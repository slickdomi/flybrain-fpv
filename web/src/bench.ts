// Control tests, run inside the page on brain time (?bench=stim | chase). web/scripts/smoke.mjs waits for
// document.documentElement.dataset.done and saves window.flybrainFpv.bench(); web/scripts/check-controls.mjs
// turns the saved results into the effect table and the pass/fail verdicts.
//
//   stim   Trials of one stimulus each: baseline, stimulus held, washout, in a seeded random order. With
//          ?loop=open the drone is held in place (so the view never changes) and what it WOULD have done is
//          integrated from the commands; with ?loop=closed it flies and the actual turn and climb are measured.
//   chase  One truck (CHASE_TRUCK) is designated as food and the drone flies for ?secs= brain seconds. Run it with and
//          without ?mb=off (the food drive's steering cut, its learning kept) to see what the food drive adds.
//          Every metric (hits, sighting, ahead, lost, the rows) is about that truck; the others drive the road as in the
//          game, and ramming one of them loses the drone (stats.crashCar), as a tree does.
//          The trucks drive the road, so "ahead" is the stretch of road nearest to ahead (World.placeCarAhead).
//          A ram sets off the drone's bomb: the truck and the drone are destroyed, and the next target appears ahead
//          of the new drone when it comes back (GAME.respawnMs later).
//
// Nothing here changes how the fly is controlled; it only schedules stimuli and reads what main.ts computes.

import { DRONE, FOOD, STIMS, WORLD } from "./config";
import { rng, type Target, type World } from "./game/world";

/** What main.ts computes on every readback, as the tests see it. */
export interface Sample {
  /** deg/s the drone is commanded to turn, all sources */
  turn: number;
  /** the parts: DNa02 (after gain and adaptation) and the food drive's readout */
  dnaTurn: number;
  mbTurn: number;
  /** raw DNa02 right - left, Hz, before adaptation */
  diff: number;
  dnp53: number;
  /** deg/s of pitch the brain commands (before the spring) */
  pitch: number;
  gf: number;
  wantL: number;
  wantR: number;
  /** looming cells per side, Hz: LC4 + LPLC2 (50 ms filter) */
  loomL: number;
  loomR: number;
  /** the escape hop's own test would have fired now (whether or not the lock suppressed it) */
  escaping: boolean;
}

export interface Harness {
  world: World;
  /** constant drive onto the cells of a STIMS key, 0 = off; returns the number of cells */
  drive(key: string, mv: number): number;
  /** drive the paint cell type within PAINT.radius of (az, el); returns the number of cells */
  paint(az: number, el: number, mv: number): number;
  unpaint(): void;
  designate(t: Target | null): void;
  /** mean rate of a stimulated group right now (probe "stim:<key>"), Hz */
  stimHz(key: string): number;
  /** mean rate over the last readback of just the cells paint(az, el) drives, painted or not, Hz */
  paintedHz(az: number, el: number): number;
  counts(): { seizures: number; escapes: number; rewardPulses: number; meanWeight: number; depressed: number };
  settings: Record<string, number | string | boolean>;
}

const PHASE = { base: 1500, stim: 2000, wash: 3000 };
/** chase: the designated truck (its slot in world.cars) */
const CHASE_TRUCK = 0;
/** chase: the target counts as lost when it stays farther than this for this long */
const LOST_DIST = 40;
const LOST_MS = 10000;
/** bench time starts after the warm-up and a settle, so the first baseline isn't the brain waking up */
const START_MS = 4000;

interface Bucket {
  n: number;
  turn: number;
  diff: number;
  dnp53: number;
  pitch: number;
  gf: number;
  gfMax: number;
  targetHz: number;
}
const bucket = (): Bucket => ({ n: 0, turn: 0, diff: 0, dnp53: 0, pitch: 0, gf: 0, gfMax: 0, targetHz: 0 });
const meanOf = (b: Bucket) => ({
  turn: b.turn / b.n, diff: b.diff / b.n, dnp53: b.dnp53 / b.n, pitch: b.pitch / b.n, gf: b.gf / b.n, gfMax: b.gfMax, targetHz: b.targetHz / b.n,
});

export interface Bench {
  /** before world.update: schedule stimuli for this frame */
  frame(ms: number): void;
  /** after world.update, with the heading and altitude it had before and the commands it was given */
  after(ms: number, before: { heading: number; y: number }, cmd: { yawRate: number; pitchRate: number }): void;
  /** on every readback, after main.ts has computed its commands */
  readback(s: Sample): void;
  /** also on every readback: an escape was triggered */
  escape(): void;
  seizure(): void;
  status(): string;
  results(): unknown;
}

/** Stimuli the stim protocol knows: the STIMS keys, and a painted phantom on either side. */
const PAINTS: Record<string, { az: number; el: number }> = { paintL: { az: -40, el: 0 }, paintR: { az: 40, el: 0 } };

export function createBench(kind: string, q: URLSearchParams, h: Harness): Bench | null {
  if (kind === "stim") return stimBench(q, h);
  if (kind === "chase") return chaseBench(q, h);
  if (kind === "approach") return approachBench(q, h);
  if (kind === "survive") return surviveBench(q, h);
  console.error(`unknown ?bench=${kind} (stim | chase | approach | survive)`);
  return null;
}

function stimBench(q: URLSearchParams, h: Harness): Bench {
  const loop = q.get("loop") === "closed" ? "closed" : "open";
  const names = (q.get("stims") ?? [...STIMS.map((s) => s.key), ...Object.keys(PAINTS)].join(",")).split(",").filter(Boolean);
  const mvs = (q.get("mvs") ?? "10,20").split(",").map(Number);
  const reps = Number(q.get("reps") ?? 3);
  const seed = Number(q.get("seed") ?? 1);
  const rand = rng(seed * 104729 + 7);
  // every stimulus x drive, shuffled within each repeat so slow drift spreads over all of them
  const trials: { stim: string; mv: number; rep: number }[] = [];
  for (let rep = 0; rep < reps; rep++) {
    const block = names.flatMap((stim) => mvs.map((mv) => ({ stim, mv, rep })));
    for (let i = block.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [block[i], block[j]] = [block[j], block[i]];
    }
    trials.push(...block);
  }
  const w = h.world;
  w.frozen = loop === "open";
  // this measures steering, not survival: contact bounces and stuns instead of blowing the drone up
  w.fatal = false;
  const out: unknown[] = [];
  let i = -1;
  let phase: "base" | "stim" | "wash" = "wash";
  let phaseStart = START_MS - PHASE.wash;
  let base = bucket();
  let stim = bucket();
  let turned = 0;
  let climbed = 0;
  let escapes = 0;
  let seized = false;
  let hitAtStart = 0;
  let cells = 0;
  const trace: number[] = [];
  let done = false;

  const setStim = (name: string, mv: number) => {
    const p = PAINTS[name];
    if (p) {
      if (mv > 0) cells = h.paint(p.az, p.el, mv);
      else h.unpaint();
    } else if (mv > 0) cells = h.drive(name, mv);
    else h.drive(name, 0);
  };
  // a painted phantom's own cells are the few LC10a it drives, not the whole side: the side's mean dilutes them
  const targetHz = (name: string) => (PAINTS[name] ? h.paintedHz(PAINTS[name].az, PAINTS[name].el) : h.stimHz(name));

  const finishTrial = () => {
    const t = trials[i];
    const b = meanOf(base);
    const s = meanOf(stim);
    const excluded = seized ? "seizure" : w.stats.treeHits > hitAtStart ? "tree" : null;
    out.push({
      ...t, loop, cells, excluded,
      base: b, during: s,
      /** degrees turned (closed) or that the commands would have turned (open) while the stimulus was held */
      turnedDeg: turned,
      /** closed: altitude change; open: degrees of pitch the commands would have added */
      climbed,
      escapes,
      trace: trace.slice(),
    });
  };

  return {
    frame() {
      if (done) return;
      const now = w.time;
      const elapsed = now - phaseStart;
      if (phase === "wash" && elapsed >= PHASE.wash) {
        if (i >= 0) finishTrial();
        i++;
        if (i >= trials.length) {
          done = true;
          document.documentElement.dataset.done = "1";
          return;
        }
        phase = "base";
        phaseStart = now;
        base = bucket();
        seized = false;
      } else if (phase === "base" && elapsed >= PHASE.base) {
        phase = "stim";
        phaseStart = now;
        stim = bucket();
        turned = 0;
        climbed = 0;
        escapes = 0;
        cells = 0;
        hitAtStart = w.stats.treeHits;
        trace.length = 0;
        const t = trials[i];
        setStim(t.stim, t.mv);
      } else if (phase === "stim" && elapsed >= PHASE.stim) {
        setStim(trials[i].stim, 0);
        phase = "wash";
        phaseStart = now;
        // closed loop: keep the drone in the air between trials, away from the floor and the ceiling
        if (loop === "closed" && (w.y < 3 || w.y > 20)) {
          w.y = 8;
          w.pitch = 0;
        }
      }
    },
    after(ms, before, cmd) {
      if (phase !== "stim" || done) return;
      if (loop === "open") {
        turned += (cmd.yawRate * ms) / 1000;
        climbed += (cmd.pitchRate * ms) / 1000;
      } else {
        turned += (Math.atan2(Math.sin(w.heading - before.heading), Math.cos(w.heading - before.heading)) * 180) / Math.PI;
        climbed += w.y - before.y;
      }
    },
    readback(s) {
      if (done || i < 0) return;
      const b = phase === "base" ? base : phase === "stim" ? stim : null;
      if (!b) return;
      b.n++;
      b.turn += s.turn;
      b.diff += s.diff;
      b.dnp53 += s.dnp53;
      b.pitch += s.pitch;
      b.gf += s.gf;
      b.gfMax = Math.max(b.gfMax, s.gf);
      b.targetHz += targetHz(trials[i].stim);
      if (phase === "stim") trace.push(Math.round(w.time - phaseStart), +s.turn.toFixed(1));
    },
    escape() {
      if (phase === "stim") escapes++;
    },
    seizure() {
      if (phase === "base" || phase === "stim") seized = true;
    },
    status() {
      if (done) return `TEST stim (${loop} loop) done: ${trials.length} trials`;
      const t = trials[Math.max(0, i)];
      return `TEST stim (${loop} loop) · trial ${i + 1}/${trials.length} · ${t.stim} at ${t.mv} mV · ${phase}`;
    },
    results: () => ({ protocol: "stim", loop, seed, mvs, reps, phases: PHASE, settings: h.settings, trials: out, done }),
  };
}

function chaseBench(q: URLSearchParams, h: Harness): Bench {
  const secs = Number(q.get("secs") ?? 180);
  const seed = Number(q.get("seed") ?? 1);
  /** ?designate=off: the trucks are there but nothing is food (no marker, no painting, no lock) */
  const designate = q.get("designate") !== "off";
  const w = h.world;
  // as measured: destroyed trucks come back by themselves after GAME.carRespawnMs, and wrecks clear (the game waits
  // for the player's Respawn trucks)
  w.autoCarRespawn = true;
  /** the one truck this test chases; every metric is about it */
  const target: Target = { kind: "car", index: CHASE_TRUCK };
  if (w.cars.length <= CHASE_TRUCK) console.error("?bench=chase needs a truck (not ?cars=0)");
  const rows: number[][] = [];
  let lastRow = -Infinity;
  let designated = false;
  let lost = 0;
  /** brain ms from first seeing the food to reaching it, per food reached */
  const latencies: number[] = [];
  let sightedAt: number | null = null;
  let reached = 0;
  let farSince = 0;
  let done = false;
  return {
    frame() {
      if (done) return;
      if (!designated && w.time >= START_MS) {
        designated = true;
        // a chase starts with its truck in sight; after a ram (or losing it) the next one appears ahead too
        w.carAhead = CHASE_TRUCK;
        w.placeCarAhead(CHASE_TRUCK);
        if (designate) h.designate(target);
      }
      // lost: out of reach for LOST_MS -> put it back ahead and count it
      const l = designated ? w.locate(target) : null;
      if (l && l.dist > LOST_DIST) {
        if (w.time - farSince > LOST_MS) {
          lost++;
          w.placeCarAhead(CHASE_TRUCK);
          farSince = w.time;
        }
      } else farSince = w.time;
      // sighting to contact: the same "in view" as the food drive's (FOOD.viewAzDeg, viewDist, no tree between)
      if (w.stats.foodReached > reached) {
        reached = w.stats.foodReached;
        if (sightedAt !== null) latencies.push(w.time - sightedAt);
        sightedAt = null;
      }
      const inView = !!l && !l.occluded && Math.abs(l.az) <= FOOD.viewAzDeg && l.dist <= FOOD.viewDist && !w.droneLost;
      if (w.droneLost) sightedAt = null;
      else if (inView && sightedAt === null) sightedAt = w.time;
      if (w.time >= START_MS + secs * 1000) {
        done = true;
        document.documentElement.dataset.done = "1";
      }
    },
    after() {},
    readback(s) {
      // not before the truck is placed: a readback can land between START_MS and the frame that places it
      if (done || !designated || w.time < START_MS || w.time - lastRow < 100) return;
      lastRow = w.time;
      const l = w.locate(target);
      rows.push([
        Math.round(w.time - START_MS), l ? +l.az.toFixed(1) : NaN, l ? +l.el.toFixed(1) : NaN, l ? +l.dist.toFixed(1) : NaN,
        l && !l.occluded ? 1 : 0, +s.turn.toFixed(2), +s.dnaTurn.toFixed(2), +s.mbTurn.toFixed(2),
        +s.wantL.toFixed(4), +s.wantR.toFixed(4), w.stats.foodReached, +w.y.toFixed(1),
      ]);
    },
    escape() {},
    seizure() {},
    status() {
      return `TEST chase · ${Math.max(0, (w.time - START_MS) / 1000).toFixed(0)}/${secs} s · food ${designate ? `truck ${CHASE_TRUCK + 1}` : "none"} · hits ${w.stats.foodReached} · other trucks rammed ${w.stats.crashCar} · trucks destroyed ${w.stats.carsDestroyed}`;
    },
    results: () => ({
      protocol: "chase", seed, secs, designate, truck: CHASE_TRUCK, trucks: w.cars.length, settings: h.settings, done,
      counts: h.counts(), stats: { ...w.stats }, lost, latencies, placements: { ...w.aheadPlacements },
      // "rams": the target's (stats.foodReached), not every truck's
      columns: ["t", "az", "el", "dist", "visible", "turn", "dnaTurn", "mbTurn", "wantL", "wantR", "rams", "alt"],
      rows,
    }),
  };
}

/**
 * approach: does anything in the brain say "obstacle ahead, turn away", and how early? The drone flies straight at
 * cruise (steerOff: the brain's commands are recorded, not obeyed) at one tree placed APPROACH_DIST ahead and a set
 * distance to the side (?offsets=, + = right), and flies on through it (ghost), so each trial records the whole
 * approach and the pass. Each trial is preceded by APPROACH_SETTLE_MS of the same pose with no tree, the baseline.
 * Nothing here steers: web/scripts/analyze-approach.mjs reads the rows.
 */
const APPROACH_DIST = 25;
const APPROACH_SETTLE_MS = 2000;
/** how far past the tree a trial runs */
const APPROACH_PAST = 4;
const APPROACH_ALT = 6;

function approachBench(q: URLSearchParams, h: Harness): Bench {
  const seed = Number(q.get("seed") ?? 1);
  const offsets = (q.get("offsets") ?? "-3,-1.5,-0.6,0,0.6,1.5,3").split(",").map(Number);
  const reps = Number(q.get("reps") ?? 3);
  const trunkR = Number(q.get("trunkr") ?? WORLD.trunkRadius);
  const rand = rng(seed * 7727 + 3);
  const trials: { offset: number; rep: number }[] = [];
  for (let rep = 0; rep < reps; rep++) {
    const block = offsets.map((offset) => ({ offset, rep }));
    for (let i = block.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [block[i], block[j]] = [block[j], block[i]];
    }
    trials.push(...block);
  }
  const w = h.world;
  w.steerOff = true;
  w.ghost = true;
  w.fatal = false;
  w.trees = [];
  const out: { offset: number; rep: number; trunkR: number; rows: number[][]; seized: boolean }[] = [];
  let i = -1;
  let phase: "settle" | "run" = "run";
  let phaseStart = 0;
  let rows: number[][] = [];
  let seized = false;
  let lastRow = -Infinity;
  let escapeNow = false;
  let done = false;
  let treeZ = 0;

  const pose = () => {
    w.x = 0;
    w.z = 0;
    w.y = APPROACH_ALT;
    w.heading = 0;
    w.pitch = 0;
    w.yawRate = 0;
    w.pitchRate = 0;
  };

  return {
    frame() {
      if (done || w.time < START_MS) return;
      if (phase === "run" && (i < 0 || w.z < treeZ - APPROACH_PAST)) {
        if (i >= 0) out.push({ ...trials[i], trunkR, rows, seized });
        i++;
        if (i >= trials.length) {
          done = true;
          w.trees = [];
          document.documentElement.dataset.done = "1";
          return;
        }
        phase = "settle";
        phaseStart = w.time;
        w.trees = [];
        rows = [];
        seized = false;
        pose();
      } else if (phase === "settle") {
        // hold the pose (the brain sees the empty field ahead), then put the tree down and let the drone go
        pose();
        if (w.time - phaseStart >= APPROACH_SETTLE_MS) {
          phase = "run";
          phaseStart = w.time;
          treeZ = -APPROACH_DIST;
          w.trees = [{ x: trials[i].offset, z: treeZ, trunkR, trunkTop: WORLD.trunkTop, canopyY: WORLD.trunkTop, canopyR: 0 }];
        }
      }
    },
    after() {},
    readback(s) {
      if (done || i < 0 || w.time - lastRow < 100) return;
      lastRow = w.time;
      // t: s relative to reaching the tree's depth (negative before); settle rows carry t = NaN
      const t = phase === "run" ? (treeZ - w.z) / (DRONE.cruise) : NaN;
      rows.push([
        phase === "run" ? +t.toFixed(2) : NaN, +s.loomL.toFixed(1), +s.loomR.toFixed(1), +s.gf.toFixed(0), +s.diff.toFixed(2),
        +s.turn.toFixed(1), s.escaping || escapeNow ? 1 : 0,
      ]);
      escapeNow = false;
    },
    escape() {
      escapeNow = true;
    },
    seizure() {
      seized = true;
    },
    status() {
      if (done) return `TEST approach done: ${trials.length} trials`;
      const t = trials[Math.max(0, i)];
      return `TEST approach · trial ${i + 1}/${trials.length} · tree ${t.offset > 0 ? "+" : ""}${t.offset} to the side · ${phase}`;
    },
    results: () => ({
      protocol: "approach", seed, reps, offsets, trunkR, dist: APPROACH_DIST, settleMs: APPROACH_SETTLE_MS, settings: h.settings, done,
      columns: ["t", "loomL", "loomR", "gf", "diff", "turn", "escape"],
      trials: out,
    }),
  };
}

/**
 * survive: how long does the drone last among the trees, every contact fatal and nothing to chase? It flies the
 * forest for ?secs= brain seconds (no trucks, no balloons, no food), respawning at the start after each crash, and
 * the drone losses by cause and the flight time between them are recorded. Run with ?avoid= and ?escape= to compare
 * avoidance mechanisms on the same seeds.
 */
function surviveBench(q: URLSearchParams, h: Harness): Bench {
  const secs = Number(q.get("secs") ?? 180);
  const seed = Number(q.get("seed") ?? 1);
  const w = h.world;
  /** where the drone was every 0.5 s (x, z), and the 10 x 10-unit patches of the field it flew over */
  const path: number[] = [];
  const patches = new Set<number>();
  let lastPos = -Infinity;
  /** brain ms of each drone's flight, from launch to its crash */
  const lives: number[] = [];
  let launched = START_MS;
  let wasLost = false;
  let done = false;
  return {
    frame() {
      if (done) return;
      if (w.droneLost && !wasLost) lives.push(w.time - launched);
      if (!w.droneLost && w.time >= START_MS && w.time - lastPos >= 500) {
        lastPos = w.time;
        path.push(+w.x.toFixed(1), +w.z.toFixed(1));
        patches.add(Math.floor((w.x + WORLD.size / 2) / 10) * 100 + Math.floor((w.z + WORLD.size / 2) / 10));
      }
      if (!w.droneLost && wasLost) launched = w.time;
      wasLost = w.droneLost;
      if (w.time >= START_MS + secs * 1000) {
        done = true;
        document.documentElement.dataset.done = "1";
      }
    },
    after() {},
    readback() {},
    escape() {},
    seizure() {},
    status() {
      return `TEST survive · ${Math.max(0, (w.time - START_MS) / 1000).toFixed(0)}/${secs} s · crashes ${w.stats.crashTree} tree, ${w.stats.crashGround} ground`;
    },
    results: () => ({
      protocol: "survive", seed, secs, settings: h.settings, done, counts: h.counts(), stats: { ...w.stats },
      lives, flying: done ? w.time - launched : 0, patches: patches.size, path,
    }),
  };
}
