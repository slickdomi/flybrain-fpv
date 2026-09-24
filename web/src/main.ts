// A fruit fly's connectome flying a quadcopter.
//
// The brain sees the world through its own compound eyes (a compute shader sampling the scene on the GPU) and
// the drone follows three of its outputs, each taken from a pathway another project measured:
//   DNa02 right - left        -> yaw               (Aimbug, PacFly, fly-addiction)
//   DNp53 - LC4/LPLC2         -> gaze pitch = climb (Aimbug)
//   DNp01 giant fiber burst   -> escape hop        (PacFly's panic)
// plus the food drive (?food=): the designated food marked black to the eye and painted onto LC10a, or (learn) what
// the mushroom body has learned to want, read out per hemisphere (fly-addiction).
// Forward speed is constant. The keys and the eye view let you stimulate cells while it flies.

import {
  BrainView, Connectome, DopamineLearning, GpuBrain, loadBrainData, Probes, Rates, requestFlyDevice, resolveLoadOptions,
  SeizureWatchdog, type Readback,
} from "flybrain";
import eyeSrc from "./shaders/eye.wgsl?raw";
import sceneSrc from "./shaders/scene.wgsl?raw";
import { createBench, type Bench, type Harness } from "./bench";
import { ALTITUDE, AVOID, CAR, CONTROL, DATA_URL, FOOD, FOOD_MODES, FOREST, GAME, LOCK, LOOP, MARK, MB, MODEL, PAINT, SEEK, STIMS, TEST_STIMS, WORLD, type FoodMode } from "./config";
import { LearnedValue } from "./game/mb";
import { Renderer, VIEW_COUNT, VIEW_NAMES, type Look, type ViewMode } from "./game/render";
import { Osd } from "./game/osd";
import { setupPanel } from "./ui/panel";
import { World, wrapDelta, type Commands, type Target } from "./game/world";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
/** brain ms between trace samples */
const TRACE_MS = 100;
/** column names of window.flybrainFpv.trace */
const TRACE_COLUMNS = ["t", "alt", "pitch", "dnaL", "dnaR", "yawCmd", "dnp53", "lc", "pitchCmd", "gf", "bAz", "bEl", "bDist", "popped"];
const query = new URLSearchParams(location.search);
const num = (key: string, fallback: number) => {
  const v = Number(query.get(key));
  return query.has(key) && Number.isFinite(v) ? v : fallback;
};

/** Something stopped the fly: the loading screen comes back (or stays) with the message and a Reload button. */
function fail(msg: string) {
  const loading = $("loading");
  loading.style.display = "";
  $("start").hidden = true;
  const el = $("loadError");
  el.hidden = false;
  el.textContent = msg;
  $("reload").hidden = false;
}

async function main() {
  // the display's frame interval (ms), from requestAnimationFrame while the brain loads and the GPU is still idle; the
  // frame loop judges a frame late against it (so a phone held at 30 Hz in low-power mode isn't taken for a busy GPU)
  let displayMs = Infinity;
  let measuring = true;
  let lastRaf = 0;
  const measure = (t: number) => {
    if (lastRaf) displayMs = Math.min(displayMs, t - lastRaf);
    lastRaf = t;
    if (measuring) requestAnimationFrame(measure);
  };
  requestAnimationFrame(measure);
  // side panel sections: collapse, pin to the top, drag by the grip, float (desktop; remembered per browser), right away, while loading
  const panelLayout = setupPanel($("panel"));
  $("resetLayout").addEventListener("click", () => panelLayout.reset());
  const loadLabel = $("loadLabel");
  const loadBar = $("loadBar");
  // a download that stops moving (a phone on a flaky connection): say so, and offer a reload, but keep waiting
  let lastProgress = performance.now();
  const stallWatch = window.setInterval(() => {
    if (!$("start").hidden || !$("loadError").hidden) return window.clearInterval(stallWatch);
    if (performance.now() - lastProgress > 25000) {
      loadLabel.textContent = "The download has stopped moving. It may still recover, or reload to start again.";
      $("reload").hidden = false;
    }
  }, 5000);
  const device = await requestFlyDevice().catch((e) => (console.warn("WebGPU init failed:", e), null));
  if (!device) {
    fail("No WebGPU here. Try a recent Chrome, Edge or Safari, or Firefox with WebGPU switched on.");
    return;
  }
  device.lost.then((info) => fail(`GPU device lost: ${info.message}`));
  device.addEventListener("uncapturederror", (ev) => console.error("WebGPU:", (ev as GPUUncapturedErrorEvent).error.message));

  const canvas = $<HTMLCanvasElement>("view");
  const format = navigator.gpu.getPreferredCanvasFormat();
  // the renderer owns the scene uniform, so it exists before the brain whose eye reads it
  const renderer = new Renderer(device, canvas, format);
  // the FPV camera by default (3); ?view=0 fly eye, 1 chase, 2 ommatidia
  renderer.mode = Math.max(0, Math.min(VIEW_COUNT - 1, num("view", 3))) as ViewMode;
  const osd = new Osd($<HTMLCanvasElement>("osd"));

  /** ?food=: how designating food steers the fly (FoodMode, SEEK in config.ts). Only "learn" modifies the brain. */
  const foodMode: FoodMode = FOOD_MODES.includes(query.get("food") as FoodMode) ? (query.get("food") as FoodMode) : SEEK.mode;
  const learns = foodMode === "learn";

  const { prune, laminaWeight, ...modelParams } = MODEL;
  let data;
  try {
    data = await loadBrainData(
      resolveLoadOptions(
        {
          url: DATA_URL,
          prune,
          laminaWeight,
          // ?food=learn only: the mushroom body's memory (Kenyon cell -> MBON synapses can change, dopamine cells say
          // where), with fly-addiction's two changes to the loaded brain. Every other mode runs the connectome as built.
          ...(learns
            ? {
                plastic: { pre: "^KC", post: "^MBON" },
                extract: [{ name: "dopamine", pre: "^(PAM|PPL1)", post: "^MBON" }],
                silence: MB.silenceKcKc ? [{ pre: "^KC", post: "^KC" }] : [],
                gain: MB.visualGain === 1 ? [] : [{ pre: MB.visualPre, post: MB.visualPost, factor: MB.visualGain }],
              }
            : {}),
        },
        MODEL,
      ),
      (label, frac) => {
        loadLabel.textContent = label;
        loadBar.style.width = `${Math.round(frac * 100)}%`;
        lastProgress = performance.now();
      },
    );
  } catch (e) {
    fail(e instanceof Error ? e.message : String(e));
    throw e;
  }
  // downloaded: the rest (building the brain and its shaders on the GPU) reports no progress, which is not a stall
  window.clearInterval(stallWatch);
  loadLabel.textContent = "Building the brain on the GPU…";
  await new Promise((r) => setTimeout(r, 30)); // let that label show before the long synchronous build
  const brain = new GpuBrain(device, data, {
    model: modelParams,
    maxStepsPerCall: LOOP.maxStepsPerFrame,
    maxSpikesPerStep: LOOP.maxSpikesPerStep,
    eye: { sceneBuffer: renderer.sceneBuffer, shader: sceneSrc + eyeSrc },
  });
  const info = brain.info;
  loadLabel.textContent = `${info.n.toLocaleString()} neurons and ${(info.stats.spikeEdges + info.stats.gradedEdges + info.stats.plasticEdges).toLocaleString()} connections, ready.`;
  loadBar.style.width = "100%";

  // ---- cells ---------------------------------------------------------------------------------
  const cells = new Connectome(info);
  const probes = new Probes();
  const MOTOR = ["DNa02", "LC10a", "LC4", "LPLC2", "DNp53", "DNp01", "MDN"];
  for (const type of MOTOR) {
    probes.add(`${type}:L`, cells.select({ type, side: "left" }));
    probes.add(`${type}:R`, cells.select({ type, side: "right" }));
  }
  const clique = cells.select({ type: "lLN1_bc" });
  const pam = cells.select({ type: /^PAM/ });
  probes.add("lLN1_bc", clique).add("PAM", pam).add("PPL1", cells.select({ type: /^PPL1/ }));
  const stimCells = new Map([...STIMS, ...TEST_STIMS].map((s) => [s.key, cells.select({ type: s.type, side: s.side })]));
  // every stimulation group's own rate, so a test can check that the drive reached its cells
  for (const [key, sel] of stimCells) probes.add(`stim:${key}`, sel);
  // DopamineLearning adds its own probe groups, so it must exist before setProbes()
  const learning = learns ? new DopamineLearning(brain, probes, { dopamineEdges: "dopamine", recoveryMs: MB.memoryMs }) : null;
  brain.setProbes(probes.indices());
  const rates = new Rates(probes);
  /** each LC10a cell's slot in rates.delta (the stim:1 and stim:2 groups), for the painted cells' own rate (bench) */
  const lc10Slot = new Map<number, number>();
  for (const key of ["1", "2"]) {
    const { start } = probes.group(`stim:${key}`);
    stimCells.get(key)!.forEach((n, k) => lc10Slot.set(n, start + k));
  }
  const paintSlots = new Map<string, number[]>();
  const value = learning ? new LearnedValue(info, learning, probes, "dopamine", MB.floor, MB.avoidFloor, MB.traceMs) : null;
  const both = (t: string, f: (name: string) => number) => f(`${t}:L`) + f(`${t}:R`);
  /** every LC10a cell with a connectome receptive field, for painting the food onto them fast */
  const lc10 = cells.select({ type: PAINT.type }).filter((i) => info.rf[3 * i + 2] > 0);
  console.log(
    `food mode ${foodMode} · cells: ${MOTOR.map((t) => `${t} ${probes.group(`${t}:L`).count}+${probes.group(`${t}:R`).count}`).join(" · ")} · ` +
      `lLN1_bc ${clique.length} · PAM ${pam.length} · LC10a with a receptive field ${lc10.length}` +
      (learning ? ` · plastic KC->MBON ${learning.weights.length}` : ""),
  );

  const watchdog = new SeizureWatchdog(brain, LOOP.seizureSpikesPerSec, LOOP.seizureMs);
  // a busy view can pass the spike threshold on its own; only a runaway clique is a seizure (PacFly)
  watchdog.confirm = () => rates.instHz("lLN1_bc") > 100;

  const benchKind = query.get("bench");
  /** Forest mode, thick trunks plus the looming turn-away (FOREST in config.ts): on unless ?forest=0 */
  const forest = query.has("forest") ? query.get("forest") !== "0" : FOREST.default;
  /** grenades are off unless GAME.grenadesEnabled or ?grenades=1: no key, no button, none on the drone */
  const grenadesOn = query.has("grenades") ? query.get("grenades") !== "0" : GAME.grenadesEnabled;
  const world = new World({
    seed: num("seed", 1),
    balloons: num("balloons", benchKind ? 0 : WORLD.balloonCount),
    // Forest mode: thick trunks (FOREST) the looming turn-away can see coming; ?trunkr= overrides
    trunkR: num("trunkr", forest ? FOREST.trunkRadius : WORLD.trunkRadius),
    // ?cars=N trucks (CAR.count by default; ?car=0 or ?cars=0: none); none in the stimulation tests
    cars: query.get("car") === "0" ? 0 : num("cars", benchKind === "stim" && !query.has("car") ? 0 : CAR.count),
  });
  world.grenadesPerDrone = world.grenadesLeft = grenadesOn ? GAME.grenades : 0;
  const look: Look = {
    groundContrast: num("ground", WORLD.groundContrast),
    treeContrast: num("trees", WORLD.treeContrast),
    /** the designated food turns black to the fly (MARK) in food modes marker, both and learn; ?mark=on|off overrides */
    mark: query.has("mark") ? query.get("mark") !== "off" : foodMode === "marker" || foodMode === "both" || foodMode === "learn",
  };
  /** the food modes that paint the food onto LC10a (SEEK) */
  const paints = foodMode === "paint" || foodMode === "both";
  /** units: no painting nearer than this (SEEK.stopWithin, ?paintnear=) */
  const paintNear = num("paintnear", SEEK.stopWithin);
  /** units: the marker comes off nearer than this, and the food is its own grey again (MARK.stopWithin, ?marknear=) */
  const markNear = num("marknear", MARK.stopWithin);
  /** the food's distance this frame, for the marker cut-off (Infinity when there is no food) */
  let foodDist = Infinity;
  const gains = {
    yaw: num("yaw", CONTROL.yawGain),
    pitch: num("pitch", CONTROL.pitchGain),
    pitchRef: num("pitchref", CONTROL.pitchRef),
    yawAdaptMs: num("yawadapt", CONTROL.yawAdaptMs),
    pitchAdaptMs: num("pitchadapt", CONTROL.pitchAdaptMs),
    escapeHz: num("escape", CONTROL.escapeHz),
    mbTurn: num("mbturn", MB.turnGain),
  };
  /**
   * The learned food drive (?food=learn). ?mb=off keeps the learning and cuts the readout's steering (fly-addiction's
   * control arm). `locked`: the target lock (LOCK) is on this frame.
   */
  const food = {
    steer: learns && query.get("mb") !== "off", pulses: 0, lastSight: -Infinity, reward: num("reward", FOOD.rewardMv),
    /** the Target steering button: false switches the marker, painting, lock and dive off together */
    on: true,
    locked: false, inView: false, painted: null as Uint32Array | null, lastPaint: -Infinity,
    /** world time until which the target counts as engaged: no turn-away, no escape hop (AVOID.holdMs) */
    engagedUntil: -Infinity,
    /** ?freeze=, ?dive=: LOCK's adaptation freeze and terminal-dive assist; `diving` is on this frame, `diveEl` its target (deg) */
    freeze: query.has("freeze") ? query.get("freeze") !== "0" : LOCK.freezeAdapt,
    dive: query.has("dive") ? query.get("dive") !== "0" : LOCK.dive,
    diving: false, diveEl: 0,
  };

  // ---- readback: rates -> commands --------------------------------------------------------------
  let yawBias = 0;
  let yawCmd = 0;
  let pitchBias = 0;
  let pitchCmd = 0;
  let wantBias = 0;
  let wantTurn = 0;
  /** AVOID: the looming asymmetry's running mean, and the turn away it gives now (deg/s) */
  let loomBias = 0;
  let avoidTurn = 0;
  const avoidGain = num("avoid", forest ? FOREST.avoidGain : AVOID.gain);
  const avoidAdaptMs = num("avoidadapt", AVOID.adaptMs);
  /** ?avoidoff=lock (default): no turn-away while the target is locked; view: none while it is in view at all */
  const avoidWhile = query.get("avoidoff") === "view" ? "view" : "lock";
  let gfMean = 0;
  let lastEscape = -Infinity;
  let escapeNow = false;
  /** brain time of the last reset (a seizure, or a new drone): constant drives and learning wait MB.learnHoldMs */
  let lastReset = -Infinity;
  /** no steering before this brain time, while the rates climb from zero (at load, and after a new drone's reset) */
  let warmUntil = CONTROL.warmupMs;
  const pitchDrive = () =>
    both("DNp53", (n) => rates.slowHz(n)) / 2 - gains.pitchRef * (both("LC4", (n) => rates.slowHz(n)) + both("LPLC2", (n) => rates.slowHz(n)));
  /** deg/s the food drive turns the drone by (0 with ?mb=off, but computed anyway so tests can compare) */
  const mbTurn = () => gains.mbTurn * wantTurn;
  /** ALTITUDE (MODELED): deg/s pulling the gaze toward a descent above softAlt, fading in over its first unit */
  const altitudePull = () => {
    const over = world.y - ALTITUDE.softAlt;
    if (over <= 0) return 0;
    const target = -Math.min(ALTITUDE.maxDescent, ALTITUDE.descentPerUnit * over);
    return ALTITUDE.pullGain * Math.min(1, over) * (target - (world.pitch * 180) / Math.PI);
  };
  const turnNow = () => gains.yaw * yawCmd + (food.steer ? mbTurn() : 0) + avoidTurn;
  /** one sample per TRACE_MS of brain time, for headless analysis (window.flybrainFpv.trace) */
  const trace: number[][] = [];
  let lastTrace = -Infinity;
  let bench: Bench | null = null;

  brain.onReadback = (r: Readback) => {
    rates.update(r);
    const dt = rates.dtMs;
    if (dt > 0) {
      // While the brain wakes up its rates climb from zero; the filters follow fast and nothing steers yet,
      // so the flight doesn't start with a lurch that is only the filters catching up.
      const warm = r.brainTime < warmUntil;
      const adapt = (tau: number) => (tau > 0 ? 1 - Math.exp(-dt / (warm ? 200 : tau)) : 0);
      // only changes in the left-right difference steer (fly-addiction's yawAdaptMs)
      const diff = rates.hz("DNa02:R") - rates.hz("DNa02:L");
      // while the food is locked (and ?freeze), the adaptation holds: the standing difference is the food's direction
      if (!(food.freeze && food.locked)) yawBias += (diff - yawBias) * adapt(gains.yawAdaptMs);
      yawCmd = warm ? 0 : diff - yawBias;
      const pd = pitchDrive();
      pitchBias += (pd - pitchBias) * adapt(gains.pitchAdaptMs);
      pitchCmd = warm ? 0 : pd - pitchBias;
      // giant fiber: a burst above its own recent mean, not a rate (PacFly)
      const gf = both("DNp01", (n) => rates.hz(n));
      // no escape hop while the target is engaged: it looms as the drone closes in, and the hop is a retreat (LOCK)
      const engaged = food.locked || world.time < food.engagedUntil;
      if (!warm && !engaged && gains.escapeHz > 0 && gf - gfMean > gains.escapeHz && r.brainTime - lastEscape > CONTROL.escapeRefractoryMs) {
        lastEscape = r.brainTime;
        escapeNow = true;
        bench?.escape();
      }
      gfMean += (gf - gfMean) * adapt(CONTROL.escapeMeanMs);
      // AVOID (MODELED): turn away from the side whose looming cells rise above their running difference; not while
      // the target is engaged (locked or close, and AVOID.holdMs after), when it is the target that looms
      const loomDiff = rates.hz("LC4:R") + rates.hz("LPLC2:R") - rates.hz("LC4:L") - rates.hz("LPLC2:L");
      loomBias += (loomDiff - loomBias) * adapt(avoidAdaptMs);
      const avoidOff = engaged || (avoidWhile === "view" && food.inView);
      // back in gradually once the target lets go (AVOID.fadeMs)
      const fadeIn = Math.min(1, Math.max(0, (world.time - food.engagedUntil) / AVOID.fadeMs));
      avoidTurn = warm || avoidOff || avoidGain <= 0 ? 0 : -avoidGain * fadeIn * (loomDiff - loomBias);

      // the learned food drive: no learning while the seizure clique builds or just fired (PacFly)
      if (learning && value) {
        learning.enabled = rates.hz("lLN1_bc") < MB.learnPauseHz && r.brainTime - lastReset > MB.learnHoldMs;
        learning.update(rates);
        value.update(rates);
        // as with DNa02, only a change in which side is wanted steers: the right hemisphere reads higher at rest
        const wdiff = value.right - value.left;
        wantBias += (wdiff - wantBias) * adapt(MB.adaptMs);
        wantTurn = warm ? 0 : wdiff - wantBias;
      }

      bench?.readback({
        turn: turnNow(), dnaTurn: gains.yaw * yawCmd, mbTurn: mbTurn(), diff, dnp53: both("DNp53", (n) => rates.slowHz(n)) / 2,
        pitch: gains.pitch * pitchCmd, gf, wantL: value?.left ?? 0, wantR: value?.right ?? 0,
        loomL: rates.hz("LC4:L") + rates.hz("LPLC2:L"), loomR: rates.hz("LC4:R") + rates.hz("LPLC2:R"), escaping: escapeNow,
      });
      if (r.brainTime - lastTrace >= TRACE_MS) {
        lastTrace = r.brainTime;
        const nb = world.nearestBalloon();
        trace.push([
          Math.round(r.brainTime), +world.y.toFixed(2), +((world.pitch * 180) / Math.PI).toFixed(1),
          +rates.hz("DNa02:L").toFixed(1), +rates.hz("DNa02:R").toFixed(1), +yawCmd.toFixed(2),
          +(both("DNp53", (n) => rates.slowHz(n)) / 2).toFixed(2),
          +(both("LC4", (n) => rates.slowHz(n)) + both("LPLC2", (n) => rates.slowHz(n))).toFixed(1),
          +pitchCmd.toFixed(2), +gf.toFixed(0), +nb.az.toFixed(1), +nb.el.toFixed(1), +nb.dist.toFixed(1),
          world.stats.popped,
        ]);
      }
    }
    if (watchdog.update(r)) {
      world.stats.seizures++;
      lastReset = r.brainTime;
      renderer.seizure = 1;
      banner(`SEIZURE ${watchdog.count}, brain rebooting`);
      console.warn(`seizure ${watchdog.count} at brain ${(r.brainTime / 1000).toFixed(1)}s`);
      bench?.seizure();
      // stimulation would re-ignite it at once: drop every held drive
      releaseAll();
    }
  };

  // ---- the food drive: designating a target -------------------------------------------------------
  const designate = (t: Target | null) => {
    world.food = t;
    $("foodTarget").textContent = t ? (t.kind === "car" ? `truck ${t.index + 1}` : `balloon ${t.index + 1}`) : "nothing";
    if (t) banner(t.kind === "car" ? `TARGET: TRUCK ${t.index + 1}` : "TARGET: BALLOON");
  };
  /**
   * Every frame: where the food is, and what that does. In view means the same test as the reward (FOOD.viewAzDeg,
   * viewDist, no tree between). Then, by food mode:
   *   learn        MODELED reward: PAM pulses while the food is in view (a long one on reaching it, in onReach)
   *   paint, both  MODELED targeting: the LC10a cells looking at the food get SEEK.paintMv
   * and in every mode the target lock (LOCK): no escape hop, and faster flight the more the food is dead ahead.
   */
  const seekFood = () => {
    const l = world.food && !world.droneLost && food.on ? world.locate(world.food) : null;
    const inView = !!l && !l.occluded && Math.abs(l.az) <= FOOD.viewAzDeg && l.dist <= FOOD.viewDist;
    foodDist = l ? l.dist : Infinity;
    food.inView = inView;
    food.locked = inView && Math.abs(l!.az) <= LOCK.azDeg && l!.dist <= LOCK.dist;
    // MODELED terminal dive (?dive): aim the gaze at the food's elevation once close
    food.diving = food.dive && food.locked && l!.dist <= LOCK.diveDist;
    food.diveEl = food.diving ? l!.el : 0;
    // engaged: locked, or close in front even if out of view; the turn-away and the hop wait AVOID.holdMs after it
    if (food.locked || (l && l.dist <= LOCK.diveDist && Math.abs(l.az) <= 90)) food.engagedUntil = world.time + AVOID.holdMs;
    world.speedScale = food.locked ? 1 + (LOCK.speedUp - 1) * (1 - Math.abs(l!.az) / LOCK.azDeg) : 1;

    if (learns && inView && food.reward > 0 && world.time - food.lastSight >= FOOD.sightEveryMs) {
      food.lastSight = world.time;
      brain.pulse(pam, food.reward, FOOD.sightPulseMs);
      food.pulses++;
    }

    // No painting when there's nothing to paint, right after a seizure (a constant drive would re-ignite the brain as
    // it restarts, PacFly), or in the last few units, where painting on top of a target filling the view seized it.
    if (!paints || !inView || world.time - lastReset < MB.learnHoldMs || (paintNear > 0 && l!.dist < paintNear)) {
      unpaintFood();
      return;
    }
    if (world.time - food.lastPaint < SEEK.paintEveryMs) return;
    food.lastPaint = world.time;
    const rf = info.rf;
    const sel = lc10.filter((i) => Math.hypot(rf[3 * i] - l!.az, rf[3 * i + 1] - l!.el) < SEEK.paintRadius);
    if (food.painted) brain.setDrive(food.painted, 0);
    food.painted = sel;
    brain.setDrive(sel, SEEK.paintMv);
  };
  const unpaintFood = () => {
    if (food.painted) brain.setDrive(food.painted, 0);
    food.painted = null;
  };
  world.onReach = (t, isFood) => {
    if (learns && isFood && food.reward > 0) {
      brain.pulse(pam, food.reward, FOOD.contactPulseMs);
      food.pulses++;
    }
    banner(t.kind === "car" ? (isFood ? "TARGET HIT" : `WRONG TRUCK (${t.index + 1})`) : isFood ? "TARGET HIT" : "POP");
  };

  // ---- the game: the drone's bomb, grenades (G), respawns ---------------------------------------------
  /** a new drone gets a fresh brain (GAME.resetBrainOnRespawn; ?brainreset=0 keeps the old one running) */
  const resetOnRespawn = query.has("brainreset") ? query.get("brainreset") !== "0" : GAME.resetBrainOnRespawn;
  const resetBrain = () => {
    // the spiking state (voltages, synaptic input, adaptation), as the seizure watchdog clears it. Not the graded
    // optic lobe: it follows the image within ~20 ms, so it holds nothing from the last drone, and flybrain's
    // reset(true) can't clear it (its buffers lack COPY_DST). Not the connectome either, nor what ?food=learn has
    // learned (the plastic weights), which is memory.
    brain.reset(false);
    lastReset = brain.brainTime;
    warmUntil = brain.brainTime + CONTROL.warmupMs;
    // a held stimulus would re-ignite it as it restarts, as after a seizure
    releaseAll();
  };
  world.onEvent = (e) => {
    if (e.kind === "droneLost") banner(`DRONE DOWN, new one in ${(GAME.respawnMs / 1000).toFixed(0)} s`);
    else if (e.kind === "carDestroyed" && e.by === "grenade") banner(`TRUCK ${e.index + 1} DESTROYED`);
    else if (e.kind === "burst" && !e.hit) banner("MISSED");
    else if (e.kind === "respawn") {
      banner("NEW DRONE");
      if (resetOnRespawn) resetBrain();
    }
    else if (e.kind === "grenade") banner(`GRENADE AWAY, ${e.left} left`);
  };
  const dropGrenade = () => {
    if (paused || !grenadesOn) return false;
    const ok = world.dropGrenade();
    if (!ok && !world.droneLost && !world.frozen) banner("OUT OF GRENADES");
    return ok;
  };
  /**
   * ?bomber=auto: drops a grenade whenever one would land within half the blast radius of where a truck will be (the
   * designated one if a truck is food, any truck otherwise). A stand-in for the player's G key in headless runs; it
   * never touches the brain or the drone's controls.
   */
  const autoBomber = query.get("bomber") === "auto";
  let lastAutoDrop = -Infinity;
  const bomb = () => {
    if (!autoBomber || world.grenadesLeft <= 0 || world.time - lastAutoDrop < 1500) return;
    const hit = world.grenadeImpact();
    if (!hit) return;
    const f = world.food;
    const aims = f?.kind === "car" ? [f.index] : world.cars.map((_, i) => i);
    for (const i of aims) {
      const car = world.carPositionIn(i, hit.t);
      if (!car || Math.hypot(wrapDelta(car[0] - hit.x), wrapDelta(car[1] - hit.z)) >= GAME.blastRadius / 2) continue;
      if (world.dropGrenade()) lastAutoDrop = world.time;
      return;
    }
  };
  /** F: whatever is nearest the middle of the view (within 30 deg); nothing there clears the designation */
  const designateAhead = () => {
    let best: Target | null = null;
    let bestAngle = 30;
    const candidates: Target[] = [
      ...world.balloons.map((_, index) => ({ kind: "balloon" as const, index })),
      ...world.cars.flatMap((c, index) => (c ? [{ kind: "car" as const, index }] : [])),
    ];
    for (const t of candidates) {
      const l = world.locate(t);
      if (!l || l.occluded || l.dist > 60) continue;
      const a = Math.hypot(l.az, l.el);
      if (a < bestAngle) (bestAngle = a), (best = t);
    }
    designate(best);
  };

  // ---- stimulation --------------------------------------------------------------------------
  let stimMv = num("mv", 10);
  const held = new Map<string, Uint32Array>();
  /** the LC10a cells being painted with a phantom, if any */
  let paintCells: Uint32Array | null = null;
  const stimButtons = new Map<string, HTMLButtonElement>();
  const press = (key: string, mv = stimMv) => {
    const sel = stimCells.get(key);
    if (!sel) return 0;
    held.set(key, sel);
    brain.setDrive(sel, mv);
    stimButtons.get(key)?.classList.add("on");
    return sel.length;
  };
  const release = (key: string) => {
    const sel = held.get(key);
    if (!sel) return;
    held.delete(key);
    brain.setDrive(sel, 0);
    stimButtons.get(key)?.classList.remove("on");
  };
  const stimList = $("stims");
  for (const s of STIMS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "stim";
    b.innerHTML = `<kbd>${s.key}</kbd><b>${s.label}</b><small>${s.what} (${stimCells.get(s.key)!.length} cells)</small>`;
    b.addEventListener("pointerdown", (e) => {
      b.setPointerCapture(e.pointerId);
      press(s.key);
    });
    const up = () => release(s.key);
    b.addEventListener("pointerup", up);
    b.addEventListener("pointercancel", up);
    // a finger held on it is a held stimulus, not a long-press menu
    b.addEventListener("contextmenu", (e) => e.preventDefault());
    stimButtons.set(s.key, b);
    stimList.append(b);
  }
  const mvInput = $<HTMLInputElement>("mv");
  mvInput.value = String(stimMv);
  const mvLabel = $("mvLabel");
  const syncMv = () => {
    stimMv = Number(mvInput.value);
    mvLabel.textContent = `${stimMv} mV`;
    // how far the slider's track is filled (styles.css)
    mvInput.style.setProperty("--fill", `${((stimMv - Number(mvInput.min)) / (Number(mvInput.max) - Number(mvInput.min))) * 100}%`);
    for (const sel of held.values()) brain.setDrive(sel, stimMv);
    if (paintCells) brain.setDrive(paintCells, stimMv);
  };
  mvInput.addEventListener("input", syncMv);
  syncMv();

  // painting a phantom: drag on the eye view to drive the LC10a cells looking there
  const paintAzEl = (az: number, el: number, mv: number) => {
    renderer.paint = { az, el, radius: PAINT.radius };
    const sel = cells.select({ type: PAINT.type, receptiveField: { az, el, radius: PAINT.radius } });
    if (paintCells) brain.setDrive(paintCells, 0);
    paintCells = sel;
    brain.setDrive(sel, mv);
    $("paintInfo").textContent = `Painting ${sel.length} ${PAINT.type} cells at ${az.toFixed(0)}°, ${el.toFixed(0)}°`;
    return sel.length;
  };
  let painting = false;
  let lastPaintSelect = 0;
  const paintAt = (e: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    const { az, el } = renderer.screenToAzEl((e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height);
    renderer.paint = { az, el, radius: PAINT.radius };
    const now = performance.now();
    if (now - lastPaintSelect < 40) return;
    lastPaintSelect = now;
    paintAzEl(az, el, stimMv);
  };
  const stopPaint = () => {
    painting = false;
    if (paintCells) brain.setDrive(paintCells, 0);
    paintCells = null;
    renderer.paint = null;
    $("paintInfo").textContent = "";
  };
  function releaseAll() {
    for (const key of [...held.keys()]) release(key);
    stopPaint();
  }

  // pointer: left button or a finger paints (eye views) or orbits (chase); right-click, or a quick tap, designates a
  // target; the wheel, or two fingers pinching, zooms
  let orbiting = false;
  let lastX = 0;
  let lastY = 0;
  /** a finger's first touch: it paints once it moves or stays (TAP), and a quick still tap designates instead */
  const TAP = { ms: 180, px: 10 };
  let touchStart: { x: number; y: number; t: number; e: PointerEvent } | null = null;
  let touchTimer = 0;
  /** fingers on the canvas (pointerId -> position), for the pinch */
  const fingers = new Map<number, { x: number; y: number }>();
  let pinchDist = 0;
  canvas.style.touchAction = "none";
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  const rayAt = (e: { clientX: number; clientY: number }) => {
    const rect = canvas.getBoundingClientRect();
    return renderer.pickRay((e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height, world);
  };
  /** a finger is fat: when the ray itself misses, the truck nearest it within 8 deg, in front of any trunk or not */
  const pickNear = (ro: [number, number, number], r: [number, number, number]): Target | null => {
    let best: Target | null = null;
    let bestCos = Math.cos((8 * Math.PI) / 180);
    world.cars.forEach((c, index) => {
      if (!c) return;
      const v = [wrapDelta(c.x - ro[0]), CAR.height / 2 - ro[1], wrapDelta(c.z - ro[2])];
      const len = Math.hypot(v[0], v[1], v[2]);
      const cos = (v[0] * r[0] + v[1] * r[1] + v[2] * r[2]) / len;
      if (len < 60 && cos > bestCos) (bestCos = cos), (best = { kind: "car", index });
    });
    return best;
  };
  const startDrag = (e: PointerEvent) => {
    if (renderer.mode === 1) {
      orbiting = true;
      lastX = e.clientX;
      lastY = e.clientY;
    } else {
      painting = true;
      paintAt(e);
    }
  };
  const endDrag = () => {
    orbiting = false;
    if (painting) stopPaint();
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button === 2) {
      const { ro, r } = rayAt(e);
      designate(world.pick(ro, r));
      return;
    }
    canvas.setPointerCapture(e.pointerId);
    if (e.pointerType !== "touch") return startDrag(e);
    fingers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (fingers.size === 2) {
      // a second finger: a pinch, not a paint or a tap
      clearTimeout(touchTimer);
      touchStart = null;
      endDrag();
      const [a, b] = [...fingers.values()];
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      return;
    }
    if (fingers.size > 2) return;
    touchStart = { x: e.clientX, y: e.clientY, t: performance.now(), e };
    touchTimer = window.setTimeout(() => {
      if (touchStart) startDrag(touchStart.e);
      touchStart = null;
    }, TAP.ms);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch" && fingers.has(e.pointerId)) {
      fingers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (fingers.size === 2) {
        const [a, b] = [...fingers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDist > 0 && d > 0) zoom(Math.log(pinchDist / d) * 600);
        pinchDist = d;
        return;
      }
      if (touchStart && Math.hypot(e.clientX - touchStart.x, e.clientY - touchStart.y) > TAP.px) {
        clearTimeout(touchTimer);
        touchStart = null;
        startDrag(e);
      }
    }
    if (painting) paintAt(e);
    if (orbiting) {
      renderer.chaseYaw += (e.clientX - lastX) * 0.006;
      renderer.chasePitch = Math.max(-0.3, Math.min(1.4, renderer.chasePitch + (e.clientY - lastY) * 0.005));
      lastX = e.clientX;
      lastY = e.clientY;
    }
  });
  const pointerUp = (e: PointerEvent) => {
    if (e.pointerType === "touch") {
      fingers.delete(e.pointerId);
      if (fingers.size > 0) return;
      pinchDist = 0;
      if (touchStart && e.type === "pointerup") {
        // a quick, still tap: designate what is under the finger (or nearest it); a tap on nothing keeps the target
        clearTimeout(touchTimer);
        const { ro, r } = rayAt(touchStart.e);
        const t = world.pick(ro, r) ?? pickNear(ro, r);
        if (t) designate(t);
      }
      touchStart = null;
    }
    endDrag();
  };
  canvas.addEventListener("pointerup", pointerUp);
  canvas.addEventListener("pointercancel", pointerUp);
  /** zoom by a wheel-like amount: + out, - in (the chase camera's distance, or the eye views' field) */
  const zoom = (delta: number) => {
    if (renderer.mode === 1) renderer.chaseDist = Math.max(1.2, Math.min(40, renderer.chaseDist * Math.exp(delta * 0.0012)));
    else renderer.fovAz = Math.max(30, Math.min(150, renderer.fovAz * Math.exp(delta * 0.0008)));
  };
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      zoom(e.deltaY);
    },
    { passive: false },
  );

  const viewBtn = $<HTMLButtonElement>("viewBtn");
  /** Q W E R: straight to a view, in the order V cycles them (FPV cam, fly eye, chase, ommatidia) */
  const VIEW_KEYS: Record<string, ViewMode> = { q: 3, w: 0, e: 1, r: 2 };
  /** a phone or tablet: the hints talk about fingers, and the key-only actions have buttons */
  const touchUI = matchMedia("(pointer: coarse)").matches;
  if (touchUI) {
    const sub = (section: string, text: string) => {
      const el = document.querySelector(`[data-section="${section}"] h2 small`);
      if (el) el.textContent = text;
    };
    sub("target", "tap a truck, or Target ahead");
    sub("stimulate", "hold a button");
  }
  const setMode = (m: ViewMode) => {
    renderer.mode = m;
    viewBtn.textContent = touchUI ? `View: ${VIEW_NAMES[m]}` : `View: ${VIEW_NAMES[m]} (Q/W/E/R)`;
    // the FPV feed carries its own OSD
    for (const id of ["title", "score"]) $(id).style.display = m === 3 ? "none" : "";
    $("hint").textContent = touchUI
      ? (m === 1 ? "Drag to orbit, pinch to zoom." : "Drag a finger to paint a fake target into the fly's eyes, pinch to zoom.") +
        " Tap a truck to target it."
      : (m === 1 ? "Drag to orbit, scroll to zoom." : "Drag to paint a fake target into the fly's eyes, scroll to zoom.") +
        ` Right-click a truck or press F to target it${grenadesOn ? ", G to drop a grenade" : ""}.`;
  };
  viewBtn.addEventListener("click", () => setMode(((renderer.mode + 1) % VIEW_COUNT) as ViewMode));
  setMode(renderer.mode);
  const mbBtn = $<HTMLButtonElement>("mbBtn");
  // ?food=learn: switches the learned readout's steering (fly-addiction's ?mb=off). Every other mode: switches all of
  // the target steering at once (marker, painting, lock, dive), so the fly flies on its own with the target designated.
  const syncMbBtn = () =>
    (mbBtn.textContent = learns
      ? `Learned steering: ${food.steer ? "on" : "off"}`
      : `Target help: ${food.on ? "on" : "off"}`);
  mbBtn.addEventListener("click", () => {
    if (learns) food.steer = !food.steer;
    else food.on = !food.on;
    syncMbBtn();
  });
  syncMbBtn();
  // the mushroom body's learned want only exists in ?food=learn
  if (!learns) $("wantL").closest<HTMLElement>(".meter")!.style.display = "none";
  // Forest mode: trunk size is built into the world, so switching reloads the page with ?forest= flipped
  const forestBtn = $<HTMLButtonElement>("forestBtn");
  forestBtn.textContent = `Forest mode: ${forest ? "on" : "off"}`;
  forestBtn.title = "Thick trees the fly can see coming, and it steers around them. Off: thin trees it can't see in time, so far more crashes, but it finds trucks more often. Reloads the page.";
  forestBtn.addEventListener("click", () => {
    const u = new URL(location.href);
    u.searchParams.set("forest", forest ? "0" : "1");
    location.href = u.toString();
  });
  let paused = false;
  window.addEventListener("keydown", (e) => {
    if (e.repeat || e.target instanceof HTMLInputElement) return;
    if (stimCells.has(e.key)) press(e.key);
    else if (e.key === "v" || e.key === "V") setMode(((renderer.mode + 1) % VIEW_COUNT) as ViewMode);
    else if (Object.hasOwn(VIEW_KEYS, e.key.toLowerCase())) setMode(VIEW_KEYS[e.key.toLowerCase()]);
    else if (e.key === "f" || e.key === "F") designateAhead();
    else if (e.key === "g" || e.key === "G") dropGrenade();
    else if (e.key === " ") {
      togglePause();
      e.preventDefault();
    }
  });
  $("grenadeBtn").addEventListener("click", () => dropGrenade());
  // F and Space as buttons, for touch screens (shown on any screen: they don't get in the way)
  $("aheadBtn").addEventListener("click", () => designateAhead());
  const pauseBtn = $<HTMLButtonElement>("pauseBtn");
  function togglePause() {
    paused = !paused;
    pauseBtn.textContent = paused ? "Resume" : "Pause";
  }
  pauseBtn.addEventListener("click", togglePause);
  if (!touchUI) {
    $("aheadBtn").textContent += " (F)";
    pauseBtn.title = "Space";
  }
  // destroyed trucks stay destroyed, their wrecks blocking the road, until this brings them all back out of sight
  const trucksBtn = $<HTMLButtonElement>("trucksBtn");
  trucksBtn.addEventListener("click", () => {
    const n = world.respawnCars();
    if (n) banner(n === 1 ? "1 TRUCK BACK ON THE ROAD" : `${n} TRUCKS BACK ON THE ROAD`);
  });
  if (!grenadesOn) for (const el of [$("grenadeBtn"), $("grenadesLeft").parentElement!]) el.style.display = "none";
  window.addEventListener("keyup", (e) => release(e.key));
  window.addEventListener("blur", releaseAll);

  // ---- HUD ------------------------------------------------------------------------------------
  let bannerTimer = 0;
  function banner(text: string) {
    const el = $("banner");
    el.textContent = text;
    el.classList.add("show");
    clearTimeout(bannerTimer);
    bannerTimer = window.setTimeout(() => el.classList.remove("show"), 1600);
  }
  const bar = (id: string, v: number, full: number) => {
    $(id).style.width = `${Math.max(0, Math.min(100, (100 * v) / full))}%`;
  };
  let fps = 60;
  let brainSpeed = 1;
  const updateHud = () => {
    bar("dnaL", rates.hz("DNa02:L"), 40);
    bar("dnaR", rates.hz("DNa02:R"), 40);
    bar("lcL", rates.hz("LC10a:L"), 20);
    bar("lcR", rates.hz("LC10a:R"), 20);
    bar("loomL", rates.hz("LC4:L") + rates.hz("LPLC2:L"), 40);
    bar("loomR", rates.hz("LC4:R") + rates.hz("LPLC2:R"), 40);
    const p = Math.max(-3, Math.min(3, pitchCmd));
    bar("pitchDown", -p, 3);
    bar("pitchUp", p, 3);
    bar("gf", both("DNp01", (n) => rates.hz(n)), 250);
    bar("mdn", both("MDN", (n) => rates.hz(n)), 40);
    bar("wantL", value?.left ?? 0, 0.5);
    bar("wantR", value?.right ?? 0, 0.5);
    // one short line per thing that is happening, most important first
    const status = [
      !food.on ? "Target help is off." : food.diving ? "Diving on the target." : food.locked ? `Locked on, ${world.speedScale.toFixed(1)}x speed.` : "",
      food.painted ? `Nudging ${food.painted.length} LC10a cells.` : "",
      learning
        ? `Learned turn ${mbTurn().toFixed(0)}°/s${food.steer ? "" : " (off)"}, ${food.pulses} reward pulses, ` +
          `${(100 * learning.stats.depressedFraction).toFixed(1)}% of synapses weakened${learning.enabled ? "." : ", learning paused."}`
        : "",
    ];
    $("foodStats").textContent = status.filter(Boolean).join(" ");
    $("rams").textContent = String(world.stats.rams);
    $("hits").textContent = String(world.stats.treeHits);
    $("escapes").textContent = String(world.stats.escapes);
    $("seizures").textContent = String(world.stats.seizures);
    $("flightTime").textContent = `${Math.floor(world.time / 1000)}s`;
    $("carsDestroyed").textContent = String(world.stats.carsDestroyed);
    $("dronesLost").textContent = String(world.stats.dronesLost);
    const gone = world.cars.filter((c, i) => !c && world.carAhead !== i).length;
    trucksBtn.disabled = gone === 0;
    trucksBtn.textContent = gone ? `Respawn trucks (${gone} destroyed)` : "Respawn trucks";
    $("grenadesLeft").textContent = world.droneLost ? "-" : String(world.grenadesLeft);
    $("grenadeBtn").textContent = world.droneLost ? "Grenade (G): no drone" : `Grenade (G): ${world.grenadesLeft} left`;
    const deg = (x: number) => ((x * 180) / Math.PI).toFixed(0);
    const f = world.food ? world.locate(world.food) : null;
    $("brainstats").textContent =
      (bench ? `${bench.status()}\n` : "") +
      `${Math.round(watchdog.spikesPerSec).toLocaleString()} spikes/s · brain ${brainSpeed.toFixed(2)}× · ${Math.round(fps)} fps · seizures ${watchdog.count}\n` +
      `DNa02 L ${rates.hz("DNa02:L").toFixed(0)} / R ${rates.hz("DNa02:R").toFixed(0)} Hz → yaw ${(gains.yaw * yawCmd).toFixed(0)}°/s · ` +
      (learns ? `learned ${mbTurn().toFixed(0)}°/s · ` : "") +
      (avoidGain > 0 ? `avoid ${avoidTurn.toFixed(0)}°/s · ` : "") +
      `pitch ${(gains.pitch * pitchCmd).toFixed(0)}°/s · GF ${both("DNp01", (n) => rates.hz(n)).toFixed(0)} Hz\n` +
      (world.droneLost ? `DRONE LOST · new drone in ${Math.max(0, (world.respawnAt - world.time) / 1000).toFixed(1)} s · ` : "") +
      `alt ${world.y.toFixed(1)} · heading ${deg(world.heading)}° · gaze ${deg(world.pitch)}°` +
      (f ? ` · target ${f.dist.toFixed(0)} away at ${f.az.toFixed(0)}°, ${f.el.toFixed(0)}°${f.occluded ? " (hidden)" : ""}` : "") +
      (paused ? " · PAUSED" : "");
  };

  // ---- tests (?bench=) --------------------------------------------------------------------------
  const harness: Harness = {
    world,
    drive: (key, mv) => (mv > 0 ? press(key, mv) : (release(key), 0)),
    paint: (az, el, mv) => paintAzEl(az, el, mv),
    unpaint: stopPaint,
    designate,
    stimHz: (key) => rates.hz(`stim:${key}`),
    paintedHz: (az, el) => {
      const key = `${az},${el}`;
      let slots = paintSlots.get(key);
      if (!slots) {
        const sel = cells.select({ type: PAINT.type, receptiveField: { az, el, radius: PAINT.radius } });
        slots = Array.from(sel, (n) => lc10Slot.get(n) ?? -1).filter((k) => k >= 0);
        paintSlots.set(key, slots);
      }
      if (!slots.length || !rates.dtMs) return 0;
      let spikes = 0;
      for (const k of slots) spikes += rates.delta[k];
      return (spikes * 1000) / rates.dtMs / slots.length;
    },
    counts: () => ({
      seizures: watchdog.count, escapes: world.stats.escapes, rewardPulses: food.pulses,
      meanWeight: learning?.stats.meanWeight ?? 1, depressed: learning?.stats.depressedFraction ?? 0,
    }),
    settings: { ...gains, trunkR: world.trees[0]?.trunkR ?? WORLD.trunkRadius, food: foodMode, freeze: food.freeze, dive: food.dive, paintNear, markNear, avoid: avoidGain, avoidAdaptMs, avoidOff: avoidWhile, foodSteer: food.steer, reward: food.reward, ground: look.groundContrast, trees: look.treeContrast, mark: look.mark },
  };
  bench = benchKind ? createBench(benchKind, query, harness) : null;

  // for headless runs (web/scripts/smoke.mjs)
  (window as unknown as { flybrainFpv: unknown }).flybrainFpv = {
    summary: () => ({
      brainSeconds: world.time / 1000,
      ...world.stats,
      distance: Math.round(world.stats.distance),
      spikesPerSec: Math.round(watchdog.spikesPerSec),
      brainSpeed: Number(brainSpeed.toFixed(2)),
      fps: Math.round(fps),
      rewardPulses: food.pulses,
      /** Chromium only: the JS heap, to catch anything that grows over a long run */
      heapMB: Math.round(((performance as unknown as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize ?? 0) / 1e6),
      meanWeight: learning?.stats.meanWeight ?? 1,
    }),
    trace: () => ({ columns: TRACE_COLUMNS, rows: trace }),
    bench: () => bench?.results() ?? null,
    /** a stimulation key, or "g" to drop a grenade (so play.sh's PRESS="g:10-11" drops one at 10 s) */
    press: (key: string) => (key === "g" || key === "G" ? Number(dropGrenade()) : press(key)),
    grenade: dropGrenade,
    /** the world itself, for headless checks (web/scripts: trucks stopping behind a wreck, the respawn button) */
    world,
    /** and the renderer (its pickRay, view and zoom), for the phone check (web/scripts/check-mobile.mjs) */
    renderer,
    release,
    /** the GPU pieces, and a way to hold the frame loop, for timing them one at a time (web/scripts/profile.mjs) */
    brain,
    get brainView() {
      return brainView;
    },
    suspend: (on: boolean) => (suspended = on),
  };

  // ---- loop -------------------------------------------------------------------------------------
  let last = performance.now();
  /** held by profile.mjs while it times the GPU passes on their own */
  let suspended = false;
  /** ms since the panel views and the HUD were last redrawn (LOOP.panelMs) */
  let panelDt = LOOP.panelMs;
  let budget = LOOP.maxBrainMsPerFrame;
  const frame = (now: number) => {
    if (suspended) {
      last = now;
      requestAnimationFrame(frame);
      return;
    }
    const elapsed = now - last;
    last = now;
    const realDt = Math.min(100, elapsed);
    fps += (1000 / Math.max(1, elapsed) - fps) * 0.05;
    // Keep the frame rate up on a busy GPU: first trace the view at fewer pixels (down to LOOP.minRenderScale), then
    // simulate less brain time per frame (fly-addiction). "Busy" is against the display's own frame interval (its
    // interval, measured while loading, at least 60 Hz's): a phone in low-power mode draws at 30 Hz and is not busy.
    const frameMs = Math.max(1000 / 60, Number.isFinite(displayMs) ? displayMs : 0);
    if (elapsed > frameMs * 1.45) {
      if (renderer.resScale > LOOP.minRenderScale) renderer.resScale = Math.max(LOOP.minRenderScale, renderer.resScale * 0.97);
      else budget = Math.max(4, budget * 0.95);
    } else if (elapsed < frameMs * 1.15) {
      if (budget < LOOP.maxBrainMsPerFrame) budget = Math.min(LOOP.maxBrainMsPerFrame, budget * 1.02);
      else renderer.resScale = Math.min(1, renderer.resScale * 1.01);
    }
    const brainMs = paused ? 0 : Math.min(realDt, budget);
    brainSpeed += (brainMs / Math.max(1, elapsed) - brainSpeed) * 0.05;

    if (brainMs > 0) {
      bench?.frame(brainMs);
      seekFood();
      bomb();
      const cmd: Commands = {
        yawRate: turnNow(),
        // diving (?dive, MODELED): the assist steers the pitch toward the food instead of the brain and the spring
        pitchRate: food.diving
          ? LOCK.diveGain * food.diveEl
          : gains.pitch * pitchCmd - CONTROL.pitchSpring * ((world.pitch * 180) / Math.PI) + altitudePull(),
        escape: escapeNow,
      };
      if (escapeNow && !world.frozen && !world.droneLost) banner("ESCAPE HOP");
      escapeNow = false;
      const before = { heading: world.heading, y: world.y };
      world.update(brainMs, cmd);
      bench?.after(brainMs, before, cmd);
    }
    renderer.seizure = Math.max(0, renderer.seizure - realDt / 700);
    renderer.escape = Math.max(0, 1 - (world.time - lastEscape) / 600);
    renderer.writeScene(world, !food.on || (markNear > 0 && foodDist < markNear) ? { ...look, mark: false } : look);

    const encoder = device.createCommandEncoder({ label: "frame" });
    if (brainMs > 0) brain.encode(encoder, brainMs);
    renderer.render(encoder, world, now / 1000);
    // the panel's brain and eye views, only while they are on screen (on a phone the panel is scrolled away), and the
    // HUD: at most every LOOP.panelMs
    panelDt += realDt;
    const panelDue = panelDt >= LOOP.panelMs;
    if (panelShown && panelDue) brainView.render(encoder, panelDt);
    device.queue.submit([encoder.finish()]);
    if (brainMs > 0) brain.afterSubmit();
    if (panelDue) {
      updateHud();
      panelDt = 0;
    }
    osd.draw(world, { mode: renderer.mode, brainSpeed, fps, seizure: renderer.seizure, escape: renderer.escape, paused, locked: food.locked, diving: food.diving });
    requestAnimationFrame(frame);
  };

  const brainView = new BrainView(brain, format, $<HTMLCanvasElement>("brain"), $<HTMLCanvasElement>("eye"));
  let panelShown = true;
  const panelSeen = new Set<Element>();
  const panelWatch = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) panelSeen.add(e.target);
      else panelSeen.delete(e.target);
    }
    panelShown = panelSeen.size > 0;
  });
  for (const id of ["brain", "eye"]) panelWatch.observe($(id));
  const start = $<HTMLButtonElement>("start");
  start.hidden = false;
  start.onclick = () => {
    $("loading").style.display = "none";
    measuring = false;
    last = performance.now();
    requestAnimationFrame(frame);
  };
}

main().catch((e) => {
  // anything that throws while the brain loads or the game is set up, instead of a loading screen that never ends
  console.error(e);
  fail(`It stopped while starting: ${e instanceof Error ? e.message : String(e)}`);
});
