// The control tests' verdicts: reads every bench.json under the given directories (scripts/test-controls.sh
// writes them), prints what each stimulus did to the drone, and grades it.
//
// Two tiers, because they mean different things:
//   PLUMBING       the drive reached the cells it was aimed at, the food designation made the mushroom body
//                  learn, and nothing crashed. A failure here is a bug; the script exits non-zero.
//   EFFECTIVENESS  the stimulus moved the drone the way its pathway predicts, consistently and by a useful
//                  amount. These can fail without anything being broken -- that is what they measure.
//
//   docker run --rm -v "$PWD:/app" -w /app node:22-alpine node web/scripts/check-controls.mjs .cache/tests
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

/** What each stimulus should do, from the pathway it drives. `turn`: -1 left, +1 right; `climb`: +1 up. */
const EXPECT = {
  1: { name: "LC10a L", turn: -1, why: "LC10 -> AOTU -> DNa02 turns toward a small dark object on that side" },
  2: { name: "LC10a R", turn: +1, why: "the same, on the right" },
  paintL: { name: "paint -40°", turn: -1, why: "LC10a cells looking 40° left" },
  paintR: { name: "paint +40°", turn: +1, why: "LC10a cells looking 40° right" },
  3: { name: "DNa02 L", turn: -1, why: "the steering command neuron itself" },
  4: { name: "DNa02 R", turn: +1, why: "the steering command neuron itself" },
  5: { name: "DNp53", climb: +1, why: "Aimbug's fly looked up with it" },
  6: { name: "LC4", escape: true, why: "looming detectors drive the giant fiber" },
  7: { name: "DNp01", escape: true, why: "the giant fiber itself" },
  8: { name: "MDN", why: "backward walking in a real fly; not wired to the drone" },
  9: { name: "P1", why: "male arousal; no motor mapping here" },
  // smell (TEST_STIMS): no expectation yet, that is what the test is for; CO2 is aversive to real flies
  vornL: { name: "ORN_V L" }, vornR: { name: "ORN_V R" }, vpnL: { name: "V PN L" }, vpnR: { name: "V PN R" },
  dm1L: { name: "ORN_DM1 L" }, dm1R: { name: "ORN_DM1 R" },
};
/** EFFECTIVENESS thresholds: in at least this share of trials, and at least this big on average */
const CONSISTENT = 0.75;
const MIN_TURN_DEG = 10; // over the 2 s the stimulus is held, i.e. 5 deg/s
const MIN_PITCH = 2; // deg/s of commanded pitch
const MIN_CLIMB = 0.5; // world units in 2 s (closed loop)
/** PLUMBING: the stimulated group's own rate must rise by this much (Hz) on average */
const MIN_TARGET_HZ = 2;

const files = [];
const walk = (d) => {
  for (const e of readdirSync(d)) {
    const p = join(d, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (e === "bench.json") files.push(p);
  }
};
for (const d of process.argv.slice(2)) walk(d);
if (!files.length) {
  console.error("no bench.json found under", process.argv.slice(2).join(" "));
  process.exit(2);
}

const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN);
const sd = (a) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
};
const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
};
const f = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "-");
const pct = (x) => (Number.isFinite(x) ? `${Math.round(100 * x)}%` : "-");
function corr(xs, ys) {
  const mx = mean(xs);
  const my = mean(ys);
  let a = 0, b = 0, c = 0;
  for (let i = 0; i < xs.length; i++) {
    a += (xs[i] - mx) * (ys[i] - my);
    b += (xs[i] - mx) ** 2;
    c += (ys[i] - my) ** 2;
  }
  return a / Math.sqrt(b * c);
}

let plumbingFailures = 0;
const verdicts = [];
const results = files.map((p) => ({ path: p, ...JSON.parse(readFileSync(p, "utf8")) }));
for (const r of results) {
  if (!r.done) {
    console.log(`PLUMBING FAIL  ${r.path}: the protocol did not finish (the run was cut off, or the page stalled)`);
    plumbingFailures++;
  }
}

// ---- stim --------------------------------------------------------------------------------------------
const stim = results.filter((r) => r.protocol === "stim");
for (const loop of ["open", "closed"]) {
  const runs = stim.filter((r) => r.loop === loop);
  if (!runs.length) continue;
  const trials = runs.flatMap((r) => r.trials.map((t) => ({ ...t, seed: r.seed })));
  const seeds = [...new Set(runs.map((r) => r.seed))].join(",");
  const excluded = trials.filter((t) => t.excluded);
  console.log(`\n=== stimulation, ${loop} loop (${loop === "open" ? "drone held still; what the commands would have done" : "drone flying; what it actually did"}) · seeds ${seeds} · ${trials.length} trials, ${excluded.length} excluded (${[...new Set(excluded.map((t) => t.excluded))].join(", ") || "none"})`);
  console.log(
    "stimulus      mV  n  cells  own rate Hz     DNa02 R-L Hz   turn °/s        turned ° (sd)    consistent  latency  pitch °/s   climb      escapes",
  );
  const keys = [...new Set(trials.map((t) => String(t.stim)))];
  const mvs = [...new Set(trials.map((t) => t.mv))].sort((a, b) => a - b);
  for (const key of keys) {
    const ex = EXPECT[key] ?? { name: key };
    for (const mv of mvs) {
      const seizedN = trials.filter((t) => String(t.stim) === key && t.mv === mv && t.excluded === "seizure").length;
      const ts = trials.filter((t) => String(t.stim) === key && t.mv === mv && !t.excluded);
      if (!ts.length) {
        if (seizedN) console.log(`${(ex.name ?? key).padEnd(13)} ${String(mv).padStart(3)}  every trial seized (${seizedN})`);
        continue;
      }
      const own = ts.map((t) => t.during.targetHz - t.base.targetHz);
      const turned = ts.map((t) => t.turnedDeg);
      const dTurn = ts.map((t) => t.during.turn - t.base.turn);
      const dDiff = ts.map((t) => t.during.diff - t.base.diff);
      const dPitch = ts.map((t) => t.during.pitch - t.base.pitch);
      const climbs = ts.map((t) => t.climbed);
      const esc = ts.filter((t) => t.escapes > 0).length / ts.length;
      // latency: first moment the commanded turn moved half of its peak change away from baseline
      const lat = ts
        .map((t) => {
          const peak = Math.max(...t.trace.filter((_, i) => i % 2 === 1).map((v) => Math.abs(v - t.base.turn)));
          for (let i = 0; i < t.trace.length; i += 2) if (Math.abs(t.trace[i + 1] - t.base.turn) >= peak / 2 && peak > 2) return t.trace[i];
          return NaN;
        })
        .filter(Number.isFinite);
      let consistent = NaN;
      if (ex.turn) consistent = turned.filter((x) => Math.sign(x) === ex.turn).length / ts.length;
      else if (ex.climb) consistent = (loop === "open" ? dPitch : climbs).filter((x) => Math.sign(x) === ex.climb).length / ts.length;
      else if (ex.escape) consistent = esc;
      const cells = ts[0].cells;
      console.log(
        `${(ex.name ?? key).padEnd(13)} ${String(mv).padStart(3)} ${String(ts.length).padStart(2)} ${String(cells).padStart(6)}  ` +
          `${f(mean(ts.map((t) => t.base.targetHz)))} -> ${f(mean(ts.map((t) => t.during.targetHz)))}`.padEnd(16) +
          `${f(mean(dDiff)).padStart(6)}        ${f(mean(dTurn)).padStart(6)}          ${f(mean(turned)).padStart(6)} (${f(sd(turned), 0)})`.padEnd(46) +
          `${pct(consistent).padStart(5)}      ${Number.isFinite(median(lat)) ? `${median(lat)} ms` : "-"}`.padEnd(20) +
          `${f(mean(dPitch)).padStart(6)}   ${f(mean(climbs), 2).padStart(6)}   ${pct(esc).padStart(5)}` +
          (seizedN ? `   (+${seizedN} trial${seizedN > 1 ? "s" : ""} excluded: seizure)` : ""),
      );
      // PLUMBING: the drive reached its own cells
      // a painted phantom's own rate is read from just the cells it drives (bench.ts paintedHz), like every other group
      const need = MIN_TARGET_HZ;
      // on average: whether each single trial clears it is the effectiveness tier's question (a lone DNa02 cell at
      // 10 mV sits barely over threshold, and one trial in three rose under 2 Hz while every trial still turned)
      const reached = cells > 0 && mean(own) >= need;
      if (!reached) {
        plumbingFailures++;
        verdicts.push(`PLUMBING FAIL  ${loop} ${ex.name} ${mv} mV: ${cells > 0 ? `its own cells rose ${f(mean(own))} Hz on average (need ≥${need})` : "no cells were selected"}`);
      }
      // EFFECTIVENESS
      let ok = null;
      let what = "";
      if (ex.turn) {
        ok = consistent >= CONSISTENT && Math.sign(mean(turned)) === ex.turn && Math.abs(mean(turned)) >= MIN_TURN_DEG;
        what = `turns ${ex.turn > 0 ? "right" : "left"}: ${f(mean(turned))}° in 2 s, ${pct(consistent)} of trials the right way (need ≥${MIN_TURN_DEG}° and ${pct(CONSISTENT)})`;
      } else if (ex.climb) {
        const m = loop === "open" ? mean(dPitch) : mean(climbs);
        const need = loop === "open" ? MIN_PITCH : MIN_CLIMB;
        ok = consistent >= CONSISTENT && m >= need;
        what = `climbs: ${loop === "open" ? `${f(m)} °/s of pitch` : `${f(m, 2)} units`}, ${pct(consistent)} of trials up (need ≥${need} and ${pct(CONSISTENT)})`;
      } else if (ex.escape) {
        ok = esc >= CONSISTENT;
        what = `escape hop in ${pct(esc)} of trials (need ${pct(CONSISTENT)})`;
      }
      if (ok !== null) verdicts.push(`${ok ? "EFFECTIVE    " : "NOT EFFECTIVE"}  ${loop.padEnd(6)} ${ex.name.padEnd(11)} ${String(mv).padStart(2)} mV  ${what}`);
    }
  }
}

// ---- chase -------------------------------------------------------------------------------------------
const chase = results.filter((r) => r.protocol === "chase");
if (chase.length) {
  console.log(`\n=== chase: a truck designated as the target, the drone flying, every contact fatal`);
  console.log(
    "arm                   seed  brain-s  hits/min  sight->hit s  crashes/min tree/ground  lost/min  truck ahead  DNa02 toward (corr)  seizures",
  );
  const arms = {};
  for (const r of chase) {
    const c = Object.fromEntries(r.columns.map((n, i) => [n, i]));
    const rows = r.rows;
    const s = r.settings;
    // the arm is named by the settings that differ between runs: the food mode, forest (thick trunks) and avoidance
    const arm = !r.designate
      ? "no target"
      : `${s.food}${s.trunkR && s.trunkR !== 0.3 ? ` trunk${s.trunkR}` : ""}${s.avoid ? ` avoid${s.avoid}${s.avoidOff === "view" ? "(off in view)" : ""}` : ""}`;
    const visible = rows.filter((x) => x[c.visible] === 1 && Math.abs(x[c.az]) < 120 && x[c.dist] < 30);
    const lateral = visible.filter((x) => Math.abs(x[c.az]) > 10);
    const moving = lateral.filter((x) => Math.abs(x[c.dnaTurn]) > 1);
    const dna = {
      share: moving.filter((x) => Math.sign(x[c.dnaTurn]) === Math.sign(x[c.az])).length / moving.length,
      corr: corr(lateral.map((x) => x[c.dnaTurn]), lateral.map((x) => x[c.az])),
    };
    const secs = rows.length ? rows[rows.length - 1][c.t] / 1000 : 0;
    const perMin = (n) => (n / Math.max(1, secs)) * 60;
    const m = {
      seed: r.seed,
      hits: perMin(r.stats.foodReached),
      latency: median(r.latencies ?? []) / 1000,
      tree: perMin(r.stats.crashTree ?? 0),
      ground: perMin(r.stats.crashGround ?? 0),
      lost: perMin(r.lost ?? 0),
      ahead: rows.filter((x) => x[c.visible] === 1 && Math.abs(x[c.az]) < 30 && x[c.dist] < 30).length / rows.length,
      dna, seizures: r.counts.seizures,
    };
    (arms[arm] ??= []).push(m);
    console.log(
      `${arm.padEnd(21)} ${String(r.seed).padStart(4)}  ${f(secs, 0).padStart(7)}  ${f(m.hits, 2).padStart(8)}  ${f(m.latency, 1).padStart(12)}  ` +
        `${f(m.tree, 2).padStart(10)} / ${f(m.ground, 2).padEnd(6)}  ${f(m.lost, 2).padStart(8)}  ${pct(m.ahead).padStart(11)}  ` +
        `${pct(dna.share).padStart(8)} (${f(dna.corr, 2)})`.padEnd(21) +
        `${String(m.seizures).padStart(8)}`,
    );
    // PLUMBING (?food=learn only): designating the target made the mushroom body learn
    if (r.designate && s.food === "learn" && !(r.counts.rewardPulses > 0 && r.counts.meanWeight < 1)) {
      plumbingFailures++;
      verdicts.push(`PLUMBING FAIL  chase seed ${r.seed} ${arm}: ${r.counts.rewardPulses} reward pulses and mean weight ${f(r.counts.meanWeight, 4)} (nothing learned)`);
    }
  }

  const summary = Object.entries(arms).map(([arm, ms]) => ({
    arm, n: ms.length, hits: mean(ms.map((m) => m.hits)), latency: mean(ms.map((m) => m.latency).filter(Number.isFinite)),
    crashes: mean(ms.map((m) => m.tree + m.ground)), ahead: mean(ms.map((m) => m.ahead)), toward: mean(ms.map((m) => m.dna.share)),
  }));
  if (summary.length > 1) {
    console.log("\narm                   runs  hits/min  sight->hit s  crashes/min  truck ahead  DNa02 toward");
    for (const s of summary.sort((a, b) => b.hits - a.hits)) {
      console.log(
        `${s.arm.padEnd(21)} ${String(s.n).padStart(4)}  ${f(s.hits, 2).padStart(8)}  ${f(s.latency, 1).padStart(12)}  ${f(s.crashes, 2).padStart(11)}  ` +
          `${pct(s.ahead).padStart(11)}  ${pct(s.toward).padStart(12)}`,
      );
    }
    // EFFECTIVENESS: each target drive against the fly on its own in the same world (food mode none)
    const trunkOf = (arm) => arm.match(/ trunk\S+/)?.[0] ?? "";
    for (const s of summary.filter((s) => !s.arm.startsWith("none") && s.arm !== "no target")) {
      const none = summary.find((n) => n.arm.startsWith("none") && trunkOf(n.arm) === trunkOf(s.arm));
      if (!none) continue;
      verdicts.push(
        `${s.hits > none.hits ? "EFFECTIVE    " : "NOT EFFECTIVE"}  chase  ${s.arm} beats the fly alone (${none.arm}): ${f(s.hits, 2)} vs ${f(none.hits, 2)} hits/min`,
      );
    }
  }
}

// ---- survive -----------------------------------------------------------------------------------------
const survive = results.filter((r) => r.protocol === "survive");
if (survive.length) {
  console.log(`\n=== survive: the forest, every contact fatal, nothing to chase`);
  console.log("arm                      seed  brain-s  crashes/min  tree  ground  mean flight s  seizures  patches visited (of 100)");
  const arms = {};
  for (const r of survive) {
    const s = r.settings;
    const arm = `trunk ${s.trunkR ?? 0.3} · avoid ${s.avoid ?? 0} · hop ${s.escapeHz > 0 ? "on" : "off"}`;
    const mins = r.secs / 60;
    const tree = r.stats.crashTree ?? 0;
    const ground = r.stats.crashGround ?? 0;
    const flights = [...(r.lives ?? []), ...(r.flying ? [r.flying] : [])];
    const m = { seed: r.seed, crashes: (tree + ground) / mins, tree: tree / mins, ground: ground / mins, flight: mean(flights) / 1000, seizures: r.counts.seizures, patches: r.patches };
    (arms[arm] ??= []).push(m);
    console.log(
      `${arm.padEnd(24)} ${String(r.seed).padStart(4)}  ${f(r.secs, 0).padStart(7)}  ${f(m.crashes, 2).padStart(11)}  ${f(m.tree, 2).padStart(4)}  ${f(m.ground, 2).padStart(6)}  ` +
        `${f(m.flight, 0).padStart(13)}  ${String(m.seizures).padStart(8)}  ${String(m.patches ?? "-").padStart(8)}`,
    );
  }
  if (Object.keys(arms).length > 1) {
    console.log("\narm                      runs  crashes/min  tree  ground  mean flight s");
    for (const [arm, ms] of Object.entries(arms).sort((a, b) => mean(a[1].map((m) => m.crashes)) - mean(b[1].map((m) => m.crashes)))) {
      console.log(
        `${arm.padEnd(24)} ${String(ms.length).padStart(4)}  ${f(mean(ms.map((m) => m.crashes)), 2).padStart(11)}  ${f(mean(ms.map((m) => m.tree)), 2).padStart(4)}  ` +
          `${f(mean(ms.map((m) => m.ground)), 2).padStart(6)}  ${f(mean(ms.map((m) => m.flight)), 0).padStart(13)}`,
      );
    }
  }
}

console.log("\n=== verdicts");
for (const v of verdicts) console.log(v);
const eff = verdicts.filter((v) => v.startsWith("EFFECTIVE ")).length;
// DECISION lines are the food-mode comparison's outcome, not a grade
const not = verdicts.filter((v) => v.startsWith("NOT EFFECTIVE")).length;
console.log(`\n${plumbingFailures ? `${plumbingFailures} PLUMBING FAILURE(S)` : "plumbing: all passed"} · effectiveness: ${eff} effective, ${not} not`);
process.exit(plumbingFailures ? 1 : 0);
