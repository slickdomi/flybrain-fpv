// Reads the approach test's bench.json files (?bench=approach) and asks: as the drone flies straight at a tree set off
// to one side, what do the looming cells, the giant fiber and DNa02 do, relative to the empty-field baseline just
// before, and how many seconds before reaching the tree? A usable avoidance signal would differ by the tree's side
// (right minus left looming positive for a tree on the right, say) early enough to turn.
//
//   docker run --rm -v "$PWD:/app" -w /app node:22-alpine node web/scripts/analyze-approach.mjs .cache/approach
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const files = [];
const walk = (d) => {
  for (const e of readdirSync(d)) {
    const p = join(d, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (e === "bench.json") files.push(p);
  }
};
for (const d of process.argv.slice(2)) walk(d);
const runs = files.map((p) => JSON.parse(readFileSync(p, "utf8"))).filter((r) => r.protocol === "approach");
if (!runs.length) {
  console.error("no approach bench.json under", process.argv.slice(2).join(" "));
  process.exit(2);
}

const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN);
const f = (x, d = 1) => (Number.isFinite(x) ? (x >= 0 ? " " : "") + x.toFixed(d) : "   -");
const c = Object.fromEntries(runs[0].columns.map((n, i) => [n, i]));
// seconds relative to reaching the tree's depth (the drone is 1.6 units from a centred trunk's axis 0.2 s before)
const BINS = [[-9, -7], [-7, -5], [-5, -3], [-3, -2], [-2, -1], [-1, -0.5], [-0.5, 0], [0, 1]];

const trials = runs.flatMap((r) => r.trials.map((t) => ({ ...t, seed: r.seed })));
const offsets = [...new Set(trials.map((t) => t.offset))].sort((a, b) => a - b);
console.log(
  `${runs.length} run(s), seeds ${[...new Set(runs.map((r) => r.seed))].join(",")}, trunk radius ${runs[0].trunkR}, ` +
    `tree ${runs[0].dist} units ahead · ${trials.length} trials, ${trials.filter((t) => t.seized).length} with a seizure (left out)`,
);

/** per trial: the change from its own baseline (the settle rows) in each time bin */
const change = (t, col) => {
  const base = mean(t.rows.filter((r) => !Number.isFinite(r[c.t])).map((r) => r[col]));
  return BINS.map(([a, b]) => mean(t.rows.filter((r) => r[c.t] >= a && r[c.t] < b).map((r) => r[col])) - base);
};
const asym = (t) => {
  const r = change(t, c.loomR);
  const l = change(t, c.loomL);
  return r.map((x, i) => x - l[i]);
};

const header = "offset   " + BINS.map(([a, b]) => `${a}..${b}s`.padStart(8)).join("");
for (const [label, fn] of [
  ["looming right - left (LC4 + LPLC2, Hz): + means the right eye's looming cells rose more", asym],
  ["giant fiber DNp01, L + R (Hz above baseline)", (t) => change(t, c.gf)],
  ["DNa02 right - left (Hz above baseline): + turns right, i.e. toward a tree on the right", (t) => change(t, c.diff)],
]) {
  console.log(`\n${label}\n${header}`);
  for (const o of offsets) {
    const ts = trials.filter((t) => t.offset === o && !t.seized);
    if (!ts.length) continue;
    const per = ts.map(fn);
    const m = BINS.map((_, i) => mean(per.map((p) => p[i]).filter(Number.isFinite)));
    console.log(`${(o > 0 ? "+" : "") + o}`.padEnd(9) + m.map((x) => f(x).padStart(8)).join("") + `   (n ${ts.length})`);
  }
}

console.log(`\nescape hops (share of trials with one in the bin)\n${header}`);
for (const o of offsets) {
  const ts = trials.filter((t) => t.offset === o && !t.seized);
  if (!ts.length) continue;
  const m = BINS.map(([a, b]) => ts.filter((t) => t.rows.some((r) => r[c.t] >= a && r[c.t] < b && r[c.escape] === 1)).length / ts.length);
  console.log(`${(o > 0 ? "+" : "") + o}`.padEnd(9) + m.map((x) => `${Math.round(100 * x)}%`.padStart(8)).join(""));
}

// the one-number answer: in each bin, does the looming asymmetry take the tree's side? (centred trees left out)
console.log("\nlooming asymmetry on the tree's side (share of side-offset trials where sign(R - L) = sign(offset))");
const sided = trials.filter((t) => t.offset !== 0 && !t.seized);
const agree = BINS.map((_, i) => {
  const xs = sided.map((t) => [Math.sign(asym(t)[i]), Math.sign(t.offset)]).filter(([a]) => a !== 0 && Number.isFinite(a));
  return xs.filter(([a, b]) => a === b).length / xs.length;
});
console.log(header.replace("offset   ", "         ") + "\n" + " ".repeat(9) + agree.map((x) => `${Math.round(100 * x)}%`.padStart(8)).join(""));
