// GPU cost of each part of a frame, timed one at a time: the brain (per 16.7 ms of brain time), the player's view in
// each mode and render scale, and the panel's brain and eye views. It flies for a while first (so the brain is as
// busy as in play), then holds the frame loop and submits each part back to back, dividing the wall time by the
// count: the throughput a slower GPU would be bound by. Serves DIST like smoke.mjs.
//
//   docker run --rm --device /dev/dri/renderD128 --ipc=host -v "$PWD/web:/app:ro" -v "$PWD/.cache/profile:/out" \
//     -e DIST=/app/dist flygames-gputest sh -c 'cp /app/scripts/profile.mjs /runner/ && node /runner/profile.mjs'
// Env: VIEWPORT (default 1500x900), DPR (device pixel ratio, default 1), FLY (seconds of flight first, default 12),
// ADAPTER=swiftshader (no GPU: everything far slower, but in proportion), REPS (submits per measurement, default 60).
import { createServer } from "node:http";
import { readFile, stat, writeFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import pw from "playwright-core";

const root = process.env.DIST ?? "/app/dist";
const out = process.env.OUT ?? "/out";
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = createServer(async (req, res) => {
  const file = join(root, normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^\/+/, "") || "index.html");
  try {
    if (!(await stat(file)).isFile()) throw new Error();
    const headers = { "content-type": types[extname(file)] ?? "application/octet-stream" };
    if (file.endsWith(".bin.gz")) headers["content-encoding"] = "gzip";
    res.writeHead(200, headers);
    res.end(await readFile(file));
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((r) => server.listen(4173, "127.0.0.1", r));
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"];
if (process.env.ADAPTER === "swiftshader") {
  args.push("--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader", "--enable-unsafe-swiftshader");
} else {
  args.push("--use-angle=vulkan");
}
const browser = await pw.chromium.launch(
  process.env.CHROMIUM
    ? { headless: true, executablePath: process.env.CHROMIUM, args: [...args, "--headless=new", "--no-sandbox"] }
    : { headless: true, channel: "chromium", args },
);
const [vw, vh] = (process.env.VIEWPORT || "1500x900").split("x").map(Number);
const page = await browser.newPage({ viewport: { width: vw, height: vh }, deviceScaleFactor: Number(process.env.DPR ?? 1) });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(`http://127.0.0.1:4173/?seed=1${process.env.QUERY ?? ""}`);
await page.waitForSelector("#start:not([hidden])", { timeout: 600_000 });
await page.click("#start");
await page.waitForTimeout(Number(process.env.FLY ?? 12) * 1000);
const before = await page.evaluate(() => window.flybrainFpv.summary());

const result = await page.evaluate(async (reps) => {
  const g = window.flybrainFpv;
  g.suspend(true);
  const { brain, renderer, world } = g;
  const view = g.brainView;
  const dev = brain.device;
  const run = async (fn, n) => {
    for (let i = 0; i < n; i++) {
      const enc = dev.createCommandEncoder();
      fn(enc);
      dev.queue.submit([enc.finish()]);
      brain.afterSubmit();
    }
    await dev.queue.onSubmittedWorkDone();
  };
  /** ms per submit, the median of three runs */
  const time = async (fn) => {
    await run(fn, 5);
    const runs = [];
    for (let k = 0; k < 3; k++) {
      const t0 = performance.now();
      await run(fn, reps);
      runs.push((performance.now() - t0) / reps);
    }
    return Number(runs.sort((a, b) => a - b)[1].toFixed(3));
  };
  const out = {};
  const canvas = document.getElementById("view");
  out.canvas = `${canvas.width}x${canvas.height}`;
  out.connectome = { n: brain.n, ng: brain.ng, ...brain.info.stats };
  out.nothing = await time(() => {});
  out.brain16ms = await time((enc) => brain.encode(enc, 1000 / 60));
  const mode = renderer.mode;
  const scale = renderer.resScale;
  for (const [name, m] of [["fpv", 3], ["chase", 1], ["flyeye", 0]]) {
    renderer.mode = m;
    for (const s of [1, 0.5]) {
      renderer.resScale = s;
      out[`${name}@${s}`] = await time((enc) => renderer.render(enc, world, performance.now() / 1000));
    }
  }
  renderer.mode = mode;
  renderer.resScale = scale;
  out.panelViews = await time((enc) => view.render(enc, 16));
  out.frameFpv = await time((enc) => {
    brain.encode(enc, 1000 / 60);
    renderer.render(enc, world, performance.now() / 1000);
    view.render(enc, 16);
  });
  g.suspend(false);
  return out;
}, Number(process.env.REPS ?? 60));

const report = { ...result, spikesPerSec: before.spikesPerSec, fpsWhileFlying: before.fps, errors };
console.log(JSON.stringify(report, null, 1));
await writeFile(`${out}/profile.json`, JSON.stringify(report, null, 1));
await browser.close();
server.close();
