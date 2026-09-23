// Headless flight: serves web/dist, opens it in Chromium with WebGPU, lets the fly fly for a while, and reports
// console errors, the flight stats, screenshots of every view and a close-up of the drone. This is the only thing
// that actually executes the WGSL -- the typecheck says nothing about shaders compiling or bind groups matching.
// Adapted from fly-addiction's runner.
//
// Runs inside Docker (the flygames-gputest image from ../fly-games): see scripts/play.sh.
//
// Environment: SECONDS (wall-clock seconds to run), QUERY (URL knobs), ADAPTER (hardware | swiftshader),
// VIEWPORT (e.g. 1600x900), APP_URL (test a running dev server instead of dist), PRESS (held stimulation, e.g.
// "2:10-15" holds key 2 from 10 s to 15 s).
import { createServer } from "node:http";
import { readFile, stat, writeFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import pw from "playwright-core";

const { chromium } = pw;

const root = process.env.DIST ?? "/app/dist";
const out = process.env.OUT ?? "/out";
const seconds = Number(process.env.SECONDS ?? 30);
const adapter = process.env.ADAPTER ?? "swiftshader";

const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".gz": "application/gzip" };
const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^\/+/, "");
  const file = join(root, path || "index.html");
  try {
    if (!(await stat(file)).isFile()) throw new Error();
    const headers = { "content-type": types[extname(file)] ?? "application/octet-stream" };
    // the connectome shards are stored gzipped and served as-is
    if (file.endsWith(".bin.gz")) headers["content-encoding"] = "gzip";
    res.writeHead(200, headers);
    res.end(await readFile(file));
  } catch {
    res.writeHead(404);
    res.end();
  }
});
if (!process.env.APP_URL) await new Promise((r) => server.listen(4173, "127.0.0.1", r));

const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"];
if (adapter === "swiftshader") {
  args.push("--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader", "--enable-unsafe-swiftshader");
} else {
  args.push("--use-angle=vulkan");
}
const browser = await chromium.launch(
  process.env.CHROMIUM
    ? { headless: true, executablePath: process.env.CHROMIUM, args: [...args, "--headless=new", "--no-sandbox"] }
    : { headless: true, channel: "chromium", args },
);
const [vw, vh] = (process.env.VIEWPORT || "1500x900").split("x").map(Number);
const page = await browser.newPage({ viewport: { width: vw || 1500, height: vh || 900 } });

const logs = [];
const errors = [];
page.on("console", (m) => {
  const line = `[${m.type()}] ${m.text()}`;
  logs.push(line);
  if (m.type() === "error") errors.push(line);
});
page.on("pageerror", (e) => errors.push(`[pageerror] ${e.message}`));

const query = new URLSearchParams(process.env.QUERY ?? "");
const url = `${process.env.APP_URL ?? "http://127.0.0.1:4173/"}${query.toString() ? `?${query}` : ""}`;
console.log(`opening ${url} (adapter ${adapter})`);
await page.goto(url);

await page.waitForSelector("#start:not([hidden]), #loadError:not([hidden])", { timeout: 300_000 });
const loadError = await page.$("#loadError:not([hidden])");
if (loadError) {
  console.error("LOAD FAILED:", (await loadError.textContent())?.trim());
  console.error(logs.slice(-25).join("\n"));
  await browser.close();
  server.close();
  process.exit(1);
}
console.log("loaded:", (await page.textContent("#loadLabel"))?.trim());

await page.click("#start");
const t0 = Date.now();
let last = 0;
// PRESS="2:10-15,4:20-25": hold stimulation key 2 from 10 s to 15 s of wall time, and so on
const presses = (process.env.PRESS ?? "").split(",").filter(Boolean).map((p) => {
  const [key, span] = p.split(":");
  const [a, b] = span.split("-").map(Number);
  return { key, a, b, on: false };
});
const samples = [];
while ((Date.now() - t0) / 1000 < seconds) {
  await page.waitForTimeout(250);
  // a test (?bench=) marks the page done when its protocol ends; SECONDS is then only a cap
  if (await page.evaluate(() => document.documentElement.dataset.done === "1")) {
    console.log(`  test done after ${((Date.now() - t0) / 1000).toFixed(0)} s wall`);
    break;
  }
  const el = (Date.now() - t0) / 1000;
  for (const p of presses) {
    const want = el >= p.a && el < p.b;
    if (want !== p.on) {
      p.on = want;
      await page.evaluate(([k, on]) => (on ? window.flybrainFpv.press(k) : window.flybrainFpv.release(k)), [p.key, want]);
      console.log(`  ${el.toFixed(1)}s ${want ? "press" : "release"} ${p.key}`);
    }
  }
  if (el - last >= 5) {
    last = el;
    const s = await page.evaluate(() => window.flybrainFpv.summary());
    samples.push({ wall: el, ...s });
    const stats = (await page.textContent("#brainstats"))?.replace(/\n/g, " · ").trim();
    console.log(`  ${el.toFixed(0)}s | brain ${s.brainSpeed}x fps ${s.fps} spikes ${s.spikesPerSec} heap ${s.heapMB} MB | rams ${s.rams} crashes ${s.crashTree}+${s.crashGround} seiz ${s.seizures}`);
  }
}

const summary = await page.evaluate(() => window.flybrainFpv.summary());
console.log("SUMMARY", JSON.stringify(summary));

try {
  // every view, by its key (main.ts VIEW_KEYS)
  for (const [key, name] of [["q", "fpv"], ["w", "eye"], ["e", "chase"], ["r", "mosaic"]]) {
    await page.keyboard.press(key);
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${out}/${name}.png` });
  }
  // the drone close up from its front quarter, paused (chase view: dragging orbits, the wheel zooms)
  await page.keyboard.press(" ");
  await page.keyboard.press("e");
  await page.mouse.move(vw * 0.4, vh * 0.5);
  await page.mouse.wheel(0, -500);
  await page.mouse.down();
  await page.mouse.move(vw * 0.4 + 330, vh * 0.5 + 40, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${out}/drone.png` });
  // and head on
  await page.mouse.down();
  await page.mouse.move(vw * 0.4 + 520, vh * 0.5 + 10, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${out}/drone-front.png` });
  // from behind and a little below, for the warhead's tail (each drag starts on the canvas, left of the panel)
  await page.mouse.move(vw * 0.2, vh * 0.5);
  await page.mouse.down();
  await page.mouse.move(vw * 0.2 + 560, vh * 0.5 - 60, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${out}/drone-back.png` });
  // high over the forest, looking down: trunk tops and fallen logs
  await page.mouse.move(vw * 0.3, vh * 0.2);
  await page.mouse.down();
  await page.mouse.move(vw * 0.3, vh * 0.2 + 320, { steps: 10 });
  await page.mouse.up();
  await page.mouse.wheel(0, 1250);
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${out}/overhead.png` });
  await writeFile(`${out}/summary.json`, JSON.stringify({ summary, samples, errors, logs: logs.slice(-60) }, null, 2));
  await writeFile(`${out}/console.log`, logs.join("\n") + "\n");
  await writeFile(`${out}/trace.json`, JSON.stringify(await page.evaluate(() => window.flybrainFpv.trace())));
  const bench = await page.evaluate(() => window.flybrainFpv.bench());
  if (bench) await writeFile(`${out}/bench.json`, JSON.stringify(bench));
} catch (e) {
  console.warn("could not write to", out, String(e));
}

if (errors.length) {
  console.error(`\n${errors.length} ERROR(S):`);
  console.error(errors.slice(0, 30).join("\n"));
}
await browser.close();
server.close();
process.exit(errors.length ? 1 : 0);
