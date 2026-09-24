// Headless check of the game UI: a destroyed truck stays destroyed, the trucks behind its wreck stop short of it, and
// the Respawn trucks button brings them back and clears the road; the side panel collapses, pins, saves and resets.
// Serves DIST like smoke.mjs.
//
//   docker run --rm --device /dev/dri/renderD128 --ipc=host -v "$PWD/web:/app:ro" -v "$PWD/.cache/ui:/out" \
//     -e DIST=/app/dist flygames-gputest sh -c 'cp /app/scripts/check-ui.mjs /runner/ && node /runner/check-ui.mjs'
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
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
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--use-angle=vulkan"];
// the flygames-gputest image names its Chromium in CHROMIUM (as smoke.mjs uses it)
const browser = await pw.chromium.launch(
  process.env.CHROMIUM
    ? { headless: true, executablePath: process.env.CHROMIUM, args: [...args, "--headless=new", "--no-sandbox"] }
    : { headless: true, channel: "chromium", args },
);
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(e.message));
await page.goto("http://127.0.0.1:4173/?seed=1&view=1");
await page.waitForSelector("#start:not([hidden])", { timeout: 300_000 });
await page.click("#start");
await page.waitForTimeout(3000);

let failures = 0;
const check = (ok, what) => {
  console.log(`${ok ? "ok  " : "FAIL"}  ${what}`);
  if (!ok) failures++;
};
const state = () =>
  page.evaluate(() => {
    const w = window.flybrainFpv.world;
    return {
      time: w.time,
      cars: w.cars.map((c) => (c ? { s: c.s, dir: c.dir, x: c.x, z: c.z } : null)),
      wrecks: w.wrecks.map((k) => ({ s: k.s, x: k.x, z: k.z })),
      L: w.roadLength,
    };
  });

// destroy truck 0 where it is, as a ram would
await page.evaluate(() => window.flybrainFpv.world.destroyCar(0, "ram"));
const t0 = await state();
check(t0.cars[0] === null && t0.wrecks.length === 1, "truck 1 destroyed, one wreck on the road");
await page.waitForTimeout(300); // the HUD updates the button on the next frames
check(await page.evaluate(() => !document.getElementById("trucksBtn").disabled), "the Respawn trucks button is enabled");

// up to 60 brain seconds: the trucks drive on until they reach the wreck; none may come within the stop gap of it
const L = t0.L;
const gapAhead = (c, s) => ((((s - c.s) * c.dir) % L) + L) % L;
let closest = Infinity;
let stopped = 0;
let last = t0;
let gone = true;
for (let k = 0; k < 30; k++) {
  await page.waitForTimeout(2000);
  const st = await state();
  gone &&= st.cars[0] === null;
  // only trucks that started clear of it: one passing the wreck as it blew up stops where it is
  st.cars.forEach((c, i) => {
    if (c && t0.cars[i] && gapAhead(t0.cars[i], t0.wrecks[0].s) >= 7.6) closest = Math.min(closest, gapAhead(c, t0.wrecks[0].s));
  });
  stopped = st.cars.filter((c, i) => c && last.cars[i] && Math.abs(c.s - last.cars[i].s) < 1e-3).length;
  last = st;
  if (stopped > 0 && closest < 9) break;
}
check(gone, `truck 1 stayed gone (${(last.time / 1000).toFixed(0)} s)`);
check(closest >= 7.6 - 0.05, `no truck drove closer than the stop gap to the wreck (closest ${closest.toFixed(2)}, gap 7.6)`);
check(stopped > 0, `${stopped} truck(s) standing still behind it`);
await page.screenshot({ path: `${out}/wreck.png` });

await page.click("#trucksBtn");
await page.waitForTimeout(500);
const t1 = await state();
check(t1.cars[0] !== null, "the button brought truck 1 back");
check(t1.wrecks.length === 0, "and cleared the wreck");
check(await page.evaluate(() => document.getElementById("trucksBtn").disabled), "the button is disabled again");
await page.waitForTimeout(3000);
const t2 = await state();
check(t2.cars.every((c, i) => c && Math.abs(c.s - t1.cars[i].s) > 0.5), "every truck drives again");

// the chase view at half resolution (as after slow frames, renderer.resScale): the whole view, scaled up
await page.evaluate(() => (window.flybrainFpv.renderer.resScale = 0.5));
await page.waitForTimeout(50);
await page.screenshot({ path: `${out}/half-res.png` });

// the side panel: collapse a section; pin one (it moves to the top, and back when unpinned); drag one by its grip
const order = () => page.evaluate(() => [...document.querySelectorAll(".panel-col > .card")].map((c) => c.dataset.section));
const before = await order();
await page.click('[data-section="eyes"] .card-toggle');
await page.click('[data-section="stimulate"] .card-pin');
const panel = await page.evaluate(() => ({
  collapsed: document.querySelector('[data-section="eyes"]').classList.contains("collapsed"),
  eyeShown: getComputedStyle(document.getElementById("eye")).display !== "none",
  saved: JSON.parse(localStorage.getItem("flybrainfpv.panel") ?? "null"),
}));
check(panel.collapsed && !panel.eyeShown, "a section collapses (its canvas hidden)");
check((await order())[0] === "stimulate", `pinning moves a section to the top (${(await order()).join(", ")})`);
check(panel.saved?.collapsed?.includes("eyes") && panel.saved?.pinned === "stimulate", "the layout is saved for the next visit");
await page.screenshot({ path: `${out}/panel.png` });
await page.click('[data-section="stimulate"] .card-pin');
check((await order()).join() === before.join(), "unpinning puts it back where it was");
// drag "What is real" (last) by its grip above the first section
const grip = await page.$('[data-section="real"] .card-grip');
await grip.scrollIntoViewIfNeeded();
const g = await grip.boundingBox();
const first = await (await page.$('[data-section="motor"]')).boundingBox();
await page.mouse.move(g.x + g.width / 2, g.y + g.height / 2);
await page.mouse.down();
for (let k = 1; k <= 20; k++) await page.mouse.move(g.x + g.width / 2, g.y + g.height / 2 + ((first.y + 5 - g.y) * k) / 20);
await page.mouse.up();
check((await order())[0] === "real", `dragging a grip moves a section (${(await order()).join(", ")})`);
await page.click("#resetLayout");
check(
  (await order()).join() === before.join() &&
    (await page.evaluate(() => !document.querySelector(".card.collapsed") && !document.querySelector(".card.pinned"))),
  "Reset layout puts everything back, expanded and unpinned",
);

check(errors.length === 0, `no page errors${errors.length ? `: ${errors.slice(0, 3).join(" | ")}` : ""}`);
await browser.close();
server.close();
console.log(failures ? `${failures} FAILED` : "all passed");
process.exit(failures ? 1 : 0);
