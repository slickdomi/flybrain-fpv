// Headless phone check: the page in Chromium's phone emulation (390 x 844 CSS px at 3x, touch), on the host GPU.
// A quick tap on a truck designates it (and paints nothing), a finger dragged over the view paints a phantom target,
// two fingers pinch-zoom, nothing overflows the screen, and it still runs in landscape. Screenshots go to OUT.
// This checks the layout and the touch handling; it says nothing about a phone GPU's speed or memory.
//
//   docker run --rm --device /dev/dri/renderD128 --ipc=host -v "$PWD/web:/app:ro" -v "$PWD/.cache/mobile:/out" \
//     -e DIST=/app/dist flygames-gputest sh -c 'cp /app/scripts/check-mobile.mjs /runner/ && node /runner/check-mobile.mjs'
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
    // served as-is, like GitHub Pages: the .gz shards arrive still compressed and the loader inflates them
    res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" });
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
const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true });
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(e.message));

let failures = 0;
const check = (ok, what) => {
  console.log(`${ok ? "ok  " : "FAIL"}  ${what}`);
  if (!ok) failures++;
};
const touch = (type, points) => cdp.send("Input.dispatchTouchEvent", { type, touchPoints: points.map(([x, y], id) => ({ x, y, id })) });

await page.goto("http://127.0.0.1:4173/?seed=1");
await page.waitForSelector("#start:not([hidden]), #loadError:not([hidden])", { timeout: 300_000 });
check(!(await page.$("#loadError:not([hidden])")), "loads (the .gz shards served without Content-Encoding, as on Pages)");
await page.tap("#start");
await page.waitForTimeout(4000);

console.log("      pointer: coarse =", await page.evaluate(() => matchMedia("(pointer: coarse)").matches));
const layout = await page.evaluate(() => ({
  scrollW: document.documentElement.scrollWidth,
  innerW: innerWidth,
  stageH: document.getElementById("stage").getBoundingClientRect().height,
  innerH: innerHeight,
  buttons: [...document.querySelectorAll("#toolbar button")]
    .filter((b) => getComputedStyle(b).display !== "none")
    .map((b) => b.getBoundingClientRect())
    .map((r) => [r.left, r.right, r.top, r.bottom]),
}));
check(layout.scrollW <= layout.innerW, `no sideways scroll (page ${layout.scrollW} px wide on a ${layout.innerW} px screen)`);
check(layout.buttons.every(([l, r]) => l >= 0 && r <= layout.innerW), `every toolbar button on screen (${layout.buttons.length})`);
check(layout.stageH > 0.5 * layout.innerH, `the view takes ${Math.round((100 * layout.stageH) / layout.innerH)}% of the height`);
await page.screenshot({ path: `${out}/portrait.png` });

// a truck on screen: scan the view for a point whose ray picks one
const rect = await page.evaluate(() => {
  const r = document.getElementById("view").getBoundingClientRect();
  return { x: r.left, y: r.top, w: r.width, h: r.height };
});
const spot = await page.evaluate(() => {
  const { world, renderer } = window.flybrainFpv;
  for (let j = 2; j < 38; j++)
    for (let i = 2; i < 38; i++) {
      const { ro, r } = renderer.pickRay(i / 40, j / 40, world);
      const t = world.pick(ro, r);
      if (t && t.kind === "car") return { u: i / 40, v: j / 40, index: t.index };
    }
  return null;
});
check(!!spot, "a truck in view to tap");
if (spot) {
  await page.evaluate(() => window.flybrainFpv.world.food = null);
  await page.touchscreen.tap(rect.x + spot.u * rect.w, rect.y + spot.v * rect.h);
  await page.waitForTimeout(300);
  const after = await page.evaluate(() => ({ food: window.flybrainFpv.world.food, paint: window.flybrainFpv.renderer.paint }));
  check(after.food?.kind === "car" && after.food.index === spot.index, `a tap designated truck ${spot.index + 1}`);
  check(!after.paint, "and painted nothing");
}

// a finger dragged across the sky paints; lifting it stops
const sx = rect.x + rect.w * 0.3;
const sy = rect.y + rect.h * 0.3;
await touch("touchStart", [[sx, sy]]);
for (let k = 1; k <= 8; k++) {
  await page.waitForTimeout(50);
  await touch("touchMove", [[sx + 12 * k, sy + 4 * k]]);
}
const during = await page.evaluate(() => ({ paint: window.flybrainFpv.renderer.paint, info: document.getElementById("paintInfo").textContent }));
check(!!during.paint, `a finger drag paints (${during.info || "no cells"})`);
await page.screenshot({ path: `${out}/painting.png` });
await touch("touchEnd", []);
await page.waitForTimeout(200);
check(!(await page.evaluate(() => window.flybrainFpv.renderer.paint)), "lifting the finger stops the painting");

// a finger held still also paints, once it has stayed a moment
await touch("touchStart", [[sx, sy]]);
await page.waitForTimeout(400);
check(!!(await page.evaluate(() => window.flybrainFpv.renderer.paint)), "a finger held still paints too");
await touch("touchEnd", []);
await page.waitForTimeout(200);

// pinch out in the fly-eye view: it zooms in (a narrower field)
await page.evaluate(() => {
  document.getElementById("viewBtn").click(); // FPV -> fly eye
});
await page.waitForTimeout(300);
const fov0 = await page.evaluate(() => window.flybrainFpv.renderer.fovAz);
const cx = rect.x + rect.w / 2;
const cy = rect.y + rect.h / 2;
await touch("touchStart", [[cx - 40, cy], [cx + 40, cy]]);
for (let k = 1; k <= 8; k++) {
  await page.waitForTimeout(30);
  await touch("touchMove", [[cx - 40 - 12 * k, cy], [cx + 40 + 12 * k, cy]]);
}
await touch("touchEnd", []);
const fov1 = await page.evaluate(() => window.flybrainFpv.renderer.fovAz);
check(fov1 < fov0 * 0.8, `two fingers pinch-zoom (field ${fov0.toFixed(0)} -> ${fov1.toFixed(0)} deg)`);
check(!(await page.evaluate(() => window.flybrainFpv.renderer.paint)), "and the pinch painted nothing");

// held stimulation buttons work with a finger
const btn = await page.$(".stim");
// in portrait the panel is below the view: scroll the button into sight, as a finger would
await btn.scrollIntoViewIfNeeded();
const box = await btn.boundingBox();
await touch("touchStart", [[box.x + box.width / 2, box.y + box.height / 2]]);
await page.waitForTimeout(300);
check(await page.evaluate(() => document.querySelector(".stim").classList.contains("on")), "holding a stimulus button with a finger drives it");
await touch("touchEnd", []);
await page.waitForTimeout(200);
check(!(await page.evaluate(() => document.querySelector(".stim").classList.contains("on"))), "and letting go releases it");

// the panel's sections collapse and pin with a tap (the pinned one moves to the top), and a finger on a grip drags one
const order = () => page.evaluate(() => [...document.querySelectorAll(".panel-col > .card")].map((c) => c.dataset.section));
await page.tap('[data-section="eyes"] .card-toggle');
await page.tap('[data-section="target"] .card-pin');
check(
  (await page.evaluate(() => document.querySelector('[data-section="eyes"]').classList.contains("collapsed"))) && (await order())[0] === "target",
  `a tap collapses a section, and one on the pin moves it to the top (${(await order()).join(", ")})`,
);
await page.tap('[data-section="target"] .card-pin');
const grip = await page.$('[data-section="stimulate"] .card-grip');
await grip.scrollIntoViewIfNeeded();
const g = await grip.boundingBox();
const top = await (await page.$('[data-section="motor"]')).boundingBox();
// the motor card is above: drag up past it (the page scrolls as the finger nears the top edge)
const gx = g.x + g.width / 2;
const gy = g.y + g.height / 2;
await touch("touchStart", [[gx, gy]]);
for (let k = 1; k <= 20; k++) {
  await page.waitForTimeout(20);
  await touch("touchMove", [[gx, gy + ((Math.max(top.y + 5, 60) - gy) * k) / 20]]);
}
await page.waitForTimeout(600);
await touch("touchEnd", []);
const moved = await order();
check(moved.indexOf("stimulate") < moved.indexOf("target"), `a finger on a grip drags a section (${moved.join(", ")})`);
await page.tap("#resetLayout");

// landscape
await page.evaluate(() => document.getElementById("viewBtn").click()); // back round to FPV takes three more
await page.evaluate(() => document.getElementById("viewBtn").click());
await page.evaluate(() => document.getElementById("viewBtn").click());
await page.setViewportSize({ width: 844, height: 390 });
await page.waitForTimeout(1500);
const land = await page.evaluate(() => ({ scrollW: document.documentElement.scrollWidth, innerW: innerWidth, mode: window.flybrainFpv.renderer.mode }));
check(land.scrollW <= land.innerW, `landscape: no sideways scroll (${land.scrollW} / ${land.innerW})`);
await page.screenshot({ path: `${out}/landscape.png` });

const s = await page.evaluate(() => ({ ...window.flybrainFpv.summary(), res: window.flybrainFpv.renderer.resScale }));
console.log(`      brain ${s.brainSpeed}x, ${s.fps} fps, render scale ${s.res.toFixed(2)} (this GPU, not a phone's)`);
check(errors.length === 0, `no page errors${errors.length ? `: ${errors.slice(0, 3).join(" | ")}` : ""}`);
await browser.close();
server.close();
console.log(failures ? `${failures} FAILED` : "all passed");
process.exit(failures ? 1 : 0);
