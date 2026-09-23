// The FPV camera's on-screen display: white pixel text with a dark outline, drawn on a 2D canvas over the view.
// The font is this file's own 5x7 bitmap. Everything here is a readout for the player; nothing reaches the brain.
//
// Mostly real numbers from the world and the brain loop. Two are made up for the look: THR (throttle) is a
// display computed from the climb and the bank, since this drone has no throttle, and the link bars show the
// brain's speed (1.00x = full bars).

import { FPV_FOV, VIEW_NAMES, type ViewMode } from "./render";
import type { World } from "./world";

export interface OsdInfo {
  mode: ViewMode;
  /** brain seconds per wall second */
  brainSpeed: number;
  fps: number;
  /** 0..1, decaying after a seizure */
  seizure: number;
  /** 0..1, decaying after a giant fiber escape */
  escape: number;
  paused: boolean;
  /** the target lock (LOCK) is on: the food is in view within 40 deg ahead */
  locked: boolean;
  /** the terminal-dive assist (?dive, MODELED) is steering the pitch */
  diving: boolean;
}

/** 5x7 glyphs, one string of 7 rows of 5 bits each */
const FONT: Record<string, string> = {
  A: "01110 10001 10001 11111 10001 10001 10001",
  B: "11110 10001 10001 11110 10001 10001 11110",
  C: "01110 10001 10000 10000 10000 10001 01110",
  D: "11110 10001 10001 10001 10001 10001 11110",
  E: "11111 10000 10000 11110 10000 10000 11111",
  F: "11111 10000 10000 11110 10000 10000 10000",
  G: "01110 10001 10000 10111 10001 10001 01111",
  H: "10001 10001 10001 11111 10001 10001 10001",
  I: "01110 00100 00100 00100 00100 00100 01110",
  J: "00111 00010 00010 00010 00010 10010 01100",
  K: "10001 10010 10100 11000 10100 10010 10001",
  L: "10000 10000 10000 10000 10000 10000 11111",
  M: "10001 11011 10101 10101 10001 10001 10001",
  N: "10001 10001 11001 10101 10011 10001 10001",
  O: "01110 10001 10001 10001 10001 10001 01110",
  P: "11110 10001 10001 11110 10000 10000 10000",
  Q: "01110 10001 10001 10001 10101 10010 01101",
  R: "11110 10001 10001 11110 10100 10010 10001",
  S: "01111 10000 10000 01110 00001 00001 11110",
  T: "11111 00100 00100 00100 00100 00100 00100",
  U: "10001 10001 10001 10001 10001 10001 01110",
  V: "10001 10001 10001 10001 10001 01010 00100",
  W: "10001 10001 10001 10101 10101 10101 01010",
  X: "10001 10001 01010 00100 01010 10001 10001",
  Y: "10001 10001 01010 00100 00100 00100 00100",
  Z: "11111 00001 00010 00100 01000 10000 11111",
  "0": "01110 10001 10011 10101 11001 10001 01110",
  "1": "00100 01100 00100 00100 00100 00100 01110",
  "2": "01110 10001 00001 00010 00100 01000 11111",
  "3": "11111 00010 00100 00010 00001 10001 01110",
  "4": "00010 00110 01010 10010 11111 00010 00010",
  "5": "11111 10000 11110 00001 00001 10001 01110",
  "6": "00110 01000 10000 11110 10001 10001 01110",
  "7": "11111 00001 00010 00100 01000 01000 01000",
  "8": "01110 10001 10001 01110 10001 10001 01110",
  "9": "01110 10001 10001 01111 00001 00010 01100",
  " ": "00000 00000 00000 00000 00000 00000 00000",
  ".": "00000 00000 00000 00000 00000 01100 01100",
  ":": "00000 01100 01100 00000 01100 01100 00000",
  "-": "00000 00000 00000 11111 00000 00000 00000",
  "+": "00000 00100 00100 11111 00100 00100 00000",
  "%": "11000 11001 00010 00100 01000 10011 00011",
  "/": "00000 00001 00010 00100 01000 10000 00000",
  "°": "01100 10010 10010 01100 00000 00000 00000",
  "!": "00100 00100 00100 00100 00100 00000 00100",
  "<": "00010 00100 01000 10000 01000 00100 00010",
  ">": "01000 00100 00010 00001 00010 00100 01000",
  "[": "01110 01000 01000 01000 01000 01000 01110",
  "]": "01110 00010 00010 00010 00010 00010 01110",
  "=": "00000 00000 11111 00000 11111 00000 00000",
  // a grenade, an up arrow and a clock, as pictograms
  "@": "00110 00100 01110 11111 11111 11111 01110",
  "^": "00100 01110 10101 00100 00100 00100 00100",
  "~": "01110 10101 10101 10111 10001 10001 01110",
};

const CELL_W = 6;
const CELL_H = 9;

export class Osd {
  private ctx: CanvasRenderingContext2D;
  /** one sprite per glyph at the current scale, outline baked in */
  private glyphs = new Map<string, HTMLCanvasElement>();
  private scale = 0;
  /** brain time the current drone took off */
  private launched = 0;
  private wasLost = false;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  private glyph(ch: string): HTMLCanvasElement | undefined {
    const g = this.glyphs.get(ch);
    if (g) return g;
    const rows = FONT[ch]?.split(" ");
    if (!rows) return undefined;
    const s = this.scale;
    const c = document.createElement("canvas");
    c.width = 5 * s + 2 * s;
    c.height = 7 * s + 2 * s;
    const x = c.getContext("2d")!;
    for (const [color, grow] of [["#000", s], ["#fff", 0]] as const) {
      x.fillStyle = color;
      for (let r = 0; r < 7; r++) {
        for (let k = 0; k < 5; k++) {
          if (rows[r][k] === "1") x.fillRect(s + k * s - grow * 0.5, s + r * s - grow * 0.5, s + grow, s + grow);
        }
      }
    }
    this.glyphs.set(ch, c);
    return c;
  }

  /** text with its top-left at cell (x, y) in pixels; align 0 left, 0.5 centre, 1 right */
  private text(str: string, x: number, y: number, align = 0) {
    const s = this.scale;
    const w = str.length * CELL_W * s;
    let px = Math.round(x - w * align);
    for (const ch of str.toUpperCase()) {
      const g = this.glyph(ch);
      if (g) this.ctx.drawImage(g, px - s, Math.round(y) - s);
      px += CELL_W * s;
    }
  }

  /** an outlined white line */
  private line(x0: number, y0: number, x1: number, y1: number, width: number) {
    const c = this.ctx;
    c.lineCap = "square";
    c.strokeStyle = "#000";
    c.lineWidth = width + 2 * this.scale * 0.67;
    c.beginPath();
    c.moveTo(x0, y0);
    c.lineTo(x1, y1);
    c.stroke();
    c.strokeStyle = "#fff";
    c.lineWidth = width;
    c.stroke();
  }

  private box(x: number, y: number, w: number, h: number, fill: boolean) {
    const c = this.ctx;
    const o = Math.max(1, Math.round(this.scale * 0.67));
    c.fillStyle = "#000";
    c.fillRect(x - o, y - o, w + 2 * o, h + 2 * o);
    c.fillStyle = fill ? "#fff" : "#000";
    c.fillRect(x, y, w, h);
    if (!fill) {
      c.strokeStyle = "#fff";
      c.lineWidth = Math.max(1, o * 0.75);
      c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    }
  }

  draw(w: World, info: OsdInfo) {
    const cv = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.round(cv.clientWidth * dpr));
    const ch = Math.max(1, Math.round(cv.clientHeight * dpr));
    if (cv.width !== cw || cv.height !== ch) {
      cv.width = cw;
      cv.height = ch;
    }
    // pixel size of the font: from the height, and from the width on a narrow (portrait phone) view
    const s = Math.max(2, Math.round(Math.min(ch, cw * 0.6) / 330));
    if (s !== this.scale) {
      this.scale = s;
      this.glyphs.clear();
    }
    const c = this.ctx;
    c.clearRect(0, 0, cw, ch);
    if (w.droneLost) this.wasLost = true;
    else if (this.wasLost) {
      this.wasLost = false;
      this.launched = w.time;
    }
    if (info.mode !== 3) return;

    const lh = CELL_H * s;
    const margin = 3 * CELL_W * s;
    const top = 2 * lh;
    const blink = performance.now() % 700 < 450;
    const DEG = 180 / Math.PI;

    // top left: link bars (the brain's speed) and the brain's numbers
    const bars = Math.max(0, Math.min(5, Math.round(info.brainSpeed * 5)));
    for (let i = 0; i < 5; i++) {
      const bh = (i + 1) * 1.4 * s;
      this.box(margin + i * 2.4 * s, top + 7 * s - bh, 1.6 * s, bh, i < bars);
    }
    this.text(`BRAIN ${info.brainSpeed.toFixed(2)}X`, margin + 14 * s, top);
    this.text(`${Math.round(info.fps)} FPS`, margin, top + lh);

    // top centre: the mode, and who is flying
    this.text(VIEW_NAMES[info.mode], cw / 2, top, 0.5);
    this.text("PILOT: FLY CNS", cw / 2, top + lh, 0.5);

    // top right: the lens and the clock
    const t = Math.max(0, (w.time - this.launched) / 1000);
    const mm = String(Math.floor(t / 60)).padStart(2, "0");
    const ss = String(Math.floor(t % 60)).padStart(2, "0");
    this.text(`FOV ${FPV_FOV}°`, cw - margin, top, 1);
    this.text(`~ ${mm}:${ss}`, cw - margin, top + lh, 1);

    // either side of centre: speed, altitude, tilt, throttle
    const speed = Math.hypot(w.vx, w.vy, w.vz) * 3.6;
    const thr = Math.max(0, Math.min(100, 42 + w.vy * 7 + Math.abs(w.bank) * 25 + w.hopUp * 30));
    const midY = ch / 2 - lh * 1.5;
    const sideX = cw * 0.2;
    this.text(`${Math.round(w.droneLost ? 0 : speed)} KM/H`, sideX, midY);
    this.text(`TILT ${(w.pitch * DEG).toFixed(0)}°`, sideX, midY + 2 * lh);
    this.text(`ALT ${w.y.toFixed(1)} M`, cw - sideX, midY, 1);
    this.text(`THR ${w.droneLost ? 0 : Math.round(thr)}%`, cw - sideX, midY + 2 * lh, 1);

    // centre: a crosshair, and a horizon line that tilts with the bank and moves with the gaze
    const cx = cw / 2;
    const cy = ch / 2;
    const arm = 5 * s;
    const lw = Math.max(1, Math.round(s * 0.67));
    this.line(cx - arm, cy, cx - 2 * s, cy, lw);
    this.line(cx + 2 * s, cy, cx + arm, cy, lw);
    this.line(cx, cy - arm, cx, cy - 2 * s, lw);
    this.line(cx, cy + 2 * s, cx, cy + arm, lw);
    if (!w.droneLost) {
      const pxPerRad = cw / 2 / ((FPV_FOV / 2) * (Math.PI / 180));
      const off = w.pitch * pxPerRad;
      const cb = Math.cos(-w.bank);
      const sb = Math.sin(-w.bank);
      const hx = cx - sb * off;
      const hy = cy + cb * off;
      for (const side of [-1, 1]) {
        const a = side * 10 * s;
        const b = side * 22 * s;
        this.line(hx + cb * a, hy + sb * a, hx + cb * b, hy + sb * b, lw);
      }
    }

    // bottom: grenades and the tally, and the brain's warnings
    const bottom = ch - 150 * dpr;
    const left = w.droneLost ? 0 : w.grenadesLeft;
    if (w.grenadesPerDrone > 0)
      this.text(`${"@".repeat(left)}${" ".repeat(Math.max(0, w.grenadesPerDrone - left))} GREN ${left}`, margin, bottom);
    this.text(`KILLS ${w.stats.carsDestroyed}  LOST ${w.stats.dronesLost}`, cw - margin, bottom, 1);
    // the convoy: trucks on the road, and the designated one's number, range and bearing
    const alive = w.cars.filter((c) => c).length;
    this.text(`TRUCKS ${alive}/${w.cars.length}`, cw - margin, bottom - lh * 1.3, 1);
    const f = w.food;
    if (f) {
      const l = w.droneLost ? null : w.locate(f);
      const name = f.kind === "car" ? `TRUCK ${f.index + 1}` : `BALLOON ${f.index + 1}`;
      const where = l ? ` ${Math.round(l.dist)} M ${l.az >= 0 ? "R" : "L"}${Math.abs(Math.round(l.az))}°` : " --";
      this.text(`TGT ${name}${where}`, cx, bottom, 0.5);
    }

    const warn: string[] = [];
    if (w.droneLost) warn.push("NO SIGNAL", `NEW DRONE ${Math.max(0, (w.respawnAt - w.time) / 1000).toFixed(1)}S`);
    if (info.seizure > 0.05) warn.push("SEIZURE");
    if (info.escape > 0.05) warn.push("GIANT FIBER");
    if (info.locked && !w.droneLost) warn.push("TARGET LOCKED");
    if (info.diving && !w.droneLost) warn.push("DIVE ASSIST");
    if (info.paused) warn.push("PAUSED");
    warn.forEach((msg, i) => {
      if (blink || msg === "PAUSED" || msg.startsWith("NEW")) this.text(msg, cx, cy + 9 * s + (i + 1) * lh * 1.3, 0.5);
    });
  }
}
