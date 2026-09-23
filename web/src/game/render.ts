// Owns the scene uniform (which the fly's eye compute pass reads straight off the GPU) and draws the player's view.
//
// Two passes: view.wgsl's `fs` ray-traces the world into an offscreen HDR target, then its `post` copies that to the
// canvas (fly eye, chase, ommatidia) or grades it into gritty FPV footage (FPV cam). Everything the View uniform
// carries (grenades, explosions, wrecks, the FPV camera and its shake) is for the player only: the fly's eye reads
// the Scene uniform alone.

import sceneSrc from "../shaders/scene.wgsl?raw";
import viewSrc from "../shaders/view.wgsl?raw";
import { CAR, GAME, MARK, WORLD } from "../config";
import { MAX_CARS, type World } from "./world";

/** must match MAX_TREES / MAX_BALLOONS / MAX_ROAD in scene.wgsl (MAX_CARS lives in world.ts) */
const MAX_TREES = 32;
const MAX_BALLOONS = 8;
const MAX_ROAD = 32;
/** header, trees (trunk, canopy), balloons, the trucks, their count and dimensions, the road's points, road info */
export const SCENE_BYTES = 64 + 16 * (2 * MAX_TREES + MAX_BALLOONS + MAX_CARS) + 32 + 8 * MAX_ROAD + 16;

/** must match MAX_GRENADES / MAX_EXPLOSIONS / MAX_WRECKS in view.wgsl */
const MAX_GRENADES = 8;
const MAX_EXPLOSIONS = 8;
const MAX_WRECKS = 8;
/** the View struct in view.wgsl, in floats */
const VIEW_FLOATS = 48 + 8 * MAX_GRENADES + 8 * MAX_EXPLOSIONS + 4 * MAX_WRECKS;

/** What the fly is shown that isn't geometry: contrasts, and whether the food marker is on. */
export interface Look {
  groundContrast: number;
  treeContrast: number;
  mark: boolean;
}

/** 0 the fly's eye, 1 chase camera, 2 ommatidia (approximate), 3 the drone's FPV camera (the default) */
export type ViewMode = 0 | 1 | 2 | 3;
export const VIEW_NAMES = ["fly eye", "chase", "ommatidia", "FPV cam"];
export const VIEW_COUNT = 4;

/** How far ahead of the drone's centre the fly's head (the camera pod) sits, level with it. */
const EYE_FORWARD = 0.36;
/** the FPV camera's horizontal field (deg, equidistant fisheye), and the resolution it renders at before grading */
export const FPV_FOV = 130;
const FPV_SCALE = 0.75;
/** FPV exposure into the grade: high enough that the overcast sky blows out */
const FPV_EXPOSURE = 1.5;

export interface Paint {
  /** degrees in the fly's field, + = right / up */
  az: number;
  el: number;
  radius: number;
}

interface Cam {
  cam: V3;
  right: V3;
  up: V3;
  fwd: V3;
  tanX: number;
  tanY: number;
}

export class Renderer {
  readonly sceneBuffer: GPUBuffer;
  mode: ViewMode = 3;
  /** half-fields of the eye view, degrees */
  fovAz = 120;
  seizure = 0;
  escape = 0;
  paint: Paint | null = null;
  /** chase camera orbit, radians */
  chaseYaw = 0;
  chasePitch = 0.25;
  chaseDist = 5;
  /**
   * Fraction of the canvas's pixels the view is traced at (then scaled up), set by main.ts's frame loop: it drops
   * when frames run late (a phone), down to LOOP.minRenderScale, before the brain is slowed.
   */
  resScale = 1;

  private sceneData = new ArrayBuffer(SCENE_BYTES);
  /** the chase and FPV cameras as last drawn, for picking */
  private chase: Cam | null = null;
  private fpv: (Cam & { aspect: number; heading: number; pitch: number }) | null = null;
  private viewBuffer: GPUBuffer;
  private pipe: GPURenderPipeline;
  private postPipe: GPURenderPipeline;
  private group: GPUBindGroup;
  private postGroup: GPUBindGroup;
  private frameGroup: GPUBindGroup | null = null;
  private frame: GPUTexture | null = null;
  private sampler: GPUSampler;
  private context: GPUCanvasContext;

  constructor(
    private device: GPUDevice,
    private canvas: HTMLCanvasElement,
    format: GPUTextureFormat,
  ) {
    const U = GPUBufferUsage;
    this.sceneBuffer = device.createBuffer({ size: SCENE_BYTES, usage: U.UNIFORM | U.COPY_DST, label: "scene" });
    this.viewBuffer = device.createBuffer({ size: VIEW_FLOATS * 4, usage: U.UNIFORM | U.COPY_DST, label: "view" });
    this.context = canvas.getContext("webgpu") as GPUCanvasContext;
    this.context.configure({ device, format, alphaMode: "opaque" });

    const module = device.createShaderModule({ code: sceneSrc + viewSrc, label: "view" });
    // A WGSL error otherwise shows up only as an invalid pipeline, and because the brain shares the frame's
    // command encoder, that silently stops the brain too (fly-addiction). Print the compiler's message.
    module.getCompilationInfo?.().then((info) => {
      for (const m of info.messages) {
        const where = `scene+view.wgsl:${m.lineNum}:${m.linePos}`;
        if (m.type === "error") console.error(`WGSL error ${where}: ${m.message}`);
        else console.warn(`WGSL ${m.type} ${where}: ${m.message}`);
      }
    });
    this.pipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
      primitive: { topology: "triangle-list" },
    });
    this.postPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "post", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.group = device.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneBuffer } },
        { binding: 1, resource: { buffer: this.viewBuffer } },
      ],
    });
    this.postGroup = device.createBindGroup({
      layout: this.postPipe.getBindGroupLayout(0),
      entries: [{ binding: 1, resource: { buffer: this.viewBuffer } }],
    });
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  }

  /** The fly's head: at the front of the drone, on a gimbal that keeps it level in roll. */
  eyePosition(w: World): [number, number, number] {
    return [w.x + Math.sin(w.heading) * EYE_FORWARD, w.y, w.z - Math.cos(w.heading) * EYE_FORWARD];
  }

  /** Uploads the world for this frame: what the brain's eye pass samples. */
  writeScene(w: World, look: Look) {
    const f = new Float32Array(this.sceneData);
    const eye = this.eyePosition(w);
    const ground = WORLD.sky * (1 - look.groundContrast);
    f.set(eye, 0);
    f[3] = w.heading;
    f[4] = w.pitch;
    f[5] = w.time / 1000;
    f[6] = WORLD.sky;
    f[7] = ground;
    f[8] = WORLD.sky * (1 - look.treeContrast);
    f[9] = WORLD.balloonLum;
    f[10] = Math.min(MAX_TREES, w.trees.length);
    f[11] = Math.min(MAX_BALLOONS, w.balloons.length);
    f[12] = WORLD.size;
    f[13] = ground * (1 - WORLD.roadContrast);
    const food = w.food;
    f[14] = food ? (food.kind === "car" ? 2 : 1) : 0;
    f[15] = food ? food.index : 0;
    let o = 16;
    for (let i = 0; i < MAX_TREES; i++) {
      const t = w.trees[i];
      f.set(t ? [t.x, t.z, t.trunkR, t.trunkTop] : [0, 0, 0, 0], o + 4 * i);
      f.set(t ? [t.canopyY, t.canopyR, 0, 0] : [0, 0, 0, 0], o + 4 * (MAX_TREES + i));
    }
    o += 8 * MAX_TREES;
    for (let i = 0; i < MAX_BALLOONS; i++) {
      const b = w.balloons[i];
      f.set(b ? [b.x, b.y, b.z, b.r] : [0, 0, 0, 0], o + 4 * i);
    }
    o += 4 * MAX_BALLOONS;
    for (let i = 0; i < MAX_CARS; i++) {
      const car = w.cars[i];
      // w: present (> 0) to the fly; 1 + the wheels' turn as a fraction of a revolution, for the view
      f.set(car ? [car.x, car.z, car.heading, 1 + (((car.wheel / (2 * Math.PI)) % 1) + 1) % 1] : [0, 0, 0, 0], o + 4 * i);
    }
    o += 4 * MAX_CARS;
    f.set([Math.min(MAX_CARS, w.cars.length), 0, 0, 0], o);
    f.set([CAR.halfLength, CAR.halfWidth, CAR.height, CAR.lum], o + 4);
    o += 8;
    const pts = w.road.points;
    for (let i = 0; i < MAX_ROAD; i++) f.set(pts[i] ?? [0, 0], o + 2 * i);
    o += 2 * MAX_ROAD;
    f.set([Math.min(MAX_ROAD, pts.length), w.road.width, MARK.markLum, look.mark ? 1 : 0], o);
    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData);
  }

  /** Screen position (0..1) -> the fly-frame azimuth / elevation it shows, degrees (eye views and FPV). */
  screenToAzEl(u: number, v: number): { az: number; el: number } {
    if (this.mode === 3 && this.fpv) {
      const f = this.fpv;
      return worldToFly(this.fisheyeDir(u, v), f.heading, f.pitch);
    }
    const fovEl = this.fovEl();
    return { az: (u * 2 - 1) * this.fovAz, el: (1 - v * 2) * fovEl };
  }

  /** The world ray through a screen position (0..1) in the current view, for picking. */
  pickRay(u: number, v: number, w: World): { ro: V3; r: V3 } {
    if (this.mode === 1 && this.chase) {
      const c = this.chase;
      const x = (u * 2 - 1) * c.tanX;
      const y = (1 - v * 2) * c.tanY;
      return { ro: c.cam, r: norm([c.fwd[0] + c.right[0] * x + c.up[0] * y, c.fwd[1] + c.right[1] * x + c.up[1] * y, c.fwd[2] + c.right[2] * x + c.up[2] * y]) };
    }
    if (this.mode === 3 && this.fpv) return { ro: this.fpv.cam, r: this.fisheyeDir(u, v) };
    const { az, el } = this.screenToAzEl(u, v);
    return { ro: this.eyePosition(w), r: flyToWorld(az, el, w.heading, w.pitch) };
  }

  /** view.wgsl's FPV lens on the CPU: an equidistant fisheye, FPV_FOV across the width. */
  private fisheyeDir(u: number, v: number): V3 {
    const f = this.fpv!;
    const px = (u * 2 - 1) * f.aspect;
    const py = 1 - v * 2;
    const len = Math.hypot(px, py);
    const th = (len / f.aspect) * ((FPV_FOV / 2) * (Math.PI / 180));
    const s = len > 1e-6 ? Math.sin(th) / len : 0;
    const c = Math.cos(th);
    return norm([
      f.fwd[0] * c + (f.right[0] * px + f.up[0] * py) * s,
      f.fwd[1] * c + (f.right[1] * px + f.up[1] * py) * s,
      f.fwd[2] * c + (f.right[2] * px + f.up[2] * py) * s,
    ]);
  }

  private fovEl() {
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    return Math.min(85, this.fovAz / aspect);
  }

  /** the offscreen target the world is traced into, at `w` x `h` */
  private target(w: number, h: number) {
    if (this.frame && this.frame.width === w && this.frame.height === h) return this.frame;
    this.frame?.destroy();
    this.frame = this.device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "view frame",
    });
    this.frameGroup = this.device.createBindGroup({
      layout: this.postPipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: this.frame.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    return this.frame;
  }

  render(encoder: GPUCommandEncoder, w: World, timeS: number) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const ch = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }
    const scale = (this.mode === 3 ? FPV_SCALE : 1) * this.resScale;
    const frame = this.target(Math.max(1, Math.round(cw * scale)), Math.max(1, Math.round(ch * scale)));
    const DEG = Math.PI / 180;
    const aspect = cw / ch;

    // chase camera: behind the drone along its heading, orbit offsets from dragging
    const yaw = w.heading + this.chaseYaw;
    const cp = Math.cos(this.chasePitch);
    const cam: V3 = [
      w.x - Math.sin(yaw) * cp * this.chaseDist,
      Math.max(0.2, w.y + Math.sin(this.chasePitch) * this.chaseDist),
      w.z + Math.cos(yaw) * cp * this.chaseDist,
    ];
    const fwd = norm([w.x - cam[0], w.y - cam[1], w.z - cam[2]]);
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    const tanY = Math.tan((60 * DEG) / 2);
    this.chase = { cam, right, up, fwd, tanX: tanY * aspect, tanY };

    // explosions, newest first, and how hard the nearest shakes the FPV camera
    const eye = this.eyePosition(w);
    const booms = w.explosions
      .map((e) => ({ e, age: (w.time - e.t0) / 1000 }))
      .filter((b) => b.age >= 0 && b.age < GAME.explosionMs / 1000 + 0.1)
      .sort((a, b) => a.age - b.age)
      .slice(0, MAX_EXPLOSIONS);
    let shake = 0;
    for (const { e, age } of booms) {
      const d = Math.hypot(wrap(e.x - eye[0]), e.y - eye[1], wrap(e.z - eye[2]));
      shake = Math.max(shake, Math.exp(-age * 4) * Math.min(1, (2.5 * e.radius) / Math.max(d, 0.1)));
    }
    const lostAge = w.droneLost ? (w.time - (w.respawnAt - GAME.respawnMs)) / 1000 : -1;

    // the FPV camera: at the fly's head, looking along the gaze, rolling with the drone's bank (view only), with a
    // little prop vibration and a kick from nearby blasts
    const jit = (a: number, b: number, c: number) => Math.sin(timeS * a) * 0.5 + Math.sin(timeS * b + 1.3) * 0.3 + Math.sin(timeS * c + 2.1) * 0.2;
    const shakeAmp = 0.0025 + 0.05 * shake;
    const roll = w.bank + jit(71, 53, 97) * shakeAmp;
    const fpvPitch = w.pitch + jit(83, 61, 109) * shakeAmp;
    const fpvHeading = w.heading + jit(67, 89, 113) * shakeAmp * 0.6;
    const gf = flyToWorld(0, 0, fpvHeading, fpvPitch);
    const gr = flyToWorld(90, 0, fpvHeading, fpvPitch);
    const gu = cross(gr, gf);
    const cr = Math.cos(roll);
    const sr = Math.sin(roll);
    const fr: V3 = [gr[0] * cr - gu[0] * sr, gr[1] * cr - gu[1] * sr, gr[2] * cr - gu[2] * sr];
    const fu: V3 = [gu[0] * cr + gr[0] * sr, gu[1] * cr + gr[1] * sr, gu[2] * cr + gr[2] * sr];
    this.fpv = { cam: eye, right: fr, up: fu, fwd: gf, tanX: aspect, tanY: 1, aspect, heading: w.heading, pitch: w.pitch };

    const c = this.mode === 3 ? this.fpv : this.chase;
    const v = new Float32Array(VIEW_FLOATS);
    v.set([this.mode, this.fovAz * DEG, this.fovEl() * DEG, timeS], 0);
    v.set([...c.cam, c.tanX], 4);
    v.set([...c.right, c.tanY], 8);
    v.set([...c.up, this.seizure], 12);
    v.set([...c.fwd, this.mode === 1 ? w.bank : 0], 16);
    v.set([w.x, w.y, w.z, w.heading], 20);
    const p = this.paint;
    v.set(p ? [p.az * DEG, p.el * DEG, p.radius * DEG, 1] : [0, 0, 0, 0], 24);
    // the drone pitches its body a little with the gaze; cosmetic
    v.set([w.pitch * 0.5, timeS * 40, this.escape, w.droneLost ? 0 : w.grenadesLeft], 28);
    const food = w.food;
    v.set(food ? [food.kind === "car" ? 2 : 1, food.index, 0, 0] : [0, 0, 0, 0], 32);
    // radians per offscreen pixel, for fading out detail finer than a pixel
    const pixAngle = this.mode === 3 ? (FPV_FOV * DEG) / frame.width : this.mode === 1 ? (2 * tanY) / frame.height : (2 * this.fovEl() * DEG) / frame.height;
    const grenades = w.grenades.slice(0, MAX_GRENADES);
    const wrecks = w.wrecks.slice(-MAX_WRECKS);
    v.set([grenades.length, booms.length, pixAngle, wrecks.length], 36);
    v.set([(FPV_FOV / 2) * DEG, lostAge, w.droneLost ? 0 : 1, scale], 40);
    v.set([Math.min(1, shake * 1.5), FPV_EXPOSURE, 0, 0], 44);
    let o = 48;
    grenades.forEach((g, i) => v.set([g.x, g.y, g.z, 0, g.vx, g.vy, g.vz, 0], o + 8 * i));
    o += 8 * MAX_GRENADES;
    booms.forEach(({ e, age }, i) => v.set([e.x, e.y, e.z, e.radius, age, (e.t0 * 0.001) % 97, 0, 0], o + 8 * i));
    o += 8 * MAX_EXPLOSIONS;
    wrecks.forEach((k, i) => v.set([k.x, k.z, k.heading, (w.time - k.t0) / 1000], o + 4 * i));
    this.device.queue.writeBuffer(this.viewBuffer, 0, v);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: frame.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.group);
    pass.draw(3);
    pass.end();

    const post = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
    });
    post.setPipeline(this.postPipe);
    post.setBindGroup(0, this.postGroup);
    post.setBindGroup(1, this.frameGroup!);
    post.draw(3);
    post.end();
  }
}

type V3 = [number, number, number];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const wrap = (d: number) => d - WORLD.size * Math.round(d / WORLD.size);

/** A fly-frame azimuth / elevation (degrees) as a world direction: scene.wgsl's toWorld on the CPU. */
function flyToWorld(azDeg: number, elDeg: number, heading: number, pitch: number): V3 {
  const az = (azDeg * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  const d: V3 = [Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)];
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const p: V3 = [d[0], d[1] * cp - d[2] * sp, d[1] * sp + d[2] * cp];
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  return [p[0] * c - p[2] * s, p[1], p[0] * s + p[2] * c];
}

/** flyToWorld inverted: a world direction as fly-frame azimuth / elevation, degrees. */
function worldToFly(r: V3, heading: number, pitch: number): { az: number; el: number } {
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const p: V3 = [r[0] * c + r[2] * s, r[1], -r[0] * s + r[2] * c];
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const d: V3 = [p[0], p[1] * cp + p[2] * sp, -p[1] * sp + p[2] * cp];
  const DEG = 180 / Math.PI;
  return { az: Math.atan2(d[0], -d[2]) * DEG, el: Math.asin(Math.max(-1, Math.min(1, d[1]))) * DEG };
}
