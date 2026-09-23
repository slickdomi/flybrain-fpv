import type { ConnectomeInfo } from "./data/types";

/** Half acceptance angle of one ommatidium (radians); the eye columns are 4.8 degrees apart. */
export const ACCEPTANCE = (2.4 * Math.PI) / 180;

/** Luminance (0 = black, 1 = white) seen along a world-space unit direction. */
export type Shader = (x: number, y: number, z: number) => number;

/**
 * The fly's compound eyes: 10,321 visual units (5,895 photoreceptors and 4,426 L1-L3 lamina cells), each with a
 * viewing direction. Point them into your world with orient(), fill `lum` (render() or your own batch
 * renderer over `world`), then normalize() and hand `contrast` to FlyBrain.setVision().
 *
 * Frames: the fly frame is +x right, +y up, -z forward. World = fly frame rotated by pitch (about x) and then
 * yaw (about y): yaw > 0 turns right, pitch > 0 looks up. With yaw 0 the fly looks along world -z.
 */
export class Retina {
  readonly count: number;
  /** unit directions in the fly frame, xyz per unit */
  readonly local: Float32Array;
  /** unit directions in the world after orient(), xyz per unit */
  readonly world: Float32Array;
  /** azimuth / elevation in the fly frame (radians), for plotting */
  readonly az: Float32Array;
  readonly el: Float32Array;
  /** 1 = left eye, 2 = right eye (the eye whose mean luminance normalises the unit) */
  readonly side: Uint8Array;
  /** 1 for lamina columns: one per ommatidium, they define each eye's mean luminance */
  readonly lamina: Uint8Array;
  /** luminance per unit, filled by render() or by you */
  readonly lum: Float32Array;
  /** contrast per unit, filled by normalize() */
  readonly contrast: Float32Array;
  yaw = 0;
  pitch = 0;

  constructor(info: Pick<ConnectomeInfo, "visUnits" | "visCount">) {
    const f = new Float32Array(info.visUnits);
    const u = new Uint32Array(info.visUnits);
    this.count = info.visCount;
    this.local = new Float32Array(3 * this.count);
    this.world = new Float32Array(3 * this.count);
    this.az = new Float32Array(this.count);
    this.el = new Float32Array(this.count);
    this.side = new Uint8Array(this.count);
    this.lamina = new Uint8Array(this.count);
    this.lum = new Float32Array(this.count).fill(1);
    this.contrast = new Float32Array(this.count);
    for (let k = 0; k < this.count; k++) {
      const x = f[8 * k];
      const y = f[8 * k + 1];
      const z = f[8 * k + 2];
      this.local.set([x, y, z], 3 * k);
      this.az[k] = Math.atan2(x, -z);
      this.el[k] = Math.asin(Math.max(-1, Math.min(1, y)));
      this.side[k] = u[8 * k + 5];
      this.lamina[k] = +(u[8 * k + 6] === 2);
    }
    this.orient(0, 0);
  }

  /** Points the eyes: yaw > 0 turns right, pitch > 0 looks up (radians). */
  orient(yaw: number, pitch = 0): this {
    this.yaw = yaw;
    this.pitch = pitch;
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const { local, world } = this;
    for (let k = 0; k < 3 * this.count; k += 3) {
      const x = local[k];
      const py = local[k + 1] * cp - local[k + 2] * sp;
      const pz = local[k + 1] * sp + local[k + 2] * cp;
      world[k] = x * c - pz * s;
      world[k + 1] = py;
      world[k + 2] = x * s + pz * c;
    }
    return this;
  }

  /** Fills `lum` by calling `shade` per unit; with blur, averages 5 rays across the acceptance angle. */
  render(shade: Shader, blur = true): this {
    const { world, lum } = this;
    for (let k = 0; k < this.count; k++) {
      const x = world[3 * k];
      const y = world[3 * k + 1];
      const z = world[3 * k + 2];
      if (!blur) {
        lum[k] = shade(x, y, z);
        continue;
      }
      // two tangents across the view direction, ACCEPTANCE long
      let ax = z;
      let az = -x;
      let al = Math.hypot(ax, az);
      if (al < 1e-3) {
        ax = 1;
        az = 0;
        al = 1;
      }
      ax = (ax / al) * ACCEPTANCE;
      az = (az / al) * ACCEPTANCE;
      const bx = y * az;
      const by = z * ax - x * az;
      const bz = -y * ax;
      const sample = (dx: number, dy: number, dz: number) => {
        const l = Math.hypot(x + dx, y + dy, z + dz);
        return shade((x + dx) / l, (y + dy) / l, (z + dz) / l);
      };
      lum[k] = (2 * shade(x, y, z) + sample(ax, 0, az) + sample(-ax, 0, -az) + sample(bx, by, bz) + sample(-bx, -by, -bz)) / 6;
    }
    return this;
  }

  /**
   * Photoreceptor input: contrast against each eye's mean luminance, clamped to [-1, 3] (as in the validated
   * model). A dark object on a bright ground gives about -1.
   */
  normalize(): Float32Array {
    const sums = [0, 0, 0];
    const counts = [0, 0, 0];
    for (let k = 0; k < this.count; k++) {
      if (!this.lamina[k]) continue;
      sums[this.side[k]] += this.lum[k];
      counts[this.side[k]]++;
    }
    const means = sums.map((s, i) => Math.max(s / Math.max(counts[i], 1), 1e-3));
    for (let k = 0; k < this.count; k++) {
      const m = means[this.side[k]];
      this.contrast[k] = Math.min(3, Math.max(-1, (this.lum[k] - m) / m));
    }
    return this.contrast;
  }
}
