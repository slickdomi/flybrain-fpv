/** Constant and time-limited external input per neuron, with the index range that changed since the last upload. */
export class DriveTable {
  readonly values: Float32Array;
  private tonic: Float32Array;
  private pulses: { idx: ArrayLike<number>; mV: number; until: number }[] = [];
  private lo = Infinity;
  private hi = -1;

  constructor(n: number) {
    this.values = new Float32Array(n);
    this.tonic = new Float32Array(n);
  }

  private touch(idx: ArrayLike<number>) {
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k];
      if (i < this.lo) this.lo = i;
      if (i > this.hi) this.hi = i;
    }
  }

  setTonic(idx: ArrayLike<number>, mV: number) {
    for (let k = 0; k < idx.length; k++) this.tonic[idx[k]] = mV;
    this.touch(idx);
  }

  pulse(idx: ArrayLike<number>, mV: number, durationMs: number, now: number) {
    this.pulses.push({ idx: Array.from(idx), mV, until: now + durationMs });
    this.touch(idx);
  }

  /** Drops pulses that ended by `now`. */
  advance(now: number) {
    const keep = [];
    for (const p of this.pulses) {
      if (p.until > now) keep.push(p);
      else this.touch(p.idx);
    }
    this.pulses = keep;
  }

  /** Recomputes the changed range and returns it as [start, end), or null if nothing changed. */
  takeDirty(): [number, number] | null {
    if (this.hi < 0) return null;
    const lo = this.lo;
    const hi = this.hi + 1;
    this.values.set(this.tonic.subarray(lo, hi), lo);
    for (const p of this.pulses) {
      for (let k = 0; k < p.idx.length; k++) {
        const i = p.idx[k];
        if (i >= lo && i < hi) this.values[i] += p.mV;
      }
    }
    this.lo = Infinity;
    this.hi = -1;
    return [lo, hi];
  }

  /** Everything, e.g. after a backend restart. */
  markAll() {
    this.lo = 0;
    this.hi = this.values.length - 1;
  }
}
