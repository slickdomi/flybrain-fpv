import type { Readback } from "./brain";

/** Named groups of neurons whose spike counters are read back every chunk. */
export class Probes {
  private list: number[] = [];
  private groups = new Map<string, { start: number; count: number }>();

  add(name: string, neurons: ArrayLike<number>): this {
    if (this.groups.has(name)) throw new Error(`probe group ${name} exists`);
    this.groups.set(name, { start: this.list.length, count: neurons.length });
    for (let k = 0; k < neurons.length; k++) this.list.push(neurons[k]);
    return this;
  }

  has(name: string) {
    return this.groups.has(name);
  }

  group(name: string): { start: number; count: number } {
    const g = this.groups.get(name);
    if (!g) throw new Error(`no probe group ${name}`);
    return g;
  }

  names(): string[] {
    return [...this.groups.keys()];
  }

  get size() {
    return this.list.length;
  }

  indices(): Uint32Array {
    return Uint32Array.from(this.list);
  }
}

/** Firing rates per probe group (mean Hz per cell), low-pass filtered in brain time. */
export class Rates {
  /** spikes of every probe neuron since the previous readback */
  delta: Uint32Array;
  /** brain ms since the previous readback */
  dtMs = 0;
  private prev: Uint32Array | null = null;
  private prevTime = 0;
  private fast = new Map<string, number>();
  private slow = new Map<string, number>();
  private inst = new Map<string, number>();

  constructor(
    readonly probes: Probes,
    /** ms, filter of hz() */
    public tauMs = 50,
    /** ms, filter of slowHz() */
    public slowTauMs = 250,
  ) {
    this.delta = new Uint32Array(probes.size);
  }

  update(r: Readback) {
    if (r.counts.length !== this.delta.length) {
      // probes changed: start over
      this.delta = new Uint32Array(r.counts.length);
      this.prev = null;
    }
    const dtMs = r.brainTime - this.prevTime;
    this.dtMs = this.prev && dtMs > 0 ? dtMs : 0;
    if (this.prev && dtMs > 0) {
      const k = 1 - Math.exp(-dtMs / this.tauMs);
      const ks = 1 - Math.exp(-dtMs / this.slowTauMs);
      for (let i = 0; i < r.counts.length; i++) {
        // counters restart from zero after a brain reset
        this.delta[i] = r.counts[i] >= this.prev[i] ? r.counts[i] - this.prev[i] : r.counts[i];
      }
      for (const name of this.probes.names()) {
        const { start, count } = this.probes.group(name);
        let spikes = 0;
        for (let i = start; i < start + count; i++) spikes += this.delta[i];
        const hz = (spikes * 1000) / dtMs / Math.max(1, count);
        this.inst.set(name, hz);
        this.fast.set(name, (this.fast.get(name) ?? 0) + (hz - (this.fast.get(name) ?? 0)) * k);
        this.slow.set(name, (this.slow.get(name) ?? 0) + (hz - (this.slow.get(name) ?? 0)) * ks);
      }
    } else {
      this.delta.fill(0);
    }
    this.prev = r.counts.slice();
    this.prevTime = r.brainTime;
  }

  /** mean Hz per cell, fast filter */
  hz(name: string) {
    return this.fast.get(name) ?? 0;
  }

  /** mean Hz per cell, slow filter */
  slowHz(name: string) {
    return this.slow.get(name) ?? 0;
  }

  /** mean Hz per cell over the last readback only */
  instHz(name: string) {
    return this.inst.get(name) ?? 0;
  }

  /** spikes of the group since the previous readback */
  spikes(name: string) {
    const { start, count } = this.probes.group(name);
    let s = 0;
    for (let i = start; i < start + count; i++) s += this.delta[i];
    return s;
  }
}
