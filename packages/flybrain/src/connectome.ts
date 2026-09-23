import type { ConnectomeInfo } from "./data/types";

export type Side = "left" | "right";

export interface NeuronQuery {
  /** exact type name, list of names, or a pattern (e.g. /^LC10/) */
  type?: string | string[] | RegExp;
  side?: Side;
  /** e.g. "descending_neuron" or /^visual_projection/ */
  superclass?: string | RegExp;
  /** only spiking (true) or only graded optic-lobe (false) cells */
  spiking?: boolean;
  /** cells whose connectome receptive field lies within `radius` degrees of (az, el) */
  receptiveField?: { az: number; el: number; radius: number };
}

/** Finds neurons by cell type, hemisphere, superclass and receptive field. */
export class Connectome {
  private byType = new Map<string, number>();

  constructor(readonly info: ConnectomeInfo) {
    info.types.forEach((t, i) => this.byType.set(t, i));
  }

  get size() {
    return this.info.n;
  }

  typeOf(neuron: number): string {
    return this.info.types[this.info.typeId[neuron]] ?? "";
  }

  sideOf(neuron: number): Side | "unknown" {
    const s = this.info.side[neuron];
    return s === 1 ? "left" : s === 2 ? "right" : "unknown";
  }

  superclassOf(neuron: number): string {
    return this.info.meta.superclasses[this.info.superclass[neuron]] ?? "";
  }

  transmitterOf(neuron: number): string {
    return this.info.meta.transmitters[this.info.nt[neuron]] ?? "";
  }

  hasType(name: string) {
    return this.byType.has(name);
  }

  select(q: NeuronQuery): Uint32Array {
    const { info } = this;
    let typeOk: ((t: number) => boolean) | null = null;
    if (typeof q.type === "string") {
      const id = this.byType.get(q.type) ?? -1;
      typeOk = (t) => t === id;
    } else if (Array.isArray(q.type)) {
      const ids = new Set(q.type.map((name) => this.byType.get(name) ?? -1));
      typeOk = (t) => ids.has(t);
    } else if (q.type instanceof RegExp) {
      const re = q.type;
      const ok = info.types.map((name) => re.test(name));
      typeOk = (t) => ok[t];
    }
    let scOk: ((s: number) => boolean) | null = null;
    if (q.superclass !== undefined) {
      const pat = q.superclass;
      const ok = info.meta.superclasses.map((name) => (typeof pat === "string" ? name === pat : pat.test(name)));
      scOk = (s) => ok[s];
    }
    const side = q.side === "left" ? 1 : q.side === "right" ? 2 : 0;
    const out: number[] = [];
    const start = q.spiking === true ? info.ng : 0;
    const end = q.spiking === false ? info.ng : info.n;
    for (let i = start; i < end; i++) {
      if (typeOk && !typeOk(info.typeId[i])) continue;
      if (side && info.side[i] !== side) continue;
      if (scOk && !scOk(info.superclass[i])) continue;
      if (q.receptiveField) {
        if (!(info.rf[3 * i + 2] > 0)) continue;
        const { az, el, radius } = q.receptiveField;
        if (Math.hypot(info.rf[3 * i] - az, info.rf[3 * i + 1] - el) >= radius) continue;
      }
      out.push(i);
    }
    return Uint32Array.from(out);
  }
}
