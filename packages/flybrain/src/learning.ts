import type { FlyBrain } from "./brain";
import { packPlastic } from "./data/build";
import type { Probes, Rates } from "./readout";

export interface DopamineLearningOptions {
  /**
   * Name of an extracted edge list (LoadOptions.extract) from dopamine neurons to the postsynaptic side of the
   * plastic synapses, e.g. { name: "dan", pre: "^(PAM|PPL1)", post: "^MBON" }. Its synapse counts decide how much
   * of each dopamine cell's release reaches each postsynaptic cell's compartment.
   */
  dopamineEdges: string;
  /** Weight change rate: relative depression per second, per eligibility spike, per Hz of dopamine. */
  rate?: number;
  /** ms, decay of the presynaptic eligibility trace (spikes) */
  eligibilityMs?: number;
  /** ms, decay of the dopamine trace */
  dopamineMs?: number;
  /**
   * ms, slow average of the dopamine each postsynaptic cell receives. Only dopamine above it teaches (phasic bursts,
   * not the dopamine cells' spontaneous firing).
   */
  baselineMs?: number;
  /** Hz of dopamine above the baseline before anything is learned */
  thresholdHz?: number;
  /** ms, weights relax back to the connectome weight with this time constant (0 = never) */
  recoveryMs?: number;
  /**
   * ms, TWO-SPEED MEMORY (0 = off, the default, and then nothing below applies). While a synapse's depression
   * is still in its fast part (the one recoveryMs relaxes), it also moves into a slow part at this rate. Only
   * edges marked in `consolidates` do this (all edges if it is null). With it off, the rule is exactly the
   * one-speed rule: the slow part stays 0.
   */
  consolidateMs?: number;
  /** ms, the slow part relaxes back to the connectome weight with this time constant (0 = never) */
  slowRecoveryMs?: number;
  /** lowest relative weight */
  minWeight?: number;
  /** ms of brain time between weight uploads to the brain */
  uploadMs?: number;
}

/**
 * Dopamine-gated plasticity of the kind that stores memories in the fly's mushroom body: a synapse from a
 * recently active Kenyon cell onto an output neuron (MBON) is depressed when dopamine arrives in that MBON's
 * compartment (Hige et al. 2015). Here the compartment match comes straight from the connectome: dopamine
 * neuron -> MBON synapse counts. Weights recover slowly toward the connectome weight.
 *
 * Optionally the depression has two parts (consolidateMs > 0): a fast one that recovers with recoveryMs and
 * transfers into a slow one at 1/consolidateMs, and the slow one, which recovers with slowRecoveryMs. `weights`
 * is always the total, so a readout needs nothing new. Both parts relax by the exact solution of that linear
 * system, so a long gap with no brain time (elapseOffline: a night, a test delay) costs one step.
 *
 * Load the brain with LoadOptions.plastic (e.g. { pre: "^KC", post: "^MBON" }) and the matching extract list,
 * create this before calling brain.setProbes(probes.indices()), and call update(rates) after every readback.
 */
export class DopamineLearning {
  enabled = true;
  rate: number;
  eligibilityMs: number;
  dopamineMs: number;
  baselineMs: number;
  thresholdHz: number;
  recoveryMs: number;
  /** TWO-SPEED MEMORY: see DopamineLearningOptions. 0 = off. */
  consolidateMs: number;
  slowRecoveryMs: number;
  /** per plastic edge, 1 if its depression consolidates; null = every edge does (when consolidateMs > 0) */
  consolidates: Uint8Array | null = null;
  minWeight: number;
  uploadMs: number;
  /** relative weight of each plastic edge (1 = connectome weight), in info.plastic order */
  readonly weights: Float32Array;
  /** the slow part of each edge's depression (1 - weight = fast + slow); all 0 unless consolidateMs > 0 */
  readonly slow: Float32Array;
  /** presynaptic cells (e.g. Kenyon cells) */
  readonly pre: Uint32Array;
  /** postsynaptic cells that receive dopamine (e.g. MBONs) */
  readonly post: Uint32Array;
  /** dopamine cells */
  readonly dopamine: Uint32Array;
  /** dopamineHz: mean rate of the dopamine cells; phasicHz: largest dopamine burst above baseline at any postsynaptic cell */
  readonly stats = { meanWeight: 1, depressedFraction: 0, dopamineHz: 0, phasicHz: 0, uploads: 0, slowMean: 0 };
  /** index into `pre` of each plastic edge's presynaptic cell */
  readonly edgePre: Uint32Array;
  /** index into `post` of each plastic edge's postsynaptic cell */
  readonly edgePost: Int32Array;
  /** eligibility trace per presynaptic cell (recent spikes, decaying with eligibilityMs) */
  readonly eligibility: Float32Array;
  /** Hz of dopamine above baseline and threshold arriving at each postsynaptic cell right now (0 = nothing teaches) */
  readonly phasic: Float32Array;

  private da: Float32Array;
  private postBase: Float32Array;
  private reach: { offsets: Uint32Array; dan: Uint32Array; frac: Float32Array };
  private packed: Uint32Array;
  private targets: Uint32Array;
  private base: Int16Array;
  private sinceUpload = 0;
  private elapsedMs = 0;
  private dirty = false;

  constructor(
    private brain: FlyBrain,
    private probes: Probes,
    opts: DopamineLearningOptions,
  ) {
    const plastic = brain.info.plastic;
    if (!plastic) throw new Error("DopamineLearning: load the brain with LoadOptions.plastic");
    const edges = brain.info.extracted[opts.dopamineEdges];
    if (!edges) throw new Error(`DopamineLearning: no extracted edge list "${opts.dopamineEdges}"`);
    this.rate = opts.rate ?? 0.005;
    this.eligibilityMs = opts.eligibilityMs ?? 1000;
    this.dopamineMs = opts.dopamineMs ?? 200;
    this.baselineMs = opts.baselineMs ?? 10000;
    this.thresholdHz = opts.thresholdHz ?? 2;
    this.recoveryMs = opts.recoveryMs ?? 120000;
    this.consolidateMs = opts.consolidateMs ?? 0;
    this.slowRecoveryMs = opts.slowRecoveryMs ?? 0;
    this.minWeight = opts.minWeight ?? 0;
    this.uploadMs = opts.uploadMs ?? 200;

    const count = plastic.packed.length;
    this.weights = new Float32Array(count).fill(1);
    this.slow = new Float32Array(count);
    this.packed = plastic.packed.slice();
    this.base = plastic.base;
    this.targets = new Uint32Array(count);
    for (let e = 0; e < count; e++) this.targets[e] = plastic.packed[e] & 0x3ffff;

    const n = brain.info.n;
    const local = new Int32Array(n).fill(-1);
    const pre: number[] = [];
    this.edgePre = new Uint32Array(count);
    for (let e = 0; e < count; e++) {
      const p = plastic.pre[e];
      if (local[p] < 0) {
        local[p] = pre.length;
        pre.push(p);
      }
      this.edgePre[e] = local[p];
    }
    this.pre = Uint32Array.from(pre);

    // Postsynaptic cells and the dopamine cells that reach them.
    const postLocal = new Int32Array(n).fill(-1);
    const post: number[] = [];
    for (let e = 0; e < count; e++) {
      const q = this.targets[e];
      if (postLocal[q] < 0) {
        postLocal[q] = post.length;
        post.push(q);
      }
    }
    this.post = Uint32Array.from(post);
    const danLocal = new Int32Array(n).fill(-1);
    const dans: number[] = [];
    const perPost: Map<number, number>[] = post.map(() => new Map());
    for (let j = 0; j < edges.pre.length; j++) {
      const q = postLocal[edges.post[j]];
      if (q < 0) continue;
      const d = edges.pre[j];
      if (danLocal[d] < 0) {
        danLocal[d] = dans.length;
        dans.push(d);
      }
      const m = perPost[q];
      m.set(danLocal[d], (m.get(danLocal[d]) ?? 0) + edges.count[j]);
    }
    this.dopamine = Uint32Array.from(dans);
    const offsets = new Uint32Array(post.length + 1);
    const dan: number[] = [];
    const frac: number[] = [];
    perPost.forEach((m, q) => {
      let total = 0;
      for (const c of m.values()) total += c;
      for (const [d, c] of m) {
        dan.push(d);
        frac.push(c / total);
      }
      offsets[q + 1] = dan.length;
    });
    this.reach = { offsets, dan: Uint32Array.from(dan), frac: Float32Array.from(frac) };
    this.edgePost = new Int32Array(count);
    for (let e = 0; e < count; e++) this.edgePost[e] = postLocal[this.targets[e]];

    this.eligibility = new Float32Array(this.pre.length);
    this.da = new Float32Array(this.dopamine.length);
    this.phasic = new Float32Array(this.post.length);
    this.postBase = new Float32Array(this.post.length);
    probes.add("learning:pre", this.pre);
    probes.add("learning:dopamine", this.dopamine);
  }

  /** Call after rates.update(readback). */
  update(rates: Rates) {
    const dt = rates.dtMs;
    if (!(dt > 0)) return;
    const pg = this.probes.group("learning:pre");
    const dg = this.probes.group("learning:dopamine");
    const ke = Math.exp(-dt / this.eligibilityMs);
    const kd = Math.exp(-dt / this.dopamineMs);
    const { eligibility: elig, da, phasic: postDa, reach } = this;
    const delta = rates.delta;
    for (let k = 0; k < elig.length; k++) elig[k] = elig[k] * ke + delta[pg.start + k];
    let daSum = 0;
    for (let d = 0; d < da.length; d++) {
      da[d] = da[d] * kd + (delta[dg.start + d] * 1000) / this.dopamineMs;
      daSum += da[d];
    }
    this.stats.dopamineHz = daSum / Math.max(1, da.length);
    // dopamine at each postsynaptic cell, minus its slow baseline: only bursts teach
    // (a running mean until baselineMs has passed, so the baseline starts where the dopamine cells are)
    this.elapsedMs += dt;
    const kb = 1 - Math.exp(-dt / Math.min(this.baselineMs, this.elapsedMs));
    const base = this.postBase;
    let maxPhasic = 0;
    for (let q = 0; q < postDa.length; q++) {
      let s = 0;
      for (let r = reach.offsets[q]; r < reach.offsets[q + 1]; r++) s += reach.frac[r] * da[reach.dan[r]];
      const phasic = s - base[q] - this.thresholdHz;
      base[q] += (s - base[q]) * kb;
      postDa[q] = phasic > 0 ? phasic : 0;
      if (phasic > maxPhasic) maxPhasic = phasic;
    }
    this.stats.phasicHz = maxPhasic;
    if (this.enabled && maxPhasic > 0) {
      const k = (this.rate * dt) / 1000;
      const w = this.weights;
      for (let e = 0; e < w.length; e++) {
        const x = elig[this.edgePre[e]] * postDa[this.edgePost[e]];
        if (x > 0) w[e] = Math.max(this.minWeight, w[e] * Math.exp(-k * x));
      }
      this.dirty = true;
    }
    this.sinceUpload += dt;
    if (this.sinceUpload >= this.uploadMs) this.flush();
  }

  /**
   * Lets `ms` pass with no brain time: both parts of every edge's depression relax (and, with consolidateMs,
   * transfer), exactly as they would have, and the result is uploaded. For a night, or a delay before a test.
   */
  elapseOffline(ms: number) {
    this.relax(ms);
    this.sinceUpload = 0;
    this.upload();
  }

  /**
   * The exact solution over `ms` of, per edge, with f the fast part of the depression and s the slow part:
   *   df/dt = -f (a + c)        a = 1/recoveryMs, c = 1/consolidateMs if the edge consolidates, else 0
   *   ds/dt = c f - b s         b = 1/slowRecoveryMs
   * With consolidation off, c = 0 and s = 0, and this is exactly the one-speed rule w += (1 - w)(1 - e^-a ms).
   */
  private relax(ms: number) {
    if (!(ms > 0)) return;
    const w = this.weights;
    const s = this.slow;
    const a = this.recoveryMs > 0 ? 1 / this.recoveryMs : 0;
    const on = this.consolidateMs > 0;
    const c = on ? 1 / this.consolidateMs : 0;
    const b = this.slowRecoveryMs > 0 ? 1 / this.slowRecoveryMs : 0;
    // per edge class (consolidating or not): fast decay factor, and the slow part's gain from the fast one
    const fOff = Math.exp(-a * ms);
    const fOn = Math.exp(-(a + c) * ms);
    const sDecay = Math.exp(-b * ms);
    const k = a + c - b;
    const gain = !on ? 0 : Math.abs(k) < 1e-15 ? c * ms * sDecay : (c * (sDecay - fOn)) / k;
    const mask = this.consolidates;
    let slowSum = 0;
    for (let e = 0; e < w.length; e++) {
      if (w[e] >= 1 && s[e] === 0) continue;
      const cons = on && (mask === null || mask[e] === 1);
      const f = Math.max(0, 1 - w[e] - s[e]);
      const sNew = cons ? s[e] * sDecay + gain * f : s[e] * sDecay;
      const fNew = f * (cons ? fOn : fOff);
      slowSum += sNew;
      if (fNew === f && sNew === s[e]) continue;
      s[e] = sNew;
      w[e] = Math.max(this.minWeight, 1 - fNew - sNew);
      this.dirty = true;
    }
    this.stats.slowMean = slowSum / Math.max(1, w.length);
  }

  private flush() {
    this.relax(this.sinceUpload);
    this.sinceUpload = 0;
    this.upload();
  }

  private upload() {
    const w = this.weights;
    if (!this.dirty) return;
    this.dirty = false;
    let sum = 0;
    let depressed = 0;
    for (let e = 0; e < w.length; e++) {
      sum += w[e];
      if (w[e] < 0.9) depressed++;
      this.packed[e] = packPlastic(this.targets[e], this.base[e] * w[e]);
    }
    this.stats.meanWeight = sum / Math.max(1, w.length);
    this.stats.depressedFraction = depressed / Math.max(1, w.length);
    this.stats.uploads++;
    this.brain.setPlasticWeights(this.packed);
  }

  /** Forgets everything: all weights back to the connectome. */
  resetWeights() {
    this.weights.fill(1);
    this.slow.fill(0);
    this.eligibility.fill(0);
    this.da.fill(0);
    this.dirty = true;
    this.flush();
  }
}
