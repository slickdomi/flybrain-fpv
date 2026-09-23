// CPU (plain JavaScript) port of the WebGPU brain. Same model, same update order as
// gpu/shaders/{retina,graded,couple,neuron,scatter}.wgsl; runs inside a worker and is simply slower.

import type { BrainData } from "../data/types";
import { FIXED_POINT, type ModelParams } from "../model";

export class CpuSim {
  brainTime = 0;
  readonly act: Float32Array;
  private n: number;
  private ng: number;
  private v: Float32Array;
  private g: Float32Array;
  private refr: Float32Array;
  private theta: Float32Array;
  private cnt: Uint32Array;
  private baseDrive: Float32Array;
  private gDrive: Float32Array;
  private ring: Int32Array;
  private ringSlots: number;
  private delaySteps: number;
  private aCur: Float32Array;
  private aNext: Float32Array;
  private ext: Float32Array;
  private mode: Uint8Array;
  private contrast: Float32Array;
  private spikeList: Int32Array;
  private t = 0;
  private nextGraded = 0;
  private probeIdx: Uint32Array = new Uint32Array(0);
  private spikeOffsets: Uint32Array;
  private spikeEdges: Uint32Array;
  private plasticOffsets: Uint32Array;
  private plasticEdges: Uint32Array;
  private gOff: Uint32Array;
  private gU: Uint32Array;
  private gF: Float32Array;
  private iOff: Uint32Array;
  private iU: Uint32Array;
  private iF: Float32Array;
  private unitsF: Float32Array;
  private unitsU: Uint32Array;
  private visCount: number;

  constructor(d: BrainData, private m: ModelParams) {
    this.n = d.n;
    this.ng = d.ng;
    const n = d.n;
    this.v = new Float32Array(n);
    this.g = new Float32Array(n);
    this.refr = new Float32Array(n);
    this.theta = new Float32Array(n);
    this.act = new Float32Array(n);
    this.cnt = new Uint32Array(n);
    this.baseDrive = new Float32Array(n);
    this.gDrive = new Float32Array(n);
    this.delaySteps = Math.max(1, Math.round(m.tDelay / m.dt));
    this.ringSlots = this.delaySteps + 1;
    this.ring = new Int32Array(this.ringSlots * n);
    this.aCur = new Float32Array(d.ng);
    this.aNext = new Float32Array(d.ng);
    this.ext = new Float32Array(d.ng);
    this.mode = new Uint8Array(d.ng);
    this.contrast = new Float32Array(d.visCount);
    this.spikeList = new Int32Array(n);
    this.spikeOffsets = d.spikeOffsets;
    this.spikeEdges = d.spikeEdges;
    this.plasticOffsets = d.plastic?.offsets ?? new Uint32Array(n + 1);
    this.plasticEdges = d.plastic?.packed ?? new Uint32Array(0);
    this.gOff = d.gradedOffsets;
    this.gU = new Uint32Array(d.gradedEntries);
    this.gF = new Float32Array(d.gradedEntries);
    this.iOff = d.ifaceOffsets;
    this.iU = new Uint32Array(d.ifaceEntries);
    this.iF = new Float32Array(d.ifaceEntries);
    this.unitsF = new Float32Array(d.visUnits);
    this.unitsU = new Uint32Array(d.visUnits);
    this.visCount = d.visCount;
    for (let u = 0; u < d.visCount; u++) this.mode[this.unitsU[8 * u + 4]] = this.unitsU[8 * u + 6];
  }

  setParams(m: ModelParams) {
    this.m = { ...m, dt: this.m.dt, tDelay: this.m.tDelay };
  }

  setDrive(start: number, values: Float32Array) {
    this.baseDrive.set(values, start);
  }

  setContrast(c: Float32Array) {
    this.contrast.set(c);
  }

  setPlastic(packed: Uint32Array) {
    this.plasticEdges.set(packed);
  }

  setProbes(idx: Uint32Array) {
    this.probeIdx = idx;
  }

  reset(includeGraded: boolean) {
    this.v.fill(0);
    this.g.fill(0);
    this.refr.fill(0);
    this.theta.fill(0);
    this.act.fill(0);
    this.cnt.fill(0);
    this.ring.fill(0);
    if (includeGraded) {
      this.aCur.fill(0);
      this.aNext.fill(0);
    }
  }

  graded(): Float32Array {
    return this.aCur;
  }

  /** Simulates `brainMs`; returns cumulative probe counts and spikes emitted. */
  step(brainMs: number): { counts: Uint32Array; spikes: number; frameMs: number } {
    const m = this.m;
    const steps = Math.max(0, Math.round(brainMs / m.dt));
    this.retina();
    const em = Math.exp(-m.dt / m.tauM);
    const es = Math.exp(-m.dt / m.tauS);
    const kg = (m.tauS / (m.tauS - m.tauM)) * (es - em);
    const adaptDecay = Math.exp(-m.dt / m.tauAdapt);
    const actDecay = Math.exp(-m.dt / m.actTau);
    const { n, ng, v, g, refr, theta, act, cnt, baseDrive, gDrive, ring, spikeList, spikeOffsets, spikeEdges, plasticOffsets, plasticEdges } = this;
    let spikes = 0;
    for (let s = 0; s < steps; s++) {
      if (this.brainTime >= this.nextGraded) {
        this.gradedStep();
        this.couple();
        this.nextGraded += m.gradedDt;
      }
      const readBase = (this.t % this.ringSlots) * n;
      const writeBase = ((this.t + this.delaySteps) % this.ringSlots) * n;
      let nsp = 0;
      for (let i = ng; i < n; i++) {
        const inp = ring[readBase + i] / FIXED_POINT;
        ring[readBase + i] = 0;
        theta[i] *= adaptDecay;
        act[i] *= actDecay;
        if (refr[i] > 0) {
          refr[i] -= m.dt;
          v[i] = m.vReset;
          g[i] += m.wSyn * inp;
          continue;
        }
        const drive = baseDrive[i] + gDrive[i];
        const vi = drive + (v[i] - drive) * em + g[i] * kg;
        g[i] = g[i] * es + m.wSyn * inp;
        if (vi > m.vThreshold + theta[i]) {
          v[i] = m.vReset;
          g[i] = 0;
          refr[i] = m.tRefractory;
          theta[i] += m.adapt;
          act[i] += 1;
          cnt[i]++;
          spikeList[nsp++] = i;
        } else {
          v[i] = vi;
        }
      }
      for (let k = 0; k < nsp; k++) {
        const pre = spikeList[k];
        for (let e = spikeOffsets[pre], end = spikeOffsets[pre + 1]; e < end; e++) {
          const packed = spikeEdges[e];
          ring[writeBase + (packed & 0x3ffff)] += ((packed | 0) >> 18) * FIXED_POINT;
        }
        for (let e = plasticOffsets[pre], end = plasticOffsets[pre + 1]; e < end; e++) {
          const packed = plasticEdges[e];
          ring[writeBase + (packed & 0x3ffff)] += (packed | 0) >> 18;
        }
      }
      spikes += nsp;
      this.brainTime += m.dt;
      this.t++;
    }
    const counts = new Uint32Array(this.probeIdx.length);
    for (let k = 0; k < counts.length; k++) counts[k] = cnt[this.probeIdx[k]];
    return { counts, spikes, frameMs: steps * m.dt };
  }

  private retina() {
    const { unitsF, unitsU, ext, contrast } = this;
    for (let u = 0; u < this.visCount; u++) ext[unitsU[8 * u + 4]] = unitsF[8 * u + 3] * contrast[u];
  }

  private gradedStep() {
    const m = this.m;
    const decay = Math.exp(-m.gradedDt / m.gradedTau);
    const { ng, gOff, gU, gF, aCur, aNext, ext, mode } = this;
    const lo = m.gradedMin;
    const hi = m.gradedMax;
    for (let i = 0; i < ng; i++) {
      if (mode[i] === 1) {
        aNext[i] = ext[i];
        continue;
      }
      let sum = 0;
      for (let e = gOff[i], end = gOff[i + 1]; e < end; e++) {
        const a = aCur[gU[2 * e]];
        sum += gF[2 * e + 1] * (a < lo ? lo : a > hi ? hi : a);
      }
      let goal = m.gradedGain * sum;
      if (mode[i] === 2) goal += ext[i];
      aNext[i] = goal + (aCur[i] - goal) * decay;
    }
    this.aCur = aNext;
    this.aNext = aCur;
  }

  private couple() {
    const m = this.m;
    const { n, ng, iOff, iU, iF, aCur, gDrive } = this;
    const lo = m.gradedMin;
    const hi = m.gradedMax;
    for (let r = 0; r < n - ng; r++) {
      let sum = 0;
      for (let e = iOff[r], end = iOff[r + 1]; e < end; e++) {
        const a = aCur[iU[2 * e]];
        sum += iF[2 * e + 1] * (a < lo ? lo : a > hi ? hi : a);
      }
      gDrive[r + ng] = m.coupling * sum;
    }
  }
}
