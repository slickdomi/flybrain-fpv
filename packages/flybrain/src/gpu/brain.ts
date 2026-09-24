import commonSrc from "./shaders/common.wgsl?raw";
import coupleSrc from "./shaders/couple.wgsl?raw";
import gradedSrc from "./shaders/graded.wgsl?raw";
import neuronSrc from "./shaders/neuron.wgsl?raw";
import probeSrc from "./shaders/probe.wgsl?raw";
import retinaSrc from "./shaders/retina.wgsl?raw";
import scatterSrc from "./shaders/scatter.wgsl?raw";
import type { Activity, FlyBrain, Readback } from "../brain";
import type { BrainData, ConnectomeInfo } from "../data/types";
import { DriveTable } from "../drive";
import { DEFAULT_MODEL, type ModelParams } from "../model";
import { ACCEPTANCE } from "../vision";

/**
 * Sample the world on the GPU instead of uploading contrast from the CPU.
 *
 * `shader` must be a WGSL module exposing two entry points, `meanPass` and `contrastPass`, over the bind
 * group [scene: uniform, eyeParams: uniform, units: read, ext: read_write, means: read_write] -- in practice
 * your scene shader concatenated with an eye shader. `meanPass` fills `means[1]` and `means[2]` with each
 * eye's mean luminance; `contrastPass` writes `ext[u.g] = u.weight * contrast` for every visual unit, exactly
 * as the built-in retina pass does. `eyeParams` is {count: u32, blur: f32} (blur = half acceptance angle).
 *
 * While this is set, setVision() is ignored: the eye reads the scene buffer directly and nothing round-trips
 * through the CPU.
 */
export interface EyeOptions {
  sceneBuffer: GPUBuffer;
  shader: string;
}

export interface GpuBrainOptions {
  model?: Partial<ModelParams>;
  /** cap on integration steps per step() call (64 steps = 32 ms of brain time) */
  maxStepsPerCall?: number;
  /** spike list capacity per 0.5 ms step */
  maxSpikesPerStep?: number;
  /** replaces the CPU-fed retina pass with a scene-sampling compute eye */
  eye?: EyeOptions;
}

type Slot = "u" | "ud" | "r" | "rw";

const align4 = (x: number) => Math.max(8, Math.ceil(x / 4) * 4);

/**
 * Workgroups in each scatter dispatch (scatter.wgsl: one spike per workgroup at a time, taken in turn). A step has a
 * few dozen spikes, a few hundred in a burst; more workgroups than spikes just return.
 */
const SCATTER_GROUPS = 256;
/** couple.wgsl: longer graded-input rows than this are summed by a workgroup, not a thread */
const LONG_ROW = 128;
/** rows per slice in slicedEll(); graded.wgsl and couple.wgsl step through a row's entries by it */
const ELL_SLICE = 64;
/** slicedEll(): a row left out (longer than its maxLen) */
const ELL_SKIP = 0xffffffff;

/**
 * A CSR matrix of {col: u32, val: f32} entries re-laid for one thread per row: rows in slices of ELL_SLICE, each
 * slice padded to its longest row and its entries interleaved, so entry k of the slice's row j sits at
 * start + k * ELL_SLICE. Neighbouring threads then read neighbouring entries (in CSR each read a cache line of its
 * own), and every row still adds its entries up in the same order, so the sums are bit for bit the same. `rows` holds
 * (start, length) per row; a row longer than `maxLen` gets length ELL_SKIP and nothing in the slice.
 */
function slicedEll(offsets: Uint32Array, entries: ArrayBuffer, maxLen = Infinity): { rows: Uint32Array; entries: Uint32Array } {
  const nRows = offsets.length - 1;
  const rows = new Uint32Array(2 * nRows);
  let total = 0;
  for (let s = 0; s < nRows; s += ELL_SLICE) {
    const end = Math.min(s + ELL_SLICE, nRows);
    let width = 0;
    for (let i = s; i < end; i++) {
      const len = offsets[i + 1] - offsets[i];
      rows[2 * i] = total + (i - s);
      rows[2 * i + 1] = len <= maxLen ? len : ELL_SKIP;
      if (len <= maxLen) width = Math.max(width, len);
    }
    total += width * ELL_SLICE;
  }
  const src = new Uint32Array(entries);
  const out = new Uint32Array(2 * Math.max(1, total));
  for (let i = 0; i < nRows; i++) {
    const len = rows[2 * i + 1];
    if (len === ELL_SKIP) continue;
    for (let k = 0; k < len; k++) {
      const from = 2 * (offsets[i] + k);
      const to = 2 * (rows[2 * i] + k * ELL_SLICE);
      out[to] = src[from];
      out[to + 1] = src[from + 1];
    }
  }
  return { rows, entries: out };
}

function infoOf(d: BrainData): ConnectomeInfo {
  return {
    meta: d.meta, types: d.types, n: d.n, ng: d.ng, typeId: d.typeId, side: d.side, superclass: d.superclass, nt: d.nt,
    pos: d.pos, rf: d.rf, visUnits: d.visUnits, visCount: d.visCount, stats: d.stats, plastic: d.plastic, extracted: d.extracted,
  };
}

/** The whole connectome in WebGPU compute shaders. */
export class GpuBrain implements FlyBrain {
  readonly backend = "webgpu";
  readonly ready = true;
  readonly info: ConnectomeInfo;
  readonly n: number;
  readonly ng: number;
  /** GPU state, used by BrainView */
  readonly neurons: GPUBuffer;
  readonly gradedA: GPUBuffer;
  readonly gradedB: GPUBuffer;
  readonly units: GPUBuffer;
  /** which of gradedA/gradedB holds the latest graded activity */
  gradedCurrent: 0 | 1 = 0;
  brainTime = 0;
  onReadback: ((r: Readback) => void) | null = null;

  private _model: ModelParams;
  private params: GPUBuffer;
  private paramsData = new ArrayBuffer(80);
  private stepBuf: GPUBuffer;
  private stepAlign: number;
  private baseDrive: GPUBuffer;
  private spikeCount: GPUBuffer;
  private ring: GPUBuffer;
  private contrast: GPUBuffer;
  private contrastData: Float32Array;
  private contrastDirty = false;
  private plasticEdges: GPUBuffer;
  private plasticCount: number;
  /** spiking cells whose graded input rows are longer than LONG_ROW (couple.wgsl longRows) */
  private longRows: number;
  private drive: DriveTable;
  private maxSteps: number;
  private maxSpikes: number;
  private ringSlots: number;
  private delaySteps: number;
  private t = 0;
  private nextGraded = 0;
  private visCount: number;
  private eye: EyeOptions | null;
  private buffers: GPUBuffer[] = [];
  private pipes: {
    neuron: GPUComputePipeline;
    scatter: GPUComputePipeline;
    graded: GPUComputePipeline;
    couple: GPUComputePipeline;
    coupleLong: GPUComputePipeline;
    probe: GPUComputePipeline;
    /** only one of these two paths exists: the CPU-fed retina, or the scene-sampling eye */
    retina?: GPUComputePipeline;
    eyeMean?: GPUComputePipeline;
    eyeContrast?: GPUComputePipeline;
  };
  private groups: {
    neuron: GPUBindGroup;
    scatter: GPUBindGroup;
    graded: [GPUBindGroup, GPUBindGroup];
    couple: [GPUBindGroup, GPUBindGroup];
    retina?: GPUBindGroup;
    eye?: GPUBindGroup;
  };
  private probeLayout: GPUBindGroupLayout;
  private probe: { idx: GPUBuffer; out: GPUBuffer; group: GPUBindGroup; count: number; stages: GPUBuffer[]; gen: number } | null = null;
  private probeGen = 0;
  private pending: (() => void) | null = null;

  constructor(
    readonly device: GPUDevice,
    d: BrainData,
    opts: GpuBrainOptions = {},
  ) {
    this.info = infoOf(d);
    this._model = { ...DEFAULT_MODEL, ...opts.model };
    this.n = d.n;
    this.ng = d.ng;
    this.visCount = d.visCount;
    this.maxSteps = opts.maxStepsPerCall ?? 64;
    this.maxSpikes = opts.maxSpikesPerStep ?? 8192;
    this.eye = opts.eye ?? null;
    this.drive = new DriveTable(d.n);
    const S = GPUBufferUsage.STORAGE;
    const CD = GPUBufferUsage.COPY_DST;
    const CS = GPUBufferUsage.COPY_SRC;
    const make = (size: number, usage: number, data?: ArrayBufferView | ArrayBuffer, label?: string) => this.make(size, usage, data, label);

    const m = this._model;
    this.delaySteps = Math.max(1, Math.round(m.tDelay / m.dt));
    this.ringSlots = this.delaySteps + 1;
    this.stepAlign = device.limits.minUniformBufferOffsetAlignment;

    this.params = make(80, GPUBufferUsage.UNIFORM | CD, undefined, "params");
    this.stepBuf = make(this.stepAlign * this.maxSteps, GPUBufferUsage.UNIFORM | CD, undefined, "steps");
    this.neurons = make(24 * d.n, S | CD | CS, undefined, "neurons");
    this.baseDrive = make(4 * d.n, S | CD, undefined, "baseDrive");
    const gDrive = make(4 * d.n, S, undefined, "gDrive");
    this.ring = make(4 * this.ringSlots * d.n, S | CD, undefined, "ring");
    const spikes = make(4 * this.maxSpikes, S, undefined, "spikes");
    this.spikeCount = make(4 * this.maxSteps, S | CD | CS, undefined, "spikeCount");
    const spikeOffsets = make(d.spikeOffsets.byteLength, S, d.spikeOffsets, "spikeOffsets");
    const spikeEdges = make(d.spikeEdges.byteLength, S, d.spikeEdges, "spikeEdges");
    const plasticOffsets = make(4 * (d.n + 1), S, d.plastic?.offsets, "plasticOffsets");
    this.plasticCount = d.plastic?.packed.length ?? 0;
    this.plasticEdges = make(4 * this.plasticCount, S | CD, d.plastic?.packed, "plasticEdges");
    // the graded optic lobe and the short rows of the coupling as sliced ELL (slicedEll); the long coupling rows stay
    // CSR, for couple.wgsl longRows
    const gEll = slicedEll(d.gradedOffsets, d.gradedEntries);
    const gRows = make(gEll.rows.byteLength, S, gEll.rows, "gradedRows");
    const gEntries = make(gEll.entries.byteLength, S, gEll.entries, "gradedEntries");
    const iEll = slicedEll(d.ifaceOffsets, d.ifaceEntries, LONG_ROW);
    const iRows = make(iEll.rows.byteLength, S, iEll.rows, "ifaceRows");
    const iEllEntries = make(iEll.entries.byteLength, S, iEll.entries, "ifaceEllEntries");
    const iOffsets = make(d.ifaceOffsets.byteLength, S, d.ifaceOffsets, "iOffsets");
    const iEntries = make(d.ifaceEntries.byteLength, S, d.ifaceEntries, "iEntries");
    this.gradedA = make(4 * d.ng, S | CD | CS, undefined, "gradedA");
    this.gradedB = make(4 * d.ng, S | CD | CS, undefined, "gradedB");
    const ext = make(4 * d.ng, S, undefined, "ext");
    const modeData = new Uint32Array(d.ng);
    const unitU = new Uint32Array(d.visUnits);
    for (let u = 0; u < d.visCount; u++) modeData[unitU[8 * u + 4]] = unitU[8 * u + 6];
    const mode = make(4 * d.ng, S, modeData, "mode");
    this.units = make(d.visUnits.byteLength, S, d.visUnits, "visUnits");
    this.contrastData = new Float32Array(d.visCount);
    this.contrast = make(4 * d.visCount, S | CD, undefined, "contrast");
    // Scene-sampling eye: its own uniform (unit count, half acceptance angle) and per-eye mean luminance.
    const eyeData = new ArrayBuffer(16);
    new Uint32Array(eyeData)[0] = d.visCount;
    new Float32Array(eyeData)[1] = ACCEPTANCE;
    const eyeParams = make(16, GPUBufferUsage.UNIFORM, eyeData, "eyeParams");
    const eyeMeans = make(16, S, new Float32Array([1, 1, 1, 1]), "eyeMeans");

    const layout = (slots: Slot[]) =>
      device.createBindGroupLayout({
        entries: slots.map((s, i): GPUBindGroupLayoutEntry => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: s === "u" || s === "ud" ? "uniform" : s === "r" ? "read-only-storage" : "storage", hasDynamicOffset: s === "ud" },
        })),
      });
    const pipeline = (code: string, slots: Slot[]) => {
      const bgl = layout(slots);
      const p = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
      return { p, bgl };
    };
    const group = (bgl: GPUBindGroupLayout, res: (GPUBuffer | [GPUBuffer, number])[]) =>
      device.createBindGroup({
        layout: bgl,
        entries: res.map((b, i) => (Array.isArray(b) ? { binding: i, resource: { buffer: b[0], size: b[1] } } : { binding: i, resource: { buffer: b } })),
      });

    const neuron = pipeline(commonSrc + neuronSrc, ["u", "ud", "rw", "r", "r", "rw", "rw", "rw"]);
    const scatter = pipeline(commonSrc + scatterSrc, ["u", "ud", "r", "rw", "r", "r", "rw", "r", "r"]);
    const ell = `const ELL_SLICE = ${ELL_SLICE}u;\nconst ELL_SKIP = ${ELL_SKIP}u;\n`;
    const graded = pipeline(ell + commonSrc + gradedSrc, ["u", "r", "r", "r", "rw", "r", "r"]);
    // couple.wgsl: rows longer than LONG_ROW (a few percent of them, up to thousands of entries) get a workgroup each
    const coupleSrcFull = ell + commonSrc + coupleSrc;
    const couple = pipeline(coupleSrcFull, ["u", "r", "r", "r", "rw", "r", "r", "r"]);
    const coupleLong = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [couple.bgl] }),
      compute: { module: device.createShaderModule({ code: coupleSrcFull }), entryPoint: "longRows" },
    });
    const io = d.ifaceOffsets;
    const longList: number[] = [];
    for (let r = 0; r + 1 < io.length; r++) if (io[r + 1] - io[r] > LONG_ROW) longList.push(r);
    this.longRows = longList.length;
    // bound at its exact size: the shader loops to arrayLength (the buffer itself is at least 8 bytes)
    const longIdx: [GPUBuffer, number] = [make(4 * longList.length, S, new Uint32Array(longList), "coupleLongRows"), 4 * Math.max(1, longList.length)];
    const probe = pipeline(commonSrc + probeSrc, ["r", "r", "rw"]);

    // Either the CPU-fed retina pass or the scene-sampling eye, never both. Both write the same `ext`
    // buffer, so nothing downstream changes. The two eye entry points share one layout and bind group.
    let retina: { p: GPUComputePipeline; bgl: GPUBindGroupLayout } | null = null;
    let eyeMean: GPUComputePipeline | undefined;
    let eyeContrast: GPUComputePipeline | undefined;
    let eyeGroup: GPUBindGroup | undefined;
    if (this.eye) {
      const bgl = layout(["u", "u", "r", "rw", "rw"]);
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const module = device.createShaderModule({ code: this.eye.shader, label: "eye" });
      eyeMean = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: "meanPass" } });
      eyeContrast = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: "contrastPass" } });
      eyeGroup = group(bgl, [this.eye.sceneBuffer, eyeParams, this.units, ext, eyeMeans]);
    } else {
      retina = pipeline(retinaSrc, ["r", "r", "rw"]);
    }

    this.probeLayout = probe.bgl;
    this.pipes = {
      neuron: neuron.p, scatter: scatter.p, graded: graded.p, couple: couple.p, coupleLong, probe: probe.p,
      retina: retina?.p, eyeMean, eyeContrast,
    };
    const step: [GPUBuffer, number] = [this.stepBuf, 16];
    this.groups = {
      neuron: group(neuron.bgl, [this.params, step, this.neurons, this.baseDrive, gDrive, this.ring, spikes, this.spikeCount]),
      scatter: group(scatter.bgl, [this.params, step, spikes, this.spikeCount, spikeOffsets, spikeEdges, this.ring, plasticOffsets, this.plasticEdges]),
      graded: [
        group(graded.bgl, [this.params, gRows, gEntries, this.gradedA, this.gradedB, ext, mode]),
        group(graded.bgl, [this.params, gRows, gEntries, this.gradedB, this.gradedA, ext, mode]),
      ],
      couple: [
        group(couple.bgl, [this.params, iOffsets, iEntries, this.gradedA, gDrive, longIdx, iRows, iEllEntries]),
        group(couple.bgl, [this.params, iOffsets, iEntries, this.gradedB, gDrive, longIdx, iRows, iEllEntries]),
      ],
      retina: retina ? group(retina.bgl, [this.units, this.contrast, ext]) : undefined,
      eye: eyeGroup,
    };
    this.writeParams();
    this.setProbes(new Uint32Array(0));
  }

  private make(size: number, usage: number, data?: ArrayBufferView | ArrayBuffer, label?: string): GPUBuffer {
    const b = this.device.createBuffer({ size: align4(size), usage, mappedAtCreation: !!data, label });
    if (data) {
      const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      new Uint8Array(b.getMappedRange()).set(src);
      b.unmap();
    }
    this.buffers.push(b);
    return b;
  }

  get model(): Readonly<ModelParams> {
    return this._model;
  }

  setModel(params: Partial<ModelParams>) {
    const { dt: _dt, tDelay: _tDelay, ...rest } = params;
    Object.assign(this._model, rest);
    this.writeParams();
  }

  private writeParams() {
    const m = this._model;
    const u = new Uint32Array(this.paramsData);
    const f = new Float32Array(this.paramsData);
    u[0] = this.n;
    u[1] = this.ng;
    u[2] = this.ringSlots;
    u[3] = this.maxSpikes;
    const em = Math.exp(-m.dt / m.tauM);
    const es = Math.exp(-m.dt / m.tauS);
    f[4] = em;
    f[5] = es;
    f[6] = (m.tauS / (m.tauS - m.tauM)) * (es - em);
    f[7] = Math.exp(-m.dt / m.tauAdapt);
    f[8] = m.vThreshold;
    f[9] = m.vReset;
    f[10] = m.wSyn;
    f[11] = m.adapt;
    f[12] = m.tRefractory;
    f[13] = m.dt;
    f[14] = Math.exp(-m.dt / m.actTau);
    f[15] = m.coupling;
    f[16] = m.gradedGain;
    f[17] = Math.exp(-m.gradedDt / m.gradedTau);
    f[18] = m.gradedMin;
    f[19] = m.gradedMax;
    this.device.queue.writeBuffer(this.params, 0, this.paramsData);
  }

  /** Ignored while a scene-sampling eye is installed: it fills `ext` itself, straight from the scene buffer. */
  setVision(contrast: Float32Array) {
    if (this.eye) return;
    if (contrast.length !== this.visCount) throw new Error(`setVision: expected ${this.visCount} values, got ${contrast.length}`);
    this.contrastData.set(contrast);
    this.contrastDirty = true;
  }

  setDrive(neurons: ArrayLike<number>, mV: number) {
    this.drive.setTonic(neurons, mV);
  }

  pulse(neurons: ArrayLike<number>, mV: number, durationMs: number) {
    this.drive.pulse(neurons, mV, durationMs, this.brainTime);
  }

  setProbes(neurons: Uint32Array) {
    const old = this.probe;
    if (old) {
      old.idx.destroy();
      old.out.destroy();
      for (const s of old.stages) s.destroy();
    }
    const count = neurons.length;
    const idx = this.device.createBuffer({ size: align4(neurons.byteLength), usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Uint32Array(idx.getMappedRange()).set(neurons);
    idx.unmap();
    const out = this.device.createBuffer({ size: align4(4 * count), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const stages = [0, 1, 2].map(() =>
      this.device.createBuffer({ size: align4(4 * (count + this.maxSteps)), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    );
    const group = this.device.createBindGroup({
      layout: this.probeLayout,
      entries: [
        { binding: 0, resource: { buffer: idx } },
        { binding: 1, resource: { buffer: this.neurons } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    this.probe = { idx, out, group, count, stages, gen: ++this.probeGen };
  }

  setPlasticWeights(packed: Uint32Array) {
    if (packed.length !== this.plasticCount) throw new Error(`setPlasticWeights: expected ${this.plasticCount} edges, got ${packed.length}`);
    if (this.plasticCount) this.device.queue.writeBuffer(this.plasticEdges, 0, packed);
  }

  /** Clears spiking state (voltages, synaptic input, adaptation); optionally the graded optic lobe too. */
  reset(includeGraded = true) {
    const q = this.device.queue;
    q.writeBuffer(this.neurons, 0, new Uint8Array(24 * this.n));
    q.writeBuffer(this.ring, 0, new Uint8Array(4 * this.ringSlots * this.n));
    if (includeGraded) {
      q.writeBuffer(this.gradedA, 0, new Float32Array(this.ng));
      q.writeBuffer(this.gradedB, 0, new Float32Array(this.ng));
    }
  }

  step(ms: number): boolean {
    const encoder = this.device.createCommandEncoder({ label: "flybrain" });
    this.encode(encoder, ms);
    this.device.queue.submit([encoder.finish()]);
    this.afterSubmit();
    return true;
  }

  /**
   * Encodes `brainMs` of simulation into your own command encoder (to share a submit with rendering).
   * Call afterSubmit() right after queue.submit(). Returns the number of 0.5 ms steps encoded.
   */
  encode(encoder: GPUCommandEncoder, brainMs: number): number {
    const m = this._model;
    const q = this.device.queue;
    const steps = Math.min(this.maxSteps, Math.max(0, Math.round(brainMs / m.dt)));
    this.drive.advance(this.brainTime);
    const dirty = this.drive.takeDirty();
    if (dirty) q.writeBuffer(this.baseDrive, 4 * dirty[0], this.drive.values, dirty[0], dirty[1] - dirty[0]);
    if (this.contrastDirty) {
      q.writeBuffer(this.contrast, 0, this.contrastData);
      this.contrastDirty = false;
    }
    const stepData = new ArrayBuffer(this.stepAlign * Math.max(1, steps));
    const sv = new Uint32Array(stepData);
    for (let s = 0; s < steps; s++) {
      const o = (this.stepAlign / 4) * s;
      const t = this.t + s;
      sv[o] = s;
      sv[o + 1] = t % this.ringSlots;
      sv[o + 2] = (t + this.delaySteps) % this.ringSlots;
    }
    if (steps > 0) q.writeBuffer(this.stepBuf, 0, stepData);
    q.writeBuffer(this.spikeCount, 0, new Uint32Array(this.maxSteps));

    const pass = encoder.beginComputePass({ label: "brain" });
    if (this.pipes.eyeMean && this.pipes.eyeContrast && this.groups.eye) {
      // One mean-luminance pass per eye, then contrast for every visual unit, straight from the scene.
      pass.setPipeline(this.pipes.eyeMean);
      pass.setBindGroup(0, this.groups.eye);
      pass.dispatchWorkgroups(2);
      pass.setPipeline(this.pipes.eyeContrast);
      pass.setBindGroup(0, this.groups.eye);
      pass.dispatchWorkgroups(Math.ceil(this.visCount / 128));
    } else if (this.pipes.retina && this.groups.retina) {
      pass.setPipeline(this.pipes.retina);
      pass.setBindGroup(0, this.groups.retina);
      pass.dispatchWorkgroups(Math.ceil(this.visCount / 128));
    }
    const ns = this.n - this.ng;
    for (let s = 0; s < steps; s++) {
      if (this.brainTime >= this.nextGraded) {
        pass.setPipeline(this.pipes.graded);
        pass.setBindGroup(0, this.groups.graded[this.gradedCurrent]);
        pass.dispatchWorkgroups(Math.ceil(this.ng / 256));
        this.gradedCurrent = this.gradedCurrent === 0 ? 1 : 0;
        pass.setPipeline(this.pipes.couple);
        pass.setBindGroup(0, this.groups.couple[this.gradedCurrent]);
        pass.dispatchWorkgroups(Math.ceil(ns / 256));
        if (this.longRows > 0) {
          pass.setPipeline(this.pipes.coupleLong);
          pass.dispatchWorkgroups(Math.min(this.longRows, 65535));
        }
        this.nextGraded += m.gradedDt;
      }
      const off = [this.stepAlign * s];
      pass.setPipeline(this.pipes.neuron);
      pass.setBindGroup(0, this.groups.neuron, off);
      pass.dispatchWorkgroups(Math.ceil(ns / 256));
      pass.setPipeline(this.pipes.scatter);
      pass.setBindGroup(0, this.groups.scatter, off);
      pass.dispatchWorkgroups(SCATTER_GROUPS);
      this.brainTime += m.dt;
    }
    this.t += steps;
    const probe = this.probe!;
    if (probe.count > 0) {
      pass.setPipeline(this.pipes.probe);
      pass.setBindGroup(0, probe.group);
      pass.dispatchWorkgroups(Math.ceil(probe.count / 64));
    }
    pass.end();

    const stage = probe.stages.pop();
    if (stage) {
      if (probe.count > 0) encoder.copyBufferToBuffer(probe.out, 0, stage, 0, 4 * probe.count);
      encoder.copyBufferToBuffer(this.spikeCount, 0, stage, 4 * probe.count, 4 * this.maxSteps);
      const brainTime = this.brainTime;
      const frameMs = steps * m.dt;
      const gen = probe.gen;
      this.pending = () => {
        stage
          .mapAsync(GPUMapMode.READ)
          .then(() => {
            const all = new Uint32Array(stage.getMappedRange().slice(0));
            stage.unmap();
            if (gen !== this.probeGen) {
              stage.destroy(); // probes changed while this readback was in flight
              return;
            }
            probe.stages.push(stage);
            let spikes = 0;
            for (let s = 0; s < steps; s++) spikes += all[probe.count + s];
            this.onReadback?.({ counts: all.subarray(0, probe.count), brainTime, spikes, frameMs });
          })
          .catch(() => stage.destroy());
      };
    }
    return steps;
  }

  /** Call right after queue.submit() of the encoder passed to encode(). */
  afterSubmit() {
    const p = this.pending;
    this.pending = null;
    p?.();
  }

  async readActivity(): Promise<Activity> {
    const d = this.device;
    const s1 = d.createBuffer({ size: 24 * this.n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const s2 = d.createBuffer({ size: align4(4 * this.ng), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.neurons, 0, s1, 0, 24 * this.n);
    enc.copyBufferToBuffer(this.currentGraded(), 0, s2, 0, 4 * this.ng);
    d.queue.submit([enc.finish()]);
    await Promise.all([s1.mapAsync(GPUMapMode.READ), s2.mapAsync(GPUMapMode.READ)]);
    const f = new Float32Array(s1.getMappedRange());
    const act = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) act[i] = f[6 * i + 4];
    const graded = new Float32Array(s2.getMappedRange().slice(0, 4 * this.ng));
    s1.destroy();
    s2.destroy();
    return { act, graded };
  }

  currentGraded(): GPUBuffer {
    return this.gradedCurrent === 0 ? this.gradedA : this.gradedB;
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    if (this.probe) {
      this.probe.idx.destroy();
      this.probe.out.destroy();
      for (const s of this.probe.stages) s.destroy();
    }
  }
}
