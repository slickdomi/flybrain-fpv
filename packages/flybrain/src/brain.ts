import type { ConnectomeInfo } from "./data/types";
import type { ModelParams } from "./model";

export interface Readback {
  /** cumulative spike counts of the probe neurons (restart from zero after a reset) */
  counts: Uint32Array;
  /** ms of simulated time when the counts were copied */
  brainTime: number;
  /** spikes emitted by the whole brain during the chunk that produced this readback */
  spikes: number;
  /** brain ms simulated in that chunk */
  frameMs: number;
}

export interface Activity {
  /** per-neuron spike trace (decays with actTau; ~ Hz * actTau / 1000) */
  act: Float32Array;
  /** graded optic-lobe activity, neurons [0, ng) */
  graded: Float32Array;
}

/** What both backends (WebGPU and the CPU worker) offer. */
export interface FlyBrain {
  readonly backend: "webgpu" | "cpu";
  readonly info: ConnectomeInfo;
  /** ms of simulated time */
  readonly brainTime: number;
  /** false while the CPU worker is still busy with the previous chunk; WebGPU is always ready */
  readonly ready: boolean;
  readonly model: Readonly<ModelParams>;
  /** Called once per simulated chunk with the probe counters. */
  onReadback: ((r: Readback) => void) | null;

  /** Changes model parameters (except dt and tDelay). */
  setModel(params: Partial<ModelParams>): void;
  /** Per visual unit contrast, as computed by Retina.normalize(). Stays until the next call. */
  setVision(contrast: Float32Array): void;
  /** Constant input (mV above rest) to these neurons, replacing their previous constant input. */
  setDrive(neurons: ArrayLike<number>, mV: number): void;
  /** Extra input (mV) to these neurons for `durationMs` of brain time, on top of any constant drive. */
  pulse(neurons: ArrayLike<number>, mV: number, durationMs: number): void;
  /** Which neurons' spike counters come back in each Readback. */
  setProbes(neurons: Uint32Array): void;
  /** New packed weights for info.plastic (same order and length as info.plastic.packed). */
  setPlasticWeights(packed: Uint32Array): void;
  /** Simulates `ms` of brain time (capped per call). Returns false if the backend was not ready. */
  step(ms: number): boolean;
  /** Clears spiking state (voltages, synaptic input, adaptation) and optionally the graded optic lobe. */
  reset(includeGraded?: boolean): void;
  /** Copies every neuron's activity back (slow on WebGPU: for analysis and the 2D views). */
  readActivity(): Promise<Activity>;
  destroy(): void;
}
