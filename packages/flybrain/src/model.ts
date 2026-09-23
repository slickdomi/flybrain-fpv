// Model constants. They come from fly-thing (Aimbug): validated there against a NumPy reference
// (pipeline/sim_hybrid.py) and tuned in closed-loop GPU trials of the aim trainer.

/** Parameters of the running model. `dt` and `tDelay` are fixed once a brain is created. */
export interface ModelParams {
  /** ms, spiking integration step */
  dt: number;
  /** mV above rest (-45 vs -52 mV) */
  vThreshold: number;
  vReset: number;
  /** ms, membrane time constant */
  tauM: number;
  /** ms, synaptic time constant */
  tauS: number;
  /** ms */
  tRefractory: number;
  /** ms, synaptic delay */
  tDelay: number;
  /** mV per synapse (Shiu et al. 2024) */
  wSyn: number;
  /**
   * mV of threshold increase per spike. Without spike-frequency adaptation recurrent cholinergic cliques
   * (e.g. the lLN1_bc antennal-lobe local neurons) lock at maximum rate and the brain never goes quiet.
   */
  adapt: number;
  /** ms */
  tauAdapt: number;
  /** ms, step of the graded (non-spiking) optic lobe */
  gradedDt: number;
  /** ms */
  gradedTau: number;
  /**
   * Recurrent gain of the graded optic lobe. Lamina feedback loops (L1/L2/C2/C3/T1) have eigenvalues ~0.42,
   * so gains above ~2.39 make them unstable.
   */
  gradedGain: number;
  gradedMin: number;
  gradedMax: number;
  /** graded -> spiking release scale (Hz-equivalent per unit of graded activity) */
  coupling: number;
  /** ms, decay of the per-neuron spike trace used by the visualisations */
  actTau: number;
}

export const DEFAULT_MODEL: Readonly<ModelParams> = Object.freeze({
  dt: 0.5,
  vThreshold: 7,
  vReset: 0,
  tauM: 20,
  tauS: 5,
  tRefractory: 2.2,
  tDelay: 1.8,
  wSyn: 0.275,
  adapt: 2.0,
  tauAdapt: 200,
  gradedDt: 8.33,
  gradedTau: 20,
  gradedGain: 2.3,
  gradedMin: -1,
  gradedMax: 4,
  coupling: 3400,
  actTau: 120,
});

/** Constants used while the connectome is split into matrices (changing them needs a reload). */
export const DEFAULT_LOAD = Object.freeze({
  /** drop graded edges below this input fraction (keeps 3.9 M of 8.9 M, no change in behaviour) */
  prune: 0.005,
  /** synthetic R1-6 input to L1-3 (the lamina lies mostly outside the reconstruction) */
  laminaWeight: 0.5,
});

/**
 * Synaptic input is accumulated in fixed point: FIXED_POINT units per synapse. Static synapses are whole
 * synapse counts; plastic synapses can take any multiple of 1/FIXED_POINT.
 */
export const FIXED_POINT = 16;
