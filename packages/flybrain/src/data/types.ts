export interface Shard {
  file: string;
  rowStart: number;
  rowEnd: number;
  edges: number;
  bytes: number;
}

export interface Meta {
  dataset: string;
  source: string;
  license: string;
  neurons: number;
  edges: number;
  synapses: number;
  superclasses: string[];
  transmitters: string[];
  shards: Shard[];
  graded: { superclasses: string[]; count: number };
  eye: { count: number };
}

/** Selects connectome edges by regular expressions over the pre- and postsynaptic cell type names. */
export interface EdgeQuery {
  name: string;
  /** e.g. "^PAM" */
  pre: string;
  /** e.g. "^MBON" */
  post: string;
}

/** Raw connectome edges picked out by an EdgeQuery (including dopaminergic ones, which the model does not run). */
export interface ExtractedEdges {
  pre: Uint32Array;
  post: Uint32Array;
  /** synapse count */
  count: Uint16Array;
  /** sign of the presynaptic transmitter (+1 / -1) */
  sign: Int8Array;
}

/** Spiking synapses whose weight can change while the brain runs (see DopamineLearning). */
export interface PlasticSynapses {
  /** n + 1 offsets into `packed`, by presynaptic neuron */
  offsets: Uint32Array;
  /** target neuron (low 18 bits) | signed weight in FIXED_POINT units per synapse (high 14 bits) */
  packed: Uint32Array;
  /** presynaptic neuron of each edge */
  pre: Uint32Array;
  /** signed synapse count of each edge: the connectome weight, i.e. relative weight 1 */
  base: Int16Array;
}

export interface LoadOptions {
  /** directory with meta.json, types.json, neurons.bin.gz and the edge shards */
  url: string;
  prune?: number;
  laminaWeight?: number;
  /** Spiking edges between these cell types become plastic, e.g. { pre: "^KC", post: "^MBON" }. */
  plastic?: { pre: string; post: string } | null;
  /** Extra edge lists to copy out of the raw connectome. */
  extract?: EdgeQuery[];
  /**
   * Spiking edges between these cell types get no fast synaptic effect, like the dopamine, serotonin and
   * octopamine synapses already have none. For connections known to act through a GPCR although their
   * transmitter is fast elsewhere, e.g. { pre: "^KC", post: "^KC" }: Kenyon cell axo-axonic synapses are
   * cholinergic but act through inhibitory muscarinic mAChR-B (Manoim et al. 2022).
   */
  silence?: { pre: string; post: string }[];
  /**
   * Spiking edges between these cell types count `factor` times their synapses. A declared departure from
   * the uniform per-synapse weight, for a pathway measured to be present but too weak to fire its targets
   * at that weight -- e.g. visual projection neurons onto the visual Kenyon cells. The first match wins.
   */
  gain?: { pre: string; post: string; factor: number }[];
}

export interface ResolvedLoadOptions {
  url: string;
  prune: number;
  laminaWeight: number;
  wSyn: number;
  tauS: number;
  plastic: { pre: string; post: string } | null;
  extract: EdgeQuery[];
  silence?: { pre: string; post: string }[];
  gain?: { pre: string; post: string; factor: number }[];
}

/** Everything about the connectome except the large matrices. */
export interface ConnectomeInfo {
  meta: Meta;
  types: string[];
  n: number;
  /** neurons [0, ng) are graded optic-lobe units, [ng, n) spike */
  ng: number;
  typeId: Uint16Array;
  /** 0 unknown, 1 left, 2 right */
  side: Uint8Array;
  superclass: Uint8Array;
  /** index into meta.transmitters */
  nt: Uint8Array;
  /** soma xyz in micrometres, NaN if unknown */
  pos: Float32Array;
  /** connectome receptive field per neuron: azimuth, elevation (deg), strength (0 = not visual) */
  rf: Float32Array;
  /**
   * Visual units, interleaved {dx, dy, dz, weight: f32; graded index, side, mode, pad: u32}.
   * Direction in the fly frame (+x right, +y up, -z forward). Mode 1 = photoreceptor (activity clamped to
   * contrast), 2 = lamina column (adds weight * contrast).
   */
  visUnits: ArrayBuffer;
  visCount: number;
  stats: { spikeEdges: number; gradedEdges: number; ifaceEdges: number; plasticEdges: number };
  plastic: PlasticSynapses | null;
  extracted: Record<string, ExtractedEdges>;
}

/** Everything a brain backend needs. */
export interface BrainData extends ConnectomeInfo {
  /** spiking -> spiking CSR by presynaptic neuron; packed = target (18 bits) | signed synapse count (14 bits) */
  spikeOffsets: Uint32Array;
  spikeEdges: Uint32Array;
  /** graded -> graded CSR by postsynaptic graded neuron: interleaved {u32 col, f32 signed input fraction} */
  gradedOffsets: Uint32Array;
  gradedEntries: ArrayBuffer;
  /** graded -> spiking CSR by postsynaptic spiking neuron (row r = neuron ng + r): {u32 graded col, f32 mV per unit} */
  ifaceOffsets: Uint32Array;
  ifaceEntries: ArrayBuffer;
}

export type LoaderOut =
  | { kind: "progress"; label: string; done: number; total: number }
  | { kind: "done"; data: BrainData }
  | { kind: "error"; message: string };

/** Transferable buffers of a BrainData (for postMessage). */
export function brainDataTransfer(d: BrainData): Transferable[] {
  const list: ArrayBufferLike[] = [
    d.typeId.buffer, d.side.buffer, d.superclass.buffer, d.nt.buffer, d.pos.buffer, d.rf.buffer, d.visUnits,
    d.spikeOffsets.buffer, d.spikeEdges.buffer, d.gradedOffsets.buffer, d.gradedEntries, d.ifaceOffsets.buffer, d.ifaceEntries,
  ];
  if (d.plastic) list.push(d.plastic.offsets.buffer, d.plastic.packed.buffer, d.plastic.pre.buffer, d.plastic.base.buffer);
  for (const e of Object.values(d.extracted)) list.push(e.pre.buffer, e.post.buffer, e.count.buffer, e.sign.buffer);
  return [...new Set(list)] as Transferable[];
}

/** A structured copy of the info part (the brain keeps the originals). */
export function copyInfo(d: ConnectomeInfo): ConnectomeInfo {
  const extracted: Record<string, ExtractedEdges> = {};
  for (const [k, e] of Object.entries(d.extracted)) extracted[k] = { pre: e.pre.slice(), post: e.post.slice(), count: e.count.slice(), sign: e.sign.slice() };
  return {
    meta: d.meta,
    types: d.types,
    n: d.n,
    ng: d.ng,
    typeId: d.typeId.slice(),
    side: d.side.slice(),
    superclass: d.superclass.slice(),
    nt: d.nt.slice(),
    pos: d.pos.slice(),
    rf: d.rf.slice(),
    visUnits: d.visUnits.slice(0),
    visCount: d.visCount,
    stats: d.stats,
    plastic: d.plastic
      ? { offsets: d.plastic.offsets.slice(), packed: d.plastic.packed.slice(), pre: d.plastic.pre.slice(), base: d.plastic.base.slice() }
      : null,
    extracted,
  };
}
