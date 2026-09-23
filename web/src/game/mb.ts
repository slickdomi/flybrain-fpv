// Copied unchanged from fly-addiction (web/src/game/mb.ts); its numbers were measured in that lab, not here.
//
// What the mushroom body has learned about what the fly is looking at right now, per hemisphere.
//
// REAL: the Kenyon cell spikes (the eye drives the visual KCs through aMe/MeVP/LoVP projection neurons), the
// KC->MBON synapse weights and how far dopamine has depressed each one (DopamineLearning), and which
// compartment every MBON sits in (dopamine cell -> MBON synapse counts from the connectome).
// MODELED: the summation below and what main.ts does with it. The MBONs' own spikes are not used, because
// they also carry large *innate* responses to a view (measured: a naive food view drives PAM-compartment
// MBONs hard), which nothing here is validated against. This reads only the learned part.
//
// For the Kenyon cells a view is driving now, it asks what fraction of their drive onto avoidance
// (PAM-compartment) MBONs reward dopamine has removed (Aso et al. 2014: depressing those tilts the fly toward
// approach). A fraction rather than an amount, so a small bright patch that drives few KCs can be as wanted
// as a big dark one that drives many -- the absolute amount scaled with salience and let the dark food patch
// soak up whatever the drug taught (experiments/visualkc.ts).
//
// The PPL1 (punishment) side reads the mirror of that: the fraction of the view's drive onto approach
// (PPL1-compartment) MBONs that dopamine has removed. Since 2026-09-20 it steers too (MB.avoidGain), and
// two things had to be true first, because PPL1 idles at ~5 Hz and jumps to 12-15 Hz for ANY near patch:
//   - its slow mean is subtracted before it steers (main.ts), as DNa02's and the value readout's are, so a
//     floor under every view cancels and only a difference between views turns the fly. Without that,
//     subtracting the raw fraction made every view read as aversive.
//   - it gets its own, larger floor (avoidFloor below). At the shared floor the fraction sat at 0.72-0.76
//     at rest with nowhere to go, so punishment moved the REWARD readout further than this one.
// Note the two channels are NOT independent: every MBON is reached by both dopamine populations and the
// compartment label is only which one has more synapses, so any dopamine pulse depresses both sets and
// raises both readouts. Measured: pairing a view with PPL1 alone raised the PAM-side fraction 0.137 -> 0.243
// (experiments/visualkc.ts, DAN=PPL1). What separates them is the DIFFERENCE, which is what steers.
//
// Left and right are the two hemispheres' KCs. Their visual input is ipsilateral enough to tell a patch on
// the left from one on the right (visualkc.ts: food at +25 degrees drives right visual KCs 9.9 Hz, left
// 2.7 Hz), which is what lets the difference steer, as bilateral familiarity comparison does in models of
// insect view-based navigation (Steinbeck et al. 2024, Adaptive Behavior).
//
// The two sides are NOT equal, and the difference is standing. The bare arena drives right visual KCs ~6-8x
// harder (0.94-0.98 against 0.12-0.17 Hz). The reconstruction offers two causes, not yet told apart: the left
// R1-R6 are traced thinner (1,112 against 2,265), and the right visual KCs get 1.24x the VPN synapses. Background
// depression piles up on the right, and `right` reads above `left` with nothing in view: 0.68 against 0.28-0.30
// in the rig, and higher at the end of every
// one of 28 CAFE runs (visualkc.ts LATERAL, .cache/lateral-0923/). main.ts high-passes right - left before it
// steers, which removes the standing part; anything reading the difference unadapted sees that, not a view.

import type { ConnectomeInfo, DopamineLearning, Probes, Rates } from "flybrain";

export class LearnedValue {
  /** per hemisphere, fraction of the view's avoidance (PAM-compartment) drive that reward has removed, 0..1 */
  left = 0;
  right = 0;
  /** the same for the approach (PPL1-compartment) drive: learned aversion. Steers via MB.avoidGain. */
  aversionLeft = 0;
  aversionRight = 0;
  /** KC activity trace, Hz, in learning.pre order */
  private trace: Float32Array;
  /** per plastic edge: +1 PAM compartment, -1 PPL1 compartment, 0 neither */
  private comp: Int8Array;
  private absBase: Float32Array;
  /** per presynaptic KC: 1 left, 2 right */
  private kcSide: Uint8Array;

  constructor(
    info: ConnectomeInfo,
    private learning: DopamineLearning,
    private probes: Probes,
    dopamineEdges: string,
    /** synapse x Hz added to each compartment's drive, so a view that drives almost nothing reads as ~0 */
    public floor: number,
    /**
     * The same, for the PPL1 (aversion) side, which needs a LARGER one. Measured 2026-09-20
     * (experiments/visualkc.ts, DAN=PPL1): at the shared floor the PPL1-side fraction sits at 0.798 before
     * any punishment is delivered and saturates at 0.889 after six pairings, while the PAM side has room to
     * move 0.137 -> 0.243. So punishment moved the REWARD readout further than the aversion one and read as
     * approach. PPL1 idles at ~5 Hz and jumps to 12-15 Hz for any near patch, so its compartments are
     * depressed by everything; a bigger floor is what buys the channel back its dynamic range.
     *
     * The criterion is maximum headroom in both directions -- resting fraction near 0.5 -- NOT whichever
     * value makes quinine work. MEASURED by it (?avoidfloor= sweep, 150 s lab runs, seed 2, 2026-09-20):
     * 0.72-0.76 at the old shared floor of 4000, 0.48-0.56 at 20000, 0.39-0.46 at 40000. Hence 20000.
     * Re-run the sweep (scripts/overnight.sh phase 0) after any change to the readout.
     */
    public avoidFloor: number,
    /** ms, how fast the KC trace follows the view */
    public tauMs: number,
  ) {
    const plastic = info.plastic;
    const edges = info.extracted[dopamineEdges];
    if (!plastic || !edges) throw new Error("LearnedValue: load the brain with plastic KC->MBON and the dopamine edges");
    const pam = new Float64Array(info.n);
    const ppl = new Float64Array(info.n);
    for (let j = 0; j < edges.pre.length; j++) {
      const type = info.types[info.typeId[edges.pre[j]]] ?? "";
      if (type.startsWith("PAM")) pam[edges.post[j]] += edges.count[j];
      else if (type.startsWith("PPL1")) ppl[edges.post[j]] += edges.count[j];
    }
    const n = learning.weights.length;
    this.comp = new Int8Array(n);
    this.absBase = new Float32Array(n);
    for (let e = 0; e < n; e++) {
      const mbon = learning.post[learning.edgePost[e]];
      this.comp[e] = pam[mbon] + ppl[mbon] === 0 ? 0 : pam[mbon] > ppl[mbon] ? 1 : -1;
      this.absBase[e] = Math.abs(plastic.base[e]);
    }
    this.kcSide = Uint8Array.from(learning.pre, (c) => info.side[c]);
    this.trace = new Float32Array(learning.pre.length);
  }

  /** per plastic edge, in learning.weights order: +1 PAM compartment, -1 PPL1, 0 neither (for MB.consolidateMs) */
  get compartments(): Readonly<Int8Array> {
    return this.comp;
  }

  /** both hemispheres together */
  get total() {
    return (this.left + this.right) / 2;
  }

  /** Call after rates.update(readback). */
  update(rates: Rates) {
    const dt = rates.dtMs;
    if (!(dt > 0)) return;
    const g = this.probes.group("learning:pre");
    const k = 1 - Math.exp(-dt / this.tauMs);
    const { trace, comp, absBase, kcSide } = this;
    for (let i = 0; i < trace.length; i++) trace[i] += ((rates.delta[g.start + i] * 1000) / dt - trace[i]) * k;
    const { weights, edgePre } = this.learning;
    // [side] -> all / removed, for the PAM and PPL1 compartments
    const pamAll = [0, 0, 0];
    const pamGone = [0, 0, 0];
    const pplAll = [0, 0, 0];
    const pplGone = [0, 0, 0];
    for (let e = 0; e < weights.length; e++) {
      if (comp[e] === 0) continue;
      const p = edgePre[e];
      const hz = trace[p];
      if (hz <= 0) continue;
      const s = kcSide[p];
      const drive = hz * absBase[e];
      const gone = drive * (1 - weights[e]);
      if (comp[e] === 1) {
        pamAll[s] += drive;
        pamGone[s] += gone;
      } else {
        pplAll[s] += drive;
        pplGone[s] += gone;
      }
    }
    const f = this.floor;
    const af = this.avoidFloor;
    this.left = pamGone[1] / (pamAll[1] + f);
    this.right = pamGone[2] / (pamAll[2] + f);
    this.aversionLeft = pplGone[1] / (pplAll[1] + af);
    this.aversionRight = pplGone[2] / (pplAll[2] + af);
  }
}
