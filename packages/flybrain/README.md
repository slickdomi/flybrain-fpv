# flybrain

The whole central nervous system of a male fruit fly, simulated in real time in the browser: all 166,700 neurons of the [MaleCNS v1.0](https://male-cns.janelia.org/) connectome, on WebGPU or in a CPU web worker. Show its compound eyes your world, read out any neuron, stimulate any neuron, and let dopamine change its mushroom body.

`flybrain` is the simulation behind [Aimbug](https://github.com/slickdomi/aimbug), pulled out of that game so other games and experiments can use it, and first used by PacFly in [fly-games](https://github.com/slickdomi/fly-games).

> **This is a vendored copy, modified for Fly Addiction.** It adds exactly one seam to `GpuBrain`:
> `EyeOptions`, which replaces the CPU-fed retina pass with a compute eye that samples your scene on the
> GPU. Pass `eye: { sceneBuffer, shader }` and the brain reads the world straight out of your uniform
> buffer with no CPU round-trip; the shader supplies `meanPass` and `contrastPass` over the bind group
> `[scene, eyeParams, units, ext, means]` and writes the same `ext` values the built-in pass would, so
> nothing downstream changes. **While an eye is installed, `setVision()` is ignored** — the quickstart
> below describes the other path, which is still there and still works.
>
> It also adds two optional `LoadOptions`, both off unless passed, so upstream behaviour is unchanged:
> `silence: [{ pre, post }]` gives spiking edges between those types no fast effect (as dopamine synapses
> already have none), and `gain: [{ pre, post, factor }]` multiplies their synapse counts. Fly Addiction uses
> the first for KC→KC (mAChR-B, see below) and the second, declared, for visual projection neurons onto the
> visual Kenyon cells. Worth porting back. Everything else is unchanged.

```ts
import { Connectome, Probes, Rates, Retina, createFlyBrain } from "flybrain";

const brain = await createFlyBrain({ url: "/data/malecns-v1" }); // WebGPU, or the CPU worker as a fallback
const cells = new Connectome(brain.info);
const retina = new Retina(brain.info);

const probes = new Probes()
  .add("DNa02:L", cells.select({ type: "DNa02", side: "left" }))
  .add("DNa02:R", cells.select({ type: "DNa02", side: "right" }));
brain.setProbes(probes.indices());
const rates = new Rates(probes);
brain.onReadback = (r) => rates.update(r);

function frame(dtMs: number) {
  const turn = 10 * (rates.hz("DNa02:R") - rates.hz("DNa02:L")); // degrees per second
  player.heading += (turn * dtMs) / 1000;

  retina.orient(player.heading); // yaw in radians, + = right
  retina.render((x, y, z) => myWorld.luminance(x, y, z)); // world-space ray -> 0..1
  brain.setVision(retina.normalize());
  brain.step(dtMs); // at most 32 ms per call
}
```

## What the model is

The model is a hybrid, validated against a NumPy reference in Aimbug:

- **Spiking neurons** make up the central brain, the ventral nerve cord and the visual projection neurons. They are leaky integrate-and-fire units with the published parameters of Shiu et al. 2024 (0.275 mV per synapse, τm 20 ms, τs 5 ms, 1.8 ms delay), plus spike-frequency adaptation. Synapse signs come from the predicted transmitter.
- **Graded rate units** stand in for the 95,501 optic-lobe cells, which are mostly non-spiking in the fly.
- **Photoreceptors** are driven by luminance contrast against each eye's mean.

About 19 M of the connectome's 25.6 M connections run. Dopamine, serotonin and octopamine have no fast synaptic effect.

It is a toy: reconstructed wiring with approximate dynamics, not a validated emulation of a fly. Some cells that matter here, among them DNa02, lLN1_bc, MDN and about half of LC4, LPLC2 and LC10a, are still marked "Prelim Roughly traced" in MaleCNS v1.0, so their wiring may change in later releases. Aimbug's README documents what it does and does not do.

What is known to work (from Aimbug's experiments):

- **Orienting toward dark objects.** Photoreceptors → medulla → LC10 → AOTU019/AOTU025 → DNa02. The difference between the right and left DNa02 rates is a turn command, and objects need to be large. The eye samples every 4.8°, and Aimbug's fly nearly lost a target below about 13° radius (26° across). PacFly steers toward pellets drawn at 8–16° radius.
- **Pitch** from DNp53, balanced against LC4/LPLC2.
- **Courtship song** from pIP10, with constant drive to P1 (`pC1_*`) neurons.

Anything else is unexplored territory. Probe, silence and stimulate before you trust a readout.

## Concepts

### Backends

`createFlyBrain(options)` downloads and decodes the connectome in a worker and returns a `FlyBrain`:

| | WebGPU (`GpuBrain`) | CPU (`CpuBrain`) |
|---|---|---|
| speed | about real time on a desktop GPU | about 0.3× real time on a desktop CPU |
| `ready` | always | false while the worker is busy with the last chunk |
| visualisation | `BrainView` draws straight from GPU buffers | `BrainView2D` from `readActivity()` snapshots |

Pass `backend: "webgpu" | "cpu"` to force one. Pass `device` to share your renderer's WebGPU device; it needs the storage-buffer limits that `requestFlyDevice()` asks for. `GpuBrain.encode(encoder, ms)` plus `afterSubmit()` lets the brain share a command buffer with your rendering.

Time is **brain time**. Advance your game by the same milliseconds you pass to `step()`, and a slow machine plays in slow motion instead of making the fly worse.

### Vision: `Retina`

The eyes have 10,321 visual units, each with a viewing direction: 5,895 photoreceptors and 4,426 lamina cells (L1–L3 with a column assigned).

1. `retina.orient(yaw, pitch)` points them into your world. The fly frame is +x right, +y up, -z forward; yaw > 0 turns right and pitch > 0 looks up.
2. Fill `retina.lum` (0 = black, 1 = white) in one of two ways. `retina.render(shader)` calls your function per unit, averaging 5 rays within 2.4° of each unit's axis (half the model's 4.8° acceptance angle). Or loop over `retina.world` (xyz per unit) with your own batch renderer.
3. `brain.setVision(retina.normalize())` converts luminance to photoreceptor contrast and uploads it.

Design your scene for the fly. In PacFly, a view crowded with objects set off seizures, and a view with no contrast at all silenced the brain. A flat background with faint structure and a few dark objects of interest works best, and `retina.az`/`retina.el` let you plot what the eyes get.

### Reading neurons: `Connectome`, `Probes`, `Rates`

- `new Connectome(brain.info).select({ type, side, superclass, spiking, receptiveField })` returns neuron indices. `type` can be a name, a list of names or a RegExp. `receptiveField: { az, el, radius }` picks visual cells by their connectome receptive field.
- `Probes` names groups of neurons. `brain.setProbes(probes.indices())` makes their spike counters come back in every `Readback`.
- `Rates.update(readback)` gives each group's mean rate per cell: `hz` (50 ms filter), `slowHz` (250 ms) and `instHz`, plus raw `spikes` and per-neuron `delta`.
- `brain.readActivity()` copies every neuron's spike trace and the graded optic-lobe state. It's slow, so use it for analysis.

### Stimulating neurons

- `brain.setDrive(neurons, mV)` sets constant input above rest (the threshold is 7 mV). For example, 6 mV into the `pC1_*` P1 cells is Aimbug's courtship arousal.
- `brain.pulse(neurons, mV, ms)` adds input for a while, in brain time, on top of any constant drive.

### Reward and learning: `DopamineLearning`

The fly learns in its mushroom body. Kenyon cells (KCs) carry a sparse code of what the fly senses. Dopamine neurons (PAM for reward, PPL1 for punishment) depress the synapses from recently active KCs onto the output neurons (MBONs) in their own compartment.

`DopamineLearning` implements that three-factor rule on the running brain:

```ts
const brain = await createFlyBrain({
  url,
  plastic: { pre: "^KC", post: "^MBON" }, // these synapses get float weights
  extract: [{ name: "dopamine", pre: "^(PAM|PPL1)", post: "^MBON" }], // which dopamine cells reach which MBON
});
const learning = new DopamineLearning(brain, probes, { dopamineEdges: "dopamine" }); // before setProbes
brain.setProbes(probes.indices());
brain.onReadback = (r) => {
  rates.update(r);
  learning.update(rates);
};

// in your game: reward and punishment are dopamine pulses
onPellet(() => brain.pulse(cells.select({ type: /^PAM/ }), 12, 100));
onDeath(() => brain.pulse(cells.select({ type: /^PPL1/ }), 12, 900));
```

The rule works as follows:
- **Eligibility:** each KC keeps a trace of its recent spikes (τ 1 s).
- **Dopamine reach:** each MBON's dopamine level is the mean of its dopamine cells' filtered rates, weighted by their DAN→MBON synapse counts. Compartments come from the connectome, not from a lookup table.
- **Depression:** a weight falls by `exp(-rate · eligibility · dopamine · dt)`.
- **Recovery:** weights relax back to the connectome weight (τ 2 min).

In MaleCNS v1.0 that covers 61,210 KC→MBON connections (463,640 synapses), 4,064 KCs, 97 MBONs, 316 PAM and 16 PPL1 cells.

Plastic weights are stored in fixed point (1/16 synapse) and are uploaded a few times per second. Any spiking pre/post type pair can be made plastic, and `setPlasticWeights()` lets you drive the weights with your own rule.

To show what has been learned, read the state directly (PacFly's memory map does):
- `weights`: relative weight per plastic edge (1 = connectome), in `info.plastic` order.
- `edgePre` / `edgePost`: each edge's cell as an index into `pre` (KCs) and `post` (MBONs).
- `eligibility`: the trace per KC; `phasic`: the dopamine burst (Hz above baseline and threshold) arriving at each MBON now.
- `stats`: mean weight, fraction weakened by more than 10%, and `uploads`, which changes whenever the weights did.

Two caveats on what learning can do here:
- **Little visual input to KCs:** 99.4% of their input comes from other central-brain cells, and only 0.5% from visual projection neurons.
- **Weak MBON links to steering:** MBONs do synapse directly onto DNa02 and AOTU019, but each of those links is under 0.3% of the steering cell's input.

So don't expect a learned change in behaviour without measuring for one.

We measured it (`scripts/conditioning.ts`, CPU backend) and found three problems with this model:
- **Silent Kenyon cells with vision alone.** With a dark blob in view and no other input, 7 of 4,064 KCs spiked in the first second, then 1–2 per second. Four 500 ms PAM pulses left the mean weight at 0.9997.
- **Odours saturate and seize.** Driving the DA2 olfactory receptor neurons with at least 8 mV (6 to 10 Hz) makes their projection neurons fire at about 50 Hz. That activates about 68% of all KCs (21% of KC input comes from other KCs, which the model treats as excitatory), and it ignites the lLN1_bc clique, seizing about once a second. At 6 mV the receptor neurons stay silent.
- **No pairing specificity.** During those seizures the dopamine cells fire at about 30 Hz on their own. Weights fell just as far without paired PPL1 pulses as with them (mean 0.58 against 0.57). Dopamine-triggered depression does happen, and the odour responses of the MBONs in PPL1's compartments halved, but it isn't specific to the pairing.

**Follow-up in Fly Addiction** (`experiments/visualkc.ts`, `kcinputs.ts` there). The KC→KC input is not just
a fifth of all KC input: onto the visual KCs (KCg-d, KCab-p) it is ~320 synapses per cell against ~30 from
visual projection neurons. In the fly those axo-axonic synapses act through inhibitory muscarinic mAChR-B
(Manoim et al. 2022, Curr Biol), so `silence: [{ pre: "^KC", post: "^KC" }]` drops them. That alone does not
wake the visual KCs — aMe12/MeVP41 respond to a dark patch (0 → 27.6 / 13.3 Hz) but ~30 synapses at 0.275 mV
cannot reach threshold — so it adds a declared `gain` of 20 on aMe/MeVP/LoVP → KCg-d/KCab-p. Then views of
the lab give sparse (≤3% of KCs), reliable, lateralized, view-specific KC patterns, and pairing one with PAM
depresses its PAM-compartment KC→MBON drive more than an unpaired view's (0.70 vs 0.46). Two new cautions from
that work: PPL1 idles at ~5 Hz and bursts to 12–15 Hz at a close dark patch, which the rule reads as
punishment, so every view quickly loses 80–90% of its PPL1-compartment drive; and KC→KC silencing does not
rescue smell, because the antennal lobe drives both glomeruli's PNs to 115–145 Hz whichever is stimulated.

The rule is in place, but a meaningful memory in this model would first need sparse Kenyon-cell coding and a stable antennal lobe. The real fly gets those from graded, non-spiking APL feedback (Lin et al. 2014), gap junctions and slow GABA-B inhibition. The model has APL, but as an ordinary spiking cell, and lacks the other two. Disable learning while the brain runs away, or it learns seizure noise. PacFly watches the seizure clique through a probe on lLN1_bc: learning pauses while it fires above 20 Hz and for 3 s after each seizure reset. A whole-brain spike limit doesn't work, because busy but healthy play can exceed it.

### Seizures: `SeizureWatchdog`

Every few minutes the cholinergic lLN1_bc clique in the antennal lobe ignites and the brain runs away (over 90 k spikes/s). `new SeizureWatchdog(brain).update(readback)` resets the spiking state when that happens. It keeps the optic lobe and learned weights, and counts seizures.

Busy visual scenes can push the whole brain past 90 k spikes/s without any seizure; PacFly's pellet corridors did. To reset only for the real thing, probe the clique and set `watchdog.confirm`:

```ts
probes.add("lLN1_bc", cells.select({ type: "lLN1_bc" }));
watchdog.confirm = () => rates.instHz("lLN1_bc") > 100; // it fires at ~200 Hz per cell in a seizure
```

Runaways that `confirm` turns down are counted in `watchdog.rejected`. Call `rates.update()` before `watchdog.update()` so the check sees the current chunk.

Visual input changes how often seizures happen. Aimbug, with one target in view, had about one every 3 minutes. In PacFly, running past rows of pellets at first triggered two a second. With fewer, non-looming blobs and no P1 arousal drive, that came down to 1–2 per minute.

## Data

The connectome files are CC BY 4.0 (FlyEM/HHMI Janelia, University of Cambridge, MRC LMB, Google Research) and are not bundled in the npm package. They're about 37 MB, gzipped, with every file under 20 MB. Serve the `malecns-v1` directory from `apps/pacman/public/data/` of this repository (or rebuild it with `pipeline/`), and pass its URL as `url`. Attribution and the description of changes are in that directory's `LICENSE.md`.

## Building

Everything runs in Docker:

```sh
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm ci
docker run --rm -v "$PWD:/app" -w /app/packages/flybrain node:22-alpine npm run build   # dist/: ES module, workers, .d.ts
```

Consumers need a bundler that understands `new Worker(new URL("./worker.js", import.meta.url))` (Vite, webpack 5, Parcel, esbuild with a plugin). They also need `@webgpu/types` for the WebGPU parts of the typings. Inside this monorepo, apps resolve the TypeScript sources directly through the `flybrain-source` export condition.

## Credits

- Connectome: Berg, S. et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, [Cell 2026](https://doi.org/10.1016/j.cell.2026.08.015) (CC BY 4.0)
- LIF parameters: Shiu, P. K. et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 2024
- Mushroom body plasticity: Hige, T. et al., "Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila", Neuron 2015
- Model, eye and optic-lobe tuning: [Aimbug](https://github.com/slickdomi/aimbug)

MIT licensed (code), © 2026 SlickDomi.
