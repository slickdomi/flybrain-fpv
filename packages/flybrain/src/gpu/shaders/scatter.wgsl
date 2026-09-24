// Event-driven spike propagation: each neuron that spiked this step adds its signed
// synaptic weights to its targets' input slot `tDelay` in the future.
//
// One workgroup per spike (workgroups take the step's spikes in turn), its lanes striding over the spiking neuron's
// out-edges. A step has a few dozen spikes and a neuron hundreds to thousands of targets: one thread per spike left
// the GPU nearly idle, each thread walking its whole edge list alone. The sums are integer atomics, so the order the
// lanes add in changes nothing.

const LANES = 64u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: Step;
@group(0) @binding(2) var<storage, read> spikes: array<u32>;
@group(0) @binding(3) var<storage, read_write> spikeCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read> edges: array<u32>;
@group(0) @binding(6) var<storage, read_write> ring: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read> plasticOffsets: array<u32>;
@group(0) @binding(8) var<storage, read> plasticEdges: array<u32>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let count = min(atomicLoad(&spikeCount[S.stepIdx]), P.maxSpikes);
  let base = S.writeSlot * P.n;
  for (var k = wid.x; k < count; k += nwg.x) {
    let pre = spikes[k];
    for (var e = offsets[pre] + lid.x; e < offsets[pre + 1u]; e += LANES) {
      let packed = edges[e];
      // target in the low 18 bits, signed synapse count in the high 14
      atomicAdd(&ring[base + (packed & 0x3FFFFu)], (bitcast<i32>(packed) >> 18u) * FIXED);
    }
    // plastic synapses: the high 14 bits already hold the weight in fixed point
    for (var e = plasticOffsets[pre] + lid.x; e < plasticOffsets[pre + 1u]; e += LANES) {
      let packed = plasticEdges[e];
      atomicAdd(&ring[base + (packed & 0x3FFFFu)], bitcast<i32>(packed) >> 18u);
    }
  }
}
