// Graded -> spiking coupling: deviations of graded release from rest become a tonic
// drive (mV) on spiking cells (visual projection neurons and beyond).
//
// Two entry points over the same rows. Row lengths are very uneven (half the spiking cells take no graded input, a
// few take thousands of entries), and one thread walking a row of thousands kept the whole pass waiting on it. So
// `main` takes the rows up to LONG_ROW entries (gpu/brain.ts), one thread each, from a sliced ELL copy (slicedEll:
// a row's entries ELL_SLICE apart, so neighbouring threads read neighbouring memory; longer rows are ELL_SKIP
// there), and `longRows` takes the rest (listed once, at load time) from the CSR, one workgroup each: its lanes
// stride the row and add up their partial sums. ELL_SLICE and ELL_SKIP are prepended by gpu/brain.ts.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> entries: array<Entry>;
@group(0) @binding(3) var<storage, read> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> gDrive: array<f32>;
@group(0) @binding(5) var<storage, read> longIdx: array<u32>;
@group(0) @binding(6) var<storage, read> ellRows: array<vec2u>;
@group(0) @binding(7) var<storage, read> ellEntries: array<Entry>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let r = id.x;
  if (r >= P.n - P.ng) {
    return;
  }
  let row = ellRows[r];
  if (row.y == ELL_SKIP) {
    return;
  }
  var sum = 0.0;
  for (var k = 0u; k < row.y; k++) {
    let en = ellEntries[row.x + k * ELL_SLICE];
    sum += en.val * clamp(a[en.col], P.gMin, P.gMax);
  }
  gDrive[r + P.ng] = P.coupling * sum;
}

const LANES = 128u;
var<workgroup> partial: array<f32, LANES>;

@compute @workgroup_size(128)
fn longRows(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  // arrayLength is uniform, so every lane takes the same trips round this loop and reaches every barrier
  for (var k = wid.x; k < arrayLength(&longIdx); k += nwg.x) {
    let r = longIdx[k];
    var sum = 0.0;
    for (var e = offsets[r] + lid.x; e < offsets[r + 1u]; e += LANES) {
      let en = entries[e];
      sum += en.val * clamp(a[en.col], P.gMin, P.gMax);
    }
    partial[lid.x] = sum;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s >>= 1u) {
      if (lid.x < s) {
        partial[lid.x] += partial[lid.x + s];
      }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      gDrive[r + P.ng] = P.coupling * partial[0];
    }
    workgroupBarrier();
  }
}
