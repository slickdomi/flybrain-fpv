// The fly's compound eyes, sampled straight from the world on the GPU. Prepended with scene.wgsl.
// Copied unchanged from fly-addiction (web/src/shaders/eye.wgsl).
//
// Writes each visual unit's contrast into `ext`, the external input of the graded optic lobe -- exactly what
// flybrain's CPU-fed retina pass writes, so nothing downstream changes. Two entry points, dispatched in
// order by GpuBrain: meanPass fills each eye's mean luminance, contrastPass turns luminance into contrast.
//
// Adapted from Aimbug (fly-thing), minus its sprite texture (this lab is analytic) and with meanPass
// parallelised -- see the note on it below.

struct Unit {
  dx: f32, dy: f32, dz: f32, weight: f32, // fly-frame view direction; weight 1 (photoreceptor) or -w (lamina)
  g: u32, side: u32, mode: u32, pad: u32,
};

struct EyeParams {
  count: u32,
  blur: f32, // radians, half acceptance angle of one ommatidium
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> eye: EyeParams;
@group(0) @binding(2) var<storage, read> units: array<Unit>;
@group(0) @binding(3) var<storage, read_write> ext: array<f32>;
@group(0) @binding(4) var<storage, read_write> means: array<f32>; // [1] left eye, [2] right eye

const MEAN_THREADS = 64u;

var<workgroup> partialSum: array<f32, 64>;
var<workgroup> partialCount: array<f32, 64>;

fn lum(d: vec3f) -> f32 {
  return sceneLum(scene.eye, toWorld(d, scene.heading, scene.pitch));
}

/** Five rays across the acceptance angle, as flybrain's CPU Retina.render does with blur on. */
fn sampleUnit(d: vec3f) -> f32 {
  var a = cross(d, vec3f(0.0, 1.0, 0.0));
  if (length(a) < 1e-3) {
    a = vec3f(1.0, 0.0, 0.0);
  }
  a = normalize(a) * eye.blur;
  let b = cross(d, a);
  return (2.0 * lum(d) + lum(normalize(d + a)) + lum(normalize(d - a)) + lum(normalize(d + b)) + lum(normalize(d - b))) / 6.0;
}

/**
 * Each eye's mean luminance, which contrastPass normalises against.
 *
 * Upstream this was `@workgroup_size(1)` with two workgroups: two threads, each walking every third unit
 * and ray-casting the scene on every lamina hit -- roughly 490 raycasts in sequence, on a GPU with
 * thousands of idle cores, every frame. It cost more than contrastPass, which does a hundred times the
 * work but spreads it over 11,222 threads.
 *
 * This samples exactly the same set (thread t takes 3t, 3t+192, ...; the union over 64 threads is every
 * third unit, as before) and reduces in shared memory, so the mean is unchanged and the fly sees the same
 * thing. One workgroup per eye, so the dispatch stays at 2.
 */
@compute @workgroup_size(64)
fn meanPass(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let side = wid.x + 1u;
  var sum = 0.0;
  var cnt = 0.0;
  for (var k = lid.x * 3u; k < eye.count; k += MEAN_THREADS * 3u) {
    let u = units[k];
    if (u.side != side || u.mode != 2u) {
      continue;
    }
    sum += lum(vec3f(u.dx, u.dy, u.dz));
    cnt += 1.0;
  }
  partialSum[lid.x] = sum;
  partialCount[lid.x] = cnt;
  workgroupBarrier();

  var stride = MEAN_THREADS / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lid.x < stride) {
      partialSum[lid.x] += partialSum[lid.x + stride];
      partialCount[lid.x] += partialCount[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    means[side] = max(partialSum[0] / max(partialCount[0], 1.0), 1e-3);
  }
}

@compute @workgroup_size(128)
fn contrastPass(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= eye.count) {
    return;
  }
  let u = units[id.x];
  let m = means[u.side];
  // contrast against the eye's mean, clamped to [-1, 3] as in the validated model
  let c = clamp((sampleUnit(vec3f(u.dx, u.dy, u.dz)) - m) / m, -1.0, 3.0);
  ext[u.g] = u.weight * c;
}
