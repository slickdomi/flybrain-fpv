// Writes the per-unit photoreceptor contrast (computed by Retina.normalize on the CPU) into the
// external input of the graded optic lobe: photoreceptors are clamped to it, lamina columns add -w * contrast.

struct Unit {
  dx: f32, dy: f32, dz: f32, weight: f32,
  g: u32, side: u32, mode: u32, pad: u32,
};

@group(0) @binding(0) var<storage, read> units: array<Unit>;
@group(0) @binding(1) var<storage, read> contrast: array<f32>;
@group(0) @binding(2) var<storage, read_write> ext: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&units)) {
    return;
  }
  let u = units[id.x];
  ext[u.g] = u.weight * contrast[id.x];
}
