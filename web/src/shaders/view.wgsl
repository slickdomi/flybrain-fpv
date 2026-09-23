// The player's view: one full-screen triangle, one ray per pixel through the same analytic world the fly's
// eye samples, then a post pass. Prepended with scene.wgsl.
//
// Four modes (view.mode):
//   0  EYE: the fly's own field of view, azimuth and elevation mapped straight onto the screen (the model's
//      eyes cover about +-140 deg by +-83 deg), drawn with the realistic look below.
//   1  CHASE: a camera behind the drone, which is drawn here (a long-range FPV strike drone: X frame, battery,
//      camera pod, a warhead slung underneath, and the contact fuze: two steel wires bent into loops ahead).
//   2  MOSAIC: sampled once per 4.8 deg hexagon in plain luminance, straight from sceneHit(): roughly the
//      resolution and the contrast the brain gets, and nothing view-only.
//   3  FPV: the drone's camera at the fly's head, a 130 deg fisheye that rolls with the bank, graded in `post`
//      into gritty low-bitrate footage. The fuze wires and the warhead's nose show at the bottom of the picture.
//
// EVERYTHING HERE IS VIEW-ONLY. The fly's eye calls sceneLum() and nothing else, so none of this can change what
// the brain sees: the ash, soil and debris are flat texture on the ground plane; the fallen logs, smoke haze, fog,
// plumes, explosions, grenades, wrecks and the drone are drawn for the player alone (the logs are solid in the
// view, but the fly can neither see nor hit them). Where the view's shape differs from the fly's silhouette it only
// ever removes: each truck is drawn inside the fly's box, and the trunks' broken tops sit up to a metre below the
// fly's flat-topped cylinders.
//
// No derivatives (fwidth, dpdx) and no implicit-LOD texture sampling anywhere: detail finer than a pixel fades
// out by distance instead (view.fx.z), so nothing depends on uniform control flow.

const MAX_GRENADES = 8u;   // must match render.ts
const MAX_EXPLOSIONS = 8u;
const MAX_WRECKS = 8u;

struct View {
  mode: f32, fovAz: f32, fovEl: f32, time: f32, // half-fields in radians (EYE, MOSAIC)
  camPos: vec3f, tanX: f32,                     // FPV: tanX = aspect
  camRight: vec3f, tanY: f32,
  camUp: vec3f, seizure: f32,
  camFwd: vec3f, bank: f32,
  drone: vec3f, droneHeading: f32,
  paint: vec4f, // az, el, radius (radians), active 0/1
  dronePitch: f32, rotor: f32, escape: f32, grenadesLeft: f32,
  food: vec4f,  // what is designated as food: kind (0 none, 1 balloon, 2 truck), its index, 0, 0
  fx: vec4f,    // grenade count, explosion count, radians per pixel, wreck count
  fpv: vec4f,   // FPV half field (rad), seconds since the drone was lost (-1: it isn't), drone drawn 1/0, render scale
  post: vec4f,  // blast glitch 0..1, FPV exposure, 0, 0
  grenades: array<vec4f, 16>,   // per grenade: (x, y, z, 0), (vx, vy, vz, 0)
  explosions: array<vec4f, 16>, // per explosion: (x, y, z, radius), (age s, seed, 0, 0)
  wrecks: array<vec4f, 8>,      // x, z, heading, age s
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> view: View;
@group(1) @binding(0) var frameTex: texture_2d<f32>;
@group(1) @binding(1) var frameSamp: sampler;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  var o: VsOut;
  o.pos = vec4f(p[i], 0.0, 1.0);
  o.ndc = p[i];
  return o;
}

// ---- light: an overcast day over a burnt forest ------------------------------------------------------------

/** a hazy sun behind the overcast: a weak directional term on top of the sky's diffuse light */
const SUN = vec3f(0.5025, 0.7035, 0.5025);
const SUN_COL = vec3f(0.55, 0.48, 0.40);
const SKY_AMB = vec3f(0.62, 0.63, 0.64);
/** The eyes' reach in the model (Aimbug README: about +-140 deg azimuth, +-83 deg elevation) */
const EYE_AZ = 2.44;
const EYE_EL = 1.45;
/** the car keeps this far right of the road's centre line (CAR.lane), for the ruts it leaves */
const CAR_LANE = 1.0;

/** emission picked up by the last shading call (embers, lights, flames), added after lighting */
var<private> gEmit: vec3f;

fn lit(albedo: vec3f, n: vec3f, ao: f32, sh: f32) -> vec3f {
  let dif = max(dot(n, SUN), 0.0) * sh;
  let amb = (0.6 + 0.4 * n.y) * ao;
  return albedo * (SUN_COL * dif + SKY_AMB * amb);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

/** HDR -> the screen, for the fly-eye and chase views (FPV has its own grade in `post`) */
fn toDisplay(c: vec3f) -> vec3f {
  return pow(aces(c * 1.15), vec3f(1.0 / 2.2));
}

// ---- noise -------------------------------------------------------------------------------------------------

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2f) -> vec2f {
  var p3 = fract(vec3f(p.x, p.y, p.x) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

fn hash13(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash41(p: f32) -> vec4f {
  var p4 = fract(vec4f(p) * vec4f(0.1031, 0.1030, 0.0973, 0.1099));
  p4 += dot(p4, p4.wzxy + 33.33);
  return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

fn wrapCell(c: vec2f, period: f32) -> vec2f {
  return c - period * floor(c / period);
}

/** 2D value noise that repeats every `period` cells, so ground texture has no seam where the torus wraps */
fn pnoise(p: vec2f, period: f32) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash12(wrapCell(i, period));
  let b = hash12(wrapCell(i + vec2f(1.0, 0.0), period));
  let c = hash12(wrapCell(i + vec2f(0.0, 1.0), period));
  let d = hash12(wrapCell(i + vec2f(1.0, 1.0), period));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/** fbm of pnoise over world xz: `freq` cycles per unit, with 100 * freq a whole number (one torus period) */
fn fbmP(p: vec2f, freq: f32, octaves: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var f = freq;
  var norm = 0.0;
  for (var i = 0; i < octaves; i++) {
    s += a * pnoise(p * f, round(100.0 * f));
    norm += a;
    a *= 0.5;
    f *= 2.0;
  }
  return s / norm;
}

fn noise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(hash13(i), hash13(i + vec3f(1.0, 0.0, 0.0)), u.x);
  let b = mix(hash13(i + vec3f(0.0, 1.0, 0.0)), hash13(i + vec3f(1.0, 1.0, 0.0)), u.x);
  let c = mix(hash13(i + vec3f(0.0, 0.0, 1.0)), hash13(i + vec3f(1.0, 0.0, 1.0)), u.x);
  let d = mix(hash13(i + vec3f(0.0, 1.0, 1.0)), hash13(i + vec3f(1.0, 1.0, 1.0)), u.x);
  return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

fn fbm3(p: vec3f) -> f32 {
  return (0.5 * noise3(p) + 0.25 * noise3(p * 2.03 + 17.1) + 0.125 * noise3(p * 4.01 + 31.7)) / 0.875;
}

/** fbm3 with two octaves: for volumes, which evaluate it many times per pixel */
fn fbm2o(p: vec3f) -> f32 {
  return (0.5 * noise3(p) + 0.25 * noise3(p * 2.03 + 17.1)) / 0.75;
}

/** 2D cellular noise, periodic every `period` cells in x: (nearest, second nearest) feature distance */
fn voronoiX(p: vec2f, period: f32) -> vec2f {
  let i = floor(p);
  let f = fract(p);
  var d1 = 8.0;
  var d2 = 8.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let g = vec2f(f32(x), f32(y));
      let cell = i + g;
      let o = hash22(vec2f(cell.x - period * floor(cell.x / period), cell.y));
      let d = length(g + o - f);
      if (d < d1) {
        d2 = d1;
        d1 = d;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  return vec2f(d1, d2);
}

// ---- shapes ------------------------------------------------------------------------------------------------

fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / max(k1, 1e-5);
}

fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn sdCylY(p: vec3f, r: f32, h: f32) -> f32 {
  let d = vec2f(length(p.xz) - r, abs(p.y) - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn sdRoundBox(p: vec3f, b: vec3f, rr: f32) -> f32 {
  let q = abs(p) - b + vec3f(rr);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - rr;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

fn opU(a: vec2f, d: f32, m: f32) -> vec2f {
  return select(a, vec2f(d, m), d < a.x);
}

/** entry and exit distance of a ray through an axis-aligned box */
fn boxInterval(o: vec3f, d: vec3f, lo: vec3f, hi: vec3f) -> vec2f {
  let inv = 1.0 / select(d, vec3f(1e-6), abs(d) < vec3f(1e-6));
  let t0 = (lo - o) * inv;
  let t1 = (hi - o) * inv;
  let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  return vec2f(tn, tf);
}

/** closest approach of a ray (unit rd) to a segment: (distance, distance along the ray) */
fn raySeg(ro: vec3f, rd: vec3f, a: vec3f, b: vec3f) -> vec2f {
  let ba = b - a;
  let oa = ro - a;
  let bb = dot(ba, ba);
  let rb = dot(rd, ba);
  let orr = dot(oa, rd);
  let ob = dot(oa, ba);
  let s = clamp((ob - rb * orr) / max(bb - rb * rb, 1e-6), 0.0, 1.0);
  let t = max(rb * s - orr, 0.0);
  return vec2f(length(oa + rd * t - ba * s), t);
}

fn sphereInterval(ro: vec3f, r: vec3f, c: vec3f, rad: f32) -> vec2f {
  let oc = ro - c;
  let b = dot(oc, r);
  let cc = dot(oc, oc) - rad * rad;
  let disc = b * b - cc;
  if (disc < 0.0) { return vec2f(NO_HIT, -NO_HIT); }
  let s = sqrt(disc);
  return vec2f(-b - s, -b + s);
}

/** Fly-frame direction for an azimuth / elevation (az > 0 = right, el > 0 = up). */
fn dirAzEl(az: f32, el: f32) -> vec3f {
  return vec3f(sin(az) * cos(el), sin(el), -cos(az) * cos(el));
}

/** the car's frame back to world (toCar inverted) */
fn fromCar(v: vec3f, heading: f32) -> vec3f {
  let c = cos(heading);
  let s = sin(heading);
  return vec3f(v.x * c - v.z * s, v.y, v.x * s + v.z * c);
}

// ---- sky, haze and fog -------------------------------------------------------------------------------------

fn skyBase(r: vec3f) -> vec3f {
  let y = clamp(r.y, 0.0, 1.0);
  var c = mix(vec3f(1.02, 0.98, 0.92), vec3f(0.80, 0.82, 0.85), pow(y, 0.6));
  let sd = max(dot(r, SUN), 0.0);
  c += vec3f(0.30, 0.26, 0.19) * pow(sd, 5.0) + vec3f(0.35, 0.30, 0.22) * pow(sd, 32.0);
  // the smoke layer: the low sky sinks into the same haze that hides the far ground, so the horizon is soft
  return mix(fogColor(r) * 1.08, c, smoothstep(-0.01, 0.32, r.y));
}

/** the smoke haze: brownish grey, a little brighter toward the hidden sun */
fn fogColor(r: vec3f) -> vec3f {
  let sd = max(dot(normalize(vec3f(r.x, 0.0, r.z) + vec3f(0.0, 1e-4, 0.0)), SUN), 0.0);
  return vec3f(0.31, 0.30, 0.285) * (1.0 + 0.35 * pow(sd, 3.0));
}

/** overcast: low stratus, streaky and slow */
fn skyColor(r: vec3f) -> vec3f {
  var c = skyBase(r);
  if (r.y > 0.0) {
    let q = r.xz / (r.y + 0.12) * 55.0 + vec2f(view.time * 0.9, view.time * 0.3);
    var n = 0.0;
    var a = 0.5;
    var f = vec2f(0.010, 0.017);
    for (var i = 0; i < 5; i++) {
      n += a * pnoise(q * f + vec2f(f32(i) * 13.1), 1e4);
      a *= 0.5;
      f *= 2.1;
    }
    let dark = smoothstep(0.38, 0.78, n);
    c = mix(c, c * vec3f(0.66, 0.66, 0.68), dark * smoothstep(0.0, 0.3, r.y));
  }
  return c;
}

/**
 * The visibility limit: fog closes in from VIS_NEAR and is opaque by VIS_FAR, just inside half the 100-unit world.
 * Every object is drawn at its nearest periodic image, so at 50 units a tree or truck would jump from one side of the
 * field to the other; without this it popped in and out of view there. View-only: the fly still sees to the seam.
 */
const VIS_NEAR = 18.0;
const VIS_FAR = 46.0;

/** smoke haze thickest near the ground (exponential height fog), a thin even haze, and the visibility limit */
fn fogAmount(ro: vec3f, r: vec3f, t: f32) -> f32 {
  let a = 0.02;
  let b = 0.1;
  var f = a * exp(-ro.y * b) * t;
  if (abs(r.y) > 1e-4) {
    f = (a / b) * exp(-ro.y * b) * (1.0 - exp(-t * r.y * b)) / r.y;
  }
  let haze = 1.0 - exp(-(max(f, 0.0) + t * 0.0025));
  return max(haze, smoothstep(VIS_NEAR, VIS_FAR, t));
}

// ---- the road, in the view: where along it a point is ------------------------------------------------------

/** (distance to the centre line, signed lateral offset (+ = right of travel along the points), arc length) */
fn roadFrame(p: vec2f) -> vec3f {
  let n = min(u32(scene.roadInfo.x), MAX_ROAD);
  var best = NO_HIT;
  var side = 1.0;
  var arc = 0.0;
  var acc = 0.0;
  for (var i = 0u; i < n; i++) {
    let a = roadPoint(i);
    let b = roadPoint((i + 1u) % n);
    let q = nearestImage(p, a) - a;
    let e = b - a;
    let len = length(e);
    let h = clamp(dot(q, e) / max(len * len, 1e-6), 0.0, 1.0);
    let d = length(q - e * h);
    if (d < best) {
      best = d;
      side = select(-1.0, 1.0, e.x * q.y - e.y * q.x < 0.0);
      arc = acc + h * len;
    }
    acc += len;
  }
  return vec3f(best, best * side, arc);
}

// ---- the military trucks (and the wrecks), inside the fly's box ---------------------------------------------
//
// A 6x6 cargo truck: a long bonnet and flat mudguards in front, a closed cab, a cargo bed under a canvas cover on
// its bows, three axles of big off-road tyres (one in front, a tandem behind), fuel tanks, an exhaust stack, olive
// drab paint with mud and dust. The model is built in a 2 x 2.4 x 5.6 reference box (TRUCK_BOX: half width, height,
// half length) and scaled to CAR's dimensions, so it always sits inside the box the fly sees. View-only.
//
// It is made of boxes and cylinders intersected analytically, one pass per ray, with the normal from whichever part
// was hit: no marching. A truck can fill the FPV view in the last seconds of a dive, and this keeps it cheap there.

const TRUCK_BOX = vec3f(1.0, 2.4, 2.8);
const TRUCK_WHEEL_R = 0.5;
/** the wheels' centre, either side, and the three axles (front, then the rear tandem) */
const TRUCK_WHEEL_X = 0.79;
const TRUCK_AXLE_F = -1.75;
const TRUCK_AXLE_R = 1.0;
const TRUCK_AXLE_GAP = 1.1;
/** the canvas cover's bows, this far apart along the bed, around its middle */
const TRUCK_BOW = 0.8;
const TRUCK_BED_Z = 1.14;
/** how far a wreck has sunk onto its rims */
const WRECK_SINK = 0.14;

/** a ray against a box: keeps it in best (t, material) and n if it is nearer */
fn tBox(o: vec3f, d: vec3f, inv: vec3f, lo: vec3f, hi: vec3f, mat: f32, best: ptr<function, vec2f>, n: ptr<function, vec3f>) {
  let t0 = (lo - o) * inv;
  let t1 = (hi - o) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tn = max(max(tmin.x, tmin.y), tmin.z);
  let tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tn > tf || tn <= 0.0 || tn >= (*best).x) { return; }
  *best = vec2f(tn, mat);
  *n = -sign(d) * step(tmin.yzx, tmin) * step(tmin.zxy, tmin);
}

/** a ray against a capped cylinder along the first axis of the frame it is given: (t, normal in that frame) */
fn tCyl(o: vec3f, d: vec3f, c: vec2f, rad: f32, lo: f32, hi: f32) -> vec4f {
  let oc = o.yz - c;
  let a = dot(d.yz, d.yz);
  let b = dot(oc, d.yz);
  let disc = b * b - a * (dot(oc, oc) - rad * rad);
  if (disc < 0.0 || a < 1e-10) { return vec4f(NO_HIT, 0.0, 0.0, 0.0); }
  let s = sqrt(disc);
  let t0 = (-b - s) / a;
  let ix = 1.0 / select(d.x, 1e-6, abs(d.x) < 1e-6);
  let x0 = (lo - o.x) * ix;
  let x1 = (hi - o.x) * ix;
  let tn = max(t0, min(x0, x1));
  let tf = min((-b + s) / a, max(x0, x1));
  if (tn > tf || tn <= 0.0) { return vec4f(NO_HIT, 0.0, 0.0, 0.0); }
  if (tn > t0) { return vec4f(tn, -sign(d.x), 0.0, 0.0); }
  return vec4f(tn, 0.0, (oc + d.yz * tn) / rad);
}

fn keep(h: vec4f, nn: vec3f, mat: f32, best: ptr<function, vec2f>, n: ptr<function, vec3f>) {
  if (h.x < (*best).x) {
    *best = vec2f(h.x, mat);
    *n = nn;
  }
}

/** a cylinder along x, centred at (y, z) = c */
fn cylX(o: vec3f, d: vec3f, c: vec2f, rad: f32, lo: f32, hi: f32, mat: f32, best: ptr<function, vec2f>, n: ptr<function, vec3f>) {
  let h = tCyl(o, d, c, rad, lo, hi);
  keep(h, h.yzw, mat, best, n);
}

/** a cylinder along z, centred at (x, y) = c */
fn cylZ(o: vec3f, d: vec3f, c: vec2f, rad: f32, lo: f32, hi: f32, mat: f32, best: ptr<function, vec2f>, n: ptr<function, vec3f>) {
  let h = tCyl(o.zxy, d.zxy, c, rad, lo, hi);
  keep(h, h.zwy, mat, best, n);
}

/** a cylinder along y, centred at (z, x) = c */
fn cylY(o: vec3f, d: vec3f, c: vec2f, rad: f32, lo: f32, hi: f32, mat: f32, best: ptr<function, vec2f>, n: ptr<function, vec3f>) {
  let h = tCyl(o.yzx, d.yzx, c, rad, lo, hi);
  keep(h, h.wyz, mat, best, n);
}

/**
 * A ray (any length of d) in the reference frame (+x right, -z front, y up from the ground) against the truck:
 * (t, material), NO_HIT on a miss, and the normal in n. Materials: 1 olive paint, 2 tyre, 3 black steel (chassis,
 * bumper, grille, mirrors, exhaust, bows), 5 canvas, 7 lamp; glass, the wheels' steel and the lights are picked out
 * by the colouring. `canvas`: false for a wreck, whose canvas burnt away and left the bows.
 */
fn truckTrace(o: vec3f, d: vec3f, canvas: bool, n: ptr<function, vec3f>) -> vec2f {
  var best = vec2f(NO_HIT, 0.0);
  let inv = 1.0 / select(d, vec3f(1e-6), abs(d) < vec3f(1e-6));
  // the cargo bed, and over it the canvas (its cross-section a rectangle with rounded upper corners) or the bows
  let bz0 = TRUCK_BED_Z - 1.62;
  let bz1 = TRUCK_BED_Z + 1.62;
  tBox(o, d, inv, vec3f(-1.0, 1.05, -0.5), vec3f(1.0, 1.63, 2.78), 1.0, &best, n);
  if (canvas) {
    tBox(o, d, inv, vec3f(-0.98, 1.5, bz0), vec3f(0.98, 2.14, bz1), 5.0, &best, n);
    tBox(o, d, inv, vec3f(-0.72, 2.14, bz0), vec3f(0.72, 2.4, bz1), 5.0, &best, n);
    cylZ(o, d, vec2f(-0.72, 2.14), 0.26, bz0, bz1, 5.0, &best, n);
    cylZ(o, d, vec2f(0.72, 2.14), 0.26, bz0, bz1, 5.0, &best, n);
  } else {
    for (var k = -2; k <= 2; k++) {
      let z = TRUCK_BED_Z + f32(k) * TRUCK_BOW;
      tBox(o, d, inv, vec3f(-0.98, 2.3, z - 0.03), vec3f(0.98, 2.36, z + 0.03), 3.0, &best, n);
      tBox(o, d, inv, vec3f(-0.98, 1.63, z - 0.03), vec3f(-0.92, 2.3, z + 0.03), 3.0, &best, n);
      tBox(o, d, inv, vec3f(0.92, 1.63, z - 0.03), vec3f(0.98, 2.3, z + 0.03), 3.0, &best, n);
    }
  }
  // the ladder frame and the fuel tanks
  tBox(o, d, inv, vec3f(-0.42, 0.64, -2.52), vec3f(0.42, 0.84, 2.72), 3.0, &best, n);
  cylZ(o, d, vec2f(-0.72, 0.66), 0.2, -0.97, -0.13, 1.0, &best, n);
  cylZ(o, d, vec2f(0.72, 0.66), 0.2, -0.97, -0.13, 1.0, &best, n);
  // the cab, the step under its doors, the mirrors at its front corners, the exhaust stack behind it
  tBox(o, d, inv, vec3f(-0.95, 0.95, -1.55), vec3f(0.95, 2.13, -0.61), 1.0, &best, n);
  tBox(o, d, inv, vec3f(-0.96, 0.91, -1.28), vec3f(0.96, 0.95, -0.88), 1.0, &best, n);
  tBox(o, d, inv, vec3f(0.955, 1.69, -1.69), vec3f(0.995, 1.95, -1.55), 3.0, &best, n);
  tBox(o, d, inv, vec3f(-0.995, 1.69, -1.69), vec3f(-0.955, 1.95, -1.55), 3.0, &best, n);
  cylY(o, d, vec2f(-0.555, 0.84), 0.045, 1.44, 2.36, 3.0, &best, n);
  // the front: bumper, bonnet and grille, the flat mudguards and their aprons, the lamps on the guards
  tBox(o, d, inv, vec3f(-0.99, 0.56, -2.79), vec3f(0.99, 0.76, -2.63), 3.0, &best, n);
  tBox(o, d, inv, vec3f(-0.6, 0.84, -2.66), vec3f(0.6, 1.5, -1.54), 1.0, &best, n);
  tBox(o, d, inv, vec3f(-0.5, 0.88, -2.69), vec3f(0.5, 1.38, -2.63), 3.0, &best, n);
  tBox(o, d, inv, vec3f(-0.99, 1.065, -2.38), vec3f(0.99, 1.135, -1.22), 1.0, &best, n);
  tBox(o, d, inv, vec3f(-0.99, 0.7, -2.385), vec3f(0.99, 1.1, -2.335), 1.0, &best, n);
  cylZ(o, d, vec2f(-0.76, 1.22), 0.085, -2.35, -2.21, 7.0, &best, n);
  cylZ(o, d, vec2f(0.76, 1.22), 0.085, -2.35, -2.21, 7.0, &best, n);
  // six wheels on three axles
  for (var k = 0; k < 3; k++) {
    let z = select(TRUCK_AXLE_R + f32(k - 1) * TRUCK_AXLE_GAP, TRUCK_AXLE_F, k == 0);
    cylX(o, d, vec2f(TRUCK_WHEEL_R, z), TRUCK_WHEEL_R, TRUCK_WHEEL_X - 0.17, TRUCK_WHEEL_X + 0.17, 2.0, &best, n);
    cylX(o, d, vec2f(TRUCK_WHEEL_R, z), TRUCK_WHEEL_R, -TRUCK_WHEEL_X - 0.17, -TRUCK_WHEEL_X + 0.17, 2.0, &best, n);
  }
  return best;
}

/** reference frame -> the fly's box: CAR's dimensions over TRUCK_BOX, so the model fits any box */
fn carScale() -> vec3f {
  return vec3f(scene.carDims.y, scene.carDims.z, scene.carDims.x) / TRUCK_BOX;
}

struct CarHit {
  t: f32,
  mat: f32,
  /** where the ray leaves the fly's box */
  exit: f32,
  /** world normal */
  n: vec3f,
};

/** The truck model (or a wreck) at pos, heading, along a world ray, inside the box the fly sees; t = NO_HIT on a miss */
fn traceCar(ro: vec3f, r: vec3f, pos: vec2f, heading: f32, burnt: bool, tMax: f32) -> CarHit {
  let lo = toCar(ro - vec3f(pos.x, 0.0, pos.y), heading);
  let ld = toCar(r, heading);
  let hi = vec3f(scene.carDims.y, scene.carDims.z, scene.carDims.x);
  let iv = boxInterval(lo, ld, vec3f(-hi.x, 0.0, -hi.z), hi);
  if (iv.y < max(iv.x, 0.0) || iv.x > tMax) { return CarHit(NO_HIT, 0.0, iv.y, vec3f(0.0, 1.0, 0.0)); }
  let s = carScale();
  var o = lo / s;
  if (burnt) { o.y += WRECK_SINK; }
  var n = vec3f(0.0, 1.0, 0.0);
  let h = truckTrace(o, ld / s, !burnt, &n);
  // a sunk wreck's lower edge is below the ground, where the ground covers it
  if (h.x >= NO_HIT || h.x > iv.y || lo.y + ld.y * h.x < 0.0) { return CarHit(NO_HIT, 0.0, iv.y, vec3f(0.0, 1.0, 0.0)); }
  return CarHit(h.x, h.y, iv.y, normalize(fromCar(n / s, heading)));
}

/** the wheel a reference-frame point belongs to, about its centre (+x outward) */
fn truckWheelLocal(q: vec3f) -> vec3f {
  let zr = q.z - TRUCK_AXLE_R - TRUCK_AXLE_GAP * clamp(round((q.z - TRUCK_AXLE_R) / TRUCK_AXLE_GAP), 0.0, 1.0);
  let zf = q.z - TRUCK_AXLE_F;
  return vec3f(abs(q.x) - TRUCK_WHEEL_X, q.y - TRUCK_WHEEL_R, select(zr, zf, abs(zf) < abs(zr)));
}

/** age < 0 for a live truck; >= 0 for a wreck that many seconds old. wheel: radians the wheels have turned. */
fn carColor(p: vec3f, n: vec3f, r: vec3f, pos: vec2f, heading: f32, mat: f32, age: f32, wheel: f32) -> vec3f {
  let burnt = age >= 0.0;
  let q = toCar(p - vec3f(pos.x, 0.0, pos.y), heading) / carScale();
  let ln = toCar(n, heading);
  // a tyre's flat outer face is its steel wheel inside the rim
  var m = mat;
  if (m == 2.0 && abs(ln.x) > 0.9 && length(truckWheelLocal(q).yz) < 0.28) { m = 4.0; }
  let fres = pow(1.0 - clamp(dot(n, -r), 0.0, 1.0), 5.0);
  let rf = reflect(r, n);
  // matte paint and canvas reflect little: the plain sky, without its clouds, is enough
  let env = select(vec3f(0.07, 0.065, 0.06), skyBase(rf), rf.y > 0.0);
  // olive drab, faded unevenly, flat
  let fade = noise3(q * vec3f(1.3, 2.1, 1.3));
  var alb = mix(vec3f(0.052, 0.058, 0.022), vec3f(0.078, 0.082, 0.038), fade);
  var spec = 0.015 + 0.2 * fres;
  var emit = vec3f(0.0);
  let front = ln.z < -0.5;
  let back = ln.z > 0.5;
  let side = abs(ln.x) > 0.6;
  // the tyres turn as far as the truck has driven (world.ts Car.wheel), so they stop when it stops
  let spin = select(0.0, wheel, !burnt);
  var mud = 1.0 - smoothstep(0.4, 1.3, q.y);
  if (m == 2.0) {
    // chunky off-road tread across the running surface, plainer sidewalls
    let wl = truckWheelLocal(q);
    let a = atan2(wl.y, wl.z) + spin;
    let lug = step(0.5, fract(a * 22.0 / 6.2832 + select(0.0, 0.5, wl.x > 0.0) + abs(wl.x) * 2.0));
    let tread = 1.0 - smoothstep(0.3, 0.6, abs(ln.x));
    alb = vec3f(0.022, 0.021, 0.02) * (1.0 - 0.45 * lug * tread);
    spec = 0.015;
    mud = 0.6 + 0.4 * noise3(q * 9.0);
  } else if (m == 4.0) {
    // the steel wheel: olive, darker in the middle, a ring of wheel nuts
    let wl = truckWheelLocal(q);
    let rho = length(wl.yz);
    let a = atan2(wl.y, wl.z) + spin;
    alb *= 1.15;
    if (rho > 0.24) { alb *= 0.5; } // the rim
    let nut = length(vec2f(rho - 0.19, (fract(a * 8.0 / 6.2832) - 0.5) * rho * 0.785)) < 0.022;
    if (nut) { alb = vec3f(0.03, 0.03, 0.025); }
    if (rho < 0.1) { alb *= 0.6; }
    spec = 0.05 + 0.3 * fres;
    mud *= 0.7;
  } else if (m == 3.0) {
    alb = vec3f(0.018, 0.018, 0.016);
    spec = 0.03;
    // the grille's slats
    if (front && abs(q.z + 2.69) < 0.02 && abs(q.x) < 0.48) {
      alb *= 0.4 + 0.9 * step(0.5, fract(q.x * 9.0));
    }
    // a stencilled unit marking on the bumper
    if (front && q.z < -2.75 && abs(q.y - 0.66) < 0.045 && abs(q.x) > 0.55 && abs(q.x) < 0.9 && fract(abs(q.x) * 18.0) < 0.5) {
      alb = vec3f(0.25, 0.25, 0.22);
    }
  } else if (m == 5.0) {
    // canvas: a paler, browner olive, creased along the bows, streaked where rain ran off
    let zc = q.z - TRUCK_BED_Z;
    let bz = zc - TRUCK_BOW * round(zc / TRUCK_BOW);
    alb = vec3f(0.078, 0.078, 0.042) * (0.85 + 0.3 * noise3(q * vec3f(3.0, 1.0, 3.0)));
    alb *= 0.8 + 0.2 * smoothstep(0.0, 0.12, abs(bz));
    // sagging between the bows: taut and lit over each bow, slack and shadowed between them
    alb *= 0.86 + 0.14 * cos(bz * 6.2832 / TRUCK_BOW);
    alb *= 1.0 - 0.12 * smoothstep(0.55, 0.8, noise3(vec3f(q.z * 5.0, q.y * 0.6, q.x * 5.0)));
    // tie-down ropes along the lower edge
    if (side && q.y < 1.72 && fract(q.z * 2.5) < 0.08) { alb *= 0.6; }
    spec = 0.01;
    mud *= 0.3;
  } else if (m == 7.0) {
    alb = vec3f(0.3);
    // dimmed headlights, on in the gloom
    emit = select(vec3f(0.0), vec3f(1.5, 1.3, 0.9), front);
  } else {
    // glass: the windscreen in two panes, and the door windows
    let wind = front && q.z < -1.5 && q.y > 1.62 && q.y < 2.02 && abs(q.x) < 0.86 && abs(q.x) > 0.04;
    let door = side && q.z > -1.48 && q.z < -0.82 && q.y > 1.6 && q.y < 2.0;
    if (wind || door) {
      alb = vec3f(0.008, 0.01, 0.012);
      spec = 0.06 + 0.9 * fres;
      mud = 0.0;
    } else if (back && q.z > 2.7 && q.y > 1.08 && q.y < 1.2 && abs(q.x) > 0.72 && abs(q.x) < 0.92) {
      alb = vec3f(0.2, 0.01, 0.01);
      emit = vec3f(0.7, 0.03, 0.02); // taillights
    } else if (side && q.z > -1.5 && q.z < -0.66) {
      // the door: its seams (no national insignia: these trucks belong to no real army)
      if (abs(q.z + 1.5) < 0.01 || abs(q.z + 0.66) < 0.01 || abs(q.y - 1.0) < 0.01) { alb *= 0.3; }
    } else if (side && q.z > -0.5 && q.y > 1.08 && q.y < 1.62) {
      // the bed's slatted sides
      alb *= 0.75 + 0.35 * step(0.18, fract((q.y - 1.08) * 7.0));
    }
  }
  // mud and dust up from the ground
  let dirt = mud * (0.45 + 0.55 * noise3(q * vec3f(6.0, 12.0, 6.0)));
  alb = mix(alb, vec3f(0.095, 0.08, 0.058), clamp(dirt, 0.0, 1.0) * 0.75);
  if (burnt) {
    // charred black and rust, glass gone, embers in the paint for the first seconds
    let n1 = noise3(q * 3.0);
    alb = mix(vec3f(0.012, 0.011, 0.01), vec3f(0.09, 0.035, 0.012), smoothstep(0.5, 0.75, n1));
    spec = 0.01;
    emit = vec3f(0.0);
    // flames die down over the first seconds; embers keep glowing while the wreck stands (until the trucks respawn)
    let burn = max(exp(-age / 3.5), 0.15);
    let fl = fbm3(vec3f(q.x * 5.0, q.y * 6.0 - scene.time * 6.0, q.z * 5.0));
    emit = vec3f(1.2, 0.3, 0.05) * burn * smoothstep(0.72, 0.85, fl) * 0.6;
  }
  let ao = 0.5 + 0.5 * smoothstep(0.3, 1.2, q.y);
  var col = lit(alb, n, ao, 1.0) * (1.0 - spec * 0.5) + env * spec;
  return col + emit;
}

// ---- trees: dead, charred trunks ---------------------------------------------------------------------------

/** where a trunk's broken top is (at most a metre below the fly's flat top), by direction around it */
fn trunkCut(d: vec2f, tr: vec4f) -> f32 {
  let a = d / max(length(d), 1e-5);
  let n = pnoise(a * 1.7 + tr.xy * 0.37 + 50.0, 1e4);
  let n2 = pnoise(a * 4.5 + tr.xy * 0.71 + 90.0, 1e4);
  let h = hash12(floor(tr.xy * 3.1));
  return tr.w - 0.05 - (0.15 + 0.45 * h) * (0.3 + 0.7 * n) - 0.25 * n2;
}

/**
 * The broken top as seen from above: at the rim it is trunkCut (the side silhouette), and it sinks a little toward
 * the middle. The fly's trunk is a flat-capped cylinder at tr.w; this surface is everywhere below it.
 */
fn trunkTopH(d: vec2f, tr: vec4f) -> f32 {
  return trunkCut(d, tr) - 0.18 * (1.0 - clamp(length(d) / tr.z, 0.0, 1.0));
}

/** the ray's distance from xz offset d (inside or on a vertical cylinder of radius rad) to where it leaves it */
fn cylExit(d: vec2f, rd: vec2f, rad: f32) -> f32 {
  let a = dot(rd, rd);
  if (a < 1e-8) { return 1e3; }
  let b = dot(d, rd);
  return (-b + sqrt(max(b * b - a * (dot(d, d) - rad * rad), 0.0))) / a;
}

/**
 * A ray that met the fly's trunk above the view's broken top: march it down inside the trunk to the top surface
 * (trunkTopH). (t from p, normal), or t = -1 if it leaves the trunk first; tExit is where it leaves.
 */
fn trunkTopHit(p: vec3f, r: vec3f, d: vec2f, tr: vec4f, tExit: f32) -> vec4f {
  var ta = 0.0;
  var tb = -1.0;
  for (var i = 1; i <= 40; i++) {
    let t = tExit * f32(i) / 40.0;
    if ((p + r * t).y <= trunkTopH(d + r.xz * t, tr)) {
      tb = t;
      break;
    }
    ta = t;
  }
  if (tb < 0.0) { return vec4f(-1.0); }
  for (var k = 0; k < 6; k++) {
    let tm = 0.5 * (ta + tb);
    if ((p + r * tm).y <= trunkTopH(d + r.xz * tm, tr)) { tb = tm; } else { ta = tm; }
  }
  let e = 0.01;
  let dd = d + r.xz * tb;
  let gx = (trunkTopH(dd + vec2f(e, 0.0), tr) - trunkTopH(dd - vec2f(e, 0.0), tr)) / (2.0 * e);
  let gz = (trunkTopH(dd + vec2f(0.0, e), tr) - trunkTopH(dd - vec2f(0.0, e), tr)) / (2.0 * e);
  return vec4f(tb, normalize(vec3f(-gx, 1.0, -gz)));
}

fn trunkColor(p: vec3f, n: vec3f, r: vec3f, i: u32) -> vec3f {
  let tr = scene.trees[i];
  let c = nearestImage(tr.xy, p.xz);
  let d = p.xz - c;
  let seed = hash12(floor(tr.xy * 5.3));
  // alligator-cracked charcoal, 13 blocks around the trunk
  let ang = atan2(d.y, d.x) / 6.2832 + 0.5;
  let uv = vec2f(ang * 13.0, p.y * 4.6 + seed * 20.0);
  let v = voronoiX(uv, 13.0);
  let crack = 1.0 - smoothstep(0.03, 0.12, v.y - v.x);
  let block = hash12(floor(uv * 1.0) + seed);
  var alb = vec3f(0.02, 0.018, 0.016) * (0.6 + 0.8 * block);
  // ash-grey weathering, and a little brown bark left high up
  let an = noise3(vec3f(d.x * 5.0, p.y * 0.9, d.y * 5.0) + seed * 30.0);
  alb = mix(alb, vec3f(0.15, 0.145, 0.14), smoothstep(0.55, 0.85, an) * 0.55);
  alb = mix(alb, vec3f(0.07, 0.045, 0.03), smoothstep(5.0, 10.0, p.y + (an - 0.5) * 4.0) * 0.45);
  alb = mix(alb, vec3f(0.003), crack);
  // the splintered top is lighter wood
  let cut = trunkCut(d, tr);
  alb = mix(alb, vec3f(0.09, 0.07, 0.05), smoothstep(cut - 0.25, cut, p.y) * 0.6);
  // the broken top from above: charred wood with growth rings, blacker toward the rim
  let top = smoothstep(0.35, 0.75, n.y);
  if (top > 0.0) {
    let rho = length(d) / tr.z;
    let wood = vec3f(0.07, 0.05, 0.035) * (0.75 + 0.25 * sin(rho * 55.0 + an * 6.0)) * (0.6 + 0.8 * an);
    alb = mix(alb, mix(wood, vec3f(0.012), smoothstep(0.55, 0.95, rho) * 0.8), top);
  }
  let ao = (0.45 + 0.55 * smoothstep(0.0, 1.5, p.y)) * (1.0 - 0.5 * crack);
  var col = lit(alb, n, ao, 1.0);
  // charcoal has a faint silvery sheen
  col += vec3f(0.05) * pow(max(dot(reflect(r, n), SUN), 0.0), 6.0) * (1.0 - crack) * block;
  return col;
}

// ---- fallen logs: charred, lying on the ground, one in about every other 10 m cell -------------------------

struct FallenLog {
  ok: bool,
  ctr: vec2f,
  dv: vec2f,
  halfL: f32,
  rad: f32,
};

/** the log of a 10 m cell, if it has one. Centred in the middle fifth of the cell and at most 3.54 long each way
 * (radius included), it never leaves its cell, so a ray only has to test the cells it crosses. */
fn fallenLog(cell: vec2f) -> FallenLog {
  let h = hash41(dot(wrapCell(cell, 10.0), vec2f(1.0, 13.0)) + 7.7);
  let a = h.w * 6.2832;
  return FallenLog(h.x < 0.55, (cell + 0.4 + 0.2 * h.yz) * 10.0, vec2f(cos(a), sin(a)),
    1.2 + 2.0 * fract(h.x * 7.3), 0.16 + 0.18 * fract(h.y * 5.1));
}

/** no log lies on or near the road. roadDistance walks the whole loop, so ask only for a log that matters. */
fn logOffRoad(lg: FallenLog) -> bool {
  return roadDistance(lg.ctr) - scene.roadInfo.y * 0.5 > lg.halfL + lg.rad + 0.3;
}

/** iq's ray / capped cylinder: (t, normal), or t = -1. rd normalized. */
fn iCylinder(ro: vec3f, rd: vec3f, pa: vec3f, pb: vec3f, ra: f32) -> vec4f {
  let ba = pb - pa;
  let oc = ro - pa;
  let baba = dot(ba, ba);
  let bard = dot(ba, rd);
  let baoc = dot(ba, oc);
  let k2 = baba - bard * bard;
  let k1 = baba * dot(oc, rd) - baoc * bard;
  let k0 = baba * dot(oc, oc) - baoc * baoc - ra * ra * baba;
  var h = k1 * k1 - k2 * k0;
  if (h < 0.0) { return vec4f(-1.0); }
  h = sqrt(h);
  var t = (-k1 - h) / k2;
  let y = baoc + t * bard;
  if (y > 0.0 && y < baba) { return vec4f(t, (oc + t * rd - ba * y / baba) / ra); }
  t = (select(baba, 0.0, y < 0.0) - baoc) / bard;
  if (abs(k1 + k2 * t) < h) { return vec4f(t, ba * sign(y) / sqrt(baba)); }
  return vec4f(-1.0);
}

/** a log sinks a little into the ash: its axis at this fraction of its radius */
const LOG_SINK = 0.85;
/** no log reaches above this */
const LOG_TOP = 0.7;
/** a Surf kind of the view alone (scene.wgsl has the fly's) */
const KIND_LOG = 20.0;

/** the nearest fallen log along the ray before tMax: (t, normal), or t = -1. Walks the 10 m cells the ray crosses
 * while it is below LOG_TOP (at most 10 of them; the fog hides anything further). */
fn traceLogs(ro: vec3f, r: vec3f, tMax: f32) -> vec4f {
  var t0 = 0.0;
  var t1 = min(tMax, 60.0);
  if (ro.y > LOG_TOP) {
    if (r.y >= -1e-4) { return vec4f(-1.0); }
    t0 = (LOG_TOP - ro.y) / r.y;
  }
  if (r.y < -1e-4) { t1 = min(t1, -ro.y / r.y + 0.01); } else if (r.y > 1e-4) { t1 = min(t1, (LOG_TOP - ro.y) / r.y); }
  if (t0 >= t1) { return vec4f(-1.0); }
  let p0 = (ro + r * t0).xz;
  var cell = floor(p0 / 10.0);
  let inv = 1.0 / max(abs(r.xz), vec2f(1e-6));
  var tNext = t0 + select(p0 - cell * 10.0, (cell + 1.0) * 10.0 - p0, r.xz > vec2f(0.0)) * inv;
  let tStep = 10.0 * inv;
  let stepC = sign(r.xz);
  for (var i = 0; i < 10; i++) {
    let lg = fallenLog(cell);
    if (lg.ok) {
      let c = vec3f(lg.ctr.x, lg.rad * LOG_SINK, lg.ctr.y);
      let a = vec3f(lg.dv.x, 0.0, lg.dv.y) * lg.halfL;
      let h = iCylinder(ro, r, c - a, c + a, lg.rad);
      if (h.x > 0.0 && h.x < t1 && logOffRoad(lg)) { return h; }
    }
    if (min(tNext.x, tNext.y) >= t1) { break; }
    if (tNext.x < tNext.y) {
      cell.x += stepC.x;
      tNext.x += tStep.x;
    } else {
      cell.y += stepC.y;
      tNext.y += tStep.y;
    }
  }
  return vec4f(-1.0);
}

/** charred bark with cracks and ash on top; the ends show paler wood rings with a burnt rim */
fn logColor(p: vec3f, n: vec3f) -> vec3f {
  let lg = fallenLog(floor(p.xz / 10.0));
  let q = p.xz - lg.ctr;
  let along = dot(q, lg.dv);
  let across = dot(q, vec2f(-lg.dv.y, lg.dv.x));
  let up = p.y - lg.rad * LOG_SINK;
  let ang = atan2(up, across);
  let bark = pnoise(vec2f(along * 2.0, ang * 1.2 + 7.0), 1e4) * 0.6 + pnoise(vec2f(along * 7.0, ang * 3.0), 1e4) * 0.4;
  let crack = (1.0 - smoothstep(0.0, 0.05, abs(fract(along * 0.9 + bark * 0.4) - 0.5))) * step(0.45, bark);
  var alb = mix(vec3f(0.014, 0.012, 0.011) * (0.6 + 0.8 * bark), vec3f(0.003), crack);
  alb = mix(alb, vec3f(0.09, 0.087, 0.084), smoothstep(0.7, 0.98, n.y) * 0.3 * (1.0 - crack));
  if (abs(along) > lg.halfL - 0.003) {
    let rho = length(vec2f(across, up)) / lg.rad;
    alb = mix(vec3f(0.06, 0.045, 0.03) * (0.7 + 0.3 * sin(rho * 18.0)), vec3f(0.01), smoothstep(0.7, 0.95, rho));
  }
  let ao = 0.45 + 0.55 * smoothstep(0.0, lg.rad * 1.6, p.y);
  return lit(alb, n, ao, 1.0) + fireLight(p, n) * alb;
}

// ---- the forest floor: ash, burnt soil, debris, all flat texture ------------------------------------------

fn soilAlbedo(p: vec2f, fp: f32) -> vec3f {
  let n1 = fbmP(p, 0.04, 3);
  let n2 = fbmP(p + vec2f(31.7, 11.3), 0.16, 3);
  var c = mix(vec3f(0.022, 0.018, 0.015), vec3f(0.048, 0.037, 0.027), n2);
  let ash = smoothstep(0.55, 0.85, n1 + 0.3 * (n2 - 0.5));
  c = mix(c, vec3f(0.15, 0.145, 0.14) * (0.75 + 0.5 * n2), ash * 0.85);
  let charP = smoothstep(0.56, 0.7, fbmP(p + vec2f(7.1, 53.9), 0.12, 2));
  c = mix(c, vec3f(0.008, 0.0075, 0.007), charP * 0.9);
  // burnt stubble and ash grain, gone where a pixel covers it
  let w1 = 1.0 - smoothstep(0.12, 0.5, fp * 3.0);
  let w2 = 1.0 - smoothstep(0.12, 0.5, fp * 12.0);
  if (w1 > 0.0) {
    c *= 1.0 + (pnoise(p * 3.0, 300.0) - 0.5) * 0.7 * w1;
  }
  if (w2 > 0.0) {
    c *= 1.0 + (pnoise(p * 12.0 + 3.7, 1200.0) - 0.5) * 0.8 * w2;
  }
  return c;
}

/** twigs, stones, the logs' shadows and embers painted onto the ground (flat texture: nothing here can be hit) */
fn debris(p: vec2f, fp: f32, cIn: vec3f, isRoad: bool) -> vec3f {
  var c = cIn;
  // charred twigs, one per half-metre cell at most
  if (fp < 0.08) {
    let cell = floor(p / 0.5);
    let h = hash41(dot(wrapCell(cell, 200.0), vec2f(1.0, 211.0)) + 0.5);
    if (h.x < 0.18) {
      let ctr = (cell + 0.25 + 0.5 * h.yz) * 0.5;
      let a = h.w * 6.2832;
      let dv = vec2f(cos(a), sin(a));
      let len = 0.06 + 0.5 * h.x;
      let q = p - ctr;
      let along = clamp(dot(q, dv), -len, len);
      let dist = length(q - dv * along);
      let wd = 0.01 + 0.012 * h.y;
      let m = 1.0 - smoothstep(wd, wd + max(fp, 0.003), dist);
      c = mix(c, vec3f(0.01), m * 0.7 * (1.0 - smoothstep(0.02, 0.05, fp)));
    }
  }
  // stones, with a lit top and a shadow on the far side
  if (fp < 0.15) {
    let cell = floor(p / 1.25);
    let h = hash41(dot(wrapCell(cell, 80.0), vec2f(1.0, 97.0)) + 3.3);
    if (h.x < 0.15) {
      let rad = 0.05 + 0.12 * h.w;
      let ctr = (cell + 0.2 + 0.6 * h.yz) * 1.25;
      let q = (p - ctr) / rad;
      let d = length(q);
      let sh = length(q + SUN.xz * 0.6);
      let fade = 1.0 - smoothstep(0.05, 0.15, fp / rad * 0.1);
      c *= 1.0 - 0.55 * (1.0 - smoothstep(0.9, 1.3, sh)) * fade;
      if (d < 1.0) {
        let top = sqrt(1.0 - d * d);
        let lum = 0.5 + 0.5 * dot(normalize(vec3f(q.x, top * 1.5, q.y)), SUN);
        c = mix(c, vec3f(0.06, 0.056, 0.052) * (0.5 + 0.8 * h.y) * lum, fade * (1.0 - smoothstep(0.85, 1.0, d)));
      }
    }
  }
  // the fallen logs' shadows, off the sun's side (the logs themselves are solid: traceLogs)
  if (!isRoad) {
    let lg = fallenLog(floor(p / 10.0));
    if (lg.ok) {
      let side = vec2f(-lg.dv.y, lg.dv.x);
      let q = p - lg.ctr;
      let endT = max(abs(dot(q, lg.dv)) - lg.halfL, 0.0);
      let su = length(vec2f(dot(q, side) + dot(SUN.xz, side) * lg.rad * 1.3, endT)) / lg.rad;
      if (su < 1.6 && logOffRoad(lg)) { c *= 1.0 - 0.6 * (1.0 - smoothstep(0.8, 1.6, su)); }
    }
  }
  // embers still glowing here and there
  if (fp < 0.04) {
    let cell = floor(p / 2.0);
    let h = hash41(dot(wrapCell(cell, 50.0), vec2f(1.0, 57.0)) + 11.1);
    if (h.x < 0.07) {
      let ctr = (cell + 0.2 + 0.6 * h.yz) * 2.0;
      let d = length(p - ctr);
      let flick = 0.6 + 0.4 * sin(view.time * (3.0 + 5.0 * h.w) + h.y * 40.0);
      gEmit += vec3f(1.6, 0.35, 0.05) * flick * (1.0 - smoothstep(0.015, 0.05, d)) * (1.0 - smoothstep(0.015, 0.04, fp));
      c = mix(c, vec3f(0.01), 1.0 - smoothstep(0.04, 0.12, d));
    }
  }
  return c;
}

/** a dirt track: packed earth, two ruts per lane, loose ash at the crown, fading into the burnt verge */
fn roadAlbedo(p: vec2f, rf: vec3f, fp: f32, soil: vec3f) -> vec3f {
  let lat = rf.y;
  let halfW = scene.roadInfo.y * 0.5;
  let n = pnoise(p * 0.7, 70.0);
  var c = vec3f(0.105, 0.088, 0.068) * (0.8 + 0.4 * n);
  let w1 = 1.0 - smoothstep(0.1, 0.4, fp * 6.0);
  if (w1 > 0.0) {
    c *= 1.0 + (pnoise(p * 6.0, 600.0) - 0.5) * 0.5 * w1;
    // gravel
    let gc = floor(p * 6.0);
    let gh = hash22(wrapCell(gc, 600.0));
    let gd = length(fract(p * 6.0) - 0.25 - 0.5 * gh);
    let stone = (1.0 - smoothstep(0.1, 0.2, gd)) * step(0.55, gh.x);
    c = mix(c, vec3f(0.07, 0.065, 0.058) * (0.7 + 0.6 * gh.y), stone * w1 * 0.8);
  }
  // wheel ruts: the car's wheels run 0.8 either side of its lane's centre
  let rut = min(abs(abs(lat) - (CAR_LANE - 0.8)), abs(abs(lat) - (CAR_LANE + 0.8)));
  let rutM = 1.0 - smoothstep(0.1, 0.32, rut + (pnoise(p * 1.3, 130.0) - 0.5) * 0.12);
  c *= 1.0 - 0.3 * rutM;
  c += vec3f(0.01) * rutM * step(0.5, fract(rf.z * 7.0)) * w1; // tread marks
  // potholes with a skin of dark water
  let ph = pnoise(p * 0.45 + 40.0, 45.0);
  let hole = smoothstep(0.78, 0.84, ph);
  c = mix(c, vec3f(0.02, 0.022, 0.024), hole * 0.8);
  // the verge: ash and soil creep in
  let edge = smoothstep(halfW - 1.0, halfW, abs(lat) + (pnoise(p * 1.1, 110.0) - 0.5) * 0.8);
  c = mix(c, soil, edge * 0.85);
  return c;
}

// ---- ambient light near things ---------------------------------------------------------------------------

/** ambient occlusion and soft shadow on the ground from trunks, the car, wrecks and the drone */
fn groundOcclusion(p: vec3f) -> f32 {
  var ao = 1.0;
  let nt = min(u32(scene.treeCount), MAX_TREES);
  for (var i = 0u; i < nt; i++) {
    let tr = scene.trees[i];
    let c = nearestImage(tr.xy, p.xz);
    let d = length(p.xz - c) - tr.z;
    // the shadow reaches about a trunk's height from its foot (the sun is ~45 deg up)
    if (d > tr.w * 1.05 + 1.0) { continue; }
    ao *= 0.55 + 0.45 * smoothstep(0.0, 1.4, d);
    // a long faint shadow away from the hazy sun
    let s = raySeg(p, SUN, vec3f(c.x, 0.0, c.y), vec3f(c.x, tr.w, c.y));
    ao *= 1.0 - 0.3 * (1.0 - smoothstep(tr.z * 0.6, tr.z * 0.6 + 0.25 + 0.03 * s.y, s.x));
  }
  let ncars = carCount();
  for (var i = 0u; i < ncars; i++) {
    let car = scene.cars[i];
    if (car.w <= 0.0) { continue; }
    let c = nearestImage(car.xy, p.xz);
    // beyond the footprint's corner plus the fade, the factor below is exactly 1
    if (distance(p.xz, c) > length(scene.carDims.xy) + 1.2) { continue; }
    let q = toCar(p - vec3f(c.x, 0.0, c.y), car.z);
    let b = abs(q.xz) - vec2f(scene.carDims.y * 0.9, scene.carDims.x * 0.95);
    let d = length(max(b, vec2f(0.0))) + min(max(b.x, b.y), 0.0);
    ao *= 0.3 + 0.7 * smoothstep(-0.4, 1.1, d);
  }
  let nw = min(u32(view.fx.w), MAX_WRECKS);
  for (var i = 0u; i < nw; i++) {
    let w = view.wrecks[i];
    let c = nearestImage(w.xy, p.xz);
    let q = toCar(p - vec3f(c.x, 0.0, c.y), w.z);
    let b = abs(q.xz) - vec2f(scene.carDims.y, scene.carDims.x);
    let d = length(max(b, vec2f(0.0))) + min(max(b.x, b.y), 0.0);
    ao *= 0.25 + 0.75 * smoothstep(-0.4, 1.4, d);
  }
  if (view.fpv.z > 0.5) {
    let c = nearestImage(view.drone.xz, p.xz);
    let d = distance(p.xz, c);
    let h = max(view.drone.y, 0.1);
    ao *= 1.0 - 0.4 * exp(-d * d / (0.25 + 0.05 * h * h)) / (1.0 + 0.08 * h);
  }
  return ao;
}

/** light from blasts and burning wrecks */
fn fireLight(p: vec3f, n: vec3f) -> vec3f {
  var l = vec3f(0.0);
  let ne = min(u32(view.fx.y), MAX_EXPLOSIONS);
  for (var i = 0u; i < ne; i++) {
    let e = view.explosions[2u * i];
    let age = view.explosions[2u * i + 1u].x;
    let c = nearestImage(e.xz, p.xz);
    let v = vec3f(c.x, e.y + e.w * 0.3, c.y) - p;
    let d2 = dot(v, v);
    let k = e.w * e.w * 6.0 * exp(-age * 4.0) * (1.0 + 2.0 * exp(-age * 30.0));
    l += vec3f(1.0, 0.55, 0.22) * k * (0.25 + 0.75 * max(dot(n, v * inverseSqrt(d2 + 1e-4)), 0.0)) / (d2 + e.w);
  }
  let nw = min(u32(view.fx.w), MAX_WRECKS);
  for (var i = 0u; i < nw; i++) {
    let w = view.wrecks[i];
    let c = nearestImage(w.xy, p.xz);
    let v = vec3f(c.x, 1.3, c.y) - p;
    let d2 = dot(v, v);
    let flick = 0.8 + 0.2 * sin(view.time * 13.0 + f32(i)) * sin(view.time * 7.3);
    let k = 3.0 * max(exp(-w.w / 3.5), 0.05) * flick;
    l += vec3f(1.0, 0.45, 0.15) * k * (0.3 + 0.7 * max(dot(n, v * inverseSqrt(d2 + 1e-4)), 0.0)) / (d2 + 1.0);
  }
  return l;
}

// ---- the drone (chase view only) ---------------------------------------------------------------------------

/** world -> drone body frame: undo heading, then pitch, then bank */
fn toDrone(p: vec3f) -> vec3f {
  let h = view.droneHeading;
  let c = cos(h);
  let s = sin(h);
  let a = vec3f(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
  let cp = cos(view.dronePitch);
  let sp = sin(view.dronePitch);
  let b = vec3f(a.x, a.y * cp + a.z * sp, -a.y * sp + a.z * cp);
  let cb = cos(view.bank);
  let sb = sin(view.bank);
  return vec3f(b.x * cb - b.y * sb, b.x * sb + b.y * cb, b.z);
}

/** a bent wire: a cubic from a to d, as six capsules of radius rad */
fn wire(q: vec3f, a: vec3f, b: vec3f, c: vec3f, d: vec3f, rad: f32) -> f32 {
  var best = 1e3;
  var prev = a;
  for (var i = 1; i <= 6; i++) {
    let t = f32(i) / 6.0;
    let u = 1.0 - t;
    let pt = a * (u * u * u) + b * (3.0 * u * u * t) + c * (3.0 * u * t * t) + d * (t * t * t);
    best = min(best, sdCapsule(q, prev, pt, rad));
    prev = pt;
  }
  return best;
}

/** a point of wireSpiral's curve, theta radians along it */
fn spiralAt(c: vec3f, a0: f32, dir: f32, r0: f32, r1: f32, total: f32, theta: f32) -> vec3f {
  let a = a0 + dir * theta;
  return c + vec3f(cos(a), 0.0, sin(a)) * mix(r0, r1, theta / total);
}

/**
 * A flat open spiral of wire around c: it starts at angle a0 (atan2(z, x)) at radius r0 and winds `turns` times
 * inward to r1, counter-clockwise from above for dir = -1. The distance is radial to the nearest turn, which is
 * close for a spiral this loose.
 */
fn wireSpiral(q: vec3f, c: vec3f, a0: f32, dir: f32, r0: f32, r1: f32, turns: f32, rad: f32) -> f32 {
  let p = q - c;
  let rho = length(p.xz);
  let total = turns * 6.2831853;
  let u = fract(dir * (atan2(p.z, p.x) - a0) / 6.2831853) * 6.2831853;
  var best = min(length(q - spiralAt(c, a0, dir, r0, r1, total, 0.0)), length(q - spiralAt(c, a0, dir, r0, r1, total, total)));
  for (var n = 0; n < 2; n++) {
    let theta = u + 6.2831853 * f32(n);
    if (theta <= total) { best = min(best, length(vec2f(rho - mix(r0, r1, theta / total), p.y))); }
  }
  return best - rad;
}

/** a capped cone along z: radius ra at z = za, rb at z = zb (iq's capped cone, turned onto z) */
fn sdConeZ(q: vec3f, za: f32, zb: f32, ra: f32, rb: f32) -> f32 {
  let h = 0.5 * (zb - za);
  let p = vec2f(length(q.xy), q.z - 0.5 * (za + zb));
  let k1 = vec2f(rb, h);
  let k2 = vec2f(rb - ra, 2.0 * h);
  let ca = vec2f(p.x - min(p.x, select(rb, ra, p.y < 0.0)), abs(p.y) - h);
  let cb = p - k1 + k2 * clamp(dot(k1 - p, k2) / dot(k2, k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

/** the warhead's axis, under the frame */
const WH_Y = -0.13;
/** radius of the warhead's threaded tail boss */
const WH_TAIL_R = 0.058;

/** the warhead, nose forward: a cone widening back to a round shoulder and a short tail boss, the nose open */
fn warheadSdf(q: vec3f) -> f32 {
  let w = q - vec3f(0.0, WH_Y, 0.0);
  var d = sdConeZ(w, -0.44, 0.08, 0.034, 0.1);
  d = smin(d, sdEllipsoid(w - vec3f(0.0, 0.0, 0.09), vec3f(0.1, 0.1, 0.15)), 0.02);
  // the tail: a threaded steel boss out of the rounded back (z 0.187-0.277), and the fuze plug in its end plate
  d = min(d, sdCylY(vec3f(w.x, w.z - 0.232, w.y), WH_TAIL_R, 0.045));
  d = min(d, sdCylY(vec3f(w.x, w.z - 0.285, w.y), 0.012, 0.009));
  // the fuze socket in the nose, hollow
  let hole = vec2f(length(w.xy) - 0.022, abs(w.z + 0.45) - 0.07);
  return max(d, -(min(max(hole.x, hole.y), 0.0) + length(max(hole, vec2f(0.0)))));
}

/**
 * The contact fuze: two electrodes of stiff steel wire. Each is a straight rod from a clamp under a front arm,
 * two-thirds of the way out, to a big open ring on the other side: the rods cross in an X ahead of the nose, and
 * the rings lie well out in front, each banked outward (the left one tips to the left, the right one to the
 * right). Impact presses the rings together. The right ring sits a little lower, so the rods pass each other at
 * the X without touching.
 */
const FUZE_ARM = vec3f(0.362, -0.012, -0.343);
const FUZE_RING_L = vec3f(-0.28, -0.24, -0.9);
const FUZE_RING_R = vec3f(0.28, -0.255, -0.9);
/** rad each ring banks outward about the forward axis */
const FUZE_TILT = 0.45;
const FUZE_R0 = 0.12;
const FUZE_R1 = 0.108;
/** each ring is open: 320 of 360 degrees, from its inboard point forward, round the outside, and back */
const FUZE_TURNS = 0.89;
const FUZE_WIRE = 0.0035;
/** q -> the frame of a ring at c banked by `bank` about the forward axis (positive: its left edge down) */
fn ringFrame(q: vec3f, c: vec3f, bank: f32) -> vec3f {
  let p = q - c;
  let cb = cos(bank);
  let sb = sin(bank);
  return c + vec3f(p.x * cb + p.y * sb, -p.x * sb + p.y * cb, p.z);
}
fn contactWires(q: vec3f) -> f32 {
  let ct = cos(FUZE_TILT);
  let st = sin(FUZE_TILT);
  // left arm -> right ring (banked right), entering at its inboard point
  var d = sdCapsule(q, FUZE_ARM * vec3f(-1.0, 1.0, 1.0), FUZE_RING_R + vec3f(-FUZE_R0 * ct, FUZE_R0 * st, 0.0), FUZE_WIRE);
  d = min(d, wireSpiral(ringFrame(q, FUZE_RING_R, -FUZE_TILT), FUZE_RING_R, 3.1415927, 1.0, FUZE_R0, FUZE_R1, FUZE_TURNS, FUZE_WIRE));
  // right arm -> left ring (banked left)
  d = min(d, sdCapsule(q, FUZE_ARM, FUZE_RING_L + vec3f(FUZE_R0 * ct, FUZE_R0 * st, 0.0), FUZE_WIRE));
  d = min(d, wireSpiral(ringFrame(q, FUZE_RING_L, FUZE_TILT), FUZE_RING_L, 0.0, -1.0, FUZE_R0, FUZE_R1, FUZE_TURNS, FUZE_WIRE));
  return d;
}

/**
 * (distance, material) in the body frame (+x right, +y up, -z forward): a long-range FPV strike drone. Materials:
 * 1 carbon, 2 flight controller, 3 battery, 4 strap, 5 motor, 6 prop nut, 7 GPS puck, 8 camera pod, 9 cable,
 * 10 warhead, 11 metal, 12 arming box, 13 red lead, 14 blue lead, 15 grenade, 16 steel wire, 17 plug.
 */
fn droneSdf(q: vec3f) -> vec2f {
  // bottom and top plates, standoffs between them, the flight controller stack
  var res = vec2f(sdRoundBox(q, vec3f(0.05, 0.006, 0.17), 0.004), 1.0);
  res = opU(res, sdRoundBox(q - vec3f(0.0, 0.05, 0.0), vec3f(0.045, 0.005, 0.15), 0.004), 1.0);
  let a = vec3f(abs(q.x), q.y, abs(q.z));
  res = opU(res, sdCylY(a - vec3f(0.035, 0.025, 0.12), 0.006, 0.022), 11.0);
  res = opU(res, sdRoundBox(q - vec3f(0.0, 0.022, 0.0), vec3f(0.032, 0.01, 0.032), 0.002), 2.0);
  // X arms: flat carbon bars, motor mounts, motors, prop nuts (one quadrant, mirrored)
  let end = vec3f(0.5, 0.0, 0.46);
  let s0 = vec2f(0.04, 0.07);
  let ab = end.xz - s0;
  let h = clamp(dot(a.xz - s0, ab) / dot(ab, ab), 0.0, 1.0);
  let arm = vec2f(length(a.xz - s0 - ab * h) - 0.02, abs(q.y) - 0.008);
  res = opU(res, min(max(arm.x, arm.y), 0.0) + length(max(arm, vec2f(0.0))), 1.0);
  res = opU(res, sdCylY(a - end, 0.056, 0.007), 1.0);
  res = opU(res, sdCylY(a - end - vec3f(0.0, 0.043, 0.0), 0.034, 0.036), 5.0);
  res = opU(res, sdCylY(a - end - vec3f(0.0, 0.105, 0.0), 0.013, 0.013), 6.0);
  // the battery: a black-wrapped pack on the top plate, two straps round it, its lead and yellow plug
  res = opU(res, sdRoundBox(q - vec3f(0.0, 0.112, 0.03), vec3f(0.055, 0.05, 0.12), 0.02), 3.0);
  res = opU(res, sdRoundBox(vec3f(q.x, q.y - 0.112, abs(q.z - 0.03) - 0.06), vec3f(0.059, 0.054, 0.009), 0.004), 4.0);
  res = opU(res, sdRoundBox(q - vec3f(0.05, 0.1, 0.17), vec3f(0.011, 0.009, 0.016), 0.002), 17.0);
  res = opU(res, wire(q, vec3f(0.02, 0.15, 0.14), vec3f(0.03, 0.16, 0.19), vec3f(0.06, 0.13, 0.2), vec3f(0.05, 0.1, 0.185), 0.005), 13.0);
  // the GPS puck on its mast, and the video antenna, at the back
  res = opU(res, wire(q, vec3f(-0.03, 0.055, 0.14), vec3f(-0.05, 0.12, 0.15), vec3f(-0.07, 0.25, 0.2), vec3f(-0.08, 0.3, 0.21), 0.004), 9.0);
  res = opU(res, sdEllipsoid(q - vec3f(-0.08, 0.315, 0.21), vec3f(0.035, 0.016, 0.035)), 7.0);
  res = opU(res, sdCapsule(q, vec3f(0.04, 0.03, 0.15), vec3f(0.19, 0.05, 0.25), 0.003), 9.0);
  // the camera pod the fly rides in, between two side plates at the front
  res = opU(res, sdRoundBox(q - vec3f(0.0, 0.022, -0.2), vec3f(0.032, 0.028, 0.026), 0.008), 8.0);
  res = opU(res, sdRoundBox(vec3f(abs(q.x) - 0.042, q.y - 0.022, q.z + 0.19), vec3f(0.004, 0.036, 0.032), 0.002), 1.0);
  // the warhead, cable-tied under the frame
  res = opU(res, warheadSdf(q), 10.0);
  let wq = q - vec3f(0.0, WH_Y, 0.0);
  for (var k = 0; k < 2; k++) {
    let zc = select(-0.05, 0.05, k == 1);
    let rr = 0.038 + 0.066 * (zc + 0.44) / 0.52;
    res = opU(res, length(vec2f(length(wq.xy) - rr, wq.z - zc)) - 0.005, 4.0);
    res = opU(res, sdRoundBox(q - vec3f(0.0, -0.028, zc), vec3f(0.012, 0.022, 0.005), 0.002), 4.0);
  }
  // the arming box on the warhead's back, its two leads up to the flight controller (red right, blue left)
  res = opU(res, sdRoundBox(q - vec3f(0.0, -0.068, -0.22), vec3f(0.018, 0.01, 0.026), 0.003), 12.0);
  let m = vec3f(abs(q.x), q.y, q.z);
  let lead = wire(m, vec3f(0.01, -0.062, -0.2), vec3f(0.05, -0.07, -0.12), vec3f(0.06, -0.02, -0.06), vec3f(0.032, 0.02, -0.02), 0.004);
  res = opU(res, lead, select(14.0, 13.0, q.x > 0.0));
  // the red lead from the fuze plug in the tail, up to the frame
  res = opU(res, wire(q, vec3f(0.003, WH_Y, 0.292), vec3f(0.035, WH_Y + 0.01, 0.335), vec3f(0.065, -0.04, 0.3), vec3f(0.035, -0.008, 0.16), 0.0025), 13.0);
  res = opU(res, contactWires(q), 16.0);
  res = opU(res, sdRoundBox(vec3f(abs(q.x), q.y, q.z) - FUZE_ARM - vec3f(0.0, 0.004, 0.0), vec3f(0.012, 0.007, 0.012), 0.002), 11.0);
  // the grenades it still carries, on side rails
  let ng = i32(view.grenadesLeft);
  for (var k = 0; k < 4; k++) {
    if (k >= ng) { break; }
    let gp = vec3f(select(-0.1, 0.1, k < 2), -0.075, select(0.0, select(-0.045, 0.045, k == 1), k < 2));
    res = opU(res, sdEllipsoid(q - gp, vec3f(0.028, 0.028, 0.038)), 15.0);
  }
  return res;
}

/** Ray-march the drone inside its bounding sphere: (distance along r, material) or (NO_HIT, 0). */
fn hitDrone(ro: vec3f, r: vec3f) -> vec2f {
  let iv = sphereInterval(ro, r, view.drone, 1.2);
  if (iv.y < 0.0 || iv.x >= NO_HIT) { return vec2f(NO_HIT, 0.0); }
  var t = max(iv.x, 0.0);
  for (var i = 0; i < 110; i++) {
    let d = droneSdf(toDrone(ro + r * t - view.drone));
    if (d.x < 0.0012) { return vec2f(t, d.y); }
    t += d.x;
    if (t > iv.y) { break; }
  }
  return vec2f(NO_HIT, 0.0);
}

fn droneNormal(p: vec3f) -> vec3f {
  let e = vec2f(0.0015, -0.0015);
  let q = p - view.drone;
  return normalize(
    e.xyy * droneSdf(toDrone(q + e.xyy)).x + e.yyx * droneSdf(toDrone(q + e.yyx)).x +
    e.yxy * droneSdf(toDrone(q + e.yxy)).x + e.xxx * droneSdf(toDrone(q + e.xxx)).x,
  );
}

// FPV: the camera sees its own warhead's nose and the contact wires at the bottom of the picture. It looks this far
// above the frame (as FPV cameras are tilted up) and sits at the camera pod's front.
const FPV_TILT = 0.17;
const FPV_CAM = vec3f(0.0, 0.022, -0.236);

/** an offset from the FPV camera (world axes) -> the body frame; the camera shake and roll carry the drone along */
fn fpvBody(p: vec3f) -> vec3f {
  let c = vec3f(dot(p, view.camRight), dot(p, view.camUp), -dot(p, view.camFwd));
  let ct = cos(FPV_TILT);
  let st = sin(FPV_TILT);
  return vec3f(c.x, c.y * ct - c.z * st, c.y * st + c.z * ct) + FPV_CAM;
}

fn fpvSdf(q: vec3f) -> vec2f {
  return opU(vec2f(warheadSdf(q), 10.0), contactWires(q), 16.0);
}

/** (distance along r from the FPV camera, material) or (NO_HIT, 0) */
fn hitFpvDrone(r: vec3f) -> vec2f {
  // everything it can see of itself is below the camera
  if ((fpvBody(r) - FPV_CAM).y > 0.05) { return vec2f(NO_HIT, 0.0); }
  var t = 0.04;
  for (var i = 0; i < 64; i++) {
    let d = fpvSdf(fpvBody(r * t));
    if (d.x < 0.0008) { return vec2f(t, d.y); }
    t += d.x;
    if (t > 1.3) { break; }
  }
  return vec2f(NO_HIT, 0.0);
}

fn fpvNormal(p: vec3f) -> vec3f {
  let e = vec2f(0.001, -0.001);
  return normalize(
    e.xyy * fpvSdf(fpvBody(p + e.xyy)).x + e.yyx * fpvSdf(fpvBody(p + e.yyx)).x +
    e.yxy * fpvSdf(fpvBody(p + e.yxy)).x + e.xxx * fpvSdf(fpvBody(p + e.xxx)).x,
  );
}

fn droneColor(p: vec3f, r: vec3f, mat: f32) -> vec3f {
  return droneShade(toDrone(p - view.drone), droneNormal(p), p, r, mat);
}

/** q: the body-frame point, n: the world normal, p: the world point */
fn droneShade(q: vec3f, n: vec3f, p: vec3f, r: vec3f, mat: f32) -> vec3f {
  let fres = pow(1.0 - clamp(dot(n, -r), 0.0, 1.0), 5.0);
  var alb = vec3f(0.02);
  var spec = 0.05;
  var gloss = 16.0;
  var emit = vec3f(0.0);
  if (mat == 1.0) {
    // carbon weave
    let wv = step(0.5, fract((q.x + q.z) * 60.0)) * 0.5 + step(0.5, fract((q.x - q.z) * 60.0)) * 0.5;
    alb = vec3f(0.018 + 0.012 * wv);
    spec = 0.12;
    gloss = 30.0;
  } else if (mat == 2.0) {
    alb = vec3f(0.02, 0.07, 0.03); // the flight controller's board, and its status LED
    if (length(q - vec3f(0.02, 0.033, 0.02)) < 0.005 && fract(view.time * 0.7) < 0.5) { emit = vec3f(0.2, 0.6, 3.0); }
  } else if (mat == 3.0) {
    // black wrap over the cells, creased
    alb = vec3f(0.022) * (0.8 + 0.4 * noise3(q * 60.0));
    spec = 0.18;
    gloss = 24.0;
  } else if (mat == 4.0) {
    alb = vec3f(0.012);
  } else if (mat == 5.0) {
    alb = select(vec3f(0.02), vec3f(0.05, 0.12, 0.3), abs(q.y - 0.031) < 0.003);
    spec = 0.35;
    emit = vec3f(1.0, 0.5, 0.2) * view.escape;
  } else if (mat == 6.0) {
    alb = select(vec3f(0.3), vec3f(0.4, 0.03, 0.02), q.z < 0.0);
    spec = 0.4;
  } else if (mat == 7.0) {
    alb = vec3f(0.7);
    spec = 0.2;
  } else if (mat == 8.0) {
    let lp = q - vec3f(0.0, 0.022, -0.2);
    alb = vec3f(0.03);
    spec = 0.15;
    if (lp.z < -0.026) {
      // the pod's window, and the fly behind it: two big red compound eyes
      alb = vec3f(0.01);
      spec = 0.1 + 0.8 * fres;
      gloss = 80.0;
      let eyeL = length(lp.xy - vec2f(-0.014, 0.004));
      let eyeR = length(lp.xy - vec2f(0.014, 0.004));
      if (min(eyeL, eyeR) < 0.011) { alb = vec3f(0.35, 0.02, 0.015); }
    }
  } else if (mat == 9.0) {
    alb = vec3f(0.01);
  } else if (mat == 10.0) {
    // olive drab paint, scuffed, with a steel band round the widest part; bare dark metal in the nose socket;
    // the tail boss is threaded steel, its end plate turned steel with a groove, the fuze plug dark
    let w = q - vec3f(0.0, WH_Y, 0.0);
    let rw = length(w.xy);
    alb = vec3f(0.14, 0.13, 0.05) * (0.75 + 0.35 * noise3(q * 45.0));
    spec = 0.22;
    gloss = 25.0;
    if (abs(w.z - 0.02) < 0.016) { alb = vec3f(0.3, 0.3, 0.29) * (0.8 + 0.2 * noise3(q * 90.0)); spec = 0.5; gloss = 40.0; }
    if (w.z < -0.38 && rw < 0.026) { alb = vec3f(0.04); spec = 0.4; }
    if (w.z > 0.2 && rw < WH_TAIL_R + 0.002) {
      spec = 0.6;
      gloss = 50.0;
      if (w.z < 0.2765) {
        alb = vec3f(0.3, 0.29, 0.27) * (0.45 + 0.55 * smoothstep(0.2, 0.5, abs(fract(w.z * 220.0) - 0.5)));
      } else if (rw < 0.0125) {
        alb = vec3f(0.05);
      } else {
        alb = vec3f(0.42, 0.41, 0.4) * (1.0 - 0.6 * (1.0 - smoothstep(0.0, 0.002, abs(rw - 0.045))));
      }
    }
  } else if (mat == 11.0) {
    alb = vec3f(0.2);
    spec = 0.3;
  } else if (mat == 12.0) {
    alb = vec3f(0.015);
    let led = length(q - vec3f(0.008, -0.056, -0.238));
    if (led < 0.006 && fract(view.time * 1.5) < 0.25) { emit = vec3f(4.0, 0.2, 0.1); }
  } else if (mat == 13.0) {
    alb = vec3f(0.4, 0.02, 0.015);
    spec = 0.2;
  } else if (mat == 14.0) {
    alb = vec3f(0.02, 0.05, 0.35);
    spec = 0.2;
  } else if (mat == 15.0) {
    alb = vec3f(0.045, 0.055, 0.03) * (0.7 + 0.3 * step(0.3, fract(q.z * 90.0)));
  } else if (mat == 16.0) {
    alb = vec3f(0.12); // weathered steel
    spec = 0.5;
    gloss = 60.0;
  } else if (mat == 17.0) {
    alb = vec3f(0.55, 0.38, 0.02);
    spec = 0.2;
  }
  var col = lit(alb, n, 1.0, 1.0) + fireLight(p, n) * alb;
  col += SUN_COL * spec * pow(max(dot(reflect(r, n), SUN), 0.0), gloss) + skyBase(reflect(r, n)) * spec * 0.4;
  return col + emit;
}

/** the four props: motion-blurred discs over whatever is behind them */
fn props(ro: vec3f, r: vec3f, depth: f32, cIn: vec3f) -> vec3f {
  var col = cIn;
  let o = toDrone(ro - view.drone);
  let d = toDrone(r);
  if (abs(d.y) < 1e-5) { return col; }
  let t = (0.118 - o.y) / d.y;
  if (t <= 0.0 || t >= depth) { return col; }
  let h = o + d * t;
  for (var i = 0u; i < 4u; i++) {
    let sx = select(-1.0, 1.0, (i & 1u) == 1u);
    let sz = select(-1.0, 1.0, (i & 2u) == 2u);
    let q = h.xz - vec2f(0.5 * sx, 0.46 * sz);
    let rr = length(q);
    if (rr < 0.24 && rr > 0.017) {
      let ang = atan2(q.y, q.x) + view.rotor * sx * sz;
      let ghost = pow(0.5 + 0.5 * cos(2.0 * ang), 6.0);
      let tip = smoothstep(0.19, 0.23, rr);
      let a = (0.12 + 0.12 * ghost + 0.18 * tip) * (1.0 - smoothstep(0.228, 0.24, rr));
      col = mix(col, SKY_AMB * 0.035, a);
    }
  }
  return col;
}

// ---- view-only things in the air: grenades, blast debris, balloon strings ----------------------------------

fn airborne(ro: vec3f, r: vec3f, depth: ptr<function, f32>, col: ptr<function, vec3f>) {
  // grenades, tumbling
  let ng = min(u32(view.fx.x), MAX_GRENADES);
  for (var i = 0u; i < ng; i++) {
    let g = view.grenades[2u * i];
    let c2 = nearestImage(g.xz, ro.xz);
    let c = vec3f(c2.x, g.y, c2.y);
    let ta = view.time * 9.0 + f32(i) * 1.7;
    let ax = normalize(vec3f(sin(ta), cos(ta), 0.35 * sin(ta * 0.7)));
    let k = 0.075 / 0.105; // across / along
    let oc = ro - c;
    let o2 = oc + ax * dot(oc, ax) * (k - 1.0);
    let d2 = r + ax * dot(r, ax) * (k - 1.0);
    let aa = dot(d2, d2);
    let bb = dot(o2, d2);
    let cc = dot(o2, o2) - 0.075 * 0.075;
    let disc = bb * bb - aa * cc;
    if (disc > 0.0) {
      let t = (-bb - sqrt(disc)) / aa;
      if (t > 0.0 && t < *depth) {
        let x = oc + r * t;
        let m = x + ax * dot(x, ax) * (k - 1.0);
        let n = normalize(m + ax * dot(m, ax) * (k - 1.0));
        let s = dot(x, ax);
        var alb = vec3f(0.045, 0.055, 0.03) * (0.6 + 0.4 * step(0.25, fract(s * 45.0)));
        if (s > 0.08) { alb = vec3f(0.25); } // the fuse and lever
        *col = lit(alb, n, 1.0, 1.0) + vec3f(0.08) * pow(max(dot(reflect(r, n), SUN), 0.0), 20.0);
        *depth = t;
      }
    }
  }
  // debris thrown by the blasts
  let ne = min(u32(view.fx.y), MAX_EXPLOSIONS);
  for (var i = 0u; i < ne; i++) {
    let e = view.explosions[2u * i];
    let e2 = view.explosions[2u * i + 1u];
    let age = e2.x;
    let c2 = nearestImage(e.xz, ro.xz);
    let base = vec3f(c2.x, e.y, c2.y);
    let reach = e.w * 5.5 * min(age, 1.6) + 1.0;
    let bi = sphereInterval(ro, r, base, reach);
    if (bi.y < 0.0 || bi.x >= *depth) { continue; }
    for (var j = 0; j < 14; j++) {
      let h = hash41(e2.y * 13.1 + f32(j) * 7.77);
      let dir = normalize(vec3f(h.x - 0.5, 0.45 + h.y, h.z - 0.5));
      let v = dir * e.w * (1.5 + 2.5 * h.w);
      let tl = (v.y + sqrt(v.y * v.y + 19.6 * max(e.y - 0.04, 0.0))) / 9.8;
      let tt = min(age, tl);
      let pos = base + vec3f(v.x * tt, v.y * tt - 4.9 * tt * tt, v.z * tt);
      let sz = (0.03 + 0.07 * h.y) * e.w * 0.25;
      let si = sphereInterval(ro, r, pos, sz);
      if (si.x > 0.0 && si.x < *depth) {
        let n = normalize(ro + r * si.x - pos);
        let hot = exp(-age * 3.0);
        *col = lit(vec3f(0.02), n, 1.0, 1.0) + vec3f(2.5, 0.8, 0.2) * hot * (0.6 + 0.4 * n.y);
        *depth = si.x;
      }
    }
  }
  // strings under the balloons
  let nb = min(u32(scene.balloonCount), MAX_BALLOONS);
  for (var i = 0u; i < nb; i++) {
    let b = scene.balloons[i];
    if (b.w <= 0.0) { continue; }
    let c = nearestImage(b.xz, ro.xz);
    let top = vec3f(c.x, b.y - b.w, c.y);
    let sway = 0.25 * sin(view.time * 1.3 + f32(i));
    let mid = top + vec3f(sway, -1.3, 0.1);
    let bot = mid + vec3f(-sway * 0.5, -1.3, 0.0);
    let s1 = raySeg(ro, r, top, mid);
    let s2 = raySeg(ro, r, mid, bot);
    let s = select(s2, s1, s1.x < s2.x);
    if (s.y > 0.0 && s.y < *depth) {
      let px = view.fx.z * s.y;
      let a = (1.0 - smoothstep(0.0, px + 0.004, s.x - 0.004)) * min(1.0, 0.008 / max(px, 1e-4));
      *col = mix(*col, SKY_AMB * 0.4, clamp(a, 0.0, 1.0));
    }
  }
}

// ---- translucent: smoke plumes and explosions --------------------------------------------------------------

/** a smoke column rising from `base`, leaning with the wind; analytic, no marching */
fn plume(ro: vec3f, r: vec3f, depth: f32, base: vec2f, height: f32, strength: f32, dark: f32, fire: f32, seed: f32, cIn: vec3f) -> vec3f {
  let lean = vec2f(0.28, 0.1);
  let o = ro.xz - lean * ro.y;
  let d = r.xz - lean * r.y;
  let dd = dot(d, d);
  if (dd < 1e-6) { return cIn; }
  let c = nearestImage(base, o);
  let t = dot(c - o, d) / dd;
  if (t <= 0.0 || t >= depth) { return cIn; }
  let y = ro.y + r.y * t;
  if (y < 0.0 || y > height) { return cIn; }
  let q = o + d * t - c;
  let w = 0.3 + 0.16 * y;
  let g = exp(-dot(q, q) / (w * w));
  if (g < 0.01) { return cIn; }
  let side = (q.x * d.y - q.y * d.x) / (sqrt(dd) * w);
  let n = pnoise(vec2f(side * 1.4 + seed, y * 0.5 - view.time * 0.6), 1e4) * 0.65
    + pnoise(vec2f(side * 3.1 + seed * 2.0, y * 1.2 - view.time * 1.1), 1e4) * 0.35;
  let fade = smoothstep(0.0, 0.6, y) * (1.0 - smoothstep(height * 0.45, height, y));
  let a = clamp(strength * g * (0.2 + n) * fade / (1.0 + 0.1 * y) * min(2.0, 1.0 / sqrt(dd)), 0.0, 0.92);
  let sc = mix(vec3f(0.2, 0.19, 0.18), vec3f(0.035, 0.032, 0.03), dark) * (0.75 + 0.35 * smoothstep(0.0, height, y));
  let f = fogAmount(ro, r, t);
  var col = mix(cIn, mix(sc, fogColor(r), f), a);
  // flames licking up out of a burning wreck
  if (fire > 0.0) {
    let flame = fire * exp(-dot(q, q) / (0.36 * w * w)) * (1.0 - smoothstep(0.9, 2.2 + 0.8 * n, y)) * smoothstep(0.5, 0.9, y);
    col += fireColor(clamp(flame * (0.3 + n), 0.0, 1.0)) * (1.0 - f) * min(2.0, 1.0 / sqrt(dd));
  }
  return col;
}

fn fireColor(h: f32) -> vec3f {
  return vec3f(1.0, 0.32, 0.05) * h * 5.0 + vec3f(1.0, 0.8, 0.5) * h * h * h * 14.0;
}

/** fireball, then soot: a short volume march per explosion (they last GAME.explosionMs, 1.5 s) */
fn explosions(ro: vec3f, r: vec3f, depth: f32, jitter: f32, cIn: vec3f) -> vec3f {
  var col = cIn;
  let ne = min(u32(view.fx.y), MAX_EXPLOSIONS);
  for (var i = 0u; i < ne; i++) {
    let e = view.explosions[2u * i];
    let e2 = view.explosions[2u * i + 1u];
    let age = e2.x;
    let R = e.w;
    let c2 = nearestImage(e.xz, ro.xz);
    let base = vec3f(c2.x, e.y, c2.y);
    let grow = 1.0 - exp(-age * 8.0);
    let rad = R * (0.2 + 0.55 * grow + 0.2 * age);
    let cen = base + vec3f(0.0, R * (0.1 + 0.45 * age) + rad * 0.25, 0.0);
    let iv = sphereInterval(ro, r, cen, rad);
    // the first instant: a flash that lights the whole frame
    let flash = exp(-age * 25.0);
    let sb = raySeg(ro, r, base, base + vec3f(0.0, 0.01, 0.0));
    if (sb.y < depth) {
      col += vec3f(3.0, 2.2, 1.4) * flash * exp(-sb.x * sb.x / (R * R * 0.6));
    }
    if (iv.x >= NO_HIT || iv.y < 0.0) { continue; }
    let t0 = max(iv.x, 0.0);
    let t1 = min(iv.y, depth);
    if (t1 <= t0) { continue; }
    let steps = 12;
    let dt = (t1 - t0) / f32(steps);
    var t = t0 + dt * jitter;
    let fire = exp(-age * 3.2);
    let fade = 1.0 - smoothstep(0.85, 1.45, age);
    var T = 1.0;
    var acc = vec3f(0.0);
    for (var k = 0; k < steps; k++) {
      let x = (ro + r * t - cen) / rad;
      let d = length(x);
      if (d < 1.0) {
        let n = fbm2o(x * 2.4 + vec3f(e2.y, e2.y * 0.7 - age * 1.6, 0.0));
        let den = clamp((1.0 - d) * 2.4 + (n - 0.55) * 2.2, 0.0, 1.0) * fade;
        if (den > 0.002) {
          let a = 1.0 - exp(-den * 6.0 / rad * dt);
          let heat = clamp((0.8 - d) * 1.8 + (n - 0.5) * 1.6, 0.0, 1.0) * fire;
          let soot = mix(vec3f(0.025, 0.023, 0.021), vec3f(0.13, 0.125, 0.12), clamp(age * 0.7, 0.0, 1.0));
          let sc = soot * SKY_AMB * (0.7 + 0.6 * clamp(x.y * 0.5 + 0.5, 0.0, 1.0)) + vec3f(0.5, 0.2, 0.05) * fire * 0.3;
          acc += T * a * (sc + fireColor(heat));
          T *= 1.0 - a;
        }
      }
      t += dt;
    }
    let f = fogAmount(ro, r, t0) * 0.7;
    col = col * T + acc * (1.0 - f) + (1.0 - T) * f * fogColor(r);
  }
  return col;
}

// ---- the world as the player sees it -----------------------------------------------------------------------

struct Surf {
  t: f32,
  kind: f32,
  index: f32,
  mat: f32,
  n: vec3f,
};

/**
 * sceneHit(), refined for the eye: each truck is marched inside the fly's box, and a ray that meets a trunk above
 * its broken top goes on down inside it to the splintered top. Where the refined shape misses, the ray carries on
 * behind it.
 */
fn sceneTrace(ro: vec3f, r: vec3f) -> Surf {
  var off = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = ro + r * off;
    let h = sceneHit(o, r);
    if (h.y >= NO_HIT) { break; }
    let p = o + r * h.y;
    if (h.z == KIND_TREE) {
      let tr = scene.trees[u32(h.w)];
      let d = p.xz - nearestImage(tr.xy, p.xz);
      if (p.y > trunkCut(d, tr)) {
        // above the broken top: down inside the trunk to the splintered top, or out the other side (the lowest the
        // top gets is about 1.1 below tr.w)
        var tExit = min(cylExit(d, r.xz, tr.z), 50.0);
        if (r.y < 0.0) { tExit = min(tExit, (p.y - (tr.w - 1.2)) / -r.y); }
        let top = trunkTopHit(p, r, d, tr, tExit);
        if (top.x >= 0.0) { return Surf(off + h.y + top.x, h.z, h.w, 0.0, top.yzw); }
        off += h.y + tExit + 0.02;
        continue;
      }
      return Surf(off + h.y, h.z, h.w, 0.0, normalize(vec3f(d.x, 0.0, d.y)));
    }
    if (h.z == KIND_CAR) {
      let car = scene.cars[u32(h.w)];
      let pos = nearestImage(car.xy, o.xz);
      let m = traceCar(o, r, pos, car.z, false, NO_HIT);
      if (m.t < NO_HIT) {
        return Surf(off + m.t, KIND_CAR, h.w, m.mat, m.n);
      }
      // missed the truck: through its box to what is behind. A ray leaving by the box's floor meets the ground there.
      let tg = select(NO_HIT, -o.y / r.y, r.y < -1e-5);
      if (tg <= m.exit + 0.01) {
        let g = o + r * tg;
        let kind = select(KIND_GROUND, KIND_ROAD, roadDistance(g.xz) < scene.roadInfo.y * 0.5);
        return Surf(off + tg, kind, 0.0, 0.0, vec3f(0.0, 1.0, 0.0));
      }
      off += max(m.exit, h.y) + 0.01;
      continue;
    }
    return Surf(off + h.y, h.z, h.w, 0.0, vec3f(0.0, 1.0, 0.0));
  }
  return Surf(NO_HIT, KIND_SKY, 0.0, 0.0, vec3f(0.0, 1.0, 0.0));
}

/** everything solid in the player's view: the fly's scene, and the fallen logs (view only) */
fn viewTrace(ro: vec3f, r: vec3f) -> Surf {
  let s = sceneTrace(ro, r);
  let lg = traceLogs(ro, r, s.t);
  if (lg.x > 0.0) { return Surf(lg.x, KIND_LOG, 0.0, 0.0, lg.yzw); }
  return s;
}

fn surfaceColor(s: Surf, ro: vec3f, r: vec3f) -> vec3f {
  let p = ro + r * s.t;
  var col = vec3f(0.0);
  if (s.kind == KIND_GROUND || s.kind == KIND_ROAD) {
    let fp = s.t * view.fx.z / max(abs(r.y), 0.08);
    var alb = soilAlbedo(p.xz, fp);
    alb = debris(p.xz, fp, alb, s.kind == KIND_ROAD);
    if (s.kind == KIND_ROAD) {
      alb = roadAlbedo(p.xz, roadFrame(p.xz), fp, alb);
    }
    // scorch where the wrecks burnt
    let nw = min(u32(view.fx.w), MAX_WRECKS);
    for (var i = 0u; i < nw; i++) {
      let w = view.wrecks[i];
      let d = distance(p.xz, nearestImage(w.xy, p.xz));
      alb *= 0.3 + 0.7 * smoothstep(1.5, 3.2, d + (pnoise(p.xz * 2.0, 200.0) - 0.5) * 1.2);
    }
    let n = vec3f(0.0, 1.0, 0.0);
    col = lit(alb, n, groundOcclusion(p), 1.0) + fireLight(p, n) * alb;
  } else if (s.kind == KIND_TREE) {
    col = trunkColor(p, s.n, r, u32(s.index)) + fireLight(p, s.n) * 0.02;
  } else if (s.kind == KIND_LOG) {
    col = logColor(p, s.n);
  } else if (s.kind == KIND_CANOPY) {
    // dead crowns, if a world has them: charred and dark
    let tr = scene.trees[u32(s.index)];
    let c = nearestImage(tr.xy, p.xz);
    let n = normalize(p - vec3f(c.x, scene.canopies[u32(s.index)].x, c.y));
    col = lit(vec3f(0.02) * (0.6 + 0.8 * noise3(p * 4.0)), n, 0.8, 1.0);
  } else if (s.kind == KIND_BALLOON) {
    let b = scene.balloons[u32(s.index)];
    let c = nearestImage(b.xz, ro.xz);
    let n = normalize(p - vec3f(c.x, b.y, c.y));
    let fres = pow(1.0 - clamp(dot(n, -r), 0.0, 1.0), 4.0);
    let wrapL = (dot(n, SUN) + 0.5) / 1.5;
    col = vec3f(0.45, 0.02, 0.03) * (SUN_COL * max(wrapL, 0.0) + SKY_AMB * (0.55 + 0.45 * n.y));
    col += vec3f(0.6) * pow(max(dot(reflect(r, n), SUN), 0.0), 60.0) + skyBase(reflect(r, n)) * (0.04 + 0.3 * fres);
  } else if (s.kind == KIND_CAR) {
    let car = scene.cars[u32(s.index)];
    let pos = nearestImage(car.xy, p.xz);
    col = carColor(p, s.n, r, pos, car.z, s.mat, -1.0, (car.w - 1.0) * 6.2831853) + fireLight(p, s.n) * 0.08;
  }
  // whatever the player designated as food glows green, pulsing (picture only)
  let isFood = s.index == view.food.y && ((view.food.x == 1.0 && s.kind == KIND_BALLOON) || (view.food.x == 2.0 && s.kind == KIND_CAR));
  if (isFood) {
    let fres = pow(1.0 - clamp(dot(s.n, -r), 0.0, 1.0), 2.0);
    col = mix(col, vec3f(0.3, 1.2, 0.4), (0.25 + 0.15 * sin(view.time * 6.0)) + 0.4 * fres * f32(s.kind == KIND_CAR));
  }
  return col;
}

/** Everything the player sees along one ray, in linear HDR. */
fn renderWorld(ro: vec3f, r: vec3f, jitter: f32, withDrone: bool) -> vec3f {
  gEmit = vec3f(0.0);
  let s = viewTrace(ro, r);
  var depth = s.t;
  var col: vec3f;
  if (s.t >= NO_HIT) {
    col = skyColor(r);
    depth = 1e5;
  } else {
    col = surfaceColor(s, ro, r) + gEmit;
  }
  // wrecks: burnt hulks the fly can't see
  let nw = min(u32(view.fx.w), MAX_WRECKS);
  for (var i = 0u; i < nw; i++) {
    let w = view.wrecks[i];
    let pos = nearestImage(w.xy, ro.xz);
    let m = traceCar(ro, r, pos, w.z, true, depth);
    if (m.t < depth) {
      let p = ro + r * m.t;
      let n = m.n;
      col = carColor(p, n, r, pos, w.z, m.mat, w.w, 0.0) + fireLight(p, n) * 0.02;
      depth = m.t;
    }
  }
  if (withDrone) {
    let d = hitDrone(ro, r);
    if (d.x < depth) {
      col = droneColor(ro + r * d.x, r, d.y);
      depth = d.x;
    }
  }
  if (view.mode == 3.0 && view.fpv.z > 0.5) {
    let d = hitFpvDrone(r);
    if (d.x < depth) {
      let p = r * d.x;
      col = droneShade(fpvBody(p), fpvNormal(p), ro + p, r, d.y);
      depth = d.x;
    }
  }
  airborne(ro, r, &depth, &col);
  if (s.t < NO_HIT || depth < 1e5) {
    col = mix(col, fogColor(r), fogAmount(ro, r, depth));
  }
  if (withDrone) {
    col = props(ro, r, depth, col);
  }
  // smoke: smouldering stumps, and the burning wrecks
  let nt = min(u32(scene.treeCount), MAX_TREES);
  for (var i = 1u; i < nt; i += 6u) {
    let tr = scene.trees[i];
    col = plume(ro, r, depth, tr.xy + vec2f(0.7, 0.4), 12.0, 0.6, 0.2, 0.0, f32(i), col);
  }
  for (var i = 0u; i < nw; i++) {
    let w = view.wrecks[i];
    // a thick column at first, then a thinner smoulder for as long as the wreck stands
    let life = smoothstep(0.0, 1.0, w.w) * mix(1.0, 0.35, smoothstep(8.0, 12.0, w.w));
    col = plume(ro, r, depth, w.xy, 18.0, 2.2 * life, 0.85, exp(-w.w / 4.0) * 1.2, 7.0 + f32(i), col);
  }
  col = explosions(ro, r, depth, jitter, col);
  return col;
}

/** Axial hex rounding: the centre of the 4.8 deg hexagon (in az/el radians) that contains `p`. */
fn hexCentre(p: vec2f, size: f32) -> vec2f {
  let q = (0.57735027 * p.x - 0.33333333 * p.y) / size;
  let r = (0.66666667 * p.y) / size;
  let s = -q - r;
  var rq = round(q);
  var rr = round(r);
  let rs = round(s);
  let dq = abs(rq - q);
  let dr = abs(rr - r);
  let ds = abs(rs - s);
  if (dq > dr && dq > ds) { rq = -rr - rs; } else if (dr > ds) { rr = -rq - rs; }
  return vec2f(size * (1.7320508 * rq + 0.8660254 * rr), size * 1.5 * rr);
}

/** the FPV lens: an equidistant fisheye, view.fpv.x from the centre to the side edge (render.ts fisheyeDir) */
fn fisheye(ndc: vec2f) -> vec3f {
  let p = vec2f(ndc.x * view.tanX, ndc.y);
  let len = length(p);
  let th = len / view.tanX * view.fpv.x;
  let s = select(0.0, sin(th) / len, len > 1e-6);
  return normalize(view.camFwd * cos(th) + (view.camRight * p.x + view.camUp * p.y) * s);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  var col: vec3f;
  let ndc = in.ndc;
  let jitter = hash12(in.pos.xy + vec2f(fract(view.time * 7.0) * 97.0, 0.0));

  if (view.mode == 3.0) {
    // FPV: linear HDR out; `post` grades it
    let r = fisheye(ndc);
    return vec4f(renderWorld(view.camPos, r, jitter, false), 1.0);
  }
  if (view.mode == 1.0) {
    // CHASE
    let r = normalize(view.camFwd + view.camRight * (ndc.x * view.tanX) + view.camUp * (ndc.y * view.tanY));
    col = toDisplay(renderWorld(view.camPos, r, jitter, view.fpv.z > 0.5));
    col *= 1.0 - 0.15 * dot(ndc, ndc);
  } else {
    // EYE and MOSAIC: screen = azimuth x elevation
    var ae = vec2f(ndc.x * view.fovAz, ndc.y * view.fovEl);
    let raw = ae;
    let mosaic = view.mode == 2.0;
    if (mosaic) { ae = hexCentre(ae, 0.0838 / 1.7320508); } // 4.8 deg between neighbouring centres
    let r = toWorld(dirAzEl(ae.x, ae.y), scene.heading, scene.pitch);
    if (mosaic) {
      // exactly what the fly gets: sceneHit's luminance, nothing view-only
      col = vec3f(sceneHit(scene.eye, r).x);
      let e = length(raw - ae) / (0.0838 / 1.7320508);
      col *= 1.0 - 0.35 * smoothstep(0.75, 1.0, e);
    } else {
      col = toDisplay(renderWorld(scene.eye, r, jitter, false));
    }
    // beyond the eyes' reach: no photoreceptor looks there
    let outside = max(abs(raw.x) - EYE_AZ, 0.0) + max(abs(raw.y) - EYE_EL, 0.0);
    col = mix(col, vec3f(0.04, 0.05, 0.07), smoothstep(0.0, 0.05, outside) * 0.85);
    // faint azimuth ticks along the horizon every 30 deg, and a frontal cross
    let deg = raw * 57.29578;
    let tick = 1.0 - smoothstep(0.0, 0.35, abs(fract(deg.x / 30.0 + 0.5) - 0.5) * 30.0);
    col = mix(col, vec3f(1.0), tick * (1.0 - smoothstep(0.0, 1.2, abs(deg.y))) * 0.5);
    let cross = (1.0 - smoothstep(0.0, 0.25, abs(deg.x))) * (1.0 - smoothstep(2.0, 2.5, abs(deg.y)))
      + (1.0 - smoothstep(0.0, 0.25, abs(deg.y))) * (1.0 - smoothstep(2.0, 2.5, abs(deg.x)));
    col = mix(col, vec3f(1.0), clamp(cross, 0.0, 1.0) * 0.6);
    // the phantom the player is painting onto LC10a
    if (view.paint.w > 0.0) {
      let d = length(raw - view.paint.xy);
      let ring = 1.0 - smoothstep(0.0, 0.008, abs(d - view.paint.z));
      let pulse = 0.6 + 0.4 * sin(view.time * 9.0);
      col = mix(col, vec3f(0.3, 1.0, 0.9), ring * pulse);
      col = mix(col, vec3f(0.3, 1.0, 0.9), (1.0 - smoothstep(0.0, view.paint.z, d)) * 0.12);
    }
    col *= 1.0 - 0.08 * dot(ndc, ndc);
  }
  col = mix(col, vec3f(0.6, 0.1, 0.12), view.seizure * 0.4);
  return vec4f(col, 1.0);
}

// ---- post: copy, or grade into FPV footage -----------------------------------------------------------------

fn tapHdr(uv: vec2f) -> vec3f {
  return textureSampleLevel(frameTex, frameSamp, uv, 0.0).rgb;
}

/** a cheap sensor and a starved encoder: blown highlights, crushed blacks, little colour */
fn grade(hdr: vec3f) -> vec3f {
  var x = 1.0 - exp(-max(hdr, vec3f(0.0)) * view.post.y);
  x = pow(x, vec3f(1.0 / 2.2));
  let l = dot(x, vec3f(0.2126, 0.7152, 0.0722));
  x = mix(vec3f(l), x, 0.42);
  x = clamp((x - 0.06) / 0.9, vec3f(0.0), vec3f(1.0));
  x = mix(x, x * x * (3.0 - 2.0 * x), 0.55);
  return x * vec3f(0.98, 1.0, 0.95);
}

fn tapG(uv: vec2f) -> vec3f {
  return grade(tapHdr(uv));
}

fn luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.299, 0.587, 0.114));
}

/** a greasy thumbprint on the lens: fixed in screen space */
fn smudgeMask(uv: vec2f) -> f32 {
  let a = pnoise(uv * vec2f(4.0, 6.0) + 3.0, 1e4) * 0.65 + pnoise(uv * vec2f(11.0, 7.0) + 9.0, 1e4) * 0.35;
  let blob1 = 1.0 - smoothstep(0.08, 0.34, length((uv - vec2f(0.8, 0.72)) * vec2f(1.0, 1.4)));
  let blob2 = 1.0 - smoothstep(0.05, 0.22, length((uv - vec2f(0.16, 0.24)) * vec2f(1.3, 1.0)));
  return clamp((blob1 + blob2 * 0.6) * smoothstep(0.35, 0.7, a) * 1.1, 0.0, 1.0);
}

fn fpvFrame(uv0: vec2f, pix: vec2f) -> vec3f {
  let dims = vec2f(textureDimensions(frameTex));
  let t = view.time;
  var uv = uv0;
  // a close blast tears the picture: rows slide sideways for a moment
  let g = view.post.x;
  if (g > 0.02) {
    let row = floor(uv.y * 40.0);
    let fr = floor(t * 24.0);
    if (hash12(vec2f(row, fr)) < g * 0.5) {
      uv.x += (hash12(vec2f(row + 3.0, fr)) - 0.5) * 0.1 * g;
    }
  }
  let c = uv - 0.5;
  let aspect = dims.x / dims.y;
  let r2 = dot(c * vec2f(aspect, 1.0), c * vec2f(aspect, 1.0)) / (0.25 * aspect * aspect + 0.25);
  // chromatic aberration, growing toward the edges
  let ca = 0.012 * r2;
  var col = vec3f(tapG(uv - c * ca).r, tapG(uv).g, tapG(uv + c * ca * 1.4).b);
  // lens smudge: blur and a halo around anything bright behind it
  let sm = smudgeMask(uv0);
  if (sm > 0.02) {
    var blur = vec3f(0.0);
    let rad = 0.014 * sm;
    for (var k = 0; k < 8; k++) {
      let a = f32(k) * 0.785 + 0.3;
      blur += tapG(uv + vec2f(cos(a) / aspect, sin(a)) * rad * (0.6 + 0.4 * f32(k & 1)));
    }
    blur *= 0.125;
    col = mix(col, blur, sm * 0.75) + max(blur - vec3f(0.6), vec3f(0.0)) * sm * 0.8;
  }
  // the encoder: chroma per 8-pixel block, and flat areas quantised toward the block's mean
  let bs = 8.0 / dims;
  let bc = (floor(uv / bs) + 0.5) * bs;
  let blk = (tapG(bc + bs * vec2f(-0.25, -0.25)) + tapG(bc + bs * vec2f(0.25, -0.25)) + tapG(bc + bs * vec2f(-0.25, 0.25)) + tapG(bc + bs * vec2f(0.25, 0.25))) * 0.25;
  let lp = luma(col);
  let lb = luma(blk);
  col = vec3f(lp) + (blk - vec3f(lb)) * 0.8 + (col - vec3f(lp)) * 0.2;
  let flatA = 1.0 - smoothstep(0.012, 0.05, abs(lp - lb));
  col += (lb - lp) * flatA * 0.45;
  // vignette, heavier at the fisheye's rim
  col *= 1.0 - 0.6 * pow(clamp(r2, 0.0, 1.5), 1.6);
  // sensor grain, worse in the dark, 30 frames a second
  let fr = floor(t * 30.0);
  let gr = hash12(pix + vec2f(fr * 17.3, fr * 3.1)) + hash12(pix * 1.7 + vec2f(fr * 5.9, 11.0)) - 1.0;
  col += gr * (0.035 + 0.06 * (1.0 - clamp(lp * 1.5, 0.0, 1.0)));
  // the drone is gone: a white flash, then snow
  let lost = view.fpv.y;
  if (lost >= 0.0) {
    let snow = hash12(floor(pix * 0.5) + vec2f(fr * 13.7, fr * 7.1));
    let roll = 0.75 + 0.25 * sin(uv.y * 9.0 - t * 14.0);
    let s = vec3f(snow * 0.6 * roll + 0.08);
    col = mix(s, vec3f(1.0), 1.0 - smoothstep(0.0, 0.3, lost));
  }
  return clamp(col, vec3f(0.0), vec3f(1.0));
}

@fragment
fn post(in: VsOut) -> @location(0) vec4f {
  if (view.mode != 3.0) {
    let dims = vec2i(textureDimensions(frameTex));
    let px = clamp(vec2i(in.pos.xy), vec2i(0), dims - vec2i(1));
    return vec4f(textureLoad(frameTex, px, 0).rgb, 1.0);
  }
  let uv = vec2f(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
  return vec4f(fpvFrame(uv, in.pos.xy), 1.0);
}
