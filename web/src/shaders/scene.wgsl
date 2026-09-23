// The world, defined analytically once and shared by the fly's eye (compute) and the player's view
// (fragment): what the fly sees is exactly what you see, minus view.wgsl's cosmetics.
//
// Requires a module-scope `scene: Scene` binding in the including shader.
//
// An open field on a torus: a ground plane a shade darker than the sky, a road loop a shade darker still,
// trees (a trunk and a round canopy, low contrast), a few balloons and the military trucks (flat grey boxes).
// Every object is drawn at its nearest periodic image from the eye, so the field has no edge. Anything half a
// world away is well under the eye's 4.8 deg sampling, so the seam never shows up in what the fly sees.
//
// THE MARKER: the target the player designated as food is drawn at markLum (black) instead of its own grey.
// This model's eye has no colour, so marking works in luminance. It changes what the brain sees (see MARK in
// config.ts); markOn = 0 turns it off.

const MAX_TREES = 32u; // must match render.ts
const MAX_BALLOONS = 8u;
const MAX_CARS = 8u;    // trucks; must match render.ts and world.ts
const MAX_ROAD = 32u;  // points; must match ROAD_POINTS in world.ts

struct Scene {
  eye: vec3f,      // fly head (the drone's camera) in world
  heading: f32,    // radians, positive = turned right
  pitch: f32,      // radians, positive = looking up
  time: f32,
  sky: f32,
  ground: f32,
  treeLum: f32,
  balloonLum: f32,
  treeCount: f32,
  balloonCount: f32,
  size: f32,       // torus period
  roadLum: f32,
  foodKind: f32,   // what is designated as food: 0 nothing, 1 a balloon, 2 a truck
  foodIndex: f32,  // which balloon or truck
  trees: array<vec4f, MAX_TREES>,       // x, z, trunk radius, trunk top
  canopies: array<vec4f, MAX_TREES>,    // canopy centre height, canopy radius, 0, 0
  balloons: array<vec4f, MAX_BALLOONS>, // x, y, z, radius (radius 0 = popped)
  cars: array<vec4f, MAX_CARS>,        // per truck: x, z, heading, > 0 = there (0 = destroyed or no truck; the view
                                       // reads 1 + the wheels' turn in revolutions from it)
  carCount: vec4f, // how many truck slots, 0, 0, 0
  carDims: vec4f,  // half length, half width, height, luminance (the same for every truck)
  road: array<vec4f, 16>,               // the loop's points, two per vec4: (x0, z0, x1, z1)
  roadInfo: vec4f, // point count, width, marker luminance, marker on (1) / off (0)
};

const PI = 3.141592653589793;
const NO_HIT = 1e6;

// Surface kinds. Only the player's view reads these, to colour the picture.
const KIND_SKY = 0.0;
const KIND_GROUND = 1.0;
const KIND_TREE = 2.0;     // the trunk; the canopy is KIND_CANOPY
const KIND_ROAD = 5.0;
const KIND_CANOPY = 6.0;
const KIND_BALLOON = 3.0;
const KIND_CAR = 4.0;

// Fly frame (+x right, +y up, -z forward) -> world: pitch about x, then heading about y.
fn toWorld(d: vec3f, heading: f32, pitch: f32) -> vec3f {
  let cp = cos(pitch);
  let sp = sin(pitch);
  let p = vec3f(d.x, d.y * cp - d.z * sp, d.y * sp + d.z * cp);
  let c = cos(heading);
  let s = sin(heading);
  return vec3f(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
}

/** The copy of a point nearest to `ro` on the torus (xz only). */
fn nearestImage(p: vec2f, ro: vec2f) -> vec2f {
  let d = p - ro;
  return ro + d - scene.size * round(d / scene.size);
}

/** Ray against an upright cylinder standing on the ground, up to `top`; distance or NO_HIT. */
fn hitTrunk(ro: vec3f, r: vec3f, c: vec2f, radius: f32, top: f32) -> f32 {
  let o = ro.xz - c;
  let a = dot(r.xz, r.xz);
  if (a < 1e-8) { return NO_HIT; }
  let b = dot(o, r.xz);
  let cc = dot(o, o) - radius * radius;
  let disc = b * b - a * cc;
  if (disc < 0.0) { return NO_HIT; }
  let t = (-b - sqrt(disc)) / a;
  if (t <= 0.0) { return NO_HIT; }
  let y = ro.y + r.y * t;
  if (y < 0.0 || y > top) { return NO_HIT; }
  return t;
}

fn hitSphere(ro: vec3f, r: vec3f, c: vec3f, radius: f32) -> f32 {
  let o = ro - c;
  let b = dot(o, r);
  let cc = dot(o, o) - radius * radius;
  let disc = b * b - cc;
  if (disc < 0.0) { return NO_HIT; }
  let t = -b - sqrt(disc);
  if (t <= 0.0) { return NO_HIT; }
  return t;
}

/** world -> a truck's frame (+x its right, -z its front), about its centre on the ground */
fn toCar(p: vec3f, heading: f32) -> vec3f {
  let c = cos(heading);
  let s = sin(heading);
  return vec3f(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

/** how many truck slots to look at */
fn carCount() -> u32 {
  return min(u32(scene.carCount.x), MAX_CARS);
}

/** Ray against truck `car`'s box (x, z, heading, there; slab test in its own frame); distance or NO_HIT. */
fn hitCar(ro: vec3f, r: vec3f, car: vec4f) -> f32 {
  if (car.w <= 0.0) { return NO_HIT; }
  let c = nearestImage(car.xy, ro.xz);
  // a quick reject: a ray that misses the box's bounding sphere (padded) misses the box, so nothing seen changes
  let rs = length(scene.carDims.xyz) + 0.1;
  let oc = ro - vec3f(c.x, scene.carDims.z * 0.5, c.y);
  let ob = dot(oc, r);
  let occ = dot(oc, oc) - rs * rs;
  if (occ > 0.0 && (ob > 0.0 || ob * ob < occ)) { return NO_HIT; }
  let o = toCar(ro - vec3f(c.x, 0.0, c.y), car.z);
  let d = toCar(r, car.z);
  let lo = vec3f(-scene.carDims.y, 0.0, -scene.carDims.x);
  let hi = vec3f(scene.carDims.y, scene.carDims.z, scene.carDims.x);
  let inv = 1.0 / select(d, vec3f(1e-6), abs(d) < vec3f(1e-6));
  let t0 = (lo - o) * inv;
  let t1 = (hi - o) * inv;
  let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tf < max(tn, 0.0) || tn <= 0.0) { return NO_HIT; }
  return tn;
}

/** Road point i of the loop. */
fn roadPoint(i: u32) -> vec2f {
  let v = scene.road[i / 2u];
  return select(v.xy, v.zw, (i & 1u) == 1u);
}

/** Horizontal distance from a ground point to the road's centre line (nearest image of the point). */
fn roadDistance(p: vec2f) -> f32 {
  let n = min(u32(scene.roadInfo.x), MAX_ROAD);
  var best = NO_HIT;
  for (var i = 0u; i < n; i++) {
    let a = roadPoint(i);
    let b = roadPoint((i + 1u) % n);
    let q = nearestImage(p, a) - a;
    let e = b - a;
    let h = clamp(dot(q, e) / max(dot(e, e), 1e-6), 0.0, 1.0);
    best = min(best, length(q - e * h));
  }
  return best;
}

/** A target's luminance to the fly: its own grey, or the marker's black if it is the designated food. */
fn targetLum(own: f32, kind: f32, index: f32) -> f32 {
  let marked = scene.roadInfo.w > 0.5 && scene.foodKind == kind && scene.foodIndex == index;
  return select(own, scene.roadInfo.z, marked);
}

/**
 * Nearest hit along a ray, as (luminance, distance, kind, object index). `r` must be unit length.
 */
fn sceneHit(ro: vec3f, r: vec3f) -> vec4f {
  var best = vec4f(scene.sky, NO_HIT, KIND_SKY, 0.0);

  if (r.y < -1e-5) {
    let t = -ro.y / r.y;
    if (t > 0.0) {
      best = vec4f(scene.ground, t, KIND_GROUND, 0.0);
      // the road: a flat band a shade darker, with a hard edge (the fly gets no texture, only this)
      if (roadDistance(ro.xz + r.xz * t) < scene.roadInfo.y * 0.5) {
        best = vec4f(scene.roadLum, t, KIND_ROAD, 0.0);
      }
    }
  }

  let nt = min(u32(scene.treeCount), MAX_TREES);
  for (var i = 0u; i < nt; i++) {
    let p = scene.trees[i];
    let c = nearestImage(p.xy, ro.xz);
    let tt = hitTrunk(ro, r, c, p.z, p.w);
    if (tt < best.y) {
      best = vec4f(scene.treeLum, tt, KIND_TREE, f32(i));
    }
    let k = scene.canopies[i];
    let tk = hitSphere(ro, r, vec3f(c.x, k.x, c.y), k.y);
    if (tk < best.y) {
      best = vec4f(scene.treeLum, tk, KIND_CANOPY, f32(i));
    }
  }

  let nb = min(u32(scene.balloonCount), MAX_BALLOONS);
  for (var i = 0u; i < nb; i++) {
    let b = scene.balloons[i];
    if (b.w <= 0.0) { continue; }
    let c = nearestImage(b.xz, ro.xz);
    let t = hitSphere(ro, r, vec3f(c.x, b.y, c.y), b.w);
    if (t < best.y) {
      best = vec4f(targetLum(scene.balloonLum, 1.0, f32(i)), t, KIND_BALLOON, f32(i));
    }
  }

  // the trucks: flat grey boxes, the designated one black
  let ncars = carCount();
  for (var i = 0u; i < ncars; i++) {
    let tc = hitCar(ro, r, scene.cars[i]);
    if (tc < best.y) {
      best = vec4f(targetLum(scene.carDims.w, 2.0, f32(i)), tc, KIND_CAR, f32(i));
    }
  }
  return best;
}

/** What the fly's eye reads. Cosmetic shading belongs to view.wgsl, so it can never change this. */
fn sceneLum(ro: vec3f, r: vec3f) -> f32 {
  return sceneHit(ro, r).x;
}
