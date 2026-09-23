// The world and the drone. Nothing here is biology: the brain only supplies the commands in `Commands`.
//
// The game on top (all mine): CAR.count military trucks drive the road loop (world.cars); the drone carries a bomb,
// so ramming a truck blows up both, and the drone comes back at the start GAME.respawnMs later while the brain keeps
// running; G drops one of GAME.grenades grenades (switched off for now), which falls and bursts on whatever it hits; a
// wrecked truck stays wrecked, blocking the road (trucks stop behind it), until the player respawns the trucks.

import { CAR, CONTROL, DRONE, GAME, WORLD } from "../config";

/** the escape hop: its speeds (world units/s) and how fast they decay */
const ESCAPE = { climb: CONTROL.escapeClimb, back: CONTROL.escapeBack, decayMs: CONTROL.escapeDecayMs };

export interface Commands {
  /** deg/s, + = right */
  yawRate: number;
  /** deg/s of gaze pitch, + = up */
  pitchRate: number;
  /** true on the readback where the giant fiber burst */
  escape: boolean;
}

/** An obstacle: an upright trunk from the ground to `trunkTop`, and a round canopy centred at `canopyY`. */
export interface Tree {
  x: number;
  z: number;
  trunkR: number;
  trunkTop: number;
  canopyY: number;
  canopyR: number;
}

/** A closed loop of road: `points` in order (world x, z), joined last to first. The trucks drive it. */
export interface Road {
  points: [number, number][];
  width: number;
}

/** A grenade in the air (world units, units/s). The scenario fills this; the view draws it; the fly never sees it. */
export interface Grenade {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}

/** An explosion that started at brain time `t0` (ms). View-only: the fly never sees it. */
export interface Explosion {
  x: number;
  y: number;
  z: number;
  t0: number;
  /** world units the blast reaches */
  radius: number;
}

/**
 * What is left of a destroyed truck, from brain time `t0` (ms), until the trucks are respawned (respawnCars; with
 * autoCarRespawn, for GAME.wreckMs). The fly never sees it, but it blocks the road: trucks stop behind it.
 */
export interface Wreck {
  x: number;
  z: number;
  heading: number;
  t0: number;
  /** where along the road it stands (as Car.s) */
  s: number;
}

export interface Balloon {
  x: number;
  y: number;
  z: number;
  r: number;
}

/** A military truck. It drives the road loop in its lane (CAR). */
export interface Car {
  x: number;
  z: number;
  /** radians, + = right, 0 = driving along -z */
  heading: number;
  /** a point on the road a little ahead of it, where it is steering */
  wx: number;
  wz: number;
  /** how far along the road loop it is (world units from road.points[0]), and which way it drives it (+1 / -1) */
  s: number;
  dir: number;
  /** radians the wheels have turned (view only: they stop when the truck does) */
  wheel: number;
}

/** Something the player can designate as food: a truck (its slot in world.cars) or a balloon (its slot). */
export type Target = { kind: "car"; index: number } | { kind: "balloon"; index: number };

/** What happened in the game, for the HUD. `index` is the truck's slot in world.cars. */
export type GameEvent =
  | { kind: "carDestroyed"; index: number; by: "ram" | "grenade" }
  | { kind: "droneLost" }
  | { kind: "respawn" }
  | { kind: "carBack"; index: number }
  | { kind: "grenade"; left: number }
  | { kind: "burst"; hit: boolean };

/** mulberry32: a small seeded RNG, so ?seed=N gives the same field and the same sequence of targets */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** shortest signed difference on the torus */
export function wrapDelta(d: number) {
  return d - WORLD.size * Math.round(d / WORLD.size);
}

const DEG = Math.PI / 180;
/** points on the road loop; must match MAX_ROAD in scene.wgsl */
const ROAD_POINTS = 32;
const angleDiff = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
/** grenades: collision radius, where one leaves the drone (below its centre), and the integration step */
const GRENADE_R = 0.15;
const GRENADE_DROP = 0.35;
const GRENADE_STEP_MS = 10;
/** where the drone starts, and comes back after it is lost (the altitude is DRONE.startAlt) */
const START = { x: 0, z: 0 };

export interface WorldOptions {
  seed: number;
  trees?: number;
  /** trunk radius (default WORLD.trunkRadius, ?trunkr=): what the fly sees of a tree, and what it collides with */
  trunkR?: number;
  balloons?: number;
  /** how many trucks (default CAR.count, at most MAX_CARS) */
  cars?: number;
}

/** trucks at most; must match MAX_CARS in scene.wgsl and render.ts */
export const MAX_CARS = 8;

export class World {
  x = START.x;
  y: number = DRONE.startAlt;
  z = START.z;
  /** radians, + = right */
  heading = 0;
  /** radians, gaze and flight path, + = up */
  pitch = 0;
  /** cosmetic, radians */
  bank = 0;
  /** deg/s actually applied, after rotor inertia */
  yawRate = 0;
  pitchRate = 0;
  /** extra velocity of an escape hop (world units/s): up and backward */
  hopUp = 0;
  hopBack = 0;
  /** the drone's velocity over the last update (world units/s); a dropped grenade starts with it */
  vx = 0;
  vy = 0;
  vz = 0;
  time = 0;
  stunnedUntil = 0;
  /**
   * Held in place: the drone neither moves nor turns, whatever the brain commands (open-loop tests). Time,
   * the trucks and the brain keep running, and the commands are still recorded by whoever reads them.
   */
  frozen = false;
  /**
   * The drone blew up with its bomb. Until `respawnAt` (brain ms) it is gone: it stays where it was, the brain's
   * commands are ignored and nothing collides with it. Then it comes back at the start. The brain never stops.
   */
  droneLost = false;
  respawnAt = 0;
  /**
   * Every contact is fatal: a tree, the ground or a balloon sets off the bomb as a truck does. The stimulation test
   * turns it off (the old bounce and stun), because it measures steering, not survival.
   */
  fatal = true;
  /**
   * Test hooks for the approach test (bench.ts): `steerOff` flies straight at cruise whatever the brain commands
   * (the commands are still recorded), and `ghost` makes nothing collide, so a trial can fly on through its tree.
   */
  steerOff = false;
  ghost = false;
  /** multiplies DRONE.cruise: the pursuit speed-up while the food is locked (main.ts, PURSUIT) */
  speedScale = 1;
  /** grenades each new drone gets (main.ts sets 0 while grenades are switched off, GAME.grenadesEnabled) */
  grenadesPerDrone: number = GAME.grenades;
  /** grenades the current drone still carries */
  grenadesLeft: number = GAME.grenades;
  trees: Tree[] = [];
  road: Road = { points: [], width: WORLD.roadWidth };
  balloons: Balloon[] = [];
  /** the trucks, one slot each; null while a destroyed one waits to come back (carRespawnAt) */
  cars: (Car | null)[] = [];
  grenades: Grenade[] = [];
  explosions: Explosion[] = [];
  wrecks: Wreck[] = [];
  /**
   * Destroyed trucks come back by themselves, GAME.carRespawnMs later, and wrecks clear after GAME.wreckMs (the chase
   * test turns this on, as it was measured that way). Off in the game: they wait for respawnCars().
   */
  autoCarRespawn = false;
  /** per slot: brain time (ms) a destroyed truck comes back (Infinity: when respawnCars() is called) */
  carRespawnAt: number[] = [];
  /** what the player has designated as food, if anything */
  food: Target | null = null;
  stats = {
    popped: 0, rams: 0, treeHits: 0, groundHits: 0, groundMs: 0, escapes: 0, distance: 0, seizures: 0, foodReached: 0,
    carsDestroyed: 0, dronesLost: 0, grenadesDropped: 0, grenadeKills: 0,
    /** drones lost to something other than the food, by cause (crashCar: ramming a truck that isn't the food) */
    crashTree: 0, crashGround: 0, crashBalloon: 0, crashCar: 0,
  };
  onGround = false;
  /** brain time of the last pop or ram */
  lastPop = -Infinity;
  lastRam = -Infinity;
  /** called when the drone reaches a target, before it respawns; `food` says whether it was the designated one */
  onReach: ((t: Target, food: boolean) => void) | null = null;
  /** game events, for the HUD */
  onEvent: ((e: GameEvent) => void) | null = null;
  private rand: () => number;
  private carRand: () => number;
  private bounceX = 0;
  private bounceZ = 0;
  /** cumulative length of the road loop at each point (roadS[n] is the whole loop) */
  private roadS: number[] = [0];

  constructor(opts: WorldOptions) {
    this.rand = rng(opts.seed);
    // its own stream, so the trucks (how many, where) leave the field and the balloons the same
    this.carRand = rng(opts.seed * 7919 + 13);
    const half = WORLD.size / 2;
    // the road: a wobbly closed loop around the middle of the field, its own seeded shape
    const roadRand = rng(opts.seed * 31 + 5);
    const p2 = roadRand() * 2 * Math.PI;
    const p3 = roadRand() * 2 * Math.PI;
    for (let i = 0; i < ROAD_POINTS; i++) {
      const a = (i / ROAD_POINTS) * 2 * Math.PI;
      const r = WORLD.roadRadius * (1 + 0.18 * Math.sin(2 * a + p2) + 0.1 * Math.sin(3 * a + p3));
      this.road.points.push([Math.sin(a) * r, -Math.cos(a) * r]);
    }
    const pts = this.road.points;
    for (let i = 0; i < pts.length; i++) {
      const [ax, az] = pts[i];
      const [bx, bz] = pts[(i + 1) % pts.length];
      this.roadS.push(this.roadS[i] + Math.hypot(bx - ax, bz - az));
    }
    const treeCount = opts.trees ?? WORLD.treeCount;
    // trees anywhere but on the road, on top of the start, or on top of each other
    for (let tries = 0; this.trees.length < treeCount && tries < 5000; tries++) {
      const x = (this.rand() * 2 - 1) * half;
      const z = (this.rand() * 2 - 1) * half;
      if (Math.hypot(x, z) < 8) continue;
      if (this.distanceToRoad(x, z) < this.road.width / 2 + WORLD.canopyRadius + 1) continue;
      if (this.trees.some((t) => Math.hypot(wrapDelta(t.x - x), wrapDelta(t.z - z)) < 6)) continue;
      this.trees.push({ x, z, trunkR: opts.trunkR ?? WORLD.trunkRadius, trunkTop: WORLD.trunkTop, canopyY: WORLD.canopyY, canopyR: WORLD.canopyRadius });
    }
    const balloonCount = opts.balloons ?? WORLD.balloonCount;
    for (let i = 0; i < balloonCount; i++) this.balloons.push(this.spawnBalloon());
    const carCount = Math.max(0, Math.min(MAX_CARS, Math.round(opts.cars ?? CAR.count)));
    for (let i = 0; i < carCount; i++) {
      this.cars.push(null);
      this.carRespawnAt.push(0);
    }
    this.placeCarsAtStart();
  }

  /** Horizontal distance from (x, z) to the road's centre line, on the torus. */
  distanceToRoad(x: number, z: number) {
    const pts = this.road.points;
    let best = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const [ax, az] = pts[i];
      const [bx, bz] = pts[(i + 1) % pts.length];
      // relative to a, nearest image of the point
      const px = wrapDelta(x - ax);
      const pz = wrapDelta(z - az);
      const ex = bx - ax;
      const ez = bz - az;
      const h = Math.max(0, Math.min(1, (px * ex + pz * ez) / (ex * ex + ez * ez)));
      best = Math.min(best, Math.hypot(px - ex * h, pz - ez * h));
    }
    return best;
  }

  // ---- the road -------------------------------------------------------------------------------------

  /** the road loop's length, world units */
  get roadLength() {
    return this.roadS[this.roadS.length - 1];
  }

  /** The point on the road's centre line `s` units along it from points[0] (any s: the loop wraps). */
  roadPoint(s: number): [number, number] {
    const L = this.roadLength;
    const u = ((s % L) + L) % L;
    const pts = this.road.points;
    let i = 0;
    while (i < pts.length - 1 && this.roadS[i + 1] < u) i++;
    const [ax, az] = pts[i];
    const [bx, bz] = pts[(i + 1) % pts.length];
    const h = (u - this.roadS[i]) / Math.max(1e-6, this.roadS[i + 1] - this.roadS[i]);
    return [ax + (bx - ax) * h, az + (bz - az) * h];
  }

  /** Where a car `s` along the road driving `dir` sits: in its lane, facing along the (smoothed) road. */
  private carPose(s: number, dir: number) {
    const [bx, bz] = this.roadPoint(s - dir * 2);
    const [fx, fz] = this.roadPoint(s + dir * 2);
    const heading = Math.atan2(fx - bx, -(fz - bz));
    const [cx, cz] = this.roadPoint(s);
    // right of the direction of travel: (cos h, sin h)
    return { x: cx + Math.cos(heading) * CAR.lane, z: cz + Math.sin(heading) * CAR.lane, heading };
  }

  private makeCar(s: number, dir: number): Car {
    const car: Car = { x: 0, z: 0, heading: 0, wx: 0, wz: 0, s, dir, wheel: 0 };
    this.poseCar(car);
    return car;
  }

  private poseCar(car: Car) {
    const L = this.roadLength;
    car.s = ((car.s % L) + L) % L;
    const p = this.carPose(car.s, car.dir);
    car.x = p.x;
    car.z = p.z;
    car.heading = p.heading;
    [car.wx, car.wz] = this.roadPoint(car.s + car.dir * 4);
  }

  /**
   * How far truck `i` may still drive before it has to stop: CAR.stopGap short of a wreck ahead in either lane (a
   * burning truck blocks the road), or of the next truck ahead in its own lane (so a column queues behind it).
   */
  private roomAhead(i: number): number {
    const car = this.cars[i]!;
    const L = this.roadLength;
    const ahead = (s: number) => ((((s - car.s) * car.dir) % L) + L) % L;
    let room = Infinity;
    for (const w of this.wrecks) room = Math.min(room, ahead(w.s) - CAR.stopGap);
    this.cars.forEach((c, j) => {
      if (c && j !== i && c.dir === car.dir) room = Math.min(room, ahead(c.s) - CAR.stopGap);
    });
    return room;
  }

  private driveCar(i: number, ms: number) {
    const car = this.cars[i]!;
    // slows over the last CAR.brakeDist (to a crawl, CAR.crawl of its speed, not an endless creep) and stops exactly
    // CAR.stopGap short of what is ahead
    const room = this.roomAhead(i);
    const v = room > 0 ? CAR.speed * Math.max(CAR.crawl, Math.min(1, room / CAR.brakeDist)) : 0;
    const step = Math.min(v * (ms / 1000), Math.max(0, room));
    car.s += car.dir * step;
    car.wheel += step / CAR.tyreRadius;
    this.poseCar(car);
  }

  /** Where truck `i` will be `secs` from now, if nothing stops it (for aiming). */
  carPositionIn(i: number, secs: number): [number, number] | null {
    const car = this.cars[i];
    if (!car) return null;
    const p = this.carPose(car.s + car.dir * CAR.speed * secs, car.dir);
    return [p.x, p.z];
  }

  /** bearing (radians, relative to the drone's heading) and horizontal distance of a point from the drone */
  private relative(x: number, z: number) {
    const dx = wrapDelta(x - this.x);
    const dz = wrapDelta(z - this.z);
    return { az: angleDiff(Math.atan2(dx, -dz) - this.heading), dist: Math.hypot(dx, dz) };
  }

  /** is (x, z) at least CAR.spacing from every truck but slot `except`? */
  private clearOfCars(x: number, z: number, except: number) {
    return this.cars.every((c, j) => j === except || !c || Math.hypot(wrapDelta(c.x - x), wrapDelta(c.z - z)) >= CAR.spacing);
  }

  /** road spots `step` apart where truck `i` could go, clear of the others (unless not `spaced`), seen from the drone */
  private roadSpots(i: number, step: number, spaced = true) {
    const spots: { s: number; x: number; z: number; az: number; dist: number }[] = [];
    for (let s = 0; s < this.roadLength; s += step) {
      const p = this.carPose(s, 1);
      if (!spaced || this.clearOfCars(p.x, p.z, i)) spots.push({ s, x: p.x, z: p.z, ...this.relative(p.x, p.z) });
    }
    return spots;
  }

  /**
   * The start: CAR.startInView trucks on the road ahead of the start pose (within CAR.startAzDeg, CAR.startMin-startMax
   * away, no trunk in the way), so the FPV view opens on them; the rest anywhere on the road. Never crowded
   * (CAR.spacing). Called while the drone sits at the start, heading 0.
   */
  private placeCarsAtStart() {
    for (let i = 0; i < this.cars.length; i++) {
      const spots = this.roadSpots(i, 1);
      const ahead = spots.filter(
        (p) => Math.abs(p.az) <= CAR.startAzDeg * DEG && p.dist >= CAR.startMin && p.dist <= CAR.startMax && !this.occluded(p.x, CAR.height / 2, p.z),
      );
      const from = i < CAR.startInView && ahead.length ? ahead : spots;
      const s = from.length ? from[Math.floor(this.carRand() * from.length)].s : this.carRand() * this.roadLength;
      this.cars[i] = this.makeCar(s, this.carRand() < 0.5 ? 1 : -1);
    }
  }

  /**
   * The chase test's target: the slot it keeps ahead (placeCarAhead), or null. Other trucks come back hidden as ever.
   * "Ahead" is on the road, in front of the drone (within 60 deg, 12-20 away), within the eyes' reach. The road doesn't
   * always pass there, so failing that 75 deg and 10-30, and failing that the road point nearest to that window.
   */
  carAhead: number | null = null;
  /** how placeCarAhead did: in the 60 deg / 12-20 window, in the wider one, or the nearest road point to it */
  aheadPlacements = { inWindow: 0, wider: 0, nearest: 0 };

  /** Put truck `i` on the road in front of the drone now (the chase test starts, or its target was lost). */
  placeCarAhead(i: number) {
    if (i < 0 || i >= this.cars.length) return;
    const windows = [
      { az: 60, min: 12, max: 20 },
      { az: 75, min: 10, max: 30 },
    ];
    // clear of the other trucks if the road allows it
    let spots = this.roadSpots(i, 1);
    if (!spots.length) spots = this.roadSpots(i, 1, false);
    let pick: number | null = null;
    for (const [k, w] of windows.entries()) {
      const ok = spots.filter((p) => Math.abs(p.az) <= w.az * DEG && p.dist >= w.min && p.dist <= w.max);
      if (ok.length) {
        pick = ok[Math.floor(this.carRand() * ok.length)].s;
        this.aheadPlacements[k === 0 ? "inWindow" : "wider"]++;
        break;
      }
    }
    if (pick === null) {
      this.aheadPlacements.nearest++;
      const w = windows[0];
      const cost = (p: { az: number; dist: number }) =>
        Math.max(0, Math.abs(p.az) / DEG - w.az) / w.az + Math.max(0, w.min - p.dist, p.dist - w.max) / 10;
      pick = spots.reduce((a, b) => (cost(b) < cost(a) ? b : a)).s;
    }
    this.cars[i] = this.makeCar(pick, this.carRand() < 0.5 ? 1 : -1);
  }

  /** Put truck `i` back on the road where the fly can't see it (CAR.hide*): behind it, far off, or behind a tree. */
  private placeCarHidden(i: number) {
    const hidden: number[] = [];
    let best = { s: 0, az: -1 };
    for (const p of this.roadSpots(i, 2)) {
      if (Math.abs(p.az) > best.az) best = { s: p.s, az: Math.abs(p.az) };
      if (p.dist < CAR.hideMin) continue;
      const behind = Math.abs(p.az) > CAR.hideAzDeg * DEG;
      if (behind || p.dist > CAR.hideDist || this.occluded(p.x, CAR.height / 2, p.z)) hidden.push(p.s);
    }
    const s = hidden.length ? hidden[Math.floor(this.carRand() * hidden.length)] : best.s;
    this.cars[i] = this.makeCar(s, this.carRand() < 0.5 ? 1 : -1);
  }

  // ---- balloons -------------------------------------------------------------------------------------

  private clearOfTrees(x: number, z: number, margin: number) {
    return this.trees.every((t) => Math.hypot(wrapDelta(t.x - x), wrapDelta(t.z - z)) > margin + t.canopyR);
  }

  /** a balloon at a random bearing and distance from the drone, clear of trees */
  spawnBalloon(): Balloon {
    for (let tries = 0; ; tries++) {
      const bearing = this.rand() * 2 * Math.PI;
      const dist = WORLD.spawnMin + this.rand() * (WORLD.spawnMax - WORLD.spawnMin);
      const b = {
        x: this.x + Math.sin(bearing) * dist,
        z: this.z - Math.cos(bearing) * dist,
        y: WORLD.spawnAltMin + this.rand() * (WORLD.spawnAltMax - WORLD.spawnAltMin),
        r: WORLD.balloonRadius,
      };
      if (this.clearOfTrees(b.x, b.z, b.r + 1) || tries > 50) return b;
    }
  }

  // ---- the game: bomb, grenades, wrecks, respawns ------------------------------------------------------

  /** G: drop a grenade from under the drone, moving as the drone moves. False if there is none to drop. */
  dropGrenade(): boolean {
    if (this.droneLost || this.frozen || this.grenadesLeft <= 0) return false;
    this.grenadesLeft--;
    this.stats.grenadesDropped++;
    this.grenades.push({ x: this.x, y: this.y - GRENADE_DROP, z: this.z, vx: this.vx, vy: this.vy, vz: this.vz });
    this.onEvent?.({ kind: "grenade", left: this.grenadesLeft });
    return true;
  }

  /**
   * Where a grenade dropped now would reach the ground (ignoring trees and trucks), and after how many seconds.
   * For the player's aim and ?bomber=auto; null while there is no drone.
   */
  grenadeImpact(): { x: number; z: number; t: number } | null {
    if (this.droneLost) return null;
    const g = GAME.gravity;
    const y0 = this.y - GRENADE_DROP;
    const t = (this.vy + Math.sqrt(this.vy * this.vy + 2 * g * Math.max(0, y0))) / g;
    return { x: this.x + this.vx * t, z: this.z + this.vz * t, t };
  }

  private updateGrenades(ms: number) {
    for (let left = ms; left > 1e-6 && this.grenades.length; left -= GRENADE_STEP_MS) {
      const h = Math.min(GRENADE_STEP_MS, left) / 1000;
      for (let i = this.grenades.length - 1; i >= 0; i--) {
        const g = this.grenades[i];
        g.vy -= GAME.gravity * h;
        g.x += g.vx * h;
        g.y += g.vy * h;
        g.z += g.vz * h;
        const half = WORLD.size / 2;
        if (g.x > half) g.x -= WORLD.size;
        if (g.x < -half) g.x += WORLD.size;
        if (g.z > half) g.z -= WORLD.size;
        if (g.z < -half) g.z += WORLD.size;
        let hit = g.y <= 0;
        if (hit) g.y = 0;
        hit ||= this.hitsTree(g.x, g.y, g.z, GRENADE_R);
        hit ||= this.carAt(g.x, g.y, g.z, GRENADE_R) >= 0;
        if (!hit) continue;
        this.grenades.splice(i, 1);
        this.burst(g.x, g.y, g.z);
      }
    }
  }

  /** the slot of a truck touching a sphere of radius r at (x, y, z), or -1 */
  private carAt(x: number, y: number, z: number, r: number) {
    return this.cars.findIndex((c) => !!c && this.nearCar(c, x, y, z, r));
  }

  /** a grenade bursts: an explosion, and every truck with any part of it within the blast destroyed */
  private burst(x: number, y: number, z: number) {
    this.explosions.push({ x, y, z, t0: this.time, radius: GAME.blastRadius });
    let hit = false;
    for (let i = 0; i < this.cars.length; i++) {
      const c = this.cars[i];
      if (!c || !this.nearCar(c, x, y, z, GAME.blastRadius)) continue;
      hit = true;
      this.stats.grenadeKills++;
      this.destroyCar(i, "grenade");
    }
    this.onEvent?.({ kind: "burst", hit });
  }

  private destroyCar(i: number, by: "ram" | "grenade") {
    const car = this.cars[i];
    if (!car) return;
    this.wrecks.push({ x: car.x, z: car.z, heading: car.heading, t0: this.time, s: car.s });
    this.cars[i] = null;
    this.stats.carsDestroyed++;
    // the chase test wants its target back at once, ahead (after a ram: ahead of the respawned drone); otherwise a
    // destroyed truck stays gone until respawnCars(), or GAME.carRespawnMs with autoCarRespawn
    this.carRespawnAt[i] = this.carAhead === i ? this.time : this.autoCarRespawn ? this.time + GAME.carRespawnMs : Infinity;
    this.onEvent?.({ kind: "carDestroyed", index: i, by });
  }

  private loseDrone() {
    this.droneLost = true;
    this.respawnAt = this.time + GAME.respawnMs;
    this.stats.dronesLost++;
    this.vx = this.vy = this.vz = 0;
    this.yawRate = this.pitchRate = 0;
    this.hopUp = this.hopBack = 0;
    this.onEvent?.({ kind: "droneLost" });
  }

  /**
   * The player's "Respawn trucks": every destroyed truck back on the road out of the fly's sight (as placeCarHidden
   * puts them), and every wreck cleared away. Returns how many came back.
   */
  respawnCars(): number {
    this.wrecks = [];
    let n = 0;
    for (let i = 0; i < this.cars.length; i++) {
      if (this.cars[i] || this.carAhead === i) continue;
      this.placeCarHidden(i);
      this.carRespawnAt[i] = 0;
      this.onEvent?.({ kind: "carBack", index: i });
      n++;
    }
    return n;
  }

  /** A new drone at the start, level, with a full load of grenades. Only the world resets: the brain runs on. */
  private respawnDrone() {
    this.droneLost = false;
    this.x = START.x;
    this.z = START.z;
    this.y = DRONE.startAlt;
    this.heading = 0;
    this.pitch = 0;
    this.bank = 0;
    this.yawRate = this.pitchRate = 0;
    this.hopUp = this.hopBack = 0;
    this.bounceX = this.bounceZ = 0;
    this.stunnedUntil = 0;
    this.onGround = false;
    this.grenadesLeft = this.grenadesPerDrone;
    this.onEvent?.({ kind: "respawn" });
  }

  private tick(ms: number) {
    this.updateGrenades(ms);
    this.explosions = this.explosions.filter((e) => this.time - e.t0 < GAME.explosionMs);
    // wrecks stay on the road until the trucks are respawned, unless trucks come back by themselves
    if (this.autoCarRespawn) this.wrecks = this.wrecks.filter((w) => this.time - w.t0 < GAME.wreckMs);
    if (this.droneLost && this.time >= this.respawnAt) this.respawnDrone();
    for (let i = 0; i < this.cars.length; i++) {
      if (this.cars[i] || this.time < this.carRespawnAt[i]) continue;
      // the chase test's target comes back ahead of the drone, so not while there is no drone
      const ahead = this.carAhead === i;
      if (ahead && this.droneLost) continue;
      if (ahead) this.placeCarAhead(i);
      else this.placeCarHidden(i);
      this.onEvent?.({ kind: "carBack", index: i });
    }
  }

  // ---- the drone ---------------------------------------------------------------------------------------

  /** Advance by `ms` of brain time under the brain's commands. */
  update(ms: number, cmd: Commands) {
    const dt = ms / 1000;
    this.time += ms;
    for (let i = 0; i < this.cars.length; i++) if (this.cars[i]) this.driveCar(i, ms);
    this.tick(ms);
    if (this.frozen || this.droneLost) {
      this.yawRate = 0;
      this.pitchRate = 0;
      return;
    }
    const stunned = this.time < this.stunnedUntil;
    const k = 1 - Math.exp(-DRONE.response * dt);
    this.yawRate += ((stunned || this.steerOff ? 0 : cmd.yawRate) - this.yawRate) * k;
    this.pitchRate += ((stunned || this.steerOff ? 0 : cmd.pitchRate) - this.pitchRate) * k;

    this.heading = angleDiff(this.heading + this.yawRate * DEG * dt);
    const maxP = DRONE.maxPitch * DEG;
    this.pitch = Math.max(-maxP, Math.min(maxP, this.pitch + this.pitchRate * DEG * dt));
    // bank into turns, cosmetic only: the fly's eye stays level on its gimbal
    this.bank += (Math.max(-0.6, Math.min(0.6, this.yawRate * 0.006)) - this.bank) * k;

    if (cmd.escape && !stunned && !this.steerOff) {
      this.hopUp = 1;
      this.hopBack = 1;
      this.stats.escapes++;
    }

    const fx = Math.sin(this.heading);
    const fz = -Math.cos(this.heading);
    const speed = stunned ? 0 : DRONE.cruise * this.speedScale;
    const cp = Math.cos(this.pitch);
    let vx = fx * cp * speed;
    let vz = fz * cp * speed;
    let vy = Math.sin(this.pitch) * speed;
    // the escape hop, decaying
    vx -= fx * this.hopBack * ESCAPE.back;
    vz -= fz * this.hopBack * ESCAPE.back;
    vy += this.hopUp * ESCAPE.climb;
    const decay = Math.exp(-ms / ESCAPE.decayMs);
    this.hopUp *= decay;
    this.hopBack *= decay;

    // stun bounce: pushed back out of whatever was hit
    if (stunned) {
      vx += this.bounceX;
      vz += this.bounceZ;
      this.bounceX *= decay;
      this.bounceZ *= decay;
    }

    const ox = this.x;
    const oz = this.z;
    this.x += vx * dt;
    this.z += vz * dt;
    this.y += vy * dt;
    // keep coordinates bounded: the world is a torus
    const half = WORLD.size / 2;
    if (this.x > half) this.x -= WORLD.size;
    if (this.x < -half) this.x += WORLD.size;
    if (this.z > half) this.z -= WORLD.size;
    if (this.z < -half) this.z += WORLD.size;
    this.stats.distance += Math.hypot(wrapDelta(this.x - ox), wrapDelta(this.z - oz));

    // skimming the ground: a contact counts once, however long the brain keeps pushing it down
    const touching = this.y <= DRONE.floor + 1e-3 && this.pitch <= 0;
    if (this.y < DRONE.floor) {
      this.y = DRONE.floor;
      if (this.pitch < 0) {
        this.pitch = 0;
        this.pitchRate = 0;
      }
    }
    if (touching && !this.onGround) this.stats.groundHits++;
    this.onGround = touching;
    if (touching) this.stats.groundMs += ms;
    if (touching && this.fatal && !this.ghost) {
      this.crash("ground");
      return;
    }
    if (this.y > DRONE.ceiling) {
      this.y = DRONE.ceiling;
      if (this.pitch > 0) this.pitch = 0;
    }

    if (this.ghost) return;

    for (const t of this.trees) {
      const dx = wrapDelta(this.x - t.x);
      const dz = wrapDelta(this.z - t.z);
      const dy = this.y - t.canopyY;
      const d = Math.hypot(dx, dz);
      const trunk = d < t.trunkR + DRONE.radius && this.y < t.trunkTop;
      const canopy = Math.hypot(d, dy) < t.canopyR + DRONE.radius;
      if (!trunk && !canopy) continue;
      if (this.fatal) {
        this.stats.treeHits++;
        this.crash("tree");
        return;
      }
      // out of the tree, then a short stun while the brain keeps running
      const nx = dx / Math.max(d, 1e-6);
      const nz = dz / Math.max(d, 1e-6);
      if (canopy) {
        const n = Math.hypot(d, dy) || 1;
        const out = t.canopyR + DRONE.radius + 0.01;
        this.x = t.x + (dx / n) * out;
        this.z = t.z + (dz / n) * out;
        this.y = t.canopyY + (dy / n) * out;
      } else {
        this.x = t.x + nx * (t.trunkR + DRONE.radius + 0.01);
        this.z = t.z + nz * (t.trunkR + DRONE.radius + 0.01);
      }
      if (!stunned) {
        this.stats.treeHits++;
        this.stunnedUntil = this.time + DRONE.stunMs;
        this.bounceX = nx * DRONE.bounce;
        this.bounceZ = nz * DRONE.bounce;
      }
    }
    // what a grenade dropped now inherits: the flight velocity (not a tree's push-out), with no sinking through the
    // floor or rising through the ceiling
    this.vx = vx;
    this.vy = (this.y <= DRONE.floor && vy < 0) || (this.y >= DRONE.ceiling && vy > 0) ? 0 : vy;
    this.vz = vz;

    for (let i = 0; i < this.balloons.length; i++) {
      const b = this.balloons[i];
      const d = Math.hypot(wrapDelta(this.x - b.x), this.y - b.y, wrapDelta(this.z - b.z));
      if (d < b.r + DRONE.radius) {
        const isFood = this.food?.kind === "balloon" && this.food.index === i;
        this.onReach?.({ kind: "balloon", index: i }, isFood);
        this.stats.popped++;
        if (isFood) this.stats.foodReached++;
        this.lastPop = this.time;
        // a new balloon in the same slot; the designation stays with the slot, so the new one is food too
        this.balloons[i] = this.spawnBalloon();
        if (this.fatal) {
          // the bomb goes off on any balloon; hitting the food one is the point, so only the others count as crashes
          this.explosions.push({ x: b.x, y: b.y, z: b.z, t0: this.time, radius: GAME.ramBlast });
          if (!isFood) this.stats.crashBalloon++;
          this.loseDrone();
          return;
        }
      }
    }

    // ramming a truck sets off the drone's bomb: both go. The designation stays with the truck's slot, as with the
    // balloons, so the one that comes back in it is food too. Ramming any other truck is a crash (crashCar).
    const hitCar = this.carAt(this.x, this.y, this.z, DRONE.radius);
    const car = this.cars[hitCar];
    if (car) {
      const isFood = this.food?.kind === "car" && this.food.index === hitCar;
      this.onReach?.({ kind: "car", index: hitCar }, isFood);
      this.stats.rams++;
      if (isFood) this.stats.foodReached++;
      else this.stats.crashCar++;
      this.lastRam = this.time;
      this.explosions.push({ x: car.x, y: CAR.height / 2, z: car.z, t0: this.time, radius: GAME.ramBlast });
      this.destroyCar(hitCar, "ram");
      this.loseDrone();
    }
  }

  /** Contact with something that isn't the food: the bomb goes off where the drone is. */
  private crash(cause: "tree" | "ground") {
    this.explosions.push({ x: this.x, y: this.y, z: this.z, t0: this.time, radius: GAME.ramBlast });
    if (cause === "tree") this.stats.crashTree++;
    else this.stats.crashGround++;
    this.loseDrone();
  }

  /** is a point within r of a tree's trunk or canopy? */
  private hitsTree(x: number, y: number, z: number, r: number) {
    for (const t of this.trees) {
      const d = Math.hypot(wrapDelta(x - t.x), wrapDelta(z - t.z));
      if ((d < t.trunkR + r && y < t.trunkTop) || Math.hypot(d, y - t.canopyY) < t.canopyR + r) return true;
    }
    return false;
  }

  /** is a sphere of radius r at (x, y, z) touching a truck's box? */
  private nearCar(car: Car, x: number, y: number, z: number, r: number) {
    const dx = wrapDelta(x - car.x);
    const dz = wrapDelta(z - car.z);
    const c = Math.cos(car.heading);
    const s = Math.sin(car.heading);
    // into the truck's frame: +x its right, -z its front
    const lx = dx * c + dz * s;
    const lz = -dx * s + dz * c;
    const qx = Math.max(Math.abs(lx) - CAR.halfWidth, 0);
    const qz = Math.max(Math.abs(lz) - CAR.halfLength, 0);
    const qy = Math.max(y - CAR.height, 0);
    return Math.hypot(qx, qy, qz) < r;
  }

  /** World position of a target, or null if it no longer exists. */
  targetPosition(t: Target): [number, number, number] | null {
    if (t.kind === "car") {
      const c = this.cars[t.index];
      return c ? [c.x, CAR.height / 2, c.z] : null;
    }
    const b = this.balloons[t.index];
    return b ? [b.x, b.y, b.z] : null;
  }

  /** Is a tree between the drone and the point (x, y, z)? Its canopy near the line of sight, or its trunk under it. */
  private occluded(x: number, y: number, z: number) {
    const dx = wrapDelta(x - this.x);
    const dz = wrapDelta(z - this.z);
    const dy = y - this.y;
    const flat = Math.hypot(dx, dz);
    for (const t of this.trees) {
      const px = wrapDelta(t.x - this.x);
      const pz = wrapDelta(t.z - this.z);
      const along = (px * dx + pz * dz) / Math.max(flat * flat, 1e-6);
      if (along <= 0 || along >= 1) continue;
      const side = Math.hypot(px - along * dx, pz - along * dz);
      const ty = this.y + along * dy;
      if (Math.hypot(side, ty - t.canopyY) < t.canopyR || (side < t.trunkR && ty < t.trunkTop)) return true;
    }
    return false;
  }

  /**
   * Where a target sits in the fly's field: azimuth (deg, + = right), elevation (deg, relative to the gaze), and
   * distance. `visible` is the geometry only (in the eyes' reach, near enough, no tree in the way), not what
   * the brain made of it.
   */
  locate(t: Target) {
    const p = this.targetPosition(t);
    if (!p) return null;
    const dx = wrapDelta(p[0] - this.x);
    const dz = wrapDelta(p[2] - this.z);
    const dy = p[1] - this.y;
    const flat = Math.hypot(dx, dz);
    const dist = Math.hypot(flat, dy);
    const az = angleDiff(Math.atan2(dx, -dz) - this.heading) / DEG;
    const el = Math.atan2(dy, flat) / DEG - this.pitch / DEG;
    return { az, el, dist, occluded: this.occluded(p[0], p[1], p[2]) };
  }

  /** Bearing, elevation and distance of the nearest balloon (for the HUD and the trace). */
  nearestBalloon() {
    let best = { az: 0, el: 0, dist: Infinity };
    for (let i = 0; i < this.balloons.length; i++) {
      const l = this.locate({ kind: "balloon", index: i });
      if (l && l.dist < best.dist) best = l;
    }
    return best;
  }

  /**
   * The balloon or truck a ray from `ro` along unit `r` hits first, if a tree or the ground isn't in the way.
   * The same geometry scene.wgsl draws, done on the CPU for clicking.
   */
  pick(ro: [number, number, number], r: [number, number, number]): Target | null {
    let best = Infinity;
    let hit: Target | null = null;
    const near = (x: number, z: number): [number, number] => [ro[0] + wrapDelta(x - ro[0]), ro[2] + wrapDelta(z - ro[2])];
    for (let i = 0; i < this.balloons.length; i++) {
      const b = this.balloons[i];
      const [cx, cz] = near(b.x, b.z);
      const t = raySphere(ro, r, [cx, b.y, cz], b.r * 1.3); // a little generous: they are small on screen
      if (t < best) (best = t), (hit = { kind: "balloon", index: i });
    }
    for (let i = 0; i < this.cars.length; i++) {
      const c = this.cars[i];
      if (!c) continue;
      const [cx, cz] = near(c.x, c.z);
      const t = raySphere(ro, r, [cx, CAR.height / 2, cz], CAR.halfLength * 1.2);
      if (t < best) (best = t), (hit = { kind: "car", index: i });
    }
    if (!hit) return null;
    if (r[1] < 0 && -ro[1] / r[1] < best) return null; // the ground is nearer
    for (const tr of this.trees) {
      const [cx, cz] = near(tr.x, tr.z);
      if (raySphere(ro, r, [cx, tr.canopyY, cz], tr.canopyR) < best) return null;
      const ox = ro[0] - cx;
      const oz = ro[2] - cz;
      const a = r[0] * r[0] + r[2] * r[2];
      const b = ox * r[0] + oz * r[2];
      const disc = b * b - a * (ox * ox + oz * oz - tr.trunkR * tr.trunkR);
      if (a < 1e-8 || disc < 0) continue;
      const t = (-b - Math.sqrt(disc)) / a;
      const y = ro[1] + r[1] * t;
      if (t > 0 && t < best && y >= 0 && y <= tr.trunkTop) return null;
    }
    return hit;
  }
}

function raySphere(ro: number[], r: number[], c: number[], radius: number) {
  const o = [ro[0] - c[0], ro[1] - c[1], ro[2] - c[2]];
  const b = o[0] * r[0] + o[1] * r[1] + o[2] * r[2];
  const disc = b * b - (o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - radius * radius);
  if (disc < 0) return Infinity;
  const t = -b - Math.sqrt(disc);
  return t > 0 ? t : Infinity;
}
