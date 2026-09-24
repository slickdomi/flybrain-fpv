// Every constant of the drone sim, split into what is the fly's and what is mine.
//
// The brain is flybrain (MaleCNS v1.0, 166,700 neurons), vendored from fly-addiction. The drone, its physics,
// the world and every gain that turns a firing rate into motion are modeled: nothing in the connectome knows
// about rotors. Where a value came from another project's measurement, the comment says which.

export const DATA_URL = `${import.meta.env.BASE_URL}data/malecns-v1`;

/** The running model: fly-addiction's MODEL, which is Aimbug's validated set. */
export const MODEL = {
  dt: 0.5,
  vThreshold: 7,
  vReset: 0,
  tauM: 20,
  tauS: 5,
  tRefractory: 2.2,
  tDelay: 1.8,
  wSyn: 0.275,
  adapt: 2.0,
  tauAdapt: 200,
  gradedDt: 8.33,
  gradedTau: 20,
  gradedGain: 2.3,
  gradedMin: -1,
  gradedMax: 4,
  prune: 0.005,
  laminaWeight: 0.5,
  coupling: 3400,
  actTau: 120,
};

/**
 * The world, as the fly sees it. Luminance only; view.wgsl colours it for you and the eye never sees that.
 *
 * Kept nearly uniform on purpose (PacFly, fly-addiction): contrast everywhere drives the looming detectors on
 * both sides and a busy view tips the lLN1_bc clique into a seizure, while a view with no contrast at all
 * silences the brain. Here the ground is a shade darker than the sky, so the horizon is always in view.
 */
export const WORLD = {
  /** the world is a torus this many units across: fly off one edge and you come back on the other */
  size: 100,
  sky: 0.5,
  /** fraction darker than the sky (?ground=) */
  groundContrast: 0.12,
  /**
   * Trees, fraction darker than the sky (?trees=). Low, like the pylons they replaced: this fly steers toward
   * dark, so dark canopies would pull the drone into them, and a busy view tips the brain into seizures.
   */
  treeContrast: 0.2,
  treeCount: 28,
  /**
   * A burnt forest: dead, branchless trunks, taller than the drone's cruise height, with no canopy (radius 0; the
   * canopy centre sits on the trunk top, so the collision there only caps the trunk). A thin trunk is a much
   * smaller target than a canopy: a 0.3 trunk plus the drone's 0.5 is 1.6 units across, against 4.6 for the old
   * 1.8 canopies. Before this (the scenario agent's low trees): 0.35 / 2.6 / 3.5 / 1.8; before that 5 / 6.5 / 2.2,
   * which the drone hit about 9 times a minute.
   */
  trunkRadius: 0.3,
  trunkTop: 11,
  canopyY: 11,
  canopyRadius: 0,
  /** the road loop: its mean radius from the field's centre, its width, and how much darker than the ground */
  roadRadius: 30,
  roadWidth: 5,
  roadContrast: 0.04,
  /**
   * Targets the player has NOT marked as food. Mid-grey rather than black, so that marking one
   * makes it stand out (MARK.markLum). Large, because the eye samples every 4.8 deg: Aimbug's fly lost a target below
   * ~13 deg radius, and a radius-2 balloon is ~9.5 deg at 12 units.
   */
  balloonLum: 0.3,
  balloonRadius: 2,
  /** none in the game: the trucks are the targets. ?balloons=N brings them back (the first flights used 3). */
  balloonCount: 0,
  /** a new balloon appears this far away, at a random bearing */
  spawnMin: 14,
  spawnMax: 30,
  spawnAltMin: 3,
  spawnAltMax: 12,
};

/** MODELED: the quadcopter. It flies along the fly's gaze at a constant cruise speed. */
export const DRONE = {
  /** units per brain-second */
  cruise: 2.5,
  /** above the canopies (WORLD.canopyY + canopyRadius = 5.3); was 6, inside them. Also the respawn altitude. */
  startAlt: 8,
  floor: 0.6,
  /** the drone can't go higher (ALTITUDE pulls it down well before; was 30, where it could fly out of the lock's reach) */
  ceiling: 14,
  /** body radius for collisions */
  radius: 0.5,
  /** 1/s, how fast heading and pitch rate follow the command (rotor inertia) */
  response: 6,
  /** deg, the pitch the gimbal allows */
  maxPitch: 50,
  /** after hitting a tree: pushed back at this speed, brain still running, controls ignored for stunMs */
  bounce: 3,
  stunMs: 800,
};

/** How firing rates become flight. The cells are the fly's; the gains and filters are mine. */
export const CONTROL = {
  /**
   * deg/s of yaw per Hz of DNa02 right - left. fly-addiction's walking fly used 2 (at 10 it span on the spot);
   * Aimbug's turret used 10. A drone at cruise needs to turn onto a balloon within a few seconds.
   */
  yawGain: 3,
  /**
   * ms: the running mean of the DNa02 difference is subtracted, so only changes steer. fly-addiction measured
   * both extremes: off, a standing imbalance circled the fly forever; at 1000 ms it cancelled real approach.
   */
  yawAdaptMs: 5000,
  /** Aimbug's pitch: rate = gain * (DNp53 - pitchRef * (LC4 + LPLC2)) - spring * pitch */
  pitchGain: 8,
  pitchRef: 0.1,
  /** 1/s, pull back toward level (Aimbug's neck spring) */
  pitchSpring: 0.3,
  /**
   * ms, same idea as yawAdaptMs, and new here: Aimbug's head could sit tilted forever, but a drone flying along
   * a tilted gaze climbs or dives forever. 0 = off.
   */
  pitchAdaptMs: 5000,
  /**
   * Giant fiber escape: DNp01 (left + right) above its own 2 s mean by this many Hz makes the drone hop up and
   * back. PacFly's `?panicBy=gf` used 60. It idles at 50-80 Hz with something in view (Aimbug). 0 = off.
   */
  escapeHz: 60,
  escapeMeanMs: 2000,
  escapeRefractoryMs: 1500,
  /** units/s of the hop, decaying with escapeDecayMs */
  escapeClimb: 5,
  escapeBack: 3,
  escapeDecayMs: 400,
  /** ms of brain time at the start with no steering, while the rates and the filters above settle */
  warmupMs: 1500,
};

/** What the player can stimulate. Constant drive (mV above rest; threshold is 7) while the key is held. */
export interface Stim {
  key: string;
  label: string;
  what: string;
  /** flybrain Connectome.select() query */
  type: string | RegExp;
  side?: "left" | "right";
}

export const STIMS: Stim[] = [
  { key: "1", label: "LC10a L", what: "left eye's small-object cells: makes it think something is on the left", type: "LC10a", side: "left" },
  { key: "2", label: "LC10a R", what: "the same for the right eye", type: "LC10a", side: "right" },
  { key: "3", label: "DNa02 L", what: "the left steering neuron", type: "DNa02", side: "left" },
  { key: "4", label: "DNa02 R", what: "the right steering neuron", type: "DNa02", side: "right" },
  { key: "5", label: "DNp53", what: "makes it look up, so the drone climbs", type: "DNp53" },
  { key: "6", label: "LC4", what: "looming detectors: something is about to hit it", type: "LC4" },
  { key: "7", label: "DNp01", what: "the giant fiber, its escape reflex", type: "DNp01" },
  { key: "8", label: "MDN", what: "makes a real fly walk backwards (not wired to the drone)", type: "MDN" },
  { key: "9", label: "P1", what: "male arousal cells", type: /^pC1_/ },
];

export const LOOP = {
  /** cap on brain ms requested per frame; the adaptive budget backs off below it when frames slow */
  maxBrainMsPerFrame: 50,
  /** the view is traced at no less than this fraction of the canvas's pixels when frames run late (renderer.resScale) */
  minRenderScale: 0.5,
  maxStepsPerFrame: 128,
  maxSpikesPerStep: 8192,
  /**
   * the panel's brain and eye views and the HUD's text and bars are redrawn at most this often (ms): every other frame at
   * 60 Hz. They are there to be glanced at, and redrawing them costs the GPU (166,700 points, twice) and the page a
   * layout each time
   */
  panelMs: 31,
  seizureSpikesPerSec: 90000,
  seizureMs: 300,
};

/** Paint-a-phantom: clicking the eye view drives LC10a cells whose receptive field is near the click. */
export const PAINT = {
  type: "LC10a",
  /** deg */
  radius: 15,
};

/**
 * MODELED: the military trucks, the chase targets. Big, for the same reason the balloons are: the eye samples every
 * 4.8 deg. CAR.count of them drive the road loop (world.road) and never leave it. To the fly each is a flat grey box
 * of these dimensions (the designated one black); view.wgsl draws a 6x6 cargo truck inside the box.
 */
export const CAR = {
  /** how many trucks drive the road (world.cars; ?cars= changes it, ?car=0 or ?cars=0 for none; at most 8) */
  count: 5,
  /**
   * Half length, half width, height (world units). A truck, not the old 4 x 2 x 1.5 car: 5.6 long and 2.4 tall, the
   * proportions of a 6x6 cargo truck at the old car's 2-unit width. The width stays so that trucks passing in
   * opposite lanes (1 either side of the centre line) only touch.
   */
  halfLength: 2.8,
  halfWidth: 1,
  height: 2.4,
  /** unmarked; the marked food is MARK.markLum */
  lum: 0.3,
  /** units per brain-second, below the drone's cruise so it can be caught */
  speed: 1.2,
  /**
   * It keeps to the right lane: its centre this far right of the road's centre line. Road half-width 2.5; at 1.25
   * an outer corner poked 0.1 past the edge on the tightest bends.
   */
  lane: 1,
  /** a truck is never put on the road nearer than this (centre to centre) to another one */
  spacing: 8,
  /**
   * A truck stops this far along the road (centre to centre) behind a wreck in either lane, or behind the next truck
   * in its own lane: a truck length and 2 units. It slows over the last brakeDist before that, to crawl.
   */
  stopGap: 7.6,
  brakeDist: 4,
  /** the fraction of its speed a braking truck slows to before it stops */
  crawl: 0.15,
  /** view only: the wheels turn by distance / this, and stop when the truck does */
  tyreRadius: 0.5,
  /**
   * At the start, this many trucks are on the road ahead of the start pose (heading 0 from the field's centre):
   * within startAzDeg of straight ahead, startMin-startMax away, no trunk in the way. The FPV view is 130 deg across.
   */
  startInView: 3,
  startAzDeg: 45,
  startMin: 15,
  startMax: 40,
  /**
   * A wrecked truck comes back on the road "out of sight": behind the fly (beyond this azimuth; the eyes reach
   * about 140 deg), farther than hideDist, or behind a tree, and never nearer than hideMin.
   */
  hideAzDeg: 150,
  hideDist: 40,
  hideMin: 12,
};

/**
 * MODELED: the game. None of it reaches the brain except through what the eye sees (the trucks and the drone's
 * position); grenades, explosions and wrecks are drawn for the player only.
 */
export const GAME = {
  /** the drone carries a bomb: ramming a truck blows up both. The drone comes back this much later, at the start. */
  respawnMs: 3000,
  /**
   * A new drone gets a fresh brain: voltages, synaptic input and adaptation cleared (the spiking state), and
   * the start-up warm-up (CONTROL.warmupMs) again before it steers. ?brainreset=0 keeps the old brain running.
   */
  resetBrainOnRespawn: true,
  /**
   * In the game a destroyed truck stays destroyed, its wreck blocking the road, until the player respawns the trucks
   * (the Respawn trucks button). Only the chase test brings them back by themselves (world.autoCarRespawn): this much
   * later, and its target at once, ahead; there wrecks also clear after wreckMs.
   */
  carRespawnMs: 3000,
  /** view-only size of the ram's explosion */
  ramBlast: 3,
  /**
   * Grenades are switched off for now: no G key, no button, none on the drone. The code is all still there;
   * ?grenades=1 turns them back on.
   */
  grenadesEnabled: false,
  /** grenades per drone (key G). They fall ballistically and burst on the ground, a tree or a truck. */
  grenades: 3,
  /** units/s^2 (a world unit is about a metre: a truck is 5.6 long) */
  gravity: 9.8,
  /** a burst destroys every truck with any part of it within this */
  blastRadius: 4,
  /** how long an explosion stays in world.explosions, and (chase test only) a wreck in world.wrecks (brain ms) */
  explosionMs: 1500,
  wreckMs: 12000,
};

/**
 * The mushroom body, loaded as fly-addiction loads it (see its README, "The mushroom body is deaf to sight"):
 * KC->KC silenced as a correction (those synapses act through inhibitory mAChR-B), and the optic lobe's
 * projections onto the visual Kenyon cells x20, a declared fudge -- at 1x those KCs never fire.
 */
export const MB = {
  silenceKcKc: true,
  visualGain: 20,
  visualPre: "^(aMe|MeVP|LoVP)",
  visualPost: "^KC(g-d|ab-p)$",
  /** ms, weights relax back to the connectome's over this */
  memoryMs: 30 * 60 * 1000,
  /** LearnedValue floors and trace, fly-addiction's */
  floor: 4000,
  avoidFloor: 20000,
  traceMs: 150,
  /**
   * MODELED. deg/s of yaw per unit of (adapted) right - left learned value. fly-addiction's value, chosen there
   * before any trial and left alone on purpose: tuning it until the fly chases would write the result in.
   */
  turnGain: 120,
  adaptMs: 5000,
  /** learning pauses while the seizure clique fires above this, and for learnHoldMs after a seizure (PacFly) */
  learnPauseHz: 20,
  learnHoldMs: 3000,
};

/**
 * MODELED: what "designate as food" does. Nothing in this brain can taste a car or a balloon, and taste
 * cannot reach the reward cells anyway (fly-addiction, finding 1), so the reward is delivered the way
 * fly-addiction delivers a bite: pulses into the PAM dopamine cells. What the fly then learns is real.
 */
export const FOOD = {
  /** mV into PAM per pulse (fly-addiction's NEURO.rewardMv) */
  rewardMv: 12,
  /** while the food is in view, a pulse of sightPulseMs every sightEveryMs */
  sightPulseMs: 100,
  sightEveryMs: 500,
  /** "in view": inside this azimuth and nearer than this */
  viewAzDeg: 120,
  viewDist: 30,
  /** reaching it: one long pulse (fly-addiction's drug pulse) */
  contactPulseMs: 500,
};

/**
 * MODELED: the target marker. This model's eye has no colour (R1-R6 and the lamina only), so "marking" the food
 * works in luminance: the marked target turns black to the fly while unmarked ones stay mid-grey. Black is also
 * what this fly steers toward hardest. It changes the brain's input, so ?mark=off turns it off, and the chase
 * test runs an arm without it.
 */
export const MARK = {
  markLum: 0.0,
  /**
   * Take the marker off when the food is nearer than this (units; 0 = never, ?marknear=). Under test: with the dive
   * assist the brain seized about 1.6 times per ram, just before contact, with the car black and filling the view;
   * with the car grey (no marker) the same approaches gave none. At 6, with painting also off inside 6: 11 -> 0.7
   * seizures a run (1, 0, 1), 3.67 -> 2.56 hits/min. A brain rebooting every ten seconds is worse than fewer hits.
   */
  stopWithin: 6,
};

/**
 * How designating a target as food steers the fly (?food=). Chosen by the five-arm chase comparison
 * (scripts/test-controls.sh, suite "food"): see README.
 *   none    nothing: the baseline arm
 *   marker  MARK: the food turns black to the fly's eye
 *   paint   SEEK: the LC10a cells whose connectome receptive field covers the food get a constant drive
 *   both    marker + paint
 *   learn   fly-addiction's food drive: PAM reward pulses, the mushroom body's own learning, the LearnedValue
 *           readout steering; plus the marker. Loads the mushroom body with KC->KC silenced and the x20 visual
 *           gain (MB), so it is the only mode that runs a modified brain.
 */
export type FoodMode = "none" | "marker" | "paint" | "both" | "learn";
export const FOOD_MODES: FoodMode[] = ["none", "marker", "paint", "both", "learn"];

/** MODELED: automatic target painting (food modes paint and both). The steering is the fly's LC10 -> DNa02 pathway. */
export const SEEK = {
  /** the default food mode */
  mode: "both" as FoodMode,
  /** mV: 20 turned the drone the right way in every trial of the control tests (paint +-40 deg, open and closed loop) */
  paintMv: 20,
  /** deg around the food's direction, as the player's own painting (PAINT.radius) */
  paintRadius: 15,
  /**
   * Stop painting when the food is nearer than this (units; 0 = never stop, ?paintnear=). With the dive assist, the
   * brain seized in pairs about a second before nearly every ram, the car 3-6 units off and filling the view while
   * 20 mV still went into LC10a (21 seizures in 180 s, seed 1). At 6: 18 -> 11 seizures a run, 4.11 -> 3.67 hits/min.
   */
  stopWithin: 6,
  /** brain ms between re-selecting which cells to paint, as the food moves across the eye */
  paintEveryMs: 100,
};

/**
 * MODELED: target lock. While the food is in view within azDeg of straight ahead and nearer than dist, the escape
 * hop is off (the food looms as the drone closes in, and the hop turned every approach into a retreat) and the
 * drone flies faster, up to speedUp x cruise when the food is dead ahead. The same in every food mode, so the
 * comparison between modes is only about how the fly is told where the food is.
 */
export const LOCK = {
  azDeg: 40,
  dist: 30,
  speedUp: 2,
  /**
   * While locked, the yaw adaptation (CONTROL.yawAdaptMs) holds still instead of following the DNa02 difference
   * (?freeze=0|1). The adaptation is there to stop a standing left-right imbalance circling the drone forever;
   * during a pursuit the standing difference IS the food's direction, and following it cancels the turn toward
   * the food within seconds. The five-arm chase comparison had DNa02 pointing at the car 51-57% of the time.
   * Measured (3 seeds x 180 s, food=both): the car within 30 deg ahead 8% -> 13% of the time, on every seed.
   */
  freezeAdapt: true,
  /**
   * MODELED, and plainly an assist (?dive=0|1): while locked and nearer than diveDist, the gaze pitch is steered
   * toward the food's elevation at diveGain deg/s per deg. The fly's pitch pathway does not track elevation here
   * (see "First flights"), and a 1.5-unit car can't be rammed from the cruise altitude without it. Left and
   * right stay the fly's. Measured (with the freeze, 3 seeds x 180 s): 0 -> 4.11 hits/min with food=both, 0.56 with
   * food=none, so the fly's own steering, told where the food is, does most of the aiming.
   */
  dive: true,
  diveDist: 15,
  diveGain: 2,
};

/**
 * Test-only stimulation groups (no key): the smell channels, for measuring whether one antenna smelling more than the
 * other steers the fly (osmotropotaxis) before any smell is built. ?bench=stim&stims=vornL,vornR,... Earlier projects
 * found any drive on a food glomerulus's ORNs or PNs sets off the lLN1_bc seizure clique, and the V (CO2) PNs the only
 * calm channel; CO2 is aversive to real flies. "Left" is the cell's own hemisphere.
 */
export const TEST_STIMS: Stim[] = [
  { key: "vornL", label: "ORN_V L", what: "CO2 receptor neurons, left", type: "ORN_V", side: "left" },
  { key: "vornR", label: "ORN_V R", what: "CO2 receptor neurons, right", type: "ORN_V", side: "right" },
  { key: "vpnL", label: "V PN L", what: "CO2 projection neurons, left", type: /^V_(ilPN|l2PN)$/, side: "left" },
  { key: "vpnR", label: "V PN R", what: "CO2 projection neurons, right", type: /^V_(ilPN|l2PN)$/, side: "right" },
  { key: "dm1L", label: "ORN_DM1 L", what: "a food-odour glomerulus (vinegar), left", type: "ORN_DM1", side: "left" },
  { key: "dm1R", label: "ORN_DM1 R", what: "a food-odour glomerulus (vinegar), right", type: "ORN_DM1", side: "right" },
];

/**
 * MODELED obstacle avoidance from the fly's own looming cells (?avoid= gain, 0 = off; ?avoidadapt= ms): the drone
 * turns away from the side whose LC4 + LPLC2 rise more, by gain deg/s per Hz of that right-minus-left difference
 * above its running mean (adapted like DNa02, but faster, since a looming obstacle is a matter of a second or two).
 * PacFly's "fear" did this from bursts and it did not help there; the approach test (bench.ts) measures first
 * whether the asymmetry points the right way and how early. Off while the food is locked (LOCK), as the escape hop is.
 */
export const AVOID = {
  gain: 0,
  adaptMs: 1500,
  /**
   * The turn-away and the escape hop also stay off while the target is within LOCK.diveDist in front (in view or
   * not), and for holdMs after that and after the lock ends; then the turn-away fades back in over fadeMs. Without
   * this, a lock that dropped for a moment late in a dive (the truck past 40 deg, or behind a trunk) turned the
   * drone hard away from the truck looming beside it.
   */
  holdMs: 1500,
  fadeMs: 1000,
};

/**
 * Forest mode (the default; ?forest=0 or the button for thin trunks): thick trunks plus the looming turn-away.
 * Measured 2026-09-23 (3 + 3 seeds x 180 s, every contact fatal): the fly's looming cells take a 1-unit trunk's side
 * 3-5 s ahead (a 0.3 trunk: under 1 s), and with the turn-away at 6 free flight had 0 crashes in 6 of 6 runs against
 * 1.7-3.4 crashes/min without it. The cost: the chase hit 2.11 targets/min against 3.56 for the thin trunks without
 * avoidance, because thick trunks hide and distract (1.89/min with them and no avoidance); 2.67/min (6 seeds) since
 * AVOID.holdMs keeps the turn-away off through the dive and a new drone gets a fresh brain (2026-09-24). On by
 * default, since a crash costs the drone.
 */
export const FOREST = {
  default: true,
  trunkRadius: 1,
  avoidGain: 6,
};

/**
 * MODELED: an altitude limit. The fly's pitch pathway can climb for good (DNp53 up while the looming cells quieten
 * high above the trunks: runs spent 77% of their time above 14 units), and from 20-30 units up a truck is out of the
 * target lock's reach. Above softAlt the gaze is pulled toward a descent, descentPerUnit deg per unit above it (at
 * most maxDescent), at pullGain per second; the brain's own pitch command still adds on top. DRONE.ceiling is the
 * hard limit. Off while the dive assist has the pitch.
 */
export const ALTITUDE = {
  softAlt: 10,
  descentPerUnit: 4,
  maxDescent: 25,
  pullGain: 1.5,
};
