# Flybrain FPV

A male fruit fly's connectome flies an FPV strike drone through its own compound eyes, live in the browser on WebGPU. You pick a truck; the fly chases it through a burnt forest; the drone blows up on contact with anything.

**Play it at [flybrain-fpv.domi.zip](https://flybrain-fpv.domi.zip)** in a browser with WebGPU (recent Chrome or Edge; Safari 26; Firefox with WebGPU switched on). It downloads the 38 MB connectome first.

All 166,700 neurons of [MaleCNS v1.0](https://male-cns.janelia.org/) run in real time: 13.2 M spiking and 3.9 M graded connections. The simulation is [`flybrain`](packages/flybrain), vendored unchanged from `../fly-addiction` (which took it from `../fly-games` and added a compute eye that samples the scene on the GPU).

## What the fly does, what the game does

**The fly:** every neuron and synapse is the real fly's. It sees the world through its own eyes and decides which way to turn and climb, and when to jump.

**The game:** everything else. The drone and its speed, the gains that turn firing rates into flight, the escape hop, the target marker, the painting, the lock and the dive assist, the turn away from looming trees, and the world: trucks, road, trees, the warhead, the respawns.

| drone | driven by | from |
|---|---|---|
| yaw | DNa02 right − left, 3 °/s per Hz, 5 s adaptation | Aimbug, PacFly, fly-addiction |
| gaze pitch (climb, dive) | DNp53 − 0.1 × (LC4 + LPLC2), 8 °/s per Hz, springs back to level | Aimbug's head pitch |
| escape hop | the DNp01 giant fiber bursting 60 Hz above its 2 s mean | PacFly |
| forward speed | constant, 2.5 units/s (up to 2× when locked on) | game |

The fly's eye sits on a gimbal: it pitches with the flight path but never rolls.

## Playing

```sh
sh scripts/install.sh      # npm install in Docker, with a dependency cooldown (.npmrc)
sh scripts/dev.sh          # http://localhost:5173 (PORT= to change) in a WebGPU browser
```

- **Right-click a truck, or press F** for the one nearest the middle of the view, to make it the target. The fly does the rest.
- **Target help** switches the target drive off and on, so you can watch the fly on its own.
- **Forest mode** (on by default) has thick trunks the fly can see coming, and it turns away from them. Off gives thin trunks: the fly finds trucks more often, but crashes a lot more. The button reloads the page.
- **Q W E R** pick a view: FPV cam (the default), fly eye, chase (drag to orbit, scroll to zoom), ommatidia. **V** cycles them.
- **Drag on the FPV or eye view** to paint a phantom: the LC10a cells looking at the pointer get a constant drive, and the fly turns toward it.
- **Hold 1–9** (or the side-panel buttons) to drive a cell group: LC10a L/R, DNa02 L/R, DNp53, LC4, DNp01, MDN (read only, not wired to the drone), P1. The *Drive* slider sets the mV.
- **Space** (or **Pause**) pauses; **Target ahead** does what F does.
- **The side panel:** click a section's title to collapse it, the pin to keep it at the top while the rest scrolls, and drag its grip (or focus the grip and use the arrow keys) to move it. The layout is remembered in the browser; **Reset layout** at the bottom puts it back.
- **On a phone:** tap a truck to make it the target (a tap near one counts), drag a finger to paint a phantom (a finger held still paints too), and pinch to zoom. Hold a stimulus button with a finger. The view traces fewer pixels when the phone falls behind (down to half, `LOOP.minRenderScale`) before it slows the brain, and the panel's brain and eye views stop drawing while they are scrolled away. A phone GPU is much slower than a desktop one, so expect the brain, and with it the whole game, to run slower than real time; the OSD's BRAIN figure shows by how much.

Every contact is fatal: a trunk, the ground or a truck sets off the warhead. A new drone launches from the start 3 s later with a fresh brain: its spiking state is cleared, as after a seizure, and it waits 1.5 s before it steers (`GAME.resetBrainOnRespawn`; `?brainreset=0` keeps the old brain running). A seizure (the lLN1_bc clique running away) reboots the spiking state and drops every held stimulus.

A destroyed truck stays destroyed. Its wreck smoulders on the road, and trucks stop behind it (in either lane), or behind a stopped truck in their own lane. **Respawn trucks** brings every destroyed truck back somewhere the fly can't see it and clears the wrecks away.

Grenades are in the code but switched off (`GAME.grenadesEnabled`; `?grenades=1` turns them on, with G to drop one).

## The target drive

Nothing in the connectome knows what a truck is, so the game tells the fly where the target is, through the fly's own eyes and its own small-object pathway (LC10 → DNa02):

- **Marker:** the target is drawn black to the fly's eye; other trucks are mid-grey. The eye has no colour, and dark is what this fly steers toward hardest.
- **Painting:** while the target is in view, the LC10a cells whose receptive field lies within 15° of it get 20 mV.
- **Lock:** with the target within 40° ahead and 30 units, the escape hop is off (otherwise every approach turns into a retreat), the yaw adaptation is held (otherwise it cancels a steady turn), and the drone speeds up to 2×.
- **Dive assist:** inside 15 units, the pitch is steered at the target. The fly's pitch pathway can't bring the drone down onto a truck. Left and right stay the fly's.
- Painting and the marker switch off inside 6 units. With them on, the brain seized about 1.5 times per hit, just before contact.

`?food=none|marker|paint|both|learn` picks the drive; `both` is the default. `learn` is fly-addiction's mushroom-body drive (reward dopamine while the target is in view, a learned-value readout that steers). It was the worst arm in the comparison, and it changes the brain (KC→KC silenced, ×20 visual gain into the Kenyon cells).

## Measured

Headless GPU runs on an RX 9070, 180 brain-seconds per run, every contact fatal. Seeds vary the hit rate 2–5×, so three seeds per arm is a rough answer.

**The chase** (a truck designated as the target, 3 seeds):

| world | target drive | hits/min | crashes/min | target within 30° ahead |
|---|---|---|---|---|
| **thick trunks, turn-away (Forest mode, the default; 6 seeds)** | both | **2.67** | 0.78 | 30% |
| the same, without the brain reset on a new drone (`?brainreset=0`, 6 seeds) | both | 2.17 | 0.78 | 29% |
| the same, before the turn-away was held off through the dive (3 seeds) | both | 2.11 | 0.89 | 29% |
| thin trunks, no turn-away | both | 3.56 | 0.44 | 42% |
| thin trunks, no turn-away | none, assists on* | 0.56 | 0.33 | 7% |

\* Measured with the earlier, smaller truck (4 × 2 × 1.5), when `both` scored 2.56 in the same world. Also from then: without the dive assist the fly got over the truck but almost never onto it (0.11 hits/min).

**Free flight** (nothing to chase, crashes per minute, seeds 1–3 / 4–6):

| | turn-away off | turn-away 6 |
|---|---|---|
| thin trunks (radius 0.3) | 0.67 / 0.22 | 0 to 1.44, not reproducible |
| **thick trunks (radius 1)** | 3.44 / 1.67 | **0 / 0** |

**Why the turn-away needs thick trunks:** the approach test flies the drone straight at one tree set off to the side, steering off. The looming cells (LC4 + LPLC2) take a thick trunk's side 3–5 s ahead. A thin trunk falls between the eye's 4.8° columns until the last second. DNa02 never turns away on its own: near the end it turns *toward* the trunk, as it does with anything it sees. The turn-away is the game reading the looming asymmetry, and it is off, as is the escape hop, while the target is locked or within 15 units in front, and for 1.5 s after (then it fades back in over 1 s): otherwise the truck looming beside the drone late in a dive turned it away.

**Smell can't steer this fly.** Driving the CO₂ or vinegar receptor neurons on one side at 8 or 12 mV seized the brain in every trial. The CO₂ projection neurons fire without a seizure but don't steer. A scent trail would be a hand-written autopilot, so there is none.

**Stimulation** (open and closed loop, 10 and 20 mV): LC10a and DNa02 turn the drone the expected way at both drives, and a painted phantom does at 20 mV; DNp53 climbs; DNp01 hops. The LC4 drive gives a hop in only some trials.

**Performance:** a 7-minute chase held 60 fps with the brain at 1.00× real time, and the JS heap stayed at 12–15 MB. Cost scales with window size, since the FPV view is rendered per pixel.

## The views

The fly's eye samples `sceneLum()` in [`scene.wgsl`](web/src/shaders/scene.wgsl) and nothing else: flat grey shapes. The sky is 0.5, the ground 12% darker, the trunks 20% darker than the sky, the trucks flat 5.6 × 2 × 2.4 boxes at 0.3, and the target black. The **ommatidia** view shows exactly that, one hexagon per 4.8°.

Everything else is drawn for the player by [`view.wgsl`](web/src/shaders/view.wgsl) and [`osd.ts`](web/src/game/osd.ts), and can't reach the brain:
- the burnt forest (charred trunks with splintered tops, fallen logs, ash, a dirt road, smoke, fog) and the olive 6x6 trucks, drawn inside the boxes the fly sees. The logs are solid in the view, but the fly can neither see nor hit them;
- the drone, a long-range FPV strike drone. It has an X frame, a taped battery, a GPS mast, and the camera pod with the fly's red eyes behind the glass. The warhead is slung underneath: a steel band round its middle, a threaded tail with the fuze plug and its red lead, and an arming box. The contact fuze is two steel rods from under the front arms, crossing in an X ahead of the nose, each ending in an open ring banked outward;
- the FPV camera: a 130° fisheye at the fly's head, graded like cheap FPV footage (blown sky, grain, block compression, a smudged lens, tearing after blasts). The fuze rods and rings and the warhead's nose show at the bottom of the picture;
- the OSD, explosions, wrecks, and the green glow on the target.

## Tests

`sh scripts/test-controls.sh` runs headless GPU protocols from [`bench.ts`](web/src/bench.ts) and grades them with [`check-controls.mjs`](web/scripts/check-controls.mjs). Results go to `.cache/tests`, and the run is resumable.

| suite | what it does |
|---|---|
| `open` | every stimulus, drone held still: the turn and pitch the commands would give |
| `closed` | the same while flying: the turn and climb it actually makes |
| `food` | the chase, once per target drive (`FOOD_MODES`); destroyed trucks come back by themselves here, as when the numbers above were measured |
| `survive` | free flight through the forest, counting crashes |
| `approach` | straight at one tree, steering off; read with [`analyze-approach.mjs`](web/scripts/analyze-approach.mjs) |

`SUITES`, `SEEDS`, `SECS` and `QUERY` (extra URL knobs, e.g. `QUERY='&forest=0'` for the thin-trunk world) select the runs. The checker has two tiers:
- **Plumbing** fails the run: every drive has to raise its own cells' rate, and in `learn` mode designating a target has to make the mushroom body learn.
- **Effectiveness** is reported only: a turn stimulus turns the expected way in 75% of trials by at least 10°, and a target drive beats the fly on its own.

One headless flight, with screenshots of every view and a close-up of the drone, goes to `.cache/smoke`:

```sh
sh scripts/build.sh && sh scripts/play.sh 40     # QUERY='?seed=2', PRESS="2:10-15" holds key 2 from 10 s to 15 s
```

It reuses `../fly-games`' `flygames-gputest` Docker image (Chromium with Mesa Vulkan).

Two more headless checks, each a script in [`web/scripts`](web/scripts) with its Docker command at the top: `check-ui.mjs` (trucks stop behind a wreck and stay destroyed, Respawn trucks, the panel's collapse, pin and reset) and `check-mobile.mjs` (a phone in Chromium's emulation: no sideways scroll, tap to target, finger painting, pinch zoom, held buttons, landscape).

## Running

Everything runs in Docker; nothing is installed on the host. `scripts/`: `install.sh`, `typecheck.sh`, `dev.sh`, `build.sh` (to `web/dist`), `play.sh`, `test-controls.sh`.

**Deploying:** every push to `main` builds `web/dist` and publishes it on GitHub Pages ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)). Once, in the repository settings: Pages > Source "GitHub Actions", custom domain `flybrain-fpv.domi.zip`, Enforce HTTPS (WebGPU only runs on a secure page); and in DNS, a CNAME record for `flybrain-fpv.domi.zip` pointing at `slickdomi.github.io`. The workflow's actions are pinned to commit SHAs of releases at least two weeks old, like the npm cooldown.

Every constant is in [`config.ts`](web/src/config.ts), with the measurement behind it. Useful URL knobs: `?seed=`, `?forest=0`, `?food=`, `?cars=N`, `?balloons=N` (the grey balloon targets of the first flights; 0 by default), `?view=0|1|2|3`, `?avoid=` (turn-away gain), `?trunkr=`, `?mark=off`, `?escape=0`, `?grenades=1`.

## Licence

Code: MIT ([LICENSE](LICENSE)). The connectome files in `web/public/data/malecns-v1` come from fly-addiction's 2026-09-23 build of MaleCNS v1.0 and are CC BY 4.0; see the `LICENSE.md` there.
