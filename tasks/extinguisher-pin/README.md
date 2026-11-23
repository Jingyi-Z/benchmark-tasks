# Get a home fire extinguisher ready to use: break the tamper seal and pull the pin

A full-size rechargeable home extinguisher (First Alert HOME1 class) stands on the bench. Its ring pin passes through the
valve head and the operating lever; a plastic tamper seal ties the pin to the carry handle. The robot gets it ready to use,
step 1 of pull-aim-squeeze-sweep: break the seal and pull the pin out, without ever squeezing the lever, and leave the
extinguisher standing. The task stops at "ready to use": no discharge, no fire, no standoff.

## Run a trial

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 800
```

To reproduce the recorded scripted controller, prefix the command with `EX_FSLOT=0.09 EX_DEPTH=0.004` and add
`--policy yam_scripted` (the bare scripted command uses a slightly narrower jaw opening, `EX_FSLOT=0.06`; scene and scorer
are the same). For Dream-1 use `--policy dream` with `-E cam_width=640 -E cam_height=352`.

- Scripted demo: `--policy yam_scripted` runs `demo()` in `task.py`. Both arms start parked. The left hand grips the carry
  handle (below the lever, never on it). The right hand pre-closes, threads one finger through the ring, swings the ring up
  about the pin's eye, turns it to break the seal, then pulls the pin straight out along its axis and lets it fall to the
  bench.
- Seeds: none; the layout is fixed (`object_jitter=0`).
- Hashes of the recorded files (sha256 of the files as run, CRLF line endings): task.py 90c719bb9a3fc687...,
  labkit.py 93dad23eb7df3262..., arena_assets_fix.py 4804d30037ebf639.... A checkout with LF endings has different byte
  hashes for the same code.
- Run limit: 800 control steps (40 s at 20 Hz). The demo settles at 1.0 by step 580.
- Scorer tests: `python test_scorer.py` (20 cases; simulator stubbed, production scoring code unchanged).

## Scoring (checked every control step)

- Latched for the rest of the trial (score 0): the lever moves more than 1 deg from rest; the extinguisher tips more than
  20 deg; the pin falls below the bench top.
- Seal broken: the PhysX joint-break event of the seal tie. This gives the time of the break, not its force.
- 1.0, all of:
  - the seal is broken;
  - the pin is fully out of the valve and resting on the bench (pin centre within 20 mm of the bench top);
  - at every step the extinguisher's base slide stays strictly under 10 mm and its axis lean strictly under 5 deg;
  - both pad tips stay at least 80 mm from the surfaces of the cylinder, valve head, pin and ring at every sampled
    step of the last second;
  - for 1 s the extinguisher and the pin each move under 2 mm.
- 0.5: the same, except the pin is currently withdrawn more than 5 mm but not yet out of the valve.
- 0 otherwise.

Checking a real trial with simple tools: film the unit from the side at bench height with a mm scale fixed behind the
lever tip and the base outline marked on the bench. The lever tip is 106 mm from its pivot, so 1 deg is 1.85 mm of tip
travel. A 10 mm slide shows against the base outline; a 5 deg lean is 25 mm of sideways shift at the 283 mm shoulder.
Pin out and on the bench, and hands clear by 80 mm, are read from the same video.

## Evidence: demo and controls (same start, same motion)

| Run | Change | Seal break (PhysX event) | Pin | Score |
|---|---|---|---|---|
| DEMO | none | during the turn (step 371) | out, on the bench | 1.0 |
| Never-break seal | tie cannot break | none | stays in | 0 |
| Squeezed lever | lever held at 0.15 rad by a diagnostic spring preload | during the turn | binds, stays in (0.2 mm) | 0 (lever latch) |
| No holding hand | left hand parked | during the turn | out | 1.0 |

In simulation a 5.4 lb unit on the bench also stays put unheld (last row). The instruction therefore asks only that the
extinguisher keeps standing where it is. Holding the carry handle is the maker's label
practice ("Pull pin. Hold unit upright.") and keeps the free hand off the lever. It is not a physical necessity here, and
the scorer does not require it.

## Forces (sources)

| Quantity | Value | Source | Model |
|---|---|---|---|
| Tamper seal break | 65 N (15 lbf) or less | UL 299 (2021), tamper indicator clause | seal tie breakForce 65 N (the upper limit) |
| Commercial seals | 15 lbf maximum | Brooks Equipment UL-listed tamper seals | at the model's limit |
| Seal under a twist | 65 N x 25.6 mm = 1.66 N m | assumed moment arm: raised ring, eye to far wire | breakTorque 1.66 N m (an approximation) |
| Pin removal | 130 N (30 lbf) maximum | UL 299 (2021) 25.1; NFPA 10 commentary | contact friction only, after the seal |
| Lever on a locked pin | no shear at 220 N (50 lbf) | UL 299 (2021) 25.2 | the lever's tangs bear on the pin by contact |

- The seal tie is a PhysX D6 joint between the pin and the valve. It allows free inward travel up to the pin's shoulder
  (a tie through the ring goes slack when pushed) and locks outward travel and twist about the pin axis.
- The shoulder, the one-way tie and the torque limit are modelling approximations, not measured HOME1 values.
- Arm: the YAM wrist joints use DAMIAO DM-J4310-2EC motors, rated 3 N m and 7 N m peak (DAMIAO 2025 catalog). The
  installed yam_arena limits are 7 N m on joints 4 to 6 and 50 N finger slide effort. No pull force was measured.

## Reproducing on real hardware

Safety: use a HOME1 that a fire-equipment service shop has emptied and depressurised (the HOME1 is rechargeable, with a
metal valve), so a squeezed lever cannot discharge powder. The valve head, lever, pin, seal and handles are unchanged. The
empty unit is lighter than the charged 2.45 kg modelled here (5.4 lb listed; about 2.5 lb of it agent). It is never lifted, only
steadied on the bench. The emptied unit is roughly half the modelled mass (about 2.9 lb against 5.4 lb, estimated
from the 2.5 lb agent charge). Whether the twist then moves an unheld real unit was not tested here; holding the carry
handle, as the label says, is the safe way to run the real trial.

| Item | Stock part | Price (USD) |
|---|---|---|
| Fire extinguisher | First Alert HOME1, 1-A:10-B:C, 3.25 in x 14.3 in, 5.4 lb | 27.99 (maker's store); 26.99 (Target) |
| Empty and depressurise | fire-equipment service shop | not priced |
| Tamper seals | UL-listed pull-tight plastic extinguisher seals, 9 in, pack of 100 (one per trial) | about 10, not verified |

Reset: push the pin back through the valve head and lever from the ring side, and fit a new tamper seal from the ring
round the carry handle. The seal is consumed every trial.

## What is modelled

- Rigid parts at HOME1 size and listed mass (2.45 kg, retailer listing, not a maker sheet).
- The visual body follows the maker's product photos: straight wall to about 60% of the height, long tapered shoulder,
  wraparound label (generic, no brand), chrome valve with round gauge and nozzle, black pressed-steel lever over the
  carry handle, round ring pin. Simple colliders do the physics.
- The lever is a sprung hinge. Its two side tangs have their own holes round the pin, 0.6 mm clear at rest; a squeeze
  brings the hole tops down onto the pin, which then binds in the valve-cheek holes.
- The ring hangs on a free hinge through the pin's eye.
- Estimated: valve and lever shapes, pin 3.2 x 50 mm, ring 24 mm (inner opening 21.6 mm).
- Not modelled: discharge, pressure, gauge reading, the bracket strap.

## Files

- `task.py`: scene, lever, pin, ring and seal, scorer, scripted demo.
- `test_scorer.py`: scorer tests.
- `labkit.py`: shared helper, plus `TIES` (a breakable joint between two parts) and smooth-normal drawn meshes.
- `arena_assets_fix.py`: shared helper.
