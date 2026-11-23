# Steer a laser spot onto a target with a rotating wedge-prism pair

Two manual rotation mounts, each holding a 2 degree glass wedge prism, sit in the beam of a Class 2 laser pointer on an
optical breadboard. A frosted acrylic screen 350 mm beyond the second prism carries a printed target. Each prism
deviates the beam by 2 degrees toward its thick edge; the two deviations add as vectors, so turning one ring swings the
spot around a circle. From the fixed start, either prism alone cannot bring the spot closer than 5.2 mm to the target
centre; both rings must be set, then released with the hands clear.

## Run a trial

From this folder, with the yam_arena / inspect-robots environment active. Scripted demo (the scripted policy ignores the
instruction text):

    inspect-robots run --embodiment yam_arena --policy yam_scripted       -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352       --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')"       --scorer task_score --max-steps 2400 --store-frames --log-dir ./logs

Model run (Dream-1, same task file, instruction and step cap):

    inspect-robots run --embodiment yam_arena --policy dream       -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352       --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')"       --scorer task_score --max-steps 2400 --store-frames --log-dir ./logs_dream

- Recorded with the cameras at 640 x 352; the policy images are 224 x 224.
- Runtime: the scripted demo reaches 1.0 in about 40 s of simulated time (about 800 steps); the run limit is 2,400 steps.
- Seeds: none. Fixed start layout (both ring angles fixed at reset), object_jitter 0.
- Scorer unit tests (pure Python, no simulator): `python test_scorer.py`. They include each prism alone at its best
  angle (scores 0), the start pose, the 0.5 band, a moved laser or screen, a hand near a mount or in the beam, a
  fingertip at the glass, missing hand data, rings still turning, and too few samples for the 1 s stillness check.

## The scripted demo

The left arm parks its wrist camera facing the screen. The right arm then works in strokes, choosing each one from
what the left wrist camera sees:

1. Find the dot in the left wrist camera image (the densest red spot on the card at the end of the beam). Cast its
   pixel ray onto the screen plane using the camera's pose, read from the arm at each look, and its lens intrinsics
   (a real trial gets the same from the arm's joint encoders and a hand-eye calibration), plus the known card plane,
   and get the dot position in mm.
2. Estimate the two ring angles that put the dot there (inverting the two-wedge relation near the angles expected
   from the start marks and the strokes so far), and pick the ring and turn that bring the dot to the target.
3. Stroke: pre-closed jaws come down across the knurled dial ring at a point the wrist can reach, pinch it, rotate
   the wrist about the optical axis, release, rise clear. The stroke gain is learned from the seen result.
   In simulation the grip ring is a plain 6 mm rim with no housing (see "Simulation versus the real mount" below).
4. Repeat until the seen dot is within 1.0 mm of the target centre, then move the hand clear. If the dot is not seen,
   no stroke is made.

The demo never reads simulator joint angles to choose a stroke; the scorer and the drawn beam use the true angles.

## Score

`task_score` comes from `prism_scorer.py` (class `Tracker`), fed one observation per step by `task.py`. The spot
position is computed from the two real ring angles by thin-prism deviation, applied in sequence at the real 50 mm
prism spacing (`spot_yz`); the same calculation draws the spot and the beam the cameras see. The numeric rule is in
`task.json` "scoring": 1.0 inside 2.0 mm of the target centre, 0.5 inside 3.0 mm, hands 80 mm clear of both mounts
and out of the beam, both rings still for 1 s; a fingertip at the glass or any fixed component moved latches 0.

## Files

- `task.py`: breadboard, laser, two rotation mounts with wedge prisms, screen; drawn beam and spot; scripted demo; per-step scoring.
- `prism_scorer.py`: spot model and scorer.
- `test_scorer.py`: scorer tests.
- `labkit.py`, `arena_assets_fix.py`: shared helpers.

## Simulation versus the real mount

The real LBTEK RME-1A ring has a 44 mm knurled flange only 3.2 mm thick, with the RME-1AS housing seated right behind
it. The simulation simplifies each mount to a plain 6 mm rim with no housing, so the demo's pinch across both faces of
the rim is easier than on the real part, where only the front face and the knurled edge can be reached. Ring holding is
a joint friction setting (0.02, with damping 0.01) in the simulator; the real holding torque is not published and should
be measured. Ring mass 30 g and pad friction 0.9 are estimates.

## Reproducing on real hardware

Positions are on the breadboard grid in `task.json` "setup", except the screen post at x 0.400 m (between grid holes,
on a 10 mm offset bracket). Prices are for materials, not an assembled unit.

| Item | Part | Price |
|---|---|---|
| Light source | Vernier LASER pointer, Class 2, under 1 mW, 650 nm | $29.00 (vernier.com/product/laser-pointer) |
| Laser holder | printed V-block (about 20 g PLA, 40 x 30 x 25 mm, 90 deg vee) on the laser's post; a switch clip holds the button on | about $0.40 of PLA ($19.99 per kg, matterhackers.com) |
| 2 x rotation mount | LBTEK RME-1A inner ring ($22.23) + RME-1AS housing ($26.15) each | $96.76 (en.lbtek.com/product/475) |
| 2 x retaining ring | LBTEK SM1 retaining ring, if not included | $8.50 (en.lbtek.com/product/135) |
| 2 x wedge prism | Thorlabs PS810, 1 in N-BK7, 2 deg beam deviation (3 deg 53 min apex), uncoated | $42.90 each, $85.80 (govolition.com V40-PS810) |
| Breadboard | 6061 plate 1/4 x 6 x 24 in cut to 600 x 150 mm, M6 drilled and tapped on 25 mm from one origin, on two end supports from one 12 in piece of 1/4 x 1/2 in bar | $54.84 + $4.50 (onlinemetals.com 1149, 1141) |
| Posts and holders | 4 x LBTEK PH-25A holder ($7.81) + 3 x OP-50 post ($2.88) + OP-75 post ($3.15) | $43.03 (en.lbtek.com/product/126, /131) |
| Screen | OPTIX 8 x 10 in clear acrylic, 2.03 mm, cut to 100 x 75 mm, one face frosted, target printed | $4.98 (lowes.com) |
| Total | | about $319 in materials with the retaining rings included in the mounts (about $328 if bought separately; about $339 buying the full PLA spool), before screws, tax and shipping; shop tools (saw, drill, M6 tap) assumed |

A ready-made LBTEK MBB-1560 board ($127.32, en.lbtek.com/product/342) can replace the drilled plate: about $387.

Mark the two ring start angles on the mount scales. Scoring tools: a ruler on a photo of the screen (spot centre to
target centre). Reset: turn both rings back to their start marks. Consumed per trial: nothing. Safety: the Class 2 laser
pointer is fixed in its holder, aimed along the board into the screen and away from the operator, and never toward a
camera; do not look into the beam.
