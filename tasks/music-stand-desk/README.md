# Unfold the desk of a folding music stand

A K&M 10065 type folding music stand stands on the floor with its height and tilt set, but its desk is still fully
folded for carrying: both halves are collapsed against the centre spine and the diagonal braces under each half of the desk are bent at their
knees. An open ukulele case with a soprano ukulele in it lies on a side table between the robot and the stand, its rim in
the path a hand would take to the stand tube at the usual height. The robot swings out both ledges until the braces are straight, so the desk
is wide and supported, while keeping the light stand from tipping or sliding and leaving the case untouched, then lets
go.

## Run a trial

From this folder, with the yam_arena / inspect-robots environment active:

    inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
      --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
      --scorer task_score --max-steps 2400 --store-frames --log-dir ./logs

Model run (Dream-1): the same command with `--policy dream` and a separate log folder.

- Settings used for the recorded demo: `MS_DFRIC=0.01` (brace pin friction); everything else default.
- Runtime: the scripted demo deploys both halves in about 35 s of simulated time (it scores 1.0 by step 700); the run
  limit is 2,400 steps.
- Seeds: none. Fixed start pose, object_jitter 0.
- Controls (environment variables): `MS_NOMOTION=1` (hands stay parked: score 0), `MS_NOSTEADY=1` (same right-hand
  motion with the left hand parked), `MS_NOPADS=1` (hand positions unavailable: no credit, the scorer fails closed),
  `MS_BOOK=open` (an open 9 x 12 in score laid on the desk, for the support check in the table below),
  `MS_TUBEZ=0.68` (the stand tube held at the lower height used without the case: the case is pushed 62 mm by step 420, latched 0),
  `MS_UKE=0` (no table and case).

## The scripted demo

The left hand holds the stand tube at 0.745 m, above the case. The right hand pinches the lip of the right ledge and swings it out past its stop,
which straightens that side's diagonal brace; the hands swap roles for the left ledge. Then both hands move clear. The
motion is a fixed plan: no decision in it depends on simulator state.

The wrist camera of the hand holding the tube is nearly black while it holds: it faces the black folded desk from a few
centimetres away. Rolling the wrist 90 deg to turn the camera away lost the grip on the tube (score 0), so the demo
keeps the roll that holds.

The case lid is off. With the lid on its hinge, every opening tried (35, 100, 145 and 180 deg) put it in the
reach of the forearms or elbows, and the case was pushed (20 to 73 mm).

## Score

`score()` in `task.py`, from the simulated joint angles, the stand pose and the fingertip positions; the numeric rule is
`task.json` "scoring". Hands clear is measured from the sampled fingertip points to sampled points of the nominal open
desk outline (ledge and top corners and midpoints) and of the stand tube, not to every surface, and must hold on every
one of the last 21 steps. The ukulele case is checked on every step from step 40 on (its pose after settling).
Stillness uses the stand position and the two ledge angles over the same 21 steps. If the
fingertips cannot be located, the clearance counts as zero (no credit).

## Files

- `task.py`: stand, desk linkage, scorer, scripted demo.
- `labkit.py`, `arena_assets_fix.py`, `cart_scene_fix.py`: shared helpers (rigid parts, joints, floor scene).

## Reproducing on real hardware

| Item | Part | Price |
|---|---|---|
| Folding music stand | K&M 10065, black, steel, 1.55 kg (3.42 lb), desk 485 x 240 x 42 mm, height 585 to 1220 mm | $72.99 (Full Compass, fullcompass.com) |
| Side table | any table with a wood top about 36 x 70 cm at 0.625 m, on steel legs | about $40 |
| Ukulele and case | soprano ukulele (wood, 53 cm) in the open base of a shaped soprano hardshell case, 62 cm long, 24 cm wide, lid off | about $50 ukulele, about $40 case |
| Support check (optional) | any 9 x 12 in music book laid open on the desk: on the deployed desk its free half stays flat (0.0 deg in simulation); on the folded desk it swings back 20 deg | about $10 |

Set the stand at 0.80 m ledge height with the desk tilted back 20 deg, both ledges folded in. Scoring by eye and a
protractor: each ledge at its stop and each brace straight. Put the side table 4 cm short of the stand's centre and the case at its stand-side edge. A tape mark on the table
shows a 5 mm move of the case. Reset: fold both ledges back in. Nothing is consumed.
