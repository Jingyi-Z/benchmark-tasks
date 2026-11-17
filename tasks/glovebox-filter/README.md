# Replace the passenger-side cabin air filter element behind a 2002-2006 Honda CR-V's glove box

A mock-up of the 2002-2006 Honda CR-V's passenger dash stands on the table: its glove box (77500-S9A-A01ZA) is
unlatched and resting open on its two stoppers (77547-S9A-G01), and the cabin air filter's passenger-side element
(Honda 80292-S5D-416, FRAM CF10135: 112 mm wide, 226 mm long, 29 mm thick, in its case with a pull tab) sits in its tray behind it,
the access cover already removed. Each stopper is a separate clip on the outside of a side wall whose catch hooks on
the dash. As on the car, a stopper is released by squeezing its tab and pushing it into the box; it stays in, and with
both in the box swings down. Then the old element slides out of its bay by its pull tab, out through the box's opening,
and the new one, waiting on a stand beside the dash, goes into the same bay through the same opening. Pulling the box
down past a stopper that is not pushed in would break it, and is scored 0. The driver-side element stays in its tray, side by side with it, and is not part of the task.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
- `labkit.py`: parts built from several shapes, with hinged and sliding joints (the box's hinge and its two stoppers).
  It hooks `yam_arena.scene` and the embodiment from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.
- `check_path.py`: offline check of the scripted demo, no simulator (needs `yam_arena` and `scipy`): replays both arms'
  commanded joints and reports joints near their limits, tracking error, arm collisions with the dash, the stand,
  the glove box and the resting elements, arm-to-arm contacts, and contacts of a carried element (it follows the hand
  from each closing of the fingers, so it stays right through re-grips) with everything but the holding pads. It
  replays the commanded path with the objects where the plan puts them (not the simulator's recorded poses), so it
  is a check of the plan, not of a run.
- `check_replay.py`: the same arm hulls placed from a run's actual joint angles (logged every control step by
  `task.py` when `GLOVE_DEBUG=1` and `GLOVE_JOINTS=<file>`), checked arm against arm and against the dash and stand,
  with the arms' closest approach. On the submitted demo: no contact; minimum sampled hull-vertex distance between the arms 65 mm (not surface clearance).

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state. A forced stopper (the
box opening through its stops while either stopper is pushed in less than 4.5 mm) is latched every control step in
`_watch()` and scores 0 for the rest of the trial. `_hold_stoppers()` keeps a stopper in once it has been pushed in
7 mm. `_watch()` also latches the box having passed its stops (open more than 54 degrees); drawing the old element from
its bay, or putting the new one into it, before that scores 0 (picking the new one up first is fine), and full credit
needs the box to have passed its stops. The dash's cheeks run up to its top panel, so the element is reached only
through the glove box's opening. Every positive score also needs every fingertip (each jaw on its own) at least 8 cm
from the nearest surface of both elements (tabs included) and the box, and every corner of both elements and the box
still within 1 mm over the last 21 control steps (`_clearance()`, `_watch()`).

## Runtime and seeds

- 2,400 control steps (120 s of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.

## Closest board task

TK-FKWT9S (not accepted) swaps a clogged filter cartridge behind a return grille. Here the filter is hidden behind a
glove box that only drops after both side stoppers are squeezed in (dropping it past a stopper breaks it), and the new
226 mm element must go in one-handed through a 118 x 35 mm slot with 3 mm of play a side, square to within 3 degrees.

## Known gaps

- Sources: element size and part numbers from Honda and FRAM listings; the stopper release (squeeze, push into the box,
  then the box drops) from two CR-V repair guides. The glove box, stopper, stop and bay sizes and positions are
  estimates, not measured, and the element's 60 g mass is an estimate.
- The stopper latches by its spring aiming fully in once pushed 7 mm in, rather than being removed; the access cover
  starts removed; the driver-side element stays in its tray. The demo reads only camera-visible poses: while it lines the new element
  up and pushes it in, small trims (a fifth of the seen error per step, at most 0.3 mm or 0.17 deg) move the hand so
  the element's far end runs on the tray's line and its turn stays square; it then steps its pinch 60 mm back along the
  edge twice without letting go, so the element never tips, and pushes it home.
