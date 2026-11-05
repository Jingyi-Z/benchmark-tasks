# Mount a gripper module on a manual tool changer and lock it

The master plate of a manual robot tool changer on a fixed stand, and a spare gripper module on a tray beside it.
One arm slides the module's T-tongue into the plate's slot and holds it against the end stop, where a spring plunger
pushes back; the other swings the red locking lever a quarter turn so that its tooth goes into a notch in the
module's side. Let go unlocked, the plunger pushes the module 8 to 10 mm back out; locked, the tooth holds it home.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The changer is one fixed labkit part with two joints: the plunger (a slide along the slot, 100 N/m, 12 mm of
    travel) and the lever (a hinge about the vertical through its pivot pin, 0 to 90 degrees, light friction, no
    spring). The lever is built where it starts, unlocked, so the first rendered frames show it there.
  - The lock is plain contact: with the module home, the tooth sweeps into the notch; with the module more than
    about 2 mm out, the tooth meets the module's side and the lever stops 3.4 to 3.9 degrees short of locked. Pushed
    by the plunger, the notch's back wall presses the tooth towards the lever's stop, so the lock holds by itself.
  - The scripted demo is closed loop where it touches the module: an `on_step` hook reads the module's position and
    moves the left hand's commanded position (the `_LeftAt` waypoints), first across the slot to line the tongue up
    before setting it down, then along the slot until the module's back is within 0.3 mm of the end stop (the arm is
    soft sideways, and the plunger and friction take about 1.6 N). The right hand drives the lever's knob round its arc
    in 10 degree steps with its fingers part-open round it, as in the spin coater task, leaning 30 degrees about
    their closing line so that its elbow stays clear of the left arm's.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 34 s of motion). Run one trial per process: the
demo's live corrections are kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: whether the module's
tongue is in the slot, how far its back is from the end stop, and the lever's angle.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
