# Load a wafer centred on a spin coater's chuck while holding the sprung lid open

A benchtop spin coater with a sprung lid on a lever, and a 2-inch silicon wafer on a tray beside it. One arm pushes
the lever's post aside and holds the lid swung back off the bowl; the other picks the wafer up by its edge and lays it
flat on the 3 cm vacuum chuck, centred within 1 mm. Then both hands come out and the spring closes the lid.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The lid and its lever are one labkit hinge about the vertical through the pivot pin, with a light spring back to
    closed and a stop at 97 degrees open.
  - The bowl wall is a ring of 48 upright ribs and the centring marks are 24 flat dots, since labkit's boxes are not
    turned.
  - The scripted demo holds the lid open with the right hand's fingers part-open round the post (16 mm apart, closing
    along the post's path), driving it round its arc in steps of about 11 degrees to 80 degrees open and holding it
    there until the left hand is up out of the bowl.
  - The scripted demo centres the wafer closed loop before setting it down: an `on_step` hook reads the wafer's position
    and moves the left hand's commanded position by part of the remaining offset from the chuck's axis each control
    step (the `_Live` waypoints), so yam_arena itself is unchanged.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1800 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 30 s of motion). Run one trial per process: the
demo's centring state is kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state.

## Runtime and seeds

- 1,800 control steps (90 s of simulated time) per trial; about 5 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
