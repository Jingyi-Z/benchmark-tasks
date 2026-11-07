# Move a wafer onto a hotplate with a transfer paddle, never touching its top

A 4-inch silicon wafer lies on a narrow pedestal, overhanging it all round. A PTFE transfer paddle lies on the table, and
an aluminium hotplate has a ring of black dots on its top plate marking where the wafer goes. One hand picks the paddle
up by the red tab on its handle and holds it level beside the pedestal, its open front edge under the wafer's overhang
and its top just below the wafer. The other hand, fingers closed, pushes the wafer by its rim off the pedestal onto the
paddle and backs off. The first hand tips the paddle back so the wafer slides against the stop, lifts it, carries it
over the hotplate, sets the paddle's front edge on the top plate and tilts it forward until the wafer creeps off the
edge onto the plate, then draws the paddle back out from under it and puts it down. The wafer's top and the hotplate's
top are never touched, and the wafer is never held by a hand.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The pedestal and the hotplate are fixed; the wafer
  and the paddle are loose. An `on_step` hook latches the violations (a fingertip over the wafer's top or on the
  hotplate's top, the wafer dropped, or the wafer off the pedestal, the paddle and the hotplate at once) and whether the
  wafer was carried on the paddle away from the pedestal, and checks that both hands are clear of the wafer.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 47 s of motion; the trial ends 10 s after full
score). The demo reads only what the cameras see: the paddle's pose, to trim the hand's aim while it holds the paddle.
Run one trial per process: the trial's record is kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 1,200 control steps (60 seconds of simulated time) per trial; about 6 minutes of wall time on one L40S with frames
  stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
