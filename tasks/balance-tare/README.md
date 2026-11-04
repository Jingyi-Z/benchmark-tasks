# Tare a balance with the empty weighing boat, then weigh a sample

A precision balance with an empty weighing boat on its pan, and an aluminium sample puck on the table. One arm presses
the balance's TARE key while the boat is still empty; the other then sets the sample upright in the middle of the boat.
Done in that order, the display reads the sample's 30 g alone.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
- `labkit.py`: parts built from several shapes, with hinges and slides (the TARE key is a sprung slide with 3 mm of
  travel). It hooks `yam_arena.scene` and the embodiment from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 21 s of motion).

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state. The balance's reading is
worked out in `task.py`: an `on_step` hook records what rests on the pan each time the TARE key goes down past half its
travel, and the reading is the pan load now minus the load at the last press.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
