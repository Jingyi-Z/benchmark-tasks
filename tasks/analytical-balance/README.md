# Weigh out 1.00 g of granules on an analytical balance with a draft shield

An analytical balance with a glass draft shield whose top door slides back. One arm runs the door and the TARE key;
the other puts the empty weighing boat on the pan and, once the balance is tared with the door shut, pours granules
from a vial into the boat until the live display reads 1.00 g. Then the door is shut again.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The balance's reading is worked out every control
  step from what rests on the pan; the task records the last TARE press (what was on the pan, whether the door was
  shut) and drives the display's segments to show the reading.
- `labkit.py`: parts built from several shapes, with hinges and slides (the top door, the sprung TARE key and the 28
  display segments are slides). It hooks `yam_arena.scene` and the embodiment from the task's import, so no change to
  yam_arena is needed. This copy adds two things: a box may carry a yaw (the vial's 16 wall faces), and colours listed
  in `labkit.GLASS_COLORS` are drawn with Omniverse's glass material (falling back to a translucent, then a plain
  surface).
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 84 s of motion). The demo's pour is closed loop:
the vial tips slowly (2.5 degrees a second from 30 degrees) until the display reads 0.95 g, then levels. Run one trial
per process: the tare record and the pour state are kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state.

## Runtime and seeds

- 2,400 control steps (120 s of simulated time) per trial; wall time not yet measured (73 bodies, 70 of them granules).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
