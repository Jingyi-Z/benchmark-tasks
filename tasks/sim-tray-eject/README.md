# Pop a phone's SIM tray out with the eject pin and put the pin back in its holder

A phone the size of an iPhone 12 stands upright in a loose desk dock, screen towards the robot. Its nano-SIM tray sits
flush in the phone's left edge. An eject tool stands pin-down in a foam block beside the dock. The tray must be popped
out and the tool stood back in its block, done the way people do it: the 0.8 mm pin goes straight into the 1 mm hole
in the tray's cap while the other hand steadies the phone, the tray pops a few millimetres out, and the tool goes back
where it is kept.

The tray is latched inside by a sprung pawl in a notch on its top edge. Only the pin, pushed in through the hole, can
lift it (a 30 degree slope on the pawl's underside lies across the hole's line). A sprung ejector then pushes the
tray 3 mm out. Pushing the tray, prying at its flush cap or tipping the phone does not free it.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
- `labkit.py`: parts built from several shapes, with sprung joints (the pawl and the ejector inside the phone). It
  hooks `yam_arena.scene` and the embodiment from the task's import, so no change to yam_arena is needed. `task.py`
  sets each spring's target every step, since labkit leaves Isaac Lab's joint targets at 0.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1000 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state. Release (every
fingertip 80 mm from the phone, tray, tool and dock) and stillness (1 s, 21 control samples) are tracked every
control step in `_watch()`.

## Runtime and seeds

- 1,000 control steps (50 s of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
