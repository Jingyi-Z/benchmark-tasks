# Focus an upright microscope on a slide without touching the objective

An upright light microscope with a slide clamped on its stage, 12 mm below focus. Turning either focus knob raises the
stage. The slide is in focus when its top is at the 10x objective's working distance, 5.2 mm below the front lens;
bringing it within 0.5 mm of the lens at any point counts as a crash.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The stage follows the knob through an `on_step`
  hook that sets the stage joint's drive target from the knob angle (24 mm per turn). The scripted demo grips the
  right knob from the side with the gripper level and rolls the wrist; `task.py` lets a demo waypoint carry a full
  gripper rotation for this (the `Orient` class), so yam_arena itself is unchanged.
- `labkit.py`: parts built from several shapes, with hinges and slides (the knob shaft is a damped hinge; the stage a
  stiff slide). It hooks `yam_arena.scene` and the embodiment from the task's import.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 24 s of motion).

The score is `task_score` in the log, computed by `score()` in `task.py`: the final gap from the slide's top to the
objective's front lens, and the smallest gap during the trial (recorded every control step).

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
