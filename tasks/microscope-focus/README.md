# Load a slide, centre the specimen and focus an upright microscope

An upright light microscope with a mechanical stage: a spring slide holder, x and y drive knobs at the stage's front
right and focus knobs on both sides. A glass slide lies on a rest beside it. The robots load the slide (one hand holds
the holder's lever open while the other lays the slide in), bring the specimen square under the 10x objective with
the drive knobs, then raise the stage with a focus knob to the objective's working distance (slide top 5.2 mm below
the front lens) without ever bringing the slide within 0.5 mm of the lens.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The holder's lever, the x and y stage axes, the drive knobs and the focus knob are labkit joints. The y axis
    rides on the stage, the x axis on the y axis and the holder on the x axis (chained joints).
  - An `on_step` hook sets the stage axes' drive targets from the knob angles: x 170 mm and y 40 mm per drive-knob
    turn, focus 24 mm per turn.
  - The scripted demo turns the drive knobs and trims the focus with a closed loop on the knob angle (the `Servo`
    class); `task.py` lets a demo waypoint carry a full gripper rotation for the side grips (the `Orient` class), so
    yam_arena itself is unchanged.
- `labkit.py`: parts built from several shapes, with hinges and slides, including joints chained on other joints
  (`parent=`). It hooks `yam_arena.scene` and the embodiment from the task's import.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 75 s of motion; 1,600 steps are enough).

The score is `task_score` in the log, computed by `score()` in `task.py` from the slide's pose in the holder, the
specimen's offset from the objective's axis, the final gap from the slide's top to the front lens, and the smallest
gap during the trial (recorded every control step).

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial; about RUNTIME minutes of wall time on one L40S with
  frames stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
