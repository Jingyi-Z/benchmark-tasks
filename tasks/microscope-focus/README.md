# Load a slide, centre a chip's alignment mark and focus an upright microscope

An upright light microscope with a mechanical stage: a spring slide holder, x and y drive knobs at the stage's front
left and coarse and fine focus knobs on both sides. A slide with a purple chip carrying an alignment mark lies on a rest beside it, and a
camera monitor to the left shows the microscope's live 10x view. The robots load the slide (one hand holds the holder's lever open while the other lays the
slide in), bring the alignment mark into the 10x objective's field of view with the drive knobs, raise the stage with a
coarse knob to near the working distance (slide top 5.2 mm below the front lens) without ever bringing the slide within
0.5 mm of the lens, then finish with a fine knob until the mark is sharp on the monitor (within the 10 um depth of field).

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The holder's lever, the x and y stage axes, the drive knobs and the coarse and fine focus knobs are labkit joints. The y axis
    rides on the stage, the x axis on the y axis and the holder on the x axis (chained joints).
  - An `on_step` hook sets the stage axes' drive targets from the knob angles: x 170 mm and y 40 mm per drive-knob
    turn, coarse focus 24 mm and fine focus 1 mm per turn.
  - The scripted demo turns the drive knobs, then the coarse and fine focus knobs, with closed loops on the knob
    angles read off their index marks (the `Servo` class); `task.py` lets a demo waypoint carry a full gripper rotation for the side grips (the `Orient` class), so
    yam_arena itself is unchanged.
  - The camera monitor draws the 10x view from the stage's position: its field purple over the chip and pale over
    glass, and the alignment mark moved with the stage and drawn sharp (within 10 um of focus), soft (0.1 mm) or
    blurred (1 mm), by showing one of three drawn copies of the mark (USD visibility and translation each step).
  - The illuminator's light is a thin pale yellow disc on the objective's axis, part of the focus slide (it rises
    with the stage) and drawn only, with no collider (labkit's `VISUAL_ONLY` colours).
- `labkit.py`: parts built from several shapes, with hinges and slides, including joints chained on other joints
  (`parent=`), and shapes drawn without a collider (`VISUAL_ONLY`). It hooks `yam_arena.scene` and the embodiment from the task's import.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 87 s of motion; 2,000 steps are enough).

The score is `task_score` in the log, computed by `score()` in `task.py` from the slide's pose in the holder, the
alignment mark's offset from the objective's axis, the stage height's distance from focus, and the smallest gap from the
slide to the front lens during the trial (recorded every control step).

## Scoring a real trial

- Centring: the alignment mark is in the camera image on the monitor at 10x (2.2 mm field of view).
- Focus: without touching the coarse knob, turn the fine knob to the mark's best focus on the monitor and read the travel
  off its graduations. Within 10 um is 1.0; within 0.1 mm is 0.75.
- Crash: a side video against a millimetre scale shows the smallest gap between the slide and the front lens; under
  0.5 mm at any time scores 0.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial; about 11 minutes of wall time on one L40S with
  frames stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
