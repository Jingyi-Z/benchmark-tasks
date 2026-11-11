# Open a backfilled tube furnace safely and load a boat of chips under its thermocouple

A compact split-tube furnace is being backfilled: its gas valve is open and the quartz tube is over atmospheric,
rising. The tube's near end is shut by a quick-release cap; a gauge on the far end shows the pressure. The cap may only
come off while the gauge is in its green band (below it the pressure holds the cap on, above it the tube is
over-pressured), and the gas must be shut as soon as it is off. Then the lid is lifted to see the thermocouple, the boat
of chips is pushed under it with a hook rod while the other hand holds the loose, slippery tube, and everything is put
back. The task:

1. **Open the tube in the green band.** Take the cap off while the gauge reads 2 to 8 kPa and shut the gas valve within
   5 s.
2. **Lift the lid** by its handle; it stays where it is left.
3. **Load the boat.** Hold the tube with one hand; push the boat with the rod, hand over hand, until its centre is
   within 8 mm of the thermocouple. The chips must not slide on the boat.
4. **Put everything back.** Pull the rod out onto its rests, shut the lid, push the cap fully home and square, let go.

## Files

- `task.py`: scene, gas model, scoring and scripted demo (yam_arena task file). The furnace and its lid, the gauge,
  the gas valve, the rod's rests and the cap's stand are fixed; the tube, the boat, the three chips, the rod and the
  cap are loose. An `on_step` hook runs the pressure model, drives the gauge's needle, latches the violations (the cap
  off below the green band, over-pressure, gas left on with the tube open, a chip that slides or tilts, the boat
  tipping, the tube moving or rolling), follows the end state and checks that everything has been still for the last
  second.
- `labkit.py`: parts built from several shapes. It hooks `yam_arena.scene` and the embodiment from the task's import,
  so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 3600 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The demo reads only what a camera sees: the rod's,
boat's and cap's poses. It pulls the cap at a fixed time that, at the gauge's rise of 0.4 kPa/s from 1 kPa, falls in the
green band. Run one trial per process. The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 3,600 control steps (180 s of simulated time) per trial. The scripted demo finishes in about 126 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
