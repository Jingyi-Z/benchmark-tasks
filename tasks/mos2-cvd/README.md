# Load a tube furnace for MoS2 growth by thermal CVD

A split-tube furnace holds a 2 inch quartz tube with both end caps off. Argon comes in at the near end, so the
near end is upstream. Three boats wait on a tray beside the near end: sulfur powder, MoO3 powder, and a quartz
boat with a SiO2/Si chip face up. A push rod lies on rests in line with the tube. The robot sets each boat in the
tube's mouth by its tongue and pushes it into place with the rod, then racks the rod. The layout is scored:

- sulfur at the furnace's upstream edge, where it stays cool enough to evaporate slowly;
- MoO3 under the thermocouple, in the hot zone;
- the chip 50 to 80 mm downstream of the MoO3 boat, centre to centre.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The furnace, tube, end caps, tray and rod
  rests are fixed; the three boats, the chip and the rod are loose. An `on_step` hook latches the violations,
  follows the end state and checks that everything has been still for the last second.
- `labkit.py`: parts built from several shapes. It hooks `yam_arena.scene` and the embodiment from the task's
  import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket.
  This task uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 4800 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The demo reads only what a camera sees: the boats'
and the rod's poses. Run one trial per process. The score is `task_score` in the log, computed by `score()` in
`task.py`.

## Runtime and seeds

- 4,800 control steps (240 s of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
