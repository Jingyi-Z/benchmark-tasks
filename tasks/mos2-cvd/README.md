# Load a tube furnace for two-precursor thermal CVD and cap the tube

A split-tube furnace holds a 2 inch quartz tube with both end caps off, each in a cradle beside its end. Argon
comes in at the near end, so the near end is upstream. Three boats wait on a tray beside the near end:
precursor 1 (a yellow powder), precursor 2 (a white powder), and a quartz boat with a silicon substrate face up. A quartz push rod with a flat grip block on its back end
lies on a rack beside the tube's line. The right hand sets each boat in the tube's mouth by its tongue, takes the
rod by its block and pushes the boat into place, then racks the rod. Meanwhile the left hand caps the far (exhaust)
end; the right hand caps the near (inlet) end last. The layout is scored:

- precursor 1 at the furnace's upstream edge, where it stays cool enough to evaporate slowly;
- precursor 2 under the thermocouple, in the hot zone;
- the substrate 50 to 80 mm downstream of precursor 2, centre to centre;
- both caps pushed home and square.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The furnace, tube, tray, rod rack and cap
  cradles are fixed; the three boats, the substrate, the rod and both caps are loose. An `on_step` hook latches the
  violations, follows the end state and checks that everything has been still for the last second.
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

Scripted demo: the same command with `--policy yam_scripted`. The demo reads only what a camera sees: the poses of
the boats (outside the tube), the rod and the caps. Run one trial per process. The score is `task_score` in the log, computed by `score()` in
`task.py`.

## Runtime and seeds

- 4,800 control steps (240 s of simulated time) per trial. The scripted demo takes about 4,150.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
