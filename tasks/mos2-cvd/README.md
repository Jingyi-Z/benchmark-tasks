# Load a tube furnace for two-precursor thermal CVD and cap the tube

A split-tube furnace holds a 2 inch quartz tube with both end caps off, each in a cradle beside its end. Argon
comes in at the near end, so the near end is upstream. Three boats wait on a tray beside the near end:
precursor 1 (a yellow powder), precursor 2 (a white powder), and a quartz boat with a silicon substrate face up. A quartz push rod with a flat grip block on its back end
lies on a rack beside the tube's line. The tube lies loose in the furnace's saddles,
centred. The left hand takes it by its far end, slides it 75 mm towards the far side so the near end stands close to the
furnace, and holds it there. The right hand sets each boat in the tube's mouth by its tongue, takes the rod by its block
and pushes the boat in, then racks the rod. The left hand slides the tube back, centred, and lets go; then each hand caps
its end (the left the far, exhaust end; the right the near, inlet end). The layout is scored against the furnace:

- precursor 1 at the furnace's upstream edge, where it stays cool enough to evaporate slowly;
- precursor 2 under the thermocouple, in the hot zone;
- the substrate 50 to 80 mm downstream of precursor 2, centre to centre;
- both caps pushed home and square;
- the tube back where it started, 120 mm out of the furnace at each end.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The furnace, tray, rod rack, cap cradles
  and the tube's two blocks are fixed; the tube (0.49 kg), the three boats, the substrate, the rod and both caps are loose. An `on_step` hook latches the
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
the boats (outside the tube), the rod, the caps and the tube. Run one trial per process. The score is `task_score` in the log, computed by `score()` in
`task.py`.

## Runtime and seeds

- 4,800 control steps (240 s of simulated time) per trial. The scripted demo takes about 4,150.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.

## Scoring a real trial

- Boats: the quartz tube is clear in a real trial, so each boat's centre is read with a ruler along the tube from the
  furnace's near end and from the thermocouple's tip.
- Tube: a ruler at each end of the furnace gives how far the tube stands out (120 mm each, within 5 mm).
- Caps: pushed home when their end plate touches the tube's end; square by eye against a set square.
- Spills, a tipped boat or a flipped substrate: seen in the video.
