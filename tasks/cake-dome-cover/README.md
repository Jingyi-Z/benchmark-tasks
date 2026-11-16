# Swap a fresh cake in under a pastry display dome and cover it

A clear polycarbonate pastry display dome (American Metalcraft LFTD8 size: 206 mm across, 168 mm tall overall, a
rounded top with a ball knob on a short neck) rests on an empty plate on the counter's display spot. A 6 in sponge cake
waits on an identical plate to the right. The cake must go where the empty plate is, under the dome, and the dome must
come down over it. How the plates and the dome are moved is up to the robot: the demo holds the dome up with one hand
while the other swaps the plates; sliding the empty plate aside with the dome still on it and moving the dome across
afterwards is also allowed.

What makes it hard: the dome hangs from its small knob about 130 mm above its rim, so it sways and the arm sags under
it while it is held; in the demo the holding hand keeps it clear of the other arm while that arm pushes the cake plate
in (which pushes the empty plate out); then the dome has to come down square round the cake, where the rounded wall leaves
about 15 mm at the cake's top edge. Any touch that moves or tilts the cake on its plate ends the trial at 0.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
- `labkit.py`: parts built from several shapes. It hooks `yam_arena.scene` and the embodiment from the task's
  import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1500 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`.

The score is `task_score` in the log, computed by `score()` in `task.py`. The cake-knocked check,
release (every fingertip and each hand's pad midpoint 80 mm from the dome, both plates and the cake) and stillness
(1 s, all pairs of 21 control samples) are tracked every control step in `_watch()`.

The scripted demo reads only what a camera sees: the dome's and the plates' positions. The left hand lifts the dome by
its knob, holds it to the left of the spot and raises itself by the arm's seen sag; the right hand pushes the cake plate
in, stopping short and closing the last gap to the spot as seen; the left hand then brings the dome over the cake plate,
centres it as seen and lowers it until the rim is seen resting on the plate before letting go.

## Runtime and seeds

- 1,500 control steps (75 s of simulated time) per trial; the demo finishes in about 1,130.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
