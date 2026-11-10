# Load a boat of chips into a tube furnace's hot zone and seal the tube

Chips with 2D materials on them are annealed in a quartz tube that runs through a split-tube furnace. They ride in a
small quartz boat that is pushed in with a long hook rod. The boat ends up hidden inside the furnace, so the rod carries
a red band that meets the tube's end when the boat sits on the thermocouple, at the hot zone's centre. A jolt slides the
chips off their places on the boat (a scratched or lost sample), so the rod is pushed slowly. The task:

1. **Push the boat to the hot zone.** The rod is as long as the push, so it is fed hand over hand: one hand takes it
   near its back, pushes one stroke and lets go, the other takes it behind and pushes on until the red band meets the
   tube's end. The boat is centred within 10 mm.
2. **Pull the rod back out** the same way and lay it back in its trough where it started.
3. **Seal the tube.** Take the quick-release end cap off its stand, hold it in line with the tube, push it fully home
   (its plate against the tube's end, square: its O-ring inside then seals all round) and let go.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The furnace with its tube, the rod's trough and the
  cap's stand are fixed; the boat, the three chips, the rod and the cap are loose. An `on_step` hook latches the
  violations (a chip that slides or tilts on the boat, a boat that tips), follows the end state and checks that
  everything has been still for the last second.
- `labkit.py`: parts built from several shapes. It hooks `yam_arena.scene` and the embodiment from the task's import,
  so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The demo reads only what a camera sees: the rod's
position and its red band, and the cap's position. Run one trial per process. The score is `task_score` in the log,
computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (120 s of simulated time) per trial. The scripted demo finishes in about 80 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
