# Recap a used dental needle one-handed, scooping its cap off the tray

A dental aspirating syringe lies on a rest in an instrument tray, its used needle uncapped; the cap lies loose on the
tray floor beside it. The needle must end up capped and the syringe back on its rest, done the way US rules
(OSHA 1910.1030) allow: one-handed, with the free hand never near the needle and the cap not held while the needle
goes in. The syringe hand scoops the lying cap onto the needle, then presses its closed end against the tray's far rim
so the cap seats on the hub.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
- `labkit.py`: parts built from several shapes (the cap's funnel is four sloped walls). It hooks `yam_arena.scene` and
  the embodiment from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state. The three violations
(both hands near the needle, the cap touched before it is seated, the needle on the tray) are latched every control
step in `_watch()` and score 0 for the rest of the trial.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
