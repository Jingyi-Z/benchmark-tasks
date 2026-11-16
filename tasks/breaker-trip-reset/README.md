# Find the tripped breaker in a home load center and reset it, leaving the others on

A Square D QO-style load center stands on the bench with six QO120 single-pole breakers in two columns of three (0.75 in
pitch). Five are ON. One has tripped: its handle rests midway between ON and OFF and a red flag shows on its face. Both
columns switch ON towards the centre bus. As on a real QO (Schneider Electric
FAQ FA353467), a tripped breaker only goes back on after its handle has been pushed all the way to OFF, which re-arms
it; pushed straight towards ON it stays tripped. The instruction names neither the breaker nor the rule.

Builds on the idea of TK-2SE38R / PR #341 and TK-Q3DMLF / PR #427, which were not accepted (final) for using
homemade sliding switches and a free-sliding bar in place of a breaker, and for an instruction that spelled out the
reset sequence; this version uses a QO-sized panel and breakers, a trip-free mechanism made of contacts, and a goal-only
instruction. None of their code or text is reused.

The working breakers' handles are a finger's width (19.05 mm row pitch) above and below the tripped one, and switching
any of them off ends the trial at 0.

## The trip-free mechanism (modelled)

The QO's internal geometry and forces are not published, so the behaviour is built from contacts between real parts,
with no state switches or scripted latches:

- the handle swings on a vertical pin and stays where it is left (pivot friction);
- behind the dead front, the handle's lever bears on a 45 degree ramp on a sliding arming block; only pushing the
  handle all the way to the OFF stop slides the block about 10 mm, where its own friction keeps it (the reset);
- the same block carries the red trip flag, proud of the breaker's face until the reset slides it out of sight;
- an arm on the same block stops a finger on the handle about 1.5 degrees past midway towards ON until the block has
  moved, so a tripped handle cannot be turned on directly.

A control run on the same scene, pushing the tripped handle straight to ON, ends with the handle at +1.5 degrees
and scores 0.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
- `labkit.py`: parts built from several shapes with sprung, damped and friction joints. It hooks `yam_arena.scene`
  and the embodiment from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 900 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`.

The score is `task_score` in the log, computed by `score()` in `task.py`. Switching a working breaker off, release
(every fingertip and each hand's pad midpoint 80 mm from the breakers' faces) and stillness (1 s, 21 control samples)
are tracked every control step in `_watch()`.

## Runtime and seeds

- 900 control steps (45 s of simulated time) per trial; the demo finishes in about 480.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
