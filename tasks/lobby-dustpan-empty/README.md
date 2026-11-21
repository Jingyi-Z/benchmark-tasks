# Empty an upright lobby dustpan into a wastebasket and park it on its mark

Two arms empty two tape cores from a full-size upright lobby dustpan (its hopper hangs on a free pivot) into a wastebasket, then stand the empty dustpan back on its floor mark with the handle upright and let go.

## Run a trial
Scripted demo (the scene, demo and scorer are all in `task.py`; two helper modules implement the scripted motion):
```
LD_MODE=integrated LD_REPINCH_HOLD=0.365,-0.230,0.650 \
inspect-robots run --embodiment yam_arena --policy yam_scripted -P hold_s=1.0 \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Empty both discarded tape cores from the upright lobby dustpan into the wastebasket, then stand the empty dustpan back on its floor mark with its handle upright." \
  --scorer task_score --max-steps 1280 --store-frames --log-dir logs/demo
```
Dream-1 command:
```
inspect-robots run --embodiment yam_arena --policy dream   -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352   --instruction "Empty both discarded tape cores from the upright lobby dustpan into the wastebasket, then stand the empty dustpan back on its floor mark with its handle upright."   --scorer task_score --max-steps 1280 --store-frames --log-dir logs/dream1
```
The recorded scripted demo also set `LD_MODE=integrated`, `LD_REPINCH_HOLD=0.365,-0.230,0.650` and `LD_LOG=1`. The first two only select the scripted demo's plan, and `LD_LOG` only prints telemetry lines; none changes the scene, the policy input or the scorer.

- Runtime per trial: 339 s wall time on one RTX 5090 for the scripted demo (1,253 control steps; the episode ends on the scorer's success hold) and 248 s for the Dream-1 run (all 1,280 steps), with frames stored at 640x352.
- Seeds: none. `object_jitter=0`, one fixed starting layout.
- Score: the `task_score` scorer reads the task's own `score()`; `[score]` lines print every 10 steps. Possible scores 0, 0.5 and 1.0.

## Scoring
See task.json `scoring`. The thresholds are the scorer constants: `RELEASE` (80 mm hand clearance), `STILL_N`/`STILL_TOL` (21 samples at 20 Hz, 5 mm), and in `score()` the 100 mm mark, 15 mm, 10 deg and 15 deg limits; a core counts in the wastebasket or the hopper only if its whole cylinder lies inside the inner walls (orientation-aware extents).

## Real trial
Stock items only: the dustpan (FG253100), the wastebasket (FG295600), two 3 in kraft tape cores, floor tape and a 100 mm scale bar. Reset per task.json `reset_notes`; read the result with a ruler and phone level, and use synchronized video for the 80 mm release and the 5 mm stillness over the last second.

## Files
- `task.py`: scene, scripted demo and scorer.
- Two helper modules implement the scripted motion; `task.py` imports them.
- `labkit.py`, `arena_assets_fix.py`: scene helpers.
