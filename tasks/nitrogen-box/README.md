# Store a sample in a nitrogen dry box and let its self-closing door latch

A small acrylic nitrogen dry box with a self-closing front door, hinged on its right edge, and a sample jar on the
bench beside it. One arm opens the door by its lever handle and holds it open against the closer; the other stands the
jar upright on the white square on the shelf inside and takes the hand out. The door is then let close, and must end
fully shut on its magnetic catch. Let go of the door early and it swings shut.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The door is a labkit hinge with a spring drive towards shut and damping (the door closer), and a hard stop at
    shut. Shut means its free edge is within 5 mm of the cabinet's front.
  - The scripted demo grips the handle's bar from above and carries it round the hinge to 95 degrees, the wrist
    turning with the door; the block on the bar's end keeps it from sliding out of the fingers against the closer. The
    other arm grips the jar from the side, fingers level and pointing into the box, tipped 15 degrees down;
    `task.py` lets a demo waypoint carry a full gripper rotation (the `Orient` class), so yam_arena itself is
    unchanged.
- `labkit.py`: parts built from several shapes, with hinges and slides (the door is a sprung hinge). It hooks
  `yam_arena.scene` and the embodiment from the task's import, so no change to yam_arena is needed. If a labkit copy
  with `GLASS_COLORS` is used (as in the analytical balance task), the acrylic is drawn see-through; this copy draws
  it opaque.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 35 s of motion).

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 4 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
