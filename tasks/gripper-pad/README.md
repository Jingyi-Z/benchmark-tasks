# Fit a tactile sensor pad on a gripper fingertip while holding the finger's holder

A spare robot gripper finger stands in a printed holder, its fingertip sticking out past the holder's head, and a
small tactile sensor pad lies on a tray. One arm holds the holder down by the block at its far end; the other picks up
the pad, turns it a quarter turn so its cable tab points along the finger away from the fingertip, sets its two pins
in the finger's holes and presses it home. The fingertip overhangs the head, so pressing without holding tips the
holder over.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The finger and holder are one loose body with one
  labkit joint: a sprung plate (a slide) under the two holes, which the pad's pins rest on. Every control step an
  `on_step` hook sets the plate's drive target: sprung up (4000 N/m) it holds the pad's underside 2 mm above the face;
  once the pad, in the holes, pushes it down 1 mm (4 N), its target drops to the bottom of its 2.5 mm travel and stays
  there, and the pad goes down onto the face. Pushing the plate with anything but the pad in the holes does not count.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 34 s of motion). The left hand presses straight
down on the top of the holder's block (about 3 N) rather than gripping it, so it cannot drag the holder sideways. The
right hand grips the pad by the two sides without the tab and turns it on the way over. Its placement is closed loop:
every control step it reads the holder's pose and the pad's, aims the pad at the holes where they are now, turns with
the holder's yaw and trims out the pad's measured offset and the tab's angle; the trim is frozen once the pins go down
towards the holes. It then lets go, closes its fingers over the middle of the pad, presses the rubber face down slowly
until the press fit gives, eases off and lifts away. Run one trial per process: the press-fit latch and the demo's live
state are kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: whether the pins are in
the holes, whether the pad is seated, and which way its tab points.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; the demo should reach success at about 22 s and end the
  trial 18 s later (`SUCCESS_HOLD_S`). About 3 minutes of wall time on one L40S with frames stored at 640 x 352
  (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
