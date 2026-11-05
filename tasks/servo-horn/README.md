# Fit a servo horn at its centre position while holding the arm link

A 3D-printed arm link for a small robot arm, an L-shaped bracket with a centred hobby servo screwed into the end of
its arm, and the servo's horn lying on the bench. The link stands on a foot with a block at its left end; at the foot's
right end an upright carries the arm, which holds the servo 10 cm out past the upright and 2.5 cm above the bench. One
arm holds the link down by the block; the other picks up the horn, turns it so its arm points along the white centre
line on the servo, sets the hub over the splined shaft and presses it down until it is seated. Pressed without holding
the link, the link tips over the upright's edge, and the pressing hand cannot keep it up by itself (see below).

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The link and servo are one loose body with two
  labkit joints: the servo's output shaft (a hinge) and a sprung collar on it (a slide). Every control step, an
  `on_step` hook sets both joints' drive targets:
  - The spline: the hub is a square hole round a round pin, so the 25 teeth are kept in the hook. When the hub first
    goes onto the shaft, the tooth it meets is fixed; while it stays on, the shaft's target follows the horn at that
    tooth, so the free shaft turns with the horn, and the horn cannot change tooth without coming off.
  - The press fit: the collar under the hub is sprung up (5000 N/m). Once the hub, on the shaft, pushes it down
    1 mm (5 N), its target drops to the bottom of its 3 mm travel and stays there, and the hub follows to its seat.
    Pushing the collar with anything but the horn's hub does not count.
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

Scripted demo: the same command with `--policy yam_scripted` (about 29 s of motion). The left hand presses straight
down on the top of the link's block (about 4 N) rather than gripping it, so it cannot drag the link sideways. The right
hand's placement is closed loop: every control step it reads the link's pose and the horn's, aims the hub at the shaft
where it is now, turns with the link's yaw and the shaft's dot, and trims out the hub's measured offset; the trim is
frozen once the hub goes down over the shaft. It then presses slowly until the press fit gives, eases off and lets go.
Run one trial per process: the spline tooth, the press-fit latch and the demo's live state are kept in the task module
and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: whether the hub is on
the shaft and seated, and where the horn's arm points with the servo at its centre.

## Why it takes two hands

Statics of the link as built (120 g with its servo, plus 15 g of shaft and collar; the hub presses 64.5 mm above the
bench and 100 mm outside the upright's edge; the block is 240 mm inside it):

- Unheld, a vertical press at the shaft tips the link at 0.8 N. The press fit needs 5 N.
- The pressing hand grips the horn, whose hub is on the shaft, so it can also push the shaft sideways, and a tip moves
  the shaft out as well as down. Even a laterally rigid hand cannot keep the link up past 2.2 N of press (2.9 N with a
  bench friction of 0.7 instead of the simulated 0.55): beyond that the bench cannot carry the sideways push at the
  edge, and the link tips or its edge slides.
- Held down by 4 N on the block, the link stays put up to 10.4 N at the shaft (8.0 N at 3 N); the 5 N press needs
  1.7 N on the block.

`onehand_check.txt` has the `sed` command for the one-hand variant (the left hand stays 6 cm above the block and never
touches it). It is expected to score below 1.0: the collar never latches, so the horn ends at best on the shaft but not
seated (0.25).

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
