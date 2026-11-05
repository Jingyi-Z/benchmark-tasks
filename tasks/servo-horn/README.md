# Fit a servo horn at its centre position while holding the arm link

A 3D-printed arm link for a small robot arm, with a centred hobby servo screwed into its right end, and the servo's
horn lying on the bench. One arm holds the link down by the block at its left end; the other picks up the horn, turns
it so its arm points along the white centre line on the servo, sets the hub over the splined shaft and presses it
down until it is seated. The servo sticks out past the link's foot, so pressing without holding the link tips it over.

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

Scripted demo: the same command with `--policy yam_scripted` (about 28 s of motion). Run one trial per process: the
spline tooth and the press-fit latch are kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: whether the hub is on
the shaft and seated, and where the horn's arm points with the servo at its centre.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
