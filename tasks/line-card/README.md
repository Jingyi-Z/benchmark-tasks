# Seat a line card by closing both ejector levers together

A small open-top card cage on the bench with a network line card resting in its guides, 7 mm short of home. The
card's front panel has an orange ejector lever at each end, both swung open towards the robots. Closing a lever levers
the card in by its claw on a catch on the cage, but the card only goes in as far as the less-closed lever takes it: a
lever closed ahead of the other cocks the card and stalls 9 degrees ahead. Both arms take one knob each and swing the
two levers closed together until they lie flat against the panel and the card is home.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The cage is one fixed labkit part with three joints: the card (a slide along the slot, 0 at the start, 7 mm at
    home, its upper limit) and the two levers (hinges about the vertical through their pivot pins, riding on the card,
    0 to 90 degrees, built open so the first frames show them there).
  - The cam is an `on_step` hook that sets drive targets only, every control step (`cam_targets`). The card's target
    is 7 mm times the less-closed lever's fraction of 90 degrees (stiff: 4 N per mm). Each lever's target is its own
    angle while it is free, so it stays where it is left and drags only lightly, and is capped at 9 degrees ahead of the
    other lever (0.7 mm of travel, the card's play before it cocks), so a lever closed alone stalls there. The lever's
    spring is 1.5 N m per radian, so a hand forcing a lone lever further gains about 2 more degrees per N.
  - The scripted demo drives both knobs at once with fingers part-open round them (pads 16 mm apart, closing along the
    knob's path, the trailing pad pushing), as in the spin coater and tool changer tasks, on one shared schedule read
    from the plan's clock: a single smooth 8 s swing to 2 degrees past closed. The knobs' path is aimed at the live
    card: an `on_step` hook reads the card's slide and the pivots move in with it. The fingers lean 30 degrees about
    their closing line, wrists outward, so the elbows stay apart, and that line runs 8 degrees behind the knob's path
    so the wrist roll stays inside its limit.
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

Scripted demo: the same command with `--policy yam_scripted` (about 19 s of motion). Run one trial per process: the
demo's live state is kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: the two levers' angles
and how far the card is from home.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; the demo ends early about 9 s after the card is seated. About
  3 minutes of wall time on one L40S with frames stored at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
