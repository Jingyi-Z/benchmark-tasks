# Swap the layer 1 chrome mask for layer 2 on a mask aligner's holder, chrome side down

A contact mask aligner's mask holder lies pulled out and upright on its loading stand. It still holds the LAYER 1 chrome
mask from the last exposure, chrome side down, under the MASK VACUUM (its toggle is ON). Behind it, a two-shelf mask rack
holds the LAYER 2 mask on its lower shelf, lying chrome side up; the upper shelf is LAYER 1's, empty.

The masks are 5 inch soda-lime plates with a chrome pattern on one face. Both faces look alike; the only cue to which
face carries the chrome is the label, written in chrome, which reads correctly from the chrome side and mirrored through
the glass. In contact printing the chrome faces the wafer, so a mask goes on the holder chrome side down, against the
holder's stop pins.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The holder, rack and toggle are fixed; the masks
  are loose. An `on_step` hook latches the violations (a fingertip at a mask's patterned area; a mask dropped; LAYER 1
  moved off the holder while the vacuum was not off) and checks that everything has been still for the last second.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The right hand switches the mask vacuum off. Both hands
pinch LAYER 1's two side edges where they overhang the holder, lift it and set it on the upper shelf. They take LAYER 2
from the lower shelf, draw it out, lift it, roll it over together about the line between them, lower it onto the holder
just in front of the stop pins and push it back against them, trimming their aim by the mask's seen position. The
right hand switches the vacuum on. The demo reads only what the cameras see: the masks' poses. Run one trial per
process.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
