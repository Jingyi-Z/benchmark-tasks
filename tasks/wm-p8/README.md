# Mount the n-type (100) wafer face down on a dicing-tape mounter

Four unlabelled 100 mm silicon wafers lie pattern side up in open single-wafer carriers. They look alike: only their
flats tell them apart (SEMI convention, seen from the pattern side). Every wafer has a primary flat; p-type (111) has
no second flat, n-type (111) has one 45 degrees counterclockwise from the primary, p-type (100) one 90 degrees
counterclockwise, and n-type (100) one opposite it. Beside them stands a manual tape-frame mounter: a chuck plate with
centring and alignment grooves and a tweezer recess, a steel film frame already on its three locating pins, and a WAFER
VACUUM toggle. Two pairs of angled wafer tweezers stand in cups.

The task follows a standard tape-station SOP: the wafer goes on the chuck face down (the tape goes on its back),
centred and aligned, then the vacuum is switched on before taping. Here the primary flat must face the frame's flat side.
Taping is not part of the task.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The carriers, cups, mounter and toggle are fixed;
  the wafers and tweezers are loose. Each pair of tweezers is two halves, one on a sprung labkit slide, so squeezing their
  arms closes the blades. An `on_step` hook latches the violations (a fingertip at a wafer's pattern or rim; a wafer
  dropped) and checks that everything has been still for the last second.
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

Scripted demo: the same command with `--policy yam_scripted`. The right tweezers take the wafer out of its carrier and
stand it on edge; the left tweezers take its top edge from behind and turn it over, so it hangs pattern side down from
blades that are above it, trim their aim by the wafer's seen position over the chuck and lay it on the chuck through
the recess. The left hand stands its tweezers back in their cup and switches the vacuum on; the right hand keeps its
tweezers, out of the way. A wafer cannot be turned face down in one pair of tweezers and still be laid down by them, so
the hand-over is needed (found with an IK search: one arm cannot turn the held wafer over on its own). The demo reads
only what the cameras see: the wafer's pose and the tweezers' tips. Run one trial per process.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
