# Load a wafer onto a spin coater's chuck with tweezers and a wafer aligner

A 2-inch silicon wafer lies on a small stand beside a benchtop spin coater whose clear lid is shut; a tweezer slot cut
into the stand's front edge runs in under the wafer's edge. Angled wafer tweezers stand tips down in a cup, and a PTFE
wafer aligner (a "fork") lies on the bench: a flat plate with a ring of posts forming a wafer pocket and a slot from the
pocket to its far end that fits round the chuck, ending in a V. One hand lifts the tweezers out of their cup, tilts them
so the blades lie flat, slides the open blades round the wafer's edge with the lower blade in the slot, pinches it,
lifts it well above the coater's lid and lowers it into the aligner's pocket, the lower blade in the pocket's notch; it
lets go, backs the blades out and stands the tweezers back in their cup. The other hand lifts the coater's lid by its
handle bar and holds it open. The first hand picks the aligner up by its handle, carries it over the chuck, lowers it
until its slot straddles the chuck, pushes it on until the V seats against the chuck, lowers it so the wafer is left on
the chuck, centred, slides it back out from under the wafer, lifts it out and sets it down. The other hand then lowers
the lid shut. The wafer's top and edge are never touched by the gripper.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The coater, the wafer stand and the tweezers' cup
  are fixed; the coater's lid is on a damped hinge and falls shut unless held; the wafer, the tweezers and the aligner
  are loose. The tweezers are two halves, one on a sprung labkit slide, so squeezing their arms closes the blades. The
  slide runs along the blades' face normal (a turned joint frame, `labkit.JOINT_FRAME_ROT`), and the sprung blade is
  set back 6.2 mm along its length, so the PEEK tips (friction 0.4 on silicon, `min` combine) meet and close straight
  onto the wafer; the hand squeezes them at about 2 N. Every tweezer collider, blades included, has a 4 mm torsional
  patch radius (minimum 2 mm), half the 8 mm arm face the pads press: an estimated contact approximation, not a
  measured calibration (without it the demo still picks and carries the wafer but centres it to 1.5 mm, not 0.5 mm).
  The wafer, drawn 1 mm thick at its real 1.3 g, has a flat collider strip flush inside the
  edge the tweezers grip (19.4 to 24.9 mm out, 10 mm wide, inside the rim), so the blades meet flat faces there. An
  `on_step` hook latches the violations (a fingertip over the wafer's top or at the wafer's rim; a tweezer blade or
  the aligner touching the wafer's top more than 6 mm inside its rim, checked over points spread across their faces;
  the wafer dropped).
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

Scripted demo: the same command with `--policy yam_scripted` (about 88 s of motion; the trial ends 3 s after full
score). The demo reads only what the cameras see: the tweezers' tips and the wafer, to trim the hand's aim while it
holds the tweezers, and the aligner's position, to line its slot up with the chuck. Run one trial per process: the
trial's record is kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py`. Every positive tier also needs both hands clear
(each fingertip, and the midpoint between a hand's fingertips, at least 80 mm from every task object's surface,
the tools and the lid included) and everything still (no surface point of any object, joint bodies included, moved 1 mm
or more over the last 21 control steps).

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial; about 11 minutes of wall time on one L40S with frames
  stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.

## Shortcut check

`WC_CHECK=tipontop` runs the same demo with the tweezers pinching 14 mm in from the rim, so the upper blade rests on
the wafer's top about 11 mm inside it, past the 6 mm handling edge. The trial must score 0.
