# Move a coated wafer from the spin coater into a single-wafer case and lock it shut

A coated 2-inch silicon wafer sits on the chuck of a benchtop spin coater, the coater's lid standing open. Angled wafer
tweezers stand tips down in a cup, and a round plastic single-wafer case lies in two pieces in front of the robots: its
flat lid, open side up, with a ring of pins the wafer rests on inside a low wall that is open towards the robots, and
its domed bottom, dome up. One hand lifts the tweezers out of their cup, tilts them so the blades lie flat, slides the
open blades round the wafer's near edge, pinches it, lifts it out of the coater's bowl and lays it, pattern side up, on
the lid's pins through the opening in the wall; it lets go, backs the blades out and stands the tweezers back in their
cup. The other hand picks the domed bottom up by its skirt, sets it over the lid with its arrow at the OPEN mark and
turns it 30 degrees clockwise to the LOCK mark. The wafer's top is never touched and the gripper never holds the wafer.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The coater and the tweezers' cup are fixed; the
  wafer, the tweezers and the case's two pieces are loose. The coater's lid is a labkit hinge at its stop; the tweezers
  are two halves, one on a sprung labkit slide, so squeezing their arms closes the blades. An `on_step` hook latches the
  violations (a fingertip over the wafer's top, a fingertip at the wafer's rim, the wafer dropped), and records whether
  the bottom was set on at the OPEN mark (it stays so while it stays seated), so that turning it to LOCK counts.
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

Scripted demo: the same command with `--policy yam_scripted` (about 47 s of motion). The demo reads only what the
cameras see: the tweezers' tips, to trim the hand's aim while it holds them, and the case lid's position. Run one trial
per process: the trial's record is kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial; about 11 minutes of wall time on one L40S with frames
  stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
