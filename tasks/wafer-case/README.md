# Move a baked wafer from the hotplate into a single-wafer case with tweezers and lock it shut

A 2-inch silicon wafer lies just baked in the middle of a benchtop hotplate's 100 mm heating plate, where tweezers
cannot get under its edge. Angled wafer tweezers stand tips down in a cup, and a round polypropylene single-wafer case
lies in two pieces in front of the robots: its flat lid, open side up, with a ledge ring the wafer's edge rests on
inside a low wall, both open towards the robots, and its clear domed bottom, dome up. One hand lifts the tweezers out
of their cup, sets their tips down on the plate behind the wafer and slides it, by its rim, out over the plate's front
edge until about 12 mm overhangs. It then tilts the tweezers so the blades lie flat, slides the open blades round the
overhanging edge, pinches it, lifts it and lays it, pattern side up, on the lid's ledge through the opening in the
wall; it lets go, backs the blades out and stands the tweezers back in their cup. The other hand picks the domed bottom
up by its skirt, sets it over the lid with its arrow at the OPEN mark and turns it 30 degrees counterclockwise to the LOCK
mark. The wafer's top, the hot plate and the wafer's edge are never touched by the gripper.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The hotplate and the tweezers' cup are fixed;
  the wafer, the tweezers and the case's two pieces are loose. The tweezers are two halves, one on a sprung labkit
  slide, so squeezing their arms closes the blades; the case's dome is a triangle mesh colliding as its convex hull.
  An `on_step` hook latches the violations (a fingertip over the wafer's top, at the wafer's rim, or on the hot plate;
  the wafer dropped), and records whether the bottom was set on at the OPEN mark (it stays so while it stays seated),
  so that turning it to LOCK counts.
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

Scripted demo: the same command with `--policy yam_scripted` (about 60 s of motion; the trial ends 10 s after full
score). The demo reads only what the cameras see: the tweezers' tips, to trim the hand's aim while it holds them, the
wafer's position, to push it and to take it by the overhang, and the case lid's position. Run one trial per process:
the trial's record is kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial; about 11 minutes of wall time on one L40S with frames
  stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
