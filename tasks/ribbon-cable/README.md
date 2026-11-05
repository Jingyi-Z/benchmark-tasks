# Plug a keyed ribbon cable connector onto a sensor board while holding its stand

A small sensor board screwed by one end to a 3D-printed bench stand, with a shrouded 2 x 5 pin box header near its
free end, and the black IDC socket connector of a ribbon cable standing on a tray. One arm holds the stand down by the
rubber pad at its left end; the other picks up the connector by its two ends, turns it a quarter turn so its key faces
the header's key slot, lowers it into the header and presses it down until the press fit gives and it seats. The board
sticks out past the stand's foot, so pressing on the header without holding the stand tips it over (at about 1.2 N; the
fit takes 5 N). The other way round, the connector's key lands on the header's wall and it cannot go in.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The stand, board and header are one loose body
  with one labkit joint: a sprung contact plate inside the header (a slide). Every control step an `on_step` hook sets
  its drive target. The plate is sprung up (5000 N/m) and carries the pin tips; once the connector, in the header with
  its key in the slot, pushes it down 1 mm (5 N), its target drops to the bottom of its 2.5 mm travel and stays there,
  and the connector follows to its seat. Pushing the plate with anything but the connector does not count.
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

Scripted demo: the same command with `--policy yam_scripted` (about 31 s of motion). The left hand presses straight
down on the stand's rubber pad (about 4 N) rather than gripping the stand, so it cannot drag it sideways; that holds
the stand flat against presses up to about 17 N on the header. The right hand grips the connector by its ends and turns
it on the way over. Its placement is closed loop: every control step it reads the stand's pose and the connector's,
aims the connector at the header where it is now, turns with the stand's yaw, and trims out the connector's measured
offset and turn; the trim is frozen once the connector goes down into the header. It then presses slowly until the
press fit gives, eases off and lets go. Run one trial per process: the press-fit latch and the demo's live state are
kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: whether the stand is
still in its outline and flat, whether the connector is in the header, and whether it is seated with its key in the
slot.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
