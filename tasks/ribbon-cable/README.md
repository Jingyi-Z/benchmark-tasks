# Plug a ribbon cable into a sensor board held up off its foam

A bare sensor board lies across two soft foam bars on an anti-static mat, with a shrouded 2 x 5 box header on top and
an electrolytic capacitor under it, and the black IDC socket connector of a ribbon cable stands on a tray. One arm
pinches the board by its stiffened left end, lifts it off the foam and holds it level; the other picks up the connector
by its two ends, lowers it into the header with its key in the slot and presses it until the press fit gives and it
seats. Pressed while the board lies on the foam, the foam gives and the capacitor's can is pushed onto the mat before
the fit gives (about 0.5 N against 1.2 N), which scores 0.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The board is one loose body with one labkit joint:
  a sprung contact plate inside the header. The mat is fixed, and each foam bar on it is a sprung slide. An `on_step`
  hook sets the plate's drive target (it drops to the seat once the keyed connector has pushed it down 1 mm), records
  whether the capacitor's can has ever come within 0.5 mm of the mat, and whether both hands' grip points are at least
  3 cm from the connector.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 1200 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 35 s of motion; the trial ends 10 s after full
score). The demo reads only what the cameras see: the board's pose and the connector's height on the board. The left
hand, level and pointing away from the robots with its fingers closing up and down, slides its lower finger under the
board's overhanging left end and closes on the board and its stiffener, lifts the board 7 cm and holds it. The right
hand aims at the header where the board is, lowers the connector in and presses slowly until the connector is seen at
its seat, eases off, lets go and lifts away; the left hand then sets the board back on the foam and parks. Run one trial
per process: the press fit and the trial's record are kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 1,200 control steps (60 seconds of simulated time) per trial; about 5 minutes of wall time on one L40S with frames
  stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
