# Load an e-beam evaporator's chuck into its load lock

An electron-beam evaporator deposits metal from sources below the samples, so the sample chuck goes into the load lock
face down. Before that, the samples have to be fixed so they cannot fall off, and nothing on the chuck may stand taller
than the load lock allows, or it jams during the transfer.

On a loading stand lies the evaporator's aluminium chuck, face up. On it are:

- a patterned 100 mm wafer, pattern up, inside six locating pins;
- six swing clips round the wafer, all parked open (the wafer's primary flat faces one of them);
- a 10 mm chip taped down by its corners with Kapton;
- a 10 mm chip mounted on a 3 mm glass block.

Beside the stand is the load lock's transfer fork, and a card on the bench reads LOAD LOCK / MAX 1.5 MM.

The task follows a Lesker PVD load-lock SOP:

1. Take off the chip on the glass block. It is far over the 1.5 mm limit.
2. Swing in clips that bite the wafer's round edge, with no gap over 150 degrees between them. The clip facing the flat
   bites nothing, so the right clips are the three at 0, 120 and 240 degrees.
3. Turn the chuck over so the samples face down, towards the sources, and set it on the transfer fork.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The stand, tray, load lock and card are fixed; the
  chuck (with six hinged clips), the wafer and the glass block are loose. An `on_step` hook latches the violations (the
  wafer off its seat, the block tipped over, a fingertip at the pattern) and checks that everything has been still for
  the last second.
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

Scripted demo: the same command with `--policy yam_scripted`.

1. The left hand swings in the 0 degree clip by pushing its knob.
2. The right hand lifts the chip on the glass block onto the tray, then swings in the 240 degree clip.
3. The left hand swings in the 120 degree clip.
4. Both hands pinch the chuck's side edges, lift it and roll it over together.
5. They carry it over the fork and trim their aim by the chuck's seen position.
6. They set it down, let go and back out.

The demo reads only what a camera sees: the chuck's pose. Run one trial per process. The score is `task_score` in the
log, computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial. The scripted demo finishes in about 65 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
