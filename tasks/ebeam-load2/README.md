# Clip a wafer onto an e-beam evaporator's sample plate and load it face down

An electron-beam evaporator deposits metal from sources below the samples, so a wafer rides into it face down, clipped
to a sample plate that hangs in a ring holder. A clip that misses the wafer's round edge, too few clips, or a screw too
long for its hole means a wafer on the chamber floor or a jammed transfer. The task:

1. **Take out the ring holder** from the open load lock and set it on the sample plate (its recess fits round the
   plate's rim).
2. **Centre the wafer.** The 100 mm wafer starts 5 mm off centre. Push it by its edge, never touching the pattern.
3. **Choose the clips.** Eight swing clips stand in the plate's 4 inch ring of holes. The wafer's flat faces one, which
   would bite nothing. One has a screw too long for its hole: its head stands 8 mm proud, and tightened, its tip would
   stick out of the plate's back and jam the transfer. Swing in clips that hold the round edge with no gap over 150
   degrees between them (the demo uses 0, 135 and 270 degrees).
4. **Tighten their screws** with the screwdriver, at least 300 degrees each, after the clip is swung in, and put the
   screwdriver back.
5. **Turn the plate and ring over together** (wafer face down; the ring now carries the plate) and set them on the
   load lock's holder.
6. **Close the door.** Pull it down by its handle until its catch holds it, then let go.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The stand, load lock, door frame and screwdriver
  holder are fixed; the plate (with eight hinged clips, each with a hinged screw), the wafer, the ring holder and the
  screwdriver are loose. The door slides on a joint. An `on_step` hook works the door's catch, latches the violations,
  follows the screws and checks that everything has been still for the last second.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 5000 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The demo reads only what a camera sees: the wafer's,
plate's, ring's, door handle's and screwdriver tip's positions and the screw slots' directions. Run one trial per
process. The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 5,000 control steps (250 s of simulated time) per trial. The scripted demo finishes in about 225 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
