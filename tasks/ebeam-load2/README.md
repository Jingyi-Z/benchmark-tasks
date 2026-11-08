# Load a 4 inch wafer into an e-beam evaporator's load lock

An electron-beam evaporator deposits metal from sources below the samples, so the samples go into its load lock face
down, on a sample plate carried by a ring holder. The task follows a Lesker PVD load-lock SOP:

1. **Vent, then open.** The load lock starts under vacuum. Press VENT and give it 10 s to reach air before lifting its
   door. Lifting it sooner is scored 0.
2. **Take out the ring holder** from the load lock's holder and set it on the sample plate (its skirt fits round the
   plate's rim).
3. **Centre the wafer.** The 100 mm wafer starts about 5 mm off centre. Push it by its edge, never touching the pattern.
4. **Choose the clips.** Eight swing clips stand in the plate's 4 inch ring of holes. The wafer's flat faces one, which
   would bite nothing. One has a screw too long for its hole: its head stands 6 mm proud, and tightened, its tip would
   stick out of the plate's back and jam the transfer. Swing in clips that hold the round edge with no gap over 150
   degrees between them (the demo uses 45, 180 and 315 degrees).
5. **Tighten their screws** with the screwdriver, at least 300 degrees each, after the clip is swung in.
6. **Turn the plate and ring over together** (samples face down; the ring now carries the plate) and set them on the
   load lock's holder.
7. **Close the door and pump.** Pull the door down, hold it shut with one hand and press PUMP with the other, and keep
   holding it for 2 s while it pumps down. Then let go.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The stand, load lock, door frame, control box and
  screwdriver holder are fixed; the plate (with eight hinged clips, each with a hinged screw), the wafer, the ring holder
  and the screwdriver are loose. The door slides on a joint; the buttons are spring-loaded slides. An `on_step` hook
  follows the load lock's pressure from the buttons, latches the violations and checks that everything has been still
  for the last second.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 4000 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The demo reads only what a camera sees: the wafer's,
plate's, ring's and screwdriver tip's positions. Run one trial per process. The score is `task_score` in the log,
computed by `score()` in `task.py`.

## Runtime and seeds

- 4,000 control steps (200 s of simulated time) per trial. The scripted demo finishes in about 170 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
