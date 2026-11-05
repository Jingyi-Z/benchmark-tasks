# Pull a 1U chassis out on its slide rails by both release tabs

A short rack section on the bench holds a shallow 1U chassis on two slide rails, pushed home and latched. A blue release
tab stands on the handle block in front of each ear, 25 cm apart. Each rail's latch lets go only while its tab is pushed
6 mm in towards the middle, and over the first 30 mm of travel letting either tab go catches the chassis again. Both
arms push the two tabs in together and, still holding them in, draw the chassis out by them to its service position,
120 mm out, then let go.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The rack is one fixed labkit part with three joints: the chassis (a slide along x, 0 at home, its upper limit, to
    -120 mm at the service position, its lower limit) and the two tabs (slides along y riding on the chassis, 0 out to
    6 mm in, sprung out at 60 N/m, built released).
  - The latches are an `on_step` hook that sets the chassis's drive target only, every control step (`rail_target`).
    The drive is 1000 N/m. Inside the first 30 mm with either tab less than 4.5 mm in, the target is where the chassis
    was caught (it can still be pushed in, not drawn out). Otherwise it runs free: the target follows it, half a
    control step ahead so the spring adds no drag on average, with 0.4 N joint friction and 6 N s/m damping. Within
    3 mm of the service position the target is the service position (the detent).
  - The scripted demo drives both posts at once with the fingers part-open round them (pads 16 mm apart), as in the
    line card and tool changer tasks. Each hand's fingers close along a line 50 degrees round from straight at the
    robots towards the middle, so one pad pushes its post in (onto the tab's stop) and out towards the robots at once.
    The hands press for 1 s, hold for 1 s, then move out together 125 mm over 5 s, 5 mm past the stop. Because the
    arm is compliant, an `on_step` hook adds up to 8 mm more push along each hand's line while its tab is not fully in,
    or, after the hands stop, while the chassis is short of the service position. The fingers lean 20 degrees, wrists
    outward, to keep the elbows apart. Release: fingers to fully open, a 0.6 s pause, then lift and park.
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

Scripted demo: the same command with `--policy yam_scripted` (about 19 s of motion). Run one trial per process: the
latch state and the demo's live state are kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: how far the chassis is
out and how far each tab is still pushed in.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; the demo ends early about 8 s after the tabs spring out at
  the service position. About 3 minutes of wall time on one L40S with frames stored at 640 x 352 (estimated from
  similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
