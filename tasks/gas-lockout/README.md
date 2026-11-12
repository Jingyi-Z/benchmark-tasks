# Lock out a gas line with a padlock and bleed it to zero

A yellow gas line on the bench has a quarter-turn ball valve, open, with a yellow lockout tab under where its handle
lies when shut. Downstream are a dial gauge reading the line's 200 kPa and a small bleed valve with a vent. An open
safety padlock with a danger tag stands on a block.

The task follows a lockout procedure:

- Shut the isolating valve: its handle a quarter turn, across the pipe, against its stop.
- Padlock it shut. The padlock's shackle is raised: the handle and the tab slide in between its short leg and its
  body, and pressing the shackle down puts the short leg through both holes into the body, where it latches. One hand
  holds the lock while the other presses.
- Release the gas trapped downstream through the bleed valve, and check that the gauge reads zero.

Opening the bleed while the isolating valve is still open vents the live supply, and scores 0.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The gas line with its two valves, the gauge and the
  tab is one fixed part; the padlock is loose, its shackle on a sprung slide. An `on_step` hook models the gas (the
  gauge's needle follows it), latches the padlock once it is pressed shut, records whether the bleed was opened on the
  live line or after the lock, and checks that everything has been still for the last second.
- `labkit.py`: parts built from several shapes. It hooks `yam_arena.scene` and the embodiment from the task's import,
  so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The left hand shuts the valve by its sleeve; the right
takes the padlock by its broad faces and slides the handle and tab into it beside the valve; the left presses the
shackle shut, then opens the bleed valve and holds it while the gauge falls to zero. Run one trial per process. The
score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (120 s of simulated time) per trial. The scripted demo takes about 800.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
