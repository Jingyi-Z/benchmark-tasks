# Clamp a glued cover plate onto a base plate with a spring clamp while holding it flush

An aluminium base plate lies on a printed gluing stand, its top glued, and a printed cover plate of the same size lies
on a mat beside it. A small sprung C-clamp hangs on the edge of a clamp rack. One arm sets the cover on the base with
its edges flush and presses it down; the other squeezes the clamp open by its grip and trigger, takes it off the rack,
slides the open jaws over the plates' front edge through a notch in the stand and lets go, so the spring closes the
jaws on both plates. Then the holding hand comes away. On the glue the cover slides easily on the base, which is
why it is held while the clamp goes on, and the clamp takes a hand of its own.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The clamp is one loose body with one labkit
  joint: a vertical slide carrying the upper jaw and the trigger. It is built where it starts, jaws 10 mm apart on the
  rack (slide at 0, drive target 0); from the first control step an `on_step` hook sets the slide's drive target to
  the closed jaws, so its 500 N/m spring presses them shut on whatever is between them.
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

Scripted demo: the same command with `--policy yam_scripted` (about 35 s of motion). The left hand grips the cover by
its long sides, lines it up on the base closed loop (every control step it reads both plates and trims the cover's
offset and yaw), sets it down, opens wide, and presses straight down on its back half with closed fingers, about 3 N,
leaned against the arm's compliance so the force is vertical. The right hand points forward and 10 degrees down, so
its fingers close square to the clamp's sloped grip and trigger; it squeezes them to the slide's stop, which opens the
jaws to 18 mm, draws the clamp off the rack and waits beside the stand until the cover is held. In front of the stack
it trims the clamp's measured position onto the stack's (closed loop), freezes the trim, slides the jaws in through
the notch, 3 mm clear above and below, and lets go: fingers wide open, a pause, then straight back and park. The left
hand lifts off once the right hand is away. Run one trial per process: the demo's live state is kept in the task
module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: whether the base plate
is still on its outline, whether the clamp's jaws grip both plates, and whether the cover is flush with the base.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
