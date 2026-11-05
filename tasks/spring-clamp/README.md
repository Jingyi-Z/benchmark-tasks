# Clamp a glued cover plate onto a base plate with a spring clamp while holding it flush

An aluminium base plate lies on a printed gluing stand, its top glued, and a printed cover plate of the same size lies
on a mat beside it. A small sprung C-clamp hangs on the edge of a clamp rack. One arm sets the cover on the base with
its edges flush and presses it down; the other squeezes the clamp open by its grip and trigger, takes it off the rack,
slides the open jaws over the plates' front edge through a notch in the stand and lets go, so the spring closes the
jaws on both plates. Then the holding hand comes away. The wet glue is modelled as low friction (0.08) between the
plates, and the clamp opens only 2 mm wider than the stack, so the soft pad under its upper jaw always rides up onto
the cover and slides along it as the jaws go on. That drags the cover off flush unless it is pressed down, which is why
it is held while the clamp goes on, and the clamp takes a hand of its own.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The clamp is one loose body with two labkit
  joints: a vertical slide carrying the upper jaw and the trigger (stop at 14 mm open), and on it a second slide, the
  foam pad under the upper jaw (2.5 mm proud when free, flat in the jaw when squashed; 120 N/m, no preload). Both are
  built where they start, jaws 10 mm apart on the rack with the pad squashed flat (both at 0, drive targets 0); from
  the first control step an `on_step` hook sets the slide's drive target to the closed jaws, so its 500 N/m spring
  presses them shut on whatever is between them, and the pad's to free, so it stands proud whenever nothing presses
  on it.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.
- `onehand_check.txt`: a sed that turns `task.py` into a one-hand variant of the demo (the holding hand never touches
  the cover), to check in simulation that it scores below 1.0.

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
jaws to 14 mm, draws the clamp off the rack and waits beside the stand until the cover is held. In front of the stack
it trims the clamp's measured position onto the stack's (closed loop), freezes the trim and slides the jaws in through
the notch, the lower jaw 1 mm under the base and the upper jaw 1 mm over the cover; the pad, 1.5 mm lower than the
cover's top, rides up onto it on its rounded nose and slides along it. Then it lets go: fingers wide open, a pause,
then straight back and park. The left hand lifts off once the right hand is away.

Why the hold is needed, in numbers: the pad presses on the cover with about 0.15 to 0.2 N as it slides (its spring in
series with the right arm's give) and drags it with 0.05 to 0.07 N (pad on cover 0.35). The wet glue alone holds the
18 g cover with only about 0.03 N (0.08 of its weight plus the pad's force), so an unheld cover goes along with the
jaws, about 18 mm, and the trial scores 0. Pressed down with 3 N, the glue holds it with about 0.27 N, four times the
drag and nearly twice the pad's peak push while its nose rides up the cover's edge (0.11 to 0.15 N), so it stays flush.
The left arm is too soft sideways (several mm per N) to hold the cover by its fingers' grip alone; the hold works by
pressing down. `onehand_check.txt` gives a sed that removes the hold from the demo, for checking this in simulation. Run one trial per process: the demo's live state is kept in the task
module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py` from the final state: whether the base plate
is still on its outline, whether the clamp's jaws grip both plates, and whether the cover is flush with the base.

## Runtime and seeds

- 1,200 control steps (60 s of simulated time) per trial; about 3 minutes of wall time on one L40S with frames stored
  at 640 x 352 (estimated from similar tasks; not yet measured).
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
