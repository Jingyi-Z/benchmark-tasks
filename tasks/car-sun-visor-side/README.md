# Swing a car's sun visor off its clip to the side window and slide it out to shade the glass

A sun visor hangs down over the windshield, its rod's tip held in the centre clip. The robot has to free it from the
clip, swing it through about 90 degrees to the side window and slide its extender out so it covers a marked band on the
side glass, then let go and move clear.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
- `labkit.py`: multi-shape parts with hinged and sliding joints (from an accepted task), plus absolute joint-axis
  friction (`AXIS_FRICTION`) so the visor holds where it is left.
- `arena_assets_fix.py`: asset path fix used by labkit.

## Run

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "<see task.json>" --scorer task_score --max-steps 950 --store-frames --log-dir logs

    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "<see task.json>" --scorer task_score --max-steps 2400 --store-frames --log-dir logs

One trial takes about 4 minutes on one GPU; no seeds (the layout is fixed). The scripted demo: the right hand's shut
fingertips press the visor's face 30 mm below the rod (low pushes roll the board up), pushing it off the clip and round
to the side; the left hand then pushes its outer end along the rod, aimed at the swing angle it is seen at, to slide the
extender out about 100 mm. `VISOR_DEBUG=1` prints the joint positions, coverage and hand clearance every second.

## Scoring

`task_score`, from `score()` in `task.py` (see task.json for the rule). Coverage is the share of the band's area inside
the visor's outline projected square-on to the glass (the hull of its eight corners). Swung over without the extender it
covers about 61% (0.5); slid out 50 mm or more, over 80% (1.0).

## Known gaps

- The roof and headliner are left out so the arms can reach; the visor's pivot mount and the centre clip are fixed in
  space where the headliner would hold them, and the corner sits lower than a car's (rod 220 mm above the bench).
- Joint friction (0.6 N m swivel and roll, 4 N slide), the clip's jaw spring (0.05 N m/rad) and the visor's size are
  estimates, not measured parts.
- An offline replay of the demo through IK shows the left arm lagging its targets by up to 122 mm while it moves behind
  the visor's end and slides it; on the GPU it reaches and slides the extender about 98 mm.
- Hand clearance uses each hand's two fingertip points and their average as the midpoint, not a separately measured
  pad centre.
