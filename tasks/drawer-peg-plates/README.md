# Make room for a dinner plate in a pegboard dish drawer

A bimanual YAM pair reconfigures a pegboard dish drawer for a larger plate. It pulls the two ring pegs nearest the
countertop, re-seats each one hole toward the counter, then lifts a 27 cm ceramic dinner plate off the counter
with a two-hand under-rim pinch and sets it flat inside the enlarged ring.

## Files

| File | Description |
|---|---|
| `task.py` | Scene (drawer, pegboard insert, nine pegs, countertop, plate), on-step logging and scorer wiring (`score()`), and the scripted `demo()` |
| `drawer_plates_score_b.py` | Pure end-state scorer over the final 21 frames (pegs, plate placement, release, stillness) |
| `drawer_plates_live.py` | Sticky drop and hard-contact latches, read from plate poses every control step |
| `hand_cover_data.py` | Certified 424-point cover of each gripper and its fingers, used for the 80 mm hand clearance |
| `labkit.py`, `arena_assets_fix.py` | Primitive-part scene builder and asset path fix |
| `ironing_floor_bay.py`, `pyproject.toml` | Embodiment `yam_floor_bay`: the calibrated bimanual YAM station on steel pedestals over a floor bay (no table), so a real-height kitchen cabinet fits in front of the arms |
| `plates_score_witnesses_b.py`, `live_latch_witnesses.py`, `full_run_check_b.py` | Offline scorer and latch tests, and a fail-closed reader that re-checks a finished run's log against the scorer |

## Install and run

Install the floor-bay embodiment once into the `yam_arena` environment:

```bash
pip install -e .
inspect-robots list
```

Scripted demo (score 1.0; 1,194 planned steps, about 4 to 6 min wall time on an RTX 5090 with frames, seed 0,
`object_jitter=0`). `yam_scripted` plays `task.py`'s `demo()`; with a task file it does not read the instruction.

```bash
inspect-robots run --embodiment yam_floor_bay --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Move the two ring pegs nearest the countertop one hole toward it and place the dinner plate flat inside the enlarged ring." \
  --scorer task_score --max-steps 1210 --store-frames --log-dir runs/demo
```

Dream-1 (2,400 steps = 2 min at 20 Hz, about 11 min wall time):

```bash
inspect-robots run --embodiment yam_floor_bay --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Move the two ring pegs nearest the countertop one hole toward it and place the dinner plate flat inside the enlarged ring." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Offline tests (no GPU): `python plates_score_witnesses_b.py` (20 cases, every score level 1.0 / 0.5 / 0.25 / 0
reached) and `python live_latch_witnesses.py` (22 cases). After a run, `python full_run_check_b.py RUN_DIR 1` re-reads
`run.log` and prints FULL 1.0, PEGS-ONLY 0.5, HAZARD 0 or FAIL.

## Scene (positions in mm; x forward from the arm-base line, y to the left, z up from the floor)

- Drawer: the deep drawer of a 33 in base cabinet, pulled fully out and fixed. Maple box 203.2 mm tall, 15.9 mm
  sides, 12.7 mm bottom, top edge z 700; interior x 106.5 to 614.5, y -580 to 130.
- Pegboard insert trimmed to fill the interior (stock route below): 5/8 in (15.9 mm) board, board top z 525.4,
  27.0 mm holes on the 2-1/2 in (63.5 mm) grid at x = 169.8 + 63.5 i (i = 0..6), y = -517.5 + 63.5 j (j = 0..8).
- Nine birch dowel pegs, 25.4 mm across, 152.4 mm above the board, 51 g. Four-peg ring at (169.8, -327.0),
  (360.3, -327.0), (169.8, -136.5), (360.3, -136.5); five spares at x 487.3, y -263.5, -200.0, -136.5, -73.0, -9.5.
- Stone countertop beside the drawer: top z 750, x 120 to 440, y 165 to 530.
- One 27 cm glazed ceramic dinner plate (270 mm across, 24 mm tall rim, 0.55 kg), flat on the countertop at (265, 320).
- Goal: the two counter-side ring pegs at (169.8, -73.0) and (360.3, -73.0), and the plate centred on the ring
  centre (265, -200). The plate cannot reach that spot while those pegs stand at y -136.5.

## Scoring

Judged over the final 1.0 s (21 frames at 20 Hz):

- Score 0 if, at any time, a plate drop or hard contact was latched: the plate fell off the counter, or tipped over 45
  deg while released, or reached a surface or a wall faster than 0.25 m/s.
- Score 0 if, in any final frame, a hand (certified gripper and finger cover) is within 80 mm of the plate or a peg,
  or anything moves 5 mm or more.
- Score 0 if any peg is not seated upright (foot within 3 mm of the drawer floor, tilt 8 deg or less, within 1.3 mm
  of a hole centre) in its own start hole or its target hole.
- Otherwise, 0.25 for each moved peg seated in its target hole.
- Plus 0.5 if the plate lies flat (5 deg or less) on the board (within 3 mm), centred within 15 mm of the ring
  centre, at least 2 mm from every wall and clear of every peg, in every frame.

Full score is 1.0.

## Results

| Run | Score | Notes |
|---|---|---|
| Scripted demo | 1.0 | Recorded with the prompt "Make room for the dinner plates." (yam_scripted plays demo() and does not read it). Pegs within 0.69 mm of their holes; plate lifted at 0.71 deg tilt, placed 1.7 mm from the ring centre, flat; no latch |
| Dream-1 | 0.0 | Neither peg moved; the plate was nudged 3 mm on the counter, not placed; no latch |

## Real-world trial

- Stock route: Rev-A-Shelf 4DPS-3021 drawer peg system (maple board, 12 pegs; factory board 768.35 x 539.75 mm), sold as
  Rockler item 1001907 at USD 180.99 (reviewed product page:
  https://www.rockler.com/30-1-4-wide-drawer-peg-system-with-12-pegs-natural-wood-maple-4dps-3021 ; maker install sheet:
  https://s1.img-b.com/build.com/mediabase/specifications/rev-a-shelf/225384/rev-a-shelf-4dps-3021-installation.pdf).
  Trim it to the 710 x 508 mm interior per the maker's measure-trim-insert steps and use nine of its twelve pegs.
  Measure its drilled grid, peg dimensions and plate fit before use; the guide gives no evidence that every dimension
  equals the simulation. The 5/8 in board and 2-1/2 in (63.5 mm) pitch are from QuikDrawers item 26696
  (https://quikdrawers.com/peg-board-drawer-organizer-with-9-pegs, from USD 134.75); its factory width is 622 mm, narrower
  than this 710 mm drawer, so it is the source for the pitch, not the trial board. The 1 in peg diameter is modeled.
- Plate: any 27 cm (10.5 in) ceramic dinner plate.
- Reset: re-seat all nine pegs in their listed start holes, upright, and put the plate back flat at its counter spot.
- Marks and measurement: mark each moved peg's foot and top and the plate's centre and rim. In a photo from above at the
  final second, check each moved peg sits fully down and upright in its target hole, the plate lies flat with its centre
  within 15 mm of the ring centre and touches no wall, and both hands are at least 80 mm from the plate and the pegs
  (tape measure). Any drop, tip-over or slammed plate on the video scores 0.
- Possible scores: 0, 0.25, 0.5, 1.0. 0.75 is unreachable: an unmoved ring peg stands 114.5 mm from the ring centre, at most 130.8 mm with the 15 mm centre and 1.3 mm hole tolerances, inside the 150.7 mm plate-plus-peg clearance the scorer requires.

## Fidelity and estimates

- The rigid insert uses 27 mm square openings with clipped corners at the six ring/target holes and five other openings
  (four spare-peg starts and the unused x 487.3, y -327 opening); the other openings remain square.
- The drawer, peg diameters and masses, and grid registration are nominal.
- The 16-sided plate and pose-based hazard scoring approximate the real objects and impacts.
- Closest board tasks: TK-DA2CEU (push a single peg home); a title-only PR scan also found #460 (a fixed drying rack, no
  reconfiguration) and #2713 (a wall hook). This task reconfigures the organizer before loading it.
