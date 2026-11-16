# Load a reusable coffee filter into a drip coffee maker's swivel basket and fold its bail flat

A 12-cup drip coffee maker (327.9 x 201.9 x 353.1 mm, 3.65 kg with water in its rear reservoir, loose on the counter)
has a front swivel filter-basket holder. A reusable basket filter (109.2 mm top, 84.0 mm base, 53.3 mm tall) with a
semicircular wire bail hinged into side rim pivot ears stands beside it. The robot steadies the brewer from above,
opens the holder, seats the filter in it, folds the bail flat inside the rim and swings the holder shut. The brew head
sits 12.0 mm above the seated rim, so a bail left upright blocks the holder ajar.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file); `task_metrics.py`: the end-state checks.
- `labkit.py`, `arena_assets_fix.py`: multi-shape parts with hinged and sliding joints, and the asset path fix.
- `check_path.py`, `hand_geometry.py`: offline checks (collision hulls along the scripted path, bail-to-head clearance
  sweep, scorer counterexamples).

## Run

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20         --instruction "<see task.json>" --scorer task_score --max-steps 960 --store-frames --log-dir logs

    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20         --instruction "<see task.json>" --scorer task_score --max-steps 960 --store-frames --log-dir logs

One trial takes about 3 to 4 minutes on one GPU; the layout is fixed (no seeds). `COFFEE_FILTER_MODE=unseated_close`
(holder closed with no filter seated, scores 0.0) and `COFFEE_FILTER_MODE=upright_bail_close` (filter seated, bail
left upright, the holder blocked ajar by the brew head, 0.5) run the two negative controls with the scripted policy.
`python check_path.py` runs the offline checks.

## Scoring

`task_score`, from `score()` (see task.json). Full credit needs the brewer upright (within 8 deg), the filter seated
in the holder, the bail folded within 14.0 deg of flat and 10.2 mm of the rim, the holder closed within 5.0 deg, both
hands' fingertips and their midpoints 80 mm clear, and everything still for 1 s.

## Known gaps

- The bail and closed-holder gates were set to the measured physical rest angles (bail 12.49 deg and 8.83 mm above the
  rim, holder 4.04 deg); 3.17 mm of head clearance remains with the bail folded.
- Internal dimensions (basket 117 mm inside, 12 mm brew-head gap, 3.5 mm bail recess), masses and hinge frictions are
  estimates from published outer sizes; the tapered filter wall is chordal boxes.
- Each run logs two PhysX friction-combine-order errors; the authored friction pairs are valid, but the friction the
  runtime applies is not independently verified.
