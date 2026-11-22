# Fit an adaptor bar across a countertop food warmer's well, then hang a quarter pan of macaroni and cheese on it

Two arms lift a loose stainless adaptor bar (531.0 x 25.4 x 8.0 mm) off the stand and seat it across the well of a
12 x 20 in full-size countertop food warmer (304.8 x 508.0 mm opening), ends on the end rims. Then they slide, square up
and pick one 1/4-size steam table pan (161.8 x 263.5 mm, 15 mm flanges) holding a 12 mm layer of macaroni and cheese
(0.28 kg, a rigid block with drawn pasta) from the stand and hang it in the far half of
the well, centred along it: one long flange on the bar, the other on the far long rim. The pan's 161.8 mm side cannot
span the 304.8 mm opening, so at this centred position it falls in without the bar (checked with the control run
below). Near either end of the well the pan's short flange can also catch an end rim, which is why the pan is seated
centred. The pan must stay level: tilting it more than 15 deg after it is lifted counts as spilling the food and
scores 0 for the rest of the trial. The pinch is aimed at the pan's measured pose after the slide.

## Run a trial

From this folder, with the yam_arena / inspect-robots environment active:

    inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
      --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
      --scorer task_score --max-steps 1200 --log-dir ./logs

- The scripted demo plan is 899 steps (44.95 s at 20 Hz). The run limit is 1,200 steps.
- Seeds: none. Fixed start layout, object_jitter 0.
- Control run without the bar: create an empty file `NO_BAR` in this folder (or export `SWBP_NO_BAR=1`).
- Scorer unit tests (numpy only): `python test_bar_pan_scorer.py`.

## Score

`task_score` is computed in `task.py` by `w50_bar_pan_scorer.py` (bar logic from `w50_baseline_bar_scorer.py` and
`w50_bar_scorer_v2.py`, pan seat check `pan_ok_bar_frame` from `w50_four_pan_waypoints.py`), checked at
20 Hz over the final 1.0 s. The numeric rule is in `task.json` "scoring".

## Files

- `task.py`: scene list, scripted demo (bar transfer, pan slide, square-up, two-hand far-channel seat, park), the pan
  guards, per-step scoring.
- `w50_bar1pan_scene.py`, `scene_w50.py`: stand, warmer, bar and pan geometry.
- `w50_bar_pan_scorer.py`, `w50_baseline_bar_scorer.py`, `w50_bar_scorer_v2.py`, `w50_four_pan_waypoints.py`: scorer.
- `test_bar_pan_scorer.py`: teleport unit tests.
- `w50_bar_waypoints.py`, `w50_two_hand_near_waypoints.py`, `w50_slide_landing_variants.py`,
  `w50_squareup_and_crossing_witness.py`, `w50_occupied_q2_and_far_isolated_audit.py`, `w50_multi_pan_isolated_runner.py`,
  `w50_live_scorer.py`: trajectory and geometry helpers.
- `labkit.py`, `arena_assets_fix.py`, `hand_cover_data.py`: shared helpers.
