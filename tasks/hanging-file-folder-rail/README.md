# Clamp a hanging file folder around a packet and hang it on frame rails

## Trial commands

Activate the Isaac Lab / `yam_arena` environment and run from the repository root:

```bash
# Scripted demonstration (900-step limit = 45 s at 20 Hz; about 14.3 s in the recorded final demo)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/hanging-file-folder-rail/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Hang the file folder on both top rails of the file frame with the document packet still inside it." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/hanging-file-demo

# Dream-1 baseline policy evaluation (900 steps = 45 s at 20 Hz)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/hanging-file-folder-rail/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Hang the file folder on both top rails of the file frame with the document packet still inside it." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/hanging-file-dream
```

Run the offline 289-hull (`578` dual-arm hulls) `YAM_URDF` trajectory, robot-vs-frame/folder/neighbor collision, OBB cross-section, and `169`-case rail-engagement scorer preflight check (including the static lug-tip counterexample and rear neighbor folder checks):

```bash
python tasks/hanging-file-folder-rail/check_path.py
```

## Runtime and seeds

- Control rate: `20 Hz` (`900` steps = `45 s` maximum; `286` steps = about `14.3 s` in the recorded final demo).
- Recorded final revision on the shared 5090: about `77 s` for the scripted demo and `250 s` for the `900`-step Dream-1 trial; queue time excluded and shared GPU load affects runtime.
- Deterministic fixed starting layout (`object_jitter=0`); no random seed configuration is required.

## Actual rail-engagement scoring, visual realism & real-trial inspection (`TK-SJ4K8R` revision)

- **Finite 3D rail-in-saddle containment (`task_metrics.py`):** Each of the target folder's four rod-end hooks (`front-left, rear-left, front-right, rear-right`) derives its `7.2 mm`-wide saddle void directly from its two physical `3.0 mm`-deep retaining lug boxes on `FRONT_LEAF_BOXES` and `REAR_LEAF_BOXES` rather than using a `6 mm` point-distance tolerance. `Metric.rail_in_saddle()` intersects the full `3.6 mm` width and upper `2.0 mm` of the `355.6 mm` rail with both finite faces (`zlo`, `zhi`) of each `3.0 mm`-thick hook and requires:
  1. Both rail side faces strictly inside the two retaining lugs (`side_margin >= 0`, zero sideways shortcut outside the saddle);
  2. The rail top bearing directly against the rod underside (`0.50 mm` separation / `0.10 mm` solver penetration simulation contact proxy); and
  3. All four hook intersections lying within the physical `355.6 mm` rail span (`x in [0.4250, 0.7806] m`) with the folder bottom crease suspended `>= 5.0 mm` above the desk, plus `>= 80 mm` six-point hand clearance and `< 1.0 mm` 21-sample stillness across the target folder, packet, and both loose rear filed folders.
- **Static lug-tip counterexample (`check_path.py`):** Shifting the hanging folder `+4.0 mm` in `Y` and `+4.49 mm` in `Z` places the rods geometrically on the inner lug tips atop the rails outside the saddle voids (`max_notch_dy = 4.0 mm <= 6.0 mm`, `max_notch_dz = 4.49 mm <= 4.50 mm`, `crease_clear = 12.49 mm`). That static geometric configuration scored `1.0` under the prior tolerance check and now scores **`0.0`** (`rail_engaged = [False, False, False, False]`, `min_saddle_side_margin_mm = -2.20 mm`). `check_path.py` verifies `140` valid in-saddle placements (`1.0`), `11` out-of-saddle/lifted/penetrating placements (`0.0` including the static lug-tip witness), and `18` structural, clearance, and rear-neighbor cases (confirming a harmless `+15 mm` slide of a rear folder along the rails still scores `1.0`).
- **Real-trial verification & retail sourcing:**
  - After `1.0 s` synchronised overhead + side video stillness and `>= 80 mm` tape-measure hand clearance from all folders, packet, frame, and oak block, inspect and photograph both ends of the left and right rails square-on from the side (`4` hook views) plus a ruler check under the bottom crease and packet (`>= 5 mm` above the desk): every target hook's inner and outer retaining lugs must straddle opposite sides of the rail blade with the rod underside seated on the rail top; any hook resting on a lug tip or rod outside its saddle scores `0.0`.
  - Low-cost retail build; listed frame, folder box, manila-folder pack and a USD 5 wood-block allowance total about USD 46 to 49 before paper, separate index tabs/inserts, clamps, tax and shipping: `Officemate 91990` (or single-pack `91992`) letter-size steel hanging file frame set to `14 in` (`355.6 mm`) depth (`~USD 14 to 17` single/2-pack retail via standard office-supply retailers; steel construction and `14 to 18 in` breakaway rail adjustment documented at https://victortech.com/products/twin-pack-hanging-file-frame , which itself is a wholesale 12-pack listing), `Pendaflex 4152` reinforced green hanging folders, plus separately purchased compatible `1/5`-cut index tabs and inserts (`~USD 18` per box of 25 folders — supplies both the target folder and the two rear pre-hung folders at `x = 0.635 m` and `0.710 m`; https://www.tops-products.com/pendaflexr-reinforced-hanging-folders-letter-size-standard-green.html ), `Smead 10330` letter-size `1/3`-cut manila folders (`~USD 9` pack) trimmed with ~`60` copy-paper sheets to the `280 x 195 x 6.0 mm` interior stack plus protruding `62 x 180 x 4.0 mm` header and `12 x 88 x 2.0 mm` tab (`200 g` estimated total packet), and a `200 x 130 x 32 mm` natural oak wood pickup block clamped at `(0.1750, 0.0000) m` (`~USD 5` allowance). Rail blade thickness (`3.6 mm` inside `7.2 mm` saddle), `2.5 mm` rigid leaves, `9.2 mm` bottom V-gusset crease, hinge damping (`0.02`), and friction (`0.45-0.58`) are disclosed engineering estimates; open-loop scripted demo uses zero simulator pose feedback and nothing is moved or latched by code.

## Occupation & specification references

- O*NET `43-9061.00` (Office Clerks, General — filing and maintaining records): https://www.onetonline.org/link/summary/43-9061.00
- O*NET `43-4071.00` (File Clerks — placing materials into storage receptacles): https://www.onetonline.org/link/summary/43-4071.00
- US Bureau of Labor Statistics May 2024 OEWS national employment (`43-9061` Office Clerks, General: `2,510,550` occupation total): https://www.bls.gov/oes/2024/may/area_emp_chart/area_emp_chart.htm

## License

Robot, station table, and `yam_arena` utilities derive from `yam-isaac-lab-arena` (`Apache-2.0` / `BSD-3-Clause`). Task scene and scoring code in this folder are released under `Apache-2.0`.
