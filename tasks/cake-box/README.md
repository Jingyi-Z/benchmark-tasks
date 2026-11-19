# Box a decorated cake for a customer without touching the frosting

## Trial commands

Activate the Isaac Lab / `yam_arena` environment and run from the repository root:

```bash
# Scripted demonstration (1,200-step limit = 60 s at 20 Hz; reaches 1.0)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/cake-box/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Box the decorated cake for the customer: slide it into the cake box on its board, close the box and tuck the lid's tab in, without touching the frosting." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir logs/cake-box-demo

# Dream-1 baseline policy evaluation (1,200 steps = 60 s at 20 Hz; reaches 0.0)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/cake-box/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Box the decorated cake for the customer: slide it into the cake box on its board, close the box and tuck the lid's tab in, without touching the frosting." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir logs/cake-box-dream
```

Scorer diagnostic controls (`CAKE_BOX_MODE=untouched` -> `0.0`, `CAKE_BOX_MODE=touch_frosting` -> `0.0`, `CAKE_BOX_MODE=near_hand` -> `0.0`, `CAKE_BOX_MODE=lid_no_tuck` -> `0.75`) and the offline `630`-hull `YAM_URDF` trajectory, dense frosting-contact (`<= 3.2 mm` surface spacing), `80.0 mm` release-gate, analytic circle projection, and `25`-case scorer witness preflight check:

```bash
python tasks/cake-box/check_path.py
python tasks/cake-box/fab/generate_templates.py
```

## Runtime and seeds

- Control rate: `20 Hz` (`1,200` steps = `60 s` maximum).
- Worst-case demo timing arithmetic: Move 1a stage slide `10.8 s` + Move 1b box entry `11.2 s` + snapshot/right clamp `2.85 s` + Stage 2A/2B front-panel fold `10.55 s` + lid lower/park `5.70 s` + circular-arc tab crease `4.45 s` + guarded lid seat & lift `4.80 s` + retreat to `>= 80.0 mm` park & `1.0 s` (`21`-sample) released stillness `~2.05 s` = **`~52.4 s` (`1,048` steps)**, leaving **`12.7%` (`7.6 s` / `152` steps) headroom** under the `1,200`-step (`60.0 s`) limit.
- Approximate wall-clock runtime on a single RTX 5090 (`640 x 352` with `--store-frames`): `~4 minutes` for the scripted demonstration and `~3 minutes` for `1,200` steps of `Dream-1`.
- Deterministic fixed starting layout (`object_jitter=0`); no random seed configuration is required.

## Real products, shipped fabrication templates & assembly (`tasks/cake-box/fab/`)

All live product URLs and retail prices below were checked in Nov 2026:

1. **Bakery window carton + declared front-panel modification (`fab/front_lifting_ear_template.svg`, `fab/templates.dxf`):**
   - **Base commercial carton:** `BRP Box Shop #245` `10 x 10 x 5 in` white `24-pt` (`0.6 mm`) SBS lock-and-tab window bakery box (`USD 29.36` per 10-pack = `USD 2.94`/carton, or `USD 117.42` per 100-pack = `USD 1.17`/carton: https://www.brpboxshop.com/245.html ; family equivalent `Southern Champion Tray #23053`: https://www.sctray.com/en/catalog/product/23053/ ; `254.0 x 254.0 x 127.0 mm` inner cavity, `174 x 174 mm` clear window, `35.0 mm` tapered front tuck tab, `~146 g` estimated total mass including lifting ear).
   - **Declared preparation & `+Y` lifting ear:** Hook the two rear (`+X`) corner lock tabs into the rear side-wall slots and secure the outer rear vertical corners with two `19 x 60 mm` strips of clear Scotch tape (`~USD 3`/roll) so `back_wall`, `left_wall`, and `right_wall` remain rigid at `90 deg`. Unhook both front (`-X`) lock tabs and slit the two front `45 deg` diagonal corner gusset webs flush along the front vertical edges of `left_wall` and `right_wall` so `front_panel` (`251.0 x 126.0 mm`) unfolds `-90 deg` flat onto the counter and seats against the side walls at `0 deg`. From `24-pt` (`0.6 mm`) white SBS cardstock (or the removed gusset scrap), cut the `116.0 x 36.0 mm` blank in `fab/front_lifting_ear_template.svg` (`CUT`/`SCORE` layers in `fab/templates.dxf`), fold `180 deg` and glue flat into a `58.0 x 36.0 x 1.2-1.6 mm` two-ply strip (`~1.3 g`), bend `90 deg` at the `20.0 mm` score line (`20.0 x 36.0 mm` flange + `38.0 x 36.0 mm` ear), and adhere the flange with `19 mm` double-sided tape (`~USD 5`/roll) onto `front_panel`'s outer face at its `+Y` top corner (`z = 88.0..124.0 mm`, placing the `38.0 x 36.0 mm` perpendicular ear at `y_box = +130.4 mm`, `3.4 mm` outside the `254.0 mm` throat, standing `39 mm` upright off the counter when `front_panel` is at `-90 deg`). When `front_panel` folds to `0 deg`, closing the `257.2 x 257.2 mm` lid onto the walls and tucking the `35.0 mm` front tab `>= 10.0 mm` behind `front_panel`'s inner face traps `front_panel` shut between the side walls and the tucked tab.

2. **Fabricated `247.6 x 4.8 mm` (`~55 g`) two-ply silver-foil cake board (`fab/cake_board_template.svg`, `fab/templates.dxf`):**
   - **Stock materials (`distinguishing nominal stock from finished stand-in`):** Two `Wilton` `10 in` nominal corrugated white cake circles (`MFR 191001569`, `USD 11.49` per 12-pack at WebstaurantStore `#143191001569`: https://www.webstaurantstore.com/wilton-10-corrugated-white-cake-circle-pack/143191001569.html ; `~254 mm` stock OD, single-wall B-flute `~2.4 mm`/ply), PVA craft glue (`~USD 4`), and `0.018 mm` household silver aluminum foil (`Reynolds Wrap`, `~USD 5`/roll).
   - **Assembly & pre-trial dimensional check:** Laminate two stock circles at `90 deg` crossed flute directions with thin PVA glue and press flat (`4.8 +/- 0.4 mm` assembled thickness; shim with kraft cardstock if caliper reads `< 4.6 mm`), wrap the top face smoothly with silver foil (`leaving the flat kraft underside exposed, mu ~ 0.20`), trim around `fab/cake_board_template.svg` (`print at 100% and verify the 100.0 mm scale bar`) to **`247.6 +/- 0.5 mm` finished OD** (`leaving 3.2 mm nominal clearance per side inside the 254.0 mm carton`), and lightly chamfer/sand the bottom kraft perimeter `3.0 mm` radially x `1.2 mm` high so the board glides cleanly onto the `1.0 mm` carton floor. Before use, verify OD (`247.6 +/- 0.5 mm`, `1 mm` steel ruler), thickness (`4.8 +/- 0.4 mm`, Vernier caliper), and mass (`55 +/- 6 g`, `1 g` digital scale; area `0.04815 m^2` x two `~520 g/m^2` plies + `~4 g` glue/foil = `~54 g`).

3. **Reusable cold decorated stand-in cake (`~300 g` estimated mass):**
   - `NY Cake` `8 x 3 in` (`203.2 x 76.2 mm`) round polystyrene foam cake dummy (`USD 7.99`: https://nycake.com/products/round-styrofoam-cake-dummy ) covered in `2.4 mm` rolled pastel blush fondant (`~USD 8` tub, `208.0 mm` OD, `78.6 mm` height), adhered centered on the foil board (`208.0 mm` guide ring in `fab/cake_board_template.svg`) with hardened royal icing (`~USD 6` meringue/icing mix), and piped with hardened snow-white royal-icing bottom and top shell borders (`217.0 mm` max OD = `108.5 mm` radius, leaving a **`15.3 mm` exposed silver pushing rim** all around) and eight top swirl rosettes (`93.5 mm` peak height, `28.7 mm` clear below the closed lid).

4. **Closure cross-section, force paths & collider relief (`fab/closure_cross_section.svg`):**
   - `fab/closure_cross_section.svg` dimensions the XZ closure path (`FRONT_ANCHOR (-127.4, 0, 0.5) mm`, `front_panel` `0.8 mm` rigid proxy & `126.0 mm` height, `127.0 mm` side walls, `LID_ANCHOR (+127.8, 0, 128.0) mm`, `TAB_ANCHOR (-124.5, 0, 128.0) mm`, and the `35.0 mm` visual `lid_tab` with rigid `tab_boxes` collider on `r = 14.5..35.0 mm` and `0..14.5 mm` hinge-root relief representing local paperboard flex near the front lid score line so an uncreased outward tab passes outside `front_panel` for `0.75` while an inward-creased tab tucks `>= 10.0 mm` behind `front_panel`'s inner face for `1.0`). All crease friction/hysteresis (`0.008-0.085 N*m`) and tab crease spring (`0.001 N*m/rad`) values are engineering estimates.

5. **Checkout vs per-trial consumed cost, witness containment & real-trial procedure:**
   - **Initial checkout cost:** `~USD 79` (`BRP #245` 10-pack `USD 29.36` + `Wilton MFR 191001569` 12-pack `USD 11.49` + `NY Cake` `8 x 3 in` dummy `USD 7.99` + fondant/royal-icing/PVA/foil/tapes `~USD 26` + food-grade unsweetened cocoa powder `~USD 5`, plus reusable `1 mm` steel ruler, `80 mm` gauge block, `30 deg` wedge, caliper, and `1 g` scale). **Per-trial consumed cost:** `~USD 0.05` of cocoa powder when reusing an uncrushed carton (`or ~USD 2.99` when swapping in a fresh `BRP #245` carton from the 10-pack).
   - **Safety & containment:** Cold room-temperature stand-in props only on a wipeable food-safe mat; wear nitrile gloves (and safety glasses when cutting templates). Apply food-grade unsweetened cocoa powder (`Hershey's 100% Cocoa`) **off-table over a catch tray** with a soft brush onto removable nitrile/silicone gripper-pad sleeves and the lid/tab underside - never dust loose powder near robot joints, and wipe the mat and sleeves with a damp microfiber cloth after each trial.
   - **Prospective real-trial verification procedure:** Record synchronised `30 fps` overhead and side video (`checking contact, drops, and 1.0 s released stillness`); inspect white borders, rosettes, and fondant under a low-angle raking lamp (`any brown cocoa transfer or dent scores 0`); measure `<= 1.0 mm` board overhang past the `254.0 x 254.0 mm` inner base lines, `<= 3.0 mm` lid-to-wall gap all round, `>= 10.0 mm` tab tuck depth behind `front_panel`'s inner face, and `>= 80.0 mm` six-point hand clearance with a `1 mm` steel ruler and `80 mm` gauge block; then lift the closed carton `50 mm` by its side walls and tilt `30 +/- 2 deg` forward against a `30 deg` wedge for `3 s` to verify `front_panel` and `lid_tab` remain tucked.

## Occupation references

- O*NET `35-2021.00` (Food Preparation Workers, *"Package take-out foods or serve food to customers"*): https://www.onetonline.org/link/summary/35-2021.00
- O*NET `51-3011.00` (Bakers): https://www.onetonline.org/link/summary/51-3011.00
- US Bureau of Labor Statistics May 2024 OEWS occupation totals (`35-2021` Food Preparation Workers: `888,770` total US jobs across the occupation; `51-3011` Bakers: `231,890` total US jobs across the occupation): https://www.bls.gov/oes/2024/may/area_emp_chart/area_emp_chart.htm and https://www.bls.gov/news.release/archives/ocwage_04022025.htm
- Closest accepted look-alike: `TK-8A2A64` (`PR #2045`, crimp a foil pan rim onto a flat board lid).

## License

Robot, station table, and `yam_arena` utilities derive from `yam-isaac-lab-arena` (`Apache-2.0` / `BSD-3-Clause`). Task scene, fabrication templates, and scoring code in this folder are released under `Apache-2.0`.
