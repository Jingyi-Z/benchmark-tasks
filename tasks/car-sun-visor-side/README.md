# Swing a car's sun visor off its clip to the side window and slide it out to shade the glass

Mounted into the overhead molded headliner of a driver-side front car-roof corner mock-up (`330 mm` rod height above the bench), an extending automotive sun visor hangs down over the windshield aperture with its steel rod tip held in a sprung centre retainer clip. Reaching underneath the overhead headliner from the cabin/front side, the robot frees the rod from the clip, swings the visor ~`86 degrees` across beneath the headliner to the driver's side window, slides the blade out along its L-rod so its square-on outline covers at least `80%` of a `300 x 120 mm` orange tape band on the window, and retreats at least `80 mm` clear.

## Files

- `task.py`: scene (with overhead roof/headliner canopy, side cantrail, 3 bench posts, 4-edge EPDM rubber weatherstrip window frame, sprung centre clip, and 3-DOF sliding visor), scorer, and under-headliner scripted demo (`VISOR_MODE=demo|swing_no_slide|unclipped_partial`).
- `generate_cad_stl.py`: zero-dependency script generating six watertight binary `.stl` 3D-printable parts (`pivot_mount.stl`, `visor_blade_full.stl`, `visor_blade_outer_half.stl`, `visor_blade_inner_half.stl`, `clip_base_front_jaw.stl`, `clip_rear_sprung_jaw.stl`) in exact `task.py` local part frames (`75.0 mm` off-centre rod bore verified to achieve `93.44%` demo target-band coverage).
- `check_visor.py`: offline 20 Hz IK and 578-hull robot/roof/headliner/visor collision verifier.
- `labkit.py`, `arena_assets_fix.py`: articulated USD rigid-body builder with absolute joint-axis friction (`AXIS_FRICTION`).

## Real-Trial Hardware Options, Roof/Headliner Placement, and Safety

Physical evaluation can use **either** commercial Honda Accord parts **or** the included exact-frame 1:1 printable STL assembly on a `2020` T-slot aluminum extrusion bench frame (`~USD 35` extrusions + gussets, clamped at pivot world origin `x = 0.34 m, y = +0.22 m` rotated `180 deg` around `Z`):

1. **Option A: Commercial `2013-2017 Honda Accord EX` extending visor & clip (`< USD 150` total):**
   - **Sliding sun visor:** `2013-2017 Honda Accord EX / EX-L / Touring` Left (Driver) Slide-on-Rod Sun Visor Assembly (`Honda OEM P/N 83280-T2F-A52ZB` Light Gray or `83280-T2F-A52ZA` Ivory Beige, superseding `83280-T2F-A51ZB`: `~USD 32 to 45` direct-fit aftermarket replacement, `~USD 95 to 110` salvaged/used OEM assembly at Go-Parts / eBay Motors, or `~USD 308` new Honda dealer list; leave vanity-mirror wire unplugged/trimmed). Its padded blade (`~410 to 430 x 170 mm`) swivels on its outer post, rolls on its `8 mm` steel L-rod, and slides axially along the rod to extend side-window shade.
   - **Matching retainer clip:** `Honda OEM Sun Visor Holder Clip P/N 88217-TR0-A01ZL` (`Light Gray`, Accord-listed) or `88217-TR0-A01ZA` (`~USD 10`).
2. **Option B: exact-frame 1:1 printable STL + standard hardware (`python generate_cad_stl.py`):**
   - **Visor blade with `75.0 mm` off-centre rod bore (`PANEL_C = (-75, 227, 0) mm`):** Print either `visor_blade_full.stl` (`430 x 170 x 25 mm`, `x in [-160.0, +10.0] mm` around the `(x = 0, z = 0)` `8.2 mm` Y rod bore, `y in [+12.0, +442.0] mm`, `z in [-12.5, +12.5] mm`) on a large-format bed **or** print `visor_blade_outer_half.stl` (`y in [12, 227] mm`) + `visor_blade_inner_half.stl` (`y in [227, 442] mm`) upright on any `220 x 220 x 250 mm` desktop printer (`0.35 kg` total PETG/ABS mass with infill) and join at `y = 227.0 mm` with two `M3 x 30 mm` steel dowel pins in the coaxial `3.1 mm` bores at `(x = -60, z = 0)` and `(x = -125, z = 0) mm` plus adhesive. Slide onto an `8.0 mm` OD steel rod bent `90 deg` (`462 mm` horizontal +Y arm at `z = 0` + `40 mm` vertical +Z pivot leg at `(0, 0)`, `~USD 14`) lined with thin PTFE/felt tape to hold roll and slide (`~0.6 N m` / `~4 N`), and mount the vertical leg in `pivot_mount.stl` (`bolted into the outer headliner boss at frame (0, 0, +20..+40) mm`). Roll the blade `-90 deg` about `+Y` at reset so `x in [-160, +10] mm` maps to `z in [-160, +10] mm` hanging straight down (`covering 93.44% of the [-165, -45] mm window target band at the demo pose`).
   - **Sprung centre clip (`frame (0, +454.0, 0) mm`):** Bolt `clip_base_front_jaw.stl` (`upper block z in [+12, +28] mm`) to the underside of the `+28 mm` centre headliner boss (`local (0, 0, 0)` aligned to frame `(0, +454.0, 0) mm`). Seat `clip_rear_sprung_jaw.stl` between the two outboard M3 clevis ears of `clip_base_front_jaw.stl` with a `90/180 deg` stainless torsion spring (any 90 deg torsion spring of this size, `4.27 mm` OD, `3.05 mm` ID, `~0.05 N m/rad`, `~USD 7/pack`) inside the `4.6 mm` central spring slot, and press an `M3 x 25 mm` dowel pin through the coaxial `3.1 / 3.2 mm` Y bores at `(x = -7.5, z = +12.0) mm` (`one spring leg bears against the z = +12 mm clip-base underside as the 0 rad closed stop, the other inside the rear jaw`).
3. **Overhead roof, headliner, and protected window placement (`frame` coordinates relative to pivot `(0.34, +0.22, TOP + 0.330)` m, `YAW = 180 deg`):**
   - **Three bench posts (`BENCH = -0.330 m` to `z = +0.044 m`):** Front A-pillar at `(-0.065, -0.145) m`, rear B-pillar at `(-0.650, -0.145) m`, and right header stanchion at `(-0.125, +0.515) m` (`30 x 30 mm` `2020` extrusion).
   - **Overhead windshield header & molded headliner canopy:** `176 x 690 mm` (`x in [-0.156, +0.020] m, y in [-0.160, +0.530] m`), `12 mm` foam-backed headliner board (`z in [+0.044, +0.056] m`) under an `8 mm` outer roof panel (`z in [+0.056, +0.064] m`), with two `56 x 72 x 16 mm` mounting bosses (`z in [+0.028, +0.044] m`) over the pivot `(0, 0)` and centre clip `(0, +0.454) m`.
   - **Driver-side cantrail & headliner valence:** `509 x 90 mm` (`x in [-0.665, -0.156] m, y in [-0.155, -0.065] m`, `z in [+0.020, +0.064] m`) above the `590 x 30 x 40 mm` upper door header.
   - **Gripper-pad edge protection (`Safety`):** Use a shatter-safe `4 mm` clear polycarbonate/acrylic window sheet (`570 x 272 mm`, `~USD 12`, never raw annealed glass) at `y = -0.120 m`, and encase **all four window edges** plus the `690 mm` front windshield header lip in continuous `14 x 14 mm` black EPDM rubber push-on U-channel weatherstrip (any push-on rubber edge trim for 3 to 5 mm panels, `~USD 2/ft`) so zero sharp or bare edges can contact the gripper pads.

## Run

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 \
    --instruction "<see task.json>" --scorer task_score --max-steps 950 --store-frames --log-dir logs

inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
    --instruction "<see task.json>" --scorer task_score --max-steps 2400 --store-frames --log-dir logs
```

One trial takes about 4 minutes on one GPU; fixed starting layout (`object_jitter=0`, no seeds). `VISOR_MODE=swing_no_slide` (`expected 0.5`: swings to side window without extending, `~61%` coverage) and `VISOR_MODE=unclipped_partial` (`expected 0.0`: unclips partway without reaching the window) run physical controls; `python check_visor.py .` runs the 578-hull 20 Hz reach and collision check.

## Known Approximations

- Joint frictions (`0.6 N m` swivel/roll, `4 N` slide), clip jaw torsion stiffness (`0.05 N m/rad`), and rigid `430 x 170 x 25 mm` blade mass (`0.35 kg`) are engineering estimates; offline 20 Hz IK replay (`check_visor.py`) has zero collisions across all 578 robot hulls with up to `18.1 mm` transient tracking error.

## License & Attribution

Helper utilities (`labkit.py` and `arena_assets_fix.py`) adapt patterns from `yam-isaac-lab-arena` and Isaac Lab (`Apache-2.0` and `BSD-3-Clause`). All scene geometry, printable STL generator (`generate_cad_stl.py`), and scoring code are authored from scratch.
