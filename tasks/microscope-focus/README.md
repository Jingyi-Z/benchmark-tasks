# Centre a slide's alignment cross on a microscope and focus it sharply

An upright microscope with a 50 x 25 mm 2-layer mechanical stage, representing the named trial configuration below
(AmScope VB-B110C binocular microscope with 1 MP USB eyepiece camera, MAGUS CS5 calibration slide, and the printed
stage-knob belt adapter in `adapter/`): a slide holder (a caliper arm on a spring), a binocular head with the eyepiece
camera, stage knobs turned through the printed belt adapter at the stage's front right, coarse and fine focus knobs on
both sides, and a bottom transmitted LED light.
A glass calibration slide with two crossed 1 mm scales (MAGUS CS5) is already in the holder, label on the left, with
the scales' crossing 12.5 mm in front of and 25.5 mm left of the objective's axis. A camera monitor (a tablet or
display lying flat on the table between the robots) shows the microscope camera's live 10x view, 0.38 x 0.22 mm of the
slide, over its whole screen. The robot turns the stage knobs to bring the crossing within 0.05 mm of the image centre,
raises the stage with a coarse knob to near the working distance (slide top 5.2 mm below the front lens in the model)
without ever bringing the slide within 0.5 mm of the lens, then finishes with a fine knob until the scales are sharp
(within 5 um of focus).

## Real-trial bill of materials and configuration

The task specifies one complete, low-cost trial route (about **USD 478 to 506** checkout cost across the commercial
bundle, slide, belt/bearing packs and shop-stock consumables, or **USD 472 to 498** consumed-part allocation, excluding
tax and shipping; no purchase is required before submission, and physical assembly/qualification is performed before
running real trials):

| Item | Specification and quantities (consumed vs. purchased) | Dated source URLs and prices checked |
|---|---|---|
| Microscope + eyepiece camera (`VB-B110C`) | AmScope `VB-B110C` bundle (1 consumed / 1 purchased): `B110C` binocular upright stand (2-layer mechanical stage with 50 x 25 mm travel and caliper slide holder, coaxial coarse/fine focus with 30 mm stroke and 0.002 mm fine divisions, 10x/0.25 DIN achromat objective, 1 W bottom transmitted LED) + 1 MP USB 2.0 CMOS eyepiece camera (`MD100` 3.84 x 2.16 mm sensor class, no reduction lens -> 0.38 x 0.22 mm field at 10x) | **USD 415.99** at `https://amscope.com/products/c-vb-b110c` (checked Nov 16, 2026); `VB-B110C-GLD` variant at `https://selectumllc.com/i/amscope-vb-b110c-gld` observed at **USD 410.65** (list 418.99, checked Nov 18, 2026; web cache also shows **USD 423.75**, live checkout varies; stand alone `B110C` USD 260.70 at `https://selectumllc.com/i/amscope-b110c`; `MD100` camera specs at `https://selectumllc.com/i/amscope-md100`) |
| Calibration slide (`MAGUS CS5`) | Optical glass 75 x 25 x 1.1 mm (drawn 76 x 26 x 2 mm), horizontal and vertical 1 mm scales in 0.01 mm divisions crossing at the slide centre (`https://www.levenhuk.com/catalogue/accessories/microscope-accessories/other-microscope-accessories/magus-cs5-calibration-slide/`, spec page; 1 consumed / 1 purchased) | **USD 42.90 to 48.00** (`$42.90` checked Nov 16, 2026 / ~USD 43 to 48 checked Nov 18, 2026 at `https://www.astroshop.eu/blank-slides-cover-plates/magus-calibration-slide-cs5/p,82613`) |
| Stage-knob belt adapter (`adapter/`) | 1x 232 mm GT2-6 closed-loop belt (1 purchased: **USD 4.09**, SKU `3423-0006-0232`, `https://www.gobilda.com/2mm-pitch-gt2-timing-belt-6mm-width-232mm-pitch-length-116-tooth/`, Nov 18, 2026); 1x 400 mm GT2-6 closed-loop belt (1 consumed from a 4-pack at **USD 8.00 to 11.00**, ~USD 2.00 to 2.75 consumed, `https://www.walmart.com/ip/4-Packs-2GT-Timing-Belt-Closed-Loop-Rubber-Belts-400-2GT-6-6mm-Width-x-400mm-Length-3D-Printer-Timing-Belt/5151299763`, Nov 18, 2026); 4x standard **unflanged** `625ZZ` 5 x 16 x 5 mm bearings (16.0 mm cylindrical OD presses flush into the 16.1 mm blind boss seats; 4 consumed = 1x 4-pack at **USD 6.00 to 9.00**, `https://www.walmart.com/ip/625ZZ-Ball-Bearing-5x16x5mm-Double-Sealed-Chrome-Steel-Bearings-4pcs/288933179`, Nov 18, 2026); plus 5 mm steel rod (cut to 51 and 66 mm), six M3 set screws with nuts, 3M VHB 4910 1 mm tape (80 x 5 mm), and ~90 g PETG filament (**USD 6.00 to 10.00** consumed from shop stock) | **USD 24.09 to 34.09** belt/bearing pack checkout + shop-stock allocation (**USD 18.09 to 25.84** consumed-part allocation) |
| **Dedicated apparatus total** | **Sum of microscope + camera bundle, slide, and adapter above** | **USD 477.64 to 505.84** pack checkout (**USD 471.64 to 497.59** consumed), excluding tax and shipping |

**Existing reusable station and lab equipment (not counted in the dedicated apparatus total above):**
- Bimanual YAM robot station and table (`top 0.7475 m`).
- Reusable USB/HDMI display or tablet/laptop placed flat at `(0.09, 0)` showing the eyepiece camera's live full-screen feed with a centre crosshair overlay.
- Desktop FDM 3D printer (for printing the five PETG adapter parts once before trials) and standard shop stock (1 kg PETG spool, M3 fastener assortment, 5 mm rod stock, VHB tape roll).
- Qualification and scoring tools: two synchronized video cameras with a millimetre tape scale, a millimetre ruler, and digital calipers (`0.01 mm` resolution).

Context references: CDC's ocular-micrometer job aid
(`https://reach.cdc.gov/sites/default/files/job-aids-resources/Calibration%20of%20Ocular%20Micrometer.pdf`) and
Evident's camera-calibration guide (`https://evidentscientific.com/en/insights/how-to-calibrate-your-microscope-camera`).

## Per-axis gearing qualification and runtime budget

AmScope publishes the `B110C`'s 50 x 25 mm stage travel, 30 mm focus stroke, and 2 um fine division, but does not
publish mm-per-turn leads for its stage or focus knobs or the 10x objective's working distance. The simulation models a
measured/documented **reference stand** available to the authors (a Zeiss Axio Imager.A2 upright stand; it appears below
**only** as the reference example row for the simulated `15 / 15 / 2 / 0.2 mm/turn` figures, never as the trial unit,
and none of its leads are attributed to the `VB-B110C` as measured `VB-B110C` values). On the `VB-B110C` trial unit,
each axis is measured before trials over a safe angle inside its available stroke:

| Axis / parameter | `VB-B110C` trial unit (before trials) | Zeiss A2 reference example row (simulated) | Candidate bench-microscope range (validate against <= 204 s) | Pre-trial safe-angle qualification method on `VB-B110C` |
|---|---|---|---|---|
| Stage `x` (front-back, 12.5 mm reset; 25 mm total travel) | Unmeasured (`25 mm` travel published; mm/turn unpublished) | **15 mm/turn** (rough ruler measurement over one 360 deg turn on Zeiss A2, 75 mm stroke) | **15.0 to 40.0 mm/turn** (15 mm/turn reference up to Nikon P-SXY64 37.5 mm/turn class) | Measure over a known angle that stays inside the available stroke, using start/end caliper readings in one direction to avoid reversal backlash. For example, measure a 90-degree turn where it is safe and calculate lead = 4 times the displacement. Record the angle, displacement, instrument resolution and available remaining travel. Use a complete revolution only where the stroke permits it. |
| Stage `y` (left-right, 25.5 mm to crossing; 50 mm total travel) | Unmeasured (`50 mm` travel published; mm/turn unpublished) | **15 mm/turn** (independent rough one-turn ruler measurement on Zeiss A2, 50 mm stroke) | **15.0 to 40.0 mm/turn** (15 mm/turn reference and Nikon P-SXY64 24.1 mm/turn class) | Same unidirectional safe-angle caliper procedure as `x` (e.g., 90-degree turn x 4 or 180-degree turn x 2 inside the 50 mm stroke; use a complete revolution only where the stroke permits it; record angle, displacement, 0.01 mm caliper resolution, and remaining travel). |
| Coarse focus (1.0 mm reset below focus; 30 mm stroke) | Unmeasured (`30 mm` coaxial stroke published; mm/turn unpublished) | **2.0 mm/turn** (Zeiss Axio Imager manual `M70-2-0020 e` & product table) | **2.0 to 38.0 mm/turn** (Zeiss 2.0 mm/turn to Nikon Eclipse E100 37.7 mm/turn) | With the objective safely clear, measure unidirectional vertical stage displacement with 0.01 mm calipers over a safe angle inside the 30 mm stroke (e.g., 90-degree turn x 4, or 180-degree turn x 2 if safe; use a complete revolution only where the stroke permits it). Record angle, displacement, resolution, and remaining travel. |
| Fine focus (finish within 5 um) | Published **0.002 mm (2 um) divisions**; mm/turn unpublished | **0.20 mm/turn** (Zeiss manual: 0.1 of coarse, 100 x 2 um divisions) | **0.15 to 0.40 mm/turn** (100- to 200-division 2 um drums) | Count 2 um drum divisions per 360 deg revolution and verify unidirectional stage displacement over 2 revolutions with 0.01 mm calipers (resolution 0.01 mm; drum-count lead and caliper displacement/2 must agree within +-0.02 mm/turn). |
| 10x/0.25 working distance | Unmeasured (`DIN 10x/0.25` published; WD unpublished) | **5.2 mm** (assumed for a 10x/0.25 achromat) | **>= 2.0 mm** (clears 1.0 mm reset and 0.5 mm crash limit with >= 1.5 mm margin at focus) | Focus on the `MAGUS CS5` scale gently by hand and measure the front-lens-to-slide gap on the calibrated side camera (0.1 mm resolution). |

**Demonstrated reference phase breakdown and runtime planning (4,800 steps = 240.0 s limit, 204.0 s 15%-margin acceptance; fixed physical reset 12.5 mm front, 25.0 mm left, 1.0 mm below focus; 0.05 mm centring and 5 um focus thresholds unchanged):**
- **Exact scripted phase durations in `task.py` (`15 / 15 / 2.0 / 0.20 mm/turn` reference):**
  1. Initial approach to stage knobs: **2.50 s**.
  2. Open quarter-turn stage cycles (`X_QUARTERS = 2`, `Y_QUARTERS = 5` -> `7` cycles x `9.50 s`, each `3.30 s` approach/grip + `4.00 s` quarter turn + `2.20 s` release/lift/wrist-back): **66.50 s**.
  3. Closed-loop stage servo cycles (`3` on `y` + `3` on `x` -> `6` cycles x `10.55 s`, each `3.30 s` approach/grip + `5.05 s` servo + `2.20 s` release/lift/wrist-back): **63.30 s**.
  4. Coarse-focus cycle (`5.80 s` approach/grip + `6.00 s` five open steps + `5.05 s` servo + `4.80 s` settle/release/unroll): **21.65 s**.
  5. Fine-focus cycle 1 (`3.20 s` approach/grip + `6.05 s` active servo at `<= 0.4 rad/s`, turning about one-third of a turn + `2.40 s` settle/release/back-off) and cycle 2 (`5.00 s` clear/unroll/re-grip + `8.05 s` active servo at `<= 0.4 rad/s` + `2.40 s` settle/release/back-off): `11.65 + 15.45` = **27.10 s**.
  6. Final clear, unroll, park, and hold (`1.0 + 1.0 + 1.5 + 2.0 + 2.0`): **7.50 s**.
  - **Total scripted waypoint sum:** `2.50 + 66.50 + 63.30 + 21.65 + 27.10 + 7.50 = 188.55 s`, plus **3.00 s** post-waypoint hold to early-success completion = **191.55 s (3,831 steps)** in native run `mf_rev6b` (`Dream-1`: **0.0** over 4,800 steps).
- **Non-reference gearings:** The native reference configuration completed in 191.55 seconds. A planning allowance of one additional 12-second fine-focus cycle would give 203.55 seconds; this is an estimate, not a demonstrated bound for every candidate gearing (in `task.py`, a full re-oriented fine-focus cycle with a 6 s active servo is 13.45 s, or 12.00 s with a 4.55 s servo, turning about one-third of a revolution per grip at `<= 0.4 rad/s`). Before using another gearing, record its leads and validate the complete motion, correction, release and settling budget against the 204-second acceptance target. The 240-second run limit and accuracy thresholds remain unchanged.

## Real trials: the stage-knob adapter (`adapter/`)

The `B110C`'s two stage knobs share one vertical shaft hanging under the stage's front right, where a robot gripper
coming from above cannot reach them. A printed adapter carries two top-turn knobs (34 mm across, tops 17.5 mm above the
stage top, as in the scene) outboard of the stage, 97 mm apart; each turns one stage knob 1:1 through a 6 mm GT2 belt
between two identical 48-tooth pulleys (a printed sleeve clamped on the stage knob, a printed pulley on the knob's 5 mm
shaft, the shafts in pairs of unflanged 625ZZ bearings). A turn of a top knob is a turn of the stage knob under it, so
the stage's own gearing and travel are unchanged. The adapter is the one geometry for both: `task.py` places the stage
knobs at the adapter's shafts and draws the adapter's own meshes (`adapter/out/asm_*.stl`), with boxes and cylinders for
its arm, wall and bosses to collide.

Mounting: the bracket's locating wall is bonded with 1 mm VHB tape (3M 4910) to the right edge face of the plate that
carries the coaxial control (assumed the lower plate, which moves only with focus). Nothing of the adapter rises above
that plate's top face inboard of its edge, so the upper plates travel over its edge without touching it; and because
the control and the bracket are on the same plate, the belts' centre distances (68 and 152 mm) never change over the
stage's 50 x 25 mm travel or its focus stroke. If the real control rides on a moving plate, bond the wall to that plate
instead: the adapter then moves with it and the centre distances are still fixed.

`adapter/knob_adapter.py` (cadquery 2.x) generates the parts as STEP files and the assembled parts and belts as STL in
`adapter/out/` (`adapter/out/clamp_bar.step` is from an earlier clamped design: obsolete, not part of this assembly).
`adapter/check_adapter.py` places them among stand-ins taken from the scene (the carrying plate, the upper stage plates
swept over their whole travel, the holder's lever over that travel, the focus housing and the right coarse and fine
focus knobs, the coaxial stage knobs on their shaft, the shafts and bearings), raises any geometry error, fails unless
every part is one solid and nothing overlaps that should not (0 unintended overlaps at the default parameters; only the
belts overlap their own pulleys and sleeves), and draws sections (`adapter/out/sections.png`). The L-shaped arm runs in
front of the right coarse focus knob, clear of it; clearances only grow as the stage rises.

**Assumed `B110C` attachment dimensions and candidate CAD adjustment bounds** (AmScope does not publish these knob and
plate dimensions; the shipped STEP/STL defaults below are unmeasured **assumptions**, not a physical fit on a `B110C`
yet). The shipped default has been checked in the simulated stand-in scene. The ranges below are candidate adjustment
bounds subject to the stated coupled constraints and a fresh CAD check (`python check_adapter.py`) for the measured
unit; they are not a verified Cartesian product of clash-free fits. The upper stage's underside in this scene is
`z = -11 mm`, so `PLATE_TOP_Z` must not exceed `-11 mm`:

| Parameter in `knob_adapter.py` | Shipped default (assumed) | Candidate adjustment bounds (run `check_adapter.py`) | Caliper measurement on `B110C` and fit rule |
|---|---|---|---|
| `X_KNOB_D`, `Y_KNOB_D` | `18.0 mm`, `24.0 mm` | `10.0 to 24.5 mm` | Outer diameters of upper (`x`) and lower (`y`) coaxial stage knobs; `sleeve()` bore `knob_d + 0.4 mm` keeps >= 1.8 mm radial wall under the 28.55 mm GT2 tooth root |
| `X_KNOB_H`, `Y_KNOB_H` | `12.0 mm`, `12.0 mm` | `8.0 to 16.0 mm` (`X_BELT_Z - Y_BELT_Z >= 11.0 mm`) | Axial heights of the two stage knobs; keep vertical separation between belt mid-planes `X_BELT_Z` and `Y_BELT_Z` at least `11.0 mm` so the 10.25 mm-tall coaxial sleeves clear |
| `X_KNOB_TOP`, `Y_KNOB_TOP` | `-20.0 mm`, `-35.0 mm` | `-20.0 to -26.0 mm` (`x`), `-33.0 to -45.0 mm` (`y`), with `X_BELT_Z <= PLATE_BOT_Z - 7.0 mm` | Distance of each knob's top face below the stage top (`X_BELT_Z = X_KNOB_TOP - X_KNOB_H / 2`); keeps the `x` sleeve and `x` pulley (`+5.75 mm` above `X_BELT_Z`) clear below the carrying plate, locating wall (`PLATE_BOT_Z - 1.0 mm`) and bracket arm |
| `PLATE_TOP_Z`, `PLATE_BOT_Z` | `-11.0 mm`, `-16.0 mm` | `-13.0 to -11.0 mm` top, `-17.5 to -15.0 mm` bottom (`PLATE_TOP_Z - PLATE_BOT_Z >= 4.0 mm`) | Carrying stage plate's top (`<= -11.0 mm`, clearing the `-11.0 mm` upper stage underside) and bottom (`>= -17.5 mm`, keeping the wall bottom above the `-18.8 mm` focus housing); set `BOSS_H = 1.0 - PLATE_TOP_Z` (`12.0 to 14.0 mm`, seating both 5 mm unflanged 625ZZ bearings) so top knobs stay `+17.5 mm` above the stage top |
| `AXIS_IN`, `AXIS_BACK` | `15.0 mm`, `5.0 mm` | `8.0 to 25.0 mm` inboard, `3.0 to 25.0 mm` back | Coaxial shaft centre inboard of carrying plate's right edge and behind its front edge; if `AXIS_IN` or `AXIS_BACK` shifts top-knob coordinates (`(0.205, -0.046)`, `(0.205, -0.144)`), select standard closed-loop GT2 belt lengths (`BELT_X_L`, `BELT_Y_L`) and `SHAFT_U` or update `task.py` and re-validate |

| Part (print, PETG, 0.2 mm layers, no supports) | Print orientation | Fits that matter | If the fit is off |
|---|---|---|---|
| `bracket` (1): arm, two bearing bosses, locating wall | flat, bosses up | unflanged 625ZZ seats 16.1 mm (press), two bearings each | `BEARING_FIT` 0.0 to 0.2 |
| `sleeve_x`, `sleeve_y` (1 each): on the stage knobs | flat, flat flange down | bores knob + 0.4 mm (free), M3 set screw | `FIT` (0.2 snug) |
| `shaft_pulley` (2) | hub down | 5.2 mm bore, M3 set screw | |
| `top_knob` (2) | upside down, top face on the bed | 5.2 mm blind bore, M3 set screw | |

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The holder's lever, the x and y stage axes, the stage knobs and the coarse and fine focus knobs are labkit joints.
    The y axis rides on the stage, the x axis on the y axis and the holder on the x axis (chained joints).
  - An `on_step` hook sets the stage axes' drive targets from the knob angles at the reference gearing: 15 mm per
    stage-knob turn, coarse focus 2 mm and fine focus 0.2 mm per turn.
  - The scripted demo (right hand only) turns the y stage knob in quarter turns and then closed loops, the x knob the
    same way, a second and third closed loop on each, then the coarse focus knob and two grips on the fine knob, with
    closed loops on the knob angles (the `Servo` class). Each closed loop on a stage knob aims 0.08 mm past the axis,
    the turn the knob slips back when let go. `task.py` lets a demo waypoint carry a full gripper rotation for the side
    grips (the `Orient` class), so yam_arena itself is unchanged.
  - The camera monitor draws the 10x view from the stage's position over its whole screen: the two scale lines moved
    with the stage and drawn across the screen with their 0.01 mm ticks (every fifth and tenth longer), sharp (within
    5 um of focus), soft (0.1 mm) or blurred (1 mm), while their crossing is in the view, by showing one of three drawn
    copies (USD visibility and translation each step).
  - The bottom light is a thin pale yellow disc on the objective's axis, drawn only, with no collider (labkit's
    `VISUAL_ONLY` colours).
- `labkit.py`: parts built from several shapes, with hinges and slides, including joints chained on other joints
  (`parent=`), and shapes drawn without a collider (`VISUAL_ONLY`). It hooks `yam_arena.scene` and the embodiment from
  the task's import.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 4800 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (completes in 3,831 steps, 191.6 s, at the simulated
15 / 2 / 0.2 mm/turn reference gearing).

The score is `task_score` in the log, computed by `score()` in `task.py` from the slide's pose in the holder, the
crossing's offset from the objective's axis, the stage height's distance from focus, the smallest gap from the slide to
the front lens during the trial, and a release and stillness gate kept every control step in `_drive()`: every positive
score needs all six jaw and pad points of both hands 80 mm or more from every task surface (`_clearance()`) and the
slide, stage, holder and every knob moving less than 1 mm over the last 21 control steps (each tracked as 3-D points
on the part: three on the slide, one on the stage carriage, the holder lever's pivot and angle with 62 mm radius bound,
and a rim point of each knob, any two samples compared). Possible scores: 0, 0.5, 0.75, 1.0.

## Scoring a real trial

- Centring: the crossing is in the camera's 0.38 x 0.22 mm view (0.5); for full credit it is within 0.05 mm of the
  image centre (the objective's axis), read with the camera software's centre crosshair against the slide's own
  0.01 mm divisions.
- Focus: without touching the coarse knob, turn the fine knob to the scales' best focus on the monitor and read the
  travel off its 2 um graduations. Within 0.1 mm is 0.75; within 5 um, with the crossing centred, is 1.0.
- Crash: a side video against a millimetre scale shows the smallest gap between the slide and the front lens; under
  0.5 mm at any time scores 0.
- Release and stillness: two synchronized videos with a tape scale show both hands 80 mm or more from the microscope,
  slide, knobs and monitor, and nothing moving in the final second.

## Runtime and seeds

- 4,800 control steps (4 minutes of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
