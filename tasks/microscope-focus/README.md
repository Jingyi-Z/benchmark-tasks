# Centre a slide's alignment cross on a microscope and focus it sharply

An upright microscope whose controls, heights and gearing are those of the measured trial stand below: a mechanical
stage with two leaf clips and a coaxial x/y drive under its front right (turned through the printed belt adapter in
`adapter/`), coaxial coarse and fine focus knobs on both sides, a binocular head and a Zeiss Axiocam 506 color camera
on a 0.5x camera adapter. A glass calibration slide with two crossed 1 mm scales (MAGUS CS5) is already under the
clips, label on the left, with the scales' crossing 12.5 mm in front of and 25 mm left of the objective's axis, the stage
1 mm below focus. A camera monitor (a tablet lying flat on the table between the robots) shows the camera's live view,
2.50 x 2.00 mm of the slide. The robot turns the stage knobs to bring the crossing within 0.05 mm of the image centre,
raises the stage with a coarse knob to near the working distance (slide top 9.0 mm below the front lens) without ever
bringing the slide within 0.5 mm of the lens, then finishes with a fine knob until the scales are sharp (within 5 um
of focus).

## The trial stand and its measured figures

The task needs an upright microscope with coaxial stage knobs, which research labs already have; the robot reaches
its stage knobs through the printed adapter. The specified trial configuration is the stand these figures were
measured on (Nov 18, 2026), a Zeiss Axio Imager.A2m. Every lead below was measured on it; none is taken from a
manual. The demonstrated configuration is the measured Zeiss Axio Imager.A2m with its stated objective, camera and
adapter. Other upright stands are candidates only after confirming stage travel, control access and adapter fit,
objective clearance, camera field, measured leads, and complete-motion timing within 4,800 steps.

| Item | Figure (used in the simulation) | How it was measured |
|---|---|---|
| Stage x (front to back), upper coaxial knob | **33.3 mm per turn**; clockwise seen from above moves the stage towards the user | the knob turned 540 degrees for the stage's full 50 mm travel, read against a ruler |
| Stage y (left to right), lower coaxial knob | **43.4 mm per turn**; clockwise seen from above moves the stage to the left | the knob turned 630 degrees for the full 76 mm travel |
| Coarse focus | **4 mm per turn**; the right knob turned clockwise, seen from the right, lowers the stage | stage travel over a counted turn |
| Fine focus | **0.333 mm per turn** | one fine turn moves the coarse knob 30 degrees (4 mm x 30 / 360) |
| Backlash | almost none on the stage knobs and the fine focus | by hand |
| Stage travel | 50 x 76 mm | ruler |
| Coaxial stage knobs | upper 32.8 mm across, 34.1 mm tall, top 40 mm below the stage top; lower 27.8 mm across, 30.6 mm tall, top 80 mm below | calipers |
| Focus knobs | coarse 60 mm across, 22.6 mm wide, axle 7 cm above the table; left fine knob 35 mm across, 25 mm long; right fine: a 50 mm graduated disc in the coarse knob's face | calipers, ruler |
| Heights | stage top 21.5 cm above the table at focus; objective front 22.5 cm; plate 5 mm thick, 16 x 28 cm | ruler |
| Objective | EC Epiplan-Neofluar 10x/0.25 HD (Zeiss 422340-9960-000), working distance **9.0 mm** | gap measured at focus |
| Slide clips | two leaf clips, 57 x 20.3 x 0.25 mm, on posts 5.8 mm across with tops 5.1 mm above the plate | calipers |
| Camera | Zeiss Axiocam 506 color, sensor 12.5 x 10.0 mm (2752 x 2208 px, 4.54 um), on a 0.5x camera adapter: with the 10x objective, **2.50 x 2.00 mm** of the slide, 0.908 um per pixel | sensor from Zeiss's product information, https://asset-downloads.zeiss.com/catalogs/download/mic/e595ff69-7731-4cdc-aaf9-e0d6e7e952af/EN_product-info_Axiocam-506_color.pdf |

The 0.05 mm centring target is about 55 camera pixels, and the 5 um focus target is within the 10x/0.25 objective's
depth of field (several um).

**Simplifications in the model:** the stage's top plate is 10 x 13 cm (the stand's is 16 x 28 cm), so the clips' posts
sit closer to the slide; the clips do not press it. The focus housing's width (16 cm) is assumed. The slide is drawn
2 mm thick (real 1.1 mm).

**A cheaper route.** The stand above costs far more than a few hundred US dollars. The AmScope VB-B110C is a named
lower-cost candidate; its configuration and runtime have not been qualified by this demonstration. It is the B110C
stand (50 x 25 mm travel, coaxial coarse and fine focus with 2 um fine divisions, 10x objective) with a 1 MP eyepiece
camera: **USD 410.65** (list 418.99) at https://selectumllc.com/i/amscope-vb-b110c-gld, checked Nov 18, 2026. Its leads
are not published and have not been measured. Before trials on it: measure each lead the same way (a counted turn
against a ruler for each stage knob and the coarse focus, the fine focus from its ratio to the coarse), confirm its
travel, knob access and the adapter's fit (new adapter dimensions), the objective's clearance and the camera's field
(its smaller sensor gives a smaller field), and rerun the demo at those values within 4,800 steps.

**Task-specific parts** (for either stand; prices checked Nov 18 to 19, 2026, before tax and shipping):

| Item | Price |
|---|---|
| MAGUS CS5 calibration slide (75 x 25 x 1.1 mm; 1 mm scales in 0.01 mm divisions, crossing at the centre; https://www.levenhuk.com/catalogue/accessories/microscope-accessories/other-microscope-accessories/magus-cs5-calibration-slide/) | about USD 43 to 48 (https://www.astroshop.eu/blank-slides-cover-plates/magus-calibration-slide-cs5/p,82613; https://astrotelescopios.com/en-us/products/portaobjetos-de-calibracion-magus-cs5-xy0-01-0-07-0-15-0-6-1-5) |
| GT2-6 closed belt, 280 mm | USD 3.96 (https://robotshop.com/products/3d-printing-canada-usa-gt2-6mm-timing-belt-loop-280mm) |
| GT2-6 closed belt, 400 mm | USD 14.95 (https://www.fruugo.co.nz/6mm-width-gt2-rubber-closed-loop-synchronous-timing-belt-for-3d-printer-gt2-400mm/p-380979672-821433681) |
| four 625ZZ bearings (5 x 16 x 5 mm) | about USD 11 a 10-pack (https://www.neweggbusiness.com/p/9B-325-003X-01535) |
| 5 mm steel rod, six M3 set screws and nuts, 3M VHB 4910 1 mm tape, about 100 g PETG | shop stock, about USD 10 (not quoted) |
| **Total, slide and adapter only** | **about USD 83 to 88**, excluding the microscope, its objective and camera, and the reusable station equipment below |

For the VB-B110C candidate, its USD 410.65 bundle comes on top of these parts, plus whatever its own adapter fit needs (not yet known).

Existing station and lab equipment (not counted): the bimanual YAM station and table; a tablet or laptop lying flat at
(0.09, 0) showing the camera's live image with a centre crosshair; a desktop FDM printer; a ruler and calipers; two
video cameras with a tape scale for scoring.

Context references: CDC's ocular-micrometer job aid
(https://reach.cdc.gov/sites/default/files/job-aids-resources/Calibration%20of%20Ocular%20Micrometer.pdf) and Evident's
camera-calibration guide (https://evidentscientific.com/en/insights/how-to-calibrate-your-microscope-camera).

## Runtime at the measured gearing (recorded demo)

Run limit 4,800 steps (240 s at 20 Hz). The recorded demo completes in **2,056 steps (102.8 s)**, 43% of the limit,
so the limit is unchanged. The two hands work at the same time: the right on the stage knobs, the left on the focus
knobs. Knob travel and timings from the run and from the demo's waypoints in `task.py`:

| Axis | Travel needed | Grips | Time (planned waypoints) |
|---|---|---|---|
| Stage y (43.4 mm/turn) | 25 mm: 0.58 turn (3.62 rad) | 1 open quarter turn, then 3 closed loops | 10.0 + 3 x 11.05 s |
| Stage x (33.3 mm/turn) | 12.5 mm: 0.38 turn (2.35 rad) | 3 closed loops | 3 x 11.05 s |
| Coarse focus (4 mm/turn) | 1 mm: a quarter turn (left hand, one grip; it aims 0.1 mm high because the stage sits that low on its drive) | 1: three open steps, then a closed loop | 17.75 s with the release |
| Fine focus (0.333 mm/turn) | the last 0.01 to 0.02 mm (about 0.2 rad) | 1 closed loop, after the right hand has parked | 15.1 s with the release |

The right hand's approach (2.5 s), the stage cycles (76.3 s) and its park (4.0 s) end at 82.8 s; the left hand waits
round the fine knob until then. Each stage cycle is 3.3 s approach and grip, a 5 s closed loop (or a 4 s open quarter
turn), and 2.7 s to let go, open wide, unroll and back out. Release, unroll and parking end at 99.75 s; the trial ends
on success after the 1 s stillness hold, at 102.8 s. Corrections recorded: each closed loop on a stage knob aims
0.02 rad counterclockwise of its target, the turn the knob makes as the fingers open. The final crossing offset in the
recorded demo is (0.007, -0.004) mm, about 0.008 mm from the image centre, and the focus error 0.9 um.

## Real trials: the stage-knob adapter (`adapter/`)

The stand's two stage knobs share one vertical shaft hanging under the stage's front right; a robot gripper is too
bulky to turn them. A printed adapter carries two top-turn knobs (34 mm across, 69 mm apart, tops 20 mm below the
stage top, as in the scene) outboard of the stage; each turns one stage knob 1:1 through a 6 mm GT2 belt between two
identical 64-tooth pulleys (a printed sleeve clamped on the stage knob, a printed pulley on the top knob's 5 mm shaft,
the shafts in pairs of 625ZZ bearings). A turn of a top knob is a turn of the stage knob under it, so the stage's own
gearing and travel are unchanged. The top knobs sit below the stage top because the arm's straight-down grip reaches
about 0.95 m above the floor of the robot frame but not the 0.98 m a knob above the stage top would need (yam_arena IK:
8 to 60 mm short). `task.py` places the stage knobs at the adapter's shafts and draws its meshes
(`adapter/out/asm_*.stl`), with boxes and cylinders for its arm, wall, web and bosses to collide.

Mounting: the bracket's locating wall is bonded with 1 mm VHB tape (3M 4910) to the right edge face of the plate that
carries the coaxial control; a drop web 75 mm behind the coaxial axis carries the arm down to 49 mm under the stage
top, clear of both coaxial knobs. Because the control and the bracket are on the same plate, the belts' centre
distances (76 and 136 mm) never change over the stage's travel or its focus stroke.

`adapter/knob_adapter.py` (cadquery 2.x) generates the parts as STEP files and the assembled parts and belts as STL in
`adapter/out/` (`clamp_bar.step` is from an earlier design, not part of this assembly). `adapter/check_adapter.py`
places them among stand-ins from the scene (the carrying plate, the upper plates swept over their whole 50 x 76 mm
travel with the clips on them, the focus housing, the right coarse knob and its fine disc, the coaxial knobs on their
shaft, the shafts and bearings), fails unless every part is one solid and nothing overlaps that should not, and draws
sections (`adapter/out/sections.png`). Result at the measured knob dimensions: every part one solid, 0 unintended
overlaps (only the belts overlap their own pulleys and sleeves).

Measured inputs: both coaxial knobs' diameters, heights and depths below the stage top. **Assumed** (measure before
printing and set at the top of `knob_adapter.py`, then rerun `check_adapter.py`): the carrying plate's top and bottom
faces (11 and 16 mm under the stage top) and the coaxial shaft's position (15 mm inboard of the plate's right edge,
5 mm behind its front edge). The adapter has not yet been fitted to a physical stand.

| Part (PETG, 0.2 mm layers, no supports) | Print orientation | Fits that matter | If the fit is off |
|---|---|---|---|
| `bracket` (1): wall, drop web, arm, two bearing bosses | flat, bosses up | 625ZZ seats 16.1 mm (press), two bearings each | `BEARING_FIT` 0.0 to 0.2 |
| `sleeve_x`, `sleeve_y` (1 each), on the coaxial knobs | flat, flange down | bores knob + 0.4 mm (33.2 and 28.2 mm, free), M3 set screw | `FIT` (0.2 snug) |
| `shaft_pulley` (2) | hub down | 5.2 mm bore, M3 set screw | |
| `top_knob` (2) | upside down, top face on the bed | 5.2 mm blind bore, M3 set screw | |

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The x and y stage axes, the stage knobs and the coarse and fine focus knobs are labkit joints; the y axis rides on
    the focus slide, the x axis on the y axis (chained joints). The leaf clips are on the x carriage.
  - An `on_step` hook sets the stage axes' drive targets from the knob angles at the measured leads and directions.
  - The scripted demo: the right hand turns the adapter's y and x knobs (straight-down grips, counterclockwise) in an
    open quarter turn where needed and then closed loops on where the crossing is seen; the left hand, at the same
    time, turns the left coarse knob a quarter turn and, once the right hand has parked, the left fine knob until
    sharp (side grips, fingers closing front and back). Closed loops are on the knob angles (the `Servo` class);
    `task.py` lets a demo waypoint carry a full gripper rotation (the `Orient` class), so yam_arena is unchanged.
  - Negative controls: `MICRO_CONTROL=untouched` (both arms stay parked) and `MICRO_CONTROL=crash` (the coarse knob
    turned on until the slide reaches the objective) both score 0.
  - The camera monitor draws the 2.50 x 2.00 mm view from the stage's position as a 5:4 picture: the two 1 mm scale
    lines (drawn in 0.05 mm segments, hidden where they leave the picture) with a tick every 0.1 mm, sharp (within
    5 um of focus), soft (0.1 mm) or blurred (1 mm), while their crossing is in view. The 0.01 mm divisions are 0.51 mm
    apart on the screen, finer than the drawn line; the real camera resolves them.
- `labkit.py`: parts built from several shapes, with hinges and slides, including chained joints (`parent=`) and shapes
  drawn without a collider (`VISUAL_ONLY`). It hooks `yam_arena.scene` and the embodiment from the task's import.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 4800 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (completes in 2,056 steps, 102.8 s).

Dream-1 run (rev9, 4,800 steps): DreamAPI refused the inference request at step 4,256 with "DA-4000 BAD_REQUEST (malformed): instruction invades privacy in this scene, refused: The task requires reading or viewing a monitor screen." The instruction is unchanged from the previous revision, whose Dream-1 run went the full 4,800 steps and scored 0.0; it is not reworded to avoid the refusal. The model-run video shows the 4,256 steps before the refusal.

The score is `task_score` in the log, computed by `score()` in `task.py` from the slide's pose under the clips, the
crossing's offset from the objective's axis, the stage height's distance from focus, the smallest gap from the slide to
the front lens during the trial, and a release and stillness gate kept every control step in `_drive()`: every positive
score needs all six jaw and pad points of both hands 80 mm or more from every task surface (`_clearance()`) and the
slide, stage and every knob moving less than 1 mm over the last 21 control steps (each tracked as 3-D points on the
part: three on the slide, one on the stage carriage, and a rim point of each knob, any two samples compared). Possible
scores: 0, 0.5, 0.75, 1.0.

## Scoring a real trial

- Centring: the crossing is in the camera's 2.50 x 2.00 mm view (0.5); for full credit it is within 0.05 mm of the
  image centre (the objective's axis), read with the camera software's centre crosshair against the slide's own
  0.01 mm divisions.
- Focus: without touching the coarse knob, turn the fine knob to the scales' best focus on the monitor and read the
  travel off its graduated ring (0.333 mm per turn). Within 0.1 mm is 0.75; within 5 um, with the crossing centred, is
  1.0.
- Crash: a side video against a millimetre scale shows the smallest gap between the slide and the front lens; under
  0.5 mm at any time scores 0.
- Release and stillness: two synchronized videos with a tape scale show both hands 80 mm or more from the microscope,
  slide, knobs and monitor, and nothing moving in the final second.

## Runtime and seeds

- 4,800 control steps (4 minutes of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
