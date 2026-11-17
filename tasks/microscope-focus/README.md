# Centre a slide's alignment cross on a microscope and focus it sharply

An upright student microscope with a mechanical stage, modelled on an AmScope B110C: a slide holder (a caliper arm on a
spring), its binocular head with the eyepiece camera, stage knobs turned through a printed belt adapter at the stage's
front right, coarse and fine focus knobs on both sides, and a bottom LED light.
A glass calibration slide with two crossed 1 mm scales (as the MAGUS CS5) is already in the holder, label on the left,
with the scales' crossing 12.5 mm in front of and 25.5 mm left of the objective's axis. A camera monitor (a tablet lying
flat on the table between the robots) shows the microscope camera's live 10x view, 0.38 x 0.22 mm of the slide, over
its whole screen. The robot turns the stage knobs to bring the crossing within 0.05 mm of the image centre, raises the
stage with a coarse knob to near the working distance (slide top 5.2 mm below the front lens) without ever bringing the
slide within 0.5 mm of the lens, then finishes with a fine knob until the scales are sharp (within 5 um of focus).

## The real setup and its figures

| Item | Real part | Figure used | Source or status |
|---|---|---|---|
| Microscope | AmScope B110C, about $261 (VB-B110C with a 1 MP USB eyepiece camera, about $416) | DIN 10x/0.25 achromat, bottom and top LEDs, coaxial coarse and fine focus, 30 mm focus range, 2 um fine divisions | https://selectumllc.com/i/amscope-b110c, https://selectumllc.com/i/amscope-vb-b110c-gld |
| Stage | the B110C's double-layer stage, 115 x 110 mm, slide caliper opened by pushing its lever outward | 50 mm left to right, 25 mm front to back | same pages; the B110C's two stage knobs share one vertical shaft under the stage's right side (manualslib 2902715), where a gripper coming from above cannot reach them; the model's two top-turn knobs are those of the knob adapter below |
| Stage knob gearing | not published | 25 mm per turn | estimate; check one turn of each knob on the stage's scales before trials |
| Coarse focus | not published | 24 mm per turn | estimate |
| Fine focus | 2 um divisions | 0.2 mm per turn (100 divisions) | estimate from the division |
| 10x working distance | not published | 5.2 mm | estimate for a DIN 10x/0.25 achromat |
| Camera | 1 MP eyepiece camera, 3.84 x 2.16 mm sensor (AmScope MD100 class), no reduction lens | 0.38 x 0.22 mm of the slide at 10x | sensor figures: https://selectumllc.com/i/amscope-md100; the field is sensor / 10 |
| Slide | MAGUS CS5 calibration slide, about $43: optical glass 75 x 25 x 1.1 mm, horizontal and vertical 1 mm scales in 0.01 mm divisions crossing at its centre | drawn 76 x 26 x 2 mm; on the monitor the two scale lines with their 0.01 mm ticks | https://www.levenhuk.com/catalogue/accessories/microscope-accessories/other-microscope-accessories/magus-cs5-calibration-slide/ |

Within the action limit (3,600 steps, 180 s): the stage moves 12.5 mm front to back (x knob) and 25.5 mm left to
right (y knob); the demo turns each knob in open quarter turns (about 9.5 s each, one grip) and then three closed-loop
turns per knob (about 10 s each), then focuses (coarse knob about 13 s, two fine-knob grips about 25 s). The stage-knob
gearing is not published, so the table covers the plausible range; the demo's other steps take about 80 s.

| Stage knob gearing | x knob turns | y knob turns | Open quarter turns (x + y) | Total, about |
|---|---|---|---|---|
| 15 mm per turn | 0.83 | 1.70 | 2 + 5 | 187 s: over 180 s |
| 17 mm per turn | 0.74 | 1.50 | 1 + 5 | 177 s |
| 20 mm per turn | 0.63 | 1.28 | 1 + 4 | 168 s |
| 25 mm per turn (the model) | 0.50 | 1.02 | 1 + 3 | 158 s (the demo) |
| 36 mm per turn | 0.35 | 0.71 | 0 + 1 | 130 s |

At the modelled 25 mm per turn, the recorded demo completes in about 163.1 s within the published 180 s limit. The table
estimates motion time and does not establish a guaranteed minimum real gearing. Measure the actual gearing (turn each
stage knob, or the adapter's top knob, exactly one turn and read the stage's travel on its vernier scales) and verify
full completion, including release and settling, before real trials. Use one published setup and run limit for every
compared trial; any changed gearing or limit must be documented and its feasibility demonstrated before those trials.

## Real trials: the stage-knob adapter (`adapter/`)

The B110C's two stage knobs share one vertical shaft hanging under the stage's front right, where a robot gripper
coming from above cannot reach them. A printed adapter carries two top-turn knobs (34 mm across, tops 17.5 mm above the stage top, as in the scene) outboard of the stage, 97 mm apart;
each turns one stage knob 1:1 through a 6 mm GT2 belt between two identical 48-tooth pulleys (a printed sleeve clamped
on the stage knob, a printed pulley on the knob's 5 mm shaft, the shafts in pairs of 625ZZ bearings). A turn of a top
knob is a turn of the stage knob under it, so the stage's own gearing and travel are unchanged. The adapter is the one
geometry for both: `task.py` places the stage knobs at the adapter's shafts and draws the adapter's own meshes
(`adapter/out/asm_*.stl`), with boxes and cylinders for its arm, wall and bosses to collide.

Mounting: the bracket's locating wall is bonded with 1 mm VHB tape (3M 4910) to the right edge face of the plate that
carries the coaxial control (assumed the lower plate, which moves only with focus). Nothing of the adapter rises above
that plate's top face inboard of its edge, so the upper plates travel over its edge without touching it; and because
the control and the bracket are on the same plate, the belts' centre distances (68 and 152 mm) never change over the
stage's 50 x 25 mm travel or its focus stroke. If the real control rides on a moving plate, bond the wall to that plate
instead: the adapter then moves with it and the centre distances are still fixed.

`adapter/knob_adapter.py` (cadquery 2.x) generates the parts as STEP files and the assembled parts and belts as STL in
`adapter/out/` (`adapter/out/clamp_bar.step` is from an earlier clamped design: obsolete, not part of this assembly). `adapter/check_adapter.py` places them among stand-ins taken from the scene (the carrying plate, the
upper stage plates swept over their whole travel, the holder's lever over that travel, the focus housing and the right
coarse and fine focus knobs, the coaxial stage knobs on their shaft, the shafts and bearings), raises any geometry
error, fails unless every part is one solid and nothing overlaps that should not (the result: none does; the only
overlaps are the belts on their own pulleys and sleeves), and draws sections through every bearing seat, bore, sleeve,
the wall and both belt planes (`adapter/out/sections.png`). The L-shaped arm runs in front of the right coarse focus knob, clear of it; the
clearances only grow as the stage rises.

**Measure first** (not published; the defaults are assumptions, set them at the top of `knob_adapter.py`): the two
stage knobs' diameters (18 and 24 mm assumed), heights (12 mm) and tops below the stage's top surface (20 and 35 mm),
the carrying plate's top and bottom faces (11 and 16 mm below the stage top) and where the knobs' shaft is (15 mm
inboard of its right edge, 5 mm behind its front edge).

| Part (print, PETG, 0.2 mm layers, no supports) | Print orientation | Fits that matter | If the fit is off |
|---|---|---|---|
| `bracket` (1): arm, two bearing bosses, locating wall | flat, bosses up | 625ZZ seats 16.1 mm (press), two bearings each | `BEARING_FIT` 0.0 to 0.2 |
| `sleeve_x`, `sleeve_y` (1 each): on the stage knobs | flat, flat flange down | bores knob + 0.4 mm (free), M3 set screw | `FIT` (0.2 snug) |
| `shaft_pulley` (2) | hub down | 5.2 mm bore, M3 set screw | |
| `top_knob` (2) | upside down, top face on the bed | 5.2 mm blind bore, M3 set screw | |

Hardware (typical prices, about $25 in all): four 625ZZ bearings (5 x 16 x 5 mm); 5 mm steel rod cut to 51 and 66 mm;
GT2 6 mm closed-loop belts, 232 mm (x) and 400 mm (y); six M3 set screws with nuts; 3M VHB 4910 tape, 1 mm, 80 x 5 mm;
about 90 g of PETG.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - The holder's lever, the x and y stage axes, the stage knobs and the coarse and fine focus knobs are labkit joints.
    The y axis rides on the stage, the x axis on the y axis and the holder on the x axis (chained joints).
  - An `on_step` hook sets the stage axes' drive targets from the knob angles: 25 mm per stage-knob turn, coarse focus
    24 mm and fine focus 0.2 mm per turn.
  - The scripted demo (right hand only) turns the y stage knob in quarter turns and then closed loops, the x knob the
    same way, a second and third closed loop on each, then the coarse focus knob and two grips on the fine knob, with
    closed loops on the knob angles (the `Servo` class). Each closed loop on a stage knob aims 0.08 mm past the axis,
    the turn the knob slips back when let go. `task.py` lets a demo waypoint carry a full gripper rotation for the side
    grips (the `Orient` class), so yam_arena itself is unchanged.
  - The camera monitor draws the 10x view from the stage's position over its whole screen: the two scale lines moved
    with the stage and drawn across the screen with their 0.01 mm ticks (every fifth and tenth longer), sharp (within
    5 um of focus), soft (0.1 mm) or blurred (1 mm), while
    their crossing is in the view, by showing one of three drawn copies (USD visibility and translation each step).
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
        --scorer task_score --max-steps 3600 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 158 s of motion).

The score is `task_score` in the log, computed by `score()` in `task.py` from the slide's pose in the holder, the
crossing's offset from the objective's axis, the stage height's distance from focus, the smallest gap from the slide to
the front lens during the trial, and a release and stillness gate kept every control step in `_drive()`: every positive
score needs all six jaw and pad points of both hands 80 mm or more from every task surface (`_clearance()`) and the
slide, stage, holder and every knob moving less than 1 mm over the last 21 control steps (each tracked as 3-D points
on the part: three on the slide, one on the stage carriage, the holder jaw's tip and a rim point of each knob, any two
samples compared). Possible scores: 0, 0.5, 0.75, 1.0.

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

- 3,600 control steps (3 minutes of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
