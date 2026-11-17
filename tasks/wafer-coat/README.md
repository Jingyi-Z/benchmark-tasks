# Load a wafer onto a spin coater's chuck with tweezers and a wafer aligner

A 2-inch silicon wafer lies on a small stand beside a benchtop spin coater whose clear lid is shut; a tweezer slot cut
into the stand's front edge runs in under the wafer's edge. Angled wafer tweezers stand tips down in a cup, and a PTFE
wafer aligner (a "fork") lies on the bench: a flat plate with a ring of posts forming a wafer pocket and a slot from the
pocket to its far end that fits round the chuck, ending in a V. One hand lifts the tweezers out of their cup, tilts them
so the blades lie flat, slides the open blades round the wafer's edge with the lower blade in the slot, pinches it,
lifts it well above the coater's lid and lowers it into the aligner's pocket, the lower blade in the pocket's notch; it
lets go, backs the blades out and stands the tweezers back in their cup. The other hand lifts the coater's lid by its
handle bar and holds it open. The first hand picks the aligner up by its handle, carries it over the chuck, lowers it
until its slot straddles the chuck, pushes it on until the V seats against the chuck, lowers it so the wafer is left on
the chuck, centred, slides it back out from under the wafer, lifts it out and sets it down. The other hand then lowers
the lid shut. The wafer's top and edge are never touched by the gripper.

## Real-trial build and scoring

A mock coater keeps the parts the task uses: the chuck, the ribbed bowl and the hinged clear lid that opens to 74
degrees and falls shut. It has no motor or vacuum, as in the simulation. `python make_parts.py --wafer-d <D>` (needs
`pip install numpy trimesh manifold3d shapely`) writes every printed part as an STL file, in mm, and the lid as a DXF to
laser-cut. `<D>` is the trial wafers' diameter, measured with calipers, about 50.8 mm. The aligner, the wafer stand,
the tweezers' cup, the bowl and the chuck are read from `task.py`, so the printed parts have the simulation's shapes.

| Part | What to use | Approx. USD |
|---|---|---|
| Printed parts | `parts/*.stl`: the housing in four 110 mm quarters, the bowl (204 mm across), the chuck, two hinge brackets, two lid knuckles, the lid's handle, the wafer stand, the tweezers' cup, the aligner, three centring gauges (206 mm across) and an edge template. PETG, about 1 kg, on any printer with a bed of at least 210 mm; or a print service | 25 to 40 (own printer); 120 to 200 (service) |
| Lid | `parts/lid_5mm_acrylic.dxf`: 5 mm clear cast acrylic, 216 x 220 mm, laser-cut with the 12 mm dispense hole, two back notches and four screw holes (about 280 g; 250 g in the simulation) | 15 to 30 |
| Hardware | Two 3 x 20 mm steel dowel pins (hinge); ten M3 x 12 screws (four for the bowl, two each for the brackets, knuckles and handle); three 1 mm felt pads for the rim | 10 |
| Tweezers | 2-inch wafer tweezers with angled flat blades about 14 mm wide and PEEK or PTFE tips, about 100 to 120 mm long | 20 to 60 |
| Wafers | 2-inch test-grade silicon, about 0.28 mm thick | 10 to 20 each |
| Scoring kit | Digital calipers with a depth rod; a yellow and a pink fluorescent highlighter; a 365 nm UV flashlight; two phones or webcams on stands | 40 to 60 |

Total: about USD 150 to 250 with your own printer, or 250 to 400 with a print service.

**Print.** PETG, 0.2 mm layers, 4 walls. Print the aligner at 0.1 mm layers, plate down, or in resin. Print the
chuck top face down, so its top is flat. Print the bowl and the gauges flat side down. Colours as in the screenshot:
housing and aligner white, bowl grey, chuck black.

**Assemble.**
1. Glue the four housing quarters together with epoxy on a flat bench.
2. Screw the bowl into the housing's recess (four M3), then press the chuck's pin into the bowl's centre hole. Check
   with the calipers that the chuck's top is 52.0 mm (+-0.2) above the bench.
3. Stick the three felt pads on the rim, 120 degrees apart, so the shut lid rests 1 mm above it, as in the simulation.
4. Screw the handle onto the lid from below (two M3). Clip a knuckle onto the lid's back edge at each outer side of
   the two notches, and screw it through the lid (M3).
5. Press a pin into each knuckle. Slide the pins through the hinge brackets' holes, then screw the brackets to the
   housing's back wall (two M3).
6. Check: the lid opens until it rests on the brackets' sloped faces at 74 degrees, and falls shut from there when let
   go. The brackets stand 34 mm above the shut lid at its back edge; in the simulation the stop is the hinge's limit.

Lay the station out as in task.json `setup` and the screenshot, with positions marked on the bench with tape.

**Before each trial.** Clean the wafer with IPA and let it dry. Mark the tools with the yellow highlighter: both
tweezer blades, both faces, full length, and the aligner's top face, posts and ramps. Mark the robot's four fingertip
pads with the pink highlighter. Film the whole trial from above and from the front at bench height, 30 frames per
second.

**Score each trial, in this order.**
1. *Video.* Score 0 if the wafer tipped past 40 degrees or touched the table. Score 0 if, at the end, any fingertip
   is within 80 mm of a task object (the lid's 220 mm width gives the scale in the frame). Score 0 if anything still
   moves in the last second.
2. *Lid.* It counts as shut if its front edge is at most 11 mm above its resting height (3 degrees over 220 mm),
   measured with a ruler. If it is not shut, the score is halved.
3. *On the chuck.* Open the lid by hand and lower `gauge_12mm` straight down until it rests on the rim. It must seat:
   resting flat all round, without rocking. If it seats, the wafer's centre is within 12 mm of the chuck's axis. Then
   use the calipers' depth rod through its open centre to measure from the gauge's top down to the wafer's top, at
   three points 15 mm from the wafer's centre, 120 degrees apart. The expected depth is 75.0 - 52.0 - t mm, where t is
   the wafer's thickness: 22.7 mm for 0.28 mm. All three readings must be within 1.0 mm of that (the wafer's underside
   within 1 mm of the chuck) and within 0.9 mm of each other (flat within 2 degrees). If either check fails, the
   wafer is not on the chuck and the score is 0. If both pass, the score is at least 0.25.
4. *Centring.* Lift off the 12 mm gauge and lower `gauge_1p5mm`. If it seats, the score is at least 0.5; then lower
   `gauge_0p5mm`. If that one seats too, the score is 1.0. Each gauge's tube has a bore of D + 2 x its tolerance, so
   it passes round the wafer's edge only if the wafer's centre is within that distance of the axis (checked with the
   generated files: 0.4 mm seats and 0.6 mm is blocked for the 0.5 mm gauge). Stop at the first gauge that does not
   seat, and don't retry it, since it may have moved the wafer.
5. *Contact.* Lay the edge template on the wafer, which centres itself on the wafer's edge. Shine the UV light from
   above. A pink mark anywhere on the wafer, top or rim, means the gripper touched or held it: score 0. A yellow mark
   seen through the template's 38.8 mm window means a tool touched the top more than 6 mm inside the rim: score 0.
   Yellow marks under the template's ring are in the 6 mm handling edge and are allowed, as are marks on the
   underside.

The simulation scores closeness (a tool within 0.5 mm of the wafer's top); real trials score contact marks.

**Gauge accuracy.** Before a session, measure each gauge's bore with the calipers. It must be within 0.1 mm of D +
2 x the gauge's tolerance; if not, reprint it, scaled to fit. With no wafer on the chuck, each gauge must seat
without rocking. Then centre a wafer on the chuck by hand with the aligner: `gauge_0p5mm` must seat. The gauges
locate on the bowl's ribs, and the bowl and the chuck's hole are one print, so the check reads centring to about
+-0.2 mm.

**Reset.** Wipe the marks off the wafer with IPA. Put it back on its stand, pattern up. Put the tweezers back in their
cup and the aligner back on the bench as in the screenshot, and shut the lid. Renew the highlighter marks.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The coater, the wafer stand and the tweezers' cup
  are fixed; the coater's lid is on a damped hinge and falls shut unless held; the wafer, the tweezers and the aligner
  are loose. The tweezers are two halves, one on a sprung labkit slide, so squeezing their arms closes the blades. The
  slide runs along the blades' face normal (a turned joint frame, `labkit.JOINT_FRAME_ROT`), and the sprung blade is
  set back 6.2 mm along its length, so the PEEK tips (friction 0.4 on silicon, `min` combine) meet and close straight
  onto the wafer; the hand squeezes them at about 2 N. Every tweezer collider, blades included, has a 4 mm torsional
  patch radius (minimum 2 mm), half the 8 mm arm face the pads press: an estimated contact approximation, not a
  measured calibration (without it the demo still picks and carries the wafer but centres it to 1.5 mm, not 0.5 mm).
  The wafer, drawn 1 mm thick at its real 1.3 g, has a flat collider strip flush inside the
  edge the tweezers grip (19.4 to 24.9 mm out, 10 mm wide, inside the rim), so the blades meet flat faces there. An
  `on_step` hook latches the violations (a fingertip over the wafer's top or at the wafer's rim; a tweezer blade or
  the aligner touching the wafer's top more than 6 mm inside its rim, checked over points spread across their faces;
  the wafer dropped).
- `make_parts.py`: writes the real-trial parts (STL files to print, the lid as a DXF to laser-cut), the aligner, stand,
  cup, bowl and chuck from `task.py`'s own shapes (see Real-trial build and scoring).
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted` (about 88 s of motion; the trial ends 3 s after full
score). The demo reads only what the cameras see: the tweezers' tips and the wafer, to trim the hand's aim while it
holds the tweezers, and the aligner's position, to line its slot up with the chuck. Run one trial per process: the
trial's record is kept in the task module and cleared at each reset.

The score is `task_score` in the log, computed by `score()` in `task.py`. Every positive tier also needs both hands clear
(each fingertip, and the midpoint between a hand's fingertips, at least 80 mm from every task object's surface,
the tools and the lid included) and everything still (no surface point of any object, joint bodies included, moved 1 mm
or more over the last 21 control steps).

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial; about 11 minutes of wall time on one L40S with frames
  stored at 640 x 352.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.

## Shortcut check

`WC_CHECK=tipontop` runs the same demo with the tweezers pinching 14 mm in from the rim, so the upper blade rests on
the wafer's top about 11 mm inside it, past the 6 mm handling edge. The trial must score 0.
