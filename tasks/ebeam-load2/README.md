# Clip a wafer onto an e-beam evaporator's sample plate and load it face down

An electron-beam evaporator deposits metal from sources below the samples, so a wafer rides into it face down, clipped
to a sample plate that hangs in a ring holder. A clip that misses the wafer's round edge, too few clips, or a screw too
long for its hole means a wafer on the chamber floor or a jammed transfer; a screw driven too tight cracks the wafer.
An open load lock takes on air and water that slow its pump-down and load the main chamber, so its door stays open as
little as possible: 2 minutes in all. The load lock starts just vented, its door shut and locked. Its light tower
(red, orange, green, blue) shows blue while a process runs, when the door must stay shut, and green when it is done. The
task:

1. **Open the load lock** once the light is green: turn the twist lock a quarter turn, then lift the door by its
   handle until it latches at the top, **take out the ring holder**
   and set it on the sample plate (its recess fits round the plate's rim). The demo shuts the door again while it works
   on the plate.
2. **Centre the wafer.** The 100 mm wafer starts 5 mm off centre. Push it by its edge, never touching the pattern.
3. **Choose the clips.** Eight swing clips stand in the plate's 4 inch ring of holes. The wafer's flat faces one, which
   would bite nothing. One has a screw too long for its hole: its head stands 8 mm proud, and tightened, its tip would
   stick out of the plate's back and jam the transfer. Swing in clips that hold the round edge with no gap over 150
   degrees between them (the demo uses 0, 135 and 270 degrees).
4. **Tighten their screws** with the screwdriver, 300 to 420 degrees each, after the clip is swung in, and put the
   screwdriver back. The demo counts each screw's turns from its slot and stops once it is tight.
5. **Turn the plate and ring over together** (wafer face down; the ring now carries the plate) and set them on the
   load lock's holder.
6. **Close and lock the door.** Pull it down by its handle, off its latch, onto its seat, let go, and turn the twist
   lock back.

## Real-trial hardware

Everything is either off the shelf or cut from the DXF files that `python make_parts.py` writes (any laser- or
water-jet-cutting service; threads tapped by hand). The listed line items sum to approximately USD 445 to 650, excluding the robot, tax and shipping; these are
estimates, not quotations.

| Part | What to use | Approx. USD |
|---|---|---|
| Sample plate | `sample_plate_1p5mm_ss304.dxf`: 304 stainless, 1.5 mm, 200 mm disc; tap the 32 holes M3 x 0.5; counterbore the eight 4 inch ring clip holes 8.5 mm across and 0.6 mm deep (drill press, by hand), leaving 0.9 mm (1.8 turns) of thread | 40 to 70 |
| Plate boss | `plate_boss_3mm_ss304.dxf`: 40 mm disc, 3 mm, four M3 x 4 countersunk screws (ISO 10642) into the plate | 10 |
| Clip tabs (8 + spare) | `clip_tab_0p45mm_ss301.dxf`: 301 stainless, full hard, 0.45 mm (0.018 in shim stock), 15 x 4 mm tab, 13 mm from the pivot hole (3.2 mm) to the tip, so a swung-in tip lies 3.5 mm inside the wafer's edge. The tab is the clamp spring | 10 to 20 |
| Clip lift springs | M3 wave spring washers, DIN 137 B, A2 stainless (3.2 x 8 mm, 0.5 mm stock, about 1.6 mm free height), one in each clip counterbore under its tab; they only hold the tab up against the screw head | 5 to 10 |
| Clip screws | 7 x M3 x 3 slotted cheese-head (ISO 1207 / DIN 84, A2 stainless, 5.5 mm head, 0.8 mm slot); the 225 degree clip gets M3 x 12 | 10 |
| Ring holder | `ring_holder_A_2p5mm_al6061.dxf` + `ring_holder_B_1p5mm_al6061.dxf`, joined with six M3 x 4 countersunk screws; B forms the 1.5 mm recess for the plate | 40 to 60 |
| Loading stand | `stand_top_6mm_steel.dxf` on a 50 x 50 mm steel square tube, cut so the top plate is 130 mm above the bench | 30 |
| Wafer | 100 mm silicon test wafer, about 0.5 mm thick (any wafer vendor) | 10 to 25 |
| Screwdriver and holder | 0.8 x 5 mm slotted screwdriver, stubby (any brand); holder: 30 x 30 x 60 mm block with a 6.5 mm square socket 12 mm deep (printed or wood) | 10 |
| Load-lock mock-up | Box 280 deep x 760 wide x 350 mm tall from 2020 aluminium extrusion and 6 mm plywood or aluminium panels, opening 100 to 320 mm above the bench; acrylic viewport on top | 120 to 180 |
| Door | `door_6mm_al6061.dxf` in two U-channel rails, 250 mm of travel; 140 mm pull handle (M6). The aluminium door weighs about 2.83 kg before fittings | 50 to 80 |
| Door counterbalance | Two 25 mm pulleys on the frame top, 2 mm stainless cable from the door's top corners to a steel counterweight in a guarded 40 mm tube behind the box, outside the arms' workspace; adjust the weight until a spring scale reads 2.45 N net downward load over the full 250 mm of travel | 45 to 60 |
| Twist lock | T-handle quarter-turn latch (compression or cam latch with T-handle, 16 mm panel hole) whose cam catches the frame under the opening | 15 to 25 |
| Top latch | Adjustable ball catch on the frame at the door's top travel, set to release at about 8 N | 5 |
| Light tower | 4-colour LED stack light (red, orange, green, blue), 24 V, with a 24 V supply and a small timer or switch that turns blue off and green on 8 s after the trial starts | 30 to 50 |
| Inner holder | Two 3D-printed or plywood arcs (radius 93 mm, between fences at 115.5 mm) on a 30 x 30 mm post, top 170 mm above the bench | 10 |

**Screws, threads and turns.** The clip screws are M3 x 0.5 (0.5 mm per turn). Each clip is stacked as: a wave spring
washer in a 0.6 mm counterbore in the plate, the 0.45 mm spring-steel tab on the spring, the screw head on the tab. There
is no rigid washer under the tab: a rigid washer concentric with the screw would take the screw's force straight into the
plate and leave the tab no travel. The wave spring's only job is to hold the tab's root up against the screw head, so the
tab rides down with the screw.

At the start of each trial every normal screw is backed off so the tab's underside stands 0.35 mm above the wafer's top
(set it with a 0.35 mm feeler gauge between the tab and the wafer, then pull the gauge out): the tab swings freely. Once a
clip is swung in over the wafer, the screw brings the tab down onto the wafer after 252 degrees (0.35 mm). From there the
screw keeps pushing the tab's root down, below the wafer's top, so the tab bears on the wafer about at its edge, 9.5 mm
from the pivot, and bends over that 9.5 mm; its tip, 3.5 mm inside the edge, lifts slightly off the wafer. The bend is the
extra advance: 0.067 mm at 300 degrees (snug) and 0.233 mm at 420 degrees, the upper end of the allowed range; more than
420 degrees risks cracking the wafer and scores 0. The travel exists in the hardware: the wave spring flattens only 0.1 mm
below the plate's surface (0.5 mm stock in a 0.6 mm counterbore), so the tab's root can move about 0.6 mm below the
wafer's top, well beyond the 0.233 mm needed.

Calculated, not measured: from the screw to its bearing point at the wafer's edge the tab is a cantilever 9.5 mm long,
4 mm wide and 0.45 mm thick. With E = 200 GPa its stiffness there is k = 3EI/L^3 = about 21 N/mm, so the tab presses the
wafer's edge with about 1.4 N at 300 degrees and 5.0 N at 420 degrees; the bending stress at 420 degrees is about 350 MPa,
below the yield of full-hard 301. The simulation uses a 20 N/mm tab spring and the same screw advance, but draws the tab 1
mm thick and scores where the tab lies (its tip at least 2 mm inside the wafer), not where it bears. The 0.9 mm (1.8 turns)
of M3 thread under each counterbore carries only the clip's load: the tab's reaction (under 5 N, calculated) plus the wave
spring's force, which is not published and is part of the torque read in qualification. The simulation's screw torques
(about 0.03 and 0.09 N m at 300 and 420 degrees) are estimates; no torque in this submission is measured. Screw tips: M3 x
3 screws end about 0.2 mm behind the plate at the start and about 0.8 mm behind it at 420 degrees; keep at least 1.0 mm of
clear space behind every screw tip throughout loading, and check it on the assembled holder. The
long M3 x 12 screw at 225 degrees cannot be set this way: at the start its head stands 8 mm proud, and tightened, its tip
would come out of the plate's back. Each screw has a white paint mark on its head so turns can be counted from the video.

**Qualifying the clips before trials (a procedure; this submission has no measured values).** On the assembled holder,
with a wafer in place, for each clip: (1) set the 0.35 mm start with the feeler gauge; (2) swing the tab in and turn the
screw with a 0.02 to 0.2 N m torque screwdriver, reading the torque at 300 and at 420 degrees from the paint mark, and
record both; (3) at 300 degrees a 0.03 mm feeler gauge must not slide under the tab where it crosses the wafer's edge; (4) with all eight clips at
300 degrees, turn the plate face down over a soft pad and tap its rim five times: the wafer must not move (no shift
against a pencil mark on the plate); repeat the whole check five times. A clip that fails any step is replaced (new tab or
new spring) and requalified. Record the torques in the trial log; they are the measured values a trial reports.

**Safety.** Fit nitrile finger cots over the robot's gripper pads and wear safety glasses: a chipped or cracked wafer has sharp edges and flakes. Use test-grade wafers.

**Assembly.** Cut and deburr the parts; tap the plate's holes M3; screw the boss under the plate's centre; put a wave
spring washer in each counterbore, then a tab and a screw in each of the eight 59.5 mm-ring holes (the long screw at 225 degrees) and set each to the
0.35 mm start; bolt the stand's top plate to its post; build the load-lock box, fit the door rails, door, handle, T-handle
latch (its cam catching the frame when upright, free after a quarter turn clockwise seen from the robot), the door's counterbalance (cables over the pulleys to the counterweight in its tube; trim the weight to a 2.45 N net
downward load), the ball catch at the top of the door's travel, the light tower and the inner holder. Check that the door
falls closed when let go and that the top catch releases with about 8 N of extra pull. The simulation uses an effective
0.25 kg door and does not reproduce the counterbalanced door's inertia.

**Reset for each trial.** Lift the ring holder off the plate and put it back on the inner holder, face up; turn every
clip screw back to the 0.35 mm start (tab 0.35 mm over the wafer, feeler gauge) and park its tab along the rim; put the wafer back pattern up, its flat facing the
90 degree clip, 5 mm off centre towards 202.5 degrees (a printed template on the plate helps); stand the screwdriver in
its holder; shut and lock the door; set the light tower to blue and start its timer with the trial.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The stand, load lock, door frame and screwdriver
  holder are fixed; the plate (with eight hinged clips, each with a hinged screw), the wafer, the ring holder and the
  screwdriver are loose. The door slides on a joint; the twist lock turns on one; each lamp's lens slides out when lit. An `on_step` hook works the light, the door's lock and latch, times how long the door is open, latches the violations,
  follows the screws, and keeps the release and stillness gate each step: every positive score needs all six jaw and
  pad points of both hands 80 mm or more from every task surface (the plate with its clips and screws, the wafer, the
  ring holder, the screwdriver, the door with its handle and lock, and the fixed apparatus) and every moving task
  part, the clips, screws, screwdriver, door and lock included, tracked as 3-D material points (rim points carry
  rotations), moving under 1 mm between any two of the last 21 samples (1 s at 20 Hz).
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `make_parts.py`: writes the DXF cutting files for the real-trial hardware (see Real-trial hardware).
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 6600 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The demo reads only what a camera sees: the wafer's,
plate's, ring's, door handle's and screwdriver tip's positions and the screw slots' directions. Run one trial per
process. The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 6,600 control steps (330 s of simulated time) per trial. The scripted demo finishes in 6,421 steps (321 s).
- Dream-1 (rev3 run): DreamAPI refused the inference request at step 1,904 of 6,600 with "DA-4000 BAD_REQUEST (malformed): unsafe instruction in this scene, refused: Over-torquing screws past limits (300 to 420 degrees) damages equipment." The instruction is unchanged from the previous revision, whose Dream-1 run went the full 6,600 steps and scored 0.0; it is not reworded to avoid the refusal. The model-run video shows the 1,904 steps before the refusal.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
