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
water-jet-cutting service; threads tapped by hand). Approximate cost of the whole rig, excluding the robot: USD 350 to
550.

| Part | What to use | Approx. USD |
|---|---|---|
| Sample plate | `sample_plate_1p5mm_ss304.dxf`: 304 stainless, 1.5 mm, 200 mm disc; tap the 30 holes M3 x 0.5 | 40 to 70 |
| Plate boss | `plate_boss_3mm_ss304.dxf`: 40 mm disc, 3 mm, four M3 x 4 countersunk screws (ISO 10642) into the plate | 10 |
| Clip tabs (8 + spare) | `clip_tab_1mm_ss301.dxf`: 301 stainless, half hard, 1.0 mm, 13 x 4 mm with a 3.2 mm hole | 10 to 20 |
| Clip spacers | M3 stainless shim washers 0.5 mm thick (DIN 988, 3.2 x 6 mm), one under each tab | 5 |
| Clip screws | 7 x M3 x 4 slotted cheese-head (ISO 1207 / DIN 84, A2 stainless, 5.5 mm head, 0.8 mm slot); the 225 degree clip gets M3 x 12 | 10 |
| Ring holder | `ring_holder_A_2p5mm_al6061.dxf` + `ring_holder_B_1p5mm_al6061.dxf`, joined with six M3 x 4 countersunk screws; B forms the 1.5 mm recess for the plate | 40 to 60 |
| Loading stand | `stand_top_6mm_steel.dxf` on a 50 x 50 mm steel square tube, cut so the top plate is 130 mm above the bench | 30 |
| Wafer | 100 mm silicon test wafer, about 0.5 mm thick (any wafer vendor) | 10 to 25 |
| Screwdriver and holder | 0.8 x 5 mm slotted screwdriver, stubby (any brand); holder: 30 x 30 x 60 mm block with a 6.5 mm square socket 12 mm deep (printed or wood) | 10 |
| Load-lock mock-up | Box 280 deep x 760 wide x 350 mm tall from 2020 aluminium extrusion and 6 mm plywood or aluminium panels, opening 100 to 320 mm above the bench; acrylic viewport on top | 120 to 180 |
| Door | `door_6mm_al6061.dxf` in two U-channel rails, 250 mm of travel; 140 mm pull handle (M6) | 50 to 80 |
| Twist lock | T-handle quarter-turn latch (compression or cam latch with T-handle, 16 mm panel hole) whose cam catches the frame under the opening | 15 to 25 |
| Top latch | Adjustable ball catch on the frame at the door's top travel, set to release at about 8 N | 5 |
| Light tower | 4-colour LED stack light (red, orange, green, blue), 24 V, with a 24 V supply and a small timer or switch that turns blue off and green on 8 s after the trial starts | 30 to 50 |
| Inner holder | Two 3D-printed or plywood arcs (radius 93 mm, between fences at 115.5 mm) on a 30 x 30 mm post, top 170 mm above the bench | 10 |

**Screws, threads and turns.** The clip screws are M3 x 0.5 (0.5 mm per turn). Each clip is stacked on the plate as:
0.5 mm shim washer, 1.0 mm tab, screw head. The washer is as thick as the wafer, so the tab lies level with the wafer's
top. At the start of each trial every screw is backed off so its head stands 0.35 mm above its tab (set it with a
0.35 mm feeler gauge under the head, then pull the gauge out): the tab swings freely. Once a clip is swung in over the
wafer, the screw closes that 0.35 mm gap after 252 degrees; 300 degrees adds about 0.07 mm of clamp, so the half-hard
tab presses the wafer's edge (snug); 420 degrees adds about 0.23 mm, which risks cracking the wafer and scores 0. The
long M3 x 12 screw at 225 degrees cannot be set this way: at the start its head stands 8 mm proud, and tightened, its tip
would come out of the plate's back. Each screw has a white paint mark on its head so turns can be counted from the video.

**Assembly.** Cut and deburr the parts; tap the plate's holes M3; screw the boss under the plate's centre; put a shim
washer, a tab and a screw in each of the eight 59.5 mm-ring holes (the long screw at 225 degrees) and set each to the
0.35 mm start; bolt the stand's top plate to its post; build the load-lock box, fit the door rails, door, handle, T-handle
latch (its cam catching the frame when upright, free after a quarter turn clockwise seen from the robot), the ball catch at
the top of the door's travel, the light tower and the inner holder.

**Reset for each trial.** Lift the ring holder off the plate and put it back on the inner holder, face up; turn every
clip screw back to the 0.35 mm start and park its tab along the rim; put the wafer back pattern up, its flat facing the
90 degree clip, 5 mm off centre towards 202.5 degrees (a printed template on the plate helps); stand the screwdriver in
its holder; shut and lock the door; set the light tower to blue and start its timer with the trial.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The stand, load lock, door frame and screwdriver
  holder are fixed; the plate (with eight hinged clips, each with a hinged screw), the wafer, the ring holder and the
  screwdriver are loose. The door slides on a joint; the twist lock turns on one; each lamp's lens slides out when lit. An `on_step` hook works the light, the door's lock and latch, times how long the door is open, latches the violations,
  follows the screws and checks that everything has been still for the last second.
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

- 6,600 control steps (330 s of simulated time) per trial. The scripted demo finishes in about 320 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
