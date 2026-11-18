# Load three sample boats into their temperature zones in a tube furnace with a push rod, then stopper the tube's inlet end

Loading a tube furnace is a routine hand step in materials and device research labs. A split-tube furnace, its lid open,
holds a 2 inch quartz tube. Each end of the tube takes a tapered one-hole silicone stopper with a stainless gas tube
through it. As in daily use, the far (exhaust) end's stopper stays in, and the near (argon inlet) end's stopper is out,
lying in a cradle beside the tube. Argon comes in at the near end, so the near end is upstream.

Three boats wait on a tray beside the near end: precursor 1 (a yellow load), precursor 2 (a white load), and a clear
boat with a substrate face up. A push rod with a flat T-handle across its back end lies in two channel rests in line
with the tube, behind its near end. The tube lies loose in the furnace's saddles, centred. In real trials the boats,
their loads, the substrate and the rod are the cold-trial stand-ins below; there is no powder on the rig.

The scripted demo runs in this order:
1. The left hand takes the tube by its far end, slides it 75 mm towards the far side so the near end stands close to the
   furnace, and holds it there.
2. For each boat, the right hand sets the boat in the tube's mouth by its tongue, takes the rod by its T-handle, pushes
   the boat in, then racks the rod.
3. The left hand slides the tube back, centred, and keeps holding it.
4. The right hand takes the inlet stopper by its gas tube, lines it up with the tube's end as seen (the loose tube can
   lie a degree off its line in its saddles), and pushes it in, held lightly so its taper can centre it, until it stops.
5. The right hand lets go and lifts clear; then the left hand lets go of the tube; both hands rest over 100 mm from
   everything.

The layout is scored against the furnace:
- precursor 1 at the furnace's upstream edge, where it stays cool enough to evaporate slowly;
- precursor 2 under the thermocouple, in the hot zone;
- the substrate 50 to 80 mm downstream of precursor 2, centre to centre;
- the inlet stopper seated in the tube's end, square to it;
- the tube back where it started, 120 mm out of the furnace at each end.

## The stopper

The stopper is Chemglass CG-1420-05: a #10 silicone stopper with one 9/16 in hole, nominally 42 mm across at its small
end, 50 mm at its large end, 25 mm long (standard #10 sizing; not measured lot tolerances). Sellers' prices seen on
Nov 17, 2026: USD 60.96 each (VWR, catalog 80068-662) and USD 72.51 each (CP Lab Safety,
https://www.cplabsafety.com/stopper-only-10-silicone-9-16-hole-each/cg-1420-05).
A 5/8 in stainless tube through its hole carries the gas. In the 46 mm bore of a 2 in quartz tube, a #10 stopper starts
in by about half its length and is pushed in until it stops.

In the simulation:
- The stopper is rigid, at its real size. It seats where its taper meets the tube's inner edge, 11.81 mm in; a real one
  squeezes further. Its small end is rounded over 3 mm, as on a moulded stopper.
- The gas tube is a hollow stainless tube, 1 mm wall, 60 mm long: 30 mm out of the large end and 5 mm past the small end.
- The tube's end is a 32-sided ring, so the stopper bears on it all round.
- The bore collides as facets (32 sides over the end lengths, 12 between), and its floor is a flat quartz strip, 20 mm
  wide, 2 mm proud, starting 15 mm inside the loading end (clear of the seated stopper). The boats' underside end edges
  are bevelled 1.2 mm up over their last 4 mm, as on a moulded boat's rounded ends, so a dragged boat rides up the strip's edge.
- Seated is scored as its small end within 1 mm of 11.81 mm in, and its axis within 2 degrees and 1 mm of the tube's.

Real trials use the real stopper's own seated depth, D_real, not 11.81 mm. Before the trials, push this stopper into
this tube by hand until it stops, pass the vacuum hold below, then read a ruler from the tube's end to the stopper's
large end at four points round it and mark that reading (D_real). A trial's stopper must read within 1 mm of it at all
four points. The vacuum hold has not been run for this submission; it is the real-trial check.

## Every positive score

Both hands must be released and everything loose must be still:
- Released: each hand's pad midpoint and both jaw points (from its two finger joints separately), six points in all,
  80 mm or more from every task surface. The fixed parts (furnace with its open lid and controller, the tube's supports,
  the stopper's cradle, the tray, the rod's rests) are bounded by boxes; the loose parts by capsules round them where
  they are: the tube with its exhaust stopper and gas tube, the inlet stopper and its gas tube, each boat with its
  tongue, the substrate, the rod's bar and paddle, and its T-handle.
- Still: points at the extremities of the tube, the inlet stopper, each boat, the substrate and the rod (both ends,
  the paddle's corners, the handle's top corners) each moved under 1 mm between any two of the last 21 samples (1 s at
  20 Hz), so turns count as well as slides.

## Cold-trial parts

All real trials are cold, with no loose precursor powder or process gas on the shared rig. The yellow and white loads
are captive solid inserts specified in the parts table below; their assembled boat masses and contact geometry match
the published cold-trial setup. Inspect the inserts and attachments before every reset, and replace any damaged
assembly before a trial.

The furnace body may be a 3D-printed or plywood shell with the same chamber, saddles and thermocouple position (the
heater is never on). The quartz tube (2 in OD, 46 mm ID, 610 mm) and the two stoppers are real items. The boats, their
loads and the push rod are printed from `make_parts.py` at the simulation's sizes (it checks them against `task.py`);
the substrate is a stainless square of the simulation's mass.

| Part | Qty | Make it from | Size (mm) | Mass |
|---|---|---|---|---|
| Precursor boat (white) | 2 | `boat_precursor.stl`, white PETG | 60 x 20 x 12, 2 mm walls; tongue 5 x 5 x 20 turned 55 deg, axis 4 mm up; underside end edges bevelled 1.2 mm up over 4 mm | 7.49 g each |
| Yellow load (precursor 1) | 1 | `load_yellow.stl`, yellow PETG | 55.6 x 15.6 x 4 slab (0.2 mm clearance inside the walls) | 4.41 g |
| White load (precursor 2) | 1 | `load_white.stl`, white PETG | as above | 4.41 g |
| Substrate boat (clear) | 1 | `boat_substrate.stl`, clear PETG; closed floor pocket 18 x 10 x 0.8 at 0.6 to 1.4 mm up | 35 x 20 x 12, as the precursor boat | 4.76 g printed |
| Ballast plate | 1 | 304 stainless sheet 0.8 mm, cut with tin snips or laser | 18 x 10 x 0.8 | 1.14 g |
| Substrate stand-in | 1 | 304 stainless sheet 0.5 mm, top painted purple | 10 x 10 x 0.5 | 0.40 g |
| Push rod | 1 | `push_rod.stl`, natural PETG, one piece | 8 x 8 bar, 293 long, hollow with closed 1.0 mm walls; paddle 4 thick, 14 to 20 wide, 9 to 11.5 tall, foot bevelled; T-handle plate 26 x 5 x 45 (hollow, 1.0 mm walls) over the bar's end | 14.74 g |
| Cyanoacrylate glue | - | any thin CA glue | bonds each load to its boat's floor | - |

Assembled masses against the simulation (`python make_parts.py` prints this table; PETG 1.27 g/cm3, steel 7.93 g/cm3):

| Assembly | Printed or cut | Simulation | Difference |
|---|---|---|---|
| Precursor boat + load | 11.89 g | 12.0 g | -0.9 % |
| Substrate boat + ballast | 5.90 g (4.76 printed + 1.14 plate) | 6.0 g | -1.6 % |
| Push rod | 14.74 g | 15.0 g | -1.7 % |
| Substrate stand-in | 0.40 g | 0.4 g | -0.9 % |

Weigh each part after printing; reprint any part more than 0.5 g from its listed mass.

How to make them:
1. `python make_parts.py out/` (Python 3 with numpy) writes the five STL files (mm). Print solid (100 % infill), 0.2 mm layers, 0.4 mm nozzle.
   The boats print upright on their floor; the loads flat. The push rod prints lying on one side of the bar, and needs a
   bed of at least 300 mm along its diagonal (it fits diagonally on a 250 x 210 mm bed); its hollow bar and plate are
   closed shells, so no support is needed inside them.
2. Substrate boat: insert a pause after the 1.4 mm layer, drop the ballast plate into the pocket, resume. The plate ends
   sealed inside the floor.
3. Loads: put a thin bead of CA glue over the boat's floor, press the load in flat against the floor, and let it cure
   10 min. The load sits 0.2 mm inside the walls all round and cannot come out without breaking the bond.
4. Weigh every part against the table, and check the tongue and bevels by eye.

Cost: about 45 g of PETG in all (about USD 1 at USD 25 per kg spool), two scraps of stainless sheet and a drop of
glue. Before each trial: turn each boat over and tap it on the tray; a load that moves is re-glued or the boat replaced.

Real-trial seal-check kit (fitted only after a trial, not in the scene): two short silicone hoses that push onto the
5/8 in gas tubes; two quarter-turn ball valves; a barbed tee; a vacuum gauge reading to -100 kPa; a hand vacuum pump
(e.g. a Mityvac-type pump); a timer in view.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file).
  - Fixed: the furnace, tray, rod rack, the stopper's cradle and the tube's two blocks.
  - Loose: the tube (0.49 kg, with its exhaust stopper), the three boats, the substrate, the rod and the inlet stopper
    (0.08 kg).
  - An `on_step` hook latches the violations, follows the end state, and keeps the release and stillness gate (see
    Every positive score).
- `labkit.py`: parts built from several shapes. It hooks `yam_arena.scene` and the embodiment from
  the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task uses
  no library assets; every object is built in `task.py`.
- `make_parts.py`: writes the cold-trial print files (STL) for the boats, loads and push rod, checks their sizes against
  `task.py` and prints the mass table (see Cold-trial parts).

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 5000 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. It aims by the poses of the boats (outside the tube), the
rod, the stopper and the tube. Run one trial per process. The score is `task_score` in the log, computed by `score()` in
`task.py`.

## Runtime and seeds

- 5,000 control steps (250 s of simulated time) per trial. The scripted demo ends at about 4,785 steps (3 minutes 59 s): the trial
  stops 2 s after the score reaches 1.0, which needs both hands resting clear and everything still.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.

## Scoring a real trial

- Boats: the quartz tube is clear in a real trial, so each boat's centre is read with a ruler along the tube from the
  furnace's near end and from the thermocouple's tip.
- Tube: a ruler at each end of the furnace gives how far the tube stands out (120 mm each, within 5 mm).
- Hands and stillness (every positive score): both hands off and 80 mm or more from every part, nothing moving, judged
  from the video's last second.
- Stopper seated: a ruler from the tube's end to the stopper's large end reads within 1 mm of D_real at four points
  round it (D_real: see The stopper, above).
- Seal (needed for real seal credit): with the tube cold, push a hose with a shut ball valve onto the exhaust stopper's
  gas tube, and a hose from the inlet stopper's gas tube to the tee, which carries the gauge and, through the second ball
  valve, the pump. Press on neither stopper. Pump to -30 kPa gauge, shut the pump's valve (the gauge stays on the tube);
  the pressure must rise under 2 kPa in 30 s, gauge and timer on video. This is a cold check, not proof of hot or
  high-vacuum performance.
- A tipped boat or a flipped substrate: seen in the video.
