# Unfold the desk of a folding music stand

A folding music stand (K&M 10065 type) stands on the floor with its height and tilt set, but its desk is still folded for
carrying: both halves lie against the centre spine and the diagonal brace under each half is bent at its knee. The robot
swings out both ledges until the braces are straight, so the desk is wide and supported for a score, keeps the light
stand from tipping or sliding, and lets go. A soprano ukulele lies face up in its open hardshell case on a small side
table just in front of the stand, in the path a hand would take to the stand tube at the usual height; neither the
ukulele nor its case may be moved.

## Why this is familiar

Opening a folding music stand is a routine part of school band, orchestra and choir rehearsals and music lessons:
players open their own stands before they play. The motion is recognizable in a few seconds (a folded black desk on a
tripod, two ledges swung out until the desk is flat), and it is not trivial for a robot: each half of the desk is a
linked frame that reaches its supported open position only when its brace straightens, the stand is light and tips
easily, and the instrument case lies close by.

## Run a trial

From this folder, with the yam_arena / inspect-robots environment active:

    MS_DFRIC=0.01 inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
      --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
      --scorer task_score --max-steps 2400 --store-frames --log-dir ./logs

Model run (Dream-1): the same command with `--policy dream` and a separate log folder.

- Settings used for the recorded demo and the Dream-1 run: `MS_DFRIC=0.01` (brace pin friction; the code default is
  0.0), as in the command above; everything else default.
- Runtime: the scripted demo deploys both halves in about 35 s of simulated time (it first scores 1.0 at step 700); the
  run limit is 2,400 steps.
- Seeds: none. Fixed start pose, object_jitter 0.
- Controls (environment variables): `MS_NOMOTION=1` (hands stay parked: score 0), `MS_NOSTEADY=1` (same right-hand
  motion with the left hand parked), `MS_NOPADS=1` (hand positions unavailable: no credit, the scorer fails closed),
  `MS_BOOK=open` (an open 9 x 12 in score laid on the desk, for the support check in the table below),
  `MS_HIGHAPP=0 MS_TUBEZ=0.68` (the earlier plan: each hand goes straight in to the tube at 0.68 m, the height used
  without the case; it pushes the ukulele, latched 0; see "Recorded results"), `MS_UKE=0` (no table, case or ukulele), `MS_CASELID=<deg>` (lid angle; default 270, hanging).

## The scripted demo

The holding hand never reaches low across the ukulele: it lifts first, travels high (0.95 m) to above the stand,
comes down slowly in front of the folded desk, slides 10 cm in under the ledges and settles on the stand tube at
0.748 m, above the ukulele and its case. It leaves the same way: up 10 mm, back out slowly, straight up, then away.
The other hand stays parked or out wide until the holding hand is down at the tube, and only then goes to its wing, so
the two hands never share the space beside the desk. The right hand pinches the lip of the right ledge and swings it out past its stop,
which straightens that side's diagonal brace; the hands swap roles for the left ledge. Then both hands move clear. The
motion is a fixed plan: no decision in it depends on simulator state.

The wrist camera of the hand holding the tube is nearly black while it holds: it faces the black folded desk from a few
centimetres away. Rolling the wrist 90 deg to turn the camera away lost the grip on the tube (score 0), and the hold
height cannot move (the case is below it and the desk just above), so the demo keeps the grip that holds.

The case lid is open past 180 deg and hangs down over the table's arm-side edge. A lid standing open (35 to 180 deg)
lies in the reach of the forearms or elbows; hanging below the table top it is clear of them.

## The ukulele and its case

The ukulele is a separate rigid body (0.40 kg) lying in the case's lining, with convex colliders for its two bouts, its
neck and its headstock; its strings, tuners and bridge are drawn but do not collide. The case (1.0 kg), its hanging lid
and the table collide as convex pieces. A hand that touches the ukulele pushes it; the score latches 0 if the ukulele
or the case moves more than 5 mm or turns more than 3 deg from its pose after settling (step 40). A touch that moves
neither by that much is not penalized.

Built in code from published figures (pages opened Nov 23, 2026):
- music stand: K&M 10065, black, product page https://www.k-m.de/en/music-stands-with-collapsible-desk/10065-music-stand-black
  (height 585 to 1,220 mm; desk 485 x 240 x 42 mm; base diameter 690 mm; 1.55 kg; steel; foldable desk; U-profile
  legs with cross braces). Pivot positions in the desk are scaled to that desk size; the page gives no drawing. The
desk head's tilt-joint discs (radius 10 mm), wing knob and clamp knobs are estimates; the page gives no sizes for them.
- ukulele: Martin S1 soprano, https://www.martinguitar.com/ukuleles/soprano-body/S1-Uke.html (fingerboard 1 13/32 in
  at the nut and 1 13/16 in at the 12th fret, neck joins at the 12th fret, 17 frets, 1 41/64 in string spacing at the
  bridge). Body length, bouts, depth and the 345 mm scale are typical soprano values, not from that page.
- case and table: generic; sizes as in task.json.

## Score

`score()` in `task.py`, from the simulated joint angles, the stand pose and the fingertip positions; the numeric rule is
`task.json` "scoring". Hands clear is measured from the sampled fingertip points to sampled points of the nominal open
desk outline (ledge and top corners and midpoints) and of the stand tube, not to every surface, and must hold on every
one of the last 21 steps. The ukulele and the case are checked on every step from step 40 on (their poses after settling).
Stillness uses the stand position and the two ledge angles over the same 21 steps. If the
fingertips cannot be located, the clearance counts as zero (no credit).

## Recorded results

All on the same task.py, with `MS_DFRIC=0.01`, 2,400-step limit, scores from the scorer output (`task_score`).

| Run | Plan | Score | Ukulele / case moved |
|---|---|---|---|
| Demo (`demo.mp4`) | scripted: lift, approach high, descend, slide in, settle; same way out | 1.0, first at step 700 | 0.0 mm / 0.0 mm |
| Control (`MS_HIGHAPP=0 MS_TUBEZ=0.68`) | the earlier plan: hand straight in to the tube at 0.68 m | 0.0, latched "ukulele disturbed" | 17.6 mm |
| Dream-1 (`model_run.mp4`) | model, instruction from task.json | 0.0 (neither ledge deployed) | 0.0 mm / 0.0 mm |

Clearances of the recorded demo, from the simulator's own link poses logged during that run (offline; exact convex
distance with a certified lower bound per pair; the ukulele covered by convex pieces containing every drawn part;
PhysX contact events were not recorded):

| Pair | Sampling | Smallest gap |
|---|---|---|
| Left arm to right arm | 805 poses at 20 Hz | 91.8 mm (step 83) |
| Arms to the ukulele | 805 poses at 20 Hz | 40.9 mm (step 97) |
| Arms to the case and its hanging lid | 805 poses at 20 Hz | 38.4 mm (step 449) |
| Arms to the side table | 805 poses at 20 Hz | over 50 mm at every sampled pose |
| Arms to the music stand, outside the intended grips | every 0.10 s | 5.0 mm (step 294, a fingertip to a tilt-joint disc) |

No sampled pose has a common point between any of these. The intended contacts are a finger pad on the upper tube while
that hand holds it, and on a ledge's own lip while that hand pinches it.

## Files

- `task.py`: stand, desk linkage, scorer, scripted demo.
- `ms_models.py`: the ukulele, the case and the side table, built in code.
- `labkit.py`, `arena_assets_fix.py`, `cart_scene_fix.py`: shared helpers (rigid parts, joints, floor scene).

## Reproducing on real hardware

| Item | Part | Price |
|---|---|---|
| Folding music stand | K&M 10065, black, steel, 1.55 kg (3.42 lb), desk 485 x 240 x 42 mm, height 585 to 1220 mm | $72.99 (Full Compass, fullcompass.com) |
| Side table | small wooden side table, top about 26 x 70 cm at 0.60 m | about $40 |
| Ukulele | any soprano ukulele (wood, about 53 cm, about 0.4 kg) | about $50 |
| Case | soprano hardshell case shaped to the instrument, about 58 x 22 x 10 cm (75 mm base), lid hinged on the long side | about $40 |
| Support check (optional) | any 9 x 12 in music book laid open on the desk: on the deployed desk its free half stays flat (0.0 deg in simulation); on the folded desk it swings back 20 deg | about $10 |

Set the stand at 0.80 m ledge height with the desk tilted back 20 deg, both ledges folded in. Scoring by eye and a
protractor: each ledge at its stop and each brace straight. Put the side table's far edge 1.7 cm short of the stand's centre and the case at its near edge, lid hanging over
that edge, the ukulele face up in it. Tape marks beside the case and at the ukulele's headstock show a 5 mm move. Reset: fold both ledges back in. Nothing is consumed.
