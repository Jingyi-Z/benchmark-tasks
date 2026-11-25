# Zero a dial comparator on a master block, then measure and sort three spacers

A cast iron comparator stand holds a 0.01 mm dial indicator over its serrated anvil. One hand presses the indicator's
spindle lifting lever to raise the contact while the other slides a part in under it; easing the lever lets the contact
rest on the part. The dial is first zeroed on a 25.000 mm master block by turning its knurled bezel until the 0 mark sits
under the needle, and the master goes back to its yellow MASTER area. Each of three spacers that look alike is then read
against that zero: within 25.00 +- 0.02 mm goes in the ACCEPT bin, the others in the REJECT bin.

## Why this is Impact on jobs

The task is one inspection cycle: set a comparator to a master, read each part as a deviation, accept or reject.

| Occupation (O*NET-SOC) | US employment, O*NET 2025 | Core task, quoted from O*NET |
|---|---|---|
| Inspectors, Testers, Sorters, Samplers, and Weighers (51-9061.00) | 602,000 | "Measure dimensions of products to verify conformance to specifications, using measuring instruments, such as rulers, calipers, gauges, or micrometers." |
| Machinists (51-4041.00) | 288,300 | "Measure, examine, or test completed units to check for defects and ensure conformance to specifications, using precision instruments, such as micrometers." |
| Computer Numerically Controlled Tool Operators (51-9161.00) | 170,000 | "Measure dimensions of finished workpieces to ensure conformance to specifications, using precision measuring instruments, templates, and fixtures." |

Inspectors and machinists together are 890,300 workers; with CNC tool operators, 1,060,300. The inspectors' O*NET task
list also names the instrument: "Clean, maintain, calibrate, or repair measuring instruments or test equipment, such as
dial indicators, fixed gauges, or height gauges."

No public source counts how many of these workers use a comparator rather than another instrument, so the figures above
are the size of the occupations whose core duty this is, not a count of comparator users. Why the comparator is a
regular instrument for that duty: gaging "is the comparative process of determining whether a part is smaller or larger
than a predetermined standard (or master part) ... the standard or master is brought to the gage and the gage is set to
that size", and "gages tend to be used in high production areas" (G. Schuetz, Mahr Gaging Tips, "Measuring vs. Gaging").
The same article says "it's not uncommon to see dial and digital indicators set up to act as comparative gages" by
zeroing them to a master, which is the setup in this task.

Sources: https://www.onetonline.org/link/details/51-9061.00 , https://www.onetonline.org/link/details/51-4041.00 ,
https://www.onetonline.org/link/details/51-9161.00 ,
https://metrology.mahr.com/en-int/gaging-tips/detail/measuring-vs-gaging-the-value-of-conserving-measuring-range

## Run a trial

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=640 -E cam_height=352 --instruction "<see task.json>" --scorer task_score --max-steps 3200
```

- Seeds: `DC_SEED=0..7` picks which spacers are in tolerance (bit i = spacer i+1 good); the layout is fixed
  (`object_jitter=0`). Evaluation seeds are withheld from the controller.
- Recorded demo: seed 2 (`DC_SEED=2`), score 1.0 at step 2,914 of 3,200, on task.py sha256 73aaa8aa9ba0...; the Dream-1
  run scored 0.0 on the same hash.
- Scripted demo: `--policy yam_scripted` runs `demo()` in `task.py`. The right hand zeroes by pinching the knurled bezel
  top and bottom and turning the wrist, then lets go, backs off and looks at the dial with its wrist camera; it turns
  again until the camera shows 0 under the needle (up to three turns). Each spacer is read the same way and goes to the
  bin the camera reading calls for (needle inside the two red markers = ACCEPT).
- The demo reads the dial ONLY from the right arm camera image (`dial_read.py`: face ellipse rectified to a circle, the
  bezel's 0 direction from its two red tolerance markers, the needle from the dark pixels between hub and numerals). It
  holds and looks again when the image read fails. Each reading is filed under the part the demo is commanded to read
  at that moment (its own plan: the master look after each bezel turn, each spacer while the lever is eased), never
  under the simulated contact state; it also uses its own hand position (settled, clear of the bezel). The log prints `DCSEEN step part seen true` for every read, so the
  camera reading can be checked against the simulated one.
- Debug trace: `DC_DEBUG=1` prints one `DCDBG` line every 10 steps; `DC_LOGALL=1` one `DCL` line per step (used by the
  close-up panel).
- Runtime per trial: run limit 3,200 steps (2 min 40 s at 20 Hz).
- Scorer tests: `python test_scorer.py` (37 cases; pure Python, the simulator modules are stubbed and the production
  scoring functions run unchanged).

## Scoring (what `score()` checks, from the simulated state)

- **Reading.** The needle is driven only by the spindle's real travel (1 mm per turn); turning the bezel moves its 0
  mark. The reading is the needle against the 0 mark.
- **Zero.** With the master under the released contact and the bezel left alone, the needle reads within one 0.01 mm
  division of 0. The out-of-tolerance spacers are at least 0.04 mm off and the good ones within 0.01 mm, so a zero one
  division off cannot flip any sort (tested). Turning the bezel again voids all spacer readings.
- **Under the contact.** The tip is inside the part's footprint by 1 mm or more and within 0.01 mm of its top face, with
  the lever released (within 5 mrad of rest).
- **Measured.** 10 consecutive steps (0.5 s) under the released contact while a zero set on the master stands.
- **Bins.** A spacer is in a bin when its whole footprint is inside the bin's inner walls, resting on its floor.
- **1.0:** zero set on the master; all three spacers measured and each in the right bin; the master's whole footprint
  back inside its MASTER area; both hands at least 80 mm from every part and the dial for the whole final second, and everything still over that second.
- **0.5:** the same with exactly two spacers measured and right.
- **0:** anything else, including no zero, a zero set on a spacer, the master in a bin or outside its area, a part off
  the bench, a hand near, or motion at the end. Once the master is back after the zero, moving it more than 2 mm or
  tilting it more than 3 deg latches 0.

## Reading the dial from the policy cameras

The top camera sees the dial edge-on. The right arm camera reads it: the right hand holds it about 5 cm in front of the
dial face between the bezel turns and while each spacer rests under the contact. At 640 x 352 the face is about 110 px
across (about 2.9 px per division at the needle tip); at 224 x 224 about 68 px (about 1.8 px per division). The two red
tolerance markers sit at +-0.02 mm on the bezel. While handling parts (loading, unloading, bins) the right hand grips with its wrist turned half a turn, so its
camera stays about 10 cm in front of the dial and never passes through the stand or the dial; it changes between this grip and the look pose in two quarter turns at a stop 9 cm in front of the dial (about 98 and 90 deg, so the wrist never turns the long way round). The left arm camera cannot reach a view of the dial face.

## What is modelled

- Stand: KBC 1-807-450 type, cast iron base 3-7/8 x 6 in, 8 in tall, serrated anvil. Its three knobs (fine adjust on
  top of the arm, column clamp on the side, indicator stem clamp at the front) are modelled where the maker's photo
  shows them and are not used: the arm stays locked and zeroing is by the bezel. Base thickness is an estimate.
- Indicator: Mitutoyo 2046A type, 57 mm dial, knurled bezel about 60 mm across, 100 divisions of 0.01 mm, 10 mm travel,
  flat contact, spindle return spring 150 N/m (estimate). The bezel is a hinged ring with friction; the face, numerals
  and two red tolerance markers turn with it, the needle does not.
- Lifting lever: Mitutoyo 21EZA198 type at the top of the spindle; 40 mm length is an estimate. Pressing its end lifts
  the spindle 0.05 mm per mrad.
- Parts: the master, a 25.000 mm rectangular steel gauge block (35 x 9 mm, yellow band, "MASTER 25.000"), and three
  ground steel spacers (40 x 10 mm footprint, painted 1 / 2 / 3) whose heights depend on the seed: 25.01 / 25.00 /
  24.99 mm when good, 25.04 / 25.04 / 24.96 mm when not. All on a blue plastic parts tray.
- Bins: ACCEPT (green rims) and REJECT (red rims), 100 x 80 x 15 mm, names printed on the bench in front of each.

## Real-trial parts and prices (listings checked Nov 22-23, 2026; not re-verified at purchase)

Published prices:

| item | route | price |
|---|---|---|
| dial comparator stand, cast iron base 3-7/8 x 6 in, raised serrated anvil, fine adjust | KBC 1-807-450 (KBC Tools) | $102.65 |
| dial indicator 0-10 mm x 0.01 mm, lug back, rotating bezel with clamp, tolerance markers (Mitutoyo 2046A type; used as a comparator, see below) | unbranded "10mm high precision dial indicator, lug back, white face" (Econo Superstore) | $13.99 |
| spindle lifting lever, to suit the indicator | Mitutoyo 21EZA198 (Penn Tool Co); it is made for Mitutoyo indicators, and its fit to the unbranded indicator is NOT verified. Sylvac sells a lever for all dial gauges, price on request | $13.95 |
| master: gauge block 25 mm, steel, grade 0 | Mitutoyo 611635-531 (TestEquity) | $57.59 |
| spacer stock: ground flat stock 10 x 25 x 500 mm, O1 tool steel | Cromwell IND4133903D | GBP 15.49 ex VAT |
| micrometer 0-1 in (0 to 25.4 mm), to measure each spacer once (the spacers run about 24.96 to 25.04 mm) | Fowler 52-240-001-1 (fowlerprecision.com) | $52.00 |
| printed parts tray and two bins | one 1 kg filament spool (MatterHackers) | $19.99 |

Spacers are not made to a size; they are measured. Six 40 mm pieces are sawn from the ground flat stock and deburred,
standing on a ground 10 x 40 mm face so the 25 mm width is the height (comparable ground stock lists +0.13/-0 mm on width; not stated for this size). Each piece
is measured once with the micrometer at its centre and corners (checked against the master first), and that measured
height decides its correct bin. High rejects are chosen from blanks that are already oversize; low rejects and pieces to
bring into tolerance are lowered by lapping the top face on abrasive paper on a flat plate (lapping only removes stock).
A piece whose reading lies within 0.005 mm of a limit (24.975 to 24.985 or 25.015 to 25.025 mm) is not used. The
simulated heights are exact examples (25.01 / 25.00 / 24.99 mm in tolerance, 25.04 / 25.04 / 24.96 mm out); a real
trial uses the measured heights.

Total: about $280 ($279.84: USD items $260.17, plus the spacer stock at GBP 15.49 ex VAT x $1.27 = $19.67). Not
counted: a flat contact tip for the indicator (for example AMPG Z9365SS, 1/4 in, #4-48 thread as on the listed
indicator, about $8), abrasive paper and a flat plate, tax and freight. The lever's fit is unverified (see the table). The
reference type is a Mitutoyo 2046A indicator, which would raise the total.

The indicator is used as a comparator: zeroed on the master, read within about +-0.06 mm of that zero, so its
full-travel accuracy does not enter; one division (0.01 mm) of repeatability is needed. Before trials, check it: each
measured spacer must read within one division of its micrometer height. An indicator that fails this check is not used.

- Loads: every part weighs under 0.1 kg; pressing the lever or turning the bezel takes well under 5 N.
- The close-up panel in the packed demo video (`closeup.py`, `pack_demo.sh`) is re-drawn from the run's logged state. The
  policy and the scorer do not use it.

## Files

- `task.py`: scene, live coupling (lever to spindle, spindle to needle, bezel held where turned), scorer and demo.
- `dial_read.py`: the demo's camera reader of the dial.
- `dcs_visuals.py`: visual-only meshes (stand, dial face, parts, trays); colliders are in `task.py`.
- `labkit.py`, `arena_assets_fix.py`: shared helpers (rigid parts, joints, surfaces).
- `test_scorer.py`: 37 scorer cases. `closeup.py`, `pack_demo.sh`, `compose_setup.py`: video and setup-image packing.
