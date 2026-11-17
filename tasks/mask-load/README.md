# Swap the layer 1 chrome mask for layer 2 on a mask aligner's holder, chrome side down

A contact mask aligner's mask holder lies pulled out and upright on its loading stand. It still holds the LAYER 1 chrome
mask from the last exposure, chrome side down, under the MASK VACUUM (its toggle is ON). Behind it, a two-shelf mask rack
holds the LAYER 2 mask on its lower shelf, lying chrome side up; the upper shelf is LAYER 1's, empty.

The masks are 5 inch soda-lime plates with a chrome pattern on one face. Both faces look alike; the only cue to which
face carries the chrome is the label, written in chrome, which reads correctly from the chrome side and mirrored through
the glass. In contact printing the chrome faces the wafer, so a mask goes on the holder chrome side down, against the
holder's stop pins.

## Real-trial mask holder and vacuum

The holder holds a mask by vacuum through a closed groove under the mask's flat border, as a contact aligner's mask
holder does. Everything is off the shelf or cut from the DXF files that `python make_holder.py` writes (any laser- or
water-jet-cutting service). Approximate cost: USD 150 to 220.

| Part | What to use | Approx. USD |
|---|---|---|
| Holder base | `holder_base_5mm_al6061.dxf`: 6061 aluminium, 5 mm, 160 x 100 mm with the 104 x 60 mm window, two 6 mm pin holes and a 2 mm port hole | 25 to 40 |
| Groove layer | `holder_top_outer_1mm_al6061.dxf` and `holder_top_inner_1mm_al6061.dxf`: 6061 aluminium, 1 mm, bonded to the base with thin structural epoxy (or vacuum-grade double-sided tape), leaving a closed 2 mm-wide groove 1 mm deep that runs 56 to 58 mm out along the length and 38 to 40 mm out across | 15 to 25 |
| Stop pins | Two 6 x 20 mm steel dowel pins, pressed in so 6 mm stands above the top | 5 |
| Column | 60 x 50 mm aluminium rectangular tube or extrusion, cut so the holder's top is 110 mm above the bench | 15 |
| Port fitting | M5 to 4 mm hose barb (brass or nylon), into the base from below over the port hole | 5 |
| Vacuum pump | 12 V mini diaphragm vacuum pump, -60 kPa or better, about 2 to 5 L/min, with a 12 V supply | 25 to 40 |
| Toggle (the MASK VACUUM switch) | Three-way, two-position toggle valve for 4 mm tube (common port to the holder; ON connects the pump, OFF closes the pump side and vents the holder to air), in a 30 x 24 x 30 mm box, lever tilting forward for ON | 20 to 35 |
| Tubing | 4 mm ID silicone or PU tube, about 1 m, and a small inline filter before the pump | 10 |
| Seal | The ground glass mask on the flat aluminium top, around the closed groove; a thin bead of vacuum grease is optional. No O-ring: the mask must sit flat and rest against the pins. | 0 to 5 |

**Assembly.** Cut and deburr the parts; bond the two 1 mm frames on the base, aligned on the window and pin holes, so
the gap between them forms the groove over the port hole; press in the pins; tap the port M5 from below and fit the
barb; mount the holder on its column; tube barb -> filter -> toggle valve (common port to the holder) -> pump; plug in
the pump's supply.

**Acceptance check (before each session).** Toggle ON with a mask on the holder against the pins: the mask must not
move when pushed sideways by hand with a light touch (about 2 N). Toggle OFF: the mask must lift freely within 2 s
(the valve vents the groove). This checks the built holder; the simulation models the vacuum only as the toggle's
state.

**Reset for each trial.** Toggle OFF; put LAYER 1 back on the holder, chrome side down, against both pins; toggle ON;
put LAYER 2 back on the lower shelf, chrome side up, label readable; the upper shelf empty.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The holder, rack and toggle are fixed; the masks
  are loose. An `on_step` hook latches the violations (a fingertip at a mask's patterned area; a mask dropped; LAYER 1
  moved off the holder while the vacuum was not off) and checks that everything has been still for the last second.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `make_holder.py`: writes the DXF cutting files for the real-trial mask holder (see Real-trial mask holder and vacuum).
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The right hand switches the mask vacuum off. Both hands
pinch LAYER 1's two side edges where they overhang the holder, lift it and set it on the upper shelf. They take LAYER 2
from the lower shelf, draw it out, lift it, roll it over together about the line between them, lower it onto the holder
just in front of the stop pins and push it back against them, trimming their aim by the mask's seen position. The
right hand switches the vacuum on. The demo reads only what the cameras see: the masks' poses. Run one trial per
process.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
