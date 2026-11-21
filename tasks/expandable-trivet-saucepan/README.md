# Expand a scissor pot stand and rest two pans on it

Pull open a folded six-bar coupled scissor pot stand (`252 x 76 x 20 mm` compact, `0.173 kg`, modeled on the `Joseph Joseph Stretch 70033` expandable silicone-nylon pot stand) symmetrically along `+-Y` to `>= 380 mm` (`446 mm` in the scripted demonstration) and place two cold, empty `26 cm` (`267 mm` rim OD, `215 mm` flat base, `0.520 kg`) enamelled steel two-handle pans (`Garcima 20226`) side by side onto the expanded stand so both pans sit level (`<= 5.0 deg` tilt, `14 to 26 mm` base seat height, `>= 10 mm` worktop clearance everywhere, `>= 265 mm` horizontal center separation) with both robot hand assemblies retreated `>= 80 mm` clear and all `9` visible reference marks moving strictly `< 5.0 mm` over a full `1.0 s` (`21`-frame) window at `20 Hz`.

Because two non-overlapping `267 mm`-rim circular pans require `>= 267 mm` horizontal center-to-center distance — strictly exceeding the `263.2 mm` compact diagonal of the folded `252 x 76 mm` stand — supporting both pans flat off the worktop is geometrically impossible without first expanding the scissor lattice. External envelope and mass follow `Joseph Joseph Stretch 70033` and `Garcima 20226` specifications; individual pot-stand rib width (`10.0 mm`), half-lap step height (`10.0 mm`) and `0.035 N*m` central hinge friction are provisional modeled estimates. The scripted demo uses a modeled 10 mm rib. At the planned 5.3° opening, the nominal adjacent-rib clearances are 5.4 mm (10 mm rib), 2.75 mm (16 mm) and 1.7 mm (20 mm). These are geometry sensitivity results; the stock grasp section and real tolerance are unmeasured.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

### Dream-1 policy evaluation (`1,200` steps)

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Expand the pot stand and set both pans on it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir logs/dream
```

### Scripted demonstration (`yam_scripted`)

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Expand the pot stand and set both pans on it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir logs/demo
```

## Runtime and seeds

- **Run limit:** `1,200` control steps (`60 s` at `20 Hz`).
- **Measured wall-clock runtime per trial (single RTX 5090 GPU):**
  - Scripted demonstration (`780` steps to `success()`, `score = 1.0`): `294 s` (`4.9 min`) headless on a shared GPU; `246 s` (`4.1 min`) with `640 x 352` 3-camera frame capture (`--store-frames`).
  - Dream-1 policy (`1,200` steps, `score = 0.0`): `323 s` (`5.4 min`) with `640 x 352` 3-camera frame capture (`--store-frames`).
- **Fixed starting layout:** `-E object_jitter=0`, deterministic, no random seeds required.
- **Third-party assets:** None (procedural rigid-body box and cylinder primitives only, built via bundled `labkit.py` and certified full-hand sphere cover `hand_cover_data.py`).

## Sourcing, simulation fidelity, and physical checks

- **Sourced product specifications:**
  - **Pot stand (`trivet`):** `Joseph Joseph Stretch 70033` expandable silicone-nylon pot stand (`252 x 76 x 20 mm` maker compact envelope, `0.173 kg` retail item mass; maker page de.josephjoseph.com/products/stretch-ausziehbarer-topfuntersetzer-schwarz lists SKU `70033`, `H2 x B25.2 x T7.6 cm`, silicone-nylon, heat resistant to `240 C`, EUR `22.00`; unboxed retail listing `203 x 51 x 19 mm` folded, expanding up to `550 x 140 x 19 mm`; ornamental US Design Patent `USD675055S1`).
  - **Pans (`pan_left`, `pan_right`):** Two `Garcima 20226` `26 cm` (`10.5 in`) enamelled steel flat-bottom serving/paella pans (`267 mm` outer rim diameter, `215 mm` flat base diameter, `38 mm` wall height, `0.520 kg` each, two riveted steel loop side handles centered at `+-152 mm` with `322 mm` outer handle span and `43 mm` handle-bar center height), tested cold and empty (thermal transfer omitted).
- **Articulation, collision, and provisional geometry details:**
  - The closed-loop six-bar scissor pantograph is modeled as an open-tree PhysX articulation rooted on `A_0` with one leader revolute hinge (`j_b0`, range `[0, 86 deg]`) and four `PhysxMimicJointAPI` follower hinges (`gearing = -1.0` on same-sign `B` bars, `gearing = +1.0` on opposite-sign `A` bars, `+-0.10 rad` follower limit buffer).
  - Individual pot-stand rib width (`10.0 mm`), half-lap step height (`10.0 mm`) and central hinge rivet friction (`0.035 N*m`) are provisional modeled estimates.
  - **Scripted demonstration method:** both hands, fingers closed and turned across the ribs, press on the rib tops and push the folded stand about `5 deg` open; each hand then lifts, opens to a `29 mm` jaw gap and regrasps an end rib at the planned `5.3 deg` pose (a fixed, open-loop pose; the demo reads no simulator state), pulls the stand open to about `446 mm`, releases, and places both pans by their handles.
  - The scripted demo uses a modeled 10 mm rib. At the planned 5.3° opening, the nominal adjacent-rib clearances are 5.4 mm (10 mm rib), 2.75 mm (16 mm) and 1.7 mm (20 mm). These are geometry sensitivity results; the stock grasp section and real tolerance are unmeasured.
  - Each pan uses `24` rigid collider primitives (`16`-segment beveled steel rim wall, `2` base cylinders, and `2` three-box loop side handles).
  - Full-hand `80 mm` release is evaluated across all `21` consecutive frames (`1.0 s` at `20 Hz`) using a `424`-sphere (`212`/hand, `14.35 mm` radius) conservative cover over both grippers, fingertips, and wrist `D405` camera housings against exact analytical oriented box and cylinder solid distances, paired with `9` visible reference marks for `< 5.0 mm` stillness.
- **Real-world physical measurement tools:** Tape measure / steel ruler (`>= 380 mm` stand length, `<= 60 mm` off-axis, `>= 30 mm` end inset, `>= 265 mm` pan center spacing, `>= 80 mm` hand clearance), `10 mm` feeler block (`>= 10 mm` worktop clearance under entire pan), `14 to 26 mm` seat-height gauge, `5.0 deg` inclinometer or side photo, and a `1.0 s` (`20 Hz`) overhead/side video check.

## Purchase route and real-world trial procedure

- **Buy:** one Joseph Joseph Stretch `70033` expandable pot stand (maker shop, EUR `22.00`, page above) and two Garcima `20226` `26 cm` enamelled steel paella pans (retail listings under MPN `20226`, about EUR `9` each, e.g. ebay.de item `303625286670`).
- **Before trials:** fold the stand closed and measure with calipers the end-rib width and height at the grasp section (about `27 mm` in from the rib tip) and the opening angle the closed-finger push actually reaches. Record both. Check that the fixed regrasp pose clears the neighbouring ribs on that measured section before running.
- **Each trial:** stand folded at the listed start pose, pans cold and empty at their start poses, then run the instruction for up to `1,200` control steps. Score with the tools below.

## Closest existing tasks

- `#1430` (egg on a fixed, already-open trivet), `#3129` (handle sleeve onto a static trivet), `#1814` (air-fryer basket onto a static pad) and `#2042` (removable handle attachment) all use a static support.
- This task first has to change the support's geometry, by expanding a coupled scissor lattice with both hands, before two pans can fit side by side. A folded stand cannot hold both pans, since their rims need `>= 267 mm` between centres against a `263.2 mm` compact diagonal.
