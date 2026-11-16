# Gold lift-off on a wet bench: dedicated beaker, rinse order, and catching every drip

Two 2 inch wafers have just come out of gold evaporation, gold over patterned photoresist. They lie pattern side down
in a small PFA lift-off cassette on a PTFE tray. On the bench stand four 800 ml beakers in a row, each with a label card
in front of it and a label on its side: ACE (acetone for ordinary resist strip), LIFT-OFF (acetone kept for lift-off
only, standing in a running ultrasonic bath), IPA and DI WATER. The two acetone beakers look the same and hold the
same liquid. The film (5 nm Ti / 30 nm Au on 1.4 um image-reversal resist) lifts off in about 60 s in the bath. In the simulation the film is drawn
in 16 patches that come away one by one while the cassette is in the running bath; taken out early, or left to dry
before IPA, patches stay on as residue and the lift-off counts as failed. A PTFE drip dish rests on its
stand, and the bench's 4 L solvent waste bottle has a funnel.

The task follows the usual wet-bench practice for lift-off:

- The cassette soaks in the LIFT-OFF beaker only. Gold flakes must never reach the general acetone, so the wet cassette
  never goes into, or drips over, the ACE beaker.
- It goes into IPA before the acetone can dry on the wafers, since drying redeposits flakes. Then it goes into DI
  water, where it is left standing.
- Whenever the wet cassette is out of the liquid and not over a beaker's mouth, it drips: the drip dish must be under
  it. Afterwards the dish is emptied into the waste bottle's funnel, where all solvent and rinse waste goes, and put back
  on its stand.

One hand carries the cassette while the other keeps the dish under it, so the task needs both arms.

Real trials run in a ventilated wet bench or fume hood rated for flammable solvents, with everything standing in a
spill tray lined with absorbent paper. Afterwards the wafers are checked under a microscope against the mask's pattern,
and any drip or spill outside the beakers, the dish or the funnel scores 0.

## Preparing the wafers (real trials)

Each trial uses two identical pre-patterned wafers. Make them in one batch (for example 20 wafers for 10 trials) so
every trial starts the same. Source: any facility of the US National Nanotechnology Coordinated Infrastructure (NNCI,
nnci.net) takes outside users. For example, the University of Washington's Washington Nanofabrication Facility (WNF)
offers a "Remote" user class whose limited-scope process work is done by WNF staff on a time-and-materials basis, with
USD 50 domestic shipping; a lab with its own trained user can run the same steps itself at academic rates.

1. Wafers: 2 inch (50.8 mm) test-grade silicon, any orientation and doping, about 280 to 500 um thick, from any wafer
   vendor (about USD 10 to 20 each).
2. Mask: one transparency photomask (film printed by a photoplotting service, under about USD 100) with a simple test
   pattern, for example 1 mm square pads and 50 um lines and gaps, no feature under 20 um. The same mask is the
   reference pattern for the microscope check.
3. Lithography: AZ 5214E image-reversal resist spun to about 1.4 um, soft bake, mask exposure, reversal bake, flood
   exposure, develop in AZ 726 MIF, following the MicroChemicals AZ 5214E datasheet. This gives the undercut profile
   that lift-off needs.
4. Metal: 5 nm Ti then 30 nm Au by e-beam (or thermal) evaporation, all wafers in one run.
5. Store the batch in a closed wafer box at room temperature, away from light, and use within about two weeks.

Batch cost (20 wafers, 10 trials), at WNF's published rates effective 1 March 2026 for external academic users
(spin/photoresist bench and contact aligner USD 63/hr, e-beam evaporator USD 89/hr, staff USD 89/hr, 15.6% overhead;
precious metals at cost plus overhead):

| Item | Estimate (USD) |
|---|---|
| 20 wafers, 2 inch test-grade Si | 200 to 400 |
| Transparency mask (one-time) | 50 to 100 |
| Resist, developer, solvents | 20 to 40 |
| Spin bench 2 h + aligner 2 h | 252 |
| E-beam evaporator 3 h (one run, all 20 wafers) | 267 |
| Gold and titanium (at cost) | 50 to 150 |
| Staff time, about 6 h (remote service only) | 534 |
| Overhead 15.6% on facility charges, shipping | about 220 |
| Total with staff service | about 1,600 to 2,000 |
| Total if the lab's own trained user does it (no staff time, no shipping) | about 900 to 1,300 |

So about USD 80 to 100 per wafer, or USD 160 to 200 per trial, with staff service (USD 90 to 130 per trial without).

Before a batch is used for trials, lift off one wafer from it as in the task (60 s in acetone in the ultrasonic bath) and check it under the microscope.

Protect the robot: fit solvent-resistant covers (for example nitrile finger cots) over both grippers' pads for real
trials. The grippers touch only the PFA cassette's handle and the PTFE dish's tab, never the liquids.

Reset between trials: two fresh wafers into the cassette, pattern side down; replace the LIFT-OFF acetone, which now
holds gold flakes, and the IPA and DI water; pour the dish and funnel waste into the solvent waste bottle; dry the
cassette and dish. The used wafers are kept for the microscope check.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The beakers, cards, tray, stand and waste bottle
  are fixed; the cassette and the dish are loose. An `on_step` hook records the cassette's baths and latches the
  violations, and checks that the cassette and dish have been still for the last second.
- `labkit.py`: parts built from several shapes. It hooks `yam_arena.scene` and the embodiment from the task's import,
  so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 3600 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`.
- The right hand takes the cassette by its handle, with the hand tilted 40 degrees back. It lifts the cassette clear of
  the rims, soaks it in LIFT-OFF, then carries it to IPA and to DI WATER, lets go and moves away.
- The left hand takes the dish by its tab and slides it along just above the rims. The dish is under the cassette
  wherever the cassette crosses between two beakers, and moves out of the way of each descent. Then the left hand pours
  the dish into the funnel and puts it back on its stand.
- Before each descent the right hand holds still and trims its aim by the cassette's seen position. The demo reads only
  what a camera sees: the cassette's pose.

Run one trial per process. The score is `task_score` in the log, computed by `score()` in `task.py`.

## Runtime and seeds

- 3,600 control steps (3 minutes of simulated time) per trial. The scripted demo finishes in about 80 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
