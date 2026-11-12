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

- 2,400 control steps (2 minutes of simulated time) per trial. The scripted demo finishes in about 80 s.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
