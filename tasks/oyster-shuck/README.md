# Shuck an oyster

A cold Pacific oyster lies cup side down on a hardwood shucking board, hinge toward the right arm, with an oyster knife
beside it and a bowl for empty shells on the bench. One hand presses the oyster down on the board by its lip end, as a
towelled hand does, keeping it level and still; the other picks up the knife, works its tip into the hinge and twists
to pop it, slides the blade flat under the top shell through the muscle, draws it out and lays the knife back down.
Then the first hand lifts the top shell off into the shell bowl.

What it tests: tool use in two physical stages on a small held object (a twist in a 4 mm gap, then a flat sweep under
a shell), with the other hand holding the work level and out of the knife's line.

Why two hands: the oyster must be held still and level while the knife pushes and twists against it, and the holding
hand must stay 3 cm or more from the knife's tip.

## How it is modelled

`yam_arena` task files place single primitives; the board, the shells, the knife and the bowl here are rigid bodies
built with `bodykit.py`, reused unchanged from the accepted ice-cube tray task
(`tasks/twist-an-ice-cube-tray-and-tip-the-cubes-into-a-dish`).

- The shells: the cup is a floor and a wall round an oyster-shaped outline (oval, narrowing to a pointed hinge end),
  the wall 4 mm lower for 24 mm at the hinge: the gap the knife goes into. The top shell is a flat plate of the same
  outline, 4 mm larger all round, resting on the rim, with a 10 mm muscle body hanging under it.
- The hinge ligament: a hinge joint along the hinge end that PhysX breaks past 6 N or 0.03 N m. A 2 mm blade pushed
  straight into the 4 mm gap doesn't load it; its 8 mm tip turned 25 to 30 degrees wedges the top shell up and breaks
  it (the pop). The muscle: a ball joint at the muscle's top that PhysX breaks past 2 N, which the blade reaches by
  sweeping through the muscle body. The task file never switches either joint off. Contact between the shells stays
  on (`task.py` sets the joints' `collisionEnabled`, which USD turns off by default).
- Scoring reads what a camera could see: the hinge end lifting 1.5 mm off the rim while the shell's centre stays seated
  (popped); the knife's tip in the muscle after that (cut); the top shell off the cup. Lifting the top shell 5 mm
  before both counts as torn off (score 0), so pulling it off by force scores nothing.
- The liquor is not simulated: the cup tipping past 20 degrees at any time counts as spilled (score 0).

The scripted demo looks twice at what a camera could see: where the knife sits in the hand once picked up, and where
its tip is when held still 30 mm and 8 mm short of the hinge; it shifts the hand by the tip's miss (the arm holds the
hand a few degrees off its commanded turn, which moves a tip 115 mm away by about 10 mm).

## Run and score a trial

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/oyster-shuck/task.py \
      -E object_jitter=0 -E control_hz=20 --instruction "$(python -c "import json;print(json.load(open('tasks/oyster-shuck/task.json'))['instruction'])")" \
      --scorer task_score --max-steps 1800

The scorer is `task_score` (see `score()` in `task.py` and `scoring` in `task.json`). One trial is 1,800 control steps
(90 s at 20 Hz); the scripted demo reaches full score in about 47 s. No seeds: the layout is fixed
(`object_jitter=0`). Set `OYSTER_DEBUG=1` to print the scorer's state every half second.
