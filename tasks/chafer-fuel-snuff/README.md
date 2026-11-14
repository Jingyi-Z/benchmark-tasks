# Cover and remove a chafing fuel holder from under a buffet stand

A cold commercial 8-quart stainless chafer stand holds a loose handled chafing-fuel holder inside its lower burner support ring with the horizontal snuffer damper swung 110 degrees open toward `-Y`. Because the open `L`-shaped assembly spans at least `156 mm` at every horizontal yaw angle, it physically jams against the stand's `144 mm` clear front leg opening if pulled out open, while pushing the snuffer tab one-handed spins the loose round cup inside the ring. The robot braces the protruding front handle with one hand while swinging the snuffer damper concentric over the cup with the other (`narrowing the width to 94 mm`), then lifts the covered holder out of the burner ring, withdraws it between the front legs, and sets it upright inside the stainless bussing pan.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1100 --store-frames --log-dir logs
```

For the scripted demonstration, pass `--policy yam_scripted` (`840` steps / `42.0 s`).

## Runtime and seeds

- **Run limit:** `1,100` control steps (`55 s` at `20 Hz`; ~2.0 min wall-clock for scripted demo, ~4.0 min for Dream-1 on a single GPU).
- **Fixed starting layout:** `-E object_jitter=0`, no random seeds required.
- **Third-party assets:** None (procedural rigid-body primitives only).
