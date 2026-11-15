# Fit, plunge, and turn the safety lid closed on a filled French press

Brace the handle of a loose `0.795 kg` filled `0.5`-litre (`4`-cup) Bodum Chambord cylindrical French press carafe (`96 mm` OD x `125 mm` glass beaker, `166 mm` overall width with handle, `190 mm` pressed overall height) on its cork coaster with one hand while the other hand picks up the `0.228 kg` plunger-lid assembly by its central guide collar from the drip stand, seats it coaxially onto the carafe rim with the one-piece safety lid rotated `90 degrees` so its solid pilot skirt closes the `-Y` pour spout (turning the lid closed either before or after pressing is valid per the Bodum Chambord manual), and presses the plunger smoothly down (`<= 110 mm/s`) through joint friction and viscous filtration damping (dimensionless legacy PhysX `jointFriction = 2.4` + `85 N s/m`). External carafe, whole-lid spout-closure topology, and assembled press dimensions follow official Bodum Chambord `0.5 L` (`1924-16` / `1928-57` manual / spare beaker `1504-10`) specifications; internal wall/skirt/dome/filter dimensions (`3 mm` glass wall, `82 mm` pilot skirt with `-Y` pour slot, `68 mm` stepped dome crown, `78 mm` filter disc, `22 mm` central collar, `36 mm` press knob), component sub-masses (`345 g` empty carafe, `148 g` safety lid, `80 g` plunger), and plunger joint parameters (dimensionless legacy `jointFriction = 2.4` + `85 N s/m` damping, not calibrated in absolute newtons) are engineering estimates.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir logs
```

For the scripted demonstration, pass `--policy yam_scripted`.

## Runtime and seeds

- **Run limit:** `1,200` control steps (`60 s` at `20 Hz`; ~2.0 min wall-clock for scripted demo, ~4.0 min for Dream-1 on a single GPU).
- **Fixed starting layout:** `-E object_jitter=0`, no random seeds required.
- **Third-party assets:** None (procedural rigid-body primitives only).
