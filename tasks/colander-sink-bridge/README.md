# Pull out both handles of an over-the-sink colander and bridge it across a sink basin

Pull out both sliding stainless U-handles of a `0.77 kg` commercial/household expanding over-the-sink colander (`340 x 230 x 120 mm` collapsed, expanding up to `550 mm` across two `105 mm`-stroke sliding arms in crimped friction-slide collars with four downward rubber corner feet, modeled 1:1 on the `Norpro #2158` Expanding Colander with Base Frame) and seat both extended handles across the left and right rims of a `455 x 276 mm` open countertop sink basin so the basket hangs suspended `>= 15 mm` clear above the sink floor. External colander dimensions and mass follow `Norpro #2158` specifications (`340 x 230 x 120 mm` collapsed, `550 mm` expanded, `0.77 kg`); sink basin dimensions (`585 x 300 x 125 mm`, `455 x 276 mm` open basin), sub-masses, and `1.6 N` slide-collar friction are engineering estimates.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1000 --store-frames --log-dir logs
```

For the scripted demonstration, pass `--policy yam_scripted`.

## Runtime and seeds

- **Run limit:** `1,000` control steps (`50 s` at `20 Hz`; ~1.8 min wall-clock for scripted demo, ~3.5 min for Dream-1 on a single GPU).
- **Fixed starting layout:** `-E object_jitter=0`, no random seeds required.
- **Third-party assets:** None (procedural rigid-body primitives only).
