# Unlock a travel mug's safety switch and pop open its spring-loaded flip lid

Steady a 1:1 insulated stainless travel mug (`Zojirushi SM-KHE48`, `67 x 73 x 232 mm`, `0.55 kg` filled with `300 mL` beverage) while sliding its front horizontal safety lock switch (`10.5 mm` stroke) clear of the push-button channel and pressing the inward spring-returned release button (`6.5 mm` stroke) so the torsion-spring flip lid pops open past `110 degrees` (`135.2 degrees` stop), leaving the mug standing upright on the counter. Outer dimensions (`67 x 73 x 232 mm`, `0.25 kg` empty mass) and slide-blocked push-button flip-lid topology follow the official Zojirushi `SM-KHE48` specification; internal liner floor (`12 mm`), spout lip (`208 mm`), catch travel/clearances, and spring rates (`60 N/m` button spring, `0.045 N m/rad` lid spring) are engineering estimates.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs
```

For the scripted demonstration, pass `--policy yam_scripted`.

## Runtime and seeds

- **Run limit:** `900` control steps (`45 s` at `20 Hz`; ~1.5 min wall-clock for scripted demo, ~3.2 min for Dream-1 on a single GPU).
- **Fixed starting layout:** `-E object_jitter=0`, no random seeds required.
- **Third-party assets:** None (procedural rigid-body primitives only).
