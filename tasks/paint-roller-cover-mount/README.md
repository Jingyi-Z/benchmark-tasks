# Slide a 9-inch paint-roller cover onto a loose cranked roller frame

## Trial commands

Activate the Isaac Lab / `yam_arena` environment and run from the repository root:

```bash
# Scripted demonstration (900-step limit = 45 s at 20 Hz; completes in ~22.5 s)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/paint-roller-cover-mount/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Slide the paint-roller cover all the way onto the roller frame's cage until it seats past the retaining clip." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/paint-roller-demo

# Dream-1 baseline policy evaluation (900 steps = 45 s at 20 Hz)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/paint-roller-cover-mount/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Slide the paint-roller cover all the way onto the roller frame's cage until it seats past the retaining clip." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/paint-roller-dream
```

Run the offline 578-hull `YAM_URDF` trajectory, swept-part collision, hollow-bore fit, and 15-case scorer preflight check:

```bash
python tasks/paint-roller-cover-mount/check_path.py
```

## Runtime and seeds

- Control rate: `20 Hz` (`900` steps = `45 s` maximum; scripted demonstration completes in ~`22.5 s`).
- Wall-clock runtime per trial on a single workstation GPU: ~`45 s` (demo), ~`90 s` (Dream-1).
- Deterministic fixed starting layout (`object_jitter=0`); no random seed configuration is required.

## Physical basis and disclosed approximations

- **Geometry & SKU reference:** Modeled 1:1 on a Wooster `Sherlock R017-9` `9-inch` cage paint-roller frame (`5/16-in` / `7.9 mm` 90-degree chrome-plated steel shank, fiberglass-reinforced nylon cage, and green quick-release retaining spring; https://www.woosterbrush.com/roller-frames/sherlock-frames/ ) and a standard `9 in x 1-1/2 in` (`228.6 mm` length, `38.1 mm` ID core, `57.2 mm` OD `3/8-in` nap) cylindrical paint-roller cover.
- **Disclosed estimates:** The hollow roller cover is modeled as a rigid 16-stave polygonal tube collider (`38.1 mm` inner bore, `57.2 mm` outer diameter; nap compressibility is omitted). Component masses (`0.34 kg` frame, `0.085 kg` cover), phenolic core wall, 5-rib cage profile, tapered nose cap, quick-release spring leaf stiffness (`1400 N/m`), cage bearing friction (`0.002 N m`), and Coulomb friction (`0.55`) are engineering estimates.

## License

Robot, station table, and `yam_arena` utilities derive from `yam-isaac-lab-arena` (`Apache-2.0` / `BSD-3-Clause`). Task scene and scoring code in this folder are released under `Apache-2.0`.
