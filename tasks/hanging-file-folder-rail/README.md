# Clamp a hanging file folder around a packet and hang it on frame rails

## Trial commands

Activate the Isaac Lab / `yam_arena` environment and run from the repository root:

```bash
# Scripted demonstration (900-step limit = 45 s at 20 Hz; completes in ~15 s)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/hanging-file-folder-rail/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Hang the file folder on both top rails of the file frame with the document packet still inside it." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/hanging-file-demo

# Dream-1 baseline policy evaluation (900 steps = 45 s at 20 Hz)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/hanging-file-folder-rail/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Hang the file folder on both top rails of the file frame with the document packet still inside it." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/hanging-file-dream
```

Run the offline 289-hull `YAM_URDF` trajectory, robot-vs-frame/folder collision, OBB cross-section, and 15-case scorer preflight check:

```bash
python tasks/hanging-file-folder-rail/check_path.py
```

## Runtime and seeds

- Control rate: `20 Hz` (`900` steps = `45 s` maximum; scripted demonstration completes in ~`15 s`).
- Wall-clock runtime per trial on a single workstation GPU: ~`40 s` (demo), ~`80 s` (Dream-1).
- Deterministic fixed starting layout (`object_jitter=0`); no random seed configuration is required.

## Physical basis and disclosed approximations

- **Geometry & SKU reference:** Modeled 1:1 on an Officemate `91990` letter-size steel hanging file frame set to its minimum `14 in` (`355.6 mm`) depth (`355.6 mm` D x `317.6 mm` center-to-center rail span x `238 mm` rail height; https://victortech.com/products/twin-pack-hanging-file-frame ) and a Pendaflex `4152` letter-size reinforced hanging file folder (`328 mm` coated-steel hanger-rod span, `298.5 x 230 mm` pressboard leaves, `0.16 kg`; https://www.tops-products.com/pendaflexr-reinforced-hanging-folders-letter-size-standard-green.html ) enclosing a loose `280 x 195 x 6.0 mm` (`0.20 kg`) manila document packet.
- **Disclosed estimates:** Fixed `200 x 130 x 32 mm` desktop inbox riser, pressboard leaves (`2.5 mm` thick), rail blade thickness (`3.6 mm` inside `7.2 mm` hook saddle), `9.2 mm` bottom V-gusset crease, un-sprung crease hinge damping (`0.02`), and Coulomb friction coefficients (`0.45-0.58`) are engineering approximations; paper bending flex is omitted.

## License

Robot, station table, and `yam_arena` utilities derive from `yam-isaac-lab-arena` (`Apache-2.0` / `BSD-3-Clause`). Task scene and scoring code in this folder are released under `Apache-2.0`.
