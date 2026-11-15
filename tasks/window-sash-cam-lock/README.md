# Close a single-hung vinyl window sash and turn its cam lock into the keeper

## Trial commands

Activate the Isaac Lab / `yam_arena` environment and run from the repository root:

```bash
# Scripted demonstration (900-step limit = 45 s at 20 Hz; completes in ~15 s)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/window-sash-cam-lock/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Push the raised lower window sash all the way down onto the sill and turn the meeting-rail cam lock into its keeper to pull the sash shut." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/window-sash-demo

# Dream-1 baseline policy evaluation (900 steps = 45 s at 20 Hz)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/window-sash-cam-lock/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Push the raised lower window sash all the way down onto the sill and turn the meeting-rail cam lock into its keeper to pull the sash shut." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/window-sash-dream
```

Run the offline 578-hull `YAM_URDF` trajectory, robot-vs-window collision, OBB/cylinder cross-section, and 15-case scorer preflight check:

```bash
python tasks/window-sash-cam-lock/check_path.py
```

## Runtime and seeds

- Control rate: `20 Hz` (`900` steps = `45 s` maximum; scripted demonstration completes in ~`15 s`).
- Wall-clock runtime per trial on a single workstation GPU: ~`40 s` (demo), ~`85 s` (Dream-1).
- Deterministic fixed starting layout (`object_jitter=0`); no random seed configuration is required.

## Physical basis and disclosed approximations

- **Geometry & SKU reference:** Modeled 1:1 on a Tafco `VSH1824B` `18 x 24 in` nominal (`17-1/2 x 23-1/2 in` / `444.5 x 596.9 mm` actual outer width/height; `2-1/2 in` / `63.5 mm` frame depth and rail wall thicknesses estimated) single-hung vinyl utility window (https://www.homedepot.com/p/TAFCO-WINDOWS-18-in-x-24-in-Single-Hung-Vinyl-Utility-Window-White-VSH1824B/203951015 and https://tafco.com/windows/single-hung-windows/ ) and a Prime-Line `F 2588` die-cast cam-action sash lock (`2-1/16 in` mounting hole centers with `5/16 in` keeper; https://www.primelineparts.com/sash-lock-2-1-16-in-hole-centers-fits-single-and-double-hung-vinyl-windows-diecast-white-single-pack/ ).
- **Disclosed estimates:** Glazing panes are omitted so overhead cameras view the meeting-rail lock clearly; jamb-balance PhysX joint-axis holding friction (`17.5 N` static / `16.0 N` dynamic), meeting-rail bulb weatherseal stiffness (`450 N/m`, `2.2 mm` preload), die-cast cam/keeper chordal box segments and `6.0 mm` cylindrical catch pin, rear bench-mounting brackets (`40 x 44 x 12 mm`), and Coulomb friction (`0.45`) are engineering approximations.

## License

Robot, station table, and `yam_arena` utilities derive from `yam-isaac-lab-arena` (`Apache-2.0` / `BSD-3-Clause`). Task scene and scoring code in this folder are released under `Apache-2.0`.
