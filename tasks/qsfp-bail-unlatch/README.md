# Unlatch and remove a pull-tab QSFP28 optical transceiver from a 1U rack switch

## Overview

A loose EIA-310 19-inch two-post 5U desktop test rack stand (`2.40 kg`) on the counter mounts a full-depth 1U leaf-switch chassis (`439.0 x 44.0 x 406.0 mm`, `8.80 kg`; `11.20 kg` combined assembly) at U3 with one SFF-8661 Type 2B pull-tab QSFP28 optical transceiver (`18.35 x 8.50 x 72.0 mm`, `0.045 kg`) seated `60.0 mm` inside its EMI cage (`19.6 x 9.6 mm` inner tunnel). Dual spring-loaded side-latching pawls inside the cage engage flat 90-degree shoulders on the transceiver housing, so pulling the exposed front zinc collar without actuating the pull-tab leaves the module locked inside the cage. Pulling the protruding release pull-tab slides its coupled 28-degree cam wedges forward `6.0 mm` onto recessed parallel dwell lands to hold both cage pawls open and withdraw the transceiver cleanly, then set it gently on the counter.

## Running and scoring a trial

Activate the Isaac Lab / `yam_arena` environment and run from the repository root:

```bash
# Scripted demonstration (reaches 1.0 in ~14 seconds / 280 steps)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/qsfp-bail-unlatch/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Use the transceiver's release pull-tab to unlatch and remove the optical transceiver from the switch port, and leave it resting on the counter." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/qsfp-demo

# Dream-1 baseline policy evaluation (900 steps = 45 s at 20 Hz)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/qsfp-bail-unlatch/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Use the transceiver's release pull-tab to unlatch and remove the optical transceiver from the switch port, and leave it resting on the counter." \
  --scorer task_score --max-steps 900 --store-frames --log-dir logs/qsfp-dream
```

## Runtime and determinism

- **Control rate:** 20 Hz (`run_limit`: 900 steps = 45 seconds).
- **Runtime per trial:** ~30 seconds for the scripted demo and ~85 seconds for a 900-step Dream-1 evaluation on a workstation GPU.
- **Seeds / layout:** Fixed deterministic starting layout (`object_jitter=0`); no random seed configuration is required.

## Disclosed approximations

- Exterior module dimensions follow SFF-8661 / Coherent FTLC9558REPM (`18.35 x 8.50 x 72.0 mm` die-cast housing with a `28.0 mm` protruding pull-tab) seated in an Arista 7050CX3-32S-scale 1U leaf-switch chassis (`439.0 x 44.0 x 406.0 mm`, `8.80 kg`) on a two-post EIA-310 19-inch (`482.6 mm`, `2.40 kg`) desktop test rack stand.
- Internal dual side-pawl cam angles (`28 deg`), `6.0 mm` pull-tab release stroke, spring stiffnesses, `0.1 mm` contact offset, geometry-bounded collider torsional patch radii (`0.5 * min(size)`), and the omitted optical patch cord are engineering estimates rather than manufacturer internal CAD.
