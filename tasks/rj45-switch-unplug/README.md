# Unplug a latched Ethernet cable from a desktop modem

## Overview
A bimanual manipulation benchmark task (`rj45-switch-unplug`) where one arm counterholds an unfixed vertical 4-port Gigabit desktop cable modem (Motorola `MB8600` exterior envelope) while the other arm pinches and depresses an IEC 60603-7 RJ45 plug's spring-return locking tab to withdraw the connector axially from `LAN 1` and place it on the counter.

## Running and Scoring a Trial

From the benchmark environment (`yam.env` sourced):

```bash
# Run scripted demonstration (900 control steps = 45 s at 20 Hz)
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 --instruction "Unplug the Ethernet connector from the desktop modem and leave both resting on the counter." --scorer task_score --max-steps 900 --store-frames --log-dir logs/demo

# Run Dream-1 model evaluation (900 control steps = 45 s at 20 Hz)
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 --instruction "Unplug the Ethernet connector from the desktop modem and leave both resting on the counter." --scorer task_score --max-steps 900 --store-frames --log-dir logs/dream
```

Optional physical negative-control probes (`RJ45_PROBE=unpressed_unbraced`, `unpressed_braced`, or `early_release`) and diagnostic logging (`RJ45_DEBUG=1`) can be passed as environment variables.

## Runtime and Determinism
- **Control horizon:** `900` steps (`45` seconds at `20 Hz`).
- **Typical wall-clock runtime per trial:** 35 to 50 seconds on an RTX 5090.
- **Seeds & reset:** Fixed deterministic starting layout (`object_jitter = 0`); `task.py` scores purely from geometric end-state poses, independent `joint7`/`joint8` fingertip clearances, and 21-step (`1.0 s`) stillness.

## Disclosed Approximations
- Modem exterior (`57 x 184 x 200 mm` on a `74 x 184 x 7 mm` base foot, `0.523 kg`) follows the Motorola `MB8600` 4-port Gigabit cable modem dimensions and vertical rear port layout without claiming internal SKU CAD; exact `LAN 1` height (`118.0 mm`) is an estimate.
- Plug mating profile (`11.68 x 6.60 mm`) follows IEC 60603-7 and the `36.07 mm` plug-and-boot length follows Stewart Connector `SS-39200-085`, terminating in a rigid `14 x 6 mm` strain-relief collar; flexible Cat6 cable drag beyond the collar and 8P8C gold-pin spring wipe are omitted.
- Front-hinged locking tab stiffness (`0.12 Nm/rad`), damping (`0.004 Nms/rad`), `1.15 mm` bezel catch overlap, and contact friction (`0.55` / `0.75`) are representative engineering estimates, with PhysX torsional patch radii bounded to individual collider dimensions.
