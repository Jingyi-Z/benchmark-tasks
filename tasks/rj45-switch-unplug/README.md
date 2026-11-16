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
- **Typical wall-clock runtime per trial:** Observed revision runs took 102 seconds for the scripted demo and 314 seconds for Dream-1 on the shared RTX 5090; runtime varies with GPU load.
- **Seeds & reset:** Fixed deterministic starting layout (`object_jitter = 0`); `task.py` scores purely from geometric end-state poses, independent `joint7`/`joint8` fingertip clearances, and 21-step (`1.0 s`) stillness.

## Disclosed Approximations
- Modem exterior (`57 x 184 x 200 mm` on a `74 x 184 x 7 mm` base foot, `0.523 kg`) follows the Motorola `MB8600` 4-port Gigabit cable modem dimensions and vertical rear port layout without claiming internal SKU CAD; exact `LAN 1` height (`118.0 mm`) is an estimate.
- Plug mating profile (`11.68 x 6.60 mm`) follows IEC 60603-7 and the `36.07 mm` plug-and-boot length follows Stewart Connector `SS-39200-085`, terminating in a rigid `14 x 6 mm` strain-relief collar; flexible Cat6 cable drag beyond the collar and 8P8C gold-pin spring wipe are omitted.
- Front-hinged locking tab stiffness (`0.12 Nm/rad`), damping (`0.004 Nms/rad`), `1.15 mm` bezel catch overlap, and contact friction (`0.55` / `0.75`) are representative engineering estimates, with PhysX torsional patch radii bounded to individual collider dimensions.



## Revision scoring and real cable reset

A fully disconnected connector resting anywhere on the counter earns 1.0 when the modem is upright, both hands are 80 mm clear and every moving part is still for 1 s. There is no rearward region or 35 mm separation requirement, and no location-based partial tier. The simulator tests disconnection against the authored LAN 1 insertion cavity; in real trials inspect that the entire mating tongue is outside the socket.

Use a 1.0 m Cat6 UTP patch cord, with only the tested end connected to LAN 1. The far end is free. Drape the cord behind the modem onto the counter near (0.28,-0.20)m; run it toward(0.58,-0.20), make a 100 mm-radius U to (0.58,-0.40), and lay the free end back toward(0.39,-0.40). Keep all slack on the counter and photograph this starting route on reset. This specifies a repeatable slack layout rather than a fixed or tensioned cable. The flexible tail remains omitted from simulation, as disclosed.

`python verify_revision.py` exercises 180 supported placements across four sides, three gaps including touching the modem, three plug yaws and five modem yaws, plus attached/partial, held, moving and tipped counterexamples. Scene, latch physics, controls and run limit are unchanged. Native revision validation: demo 1.0 (16.75s video), Dream-1 0.0 (45.05s), on the identical scene and policy. Both runs use this revised scorer. Demo final six-point clearance 137.48 mm and 21-sample displacement 0.0 mm; Dream ends with the plug still inserted.
