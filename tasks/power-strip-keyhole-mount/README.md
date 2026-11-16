# Hang a 6-Outlet Power Strip onto Workbench Keyhole Studs and Slide It Locked

Builds on the idea of TK-VAB5G6 / PR #694 and TK-N2JLLX / PR #2152, which were not accepted because the hung plate was an abstract holder rather than a realistic job-relevant object and a generic robot-cover label did not establish domain relevance; this version models a 1:1 Eaton Tripp Lite `TLP606` 6-outlet surge-protector power strip (`276.0 x 47.0 x 33.0 mm`, `0.460 kg`, six recessed 3-prong `NEMA 5-15R` receptacles, hinged illuminated `RESET/OFF` rocker switch, indicator LEDs, articulated `SJT` AC power cable with 3-prong `NEMA 5-15P` plug, and four rear keyhole slots on `101.0 x 28.0 mm` centers) starting flat outlet-face-up on the bench from scratch, scoring any securely engaged keyhole row or stable angle without arbitrary orientation vetoes.

## Overview

This task evaluates bimanual blind keyhole mounting at 1:1 scale: picking up a `276.0 mm` 6-outlet surge-protector power strip (`with articulated AC power cable and 3-prong NEMA 5-15P plug`) from a flat outlet-face-up bench pose, pitching it `90 deg` vertical to present its rear four-keyhole backplate (`101.0 mm` lengthwise x `28.0 mm` transverse centers) to a workbench riser backboard, aligning two `101.0 mm`-spaced blind rear openings simultaneously over two `#10` pan-head mounting studs occluded behind the `47.0 mm`-wide strip body (`+-1.4 mm` clearance per side, unlike exposed shelf hooks or open top-rail rods), pressing flush into the internal undercut pockets, and sliding the strip `12.4 mm` left along its longitudinal keyhole slots so both `9.6 mm` stud heads lock behind the `6.2 mm` neck flanges rather than seating under gravity alone.

## Run

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 --instruction "<see task.json>" --scorer task_score --max-steps 680 --store-frames --log-dir logs
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 --instruction "<see task.json>" --scorer task_score --max-steps 680 --store-frames --log-dir logs
```

One trial takes about 2.5 to 3 minutes on one GPU; the layout is fixed (no seeds). `STRIP_MOUNT_MODE=early_slide_push` (sliding +12.4 mm left before pressing onto the studs, physically blocked by the 6.2 mm neck flanges against the 9.6 mm stud heads, scores 0.0) and `STRIP_MOUNT_MODE=entry_only_no_slide` (seating onto both 12.4 mm entry openings without sliding left into the undercut necks, scores 0.5) run the two physical controls with `yam_scripted`. `python check_path.py` runs the offline 578-hull trajectory (including articulated cord and plug hulls), 3D keyhole undercut interlock, and 18-case scorer checks.

## License & Attribution

Helper utilities (`labkit.py`, `arena_assets_fix.py`, `hand_geometry.py`) adapt patterns from `yam-isaac-lab-arena` and Isaac Lab (`Apache-2.0` and `BSD-3-Clause`). All geometry, physics, and scoring code is authored from scratch from published Eaton Tripp Lite `TLP606` (`276.0 x 47.0 x 33.0 mm`, `0.460 kg`, `101.0 x 28.0 mm` keyhole spacing) specifications, with internal keyhole pocket, shortened articulated cord, and `#10` stud dimensions estimated.
