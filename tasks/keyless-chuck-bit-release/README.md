Remove a clamped round-shank bit from an unpowered handheld keyless drill and leave it in the parts tray.

Run from this folder in the installed Isaac Lab-Arena YAM environment:

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 --instruction "Remove the drill bit from the unpowered drill and leave the bit in the tray." --scorer task_score --max-steps 1600 --store-frames --log-dir logs/demo

Replace yam_scripted with dream for a model trial in a separate log directory. Limit1600steps/80s at20Hz. Fixed layout, object_jitter=0, no random seeds. Planned controller lasts72s; the native demonstration terminates successfully after66.5s, about10minutes wall time for R1 on the shared5090. The retained prior Dream recording lasts80.05s; its original wall time was about5minutes.

The left hand holds the existing pistol grip; the right opens the hollow sleeve through six90degree strokes with release/withdraw/reset/regrasp. The prefix uses fixed development calibration. The right then grasps the exposed bit on its flat pads, extracts50mm and another30mm along the visibly observed axis, and waits while the left sets the drill down. Only afterward it transfers the bit over the tray and lowers it for release. Carry alignment uses the visible bit relative to the hand, not an assumed grasp angle. A conservative installed-hand envelope selects14mm bit-centre release height, leaving an8mm fall to the3mm tray floor. The demo reads camera-observable bit/tray poses and robot encoders; no contact forces, latch flags, attachments, injected forces or object-pose writes. Optional CHUCK_DEBUG only prints diagnostic state. camera-replay-d16c.json is development pose data for offline checks; native control uses current visible poses. hand-envelope.json contains conservative hand bounds, not a substituted collision mesh.

Revision R1 preserves the successful D18 physical scene and controller. It replaces the signed infinite nose plane with a finite chuck/bit separation test. Native R1 demo scored1.0 after1330steps (66.5s). Final whole-bit tray margin30.705mm and hand clearance100.739mm; support and21-sample stillness pass. R1 demo took about10minutes wall time on the shared5090.

The185x152x60mm exterior and0.87kg bare-tool mass follow DCD701 references. Sleeve42mm diameter,20degree jaw guides,2mm/rev ideal passive nut/sleeve transmission, mass split and20um initial interference are estimates. Legacy PhysX jointFriction0.15 is dimensionless/load-dependent, not an absolute0.15Nm brake. Force calibration is not measured; bit flutes, battery, spindle drive and damage are omitted. Original fixed-root retention controls validate the model, not commercial chuck forces.

Scoring is numeric in task.json. Removal means the entire bit is outside the finite42mm-diameter,40mm-long chuck envelope, in any drill orientation. A complete separating-axis test on circumscribed32-sided cylinders uses0.15mm tolerance, exceeding their combined radial enclosure error. Supported parts, independent hand endpoints/midpoints80mm clear,21-sample stillness and all tray margins remain required. No arbitrary20mm infinite nose-plane gap is required.

Repeatable real-world reset: With battery removed, insert the 6 mm bit 19.0 +/- 0.5 mm past the chuck nose. Counterhold the drill body and turn the sleeve clockwise as viewed from the bit tip, using a low-range torque driver with a nonslip strap adapter, until 0.50 +/- 0.05 N m is reached; hold that torque for 1 s, then remove the reset tool. This is a chosen trial-reset condition, not a manufacturer torque specification. Restore the stated drill and tray poses. The simulator retains its unchanged20um interference estimate; its grip has not been calibrated to this chosen real-world tightening torque.

Primary exterior reference: https://www.cee.dewalt.global/en-ee/product/dcd701d2-qw/12v-xr-brushless-sub-compact-drill-driver-2-2ah-batteries . Generic jaw/nut topology: https://patents.google.com/patent/US4682918A/en . Neither supplies commercial internal CAD. Task objects are procedural primitives/convex wedges; no third-party object meshes or scans. Installed YAM/Isaac Lab-Arena supplies robot and station.


Dream-1 run: the scene, physics and instruction are unchanged in this revision, so the earlier full Dream-1 run (1,600 steps) is kept. Its logged object poses were rescored with the revised scorer: the bit is never removed from the chuck, so the score stays 0.0.

## Reviewer-requested tray/orientation scorer checks (Nov 16)

Run this CPU-only check in the installed YAM environment:

    python check_tray_orientations.py

All **576 valid bit-in-tray cases receive full credit, 1.0**. The matrix covers 16 drill orientations: yaw 0, 45, 90, 135, 180, 225, 270, 315 degrees, each at roll 0 and 180 degrees. For each drill orientation it tests 36 tray placements: bit yaw 0, 45, 90, 135 degrees, with a 3x3 grid of centre/edge/corner positions that leaves at least 2 mm to every inner wall. The bit rests 3 mm above the tray base, matching the physical floor. The drill's height is set from its full geometry to touch the table. For each case the unchanged production Metric.update runs 21 identical observations and verifies removal, support, whole-bit containment, six-point hand clearance and stillness.

Only installed-hand runtime access is stubbed, with two distinct 12 mm-open jaws per hand placed clear of all parts; every object-scoring predicate is the submitted production implementation. These are scorer checks with prescribed static poses, not 576 native simulations or a proof of dynamic stability. They implement the review's condition that other scoring requirements are met.

| Drill yaw (deg) | Drill roll (deg) | Valid tray cases | Minimum score | Maximum score |
|---:|---:|---:|---:|---:|
| 0 | 0 | 36 | 1.0 | 1.0 |
| 45 | 0 | 36 | 1.0 | 1.0 |
| 90 | 0 | 36 | 1.0 | 1.0 |
| 135 | 0 | 36 | 1.0 | 1.0 |
| 180 | 0 | 36 | 1.0 | 1.0 |
| 225 | 0 | 36 | 1.0 | 1.0 |
| 270 | 0 | 36 | 1.0 | 1.0 |
| 315 | 0 | 36 | 1.0 | 1.0 |
| 0 | 180 | 36 | 1.0 | 1.0 |
| 45 | 180 | 36 | 1.0 | 1.0 |
| 90 | 180 | 36 | 1.0 | 1.0 |
| 135 | 180 | 36 | 1.0 | 1.0 |
| 180 | 180 | 36 | 1.0 | 1.0 |
| 225 | 180 | 36 | 1.0 | 1.0 |
| 270 | 180 | 36 | 1.0 | 1.0 |
| 315 | 180 | 36 | 1.0 | 1.0 |

Guard controls: bit outside tray 0.5; fewer than 21 still samples 0; a hand within 1 mm of the bit 0; 2 mm final-second bit motion 0; bit still in chuck 0. All five match their expected scores. The full CSV is tray-orientation-checks.csv; summary and source hashes are in tray-orientation-summary.json. The script regenerates the full JSON report and CSV. Minimum positive-case hand clearance is 518.877 mm; minimum inner-wall margin is 2.0 mm.

This evidence-only revision preserves the physical scene, controller, scorer, reset procedure and instruction from submitted 650061ff003499cacd36472f31e338a74441745f. Existing native demo 1.0 and retained completed Dream-1 regraded 0.0 remain the media evidence. No new native run is claimed or needed for these CPU checks.
