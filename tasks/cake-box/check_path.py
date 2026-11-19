"""Offline CPU preflight for cake-box:
1. Static initial scene check (untouched start = 0.0, zero board-vs-box/front_panel overlap, zero robot READY overlap).
2. Comprehensive 14-case scorer witness suite covering 0.0, 0.25, 0.50, 0.75, 1.00 and every sticky-zero latch.
3. Full 630-hull dual-arm YAM_URDF trajectory IK & dense frosting-clearance sweep across all demo steps.
"""
from __future__ import annotations

import math
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task
from task_metrics import (
    FRONT_INIT,
    FROSTING_PTS_LOCAL,
    HAND_CLEAR_M,
    LID_INIT,
    R_BOARD,
    TAB_INIT,
    T_BASE,
    T_BOARD,
    W_IN,
    articulated_box_obbs_in_box_frame,
    box_panel_touches_frosting,
    evaluate_final_tier,
    get_robot_urdf_hulls,
    hand_clearance_m,
    robot_hulls_touch_frosting,
    ry,
    step_violation_metrics,
)
from yam_arena import YAM_URDF
from yam_arena import demo as _demo
from yam_arena import ik as _ik
from yam_arena import urdf as _urdf
from yam_arena.constants import LEFT_BASE_POS, READY, RIGHT_BASE_POS


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _fk_arm_world_hulls(arm_label: str, base_pos, q6: np.ndarray, finger: float):
    """Exact world convex hulls for all links of one YAM arm via yam_arena.urdf.forward_kinematics."""
    jmap = {f"joint{i + 1}": float(q6[i]) for i in range(6)}
    # In YAM URDF, joint7/joint8 prismatic range is [-0.04, 0.0] (or magnitude); pass signed finger in [-0.04, 0.0]
    f_val = -abs(float(finger))
    jmap["joint7"] = f_val
    jmap["joint8"] = f_val

    poses = _urdf.forward_kinematics(YAM_URDF, jmap)
    base_vec = np.asarray(base_pos, float)
    hulls = get_robot_urdf_hulls()
    out = []
    for link, T in poses.items():
        R = T[:3, :3]
        p = T[:3, 3] + base_vec
        for v, eq in hulls.get(link, []):
            w = v @ R.T + p
            nw = eq[:, :3] @ R.T
            out.append((f"{arm_label}:{link}", w, np.c_[nw, eq[:, 3] - nw @ p], w.min(0), w.max(0)))
    return out


def _board_min_dist_to_box_panels(bpos: np.ndarray, R_box: np.ndarray, pl_pos: np.ndarray) -> tuple[float, str]:
    """Minimum 3D distance (m) from 72 perimeter points of CAKE_BOARD to all CAKE_BOX panels (incl. front_panel)."""
    obbs = articulated_box_obbs_in_box_frame(FRONT_INIT, LID_INIT, TAB_INIT)
    best = float("inf")
    worst_name = ""
    for a in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
        for z in (0.0, T_BOARD):
            pw = pl_pos + np.array([R_BOARD * math.cos(a), R_BOARD * math.sin(a), z])
            p_box = R_box.T @ (pw - bpos)
            for name, c, R, h in obbs:
                d = float(np.linalg.norm(np.maximum(np.abs(R.T @ (p_box - c)) - h, 0.0)))
                if d < best:
                    best = d
                    worst_name = name
    return best, worst_name


def run_checks():
    print(f"Frosting surface samples: {len(FROSTING_PTS_LOCAL)} points (<= 3.2 mm spacing)")
    assert len(FROSTING_PTS_LOCAL) >= 4000

    I3 = np.eye(3)
    R_box = _rz(task.BOX_YAW)
    bpos = np.array([task.BOX_XY[0], task.BOX_XY[1], task.TOP])
    pl_start = np.array([task.CAKE_START_XY[0], task.CAKE_START_XY[1], task.TOP])
    ck_start = np.array([task.CAKE_START_XY[0], task.CAKE_START_XY[1], task.TOP + T_BOARD])

    # 1. Verify starting configuration (including board-vs-box/front_panel static gap AND robot READY gap!)
    board_box_gap, closest_panel = _board_min_dist_to_box_panels(bpos, R_box, pl_start)
    print(f"Initial CAKE_BOARD to CAKE_BOX ({closest_panel}) clearance: {board_box_gap * 1000.0:.2f} mm")
    assert board_box_gap >= 0.020, f"CAKE_BOARD too close to {closest_panel} at start: {board_box_gap * 1000:.2f} mm!"

    ready_hulls = _fk_arm_world_hulls("left", LEFT_BASE_POS, READY, -0.04) + _fk_arm_world_hulls("right", RIGHT_BASE_POS, READY, -0.04)
    ready_hit = robot_hulls_touch_frosting(ready_hulls, ck_start, I3, tol=0.050)
    assert ready_hit is None, f"READY arm within 50 mm of starting cake: {ready_hit}"

    t0 = evaluate_final_tier(bpos, R_box, FRONT_INIT, LID_INIT, TAB_INIT, pl_start, I3)
    assert t0["tier"] == 0.0 and not t0["board_inside"], f"Start must score 0.0, got {t0}"
    hit0 = box_panel_touches_frosting(bpos, R_box, FRONT_INIT, LID_INIT, TAB_INIT, ck_start, I3)
    assert hit0 is None, f"Start box panel touches frosting: {hit0}"
    sv0 = step_violation_metrics(task.TOP, bpos, R_box, FRONT_INIT, LID_INIT, TAB_INIT, pl_start, I3, ck_start, I3, np.array([0.0, 0.0, T_BOARD]))
    assert not (sv0["cake_disturbed"] or sv0["board_dropped"] or sv0["box_crushed"]), f"Start violation: {sv0}"
    print("[PASS] Initial scene configuration: score = 0.0, >50 mm clear of READY arms, board-to-wall nominal clearance =",
          round((W_IN / 2 - R_BOARD) * 1000.0, 2), "mm per side")

    # 2. Scorer witness suite
    pl_in = np.array([task.BOX_XY[0], task.BOX_XY[1], task.TOP + T_BASE])
    ck_in = np.array([task.BOX_XY[0], task.BOX_XY[1], task.TOP + T_BASE + T_BOARD])
    far_pads = [np.array([0.10, +0.38, task.TOP + 0.25])] * 3 + [np.array([0.10, -0.38, task.TOP + 0.25])] * 3

    witnesses = []
    w_025 = evaluate_final_tier(bpos, R_box, FRONT_INIT, LID_INIT, TAB_INIT, pl_in, I3)
    assert w_025["tier"] == 0.25, w_025
    witnesses.append(("board_inside_open_box", 0.25))

    w_050 = evaluate_final_tier(bpos, R_box, math.radians(-2.0), LID_INIT, TAB_INIT, pl_in, I3)
    assert w_050["tier"] == 0.50, w_050
    witnesses.append(("front_closed_lid_open", 0.50))

    w_050_gap = evaluate_final_tier(bpos, R_box, 0.0, math.radians(1.5), math.radians(-20.0), pl_in, I3)
    assert w_050_gap["tier"] == 0.50 and not w_050_gap["lid_closed"], w_050_gap
    witnesses.append(("lid_gap_6.7mm_exceeds_3mm", 0.50))

    w_075 = evaluate_final_tier(bpos, R_box, 0.0, math.radians(0.4), TAB_INIT, pl_in, I3)
    assert w_075["tier"] == 0.75 and w_075["lid_closed"] and not w_075["tab_tucked"], w_075
    witnesses.append(("lid_closed_tab_outside", 0.75))

    w_075_shallow = evaluate_final_tier(bpos, R_box, 0.0, 0.0, math.radians(-6.0), pl_in, I3)
    assert w_075_shallow["tier"] == 0.75 and w_075_shallow["tab_inward_depth_mm"] < 10.0, w_075_shallow
    witnesses.append(("tab_shallow_6.2mm_under_10mm", 0.75))

    for q_tab_deg in (-14.0, -20.0, -28.0, -35.0, -45.0, -52.0):
        qt = math.radians(q_tab_deg)
        wt = evaluate_final_tier(bpos, R_box, 0.0, math.radians(0.4), qt, pl_in, I3)
        ph = box_panel_touches_frosting(bpos, R_box, 0.0, math.radians(0.4), qt, ck_in, I3)
        assert wt["tier"] == 1.0 and ph is None, (q_tab_deg, wt, ph)
        witnesses.append((f"full_tucked_{int(q_tab_deg)}deg", 1.0))

    ph_lid_hit = box_panel_touches_frosting(bpos, R_box, 0.0, math.radians(-22.0), 0.0, ck_in, I3)
    assert ph_lid_hit is not None
    witnesses.append(("panel_touches_frosting", 0.0))

    sv_tilt = step_violation_metrics(task.TOP, bpos, R_box, 0.0, 0.0, math.radians(-16.0), pl_in, I3, ck_in, ry(math.radians(11.5)), np.array([0.0, 0.0, T_BOARD]))
    assert sv_tilt["cake_disturbed"]
    witnesses.append(("cake_tilted_11.5deg", 0.0))

    sv_slide = step_violation_metrics(task.TOP, bpos, R_box, 0.0, 0.0, math.radians(-16.0), pl_in, I3, ck_in + np.array([0.012, 0.0, 0.0]), I3, np.array([0.0, 0.0, T_BOARD]))
    assert sv_slide["cake_disturbed"]
    witnesses.append(("cake_slid_12mm_on_board", 0.0))

    sv_drop = step_violation_metrics(task.TOP, bpos, R_box, 0.0, 0.0, math.radians(-16.0), pl_in + np.array([0.0, 0.0, 0.025]), I3, ck_in + np.array([0.0, 0.0, 0.025]), I3, np.array([0.0, 0.0, T_BOARD]))
    assert sv_drop["board_dropped"]
    witnesses.append(("board_lifted_25mm_drop", 0.0))

    sv_crush = step_violation_metrics(task.TOP, bpos, R_box, math.radians(12.0), 0.0, 0.0, pl_in, I3, ck_in, I3, np.array([0.0, 0.0, T_BOARD]))
    assert sv_crush["box_crushed"]
    witnesses.append(("front_panel_crushed_inward_12deg", 0.0))

    assert math.isclose(HAND_CLEAR_M, 0.0800, rel_tol=0.0, abs_tol=1e-9), f"HAND_CLEAR_M must be 80.0 mm, got {HAND_CLEAR_M}"
    hc_far = hand_clearance_m(far_pads, bpos, R_box, 0.0, 0.0, math.radians(-16.0), pl_in, I3, ck_in, I3)
    assert hc_far >= HAND_CLEAR_M
    near_pads_30 = list(far_pads)
    near_pads_30[0] = bpos + R_box @ np.array([-W_IN / 2 - 0.030, 0.0, 0.080])
    assert hand_clearance_m(near_pads_30, bpos, R_box, 0.0, 0.0, math.radians(-16.0), pl_in, I3, ck_in, I3) < HAND_CLEAR_M
    witnesses.append(("hand_too_close_30mm", 0.0))

    near_pads_73 = list(far_pads)
    near_pads_73[0] = bpos + R_box @ np.array([-W_IN / 2 - 0.07349, 0.0, 0.080])
    hc_73 = hand_clearance_m(near_pads_73, bpos, R_box, 0.0, 0.0, math.radians(-16.0), pl_in, I3, ck_in, I3)
    assert hc_73 < HAND_CLEAR_M, f"73.49 mm clearance must fail 80 mm gate: {hc_73}"
    witnesses.append(("hand_too_close_73.49mm_fails_80mm_gate", 0.0))

    # Direct production task.score(objects) boundary checks across 50..80.5 mm clearance
    from types import SimpleNamespace
    obj_ok = {
        "cake_box": SimpleNamespace(pos=bpos, rotation=lambda: R_box, joints={"front_panel": 0.0, "lid": 0.0, "lid_tab": math.radians(-57.84)}),
        "cake_board": SimpleNamespace(pos=pl_in, rotation=lambda: I3),
    }
    for mm, expected in ((50.0, 0.0), (73.49, 0.0), (79.5, 0.0), (80.5, 1.0)):
        task._STATE.update(frosting_touch=None, cake_disturbed=False, board_dropped=False, box_crushed=False, hand_clear_m=mm / 1000.0, still_ok=True)
        got = task.score(obj_ok)
        assert got == expected, f"score() at {mm} mm: expected {expected}, got {got}"
        witnesses.append((f"score_hand_clear_{mm}mm", expected))

    # Analytic 3D projected circle overhang witnesses (including 7.5 deg yaw where sparse 24-pt sampling missed 1.06 mm)
    R_yaw75 = _rz(math.radians(7.5))
    w_over_2mm = evaluate_final_tier(bpos, R_box, 0.0, 0.0, math.radians(-57.84), bpos + R_box @ np.array([0.0052, 0.0, T_BASE]), R_box @ R_yaw75)
    assert not w_over_2mm["board_inside"] and w_over_2mm["tier"] == 0.0, w_over_2mm
    witnesses.append(("analytic_circle_2mm_overhang_at_7.5deg_yaw_rejected", 0.0))

    w_over_1p05mm = evaluate_final_tier(bpos, R_box, 0.0, 0.0, math.radians(-57.84), bpos + R_box @ np.array([0.00425, 0.0, T_BASE]), R_box @ R_yaw75)
    assert not w_over_1p05mm["board_inside"] and w_over_1p05mm["tier"] == 0.0, w_over_1p05mm
    witnesses.append(("analytic_circle_1.05mm_overhang_at_7.5deg_yaw_rejected", 0.0))

    w_allow_0p8mm = evaluate_final_tier(bpos, R_box, 0.0, 0.0, math.radians(-57.84), bpos + R_box @ np.array([0.0040, 0.0, T_BASE]), R_box @ R_yaw75)
    assert w_allow_0p8mm["board_inside"] and w_allow_0p8mm["tier"] == 1.0, w_allow_0p8mm
    witnesses.append(("analytic_circle_0.8mm_within_1mm_wall_allowance_accepted", 1.0))

    w_board_tilt = evaluate_final_tier(bpos, R_box, 0.0, 0.0, math.radians(-57.84), pl_in, R_box @ ry(math.radians(4.0)))
    assert not w_board_tilt["board_inside"] and w_board_tilt["tier"] == 0.0, w_board_tilt
    witnesses.append(("analytic_tilted_board_4deg_exceeds_3.5mm_floor_tol", 0.0))

    print(f"[PASS] {len(witnesses)}/{len(witnesses)} scorer witness checks passed!")

    # 3. Full 630-hull dual-arm trajectory IK & frosting-clearance sweep across normal demo
    plan_dict = task.demo()
    lp = _demo.ArmPlan(plan_dict["left"], LEFT_BASE_POS)
    rp = _demo.ArmPlan(plan_dict["right"], RIGHT_BASE_POS)
    total_dur = max(sum(w.duration for w in plan_dict["left"]), sum(w.duration for w in plan_dict["right"]))
    n_steps = int(math.ceil(total_dur * 20.0))

    frosting_hits = []
    n_hulls_checked = 0
    # Phase 1 exact waypoint clocks:
    # Step 1a (total 10.0 s): 0..3.2 approach, 3.2..7.4 slide CAKE_START_XY -> STAGE_XY, 7.4..10.0 align & lift
    # Step 1b (10.0..20.6 s): 10.0..13.2 arc & behind & engage, 13.2..18.0 slide STAGE_XY -> BOX_XY
    for step_i in range(0, n_steps + 1, 2):
        t_sec = step_i / 20.0
        al, lbl_l = lp.action7(t_sec)
        ar, lbl_r = rp.action7(t_sec)
        ql, gl = al[:6], al[6] * (-0.04695)
        qr, gr = ar[:6], ar[6] * (-0.04695)

        if t_sec < 3.4:
            ck_xy = task.CAKE_START_XY.copy()
        elif t_sec < 8.2:
            u = _ik.smoothstep(min(1.0, max(0.0, (t_sec - 3.4) / 4.8)))
            ck_xy = task.CAKE_START_XY + u * (task.STAGE_XY - task.CAKE_START_XY)
        elif t_sec < 14.0:
            ck_xy = task.STAGE_XY.copy()
        elif t_sec < 19.4:
            u = _ik.smoothstep(min(1.0, max(0.0, (t_sec - 14.0) / 5.4)))
            ck_xy = task.STAGE_XY + u * (task.BOX_XY - task.STAGE_XY)
        else:
            ck_xy = task.BOX_XY.copy()

        ck_z = task.TOP + T_BOARD + (T_BASE if t_sec >= 15.4 else 0.0)
        ck_p = np.array([ck_xy[0], ck_xy[1], ck_z])

        hulls_w = _fk_arm_world_hulls("left", LEFT_BASE_POS, ql, gl) + _fk_arm_world_hulls("right", RIGHT_BASE_POS, qr, gr)
        n_hulls_checked = len(hulls_w)
        hit = robot_hulls_touch_frosting(hulls_w, ck_p, I3, tol=0.0050)
        if hit is not None:
            frosting_hits.append((round(t_sec, 2), hit, lbl_l, lbl_r, ck_xy.round(3).tolist()))

    max_ik_mm = max(lp.worst_ik_mm, rp.worst_ik_mm)
    max_ik_deg = max(lp.worst_ik_deg, rp.worst_ik_deg)
    print(f"Trajectory duration: {total_dur:.2f} s ({n_steps} steps at 20 Hz), dual-arm hulls/step: {n_hulls_checked}")
    print(f"Worst IK error: {max_ik_mm:.4f} mm, {max_ik_deg:.4f} deg")
    if frosting_hits:
        print("FROSTING NEAR-HITS (< 5.0 mm):", frosting_hits[:5])
        raise AssertionError(f"Found {len(frosting_hits)} frosting near-hits!")
    assert max_ik_mm < 1.0 and max_ik_deg < 1.0, f"IK error too high: {max_ik_mm} mm, {max_ik_deg} deg"
    print("[PASS] 0 frosting near-hits (< 5.0 mm) across entire 630-hull trajectory!")

    # 4. Verify final parked 6-point pad/jaw clearance against closed box, board, and cake (>= 120 mm >> 80 mm)
    al_end, _ = lp.action7(total_dur)
    ar_end, _ = rp.action7(total_dur)
    end_pads: list[np.ndarray] = []
    for base_vec, a7 in ((LEFT_BASE_POS, al_end), (RIGHT_BASE_POS, ar_end)):
        f_mag = abs(float(a7[6])) * 0.04
        jmap = {f"joint{i + 1}": float(a7[i]) for i in range(6)}
        jmap["joint7"] = -f_mag
        jmap["joint8"] = -f_mag
        poses = _urdf.forward_kinematics(YAM_URDF, jmap)
        p_tcp, _ = _ik.tcp_pose(a7[:6], base_vec)
        g_rot = poses["gripper"][:3, :3]
        end_pads.extend([p_tcp - f_mag * g_rot[:, 0], p_tcp + f_mag * g_rot[:, 0], p_tcp])
    hc_end = hand_clearance_m(end_pads, bpos, R_box, 0.0, 0.0, math.radians(-57.84), pl_in, I3, ck_in, I3)
    print(f"Final parked 6-point hand clearance: {hc_end * 1000.0:.2f} mm (gate = {HAND_CLEAR_M * 1000.0:.1f} mm)")
    assert hc_end >= 0.120, f"Final parked hand clearance too small: {hc_end * 1000:.2f} mm"
    print("[PASS] Final parked 6-point hand clearance >= 120.0 mm!")


if __name__ == "__main__":
    run_checks()
