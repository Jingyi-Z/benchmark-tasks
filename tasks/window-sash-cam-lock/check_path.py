"""578-hull installed-robot ArmPlan.action7(t) + WINDOW/SASH/CAM collision screen + 15 scorer counterexamples."""
import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import task
from hand_geometry import read_stl, rotation, vector, world_planes
from task_metrics import box_rot, cyl_as_boxes, part_local_corners, rz
from yam_arena import YAM_URDF, ik
from yam_arena.constants import LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import ArmPlan
from yam_arena.urdf import forward_kinematics


class FakeState:
    def __init__(self, pos, R=None, joints=None):
        self.pos = np.asarray(pos, float)
        self._R = np.eye(3) if R is None else np.asarray(R, float)
        self.joints = {} if joints is None else dict(joints)

    def rotation(self):
        return self._R


def check_metadata():
    tj = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    allowed = {
        "Impact on jobs",
        "Impact on AI timelines",
        "Safety and alignment",
        "Societal relatability",
        "Other",
    }
    assert tj["category"] in allowed, f"Bad category: {tj['category']}"
    fid_len = len(tj["fidelity"])
    assert fid_len < 500, f"fidelity too long: {fid_len}"
    print(f"[PASS] task.json category='{tj['category']}', len(fidelity)={fid_len} < 500")


def obb_overlap(box_a, pos_a, R_a, box_b, pos_b, R_b, eps=1e-5) -> float:
    ca = R_a @ np.asarray(box_a[0], float) + pos_a
    ha = np.asarray(box_a[1], float) / 2.0
    Ua = R_a @ box_rot(box_a)

    cb = R_b @ np.asarray(box_b[0], float) + pos_b
    hb = np.asarray(box_b[1], float) / 2.0
    Ub = R_b @ box_rot(box_b)

    axes = [Ua[:, i] for i in range(3)] + [Ub[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(Ua[:, i], Ub[:, j])
            n = np.linalg.norm(c)
            if n > 1e-8:
                axes.append(c / n)

    d = cb - ca
    min_pen = float("inf")
    for ax in axes:
        ra = float(np.sum(ha * np.abs(Ua.T @ ax)))
        rb = float(np.sum(hb * np.abs(Ub.T @ ax)))
        sep = abs(float(np.dot(d, ax))) - (ra + rb)
        if sep >= -eps:
            return 0.0
        min_pen = min(min_pen, -sep)
    return min_pen


def obb_cyl_z_overlap(box_a, pos_a, R_a, cyl_b, pos_b, R_b) -> float:
    # Exact 3D check for Z-axis cylinder on R_b==I3 vs box_a:
    ca = R_a @ np.asarray(box_a[0], float) + pos_a
    ha = np.asarray(box_a[1], float) / 2.0
    Ua = R_a @ box_rot(box_a)
    cc = pos_b + R_b @ np.asarray(cyl_b[0], float)
    rad = float(cyl_b[1])
    hc = float(cyl_b[2]) * 0.5
    # Vertical overlap check first:
    az_rad = float(np.sum(ha * np.abs(Ua[2, :])))
    z_pen = (az_rad + hc) - abs(float(ca[2] - cc[2]))
    if z_pen <= 1e-5:
        return 0.0
    # 2D XY box vs circle check (since box_a only yaws about Z):
    rel_xy = cc[:2] - ca[:2]
    lx = float(rel_xy[0] * Ua[0, 0] + rel_xy[1] * Ua[1, 0])
    ly = float(rel_xy[0] * Ua[0, 1] + rel_xy[1] * Ua[1, 1])
    dx = max(abs(lx) - ha[0], 0.0)
    dy = max(abs(ly) - ha[1], 0.0)
    dist = math.hypot(dx, dy)
    xy_pen = (rad + min(ha[0] - abs(lx), ha[1] - abs(ly))) if dist == 0.0 else max(0.0, rad - dist)
    return min(z_pen, xy_pen)


def check_state_penetration(dz: float, dx: float, theta_deg: float, label: str, expect_clear: bool = True):
    m = task.Metric(vars(task))
    st = FakeState(task.WINDOW.pos, joints={"sash_z": dz, "sash_x": dx, "cam_lock": math.radians(theta_deg)})
    p_0, R_0, p_z, R_z, p_s, R_s, p_c, R_c, _, _, _ = m.link_poses(st)

    worst = 0.0
    # PhysX self_collide checks (main vs sash_x) and (main vs cam_lock):
    for bf in task.FRAME_BOXES:
        for bs in task.SASH_X_BOXES:
            worst = max(worst, obb_overlap(bf, p_0, R_0, bs, p_s, R_s))
        for bc in list(task.CAM_BOXES) + cyl_as_boxes(task.CAM_CYLS):
            worst = max(worst, obb_overlap(bf, p_0, R_0, bc, p_c, R_c))
    for cyf in task.FRAME_CYLS:
        for bs in task.SASH_X_BOXES:
            worst = max(worst, obb_cyl_z_overlap(bs, p_s, R_s, cyf, p_0, R_0))
        for bc in list(task.CAM_BOXES) + cyl_as_boxes(task.CAM_CYLS):
            worst = max(worst, obb_cyl_z_overlap(bc, p_c, R_c, cyf, p_0, R_0))

    if expect_clear:
        assert worst <= 1e-4, f"{label} overlap: {worst * 1000:.3f} mm"
        print(f"[PASS] {label}: max OBB/cyl penetration = {worst * 1000:.4f} mm")
    else:
        assert worst > 0.001, f"{label} expected keeper roof block, got {worst * 1000:.3f} mm"
        print(f"[PASS] {label}: physical keeper block confirmed ({worst * 1000:.2f} mm overlap if forced)")


def check_static_sections():
    check_state_penetration(task.SASH_Z_INIT, task.SASH_X_INIT, 0.0, "t=0 Open Unlocked")
    check_state_penetration(0.0, task.SASH_X_INIT, 0.0, "Closed Unlocked (dz=0, dx=-2.2mm, th=0)")
    check_state_penetration(0.0, -0.0011, 148.0, "Closed Mid-Ramp (dz=0, dx=-1.1mm, th=148)")
    check_state_penetration(0.0, -0.0002, 165.0, "Closed Locked (dz=0, dx=-0.2mm, th=165)")
    check_state_penetration(0.0, -0.0003, 170.0, "Closed Locked Stop (dz=0, dx=-0.3mm, th=170)")
    check_state_penetration(0.0120, task.SASH_X_INIT, 165.0, "Above Keeper Roof (dz=12mm, th=165)")
    check_state_penetration(0.0070, task.SASH_X_INIT, 165.0, "Turn-Before-Close Blocked by Keeper Roof (dz=7mm, th=165)", expect_clear=False)


def check_scorer_counterexamples():
    m = task.Metric(vars(task))
    p0 = np.asarray(task.WINDOW.pos, float)
    far_tips = np.tile(np.array([[0.10, +0.35, 0.95], [0.10, -0.35, 0.95]]), (3, 1))

    def eval_s(dz, dx, th_deg, pos=None, tips=far_tips, jitter=0.0, n_hist=21, reset_first=True):
        if reset_first:
            m.reset()
        st = FakeState(p0 if pos is None else pos, joints={"sash_z": dz, "sash_x": dx, "cam_lock": math.radians(th_deg)})
        _, _, _, _, p_s, R_s, p_c, R_c, _, _, _ = m.link_poses(st)
        pts = np.vstack([m.sash_x_corners @ R_s.T + p_s, m.cam_corners @ R_c.T + p_c])
        hist = [pts + (np.array([0.0, 0.0, jitter]) if k == n_hist - 1 else 0.0) for k in range(n_hist)]
        return m.evaluate_states(st, tips, hist_override=hist)

    # 1. Reset open state -> 0.0
    assert eval_s(0.0550, -0.0022, 0.0) == 0.0
    # 2. Closed unlocked on sill -> 0.5
    assert eval_s(0.0000, -0.0022, 0.0) == 0.5
    # 3. Closed unlocked 5.5 mm above sill -> 0.5
    assert eval_s(0.0055, -0.0022, 0.0) == 0.5
    # 4. Blocked on keeper roof 11.0 mm above sill (turn_before_close) -> 0.0
    assert eval_s(0.0110, -0.0022, 165.0) == 0.0
    # 5. Closed on sill + cam only turned 130 deg (not captured, weatherseal -2.1mm) -> 0.5
    assert eval_s(0.0000, -0.0021, 130.0) == 0.5
    # 6. Closed on sill + cam turned 165 deg WITHOUT weatherseal draw-in (dx=-1.8mm) -> 0.5
    assert eval_s(0.0000, -0.0018, 165.0) == 0.5
    # 7. Closed on sill + weatherseal compressed WITHOUT cam capture (th=0) -> 0.5
    assert eval_s(0.0000, -0.0001, 0.0) == 0.5
    # 8. Full closed + keeper captured + weatherseal drawn snug (th=165, dx=-0.2mm) -> 1.0
    assert eval_s(0.0000, -0.0002, 165.0) == 1.0
    # 9. Full closed + keeper captured across dwell band (dz=2.0mm, dx=-0.6mm, th=158) -> 1.0
    assert eval_s(0.0020, -0.0006, 158.0) == 1.0
    # 10. Hand within 70 mm of lower sash meeting rail -> 0.0
    near_sash = far_tips.copy()
    near_sash[0] = np.array([task.WIN_X - 0.0265 - 0.070, +0.100, task.TOP + 0.285])
    assert eval_s(0.0000, -0.0002, 165.0, tips=near_sash) == 0.0
    # 11. Hand within 72 mm of outer frame top jamb -> 0.0
    near_frame = far_tips.copy()
    near_frame[0] = np.array([task.WIN_X - 0.03175 - 0.065, +0.200, task.TOP + 0.500])
    assert eval_s(0.0000, -0.0002, 165.0, tips=near_frame) == 0.0
    # 12. Hand at 90 mm clear -> 1.0
    ok_tips = far_tips.copy()
    ok_tips[0] = np.array([task.WIN_X - 0.03175 - 0.095, 0.0, task.TOP + 0.250])
    assert eval_s(0.0000, -0.0002, 165.0, tips=ok_tips) == 1.0
    # 13. Stillness drift 1.4 mm -> 0.0
    assert eval_s(0.0000, -0.0002, 165.0, jitter=0.0014) == 0.0
    # 14. Fewer than 21 samples -> 0.0
    assert eval_s(0.0000, -0.0002, 165.0, n_hist=18) == 0.0
    # 15. Final-state off_table check (0.0 when below bench, recovers to 1.0 if final state is on bench)
    assert eval_s(0.0000, -0.0002, 165.0, pos=[task.WIN_X, 0.0, task.TOP - 0.040]) == 0.0
    assert eval_s(0.0000, -0.0002, 165.0, reset_first=False) == 1.0
    print("[PASS] 15/15 CPU scorer counterexamples verified")


def _boxes_to_hulls(prefix, boxes, p_obj, R_obj=None):
    R_obj = np.eye(3) if R_obj is None else np.asarray(R_obj, float)
    out = []
    for idx, b in enumerate(boxes):
        c = p_obj + R_obj @ np.asarray(b[0], float)
        h = np.asarray(b[1], float) * 0.5
        rb = R_obj @ box_rot(b)
        verts = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    verts.append(c + rb @ (h * np.array([sx, sy, sz])))
        vv = np.asarray(verts, float)
        planes = []
        for ax in range(3):
            n = rb[:, ax]
            planes.append(np.r_[n, -float(np.dot(n, c)) - h[ax]])
            planes.append(np.r_[-n, float(np.dot(n, c)) - h[ax]])
        out.append((f"{prefix}_{idx}", vv, np.asarray(planes, float), vv.min(0), vv.max(0)))
    return out


def _polyhedra_intersect_exact(va, pa, vb, pb) -> bool:
    if np.any(np.min(vb @ pa[:, :3].T + pa[:, 3], axis=0) > 1e-6):
        return False
    if np.any(np.min(va @ pb[:, :3].T + pb[:, 3], axis=0) > 1e-6):
        return False
    P = np.vstack([pa, pb])
    res = linprog(np.zeros(3), A_ub=P[:, :3], b_ub=-P[:, 3] - 1e-6, bounds=(None, None), method="highs")
    return bool(res.success)


def _nominal_window_state_at_t(mode: str, t: float):
    # Expected sash_z, sash_x, cam_lock along each trajectory so all 578 robot hulls are checked against moving sash & cam too!
    if mode == "close_without_lock":
        if t <= 2.4:
            dz = task.SASH_Z_INIT
        elif t <= 4.4:
            u = (t - 2.4) / 2.0
            dz = task.SASH_Z_INIT * (1.0 - u)
        else:
            dz = 0.0
        return dz, task.SASH_X_INIT, 0.0

    if mode == "turn_before_close":
        if t <= 2.1:
            th = 0.0
        elif t <= 5.1:
            th = 168.0 * (t - 2.1) / 3.0
        else:
            th = 168.0
        if t <= 8.9:
            dz = task.SASH_Z_INIT
        else:
            dz = max(0.0115, task.SASH_Z_INIT - 0.045 * min(1.0, (t - 8.9) / 2.0))
        return dz, task.SASH_X_INIT, th

    if mode == "unlocked_pull":
        if t <= 2.2:
            dz = task.SASH_Z_INIT
        elif t <= 4.2:
            dz = task.SASH_Z_INIT * (1.0 - (t - 2.2) / 2.0)
        elif t <= 9.6:
            dz = 0.0
        elif t <= 11.0:
            dz = 0.042 * (t - 9.6) / 1.4
        else:
            dz = 0.042
        return dz, task.SASH_X_INIT, 0.0

    # Nominal full demo (and locked_pull):
    if t <= 2.2:
        dz = task.SASH_Z_INIT
    elif t <= 4.2:
        dz = task.SASH_Z_INIT * (1.0 - (t - 2.2) / 2.0)
    else:
        dz = 0.0
    if t <= 7.2:
        th = 0.0
        dx = task.SASH_X_INIT
    elif t <= 10.3:
        th = 168.0 * (t - 7.2) / 3.1
        dx = task.SASH_X_INIT if th < 136.0 else (-0.0022 + 0.0020 * min(1.0, (th - 136.0) / 30.0))
    else:
        th = 168.0
        dx = -0.0002
    return dz, dx, th


def check_full_trajectory():
    urdf = pathlib.Path(YAM_URDF)
    root = ET.parse(urdf).getroot()
    shapes = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        if name in ("base", "world"):
            continue
        for c in link.findall("collision"):
            mesh = c.find("geometry/mesh")
            if mesh is None:
                continue
            path = (urdf.parent / mesh.get("filename")).resolve()
            o = c.find("origin")
            r = rotation(vector(o.get("rpy") if o is not None else None, [0, 0, 0]))
            p = vector(o.get("xyz") if o is not None else None, [0, 0, 0])
            v = (read_stl(path) * vector(mesh.get("scale"), [1, 1, 1])) @ r.T + p
            v = np.unique(np.round(v.reshape(-1, 3), 7), axis=0)
            hull = ConvexHull(v)
            shapes.append((name, v[hull.vertices], hull.equations.copy()))
    print(f"[INFO] Loaded {len(shapes)} collision hulls from YAM_URDF (2 arms = {2 * len(shapes)} hulls)")

    m = task.Metric(vars(task))
    p_win = np.asarray(task.WINDOW.pos, float)
    static_frame_hulls = _boxes_to_hulls("frame", list(task.FRAME_BOXES) + cyl_as_boxes(task.FRAME_CYLS), p_win)
    bases = [np.array(RIGHT_BASE_POS, float), np.array(LEFT_BASE_POS, float)]

    def placed(i, q, f):
        joints = {f"joint{k + 1}": float(q[k]) for k in range(6)}
        joints.update(joint7=f, joint8=f)
        fk = forward_kinematics(urdf, joints)
        out = []
        for name, v, planes in shapes:
            tf = fk[name].copy()
            tf[:3, 3] += bases[i]
            vv = v @ tf[:3, :3].T + tf[:3, 3]
            out.append((name, vv, planes, tf[:3, :3], tf[:3, 3], vv.min(0), vv.max(0)))
        return out

    for mode in ("", "close_without_lock", "turn_before_close", "unlocked_pull", "locked_pull"):
        task._PROBE = mode
        plan = task.demo()
        aps = [ArmPlan(plan["right"], RIGHT_BASE_POS), ArmPlan(plan["left"], LEFT_BASE_POS)]
        duration = max(sum(w.duration for w in v) for v in plan.values())
        worst_pos = worst_rot = 0.0
        hits = []
        frame_hits = []
        moving_non_tip_hits = []
        prepost_tip_hits = []
        below = []
        min_inter_aabb_mm = 999.0
        min_frame_aabb_mm = 999.0
        min_top_mm = 999.0
        dt = 0.05 if mode == "" else 0.10

        flo = np.array([fh[3] for fh in static_frame_hulls])
        fhi = np.array([fh[4] for fh in static_frame_hulls])

        for t in np.arange(0.0, duration + 0.001, dt):
            bodies = []
            for i in range(2):
                tcp, rot, _, _ = aps[i].target(float(t))
                act7, _ = aps[i].action7(float(t))
                q = np.asarray(act7[:6], float)
                tcp_act, rot_act = ik.tcp_pose(q, bases[i])
                pe = float(np.linalg.norm(tcp_act - tcp))
                cos_ang = float(np.clip((np.trace(rot_act.T @ rot) - 1.0) * 0.5, -1.0, 1.0))
                re = math.acos(cos_ang)
                worst_pos = max(worst_pos, pe * 1000.0)
                worst_rot = max(worst_rot, float(np.degrees(re)))
                bodies.append(placed(i, q, float(act7[6]) * float(task.FINGER_OPEN)))

            alo = np.array([v[5] for v in bodies[0]])
            ahi = np.array([v[6] for v in bodies[0]])
            blo = np.array([v[5] for v in bodies[1]])
            bhi = np.array([v[6] for v in bodies[1]])
            gap = np.maximum(0.0, np.maximum(alo[:, None, :] - bhi[None, :, :], blo[None, :, :] - ahi[:, None, :]))
            min_inter_aabb_mm = min(min_inter_aabb_mm, float(np.min(np.linalg.norm(gap, axis=-1)) * 1000.0))
            cands = np.all(np.minimum(ahi[:, None], bhi[None, :]) - np.maximum(alo[:, None], blo[None, :]) > 1e-6, axis=-1)
            for ia, ib in np.argwhere(cands):
                aa, bb = bodies[0][ia], bodies[1][ib]
                pa = world_planes(aa[2], aa[3], aa[4])
                pb = world_planes(bb[2], bb[3], bb[4])
                if _polyhedra_intersect_exact(aa[1], pa, bb[1], pb):
                    hits.append([round(float(t), 2), aa[0], bb[0]])

            # All 578 robot hulls vs static outer frame, upper sash, and C-keeper at EVERY t:
            for arm_idx, (rlo, rhi) in enumerate([(alo, ahi), (blo, bhi)]):
                fgap = np.maximum(0.0, np.maximum(rlo[:, None, :] - fhi[None, :, :], flo[None, :, :] - rhi[:, None, :]))
                min_frame_aabb_mm = min(min_frame_aabb_mm, float(np.min(np.linalg.norm(fgap, axis=-1)) * 1000.0))
                fcands = np.all(np.minimum(rhi[:, None], fhi[None, :]) - np.maximum(rlo[:, None], flo[None, :]) > 1e-6, axis=-1)
                for ir, jf in np.argwhere(fcands):
                    rr = bodies[arm_idx][ir]
                    ff = static_frame_hulls[jf]
                    pr = world_planes(rr[2], rr[3], rr[4])
                    if _polyhedra_intersect_exact(rr[1], pr, ff[1], ff[2]):
                        frame_hits.append([round(float(t), 2), arm_idx, rr[0], ff[0]])

            # Moving lower sash + cam lock hulls at nominal state(t):
            dz_t, dx_t, th_t = _nominal_window_state_at_t(mode, float(t))
            st_t = FakeState(p_win, joints={"sash_z": dz_t, "sash_x": dx_t, "cam_lock": math.radians(th_t)})
            _, _, _, _, p_s, R_s, p_c, R_c, _, _, _ = m.link_poses(st_t)
            moving_hulls = (
                _boxes_to_hulls("sash", task.SASH_X_BOXES, p_s, R_s)
                + _boxes_to_hulls("cam", list(task.CAM_BOXES) + cyl_as_boxes(task.CAM_CYLS), p_c, R_c)
            )
            mlo = np.array([mh[3] for mh in moving_hulls])
            mhi = np.array([mh[4] for mh in moving_hulls])

            # Check ALL non-tip links (link1..gripper) at ALL t, plus tip_left/tip_right outside contact windows!
            left_in_contact = (
                (1.5 <= t <= 5.7) if mode in ("", "close_without_lock") else (
                    (10.5 <= t <= 14.4) if mode == "turn_before_close" else (
                        (1.5 <= t <= 5.7) or (8.9 <= t <= 12.8) if mode == "unlocked_pull" else (
                            (1.5 <= t <= 5.7) or (17.1 <= t <= 21.0)
                        )
                    )
                )
            )
            right_in_contact = (
                (6.6 <= t <= 13.6) if mode == "" else (
                    (1.2 <= t <= 8.2) if mode == "turn_before_close" else (
                        (8.9 <= t <= 12.8) if mode == "unlocked_pull" else (
                            (6.6 <= t <= 13.6) or (17.1 <= t <= 21.0)
                        )
                    )
                )
            )
            for arm_idx, (rlo, rhi, in_contact) in enumerate([(alo, ahi, right_in_contact), (blo, bhi, left_in_contact)]):
                mcands = np.all(np.minimum(rhi[:, None], mhi[None, :]) - np.maximum(rlo[:, None], mlo[None, :]) > 1e-6, axis=-1)
                for ir, jm in np.argwhere(mcands):
                    rr = bodies[arm_idx][ir]
                    mm = moving_hulls[jm]
                    is_tip = rr[0] in ("tip_left", "tip_right")
                    if is_tip and in_contact:
                        continue
                    pr = world_planes(rr[2], rr[3], rr[4])
                    if _polyhedra_intersect_exact(rr[1], pr, mm[1], mm[2]):
                        if is_tip:
                            prepost_tip_hits.append([round(float(t), 2), arm_idx, rr[0], mm[0]])
                        else:
                            moving_non_tip_hits.append([round(float(t), 2), arm_idx, rr[0], mm[0]])

            for arm_idx, b in enumerate(bodies):
                for name, vv, *_ in b:
                    if name == "link1":
                        continue
                    zmin = float(vv[:, 2].min())
                    min_top_mm = min(min_top_mm, (zmin - float(task.TOP)) * 1000.0)
                    if zmin < float(task.TOP) - 1e-4:
                        below.append([round(float(t), 2), arm_idx, name, round((zmin - float(task.TOP)) * 1000.0, 2)])

        label = mode or "demo"
        print(
            f"[INFO] mode={label}: dur={duration:.2f}s worst_pos={worst_pos:.4f}mm worst_rot={worst_rot:.4f}deg "
            f"inter_hits={len(hits)} frame_hits={len(frame_hits)} non_tip_moving_hits={len(moving_non_tip_hits)} "
            f"prepost_tip_hits={len(prepost_tip_hits)} below_top={len(below)} "
            f"min_inter_aabb={min_inter_aabb_mm:.2f}mm min_frame_aabb={min_frame_aabb_mm:.2f}mm min_top={min_top_mm:.2f}mm"
        )
        assert worst_pos < 0.5, f"{label}: IK pos error {worst_pos:.3f} mm"
        assert worst_rot < 0.5, f"{label}: IK rot error {worst_rot:.3f} deg"
        assert not hits, f"{label}: inter-arm collisions: {hits[:5]}"
        assert not frame_hits, f"{label}: static frame/keeper collisions: {frame_hits[:5]}"
        assert not moving_non_tip_hits, f"{label}: wrist/arm collisions with moving sash/cam: {moving_non_tip_hits[:5]}"
        assert not prepost_tip_hits, f"{label}: pre/post tip collisions with moving sash/cam: {prepost_tip_hits[:5]}"
        assert not below, f"{label}: below-table collisions: {below[:5]}"
    task._PROBE = ""
    print("[PASS] Full 578-hull trajectory + static/moving window clearance verified across demo and both negative controls")


if __name__ == "__main__":
    check_metadata()
    check_static_sections()
    check_scorer_counterexamples()
    check_full_trajectory()
    print("ALL PREFLIGHT CHECKS PASSED")
