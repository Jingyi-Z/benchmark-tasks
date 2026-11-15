"""578-hull installed-robot ArmPlan.action7(t) + swept ROLLER_FRAME/ROLLER_COVER collision screen + 14 scorer counterexamples."""
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
from task_metrics import box_rot, cyl_as_boxes, part_local_corners, rx, ry, rz
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


def obb_cyl_y_overlap(box_a, pos_a, R_a, cyl_b, pos_b, R_b) -> float:
    ca = R_a @ np.asarray(box_a[0], float) + pos_a
    ha = np.asarray(box_a[1], float) / 2.0
    Ua = R_a @ box_rot(box_a)
    cc = pos_b + R_b @ np.asarray(cyl_b[0], float)
    rad = float(cyl_b[1])
    hc = float(cyl_b[2]) * 0.5
    ay_rad = float(np.sum(ha * np.abs(Ua[1, :])))
    y_pen = (ay_rad + hc) - abs(float(ca[1] - cc[1]))
    if y_pen <= 1e-5:
        return 0.0
    rel_xz = np.array([cc[0] - ca[0], 0.0, cc[2] - ca[2]])
    q_local = Ua.T @ rel_xz
    dx = max(abs(float(q_local[0])) - float(ha[0]), 0.0)
    dz = max(abs(float(q_local[2])) - float(ha[2]), 0.0)
    dist_xz = math.hypot(dx, dz)
    if dist_xz >= rad - 1e-5:
        return 0.0
    return min(y_pen, rad - dist_xz)


def _hull_from_pts(name: str, pts_w: np.ndarray):
    hull = ConvexHull(pts_w)
    return (name, pts_w[hull.vertices], hull.equations.copy(), pts_w.min(0), pts_w.max(0))


def _poly_penetration_z(va: np.ndarray, pa: np.ndarray, vb: np.ndarray, pb: np.ndarray) -> float:
    """Exact minimum downward shift dz >= 0 of polyhedron B needed so B + [0,0,-dz] separates from A."""
    if not _polyhedra_intersect_exact(va, pa, vb, pb):
        return 0.0
    lo, hi = 0.0, 0.0050
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        vb_s = vb + np.array([0.0, 0.0, -mid])
        pb_s = pb.copy()
        pb_s[:, 3] = pb[:, 3] + pb[:, 2] * mid
        if _polyhedra_intersect_exact(va, pa, vb_s, pb_s):
            lo = mid
        else:
            hi = mid
    return hi


def check_bore_and_detent_fit():
    I3 = np.eye(3)
    p_f = np.array([task.FRAME_X0, task.FRAME_Y0, task.Z_CAGE_LIFTED])

    # 1. Audit exact 3-piece detent collider-union profile (outboard 14-deg wedge + crown + inboard -14-deg wedge):
    w_out = np.asarray(task.DETENT_MESHES[0].points, float)
    w_inb = np.asarray(task.DETENT_MESHES[1].points, float)
    crn_pts = part_local_corners(task.DETENT_BOXES)
    all_det_pts = np.vstack([w_out, crn_pts, w_inb])
    max_z = float(np.max(all_det_pts[:, 2]))
    assert abs(max_z - 0.0204) < 1e-9, f"Detent union peak z={max_z * 1000:.6f} mm != 20.400000 mm"

    # Seam continuity at y = -0.033 (outboard -> crown) and y = -0.023 (crown -> inboard):
    out_join_top = float(np.max(w_out[np.isclose(w_out[:, 1], -0.033), 2]))
    out_join_bot = float(np.min(w_out[np.isclose(w_out[:, 1], -0.033), 2]))
    crn_out_top = float(np.max(crn_pts[np.isclose(crn_pts[:, 1], -0.033), 2]))
    crn_out_bot = float(np.min(crn_pts[np.isclose(crn_pts[:, 1], -0.033), 2]))
    crn_inb_top = float(np.max(crn_pts[np.isclose(crn_pts[:, 1], -0.023), 2]))
    crn_inb_bot = float(np.min(crn_pts[np.isclose(crn_pts[:, 1], -0.023), 2]))
    inb_join_top = float(np.max(w_inb[np.isclose(w_inb[:, 1], -0.023), 2]))
    inb_join_bot = float(np.min(w_inb[np.isclose(w_inb[:, 1], -0.023), 2]))
    step_out_um = max(abs(out_join_top - crn_out_top), abs(out_join_bot - crn_out_bot)) * 1e6
    step_inb_um = max(abs(inb_join_top - crn_inb_top), abs(inb_join_bot - crn_inb_bot)) * 1e6
    assert step_out_um < 1e-3 and step_inb_um < 1e-3, f"Seam step: out={step_out_um:.4f} um, inb={step_inb_um:.4f} um"

    slope_out_deg = math.degrees(math.atan2(0.0204 - 0.0172, -0.033 - float(np.min(w_out[:, 1]))))
    slope_inb_deg = math.degrees(math.atan2(0.0172 - 0.0204, float(np.max(w_inb[:, 1])) - (-0.023)))
    assert abs(slope_out_deg - 14.0) < 1e-4 and abs(slope_inb_deg + 14.0) < 1e-4

    for m_idx, m in enumerate(task.DETENT_MESHES):
        pts = np.asarray(m.points, float)
        ctr = np.mean(pts, axis=0)
        for t_idx, (ia, ib, ic) in enumerate(np.asarray(m.tris, int)):
            n = np.cross(pts[ib] - pts[ia], pts[ic] - pts[ia])
            fc = (pts[ia] + pts[ib] + pts[ic]) / 3.0 - ctr
            assert float(np.dot(n, fc)) > 1e-9, f"Mesh {m_idx} tri {t_idx} inward/degenerate normal"
    print(
        f"[PASS] Detent collider-union profile verified: max_z={max_z * 1000:.4f}mm, "
        f"out_slope=+{slope_out_deg:.2f}deg, inb_slope={slope_inb_deg:.2f}deg, "
        f"seam_steps=({step_out_um:.4f}um, {step_inb_um:.4f}um)"
    )

    # 2. Bore vs cage clearance across full 228.6 mm stroke:
    max_cage_pen = 0.0
    for y_cov in np.linspace(task.FRAME_Y0 - task.CAGE_LEN - 0.5 * task.COVER_LEN, task.FRAME_Y0 - 0.5 * task.COVER_LEN, 12):
        p_v = np.array([task.FRAME_X0, y_cov, task.Z_CAGE_LIFTED])
        for cb in task.COVER_BOXES:
            for gb in task.CAGE_BOXES:
                max_cage_pen = max(max_cage_pen, obb_overlap(cb, p_v, I3, gb, p_f, I3))
            for gc in task.CAGE_CYLS:
                max_cage_pen = max(max_cage_pen, obb_cyl_y_overlap(cb, p_v, I3, gc, p_f, I3))
    assert max_cage_pen < 1e-4, f"Cover bore interferes with cage: {max_cage_pen * 1000:.2f} mm"
    print(f"[PASS] 16-stave hollow cover (38.1 mm ID) clears 36.0 mm cage across entire 228.6 mm stroke (max pen = {max_cage_pen * 1000:.3f} mm)")

    # 3. Dense 60-step axial sweep of cover leading rim across entire detent union [-55 mm, 0 mm]:
    det_hulls_rest = _boxes_to_hulls("det_b", task.DETENT_BOXES, p_f, I3) + [
        _hull_from_pts(f"det_m_{i}", np.asarray(m.points, float) + p_f) for i, m in enumerate(task.DETENT_MESHES)
    ]
    det_hulls_comp = _boxes_to_hulls("det_bc", task.DETENT_BOXES, p_f + np.array([0.0, 0.0, -0.0016]), I3) + [
        _hull_from_pts(f"det_mc_{i}", np.asarray(m.points, float) + p_f + np.array([0.0, 0.0, -0.0016]))
        for i, m in enumerate(task.DETENT_MESHES)
    ]
    dz_profile = []
    comp_hits = 0
    for y_inb in np.linspace(-0.0550, 0.0000, 56):
        p_v = np.array([task.FRAME_X0, task.FRAME_Y0 + y_inb - 0.5 * task.COVER_LEN, task.Z_CAGE_LIFTED])
        cov_hulls = _boxes_to_hulls("cov", task.COVER_BOXES, p_v, I3)
        req_dz = 0.0
        for _, cv, cp, clo, chi in cov_hulls:
            for _, dv, dp, dlo, dhi in det_hulls_rest:
                if np.all(np.minimum(chi, dhi) - np.maximum(clo, dlo) > -1e-5):
                    req_dz = max(req_dz, _poly_penetration_z(cv, cp, dv, dp))
            for _, dv, dp, dlo, dhi in det_hulls_comp:
                if np.all(np.minimum(chi, dhi) - np.maximum(clo, dlo) > 1e-6):
                    if _polyhedra_intersect_exact(cv, cp, dv, dp):
                        comp_hits += 1
        dz_profile.append((y_inb, req_dz))

    pre_max = max(dz for y, dz in dz_profile if y <= -0.0390)
    peak_dz = max(dz for _, dz in dz_profile)
    # Monotonic non-decreasing across [-55 mm, 0 mm] up to numerical tolerance 5 um:
    diffs = [dz_profile[i + 1][1] - dz_profile[i][1] for i in range(len(dz_profile) - 1)]
    assert pre_max == 0.0, f"Premature detent contact before ramp entry: {pre_max * 1000:.3f} mm"
    assert min(diffs) >= -1e-5, f"Non-monotonic detent depression step: {min(diffs) * 1000:.4f} mm"
    assert 0.00130 <= peak_dz <= 0.00150, f"Unexpected peak detent depression: {peak_dz * 1000:.3f} mm"
    assert comp_hits == 0, f"Compressed detent (dz=-1.6mm) still hits cover ({comp_hits} hits)"
    print(
        f"[PASS] Swept detent-union engagement monotonic across [-55, 0] mm: "
        f"pre_entry=0.000mm, peak_dz={peak_dz * 1000:.3f}mm, compressed_hits=0"
    )


def check_scorer_counterexamples():
    m = task.Metric(vars(task))
    far_tips = np.tile(np.array([0.0, +0.48, task.TOP + 0.42]), (6, 1))
    p_f = np.array([task.FRAME_X0, task.FRAME_Y0, task.TOP + task.COVER_R_OUT])

    def eval_case(p_frame, R_frame, p_cov, R_cov, tips=far_tips, drift=0.0):
        fs = FakeState(p_frame, R=R_frame, joints={"cage_spin": 0.0, "detent_z": -0.0014})
        cs = FakeState(p_cov, R=R_cov)
        _, _, p_c, R_c, p_d, R_d, _, _ = m.link_poses(fs)
        pts0 = np.vstack([
            m.frame_corners @ R_frame.T + p_frame,
            m.cage_corners @ R_c.T + p_c,
            m.detent_corners @ R_d.T + p_d,
            m.cover_corners @ R_cov.T + p_cov,
        ])
        hist = [pts0.copy() for _ in range(20)] + [pts0 + np.array([drift, 0.0, 0.0])]
        return m.evaluate_states(fs, cs, tips, hist_override=hist)

    I3 = np.eye(3)
    p_cov_full = p_f + np.array([0.0, -0.5 * task.COVER_LEN - 0.0020, 0.0])
    p_cov_flip = p_f + np.array([0.0, -0.5 * task.COVER_LEN - 0.0020, 0.0])
    p_cov_half = p_f + np.array([0.0, -task.CAGE_LEN + 0.1450 - 0.5 * task.COVER_LEN, 0.0])
    p_cov_barely = p_f + np.array([0.0, -task.CAGE_LEN + 0.0600 - 0.5 * task.COVER_LEN, 0.0])

    assert eval_case(np.array([task.FRAME_X0, task.FRAME_Y0, task.FRAME_Z0]), I3, np.array([task.COVER_X0, task.COVER_Y0, task.COVER_Z0]), I3) == 0.0
    assert eval_case(p_f, I3, p_cov_full, I3) == 1.0
    assert eval_case(p_f, I3, p_cov_flip, rz(math.pi)) == 1.0
    R_rot = rz(0.4)
    assert eval_case(p_f, R_rot, p_f + R_rot @ np.array([0.0, -0.5 * task.COVER_LEN - 0.002, 0.0]), R_rot) == 1.0
    assert eval_case(p_f, I3, p_cov_half, I3) == 0.5
    assert eval_case(p_f, I3, p_cov_barely, I3) == 0.0
    assert eval_case(p_f, I3, p_cov_full + np.array([0.058, 0.0, 0.0]), I3) == 0.0
    assert eval_case(p_f, I3, p_cov_half, rz(math.radians(9.0))) == 0.0
    near_cov = far_tips.copy()
    near_cov[3] = p_cov_full + np.array([0.0, 0.0, task.COVER_R_OUT + 0.045])
    assert eval_case(p_f, I3, p_cov_full, I3, tips=near_cov) == 0.0
    near_shank = far_tips.copy()
    near_shank[0] = p_f + np.array([-0.0675, +0.0320, 0.045])
    assert eval_case(p_f, I3, p_cov_full, I3, tips=near_shank) == 0.0
    near_hdl = far_tips.copy()
    near_hdl[1] = p_f + np.array([-0.1350, +0.1260, 0.055])
    assert eval_case(p_f, I3, p_cov_full, I3, tips=near_hdl) == 0.0
    assert eval_case(p_f, I3, p_cov_full, I3, drift=0.0014) == 0.0
    assert eval_case(p_f, I3, np.array([0.35, -0.20, task.TOP - 0.045]), I3) == 0.0
    assert eval_case(p_f, I3, p_f + np.array([0.0, -0.5 * task.COVER_LEN - 0.0095, 0.0]), I3) == 0.5
    near_wedge = far_tips.copy()
    near_wedge[2] = p_f + np.array([0.0, -0.040, +0.018 + 0.050])
    assert eval_case(p_f, I3, p_cov_barely, I3, tips=near_wedge) == 0.0
    print("[PASS] 15/15 CPU scorer counterexamples verified")


def _polyhedra_intersect_exact(va, pa, vb, pb):
    if np.any(np.min(vb @ pa[:, :3].T + pa[:, 3], axis=0) > 1e-6):
        return False
    if np.any(np.min(va @ pb[:, :3].T + pb[:, 3], axis=0) > 1e-6):
        return False
    P = np.vstack([pa, pb])
    res = linprog(np.zeros(3), A_ub=P[:, :3], b_ub=-P[:, 3] - 1e-6, bounds=(None, None), method="highs")
    return bool(res.success)


def _boxes_to_hulls(prefix: str, boxes, pos: np.ndarray, R: np.ndarray = None):
    if R is None:
        R = np.eye(3)
    out = []
    for k, b in enumerate(boxes):
        pts_l = part_local_corners([b])
        pts_w = pts_l @ R.T + pos
        hull = ConvexHull(pts_w)
        out.append((f"{prefix}_{k}", pts_w[hull.vertices], hull.equations.copy(), pts_w.min(0), pts_w.max(0)))
    return out


def _expected_part_poses(mode: str, t: float):
    I3 = np.eye(3)
    if mode == "unbraced_push":
        p_f = np.array([task.FRAME_X0, task.FRAME_Y0, task.FRAME_Z0])
        if t <= 3.4:
            p_v = np.array([task.COVER_X0, task.COVER_Y0, task.COVER_Z0])
        else:
            p_v = np.array([task.FRAME_X0, task.Y_COV_CTR_PRE, task.COVER_Z0 + 0.004])
        return p_f, I3, p_v, I3, False, False

    t_low_start = 11.2 if mode == "half_seated" else 16.2
    p_f0 = np.array([task.FRAME_X0, task.FRAME_Y0, task.FRAME_Z0])
    if t <= 3.4:
        p_f = p_f0
        R_f = I3
        l_hold = False
    elif t <= 4.6:
        u = (t - 3.4) / 1.2
        R_f = ry(u * task.ROLL_BRACE_RAD)
        p_f = task._BOLSTER_PIV_W - R_f @ task._BOLSTER_PIV_LOC
        l_hold = True
    elif t <= t_low_start:
        p_f = task.P_FRM_NOM.copy()
        R_f = task.R_FRM_NOM.copy()
        l_hold = True
    elif t <= t_low_start + 1.2:
        u = (t - t_low_start) / 1.2
        R_f = ry((1.0 - u) * task.ROLL_BRACE_RAD)
        p_end = np.array([task.FRAME_X0, task.FRAME_Y0, task.TOP + task.COVER_R_OUT])
        p_f = (1.0 - u) * task.P_FRM_NOM + u * p_end
        l_hold = True
    else:
        p_f = np.array([task.FRAME_X0, task.FRAME_Y0, task.TOP + task.COVER_R_OUT])
        R_f = I3
        l_hold = t <= t_low_start + 2.0

    u_cage = R_f[:, 1]
    p_nose = p_f - task.CAGE_LEN * u_cage

    if t <= 3.4:
        p_v = np.array([task.COVER_X0, task.COVER_Y0, task.COVER_Z0])
        R_v = I3
        r_hold = False
    elif t <= 5.1:
        u = min(1.0, (t - 3.4) / 1.2)
        p_v0 = np.array([task.COVER_X0, task.COVER_Y0, task.COVER_Z0])
        p_v = (1.0 - u) * p_v0 + u * task._P_COV_PRENOSE
        R_v = I3
        r_hold = True
    elif t <= 6.0:
        u = (t - 5.1) / 0.9
        s_ax = -0.0160 + u * (-0.0080 - (-0.0160))
        p_v = p_nose + (s_ax - 0.5 * task.COVER_LEN) * u_cage
        R_v = I3
        r_hold = True
    elif t <= 9.0:
        u = (t - 6.0) / 3.0
        s_ax = -0.0080 + u * (0.1520 - (-0.0080))
        p_v = p_nose + (s_ax - 0.5 * task.COVER_LEN) * u_cage
        R_v = I3
        r_hold = True
    elif mode == "half_seated" or t <= 12.6:
        p_v = p_nose + (0.1520 - 0.5 * task.COVER_LEN) * u_cage
        R_v = I3
        r_hold = t <= 9.8
    elif t <= 14.7:
        u = (t - 12.6) / 2.1
        s_ax = 0.1520 + u * (task.CAGE_LEN - 0.1520)
        p_v = p_nose + (s_ax - 0.5 * task.COVER_LEN) * u_cage
        R_v = I3
        r_hold = True
    else:
        p_v = p_f - 0.5 * task.COVER_LEN * u_cage
        R_v = I3
        r_hold = t <= 15.3

    return p_f, R_f, p_v, R_v, l_hold, r_hold


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

    bases = [np.array(RIGHT_BASE_POS, float), np.array(LEFT_BASE_POS, float)]
    frame_all_boxes = list(task.FRAME_BOXES) + cyl_as_boxes(task.FRAME_CYLS) + list(task.CAGE_BOXES) + cyl_as_boxes(task.CAGE_CYLS) + list(task.DETENT_BOXES)
    cover_all_boxes = list(task.COVER_BOXES)

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

    task._LIVE_FRM = None
    task._SNAP1 = None
    task._SNAP2 = None

    for mode in ("", "half_seated", "unbraced_push"):
        task._PROBE = mode
        plan = task.demo()
        aps = [ArmPlan(plan["right"], RIGHT_BASE_POS), ArmPlan(plan["left"], LEFT_BASE_POS)]
        duration = max(sum(w.duration for w in v) for v in plan.values())
        worst_pos = worst_rot = 0.0
        inter_hits = []
        below_hits = []
        unintended_hits = []
        min_inter_aabb_mm = 999.0
        min_top_mm = 999.0
        dt = 0.05 if mode == "" else 0.10

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
                if _polyhedra_intersect_exact(aa[1], world_planes(aa[2], aa[3], aa[4]), bb[1], world_planes(bb[2], bb[3], bb[4])):
                    inter_hits.append((round(float(t), 2), aa[0], bb[0]))

            for side, arm_b in (("right", bodies[0]), ("left", bodies[1])):
                for lname, vv, planes, R_l, p_l, lo, hi in arm_b:
                    if lname in ("link1", "link2"):
                        continue
                    dz_top = float(lo[2] - task.TOP) * 1000.0
                    min_top_mm = min(min_top_mm, dz_top)
                    if lo[2] < task.TOP + 0.0020:
                        below_hits.append((round(float(t), 2), side, lname, round(dz_top, 2)))

            p_f, R_f, p_v, R_v, l_hold, r_hold = _expected_part_poses(mode, float(t))
            fhulls = _boxes_to_hulls("frame", frame_all_boxes, p_f, R_f) + [
                _hull_from_pts(f"det_mesh_{mi}", np.asarray(m.points, float) @ R_f.T + p_f)
                for mi, m in enumerate(task.DETENT_MESHES)
            ]
            vhulls = _boxes_to_hulls("cover", cover_all_boxes, p_v, R_v)

            for lname, vv, planes, R_l, p_l, lo, hi in bodies[0]:
                pl_arm = None
                for hname, hpts, heq, hlo, hhi in fhulls:
                    if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                        if pl_arm is None:
                            pl_arm = world_planes(planes, R_l, p_l)
                        if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                            unintended_hits.append((round(float(t), 2), "right", lname, hname))
                is_tip = lname in ("tip_left", "tip_right")
                if not (is_tip and (2.1 <= t <= 10.6 or 12.0 <= t <= 15.9)):
                    for hname, hpts, heq, hlo, hhi in vhulls:
                        if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                            if pl_arm is None:
                                pl_arm = world_planes(planes, R_l, p_l)
                            if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                                unintended_hits.append((round(float(t), 2), "right", lname, hname))

            for lname, vv, planes, R_l, p_l, lo, hi in bodies[1]:
                pl_arm = None
                for hname, hpts, heq, hlo, hhi in vhulls:
                    if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                        if pl_arm is None:
                            pl_arm = world_planes(planes, R_l, p_l)
                        if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                            unintended_hits.append((round(float(t), 2), "left", lname, hname))
                is_tip = lname in ("tip_left", "tip_right")
                if not (is_tip and 2.1 <= t <= (13.6 if mode == "half_seated" else 18.8)):
                    for hname, hpts, heq, hlo, hhi in fhulls:
                        if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                            if pl_arm is None:
                                pl_arm = world_planes(planes, R_l, p_l)
                            if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                                unintended_hits.append((round(float(t), 2), "left", lname, hname))

        label = mode if mode else "demo"
        print(
            f"[INFO] mode={label}: dur={duration:.2f}s worst_pos={worst_pos:.4f}mm worst_rot={worst_rot:.4f}deg "
            f"inter_hits={len(inter_hits)} below_top={len(below_hits)} unintended_hits={len(unintended_hits)} "
            f"min_inter_aabb={min_inter_aabb_mm:.2f}mm min_top={min_top_mm:.2f}mm"
        )
        assert worst_pos < 1.0, f"IK pos err {worst_pos:.3f}mm in {label}"
        assert worst_rot < 1.0, f"IK rot err {worst_rot:.3f}deg in {label}"
        assert not inter_hits, f"Inter-arm hits in {label}: {inter_hits[:5]}"
        assert not below_hits, f"Below-table hits in {label}: {below_hits[:5]}"
        assert not unintended_hits, f"Unintended part hits in {label}: {unintended_hits[:8]}"

    # Also verify IK tracking across +-15 mm, +-4 deg live-frame perturbations of _LIVE_FRM + _SNAP1/_SNAP2:
    for dx, dy, dz, dpitch, dyaw in [
        (-0.012, -0.008, -0.006, +0.05, -0.05),
        (+0.012, +0.008, +0.010, -0.05, +0.05),
    ]:
        R_pert = rz(dyaw) @ rx(dpitch) @ task.R_FRM_NOM
        p_pert = task.P_FRM_NOM + np.array([dx, dy, dz])
        task._LIVE_FRM = dict(p_f=p_pert, R_f=R_pert)
        task._SNAP1 = dict(
            R_v=np.eye(3),
            R_cov_from_cmd=task.ROT_PRENOSE_R.T @ np.eye(3),
            p_cov_from_cmd=task.ROT_PRENOSE_R.T @ (task._P_COV_PRENOSE - task.P_R_PRENOSE),
        )
        task._SNAP2 = dict(out_s=-task.CAGE_LEN + 0.1520 - task.COVER_LEN)
        task._PROBE = ""
        ap_r = ArmPlan(task.demo()["right"], RIGHT_BASE_POS)
        w_pe = 0.0
        for t in np.arange(0.0, ap_r.duration + 0.001, 0.10):
            tcp, _, _, _ = ap_r.target(float(t))
            act7, _ = ap_r.action7(float(t))
            tcp_a, _ = ik.tcp_pose(act7[:6], np.array(RIGHT_BASE_POS, float))
            w_pe = max(w_pe, float(np.linalg.norm(tcp_a - tcp)) * 1000.0)
        assert w_pe < 1.0, f"Perturbed seen-pose IK error {w_pe:.2f} mm"
    task._LIVE_FRM = None
    task._SNAP1 = None
    task._SNAP2 = None
    task._PROBE = ""
    print("[PASS] Full 578-hull trajectory + swept ROLLER_FRAME/ROLLER_COVER + perturbed live-pose IK verified")


if __name__ == "__main__":
    check_metadata()
    check_bore_and_detent_fit()
    check_scorer_counterexamples()
    check_full_trajectory()
    print("ALL PREFLIGHT CHECKS PASSED")
