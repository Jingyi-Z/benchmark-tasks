"""Offline 578-hull YAM_URDF + carried-strip/cord/plug + 3D 4-slot TLP606 keyhole interlock + 18 scorer counterexamples."""
import json
import math
import os
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
from task_metrics import box_rot, rx, ry, rz
from yam_arena import YAM_URDF, ik
from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS
from yam_arena.demo import ArmPlan
from yam_arena.urdf import forward_kinematics


class FakeState:
    def __init__(self, pos, R=None, joints=None):
        self.pos = np.asarray(pos, float)
        self._R = np.eye(3) if R is None else np.asarray(R, float)
        self.joints = {} if joints is None else dict(joints)

    def rotation(self):
        return self._R


def _polyhedra_intersect_exact(va, pa, vb, pb):
    if np.any(np.min(vb @ pa[:, :3].T + pa[:, 3], axis=0) > 1e-6):
        return False
    if np.any(np.min(va @ pb[:, :3].T + pb[:, 3], axis=0) > 0.0 + 1e-6):
        return False
    P = np.vstack([pa, pb])
    res = linprog(np.zeros(3), A_ub=P[:, :3], b_ub=-P[:, 3] - 1e-6, bounds=(None, None), method="highs")
    return bool(res.success)


_UNIT_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    float,
)
_BOX_LOCAL_NORMALS = np.array(
    [
        [+1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, +1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, +1.0],
        [0.0, 0.0, -1.0],
    ],
    float,
)


def _boxes_to_hulls(prefix: str, boxes, pos: np.ndarray, R: np.ndarray = None):
    if R is None:
        R = np.eye(3)
    out = []
    for k, b in enumerate(boxes):
        c_l = np.asarray(b[0], float)
        h = 0.5 * np.asarray(b[1], float)
        U = R @ box_rot(b)
        c_w = pos + R @ c_l
        pts_w = (_UNIT_SIGNS * h) @ U.T + c_w
        n_w = _BOX_LOCAL_NORMALS @ U.T
        ext = np.array([h[0], h[0], h[1], h[1], h[2], h[2]], float)
        d_w = -np.einsum("ij,j->i", n_w, c_w) - ext
        eq = np.column_stack([n_w, d_w])
        out.append((f"{prefix}_{k}", pts_w, eq, pts_w.min(0), pts_w.max(0)))
    return out


def check_metadata_and_blind_review():
    tj = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    assert tj["category"] == "Impact on jobs", f"Bad category: {tj['category']}"
    fid_len = len(tj["fidelity"])
    assert fid_len < 500, f"fidelity too long: {fid_len}"
    banned = tuple(t.strip().lower() for t in os.environ.get("BLIND_TERMS", "").split(",") if t.strip())
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in (".py", ".json", ".md") and p.name != "check_path.py":
            low = p.read_text(encoding="utf-8", errors="replace").lower()
            for b in banned:
                assert b not in low, f"Banned identifier '{b}' found in {p.name}"
    print(f"[PASS] task.json category='{tj['category']}', len(fidelity)={fid_len} < 500, 0 identifiers")


def check_3d_keyhole_undercut_interlock():
    # 0. Verify zero internal box-box overlap inside all 32 STRIP_BOXES:
    sh0 = _boxes_to_hulls("st", task.STRIP_BOXES, np.zeros(3), np.eye(3))
    for i in range(len(sh0)):
        for j in range(i + 1, len(sh0)):
            n1, v1, p1, lo1, hi1 = sh0[i]
            n2, v2, p2, lo2, hi2 = sh0[j]
            if np.all(np.minimum(hi1, hi2) - np.maximum(lo1, lo2) > 1e-6):
                assert not _polyhedra_intersect_exact(v1, p1, v2, p2), f"Internal STRIP_BOXES overlap: {n1} vs {n2}"

    p_rs = np.array([task.RISER_X0, task.RISER_Y0, task.RISER_Z0], float)
    rs_hulls = _boxes_to_hulls("rs", task.RISER_BOXES, p_rs, rz(task.RISER_YAW0))
    R_mnt = ry(-math.pi / 2.0)

    def hits_riser(x_rear: float, dy_slide: float, x_row: float = +0.0140) -> list:
        ps = np.array([x_rear, dy_slide, task.STUD_Z_W - x_row], float)
        st_h = _boxes_to_hulls("st", task.STRIP_BOXES, ps, R_mnt)
        hits = []
        for n1, v1, p1, lo1, hi1 in st_h:
            for n2, v2, p2, lo2, hi2 in rs_hulls:
                if np.all(np.minimum(hi1, hi2) - np.maximum(lo1, lo2) > 1e-6):
                    if _polyhedra_intersect_exact(v1, p1, v2, p2):
                        hits.append((n1, n2))
        return hits

    # Verify BOTH upper keyhole row (+0.0140 m) AND lower keyhole row (-0.0140 m) of the 101x28 mm TLP606 grid!:
    for row in (+0.0140, -0.0140):
        for x_mm in range(384, 415, 2):
            h = hits_riser(x_mm / 1000.0, 0.0, row)
            assert not h, f"Entry push (row={row}) hit riser/studs at x={x_mm}mm: {h}"
        for dy_mm in range(0, 13, 2):
            h = hits_riser(0.4148, dy_mm / 1000.0, row)
            assert not h, f"Lateral locking slide (row={row}) hit studs at dy={dy_mm}mm: {h}"
        h_pull = hits_riser(0.4095, +0.0124, row)
        hit_studs = {r for _, r in h_pull}
        assert "rs_3" in hit_studs and "rs_5" in hit_studs, f"Locked -X pull (row={row}) missed stud heads: {h_pull}"
        h_early = hits_riser(0.4066, +0.0124, row)
        early_studs = {r for _, r in h_early}
        assert "rs_3" in early_studs and "rs_5" in early_studs, f"Early slide (row={row}) missed stud heads: {h_early}"

    print(
        f"[PASS] 3D 4-slot (101x28 mm) TLP606 keyhole interlock verified across all {len(sh0)} boxes: "
        "0 internal overlaps; BOTH upper & lower keyhole rows pass entry + +12.4 mm lock & block -X pull-off"
    )


def check_scorer_counterexamples():
    m = task.Metric(vars(task))
    far_tips = np.tile(np.array([0.0, +0.48, task.TOP + 0.45]), (6, 1))
    p_rs = np.array([task.RISER_X0, task.RISER_Y0, task.RISER_Z0], float)
    R_rs = rz(task.RISER_YAW0)
    R_mnt = ry(-math.pi / 2.0)
    p_start = np.array([task.STRIP_X0, task.STRIP_Y0, task.STRIP_Z0], float)
    z_up = task.STUD_Z_W - 0.0140
    z_lo = task.STUD_Z_W + 0.0140
    p_locked_up = np.array([0.4148, +0.0124, z_up - 0.0008], float)
    p_locked_lo = np.array([0.4148, +0.0124, z_lo - 0.0008], float)
    p_entry_up = np.array([0.4148, 0.0000, z_up - 0.0038], float)

    def eval_case(ps=p_locked_up, Rs=R_mnt, tips=far_tips, drift=0.0):
        rss = FakeState(p_rs, R=R_rs)
        sts = FakeState(ps, R=Rs)
        pts0 = m.strip_corners @ Rs.T + ps
        hist = [pts0.copy() for _ in range(20)] + [pts0 + np.array([drift, 0.0, 0.0])]
        return m.evaluate_states(rss, sts, tips, hist_override=hist)

    # 1-2: Start state & dropped on bench score 0.0:
    assert eval_case(ps=p_start, Rs=np.eye(3)) == 0.0
    assert eval_case(ps=np.array([0.395, +0.012, task.TOP + 0.024])) == 0.0
    # 3-6: Locked on upper row, locked on lower row, OR locked with 8-deg pitch (no arbitrary 5-deg veto!) ALL score 1.0!:
    assert eval_case(ps=p_locked_up) == 1.0
    assert eval_case(ps=p_locked_lo) == 1.0
    assert eval_case(ps=np.array([0.4138, +0.0105, z_up - 0.0010]), Rs=R_mnt @ ry(math.radians(8.5))) == 1.0
    # Diagonal two-stud lock across (-0.0140, -0.0505) and (+0.0140, +0.0505) (~15.5 deg roll!) ALSO scores 1.0!:
    ang_diag = math.atan2(0.0280, 0.1010)
    R_diag = rx(-ang_diag) @ R_mnt
    # Find ps so stud 0 maps to (-0.0140, -0.0625, 0.0070) and stud 1 maps to (+0.0140, +0.0385, 0.0070):
    s0_w = np.asarray(task.STUD_HEAD_CENTERS_W[0], float)
    ps_diag = s0_w - R_diag @ np.array([-0.0140, -0.0620, 0.0072])
    assert eval_case(ps=ps_diag, Rs=R_diag) == 1.0
    # 7-9: Hung on entry bulbs without sliding left (or partial 5 mm slide) scores 0.5:
    assert eval_case(ps=p_entry_up) == 0.5
    assert eval_case(ps=np.array([0.4145, +0.0050, z_up - 0.0010])) == 0.5
    assert eval_case(ps=np.array([0.4148, 0.0000, z_lo - 0.0038])) == 0.5
    # 10-13: Pulled off riser face, outlet-inward against board, missed one stud, or protruding past riser Y -> 0.0:
    assert eval_case(ps=np.array([0.4060, +0.0124, z_up])) == 0.0
    assert eval_case(Rs=R_mnt @ ry(math.pi)) == 0.0
    assert eval_case(ps=np.array([0.4148, +0.0350, z_up])) == 0.0
    assert eval_case(ps=np.array([0.4148, +0.0800, z_up])) == 0.0
    # 14-16: Hand clearance < 80 mm from riser or strip scores 0.0:
    near_rs = far_tips.copy()
    near_rs[0] = np.array([0.380, 0.0, task.TOP + 0.14])
    assert eval_case(tips=near_rs) == 0.0
    near_st = far_tips.copy()
    near_st[3] = np.array([0.350, -0.09, z_up])
    assert eval_case(tips=near_st) == 0.0
    # Hand clearance < 80 mm from the articulated AC power cable / NEMA 5-15P plug ALSO scores 0.0:
    near_plug = far_tips.copy()
    plug_w = p_locked_up + R_mnt @ np.array([-0.0860, -0.1625, 0.0108])
    near_plug[4] = plug_w + np.array([-0.045, 0.0, 0.0])
    assert eval_case(tips=near_plug) == 0.0
    # 17-18: Stillness >= 1.0 mm or fallen off bench scores 0.0:
    assert eval_case(drift=0.0015) == 0.0
    assert eval_case(ps=np.array([0.30, 0.0, task.TOP - 0.045])) == 0.0
    print("[PASS] 18/18 CPU scorer counterexamples verified (including upper/lower row, 8.5-deg pitch, 15.5-deg diagonal, and cord/plug clearance!)")


def _slerp_ry(a0: float, a1: float, u: float) -> np.ndarray:
    return ry(a0 + u * (a1 - a0))


def _expected_strip_pose(mode: str, t: float):
    early_slide = (mode == "early_slide_push")
    entry_only = (mode == "entry_only_no_slide")
    y0 = +0.0124 if early_slide else 0.0000
    z_align = task.STUD_Z_W - task.KEYHOLE_X_ROWS[1]

    p0 = np.array([task.STRIP_X0, task.STRIP_Y0, task.STRIP_Z0], float)
    p_lift = np.array([task.STRIP_X0, 0.0000, task.TOP + 0.0880], float)
    p_mid = np.array([0.3150, 0.0000, z_align + 0.0060], float)
    p_stage = np.array([0.3840, y0, z_align], float)
    p_press = np.array([0.4015 if early_slide else 0.4148, y0, z_align], float)
    p_lock = np.array([0.4148, +0.0124, z_align - 0.0008], float)

    if t <= 2.75:
        return p0, np.eye(3)
    if t <= 3.80:
        u = ik.smoothstep((t - 2.75) / 1.05)
        return (1.0 - u) * p0 + u * p_lift, np.eye(3)
    if t <= 5.05:
        u = ik.smoothstep((t - 3.80) / 1.25)
        return (1.0 - u) * p_lift + u * p_mid, _slerp_ry(0.0, -math.pi / 4.0, u)
    if t <= 6.30:
        u = ik.smoothstep((t - 5.05) / 1.25)
        return (1.0 - u) * p_mid + u * p_stage, _slerp_ry(-math.pi / 4.0, -math.pi / 2.0, u)
    if t <= 7.70:
        return p_stage, ry(-math.pi / 2.0)
    if t <= 9.00:
        u = ik.smoothstep((t - 7.70) / 1.30)
        return (1.0 - u) * p_stage + u * p_press, ry(-math.pi / 2.0)

    if early_slide:
        return p_press, ry(-math.pi / 2.0)
    if entry_only:
        p_entry_rest = np.array([0.4148, 0.0000, z_align - 0.0031], float)
        if t <= 9.55:
            u = ik.smoothstep(min(1.0, (t - 9.00) / 0.55))
            return (1.0 - u) * p_press + u * p_entry_rest, ry(-math.pi / 2.0)
        return p_entry_rest, ry(-math.pi / 2.0)

    p_slide = np.array([0.4148, +0.0124, z_align], float)
    p_lock = np.array([0.4148, +0.0124, z_align - 0.0007], float)
    if t <= 10.35:
        u = ik.smoothstep(min(1.0, (t - 9.00) / 1.35))
        return (1.0 - u) * p_press + u * p_slide, ry(-math.pi / 2.0)
    if t <= 10.90:
        u = ik.smoothstep(min(1.0, (t - 10.35) / 0.55))
        return (1.0 - u) * p_slide + u * p_lock, ry(-math.pi / 2.0)
    return p_lock, ry(-math.pi / 2.0)


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
    p_rs = np.array([task.RISER_X0, task.RISER_Y0, task.RISER_Z0], float)
    rs_hulls = _boxes_to_hulls("rs", task.RISER_BOXES, p_rs, rz(task.RISER_YAW0))

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

    for preload_tag, preload_val in (("achieved_kinematic", 0.0), ("pd_preloaded_cmd", None)):
        for mode in ("demo", "entry_only_no_slide", "early_slide_push"):
            task._MODE = mode
            task._STUD_TRIM = None
            task._Z_PRELOAD_OVERRIDE = preload_val
            plan = task.demo()
            aps = [ArmPlan(plan["right"], RIGHT_BASE_POS), ArmPlan(plan["left"], LEFT_BASE_POS)]
            duration = max(sum(w.duration for w in v) for v in plan.values())
            worst_pos = worst_rot = 0.0
            inter_hits = []
            below_hits = []
            unintended_hits = []
            min_inter_aabb_mm = 999.0
            min_top_mm = 999.0
            t_rel_done = 9.80 if mode == "early_slide_push" else (10.35 if mode == "entry_only_no_slide" else 11.65)
            all_strip_cord_boxes = list(task.STRIP_BOXES) + list(task.STRIP_CORD_REST_BOXES)

            for t in np.arange(0.0, duration + 0.001, 0.05):
                ps, Rs = _expected_strip_pose(mode, float(t))
                st_hulls = _boxes_to_hulls("st", all_strip_cord_boxes, ps, Rs)

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
                    if _polyhedra_intersect_exact(
                        aa[1], world_planes(aa[2], aa[3], aa[4]), bb[1], world_planes(bb[2], bb[3], bb[4])
                    ):
                        inter_hits.append((round(float(t), 2), aa[0], bb[0]))

                for side, arm_b in (("right", bodies[0]), ("left", bodies[1])):
                    for lname, vv, planes, R_l, p_l, lo, hi in arm_b:
                        if lname in ("link1", "link2"):
                            continue
                        dz_top = float(lo[2] - task.TOP) * 1000.0
                        min_top_mm = min(min_top_mm, dz_top)
                        if lo[2] < task.TOP + 0.0040:
                            below_hits.append((round(float(t), 2), side, lname, round(dz_top, 2)))

                if preload_val is not None and mode != "early_slide_push":
                    for fn, fv, fp, flo, fhi in st_hulls:
                        for sn, sv, sp, slo, shi in rs_hulls:
                            if np.all(np.minimum(fhi, shi) - np.maximum(flo, slo) > 1e-6):
                                if _polyhedra_intersect_exact(fv, fp, sv, sp):
                                    unintended_hits.append((round(float(t), 2), "strip_vs_riser", fn, sn))

                for side, arm_b in (("right", bodies[0]), ("left", bodies[1])):
                    for lname, vv, planes, R_l, p_l, lo, hi in arm_b:
                        eq_a = world_planes(planes, R_l, p_l)
                        for sn, sv, sp, slo, shi in rs_hulls:
                            if np.all(np.minimum(hi, shi) - np.maximum(lo, slo) > 1e-6):
                                if _polyhedra_intersect_exact(vv, eq_a, sv, sp):
                                    unintended_hits.append((round(float(t), 2), f"{side}_vs_riser", lname, sn))
                        if preload_val is not None:
                            if lname not in ("tip_left", "tip_right"):
                                for sn, sv, sp, slo, shi in st_hulls:
                                    if np.all(np.minimum(hi, shi) - np.maximum(lo, slo) > 1e-6):
                                        if _polyhedra_intersect_exact(vv, eq_a, sv, sp):
                                            unintended_hits.append((round(float(t), 2), f"{side}_link_vs_strip", lname, sn))
                            elif float(t) <= 1.90 or float(t) >= t_rel_done:
                                for sn, sv, sp, slo, shi in st_hulls:
                                    if np.all(np.minimum(hi, shi) - np.maximum(lo, slo) > 1e-6):
                                        if _polyhedra_intersect_exact(vv, eq_a, sv, sp):
                                            unintended_hits.append((round(float(t), 2), f"{side}_tip_approach_strip", lname, sn))

            print(
                f"[INFO] [{preload_tag}] mode={mode}: dur={duration:.2f}s worst_pos={worst_pos:.4f}mm worst_rot={worst_rot:.4f}deg "
                f"inter_hits={len(inter_hits)} below_top={len(below_hits)} unintended_hits={len(unintended_hits)} "
                f"min_inter_aabb={min_inter_aabb_mm:.2f}mm min_top={min_top_mm:.2f}mm"
            )
            if inter_hits or below_hits or unintended_hits:
                print("  inter_hits:", inter_hits[:5])
                print("  below_hits:", below_hits[:5])
                print("  unintended_hits:", unintended_hits[:10])
            assert worst_pos <= 0.50 and worst_rot <= 0.25, f"IK error too large: {worst_pos}mm, {worst_rot}deg"
            assert not inter_hits and not below_hits and not unintended_hits, f"[{preload_tag}] {mode} failed hull checks!"
    task._Z_PRELOAD_OVERRIDE = None


if __name__ == "__main__":
    check_metadata_and_blind_review()
    check_3d_keyhole_undercut_interlock()
    check_scorer_counterexamples()
    check_full_trajectory()
    print("ALL PREFLIGHT CHECKS PASSED")
