"""289-hull installed-robot ArmPlan.action7(t) + FILE_FRAME/INBOX_RISER/REAR_FOLDERS + swept-FOLDER/PACKET collision screen + 170 scorer witnesses."""
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
from task_metrics import box_rot, part_local_corners, ry
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
    rst_len = len(tj.get("reset_notes", ""))
    assert fid_len < 500, f"fidelity too long: {fid_len}"
    assert rst_len < 1500, f"reset_notes too long: {rst_len}"
    assert "Possible scores: 0, 0.5, 1.0." in tj["scoring"], "Missing Possible scores"
    assert "Real trial:" in tj["scoring"], "Missing Real trial sentence"
    assert "Closest look-alikes:" in tj["context"], "Missing Closest look-alikes"
    print(
        f"[PASS] task.json category='{tj['category']}', len(fidelity)={fid_len} < 500, "
        f"len(reset_notes)={rst_len} < 1500"
    )


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


def check_boxes_pair(boxes_a, pos_a, R_a, boxes_b, pos_b, R_b, label: str):
    worst = 0.0
    for ba in boxes_a:
        for bb in boxes_b:
            worst = max(worst, obb_overlap(ba, pos_a, R_a, bb, pos_b, R_b))
    assert worst <= 1e-4, f"{label} overlap: {worst * 1000:.3f} mm"
    print(f"[PASS] {label}: max OBB penetration = {worst * 1000:.4f} mm")


def check_static_sections():
    I3 = np.eye(3)
    p_ris = np.asarray(task.INBOX_RISER.pos, float)
    p_frm = np.asarray(task.FILE_FRAME.pos, float)
    p_fld0 = np.asarray(task.FOLDER.pos, float)
    p_pkt0 = np.asarray(task.PACKET.pos, float)
    p_rf1 = np.asarray(task.REAR_FOLDER_1.pos, float)
    p_rf2 = np.asarray(task.REAR_FOLDER_2.pos, float)

    check_boxes_pair(task.INBOX_RISER.boxes, p_ris, I3, task.FILE_FRAME.boxes, p_frm, I3, "t=0 RISER vs FRAME")
    check_boxes_pair(task.INBOX_RISER.boxes, p_ris, I3, task.FRONT_LEAF_BOXES, p_fld0, I3, "t=0 RISER vs FRONT_LEAF")
    check_boxes_pair(task.INBOX_RISER.boxes, p_ris, I3, task.REAR_LEAF_BOXES, p_fld0, I3, "t=0 RISER vs REAR_LEAF")
    check_boxes_pair(task.INBOX_RISER.boxes, p_ris, I3, task.PACKET.boxes, p_pkt0, I3, "t=0 RISER vs PACKET")
    check_boxes_pair(task.FILE_FRAME.boxes, p_frm, I3, task.FRONT_LEAF_BOXES, p_fld0, I3, "t=0 FRAME vs FRONT_LEAF")
    check_boxes_pair(task.FILE_FRAME.boxes, p_frm, I3, task.REAR_LEAF_BOXES, p_fld0, I3, "t=0 FRAME vs REAR_LEAF")
    check_boxes_pair(task.FRONT_LEAF_BOXES, p_fld0, I3, task.PACKET.boxes, p_pkt0, I3, "t=0 FRONT_LEAF vs PACKET")
    check_boxes_pair(task.REAR_LEAF_BOXES, p_fld0, I3, task.PACKET.boxes, p_pkt0, I3, "t=0 REAR_LEAF vs PACKET")
    check_boxes_pair(task.FILE_FRAME.boxes, p_frm, I3, task.REAR_FOLDER_1.boxes, p_rf1, I3, "t=0 FRAME vs REAR_FOLDER_1")
    check_boxes_pair(task.FILE_FRAME.boxes, p_frm, I3, task.REAR_FOLDER_2.boxes, p_rf2, I3, "t=0 FRAME vs REAR_FOLDER_2")
    check_boxes_pair(task.REAR_FOLDER_1.boxes, p_rf1, I3, task.REAR_FOLDER_2.boxes, p_rf2, I3, "t=0 REAR_FOLDER_1 vs REAR_FOLDER_2")

    R_hang = ry(-0.5 * math.pi)
    p_fld_goal = np.array([task.X_HANG, 0.0, task.RAIL_Z_TOP - task.ROD_SEAT_X + 0.0002])
    p_pkt_goal = p_fld_goal + R_hang @ np.array([0.1050, 0.0, 0.0058])
    check_boxes_pair(task.FILE_FRAME.boxes, p_frm, I3, task.FRONT_LEAF_BOXES, p_fld_goal, R_hang, "Goal FRAME vs FRONT_LEAF")
    check_boxes_pair(task.FILE_FRAME.boxes, p_frm, I3, task.REAR_LEAF_BOXES, p_fld_goal, R_hang, "Goal FRAME vs REAR_LEAF")
    check_boxes_pair(task.FRONT_LEAF_BOXES, p_fld_goal, R_hang, task.PACKET.boxes, p_pkt_goal, R_hang, "Goal FRONT_LEAF vs PACKET")
    check_boxes_pair(task.REAR_LEAF_BOXES, p_fld_goal, R_hang, task.PACKET.boxes, p_pkt_goal, R_hang, "Goal REAR_LEAF vs PACKET")
    check_boxes_pair(task.FRONT_LEAF_BOXES, p_fld_goal, R_hang, task.REAR_FOLDER_1.boxes, p_rf1, I3, "Goal FRONT_LEAF vs REAR_FOLDER_1")
    check_boxes_pair(task.REAR_LEAF_BOXES, p_fld_goal, R_hang, task.REAR_FOLDER_1.boxes, p_rf1, I3, "Goal REAR_LEAF vs REAR_FOLDER_1")
    check_boxes_pair(task.PACKET.boxes, p_pkt_goal, R_hang, task.REAR_FOLDER_1.boxes, p_rf1, I3, "Goal PACKET vs REAR_FOLDER_1")


def check_scorer_counterexamples():
    m = task.Metric(vars(task))
    I3 = np.eye(3)
    R_hang = ry(-0.5 * math.pi)
    p_fld0 = np.asarray(task.FOLDER.pos, float)
    p_pkt0 = np.asarray(task.PACKET.pos, float)
    p_fld_goal = np.array([task.X_HANG, 0.0, task.RAIL_Z_TOP - task.ROD_SEAT_X])
    p_pkt_goal = p_fld_goal + R_hang @ np.array([0.1050, 0.0, 0.0058])
    far_tips = np.tile(np.array([[0.10, +0.35, 0.95], [0.10, -0.35, 0.95]]), (3, 1))
    default_rears = [FakeState(rf.pos, I3) for rf in task.REAR_FOLDERS]

    def eval_static(fld_s, pkt_s, tips=far_tips, rear_states=None, jitter=0.0, reset_first=True):
        if reset_first:
            m.reset()
        rs = default_rears if rear_states is None else rear_states
        pts = m.all_world_corners(fld_s, pkt_s, rs)
        hist = [pts + (np.array([0, 0, jitter]) if k == 20 else 0.0) for k in range(21)]
        return m.evaluate_states(fld_s, pkt_s, tips, rear_states=rs, hist_override=hist)

    assert eval_static(FakeState(p_fld0), FakeState(p_pkt0)) == 0.0
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang)) == 1.0
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt0, I3)) == 0.5
    assert eval_static(FakeState(p_fld_goal, R_hang, {"rear_leaf": -0.60}), FakeState(p_pkt0, I3)) == 0.0
    rx8 = np.array([[1, 0, 0], [0, math.cos(0.14), -math.sin(0.14)], [0, math.sin(0.14), math.cos(0.14)]])
    assert eval_static(FakeState(p_fld_goal, rx8 @ R_hang), FakeState(p_pkt_goal, R_hang)) == 0.0
    assert eval_static(FakeState([task.X_HANG, 0.0, task.FRAME_BASE_TOP], R_hang), FakeState(p_pkt_goal, R_hang)) == 0.0
    near_tips = far_tips.copy()
    near_tips[0] = p_fld_goal + np.array([-0.060, 0.0, 0.120])
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), tips=near_tips) == 0.0
    near_rail_tips = far_tips.copy()
    near_rail_tips[0] = np.array([task.RAIL_X_MAX - 0.010, +task.RAIL_Y, task.RAIL_Z_TOP + 0.070])
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), tips=near_rail_tips) == 0.0
    near_riser_tips = far_tips.copy()
    near_riser_tips[0] = np.array([task.INBOX_RISER.pos[0], 0.0, task.TOP + task.RISER_H + 0.070])
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), tips=near_riser_tips) == 0.0
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), jitter=0.0015) == 0.0
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal + [0, 0.080, 0], R_hang)) == 0.5
    assert eval_static(FakeState([task.RAIL_X_MIN + 0.005, 0.0, task.RAIL_Z_TOP - task.ROD_SEAT_X], R_hang), FakeState(p_pkt_goal, R_hang)) == 0.0
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState([0.20, 0.0, task.TOP - 0.050], I3)) == 0.0
    eval_static(FakeState([0.05, 0.0, task.TOP - 0.050], R_hang), FakeState([0.05, 0.0, task.TOP - 0.040], R_hang))
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), reset_first=False) == 1.0

    # Neighbor rear-folder witnesses (hand clearance, dropped neighbor, harmless +15 mm slide along rail):
    near_rear_tips = far_tips.copy()
    near_rear_tips[0] = np.array([task.REAR_FOLDER_1.pos[0], 0.0, task.RAIL_Z_TOP + 0.060])
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), tips=near_rear_tips) == 0.0
    dropped_rears = [FakeState([0.635, 0.0, task.TOP - 0.050], I3), FakeState(task.REAR_FOLDER_2.pos, I3)]
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), rear_states=dropped_rears) == 0.0
    bumped_rears = [FakeState(np.asarray(task.REAR_FOLDER_1.pos) + [0.015, 0.0, 0.0], I3), FakeState(task.REAR_FOLDER_2.pos, I3)]
    assert eval_static(FakeState(p_fld_goal, R_hang), FakeState(p_pkt_goal, R_hang), rear_states=bumped_rears) == 1.0

    # Witness (TK-SJ4K8R counterexample): rod balanced +4.0 mm sideways and +4.49 mm high on inner lug tips
    # outside the 7.2 mm saddle void (within old 6.0 mm / 4.5 mm point tolerances) MUST score 0.0:
    p_outside_saddle = np.array([task.X_HANG, +0.0040, task.RAIL_Z_TOP - task.ROD_SEAT_X + 0.00449])
    p_pkt_outside = p_outside_saddle + R_hang @ np.array([0.1050, 0.0, 0.0058])
    assert eval_static(FakeState(p_outside_saddle, R_hang), FakeState(p_pkt_outside, R_hang)) == 0.0
    assert m.detail["rail_engaged"] == [False, False, False, False], m.detail
    assert m.detail["min_saddle_side_margin_mm"] < -2.0, m.detail

    # Grid audit: 140 valid in-saddle placements -> 1.0, and 10 out-of-saddle / unseated offsets -> 0.0:
    for dy in np.linspace(-0.0017, +0.0017, 35):
        for dz in (-0.00005, 0.0, 0.00020, 0.00045):
            pf_v = np.array([task.X_HANG, float(dy), task.RAIL_Z_TOP - task.ROD_SEAT_X + dz])
            pp_v = pf_v + R_hang @ np.array([0.1050, 0.0, 0.0058])
            assert eval_static(FakeState(pf_v, R_hang), FakeState(pp_v, R_hang)) == 1.0, (dy, dz, m.detail)
    for dy in (-0.006, -0.004, -0.002, +0.002, +0.004, +0.006):
        pf_n = np.array([task.X_HANG, dy, task.RAIL_Z_TOP - task.ROD_SEAT_X])
        pp_n = pf_n + R_hang @ np.array([0.1050, 0.0, 0.0058])
        assert eval_static(FakeState(pf_n, R_hang), FakeState(pp_n, R_hang)) == 0.0, (dy, m.detail)
    for dz in (-0.00020, +0.00070, +0.00200, +0.00449):
        pf_n = np.array([task.X_HANG, 0.0, task.RAIL_Z_TOP - task.ROD_SEAT_X + dz])
        pp_n = pf_n + R_hang @ np.array([0.1050, 0.0, 0.0058])
        assert eval_static(FakeState(pf_n, R_hang), FakeState(pp_n, R_hang)) == 0.0, (dz, m.detail)
    print(
        "[PASS] 169/169 CPU scorer checks verified (140 valid in-saddle -> 1.0, "
        "outside-saddle lug-tip balance + 10 offsets -> 0.0, 18 structural & neighbor cases)"
    )


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
        out.append((f"{prefix}_box_{idx}", vv, np.asarray(planes, float), vv.min(0), vv.max(0)))
    return out


def _scene_box_hulls():
    hulls = (
        _boxes_to_hulls("riser", task.INBOX_RISER.boxes, np.asarray(task.INBOX_RISER.pos, float))
        + _boxes_to_hulls("frame", task.FILE_FRAME.boxes, np.asarray(task.FILE_FRAME.pos, float))
    )
    for idx, rf in enumerate(task.REAR_FOLDERS, start=1):
        hulls = hulls + _boxes_to_hulls(f"rear_folder_{idx}", rf.boxes, np.asarray(rf.pos, float))
    return hulls


def _polyhedra_intersect_exact(va, pa, vb, pb) -> bool:
    if np.any(np.min(vb @ pa[:, :3].T + pa[:, 3], axis=0) > 1e-6):
        return False
    if np.any(np.min(va @ pb[:, :3].T + pb[:, 3], axis=0) > 1e-6):
        return False
    P = np.vstack([pa, pb])
    res = linprog(np.zeros(3), A_ub=P[:, :3], b_ub=-P[:, 3] - 1e-6, bounds=(None, None), method="highs")
    return bool(res.success)


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

    scene_hulls = _scene_box_hulls()
    p_fld0 = np.asarray(task.FOLDER.pos, float)
    p_pkt0 = np.asarray(task.PACKET.pos, float)
    pregrasp_loose_hulls = (
        _boxes_to_hulls("fld0_front", task.FRONT_LEAF_BOXES, p_fld0)
        + _boxes_to_hulls("fld0_rear", task.REAR_LEAF_BOXES, p_fld0)
        + _boxes_to_hulls("pkt0", task.PACKET.boxes, p_pkt0)
    )
    R_hang = ry(-0.5 * math.pi)
    p_fld_goal = np.array([task.X_HANG, 0.0, task.RAIL_Z_TOP - task.ROD_SEAT_X])
    p_pkt_goal = p_fld_goal + R_hang @ np.array([0.1050, 0.0, 0.0058])
    post_loose_hulls = (
        _boxes_to_hulls("fldg_front", task.FRONT_LEAF_BOXES, p_fld_goal, R_hang)
        + _boxes_to_hulls("fldg_rear", task.REAR_LEAF_BOXES, p_fld_goal, R_hang)
        + _boxes_to_hulls("pktg", task.PACKET.boxes, p_pkt_goal, R_hang)
    )
    bases = [np.array(RIGHT_BASE_POS, float), np.array(LEFT_BASE_POS, float)]
    I3 = np.eye(3)
    p_ris = np.asarray(task.INBOX_RISER.pos, float)
    p_frm = np.asarray(task.FILE_FRAME.pos, float)
    pkt_boxes_in_fld = [
        (tuple(np.asarray(b[0], float) + np.array([0.1050, 0.0, 0.0058])), b[1], b[2])
        for b in task.PACKET.boxes
    ]
    all_fld_boxes = list(task.FRONT_LEAF_BOXES) + list(task.REAR_LEAF_BOXES) + pkt_boxes_in_fld
    fld_corners_local = part_local_corners(all_fld_boxes)
    q_grasp_local = np.array([task.GRASP_X_LOCAL, 0.0, 0.00585], float)

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

    for mode in ("", "one_hand_only", "unclamped_lift"):
        task._PROBE = mode
        plan = task.demo()
        aps = [ArmPlan(plan["right"], RIGHT_BASE_POS), ArmPlan(plan["left"], LEFT_BASE_POS)]
        duration = max(sum(w.duration for w in v) for v in plan.values())
        qs = [np.array(READY, float), np.array(READY, float)]
        worst_pos = worst_rot = 0.0
        hits = []
        scene_hits = []
        swept_fld_hits = []
        below = []
        min_top_mm = 999.0
        min_fld_top_mm = 999.0
        min_inter_aabb_mm = 999.0
        min_scene_aabb_mm = 999.0
        dt = 0.05 if mode == "" else 0.10
        for t in np.arange(0.0, duration + 0.001, dt):
            bodies = []
            tcp_acts = []
            rot_acts = []
            for i in range(2):
                tcp, rot, f, _ = aps[i].target(float(t))
                act7, _ = aps[i].action7(float(t))
                qs[i] = np.asarray(act7[:6], float)
                tcp_act, rot_act = ik.tcp_pose(qs[i], bases[i])
                tcp_acts.append(tcp_act)
                rot_acts.append(rot_act)
                pe = float(np.linalg.norm(tcp_act - tcp))
                cos_ang = float(np.clip((np.trace(rot_act.T @ rot) - 1.0) * 0.5, -1.0, 1.0))
                re = math.acos(cos_ang)
                worst_pos = max(worst_pos, pe * 1000.0)
                worst_rot = max(worst_rot, float(np.degrees(re)))
                bodies.append(placed(i, qs[i], float(act7[6]) * float(task.FINGER_OPEN)))
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
            active_hulls = list(scene_hulls)
            if t <= 2.50:
                active_hulls = active_hulls + pregrasp_loose_hulls
            elif mode == "" and t >= 11.80:
                active_hulls = active_hulls + post_loose_hulls
            aflo = np.array([fh[3] for fh in active_hulls])
            afhi = np.array([fh[4] for fh in active_hulls])
            for arm_idx, (rlo, rhi) in enumerate([(alo, ahi), (blo, bhi)]):
                fgap = np.maximum(0.0, np.maximum(rlo[:, None, :] - afhi[None, :, :], aflo[None, :, :] - rhi[:, None, :]))
                min_scene_aabb_mm = min(min_scene_aabb_mm, float(np.min(np.linalg.norm(fgap, axis=-1)) * 1000.0))
                fcands = np.all(np.minimum(rhi[:, None], afhi[None, :]) - np.maximum(rlo[:, None], aflo[None, :]) > 1e-6, axis=-1)
                for ir, jf in np.argwhere(fcands):
                    rr = bodies[arm_idx][ir]
                    ff = active_hulls[jf]
                    pr = world_planes(rr[2], rr[3], rr[4])
                    if _polyhedra_intersect_exact(rr[1], pr, ff[1], ff[2]):
                        scene_hits.append([round(float(t), 2), arm_idx, rr[0], ff[0]])
            # Swept carried-FOLDER+PACKET OBB check during clamped carry/seating phase (t in [3.75, 10.00] s):
            if mode == "" and 3.75 <= t <= 10.00:
                c_r = tcp_acts[0] + task.PAD_OFFSET * rot_acts[0][:, 2]
                c_l = tcp_acts[1] + task.PAD_OFFSET * rot_acts[1][:, 2]
                c_mid = 0.5 * (c_r + c_l)
                R_fld = rot_acts[1] @ task.ROT_FLAT_L.T
                p_fld = c_mid - R_fld @ q_grasp_local
                fc_world = fld_corners_local @ R_fld.T + p_fld
                fld_z_clear_mm = float(np.min(fc_world[:, 2]) - task.TOP) * 1000.0
                min_fld_top_mm = min(min_fld_top_mm, fld_z_clear_mm)
                if fld_z_clear_mm < 2.0:
                    swept_fld_hits.append([round(float(t), 2), "table", round(fld_z_clear_mm, 2)])
                for fb in all_fld_boxes:
                    if t >= 4.40:
                        for idx_r, rb in enumerate(task.INBOX_RISER.boxes):
                            pen = obb_overlap(fb, p_fld, R_fld, rb, p_ris, I3)
                            if pen > 2e-4:
                                swept_fld_hits.append([round(float(t), 2), f"riser_{idx_r}", round(pen * 1000.0, 2)])
                    for idx_f, mb in enumerate(task.FILE_FRAME.boxes):
                        pen = obb_overlap(fb, p_fld, R_fld, mb, p_frm, I3)
                        if pen > 2e-4:
                            swept_fld_hits.append([round(float(t), 2), f"frame_{idx_f}", round(pen * 1000.0, 2)])
                    for rfi, rf_spec in enumerate(task.REAR_FOLDERS, start=1):
                        p_rf = np.asarray(rf_spec.pos, float)
                        for idx_rf, rfb in enumerate(rf_spec.boxes):
                            pen = obb_overlap(fb, p_fld, R_fld, rfb, p_rf, I3)
                            if pen > 2e-4:
                                swept_fld_hits.append([round(float(t), 2), f"rear{rfi}_{idx_rf}", round(pen * 1000.0, 2)])
            for i in range(2):
                lows = [
                    (v[0], float(v[5][2] - task.TOP) * 1000.0)
                    for v in bodies[i]
                    if v[0] != "link1"
                ]
                mn = min(lows, key=lambda v: v[1])
                min_top_mm = min(min_top_mm, mn[1])
                if mn[1] < 0.0:
                    below.append([round(float(t), 2), i, mn])
        info = dict(
            duration=round(duration, 2),
            max_pos_mm=round(worst_pos, 4),
            max_rot_deg=round(worst_rot, 4),
            min_top_mm=round(min_top_mm, 2),
            min_fld_top_mm=round(min_fld_top_mm, 2) if mode == "" else None,
            min_inter_aabb_mm=round(min_inter_aabb_mm, 2),
            min_scene_aabb_mm=round(min_scene_aabb_mm, 2),
            inter_arm_hits=len(hits),
            scene_hits=len(scene_hits),
            swept_fld_hits=len(swept_fld_hits),
            below_top=len(below),
        )
        print(f"[INFO] {mode or 'demo'}: {json.dumps(info)}")
        assert worst_pos < 0.5, f"{mode}: IK pos err {worst_pos} mm >= 0.5 mm"
        assert worst_rot < 0.5, f"{mode}: IK rot err {worst_rot} deg >= 0.5 deg"
        assert len(hits) == 0, f"{mode}: inter-arm collisions: {hits[:5]}"
        assert len(scene_hits) == 0, f"{mode}: robot-vs-scene collisions: {scene_hits[:5]}"
        assert len(swept_fld_hits) == 0, f"{mode}: swept-folder collisions: {swept_fld_hits[:5]}"
        assert len(below) == 0, f"{mode}: below table: {below[:5]}"


if __name__ == "__main__":
    check_metadata()
    check_static_sections()
    check_scorer_counterexamples()
    check_full_trajectory()
    print("ALL OFFLINE PREFLIGHT CHECKS PASSED")
