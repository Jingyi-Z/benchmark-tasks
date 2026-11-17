"""Offline check of the scripted demo, no simulator: replays both arms' commanded joints (ArmPlan.action7, as the run
does) every 0.05 s and reports (1) joints within 0.1 rad of a limit, (2) TCP tracking error over 2 mm or 2 deg, and
(3) any arm collision hull (installed YAM URDF) overlapping the dash, the stand, the hanging glove box (after it has
dropped) or a filter while it rests untouched.

Usage (yam_arena importable):  python check_path.py
"""
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import task  # noqa: E402
from yam_arena import YAM_URDF, ik  # noqa: E402
from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402


def _rpy(r):
    cr, sr, cp, sp, cy, sy = math.cos(r[0]), math.sin(r[0]), math.cos(r[1]), math.sin(r[1]), math.cos(r[2]), math.sin(r[2])
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _stl(path):
    data = pathlib.Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [list(map(float, ln.split()[1:4])) for ln in data.decode(errors="ignore").splitlines() if ln.strip().startswith("vertex")]
        return np.array(v, float)
    n = int.from_bytes(data[80:84], "little")
    a = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


def _box_hull(c, h, R=np.eye(3), p=np.zeros(3)):
    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h + c
    v = v @ R.T + p
    hull = ConvexHull(v)
    return v, hull.equations


def _overlap(va, pa, vb, pb):
    if np.any(np.min(vb @ pa[:, :3].T + pa[:, 3], axis=0) > 1e-6):
        return False
    if np.any(np.min(va @ pb[:, :3].T + pb[:, 3], axis=0) > 1e-6):
        return False
    P = np.vstack([pa, pb])
    r = linprog(np.zeros(3), A_ub=P[:, :3], b_ub=-P[:, 3] - 1e-6, bounds=(None, None), method="highs")
    return r.status == 0


def main():
    urdf = pathlib.Path(YAM_URDF)
    root = ET.parse(urdf).getroot()
    lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
           for j in root.findall("joint") if j.find("limit") is not None}
    shapes = []
    for link in root.findall("link"):
        for c in link.findall("collision"):
            mesh = c.find("geometry/mesh")
            if mesh is None:
                continue
            o = c.find("origin")
            R = _rpy([float(x) for x in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.eye(3)
            p = np.array([float(x) for x in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            sc = np.array([float(x) for x in (mesh.get("scale") or "1 1 1").split()])
            v = (_stl((urdf.parent / mesh.get("filename")).resolve()) * sc) @ R.T + p
            v = np.unique(np.round(v, 6), axis=0)
            h = ConvexHull(v)
            shapes.append((link.get("name"), v[h.vertices]))
    print(f"{len(shapes)} arm hulls per arm")

    T = task
    dash_p = np.array([T.DASH_X, 0.0, T.TOP])
    def _rz(b):
        if len(b) < 4:
            return np.eye(3)
        y = float(b[3][0] if isinstance(b[3], (tuple, list)) else b[3])
        return np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])

    static = [(f"dash{i}", *_box_hull(np.zeros(3), np.array(b[1], float) / 2, _rz(b), dash_p + np.array(b[0], float)))
              for i, b in enumerate(T.DASH_BOXES)]
    static += [("stand", *_box_hull(np.array(b[0], float), np.array(b[1], float) / 2, np.eye(3),
                                    np.array([T.STAND_XY[0], T.STAND_XY[1], T.TOP]))) for b in T.STAND.boxes[:1]]
    hinge = np.array([0.0, 0.0, T.HZ])

    def box_at(deg, tag):
        a = math.radians(deg)
        Ry = np.array([[math.cos(-a), 0, math.sin(-a)], [0, 1, 0], [-math.sin(-a), 0, math.cos(-a)]])
        return [(f"{tag}{i}", *_box_hull(Ry @ (np.array(b[0], float) - hinge) + hinge, np.array(b[1], float) / 2, Ry,
                                         dash_p)) for i, b in enumerate(T.BOX_BOXES)]
    box_hang, box_rest = box_at(92.0, "box_hanging"), box_at(45.0, "box_at_45deg")
    fh = np.array([T.F_L / 2, T.F_W / 2, T.F_T / 2])
    new0 = ("new_filter_resting", *_box_hull(np.zeros(3), fh, np.eye(3), np.array(T.NEW.pos, float)))
    old0 = ("old_filter_resting", *_box_hull(np.zeros(3), fh, np.eye(3), np.array(T.OLD.pos, float)))

    plan = T.demo()
    aps = {"right": ArmPlan(plan["right"], RIGHT_BASE_POS), "left": ArmPlan(plan["left"], LEFT_BASE_POS)}
    bases = {"right": np.array(RIGHT_BASE_POS, float), "left": np.array(LEFT_BASE_POS, float)}
    dur = max(sum(w.duration for w in v) for v in plan.values())

    def first_time(arm, word):
        acc = 0.0
        for w in plan[arm]:
            if word in w.label:
                return acc
            acc += w.duration
        return None
    t_old = first_time("right", "pinch the old element") or 1e9
    t_new = first_time("right", "pinch the new element") or 1e9
    t_drop = (first_time("right", "let go: the box swings down") or 14.0) + 1.0
    t_grab = first_time("right", "carry the box down") or t_drop
    print(f"plan {dur:.1f} s; old filter untouched until {t_old:.1f} s, new until {t_new:.1f} s")
    # carried elements, phase by phase: each element follows the right hand rigidly while the fingers are closed inside
    # its holding window, with the grasp offset taken at each (re)closure from where the element then is; while the
    # fingers are open (the two 60 mm steps along the edge, the release before the tab) it stays where it was left
    windows = {"old": (first_time("right", "pinch the old element"), first_time("right", "let go: it drops")),
               "new": (first_time("right", "pinch the new element"), first_time("right", "let go of the tab"))}
    pose = {"old": (np.array(T.OLD.pos, float), np.eye(3)), "new": (np.array(T.NEW.pos, float), np.eye(3))}
    grip = {"old": None, "new": None}
    carried_hits = {}

    def carried(t, tcp, rot, closed):
        """Update and return (name, (centre, rotation)) of the element the right hand holds at t, else None."""
        out = None
        for e, (t0, t1) in windows.items():
            if t0 is None or t1 is None or not (t0 <= t < t1):
                grip[e] = None
                continue
            if closed:
                if grip[e] is None:  # (re)closure: the grasp offset from where the element is now
                    pe, Re = pose[e]
                    grip[e] = (np.asarray(rot).T @ (pe - np.asarray(tcp)), np.asarray(rot).T @ Re)
                po, Ro = grip[e]
                pose[e] = (np.asarray(tcp) + np.asarray(rot) @ po, np.asarray(rot) @ Ro)
                out = (f"{e}_element_carried", pose[e])
            else:
                grip[e] = None
        return out

    near_lim, track, hits = {}, {}, {}
    t_left_tab = 1e9  # (the left hand no longer holds the element)
    t_left_go = first_time("left", "let go of the tab") or 0.0
    HOLD_LINKS = ("tip_left", "tip_right")  # only the pads touch a held element on purpose
    arm_hits = {}
    for t in np.arange(0.0, dur + 1e-6, 0.05):
        held = None  # (name, verts, planes) of an element in the right hand at this sample
        tcp_r, rot_r, _, _ = aps["right"].target(float(t))
        act_r, _ = aps["right"].action7(float(t))
        cr = carried(float(t), tcp_r, rot_r, float(act_r[6]) < 0.2)
        if cr is not None:
            held = (cr[0], *_box_hull(np.zeros(3), fh, cr[1][1], cr[1][0]))
        arm_v = {}
        for arm, ap in aps.items():
            tcp, rot, _, label = ap.target(float(t))
            act, _ = ap.action7(float(t))
            q = np.asarray(act[:6], float)
            for k in range(6):
                lo, hi = lim[f"joint{k + 1}"]
                m = min(q[k] - lo, hi - q[k])
                if m < 0.1:
                    key = (arm, k + 1, label)
                    near_lim[key] = min(near_lim.get(key, 9), m)
            p_act, r_act = ik.tcp_pose(q, bases[arm])
            pe = float(np.linalg.norm(p_act - tcp)) * 1000
            re = math.degrees(math.acos(float(np.clip((np.trace(r_act.T @ rot) - 1) / 2, -1, 1))))
            if pe > 2 or re > 2:
                key = (arm, label)
                track[key] = max(track.get(key, (0, 0)), (round(pe, 1), round(re, 1)))
            joints = {f"joint{k + 1}": float(q[k]) for k in range(6)}
            f = float(act[6]) * float(task.FINGER_OPEN)  # the commanded opening (as the run's gripper)
            joints.update(joint7=f, joint8=f)
            fk = forward_kinematics(urdf, joints)
            objs = (list(static) + (box_hang if t >= t_drop else []) + (box_rest if t < t_grab else [])
                    + ([new0] if t < t_new else []) + ([old0] if t < t_old else []))
            if arm == "right" and held is not None:
                if True:
                    cname, cv, cp_ = held
                    other = [old0] if t >= t_old and cname.startswith("new") else []
                    for oname, ov, op in list(static) + (box_hang if t >= t_drop else []) + other:
                        if np.any(ov.max(0) < cv.min(0)) or np.any(ov.min(0) > cv.max(0)):
                            continue
                        if _overlap(cv, cp_, ov, op):
                            key = (cname, oname, label)
                            carried_hits.setdefault(key, round(float(t), 2))
            arm_objs = list(objs)
            if held is not None:
                holding = arm == "right" or (t_left_tab <= t < t_left_go)
                arm_objs.append(held + ((HOLD_LINKS if holding else ()),))
            for name, v in shapes:
                tf = fk[name]
                vv = v @ tf[:3, :3].T + tf[:3, 3] + bases[arm]
                hv = ConvexHull(vv)
                lo_, hi_ = vv.min(0), vv.max(0)
                arm_v[(arm, name)] = (vv, hv.equations, lo_, hi_, label)
                for ob in arm_objs:
                    oname, ov, op = ob[0], ob[1], ob[2]
                    if len(ob) > 3 and name in ob[3]:
                        continue  # (the holding hand's own pads and gripper touch it on purpose)
                    if np.any(ov.max(0) < lo_) or np.any(ov.min(0) > hi_):
                        continue
                    if _overlap(vv, hv.equations, ov, op):
                        key = (arm, name, oname, label)
                        if key not in hits:
                            hits[key] = round(float(t), 2)
        rk = [k for k in arm_v if k[0] == "right"]
        lk = [k for k in arm_v if k[0] == "left"]
        for a_ in rk:
            va, pa, la, ha, laba = arm_v[a_]
            for b_ in lk:
                vb, pb, lb, hb, labb = arm_v[b_]
                if np.any(ha < lb) or np.any(la > hb):
                    continue
                if _overlap(va, pa, vb, pb):
                    key = (a_[1], b_[1], laba, labb)
                    arm_hits.setdefault(key, round(float(t), 2))
    print("\nARM-TO-ARM (right link, left link, right waypoint, left waypoint: first time s):")
    for k, v in sorted(arm_hits.items(), key=lambda x: x[1]):
        print(" ", v, k)
    print("\nJOINTS within 0.1 rad of a limit (arm, joint, waypoint: margin rad):")
    for k, m in sorted(near_lim.items(), key=lambda x: x[1]):
        print(" ", k, round(m, 3))
    print("\nTRACKING over 2 mm / 2 deg (arm, waypoint: worst mm, deg):")
    for k, v in track.items():
        print(" ", k, v)
    print("\nCOLLISIONS (arm, link, object, waypoint: first time s):")
    for k, v in sorted(hits.items(), key=lambda x: x[1]):
        print(" ", v, k)
    print("\nCARRIED ELEMENT CONTACTS (element, object, waypoint: first time s):")
    for k, v in sorted(carried_hits.items(), key=lambda x: x[1]):
        print(" ", v, k)
    print("\ndone")


if __name__ == "__main__":
    main()
