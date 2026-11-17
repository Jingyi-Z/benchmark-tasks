"""Fast exact 578-hull offline check of the visor demo: 20 Hz IK tracking on both arms and exact half-space LP
collisions of every installed robot collision hull with the fixed roof/headliner/weatherstrip mock-up, sprung clip,
L-rod, and carried visor blade, plus inter-arm collisions."""
import importlib.util as iu
import math
import pathlib
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

HERE = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE))
spec = iu.spec_from_file_location("task", HERE / "task.py")
task = iu.module_from_spec(spec)
spec.loader.exec_module(task)
from yam_arena import YAM_URDF, ik  # noqa: E402
from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, RIGHT_BASE_POS  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402


def read_stl(path):
    data = pathlib.Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) == 84 + 50 * n:
        dt = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(float).reshape(-1, 3)
    v = [[float(x) for x in ln.split()[1:]] for ln in data.decode("ascii").splitlines() if ln.split()[:1] == ["vertex"]]
    return np.asarray(v)


def rpy_rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


urdf = pathlib.Path(YAM_URDF)
shapes = []
for link in ET.parse(urdf).getroot().findall("link"):
    name = link.attrib["name"]
    if name in ("base", "world"):
        continue
    for c in link.findall("collision"):
        m = c.find("geometry/mesh")
        if m is None:
            continue
        o = c.find("origin")
        rpy = [float(x) for x in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
        xyz = [float(x) for x in ((o.get("xyz") if o is not None else None) or "0 0 0").split()]
        sc = [float(x) for x in (m.get("scale") or "1 1 1").split()]
        v = (read_stl((urdf.parent / m.get("filename")).resolve()) * sc) @ rpy_rot(rpy).T + xyz
        v = np.unique(np.round(v, 7), axis=0)
        ch = ConvexHull(v)
        shapes.append((name, v[ch.vertices], ch.equations))
print(f"{len(shapes)} hulls per arm, {2 * len(shapes)} in all", flush=True)


def transform_hull(v0, eq0, R, p):
    v = v0 @ R.T + p
    n = eq0[:, :3] @ R.T
    d = eq0[:, 3] - np.sum(n * p, axis=1)
    return v.min(0), v.max(0), np.column_stack([n, d])


def box_hull_eq(c, s, R=np.eye(3), p=np.zeros(3)):
    c_w = p + R @ np.asarray(c, float)
    h = np.asarray(s, float) / 2
    v = np.array([c_w + R @ (h * np.array([a, b, d])) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    # Exact 6 axis-aligned box half-spaces transformed by (R, c_w)
    axes = R.T  # rows are world directions of local X, Y, Z
    rows = []
    for k in range(3):
        nk = R[:, k]
        rows.append([nk[0], nk[1], nk[2], -float(nk @ c_w) - h[k]])
        rows.append([-nk[0], -nk[1], -nk[2], float(nk @ c_w) - h[k]])
    return v.min(0), v.max(0), np.asarray(rows, float)


def intersect(ha, hb):
    mina, maxa, eqa = ha
    minb, maxb, eqb = hb
    if np.any(mina > maxb + 1e-6) or np.any(maxa < minb - 1e-6):
        return False
    P = np.vstack([eqa, eqb])
    res = linprog(np.zeros(3), A_ub=P[:, :3], b_ub=-P[:, 3] - 1e-6, bounds=(None, None), method="highs")
    return bool(res.success)


origin = np.array([task.PX, task.PY, task.ROD_Z])
RY = task.RYAW
static = [(f"roof{i}", box_hull_eq(b[0], b[1], R=RY, p=origin)) for i, b in enumerate(task.FRAME.boxes)]
static += [("jaw", box_hull_eq(b[0], b[1], R=RY, p=origin)) for b in task.REAR_JAW.boxes]
static += [("mount", box_hull_eq(task.VISOR.boxes[0][0], task.VISOR.boxes[0][1], R=RY, p=origin))]
static += [("bench", box_hull_eq((0.3, 0.0, task.TOP - 0.05), (1.6, 1.6, 0.1)))]


def visor_hulls(q):
    rs = task._rot(task._AXV["Z"], q["swivel"])
    rr = task._rot(task._AXV["Y"], q["roll"])
    rod = box_hull_eq((0, task.ROD_L / 2, 0), (2 * task.ROD_R, task.ROD_L, 2 * task.ROD_R), R=RY @ rs, p=origin)
    out = [("rod", rod)]
    for bi, b in enumerate(task.SLIDE.boxes):
        c = np.asarray(b[0], float) + [0, q["slide"], 0]
        out.append(("board" if bi == 0 else f"board_trim{bi}", box_hull_eq(c, b[1], R=RY @ rs @ rr, p=origin)))
    return out


bases = [np.array(RIGHT_BASE_POS, float), np.array(LEFT_BASE_POS, float)]
plan = task.demo()
aps = [ArmPlan(plan["right"], RIGHT_BASE_POS), ArmPlan(plan["left"], LEFT_BASE_POS)]
dur = max(sum(w.duration for w in v) for v in plan.values())
q = {"swivel": 0.0, "roll": math.radians(-90.0), "slide": 0.0}
PL = sorted(task.PLANNED)


def planned(t):
    sw, sl, who = 0.0, 0.0, None
    prev_sw, prev_sl = 0.0, 0.0
    for te, s_, l_, arm in PL:
        if t >= te:
            sw, sl = max(sw, s_), max(sl, l_)
            prev_sw, prev_sl = sw, sl
            continue
        if te - t < 1.05:
            f = 1.0 - (te - t) / 1.0
            sw = prev_sw + (s_ - prev_sw) * max(0.0, f) if s_ > prev_sw else sw
            sl = prev_sl + (l_ - prev_sl) * max(0.0, f) if l_ > prev_sl else sl
            who = arm
        break
    return sw, sl, who


worst = [0.0, 0.0]
hits = set()
tip_touch = set()
log = []
for t in np.arange(0.0, dur + 1e-9, 0.05):
    bodies = []
    for i in range(2):
        tcp, rot, f, _ = aps[i].target(float(t))
        act7, _ = aps[i].action7(float(t))
        qa = np.asarray(act7[:6], float)
        tcp_a, rot_a = ik.tcp_pose(qa, bases[i])
        pe = float(np.linalg.norm(tcp_a - tcp)) * 1000
        re = math.degrees(math.acos(float(np.clip((np.trace(rot_a.T @ rot) - 1) / 2, -1, 1))))
        worst[0], worst[1] = max(worst[0], pe), max(worst[1], re)
        if pe > 5 or re > 4:
            log.append(f"t={t:.2f} arm{i} {pe:.0f}mm/{re:.0f}deg")
        finger = float(act7[6]) * float(FINGER_OPEN)
        joints = {f"joint{k + 1}": float(qa[k]) for k in range(6)}
        joints.update(joint7=finger, joint8=finger)
        fk = forward_kinematics(urdf, joints)
        bodies.append([(name, transform_hull(v0, eq0, fk[name][:3, :3], fk[name][:3, 3] + bases[i])) for name, v0, eq0 in shapes])
    sw_p, sl_p, who = planned(float(t))
    q = {"swivel": sw_p, "roll": q["roll"], "slide": sl_p}
    vis = visor_hulls(q)
    for i in range(2):
        for name, h_arm in bodies[i]:
            for sn, h_sc in static + vis:
                if sn.startswith("board") and name in ("tip_left", "tip_right"):
                    if intersect(h_arm, h_sc):
                        tip_touch.add((round(float(t), 1), "right" if i == 0 else "left", name))
                    continue
                if intersect(h_arm, h_sc):
                    hits.add((round(float(t), 1), "right" if i == 0 else "left", name, sn))
    for na, ha in bodies[0]:
        for nb, hb in bodies[1]:
            if intersect(ha, hb):
                hits.add((round(float(t), 1), "arm-arm", na, nb))
    k = int(round(t * 20))
    if k % 40 == 0:
        print(f"t={t:5.1f} swivel={math.degrees(q['swivel']):5.1f} slide={q['slide'] * 1000:5.1f}mm "
              f"cover={task.coverage(q):.2f}", flush=True)
print("worst IK tracking:", f"{worst[0]:.1f} mm / {worst[1]:.1f} deg")
print("tracking problems:", log, f"({len(log)} samples)")
hl = sorted(hits)
print("collisions:", hl, f"({len(hl)} total)")
print("fingertip contacts with the board:", sorted(tip_touch)[:20], f"({len(tip_touch)} total)")
print("final coverage:", round(task.coverage(q), 3), {k: round(v, 3) for k, v in q.items()})
