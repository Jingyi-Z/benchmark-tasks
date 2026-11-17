"""Offline collision check of a recorded run: both arms' actual joint angles (logged every control step by task.py
when GLOVE_DEBUG=1 and GLOVE_JOINTS=<file>), checked arm against arm and against the fixed dash and stand.

    python check_replay.py <joints file>

Each arm is the YAM URDF's convex collision hulls placed by forward kinematics from the logged angles (the gripper's
two finger joints included). It reports every pair of hulls from the two arms that overlap, and every arm hull that
overlaps a fixed dash or stand box, with the first time each pair was seen. The elements and the glove box move during
the run and are not checked here (check_path.py covers them along the planned path).
"""
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_path as cp  # noqa: E402
import task  # noqa: E402
from yam_arena import YAM_URDF  # noqa: E402
from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402


def main(path):
    urdf = pathlib.Path(YAM_URDF)
    root = ET.parse(urdf).getroot()
    shapes = []
    for link in root.findall("link"):
        for c in link.findall("collision"):
            mesh = c.find("geometry/mesh")
            if mesh is None:
                continue
            o = c.find("origin")
            R = cp._rpy([float(x) for x in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.eye(3)
            p = np.array([float(x) for x in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            sc = np.array([float(x) for x in (mesh.get("scale") or "1 1 1").split()])
            v = (cp._stl((urdf.parent / mesh.get("filename")).resolve()) * sc) @ R.T + p
            v = np.unique(np.round(v, 6), axis=0)
            h = ConvexHull(v)
            shapes.append((link.get("name"), v[h.vertices]))
    T = task
    dash_p = np.array([T.DASH_X, 0.0, T.TOP])
    static = [(f"dash{i}", *cp._box_hull(np.zeros(3), np.array(b[1], float) / 2, np.eye(3), dash_p + np.array(b[0], float)))
              for i, b in enumerate(T.DASH_BOXES)]
    static += [("stand", *cp._box_hull(np.array(b[0], float), np.array(b[1], float) / 2, np.eye(3),
                                       np.array([T.STAND_XY[0], T.STAND_XY[1], T.TOP]))) for b in T.STAND.boxes[:1]]
    bases = {"left": np.array(LEFT_BASE_POS, float), "right": np.array(RIGHT_BASE_POS, float)}
    names, rows = None, []
    for line in open(path):
        if line.startswith("#"):
            names = line[1:].split()
            continue
        v = [float(x) for x in line.split()]
        if len(v) == 17:
            rows.append(v)
    names = names or [f"joint{k + 1}" for k in range(8)]
    print(f"{len(rows)} logged steps, {rows[0][0] if rows else 0:.2f} to {rows[-1][0] if rows else 0:.2f} s; joints {names}")
    arm_hits, static_hits = {}, {}
    gap = (9.0, None, None)  # the closest the two arms come: (m, time, left link)
    for r in rows[::2]:  # every second step (10 Hz)
        t = r[0]
        hulls = {}
        for arm, q in (("left", r[1:9]), ("right", r[9:17])):
            fk = forward_kinematics(urdf, dict(zip(names, q)))
            for name, v in shapes:
                tf = fk[name]
                vv = v @ tf[:3, :3].T + tf[:3, 3] + bases[arm]
                hv = ConvexHull(vv)
                hulls[(arm, name)] = (vv, hv.equations, vv.min(0), vv.max(0))
                for oname, ov, op in static:
                    if np.any(ov.max(0) < vv.min(0)) or np.any(ov.min(0) > vv.max(0)):
                        continue
                    if cp._overlap(vv, hv.equations, ov, op):
                        static_hits.setdefault((arm, name, oname), t)
        right_v = np.vstack([h[0] for (a_, n_), h in hulls.items() if a_ == "right"])
        for (a_, n_), h in hulls.items():
            if a_ == "left":
                d = float(np.min(np.linalg.norm(right_v[:, None, :] - h[0][None, ::2, :], axis=2)))
                if d < gap[0]:
                    gap = (d, t, n_)
        for (a1, n1), (v1, p1, lo1, hi1) in hulls.items():
            if a1 != "right":
                continue
            for (a2, n2), (v2, p2, lo2, hi2) in hulls.items():
                if a2 != "left" or np.any(hi1 < lo2) or np.any(lo1 > hi2):
                    continue
                if cp._overlap(v1, p1, v2, p2):
                    arm_hits.setdefault((n1, n2), t)
    print(f"\nMinimum sampled hull-vertex distance between the arms (not surface clearance): {gap[0] * 1000:.1f} mm at {gap[1]} s (left {gap[2]})")
    print("\nARM-TO-ARM in the recorded run (right link, left link: first time s):")
    for k, v in sorted(arm_hits.items(), key=lambda x: x[1]):
        print(" ", v, k)
    print("\nARM-TO-DASH/STAND in the recorded run (arm, link, object: first time s):")
    for k, v in sorted(static_hits.items(), key=lambda x: x[1]):
        print(" ", v, k)
    print("\ndone")


if __name__ == "__main__":
    main(sys.argv[1])
