"""Offline reach check: solve IK for every unique demo waypoint of a task.py (no simulator needed).
Usage: python tools/ik_check.py tasks/mos2-cvd/task.py   (needs yam_arena importable)
Prints each waypoint whose best IK error (mm + deg) is over 1, then "done N"."""
import sys, math, importlib.util
import numpy as np
from yam_arena import ik
from yam_arena.constants import RIGHT_BASE_POS, LEFT_BASE_POS, READY

spec = importlib.util.spec_from_file_location("task", sys.argv[1])
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)


def err(pos, R, B):
    best = 99.0
    seeds = [np.array(READY, float)] + [np.array(READY, float) + np.random.default_rng(i).normal(0, 0.8, 6) for i in range(12)]
    for s in seeds:
        _, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(R), s, base_pos=np.array(B), iters=300)
        best = min(best, pe * 1000 + math.degrees(re))
        if best < 1:
            break
    return round(best, 2)


d = T.demo(); seen = set()
for side, B in (("right", RIGHT_BASE_POS), ("left", LEFT_BASE_POS)):
    for w in d[side]:
        if not hasattr(w.yaw, "rot"):
            continue
        p = np.asarray(w.pos, float); key = (side, tuple(np.round(p, 3)), w.label)
        if key in seen:
            continue
        seen.add(key)
        e = err(p, w.yaw.rot, B)
        if e > 1:
            print(side, w.label, np.round(p, 3), e)
print("done", len(seen))
