"""Door test (not a submission)."""
import math, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit
from labkit import Joint, on_reset, on_step, part
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP
SCENE = "station"
TOP = TABLE_TOP[SCENE]
DARK = (0.2, 0.2, 0.2); ALU = (0.8, 0.8, 0.8)
DOOR_M = 0.25
DOORS = []
for i, (fr, cw) in enumerate(((0.0, True), (0.0, False), (0.0, None))):
    DOORS.append(part(f"door{i}", pos=(0.40, -0.25 + 0.25 * i, TOP), fixed=True, friction=1.0,
        boxes=[((0.0, 0.0, 0.003), (0.004, 0.004, 0.006), DARK)],
        joints=[Joint(f"door{i}", kind="slide", axis="Z", anchor=(0.0, 0.0, 0.095), lower=0.0, upper=0.22,
                      initial=0.0, damping=6.0, mass=DOOR_M,
                      boxes=[((0.0, 0.0, 0.2), (0.006, 0.2, 0.21), ALU)])]))
OBJECTS = DOORS
_n = {"n": 0}

@on_reset
def _r(env):
    import torch
    for i in (0, 1):
        d = env.unwrapped.scene[f"door{i}"]
        ids = d.find_bodies(f"door{i}")[0]
        f = torch.zeros((1, len(ids), 3), device=d.device)
        f[..., 2] = DOOR_M * 9.81 * (1.0 if i == 0 else 3.0)
        d.set_external_force_and_torque(f, torch.zeros_like(f), body_ids=ids, is_global=True)
        open("/home/ubuntu/dbg_door.txt", "a").write(f"reset door{i} ids={ids} bodies={d.body_names}\n")

@on_step
def _s(env):
    _n["n"] += 1
    if _n["n"] % 10:
        return
    vals = [float(labkit._part_state(env, labkit._SPECS[f"door{i}"]).joints[f"door{i}"]) for i in range(3)]
    open("/home/ubuntu/dbg_door.txt", "a").write(f"{_n['n']} {np.round(np.array(vals)*1000,1).tolist()}\n")

def score(objects): return 0.0
def success(objects): return False
def distance(objects): return 0.0
def demo():
    return {"left": [Waypoint(10.0, (0.15, 0.25, TOP + 0.2), FINGER_OPEN, "rest")],
            "right": [Waypoint(10.0, (0.15, -0.25, TOP + 0.2), FINGER_OPEN, "rest")]}
