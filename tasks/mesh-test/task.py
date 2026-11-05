"""Mesh render test (not a task)."""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import part, stl_mesh  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

TOP = TABLE_TOP["station"]
HERE = pathlib.Path(__file__).resolve().parent
R = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], float)  # assembly x -> world y, y -> up, z (fingers) -> towards robots
COL = {"structure": (0.93, 0.92, 0.88), "mount": (0.08, 0.08, 0.09), "jaw_left": (0.08, 0.08, 0.09),
       "jaw_right": (0.08, 0.08, 0.09), "rail": (0.70, 0.71, 0.73), "motor": (0.25, 0.25, 0.27),
       "board": (0.10, 0.25, 0.60)}
GRIPPER = part("umi", pos=(0.36, 0.0, TOP + 0.0785), fixed=True,
               meshes=[stl_mesh(HERE / "umi" / f"{k}.stl", c, rpy=R) for k, c in COL.items()])
FINGER = part("finger", pos=(0.20, 0.20, TOP + 0.05), mass=0.03,
              meshes=[stl_mesh(HERE / "umi" / "jaw_left.stl", (0.08, 0.08, 0.09), rpy=R, collide="hull",
                               origin=(-38.0, 0.0, 104.0))])
OBJECTS = [GRIPPER, FINGER]


def score(objects):
    return 0.0


def success(objects):
    return False


def demo():
    return {"left": [Waypoint(3.0, (0.15, 0.25, TOP + 0.14), FINGER_OPEN, "park")],
            "right": [Waypoint(3.0, (0.15, -0.25, TOP + 0.14), FINGER_OPEN, "park")]}
