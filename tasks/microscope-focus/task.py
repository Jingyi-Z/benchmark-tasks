"""On the station table: an upright light microscope (modelled on a research stand with focus knobs on both sides) with a
slide on its stage under the 10x objective. The stage is 12 mm too low, so the specimen is out of focus.

The task: turn a focus knob to raise the stage until the slide's top is at the objective's working distance, without
letting the slide come within 0.5 mm of the objective. One shaft carries both knobs, so either arm can focus.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

STAND = (0.90, 0.90, 0.88)
DARK = (0.16, 0.17, 0.19)
BLACK = (0.06, 0.06, 0.07)
STEEL = (0.70, 0.71, 0.73)
GLASS = (0.80, 0.90, 0.95)
SPECIMEN = (0.55, 0.20, 0.45)
WHITE = (0.97, 0.97, 0.97)
RED = (0.85, 0.12, 0.12)

# -- the microscope (fixed). Frame: origin on the table under the stand, x away from the robots, y to their left. -----
MX, MY = 0.34, 0.06
MYAW = 0.0  # facing the robots squarely
BASE = (0.26, 0.16, 0.03)
PILLAR = (0.07, 0.11, 0.22)
PILLAR_X = 0.085
HEAD = (0.18, 0.09, 0.05)  # the arm over the stage
HEAD_TOP = 0.25
STAGE_X = -0.03  # stage and objective centre (local x)
OBJ_R, OBJ_TIP = 0.012, 0.155  # the 10x objective: 24 mm across, front lens 15.5 cm above the table
WD = 0.0052  # working distance: in focus when the slide's top is 5.2 mm below the front lens
START_OFF = 0.012  # the stage starts 12 mm below focus
SLIDE = (0.026, 0.076, 0.001)  # x by y by thickness (long axis across the stage)
PLATE = (0.12, 0.15, 0.012)
PLATE_TOP0 = OBJ_TIP - WD - START_OFF - SLIDE[2]  # stage plate top at the start
STAGE_MAX = START_OFF + WD - 0.0002  # the stage stops just short of touching the objective
# Focus drive housing under the stage, with the knob shaft through it along y.
HOUSING = (0.08, 0.20, 0.085)  # x by y by height, from 3 cm above the table
KNOB_X, KNOB_Z = -0.04, 0.10  # knob axis (local)
KNOB_R, KNOB_W = 0.021, 0.020
KNOB_Y0 = HOUSING[1] / 2  # inner face of each knob; the knob spans KNOB_Y0 .. KNOB_Y0 + KNOB_W on each side
LEAD = 0.024  # m of stage travel per turn: 2 mm per 30 degrees
KNOB_MIN = -2 * math.pi * STAGE_MAX / LEAD  # knob angle at the stage's upper stop (turning so the top comes towards you)

# Ribs round each knob's rim (axis y), so the fingers bite.
RIBS = [((KNOB_X + KNOB_R * math.cos(i * math.pi / 6), sgn * (KNOB_Y0 + KNOB_W / 2), KNOB_Z + KNOB_R * math.sin(i * math.pi / 6)),
         0.0025, KNOB_W, BLACK, "Y") for sgn in (1, -1) for i in range(12)]
FOCUS = Joint(
    "focus", kind="hinge", axis="Y", anchor=(KNOB_X, 0.0, KNOB_Z), lower=KNOB_MIN - 0.3, upper=0.3, initial=0.0,
    stiffness=0.0, damping=0.3, target=0.0, mass=0.05,  # a firm knob: viscous drag, it stays where it is left
    cylinders=[
        ((KNOB_X, 0.0, KNOB_Z), 0.006, 2 * KNOB_Y0 + 2 * KNOB_W - 0.002, STEEL, "Y"),  # shaft
        ((KNOB_X, KNOB_Y0 + KNOB_W / 2, KNOB_Z), KNOB_R, KNOB_W, BLACK, "Y"),
        ((KNOB_X, -(KNOB_Y0 + KNOB_W / 2), KNOB_Z), KNOB_R, KNOB_W, BLACK, "Y"),
        *RIBS,
    ],
    boxes=[  # a white index line on each knob's outer face, from the centre outwards (pointing up at the start)
        ((KNOB_X, sgn * (KNOB_Y0 + KNOB_W + 0.0005), KNOB_Z + KNOB_R / 2), (0.003, 0.001, KNOB_R), WHITE)
        for sgn in (1, -1)
    ],
)
STAGE = Joint(
    "stage", kind="slide", axis="Z", anchor=(STAGE_X, 0.0, PLATE_TOP0), lower=0.0, upper=STAGE_MAX, initial=0.0,
    stiffness=20000.0, damping=400.0, target=0.0, mass=0.05,  # follows the knob through the drive (see _drive)
    boxes=[
        ((STAGE_X, 0.0, PLATE_TOP0 - PLATE[2] / 2), PLATE, DARK),
        ((STAGE_X + 0.075, 0.0, PLATE_TOP0 - 0.025), (0.03, 0.07, 0.03), DARK),  # stage carrier, towards the pillar
        ((STAGE_X, 0.0, PLATE_TOP0 + SLIDE[2] / 2), SLIDE, GLASS),  # the slide, held in the stage's slide holder
        ((STAGE_X, 0.0, PLATE_TOP0 + SLIDE[2] + 0.0002), (0.010, 0.010, 0.0004), SPECIMEN),  # the specimen
        ((STAGE_X, 0.030, PLATE_TOP0 + SLIDE[2] + 0.0002), (0.024, 0.016, 0.0004), WHITE),  # slide label
    ],
)
MICROSCOPE = part(
    "microscope", pos=(MX, MY, TOP), yaw=MYAW, fixed=True, friction=1.0,
    boxes=[
        ((0.0, 0.0, BASE[2] / 2), BASE, STAND),
        ((PILLAR_X, 0.0, BASE[2] + PILLAR[2] / 2), PILLAR, STAND),
        ((PILLAR_X + PILLAR[0] / 2 - HEAD[0] / 2, 0.0, HEAD_TOP - HEAD[2] / 2), HEAD, STAND),
        ((KNOB_X, 0.0, 0.03 + HOUSING[2] / 2), HOUSING, STAND),
    ],
    cylinders=[
        ((STAGE_X, 0.0, HEAD_TOP - HEAD[2] - 0.006), 0.032, 0.012, STEEL),  # nosepiece
        ((STAGE_X, 0.0, (OBJ_TIP + HEAD_TOP - HEAD[2] - 0.012) / 2), OBJ_R,
         HEAD_TOP - HEAD[2] - 0.012 - OBJ_TIP, STEEL),  # the 10x objective
        ((STAGE_X, 0.0, OBJ_TIP + 0.004), OBJ_R + 0.0005, 0.002, RED),  # its colour ring
    ],
    joints=[FOCUS, STAGE],
)

OBJECTS = [MICROSCOPE]


# -- the focus drive: the stage follows the knob ----------------------------------------------------------------------
def stage_for(knob_angle: float) -> float:
    """Stage height (m above its start) for a knob angle (rad): turning so the top comes towards you raises it."""
    return float(np.clip(-LEAD * knob_angle / (2 * math.pi), 0.0, STAGE_MAX))


def gap(stage_q: float) -> float:
    """Clearance (m) from the slide's top to the objective's front lens."""
    return OBJ_TIP - (PLATE_TOP0 + SLIDE[2] + stage_q)


_trial = {"min_gap": None}


@on_reset
def _clear(env):
    _trial["min_gap"] = None


@on_step
def _drive(env):
    import torch

    import labkit

    s = labkit._part_state(env, labkit._SPECS["microscope"])
    g = gap(s.joints["stage"])
    _trial["min_gap"] = g if _trial["min_gap"] is None else min(_trial["min_gap"], g)
    asset = env.unwrapped.scene["microscope"]
    idx = list(asset.joint_names).index(STAGE.joint_name)
    target = torch.tensor([[stage_for(s.joints["focus"])]], device=asset.device)
    asset.set_joint_position_target(target, joint_ids=[idx])


# -- scoring ----------------------------------------------------------------------------------------------------------
CRASH_GAP = 0.0005
FOCUS_TOL, NEAR_TOL, COARSE_TOL = 0.0005, 0.002, 0.006


def focus_error(objects) -> float:
    return abs(gap(objects["microscope"].joints["stage"]) - WD)


def score(objects):
    """0 if the slide came within 0.5 mm of the objective at any time. Otherwise, by the final distance of the slide's
    top from the working distance: 1.0 within 0.5 mm (in focus), 0.5 within 2 mm, 0.25 within 6 mm, else 0."""
    if _trial["min_gap"] is not None and _trial["min_gap"] < CRASH_GAP:
        return 0.0
    e = focus_error(objects)
    return 1.0 if e <= FOCUS_TOL else 0.5 if e <= NEAR_TOL else 0.25 if e <= COARSE_TOL else 0.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return focus_error(objects)


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm: comes in from the right side with the gripper level (pointing along the knob axis, fingers vertical), grips the
# right knob's rim and turns it half a turn, top towards the robots, in 30-degree steps; lets go, backs off and parks.
# The task file widens the scripted planner's grip orientations: a Waypoint whose ``yaw`` is an ``Orient`` carries a
# full rotation matrix.


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_orient", False):
    def _grip_rotation(yaw=0.0):
        return yaw.rot if isinstance(yaw, Orient) else _down_grip(yaw)

    _grip_rotation._orient = True
    _demo.grip_rotation = _grip_rotation

SIDE = axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.pi / 2) @ _down_grip(0.0)  # pointing +y, fingers vertical


def side(roll: float) -> Orient:
    """Level grip pointing +y, rolled ``roll`` rad about +y (negative: the top of a gripped knob comes towards you)."""
    return Orient(_RZ @ axis_angle_matrix(np.array([0.0, 1.0, 0.0]), roll) @ SIDE)


TIP_ABOVE_TCP = 0.013
_RZ = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), MYAW)


def world(local) -> tuple:
    """A point in the microscope's frame, in the world."""
    return tuple((np.array([MX, MY, TOP]) + _RZ @ np.asarray(local, float)).tolist())


_GRIP_LY = -(KNOB_Y0 + KNOB_W) + 0.009 - TIP_ABOVE_TCP  # pad tips 9 mm onto the right knob
GRIP = world((KNOB_X, _GRIP_LY, KNOB_Z))
OUT = world((KNOB_X, _GRIP_LY - 0.04, KNOB_Z))
OUT_UP = world((KNOB_X, _GRIP_LY - 0.04, KNOB_Z + 0.04))
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
TURN = -2 * math.pi * START_OFF / LEAD  # half a turn
STEPS = 6

def _aabb(name, centre, size):
    """World-aligned box round a box in the microscope's frame (plancheck takes axis-aligned boxes only)."""
    half = np.abs(_RZ) @ (np.asarray(size, float) / 2)
    return (name, world(centre), tuple((2 * half).tolist()))


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    _aabb("base", (0.0, 0.0, BASE[2] / 2), BASE),
    _aabb("housing", (KNOB_X, 0.0, 0.03 + HOUSING[2] / 2), HOUSING),
    _aabb("stage", (STAGE_X, 0.0, PLATE_TOP0 - PLATE[2] / 2), PLATE),
    _aabb("pillar", (PILLAR_X, 0.0, BASE[2] + PILLAR[2] / 2), PILLAR),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    right = [
        Waypoint(2.5, OUT_UP, FINGER_OPEN, "above, beside the right knob"),
        Waypoint(1.5, OUT, FINGER_OPEN, "level the gripper beside the knob", side(0.0)),
        Waypoint(1.5, GRIP, FINGER_OPEN, "onto the knob", side(0.0)),
        Waypoint(0.8, GRIP, FINGER_CLOSED, "grip the rim", side(0.0)),
    ]
    for k in range(1, STEPS + 1):
        right.append(Waypoint(1.2, GRIP, FINGER_CLOSED, "turn, top towards the robots", side(TURN * k / STEPS)))
    right += [
        Waypoint(0.6, GRIP, FINGER_CLOSED, "settle", side(TURN)),
        Waypoint(0.8, GRIP, FINGER_OPEN, "let go", side(TURN)),
        Waypoint(1.0, OUT, FINGER_OPEN, "back off", side(TURN)),
        Waypoint(1.2, OUT, FINGER_OPEN, "unroll", side(TURN / 2)),
        Waypoint(1.2, OUT, FINGER_OPEN, "unroll", side(0.0)),
        Waypoint(1.5, OUT_UP, FINGER_OPEN, "point down again"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = _dur(right) + 2.0
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "hold"))
    left = [Waypoint(t_end, LEFT_PARK, FINGER_OPEN, "rest")]
    return {"left": left, "right": right}
