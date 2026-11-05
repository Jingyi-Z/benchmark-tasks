"""On the station table: a benchtop spin coater for small wafers, a 2-inch silicon wafer on a pad stand to its right,
a pair of angled wafer tweezers resting on a stand, and a PTFE wafer aligner lying on the bench in front.

The coater: a white housing 22 x 22 cm with a round catch bowl and, in the middle of the bowl, a black vacuum chuck
2.4 cm across. Its clear lid is hinged along the back edge and lifts at the front by a black handle bar; it falls shut
under its own weight unless held. A 12 mm hole in the middle of the lid, over the chuck, is where resist is dispensed
with the lid shut.

The aligner (a "fork"): a flat PTFE plate with a ring of short posts making a wafer pocket, and a slot from the pocket
out to the plate's far end that fits round the chuck. The slot ends, near the pocket's centre, in a V. Lowered over the
chuck with the wafer in its pocket and pushed onto it, the V seats against the chuck with the pocket centred on the
chuck's axis; lowered further, the plate sinks below the chuck's top, leaving the wafer centred on the chuck, and it
then slides out from under the wafer and lifts away. A notch in the pocket's near rim lets the tweezers' lower blade
under the wafer's edge.

The task: hold the lid open; with the tweezers, move the wafer from its stand into the aligner's pocket; with the
aligner, set the wafer on the chuck, centred within 0.5 mm; take the aligner out; let the lid close.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

HOUSING_C = (0.92, 0.92, 0.90)
BOWL_C = (0.62, 0.63, 0.66)
CHUCK_C = (0.08, 0.08, 0.09)
LID_C = (0.82, 0.90, 0.96)
labkit.GLASS_COLORS.update({LID_C})
BLACK = (0.06, 0.06, 0.07)
DARK = (0.16, 0.17, 0.19)
PTFE = (0.95, 0.95, 0.93)
SI = (0.32, 0.34, 0.42)
STEEL = (0.72, 0.73, 0.75)
STAND_C = (0.55, 0.56, 0.58)

# -- the spin coater (fixed). Frame: origin on the table under the chuck's axis, axes along the world's. --------------
CX, CY = 0.27, 0.03
HW = 0.110  # housing half width
HOUSING_H = 0.040  # its top is the bowl's floor
BOWL_IN, RIB_R = 0.095, 0.0035
N_RIBS = 88
RIM = 0.072
CHUCK_R, CHUCK_TOP = 0.012, 0.052
LID_T = 0.005
HINGE = (HW, 0.0, RIM + 0.0035)  # the lid's hinge, along y at the housing's back top edge
HOLE = 0.012  # the dispense hole, square, over the chuck
HANDLE_Y, HANDLE_L, HANDLE_S, HANDLE_UP = 0.040, 0.060, 0.010, 0.012
HANDLE_X = -HW + 0.010
LID_MAX = 1.30  # the lid's stop (rad, 74 degrees): short of upright, so it always falls shut
_lz = RIM + 0.001 + LID_T / 2
_half = (2 * HW - HOLE) / 4  # width of each strip round the hole
LID_BOXES = [
    ((-(HOLE / 2 + _half), 0.0, _lz), (2 * _half, 2 * HW, LID_T), LID_C),  # in front of the hole
    ((HOLE / 2 + _half, 0.0, _lz), (2 * _half, 2 * HW, LID_T), LID_C),  # behind it
    ((0.0, HOLE / 2 + _half, _lz), (HOLE, 2 * _half, LID_T), LID_C),
    ((0.0, -(HOLE / 2 + _half), _lz), (HOLE, 2 * _half, LID_T), LID_C),
    # a dark collar round the hole
    *[((sx * (HOLE / 2 + 0.002), 0.0, _lz + LID_T / 2 + 0.001), (0.004, HOLE + 0.008, 0.002), DARK) for sx in (-1, 1)],
    *[((0.0, sy * (HOLE / 2 + 0.002), _lz + LID_T / 2 + 0.001), (HOLE, 0.004, 0.002), DARK) for sy in (-1, 1)],
    # the handle: a bar along y on two posts at the front
    ((HANDLE_X, HANDLE_Y, _lz + LID_T / 2 + HANDLE_UP + HANDLE_S / 2), (HANDLE_S, HANDLE_L, HANDLE_S), BLACK),
    *[((HANDLE_X, HANDLE_Y + e * (HANDLE_L / 2 - 0.005), _lz + LID_T / 2 + HANDLE_UP / 2), (HANDLE_S, 0.010, HANDLE_UP),
       BLACK) for e in (-1, 1)],
    # hinge knuckles
    *[((HW - 0.004, e * 0.07, _lz), (0.008, 0.03, 0.008), STEEL) for e in (-1, 1)],
]
LID = Joint("lid", kind="hinge", axis="Y", anchor=HINGE, lower=0.0, upper=LID_MAX, initial=0.0, target=0.0,
            stiffness=0.0, damping=0.08, mass=0.25, boxes=LID_BOXES)


def _ring(r, n, rad, z0, h, colour):
    return [((r * math.cos(2 * math.pi * (i + 0.5) / n), r * math.sin(2 * math.pi * (i + 0.5) / n), z0 + h / 2), rad,
             h, colour) for i in range(n)]


COATER = part(
    "coater", pos=(CX, CY, TOP), fixed=True, friction=0.9,
    boxes=[
        ((0.0, 0.0, HOUSING_H / 2), (2 * HW, 2 * HW, HOUSING_H), HOUSING_C),
        ((-HW + 0.012, -HW + 0.030, HOUSING_H + 0.0005), (0.016, 0.040, 0.001), DARK),  # control panel
        *[((HW - 0.004, e * 0.07, (HOUSING_H + RIM) / 2), (0.008, 0.012, RIM - HOUSING_H), HOUSING_C)
          for e in (-1, 1)],  # hinge brackets
    ],
    cylinders=[
        ((0.0, 0.0, HOUSING_H + 0.0005), BOWL_IN + 2 * RIB_R, 0.001, BOWL_C),
        ((0.0, 0.0, (HOUSING_H + CHUCK_TOP) / 2), CHUCK_R, CHUCK_TOP - HOUSING_H, CHUCK_C),
        *_ring(BOWL_IN + RIB_R, N_RIBS, RIB_R, HOUSING_H, RIM - HOUSING_H, BOWL_C),
    ],
    joints=[LID],
)

# -- the wafer stand (fixed) and the wafer (loose) on its pad -------------------------------------------------------
STAND_XY = (0.29, -0.16)
PAD_R, PAD_TOP = 0.010, 0.025  # narrower than the wafer, so the tweezers' lower blade fits under its edge
WAFER_R, WAFER_T = 0.0254, 0.0010  # 2-inch wafer, drawn 1 mm thick (real: 0.28 mm) for stable contact
STAND = part("stand", pos=(*STAND_XY, TOP), fixed=True, friction=0.6,
             boxes=[((0.0, 0.0, 0.002), (0.07, 0.07, 0.004), STAND_C)],
             cylinders=[((0.0, 0.0, (0.004 + PAD_TOP) / 2), PAD_R, PAD_TOP - 0.004, STAND_C)])
WAFER = part("wafer", pos=(*STAND_XY, TOP + PAD_TOP), mass=0.004, friction=0.5,
             cylinders=[((0.0, 0.0, WAFER_T / 2), WAFER_R, WAFER_T, SI)])

# -- the aligner (loose, PTFE). Frame: origin at the pocket's centre, on the plate's underside; the slot opens to +x. ---
PT = 0.003  # plate thickness
SLOT_HW = 0.0135
HEAD_X = (-0.036, 0.026)
HEAD_HW = 0.034
POCKET_R = WAFER_R + 0.0002 + 0.0015  # post centres
POST_R, POST_H = 0.0015, 0.0015
NOTCH = ((-0.036, -0.0165), 0.007)  # the tweezer notch in the near rim, up to the V: x span, half width
RISER = ((-0.040, -0.036), 0.010, 0.032)  # x span, half width, height
HANDLE_A = ((-0.115, -0.040), 0.006, (0.026, 0.032))  # the aligner's handle: x span, half width, z span
_t = 0.003  # the V's walls
_u = (CHUCK_R + _t / 2) / math.sqrt(2)
ALIGNER_BOXES = [
    # rails either side of the slot, from behind the V to the far end
    *[((sum((-0.019, HEAD_X[1])) / 2, s * (SLOT_HW + HEAD_HW) / 2, PT / 2),
       (HEAD_X[1] + 0.019, HEAD_HW - SLOT_HW, PT), PTFE) for s in (1, -1)],
    # the near rim, either side of the tweezer notch
    *[(((NOTCH[0][0] + NOTCH[0][1]) / 2, s * (NOTCH[1] + HEAD_HW) / 2, PT / 2),
       (NOTCH[0][1] - NOTCH[0][0], HEAD_HW - NOTCH[1], PT), PTFE) for s in (1, -1)],
    # the V at the slot's end, its walls tangent to the chuck when the pocket is centred on it
    ((-_u, _u, PT / 2), (0.020, _t, PT), PTFE, math.radians(45.0)),
    ((-_u, -_u, PT / 2), (0.020, _t, PT), PTFE, math.radians(-45.0)),
    # riser and handle
    ((sum(RISER[0]) / 2, 0.0, RISER[2] / 2), (RISER[0][1] - RISER[0][0], 2 * RISER[1], RISER[2]), PTFE),
    ((sum(HANDLE_A[0]) / 2, 0.0, sum(HANDLE_A[2]) / 2),
     (HANDLE_A[0][1] - HANDLE_A[0][0], 2 * HANDLE_A[1], HANDLE_A[2][1] - HANDLE_A[2][0]), PTFE),
]
_post_angles = [a for a in range(30, 331, 15) if not 165 <= a <= 195]
ALIGNER_CYLS = [((POCKET_R * math.cos(math.radians(a)), POCKET_R * math.sin(math.radians(a)), PT + POST_H / 2),
                 POST_R, POST_H, PTFE) for a in _post_angles]
ALIGNER_XY = (0.118, -0.06)  # where it lies at the start: pocket centre, slot towards the coater (+x)
ALIGNER = part("aligner", pos=(*ALIGNER_XY, TOP), mass=0.030, friction=0.5, boxes=ALIGNER_BOXES,
               cylinders=ALIGNER_CYLS)

# -- the tweezers (loose): angled wafer tweezers. Frame: origin between the two flat blades' tips; the blades lie flat
# along +x; the arms rise behind them at 45 degrees. The lower blade and arm are the body; the upper blade and arm
# slide together along z (sprung open).
BLADE_L, BLADE_W, BLADE_T = 0.012, 0.010, 0.0012
OPEN_GAP = 0.005  # between the blades, open
ARM_L, ARM_W, ARM_T = 0.085, 0.008, 0.0025
_c, _s = math.cos(math.radians(45.0)), math.sin(math.radians(45.0))


def _arm(z_blade):
    """An arm rising at 45 degrees from the blade's back end (x = -BLADE_L), its blade at height z_blade."""
    mx = -BLADE_L - ARM_L / 2 * _c
    mz = z_blade + ARM_L / 2 * _s
    return ((mx, 0.0, mz), (ARM_L, ARM_W, ARM_T), STEEL, (0.0, math.radians(-45.0)))


_zl = -OPEN_GAP / 2 - BLADE_T / 2  # lower blade's centre
_zu = OPEN_GAP / 2 + BLADE_T / 2
_back = (-BLADE_L - ARM_L * _c, 0.0, (_zl + _zu) / 2 + ARM_L * _s)
TWEEZER_JAW = Joint(
    "jaw", kind="slide", axis="Z", anchor=(-BLADE_L / 2, 0.0, _zu), lower=-(OPEN_GAP - 0.0002), upper=0.0, initial=0.0,
    target=0.0, stiffness=300.0, damping=2.0, mass=0.008,
    boxes=[((-BLADE_L / 2, 0.0, _zu), (BLADE_L, BLADE_W, BLADE_T), STEEL), _arm(_zu)],
)
# The tweezers' stand: a base with two posts holding them at their 45 degree pose, the blades in the air: a low post
# under the lower blade and a tall one under the joined end, each with side lips.
TW_XY = (0.20, -0.215)
TW_Z = 0.040  # the tweezers' origin above the table
_blade_bottom = TW_Z + _zl - BLADE_T / 2
_end_bottom = TW_Z + _back[2] - (OPEN_GAP + 2 * ARM_T) / 2
_px1, _px2 = -BLADE_L / 2, _back[0] - 0.004


def _post(px, top):
    return [((px, 0.0, (0.003 + top) / 2), (0.008, 0.008, top - 0.003), STAND_C),
            *[((px, e * (BLADE_W / 2 + 0.002), top + 0.002), (0.008, 0.002, 0.004), STAND_C) for e in (-1, 1)]]


TSTAND = part("tweezer_stand", pos=(*TW_XY, TOP), fixed=True, friction=0.8,
              boxes=[((-0.035, 0.0, 0.0015), (0.090, 0.040, 0.003), STAND_C), *_post(_px1, _blade_bottom),
                     *_post(_px2, _end_bottom)])
TWEEZERS = part(
    "tweezers", pos=(*TW_XY, TOP + TW_Z), mass=0.020, friction=0.9,
    boxes=[((-BLADE_L / 2, 0.0, _zl), (BLADE_L, BLADE_W, BLADE_T), STEEL), _arm(_zl),
           ((_back[0] - 0.004, 0.0, _back[2]), (0.010, ARM_W, OPEN_GAP + 2 * ARM_T), STEEL)],  # the joined end
    joints=[TWEEZER_JAW],
)

OBJECTS = [COATER, STAND, WAFER, ALIGNER, TWEEZERS, TSTAND]


def C(x, y, z):
    """Coater coordinates to world."""
    return (CX + x, CY + y, TOP + z)


# -- scoring ----------------------------------------------------------------------------------------------------------
CHUCK_XY = np.array([CX, CY])
CHUCK_Z = TOP + CHUCK_TOP
FLAT_DEG = 2.0
SEAT_TOL = 0.001
CENTRE_TOL, NEAR_TOL = 0.0005, 0.0015
LID_SHUT = math.radians(3.0)


def wafer_offset(w) -> float:
    return float(np.hypot(*(w.pos[:2] - CHUCK_XY)))


def on_chuck(w) -> bool:
    return w.tilt_deg() <= FLAT_DEG and abs(w.pos[2] - CHUCK_Z) <= SEAT_TOL and wafer_offset(w) <= CHUCK_R


def score(objects):
    """0 unless the wafer rests flat on the chuck (tilt up to 2 degrees, underside within 1 mm of the chuck's top,
    centre over the chuck). 0.25 on the chuck; 0.5 centred within 1.5 mm; 1.0 within 0.5 mm. Halved if the lid is not
    shut (within 3 degrees)."""
    w = objects["wafer"]
    if not on_chuck(w):
        return 0.0
    e = wafer_offset(w)
    s = 1.0 if e <= CENTRE_TOL else 0.5 if e <= NEAR_TOL else 0.25
    if objects["coater"].joints["lid"] > LID_SHUT:
        s *= 0.5
    return s


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    w = objects["wafer"]
    return wafer_offset(w) + abs(float(w.pos[2]) - CHUCK_Z)


def demo():
    return {"left": [Waypoint(4.0, (0.15, 0.25, TOP + 0.14), FINGER_OPEN, "park")],
            "right": [Waypoint(4.0, (0.15, -0.25, TOP + 0.14), FINGER_OPEN, "park")]}
