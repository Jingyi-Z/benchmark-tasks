"""On the station table: a benchtop spin coater for small wafers, a 2-inch silicon wafer on a pad stand to its right,
a pair of angled wafer tweezers standing tips down in a cup, and a PTFE wafer aligner lying on the bench in front.

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
CX, CY = 0.28, -0.02
HW = 0.110  # housing half width
HOUSING_H = 0.040  # its top is the bowl's floor
BOWL_IN, RIB_R = 0.095, 0.0035
N_RIBS = 88
RIM = 0.072
CHUCK_R, CHUCK_TOP = 0.012, 0.052
LID_T = 0.005
HINGE = (HW, 0.0, RIM + 0.0035)  # the lid's hinge, along y at the housing's back top edge
HOLE = 0.012  # the dispense hole, square, over the chuck
HANDLE_Y, HANDLE_L, HANDLE_S, HANDLE_UP = 0.080, 0.050, 0.010, 0.020
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
    *[((HANDLE_X, HANDLE_Y + e * (HANDLE_L / 2 - 0.005), _lz + LID_T / 2 + HANDLE_UP / 2), (HANDLE_S, 0.010, HANDLE_UP),
       BLACK) for e in (-1, 1)],
    # hinge knuckles
    *[((HW - 0.004, e * 0.07, _lz), (0.008, 0.03, 0.008), STEEL) for e in (-1, 1)],
]
BAR_Z = _lz + LID_T / 2 + HANDLE_UP + HANDLE_S / 2  # the handle bar's axis, lid shut
LID = Joint("lid", kind="hinge", axis="Y", anchor=HINGE, lower=0.0, upper=LID_MAX, initial=0.0, target=0.0,
            stiffness=0.0, damping=0.08, mass=0.25, boxes=LID_BOXES,
            cylinders=[((HANDLE_X, HANDLE_Y, BAR_Z), HANDLE_S / 2, HANDLE_L, BLACK, "Y")])


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
STAND_XY = (0.40, -0.20)
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
ALIGNER_XY = (0.125, -0.10)  # where it lies at the start: pocket centre, slot towards the coater (+x)
ALIGNER = part("aligner", pos=(*ALIGNER_XY, TOP), mass=0.030, friction=0.5, boxes=ALIGNER_BOXES,
               cylinders=ALIGNER_CYLS)

# -- the tweezers (loose): angled wafer tweezers, standing upright in a cup, tips down. Frame: origin between the two
# blades' tips, with the arms vertical (joined end up) and the blades bent 45 degrees forward (+x) and down from the
# arms' lower ends. The arms are flat strips, their faces normal to x; the lower blade and arm (x < 0) and the joined
# end are the body, the upper blade and arm (x > 0) slide together along x (sprung open). Held in a hand turned 45
# degrees (arms rising back at 45 degrees), the blades lie flat, pointing forward.
BLADE_L, BLADE_W, BLADE_T = 0.012, 0.010, 0.0012
OPEN_GAP = 0.005  # between the blades' faces, open
ARM_L, ARM_W, ARM_T = 0.085, 0.008, 0.0025
_c = _s = math.sqrt(0.5)
SEP = (OPEN_GAP + BLADE_T) / _s  # the two halves' offset along x (normal to the arms), open
_ax = -_c * BLADE_L  # the arm stack's centre line, x
_az = _s * BLADE_L  # the arms' lower ends, z


def _half(sx):
    """One half's arm and blade, at x offset sx: the arm vertical, the blade bent forward-down 45 degrees."""
    return [((_ax + sx, 0.0, _az + ARM_L / 2), (ARM_T, ARM_W, ARM_L), STEEL),
            ((_ax + sx + _c * BLADE_L / 2, 0.0, _az - _s * BLADE_L / 2), (BLADE_L, BLADE_W, BLADE_T), STEEL,
             (0.0, math.radians(45.0)))]


TOP_Z = _az + ARM_L  # the arms' upper ends
TWEEZER_JAW = Joint("jaw", kind="slide", axis="X", anchor=(_ax + SEP / 2, 0.0, _az + ARM_L / 2),
                    lower=-(SEP - BLADE_T / _s - 0.0002), upper=0.0, initial=0.0, target=0.0, stiffness=2000.0,
                    damping=5.0, mass=0.008, boxes=_half(SEP / 2))
# The cup: a block on the bench; the tweezers stand in it on their blade tips, upright.
TW_XY = (0.30, -0.28)  # off to the side, clear of the path to the wafer
CUP_FLOOR, CUP_H = 0.020, 0.035  # the cup's floor above the table; its walls above the floor
CUP_IN = ((-0.0150, 0.0065), 0.0065)  # inner x span, inner half width (y): a snug fit, so they stand upright
_cw = 0.003
TSTAND = part("tweezer_cup", pos=(*TW_XY, TOP), fixed=True, friction=0.8,
              boxes=[((sum(CUP_IN[0]) / 2, 0.0, CUP_FLOOR / 2), (CUP_IN[0][1] - CUP_IN[0][0] + 2 * _cw,
                                                                  2 * CUP_IN[1] + 2 * _cw, CUP_FLOOR), STAND_C),
                     *[((x, 0.0, CUP_FLOOR + CUP_H / 2), (_cw, 2 * CUP_IN[1] + 2 * _cw, CUP_H), STAND_C)
                       for x in (CUP_IN[0][0] - _cw / 2, CUP_IN[0][1] + _cw / 2)],
                     *[((sum(CUP_IN[0]) / 2, e * (CUP_IN[1] + _cw / 2), CUP_FLOOR + CUP_H / 2),
                        (CUP_IN[0][1] - CUP_IN[0][0], _cw, CUP_H), STAND_C) for e in (-1, 1)]])
TWEEZERS = part(
    "tweezers", pos=(*TW_XY, TOP + CUP_FLOOR + 0.0012), mass=0.020, friction=0.9,
    boxes=[*_half(-SEP / 2), ((_ax, 0.0, TOP_Z + 0.005), (SEP + ARM_T, ARM_W, 0.010), STEEL)],  # the joined end
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


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm, with the tweezers: lifts them out of their cup from above, gripping their two arms near the joined end,
# tilts them 45 degrees so the blades lie flat, slides the open blades round the wafer's near edge on its stand, pinches it, carries it to the
# aligner on the bench and lowers it into the pocket (the lower blade drops into the notch), lets go, backs the blades
# out, and stands the tweezers back in their cup. Left arm: lifts the lid by its handle bar, the bar riding between its
# part-open fingers along the lid's arc, to 70 degrees, and holds it there. Right arm: picks the aligner up by its
# handle, carries it over the chuck, lowers it until its slot straddles the chuck (closed loop on the aligner's live
# position), pushes it on until the V seats against the chuck, lowers it 4.5 mm (the wafer is left on the chuck), slides
# it back out from under the wafer, lifts it out of the bowl and sets it down. Left arm: lowers the lid shut.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._rot = True
    _demo.grip_rotation = _grip

TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.15, -0.25, TOP + 0.16)
Y = np.array([0.0, 1.0, 0.0])
TW_ROT = Orient(axis_angle_matrix(Y, math.radians(-45.0)) @ _prev_grip(math.pi / 2))  # approach forward-down 45 deg
PAD_IN = 0.006  # the pad tips reach this far past the point gripped


def _tcp(point, rot):
    """TCP that puts the pads round ``point`` (world), their tips PAD_IN beyond it along the approach."""
    a = rot.rot[:, 2]
    return tuple(np.asarray(point, float) - a * (TIP_ABOVE_TCP - PAD_IN))


TW_GRIP_F = np.array([_ax, 0.0, TOP_Z - 0.020])  # gripped 2 cm below the joined end (tweezer frame, upright)
TW_TILT = axis_angle_matrix(Y, math.radians(-45.0))  # upright -> held: arms rising back at 45 degrees, blades flat
TW_GRIP = TW_TILT @ TW_GRIP_F  # the grip point, from the blade tips, when held
DOWN_X = Orient(_prev_grip(math.pi / 2))  # pointing down, fingers closing along x (across the arms' faces)
TW_WIDE = -0.014  # pads 28 mm apart round the arms (11.3 mm over their outer faces)
TW_HOLD = -0.0050  # just closed on the arms' outer faces: the blades stay open
TW_PINCH = -0.0024  # squeezes the arms: the blades close on the wafer


def _tw_tcp(origin):
    """TCP holding the tweezers (turned to blades flat) with their origin (between the blade tips) at ``origin``."""
    return _tcp(np.asarray(origin, float) + TW_GRIP, TW_ROT)


def _cup_tcp(origin):
    """TCP holding the tweezers upright (pointing down) with their origin at ``origin``."""
    return _tcp(np.asarray(origin, float) + TW_GRIP_F, DOWN_X)


PINCH_IN = 0.006  # the blade tips this far inside the wafer's edge
_wb = TOP + PAD_TOP  # the wafer's underside on its stand
WAFER_PICK = (STAND_XY[0] - WAFER_R + PINCH_IN, STAND_XY[1], _wb + 0.0002 + OPEN_GAP / 2 - 0.0005)
POCKET_PLACE = (ALIGNER_XY[0] - WAFER_R + PINCH_IN, ALIGNER_XY[1], TOP + PT + 0.0008 + OPEN_GAP / 2 - 0.0005)
TW_HOME = np.array([TW_XY[0], TW_XY[1], TOP + CUP_FLOOR + 0.0012])

AL_HANDLE_X = -0.078  # gripped here along its handle (aligner frame)
AL_WIDE, AL_HOLD = -0.012, -0.004
AL_GRIP_Z = HANDLE_A[2][1] - 0.004  # pad tips 4 mm down the 6 mm handle
DOCK_Z = CHUCK_TOP - 0.002  # the plate's underside: its top 1 mm over the chuck's, the wafer resting on it
DROP = 0.0045
BACK = 0.055  # slide the aligner back this far after the drop, clear of the wafer
DOCK_PRE = 0.004  # come down this far short of the V seating
DOCK_PUSH = 0.0015  # aim this far past seating: a light push


def _al_tcp(pocket):
    """TCP holding the aligner by its handle, its pocket centre (plate underside) at ``pocket`` (world)."""
    p = np.asarray(pocket, float)
    return (p[0] + AL_HANDLE_X, p[1], p[2] + AL_GRIP_Z + TIP_ABOVE_TCP)


_live = {"corr": np.zeros(2), "on": False}


def _live_reset():
    _live.update(corr=np.zeros(2), on=False)


@on_reset
def _reset(env):
    _live_reset()


@on_step
def _track(env):
    """While ``on``, trim the aligner hand so the pocket's centre comes over the chuck's axis (part of the error per
    step)."""
    if not _live["on"]:
        return
    a = labkit._part_state(env, labkit._SPECS["aligner"])
    err = CHUCK_XY + np.array([-DOCK_PRE, 0.0]) - a.pos[:2]
    _live["corr"] = np.clip(_live["corr"] + np.clip(0.15 * err, -0.0004, 0.0004), -0.004, 0.004)


class _AlLive:
    """The aligner hand's TCP read every step: ``pocket`` plus the closed-loop trim (frozen once ``freeze``)."""

    def __init__(self, pocket, track=True):
        self.pocket = np.asarray(pocket, float)
        self.track = track

    def __array__(self, dtype=None, copy=None):
        _live["on"] = self.track
        c = _live["corr"]
        return np.asarray(_al_tcp(self.pocket + np.array([c[0], c[1], 0.0])), dtype=dtype or float)


def _bar_at(th: float):
    """World position of the lid's handle bar axis with the lid open th (rad)."""
    d = np.array([HANDLE_X - HINGE[0], 0.0, BAR_Z - HINGE[2]])
    x = d[0] * math.cos(th) + d[2] * math.sin(th)
    z = -d[0] * math.sin(th) + d[2] * math.cos(th)
    return np.array(C(HINGE[0] + x, HANDLE_Y, HINGE[2] + z))


LID_OPEN = math.radians(70.0)
BAR_WIDE = -0.008  # pads 16 mm apart round the 10 mm bar


def _lid_rot(th):
    """The holding hand's rotation: approach nearly level (+x), fingers closing nearly vertically, so the pads lie
    above and below the round bar and it can turn between them as the lid swings."""
    pitch = math.radians(-75.0 + 15.0 * th / LID_OPEN)  # 75 degrees from straight down, 60 when held open
    return Orient(axis_angle_matrix(Y, pitch) @ _prev_grip(-math.pi / 2))


def _lid_tcp(th, lift=0.0):
    """TCP with the bar between the part-open pads, the lower pad bearing on it (``lift`` m below the bar's centre)."""
    rot = _lid_rot(th)
    n = rot.rot[:, 0]  # the fingers' closing direction (mostly vertical)
    if n[2] < 0:
        n = -n
    return _tcp(_bar_at(th) - n * lift, rot)


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    q, d = TW_ROT, DOWN_X
    right = [
        Waypoint(3.0, _cup_tcp(TW_HOME + [0.0, 0.0, 0.05]), TW_WIDE, "above the tweezers standing in their cup", d),
        Waypoint(1.5, _cup_tcp(TW_HOME), TW_WIDE, "down: pads either side of their arms", d),
        Waypoint(0.8, _cup_tcp(TW_HOME), TW_HOLD, "hold the tweezers, blades open", d),
        Waypoint(1.5, _cup_tcp(TW_HOME + [0.0, 0.0, 0.06]), TW_HOLD, "lift them out of the cup", d),
        Waypoint(3.0, _tw_tcp(np.array(WAFER_PICK) + [-0.012, 0.0, 0.03]), TW_HOLD,
                 "tilt them, blades flat, towards the wafer's near edge", q),
        Waypoint(1.2, _tw_tcp(np.array(WAFER_PICK) + [-0.012, 0.0, 0.0]), TW_HOLD, "blades level with the wafer", q),
        Waypoint(1.5, _tw_tcp(WAFER_PICK), TW_HOLD, "slide the open blades round its edge", q),
        Waypoint(0.8, _tw_tcp(WAFER_PICK), TW_PINCH, "pinch the wafer by its edge", q),
        Waypoint(1.2, _tw_tcp(np.array(WAFER_PICK) + [0.0, 0.0, 0.03]), TW_PINCH, "lift it off the stand", q),
        Waypoint(3.0, _tw_tcp(np.array(POCKET_PLACE) + [0.0, 0.0, 0.03]), TW_PINCH, "over the aligner's pocket", q),
        Waypoint(1.5, _tw_tcp(POCKET_PLACE), TW_PINCH, "lower it into the pocket, the lower blade in the notch", q),
        Waypoint(0.8, _tw_tcp(POCKET_PLACE), TW_HOLD, "let go: open the blades", q),
        Waypoint(1.2, _tw_tcp(np.array(POCKET_PLACE) + [-0.015, 0.0, 0.0]), TW_HOLD, "back the blades out", q),
        Waypoint(1.0, _tw_tcp(np.array(POCKET_PLACE) + [-0.015, 0.0, 0.03]), TW_HOLD, "up", q),
        Waypoint(3.0, _cup_tcp(TW_HOME + [0.0, 0.0, 0.06]), TW_HOLD, "upright again, over the cup", d),
        Waypoint(1.5, _cup_tcp(TW_HOME + [0.0, 0.0, 0.002]), TW_HOLD, "stand them back in the cup", d),
        Waypoint(0.6, _cup_tcp(TW_HOME + [0.0, 0.0, 0.002]), TW_WIDE, "let go", d),
        Waypoint(1.2, _cup_tcp(TW_HOME + [0.0, 0.0, 0.05]), TW_WIDE, "up", d),
    ]
    ax, ay = ALIGNER_XY
    al0 = np.array([ax, ay, TOP])
    right += [
        Waypoint(2.5, _al_tcp(al0 + [0, 0, 0.04]), AL_WIDE, "above the aligner's handle"),
        Waypoint(1.2, _al_tcp(al0), AL_WIDE, "fingers round the handle"),
        Waypoint(0.8, _al_tcp(al0), AL_HOLD, "grip the handle"),
        Waypoint(1.2, _al_tcp(al0 + [0, 0, 0.05]), AL_HOLD, "lift the aligner level, the wafer in its pocket"),
    ]
    t_al = _dur(right)

    left = [Waypoint(3.0, LEFT_PARK, FINGER_OPEN, "wait")]
    left.append(Waypoint(max(0.3, t_al - 10.0 - _dur(left)), LEFT_PARK, FINGER_OPEN, "wait"))
    left += [
        Waypoint(2.5, tuple(np.array(_lid_tcp(0.0)) + [-0.03, 0.0, 0.02]), BAR_WIDE, "in front of the lid's handle",
                 _lid_rot(0.0)),
        Waypoint(1.2, _lid_tcp(0.0), BAR_WIDE, "the handle bar between the fingers", _lid_rot(0.0)),
    ]
    for th in np.linspace(0.12, LID_OPEN, 8):
        left.append(Waypoint(0.7, _lid_tcp(th, 0.0015), BAR_WIDE, "lift the lid along its arc", _lid_rot(th)))
    t_open = _dur(left)

    dock_hi = np.array(C(-DOCK_PRE, 0.0, DOCK_Z + 0.030))
    dock = np.array(C(-DOCK_PRE, 0.0, DOCK_Z))
    right.append(Waypoint(max(0.3, t_open - _dur(right)), _al_tcp(al0 + [0, 0, 0.05]), AL_HOLD, "wait for the lid"))
    right += [
        Waypoint(3.0, _al_tcp(dock_hi), AL_HOLD, "over the chuck, the slot lined up with it"),
        Waypoint(0.1, _AlLive(dock_hi), AL_HOLD, "line the slot up with the chuck"),
        Waypoint(2.0, _AlLive(dock_hi), AL_HOLD, "line the slot up with the chuck"),
        Waypoint(2.0, _AlLive(dock), AL_HOLD, "lower it: the slot straddles the chuck"),
        Waypoint(1.5, _AlLive(dock + [DOCK_PRE + DOCK_PUSH, 0, 0], track=False), AL_HOLD,
                 "push it on: the V seats against the chuck, the pocket centred on it"),
        Waypoint(0.5, _AlLive(dock + [DOCK_PRE + DOCK_PUSH, 0, 0], track=False), AL_HOLD, "steady"),
        Waypoint(1.5, _AlLive(dock + [DOCK_PRE + DOCK_PUSH, 0, -DROP], track=False), AL_HOLD,
                 "lower it: the wafer is left on the chuck"),
        Waypoint(2.5, _AlLive(dock + [DOCK_PRE - BACK, 0, -DROP], track=False), AL_HOLD,
                 "slide it back out from under the wafer"),
        Waypoint(1.5, _AlLive(dock + [DOCK_PRE - BACK, 0, 0.04], track=False), AL_HOLD, "lift it out of the bowl"),
        Waypoint(3.0, _al_tcp(al0 + [0, 0, 0.02]), AL_HOLD, "back over the bench"),
        Waypoint(1.2, _al_tcp(al0 + [0, 0, 0.002]), AL_HOLD, "set it down"),
        Waypoint(0.6, _al_tcp(al0 + [0, 0, 0.002]), AL_WIDE, "let go"),
        Waypoint(1.0, _al_tcp(al0 + [0, 0, 0.04]), AL_WIDE, "up"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_out = _dur(right) - 6.3
    left.append(Waypoint(max(0.3, t_out - _dur(left)), _lid_tcp(LID_OPEN, 0.0015), BAR_WIDE, "hold the lid open",
                         _lid_rot(LID_OPEN)))
    for th in np.linspace(LID_OPEN - 0.15, 0.0, 8):
        left.append(Waypoint(0.7, _lid_tcp(th, 0.0015), BAR_WIDE, "lower the lid shut", _lid_rot(th)))
    left += [
        Waypoint(1.0, tuple(np.array(_lid_tcp(0.0)) + [-0.03, 0.0, 0.02]), BAR_WIDE, "off the handle", _lid_rot(0.0)),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
