"""On the station table: a benchtop spin coater for small wafers with a coated 2-inch silicon wafer on its chuck, a
pair of angled wafer tweezers standing tips down in a cup, and a round plastic single-wafer case in two pieces: its
flat lid lying open side up, and its domed bottom lying beside it, dome up.

The coater: a white housing 22 x 22 cm with a round catch bowl and, in the middle of the bowl, a black vacuum chuck
2.4 cm across, the wafer resting on it (vacuum off). Its clear lid is hinged along the back edge and lifts at the front
by a black handle bar; it stands open against its stop, past upright, as it was left after spinning.

The case (polypropylene): the lid is a flat round tray with a ring of low pins the wafer rests on and a low wall round
them, open on the side towards the robots for the tweezers' blades. The domed bottom fits over the wall and sits on
the lid's wide flange; a bayonet closes it: set on with its arrow at the lid's OPEN mark and turned 30 degrees
clockwise to the LOCK mark.

The task: with the tweezers, take the wafer off the chuck by its edge and lay it, pattern side up, in the case's lid;
set the domed bottom on the case's lid and turn it to lock while holding the case's lid down. The wafer is never
touched by the gripper.

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
SI = (0.32, 0.34, 0.42)
STEEL = (0.72, 0.73, 0.75)
STAND_C = (0.55, 0.56, 0.58)
PP = (0.90, 0.91, 0.93)  # the case's polypropylene
PP_B = (0.80, 0.86, 0.93)  # the domed bottom, tinted
MARK = (0.85, 0.20, 0.12)

# -- the spin coater (fixed). Frame: origin on the table under the chuck's axis, axes along the world's. --------------
CX, CY = 0.31, -0.12  # to the robots' right; the case sits in front of the robots' middle
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
LID_MAX = 1.75  # the lid's stop (rad, 100 degrees): past upright, so it stays open, as it was left after spinning
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
LID = Joint("lid", kind="hinge", axis="Y", anchor=HINGE, lower=0.0, upper=LID_MAX, initial=LID_MAX, target=0.0,
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

# -- the wafer (loose), on the chuck -----------------------------------------------------------------------------------
WAFER_R, WAFER_T = 0.0254, 0.0010  # 2-inch wafer, drawn 1 mm thick (real: 0.28 mm) for stable contact
WAFER = part("wafer", pos=(CX, CY, TOP + CHUCK_TOP), mass=0.004, friction=0.8,
             cylinders=[((0.0, 0.0, WAFER_T / 2), WAFER_R, WAFER_T, SI)])

# -- the tweezers (loose): angled wafer tweezers, standing upright in a cup, tips down. Frame: origin between the two
# blades' tips, with the arms vertical (joined end up) and the blades bent 45 degrees forward (+x) and down from the
# arms' lower ends. The arms are flat strips, their faces normal to x; the lower blade and arm (x < 0) and the joined
# end are the body, the upper blade and arm (x > 0) slide together along x (sprung open). Held in a hand turned 45
# degrees (arms rising back at 45 degrees), the blades lie flat, pointing forward.
BLADE_L, BLADE_W, BLADE_T = 0.012, 0.014, 0.0012
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
                    lower=-(SEP - BLADE_T / _s - 0.0002), upper=0.0, initial=0.0, target=0.0, stiffness=600.0,
                    damping=5.0, mass=0.008, boxes=_half(SEP / 2))
# The cup: a block on the bench; the tweezers stand in it on their blade tips, upright.
TW_XY = (0.17, -0.27)  # in front of the coater, to the right
CUP_FLOOR, CUP_H = 0.020, 0.035  # the cup's floor above the table; its walls above the floor
CUP_IN = ((-0.0150, 0.0065), 0.0085)  # inner x span, inner half width (y): a snug fit, so they stand upright
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
    boxes=_half(-SEP / 2),  # the two halves' joined end is the slide joint: nothing stands above the arms' tops
    joints=[TWEEZER_JAW],
)

# -- the case (loose): lid and domed bottom. Frames: origin on the axis, at the underside. ------------------------------
CASE_XY = (0.15, 0.02)  # the lid, open side up, in front of the robots' middle
COVER_XY = (0.30, -0.32)  # the domed bottom, dome up, to the right, in front of the coater
BASE_R, BASE_T = 0.053, 0.002  # the lid's base: floor and a wide flange, to hold it down by
PIN_R, PIN_RAD, PIN_H = 0.0235, 0.0012, 0.003  # the pins the wafer rests on
WALL_R, WALL_T, WALL_H = 0.0285, 0.002, 0.007  # the low wall round the pocket
GAP = math.radians(32.0)  # half the opening in the wall (and the pins) towards -x, for the tweezers' blades
GAP_DIR = math.pi
N_WALL, N_PIN = 36, 18
_wseg = 2 * math.pi * WALL_R / N_WALL * 1.08


def _off_gap(a):
    return abs(math.atan2(math.sin(a - GAP_DIR), math.cos(a - GAP_DIR))) > GAP


_wall = [((WALL_R * math.cos(a), WALL_R * math.sin(a), BASE_T + WALL_H / 2), (WALL_T, _wseg, WALL_H), PP, a)
         for a in (2 * math.pi * (i + 0.5) / N_WALL for i in range(N_WALL)) if _off_gap(a)]
_pins = [((PIN_R * math.cos(a), PIN_R * math.sin(a), BASE_T + PIN_H / 2), PIN_RAD, PIN_H, PP)
         for a in (2 * math.pi * (i + 0.5) / N_PIN for i in range(N_PIN)) if _off_gap(a)]
LOCK = math.radians(-30.0)  # the bayonet's turn, clockwise seen from above
_marks = [((0.0415 * math.cos(a), 0.0415 * math.sin(a), BASE_T + 0.0002), (0.007, 0.0015, 0.0004), MARK, a)
          for a in (math.pi, math.pi + LOCK)]  # OPEN and LOCK, at the front
CASE_LID = part("case_lid", pos=(*CASE_XY, TOP), mass=0.020, friction=0.7,
                boxes=[*_wall, *_marks],
                cylinders=[((0.0, 0.0, BASE_T / 2), BASE_R, BASE_T, PP), *_pins])

SKIRT_R, SKIRT_T, SKIRT_H = 0.0355, 0.002, 0.014  # the bottom's skirt: centre radius, thickness, height
N_SKIRT = 40
_sseg = 2 * math.pi * SKIRT_R / N_SKIRT * 1.08
_dome = [(0.0365, 0.002), (0.031, 0.002), (0.023, 0.0015), (0.013, 0.001)]  # stepped dome: radius, thickness
_dz, _dome_cyl = SKIRT_H, []
for r, t in _dome:
    _dome_cyl.append(((0.0, 0.0, _dz + t / 2), r, t, PP_B))
    _dz += t
COVER_TOP = _dz
CASE_COVER = part("case_bottom", pos=(*COVER_XY, TOP), mass=0.015, friction=0.6,
                  boxes=[*[((SKIRT_R * math.cos(a), SKIRT_R * math.sin(a), SKIRT_H / 2), (SKIRT_T, _sseg, SKIRT_H), PP_B,
                            a) for a in (2 * math.pi * (i + 0.5) / N_SKIRT for i in range(N_SKIRT))],
                         ((-(SKIRT_R + SKIRT_T / 2 + 0.0004), 0.0, SKIRT_H / 2), (0.0008, 0.003, 0.010), MARK)],  # arrow
                  cylinders=_dome_cyl)

OBJECTS = [COATER, WAFER, TWEEZERS, TSTAND, CASE_LID, CASE_COVER]


def C(x, y, z):
    """Coater coordinates to world."""
    return (CX + x, CY + y, TOP + z)


# -- scoring ----------------------------------------------------------------------------------------------------------
FLAT_DEG = 3.0
POCKET_TOL = 0.0025  # the wafer's centre within 2.5 mm of the case's axis: inside the wall
REST_Z = BASE_T + PIN_H  # the wafer's underside on the pins, above the lid's underside
SEAT_TOL = 0.0025
LOCK_WIN = (math.radians(-45.0), math.radians(-22.0))  # turned into the lock: 22 to 45 degrees clockwise
OPEN_WIN = math.radians(10.0)  # set on within 10 degrees of the OPEN mark
TOP_TOUCH = 0.006  # a pad tip within 6 mm above the wafer's top, more than 3 mm inside its rim: the top was touched
HANDLED = 0.002  # a pad tip within 2 mm outside the wafer's rim and 8 mm of its height: held by the gripper
HANDS_FAR = 0.08  # both pad tips this far from the case's axis at the end

_state = {"touched": False, "handled": False, "dropped": False, "armed": False}


@on_reset
def _reset_state(env):
    _state.update(touched=False, handled=False, dropped=False, armed=False, far=True)


def _tips(env):
    """Both pads' tips of both hands (world, env frame): either side of the gripper's centre line by half the opening,
    along the fingers' closing direction."""
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7"], preserve_order=True)[0]
        half = abs(float(robot.data.joint_pos.torch[0, ids].cpu().numpy()[0]))
        out += [mid + half * g[:3, 0], mid - half * g[:3, 0]]
    return out


def _yaw(s) -> float:
    r = s.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def rel_turn(lid, cover) -> float:
    d = _yaw(cover) - _yaw(lid)
    return math.atan2(math.sin(d), math.cos(d))


def seated(lid, cover) -> bool:
    lc = lid.to_local(cover.pos)
    return (np.hypot(lc[0], lc[1]) <= SEAT_TOL and abs(lc[2] - BASE_T) <= 0.0015 and cover.tilt_deg() <= FLAT_DEG
            and lid.tilt_deg() <= FLAT_DEG)


def in_pocket(lid, w) -> bool:
    lc = lid.to_local(w.pos)
    return np.hypot(lc[0], lc[1]) <= POCKET_TOL and abs(lc[2] - REST_Z) <= 0.001 and w.tilt_deg() <= FLAT_DEG


@on_step
def _watch(env):
    """Latch the violations (the wafer's top touched, the wafer held by the gripper, the wafer dropped), and whether the
    bottom was set on at the OPEN mark (it stays armed while it stays seated)."""
    S = labkit._SPECS
    w = labkit._part_state(env, S["wafer"])
    lid = labkit._part_state(env, S["case_lid"])
    cover = labkit._part_state(env, S["case_bottom"])
    p = np.asarray(w.pos, float)
    top = p[2] + WAFER_T
    if w.tilt_deg() > 40.0 or p[2] < TOP + 0.003:
        _state["dropped"] = True
    tips = _tips(env)
    for tip in tips:
        r = np.hypot(*(tip[:2] - p[:2]))
        if r < WAFER_R - 0.003 and top - 0.001 < tip[2] < top + TOP_TOUCH:
            _state["touched"] = True
        if r < WAFER_R + HANDLED and abs(tip[2] - (p[2] + WAFER_T / 2)) < 0.008:
            _state["handled"] = True
    if seated(lid, cover):
        if abs(rel_turn(lid, cover)) <= OPEN_WIN:
            _state["armed"] = True
    else:
        _state["armed"] = False
    _state["far"] = all(np.hypot(*(t[:2] - np.asarray(lid.pos[:2]))) >= HANDS_FAR for t in tips)


def locked(lid, cover) -> bool:
    return seated(lid, cover) and _state["armed"] and LOCK_WIN[0] <= rel_turn(lid, cover) <= LOCK_WIN[1]


def score(objects):
    """0 if the wafer's top was ever touched, the wafer was ever held by the gripper itself, or it was dropped (tilted
    past 40 degrees, or on the table). Otherwise, with the wafer lying flat in the case's lid on its pins, its centre
    within 2 mm of the case's axis: 1.0 if the domed bottom is seated on the lid and locked (set on at the OPEN mark,
    within 10 degrees, and turned 22 to 45 degrees clockwise while seated) with both hands 8 cm from the case's axis; 0.5
    if the bottom is seated but not locked, or a hand is still at the case; 0.25 if the wafer is in the lid but the
    bottom is not seated. 0 if the wafer is not in the lid."""
    if _state["touched"] or _state["handled"] or _state["dropped"]:
        return 0.0
    w, lid, cover = objects["wafer"], objects["case_lid"], objects["case_bottom"]
    if not in_pocket(lid, w):
        return 0.0
    if locked(lid, cover):
        return 1.0 if _state.get("far", True) else 0.5
    return 0.5 if seated(lid, cover) else 0.25


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    w, lid = objects["wafer"], objects["case_lid"]
    lc = lid.to_local(w.pos)
    return float(np.hypot(lc[0], lc[1]) + abs(lc[2] - REST_Z))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm, with the tweezers: lifts them out of their cup from above, gripping their two arms near the top, tilts
# them 45 degrees so the blades lie flat, slides the open blades round the wafer's near edge on the chuck (closed loop
# on the tweezers' live tips), pinches it, lifts it out of the bowl and lays it in the case's lid through the gap in its
# wall, lets go, backs the blades out and stands the tweezers back in their cup. Then it picks the domed bottom up by
# its skirt and sets it on the case's lid with its arrow at the OPEN mark (closed loop on the lid's live position). Left arm: holds the lid's flange down with closed fingertips while the right
# turns the bottom 30 degrees clockwise to lock it; both let go.
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


TW_GRIP_F = np.array([_ax, 0.0, TOP_Z - 0.014])  # gripped 14 mm below the arms' tops, the palm just clear
# of them (tweezer frame, upright)
TW_TILT = axis_angle_matrix(Y, math.radians(-45.0))  # upright -> held: arms rising back at 45 degrees, blades flat
TW_GRIP = TW_TILT @ TW_GRIP_F  # the grip point, from the blade tips, when held
DOWN_X = Orient(_prev_grip(math.pi / 2))  # pointing down, fingers closing along x (across the arms' faces)
TW_WIDE = -0.014  # pads 28 mm apart round the arms (11.3 mm over their outer faces)
TW_HOLD = -0.0050  # just closed on the arms' outer faces: the blades stay open
TW_PINCH = -0.0016  # squeezes the arms: the blades close on the wafer


Z = np.array([0.0, 0.0, 1.0])


def tw_rot(psi=0.0):
    """The hand's rotation holding the tweezers blades flat, pointing along heading ``psi`` (0: +x)."""
    return Orient(axis_angle_matrix(Z, psi) @ TW_ROT.rot)


def _tw_tcp(origin, psi=0.0):
    """TCP holding the tweezers (turned to blades flat, pointing along ``psi``) with their origin (between the blade
    tips) at ``origin``."""
    return _tcp(np.asarray(origin, float) + axis_angle_matrix(Z, psi) @ TW_GRIP, tw_rot(psi))


def _cup_tcp(origin):
    """TCP holding the tweezers upright (pointing down) with their origin at ``origin``."""
    return _tcp(np.asarray(origin, float) + TW_GRIP_F, DOWN_X)


PINCH_IN = 0.0085  # the blade tips this far inside the wafer's edge (the upper blade draws back ~3 mm as it closes)
BLADE_Z = 0.0022  # the tips' midpoint this far above the wafer's underside: the held-open lower blade 0.3 mm under
# it (only the upper blade moves as they close), the upper 3 mm over it
WAFER_PICK = np.array([CX - WAFER_R + PINCH_IN, CY, TOP + CHUCK_TOP + BLADE_Z])
CASE_PLACE = np.array([CASE_XY[0] - WAFER_R + PINCH_IN, CASE_XY[1], TOP + REST_Z + 0.0008 + BLADE_Z])  # from -x
TW_HOME = np.array([TW_XY[0], TW_XY[1], TOP + CUP_FLOOR + 0.0012])
DOWN_Y = Orient(_prev_grip(0.0))  # pointing down, fingers closing along y
COVER_WIDE, COVER_HOLD = -0.044, -0.0335  # pads round the 75 mm skirt; squeezing it
COVER_GRIP_Z = 0.008  # pad tips this far above the skirt's lower edge
PRESS_R = 0.049  # the left fingertips press the lid's flange here, on its left (+y), clear of the right's fingers
PRESS_IN = 0.0025  # aim this far into the flange: a firm press

# what the demo sees of the case's lid: its position, every step
_seen = {"lid": np.array([*CASE_XY, TOP])}


@on_step
def _see(env):
    _seen["lid"] = np.asarray(labkit._part_state(env, labkit._SPECS["case_lid"]).pos, float)
    if getattr(_demo, "_plan_clock", 0.0) < _seen.get("freeze_t", 1e9):
        _seen["held"] = _seen["lid"].copy()


@on_reset
def _reset_seen(env):
    _seen["lid"] = np.array([*CASE_XY, TOP])
    _seen["held"] = _seen["lid"].copy()


class _OnLid:
    """A TCP given relative to the case lid's position as seen up to ``_seen["freeze_t"]`` (set by the demo to when the
    hands start bearing on the case, so they do not chase it). ``track`` is kept for readability only."""

    def __init__(self, fn, track=True):
        self.fn = fn
        self.track = track

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.fn(_seen["held"]), dtype=dtype or float)


def _cover_tcp(axis_pt):
    """TCP holding the bottom by its skirt, its axis (underside) at ``axis_pt``."""
    return tuple(np.asarray(axis_pt, float) + [0.0, 0.0, COVER_GRIP_Z + TIP_ABOVE_TCP])


# what the demo sees of the tweezers once held: their origin (between the blade tips); the right hand's aim is trimmed
# by the offset between where it means the tips to be and where they are, integrated only while they are still
_tw = {"corr": np.zeros(3), "goal": None, "prev": None}


@on_reset
def _reset_tw(env):
    _tw.update(corr=np.zeros(3), goal=None, prev=None)


@on_step
def _track_tw(env):
    pos = np.asarray(labkit._part_state(env, labkit._SPECS["tweezers"]).pos, float)
    prev, _tw["prev"] = _tw["prev"], pos
    if _tw["goal"] is None or prev is None:
        return
    err = _tw["goal"] - pos
    if np.linalg.norm(pos - prev) < 0.0002 and np.linalg.norm(err) < 0.04:
        _tw["corr"] = np.clip(_tw["corr"] + np.clip(0.3 * err, -0.0005, 0.0005), -0.03, 0.03)


class _TwAt:
    """The right hand's TCP to put the held tweezers' tips at ``origin`` (blades flat, or upright when ``cup``),
    trimmed by the live correction; ``track`` False holds the correction (and, before they are held, it is zero)."""

    def __init__(self, origin, cup=False, track=True, psi=0.0):
        self.origin = np.asarray(origin, float)
        self.cup = cup
        self.track = track
        self.psi = psi

    def __array__(self, dtype=None, copy=None):
        _tw["goal"] = self.origin if self.track else None
        tcp = _cup_tcp(self.origin) if self.cup else _tw_tcp(self.origin, self.psi)
        return np.asarray(np.asarray(tcp) + _tw["corr"], dtype=dtype or float)


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
    _tw.update(corr=np.zeros(3), goal=None, prev=None)
    _seen.pop("freeze_t", None)
    d, q0, q1 = DOWN_X, tw_rot(0.0), tw_rot(0.0)
    P, K = WAFER_PICK, CASE_PLACE
    H = 0.0  # heading into the case's lid: +x, through the gap in its wall
    right = [
        Waypoint(3.0, _TwAt(TW_HOME + [0.0, 0.0, 0.05], cup=True, track=False), TW_WIDE,
                 "above the tweezers standing in their cup", d),
        Waypoint(1.5, _TwAt(TW_HOME, cup=True, track=False), TW_WIDE, "down: pads either side of their arms", d),
        Waypoint(0.8, _TwAt(TW_HOME, cup=True, track=False), TW_HOLD, "hold the tweezers, blades open", d),
        Waypoint(1.5, _TwAt(TW_HOME + [0.0, 0.0, 0.06], cup=True, track=False), TW_HOLD, "lift them out of the cup", d),
        Waypoint(3.0, _TwAt(P + [-0.03, 0.0, 0.06]), TW_HOLD, "tilt them, blades flat, above the bowl", q0),
        Waypoint(2.0, _TwAt(P + [-0.015, 0.0, 0.02]), TW_HOLD, "down into the bowl, towards the wafer's near edge", q0),
        Waypoint(2.0, _TwAt(P + [-0.015, 0.0, 0.0]), TW_HOLD, "blades level with the wafer", q0),
        Waypoint(1.5, _TwAt(P + [-0.015, 0.0, 0.0]), TW_HOLD, "line up, seen against the wafer", q0),
        Waypoint(2.0, _TwAt(P), TW_HOLD, "slide the open blades round its edge", q0),
        Waypoint(1.5, _TwAt(P), TW_HOLD, "line up: the lower blade just under the wafer", q0),
        Waypoint(0.8, _TwAt(P, track=False), TW_PINCH, "pinch the wafer by its edge", q0),
        Waypoint(1.5, _TwAt(P + [0.0, 0.0, 0.04], track=False), TW_PINCH, "lift it off the chuck", q0),
        Waypoint(2.0, _TwAt(P + [-0.04, 0.0, 0.06], track=False), TW_PINCH, "up and out over the bowl's rim", q0),
        Waypoint(4.0, _TwAt(K + [0.0, 0.0, 0.03], psi=H), TW_PINCH, "carry it over the case's lid", q1),
        Waypoint(2.0, _TwAt(K + [0.0, 0.0, 0.03], psi=H), TW_PINCH, "line up over the pocket", q1),
        Waypoint(1.5, _TwAt(K, track=False, psi=H), TW_PINCH, "lower it onto the pins, the blades in the wall's gap", q1),
        Waypoint(0.8, _TwAt(K, track=False, psi=H), TW_HOLD, "let go: open the blades", q1),
        Waypoint(1.2, _TwAt(K + [-0.016, 0.0, 0.0], track=False, psi=H), TW_HOLD, "back the blades out", q1),
        Waypoint(1.0, _TwAt(K + [-0.016, 0.0, 0.03], track=False, psi=H), TW_HOLD, "up", q1),
        Waypoint(3.0, _TwAt(TW_HOME + [0.0, 0.0, 0.06], cup=True), TW_HOLD, "upright again, over the cup", d),
        Waypoint(2.0, _TwAt(TW_HOME + [0.0, 0.0, 0.06], cup=True), TW_HOLD, "line them up over the cup", d),
        Waypoint(1.5, _TwAt(TW_HOME + [0.0, 0.0, 0.002], cup=True, track=False), TW_HOLD, "lower them into the cup", d),
        Waypoint(0.6, _TwAt(TW_HOME + [0.0, 0.0, 0.002], cup=True, track=False), TW_WIDE, "let go: they stand in it", d),
        Waypoint(1.2, _TwAt(TW_HOME + [0.0, 0.0, 0.05], cup=True, track=False), TW_WIDE, "up", d),
    ]

    # right, then: the domed bottom onto the case's lid, arrow at OPEN; with the lid held down, turn it to LOCK.
    # Fingers closing along x, so the left hand can come in on the lid's left side.
    c0 = np.array([*COVER_XY, TOP])
    Q = math.pi / 2  # the hand's yaw: fingers closing along x
    on_lid = lambda dz: _OnLid(lambda L: _cover_tcp(L + [0, 0, dz]))  # noqa: E731
    right += [
        Waypoint(2.5, _cover_tcp(c0 + [0, 0, 0.04]), COVER_WIDE, "above the domed bottom", Q),
        Waypoint(1.5, _cover_tcp(c0), COVER_WIDE, "fingers either side of its skirt", Q),
        Waypoint(0.8, _cover_tcp(c0), COVER_HOLD, "grip the skirt", Q),
        Waypoint(1.5, _cover_tcp(c0 + [0, 0, 0.10]), COVER_HOLD, "lift it, clear of the coater", Q),
        Waypoint(3.0, on_lid(0.10), COVER_HOLD, "over the case's lid, its arrow at the OPEN mark", Q),
        Waypoint(1.5, on_lid(0.03), COVER_HOLD, "down towards the lid", Q),
        Waypoint(1.5, on_lid(0.03), COVER_HOLD, "line up over the lid", Q),
    ]
    _seen["freeze_t"] = _dur(right)  # from here the hands bear on the case: aim at where it was seen last
    right += [Waypoint(2.0, on_lid(BASE_T - 0.001), COVER_HOLD, "lower it over the wall onto the flange", Q)]
    t_seated = _dur(right)
    right.append(Waypoint(3.0, on_lid(BASE_T - 0.001), COVER_HOLD, "wait for the other hand on the flange", Q))
    for f in np.linspace(0.25, 1.0, 4):
        right.append(Waypoint(0.8, on_lid(BASE_T - 0.001), COVER_HOLD, "turn it clockwise to the LOCK mark",
                              float(Q + LOCK * f)))
    right += [Waypoint(0.5, on_lid(BASE_T - 0.001), COVER_HOLD, "steady", float(Q + LOCK))]
    t_turned = _dur(right)
    right += [
        Waypoint(0.6, on_lid(BASE_T - 0.001), COVER_WIDE, "let go", float(Q + LOCK)),
        Waypoint(1.2, on_lid(0.05), COVER_WIDE, "up", float(Q + LOCK)),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]

    # left: fingertips closed, hold the case's lid down by its flange while the bottom is turned
    press = lambda dz: _OnLid(lambda L: tuple(L + [0.0, PRESS_R, BASE_T + TIP_ABOVE_TCP + dz]))  # noqa: E731
    left = [Waypoint(3.0, LEFT_PARK, FINGER_OPEN, "wait")]
    left.append(Waypoint(max(0.3, t_seated - 2.5 - _dur(left)), LEFT_PARK, FINGER_OPEN, "wait"))
    left += [
        Waypoint(2.5, press(0.03), FINGER_CLOSED, "fingers closed, above the case's flange", 0.0),
        Waypoint(1.5, press(0.0), FINGER_CLOSED, "fingertips on the flange", 0.0),
        Waypoint(1.0, press(-PRESS_IN), FINGER_CLOSED, "hold the case's lid down", 0.0),
    ]
    left.append(Waypoint(max(0.3, t_turned + 0.6 - _dur(left)), press(-PRESS_IN), FINGER_CLOSED,
                         "hold it while the bottom is turned", 0.0))
    left += [
        Waypoint(1.0, press(0.04), FINGER_CLOSED, "up", 0.0),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
