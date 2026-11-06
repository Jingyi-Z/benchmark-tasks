"""On the station table: a bare sensor board lying on two soft foam bars on an anti-static mat, a flat flex cable (FPC)
lying on a cable rest beside it, and a small LED work lamp lighting the bench.

The board is green, 50 x 80 mm, with a flip-lock ZIF connector for the FPC on its top near its right end: a black
housing with a slot facing the board's right end, and a brown actuator hinged along the housing's back edge, with a
small upright tab on its front. The actuator lies flat (locked); flipped up about 100 degrees it is open, and the cable
can be slid into the slot. Flipped back down it clamps the cable. Closed, its clamping rib blocks the slot, so a cable
cannot go in. Under the board is an upright electrolytic capacitor, 15 mm tall. The board lies across two grey foam bars,
18 mm tall, on a dark anti-static mat on a 4 cm riser block, its near edge overhanging the riser. The foam is soft:
pushed on while it lies there, the board sinks or slides and the capacitor's can hits the mat, which bends its leads.
So one hand holds the board up off the foam while the other works the connector: unclip, insert, clip.

The cable is a 10 mm wide, 50 mm long amber flat flex cable with a blue stiffener on its contact end, lying flat along y
on a two-post rest, contact end towards the board.

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

FR4 = (0.10, 0.45, 0.20)
BLACK = (0.05, 0.05, 0.06)
ACT_C = (0.45, 0.28, 0.12)  # the actuator's brown plastic
GOLD = (0.90, 0.72, 0.30)
CAP_C = (0.10, 0.20, 0.55)
MAT_C = (0.20, 0.22, 0.24)
FOAM_C = (0.45, 0.47, 0.50)
RISER_C = (0.85, 0.85, 0.82)
PLA = (0.93, 0.45, 0.10)
FPC_C = (0.80, 0.50, 0.10)  # polyimide amber
STIFF_C = (0.15, 0.35, 0.80)  # the cable's blue stiffener
REST_C = (0.62, 0.63, 0.66)
LAMP_C = (0.90, 0.90, 0.92)

# -- the foam bars on the mat (fixed; each bar a sprung slide) -------------------------------------------------------------
PED_H = 0.040
PED_X0, PED_X1 = -0.009, 0.036
MAT_T = 0.003
FOAM_H, FOAM_W = 0.018, 0.012
FOAM_X0, FOAM_X1 = -0.007, 0.032
FOAM_K, FOAM_D = 150.0, 3.0  # N/m: soft foam
FOAM_TRAVEL = 0.010

# -- the board (loose). Frame: origin on the board's underside under the connector's centre, x away from the robots, y to
# their left. -------------------------------------------------------------------------------------------------------------
HX, HY = 0.30, -0.08  # the connector's centre at the start, world
BOARD_HX, BOARD_Y0, BOARD_Y1, BOARD_T = 0.025, -0.020, 0.060, 0.0016
ZT = BOARD_T
BAR_Y = (-0.014, 0.036)
STIFF_Y0, STIFF_T = 0.046, 0.003
GRIP_Y = 0.054
CAP_X, CAP_Y, CAP_R, CAP_H = 0.012, 0.011, 0.004, 0.015
BOARD_MASS = 0.028
MAT_TOP = TOP + PED_H + MAT_T
BOARD_Z0 = MAT_TOP + FOAM_H

# The ZIF connector (heights above the board's top). The slot runs along +y from its mouth at y = -2.5 mm to its back
# at +1.5 mm, 11 mm wide, its floor 0.2 mm and its fixed roof 1.4 mm up over the front 2.5 mm. Behind the roof the
# actuator closes over the slot: a bar hinged at y = +2.5 mm, 1.8 mm up, with a clamping rib under it that comes down to
# 0.95 mm (onto a 0.8 mm cable on the floor) and an upright tab, 4 mm tall, on its front edge.
ZW, ZY0, ZY1 = 0.008, -0.0025, 0.0025  # housing: 16 mm wide, 5 mm deep
SLOT_HW, SLOT_BACK = 0.0055, 0.0015
FLOOR, ROOF0, ROOF1, WALL_H = 0.0002, 0.0014, 0.0018, 0.0018
HINGE_Y, HINGE_Z = 0.0025, 0.0018
ACT_T, RIB_Z, TAB_H, TAB_T = 0.0006, 0.00095, 0.004, 0.001
ACT_OPEN = -math.radians(100.0)  # the actuator flipped up (negative: its front edge rises)
DETENT = -math.radians(35.0)  # past this it snaps open, short of it it springs shut
ACT_K, ACT_D = 0.02, 0.0004  # N m/rad

ACTUATOR = Joint(
    "actuator", kind="hinge", axis="X", anchor=(0.0, HINGE_Y, ZT + HINGE_Z), lower=ACT_OPEN - 0.1, upper=0.02,
    initial=0.0, stiffness=ACT_K, damping=ACT_D, target=0.0, mass=0.0005,
    boxes=[
        ((0.0, HINGE_Y / 2, ZT + HINGE_Z + ACT_T / 2), (2 * ZW - 0.001, HINGE_Y, ACT_T), ACT_C),  # the bar
        ((0.0, (SLOT_BACK + 0.0003) / 2, ZT + (RIB_Z + HINGE_Z) / 2), (2 * SLOT_HW - 0.0004, SLOT_BACK - 0.0003,
                                                                      HINGE_Z - RIB_Z), ACT_C),  # the clamping rib
        ((0.0, TAB_T / 2, ZT + HINGE_Z + ACT_T + TAB_H / 2), (0.010, TAB_T, TAB_H), ACT_C),  # the tab
    ],
)

_ys = (ZY0 + ZY1) / 2
BOARD = part(
    "sensor_board", pos=(HX, HY, BOARD_Z0), mass=BOARD_MASS, friction=0.7,
    boxes=[
        ((0.0, (BOARD_Y0 + BOARD_Y1) / 2, BOARD_T / 2), (2 * BOARD_HX, BOARD_Y1 - BOARD_Y0, BOARD_T), FR4),
        ((-0.008, 0.032, ZT + 0.0005), (0.008, 0.008, 0.001), BLACK),  # a chip
        ((0.0, (STIFF_Y0 + BOARD_Y1) / 2, -STIFF_T / 2), (2 * BOARD_HX, BOARD_Y1 - STIFF_Y0, STIFF_T), PLA),
        # the ZIF housing: floor, side walls, back wall under the hinge, and the fixed roof over the slot's mouth
        ((0.0, _ys, ZT + FLOOR / 2), (2 * ZW, ZY1 - ZY0, FLOOR), BLACK),
        *[((s * (SLOT_HW + ZW) / 2, _ys, ZT + WALL_H / 2), (ZW - SLOT_HW, ZY1 - ZY0, WALL_H), BLACK) for s in (1, -1)],
        ((0.0, (SLOT_BACK + ZY1) / 2, ZT + WALL_H / 2), (2 * SLOT_HW, ZY1 - SLOT_BACK, WALL_H), BLACK),
        ((0.0, ZY0 / 2, ZT + (ROOF0 + ROOF1) / 2), (2 * ZW, -ZY0, ROOF1 - ROOF0), BLACK),
        # the contacts' gold tails in front of the housing
        *[((x, ZY1 + 0.0008, ZT + 0.0001), (0.0003, 0.0016, 0.0002), GOLD) for x in np.linspace(-0.0045, 0.0045, 10)],
    ],
    cylinders=[((CAP_X, CAP_Y, -CAP_H / 2), CAP_R, CAP_H, CAP_C)],
    joints=[ACTUATOR],
)

_fx, _fl = (FOAM_X0 + FOAM_X1) / 2, FOAM_X1 - FOAM_X0
_bars = [Joint(f"foam{i}", kind="slide", axis="Z", anchor=(_fx, y, PED_H + MAT_T + FOAM_H), lower=-FOAM_TRAVEL,
               upper=0.0, initial=0.0, stiffness=FOAM_K, damping=FOAM_D, target=0.0, mass=0.003,
               boxes=[((_fx, y, PED_H + MAT_T + FOAM_H / 2), (_fl, FOAM_W, FOAM_H), FOAM_C)])
         for i, y in enumerate(BAR_Y)]
_px, _pl, _py, _pw = (PED_X0 + PED_X1) / 2, PED_X1 - PED_X0, (BOARD_Y0 + BOARD_Y1) / 2, BOARD_Y1 - BOARD_Y0 + 0.03
MAT_PART = part("mat", pos=(HX, HY, TOP), fixed=True, friction=0.8,
                boxes=[((_px, _py, PED_H / 2), (_pl, _pw, PED_H), RISER_C),
                       ((_px, _py, PED_H + MAT_T / 2), (_pl, _pw, MAT_T), MAT_C)],
                joints=_bars)

# -- the cable (loose). Frame: origin at the middle of its contact end, on its underside; it runs along -y. -----------------
FPC_HW, FPC_L, FPC_T = 0.005, 0.050, 0.0008
FPC_STIFF = 0.008  # the blue stiffener's length on the contact end
RX, RY, REST_H = 0.24, -0.16, 0.030  # the cable's contact end at the start, and the rest's height
FPC = part(
    "fpc", pos=(RX, RY, TOP + REST_H), mass=0.0012, friction=0.6,
    boxes=[
        ((0.0, -FPC_L / 2, FPC_T / 2), (2 * FPC_HW, FPC_L, FPC_T), FPC_C),
        ((0.0, -FPC_STIFF / 2, FPC_T + 0.0001), (2 * FPC_HW, FPC_STIFF, 0.0002), STIFF_C),
        *[((x, -0.0015, FPC_T + 0.0002), (0.0004, 0.003, 0.0001), GOLD) for x in np.linspace(-0.0045, 0.0045, 10)],
    ],
)
# the rest: a post under the contact end and one under the tail, with the middle 25 mm free for the fingers
REST = part("cable_rest", pos=(RX, RY, TOP), fixed=True, friction=0.6,
            boxes=[((0.0, -0.010, REST_H / 2), (0.016, 0.020, REST_H), REST_C),
                   ((0.0, -0.048, REST_H / 2), (0.016, 0.004, REST_H), REST_C)])

# -- the work lamp (fixed): a weighted base, a post and a head; the light itself is added to the stage at reset ----------
LAMP_XY, LAMP_HEAD_Z = (0.47, -0.17), 0.42
LAMP = part("work_lamp", pos=(*LAMP_XY, TOP), fixed=True, friction=0.6,
            boxes=[((0.0, 0.0, 0.006), (0.10, 0.10, 0.012), LAMP_C),
                   ((0.0, 0.0, LAMP_HEAD_Z / 2), (0.016, 0.016, LAMP_HEAD_Z), LAMP_C),
                   ((-0.04, 0.0, LAMP_HEAD_Z + 0.01), (0.10, 0.05, 0.02), LAMP_C)])
LIGHT_POS = (LAMP_XY[0] - 0.06, LAMP_XY[1], TOP + LAMP_HEAD_Z - 0.005)
LIGHT_INTENSITY, LIGHT_RADIUS = 2.0e5, 0.02

OBJECTS = [BOARD, MAT_PART, FPC, REST, LAMP]

# -- the cable in the connector -------------------------------------------------------------------------------------------
IN_X = 0.001  # inserted: within 1 mm of the slot's centre line,
IN_YAW = math.radians(10.0)  # turned no more than 10 degrees from it,
IN_DEPTH = SLOT_BACK - 0.0005  # its end within 0.5 mm of the slot's back,
IN_Z = 0.0008  # and its underside within 0.8 mm of the board's top
ENTER = ZY0 + 0.002  # its end 2 mm into the slot counts as entering
CLOSED = math.radians(6.0)
OPENED = math.radians(60.0)
FLIP_MAX = 30.0
CAP_TOUCH = 0.0005
RELEASED = 0.030


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _yaw(rot) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def cable_in_slot(board, fpc):
    """(the cable end's x and y in the board's frame, its underside's height above the board's top, its turn from the
    slot's axis in rad)."""
    p = board.to_local(fpc.pos)
    rel = board.rotation().T @ fpc.rotation()
    return float(p[0]), float(p[1]), float(p[2] - ZT), _wrap(_yaw(rel))


def inserted(board, fpc) -> bool:
    x, y, z, turn = cable_in_slot(board, fpc)
    return abs(x) <= IN_X and y >= IN_DEPTH and -0.0003 <= z <= IN_Z and abs(turn) <= IN_YAW


def cap_clearance(board) -> float:
    rot = board.rotation()
    axis = rot[:, 2]
    centre = board.to_world((CAP_X, CAP_Y, -CAP_H))
    down = np.array([0.0, 0.0, -1.0])
    radial = down - axis * float(down @ axis)
    n = float(np.linalg.norm(radial))
    low = centre + (CAP_R * radial / n if n > 1e-9 else 0.0)
    return float(low[2] - MAT_TOP)


# -- the actuator's detent and the trial's record ------------------------------------------------------------------------
_state = {"opened": False, "forced": False, "pressed_cap": False, "hands_far": True, "lamp": False, "was_in": False}


@on_reset
def _reset_state(env):
    _state.update(opened=False, forced=False, pressed_cap=False, hands_far=True, was_in=False)
    if not _state["lamp"]:
        _add_lamp(env)


def _add_lamp(env):
    """The work lamp's light: a small sphere light in the lamp's head, over the bench."""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        o = env.unwrapped.scene.env_origins[0].cpu().numpy()
        light = UsdLux.SphereLight.Define(stage, "/World/task_work_lamp")
        light.CreateIntensityAttr(LIGHT_INTENSITY)
        light.CreateRadiusAttr(LIGHT_RADIUS)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(*(np.asarray(LIGHT_POS) + o).tolist()))
        _state["lamp"] = True
    except Exception as e:  # rendering only: never fail a trial over the lamp
        print("work lamp not added:", e)


@on_step
def _watch(env):
    """The actuator's detent (snaps open past 35 degrees, springs shut short of it); whether it has been opened; whether
    the cable was pushed into the slot while it was shut (forced); whether the capacitor has ever been pressed onto the
    mat; whether both hands are clear of the connector."""
    import torch

    import labkit
    from yam_arena import runtime as _rt

    board = labkit._part_state(env, labkit._SPECS["sensor_board"])
    fpc = labkit._part_state(env, labkit._SPECS["fpc"])
    a = board.joints["actuator"]
    if -a >= OPENED:
        _state["opened"] = True
    x, y, z, turn = cable_in_slot(board, fpc)
    now_in = y >= ENTER and abs(x) <= SLOT_HW and z <= ROOF0
    if now_in and not _state["was_in"] and -a < -DETENT:  # it entered the slot while the actuator was shut
        _state["forced"] = True
    _state["was_in"] = now_in
    if cap_clearance(board) <= CAP_TOUCH:
        _state["pressed_cap"] = True
    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    zif = board.to_world((0.0, 0.0, ZT))
    _state["hands_far"] = all(float(np.linalg.norm(_rt.tcp_world(b.scene[n]) - o - zif)) >= RELEASED
                              for n in ("left_robot", "right_robot"))
    asset = b.scene["sensor_board"]
    ids = [list(asset.joint_names).index(ACTUATOR.joint_name)]
    target = ACT_OPEN if a <= DETENT else 0.0
    asset.set_joint_position_target(torch.tensor([[target]], device=asset.device), joint_ids=ids)


# -- scoring -----------------------------------------------------------------------------------------------------------
SUCCESS_HOLD_S = 10.0


def score(objects):
    """0 if the capacitor under the board was ever pressed onto the mat (its can within 0.5 mm of it), if the cable was
    ever pushed 2 mm into the slot while the actuator was shut (forced), or if the board ends tilted more than 30 degrees
    from component side up. Otherwise: 1.0 if the cable is fully inserted (its end within 0.5 mm of the slot's back,
    within 1 mm of its centre line, turned no more than 10 degrees, lying on the slot's floor), the actuator is shut
    (within 6 degrees of flat) and both hands are at least 3 cm from the connector; 0.5 if the cable is fully inserted
    but the actuator is not shut or a hand is still at the connector; 0.25 if the actuator has been flipped open (60
    degrees or more) at some point but the cable is not fully inserted; 0 otherwise."""
    board, fpc = objects["sensor_board"], objects["fpc"]
    if _state["pressed_cap"] or _state["forced"] or board.tilt_deg() > FLIP_MAX:
        return 0.0
    if inserted(board, fpc):
        shut = abs(board.joints["actuator"]) <= CLOSED
        return 1.0 if shut and _state["hands_far"] else 0.5
    return 0.25 if _state["opened"] else 0.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    x, y, _, _ = cable_in_slot(objects["sensor_board"], objects["fpc"])
    return abs(x) + max(0.0, SLOT_BACK - y) + abs(objects["sensor_board"].joints["actuator"])


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Left arm: level, pointing away from the robots, fingers closing up and down; slides its lower finger under the board's
# overhanging left end, pinches the board and its stiffener, lifts it 7 cm off the foam and holds it level until the
# right hand is done, then sets it back and parks. Right arm, once the board is up, aiming at the connector where the
# board is now (read from the board's pose, as a camera sees it): with its fingers closed, pushes the actuator's tab back
# until it flips open; picks the cable off its rest by its two edges, 35 mm from its end; carries it level to the slot
# and slides it straight in to the back; lets go and lifts off; then, fingers closed, pushes the actuator's tab forward
# from behind until it snaps shut; and parks.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
LIFT = 0.070
BOARD_WIDE, BOARD_HOLD = -0.009, FINGER_CLOSED
PINCH_IN = 0.013
FPC_WIDE, FPC_HOLD = -0.008, -0.0015  # pads 16 mm apart round the 10 mm cable, then a firm pinch
FPC_PINCH = 0.035  # the right hand pinches the cable this far from its contact end
TIPS_BELOW = 0.0025  # the pad tips this far below the cable's underside
FLAT_YAW = math.pi / 2  # fingers closing along x

_orig_grip = _demo.grip_rotation
_down = getattr(_orig_grip, "_base", _orig_grip)
if not getattr(_orig_grip, "_rot_duck", False):
    def _grip_rotation(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _down(yaw)

    _grip_rotation._rot_duck = True
    _grip_rotation._base = _down
    _demo.grip_rotation = _grip_rotation


class Orient(float):
    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


POINT_X = Orient(axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -math.pi / 2) @ _down(math.pi / 2))

_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7

# What the demo sees: the board's pose (the cameras see it), read each step.
_seen = {"zif": None, "psi": 0.0}


@on_reset
def _reset_seen(env):
    _seen.update(zif=None, psi=0.0)


@on_step
def _look(env):
    import labkit

    board = labkit._part_state(env, labkit._SPECS["sensor_board"])
    _seen.update(zif=np.asarray(board.to_world((0.0, 0.0, ZT)), float), psi=_yaw(board.rotation()))


HELD = np.array([HX, HY, BOARD_Z0 + BOARD_T + LIFT])  # the connector's base while held (nominal)


class _AtZif:
    """A TCP point given in the connector's frame (x, y along the board, z above the board's top), where the board is
    now."""

    def __init__(self, x, y, z):
        self.p = np.array([x, y, z])

    def __array__(self, dtype=None, copy=None):
        base = _seen["zif"] if _seen["zif"] is not None else HELD
        c, s = math.cos(_seen["psi"]), math.sin(_seen["psi"])
        x, y, z = self.p
        return np.asarray((base[0] + c * x - s * y, base[1] + s * x + c * y, base[2] + z), dtype=dtype or float)


class _ZifYaw(float):
    def __new__(cls):
        return float.__new__(cls, FLAT_YAW)

    @property
    def rot(self):
        return _down(FLAT_YAW + _seen["psi"])


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _seen.update(zif=None, psi=0.0)
    gx = HX - BOARD_HX + PINCH_IN - TIP_ABOVE_TCP
    gy = HY + GRIP_Y
    gz = BOARD_Z0 - 0.001 + (BOARD_T - STIFF_T) / 2
    out = gx - 0.06
    left = [
        Waypoint(3.0, (out, gy, gz + 0.03), BOARD_WIDE, "level, pointing at the board's overhang", POINT_X),
        Waypoint(1.5, (out, gy, gz), BOARD_WIDE, "fingers either side of the board's edge", POINT_X),
        Waypoint(2.0, (gx, gy, gz), BOARD_WIDE, "slide in: lower finger under the overhang", POINT_X),
        Waypoint(0.8, (gx, gy, gz), BOARD_HOLD, "pinch the board by its faces", POINT_X),
        Waypoint(2.0, (gx, gy, gz + LIFT), BOARD_HOLD, "lift it off the foam, level", POINT_X),
        Waypoint(1.0, (gx, gy, gz + LIFT), BOARD_HOLD, "hold it up", POINT_X),
    ]
    t_up = _dur(left)
    yaw = _ZifYaw()
    shut = FINGER_CLOSED
    tip = TIP_ABOVE_TCP
    # 1. unclip: closed fingertips in front of the tab, 3 mm up it, push it back 4 mm until it flips open
    tz = HINGE_Z + ACT_T + 0.0025 + tip  # the pad tips 2.5 mm up the tab
    right = [
        Waypoint(2.5, (0.22, -0.16, TOP + 0.16), shut, "fingers closed, up", FLAT_YAW),
    ]
    right.append(Waypoint(max(0.5, t_up - _dur(right)), (0.22, -0.16, TOP + 0.16), shut, "wait for the board", FLAT_YAW))
    right += [
        Waypoint(2.0, _AtZif(0.0, -0.006, 0.03), shut, "above the connector, in front of the actuator's tab", yaw),
        Waypoint(1.5, _AtZif(0.0, -0.006, tz), shut, "fingertips down in front of the tab", yaw),
        Waypoint(2.0, _AtZif(0.0, 0.001, tz), shut, "push the tab back until the actuator flips open", yaw),
        Waypoint(1.0, _AtZif(0.0, 0.001, 0.03), shut, "lift off", yaw),
    ]
    # 2. pick the cable off its rest by its edges
    px, py = RX, RY - FPC_PINCH
    pz = TOP + REST_H - TIPS_BELOW + tip
    right += [
        Waypoint(2.5, (px, py, pz + 0.04), FPC_WIDE, "above the cable, between its rests", FLAT_YAW),
        Waypoint(1.5, (px, py, pz), FPC_WIDE, "pads either side of the cable", FLAT_YAW),
        Waypoint(0.8, (px, py, pz), FPC_HOLD, "pinch it by its edges", FLAT_YAW),
        Waypoint(1.2, (px, py, pz + 0.04), FPC_HOLD, "lift it", FLAT_YAW),
    ]
    # 3. insert: level, its end 10 mm in front of the slot's mouth, then straight in to the back
    cz = 0.0004 - TIPS_BELOW + tip  # its underside 0.4 mm above the board's top: in the middle of the slot's mouth
    right += [
        Waypoint(3.0, _AtZif(0.0, ZY0 - 0.010 - FPC_PINCH, 0.03), FPC_HOLD, "carry it over, in line with the slot", yaw),
        Waypoint(1.5, _AtZif(0.0, ZY0 - 0.010 - FPC_PINCH, cz), FPC_HOLD, "down level with the slot", yaw),
        Waypoint(1.0, _AtZif(0.0, ZY0 - 0.010 - FPC_PINCH, cz), FPC_HOLD, "line up", yaw),
        Waypoint(3.0, _AtZif(0.0, SLOT_BACK + 0.0005 - FPC_PINCH, cz), FPC_HOLD, "slide it straight in to the back", yaw),
        Waypoint(0.5, _AtZif(0.0, SLOT_BACK + 0.0005 - FPC_PINCH, cz), FPC_HOLD, "settle", yaw),
        Waypoint(0.8, _AtZif(0.0, SLOT_BACK + 0.0005 - FPC_PINCH, cz), FPC_WIDE, "let go", yaw),
        Waypoint(1.2, _AtZif(0.0, SLOT_BACK + 0.0005 - FPC_PINCH, 0.03), FPC_WIDE, "up off the cable", yaw),
    ]
    # 4. clip: closed fingertips behind the open actuator's tab, push it forward until it snaps shut
    bz = HINGE_Z + 0.0015 + tip  # the open actuator's tab sticks back, 0.7 to 2.4 mm above the hinge
    right += [
        Waypoint(2.0, _AtZif(0.0, 0.014, 0.03), shut, "fingers closed, behind the open actuator", yaw),
        Waypoint(1.5, _AtZif(0.0, 0.014, bz), shut, "fingertips down behind its tab", yaw),
        Waypoint(2.0, _AtZif(0.0, 0.006, bz), shut, "push it forward until it snaps shut", yaw),
        Waypoint(1.0, _AtZif(0.0, 0.006, 0.03), shut, "lift off", yaw),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_free = _dur(right) - 2.0
    left.append(Waypoint(max(0.5, t_free - _dur(left)), (gx, gy, gz + LIFT), BOARD_HOLD,
                         "hold the board up while the cable is connected", POINT_X))
    left += [
        Waypoint(2.0, (gx, gy, gz + 0.004), BOARD_HOLD, "set it back on the foam, level", POINT_X),
        Waypoint(0.8, (gx, gy, gz), BOARD_WIDE, "let go", POINT_X),
        Waypoint(1.5, (out, gy, gz), BOARD_WIDE, "slide out", POINT_X),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 1.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    return {"left": left, "right": right}
