"""On the station table: a bare sensor board lying on two soft foam bars on an anti-static mat, and the black socket
connector of a ribbon cable standing on a tray beside it.

The board is green, 50 x 80 mm, with a black shrouded 2 x 5 pin header on its top near its right end: a box with walls
round ten gold pins. The header's wall facing the robots has a 5 mm notch in its middle: the key slot. Under the board
are the solder tails of the header's pins and an upright electrolytic capacitor, 15 mm tall, next to the header. The
board lies across two grey foam bars, 18 mm tall, on a dark anti-static mat on a 4 cm riser block; its near edge
overhangs the riser. The foam is soft: pressed down on the header with the force the connector takes, the board sinks into the bars
and the capacitor's can hits the mat, which bends its leads. So one hand holds the board up off the foam while the
other plugs the cable in.

The connector is a black block, 16 mm long, 8 mm wide and 11 mm tall, with a raised key on the middle of one long side
and a short stub of grey ribbon cable, red stripe on one edge, standing up out of its top. It stands on the tray with
its length along y and its key towards the robots. Plugging it in means: line it up over the header with its key at the
key slot, lower it in and press it down until it gives and seats, about 1.2 N. The other way round, the key lands on the
top of the header's wall and it cannot go in.

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
GOLD = (0.90, 0.72, 0.30)
SOLDER = (0.75, 0.75, 0.78)
CAP_C = (0.10, 0.20, 0.55)  # the capacitor's blue sleeve
RIBBON_C = (0.72, 0.72, 0.74)
RED = (0.80, 0.10, 0.08)
MAT_C = (0.20, 0.22, 0.24)
FOAM_C = (0.45, 0.47, 0.50)
RISER_C = (0.85, 0.85, 0.82)
PLA = (0.93, 0.45, 0.10)
TRAY_C = (0.62, 0.63, 0.66)

# -- the foam bars on the mat (fixed mat; each bar a sprung slide) -------------------------------------------------------
PED_H = 0.040  # the riser block the mat lies on, 4 cm tall
PED_X0, PED_X1 = -0.009, 0.036  # its extent along x in the board's frame: the board's near edge overhangs it 16 mm
MAT_T = 0.003  # anti-static mat, 3 mm thick, on the riser
FOAM_H, FOAM_W = 0.018, 0.012  # each bar 18 mm tall, 12 mm wide, along x across the board
FOAM_X0, FOAM_X1 = -0.007, 0.032
FOAM_K, FOAM_D = 150.0, 3.0  # N/m: soft foam, 1 N on a bar sinks it 6.7 mm
FOAM_TRAVEL = 0.010

# -- the board (loose). Frame: origin on the board's underside under the header's centre, x away from the robots, y to
# their left. ------------------------------------------------------------------------------------------------------------
HX, HY = 0.30, -0.08  # the header's centre at the start, world
BOARD_HX, BOARD_Y0, BOARD_Y1, BOARD_T = 0.025, -0.020, 0.060, 0.0016  # 50 x 80 mm; the header 20 mm from its right end
ZT = BOARD_T  # the board's top, in its frame
BAR_Y = (-0.014, 0.036)  # the foam bars' centres, board frame
STIFF_Y0, STIFF_T = 0.046, 0.003  # a printed stiffener, 3 mm thick, screwed under the board's left end (y 46 to 60 mm)
GRIP_Y = 0.054  # the left hand pinches the board and its stiffener here, at the near edge, over no riser
CAP_X, CAP_Y, CAP_R, CAP_H = 0.012, 0.011, 0.004, 0.015  # the capacitor under the board, beside the header
BOARD_MASS = 0.028
MAT_TOP = TOP + PED_H + MAT_T  # the mat's top, world
BOARD_Z0 = MAT_TOP + FOAM_H  # the board's underside on the foam, world (before it sinks under its own weight)

# The connector (below) is 16 x 8 mm; the header's cavity leaves 0.6 mm round it, inside walls 1.2 mm thick.
CONN_L, CONN_W, CONN_H = 0.016, 0.008, 0.011
CLEAR, WALL = 0.0006, 0.0012
CAV_X, CAV_Y = CONN_L + 2 * CLEAR, CONN_W + 2 * CLEAR  # 17.2 x 9.2 mm, long along x
OUT_X, OUT_Y = CAV_X + 2 * WALL, CAV_Y + 2 * WALL
FLOOR, WALL_TOP = 0.001, 0.007  # the header's floor and the top of its walls, above the board's top
SLOT_W = 0.005  # the key slot through the middle of the header's wall facing the robots (-x)
_wall_h = WALL_TOP - FLOOR

# The press fit: the pins are not contacts. The connector rests on a sprung contact plate inside the header, 4 mm above
# the board. The plate gives way once the connector, in the header the right way round, pushes it down 1 mm (1.2 N); its
# target then drops to the bottom of its 2.5 mm travel and stays there, and the connector follows to its seat.
SEAT_GAP = 0.0015
TRAVEL, TRIGGER = 0.0025, 0.001
PLATE_K, PLATE_C = 1200.0, 4.0  # 1 mm of give, the trigger, takes 1.2 N
PIN, PIN_PROUD, PLATE_T = 0.00064, 0.0001, 0.0004
PIN_TOP0 = ZT + SEAT_GAP + TRAVEL
_plate_top = PIN_TOP0 - PIN_PROUD
_pin_h = PIN_PROUD + 0.0002
# the header lies with its length along y here (the board's long axis): pins in 5 rows along y, 2 along x
PINS = [(x, y) for y in (-0.00508, -0.00254, 0.0, 0.00254, 0.00508) for x in (-0.00127, 0.00127)]
CAV_LX, CAV_LY = CAV_Y, CAV_X  # the cavity: 9.2 mm along x, 17.2 mm along y
OUT_LX, OUT_LY = OUT_Y, OUT_X
_seg = (CAV_LY / 2 - SLOT_W / 2)

PLATE = Joint(
    "contacts", kind="slide", axis="Z", anchor=(0.0, 0.0, _plate_top - PLATE_T / 2), lower=-TRAVEL, upper=0.0,
    initial=0.0, stiffness=PLATE_K, damping=PLATE_C, target=0.0, mass=0.002,
    boxes=[((0.0, 0.0, _plate_top - PLATE_T / 2), (CAV_LX - 0.0004, CAV_LY - 0.0004, PLATE_T), BLACK)]
    + [((x, y, PIN_TOP0 - _pin_h / 2), (PIN, PIN, _pin_h), GOLD) for x, y in PINS],
)

_wz = ZT + FLOOR + _wall_h / 2
BOARD = part(
    "sensor_board", pos=(HX, HY, BOARD_Z0), mass=BOARD_MASS, friction=0.7,
    boxes=[
        ((0.0, (BOARD_Y0 + BOARD_Y1) / 2, BOARD_T / 2), (2 * BOARD_HX, BOARD_Y1 - BOARD_Y0, BOARD_T), FR4),
        ((-0.008, 0.032, ZT + 0.0005), (0.008, 0.008, 0.001), BLACK),  # a chip
        # the printed stiffener under the left end
        ((0.0, (STIFF_Y0 + BOARD_Y1) / 2, -STIFF_T / 2), (2 * BOARD_HX, BOARD_Y1 - STIFF_Y0, STIFF_T), PLA),
        # the header: floor, the two end walls, the far wall and the near wall in two pieces either side of the slot
        ((0.0, 0.0, ZT + FLOOR / 2), (OUT_LX, OUT_LY, FLOOR), BLACK),
        ((0.0, CAV_LY / 2 + WALL / 2, _wz), (OUT_LX, WALL, _wall_h), BLACK),
        ((0.0, -CAV_LY / 2 - WALL / 2, _wz), (OUT_LX, WALL, _wall_h), BLACK),
        ((CAV_LX / 2 + WALL / 2, 0.0, _wz), (WALL, CAV_LY, _wall_h), BLACK),
        ((-CAV_LX / 2 - WALL / 2, SLOT_W / 2 + _seg / 2, _wz), (WALL, _seg, _wall_h), BLACK),
        ((-CAV_LX / 2 - WALL / 2, -SLOT_W / 2 - _seg / 2, _wz), (WALL, _seg, _wall_h), BLACK),
        # the pins' solder tails under the board
        *[((x, y, -0.0012), (0.0012, 0.0012, 0.0024), SOLDER) for x, y in PINS],
    ],
    cylinders=[((CAP_X, CAP_Y, -CAP_H / 2), CAP_R, CAP_H, CAP_C)],
    joints=[PLATE],
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

# -- the connector (loose). Frame: origin at the middle of its underside; its length along x, its key on its -y side. -
KEY_W, KEY_P = 0.0035, 0.002
RIBBON_W, RIBBON_T, RIBBON_H = 0.0127, 0.001, 0.012
TRAY_XY, TRAY = (0.22, -0.20), (0.060, 0.060, 0.004)
CONN_YAW0 = -math.pi / 2  # on the tray: its length along y, its key towards the robots (-x)
_rz = CONN_H + RIBBON_H / 2
CONNECTOR = part(
    "connector", pos=(*TRAY_XY, TOP + TRAY[2]), yaw=CONN_YAW0, mass=0.004, friction=0.8,
    boxes=[
        ((0.0, 0.0, CONN_H / 2), (CONN_L, CONN_W, CONN_H), BLACK),
        ((0.0, -CONN_W / 2 - KEY_P / 2, CONN_H / 2), (KEY_W, KEY_P, CONN_H), BLACK),
        ((0.0, 0.0, _rz), (RIBBON_W, RIBBON_T, RIBBON_H), RIBBON_C),
        ((-RIBBON_W / 2 + 0.0005, 0.0, _rz), (0.001, RIBBON_T + 0.0002, RIBBON_H), RED),
    ],
)
TRAY_PART = part("tray", pos=(*TRAY_XY, TOP), fixed=True, friction=0.6,
                 boxes=[((0.0, 0.0, TRAY[2] / 2), TRAY, TRAY_C)])

OBJECTS = [BOARD, MAT_PART, CONNECTOR, TRAY_PART]

# -- where the connector is in the header -----------------------------------------------------------------------------
IN_R = 0.0015
IN_TILT = 10.0
IN_GAP = WALL_TOP - 0.001
SEATED_GAP = SEAT_GAP + 0.0005
KEY_TOL = math.radians(10.0)
KEY_YAW = -math.pi / 2  # in the board's frame the keyed connector lies along y with its key towards -x
FLIP_MAX = 30.0  # the board component side up: tilted no more than 30 degrees
CAP_TOUCH = 0.0005  # the capacitor's can within 0.5 mm of the mat counts as pressed onto it
RELEASED = 0.030  # both hands' grip points at least 3 cm from the connector at the end


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _yaw(rot) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def connector_in_header(board, conn):
    """(offset of the connector's underside centre from the header's axis in m, its underside's height above the
    board's top in m, its tilt from the header's axis in degrees, its turn from the keyed orientation in rad), in the
    board's frame."""
    p = board.to_local(conn.pos)
    rel = board.rotation().T @ conn.rotation()
    tilt = float(np.degrees(np.arccos(np.clip(rel[2, 2], -1.0, 1.0))))
    return float(math.hypot(p[0], p[1])), float(p[2] - ZT), tilt, _wrap(_yaw(rel) - KEY_YAW)


def _in(r, gap, tilt) -> bool:
    return r <= IN_R and gap <= IN_GAP and tilt <= IN_TILT


def cap_clearance(board) -> float:
    """Height of the capacitor can's lowest point above the mat's top (m)."""
    rot = board.rotation()
    axis = rot[:, 2]
    centre = board.to_world((CAP_X, CAP_Y, -CAP_H))  # the middle of the can's end
    # the end's lowest rim point
    down = np.array([0.0, 0.0, -1.0])
    radial = down - axis * float(down @ axis)
    n = float(np.linalg.norm(radial))
    low = centre + (CAP_R * radial / n if n > 1e-9 else 0.0)
    return float(low[2] - MAT_TOP)


# -- the press fit and the trial's record ---------------------------------------------------------------------------------
_state = {"latched": False, "pressed_cap": False, "hands_far": True}


@on_reset
def _reset_fit(env):
    _state.update(latched=False, pressed_cap=False, hands_far=True)


@on_step
def _fit(env):
    """The contact plate's drive target (sprung up until the keyed connector has pushed it down 1 mm, then at the bottom
    of its travel); whether the capacitor has ever been pressed onto the mat; whether both hands are clear of the
    connector."""
    import torch

    import labkit
    from yam_arena import runtime as _rt

    board = labkit._part_state(env, labkit._SPECS["sensor_board"])
    conn = labkit._part_state(env, labkit._SPECS["connector"])
    r, gap, tilt, turn = connector_in_header(board, conn)
    if _in(r, gap, tilt) and abs(turn) <= KEY_TOL and board.joints["contacts"] <= -TRIGGER:
        _state["latched"] = True
    if cap_clearance(board) <= CAP_TOUCH:
        _state["pressed_cap"] = True
    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    c = np.asarray(conn.pos, float)
    far = all(float(np.linalg.norm(_rt.tcp_world(b.scene[n]) - o - c)) >= RELEASED for n in ("left_robot", "right_robot"))
    _state["hands_far"] = far
    asset = b.scene["sensor_board"]
    ids = [list(asset.joint_names).index(PLATE.joint_name)]
    target = -TRAVEL if _state["latched"] else 0.0
    asset.set_joint_position_target(torch.tensor([[target]], device=asset.device), joint_ids=ids)


# -- scoring -----------------------------------------------------------------------------------------------------------
SUCCESS_HOLD_S = 6.0


def score(objects):
    """0 if the capacitor under the board was ever pressed onto the mat (its can within 0.5 mm of the mat's top at any
    step: that bends its leads), or the board ends tilted more than 30 degrees from component side up. Otherwise 0
    unless the connector is in the header (underside centre within 1.5 mm of the header's axis, tilted no more than 10
    degrees from it, underside no more than 6 mm above the board's top). 0.25 in the header but not seated, or seated but
    still touched by a hand (a hand's grip point within 3 cm of it). 1.0 seated (underside within 0.5 mm of its seat,
    2 mm above the board's top) with its key in the header's slot and both hands clear of it."""
    board, conn = objects["sensor_board"], objects["connector"]
    if _state["pressed_cap"] or board.tilt_deg() > FLIP_MAX:
        return 0.0
    r, gap, tilt, turn = connector_in_header(board, conn)
    if not _in(r, gap, tilt):
        return 0.0
    if gap <= SEATED_GAP and abs(turn) <= KEY_TOL and _state["hands_far"]:
        return 1.0
    return 0.25


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    r, gap, _, _ = connector_in_header(objects["sensor_board"], objects["connector"])
    return r + max(0.0, gap - SEAT_GAP)


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Left arm: level, pointing away from the robots, fingers closing up and down; slides its lower finger under the board's
# overhanging left end, pinches the board there by its faces, lifts it 7 cm off the foam and holds it level. Right arm:
# meanwhile picks the connector by its two ends and turns it so its key faces the header's slot; once the board is up it
# aims at the header where the board is now (read from the board's pose, as a camera sees it), lowers the connector in
# and presses it slowly until the connector has sunk to its seat (read from the connector's height on the board), eases
# off, lets go and lifts off. Then the left hand sets the board back on the foam, slides out and parks.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_UP = 0.007
CONN_WIDE = -0.012
CONN_HOLD = -0.005
PICK_Z = TOP + TRAY[2] + GRIP_UP + TIP_ABOVE_TCP
PICK_HOVER = TOP + 0.07
LIFT = 0.070  # the board held this far above where it lay
BOARD_WIDE, BOARD_HOLD = -0.009, FINGER_CLOSED  # pads 18 mm apart round the board's 4.6 mm end, then closed on it
PINCH_IN = 0.013  # the pads' tips this far in from the board's near edge (it overhangs the riser 16 mm)
C_RIGHT = np.array([0.94, 0.61, 2.20]) / 1000.0
PRESS_RATE = 0.002  # m/s
PRESS_DEPTH = 0.010  # commanded at most this far below the seat
PRESS_F_MAX = 4.0

_orig_grip = _demo.grip_rotation
_down = getattr(_orig_grip, "_base", _orig_grip)  # the plain top-down grip by yaw
if not getattr(_orig_grip, "_rot_duck", False):
    def _grip_rotation(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _down(yaw)

    _grip_rotation._rot_duck = True
    _grip_rotation._base = _down
    _demo.grip_rotation = _grip_rotation


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


# level, pointing +x (away from the robots), fingers closing along z: a yaw of 90 degrees (fingers along x), then turned
# a quarter turn about y so the fingers point +x and close up and down
POINT_X = Orient(axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -math.pi / 2) @ _down(math.pi / 2))

_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


def _clock() -> float:
    return float(getattr(_demo, "_plan_clock", 0.0))


# What the demo sees: the board's pose and the connector's height on it (both visible to the cameras), read each step.
_seen = {"header": None, "psi": 0.0, "gap": None, "z": None}
_plan = {"t_press0": 1e9, "t_press1": 1e9, "t_seat": None, "z_seat": None}


@on_reset
def _reset_seen(env):
    _seen.update(header=None, psi=0.0, gap=None, z=None)
    _plan.update(t_seat=None, z_seat=None)


@on_step
def _look(env):
    import labkit

    board = labkit._part_state(env, labkit._SPECS["sensor_board"])
    conn = labkit._part_state(env, labkit._SPECS["connector"])
    _, gap, _, _ = connector_in_header(board, conn)
    top = board.to_world((0.0, 0.0, ZT))
    _seen.update(header=np.asarray(top[:2], float), psi=_yaw(board.rotation()), gap=gap, z=float(top[2]))


def conn_z(board_top: float, underside: float) -> float:
    return board_top + underside + GRIP_UP + TIP_ABOVE_TCP


HELD_TOP = BOARD_Z0 + BOARD_T + LIFT  # the board's top while held (nominal)


def _held_top() -> float:
    return _seen["z"] if _seen["z"] is not None else HELD_TOP


def _press_z() -> float:
    """Down from the contact plate at PRESS_RATE until the connector is seen at its seat, then back to a light push."""
    t = min(_clock(), _plan["t_press1"])
    rest = conn_z(_held_top(), SEAT_GAP + TRAVEL + 0.0003)
    seat = conn_z(_held_top(), SEAT_GAP)
    if _plan["t_seat"] is None:
        z = max(seat - PRESS_DEPTH, rest - PRESS_RATE * max(0.0, t - _plan["t_press0"]))
        if _seen["gap"] is not None and _seen["gap"] <= SEAT_GAP + 0.0003:
            _plan["t_seat"], _plan["z_seat"] = t, z
        return z
    goal = seat - 0.0015
    d = goal - _plan["z_seat"]
    return _plan["z_seat"] + math.copysign(min(abs(d), PRESS_RATE * max(0.0, t - _plan["t_seat"])), d)


class _AtHeader:
    """The right hand's TCP over the header where the board is now, at ``underside`` above the board's top (a number) or
    pressing (``"press"``)."""

    def __init__(self, underside):
        self.u = underside

    def __array__(self, dtype=None, copy=None):
        xy = _seen["header"] if _seen["header"] is not None else np.array([HX, HY])
        if self.u == "press":
            z = _press_z()
            now = conn_z(_held_top(), _seen["gap"]) if _seen["gap"] is not None else z
            f = float(np.clip((now - z) / C_RIGHT[2], 0.0, PRESS_F_MAX))
            xy = xy - f * C_RIGHT[:2]
        else:
            z = conn_z(_held_top(), self.u)
        return np.asarray((xy[0], xy[1], z), dtype=dtype or float)


class _HeaderYaw(float):
    """The right hand's yaw over the header: the connector along the header (the board's y), key to the slot."""

    def __new__(cls):
        return float.__new__(cls, 0.0)

    @property
    def rot(self):
        return _down(_seen["psi"])


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _seen.update(header=None, psi=0.0, gap=None, z=None)
    _plan.update(t_seat=None, z_seat=None)
    # the left hand's pinch point: at the board's near edge (-x), PINCH_IN in, at GRIP_Y, at the board's mid-thickness
    gx = HX - BOARD_HX + PINCH_IN - TIP_ABOVE_TCP  # the TCP sits TIP_ABOVE_TCP behind the pad tips
    gy = HY + GRIP_Y
    gz = BOARD_Z0 - 0.001 + (BOARD_T - STIFF_T) / 2  # the middle of the board and stiffener (sunk 1 mm into the foam)
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
    yaw = _HeaderYaw()
    tx, ty = TRAY_XY
    right = [
        Waypoint(2.5, (tx, ty, PICK_HOVER), CONN_WIDE, "above the connector", 0.0),
        Waypoint(1.5, (tx, ty, PICK_Z), CONN_WIDE, "pads either side of its ends", 0.0),
        Waypoint(0.8, (tx, ty, PICK_Z), CONN_HOLD, "grip the connector by its ends", 0.0),
        Waypoint(1.2, (tx, ty, PICK_HOVER), CONN_HOLD, "lift", 0.0),
        Waypoint(2.0, (tx, ty + 0.04, TOP + 0.16), CONN_HOLD, "up, out of the left hand's way", 0.0),
    ]
    right.append(Waypoint(max(0.5, t_up - _dur(right)), (tx, ty + 0.04, TOP + 0.16), CONN_HOLD,
                          "wait for the board to be up", 0.0))
    right += [
        Waypoint(2.5, _AtHeader(WALL_TOP + 0.025), CONN_HOLD, "over the header, key to the slot", yaw),
        Waypoint(1.5, _AtHeader(WALL_TOP + 0.002), CONN_HOLD, "connector 2 mm above the header's walls", yaw),
        Waypoint(1.0, _AtHeader(WALL_TOP + 0.002), CONN_HOLD, "line it up", yaw),
        Waypoint(1.5, _AtHeader(SEAT_GAP + TRAVEL + 0.0003), CONN_HOLD, "straight down onto the contacts", yaw),
        Waypoint(0.5, _AtHeader(SEAT_GAP + TRAVEL + 0.0003), CONN_HOLD, "settle", yaw),
    ]
    _plan["t_press0"] = _dur(right)
    press_s = PRESS_DEPTH / PRESS_RATE + 1.5
    right += [
        Waypoint(0.1, _AtHeader("press"), CONN_HOLD, "press it down slowly", yaw),
        Waypoint(press_s, _AtHeader("press"), CONN_HOLD, "press until it seats, then ease to a light push", yaw),
    ]
    _plan["t_press1"] = _dur(right)
    right += [
        Waypoint(0.8, _AtHeader(SEAT_GAP), CONN_HOLD, "ease off at the seat", yaw),
        Waypoint(0.8, _AtHeader(SEAT_GAP), FINGER_OPEN, "let go", yaw),
        Waypoint(0.5, _AtHeader(SEAT_GAP), FINGER_OPEN, "wait for the fingers to clear", yaw),
        Waypoint(1.5, _AtHeader(WALL_TOP + 0.03), FINGER_OPEN, "straight up off the connector", yaw),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_free = _dur(right) - 2.0
    left.append(Waypoint(max(0.5, t_free - _dur(left)), (gx, gy, gz + LIFT), BOARD_HOLD,
                         "hold the board up while the cable is plugged", POINT_X))
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
