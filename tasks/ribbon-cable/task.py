"""On the station table: a small sensor board on an orange 3D-printed bench stand, and the black socket connector of a
ribbon cable lying on a tray in front of it.

The stand lies along y. Its left end is a raised block with a black rubber press pad on top; its middle is a flat foot
on the bench; at its right end a riser carries the board, screwed down by its two mounting holes at one end. The rest
of the board sticks out 5 cm past the end of the foot, 2.5 cm above the bench, and near its free end it carries a black
shrouded 2 x 5 pin header: a box with walls round ten gold pins. The header's right wall, the one facing the board's
free edge, has a 5 mm notch in its middle: the key slot. A white tape outline on the bench marks where the stand's foot
starts.

The connector is a black block, 16 mm long, 8 mm wide and 11 mm tall, with a raised key on the middle of one long side
and a short stub of grey ribbon cable, red stripe on one edge, standing up out of its top. It stands on the tray with
its length along y and its key towards the robots. Plugging it in means: turn it a quarter turn so its key faces the
header's key slot, line it up over the header, lower it in and press it down until it gives and seats, about 5 N. The
other way round, the key lands on the top of the header's wall and it cannot go in. Pressing on the header alone tips
the stand over the end of its foot, so one hand holds the stand down by its press pad while the other plugs.

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

SCENE = "station"
TOP = TABLE_TOP[SCENE]

PLA = (0.93, 0.45, 0.10)  # orange printed stand
RUBBER = (0.08, 0.08, 0.09)
BRASS = (0.80, 0.64, 0.30)
FR4 = (0.05, 0.36, 0.20)  # green solder mask
BLACK = (0.05, 0.05, 0.06)
GOLD = (0.90, 0.72, 0.30)
RIBBON_C = (0.72, 0.72, 0.74)
RED = (0.80, 0.10, 0.08)
WHITE = (0.96, 0.96, 0.96)
TRAY_C = (0.62, 0.63, 0.66)

# -- the stand with its board and header (one loose body). Frame: origin on the bench under the header's centre, x away
# from the robots, y to their left. -----------------------------------------------------------------------------------
SX, SY = 0.30, -0.08  # the header's centre, world
FOOT_HW, FOOT_Y0, FOOT_Y1, FOOT_T = 0.020, 0.040, 0.220, 0.005  # the foot: 40 mm wide, 180 mm long, 5 mm thick
RISER_Y1, RISER_TOP = 0.075, 0.020  # the riser on the foot's right end, 2 cm tall
BLOCK_Y0, BLOCK_TOP = 0.190, 0.025  # the block at the foot's left end, 2.5 cm tall
PAD, PAD_T = 0.024, 0.001  # the rubber press pad on the block
PAD_Y = (BLOCK_Y0 + FOOT_Y1) / 2
STANDOFF_R, STANDOFF_XY = 0.0025, ((-0.019, 0.064), (0.019, 0.064))
BOARD_Z0, BOARD_T = 0.025, 0.0016  # the board's underside 2.5 cm above the bench, on the standoffs
BOARD_HX, BOARD_Y0, BOARD_Y1 = 0.025, -0.010, 0.070  # 50 x 80 mm; 5 cm of it past the foot's end at y = 0.040
ZT = BOARD_Z0 + BOARD_T  # the board's top
CHIP, CHIP_T, CHIP_Y = 0.006, 0.001, 0.030
STAND_MASS = 0.055  # stand, board and header

# The connector (below) is 16 x 8 mm; the header's cavity leaves 0.6 mm round it, inside walls 1.2 mm thick.
CONN_L, CONN_W, CONN_H = 0.016, 0.008, 0.011
CLEAR, WALL = 0.0006, 0.0012
CAV_X, CAV_Y = CONN_L + 2 * CLEAR, CONN_W + 2 * CLEAR  # 17.2 x 9.2 mm, long along x
OUT_X, OUT_Y = CAV_X + 2 * WALL, CAV_Y + 2 * WALL  # 19.6 x 11.6 mm
FLOOR, WALL_TOP = 0.001, 0.007  # the header's floor and the top of its walls, above the board's top
SLOT_W = 0.005  # the key slot through the middle of the -y wall
_wall_h = WALL_TOP - FLOOR
_seg = (CAV_X / 2 - SLOT_W / 2)  # each piece of the slotted wall, 6.1 mm

# The press fit: the ten pins are not contacts. The connector rests on a sprung contact plate inside the header (black,
# with the pins' gold tips on it), 4 mm above the board. Sprung up, the plate gives way once the connector, in the
# header the right way round, pushes it down 1 mm (5 N); its target then drops to the bottom of its 2.5 mm travel and
# stays there, and the connector follows to its seat, 1.5 mm above the board.
SEAT_GAP = 0.0015  # the seated connector's underside above the board's top
TRAVEL, TRIGGER = 0.0025, 0.001
PLATE_K, PLATE_C = 5000.0, 10.0  # N/m: 1 mm of give, the trigger, takes 5 N
PIN, PIN_PROUD, PLATE_T = 0.00064, 0.0001, 0.0004
PIN_TOP0 = ZT + SEAT_GAP + TRAVEL  # the pin tips at rest: the unpressed connector's underside, 4 mm above the board
_plate_top = PIN_TOP0 - PIN_PROUD
_pin_h = PIN_PROUD + 0.0002
PINS = [(x, y) for x in (-0.00508, -0.00254, 0.0, 0.00254, 0.00508) for y in (-0.00127, 0.00127)]

PLATE = Joint(
    "contacts", kind="slide", axis="Z", anchor=(0.0, 0.0, _plate_top - PLATE_T / 2), lower=-TRAVEL, upper=0.0,
    initial=0.0, stiffness=PLATE_K, damping=PLATE_C, target=0.0, mass=0.002,
    boxes=[((0.0, 0.0, _plate_top - PLATE_T / 2), (CAV_X - 0.0004, CAV_Y - 0.0004, PLATE_T), BLACK)]
    + [((x, y, PIN_TOP0 - _pin_h / 2), (PIN, PIN, _pin_h), GOLD) for x, y in PINS],
)

_wz = ZT + FLOOR + _wall_h / 2
STAND = part(
    "sensor_board", pos=(SX, SY, TOP), mass=STAND_MASS, friction=0.6,
    boxes=[
        # the printed stand: foot, riser, block, and the rubber pad on the block
        ((0.0, (FOOT_Y0 + FOOT_Y1) / 2, FOOT_T / 2), (2 * FOOT_HW, FOOT_Y1 - FOOT_Y0, FOOT_T), PLA),
        ((0.0, (FOOT_Y0 + RISER_Y1) / 2, (FOOT_T + RISER_TOP) / 2), (2 * FOOT_HW, RISER_Y1 - FOOT_Y0,
                                                                     RISER_TOP - FOOT_T), PLA),
        ((0.0, PAD_Y, (FOOT_T + BLOCK_TOP) / 2), (2 * FOOT_HW, FOOT_Y1 - BLOCK_Y0, BLOCK_TOP - FOOT_T), PLA),
        ((0.0, PAD_Y, BLOCK_TOP + PAD_T / 2), (PAD, PAD, PAD_T), RUBBER),
        # the board and a chip on it
        ((0.0, (BOARD_Y0 + BOARD_Y1) / 2, BOARD_Z0 + BOARD_T / 2), (2 * BOARD_HX, BOARD_Y1 - BOARD_Y0, BOARD_T), FR4),
        ((0.0, CHIP_Y, ZT + CHIP_T / 2), (CHIP, CHIP, CHIP_T), BLACK),
        # the header: its floor, two end walls, the plain +y wall and the -y wall in two pieces either side of the slot
        ((0.0, 0.0, ZT + FLOOR / 2), (OUT_X, OUT_Y, FLOOR), BLACK),
        ((CAV_X / 2 + WALL / 2, 0.0, _wz), (WALL, OUT_Y, _wall_h), BLACK),
        ((-CAV_X / 2 - WALL / 2, 0.0, _wz), (WALL, OUT_Y, _wall_h), BLACK),
        ((0.0, CAV_Y / 2 + WALL / 2, _wz), (CAV_X, WALL, _wall_h), BLACK),
        ((SLOT_W / 2 + _seg / 2, -CAV_Y / 2 - WALL / 2, _wz), (_seg, WALL, _wall_h), BLACK),
        ((-SLOT_W / 2 - _seg / 2, -CAV_Y / 2 - WALL / 2, _wz), (_seg, WALL, _wall_h), BLACK),
    ],
    cylinders=[((x, y, (RISER_TOP + BOARD_Z0) / 2), STANDOFF_R, BOARD_Z0 - RISER_TOP, BRASS) for x, y in STANDOFF_XY],
    joints=[PLATE],
)

# -- the connector (loose). Frame: origin at the middle of its underside; its length along x, its key on its -y side. -
KEY_W, KEY_P = 0.0035, 0.002  # the key: 3.5 mm wide, 2 mm proud, the connector's full height
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
        ((-RIBBON_W / 2 + 0.0005, 0.0, _rz), (0.001, RIBBON_T + 0.0002, RIBBON_H), RED),  # the stripe: pin 1
    ],
)
TRAY_PART = part("tray", pos=(*TRAY_XY, TOP), fixed=True, friction=0.6,
                 boxes=[((0.0, 0.0, TRAY[2] / 2), TRAY, TRAY_C)])

# -- the tape outline round the stand's foot (fixed): 3 mm wide, its inner edges 2 mm outside the foot. -------------
TAPE_W, TAPE_T, TAPE_GAP = 0.003, 0.0004, 0.002
_ox, _oy0, _oy1 = FOOT_HW + TAPE_GAP + TAPE_W / 2, FOOT_Y0 - TAPE_GAP - TAPE_W / 2, FOOT_Y1 + TAPE_GAP + TAPE_W / 2
OUTLINE = part(
    "outline", pos=(SX, SY, TOP), fixed=True, friction=0.6,
    boxes=[
        *[((s * _ox, (_oy0 + _oy1) / 2, TAPE_T / 2), (TAPE_W, _oy1 - _oy0 + TAPE_W, TAPE_T), WHITE) for s in (1, -1)],
        *[((0.0, y, TAPE_T / 2), (2 * _ox - TAPE_W, TAPE_W, TAPE_T), WHITE) for y in (_oy0, _oy1)],
    ],
)

OBJECTS = [STAND, CONNECTOR, TRAY_PART, OUTLINE]


# -- where the connector is in the header -----------------------------------------------------------------------------
IN_R = 0.0015  # in the header: its underside's centre within 1.5 mm of the header's axis,
IN_TILT = 10.0  # tilted no more than 10 degrees from it,
IN_GAP = WALL_TOP - 0.001  # and its underside no more than 6 mm above the board (5 mm above the floor, 1 mm inside)
SEATED_GAP = SEAT_GAP + 0.0005  # seated: within 0.5 mm of its seat
KEY_TOL = math.radians(10.0)  # the key in the slot: its long axis within 10 degrees, key side towards the slot
MOVE_MAX = 0.010  # the stand's foot no corner more than 10 mm from where it started
TIP_MAX = 15.0  # and tilted no more than 15 degrees
FOOT_CORNERS = [(sx * FOOT_HW, y, 0.0) for sx in (1, -1) for y in (FOOT_Y0, FOOT_Y1)]


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _yaw(rot) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def connector_in_header(stand, conn):
    """(offset of the connector's underside centre from the header's axis in m, its underside's height above the
    board's top in m, its tilt from the header's axis in degrees, its turn from the keyed orientation in rad), in the
    stand's frame. The turn is 0 with its key towards the slot, pi the other way round."""
    p = stand.to_local(conn.pos)
    rel = stand.rotation().T @ conn.rotation()
    tilt = float(np.degrees(np.arccos(np.clip(rel[2, 2], -1.0, 1.0))))
    return float(math.hypot(p[0], p[1])), float(p[2] - ZT), tilt, _yaw(rel)


def _in(r, gap, tilt) -> bool:
    return r <= IN_R and gap <= IN_GAP and tilt <= IN_TILT


def stand_moved(stand) -> float:
    """How far the stand's foot has moved: the largest horizontal distance of a bottom corner from where it started
    (m)."""
    start = np.array([SX, SY])
    return max(float(np.linalg.norm(stand.to_world(c)[:2] - (start + np.array(c[:2])))) for c in FOOT_CORNERS)


def stand_ok(stand) -> bool:
    return stand_moved(stand) <= MOVE_MAX and stand.tilt_deg() <= TIP_MAX


# -- the press fit -----------------------------------------------------------------------------------------------------
_state = {"latched": False}


@on_reset
def _reset_fit(env):
    _state["latched"] = False


@on_step
def _fit(env):
    """The contact plate's drive target: sprung up until the connector, in the header with its key in the slot, has
    pushed it down 1 mm; from then on at the bottom of its travel."""
    import torch

    import labkit

    stand = labkit._part_state(env, labkit._SPECS["sensor_board"])
    conn = labkit._part_state(env, labkit._SPECS["connector"])
    r, gap, tilt, turn = connector_in_header(stand, conn)
    if _in(r, gap, tilt) and abs(turn) <= KEY_TOL and stand.joints["contacts"] <= -TRIGGER:
        _state["latched"] = True
    asset = env.unwrapped.scene["sensor_board"]
    ids = [list(asset.joint_names).index(PLATE.joint_name)]
    target = -TRAVEL if _state["latched"] else 0.0
    asset.set_joint_position_target(torch.tensor([[target]], device=asset.device), joint_ids=ids)


# -- scoring -----------------------------------------------------------------------------------------------------------
SUCCESS_HOLD_S = 17.0  # end a trial early only once it has stayed seated for 17 s: in the demo the press starts at
#                        14.5 s and cannot seat before 14.8 s (it takes 1 mm of give), and the plan ends at 30.7 s with
#                        both hands parked, so the video plays to its end


def score(objects):
    """0 if the stand's foot has moved more than 10 mm from where it started (any bottom corner) or the stand is tilted
    more than 15 degrees. Otherwise 0 unless the connector is in the header (underside centre within 1.5 mm of the
    header's axis, tilted no more than 10 degrees from it, underside no more than 6 mm above the board's top, inside
    the walls). 0.25 in the header but not seated. 1.0 seated: underside within 0.5 mm of its seat (at most 2 mm above
    the board's top) with its key in the header's slot."""
    stand, conn = objects["sensor_board"], objects["connector"]
    if not stand_ok(stand):
        return 0.0
    r, gap, tilt, turn = connector_in_header(stand, conn)
    if not _in(r, gap, tilt):
        return 0.0
    return 1.0 if gap <= SEATED_GAP and abs(turn) <= KEY_TOL else 0.25


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The connector's offset from the header's axis plus its height above its seat (m)."""
    r, gap, _, _ = connector_in_header(objects["sensor_board"], objects["connector"])
    return r + max(0.0, gap - SEAT_GAP)


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Left arm: comes straight down onto the stand's rubber pad with its fingers closed and presses the stand into the bench
# (about 4 N) until the connector is seated; then lifts straight off and parks. Unheld, the stand tips over the end of
# its foot once the header is pressed with about 1.2 N (its weight acts 85 mm behind that edge, the header 40 mm in
# front of it); held with 4 N on the pad, 165 mm behind the edge, it stays flat up to about 17 N. It does not grip the
# stand: the arm is laterally soft (about 7 mm per N at the pads), so a grip would let the hand drag the stand. The
# commanded press point is leaned against the arm's compliance so that the pads go straight down and the force is
# vertical.
# Right arm: picks the connector by its two ends (fingers closing along its length, pad tips 7 mm above its underside)
# and turns it a quarter turn anticlockwise on the way, so its key faces the slot. From there it is closed loop on the
# live stand (see ``_track``): every control step the hand is aimed at the header where the stand is now, turned with
# the stand's yaw, and trimmed by the measured offset of the connector from the header's axis and of its long axis from
# the header's. With the connector 2 mm above the header's walls the trim settles and is then frozen; the connector goes
# straight down into the header onto the contact plate and is pressed slowly (3 mm/s, commanded up to 16 mm below the
# seat: the arm gives about 2.2 mm per N pushed up, so the 5 N trigger is reached about 12 mm in) until the fit gives,
# then eased to a light push at the seat, to no push, let go and lifted straight off. Seated, the pad tips are 1.5 mm
# above the header's walls.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_UP = 0.007  # pad tips 7 mm above the connector's underside
CONN_WIDE = -0.012  # pads 24 mm apart round its 16 mm length
PICK_Z = TOP + TRAY[2] + GRIP_UP + TIP_ABOVE_TCP
PICK_HOVER = TOP + 0.07

# The arms' compliance at the pad tips (m of give per N pushed up on the pads, x, y, z), as measured on the servo-horn
# rig at nearly the same hand poses: a press commanded at (point - F * C) leaves the pads at the point with a purely
# vertical force F.
C_LEFT = np.array([1.12, -0.73, 2.16]) / 1000.0
C_RIGHT = np.array([0.94, 0.61, 2.20]) / 1000.0


def conn_z(underside: float) -> float:
    """TCP height that puts the connector's underside ``underside`` above the board's top."""
    return TOP + ZT + underside + GRIP_UP + TIP_ABOVE_TCP


CARRY_Z = conn_z(WALL_TOP + 0.025)
ENTRY_Z = conn_z(WALL_TOP + 0.002)
REST_Z = conn_z(SEAT_GAP + TRAVEL + 0.0003)
SEAT_Z = conn_z(SEAT_GAP)
PRESS_MIN_Z = SEAT_Z - 0.016  # the deepest the press is commanded
PRESS_RATE = 0.003  # m/s
PUSH_Z = SEAT_Z - 0.003  # the light push on the seated connector (about 1.4 N)
PRESS_F_MAX = 8.0  # N, cap on the estimated press force used for the lean
CONN_YAW_PICK = 0.0  # fingers close along y, on its ends, while its length is along y
CONN_YAW_PLUG = math.pi / 2  # and along x once it is turned to the header (stand at its start)
BX, BY = SX, SY + PAD_Y  # the press pad's centre (stand at its start)
PAD_TOUCH = TOP + BLOCK_TOP + PAD_T + TIP_ABOVE_TCP  # closed pad tips on the rubber pad
PAD_HOVER = PAD_TOUCH + 0.03
HOLD_F = 4.0  # N, the hold-down force on the pad
HOLD = (BX - HOLD_F * C_LEFT[0], BY - HOLD_F * C_LEFT[1], PAD_TOUCH - HOLD_F * C_LEFT[2])  # 8.6 mm in, leaned

# The live part of the plan. ``_track`` reads the stand and the connector every control step; the right arm's waypoints
# from the carry on read their pose from ``_live``. Offline (no simulator) the values stay nominal, so the plan checks
# see the stand at its start.
TRACK_GAIN, TRACK_STEP, TRACK_MAX = 0.15, 0.0004, 0.004  # per step: 15% of the offset, at most 0.4 mm; 4 mm in all
YAW_GAIN, YAW_STEP, YAW_MAX = 0.15, math.radians(0.5), math.radians(8.0)
_live = {"header": np.array([SX, SY]), "psi": 0.0, "corr": np.zeros(2), "cyaw": 0.0, "gap": None, "frozen": False}
_plan = {"t_trim": 1e9, "t_freeze": 1e9, "t_press0": 1e9, "t_press1": 1e9, "t_latch": None, "z_latch": None}

_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot_duck", False):
    def _grip_rotation(yaw=0.0, _prev=_prev_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rotation._rot_duck = True
    _demo.grip_rotation = _grip_rotation

_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)  # the plan's time, for waypoints that move with the trial
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


def _clock() -> float:
    return float(getattr(_demo, "_plan_clock", 0.0))


def _live_reset():
    _live.update(header=np.array([SX, SY]), psi=0.0, corr=np.zeros(2), cyaw=0.0, gap=None, frozen=False)
    _plan.update(t_latch=None, z_latch=None)
    _demo._plan_clock = 0.0


@on_reset
def _reset_live(env):
    _live_reset()


@on_step
def _track(env):
    """Until the trim is frozen: the header's position and the stand's yaw, live; from the end of the carry, the trim,
    integrated from the connector's offset from the header's axis and its long axis's angle from the header's. Frozen
    once the connector is lowered into the header (or is in it), so the press cannot chase a stand it pushes. The
    connector's height is read every step, for the press's lean."""
    import labkit

    stand = labkit._part_state(env, labkit._SPECS["sensor_board"])
    conn = labkit._part_state(env, labkit._SPECS["connector"])
    r, gap, tilt, _ = connector_in_header(stand, conn)
    _live["gap"] = gap
    t = _clock()
    if _live["frozen"] or _in(r, gap, tilt) or t >= _plan["t_freeze"]:
        _live["frozen"] = True
        return
    rot = stand.rotation()
    header = stand.to_world((0.0, 0.0, ZT))[:2]
    _live.update(header=header, psi=_yaw(rot))
    if t >= _plan["t_trim"]:
        err = header - conn.pos[:2]
        corr = _live["corr"] + np.clip(TRACK_GAIN * err, -TRACK_STEP, TRACK_STEP)
        _live["corr"] = np.clip(corr, -TRACK_MAX, TRACK_MAX)
        e = _wrap(_yaw(rot) - _yaw(conn.rotation()))
        _live["cyaw"] = float(np.clip(_live["cyaw"] + np.clip(YAW_GAIN * e, -YAW_STEP, YAW_STEP), -YAW_MAX, YAW_MAX))


def _press_z() -> float:
    """The press's commanded TCP height: down from the contact plate at PRESS_RATE (to PRESS_MIN_Z at most) until the
    fit gives, then back at PRESS_RATE to the light push. Offline (no latch) it goes all the way down: the plan checks
    see the deepest pose."""
    t = min(_clock(), _plan["t_press1"])
    p = _plan
    if p["t_latch"] is None:
        z = max(PRESS_MIN_Z, REST_Z - PRESS_RATE * max(0.0, t - p["t_press0"]))
        if _state["latched"]:
            p["t_latch"], p["z_latch"] = t, z
        return z
    d = PUSH_Z - p["z_latch"]
    return p["z_latch"] + math.copysign(min(abs(d), PRESS_RATE * max(0.0, t - p["t_latch"])), d)


class _ConnAt:
    """The right hand's TCP over the header: the live header position plus the trim, at height ``z`` (a number, or
    ``"press"`` for the press, which also leans against the arm's compliance by the estimated press force)."""

    def __init__(self, z):
        self.z = z

    def __array__(self, dtype=None, copy=None):
        xy = _live["header"] + _live["corr"]
        if self.z == "press":
            z = _press_z()
            now = conn_z(_live["gap"]) if _live["gap"] is not None else REST_Z  # offline: on the contact plate
            f = float(np.clip((now - z) / C_RIGHT[2], 0.0, PRESS_F_MAX))
            xy = xy - f * C_RIGHT[:2]
        else:
            z = self.z
        return np.asarray((xy[0], xy[1], z), dtype=dtype or float)


class _ConnYaw(float):
    """The right hand's yaw over the header: the connector's long axis along the header's, key to the slot, turned with
    the stand, plus the trim."""

    def __new__(cls):
        return float.__new__(cls, CONN_YAW_PLUG)

    @property
    def rot(self):
        return _prev_grip(CONN_YAW_PLUG + _live["psi"] + _live["cyaw"])


def S(x, y, z):
    """Stand coordinates to world."""
    return (SX + x, SY + y, TOP + z)


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("foot", S(0.0, (FOOT_Y0 + FOOT_Y1) / 2, FOOT_T / 2), (2 * FOOT_HW, FOOT_Y1 - FOOT_Y0, FOOT_T)),
    ("riser", S(0.0, (FOOT_Y0 + RISER_Y1) / 2, RISER_TOP / 2), (2 * FOOT_HW, RISER_Y1 - FOOT_Y0, RISER_TOP)),
    ("block", S(0.0, PAD_Y, (BLOCK_TOP + PAD_T) / 2), (2 * FOOT_HW, FOOT_Y1 - BLOCK_Y0, BLOCK_TOP + PAD_T)),
    ("board", S(0.0, (BOARD_Y0 + BOARD_Y1) / 2, BOARD_Z0 + BOARD_T / 2), (2 * BOARD_HX, BOARD_Y1 - BOARD_Y0, BOARD_T)),
    ("header", S(0.0, 0.0, ZT + WALL_TOP / 2), (OUT_X, OUT_Y, WALL_TOP)),
    ("tray", (*TRAY_XY, TOP + TRAY[2] / 2), TRAY),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    shut = FINGER_CLOSED
    left = [
        Waypoint(2.5, (BX, BY, PAD_HOVER), shut, "above the stand's press pad, fingers closed"),
        Waypoint(1.2, (BX, BY, PAD_TOUCH + 0.003), shut, "straight down to just above the pad"),
        Waypoint(0.8, (BX, BY, PAD_TOUCH), shut, "pad tips onto the rubber pad"),
        Waypoint(1.2, HOLD, shut, "press the stand down by its pad"),
        Waypoint(0.5, HOLD, shut, "settle"),
    ]
    t_held = _dur(left)
    yaw = _ConnYaw()
    tx, ty = TRAY_XY
    right = [
        Waypoint(2.5, (tx, ty, PICK_HOVER), CONN_WIDE, "above the connector", CONN_YAW_PICK),
        Waypoint(1.5, (tx, ty, PICK_Z), CONN_WIDE, "pads either side of its ends", CONN_YAW_PICK),
        Waypoint(0.8, (tx, ty, PICK_Z), FINGER_CLOSED, "grip the connector by its ends", CONN_YAW_PICK),
        Waypoint(1.2, (tx, ty, PICK_HOVER), FINGER_CLOSED, "lift", CONN_YAW_PICK),
        Waypoint(2.5, _ConnAt(CARRY_Z), FINGER_CLOSED, "over the header, key turned to the slot", yaw),
    ]
    _plan["t_trim"] = _dur(right)  # the connector is turned and over the header: the trim starts
    right.append(Waypoint(max(0.5, t_held - _dur(right)), _ConnAt(CARRY_Z), FINGER_CLOSED,
                          "wait for the stand to be held", yaw))
    right += [
        Waypoint(1.5, _ConnAt(ENTRY_Z), FINGER_CLOSED, "connector 2 mm above the header's walls", yaw),
        Waypoint(2.0, _ConnAt(ENTRY_Z), FINGER_CLOSED, "line the connector up on the header", yaw),
    ]
    _plan["t_freeze"] = _dur(right)
    right += [
        Waypoint(1.5, _ConnAt(REST_Z), FINGER_CLOSED, "straight down into the header onto the contacts", yaw),
        Waypoint(0.5, _ConnAt(REST_Z), FINGER_CLOSED, "settle", yaw),
    ]
    _plan["t_press0"] = _dur(right)
    press_s = (REST_Z - PRESS_MIN_Z) / PRESS_RATE + 1.0
    right += [
        Waypoint(0.1, _ConnAt("press"), FINGER_CLOSED, "press the connector down slowly", yaw),
        Waypoint(press_s, _ConnAt("press"), FINGER_CLOSED, "press until the fit gives, then ease to a light push",
                 yaw),
    ]
    _plan["t_press1"] = _dur(right)
    right += [
        Waypoint(1.0, _ConnAt(SEAT_Z), FINGER_CLOSED, "ease off at the seat", yaw),
        Waypoint(0.8, _ConnAt(SEAT_Z), CONN_WIDE, "let go", yaw),
        Waypoint(1.2, _ConnAt(CARRY_Z), CONN_WIDE, "straight up off the connector", yaw),
    ]
    t_done = _dur(right)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    left.append(Waypoint(t_done - _dur(left), HOLD, shut, "hold the stand down while the connector is pressed"))
    left += [
        Waypoint(0.8, (BX, BY, PAD_TOUCH + 0.003), shut, "lift straight off the pad"),
        Waypoint(1.0, (BX, BY, PAD_HOVER), shut, "up"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    _plan["t_end"] = t_end
    return {"left": left, "right": right}
