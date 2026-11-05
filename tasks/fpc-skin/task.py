"""On the station table: a UMI-style parallel gripper (the open-source actuated-UMI design, its real CAD, MIT licence)
lying on its side on the bench, fin-ray fingers pointing at the robots. The finger on the robots' left carries a tactile
skin: a flexible sensor sheet over its whole inner face, 11 x 2.2 cm. The skin's flat flexible cable (FPC) runs back
along the jaw to the top of the gripper body, where a sensor communication board sits on standoffs. The FPC's last
3.3 cm is free: it ends in a stiffened tip that plugs into a flip-lock ZIF connector on the board.

The task: plug the FPC into the connector and lock it. The connector's actuator (latch) starts open, standing up behind
the slot. The free FPC holds a curl (bend memory) that keeps its tip lifted 1.5 cm and 3 mm short of the slot, 2 mm
off to one side. Pressed flat and slid in, the tip reaches the slot's back stop (3.5 mm in), but let go with the latch
open, the curl pulls it back out; the latch has to be closed while the FPC is held in. Closed onto an inserted FPC, the
latch clamps it; closed on an empty slot it clamps nothing, and an FPC pushed in under a closed latch is not clamped
either. Bending any hinge of the FPC past 60 degrees creases it.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import Joint, on_reset, on_step, part, stl_mesh  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
HERE = pathlib.Path(__file__).resolve().parent

BLACK = (0.07, 0.07, 0.08)
WHITE = (0.93, 0.92, 0.88)
STEEL = (0.70, 0.71, 0.73)
MOTOR = (0.25, 0.25, 0.27)
BLUE = (0.10, 0.25, 0.60)
PCB = (0.05, 0.30, 0.15)  # green solder mask
CHIP = (0.10, 0.10, 0.11)
GOLD = (0.80, 0.62, 0.25)
POLYIMIDE = (0.85, 0.50, 0.12)  # amber FPC film
STIFF = (0.55, 0.32, 0.08)  # the FPC tip's stiffener, darker
SKIN = (0.20, 0.21, 0.23)
TAXEL = (0.62, 0.64, 0.67)
CONN = (0.92, 0.90, 0.84)  # connector housing, ivory LCP
LATCH = (0.12, 0.12, 0.13)
FOAM = (0.15, 0.15, 0.16)

# -- the gripper (fixed). Part frame: origin at the gripper's assembly origin, axes along the world's. -----------------
G0 = (0.36, 0.0, TOP + 0.0785)
# Assembly x (jaw travel) -> world y; assembly y -> up; assembly z (fingers) -> towards the robots (-x).
R_ASSY = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
MESHES = [stl_mesh(HERE / "umi" / f"{k}.stl", c, rpy=R_ASSY) for k, c in (
    ("structure", WHITE), ("mount", BLACK), ("jaw_left", BLACK), ("jaw_right", BLACK), ("rail", STEEL),
    ("motor", MOTOR), ("board", BLUE))]
ST_TOP = 0.027  # the gripper body's flat top (part frame z)

# Colliders for what the arms or the FPC can touch (the meshes are drawn only).
BODY_BOXES = [
    ((0.0018, 0.009, ST_TOP - 0.0265), (0.071, 0.126, 0.053), WHITE),  # the body, under its top
    ((-0.104, 0.0383, 0.0), (0.166, 0.038, 0.038), BLACK),  # left jaw (on the robots' left, +y)
    ((-0.104, -0.0383, 0.0), (0.166, 0.038, 0.038), BLACK),  # right jaw
]

# The tactile skin over the left finger's inner face (its flat face at y = 0.0573), with a grid of taxels.
FACE_Y = 0.0573
SKIN_T = 0.0008
SKIN_X, SKIN_Z = (-0.185, -0.075), 0.011
SKIN_BOXES = [((sum(SKIN_X) / 2, FACE_Y - SKIN_T / 2, 0.0), (SKIN_X[1] - SKIN_X[0], SKIN_T, 2 * SKIN_Z), SKIN)]
for _i in range(14):
    for _k in range(4):
        SKIN_BOXES.append(((SKIN_X[0] + 0.004 + _i * 0.0078, FACE_Y - SKIN_T - 0.0002, -0.0075 + _k * 0.005),
                           (0.0035, 0.0004, 0.0035), TAXEL))

# The communication board on 4 mm standoffs on the body's top, and its ZIF connector, entry facing +y.
STANDOFF = 0.004
BOARD = ((0.0, 0.045), (0.042, 0.060), 0.0016)  # centre xy, size xy, thickness
BT = ST_TOP + STANDOFF + BOARD[2]  # board top (part frame z)
FPC_W, FPC_T = 0.012, 0.0010  # 24 contacts at 0.5 mm pitch; drawn 1 mm thick (real: 0.12 mm) for stable contact
SLOT_W, SLOT_H = 0.0128, 0.0015
SLOT_FLOOR = BT  # the slot's floor is the board
HOUSING_W, HOUSING_H = 0.018, 0.0026
ENTRY_Y = 0.030  # the slot's mouth
DEPTH = 0.0035  # from the mouth to the back stop
STOP_Y = ENTRY_Y - DEPTH
BACK_Y = STOP_Y - 0.0015  # the housing's back face
HTOP = BT + HOUSING_H
_hz = (BT + HTOP) / 2
_sx = (HOUSING_W - SLOT_W) / 4 + SLOT_W / 2  # side walls' centre x
_chamfer = 0.0018
CONNECTOR_BOXES = [
    ((_sx, (BACK_Y + ENTRY_Y) / 2, _hz), ((HOUSING_W - SLOT_W) / 2, ENTRY_Y - BACK_Y, HOUSING_H), CONN),
    ((-_sx, (BACK_Y + ENTRY_Y) / 2, _hz), ((HOUSING_W - SLOT_W) / 2, ENTRY_Y - BACK_Y, HOUSING_H), CONN),
    ((0.0, (BACK_Y + STOP_Y) / 2, _hz), (SLOT_W, STOP_Y - BACK_Y, HOUSING_H), CONN),  # back, the stop
    ((0.0, (STOP_Y + ENTRY_Y) / 2, (SLOT_FLOOR + SLOT_H + HTOP) / 2),
     (SLOT_W, DEPTH, HTOP - SLOT_FLOOR - SLOT_H), CONN),  # roof over the slot
    # lead-in: the side walls flare out at the mouth
    ((SLOT_W / 2 + 0.0004, ENTRY_Y + _chamfer / 2, _hz), (0.0008, _chamfer * 1.15, HOUSING_H), CONN, -0.5),
    ((-SLOT_W / 2 - 0.0004, ENTRY_Y + _chamfer / 2, _hz), (0.0008, _chamfer * 1.15, HOUSING_H), CONN, 0.5),
    # contacts showing at the mouth
    *[((-SLOT_W / 2 + 0.0008 + 0.0005 * i, ENTRY_Y - 0.0004, SLOT_FLOOR + 0.0001), (0.0002, 0.0008, 0.0002), GOLD)
      for i in range(0, 24, 2)],
]

# The latch: a flip-lock actuator hinged along the housing's back top edge. Open (0) it stands upright; closing turns
# its top towards the mouth (+y) until it lies on the housing's roof (-pi/2). Light joint friction keeps it where it is
# left.
LATCH_T, LATCH_H = 0.0012, 0.0045
LATCH_AXIS = (0.0, BACK_Y + LATCH_T / 2, HTOP)
LATCH_CLOSED = -math.pi / 2
LATCH = Joint(
    "latch", kind="hinge", axis="X", anchor=LATCH_AXIS, lower=LATCH_CLOSED - 0.05, upper=0.15, initial=0.0, target=0.0,
    stiffness=0.0, damping=0.0005, friction=0.0006, mass=0.002,
    boxes=[((0.0, LATCH_AXIS[1], HTOP + LATCH_H / 2), (HOUSING_W, LATCH_T, LATCH_H), LATCH)],
)

# The free FPC: from where it leaves its foam pad (y = PAD_Y) towards the connector (-y): a yaw hinge (about z) at the
# pad, five 5 mm segments joined by bend hinges (about x), then the 12 mm stiffened tip. Its rest shape (bend memory) is
# an arch 6 mm high that leaves the tip flat on the board 1.7 mm short of the mouth and 1.7 mm off to the side; flat
# (all hinges at 0) the tip sits 0.3 mm short of the back stop. Bending up is negative.
ARCH = 0.005
TIP_L = 0.012
FPC_L = 5 * ARCH + TIP_L
PAD_Y = STOP_Y + 0.0003 + FPC_L
FPC_Z = BT + 0.0002 + FPC_T / 2  # the FPC's mid-plane when flat, 0.2 mm over the board (the slot's floor)
FPC_NAMES = ["fpc_yaw", "fpc_b1", "fpc_b2", "fpc_b3", "fpc_b4", "fpc_b5"]
REST = dict(zip(FPC_NAMES, [math.radians(a) for a in (3.0, -55.0, 35.0, 40.0, 35.0, -55.0)]))
FPC_K = {n: (0.004 if n == "fpc_yaw" else 0.006) for n in FPC_NAMES}  # N m/rad
CREASE = math.radians(70.0)
SEG_L = [ARCH] * 5 + [TIP_L]  # the segment after each hinge
_seg_y0 = [PAD_Y - ARCH * k for k in range(6)]  # each segment's start (flat), = its hinge's position

FPC_JOINTS = []
_prev = ""
for _k, _n in enumerate(FPC_NAMES):
    _y0, _L = _seg_y0[_k], SEG_L[_k]
    if _n == "fpc_b5":  # the stiffened tip, with its white insertion-depth line DEPTH from the end
        _end = _y0 - _L
        _boxes = [((0.0, _end + 0.003, FPC_Z), (FPC_W, 0.006, FPC_T * 1.05), STIFF),
                  ((0.0, (_end + 0.006 + _y0) / 2, FPC_Z), (FPC_W, _y0 - _end - 0.006, FPC_T), POLYIMIDE),
                  ((0.0, _end + DEPTH, FPC_Z + FPC_T / 2 + 0.00006), (FPC_W, 0.0005, 0.00012), WHITE)]
    else:
        _boxes = [((0.0, _y0 - _L / 2, FPC_Z), (FPC_W, _L, FPC_T), POLYIMIDE)]
    FPC_JOINTS.append(Joint(_n, kind="hinge", axis="Z" if _n == "fpc_yaw" else "X", anchor=(0.0, _y0, FPC_Z),
                            lower=-0.6 if _n == "fpc_yaw" else -1.45, upper=0.6 if _n == "fpc_yaw" else 1.45,
                            initial=REST[_n], target=REST[_n], stiffness=FPC_K[_n], damping=0.0003, mass=0.0004,
                            parent=_prev, boxes=_boxes))
    _prev = _n

# The FPC's fixed run, drawn: from the skin's back end, along the top of the left jaw, up onto the body, through a
# foam pad that holds it at the connector's height and points it at the slot.
_jaw_top = 0.019
ROUTE_BOXES = [
    ((-0.071, FACE_Y - 0.004, 0.0), (0.008, 0.008, 0.016), FOAM),  # the skin's tail, folded round the finger root
    ((-0.050, 0.050, _jaw_top + 0.0006), (0.044, FPC_W, 0.0008), POLYIMIDE),  # along the jaw's top
    ((-0.029, 0.050, (_jaw_top + ST_TOP) / 2 + 0.001), (0.0008, FPC_W, ST_TOP - _jaw_top + 0.002), POLYIMIDE),
    ((-0.015, 0.050, ST_TOP + 0.0004), (0.028, FPC_W, 0.0008), POLYIMIDE),  # across the body's top
    ((0.0, (0.050 + PAD_Y + 0.009) / 2, ST_TOP + 0.0004), (FPC_W, PAD_Y + 0.009 - 0.044, 0.0008), POLYIMIDE),
    ((0.0, PAD_Y + 0.005, (BT + FPC_Z - FPC_T / 2) / 2), (0.016, 0.010, FPC_Z - FPC_T / 2 - BT), FOAM),  # pad
    ((0.0, PAD_Y + 0.005, FPC_Z + FPC_T / 2 + 0.0006), (0.016, 0.010, 0.0012), FOAM),  # foam tape over it
    ((0.0, PAD_Y + 0.004, FPC_Z), (FPC_W, 0.008, FPC_T), POLYIMIDE),  # held under the tape
    ((0.0, PAD_Y + 0.0085, (ST_TOP + FPC_Z) / 2 + 0.0004), (FPC_W, 0.0008, FPC_Z - ST_TOP + 0.0008), POLYIMIDE),
]

_bx, _by = BOARD[0]
_bw, _bl = BOARD[1]
BOARD_BOXES = [
    ((_bx, _by, BT - BOARD[2] / 2), (_bw, _bl, BOARD[2]), PCB),
    *[((_bx + sx * (_bw / 2 - 0.003), _by + sy * (_bl / 2 - 0.003), ST_TOP + STANDOFF / 2), (0.004, 0.004, STANDOFF),
       STEEL) for sx in (-1, 1) for sy in (-1, 1)],
    ((_bx + 0.008, _by + 0.012, BT + 0.0006), (0.009, 0.009, 0.0012), CHIP),  # microcontroller
    ((_bx - 0.010, _by + 0.014, BT + 0.0005), (0.006, 0.005, 0.001), CHIP),
    ((_bx, _by + _bl / 2 - 0.004, BT + 0.0016), (0.010, 0.006, 0.0032), STEEL),  # cable out to the arm
]

GRIPPER = part(
    "gripper", pos=G0, fixed=True, friction=0.6,
    boxes=BODY_BOXES + SKIN_BOXES + CONNECTOR_BOXES + ROUTE_BOXES + BOARD_BOXES,
    meshes=MESHES,
    joints=[LATCH, *FPC_JOINTS],
)
OBJECTS = [GRIPPER]


def W(p):
    """Part frame to world."""
    return np.asarray(G0) + np.asarray(p, float)


# -- the latch's clamp ------------------------------------------------------------------------------------------------
LATCH_SHUT = LATCH_CLOSED + math.radians(15.0)  # closed within 15 degrees of flat on the roof
CLAMP_DEPTH = 0.0025  # the tip at least 2.5 mm into the slot when the latch shuts on it
_trial = {"clamped": False, "was_shut": False, "tip": None, "ids": None, "set": None, "q": None}


def _fpc_points(q: dict):
    """Forward kinematics of the FPC from its joint angles (part frame): the hinge points and the tip's centre, and
    the tip segment's direction."""
    yaw = q["fpc_yaw"]
    p = np.array([0.0, PAD_Y, FPC_Z])
    pts, pitch, d = [p], 0.0, None
    for k, n in enumerate(FPC_NAMES):
        if n != "fpc_yaw":
            pitch += q[n]
        # bending about the segment's own x axis (turned by the yaw): -y tips up for a negative pitch
        d = np.array([math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
        p = p + SEG_L[k] * d
        pts.append(p)
    return pts, d


def insertion(q: dict):
    """(depth of the tip past the mouth along the slot (m), its centre's sideways offset (m), its yaw (deg), its height
    off the slot's mid-plane (m))."""
    pts, d = _fpc_points(q)
    tip = pts[-1]
    return (ENTRY_Y - tip[1], tip[0], math.degrees(math.atan2(d[0], -d[1])), tip[2] - FPC_Z)


@on_reset
def _reset(env):
    _trial.update(clamped=False, was_shut=False, set=None, q=None)
    _live_reset()


@on_step
def _step(env):
    import torch

    import labkit

    st = labkit._part_state(env, labkit._SPECS["gripper"])
    q = st.joints
    _trial["q"] = q
    asset = env.unwrapped.scene["gripper"]
    if _trial["ids"] is None:
        names = list(asset.joint_names)
        _trial["ids"] = [names.index(j.joint_name) for j in FPC_JOINTS]
    shut = q["latch"] <= LATCH_SHUT
    depth, side, _, _ = insertion(q)
    if shut and not _trial["was_shut"]:  # the latch has just shut: it clamps the FPC only if it is in the slot
        _trial["clamped"] = depth >= CLAMP_DEPTH and abs(side) <= (SLOT_W - FPC_W) / 2 + 0.0003
    if not shut:
        _trial["clamped"] = False
    _trial["was_shut"] = shut
    want = "flat" if _trial["clamped"] else "rest"
    if want != _trial["set"]:  # clamped, the contacts hold the FPC flat in the slot; otherwise its curl returns
        tgt = [0.0 if want == "flat" else REST[n] for n in FPC_NAMES]
        asset.set_joint_position_target(torch.tensor([tgt], device=asset.device), joint_ids=_trial["ids"])
        _trial["set"] = want
    _live_update(st)


# -- scoring ----------------------------------------------------------------------------------------------------------
SEAT_DEPTH = 0.0030  # the depth line within 0.5 mm of the mouth
SIDE_TOL = 0.0004
YAW_TOL = 3.0


def creased(q: dict) -> bool:
    return any(abs(q[n]) > CREASE for n in FPC_NAMES)


def score(objects):
    """0 if the FPC is creased (a hinge bent past 60 degrees) or its tip is not in the slot (at least 1 mm past the
    mouth, its centre within 1 mm of the slot's). 0.25 in the slot; 0.5 seated (at least 3.0 mm in, within 0.4 mm of
    centre, square within 3 degrees) with the latch open; 0.75 seated with the latch shut but not clamping it (shut
    before it was in); 1.0 seated and clamped."""
    g = objects["gripper"]
    q = g.joints
    if creased(q):
        return 0.0
    depth, side, yaw, _ = insertion(q)
    if depth < 0.001 or abs(side) > 0.001:
        return 0.0
    seated = depth >= SEAT_DEPTH and abs(side) <= SIDE_TOL and abs(yaw) <= YAW_TOL
    if not seated:
        return 0.25
    if q["latch"] > LATCH_SHUT:
        return 0.5
    return 1.0 if _trial["clamped"] else 0.75


SUCCESS_HOLD_S = 4.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    q = objects["gripper"].joints
    depth, side, _, _ = insertion(q)
    return max(0.0, DEPTH - depth) + abs(side) + max(0.0, q["latch"] - LATCH_CLOSED) * 0.01


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left arm (from the +y side): comes down with closed fingers, leaning 25 degrees towards -y, onto the FPC's third
# segment, presses it flat against the board, then drags it towards the slot: closed loop on the FPC's live shape, it
# aims the press point so the tip lines up with the slot (sideways) and slides on until the tip is seated, then holds.
# Right arm (from the -y side): leaning towards +y, puts its closed fingertips behind the upright latch and swings it
# over onto the roof along its arc, then lifts away. Then the left arm lets go.
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


_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_orient", False):
    def _grip_rotation(yaw=0.0):
        return yaw.rot if isinstance(yaw, Orient) else _down_grip(yaw)

    _grip_rotation._orient = True
    _demo.grip_rotation = _grip_rotation

TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
LEAN = math.radians(25.0)
# fingers closing along x (yaw pi/2), so the two closed fingertips sit side by side across the FPC and the latch
PRESS_ROT = Orient(axis_angle_matrix(np.array([1.0, 0.0, 0.0]), -LEAN) @ _down_grip(math.pi / 2))  # leaning to -y
LATCH_ROT = Orient(axis_angle_matrix(np.array([1.0, 0.0, 0.0]), LEAN) @ _down_grip(math.pi / 2))  # leaning to +y
PRESS_ON = 0.0006  # aim the fingertips 0.6 mm into the FPC: a light press
DRAG_V = 0.004  # m/s
_live = {"push": 0.0, "side": 0.0, "t0": None, "tl": 0.0, "seated": False}


def _live_reset():
    _live.update(push=0.0, side=0.0, t0=None, tl=0.0, seated=False)


def _live_update(st):
    """Every control step: how far to drag, and how far to steer sideways so the tip meets the slot's centre."""
    q = _trial["q"]
    depth, side, _, _ = insertion(q)
    _live["side"] = float(np.clip(_live["side"] - 0.3 * side, -0.004, 0.004))
    _live["seated"] = depth >= DEPTH - 0.0002


def _clock():
    return float(getattr(_demo, "_plan_clock", 0.0))


def _fingertip_offset(rot):
    """The closed fingertips minus the TCP: TIP_ABOVE_TCP along the approach (the TCP frame's z axis)."""
    return rot[:, 2] * TIP_ABOVE_TCP


PRESS_Y0 = ENTRY_Y + 0.0090  # the press point at the start: on the tip segment, 9 mm out from the mouth
PRESS_Z = FPC_Z + FPC_T / 2 - PRESS_ON


class _Drag:
    """The pressing hand's TCP: the fingertips on the FPC, dragged towards the slot at DRAG_V from ``t0`` until the
    tip is seated (then held), steered sideways to centre the tip."""

    def __init__(self, extra=0.0, dz=0.0):
        self.extra, self.dz = extra, dz

    def __array__(self, dtype=None, copy=None):
        t = _clock()
        if _live["t0"] is None or t < _live["tl"] - 0.5:
            _live.update(t0=t, tl=t, push=0.0)
        if not _live["seated"]:
            _live["push"] = min(0.0065, _live["push"] + DRAG_V * max(0.0, t - _live["tl"]))
        _live["tl"] = t
        p = W((_live["side"], PRESS_Y0 - _live["push"] - self.extra, PRESS_Z + self.dz))
        return np.asarray(p - _fingertip_offset(PRESS_ROT.rot), dtype=dtype or float)


def _latch_tip(phi: float, r: float):
    """Part-frame point at radius r from the latch's hinge, with the latch at angle phi (0 upright, -pi/2 flat)."""
    return np.array([0.0, LATCH_AXIS[1] - r * math.sin(phi), LATCH_AXIS[2] + r * math.cos(phi)])


def _latch_push(phi: float):
    """TCP for the fingertips behind (at -y of) the latch's top at angle phi, pushing it shut."""
    p = _latch_tip(phi, LATCH_H * 0.8) + np.array([0.0, -0.0025 * math.cos(phi), -0.0025 * math.sin(phi)])
    return W(p) - _fingertip_offset(LATCH_ROT.rot)


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    above = W((0.0, PRESS_Y0, PRESS_Z + 0.04)) - _fingertip_offset(PRESS_ROT.rot)
    left = [
        Waypoint(3.0, tuple(above), FINGER_CLOSED, "above the FPC's free end, leaning", PRESS_ROT),
        Waypoint(1.5, _Drag(dz=0.004), FINGER_CLOSED, "down onto the curled FPC", PRESS_ROT),
        Waypoint(1.0, _Drag(), FINGER_CLOSED, "press it flat on the board", PRESS_ROT),
        Waypoint(0.1, _Drag(), FINGER_CLOSED, "slide it into the slot, steering the tip to the slot's centre",
                 PRESS_ROT),
        Waypoint(4.0, _Drag(), FINGER_CLOSED, "slide it into the slot until seated", PRESS_ROT),
        Waypoint(1.0, _Drag(), FINGER_CLOSED, "hold it in", PRESS_ROT),
    ]
    t_in = _dur(left)
    right = [Waypoint(3.0, tuple(_latch_push(0.0) + np.array([0.0, -0.02, 0.03])), FINGER_CLOSED,
                      "above, behind the upright latch, leaning", LATCH_ROT)]
    right.append(Waypoint(max(0.3, t_in - 1.5 - _dur(right)), tuple(_latch_push(0.0) + np.array([0.0, -0.02, 0.03])),
                          FINGER_CLOSED, "wait for the FPC to be in", LATCH_ROT))
    right.append(Waypoint(1.5, tuple(_latch_push(0.0)), FINGER_CLOSED, "fingertips behind the latch", LATCH_ROT))
    for phi in (-0.3, -0.6, -0.9, -1.2, -1.45, LATCH_CLOSED - 0.03):
        right.append(Waypoint(0.6, tuple(_latch_push(phi)), FINGER_CLOSED, "swing the latch shut over the FPC",
                              LATCH_ROT))
    right += [
        Waypoint(0.8, tuple(_latch_push(LATCH_CLOSED - 0.03) + np.array([0.0, -0.01, 0.03])), FINGER_CLOSED,
                 "up off the latch", LATCH_ROT),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_shut = _dur(right) - 2.0
    left.append(Waypoint(max(0.3, t_shut - _dur(left)), _Drag(), FINGER_CLOSED, "hold it in while the latch shuts",
                         PRESS_ROT))
    left += [
        Waypoint(1.0, _Drag(dz=0.03), FINGER_CLOSED, "lift off: the latch holds it", PRESS_ROT),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
