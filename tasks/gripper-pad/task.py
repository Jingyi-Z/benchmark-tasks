"""On the station table: a spare robot gripper finger standing in a printed holder, and a tactile sensor pad lying on a
small tray in front of it.

The finger is an aluminium bar, 60 mm long, 20 mm wide and 10 mm thick, lying along y with its gripping face up. Its
root end sits in a slot in the holder's tall head, 5 cm above the bench; its fingertip end, marked with a red band,
sticks out 45 mm past the head to the right, in the air. Behind the head, the holder runs to the left as a flat foot
on the bench and ends in a block. On the finger's gripping face, centred 11 mm from the fingertip, are two locating
holes 10 mm apart along the finger.

The pad is a small tactile sensor: an 18 x 18 mm green board with a 2.4 mm grey rubber face on top, two steel
locating pins 2 mm across sticking 3.5 mm out of its underside, 10 mm apart, and a flat amber cable tab sticking out
8 mm from one edge, in line with the pins. It lies rubber side up on the tray, pins down in the tray's slot, its tab
pointing away from the robots.

Fitting it means: turn the pad so its tab points along the finger, away from the fingertip, set the pins in the holes
and press the pad down until it sits flat on the face. The pins are a light press fit: the pad stops with its
underside 2 mm above the face until it is pushed down with about 4 N, then it goes home. Pressed on its own, the
finger and holder tip over the edge of the head, so one hand holds the holder down by its block while the other
presses.

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

ALU = (0.74, 0.75, 0.77)  # the finger
PLA = (0.93, 0.47, 0.13)  # the orange printed holder
RED = (0.80, 0.08, 0.08)  # the fingertip's paint mark
BLACK = (0.05, 0.05, 0.06)
TRAY_C = (0.12, 0.12, 0.14)  # black anti-static tray
BOARD_C = (0.10, 0.42, 0.20)  # green circuit board
RUBBER_C = (0.55, 0.56, 0.58)  # grey silicone face
STEEL = (0.80, 0.81, 0.83)
TAB_C = (0.85, 0.55, 0.15)  # amber polyimide flex cable

# -- the finger in its holder (one loose body). Frame: origin on the bench under the middle of the two holes, x away
# from the robots, y to their left. The finger runs along y, its fingertip at -y. --------------------------------------
LX, LY = 0.30, -0.10  # the holes' centre, world
Z_FACE = 0.050  # the gripping face, 5 cm above the bench
FINGER_T, FINGER_HW = 0.010, 0.010  # the finger is 10 mm thick and 20 mm wide
FINGER_Z0 = Z_FACE - FINGER_T
TIP_Y, ROOT_Y = -0.011, 0.049  # the fingertip end and the root end; 60 mm long
MARK_Y = TIP_Y + 0.0015  # the red band across the face at the fingertip, 1.5 mm wide
HOLE, HOLE_D, PIN_DY = 0.0034, 0.0045, 0.005  # square holes 3.4 mm across, 4.5 mm deep, at y = +-5 mm
HEAD_Y0, HEAD_Y1, HEAD_HW = 0.034, 0.064, 0.016  # the holder's head: 3 cm long, 32 mm wide, 5 cm tall
FOOT_Y1, FOOT_T = 0.204, 0.008
BLOCK_Y1, BLOCK_H = 0.234, 0.034  # the block at the holder's left end, 22 cm from the holes
HOLDER_MASS = 0.09  # finger and holder

_h0, _h1 = PIN_DY - HOLE / 2, PIN_DY + HOLE / 2  # each hole spans 3.3 to 6.7 mm from the centre along y
_lz = Z_FACE - HOLE_D / 2  # the face layer the holes are cut in
_fz = (FINGER_Z0 + Z_FACE - HOLE_D) / 2  # the solid part of the finger under it
_fw = 2 * FINGER_HW
_side = FINGER_HW - HOLE / 2


def _layer(y0, y1, color=ALU):
    return ((0.0, (y0 + y1) / 2, _lz), (_fw, y1 - y0, HOLE_D), color)


_finger = [
    ((0.0, (TIP_Y + ROOT_Y) / 2, _fz), (_fw, ROOT_Y - TIP_Y, Z_FACE - HOLE_D - FINGER_Z0), ALU),
    _layer(TIP_Y, MARK_Y, RED),
    _layer(MARK_Y, -_h1),
    _layer(-_h0, _h0),
    _layer(_h1, ROOT_Y),
]
for _yc in (-PIN_DY, PIN_DY):  # either side of each hole
    for _sx in (-1, 1):
        _finger.append(((_sx * (HOLE / 2 + _side / 2), _yc, _lz), (_side, HOLE, HOLE_D), ALU))

_holder = [
    ((0.0, (HEAD_Y0 + ROOT_Y) / 2, FINGER_Z0 / 2), (_fw, ROOT_Y - HEAD_Y0, FINGER_Z0), PLA),  # under the finger
    ((FINGER_HW + (HEAD_HW - FINGER_HW) / 2, (HEAD_Y0 + ROOT_Y) / 2, Z_FACE / 2),
     (HEAD_HW - FINGER_HW, ROOT_Y - HEAD_Y0, Z_FACE), PLA),  # the slot's two walls
    ((-FINGER_HW - (HEAD_HW - FINGER_HW) / 2, (HEAD_Y0 + ROOT_Y) / 2, Z_FACE / 2),
     (HEAD_HW - FINGER_HW, ROOT_Y - HEAD_Y0, Z_FACE), PLA),
    ((0.0, (ROOT_Y + HEAD_Y1) / 2, Z_FACE / 2), (2 * HEAD_HW, HEAD_Y1 - ROOT_Y, Z_FACE), PLA),  # behind the finger
    ((0.0, (HEAD_Y1 + FOOT_Y1) / 2, FOOT_T / 2), (2 * HEAD_HW, FOOT_Y1 - HEAD_Y1, FOOT_T), PLA),
    ((0.0, (FOOT_Y1 + BLOCK_Y1) / 2, BLOCK_H / 2), (2 * HEAD_HW, BLOCK_Y1 - FOOT_Y1, BLOCK_H), PLA),
]

# The press fit: a sprung plate under the two holes, which the pins' ends rest on (the holder's body does not collide
# with its own joints). Sprung up, it holds the pad's underside 2 mm above the face; once the pad, in the holes, pushes
# it down 1 mm (4 N), its target drops to the bottom of its travel and stays there, and the pad goes home onto the face.
PIN_R, PIN_L = 0.001, 0.0035
REST_GAP = 0.002  # the unpressed pad's underside above the face
TRAVEL = 0.0025  # latched, the plate is 0.5 mm below the pins' ends with the pad on the face
SEAT_K, SEAT_C = 4000.0, 10.0  # N/m: 1 mm of give, the trigger below, takes 4 N
TRIGGER = 0.001
PLATE_T = 0.0005
PLATE_TOP0 = Z_FACE - (PIN_L - REST_GAP)  # 1.5 mm below the face
SEAT = Joint(
    "seat", kind="slide", axis="Z", anchor=(0.0, 0.0, PLATE_TOP0 - PLATE_T / 2), lower=-TRAVEL, upper=0.0,
    initial=0.0, stiffness=SEAT_K, damping=SEAT_C, target=0.0, mass=0.005,
    boxes=[((0.0, 0.0, PLATE_TOP0 - PLATE_T / 2), (0.0032, 2 * _h1, PLATE_T), BLACK)],
)

HOLDER = part("holder", pos=(LX, LY, TOP), mass=HOLDER_MASS, friction=0.6, boxes=_finger + _holder, joints=[SEAT])

# -- the tray (fixed): a black plate 5 x 5 cm, 8 mm thick, with a slot along x the pad's pins stand in. --------------
PX, PY = 0.22, -0.20  # the pad's centre on the tray, world
TRAY, TRAY_T = 0.050, 0.008
SLOT_W, SLOT_L, SLOT_D = 0.0034, 0.0144, 0.0045
_sz = TRAY_T - SLOT_D
TRAY_PART = part(
    "tray", pos=(PX, PY, TOP), fixed=True, friction=0.6,
    boxes=[
        ((0.0, 0.0, _sz / 2), (TRAY, TRAY, _sz), TRAY_C),
        ((0.0, (TRAY + SLOT_W) / 4, TRAY_T - SLOT_D / 2), (TRAY, (TRAY - SLOT_W) / 2, SLOT_D), TRAY_C),
        ((0.0, -(TRAY + SLOT_W) / 4, TRAY_T - SLOT_D / 2), (TRAY, (TRAY - SLOT_W) / 2, SLOT_D), TRAY_C),
        (((TRAY + SLOT_L) / 4, 0.0, TRAY_T - SLOT_D / 2), ((TRAY - SLOT_L) / 2, SLOT_W, SLOT_D), TRAY_C),
        ((-(TRAY + SLOT_L) / 4, 0.0, TRAY_T - SLOT_D / 2), ((TRAY - SLOT_L) / 2, SLOT_W, SLOT_D), TRAY_C),
    ],
)

# -- the pad (loose). Frame: origin at the middle of its underside; pins and tab along +x. ---------------------------
PAD_W, BOARD_T, RUBBER_T = 0.018, 0.0016, 0.0024
PAD_H = BOARD_T + RUBBER_T  # 4 mm
TAB_L, TAB_W, TAB_T, TAB_Z = 0.008, 0.010, 0.0005, 0.001  # the tab's underside 1 mm above the pad's
PAD = part(
    "pad", pos=(PX, PY, TOP + TRAY_T), mass=0.003, friction=0.8,
    boxes=[
        ((0.0, 0.0, BOARD_T / 2), (PAD_W, PAD_W, BOARD_T), BOARD_C),
        ((0.0, 0.0, BOARD_T + RUBBER_T / 2), (PAD_W, PAD_W, RUBBER_T), RUBBER_C),
        ((PAD_W / 2 + TAB_L / 2, 0.0, TAB_Z + TAB_T / 2), (TAB_L, TAB_W, TAB_T), TAB_C),
    ],
    cylinders=[((sx * PIN_DY, 0.0, -PIN_L / 2), PIN_R, PIN_L, STEEL) for sx in (-1, 1)],
)

OBJECTS = [HOLDER, TRAY_PART, PAD]


# -- where the pad is on the finger ---------------------------------------------------------------------------------
ON_R = 0.001  # in the holes: the pad's centre within 1 mm of the holes' centre,
ON_TILT = 10.0  # tilted no more than 10 degrees from the face,
ON_GAP = 0.003  # and its underside no more than 3 mm above the face (the pins at least 0.5 mm in)
ON_GAP_MIN = -0.001  # nor more than 1 mm below it (under the fingertip, say, rather than on it)
SEATED_GAP = 0.0005  # seated: underside within 0.5 mm of the face
TAB_OK = 20.0  # the tab within 20 degrees of pointing along the finger, away from the fingertip


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def pad_on_finger(holder, pad):
    """(offset of the pad's centre from the holes' centre in m, gap under the pad in m, tilt in degrees, the tab's
    angle from the finger's root direction in rad), in the holder's frame. The tab's angle is anticlockwise seen from
    above, 0 along +y (away from the fingertip)."""
    c = holder.to_local(pad.pos)
    rel = holder.rotation().T @ pad.rotation()
    tilt = float(np.degrees(np.arccos(np.clip(rel[2, 2], -1.0, 1.0))))
    tab = rel[:, 0]
    return float(math.hypot(c[0], c[1])), float(c[2] - Z_FACE), tilt, math.atan2(-tab[0], tab[1])


def _on(r, gap, tilt) -> bool:
    return r <= ON_R and ON_GAP_MIN <= gap <= ON_GAP and tilt <= ON_TILT


_state = {"latched": False}


@on_reset
def _reset_fit(env):
    _state["latched"] = False


@on_step
def _latch(env):
    """The press fit: once the pad is in the holes and has pushed the plate down 1 mm, the plate's target drops to the
    bottom of its travel and stays there."""
    import torch

    import labkit

    holder = labkit._part_state(env, labkit._SPECS["holder"])
    pad = labkit._part_state(env, labkit._SPECS["pad"])
    r, gap, tilt, _ = pad_on_finger(holder, pad)
    if _on(r, gap, tilt) and holder.joints["seat"] <= -TRIGGER:
        _state["latched"] = True
    asset = env.unwrapped.scene["holder"]
    ids = [list(asset.joint_names).index(SEAT.joint_name)]
    target = [-TRAVEL if _state["latched"] else 0.0]
    asset.set_joint_position_target(torch.tensor([target], device=asset.device), joint_ids=ids)


# -- scoring --------------------------------------------------------------------------------------------------------
SUCCESS_HOLD_S = 18.0  # end a trial early only once the pad has stayed fitted for 18 s, so the demo plays to its end


def score(objects):
    """0 unless the pad's pins are in the holes: its centre within 1 mm of the holes' centre, tilted no more than 10
    degrees from the face, its underside no more than 3 mm above the face and no more than 1 mm below it. 0.25 in the
    holes but not seated (underside more than 0.5 mm above the face). Seated: 1.0 if the tab points within 20 degrees
    of along the finger away from the fingertip, 0.5 otherwise (the pins only go in with the tab along the finger, so
    this is the tab turned towards the fingertip)."""
    r, gap, tilt, tab = pad_on_finger(objects["holder"], objects["pad"])
    if not _on(r, gap, tilt):
        return 0.0
    if gap > SEATED_GAP:
        return 0.25
    return 1.0 if abs(math.degrees(tab)) <= TAB_OK else 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The pad's offset from the holes' centre plus its underside's distance from the face (m)."""
    r, gap, _, _ = pad_on_finger(objects["holder"], objects["pad"])
    return r + abs(gap)


# -- scripted demo --------------------------------------------------------------------------------------------------
# Left arm: comes straight down onto the top of the block at the holder's left end with its fingers closed and presses
# it into the bench (about 3 N) until the pad is home; then lifts straight off and parks. It does not grip the block:
# the arm is laterally soft (about 7 mm/N at the pads), so a grip would let any sideways give of the hand drag the
# holder; pressing on the top, the commanded point is leaned against the arm's compliance so the force is vertical.
# Right arm: picks the pad by its two sides without the tab (pad tips 1.5 mm above its underside), lifting it straight
# up off the tray's slot, and turns it a quarter turn anticlockwise on the way, so its tab points along the finger,
# away from the fingertip. From there it is closed loop on the live holder (see ``_track``): every control step the
# hand is aimed at the holes where the holder is now, turned with the holder's yaw, and trimmed by the measured offset
# of the pad from the holes and of its tab from the finger's line. With the pins 2 mm above the face the trim settles
# and is frozen; the pins go straight down into the holes onto the sprung plate. The hand lets go, lifts, closes its
# fingers over the pad's middle and presses the rubber face down slowly (3 mm/s, commanded up to 16 mm below the seat:
# the arm gives about 2.2 mm/N, so the 4 N latch is reached about 10 mm in) until the press fit gives, then eases to a
# light push at the seat, lifts straight off and parks.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_UP = 0.0015  # pad tips 1.5 mm above the pad's underside
PAD_WIDE = -0.013  # pads 26 mm apart round the 18 mm pad
PICK_Z = TOP + TRAY_T + GRIP_UP + TIP_ABOVE_TCP
PICK_HOVER = TOP + 0.06

# The arms' compliance at the pad tips (mm of give per N pushed up on the pads, x, y, z), as measured for the same
# reach on a press fit at (0.30, -0.11); see the servo horn task. A press commanded at (point - F * C) leaves the pads
# at the point with a vertical force F.
C_LEFT = np.array([1.12, -0.73, 2.16]) / 1000.0  # m/N, left arm on the block
C_RIGHT = np.array([0.94, 0.61, 2.20]) / 1000.0  # m/N, right arm at the holes


def pad_z(underside: float) -> float:
    """TCP height, gripping the pad, that puts its underside ``underside`` above the face."""
    return TOP + Z_FACE + underside + GRIP_UP + TIP_ABOVE_TCP


def top_z(underside: float) -> float:
    """TCP height that puts the tips of the closed fingers on the rubber face of a pad ``underside`` above the face."""
    return TOP + Z_FACE + underside + PAD_H + TIP_ABOVE_TCP


CARRY_Z = pad_z(PIN_L + 0.025)
ENTRY_Z = pad_z(PIN_L + 0.002)
REST_Z = pad_z(REST_GAP + 0.0003)
OPEN_UP_Z = top_z(REST_GAP + 0.004)  # open tips up clear of the pad, closed tips 4 mm over its face
TOUCH_Z = top_z(REST_GAP + 0.001)
SEAT_Z = top_z(0.0)
PRESS_MIN_Z = SEAT_Z - 0.016  # the deepest the press is commanded
PRESS_RATE = 0.003  # m/s, down until the fit gives, then back up to the light push
PUSH_Z = SEAT_Z - 0.003  # the light push on the seated pad (about 1.4 N)
PRESS_F_MAX = 8.0  # N, cap on the estimated press force used for the lean
PAD_YAW0 = 0.0  # fingers close along y, on the pad's sides, while the tab points along +x
PAD_YAW1 = math.pi / 2  # and along x once the tab points along +y, away from the fingertip (holder at 0)
BX, BY = LX, LY + (FOOT_Y1 + BLOCK_Y1) / 2  # the block's centre (holder at its start)
BLOCK_TOUCH = TOP + BLOCK_H + TIP_ABOVE_TCP  # closed pad tips on the block's top
BLOCK_HOVER = BLOCK_TOUCH + 0.03
HOLD_F = 3.0  # N, the hold-down force on the block
HOLD = (BX - HOLD_F * C_LEFT[0], BY - HOLD_F * C_LEFT[1], BLOCK_TOUCH - HOLD_F * C_LEFT[2])  # 6.5 mm in, leaned

# The live part of the plan. ``_track`` (below) reads the holder and the pad every control step; the right arm's
# waypoints from the carry on read their pose from ``_live``. Offline (no simulator) the values stay nominal, so the
# plan checks see the holder at its start.
TRACK_GAIN, TRACK_STEP, TRACK_MAX = 0.15, 0.0004, 0.004  # per step: 15% of the pad's offset, at most 0.4 mm; 4 mm all
YAW_GAIN, YAW_STEP, YAW_MAX = 0.15, math.radians(0.5), math.radians(8.0)
_live = {"holes": np.array([LX, LY]), "psi": 0.0, "corr": np.zeros(2), "cyaw": 0.0, "gap": None, "frozen": False}
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
    _live.update(holes=np.array([LX, LY]), psi=0.0, corr=np.zeros(2), cyaw=0.0, gap=None, frozen=False)
    _plan.update(t_latch=None, z_latch=None)
    _demo._plan_clock = 0.0


@on_reset
def _reset_live(env):
    _live_reset()


@on_step
def _track(env):
    """Until the trim is frozen: the holes' position and the holder's yaw, live; from the end of the carry, the trim,
    integrated from the pad's offset from the holes and its tab's angle from the finger's line. Frozen once the pins
    are lowered towards the holes (or the pad is in them), so the press cannot chase a holder it pushes. The gap under
    the pad is read every step, for the press's lean."""
    import labkit

    holder = labkit._part_state(env, labkit._SPECS["holder"])
    pad = labkit._part_state(env, labkit._SPECS["pad"])
    r, gap, tilt, tab = pad_on_finger(holder, pad)
    _live["gap"] = gap
    t = _clock()
    if _live["frozen"] or _on(r, gap, tilt) or t >= _plan["t_freeze"]:
        _live["frozen"] = True
        return
    rot = holder.rotation()
    holes = holder.to_world((0.0, 0.0, Z_FACE))[:2]
    _live.update(holes=holes, psi=float(math.atan2(rot[1, 0], rot[0, 0])))
    if t >= _plan["t_trim"]:
        err = holes - pad.pos[:2]
        corr = _live["corr"] + np.clip(TRACK_GAIN * err, -TRACK_STEP, TRACK_STEP)
        _live["corr"] = np.clip(corr, -TRACK_MAX, TRACK_MAX)
        e = _wrap(-tab)
        _live["cyaw"] = float(np.clip(_live["cyaw"] + np.clip(YAW_GAIN * e, -YAW_STEP, YAW_STEP), -YAW_MAX, YAW_MAX))


def _press_z() -> float:
    """The press's commanded TCP height: down from touching the pad at PRESS_RATE (to PRESS_MIN_Z at most) until the
    press fit gives, then back at PRESS_RATE to the light push. Offline (no latch) it goes all the way down: the plan
    checks see the deepest pose."""
    t = min(_clock(), _plan["t_press1"])
    p = _plan
    if p["t_latch"] is None:
        z = max(PRESS_MIN_Z, TOUCH_Z - PRESS_RATE * max(0.0, t - p["t_press0"]))
        if _state["latched"]:
            p["t_latch"], p["z_latch"] = t, z
        return z
    d = PUSH_Z - p["z_latch"]
    return p["z_latch"] + math.copysign(min(abs(d), PRESS_RATE * max(0.0, t - p["t_latch"])), d)


class _PadAt:
    """The right hand's TCP over the holes, where the holder is now: plus the trim while it holds the pad
    (``trim=True``), or right over the holes' centre for the press with closed fingers. ``z`` is a height, or
    ``"press"`` for the press, which also leans against the arm's compliance by the estimated press force."""

    def __init__(self, z, trim=True):
        self.z, self.trim = z, trim

    def __array__(self, dtype=None, copy=None):
        xy = _live["holes"] + (_live["corr"] if self.trim else 0.0)
        if self.z == "press":
            z = _press_z()
            top = top_z(_live["gap"]) if _live["gap"] is not None else TOUCH_Z  # offline: the pad on the plate
            f = float(np.clip((top - z) / C_RIGHT[2], 0.0, PRESS_F_MAX))
            xy = xy - f * C_RIGHT[:2]
        else:
            z = self.z
        return np.asarray((xy[0], xy[1], z), dtype=dtype or float)


class _PadYaw(float):
    """The right hand's yaw over the holes: the tab along the finger, turned with the holder, plus the trim."""

    def __new__(cls):
        return float.__new__(cls, PAD_YAW1)

    @property
    def rot(self):
        return _prev_grip(PAD_YAW1 + _live["psi"] + _live["cyaw"])


def H(x, y, z):
    """Holder coordinates to world."""
    return (LX + x, LY + y, TOP + z)


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("finger", H(0.0, (TIP_Y + ROOT_Y) / 2, (FINGER_Z0 + Z_FACE) / 2), (_fw, ROOT_Y - TIP_Y, FINGER_T)),
    ("head", H(0.0, (HEAD_Y0 + HEAD_Y1) / 2, Z_FACE / 2), (2 * HEAD_HW, HEAD_Y1 - HEAD_Y0, Z_FACE)),
    ("foot", H(0.0, (HEAD_Y1 + FOOT_Y1) / 2, FOOT_T / 2), (2 * HEAD_HW, FOOT_Y1 - HEAD_Y1, FOOT_T)),
    ("block", H(0.0, (FOOT_Y1 + BLOCK_Y1) / 2, BLOCK_H / 2), (2 * HEAD_HW, BLOCK_Y1 - FOOT_Y1, BLOCK_H)),
    ("tray", (PX, PY, TOP + TRAY_T / 2), (TRAY, TRAY, TRAY_T)),
    ("seated pad", H(0.0, 0.0, Z_FACE + PAD_H / 2), (PAD_W, PAD_W, PAD_H)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    shut = FINGER_CLOSED
    left = [
        Waypoint(2.5, (BX, BY, BLOCK_HOVER), shut, "above the block at the holder's left end, fingers closed"),
        Waypoint(1.2, (BX, BY, BLOCK_TOUCH + 0.003), shut, "straight down to just above the block"),
        Waypoint(0.8, (BX, BY, BLOCK_TOUCH), shut, "pad tips onto the block's top"),
        Waypoint(1.2, HOLD, shut, "press the holder down by its block"),
        Waypoint(0.5, HOLD, shut, "settle"),
    ]
    t_held = _dur(left)
    yaw = _PadYaw()
    right = [
        Waypoint(2.5, (PX, PY, PICK_HOVER), PAD_WIDE, "above the pad", PAD_YAW0),
        Waypoint(1.5, (PX, PY, PICK_Z), PAD_WIDE, "pads either side of the sensor pad", PAD_YAW0),
        Waypoint(0.8, (PX, PY, PICK_Z), FINGER_CLOSED, "grip the pad", PAD_YAW0),
        Waypoint(1.2, (PX, PY, PICK_HOVER), FINGER_CLOSED, "lift it off the tray's slot", PAD_YAW0),
        Waypoint(2.5, _PadAt(CARRY_Z), FINGER_CLOSED, "over the holes, tab turned along the finger", yaw),
    ]
    _plan["t_trim"] = _dur(right)  # the pad is turned and over the holes: the trim starts
    right.append(Waypoint(max(0.5, t_held - _dur(right)), _PadAt(CARRY_Z), FINGER_CLOSED,
                          "wait for the holder to be held", yaw))
    right += [
        Waypoint(1.5, _PadAt(ENTRY_Z), FINGER_CLOSED, "pins 2 mm above the face", yaw),
        Waypoint(2.0, _PadAt(ENTRY_Z), FINGER_CLOSED, "line the pins up on the holes", yaw),
    ]
    _plan["t_freeze"] = _dur(right)
    right += [
        Waypoint(1.5, _PadAt(REST_Z), FINGER_CLOSED, "pins straight down into the holes", yaw),
        Waypoint(0.5, _PadAt(REST_Z), FINGER_CLOSED, "settle", yaw),
        Waypoint(0.8, _PadAt(REST_Z), PAD_WIDE, "let go", yaw),
        Waypoint(1.0, _PadAt(OPEN_UP_Z), PAD_WIDE, "up off the pad", yaw),
        Waypoint(1.0, _PadAt(OPEN_UP_Z, trim=False), FINGER_CLOSED, "close the fingers over the pad's middle", yaw),
        Waypoint(1.0, _PadAt(TOUCH_Z, trim=False), FINGER_CLOSED, "down to just above the rubber face", yaw),
    ]
    _plan["t_press0"] = _dur(right)
    press_s = (TOUCH_Z - PRESS_MIN_Z) / PRESS_RATE + 1.0
    right += [
        Waypoint(0.1, _PadAt("press", trim=False), FINGER_CLOSED, "press the pad down slowly", yaw),
        Waypoint(press_s, _PadAt("press", trim=False), FINGER_CLOSED,
                 "press until the fit gives, then ease to a light push", yaw),
    ]
    _plan["t_press1"] = _dur(right)
    right += [
        Waypoint(1.0, _PadAt(SEAT_Z + 0.001, trim=False), FINGER_CLOSED, "ease off the seated pad", yaw),
        Waypoint(1.2, _PadAt(SEAT_Z + 0.03, trim=False), FINGER_CLOSED, "straight up off the pad", yaw),
    ]
    t_done = _dur(right)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    left.append(Waypoint(t_done - _dur(left), HOLD, shut, "hold the holder down while the pad is pressed"))
    left += [
        Waypoint(0.8, (BX, BY, BLOCK_TOUCH + 0.003), shut, "lift straight off the block"),
        Waypoint(1.0, (BX, BY, BLOCK_HOVER), shut, "up"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    _plan["t_end"] = t_end
    return {"left": left, "right": right}
