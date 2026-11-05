"""On the station table: the master plate of a manual robot tool changer, bolted on a fixed stand, and to its left a
tray with a spare gripper module standing on it.

The master plate: a grey stand with an aluminium T-slot guide on top, the slot running away from the robots. The slot
is open at its front end, where a short entry ledge lets a module be set down between the guide's low side walls, and
closed at its back end by an end stop. Under the end stop's face a spring plunger sticks out 12 mm: a module pushed
home compresses it, and as soon as nothing holds the module, the plunger pushes it 8 to 10 mm back out along the slot.
To the right of the slot a red locking lever turns on a vertical pivot pin at the front right corner of the plate. Its
handle, ending in a black knob, starts pointing at the robots; its steel hook arm runs out to the right. Swinging the
knob a quarter turn round to the right, until the handle rests against a black stop pin, turns the hook arm back
along the slot's right side, and its tooth goes into a notch in the module's right side. That only fits with the
module pushed fully home against the end stop; once the tooth is in, the plunger cannot push the module out.

The gripper module: a blue block 4 cm square and 3 cm tall with two black jaw fingers on top and, underneath, an
aluminium T-tongue that fits the slot. It stands upright on its tongue on the tray.

The task: slide the module onto the guide until it stops against the end stop and hold it there with one hand; with
the other, swing the lever over to locked. Then let go of both.

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

STAND_C = (0.30, 0.31, 0.33)  # grey powder-coated steel
ALU = (0.78, 0.79, 0.81)  # aluminium
STEEL = (0.62, 0.63, 0.66)
RED = (0.80, 0.12, 0.10)
BLACK = (0.06, 0.06, 0.07)
BRASS = (0.78, 0.62, 0.28)
BLUE = (0.16, 0.30, 0.62)  # the module's anodised body
TRAY_C = (0.55, 0.56, 0.58)

# -- the master plate on its stand (fixed). Frame: origin on the table under the lever's pivot pin in x and under the
# slot's centre line in y; x away from the robots, y to their left. ------------------------------------------------------
CX, CY = 0.255, 0.04
SB = 0.030  # the stand's top: the slot's floor, 3 cm above the table
X_FRONT = -0.035  # the front edge of the stand and of the entry ledge
X_LIP0 = 0.015  # the guide's lips start here: from here on the module's tongue is held down
X_STOP = 0.065  # the end stop's face
STOP_T, STOP_H = 0.010, 0.030
PLATE_Y = 0.030  # the guide is 6 cm wide
CHANNEL_Y = 0.0115  # half width of the slot under the lips (23 mm for the 20 mm tongue)
OPENING_Y = 0.0065  # half width of the opening between the lips (13 mm for the 10 mm neck)
WALL_H = 0.006  # the slot under the lips is 6 mm tall (for the 5 mm tongue)
LIP_T = 0.004
LIP_TOP = SB + WALL_H + LIP_T  # 4 cm above the table
STAND_Y = 0.035

# The spring plunger: a 10 mm steel head on a slide along x, sticking out 12 mm from the end stop's face at rest.
# A module pushed home presses it flush (1.2 N); let go, it pushes the module back out.
PLUNGER_TRAVEL = 0.012
PLUNGER_K, PLUNGER_C = 100.0, 2.0  # N/m, N s/m
PLUNGER_Z = SB + 0.018
PLUNGER = Joint(
    "plunger", kind="slide", axis="X", anchor=(X_STOP - PLUNGER_TRAVEL + 0.005, 0.0, PLUNGER_Z), lower=0.0,
    upper=PLUNGER_TRAVEL, initial=0.0, stiffness=PLUNGER_K, damping=PLUNGER_C, target=0.0, mass=0.01,
    boxes=[((X_STOP - PLUNGER_TRAVEL + 0.005, 0.0, PLUNGER_Z), (0.010, 0.010, 0.010), STEEL)],
    cylinders=[((X_STOP + 0.004, 0.0, PLUNGER_Z), 0.003, 0.010, STEEL, "X")],
)

# The locking lever, built where it starts (unlocked, joint at 0) so that the first rendered frames show it there.
# Pivot pin at (0, -29 mm). Handle: red, 88 mm long, pointing at the robots, a black knob 85 mm from the pin. Hook arm:
# steel, 50 mm long, pointing right, with a tooth on its end pointing back. Locking turns it anticlockwise seen from
# above by 90 degrees (the joint's upper limit): the handle then points right, the arm runs back along the slot's right
# side, 3 mm clear of it, and the tooth points left, 9 mm into the module's side: into its notch.
PIVOT = (0.0, -0.029)
LEVER_Z0, LEVER_T = SB + 0.012, 0.010  # the lever's underside 2 mm above the lips
LEVER_ZC = LEVER_Z0 + LEVER_T / 2
LOCK = math.pi / 2
ARM_L, ARM_W = 0.050, 0.006
TOOTH_X, TOOTH_Y = (0.003, 0.012), (-0.050, -0.044)  # from the pin, unlocked
HANDLE_L, HANDLE_W = 0.088, 0.008
KNOB_R, KNOB_H, KNOB_AT = 0.005, 0.025, 0.085  # the knob: 10 mm across, 25 mm tall, its axis 85 mm from the pin
KNOB_Z0 = LEVER_Z0 + LEVER_T
LEVER_FRICTION, LEVER_DAMP = 0.02, 0.005  # N m: it stays where it is left
STOP_PIN = (0.0085, -0.058)  # from the pin: just behind the handle when locked, outside the hook arm's sweep


def _pv(x, y):
    """Lever coordinates (from the pivot pin, lever unlocked) to the plate's frame."""
    return PIVOT[0] + x, PIVOT[1] + y


LEVER = Joint(
    "lever", kind="hinge", axis="Z", anchor=(*PIVOT, LEVER_ZC), lower=0.0, upper=LOCK, initial=0.0,
    stiffness=0.0, damping=LEVER_DAMP, friction=LEVER_FRICTION, target=0.0, mass=0.03,
    boxes=[
        ((*_pv(-HANDLE_L / 2, 0.0), LEVER_ZC), (HANDLE_L, HANDLE_W, LEVER_T), RED),  # handle
        ((*_pv(0.0, -ARM_L / 2), LEVER_ZC), (ARM_W, ARM_L, LEVER_T), STEEL),  # hook arm
        ((*_pv(sum(TOOTH_X) / 2, sum(TOOTH_Y) / 2), LEVER_ZC), (TOOTH_X[1] - TOOTH_X[0], TOOTH_Y[1] - TOOTH_Y[0],
                                                              LEVER_T), STEEL),  # tooth
    ],
    cylinders=[
        ((*PIVOT, LEVER_ZC), 0.007, LEVER_T, STEEL),  # hub
        ((*_pv(-KNOB_AT, 0.0), KNOB_Z0 + KNOB_H / 2), KNOB_R, KNOB_H, BLACK),  # knob
    ],
)

_sp = _pv(*STOP_PIN)
CHANGER = part(
    "changer", pos=(CX, CY, TOP), fixed=True, friction=0.4,
    boxes=[
        (((X_FRONT + X_STOP + STOP_T) / 2, 0.0, SB / 2), (X_STOP + STOP_T - X_FRONT, 2 * STAND_Y, SB), STAND_C),
        ((0.003, -0.0675, SB - 0.003), (0.026, 0.065, 0.006), STAND_C),  # bracket for the stop pin
        # side walls of the slot, along its whole length; the lips on them from X_LIP0 back
        *[(((X_FRONT + X_STOP) / 2, s * (CHANNEL_Y + PLATE_Y) / 2, SB + WALL_H / 2),
           (X_STOP - X_FRONT, PLATE_Y - CHANNEL_Y, WALL_H), ALU) for s in (1, -1)],
        *[(((X_LIP0 + X_STOP) / 2, s * (OPENING_Y + PLATE_Y) / 2, SB + WALL_H + LIP_T / 2),
           (X_STOP - X_LIP0, PLATE_Y - OPENING_Y, LIP_T), ALU) for s in (1, -1)],
        ((X_STOP + STOP_T / 2, 0.0, SB + STOP_H / 2), (STOP_T, 2 * PLATE_Y, STOP_H), ALU),  # end stop
    ],
    cylinders=[
        ((*PIVOT, (SB + WALL_H + LEVER_Z0) / 2), 0.0035, LEVER_Z0 - SB - WALL_H, STEEL),  # pivot pin
        ((*_sp, (SB + LEVER_Z0 + LEVER_T) / 2), 0.004, LEVER_Z0 + LEVER_T - SB, BLACK),  # stop pin
    ],
    joints=[PLUNGER, LEVER],
)

# -- the gripper module (loose). Frame: origin at the middle of its tongue's underside; x along the slot. --------------
MOD_HALF_L = 0.020  # 4 cm long
TONGUE_W, TONGUE_T = 0.020, 0.005
NECK_W, NECK_TOP = 0.010, 0.011
BODY_HALF_W, BODY_TOP = 0.020, 0.041
JAW = (0.014, 0.008, 0.020)
JAW_Y = 0.009  # jaw centres: 10 mm apart inside, 26 mm outside
MOD_H = BODY_TOP + JAW[2]
NOTCH_X, NOTCH_IN, NOTCH_TOP = (-0.0025, 0.007), -0.016, 0.025  # the notch in its right side, open underneath
MOD_MASS = 0.10
TRAY_XY = (0.20, 0.20)
TRAY = (0.080, 0.080, 0.004)
_hl, _bw = MOD_HALF_L, BODY_HALF_W
MODULE = part(
    "module", pos=(*TRAY_XY, TOP + TRAY[2]), mass=MOD_MASS, friction=0.4,
    boxes=[
        ((0.0, 0.0, TONGUE_T / 2), (2 * _hl, TONGUE_W, TONGUE_T), ALU),  # tongue
        ((0.0, 0.0, (TONGUE_T + NECK_TOP) / 2), (2 * _hl, NECK_W, NECK_TOP - TONGUE_T), ALU),  # neck
        ((0.0, 0.0, (NOTCH_TOP + BODY_TOP) / 2), (2 * _hl, 2 * _bw, BODY_TOP - NOTCH_TOP), BLUE),  # body, upper
        ((0.0, (NOTCH_IN + _bw) / 2, (NECK_TOP + NOTCH_TOP) / 2), (2 * _hl, _bw - NOTCH_IN, NOTCH_TOP - NECK_TOP),
         BLUE),  # body, lower, left of the notch
        (((-_hl + NOTCH_X[0]) / 2, (-_bw + NOTCH_IN) / 2, (NECK_TOP + NOTCH_TOP) / 2),
         (NOTCH_X[0] + _hl, NOTCH_IN + _bw, NOTCH_TOP - NECK_TOP), BLUE),  # in front of the notch
        (((NOTCH_X[1] + _hl) / 2, (-_bw + NOTCH_IN) / 2, (NECK_TOP + NOTCH_TOP) / 2),
         (_hl - NOTCH_X[1], NOTCH_IN + _bw, NOTCH_TOP - NECK_TOP), BLUE),  # behind it
        *[((0.0, s * JAW_Y, BODY_TOP + JAW[2] / 2), JAW, BLACK) for s in (1, -1)],  # jaw fingers
    ],
)
TRAY_PART = part("tray", pos=(*TRAY_XY, TOP), fixed=True, friction=0.6,
                 boxes=[((0.0, 0.0, TRAY[2] / 2), TRAY, TRAY_C)])

OBJECTS = [CHANGER, MODULE, TRAY_PART]

# -- scoring ----------------------------------------------------------------------------------------------------------
ON_TILT = 5.0  # degrees
ON_DZ = 0.002  # the tongue's underside within 2 mm of the slot's floor
ON_DY = 0.0025  # its centre line within 2.5 mm of the slot's
ENGAGED = 0.010  # at least 10 mm of tongue under the lips
HOME_GAP = 0.003  # against the stop: its back within 3 mm of the stop's face
LOCK_TOL = math.radians(2.0)
SCORE_OVER = "end"


def stop_gap(m) -> float:
    """Distance (m) from the module's back face to the end stop's face, along the slot."""
    return float(CX + X_STOP - m.to_world((MOD_HALF_L, 0.0, 0.0))[0])


def on_guide(m) -> bool:
    """The module's tongue is in the slot: tilted no more than 5 degrees, its underside within 2 mm of the slot's
    floor, its centre within 2.5 mm of the slot's centre line, and at least 10 mm of it under the lips."""
    back = m.to_world((MOD_HALF_L, 0.0, 0.0))[0]
    return (m.tilt_deg() <= ON_TILT and abs(float(m.pos[2]) - (TOP + SB)) <= ON_DZ
            and abs(float(m.pos[1]) - CY) <= ON_DY and back >= CX + X_LIP0 + ENGAGED)


def locked(j: dict) -> bool:
    """The lever is within 2 degrees of fully locked (its handle on the stop pin)."""
    return j["lever"] >= LOCK - LOCK_TOL


def score(objects):
    """0 unless the module's tongue is in the slot (see on_guide). 0.25 if it is but the module's back is more than
    3 mm from the end stop's face; 0.5 if it is within 3 mm and the lever is not locked; 1.0 if it is within 3 mm and
    the lever is within 2 degrees of locked. Read at the end of the trial, so the module must still be there."""
    m, j = objects["module"], objects["changer"].joints
    if not on_guide(m):
        return 0.0
    if stop_gap(m) > HOME_GAP:
        return 0.25
    return 1.0 if locked(j) else 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The module's back face from the end stop's face (m, straight line), plus the lever's knob's remaining travel to
    locked (m, along its arc)."""
    m, j = objects["module"], objects["changer"].joints
    back = m.to_world((MOD_HALF_L, 0.0, MOD_H / 2))
    home = np.array([CX + X_STOP, CY, TOP + SB + MOD_H / 2])
    return float(np.linalg.norm(back - home)) + max(0.0, LOCK - j["lever"]) * KNOB_AT


SUCCESS_HOLD_S = 11.0  # end a trial early only after 11 s locked: in the demo the lever locks at about 24.3 s and the
#                        plan ends at 33.6 s with both hands parked, so the video plays to its end


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left arm: grips the module by its two jaw fingers (fingers closing along y on their outer faces, pad tips 4 mm above
# the body), lifts it off the tray and carries it over the entry ledge. With the tongue 3 mm above the side walls it
# centres it on the slot closed loop: an ``on_step`` hook reads the module's position and moves the commanded hand
# position across by part of the remaining offset each control step. Then it sets the module down on the ledge (1 mm
# below the floor: a light press), slides it straight back along the slot and, from where it should meet the end stop,
# goes on closed loop: while the module's back is more than 0.3 mm from the stop's face the commanded position moves on
# by part of the gap each step (the arm gives about 7 mm per N sideways, and the plunger and friction take about 1.6 N),
# up to 20 mm. It holds the module there while the right arm locks, then eases back to its nominal position (the
# plunger pushes the module 2 mm onto the tooth), lets go and lifts off.
# Right arm: comes down with its fingers part-open round the lever's knob, the pads either side of it along its path
# (the fingers close along that path), and drives the knob round its arc in steps of 10 degrees, the trailing pad's flat
# inner face pushing it, to locked (the lever's stop). Its fingers lean 30 degrees about their closing line, the wrist
# out away from the pivot, which keeps its elbow clear of the left arm's. It holds there until the left hand has let
# go, then opens its fingers and lifts off.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_DZ = BODY_TOP + 0.004 + TIP_ABOVE_TCP  # TCP above the tongue's underside: pad tips 4 mm above the body
JAW_WIDE = -0.022  # pads 44 mm apart round the 26 mm pair of jaws
PICK_Z = TOP + TRAY[2] + GRIP_DZ
CARRY_Z = TOP + SB + WALL_H + 0.015 + GRIP_DZ  # the tongue 15 mm above the side walls
HOVER_Z = TOP + SB + WALL_H + 0.003 + GRIP_DZ  # 3 mm above them
SET_Z = TOP + SB - 0.001 + GRIP_DZ  # on the floor, pressed 1 mm
ENTRY_X = CX + X_LIP0 - 0.003 - MOD_HALF_L  # the module's back 3 mm short of the lips
HOME_X = CX + X_STOP - MOD_HALF_L  # its back on the stop's face
TRIM_GAIN, TRIM_STEP, TRIM_MAX = 0.15, 0.0004, 0.006  # per control step: 15% of the offset, at most 0.4 mm; 6 mm all
PUSH_GAIN, PUSH_STEP, PUSH_MAX = 0.15, 0.0004, 0.020
GAP_OK = 0.0003
PUSH_OFFLINE = PUSH_MAX  # offline (no simulator) the plan checks see the push at its deepest

KNOB_Z = TOP + KNOB_Z0 + 0.009 + TIP_ABOVE_TCP  # pad tips 9 mm above the handle: on the knob's top 16 mm
KNOB_GAP = 0.016  # pads 16 mm apart round the 10 mm knob
KNOB_WIDE = -KNOB_GAP / 2
KNOB_DOWN = -0.010  # 20 mm apart while coming down over it
KNOB_FREE = -0.016  # 32 mm apart to lift off it
KNOB_PRELOAD = 0.001  # the trailing pad 1 mm into the knob
SWING_STEPS = [math.radians(d) for d in range(10, 91, 10)]
LEAN = math.radians(30.0)  # the right hand's fingers lean 30 degrees, wrist outward from the pivot: its elbow then
#                            stays clear of the left arm's (straight down, they would touch at the locked pose)

_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)  # the plan's time, for waypoints that move with the trial
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7

_live = {"dx": 0.0, "dy": 0.0, "sim": False}
_plan = {"trim0": 1e9, "trim1": 1e9, "push0": 1e9, "push1": 1e9}


def _clock() -> float:
    return float(getattr(_demo, "_plan_clock", 0.0))


def _live_reset():
    _live.update(dx=0.0, dy=0.0, sim=False)
    _demo._plan_clock = 0.0


@on_reset
def _reset_live(env):
    _live_reset()


@on_step
def _track(env):
    """The left hand's live corrections: across the slot while it hovers over the ledge, along it while it pushes."""
    import labkit

    _live["sim"] = True
    t = _clock()
    m = labkit._part_state(env, labkit._SPECS["module"])
    if _plan["trim0"] <= t <= _plan["trim1"]:
        step = float(np.clip(TRIM_GAIN * (CY - float(m.pos[1])), -TRIM_STEP, TRIM_STEP))
        _live["dy"] = float(np.clip(_live["dy"] + step, -TRIM_MAX, TRIM_MAX))
    if _plan["push0"] <= t <= _plan["push1"]:
        gap = stop_gap(m)
        if gap > GAP_OK:
            _live["dx"] = float(min(PUSH_MAX, _live["dx"] + min(PUSH_STEP, PUSH_GAIN * gap)))


class _LeftAt:
    """The left hand's TCP over the slot, read every control step: x (plus the push, if ``push``), the slot's centre
    line plus the trim, and z."""

    def __init__(self, x, z, push=False):
        self.x, self.z, self.push = x, z, push

    def __array__(self, dtype=None, copy=None):
        dx = (_live["dx"] if _live["sim"] else PUSH_OFFLINE) if self.push else 0.0
        return np.asarray((self.x + dx, CY + _live["dy"], self.z), dtype=dtype or float)


def W(x, y, z):
    """The plate's frame to world."""
    return (CX + x, CY + y, TOP + z)


def knob_at(phi: float):
    """World xy of the knob's axis with the lever turned phi towards locked."""
    return CX + PIVOT[0] - KNOB_AT * math.cos(phi), CY + PIVOT[1] - KNOB_AT * math.sin(phi)


def _push_at(phi: float, lead: float | None = None):
    """TCP xy with the knob between the part-open pads, ``lead`` ahead of it along its path: by default so that the
    trailing pad's inner face is KNOB_PRELOAD into the knob (it pushes it)."""
    if lead is None:
        lead = KNOB_GAP / 2 - KNOB_R + KNOB_PRELOAD
    kx, ky = knob_at(phi)
    return kx + lead * math.sin(phi), ky - lead * math.cos(phi)


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_rot_duck", False):
    def _grip_rotation(yaw=0.0, _prev=_down_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rotation._rot_duck = True
    _demo.grip_rotation = _grip_rotation


def _push_yaw(phi: float) -> Orient:
    """The right hand's rotation round the knob: the fingers close along its path (along y at phi 0) and lean LEAN
    about that line, the wrist out away from the pivot. Leaning about the closing line keeps the pads' faces square to
    the path."""
    u = np.array([math.sin(phi), -math.cos(phi), 0.0])
    return Orient(axis_angle_matrix(u, LEAN) @ _down_grip(phi))


def _lever_box(name, x0, x1, y0, y1, phi):
    """A box of the lever (lever coordinates, unlocked) turned by phi = 0 or LOCK, as a plan-check box."""
    pts = np.array([(x0, y0), (x1, y1)])
    if phi:
        pts = np.array([(-y, x) for x, y in pts])
    lo, hi = pts.min(0), pts.max(0)
    c = W(PIVOT[0] + (lo[0] + hi[0]) / 2, PIVOT[1] + (lo[1] + hi[1]) / 2, LEVER_ZC)
    return (name, c, (hi[0] - lo[0], hi[1] - lo[1], LEVER_T))


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("stand", W((X_FRONT + X_STOP + STOP_T) / 2, 0.0, SB / 2), (X_STOP + STOP_T - X_FRONT, 2 * STAND_Y, SB)),
    ("guide walls and lips", W((X_FRONT + X_STOP) / 2, 0.0, SB + (WALL_H + LIP_T) / 2),
     (X_STOP - X_FRONT, 2 * PLATE_Y, WALL_H + LIP_T)),
    ("end stop", W(X_STOP + STOP_T / 2, 0.0, SB + STOP_H / 2), (STOP_T, 2 * PLATE_Y, STOP_H)),
    ("stop pin", W(*_sp, (SB + LEVER_Z0 + LEVER_T) / 2), (0.008, 0.008, LEVER_Z0 + LEVER_T - SB)),
    _lever_box("handle, unlocked", -HANDLE_L, 0.0, -HANDLE_W / 2, HANDLE_W / 2, 0.0),
    _lever_box("hook arm, unlocked", -ARM_W / 2, ARM_W / 2, -ARM_L, 0.0, 0.0),
    _lever_box("handle, locked", -HANDLE_L, 0.0, -HANDLE_W / 2, HANDLE_W / 2, LOCK),
    _lever_box("hook arm, locked", -ARM_W / 2, ARM_W / 2, -ARM_L, 0.0, LOCK),
    ("knob, unlocked", (*knob_at(0.0), TOP + KNOB_Z0 + KNOB_H / 2), (2 * KNOB_R, 2 * KNOB_R, KNOB_H)),
    ("module body, home", (HOME_X, CY, TOP + SB + (NECK_TOP + BODY_TOP) / 2), (2 * MOD_HALF_L, 2 * BODY_HALF_W,
                                                                             BODY_TOP - NECK_TOP)),
    ("tray", (*TRAY_XY, TOP + TRAY[2] / 2), TRAY),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    tx, ty = TRAY_XY
    left = [
        Waypoint(2.5, (tx, ty, PICK_Z + 0.05), JAW_WIDE, "above the gripper module"),
        Waypoint(1.2, (tx, ty, PICK_Z), JAW_WIDE, "fingers either side of its jaws"),
        Waypoint(0.8, (tx, ty, PICK_Z), FINGER_CLOSED, "grip the jaws"),
        Waypoint(1.2, (tx, ty, CARRY_Z), FINGER_CLOSED, "lift"),
        Waypoint(2.5, _LeftAt(ENTRY_X, CARRY_Z), FINGER_CLOSED, "over the entry ledge"),
        Waypoint(1.5, _LeftAt(ENTRY_X, HOVER_Z), FINGER_CLOSED, "lower: the tongue 3 mm above the side walls"),
        Waypoint(0.05, _LeftAt(ENTRY_X, HOVER_Z), FINGER_CLOSED, "line the tongue up on the slot"),
    ]
    _plan["trim0"] = _dur(left)
    left.append(Waypoint(1.5, _LeftAt(ENTRY_X, HOVER_Z), FINGER_CLOSED, "line the tongue up on the slot"))
    _plan["trim1"] = _dur(left)
    left += [
        Waypoint(1.0, _LeftAt(ENTRY_X, SET_Z), FINGER_CLOSED, "set it down on the ledge"),
        Waypoint(0.5, _LeftAt(ENTRY_X, SET_Z), FINGER_CLOSED, "settle"),
        Waypoint(4.0, _LeftAt(HOME_X, SET_Z), FINGER_CLOSED, "slide it back along the slot"),
    ]
    _plan["push0"] = _dur(left)
    left.append(Waypoint(2.0, _LeftAt(HOME_X, SET_Z, push=True), FINGER_CLOSED, "push it against the end stop"))
    t_held = _dur(left)

    # Right arm: waits clear of the left arm, comes down round the knob as the module reaches the stop, and swings the
    # lever over once the module is held there.
    right = [Waypoint(2.5, RIGHT_PARK, FINGER_OPEN, "park")]
    above = (*_push_at(0.0, 0.0), KNOB_Z + 0.04)
    right.append(Waypoint(max(0.3, _plan["push0"] - 2.5 - _dur(right)), RIGHT_PARK, FINGER_OPEN,
                          "wait for the module to be in the slot"))
    right += [
        Waypoint(2.5, above, KNOB_DOWN, "above the lever's knob, fingers part-open", _push_yaw(0.0)),
        Waypoint(1.0, (*_push_at(0.0, 0.0), KNOB_Z), KNOB_DOWN, "down round the knob", _push_yaw(0.0)),
        Waypoint(0.4, (*_push_at(0.0, 0.0), KNOB_Z), KNOB_WIDE, "pads 16 mm apart round the knob", _push_yaw(0.0)),
        Waypoint(0.5, (*_push_at(0.0), KNOB_Z), KNOB_WIDE, "trailing pad onto the knob", _push_yaw(0.0)),
    ]
    right.append(Waypoint(max(0.3, t_held - _dur(right)), (*_push_at(0.0), KNOB_Z), KNOB_WIDE,
                          "wait for the module to be held at the stop", _push_yaw(0.0)))
    for phi in SWING_STEPS:
        right.append(Waypoint(0.6, (*_push_at(phi), KNOB_Z), KNOB_WIDE, "swing the lever over to locked",
                              _push_yaw(phi)))
    right.append(Waypoint(0.8, (*_push_at(LOCK), KNOB_Z), KNOB_WIDE, "hold the lever on its stop", _push_yaw(LOCK)))
    t_locked = _dur(right)

    left.append(Waypoint(max(0.3, t_locked - _dur(left)), _LeftAt(HOME_X, SET_Z, push=True), FINGER_CLOSED,
                         "hold it against the stop while the lever is locked"))
    _plan["push1"] = _dur(left)
    left += [
        Waypoint(1.5, _LeftAt(HOME_X, SET_Z), FINGER_CLOSED, "ease off: the lever's tooth holds it"),
        Waypoint(0.6, _LeftAt(HOME_X, SET_Z), JAW_WIDE, "let go"),
        Waypoint(1.0, _LeftAt(HOME_X, SET_Z + 0.04), JAW_WIDE, "lift off"),
    ]
    t_left_off = _dur(left)
    left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"))

    right.append(Waypoint(max(0.3, t_left_off - _dur(right)), (*_push_at(LOCK), KNOB_Z), KNOB_WIDE,
                          "hold the lever until the left hand is off", _push_yaw(LOCK)))
    off = (*_push_at(LOCK, 0.0), KNOB_Z)
    right += [
        Waypoint(0.5, off, KNOB_FREE, "open the fingers", _push_yaw(LOCK)),
        Waypoint(0.8, (off[0], off[1], KNOB_Z + 0.04), KNOB_FREE, "up off the knob", _push_yaw(LOCK)),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    _plan["t_locked"], _plan["t_end"] = t_locked, t_end
    return {"left": left, "right": right}
