"""On the station table: a 4-inch silicon wafer lying on a narrow pedestal, a PTFE wafer transfer paddle lying on the
bench, and a hotplate with a ring of low locating pins on its top plate.

The wafer is 100 mm across and overhangs its pedestal, 30 mm across and 8 cm tall, by 35 mm all round. Its top is the
patterned side and must never be touched. The paddle is a flat PTFE plate 11 x 11.5 cm with low lips along its two
sides, a stop along its back edge and a handle running back from the stop; its front edge is open. The hotplate (shown
at 115 C) is an aluminium block with a top plate 16 x 16 cm; a printed ring of white dots, 4 mm outside a wafer's rim,
marks where the wafer goes in the middle of it.

The task: hold the paddle level, front edge under the wafer's overhang, and nudge the wafer by its rim off the pedestal
onto it; carry it to the hotplate and set it down inside the ring off the paddle's front edge: tilt the paddle forward
with its front edge on the top plate, so the wafer slides down onto the plate, then draw the paddle back out from under
it.
The wafer's top is never touched, no hand touches the hotplate's top plate, and the wafer is never dropped.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

SI = (0.32, 0.34, 0.42)
PTFE = (0.95, 0.95, 0.93)
STAND_C = (0.55, 0.56, 0.58)
ALU = (0.78, 0.79, 0.81)
DARK = (0.16, 0.17, 0.19)
RED = (0.80, 0.15, 0.12)

# -- the wafer on its pedestal ---------------------------------------------------------------------------------------
PED_XY = (0.24, -0.10)
PED_R, PED_TOP = 0.015, 0.080
WAFER_R, WAFER_T = 0.0508, 0.0010  # 4-inch wafer, drawn 1 mm thick (real 0.525 mm), 10 g as real
PEDESTAL = part("pedestal", pos=(*PED_XY, TOP), fixed=True, friction=0.6,
                boxes=[((0.0, 0.0, 0.003), (0.06, 0.06, 0.006), STAND_C)],
                cylinders=[((0.0, 0.0, (0.006 + PED_TOP) / 2), PED_R, PED_TOP - 0.006, STAND_C)])
WAFER = part("wafer", pos=(*PED_XY, TOP + PED_TOP), mass=0.010, friction=0.15,
             cylinders=[((0.0, 0.0, WAFER_T / 2), WAFER_R, WAFER_T, SI)])

# -- the paddle (loose, PTFE). Frame: origin at the plate's centre on its underside; front edge at -y, handle to +y. ---
PW, PL, PT = 0.110, 0.115, 0.003
LIP_W, LIP_H, STOP_H = 0.003, 0.003, 0.005
RISER = (0.024, 0.012, 0.022)  # x, y, height above the plate's top
HANDLE = (0.012, 0.070, 0.010)  # x width, y length, z thickness; its top flush with the riser's
TAB = (0.030, 0.010, 0.030)  # a grip tab across the handle's end, rising 2 cm above it: x, y (gripped across), z
HANDLE_Y0 = PL / 2 + 0.004  # the handle starts here, over the riser
_hz = PT + RISER[2] - HANDLE[2] / 2  # handle's centre height
PADDLE_BOXES = [
    ((0.0, 0.0, PT / 2), (PW, PL, PT), PTFE),
    *[((s * (PW / 2 - LIP_W / 2), 0.0, PT + LIP_H / 2), (LIP_W, PL, LIP_H), PTFE) for s in (1, -1)],
    ((0.0, PL / 2 - LIP_W / 2, PT + STOP_H / 2), (PW, LIP_W, STOP_H), PTFE),
    ((0.0, PL / 2 + RISER[1] / 2 - LIP_W, PT + RISER[2] / 2), (RISER[0], RISER[1], RISER[2]), PTFE),
    ((0.0, HANDLE_Y0 + HANDLE[1] / 2, _hz), HANDLE, PTFE),
    ((0.0, HANDLE_Y0 + HANDLE[1] + TAB[1] / 2, _hz - HANDLE[2] / 2 + TAB[2] / 2), TAB, RED),  # the grip tab
]
PADDLE0 = (0.20, 0.07)  # where it lies at the start: plate centre, flat on the bench
PADDLE = part("paddle", pos=(*PADDLE0, TOP), mass=0.060, friction=0.10, boxes=PADDLE_BOXES)

# -- the hotplate (fixed): an aluminium block; a printed ring of white dots on its top plate marks the target --------
HP_XY = (0.38, -0.08)
HP_HALF, HP_H = 0.080, 0.040
DOT_R, DOT_H, N_DOTS = 0.0012, 0.0002, 24
RING_R = WAFER_R + 0.004 + DOT_R  # the dots' centres: 4 mm outside a centred wafer's rim
HOTPLATE = part(
    "hotplate", pos=(*HP_XY, TOP), fixed=True, friction=0.4,
    boxes=[((0.0, 0.0, HP_H / 2), (2 * HP_HALF, 2 * HP_HALF, HP_H), ALU),
           ((-HP_HALF - 0.004, 0.0, 0.012), (0.008, 0.10, 0.016), DARK),  # front control panel
           ((-HP_HALF - 0.0085, 0.03, 0.012), (0.001, 0.02, 0.006), RED)],  # "hot" lamp
    cylinders=[((RING_R * math.cos(2 * math.pi * (i + 0.5) / N_DOTS), RING_R * math.sin(2 * math.pi * (i + 0.5) / N_DOTS),
                 HP_H + DOT_H / 2), DOT_R, DOT_H, PTFE) for i in range(N_DOTS)],
)

OBJECTS = [PEDESTAL, WAFER, PADDLE, HOTPLATE]

# -- scoring ----------------------------------------------------------------------------------------------------------
HP_TOP = TOP + HP_H
CENTRE_TOL = 0.003  # on target: centre within 3 mm of the ring's centre,
FLAT_DEG = 2.0  # lying flat on the top plate
FLIP_DEG = 40.0  # tilted more than this at any time: dropped or flipped
TOP_TOUCH = 0.006  # a pad tip within 6 mm above the wafer's top, inside its rim less 3 mm: the top was touched
HOT_TOUCH = 0.004  # a pad tip within 4 mm of the top plate, over it: a hand touched the hotplate
HANDS_FAR = 0.050
SUCCESS_HOLD_S = 2.0

_state = {"touched": False, "dropped": False, "hot": False, "loaded": False}


@on_reset
def _reset_state(env):
    _state.update(touched=False, dropped=False, hot=False, loaded=False)


def _tips(env):
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    return [_rt.pad_tips_world(b.scene[n]) - o for n in ("left_robot", "right_robot")]


@on_step
def _watch(env):
    """Latch the violations: the wafer's top touched, the wafer dropped or flipped, a hand on the hotplate's top plate;
    and whether the wafer has been carried off the pedestal on the paddle."""
    w = labkit._part_state(env, labkit._SPECS["wafer"])
    p = np.asarray(w.pos, float)
    top = p[2] + WAFER_T
    if w.tilt_deg() > FLIP_DEG or p[2] < TOP + 0.020:
        _state["dropped"] = True
    for tip in _tips(env):
        if np.hypot(*(tip[:2] - p[:2])) < WAFER_R - 0.003 and top - 0.001 < tip[2] < top + TOP_TOUCH:
            _state["touched"] = True
        if (abs(tip[0] - HP_XY[0]) < HP_HALF and abs(tip[1] - HP_XY[1]) < HP_HALF
                and tip[2] < HP_TOP + HOT_TOUCH):
            _state["hot"] = True
    if np.hypot(*(p[:2] - np.asarray(PED_XY))) > 0.04 and p[2] > TOP + PED_TOP + 0.005:
        _state["loaded"] = True


def on_target(w) -> bool:
    return (w.tilt_deg() <= FLAT_DEG and abs(w.pos[2] - HP_TOP) <= 0.0005
            and float(np.hypot(*(w.pos[:2] - np.asarray(HP_XY)))) <= CENTRE_TOL)


def on_hotplate(w) -> bool:
    return (abs(w.pos[0] - HP_XY[0]) < HP_HALF and abs(w.pos[1] - HP_XY[1]) < HP_HALF
            and HP_TOP - 0.001 <= w.pos[2] <= HP_TOP + 0.005)


def score(objects):
    """0 if the wafer's top was ever touched, the wafer was ever dropped or flipped (tilted past 40 degrees, or below 2
    cm above the table), or a hand ever touched the hotplate's top plate. Otherwise: 1.0 if the wafer lies flat on the
    top plate inside the ring (centre within 3 mm of the ring's centre) with both hands at least 5 cm from it; 0.5 if it
    rests on the hotplate but off the mark (or a hand is still near); 0.25 if it has been carried off the pedestal on
    the paddle; 0 otherwise."""
    w = objects["wafer"]
    if _state["touched"] or _state["dropped"] or _state["hot"]:
        return 0.0
    if on_target(w):
        return 1.0 if _state.get("far", True) else 0.5
    if on_hotplate(w):
        return 0.5
    return 0.25 if _state["loaded"] else 0.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    w = objects["wafer"]
    return float(np.hypot(*(w.pos[:2] - np.asarray(HP_XY)))) + abs(float(w.pos[2]) - HP_TOP)


@on_step
def _hands(env):
    w = labkit._part_state(env, labkit._SPECS["wafer"])
    _state["far"] = all(float(np.linalg.norm(t - np.asarray(w.pos))) >= HANDS_FAR for t in _tips(env))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left arm: picks the paddle up by its handle, pointing down, fingers closing across the handle; holds it level beside
# the pedestal with its front edge under the wafer's overhang and its top 1.5 mm below the wafer. Right arm, fingers
# closed and pointing down, lowers its fingertips beside the wafer's far rim, below the wafer's top, and pushes the
# wafer by its rim towards the paddle until it rests on the paddle, then backs off. Left arm: lifts the paddle off the
# pedestal, tips it back so the wafer slides to the back stop, levels it, carries it to the hotplate, sets its front
# edge on the top plate one wafer radius short of the ring's centre and tilts it forward 15 degrees so the wafer slides
# down onto the plate, then draws the paddle back out from under the wafer and sets it down on the bench. The left hand's aim is trimmed each step by where the paddle is seen to be (a
# camera sees it), so the arm's sag does not move the paddle off its mark.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


class Orient(float):
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
PAD_IN = 0.006
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.15, -0.25, TOP + 0.16)
X = np.array([1.0, 0.0, 0.0])
DOWN_X = _prev_grip(0.0)  # pointing down, fingers closing along y: across the grip tab, which holds the paddle's pitch
GRIP_LOCAL = np.array([0.0, HANDLE_Y0 + HANDLE[1] + TAB[1] / 2, _hz + HANDLE[2] / 2 + 0.013])  # the tab, the
# pad tips 7 mm above the handle (the inner finger must clear it)
H_WIDE, H_HOLD = -0.012, FINGER_CLOSED


def _rx(phi):
    return axis_angle_matrix(X, phi)


def _hand_rot(phi):
    """The left hand's rotation with the paddle tilted phi about x (positive: front edge down)."""
    return Orient(_rx(phi) @ DOWN_X)


def _paddle_tcp(origin, phi=0.0):
    """TCP holding the paddle with its origin (plate centre, underside) at ``origin`` (world), tilted phi."""
    r = _rx(phi)
    grip = np.asarray(origin, float) + r @ GRIP_LOCAL
    a = r @ DOWN_X[:, 2]
    return grip - a * (TIP_ABOVE_TCP - PAD_IN)


# what the demo sees of the paddle: its origin, every step; the left hand's aim is trimmed by the offset between where
# it means the paddle to be and where it is
_live = {"corr": np.zeros(3), "goal": None}


@on_reset
def _reset_live(env):
    _live.update(corr=np.zeros(3), goal=None, prev=None)


@on_step
def _track(env):
    pad = np.asarray(labkit._part_state(env, labkit._SPECS["paddle"]).pos, float)
    prev, _live["prev"] = _live.get("prev"), pad
    if _live["goal"] is None or prev is None:
        return
    err = np.asarray(_live["goal"], float) - pad
    still = np.linalg.norm(pad - prev) < 0.0002  # the paddle has (nearly) stopped: the hand has reached its aim
    if still and np.linalg.norm(err) < 0.03:
        _live["corr"] = np.clip(_live["corr"] + np.clip(0.3 * err, -0.0005, 0.0005), -0.03, 0.03)


class _PaddleAt:
    """The left hand's TCP to put the paddle's origin at ``origin`` tilted ``phi``, trimmed by the live correction."""

    def __init__(self, origin, phi=0.0, track=True):
        self.origin = np.asarray(origin, float)
        self.phi = phi
        self.track = track

    def __array__(self, dtype=None, copy=None):
        if self.track:
            _live["goal"] = self.origin
        return np.asarray(_paddle_tcp(self.origin, self.phi) + _live["corr"], dtype=dtype or float)


def _dur(wps):
    return sum(w.duration for w in wps)


def _wait_until(wps, t, pos, finger, label, orient=0.0):
    wps.append(Waypoint(max(0.3, t - _dur(wps)), pos, finger, label, orient))


LOAD_GAP = 0.0015  # the paddle's top this far below the wafer's underside
FRONT_CLEAR = 0.003  # its front edge this far from the pedestal's side
BACK_TILT = math.radians(-15.0)
CARRY_TILT = math.radians(-6.0)  # carried tipped back a little, the wafer against the stop
SET_TILT = math.radians(15.0)
SET_SHIFT = 0.0  # where the front edge touches down, from one wafer radius short of the ring's centre


def demo():
    _live.update(corr=np.zeros(3), goal=None, prev=None)
    px, py = PED_XY
    wz = TOP + PED_TOP  # the wafer's underside
    # the paddle at the pedestal: front edge under the overhang, top LOAD_GAP below the wafer
    load = np.array([px, py + PED_R + FRONT_CLEAR + PL / 2, wz - LOAD_GAP - PT])
    start = np.array([*PADDLE0, TOP])
    q0 = _hand_rot(0.0)
    left = [
        Waypoint(3.0, _PaddleAt(start + [0, 0, 0.04], track=False), H_WIDE, "above the paddle's handle", q0),
        Waypoint(1.5, _PaddleAt(start, track=False), H_WIDE, "fingers either side of the handle", q0),
        Waypoint(0.8, _PaddleAt(start, track=False), H_HOLD, "grip the handle", q0),
        Waypoint(1.5, _PaddleAt(start + [0, 0, 0.03]), H_HOLD, "lift the paddle level", q0),
        Waypoint(3.0, _PaddleAt(load + [0, 0.02, -0.02]), H_HOLD, "beside the pedestal, below the wafer", q0),
        Waypoint(1.5, _PaddleAt(load + [0, 0, -0.02]), H_HOLD, "front edge under the wafer's overhang", q0),
        Waypoint(2.0, _PaddleAt(load), H_HOLD, "up to just under the wafer", q0),
        Waypoint(1.0, _PaddleAt(load), H_HOLD, "hold it there", q0),
    ]
    t_ready = _dur(left)

    # right: fingertips beside the wafer's far rim, below its top, then push it towards the paddle by its rim
    qr = 0.0  # pointing down, fingers closing along y: the closed tips' 15 mm face pushes along y
    tip_z = wz - 0.002  # the fingertips' bottom 2 mm below the wafer's underside, beside the pedestal
    tcp_z = tip_z + TIP_ABOVE_TCP
    y_rim = py - WAFER_R - 0.004 - 0.006  # 6 mm clear of the rim (the tips are 8 mm thick)
    y_end = py - PED_R - 0.004 - 0.003  # stop 3 mm short of the pedestal's side
    right = [Waypoint(3.0, (px, y_rim - 0.03, tcp_z + 0.06), FINGER_CLOSED, "fingers closed, above the bench", qr)]
    _wait_until(right, t_ready, (px, y_rim - 0.03, tcp_z + 0.06), FINGER_CLOSED, "wait for the paddle", qr)
    right += [
        Waypoint(2.0, (px, y_rim, tcp_z + 0.02), FINGER_CLOSED, "beside the wafer's far rim", qr),
        Waypoint(1.5, (px, y_rim, tcp_z), FINGER_CLOSED, "tips down beside the rim, below the wafer's top", qr),
        Waypoint(4.0, (px, y_end, tcp_z), FINGER_CLOSED, "push the wafer by its rim onto the paddle", qr),
        Waypoint(0.5, (px, y_end, tcp_z), FINGER_CLOSED, "settle", qr),
        Waypoint(1.2, (px, y_end - 0.02, tcp_z), FINGER_CLOSED, "back off the rim", qr),
        Waypoint(1.2, (px, y_end - 0.02, tcp_z + 0.06), FINGER_CLOSED, "up", qr),
        Waypoint(2.5, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_pushed = _dur(right) - 3.7

    # left: lift off the pedestal, tip back so the wafer slides to the stop, level, carry; then set the wafer down off
    # the front edge: the front edge on the top plate one wafer radius short of the ring's centre, tilt forward so the
    # wafer slides down onto the plate, draw the paddle back out from under it
    hx, hy = HP_XY
    front_local = np.array([0.0, -PL / 2, 0.0])  # the front edge's underside, paddle frame
    touch = np.array([hx, hy - WAFER_R + SET_SHIFT, HP_TOP + 0.0005])  # the front edge, on the plate
    above = touch + [0, 0, 0.006]

    def at_front(front, phi):
        return front - _rx(phi) @ front_local

    _wait_until(left, t_pushed, _PaddleAt(load), H_HOLD, "hold the paddle while the wafer is pushed on", q0)
    load_front = load + front_local
    up = load_front + [0, 0, 0.03]
    left += [
        Waypoint(2.0, _PaddleAt(at_front(load_front, BACK_TILT), BACK_TILT), H_HOLD,
                 "tip it back about its front edge: the wafer slides off the pedestal to the stop", _hand_rot(BACK_TILT)),
        Waypoint(1.5, _PaddleAt(at_front(load_front, BACK_TILT), BACK_TILT), H_HOLD, "steady", _hand_rot(BACK_TILT)),
        Waypoint(1.5, _PaddleAt(at_front(up, BACK_TILT), BACK_TILT), H_HOLD, "lift it clear of the pedestal",
                 _hand_rot(BACK_TILT)),
        Waypoint(1.5, _PaddleAt(at_front(up, CARRY_TILT), CARRY_TILT), H_HOLD, "nearly level, the wafer at the stop",
                 _hand_rot(CARRY_TILT)),
        Waypoint(4.0, _PaddleAt(at_front(above + [0, 0, 0.03], CARRY_TILT), CARRY_TILT), H_HOLD,
                 "carry it over the hotplate", _hand_rot(CARRY_TILT)),
        Waypoint(1.5, _PaddleAt(at_front(above, CARRY_TILT), CARRY_TILT), H_HOLD,
                 "front edge just over the plate, short of the ring's centre", _hand_rot(CARRY_TILT)),
    ]
    for f in np.linspace(0.25, 1.0, 4):
        phi = CARRY_TILT + f * (SET_TILT - CARRY_TILT)
        front = above + (touch - above) * f
        left.append(Waypoint(1.0, _PaddleAt(at_front(front, phi), phi), H_HOLD,
                             "tilt it forward, front edge onto the plate: the wafer slides down onto the plate",
                             _hand_rot(phi)))
    left.append(Waypoint(1.5, _PaddleAt(at_front(touch, SET_TILT), SET_TILT), H_HOLD, "let it settle",
                         _hand_rot(SET_TILT)))
    out = touch + [0, 2 * WAFER_R + 0.015, 0]
    left += [
        Waypoint(3.0, _PaddleAt(at_front(out, SET_TILT), SET_TILT, track=False), H_HOLD,
                 "draw the paddle back out from under the wafer, front edge on the plate", _hand_rot(SET_TILT)),
        Waypoint(1.5, _PaddleAt(at_front(out + [0, 0.01, 0.04], 0.0), track=False), H_HOLD, "lift it away, level", q0),
        Waypoint(3.0, _PaddleAt(start + [0, 0, 0.02], track=False), H_HOLD, "back over the bench", q0),
        Waypoint(1.2, _PaddleAt(start + [0, 0, 0.002], track=False), H_HOLD, "set it down", q0),
        Waypoint(0.6, _PaddleAt(start + [0, 0, 0.002], track=False), H_WIDE, "let go", q0),
        Waypoint(1.0, _PaddleAt(start + [0, 0, 0.05], track=False), H_WIDE, "up", q0),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 1.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
