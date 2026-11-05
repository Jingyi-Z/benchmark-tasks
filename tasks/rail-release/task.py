"""On the station table: a short rack section, fixed, holding a shallow 1U chassis on two slide rails, pushed home and
latched.

The rack: four grey steel posts, joined along each side at top and bottom, its open front towards the robots. A fixed
outer rail runs along the inside of each side. The chassis, 300 mm deep, 220 mm wide and 44 mm tall, rides on an inner
rail along each of its sides, inside the outer rails, and slides out towards the robots. At home its steel ears sit
against the front posts. Each ear carries a grey handle block that sticks out 40 mm in front, and standing up out of a
slot in the top of each block is a blue release tab: a round post 10 mm across and 34 mm tall. The two tabs are 25 cm
apart. Each tab is sprung outwards; pushing it 6 mm in towards the middle of the chassis, until it stops, releases the
latch on its rail.

Each rail latch holds the chassis in over the first 30 mm of its travel. It lets go only while its tab is held in: with
both tabs held in, the chassis slides out; let either tab go while the chassis is still inside those first 30 mm, and
that latch catches it again and it stops where it is (it can still be pushed back in). Past 30 mm the rails run free,
with a little friction, and at 120 mm out the chassis reaches its service position: the rails' end stops, with a
detent that holds it there. A yellow band on the top of each inner rail lines up with the front face of the front posts
at the service position.

The task: press and hold both blue release tabs at once, one with each hand, and draw the chassis out by them until it
stops at the service position. Then let go of both tabs.

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
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

RACK_C = (0.30, 0.31, 0.33)  # grey powder-coated steel
STEEL = (0.62, 0.63, 0.66)
LID = (0.70, 0.71, 0.73)  # the chassis's zinc-plated steel cover
BLACK = (0.06, 0.06, 0.07)
VENT = (0.16, 0.16, 0.18)
HANDLE_C = (0.40, 0.41, 0.43)
BLUE = (0.10, 0.36, 0.85)  # the release tabs
YELLOW = (0.95, 0.80, 0.10)
GREEN = (0.20, 0.85, 0.30)

# -- frame: origin on the table at the front face of the front posts, on the rack's centre line; x away from the robots
# (into the rack), y to their left. Everything below is in this frame. -------------------------------------------------
CX, CY = 0.380, 0.0


def _mid(a, b):
    return (a + b) / 2


def _box(xr, yr, zr, colour):
    return ((_mid(*xr), _mid(*yr), _mid(*zr)), (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), colour)


def _pair(xr, hw, zr, colour):
    """Two boxes mirrored across the centre line, ``hw`` the (inner, outer) half widths."""
    return [_box(xr, (hw[0], hw[1]), zr, colour), _box(xr, (-hw[1], -hw[0]), zr, colour)]


# The rack (fixed): a base plate, four posts, top side rails and the two outer slide rails.
DEPTH = 0.330
BASE_HW, BASE_T = 0.150, 0.010
POST_D, POST_HW, POST_H = 0.020, (0.128, 0.150), 0.110
OUTER_HW, RAIL_Z = (0.118, 0.128), (0.028, 0.056)

# The chassis, one body on a slide along x. Joint value: minus how far out it is (0 home, -120 mm at the service
# position, the slide's lower limit; its upper limit, home, is the ears against the front posts).
BODY_X, BODY_HW, BODY_Z = (0.000, 0.300), 0.110, (0.020, 0.064)
BEZEL_X = (-0.004, 0.000)
EAR_HW = (0.110, 0.150)
HANDLE_X, HANDLE_HW = (-0.056, -0.004), (0.112, 0.146)
INNER_HW = (0.110, 0.118)
BAND_X = (0.118, 0.122)  # the yellow band: at x 0 (the posts' front face) when the chassis is 120 mm out
SERVICE = 0.120
CHASSIS_MASS = 1.5
# Its drive (see ``_rails``): 1000 N/m, damping 6 N s/m, joint friction 0.4 N. While a latch holds it, the target is
# where it was caught (stiff: 1 N moves it 1 mm). While it runs free, the target follows it, half a control step ahead
# so the spring neither drags nor pushes on average. In the detent, the target is the service position.
CHASSIS_K, CHASSIS_C, CHASSIS_F = 1000.0, 6.0, 0.4

# The release tabs, a slide along y each, riding on the chassis, built released (joint at 0). Pressing moves the left
# tab towards -y and the right one towards +y, 6 mm to the slide's limit (the tab bottomed in its slot). A spring of
# 60 N/m pushes each back out: 0.36 N held fully in.
KX, KY = -0.040, 0.125  # the tabs' axes, released: 25 cm apart
TAB_TRAVEL = 0.006
KNOB_R, KNOB_Z = 0.005, (BODY_Z[1], BODY_Z[1] + 0.034)  # 10 mm across, 34 mm tall above the handle block
TAB_K, TAB_C = 60.0, 0.5
SLOT_HW = (0.114, 0.131)

SIDES = {"left": 1, "right": -1}


def _tab(name: str, s: int) -> Joint:
    py = s * KY
    return Joint(
        name, kind="slide", axis="Y", anchor=(KX, py, KNOB_Z[0]), lower=-TAB_TRAVEL if s > 0 else 0.0,
        upper=0.0 if s > 0 else TAB_TRAVEL, initial=0.0, stiffness=TAB_K, damping=TAB_C, target=0.0, mass=0.01,
        parent="chassis",
        boxes=[((KX, py, KNOB_Z[0] + 0.0015), (0.012, 0.012, 0.003), BLUE)],  # the slider under the post
        cylinders=[((KX, py, _mid(*KNOB_Z)), KNOB_R, KNOB_Z[1] - KNOB_Z[0], BLUE)],
    )


CHASSIS = Joint(
    "chassis", kind="slide", axis="X", anchor=(0.150, 0.0, _mid(*BODY_Z)), lower=-SERVICE, upper=0.0, initial=0.0,
    stiffness=CHASSIS_K, damping=CHASSIS_C, friction=CHASSIS_F, target=0.0, mass=CHASSIS_MASS,
    boxes=[
        _box(BODY_X, (-BODY_HW, BODY_HW), BODY_Z, LID),
        _box(BEZEL_X, (-BODY_HW, BODY_HW), BODY_Z, BLACK),
        *_pair(BEZEL_X, EAR_HW, BODY_Z, STEEL),  # ears
        *_pair(HANDLE_X, HANDLE_HW, BODY_Z, HANDLE_C),  # handle blocks
        *_pair((KX - 0.005, KX + 0.005), SLOT_HW, (BODY_Z[1] - 0.001, BODY_Z[1] + 0.0002), BLACK),  # tab slots
        *_pair(BODY_X, INNER_HW, (0.030, 0.054), STEEL),  # inner rails
        *_pair(BAND_X, INNER_HW, (0.054, 0.0548), YELLOW),  # service marks
        *[_box((-0.0048, -0.004), (y - 0.025, y + 0.025), (0.030, 0.054), VENT) for y in (-0.060, 0.0, 0.060)],
        _box((-0.0048, -0.004), (0.090, 0.096), (0.050, 0.056), GREEN),  # power LED
    ],
)
TAB_L = _tab("tab_left", 1)
TAB_R = _tab("tab_right", -1)

RACK = part(
    "rack", pos=(CX, CY, TOP), fixed=True, friction=0.5,
    boxes=[
        _box((0.0, DEPTH), (-BASE_HW, BASE_HW), (0.0, BASE_T), RACK_C),
        *_pair((0.0, POST_D), POST_HW, (0.0, POST_H), RACK_C),  # front posts
        *_pair((DEPTH - POST_D, DEPTH), POST_HW, (0.0, POST_H), RACK_C),  # rear posts
        *_pair((0.0, DEPTH), POST_HW, (POST_H - 0.010, POST_H), RACK_C),  # top side rails
        *_pair((0.0, DEPTH), OUTER_HW, RAIL_Z, STEEL),  # outer slide rails
    ],
    joints=[CHASSIS, TAB_L, TAB_R],
)

OBJECTS = [RACK]


# -- the rail latches and the service detent ---------------------------------------------------------------------------
LATCH_ZONE = 0.030  # the latches hold the chassis over its first 30 mm
TAB_IN = 0.0045  # a tab counts as held in (its latch released) when pushed in 4.5 mm or more
DETENT = 0.003  # within 3 mm of the service position, the detent draws the chassis onto its stop
CONTROL_DT = 0.05

_rail = {"caught": 0.0}


def out_of(joints: dict) -> float:
    """How far the chassis is out from home (m)."""
    return -float(joints["chassis"])


def pressed(joints: dict) -> tuple[float, float]:
    """How far each tab is pushed in (m, 0 released to 6 mm against its stop): left, right."""
    return -float(joints["tab_left"]), float(joints["tab_right"])


def rail_target(out: float, v_out: float, in_l: float, in_r: float) -> float:
    """The chassis's drive target, as how far out (m), from where it is, how fast it moves out (m/s) and how far each
    tab is pushed in. Inside the latch zone with either tab not held in, a latch has it: the target stays where it was
    caught, or wherever it has since been pushed in to. Otherwise it runs free (the target follows it) or, within 3 mm
    of the service position, the detent holds it there."""
    if out < LATCH_ZONE and not (in_l >= TAB_IN and in_r >= TAB_IN):
        caught = _rail["caught"]
        _rail["caught"] = out if caught is None else min(caught, max(out, 0.0))
        return max(_rail["caught"], 0.0)
    _rail["caught"] = None
    if out >= SERVICE - DETENT:
        return SERVICE
    return float(np.clip(out + v_out * CONTROL_DT / 2, 0.0, SERVICE))


def _rack_asset(env):
    return env.unwrapped.scene["rack"]


def _set_chassis_target(env, out_target: float):
    import torch

    asset = _rack_asset(env)
    ids = [list(asset.joint_names).index(CHASSIS.joint_name)]
    asset.set_joint_position_target(torch.tensor([[-out_target]], device=asset.device), joint_ids=ids)


@on_reset
def _rails_reset(env):
    _rail["caught"] = 0.0
    _set_chassis_target(env, 0.0)
    _live_reset()


@on_step
def _rails(env):
    """Every control step: the chassis's drive target from the latches (targets only)."""
    import labkit

    s = labkit._part_state(env, labkit._SPECS["rack"])
    asset = _rack_asset(env)
    i = list(asset.joint_names).index(CHASSIS.joint_name)
    v_out = -float(labkit._np(asset.data.joint_vel)[0][i])
    out, (in_l, in_r) = out_of(s.joints), pressed(s.joints)
    _set_chassis_target(env, rail_target(out, v_out, in_l, in_r))
    _live.update(out=out, tabs={1: in_l, -1: in_r}, sim=True)


# -- scoring -----------------------------------------------------------------------------------------------------------
HALF_OUT = 0.060
SERVICE_TOL = 0.005
RELEASED_TOL = 0.001  # a tab counts as let go when within 1 mm of fully out
SCORE_OVER = "end"


def score(objects):
    """0 if the chassis is 30 mm or less out from home (inside the latch zone). 0.25 if it is out more than 30 mm but
    less than 60 mm. 1.0 if it is within 5 mm of the service position (120 mm out) and both tabs are let go (each
    within 1 mm of fully out). Otherwise (60 mm or more out) 0.5. Read at the end of the trial."""
    j = objects["rack"].joints
    out = out_of(j)
    if out <= LATCH_ZONE:
        return 0.0
    if out < HALF_OUT:
        return 0.25
    if abs(out - SERVICE) <= SERVICE_TOL and max(pressed(j)) <= RELEASED_TOL:
        return 1.0
    return 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The chassis's remaining travel to the service position (m) plus how far each tab is still pushed in (m)."""
    j = objects["rack"].joints
    return max(0.0, SERVICE - out_of(j)) + sum(max(0.0, p) for p in pressed(j))


# The demo's hands open at the service position from 12.9 s and the tabs are out by about 13.5 s; the plan ends at
# 19.1 s with both hands parked. Success must hold 8 s before the trial ends early, so the video plays to its end.
SUCCESS_HOLD_S = 8.0


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Both arms do the same thing, mirrored, at the same time. Each comes down with its fingers part-open round its tab's
# post, pads 16 mm apart, the fingers closing along a line 50 degrees round from straight at the robots towards the
# middle of the chassis. It then moves along that line, so the outer rear pad's flat face pushes the post in towards the
# middle and out towards the robots at once: in, the tab goes to its stop (0.36 N); out, the latched chassis gives about
# 1 mm. Once both tabs are in, the latches let go, and both hands move out 125 mm together, the same push still holding
# each tab against its stop while it draws the chassis out by it, 5 mm past the service position so the chassis lands
# on the end stops and the detent. The arm is compliant (about 7 mm per N sideways), so the hand is aimed into the post:
# 4 mm, and closed loop more (an ``on_step`` hook reads both tabs and the chassis every control step and adds up to
# 8 mm along the push while its tab is not fully in or, once the hands have stopped, while the chassis is short of the
# service position). The fingers lean 20 degrees about their closing line, wrists outward, which keeps the elbows
# apart. Then each hand opens its fingers fully (the tab springs out), pauses, lifts off and parks.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_Z = TOP + BODY_Z[1] + 0.006 + TIP_ABOVE_TCP  # pad tips 6 mm above the handle block, on the post's top 28 mm
POST_GAP = 0.016  # pads 16 mm apart round the 10 mm post
POST_WIDE = -POST_GAP / 2
POST_DOWN = -0.010  # 20 mm apart while coming down over it
PRELOAD = 0.004
LEAD = POST_GAP / 2 - KNOB_R + PRELOAD  # the TCP this far ahead of the post along the push: the pad 4 mm into it
PUSH_ANGLE = math.radians(50.0)  # the push: 50 degrees round from straight at the robots towards the middle
OVERSHOOT = 0.005
LEAN, LEAN_SIGN = math.radians(20.0), 1  # the fingers lean 20 degrees about their closing line, the wrist out
#                                         away from the middle: straight down, the elbows come within 27 mm
PRESS_S, SETTLE_S, PULL_S, HOLD_S = 1.0, 1.0, 5.0, 1.5
EXTRA_STEP, EXTRA_MAX = 0.0003, 0.008  # per control step, at most
TAB_FULL = TAB_TRAVEL - 0.0005
EXTRA_OFFLINE = EXTRA_MAX  # offline (no simulator) the plan checks see the push at its deepest

_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)  # the plan's time, for waypoints that move with the trial
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7

_live = {"out": 0.0, "tabs": {1: 0.0, -1: 0.0}, "extra": {1: 0.0, -1: 0.0}, "sim": False}
_plan = {"press0": 1e9, "pull1": 1e9, "hold1": 1e9}


def _clock() -> float:
    return float(getattr(_demo, "_plan_clock", 0.0))


def _live_reset():
    _live.update(out=0.0, tabs={1: 0.0, -1: 0.0}, extra={1: 0.0, -1: 0.0}, sim=False)
    _demo._plan_clock = 0.0


@on_step
def _track(env):
    """The demo's closed loop: more push along each hand's line while its tab is not fully in, or while the chassis is
    short of the service position after the hands have stopped."""
    t = _clock()
    if not _plan["press0"] <= t <= _plan["hold1"]:
        return
    short = t >= _plan["pull1"] and _live["out"] < SERVICE - 0.001
    for s in (1, -1):
        if _live["tabs"][s] < TAB_FULL or short:
            _live["extra"][s] = min(EXTRA_MAX, _live["extra"][s] + EXTRA_STEP)


def push_dir(s: int) -> np.ndarray:
    """The way the hand on side s pushes its post: towards the robots and in towards the middle."""
    return np.array([-math.cos(PUSH_ANGLE), -s * math.sin(PUSH_ANGLE)])


def post_at(s: int, out: float, pushed_in: float) -> np.ndarray:
    """World xy of side s's tab post with the chassis ``out`` from home and the tab pushed in ``pushed_in``."""
    return np.array([CX + KX - out, CY + s * (KY - pushed_in)])


_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_rot_duck", False):
    def _grip_rotation(yaw=0.0, _prev=_down_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rotation._rot_duck = True
    _demo.grip_rotation = _grip_rotation


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


def _yaw(s: int) -> Orient:
    """The hand's rotation on side s: the fingers close along its push line, leaning LEAN about that line."""
    d = push_dir(s)
    turn = _down_grip(-s * (math.pi / 2 - PUSH_ANGLE))
    return Orient(axis_angle_matrix(np.array([d[0], d[1], 0.0]), LEAN_SIGN * s * LEAN) @ turn)


class _PostAt:
    """The TCP on side s's post, the chassis ``out`` from home and the tab ``pushed_in``, ``lead`` ahead of the post
    along the push (plus the live extra if ``push``), at height z."""

    def __init__(self, s, out, pushed_in=0.0, lead=0.0, push=False, z=GRIP_Z):
        self.s, self.out, self.pushed_in, self.lead, self.push, self.z = s, out, pushed_in, lead, push, z

    def __array__(self, dtype=None, copy=None):
        extra = (_live["extra"][self.s] if _live["sim"] else EXTRA_OFFLINE) if self.push else 0.0
        xy = post_at(self.s, self.out, self.pushed_in) + (self.lead + extra) * push_dir(self.s)
        return np.asarray((*xy, self.z), dtype=dtype or float)


def W(x, y, z):
    """The rack's frame to world."""
    return (CX + x, CY + y, TOP + z)


def _pbox(name, xr, yr, zr):
    return (name, W(_mid(*xr), _mid(*yr), _mid(*zr)), (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]))


def _sweep(xr):
    """An x range on the chassis, swept from home to the service position."""
    return (xr[0] - SERVICE, xr[1])


def _post_boxes(side: str):
    s = SIDES[side]
    out = []
    for label, o, p in (("released, home", 0.0, 0.0), ("in, home", 0.0, TAB_TRAVEL),
                        ("in, service", SERVICE, TAB_TRAVEL), ("released, service", SERVICE, 0.0)):
        x, y = post_at(s, o, p)
        out.append((f"{side} tab post, {label}", (x, y, TOP + _mid(*KNOB_Z)), (2 * KNOB_R, 2 * KNOB_R,
                                                                              KNOB_Z[1] - KNOB_Z[0])))
    return out


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    _pbox("base plate", (0.0, DEPTH), (-BASE_HW, BASE_HW), (0.0, BASE_T)),
    _pbox("left front post", (0.0, POST_D), POST_HW, (0.0, POST_H)),
    _pbox("right front post", (0.0, POST_D), (-POST_HW[1], -POST_HW[0]), (0.0, POST_H)),
    _pbox("left top side rail", (0.0, DEPTH), POST_HW, (POST_H - 0.010, POST_H)),
    _pbox("right top side rail", (0.0, DEPTH), (-POST_HW[1], -POST_HW[0]), (POST_H - 0.010, POST_H)),
    _pbox("chassis body, home to service", _sweep((BEZEL_X[0], BODY_X[1])), (-BODY_HW, BODY_HW), BODY_Z),
    _pbox("left handle block, home to service", _sweep(HANDLE_X), HANDLE_HW, BODY_Z),
    _pbox("right handle block, home to service", _sweep(HANDLE_X), (-HANDLE_HW[1], -HANDLE_HW[0]), BODY_Z),
    _pbox("left ear, home to service", _sweep(BEZEL_X), EAR_HW, BODY_Z),
    _pbox("right ear, home to service", _sweep(BEZEL_X), (-EAR_HW[1], -EAR_HW[0]), BODY_Z),
    *_post_boxes("left"),
    *_post_boxes("right"),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def _arm(side: str):
    s = SIDES[side]
    y = _yaw(s)
    name = f"the {side} release tab's post"
    far = SERVICE + OVERSHOOT
    wps = [
        Waypoint(2.5, _PostAt(s, 0.0, z=GRIP_Z + 0.04), POST_DOWN, f"above {name}, fingers part-open", y),
        Waypoint(1.0, _PostAt(s, 0.0), POST_DOWN, "down round the post", y),
        Waypoint(0.4, _PostAt(s, 0.0), POST_WIDE, "pads 16 mm apart round the post", y),
        Waypoint(0.5, _PostAt(s, 0.0, lead=POST_GAP / 2 - KNOB_R), POST_WIDE, "pad onto the post", y),
    ]
    press0 = _dur(wps)
    wps += [
        Waypoint(PRESS_S, _PostAt(s, 0.0, TAB_TRAVEL, LEAD, True), POST_WIDE, "press both tabs in together", y),
        Waypoint(SETTLE_S, _PostAt(s, 0.0, TAB_TRAVEL, LEAD, True), POST_WIDE, "hold both tabs in", y),
        Waypoint(PULL_S, _PostAt(s, far, TAB_TRAVEL, LEAD, True), POST_WIDE,
                 "draw the chassis out by both tabs together", y),
    ]
    pull1 = _dur(wps)
    wps.append(Waypoint(HOLD_S, _PostAt(s, far, TAB_TRAVEL, LEAD, True), POST_WIDE,
                        "hold it at the service position", y))
    hold1 = _dur(wps)
    wps += [
        Waypoint(0.6, _PostAt(s, far, TAB_TRAVEL, LEAD, True), FINGER_OPEN, "let go: open the fingers", y),
        Waypoint(0.6, _PostAt(s, far, TAB_TRAVEL, LEAD, True), FINGER_OPEN, "pause", y),
        Waypoint(1.0, _PostAt(s, far, TAB_TRAVEL, LEAD, True, GRIP_Z + 0.06), FINGER_OPEN, "lift off", y),
        Waypoint(2.0, LEFT_PARK if s > 0 else RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    return wps, press0, pull1, hold1


def demo():
    _live_reset()
    left, press0, pull1, hold1 = _arm("left")
    right, *_ = _arm("right")
    _plan.update(press0=press0, pull1=pull1, hold1=hold1)
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    _plan["t_end"] = t_end
    return {"left": left, "right": right}
