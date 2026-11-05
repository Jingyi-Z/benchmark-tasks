"""On the station table: a small card cage, fixed, with a network line card sitting in its slot, not yet seated.

The cage: an open-top grey steel box on the bench, its open front towards the robots. Inside, guide rails on both side
walls hold a horizontal slot, and at the back a green backplane carries a black connector. The line card lies flat in
the guides: a green circuit board on a steel carrier frame with a black connector along its back edge and, across its
front, an aluminium front panel with six network ports. It has been pushed in until its connector meets the
backplane's, but it is not seated: it sits 7 mm short of home, and a 7 mm gap shows between the panel's back and the
cage's front posts. Pushing on the panel does not move it further: the connector's pins need the levers' force.

The front panel has an ejector lever at each end, 26 cm apart at their pivots. Each is an orange bar on a vertical
pivot pin on a bracket at the panel's end, with a black knob standing up from its free end. Both start open, swung
out to point straight at the robots. A short steel claw on each lever's hub reaches behind a steel catch on the cage's
front post beside it. Closing a lever, swinging its knob in towards the middle of the panel until the bar lies flat
along the panel's front face, levers the card in: each lever's claw draws the card in by an amount proportional to how
far it has closed, 7 mm over the full 90 degrees. The card only goes in as far as the less-closed lever takes it: a
lever closed ahead of the other cocks the card in its guides, and it stops 9 degrees ahead of the other and goes no
further. So the two levers must close together, level with each other, until both lie flat against the panel and the
card is home.

The task: close both ejector levers together until both lie flat against the front panel and the card is seated.

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

CAGE_C = (0.30, 0.31, 0.33)  # grey powder-coated steel
STEEL = (0.62, 0.63, 0.66)
ALU = (0.78, 0.79, 0.81)  # the front panel
FR4 = (0.10, 0.36, 0.18)  # green circuit board
BLACK = (0.06, 0.06, 0.07)
PORT_C = (0.16, 0.16, 0.18)
ORANGE = (0.92, 0.45, 0.08)  # the levers
GOLD = (0.80, 0.66, 0.30)

# -- frame: origin on the table at the panel's front face where the card starts, in the middle of the slot; x away from
# the robots (the way the card goes in), y to their left. Everything below is in this frame. ---------------------------
CX, CY = 0.25, 0.0
TRAVEL = 0.007  # the card starts 7 mm short of home

# The line card, one body on a slide along x: card = 0 where it starts, TRAVEL when home (its connector bottomed on the
# backplane's: the slide's upper limit). Its drive target is set every control step from the levers (see ``_cam``):
# stiff (4 N per mm), it stands for the claws on the catches and the connector's pins, so the card cannot be pushed in
# by hand or pulled out without the levers.
PANEL_T, PANEL_HW, PANEL_Z = 0.003, 0.124, (0.012, 0.058)  # 3 mm aluminium, 248 mm wide, 46 mm tall
BOARD_X, BOARD_HW, BOARD_Z = (0.003, 0.123), 0.114, (0.030, 0.032)  # 120 x 228 x 2 mm FR4
RAIL_HW, RAIL_Z = (0.114, 0.120), (0.028, 0.034)  # the carrier frame's steel side rails, 6 mm square
CONN_X, CONN_HW, CONN_Z = (0.113, 0.127), 0.080, (0.032, 0.044)  # the card's connector, 14 x 160 x 12 mm
SINK_X, SINK_HW, SINK_Z = (0.045, 0.085), 0.035, (0.032, 0.046)  # a heat sink on the board
PORT_Y, PORT_W, PORT_Z = (-0.075, -0.045, -0.015, 0.015, 0.045, 0.075), 0.016, (0.022, 0.036)
CARD_K, CARD_C = 4000.0, 60.0  # N/m, N s/m
CARD_MASS = 0.25

# The ejector levers, one hinge each about the vertical through its pivot pin, riding on the card. Built open (joint
# at 0), pointing at the robots. Closing turns the left lever anticlockwise seen from above (its joint goes to +90
# degrees) and the right one clockwise (its joint to -90): the bar then lies along the panel's front face, 2 mm off it,
# pointing in towards the middle. The joint's limit at 90 degrees is the bar against the panel.
PX, PY = -0.006, 0.130  # the pivots, 6 mm in front of the panel, 26 cm apart
CLOSED = math.pi / 2
LEVER_Z = (0.046, 0.054)  # 8 mm thick, its top 4 mm below the panel's top
LEVER_ZC = sum(LEVER_Z) / 2
LEVER_T = LEVER_Z[1] - LEVER_Z[0]
LEVER_W = 0.008
KNOB_AT = 0.055  # the knob's axis from the pivot
LEVER_L = KNOB_AT + 0.006
HUB_R = 0.006
KNOB_R, KNOB_Z = 0.005, (LEVER_Z[1], 0.088)  # 10 mm across, 34 mm tall, top 30 mm above the panel's top
CLAW_AT, CLAW = 0.008, (0.004, 0.004, LEVER_T)  # the claw on the hub, pointing into the cage when open
LEVER_MASS = 0.02
# The levers' drives (see ``_cam``): a spring of 1.5 N m/rad whose target is the lever's own angle while it is free, so
# it stays where it is left and drags only lightly when turned (about 0.2 N at the knob at the demo's pace), and its
# limit while it is ahead of the other lever by the cocking allowance, so it stalls there.
LEVER_K, LEVER_C = 1.5, 0.002
SLACK = 0.0007  # the card's play in its guides before it cocks: one lever may lead the other by 0.7 mm of travel,
SLACK_ANGLE = SLACK / TRAVEL * CLOSED  # 9 degrees


def _lever(name: str, s: int) -> Joint:
    """The lever on side s (+1 left, -1 right), open."""
    py = s * PY
    return Joint(
        name, kind="hinge", axis="Z", anchor=(PX, py, LEVER_ZC), lower=0.0 if s > 0 else -CLOSED,
        upper=CLOSED if s > 0 else 0.0, initial=0.0, stiffness=LEVER_K, damping=LEVER_C, target=0.0,
        mass=LEVER_MASS, parent="card",
        boxes=[
            ((PX - LEVER_L / 2, py, LEVER_ZC), (LEVER_L, LEVER_W, LEVER_T), ORANGE),  # the bar
            ((PX + CLAW_AT, py, LEVER_ZC), CLAW, STEEL),  # the claw
        ],
        cylinders=[
            ((PX, py, LEVER_ZC), HUB_R, LEVER_T, ORANGE),  # hub
            ((PX - KNOB_AT, py, sum(KNOB_Z) / 2), KNOB_R, KNOB_Z[1] - KNOB_Z[0], BLACK),  # knob
        ],
    )


def _mid(a, b):
    return (a + b) / 2


CARD = Joint(
    "card", kind="slide", axis="X", anchor=(0.06, 0.0, _mid(*BOARD_Z)), lower=-0.030, upper=TRAVEL, initial=0.0,
    stiffness=CARD_K, damping=CARD_C, target=0.0, mass=CARD_MASS,
    boxes=[
        ((PANEL_T / 2, 0.0, _mid(*PANEL_Z)), (PANEL_T, 2 * PANEL_HW, PANEL_Z[1] - PANEL_Z[0]), ALU),
        ((_mid(*BOARD_X), 0.0, _mid(*BOARD_Z)), (BOARD_X[1] - BOARD_X[0], 2 * BOARD_HW, BOARD_Z[1] - BOARD_Z[0]), FR4),
        *[((_mid(*BOARD_X), s * _mid(*RAIL_HW), _mid(*RAIL_Z)),
           (BOARD_X[1] - BOARD_X[0], RAIL_HW[1] - RAIL_HW[0], RAIL_Z[1] - RAIL_Z[0]), STEEL) for s in (1, -1)],
        ((_mid(*CONN_X), 0.0, _mid(*CONN_Z)), (CONN_X[1] - CONN_X[0], 2 * CONN_HW, CONN_Z[1] - CONN_Z[0]), BLACK),
        ((_mid(*SINK_X), 0.0, _mid(*SINK_Z)), (SINK_X[1] - SINK_X[0], 2 * SINK_HW, SINK_Z[1] - SINK_Z[0]), ALU),
        *[((-0.00075, y, _mid(*PORT_Z)), (0.0015, PORT_W, PORT_Z[1] - PORT_Z[0]), PORT_C) for y in PORT_Y],
        # the pivot brackets off the panel's ends, under the levers' hubs
        *[((-0.0045, s * 0.127, 0.043), (0.015, 0.018, 0.006), STEEL) for s in (1, -1)],
    ],
    cylinders=[((PX, s * PY, 0.0485), 0.0025, 0.017, STEEL) for s in (1, -1)],  # pivot pins
)
LEVER_L_J = _lever("lever_left", 1)
LEVER_R_J = _lever("lever_right", -1)

# The cage (fixed): base plate, side walls with guide rails, front posts with the catches, backplane and rear wall.
BASE_X, BASE_HW, BASE_T = (0.010, 0.155), 0.146, 0.008
WALL_X, WALL_HW, WALL_Z = (0.020, 0.149), (0.122, 0.134), (BASE_T, 0.050)
GUIDE_X, GUIDE_HW = (0.010, 0.146), (0.116, 0.122)
GUIDE_LO, GUIDE_HI = (0.024, 0.028), (0.034, 0.038)  # under and over the frame's rails, 6 mm slot
POST_X, POST_HW, POST_Z = (0.010, 0.020), (0.122, 0.146), (BASE_T, 0.058)  # the panel seats against these
CATCH_X, CATCH_HW = (-0.0095, -0.0015), (0.137, 0.146)  # the catch, its back face 1.5 mm in front of the panel
CATCH_ARM_X, CATCH_ARM_HW = (-0.0015, 0.010), (0.141, 0.146)
BP_CONN_X, BP_CONN_HW, BP_CONN_Z = (0.134, 0.146), 0.085, (0.026, 0.048)  # its face: the card's connector home
BACKPLANE_X, BACKPLANE_HW = (0.146, 0.149), 0.122
REAR_X, REAR_Z = (0.149, 0.155), (0.0, 0.060)


def _box(xr, yr, zr, colour):
    return ((_mid(*xr), _mid(*yr), _mid(*zr)), (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), colour)


def _pair(xr, hw, zr, colour):
    """Two boxes mirrored across the slot's centre line, ``hw`` the (inner, outer) half widths."""
    return [_box(xr, (hw[0], hw[1]), zr, colour), _box(xr, (-hw[1], -hw[0]), zr, colour)]


CAGE = part(
    "cage", pos=(CX, CY, TOP), fixed=True, friction=0.5,
    boxes=[
        _box(BASE_X, (-BASE_HW, BASE_HW), (0.0, BASE_T), CAGE_C),
        *_pair(WALL_X, WALL_HW, WALL_Z, CAGE_C),
        *_pair(GUIDE_X, GUIDE_HW, GUIDE_LO, STEEL),
        *_pair(GUIDE_X, GUIDE_HW, GUIDE_HI, STEEL),
        *_pair(POST_X, POST_HW, POST_Z, CAGE_C),
        *_pair(CATCH_X, CATCH_HW, LEVER_Z, STEEL),
        *_pair(CATCH_ARM_X, CATCH_ARM_HW, LEVER_Z, STEEL),
        _box(BP_CONN_X, (-BP_CONN_HW, BP_CONN_HW), BP_CONN_Z, BLACK),
        _box(BACKPLANE_X, (-BACKPLANE_HW, BACKPLANE_HW), (BASE_T, 0.058), FR4),
        _box(REAR_X, (-POST_HW[1], POST_HW[1]), REAR_Z, CAGE_C),
        _box((BP_CONN_X[0] - 0.0005, BP_CONN_X[0]), (-0.070, 0.070), (0.035, 0.039), GOLD),  # contact row
    ],
    joints=[CARD, LEVER_L_J, LEVER_R_J],
)

OBJECTS = [CAGE]


# -- the cam: how the levers draw the card in -----------------------------------------------------------------------
def closing(joints: dict) -> tuple[float, float]:
    """How far each lever has closed (rad, 0 open to 90 degrees flat against the panel): left, right."""
    return float(joints["lever_left"]), float(-joints["lever_right"])


def cam_targets(th_l: float, th_r: float):
    """Drive targets for (card, left lever joint, right lever joint) from the levers' closing angles. The card goes in
    as far as the less-closed lever's claw takes it (TRAVEL in proportion to its angle). A lever may close at most
    SLACK_ANGLE ahead of the other: past that the card would cock in its guides, so its target is held there and the
    lever stalls. Otherwise its target is where it is, so it stays where it is left."""
    th_l = float(np.clip(th_l, 0.0, CLOSED))
    th_r = float(np.clip(th_r, 0.0, CLOSED))
    card = TRAVEL * min(th_l, th_r) / CLOSED
    t_l = min(th_l, th_r + SLACK_ANGLE, CLOSED)
    t_r = min(th_r, th_l + SLACK_ANGLE, CLOSED)
    return card, t_l, -t_r


def _set_targets(env, card, q_l, q_r):
    import torch

    asset = env.unwrapped.scene["cage"]
    names = list(asset.joint_names)
    ids = [names.index(CARD.joint_name), names.index(LEVER_L_J.joint_name), names.index(LEVER_R_J.joint_name)]
    asset.set_joint_position_target(torch.tensor([[card, q_l, q_r]], device=asset.device), joint_ids=ids)


@on_reset
def _cam_reset(env):
    _set_targets(env, 0.0, 0.0, 0.0)
    _live_reset()


@on_step
def _cam(env):
    """Every control step: the card's and the levers' drive targets from where the levers are (targets only)."""
    import labkit

    s = labkit._part_state(env, labkit._SPECS["cage"])
    _set_targets(env, *cam_targets(*closing(s.joints)))
    _live.update(card=float(s.joints["card"]), sim=True)


# -- scoring --------------------------------------------------------------------------------------------------------
CLOSED_TOL = math.radians(3.0)
HOME_TOL = 0.0005
PULLED_OUT = 0.010
SCORE_OVER = "end"


def card_short(joints: dict) -> float:
    """How far the card is short of home (m): 7 mm at the start, 0 seated."""
    return TRAVEL - float(joints["card"])


def lever_closed(th: float) -> bool:
    return th >= CLOSED - CLOSED_TOL


def score(objects):
    """0 if the card has been pulled out more than 10 mm from where it started, or if neither lever is closed (within
    3 degrees of flat against the panel). 0.25 if only one is closed; 0.5 if both are but the card is more than 0.5 mm
    short of home; 1.0 if both are closed and the card is within 0.5 mm of home. Read at the end of the trial."""
    j = objects["cage"].joints
    if float(j["card"]) < -PULLED_OUT:
        return 0.0
    n = sum(lever_closed(th) for th in closing(j))
    if n == 0:
        return 0.0
    if n == 1:
        return 0.25
    return 1.0 if card_short(j) <= HOME_TOL else 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The card's remaining travel to home (m) plus each knob's remaining travel along its arc to closed (m)."""
    j = objects["cage"].joints
    return max(0.0, card_short(j)) + sum(max(0.0, CLOSED - th) * KNOB_AT for th in closing(j))


# The demo's levers are closed (within 3 degrees) with the card home at about 12.0 s; the plan ends at 19.1 s with both
# hands parked. Success must hold 9 s before the trial ends early, so the video plays to its end.
SUCCESS_HOLD_S = 9.0


# -- scripted demo --------------------------------------------------------------------------------------------------
# Both arms do the same thing, mirrored, at the same time. Each comes down with its fingers part-open round its lever's
# knob, the pads either side of it along its path (the fingers close along that path), and drives the knob round its
# arc, the trailing pad's flat inner face pushing it, from open to flat against the panel. The two arcs follow one
# shared schedule, read from the plan's clock every control step (``_Arc``): one smooth swing over 8 s to 2 degrees
# past closed, so the levers stay level with each other and press lightly on their limits at the end. The knob's path
# is aimed at the live card (closed loop): an ``on_step`` hook reads the card's slide every control step, and the knob's
# pivot moves in with it. Offline (no simulator) the card is taken where the cam puts it. The pads hold the knob over
# its top 24 mm, 6 mm above the panel's top, so the leading pad passes over the panel as the lever closes. The fingers
# lean 30 degrees about their closing line, wrists out away from the middle, which keeps the two elbows apart, and
# their closing line runs 8 degrees behind the knob's path, which keeps the wrist roll inside its limit. Then each hand
# holds its lever closed, opens its fingers, lifts straight off and parks.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_Z = TOP + PANEL_Z[1] + 0.006 + TIP_ABOVE_TCP  # pad tips 6 mm above the panel's top, 10 mm above the lever
KNOB_GAP = 0.016  # pads 16 mm apart round the 10 mm knob
KNOB_WIDE = -KNOB_GAP / 2
KNOB_DOWN = -0.010  # 20 mm apart while coming down over it
KNOB_FREE = -0.016  # 32 mm apart to lift off it
KNOB_PRELOAD = 0.001  # the trailing pad 1 mm into the knob
LEAD = KNOB_GAP / 2 - KNOB_R + KNOB_PRELOAD
OVERDRIVE = math.radians(2.0)
PHI_END = CLOSED + OVERDRIVE
SWING_S = 8.0
LEAN = math.radians(30.0)  # the fingers lean 30 degrees about their closing line, the wrist outward, away from the
#                            middle: straight down, the two elbows would touch as the knobs come in (17 mm overlap)
YAW_BACK = math.radians(8.0)  # the closing line turned 8 degrees behind the knob's path all along: closed with it on
#                               the path, the wrist roll would need about 5 degrees past its limit
SIDES = {"left": 1, "right": -1}

_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)  # the plan's time, for waypoints that move with the trial
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7

_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_rot_duck", False):
    def _grip_rotation(yaw=0.0, _prev=_down_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rotation._rot_duck = True
    _demo.grip_rotation = _grip_rotation

_live = {"card": 0.0, "sim": False}
_plan = {"swing0": 1e9}


def _clock() -> float:
    return float(getattr(_demo, "_plan_clock", 0.0))


def _live_reset():
    _live.update(card=0.0, sim=False)
    _demo._plan_clock = 0.0


def _smooth(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def phi_now() -> float:
    """The shared schedule: how far both knobs are being driven round their arcs now (rad)."""
    return PHI_END * _smooth((_clock() - _plan["swing0"]) / SWING_S)


def _card_at(phi: float) -> float:
    """The card's slide: live in the simulator; offline, where the cam puts it with both levers at phi."""
    if _live["sim"]:
        return _live["card"]
    return TRAVEL * min(max(phi, 0.0), CLOSED) / CLOSED


def knob_at(s: int, phi: float, card: float = 0.0):
    """World xy of the knob's axis on side s with the lever closed by phi and the card at ``card``."""
    return (CX + PX + card - KNOB_AT * math.cos(phi), CY + s * (PY - KNOB_AT * math.sin(phi)))


def tangent(s: int, phi: float) -> np.ndarray:
    """The way the knob moves as its lever closes."""
    return np.array([math.sin(phi), -s * math.cos(phi)])


def tcp_at(s: int, phi: float, lead: float = LEAD, card: float | None = None):
    """TCP xy with the knob between the part-open pads, ``lead`` ahead of it along its path: by default so that the
    trailing pad's inner face is KNOB_PRELOAD into the knob (it pushes it)."""
    c = _card_at(phi) if card is None else card
    return tuple(np.array(knob_at(s, phi, c)) + lead * tangent(s, phi))


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


def _rot(s: int, phi: float) -> np.ndarray:
    """The hand's rotation round the knob: the fingers close along its path (along y with the lever open) turned
    YAW_BACK behind it, leaning LEAN about that line, the wrist out away from the middle. With the knob 4 mm behind the
    TCP along the path, the trailing pad's face is then 1 mm into it, 0.6 mm off the pad's centre line."""
    line = phi - YAW_BACK
    t = tangent(s, line)
    return axis_angle_matrix(np.array([t[0], t[1], 0.0]), -s * LEAN) @ _down_grip(s * line)


def _yaw(s: int, phi: float) -> Orient:
    return Orient(_rot(s, phi))


class _KnobAt:
    """The TCP round side s's knob at a fixed phi, on the live card, at height z."""

    def __init__(self, s, phi, lead=LEAD, z=GRIP_Z):
        self.s, self.phi, self.lead, self.z = s, phi, lead, z

    def __array__(self, dtype=None, copy=None):
        return np.asarray((*tcp_at(self.s, self.phi, self.lead), self.z), dtype=dtype or float)


class _Arc:
    """The TCP round side s's knob, driven along the shared schedule, on the live card."""

    def __init__(self, s):
        self.s = s

    def __array__(self, dtype=None, copy=None):
        return np.asarray((*tcp_at(self.s, phi_now()), GRIP_Z), dtype=dtype or float)


class _ArcYaw(float):
    """The hand's rotation along the shared schedule."""

    def __new__(cls, s):
        obj = float.__new__(cls, 0.0)
        obj.s = s
        return obj

    @property
    def rot(self):
        return _rot(self.s, phi_now())


def W(x, y, z):
    """The cage's frame to world."""
    return (CX + x, CY + y, TOP + z)


def _pbox(name, xr, yr, zr, dx=0.0):
    return (name, W(_mid(*xr) + dx, _mid(*yr), _mid(*zr)), (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]))


def _lever_boxes(side: str):
    """Plan-check boxes for side's lever bar and knob, open (card at 0) and closed (card home)."""
    s = SIDES[side]
    out = [
        _pbox(f"{side} lever, open", (PX - LEVER_L, PX), (s * PY - LEVER_W / 2, s * PY + LEVER_W / 2), LEVER_Z),
        _pbox(f"{side} lever, closed", (PX - LEVER_W / 2, PX + LEVER_W / 2),
              tuple(sorted((s * PY, s * (PY - LEVER_L)))), LEVER_Z, TRAVEL),
    ]
    for label, phi, c in (("open", 0.0, 0.0), ("closed", CLOSED, TRAVEL)):
        kx, ky = knob_at(s, phi, c)
        out.append((f"{side} knob, {label}", (kx, ky, TOP + _mid(*KNOB_Z)), (2 * KNOB_R, 2 * KNOB_R,
                                                                            KNOB_Z[1] - KNOB_Z[0])))
    return out


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    _pbox("panel, start to home", (0.0, PANEL_T + TRAVEL), (-PANEL_HW, PANEL_HW), PANEL_Z),
    _pbox("heat sink, home", SINK_X, (-SINK_HW, SINK_HW), SINK_Z, TRAVEL),
    _pbox("card connector, home", CONN_X, (-CONN_HW, CONN_HW), CONN_Z, TRAVEL),
    _pbox("cage walls and posts", (POST_X[0], WALL_X[1]), (-POST_HW[1], POST_HW[1]), POST_Z),
    _pbox("left catch", CATCH_X, CATCH_HW, LEVER_Z),
    _pbox("right catch", CATCH_X, (-CATCH_HW[1], -CATCH_HW[0]), LEVER_Z),
    _pbox("left pivot bracket", (-0.012, 0.003 + TRAVEL), (0.118, 0.136), (0.040, 0.057)),
    _pbox("right pivot bracket", (-0.012, 0.003 + TRAVEL), (-0.136, -0.118), (0.040, 0.057)),
    _pbox("backplane and rear wall", (BP_CONN_X[0], REAR_X[1]), (-POST_HW[1], POST_HW[1]), REAR_Z),
    *_lever_boxes("left"),
    *_lever_boxes("right"),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def _arm(side: str) -> list:
    s = SIDES[side]
    name = f"the {side} lever's knob"
    y0 = _yaw(s, 0.0)
    above = (*tcp_at(s, 0.0, 0.0, 0.0), GRIP_Z + 0.04)
    wps = [
        Waypoint(2.5, above, KNOB_DOWN, f"above {name}, fingers part-open", y0),
        Waypoint(1.0, _KnobAt(s, 0.0, 0.0), KNOB_DOWN, "down round the knob", y0),
        Waypoint(0.4, _KnobAt(s, 0.0, 0.0), KNOB_WIDE, "pads 16 mm apart round the knob", y0),
        Waypoint(0.5, _KnobAt(s, 0.0), KNOB_WIDE, "trailing pad onto the knob", y0),
        Waypoint(0.5, _KnobAt(s, 0.0), KNOB_WIDE, "settle", y0),
        Waypoint(0.05, _Arc(s), KNOB_WIDE, "close both levers together", _ArcYaw(s)),
    ]
    swing0 = _dur(wps)
    y1 = _yaw(s, PHI_END)
    wps += [
        Waypoint(SWING_S, _Arc(s), KNOB_WIDE, "close both levers together", _ArcYaw(s)),
        Waypoint(0.8, _KnobAt(s, PHI_END), KNOB_WIDE, "hold the lever flat against the panel", y1),
        Waypoint(0.5, _KnobAt(s, CLOSED, 0.0), KNOB_FREE, "open the fingers", y1),
        Waypoint(0.8, _KnobAt(s, CLOSED, 0.0, GRIP_Z + 0.04), KNOB_FREE, "up off the knob", y1),
        Waypoint(2.0, LEFT_PARK if s > 0 else RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    return wps, swing0


def demo():
    _live_reset()
    left, t0 = _arm("left")
    right, _ = _arm("right")
    _plan["swing0"] = t0
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    _plan["t_end"] = t_end
    return {"left": left, "right": right}
