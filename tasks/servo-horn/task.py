"""On the station table: a 3D-printed arm link for a small robot arm with a hobby servo screwed into its right end, and
the servo's horn lying on the bench in front of it.

The link lies along y, an L-shaped printed bracket 38 cm long. Its left end is a printed block; its middle is a flat
foot on the bench; at the foot's right end an upright rises, and from its top an arm reaches out past the upright,
ending in a shelf that holds the servo by its mounting tabs: the case 2.5 cm above the bench, the output shaft pointing
up, 10 cm out from the upright. The servo has been centred: the white dot on top of its splined shaft points along a
white line printed on the servo's top, which runs from the shaft towards the link's left end. The shaft is not powered
and turns if it is twisted.

The horn is a white plastic arm, 37 mm long, flat underneath, with a square hub around a hole for the shaft. It lies
flat side down, its arm pointing away from the robots. Fitting it means: turn it so its arm points along the white
line, set the hub over the shaft and press it down until the hub sits on the servo. The spline has 25 teeth, so the
horn goes on in 14.4 degree steps, and once on, the shaft turns with it. The press takes about 5 N; pressing on the
servo alone tips the link over the upright's foot edge, so one hand holds the link down while the other presses.

Why the shaft is 10 cm out (statics: link and servo 120 g, W, plus 15 g of shaft and collar on the shaft's axis; the
hub presses at h = 64.5 mm above the bench, d = 100 mm outside the tipping edge; the COM is c = 83 mm and the hold
point b = 240 mm inside it):
- Unheld, a vertical press at the shaft tips the link at 0.8 N.
- The pressing hand also grips the horn, whose hub is on the shaft, so it can push the shaft sideways. A tip about the
  edge moves the shaft out by h and down by d per radian. A hand that stops the sideways part makes the link a ladder
  leaning on the hand: it stays put only while the bench's friction at the edge (mu) carries the hand's sideways push,
  that is while F (d - 5 mm - mu h) <= M + mu W h for a press F, M = 82 N mm being the weights' moment about the edge.
  So even a laterally rigid hand cannot press more than 2.2 N (mu 0.55: the link's 0.6 on the table's 0.5) or 2.9 N
  (mu 0.7) before the link tips or its edge slides; the fit needs 5 N. Tipped, the case's bottom
  reaches the bench only once the hub has dropped 24 mm, deeper than the demo's press goes.
- Held down by 4 N on the block (the demo), the link stays put up to 10.4 N at the shaft (9.2 N at 3.5 N, 8.0 N at
  3 N); the 5 N press needs 1.7 N on the block.

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

PLA = (0.22, 0.42, 0.78)  # blue printed parts
CASE_C = (0.10, 0.11, 0.13)  # the servo's black case
BRASS = (0.80, 0.64, 0.30)  # the spline shaft
BLACK = (0.04, 0.04, 0.05)
WHITE = (0.97, 0.97, 0.97)
HORN_C = (0.95, 0.95, 0.93)

# -- the arm link with its servo (one loose body). Frame: origin on the bench under the shaft, x away from the robots,
# y to their left. The servo's case spans y -10 to +30 mm (the shaft is 10 mm from its right end), x +-10 mm. --------
LX, LY = 0.30, -0.16  # the shaft, world
SB = 0.025  # the case's bottom, above the bench
CASE = (0.020, 0.040, 0.036)
CASE_Y0, CASE_Y1 = -0.010, 0.030
Z_TOP = SB + CASE[2]  # the case's top, 66 mm above the bench
TAB_Z0, TAB_T, TAB_L = SB + 0.0245, 0.0025, 0.007  # mounting tabs at each end, resting on the shelf
SHELF_T, HALF_W = 0.004, 0.016  # the link is 32 mm wide
SHELF_Y0, SHELF_Y1 = -0.020, 0.050
CUT = 0.0005  # the shelf's cut-out clears the case by 0.5 mm
BEAM_T = 0.008  # the arm from the shelf to the upright, its top flush with the shelf's
RISER_Y0, RISER_T = 0.100, 0.010  # the upright's right face: the edge the link tips over, 10 cm from the shaft
FOOT_Y1, FOOT_T = 0.320, 0.008
BLOCK_Y1, BLOCK_H = 0.360, 0.034  # the block at the link's left end, 34 cm from the shaft
LINK_MASS = 0.12  # link and servo

LINE_Y0, LINE_Y1, LINE_W = 0.0065, 0.0295, 0.001  # the centre line on the case's top, along +y

SHAFT_R, SHAFT_H = 0.003, 0.0055  # the 25-tooth spline: 6 mm across, 5.5 mm above the case
TEETH = 25
STEP = 2 * math.pi / TEETH  # 14.4 degrees
SHAFT_K, SHAFT_C = 0.1, 0.003  # the shaft's drive (N m/rad, N m s/rad); its target is set every step (see _drive)
COLLAR_R, COLLAR_T = 0.005, 0.001
COLLAR_TOP0 = Z_TOP + 0.0035  # the hub rests here, 2 mm onto the spline, before it is pressed
TRAVEL = 0.003  # the collar gives 3 mm: the seated hub's underside is 0.5 mm above the case
COLLAR_K, COLLAR_C = 5000.0, 10.0  # N/m: 1 mm of give, the trigger below, takes 5 N
TRIGGER = 0.001

SHAFT = Joint(
    "shaft", kind="hinge", axis="Z", anchor=(0.0, 0.0, Z_TOP), lower=-math.pi, upper=math.pi, initial=0.0,
    stiffness=SHAFT_K, damping=SHAFT_C, target=0.0, mass=0.01,
    cylinders=[((0.0, 0.0, Z_TOP + (SHAFT_H - 0.001) / 2), SHAFT_R, SHAFT_H + 0.001, BRASS)],
    boxes=[((0.0, 0.002, Z_TOP + SHAFT_H - 0.0001), (0.0008, 0.0016, 0.0003), WHITE)],  # the centre dot, along +y
)
COLLAR = Joint(
    "collar", kind="slide", axis="Z", anchor=(0.0, 0.0, COLLAR_TOP0 - COLLAR_T / 2), lower=-TRAVEL, upper=0.0,
    initial=0.0, stiffness=COLLAR_K, damping=COLLAR_C, target=0.0, mass=0.005,
    cylinders=[((0.0, 0.0, COLLAR_TOP0 - COLLAR_T / 2), COLLAR_R, COLLAR_T, BLACK)],
)

_shelf_z = SB + 0.0245 - SHELF_T / 2
_cx, _cy0, _cy1 = CASE[0] / 2 + CUT, CASE_Y0 - CUT, CASE_Y1 + CUT
_beam_z0 = TAB_Z0 - BEAM_T  # the arm's underside, and the upright's top
_riser_y1 = RISER_Y0 + RISER_T
LINK = part(
    "arm_link", pos=(LX, LY, TOP), mass=LINK_MASS, friction=0.6,
    boxes=[
        # the servo's case and its two mounting tabs
        ((0.0, (CASE_Y0 + CASE_Y1) / 2, SB + CASE[2] / 2), CASE, CASE_C),
        ((0.0, CASE_Y0 - TAB_L / 2, TAB_Z0 + TAB_T / 2), (CASE[0], TAB_L, TAB_T), CASE_C),
        ((0.0, CASE_Y1 + TAB_L / 2, TAB_Z0 + TAB_T / 2), (CASE[0], TAB_L, TAB_T), CASE_C),
        ((0.0, (LINE_Y0 + LINE_Y1) / 2, Z_TOP - 0.00005), (LINE_W, LINE_Y1 - LINE_Y0, 0.0003), WHITE),
        # the shelf round the case, the arm out to the upright, the upright, the foot and the block
        ((0.0, (SHELF_Y0 + _cy0) / 2, _shelf_z), (2 * HALF_W, _cy0 - SHELF_Y0, SHELF_T), PLA),
        ((0.0, (_cy1 + SHELF_Y1) / 2, _shelf_z), (2 * HALF_W, SHELF_Y1 - _cy1, SHELF_T), PLA),
        (((_cx + HALF_W) / 2, (_cy0 + _cy1) / 2, _shelf_z), (HALF_W - _cx, _cy1 - _cy0, SHELF_T), PLA),
        ((-(_cx + HALF_W) / 2, (_cy0 + _cy1) / 2, _shelf_z), (HALF_W - _cx, _cy1 - _cy0, SHELF_T), PLA),
        ((0.0, (SHELF_Y1 + _riser_y1) / 2, _beam_z0 + BEAM_T / 2), (2 * HALF_W, _riser_y1 - SHELF_Y1, BEAM_T), PLA),
        ((0.0, (RISER_Y0 + _riser_y1) / 2, _beam_z0 / 2), (2 * HALF_W, RISER_T, _beam_z0), PLA),
        ((0.0, (_riser_y1 + FOOT_Y1) / 2, FOOT_T / 2), (2 * HALF_W, FOOT_Y1 - _riser_y1, FOOT_T), PLA),
        ((0.0, (FOOT_Y1 + BLOCK_Y1) / 2, BLOCK_H / 2), (2 * HALF_W, BLOCK_Y1 - FOOT_Y1, BLOCK_H), PLA),
    ],
    joints=[SHAFT, COLLAR],
)

# -- the horn (loose). Frame: origin on the hub's axis at the horn's flat underside; the arm along +x. --------------
HUB, HOLE, HUB_H = 0.012, 0.0084, 0.006  # square hub 12 mm, square hole 8.4 mm (1.2 mm play round the shaft)
ARM_W, ARM_T, ARM_END = 0.006, 0.0025, 0.028  # the arm plate; a round end of radius 3 mm at x = 28 mm
_wall = (HUB - HOLE) / 2
HX, HY = 0.22, -0.20
HORN = part(
    "horn", pos=(HX, HY, TOP), mass=0.005, friction=0.8,
    boxes=[
        ((HOLE / 2 + _wall / 2, 0.0, HUB_H / 2), (_wall, HUB, HUB_H), HORN_C),
        ((-HOLE / 2 - _wall / 2, 0.0, HUB_H / 2), (_wall, HUB, HUB_H), HORN_C),
        ((0.0, HOLE / 2 + _wall / 2, HUB_H / 2), (HOLE, _wall, HUB_H), HORN_C),
        ((0.0, -HOLE / 2 - _wall / 2, HUB_H / 2), (HOLE, _wall, HUB_H), HORN_C),
        (((HOLE / 2 + ARM_END) / 2, 0.0, ARM_T / 2), (ARM_END - HOLE / 2, ARM_W, ARM_T), HORN_C),
    ],
    cylinders=[((ARM_END, 0.0, ARM_T / 2), ARM_W / 2, ARM_T, HORN_C)],
)

OBJECTS = [LINK, HORN]


# -- where the horn is on the servo ---------------------------------------------------------------------------------
ON_R = 0.002  # on the shaft: hub centred within 2 mm,
ON_TILT = 10.0  # tilted no more than 10 degrees,
ON_GAP = SHAFT_H - 0.001  # and the underside no more than 4.5 mm above the case (the shaft 1 mm into the hub)
SEAT_GAP = COLLAR_TOP0 - TRAVEL - Z_TOP  # 0.5 mm
SEATED_GAP = SEAT_GAP + 0.001  # seated: within 1 mm of the seat


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def horn_on_servo(link, horn):
    """(radial offset of the hub from the shaft in m, gap under the hub in m, tilt in degrees, arm angle from the line
    in rad), in the link's frame. The arm angle is anticlockwise seen from above, 0 along the line (+y)."""
    hub = link.to_local(horn.pos)
    rel = link.rotation().T @ horn.rotation()
    tilt = float(np.degrees(np.arccos(np.clip(rel[2, 2], -1.0, 1.0))))
    arm = rel[:, 0]
    return float(math.hypot(hub[0], hub[1])), float(hub[2] - Z_TOP), tilt, math.atan2(-arm[0], arm[1])


def _on(r, gap, tilt) -> bool:
    return r <= ON_R and gap <= ON_GAP and tilt <= ON_TILT


# -- the spline and the press fit -----------------------------------------------------------------------------------
# The hub is a square hole round a round pin, so the teeth are kept here. When the hub first goes onto the shaft, the
# tooth it meets is fixed (``tooth``: the arm's angle minus the shaft's, in whole steps); while it stays on, the shaft's
# drive target follows the horn at that tooth, so the free shaft turns with the horn and the horn cannot change tooth
# without coming off. Off the shaft, the target is wherever the shaft is, so it stays where it is left. The collar under
# the hub is the interference fit: sprung up, it gives way once the hub on the shaft pushes it down 1 mm (5 N); its
# target then drops to the bottom of its travel and stays there, and the hub follows to its seat.
_state = {"tooth": None, "latched": False}


@on_reset
def _reset_fit(env):
    _state["tooth"], _state["latched"] = None, False


@on_step
def _drive(env):
    import torch

    import labkit

    link = labkit._part_state(env, labkit._SPECS["arm_link"])
    horn = labkit._part_state(env, labkit._SPECS["horn"])
    r, gap, tilt, arm = horn_on_servo(link, horn)
    s = link.joints["shaft"]
    on = _on(r, gap, tilt)
    if on and _state["tooth"] is None:
        _state["tooth"] = round(_wrap(arm - s) / STEP)
    elif not on:
        _state["tooth"] = None
    if _state["tooth"] is None:
        shaft_target = s
    else:
        shaft_target = s + _wrap(arm - s - _state["tooth"] * STEP)
    if on and link.joints["collar"] <= -TRIGGER:
        _state["latched"] = True
    collar_target = -TRAVEL if _state["latched"] else 0.0
    asset = env.unwrapped.scene["arm_link"]
    names = list(asset.joint_names)
    ids = [names.index(SHAFT.joint_name), names.index(COLLAR.joint_name)]
    target = [float(np.clip(shaft_target, -math.pi, math.pi)), collar_target]
    asset.set_joint_position_target(torch.tensor([target], device=asset.device), joint_ids=ids)


# -- scoring --------------------------------------------------------------------------------------------------------
RIGHT_TOOTH = STEP / 2  # within 7.2 degrees: the tooth on the line
ONE_OFF = 1.5 * STEP  # within 21.6 degrees: one tooth off
SUCCESS_HOLD_S = 15.0  # end a trial early only once the horn has stayed on for 15 s, so the demo plays to its end


def zero_error(link, horn) -> float:
    """The arm's angle from the line with the servo at its centre (rad): the arm's angle minus the shaft's."""
    return _wrap(horn_on_servo(link, horn)[3] - link.joints["shaft"])


def score(objects):
    """0 unless the horn is on the shaft (hub centred within 2 mm, tilted no more than 10 degrees, underside no more
    than 4.5 mm above the case, so the shaft is at least 1 mm into the hub). 0.25 on the shaft but not seated
    (underside more than 1.5 mm above the case, 1 mm short of its seat). Seated: by the arm's angle from the line minus
    the angle the shaft's dot has turned from it (where the arm points with the servo centred): 1.0 within 7.2 degrees
    (the tooth on the line), 0.75 within 21.6 degrees (one tooth off), 0.5 further off."""
    link, horn = objects["arm_link"], objects["horn"]
    r, gap, tilt, _ = horn_on_servo(link, horn)
    if not _on(r, gap, tilt):
        return 0.0
    if gap > SEATED_GAP:
        return 0.25
    e = abs(zero_error(link, horn))
    if e <= RIGHT_TOOTH:
        return 1.0
    return 0.75 if e <= ONE_OFF else 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The hub's offset from the shaft plus its height above its seat (m)."""
    r, gap, _, _ = horn_on_servo(objects["arm_link"], objects["horn"])
    return r + max(0.0, gap - SEAT_GAP)


# -- scripted demo --------------------------------------------------------------------------------------------------
# Left arm: comes straight down onto the top of the block at the link's left end with its fingers closed and presses
# it into the bench (about 4 N) until the horn is on; then lifts straight off and parks. It does not grip the block:
# the arm is laterally soft (about 7 mm/N at the pads), so a grip locks the link to the hand and any sideways give of
# the hand drags the link with it (the earlier demo shoved it 10 mm this way). Pressing on the top, the pads can only
# pass on friction from that 4 N, less than the bench holds the link with (that 4 N plus its weight), so the link
# cannot be dragged; and the commanded press point is leaned against the arm's measured compliance so that the pads go
# straight down and the contact force is vertical.
# Right arm: picks the horn by the sides of its hub (pad tips 2 mm above its underside) and turns it a quarter turn
# anticlockwise on the way, so its arm points along the line. From there it is closed loop on the live link (see
# ``_track``): every control step the hand is aimed at the shaft where the link is now, turned with the link's yaw and
# the shaft's dot, and trimmed by the measured offset of the hub from the shaft and of the horn's arm from the dot.
# With the hub 2 mm above the shaft the trim settles and is then frozen; the hub goes straight down over the shaft onto
# the collar and is pressed slowly (3 mm/s, commanded up to 16 mm below the seat: the arm gives about 2 mm/N, so the
# 5 N latch is reached about 11 mm in) until the press fit gives, then eased to a light push at the seat, to no push,
# let go and lifted straight off.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_UP = 0.002  # pad tips 2 mm above the hub's underside
HUB_WIDE = -0.012  # pads 24 mm apart round the 12 mm hub
PICK_Z = TOP + GRIP_UP + TIP_ABOVE_TCP
PICK_HOVER = TOP + 0.06

# The arms' compliance at the pad tips (mm of give per N pushed up on the pads, x, y, z), from the URDF Jacobian and the
# joint drives (80 N m/rad shoulder, 10 N m/rad wrist): pushed up, the pads also move sideways. A press commanded at
# (point - F * C) leaves the pads at the point with a purely vertical force F.
C_LEFT = np.array([1.22, -0.51, 1.88]) / 1000.0  # m/N, left arm on the block
C_RIGHT = np.array([1.04, 0.50, 1.99]) / 1000.0  # m/N, right arm at the shaft


def hub_z(underside: float) -> float:
    """TCP height that puts the hub's underside ``underside`` above the case's top."""
    return TOP + Z_TOP + underside + GRIP_UP + TIP_ABOVE_TCP


CARRY_Z = hub_z(SHAFT_H + 0.025)
ENTRY_Z = hub_z(SHAFT_H + 0.002)
REST_Z = hub_z(COLLAR_TOP0 - Z_TOP + 0.0003)
SEAT_Z = hub_z(SEAT_GAP)
PRESS_MIN_Z = SEAT_Z - 0.016  # the deepest the press is commanded
PRESS_RATE = 0.003  # m/s, down until the fit gives, then back up to the light push
PUSH_Z = SEAT_Z - 0.003  # the light push on the seated hub (about 1.5 N)
PRESS_F_MAX = 8.0  # N, cap on the estimated press force used for the lean
HORN_YAW0 = 0.0  # fingers close along y, on the hub's sides, while the arm points along +x
HORN_YAW1 = math.pi / 2  # and along x once the arm points along +y, the line (link and dot at 0)
BX, BY = LX, LY + (FOOT_Y1 + BLOCK_Y1) / 2  # the block's centre (link at its start)
BLOCK_TOUCH = TOP + BLOCK_H + TIP_ABOVE_TCP  # closed pad tips on the block's top
BLOCK_HOVER = BLOCK_TOUCH + 0.03
HOLD_F = 4.0  # N, the hold-down force on the block (the 5 N press needs 1.7 N there)
HOLD = (BX - HOLD_F * C_LEFT[0], BY - HOLD_F * C_LEFT[1], BLOCK_TOUCH - HOLD_F * C_LEFT[2])  # 7.5 mm in, leaned

# The live part of the plan. ``_track`` (below) reads the link and the horn every control step; the right arm's
# waypoints from the carry on read their pose from ``_live``. Offline (no simulator) the values stay nominal, so the
# plan checks see the link at its start.
TRACK_GAIN, TRACK_STEP, TRACK_MAX = 0.15, 0.0004, 0.004  # per step: 15% of the hub's offset, at most 0.4 mm; 4 mm all
YAW_GAIN, YAW_STEP, YAW_MAX = 0.15, math.radians(0.5), math.radians(8.0)
_live = {"shaft": np.array([LX, LY]), "psi": 0.0, "dot": 0.0, "corr": np.zeros(2), "cyaw": 0.0, "gap": None,
         "frozen": False}
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
    _live.update(shaft=np.array([LX, LY]), psi=0.0, dot=0.0, corr=np.zeros(2), cyaw=0.0, gap=None, frozen=False)
    _plan.update(t_latch=None, z_latch=None)
    _demo._plan_clock = 0.0


@on_reset
def _reset_live(env):
    _live_reset()


@on_step
def _track(env):
    """Until the trim is frozen: the shaft's position, the link's yaw and the dot's angle, live; from the end of the
    carry, the trim, integrated from the hub's offset from the shaft and the horn's arm's angle from the dot. Frozen
    once the hub is lowered over the shaft (or the hub is on it), so the press cannot chase a link it pushes. The gap
    under the hub is read every step, for the press's lean."""
    import labkit

    link = labkit._part_state(env, labkit._SPECS["arm_link"])
    horn = labkit._part_state(env, labkit._SPECS["horn"])
    _, gap, _, arm = horn_on_servo(link, horn)
    _live["gap"] = gap
    t = _clock()
    if _live["frozen"] or _state["tooth"] is not None or t >= _plan["t_freeze"]:
        _live["frozen"] = True
        return
    rot = link.rotation()
    shaft = link.to_world((0.0, 0.0, Z_TOP))[:2]
    dot = link.joints["shaft"]
    _live.update(shaft=shaft, psi=float(math.atan2(rot[1, 0], rot[0, 0])), dot=dot)
    if t >= _plan["t_trim"]:
        err = shaft - horn.pos[:2]
        corr = _live["corr"] + np.clip(TRACK_GAIN * err, -TRACK_STEP, TRACK_STEP)
        _live["corr"] = np.clip(corr, -TRACK_MAX, TRACK_MAX)
        e = _wrap(dot - arm)
        _live["cyaw"] = float(np.clip(_live["cyaw"] + np.clip(YAW_GAIN * e, -YAW_STEP, YAW_STEP), -YAW_MAX, YAW_MAX))


def _press_z() -> float:
    """The press's commanded TCP height: down from the collar at PRESS_RATE (to PRESS_MIN_Z at most) until the press fit
    gives, then back at PRESS_RATE to the light push. Offline (no latch) it goes all the way down: the plan checks see
    the deepest pose."""
    t = min(_clock(), _plan["t_press1"])
    p = _plan
    if p["t_latch"] is None:
        z = max(PRESS_MIN_Z, REST_Z - PRESS_RATE * max(0.0, t - p["t_press0"]))
        if _state["latched"]:
            p["t_latch"], p["z_latch"] = t, z
        return z
    d = PUSH_Z - p["z_latch"]
    return p["z_latch"] + math.copysign(min(abs(d), PRESS_RATE * max(0.0, t - p["t_latch"])), d)


class _HornAt:
    """The right hand's TCP over the shaft: the live shaft position plus the trim, at height ``z`` (a number, or
    ``"press"`` for the press, which also leans against the arm's compliance by the estimated press force)."""

    def __init__(self, z):
        self.z = z

    def __array__(self, dtype=None, copy=None):
        xy = _live["shaft"] + _live["corr"]
        if self.z == "press":
            z = _press_z()
            hub = hub_z(_live["gap"]) if _live["gap"] is not None else REST_Z  # offline: the hub on the collar
            f = float(np.clip((hub - z) / C_RIGHT[2], 0.0, PRESS_F_MAX))
            xy = xy - f * C_RIGHT[:2]
        else:
            z = self.z
        return np.asarray((xy[0], xy[1], z), dtype=dtype or float)


class _HornYaw(float):
    """The right hand's yaw over the shaft: the arm along the line, turned with the link and the dot, plus the trim."""

    def __new__(cls):
        return float.__new__(cls, HORN_YAW1)

    @property
    def rot(self):
        return _prev_grip(HORN_YAW1 + _live["psi"] + _live["dot"] + _live["cyaw"])


def L(x, y, z):
    """Link coordinates to world."""
    return (LX + x, LY + y, TOP + z)


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("servo case", L(0.0, (CASE_Y0 + CASE_Y1) / 2, SB + CASE[2] / 2), CASE),
    ("shelf", L(0.0, (SHELF_Y0 + SHELF_Y1) / 2, _shelf_z), (2 * HALF_W, SHELF_Y1 - SHELF_Y0, SHELF_T)),
    ("arm", L(0.0, (SHELF_Y1 + _riser_y1) / 2, _beam_z0 + BEAM_T / 2), (2 * HALF_W, _riser_y1 - SHELF_Y1, BEAM_T)),
    ("upright", L(0.0, (RISER_Y0 + _riser_y1) / 2, _beam_z0 / 2), (2 * HALF_W, RISER_T, _beam_z0)),
    ("foot", L(0.0, (_riser_y1 + FOOT_Y1) / 2, FOOT_T / 2), (2 * HALF_W, FOOT_Y1 - _riser_y1, FOOT_T)),
    ("block", L(0.0, (FOOT_Y1 + BLOCK_Y1) / 2, BLOCK_H / 2), (2 * HALF_W, BLOCK_Y1 - FOOT_Y1, BLOCK_H)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    shut = FINGER_CLOSED
    left = [
        Waypoint(2.5, (BX, BY, BLOCK_HOVER), shut, "above the block at the link's left end, fingers closed"),
        Waypoint(1.2, (BX, BY, BLOCK_TOUCH + 0.003), shut, "straight down to just above the block"),
        Waypoint(0.8, (BX, BY, BLOCK_TOUCH), shut, "pad tips onto the block's top"),
        Waypoint(1.2, HOLD, shut, "press the link down by its block"),
        Waypoint(0.5, HOLD, shut, "settle"),
    ]
    t_held = _dur(left)
    yaw = _HornYaw()
    right = [
        Waypoint(2.5, (HX, HY, PICK_HOVER), HUB_WIDE, "above the horn", HORN_YAW0),
        Waypoint(1.5, (HX, HY, PICK_Z), HUB_WIDE, "pads either side of the hub", HORN_YAW0),
        Waypoint(0.8, (HX, HY, PICK_Z), FINGER_CLOSED, "grip the hub", HORN_YAW0),
        Waypoint(1.2, (HX, HY, PICK_HOVER), FINGER_CLOSED, "lift", HORN_YAW0),
        Waypoint(2.5, _HornAt(CARRY_Z), FINGER_CLOSED, "over the shaft, arm turned onto the line", yaw),
    ]
    _plan["t_trim"] = _dur(right)  # the horn is turned and over the shaft: the trim starts
    right.append(Waypoint(max(0.5, t_held - _dur(right)), _HornAt(CARRY_Z), FINGER_CLOSED,
                          "wait for the link to be held", yaw))
    right += [
        Waypoint(1.5, _HornAt(ENTRY_Z), FINGER_CLOSED, "hub 2 mm above the shaft", yaw),
        Waypoint(2.0, _HornAt(ENTRY_Z), FINGER_CLOSED, "line the hub up on the shaft", yaw),
    ]
    _plan["t_freeze"] = _dur(right)
    right += [
        Waypoint(1.5, _HornAt(REST_Z), FINGER_CLOSED, "hub straight down over the shaft onto the collar", yaw),
        Waypoint(0.5, _HornAt(REST_Z), FINGER_CLOSED, "settle", yaw),
    ]
    _plan["t_press0"] = _dur(right)
    press_s = (REST_Z - PRESS_MIN_Z) / PRESS_RATE + 1.0
    right += [
        Waypoint(0.1, _HornAt("press"), FINGER_CLOSED, "press the horn down slowly", yaw),
        Waypoint(press_s, _HornAt("press"), FINGER_CLOSED, "press until the fit gives, then ease to a light push",
                 yaw),
    ]
    _plan["t_press1"] = _dur(right)
    right += [
        Waypoint(1.0, _HornAt(SEAT_Z), FINGER_CLOSED, "ease off at the seat", yaw),
        Waypoint(0.8, _HornAt(SEAT_Z), HUB_WIDE, "let go", yaw),
        Waypoint(1.2, _HornAt(CARRY_Z), HUB_WIDE, "straight up off the horn", yaw),
    ]
    t_done = _dur(right)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    left.append(Waypoint(t_done - _dur(left), HOLD, shut, "hold the link down while the horn is pressed"))
    left += [
        Waypoint(0.8, (BX, BY, BLOCK_TOUCH + 0.003), shut, "lift straight off the block"),
        Waypoint(1.0, (BX, BY, BLOCK_HOVER), shut, "up"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    return {"left": left, "right": right}
