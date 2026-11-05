"""On the station table: a 3D-printed arm link for a small robot arm with a hobby servo screwed into its right end, and
the servo's horn lying on the bench in front of it.

The link lies along y. Its left end is a printed block; its middle is a flat foot on the bench; at its right end an
upright and a shelf hold the servo by its mounting tabs, 3 cm above the bench and out past the end of the foot, with
the servo's output shaft pointing up. The servo has been centred: the white dot on top of its splined shaft points
along a white line printed on the servo's top, which runs from the shaft towards the link's left end. The shaft is not
powered and turns if it is twisted.

The horn is a white plastic arm, 37 mm long, flat underneath, with a square hub around a hole for the shaft. It lies
flat side down, its arm pointing away from the robots. Fitting it means: turn it so its arm points along the white
line, set the hub over the shaft and press it down until the hub sits on the servo. The spline has 25 teeth, so the
horn goes on in 14.4 degree steps, and once on, the shaft turns with it. The press takes about 5 N; pressing on the
servo alone tips the link over its foot, so one hand holds the link down while the other presses.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import Joint, on_reset, on_step, part  # noqa: E402
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
LX, LY = 0.30, -0.11  # the shaft, world
SB = 0.030  # the case's bottom, above the bench
CASE = (0.020, 0.040, 0.036)
CASE_Y0, CASE_Y1 = -0.010, 0.030
Z_TOP = SB + CASE[2]  # the case's top, 66 mm above the bench
TAB_Z0, TAB_T, TAB_L = SB + 0.0245, 0.0025, 0.007  # mounting tabs at each end, resting on the shelf
SHELF_T, HALF_W = 0.004, 0.016  # the link is 32 mm wide
SHELF_Y0, SHELF_Y1 = -0.020, 0.050
CUT = 0.0005  # the shelf's cut-out clears the case by 0.5 mm
RISER_Y0 = 0.040  # the upright's right face: the edge the link tips over
FOOT_Y1, FOOT_T = 0.204, 0.008
BLOCK_Y1, BLOCK_H = 0.236, 0.034  # the block at the link's left end, 22 cm from the shaft
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
_riser_h = TAB_Z0 - SHELF_T
LINK = part(
    "arm_link", pos=(LX, LY, TOP), mass=LINK_MASS, friction=0.6,
    boxes=[
        # the servo's case and its two mounting tabs
        ((0.0, (CASE_Y0 + CASE_Y1) / 2, SB + CASE[2] / 2), CASE, CASE_C),
        ((0.0, CASE_Y0 - TAB_L / 2, TAB_Z0 + TAB_T / 2), (CASE[0], TAB_L, TAB_T), CASE_C),
        ((0.0, CASE_Y1 + TAB_L / 2, TAB_Z0 + TAB_T / 2), (CASE[0], TAB_L, TAB_T), CASE_C),
        ((0.0, (LINE_Y0 + LINE_Y1) / 2, Z_TOP - 0.00005), (LINE_W, LINE_Y1 - LINE_Y0, 0.0003), WHITE),
        # the shelf round the case, the upright under its left part, the foot and the block
        ((0.0, (SHELF_Y0 + _cy0) / 2, _shelf_z), (2 * HALF_W, _cy0 - SHELF_Y0, SHELF_T), PLA),
        ((0.0, (_cy1 + SHELF_Y1) / 2, _shelf_z), (2 * HALF_W, SHELF_Y1 - _cy1, SHELF_T), PLA),
        (((_cx + HALF_W) / 2, (_cy0 + _cy1) / 2, _shelf_z), (HALF_W - _cx, _cy1 - _cy0, SHELF_T), PLA),
        ((-(_cx + HALF_W) / 2, (_cy0 + _cy1) / 2, _shelf_z), (HALF_W - _cx, _cy1 - _cy0, SHELF_T), PLA),
        ((0.0, (RISER_Y0 + SHELF_Y1) / 2, _riser_h / 2), (2 * HALF_W, SHELF_Y1 - RISER_Y0, _riser_h), PLA),
        ((0.0, (SHELF_Y1 + FOOT_Y1) / 2, FOOT_T / 2), (2 * HALF_W, FOOT_Y1 - SHELF_Y1, FOOT_T), PLA),
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
SUCCESS_HOLD_S = 3.0


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
# Left arm: grips the block at the link's left end across its width and holds it, 2 mm down into the bench, until the
# horn is pressed on; then lets go and parks. Right arm: picks the horn by the sides of its hub (pad tips 2 mm above
# its underside), turns it a quarter turn anticlockwise on the way so its arm points along the line, lowers the hub
# over the shaft onto the collar, presses down (commanded 10 mm past the seat), eases off to the seat, lets go and
# parks.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_UP = 0.002  # pad tips 2 mm above the hub's underside
HUB_WIDE = -0.012  # pads 24 mm apart round the 12 mm hub
PICK_Z = TOP + GRIP_UP + TIP_ABOVE_TCP
PICK_HOVER = TOP + 0.06


def hub_z(underside: float) -> float:
    """TCP height that puts the hub's underside ``underside`` above the case's top."""
    return TOP + Z_TOP + underside + GRIP_UP + TIP_ABOVE_TCP


CARRY_Z = hub_z(SHAFT_H + 0.025)
ENTRY_Z = hub_z(SHAFT_H + 0.002)
REST_Z = hub_z(COLLAR_TOP0 - Z_TOP + 0.0003)
SEAT_Z = hub_z(SEAT_GAP)
PRESS_DEPTH = 0.010
PRESS_Z = SEAT_Z - PRESS_DEPTH
HORN_YAW0 = 0.0  # fingers close along y, on the hub's sides, while the arm points along +x
HORN_YAW1 = math.pi / 2  # and along x once the arm points along +y, the line
BX, BY = LX, LY + (FOOT_Y1 + BLOCK_Y1) / 2
BLOCK_YAW = -math.pi / 2  # fingers close along x, on the block's sides
BLOCK_WIDE = -0.03  # pads 60 mm apart round the 32 mm block
BLOCK_Z = TOP + 0.010 + TIP_ABOVE_TCP  # pad tips 10 mm above the bench
HOLD_Z = BLOCK_Z - 0.002
BLOCK_HOVER = TOP + BLOCK_H + 0.05


def L(x, y, z):
    """Link coordinates to world."""
    return (LX + x, LY + y, TOP + z)


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("servo case", L(0.0, (CASE_Y0 + CASE_Y1) / 2, SB + CASE[2] / 2), CASE),
    ("shelf", L(0.0, (SHELF_Y0 + SHELF_Y1) / 2, _shelf_z), (2 * HALF_W, SHELF_Y1 - SHELF_Y0, SHELF_T)),
    ("upright", L(0.0, (RISER_Y0 + SHELF_Y1) / 2, _riser_h / 2), (2 * HALF_W, SHELF_Y1 - RISER_Y0, _riser_h)),
    ("foot", L(0.0, (SHELF_Y1 + FOOT_Y1) / 2, FOOT_T / 2), (2 * HALF_W, FOOT_Y1 - SHELF_Y1, FOOT_T)),
    ("block", L(0.0, (FOOT_Y1 + BLOCK_Y1) / 2, BLOCK_H / 2), (2 * HALF_W, BLOCK_Y1 - FOOT_Y1, BLOCK_H)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    left = [
        Waypoint(2.5, (BX, BY, BLOCK_HOVER), BLOCK_WIDE, "above the block at the link's left end", BLOCK_YAW),
        Waypoint(1.5, (BX, BY, BLOCK_Z), BLOCK_WIDE, "fingers either side of the block", BLOCK_YAW),
        Waypoint(0.8, (BX, BY, BLOCK_Z), FINGER_CLOSED, "grip", BLOCK_YAW),
        Waypoint(0.6, (BX, BY, HOLD_Z), FINGER_CLOSED, "hold the link down", BLOCK_YAW),
    ]
    t_held = _dur(left)
    right = [
        Waypoint(2.5, (HX, HY, PICK_HOVER), HUB_WIDE, "above the horn", HORN_YAW0),
        Waypoint(1.5, (HX, HY, PICK_Z), HUB_WIDE, "pads either side of the hub", HORN_YAW0),
        Waypoint(0.8, (HX, HY, PICK_Z), FINGER_CLOSED, "grip the hub", HORN_YAW0),
        Waypoint(1.2, (HX, HY, PICK_HOVER), FINGER_CLOSED, "lift", HORN_YAW0),
        Waypoint(2.5, (LX, LY, CARRY_Z), FINGER_CLOSED, "over the shaft, arm turned onto the line", HORN_YAW1),
    ]
    right.append(Waypoint(max(0.5, t_held - _dur(right)), (LX, LY, CARRY_Z), FINGER_CLOSED,
                          "wait for the link to be held", HORN_YAW1))
    right += [
        Waypoint(1.5, (LX, LY, ENTRY_Z), FINGER_CLOSED, "hub just above the shaft", HORN_YAW1),
        Waypoint(0.8, (LX, LY, ENTRY_Z), FINGER_CLOSED, "settle", HORN_YAW1),
        Waypoint(1.5, (LX, LY, REST_Z), FINGER_CLOSED, "hub down over the shaft onto the collar", HORN_YAW1),
        Waypoint(0.5, (LX, LY, REST_Z), FINGER_CLOSED, "settle", HORN_YAW1),
        Waypoint(2.5, (LX, LY, PRESS_Z), FINGER_CLOSED, "press the horn down", HORN_YAW1),
        Waypoint(1.5, (LX, LY, PRESS_Z), FINGER_CLOSED, "keep pressing", HORN_YAW1),
        Waypoint(1.0, (LX, LY, SEAT_Z), FINGER_CLOSED, "ease off at the seat", HORN_YAW1),
        Waypoint(0.6, (LX, LY, SEAT_Z), HUB_WIDE, "let go", HORN_YAW1),
        Waypoint(1.0, (LX, LY, CARRY_Z), HUB_WIDE, "up", HORN_YAW1),
    ]
    t_done = _dur(right)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    left.append(Waypoint(t_done - _dur(left), (BX, BY, HOLD_Z), FINGER_CLOSED, "hold the link while the horn is pressed",
                         BLOCK_YAW))
    left += [
        Waypoint(0.6, (BX, BY, BLOCK_Z), BLOCK_WIDE, "let go", BLOCK_YAW),
        Waypoint(1.0, (BX, BY, BLOCK_HOVER), BLOCK_WIDE, "up", BLOCK_YAW),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    return {"left": left, "right": right}
