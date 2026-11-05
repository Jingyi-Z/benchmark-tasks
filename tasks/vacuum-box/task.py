"""On the station table: a clear acrylic vacuum box holding a dish of a moisture-sensitive powder, under vacuum, with
its valve block on its right side, and a sample vial on the bench to the left.

The vacuum box: a clear acrylic chamber 18 x 18 cm and 8 cm tall, its walls 8 mm thick, with a black O-ring along
the top of its walls. Its lid is a 20 x 20 cm acrylic plate 1.5 cm thick (1.2 kg) with a handle bar along each side;
under vacuum the air presses it onto the O-ring with far more force than a robot can pull, so it cannot be lifted.
The valve block on the box's right side carries a vacuum gauge (needle dial on its front face, 0 to -100 kPa), a
needle VENT valve (a knob on the block's top: a quarter turn anticlockwise opens it fully) and the PUMP valve to the
vacuum line (a lever on the block's right side: pointing at the robots is closed, turned out to the right is open).
The pump runs all the time.

The task: break the vacuum gently with the vent knob (venting faster than 12 kPa a second blows the powder in the
dish about), lift the lid off level with both hands and set it down in front of the box, put the vial upright on the
white circle in the middle of the box floor, lower the lid back level onto the O-ring, close the vent, open the pump
valve until the gauge reads -90 kPa or lower, close the pump valve, then pull up gently on the lid's handles to check
it is held.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import labkit  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

ACRYLIC = (0.80, 0.90, 0.96)  # clear acrylic, drawn as glass
labkit.GLASS_COLORS.update({ACRYLIC})
BLACK = (0.06, 0.06, 0.07)
DARK = (0.18, 0.19, 0.21)
ALU = (0.78, 0.79, 0.81)
WHITE = (0.96, 0.96, 0.95)
RED = (0.80, 0.12, 0.10)
BRASS = (0.78, 0.62, 0.30)
DISH_C = (0.85, 0.92, 0.95)
POWDER = (0.93, 0.80, 0.30)
VIAL_C = (0.88, 0.93, 0.95)
CAP_C = (0.12, 0.35, 0.70)
HOSE = (0.75, 0.80, 0.70)  # translucent PVC hose, drawn opaque

# -- the box (fixed). Frame: origin on the table under the box's centre, axes along the world's. -----------------------
BX, BY = 0.33, 0.0
HALF, WALL, H = 0.090, 0.008, 0.080  # outer half width, wall, height
ORING_W, ORING_H = 0.006, 0.002
SEAT_Z = H + ORING_H  # the lid's underside when seated
IN_HALF = HALF - WALL
MARK_R = 0.022

BOX_BOXES = [
    ((0.0, 0.0, 0.004), (2 * HALF, 2 * HALF, 0.008), ACRYLIC),  # floor
    ((HALF - WALL / 2, 0.0, H / 2), (WALL, 2 * HALF, H), ACRYLIC),
    ((-HALF + WALL / 2, 0.0, H / 2), (WALL, 2 * HALF, H), ACRYLIC),
    ((0.0, HALF - WALL / 2, H / 2), (2 * HALF - 2 * WALL, WALL, H), ACRYLIC),
    ((0.0, -HALF + WALL / 2, H / 2), (2 * HALF - 2 * WALL, WALL, H), ACRYLIC),
    # the O-ring along the top of the walls
    ((HALF - WALL / 2, 0.0, H + ORING_H / 2), (ORING_W, 2 * HALF - WALL, ORING_H), BLACK),
    ((-HALF + WALL / 2, 0.0, H + ORING_H / 2), (ORING_W, 2 * HALF - WALL, ORING_H), BLACK),
    ((0.0, HALF - WALL / 2, H + ORING_H / 2), (2 * HALF - WALL, ORING_W, ORING_H), BLACK),
    ((0.0, -HALF + WALL / 2, H + ORING_H / 2), (2 * HALF - WALL, ORING_W, ORING_H), BLACK),
]
BOX_CYLS = [((0.0, 0.0, 0.0082), MARK_R, 0.0004, WHITE),  # the white circle for the vial
            ((0.0, 0.0, 0.0084), MARK_R - 0.002, 0.0004, (0.85, 0.85, 0.84))]

# The valve block, on the box's right side: x from -0.025 to 0.025, y from -0.140 to -0.100, 7 cm tall.
VB = ((-0.025, 0.025), (-0.140, -0.100), 0.070)
VB_C = ((VB[0][0] + VB[0][1]) / 2, (VB[1][0] + VB[1][1]) / 2)
GAUGE_R = 0.019
GAUGE_Z = 0.042
FACE_X = VB[0][0] - 0.004  # the gauge's glass, in front of the block
KNOB_R, KNOB_H = 0.009, 0.014
KNOB_XY = (VB_C[0] - 0.006, VB_C[1])
LEVER_AXIS = (0.008, VB[1][0] - 0.004, 0.034)  # on the block's right face
LEVER_L = 0.050
POST_R, POST_H = 0.005, 0.016
VENT_FULL = math.pi / 2  # the knob's quarter turn, anticlockwise from above (positive)
PUMP_OPEN = math.pi / 2  # the lever's quarter turn, out to the right (clockwise from above: negative)

# Gauge needle: about x (the dial faces -x, towards the robots); 0 points up at 0 kPa, and it turns clockwise as seen
# from the front, 0.044 rad per kPa of vacuum (270 degrees for 100 kPa).
NEEDLE_PER_KPA = math.radians(270.0) / 100.0
NEEDLE = Joint(
    "needle", kind="hinge", axis="X", anchor=(FACE_X, VB_C[1], GAUGE_Z), lower=-0.1, upper=4.8, initial=0.0,
    target=0.0, stiffness=0.02, damping=0.0008, mass=0.001,
    boxes=[((FACE_X - 0.0010, VB_C[1], GAUGE_Z + 0.007), (0.0008, 0.0012, 0.015), RED)],
)
KNOB = Joint(
    "vent", kind="hinge", axis="Z", anchor=(*KNOB_XY, VB[2]), lower=0.0, upper=VENT_FULL + 0.05, initial=0.0,
    target=0.0, stiffness=0.0, damping=0.002, friction=0.02, mass=0.01,
    cylinders=[((*KNOB_XY, VB[2] + KNOB_H / 2), KNOB_R, KNOB_H, BLACK)],
    boxes=[((KNOB_XY[0] - KNOB_R + 0.002, KNOB_XY[1], VB[2] + KNOB_H + 0.0003), (0.004, 0.0015, 0.0006), WHITE)],
)
LEVER = Joint(
    "pump", kind="hinge", axis="Z", anchor=LEVER_AXIS, lower=-PUMP_OPEN - 0.05, upper=0.05, initial=0.0, target=0.0,
    stiffness=0.0, damping=0.004, friction=0.03, mass=0.02,
    boxes=[((LEVER_AXIS[0] - LEVER_L / 2, LEVER_AXIS[1], LEVER_AXIS[2]), (LEVER_L, 0.008, 0.004), RED)],
    cylinders=[((LEVER_AXIS[0] - LEVER_L + 0.006, LEVER_AXIS[1], LEVER_AXIS[2] + POST_H / 2), POST_R, POST_H, RED)],
)
_ticks = []
for _k in range(11):  # tick marks every 10 kPa, from 0 (up) clockwise
    _a = _k * 10 * NEEDLE_PER_KPA
    _ticks.append(((FACE_X - 0.0004, VB_C[1] - 0.0155 * math.sin(_a), GAUGE_Z + 0.0155 * math.cos(_a)),
                   (0.0004, 0.0012, 0.0012), BLACK if _k < 9 else RED))
VALVE_BOXES = [
    (((VB[0][0] + VB[0][1]) / 2, VB_C[1], VB[2] / 2), (VB[0][1] - VB[0][0], VB[1][1] - VB[1][0], VB[2]), ALU),
    ((VB[0][0] - 0.002, VB_C[1], GAUGE_Z), (0.004, 2 * GAUGE_R, 2 * GAUGE_R), DARK),  # the gauge's case (square)
    ((FACE_X + 0.0002, VB_C[1], GAUGE_Z), (0.0004, 2 * GAUGE_R - 0.004, 2 * GAUGE_R - 0.004), WHITE),  # dial
    *_ticks,
    ((0.0, (VB[1][1] - WALL / 2 - HALF + WALL) / 2 - 0.004, 0.030), (0.010, 0.018, 0.010), BRASS),  # into the box
    ((VB[0][1] + 0.010, VB_C[1] + 0.008, 0.020), (0.020, 0.010, 0.010), BRASS),  # to the pump
    ((VB[0][1] + 0.060, VB_C[1] + 0.008, 0.006), (0.090, 0.012, 0.012), HOSE),  # vacuum hose, off to the pump
]
VALVE_CYLS = [((LEVER_AXIS[0], LEVER_AXIS[1] + 0.002, LEVER_AXIS[2]), 0.006, 0.008, BRASS, "Y")]

BOX = part("vacuum_box", pos=(BX, BY, TOP), fixed=True, friction=0.6,
           boxes=BOX_BOXES + VALVE_BOXES, cylinders=BOX_CYLS + VALVE_CYLS, joints=[NEEDLE, KNOB, LEVER])

# -- the lid (loose), seated on the O-ring -------------------------------------------------------------------------
LID_HALF, LID_T = 0.100, 0.015
LID_KG = 1.2
BAR_Y = 0.078  # handle bars along x at y = +-0.078 on the lid's top
BAR_L, BAR_S, BAR_UP = 0.090, 0.012, 0.012  # bar length, section, gap under it
_bar_z = LID_T + BAR_UP + BAR_S / 2
LID_BOXES = [((0.0, 0.0, LID_T / 2), (2 * LID_HALF, 2 * LID_HALF, LID_T), ACRYLIC)]
for _s in (1, -1):
    LID_BOXES += [((0.0, _s * BAR_Y, _bar_z), (BAR_L, BAR_S, BAR_S), BLACK)]
    LID_BOXES += [((_e * (BAR_L / 2 - 0.006), _s * BAR_Y, LID_T + BAR_UP / 2), (0.012, BAR_S, BAR_UP), BLACK)
                  for _e in (1, -1)]
LID = part("lid", pos=(BX, BY, TOP + SEAT_Z), mass=LID_KG, friction=0.8, boxes=LID_BOXES)

# -- the dish of powder already inside, and the sample vial on the bench -----------------------------------------------
DISH_R, DISH_H = 0.025, 0.010
DISH_XY = (BX + 0.040, BY + 0.040)
DISH = part("dish", pos=(*DISH_XY, TOP + 0.008), mass=0.020, friction=0.5,
            cylinders=[((0.0, 0.0, 0.001), DISH_R, 0.002, DISH_C),
                       ((0.0, 0.0, 0.004), DISH_R - 0.003, 0.004, POWDER)],
            boxes=[((_r * math.cos(_a), _r * math.sin(_a), DISH_H / 2), (0.002, 0.012, DISH_H), DISH_C, _a)
                   for _r in (DISH_R - 0.001,) for _a in np.linspace(0, 2 * math.pi, 14, endpoint=False)])
VIAL_R, VIAL_H, CAP_H = 0.012, 0.040, 0.012
VIAL_XY = (0.20, 0.20)
VIAL = part("vial", pos=(*VIAL_XY, TOP), mass=0.030, friction=0.9,
            cylinders=[((0.0, 0.0, VIAL_H / 2), VIAL_R, VIAL_H, VIAL_C),
                       ((0.0, 0.0, VIAL_H + CAP_H / 2), VIAL_R + 0.0005, CAP_H, CAP_C)])

OBJECTS = [BOX, LID, DISH, VIAL]


def W(x, y, z):
    """Box coordinates to world."""
    return (BX + x, BY + y, TOP + z)


# -- the air: pressure in the box, the vent, the pump, and the lid held down ---------------------------------------
ATM = 101.3  # kPa
P0 = 8.0  # kPa absolute at the start (-93 kPa on the gauge)
VENT_K = 60.0  # kPa/s with the vent fully open at full vacuum
PUMP_K = 0.6  # 1/s: the pump pulls the box down towards 3 kPa
P_PUMP = 3.0
LEAK = 0.002  # kPa/s through a seated lid
GENTLE = 12.0  # kPa/s: venting faster blows the powder about
HOLD_MAX = 150.0  # N: the air's push on a seated lid under vacuum (capped; the real push is thousands of newtons)
SEAL_TILT, SEAL_Z, SEAL_XY, SEAL_YAW = 1.5, 0.0015, 0.008, 4.0  # deg, m, m, deg
VENT_OPEN, PUMP_ON = 0.05, PUMP_OPEN * 0.8
_trial = {"p": P0, "max_rate": 0.0, "blown": False, "ids": None, "n": None}


def lid_sealed(lid) -> bool:
    """Seated on the O-ring: flat within 1.5 degrees, its underside within 1.5 mm of the O-ring's top, its centre
    within 8 mm of the box's and square to it within 4 degrees (so it covers the O-ring all round)."""
    R = lid.rotation()
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    yaw = (yaw + 45.0) % 90.0 - 45.0
    return (lid.tilt_deg() <= SEAL_TILT and abs(lid.pos[2] - (TOP + SEAT_Z)) <= SEAL_Z
            and math.hypot(lid.pos[0] - BX, lid.pos[1] - BY) <= SEAL_XY and abs(yaw) <= SEAL_YAW)


def vent_fraction(q: float) -> float:
    return float(np.clip(q / VENT_FULL, 0.0, 1.0)) ** 2


@on_reset
def _reset(env):
    _trial.update(p=P0, max_rate=0.0, blown=False)
    _live_reset()


@on_step
def _air(env):
    import torch

    base = env.unwrapped
    dt = float(getattr(base, "step_dt", 0.05))
    s = {n: labkit._part_state(env, labkit._SPECS[n]) for n in ("vacuum_box", "lid", "dish")}
    q = s["vacuum_box"].joints
    sealed = lid_sealed(s["lid"])
    p = _trial["p"]
    if not sealed:
        rate = 0.0
        p = ATM
    else:
        rate = VENT_K * vent_fraction(q["vent"]) * math.sqrt(max(0.0, ATM - p) / ATM)  # in through the vent
        p += rate * dt
        if q["pump"] <= -PUMP_ON:
            p -= PUMP_K * (p - P_PUMP) * dt
        p = min(ATM, p + LEAK * dt)
    _trial["p"] = p
    _trial["max_rate"] = max(_trial["max_rate"], rate)
    blow = rate > GENTLE
    _trial["blown"] = _trial["blown"] or blow
    box = base.scene["vacuum_box"]
    if _trial["ids"] is None:
        _trial["ids"] = [list(box.joint_names).index(NEEDLE.joint_name)]
    box.set_joint_position_target(torch.tensor([[(ATM - p) * NEEDLE_PER_KPA]], device=box.device),
                                  joint_ids=_trial["ids"])
    lid = base.scene["lid"]
    push = min(HOLD_MAX, (ATM - p) * (2 * IN_HALF) ** 2 * 1000.0) if sealed and p < ATM - 1.0 else 0.0
    _wrench(lid, (0.0, 0.0, -push))
    dish = base.scene["dish"]
    _wrench(dish, (0.0, 0.8, 0.6) if blow else (0.0, 0.0, 0.0))  # a gust from the vent port on the right wall
    _live_update(s, q)


def _wrench(asset, force):
    import torch

    f = torch.tensor([[[*force]]], device=asset.device, dtype=torch.float32)
    t = torch.zeros_like(f)
    comp = getattr(asset, "permanent_wrench_composer", None)
    if comp is not None:
        comp.set_forces_and_torques_index(forces=f, torques=t, body_ids=[0], is_global=True)
    else:
        asset.set_external_force_and_torque(f, t, body_ids=[0], is_global=True)


def gauge_kpa() -> float:
    """What the gauge reads (kPa, negative under vacuum)."""
    return -(ATM - _trial["p"])


# -- scoring ----------------------------------------------------------------------------------------------------------
UPRIGHT_DEG = 10.0
MARK_TOL = 0.015
TARGET_KPA = -90.0
CLOSED_TOL = math.radians(10.0)


def vial_inside(v) -> bool:
    """Upright (within 10 degrees) on the box's floor, wholly inside its walls."""
    return (v.tilt_deg() <= UPRIGHT_DEG and abs(v.pos[2] - (TOP + 0.008)) <= 0.003
            and abs(v.pos[0] - BX) <= IN_HALF - VIAL_R and abs(v.pos[1] - BY) <= IN_HALF - VIAL_R)


def score(objects):
    """0 unless the vial stands upright inside the box and the lid is seated on the O-ring. 0.25 then; 0.5 if the
    gauge also reads -90 kPa or lower; 0.75 if, besides, both valves are closed (vent knob back within 10 degrees of
    shut, pump lever within 10 degrees of pointing at the robots); 1.0 if, besides, the vial stands on the white circle
    (its centre within 1.5 cm of the circle's). Capped at 0.25 if the box was ever vented faster than 12 kPa a
    second."""
    v, lid, q = objects["vial"], objects["lid"], objects["vacuum_box"].joints
    if not (vial_inside(v) and lid_sealed(lid)):
        return 0.0
    s = 0.25
    if gauge_kpa() <= TARGET_KPA:
        s = 0.5
        if q["vent"] <= CLOSED_TOL and q["pump"] >= -CLOSED_TOL:
            s = 0.75
            if math.hypot(v.pos[0] - BX, v.pos[1] - BY) <= MARK_TOL:
                s = 1.0
    return min(s, 0.25) if _trial["blown"] else s


SUCCESS_HOLD_S = 4.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    v = objects["vial"]
    return math.hypot(v.pos[0] - BX, v.pos[1] - BY) + max(0.0, gauge_kpa() - TARGET_KPA) * 0.001


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm: grips the vent knob from above and turns it anticlockwise in small steps, closed loop on the gauge: it
# opens it a little at a time, keeping the inflow under about 9 kPa a second, until the box is at atmospheric pressure.
# Both arms: grip the lid's two handle bars from above, lift it level 4 cm, carry it in front of the box and set it
# down. Left arm: carries the vial into the box onto the white circle. Both arms: carry the lid back and lower it level
# onto the O-ring. Right arm: shuts the vent knob, swings the pump lever open, waits for the gauge to read -92 kPa,
# swings it shut. Both arms: grip the handles and pull up gently (the lid stays put), let go and park.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7

TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.15, -0.25, TOP + 0.16)
BAR_GRIP_Z = TOP + SEAT_Z + _bar_z - 0.002 + TIP_ABOVE_TCP  # pad tips 2 mm below the bar's middle
BAR_WIDE, BAR_HOLD = -0.016, -0.003  # fingers closing along y round the 12 mm bar
LIFT = 0.040
ASIDE = (0.120, 0.0)  # where the lid is set down, in front of the box
KNOB_Z = TOP + VB[2] + KNOB_H - 0.004 + TIP_ABOVE_TCP  # pad tips 4 mm down the knob
KNOB_WIDE, KNOB_HOLD = -0.016, -0.0055
POST_Z = TOP + LEVER_AXIS[2] + 0.004 + POST_H - 0.006 + TIP_ABOVE_TCP
POST_WIDE = -0.008  # pads 16 mm apart round the 10 mm post
VIAL_GRIP_Z = TOP + 0.025 + TIP_ABOVE_TCP
VIAL_WIDE, VIAL_HOLD = -0.020, -0.009
CARRY_Z = TOP + H + 0.010 + 0.025 + TIP_ABOVE_TCP  # the vial's bottom 1 cm over the walls
VENT_RATE_AIM = 9.0  # kPa/s
_live = {"vent": 0.0, "tl": None, "p": P0}


def _live_reset():
    _live.update(vent=0.0, tl=None, p=P0)


def _live_update(s, q):
    _live["p"] = _trial["p"]


def _vent_target(dt: float) -> float:
    """The knob angle to turn to next: open it further while the inflow is under VENT_RATE_AIM (the gauge rising
    slower than that), never more than a quarter turn."""
    p = _live["p"]
    rate_now = VENT_K * vent_fraction(_live["vent"]) * math.sqrt(max(0.0, ATM - p) / ATM)
    if rate_now < VENT_RATE_AIM and p < ATM - 0.5:
        _live["vent"] = min(VENT_FULL, _live["vent"] + math.radians(8.0) * dt)
    return _live["vent"]


def _clock():
    return float(getattr(_demo, "_plan_clock", 0.0))


class _KnobYaw(float):
    """A wrist yaw read every control step: ``base`` plus the vent knob's angle, which the closed loop advances while
    ``turning`` (else it stays where the loop left it)."""

    def __new__(cls, base=0.0, turning=False):
        obj = float.__new__(cls, base)
        obj.base, obj.turning = base, turning
        return obj

    @property
    def rot(self):
        if self.turning:
            t = _clock()
            dt = 0.0 if _live["tl"] is None else max(0.0, t - _live["tl"])
            _live["tl"] = t
            _vent_target(dt)
        return _prev_grip(self.base + _live["vent"])


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._rot = True
    _demo.grip_rotation = _grip


def _lever_post(phi: float):
    """World xy of the lever's post with the lever turned phi (0 closed, -pi/2 open)."""
    ax, ay = W(LEVER_AXIS[0], LEVER_AXIS[1], 0.0)[:2]
    r = LEVER_L - 0.006
    return ax - r * math.cos(phi), ay + r * math.sin(phi)


def _dur(wps):
    return sum(w.duration for w in wps)


def _both(left, right, dur, xy, z, finger, label):
    """Both hands on the lid's handle bars, the lid's centre at xy, the bars' grip height z."""
    left.append(Waypoint(dur, (xy[0], xy[1] + BAR_Y, z), finger, label, 0.0))
    right.append(Waypoint(dur, (xy[0], xy[1] - BAR_Y, z), finger, label, 0.0))


def _sync(left, right, label):
    t = max(_dur(left), _dur(right))
    for wps in (left, right):
        last = wps[-1]
        wps.append(Waypoint(max(0.2, t - _dur(wps) + 0.2), last.pos, last.finger, label, last.yaw))


def demo():
    _live_reset()
    kx, ky = W(*KNOB_XY, 0.0)[:2]
    knob_yaw = 0.0  # fingers closing along y round the knob
    left = [Waypoint(3.0, LEFT_PARK, FINGER_OPEN, "wait")]
    right = [
        Waypoint(2.5, (kx, ky, KNOB_Z + 0.04), KNOB_WIDE, "above the vent knob", knob_yaw),
        Waypoint(1.2, (kx, ky, KNOB_Z), KNOB_WIDE, "fingers round the knob", knob_yaw),
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_HOLD, "grip the knob", knob_yaw),
        Waypoint(0.1, (kx, ky, KNOB_Z), KNOB_HOLD, "turn it open slowly, watching the gauge", _KnobYaw(knob_yaw, True)),
        Waypoint(14.0, (kx, ky, KNOB_Z), KNOB_HOLD, "turn it open slowly, watching the gauge",
                 _KnobYaw(knob_yaw, True)),
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_WIDE, "let go of the knob", _KnobYaw(knob_yaw)),
        Waypoint(1.0, (kx, ky, KNOB_Z + 0.04), KNOB_WIDE, "up", _KnobYaw(knob_yaw)),
    ]
    _sync(left, right, "the box is vented")
    c = (BX, BY)
    _both(left, right, 2.5, c, BAR_GRIP_Z + 0.04, BAR_WIDE, "above the lid's handle bars")
    _both(left, right, 1.2, c, BAR_GRIP_Z, BAR_WIDE, "fingers round the bars")
    _both(left, right, 0.6, c, BAR_GRIP_Z, BAR_HOLD, "grip both bars")
    _both(left, right, 1.5, c, BAR_GRIP_Z + LIFT, BAR_HOLD, "lift the lid level")
    _both(left, right, 3.0, ASIDE, BAR_GRIP_Z + LIFT, BAR_HOLD, "carry it in front of the box")
    aside_z = BAR_GRIP_Z - SEAT_Z - 0.001
    _both(left, right, 1.5, ASIDE, aside_z, BAR_HOLD, "set it down on the bench")
    _both(left, right, 0.6, ASIDE, aside_z, BAR_WIDE, "let go")
    _both(left, right, 0.8, ASIDE, aside_z + 0.04, BAR_WIDE, "up")
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "out of the way"))
    vx, vy = VIAL_XY
    left += [
        Waypoint(2.5, (vx, vy, VIAL_GRIP_Z + 0.05), VIAL_WIDE, "above the vial"),
        Waypoint(1.2, (vx, vy, VIAL_GRIP_Z), VIAL_WIDE, "fingers round the vial"),
        Waypoint(0.6, (vx, vy, VIAL_GRIP_Z), VIAL_HOLD, "grip it"),
        Waypoint(1.2, (vx, vy, CARRY_Z), VIAL_HOLD, "lift it over the wall"),
        Waypoint(2.5, (BX, BY, CARRY_Z), VIAL_HOLD, "over the white circle"),
        Waypoint(2.0, (BX, BY, VIAL_GRIP_Z + 0.008 + 0.001), VIAL_HOLD, "lower it onto the circle"),
        Waypoint(0.6, (BX, BY, VIAL_GRIP_Z + 0.008 + 0.001), VIAL_WIDE, "let go"),
        Waypoint(1.5, (BX, BY, CARRY_Z), VIAL_WIDE, "up out of the box"),
    ]
    _sync(left, right, "the vial is in")
    _both(left, right, 2.5, ASIDE, aside_z + 0.04, BAR_WIDE, "back above the lid's bars")
    _both(left, right, 1.2, ASIDE, aside_z, BAR_WIDE, "fingers round the bars")
    _both(left, right, 0.6, ASIDE, aside_z, BAR_HOLD, "grip both bars")
    _both(left, right, 1.5, ASIDE, BAR_GRIP_Z + LIFT, BAR_HOLD, "lift the lid level")
    _both(left, right, 3.0, c, BAR_GRIP_Z + LIFT, BAR_HOLD, "carry it over the box")
    _both(left, right, 2.5, c, BAR_GRIP_Z - 0.002, BAR_HOLD, "lower it level onto the O-ring")
    _both(left, right, 0.6, c, BAR_GRIP_Z - 0.002, BAR_WIDE, "let go")
    _both(left, right, 0.8, c, BAR_GRIP_Z + 0.04, BAR_WIDE, "up")
    left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "out of the way"))
    # vent shut, pump open
    right += [
        Waypoint(2.5, (kx, ky, KNOB_Z + 0.04), KNOB_WIDE, "above the vent knob", _KnobYaw(knob_yaw)),
    ]
    _knob_shut(right, kx, ky, knob_yaw)
    for wps in [right]:
        px, py = _lever_post(0.0)
        wps += [Waypoint(2.0, (px, py - 0.004, POST_Z + 0.04), POST_WIDE, "above the pump lever's post", 0.0),
                Waypoint(1.0, (px, py - 0.004, POST_Z), POST_WIDE, "fingers round the post", 0.0)]
        for phi in np.linspace(-0.25, -PUMP_OPEN, 6):
            x, y = _lever_post(phi)
            wps.append(Waypoint(0.5, (x, y, POST_Z), POST_WIDE, "swing the pump lever open", -phi))
        wps.append(Waypoint(7.0, (x, y, POST_Z), POST_WIDE, "pump down: wait for -90 kPa or lower", -phi))
        for phi in np.linspace(-PUMP_OPEN + 0.25, 0.04, 6):
            x, y = _lever_post(phi)
            wps.append(Waypoint(0.5, (x, y, POST_Z), POST_WIDE, "swing the pump lever shut", -phi))
        wps.append(Waypoint(0.8, (x, y, POST_Z + 0.04), POST_WIDE, "up off the post", -phi))
    _sync(left, right, "pumped down")
    _both(left, right, 2.5, c, BAR_GRIP_Z + 0.04, BAR_WIDE, "above the lid's handle bars")
    _both(left, right, 1.2, c, BAR_GRIP_Z, BAR_WIDE, "fingers round the bars")
    _both(left, right, 0.6, c, BAR_GRIP_Z, BAR_HOLD, "grip both bars")
    _both(left, right, 1.5, c, BAR_GRIP_Z + 0.006, BAR_HOLD, "pull up gently: the lid is held")
    _both(left, right, 0.6, c, BAR_GRIP_Z + 0.006, BAR_WIDE, "let go")
    _both(left, right, 0.8, c, BAR_GRIP_Z + 0.04, BAR_WIDE, "up")
    left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"))
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    _sync(left, right, "rest")
    for wps in (left, right):
        wps.append(Waypoint(3.0, wps[-1].pos, wps[-1].finger, "rest", wps[-1].yaw))
    return {"left": left, "right": right}


def _knob_shut(right, kx, ky, knob_yaw):
    """Grip the knob at the angle the vent loop left it at and turn it back shut."""
    right += [
        Waypoint(1.2, (kx, ky, KNOB_Z), KNOB_WIDE, "fingers round the knob", _KnobYaw(knob_yaw)),
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_HOLD, "grip the knob", _KnobYaw(knob_yaw)),
        Waypoint(2.5, (kx, ky, KNOB_Z), KNOB_HOLD, "turn the vent shut", knob_yaw - 0.03),
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_WIDE, "let go", knob_yaw),
        Waypoint(1.0, (kx, ky, KNOB_Z + 0.04), KNOB_WIDE, "up", knob_yaw),
    ]
