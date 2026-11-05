"""On the station table: a polycarbonate vacuum desiccator holding a dish of a moisture-sensitive powder, under vacuum;
its vacuum line with an inline gauge on the bench to its right; and a sample vial on the bench to its left.

The desiccator: a white polypropylene bowl base, 17 cm across its flange, with a perforated white plate inside, and a
clear dome lid that sits on the flange (an O-ring seals it). On top of the dome, a clear knob is the VENT valve: turned
anticlockwise (seen from above) it lets air in, faster the further it is turned, a quarter turn fully open. On the
base's right side, a red stopcock connects the desiccator to the vacuum line through a hose barb: its lever pointing at
the robots is shut, turned a quarter turn anticlockwise (pointing right) it is open to the pump, which runs all the
time. The gauge on the vacuum line reads the desiccator's pressure whenever the stopcock is open, and holds its last
reading when it is shut. The base is light: turning the stopcock (or the vent knob) without steadying it turns or
slides the whole desiccator on the bench instead.

Under vacuum the air holds the lid down with far more force than a robot can pull. Venting faster than 12 kPa a second
blows the powder in the dish about.

The task: vent the desiccator gently, lift the lid off by its knob and set it down, stand the vial on the middle of the
plate, put the lid back centred on the flange, shut the vent, open the stopcock until the gauge reads -90 kPa or
lower, shut the stopcock, and pull up gently on the knob to check the lid is held.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, lathe_mesh, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

PP = (0.95, 0.95, 0.93)  # white polypropylene
CLEAR = (0.82, 0.90, 0.96)  # clear polycarbonate, drawn as glass
labkit.GLASS_COLORS.update({CLEAR})
RED = (0.82, 0.12, 0.10)
BLACK = (0.06, 0.06, 0.07)
DARK = (0.18, 0.19, 0.21)
STEEL = (0.75, 0.76, 0.78)
HOSE = (0.85, 0.75, 0.55)  # amber rubber vacuum hose
DISH_C = (0.85, 0.92, 0.95)
POWDER = (0.93, 0.80, 0.30)
VIAL_C = (0.88, 0.93, 0.95)
CAP_C = (0.12, 0.35, 0.70)
WHITE = (0.97, 0.97, 0.96)

# -- the base (loose). Frame: origin on the table under its axis. -----------------------------------------------------
BX, BY = 0.24, -0.05
BASE_KG = 0.8
FLANGE_R, FLANGE_TOP = 0.085, 0.075
WALL_R = 0.072  # inside of the wall at the top
PLATE_Z = 0.032  # the perforated plate's top
PLATE_R = 0.066
BASE_PROFILE = [(0.0, 0.0), (0.056, 0.0), (0.060, 0.004), (0.064, 0.020), (0.070, 0.045), (0.074, 0.062),
                (0.076, 0.069), (FLANGE_R, 0.069), (FLANGE_R, FLANGE_TOP), (WALL_R, FLANGE_TOP), (0.070, 0.050),
                (0.062, 0.020), (0.056, 0.006), (0.0, 0.006)]
N_WALL = 28
STOP_ANGLE = -math.pi / 2  # the stopcock on the base's right (-y)
STOP_R = FLANGE_R + 0.008
STOP_Z = 0.050
STOP_XY = (STOP_R * math.cos(STOP_ANGLE), STOP_R * math.sin(STOP_ANGLE))
LEVER_L = 0.028
STOP_OPEN = math.pi / 2  # the lever's quarter turn anticlockwise (positive about z)
_wall_boxes = []
for _i in range(N_WALL):
    _a = 2 * math.pi * (_i + 0.5) / N_WALL
    _r = WALL_R + 0.002
    _wall_boxes.append(((_r * math.cos(_a), _r * math.sin(_a), (0.006 + FLANGE_TOP) / 2),
                        (0.004, 2 * math.pi * _r / N_WALL + 0.002, FLANGE_TOP - 0.006), PP, _a))
    _rf = (WALL_R + FLANGE_R) / 2
    _wall_boxes.append(((_rf * math.cos(_a), _rf * math.sin(_a), FLANGE_TOP - 0.003),
                        (FLANGE_R - WALL_R, 2 * math.pi * _rf / N_WALL + 0.002, 0.006), PP, _a))
STOPCOCK = Joint(
    "stopcock", kind="hinge", axis="Z", anchor=(*STOP_XY, STOP_Z + 0.010), lower=-0.05, upper=STOP_OPEN + 0.05,
    initial=0.0, target=0.0, stiffness=0.0, damping=0.02, friction=0.30, mass=0.01,
    boxes=[((STOP_XY[0] - LEVER_L / 2, STOP_XY[1], STOP_Z + 0.014), (LEVER_L, 0.008, 0.004), RED)],
    cylinders=[((STOP_XY[0], STOP_XY[1], STOP_Z + 0.013), 0.007, 0.006, RED),
               ((STOP_XY[0] - LEVER_L + 0.004, STOP_XY[1], STOP_Z + 0.020), 0.0045, 0.012, RED)],  # lever post
)
BASE = part(
    "base", pos=(BX, BY, TOP), mass=BASE_KG, friction=0.30,
    meshes=[lathe_mesh(BASE_PROFILE, PP, segments=72),
            lathe_mesh([(0.0, PLATE_Z), (PLATE_R, PLATE_Z), (PLATE_R, PLATE_Z - 0.003), (0.0, PLATE_Z - 0.003)],
                       WHITE, segments=72)],
    boxes=[*_wall_boxes,
           ((STOP_XY[0], STOP_XY[1] + 0.008, STOP_Z), (0.010, 0.020, 0.010), PP),  # the port through the wall
           ((STOP_XY[0], STOP_XY[1] - 0.015, STOP_Z), (0.006, 0.020, 0.006), PP)],  # hose barb
    cylinders=[((0.0, 0.0, 0.003), 0.058, 0.006, PP),  # floor
               ((0.0, 0.0, PLATE_Z - 0.0015), PLATE_R, 0.003, WHITE),  # the plate
               ((0.0, 0.0, (0.006 + PLATE_Z - 0.003) / 2), 0.008, PLATE_Z - 0.009, WHITE),  # its centre leg
               *[((0.045 * math.cos(a), 0.045 * math.sin(a), PLATE_Z + 0.0002), 0.0035, 0.0004, (0.80, 0.80, 0.80))
                 for a in np.linspace(0, 2 * math.pi, 10, endpoint=False)],  # the plate's holes, drawn
               ((STOP_XY[0], STOP_XY[1], STOP_Z), 0.008, 0.016, WHITE)],  # stopcock body
    joints=[STOPCOCK],
)

# -- the lid (loose): the dome sits on the flange; the knob on top is the vent valve ----------------------------------
LID_KG = 0.5
DOME_TOP = 0.148
KNOB_R, KNOB_H = 0.013, 0.022
LID_PROFILE = [(FLANGE_R, 0.0), (FLANGE_R, 0.008), (0.080, 0.010), (0.078, 0.025), (0.072, 0.045), (0.060, 0.060),
               (0.042, 0.069), (0.020, 0.073), (0.016, 0.073), (0.016, 0.076), (0.0, 0.076)]
VENT_FULL = math.pi / 2
KNOB = Joint(
    "vent", kind="hinge", axis="Z", anchor=(0.0, 0.0, 0.076), lower=-0.05, upper=VENT_FULL + 0.05, initial=0.0,
    target=0.0, stiffness=0.0, damping=0.01, friction=0.25, mass=0.01,
    cylinders=[((0.0, 0.0, 0.076 + KNOB_H / 2), KNOB_R, KNOB_H, CLEAR)],
    boxes=[((KNOB_R * math.cos(a), KNOB_R * math.sin(a), 0.076 + KNOB_H / 2), (0.003, 0.003, KNOB_H), CLEAR, a)
           for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]  # grip ribs
          + [((KNOB_R - 0.003, 0.0, 0.076 + KNOB_H + 0.0002), (0.006, 0.0015, 0.0005), RED)],  # its mark
)
_rim_boxes = []
for _i in range(N_WALL):
    _a = 2 * math.pi * (_i + 0.5) / N_WALL
    _r = (0.078 + FLANGE_R) / 2
    _rim_boxes.append(((_r * math.cos(_a), _r * math.sin(_a), 0.004), (FLANGE_R - 0.078, 2 * math.pi * _r / N_WALL
                                                                        + 0.002, 0.008), CLEAR, _a))
LID = part(
    "lid", pos=(BX, BY, TOP + FLANGE_TOP), mass=LID_KG, friction=1.0,
    meshes=[lathe_mesh(LID_PROFILE, CLEAR, segments=72)],
    boxes=_rim_boxes,
    cylinders=[((0.0, 0.0, 0.0745), 0.018, 0.003, CLEAR)],  # the knob's boss
    joints=[KNOB],
)

# -- the vacuum line (fixed): a hose from the stopcock's barb to a small manifold with the gauge ----------------------
LINE_XY = (BX, BY - 0.150)
GAUGE_R = 0.020
GAUGE_TOP = 0.050
NEEDLE_PER_KPA = math.radians(270.0) / 100.0
NEEDLE = Joint(
    "needle", kind="hinge", axis="Z", anchor=(0.0, 0.0, GAUGE_TOP + 0.001), lower=-4.8, upper=0.1, initial=0.0,
    target=0.0, stiffness=0.02, damping=0.0008, mass=0.001,
    boxes=[((0.0, 0.007, GAUGE_TOP + 0.0012), (0.0012, 0.015, 0.0008), RED)],
)
_ticks = []
for _k in range(11):  # every 10 kPa, from 0 (pointing +y, away from the base) clockwise seen from above
    _a = -_k * 10 * NEEDLE_PER_KPA
    _ticks.append(((-0.0155 * math.sin(_a), 0.0155 * math.cos(_a), GAUGE_TOP + 0.0003), (0.0012, 0.0012, 0.0004),
                   BLACK if _k < 9 else RED))
_hose_len = LINE_XY[1] + 0.012 - (BY + STOP_XY[1] - 0.025)
LINE = part(
    "vacuum_line", pos=(*LINE_XY, TOP), fixed=True, friction=0.6,
    boxes=[((0.0, 0.0, 0.010), (0.040, 0.030, 0.020), STEEL),  # manifold
           ((0.0, 0.0, (0.020 + GAUGE_TOP - 0.008) / 2), (0.008, 0.008, GAUGE_TOP - 0.028), STEEL),  # gauge stem
           *_ticks,
           ((0.0, 0.012 - _hose_len / 2, STOP_Z * 0.5), (0.010, _hose_len, 0.010), HOSE),  # hose, low on the bench
           ((0.0, -0.040, 0.008), (0.010, 0.050, 0.010), HOSE)],  # on to the pump
    cylinders=[((0.0, 0.0, GAUGE_TOP - 0.004), GAUGE_R, 0.008, DARK), ((0.0, 0.0, GAUGE_TOP + 0.0001), GAUGE_R - 0.002,
                                                                       0.0002, WHITE)],
    joints=[NEEDLE],
)

# -- the dish of powder inside, and the vial ------------------------------------------------------------------------
DISH_R, DISH_H = 0.020, 0.008
DISH_XY = (BX + 0.035, BY + 0.025)
DISH = part("dish", pos=(*DISH_XY, TOP + PLATE_Z), mass=0.020, friction=0.5,
            cylinders=[((0.0, 0.0, 0.001), DISH_R, 0.002, DISH_C), ((0.0, 0.0, 0.0035), DISH_R - 0.003, 0.003, POWDER)],
            boxes=[((DISH_R * math.cos(a), DISH_R * math.sin(a), DISH_H / 2), (0.002, 0.011, DISH_H), DISH_C, a)
                   for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
VIAL_R, VIAL_H, CAP_H = 0.011, 0.035, 0.010
VIAL_XY = (0.20, 0.20)
VIAL = part("vial", pos=(*VIAL_XY, TOP), mass=0.030, friction=0.9,
            cylinders=[((0.0, 0.0, VIAL_H / 2), VIAL_R, VIAL_H, VIAL_C),
                       ((0.0, 0.0, VIAL_H + CAP_H / 2), VIAL_R + 0.0005, CAP_H, CAP_C)])

OBJECTS = [BASE, LID, LINE, DISH, VIAL]


# -- the air -----------------------------------------------------------------------------------------------------------
ATM = 101.3  # kPa
P0 = 8.0
VENT_K = 60.0  # kPa/s, vent fully open at full vacuum
PUMP_K = 0.6  # 1/s towards P_PUMP
P_PUMP = 3.0
LEAK = 0.002
GENTLE = 12.0
HOLD_MAX = 150.0  # N on a seated lid under vacuum (capped)
SEAL_TILT, SEAL_Z, SEAL_XY = 1.5, 0.0015, 0.004
_trial = {"p": P0, "gauge": P0, "max_rate": 0.0, "blown": False, "ids": None}


def lid_sealed(lid, base) -> bool:
    """Seated on the flange: flat within 1.5 degrees, its rim within 1.5 mm of the flange's top, and its centre within
    4 mm of the base's axis."""
    rim = base.to_world((0.0, 0.0, FLANGE_TOP))
    rel = base.to_local(lid.pos)
    return (lid.tilt_deg() <= SEAL_TILT and abs(lid.pos[2] - rim[2]) <= SEAL_Z and math.hypot(rel[0], rel[1]) <= SEAL_XY)


def vent_fraction(q: float) -> float:
    return float(np.clip(q / VENT_FULL, 0.0, 1.0)) ** 2


@on_reset
def _reset(env):
    _trial.update(p=P0, gauge=P0, max_rate=0.0, blown=False)
    _live_reset()


def _wrench(asset, force):
    import torch

    f = torch.tensor([[[*force]]], device=asset.device, dtype=torch.float32)
    t = torch.zeros_like(f)
    comp = getattr(asset, "permanent_wrench_composer", None)
    if comp is not None:
        comp.set_forces_and_torques_index(forces=f, torques=t, body_ids=[0], is_global=True)
    else:
        asset.set_external_force_and_torque(f, t, body_ids=[0], is_global=True)


@on_step
def _air(env):
    import torch

    base_env = env.unwrapped
    dt = float(getattr(base_env, "step_dt", 0.05))
    s = {n: labkit._part_state(env, labkit._SPECS[n]) for n in ("base", "lid")}
    sealed = lid_sealed(s["lid"], s["base"])
    p = _trial["p"]
    rate = 0.0
    if not sealed:
        p = ATM
    else:
        rate = VENT_K * vent_fraction(s["lid"].joints["vent"]) * math.sqrt(max(0.0, ATM - p) / ATM)
        p += rate * dt
        if s["base"].joints["stopcock"] >= STOP_OPEN * 0.8:
            p -= PUMP_K * (p - P_PUMP) * dt
        p = min(ATM, p + LEAK * dt)
    _trial["p"] = p
    if s["base"].joints["stopcock"] >= STOP_OPEN * 0.2:  # the gauge sees the desiccator through the open stopcock
        _trial["gauge"] = p
    _trial["max_rate"] = max(_trial["max_rate"], rate)
    blow = rate > GENTLE
    _trial["blown"] = _trial["blown"] or blow
    line = base_env.scene["vacuum_line"]
    if _trial["ids"] is None:
        _trial["ids"] = [list(line.joint_names).index(NEEDLE.joint_name)]
    line.set_joint_position_target(torch.tensor([[-(ATM - _trial["gauge"]) * NEEDLE_PER_KPA]], device=line.device),
                                   joint_ids=_trial["ids"])
    push = min(HOLD_MAX, (ATM - p) * math.pi * WALL_R ** 2 * 1000.0) if sealed and p < ATM - 1.0 else 0.0
    _wrench(base_env.scene["lid"], (0.0, 0.0, -push))
    _wrench(base_env.scene["base"], (0.0, 0.0, push))  # the same push holds the base up against the lid
    _wrench(base_env.scene["dish"], (0.0, 0.5, 0.4) if blow else (0.0, 0.0, 0.0))
    _live["p"] = p


def gauge_kpa() -> float:
    return -(ATM - _trial["gauge"])


# -- scoring ----------------------------------------------------------------------------------------------------------
UPRIGHT_DEG = 10.0
MARK_TOL = 0.015
TARGET_KPA = -90.0
CLOSED_TOL = math.radians(10.0)


def vial_on_plate(v, base) -> bool:
    rel = base.to_local(v.pos)
    return (v.tilt_deg() <= UPRIGHT_DEG and abs(rel[2] - PLATE_Z) <= 0.003 and math.hypot(rel[0], rel[1]) <= PLATE_R
            - VIAL_R)


def score(objects):
    """0 unless the vial stands upright on the plate inside the base and the lid is seated on the flange. 0.25 then;
    0.5 if the desiccator is also at -90 kPa or lower (the gauge, read with the stopcock open); 0.75 if, besides, the
    vent knob and the stopcock are both shut (within 10 degrees); 1.0 if, besides, the vial's centre is within 1.5 cm of
    the plate's. Capped at 0.25 if the desiccator was ever vented faster than 12 kPa a second."""
    v, lid, base = objects["vial"], objects["lid"], objects["base"]
    if not (vial_on_plate(v, base) and lid_sealed(lid, base)):
        return 0.0
    s = 0.25
    if -(ATM - _trial["p"]) <= TARGET_KPA:
        s = 0.5
        if lid.joints["vent"] <= CLOSED_TOL and base.joints["stopcock"] <= CLOSED_TOL:
            s = 0.75
            rel = base.to_local(v.pos)
            if math.hypot(rel[0], rel[1]) <= MARK_TOL:
                s = 1.0
    return min(s, 0.25) if _trial["blown"] else s


SUCCESS_HOLD_S = 4.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    v, base = objects["vial"], objects["base"]
    rel = base.to_local(v.pos)
    return math.hypot(rel[0], rel[1]) + max(0.0, -(ATM - _trial["p"]) - TARGET_KPA) * 0.001


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left arm steadies the base whenever a valve is turned: closed fingertips pressing down on the flange's left edge.
# Right arm: grips the vent knob and turns it open in small steps, closed loop on the pressure (keeping the inflow
# under about 9 kPa a second), lets go; grips the knob again, lifts the lid off 2 cm and carries it to the bench in front,
# sets it down. Left arm: puts the vial on the plate's middle. Right arm: carries the lid back and lowers it onto the
# flange, turns the vent shut. Then, with the left hand steadying the base again, the right hand swings the stopcock
# open, waits for -92 kPa on the gauge, swings it shut; finally pulls up gently on the knob, lets go, and both park.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7

TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.17, -0.24, TOP + 0.13)
KNOB_Z = TOP + FLANGE_TOP + 0.076 + KNOB_H - 0.007 + TIP_ABOVE_TCP  # pad tips 7 mm down the knob
KNOB_WIDE, KNOB_HOLD = -0.020, -0.0095  # round the 26 mm knob, squeezing it
LIFT = 0.020
LID_ASIDE = (0.16, -0.23)
STEADY_XY = (BX, BY + FLANGE_R - 0.004)  # on the flange's left edge, fingers closing along x so the hand stays
#   clear of the knob
STEADY_Z = TOP + FLANGE_TOP - 0.003 + TIP_ABOVE_TCP  # closed fingertips 3 mm into the flange: pressing on it
VIAL_GRIP_Z = TOP + 0.022 + TIP_ABOVE_TCP
VIAL_WIDE, VIAL_HOLD = -0.020, -0.008
VIAL_CARRY_Z = TOP + FLANGE_TOP + 0.015 + 0.022 + TIP_ABOVE_TCP
POST_Z = TOP + STOP_Z + 0.020 + 0.003 - 0.004 + TIP_ABOVE_TCP
POST_WIDE = -0.008
VENT_RATE_AIM = 9.0
_live = {"vent": 0.0, "tl": None, "p": P0}


def _live_reset():
    _live.update(vent=0.0, tl=None, p=P0)


def _clock():
    return float(getattr(_demo, "_plan_clock", 0.0))


def _vent_target(dt: float) -> float:
    p = _live["p"]
    rate_now = VENT_K * vent_fraction(_live["vent"]) * math.sqrt(max(0.0, ATM - p) / ATM)
    if rate_now < VENT_RATE_AIM and p < ATM - 0.5:
        _live["vent"] = min(VENT_FULL, _live["vent"] + math.radians(8.0) * dt)
    return _live["vent"]


class _KnobYaw(float):
    """Wrist yaw read every step: ``base`` plus the vent loop's knob angle (advanced while ``turning``)."""

    def __new__(cls, base=0.0, turning=False, scale=1.0):
        obj = float.__new__(cls, base)
        obj.base, obj.turning, obj.scale = base, turning, scale
        return obj

    @property
    def rot(self):
        if self.turning:
            t = _clock()
            dt = 0.0 if _live["tl"] is None else max(0.0, t - _live["tl"])
            _live["tl"] = t
            _vent_target(dt)
        return _prev_grip(self.base + self.scale * _live["vent"])


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._rot = True
    _demo.grip_rotation = _grip


def _post(phi):
    """World xy of the stopcock lever's post with the lever turned phi (0 shut, pointing at the robots)."""
    ax, ay = BX + STOP_XY[0], BY + STOP_XY[1]
    r = LEVER_L - 0.004
    return ax - r * math.cos(phi), ay - r * math.sin(phi)


def _dur(wps):
    return sum(w.duration for w in wps)


def _steady(left, t_from, t_to, label):
    """Left hand: be pressing on the flange from t_from to t_to, then lift off."""
    sx, sy = STEADY_XY
    left.append(Waypoint(max(0.3, t_from - 2.0 - _dur(left)), left[-1].pos, left[-1].finger, "wait", left[-1].yaw))
    q = -math.pi / 2
    left += [Waypoint(1.2, (sx, sy, STEADY_Z + 0.03), FINGER_CLOSED, "above the base's flange", q),
             Waypoint(0.8, (sx, sy, STEADY_Z), FINGER_CLOSED, label, q)]
    left.append(Waypoint(max(0.3, t_to - _dur(left)), (sx, sy, STEADY_Z), FINGER_CLOSED, label, q))
    left.append(Waypoint(0.8, (sx, sy, STEADY_Z + 0.04), FINGER_CLOSED, "off the flange", q))


def demo():
    _live_reset()
    kx, ky = BX, BY
    right = [
        Waypoint(2.5, (kx, ky, KNOB_Z + 0.03), KNOB_WIDE, "above the vent knob"),
        Waypoint(1.2, (kx, ky, KNOB_Z), KNOB_WIDE, "fingers round the knob"),
    ]
    t_vent0 = _dur(right)
    right += [
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_HOLD, "grip the knob"),
        Waypoint(0.1, (kx, ky, KNOB_Z), KNOB_HOLD, "turn it open slowly, watching the gauge", _KnobYaw(0.0, True)),
        Waypoint(13.0, (kx, ky, KNOB_Z), KNOB_HOLD, "turn it open slowly, watching the gauge", _KnobYaw(0.0, True)),
    ]
    t_vent1 = _dur(right)
    left = [Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "wait")]
    _steady(left, t_vent0, t_vent1 + 0.5, "steady the base: press on its flange")
    # lift the lid off by the knob (the vent stays open), set it down in front
    lid_xy = np.array([kx, ky])
    aside = np.array(LID_ASIDE)
    right += [
        Waypoint(1.5, (kx, ky, KNOB_Z + LIFT), KNOB_HOLD, "lift the lid off by its knob", _KnobYaw()),
        Waypoint(3.0, (*aside, KNOB_Z + LIFT), KNOB_HOLD, "carry it to the bench in front", _KnobYaw()),
        Waypoint(1.5, (*aside, KNOB_Z - FLANGE_TOP - 0.001), KNOB_HOLD, "set it down on its rim", _KnobYaw()),
        Waypoint(0.6, (*aside, KNOB_Z - FLANGE_TOP - 0.001), KNOB_WIDE, "let go", _KnobYaw()),
        Waypoint(1.0, (*aside, KNOB_Z - FLANGE_TOP + 0.03), KNOB_WIDE, "up", _KnobYaw()),
    ]
    t_lid_off = _dur(right)
    vx, vy = VIAL_XY
    left.append(Waypoint(max(0.3, t_lid_off - _dur(left)), LEFT_PARK, FINGER_OPEN, "wait for the lid to be off"))
    left += [
        Waypoint(2.5, (vx, vy, VIAL_GRIP_Z + 0.05), VIAL_WIDE, "above the vial"),
        Waypoint(1.2, (vx, vy, VIAL_GRIP_Z), VIAL_WIDE, "fingers round the vial"),
        Waypoint(0.6, (vx, vy, VIAL_GRIP_Z), VIAL_HOLD, "grip it"),
        Waypoint(1.2, (vx, vy, VIAL_CARRY_Z), VIAL_HOLD, "lift it over the flange"),
        Waypoint(2.5, (BX, BY, VIAL_CARRY_Z), VIAL_HOLD, "over the plate's middle"),
        Waypoint(2.0, (BX, BY, VIAL_GRIP_Z + PLATE_Z + 0.001), VIAL_HOLD, "stand it on the plate"),
        Waypoint(0.6, (BX, BY, VIAL_GRIP_Z + PLATE_Z + 0.001), VIAL_WIDE, "let go"),
        Waypoint(1.5, (BX, BY, VIAL_CARRY_Z + 0.03), VIAL_WIDE, "up out of the base"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "out of the way"),
    ]
    t_vial = _dur(left) - 2.0
    right.append(Waypoint(max(0.3, t_vial - _dur(right)), (*aside, KNOB_Z - FLANGE_TOP + 0.03), KNOB_WIDE,
                          "wait for the vial", _KnobYaw()))
    right += [
        Waypoint(1.2, (*aside, KNOB_Z - FLANGE_TOP - 0.001), KNOB_WIDE, "fingers round the knob", _KnobYaw()),
        Waypoint(0.6, (*aside, KNOB_Z - FLANGE_TOP - 0.001), KNOB_HOLD, "grip it", _KnobYaw()),
        Waypoint(1.2, (*aside, KNOB_Z + LIFT), KNOB_HOLD, "lift the lid", _KnobYaw()),
        Waypoint(3.0, (*lid_xy, KNOB_Z + LIFT), KNOB_HOLD, "over the base", _KnobYaw()),
        Waypoint(2.0, (*lid_xy, KNOB_Z - 0.0015), KNOB_HOLD, "lower it onto the flange, centred", _KnobYaw()),
    ]
    t_shut0 = _dur(right)
    right += [
        Waypoint(2.5, (kx, ky, KNOB_Z - 0.0015), KNOB_HOLD, "turn the vent shut", -0.04),
        Waypoint(0.6, (kx, ky, KNOB_Z - 0.0015), KNOB_WIDE, "let go", 0.0),
        Waypoint(1.0, (kx, ky, KNOB_Z + 0.03), KNOB_WIDE, "up", 0.0),
    ]
    t_shut1 = _dur(right) - 1.6
    # stopcock: open, wait for the pump, shut
    px, py = _post(0.0)
    right += [Waypoint(2.0, (px, py, POST_Z + 0.03), POST_WIDE, "above the stopcock lever's post", math.pi / 2),
              Waypoint(1.0, (px, py, POST_Z), POST_WIDE, "fingers round the post", math.pi / 2)]
    t_stop0 = _dur(right)
    for phi in np.linspace(0.25, STOP_OPEN, 6):
        x, y = _post(phi)
        right.append(Waypoint(0.5, (x, y, POST_Z), POST_WIDE, "swing the stopcock open", math.pi / 2 + 0.35 * phi))
    right.append(Waypoint(7.0, (x, y, POST_Z), POST_WIDE, "pump down: wait for -90 kPa or lower", math.pi / 2 + 0.35 * phi))
    for phi in np.linspace(STOP_OPEN - 0.25, -0.03, 6):
        x, y = _post(phi)
        right.append(Waypoint(0.5, (x, y, POST_Z), POST_WIDE, "swing the stopcock shut", math.pi / 2 + 0.35 * phi))
    t_stop1 = _dur(right)
    right.append(Waypoint(0.8, (x, y, POST_Z + 0.03), POST_WIDE, "up off the post", math.pi / 2 + 0.35 * phi))
    _steady(left, t_shut0, t_shut1, "steady the base while the vent is shut")
    _steady(left, t_stop0, t_stop1, "steady the base while the stopcock turns")
    right += [
        Waypoint(2.0, (kx, ky, KNOB_Z + 0.03), KNOB_WIDE, "above the knob"),
        Waypoint(1.2, (kx, ky, KNOB_Z), KNOB_WIDE, "fingers round the knob"),
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_HOLD, "grip it"),
        Waypoint(1.5, (kx, ky, KNOB_Z + 0.006), KNOB_HOLD, "pull up gently: the lid is held"),
        Waypoint(0.6, (kx, ky, KNOB_Z + 0.006), KNOB_WIDE, "let go"),
        Waypoint(1.0, (kx, ky, KNOB_Z + 0.04), KNOB_WIDE, "up"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"))
    t_end = max(_dur(left), _dur(right)) + 3.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
