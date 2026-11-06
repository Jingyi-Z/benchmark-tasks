"""On the station table: a polycarbonate vacuum desiccator holding a dish of a moisture-sensitive powder, under vacuum;
its vacuum line with an inline gauge on the bench to its right; and a sample vial on the bench to its left.

The desiccator: a white polypropylene bowl base, 17 cm across its flange, with a perforated white plate inside, and a
clear glass-like dome lid that sits on the flange (an O-ring seals it). On top of the dome, a clear knob is the VENT
valve: turned anticlockwise (seen from above) it lets air in, faster the further it is turned, a quarter turn fully open.
On the base's right side, a red stopcock connects the desiccator to the vacuum line through a hose barb: its 4 cm lever
pointing at the robots is shut; turned a quarter turn anticlockwise (pointing right) it is open to the pump, which runs
all the time. The gauge on the vacuum line reads the desiccator's pressure whenever the stopcock is open and holds its
last reading when it is shut. The base stands on rubber feet and does not slide when a valve is turned.

Under vacuum the air holds the lid down with far more force than a robot can pull. Venting faster than 12 kPa a second
blows the powder in the dish about: with the box at full vacuum that happens once the knob is past about 40 degrees.

The task: vent the desiccator gently; lift the dome off by its knob and hold it up clear of the base (it is not set down)
while the other hand stands the vial on the middle of the plate; lower the dome back centred on the flange; shut the
vent; open the stopcock until the gauge reads -90 kPa or lower, then shut it; pull up gently on the knob to check the
lid is held.

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
# A 24 cm desiccator, kept low (7 cm base, 6.5 cm dome) so the arms reach its knob.
BX, BY = 0.25, -0.06
BASE_KG = 2.0  # with its rubber feet, it does not turn on the bench when the stopcock is turned
FLANGE_R, FLANGE_TOP = 0.120, 0.070
WALL_R = 0.104  # inside of the wall at the top
PLATE_Z = 0.030  # the perforated plate's top
PLATE_R = 0.094
_sr, _sz = FLANGE_R / 0.085, FLANGE_TOP / 0.075
BASE_PROFILE = ([(r * _sr, z * _sz) for r, z in [(0.0, 0.0), (0.056, 0.0), (0.060, 0.004), (0.064, 0.020),
                                                  (0.070, 0.045), (0.074, 0.062), (0.076, 0.069)]]
                + [(FLANGE_R, 0.069 * _sz), (FLANGE_R, FLANGE_TOP), (WALL_R, FLANGE_TOP)]
                + [(r * _sr, z * _sz) for r, z in [(0.070, 0.050), (0.062, 0.020), (0.056, 0.006), (0.0, 0.006)]])
N_WALL = 40
MARK_XY = (0.0, 0.045)  # the vial's spot on the plate, a ring towards the robots' left
MARK_R = 0.016
# The stopcock stands out from the base's right side on a 2.5 cm port, so a hand on its lever stays clear of the dome.
STOP_ANGLE = -math.pi / 2
STOP_R = FLANGE_R + 0.025
STOP_Z = 0.045
STOP_XY = (STOP_R * math.cos(STOP_ANGLE), STOP_R * math.sin(STOP_ANGLE))
LEVER_L = 0.040
STOP_OPEN = math.pi / 2  # the lever's quarter turn anticlockwise (positive about z): open
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
    initial=0.0, target=0.0, stiffness=0.0, damping=0.05, friction=0.0, mass=0.01,  # PhysX joint friction scales with
    #   the load on the hinge, so a push on the post would lock it: plain damping instead
    boxes=[((STOP_XY[0] - LEVER_L / 2, STOP_XY[1], STOP_Z + 0.014), (LEVER_L, 0.008, 0.004), RED)],
    cylinders=[((STOP_XY[0], STOP_XY[1], STOP_Z + 0.013), 0.007, 0.006, RED),
               ((STOP_XY[0] - LEVER_L + 0.004, STOP_XY[1], STOP_Z + 0.024), 0.0045, 0.020, RED)],  # lever post
)
_port_len = STOP_R - (WALL_R + 0.004) + 0.004
_mark = [((MARK_XY[0] + MARK_R * math.cos(a), MARK_XY[1] + MARK_R * math.sin(a), PLATE_Z + 0.0002),
          (0.0025, 0.0025, 0.0004), DARK, a) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
BASE = part(
    "base", pos=(BX, BY, TOP), mass=BASE_KG, friction=0.8,
    meshes=[lathe_mesh(BASE_PROFILE, PP, segments=96),
            lathe_mesh([(0.0, PLATE_Z), (PLATE_R, PLATE_Z), (PLATE_R, PLATE_Z - 0.003), (0.0, PLATE_Z - 0.003)],
                       WHITE, segments=96)],
    boxes=[*_wall_boxes, *_mark,
           ((STOP_XY[0], STOP_XY[1] + _port_len / 2, STOP_Z), (0.010, _port_len, 0.010), PP),  # the port
           ((STOP_XY[0], STOP_XY[1] - 0.015, STOP_Z), (0.006, 0.020, 0.006), PP)],  # hose barb
    cylinders=[((0.0, 0.0, 0.003), 0.079, 0.006, PP),  # floor
               ((0.0, 0.0, PLATE_Z - 0.0015), PLATE_R, 0.003, WHITE),  # the plate
               ((0.0, 0.0, (0.006 + PLATE_Z - 0.003) / 2), 0.010, PLATE_Z - 0.009, WHITE),  # its centre leg
               *[((0.072 * math.cos(a), 0.072 * math.sin(a), PLATE_Z + 0.0002), 0.004, 0.0004, (0.80, 0.80, 0.80))
                 for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)],  # the plate's holes, drawn
               ((STOP_XY[0], STOP_XY[1], STOP_Z), 0.008, 0.016, WHITE)],  # stopcock body
    joints=[STOPCOCK],
)

# -- the lid (loose): a clear dome sitting on the flange, a solid knob on top to lift it by -------------------------
LID_KG = 0.45  # polycarbonate
DOME_H = 0.065
KNOB_R, KNOB_H = 0.016, 0.022
DOME_TOP = FLANGE_TOP + DOME_H + KNOB_H  # the knob's top above the table
_dr, _dz = FLANGE_R / 0.085, DOME_H / 0.076
LID_PROFILE = ([(FLANGE_R, 0.0), (FLANGE_R, 0.008), (FLANGE_R - 0.006, 0.010)]
               + [(r * _dr, z * _dz) for r, z in [(0.078, 0.025), (0.072, 0.045), (0.060, 0.060), (0.042, 0.069),
                                                  (0.024, 0.074)]]
               + [(0.020, DOME_H), (0.0, DOME_H)])
_rim_boxes = []
for _i in range(N_WALL):
    _a = 2 * math.pi * (_i + 0.5) / N_WALL
    _r = (FLANGE_R - 0.007 + FLANGE_R) / 2
    _rim_boxes.append(((_r * math.cos(_a), _r * math.sin(_a), 0.004), (0.007, 2 * math.pi * _r / N_WALL + 0.002, 0.008),
                       CLEAR, _a))
LID = part(
    "lid", pos=(BX, BY, TOP + FLANGE_TOP), mass=LID_KG, friction=1.0,
    meshes=[lathe_mesh(LID_PROFILE, CLEAR, segments=96)],
    boxes=_rim_boxes + [((KNOB_R * math.cos(a), KNOB_R * math.sin(a), DOME_H + KNOB_H / 2), (0.003, 0.003, KNOB_H),
                         CLEAR, a) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)],  # the knob's grip ribs
    cylinders=[((0.0, 0.0, DOME_H - 0.0015), 0.022, 0.003, CLEAR),  # the knob's boss
               ((0.0, 0.0, DOME_H + KNOB_H / 2), KNOB_R, KNOB_H, CLEAR)],  # the knob
)

# -- the vacuum line (fixed): a hose from the stopcock's barb to a manifold with the gauge and the pump's switch -------
LINE_XY = (BX, BY + STOP_XY[1] - 0.100)
BOX = (0.080, 0.040, 0.030)  # the manifold
GAUGE_X = -0.018  # the gauge on the manifold's near half, the switch on its far half
GAUGE_R = 0.020
GAUGE_TOP = BOX[2] + 0.025
NEEDLE_PER_KPA = math.radians(270.0) / 100.0
NEEDLE = Joint(
    "needle", kind="hinge", axis="Z", anchor=(GAUGE_X, 0.0, GAUGE_TOP + 0.001), lower=-4.8, upper=0.1, initial=0.0,
    target=0.0, stiffness=0.02, damping=0.0008, mass=0.001,
    boxes=[((GAUGE_X, 0.007, GAUGE_TOP + 0.0012), (0.0012, 0.015, 0.0008), RED)],
)
_ticks = []
for _k in range(11):  # every 10 kPa, from 0 (pointing +y, towards the desiccator) clockwise seen from above
    _a = -_k * 10 * NEEDLE_PER_KPA
    _ticks.append(((GAUGE_X - 0.0155 * math.sin(_a), 0.0155 * math.cos(_a), GAUGE_TOP + 0.0003),
                   (0.0012, 0.0012, 0.0004), BLACK if _k < 9 else RED))
# The pump switch: a rocker on the manifold's far end, pivot across it (about y). Red PUMP ON half away from the robots,
# black OFF half towards them. Off: the red end tilted up. Pressing an end down tips it there; it stays where pushed.
SWITCH_X = 0.020
ROCKER_PIVOT = (SWITCH_X, 0.0, BOX[2] + 0.004)
ROCKER = Joint(
    "pump", kind="hinge", axis="Y", anchor=ROCKER_PIVOT, lower=-0.25, upper=0.25, initial=-0.25,
    stiffness=0.0, damping=0.02, target=0.0,
    boxes=[((SWITCH_X + 0.008, 0.0, ROCKER_PIVOT[2]), (0.016, 0.018, 0.004), RED),
           ((SWITCH_X - 0.008, 0.0, ROCKER_PIVOT[2]), (0.016, 0.018, 0.004), BLACK)],
)
ON_ANGLE = 0.10  # the rocker counts as on past this (rad)
_hose_len = (BY + STOP_XY[1] - 0.025) - (LINE_XY[1] + BOX[1] / 2)
LINE = part(
    "vacuum_line", pos=(*LINE_XY, TOP), fixed=True, friction=0.6,
    boxes=[((0.0, 0.0, BOX[2] / 2), BOX, STEEL),  # manifold
           ((GAUGE_X, 0.0, (BOX[2] + GAUGE_TOP - 0.008) / 2), (0.008, 0.008, GAUGE_TOP - 0.008 - BOX[2]), STEEL),
           *_ticks,
           ((0.0, BOX[1] / 2 + _hose_len / 2, STOP_Z * 0.6), (0.010, _hose_len, 0.010), HOSE),  # hose up to the barb
           ((0.0, -BOX[1] / 2 - 0.025, 0.008), (0.010, 0.050, 0.010), HOSE)],  # on to the pump
    cylinders=[((GAUGE_X, 0.0, GAUGE_TOP - 0.004), GAUGE_R, 0.008, DARK),
               ((GAUGE_X, 0.0, GAUGE_TOP + 0.0001), GAUGE_R - 0.002, 0.0002, WHITE)],
    joints=[NEEDLE, ROCKER],
)

# -- the dish of powder inside, and the vial ------------------------------------------------------------------------
DISH_R, DISH_H = 0.022, 0.008
DISH_XY = (BX + 0.025, BY - 0.045)
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
# The pump starts off, and its line is then open to the room (its air-admittance valve). With the stopcock open, air
# flows in through the line while the pump is off (faster the further the stopcock is turned, a quarter turn fully
# open), and the box is pumped down while it is on. The gauge reads the line: the box's pressure while the stopcock is
# open, else the pump's (on) or the room's (off).
ATM = 101.3  # kPa
P0 = 8.0
VENT_K = 60.0  # kPa/s, stopcock fully open at full vacuum
PUMP_K = 0.6  # 1/s towards P_PUMP, stopcock fully open
P_PUMP = 3.0
LEAK = 0.002
GENTLE = 12.0
HOLD_MAX = 150.0  # N on a seated lid under vacuum (capped)
SEAL_TILT, SEAL_Z, SEAL_XY = 1.5, 0.0015, 0.004
_trial = {"p": P0, "gauge": ATM, "max_rate": 0.0, "blown": False, "ids": None}


def lid_sealed(lid, base) -> bool:
    """Seated on the flange: flat within 1.5 degrees, its rim within 1.5 mm of the flange's top, and its centre within
    4 mm of the base's axis."""
    rim = base.to_world((0.0, 0.0, FLANGE_TOP))
    rel = base.to_local(lid.pos)
    return (lid.tilt_deg() <= SEAL_TILT and abs(lid.pos[2] - rim[2]) <= SEAL_Z and math.hypot(rel[0], rel[1]) <= SEAL_XY)


def open_fraction(q: float) -> float:
    return float(np.clip(q / STOP_OPEN, 0.0, 1.0)) ** 2


def pump_on(line) -> bool:
    return line.joints["pump"] > ON_ANGLE


@on_reset
def _reset(env):
    _trial.update(p=P0, gauge=ATM, max_rate=0.0, blown=False)
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
    s = {n: labkit._part_state(env, labkit._SPECS[n]) for n in ("base", "lid", "vacuum_line")}
    sealed = lid_sealed(s["lid"], s["base"])
    f = open_fraction(s["base"].joints["stopcock"])
    on = pump_on(s["vacuum_line"])
    p = _trial["p"]
    rate = 0.0
    if not sealed:
        p = ATM
    else:
        if on:
            p -= PUMP_K * f * (p - P_PUMP) * dt
        else:
            rate = VENT_K * f * math.sqrt(max(0.0, ATM - p) / ATM)
            p += rate * dt
        p = min(ATM, p + LEAK * dt)
    _trial["p"] = p
    if s["base"].joints["stopcock"] >= math.radians(5.0):
        _trial["gauge"] = p
    else:
        _trial["gauge"] = P_PUMP if on else ATM
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
    _live["lid_xy"] = np.asarray(s["lid"].pos[:2], float)
    _live["base_xy"] = np.asarray(s["base"].pos[:2], float)


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
    """0 unless the vial stands upright on the plate inside the base and the dome is seated on the flange. 0.25 then;
    0.5 if the desiccator is also at -90 kPa or lower; 0.75 if, besides, the stopcock is shut (within 10 degrees) and the
    pump switched off; 1.0 if, besides, the vial's centre is within 1.5 cm of the ring's. Capped at 0.25 if the
    desiccator was ever vented faster than 12 kPa a second."""
    v, lid, base, line = objects["vial"], objects["lid"], objects["base"], objects["vacuum_line"]
    if not (vial_on_plate(v, base) and lid_sealed(lid, base)):
        return 0.0
    s = 0.25
    if -(ATM - _trial["p"]) <= TARGET_KPA:
        s = 0.5
        if base.joints["stopcock"] <= CLOSED_TOL and not pump_on(line):
            s = 0.75
            rel = base.to_local(v.pos)
            if math.hypot(rel[0] - MARK_XY[0], rel[1] - MARK_XY[1]) <= MARK_TOL:
                s = 1.0
    return min(s, 0.25) if _trial["blown"] else s


SUCCESS_HOLD_S = 4.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    v, base = objects["vial"], objects["base"]
    rel = base.to_local(v.pos)
    return math.hypot(rel[0] - MARK_XY[0], rel[1] - MARK_XY[1]) + max(0.0, -(ATM - _trial["p"]) - TARGET_KPA) * 0.001


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm: straddles the stopcock's post with its fingers and swings the lever open in small steps, closed loop on the
# pressure (keeping the inflow under about 9 kPa a second), then leaves it open; grips the dome's knob, lifts the dome and
# holds it up beside the base. Left arm (meanwhile): picks up the vial and, once the dome is up, stands it on the ring.
# Right arm: sets the dome back centred on the flange, lets go; presses the pump switch on; when the box is at about
# -95 kPa swings the stopcock shut; presses the switch off; pulls up gently on the knob, lets go; both park.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7

TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.17, -0.22, TOP + 0.14)
KNOB_Z = TOP + FLANGE_TOP + DOME_H + KNOB_H - 0.008 + TIP_ABOVE_TCP  # pad tips 8 mm down the knob
KNOB_WIDE, KNOB_HOLD = -0.032, -0.0125  # round the 35 mm knob (ribs included), squeezing it
LIFT = 0.020
KNOB_YAW = math.pi / 2  # fingers close along x on the knob
LID_HELD = (0.33, -0.22)  # where the right hand holds the dome while the vial goes in: beside the base, away from the
#   ring the vial goes on
HELD_UP = 0.050  # the dome's rim this far above the flange's top while held
VIAL_GRIP_Z = TOP + 0.022 + TIP_ABOVE_TCP
VIAL_WIDE, VIAL_HOLD = -0.020, -0.008
VIAL_CARRY_Z = TOP + FLANGE_TOP + 0.015 + 0.022 + TIP_ABOVE_TCP
POST_Z = TOP + STOP_Z + 0.019 + TIP_ABOVE_TCP  # tips 3 mm above the lever bar, 15 mm down the 20 mm post
POST_WIDE = -0.010
SWITCH_Z = TOP + BOX[2] + 0.004 - 0.003 + TIP_ABOVE_TCP  # closed tips 3 mm below the pivot, near the paddle's end
VENT_RATE_AIM = 9.0
_live = {"stop": 0.0, "tl": None, "p": P0, "lid_xy": None, "base_xy": None, "off": np.zeros(2), "t_back": 1e9}


def _live_reset():
    _live.update(stop=0.0, tl=None, p=P0, lid_xy=None, base_xy=None, off=np.zeros(2), t_back=1e9)


def _clock():
    return float(getattr(_demo, "_plan_clock", 0.0))


def _post(phi):
    """World xy of the stopcock lever's post with the lever turned phi (0 shut, pointing at the robots)."""
    ax, ay = BX + STOP_XY[0], BY + STOP_XY[1]
    r = LEVER_L - 0.004
    return ax - r * math.cos(phi), ay - r * math.sin(phi)


def _vent_step():
    """Advance the stopcock's opening while the inflow it gives is under the aim (it rises as the box fills)."""
    t = _clock()
    dt = 0.0 if _live["tl"] is None else max(0.0, t - _live["tl"])
    _live["tl"] = t
    p = _live["p"]
    rate_now = VENT_K * open_fraction(_live["stop"]) * math.sqrt(max(0.0, ATM - p) / ATM)
    if rate_now < VENT_RATE_AIM and p < ATM - 0.5:
        _live["stop"] = min(STOP_OPEN, _live["stop"] + math.radians(8.0) * dt)


class _PostAt:
    """The right hand on the lever's post at the vent loop's opening (advanced each step while ``venting``)."""

    def __init__(self, z, venting=False):
        self.z, self.venting = z, venting

    def __array__(self, dtype=None, copy=None):
        if self.venting:
            _vent_step()
        x, y = _post(_live["stop"])
        return np.asarray((x, y, self.z), dtype=dtype or float)


class _PostYaw(float):
    """The hand's yaw on the post: turned with the lever (60 percent of its angle; the post turns in the open fingers)."""

    def __new__(cls):
        return float.__new__(cls, 0.0)

    @property
    def rot(self):
        return _prev_grip(0.6 * _live["stop"])


class _Held:
    """Where the right hand holds the dome aside; also measures how far the hanging dome's centre sits from the hand
    (the knob slips a little in the grip) until the hand starts back, so the dome can be set back centred."""

    def __init__(self, z):
        self.z = z

    def __array__(self, dtype=None, copy=None):
        held = np.asarray(LID_HELD, float)
        if _live["lid_xy"] is not None and _clock() < _live["t_back"] - 0.2:
            _live["off"] = held - _live["lid_xy"]
        return np.asarray((held[0], held[1], self.z), dtype=dtype or float)


class _Back:
    """The right hand over the base, offset so the dome it holds comes down centred on the base."""

    def __init__(self, z):
        self.z = z

    def __array__(self, dtype=None, copy=None):
        b = np.asarray((BX, BY), float) if _live["base_xy"] is None else _live["base_xy"]
        xy = b + _live["off"]
        return np.asarray((xy[0], xy[1], self.z), dtype=dtype or float)


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._rot = True
    _demo.grip_rotation = _grip


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    py = _PostYaw()
    # 1. vent: swing the stopcock open slowly with the pump off
    px, py0 = _post(0.0)
    right = [
        Waypoint(2.5, (px, py0, POST_Z + 0.03), POST_WIDE, "above the stopcock lever's post", 0.0),
        Waypoint(1.0, (px, py0, POST_Z + 0.03), POST_WIDE, "settle over the post", 0.0),
        Waypoint(1.5, (px, py0, POST_Z), POST_WIDE, "fingers either side of the post", 0.0),
        Waypoint(0.1, _PostAt(POST_Z, True), POST_WIDE, "swing the stopcock open slowly, watching the gauge", py),
        Waypoint(17.0, _PostAt(POST_Z, True), POST_WIDE, "swing the stopcock open slowly, watching the gauge", py),
        Waypoint(0.8, _PostAt(POST_Z + 0.03), POST_WIDE, "up off the post", py),
    ]
    # 2. lift the dome off by its knob and hold it up beside the base
    kx, ky = BX, BY
    right += [
        Waypoint(2.5, (kx, ky, KNOB_Z + 0.03), KNOB_WIDE, "above the dome's knob", KNOB_YAW),
        Waypoint(1.0, (kx, ky, KNOB_Z + 0.03), KNOB_WIDE, "settle over the knob", KNOB_YAW),
        Waypoint(2.0, (kx, ky, KNOB_Z), KNOB_WIDE, "fingers round the knob", KNOB_YAW),
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_HOLD, "grip the knob", KNOB_YAW),
        Waypoint(1.5, (kx, ky, KNOB_Z + LIFT), KNOB_HOLD, "lift the dome off, level", KNOB_YAW),
        Waypoint(1.5, (kx, ky, KNOB_Z + HELD_UP), KNOB_HOLD, "lift it clear of the base", KNOB_YAW),
        Waypoint(3.0, _Held(KNOB_Z + HELD_UP), KNOB_HOLD, "hold it up beside the base, level", KNOB_YAW),
    ]
    t_lid_off = _dur(right)
    # the left hand picks the vial up meanwhile and waits with it
    vx, vy = VIAL_XY
    mx, my = BX + MARK_XY[0], BY + MARK_XY[1]
    left = [
        Waypoint(max(0.3, t_lid_off - 11.0), LEFT_PARK, FINGER_OPEN, "wait"),
        Waypoint(2.5, (vx, vy, VIAL_GRIP_Z + 0.05), VIAL_WIDE, "above the vial"),
        Waypoint(1.2, (vx, vy, VIAL_GRIP_Z), VIAL_WIDE, "fingers round the vial"),
        Waypoint(0.6, (vx, vy, VIAL_GRIP_Z), VIAL_HOLD, "grip it"),
        Waypoint(1.2, (vx, vy, VIAL_CARRY_Z), VIAL_HOLD, "lift it"),
    ]
    left.append(Waypoint(max(0.3, t_lid_off - _dur(left)), (vx, vy, VIAL_CARRY_Z), VIAL_HOLD,
                         "wait for the dome to be off"))
    left += [
        Waypoint(2.5, (mx, my, VIAL_CARRY_Z), VIAL_HOLD, "over the ring on the plate"),
        Waypoint(2.0, (mx, my, VIAL_GRIP_Z + PLATE_Z + 0.001), VIAL_HOLD, "stand it on the ring"),
        Waypoint(0.6, (mx, my, VIAL_GRIP_Z + PLATE_Z + 0.001), VIAL_WIDE, "let go"),
        Waypoint(1.5, (mx, my, VIAL_CARRY_Z + 0.03), VIAL_WIDE, "up out of the base"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "out of the way"),
    ]
    t_vial = _dur(left) - 2.0
    right.append(Waypoint(max(0.3, t_vial - _dur(right)), _Held(KNOB_Z + HELD_UP), KNOB_HOLD,
                          "hold the dome while the vial goes in", KNOB_YAW))
    _live["t_back"] = _dur(right)  # the offset is measured only until the hand starts back
    # 3. the dome back on, centred
    right += [
        Waypoint(3.0, _Back(KNOB_Z + HELD_UP), KNOB_HOLD, "back over the base", KNOB_YAW),
        Waypoint(1.5, _Back(KNOB_Z + LIFT), KNOB_HOLD, "down, level", KNOB_YAW),
        Waypoint(2.0, _Back(KNOB_Z - 0.0015), KNOB_HOLD, "lower it onto the flange, centred", KNOB_YAW),
        Waypoint(0.8, _Back(KNOB_Z - 0.0015), KNOB_HOLD, "let it settle on the seal", KNOB_YAW),
        Waypoint(0.6, _Back(KNOB_Z - 0.0015), KNOB_WIDE, "let go", KNOB_YAW),
        Waypoint(1.0, _Back(KNOB_Z + 0.03), KNOB_WIDE, "up", KNOB_YAW),
    ]
    # 4. pump down: switch the pump on (the stopcock is still open), wait, swing the stopcock shut, switch it off
    sx, sy = LINE_XY[0] + SWITCH_X, LINE_XY[1]
    on_x, off_x = sx + 0.012, sx - 0.012  # near each end of the 32 mm paddle
    right += [
        Waypoint(2.5, (on_x, sy, SWITCH_Z + 0.04), FINGER_CLOSED, "above the pump switch's red end", 0.0),
        Waypoint(1.0, (on_x, sy, SWITCH_Z), FINGER_CLOSED, "press it down: pump on", 0.0),
        Waypoint(0.6, (on_x, sy, SWITCH_Z + 0.04), FINGER_CLOSED, "up", 0.0),
    ]
    pxo, pyo = _post(STOP_OPEN)
    right += [
        Waypoint(2.0, (pxo, pyo, POST_Z + 0.03), POST_WIDE, "above the open lever's post, watching the gauge", py),
        Waypoint(1.0, (pxo, pyo, POST_Z + 0.03), POST_WIDE, "settle over the post", py),
        Waypoint(1.5, (pxo, pyo, POST_Z), POST_WIDE, "fingers either side of the post", py),
        Waypoint(3.0, (pxo, pyo, POST_Z), POST_WIDE, "wait for -90 kPa or lower on the gauge", py),
    ]
    for phi in np.linspace(STOP_OPEN - 0.25, -0.03, 6):
        x, y = _post(phi)
        right.append(Waypoint(0.5, (x, y, POST_Z), POST_WIDE, "swing the stopcock shut", 0.6 * phi))
    right += [
        Waypoint(0.8, (x, y, POST_Z + 0.03), POST_WIDE, "up off the post", 0.0),
        Waypoint(2.0, (off_x, sy, SWITCH_Z + 0.04), FINGER_CLOSED, "above the switch's black end", 0.0),
        Waypoint(1.0, (off_x, sy, SWITCH_Z), FINGER_CLOSED, "press it down: pump off", 0.0),
        Waypoint(0.6, (off_x, sy, SWITCH_Z + 0.04), FINGER_CLOSED, "up", 0.0),
    ]
    # 5. check the dome is held: pull up gently on the knob
    right += [
        Waypoint(2.5, (kx, ky, KNOB_Z + 0.03), KNOB_WIDE, "above the knob", KNOB_YAW),
        Waypoint(1.0, (kx, ky, KNOB_Z + 0.03), KNOB_WIDE, "settle over the knob", KNOB_YAW),
        Waypoint(2.0, (kx, ky, KNOB_Z), KNOB_WIDE, "fingers round the knob", KNOB_YAW),
        Waypoint(0.6, (kx, ky, KNOB_Z), KNOB_HOLD, "grip it", KNOB_YAW),
        Waypoint(1.5, (kx, ky, KNOB_Z + 0.006), KNOB_HOLD, "pull up gently: the dome is held", KNOB_YAW),
        Waypoint(0.6, (kx, ky, KNOB_Z + 0.006), KNOB_WIDE, "let go", KNOB_YAW),
        Waypoint(1.0, (kx, ky, KNOB_Z + 0.04), KNOB_WIDE, "up", KNOB_YAW),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 3.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
