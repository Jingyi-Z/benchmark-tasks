"""On the station table: a precision lab balance with a plastic weighing boat already on its pan, and an aluminium sample
puck on the table beside it.

The weighing procedure: tare with the empty boat on the pan, so the boat's own mass is subtracted, then put the sample
in the middle of the boat. Done in that order, the display shows the sample's mass alone. One arm presses TARE, the
other moves the sample.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import Joint, on_reset, on_step, part  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

CASE = (0.93, 0.93, 0.91)
DARK = (0.16, 0.17, 0.19)
STEEL = (0.70, 0.71, 0.73)
GREEN = (0.10, 0.55, 0.30)
BOAT_C = (0.92, 0.92, 0.95)
ALU = (0.78, 0.79, 0.80)
LCD = (0.55, 0.65, 0.55)

# -- the balance (fixed; frame origin on the table under the case centre) -------------------------------------------
BX, BY = 0.30, 0.06
BODY = (0.21, 0.19, 0.07)  # x (towards the robots is -x) by y by height
PAN_C = (0.02, 0.0)  # pan centre in the balance frame, a little behind the case centre
PAN_R, PAN_T = 0.06, 0.004
PAN_TOP = BODY[2] + 0.006 + PAN_T  # the pan sits on a 6 mm post: 8 cm above the table
KEY_X = -0.085  # the key strip along the front edge
TARE_Y = -0.055  # TARE key: on the right of the strip, seen by the robots
KEY = (0.016, 0.016, 0.004)
KEY_TRAVEL = 0.003
TARE = Joint(
    "tare", kind="slide", axis="Z", anchor=(KEY_X, TARE_Y, BODY[2]), lower=-KEY_TRAVEL, upper=0.0, initial=0.0,
    stiffness=60.0, damping=2.0, target=0.0, mass=0.01,  # a sprung key: about 1.8 N at the bottom of its travel
    boxes=[((KEY_X, TARE_Y, BODY[2] + KEY[2] / 2), KEY, GREEN)],
)
BALANCE = part(
    "balance", pos=(BX, BY, TOP), fixed=True, friction=0.8,
    boxes=[
        ((0.0, 0.0, BODY[2] / 2), BODY, CASE),
        ((KEY_X, 0.035, BODY[2] + 0.0005), (0.018, 0.07, 0.001), LCD),  # display window (static)
        ((KEY_X, 0.005, BODY[2] + 0.002), KEY, DARK),  # other keys (static: ON/OFF, MODE)
        ((KEY_X, -0.025, BODY[2] + 0.002), KEY, DARK),
    ],
    cylinders=[
        ((*PAN_C, BODY[2] + 0.003), 0.008, 0.006, STEEL),  # pan post
        ((*PAN_C, BODY[2] + 0.006 + PAN_T / 2), PAN_R, PAN_T, STEEL),  # pan
    ],
    joints=[TARE],
)

# -- the weighing boat (loose, on the pan) and the sample (loose, on the table) ---------------------------------------
BOAT = 0.07  # outer square
BOAT_FLOOR, BOAT_WALL, BOAT_H = 0.0015, 0.002, 0.014
BOAT_MASS = 0.004
_w = BOAT / 2 - BOAT_WALL / 2
BOAT_PART = part(
    "boat", pos=(BX + PAN_C[0], BY + PAN_C[1], TOP + PAN_TOP), mass=BOAT_MASS, friction=0.6,
    boxes=[
        ((0.0, 0.0, BOAT_FLOOR / 2), (BOAT, BOAT, BOAT_FLOOR), BOAT_C),
        ((_w, 0.0, BOAT_H / 2), (BOAT_WALL, BOAT, BOAT_H), BOAT_C),
        ((-_w, 0.0, BOAT_H / 2), (BOAT_WALL, BOAT, BOAT_H), BOAT_C),
        ((0.0, _w, BOAT_H / 2), (BOAT - 2 * BOAT_WALL, BOAT_WALL, BOAT_H), BOAT_C),
        ((0.0, -_w, BOAT_H / 2), (BOAT - 2 * BOAT_WALL, BOAT_WALL, BOAT_H), BOAT_C),
    ],
)
SAMPLE_R, SAMPLE_H, SAMPLE_MASS = 0.012, 0.025, 0.030  # an aluminium puck, 30 g
SX, SY = 0.15, 0.21
SAMPLE = part(
    "sample", pos=(SX, SY, TOP), mass=SAMPLE_MASS, friction=0.8,
    cylinders=[((0.0, 0.0, SAMPLE_H / 2), SAMPLE_R, SAMPLE_H, ALU)],
)

OBJECTS = [BALANCE, BOAT_PART, SAMPLE]

# -- scoring: the balance's reading, worked out from what is on the pan --------------------------------------------
PRESSED = -0.0015  # the key counts as pressed past half its travel
PAN_WORLD = (BX + PAN_C[0], BY + PAN_C[1])
CENTRE_TOL = 0.010
READ_TOL = 0.001  # kg: the display reads within 1 g of the sample's mass
SUCCESS_HOLD_S = 1.0
_tare = {"kg": 0.0, "pressed": False, "count": 0}


def _on_pan(state, half_height) -> bool:
    """Resting on the pan (directly, or in the boat on it): over the pan and less than 3 cm above it."""
    c = state.pos
    return (math.hypot(c[0] - PAN_WORLD[0], c[1] - PAN_WORLD[1]) <= PAN_R + 0.02
            and c[2] - half_height <= TOP + PAN_TOP + 0.03 and c[2] > TOP + PAN_TOP - 0.01)


def _pan_load(boat, sample) -> float:
    kg = 0.0
    if _on_pan(boat, 0.0):
        kg += BOAT_MASS
    if _on_pan(sample, 0.0) and sample.lin_vel is not None and abs(float(sample.lin_vel[2])) < 0.05:
        kg += SAMPLE_MASS
    return kg


@on_reset
def _clear(env):
    _tare.update(kg=0.0, pressed=False, count=0)


@on_step
def _watch_tare(env):
    import labkit

    s = {n: labkit._part_state(env, labkit._SPECS[n]) for n in ("balance", "boat", "sample")}
    down = s["balance"].joints["tare"] < PRESSED
    if down and not _tare["pressed"]:  # a new press: the balance zeroes on what is on the pan now
        _tare["kg"] = _pan_load(s["boat"], s["sample"])
        _tare["count"] += 1
    _tare["pressed"] = down


def reading(objects) -> float:
    """What the display shows, in kg."""
    return _pan_load(objects["boat"], objects["sample"]) - _tare["kg"]


def _in_boat(boat, sample) -> bool:
    d = sample.pos - boat.pos
    inner = BOAT / 2 - BOAT_WALL
    return (abs(d[0]) <= inner - SAMPLE_R + 0.001 and abs(d[1]) <= inner - SAMPLE_R + 0.001
            and sample.low[2] - boat.low[2] < 0.006 and sample.tilt_deg() < 15.0)


def score(objects):
    """0 if the sample is not resting upright in the boat on the pan. 0.25 if it is, but the display does not read the
    sample's 30 g within 1 g (the balance was not tared with the empty boat on it, or was tared again afterwards).
    0.5 if it reads 30 g but the sample is more than 1 cm from the pan's centre. 1.0 if it reads 30 g and the sample is
    within 1 cm of the pan's centre."""
    boat, sample = objects["boat"], objects["sample"]
    if not (_on_pan(boat, 0.0) and _in_boat(boat, sample)):
        return 0.0
    if abs(reading(objects) - SAMPLE_MASS) > READ_TOL:
        return 0.25
    off = math.hypot(sample.pos[0] - PAN_WORLD[0], sample.pos[1] - PAN_WORLD[1])
    return 1.0 if off <= CENTRE_TOL else 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    s = objects["sample"].pos
    return math.hypot(s[0] - PAN_WORLD[0], s[1] - PAN_WORLD[1])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm: presses TARE with closed fingers while the empty boat is on the pan, then parks.
# Left arm: waits at its park until the tare is done and the right arm is clear, then picks the sample puck up by its sides and lowers it into the middle of the boat.
TIP_ABOVE_TCP = 0.013
KEY_TOP = TOP + BODY[2] + KEY[2]
PRESS_Z = KEY_TOP - KEY_TRAVEL + 0.0005 + TIP_ABOVE_TCP  # pad tips 2.5 mm down: past the 1.5 mm press point, clear of the case
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
FINGER_WIDE = -0.03
FINGER_RELEASE = -0.02  # opens enough to free the puck, not so far that the fingers touch the boat walls
GRIP_Z = TOP + 0.008 + TIP_ABOVE_TCP  # pad tips 8 mm above the puck's base
CARRY_Z = TOP + PAN_TOP + BOAT_H + SAMPLE_H + 0.03 + TIP_ABOVE_TCP
PLACE_Z = TOP + PAN_TOP + BOAT_FLOOR + 0.002 + 0.008 + TIP_ABOVE_TCP  # base 2 mm above the boat floor

PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("balance", (BX, BY, TOP + BODY[2] / 2), BODY),
] + [  # the boat's four walls (its inside is free space for the fingers)
    (f"boat wall {i}", (BX + PAN_C[0] + dx, BY + PAN_C[1] + dy, TOP + PAN_TOP + BOAT_H / 2), size)
    for i, (dx, dy, size) in enumerate([
        (_w, 0.0, (BOAT_WALL, BOAT, BOAT_H)), (-_w, 0.0, (BOAT_WALL, BOAT, BOAT_H)),
        (0.0, _w, (BOAT, BOAT_WALL, BOAT_H)), (0.0, -_w, (BOAT, BOAT_WALL, BOAT_H)),
    ])
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    kx, ky = BX + KEY_X, BY + TARE_Y
    right = [
        Waypoint(2.0, (kx, ky, KEY_TOP + 0.05), FINGER_CLOSED, "above TARE"),
        Waypoint(1.2, (kx, ky, KEY_TOP + 0.004 + TIP_ABOVE_TCP), FINGER_CLOSED, "onto the key"),
        Waypoint(0.8, (kx, ky, PRESS_Z), FINGER_CLOSED, "press TARE (empty boat on the pan)"),
        Waypoint(0.6, (kx, ky, PRESS_Z), FINGER_CLOSED, "hold"),
        Waypoint(0.8, (kx, ky, KEY_TOP + 0.05), FINGER_CLOSED, "release"),
        Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_tared = _dur(right)
    px, py = PAN_WORLD
    left = [
        Waypoint(t_tared, LEFT_PARK, FINGER_OPEN, "wait while the right arm tares"),
        Waypoint(2.0, (SX, SY, CARRY_Z), FINGER_WIDE, "above the sample"),
        Waypoint(1.5, (SX, SY, GRIP_Z), FINGER_WIDE, "down round it"),
        Waypoint(0.8, (SX, SY, GRIP_Z), FINGER_CLOSED, "grip its sides"),
        Waypoint(1.2, (SX, SY, CARRY_Z), FINGER_CLOSED, "lift"),
    ]
    left += [
        Waypoint(2.0, (px, py, CARRY_Z), FINGER_CLOSED, "over the middle of the pan"),
        Waypoint(1.5, (px, py, PLACE_Z), FINGER_CLOSED, "lower into the boat"),
        Waypoint(0.8, (px, py, PLACE_Z), FINGER_RELEASE, "let go"),
        Waypoint(1.0, (px, py, CARRY_Z), FINGER_RELEASE, "up"),
        Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    return {"left": left, "right": right}
