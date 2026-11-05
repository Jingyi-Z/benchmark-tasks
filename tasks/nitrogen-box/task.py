"""On the station table: a small acrylic nitrogen dry box (a desiccator cabinet), fixed, its front facing the robots, and
a sample jar standing on the bench to its left.

The dry box has one front door, hinged on its right edge, that opens towards the robots. A door closer (a spring)
swings it shut whenever it is let go, and a magnetic catch on the left wall holds it shut. A black lever handle sticks
out from the door near its free (left) edge, with a wider end on it. Inside, on the floor shelf, a white square marks
where the sample goes.

The task: open the door by its handle and hold it open with one hand; with the other, take the jar, stand it upright on
the white square, and take the hand out; then let the door close so the catch holds it shut. Two hands: the door swings
shut by itself as soon as the handle is let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

ACRYLIC = (0.80, 0.88, 0.92)  # clear acrylic, slightly blue
STEEL = (0.70, 0.71, 0.73)
BLACK = (0.06, 0.06, 0.07)
DARK = (0.18, 0.19, 0.21)
WHITE = (0.97, 0.97, 0.97)
BRASS = (0.78, 0.62, 0.28)
JAR_C = (0.92, 0.92, 0.88)  # natural polypropylene
CAP_C = (0.15, 0.30, 0.70)
LABEL_C = (0.95, 0.55, 0.12)
if hasattr(labkit, "GLASS_COLORS"):  # a labkit copy that draws see-through glass shows the acrylic clear
    labkit.GLASS_COLORS.add(ACRYLIC)

# -- the dry box (fixed). Frame: origin on the table at the middle of the cabinet's front edge, x away from the robots,
# y to their left. -----------------------------------------------------------------------------------------------------
CX, CY = 0.33, -0.03
W, D = 0.18, 0.16  # outside width (y) and depth (x)
WALL = 0.006
BASE_T = 0.008  # acrylic base plate
SHELF_T = 0.004  # steel shelf plate on it
FLOOR = BASE_T + SHELF_T  # the shelf's top, 12 mm above the table
ROOF_UNDER = FLOOR + 0.120  # 12 cm of room inside
ROOF_TOP = ROOF_UNDER + WALL
IN_X0, IN_X1 = 0.0, D - WALL  # inside, x: from the door to the back wall
IN_Y = W / 2 - WALL  # inside half width
MARK_X, MARK = 0.070, 0.060  # the white 6 cm square, its centre 7 cm behind the door

# Door: a 5 mm acrylic plate over the whole front, hinged on its right edge; opening swings its free edge towards the
# robots (positive angle). The closer pulls it back to 0 (shut); the stop at 0 is the cabinet's front.
DOOR_T = 0.005
DOOR_Z0, DOOR_Z1 = 0.003, ROOF_TOP
HINGE = (-DOOR_T / 2, -W / 2)
DOOR_OPEN_MAX = math.radians(100.0)
CLOSER_K, CLOSER_C = 0.20, 0.06  # N m/rad, N m s/rad: about 0.33 N m held open at 95 degrees; shuts in about a second

# Lever handle near the free edge: a black bar straight out from the door, ending in a wider block.
HANDLE_Y = W / 2 - 0.025
HANDLE_Z = 0.080  # the bar's middle
NECK = (0.065, 0.010, 0.012)  # out from the door face (x), across (y), tall (z)
NECK_X0 = -DOOR_T  # the door's outer face
KNOB = (0.010, 0.022, 0.022)
KNOB_X = NECK_X0 - NECK[0] + KNOB[0] / 2 - 0.002  # the end block, its centre
KNOB_Z = HANDLE_Z + NECK[2] / 2 - KNOB[2] / 2  # flush with the bar's top, reaching 10 mm below it

DOOR = Joint(
    "door", kind="hinge", axis="Z", anchor=(*HINGE, (DOOR_Z0 + DOOR_Z1) / 2), lower=0.0, upper=DOOR_OPEN_MAX,
    initial=0.0, stiffness=CLOSER_K, damping=CLOSER_C, target=0.0, mass=0.18,
    boxes=[
        ((-DOOR_T / 2, 0.0, (DOOR_Z0 + DOOR_Z1) / 2), (DOOR_T, W, DOOR_Z1 - DOOR_Z0), ACRYLIC),
        ((NECK_X0 - 0.0015, HANDLE_Y, HANDLE_Z - 0.004), (0.003, 0.026, 0.034), BLACK),  # handle's base plate
        ((NECK_X0 - NECK[0] / 2, HANDLE_Y, HANDLE_Z), NECK, BLACK),
        ((KNOB_X, HANDLE_Y, KNOB_Z), KNOB, BLACK),
        ((0.0004, W / 2 - WALL - 0.006, HANDLE_Z), (0.0008, 0.012, 0.020), STEEL),  # strike plate for the catch
    ],
)

CABINET = part(
    "drybox", pos=(CX, CY, TOP), fixed=True, friction=0.6,
    boxes=[
        ((D / 2, 0.0, BASE_T / 2), (D, W, BASE_T), ACRYLIC),
        (((IN_X0 + IN_X1) / 2 + 0.001, 0.0, BASE_T + SHELF_T / 2), (IN_X1 - IN_X0 - 0.004, 2 * IN_Y - 0.004, SHELF_T),
         STEEL),
        ((MARK_X, 0.0, FLOOR + 0.0002), (MARK, MARK, 0.0004), WHITE),
        ((D / 2, W / 2 - WALL / 2, (BASE_T + ROOF_UNDER) / 2), (D, WALL, ROOF_UNDER - BASE_T), ACRYLIC),  # left wall
        ((D / 2, -W / 2 + WALL / 2, (BASE_T + ROOF_UNDER) / 2), (D, WALL, ROOF_UNDER - BASE_T), ACRYLIC),  # right
        ((D - WALL / 2, 0.0, (BASE_T + ROOF_UNDER) / 2), (WALL, W - 2 * WALL, ROOF_UNDER - BASE_T), ACRYLIC),  # back
        ((D / 2, 0.0, (ROOF_UNDER + ROOF_TOP) / 2), (D, W, WALL), ACRYLIC),  # roof
        ((0.004, IN_Y - 0.006, HANDLE_Z), (0.006, 0.012, 0.020), DARK),  # magnetic catch, inside the left wall
        ((D - WALL - 0.002, 0.04, 0.085), (0.004, 0.035, 0.035), WHITE),  # humidity card on the back wall
    ],
    cylinders=[
        ((HINGE[0], HINGE[1], 0.030), 0.004, 0.020, STEEL),  # hinge knuckles
        ((HINGE[0], HINGE[1], ROOF_TOP - 0.030), 0.004, 0.020, STEEL),
        ((D + 0.010, -0.050, 0.100), 0.004, 0.020, BRASS, "X"),  # nitrogen inlet, out of the back wall
        ((D + 0.010, 0.050, 0.030), 0.004, 0.020, BRASS, "X"),  # purge outlet
    ],
    joints=[DOOR],
)

# -- the sample jar (loose): a 4 cm polypropylene jar with a blue screw cap, standing on the bench ----------------------
JAR_R, JAR_BODY, CAP_R, CAP_H = 0.020, 0.048, 0.021, 0.012
JAR_H = JAR_BODY + CAP_H
JAR_XY = (0.24, 0.17)
JAR = part(
    "jar", pos=(*JAR_XY, TOP), mass=0.04, friction=0.8,
    cylinders=[
        ((0.0, 0.0, JAR_BODY / 2), JAR_R, JAR_BODY, JAR_C),
        ((0.0, 0.0, JAR_BODY + CAP_H / 2), CAP_R, CAP_H, CAP_C),
        ((0.0, 0.0, 0.026), JAR_R + 0.0003, 0.018, LABEL_C),  # label band
    ],
)

OBJECTS = [CABINET, JAR]


# -- scoring ----------------------------------------------------------------------------------------------------------
SHUT = math.asin(0.005 / W)  # the free edge within 5 mm of the cabinet's front: the catch holds it (1.4 degrees)
MARK_TOL = 0.010  # the jar's base centre within 1 cm of the square's centre along x and y: its base wholly on the square
UPRIGHT_DEG = 10.0
REST_TOL = 0.005  # the jar's base within 5 mm of the shelf
MARK_WORLD = (CX + MARK_X, CY)
SUCCESS_HOLD_S = 3.0  # end a trial early only once it has stayed done for 3 s


def door_gap(angle: float) -> float:
    """How far (m) the door's free edge stands off the cabinet's front."""
    return W * math.sin(max(0.0, angle))


def _inside(jar) -> bool:
    """The jar's middle inside the cabinet: behind the door's line, between the walls, under the roof."""
    c = jar.to_world((0.0, 0.0, JAR_H / 2)) - np.array([CX, CY, TOP])
    return bool(IN_X0 < c[0] < IN_X1 and abs(c[1]) < IN_Y and FLOOR < c[2] < ROOF_UNDER)


def _on_mark(jar) -> bool:
    base = jar.to_world((0.0, 0.0, 0.0))
    return bool(abs(base[0] - MARK_WORLD[0]) <= MARK_TOL and abs(base[1] - MARK_WORLD[1]) <= MARK_TOL
                and abs(base[2] - (TOP + FLOOR)) <= REST_TOL and jar.tilt_deg() <= UPRIGHT_DEG)


def score(objects):
    """0 unless the jar is inside the cabinet (its middle behind the door's line, between the walls, under the roof).
    Inside but not on the mark (base centre more than 1 cm from the white square's centre along x or y, base more than
    5 mm off the shelf, or tilted more than 10 degrees): 0.5 with the door shut, 0.25 with it open. On the mark: 1.0 with
    the door shut, 0.75 with it open. Shut: the door's free edge within 5 mm of the cabinet's front."""
    jar, door = objects["jar"], objects["drybox"].joints["door"]
    if not _inside(jar):
        return 0.0
    shut = door <= SHUT
    if _on_mark(jar):
        return 1.0 if shut else 0.75
    return 0.5 if shut else 0.25


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The jar's base centre from the square's centre (m, horizontal) plus the door's gap at its free edge."""
    base = objects["jar"].to_world((0.0, 0.0, 0.0))
    return math.hypot(base[0] - MARK_WORLD[0], base[1] - MARK_WORLD[1]) + door_gap(objects["drybox"].joints["door"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm: grips the handle's bar from above (fingers closing across it, pads 4 cm out from the door) and carries it
# round the hinge to 95 degrees open, the wrist turning with the door; the end block keeps the bar from sliding out of
# the fingers, against the closer. It holds the door there while the left arm works, then carries it back to 8 degrees
# and opens its fingers: the closer shuts the door (the end block passes between the open fingers) and the catch holds
# it. Left arm: grips the jar from the side, fingers level, pointing away from the robots and tipped 15 degrees down;
# lifts it 12 mm, carries it in front of the doorway, slides it in over the shelf, sets it on the white square, lets go
# and backs out. The task file lets a Waypoint's ``yaw`` carry a full gripper rotation (``Orient``).


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_orient", False):
    def _grip_rotation(yaw=0.0):
        return yaw.rot if isinstance(yaw, Orient) else _down_grip(yaw)

    _grip_rotation._orient = True
    _demo.grip_rotation = _grip_rotation

PITCH = math.radians(15.0)  # the side grip's tip tipped down: level reach is short of the joint limits this low
FORWARD = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -math.pi / 2 + PITCH) @ _down_grip(0.0)
AHEAD = np.array([math.cos(PITCH), 0.0, -math.sin(PITCH)])  # the side grip's pointing direction


def forward() -> Orient:
    """Side grip: fingers closing along y, pointing +x and tipped down by PITCH."""
    return Orient(FORWARD)


def W_(x, y, z):
    """Local cabinet coordinates to world."""
    return (CX + x, CY + y, TOP + z)


def door_point(local, theta: float):
    """World xy of a point on the door, given in cabinet coordinates with the door shut, with the door open by theta."""
    dx, dy = local[0] - HINGE[0], local[1] - HINGE[1]
    c, s = math.cos(theta), math.sin(theta)
    return CX + HINGE[0] + c * dx - s * dy, CY + HINGE[1] + s * dx + c * dy


TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
GRIP_OUT = 0.040  # pads centred 4 cm out from the door's face: the fingers stay clear of the door
HANDLE_GRIP = (NECK_X0 - GRIP_OUT, HANDLE_Y)
HANDLE_GRIP_Z = TOP + HANDLE_Z - NECK[2] / 2 - 0.003 + TIP_ABOVE_TCP  # pad tips 3 mm below the bar
HANDLE_HOVER = HANDLE_GRIP_Z + 0.04
HANDLE_WIDE = -0.012  # pads 24 mm apart round the 10 mm bar
HANDLE_HOLD = -0.002
HANDLE_LET_GO = -0.022  # pads 44 mm apart: the 22 mm end block passes between them as the door shuts
OPEN = math.radians(95.0)
ARC_STEP = math.radians(10.0)
LET_GO_AT = math.radians(8.0)

JAR_GRIP_H = 0.030  # pads centred 30 mm up the jar
JAR_WIDE = -0.032  # pads 64 mm apart round the 40 mm jar
JAR_HOLD = -0.014
JAR_RELEASE = -0.028
CARRY_UP = 0.012  # the jar carried with its base 12 mm above the shelf
PAD_AHEAD = 0.005  # the pads' middle, ahead of the TCP


def jar_tcp(x, y, base_z):
    """TCP for the side grip on a jar standing with its base centre at (x, y, base_z) (world): pads on its axis,
    JAR_GRIP_H up."""
    return tuple(np.array([x, y, base_z + JAR_GRIP_H]) - PAD_AHEAD * AHEAD)


def _back(p, d):
    """``p`` moved back d along the side grip's pointing direction."""
    return tuple(np.asarray(p) - d * AHEAD)


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("base and shelf", W_(D / 2, 0.0, FLOOR / 2), (D, W, FLOOR)),
    ("left wall", W_(D / 2, W / 2 - WALL / 2, ROOF_TOP / 2), (D, WALL, ROOF_TOP)),
    ("right wall", W_(D / 2, -W / 2 + WALL / 2, ROOF_TOP / 2), (D, WALL, ROOF_TOP)),
    ("back wall", W_(D - WALL / 2, 0.0, ROOF_TOP / 2), (WALL, W, ROOF_TOP)),
    ("roof", W_(D / 2, 0.0, (ROOF_UNDER + ROOF_TOP) / 2), (D, W, WALL)),
    ("door, drawn shut (it is open while the left arm is inside)", W_(-DOOR_T / 2, 0.0, (DOOR_Z0 + DOOR_Z1) / 2), (DOOR_T, W, DOOR_Z1 - DOOR_Z0)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def _arc(wps, th0, th1, label, finger):
    """Carry the handle round the hinge from th0 to th1 in steps of at most ARC_STEP, the wrist turning with it."""
    n = max(1, math.ceil(abs(th1 - th0) / ARC_STEP - 1e-9))
    for i in range(1, n + 1):
        th = th0 + (th1 - th0) * i / n
        wps.append(Waypoint(0.8, (*door_point(HANDLE_GRIP, th), HANDLE_GRIP_Z), finger, label, th))


def demo():
    # Right arm: open the door and hold it.
    h0 = door_point(HANDLE_GRIP, 0.0)
    right = [
        Waypoint(2.5, (*h0, HANDLE_HOVER), HANDLE_WIDE, "above the door handle"),
        Waypoint(1.2, (*h0, HANDLE_GRIP_Z), HANDLE_WIDE, "fingers either side of the bar"),
        Waypoint(0.8, (*h0, HANDLE_GRIP_Z), HANDLE_HOLD, "grip the handle"),
    ]
    _arc(right, 0.0, OPEN, "pull the door open round its hinge", HANDLE_HOLD)
    t_open = _dur(right)

    # Left arm: take the jar from the side, carry it in, set it on the square.
    jx, jy = JAR_XY
    pick = jar_tcp(jx, jy, TOP)
    lifted = jar_tcp(jx, jy, TOP + FLOOR + CARRY_UP)
    mx, my = MARK_WORLD
    place = jar_tcp(mx, my, TOP + FLOOR + 0.002)
    inside_high = jar_tcp(mx, my, TOP + FLOOR + CARRY_UP)
    doorway = jar_tcp(CX - JAR_R - 0.030, my, TOP + FLOOR + CARRY_UP)  # the jar 3 cm in front of the door's line
    fwd = forward()
    left = [
        Waypoint(2.5, (*_back(pick, 0.07)[:2], pick[2] + 0.04), JAR_WIDE, "behind the jar, fingers pointing at it", fwd),
        Waypoint(1.2, _back(pick, 0.07), JAR_WIDE, "down beside the bench", fwd),
        Waypoint(1.5, pick, JAR_WIDE, "fingers either side of the jar", fwd),
        Waypoint(0.8, pick, JAR_HOLD, "grip the jar", fwd),
        Waypoint(1.0, lifted, JAR_HOLD, "lift it 2.4 cm", fwd),
    ]
    left.append(Waypoint(max(0.3, t_open - 2.0 - _dur(left)), lifted, JAR_HOLD, "wait for the door", fwd))
    left += [
        Waypoint(2.0, doorway, JAR_HOLD, "in front of the doorway", fwd),
        Waypoint(2.0, inside_high, JAR_HOLD, "slide it in over the shelf", fwd),
        Waypoint(1.0, place, JAR_HOLD, "set it down on the white square", fwd),
        Waypoint(0.6, place, JAR_RELEASE, "let go", fwd),
        Waypoint(2.0, _back(place, 0.11), JAR_RELEASE, "back out of the cabinet", fwd),
        Waypoint(1.0, _back(place, 0.11)[:2] + (place[2] + 0.04,), JAR_WIDE, "up", fwd),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_out = _dur(left)  # the left arm is parked, clear of the swinging door

    right.append(Waypoint(max(0.3, t_out - _dur(right)), (*door_point(HANDLE_GRIP, OPEN), HANDLE_GRIP_Z), HANDLE_HOLD,
                          "hold the door open", OPEN))
    _arc(right, OPEN, LET_GO_AT, "ease the door back", HANDLE_HOLD)
    hl = door_point(HANDLE_GRIP, LET_GO_AT)
    right += [
        Waypoint(0.5, (*hl, HANDLE_GRIP_Z), HANDLE_LET_GO, "let go: the closer shuts the door", LET_GO_AT),
        Waypoint(1.0, (*hl, HANDLE_HOVER + 0.02), HANDLE_LET_GO, "up", LET_GO_AT),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 3.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
