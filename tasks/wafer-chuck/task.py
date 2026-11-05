"""On the station table: a benchtop spin coater for small wafers and, to its left, a wafer tray holding a 2-inch
silicon wafer on a raised pad.

The coater: a white housing with a round grey catch bowl on top and a black vacuum chuck, 3 cm across, raised in the
middle of the bowl. A ring of white dots on the bowl floor sits 2 mm outside the edge of a centred wafer. The bowl is
covered by a lid on an L-shaped lever that pivots on a pin to the right of the bowl. The lever's handle runs back from
the pin, away from the robots, and ends in a black post. Pushing the post to the right swings the lid back, off the
bowl; a spring swings it shut again when the post is let go, so one hand holds the lid open while the other loads the
wafer.

The task: hold the lid open, lay the wafer flat on the chuck with its centre within 1 mm of the chuck's axis (it may be
nudged by its edge to centre it), take both hands out and let the lid close.

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

HOUSING_C = (0.92, 0.92, 0.90)
BOWL_C = (0.62, 0.63, 0.66)  # grey polypropylene
CHUCK_C = (0.08, 0.08, 0.09)  # black PEEK
WHITE = (0.97, 0.97, 0.97)
LID_C = (0.72, 0.80, 0.86)  # tinted acrylic
DARK = (0.16, 0.17, 0.19)
BLACK = (0.06, 0.06, 0.07)
TRAY_C = (0.55, 0.56, 0.58)  # anodised aluminium
SI = (0.32, 0.34, 0.42)  # silicon

# -- the spin coater (fixed). Frame: origin on the table under the chuck's axis, x away from the robots, y to their left
CX, CY = 0.30, 0.02
HOUSING = (-0.075, 0.075, -0.115, 0.075)  # x from, x to, y from, y to
HOUSING_H = 0.030  # its top is the bowl floor
CHUCK_R, CHUCK_TOP = 0.015, 0.042  # the chuck: 3 cm across, 12 mm proud of the bowl floor
BOWL_IN = 0.050  # inside radius of the bowl wall
RIB_R, N_RIBS = 0.0035, 48  # the wall: 48 upright ribs 7 mm across, touching, in a ring
RIM = 0.062  # the wall's top
WAFER_R, WAFER_T = 0.0254, 0.0015  # 2-inch wafer; drawn 1.5 mm thick (real: 0.28 mm) for stable contact
DOT_R, N_DOTS = 0.0012, 24
DOT_RING = WAFER_R + 0.002 + DOT_R  # the dots' inner edges are 2 mm outside a centred wafer's edge

# The lid and its lever, one body on a hinge about the vertical through the pivot pin. Closed, the lid is centred over
# the bowl, its underside 1 mm above the rim; the lever's arm runs from the lid to the pivot, 9.5 cm right of the axis,
# and its handle runs 9 cm back from the pivot (+x), to the post. Opening turns it clockwise seen from above (the joint
# goes negative): the lid swings back, away from the robots, and the post swings out to the right, away from the bowl.
PIVOT_DY = 0.095
LID_R, LID_T = 0.058, 0.005
LID_Z0 = RIM + 0.001
HANDLE_L = 0.090
HANDLE_T = 0.004
POST_R, POST_H = 0.005, 0.020
POST_Z0 = LID_Z0 + HANDLE_T  # the post stands on the handle's top
LID_MAX = 1.70  # the lid's stop when open (rad, about 97 degrees)
PIN_R = 0.005


def _ring(r, n, rad, z0, h, colour):
    return [((r * math.cos(2 * math.pi * (i + 0.5) / n), r * math.sin(2 * math.pi * (i + 0.5) / n), z0 + h / 2), rad,
             h, colour) for i in range(n)]


LID = Joint(
    "lid", kind="hinge", axis="Z", anchor=(0.0, -PIVOT_DY, LID_Z0), lower=-LID_MAX, upper=0.0, initial=0.0,
    stiffness=0.2, damping=0.04, target=0.0, mass=0.08,  # spring: 0.2 N m/rad, about 3 N at the post held at 80 deg
    cylinders=[
        ((0.0, 0.0, LID_Z0 + LID_T / 2), LID_R, LID_T, LID_C),  # the lid
        ((0.0, -PIVOT_DY, LID_Z0 + 0.003), 0.009, 0.006, DARK),  # hub over the pin
        ((HANDLE_L - 0.003, -PIVOT_DY, POST_Z0 + POST_H / 2), POST_R, POST_H, BLACK),  # the post
    ],
    boxes=[
        ((0.0, -(PIVOT_DY + LID_R - 0.008) / 2, LID_Z0 + LID_T / 2), (0.016, PIVOT_DY - LID_R + 0.008, LID_T), DARK),
        ((HANDLE_L / 2, -PIVOT_DY, LID_Z0 + HANDLE_T / 2), (HANDLE_L, 0.008, HANDLE_T), DARK),  # handle
    ],
)
COATER = part(
    "coater", pos=(CX, CY, TOP), fixed=True, friction=0.9,
    boxes=[
        (((HOUSING[0] + HOUSING[1]) / 2, (HOUSING[2] + HOUSING[3]) / 2, HOUSING_H / 2),
         (HOUSING[1] - HOUSING[0], HOUSING[3] - HOUSING[2], HOUSING_H), HOUSING_C),
        ((-0.062, 0.055, HOUSING_H + 0.0005), (0.020, 0.030, 0.001), DARK),  # control panel
    ],
    cylinders=[
        ((0.0, 0.0, HOUSING_H + 0.0005), BOWL_IN + 2 * RIB_R, 0.001, BOWL_C),  # bowl floor
        ((0.0, 0.0, (HOUSING_H + CHUCK_TOP) / 2), CHUCK_R, CHUCK_TOP - HOUSING_H, CHUCK_C),  # the chuck
        *_ring(BOWL_IN + RIB_R, N_RIBS, RIB_R, HOUSING_H, RIM - HOUSING_H, BOWL_C),  # the bowl wall
        *_ring(DOT_RING, N_DOTS, DOT_R, HOUSING_H + 0.001, 0.0004, WHITE),  # centring dots
        ((0.0, -PIVOT_DY, (HOUSING_H + LID_Z0 - 0.0005) / 2), PIN_R, LID_Z0 - 0.0005 - HOUSING_H, DARK),  # pivot pin
    ],
    joints=[LID],
)

# -- the wafer tray (fixed) and the wafer (loose) on its pad ---------------------------------------------------------
TRAY_XY = (0.20, 0.21)
TRAY = (0.080, 0.080, 0.004)
PAD_R, PAD_TOP = 0.012, 0.025  # the pad is narrower than the wafer, so fingers can close on the wafer's edge
TRAY_PART = part(
    "tray", pos=(*TRAY_XY, TOP), fixed=True, friction=0.6,
    boxes=[((0.0, 0.0, TRAY[2] / 2), TRAY, TRAY_C)],
    cylinders=[((0.0, 0.0, (TRAY[2] + PAD_TOP) / 2), PAD_R, PAD_TOP - TRAY[2], TRAY_C)],
)
WAFER = part("wafer", pos=(*TRAY_XY, TOP + PAD_TOP), mass=0.004, friction=0.6,
             cylinders=[((0.0, 0.0, WAFER_T / 2), WAFER_R, WAFER_T, SI)])

OBJECTS = [COATER, TRAY_PART, WAFER]

# -- scoring ----------------------------------------------------------------------------------------------------------
CHUCK_XY = np.array([CX, CY])
CHUCK_Z = TOP + CHUCK_TOP
FLAT_DEG = 2.0
SEAT_TOL = 0.001  # the wafer's underside within 1 mm of the chuck's top
CENTRE_TOL, NEAR_TOL = 0.001, 0.003
LID_SHUT = math.radians(3.0)


def wafer_offset(w) -> float:
    """Horizontal distance (m) from the wafer's centre to the chuck's axis."""
    return float(np.hypot(*(w.pos[:2] - CHUCK_XY)))


def on_chuck(w) -> bool:
    """Flat on the chuck: tilted 2 degrees or less, its underside within 1 mm of the chuck's top, its centre over the
    chuck (within its 15 mm radius)."""
    return w.tilt_deg() <= FLAT_DEG and abs(w.pos[2] - CHUCK_Z) <= SEAT_TOL and wafer_offset(w) <= CHUCK_R


def lid_open(j: dict) -> float:
    """How far the lid is open (rad)."""
    return -j["lid"]


def score(objects):
    """0 unless the wafer rests flat on the chuck (tilt up to 2 degrees, underside within 1 mm of the chuck's top,
    centre within the chuck's 15 mm radius). 0.25 on the chuck; 0.5 with its centre within 3 mm of the chuck's axis;
    1.0 within 1 mm. Halved if the lid is open by more than 3 degrees."""
    w = objects["wafer"]
    if not on_chuck(w):
        return 0.0
    e = wafer_offset(w)
    s = 1.0 if e <= CENTRE_TOL else 0.5 if e <= NEAR_TOL else 0.25
    if lid_open(objects["coater"].joints) > LID_SHUT:
        s *= 0.5
    return s


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The wafer's centre from the chuck's axis plus its underside's height off the chuck's top (m)."""
    w = objects["wafer"]
    return wafer_offset(w) + abs(float(w.pos[2]) - CHUCK_Z)


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm: comes down with its fingers part-open round the lever's post, the pads either side of it along its path
# (the fingers close along that path), and drives the post round its arc in steps of about 11 degrees, the trailing
# pad's flat inner face pushing it, out to 80 degrees open (the whole bowl uncovered); holds it there while the left arm
# loads the wafer and comes up out of the bowl, then takes it back along the arc so the lid closes, opens and lifts off.
# The pads hold the post over its top 16 mm and 26 mm along the lever, so it cannot slip out from under or beside them.
# Left arm: grips the wafer across its diameter by the edge, pad tips 3 mm below its underside beside
# the narrow pad, carries it over the chuck and lowers it to 3 mm above. There it centres the wafer closed loop: an
# ``on_step`` hook reads the wafer's position and moves the commanded hand position by part of the remaining offset
# from the chuck's axis each step (``_Live`` waypoints read it every control step). Then it lowers the wafer until its
# underside is on the chuck's top, lets go and lifts out.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
OPEN = 1.40  # how far the lid is held open (rad, 80 degrees): its near edge 6.4 cm from the chuck's axis, outside the
#              bowl wall (5.7 cm). The left hand's fingers would touch the lid below about 55 degrees; at 80 they
#              clear it by 26 mm. The stop is at 97.
OPEN_STEPS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, OPEN)  # waypoints along the arc, about 11 degrees apart
POST_GRIP_Z = TOP + POST_Z0 + 0.004 + TIP_ABOVE_TCP  # pad tips 4 mm above the handle: on the post's top 16 mm
POST_GAP = 0.016  # pads 16 mm apart round the 10 mm post
POST_WIDE = -POST_GAP / 2
POST_DOWN = -0.010  # 20 mm apart while coming down over it
POST_FREE = -0.016  # 32 mm apart to lift off it
POST_PRELOAD = 0.001  # the trailing pad 1 mm into the post: it stays on it
WAFER_WIDE = -0.029  # pads 58 mm apart round the 50.8 mm wafer
WAFER_HOLD = -0.021  # squeezes it: the pads stop on its edge at 25.4 mm
PICK_Z = TOP + PAD_TOP - 0.003 + TIP_ABOVE_TCP  # pad tips 3 mm below the wafer's underside, beside the 24 mm pad
CARRY_Z = TOP + RIM + 0.015 - 0.003 + TIP_ABOVE_TCP  # the wafer's underside 15 mm above the rim
CLEAR_Z = CARRY_Z + 0.02  # pad tips 2.6 cm above the lid's top: clear of it as it closes
HOVER_Z = TOP + CHUCK_TOP + 0.003 - 0.003 + TIP_ABOVE_TCP  # the wafer's underside 3 mm above the chuck
PLACE_Z = TOP + CHUCK_TOP - 0.003 + TIP_ABOVE_TCP  # the wafer's underside on the chuck's top
CENTRE_GAIN, CENTRE_STEP, CENTRE_MAX = 0.15, 0.0004, 0.006  # per control step: 15% of the offset, at most 0.4 mm;
#                                                              at most 6 mm in all


def W(x, y, z):
    """Local coater coordinates to world."""
    return (CX + x, CY + y, TOP + z)


def _post_at(phi: float):
    """World xy of the lever's post when the lid is open by phi."""
    r = HANDLE_L - 0.003
    return CX + r * math.cos(phi), CY - PIVOT_DY - r * math.sin(phi)


def _push_at(phi: float, lead: float | None = None):
    """TCP xy with the post between the part-open pads when the lid is open by phi: ``lead`` ahead of the post along
    its path, by default so that the trailing pad's inner face is POST_PRELOAD into the post (it pushes it)."""
    if lead is None:
        lead = POST_GAP / 2 - POST_R + POST_PRELOAD
    px, py = _post_at(phi)
    ux, uy = -math.sin(phi), -math.cos(phi)  # the way the post moves as the lid opens
    return px + lead * ux, py + lead * uy


def _push_yaw(phi: float) -> float:
    """Wrist yaw round the post: the fingers close along the post's path (along y at yaw 0, as at phi 0) and the pads'
    26 mm width lies along the lever."""
    return -phi


def _lid_centre(phi: float):
    return CX + PIVOT_DY * math.sin(phi), CY - PIVOT_DY + PIVOT_DY * math.cos(phi)


_place = {"dx": 0.0, "dy": 0.0, "t0": 0.0, "t1": 0.0}
_clock = {"t": 0.0}


@on_reset
def _centre_reset(env):
    _clock["t"] = 0.0
    _place.update(dx=0.0, dy=0.0)


@on_step
def _centre_step(env):
    import labkit

    _clock["t"] += float(getattr(env.unwrapped, "step_dt", 0.05))
    if not _place["t0"] <= _clock["t"] <= _place["t1"]:
        return
    w = labkit._part_state(env, labkit._SPECS["wafer"])
    step = np.clip(CENTRE_GAIN * (CHUCK_XY - w.pos[:2]), -CENTRE_STEP, CENTRE_STEP)
    _place["dx"] = float(np.clip(_place["dx"] + step[0], -CENTRE_MAX, CENTRE_MAX))
    _place["dy"] = float(np.clip(_place["dy"] + step[1], -CENTRE_MAX, CENTRE_MAX))


class _Live:
    """A TCP position read every control step: ``base`` moved by the centring correction."""

    def __init__(self, base):
        self.base = np.asarray(base, float)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.base + np.array([_place["dx"], _place["dy"], 0.0]), dtype=dtype or float)


_lc = _lid_centre(OPEN)
_ribs = _ring(BOWL_IN + RIB_R, N_RIBS, RIB_R, HOUSING_H, RIM - HOUSING_H, BOWL_C)
PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("housing", W((HOUSING[0] + HOUSING[1]) / 2, (HOUSING[2] + HOUSING[3]) / 2, HOUSING_H / 2),
     (HOUSING[1] - HOUSING[0], HOUSING[3] - HOUSING[2], HOUSING_H)),
    ("chuck", W(0.0, 0.0, (HOUSING_H + CHUCK_TOP) / 2), (2 * CHUCK_R, 2 * CHUCK_R, CHUCK_TOP - HOUSING_H)),
    *[(f"bowl rib {i}", W(c[0], c[1], c[2]), (2 * RIB_R, 2 * RIB_R, RIM - HOUSING_H))
      for i, (c, *_) in enumerate(_ribs) if i % 3 == 0],
    ("open lid", (_lc[0], _lc[1], TOP + LID_Z0 + LID_T / 2), (2 * LID_R, 2 * LID_R, LID_T)),
    ("pivot hub", W(0.0, -PIVOT_DY, LID_Z0 + 0.003), (0.018, 0.018, 0.006)),
    *[(f"open lid arm {k}", W(d * math.sin(OPEN), -PIVOT_DY + d * math.cos(OPEN), LID_Z0 + LID_T / 2),
       (0.016, 0.016, LID_T)) for k, d in enumerate((0.015, 0.030))],
    *[(f"open lever handle {k}", W(d * math.cos(OPEN), -PIVOT_DY - d * math.sin(OPEN), LID_Z0 + HANDLE_T / 2),
       (0.008, 0.008, HANDLE_T)) for k, d in enumerate((0.015, 0.030, 0.045, 0.060, 0.075))],
    ("tray",(*TRAY_XY, TOP + TRAY[2] / 2), TRAY),
    ("tray pad", (*TRAY_XY, TOP + PAD_TOP / 2), (2 * PAD_R, 2 * PAD_R, PAD_TOP)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _place.update(dx=0.0, dy=0.0)
    tx, ty = TRAY_XY
    right = [
        Waypoint(2.5, (*_push_at(0.0, 0.0), POST_GRIP_Z + 0.04), POST_DOWN, "above the lever post, fingers part-open",
                 _push_yaw(0.0)),
        Waypoint(1.0, (*_push_at(0.0, 0.0), POST_GRIP_Z), POST_DOWN, "down round the post", _push_yaw(0.0)),
        Waypoint(0.4, (*_push_at(0.0, 0.0), POST_GRIP_Z), POST_WIDE, "pads 16 mm apart round the post",
                 _push_yaw(0.0)),
    ]
    for phi in OPEN_STEPS:
        right.append(Waypoint(0.7, (*_push_at(phi), POST_GRIP_Z), POST_WIDE, "push the post: the lid swings back",
                              _push_yaw(phi)))
    right.append(Waypoint(0.5, (*_push_at(OPEN), POST_GRIP_Z), POST_WIDE, "hold the lid open", _push_yaw(OPEN)))
    t_open = _dur(right)

    left = [
        Waypoint(2.5, (tx, ty, PICK_Z + 0.05), FINGER_OPEN, "above the wafer"),
        Waypoint(1.2, (tx, ty, PICK_Z), WAFER_WIDE, "fingers either side of the wafer"),
        Waypoint(0.8, (tx, ty, PICK_Z), WAFER_HOLD, "grip its edge"),
        Waypoint(1.0, (tx, ty, CARRY_Z), WAFER_HOLD, "lift"),
    ]
    left.append(Waypoint(max(0.3, t_open - _dur(left)), (tx, ty, CARRY_Z), WAFER_HOLD, "wait for the lid to open"))
    # the lid is held open from t_open: the left hand goes over the bowl only after that
    hover, place, up = _Live((CX, CY, HOVER_Z)), _Live((CX, CY, PLACE_Z)), _Live((CX, CY, CLEAR_Z))
    left += [
        Waypoint(2.0, (CX, CY, CARRY_Z), WAFER_HOLD, "over the chuck"),
        Waypoint(1.5, (CX, CY, HOVER_Z), WAFER_HOLD, "lower into the bowl, 3 mm above the chuck"),
        Waypoint(0.05, hover, WAFER_HOLD, "centre the wafer over the chuck"),
    ]
    _place["t0"] = _dur(left)
    left += [
        Waypoint(2.5, hover, WAFER_HOLD, "centre the wafer over the chuck"),
        Waypoint(1.5, place, WAFER_HOLD, "lower it onto the chuck"),
        Waypoint(0.5, place, WAFER_HOLD, "steady"),
    ]
    _place["t1"] = _dur(left)
    left += [
        Waypoint(0.6, place, WAFER_WIDE, "let go"),
        Waypoint(1.2, up, WAFER_WIDE, "up out of the bowl"),
    ]
    t_left_clear = _dur(left)
    left.append(Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "park"))

    right.append(Waypoint(max(0.3, t_left_clear - _dur(right)), (*_push_at(OPEN), POST_GRIP_Z), POST_WIDE,
                          "hold the lid open until the left arm is clear", _push_yaw(OPEN)))
    for phi in reversed((0.0, *OPEN_STEPS[:-1])):
        right.append(Waypoint(0.7, (*_push_at(phi), POST_GRIP_Z), POST_WIDE, "ease off: the lid closes",
                              _push_yaw(phi)))
    shut = (*_push_at(-0.06), POST_GRIP_Z)  # the trailing pad 4 mm off the closed post, the leading one 2 mm
    right += [
        Waypoint(0.6, shut, POST_WIDE, "off the post: the lid shut", _push_yaw(-0.06)),
        Waypoint(0.4, shut, POST_FREE, "open the fingers", _push_yaw(-0.06)),
        Waypoint(0.8, (shut[0], shut[1], POST_GRIP_Z + 0.04), POST_FREE, "up off the post", _push_yaw(-0.06)),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    return {"left": left, "right": right}
