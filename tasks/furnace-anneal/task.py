"""Loading chips into a 1 inch tube furnace for an anneal (2D materials), after the lab's procedure.

A compact split-tube furnace stands in front of the robots, a quartz tube across it. The tube is still being
backfilled: its gas valve, on the end fitting at the far end, is open, and the gauge there reads a little over
atmospheric and rising. Its near end is shut by a quick-release cap. The cap can come off only while the gauge is in
its green band: below it, the pressure difference holds the cap on; above it (red), the tube is over-pressured. As soon
as the cap is off the gas must be shut. Then the lid is lifted, which shows the thermocouple's tip at the hot zone's
centre; one hand holds the tube (it slides and rolls) while the other pushes the alumina boat with three chips in with a
hook rod, stroke by stroke, until it sits under the thermocouple, without jolting the chips off their places. Then the
rod comes back out onto its rests, the lid is shut and the cap goes back on.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
DT = 0.05  # one control step at 20 Hz


QUARTZ = (0.86, 0.90, 0.92)
STAINLESS = (0.62, 0.63, 0.66)
STEEL = (0.72, 0.73, 0.75)
DARK = (0.20, 0.21, 0.23)
FURNACE_C = (0.86, 0.86, 0.84)
INSUL = (0.93, 0.92, 0.88)
CHIP_C = (0.42, 0.33, 0.58)
RED = (0.80, 0.12, 0.10)
WHITE = (0.96, 0.96, 0.96)
labkit.GLASS_COLORS.update({QUARTZ})
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
FONT = {}


def Rz(a):
    return axis_angle_matrix(Z, a)


def _boxes_mesh(boxes, colour):
    """One drawn-only mesh (no collision) from many boxes: (centre, size[, yaw]) tuples."""
    pts, tris = [], []
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    for box in boxes:
        c, h = np.asarray(box[0], float), np.asarray(box[1], float) / 2
        rot = Rz(box[2]) if len(box) > 2 else np.eye(3)
        b = len(pts)
        pts += [c + rot @ (h * [sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        tris += [(b + i, b + j, b + k) for i, j, k in faces]
    return labkit.Mesh(points=np.array(pts), tris=np.array(tris), color=colour, collide="none")


def _glyphs(text, pix):
    w = len(text) * 6 * pix - pix
    out = []
    for n, ch in enumerate(text):
        for r, row in enumerate(FONT[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    out.append((-w / 2 + (n * 6 + c + 0.5) * pix, (3 - r) * pix))
    return out


def _text(line, pix, cx, cy, z):
    """Letters printed on a flat face, read from the robots' side: the line runs along -y, letters' tops towards +x."""
    return [((cx + v, cy - u, z), (pix * 0.96, pix * 0.96, 0.0006)) for u, v in _glyphs(line, pix)]


def _prism(outline, z0, z1, colour):
    n = len(outline)
    pts = [(*p, z1) for p in outline] + [(*p, z0) for p in outline] + [(0.0, 0.0, z1), (0.0, 0.0, z0)]
    ct, cb = 2 * n, 2 * n + 1
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(ct, i, j), (cb, n + j, n + i), (i, n + i, n + j), (i, n + j, j)]
    return labkit.Mesh(points=np.array(pts, float), tris=np.array(tris), color=colour, collide="none")


def _disc(r, n=96):
    return np.array([(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)])


def _arc_boxes(r0, r1, z0, z1, degs, width, colour):
    """Boxes along a ring between radii r0 and r1, one per angle in degs (tangential size ``width``)."""
    out = []
    for d in degs:
        a = math.radians(d)
        rm = (r0 + r1) / 2
        out.append(((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (r1 - r0, width, z1 - z0), colour, a))
    return out


def _tube_mesh(x0, x1, r_out, r_in, colour, n=48):
    """A drawn-only hollow cylinder along x from x0 to x1 (part frame, axis at y = 0, z = AXZ)."""
    pts, tris = [], []
    for x in (x0, x1):
        for r in (r_out, r_in):
            pts += [(x, r * math.cos(2 * math.pi * i / n), AXZ + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    o0, i0, o1, i1 = 0, n, 2 * n, 3 * n
    for i in range(n):
        j = (i + 1) % n
        tris += [(o0 + i, o1 + i, o1 + j), (o0 + i, o1 + j, o0 + j)]  # outside
        tris += [(i0 + i, i1 + j, i1 + i), (i0 + i, i0 + j, i1 + j)]  # inside
        tris += [(o0 + i, o0 + j, i0 + j), (o0 + i, i0 + j, i0 + i)]  # end at x0
        tris += [(o1 + i, i1 + j, o1 + j), (o1 + i, i1 + i, i1 + j)]  # end at x1
    return labkit.Mesh(points=np.array(pts, float), tris=np.array(tris), color=colour, collide="none")


_QUADS = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def _hex(pts, colour, collide="hull"):
    """A six-faced convex solid from its 8 corners (corner 4i + 2j + k), its faces wound outwards."""
    pts = np.asarray(pts, float)
    c, tris = pts.mean(axis=0), []
    for a, b, d, e in _QUADS:
        n = np.cross(pts[b] - pts[a], pts[d] - pts[a])
        f = [(a, b, d), (a, d, e)]
        if float(n @ (pts[[a, b, d, e]].mean(axis=0) - c)) < 0:
            f = [(a, d, b), (a, e, d)]
        tris += f
    return labkit.Mesh(points=pts, tris=np.array(tris), color=colour, collide=collide)


def _ring(x0, x1, r_in, r_out, n, colour, cy=0.0, cz=0.0, a0=0.0, a1=2 * math.pi, collide="hull", r_in1=None):
    """A ring (or an arc of one) round an axis along x through (cy, cz), from x0 to x1: n convex segments, each one
    colliding as itself (a hollow part's walls, round, where boxes would be square). ``r_in1``: the inside radius at x1,
    for a flared mouth."""
    out = []
    r_in1 = r_in if r_in1 is None else r_in1
    for i in range(n):
        aa = (a0 + (a1 - a0) * i / n, a0 + (a1 - a0) * (i + 1) / n)
        out.append(_hex([(x, cy + r * math.cos(a), cz + r * math.sin(a)) for x, ri in ((x0, r_in), (x1, r_in1))
                         for r in (ri, r_out) for a in aa], colour, collide))
    return out


def _box_mesh(c, s, colour, collide="none"):
    return _hex([(c[0] + (i - 0.5) * s[0], c[1] + (j - 0.5) * s[1], c[2] + (k - 0.5) * s[2]) for i in (0, 1)
                 for j in (0, 1) for k in (0, 1)], colour, collide)





def _ring_y(y0, y1, r_in, r_out, n, colour, cx=0.0, cz=0.0, a0=0.0, a1=2 * math.pi, collide="hull", r_in1=None):
    """A ring (or an arc of one) round an axis along y through (cx, cz), from y0 to y1, as n convex segments that each
    collide as themselves (a round hollow part, where boxes would be square). Angles from +x towards +z. ``r_in1``: the
    inside radius at y1 (a flared mouth)."""
    out = []
    r_in1 = r_in if r_in1 is None else r_in1
    for i in range(n):
        aa = (a0 + (a1 - a0) * i / n, a0 + (a1 - a0) * (i + 1) / n)
        out.append(_hex([(cx + r * math.cos(a), y, cz + r * math.sin(a)) for y, ri in ((y0, r_in), (y1, r_in1))
                         for r in (ri, r_out) for a in aa], colour, collide))
    return out


SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
       "8": "abcdefg", "9": "abcdfg"}


def _digits(text, x, y0, z0, h, colour):
    """Seven-segment digits on a face at x, facing -x (read from the robots' side): text from left (+y) to right."""
    w, t = 0.55 * h, 0.12 * h
    out = []
    for n, ch in enumerate(text):
        yc = y0 - n * (w + 0.35 * h)
        segs = {"a": (0, h / 2, w, t), "g": (0, 0, w, t), "d": (0, -h / 2, w, t),
                "f": (w / 2, h / 4, t, h / 2), "b": (-w / 2, h / 4, t, h / 2),
                "e": (w / 2, -h / 4, t, h / 2), "c": (-w / 2, -h / 4, t, h / 2)}
        for s in SEG.get(ch, ""):
            dy, dz, sy, sz = segs[s]
            out.append(((x, yc + dy, z0 + dz), (0.0008, sy, sz)))
    return _boxes_mesh(out, colour)


# -- the furnace (fixed): a compact split-tube furnace (after a 1 inch, 12 inch long laboratory model), its front towards
# the robots: a base with the controller, the heated lower half, and a lid hinged at the back that lifts by a handle on
# its front. Frame: on the bench. The tube runs across it along y. ---------------------------------------------------------
XA, AXZ = 0.355, 0.220  # the tube's axis: x, height above the bench
TUBE_R, TUBE_IN = 0.0127, 0.0110  # 1 inch quartz tube: 25.4 mm outside, 22 mm inside
TUBE_Y0, TUBE_Y1 = -0.225, 0.385  # the tube's open (loading) end, towards the right arm, and its far end
FX0, FX1, FY0, FY1 = 0.215, 0.495, -0.125, 0.285  # the furnace body: front, back, ends (410 mm wide, 280 mm deep)
BASE_Z = 0.110  # the base (controller) under the heated chamber
LID_Z1 = 0.360  # the lid's top
HOT_Y = (FY0 + FY1) / 2  # the hot zone's centre, where the thermocouple's tip is
HINGE = (FX1 - 0.010, AXZ)  # the lid's hinge along the back, at the split
SLOT = 0.019  # half the gap the tube lies in, front to back, in both halves
WHITE_F = (0.90, 0.90, 0.88)
PANEL_C = (0.40, 0.52, 0.70)
INSUL_C = (0.95, 0.93, 0.86)
COIL_C = (0.55, 0.42, 0.33)
SCREEN_C = (0.06, 0.07, 0.08)
LED_C = (0.95, 0.20, 0.12)
TC_C = (0.70, 0.71, 0.73)
GREEN = (0.15, 0.70, 0.25)
_mid_y, _len_y = (FY0 + FY1) / 2, FY1 - FY0
FURNACE_BOXES = [
    # base with the controller panel
    ((((FX0 + FX1) / 2), _mid_y, BASE_Z / 2), (FX1 - FX0, _len_y, BASE_Z), WHITE_F),
    ((FX0 - 0.0005, _mid_y, BASE_Z / 2), (0.001, _len_y - 0.02, BASE_Z - 0.02), PANEL_C),
    ((FX0 - 0.0012, _mid_y + 0.05, 0.060), (0.001, 0.090, 0.045), SCREEN_C),
    # the heated lower half: front and back blocks either side of the tube's gap, ends, a floor
    (((FX0 + XA - SLOT) / 2, _mid_y, (BASE_Z + AXZ) / 2), (XA - SLOT - FX0, _len_y, AXZ - BASE_Z), WHITE_F),
    (((XA + SLOT + FX1) / 2, _mid_y, (BASE_Z + AXZ) / 2), (FX1 - XA - SLOT, _len_y, AXZ - BASE_Z), WHITE_F),
    ((XA, _mid_y, BASE_Z + 0.01), (2 * SLOT, _len_y, 0.02), INSUL_C),
    *[((XA, y, (BASE_Z + AXZ - 0.030) / 2), (2 * SLOT, 0.020, AXZ - 0.030 - BASE_Z), WHITE_F)
      for y in (FY0 + 0.010, FY1 - 0.010)],
    # the thermocouple: in through the back wall, its tip beside the tube at the hot zone's centre (seen with the lid up)
    ((((XA + 0.020) + FX1) / 2 - 0.02, HOT_Y, AXZ - 0.004), ((FX1 - 0.04) - (XA + 0.020), 0.003, 0.003), TC_C),
]
FURNACE_CYLS = [((XA + 0.020, HOT_Y, AXZ - 0.004), 0.0028, 0.004, TC_C, "X")]  # the junction bead
# the tube lies on two saddles in the furnace's end walls: round, open on top
SADDLES = [m for y in (FY0, FY1 - 0.020) for m in _ring_y(y, y + 0.020, TUBE_R + 0.0006, TUBE_R + 0.008, 8, INSUL_C,
                                                        cx=XA, cz=AXZ, a0=math.radians(200), a1=math.radians(340))]
# drawn only: the heating coils in the lower half, the lid's inside, the temperature on the controller's screen
_coils = [((XA + s * (SLOT - 0.003), y, AXZ - 0.012), (0.003, 0.004, 0.012)) for s in (-1, 1)
          for y in np.arange(FY0 + 0.03, FY1 - 0.03, 0.012)]
FURNACE_MESHES = [*SADDLES, _boxes_mesh(_coils, COIL_C), _digits("28", FX0 - 0.0018, _mid_y + 0.075, 0.062, 0.022, LED_C)]

# the lid: front and back blocks either side of the tube's gap, the top over it; a handle bar on its front
LID_HANDLE_Y, LID_HANDLE_Z, LID_HANDLE_OUT = 0.240, 0.280, 0.010  # the handle bar: 10 mm out from the lid's front
_lid_boxes = [
    (((FX0 + XA - SLOT) / 2, _mid_y, (AXZ + LID_Z1) / 2), (XA - SLOT - FX0, _len_y, LID_Z1 - AXZ), WHITE_F),
    (((XA + SLOT + FX1) / 2, _mid_y, (AXZ + LID_Z1) / 2), (FX1 - XA - SLOT, _len_y, LID_Z1 - AXZ), WHITE_F),
    ((XA, _mid_y, (AXZ + 0.030 + LID_Z1) / 2), (2 * SLOT, _len_y, LID_Z1 - AXZ - 0.030), WHITE_F),
    *[((FX0 - LID_HANDLE_OUT / 2, y, LID_HANDLE_Z), (LID_HANDLE_OUT, 0.008, 0.010), DARK)
      for y in (LID_HANDLE_Y - 0.035, LID_HANDLE_Y + 0.035)],
    ((FX0 - LID_HANDLE_OUT, LID_HANDLE_Y, LID_HANDLE_Z), (0.008, 0.080, 0.008), DARK),
]
LID_UP = 1.0  # rad: how far the lid can lift (it stays where it is left: a stiff hinge, part sprung)
LID = Joint("lid", kind="hinge", axis="Y", anchor=(HINGE[0], _mid_y, HINGE[1]), lower=0.0, upper=LID_UP,
            stiffness=0.8, target=LID_UP, damping=0.6, friction=0.6, initial=0.0, mass=0.8, boxes=_lid_boxes)
FURNACE = part("tube_furnace", pos=(0.0, 0.0, TOP), fixed=True, friction=0.15, boxes=FURNACE_BOXES,
               cylinders=FURNACE_CYLS, meshes=FURNACE_MESHES, joints=[LID])

# -- the tube (loose): fused quartz, 25.4 mm outside, 22 mm inside, 610 mm long, open at its near end; at its far end a
# stainless end fitting with a compound gauge (its face towards the robots: -20 to +20 kPa, the safe band for opening
# the tube marked green, red above it) and the gas valve, a quarter-turn ball valve: its lever pointing at the robots is
# open, turned a quarter turn towards the furnace is shut. Frame: on the bench. ---------------------------------------
FIT_Y0, FIT_Y1, FIT_R = TUBE_Y1 - 0.030, TUBE_Y1 + 0.010, 0.017
GAUGE_C = (XA, (FIT_Y0 + FIT_Y1) / 2, AXZ + 0.075)  # the gauge's centre; its face at x - 0.012
GAUGE_R = 0.028
P_MIN, P_MAX = -20.0, 20.0  # kPa over the dial
P_LO, P_HI = 2.0, 8.0  # the green band: the cap comes off only above P_LO (below, the pressure difference holds it);
# above P_HI the tube is over-pressured (it can burst)
DIAL_SWEEP = math.radians(270.0)


def dial_angle(p):
    """The needle's angle about x (rad, 0 = straight up), for pressure p (kPa): P_MIN at -135 deg, P_MAX at +135 deg,
    clockwise as the robots see it."""
    s = (min(max(p, P_MIN), P_MAX) - P_MIN) / (P_MAX - P_MIN)
    return -DIAL_SWEEP / 2 + s * DIAL_SWEEP


def _dial_arc(p0, p1, colour, r0=0.019, r1=0.026):
    """A coloured band on the dial between two pressures (drawn on its face)."""
    xs = GAUGE_C[0] - 0.0058
    out = []
    n = max(2, int(abs(dial_angle(p1) - dial_angle(p0)) / math.radians(6)) + 1)
    for i in range(n):
        a, b = dial_angle(p0 + (p1 - p0) * i / n), dial_angle(p0 + (p1 - p0) * (i + 1) / n)
        # angle 0 is up; clockwise seen from -x means towards -y... the needle turns about +x
        pts = []
        for x in (xs, xs + 0.0006):
            for r in (r0, r1):
                for ang in (a, b):
                    pts.append((x, GAUGE_C[1] - r * math.sin(ang), GAUGE_C[2] + r * math.cos(ang)))
        out.append(_hex(pts, colour, collide="none"))
    return out


VALVE_PIVOT = (XA - 0.040, (FIT_Y0 + FIT_Y1) / 2, AXZ + 0.016)
LEVER_L = 0.045
TUBE_MESHES = [*_ring_y(TUBE_Y0, FIT_Y0 + 0.005, TUBE_IN, TUBE_R, 24, QUARTZ, cx=XA, cz=AXZ),
               *_ring_y(FIT_Y0, FIT_Y1, TUBE_R, FIT_R, 16, STAINLESS, cx=XA, cz=AXZ),
               *_dial_arc(P_MIN, P_LO, WHITE), *_dial_arc(P_LO, P_HI, GREEN), *_dial_arc(P_HI, P_MAX, RED)]
TUBE_DRAWN = [
    _box_mesh((XA, FIT_Y1 + 0.002, AXZ), (0.020, 0.004, 0.020), STAINLESS),  # the far end's blank
    _box_mesh((XA, GAUGE_C[1], (AXZ + FIT_R + GAUGE_C[2] - GAUGE_R) / 2),
              (0.008, 0.008, GAUGE_C[2] - GAUGE_R - AXZ - FIT_R + 0.004), STAINLESS),  # the gauge's stem
    _box_mesh((XA - 0.025, VALVE_PIVOT[1], AXZ), (0.022, 0.010, 0.010), STAINLESS),  # the valve's port, to the front
    _box_mesh((VALVE_PIVOT[0], VALVE_PIVOT[1], AXZ + 0.002), (0.018, 0.016, 0.018), (0.78, 0.66, 0.30)),  # the valve
    *_ring(GAUGE_C[0] - 0.004, GAUGE_C[0] + 0.012, 0.0005, GAUGE_R + 0.002, 24, STAINLESS, cy=GAUGE_C[1],
           cz=GAUGE_C[2], collide="none"),  # the gauge's case
    *_ring(GAUGE_C[0] - 0.0050, GAUGE_C[0] - 0.0042, 0.0005, GAUGE_R, 24, WHITE, cy=GAUGE_C[1], cz=GAUGE_C[2],
           collide="none"),  # its face
]
TUBE_MESHES += TUBE_DRAWN
NEEDLE = Joint("needle", kind="hinge", axis="X", anchor=(GAUGE_C[0] - 0.006, GAUGE_C[1], GAUGE_C[2]),
               lower=-DIAL_SWEEP / 2 - 0.05, upper=DIAL_SWEEP / 2 + 0.05, stiffness=0.05, damping=0.003,
               target=0.0, initial=0.0, mass=0.002,
               meshes=[_box_mesh((GAUGE_C[0] - 0.0064, GAUGE_C[1], GAUGE_C[2] + 0.010), (0.001, 0.0016, 0.024), DARK)],
               boxes=[((GAUGE_C[0] - 0.0064, GAUGE_C[1], GAUGE_C[2]), (0.001, 0.003, 0.003), DARK)])
VALVE = Joint("vent_valve", kind="hinge", axis="Z", anchor=VALVE_PIVOT, lower=0.0, upper=math.pi / 2,
              stiffness=0.0, damping=0.02, friction=0.03, initial=0.0, mass=0.01,
              boxes=[((VALVE_PIVOT[0] - LEVER_L / 2, VALVE_PIVOT[1], VALVE_PIVOT[2] + 0.004), (LEVER_L, 0.010, 0.004),
                      RED)],
              cylinders=[((VALVE_PIVOT[0] - LEVER_L + 0.004, VALVE_PIVOT[1], VALVE_PIVOT[2] + 0.013), 0.005, 0.014, RED,
                          "Z")])
TUBE = part("quartz_tube", pos=(0.0, 0.0, TOP), mass=0.30, friction=0.15, meshes=TUBE_MESHES, joints=[NEEDLE, VALVE])
KNOB_AT = (VALVE_PIVOT[0] - LEVER_L + 0.004, VALVE_PIVOT[1], VALVE_PIVOT[2] + 0.013)  # the lever's knob (open)

# -- the boat (loose): an alumina boat 80 x 15 x 8 mm, open on top, its back end a 12 mm tab the rod's pusher meets. In the
# round bore it sits 8 mm under the axis. Frame: origin at the centre of its underside. ----------------------------------
BOAT_L, BOAT_W, BOAT_H, BOAT_T, BACK_H = 0.080, 0.015, 0.008, 0.0015, 0.012
ALUMINA = (0.97, 0.96, 0.93)
BOAT_Z = AXZ - math.sqrt(TUBE_IN ** 2 - (BOAT_W / 2) ** 2) * math.cos(math.pi / 24) + 0.0005
BOAT_START = np.array([XA, TUBE_Y0 + 0.006 + BOAT_L / 2, TOP + BOAT_Z])
BOAT = part("boat", pos=tuple(BOAT_START), mass=0.020, friction=0.5,
            boxes=[((0.0, 0.0, BOAT_T / 2), (BOAT_W, BOAT_L, BOAT_T), ALUMINA),
                   *[((e * (BOAT_W - BOAT_T) / 2, 0.0, BOAT_H / 2), (BOAT_T, BOAT_L, BOAT_H), ALUMINA) for e in (-1, 1)],
                   ((0.0, -(BOAT_L - BOAT_T) / 2, BACK_H / 2), (BOAT_W - 2 * BOAT_T, BOAT_T, BACK_H), ALUMINA),
                   ((0.0, (BOAT_L - BOAT_T) / 2, BOAT_H / 2), (BOAT_W - 2 * BOAT_T, BOAT_T, BOAT_H), ALUMINA)])

# -- the chips (loose): three 8 x 8 x 0.5 mm silicon chips (with 2D flakes on their oxide, which shows purple) in a row
CHIP, CHIP_T = 0.008, 0.0005
CHIP_DY = (-0.024, 0.0, 0.024)
CHIP_START = [BOAT_START + [0.0, dy, BOAT_T + CHIP_T / 2] for dy in CHIP_DY]
CHIPS = [part(f"chip{i}", pos=tuple(CHIP_START[i]), mass=0.00008, friction=0.8,
              boxes=[((0.0, 0.0, 0.0), (CHIP, CHIP, CHIP_T), CHIP_C)]) for i in range(3)]

# -- the hook rod (loose): stainless, 340 mm: a 10 mm pusher disc at its front for the boat's tab, 4 mm round for 100 mm,
# then a 6 mm square grip to its back end. Frame: origin on its axis at the pusher's front face, the rod along -y. ----
ROD_L, ROD_R, GRIP_W, PUSHER_R, ROUND_L = 0.340, 0.002, 0.006, 0.005, 0.100
ROD_START = np.array([XA, TUBE_Y0 - 0.045, TOP + AXZ - 0.0015])  # on its rests in line with the tube, 45 mm short of its end
ROD = part("hook_rod", pos=tuple(ROD_START), mass=0.090, friction=0.7,
           cylinders=[((0.0, -ROUND_L / 2, 0.0), ROD_R, ROUND_L, STEEL, "Y"),
                      ((0.0, -0.0015, 0.0), PUSHER_R, 0.003, STEEL, "Y")],
           boxes=[((0.0, -(ROUND_L + ROD_L) / 2, 0.0), (GRIP_W, ROD_L - ROUND_L, GRIP_W), DARK)])

# -- the rod's rests (fixed): two short troughs on posts, in line with the tube, 1.5 mm under the rod's grip (its back end,
# coming back over the near one, must not catch on it). --------------------------------------------------------------
REST_Y = (-0.350, -0.520)
REST_Z = AXZ - GRIP_W / 2 - 0.0015
REST = part("rod_rests", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5,
            boxes=[b for y in REST_Y for b in (
                ((XA, y, REST_Z / 2), (0.020, 0.014, REST_Z), DARK),
                *[((XA + e * 0.0075, y, REST_Z + 0.004), (0.004, 0.014, 0.008), DARK) for e in (-1, 1)])])

# -- the end cap (loose): an aluminium quick-release cap for the 1 inch tube: a 5 mm plate and a 25 mm sleeve that slides
# over the tube's end until the plate meets it (its O-ring inside then seals), knurled outside where it is held. Frame:
# origin on its axis at the plate's inner face, the sleeve along +y. ---------------------------------------------------
ALUM = (0.80, 0.81, 0.83)
CAP_IN, CAP_D, CAP_PLATE = TUBE_R + 0.0015, 0.025, 0.005
CAP_OUT = CAP_IN + 0.004  # 36.4 mm across
CAP_START = np.array([XA, TUBE_Y0, TOP + AXZ])  # on the tube's end
CAP_REST = np.array([0.250, -0.280, TOP + 0.020 + CAP_OUT + 0.0012])  # its stand, beside the tube's end
CAP = part("end_cap", pos=tuple(CAP_START), mass=0.06, friction=0.35,
           cylinders=[((0.0, -CAP_PLATE / 2, 0.0), CAP_OUT, CAP_PLATE, ALUM, "Y")],
           meshes=[*_ring_y(0.0, CAP_D - 0.004, CAP_IN, CAP_OUT, 24, ALUM),
                   *_ring_y(CAP_D - 0.004, CAP_D, CAP_IN, CAP_OUT, 24, ALUM, r_in1=CAP_IN + 0.0015)])
CAP_STAND = part("cap_stand", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5,
                 boxes=[((CAP_REST[0], CAP_REST[1] + CAP_D / 2, 0.010), (0.030, 0.030, 0.020), DARK)],
                 meshes=_ring_y(CAP_REST[1] + 0.002, CAP_REST[1] + CAP_D - 0.002, CAP_OUT + 0.0012, CAP_OUT + 0.005, 6,
                                DARK, cx=CAP_REST[0], cz=CAP_REST[2] - TOP, a0=math.radians(215), a1=math.radians(325)))

OBJECTS = [FURNACE, TUBE, BOAT, *CHIPS, ROD, REST, CAP, CAP_STAND]
PLANCHECK_BOXES = [
    ("furnace", (((FX0 + FX1) / 2), _mid_y, TOP + LID_Z1 / 2), (FX1 - FX0, _len_y, LID_Z1)),
]


# -- the gas and the cap --------------------------------------------------------------------------------------------
# The trial starts mid-backfill: the gas valve open, the tube 1 kPa over atmospheric and rising. With the cap on and the
# valve open the pressure rises 0.4 kPa/s; shutting the valve holds it. Below the green band (P_LO) the pressure
# difference holds the cap on (40 N); in the band only its O-ring does (1.5 N); over P_HI the tube is over-pressured.
# Once the cap is off the tube is at atmospheric; the valve must then be shut within VENT_OPEN_S.
P0, P_RATE = 1.0, 0.4  # kPa, kPa/s
CAP_HOLD_LOCKED, CAP_HOLD_ORING = 40.0, 1.5  # N
CAP_K = 4000.0  # N/m: the hold rises over the cap's first mm off its seat
CAP_ON, CAP_OFF = 0.010, 0.020  # the cap seals within 10 mm of its seat, is off past 20 mm
CAP_AREA = math.pi * TUBE_R ** 2
VENT_OPEN_S = 5.0
VALVE_SHUT = math.radians(80.0)


def _asset(env, name):
    return env.unwrapped.scene[name]


def _t(x):
    return getattr(x, "torch", x)


def _push_cap(env, fy):
    """A force fy (N) along +y on the cap (towards its seat): the pressure and the O-ring, which the sim does not have."""
    import torch

    a = _asset(env, "end_cap")
    q = labkit._part_state(env, labkit._SPECS["end_cap"]).rotation()
    f = q.T @ np.array([0.0, fy, 0.0])  # in the cap's frame
    dev = _t(a.data.root_link_pose_w).device
    F = torch.tensor(f, dtype=torch.float32, device=dev).reshape(1, 1, 3)
    try:
        a.set_external_force_and_torque(F, torch.zeros_like(F))
    except Exception as e:  # noqa: BLE001
        if not _gas.get("warned"):
            _gas["warned"] = True
            print(f"[furnace-anneal] cap force not applied: {e!r}")


def _show_pressure(env, p):
    """Turn the gauge's needle to the pressure (its joint's drive target)."""
    import torch

    a = _asset(env, "quartz_tube")
    try:
        i = list(a.joint_names).index(NEEDLE.joint_name)
        dev = _t(a.data.joint_pos).device
        a.set_joint_position_target(torch.tensor([[dial_angle(p)]], dtype=torch.float32, device=dev), joint_ids=[i])
    except Exception as e:  # noqa: BLE001
        if not _gas.get("warned_n"):
            _gas["warned_n"] = True
            print(f"[furnace-anneal] gauge not driven: {e!r}")


_gas = {}


@on_reset
def _reset_gas(env):
    _gas.clear()
    _gas.update(p=P0, removed=False, open_off=0.0)


def cap_out(c):
    """How far the cap is off its seat along the tube (m; 0 when home)."""
    return float(TUBE_Y0 - np.asarray(c.pos, float)[1])


@on_step
def _gas_step(env):
    S = labkit._SPECS
    c = labkit._part_state(env, S["end_cap"])
    t = labkit._part_state(env, S["quartz_tube"])
    valve_open = t.joints.get("vent_valve", 0.0) < VALVE_SHUT
    d = cap_out(c)
    p = _gas.get("p", P0)
    on = d < CAP_ON and math.hypot(c.pos[0] - XA, c.pos[2] - (TOP + AXZ)) < 0.006
    if on and not _gas.get("removed"):
        p += P_RATE * DT if valve_open else 0.0
    else:
        p *= math.exp(-DT / 0.3)  # open to the room
        if d > CAP_OFF:
            _gas["removed"] = True
    if on and _gas.get("removed"):
        p = 0.0  # back on after loading: at atmospheric
    _gas["p"] = p
    _gas["open_off"] = _gas.get("open_off", 0.0) + DT if (valve_open and d > CAP_ON) else 0.0
    # the hold on the cap: the pressure difference below the band, the O-ring above it; the pressure itself pushes it off
    hold = 0.0
    if d < CAP_OFF:
        locked = p < P_LO and not _gas.get("removed")
        hold = min(CAP_HOLD_LOCKED if locked else CAP_HOLD_ORING, CAP_K * max(d, 0.0))
    push = (p * 1000.0 * CAP_AREA) if d < CAP_ON else 0.0
    _push_cap(env, hold - push)
    _show_pressure(env, p)


# -- scoring ----------------------------------------------------------------------------------------------------------
CHIP_SLIDE = 0.003  # a chip that moves over 3 mm on the boat has slid (lost or scratched)
CHIP_FLIP = 30.0  # or tilted over 30 degrees
BOAT_TILT = 20.0
HOT_TOL = 0.008  # the boat centred under the thermocouple within 8 mm
TUBE_MOVE, TUBE_TURN = 0.003, 3.0  # the tube slid over 3 mm or rolled over 3 degrees
RACK_TOL = 0.015  # the rod back on its rests: within 15 mm of where it lay
LID_SHUT = math.radians(2.0)
SEAT_TOL, AXIS_TOL, CAP_TILT = 0.002, 0.003, 3.0
HANDS_FAR = 0.08
SETTLE_STEPS, SETTLE_TOL = 20, 0.001

_state = {}
_hist = []


@on_reset
def _reset_state(env):
    _state.clear()
    _state.update(step=0, viol=set(), settled=False, far=True, centred=False, racked=False, lid=True, sealed=False)
    _hist.clear()


def chip_offsets(b, chips):
    """Each chip's place and tilt on the boat: (offset from its starting place on the boat (m), tilt (degrees))."""
    Rb = b.rotation()
    out = []
    for i, c in enumerate(chips):
        q = Rb.T @ (np.asarray(c.pos, float) - np.asarray(b.pos, float))
        q0 = np.array([0.0, CHIP_DY[i], BOAT_T + CHIP_T / 2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float((Rb.T @ c.rotation())[2, 2])))))
        out.append((float(np.linalg.norm(q - q0)), tilt))
    return out


def axis_off(state, axis=1):
    """How far a part's own y axis is off the world's y axis (degrees)."""
    a = state.rotation()[:, axis]
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(a[1]))))))


def rod_racked(r):
    p = np.asarray(r.pos, float)
    return float(np.linalg.norm(p - ROD_START)) <= RACK_TOL and axis_off(r) <= 5.0


def cap_sealed(c, tube):
    p = np.asarray(c.pos, float)
    return (abs(cap_out(c)) <= SEAT_TOL and math.hypot(p[0] - XA, p[2] - (TOP + AXZ)) <= AXIS_TOL
            and axis_off(c) <= CAP_TILT and tube.joints.get("vent_valve", 0.0) >= VALVE_SHUT)


def _tube_ref(t):
    return np.asarray(t.pos, float), t.rotation()


@on_step
def _watch(env):
    """Latch the violations and follow the end state."""
    S = labkit._SPECS
    st = _state
    st["step"] += 1
    b, r, c = (labkit._part_state(env, S[n]) for n in ("boat", "hook_rod", "end_cap"))
    t, f = labkit._part_state(env, S["quartz_tube"]), labkit._part_state(env, S["tube_furnace"])
    chips = [labkit._part_state(env, S[f"chip{i}"]) for i in range(3)]
    if st["step"] == 10:
        st["tube0"] = _tube_ref(t)  # where the tube lies once settled
    for d, tilt in chip_offsets(b, chips):
        if d > CHIP_SLIDE:
            st["viol"].add("a chip slid on the boat")
        if tilt > CHIP_FLIP:
            st["viol"].add("a chip flipped")
    if b.tilt_deg() > BOAT_TILT:
        st["viol"].add("the boat tipped")
    if _gas.get("p", 0.0) > P_HI:
        st["viol"].add("the tube was over-pressured")
    if _gas.get("open_off", 0.0) > VENT_OPEN_S:
        st["viol"].add("the gas ran with the tube open")
    if "tube0" in st:
        p0, R0 = st["tube0"]
        turn = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ t.rotation()) - 1) / 2))))
        if np.linalg.norm(np.asarray(t.pos, float) - p0) > TUBE_MOVE or turn > TUBE_TURN:
            st["viol"].add("the tube slid or rolled")
    bp = np.asarray(b.pos, float)
    st["centred"] = abs(bp[1] - HOT_Y) <= HOT_TOL
    st["racked"] = rod_racked(r)
    st["lid"] = f.joints.get("lid", 0.0) <= LID_SHUT
    st["sealed"] = cap_sealed(c, t)
    tips = _tips(env)
    keep = [np.asarray(r.pos, float), np.asarray(c.pos, float), np.array([FX0 - LID_HANDLE_OUT, LID_HANDLE_Y, TOP + LID_HANDLE_Z])]
    st["far"] = all(np.linalg.norm(np.asarray(tp) - k) >= HANDS_FAR for tp in tips for k in keep)
    _hist.append(np.concatenate([bp, np.asarray(r.pos, float), np.asarray(c.pos, float),
                                 *[np.asarray(x.pos, float) for x in chips], [f.joints.get("lid", 0.0) * 0.1]]))
    del _hist[:-(SETTLE_STEPS + 1)]
    st["settled"] = (len(_hist) > SETTLE_STEPS
                     and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 after any violation (the tube over-pressured; the gas left running over 5 s with the cap off; a chip slid over
    3 mm on the boat or tilted over 30 degrees; the boat tipped over 20 degrees; the tube slid over 3 mm or rolled over
    3 degrees), unless everything has been still (under 1 mm) for the last second, or unless the boat's centre is within
    8 mm of the thermocouple along the tube. Then 0.2, plus 0.2 each for: the rod back on its rests (within 15 mm of
    where it lay), the lid shut (within 2 degrees), the cap home and square (its plate within 2 mm of the tube's end, its
    axis within 3 mm and 3 degrees of the tube's) with the gas valve shut. 1.0 if all three, and every fingertip is 8 cm
    from the rod, the cap and the lid's handle."""
    st = _state
    if st.get("viol") or not st.get("settled", False) or not st.get("centred"):
        return 0.0
    parts = [bool(st.get("racked")), bool(st.get("lid")), bool(st.get("sealed"))]
    if all(parts) and st.get("far", True):
        return 1.0
    return 0.2 + 0.2 * sum(parts)


SUCCESS_HOLD_S = 2.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return float(abs(np.asarray(objects["boat"].pos, float)[1] - HOT_Y))



# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right takes the cap, left the gas valve's knob; with the gauge in the green right pulls the cap off and left shuts
# the gas at once. Left lifts the lid; right sets the cap on its stand. Left holds the tube while right pushes the boat
# in with the rod, two strokes, until the boat is seen under the thermocouple; then pulls the rod back out onto its
# rests. Left shuts the lid; right puts the cap back on, in line with the tube (seen), fully home.
# Aims are trimmed only by what a camera sees: the boat, the cap.

from yam_arena import ik as _ik  # noqa: E402
from yam_arena.constants import READY as _READY  # noqa: E402


def _tips(env):
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7"], preserve_order=True)[0]
        half = abs(float(robot.data.joint_pos.torch[0, ids].cpu().numpy()[0]))
        out += [mid + half * g[:3, 0], mid - half * g[:3, 0]]
    return out


def tilt_deg(R, down=False):
    c = float(R[2, 2]) * (-1.0 if down else 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))

def _ik_err(pe, re):
    return pe * 1000.0 + math.degrees(re)


FREE_MOVES = ("wrist turned", "hand turned", "up and clear")


def _endpoints(plan):
    """Joint angles for every waypoint's end, chained: warm-started from the previous one, and, where that solve misses,
    the closest of several seeded solves that reach it (so the wrist turns the way the waypoints mean)."""
    rng = np.random.default_rng(7)
    seeds = [np.array(_READY, float)] + [np.array(_READY, float) + rng.normal(0.0, 0.8, 6) for _ in range(10)]
    q_prev, out, switch, forced = plan.q.copy(), [], set(), set()
    for i, w in enumerate(plan.waypoints):
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, q_prev, base_pos=plan.base_pos, iters=300)
        if any(m in w.label for m in FREE_MOVES):
            forced.add(i)  # a free-space move: always a joint-space blend
        if _ik_err(pe, re) > 1.5:
            switch.add(i)
            best = (_ik_err(pe, re), 0.0, q)
            for s in [q_prev + rng.normal(0.0, 0.3, 6) for _ in range(4)] + seeds:
                qs, ps, rs = _ik.solve(pos, quat, s, base_pos=plan.base_pos, iters=200)
                e, d = _ik_err(ps, rs), float(np.max(np.abs(qs - q_prev)))
                if (e < 1.0 and (best[0] >= 1.0 or d < best[1])) or (best[0] >= 1.0 and e < best[0]):
                    best = (e, d, qs)
            q = best[2]
        elif (float(np.max(np.abs(q[3:5] - q_prev[3:5]))) > math.radians(70.0)
              or float(np.max(np.abs(q - q_prev))) > math.radians(100.0)):
            switch.add(i)  # reached, but by turning the wrist over
        out.append(q)
        q_prev = q
    # then backwards: where the arm changes its wrist's branch between two waypoints, move the change earlier if the
    # earlier waypoint can be reached in the later branch (so it happens in free space, before an approach)
    for i in range(len(out) - 1, 0, -1):
        if i not in switch or i in forced:
            continue
        w = plan.waypoints[i - 1]
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, out[i], base_pos=plan.base_pos, iters=300)
        if (_ik_err(pe, re) < 1.5 and float(np.max(np.abs(q - out[i])))
                < float(np.max(np.abs(out[i - 1] - out[i]))) - math.radians(10.0)):
            out[i - 1] = q
            switch.discard(i)
            switch.add(i - 1)
    return out, switch | forced


def _seg(plan, t):
    acc = 0.0
    for i, w in enumerate(plan.waypoints):
        if t < acc + w.duration:
            return i, _ik.smoothstep((t - acc) / w.duration)
        acc += w.duration
    return len(plan.waypoints) - 1, 1.0


def _action7(self, t):
    """[j1..j6, grip] at time t: IK warm-started from the previous step, or, where that misses, from the joint-space
    blend of the segment's end solutions. It also sets the plan clock the trims run on."""
    _demo._plan_clock = float(t)
    if getattr(self, "_qends", None) is None:
        self._qends, self._switch = _endpoints(self)
    pos, rot, finger, label = self.target(t)
    quat = _ik.matrix_to_quat_xyzw(rot)
    i, s = _seg(self, t)
    q0 = self._qends[i - 1] if i > 0 else self._q_start
    seed = q0 + s * (self._qends[i] - q0)
    if i in self._switch:
        # the wrist changes branch on this (free-space) segment: follow the joint-space blend
        self.q = seed
        grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
        return np.r_[seed, grip], label
    q, pe, re = _ik.solve(pos, quat, self.q, base_pos=self.base_pos, iters=80)
    if _ik_err(pe, re) > 2.0:
        q2, p2, r2 = _ik.solve(pos, quat, seed, base_pos=self.base_pos, iters=80)
        if _ik_err(p2, r2) < _ik_err(pe, re) - 1.0:
            q, pe, re = q2, p2, r2
    self.q = q
    self.worst_ik_mm = max(self.worst_ik_mm, pe * 1000)
    self.worst_ik_deg = max(self.worst_ik_deg, float(np.degrees(re)))
    grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
    return np.r_[q, grip], label


_prev_init = _demo.ArmPlan.__init__
if not getattr(_prev_init, "_fa", False):
    def _init(self, *a, _prev=_prev_init, **k):
        _prev(self, *a, **k)
        self._q_start, self._qends = self.q.copy(), None

    _init._fa = True
    _demo.ArmPlan.__init__ = _init
    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._rot = True
    _demo.grip_rotation = _grip

TIP_ABOVE_TCP = 0.013

class _Plan:
    """Two arms' waypoints, built in phases that start together; the clock is the phases' running time."""

    def __init__(self):
        self.L, self.R = [], []
        self.t = 0.0

    def hold(self, plan, dur, label="hold"):
        w = plan[-1]
        plan.append(Waypoint(dur, w.pos, w.finger, label, w.yaw))

    def phase(self, right=(), left=()):
        """Each list: (duration, pos, finger, label, yaw) tuples. The shorter one is padded with a hold."""
        tr, tl = sum(w[0] for w in right), sum(w[0] for w in left)
        dur = max(tr, tl)
        for plan, ws, tot in ((self.R, right, tr), (self.L, left, tl)):
            for w in ws:
                plan.append(Waypoint(*w))
            if dur - tot > 1e-9:
                self.hold(plan, dur - tot)
        t0 = self.t
        self.t += dur
        return t0

    def window(self, t0, t1, key, fn, watch):
        _ct["wins"].append((t0, t1, key, fn, watch))


# trims: each a 3-vector added to the TCPs of _Dyn waypoints with its key; integrated in windows from seen positions
_ct = {"trim": {}, "wins": [], "last": {}}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}


def _seen(env, name):
    return np.asarray(labkit._part_state(env, labkit._SPECS[name]).pos, float)


@on_step
def _ctrack(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, fn, watch in _ct["wins"]:
        if not t0 <= t < t1:
            continue
        p = _seen(env, watch)
        last = _ct["last"].get(key)
        _ct["last"][key] = p
        if last is None or np.max(np.abs(p - last)) > 0.0005:
            continue  # only while what is watched holds still
        e = np.asarray(fn(env), float)
        tr = _ct["trim"].get(key, np.zeros(3))
        _ct["trim"][key] = np.clip(tr + np.clip(0.2 * e, -0.0003, 0.0003), -0.012, 0.012)


class _Dyn:
    """A TCP plus the live trim ``key``."""

    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)




LEFT_PARK = (0.12, 0.36, TOP + 0.20)
RIGHT_PARK = (0.20, -0.34, TOP + 0.20)
TILT = math.radians(40.0)  # hands on the tube, the rod, the cap and the valve's knob: tipped 40 degrees forward


def _tilted(p=TILT):
    """Fingers closing across a part that runs along y (nearly along x), the hand tipped forward by p."""
    return axis_angle_matrix(Y, -p) @ _prev_grip(math.pi / 2)


def _appr(R):
    return R @ _prev_grip(0.0).T @ np.array([0.0, 0.0, -1.0])


ROT_T = _tilted()
APPR_T = _appr(ROT_T)
O_T = Orient(ROT_T)


def _tcp(tip, R=ROT_T, past=0.004):
    """TCP for fingertips meeting at ``tip`` (moved ``past`` on along the fingers: their ends just beyond an axis)."""
    a = _appr(R)
    return np.asarray(tip, float) + a * past - a * TIP_ABOVE_TCP


ROD_OPEN, ROD_HOLD = -0.012, -0.0005  # round the rod's 6 mm square grip, squeezed hard
TUBE_OPEN, TUBE_HOLD = -0.022, -0.0115  # round the 25.4 mm tube
CAP_OPEN, CAP_HOLD = -0.030, -0.0170  # round the cap's 36.4 mm sleeve
KNOB_OPEN, KNOB_HOLD = -0.014, -0.0035  # round the valve's 10 mm knob
BAR_OPEN, BAR_HOLD = -0.016, -0.0030  # round the lid's 8 mm handle bar
HOLD_Y = 0.322  # where the left hand holds the tube: past the furnace's end, before the end fitting
FEED = (-0.450, -0.270)  # each stroke on the rod: from here to here along y (the hand's reach either side)
H_IN = HOT_Y - BOAT_L / 2  # the pusher at the hot zone (the boat's back wall)
LID_OPEN = 0.60  # rad, the lid lifted this far (34 degrees): the inside and the thermocouple's tip are seen
BAR = np.array([FX0 - LID_HANDLE_OUT, LID_HANDLE_Y, TOP + LID_HANDLE_Z])


def _bar(theta):
    """Where the lid's handle bar is, the lid lifted by theta, and the hand's rotation holding it (pointing in at the
    lid's front, the fingers closing up and down round the bar)."""
    hx, hz = HINGE[0], TOP + HINGE[1]
    dx, dz = BAR[0] - hx, BAR[2] - hz
    p = np.array([hx + dx * math.cos(theta) + dz * math.sin(theta), BAR[1], hz - dx * math.sin(theta) + dz * math.cos(theta)])
    R = axis_angle_matrix(Y, theta) @ axis_angle_matrix(Y, -math.pi / 2) @ _prev_grip(math.pi / 2)
    return p, R


def _knob(phi):
    """The valve knob's centre, the lever turned phi from open (pointing at the robots) towards shut."""
    a = math.pi + phi
    return np.array([VALVE_PIVOT[0] + (LEVER_L - 0.004) * math.cos(a), VALVE_PIVOT[1] + (LEVER_L - 0.004) * math.sin(a),
                     TOP + VALVE_PIVOT[2] + 0.013])


def _rod_at(y, z=AXZ):
    return np.array([XA, y, TOP + z])


def demo():
    _ct["trim"], _ct["wins"] = {}, []
    pl = _Plan()
    O = Orient(_prev_grip(0.0))
    up = np.array([0.0, 0.0, 1.0])
    cap_g = np.array([0.0, CAP_D / 2, 0.0])  # where the cap is held: its sleeve's middle

    # -- 1. right takes the cap; left takes the valve's knob ---------------------------------------------------------
    c0 = CAP_START + cap_g
    k0 = _knob(0.0)
    pl.phase(right=[(2.5, _tcp(c0 + 0.04 * up), CAP_OPEN, "hand turned over the cap", O_T),
                    (1.5, _tcp(c0), CAP_OPEN, "down round the cap", O_T),
                    (0.6, _tcp(c0), CAP_HOLD, "close on the cap", O_T)],
             left=[(2.5, _tcp(k0 + 0.03 * up, past=0.0), KNOB_OPEN, "hand turned over the valve's knob", O_T),
                   (1.5, _tcp(k0, past=0.0), KNOB_OPEN, "down round the knob", O_T),
                   (0.6, _tcp(k0, past=0.0), KNOB_HOLD, "close on the knob", O_T)])
    # -- 2. with the gauge in the green, right pulls the cap off; at once left shuts the gas -------------------------
    c1 = c0 - [0.0, 0.035, 0.0]
    pl.phase(right=[(2.0, _tcp(c1), CAP_HOLD, "pull the cap off (the gauge in the green band)", O_T)])
    pl.phase(left=[(0.6, _tcp(_knob(f * math.pi / 2), past=0.0), KNOB_HOLD, "shut the gas", O_T) for f in (1 / 3, 2 / 3, 1.0)]
             + [(0.4, _tcp(_knob(math.pi / 2), past=0.0), KNOB_HOLD, "shut the gas", O_T)],
             right=[(1.0, _tcp(c1 + 0.04 * up), CAP_HOLD, "lift the cap", O_T)])
    # -- 3. right sets the cap on its stand; left lifts the lid ----------------------------------------------------------
    s0 = CAP_REST + cap_g
    p0, R0 = _bar(0.0)
    lid = [(2.5, _tcp(p0 - [0.012, 0.0, 0.0], R0, past=0.0), BAR_OPEN, "hand turned to the lid's handle", Orient(R0)),
           (1.0, _tcp(p0, R0, past=0.002), BAR_OPEN, "round the handle", Orient(R0)),
           (0.6, _tcp(p0, R0, past=0.002), BAR_HOLD, "close on the handle", Orient(R0))]
    for f in (0.25, 0.5, 0.75, 1.0):
        p, R = _bar(f * LID_OPEN)
        lid.append((1.2, _tcp(p, R, past=0.002), BAR_HOLD, "lift the lid", Orient(R)))
    p1, R1 = _bar(LID_OPEN)
    lid += [(0.6, _tcp(p1, R1, past=0.002), BAR_OPEN, "let go of the lid", Orient(R1)),
            (1.0, _tcp(p1 - _appr(R1) * 0.03, R1, past=0.002), BAR_OPEN, "back off the handle", Orient(R1))]
    pl.phase(left=[(0.6, _tcp(_knob(math.pi / 2), past=0.0), KNOB_OPEN, "let go of the knob", O_T),
                   (0.8, _tcp(_knob(math.pi / 2) + 0.03 * up, past=0.0), KNOB_OPEN, "up off the knob", O_T), *lid],
             right=[(2.0, _tcp(s0 + 0.05 * up), CAP_HOLD, "the cap over its stand", O_T),
                    (1.5, _tcp(s0 + 0.002 * up), CAP_HOLD, "the cap down on its stand", O_T),
                    (0.6, _tcp(s0 + 0.002 * up), CAP_OPEN, "let go of the cap", O_T),
                    (1.0, _tcp(s0 + 0.05 * up), CAP_OPEN, "up off the cap", O_T)])
    # -- 4. left holds the tube; right takes the rod ---------------------------------------------------------------------
    h = _rod_at(HOLD_Y)
    r_g0 = _rod_at(FEED[0], AXZ - 0.0015)
    pl.phase(left=[(2.5, _tcp(h + 0.04 * up), TUBE_OPEN, "hand turned over the tube's far end", O_T),
                   (1.5, _tcp(h), TUBE_OPEN, "down round the tube", O_T),
                   (0.6, _tcp(h), TUBE_HOLD, "hold the tube (it rolls and slides)", O_T)],
             right=[(2.5, _tcp(r_g0 + 0.04 * up), ROD_OPEN, "hand turned over the rod", O_T),
                    (1.5, _tcp(r_g0), ROD_OPEN, "down round the rod", O_T),
                    (0.6, _tcp(r_g0), ROD_HOLD, "close on the rod", O_T)])
    # -- 5. right pushes the boat in, two strokes, until it sits under the thermocouple (seen) ---------------------------
    def stroke(y0, y1, key, label, dur):
        return [(dur, _Dyn(_tcp(_rod_at(y1)), key), ROD_HOLD, label, O_T)]

    def regrip(y_from, y_to, key):
        return [(0.6, _Dyn(_tcp(_rod_at(y_from)), key), ROD_OPEN, "let go of the rod", O_T),
                (0.8, _Dyn(_tcp(_rod_at(y_from) + 0.03 * up), key), ROD_OPEN, "up off the rod", O_T),
                (1.5, _Dyn(_tcp(_rod_at(y_to) + 0.03 * up), key), ROD_OPEN, "along the rod", O_T),
                (1.0, _Dyn(_tcp(_rod_at(y_to)), key), ROD_OPEN, "down round the rod", O_T),
                (0.6, _Dyn(_tcp(_rod_at(y_to)), key), ROD_HOLD, "close on the rod", O_T)]

    first = FEED[1] - FEED[0]  # the first stroke
    second = H_IN - (ROD_START[1] + first)
    pl.phase(right=[(1.0, _tcp(_rod_at(FEED[0])), ROD_HOLD, "lift the rod off its rest", O_T)]
             + stroke(FEED[0], FEED[1], "r1", "push the boat in", 6.0)
             + regrip(FEED[1], FEED[1] - second, "r2")
             + stroke(FEED[1] - second, FEED[1], "r2", "push on until the boat is under the thermocouple", 5.0))
    tw = pl.phase(right=[(2.0, _Dyn(_tcp(_rod_at(FEED[1])), "r2"), ROD_HOLD,
                          "hold still: the boat seen under the thermocouple", O_T)])
    pl.window(tw + 0.3, tw + 2.0, "r2",
              lambda env: np.array([0.0, HOT_Y - _seen(env, "boat")[1], 0.0]), "boat")
    # -- 6. right pulls the rod back out, two strokes, and lays it on its rests ------------------------------------------
    pl.phase(right=[(0.5, _Dyn(_tcp(_rod_at(FEED[1])), "r2"), ROD_HOLD, "hold the rod", O_T)]
             + stroke(FEED[1], FEED[1] - second, "r3", "pull the rod back", 5.0)
             + regrip(FEED[1] - second, FEED[1], "r4")
             + stroke(FEED[1], FEED[0], "r4", "pull the rod out", 6.0)
             + [(1.0, _Dyn(_tcp(_rod_at(FEED[0], AXZ - 0.0015)), "r4"), ROD_HOLD, "down onto its rests", O_T),
                (0.6, _Dyn(_tcp(_rod_at(FEED[0], AXZ - 0.0015)), "r4"), ROD_OPEN, "let go of the rod", O_T),
                (1.0, _Dyn(_tcp(_rod_at(FEED[0], AXZ - 0.0015) + 0.04 * up), "r4"), ROD_OPEN, "up off the rod", O_T),
                (2.0, RIGHT_PARK, ROD_OPEN, "up and clear", O_T)])
    # -- 7. left lets go of the tube and shuts the lid -------------------------------------------------------------------
    shut = [(2.5, _tcp(p1 - _appr(R1) * 0.03, R1, past=0.002), BAR_OPEN, "hand turned to the lid's handle", Orient(R1)),
            (1.0, _tcp(p1, R1, past=0.002), BAR_OPEN, "round the handle", Orient(R1)),
            (0.6, _tcp(p1, R1, past=0.002), BAR_HOLD, "close on the handle", Orient(R1))]
    for f in (0.75, 0.5, 0.25, 0.0):
        p, R = _bar(f * LID_OPEN)
        shut.append((1.0, _tcp(p, R, past=0.002), BAR_HOLD, "lower the lid", Orient(R)))
    shut += [(0.6, _tcp(p0, R0, past=0.002), BAR_OPEN, "let go of the lid", Orient(R0)),
             (1.0, _tcp(p0 - _appr(R0) * 0.012, R0, past=0.002), BAR_OPEN, "back off the handle", Orient(R0)),
             (2.0, LEFT_PARK, BAR_OPEN, "up and clear", O_T)]
    pl.phase(left=[(0.6, _tcp(h), TUBE_OPEN, "let go of the tube", O_T),
                   (0.8, _tcp(h + 0.04 * up), TUBE_OPEN, "up off the tube", O_T), *shut])
    # -- 8. right puts the cap back on: in line with the tube (seen), pushed fully home -------------------------------
    pre = CAP_START - [0.0, CAP_D + 0.005, 0.0]
    pl.phase(right=[(2.5, _tcp(s0 + 0.05 * up), CAP_OPEN, "hand turned over the cap", O_T),
                    (1.5, _tcp(s0 + 0.002 * up), CAP_OPEN, "down round the cap", O_T),
                    (0.6, _tcp(s0 + 0.002 * up), CAP_HOLD, "close on the cap", O_T),
                    (1.0, _tcp(s0 + 0.05 * up), CAP_HOLD, "lift the cap", O_T),
                    (2.0, _tcp(pre + cap_g + 0.045 * up), CAP_HOLD, "the cap over the tube's end", O_T),
                    (1.5, _Dyn(_tcp(pre + cap_g), "c1"), CAP_HOLD, "the cap in front of the tube's end", O_T)])
    tw = pl.phase(right=[(2.0, _Dyn(_tcp(pre + cap_g), "c1"), CAP_HOLD, "hold still: the cap seen in line with the tube",
                          O_T)])
    pl.window(tw + 0.5, tw + 2.0, "c1",
              lambda env: np.array([XA - _seen(env, "end_cap")[0], 0.0, TOP + AXZ - _seen(env, "end_cap")[2]]),
              "end_cap")
    home = CAP_START + [0.0, 0.0015, 0.0] + cap_g
    pl.phase(right=[(2.5, _Dyn(_tcp(home), "c1"), CAP_HOLD, "push the cap fully home", O_T),
                    (0.8, _Dyn(_tcp(home), "c1"), CAP_HOLD, "hold it home", O_T),
                    (0.6, _Dyn(_tcp(home), "c1"), CAP_OPEN, "let go of the cap", O_T),
                    (1.0, _Dyn(_tcp(home) + 0.04 * up, "c1"), CAP_OPEN, "up off the cap", O_T),
                    (2.0, RIGHT_PARK, CAP_OPEN, "up and clear", O_T)])
    pl.phase(right=[(2.0, RIGHT_PARK, FINGER_CLOSED, "rest", O_T)], left=[(2.0, LEFT_PARK, FINGER_CLOSED, "rest", O_T)])
    return {"left": pl.L, "right": pl.R}
