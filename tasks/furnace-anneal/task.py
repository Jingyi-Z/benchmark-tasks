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

import os
_SKIP = os.environ.get("FA_SKIP", "")  # DEBUG-SKIP: remove before submitting
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


def _seg(p0, p1, w, colour):
    """A drawn-only square rod of width w from p0 to p1 (a hose's or a line's length)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d = d / max(np.linalg.norm(d), 1e-9)
    a = np.cross(d, Z if abs(d[2]) < 0.9 else X)
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    return _hex([p + (i - 0.5) * w * a + (j - 0.5) * w * b for p in (p0, p1) for i in (0, 1) for j in (0, 1)],
                 colour, collide="none")


def _hose(pts, w, colour):
    return [_seg(a, b, w, colour) for a, b in zip(pts[:-1], pts[1:])]


# -- the furnace (fixed), on a small desk: a compact split-tube furnace (after a 1 inch, 12 inch long laboratory model),
# its front towards the robots: a base with the controller, the heated lower half, and a lid hinged along the back that
# lifts (to 90 degrees, fully open) by a handle at the middle of its front. The tube runs across it along y. -------------
DESK = 0.050  # the desk's top: the furnace stands on it, so the controller and the gauge face the camera more squarely
XA, AXZ = 0.240, DESK + 0.220  # the tube's axis: x, height above the bench
TUBE_R, TUBE_IN = 0.0127, 0.0110  # 1 inch quartz tube: 25.4 mm outside, 22 mm inside
TUBE_Y0, TUBE_Y1 = -0.225, 0.385  # the tube's ends: the loading end (towards the right arm) and the far end
FX0, FX1, FY0, FY1 = 0.100, 0.380, -0.180, 0.230  # the furnace body: front, back, ends (410 mm wide, 280 mm deep)
BASE_Z = DESK + 0.110  # the base (controller) under the heated chamber
LID_Z1 = DESK + 0.360  # the lid's top
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
DESK_C = (0.36, 0.37, 0.40)
_mid_y, _len_y = (FY0 + FY1) / 2, FY1 - FY0
# the controller: a sloped console on the base's front, its display tilted up towards the operator (and the camera)
CON_N = np.array([-0.6, 0.0, 0.8])  # the console face's normal
CON_U = np.array([0.8, 0.0, 0.6])  # up the face
CON_R = -Y  # across it, left to right as the robots see it
CON_C = np.array([FX0 - 0.035, _mid_y, DESK + 0.060])  # the face's centre
CON_W, CON_H = 0.190, 0.090
DESK_X0 = CON_C[0] - CON_H / 2 * CON_U[0] - 0.012  # the desk's front edge, under the console
FURNACE_BOXES = [
    # the desk
    (((DESK_X0 + FX1 + 0.015) / 2, _mid_y, DESK / 2), (FX1 + 0.015 - DESK_X0, _len_y + 0.030, DESK), DESK_C),
    # base with the controller panel and its display
    (((FX0 + FX1) / 2, _mid_y, (DESK + BASE_Z) / 2), (FX1 - FX0, _len_y, BASE_Z - DESK), WHITE_F),
    ((FX0 - 0.0005, _mid_y, (DESK + BASE_Z) / 2), (0.001, _len_y - 0.02, BASE_Z - DESK - 0.016), PANEL_C),
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
SEG.update({"C": "adef", "°": "abgf"})
def _face_box(c, du, dr, su, sr, t, colour):
    """A thin drawn box on the console's face: centre c + du up + dr across, size su x sr, t thick."""
    o = c + du * CON_U + dr * CON_R
    return _hex([o + (i - 0.5) * sr * CON_R + (j - 0.5) * su * CON_U + k * t * CON_N for i in (0, 1) for j in (0, 1)
                 for k in (0, 1)], colour, collide="none")


def _console():
    """The console (a wedge, collides) with its display: "28°C" in red seven-segment digits (the furnace is cold)."""
    lo, hi = CON_C - CON_H / 2 * CON_U, CON_C + CON_H / 2 * CON_U
    w2 = CON_W / 2
    wedge = _hex([(x, _mid_y + sy * w2, z) for x, z in ((lo[0], lo[2]), (hi[0], hi[2]), (FX0 + 0.005, lo[2]),
                                                       (FX0 + 0.005, hi[2])) for sy in (-1, 1)][::1], PANEL_C)
    base = _box_mesh(((lo[0] + FX0) / 2 + 0.0025, _mid_y, (DESK + lo[2]) / 2), (FX0 + 0.005 - lo[0], CON_W, lo[2] - DESK),
                     PANEL_C, collide="hull")
    out = [wedge, base, _face_box(CON_C, 0.004, 0.0, 0.068, 0.170, 0.0008, SCREEN_C)]
    h = 0.050
    w, tk = 0.55 * h, 0.12 * h
    segs = {"a": (h / 2, 0, tk, w), "g": (0, 0, tk, w), "d": (-h / 2, 0, tk, w), "f": (h / 4, -w / 2, h / 2, tk),
            "b": (h / 4, w / 2, h / 2, tk), "e": (-h / 4, -w / 2, h / 2, tk), "c": (-h / 4, w / 2, h / 2, tk)}
    text = "28°C"
    step = w + 0.30 * h
    for n, ch in enumerate(text):
        x0 = (n - (len(text) - 1) / 2) * step
        sc = 0.5 if ch == "°" else 1.0
        for sg in SEG.get(ch, ""):
            du, dr, su, sr = segs[sg]
            if ch == "°":
                du, dr, su, sr = h / 4 + du * sc, dr * sc, su * sc if su > tk else tk, sr * sc if sr > tk else tk
            out.append(_face_box(CON_C, 0.004 + du, x0 + dr, su, sr, 0.0016, LED_C))
    return out


_coils = [((XA + s * (SLOT - 0.003), y, AXZ - 0.012), (0.003, 0.004, 0.012)) for s in (-1, 1)
          for y in np.arange(FY0 + 0.03, FY1 - 0.03, 0.012)]
FURNACE_MESHES = [*SADDLES, _boxes_mesh(_coils, COIL_C),
                  *_console()]

# the lid: front and back blocks either side of the tube's gap, the top over it; a handle bar at the middle of its front
LID_HANDLE_Y, LID_HANDLE_Z, LID_HANDLE_OUT = _mid_y, AXZ + 0.060, 0.030  # the handle's bar: 30 mm out from the front
_lid_boxes = [
    (((FX0 + XA - SLOT) / 2, _mid_y, (AXZ + LID_Z1) / 2), (XA - SLOT - FX0, _len_y, LID_Z1 - AXZ), WHITE_F),
    (((XA + SLOT + FX1) / 2, _mid_y, (AXZ + LID_Z1) / 2), (FX1 - XA - SLOT, _len_y, LID_Z1 - AXZ), WHITE_F),
    ((XA, _mid_y, (AXZ + 0.030 + LID_Z1) / 2), (2 * SLOT, _len_y, LID_Z1 - AXZ - 0.030), WHITE_F),
    *[((FX0 - LID_HANDLE_OUT / 2, y, LID_HANDLE_Z), (LID_HANDLE_OUT, 0.008, 0.010), DARK)
      for y in (LID_HANDLE_Y - 0.040, LID_HANDLE_Y + 0.040)],
    ((FX0 - LID_HANDLE_OUT, LID_HANDLE_Y, LID_HANDLE_Z), (0.008, 0.090, 0.008), DARK),
]
LID_UP = math.radians(92.0)  # the lid lifts to 92 degrees (its stop), fully clear of the tube
LID_K, LID_SLIP = 15.0, 0.04  # its stiff hinge: it holds where it is left (see _hinge_hold)
LID = Joint("lid", kind="hinge", axis="Y", anchor=(HINGE[0], _mid_y, HINGE[1]), lower=0.0, upper=LID_UP,
            stiffness=LID_K, target=0.0, damping=0.5, initial=0.0, mass=0.4, boxes=_lid_boxes)
FURNACE = part("tube_furnace", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5, boxes=FURNACE_BOXES,
               cylinders=FURNACE_CYLS, meshes=FURNACE_MESHES, joints=[LID])

# -- the tube (loose): fused quartz, 25.4 mm outside, 22 mm inside, 610 mm long, with a stainless vacuum flange on each
# end. The loading end's flange is a quick-clamp one: a collar on the tube, a blank plate (loose) with an O-ring held to
# it by a hinged clamp, opened by flipping its lever up. The argon line joins this collar's side port. The far end's
# flange is closed; on it, the compound gauge (green band: the plate may be opened; red: over-pressured) and the bleed
# line's port. Frame: on the bench. -------------------------------------------------------------------------------------
COLLAR_R, FLANGE_R, COLLAR_L = 0.019, 0.025, 0.035
NC_Y1 = TUBE_Y0 + COLLAR_L  # the loading collar: TUBE_Y0 (its face, the tube's end) to NC_Y1
FIT_Y0, FIT_Y1 = TUBE_Y1 - COLLAR_L, TUBE_Y1 + 0.006  # the far flange (closed)
GAUGE_C = np.array([XA, FIT_Y1 + 0.045, AXZ + 0.085])  # the gauge's centre: on a tee out of the far blank, in view
GAUGE_R = 0.042
GAUGE_N = Z  # a back-mounted gauge on an upright stem: its face looks up (at the top camera)
P_MIN, P_MAX = -10.0, 15.0  # kPa over the dial
P_LO, P_HI = 2.0, 8.0  # the green band: the plate may be opened only here; under it the pressure difference holds it,
P_BURST = 12.0  # over P_HI it would blow off; over P_BURST the tube is over-pressured (it can burst)
DIAL_SWEEP = math.radians(270.0)
_gu, _gr = X, -Y  # the dial's "up" and "right" as the top camera (behind the robots, looking along +x) sees it


def dial_angle(p):
    """The needle's angle on the dial (rad, 0 = up as the camera sees it, clockwise positive), for pressure p (kPa):
    P_MIN at -135 deg, P_MAX at +135 deg. Its joint (about +z) turns the other way: its value is minus this."""
    s = (min(max(p, P_MIN), P_MAX) - P_MIN) / (P_MAX - P_MIN)
    return -DIAL_SWEEP / 2 + s * DIAL_SWEEP


def _on_dial(r, ang, h=0.0):
    """A point on the dial's face: radius r, angle ang (0 up, clockwise as seen), h out of the face."""
    return GAUGE_C + r * (math.cos(ang) * _gu + math.sin(ang) * _gr) + h * GAUGE_N


def _dial_arc(p0, p1, colour, r0=0.028, r1=0.038):
    out = []
    n = max(2, int(abs(dial_angle(p1) - dial_angle(p0)) / math.radians(6)) + 1)
    for i in range(n):
        a, b = dial_angle(p0 + (p1 - p0) * i / n), dial_angle(p0 + (p1 - p0) * (i + 1) / n)
        out.append(_hex([_on_dial(r, ang, h) for h in (0.0011, 0.0017) for r in (r0, r1) for ang in (a, b)],
                        colour, collide="none"))
    return out


def _disc_mesh(c, n, r, t, colour, k=32):
    """A drawn disc (centre c, axis n, radius r, thickness t), as k wedges."""
    u = np.cross(n, X if abs(n[0]) < 0.9 else Y)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    out = []
    for i in range(k):
        a, b = 2 * math.pi * i / k, 2 * math.pi * (i + 1) / k
        out.append(_hex([c + h * n + rr * (math.cos(q) * u + math.sin(q) * v) for h in (0.0, t) for rr in (0.0005, r)
                         for q in (a, b)], colour, collide="none"))
    return out


# the tube's wall in 45 mm lengths (PhysX cannot build long, thin convex pieces on the GPU), the two collars with their
# flange lips, the gauge and the ports
_TUBE_CUTS = np.linspace(TUBE_Y0, FIT_Y0 + 0.005, 14)
ARGON_PORT = (XA + COLLAR_R + 0.010, (TUBE_Y0 + NC_Y1) / 2 + 0.004, AXZ)  # the argon line's port, on the collar's back
BLEED_PORT = (XA + 0.012, FIT_Y1 + 0.020, AXZ)  # the bleed line's port, on the tee's back
CLAMP_PIVOT = (XA, NC_Y1 - 0.012, AXZ + COLLAR_R + 0.010)  # the clamp's hinge, on top of the loading collar
# (the flanges first: PhysX keeps a limited number of shapes on one body, so the pieces that must collide come first)
TUBE_MESHES = [*_ring_y(TUBE_Y0, TUBE_Y0 + 0.006, TUBE_R, FLANGE_R, 12, STAINLESS, cx=XA, cz=AXZ),
               *_ring_y(TUBE_Y0 + 0.006, NC_Y1, TUBE_R, COLLAR_R, 12, STAINLESS, cx=XA, cz=AXZ),
               *_ring_y(FIT_Y0, FIT_Y1 - 0.006, TUBE_R, COLLAR_R, 12, STAINLESS, cx=XA, cz=AXZ),
               *_ring_y(FIT_Y1 - 0.006, FIT_Y1, TUBE_R, FLANGE_R, 12, STAINLESS, cx=XA, cz=AXZ),
               *[m for y0, y1 in zip(_TUBE_CUTS[:-1], _TUBE_CUTS[1:])
                 for m in _ring_y(y0, y1, TUBE_IN, TUBE_R, 12, QUARTZ, cx=XA, cz=AXZ)]]
TUBE_DRAWN = [
    *_disc_mesh(np.array([XA, FIT_Y1, AXZ]), Y, FLANGE_R, 0.006, STAINLESS, 16),  # the far flange's blank
    _seg((XA, FIT_Y1, AXZ), (XA, GAUGE_C[1] + 0.006, AXZ), 0.014, STAINLESS),  # the tee's nipple out of the blank
    _seg((XA, GAUGE_C[1], AXZ), GAUGE_C - 0.010 * GAUGE_N, 0.009, STAINLESS),  # the gauge's stem, up off the tee
    _box_mesh((XA + COLLAR_R + 0.005, ARGON_PORT[1], AXZ), (0.012, 0.010, 0.010), STAINLESS),  # the argon port
    _box_mesh((XA + 0.008, BLEED_PORT[1], AXZ), (0.012, 0.010, 0.010), STAINLESS),  # the bleed port
    _box_mesh((XA, CLAMP_PIVOT[1], AXZ + COLLAR_R + 0.005), (0.014, 0.012, 0.012), STAINLESS),  # the clamp's hinge block
    *_disc_mesh(GAUGE_C - 0.012 * GAUGE_N, GAUGE_N, GAUGE_R + 0.003, 0.012, STAINLESS),  # the gauge's case
    *_disc_mesh(GAUGE_C, GAUGE_N, GAUGE_R, 0.001, WHITE),  # its face
    *_dial_arc(P_MIN, P_LO, (0.55, 0.56, 0.58)), *_dial_arc(P_LO, P_HI, GREEN), *_dial_arc(P_HI, P_MAX, RED),
]
TUBE_MESHES += TUBE_DRAWN
_na = GAUGE_C + 0.0024 * GAUGE_N  # the needle's pivot, just in front of the face; it turns about the face's normal
NEEDLE = Joint("needle", kind="hinge", axis="Z", anchor=tuple(_na),
               lower=-DIAL_SWEEP / 2 - 0.05, upper=DIAL_SWEEP / 2 + 0.05, stiffness=0.05, damping=0.003,
               target=0.0, initial=0.0, mass=0.0002,
               meshes=[_seg(_na - 0.006 * _gu, _na + 0.034 * _gu, 0.0022, DARK)],
               boxes=[(tuple(_na), (0.003, 0.003, 0.003), DARK)])

# the quick clamp: an arm hinged on the collar's top that reaches over the plate and hooks its outside face; flipped up
# by its lever (opening it), pushed back down over the plate (closing it). Closed is 0; open is turned up past 80 deg.
CLAMP_OPEN, CLAMP_UP = math.radians(80.0), math.radians(110.0)
CLAMP_K, CLAMP_SLIP = 0.5, 0.08  # its hinge is stiff too: it stays open or shut where it is left
PLATE_T, PLATE_R = 0.006, FLANGE_R  # the blank plate: 6 mm stainless, as wide as the flange
_cz = CLAMP_PIVOT[2]
_hook_y = TUBE_Y0 - PLATE_T - 0.0012  # the hook's inside face, 1.2 mm off the plate's outside
CLAMP = Joint("clamp", kind="hinge", axis="X", anchor=CLAMP_PIVOT, lower=-CLAMP_UP, upper=0.0,
              stiffness=CLAMP_K, damping=0.02, friction=0.02, initial=0.0, mass=0.030,
              boxes=[((XA, (CLAMP_PIVOT[1] + _hook_y - 0.004) / 2, _cz + 0.002),
                      (0.012, CLAMP_PIVOT[1] - _hook_y + 0.004, 0.004), STAINLESS),  # the arm
                     ((XA, _hook_y - 0.002, _cz - 0.006), (0.012, 0.004, 0.020), STAINLESS),  # the hook
                     ((XA, _hook_y - 0.002, _cz + 0.016), (0.014, 0.004, 0.024), DARK)])  # the lever, up off the hook
# a smooth strip along the tube's floor, one piece (the boat would catch on the seams between the wall's lengths)
FLOOR_Z = AXZ - 0.0075
# under each collar a flat foot, which stands on a support (the tube rests on the two supports, flat on flat, so it cannot
# roll; the furnace's saddles just clear it)
PAD_T, PAD_W = 0.003, 0.050
_pads = [((TUBE_Y0 + 0.006 + NC_Y1) / 2, NC_Y1 - TUBE_Y0 - 0.010), ((FIT_Y0 + FIT_Y1 - 0.006) / 2, FIT_Y1 - FIT_Y0 - 0.010)]
PAD_Z = AXZ - COLLAR_R - PAD_T / 2 + 0.0008  # (its top a little into the collar: one solid with it)
TUBE_BOXES = [((XA, (TUBE_Y0 + FIT_Y0) / 2, FLOOR_Z - 0.001), (0.012, FIT_Y0 - TUBE_Y0, 0.002), QUARTZ),
              *[((XA, y, PAD_Z), (PAD_W, l, PAD_T), STAINLESS) for y, l in _pads]]
# (the clamp and the needle are light: the tube's weight must stay under its axis, or it rolls in its saddles)
TUBE = part("quartz_tube", pos=(0.0, 0.0, TOP), mass=0.40, friction=0.3, boxes=TUBE_BOXES, meshes=TUBE_MESHES,
            joints=[NEEDLE, CLAMP])

# -- the gas lines (fixed), each with a quarter-turn ball valve on a post, so working a valve does not push the loose
# tube: the argon line from a gas bottle, through its regulator and the inlet valve, to the loading collar's port; the
# bleed line from the far end's tee, through the bleed valve, down to the exhaust. A lever along its line is open;
# turned a quarter turn across it, shut. The argon's line runs along x at its valve, the bleed's along y. ---------------
LEVER_L = 0.045
BRASS = (0.78, 0.66, 0.30)
HOSE_C = (0.30, 0.55, 0.85)
ARGON_C = (0.25, 0.48, 0.30)  # the gas bottle (argon: green shoulder)
VALVE_SHUT = math.radians(10.0)  # a valve counts as shut with its lever within 10 degrees of across its line
SHUT_AT = {"argon_valve": -math.pi / 2, "bleed_valve": 0.0}  # each valve's lever angle (about z, 0 along +x) when shut
ARGON_PIVOT = (XA + 0.060, -0.215, AXZ + 0.004)  # the inlet valve, behind the loading end's collar
BLEED_PIVOT = (XA + 0.070, 0.425, AXZ + 0.004)  # the bleed valve, behind the far end's tee (clear of the left hand)
BOTTLE = (0.42, -0.62)  # the argon bottle, off to the side of the loading end


def _valve(name, pivot, lo, hi, start):
    """A ball valve on its post: its lever (along +x at 0) turns about z; along its line it is open, a quarter turn
    across it shut. ``lo``/``hi``: the lever's travel; ``start``: where it starts."""
    j = Joint(name, kind="hinge", axis="Z", anchor=pivot, lower=lo, upper=hi, stiffness=0.0, damping=0.02,
              friction=0.03, initial=start, mass=0.01,
              boxes=[((pivot[0] + LEVER_L / 2, pivot[1], pivot[2] + 0.010), (LEVER_L, 0.010, 0.004), RED)],
              cylinders=[((pivot[0] + LEVER_L - 0.004, pivot[1], pivot[2] + 0.027), 0.005, 0.030, RED, "Z")])  # its grip
    post = [((pivot[0], pivot[1], (pivot[2] - 0.012) / 2), (0.012, 0.012, pivot[2] - 0.012), DARK),
            ((pivot[0], pivot[1], pivot[2]), (0.022, 0.020, 0.016), BRASS)]
    return j, post


ARGON_V, _ap = _valve("argon_valve", ARGON_PIVOT, -math.pi / 2, 0.0,
                      -math.pi / 2 if _SKIP else 0.0)  # open; shut turned towards the right arm
BLEED_V, _bp = _valve("bleed_valve", BLEED_PIVOT, -0.05, math.pi / 2, 0.0)  # starts shut (along x, across its line)
LINE_MESHES = [
    # the argon bottle with its regulator, the line along the bench to the inlet valve, and on to the collar's port
    *_disc_mesh(np.array([BOTTLE[0], BOTTLE[1], 0.0]), Z, 0.040, 0.250, (0.55, 0.56, 0.58)),
    *_disc_mesh(np.array([BOTTLE[0], BOTTLE[1], 0.250]), Z, 0.040, 0.050, ARGON_C),
    _box_mesh((BOTTLE[0], BOTTLE[1], 0.315), (0.030, 0.030, 0.030), BRASS),
    *_hose([(BOTTLE[0] - 0.015, BOTTLE[1], 0.315), (BOTTLE[0] - 0.05, BOTTLE[1], 0.29), (BOTTLE[0] - 0.06, BOTTLE[1], 0.004),
            (ARGON_PIVOT[0] + 0.04, ARGON_PIVOT[1] - 0.03, 0.004), (ARGON_PIVOT[0] + 0.03, ARGON_PIVOT[1], 0.03),
            (ARGON_PIVOT[0] + 0.012, ARGON_PIVOT[1], ARGON_PIVOT[2] - 0.004)], 0.006, HOSE_C),
    *_hose([(ARGON_PIVOT[0] - 0.012, ARGON_PIVOT[1], ARGON_PIVOT[2] - 0.004),
            (ARGON_PORT[0] + 0.010, ARGON_PIVOT[1] + 0.004, AXZ - 0.002), (ARGON_PORT[0] + 0.004, ARGON_PORT[1], AXZ)],
           0.006, HOSE_C),
    # the bleed line: from the tee's port to its valve (in along y), then out and down off the bench to the exhaust
    *_hose([(BLEED_PORT[0] + 0.004, BLEED_PORT[1], AXZ), (BLEED_PIVOT[0] - 0.004, BLEED_PIVOT[1] - 0.020, AXZ - 0.002),
            (BLEED_PIVOT[0], BLEED_PIVOT[1] - 0.012, BLEED_PIVOT[2] - 0.004)], 0.006, HOSE_C),
    *_hose([(BLEED_PIVOT[0], BLEED_PIVOT[1] + 0.012, BLEED_PIVOT[2] - 0.004), (BLEED_PIVOT[0], BLEED_PIVOT[1] + 0.05, 0.12),
            (BLEED_PIVOT[0] + 0.05, BLEED_PIVOT[1] + 0.10, 0.0)], 0.006, (0.25, 0.26, 0.28)),
]
GAS = part("gas_lines", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5, boxes=[*_ap, *_bp], meshes=LINE_MESHES,
           joints=[ARGON_V, BLEED_V])


# -- the blank plate (loose): 6 mm stainless with an O-ring and a centring spigot on its inside (in the tube's bore), a
# black knob on its outside to hold it by.
# Frame: origin on its axis at its inside face, the knob along -y. ------------------------------------------------------
KNOB_R, KNOB_L = 0.008, 0.016
SPIGOT_R, SPIGOT_L = 0.0085, 0.010  # its centring ring, in the bore: it rests on the floor strip, and keeps the plate
# from tipping out when the clamp is open (it has 1 mm to the bore's top)
_SPZ = FLOOR_Z + 0.0003 + SPIGOT_R - (AXZ + 0.0004)  # the ring's centre, from the plate's
PLATE_START = np.array([XA, TUBE_Y0 - 0.0008, TOP + AXZ + 0.0004])
PLATE = part("end_plate", pos=tuple(PLATE_START), mass=0.10, friction=0.4,
             cylinders=[((0.0, -PLATE_T / 2, 0.0), PLATE_R, PLATE_T, STAINLESS, "Y"),
                        ((0.0, SPIGOT_L / 2, _SPZ), SPIGOT_R, SPIGOT_L, DARK, "Y"),
                        ((0.0, -PLATE_T - KNOB_L / 2, 0.0), KNOB_R, KNOB_L, DARK, "Y")])
# its stand (fixed): a post with a V cradle the plate stands in, on edge, knob towards -y
PLATE_REST = np.array([XA + 0.090, -0.440, TOP + AXZ - 0.020])  # the plate's centre on the stand
_vz = PLATE_REST[2] - TOP - PLATE_R * 0.90  # the V's bottom
PLATE_STAND = part("plate_stand", pos=(0.0, 0.0, TOP), fixed=True, friction=0.6,
                   boxes=[((PLATE_REST[0], PLATE_REST[1] - PLATE_T / 2, (_vz - 0.004) / 2),
                           (0.016, 0.016, _vz - 0.004), DARK),
                          *[((PLATE_REST[0], PLATE_REST[1] - PLATE_T / 2 + s * (PLATE_T / 2 + 0.0035), _vz + 0.004),
                             (0.040, 0.003, 0.016), DARK) for s in (-1, 1)]],
                   meshes=[_hex([(PLATE_REST[0] + sx * xo, PLATE_REST[1] - PLATE_T / 2 + dy, zz)
                                 for xo, zt in ((0.0, _vz), (0.020, _vz + 0.009)) for dy in (-0.005, 0.005)
                                 for zz in (_vz - 0.004, zt)], DARK) for sx in (-1, 1)])

# -- the boat (loose): an alumina boat 80 x 15 x 8 mm, open on top, its back end a 12 mm wall with a 14 mm tongue out
# behind it at its floor, to hold it by and for the rod's pusher. In the tube it sits on the floor strip, 7.5 mm under
# the axis. It starts on a tray in front of the loading end, in line with the tube. Frame: origin at the centre of its underside. ---------------------
BOAT_L, BOAT_W, BOAT_H, BOAT_T, BACK_H = 0.080, 0.015, 0.008, 0.0015, 0.012
TONGUE_L, TONGUE_W, TONGUE_T = 0.016, 0.012, 0.004
ALUMINA = (0.97, 0.96, 0.93)
BOAT_Z = FLOOR_Z + 0.0003  # on the floor strip
TRAY = np.array([XA, -0.296, AXZ - 0.032])  # the tray's top centre: in line with the tube, in front of its mouth
BOAT_START = TRAY + [0.0, 0.006, TOP + 0.0003]
BOAT = part("boat", pos=tuple(BOAT_START), mass=0.020, friction=0.5,
            boxes=[((0.0, 0.0, BOAT_T / 2), (BOAT_W, BOAT_L, BOAT_T), ALUMINA),
                   *[((e * (BOAT_W - BOAT_T) / 2, 0.0, BOAT_H / 2), (BOAT_T, BOAT_L, BOAT_H), ALUMINA) for e in (-1, 1)],
                   ((0.0, -(BOAT_L - BOAT_T) / 2, BACK_H / 2), (BOAT_W - 2 * BOAT_T, BOAT_T, BACK_H), ALUMINA),
                   ((0.0, (BOAT_L - BOAT_T) / 2, BOAT_H / 2), (BOAT_W - 2 * BOAT_T, BOAT_T, BOAT_H), ALUMINA),
                   ((0.0, -BOAT_L / 2 - TONGUE_L / 2, TONGUE_T / 2), (TONGUE_W, TONGUE_L, TONGUE_T), ALUMINA)])
# the tray (fixed): a stainless tray on a post, shorter than the boat so its tongue hangs out over the tray's end
TRAY_L = 0.060
BOAT_TRAY = part("boat_tray", pos=(0.0, 0.0, TOP), fixed=True, friction=0.6,
                 boxes=[((TRAY[0], TRAY[1] + 0.012, (TRAY[2] - 0.003) / 2), (0.012, 0.012, TRAY[2] - 0.003), DARK),
                        ((TRAY[0], TRAY[1] + 0.012, TRAY[2] - 0.0015), (0.030, TRAY_L, 0.003), STAINLESS),
                        *[((TRAY[0] + s * 0.0125, TRAY[1] + 0.012, TRAY[2] + 0.002), (0.003, TRAY_L, 0.004), STAINLESS)
                          for s in (-1, 1)]])

# -- the chips (loose): three 8 x 8 x 0.5 mm silicon chips (with 2D flakes on their oxide, which shows purple) in a row
CHIP, CHIP_T = 0.008, 0.0005
CHIP_DY = (-0.024, 0.0, 0.024)
CHIP_START = [BOAT_START + [0.0, dy, BOAT_T + CHIP_T / 2] for dy in CHIP_DY]
CHIPS = [part(f"chip{i}", pos=tuple(CHIP_START[i]), mass=0.00008, friction=0.8,
              boxes=[((0.0, 0.0, 0.0), (CHIP, CHIP, CHIP_T), CHIP_C)]) for i in range(3)]

# -- the hook rod (loose): stainless, 390 mm, 6 mm round its whole length, a 10 mm pusher disc at its front. Frame: origin
# on its axis at the pusher's front face, the rod along -y. -----------------------------------------------------------------
ROD_L, GRIP_W, PUSHER_R = 0.390, 0.006, 0.005
ROD_START = np.array([XA, TUBE_Y0 - 0.150, TOP + AXZ - 0.0015])  # on its rests in line with the tube, room in front for the boat
ROD = part("hook_rod", pos=tuple(ROD_START), mass=0.090, friction=0.7,
           cylinders=[((0.0, -0.0015, 0.0), PUSHER_R, 0.003, STEEL, "Y"),
                      ((0.0, -(0.003 + ROD_L) / 2, 0.0), GRIP_W / 2, ROD_L - 0.003, STEEL, "Y")])
# its rests (fixed): two posts with shallow V tops (1.5 mm deep over 10 mm a side) that centre the rod
REST_Y = (-0.460, -0.700)  # both behind the hand's strokes on the rod
REST_Z = AXZ - GRIP_W / 2 - 0.0015
V_RISE = 0.0015
REST = part("rod_rests", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5,
            boxes=[((XA, y, REST_Z / 2), (0.020, 0.014, REST_Z), DARK) for y in REST_Y],
            meshes=[_hex([(XA + sx * xo, y + dy, REST_Z + zo) for xo, zt in ((0.0, 0.0002), (0.010, V_RISE))
                          for dy in (-0.007, 0.007) for zo in (0.0, zt)], DARK)
                    for y in REST_Y for sx in (-1, 1)])

_stz = PAD_Z - PAD_T / 2  # the supports' tops
SUPPORTS = part("tube_supports", pos=(0.0, 0.0, TOP), fixed=True, friction=0.8,
                boxes=[b for y, l in _pads for b in (((XA, y, _stz - 0.003), (PAD_W + 0.008, l + 0.004, 0.006), DARK),
                                                     ((XA, y, (_stz - 0.006) / 2), (0.014, 0.014, _stz - 0.006), DARK))])

OBJECTS = [FURNACE, TUBE, GAS, PLATE, PLATE_STAND, BOAT, BOAT_TRAY, *CHIPS, ROD, REST, SUPPORTS]
PLANCHECK_BOXES = [
    ("furnace", (((FX0 + FX1) / 2), _mid_y, TOP + LID_Z1 / 2), (FX1 - FX0, _len_y, LID_Z1)),
]


# -- the gas ----------------------------------------------------------------------------------------------------------
# The trial starts mid-backfill: the argon inlet open, the sealed tube 7 kPa over atmospheric and rising 0.4 kPa/s (over
# the green band in a few seconds; over-pressured, P_BURST, in 12 s). Shutting the argon holds the pressure; the bleed
# valve lets it out (in proportion to how far it is opened). The plate may be taken off only with the argon shut and the
# gauge in its green band; once off, the tube is at atmospheric.
P0, P_RATE = (5.0 if _SKIP else 6.0), 0.4  # kPa, kPa/s (the argon fully open; less as its valve is turned towards shut)
BLEED_K = 0.6  # 1/s: the bleed, fully open, lets the pressure fall by this fraction each second
SEAT_ON, SEAT_OFF = 0.002, 0.004  # the plate seals within 2 mm of its seat; it is off (opened) past 4 mm


def _asset(env, name):
    return env.unwrapped.scene[name]


def _t(x):
    return getattr(x, "torch", x)


def _drive(env, part_name, joint, value):
    """Set a joint's drive target (the gauge's needle; the stiff hinges)."""
    import torch

    a = _asset(env, part_name)
    try:
        i = list(a.joint_names).index(joint.joint_name)
        dev = _t(a.data.joint_pos).device
        a.set_joint_position_target(torch.tensor([[value]], dtype=torch.float32, device=dev), joint_ids=[i])
    except Exception as e:  # noqa: BLE001
        if not _gas.get("warned_" + joint.name):
            _gas["warned_" + joint.name] = True
            print(f"[furnace-anneal] {joint.name} not driven: {e!r}")


def _hinge_hold(env, part_name, joint, key, lo, hi, slip):
    """A stiff hinge: its drive holds it where it was left and gives way (the target follows) only when it is pushed
    more than ``slip`` from it."""
    th = labkit._part_state(env, labkit._SPECS[part_name]).joints.get(joint.name, 0.0)
    tgt = _gas.get(key, th)
    if th - tgt > slip:
        tgt = th - slip
    elif tgt - th > slip:
        tgt = th + slip
    tgt = min(max(tgt, lo), hi)
    _gas[key] = tgt
    _drive(env, part_name, joint, tgt)


@on_step
def _holds(env):
    _hinge_hold(env, "tube_furnace", LID, "lid_t", 0.0, LID_UP, LID_SLIP)  # the lid stays where it is left
    _hinge_hold(env, "quartz_tube", CLAMP, "clamp_t", -CLAMP_UP, 0.0, CLAMP_SLIP)  # so does the clamp (open or shut)


_gas = {}


@on_reset
def _reset_gas(env):
    _gas.clear()
    _gas.update(p=P0, opened=None)


def plate_out(pl):
    """How far the plate is off its seat along the tube (m; 0 when home)."""
    return float(TUBE_Y0 - np.asarray(pl.pos, float)[1])


def plate_on(pl):
    p = np.asarray(pl.pos, float)
    return plate_out(pl) < SEAT_ON and math.hypot(p[0] - XA, p[2] - (TOP + AXZ)) < 0.002


def valve_turn(g, name):
    """How far a valve's lever is turned from shut (radians)."""
    return abs(g.joints.get(name, SHUT_AT[name]) - SHUT_AT[name])


def valve_shut(g, name):
    return valve_turn(g, name) <= VALVE_SHUT


def argon_open(g):
    return 0.0 if valve_shut(g, "argon_valve") else min(1.0, valve_turn(g, "argon_valve") / (math.pi / 2))


def bleed_open(g):
    return min(1.0, valve_turn(g, "bleed_valve") / (math.pi / 2))


@on_step
def _gas_step(env):
    S = labkit._SPECS
    pl = labkit._part_state(env, S["end_plate"])
    g = labkit._part_state(env, S["gas_lines"])
    argon = not valve_shut(g, "argon_valve")
    p = _gas.get("p", P0)
    if plate_out(pl) < SEAT_OFF and _gas.get("opened") is None:
        p += P_RATE * argon_open(g) * DT
        p -= BLEED_K * bleed_open(g) * max(p, 0.0) * DT
    else:
        if _gas.get("opened") is None:
            _gas["opened"] = (p, argon)  # the pressure, and whether the argon still ran, when the plate came off
        p = 0.0 if not plate_on(pl) else p  # open to the room; back on after loading, at atmospheric
        if plate_on(pl):
            p += P_RATE * argon_open(g) * DT
    _gas["p"] = p
    _drive(env, "quartz_tube", NEEDLE, -dial_angle(p))


# -- scoring ----------------------------------------------------------------------------------------------------------
CHIP_SLIDE = 0.003  # a chip that moves over 3 mm on the boat has slid (lost or scratched)
CHIP_FLIP = 30.0  # or tilted over 30 degrees
BOAT_TILT = 20.0
BOAT_DROP = 0.030  # the boat dropped: over 30 mm under the tray or the tube's floor
HOT_TOL = 0.008  # the boat centred under the thermocouple within 8 mm
TUBE_MOVE, TUBE_TURN = 0.003, 3.0  # the tube slid over 3 mm or rolled over 3 degrees
RACK_TOL = 0.015  # the rod back on its rests: within 15 mm of where it lay
LID_SHUT = math.radians(2.0)
AXIS_TOL, PLATE_TILT = 0.002, 3.0
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


def flange_sealed(pl, t, g):
    """The plate home and square, the clamp shut over it, both gas valves shut."""
    return (plate_on(pl) and axis_off(pl) <= PLATE_TILT and abs(t.joints.get("clamp", 0.0)) <= math.radians(5.0)
            and valve_shut(g, "argon_valve") and valve_shut(g, "bleed_valve"))


TUBE_MID = np.array([XA, (TUBE_Y0 + TUBE_Y1) / 2, AXZ])  # a point on the tube's axis, at its middle (part frame)


def _tube_ref(t):
    """Where the tube's middle is (on its axis), and how it is turned."""
    return np.asarray(t.pos, float) + t.rotation() @ TUBE_MID, t.rotation()


@on_step
def _watch(env):
    """Latch the violations and follow the end state."""
    S = labkit._SPECS
    st = _state
    st["step"] += 1
    b, r, pl = (labkit._part_state(env, S[n]) for n in ("boat", "hook_rod", "end_plate"))
    t, f, g = (labkit._part_state(env, S[n]) for n in ("quartz_tube", "tube_furnace", "gas_lines"))
    chips = [labkit._part_state(env, S[f"chip{i}"]) for i in range(3)]
    if st["step"] == 10:
        st["tube0"] = _tube_ref(t)  # where the tube lies once settled
    for d, tilt in chip_offsets(b, chips):
        if d > CHIP_SLIDE:
            st["viol"].add("a chip slid on the boat")
        if tilt > CHIP_FLIP:
            st["viol"].add("a chip flipped")
    if b.tilt_deg() > BOAT_TILT or np.asarray(b.pos, float)[2] < TOP + min(TRAY[2], FLOOR_Z) - BOAT_DROP:
        st["viol"].add("the boat tipped or dropped")
    p = _gas.get("p", 0.0)
    if p > P_BURST:
        st["viol"].add("the tube was over-pressured")
    if abs(t.joints.get("clamp", 0.0)) > math.radians(20.0) and _gas.get("opened") is None and p > P_HI:
        st["viol"].add("the clamp was opened over the green band")
    op = _gas.get("opened")
    if op is not None:
        if op[1]:
            st["viol"].add("the plate came off with the argon on")
        if op[0] > P_HI:
            st["viol"].add("the plate came off over the green band")
        if op[0] < P_LO:
            st["viol"].add("the plate was forced off under the green band")
    if "tube0" in st:
        p0, R0 = st["tube0"]
        turn = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ t.rotation()) - 1) / 2))))
        if np.linalg.norm(_tube_ref(t)[0] - p0) > TUBE_MOVE or turn > TUBE_TURN:
            st["viol"].add("the tube slid or rolled")
    bp = np.asarray(b.pos, float)
    st["centred"] = abs(bp[1] - HOT_Y) <= HOT_TOL and abs(bp[2] - (TOP + BOAT_Z)) < 0.004
    st["racked"] = rod_racked(r)
    st["lid"] = f.joints.get("lid", 0.0) <= LID_SHUT
    st["sealed"] = flange_sealed(pl, t, g)
    tips = _tips(env)
    keep = [np.asarray(r.pos, float), np.asarray(pl.pos, float) - [0.0, PLATE_T + KNOB_L / 2, 0.0],
            np.array([FX0 - LID_HANDLE_OUT, LID_HANDLE_Y, TOP + LID_HANDLE_Z]),
            np.array([XA, _hook_y, TOP + _cz + 0.016])]
    st["far"] = all(np.linalg.norm(np.asarray(tp) - k) >= HANDS_FAR for tp in tips for k in keep)
    _hist.append(np.concatenate([bp, np.asarray(r.pos, float), np.asarray(pl.pos, float),
                                 *[np.asarray(x.pos, float) for x in chips], [f.joints.get("lid", 0.0) * 0.1]]))
    del _hist[:-(SETTLE_STEPS + 1)]
    st["settled"] = (len(_hist) > SETTLE_STEPS
                     and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 after any violation: the tube over-pressured (over 12 kPa); the clamp opened or the plate taken off with the
    gauge over its green band (8 kPa), or the plate forced off under it (2 kPa), or with the argon still on; a chip slid
    over 3 mm on the boat or tilted over 30 degrees; the boat tipped over 20 degrees or dropped; the tube slid over 3 mm
    or rolled over 3 degrees. 0 also unless everything has been still (under 1 mm) for the last second, and the boat's
    centre lies on the tube's floor within 8 mm of the thermocouple along the tube. Then 0.2, plus 0.2 each for: the rod
    back on its rests (within 15 mm of where it lay); the lid shut (within 2 degrees); the flange sealed (the plate home
    and square, within 2 mm and 3 degrees, the clamp shut, both gas valves shut). 1.0 if all three, and every fingertip
    is 8 cm from the rod, the plate's knob, the clamp's lever and the lid's handle."""
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

    def gate(self, t0, t1, key, cond):
        _ct["gates"].append((t0, t1, key, cond))


# trims: each a 3-vector added to the TCPs of _Dyn waypoints with its key; integrated in windows from seen positions
_ct = {"trim": {}, "wins": [], "last": {}, "gates": [], "gate": {}}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}
    _ct["gate"] = {}


@on_step
def _gtrack(env):
    """Gates: from t0 on, latch ``key`` once ``cond`` sees what it waits for (at t1 at the latest)."""
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, cond in _ct["gates"]:
        if t0 <= t and not _ct["gate"].get(key) and (t >= t1 or cond(env)):
            _ct["gate"][key] = t


class _Gate:
    """A TCP that is ``a`` until the gate ``key`` latches, then ``b``."""

    def __init__(self, a, b, key):
        self.a, self.b, self.key = np.asarray(a, float), np.asarray(b, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.b if _ct["gate"].get(self.key) else self.a, dtype=dtype or float)


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
RIGHT_PARK = (0.14, -0.30, TOP + 0.24)
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


ROD_OPEN, ROD_HOLD = -0.012, 0.0  # round the 6 mm rod, squeezed hard (a looser grip slips)
TUBE_OPEN, TUBE_HOLD = -0.022, -0.0115  # round the 25.4 mm tube
KNOB_OPEN, KNOB_HOLD = -0.016, -0.0040  # round a valve lever's 10 mm upright grip
PK_OPEN, PK_HOLD = -0.018, -0.0050  # round the plate's 16 mm knob
TAB_OPEN, TAB_HOLD = -0.014, 0.0  # across the clamp's 4 mm lever (squeezed)
TONGUE_OPEN, TONGUE_HOLD = -0.012, 0.0  # on the boat's 4 mm thick tongue (squeezed)
BAR_OPEN, BAR_HOLD = -0.016, -0.0010  # round the lid's 8 mm handle bar (squeezed)
WIDE = -0.030
HOLD_Y = 0.290  # where the left hand holds the tube: past the furnace's end, before the far flange
FEED = (-0.395, -0.275)  # each stroke on the rod: from here to here along y (reach behind, the near rest in front)
H_IN = HOT_Y - BOAT_L / 2 - TONGUE_L  # the pusher at the boat centred under the thermocouple (its tongue's end)
LID_OPEN = math.radians(85.0)  # the lid lifted (all but upright): the tube, the boat and the thermocouple's tip in view
BAR = np.array([FX0 - LID_HANDLE_OUT, LID_HANDLE_Y, TOP + LID_HANDLE_Z])
P_OPEN_AT = 5.0  # the bleed is shut again once the gauge reads this (kPa): in the middle of the green band
BOAT_IN = -0.240  # the boat set in the tube's mouth: its back end here (15 mm out; most of it, and its weight, inside)


def _bar(theta):
    """Where the lid's handle bar is, the lid lifted by theta, and the hand's rotation holding it (pointing in at the
    lid's front, the fingers closing up and down round the bar)."""
    hx, hz = HINGE[0], TOP + HINGE[1]
    dx, dz = BAR[0] - hx, BAR[2] - hz
    p = np.array([hx + dx * math.cos(theta) + dz * math.sin(theta), BAR[1], hz - dx * math.sin(theta) + dz * math.cos(theta)])
    R = axis_angle_matrix(Y, theta) @ axis_angle_matrix(Y, -math.pi / 2) @ _prev_grip(math.pi / 2)
    return p, R


def _lid_grip_rot(phi, sgn, psi):
    """A grip on the lid's handle bar (along y), the lid shut: the fingers closing along (cos phi, 0, sin phi), the hand
    coming in at psi degrees out of the x-z plane (towards -y for psi < 0)."""
    G = _prev_grip(math.pi / 2)
    a_l, c_l = G.T @ np.array([0.0, 0.0, -1.0]), G.T @ np.array([1.0, 0.0, 0.0])
    ml = np.column_stack([a_l, c_l, np.cross(a_l, c_l)])
    f, s_ = math.radians(phi), math.radians(psi)
    c = np.array([math.cos(f), 0.0, math.sin(f)])
    a = sgn * math.cos(s_) * np.array([-math.sin(f), 0.0, math.cos(f)]) + math.sin(s_) * Y
    a /= np.linalg.norm(a)
    return np.column_stack([a, c, np.cross(a, c)]) @ ml.T


LID_GRIP_A = _lid_grip_rot(120.0, -1, -55.0)  # holds the handle from shut to 55 degrees
LID_GRIP_B = _lid_grip_rot(75.0, 1, 0.0)  # and from 55 degrees to (nearly) upright
LID_SWAP = math.radians(55.0)
LID_VIA = [np.array([0.17, 0.27, TOP + 0.50]),  # up over the furnace's end, then down in front of it to take its handle
           np.array([float(os.environ.get("FA_VX", "0.04")), float(os.environ.get("FA_VY", "0.10")),
                     TOP + float(os.environ.get("FA_VZ", "0.36"))])]


def _lid(theta, R0):
    """The handle's bar and the hand's rotation (grip R0, turned with the lid) with the lid lifted by theta."""
    p, _ = _bar(theta)
    return p, axis_angle_matrix(Y, theta) @ R0


def _vpush(pivot, th, side, r=0.032):
    """Where a fingertip pushes a valve's lever (the lever at th): beside it, r along it, on its side ``side`` (+1: the
    side towards larger th, so pushing turns it to smaller th; -1 the other way)."""
    d = np.array([math.cos(th), math.sin(th), 0.0])
    n = np.array([-math.sin(th), math.cos(th), 0.0])
    return np.array([pivot[0], pivot[1], TOP + pivot[2] + 0.010]) + r * d + side * 0.014 * n  # (the fingers' half width)


class _GateSeq:
    """Waypoint targets that wait at the first until the gate ``key`` latches, then step through the rest, dt each."""

    def __init__(self, targets, key, dt):
        self.targets, self.key, self.dt = [np.asarray(x, float) for x in targets], key, dt

    def __array__(self, dtype=None, copy=None):
        t0 = _ct["gate"].get(self.key)
        if t0 is None:
            return np.asarray(self.targets[0], dtype=dtype or float)
        k = min(len(self.targets) - 1, 1 + int((getattr(_demo, "_plan_clock", 0.0) - t0) / self.dt))
        return np.asarray(self.targets[k], dtype=dtype or float)


class _GateRot(float):
    """A waypoint rotation that, like _GateSeq, waits at the first until the gate latches, then steps through the rest."""

    def __new__(cls, rots, key, dt):
        obj = float.__new__(cls, 0.0)
        obj.rots, obj.key, obj.dt = [np.asarray(r, float) for r in rots], key, dt
        return obj

    @property
    def rot(self):
        t0 = _ct["gate"].get(self.key)
        if t0 is None:
            return self.rots[0]
        return self.rots[min(len(self.rots) - 1, 1 + int((getattr(_demo, "_plan_clock", 0.0) - t0) / self.dt))]


def _vknob(pivot, th):
    """A valve lever's knob, the lever at th about z (0: along +x, open)."""
    k = LEVER_L - 0.004
    return np.array([pivot[0] + k * math.cos(th), pivot[1] + k * math.sin(th), TOP + pivot[2] + 0.034])  # its grip's top


ROT_C = axis_angle_matrix(Y, -TILT) @ _prev_grip(0.0)  # tipped forward, the fingers closing along y: pushing along y
O_TC = Orient(ROT_C)  # with the flat of a finger (its thin side)
_LEVER_OUT = _hook_y - 0.004 - CLAMP_PIVOT[1]  # the clamp lever's outside face (towards -y), from the hinge, shut


def _push_open(th):
    """Where a fingertip pushes the clamp's lever open (on its outside face), the clamp turned th about x."""
    return np.array(CLAMP_PIVOT) + [0.0, 0.0, TOP] + axis_angle_matrix(X, th) @ np.array([0.0, _LEVER_OUT - 0.005, 0.016])


ROT_K = axis_angle_matrix(Y, -math.radians(55.0)) @ _prev_grip(0.0)  # tipped 55 deg, closing along y (across the lever)
_TAB = np.array([0.0, _hook_y - 0.002 - CLAMP_PIVOT[1], 0.016])  # the clamp lever's middle, from its hinge (shut)
CLAMP_TO = -45.0  # degrees: the clamp turned up this far, its hook is clear of the plate (it slides out under it)


def _tab(th):
    """The clamp lever's middle and the hand holding it (fingers across it), the clamp turned th degrees about x."""
    Rx = axis_angle_matrix(X, math.radians(th))
    return np.array(CLAMP_PIVOT) + [0.0, 0.0, TOP] + Rx @ _TAB, Rx @ ROT_K


def _push_shut(th):
    """Where a fingertip pushes the clamp shut (on top of its arm, by the hook), the clamp turned th about x."""
    return np.array(CLAMP_PIVOT) + [0.0, 0.0, TOP] + axis_angle_matrix(X, th) @ np.array([0.0, -0.026, 0.0045])


def _rod_at(y, z=AXZ):
    return np.array([XA, y, TOP + z])


def _rod_seen_at(env, y):
    """Where the rod's axis is (seen) where it crosses y."""
    r = labkit._part_state(env, labkit._SPECS["hook_rod"])
    d = r.rotation()[:, 1]
    p = np.asarray(r.pos, float)
    return p + d * ((y - p[1]) / d[1])


def _on_rod(p):
    """TCP for the fingers round the rod at p, their tips 6 mm past its axis (so the pads, not the tips, hold it)."""
    return _tcp(p, past=0.006)


def _gauge_kpa(env):
    """The pressure as read off the gauge's needle (where it points on the dial)."""
    a = -labkit._part_state(env, labkit._SPECS["quartz_tube"]).joints.get("needle", 0.0)
    return P_MIN + (a + DIAL_SWEEP / 2) / DIAL_SWEEP * (P_MAX - P_MIN)


def demo():
    _ct["trim"], _ct["wins"], _ct["gates"] = {}, [], []
    pl = _Plan()
    up = np.array([0.0, 0.0, 1.0])
    ROT_D = axis_angle_matrix(Y, -math.radians(70.0)) @ _prev_grip(math.pi / 2)  # in from the front, 20 deg down,
    O_D = Orient(ROT_D)  # the fingers closing nearly up and down (on the boat's tongue)

    def _lid_moves(thetas, R0, label, first, pre=0.03, via=None):
        """Take the handle (grip R0) at thetas[0] coming in along the grip, turn the lid through thetas, let go, back
        off along the grip. ``via``: first to here (the hand already turned), clear of the furnace."""
        p, R = _lid(thetas[0], R0)
        a = _appr(R)
        out = []
        for k, v in enumerate(via or []):
            out.append((2.5 if k == 0 else 1.5, _tcp(np.asarray(v, float), R, past=0.002), BAR_OPEN,
                        first if k == 0 else "down in front of the lid", Orient(R)))
        out += [(2.0, _tcp(p - a * pre, R, past=0.002), BAR_OPEN, first if not via else "in to the handle", Orient(R)),
               (1.0, _tcp(p, R, past=0.002), BAR_OPEN, "round the handle", Orient(R)),
               (0.6, _tcp(p, R, past=0.002), BAR_HOLD, "close on the handle", Orient(R))]
        for th in thetas[1:]:
            p, R = _lid(th, R0)
            out.append((1.4, _tcp(p, R, past=0.002), BAR_HOLD, label, Orient(R)))
        out += [(0.6, _tcp(p, R, past=0.002), BAR_OPEN, "let go of the lid", Orient(R)),
                (0.8, _tcp(p - _appr(R) * pre, R, past=0.002), BAR_OPEN, "back off the handle", Orient(R))]
        return out

    # -- 1. right turns the argon valve shut, a third of a turn at a time (the wrist turning with the lever; the tube is
    # over its green band and still filling); left takes the bleed valve's grip ------------------------------------------
    def vturn(pivot, a, b, label, n=3, release=True):
        """Take the lever's grip at a (degrees) and turn it to b, the wrist turning with it."""
        k0 = _vknob(pivot, math.radians(a))
        out = [(1.0, _tcp(k0 + 0.03 * up, past=0.002), KNOB_OPEN, "over the valve's grip", O_T),
               (0.8, _tcp(k0, past=0.002), KNOB_OPEN, "down round the grip", O_T),
               (0.5, _tcp(k0, past=0.002), KNOB_HOLD, "close on the grip", O_T)]
        for t in np.linspace(a, b, n + 1)[1:]:
            R = axis_angle_matrix(Z, math.radians(t - a)) @ ROT_T
            out.append((0.6, _tcp(_vknob(pivot, math.radians(t)), R, past=0.002), KNOB_HOLD, label, Orient(R)))
        if release:
            R = axis_angle_matrix(Z, math.radians(b - a)) @ ROT_T
            kb = _vknob(pivot, math.radians(b))
            out += [(0.4, _tcp(kb, R, past=0.002), KNOB_OPEN, "let go of the grip", Orient(R)),
                    (0.6, _tcp(kb + 0.03 * up, R, past=0.002), KNOB_OPEN, "up off the grip", Orient(R))]
        return out

    from yam_arena.constants import LEFT_BASE_POS
    _READY_L = _ik.tcp_pose(np.array(_READY, float), np.array(LEFT_BASE_POS))  # where the left hand starts
    pl.phase(right=[(2.0, np.array([0.37, -0.27, TOP + 0.37]), KNOB_OPEN, "hand turned, up clear of the tube's end", O_T),
                    *vturn(ARGON_PIVOT, 0, -30, "turn the argon off"), *vturn(ARGON_PIVOT, -30, -60, "turn the argon off"),
                    *vturn(ARGON_PIVOT, -60, -92, "turn the argon off")],
             left=[(0.8, _READY_L[0] + 0.07 * up, KNOB_OPEN, "straight up, off the bleed valve's grip",
                    Orient(_READY_L[1])),
                   (1.5, np.array([0.37, 0.27, TOP + 0.37]), KNOB_OPEN, "hand turned, up clear of the bleed valve", O_T),
                   (1.5, _tcp(_vknob(BLEED_PIVOT, 0.0) + 0.03 * up, past=0.002), KNOB_OPEN,
                    "over the bleed valve's grip", O_T)])
    # -- 2. left opens the bleed a third and holds it, watching the gauge; in the green band, turns it shut -------------
    pl.phase(left=vturn(BLEED_PIVOT, 0, 30, "open the bleed (a third)", release=False))
    pl.gate(pl.t + 0.2, pl.t + 16.0, "bleed", lambda env: _gauge_kpa(env) <= P_OPEN_AT)  # once it is open and held
    back = [30, 20, 10, 0, -2]
    rots = [axis_angle_matrix(Z, math.radians(t)) @ ROT_T for t in back]
    pts = [_tcp(_vknob(BLEED_PIVOT, math.radians(t)), R, past=0.002) for t, R in zip(back, rots)]
    gr = _GateRot(rots, "bleed", 0.6)
    pl.phase(left=[(0.6, _GateSeq(pts, "bleed", 0.6), KNOB_HOLD, "bleed until the gauge is in the green; then shut it", gr)
                   for _ in range(30)]
             + [(0.4, pts[-1], KNOB_OPEN, "let go of the grip", Orient(rots[-1])),
                (0.6, pts[-1] + 0.03 * up, KNOB_OPEN, "up off the grip", Orient(rots[-1]))])
    # -- 3. left lifts the lid (fully open), so the hand on the clamp has room -------------------------------------------
    if _SKIP:  # DEBUG-SKIP: start here (the valves already shut)
        pl = _Plan()
        _ct["gates"] = []
        pl.phase(right=[(2.0, np.array([0.37, -0.27, TOP + 0.37]), KNOB_OPEN, "hand turned, up clear of the tube's end", O_T)],
                 left=[(2.0, _READY_L[0] + 0.07 * up, KNOB_OPEN, "straight up, off the bleed valve's grip",
                        Orient(_READY_L[1]))])
    lid = [*_lid_moves([0.0, math.radians(15), math.radians(30), math.radians(45), LID_SWAP], LID_GRIP_A,
                       "lift the lid", first="hand turned, in front of the lid", via=LID_VIA),
           *_lid_moves([LID_SWAP, math.radians(70), LID_OPEN], LID_GRIP_B, "lift the lid (on up)",
                       first="the hand round to hold the handle from above")]
    pl.phase(left=lid)
    # -- 4. left holds the tube; right flips the clamp open ---------------------------------------------------------------
    h = _rod_at(HOLD_Y)
    clear = np.array([0.215, 0.350, TOP + AXZ + 0.10])
    pl.phase(left=[(2.5, _tcp(clear), TUBE_OPEN, "hand turned, past the lid's end", O_T),
                   (1.5, _tcp(h + 0.08 * up), TUBE_OPEN, "over the tube", O_T),
                   (1.0, _tcp(h), TUBE_OPEN, "down round the tube", O_T),
                   (0.6, _tcp(h), TUBE_HOLD, "hold the tube (it slides and rolls)", O_T)],
             right=[(2.0, _tcp(_tab(0.0)[0] + 0.03 * up, _tab(0.0)[1], past=0.004), TAB_OPEN,
                     "hand turned over the clamp's lever", Orient(_tab(0.0)[1])),
                    (1.0, _tcp(_tab(0.0)[0], _tab(0.0)[1], past=0.004), TAB_OPEN, "down round the lever",
                     Orient(_tab(0.0)[1]))])
    pl.phase(right=[(0.5, _tcp(_tab(0.0)[0], _tab(0.0)[1], past=0.004), TAB_HOLD, "close on the lever",
                     Orient(_tab(0.0)[1])),
                    *[(0.7, _tcp(_tab(a)[0], _tab(a)[1], past=0.004), TAB_HOLD, "turn the clamp up, off the plate",
                       Orient(_tab(a)[1])) for a in (-15.0, -30.0, CLAMP_TO)],
                    (0.4, _tcp(_tab(CLAMP_TO)[0], _tab(CLAMP_TO)[1], past=0.004), TAB_OPEN, "let go of the lever",
                     Orient(_tab(CLAMP_TO)[1])),
                    (0.8, _tcp(_tab(CLAMP_TO)[0] - 0.035 * X + 0.02 * up, _tab(-10.0)[1], past=0.004), TAB_OPEN,
                     "back off the lever, the wrist turning back", Orient(_tab(-10.0)[1]))])
    # -- 5. right takes the plate off (the gauge in its green band, the argon shut) and stands it on its stand ---------
    k0 = PLATE_START + [0.0, -PLATE_T - KNOB_L / 2, 0.0]
    kr = PLATE_REST + [0.0, -PLATE_T - KNOB_L / 2, 0.0]
    pl.phase(right=[(1.5, _tcp(k0 + 0.04 * up), PK_OPEN, "hand turned over the plate's knob", O_T),
                    (1.0, _tcp(k0), PK_OPEN, "down round the knob", O_T),
                    (0.6, _tcp(k0), PK_HOLD, "close on the knob", O_T),
                    (1.5, _tcp(k0 - 0.018 * Y), PK_HOLD, "pull the plate off (the gauge in the green)", O_T),
                    (1.0, _tcp(k0 - 0.018 * Y + 0.035 * up), PK_HOLD, "lift the plate", O_T),
                    (2.0, _tcp(kr + 0.04 * up), PK_HOLD, "the plate over its stand", O_T),
                    (1.2, _tcp(kr + 0.002 * up), PK_HOLD, "the plate down in its stand", O_T),
                    (0.5, _tcp(kr + 0.002 * up), PK_OPEN, "let go of the plate", O_T),
                    (0.8, _tcp(kr + 0.05 * up), PK_OPEN, "up off the plate", O_T)])
    # -- 6. right picks the boat up from its tray by its tongue and sets it in the tube's mouth, lined up by eye --------
    tg = BOAT_START + [0.0, -BOAT_L / 2 - TONGUE_L / 2, TONGUE_T / 2]
    a_d = _appr(ROT_D)
    pick = [(2.0, _tcp(tg - 0.04 * a_d + 0.05 * up, ROT_D, past=0.006), TONGUE_OPEN, "hand turned over the boat's tongue",
             O_D),
            (0.8, _tcp(tg - 0.04 * a_d, ROT_D, past=0.006), TONGUE_OPEN, "down, level with the tongue", O_D),
            (1.0, _tcp(tg, ROT_D, past=0.006), TONGUE_OPEN, "in round the tongue", O_D),
            (0.6, _tcp(tg, ROT_D, past=0.006), TONGUE_HOLD, "close on the tongue", O_D),
            (1.5, _tcp(tg + 0.035 * up, ROT_D, past=0.006), TONGUE_HOLD, "lift the boat, level", O_D)]
    sb = np.array([XA, BOAT_IN + BOAT_L / 2, TOP + BOAT_Z])  # the boat set in the mouth (its frame's origin)
    pre = np.array([XA, TUBE_Y0 - 0.006 - BOAT_L / 2, TOP + BOAT_Z + 0.0012])  # its front 6 mm short of the mouth
    g_pre = pre + (tg - BOAT_START)  # the tongue's grip point, the boat in front of the mouth
    pl.phase(right=pick + [(2.5, _Dyn(_tcp(g_pre + 0.02 * up, ROT_D, past=0.006), "b"), TONGUE_HOLD,
                            "the boat over the tube's mouth", O_D),
                           (1.5, _Dyn(_tcp(g_pre, ROT_D, past=0.006), "b"), TONGUE_HOLD,
                            "the boat down, in line with the tube", O_D)])
    tw = pl.phase(right=[(1.8, _Dyn(_tcp(g_pre, ROT_D, past=0.006), "b"), TONGUE_HOLD, "hold still: the boat seen in line", O_D)])
    pl.window(tw + 0.3, tw + 1.8, "b",
              lambda env: np.array([XA, 0.0, pre[2]]) - _seen(env, "boat") * [1.0, 0.0, 1.0], "boat")
    g_in = g_pre + (sb - pre)
    pl.phase(right=[(2.5, _Dyn(_tcp(g_in, ROT_D, past=0.006), "b"), TONGUE_HOLD, "slide the boat into the mouth", O_D),
                    (0.5, _Dyn(_tcp(g_in, ROT_D, past=0.006), "b"), TONGUE_OPEN, "let go of the boat", O_D),
                    (0.8, _Dyn(_tcp(g_in - 0.035 * a_d, ROT_D, past=0.006), "b"), TONGUE_OPEN, "back off the boat", O_D),
                    (0.8, _Dyn(_tcp(g_in - 0.035 * a_d + 0.04 * up, ROT_D, past=0.006), "b"), TONGUE_OPEN, "up and away",
                     O_D)])
    # -- 7. right takes the rod and pushes the boat in, stroke by stroke, until it sits under the thermocouple (seen) ---
    r_g0 = _rod_at(FEED[0], AXZ - 0.0015)
    pl.phase(right=[(2.0, _on_rod(r_g0 + 0.09 * up), ROD_OPEN, "hand turned over the rod", O_T),
                    (1.0, _on_rod(r_g0 + 0.03 * up), ROD_OPEN, "down over the rod", O_T),
                    (1.0, _on_rod(r_g0), ROD_OPEN, "down round the rod", O_T),
                    (0.6, _on_rod(r_g0), ROD_HOLD, "close on the rod", O_T)])
    seq = []
    cur = {"k": "r"}
    keys = iter(f"r{i}" for i in range(1, 20))

    def stroke(y1, label, dur):
        seq.append((dur, _Dyn(_on_rod(_rod_at(y1)), cur["k"]), ROD_HOLD, label, O_T))

    def regrip(y_from, y_to):
        """Let go, move along, line up on the rod as seen while hovering 3 cm over it (the open fingers clear of it),
        keep trimming on the way down, then close."""
        k_old, k_new = cur["k"], next(keys)
        cur["k"] = k_new
        over = _on_rod(_rod_at(y_to) + 0.03 * up)
        seq.extend([(0.6, _Dyn(_on_rod(_rod_at(y_from)), k_old), ROD_OPEN, "let go of the rod", O_T),
                    (0.8, _Dyn(_on_rod(_rod_at(y_from) + 0.03 * up), k_old), ROD_OPEN, "up off the rod", O_T),
                    (1.2, _Dyn(over, k_new), ROD_OPEN, "along the rod", O_T)])
        t0 = pl.t + sum(w[0] for w in seq)
        pl.window(t0 + 0.1, t0 + 2.4, k_new,
                  lambda env, y=y_to, key=k_new: (_rod_seen_at(env, y) - _rod_at(y) - _ct["trim"].get(key, np.zeros(3)))
                  * [1.0, 0.0, 1.0], "hook_rod")
        seq.extend([(1.4, _Dyn(over, k_new), ROD_OPEN, "line up over the rod (seen)", O_T),
                    (1.0, _Dyn(_on_rod(_rod_at(y_to)), k_new), ROD_OPEN, "down round the rod", O_T),
                    (0.6, _Dyn(_on_rod(_rod_at(y_to)), k_new), ROD_HOLD, "close on the rod", O_T)])

    tongue_end = BOAT_IN - TONGUE_L
    touch = FEED[0] + (tongue_end - 0.004) - ROD_START[1]  # the hand, the pusher 4 mm from the tongue's end
    last = H_IN - (ROD_START[1] + 2 * (FEED[1] - FEED[0]))  # the third stroke's length
    seq.append((1.0, _on_rod(_rod_at(FEED[0])), ROD_HOLD, "lift the rod off its rests", O_T))
    stroke(touch - 0.010, "the pusher up behind the boat", 1.5)
    t0 = pl.t + sum(w[0] for w in seq)
    stroke(touch - 0.010, "hold still: the rod's tip seen in line with the tube", 1.5)
    pl.window(t0 + 0.2, t0 + 1.5, "r",
              lambda env: np.array([XA, 0.0, TOP + AXZ]) - _rod_seen_at(env, tongue_end - 0.014) * [1.0, 0.0, 1.0],
              "hook_rod")
    stroke(touch, "the pusher to the boat's tongue", 1.0)
    stroke(FEED[1], "push the boat in, slowly", 6.0)
    regrip(FEED[1], FEED[0])
    stroke(FEED[1], "push the boat on", 5.0)
    regrip(FEED[1], FEED[0])
    stroke(FEED[0] + last, "push on until the boat is under the thermocouple", 3.0)
    k_push = cur["k"]
    pl.phase(right=seq)
    tw = pl.phase(right=[(2.0, _Dyn(_on_rod(_rod_at(FEED[0] + last)), k_push), ROD_HOLD,
                          "hold still: the boat seen under the thermocouple", O_T)])
    pl.window(tw + 0.3, tw + 2.0, k_push,
              lambda env: np.array([0.0, HOT_Y - (_seen(env, "boat")[1]), 0.0]), "boat")
    # -- 8. right pulls the rod back out, stroke by stroke, and lays it on its rests ------------------------------------
    seq = []
    stroke(FEED[0], "pull the rod back", 2.5)
    regrip(FEED[0], FEED[1])
    stroke(FEED[0], "pull the rod back", 4.0)
    regrip(FEED[0], FEED[1])
    stroke(FEED[0], "pull the rod out", 4.0)
    k = cur["k"]
    seq += [(1.0, _Dyn(_on_rod(_rod_at(FEED[0], AXZ - 0.0015)), k), ROD_HOLD, "down onto its rests", O_T),
            (0.6, _Dyn(_on_rod(_rod_at(FEED[0], AXZ - 0.0015)), k), ROD_OPEN, "let go of the rod", O_T),
            (1.0, _Dyn(_on_rod(_rod_at(FEED[0], AXZ - 0.0015)) + 0.04 * up, k), ROD_OPEN, "up off the rod", O_T)]
    pl.phase(right=seq)
    # -- 9. right puts the plate back on, in line with the tube (seen), fully home; then shuts the clamp over it --------
    pre_k = k0 - [0.0, 0.025, 0.0]
    pl.phase(right=[(2.5, _tcp(kr + 0.05 * up), PK_OPEN, "hand turned over the plate on its stand", O_T),
                    (1.2, _tcp(kr + 0.002 * up), PK_OPEN, "down round the knob", O_T),
                    (0.6, _tcp(kr + 0.002 * up), PK_HOLD, "close on the knob", O_T),
                    (1.0, _tcp(kr + 0.04 * up), PK_HOLD, "lift the plate", O_T),
                    (2.0, _tcp(pre_k + 0.035 * up), PK_HOLD, "the plate over the tube's end", O_T),
                    (1.5, _Dyn(_tcp(pre_k), "c1"), PK_HOLD, "the plate in front of the tube's end", O_T)])
    tw = pl.phase(right=[(2.0, _Dyn(_tcp(pre_k), "c1"), PK_HOLD, "hold still: the plate seen in line", O_T)])
    pl.window(tw + 0.4, tw + 2.0, "c1",
              lambda env: np.array([XA - _seen(env, "end_plate")[0], 0.0, TOP + AXZ - _seen(env, "end_plate")[2]]),
              "end_plate")
    home = k0 + [0.0, 0.0005, 0.0]
    pl.phase(right=[(2.0, _Dyn(_tcp(home), "c1"), PK_HOLD, "push the plate home", O_T),
                    (0.6, _Dyn(_tcp(home), "c1"), PK_HOLD, "hold it home", O_T),
                    (0.5, _Dyn(_tcp(home), "c1"), PK_OPEN, "let go of the plate", O_T),
                    (0.8, _Dyn(_tcp(home) - 0.02 * Y + 0.03 * up, "c1"), PK_OPEN, "back off the plate", O_T),
                    (1.5, _tcp(_tab(CLAMP_TO)[0] - 0.035 * X + 0.02 * up, _tab(-10.0)[1], past=0.004), TAB_OPEN,
                     "hand turned to the clamp's lever", Orient(_tab(-10.0)[1])),
                    (0.8, _tcp(_tab(CLAMP_TO)[0], _tab(CLAMP_TO)[1], past=0.004), TAB_OPEN, "down round the lever",
                     Orient(_tab(CLAMP_TO)[1])),
                    (0.5, _tcp(_tab(CLAMP_TO)[0], _tab(CLAMP_TO)[1], past=0.004), TAB_HOLD, "close on the lever",
                     Orient(_tab(CLAMP_TO)[1])),
                    *[(0.7, _tcp(_tab(a)[0], _tab(a)[1], past=0.004), TAB_HOLD, "turn the clamp down over the plate",
                       Orient(_tab(a)[1])) for a in (-30.0, -15.0, 0.0, 3.0)],
                    (0.4, _tcp(_tab(0.0)[0], _tab(0.0)[1], past=0.004), TAB_OPEN, "let go of the lever", Orient(_tab(0.0)[1])),
                    (0.8, _tcp(_tab(0.0)[0] + 0.05 * up, _tab(0.0)[1], past=0.004), TAB_OPEN, "up off the clamp",
                     Orient(_tab(0.0)[1])),
                    (2.0, RIGHT_PARK, TAB_OPEN, "up and clear", O_T)])
    # -- 10. left lets go of the tube and lowers the lid by its handle -----------------------------------------------------
    shut_lid = [(0.6, _tcp(h), TUBE_OPEN, "let go of the tube", O_T),
                (0.8, _tcp(h) - APPR_T * 0.025, TUBE_OPEN, "back off the tube along the hand", O_T),
                (1.0, _tcp(h + 0.08 * up), TUBE_OPEN, "up off the tube", O_T),
                (1.5, _tcp(clear), TUBE_OPEN, "back past the lid's end", O_T),
                (2.0, _tcp(_lid(math.radians(68), LID_GRIP_B)[0] - _appr(_lid(math.radians(68), LID_GRIP_B)[1]) * 0.02,
                           _lid(math.radians(68), LID_GRIP_B)[1], past=0.002), BAR_OPEN,
                 "hand turned, under the lifted lid's handle", Orient(_lid(math.radians(68), LID_GRIP_B)[1])),
                *_lid_moves([LID_OPEN, math.radians(70), LID_SWAP], LID_GRIP_B, "lower the lid",
                            first="up to the lid's handle", pre=0.012),
                *_lid_moves([LID_SWAP, math.radians(45), math.radians(30), math.radians(15), 0.0], LID_GRIP_A,
                            "lower the lid (on down)", first="the hand round to hold the handle from the front")]
    pa, Ra = _lid(0.0, LID_GRIP_A)
    shut_lid += [(1.0, _tcp(pa - _appr(Ra) * 0.03 + 0.02 * up, Ra, past=0.002), BAR_OPEN, "away from the handle",
                  Orient(Ra)),
                 (1.5, _tcp(np.array([0.06, 0.12, TOP + 0.36]), Ra, past=0.0), BAR_OPEN, "out in front of the furnace",
                  Orient(Ra)),
                 (1.5, LEFT_PARK, BAR_OPEN, "hand turned at the park", O_T)]
    pl.phase(left=shut_lid)
    pl.phase(right=[(2.0, RIGHT_PARK, FINGER_CLOSED, "rest", O_T)], left=[(2.0, LEFT_PARK, FINGER_CLOSED, "rest", O_T)])
    return {"left": pl.L, "right": pl.R}
