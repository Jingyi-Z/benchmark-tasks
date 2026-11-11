"""Laying out a 2 inch tube furnace for MoS2 growth by CVD, after the lab's procedure.

A split-tube furnace stands in front of the robots, its lid open, a quartz tube across it. Its end caps are off for
loading, each hung beside its end of the tube on its line: the argon inlet's cap (its hose from the gas bottle) at the
loading end, the exhaust's (its hose to an oil bubbler) at the far end. On a tray in front of the loading end, under the
push rod's line: an alumina boat of sulfur, an alumina boat of MoO3 and a quartz boat with a 1 cm SiO2/Si chip, face up.
For MoS2 the sulfur goes upstream at the furnace's edge (cool), the MoO3 under the thermocouple (the hot zone's centre)
and the chip a few cm downstream of it. Upstream is where the gas comes in, here the loading end, so the chip goes in
first and deepest; the boats cannot pass each other in the tube.

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



# -- the furnace (fixed), on a small desk: a split-tube furnace for a 2 inch tube, its front towards the robots: a base
# with the controller, the heated lower half, and the lid, hinged along the back, standing open. The tube runs across it
# along y. --------------------------------------------------------------------------------------------------------------
DESK = 0.0  # no desk: the tube's far end stays under the left hand's start pose
XA, AXZ = 0.240, DESK + 0.210  # the tube's axis: x, height above the bench
TUBE_R, TUBE_IN = 0.0254, 0.0230  # 2 inch quartz tube: 50.8 mm outside, 46 mm inside
TUBE_Y0, TUBE_Y1 = -0.225, 0.385  # the tube's ends: the loading end (towards the right arm) and the far end
FX0, FX1, FY0, FY1 = 0.100, 0.380, -0.180, 0.190  # the furnace body: front, back, ends (370 mm wide, 280 mm deep)
BASE_Z = DESK + 0.110  # the base (controller) under the heated chamber
LID_Z1 = DESK + 0.360  # the lid's top
HOT_Y = (FY0 + FY1) / 2  # the hot zone's centre, where the thermocouple's tip is
EDGE_Y = FY0 + 0.040  # the heated zone's upstream edge (the near end wall is 20 mm thick, the heater stops 20 mm in)
HINGE = (FX1 - 0.010, AXZ)  # the lid's hinge along the back, at the split
SLOT = 0.029  # half the gap the tube lies in, front to back, in both halves
WHITE_F = (0.90, 0.90, 0.88)
PANEL_C = (0.40, 0.52, 0.70)
INSUL_C = (0.95, 0.93, 0.86)
COIL_C = (0.55, 0.42, 0.33)
SCREEN_C = (0.06, 0.07, 0.08)
LED_C = (0.95, 0.20, 0.12)
TC_C = (0.70, 0.71, 0.73)
DESK_C = (0.36, 0.37, 0.40)
_mid_y, _len_y = (FY0 + FY1) / 2, FY1 - FY0
CON_N = np.array([-0.6, 0.0, 0.8])  # the controller's sloped face: its normal, up the face, across it
CON_U = np.array([0.8, 0.0, 0.6])
CON_R = -Y
CON_C = np.array([FX0 - 0.035, _mid_y, DESK + 0.060])
CON_W, CON_H = 0.190, 0.090
DESK_X0 = CON_C[0] - CON_H / 2 * CON_U[0] - 0.012
FURNACE_BOXES = [
    (((DESK_X0 + FX1 + 0.015) / 2, _mid_y, DESK / 2), (FX1 + 0.015 - DESK_X0, _len_y + 0.030, DESK), DESK_C),
    (((FX0 + FX1) / 2, _mid_y, (DESK + BASE_Z) / 2), (FX1 - FX0, _len_y, BASE_Z - DESK), WHITE_F),
    ((FX0 - 0.0005, _mid_y, (DESK + BASE_Z) / 2), (0.001, _len_y - 0.02, BASE_Z - DESK - 0.016), PANEL_C),
    # the heated lower half: front and back blocks either side of the tube's gap, a floor, the ends
    (((FX0 + XA - SLOT) / 2, _mid_y, (BASE_Z + AXZ) / 2), (XA - SLOT - FX0, _len_y, AXZ - BASE_Z), WHITE_F),
    (((XA + SLOT + FX1) / 2, _mid_y, (BASE_Z + AXZ) / 2), (FX1 - XA - SLOT, _len_y, AXZ - BASE_Z), WHITE_F),
    ((XA, _mid_y, BASE_Z + 0.01), (2 * SLOT, _len_y, 0.02), INSUL_C),
    *[((XA, y, (BASE_Z + AXZ - TUBE_R - 0.004) / 2), (2 * SLOT, 0.020, AXZ - TUBE_R - 0.004 - BASE_Z), WHITE_F)
      for y in (FY0 + 0.010, FY1 - 0.010)],
    # the thermocouple: in through the back wall, its tip beside the tube at the hot zone's centre
    ((((XA + SLOT - 0.002) + FX1) / 2 - 0.02, HOT_Y, AXZ - 0.004), ((FX1 - 0.04) - (XA + SLOT - 0.002), 0.003, 0.003),
     TC_C),
]
FURNACE_CYLS = [((XA + SLOT - 0.002, HOT_Y, AXZ - 0.004), 0.0028, 0.004, TC_C, "X")]  # the junction bead
SADDLES = [m for y in (FY0, FY1 - 0.020) for m in _ring_y(y, y + 0.020, TUBE_R + 0.0006, TUBE_R + 0.008, 8, INSUL_C,
                                                        cx=XA, cz=AXZ, a0=math.radians(200), a1=math.radians(340),
                                                        collide="none")]
SEG.update({"C": "adef", "°": "abgf"})


def _face_box(c, du, dr, su, sr, t, colour):
    o = c + du * CON_U + dr * CON_R
    return _hex([o + (i - 0.5) * sr * CON_R + (j - 0.5) * su * CON_U + k * t * CON_N for i in (0, 1) for j in (0, 1)
                 for k in (0, 1)], colour, collide="none")


def _console():
    """The controller (a wedge) with its display: "28°C" in red seven-segment digits (the furnace is cold)."""
    lo, hi = CON_C - CON_H / 2 * CON_U, CON_C + CON_H / 2 * CON_U
    w2 = CON_W / 2
    wedge = _hex([(x, _mid_y + sy * w2, z) for x, z in ((lo[0], lo[2]), (hi[0], hi[2]), (FX0 + 0.005, lo[2]),
                                                       (FX0 + 0.005, hi[2])) for sy in (-1, 1)], PANEL_C)
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


_coils = [((XA + s * (SLOT - 0.003), y, AXZ - 0.016), (0.003, 0.004, 0.016)) for s in (-1, 1)
          for y in np.arange(EDGE_Y, FY1 - 0.040, 0.012)]
FURNACE_MESHES = [*SADDLES, _boxes_mesh(_coils, COIL_C), *_console()]
# the lid, standing open (held by its stop): front and back blocks either side of the tube's gap, the top over it
_lid_boxes = [
    (((FX0 + XA - SLOT) / 2, _mid_y, (AXZ + LID_Z1) / 2), (XA - SLOT - FX0, _len_y, LID_Z1 - AXZ), WHITE_F),
    (((XA + SLOT + FX1) / 2, _mid_y, (AXZ + LID_Z1) / 2), (FX1 - XA - SLOT, _len_y, LID_Z1 - AXZ), WHITE_F),
    ((XA, _mid_y, (AXZ + 0.040 + LID_Z1) / 2), (2 * SLOT, _len_y, LID_Z1 - AXZ - 0.040), WHITE_F),
]
LID_OPEN = math.radians(100.0)  # opened just past upright, leaning back off the left hand
LID = Joint("lid", kind="hinge", axis="Y", anchor=(HINGE[0], _mid_y, HINGE[1]), lower=0.0, upper=LID_OPEN + 0.02,
            stiffness=40.0, target=LID_OPEN, damping=1.0, initial=LID_OPEN, mass=0.4, boxes=_lid_boxes)
FURNACE = part("tube_furnace", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5, boxes=FURNACE_BOXES,
               cylinders=FURNACE_CYLS, meshes=FURNACE_MESHES, joints=[LID])

# -- the tube (fixed in its end clamps): fused quartz, 2 inch, 610 mm long, open at both ends (its end caps are off for
# loading). Its wall in 45 mm lengths, 12-sided, and a smooth strip along its floor that the boats slide on. ------------
FLOOR_W = 0.020
FLOOR_Z = AXZ - math.sqrt(TUBE_IN ** 2 - (FLOOR_W / 2) ** 2) + 0.0006  # the strip's top: as wide as a boat
_CUTS = list(np.arange(TUBE_Y0, TUBE_Y1, 0.045)) + [TUBE_Y1]
TUBE_MESHES = [m for y0, y1 in zip(_CUTS[:-1], _CUTS[1:]) for m in _ring_y(y0, y1, TUBE_IN, TUBE_R, 12, QUARTZ, cx=XA,
                                                                          cz=AXZ)]
TUBE_BOXES = [((XA, (TUBE_Y0 + TUBE_Y1) / 2, FLOOR_Z - 0.0015), (FLOOR_W, TUBE_Y1 - TUBE_Y0, 0.003), QUARTZ)]
# the tube's end clamps: a block under each end outside the furnace, with a strap over it
TUBE_HOLD = [((XA, y, (AXZ - TUBE_R) / 2), (0.030, 0.020, AXZ - TUBE_R), DARK) for y in (TUBE_Y0 + 0.020, TUBE_Y1 - 0.020)]
TUBE = part("quartz_tube", pos=(0.0, 0.0, TOP), fixed=True, friction=0.3, boxes=TUBE_BOXES + TUBE_HOLD,
            meshes=TUBE_MESHES)

# -- the end caps, off for loading, each on a hook beside its end of the tube, still on its line: the argon inlet's cap
# (its hose from the gas bottle's regulator) at the loading end, the exhaust's cap (its hose to an oil bubbler) at the far
# end. Upstream is the end the gas comes in. Fixed. ----------------------------------------------------------------------
HOSE_C = (0.30, 0.55, 0.85)
EXH_C = (0.25, 0.26, 0.28)
BRASS = (0.78, 0.66, 0.30)
ARGON_C = (0.25, 0.48, 0.30)  # the gas bottle (argon: green shoulder)
CAP_R = 0.035
CAP_NEAR = np.array([XA - 0.110, TUBE_Y0 - 0.010, AXZ - 0.020])  # the inlet cap's centre, hung face up on its hook
CAP_FAR = np.array([XA - 0.110, TUBE_Y1 + 0.010, AXZ - 0.020])
BOTTLE = (0.40, -0.66)
BUBBLER = (0.05, 0.62)


def _cap(c, colour):
    return [*_disc_mesh(c, Z, CAP_R, 0.012, STAINLESS), _seg(c + [0.0, 0.0, 0.006], c + [0.0, 0.0, 0.030], 0.010, BRASS),
            _box_mesh((c[0] - 0.050, c[1], (c[2] - 0.006) / 2), (0.012, 0.012, c[2] - 0.006), DARK, collide="hull"),
            _seg((c[0] - 0.050, c[1], c[2] - 0.006), (c[0] - CAP_R + 0.004, c[1], c[2] - 0.006), 0.008, DARK)]


CAP_MESHES = [
    *_cap(CAP_NEAR, HOSE_C), *_cap(CAP_FAR, EXH_C),
    # the argon bottle and its regulator; its hose to the inlet cap's barb
    *_disc_mesh(np.array([BOTTLE[0], BOTTLE[1], 0.0]), Z, 0.040, 0.250, (0.55, 0.56, 0.58)),
    *_disc_mesh(np.array([BOTTLE[0], BOTTLE[1], 0.250]), Z, 0.040, 0.050, ARGON_C),
    _box_mesh((BOTTLE[0], BOTTLE[1], 0.315), (0.030, 0.030, 0.030), BRASS),
    *_hose([(BOTTLE[0] - 0.015, BOTTLE[1], 0.315), (BOTTLE[0] - 0.06, BOTTLE[1] + 0.02, 0.30),
            (BOTTLE[0] - 0.10, BOTTLE[1] + 0.04, 0.012), (0.10, BOTTLE[1] + 0.04, 0.012), (0.10, CAP_NEAR[1] - 0.07, 0.012),
            (CAP_NEAR[0] - 0.02, CAP_NEAR[1] - 0.04, CAP_NEAR[2] + 0.02), (CAP_NEAR[0], CAP_NEAR[1], CAP_NEAR[2] + 0.030)],
           0.008, HOSE_C),  # (along the bench, round the rod's rack)
    # the oil bubbler (a glass bottle) and the exhaust hose from the far cap
    *_disc_mesh(np.array([BUBBLER[0], BUBBLER[1], 0.0]), Z, 0.035, 0.160, QUARTZ),
    *_hose([(CAP_FAR[0], CAP_FAR[1], CAP_FAR[2] + 0.030), (CAP_FAR[0], CAP_FAR[1] + 0.06, CAP_FAR[2] + 0.06),
            (BUBBLER[0], BUBBLER[1], 0.20), (BUBBLER[0], BUBBLER[1], 0.12)], 0.008, EXH_C),
]
CAPS = part("end_caps", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5, meshes=CAP_MESHES)

# -- the boats (loose), side by side on a tray beside the loading end (behind the tube's line), level with the tube's
# floor: an alumina boat of sulfur powder, an alumina boat of MoO3 powder, and a short quartz boat carrying the 1 cm chip,
# face up. Each has a square tongue out of its back end, to hold it by and for the rod's pusher. Frame: the centre of its
# underside. -----------------------------------------------------------------------------------------------------------
ALUMINA = (0.97, 0.96, 0.93)
SULFUR_C = (0.95, 0.85, 0.15)
MOO3_C = (0.86, 0.88, 0.78)
QBOAT_C = (0.80, 0.86, 0.88)
CHIP_C = (0.42, 0.33, 0.58)
BOAT_W, BOAT_H, BOAT_T = 0.020, 0.012, 0.002
TONGUE_L, TONGUE_W = 0.020, 0.005  # a 5 mm square bar out of the back wall, turned on its axis
TONGUE_TURN = math.radians(-55.0)  # its faces square to the tipped hand's closing (checked against ROT_D in the demo)
TONGUE_Z = 0.004  # the bar's axis over the boat's underside (its lowest edge 0.5 mm above it)
BOATS = {"s_boat": (0.060, ALUMINA, SULFUR_C), "moo3_boat": (0.060, ALUMINA, MOO3_C), "chip_boat": (0.035, QBOAT_C, None)}
BOAT_Z = FLOOR_Z + 0.0003  # a boat on the tube's floor strip
TRAY_Z = FLOOR_Z  # the tray's top: level with the tube's floor
TRAY_X = {"chip_boat": XA + 0.050, "moo3_boat": XA + 0.085, "s_boat": XA + 0.120}  # in loading order from the front
MOUTH_OUT_SET = 0.008  # a boat set in the tube's mouth by the demo: its back end this far out of the tube
BACK_Y = -0.300  # the boats' back ends on the tray (the hand on a tongue clear of the rod's pusher)


def _boat(name):
    L, wall, powder = BOATS[name]
    bx = [((0.0, 0.0, BOAT_T / 2), (BOAT_W, L, BOAT_T), wall),
          *[((e * (BOAT_W - BOAT_T) / 2, 0.0, BOAT_H / 2), (BOAT_T, L, BOAT_H), wall) for e in (-1, 1)],
          *[((0.0, e * (L - BOAT_T) / 2, BOAT_H / 2), (BOAT_W - 2 * BOAT_T, BOAT_T, BOAT_H), wall) for e in (-1, 1)],
          ((0.0, -L / 2 - TONGUE_L / 2 + 0.001, TONGUE_Z), (TONGUE_W, TONGUE_L + 0.002, TONGUE_W), wall, (0.0, TONGUE_TURN))]
    if powder:  # a layer of powder in it, 4 mm deep
        bx.append(((0.0, 0.0, BOAT_T + 0.002), (BOAT_W - 2 * BOAT_T - 0.0004, L - 2 * BOAT_T - 0.0004, 0.004), powder))
    return part(name, pos=tuple(boat_start(name)), mass=0.012 if powder else 0.006, friction=0.5, boxes=bx)


def boat_start(name):
    """A boat's frame on the tray."""
    return np.array([TRAY_X[name], BACK_Y + BOATS[name][0] / 2, TOP + TRAY_Z + 0.0003])


BOAT_PARTS = [_boat(n) for n in TRAY_X]
# the tray (fixed): a stainless plate on a post, its back edge short of the boats' (their tongues hang out over it)
_tx0, _tx1 = min(TRAY_X.values()) - 0.015, max(TRAY_X.values()) + 0.015
BOAT_TRAY = part("boat_tray", pos=(0.0, 0.0, TOP), fixed=True, friction=0.6,
                 boxes=[(((_tx0 + _tx1) / 2, BACK_Y + 0.035, TRAY_Z - 0.0015), (_tx1 - _tx0, 0.060, 0.003), STAINLESS),
                        (((_tx0 + _tx1) / 2, BACK_Y + 0.035, (TRAY_Z - 0.003) / 2), (0.014, 0.014, TRAY_Z - 0.003),
                         DARK)])

# -- the chip (loose): 10 x 10 x 0.5 mm SiO2/Si, face up (its oxide shows purple) in the quartz boat ---------------------
CHIP, CHIP_T = 0.010, 0.0005
CHIP_START = boat_start("chip_boat") + [0.0, 0.0, BOAT_T + CHIP_T / 2]
CHIP_PART = part("chip", pos=tuple(CHIP_START), mass=0.0004, friction=0.8,  # (4x a real chip: steadier contact)
                 boxes=[((0.0, 0.0, 0.0), (CHIP, CHIP, CHIP_T), CHIP_C)])

# -- the push rod (loose): fused quartz tube, 8 mm OD, a 12 mm knob fused on its front (the pusher) and a flat ring
# (40 mm across, 5 mm thick) fused on its back end, to hold it by, its face turned 40 degrees from upright (towards the robots);
# on a rack of two V rests beside the tube's line, nearer the robots. Frame: on its axis at the pusher's front face, the
# rod along -y. ----------------------------------------------------------------------------------------------------------
ROD_L, GRIP_W, PUSHER_R = 0.218, 0.008, 0.006
RING_RO, RING_RI, RING_T = 0.020, 0.008, 0.005
RING_Y = -(ROD_L + RING_RO - 0.001)  # the ring's centre, from the frame (on the rod's axis, behind its end)
RING_TURN = math.radians(40.0)
RING_U = np.array([math.cos(RING_TURN), 0.0, -math.sin(RING_TURN)])  # in the ring's plane, square to the rod
RING_N = -np.array([math.sin(RING_TURN), 0.0, math.cos(RING_TURN)])  # the ring's face normal
ROD_Z = BOAT_Z + 0.0075  # the rod's axis in use: its pusher on a boat's tongue, its lower edge 1.5 mm over the tube's floor
RACK_X = XA - 0.060
ROD_START = np.array([RACK_X, TUBE_Y0 - MOUTH_OUT_SET - 0.077, TOP + ROD_Z])  # racked (its pusher clear of the hand
# setting a boat in the mouth)
_RING = [_hex([sx * RING_T / 2 * RING_N + [0.0, RING_Y + r * math.cos(a), 0.0] + r * math.sin(a) * RING_U
               for sx in (-1, 1) for r in (RING_RI, RING_RO) for a in (2 * math.pi * i / 12, 2 * math.pi * (i + 1) / 12)],
              QUARTZ) for i in range(12)]
ROD = part("push_rod", pos=tuple(ROD_START), mass=0.015,  # (a tube: 8 mm OD, 6 mm ID) friction=0.7,
           cylinders=[((0.0, -0.002, 0.0), PUSHER_R, 0.004, QUARTZ, "Y"),
                      ((0.0, -(0.004 + ROD_L) / 2, 0.0), GRIP_W / 2, ROD_L - 0.004, QUARTZ, "Y")],
           meshes=_RING)
REST_Y = (ROD_START[1] - 0.030, ROD_START[1] - 0.150)  # the rack's two Vs, under the rod's body (clear of the ring)
V_RISE = 0.006  # each rest's V: 6 mm deep over 10 mm each side (about 30 degrees), so the rod lies in it, not on it
REST_Z = ROD_Z - 0.0002 - GRIP_W / 2 * math.hypot(1.0, (V_RISE - 0.0002) / 0.010)  # the rod's axis at ROD_Z in the V
REST = part("rod_rack", pos=(0.0, 0.0, TOP), fixed=True, friction=0.3,
            boxes=[((RACK_X, y, REST_Z / 2), (0.020, 0.014, REST_Z), DARK) for y in REST_Y],
            meshes=[_hex([(RACK_X + sx * xo, y + dy, REST_Z + zo) for xo, zt in ((0.0, 0.0002), (0.010, V_RISE))
                          for dy in (-0.007, 0.007) for zo in (0.0, zt)], DARK)
                    for y in REST_Y for sx in (-1, 1)])

OBJECTS = [FURNACE, TUBE, CAPS, *BOAT_PARTS, BOAT_TRAY, CHIP_PART, ROD, REST]

# -- the layout for MoS2 (scored): upstream (the gas inlet's end, the loading end) to downstream ------------------------
S_AT, S_TOL = FY0 + 0.010, 0.015  # the sulfur boat's centre: at the furnace's upstream edge (its end wall; cool)
MO_AT, MO_TOL = HOT_Y, 0.010  # the MoO3 boat's centre: under the thermocouple
CHIP_DMIN, CHIP_DMAX = 0.050, 0.080  # the chip boat's centre: this far downstream of the MoO3 boat's (2.5 to 32.5 mm gap)


# -- scoring ----------------------------------------------------------------------------------------------------------
BOAT_TILT = 20.0  # a boat tipped over 20 degrees (its powder spilled)
BOAT_DROP = 0.030  # or dropped: over 30 mm under the tray or the tube's floor
CHIP_FLIP = 30.0  # the chip tilted over 30 degrees on its boat (flipping); it may slide about inside the boat
RACK_TOL = 0.015
HANDS_FAR = 0.08
SETTLE_STEPS, SETTLE_TOL = 20, 0.001
NAMES = ("s_boat", "moo3_boat", "chip_boat")

_state = {}
_hist = []


@on_reset
def _reset_state(env):
    _state.clear()
    _state.update(step=0, viol=set(), settled=False, far=True, ok={})
    _hist.clear()


def axis_off(state, axis=1):
    """How far a part's own y axis is off the world's y axis (degrees)."""
    a = state.rotation()[:, axis]
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(a[1]))))))


def in_tube(p):
    return TUBE_Y0 < p[1] < TUBE_Y1 and abs(p[0] - XA) < 0.010 and abs(p[2] - (TOP + BOAT_Z)) < 0.004


def layout_ok(bs):
    """Each boat in its place, all on the tube's floor: (sulfur, MoO3, chip) booleans."""
    s, m, c = (np.asarray(bs[n].pos, float) for n in NAMES)
    return (in_tube(s) and abs(s[1] - S_AT) <= S_TOL,
            in_tube(m) and abs(m[1] - MO_AT) <= MO_TOL,
            in_tube(c) and in_tube(m) and CHIP_DMIN <= c[1] - m[1] <= CHIP_DMAX)


def rod_racked(r):
    return float(np.linalg.norm(np.asarray(r.pos, float) - ROD_START)) <= RACK_TOL and axis_off(r) <= 5.0


@on_step
def _watch(env):
    """Latch the violations and follow the end state."""
    S = labkit._SPECS
    st = _state
    st["step"] += 1
    bs = {n: labkit._part_state(env, S[n]) for n in NAMES}
    ch, r = labkit._part_state(env, S["chip"]), labkit._part_state(env, S["push_rod"])
    for n, b in bs.items():
        if b.tilt_deg() > BOAT_TILT or np.asarray(b.pos, float)[2] < TOP + min(TRAY_Z, FLOOR_Z) - BOAT_DROP:
            st["viol"].add("a boat tipped or dropped")
    cb = bs["chip_boat"]
    q = cb.rotation().T @ (np.asarray(ch.pos, float) - np.asarray(cb.pos, float))
    L_c = BOATS["chip_boat"][0]
    out = (abs(q[0]) > BOAT_W / 2 - BOAT_T or abs(q[1]) > L_c / 2 - BOAT_T  # its centre past a wall's inside
           or not 0.0 < q[2] < BOAT_H)  # or off the boat's floor (over its walls, or under it)
    if out or math.degrees(math.acos(max(-1.0, min(1.0, float((cb.rotation().T @ ch.rotation())[2, 2]))))) > CHIP_FLIP:
        st["viol"].add("the chip flipped or left its boat")
    st["ok"] = dict(zip(("s", "mo", "chip"), layout_ok(bs)))
    st["racked"] = rod_racked(r)
    keep = [np.asarray(r.pos, float) + [0.0, -0.10, 0.0], *[np.asarray(b.pos, float) for b in bs.values()]]
    st["far"] = all(np.linalg.norm(np.asarray(tp) - k) >= HANDS_FAR for tp in _tips(env) for k in keep)
    _hist.append(np.concatenate([np.asarray(x.pos, float) for x in (*bs.values(), ch, r)]))
    del _hist[:-(SETTLE_STEPS + 1)]
    st["settled"] = (len(_hist) > SETTLE_STEPS
                     and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 after any violation (a boat tipped over 20 degrees or dropped; the chip tilted over 30 degrees on its boat or
    its centre left the boat's inside), and 0 unless the boats, the chip and the rod have moved under 1 mm over the last second. Then 0.25
    for each boat in its place, on the tube's floor: sulfur centred within 15 mm of the furnace's upstream edge (10 mm in
    from its near end); MoO3 centred within 10 mm of the thermocouple; the chip boat's centre 50 to 80 mm downstream of
    the MoO3 boat's. 1.0 if all three, the rod back on its rack (within 15 mm) and every fingertip 8 cm from the rod and
    the boats."""
    st = _state
    if st.get("viol") or not st.get("settled", False):
        return 0.0
    ok = st.get("ok", {})
    n = sum(bool(v) for v in ok.values())
    if n == 3 and st.get("racked") and st.get("far", True):
        return 1.0
    return 0.25 * n


SUCCESS_HOLD_S = 2.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return float(abs(np.asarray(objects["moo3_boat"].pos, float)[1] - MO_AT))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The right hand loads the boats deepest first: picks each up from the tray by its tongue, sets it in the tube's mouth
# (lined up by eye), then takes the push rod by its ring and pushes the boat to its place (judged by the rod's depth),
# and puts the rod back on its rack. Aims are trimmed only by what a camera sees: the boats outside the tube, the rod.

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


ROD_OPEN, ROD_HOLD = -0.012, 0.0  # across the rod's 5 mm ring, squeezed hard
TONGUE_OPEN, TONGUE_HOLD = -0.012, -0.0015  # on a boat's 5 mm tongue: a light squeeze (a hard one shakes the boat)
ROT_R = np.column_stack([RING_N, np.cross(RING_U, RING_N), RING_U])  # in from the robots' side, tipped 40 deg down,
O_R = Orient(ROT_R)  # closing square across the ring's faces
RING_IN = 0.040  # the open hand comes in onto the ring (and backs off it) this far along its approach
RING_LIFT = 0.012  # the rod lifted this much off its rack (clear of its Vs) to carry it over
ROT_D = axis_angle_matrix(Y, TONGUE_TURN) @ _prev_grip(math.pi / 2)  # in from the front, tipped 55 deg,
O_D = Orient(ROT_D)  # closing square across a boat's tongue
assert abs(math.atan2(-ROT_D[2, 0], ROT_D[0, 0]) - TONGUE_TURN) < 1e-6  # the pads meet the tongue's faces flat
LOAD_ORDER = ("chip_boat", "moo3_boat", "s_boat")  # the deepest first: boats cannot pass each other in the tube
# (upstream is the loading end, where the argon comes in: the chip, furthest downstream, goes in first, sulfur last)
TARGET = {"chip_boat": -0.0075, "moo3_boat": MO_AT, "s_boat": S_AT}
# (the chip boat goes as deep as the rod reaches; the MoO3 boat then pushes it on, by its tongue, to 67.5 mm downstream)
MOUTH_HI = 0.004  # carried in this much above the tube's floor (a held boat droops a little)
MOUTH_OUT = MOUTH_OUT_SET  # a boat set in the mouth: its back end this far out (most of it, and its weight, inside)
CARRY_UP = 0.030  # carried this high over the tray and the tube's end


def _ring_at(pusher_y, x=XA):
    """The rod's ring centre with its pusher at pusher_y (the rod along y at ROD_Z, at x)."""
    return np.array([x, pusher_y + RING_Y, TOP + ROD_Z])


def _rod_seen_at(env, y):
    """Where the rod's axis is (seen) where it crosses y."""
    r = labkit._part_state(env, labkit._SPECS["push_rod"])
    d = r.rotation()[:, 1]
    p = np.asarray(r.pos, float)
    return p + d * ((y - p[1]) / d[1])


def _on_ring(c, back=0.0):
    """TCP for the fingers across the ring centred at c, on the part of it nearest the robots (``back`` short of it)."""
    band = np.asarray(c, float) - (RING_RI + RING_RO) / 2 * RING_U
    return _tcp(band - back * RING_U, ROT_R, past=0.004)


def _tongue(name, centre):
    """Where to hold boat ``name`` by its tongue (3 mm back, clear of its back wall), its underside's centre at centre."""
    L = BOATS[name][0]
    return np.asarray(centre, float) + [0.0, -L / 2 - TONGUE_L / 2 - 0.003, TONGUE_Z]


def demo():
    up = np.array([0.0, 0.0, 1.0])
    a_d = _appr(ROT_D)
    pl = _Plan()
    from yam_arena.constants import LEFT_BASE_POS
    _READY_L = _ik.tcp_pose(np.array(_READY, float), np.array(LEFT_BASE_POS))
    pl.phase(left=[(1.0, _READY_L[0] + 0.06 * up, ROD_OPEN, "straight up", Orient(_READY_L[1])),
                   (2.0, LEFT_PARK, ROD_OPEN, "hand turned at the park", O_T)],
             right=[(2.0, np.array([0.33, -0.27, TOP + 0.33]), TONGUE_OPEN, "hand turned, up clear of the tube's end",
                     O_D)])
    keys = iter(f"r{i}" for i in range(1, 99))
    for bi, name in enumerate(LOAD_ORDER):
        L = BOATS[name][0]
        on_tray = boat_start(name)
        tg = _tongue(name, on_tray)
        mouth = np.array([XA, TUBE_Y0 - MOUTH_OUT + L / 2, TOP + BOAT_Z + 0.0012])
        g_mouth = _tongue(name, mouth)
        hi = np.array([0.0, 0.0, MOUTH_HI])
        pre = g_mouth + hi - [0.0, L - MOUTH_OUT + 0.008, 0.0]  # lined up first: its front 8 mm short of the tube's end
        lift = tg + [0.0, 0.0, CARRY_UP]
        # -- pick the boat up by its tongue; carry it level, over the others and under the rod, to the tube's mouth -----
        k_b = f"b{bi}"
        pl.phase(right=[
            (2.0, _tcp(tg - 0.04 * a_d + 0.05 * up, ROT_D, past=0.006), TONGUE_OPEN, "hand turned over the boat's tongue",
             O_D),
            (0.8, _tcp(tg - 0.04 * a_d, ROT_D, past=0.006), TONGUE_OPEN, "down, level with the tongue", O_D),
            (1.0, _tcp(tg, ROT_D, past=0.006), TONGUE_OPEN, "in round the tongue", O_D),
            (0.6, _tcp(tg, ROT_D, past=0.006), TONGUE_HOLD, "close on the tongue", O_D),
            (1.6, _tcp(lift, ROT_D, past=0.006), TONGUE_HOLD, "lift the boat, level", O_D),
            (3.0, _tcp(np.array([XA + 0.010, pre[1] - 0.020, lift[2]]), ROT_D, past=0.006), TONGUE_HOLD,
             "carry it over to the tube's line", O_D),
            (2.0, _Dyn(_tcp(pre + [0.0, -0.010, 0.0], ROT_D, past=0.006), k_b), TONGUE_HOLD, "down in line with the tube",
             O_D),
            (1.0, _Dyn(_tcp(pre, ROT_D, past=0.006), k_b), TONGUE_HOLD, "to the tube's mouth", O_D)])
        tw = pl.phase(right=[(1.8, _Dyn(_tcp(pre, ROT_D, past=0.006), k_b), TONGUE_HOLD,
                              "hold still: the boat seen in line", O_D)])
        pl.window(tw + 0.3, tw + 1.8, k_b,
                  lambda env, n=name, z=mouth[2] + MOUTH_HI: np.array([XA, 0.0, z]) - _seen(env, n) * [1.0, 0.0, 1.0],
                  name)
        pl.phase(right=[
            (2.0, _Dyn(_tcp(g_mouth + hi, ROT_D, past=0.006), k_b), TONGUE_HOLD, "slide it into the mouth", O_D),
            (0.8, _Dyn(_tcp(g_mouth, ROT_D, past=0.006), k_b), TONGUE_HOLD, "set it down on the tube's floor", O_D),
            (0.6, _Dyn(_tcp(g_mouth, ROT_D, past=0.006), k_b), TONGUE_OPEN, "let go of the boat", O_D),
            (0.8, _Dyn(_tcp(g_mouth - 0.035 * a_d, ROT_D, past=0.006), k_b), TONGUE_OPEN, "back off the boat", O_D),
            (0.8, _Dyn(_tcp(g_mouth - 0.035 * a_d + 0.05 * up, ROT_D, past=0.006), k_b), TONGUE_OPEN, "up and away", O_D)])
        # -- take the rod by its ring off its rack; push the boat to its place in one stroke; pull the rod back out and
        # put it back on its rack ---------------------------------------------------------------------------------------
        tongue_end = TUBE_Y0 - MOUTH_OUT - TONGUE_L
        p_goal = TARGET[name] - L / 2 - TONGUE_L  # the pusher there: the boat centred at its target
        on_rack, k_r = _ring_at(ROD_START[1], RACK_X), next(keys)
        p_line = tongue_end - 0.008  # the pusher, in line, before going in
        in_line = _ring_at(p_line)
        seq = [(2.0, _on_ring(on_rack, RING_IN) + 0.05 * up, ROD_OPEN, "hand turned to the rod's ring, over it", O_R),
               (1.0, _on_ring(on_rack, RING_IN), ROD_OPEN, "down beside the ring", O_R),
               (1.0, _on_ring(on_rack), ROD_OPEN, "in round the ring", O_R),
               (0.6, _on_ring(on_rack), ROD_HOLD, "close on the ring", O_R),
               (1.0, _on_ring(on_rack + RING_LIFT * up), ROD_HOLD, "lift the rod off its rack", O_R),
               (2.0, _on_ring(in_line + RING_LIFT * up), ROD_HOLD, "over to the tube's line", O_R),
               (1.0, _Dyn(_on_ring(in_line), k_r), ROD_HOLD, "down in line with the tube", O_R)]
        pl.phase(right=seq)
        tw = pl.phase(right=[(1.8, _Dyn(_on_ring(in_line), k_r), ROD_HOLD, "hold still: the rod's tip seen in line", O_R)])
        pl.window(tw + 0.3, tw + 1.8, k_r,
                  lambda env: np.array([XA, 0.0, TOP + ROD_Z]) - _rod_seen_at(env, tongue_end - 0.010) * [1.0, 0.0, 1.0],
                  "push_rod")
        touch = _ring_at(tongue_end - 0.003)
        pl.phase(right=[(1.5, _Dyn(_on_ring(touch), k_r), ROD_HOLD, "the pusher up to the boat's tongue", O_R),
                        (max(1.5, 40.0 * (p_goal - tongue_end)), _Dyn(_on_ring(_ring_at(p_goal)), k_r), ROD_HOLD,
                         "push the boat in, slowly", O_R)])
        tw = pl.phase(right=[(2.0, _Dyn(_on_ring(_ring_at(p_goal)), k_r), ROD_HOLD, "hold still: the rod seen at its depth",
                              O_R)])
        pl.window(tw + 0.3, tw + 2.0, k_r,  # (the boat is out of sight in the tube: judged by the rod's depth)
                  lambda env, g=p_goal: np.array([0.0, g - _seen(env, "push_rod")[1], 0.0]), "push_rod")
        pl.phase(right=[(max(1.5, 25.0 * (p_goal - p_line)), _Dyn(_on_ring(in_line), k_r), ROD_HOLD, "pull the rod back out",
                         O_R),
                        (1.0, _on_ring(in_line + RING_LIFT * up), ROD_HOLD, "lift it clear", O_R),
                        (2.0, _on_ring(on_rack + RING_LIFT * up), ROD_HOLD, "over to its rack", O_R),
                        (1.0, _on_ring(on_rack), ROD_HOLD, "down onto its rack", O_R),
                        (0.6, _on_ring(on_rack), ROD_OPEN, "let go of the ring", O_R),
                        (1.0, _on_ring(on_rack, RING_IN), ROD_OPEN, "back off the ring", O_R),
                        (1.0, _on_ring(on_rack, RING_IN) + 0.05 * up, ROD_OPEN, "up, clear of the rod", O_R)])
    pl.phase(right=[(2.0, RIGHT_PARK, ROD_OPEN, "up and clear", O_T)])
    pl.phase(right=[(3.0, RIGHT_PARK, ROD_OPEN, "rest", O_T)], left=[(3.0, LEFT_PARK, ROD_OPEN, "rest", O_T)])
    return {"left": pl.L, "right": pl.R}
