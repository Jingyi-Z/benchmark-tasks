"""Laying out a 2 inch tube furnace for two-precursor thermal CVD (as for a 2D semiconductor), after a lab's procedure.

A split-tube furnace stands in front of the robots, its lid open, a loose quartz tube across it, centred (120 mm out at
each end). For loading, the left hand slides the tube 75 mm towards the far end and holds it, and slides it back after.
The tube's ends are closed with tapered one-hole silicone stoppers carrying a short stainless gas tube, as many research
labs do (both ends stand 120 mm out of the furnace, away from the heat): the exhaust end's is in place; the argon inlet
end's lies in a cradle beside the tube. After loading, the left hand holds the tube still while the right pushes the
inlet stopper into the tube's bore until its taper bears all round on the tube's inner edge. On a tray in front of the loading end, under the push rod's line: an alumina boat of precursor 1 (a yellow powder),
an alumina boat of precursor 2 (a white powder) and a quartz boat with a 1 cm silicon substrate, face up. Precursor 1
goes upstream at the furnace's edge (cool), precursor 2 under the thermocouple (the hot zone's centre) and the
substrate a few cm downstream of it. Upstream is where the gas comes in, here the loading end, so the substrate goes in
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
TUBE_Y0, TUBE_Y1 = -0.300, 0.310  # the tube's ends, centred in the furnace (120 mm out at each end): the loading end
# (towards the right arm) and the far end. The tube is loose in its saddles: for loading, the left hand slides it 75 mm
# towards the far end (the loading end then 45 mm out), holds it while the boats go in, and slides it back.
SHIFT = 0.075
LOAD_Y0, LOAD_Y1 = TUBE_Y0 + SHIFT, TUBE_Y1 + SHIFT  # the tube's ends while it is slid over for loading
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
    ((((XA + SLOT + 0.001) + FX1) / 2 - 0.02, HOT_Y, AXZ - 0.004), ((FX1 - 0.04) - (XA + SLOT + 0.001), 0.003, 0.003),
     TC_C),
]
FURNACE_CYLS = [((XA + SLOT + 0.001, HOT_Y, AXZ - 0.004), 0.0028, 0.004, TC_C, "X")]  # the junction bead
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
FURNACE_MESHES = [*SADDLES, _boxes_mesh(_coils, COIL_C)]  # (its controller is off the bench: not modelled)
# the lid, standing open (held by its stop): front and back blocks either side of the tube's gap, the top over it
_lid_boxes = [
    (((FX0 + XA - SLOT) / 2, _mid_y, (AXZ + LID_Z1) / 2), (XA - SLOT - FX0, _len_y, LID_Z1 - AXZ), WHITE_F),
    (((XA + SLOT + FX1) / 2, _mid_y, (AXZ + LID_Z1) / 2), (FX1 - XA - SLOT, _len_y, LID_Z1 - AXZ), WHITE_F),
    ((XA, _mid_y, (AXZ + 0.040 + LID_Z1) / 2), (2 * SLOT, _len_y, LID_Z1 - AXZ - 0.040), WHITE_F),
    # its handle, on the lid's front (on top, with the lid open): a 120 mm steel bar on two 25 mm standoffs
    *[((FX0 - 0.0125, _mid_y + sy * 0.050, AXZ + 0.70 * (LID_Z1 - AXZ)), (0.025, 0.010, 0.010), STAINLESS)
      for sy in (-1, 1)],
    ((FX0 - 0.030, _mid_y, AXZ + 0.70 * (LID_Z1 - AXZ)), (0.010, 0.120, 0.012), STAINLESS),
]
LID_OPEN = math.radians(100.0)  # opened just past upright, leaning back off the left hand, resting on its stop


def _open_box(c, size, colour):
    """A lid box (given with the lid closed) swung open about the hinge: fixed, as the lid rests on its stop."""
    ca, sa = math.cos(LID_OPEN), math.sin(LID_OPEN)
    pts = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                x, y, z = (np.asarray(c, float) + (np.array([i, j, k]) - 0.5) * np.asarray(size, float))
                dx, dz = x - HINGE[0], z - HINGE[1]
                pts.append((HINGE[0] + dx * ca + dz * sa, y, HINGE[1] - dx * sa + dz * ca))
    return _hex(pts, colour)


FURNACE = part("tube_furnace", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5, boxes=FURNACE_BOXES,
               cylinders=FURNACE_CYLS, meshes=[*FURNACE_MESHES, *(_open_box(*b) for b in _lid_boxes)])

# -- the tube (loose, in the furnace's saddles and on two supports just outside it): fused quartz, 2 inch, 610 mm long,
# 0.49 kg, its inlet end open for loading. Its wall in 45 mm lengths, 12-sided, and a smooth strip
# along its floor that the boats slide on. Frame: on its axis, at its middle. ---------------------------------------------
FLOOR_W = 0.020
FLOOR_Z = AXZ - math.sqrt(TUBE_IN ** 2 - (FLOOR_W / 2) ** 2) + 0.0020  # the strip's top: as wide as a boat, and 2 mm
# over the wall at its edges (a boat slides on it alone, never catching on the wall's facets)
TUBE_C = np.array([XA, (TUBE_Y0 + TUBE_Y1) / 2, TOP + AXZ])  # the tube's frame at the start (centred)
_HL = (TUBE_Y1 - TUBE_Y0) / 2
_CUTS = list(np.arange(-_HL, _HL, 0.045)) + [_HL]
TUBE_MESHES = [m for k, (y0, y1) in enumerate(zip(_CUTS[:-1], _CUTS[1:]))
               for m in _ring_y(y0, y1, TUBE_IN, TUBE_R, 32 if k in (0, len(_CUTS) - 2) else 12, QUARTZ)]
# (its two end lengths 32-sided, so a round stopper seats in them as in the round tube)
STRIP_IN = 0.015  # the floor strip starts this far inside the loading end (clear of the seated stopper, at most
# 12.8 mm in); a boat's bevelled underside edges ride up its 2 mm step
TUBE_BOXES = [((0.0, STRIP_IN / 2, FLOOR_Z - AXZ - 0.0015), (FLOOR_W, 2 * _HL - STRIP_IN, 0.003), QUARTZ)]
# -- the stoppers: tapered one-hole silicone stoppers, size 10 (Chemglass CG-1420-05: #10, 9/16 in hole; a #10 is 42 mm
# across at its small end, 50 mm at its large end, 25 mm long), each with a 5/8 in (15.9 mm) stainless gas tube through
# its hole, 60 mm long, 1 mm wall. Frame: the centre of its small end; its large end runs back towards -y.
STOP_R0, STOP_R1, STOP_L = 0.021, 0.025, 0.025  # small end, large end (radii), length
STUB_R, STUB_IN, STUB_OUT = 0.0079, 0.005, 0.030  # the gas tube: radius, how far it stands in past the small end, out
# past the large end
STOP_EDGE, STOP_RE = 0.003, 0.0195  # its small end's moulded edge: rounded off over 3 mm, down to 39 mm across
SEAT = (TUBE_IN * math.cos(math.pi / 32) - STOP_R0) / (STOP_R1 - STOP_R0) * STOP_L  # the small end this far inside the
# tube when the taper bears on the tube's 32-sided inner edge (about 12 mm)
SILICONE = (0.80, 0.78, 0.74)


def _stopper_shapes(sd, y0):
    """A stopper's taper and gas tube as (meshes, cylinders), its small end at y0, its large end towards sd * y."""
    m = []  # (a solid taper: 32 wedges from close to the axis out, each its own hull)
    for i in range(32):
        aa = (2 * math.pi * i / 32, 2 * math.pi * (i + 1) / 32)
        r_e = STOP_R0 + STOP_EDGE / STOP_L * (STOP_R1 - STOP_R0)  # the taper where the rounded edge ends
        for (ya, ra), (yb, rb) in (((y0, STOP_RE), (y0 + sd * STOP_EDGE, r_e)),
                                   ((y0 + sd * STOP_EDGE, r_e), (y0 + sd * STOP_L, STOP_R1))):
            pts = [(r * math.cos(t), y, r * math.sin(t)) for y, ro in ((ya, ra), (yb, rb)) for r in (0.001, ro)
                   for t in aa]
            m.append(_hex(pts, SILICONE))
    # the gas tube: a stainless tube, 1 mm wall (hollow, so its weight is small), as 16 hull segments
    m += _ring_y(*sorted((y0 - sd * STUB_IN, y0 + sd * (STOP_L + STUB_OUT))), STUB_R - 0.001, STUB_R, 16, STAINLESS)
    return m, []


_ex_m, _ex_c = _stopper_shapes(1, _HL - SEAT)  # the exhaust stopper, in place
TUBE_MESHES += _ex_m
TUBE = part("quartz_tube", pos=tuple(TUBE_C), mass=0.49 + 0.08, friction=0.3, boxes=TUBE_BOXES, meshes=TUBE_MESHES,
            cylinders=_ex_c)  # (the tube 0.49 kg, its exhaust stopper with its gas tube 0.08 kg)
# its two supports (fixed): a block under the tube just outside each end of the furnace
TUBE_STAND = part("tube_stands", pos=(0.0, 0.0, TOP), fixed=True, friction=0.3,
                  boxes=[((XA, y, (AXZ - TUBE_R) / 2), (0.030, 0.012, AXZ - TUBE_R), DARK) for y in (FY0 - 0.012,
                                                                                                    FY1 + 0.012)])

# -- the inlet stopper (loose), out for loading, lying in a cradle beside the loading end (beyond the boat tray), its
# small end towards the tube. The hand carries it by its gas tube. -----------------------------------------------------
BRASS = (0.78, 0.66, 0.30)
CAP_TURN = math.radians(40.0)
CAP_U = np.array([math.cos(CAP_TURN), 0.0, -math.sin(CAP_TURN)])  # square to the stopper's axis, tipped 40 deg down
CAP_N = -np.array([math.sin(CAP_TURN), 0.0, math.cos(CAP_TURN)])  # the hand's fingers close along this
CRADLE_X = {"inlet_cap": XA + 0.200}
CAP_HOME = {"inlet_cap": TUBE_Y0 + SEAT}  # its small end, seated in the tube's loading end
CRADLE_OUT = {"inlet_cap": 0.012}  # its small end this far out beyond the tube's loading end (slid over for loading)


def cap_start(name):
    """The stopper's frame in its cradle, beside the tube's line, out beyond the loading end."""
    return np.array([CRADLE_X[name], LOAD_Y0 - CRADLE_OUT[name], TOP + AXZ])


def _cap_part(name):
    m, c = _stopper_shapes(-1, 0.0)
    return part(name, pos=tuple(cap_start(name)), mass=0.08, friction=0.9, meshes=m, cylinders=c)  # (silicone grips)


CAP_PARTS = [_cap_part("inlet_cap")]
# the cradle: three pairs of short rails, 14 mm either side of its axis, along the taper, each as high as the
# taper at its inner edge (11 mm off the axis, where the stopper's round side is lowest), so it lies level
_RAILS = [(STOP_R0 + (f * STOP_L + 0.002) / STOP_L * (STOP_R1 - STOP_R0), -f * STOP_L) for f in (0.3, 0.6, 0.9)]
# (each rail as high as the taper at its wider edge, so none pokes into it)
CRADLES = part("cap_cradles", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5,
               boxes=[((cx + e * 0.014, cap_start(n)[1] + yr, (AXZ - math.sqrt(r ** 2 - 0.011 ** 2) - 0.0003) / 2),
                       (0.006, 0.004, AXZ - math.sqrt(r ** 2 - 0.011 ** 2) - 0.0003), DARK)
                      for n, cx in CRADLE_X.items() for e in (-1, 1) for r, yr in _RAILS])
# -- the boats (loose), side by side on a tray beside the loading end (behind the tube's line), level with the tube's
# floor: an alumina boat of precursor 1, an alumina boat of precursor 2, and a short quartz boat carrying the 1 cm substrate,
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
    bev = 0.004  # its underside's end edges bevelled up 2 mm over 4 mm, as on a moulded boat's rounded ends: a
    # dragged boat rides up the tube floor strip's step instead of catching on it
    bx = [((0.0, 0.0, BOAT_T / 2), (BOAT_W, L - 2 * bev, BOAT_T), wall),
          *[((e * (BOAT_W - BOAT_T) / 2, 0.0, (BOAT_H + BOAT_T) / 2), (BOAT_T, L, BOAT_H - BOAT_T), wall)
            for e in (-1, 1)],
          *[((0.0, e * (L - BOAT_T) / 2, (BOAT_H + BOAT_T) / 2), (BOAT_W - 2 * BOAT_T, BOAT_T, BOAT_H - BOAT_T), wall)
            for e in (-1, 1)],
          ((0.0, -L / 2 - TONGUE_L / 2 + 0.001, TONGUE_Z), (TONGUE_W, TONGUE_L + 0.002, TONGUE_W), wall, (0.0, TONGUE_TURN))]
    bm = [_hex([(x, e * y, z) for x in (-BOAT_W / 2, BOAT_W / 2) for y, z0 in ((e * (L / 2 - bev), 0.0),
                (e * L / 2, BOAT_T - 0.0008)) for z in (z0, BOAT_T)], wall) for e in (-1, 1)]
    if powder:  # a layer of powder in it, 4 mm deep
        bx.append(((0.0, 0.0, BOAT_T + 0.002), (BOAT_W - 2 * BOAT_T - 0.0004, L - 2 * BOAT_T - 0.0004, 0.004), powder))
    return part(name, pos=tuple(boat_start(name)), mass=0.012 if powder else 0.006, friction=0.5, boxes=bx,
                meshes=bm)


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

# -- the substrate (loose): a 10 x 10 x 0.5 mm silicon chip, oxidised face up (purple) in the quartz boat ---------------------
CHIP, CHIP_T = 0.010, 0.0005
CHIP_START = boat_start("chip_boat") + [0.0, 0.0, BOAT_T + CHIP_T / 2]
CHIP_PART = part("chip", pos=tuple(CHIP_START), mass=0.0004, friction=0.8,  # (4x a real chip: steadier contact)
                 boxes=[((0.0, 0.0, 0.0), (CHIP, CHIP, CHIP_T), CHIP_C)])

# -- the push rod (loose): fused quartz tube, 8 mm OD, a flat paddle fused on its front (the pusher) and a flat T-handle
# fused across its back end, to hold it by: a 5 mm plate square to the rod, 51 x 26 mm, rising from the rod 45 mm up and
# back towards the robots (40 degrees from flat). The hand takes it across its two faces, so a push goes straight into
# the pads and the rod's weight cannot turn it in the grip;
# on a rack of two V rests beside the tube's line, nearer the robots. Frame: on its axis at the pusher's front face, the
# rod along -y. ----------------------------------------------------------------------------------------------------------
ROD_L, GRIP_W, PUSHER_R = 0.293, 0.008, 0.006
RING_T = 0.005  # the T-handle's plate: 5 mm thick
TH_UP, TH_W = 0.045, 0.026  # the handle: a 5 mm plate square to the rod, 26 mm wide, standing 45 mm up over the rod's end
RING_Y = -(ROD_L + RING_T / 2)  # the plate's mid-plane, from the frame (behind the rod's end)
RING_TURN = math.radians(40.0)
RING_U = np.array([math.cos(RING_TURN), 0.0, -math.sin(RING_TURN)])  # down the handle's slope (the hand's approach)
RING_N = -np.array([math.sin(RING_TURN), 0.0, math.cos(RING_TURN)])  # across the handle, square to the slope
ROD_Z = BOAT_Z + 0.0075  # the rod's axis in use: its pusher on a boat's tongue, its lower edge 1.5 mm over the tube's floor
RACK_X = XA  # the rod lies in line with the tube, behind its loading end, on two low rests: it is slid in and out along
# them, never lifted
ROD_START = np.array([RACK_X, TUBE_Y0 - 0.100, TOP + ROD_Z])  # its pusher 100 mm behind the centred tube's loading end
# (clear of the inlet cap as it is lined up)
# (clear of a boat set in the mouth, and of the inlet cap)
_RING = [_hex([(sx * TH_W / 2, RING_Y + dy, z) for sx in (-1, 1) for dy in (-RING_T / 2, RING_T / 2)  # (all above
               for z in (0.002, 0.002 + TH_UP)], QUARTZ)]  # the rod's axis, and centred over it: its weight cannot turn it)
# the pusher: a flat quartz paddle 4 mm thick, 9 to 11.5 mm tall, 14 mm wide at its foot and 20 mm at its top (it keeps a
# boat's tongue on its face when the rod's tip wanders sideways, and clears the tube's wall)
_PADDLE = [_hex([(sx * (0.007 if z < 0 else 0.010), y, (z if z > 0 else (-0.003 if y == 0.0 else -0.0055)))
                 for y in (0.0, -0.004) for z in (-0.006, 0.006) for sx in (-1, 1)], QUARTZ)]
# (its foot bevelled up at the front, 4.5 mm over the tube's floor there: if the rod's tip sags as it goes in, it rides
# onto the floor at the tube's mouth instead of stubbing on its end)


ROD = part("push_rod", pos=tuple(ROD_START), mass=0.015, friction=0.3,  # (a tube: 8 mm OD, 6 mm ID)
           boxes=[((0.0, -(0.004 + ROD_L) / 2, 0.0), (GRIP_W, ROD_L - 0.004, GRIP_W), QUARTZ)],  # (square: it cannot
           meshes=[*_RING, *_PADDLE])  # roll in its channel under the handle's offset weight)
REST_Y = (ROD_START[1] - 0.030, ROD_START[1] - 0.250)  # its two rests, in line with the tube, under the parked rod's body
CH_W, CH_WALL, CH_UP = 0.014, 0.003, 0.0015  # each rest: a flat channel 14 mm wide, its walls 1.5 mm up the rod's sides (all
REST_Z = ROD_Z - GRIP_W / 2 - 0.0002  # under the rod's axis, so the handle slides over them); its floor under the rod
REST = part("rod_rack", pos=(0.0, 0.0, TOP), fixed=True, friction=0.3,
            boxes=[b for y in REST_Y for b in (
                ((RACK_X, y, REST_Z / 2), (CH_W + 2 * CH_WALL, 0.014, REST_Z), DARK),
                *[((RACK_X + sx * (CH_W + CH_WALL) / 2, y, REST_Z + CH_UP / 2), (CH_WALL, 0.014, CH_UP), DARK)
                  for sx in (-1, 1)])])

OBJECTS = [FURNACE, TUBE, TUBE_STAND, *CAP_PARTS, CRADLES, *BOAT_PARTS, BOAT_TRAY, CHIP_PART, ROD, REST]

# -- the layout (scored): upstream (the gas inlet's end, the loading end) to downstream ------------------------
S_AT, S_TOL = FY0 + 0.010, 0.015  # precursor 1's boat's centre: at the furnace's upstream edge (cool)
MO_AT, MO_TOL = HOT_Y, 0.010  # precursor 2's boat's centre: under the thermocouple
CHIP_DMIN, CHIP_DMAX = 0.050, 0.080  # the substrate's boat's centre: this far downstream of precursor 2's (2.5 to 32.5 mm gap)


# -- scoring ----------------------------------------------------------------------------------------------------------
BOAT_TILT = 20.0  # a boat tipped over 20 degrees (its powder spilled)
BOAT_DROP = 0.030  # or dropped: over 30 mm under the tray or the tube's floor
CHIP_FLIP = 30.0  # the substrate tilted over 30 degrees on its boat (flipping); it may slide about inside the boat
RACK_TOL = 0.015
NAMES = ("s_boat", "moo3_boat", "chip_boat")
CAP_NAMES = ("inlet_cap",)
CAP_TOL, CAP_TILT, CAP_OFF = 0.001, 2.0, 0.001  # seated: its small end within 1 mm of its seating depth (the taper
# bearing all round on the tube's inner edge), its axis within 1 mm of the tube's at the tube's end and 2 degrees

_state = {}
_hist = []
_DEBUG = bool(__import__("os").environ.get("TF_DEBUG"))  # a trace of the scored state every 2 s (no effect on it)


@on_reset
def _reset_state(env):
    _state.clear()
    _state.update(step=0, viol=set(), settled=False, far=False, clear=0.0, travel=9.0, ok={})
    _hist.clear()


def axis_off(state, axis=1):
    """How far a part's own y axis is off the world's y axis (degrees)."""
    a = state.rotation()[:, axis]
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(a[1]))))))


TUBE_TOL, TUBE_TILT = 0.005, 1.0  # the tube back in place: within 5 mm along its axis of where it started (its ends 120 mm out of the
# furnace each), its axis within 1 degree of it
TUBE_OFF, TUBE_KNOCK = 0.010, 3.0  # knocked: over 10 mm up, down or sideways, or tilted over 3 degrees, at any time


def in_tube(p, tp):
    """On the tube's floor, between its ends (the tube's frame at tp)."""
    d = np.asarray(p, float) - np.asarray(tp, float)
    return (-_HL < d[1] < _HL and abs(d[0]) < 0.010 and abs(d[2] - (BOAT_Z - AXZ)) < 0.004)


def tube_home(t):
    p = np.asarray(t.pos, float)
    return abs(p[1] - TUBE_C[1]) <= TUBE_TOL and axis_off(t) <= TUBE_TILT


def layout_ok(bs, tp):
    """Each boat in its place (against the furnace), all on the tube's floor: (precursor 1, precursor 2, substrate)."""
    s, m, c = (np.asarray(bs[n].pos, float) for n in NAMES)
    return (in_tube(s, tp) and abs(s[1] - S_AT) <= S_TOL,
            in_tube(m, tp) and abs(m[1] - MO_AT) <= MO_TOL,
            in_tube(c, tp) and in_tube(m, tp) and CHIP_DMIN <= c[1] - m[1] <= CHIP_DMAX)


def cap_seated(c, name, tp, ty=Y):
    """The inlet stopper seated in the tube's loading end (the tube's frame at tp, its axis ty): its small end within
    1 mm of its seating depth, where its taper bears all round on the tube's inner edge; its axis within 2 degrees of
    the tube's and within 1 mm of it at the tube's end."""
    p, tp, ty = np.asarray(c.pos, float), np.asarray(tp, float), np.asarray(ty, float)
    a = c.rotation()[:, 1]
    a = a if np.dot(a, ty) > 0 else -a
    end = tp - _HL * ty
    depth = float(np.dot(p - end, ty))
    at_end = p - a * (depth / max(1e-6, float(np.dot(a, ty))))
    d = at_end - tp
    off = float(np.linalg.norm(d - np.dot(d, ty) * ty))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, ty))))))  # against the tube's own axis
    return abs(depth - SEAT) <= CAP_TOL and off <= CAP_OFF and tilt <= CAP_TILT


def rod_racked(r):
    return float(np.linalg.norm(np.asarray(r.pos, float) - ROD_START)) <= RACK_TOL and axis_off(r) <= 5.0


# -- released hands and stillness (every positive score) --------------------------------------------------------------
RELEASE = 0.080  # every jaw point and pad midpoint of both hands at least this far from every task surface
STILL_N, STILL_TOL = 21, 0.001  # and no material point of a loose part moved 1 mm between any two of the last 21 samples
_EX_END = _HL - SEAT + STOP_L + STUB_OUT  # the exhaust stopper's gas tube's outer end, from the tube's frame


def _fixed_boxes():
    """World axis-aligned boxes (centre, half sizes) bounding every fixed task part: the furnace's body, chamber
    blocks, saddles, thermocouple, controller and open lid; the tube's supports, the stopper's cradle, the boat tray
    and the rod's rests."""
    o = np.array([0.0, 0.0, TOP])
    out = [(o + np.asarray(b[0], float), np.asarray(b[1], float) / 2) for b in FURNACE_BOXES]
    out.append((o + np.array([XA, _mid_y, (BASE_Z + AXZ) / 2]),  # the saddles and coils in the chamber's gap
                np.array([TUBE_R + 0.008, _len_y / 2, (AXZ - BASE_Z) / 2])))
    ca, sa = math.cos(LID_OPEN), math.sin(LID_OPEN)
    for c, size, *_ in _lid_boxes:  # the open lid: each block's swung corners
        pts = []
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    x, y, z = np.asarray(c, float) + (np.array([i, j, k]) - 0.5) * np.asarray(size, float)
                    dx, dz = x - HINGE[0], z - HINGE[1]
                    pts.append((HINGE[0] + dx * ca + dz * sa, y, HINGE[1] - dx * sa + dz * ca))
        pts = np.array(pts)
        out.append((o + (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)) / 2))
    hi = CON_C + CON_H / 2 * CON_U + 0.002 * CON_N  # the controller's wedge and display
    lo = np.array([DESK_X0, _mid_y - CON_W / 2, DESK])
    hi = np.array([FX0 + 0.005, _mid_y + CON_W / 2, hi[2]])
    out.append((o + (hi + lo) / 2, (hi - lo) / 2))
    for p in (TUBE_STAND, CRADLES, BOAT_TRAY, REST):
        for b in p.boxes:
            out.append((o + np.asarray(b[0], float), np.asarray(b[1], float) / 2))
    return out


FIXED_BOXES = _fixed_boxes()
# Loose parts as capsules (segment ends in the part's own frame, radius) that contain them: the tube with its exhaust
# stopper and gas tube; the inlet stopper with its gas tube; each boat with its tongue; the substrate; the rod's bar
# and paddle, and its T-handle.
CAPSULES = {
    "quartz_tube": [((0.0, -_HL, 0.0), (0.0, _EX_END, 0.0), TUBE_R)],
    "inlet_cap": [((0.0, STUB_IN, 0.0), (0.0, -(STOP_L + STUB_OUT), 0.0), STOP_R1)],
    **{n: [((0.0, -BOATS[n][0] / 2 - TONGUE_L, BOAT_H / 2), (0.0, BOATS[n][0] / 2, BOAT_H / 2), 0.016)] for n in NAMES},
    "chip": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.008)],
    "push_rod": [((0.0, 0.0, 0.0), (0.0, -(ROD_L + RING_T), 0.0), 0.012),
                 ((0.0, RING_Y, 0.002), (0.0, RING_Y, 0.002 + TH_UP), 0.014)],
}
# Material points spanning each loose part's extremities (its own frame), for the stillness gate: they catch turns as
# well as slides.
MAT_POINTS = {
    "quartz_tube": [(0.0, -_HL, 0.0), (TUBE_R, -_HL, 0.0), (0.0, -_HL, TUBE_R), (0.0, _HL, 0.0), (TUBE_R, _HL, 0.0),
                    (0.0, _HL, TUBE_R), (0.0, _EX_END, 0.0)],
    "inlet_cap": [(0.0, STUB_IN, 0.0), (0.0, -(STOP_L + STUB_OUT), 0.0), (STOP_R1, -STOP_L, 0.0),
                  (0.0, -STOP_L, STOP_R1), (-STOP_R1, -STOP_L, 0.0)],
    **{n: [(sx * BOAT_W / 2, sy * BOATS[n][0] / 2, BOAT_H) for sx in (-1, 1) for sy in (-1, 1)]
       + [(0.0, -BOATS[n][0] / 2 - TONGUE_L, TONGUE_Z)] for n in NAMES},
    "chip": [(-CHIP / 2, -CHIP / 2, 0.0), (CHIP / 2, CHIP / 2, 0.0), (CHIP / 2, -CHIP / 2, 0.0)],
    "push_rod": [(0.0, 0.0, 0.0), (-0.010, 0.0, 0.006), (0.010, 0.0, 0.006), (0.0, -ROD_L, 0.0),
                 (-TH_W / 2, RING_Y, 0.002 + TH_UP), (TH_W / 2, RING_Y, 0.002 + TH_UP)],
}


def _seg_dist(p, a, b):
    ab = b - a
    t = 0.0 if float(ab @ ab) < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / float(ab @ ab)))
    return float(np.linalg.norm(p - (a + t * ab)))


def clearance(parts, tips) -> float:
    """The least distance (m) from any of the hands' points to any task surface: the fixed parts' boxes and the loose
    parts' capsules, where they now are (parts: name -> state with .pos and .rotation())."""
    best = 9.0
    for p in tips:
        p = np.asarray(p, float)
        for c, h in FIXED_BOXES:
            best = min(best, float(np.linalg.norm(np.maximum(np.abs(p - c) - h, 0.0))))
        for n, caps in CAPSULES.items():
            s = parts[n]
            R, o = s.rotation(), np.asarray(s.pos, float)
            for a, b, r in caps:
                best = min(best, _seg_dist(p, o + R @ np.asarray(a, float), o + R @ np.asarray(b, float)) - r)
    return best


def material_points(parts) -> np.ndarray:
    out = []
    for n, pts in MAT_POINTS.items():
        s = parts[n]
        R, o = s.rotation(), np.asarray(s.pos, float)
        out += [o + R @ np.asarray(q, float) for q in pts]
    return np.array(out)


def max_travel(hist) -> float:
    """The largest distance any one tracked point moved between any two of the recorded samples (m)."""
    h = np.asarray(hist)  # samples x points x 3
    return float(np.linalg.norm(h[:, None, :, :] - h[None, :, :, :], axis=3).max())


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
        st["viol"].add("the substrate flipped or left its boat")
    t = labkit._part_state(env, S["quartz_tube"])
    tp = np.asarray(t.pos, float)
    if (math.hypot(tp[0] - TUBE_C[0], tp[2] - TUBE_C[2]) > TUBE_OFF or axis_off(t) > TUBE_KNOCK):
        st["viol"].add("the tube was knocked off its saddles")
    st["tube"] = tube_home(t)
    st["ok"] = dict(zip(("s", "mo", "chip"), layout_ok(bs, tp)))
    st["racked"] = rod_racked(r)
    cs = {n: labkit._part_state(env, S[n]) for n in CAP_NAMES}
    st["caps"] = {n: cap_seated(c, n, tp, t.rotation()[:, 1]) for n, c in cs.items()}
    parts = {**bs, **cs, "chip": ch, "push_rod": r, "quartz_tube": t}
    st["clear"] = clearance(parts, _tips(env))
    st["far"] = st["clear"] >= RELEASE
    _hist.append(material_points(parts))
    del _hist[:-STILL_N]
    st["travel"] = max_travel(_hist) if len(_hist) == STILL_N else 9.0
    st["settled"] = st["travel"] < STILL_TOL
    if _DEBUG and st["step"] % 40 == 0:
        c = cs["inlet_cap"]
        ty = t.rotation()[:, 1]
        depth = float(np.dot(np.asarray(c.pos, float) - (tp - _HL * ty), ty))
        print("TF_DBG", {"step": st["step"], "viol": sorted(st["viol"]), "ok": {k: bool(v) for k, v in st["ok"].items()},
                         "seated": bool(st["caps"]["inlet_cap"]), "depth_mm": round(depth * 1000, 2),
                         "seat_mm": round(SEAT * 1000, 2), "tilt": round(axis_off(c), 2),
                         "xz_mm": np.round((np.asarray(c.pos, float) - tp)[[0, 2]] * 1000, 2).tolist(),
                         "tube_mm": np.round((tp - TUBE_C) * 1000, 2).tolist(), "racked": bool(st["racked"]),
                         "far": bool(st["far"]), "clear_mm": round(st["clear"] * 1000, 1),
                         "travel_mm": round(st["travel"] * 1000, 3), "settled": bool(st["settled"])},
              flush=True)


def score(objects):
    """0 after any violation (a boat tipped over 20 degrees or dropped; the substrate tilted over 30 degrees on its boat or
    its centre left the boat's inside; the tube knocked over 10 mm up, down or sideways or tilted over 3 degrees). Every
    positive score also needs both hands released (each hand's pad midpoint and both jaw points 80 mm or more from
    every task surface: the furnace, supports, cradle, tray and rests, and the tube, its stoppers and gas tubes, the
    boats, the substrate and the whole rod with its handle) and everything loose still (no point on the tube, the
    stoppers, the boats, the substrate or the rod moved 1 mm between any two of the last 21 samples, 1 s at 20 Hz).
    Then 0.2 for each boat in its place, on the tube's floor: precursor 1 centred within 15 mm of the furnace's upstream
    edge (10 mm in from its near end); precursor 2 centred within 10 mm of the thermocouple; the substrate's boat's
    centre 50 to 80 mm downstream of precursor 2's. 0.8 if all three are placed and the inlet stopper is seated (its
    small end within 1 mm of its rigid seating depth, 11.81 mm in, its axis within 2 degrees and 1 mm of the tube's):
    the seal counts only on top of full boat credit. 1.0 if also the rod is back on its rack (within 15 mm and 5
    degrees) and the tube back where it started (within 5 mm along its axis and 1 degree).
    """
    st = _state
    if st.get("viol") or not st.get("settled", False) or not st.get("far", False):
        return 0.0
    n = sum(bool(v) for v in st.get("ok", {}).values())
    c = all(st.get("caps", {}).values()) and bool(st.get("caps"))
    if n == 3 and c:
        if st.get("tube") and st.get("racked"):
            return 1.0
        return 0.8
    return round(0.2 * n, 3)


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
    """Both hands' pad midpoints (the runtime's own pad datum) and each hand's two jaw points, from its two finger
    joints separately (world, env frame): six points."""
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        q7, q8 = (abs(float(v)) for v in robot.data.joint_pos.torch[0, ids].cpu().numpy())
        out += [mid, mid - q7 * g[:3, 0], mid + q8 * g[:3, 0]]
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
_ct = {"trim": {}, "wins": [], "last": {}, "gates": [], "gate": {}, "rtrim": {}, "rwins": [], "rlast": {}}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}
    _ct["gate"] = {}
    _ct["rtrim"] = {}
    _ct["rlast"] = {}


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


def _tube_end_seen(env):
    """The centre of the tube's loading end, where it is seen."""
    t = labkit._part_state(env, labkit._SPECS["quartz_tube"])
    return np.asarray(t.pos, float) - _HL * t.rotation()[:, 1]


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
        _ct["trim"][key] = np.clip(tr + np.clip(0.2 * e, -0.0003, 0.0003), -0.020, 0.020)


@on_step
def _rtrack(env):
    """The live turn trims: as the position trims, a turn of the hand (a rotation vector, rad) from what is seen."""
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, fn, watch in _ct["rwins"]:
        if not t0 <= t < t1:
            continue
        p = _seen(env, watch)
        last = _ct["rlast"].get(key)
        _ct["rlast"][key] = p
        if last is None or np.max(np.abs(p - last)) > 0.0005:
            continue
        e = np.asarray(fn(env), float)
        tr = _ct["rtrim"].get(key, np.zeros(3))
        _ct["rtrim"][key] = np.clip(tr + np.clip(0.2 * e, -0.002, 0.002), -0.06, 0.06)


def _rotvec(v):
    a = float(np.linalg.norm(v))
    if a < 1e-9:
        return np.eye(3)
    k = np.asarray(v, float) / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


class _DynO(float):
    """A Waypoint ``yaw`` whose TCP rotation is ``base`` turned by the live turn trim ``key``."""

    def __new__(cls, base, key):
        obj = float.__new__(cls, 0.0)
        obj.base, obj.key = np.asarray(getattr(base, "rot", base), float), key
        return obj

    @property
    def rot(self):
        return _rotvec(_ct["rtrim"].get(self.key, np.zeros(3))) @ self.base


class _Dyn:
    """A TCP plus the live trim ``key``."""

    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)




LEFT_PARK = (0.12, 0.36, TOP + 0.20)
RIGHT_PARK = (0.14, -0.30, TOP + 0.24)
# where both hands rest at the end: over 100 mm from every task surface (the release gate needs 80 mm)
LEFT_REST = (0.08, 0.32, TOP + 0.22)
RIGHT_REST = (0.10, -0.30, TOP + 0.30)
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


STROKE = 0.050  # the rod pushes a boat in strokes of at most this
ROD_OPEN, ROD_HOLD = -0.012, 0.0  # across the rod's 5 mm ring, squeezed hard
TONGUE_OPEN, TONGUE_HOLD = -0.012, -0.0015  # on a boat's 5 mm tongue: a light squeeze (a hard one shakes the boat)
ROT_R = np.column_stack([-Y, np.cross(RING_U, -Y), RING_U])  # on the T-handle: in from the robots' side down its slope,
O_R = Orient(ROT_R)  # the fingers closing along the rod across the plate's two faces
ROT_L = ROT_R @ np.diag([-1.0, -1.0, 1.0])  # the left hand's (the same, its fingers swapped: it reaches from the other side)
O_L = Orient(ROT_L)
ROT_CR = np.column_stack([CAP_N, np.cross(CAP_U, CAP_N), CAP_U])  # on a cap's tab: in from the robots' side, tipped
O_CR = Orient(ROT_CR)  # 40 deg down, closing square across the tab's faces
CAP_IN = 0.030  # the open hand comes in onto a cap's tab (and backs off it) this far along its approach
STOP_PUSH = -0.006  # pushing the stopper in, its gas tube held lightly (pads 12 mm apart on the 15.9 mm tube), so its
# taper can turn it into line with the tube, as a hand pushing a stopper does
CAP_SAG = {"inlet_cap": 0.004}  # (the stopper hangs a little low off its gas tube: the hand aims this much high)
CAP_LIFT = 0.012  # a cap lifted this much out of its cradle (clear of its rails) to carry it over
RING_IN = 0.050  # the open hand comes in onto the handle (and backs off it) this far along its approach
RING_LIFT = 0.012  # the rod lifted this much off its rack (clear of its Vs) to carry it over
ROT_D = axis_angle_matrix(Y, TONGUE_TURN) @ _prev_grip(math.pi / 2)  # in from the front, tipped 55 deg,
O_D = Orient(ROT_D)  # closing square across a boat's tongue
assert abs(math.atan2(-ROT_D[2, 0], ROT_D[0, 0]) - TONGUE_TURN) < 1e-6  # the pads meet the tongue's faces flat
LOAD_ORDER = ("chip_boat", "moo3_boat", "s_boat")  # the deepest first: boats cannot pass each other in the tube
# (upstream is the loading end, where the argon comes in: the substrate, furthest downstream, goes in first, precursor 1 last)
TARGET = {"chip_boat": -0.0075, "moo3_boat": MO_AT, "s_boat": S_AT}  # (against the furnace, with the tube slid back)
TUBE_OPEN, TUBE_HOLD = -0.040, -0.016  # round the 50.8 mm tube (pads 80 mm apart), and squeezed hard on it
GRIP_Y = FY1 + 0.060  # the left hand on the tube's far end, outside the furnace (at the start)
# (the substrate's boat goes as deep as the rod reaches; precursor 2's then pushes it on, by its tongue, to 67.5 mm downstream)
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


TH_TIPS = 0.012  # the fingertips stop this far up the handle from the rod's axis (clear of the rod)


def _on_ring(c, back=0.0):
    """TCP for the fingers across the T-handle whose plate crosses the rod's axis at c (``back`` short of it)."""
    tip = np.asarray(c, float) + [0.008, 0.0, 0.016] - back * RING_U  # (the tips 12 mm over the rod's top, the pads up the
    return _tcp(tip, ROT_R, past=0.0)  # plate)


def _tongue(name, centre):
    """Where to hold boat ``name`` by its tongue (3 mm back, clear of its back wall), its underside's centre at centre."""
    L = BOATS[name][0]
    return np.asarray(centre, float) + [0.0, -L / 2 - TONGUE_L / 2 - 0.003, TONGUE_Z]


def _tab_at(name, p):
    """The middle of the stopper's gas tube where it stands out of the large end, its small end at p."""
    return np.asarray(p, float) - (STOP_L + STUB_OUT / 2) * Y


def _on_tab(name, p, R, back=0.0):
    """TCP for the fingers across cap ``name``'s tab (``back`` short of it along the approach), its open end at p."""
    return _tcp(_tab_at(name, p) - back * CAP_U, R, past=0.004)


def _cap_moves(pl, name, R, O, key, t_start):
    """Take cap ``name`` out of its cradle by its tab, line it up with the tube (as seen), push it home, let go: the
    waypoints, and the line-up window (from t_start, when they begin)."""
    up = np.array([0.0, 0.0, 1.0])
    sd = -1
    p0 = cap_start(name)
    end = TUBE_Y0
    p_line = np.array([XA, end + sd * 0.012, TOP + AXZ])  # in line, its open end 12 mm short of the tube's end
    p_turn = np.array([p0[0], p_line[1], TOP + AXZ])  # (along its cradle's side first, then across to the line)
    p_push = np.array([XA, CAP_HOME[name] + 0.005, TOP + AXZ])  # (pressed 5 mm past its seat: the taper stops it;
    # the left hand holds the tube)
    lift = CAP_LIFT * up
    sag = CAP_SAG[name] * up  # (a cap hangs off its tab: the hand aims this much high; the line-up window does the rest)
    a = [(2.0, _on_tab(name, p0, R, CAP_IN) + 0.06 * up, ROD_OPEN, f"hand turned to the {name}'s tab, high over it", O),
         (1.0, _on_tab(name, p0, R, CAP_IN) + 0.03 * up, ROD_OPEN, f"over the {name}'s tab", O),
         (1.0, _on_tab(name, p0, R, CAP_IN), ROD_OPEN, "down beside the tab", O),
         (1.0, _on_tab(name, p0, R), ROD_OPEN, "in round the tab", O),
         (0.6, _on_tab(name, p0, R), ROD_HOLD, "close on the tab", O),
         (1.0, _on_tab(name, p0, R) + lift, ROD_HOLD, "lift the cap out of its cradle", O),
         (1.5, _on_tab(name, p_turn, R) + lift, ROD_HOLD, "along to the tube's end", O),
         (2.0, _on_tab(name, p_line, R) + lift, ROD_HOLD, "over to the tube's line", O),
         (1.0, _Dyn(_on_tab(name, p_line, R) + sag, key), ROD_HOLD, "down in line with the tube", O)]
    tw = t_start + sum(w[0] for w in a)
    O = _DynO(O, key)  # (from here on the hand also turns the stopper into line with the tube, as seen)
    p_near = np.array([XA, end - 0.003, TOP + AXZ])  # its small end 3 mm short of the tube's end
    p_mid = np.array([XA, CAP_HOME[name] - 0.008, TOP + AXZ])  # its small end about 4 mm in: far enough to be held
    # by the mouth, short enough that a slight cock can still be squared up before the full press
    p_seat = np.array([XA, CAP_HOME[name], TOP + AXZ])  # (at its seat: the hand stops pressing before it lets go)
    a += [(3.0, _Dyn(_on_tab(name, p_line, R) + sag, key), ROD_HOLD, "hold still: the stopper seen in line", O),
          (1.0, _Dyn(_on_tab(name, p_near, R) + sag, key), ROD_HOLD, "up to the tube's end", O),
          (2.0, _Dyn(_on_tab(name, p_near, R) + sag, key), ROD_HOLD, "hold still: seen in line again", O),
          (1.6, _Dyn(_on_tab(name, p_mid, R) + sag, key), ROD_HOLD, "start the stopper into the tube's end, held "
                                                                    "firmly so it cannot pivot on the rim", O),
          (2.0, _Dyn(_on_tab(name, p_mid, R) + sag, key), ROD_HOLD, "hold: square in the mouth", O),
          (0.4, _Dyn(_on_tab(name, p_mid, R) + sag, key), STOP_PUSH, "ease the grip on the gas tube", O),
          (2.4, _Dyn(_on_tab(name, p_push, R) + sag, key), STOP_PUSH, "push it home (held lightly: its taper "
                                                                      "centres it)", O),
          (1.5, _Dyn(_on_tab(name, p_push, R) + sag, key), STOP_PUSH, "hold it pressed in, squared up", O),
          (0.8, _Dyn(_on_tab(name, p_push, R) + sag, key), ROD_OPEN, "let go of the gas tube", O),
          (0.6, _Dyn(_on_tab(name, p_push, R) + sag, key), ROD_OPEN, "wait, open", O),
          (1.2, _Dyn(_on_tab(name, p_seat, R, CAP_IN) + sag, key), ROD_OPEN, "back off the gas tube", O),
          (1.0, _Dyn(_on_tab(name, p_seat, R, CAP_IN) + sag + 0.05 * up, key), ROD_OPEN, "straight up, clear of it",
           O)]
    # before it goes in: its small end onto the tube's end as seen (the loose tube may lie a degree off its line in its
    # saddles, which puts its end several mm to the side)
    pl.window(tw + 0.3, tw + 3.0, key, lambda env, n=name: (_tube_end_seen(env) - _seen(env, n)) * [1.0, 0.0, 1.0], name)
    # going in: its back end (the gas tube, where the hand holds it) onto the tube's axis as seen. (Its small end, once
    # in, is held by the tube itself: steering by it would only cock the stopper.)
    # in line again at the tube's end: still outside, so its small end is steered straight onto the end as seen
    pl.window(tw + 4.3, tw + 6.0, key, lambda env, n=name: (_tube_end_seen(env) - _seen(env, n)) * [1.0, 0.0, 1.0], name)
    # through the entry, the square-up dwell, the press home and the hold pressed in: the small end is held by the
    # tube, so its back end (the gas tube) is steered onto the axis (the light grip lets it pivot between the pads)
    pl.window(tw + 6.2, tw + 13.9, key, lambda env, n=name: _back_on_axis(env, n) * [1.0, 0.0, 1.0], name)
    for t0, t1 in ((tw + 0.3, tw + 6.0), (tw + 6.2, tw + 13.9)):  # and its axis into line with the tube's
        _ct["rwins"].append((t0, t1, key, lambda env, n=name: _axis_err(env, n), name))
    return a


def _axis_err(env, name):
    """The turn (a rotation vector, rad) that would bring the stopper's axis into line with the tube's, as seen."""
    ty = labkit._part_state(env, labkit._SPECS["quartz_tube"]).rotation()[:, 1]
    a = labkit._part_state(env, labkit._SPECS[name]).rotation()[:, 1]
    a = a if np.dot(a, ty) > 0 else -a
    return np.cross(a, ty)


def _back_on_axis(env, name):
    """How far the stopper's back end (its gas tube, 40 mm behind its small end) is off the tube's axis, as seen."""
    t = labkit._part_state(env, labkit._SPECS["quartz_tube"])
    ty = t.rotation()[:, 1]
    e = np.asarray(t.pos, float) - _HL * ty
    c = labkit._part_state(env, labkit._SPECS[name])
    b = np.asarray(c.pos, float) - (STOP_L + STUB_OUT / 2) * c.rotation()[:, 1]
    return e + np.dot(b - e, ty) * ty - b


def demo():
    up = np.array([0.0, 0.0, 1.0])
    a_d = _appr(ROT_D)
    pl = _Plan()
    from yam_arena.constants import LEFT_BASE_POS
    _READY_L = _ik.tcp_pose(np.array(_READY, float), np.array(LEFT_BASE_POS))
    # the left hand takes the tube by its far end and slides it 75 mm towards the far end, so the loading end stands
    # close to the furnace; it holds the tube there while the right hand loads the boats
    a_t = _appr(ROT_T)
    g0 = np.array([XA, GRIP_Y, TOP + AXZ])
    on_tube = lambda g: _tcp(g, ROT_T, past=0.012) - 0.004 * up  # the fingertips 12 mm past the tube's axis: the
    # pads across it (aimed 4 mm low: the closing pads draw the tube up into the hand, and a raised far end tips the
    # tube's floor down at the loading end)
    left = [(1.0, _READY_L[0] + 0.06 * up, TUBE_OPEN, "straight up", Orient(_READY_L[1])),
            (2.0, on_tube(g0) - 0.06 * a_t, TUBE_OPEN, "hand turned over the tube's far end", O_T),
            (1.0, on_tube(g0) - 0.02 * a_t, TUBE_OPEN, "down beside the tube", O_T),
            (1.0, on_tube(g0), TUBE_OPEN, "round the tube", O_T),
            (0.8, on_tube(g0), TUBE_HOLD, "close on the tube", O_T),
            (5.0, _Dyn(on_tube(g0 + SHIFT * Y), "ts"), TUBE_HOLD, "slide the tube 75 mm towards the far end", O_T)]
    pl.phase(left=left,
             right=[(2.0, np.array([0.33, -0.27, TOP + 0.33]), TONGUE_OPEN, "hand turned, up clear of the tube's end",
                     O_D)])
    tw = pl.phase(left=[(4.0, _Dyn(on_tube(g0 + SHIFT * Y), "ts"), TUBE_HOLD, "hold still: the tube seen 75 mm over",
                         O_T)])  # (the pads creep a little on the round tube: made up from where the tube is seen)
    pl.window(tw + 0.3, tw + 4.0, "ts",
              lambda env: (TUBE_C + SHIFT * Y - _seen(env, "quartz_tube")) * [0.0, 1.0, 0.0], "quartz_tube")
    keys = iter(f"r{i}" for i in range(1, 99))
    for bi, name in enumerate(LOAD_ORDER):
        L = BOATS[name][0]
        on_tray = boat_start(name)
        tg = _tongue(name, on_tray)
        mouth = np.array([XA, LOAD_Y0 - MOUTH_OUT + L / 2, TOP + BOAT_Z + 0.0012])
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
                  lambda env, n=name, z=mouth[2] + MOUTH_HI, yl=mouth[1] - (L - MOUTH_OUT + 0.008):
                  np.array([XA, yl + _seen(env, "quartz_tube")[1] - TUBE_C[1] - SHIFT, z] - _seen(env, n)),
                  name)
        pl.phase(right=[
            (2.0, _Dyn(_tcp(g_mouth + hi, ROT_D, past=0.006), k_b), TONGUE_HOLD, "slide it into the mouth", O_D),
            (0.8, _Dyn(_tcp(g_mouth, ROT_D, past=0.006), k_b), TONGUE_HOLD, "set it down on the tube's floor", O_D),
            (0.6, _Dyn(_tcp(g_mouth, ROT_D, past=0.006), k_b), TONGUE_OPEN, "let go of the boat", O_D),
            (0.8, _Dyn(_tcp(g_mouth - 0.035 * a_d, ROT_D, past=0.006), k_b), TONGUE_OPEN, "back off the boat", O_D),
            (0.8, _Dyn(_tcp(g_mouth - 0.035 * a_d + 0.05 * up, ROT_D, past=0.006), k_b), TONGUE_OPEN, "up and away", O_D)])
        # -- take the rod by its ring off its rack; push the boat to its place in one stroke; pull the rod back out and
        # put it back on its rack ---------------------------------------------------------------------------------------
        tongue_end = LOAD_Y0 - MOUTH_OUT - TONGUE_L
        p_goal = TARGET[name] + SHIFT - L / 2 - TONGUE_L  # the pusher there: the boat centred at its target (once the
        # tube is slid back)
        on_rack, k_r = _ring_at(ROD_START[1], RACK_X), next(keys)
        p_line = tongue_end - 0.008  # the pusher, in line, before going in
        in_line = _ring_at(p_line)
        seq = [(2.0, _on_ring(on_rack, RING_IN) + 0.03 * up, ROD_OPEN, "hand turned to the rod's handle, over it", O_R),
               (1.0, _on_ring(on_rack, RING_IN), ROD_OPEN, "down beside the handle", O_R),
               (1.0, _on_ring(on_rack), ROD_OPEN, "in round the handle", O_R),
               (0.6, _on_ring(on_rack), ROD_HOLD, "close on the handle", O_R),
               (max(1.5, 25.0 * (p_line - ROD_START[1])), _Dyn(_on_ring(in_line), k_r), ROD_HOLD,
                "slide the rod along its rests to the tube's mouth", O_R)]
        pl.phase(right=seq)
        tw = pl.phase(right=[(1.8, _Dyn(_on_ring(in_line), k_r), ROD_HOLD, "hold still: the rod's tip seen in line", O_R)])
        pl.window(tw + 0.3, tw + 1.8, k_r,
                  lambda env: np.array([XA, 0.0, TOP + ROD_Z]) - _rod_seen_at(env, tongue_end - 0.010) * [1.0, 0.0, 1.0],
                  "push_rod")
        touch = _ring_at(tongue_end - 0.003)
        pl.phase(right=[(1.5, _Dyn(_on_ring(touch), k_r), ROD_HOLD, "the pusher up to the boat's tongue", O_R)])
        # in strokes of about 50 mm, the rod's tip re-centred on the tube's axis (as seen) between them: a hand-held rod
        # turns a little in the grip as it pushes, and over one long stroke its tip walks off the boat's tongue
        n_st = max(1, int(math.ceil((p_goal - tongue_end) / STROKE)))
        y_prev = tongue_end - 0.003
        for i in range(1, n_st + 1):
            y_i = tongue_end - 0.003 + (p_goal - tongue_end + 0.003) * i / n_st
            pl.phase(right=[(max(1.0, 40.0 * (y_i - y_prev)), _Dyn(_on_ring(_ring_at(y_i)), k_r), ROD_HOLD,
                             f"push the boat in, slowly (stroke {i} of {n_st})", O_R)])
            y_prev = y_i
            if i < n_st:
                tw = pl.phase(right=[(1.6, _Dyn(_on_ring(_ring_at(y_i)), k_r), ROD_HOLD,
                                      "hold still: the rod's tip seen, re-centred", O_R)])
                pl.window(tw + 0.3, tw + 1.6, k_r,
                          lambda env, y=y_i: (np.array([XA, 0.0, TOP + ROD_Z]) - _rod_seen_at(env, y - 0.002))
                          * [1.0, 0.0, 1.0], "push_rod")
        tw = pl.phase(right=[(2.0, _Dyn(_on_ring(_ring_at(p_goal)), k_r), ROD_HOLD, "hold still: the rod seen at its depth",
                              O_R)])
        pl.window(tw + 0.3, tw + 2.0, k_r,  # (the boat is out of sight in the tube: judged by the rod's depth)
                  lambda env, g=p_goal: np.array([0.0, g + _seen(env, "quartz_tube")[1] - TUBE_C[1] - SHIFT
                                                  - _seen(env, "push_rod")[1], 0.0]), "push_rod")
        pl.phase(right=[(max(1.5, 25.0 * (p_goal - p_line)), _Dyn(_on_ring(in_line), k_r), ROD_HOLD, "pull the rod back out",
                         O_R),
                        (max(1.5, 25.0 * (p_line - ROD_START[1])), _on_ring(on_rack), ROD_HOLD,
                         "slide it back along its rests to its place", O_R),
                        (0.6, _on_ring(on_rack), ROD_OPEN, "let go of the handle", O_R),
                        (1.0, _on_ring(on_rack, RING_IN), ROD_OPEN, "back off the handle", O_R),
                        (1.0, _on_ring(on_rack, RING_IN) + 0.03 * up, ROD_OPEN, "up, clear of the rod", O_R)])
    # the left hand slides the tube back, centred again (as seen: its ends against the furnace), and lets go of it
    pl.phase(left=[(6.0, _Dyn(on_tube(g0), "tb"), TUBE_HOLD, "slide the tube back, centred in the furnace", O_T)],
             right=[(2.0, RIGHT_PARK, ROD_OPEN, "up and clear of the tube's end", O_T)])
    tw = pl.phase(left=[(5.0, _Dyn(on_tube(g0), "tb"), TUBE_HOLD, "hold still: the tube seen centred", O_T)])
    pl.window(tw + 0.3, tw + 5.0, "tb", lambda env: (TUBE_C - _seen(env, "quartz_tube")) * [0.0, 1.0, 0.0], "quartz_tube")
    # the right hand pushes the inlet stopper into the tube's loading end while the left hand holds the tube still
    t_c = pl.t
    right = _cap_moves(pl, "inlet_cap", ROT_CR, O_CR, "cn", t_c)
    pl.phase(right=right, left=[(sum(w[0] for w in right), _Dyn(on_tube(g0), "tb"), TUBE_HOLD,
                                 "hold the tube while the stopper is pushed in", O_T)])
    pl.phase(left=[(0.8, _Dyn(on_tube(g0), "tb"), TUBE_OPEN, "let go of the tube", O_T),
                   (1.0, _Dyn(on_tube(g0) - 0.05 * a_t, "tb"), TUBE_OPEN, "back off the tube", O_T),
                   (1.0, _Dyn(on_tube(g0) - 0.05 * a_t + 0.10 * up, "tb"), TUBE_OPEN, "up, well clear of the tube", O_T)],
             right=[(2.0, RIGHT_PARK, ROD_OPEN, "up and clear", O_T)])
    pl.phase(right=[(2.0, RIGHT_REST, ROD_OPEN, "rest", O_T)], left=[(2.0, LEFT_REST, ROD_OPEN, "to the park", O_T)])
    pl.phase(right=[(3.0, RIGHT_REST, ROD_OPEN, "rest", O_T)], left=[(3.0, LEFT_REST, ROD_OPEN, "rest", O_T)])
    return {"left": pl.L, "right": pl.R}
