"""Geometry for the music stand task's props, built in code from published dimensions (no third-party meshes).

Every builder returns a list of (points (N,3), tris (M,3), rgb, collide) tuples in the object's own frame, metres.
collide: "hull" (one convex collider), "none" (visual only).

Soprano ukulele, typical published figures: overall about 530 mm, body 240 mm long, lower bout about 170 mm, waist about
125 mm, upper bout about 135 mm, body depth about 60 mm, scale 345 mm (nut to saddle), 12 frets clear of the body,
nut width about 35 mm, about 0.35 to 0.45 kg.
"""
import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull, Delaunay

# ------------------------------------------------------------------------------------------------ small helpers


def hull(pts, color, collide="none"):
    p = np.asarray(pts, float)
    try:
        h = ConvexHull(p)
    except Exception:                      # flat or degenerate (a wall quad where the outlines meet): joggle
        h = ConvexHull(p, qhull_options="QJ")
    return (p, h.simplices.astype(np.int64), color, collide)


def box(c, sz, color, collide="none"):
    return hull([(c[0] + a * sz[0] / 2, c[1] + b * sz[1] / 2, c[2] + d * sz[2] / 2)
                 for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], color, collide)


def rod(p0, p1, r, color, n=6, collide="none"):
    """A thin prism from p0 to p1 of radius r (strings, shafts)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = p1 - p0
    a /= np.linalg.norm(a)
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    ring = [r * (math.cos(t) * u + math.sin(t) * v) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    return hull([p0 + q for q in ring] + [p1 + q for q in ring], color, collide)


def disc(c, r, x0, x1, color, n=32):
    """A flat disc in the y-z plane between x0 and x1 (decals on a face that looks along -x)."""
    return hull([(x, c[0] + r * math.cos(t), c[1] + r * math.sin(t))
                 for t in np.linspace(0, 2 * math.pi, n, endpoint=False) for x in (x0, x1)], color)


def ring(c, r0, r1, x, color, n=48):
    """An annulus in the y-z plane at depth x (a rosette)."""
    pts, tris = [], []
    for k, t in enumerate(np.linspace(0, 2 * math.pi, n, endpoint=False)):
        pts += [(x, c[0] + r0 * math.cos(t), c[1] + r0 * math.sin(t)), (x, c[0] + r1 * math.cos(t), c[1] + r1 * math.sin(t))]
        a, b, c2, d = 2 * k, 2 * k + 1, (2 * k + 2) % (2 * n), (2 * k + 3) % (2 * n)
        tris += [(a, b, d), (a, d, c2)]
    return (np.asarray(pts, float), np.asarray(tris, np.int64), color, "none")


def polygon_fill(loop2d):
    """Triangles filling a simple 2-D polygon (Delaunay of boundary + interior grid, kept if inside)."""
    loop2d = np.asarray(loop2d, float)
    lo, hi = loop2d.min(0), loop2d.max(0)
    g = np.mgrid[lo[0]:hi[0]:0.008, lo[1]:hi[1]:0.008].reshape(2, -1).T
    from matplotlib.path import Path
    path = Path(loop2d)
    inner = g[path.contains_points(g, radius=-0.003)]
    pts = np.vstack([loop2d, inner])
    tri = Delaunay(pts).simplices
    keep = path.contains_points(pts[tri].mean(axis=1), radius=-1e-6)
    return pts, tri[keep]


# ------------------------------------------------------------------------------------------------ ukulele

UKE_WOOD_TOP = (0.66, 0.40, 0.19)      # koa / mahogany top, warm
UKE_WOOD_GRAIN = (0.58, 0.34, 0.15)
UKE_WOOD_SIDE = (0.47, 0.26, 0.12)     # sides and back, darker
UKE_NECK = (0.55, 0.33, 0.17)
UKE_BOARD = (0.17, 0.10, 0.07)         # rosewood fretboard, bridge, headstock veneer
UKE_BIND = (0.90, 0.86, 0.76)          # cream binding
UKE_BONE = (0.93, 0.91, 0.84)
UKE_NICKEL = (0.80, 0.80, 0.83)
UKE_STRING = (0.95, 0.94, 0.90)        # white nylon
UKE_HOLE = (0.03, 0.02, 0.015)
UKE_PEARL = (0.94, 0.93, 0.90)

BODY_L, DEPTH = 0.240, 0.060
Z_NUT = 0.4125                         # 12th fret at the body joint (z = 0.240): nut at 0.240 + 0.345 / 2
Z_SADDLE = Z_NUT - 0.345
FACE = -DEPTH / 2                      # the top (soundboard) is the -x face
NX = -0.026                            # neck group offset: fingerboard underside on the soundboard


def uke_outline(n=120):
    """Closed body outline (y, z): lower bout 170 mm, waist about 125 mm, upper bout 135 mm, 240 mm long."""
    half = [(0.0, 0.0), (0.030, 0.004), (0.058, 0.016), (0.078, 0.036), (0.085, 0.064), (0.080, 0.094),
            (0.068, 0.116), (0.0625, 0.132), (0.064, 0.150), (0.0675, 0.178), (0.063, 0.204), (0.048, 0.226),
            (0.026, 0.237), (0.0, 0.240)]
    loop = half + [(-y, z) for (y, z) in reversed(half[1:-1])]
    p = np.asarray(loop + [loop[0]], float)
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    cs = CubicSpline(u, p, bc_type="periodic")
    return cs(np.linspace(0, u[-1], n, endpoint=False))


def _band(loop_a, xa, loop_b, xb, color):
    """Side band between two closed loops at depths xa and xb."""
    n = len(loop_a)
    pts = [(xa, y, z) for (y, z) in loop_a] + [(xb, y, z) for (y, z) in loop_b]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + j), (i, n + j, n + i)]
    return (np.asarray(pts, float), np.asarray(tris, np.int64), color, "none")


def _face(loop, x, color, flip=False):
    pts2, tri = polygon_fill(loop)
    if flip:
        tri = tri[:, ::-1]
    return (np.c_[np.full(len(pts2), x), pts2], tri.astype(np.int64), color, "none")


def ukulele():
    out = []
    L = uke_outline()
    c = L.mean(axis=0)
    shrink = lambda s: c + (L - c) * s
    # body: rounded edges (four loops), soundboard and back faces, cream binding on the top edge
    xs = [FACE, FACE + 0.003, -FACE - 0.003, -FACE]
    sc = [0.975, 1.0, 1.0, 0.98]
    loops = [c + (L - c) * s for s in sc]
    out.append(_band(loops[0], xs[0], loops[1], xs[1], UKE_BIND))
    out.append(_band(loops[1], xs[1], loops[2], xs[2], UKE_WOOD_SIDE))
    out.append(_band(loops[2], xs[2], loops[3], xs[3], UKE_WOOD_SIDE))
    out.append(_face(loops[0], FACE - 0.0002, UKE_BIND))
    out.append(_face(c + (L - c) * 0.955, FACE - 0.0004, UKE_WOOD_TOP))
    out.append(_face(loops[3], -FACE + 0.0002, UKE_WOOD_SIDE, flip=True))
    # grain: a few long strips on the soundboard, clipped to the outline
    from matplotlib.path import Path
    top = Path(c + (L - c) * 0.95)
    for k, y in enumerate(np.linspace(-0.072, 0.072, 13)):
        zs = np.linspace(0.0, BODY_L, 241)
        inside = top.contains_points(np.c_[np.full_like(zs, y), zs])
        runs, start = [], None
        for i, f in enumerate(inside):
            if f and start is None:
                start = i
            if (not f or i == len(zs) - 1) and start is not None:
                runs.append((zs[start], zs[i - 1 if not f else i]))
                start = None
        for z0, z1 in runs:
            if z1 - z0 > 0.01:
                out.append(box((FACE - 0.0006, y + 0.002 * math.sin(3 * k), (z0 + z1) / 2), (0.0002, 0.0035, z1 - z0),
                               UKE_WOOD_GRAIN))
    # sound hole 50 mm with a rosette, centred in the upper bout
    zh = 0.168
    out.append(disc((0.0, zh), 0.025, FACE - 0.0012, FACE - 0.0008, UKE_HOLE))
    out.append(ring((0.0, zh), 0.0262, 0.0275, FACE - 0.0010, UKE_BIND))
    out.append(ring((0.0, zh), 0.0278, 0.0300, FACE - 0.0010, UKE_BOARD))
    out.append(ring((0.0, zh), 0.0303, 0.0312, FACE - 0.0010, UKE_BIND))
    # tie-bar bridge with a bone saddle at the 345 mm scale
    out.append(box((FACE - 0.004, 0.0, Z_SADDLE - 0.002), (0.008, 0.074, 0.016), UKE_BOARD))
    out.append(box((FACE - 0.0095, 0.0, Z_SADDLE), (0.003, 0.050, 0.0025), UKE_BONE))
    # neck (heel inside the body end), fretboard, frets, dots, nut: built with the neck surface at x = -0.004, then
    # raised by NX so the fingerboard sits on the soundboard
    i_neck = len(out)
    nk = []
    for z, w, d in ((0.232, 0.0228, 0.020), (Z_NUT, 0.0178, 0.016)):
        for t in np.linspace(0, math.pi, 9):
            nk.append((-0.004 + d * math.sin(t) * 1.0, w * math.cos(t), z))
    out.append(hull(nk, UKE_NECK, "hull"))
    out.append(hull([(x, y, z) for x in (-0.004, -FACE - NX) for y in (-0.020, 0.020) for z in (0.226, 0.250)], UKE_NECK))  # heel
    # fingerboard (Martin S1 page: 1 13/32 in at the nut, 1 13/16 in at the 12th fret, 17 frets, joins at the 12th)
    FB_END = 0.194
    wfb = lambda z: 0.0178 + (0.0230 - 0.0178) * (Z_NUT - z) / (Z_NUT - 0.240)
    fb = [(-wfb(FB_END), FB_END), (wfb(FB_END), FB_END), (0.0178, Z_NUT), (-0.0178, Z_NUT)]
    out.append(hull([(x, y, z) for x in (-0.010, -0.004) for (y, z) in fb], UKE_BOARD))
    fz = lambda k: Z_NUT - 0.345 * (1 - 2 ** (-k / 12.0))
    for k in range(1, 18):
        z = fz(k)
        out.append(box((-0.0108, 0.0, z), (0.0016, 2 * wfb(z) - 0.001, 0.0012), UKE_NICKEL))
    for k, ys in ((5, (0.0,)), (7, (0.0,)), (10, (0.0,)), (12, (-0.007, 0.007))):
        z = 0.5 * (fz(k) + fz(k - 1))
        for y in ys:
            out.append(disc((y, z), 0.0022, -0.0106, -0.0101, UKE_PEARL, n=12))
    out.append(box((-0.0115, 0.0, Z_NUT + 0.0015), (0.0035, 0.0362, 0.004), UKE_BONE))
    # headstock: flared, crowned top, rosewood face veneer
    hs = [(-0.0185, Z_NUT + 0.003), (0.0185, Z_NUT + 0.003), (0.0265, 0.480), (0.0295, 0.512), (0.022, 0.527),
          (0.0, 0.521), (-0.022, 0.527), (-0.0295, 0.512), (-0.0265, 0.480)]
    out.append(hull([(x, y, z) for x in (-0.0045, 0.0095) for (y, z) in hs], UKE_NECK, "hull"))
    hs_loop = np.asarray(hs, float)
    out.append(_face(np.c_[hs_loop[:, 0] * 0.96, hs_loop[:, 1]], -0.0047, UKE_BOARD))
    # four geared tuners: posts on the face, gear housings and pearloid buttons out the sides
    posts = [(-0.012, 0.455), (0.012, 0.455), (-0.014, 0.497), (0.014, 0.497)]
    for (y, z) in posts:
        side = 1 if y > 0 else -1
        out.append(rod((-0.0045, y, z), (-0.016, y, z), 0.0032, UKE_NICKEL, n=10))      # bushing + post
        out.append(box((0.0125, y + side * 0.002, z), (0.006, 0.014, 0.012), UKE_NICKEL))  # gear housing (back)
        out.append(rod((0.0125, y + side * 0.009, z), (0.0125, side * 0.040, z), 0.0016, UKE_NICKEL))  # shaft
        out.append(hull([(0.0125 + a * 0.0035, side * 0.040 + b * 0.008 + side * 0.008, z + d * 0.007)
                         for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], UKE_PEARL))            # button
    for k in range(i_neck, len(out)):                                   # raise the neck group onto the soundboard
        p_, t_, c_, col_ = out[k]
        out[k] = (p_ + [NX, 0.0, 0.0], t_, c_, col_)
    # four nylon strings: tie block -> saddle -> nut -> posts (G C E A, re-entrant)
    ys_saddle, ys_nut = (-0.0209, -0.0070, 0.0070, 0.0209), (-0.0140, -0.0047, 0.0047, 0.0140)   # 41.7 mm at the bridge
    for i in range(4):
        r = (0.00045, 0.0006, 0.0005, 0.0004)[i]
        tie = (FACE - 0.008, ys_saddle[i], Z_SADDLE - 0.008)
        sad = (FACE - 0.0110, ys_saddle[i], Z_SADDLE)
        nut = (-0.0135 + NX, ys_nut[i], Z_NUT + 0.0015)
        py, pz = posts[(0, 2, 3, 1)[i]]
        out.append(rod(tie, sad, r, UKE_STRING, n=5))
        out.append(rod(sad, nut, r, UKE_STRING, n=5))
        out.append(rod(nut, (-0.014 + NX, py, pz), r, UKE_STRING, n=5))
    # colliders: lower and upper bout as convex hulls 1 mm inside the closed body shell (labkit renders colliders,
    # so they sit hidden inside it); the neck and the headstock collide as their own convex visuals (above)
    Ls = c + (L - c) * 0.975
    for part in (Ls[Ls[:, 1] <= 0.132], Ls[Ls[:, 1] >= 0.132]):
        out.append(hull([(x, y, z) for x in (FACE + 0.001, -FACE - 0.001) for (y, z) in part], UKE_WOOD_SIDE, "hull"))
    return out


UKE_MASS = 0.40   # kg, typical soprano (0.35 to 0.45 kg)


# ------------------------------------------------------------------------------------------------ hardshell case
# A soprano hardshell case shaped on the ukulele's own outline (cavity 11 mm outside it, shell 14 mm; base rim 75 mm, lid 25 mm), black textured shell,
# plush lining, a padded neck rest, two chrome draw latches and a handle on the front (+x), two barrel hinges on the
# back (-x). Origin: centre of the case bottom; +y toward the head end.

CASE_SHELL = (0.045, 0.045, 0.05)
CASE_VALANCE = (0.30, 0.30, 0.32)      # aluminium rim valance
CASE_LINING = (0.11, 0.15, 0.40)       # blue plush
CASE_CHROME = (0.82, 0.82, 0.85)
CASE_RIM = float(__import__("os").environ.get("MS_CASERIM", "0.075"))   # base rim height
CASE_FLOOR, CASE_WALL, CASE_LIDT = 0.010, 0.010, 0.025
UKE_Y0 = -0.28                         # the ukulele's body end in the case frame (case y = ukulele z + UKE_Y0)
CAV_GAP, SHELL_T = 0.011, 0.014        # cavity clearance around the ukulele; plush lining plus shell wall


def _uke_half_profile():
    """Half-width (m) of the shape the case cavity follows, along the ukulele's length z: its own body outline, a
    constant-width neck channel, and a headstock pocket wide enough for the tuner buttons."""
    L = uke_outline(240)
    r = L[L[:, 0] >= 0]
    r = r[np.argsort(r[:, 1])]
    zs = np.linspace(0.0, 0.53, 531)
    body = np.where(zs <= BODY_L, np.interp(zs, r[:, 1], r[:, 0], left=0.0, right=0.0), 0.0)
    neck = np.where((zs >= 0.200) & (zs <= Z_NUT), 0.023, 0.0)
    head = np.interp(zs, [Z_NUT, 0.437, 0.515, 0.527, 0.530], [0.023, 0.056, 0.056, 0.031, 0.0], left=0.0, right=0.0)
    return zs, np.maximum(body, np.maximum(neck, head))


_ZS, _HZ = _uke_half_profile()


def _offset_half(z, r):
    """Half-width at z of the profile grown by r all round (Minkowski sum with a disc)."""
    d = np.abs(_ZS - z)
    m = d < r
    if not m.any():
        return 0.0
    return float((_HZ[m] + np.sqrt(r * r - d[m] ** 2)).max())


def case_half(y):
    """Half-width of the case outline at y (outside): the ukulele's own outline grown by the cavity gap and the
    lining plus shell, so lower bout, waist, upper bout, neck channel and headstock pocket follow the instrument."""
    return _offset_half(y - UKE_Y0, CAV_GAP + SHELL_T)


CASE_YMIN = UKE_Y0 - CAV_GAP - SHELL_T
CASE_YMAX = UKE_Y0 + 0.530 + CAV_GAP + SHELL_T


def _outline(inset=0.0, n=110):
    ys = np.linspace(CASE_YMIN + 0.0005, CASE_YMAX - 0.0005, n)
    right = [(max(case_half(y) - inset, 0.002), y) for y in ys]
    return np.asarray(right + [(-x, y) for (x, y) in reversed(right)], float)


def _slices(z0, z1, tf=None, band=0.04):
    """Convex collider slices of the outline (floor or lid plate), each a band along y."""
    O = _outline()
    out = []
    edges = np.arange(CASE_YMIN, CASE_YMAX + band, band)
    for y0, y1 in zip(edges[:-1], edges[1:]):
        sel = O[(O[:, 1] >= y0 - 0.002) & (O[:, 1] <= y1 + 0.002)]
        if len(sel) < 3:
            continue
        pts = [(x, y, z) for (x, y) in sel for z in (z0, z1)]
        if tf is not None:
            pts = [tf(p) for p in pts]
        out.append(hull(pts, CASE_SHELL, "hull"))
    return out


def _walls(outer, inner, z0, z1, color, tf=None, collide="hull"):
    """Wall segments between an outer and an inner closed outline (same point count), as convex quads."""
    out = []
    n = len(outer)
    for i in range(n):
        j = (i + 1) % n
        q = [outer[i], outer[j], inner[j], inner[i]]
        pts = [(x, y, z) for (x, y) in q for z in (z0, z1)]
        if tf is not None:
            pts = [tf(p) for p in pts]
        out.append(hull(pts, color, collide))
    return out


def _slab(loop, z0, z1, color, collide="none", tf=None):
    """A flat slab filling a closed outline (triangulated top, bottom and side), visual."""
    pts2, tri = polygon_fill(loop)
    n = len(pts2)
    P = np.r_[np.c_[pts2, np.full(n, z0)], np.c_[pts2, np.full(n, z1)]]
    T = np.r_[tri[:, ::-1], tri + n]
    m = len(loop)
    side = []
    for i in range(m):
        j = (i + 1) % m
        side += [(i, j, n + j), (i, n + j, n + i)]
    T = np.r_[T, np.asarray(side, np.int64)]
    if tf is not None:
        P = np.asarray([tf(p) for p in P])
    return (P, T.astype(np.int64), color, collide)


def case_hinge_x():
    return -max(case_half(y) for y in np.linspace(UKE_Y0, UKE_Y0 + 0.12, 61))      # back of the lower bout


def case(lid_deg=270.0):
    """Base plus lid. lid_deg: 0 closed; 180 flat open behind (-x); 270 hanging straight down behind the back wall."""
    out = []
    O = _outline()
    I = _outline(inset=CASE_WALL)
    L = _outline(inset=CASE_WALL + 0.004)
    # bottom (collider in convex pieces) and plush floor
    out += _slices(0.0, CASE_FLOOR)
    out.append(_slab(L, CASE_FLOOR, CASE_FLOOR + 0.002, CASE_LINING))
    # walls (shell, colliding), plush wall lining, rim valance
    out += _walls(O, I, 0.0, CASE_RIM, CASE_SHELL)
    out += _walls(I, L, CASE_FLOOR, CASE_RIM - 0.004, CASE_LINING, collide="none")
    out += _walls(_outline(inset=-0.0015), _outline(inset=0.0025), CASE_RIM - 0.006, CASE_RIM + 0.001, CASE_VALANCE,
                  collide="none")
    # padded neck rest under the ukulele's neck (its top meets the neck's back)
    out.append(hull([(x, y, z) for x in (-0.030, 0.030) for y in (0.040, 0.085) for z in (CASE_FLOOR, 0.0510)],
                    CASE_LINING, "hull"))
    # front (+x): two draw latches and a handle
    for yl in (-0.17, 0.00):
        hx = case_half(yl)
        out.append(box((hx + 0.004, yl, CASE_RIM - 0.010), (0.008, 0.030, 0.022), CASE_CHROME))
        out.append(box((hx + 0.006, yl, CASE_RIM - 0.024), (0.004, 0.014, 0.010), CASE_CHROME))
    hy = -0.085
    hx = case_half(hy)
    for b in (-0.045, 0.045):
        out.append(box((hx + 0.003, hy + b, CASE_RIM - 0.020), (0.006, 0.012, 0.016), CASE_SHELL))
    out.append(hull([(hx + 0.006 + a, hy + b, CASE_RIM - 0.020 + d) for a in (0.0, 0.016) for b in (-0.05, 0.05)
                     for d in (-0.006, 0.006)], CASE_SHELL))
    # lid: same outline, 25 mm deep, hinged on the back (-x); pivot at the outer shell corner above the rim
    x0, zp = case_hinge_x(), CASE_RIM + CASE_LIDT
    th = math.radians(lid_deg)
    cl, sl = math.cos(th), math.sin(th)

    def rot(p):
        x, y, z = p
        dx, dz = x - x0, z - zp
        return (x0 + dx * cl - dz * sl, y, zp + dx * sl + dz * cl)
    lz = lambda z: CASE_RIM + z                                          # lid built closed, then rotated
    out += _walls(O, I, lz(0.0), lz(CASE_LIDT - 0.006), CASE_SHELL, tf=rot)
    out.append(_slab(O, lz(CASE_LIDT - 0.006), lz(CASE_LIDT), CASE_SHELL, tf=rot))
    out += _slices(lz(CASE_LIDT - 0.006), lz(CASE_LIDT - 0.001), tf=rot)
    out.append(_slab(L, lz(CASE_LIDT - 0.009), lz(CASE_LIDT - 0.006), CASE_LINING, tf=rot))
    out += _walls(_outline(inset=-0.0015), _outline(inset=0.0025), lz(-0.001), lz(0.006), CASE_VALANCE, tf=rot,
                  collide="none")
    for yh in (-0.17, 0.02):                                             # two barrel hinges on the back
        out.append(rod((x0 - 0.002, yh - 0.025, CASE_RIM), (x0 - 0.002, yh + 0.025, CASE_RIM), 0.004, CASE_CHROME, n=10))
    return out


CASE_MASS = 1.0   # kg, empty soprano hardshell case (typical 0.9 to 1.2 kg)

# ------------------------------------------------------------------------------------------------ small side table
TABLE_TOP = (0.56, 0.38, 0.22)         # oak
TABLE_EDGE = (0.48, 0.32, 0.18)


def side_table(depth=0.26, width=0.70, height=0.625, top_t=0.022):
    """A small wooden side table: top, 70 mm apron, four tapered square legs. Origin: footprint centre on the floor."""
    out = []
    zt = height - top_t
    out.append(box((0, 0, zt + top_t / 2), (depth, width, top_t), TABLE_TOP, "hull"))
    ia, ib = depth / 2 - 0.045, width / 2 - 0.045
    for sx in (-1, 1):                                                   # apron
        out.append(box((sx * ia, 0, zt - 0.035), (0.018, 2 * ib, 0.070), TABLE_EDGE, "hull"))
    for sy in (-1, 1):
        out.append(box((0, sy * ib, zt - 0.035), (2 * ia, 0.018, 0.070), TABLE_EDGE, "hull"))
    for sx in (-1, 1):                                                   # legs, 40 mm at the top tapering to 25 mm
        for sy in (-1, 1):
            cx, cy = sx * (depth / 2 - 0.035), sy * (width / 2 - 0.035)
            out.append(hull([(cx + a * w, cy + b * w, z) for (w, z) in ((0.020, zt), (0.0125, 0.0))
                             for a in (-1, 1) for b in (-1, 1)], TABLE_EDGE, "hull"))
    return out
