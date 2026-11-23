"""Visual-only meshes for the dial comparator task (collide "none"; the colliders in task.py are unchanged and are enclosed
by these meshes by 0.1-0.3 mm where a part is restyled). get(key, g) returns a list of labkit Mesh for that body; g is
task.py's globals at call time. Joint keys (arm, spindle, needle, lever, knob) are in WORLD coordinates at rest; part keys
(stand at the origin, parts_tray, gauge_block, spacer_*, tray_*) are in the part frame."""
import math

import numpy as np

try:
    import labkit as _lk
    from scipy.spatial import ConvexHull as _Hull
    _OK = isinstance(_lk.Mesh, type)
except Exception:  # noqa: BLE001
    _OK = False

STEEL_BRIGHT = (0.86, 0.87, 0.89)
STEEL_GROUND = (0.70, 0.71, 0.73)
CHROME_V = (0.90, 0.91, 0.92)
CAST = (0.22, 0.23, 0.25)
KNOB_BLACK = (0.07, 0.07, 0.08)
FACE_WHITE = (0.97, 0.97, 0.95)
INK = (0.04, 0.04, 0.05)
LAPPED = (0.40, 0.41, 0.43)
SPECK = (0.34, 0.34, 0.36)
CARBIDE = (0.30, 0.30, 0.33)
BORE = (0.10, 0.10, 0.11)
POCKET = (0.15, 0.28, 0.50)

if _OK:
    try:
        _lk.SURFACES.update({tuple(round(c, 4) for c in STEEL_BRIGHT): {"roughness": 0.12, "metallic": 0.95},
                             tuple(round(c, 4) for c in STEEL_GROUND): {"roughness": 0.35, "metallic": 0.85},
                             tuple(round(c, 4) for c in CHROME_V): {"roughness": 0.08, "metallic": 0.98},
                             tuple(round(c, 4) for c in CAST): {"roughness": 0.85, "metallic": 0.1},
                             tuple(round(c, 4) for c in LAPPED): {"roughness": 0.3, "metallic": 0.0}})
    except Exception:  # noqa: BLE001
        pass

FONT = {"0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "001", "001", "001"],
        "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"],
        "M": ["101", "111", "111", "101", "101"], "A": ["010", "101", "111", "101", "101"],
        "S": ["111", "100", "111", "001", "111"], "T": ["111", "010", "010", "010", "010"],
        "E": ["111", "100", "110", "100", "111"], "R": ["110", "101", "110", "101", "101"],
        ".": ["000", "000", "000", "000", "010"], " ": ["000", "000", "000", "000", "000"]}
MASTER_PAINT = (0.95, 0.75, 0.10)              # yellow paint band and pocket: the setting master, not a workpiece
TOL_RED = (0.85, 0.10, 0.08)


def _hull(pts, color):
    P = np.asarray(pts, float)
    tris = _Hull(P).simplices.astype(np.int64)
    c = P.mean(axis=0)
    for k, (a, b, d) in enumerate(tris):                 # outward winding so the computed normals face out
        n = np.cross(P[b] - P[a], P[d] - P[a])
        if np.dot(n, P[a] - c) < 0:
            tris[k] = (a, d, b)
    return _lk.Mesh(P, tris, color, "none")


def box(c, s, color, ch=0.0):
    """box centre c, size s; optional chamfer ch on all edges"""
    c = np.asarray(c, float); h = np.asarray(s, float) / 2
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                if ch <= 0:
                    pts.append(c + h * (sx, sy, sz))
                else:
                    for d in range(3):
                        v = h.copy(); v[d] -= ch
                        pts.append(c + v * (sx, sy, sz))
    return _hull(pts, color)


def cyl(c, r, ht, axis, color, n=32, ch=0.0, a0=0.0, a1=2 * math.pi):
    """cylinder (or sector a0..a1) centre c, radius r, height ht along axis X/Y/Z; chamfer ch on both rims"""
    c = np.asarray(c, float)
    ax = {"X": 0, "Y": 1, "Z": 2}[axis]
    u, v = [i for i in range(3) if i != ax]
    full = abs(a1 - a0) >= 2 * math.pi - 1e-9
    angs = [a0 + (a1 - a0) * k / n for k in range(n if full else n + 1)]
    pts = []
    for zz, rr in ((-ht / 2, r - ch), (-ht / 2 + ch, r), (ht / 2 - ch, r), (ht / 2, r - ch)) if ch > 0 else ((-ht / 2, r), (ht / 2, r)):
        for a in angs:
            p = c.copy(); p[ax] += zz; p[u] += rr * math.cos(a); p[v] += rr * math.sin(a)
            pts.append(p)
        if not full:
            p = c.copy(); p[ax] += zz; pts.append(p)
    return _hull(pts, color)


def ring(c, r_in, r_out, ht, axis, color, n=48):
    return [cyl(c, r_out, ht, axis, color, n=4, a0=2 * math.pi * k / n, a1=2 * math.pi * (k + 1) / n)
            for k in range(n)] if r_in <= 0 else _ring_segs(c, r_in, r_out, ht, axis, color, n)


def _ring_segs(c, r_in, r_out, ht, axis, color, n):
    c = np.asarray(c, float)
    ax = {"X": 0, "Y": 1, "Z": 2}[axis]
    u, v = [i for i in range(3) if i != ax]
    out = []
    for k in range(n):
        pts = []
        for a in (2 * math.pi * k / n, 2 * math.pi * (k + 1) / n):
            for rr in (r_in, r_out):
                for zz in (-ht / 2, ht / 2):
                    p = c.copy(); p[ax] += zz; p[u] += rr * math.cos(a); p[v] += rr * math.sin(a)
                    pts.append(p)
        out.append(_hull(pts, color))
    return out


def text_on_plane(s, origin, right, up, normal, px, depth, color):
    """block-pixel digits: origin = centre of the text on the surface, right/up unit vectors in the surface,
    normal = outward unit normal; pixels stand proud by depth"""
    o = np.asarray(origin, float); R = np.asarray(right, float); U = np.asarray(up, float); N = np.asarray(normal, float)
    ncol = len(s) * 4 - 1
    out = []
    for i, ch in enumerate(s):
        for r, row in enumerate(FONT[ch]):
            for cc, bit in enumerate(row):
                if bit != "1":
                    continue
                cx = (i * 4 + cc) * px - (ncol - 1) * px / 2
                cz = (2 - r) * px
                ctr = o + R * cx + U * cz + N * depth / 2
                pts = [ctr + R * sx * px / 2 + U * sz * px / 2 + N * sn * depth / 2
                       for sx in (-1, 1) for sz in (-1, 1) for sn in (-1, 1)]
                out.append(_hull(pts, color))
    return out


# ---------------------------------------------------------------------------------------------------------------------
def _stand(g):
    XS, YS, TOP, GH, GB = g["XS"], g["YS"], g["TOP"], g["GH"], g["GB"]
    GBX, GBY = g.get("GBX", GB), g.get("GBY", GB)
    XC, YC = g.get("XC", XS + 0.08), g.get("YCOL", g.get("YC", YS))
    COL_H = g.get("COL_H", 0.20)
    ANV = TOP + GH
    GHB = g.get("GHB", GH)                                                             # base slab (anvil raised above it)
    out = [box((XS, YS, TOP + GHB / 2 - 0.00005), (GBX + 0.0006, GBY + 0.0006, GHB + 0.0002), g.get("GRANITE", CAST), ch=0.003)]
    if GHB < GH:                                                                       # raised anvil: ground steel side
        out.append(cyl((XS, YS, TOP + GHB + (GH - GHB) / 2), g.get("ANV_R", 0.030) + 0.0003, GH - GHB + 0.0002, "Z", STEEL_BRIGHT, n=48, ch=0.0008))
    out.append(cyl((XS, YS, ANV + 0.00015), 0.030, 0.0003, "Z", LAPPED, n=48))          # ground anvil face (0.3 mm)
    out += ring((XS, YS, ANV + 0.0002), 0.030, 0.0315, 0.0004, "Z", (0.55, 0.56, 0.58), n=40)
    for k in range(-5, 6):                                                             # serrations on the anvil face
        out.append(box((XS + k * 0.005, YS, ANV + 0.00032), (0.0006, 2 * math.sqrt(max(0.0, 0.029 ** 2 - (k * 0.005) ** 2)), 0.0001), (0.50, 0.51, 0.53)))
    rng = np.random.default_rng(7)
    for _ in range(70 if g.get("SPECKLE", True) else 0):                               # granite speckle on the margin
        x, y = rng.uniform(-GBX / 2 + 0.006, GBX / 2 - 0.006), rng.uniform(-GBY / 2 + 0.006, GBY / 2 - 0.006)
        if math.hypot(x, y) < 0.04:
            continue
        out.append(box((XS + x, YS + y, TOP + GHB + 0.0001), (0.0016, 0.0013, 0.0002), SPECK))
    zc0 = TOP + GHB
    out.append(cyl((XC, YC, (zc0 + TOP + COL_H) / 2), 0.0130, TOP + COL_H - zc0, "Z", STEEL_BRIGHT, n=40))   # ground column
    out.append(cyl((XC, YC, TOP + COL_H - 0.002), 0.0133, 0.004, "Z", STEEL_BRIGHT, n=40, ch=0.0012))         # chamfered top
    out.append(cyl((XC, YC, zc0 + 0.004), 0.0205, 0.0082, "Z", CAST, n=40, ch=0.0015))                       # column foot
    for k in range(4):                                                                 # foot cap screws
        a = math.pi / 4 + k * math.pi / 2
        out.append(cyl((XC + 0.016 * math.cos(a), YC + 0.016 * math.sin(a), zc0 + 0.0084), 0.0022, 0.0012, "Z", KNOB_BLACK, n=12))
    return out


def _arm(g):
    XS, YS, DIAL_Z, DIAL_R, DIAL_T = g["XS"], g["YS"], g["DIAL_Z"], g["DIAL_R"], g["DIAL_T"]
    XC, YC = g.get("XC", XS), g.get("YCOL", YS + 0.065)
    KX, KZ = g.get("KNOB_X", XC - 0.030), g.get("KNOB_Z", DIAL_Z + 0.030)
    za = DIAL_Z + 0.030
    yf = YS - DIAL_T / 2                                                               # dial front plane (case)
    out = []
    ly = YC - YS - 0.008                                                               # cast arm along +y (encloses the bar)
    out.append(box((XS, (YS + YC) / 2 + 0.004, za), (0.0166, ly + 0.0006, 0.0146), CAST, ch=0.002))
    out.append(box((XC, YC, za), (0.0366, 0.0366, 0.0206), CAST, ch=0.0025))         # column clamp block
    out.append(cyl((XC + 0.0183 + 0.0055, YC, za), 0.0075, 0.010, "X", KNOB_BLACK, n=20, ch=0.0012))   # clamp lock knob
    out.append(cyl((XC + 0.0183 + 0.0008, YC, za), 0.0032, 0.0016, "X", STEEL_GROUND, n=12))
    # the stand's real knobs (KBC photo): fine adjust on TOP of the arm beside the column (vertical axis, untouched),
    # the column clamp on the side (above), and the indicator stem clamp at the arm's front end (horizontal axis)
    out.append(cyl((XS, YC - 0.026, za + 0.0073 + 0.005), 0.0075, 0.010, "Z", KNOB_BLACK, n=20, ch=0.0012))   # fine adjust
    out.append(cyl((XS, YC - 0.026, za + 0.0073 + 0.0002), 0.0030, 0.0008, "Z", STEEL_GROUND, n=12))
    out.append(cyl((XS + 0.0083 + 0.005, YS + 0.019, za), 0.0060, 0.009, "X", KNOB_BLACK, n=18, ch=0.0010))  # stem clamp
    # dial indicator body: chrome case + bezel + back
    out.append(cyl((XS, YS, DIAL_Z), DIAL_R + 0.0003, DIAL_T + 0.0004, "Y", CHROME_V, n=48))
    out.append(cyl((XS, YS + DIAL_T / 2 + 0.0008, DIAL_Z), DIAL_R - 0.002, 0.0016, "Y", KNOB_BLACK, n=40))  # back cover
    # stem and lug
    out.append(cyl((XS, YS, DIAL_Z - DIAL_R - 0.0075), 0.0046, 0.0150, "Z", CHROME_V, n=24))
    out.append(box((XS, YS + DIAL_T / 2 + 0.006, DIAL_Z), (0.012, 0.012, 0.0314), CHROME_V, ch=0.0015))
    return out


def _bezel(g):
    """the turning bezel: knurled ring, white face, graduations, numerals and the two tolerance markers all turn with
    it (world coords at rest; the needle is a separate part and does not turn with it)"""
    XS, YS, DIAL_Z, DIAL_R, DIAL_T = g["XS"], g["YS"], g["DIAL_Z"], g["DIAL_R"], g["DIAL_T"]
    yf = YS - DIAL_T / 2
    out = []
    out.append(cyl((XS, yf - 0.0005, DIAL_Z), DIAL_R - 0.0022, 0.0008, "Y", (0.96, 0.96, 0.94), n=48))   # white face
    out += ring((XS, yf - 0.0013, DIAL_Z), DIAL_R - 0.0022, DIAL_R + 0.0018, 0.0050, "Y", CHROME_V, n=48)   # bezel
    for k in range(60):                                                                # bezel knurl
        a = 2 * math.pi * k / 60
        out.append(box((XS + (DIAL_R + 0.0013) * math.cos(a), yf - 0.0005, DIAL_Z + (DIAL_R + 0.0013) * math.sin(a)),
                       (0.0007, 0.0024, 0.0007), STEEL_GROUND))
    for sg in (() if __import__("os").environ.get("DC_HIDE_MARKERS") else (-1, 1)):   # tolerance markers: +-2 divisions
                                                                                       # (DC_HIDE_MARKERS: obscured-dial negative test)
        a = sg * 2 * math.pi * 2 / 100
        for rr, ww in ((DIAL_R - 0.0004, 0.0022), (DIAL_R - 0.0020, 0.0012)):          # a red wedge clamped on the bezel
            out.append(box((XS + rr * math.sin(a), yf - 0.0032, DIAL_Z + rr * math.cos(a)), (ww, 0.0012, 0.0018), TOL_RED))
    # face: graduations (100) and numerals 0..90 between the face (front at yf - 0.001) and the needle
    yt = yf - 0.0012
    for k in range(100):
        a = 2 * math.pi * k / 100
        L = 0.0045 if k % 10 == 0 else (0.0033 if k % 5 == 0 else 0.0024)
        w = 0.00055 if k % 10 == 0 else 0.00035
        r_out = DIAL_R - 0.0032
        rc = r_out - L / 2
        cx, cz = XS + rc * math.sin(a), DIAL_Z + rc * math.cos(a)
        R = np.array([math.cos(a), 0, -math.sin(a)]); U = np.array([math.sin(a), 0, math.cos(a)]); N = np.array([0, -1.0, 0])
        ctr = np.array([cx, yt, cz])
        pts = [ctr + R * sx * w / 2 + U * su * L / 2 + N * sn * 0.0001 for sx in (-1, 1) for su in (-1, 1) for sn in (-1, 1)]
        out.append(_hull(pts, INK))
    for k in range(10):
        a = 2 * math.pi * k / 10
        rc = DIAL_R - 0.0118
        out += text_on_plane(str(k * 10), (XS + rc * math.sin(a), yt, DIAL_Z + rc * math.cos(a)),
                             (1, 0, 0), (0, 0, 1), (0, -1, 0), 0.00062, 0.0002, INK)
    out += text_on_plane("001", (XS, yt, DIAL_Z - 0.0095), (1, 0, 0), (0, 0, 1), (0, -1, 0), 0.00040, 0.0002, INK)
    return out


def _spindle(g):
    XS, YS, TIP0, DIAL_Z, DIAL_R = g["XS"], g["YS"], g["TIP0"], g["DIAL_Z"], g["DIAL_R"]
    out = [cyl((XS, YS, TIP0 + 0.0019), 0.0034, 0.0038, "Z", CARBIDE, n=24),            # flat contact point (bottom at the tip)
           cyl((XS, YS, TIP0 + 0.0060), 0.0042, 0.0045, "Z", STEEL_GROUND, n=6),        # hex of the contact point
           cyl((XS, YS, TIP0 + 0.0140), 0.0031, 0.0120, "Z", STEEL_BRIGHT, n=24)]       # spindle
    out.append(cyl((XS, YS, DIAL_Z + DIAL_R + 0.024), 0.0063, 0.0084, "Z", KNOB_BLACK, n=24, ch=0.0012))   # lift cap
    out.append(cyl((XS, YS, DIAL_Z + DIAL_R + 0.012), 0.0022, 0.018, "Z", STEEL_BRIGHT, n=16))
    return out


def _needle(g):
    XS, YS, DIAL_Z, DIAL_T = g["XS"], g["YS"], g["DIAL_Z"], g["DIAL_T"]
    y = YS - DIAL_T / 2 - 0.0012
    return [cyl((XS, y - 0.0009, DIAL_Z), 0.0021, 0.0012, "Y", INK, n=20),
            cyl((XS, y - 0.0016, DIAL_Z), 0.0009, 0.0006, "Y", (0.75, 0.12, 0.10), n=12)]


def _lever(g):
    XS, LEV_Y0, LEV_Z, LEV_L = g.get("XS"), g.get("LEV_Y0"), g.get("LEV_Z"), g.get("LEV_L")
    if None in (XS, LEV_Y0, LEV_Z, LEV_L):
        return []
    out = [box((XS, LEV_Y0 + LEV_L / 2, LEV_Z), (0.0066, LEV_L + 0.0006, 0.0046), STEEL_BRIGHT, ch=0.0008)]
    out.append(cyl((XS, LEV_Y0 + LEV_L - 0.006, LEV_Z + 0.003), 0.0102, 0.0047, "Z", KNOB_BLACK, n=24, ch=0.001))  # thumb pad
    out.append(cyl((XS, LEV_Y0, LEV_Z), 0.0026, 0.0110, "X", STEEL_GROUND, n=12))                  # pivot pin
    return out


def _knob(g):
    XC, YC = g.get("XC"), g.get("YCOL")
    if g.get("KNOB_X") is None or YC is None:
        return []
    p = (g["KNOB_X"], YC, g.get("KNOB_Z", g["DIAL_Z"] + 0.030))
    out = [cyl(p, 0.0103, 0.0121, "X", KNOB_BLACK, n=24, ch=0.0010)]
    for k in range(20):                                                                # knurl ridges along the axis (x)
        a = 2 * math.pi * k / 20
        out.append(box((p[0], p[1] + 0.0105 * math.cos(a), p[2] + 0.0105 * math.sin(a)), (0.0092, 0.0013, 0.0013), KNOB_BLACK))
    return out


def _gauge_block(g):
    out = [box((0, 0, 0.0125), (0.0095, 0.0355, 0.0254), STEEL_BRIGHT, ch=0.0005)]
    out.append(box((0, 0, 0.0045), (0.0099, 0.0359, 0.0050), MASTER_PAINT))                       # painted band, low
    out += text_on_plane("MASTER", (-0.00495, 0.0, 0.0190), (0, -1, 0), (0, 0, 1), (-1, 0, 0), 0.0012, 0.0002, INK)
    out += text_on_plane("25.000", (-0.00495, 0.0, 0.0120), (0, -1, 0), (0, 0, 1), (-1, 0, 0), 0.0009, 0.0002, INK)
    out += text_on_plane("MASTER", (0.00495, 0.0, 0.0190), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.0012, 0.0002, INK)
    return out


def _spacer(g, k):
    h = g["SPACER_H"][k]; SL = g.get("SL", 0.040)
    out = [box((0, 0, h / 2), (0.0105, SL + 0.0005, h + 0.0004), STEEL_GROUND, ch=0.0008)]
    out.append(cyl((0, -SL / 2 + 0.010, h + 0.00025), 0.0032, 0.0003, "Z", BORE, n=24))            # through bore (top)
    out += ring((0, -SL / 2 + 0.010, h + 0.0003), 0.0032, 0.0040, 0.0003, "Z", STEEL_BRIGHT, n=24)  # chamfer ring
    idx = sorted(g["SPACER_H"]).index(k) + 1
    out += text_on_plane(str(idx), (-0.00525, 0.004, h * 0.55), (0, -1, 0), (0, 0, 1), (-1, 0, 0), 0.0016, 0.0002,
                         (0.85, 0.15, 0.10))                                                        # paint-marked ID
    return out


def _parts_tray(g):
    X0, X1, W = g.get("PT_X0"), g.get("PT_X1"), g.get("PT_W")
    FL = g.get("PT_FLOOR", 0.0015)
    if None in (X0, X1, W):
        return []
    col = g.get("PTC", (0.20, 0.36, 0.62))
    cx = (X0 + X1) / 2; lx = X1 - X0
    out = []
    t, h = 0.002, 0.006
    for sy in (-1, 1):                                                                 # lip flange on the long rims
        out.append(box((0, sy * (W / 2 + 0.0015), h + 0.0005), (lx + 0.006, 0.005, 0.001), col, ch=0.0003))
    for sx in (-1, 1):
        out.append(box((sx * (lx / 2 + 0.0015), 0, h + 0.0005), (0.005, W + 0.006, 0.001), col, ch=0.0003))
    xs = list(g.get("PART_XS", (0.33, 0.38, 0.43, 0.48)))
    for i, x in enumerate(xs):                                                         # one pocket per part (master: yellow)
        if i == 0 and g.get("MASTER_POCKET"):
            mx, mlx, mly = g["MASTER_POCKET"]
            out.append(box((mx - cx, 0, FL + 0.0001), (mlx, mly, 0.0002), MASTER_PAINT))
        else:
            out.append(box((x - cx, 0, FL + 0.0001), (0.020, 0.050, 0.0002), MASTER_PAINT if i == 0 else POCKET))
    for i, (a, b) in enumerate(zip(xs[:-1], xs[1:])):                                  # low ribs between pockets
        out.append(box(((a + b) / 2 - cx, 0, FL + 0.00125), (0.0015 if i else 0.0030, W - 2 * t, 0.0025),
                       MASTER_PAINT if i == 0 else col, ch=0.0004))
    out += text_on_plane("MASTER", (-lx / 2 - 0.0015, 0.0, h + 0.0010), (0, -1, 0), (1, 0, 0), (0, 0, 1),
                         0.0008, 0.0002, INK)                                          # printed on the front lip, at the master pocket
    return out


def _tray(g, name):
    col = g.get("GREEN", (0.12, 0.55, 0.20)) if name == "tray_pass" else g.get("RED", (0.75, 0.12, 0.10))
    dark = tuple(c * 0.75 for c in col)
    w, d, h, t = 0.10, 0.08, 0.015, 0.003
    out = []
    for sy in (-1, 1):                                                                 # rolled moulded rim
        out.append(box((0, sy * (d / 2 + 0.002), h + 0.0008), (w + 0.010, 0.008, 0.0016), col, ch=0.0006))
    for sx in (-1, 1):
        out.append(box((sx * (w / 2 + 0.002), 0, h + 0.0008), (0.008, d + 0.010, 0.0016), col, ch=0.0006))
    for sx in (-1, 1):
        for sy in (-1, 1):                                                             # rounded corner posts
            out.append(cyl((sx * (w / 2 - t / 2), sy * (d / 2 - t / 2), h / 2), 0.0032, h, "Z", col, n=16))
    out.append(box((0, 0, t + 0.0001), (w - 2 * t - 0.004, d - 2 * t - 0.004, 0.0002), dark))   # inner floor
    return out


def get(key, g):
    if not _OK:
        return []
    try:
        if key == "stand":
            return _stand(g)
        if key == "arm":
            return _arm(g)
        if key == "spindle":
            return _spindle(g)
        if key == "needle":
            return _needle(g)
        if key == "lever":
            return _lever(g)
        if key == "bezel":
            return _bezel(g)
        if key == "knob":
            return _knob(g)
        if key == "gauge_block":
            return _gauge_block(g)
        if key.startswith("spacer_"):
            return _spacer(g, key[len("spacer_"):])
        if key == "parts_tray":
            return _parts_tray(g)
        if key in ("tray_pass", "tray_rework"):
            return _tray(g, key)
    except KeyError:
        return []
    return []
