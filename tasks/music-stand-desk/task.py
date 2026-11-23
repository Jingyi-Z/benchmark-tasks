"""music-stand GATE 1 coupon (claude-b50, own primitives): one desk wing of a folding music stand swings open on its own
linked members while the other arm steadies the stand.

Stand: loose, 1.55 kg, tripod base 69 cm, standing on the floor in front of the arms (the station table is replaced by a
short deck under the arm bases, cart_scene_fix). Desk 48.5 x 24 cm with a 4.2 cm ledge at 0.80 m (low seated height),
steel strips. Each wing is a parallelogram on real pivots: ledge arm hinged at the spine bottom, upper strut hinged at
the spine top, end bar hinged at the ledge-arm tip; the loop is closed by PhysX mimic constraints (strut follows the ledge
angle, end bar keeps vertical), which is exact for a parallelogram. Strut lengths and pivots are scaled to the published
desk size; the maker's drawing was not reachable. Folded = wing swung up against the spine, open = ledge horizontal.
Right hand pinches the right wing's end bar and carries it along its arc; left hand holds the stand tube."""
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401,E402
import cart_scene_fix  # noqa: F401,E402
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.ik import quat_xyzw_to_matrix as _q2m  # noqa: E402

SCENE = "station"
FLOOR = float(os.environ.get("MS_FLOOR", "0.0"))
RUN_STEPS = 700
labkit.CONTACT_OFFSET = 0.0005

BLACK = (0.05, 0.05, 0.055)
RUBBER = (0.02, 0.02, 0.02)
labkit.SURFACES.update({tuple(round(c, 4) for c in BLACK): {"roughness": 0.35, "metallic": 0.6}})

SX, SY = 0.58, 0.0                 # stand (tube) position on the floor
P0 = 0.80                          # ledge height above the floor
H = 0.24                           # desk height
HALF = 0.2425                      # half the 48.5 cm desk width
XD = -0.03                         # desk plane, in front of the tube (toward the arms)
T = 0.003                          # strip thickness
YP = 0.015                         # wing pivots sit at |y| = 15 mm (edge of the 30 mm spine)
ARM = HALF - YP - 0.0075           # pivot-to-pivot length of the ledge arm and the strut (0.22 m)
ZA = P0 + 0.0125                   # ledge-arm centre line (25 mm strip)
ZB = P0 + H - 0.0075               # strut centre line (15 mm strip)
LINK = ZB - ZA                     # end-bar pivot spacing (= ZB - ZA, parallelogram)
TILT = math.radians(float(os.environ.get("MS_TILT_DEG", "20")))   # desk tilted back (K&M detents every 10 deg)
KP = np.array([XD, 0.0, P0 - 0.015])                             # tilt pivot at the head joint (stand frame)
_RT = np.array([[math.cos(TILT), 0, math.sin(TILT)], [0, 1, 0], [-math.sin(TILT), 0, math.cos(TILT)]])


def tp(p):
    """A desk point (stand frame, untilted) moved to the tilted desk."""
    return tuple((KP + _RT @ (np.asarray(p, float) - KP)).tolist())


def tboxes(bs):
    return [(tp(c), sz, col, (0.0, TILT)) for c, sz, col in bs]


def _hull(points, color, collide="hull"):
    from scipy.spatial import ConvexHull
    p = np.asarray(points, float)
    return labkit.Mesh(p, ConvexHull(p).simplices.astype(np.int64), color, collide)


def _rod(p0, p1, r, color, collide="hull", n=10):
    """A round rod from p0 to p1 as a convex hull (legs, braces)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(a, [0, 0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    ring = [r * (math.cos(t) * u + math.sin(t) * v) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    return _hull([p0 + q for q in ring] + [p1 + q for q in ring], color, collide)


def _legs():
    """Tripod: three round legs from the collar down to rubber feet at 34.5 cm radius, plus three short braces."""
    meshes, boxes = [], []
    for k in range(3):
        a = math.radians(120 * k)
        foot = np.array([0.345 * math.cos(a), 0.345 * math.sin(a), 0.012])
        meshes.append(_rod((0.012 * math.cos(a), 0.012 * math.sin(a), 0.24), foot, 0.0055, BLACK))
        meshes.append(_rod((0.0, 0.0, 0.30), 0.55 * foot + np.array([0, 0, 0.11]), 0.0035, BLACK, collide="none"))
        boxes.append(((foot[0], foot[1], 0.006), (0.022, 0.022, 0.012), RUBBER))
    return meshes, boxes


_LEG_MESH, _LEG_BOX = _legs()
_LYRE = [_rod(tp((XD - 0.002, s * 0.004, P0 + H - 0.005)), tp((XD - 0.002, s * 0.014, P0 + H + 0.035)), 0.0015, BLACK, "none")
         for s in (-1, 1)]


BODY_BOXES = _LEG_BOX + [
    ((XD / 2, 0, P0 - 0.015), (abs(XD) + 0.01, 0.03, 0.02), BLACK),                     # head: tube to desk
] + tboxes([((XD, 0, P0 + H / 2), (T, 0.03, H), BLACK),                                # spine
             ((XD - 0.021, 0, P0 - 0.0015), (0.042, 0.03, T), BLACK)])                     # centre ledge
_ZT = 0.22 + 0.5 * (P0 - 0.03 - 0.22)
BODY_CYLS = [((0, 0, (0.22 + _ZT) / 2), 0.010, _ZT - 0.22, BLACK, "Z"),                  # lower tube (20 mm)
             ((0, 0, (_ZT + P0 - 0.03) / 2), 0.0075, P0 - 0.03 - _ZT, BLACK, "Z"),     # upper tube (15 mm)
             ((0, 0, _ZT), 0.012, 0.025, BLACK, "Z"),                                  # height clamp collar
             ((0, 0, 0.24), 0.016, 0.03, BLACK, "Z")]                                  # leg collar


def _wing(side):
    """side -1 = right wing (toward -y), +1 = left wing. Shapes at the OPEN pose."""
    s = side
    n = "r" if s < 0 else "l"
    lo, hi = (-math.pi / 2, 0.0) if s < 0 else (0.0, math.pi / 2)
    folded = lo if s < 0 else hi
    start = os.environ.get("MS_START", "folded")       # teleport tests: open, right (right wing open), folded
    if start == "open" or (start == "right" and s < 0):
        folded = 0.0
    ym = s * (YP + ARM / 2)
    ledge = Joint(f"{n}_ledge", kind="hinge", axis="X", anchor=tp((XD - T, s * YP, ZA)),
                  boxes=tboxes([((XD - T, ym, ZA), (T, ARM + 0.015, 0.025), BLACK),          # ledge arm
                                ((XD - T - 0.021, ym, P0 - 0.0015), (0.042, ARM + 0.015, T), BLACK)]),  # its ledge lip
                  lower=lo, upper=hi, initial=folded, target=folded if os.environ.get("MS_HOLD", "1") == "1" else 0.0,
                  stiffness=0.0, damping=0.05, friction=float(os.environ.get("MS_FRIC", "0.3")), mass=0.12)
    endb = Joint(f"{n}_end", kind="hinge", axis="X", anchor=tp((XD - 2 * T, s * (YP + ARM), ZA)), parent=f"{n}_ledge",
                 boxes=tboxes([((XD - 2 * T, s * (YP + ARM), ZA + LINK / 2), (T, 0.015, LINK + 0.02), BLACK)]),
                 lower=-hi, upper=-lo, initial=-folded, target=-folded, stiffness=0.0, damping=0.02, mass=0.05)
    strut = Joint(f"{n}_strut", kind="hinge", axis="X", anchor=tp((XD - T, s * YP, ZB)),
                  boxes=tboxes([((XD - T, ym, ZB), (T, ARM + 0.015, 0.015), BLACK)]),
                  lower=lo, upper=hi, initial=folded, target=folded, stiffness=0.0, damping=0.02, mass=0.05)
    # diagonal brace 5.k in two halves with a knee: upper half hinged at the centre-brace top pivot, lower half at the
    # outer bottom corner (riding on the ledge arm); the knee is closed by a loop joint. Straight (both halves at their
    # 0 stop) locks the open rectangle; folded, both halves hang down along the spine.
    xdg = XD - 4 * T
    yc, zc = s * (YP + ARM), ZA                       # outer bottom corner (open)
    yp, zp = s * YP, ZB                               # centre top pivot
    ym_, zm_ = (yc + yp) / 2, (zc + zp) / 2           # knee (open)
    dl = math.hypot(yc - yp, zc - zp) / 2
    fu = (math.pi / 4) if s < 0 else (-math.pi / 4)   # folded angles (see notes in the 3:58 PM queue post)
    fl = -fu
    if start == "open" or (start == "right" and s < 0):
        fu = fl = 0.0
    def _strip(p0, p1, w=0.012):
        import numpy as _np
        p0, p1 = _np.asarray(p0, float), _np.asarray(p1, float)
        c = (p0 + p1) / 2
        ang = math.atan2(p1[2] - p0[2], p1[1] - p0[1])
        L = float(_np.linalg.norm(p1 - p0))
        pts = []
        for dx in (-T / 2, T / 2):
            for a_ in (-L / 2, L / 2):
                for b_ in (-w / 2, w / 2):
                    pts.append((c[0] + dx, c[1] + a_ * math.cos(ang) - b_ * math.sin(ang), c[2] + a_ * math.sin(ang) + b_ * math.cos(ang)))
        return _hull([tp(q) for q in pts], BLACK)
    dup = Joint(f"{n}_dup", kind="hinge", axis="X", anchor=tp((xdg, yp, zp)), meshes=[_strip((xdg, yp, zp), (xdg, ym_, zm_))],
                lower=min(0.0, fu), upper=max(0.0, fu), initial=fu, target=fu, stiffness=0.0, damping=0.02, friction=float(os.environ.get("MS_DFRIC", "0.0")), mass=0.03)
    dlo = Joint(f"{n}_dlo", kind="hinge", axis="X", anchor=tp((xdg, yc, zc)), parent=f"{n}_ledge",
                meshes=[_strip((xdg, yc, zc), (xdg, ym_, zm_))],
                lower=min(0.0, fl), upper=max(0.0, fl), initial=fl, target=fl, stiffness=0.0, damping=0.02, friction=float(os.environ.get("MS_DFRIC", "0.0")), mass=0.03)
    KNEES.append((n, (xdg, ym_ - yp, zm_ - zp), (xdg, ym_ - yc, zm_ - zc), dl))
    mimic = [(f"{n}_end", f"{n}_ledge", 1.0, 0.0),        # q_end + q_ledge = 0: end bar stays vertical
             (f"{n}_strut", f"{n}_ledge", -1.0, 0.0)]     # q_strut - q_ledge = 0: strut parallel to the ledge arm
    return [ledge, endb, strut, dup, dlo], mimic


KNEES: list = []
_jr, _mr = _wing(-1)
_jl, _ml = _wing(+1)
LOOP = os.environ.get("MS_LOOP", "1") == "1"
labkit.JOINT_FRAME_PITCH["music_stand"] = TILT
if not LOOP:
    labkit.PART_MIMIC["music_stand"] = _mr + _ml   # diagnostic only (codex 1511: an angle rule cannot show the move)
else:
    # real loop closure (claude-ebeam 1513): the end bar's upper pivot is pinned to the strut tip by a revolute joint
    # solved outside the articulation tree; each wing's members are free on their own pivots otherwise
    _orig_author = labkit._author

    def _author_loop(prim_path, cfg, *a, **k):
        prim = _orig_author(prim_path, cfg, *a, **k)
        if getattr(cfg, "part_name", None) == "music_stand":
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import Gf, Sdf, UsdPhysics
            st = get_current_stage()
            for n, s_ in (("r", -1), ("l", 1)):
                jp = UsdPhysics.RevoluteJoint.Define(st, f"{prim_path}/{n}_end/loop_close")
                jp.CreateAxisAttr("X")
                jp.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/{n}_end")])
                jp.CreateBody1Rel().SetTargets([Sdf.Path(f"{prim_path}/{n}_strut")])
                q_ = Gf.Quatf(math.cos(TILT / 2), 0.0, math.sin(TILT / 2), 0.0)
                jp.CreateLocalPos0Attr(Gf.Vec3f(*(_RT @ np.array([0.0, 0.0, LINK])).tolist()))           # end-bar upper pivot
                jp.CreateLocalRot0Attr(q_)
                jp.CreateLocalPos1Attr(Gf.Vec3f(*(_RT @ np.array([-T, s_ * ARM, 0.0])).tolist()))      # strut tip
                jp.CreateLocalRot1Attr(q_)
                jp.CreateExcludeFromArticulationAttr(True)
            for n, _rel_u, _rel_l, _dl in KNEES:          # knee of each diagonal brace: upper-half tip = lower-half tip
                kj = UsdPhysics.RevoluteJoint.Define(st, f"{prim_path}/{n}_dlo/knee")
                kj.CreateAxisAttr("X")
                kj.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/{n}_dup")])
                kj.CreateBody1Rel().SetTargets([Sdf.Path(f"{prim_path}/{n}_dlo")])
                q_ = Gf.Quatf(math.cos(TILT / 2), 0.0, math.sin(TILT / 2), 0.0)
                kj.CreateLocalPos0Attr(Gf.Vec3f(*(_RT @ np.array([0.0, _rel_u[1], _rel_u[2]])).tolist()))
                kj.CreateLocalRot0Attr(q_)
                kj.CreateLocalPos1Attr(Gf.Vec3f(*(_RT @ np.array([0.0, _rel_l[1], _rel_l[2]])).tolist()))
                kj.CreateLocalRot1Attr(q_)
                kj.CreateExcludeFromArticulationAttr(True)
            print("MSLOOP authored 2 loop joints + 2 knees on " + prim_path, flush=True)
        return prim

    labkit._author = _author_loop
STAND = part("music_stand", pos=(SX, SY, FLOOR), roll=math.radians(float(os.environ.get("MS_ROLL_DEG", "0"))), mass=1.10, friction=0.8, boxes=BODY_BOXES, cylinders=BODY_CYLS,
             meshes=_LEG_MESH + _LYRE,
             joints=_jr + _jl)
OBJECTS = [STAND]
BOOK = None
if os.environ.get("MS_BOOK") == "1":
    # support check (codex 1539): a standard 9 x 12 in music book (229 x 305 x 5 mm, 300 g) set onto the desk, its lower
    # edge on the ledge lip and its back against the desk face; it must stay there
    _bc = tp((XD - 2 * T - 0.0015 - 0.0025 - 0.0005, 0.0, P0 + 0.0005 + 0.1525))
    BOOK = part("score_book", pos=(SX + _bc[0], SY + _bc[1], FLOOR + _bc[2]), mass=0.30, friction=0.5,
                boxes=[((0, 0, 0), (0.005, 0.229, 0.305), (0.80, 0.78, 0.72), (0.0, TILT))])
    OBJECTS.append(BOOK)
elif os.environ.get("MS_BOOK") == "open":
    # an open score: two 229 x 305 x 3 mm halves (150 g each) hinged at the spine, laid open across the desk. On the
    # deployed desk both halves rest on the wings; with nothing behind them they swing back about the spine.
    _bc = tp((XD - 2 * T - 0.0015 - 0.0015 - 0.0005, 0.0, P0 + 0.0005 + 0.1525))
    labkit.JOINT_FRAME_PITCH["score_book"] = TILT
    _half = Joint("right_half", kind="hinge", axis="Z", anchor=(0.0, 0.0, 0.0),
                  boxes=[((0.0, -0.1145, 0.0), (0.003, 0.229, 0.305), (0.82, 0.80, 0.74), (0.0, TILT))],
                  lower=-math.pi / 2, upper=math.pi / 2, initial=0.0, target=0.0, stiffness=0.0, damping=0.002,
                  friction=0.002, mass=0.15)
    BOOK = part("score_book", pos=(SX + _bc[0], SY + _bc[1], FLOOR + _bc[2]), mass=0.15, friction=0.5,
                boxes=[((0.0, 0.1145, 0.0), (0.003, 0.229, 0.305), (0.82, 0.80, 0.74), (0.0, TILT))], joints=[_half])
    OBJECTS.append(BOOK)

# A soprano ukulele (modelled here from published soprano dimensions: overall 530 mm, body 240 mm long, lower bout
# 170 mm, upper bout 135 mm, depth 60 mm, scale 345 mm, 12 frets on the neck, about 0.38 kg, mahogany) standing upright
# on a small A-frame floor stand directly in front of the music stand, face toward the arms. It is not to be touched.
UKE = os.environ.get("MS_UKE", "1") == "1"
UX, UY = float(os.environ.get("MS_UKX", "0.42")), float(os.environ.get("MS_UKY", "0.0"))
UKE_BASE = FLOOR + float(os.environ.get("MS_UKBASE", "0.20"))   # lower edge of the body sits in the stand's cradle
UKE_LEAN = math.radians(float(os.environ.get("MS_UKLEAN", "10")))   # leaning back (toward the music stand)
WOODC, DARKW, BONE, NICKEL, SOUND = (0.55, 0.32, 0.16), (0.20, 0.12, 0.07), (0.92, 0.89, 0.80), (0.75, 0.75, 0.77), (0.07, 0.04, 0.02)


def _hullmesh(pts, color, collide="none"):
    from scipy.spatial import ConvexHull
    p = np.asarray(pts, float)
    return labkit.Mesh(points=p, tris=ConvexHull(p).simplices.astype(np.int64), color=color, collide=collide)


def _uke_meshes():
    """Ukulele in its own frame: origin at the bottom of the body, +z up the neck, -x toward the arms (the face)."""
    D, n = 0.060, 28
    ms = []

    def ell(cz, hw, hh, x0, x1, color, collide):
        pts = [(x, hw * math.cos(t), cz + hh * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)
               for x in (x0, x1)]
        return _hullmesh(pts, color, collide)
    ms.append(ell(0.072, 0.085, 0.072, -D / 2, D / 2, WOODC, "hull"))            # lower bout (170 mm)
    ms.append(ell(0.178, 0.0675, 0.062, -D / 2, D / 2, WOODC, "hull"))           # upper bout (135 mm)
    ms.append(ell(0.072, 0.0835, 0.0705, -D / 2 - 0.0006, -D / 2, DARKW, "none"))   # face edge binding hints
    ms.append(ell(0.072, 0.081, 0.068, -D / 2 - 0.0012, -D / 2 - 0.0006, WOODC, "none"))
    ms.append(ell(0.178, 0.0655, 0.060, -D / 2 - 0.0012, -D / 2 - 0.0006, WOODC, "none"))
    ms.append(ell(0.150, 0.025, 0.025, -D / 2 - 0.0016, -D / 2 - 0.0010, SOUND, "none"))   # sound hole 50 mm
    ms.append(ell(0.150, 0.029, 0.029, -D / 2 - 0.0014, -D / 2 - 0.0011, DARKW, "none"))   # rosette ring

    def box(c, sz, color, collide="none"):
        pts = [(c[0] + sx * sz[0] / 2, c[1] + sy * sz[1] / 2, c[2] + sz2 * sz[2] / 2)
               for sx in (-1, 1) for sy in (-1, 1) for sz2 in (-1, 1)]
        return _hullmesh(pts, color, collide)
    ms.append(box((-D / 2 - 0.004, 0, 0.050), (0.008, 0.070, 0.010), DARKW))           # bridge
    ms.append(box((-D / 2 - 0.009, 0, 0.051), (0.002, 0.050, 0.002), BONE))            # saddle
    # neck and fretboard (body join at 0.236 m), nut at 0.236 + 0.175 m
    ms.append(_hullmesh([(x, y, z) for x in (-0.010, 0.012) for (y, z) in
                         ((-0.019, 0.236), (0.019, 0.236), (-0.0175, 0.411), (0.0175, 0.411))], WOODC, "hull"))
    ms.append(_hullmesh([(x, y, z) for x in (-0.015, -0.010) for (y, z) in
                         ((-0.019, 0.200), (0.019, 0.200), (-0.0175, 0.411), (0.0175, 0.411))], DARKW, "none"))
    scale = 0.345
    for k in range(1, 13):                                                              # 12 frets on the neck
        dz = scale * (1 - 2 ** (-k / 12.0))
        z = 0.411 - dz
        if z > 0.200:
            ms.append(box((-0.0158, 0, z), (0.0016, 0.036, 0.0012), NICKEL))
    ms.append(box((-0.0158, 0, 0.412), (0.0022, 0.036, 0.004), BONE))                   # nut
    ms.append(_hullmesh([(x, y, z) for x in (-0.006, 0.010) for (y, z) in
                         ((-0.022, 0.413), (0.022, 0.413), (-0.026, 0.525), (0.026, 0.525))], DARKW, "hull"))   # headstock
    for side in (-1, 1):                                                                # 4 tuners
        for z in (0.450, 0.495):
            ms.append(box((0.002, side * 0.034, z), (0.010, 0.016, 0.008), NICKEL))
    for y in (-0.0135, -0.0045, 0.0045, 0.0135):                                        # 4 strings
        ms.append(box((-0.0185, y * 1.0, 0.230), (0.0006, 0.0008, 0.360), BONE))
    return ms


def _stand_boxes():
    """Ukulele floor stand (black steel tube): feet, an upright behind the body, a cradle under the body and a padded
    yoke that the back of the neck rests in."""
    BK = (0.06, 0.06, 0.065)
    cz = UKE_BASE - FLOOR - 0.006                                   # cradle top under the body
    yz = cz + 0.300                                                 # yoke height (behind the neck)
    yx = 0.010 + 0.300 * math.tan(UKE_LEAN) + 0.016                 # just behind the leaned neck
    return [((0.0, 0.0, 0.004), (0.22, 0.22, 0.008), BK),                       # floor feet
            ((0.045, 0.0, cz / 2), (0.014, 0.014, cz), BK),                     # lower upright
            ((-0.010, -0.06, cz - 0.006), (0.09, 0.012, 0.012), BK),            # cradle arms (padded)
            ((-0.010, 0.06, cz - 0.006), (0.09, 0.012, 0.012), BK),
            ((0.045 + (yx - 0.045) / 2, 0.0, cz + 0.15), (abs(yx - 0.045) + 0.014, 0.014, 0.30), BK),   # upper upright
            ((yx, -0.026, yz), (0.012, 0.010, 0.040), BK),                      # yoke forks either side of the neck
            ((yx, 0.026, yz), (0.012, 0.010, 0.040), BK),
            ((yx + 0.006, 0.0, yz - 0.012), (0.010, 0.062, 0.012), BK)]         # yoke back (neck rests on it)


def _case_half(y):
    """Half-width of a soprano ukulele hardshell case outline at y (long axis), outside, m."""
    h = 0.0
    for c, ay, ax in ((-0.175, 0.125, 0.120), (-0.005, 0.090, 0.095)):   # lower bout, upper bout
        u = (y - c) / ay
        if abs(u) < 1:
            h = max(h, ax * math.sqrt(1 - u * u))
    if 0.04 <= y <= 0.31:
        h = max(h, 0.055 + 0.007 * (y - 0.04) / 0.27)                    # neck section, wider at the head end
    return h


def _case_meshes():
    """Shaped hardshell case lying flat, lid hinged on the arm side and standing open at CASE_LID deg, with the
    ukulele lying face up in its lining. Origin: case centre on the table top; +y toward the head end."""
    RIM, FL, T = 0.075, 0.012, 0.010
    lid = math.radians(float(os.environ.get("MS_CASELID", "-1")))
    ys = list(np.linspace(-0.299, 0.31, 40))
    out = []

    def prism(poly, z0, z1, color, collide="hull"):
        return _hullmesh([(x, y, z) for (x, y) in poly for z in (z0, z1)], color, collide)

    def seg_walls(z0, z1, color, tf=lambda p: p):
        ms = []
        for side in (-1, 1):
            for y0, y1 in zip(ys[:-1], ys[1:]):
                h0, h1 = _case_half(y0), _case_half(y1)
                q = [(side * h0, y0), (side * h1, y1), (side * max(h1 - T, 0), y1), (side * max(h0 - T, 0), y0)]
                ms.append(_hullmesh([tf((x, y, z)) for (x, y) in q for z in (z0, z1)], color, "hull"))
        h = _case_half(0.31)
        ms.append(_hullmesh([tf((x, y, z)) for x in (-h, h) for y in (0.30, 0.31) for z in (z0, z1)], color, "hull"))
        return ms
    # base tray: shell floor, lining, walls
    for c, ay, ax in ((-0.175, 0.125, 0.120), (-0.005, 0.090, 0.095)):
        e = [(ax * math.cos(t), c + ay * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
        out.append(prism(e, 0.0, FL - 0.002, BLK))
        out.append(prism([(0.93 * x, c + 0.95 * (y - c)) for x, y in e], FL - 0.002, FL, LINE, "none"))
    nk = [(-0.055, 0.04), (0.055, 0.04), (0.062, 0.31), (-0.062, 0.31)]
    out.append(prism(nk, 0.0, FL - 0.002, BLK))
    out.append(prism([(0.88 * x, min(y, 0.30)) for x, y in nk], FL - 0.002, FL, LINE, "none"))
    out += seg_walls(0.0, RIM, BLK)
    for yl in (-0.17, -0.02):                                             # latches on the stand side
        out.append(prism([(_case_half(yl) - 0.002, yl - 0.012), (_case_half(yl) + 0.006, yl - 0.012),
                          (_case_half(yl) + 0.006, yl + 0.012), (_case_half(yl) - 0.002, yl + 0.012)],
                         RIM - 0.022, RIM - 0.004, METAL, "none"))
    # lid: same outline, hinged along x = -0.12 at the rim, opened about the y axis toward the arms
    x0, z0, cl, sl = -0.120, RIM, math.cos(lid), math.sin(lid)

    mir = -1.0 if os.environ.get("MS_CASEHINGE", "arm") == "stand" else 1.0   # hinge on the stand side: mirrored

    def rot(p):
        x, y, z = p
        x = mir * x
        dx, dz = x - x0, z - z0
        return (mir * (x0 + dx * cl - dz * sl), y, z0 + dx * sl + dz * cl)
    if lid < 0:                                                           # MS_CASELID=-1: lid off, set aside
        nolid = True
    else:
        nolid = False
        out += seg_walls(0.0, 0.020, BLK, rot)
    for c, ay, ax in ([] if nolid else ((-0.175, 0.125, 0.120), (-0.005, 0.090, 0.095))):
        e = [(ax * math.cos(t), c + ay * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
        out.append(_hullmesh([rot((x, y, z)) for (x, y) in e for z in (0.020, 0.028)], BLK, "hull"))
        out.append(_hullmesh([rot((0.93 * x, c + 0.95 * (y - c), z)) for (x, y) in e for z in (0.019, 0.0195)],
                             LINE, "none"))
    if not nolid:
        out.append(_hullmesh([rot((x, y, z)) for (x, y) in nk for z in (0.020, 0.028)], BLK, "hull"))
        out.append(_hullmesh([rot((0.88 * x, min(y, 0.30), z)) for (x, y) in nk for z in (0.019, 0.0195)], LINE,
                             "none"))
    # the ukulele, face up, body end toward -y (uke frame: +z up the neck, -x the face)
    R = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)
    off = np.array([0.0, -0.28, FL + 0.030])
    for m in _uke_meshes():
        out.append(labkit.Mesh(points=m.points @ R.T + off, tris=m.tris, color=m.color, collide="none"))
    return out


UKMODE = os.environ.get("MS_UKMODE", "case")
TBL_H = float(os.environ.get("MS_TBLH", "0.625"))        # side table top above the floor
CASE_L, CASE_W, CASE_D = 0.62, 0.24, 0.095              # soprano hardshell case, outside
if UKE and UKMODE == "case":
    WOOD2, LEGC = (0.62, 0.47, 0.30), (0.20, 0.20, 0.21)
    tx = float(os.environ.get("MS_TBLX", "0.36"))
    SIDE_TABLE = part("side_table", pos=(tx, UY, FLOOR), fixed=True, friction=0.6,
                      boxes=[((0, 0, TBL_H - 0.012), (0.36, 0.70, 0.024), WOOD2)] +
                            [((sx * 0.15, sy * 0.25, (TBL_H - 0.024) / 2), (0.03, 0.03, TBL_H - 0.024), LEGC)
                             for sx in (-1, 1) for sy in (-1, 1)])
    BLK, METAL, LINE = (0.05, 0.05, 0.055), (0.70, 0.70, 0.72), (0.10, 0.14, 0.38)
    cx = tx + 0.18 - CASE_W / 2 - 0.01                  # at the table edge nearest the music stand
    UKULELE = part("ukulele", pos=(cx, UY, TBL_H + 0.001), mass=1.35, friction=0.7, meshes=_case_meshes())
    OBJECTS += [SIDE_TABLE, UKULELE]
if UKE and UKMODE == "stand":
    UK_STAND = part("uke_stand", pos=(UX, UY, FLOOR), fixed=True, friction=0.8, boxes=_stand_boxes())
    _cl, _sl = math.cos(UKE_LEAN), math.sin(UKE_LEAN)
    _RL = np.array([[_cl, 0, _sl], [0, 1, 0], [-_sl, 0, _cl]])          # lean the neck back toward the music stand
    _UM = [labkit.Mesh(points=m.points @ _RL.T, tris=m.tris, color=m.color, collide=m.collide) for m in _uke_meshes()]
    UKULELE = part("ukulele", pos=(UX - 0.01, UY, UKE_BASE + 0.002), mass=0.38, friction=0.7, meshes=_UM)
    OBJECTS += [UK_STAND, UKULELE]

_log: dict = {}


@on_reset
def _reset(env):
    _log.clear()
    _log.update(step=0, p0=None)


@on_step
def _track(env):
    _log["step"] += 1
    st = labkit._part_state(env, STAND)
    if _log["p0"] is None:
        _log["p0"] = np.asarray(st.pos, float).copy()
    R = st.rotation()
    tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
    j = {k: round(math.degrees(float(v)), 2) for k, v in st.joints.items()}
    if _log["step"] % 5 == 0 or _log["step"] <= 2:
        tips = None
        try:
            from yam_arena import runtime as _rt
            o = env.unwrapped.scene.env_origins[0].cpu().numpy()
            tips = [[round(float(x), 3) for x in (_rt.pad_tips_world(env.unwrapped.scene[n]) - o)] for n in ("left_robot", "right_robot")]
            fr = [round(float(v), 1) for v in env.unwrapped.scene["right_robot"].data.applied_torque[0, 6:8].tolist()]
            fl = [round(float(v), 1) for v in env.unwrapped.scene["left_robot"].data.applied_torque[0, 6:8].tolist()]
        except Exception:
            fr = fl = None
        if BOOK is not None:
            bk = labkit._part_state(env, BOOK)
            _log.setdefault("b0", np.asarray(bk.pos, float).copy())
            hj = {k: round(math.degrees(float(v)), 1) for k, v in bk.joints.items()}
            print("MSBOOK " + json.dumps(dict(step=_log["step"], moved_mm=round(float(np.linalg.norm(np.asarray(bk.pos) - _log["b0"])) * 1e3, 1),
                                              z=round(float(bk.pos[2]), 4), book_tilt_deg=round(math.degrees(math.acos(float(np.clip(bk.rotation()[2, 2], -1, 1)))), 1),
                                              half_deg=hj)), flush=True)
        print("MSDBG " + json.dumps(dict(step=_log["step"], joints_deg=j, tilt_deg=round(tilt, 2),
                                         base_xyz_mm=[round(float(v) * 1e3, 1) for v in (np.asarray(st.pos) - _log["p0"])],
                                         base_z=round(float(st.pos[2]), 4), pads=tips, f_right=fr, f_left=fl)), flush=True)


# --- demo -------------------------------------------------------------------------------------------------------------
TIP = 0.0128
OPEN_WIDE, SQUEEZE, SQUEEZE_HARD = -0.024, -0.010, -0.004


def _Rx(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def _Ry(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def _Rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


class Orient(float):
    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_alt", False):
    def _grip_rot(yaw=0.0, _prev=_prev_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rot._alt = True
    _demo.grip_rotation = _grip_rot

_D = _q2m(np.asarray(_demo.DOWN_GRIP))


def _lip(side, phi):
    """TCP pose pinching a wing's support-angle lip 16 cm from its pivot (side -1 right, +1 left; phi = ledge angle)."""
    r_lip = float(os.environ.get("MS_LIP_R", "0.16"))
    a = math.radians(phi)
    dy, dz = side * r_lip, -0.014
    y = side * YP + dy * math.cos(a) - dz * math.sin(a)
    z = ZA + dy * math.sin(a) + dz * math.cos(a)
    pw = np.array(tp((XD - T - 0.032, y, z))) + np.array([SX, SY, FLOOR])
    Rr = Orient(_RT @ _Rx(a - side * math.pi / 2) @ _Ry(math.radians(-90)) @ _D)
    return pw - Rr.rot @ np.array([0.0, 0.0, TIP]), Rr


def _tube():
    # level grip pointing at the tube; the wrist is rolled about the approach axis so the wrist camera sits beside the
    # tube looking past it, not pressed against the black folded desk
    roll = math.radians(float(os.environ.get("MS_TUBEROLL", "0")))
    Rl = Orient(_Ry(math.radians(-90)) @ _D @ _Rz(roll))
    tz = float(os.environ.get("MS_TUBEZ", "0.745" if UKE else "0.68"))     # grip above the ukulele's headstock
    return np.array([SX, SY, FLOOR + tz]) - Rl.rot @ np.array([0.0, 0.0, TIP]), Rl


class _Track:
    """Waypoints given by absolute end times."""

    def __init__(self, start):
        self.w, self.t, self.last = [], 0.0, start

    def to(self, t_end, pos, finger, label, rot=0.0):
        self.w.append(Waypoint(max(0.05, t_end - self.t), tuple(np.asarray(pos, float)), finger, label, yaw=rot))
        self.t, self.last = max(t_end, self.t + 0.05), (pos, finger, rot)

    def hold(self, t_end, label):
        pos, finger, rot = self.last
        self.to(t_end, pos, finger, label, rot)


def _demo_both():
    """Both wings, one at a time: the free hand holds the stand tube while the other arm swings its own wing open."""
    parked_l, parked_r = (0.21, 0.28, 0.93), (0.21, -0.28, 0.93)
    tube, Rt = _tube()
    L, R = _Track(parked_l), _Track(parked_r)
    # phase A: left hand holds the tube, right arm opens the right wing
    L.to(2.0, tube + [-0.08, 0, 0], OPEN_WIDE, "left hand in front of the stand tube", Rt)
    L.to(3.5, tube, OPEN_WIDE, "jaws either side of the tube", Rt)
    L.to(4.5, tube, SQUEEZE, "hold the stand tube", Rt)
    p, Rr = _lip(-1, -90.0)
    R.to(2.0, p + [-0.07, 0, 0], OPEN_WIDE, "right hand in front of the folded right wing", Rr)
    R.to(3.5, p, OPEN_WIDE, "jaws either side of the right ledge lip", Rr)
    R.to(4.5, p, 0.0, "pinch the right ledge lip", Rr)
    t = 4.5
    for k, phi in enumerate((-72.0, -54.0, -36.0, -18.0, 0.0, 4.0)):
        t += 0.8
        p, Rr = _lip(-1, phi)
        R.to(t, p, 0.0, "swing the right wing open (%d/6)" % (k + 1), Rr)
    R.hold(t + 1.0, "press the right wing onto its stop so the diagonal brace locks straight")
    R.to(t + 1.6, p, OPEN_WIDE, "let go of the right wing", Rr)
    R.to(t + 3.1, p + [-0.08, 0, 0], OPEN_WIDE, "back off", Rr)
    tA = t + 3.1
    L.hold(tA, "keep holding the tube")
    L.to(tA + 0.6, tube, OPEN_WIDE, "let go of the tube", Rt)
    L.to(tA + 1.6, (0.42, 0.22, 0.88), OPEN_WIDE, "move clear to the left", Rt)
    # phase B: right hand holds the tube, left arm opens the left wing
    tB = tA + 1.6
    R.hold(tB, "wait")
    R.to(tB + 2.0, tube + [-0.08, 0, 0], OPEN_WIDE, "right hand in front of the stand tube", Rt)
    R.to(tB + 3.5, tube, OPEN_WIDE, "jaws either side of the tube", Rt)
    R.to(tB + 4.5, tube, SQUEEZE, "hold the stand tube", Rt)
    p, Rl = _lip(+1, 90.0)
    L.to(tB + 2.0, p + [-0.07, 0, 0], OPEN_WIDE, "left hand in front of the folded left wing", Rl)
    L.to(tB + 3.5, p, OPEN_WIDE, "jaws either side of the left ledge lip", Rl)
    L.to(tB + 4.5, p, 0.0, "pinch the left ledge lip", Rl)
    t = tB + 4.5
    for k, phi in enumerate((72.0, 54.0, 36.0, 18.0, 0.0, -4.0)):
        t += 0.8
        p, Rl = _lip(+1, phi)
        L.to(t, p, 0.0, "swing the left wing open (%d/6)" % (k + 1), Rl)
    L.hold(t + 1.0, "press the left wing onto its stop so the diagonal brace locks straight")
    L.to(t + 1.6, p, OPEN_WIDE, "let go of the left wing", Rl)
    L.to(t + 3.1, p + [-0.08, 0, 0], OPEN_WIDE, "back off", Rl)
    tC = t + 3.1
    R.hold(tC, "keep holding the tube")
    R.to(tC + 0.6, tube, OPEN_WIDE, "let go of the tube", Rt)
    R.to(tC + 1.6, tube + [-0.10, -0.05, 0.05], OPEN_WIDE, "back off", Rt)
    R.to(tC + 4.0, parked_r, 0.0, "park")
    L.to(tC + 4.0, parked_l, 0.0, "park")
    R.hold(tC + 6.0, "hold")
    L.hold(tC + 6.0, "hold")
    return dict(left=L.w, right=R.w)


def demo():
    if os.environ.get("MS_DEMO", "both") == "both" and os.environ.get("MS_NOMOTION") != "1":
        return _demo_both()
    parked_l, parked_r = (0.21, 0.28, 0.93), (0.21, -0.28, 0.93)
    if os.environ.get("MS_NOMOTION") == "1":
        return dict(left=[Waypoint(30.0, parked_l, 0.0, "parked")], right=[Waypoint(30.0, parked_r, 0.0, "parked")])
    # right hand: pinch the right wing's ledge lip (the support angle, which sticks out toward the arms when folded),
    # 16 cm from the pivot, jaws closing across the 3 mm lip; the wrist rolls with the wing angle as it swings open
    r_lip = float(os.environ.get("MS_LIP_R", "0.16"))
    xp = SX + XD - T - 0.032

    def lip(phi):
        a = math.radians(phi)
        dy, dz = -r_lip, -0.014
        y = SY - YP + dy * math.cos(a) - dz * math.sin(a)
        z = FLOOR + ZA + dy * math.sin(a) + dz * math.cos(a)
        Rr = Orient(_Rx(a + math.pi / 2) @ _Ry(math.radians(-90)) @ _D)
        return np.array([xp, y, z]) - Rr.rot @ np.array([0.0, 0.0, TIP]), Rr

    p_f, R_f = lip(-90.0)
    R = [Waypoint(2.0, tuple(p_f + [-0.07, 0, 0]), OPEN_WIDE, "right hand in front of the folded wing", yaw=R_f),
         Waypoint(1.5, tuple(p_f), OPEN_WIDE, "jaws either side of the ledge lip", yaw=R_f),
         Waypoint(1.0, tuple(p_f), 0.0, "pinch the ledge lip", yaw=R_f)]
    # left hand: hold the stand tube, horizontal grip toward +x, jaws across the tube (y)
    Rl = Orient(_Ry(math.radians(-90)) @ _D)
    tl = Rl.rot @ np.array([0.0, 0.0, TIP])
    tube = np.array([SX, SY, FLOOR + 0.68]) - tl
    L = [Waypoint(2.0, tuple(tube + [-0.08, 0, 0]), OPEN_WIDE, "left hand in front of the stand tube", yaw=Rl),
         Waypoint(1.5, tuple(tube), OPEN_WIDE, "jaws either side of the tube", yaw=Rl),
         Waypoint(1.0, tuple(tube), SQUEEZE, "hold the stand tube", yaw=Rl)]
    for k, phi in enumerate((-72.0, -54.0, -36.0, -18.0, 0.0)):
        pk, Rk = lip(phi)
        R.append(Waypoint(0.8, tuple(pk), 0.0, "swing the wing open (%d/5)" % (k + 1), yaw=Rk))
    p_o, R_o = lip(0.0)
    R.append(Waypoint(1.0, tuple(p_o), 0.0, "wing open on its stop", yaw=R_o))
    R.append(Waypoint(0.6, tuple(p_o), OPEN_WIDE, "let go", yaw=R_o))
    R.append(Waypoint(1.5, tuple(p_o + [-0.08, 0, 0.0]), OPEN_WIDE, "back off", yaw=R_o))
    R.append(Waypoint(3.0, parked_r, 0.0, "park"))
    L.append(Waypoint(sum(w.duration for w in R) - sum(w.duration for w in L) - 3.0, tuple(tube), SQUEEZE, "keep holding", yaw=Rl))
    L.append(Waypoint(0.6, tuple(tube), OPEN_WIDE, "let go", yaw=Rl))
    L.append(Waypoint(2.4, parked_l, 0.0, "park"))
    if os.environ.get("MS_NOSTEADY") == "1":   # control: same right-hand motion, left hand stays parked
        L = [Waypoint(sum(w.duration for w in R), parked_l, 0.0, "parked")]
    return dict(left=L, right=R)


HAND_CLEAR = 0.08
DEPLOY_TOL = 3.0


def _desk_points(st):
    """Sample points on the deployed-desk outline (world), for the hands-clear distance."""
    pts = []
    for yy in (-HALF, -HALF / 2, 0.0, HALF / 2, HALF):
        for zz in (P0, P0 + H):
            pts.append(np.asarray(st.pos) + st.rotation() @ np.array(tp((XD, yy, zz))))
    for zz in np.linspace(0.25, P0 - 0.03, 8):          # the stand tube too: hands must be off the whole stand
        pts.append(np.asarray(st.pos) + st.rotation() @ np.array((0.0, 0.0, zz)))
    return np.array(pts)


def wings_from(j):
    """[right, left] deployed: ledge within 3 deg of its stop, end bar within 3 deg of vertical, and both halves of the
    diagonal brace within 3 deg of straight (locked). j: joint angles in degrees by name."""
    return [abs(j[f"{n}_ledge"]) <= DEPLOY_TOL and abs(j[f"{n}_end"] + j[f"{n}_ledge"]) <= DEPLOY_TOL
            and abs(j[f"{n}_dup"]) <= DEPLOY_TOL and abs(j[f"{n}_dlo"]) <= DEPLOY_TOL for n in ("r", "l")]


def _eval(env):
    st = labkit._part_state(env, STAND)
    j = {k: math.degrees(float(v)) for k, v in st.joints.items()}
    tilt = math.degrees(math.acos(float(np.clip(st.rotation()[2, 2], -1, 1))))
    xy = float(np.linalg.norm(np.asarray(st.pos)[:2] - _log["p0"][:2])) if _log.get("p0") is not None else 0.0
    wings = wings_from(j)
    try:
        if os.environ.get("MS_NOPADS") == "1":            # control: hands cannot be located -> no credit
            raise RuntimeError("pads unavailable (control)")
        from yam_arena import runtime as _rt
        o = env.unwrapped.scene.env_origins[0].cpu().numpy()
        tips = [np.asarray(_rt.pad_tips_world(env.unwrapped.scene[n])) - o for n in ("left_robot", "right_robot")]
        dp = _desk_points(st)
        clear = min(float(np.min(np.linalg.norm(dp - t, axis=1))) for t in tips)
    except Exception:
        clear = 0.0                                        # fail closed: unknown hand position never earns credit
    um, ur = None, None
    if UKE:
        us = labkit._part_state(env, UKULELE)
        up, uR = np.asarray(us.pos, float), us.rotation()
        if _log.get("u0") is None and _log.get("step", 0) >= 40:          # reference after the instrument has settled
            _log["u0"] = (up.copy(), uR.copy())
        if _log.get("u0") is not None:
            um = float(np.linalg.norm(up - _log["u0"][0]))
            ur = math.degrees(math.acos(float(np.clip((np.trace(_log["u0"][1].T @ uR) - 1) / 2, -1, 1))))
    return dict(j=j, tilt=tilt, xy=xy, wings=wings, clear=clear, pos=np.asarray(st.pos, float).copy(),
                uke_moved=um, uke_rot=ur)


def _score_now(env):
    e = _eval(env)
    if e["tilt"] > 10.0 or e["xy"] > 0.05:
        _log["latched"] = "stand tipped or slid"
    if UKE and e.get("uke_moved") is not None and (e["uke_moved"] > 0.005 or e["uke_rot"] > 3.0):
        _log["latched"] = _log.get("latched") or "ukulele disturbed"
    hist = _log.setdefault("hist", [])
    hist.append((e["pos"], e["j"]["r_ledge"], e["j"]["l_ledge"], e["clear"]))
    if len(hist) > 21:
        hist.pop(0)
    still = len(hist) == 21 and max(float(np.linalg.norm(h[0] - hist[0][0])) for h in hist) < 0.001 and         max(abs(h[1] - hist[0][1]) + abs(h[2] - hist[0][2]) for h in hist) < 0.5
    sc = 0.0
    clear_1s = min(h[3] for h in hist)                    # hands clear on every sample of the last second
    if not _log.get("latched") and still and clear_1s >= HAND_CLEAR and e["tilt"] <= 3.0 and e["xy"] <= 0.03:
        sc = 1.0 if all(e["wings"]) else (0.5 if any(e["wings"]) else 0.0)
    _log["score"] = sc
    _log["eval"] = dict(wings=e["wings"], tilt=round(e["tilt"], 2), xy_mm=round(e["xy"] * 1e3, 1), clear_m=round(e["clear"], 3),
                        still=still, latched=_log.get("latched"), score=sc,
                        uke_mm=None if e["uke_moved"] is None else round(e["uke_moved"] * 1e3, 1),
                        uke_deg=None if e["uke_rot"] is None else round(e["uke_rot"], 2))
    return sc


@on_step
def _scorer_step(env):
    _score_now(env)
    if _log["step"] % 20 == 0:
        print("MSSCORE " + json.dumps(dict(step=_log["step"], **_log["eval"])), flush=True)


def score(objects) -> float:
    """1.0: both desk wings deployed (each ledge within 3 deg of its stop, end bar within 3 deg of vertical), stand
    upright within 3 deg and within 3 cm of its start, both hands at least 8 cm from the desk, all still for 1 s.
    0.5: one wing deployed (same other conditions). 0 otherwise. Latched 0 if the stand tips over 10 deg or slides
    over 5 cm at any time."""
    return float(_log.get("score", 0.0))


def success(objects) -> bool:
    return False
