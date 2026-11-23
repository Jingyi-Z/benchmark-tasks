"""Steer a laser spot onto a target with a rotating wedge-prism pair.

On a matte black optical breadboard (150 x 600 mm, M6 holes on a 25 mm grid) lying on the bench, along an optical axis
that runs sideways (+y) at x 0.36 m and 90 mm above the board:
  - a Class 2 red laser pointer in a V-holder at y -0.30, pointing +y;
  - prism A (1 in N-BK7 wedge, 2 deg deviation, Thorlabs PS810 class) in a manual rotation mount at y -0.04;
  - prism B, same, in its own mount at y 0.00 (40 mm apart: a fingertip turns one ring without touching the other);
  - a white card screen at y 0.35 (L = 350 mm beyond B) with a printed 4 mm target ring 12.25 mm above the undeflected
    spot.
Start (fixed reset): A's base (thick edge) at +30 deg, B's at -30 deg (0 = up, positive turns the top toward +x).
Task: turn both rings so the spot lands inside the target ring, then let go.

The spot and the visible beam are computed every step from the two REAL ring angles by thin-prism deviation in sequence
(prism_scorer.spot_yz) and drawn: the spot by a render-only marker on two slides on the screen, the beam after B by a
render-only rod on two hinges at B. Colliders: board, holders, posts, mount rings and screen card are simple boxes and
cylinders; the glass, beam, spot, target ring, hole grid and engraved marks are drawn only."""
import math, os, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa
import labkit
from labkit import Joint, Mesh, on_reset, on_step, part
import yam_arena.demo as _demo
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
import prism_scorer as PS

TOP = TABLE_TOP["station"]
BLACK, DARK, STEEL, WHITE, RED = (0.05, 0.05, 0.06), (0.15, 0.15, 0.17), (0.62, 0.63, 0.66), (0.95, 0.95, 0.93), (0.95, 0.08, 0.06)
GLASS, HOLE, YELLOW = (0.78, 0.88, 0.93), (0.02, 0.02, 0.02), (0.95, 0.80, 0.10)
X0 = 0.36                            # optical axis x (a hole row of the board)
BT = 0.0127                          # breadboard thickness (1/2 in)
BTOP = TOP + BT
ZB = BTOP + 0.09                     # beam height
YL, YA, YB, YS = -0.225, PS.SA, PS.SB, PS.SS     # all on 25 mm grid holes
RO = float(os.environ.get("PR_RO", "0.022"))   # knurled dial ring outer radius: 44 mm OD (LBTEK RME-1A CAD, +-22 mm)
TH, TAB = 0.006, 0.010                          # ring thickness, (legacy) tab height
START = {"A": math.radians(float(os.environ.get("PR_SA", "30"))), "B": math.radians(float(os.environ.get("PR_SB", "-30")))}


def quads_mesh(quads, color):
    pts, tris = [], []
    for q in quads:
        k = len(pts)
        pts += list(q)
        tris += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return Mesh(points=np.asarray(pts, float), tris=np.asarray(tris, np.int64), color=color, collide="none")


def box_mesh(c, size, color):
    c, h = np.asarray(c, float), np.asarray(size, float) / 2
    pts = np.array([c + h * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                     (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return Mesh(points=pts, tris=tris, color=color, collide="none")


# breadboard with its hole grid (drawn)
_holes = []
for i in range(6):
    for j in range(24):
        x = 0.31 + 0.025 * i
        y = -0.225 + 0.025 * j
        h = 0.0018
        _holes.append([(x - h, y - h, BTOP + 0.0002), (x + h, y - h, BTOP + 0.0002), (x + h, y + h, BTOP + 0.0002), (x - h, y + h, BTOP + 0.0002)])
BOARD = part("breadboard", pos=(0, 0, 0), fixed=True, friction=0.6,
             boxes=[((0.3725, 0.0625, TOP + BT / 2), (0.150, 0.600, BT), DARK)], meshes=[quads_mesh(_holes, HOLE)])


def post(x, y, ztop):
    """post holder (1 in tall) + post up to ztop"""
    return [((x, y, BTOP + 0.0127), (0.016, 0.016, 0.0254), BLACK),
            ((x, y, (BTOP + 0.0254 + ztop) / 2), (0.0127, 0.0127, ztop - BTOP - 0.0254), STEEL)]


# laser pointer (Class 2, about 15 mm dia x 140 mm, EST) in a V-holder on a post
LASER = part("laser", pos=(0, 0, 0), fixed=True, friction=0.5,
             boxes=post(X0, YL, ZB - 0.010) + [((X0, YL, ZB - 0.008), (0.03, 0.04, 0.006), BLACK)],
             cylinders=[((X0, YL + 0.02, ZB), 0.0075, 0.140, (0.20, 0.22, 0.26), "Y"),
                        ((X0, YL + 0.091, ZB), 0.0045, 0.003, BLACK, "Y")],
             meshes=[box_mesh((X0 + 0.0076, YL, ZB), (0.0004, 0.03, 0.008), YELLOW),        # warning label
                     box_mesh((X0, (YL + 0.092 + YA) / 2, ZB), (0.0008, YA - YL - 0.092, 0.0008), RED)])   # beam to A


def wedge_glass(y):
    """a 1 in wedge disc, drawn only: thick edge 4.8 mm at the base (+z in the ring frame), thin edge 3.0 mm"""
    pts, tris = [], []
    n = 24
    for k in range(n):
        t = 2 * math.pi * k / n
        xx, zz = 0.0127 * math.sin(t), 0.0127 * math.cos(t)
        half = (0.0039 + 0.0009 * math.cos(t)) / 2
        pts += [(X0 + xx, y - half, ZB + zz), (X0 + xx, y + half, ZB + zz)]
    for k in range(n):
        a0, a1, b0, b1 = 2 * k, 2 * ((k + 1) % n), 2 * k + 1, 2 * ((k + 1) % n) + 1
        tris += [(a0, a1, b1), (a0, b1, b0)]
    c0 = len(pts); pts += [(X0, y - 0.0019, ZB), (X0, y + 0.0019, ZB)]
    for k in range(n):
        tris += [(c0, 2 * ((k + 1) % n), 2 * k), (c0 + 1, 2 * k + 1, 2 * ((k + 1) % n) + 1)]
    return Mesh(points=np.asarray(pts, float), tris=np.asarray(tris, np.int64), color=GLASS, collide="none")


def _hull_tris(pts):
    from scipy.spatial import ConvexHull
    return ConvexHull(pts).simplices


# thumb tab on each ring (part of the ring; 6 mm wide, 10 mm tall). Placed at TAB_OFF from the ring's zero mark so it
# stays on the robots' side of the ring top from the start angle to the target angle
GRIP = os.environ.get("PR_GRIP", "face")       # "face": pinch the knurled ring across its faces; "tab": legacy
TAB_OFF = {"A": math.radians(float(os.environ.get("PR_OFFA", "-95"))), "B": math.radians(float(os.environ.get("PR_OFFB", "12")))}
# (the right arm reaches the tab only on the robots' side of the ring: wrist tilt -80 to -10 deg, offline IK)
TABW = 0.006


def _tab(y, off):
    pts = []
    for r in (RO - 0.001, RO + TAB):
        h = TABW / 2 / r
        for t in (off - h, off + h):
            for dy in (-TH / 2, TH / 2):
                pts.append((X0 + r * math.sin(t), y + dy, ZB + r * math.cos(t)))
    pts = np.array(pts)
    return Mesh(points=pts, tris=np.asarray(_hull_tris(pts), np.int64), color=BLACK, collide="hull")


def ring(name, y):
    ticks = []
    for k in range(36):                      # engraved marks every 10 deg on the ring face (-y side)
        t = 2 * math.pi * k / 36
        r0, r1 = RO - (0.004 if k % 9 == 0 else 0.0025), RO - 0.0005
        w = 0.0004
        c, s = math.sin(t), math.cos(t)
        ticks.append([(X0 + r0 * c - w * s, y - TH / 2 - 0.0002, ZB + r0 * s + w * c),
                      (X0 + r1 * c - w * s, y - TH / 2 - 0.0002, ZB + r1 * s + w * c),
                      (X0 + r1 * c + w * s, y - TH / 2 - 0.0002, ZB + r1 * s - w * c),
                      (X0 + r0 * c + w * s, y - TH / 2 - 0.0002, ZB + r0 * s - w * c)])
    segs = []
    for k in range(24):                      # knurled rotating dial ring: 24 convex segments, r 14.0 mm to RO
        t0, t1 = 2 * math.pi * k / 24, 2 * math.pi * (k + 1) / 24
        pts = np.array([(X0 + r * math.sin(t), y + dy, ZB + r * math.cos(t)) for r in (0.0140, RO) for t in (t0, t1)
                        for dy in (-TH / 2, TH / 2)])
        segs.append(Mesh(points=pts, tris=np.asarray(_hull_tris(pts), np.int64),
                         color=(0.10, 0.10, 0.11) if k % 2 else (0.20, 0.20, 0.22), collide="hull"))
    return Joint(name, kind="hinge", axis="Y", anchor=(X0, y, ZB), lower=-4 * math.pi, upper=4 * math.pi, mass=0.03,
                 initial=START[name], target=START[name], friction=float(os.environ.get("PR_FRIC", "0.02")),
                 damping=0.01,
                 meshes=segs + ([_tab(y, TAB_OFF[name])] if GRIP == "tab" else []) + [wedge_glass(y), quads_mesh(ticks, WHITE),
                                box_mesh((X0, y - TH / 2 - 0.0003, ZB + RO - 0.002), (0.0012, 0.0002, 0.004), RED)])


# a pinch across the ring faces turns the ring by torsional friction about the contact normal (the ring axis): give
# the knurled ring a real finger-pad contact patch so the pinch can carry torque (point contacts carry none)
labkit.TORSIONAL_PATCH["prism_mounts"] = float(os.environ.get("PR_TP", "0.004"))
MOUNTS = part("prism_mounts", pos=(0, 0, 0), fixed=True, friction=float(os.environ.get("PR_MU", "0.9")), joints=[ring("A", YA), ring("B", YB)],
              boxes=sum((post(X0, y, ZB - RO - 0.004) + [((X0, y + TH / 2 + 0.003, ZB - RO - 0.001), (0.030, 0.006, 0.006), BLACK)]
                         for y in (YA, YB)), []))

# screen: white card on a post, printed target ring; spot on two slides; beam after B on two hinges (all drawn)
RT = 0.002
_ring_q = []
for k in range(32):
    t0, t1 = 2 * math.pi * k / 32, 2 * math.pi * (k + 1) / 32
    def pnt(r, t):
        return (X0 + r * math.sin(t), YS - 0.0012, ZB + PS.TARGET[1] + r * math.cos(t))
    _ring_q.append([pnt(RT - 0.0003, t0), pnt(RT + 0.0003, t0), pnt(RT + 0.0003, t1), pnt(RT - 0.0003, t1)])
spx = Joint("spx", kind="slide", axis="X", anchor=(X0, YS - 0.0015, ZB), lower=-0.1, upper=0.1, mass=0.001,
            stiffness=50.0, damping=0.5, meshes=[box_mesh((X0, YS - 0.0017, ZB), (0.0001, 0.0001, 0.0001), RED)])
spz = Joint("spz", kind="slide", axis="Z", anchor=(X0, YS - 0.0015, ZB), parent="spx", lower=-0.1, upper=0.1,
            mass=0.001, stiffness=50.0, damping=0.5,
            meshes=[box_mesh((X0, YS - 0.0016, ZB), (0.0025, 0.0004, 0.0025), RED),           # the visible spot (x and z)
                    box_mesh((X0, YS + 0.0016, ZB), (0.0025, 0.0004, 0.0025), RED)])          # seen through the frosted screen
for k in range(32):                      # second printed ring at the 3.0 mm half-credit radius
    t0, t1 = 2 * math.pi * k / 32, 2 * math.pi * (k + 1) / 32
    _ring_q.append([pnt(0.003 - 0.0002, t0), pnt(0.003 + 0.0002, t0), pnt(0.003 + 0.0002, t1), pnt(0.003 - 0.0002, t1)])
_tc = (X0, ZB + PS.TARGET[1])
for (ux, uz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # crosshair ticks from 4 to 10 mm off the target centre
    w = 0.0003
    p0, p1 = 0.004, 0.010
    _ring_q.append([(_tc[0] + ux * p0 - uz * w, YS - 0.0012, _tc[1] + uz * p0 - ux * w),
                    (_tc[0] + ux * p1 - uz * w, YS - 0.0012, _tc[1] + uz * p1 - ux * w),
                    (_tc[0] + ux * p1 + uz * w, YS - 0.0012, _tc[1] + uz * p1 + ux * w),
                    (_tc[0] + ux * p0 + uz * w, YS - 0.0012, _tc[1] + uz * p0 + ux * w)])
_ring_back = [[(px, YS + 0.0012, pz) for (px, py, pz) in q] for q in _ring_q]
SCREEN = part("screen", pos=(0, 0, 0), fixed=True, joints=[spx, spz], meshes=[quads_mesh(_ring_q, BLACK), quads_mesh(_ring_back, BLACK)],
              boxes=post(X0 + 0.04, YS, ZB - 0.035) + [((X0, YS, ZB + 0.005), (0.10, 0.002, 0.075), (0.93, 0.94, 0.95))])   # frosted acrylic
bz = Joint("byaw", kind="hinge", axis="Z", anchor=(X0, YB, ZB), lower=-0.5, upper=0.5, mass=0.001, stiffness=5.0,
           damping=0.05, meshes=[box_mesh((X0, YB + 0.002, ZB), (0.0001, 0.0001, 0.0001), RED)])
bp = Joint("bpitch", kind="hinge", axis="X", anchor=(X0, YB, ZB), parent="byaw", lower=-0.5, upper=0.5, mass=0.001,
           stiffness=5.0, damping=0.05,
           meshes=[box_mesh((X0, YB + (YS - YB) / 2, ZB), (0.0008, YS - YB - 0.003, 0.0008), RED)])
bza = Joint("bayaw", kind="hinge", axis="Z", anchor=(X0, YA, ZB), lower=-0.5, upper=0.5, mass=0.001, stiffness=5.0,
            damping=0.05, meshes=[box_mesh((X0, YA + 0.002, ZB), (0.0001, 0.0001, 0.0001), RED)])
bpa = Joint("bapitch", kind="hinge", axis="X", anchor=(X0, YA, ZB), parent="bayaw", lower=-0.5, upper=0.5, mass=0.001,
            stiffness=5.0, damping=0.05,
            meshes=[box_mesh((X0, (YA + YB) / 2, ZB), (0.0008, YB - YA - 0.003, 0.0008), RED)])
BEAM_AB = part("beam_ab", pos=(0, 0, 0), fixed=True, joints=[bza, bpa],
               boxes=[((X0 + 0.07, YA, BTOP + 0.002), (0.004, 0.004, 0.004), DARK)])
BEAM = part("beam", pos=(0, 0, 0), fixed=True, joints=[bz, bp],
            boxes=[((X0 + 0.07, YB, BTOP + 0.002), (0.004, 0.004, 0.004), DARK)])     # tiny anchor block (part needs a body)
OBJECTS = [BOARD, LASER, MOUNTS, SCREEN, BEAM, BEAM_AB]

# ------------------------------------------------------------------------------------------------------------------
_S = {}


class Orient(float):
    def __new__(cls, rot):
        o = float.__new__(cls, 0.0)
        o.rot = np.asarray(rot, float)
        return o


_prev = _demo.grip_rotation
if not getattr(_prev, "_rot", False):
    def _g(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)
    _g._rot = True
    _demo.grip_rotation = _g


def app_rot(yaw, pitch, roll):
    a = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    up = np.array([0, 0, 1.0])
    c0 = np.cross(a, up)
    c0 /= np.linalg.norm(c0)
    v = np.cross(a, c0)
    c = math.cos(roll) * c0 + math.sin(roll) * v
    return np.column_stack([c, np.cross(a, c), a])


DOWN = app_rot(0.0, math.radians(-89.0), float(os.environ.get("PR_ROLL", "0.0")))   # straight down, jaws along y


PRE = float(os.environ.get("PR_PRE", "-0.008"))   # jaws pre-closed to about the ring width + margin


def _ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def strokes(y, total_deg, max_deg, sign_x, pad):
    """turn the ring at y by total_deg: pinch the knurled rim at the top with pre-closed jaws (straddling only this
    ring), rotate the wrist about the optical axis so the pinch rides round with the ring, release, lift out
    radially, regrip at the top; strokes of at most max_deg"""
    W = []
    # the commanded point sits about 16 mm above the pad centre (measured g30): put the pad centre 2.5 mm under the
    # rim (a shallow pinch on the knurl, clear of the aperture) and rotate about the ring axis through the pads
    r = RO - float(os.environ.get("PR_DEPTH", "0.006")) + float(os.environ.get("PR_H", "0.016"))
    n = max(1, int(math.ceil(abs(total_deg) / max_deg - 1e-9)))
    d = math.radians(abs(total_deg) / n)
    m = 4
    for k in range(n):
        top = np.array([X0 + pad[0], y + pad[1], ZB + r])
        W += [Waypoint(1.5, tuple(top + [0, 0, 0.05]), PRE, f"over the ring top, jaws pre-closed ({k})", Orient(DOWN)),
              Waypoint(1.0, tuple(top), PRE, "down onto the knurled rim, straddling only this ring", Orient(DOWN)),
              Waypoint(0.8, tuple(top), FINGER_CLOSED, "pinch the knurled rim", Orient(DOWN))]
        for j in range(1, m + 1):
            t = sign_x * d * j / m
            p = np.array([X0 + pad[0] + r * math.sin(t), y + pad[1], ZB + r * math.cos(t)])
            W.append(Waypoint(0.5, tuple(p), FINGER_CLOSED, "rotate the wrist about the optical axis", Orient(_ry(t) @ DOWN)))
        t = sign_x * d
        out = np.array([math.sin(t), 0.0, math.cos(t)])
        W += [Waypoint(0.6, tuple(p), PRE, "release (open only to the pre-close width)", Orient(_ry(t) @ DOWN)),
              Waypoint(0.8, tuple(p + 0.04 * out), PRE, "lift out radially, clear of the rim", Orient(_ry(t) @ DOWN))]
    # straight up, high above this ring, before moving on (a direct move swept through ring A in j1)
    W.append(Waypoint(1.0, (X0 + pad[0], y + pad[1], ZB + RO + float(os.environ.get("PR_H", "0.016")) + 0.08), PRE,
                      "rise clear above the mounts", Orient(DOWN)))
    return W


# ---------------- closed loop on what the cameras see: the two dial angles and the dot against the target ----------
# The demo reads each ring's dial angle (engraved ticks, visible) and the dot on the screen. Before every stroke it
# picks the ring and the turn that bring the dot to the target, strokes, re-reads, and repeats until the dot is well
# inside the small ring; then it lets go and clears. No fixed stroke count.
def _solve_targets():
    best = []
    for i in range(0, 721):
        for j in range(0, 721):
            a, b = math.radians(i * 0.5 - 180), math.radians(j * 0.5 - 180)
            m = PS.miss_m(a, b)
            if m < 0.0006:
                best.append((m, a, b))
    sols = []
    for m, a, b in sorted(best):
        if all(abs(math.remainder(a - s[0], 2 * math.pi)) > 0.3 for s in sols):
            sols.append((a, b))
    return sols


_SOLS = _solve_targets()
_TGT = min(_SOLS, key=lambda s: abs(math.remainder(s[0] - START["A"], 2 * math.pi)) +
           abs(math.remainder(s[1] - START["B"], 2 * math.pi)))
NSLOT = int(os.environ.get("PR_SLOTS", "12"))
DUR = [1.2, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.5, 0.6, 0.8]      # hover, down, pinch, rotate x4, release, lift, rise
TSLOT = sum(DUR)
T0 = float(os.environ.get("PR_T0", "7.5"))            # first look after the left wrist camera has settled
DONE_M = float(os.environ.get("PR_DONE", "0.0010"))
TAB_R = RO + float(os.environ.get("PR_TIPR", "0.002"))      # finger tips just above the rim, on the tab
H = float(os.environ.get("PR_H", "0.016"))                   # commanded point to finger tips
DOWNX = app_rot(math.pi / 2, math.radians(-89.0), 0.0)       # straight down, jaws along x (tangent at the top)
CLEAR = (X0 - 0.10, YA - 0.07, ZB + 0.15)                   # robots' side, about 190 mm from both mounts


def _qm(q):
    """rotation matrix from a camera quaternion as the data stores it (PR_QORD xyzw, Isaac default here, or wxyz)"""
    q = [float(c) for c in q]
    if os.environ.get("PR_QORD", "xyzw") == "xyzw":
        return _quat_mat(q[3], q[0], q[1], q[2])
    return _quat_mat(*q)


def _quat_mat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _see_dot(env):
    """(u, v) of the laser dot on the screen, in m from the beam axis, found ONLY in the left wrist camera image:
    red pixels, a morphological opening removes the thin beam rod, the largest remaining blob is the dot, its centre
    ray (camera pose from the arm's kinematics, lens from its intrinsics) meets the screen plane. None if not seen."""
    from scipy import ndimage
    cam = env.unwrapped.scene["left_cam"]
    img = labkit._np(cam.data.output["rgb"])[0][..., :3].astype(float)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    red = (r > 120) & (r > 2.2 * g) & (r > 2.2 * b)
    K = labkit._np(cam.data.intrinsic_matrices)[0]
    o = labkit._np(cam.data.pos_w)[0] - env.unwrapped.scene.env_origins[0].cpu().numpy()
    Rc = _qm(labkit._np(cam.data.quat_w_ros)[0])
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        return None
    rays = Rc @ np.linalg.solve(K, np.stack([xs, ys, np.ones_like(xs)]).astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = (YS - 0.0016 - o[1]) / rays[1]
    hu, hw = o[0] + sc * rays[0] - X0, o[2] + sc * rays[2] - ZB
    on = (sc > 0) & (np.abs(hu) < 0.05) & (np.abs(hw - 0.005) < 0.0375)       # only red seen on the card
    card = np.zeros_like(red)
    card[ys[on], xs[on]] = True
    k = max(1, int(round(img.shape[1] / 224.0)))
    win = 2 * k + 1                                  # the dot is about 2.5 mm; the beam rod is far thinner
    dens = ndimage.uniform_filter(card.astype(float), win) * win * win
    if dens.max() < 0.6 * win * win:
        return None                                  # no dot on the card (lost, covered or off the card)
    py0, px0 = np.unravel_index(int(np.argmax(dens)), dens.shape)
    y0, y1, x0, x1 = max(0, py0 - 2 * win), py0 + 2 * win + 1, max(0, px0 - 2 * win), px0 + 2 * win + 1
    sub = dens[y0:y1, x0:x1] >= dens.max() - 0.5 * k * k                     # one dot only: around the peak
    yy, xx = np.nonzero(sub)
    px, py = float(xx.mean() + x0), float(yy.mean() + y0)
    _S["px"] = (round(px, 1), round(py, 1), img.shape[1])
    d = np.linalg.solve(K, np.array([px, py, 1.0]))
    conv = os.environ.get("PR_CAMQ", "ros")
    if conv == "ros":
        ray = _qm(labkit._np(cam.data.quat_w_ros)[0]) @ d
    elif conv == "opengl":                      # opengl camera: -z forward, +y up, +x right
        ray = _qm(labkit._np(cam.data.quat_w_opengl)[0]) @ np.array([d[0], -d[1], -d[2]])
    else:                                       # world convention: +x forward, +y left, +z up
        ray = _qm(labkit._np(cam.data.quat_w_world)[0]) @ np.array([d[2], -d[0], -d[1]])
    if "dbg" not in _S:
        _S["dbg"] = 1
        for nm, qq in (("ros", cam.data.quat_w_ros), ("world", cam.data.quat_w_world), ("opengl", cam.data.quat_w_opengl)):
            Rq = _qm(labkit._np(qq)[0])
            print(f"[pr] cam {nm} o={np.round(o, 3)} fwd_cols={np.round(Rq, 2).tolist()} K={np.round(K, 1).tolist()}",
                  file=sys.stderr, flush=True)
    if abs(ray[1]) < 1e-6:
        return None
    s = (YS - 0.0016 - o[1]) / ray[1]
    if s <= 0:
        return None
    p = o + s * ray
    u, w = p[0] - X0, p[2] - ZB
    _S["px"] = _S["px"] + (round(u * 1e3, 2), round(w * 1e3, 2))
    if abs(u) > 0.05 or abs(w - 0.005) > 0.0375:          # off the card
        return None
    return u, w


def _angles_from_dot(uv, guess):
    """the ring angles that put the dot where it is seen, nearest the dead-reckoned guess (start marks + strokes)"""
    best = None
    for i in range(-60, 61):
        for j in range(-60, 61):
            a, b = guess[0] + math.radians(i * 0.5), guess[1] + math.radians(j * 0.5)
            su, sv = PS.spot_yz(a, b)
            c = (su - uv[0]) ** 2 + (sv - uv[1]) ** 2 + 1e-9 * ((i * 0.5) ** 2 + (j * 0.5) ** 2)
            if best is None or c < best[0]:
                best = (c, a, b)
    return {"A": best[1], "B": best[2]}


def _plan(q):
    """decide the next stroke from the observed dial angles"""
    if math.hypot(q["u"] - PS.TARGET[0], q["v"] - PS.TARGET[1]) < DONE_M:
        return None
    e = {k: math.remainder(_TGT[i] - q[k], 2 * math.pi) for i, k in enumerate(("A", "B"))}
    k = max(e, key=lambda kk: abs(e[kk]))
    g = _S.setdefault("gain", {"A": 0.9, "B": 0.9})[k]
    cmd = min(math.radians(float(os.environ.get("PR_STEPMAX", "15"))), max(math.radians(3.0), abs(e[k]) / g))
    sg = 1.0 if e[k] > 0 else -1.0
    if GRIP == "face":   # any point of the rim will do: start where the whole stroke stays at a reachable wrist tilt
        psi0 = math.radians(-55.0 if sg > 0 else -20.0)
    else:
        psi0 = q[k] + TAB_OFF[k]
    return dict(ring=k, y=YA if k == "A" else YB, s=sg, psi0=psi0, d=cmd, q0=q[k])


def _slot_update(t, env, qtrue):
    k = int((t - T0) // TSLOT)
    if k < 0 or k >= NSLOT:
        return
    P = _S.setdefault("plan", {})
    if k in P:
        return
    est = _S.setdefault("est", {"A": START["A"], "B": START["B"]})     # dead reckoning from the start marks
    prev = P.get(k - 1)
    uv = _see_dot(env)
    if uv is None:
        P[k] = False
        print(f"[pr] slot {k} at t={t:.1f}: dot NOT seen -> no stroke {_S.pop('px', None)}", file=sys.stderr, flush=True)
        return
    if prev:
        est[prev["ring"]] = prev["q0"] + prev["s"] * prev["d"] * _S.setdefault("gain", {"A": 0.9, "B": 0.9})[prev["ring"]]
    seen = _angles_from_dot(uv, (est["A"], est["B"]))
    if prev:                                                    # learn the stroke gain from the seen result
        got = abs(math.remainder(seen[prev["ring"]] - prev["q0"], 2 * math.pi))
        _S["gain"][prev["ring"]] = min(1.5, max(0.3, got / prev["d"]))
    est.update(seen)
    obs = dict(seen, u=uv[0], v=uv[1])
    pl = _plan(obs)
    P[k] = pl if pl else False
    msg = "done, hands clear" if not pl else f"ring {pl['ring']} {math.degrees(pl['s'] * pl['d']):+.1f} deg"
    print(f"[pr] slot {k} at t={t:.1f}: seen dot ({uv[0] * 1e3:.2f},{uv[1] * 1e3:.2f}) mm, seen miss "
          f"{math.hypot(uv[0] - PS.TARGET[0], uv[1] - PS.TARGET[1]) * 1e3:.2f} (true {PS.miss_m(qtrue['A'], qtrue['B']) * 1e3:.2f}); "
          f"est A {math.degrees(seen['A']):.1f} B {math.degrees(seen['B']):.1f} (true {math.degrees(qtrue['A']):.1f} "
          f"{math.degrees(qtrue['B']):.1f}) -> {msg}", file=sys.stderr, flush=True)


def _pose(k, st):
    pl = _S.get("plan", {}).get(k)
    if not pl:
        return np.array(CLEAR, float), DOWN
    rt = (RO - float(os.environ.get("PR_DEPTH", "0.006")) if GRIP == "face" else TAB_R) + H
    G = DOWN if GRIP == "face" else DOWNX          # face pinch: jaws along the ring axis, across the ring faces
    if st <= 2:
        psi = pl["psi0"]
    elif st <= 6:
        psi = pl["psi0"] + pl["s"] * pl["d"] * (st - 2) / 4
    else:
        psi = pl["psi0"] + pl["s"] * pl["d"]
    rr = rt + (0.04 if st in (0, 8) else 0.0)
    p = np.array([X0 + rr * math.sin(psi), pl["y"], ZB + rr * math.cos(psi)])
    if st == 9:
        p = np.array([p[0], pl["y"], ZB + RO + H + 0.08])
        return p, G
    return p, _ry(psi) @ G


class _DP:
    def __init__(self, k, st):
        self.k, self.st = k, st

    def __array__(self, dtype=None, copy=None):
        return np.asarray(_pose(self.k, self.st)[0], dtype=dtype or float)


class _DO(float):
    def __new__(cls, k, st):
        o = float.__new__(cls, 0.0)
        o.k, o.st = k, st
        return o

    @property
    def rot(self):
        return np.asarray(_pose(self.k, self.st)[1], float)


def demo():
    # Right hand: closed-loop strokes. Each stroke pinches the ring's thumb tab with pre-closed jaws (jaws across the
    # tab), rotates the wrist about the optical axis, releases, lifts out and rises clear; then the dial and dot are
    # re-read and the next stroke chosen. Left hand parks its wrist camera facing the screen and watches the dot.
    GR = [PRE, PRE, FINGER_CLOSED, FINGER_CLOSED, FINGER_CLOSED, FINGER_CLOSED, FINGER_CLOSED, PRE, PRE, PRE]
    LB = ["over the tab", "down onto the tab", "pinch the thumb tab", "rotate the wrist", "rotate the wrist",
          "rotate the wrist", "rotate the wrist", "release to the pre-close width", "lift out", "rise clear"]
    R = [Waypoint(T0, CLEAR, PRE, "start clear", Orient(DOWN))]
    for k in range(NSLOT):
        for st in range(10):
            R.append(Waypoint(DUR[st], _DP(k, st), GR[st], f"slot {k}: {LB[st]}", _DO(k, st)))
    R.append(Waypoint(3.0, CLEAR, PRE, "hands clear", Orient(DOWN)))
    if os.environ.get("PR_LKMODE", "front") == "front":
        # beside the beam on the robots' side, looking diagonally at the card's front face
        look = np.array([float(os.environ.get("PR_LKXA", "0.34")), float(os.environ.get("PR_LKYA", "0.29")),
                         ZB + PS.TARGET[1] + float(os.environ.get("PR_LKZ", "0.02"))])
        tgt = np.array([X0, YS, ZB + PS.TARGET[1]])
        dv = tgt - look
        yaw = math.atan2(dv[1], dv[0])
        pitch = math.atan2(dv[2], math.hypot(dv[0], dv[1])) + math.radians(float(os.environ.get("PR_LKP", "10")))
        LR = app_rot(yaw, pitch, float(os.environ.get("PR_LKR", "0.0")))
    else:
        look = np.array([X0 + float(os.environ.get("PR_LKX", "0.0")), YS + float(os.environ.get("PR_LKD", "0.10")),
                         ZB + PS.TARGET[1] + float(os.environ.get("PR_LKZ", "0.03"))])
        LR = app_rot(math.radians(-90.0), math.radians(float(os.environ.get("PR_LKP", "-15"))), float(os.environ.get("PR_LKR", "0.0")))
    L = [Waypoint(4.0, tuple(look + [0, 0.04, 0.05]), FINGER_CLOSED, "behind the screen", Orient(LR)),
         Waypoint(2.0, tuple(look), FINGER_CLOSED, "look at the dot through the frosted screen", Orient(LR)),
         Waypoint(95.0, tuple(look), FINGER_CLOSED, "watch while the right hand turns the rings", Orient(LR))]
    return {"left": L, "right": R}


@on_reset
def _r(env):
    _S.clear()
    _S["T"] = PS.Tracker()


def _jt(art):
    return [x.replace("_joint", "") for x in art.joint_names]


@on_step
def _w(env):
    m = env.unwrapped.scene["prism_mounts"]
    q = dict(zip(_jt(m), (float(v) for v in labkit._np(m.data.joint_pos)[0])))
    u, v = PS.spot_yz(q["A"], q["B"])
    sc = env.unwrapped.scene["screen"]
    sn = _jt(sc)
    tgt = sc.data.joint_pos.clone()
    tgt[:, sn.index("spx")] = float(u)
    tgt[:, sn.index("spz")] = float(v)
    sc.set_joint_position_target(tgt)
    bm = env.unwrapped.scene["beam"]
    bn = _jt(bm)
    tb = bm.data.joint_pos.clone()
    # direction after B: slopes du/dy, dv/dy = dA + dB; offset at B (tiny) ignored for the drawn rod
    du = PS.DEV * (math.sin(q["A"]) + math.sin(q["B"]))
    dv = PS.DEV * (math.cos(q["A"]) + math.cos(q["B"]))
    tb[:, bn.index("byaw")] = -math.atan(du)      # +x deflection = rotation about z toward -angle (rod along +y)
    tb[:, bn.index("bpitch")] = math.atan(dv)     # +z deflection = rotation about x
    bm.set_joint_position_target(tb)
    ba = env.unwrapped.scene["beam_ab"]
    an = _jt(ba)
    ta = ba.data.joint_pos.clone()
    ta[:, an.index("bayaw")] = -math.atan(PS.DEV * math.sin(q["A"]))
    ta[:, an.index("bapitch")] = math.atan(PS.DEV * math.cos(q["A"]))
    ba.set_joint_position_target(ta)
    try:
        from yam_arena import runtime as _rt
        o = env.unwrapped.scene.env_origins[0].cpu().numpy()
        pads = [(_rt.pad_tips_world(env.unwrapped.scene[n]) - o).reshape(-1, 3) for n in ("left_robot", "right_robot")]
        mounts = [np.array([X0, y, ZB]) for y in (YA, YB)]
        hc = min(float(np.linalg.norm(p_ - c)) for P in pads for p_ in P for c in mounts) - RO
        bc = PS.beam_clear([(p_[0] - X0, p_[1], p_[2] - ZB) for P in pads for p_ in P], q["A"], q["B"], YL + 0.095)
        _S["pads"] = [P.mean(0) for P in pads]
        gt = any(math.hypot(p_[0] - X0, p_[2] - ZB) < 0.0135 and abs(p_[1] - y) < TH / 2 + 0.004
                 for P in pads for p_ in P for y in (YA, YB))
    except Exception:  # noqa: BLE001
        hc = bc = None
        gt = False
    fixed = {k: (0.0, 0.0, 0.0, 0.0) for k in ("breadboard", "laser", "screen", "post_A", "post_B")}   # fixed parts
    _S["T"].step(dict(a=q["A"], b=q["B"], fixed=fixed, hands_clear_m=hc, beam_clear_m=bc, glass_touch=gt))
    try:
        _slot_update(_S.get("n", 0) / 20.0, env, q)
    except Exception as ex:  # noqa: BLE001
        print(f"[pr] slot update error {ex!r}", file=sys.stderr, flush=True)
    _S["score"] = _S["T"].score()
    _S["n"] = _S.get("n", 0) + 1
    print(f"PRL {_S['n']} {q['A']:.5f} {q['B']:.5f}", file=sys.stderr, flush=True)   # per-step log for the close-up
    if _S["n"] % 20 == 0:
        pd = " ".join(str(np.round(P * 1e3).astype(int).tolist()) for P in _S.get("pads", []))
        print(f"[pr] t={_S['n'] / 20:5.1f} A={math.degrees(q['A']):7.1f} B={math.degrees(q['B']):7.1f} "
              f"spot=({u * 1e3:6.1f},{v * 1e3:6.1f})mm miss={PS.miss_m(q['A'], q['B']) * 1e3:6.2f} "
              f"score={_S['score']} latch={_S['T'].latch} hc={hc if hc is None else round(hc, 3)} pads L,R={pd}", file=sys.stderr, flush=True)


def score(objects):
    return float(_S.get("score", 0.0))


def success(objects):
    return score(objects) >= 1.0
