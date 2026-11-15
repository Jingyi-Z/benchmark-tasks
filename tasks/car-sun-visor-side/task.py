"""On the station table: a fixed mock-up of a car's driver-side front corner (a side window with its glass between two
pillars; the roof itself left out, the visor's pivot mount and centre clip fixed where the headliner would hold them)
and a sun visor hanging down over the windshield, its rod's tip clipped into the centre clip. The mock-up is turned
so the visor lies across in front of the arms, pivot on the left, and swings away from them to the window.

The visor: a padded board 430 x 170 x 25 mm (0.35 kg) on an 8 mm steel L-rod. The rod's short leg turns in a mount at
the outer (pillar) corner about a vertical axis (the visor swings from the windshield to the side window), the board
turns about the rod (down to shade, up to stow), and it slides along the rod by up to 120 mm (the extender). Each
joint holds where it is left by friction. The rod's tip is a pin held in the centre clip: a fixed front jaw and a
sprung rear jaw. Swinging the visor off the windshield pulls the pin back through the clip, pushing the sprung jaw
open; with the pin out it springs back. It starts down over the windshield (shading it), clipped.

The task: shade the side window. The window's glass carries a marked band, 300 x 120 mm, along its top toward the
back. Seen square-on to the glass from inside, the visor has to cover that band. Swung over without the extender it
covers about 61%; slid out 50 mm or more, 80% or more.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score --max-steps 2400 --store-frames --log-dir logs
"""

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
DEBUG = bool(os.environ.get("VISOR_DEBUG"))

TRIM = (0.30, 0.30, 0.32)  # dark grey plastic trim
POST_C = (0.22, 0.22, 0.24)
VISOR_C = (0.78, 0.74, 0.66)  # beige cloth-covered board
STEEL = (0.72, 0.73, 0.75)
CLIP_C = (0.12, 0.12, 0.13)
GLASS_C = (0.70, 0.80, 0.86)
BAND_C = (0.95, 0.55, 0.10)  # orange marking tape on the glass
labkit.GLASS_COLORS.update({GLASS_C})

# -- frame: origin on the swivel axis at the rod's height; axes along the world's. x forward (windshield), y toward the
# car's centre, z up. The visor stows along +y from the pivot, and swings to lie along -x (toward the back).
PX, PY = float(os.environ.get("VISOR_PX", "0.30")), float(os.environ.get("VISOR_PY", "0.22"))
ROD_Z = TOP + float(os.environ.get("VISOR_ROD_H", "0.220"))  # the rod this far above the bench
YAW = math.radians(float(os.environ.get("VISOR_YAW", "180")))  # the mock-up turned this far about the vertical
_CY, _SY = math.cos(YAW), math.sin(YAW)
RYAW = np.array([[_CY, -_SY, 0.0], [_SY, _CY, 0.0], [0.0, 0.0, 1.0]])
BENCH = TOP - ROD_Z  # the bench in frame coordinates

ROD_R = 0.004
ROD_L = 0.462  # pivot to the pin's tip
PANEL = (0.170, 0.430, 0.025)  # depth (away from the rod), length (along it), thickness
PANEL_Y0 = 0.012  # the board's outer end, this far along the rod from the pivot
PANEL_C = (-0.075, PANEL_Y0 + PANEL[1] / 2, 0.0)  # stowed: lying flat, back from the rod (the rod 10 mm inside it)
PANEL_MASS = 0.35
SLIDE_MAX = 0.120

GLASS_Y = -0.120  # the side glass's inner face
BAND_X = (-0.550, -0.250)  # the marked band on the glass (frame x, z)
BAND_Z = (-0.165, -0.045)
CLIP_Y = (0.446, 0.462)  # the centre clip's jaws along the rod
CLIP_YC, CLIP_YS = (CLIP_Y[0] + CLIP_Y[1]) / 2, CLIP_Y[1] - CLIP_Y[0]
JAW_K = 0.05  # N m/rad: the rear jaw's spring (estimate; about 5 N to pull the pin through)


def F(x, y, z):
    """Frame coordinates to world."""
    w = RYAW @ np.array([x, y, z], float)
    return (PX + float(w[0]), PY + float(w[1]), ROD_Z + float(w[2]))


# The visor: swivel (about z at the pivot) -> roll (about the rod, y) -> slide (along the rod). Rolling down is
# negative (the board's back edge goes down).
SWIVEL = Joint("swivel", kind="hinge", axis="Z", anchor=(0.0, 0.0, 0.0), lower=0.0, upper=math.radians(88.0),
               damping=0.02, mass=0.06,
               cylinders=[((0.0, ROD_L / 2, 0.0), ROD_R, ROD_L, STEEL, "Y"),
                          ((0.0, 0.0, 0.010), ROD_R + 0.001, 0.020, STEEL, "Z")])
ROLL = Joint("roll", kind="hinge", axis="Y", anchor=(0.0, 0.0, 0.0), lower=math.radians(-100.0), upper=0.0,
             initial=math.radians(-90.0), damping=0.02, mass=0.01, parent="swivel",
             cylinders=[((0.0, 0.006, 0.0), 0.007, 0.010, CLIP_C, "Y")])
SLIDE = Joint("slide", kind="slide", axis="Y", anchor=(0.0, 0.0, 0.0), lower=0.0, upper=SLIDE_MAX, damping=0.5,
              mass=PANEL_MASS, parent="roll",
              boxes=[(PANEL_C, PANEL, VISOR_C)])
labkit.AXIS_FRICTION.update({"swivel": (0.60, 0.50), "roll": (0.60, 0.50), "slide": (4.0, 3.0)})

VISOR = part("visor", pos=F(0.0, 0.0, 0.0), fixed=True, friction=0.8, yaw=YAW,
             boxes=[((0.0, 0.0, 0.030), (0.024, 0.024, 0.020), TRIM)],  # the pivot mount
             joints=[SWIVEL, ROLL, SLIDE])

# The roof corner, the glass and the centre clip (another fixed part, so the visor collides with it).
REAR_JAW = Joint("jaw", kind="hinge", axis="Y", anchor=(-0.0075, CLIP_YC, 0.012), lower=0.0, upper=1.4,
                 stiffness=JAW_K, damping=0.002, target=0.0, mass=0.003,
                 boxes=[((-0.0075, CLIP_YC, -0.002), (0.006, CLIP_YS, 0.028), CLIP_C)])
_GX0, _GX1 = -0.620, -0.050  # the side glass from the B- to the A-pillar (its front corner 50 mm behind the pivot)
_GZ0, _GZ1 = BENCH + 0.010, -0.040
FRAME = part(
    "roof", pos=F(0.0, 0.0, 0.0), fixed=True, friction=0.8, yaw=YAW,
    boxes=[
        ((_GX1 - 0.015, GLASS_Y - 0.025, (BENCH + _GZ1 + 0.016) / 2), (0.030, 0.030, _GZ1 + 0.016 - BENCH), POST_C),  # A-pillar
        ((-0.650, GLASS_Y - 0.025, (BENCH + _GZ1 + 0.016) / 2), (0.030, 0.030, _GZ1 + 0.016 - BENCH), POST_C),  # B-pillar
        (((_GX0 + _GX1) / 2, GLASS_Y - 0.002, (_GZ0 + _GZ1) / 2), (_GX1 - _GX0, 0.004, _GZ1 - _GZ0), GLASS_C),
        (((_GX0 + _GX1) / 2, GLASS_Y - 0.010, BENCH + 0.005), (_GX1 - _GX0 + 0.06, 0.030, 0.010), TRIM),  # sill
        (((_GX0 + _GX1) / 2, GLASS_Y - 0.010, _GZ1 + 0.008), (_GX1 - _GX0 + 0.06, 0.030, 0.016), TRIM),  # upper frame
        # the marked band: four strips of tape on the glass's inner face
        (((BAND_X[0] + BAND_X[1]) / 2, GLASS_Y + 0.0003, BAND_Z[1]), (BAND_X[1] - BAND_X[0], 0.0006, 0.004), BAND_C),
        (((BAND_X[0] + BAND_X[1]) / 2, GLASS_Y + 0.0003, BAND_Z[0]), (BAND_X[1] - BAND_X[0], 0.0006, 0.004), BAND_C),
        ((BAND_X[0], GLASS_Y + 0.0003, sum(BAND_Z) / 2), (0.004, 0.0006, BAND_Z[1] - BAND_Z[0]), BAND_C),
        ((BAND_X[1], GLASS_Y + 0.0003, sum(BAND_Z) / 2), (0.004, 0.0006, BAND_Z[1] - BAND_Z[0]), BAND_C),
        # the centre clip, fixed where the headliner would hold it (the roof itself is left out): block and front jaw
        ((0.000, CLIP_YC, 0.020), (0.024, CLIP_YS, 0.016), CLIP_C),
        ((0.0075, CLIP_YC, -0.001), (0.006, CLIP_YS, 0.026), CLIP_C),
    ],
    joints=[REAR_JAW],
)

OBJECTS = [VISOR, FRAME]


# -- kinematics of the visor's chain (frame coordinates, from the joint positions) ----------------------------------
_AXV = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}


def _rot(axis, ang):
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * kx + (1.0 - math.cos(ang)) * kx @ kx


def panel_pose(q):
    """The board's rotation and the map from its stowed (zero) frame coordinates to frame coordinates now."""
    rs = _rot(_AXV["Z"], q["swivel"])
    rr = _rot(_AXV["Y"], q["roll"])
    shift = np.array([0.0, q["slide"], 0.0])
    rot = rs @ rr

    def place(p0):
        return (rot @ (np.asarray(p0, float) + shift).T).T

    return rot, place


def panel_corners(q):
    c, s = np.array(PANEL_C), np.array(PANEL) / 2
    pts = np.array([c + s * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return panel_pose(q)[1](pts)


def _hull(pts):
    pts = sorted(set((round(float(a), 7), round(float(b), 7)) for a, b in pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _clip(poly, x0, x1, z0, z1):
    """Sutherland-Hodgman: the part of a convex polygon (x, z) inside the band's rectangle."""
    def cut(pts, inside, inter):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out

    def at_x(xc):
        return lambda a, b: (xc, a[1] + (b[1] - a[1]) * (xc - a[0]) / (b[0] - a[0]))

    def at_z(zc):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (zc - a[1]) / (b[1] - a[1]), zc)

    p = list(poly)
    for inside, inter in ((lambda v: v[0] >= x0, at_x(x0)), (lambda v: v[0] <= x1, at_x(x1)),
                          (lambda v: v[1] >= z0, at_z(z0)), (lambda v: v[1] <= z1, at_z(z1))):
        if not p:
            break
        p = cut(p, inside, inter)
    return p


def _area(poly):
    return 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))) if len(poly) > 2 else 0.0


def coverage(q) -> float:
    """Share of the marked band the visor covers, seen square-on to the glass from inside (frame y is the glass's
    normal): the band rectangle clipped by the outline of the board's eight corners projected onto the glass. The board
    lies inboard of the glass at every joint position, so it is always between the band and the driver's side."""
    poly = _hull([(p[0], p[2]) for p in panel_corners(q)])
    band = (BAND_X[1] - BAND_X[0]) * (BAND_Z[1] - BAND_Z[0])
    return _area(_clip(poly, BAND_X[0], BAND_X[1], BAND_Z[0], BAND_Z[1])) / band


# -- scoring ----------------------------------------------------------------------------------------------------------
FULL_COVER, HALF_COVER = 0.80, 0.40
HAND_CLEAR = 0.080  # at the end, each jaw endpoint and each hand's midpoint at least this far from every part
STILL_MM, STILL_N = 0.001, 21  # and no surface point moved 1 mm or more between any two of the last 21 samples (1 s)
_end = {"clear": 0.0, "hist": [], "q": None}


def _box_points(c, s, n=5):
    c, h = np.asarray(c, float), np.asarray(s, float) / 2
    g = np.linspace(-1.0, 1.0, n)
    return np.array([c + h * np.array([a, b, d]) for a in g for b in g for d in g
                     if max(abs(a), abs(b), abs(d)) == 1.0])


def _cyl_points(c, r, h, axis, n=16):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = [(r * math.cos(x), r * math.sin(x)) for x in a]
    p = np.array([(u, v, z) for z in (-h / 2, 0.0, h / 2) for u, v in ring])
    if axis == "X":
        p = p[:, [2, 0, 1]]
    elif axis == "Y":
        p = p[:, [0, 2, 1]]
    return p + np.asarray(c, float)


def _box_dist(p, c, s):
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(p, float) - np.asarray(c, float)) - np.asarray(s) / 2, 0)))


def _cyl_dist(p, c, r, h, axis):
    ax = _AXV[axis]
    v = np.asarray(p, float) - np.asarray(c, float)
    a = float(v @ ax)
    return math.hypot(max(float(np.linalg.norm(v - a * ax)) - r, 0.0), max(abs(a) - h / 2, 0.0))


def _to_zero(p, q, joint):
    """A frame point expressed in the stowed coordinates of a visor joint's body (the chain undone)."""
    p = np.asarray(p, float)
    p = _rot(_AXV["Z"], -q["swivel"]) @ p
    if joint == "swivel":
        return p
    p = _rot(_AXV["Y"], -q["roll"]) @ p
    if joint == "roll":
        return p
    return p - np.array([0.0, q["slide"], 0.0])


def part_distance(p_frame, q, jaw_open) -> float:
    """Shortest distance (m) from a frame point to any part's solid, the visor's bodies and the clip's sprung jaw at
    their current positions (0 inside)."""
    best = math.inf
    for b in FRAME.boxes + VISOR.boxes:
        best = min(best, _box_dist(p_frame, b[0], b[1]))
    jz = _rot(_AXV["Y"], -jaw_open) @ (np.asarray(p_frame, float) - np.array(REAR_JAW.anchor)) + np.array(REAR_JAW.anchor)
    for b in REAR_JAW.boxes:
        best = min(best, _box_dist(jz, b[0], b[1]))
    for j in (SWIVEL, ROLL, SLIDE):
        z = _to_zero(p_frame, q, j.name)
        for b in j.boxes:
            best = min(best, _box_dist(z, b[0], b[1]))
        for c in j.cylinders:
            best = min(best, _cyl_dist(z, c[0], c[1], c[2], c[4]))
    return best


_STATIC = np.vstack([_box_points(b[0], b[1]) for b in FRAME.boxes + VISOR.boxes])


def surface_points(q, jaw_open):
    """Frame points over every part's surface at the current joint positions (for stillness)."""
    a = np.array(REAR_JAW.anchor)
    jaw = np.vstack([(_rot(_AXV["Y"], jaw_open) @ (_box_points(b[0], b[1]) - a).T).T + a for b in REAR_JAW.boxes])
    rs = _rot(_AXV["Z"], q["swivel"])
    rod = np.vstack([_cyl_points(c[0], c[1], c[2], c[4]) for c in SWIVEL.cylinders])
    board = panel_pose(q)[1](np.vstack([_box_points(b[0], b[1]) for b in SLIDE.boxes]))
    return np.vstack([_STATIC, jaw, (rs @ rod.T).T, board])


def _tips(env):
    """Each hand's two pad tips (world, env frame), then the midpoint between them."""
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
        out += [mid - q7 * g[:3, 0], mid + q8 * g[:3, 0]]  # joint7's pad towards -x, joint8's towards +x
    return out + [(out[0] + out[1]) / 2, (out[2] + out[3]) / 2]


def _joints(env):
    v = labkit._part_state(env, labkit._SPECS["visor"])
    r = labkit._part_state(env, labkit._SPECS["roof"])
    q = {k: float(v.joints.get(k, 0.0)) for k in ("swivel", "roll", "slide")}
    return q, float(r.joints.get("jaw", 0.0))


@on_reset
def _reset(env):
    _end.update(clear=0.0, hist=[], q=None)
    _seen.update(q=None)


@on_step
def _watch(env):
    q, jaw = _joints(env)
    origin = np.array([PX, PY, ROD_Z])
    _end["clear"] = float(min(part_distance(RYAW.T @ (t - origin), q, jaw) for t in _tips(env)))
    _end["hist"] = (_end["hist"] + [surface_points(q, jaw)])[-STILL_N:]
    _end["q"] = q
    _seen.update(q=q)  # the visor's pose as a camera sees it (the board's position), for the demo's aim
    if DEBUG:
        _seen["n"] = _seen.get("n", 0) + 1
        if _seen["n"] % 20 == 0:
            print(f"[visor] t={_seen['n'] / 20:5.1f}s swivel={math.degrees(q['swivel']):6.1f} roll="
                  f"{math.degrees(q['roll']):6.1f} slide={q['slide'] * 1000:5.1f}mm jaw={math.degrees(jaw):5.1f} "
                  f"cover={coverage(q):.2f} clear={_end['clear'] * 1000:.0f}mm", file=sys.stderr, flush=True)


def _still() -> bool:
    h = _end["hist"]
    if len(h) < STILL_N:
        return False
    moved = max(float(np.max(np.linalg.norm(a - b, axis=1))) for i, a in enumerate(h) for b in h[i + 1:])
    return moved < STILL_MM


def score(objects) -> float:
    """End state only. 0 unless each hand's two jaw endpoints and their midpoint are 80 mm or more from every part
    and no surface point of any part moved 1 mm or more between any two of the last 21 samples (1 s at 20 Hz). Then
    1.0 if the visor covers 80% or more of the marked 300 x 120 mm band on the side glass, seen square-on to the glass
    from inside; 0.5 at 40% or more; 0 below."""
    v = objects["visor"]
    q = {k: float(v.joints.get(k, 0.0)) for k in ("swivel", "roll", "slide")}
    if _end["clear"] < HAND_CLEAR or not _still():
        return 0.0
    c = coverage(q)
    return 1.0 if c >= FULL_COVER else 0.5 if c >= HALF_COVER else 0.0


def success(objects) -> bool:
    return score(objects) == 1.0


def distance(objects) -> float:
    v = objects["visor"]
    q = {k: float(v.joints.get(k, 0.0)) for k in ("swivel", "roll", "slide")}
    return 1.0 - coverage(q)


# -- scripted demo ----------------------------------------------------------------------------------------------------
# It reads only what a camera sees: the board's pose (from the joint positions, which fix where the board is). It
# never reads the clip's jaw or any force.
_seen: dict = {"q": None}


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


def frame_rot(approach, closing):
    """A TCP rotation from its approach direction and fingers' closing direction (world)."""
    a = np.asarray(approach, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(closing, float)
    c = c - a * (c @ a)
    c = c / np.linalg.norm(c)
    return Orient(np.column_stack([c, np.cross(a, c), a]))


PAD_PAST_TCP = 0.0035  # the pads' centres this far past the TCP along the approach
SHUT_ON_BOARD = -0.0105  # pads round the 25 mm board, squeezed 2 mm
OPEN_WIDE = -0.024
LEFT_PARK = (0.15, 0.45, TOP + 0.16)
RIGHT_PARK = (0.10, -0.42, TOP + 0.16)
PINCH_R0, PINCH_R1 = 0.180, 0.260  # the pinch this far from the pivot along the rod, before and after the slide
PINCH_DOWN = 0.020  # the pads' centres this far below the rod (the board's top edge is 10 mm above it)
TILT = math.radians(float(os.environ.get("VISOR_TILT", "0")))  # fingertips leaning toward the pivot
SWING_END = math.radians(86.0)


def _w(t, pos, finger, label, rot):
    return Waypoint(t, tuple(float(v) for v in np.asarray(pos, float)), finger, label, rot)


def _dur(wps):
    return sum(w.duration for w in wps)


def pinch(sw, r, up=0.0):
    """TCP and rotation for the right hand pointing down, its fingers closing across the hanging board's thickness,
    the pads ``r`` along the rod from the pivot, the board swung ``sw`` (rad); ``up`` lifts it."""
    along = RYAW @ np.array([math.sin(sw), -math.cos(sw), 0.0])  # from the board's inner end toward the pivot
    across = RYAW @ np.array([math.cos(sw), math.sin(sw), 0.0])
    a = np.array([0.0, 0.0, -math.cos(TILT)]) - along * math.sin(TILT)
    rot = frame_rot(a, across)
    pad = np.array(F(0.0, 0.0, -PINCH_DOWN + up)) - along * r
    return pad - rot.rot[:, 2] * PAD_PAST_TCP, rot


TIP_PAST_TCP = 0.0128  # the shut fingertips this far past the TCP
PUSH_DOWN = float(os.environ.get("VISOR_PUSH_DOWN", "0.030"))  # pushes land this far below the rod
PUSH_TILT = math.radians(float(os.environ.get("VISOR_PUSH_TILT", "20")))  # fingertips pointing this far below level
PUSH_INTO = 0.004  # each push target is this far inside the face, so the fingertips keep pressing
FINGERS_SHUT = 0.0


def _wv(v):
    return RYAW @ np.asarray(v, float)


def face_push(sw, r, slide=0.0, back=0.0):
    """TCP and rotation for a hand whose shut fingertips press the board's robot-side face (its windshield-side face
    when stowed), ``r`` along the rod from the pivot and PUSH_DOWN below the rod, the board swung ``sw``; the push
    is along the face's normal, level but for PUSH_TILT. ``back`` holds the tips that far off the face."""
    n_in = _wv([-math.cos(sw), -math.sin(sw), 0.0])  # into the board from its robot-side face
    pt = np.array(F(*(np.array([math.cos(sw), math.sin(sw), 0.0]) * (PANEL[2] / 2 + 0.010)
                     + np.array([-math.sin(sw), math.cos(sw), 0.0]) * (r + slide) + [0.0, 0.0, -PUSH_DOWN])))
    pt = pt - np.array(_wv([math.cos(sw), math.sin(sw), 0.0])) * 0.010  # back onto the face itself
    a = n_in * math.cos(PUSH_TILT) + np.array([0.0, 0.0, -math.sin(PUSH_TILT)])
    rot = frame_rot(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
    tip = pt + n_in * (PUSH_INTO - back)
    return tip - rot.rot[:, 2] * TIP_PAST_TCP, rot


END_DOWN = float(os.environ.get("VISOR_END_DOWN", "0.110"))  # the end push lands this far below the rod: the hand
# follows the board out under the rod (a push along the rod turns nothing, so it can land low)
END_TILT = math.radians(float(os.environ.get("VISOR_END_TILT", "0")))


def end_push(sw, slide, back=0.0):
    """TCP and rotation for a hand whose shut fingertips press the board's outer end (by the pivot), PUSH_DOWN below
    the rod, pushing it out along the rod by ``slide``."""
    along_out = _wv([-math.sin(sw), math.cos(sw), 0.0])  # along the rod, away from the pivot
    pt = np.array(F(*(np.array([-math.sin(sw), math.cos(sw), 0.0]) * (PANEL_Y0 + slide) + [0.0, 0.0, -END_DOWN])))
    a = along_out * math.cos(END_TILT) + np.array([0.0, 0.0, -math.sin(END_TILT)])
    rot = frame_rot(a, [0.0, 0.0, 1.0])
    tip = pt + along_out * (PUSH_INTO - back)
    return tip - rot.rot[:, 2] * TIP_PAST_TCP, rot


class _EndAt:
    """A TCP target for the end push, aimed at the swing angle the visor is actually seen at (a camera sees the board's
    pose); planned SWING_END until it is seen."""

    def __init__(self, slide, back=0.0):
        self.slide, self.back = slide, back

    def __array__(self, dtype=None, copy=None):
        q = _seen.get("q")
        sw = float(q["swivel"]) if q else SWING_END
        return np.asarray(end_push(sw, self.slide, self.back)[0], dtype=dtype or float)


SLIDE_TO = float(os.environ.get("VISOR_SLIDE", "0.100"))
R_NEAR, R_MID = float(os.environ.get("VISOR_R1", "0.25")), float(os.environ.get("VISOR_R2", "0.20"))
SLIDE_ARM = os.environ.get("VISOR_SLIDE_ARM", "left")
SW_HAND = math.radians(float(os.environ.get("VISOR_SW_HAND", "50")))


PLANNED: list = []  # (time, swivel, slide, contact) along the demo, for the offline reach and collision check


def demo():
    _seen.update(q=None)
    PLANNED.clear()
    right = []
    tcp, rot = face_push(0.0, R_NEAR, back=0.060)
    right += [_w(3.0, (0.20, -0.40, TOP + 0.24), FINGERS_SHUT, "out, clear of the centre clip", rot),
              _w(3.0, (0.16, -0.42, tcp[2]), FINGERS_SHUT, "down, outside the visor's inner end", rot),
              _w(2.5, tcp, FINGERS_SHUT, "in front of the visor's face, near its inner end", rot)]
    tcp, rot = face_push(0.0, R_NEAR, back=0.004)
    right += [_w(1.5, tcp, FINGERS_SHUT, "fingertips to the face", rot)]
    for sw in np.linspace(0.0, SW_HAND, 11)[1:]:
        tcp, rot = face_push(float(sw), R_NEAR)
        right.append(_w(1.0 if sw < 0.15 else 0.8, tcp, FINGERS_SHUT, "push it off the centre clip and round", rot))
        PLANNED.append((_dur(right), float(sw), 0.0, "right"))
    tcp, rot = face_push(SW_HAND, R_NEAR, back=0.050)
    right += [_w(1.0, tcp, FINGERS_SHUT, "off the face", rot)]
    tcp, rot = face_push(SW_HAND, R_MID, back=0.050)
    right += [_w(1.5, tcp, FINGERS_SHUT, "nearer the pivot", rot)]
    tcp, rot = face_push(SW_HAND, R_MID, back=0.004)
    right += [_w(1.0, tcp, FINGERS_SHUT, "fingertips to the face", rot)]
    for sw in np.linspace(SW_HAND, SWING_END, 8)[1:]:
        tcp, rot = face_push(float(sw), R_MID)
        right.append(_w(0.8, tcp, FINGERS_SHUT, "push it on round to the side window", rot))
        PLANNED.append((_dur(right), float(sw), 0.0, "right"))
    tcp, rot = face_push(SWING_END, R_MID, back=0.090)
    right += [_w(1.5, tcp, FINGERS_SHUT, "back off", rot),
              _w(2.0, (0.30, 0.05, TOP + 0.10), FINGERS_SHUT, "down and back toward the bench's middle", rot),
              _w(2.0, (0.15, -0.12, TOP + 0.10), FINGERS_SHUT, "back past the centre clip, low", 0.0)]
    if SLIDE_ARM == "right":
        tcp, rot = end_push(SWING_END, 0.0, back=0.060)
        right += [_w(2.5, tcp, FINGERS_SHUT, "behind the visor's outer end", rot)]
        tcp, rot = end_push(SWING_END, 0.0, back=0.004)
        right += [_w(1.5, tcp, FINGERS_SHUT, "fingertips to its end", rot)]
        for sl in np.linspace(0.0, SLIDE_TO, 6)[1:]:
            tcp, rot = end_push(SWING_END, float(sl))
            right.append(_w(0.8, tcp, FINGERS_SHUT, "push the extender out along the rod", rot))
            PLANNED.append((_dur(right), SWING_END, float(sl), "right"))
        tcp, rot = end_push(SWING_END, SLIDE_TO, back=0.090)
        right += [_w(1.5, tcp, FINGERS_SHUT, "back off", rot)]
    right += [_w(2.5, RIGHT_PARK, FINGER_OPEN, "park", 0.0)]
    t_swung = _dur(right) - 4.0
    if SLIDE_ARM == "right":
        t_end = _dur(right) + 2.0
        right.append(_w(2.0, RIGHT_PARK, FINGER_OPEN, "rest", 0.0))
        return {"left": [_w(3.0, (0.25, 0.42, TOP + 0.25), FINGER_OPEN, "out, clear of the visor's mount", 0.0),
                         _w(t_end - 2.0, LEFT_PARK, FINGER_OPEN, "wait", 0.0)], "right": right}

    left = [_w(3.0, (0.25, 0.42, TOP + 0.25), FINGER_OPEN, "out, clear of the visor's mount", 0.0),
            _w(2.0, LEFT_PARK, FINGER_OPEN, "wait", 0.0)]
    tcp, rot = end_push(SWING_END, 0.0, back=0.060)
    left.append(_w(max(0.3, t_swung - _dur(left)), LEFT_PARK, FINGER_OPEN, "wait", 0.0))
    left += [Waypoint(3.0, _EndAt(0.0, back=0.060), FINGERS_SHUT, "behind the visor's outer end, where it is seen", rot)]
    left += [Waypoint(1.5, _EndAt(0.0, back=0.004), FINGERS_SHUT, "fingertips to its end", rot)]
    for sl in np.linspace(0.0, SLIDE_TO, 6)[1:]:
        left.append(Waypoint(0.8, _EndAt(float(sl)), FINGERS_SHUT, "push the extender out along the rod", rot))
        PLANNED.append((_dur(left), SWING_END, float(sl), "left"))
    tcp, rot = end_push(SWING_END, SLIDE_TO, back=0.090)
    left += [Waypoint(1.5, _EndAt(SLIDE_TO, back=0.090), FINGERS_SHUT, "back off", rot),
             _w(2.5, LEFT_PARK, FINGER_OPEN, "park", 0.0)]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(_w(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest", 0.0))
    right.append(_w(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest", 0.0))
    return {"left": left, "right": right}
