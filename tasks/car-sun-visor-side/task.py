"""On the station table: a self-supporting benchtop mock-up of a car's driver-side front roof corner with an overhead
roof panel, molded cloth headliner ceiling around both mounting bosses, driver-side roof cantrail, A-pillar, B-pillar,
right header stanchion, and a shatter-safe side window pane whose four edges are fully encased in black EPDM rubber
U-channel weatherstrip molding. Mounted into the underside of the headliner are an extending automotive sun visor
(2013-2017 Honda Accord EX Left Driver Sliding Sun Visor 83280-T2F-A52ZB or 1:1 printable STL equivalent from
generate_cad_stl.py: 430 x 170 x 25 mm padded blade, 0.35 kg, on an 8 mm steel L-rod 330 mm above the bench) and its
matching snap-in centre retainer clip (Honda 88217-TR0-A01ZA: fixed front jaw and torsion-sprung rear jaw).

The visor hangs down over the windshield aperture with its rod tip clipped into the centre clip. Reaching underneath
the overhead headliner from the cabin/front side, the robot frees the visor rod from the sprung centre clip, swings the
visor ~86 degrees across to the driver's side window, and slides the blade out along its L-rod so its projected outline
covers at least 80% of the 300 x 120 mm orange target band on the side glass, then moves both hands at least 80 mm clear.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score --max-steps 2400 --store-frames --log-dir logs
"""

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401,E402
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
DEBUG = bool(os.environ.get("VISOR_DEBUG"))
MODE = os.environ.get("VISOR_MODE", "demo").strip().lower()

TRIM = (0.28, 0.29, 0.31)          # interior pillar / header trim
POST_C = (0.20, 0.21, 0.23)        # structural pillar / 2020 stanchion
ROOF_SKIN_C = (0.23, 0.25, 0.28)   # painted steel outer roof panel
HEADLINER_C = (0.73, 0.71, 0.67)   # molded woven automotive headliner fabric
RUBBER_C = (0.11, 0.11, 0.12)      # black EPDM rubber U-channel glass-edge weatherstrip
VISOR_C = (0.78, 0.74, 0.66)       # beige/light-grey cloth sun-visor blade
MIRROR_TRIM_C = (0.58, 0.56, 0.52) # flush vanity-mirror sliding cover bezel
STEEL = (0.72, 0.73, 0.75)
CLIP_C = (0.14, 0.14, 0.15)
GLASS_C = (0.70, 0.80, 0.86)
BAND_C = (0.95, 0.55, 0.10)        # orange marking tape on the side window
labkit.GLASS_COLORS.update({GLASS_C})

# -- frame: origin on the swivel axis at the rod's height; x forward (windshield header), y toward the car's centre,
# z up. Turned 180 deg so the visor swings away from the arm bases (+X world) toward the left-hand side window.
PX, PY = float(os.environ.get("VISOR_PX", "0.34")), float(os.environ.get("VISOR_PY", "0.22"))
ROD_Z = TOP + float(os.environ.get("VISOR_ROD_H", "0.330"))  # 330 mm above bench (installed cabin headliner height)
YAW = math.radians(float(os.environ.get("VISOR_YAW", "180")))
_CY, _SY = math.cos(YAW), math.sin(YAW)
RYAW = np.array([[_CY, -_SY, 0.0], [_SY, _CY, 0.0], [0.0, 0.0, 1.0]])
BENCH = TOP - ROD_Z  # -0.330 m in frame coordinates

ROD_R = 0.004
ROD_L = 0.462  # pivot axis to rod pin tip
PANEL = (0.170, 0.430, 0.025)  # depth (away from rod), length (along rod), thickness
PANEL_Y0 = 0.012  # outer end of blade, 12 mm inboard of pivot
PANEL_C = (-0.075, PANEL_Y0 + PANEL[1] / 2, 0.0)  # stowed flat: rod 10 mm inside top edge
PANEL_MASS = 0.35
SLIDE_MAX = 0.120

GLASS_Y = -0.120  # inner face of side window pane
BAND_X = (-0.550, -0.250)  # 300 x 120 mm marked target band on side window (frame x, z)
BAND_Z = (-0.165, -0.045)
CLIP_Y = (0.446, 0.462)    # centre clip jaws along the rod
CLIP_YC, CLIP_YS = (CLIP_Y[0] + CLIP_Y[1]) / 2, CLIP_Y[1] - CLIP_Y[0]
JAW_K = 0.05  # N m/rad: sprung rear jaw torsion spring (engineering estimate; ~5 N pull-out)


def F(x, y, z):
    """Frame coordinates to world."""
    w = RYAW @ np.array([x, y, z], float)
    return (PX + float(w[0]), PY + float(w[1]), ROD_Z + float(w[2]))


# Visor kinematic chain: swivel (about Z at outer pivot) -> roll (about rod Y) -> slide (along rod Y).
SWIVEL = Joint(
    "swivel", kind="hinge", axis="Z", anchor=(0.0, 0.0, 0.0), lower=0.0, upper=math.radians(88.0),
    damping=0.02, mass=0.06,
    cylinders=[
        ((0.0, ROD_L / 2, 0.0), ROD_R, ROD_L, STEEL, "Y"),
        ((0.0, 0.0, 0.010), ROD_R + 0.001, 0.020, STEEL, "Z"),
    ],
)
ROLL = Joint(
    "roll", kind="hinge", axis="Y", anchor=(0.0, 0.0, 0.0), lower=math.radians(-100.0), upper=0.0,
    initial=math.radians(-90.0), damping=0.02, mass=0.01, parent="swivel",
    cylinders=[((0.0, 0.006, 0.0), 0.007, 0.010, CLIP_C, "Y")],
)
# Main 430 x 170 x 25 mm padded visor blade (first box defines outer shading envelope) + recessed vanity cover detail
SLIDE = Joint(
    "slide", kind="slide", axis="Y", anchor=(0.0, 0.0, 0.0), lower=0.0, upper=SLIDE_MAX, damping=0.5,
    mass=PANEL_MASS, parent="roll",
    boxes=[
        (PANEL_C, PANEL, VISOR_C),
        ((-0.080, PANEL_Y0 + 0.225, -0.0122), (0.082, 0.165, 0.0012), MIRROR_TRIM_C),
    ],
)
labkit.AXIS_FRICTION.update({"swivel": (0.60, 0.50), "roll": (0.60, 0.50), "slide": (4.0, 3.0)})

VISOR = part(
    "visor", pos=F(0.0, 0.0, 0.0), fixed=True, friction=0.8, yaw=YAW,
    boxes=[((0.0, 0.0, 0.030), (0.028, 0.028, 0.020), TRIM)],  # outer swivel bezel bolted into headliner boss
    joints=[SWIVEL, ROLL, SLIDE],
)

# Roof corner, molded headliner ceiling, EPDM rubber-trimmed side window, pillars, and sprung centre clip.
REAR_JAW = Joint(
    "jaw", kind="hinge", axis="Y", anchor=(-0.0075, CLIP_YC, 0.012), lower=0.0, upper=1.4,
    stiffness=JAW_K, damping=0.002, target=0.0, mass=0.003,
    boxes=[((-0.0075, CLIP_YC, -0.002), (0.006, CLIP_YS, 0.028), CLIP_C)],
)

_GX0, _GX1 = -0.620, -0.050  # 570 mm side window pane from B-pillar to A-pillar
_GZ0, _GZ1 = BENCH + 0.018, -0.040  # 272 mm pane height (sill at BENCH+18 mm, upper channel at -40 mm)

# Overhead roof & headliner envelope:
#   - Windshield header + stowed-visor ceiling canopy: frame x in [-0.156, +0.024] (180 mm deep),
#     y in [-0.150, +0.530] (680 mm wide, spanning from left A-pillar past pivot y=0 and centre clip y=+0.454 to right stanchion),
#     molded headliner ceiling at z in [+0.038, +0.050] + steel outer roof skin at z in [+0.050, +0.058],
#     with downward-molded mounting bosses at z in [+0.028, +0.038] enveloping the pivot bezel and centre clip base.
#   - Driver-side cantrail / door roof header: frame x in [-0.665, -0.156] (509 mm rearward to B-pillar),
#     y in [-0.150, -0.060] (90 mm wide overhanging the side window), headliner valence z in [-0.024, +0.038] + roof skin [+0.038, +0.058].
FRAME = part(
    "roof", pos=F(0.0, 0.0, 0.0), fixed=True, friction=0.8, yaw=YAW,
    boxes=[
        # 1. Three structural support posts standing solidly on the bench (BENCH = -0.330 m up to roof header z = +0.044 m)
        ((-0.065, -0.145, (BENCH + 0.044) / 2), (0.030, 0.030, 0.044 - BENCH), POST_C),   # Left front A-pillar
        ((-0.650, -0.145, (BENCH + 0.044) / 2), (0.030, 0.030, 0.044 - BENCH), POST_C),   # Left rear B-pillar
        ((-0.125, +0.515, (BENCH + 0.044) / 2), (0.030, 0.030, 0.044 - BENCH), POST_C),   # Right header stanchion post
        ((-0.080, +0.515, BENCH + 0.008), (0.120, 0.030, 0.016), POST_C),                 # Right stanchion base foot

        # 2. Overhead windshield header + stowed-visor molded headliner canopy & painted steel outer roof panel
        ((-0.068, +0.185, +0.050), (0.176, 0.690, 0.012), HEADLINER_C),  # Molded cloth headliner ceiling (176 x 690 mm)
        ((-0.068, +0.185, +0.060), (0.176, 0.690, 0.008), ROOF_SKIN_C),  # Outer painted steel roof panel (176 x 690 mm)
        ((+0.020, +0.185, +0.052), (0.008, 0.690, 0.020), RUBBER_C),     # Front windshield header EPDM pinch-weld trim
        ((-0.010, +0.000, +0.036), (0.056, 0.072, 0.016), HEADLINER_C),  # Molded headliner boss around outer pivot mount
        ((-0.010, CLIP_YC, +0.036), (0.056, 0.072, 0.016), HEADLINER_C), # Molded headliner boss around centre clip

        # 3. Driver-side cantrail roof rail & headliner soffit connecting A-pillar/header to B-pillar above side window
        ((-0.4105, -0.110, +0.054), (0.509, 0.090, 0.020), ROOF_SKIN_C),  # Side roof cantrail sheet-metal rail
        ((-0.4105, -0.110, +0.032), (0.509, 0.090, 0.024), HEADLINER_C),  # Side cantrail molded headliner valence
        ((-0.345, -0.135, -0.006), (0.590, 0.030, 0.040), TRIM),          # Upper door-frame header rail

        # 4. Side window pane + lower beltline sill + full 4-edge black EPDM rubber U-channel weatherstrip edge guards
        (((_GX0 + _GX1) / 2, GLASS_Y - 0.002, (_GZ0 + _GZ1) / 2), (_GX1 - _GX0, 0.004, _GZ1 - _GZ0), GLASS_C),
        (((_GX0 + _GX1) / 2, GLASS_Y - 0.010, BENCH + 0.009), (_GX1 - _GX0 + 0.060, 0.030, 0.018), TRIM),     # Lower sill
        (((_GX0 + _GX1) / 2, GLASS_Y - 0.002, _GZ1 + 0.004), (_GX1 - _GX0 + 0.016, 0.014, 0.014), RUBBER_C), # Top EPDM U-channel
        (((_GX0 + _GX1) / 2, GLASS_Y - 0.002, _GZ0 - 0.004), (_GX1 - _GX0 + 0.016, 0.014, 0.014), RUBBER_C), # Bottom EPDM U-channel
        ((_GX1 + 0.003, GLASS_Y - 0.002, (_GZ0 + _GZ1) / 2), (0.014, 0.014, _GZ1 - _GZ0 + 0.016), RUBBER_C),  # Front EPDM U-channel
        ((_GX0 - 0.003, GLASS_Y - 0.002, (_GZ0 + _GZ1) / 2), (0.014, 0.014, _GZ1 - _GZ0 + 0.016), RUBBER_C),  # Rear EPDM U-channel

        # 5. Marked 300 x 120 mm orange target band on the inner face of the side glass
        (((BAND_X[0] + BAND_X[1]) / 2, GLASS_Y + 0.0003, BAND_Z[1]), (BAND_X[1] - BAND_X[0], 0.0006, 0.004), BAND_C),
        (((BAND_X[0] + BAND_X[1]) / 2, GLASS_Y + 0.0003, BAND_Z[0]), (BAND_X[1] - BAND_X[0], 0.0006, 0.004), BAND_C),
        ((BAND_X[0], GLASS_Y + 0.0003, sum(BAND_Z) / 2), (0.004, 0.0006, BAND_Z[1] - BAND_Z[0]), BAND_C),
        ((BAND_X[1], GLASS_Y + 0.0003, sum(BAND_Z) / 2), (0.004, 0.0006, BAND_Z[1] - BAND_Z[0]), BAND_C),

        # 6. Centre retainer clip base & fixed front jaw bolted directly into the headliner boss at z = +0.028
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
    """Share of the marked band the visor covers, seen square-on to the glass from inside."""
    poly = _hull([(p[0], p[2]) for p in panel_corners(q)])
    band = (BAND_X[1] - BAND_X[0]) * (BAND_Z[1] - BAND_Z[0])
    return _area(_clip(poly, BAND_X[0], BAND_X[1], BAND_Z[0], BAND_Z[1])) / band


# -- scoring ----------------------------------------------------------------------------------------------------------
FULL_COVER, HALF_COVER = 0.80, 0.40
HAND_CLEAR = 0.080  # all 6 independent fingertip & pad-midpoint points >= 80 mm from every part
STILL_MM, STILL_N = 0.001, 21  # no surface point moves >= 1 mm across the final 21 samples (1.0 s at 20 Hz)
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
    """Shortest distance (m) from a frame point to any solid of the roof/headliner/glass frame, visor, or clip jaw."""
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
    """Frame points over every part's surface at the current joint positions (for 21-sample stillness)."""
    a = np.array(REAR_JAW.anchor)
    jaw = np.vstack([(_rot(_AXV["Y"], jaw_open) @ (_box_points(b[0], b[1]) - a).T).T + a for b in REAR_JAW.boxes])
    rs = _rot(_AXV["Z"], q["swivel"])
    rod = np.vstack([_cyl_points(c[0], c[1], c[2], c[4]) for c in SWIVEL.cylinders])
    board = panel_pose(q)[1](np.vstack([_box_points(b[0], b[1]) for b in SLIDE.boxes]))
    return np.vstack([_STATIC, jaw, (rs @ rod.T).T, board])


def _tips(env):
    """All 6 independent gripper points in world env frame: each jaw pad tip (-x via joint7, +x via joint8) plus the
    independent gripper pad-center datum ``mid`` on both left and right robots."""
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
        out += [mid - q7 * g[:3, 0], mid + q8 * g[:3, 0], mid]
    return out


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
    _seen.update(q=q)
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
    """End state only. 0 unless all 6 independent jaw/pad points are >= 80 mm from every part (including the roof and
    headliner) and no surface point moved >= 1 mm across the final 21 samples (1 s at 20 Hz). Then 1.0 if the visor
    covers >= 80% of the marked 300 x 120 mm band on the side glass (seen square-on from inside); 0.5 at >= 40%; 0 below."""
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


# -- scripted demo (under-headliner cabin reach, zero wrist/elbow protrusion into roof) -------------------------------
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


TIP_PAST_TCP = 0.0128
PUSH_DOWN_CLIP = float(os.environ.get("VISOR_PUSH_CLIP", "0.038"))  # near rod while freeing clip (low roll torque)
PUSH_DOWN_SWING = float(os.environ.get("VISOR_PUSH_SWING", "0.080"))  # lower under headliner once unclipped
PUSH_TILT = math.radians(float(os.environ.get("VISOR_PUSH_TILT", "0")))  # level approach keeps wrist below headliner
PUSH_INTO = 0.004
FINGERS_SHUT = 0.0
LEFT_PARK = (0.14, 0.45, TOP + 0.16)
RIGHT_PARK = (0.10, -0.42, TOP + 0.16)
SWING_END = math.radians(86.0)


def _w(t, pos, finger, label, rot):
    return Waypoint(t, tuple(float(v) for v in np.asarray(pos, float)), finger, label, rot)


def _dur(wps):
    return sum(w.duration for w in wps)


def _wv(v):
    return RYAW @ np.asarray(v, float)


def _push_down_at(sw: float) -> float:
    """Smoothly lowers the push height from 38 mm below the rod during clip release (0..8 deg) to 80 mm below the rod
    by 24 deg so the forearm and wrist stay well beneath the overhead molded headliner canopy."""
    s_deg = math.degrees(sw)
    if s_deg <= 8.0:
        return PUSH_DOWN_CLIP
    if s_deg >= 24.0:
        return PUSH_DOWN_SWING
    u = (s_deg - 8.0) / 16.0
    return PUSH_DOWN_CLIP + u * (PUSH_DOWN_SWING - PUSH_DOWN_CLIP)


def face_push(sw, r, slide=0.0, back=0.0, pd_override=None):
    """Shut-fingertip press on the visor's front/cabin-accessible face beneath the headliner."""
    pd = _push_down_at(sw) if pd_override is None else float(pd_override)
    n_in = _wv([-math.cos(sw), -math.sin(sw), 0.0])
    pt = np.array(F(*(np.array([math.cos(sw), math.sin(sw), 0.0]) * (PANEL[2] / 2)
                     + np.array([-math.sin(sw), math.cos(sw), 0.0]) * (r + slide) + [0.0, 0.0, -pd])))
    a = n_in * math.cos(PUSH_TILT) + np.array([0.0, 0.0, -math.sin(PUSH_TILT)])
    rot = frame_rot(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
    tip = pt + n_in * (PUSH_INTO - back)
    return tip - rot.rot[:, 2] * TIP_PAST_TCP, rot


END_DOWN = float(os.environ.get("VISOR_END_DOWN", "0.110"))
END_TILT = math.radians(float(os.environ.get("VISOR_END_TILT", "0")))


def end_push(sw, slide, back=0.0):
    """Shut-fingertip press on the visor's outer end 110 mm below the rod, sliding the blade rearward along the rod."""
    along_out = _wv([-math.sin(sw), math.cos(sw), 0.0])
    pt = np.array(F(*(np.array([-math.sin(sw), math.cos(sw), 0.0]) * (PANEL_Y0 + slide) + [0.0, 0.0, -END_DOWN])))
    a = along_out * math.cos(END_TILT) + np.array([0.0, 0.0, -math.sin(END_TILT)])
    rot = frame_rot(a, [0.0, 0.0, 1.0])
    tip = pt + along_out * (PUSH_INTO - back)
    return tip - rot.rot[:, 2] * TIP_PAST_TCP, rot


class _EndAt:
    """Camera-observable end-push target aligned to the visor's seen swing angle."""

    def __init__(self, slide, back=0.0):
        self.slide, self.back = slide, back

    def __array__(self, dtype=None, copy=None):
        q = _seen.get("q")
        sw = float(q["swivel"]) if q else SWING_END
        return np.asarray(end_push(sw, self.slide, self.back)[0], dtype=dtype or float)


SLIDE_TO = float(os.environ.get("VISOR_SLIDE", "0.100"))
R_NEAR, R_MID = float(os.environ.get("VISOR_R1", "0.25")), float(os.environ.get("VISOR_R2", "0.20"))
SW_HAND = math.radians(float(os.environ.get("VISOR_SW_HAND", "50")))

PLANNED: list = []


def demo():
    _seen.update(q=None)
    PLANNED.clear()
    right = []

    if MODE == "unclipped_partial":
        sw_partial = math.radians(28.0)
        tcp, rot = face_push(0.0, R_NEAR, back=0.060)
        right += [
            _w(2.5, (0.18, -0.40, TOP + 0.22), FINGERS_SHUT, "approach below headliner", rot),
            _w(2.5, tcp, FINGERS_SHUT, "in front of visor face", rot),
        ]
        tcp, rot = face_push(0.0, R_NEAR, back=0.004)
        right.append(_w(1.2, tcp, FINGERS_SHUT, "touch face", rot))
        for sw in np.linspace(0.0, sw_partial, 7)[1:]:
            tcp, rot = face_push(float(sw), R_NEAR)
            right.append(_w(0.9, tcp, FINGERS_SHUT, "unclip only partway", rot))
            PLANNED.append((_dur(right), float(sw), 0.0, "right"))
        tcp, rot = face_push(sw_partial, R_NEAR, back=0.080)
        right += [
            _w(1.5, tcp, FINGERS_SHUT, "back off", rot),
            _w(2.5, RIGHT_PARK, FINGER_OPEN, "park", 0.0),
            _w(2.0, RIGHT_PARK, FINGER_OPEN, "rest", 0.0),
        ]
        t_end = _dur(right)
        left = [
            _w(2.5, LEFT_PARK, FINGER_OPEN, "park", 0.0),
            _w(t_end - 2.5, LEFT_PARK, FINGER_OPEN, "rest", 0.0),
        ]
        return {"left": left, "right": right}

    tcp, rot = face_push(0.0, R_NEAR, back=0.060)
    right += [
        _w(2.5, (0.20, -0.34, TOP + 0.22), FINGERS_SHUT, "approach below headliner", rot),
        _w(2.5, (0.22, -0.26, tcp[2]), FINGERS_SHUT, "align below windshield header", rot),
        _w(2.5, tcp, FINGERS_SHUT, "in front of visor face near inner end", rot),
    ]
    tcp, rot = face_push(0.0, R_NEAR, back=0.004)
    right += [_w(1.5, tcp, FINGERS_SHUT, "fingertips to visor face", rot)]
    for sw in np.linspace(0.0, SW_HAND, 11)[1:]:
        tcp, rot = face_push(float(sw), R_NEAR)
        right.append(_w(1.0 if sw < 0.15 else 0.8, tcp, FINGERS_SHUT, "unclip and swing beneath headliner", rot))
        PLANNED.append((_dur(right), float(sw), 0.0, "right"))
    tcp, rot = face_push(SW_HAND, R_NEAR, back=0.050)
    right += [_w(1.0, tcp, FINGERS_SHUT, "step off face", rot)]
    tcp, rot = face_push(SW_HAND, R_MID, back=0.050)
    right += [_w(1.5, tcp, FINGERS_SHUT, "shift nearer pivot beneath headliner", rot)]
    tcp, rot = face_push(SW_HAND, R_MID, back=0.004)
    right += [_w(1.0, tcp, FINGERS_SHUT, "re-engage face", rot)]
    for sw in np.linspace(SW_HAND, SWING_END, 8)[1:]:
        tcp, rot = face_push(float(sw), R_MID)
        right.append(_w(0.8, tcp, FINGERS_SHUT, "swing visor to side window", rot))
        PLANNED.append((_dur(right), float(sw), 0.0, "right"))
    tcp, rot = face_push(SWING_END, R_MID, back=0.090)
    right += [
        _w(1.5, tcp, FINGERS_SHUT, "retreat from visor face", rot),
        _w(2.0, (0.22, 0.00, TOP + 0.10), FINGERS_SHUT, "drop low beneath headliner", rot),
        _w(2.0, (0.15, -0.22, TOP + 0.12), FINGERS_SHUT, "clear cabin center", 0.0),
        _w(2.5, RIGHT_PARK, FINGER_OPEN, "park", 0.0),
    ]

    if MODE == "swing_no_slide":
        right.append(_w(2.0, RIGHT_PARK, FINGER_OPEN, "rest", 0.0))
        t_end = _dur(right)
        left = [
            _w(2.5, LEFT_PARK, FINGER_OPEN, "park", 0.0),
            _w(t_end - 2.5, LEFT_PARK, FINGER_OPEN, "rest", 0.0),
        ]
        return {"left": left, "right": right}

    t_swung = _dur(right) - 4.0
    left = [_w(2.5, LEFT_PARK, FINGER_OPEN, "park clear of visor mount", 0.0)]
    _, rot = end_push(SWING_END, 0.0, back=0.060)
    left.append(_w(max(0.3, t_swung - _dur(left)), LEFT_PARK, FINGER_OPEN, "wait for swing", 0.0))
    left += [Waypoint(3.0, _EndAt(0.0, back=0.060), FINGERS_SHUT, "approach visor outer end beneath roof rail", rot)]
    left += [Waypoint(1.5, _EndAt(0.0, back=0.004), FINGERS_SHUT, "fingertips to outer end", rot)]
    for sl in np.linspace(0.0, SLIDE_TO, 6)[1:]:
        left.append(Waypoint(0.8, _EndAt(float(sl)), FINGERS_SHUT, "slide extender along rod", rot))
        PLANNED.append((_dur(left), SWING_END, float(sl), "left"))
    left += [
        Waypoint(1.5, _EndAt(SLIDE_TO, back=0.090), FINGERS_SHUT, "retreat along rod", rot),
        _w(2.5, LEFT_PARK, FINGER_OPEN, "park", 0.0),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(_w(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest", 0.0))
    right.append(_w(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest", 0.0))
    return {"left": left, "right": right}
