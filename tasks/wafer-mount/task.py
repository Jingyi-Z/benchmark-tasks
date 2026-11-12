"""On the station table: four unlabelled 100 mm silicon wafers, pattern side up, each in its own open single-wafer
carrier; two pairs of angled wafer tweezers standing tips down in cups; and a manual tape-frame mounter with a steel
film frame already on its three locating pins and a WAFER VACUUM toggle switch.

The wafers look alike: their type and crystal orientation show only in their flats (SEMI convention, seen from the
pattern side): one primary flat on every wafer; p-type (111) has no second flat, n-type (111) a second flat 45 degrees
counterclockwise from the primary, p-type (100) one 90 degrees counterclockwise, n-type (100) one opposite it (180).

The task: mount the n-type (100) wafer for dicing: lay it pattern side DOWN on the mounter's chuck (the tape goes on
its back), centred, with its primary flat towards the frame's flat side, and switch the wafer vacuum on. Wafers are
handled with tweezers only, never touched on the pattern, never dropped, and no other wafer goes on the chuck.

The demo flips the wafer in the air with one pair of tweezers and hands it over to the other pair (tweezers can only
set down a wafer whose top they reach from above), which lays it on the chuck through the chuck's tweezer recess.

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

SI_BACK = (0.58, 0.59, 0.62)  # the wafers' lapped back and polished rim: plain grey
DIE = (0.20, 0.22, 0.42)  # the patterned area: dark blue-violet
STREET = (0.62, 0.60, 0.48)  # the scribe streets between the dies: pale gold
STEEL = (0.72, 0.73, 0.75)
STAND_C = (0.55, 0.56, 0.58)
PP = (0.86, 0.87, 0.85)  # natural polypropylene carriers
ALU = (0.70, 0.71, 0.73)  # the mounter's chuck plate
BODY = (0.88, 0.88, 0.86)  # the mounter's housing
GROOVE = (0.35, 0.36, 0.38)
BLACK = (0.06, 0.06, 0.07)
RED = (0.80, 0.14, 0.10)
GREEN = (0.10, 0.55, 0.20)

# -- the wafers (loose): 100 mm silicon, drawn 1 mm thick (real: 0.525 mm) for stable contact. Frame: origin at the
# mid-plane centre, +z out of the pattern side, the primary flat's middle on +x. SEMI M1 flats for 100 mm wafers:
# primary 32.5 mm long, secondary 18 mm. -------------------------------------------------------------------------------
WAFER_R, WAFER_T = 0.0508, 0.0010
PRI_L, SEC_L = 0.0325, 0.018
EDGE_EXCL = 0.005  # the pattern stops 5 mm inside the rim (edge exclusion): a 90 mm field
COLLIDE_R = WAFER_R - 0.0031


def _chord_d(length):
    return math.sqrt(WAFER_R ** 2 - (length / 2) ** 2)


def _outline(sec_deg):
    """The wafer's outline (counterclockwise, seen from the pattern side): a circle cut by the primary flat (at +x)
    and, unless ``sec_deg`` is None, a secondary flat ``sec_deg`` counterclockwise from it."""
    chords = [(0.0, _chord_d(PRI_L), PRI_L)]
    if sec_deg is not None:
        chords.append((math.radians(sec_deg), _chord_d(SEC_L), SEC_L))
    pts = []
    for i in range(144):
        a = 2 * math.pi * (i + 0.5) / 144
        p = np.array([WAFER_R * math.cos(a), WAFER_R * math.sin(a)])
        if all(p @ [math.cos(t), math.sin(t)] <= d + 1e-9 for t, d, _ in chords):
            pts.append(p)
    for t, d, L in chords:
        n, tg = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
        pts += [d * n + L / 2 * tg, d * n - L / 2 * tg]
    pts.sort(key=lambda p: math.atan2(p[1], p[0]))
    return np.array(pts)


def _prism(outline, z0, z1, colour, collide):
    n = len(outline)
    pts = [(*p, z1) for p in outline] + [(*p, z0) for p in outline] + [(0.0, 0.0, z1), (0.0, 0.0, z0)]
    ct, cb = 2 * n, 2 * n + 1
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(ct, i, j), (cb, n + j, n + i), (i, n + i, n + j), (i, n + j, j)]
    return labkit.Mesh(points=np.array(pts, float), tris=np.array(tris), color=colour, collide=collide)


def _box_mesh(c, size, colour):
    """A box drawn only (no collision)."""
    c, h = np.asarray(c, float), np.asarray(size, float) / 2
    pts = np.array([c + h * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                     (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return labkit.Mesh(points=pts, tris=tris, color=colour, collide="none")


def _pattern():
    """The patterned area on the pattern side: a dark disc inside the edge exclusion, ruled into 10 mm dies by pale
    scribe streets (drawn only)."""
    r = WAFER_R - EDGE_EXCL
    ring = np.array([(r * math.cos(2 * math.pi * i / 96), r * math.sin(2 * math.pi * i / 96)) for i in range(96)])
    out = [_prism(ring, WAFER_T / 2, WAFER_T / 2 + 0.00008, DIE, "none")]
    for k in range(-4, 5):
        u = 0.010 * k + 0.005
        half = math.sqrt(max(r * r - u * u, 0.0)) - 0.0005
        if half <= 0.002:
            continue
        out.append(_box_mesh((u, 0.0, WAFER_T / 2 + 0.00012), (0.0006, 2 * half, 0.00004), STREET))
        out.append(_box_mesh((0.0, u, WAFER_T / 2 + 0.00012), (2 * half, 0.0006, 0.00004), STREET))
    return out


RIM_N, RIM_IN, RIM_GAP = 48, WAFER_R - 0.0050, 0.0006


def _rim_boxes(sec_deg):
    """Collision boxes filling the band from 5 mm to 0.6 mm inside the rim, cut short at the flats."""
    chords = [(0.0, _chord_d(PRI_L))] + ([(math.radians(sec_deg), _chord_d(SEC_L))] if sec_deg is not None else [])
    w = math.pi * RIM_IN / RIM_N * 1.05  # half the tangential width
    out = []
    for i in range(RIM_N):
        a = 2 * math.pi * (i + 0.5) / RIM_N
        r_o = WAFER_R - RIM_GAP
        for t, d in chords:
            c = math.cos(a - t)
            if c > 1e-6:
                r_o = min(r_o, (d - 0.0003 - w * abs(math.sin(a - t))) / c)
        if r_o - RIM_IN < 0.001:
            continue
        r = (r_o + RIM_IN) / 2
        out.append(((r * math.cos(a), r * math.sin(a), 0.0), (r_o - RIM_IN, 2 * w, WAFER_T * 0.9), SI_BACK, a))
    return out


def _wafer(name, sec_deg, xy, yaw_deg, z):
    # It collides as 16 thin strips crossing its centre, their ends 3 mm inside its rim (inside the flats too), hidden
    # in the drawn wafer: a thin convex mesh collider sinks slowly through thin supports in PhysX, and a primitive disc
    # gives tweezer blades a single contact point, so a wafer held by one edge twists out of them; flat strip faces
    # give the blades a contact patch.
    # A ring of short boxes then fills the band out to 0.6 mm inside the rim (short of the flats), so tweezers pinching
    # the outer 4 mm grip flat faces.
    strips = [((0.0, 0.0, 0.0), (2 * COLLIDE_R, 0.010, WAFER_T * 0.9), SI_BACK, math.pi * k / 16) for k in range(16)]
    return part(name, pos=(*xy, z), yaw=math.radians(yaw_deg), mass=0.010, friction=1.1,
                boxes=strips + _rim_boxes(sec_deg),
                meshes=[_prism(_outline(sec_deg), -WAFER_T / 2, WAFER_T / 2, SI_BACK, "none"), *_pattern()])


# -- the carriers (fixed): open single-wafer carriers for 100 mm wafers, all turned so the opening in their ledge and
# wall faces the right-hand robot (towards 200 degrees). The wafer rests by its rim on a ledge ring inside a low wall;
# the wall is open there, and a slot through the ledge lets the tweezers' lower blade go under the wafer's edge. -------------------------------------------------------
N_SEG = 72
C_BASE_R, C_BASE_T = 0.060, 0.002
LEDGE_IN, LEDGE_OUT, LEDGE_H = 0.0430, 0.0520, 0.006  # the wafer 8 mm above the bench: room under it for the lower blade
C_WALL_R, C_WALL_T, C_WALL_H = 0.0535, 0.002, 0.010
GAP = math.radians(32.0)
PICK_H = math.radians(20.0)  # the right hand's heading into the carriers (the blades point along it)
GAP_DIR = PICK_H + math.pi
REST_Z = TOP + C_BASE_T + LEDGE_H + WAFER_T / 2  # a wafer's centre, resting in its carrier


def _ring(r_in, r_out, z0, h, colour, n=N_SEG, skip=None, rib=False):
    seg = 2 * math.pi * (r_in + r_out) / 2 / n * (0.45 if rib else 1.06)
    out = []
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        if skip and skip(a):
            continue
        r = (r_in + r_out) / 2
        out.append(((r * math.cos(a), r * math.sin(a), z0 + h / 2), (r_out - r_in, seg, h), colour, a))
    return out


def _in_gap(a):
    return abs(math.atan2(math.sin(a - GAP_DIR), math.cos(a - GAP_DIR))) <= GAP


SLOT_HW = 0.011  # half the width of the tweezer slot cut through the ledge, in the opening


def _in_slot(a):
    return abs(math.atan2(math.sin(a - GAP_DIR), math.cos(a - GAP_DIR))) <= SLOT_HW / LEDGE_IN + math.radians(2.5)


def _carrier(name, xy):
    return part(name, pos=(*xy, TOP), fixed=True, friction=0.8,
                boxes=[*_ring(LEDGE_IN, LEDGE_OUT, C_BASE_T, LEDGE_H, PP, skip=_in_slot),
                       *_ring(C_WALL_R - C_WALL_T / 2, C_WALL_R + C_WALL_T / 2, C_BASE_T, C_WALL_H, PP, skip=_in_gap)],
                cylinders=[((0.0, 0.0, C_BASE_T / 2), C_BASE_R, C_BASE_T, PP)])


# Which wafer is where (the scene's only record of the types is the flats): the n-type (100) wafer is the front-left
# one. Yaws: where each primary flat points, counterclockwise from +x.
SPOTS = {"front_left": (0.24, -0.10), "front_right": (0.24, -0.235), "back_left": (0.375, -0.10),
         "back_right": (0.375, -0.235)}
TARGET_YAW = 110.0  # degrees (where its primary flat points); the demo's flip leaves it towards the frame's flat
WAFERS = [
    _wafer("wafer_1", 180.0, SPOTS["front_left"], TARGET_YAW, REST_Z),  # n-type (100): the one to mount
    _wafer("wafer_2", None, SPOTS["front_right"], 75.0, REST_Z),  # p-type (111)
    _wafer("wafer_3", 90.0, SPOTS["back_left"], -100.0, REST_Z),  # p-type (100)
    _wafer("wafer_4", 45.0, SPOTS["back_right"], 135.0, REST_Z),  # n-type (111)
]
TARGET = "wafer_1"
CARRIERS = [_carrier(f"carrier_{i + 1}", xy) for i, xy in enumerate(SPOTS.values())]

# -- the tweezers (loose), two pairs: angled wafer tweezers standing upright in cups, tips down. Frame: origin between
# the blades' tips, the arms vertical (joined end up), the blades bent 45 degrees forward (+x) and down. The lower blade
# and arm (x < 0) and the joined end are the body; the upper blade and arm slide together along x (sprung open). ------
BLADE_L, BLADE_W, BLADE_T = 0.012, 0.014, 0.0012
OPEN_GAP = 0.005
ARM_L, ARM_W, ARM_T = 0.085, 0.008, 0.0025
_c = _s = math.sqrt(0.5)
SEP = (OPEN_GAP + BLADE_T) / _s
_ax = -_c * BLADE_L
_az = _s * BLADE_L
TOP_Z = _az + ARM_L


def _half(sx):
    return [((_ax + sx, 0.0, _az + ARM_L / 2), (ARM_T, ARM_W, ARM_L), STEEL),
            ((_ax + sx + _c * BLADE_L / 2, 0.0, _az - _s * BLADE_L / 2), (BLADE_L, BLADE_W, BLADE_T), STEEL,
             (0.0, math.radians(45.0)))]


CUP_FLOOR, CUP_H = 0.020, 0.035
CUP_IN = ((-0.0150, 0.0065), 0.0085)
_cw = 0.003


def _cup(name, xy):
    return part(name, pos=(*xy, TOP), fixed=True, friction=0.8,
                boxes=[((sum(CUP_IN[0]) / 2, 0.0, CUP_FLOOR / 2), (CUP_IN[0][1] - CUP_IN[0][0] + 2 * _cw,
                                                                    2 * CUP_IN[1] + 2 * _cw, CUP_FLOOR), STAND_C),
                       *[((x, 0.0, CUP_FLOOR + CUP_H / 2), (_cw, 2 * CUP_IN[1] + 2 * _cw, CUP_H), STAND_C)
                         for x in (CUP_IN[0][0] - _cw / 2, CUP_IN[0][1] + _cw / 2)],
                       *[((sum(CUP_IN[0]) / 2, e * (CUP_IN[1] + _cw / 2), CUP_FLOOR + CUP_H / 2),
                          (CUP_IN[0][1] - CUP_IN[0][0], _cw, CUP_H), STAND_C) for e in (-1, 1)]])


def _tweezers(name, xy):
    jaw = Joint("jaw", kind="slide", axis="X", anchor=(_ax + SEP / 2, 0.0, _az + ARM_L / 2),
                lower=-(SEP - BLADE_T / _s - 0.0002), upper=0.0, initial=0.0, target=0.0, stiffness=600.0,
                damping=5.0, mass=0.008, boxes=_half(SEP / 2))
    return part(name, pos=(*xy, TOP + CUP_FLOOR + 0.0012), mass=0.020, friction=1.1, boxes=_half(-SEP / 2),
                meshes=[_box_mesh((_ax, 0.0, TOP_Z - 0.004), (SEP - ARM_T, ARM_W, 0.008), STEEL)], joints=[jaw])


R_CUP_XY = (0.11, -0.215)
L_CUP_XY = (0.12, 0.25)
R_CUP, L_CUP = _cup("tweezer_cup_r", R_CUP_XY), _cup("tweezer_cup_l", L_CUP_XY)
R_TW, L_TW = _tweezers("tweezers_r", R_CUP_XY), _tweezers("tweezers_l", L_CUP_XY)

# -- the tape-frame mounter (fixed): a housing whose top is the chuck plate, flush, with concentric centring grooves
# and straight alignment grooves, and a tweezer recess cut into the front of the chuck. A steel film frame (an
# annulus with a straight flat on its front side) lies on the plate round the chuck, seated against three locating
# pins. The mounter is turned a quarter turn so the recess and the frame's flat face the left-hand robot (+y). Frame:
# origin on the table under the chuck's centre, -x (the recess side) towards +y in the world. -------------------------
M_XY = (0.30, 0.14)
M_YAW = -math.pi / 2
M_HALF, M_H = 0.12, 0.040  # the housing's half width; the chuck plate's top above the table
CHUCK_TOP = TOP + M_H
RECESS_W, RECESS_D = 0.020, 0.004  # the tweezer recess: width, depth
RECESS_X = (-0.078, -0.030)  # its x span (mounter frame)
_pt = 0.006  # the chuck plate's thickness
FRAME_IN, FRAME_OUT, FRAME_T = 0.075, 0.100, 0.0012
FRAME_FLAT = 0.093  # the frame's front flat, this far in front of the centre


def _frame_boxes():
    """The film frame: an annulus of boxes (drawn and colliding) with a straight flat on its front (-x) side."""
    out = []
    n = 72
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        r = (FRAME_IN + FRAME_OUT) / 2
        if r * math.cos(a) < -FRAME_FLAT + 0.006:
            continue
        out.append(((r * math.cos(a), r * math.sin(a), M_H + FRAME_T / 2),
                    (FRAME_OUT - FRAME_IN, 2 * math.pi * r / n * 1.06, FRAME_T), STEEL, a))
    w = 2 * math.sqrt(FRAME_OUT ** 2 - FRAME_FLAT ** 2)
    out.append(((-(FRAME_FLAT + FRAME_IN) / 2, 0.0, M_H + FRAME_T / 2), (FRAME_FLAT - FRAME_IN, w, FRAME_T), STEEL))
    return out


_rx0, _rx1 = RECESS_X
_marks = []
for _r in (0.0254, 0.0381, 0.0508, 0.0635):  # centring grooves for 2, 3, 4 and 5 inch wafers
    _ring_pts = np.array([(_r * math.cos(2 * math.pi * i / 96), _r * math.sin(2 * math.pi * i / 96))
                          for i in range(96)])
    _in = np.array([(p[0] * (_r - 0.0004) / _r, p[1] * (_r - 0.0004) / _r) for p in _ring_pts])
    pts = [(*p, M_H + 0.00005) for p in _ring_pts] + [(*p, M_H + 0.00005) for p in _in]
    tris = [(i, (i + 1) % 96, 96 + i) for i in range(96)] + [((i + 1) % 96, 96 + (i + 1) % 96, 96 + i)
                                                             for i in range(96)]
    _marks.append(labkit.Mesh(points=np.array(pts), tris=np.array(tris), color=GROOVE, collide="none"))
_marks += [_box_mesh((0.0, 0.0, M_H + 0.00005), (0.140, 0.0008, 0.0001), GROOVE),  # straight grooves
           _box_mesh((0.010, 0.0, M_H + 0.00005), (0.0008, 0.140, 0.0001), GROOVE),
           _box_mesh((-_chord_d(PRI_L), 0.0, M_H + 0.00005), (0.0008, 0.070, 0.0001), GROOVE)]  # the flat line

MOUNTER = part(
    "mounter", pos=(*M_XY, TOP), fixed=True, friction=0.6,
    boxes=[
        ((0.0, 0.0, (M_H - _pt) / 2), (2 * M_HALF, 2 * M_HALF, M_H - _pt), BODY),
        # the chuck plate, round the tweezer recess: behind it, and either side of it in front
        (((_rx1 + M_HALF) / 2, 0.0, M_H - _pt / 2), (M_HALF - _rx1, 2 * M_HALF, _pt), ALU),
        (((_rx1 - M_HALF) / 2, (M_HALF + RECESS_W / 2) / 2, M_H - _pt / 2),
         (M_HALF + _rx1, M_HALF - RECESS_W / 2, _pt), ALU),
        (((_rx1 - M_HALF) / 2, -(M_HALF + RECESS_W / 2) / 2, M_H - _pt / 2),
         (M_HALF + _rx1, M_HALF - RECESS_W / 2, _pt), ALU),
        (((_rx0 - M_HALF) / 2, 0.0, M_H - _pt / 2), (M_HALF + _rx0, RECESS_W, _pt), ALU),  # in front of the recess
        ((sum(RECESS_X) / 2, 0.0, M_H - _pt + (_pt - RECESS_D) / 2), (_rx1 - _rx0, RECESS_W, _pt - RECESS_D), ALU),
        *_frame_boxes(),
    ],
    cylinders=[((FRAME_OUT * math.cos(a) + 0.003 * math.cos(a), FRAME_OUT * math.sin(a) + 0.003 * math.sin(a),
                 M_H + 0.003), 0.003, 0.006, STEEL) for a in (0.0, math.radians(120.0), math.radians(-120.0))],
    meshes=_marks, yaw=M_YAW,
)
CHUCK_C = np.array([*M_XY, CHUCK_TOP])
FLAT_DIR = math.pi + M_YAW  # the frame's flat side, seen from above: towards +y

# -- the WAFER VACUUM toggle (fixed) on the mounter's housing, at its corner nearest the right-hand robot: its lever
# tilted back (+x) is OFF, forward (-x, towards the green mark) is ON. Frame: origin on the housing's top. -------------
VAC_XY = (0.205, 0.040)
VAC = Joint("vacuum", kind="hinge", axis="Y", anchor=(0.0, 0.0, 0.006), lower=-0.45, upper=0.45, initial=0.42,
            target=0.0, friction=0.012, damping=0.02, mass=0.005,
            boxes=[((0.0, 0.0, 0.006 + 0.012), (0.004, 0.004, 0.024), STEEL)],
            cylinders=[((0.0, 0.0, 0.006 + 0.025), 0.0035, 0.004, BLACK)])
VAC_ON = -0.20  # the toggle counts as ON when tilted forward past this (rad)
VAC_SWITCH = part("vacuum_switch", pos=(*VAC_XY, CHUCK_TOP), fixed=True, friction=0.6,
                  boxes=[((0.0, 0.0, 0.003), (0.022, 0.016, 0.006), BLACK),
                         ((-0.016, 0.0, 0.0003), (0.005, 0.012, 0.0006), GREEN)],  # the ON mark, in front
                  joints=[VAC])

OBJECTS = [*CARRIERS, *WAFERS, R_CUP, L_CUP, R_TW, L_TW, MOUNTER, VAC_SWITCH]
WAFER_NAMES = [w.name for w in WAFERS]

# -- scoring ----------------------------------------------------------------------------------------------------------
CENTRE_TOL = 0.002  # the wafer's centre within 2 mm of the chuck's centre (centred by eye on the grooves)
SEAT_TOL = 0.0012  # its mid-plane within 1.2 mm of where it lies on the chuck
FACE_TILT = 3.0  # its pattern side facing straight down, within 3 degrees
FLAT_TOL = math.radians(2.0)  # its primary flat within 2 degrees of facing the frame's flat side
TOP_TOUCH = 0.004  # a pad tip within 4 mm of a wafer's pattern side, inside the edge exclusion: the pattern touched
HANDLED = 0.002  # a pad tip within 2 mm outside a wafer's rim and 5 mm of its mid-plane: held by the gripper
HANDS_FAR = 0.08  # both hands' pad tips and both tweezers' tips this far from the chuck's centre at the end
OTHER_NEAR = 0.06  # any other wafer's centre within 6 cm of the chuck's centre: the wrong wafer is on the mounter

_state = {}
SETTLE_STEPS, SETTLE_TOL = 20, 0.001
_hist = []


@on_reset
def _reset_state(env):
    _state.update(touched=False, handled=False, dropped=False, far=True, settled=False)
    _hist.clear()


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


def flat_dir(w) -> float:
    """Where the wafer's primary flat points, seen from above (rad)."""
    v = w.rotation()[:, 0]
    return math.atan2(v[1], v[0])


def face_down_deg(w) -> float:
    """The angle between the wafer's pattern-side normal and straight down (degrees)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(w.rotation()[2, 2])))))


def on_chuck(w) -> bool:
    p = np.asarray(w.pos, float)
    return (np.hypot(*(p[:2] - CHUCK_C[:2])) <= CENTRE_TOL and abs(p[2] - (CHUCK_TOP + WAFER_T / 2)) <= SEAT_TOL
            and face_down_deg(w) <= FACE_TILT)


def flat_ok(w) -> bool:
    d = flat_dir(w) - FLAT_DIR
    return abs(math.atan2(math.sin(d), math.cos(d))) <= FLAT_TOL


TW_TOUCH = 0.0003  # a tweezer blade within 0.3 mm of a wafer's pattern side, inside the edge exclusion: pattern touched
_BLADE_PTS = [np.array([u * BLADE_L / 2, v * BLADE_W / 2, t * BLADE_T / 2]) for u in (-1, 0, 1) for v in (-1, 0, 1)
               for t in (-1, 1)]
_EX = np.array([_c, 0.0, -_s])  # the blades' long axis (bent 45 degrees forward and down), in a tweezers' frame
_EZ = np.array([_s, 0.0, _c])


def _blade_points(tw):
    """Points over both blades of a pair of tweezers (a 3 x 3 grid on each face), in world coordinates."""
    out = []
    for sx in (-SEP / 2, SEP / 2 + tw.joints.get("jaw", 0.0)):
        c = np.array([_ax + sx + _c * BLADE_L / 2, 0.0, _az - _s * BLADE_L / 2])
        for q in _BLADE_PTS:
            out.append(tw.to_world(c + q[0] * _EX + [0.0, q[1], 0.0] + q[2] * _EZ))
    return out


@on_step
def _watch(env):
    """Latch the violations: a wafer's pattern touched by the gripper, a wafer held by the gripper itself, a wafer
    dropped (on the table, or below its carrier's ledge or the chuck without being on either)."""
    S = labkit._SPECS
    tips = _tips(env)
    blades = [q for n in ("tweezers_r", "tweezers_l") for q in _blade_points(labkit._part_state(env, S[n]))]
    track = []
    for n in WAFER_NAMES:
        w = labkit._part_state(env, S[n])
        p = np.asarray(w.pos, float)
        R = w.rotation()
        if p[2] < REST_Z - 0.0015:  # below a carrier's ledge: fallen in, or on the table
            _state["dropped"] = True
        for b in blades:
            q = R.T @ (np.asarray(b, float) - p)
            if math.hypot(q[0], q[1]) < WAFER_R - EDGE_EXCL and -0.0005 < q[2] - WAFER_T / 2 < TW_TOUCH:
                _state["touched"] = True  # tweezers on the pattern, inside the edge exclusion
        for tip in tips:
            q = R.T @ (np.asarray(tip, float) - p)
            r = math.hypot(q[0], q[1])
            if r < WAFER_R - EDGE_EXCL and WAFER_T / 2 - 0.0005 < q[2] < WAFER_T / 2 + TOP_TOUCH:
                _state["touched"] = True
            if r < WAFER_R + HANDLED and abs(q[2]) < 0.005:
                _state["handled"] = True
        track.append(p)
    tw = [np.asarray(labkit._part_state(env, S[n]).pos, float) for n in ("tweezers_r", "tweezers_l")]
    _state["far"] = all(np.hypot(*(t[:2] - CHUCK_C[:2])) >= HANDS_FAR for t in [*tips, *tw])
    vac = labkit._part_state(env, S["vacuum_switch"]).joints["vacuum"]
    _hist.append(np.concatenate([np.concatenate(track), [vac * 0.01]]))
    del _hist[:-(SETTLE_STEPS + 1)]
    _state["settled"] = (len(_hist) > SETTLE_STEPS
                         and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def vacuum_on(objects) -> bool:
    return objects["vacuum_switch"].joints["vacuum"] <= VAC_ON


def score(objects):
    """0 if a wafer's pattern (inside its 5 mm edge exclusion) was ever touched by the gripper or the tweezers, a wafer was ever held by the gripper itself or dropped,
    another wafer is on the mounter, or the wafers have not all been still (under 1 mm of movement) for the last
    second. Otherwise, with the n-type (100) wafer on the chuck pattern side down (within 3 degrees), its centre
    within 2 mm of the chuck's centre: 1.0 if its primary flat faces the frame's flat side within 2 degrees, the
    wafer vacuum is ON and both hands and both tweezers are 8 cm from the chuck's centre; 0.75 if all that but a
    hand or tweezers still at the chuck; 0.5 if the flat is right but the vacuum is OFF; 0.25 if the flat is off. 0
    if it is not on the chuck that way."""
    if _state.get("touched") or _state.get("handled") or _state.get("dropped"):
        return 0.0
    if not _state.get("settled", False):
        return 0.0
    for n in WAFER_NAMES:
        if n != TARGET and np.hypot(*(np.asarray(objects[n].pos[:2]) - CHUCK_C[:2])) < OTHER_NEAR:
            return 0.0
    w = objects[TARGET]
    if not on_chuck(w):
        return 0.0
    if not flat_ok(w):
        return 0.25
    if not vacuum_on(objects):
        return 0.5
    return 1.0 if _state.get("far", True) else 0.75


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    p = np.asarray(objects[TARGET].pos, float)
    return float(np.hypot(*(p[:2] - CHUCK_C[:2])) + abs(p[2] - (CHUCK_TOP + WAFER_T / 2)))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right hand: takes its tweezers out of their cup, slides the open blades into the front-left carrier's opening under
# the n-type (100) wafer's edge, pinches it, lifts it high and rolls the tweezers half a turn about the blades, so the
# wafer hangs pattern side down from them, then turns it to face the left hand. Left hand: takes its tweezers, pinches
# the wafer's free edge from above, and the right hand lets go and backs out. The left hand carries the wafer over the
# chuck, turns it so its primary flat faces the frame's flat side and lays it on the chuck, the lower blade in the
# tweezer recess; lets go and backs out. Both stand their tweezers back in their cups; the right hand switches the
# vacuum on. The demo reads only what the cameras see: the tweezers' tips (to trim each hand's aim) and the wafer.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

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
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.15, -0.25, TOP + 0.16)
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
TW_ROT = Orient(axis_angle_matrix(Y, math.radians(-45.0)) @ _prev_grip(math.pi / 2))
PAD_IN = 0.006
TW_GRIP_F = np.array([_ax, 0.0, TOP_Z - 0.014])
TW_TILT = axis_angle_matrix(Y, math.radians(-45.0))
TW_GRIP = TW_TILT @ TW_GRIP_F
DOWN_X = Orient(_prev_grip(math.pi / 2))
TW_WIDE = -0.014
TW_HOLD = -0.0050
TW_PINCH = -0.0010  # squeezed hard: the pinch must hold the wafer on edge against its weight's twist
PINCH_IN = 0.003  # the tips 3 mm inside the edge, inside the 5 mm edge exclusion
BLADE_Z = 0.0022  # the tips' midpoint above the wafer's underside, as taken in a carrier
V0 = np.array([WAFER_R - PINCH_IN, 0.0, WAFER_T / 2 - (BLADE_Z - 0.0016)])  # lower blade ~1 mm under it  # the wafer's centre from the tips, held
# blades flat along +x (tweezer heading 0)


def Rz(a):
    return axis_angle_matrix(Z, a)


def _tcp(point, rot):
    a = rot[:, 2]
    return np.asarray(point, float) - a * (TIP_ABOVE_TCP - PAD_IN)


def _tw_pose_tcp(origin, Rt):
    """TCP for a hand holding tweezers whose blades lie along Rt's x (Rt = Rz(heading), or rolled), tips at ``origin``."""
    return _tcp(np.asarray(origin, float) + Rt @ TW_GRIP, Rt @ TW_ROT.rot), Orient(Rt @ TW_ROT.rot)


def _cup_tcp(origin, psi=0.0):
    rot = Rz(psi) @ DOWN_X.rot
    return _tcp(np.asarray(origin, float) + Rz(psi) @ TW_GRIP_F, rot), Orient(rot)


# what each hand's tweezers are seen doing: the tips' (origin's) live position; each hand's aim is trimmed by the
# offset between where it means them to be and where they are, integrated only while they are still and near it
_tw = {s: {"corr": np.zeros(3), "goal": None, "prev": None} for s in ("r", "l")}
_seen = {}


@on_reset
def _reset_demo(env):
    for d in _tw.values():
        d.update(corr=np.zeros(3), goal=None, prev=None)
    _seen.pop("w", None)
    _seen.pop("wp", None)
    _seen.pop("w_live", None)
    _seen.pop("w_prev", None)
    _seen["wtrim"] = np.zeros(3)


@on_step
def _track(env):
    S = labkit._SPECS
    for s, n in (("r", "tweezers_r"), ("l", "tweezers_l")):
        d = _tw[s]
        pos = np.asarray(labkit._part_state(env, S[n]).pos, float)
        prev, d["prev"] = d["prev"], pos
        if d["goal"] is None or prev is None:
            continue
        err = d["goal"] - pos
        if np.linalg.norm(pos - prev) < 0.0002 and np.linalg.norm(err) < 0.04:
            d["corr"] = np.clip(d["corr"] + np.clip(0.2 * err, -0.0003, 0.0003), -0.03, 0.03)
    w = labkit._part_state(env, S[TARGET])
    _seen["w_live"] = (np.asarray(w.pos, float), w.rotation())
    t = getattr(_demo, "_plan_clock", 0.0)
    if t < _seen.get("w_t", 1e9):
        _seen["w"] = _seen["w_live"]
    if t < _seen.get("wp_t", 1e9):
        _seen["wp"] = _seen["w_live"]
    # laying it down: trim the left hand's aim by where the held wafer is seen over the chuck (while it is still)
    p, win = _seen["w_live"][0], _seen.get("trim_win")
    prev, _seen["w_prev"] = _seen.get("w_prev"), p
    if win and win[0] <= t < win[1] and prev is not None and np.linalg.norm(p - prev) < 0.0002:
        tr = _seen.setdefault("wtrim", np.zeros(3))
        aim = PLACE_C[:2] - [0.0, SETTLE_Y]
        tr[:2] = np.clip(tr[:2] + np.clip(0.2 * (aim - p[:2]), -0.0003, 0.0003), -0.02, 0.02)


class _At:
    """A hand's TCP from a function giving (tweezer origin, Rt) or, with ``cup``, an upright pose; trimmed by that
    hand's live correction (held fixed when ``track`` is False)."""

    def __init__(self, side, fn, track=True, cup=False):
        self.side, self.fn, self.track, self.cup = side, fn, track, cup
        self._rot = None

    def pose(self):
        o, R = self.fn()
        tcp, rot = (_cup_tcp(o, R) if self.cup else _tw_pose_tcp(o, R))
        return o, tcp, rot

    def __array__(self, dtype=None, copy=None):
        o, tcp, _ = self.pose()
        d = _tw[self.side]
        d["goal"] = np.asarray(o, float) if self.track else None
        return np.asarray(tcp + d["corr"], dtype=dtype or float)


def _wp(dur, side, fn, finger, label, track=True, cup=False):
    at = _At(side, fn, track, cup)
    return Waypoint(dur, at, finger, label, at.pose()[2])


def _dur(wps):
    return sum(w.duration for w in wps)


def _const(o, R):
    o = np.asarray(o, float)
    return lambda: (o, R)


R_HOME = np.array([R_CUP_XY[0], R_CUP_XY[1], TOP + CUP_FLOOR + 0.0012])
L_HOME = np.array([L_CUP_XY[0], L_CUP_XY[1], TOP + CUP_FLOOR + 0.0012])
W0 = np.array([*SPOTS["front_left"], REST_Z])
HAND_C = np.array([0.26, 0.0, TOP + 0.20])  # the wafer's centre when the right hand starts to stand it on edge
U = Rz(PICK_H) @ X  # the right tweezers' blades (and the axis they are rolled about)
ROLL = math.pi / 2  # the right hand stands the wafer on edge: a quarter roll about the blades
ROLL_STEPS = 6
T_UP = axis_angle_matrix(U, ROLL) @ Rz(PICK_H)  # the right tweezers' attitude, the wafer on edge
O_R = HAND_C - Rz(PICK_H) @ V0  # the right tweezers' tips during the roll (fixed)
C_EDGE = O_R + T_UP @ V0  # the wafer's centre once on edge
# The left tweezers take the wafer's top edge from behind, blades pointing down, their arms on its back side; they
# then turn it over (a 166 degree turn about a fixed axis through their tips) so it lies flat, pattern side down,
# with their blades pointing -y and their arms above: an attitude the left arm can reach, and one that lays the wafer
# on the chuck through the recess on the chuck's +y side.
_xb = np.array([0.0, -1.0, 0.0])  # the left blades' direction in the right tweezers' frame: from the top edge down
_zb = np.array([0.0, 0.0, -1.0])  # their blade plane's normal: the wafer's back side
REL_L = np.column_stack([_xb, np.cross(_zb, _xb), _zb])
L_R0 = T_UP @ REL_L  # the left tweezers' attitude taking the top edge
L_R1 = Rz(math.pi) @ axis_angle_matrix(X, math.pi) @ REL_L  # once turned over: equals Rz(-90 deg)
O_L = C_EDGE - L_R0 @ V0  # the left tweezers' tips at the top edge (fixed while turning it over)


def _turn_axis():
    d = L_R1 @ L_R0.T
    q = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    ang = math.atan2(np.linalg.norm(q) / 2, (np.trace(d) - 1) / 2)
    return q / np.linalg.norm(q), ang


TURN_AX, TURN_ANG = _turn_axis()
TURN_STEPS = 8
L_H1 = -math.pi / 2  # the left tweezers' heading laying the wafer down
PLACE_C = CHUCK_C + [0.0, 0.0, WAFER_T / 2 + 0.0006]
SETTLE_Y = 0.0015  # the wafer settles this far towards the recess (+y) as the blades let go: aimed short of it


def _r_roll(k):
    R = axis_angle_matrix(U, ROLL * k / ROLL_STEPS) @ Rz(PICK_H)
    return O_R, R


def _l_turn(k):
    return O_L, axis_angle_matrix(TURN_AX, TURN_ANG * k / TURN_STEPS) @ L_R0


def demo():
    for d in _tw.values():
        d.update(corr=np.zeros(3), goal=None, prev=None)
    _seen.pop("w", None)
    _seen.pop("wp", None)
    _seen["wtrim"] = np.zeros(3)
    right, left = [], []
    # right: tweezers out of their cup
    right += [
        _wp(3.0, "r", _const(R_HOME + [0, 0, 0.05], 0.0), TW_WIDE, "above the right tweezers in their cup", False, True),
        _wp(1.5, "r", _const(R_HOME, 0.0), TW_WIDE, "down: pads either side of their arms", False, True),
        _wp(0.8, "r", _const(R_HOME, 0.0), TW_HOLD, "hold them, blades open", False, True),
        _wp(1.5, "r", _const(R_HOME + [0, 0, 0.06], 0.0), TW_HOLD, "lift them out", False, True),
    ]
    # right: into the front-left carrier's opening, under the wafer's edge
    def pick(back, up):  # aimed at the wafer as seen before the blades go round its edge, at its edge's height
        def f():
            c, Rw = _seen.get("wp", (W0, np.eye(3)))
            v = Rw.T @ (-U)
            v = np.array([v[0], v[1], 0.0]) / math.hypot(v[0], v[1])
            dz = float((Rw @ v)[2]) * WAFER_R  # the edge there above (or below) the centre, if the wafer tilts
            return c - Rz(PICK_H) @ V0 + Z * dz - U * back + Z * up, Rz(PICK_H)
        return f
    right += [
        _wp(3.5, "r", pick(0.03, 0.03), TW_HOLD, "tilt them, blades flat, in front of the carrier's opening"),
        _wp(2.0, "r", pick(0.03, 0.03), TW_HOLD, "line the tips up with the wafer's edge"),
        _wp(1.5, "r", pick(0.015, 0.0), TW_HOLD, "blades level with the wafer"),
        _wp(1.5, "r", pick(0.015, 0.0), TW_HOLD, "line up"),
    ]
    _seen["wp_t"] = _dur(right)  # the wafer, seen before the blades go round its edge
    right += [
        _wp(2.0, "r", pick(0.0, 0.0), TW_HOLD, "slide the open blades round the edge in the opening"),
        _wp(1.2, "r", pick(0.0, 0.0), TW_HOLD, "line up: the lower blade just under the wafer"),
        _wp(0.8, "r", pick(0.0, 0.0), TW_PINCH, "pinch the wafer by its edge", False),
        _wp(2.0, "r", pick(0.0, 0.03), TW_PINCH, "lift it out of the carrier", False),
        _wp(4.5, "r", _const(O_R, Rz(PICK_H)), TW_PINCH, "carry it up, clear of everything", False),
    ]
    for k in range(1, ROLL_STEPS + 1):
        right.append(_wp(0.8, "r", (lambda k=k: _r_roll(k)), TW_PINCH,
                         "roll the tweezers a quarter turn about the blades: the wafer stands on edge", False))
    right.append(_wp(1.0, "r", lambda: _r_roll(ROLL_STEPS), TW_PINCH, "hold it steady", False))
    t_offer = _dur(right)

    # left: tweezers out of their cup, then down onto the wafer's top edge from above, arms on its back side
    left += [_wp(2.0, "l", _const(L_HOME + [0, 0, 0.05], 0.0), TW_WIDE, "wait", False, True)]
    left.append(_wp(max(0.3, t_offer - 16.0 - _dur(left)), "l", _const(L_HOME + [0, 0, 0.05], 0.0), TW_WIDE,
                    "wait", False, True))
    left += [
        _wp(1.5, "l", _const(L_HOME, 0.0), TW_WIDE, "down: pads either side of the left tweezers' arms", False, True),
        _wp(0.8, "l", _const(L_HOME, 0.0), TW_HOLD, "hold them, blades open", False, True),
        _wp(1.5, "l", _const(L_HOME + [0, 0, 0.06], 0.0), TW_HOLD, "lift them out", False, True),
    ]
    _seen["w_t"] = t_offer + 2.6  # the wafer, seen on edge and steady
    seen = lambda: _seen.get("w", (C_EDGE, None))[0]  # noqa: E731
    at_top = lambda back: (lambda: (seen() - C_EDGE + O_L - L_R0 @ X * back, L_R0))  # noqa: E731
    stage = C_EDGE + [0.0, 0.12, 0.02]
    left += [
        _wp(3.0, "l", _const(stage, Rz(-math.pi / 2)), TW_HOLD, "tilt them, blades flat, beside the wafer"),
        _wp(max(0.5, t_offer - _dur(left)), "l", _const(stage, Rz(-math.pi / 2)), TW_HOLD,
            "wait, clear of the wafer, while it is stood on edge"),
        _wp(2.0, "l", at_top(0.04), TW_HOLD, "turn them, blades pointing down, above the wafer's top edge"),
        _wp(0.8, "l", at_top(0.04), TW_HOLD, "wait for it to stand still"),
        _wp(1.5, "l", at_top(0.015), TW_HOLD, "down to its top edge"),
        _wp(1.2, "l", at_top(0.015), TW_HOLD, "line up, seen against the wafer"),
        _wp(1.5, "l", at_top(0.0), TW_HOLD, "slide the open blades round the top edge"),
        _wp(1.0, "l", at_top(0.0), TW_HOLD, "line up"),
        _wp(0.8, "l", at_top(0.0), TW_PINCH, "pinch it", False),
    ]
    t_taken = _dur(left)
    right.append(_wp(max(0.3, t_taken + 0.2 - _dur(right)), "r", lambda: _r_roll(ROLL_STEPS), TW_PINCH,
                     "hold it while the left hand takes it", False))
    rb = lambda back, dy=0.0: (lambda: (O_R - (T_UP @ X) * back + [0.0, dy, 0.0], T_UP))  # noqa: E731
    right += [
        _wp(0.8, "r", rb(0.0), TW_HOLD, "let go: open the blades", False),
        _wp(1.2, "r", rb(0.025), TW_HOLD, "back the blades out along the edge", False),
        _wp(1.2, "r", rb(0.06), TW_HOLD, "away from the wafer", False),
    ]
    # right: keeps its tweezers; rolls them back, blades flat, and moves out of the way above the carriers
    for k in range(ROLL_STEPS - 1, -1, -1):
        right.append(_wp(0.6, "r", (lambda k=k: (O_R - U * 0.06, _r_roll(k)[1])), TW_HOLD,
                         "roll them back, blades flat", False))
    right.append(_wp(2.5, "r", _const(W0 - Rz(PICK_H) @ V0 - U * 0.06 + Z * 0.10, Rz(PICK_H)), TW_HOLD,
                     "out of the way above the carriers, keeping the tweezers", False))
    r_rest = right[-1]

    # left: hold it while the right hand lets go, then turn it over, carry it over the chuck and lay it down
    held = (lambda: (seen() - C_EDGE + O_L, L_R0))  # noqa: E731
    left.append(_wp(max(0.3, t_taken + 3.5 - _dur(left)), "l", held, TW_PINCH,
                    "hold it while the right hand lets go and backs out", False))
    lift = lambda k: (lambda: (seen() - C_EDGE + O_L, _l_turn(k)[1]))  # noqa: E731
    for k in range(1, TURN_STEPS + 1):
        left.append(_wp(0.8, "l", lift(k), TW_PINCH, "turn it over: it lies flat, pattern side down", False))
    over = PLACE_C + [0.0, 0.0, 0.06]
    lay = lambda up, back=0.0: (lambda: (PLACE_C + Z * up - L_R1 @ V0 - L_R1 @ X * back  # noqa: E731
                                         + _seen.get("wtrim", np.zeros(3)), L_R1))
    left += [_wp(3.0, "l", lay(0.06), TW_PINCH, "carry it over the chuck", False)]
    t0 = _dur(left)
    left += [
        _wp(2.0, "l", lay(0.02), TW_PINCH, "down over the chuck"),
        _wp(3.5, "l", lay(0.02), TW_PINCH, "line it up over the chuck, seen against the chuck"),
    ]
    _seen["trim_win"] = (t0 + 0.8, _dur(left))
    left += [
        _wp(1.5, "l", lay(0.0), TW_PINCH, "lay it on the chuck, the lower blade in the recess", False),
        _wp(0.8, "l", lay(0.0), TW_HOLD, "let go: open the blades", False),
        _wp(0.6, "l", lay(-0.0015), TW_HOLD, "drop the blades into the recess, clear of the wafer", False),
        _wp(1.2, "l", lay(-0.0015, 0.020), TW_HOLD, "back the blades out along the recess", False),
        _wp(1.0, "l", lay(0.03, 0.020), TW_HOLD, "up", False),
        _wp(3.0, "l", _const(L_HOME + [0, 0, 0.06], 0.0), TW_HOLD, "upright again, over the cup", True, True),
        _wp(2.0, "l", _const(L_HOME + [0, 0, 0.06], 0.0), TW_HOLD, "line them up over the cup", True, True),
        _wp(1.5, "l", _const(L_HOME + [0, 0, 0.002], 0.0), TW_HOLD, "lower them into the cup", False, True),
        _wp(0.6, "l", _const(L_HOME + [0, 0, 0.002], 0.0), TW_WIDE, "let go", False, True),
        _wp(1.2, "l", _const(L_HOME + [0, 0, 0.05], 0.0), TW_WIDE, "up", False, True),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_laid = _dur(left) - 2.0  # the left hand has stood its tweezers in their cup

    # left: switch the wafer vacuum on once its tweezers are back in their cup (closed fingers push the lever forward)
    tg = np.array([VAC_XY[0], VAC_XY[1], CHUCK_TOP + 0.006])
    knob_off = tg + [0.014 * math.sin(0.42), 0.0, 0.014 * math.cos(0.42)]  # 14 mm up the lever: pushed mid-rod
    knob_on = tg + [-0.014 * math.sin(0.42), 0.0, 0.014 * math.cos(0.42)]
    dn = Orient(_prev_grip(0.0))
    tcp = lambda p: tuple(np.asarray(p, float) + [0.0, 0.0, TIP_ABOVE_TCP])  # noqa: E731
    left = left[:-1]  # (not to the park first)
    left += [
        Waypoint(2.5, tcp(knob_off + [0.012, 0.0, 0.03]), FINGER_CLOSED, "above, behind the vacuum toggle", dn),
        Waypoint(1.2, tcp(knob_off + [0.012, 0.0, -0.002]), FINGER_CLOSED, "fingertips down behind its knob", dn),
        Waypoint(1.5, tcp(knob_on + [-0.006, 0.0, -0.004]), FINGER_CLOSED, "push it forward: WAFER VACUUM on", dn),
        Waypoint(1.0, tcp(knob_on + [0.0, 0.0, 0.05]), FINGER_CLOSED, "up", dn),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), r_rest.pos, TW_HOLD, "rest", r_rest.yaw))
    return {"left": left, "right": right}
