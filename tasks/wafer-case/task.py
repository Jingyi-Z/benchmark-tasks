"""On the station table: a small benchtop hotplate with a 2-inch silicon wafer just baked on it, a pair of angled wafer
tweezers standing tips down in a cup, and a round polypropylene single-wafer case in two pieces: its flat lid lying
open side up, and its clear domed bottom lying beside it, dome up.

The hotplate: a light grey housing with a 100 x 100 mm aluminium heating plate, edged in red, its top 4 cm above the
table, and a dark control strip along its front with a red temperature display and two knobs; the wafer lies flat in
the middle of the plate, so the tweezers cannot get under its edge until it is slid out over the plate's edge.

The case: the lid is a shallow tray; the wafer rests by its edge on a ledge ring inside a low wall, both open on the
side towards the robots for the tweezers' blades, and a wide ribbed flange outside the wall takes the domed bottom.
The bottom is a clear shell, a ribbed skirt under a shallow dome; a bayonet closes the case: the bottom is set on with
its arrow at the lid's OPEN mark and turned 30 degrees clockwise to the LOCK mark.

The task: with the tweezers, slide the wafer to the plate's edge until it overhangs, pinch it by that edge and lay it, pattern side up, in the case's lid;
set the domed bottom on and turn it to lock. The gripper never holds the wafer, never touches its top and never
touches the hotplate's top.

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
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

HOUSING_C = (0.92, 0.92, 0.90)
ALU = (0.78, 0.79, 0.81)
HOT_C = (0.85, 0.15, 0.10)
BLACK = (0.06, 0.06, 0.07)
DARK = (0.16, 0.17, 0.19)
SI = (0.32, 0.34, 0.42)
STEEL = (0.72, 0.73, 0.75)
STAND_C = (0.55, 0.56, 0.58)
PP = (0.86, 0.87, 0.85)  # the case's lid: natural polypropylene
PP_B = (0.80, 0.84, 0.86)  # the domed bottom: clear polypropylene
labkit.GLASS_COLORS.update({PP_B})
MARK = (0.85, 0.20, 0.12)

WAFER_R, WAFER_T = 0.0254, 0.0010  # 2-inch wafer, drawn 1 mm thick (real: 0.28 mm) for stable contact

# -- the hotplate (fixed): a benchtop hotplate. A light grey housing; on top, a 100 x 100 mm aluminium heating plate,
# edged with a red warning band; a dark control strip along the front with a red temperature display and two knobs at
# its ends. ------------------------------------------------------------------------------------------------------------
HP_XY = (0.29, -0.10)  # the heating plate's centre, to the robots' right
HP_HALF, HP_PLATE_T, HP_H = 0.050, 0.005, 0.040  # the plate's half width and thickness; its top above the table
HP_TOP = TOP + HP_H
_bh = HP_H - HP_PLATE_T - 0.002  # the housing's top, under the plate
_bx = (-HP_HALF - 0.032, HP_HALF + 0.006)  # the housing's x span (plate frame): the control strip in front
_by = HP_HALF + 0.004  # the housing's half width: the plate's sides overhang nothing but the red band
_pz = HP_H - HP_PLATE_T / 2
HOTPLATE = part(
    "hotplate", pos=(*HP_XY, TOP), fixed=True, friction=0.3,
    boxes=[
        ((sum(_bx) / 2, 0.0, _bh / 2), (_bx[1] - _bx[0], 2 * _by, _bh), HOUSING_C),
        ((0.0, 0.0, _bh + 0.001), (2 * HP_HALF + 0.006, 2 * HP_HALF + 0.006, 0.002), HOT_C),  # red band
        ((0.0, 0.0, _pz), (2 * HP_HALF, 2 * HP_HALF, HP_PLATE_T), ALU),  # the heating plate
        ((_bx[0] + 0.015, 0.0, _bh + 0.0005), (0.026, 2 * _by - 0.008, 0.001), DARK),  # control strip
        ((_bx[0] + 0.015, 0.012, _bh + 0.0012), (0.012, 0.020, 0.0008), HOT_C),  # temperature display
    ],
    cylinders=[((_bx[0] + 0.015, e, _bh + 0.004), 0.0055, 0.007, BLACK) for e in (-0.034, 0.040)],  # the knobs
)

# -- the wafer (loose), just baked, in the middle of the hotplate ------------------------------------------------------
WAFER = part("wafer", pos=(*HP_XY, HP_TOP), mass=0.004, friction=0.8,
             cylinders=[((0.0, 0.0, WAFER_T / 2), WAFER_R, WAFER_T, SI)])

# -- the tweezers (loose): angled wafer tweezers, standing upright in a cup, tips down. Frame: origin between the two
# blades' tips, with the arms vertical (joined end up) and the blades bent 45 degrees forward (+x) and down from the
# arms' lower ends. The arms are flat strips, their faces normal to x; the lower blade and arm (x < 0) and the joined
# end are the body, the upper blade and arm (x > 0) slide together along x (sprung open). Held in a hand turned 45
# degrees (arms rising back at 45 degrees), the blades lie flat, pointing forward.
BLADE_L, BLADE_W, BLADE_T = 0.012, 0.014, 0.0012
OPEN_GAP = 0.005  # between the blades' faces, open
ARM_L, ARM_W, ARM_T = 0.085, 0.008, 0.0025
_c = _s = math.sqrt(0.5)
SEP = (OPEN_GAP + BLADE_T) / _s  # the two halves' offset along x (normal to the arms), open
_ax = -_c * BLADE_L  # the arm stack's centre line, x
_az = _s * BLADE_L  # the arms' lower ends, z


def _half(sx):
    """One half's arm and blade, at x offset sx: the arm vertical, the blade bent forward-down 45 degrees."""
    return [((_ax + sx, 0.0, _az + ARM_L / 2), (ARM_T, ARM_W, ARM_L), STEEL),
            ((_ax + sx + _c * BLADE_L / 2, 0.0, _az - _s * BLADE_L / 2), (BLADE_L, BLADE_W, BLADE_T), STEEL,
             (0.0, math.radians(45.0)))]


TOP_Z = _az + ARM_L  # the arms' upper ends


def _box_mesh(c, size, colour):
    """A box drawn only (no collision): the tweezers' joined end, a bridge between the arms' tops, inside their
    outline; the sprung half closes through it, and it never stops the gripper's pads (on the arms' outer faces)."""
    c, h = np.asarray(c, float), np.asarray(size, float) / 2
    pts = np.array([c + h * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                     (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return labkit.Mesh(points=pts, tris=tris, color=colour, collide="none")
TWEEZER_JAW = Joint("jaw", kind="slide", axis="X", anchor=(_ax + SEP / 2, 0.0, _az + ARM_L / 2),
                    lower=-(SEP - BLADE_T / _s - 0.0002), upper=0.0, initial=0.0, target=0.0, stiffness=600.0,
                    damping=5.0, mass=0.008, boxes=_half(SEP / 2))
# The cup: a block on the bench; the tweezers stand in it on their blade tips, upright.
TW_XY = (0.17, -0.27)  # in front of the hotplate, to the right
CUP_FLOOR, CUP_H = 0.020, 0.035  # the cup's floor above the table; its walls above the floor
CUP_IN = ((-0.0150, 0.0065), 0.0085)  # inner x span, inner half width (y): a snug fit, so they stand upright
_cw = 0.003
TSTAND = part("tweezer_cup", pos=(*TW_XY, TOP), fixed=True, friction=0.8,
              boxes=[((sum(CUP_IN[0]) / 2, 0.0, CUP_FLOOR / 2), (CUP_IN[0][1] - CUP_IN[0][0] + 2 * _cw,
                                                                  2 * CUP_IN[1] + 2 * _cw, CUP_FLOOR), STAND_C),
                     *[((x, 0.0, CUP_FLOOR + CUP_H / 2), (_cw, 2 * CUP_IN[1] + 2 * _cw, CUP_H), STAND_C)
                       for x in (CUP_IN[0][0] - _cw / 2, CUP_IN[0][1] + _cw / 2)],
                     *[((sum(CUP_IN[0]) / 2, e * (CUP_IN[1] + _cw / 2), CUP_FLOOR + CUP_H / 2),
                        (CUP_IN[0][1] - CUP_IN[0][0], _cw, CUP_H), STAND_C) for e in (-1, 1)]])
TWEEZERS = part(
    "tweezers", pos=(*TW_XY, TOP + CUP_FLOOR + 0.0012), mass=0.020, friction=0.9,
    boxes=[*_half(-SEP / 2),
           ],
    meshes=[_box_mesh((_ax, 0.0, TOP_Z - 0.004), (SEP - ARM_T, ARM_W, 0.008), STEEL)],
    joints=[TWEEZER_JAW],
)

# -- the case (loose): a round polypropylene single-wafer case in two pieces. Frames: origin on the axis, at the
# underside. The lid is a shallow tray: the wafer rests by its edge on a ledge ring inside a low wall, both open over
# 64 degrees towards the robots (-x) for the tweezers' blades; a wide flange outside the wall, ribbed round its edge,
# takes the domed bottom. The bottom is a clear shell: a ribbed skirt that fits over the wall onto the flange, under a
# shallow dome. They close with a bayonet: the bottom set on at the OPEN mark and turned 30 degrees clockwise to LOCK.
CASE_XY = (0.15, 0.02)  # the lid, open side up, in front of the robots' middle
COVER_XY = (0.20, 0.17)  # the domed bottom, dome up, to the case's left
N_SEG = 64  # segments round each ring: smooth enough to read as round
BASE_R, BASE_T = 0.042, 0.002  # the lid's floor and flange
LEDGE_IN, LEDGE_OUT, LEDGE_H = 0.0230, 0.0265, 0.003  # the ledge ring the wafer's edge rests on
WALL_R, WALL_T, WALL_H = 0.0285, 0.002, 0.007  # the low wall round the pocket
GAP = math.radians(32.0)  # half the opening in the ledge and the wall, towards -x
GAP_DIR = math.pi
LOCK = math.radians(-30.0)  # the bayonet's turn, clockwise seen from above


def _off_gap(a):
    return abs(math.atan2(math.sin(a - GAP_DIR), math.cos(a - GAP_DIR))) > GAP


def _ring(r_in, r_out, z0, h, colour, n=N_SEG, skip=None, rib=False):
    """A ring of ``n`` boxes, radial thickness r_out - r_in, from z0 up by h (``skip(angle)`` drops some)."""
    seg = 2 * math.pi * (r_in + r_out) / 2 / n * (0.45 if rib else 1.06)
    out = []
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        if skip and skip(a):
            continue
        r = (r_in + r_out) / 2
        out.append(((r * math.cos(a), r * math.sin(a), z0 + h / 2), (r_out - r_in, seg, h), colour, a))
    return out


_marks = [((0.0375 * math.cos(a), 0.0375 * math.sin(a), BASE_T + 0.0002), (0.006, 0.0015, 0.0004), MARK, a)
          for a in (math.pi, math.pi + LOCK)]  # OPEN and LOCK, at the front of the flange
CASE_LID = part(
    "case_lid", pos=(*CASE_XY, TOP), mass=0.020, friction=1.0,
    boxes=[*_ring(LEDGE_IN, LEDGE_OUT, BASE_T, LEDGE_H, PP, skip=lambda a: not _off_gap(a)),
           *_ring(WALL_R - WALL_T / 2, WALL_R + WALL_T / 2, BASE_T, WALL_H, PP, skip=lambda a: not _off_gap(a)),
           *_ring(BASE_R, BASE_R + 0.0012, 0.0, BASE_T, PP, n=48, rib=True),  # grip ribs round the flange
           *_marks],
    cylinders=[((0.0, 0.0, BASE_T / 2), BASE_R, BASE_T, PP)])

SKIRT_R, SKIRT_T, SKIRT_H = 0.0355, 0.002, 0.012  # the bottom's skirt: centre radius, thickness, height
DOME_H = 0.009  # the dome's rise above the skirt


def _dome_mesh(r0, h, colour, n=N_SEG, rings=8):
    """A spherical cap of base radius r0 and rise h (a solid lens for collision: its convex hull), on the skirt."""
    rho = (r0 * r0 + h * h) / (2 * h)  # the sphere's radius
    pts = [(0.0, 0.0, SKIRT_H + h)]
    for k in range(1, rings + 1):
        r = r0 * k / rings
        z = SKIRT_H + h - (rho - math.sqrt(rho * rho - r * r))
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((r * math.cos(a), r * math.sin(a), z))
    base = len(pts)
    pts.append((0.0, 0.0, SKIRT_H))
    tris = [(0, 1 + (i + 1) % n, 1 + i) for i in range(n)]
    for k in range(rings - 1):
        o0, o1 = 1 + k * n, 1 + (k + 1) * n
        for i in range(n):
            a, b, c, d = o0 + i, o0 + (i + 1) % n, o1 + i, o1 + (i + 1) % n
            tris += [(a, b, d), (a, d, c)]
    o = 1 + (rings - 1) * n
    tris += [(base, o + i, o + (i + 1) % n) for i in range(n)]  # the flat underside
    return labkit.Mesh(points=np.array(pts), tris=np.array(tris), color=colour, collide="hull")


CASE_COVER = part(
    "case_bottom", pos=(*COVER_XY, TOP), mass=0.015, friction=0.25,
    boxes=[*_ring(SKIRT_R - SKIRT_T / 2, SKIRT_R + SKIRT_T / 2, 0.0, SKIRT_H, PP_B),
           *_ring(SKIRT_R + SKIRT_T / 2, SKIRT_R + SKIRT_T / 2 + 0.0012, 0.002, SKIRT_H - 0.004, PP_B, n=36, rib=True),
           ((-(SKIRT_R + SKIRT_T / 2 + 0.0016), 0.0, SKIRT_H / 2), (0.0008, 0.003, 0.008), MARK)],  # the arrow
    meshes=[_dome_mesh(SKIRT_R + SKIRT_T / 2, DOME_H, PP_B)])
COVER_TOP = SKIRT_H + DOME_H

OBJECTS = [HOTPLATE, WAFER, TWEEZERS, TSTAND, CASE_LID, CASE_COVER]


# -- scoring ----------------------------------------------------------------------------------------------------------
FLAT_DEG = 3.0
POCKET_TOL = 0.0025  # the wafer's centre within 2.5 mm of the case's axis: inside the wall
REST_Z = BASE_T + LEDGE_H  # the wafer's underside on the ledge, above the lid's underside
SEAT_TOL = 0.0025
LOCK_WIN = (math.radians(-45.0), math.radians(-22.0))  # turned into the lock: 22 to 45 degrees clockwise
OPEN_WIN = math.radians(10.0)  # set on within 10 degrees of the OPEN mark
TOP_TOUCH = 0.006  # a pad tip within 6 mm above the wafer's top, more than 3 mm inside its rim: the top was touched
HANDLED = 0.002  # a pad tip within 2 mm outside the wafer's rim and 8 mm of its height: held by the gripper
HOT_TOUCH = 0.004  # a pad tip within 4 mm of the hotplate's top plate, over it: the gripper touched the hot plate
HANDS_FAR = 0.08  # both pad tips this far from the case's axis at the end

_state = {"touched": False, "handled": False, "dropped": False, "hot": False, "armed": False}


@on_reset
def _reset_state(env):
    _state.update(touched=False, handled=False, dropped=False, hot=False, armed=False, far=True)


def _tips(env):
    """Both pads' tips of both hands (world, env frame): either side of the gripper's centre line by half the opening,
    along the fingers' closing direction."""
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


def _yaw(s) -> float:
    r = s.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def rel_turn(lid, cover) -> float:
    d = _yaw(cover) - _yaw(lid)
    return math.atan2(math.sin(d), math.cos(d))


def seated(lid, cover) -> bool:
    lc = lid.to_local(cover.pos)
    return (np.hypot(lc[0], lc[1]) <= SEAT_TOL and abs(lc[2] - BASE_T) <= 0.0015 and cover.tilt_deg() <= FLAT_DEG
            and lid.tilt_deg() <= FLAT_DEG)


def in_pocket(lid, w) -> bool:
    lc = lid.to_local(w.pos)
    return np.hypot(lc[0], lc[1]) <= POCKET_TOL and abs(lc[2] - REST_Z) <= 0.001 and w.tilt_deg() <= FLAT_DEG


@on_step
def _watch(env):
    """Latch the violations (the wafer's top touched, the wafer held by the gripper, the wafer dropped, the hotplate's
    top touched by the gripper), and whether the bottom was set on at the OPEN mark (it stays armed while seated)."""
    S = labkit._SPECS
    w = labkit._part_state(env, S["wafer"])
    lid = labkit._part_state(env, S["case_lid"])
    cover = labkit._part_state(env, S["case_bottom"])
    p = np.asarray(w.pos, float)
    top = p[2] + WAFER_T
    if w.tilt_deg() > 40.0 or p[2] < TOP + 0.003:
        _state["dropped"] = True
    tips = _tips(env)
    for tip in tips:
        r = np.hypot(*(tip[:2] - p[:2]))
        if r < WAFER_R - 0.003 and top - 0.001 < tip[2] < top + TOP_TOUCH:
            _state["touched"] = True
        if r < WAFER_R + HANDLED and abs(tip[2] - (p[2] + WAFER_T / 2)) < 0.008:
            _state["handled"] = True
        if (abs(tip[0] - HP_XY[0]) < HP_HALF and abs(tip[1] - HP_XY[1]) < HP_HALF
                and tip[2] < HP_TOP + HOT_TOUCH):
            _state["hot"] = True
    if seated(lid, cover):
        if abs(rel_turn(lid, cover)) <= OPEN_WIN:
            _state["armed"] = True
    else:
        _state["armed"] = False
    _state["far"] = all(np.hypot(*(t[:2] - np.asarray(lid.pos[:2]))) >= HANDS_FAR for t in tips)


def locked(lid, cover) -> bool:
    return seated(lid, cover) and _state["armed"] and LOCK_WIN[0] <= rel_turn(lid, cover) <= LOCK_WIN[1]


def score(objects):
    """0 if the wafer's top was ever touched, the wafer was ever held by the gripper itself, it was dropped (tilted past
    40 degrees, or on the table), or the gripper touched the hotplate's top. Otherwise, with the wafer lying flat
    in the case's lid on its ledge, its centre within 2.5 mm of the case's axis: 1.0 if the domed bottom is
    seated on the lid and locked (set on at the OPEN mark, within 10 degrees, and turned 22 to 45 degrees
    clockwise while seated) with both hands 8 cm from the case's axis; 0.5 if the bottom is seated but not
    locked, or a hand is still at the case; 0.25 if the wafer is in the lid but the bottom is not seated. 0 if
    the wafer is not in the lid."""
    if _state["touched"] or _state["handled"] or _state["dropped"] or _state["hot"]:
        return 0.0
    w, lid, cover = objects["wafer"], objects["case_lid"], objects["case_bottom"]
    if not in_pocket(lid, w):
        return 0.0
    if locked(lid, cover):
        return 1.0 if _state.get("far", True) else 0.5
    return 0.5 if seated(lid, cover) else 0.25


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    w, lid = objects["wafer"], objects["case_lid"]
    lc = lid.to_local(w.pos)
    return float(np.hypot(lc[0], lc[1]) + abs(lc[2] - REST_Z))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm, with the tweezers: lifts them out of their cup from above, gripping their two arms near the top, carries
# them over the plate, still upright, sets their tips down beside the wafer and slides the wafer with the blades' sides
# out over the plate's right edge until it overhangs; tilts them so the blades lie flat, goes round to the right, slides
# the open blades round the overhanging edge (closed loop on the tweezers' live tips and the wafer's seen position),
# pinches it, lifts it and lays it in the case's lid through the gap in its wall, lets go, backs the blades out, stands
# the tweezers back in their cup and moves out of the way. Left arm: picks the domed bottom up by its skirt, sets it on
# the case's lid with its arrow at the OPEN mark
# (closed loop on the lid's live position) and turns it 30 degrees clockwise to the LOCK mark; lets go.
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
Y = np.array([0.0, 1.0, 0.0])
TW_ROT = Orient(axis_angle_matrix(Y, math.radians(-45.0)) @ _prev_grip(math.pi / 2))  # approach forward-down 45 deg
PAD_IN = 0.006  # the pad tips reach this far past the point gripped


def _tcp(point, rot):
    """TCP that puts the pads round ``point`` (world), their tips PAD_IN beyond it along the approach."""
    a = rot.rot[:, 2]
    return tuple(np.asarray(point, float) - a * (TIP_ABOVE_TCP - PAD_IN))


TW_GRIP_F = np.array([_ax, 0.0, TOP_Z - 0.014])  # gripped 14 mm below the arms' tops, the palm just clear
# of them (tweezer frame, upright)
TW_TILT = axis_angle_matrix(Y, math.radians(-45.0))  # upright -> held: arms rising back at 45 degrees, blades flat
TW_GRIP = TW_TILT @ TW_GRIP_F  # the grip point, from the blade tips, when held
DOWN_X = Orient(_prev_grip(math.pi / 2))  # pointing down, fingers closing along x (across the arms' faces)
TW_WIDE = -0.014  # pads 28 mm apart round the arms (11.3 mm over their outer faces)
TW_HOLD = -0.0050  # just closed on the arms' outer faces: the blades stay open
TW_PINCH = -0.0016  # squeezes the arms: the blades close on the wafer


Z = np.array([0.0, 0.0, 1.0])


def tw_rot(psi=0.0):
    """The hand's rotation holding the tweezers blades flat, pointing along heading ``psi`` (0: +x)."""
    return Orient(axis_angle_matrix(Z, psi) @ TW_ROT.rot)


def _tw_tcp(origin, psi=0.0):
    """TCP holding the tweezers (turned to blades flat, pointing along ``psi``) with their origin (between the blade
    tips) at ``origin``."""
    return _tcp(np.asarray(origin, float) + axis_angle_matrix(Z, psi) @ TW_GRIP, tw_rot(psi))


def up_rot(psi=0.0):
    """The hand's rotation holding the tweezers upright, tips down, their blades pointing along heading ``psi``."""
    return Orient(axis_angle_matrix(Z, psi) @ DOWN_X.rot)


def _cup_tcp(origin, psi=0.0):
    """TCP holding the tweezers upright (pointing down), blades along ``psi``, with their origin at ``origin``."""
    return _tcp(np.asarray(origin, float) + axis_angle_matrix(Z, psi) @ TW_GRIP_F, up_rot(psi))


PINCH_IN = 0.0085  # the blade tips this far inside the wafer's edge (the upper blade draws back ~3 mm as it closes)
BLADE_Z = 0.0022  # the tips' midpoint this far above the wafer's underside: the held-open lower blade 0.3 mm under
# it (only the upper blade moves as they close), the upper 3 mm over it
OVERHANG = 0.012  # slide the wafer this far out over the plate's right edge, so the blades can take it from that side
SLIDE_Y = HP_XY[1] - HP_HALF + WAFER_R - OVERHANG  # the wafer's centre (y) once slid
PUSH_Z = HP_TOP + 0.0006  # pushing, the tweezers upright: their tips just clear of the plate
PUSH_X = HP_XY[0]  # the tips level with the wafer's centre: the blades' side edges meet its rim near the tips
PUSH_Y0 = HP_XY[1] + WAFER_R + BLADE_W / 2 + 0.004  # start: the blades' right side 4 mm from the wafer's rim
PUSH_Y1 = SLIDE_Y + WAFER_R + BLADE_W / 2 - 0.0005  # end: the wafer pushed to SLIDE_Y
PICK_PSI = math.pi / 2  # take it by the overhang from the right: blades pointing +y
CASE_PLACE = np.array([CASE_XY[0] - WAFER_R + PINCH_IN, CASE_XY[1], TOP + REST_Z + 0.0008 + BLADE_Z])  # from -x
TW_HOME = np.array([TW_XY[0], TW_XY[1], TOP + CUP_FLOOR + 0.0012])
DOWN_Y = Orient(_prev_grip(0.0))  # pointing down, fingers closing along y
COVER_WIDE, COVER_HOLD = -0.047, -0.0355  # pads round the 75 mm ribbed skirt; squeezing it
COVER_GRIP_Z = 0.008  # pad tips this far above the skirt's lower edge

# what the demo sees of the case's lid: its position, every step
_seen = {"lid": np.array([*CASE_XY, TOP])}


@on_step
def _see(env):
    _seen["lid"] = np.asarray(labkit._part_state(env, labkit._SPECS["case_lid"]).pos, float)
    if getattr(_demo, "_plan_clock", 0.0) < _seen.get("freeze_t", 1e9):
        _seen["held"] = _seen["lid"].copy()


@on_step
def _see_wafer(env):
    if getattr(_demo, "_plan_clock", 0.0) < _seen.get("wafer_t", 1e9):
        _seen["wafer"] = np.asarray(labkit._part_state(env, labkit._SPECS["wafer"]).pos, float)


@on_reset
def _reset_seen(env):
    _seen["lid"] = np.array([*CASE_XY, TOP])
    _seen["held"] = _seen["lid"].copy()
    _seen["wafer"] = np.array([HP_XY[0], SLIDE_Y, HP_TOP])


def _pick_at(dy=0.0, dz=0.0):
    """Where the tips go to take the slid wafer from the right, seen where it came to rest (frozen once the hand is
    under it): ``dy`` further out (-y) of the pinch point, ``dz`` above it."""
    def f():
        w = _seen["wafer"]
        return np.array([w[0], w[1] - WAFER_R + PINCH_IN - dy, HP_TOP + BLADE_Z + dz])
    return f


class _OnLid:
    """A TCP given relative to the case lid's position as seen up to ``_seen["freeze_t"]`` (set by the demo to when the
    hands start bearing on the case, so they do not chase it). ``track`` is kept for readability only."""

    def __init__(self, fn, track=True):
        self.fn = fn
        self.track = track

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.fn(_seen["held"]), dtype=dtype or float)


def _cover_tcp(axis_pt):
    """TCP holding the bottom by its skirt, its axis (underside) at ``axis_pt``."""
    return tuple(np.asarray(axis_pt, float) + [0.0, 0.0, COVER_GRIP_Z + TIP_ABOVE_TCP])


# what the demo sees of the tweezers once held: their origin (between the blade tips); the right hand's aim is trimmed
# by the offset between where it means the tips to be and where they are, integrated only while they are still
_tw = {"corr": np.zeros(3), "goal": None, "prev": None}


@on_reset
def _reset_tw(env):
    _tw.update(corr=np.zeros(3), goal=None, prev=None)


@on_step
def _track_tw(env):
    pos = np.asarray(labkit._part_state(env, labkit._SPECS["tweezers"]).pos, float)
    prev, _tw["prev"] = _tw["prev"], pos
    if _tw["goal"] is None or prev is None:
        return
    err = _tw["goal"] - pos
    if np.linalg.norm(pos - prev) < 0.0002 and np.linalg.norm(err) < 0.04:
        _tw["corr"] = np.clip(_tw["corr"] + np.clip(0.3 * err, -0.0005, 0.0005), -0.03, 0.03)


class _TwAt:
    """The right hand's TCP to put the held tweezers' tips at ``origin`` (blades flat, or upright when ``cup``),
    trimmed by the live correction; ``track`` False holds the correction (and, before they are held, it is zero)."""

    def __init__(self, origin, cup=False, track=True, psi=0.0):
        self.fn = origin if callable(origin) else None
        self.origin = None if callable(origin) else np.asarray(origin, float)
        self.cup = cup
        self.track = track
        self.psi = psi

    def __array__(self, dtype=None, copy=None):
        o = self.fn() if self.fn is not None else self.origin
        _tw["goal"] = o if self.track else None
        tcp = _cup_tcp(o, self.psi) if self.cup else _tw_tcp(o, self.psi)
        return np.asarray(np.asarray(tcp) + _tw["corr"], dtype=dtype or float)


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _tw.update(corr=np.zeros(3), goal=None, prev=None)
    _seen.pop("freeze_t", None)
    d, q0, q1 = DOWN_X, tw_rot(0.0), tw_rot(0.0)
    K = CASE_PLACE
    H = 0.0  # heading into the case's lid: +x, through the gap in its wall
    right = [
        Waypoint(3.0, _TwAt(TW_HOME + [0.0, 0.0, 0.05], cup=True, track=False), TW_WIDE,
                 "above the tweezers standing in their cup", d),
        Waypoint(1.5, _TwAt(TW_HOME, cup=True, track=False), TW_WIDE, "down: pads either side of their arms", d),
        Waypoint(0.8, _TwAt(TW_HOME, cup=True, track=False), TW_HOLD, "hold the tweezers, blades open", d),
        Waypoint(1.5, _TwAt(TW_HOME + [0.0, 0.0, 0.06], cup=True, track=False), TW_HOLD, "lift them out of the cup", d),
    ]
    a0, a1 = np.array([PUSH_X, PUSH_Y0, PUSH_Z]), np.array([PUSH_X, PUSH_Y1, PUSH_Z])
    qp = tw_rot(PICK_PSI)
    right += [
        Waypoint(3.0, _TwAt(a0 + [0.0, 0.0, 0.03], cup=True), TW_HOLD, "upright, over the plate left of the wafer", d),
        Waypoint(1.5, _TwAt(a0 + [0.0, 0.0, 0.004], cup=True), TW_HOLD, "down, tips just over the plate", d),
        Waypoint(1.5, _TwAt(a0 + [0.0, 0.0, 0.004], cup=True), TW_HOLD, "line the tips up beside the wafer's rim", d),
        Waypoint(0.8, _TwAt(a0, cup=True, track=False), TW_HOLD, "tips down to the plate", d),
        Waypoint(5.0, _TwAt(a1, cup=True, track=False), TW_HOLD,
                 "slide the wafer with the blades' sides to the plate's right edge, until it overhangs", d),
        Waypoint(0.8, _TwAt(a1 + [0.0, 0.006, 0.0], cup=True, track=False), TW_HOLD, "draw back off its rim", d),
        Waypoint(1.2, _TwAt(a1 + [0.0, 0.006, 0.04], cup=True, track=False), TW_HOLD, "up", d),
    ]
    _seen["wafer_t"] = _dur(right) + 0.5  # where the wafer came to rest, seen before the hand goes round it
    right += [
        Waypoint(3.5, _TwAt(_pick_at(0.03, 0.03), psi=PICK_PSI), TW_HOLD,
                 "round to the right of the plate, blades pointing at the overhang", qp),
        Waypoint(2.0, _TwAt(_pick_at(0.015, 0.0), psi=PICK_PSI), TW_HOLD, "blades level with the wafer", qp),
        Waypoint(1.5, _TwAt(_pick_at(0.015, 0.0), psi=PICK_PSI), TW_HOLD, "line up, seen against the wafer", qp),
        Waypoint(2.0, _TwAt(_pick_at(), psi=PICK_PSI), TW_HOLD, "slide the open blades round the overhanging edge",
                 qp),
        Waypoint(1.5, _TwAt(_pick_at(), psi=PICK_PSI), TW_HOLD, "line up: the lower blade just under the wafer", qp),
        Waypoint(0.8, _TwAt(_pick_at(), track=False, psi=PICK_PSI), TW_PINCH, "pinch the wafer by its edge", qp),
        Waypoint(1.5, _TwAt(_pick_at(0.01, 0.03), track=False, psi=PICK_PSI), TW_PINCH, "lift it off the plate", qp),
        Waypoint(4.0, _TwAt(K + [0.0, 0.0, 0.03], psi=H), TW_PINCH, "carry it over the case's lid", q1),
        Waypoint(2.0, _TwAt(K + [0.0, 0.0, 0.03], psi=H), TW_PINCH, "line up over the pocket", q1),
        Waypoint(1.5, _TwAt(K, track=False, psi=H), TW_PINCH, "lower it onto the ledge, the blades in the gap", q1),
        Waypoint(0.8, _TwAt(K, track=False, psi=H), TW_HOLD, "let go: open the blades", q1),
        Waypoint(1.2, _TwAt(K + [-0.016, 0.0, 0.0], track=False, psi=H), TW_HOLD, "back the blades out", q1),
        Waypoint(1.0, _TwAt(K + [-0.016, 0.0, 0.03], track=False, psi=H), TW_HOLD, "up", q1),
        Waypoint(3.0, _TwAt(TW_HOME + [0.0, 0.0, 0.06], cup=True), TW_HOLD, "upright again, over the cup", d),
        Waypoint(2.0, _TwAt(TW_HOME + [0.0, 0.0, 0.06], cup=True), TW_HOLD, "line them up over the cup", d),
        Waypoint(1.5, _TwAt(TW_HOME + [0.0, 0.0, 0.002], cup=True, track=False), TW_HOLD, "lower them into the cup", d),
        Waypoint(0.6, _TwAt(TW_HOME + [0.0, 0.0, 0.002], cup=True, track=False), TW_WIDE,
                 "let go: they stand in it", d),
        Waypoint(1.2, _TwAt(TW_HOME + [0.0, 0.0, 0.05], cup=True, track=False), TW_WIDE, "up", d),
    ]

    right += [Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park, out of the way")]
    t_placed = _dur(right) - 10.3  # the wafer is in the case's lid and the tweezers are up, clear of it

    # left: the domed bottom onto the case's lid, arrow at OPEN; then turn it clockwise to LOCK. The lid stays put
    # under the turn: its underside grips the bench better than the bottom's rim grips its flange.
    c0 = np.array([*COVER_XY, TOP])
    on_lid = lambda dz: _OnLid(lambda L: _cover_tcp(L + [0, 0, dz]))  # noqa: E731
    left = [Waypoint(3.0, LEFT_PARK, FINGER_OPEN, "wait")]
    left.append(Waypoint(max(0.3, t_placed - 8.0 - _dur(left)), LEFT_PARK, FINGER_OPEN, "wait"))
    left += [
        Waypoint(2.5, _cover_tcp(c0 + [0, 0, 0.04]), COVER_WIDE, "above the domed bottom", 0.0),
        Waypoint(1.5, _cover_tcp(c0), COVER_WIDE, "fingers either side of its skirt", 0.0),
        Waypoint(0.8, _cover_tcp(c0), COVER_HOLD, "grip the skirt", 0.0),
        Waypoint(1.2, _cover_tcp(c0 + [0, 0, 0.05]), COVER_HOLD, "lift it", 0.0),
    ]
    left.append(Waypoint(max(0.3, t_placed - _dur(left)), _cover_tcp(c0 + [0, 0, 0.05]), COVER_HOLD,
                         "wait for the wafer to be in the case", 0.0))
    left += [
        Waypoint(3.0, on_lid(0.04), COVER_HOLD, "over the case's lid, its arrow at the OPEN mark", 0.0),
        Waypoint(1.5, on_lid(0.04), COVER_HOLD, "line up over the lid", 0.0),
    ]
    _seen["freeze_t"] = _dur(left)  # from here the hand bears on the case: aim at where it was seen last
    left += [
        Waypoint(2.0, on_lid(BASE_T - 0.001), COVER_HOLD, "lower it over the wall onto the flange", 0.0),
        Waypoint(0.5, on_lid(BASE_T - 0.001), COVER_HOLD, "seated", 0.0),
    ]
    for f in np.linspace(0.25, 1.0, 4):
        left.append(Waypoint(0.8, on_lid(BASE_T - 0.001), COVER_HOLD, "turn it clockwise to the LOCK mark",
                             float(LOCK * f)))
    left += [
        Waypoint(0.5, on_lid(BASE_T - 0.001), COVER_HOLD, "steady", float(LOCK)),
        Waypoint(0.6, on_lid(BASE_T - 0.001), COVER_WIDE, "let go", float(LOCK)),
        Waypoint(1.2, on_lid(0.05), COVER_WIDE, "up", float(LOCK)),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "rest"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
