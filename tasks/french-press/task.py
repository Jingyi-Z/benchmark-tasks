"""french-press: fit the plunger lid onto a filled Bodum Chambord French press, plunge smoothly while bracing the carafe, and rotate the safety lid to close the pour spout."""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arena_assets_fix  # noqa: F401
import labkit
from labkit import GLASS_COLORS, Joint, on_reset, on_step, part

from yam_arena import demo as _demo
from yam_arena import ik as _ik
from yam_arena.constants import FINGER_OPEN, READY as _READY
from yam_arena.demo import ArmPlan, Waypoint
from yam_arena.task_file import TABLE_TOP
from yam_arena.urdf import axis_angle_matrix

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 1200

# Colours (sRGB)
CORK_TAN = (0.68, 0.50, 0.32)
STEEL_BRIGHT = (0.78, 0.80, 0.84)
STEEL_MID = (0.58, 0.61, 0.65)
STEEL_DARK = (0.34, 0.37, 0.41)
GLASS_AMBER = (0.46, 0.28, 0.14)
GROUNDS_DARK = (0.20, 0.13, 0.08)
HANDLE_BLACK = (0.16, 0.17, 0.20)
KNOB_BRASS = (0.86, 0.70, 0.22)
SPOUT_MARK_DARK = (0.20, 0.22, 0.26)

GLASS_COLORS.add(GLASS_AMBER)

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def Rx(a: float) -> np.ndarray:
    return axis_angle_matrix(X, a)


def Ry(a: float) -> np.ndarray:
    return axis_angle_matrix(Y, a)


def Rz(a: float) -> np.ndarray:
    return axis_angle_matrix(Z, a)


# -- 1. Fixed cylindrical cork coaster & filled 0.5 L Bodum Chambord 4-cup carafe (1924-16 + 1504-10 beaker, 795 g) ----
# Official Bodum specs: 1504-10 spare glass beaker = 96.0 mm OD (9.6 cm) x 125.0 mm height (12.5 cm);
# 1924-16 Chambord 4-cup overall width with handle = 166.0 mm (16.6 cm), pressed overall height = 190.0 mm (19.0 cm).
COASTER_XY = np.array([0.235, +0.025])
COASTER_H = 0.006
COASTER_CYLS = [
    ((0.0, 0.0, COASTER_H / 2), 0.064, COASTER_H, CORK_TAN, "Z"),
]
CORK_COASTER = part(
    "cork_coaster",
    pos=(COASTER_XY[0], COASTER_XY[1], TOP),
    fixed=True,
    friction=0.28,
    cylinders=COASTER_CYLS,
)

FRAME_BASE_H = 0.008      # 8.0 mm stainless footed frame ring supporting 125.0 mm glass beaker -> 133.0 mm carafe rim
GLASS_H = 0.125           # 125.0 mm official Bodum 1504-10 replacement beaker height
CARAFE_H = FRAME_BASE_H + GLASS_H  # 0.133 m (133.0 mm)
BEAKER_R_OUT = 0.0480     # 96.0 mm official Bodum 1504-10 beaker outer diameter
BEAKER_WALL = 0.0030      # 3.0 mm engineering-estimated borosilicate wall -> 90.0 mm cylindrical inner bore (r_in = 0.0450 m)
BEAKER_R_IN = BEAKER_R_OUT - BEAKER_WALL
BEAKER_R_MID = BEAKER_R_IN + 0.5 * BEAKER_WALL  # 0.0465 m
N_STAVES = 16
STAVE_W = 0.0192          # seamless 16-gon chord at r_out = 48.0 mm
HANDLE_GRIP_LOCAL = np.array([0.0, +0.106, 0.075])  # outer handle face at y = +0.118 m, opposite beaker at -0.048 m -> 166 mm width

CARAFE_CYLS = [
    # Stainless footed base ring & beaker floor support (98 mm OD x 8 mm, z 0.000..0.008)
    ((0.0, 0.0, 0.004), 0.049, FRAME_BASE_H, STEEL_MID, "Z"),
    # Hydrated 25 g coarse coffee grounds bed at bottom of glass beaker (88 mm dia x 18 mm, z 0.008..0.026)
    ((0.0, 0.0, 0.017), 0.044, 0.018, GROUNDS_DARK, "Z"),
]

CARAFE_BOXES = []
for _k in range(N_STAVES):
    _th = _k * (2.0 * math.pi / N_STAVES)
    _cx = BEAKER_R_MID * math.cos(_th)
    _cy = BEAKER_R_MID * math.sin(_th)
    if _k == 12:
        # Front -Y stave (-90 deg) shortened by 18 mm at top for the open pour-spout notch (z 0.008..0.115)
        CARAFE_BOXES.append(((_cx, _cy, 0.0615), (BEAKER_WALL, STAVE_W, 0.107), GLASS_AMBER, _th))
    else:
        # 125.0 mm tall borosilicate glass wall stave (z 0.008..0.133)
        CARAFE_BOXES.append(((_cx, _cy, 0.0705), (BEAKER_WALL, STAVE_W, GLASS_H), GLASS_AMBER, _th))
        # Chrome lower & upper retaining bands around 96 mm OD beaker exterior
        CARAFE_BOXES.append(((0.0492 * math.cos(_th), 0.0492 * math.sin(_th), 0.034), (0.0024, 0.0198, 0.011), STEEL_BRIGHT, _th))
        CARAFE_BOXES.append(((0.0492 * math.cos(_th), 0.0492 * math.sin(_th), 0.116), (0.0024, 0.0198, 0.011), STEEL_BRIGHT, _th))

CARAFE_BOXES += [
    # Projecting -Y stainless pour-spout lip at z = 0.115 m
    ((0.0, -0.052, 0.115), (0.022, 0.012, 0.005), STEEL_BRIGHT),
    # Four vertical chrome frame legs at 45, 135, 225, 315 deg
    *[
        ((0.0498 * math.cos(_a), 0.0498 * math.sin(_a), 0.065), (0.003, 0.009, 0.122), STEEL_BRIGHT, _a)
        for _a in (0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi)
    ],
    # Matte black PP side handle on +Y (outer face at +0.118 m; -0.048..+0.118 m = 166.0 mm official Chambord 1924-16 width)
    ((0.0, +0.083, 0.116), (0.026, 0.070, 0.011), HANDLE_BLACK),
    ((0.0, +0.083, 0.034), (0.026, 0.070, 0.011), HANDLE_BLACK),
    ((0.0, +0.106, 0.075), (0.026, 0.024, 0.069), HANDLE_BLACK),
]

CARAFE_START = np.array([COASTER_XY[0], COASTER_XY[1], TOP + COASTER_H + 0.0005])
CARAFE = part(
    "carafe",
    pos=(float(CARAFE_START[0]), float(CARAFE_START[1]), float(CARAFE_START[2])),
    mass=0.795,   # 345 g Chambord 4-cup carafe + 425 mL brew water + 25 g wet coffee grounds
    friction=0.28,
    boxes=CARAFE_BOXES,
    cylinders=CARAFE_CYLS,
)


# -- 2. Fixed stainless drip stand on the right (-Y) -------------------------------------------------------------------
STAND_XY = np.array([0.245, -0.165])
STAND_CRADLE_Z = 0.068    # supports 98 mm round lid flange at z_rel = 68 mm (mesh disc hangs 36 mm above base)

STAND_CYLS = [
    ((0.0, 0.0, 0.003), 0.066, 0.006, STEEL_MID, "Z"),
]
STAND_BOXES = [
    # Flush cradle ledges (89.6 mm inner gap clears 82.0 mm cylindrical pilot skirt while supporting 98 mm lid flange)
    ((+0.0478, 0.0, 0.037), (0.006, 0.1016, 0.062), STEEL_DARK),
    ((-0.0478, 0.0, 0.037), (0.006, 0.1016, 0.062), STEEL_DARK),
    ((0.0, -0.0478, 0.037), (0.0896, 0.006, 0.062), STEEL_DARK),
    ((0.0, +0.0478, 0.037), (0.0896, 0.006, 0.062), STEEL_DARK),
]
DRIP_STAND = part(
    "drip_stand",
    pos=(STAND_XY[0], STAND_XY[1], TOP),
    fixed=True,
    friction=0.55,
    boxes=STAND_BOXES,
    cylinders=STAND_CYLS,
)


# -- 3. Authentic one-piece Bodum Chambord stepped safety lid + plunger assembly (228 g total: 148 g lid + 80 g plunger) --
# Per Bodum Chambord 1928-57 / 1924-16 instructions, rotating the safety lid itself opens or closes the pour spout:
# its cylindrical lower pilot skirt has a -Y pour opening (staves 11..13) and solid stainless wall elsewhere; rotating
# the lid ~90 deg CW turns the solid pilot-skirt wall across the carafe's -Y pour-spout notch.
# Local origin (0, 0, 0) = underside of 98 mm OD round stainless rim lip (rests on carafe rim z = CARAFE_H = 0.133 m;
# pressed knob top at z_local = 0.117 - 0.060 = 0.057 m -> 133 + 57 = 190.0 mm exact official Bodum 1924-16 height).
PLUNGE_STROKE = 0.060     # 60.0 mm full stroke (q_plunge: 0.000 -> -0.060 m)
COLLAR_GRIP_LOCAL = np.array([0.0, 0.0, 0.0305]) # authentic 22 mm OD central guide collar on central axis (z 0.0125..0.0465 -> +-4.84 mm centered clearance!)
DOME_SPIN_LOCAL_Z = 0.0085                       # upper half of 68 mm OD knurled dome crown (z 0.0035..0.0125 -> +5.0 mm above 3.5 mm rim lip)
KNOB_CUP_LOCAL_Z = 0.112

LID_CYLS = [
    # Inner stainless dome core (65 mm OD x 12.5 mm, z 0.000..0.0125)
    ((0.0, 0.0, 0.00625), 0.0325, 0.0125, STEEL_BRIGHT, "Z"),
    # Authentic cylindrical central guide collar (22 mm OD x 34 mm, z 0.0125..0.0465) + 26 mm OD knurled top nut (z 0.0465..0.0505)
    ((0.0, 0.0, 0.0295), 0.011, 0.034, STEEL_DARK, "Z"),
    ((0.0, 0.0, 0.0485), 0.013, 0.004, STEEL_BRIGHT, "Z"),
]

LID_BOXES = []
for _k in range(N_STAVES):
    _th = _k * (2.0 * math.pi / N_STAVES)
    # 360-deg 98 mm OD round outer stainless rim lip (r = 0.034..0.049 m, z 0.000..0.0035);
    # stave 12 (-Y) carries a flush dark pour-slot alignment mark directly above the -Y pour opening
    _rim_col = SPOUT_MARK_DARK if _k == 12 else STEEL_BRIGHT
    LID_BOXES.append(((0.0415 * math.cos(_th), 0.0415 * math.sin(_th), 0.00175), (0.015, 0.0195, 0.0035), _rim_col, _th))
    # Stepped 68 mm OD 16-stave knurled chrome dome crown (r_out = 0.0340 m, z 0.0035..0.0125)
    _dome_col = SPOUT_MARK_DARK if _k == 12 else STEEL_BRIGHT
    LID_BOXES.append(((0.0325 * math.cos(_th), 0.0325 * math.sin(_th), 0.0080), (0.003, 0.0136, 0.0090), _dome_col, _th))
    # 82.0 mm OD cylindrical lower pilot skirt (r_out = 0.0410 m -> 4.0 mm radial clearance inside 90.0 mm bore,
    # z -0.016..0.000 covering the 18 mm deep -Y spout notch when rotated closed), slotted on -Y (_k in 11, 12, 13)
    if _k not in (11, 12, 13):
        LID_BOXES.append(((0.0390 * math.cos(_th), 0.0390 * math.sin(_th), -0.008), (0.004, 0.0162, 0.016), SPOUT_MARK_DARK, _th))

PLUNGER_CYLS = [
    # Bottom stainless spiral/cross-plate mesh filter assembly (78 mm OD x 8 mm at z = -0.028 when q = 0 -> -0.088 at q = -0.060)
    ((0.0, 0.0, -0.028), 0.039, 0.008, STEEL_MID, "Z"),
    # Central 8 mm stainless plunger stem (z -0.024..+0.109)
    ((0.0, 0.0, 0.0425), 0.004, 0.133, STEEL_BRIGHT, "Z"),
    # Dished 36 mm OD brass press knob (z 0.107..0.113 -> pressed z = 0.047..0.053, 30 mm above 72 mm dome crown!)
    ((0.0, 0.0, 0.110), 0.018, 0.006, KNOB_BRASS, "Z"),
]
PLUNGER_BOXES = [
    # Cylindrical 8-segment dished thumb rim on top of the 36 mm knob (z 0.113..0.117 -> pressed height = 133 + 57 = 190 mm!)
    *[
        ((0.0155 * math.cos(_m * 0.25 * math.pi), 0.0155 * math.sin(_m * 0.25 * math.pi), 0.115),
         (0.0035, 0.0125, 0.004), KNOB_BRASS, _m * 0.25 * math.pi)
        for _m in range(8)
    ],
]

# Plunger joint: dimensionless legacy PhysX jointFriction = 2.4 + estimated 85 N*s/m Darcy filtration damping
PLUNGER_JOINT = Joint(
    "plunger", kind="slide", axis="Z", anchor=(0.0, 0.0, 0.0),
    boxes=PLUNGER_BOXES, cylinders=PLUNGER_CYLS,
    lower=-0.060, upper=0.002, initial=0.0, target=0.0,
    stiffness=0.0, damping=85.0, friction=2.4, mass=0.080,
)

LID_STAND_START = np.array([STAND_XY[0], STAND_XY[1], TOP + STAND_CRADLE_Z + 0.0005])
PLUNGER_LID = part(
    "plunger_lid",
    pos=(float(LID_STAND_START[0]), float(LID_STAND_START[1]), float(LID_STAND_START[2])),
    mass=0.148,
    friction=0.85,
    boxes=LID_BOXES,
    cylinders=LID_CYLS,
    joints=[PLUNGER_JOINT],
    self_collide=False,
)

OBJECTS = [CORK_COASTER, CARAFE, DRIP_STAND, PLUNGER_LID]


# -- Scoring & continuous safety/validity monitor ----------------------------------------------------------------------
# Exact hydrostatic fill/spout overflow geometry: 425 mL brew water + 25 mL wet grounds = 450 cm^3 inside 90.0 mm ID
# cylindrical beaker (r_in = 0.0450 m, floor z_local = 0.0080 m) -> undisturbed center fill z_local = 0.07874 m
# (overflows -Y z=0.115 m pour-spout notch above 38.9 deg toward -Y, or z=0.133 m rim above 50.3 deg elsewhere).
Z_FILL_LOCAL = FRAME_BASE_H + 450e-6 / (math.pi * BEAKER_R_IN * BEAKER_R_IN)
_FILL_ORIGIN_LOCAL = np.array([0.0, 0.0, Z_FILL_LOCAL], float)
_CARAFE_LIP_PTS = tuple(
    [
        np.array([0.0, -BEAKER_R_IN, 0.1150], float),
        np.array([+0.0096, -math.sqrt(BEAKER_R_IN * BEAKER_R_IN - 0.0096 * 0.0096), 0.1150], float),
        np.array([-0.0096, -math.sqrt(BEAKER_R_IN * BEAKER_R_IN - 0.0096 * 0.0096), 0.1150], float),
    ]
    + [
        np.array([BEAKER_R_IN * math.cos(_k * 2.0 * math.pi / N_STAVES), BEAKER_R_IN * math.sin(_k * 2.0 * math.pi / N_STAVES), CARAFE_H], float)
        for _k in range(N_STAVES)
        if _k != 12
    ]
)
CARAFE_END_TILT = 3.0        # <=3 deg upright at end
LID_RAD_TOL = 0.006          # <=6 mm coaxial with carafe rim
LID_Z_TOL = 0.006            # within 6 mm of carafe rim height
LID_TILT_TOL = 5.0           # <=5 deg tilt
# Modeled spout/slot ray occlusion: carafe -Y spout gap spans +-10.46 deg between staves 11 & 13 (r_in=45.0 mm) and lid
# -Y pour slot spans +-32.65 deg between solid pilot-skirt staves 10 & 14 (r_in=37.0 mm, solid across remaining 294.7 deg
# continuously through 180 deg). Exact 2D box ray-coverage transition is 32.65 + 10.46 = 43.11 deg -> >=43.15 deg blocks all rays.
SPOUT_CLOSED_MIN_DEG = 43.15
PLUNGE_GOAL = 0.052          # >=52 mm final depth AND >=52 mm net downward travel accumulated while seated on carafe
MAX_PLUNGE_SPEED = 0.110     # >110 mm/s filter descent inside 90 mm carafe bore (insertion or seated press) latches 0
HANDS_CLEAR = 0.080          # >=80 mm exact collider-surface clearance from both pad midpoints + all 4 jaw endpoints
SETTLE_STEPS = 20            # 21 samples = 1.0 s at 20 Hz
SETTLE_TOL = 0.001           # <1 mm 3D Euclidean extremal-point motion


def _prep_boxes(box_list) -> tuple:
    out = []
    for spec in box_list:
        c = np.asarray(spec[0], float)
        h = 0.5 * np.asarray(spec[1], float)
        yaw = float(spec[3]) if len(spec) > 3 else 0.0
        out.append((c, h, Rz(-yaw)))
    return tuple(out)


def _prep_cyls(cyl_list) -> tuple:
    return tuple((np.asarray(c, float), float(r), 0.5 * float(h)) for c, r, h, *_ in cyl_list)


_CARAFE_BOX_SPECS = _prep_boxes(CARAFE_BOXES)
_CARAFE_CYL_SPECS = _prep_cyls(CARAFE_CYLS)
_COASTER_CYL_SPECS = _prep_cyls(COASTER_CYLS)
_STAND_BOX_SPECS = _prep_boxes(STAND_BOXES)
_STAND_CYL_SPECS = _prep_cyls(STAND_CYLS)
_LID_BOX_SPECS = _prep_boxes(LID_BOXES)
_LID_CYL_SPECS = _prep_cyls(LID_CYLS)
_PLG_BOX_SPECS = _prep_boxes(PLUNGER_BOXES)
_PLG_CYL_SPECS = _prep_cyls(PLUNGER_CYLS)
_COASTER_ORIGIN = np.array([COASTER_XY[0], COASTER_XY[1], TOP], float)
_STAND_ORIGIN = np.array([STAND_XY[0], STAND_XY[1], TOP], float)


def _dedup_shape_verts(box_list, cyl_list=()) -> np.ndarray:
    verts = []
    for spec in box_list:
        c = np.asarray(spec[0], float)
        hx, hy, hz = 0.5 * np.asarray(spec[1], float)
        yaw = float(spec[3]) if len(spec) > 3 else 0.0
        ry = Rz(yaw)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    verts.append(c + ry @ np.array([sx * hx, sy * hy, sz * hz]))
    for c, r, h, *_ in cyl_list:
        cc = np.asarray(c, float)
        hh = 0.5 * float(h)
        rr = float(r)
        for k in range(8):
            th = k * (0.25 * math.pi)
            for sz in (-1.0, 1.0):
                verts.append(cc + np.array([rr * math.cos(th), rr * math.sin(th), sz * hh]))
    arr = np.asarray(verts, float)
    _, idx = np.unique(np.round(arr, 4), axis=0, return_index=True)
    return arr[np.sort(idx)]


_CARAFE_CORNERS = _dedup_shape_verts(CARAFE_BOXES, CARAFE_CYLS)
_LID_CORNERS = _dedup_shape_verts(LID_BOXES, LID_CYLS)

_state: dict = {}
_hist: list = []

IMPACT_SPEED = 0.50
TABLE_X0, TABLE_X1 = -0.19, 0.91
TABLE_Y_HALF = 0.65
FLOOR_Z = -0.0125
CONTACT_TOL = 0.0008
PENETRATION_FLOOR = -0.004
CONTROL_DT = 0.05


@on_reset
def _reset(env):
    _state.clear()
    _state.update(
        step=0, viol=set(), settled=False, clear_all=False, prev_q=None,
        prev_seated=False, seated_travel=0.0, prev_filter_cr=None,
        prev_pts={}, prev_down={},
    )
    _hist.clear()


def _carafe_spilled(cr) -> bool:
    u_up = cr.rotation().T @ Z
    if float(u_up[2]) <= 0.0:
        return True
    return any(float(np.dot(p - _FILL_ORIGIN_LOCAL, u_up)) <= 0.0 for p in _CARAFE_LIP_PTS)


def _tips(env):
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        j7, j8 = np.abs(robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(float))
        out += [mid, mid - float(j7) * g[:3, 0], mid + float(j8) * g[:3, 0]]
    return out


def _min_box_dist(ql: np.ndarray, specs) -> float:
    return min(float(np.linalg.norm(np.maximum(np.abs(rt @ (ql - c)) - h, 0.0))) for c, h, rt in specs) if specs else 1e9


def _min_zcyl_dist(ql: np.ndarray, specs) -> float:
    return min(
        float(math.hypot(max(math.hypot(float(ql[0] - c[0]), float(ql[1] - c[1])) - r, 0.0),
                         max(abs(float(ql[2] - c[2])) - hh, 0.0)))
        for c, r, hh in specs
    ) if specs else 1e9


def _min_surf_dist_world(q_world: np.ndarray, cr, pl) -> float:
    q = np.asarray(q_world, float)
    qc = cr.rotation().T @ (q - np.asarray(cr.pos, float))
    d_cr = min(_min_box_dist(qc, _CARAFE_BOX_SPECS), _min_zcyl_dist(qc, _CARAFE_CYL_SPECS))
    d_co = _min_zcyl_dist(q - _COASTER_ORIGIN, _COASTER_CYL_SPECS)
    qst = q - _STAND_ORIGIN
    d_st = min(_min_box_dist(qst, _STAND_BOX_SPECS), _min_zcyl_dist(qst, _STAND_CYL_SPECS))

    ql = pl.rotation().T @ (q - np.asarray(pl.pos, float))
    d_ld = min(_min_box_dist(ql, _LID_BOX_SPECS), _min_zcyl_dist(ql, _LID_CYL_SPECS))
    qp = ql - np.array([0.0, 0.0, float(pl.joints["plunger"])])
    d_pl = min(_min_box_dist(qp, _PLG_BOX_SPECS), _min_zcyl_dist(qp, _PLG_CYL_SPECS))
    return min(d_cr, d_co, d_st, d_ld, d_pl)


def _spout_angle_deg(cr, pl) -> float:
    """Unsigned azimuthal angle (0..180 deg) in the carafe's horizontal frame between -Y pour spout and lid -Y pour slot."""
    rel_y = (cr.rotation().T @ pl.rotation())[:2, 1]
    denom = float(math.hypot(float(rel_y[0]), float(rel_y[1])))
    if denom < 1e-9:
        return 180.0
    cos_a = float(np.clip(float(rel_y[1]) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def _lid_seated_on_carafe(cr, pl) -> bool:
    rim_w = cr.to_world(np.array([0.0, 0.0, CARAFE_H]))
    dp = np.asarray(pl.pos, float) - rim_w
    rad = float(math.hypot(dp[0], dp[1]))
    return rad <= LID_RAD_TOL and abs(float(dp[2])) <= LID_Z_TOL and pl.tilt_deg() <= LID_TILT_TOL


def _extremal_pts_3d(cr, pl) -> np.ndarray:
    c_pts = [cr.to_world(p) for p in _CARAFE_CORNERS]
    l_pts = [pl.to_world(p) for p in _LID_CORNERS]
    q_plg = float(pl.joints["plunger"])
    p_pts = [
        pl.to_world(np.array([+0.018, 0.0, KNOB_CUP_LOCAL_Z + q_plg])),
        pl.to_world(np.array([0.0, +0.018, KNOB_CUP_LOCAL_Z + q_plg])),
        pl.to_world(np.array([+0.039, 0.0, -0.028 + q_plg])),
        pl.to_world(np.array([0.0, +0.039, -0.028 + q_plg])),
    ]
    return np.asarray(c_pts + l_pts + p_pts, float)


@on_step
def _watch(env):
    S = labkit._SPECS
    st = _state
    st["step"] += 1
    dt = float(getattr(getattr(env, "unwrapped", None), "step_dt", CONTROL_DT))
    cr = labkit._part_state(env, S["carafe"])
    pl = labkit._part_state(env, S["plunger_lid"])
    q_plg = float(pl.joints["plunger"])
    plunge_stroke = float(globals().get("PLUNGE_STROKE", 0.060))
    r_bore = float(globals().get("BEAKER_R_IN", 0.0450))
    z_fill = float(globals().get("Z_FILL_LOCAL", 0.07874))
    c_corners = np.asarray([cr.to_world(p) for p in _CARAFE_CORNERS], float)
    filter_local = np.array([
        [0.0, 0.0, -0.032 + q_plg],
        [+0.039, 0.0, -0.032 + q_plg],
        [-0.039, 0.0, -0.032 + q_plg],
        [0.0, +0.039, -0.032 + q_plg],
        [0.0, -0.039, -0.032 + q_plg],
        [0.0, 0.0, -0.024 + q_plg],
    ], float)
    pl_extra = np.vstack([filter_local[:5], np.array([[0.0, 0.0, 0.117 + q_plg]], float)])
    l_corners = np.asarray([pl.to_world(p) for p in _LID_CORNERS] + [pl.to_world(p) for p in pl_extra], float)
    cr_pos = np.asarray(cr.pos, float)
    cr_R_T = cr.rotation().T
    filter_cr = np.asarray([cr_R_T @ (pl.to_world(p) - cr_pos) for p in filter_local], float)
    seated = _lid_seated_on_carafe(cr, pl)
    rim_dp = np.asarray(pl.pos, float) - cr.to_world(np.array([0.0, 0.0, CARAFE_H]))
    removed_off_carafe = (float(math.hypot(float(rim_dp[0]), float(rim_dp[1]))) > 0.025) or (abs(float(rim_dp[2])) > 0.025)

    if st["step"] > 5:
        if _carafe_spilled(cr):
            st["viol"].add("carafe_spilled")
        for pname, p_st, p_pts in (("carafe", cr, c_corners), ("plunger_lid", pl, l_corners)):
            pos_p = np.asarray(p_st.pos, float)
            vel_pts = np.asarray(p_st.lin_vel, float)[None, :] + np.cross(
                np.asarray(p_st.ang_vel, float)[None, :], p_pts - pos_p[None, :]
            )
            down_now = -vel_pts[:, 2]
            prev_p = st["prev_pts"].get(pname)
            prev_d = st["prev_down"].get(pname)
            if prev_p is not None and prev_d is not None and dt > 0.0:
                fd_down = (prev_p[:, 2] - p_pts[:, 2]) / dt
                for i in range(len(p_pts)):
                    x, y, z = float(p_pts[i, 0]), float(p_pts[i, 1]), float(p_pts[i, 2])
                    on_table = ((TABLE_X0 <= x <= TABLE_X1)
                                and (abs(y) <= TABLE_Y_HALF)
                                and (PENETRATION_FLOOR <= (z - TOP) <= CONTACT_TOL))
                    on_floor = z <= (FLOOR_Z + CONTACT_TOL)
                    if on_table or on_floor:
                        z_surf = FLOOR_Z if on_floor else TOP
                        inc = max(float(down_now[i]), float(fd_down[i]))
                        prev_z = float(prev_p[i, 2])
                        if prev_z > (z_surf + CONTACT_TOL) and (prev_z - float(prev_d[i]) * dt) <= (z_surf + CONTACT_TOL):
                            inc = max(inc, float(prev_d[i]))
                        if inc > IMPACT_SPEED:
                            st["viol"].add(f"{pname}_impact")
                            break
            st["prev_pts"][pname] = p_pts
            st["prev_down"][pname] = down_now

        # Monitor both seated plunger joint speed and 3D filter entry/descent inside the filled brew bore (0..Z_FILL_LOCAL)
        if st["prev_q"] is not None and seated and dt > 0.0:
            v_down = (st["prev_q"] - q_plg) / dt
            if v_down > MAX_PLUNGE_SPEED:
                st["viol"].add("plunged_too_fast")
        prev_fc = st.get("prev_filter_cr")
        if prev_fc is not None and dt > 0.0:
            for i in range(len(filter_cr)):
                x0, y0, z0 = float(prev_fc[i, 0]), float(prev_fc[i, 1]), float(prev_fc[i, 2])
                x1, y1, z1 = float(filter_cr[i, 0]), float(filter_cr[i, 1]), float(filter_cr[i, 2])
                if z1 >= z0:
                    continue
                dz = z1 - z0
                s_lo = max(0.0, min((0.0 - z0) / dz, (z_fill - z0) / dz))
                s_hi = min(1.0, max((0.0 - z0) / dz, (z_fill - z0) / dz))
                if s_hi <= s_lo:
                    continue
                dx, dy = x1 - x0, y1 - y0
                qa = dx * dx + dy * dy
                qb = 2.0 * (x0 * dx + y0 * dy)
                qc = x0 * x0 + y0 * y0 - r_bore * r_bore
                if qa < 1e-12:
                    if qc > 0.0:
                        continue
                else:
                    disc = qb * qb - 4.0 * qa * qc
                    if disc < 0.0:
                        continue
                    sq = math.sqrt(disc)
                    s_lo = max(s_lo, (-qb - sq) / (2.0 * qa))
                    s_hi = min(s_hi, (-qb + sq) / (2.0 * qa))
                if (z0 - z1) * (s_hi - s_lo) > 1e-5 and (z0 - z1) / dt > MAX_PLUNGE_SPEED:
                    st["viol"].add("plunged_too_fast")
                    break
    else:
        for pname, p_st, p_pts in (("carafe", cr, c_corners), ("plunger_lid", pl, l_corners)):
            pos_p = np.asarray(p_st.pos, float)
            vel_pts = np.asarray(p_st.lin_vel, float)[None, :] + np.cross(
                np.asarray(p_st.ang_vel, float)[None, :], p_pts - pos_p[None, :]
            )
            st["prev_pts"][pname] = p_pts
            st["prev_down"][pname] = -vel_pts[:, 2]

    # Credit net downward plunger travel strictly while lid is seated on carafe (zero credit off-carafe; upward return
    # or full lid removal deducts/resets so oscillations or unseat-repress cycles cannot manufacture stroke credit).
    if removed_off_carafe:
        st["seated_travel"] = 0.0
    elif st["prev_q"] is not None:
        d_prev = float(np.clip(-float(st["prev_q"]), 0.0, plunge_stroke))
        d_cur = float(np.clip(-q_plg, 0.0, plunge_stroke))
        dd = d_cur - d_prev
        if seated and st.get("prev_seated", False):
            st["seated_travel"] = float(np.clip(float(st.get("seated_travel", 0.0)) + dd, 0.0, plunge_stroke))
        elif dd < 0.0:
            st["seated_travel"] = max(0.0, float(st.get("seated_travel", 0.0)) + dd)

    st["prev_q"] = q_plg
    st["prev_seated"] = seated
    st["prev_filter_cr"] = filter_cr

    st["clear_all"] = all(_min_surf_dist_world(tp, cr, pl) >= HANDS_CLEAR for tp in _tips(env))

    pts = _extremal_pts_3d(cr, pl)
    _hist.append(pts)
    if len(_hist) > SETTLE_STEPS + 1:
        _hist.pop(0)
    if len(_hist) == SETTLE_STEPS + 1:
        h = np.stack(_hist, axis=0)
        disp = np.linalg.norm(h[:, None, :, :] - h[None, :, :, :], axis=-1)
        st["settled"] = bool(float(np.max(disp)) < SETTLE_TOL)
    else:
        st["settled"] = False


def task_score(env) -> float:
    if _state.get("step", 0) == 0:
        _watch(env)
    st = _state
    if st["viol"] or not st["settled"] or not st["clear_all"]:
        return 0.0

    cr = labkit._part_state(env, labkit._SPECS["carafe"])
    pl = labkit._part_state(env, labkit._SPECS["plunger_lid"])
    if cr.tilt_deg() > CARAFE_END_TILT or not _lid_seated_on_carafe(cr, pl):
        return 0.0

    plunge_stroke = float(globals().get("PLUNGE_STROKE", 0.060))
    travel = float(np.clip(-float(pl.joints["plunger"]), 0.0, plunge_stroke))
    credited_travel = float(st.get("seated_travel", 0.0))
    spout_deg = _spout_angle_deg(cr, pl)
    spout_closed = spout_deg >= SPOUT_CLOSED_MIN_DEG
    plunged = (travel >= PLUNGE_GOAL) and (credited_travel >= PLUNGE_GOAL)
    if plunged and spout_closed:
        return 1.0
    if plunged:
        return 0.70
    if spout_closed:
        return 0.50
    return 0.30


SUCCESS_HOLD_S = 1.0


def score(objects) -> float:
    e = labkit.env()
    return float(task_score(e)) if e is not None else 0.0


def success(objects) -> bool:
    return score(objects) == 1.0


# -- Arbitrary-3D-orientation Waypoint monkeypatch + closed-loop live TCP/tip trim ------------------------------------
TIP_ABOVE_TCP = 0.013


class Orient(float):
    """Waypoint yaw carrying a full 3x3 TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_fp", False):
    def _grip_rot(yaw=0.0, _prev=_prev_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rot._fp = True
    _demo.grip_rotation = _grip_rot


def _ik_err(pe, re):
    return pe * 1000.0 + math.degrees(re)


def _endpoints(plan):
    rng = np.random.default_rng(7)
    seeds = [np.array(_READY, float)] + [np.array(_READY, float) + rng.normal(0.0, 0.8, 6) for _ in range(10)]
    plan._q0 = plan.q.copy()
    q_prev, out, switch = plan.q.copy(), [], set()
    for i, w in enumerate(plan.waypoints):
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, q_prev, base_pos=plan.base_pos)
        best = (_ik_err(pe, re), q, pe, re)
        if best[0] > 1.0 or float(np.max(np.abs(q - q_prev))) > 1.75:
            for s in seeds:
                qs, ps, rs = _ik.solve(pos, quat, s, base_pos=plan.base_pos, iters=200)
                cand = (_ik_err(ps, rs), qs, ps, rs)
                if cand[0] < best[0]:
                    best = cand
        _, q, pe, re = best
        if i > 0 and float(np.max(np.abs(q - q_prev))) > 1.75:
            switch.add(i)
        plan.worst_ik_mm = max(plan.worst_ik_mm, pe * 1000.0)
        plan.worst_ik_deg = max(plan.worst_ik_deg, math.degrees(re))
        out.append(q)
        q_prev = q
    plan._q_end, plan._switch = out, switch


def _action7(plan, t):
    _demo._plan_clock = t
    has_dyn = any(isinstance(w.pos, _Dyn) for w in plan.waypoints)
    if not hasattr(plan, "_q_end"):
        _endpoints(plan)
    if not has_dyn and not plan._switch:
        pos, rot, finger, label = plan.target(t)
        quat = _ik.matrix_to_quat_xyzw(rot)
        plan.q, pe, re = _ik.solve(pos, quat, plan.q, base_pos=plan.base_pos)
        plan.worst_ik_mm = max(plan.worst_ik_mm, pe * 1000.0)
        plan.worst_ik_deg = max(plan.worst_ik_deg, math.degrees(re))
        return np.append(plan.q, finger / FINGER_OPEN), label
    acc = 0.0
    for i, w in enumerate(plan.waypoints):
        if t <= acc + w.duration or i == len(plan.waypoints) - 1:
            u = 1.0 if w.duration <= 0 else min(max((t - acc) / w.duration, 0.0), 1.0)
            s = u * u * (3.0 - 2.0 * u)
            finger = plan.finger_start if i == 0 else plan.waypoints[i - 1].finger
            finger = finger + s * (w.finger - finger)
            grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            if i not in plan._switch and not isinstance(w.pos, _Dyn) and (i == 0 or not isinstance(plan.waypoints[i - 1].pos, _Dyn)):
                pos, rot, _, label = plan.target(t)
                quat = _ik.matrix_to_quat_xyzw(rot)
                plan.q, pe, re = _ik.solve(pos, quat, plan.q, base_pos=plan.base_pos, iters=80)
                plan.worst_ik_mm = max(plan.worst_ik_mm, pe * 1000.0)
                plan.worst_ik_deg = max(plan.worst_ik_deg, math.degrees(re))
                return np.append(plan.q, grip), label
            if not isinstance(w.pos, _Dyn) and (i == 0 or not isinstance(plan.waypoints[i - 1].pos, _Dyn)):
                q0 = plan._q0 if i == 0 else plan._q_end[i - 1]
                plan.q = (1.0 - s) * q0 + s * plan._q_end[i]
                return np.append(plan.q, grip), w.label
            _, rot, finger, label = plan.target(t)
            grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            p0 = np.asarray(plan.waypoints[i - 1].pos, float) if i > 0 else plan.start_pos
            p1 = np.asarray(w.pos, float)
            pos = (1.0 - s) * p0 + s * p1
            quat = _ik.matrix_to_quat_xyzw(rot)
            plan.q, pe, re = _ik.solve(pos, quat, plan.q, base_pos=plan.base_pos, iters=80)
            plan.worst_ik_mm = max(plan.worst_ik_mm, pe * 1000.0)
            plan.worst_ik_deg = max(plan.worst_ik_deg, math.degrees(re))
            return np.append(plan.q, grip), label
        acc += w.duration


ArmPlan.action7 = _action7


def _appr(R: np.ndarray) -> np.ndarray:
    return R @ _prev_grip(0.0).T @ np.array([0.0, 0.0, -1.0])


def _tcp(tip, R: np.ndarray, past: float = TIP_ABOVE_TCP) -> np.ndarray:
    a = _appr(R)
    return np.asarray(tip, float) + a * (past - TIP_ABOVE_TCP)


class _Plan:
    def __init__(self):
        self.L, self.R = [], []
        self.t = 0.0

    def hold(self, plan, dur, label="hold"):
        w = plan[-1]
        plan.append(Waypoint(dur, w.pos, w.finger, label, w.yaw))

    def phase(self, left=(), right=()):
        tl, tr = sum(w[0] for w in left), sum(w[0] for w in right)
        dur = max(tl, tr)
        for plan, ws, tot in ((self.L, left, tl), (self.R, right, tr)):
            for w in ws:
                plan.append(Waypoint(*w))
            if dur - tot > 1e-9:
                self.hold(plan, dur - tot)
        t0 = self.t
        self.t += dur
        return t0

    def window(self, t0, t1, key, fn):
        _ct["wins"].append((t0, t1, key, fn))


_ct = {"trim": {}, "wins": [], "last": {}, "lid_pose": None}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}
    _ct["lid_pose"] = None


def _live_tcp(env, robot_name: str) -> np.ndarray:
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    return _rt.tcp_world(b.scene[robot_name], 0, past_tcp=0.0) - o


def _live_tip(env, robot_name: str) -> np.ndarray:
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    return _rt.pad_tips_world(b.scene[robot_name], 0) - o


@on_step
def _ctrack(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, fn in _ct["wins"]:
        if not t0 <= t < t1:
            continue
        rname = "left_robot" if key == "hnd" else "right_robot"
        p = _live_tcp(env, rname)
        last = _ct["last"].get(key)
        _ct["last"][key] = p
        if last is None or np.max(np.abs(p - last)) > 0.0038:
            continue
        e = np.asarray(fn(env), float)
        tr = _ct["trim"].get(key, np.zeros(3))
        _ct["trim"][key] = np.clip(tr + np.clip(0.40 * e, -0.0032, 0.0032), -0.040, 0.040)
    if os.environ.get("FP_DEBUG", "0") == "1" and int(round(t * 20)) % 10 == 0:
        pl = labkit._part_state(env, labkit._SPECS["plunger_lid"])
        cr = labkit._part_state(env, labkit._SPECS["carafe"])
        print(
            f"FPDBG step={_state.get('step',0)} t={t:.2f} cr_pos={[round(float(v),4) for v in cr.pos]} "
            f"lid_pos={[round(float(v),4) for v in pl.pos]} "
            f"lid_tilt={pl.tilt_deg():.2f} cr_tilt={cr.tilt_deg():.2f} q_plg={float(pl.joints['plunger']):.4f} "
            f"seated_trv={_state.get('seated_travel',0.0):.4f} spout_deg={_spout_angle_deg(cr, pl):.1f} "
            f"viol={sorted(_state.get('viol',()))}",
            flush=True,
        )


class _Dyn:
    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)


# -- Grasp, mid-air whole-lid spout-closing turn, and press helpers (robot-encoder PD trim only) ----------------------
DOWN = _prev_grip(0.0)

# 1. Left-arm carafe handle brace (side-approach from +Y/-X at 74 deg from vertical, clamping 26 mm X-width inside 69 mm window)
HND_OPEN = -0.028
HND_HOLD = 0.0
HND_PAST = 0.012
R_HND = Rz(-math.radians(78.0)) @ Ry(-math.radians(74.0)) @ DOWN


def _hnd_pose(back: float = 0.0):
    a = _appr(R_HND)
    p = CARAFE_START + HANDLE_GRIP_LOCAL - back * a
    return _tcp(p, R_HND, past=HND_PAST), Orient(R_HND)


# 2. Right-arm on-axis pinch on the authentic 22 mm OD central guide collar (z_local = 0.0305 m):
#    Grasps at yaw = -35 deg on the drip stand, lifts +95 mm above carafe rim, and sweeps smoothly in 10 short 9.2-deg
#    legs to yaw = +57 deg (+92 deg CCW turn, max step |dq| = 0.0382 rad, 0 branch switches!) so the lid's solid pilot
#    skirt already occludes the carafe's -Y pour spout upon coaxial seating (per Bodum Chambord manual)!
COL_OPEN = -0.024
COL_HOLD = 0.0
COL_PAST = 0.009
COL_PITCH_DEG = 50.0
COL_YAW_START = -35.0
COL_YAW_END = +57.0


def _col_rot(yaw_deg: float) -> np.ndarray:
    return Rz(math.radians(yaw_deg)) @ Ry(-math.radians(COL_PITCH_DEG)) @ DOWN


def _col_pose(lid_origin, yaw_deg: float, back: float = 0.0):
    R = _col_rot(yaw_deg)
    a = _appr(R)
    p = np.asarray(lid_origin, float) + COLLAR_GRIP_LOCAL - back * a
    return _tcp(p, R, past=COL_PAST), Orient(R)


# 3. Right-arm closed-fingertip vertical press on dished 36 mm brass knob (28 deg from vertical from -Y/-X)
KNOB_CLOSED = 0.0
R_PRESS = Rz(math.radians(35.0)) @ Ry(-math.radians(28.0)) @ DOWN


def _press_pose(z_world: float):
    p = np.array([COASTER_XY[0], COASTER_XY[1], z_world], float)
    return _tcp(p, R_PRESS, past=0.0), Orient(R_PRESS)


LEFT_PARK = np.array([0.16, +0.27, TOP + 0.21])
RIGHT_PARK = np.array([0.16, -0.27, TOP + 0.21])


def _stand_press_pose(z_world: float):
    """Right-arm vertical-ish press on the brass knob directly on LID_STAND_START."""
    R_sp = Rz(-math.radians(25.0)) @ Ry(-math.radians(28.0)) @ DOWN
    p = np.array([LID_STAND_START[0], LID_STAND_START[1], z_world], float)
    return _tcp(p, R_sp, past=0.0), Orient(R_sp)


def _shortcut_demo(fast_insert: bool = False):
    """Negative control: depress plunger 30 mm ON THE DRIP STAND first (earning 0 credit), then carry & seat via demo().
    - stand_plunge (fast_insert=False): seats slowly and presses remaining 30 mm to -60 mm bottom stop -> scores 0.50 (NOT 1.0!).
    - fast_insert (fast_insert=True): plunges pre-pressed filter into filled brew bore at >200 mm/s -> latches plunged_too_fast (0.0).
    """
    _ct["wins"].clear()
    pl = _Plan()
    o_wait = Orient(Ry(-math.radians(35.0)) @ DOWN)
    p_h_pre, o_h = _hnd_pose(back=0.060)
    p_h_hold, _ = _hnd_pose(back=0.016)

    # Phase 0: Depress plunger 30 mm directly ON THE DRIP STAND before pickup (seated_travel stays 0.0000!)
    z_st_knob0 = float(LID_STAND_START[2] + KNOB_CUP_LOCAL_Z)
    p_sp_pre, o_sp = _stand_press_pose(z_st_knob0 + 0.024)
    p_sp_touch, _ = _stand_press_pose(z_st_knob0 - 0.002)
    p_sp_bot, _ = _stand_press_pose(z_st_knob0 - 0.038)
    p_sp_up, _ = _stand_press_pose(z_st_knob0 + 0.025)
    p_t_safe = np.array([0.215, -0.275, TOP + 0.215])
    p_t_pre, o_t0 = _col_pose(LID_STAND_START, COL_YAW_START, back=0.060)
    p_t_grasp, _ = _col_pose(LID_STAND_START, COL_YAW_START, back=0.0)

    pl.phase(
        left=[(1.20, p_h_pre, HND_OPEN, "align short of carafe handle", o_h)],
        right=[(1.20, p_sp_pre, KNOB_CLOSED, "align closed fingertips over knob on drip stand", o_sp)],
    )
    t_sp = pl.phase(
        left=[(0.85, _Dyn(p_h_pre, "hnd"), HND_OPEN, "encoder PD trim at carafe handle", o_h)],
        right=[(0.85, _Dyn(p_sp_pre, "st_knb"), KNOB_CLOSED, "encoder PD trim over stand knob", o_sp)],
    )
    pl.window(t_sp + 0.20, t_sp + 0.85, "hnd", lambda env, _p=p_h_pre: _p - _live_tcp(env, "left_robot"))
    pl.window(t_sp + 0.20, t_sp + 0.85, "st_knb", lambda env, _p=p_sp_pre: _p - _live_tcp(env, "right_robot"))

    pl.phase(
        left=[(4.50, _Dyn(p_h_pre, "hnd"), HND_OPEN, "wait clear during on-stand pre-plunge", o_h)],
        right=[
            (0.55, _Dyn(p_sp_touch, "st_knb"), KNOB_CLOSED, "engage brass knob on stand", o_sp),
            (1.60, _Dyn(p_sp_bot, "st_knb"), KNOB_CLOSED, "depress plunger 30 mm on drip stand", o_sp),
            (0.35, _Dyn(p_sp_bot, "st_knb"), KNOB_CLOSED, "hold stand bottom", o_sp),
            (0.65, _Dyn(p_sp_up, "st_knb"), COL_OPEN, "lift clear above stand-pressed knob", o_sp),
            (0.65, p_t_safe, COL_OPEN, "swing outboard to collar approach", o_wait),
            (0.70, p_t_pre, COL_OPEN, "align short of lid guide collar", o_t0),
        ],
    )
    pl.phase(
        left=[(0.75, _Dyn(p_h_pre, "hnd"), HND_OPEN, "wait short of carafe handle", o_h)],
        right=[(0.75, p_t_grasp, COL_OPEN, "advance open jaws onto guide collar", o_t0)],
    )
    t_look1 = pl.phase(
        left=[(0.85, _Dyn(p_h_pre, "hnd"), HND_OPEN, "hold carafe handle trim", o_h)],
        right=[(0.85, _Dyn(p_t_grasp, "tur0"), COL_OPEN, "encoder PD trim at collar grasp height", o_t0)],
    )
    pl.window(t_look1 + 0.20, t_look1 + 0.85, "tur0", lambda env, _p=p_t_grasp: _p - _live_tcp(env, "right_robot"))

    # Phase 1: 10-leg +92 deg CCW collar carry with [+5.5, +9.5] mm bore-entry centering for the -31 mm stand-pressed filter!
    lid_lift_stand = np.array([LID_STAND_START[0], LID_STAND_START[1], CARAFE_START[2] + CARAFE_H + 0.095])
    lid_over_carafe = CARAFE_START + np.array([+0.0055, +0.0095, CARAFE_H + 0.095])
    lid_rim_clear = CARAFE_START + np.array([+0.0055, +0.0095, CARAFE_H + 0.048])
    lid_rim_inside = CARAFE_START + np.array([+0.0040, +0.0065, CARAFE_H + 0.018])
    lid_seated = CARAFE_START + np.array([+0.0025, +0.0045, CARAFE_H - 0.004])

    p_t_lift, _ = _col_pose(lid_lift_stand, COL_YAW_START)
    carry_wps = []
    for _k in range(1, 11):
        _f = _k / 10.0
        _yk = COL_YAW_START + _f * (COL_YAW_END - COL_YAW_START)
        _p_c, _o_c = _col_pose((1.0 - _f) * lid_lift_stand + _f * lid_over_carafe, _yk)
        carry_wps.append((0.26, _p_c, COL_HOLD, f"carry & turn lid leg {_k}/10", _o_c))

    p_t_rim_clear, o_t1 = _col_pose(lid_rim_clear, COL_YAW_END)
    p_t_inside, _ = _col_pose(lid_rim_inside, COL_YAW_END)
    p_t_seat, _ = _col_pose(lid_seated, COL_YAW_END)
    p_t_rel, _ = _col_pose(lid_seated, COL_YAW_END, back=0.070)

    pl.phase(
        left=[(5.10, _Dyn(p_h_pre, "hnd"), HND_OPEN, "wait clear of carafe during lid carry", o_h)],
        right=[
            (0.50, _Dyn(p_t_grasp, "tur0"), COL_HOLD, "grasp central guide collar on axis", o_t0),
            (1.25, _Dyn(p_t_lift, "tur0"), COL_HOLD, "lift stand-pressed lid plumb clear of stand", o_t0),
            *carry_wps,
            (0.75, p_t_rim_clear, COL_HOLD, "descend above carafe bore rim", o_t1),
        ],
    )
    t_rim_trim = pl.phase(
        left=[(1.00, _Dyn(p_h_pre, "hnd"), HND_OPEN, "stay clear during rim-plane XY trim", o_h)],
        right=[(1.00, _Dyn(p_t_rim_clear, "tur1"), COL_HOLD, "converge encoder trim at bore rim plane", o_t1)],
    )
    pl.window(t_rim_trim + 0.20, t_rim_trim + 1.00, "tur1", lambda env, _p=p_t_rim_clear: _p - _live_tcp(env, "right_robot"))

    if fast_insert:
        p_t_fast_deep, _ = _col_pose(CARAFE_START + np.array([+0.0045, +0.0075, CARAFE_H - 0.012]), COL_YAW_END)
        pl.phase(
            left=[(3.10, LEFT_PARK, HND_OPEN, "left park clear", o_wait)],
            right=[
                (0.15, _Dyn(p_t_fast_deep, "tur1"), COL_HOLD, "fast pre-pressed insertion (>300 mm/s into brew)", o_t1),
                (0.50, _Dyn(p_t_seat, "tur1"), COL_OPEN, "release collar", o_t1),
                (0.85, p_t_rel, COL_OPEN, "back clear of collar", o_t1),
                (1.60, RIGHT_PARK, COL_OPEN, "right park clear", o_wait),
            ],
        )
        pl.phase(
            left=[(2.2, LEFT_PARK, HND_OPEN, "left final settle", o_wait)],
            right=[(2.2, RIGHT_PARK, COL_OPEN, "right final settle", o_wait)],
        )
        return {"left": pl.L, "right": pl.R}

    pl.phase(
        left=[(3.25, _Dyn(p_h_pre, "hnd"), HND_OPEN, "stay clear while pre-pressed lid seats coaxially", o_h)],
        right=[
            (0.95, _Dyn(p_t_inside, "tur1"), COL_HOLD, "descend slowly into 90 mm bore", o_t1),
            (1.05, _Dyn(p_t_seat, "tur1"), COL_HOLD, "seat pre-closed lid onto carafe rim", o_t1),
            (0.45, _Dyn(p_t_seat, "tur1"), COL_OPEN, "release seated guide collar", o_t1),
            (0.80, p_t_rel, COL_OPEN, "back clear of guide collar", o_t1),
        ],
    )

    # Phase 2: Complete remaining ~27 mm stroke while seated -> reaches ~-58 mm final depth (>=52 mm), earns 0.50 (NOT 1.0!)
    z_knob0 = float(CARAFE_START[2] + CARAFE_H + KNOB_CUP_LOCAL_Z)
    p_k_clear = np.array([0.205, -0.095, z_knob0 + 0.045])
    p_k_pre, o_k = _press_pose(z_knob0 + 0.024)
    p_k_mid, _ = _press_pose(z_knob0 - 0.034)
    p_k_bot, _ = _press_pose(z_knob0 - 0.066)
    p_k_up, _ = _press_pose(z_knob0 - 0.015)
    p_h_rel, _ = _hnd_pose(back=0.070)

    pl.phase(
        left=[
            (0.85, _Dyn(p_h_hold, "hnd"), HND_OPEN, "advance onto seated carafe handle", o_h),
            (0.55, _Dyn(p_h_hold, "hnd"), HND_HOLD, "brace carafe handle", o_h),
        ],
        right=[
            (0.70, p_k_clear, KNOB_CLOSED, "lift clear outboard of knob", o_k),
            (0.70, p_k_pre, KNOB_CLOSED, "close fingers above knob", o_k),
        ],
    )
    t_look2 = pl.phase(
        left=[(0.9, _Dyn(p_h_hold, "hnd"), HND_HOLD, "hold carafe for knob encoder trim", o_h)],
        right=[(0.9, _Dyn(p_k_pre, "knb"), KNOB_CLOSED, "encoder PD trim above knob", o_k)],
    )
    pl.window(t_look2 + 0.20, t_look2 + 0.9, "knb", lambda env, _p=p_k_pre: _p - _live_tcp(env, "right_robot"))

    pl.phase(
        left=[
            (5.05, _Dyn(p_h_hold, "hnd"), HND_HOLD, "brace carafe during remaining stroke", o_h),
            (0.50, _Dyn(p_h_hold, "hnd"), HND_OPEN, "release carafe handle", o_h),
            (0.90, p_h_rel, HND_OPEN, "back clear of carafe handle", o_h),
            (1.10, LEFT_PARK, HND_OPEN, "left park clear", o_wait),
        ],
        right=[
            (1.90, _Dyn(p_k_mid, "knb"), KNOB_CLOSED, "engage partly-pressed knob", o_k),
            (1.90, _Dyn(p_k_bot, "knb"), KNOB_CLOSED, "complete remaining 30 mm stroke to bottom stop", o_k),
            (0.50, _Dyn(p_k_bot, "knb"), KNOB_CLOSED, "hold at bottom stop", o_k),
            (0.75, _Dyn(p_k_up, "knb"), COL_OPEN, "lift clear above pressed knob", o_k),
            (1.30, RIGHT_PARK, COL_OPEN, "right park clear", o_wait),
        ],
    )
    pl.phase(
        left=[(2.2, LEFT_PARK, HND_OPEN, "left final settle", o_wait)],
        right=[(2.2, RIGHT_PARK, COL_OPEN, "right final settle", o_wait)],
    )
    return {"left": pl.L, "right": pl.R}


def demo():
    mode = os.environ.get("FRENCH_PRESS_MODE", "demo").strip().lower()
    if mode == "stand_plunge":
        return _shortcut_demo(fast_insert=False)
    if mode == "fast_insert":
        return _shortcut_demo(fast_insert=True)
    _ct["wins"].clear()
    pl = _Plan()
    o_wait = Orient(Ry(-math.radians(35.0)) @ DOWN)

    # Phase 1: Left arm aligns & robot-encoder trims short of carafe handle; Right arm swings outboard (-Y = -0.275 m,
    # 110 mm clear of the raised brass plunger knob), advances onto the 22 mm central guide collar, and encoder-trims
    # directly at grasp height so jaws close dead-centered on axis!
    p_h_pre, o_h = _hnd_pose(back=0.060)
    p_h_hold, _ = _hnd_pose(back=0.016)
    p_t_safe = np.array([0.215, -0.275, TOP + 0.215])
    p_t_pre, o_t0 = _col_pose(LID_STAND_START, COL_YAW_START, back=0.060)
    p_t_grasp, _ = _col_pose(LID_STAND_START, COL_YAW_START, back=0.0)

    pl.phase(
        left=[(1.3, p_h_pre, HND_OPEN, "align short of carafe handle", o_h)],
        right=[
            (0.55, p_t_safe, COL_OPEN, "swing outboard clear of raised plunger knob", o_wait),
            (0.75, p_t_pre, COL_OPEN, "align short of lid guide collar", o_t0),
        ],
    )

    pl.phase(
        left=[(0.75, p_h_pre, HND_OPEN, "wait short of carafe handle", o_h)],
        right=[(0.75, p_t_grasp, COL_OPEN, "advance open jaws onto guide collar", o_t0)],
    )
    t_look1 = pl.phase(
        left=[(0.85, _Dyn(p_h_pre, "hnd"), HND_OPEN, "encoder PD trim at carafe handle", o_h)],
        right=[(0.85, _Dyn(p_t_grasp, "tur0"), COL_OPEN, "encoder PD trim at collar grasp height", o_t0)],
    )
    pl.window(t_look1 + 0.20, t_look1 + 0.85, "hnd", lambda env, _p=p_h_pre: _p - _live_tcp(env, "left_robot"))
    pl.window(t_look1 + 0.20, t_look1 + 0.85, "tur0", lambda env, _p=p_t_grasp: _p - _live_tcp(env, "right_robot"))

    # Lift plunger_lid +95 mm above carafe rim, rotate +92 deg CCW across 10 smooth 9.2-deg high-clearance carry legs,
    # lower to +48 mm over carafe rim (filter disc 16 mm clear above bore plane), converge XY encoder trim, freeze trim,
    # and descend straight down along Z into the untouched 90 mm bore!
    lid_lift_stand = np.array([LID_STAND_START[0], LID_STAND_START[1], CARAFE_START[2] + CARAFE_H + 0.095])
    lid_over_carafe = CARAFE_START + np.array([+0.0020, +0.0035, CARAFE_H + 0.095])
    lid_rim_clear = CARAFE_START + np.array([+0.0020, +0.0035, CARAFE_H + 0.048])
    lid_rim_inside = CARAFE_START + np.array([+0.0020, +0.0035, CARAFE_H + 0.020])
    lid_seated = CARAFE_START + np.array([+0.0020, +0.0035, CARAFE_H - 0.004])

    p_t_lift, _ = _col_pose(lid_lift_stand, COL_YAW_START)
    carry_wps = []
    for _k in range(1, 11):
        _f = _k / 10.0
        _yk = COL_YAW_START + _f * (COL_YAW_END - COL_YAW_START)
        _p_c, _o_c = _col_pose((1.0 - _f) * lid_lift_stand + _f * lid_over_carafe, _yk)
        carry_wps.append((0.26, _p_c, COL_HOLD, f"carry & turn lid leg {_k}/10", _o_c))

    p_t_rim_clear, o_t1 = _col_pose(lid_rim_clear, COL_YAW_END)
    p_t_inside, _ = _col_pose(lid_rim_inside, COL_YAW_END)
    p_t_seat, _ = _col_pose(lid_seated, COL_YAW_END)
    p_t_rel, _ = _col_pose(lid_seated, COL_YAW_END, back=0.070)

    pl.phase(
        left=[(5.10, _Dyn(p_h_pre, "hnd"), HND_OPEN, "wait clear of carafe during lid carry", o_h)],
        right=[
            (0.50, _Dyn(p_t_grasp, "tur0"), COL_HOLD, "grasp central guide collar on axis", o_t0),
            (1.25, _Dyn(p_t_lift, "tur0"), COL_HOLD, "lift plunger_lid plumb clear of stand", o_t0),
            *carry_wps,
            (0.75, p_t_rim_clear, COL_HOLD, "descend to 16 mm above carafe bore rim", o_t1),
        ],
    )
    t_rim_trim = pl.phase(
        left=[(1.00, _Dyn(p_h_pre, "hnd"), HND_OPEN, "stay clear during rim-plane XY trim", o_h)],
        right=[(1.00, _Dyn(p_t_rim_clear, "tur1"), COL_HOLD, "converge encoder trim at bore rim plane", o_t1)],
    )
    pl.window(t_rim_trim + 0.20, t_rim_trim + 1.00, "tur1", lambda env, _p=p_t_rim_clear: _p - _live_tcp(env, "right_robot"))

    pl.phase(
        left=[(3.25, _Dyn(p_h_pre, "hnd"), HND_OPEN, "stay clear while lid seats coaxially", o_h)],
        right=[
            (0.95, _Dyn(p_t_inside, "tur1"), COL_HOLD, "descend straight with frozen trim into 90 mm bore", o_t1),
            (1.05, _Dyn(p_t_seat, "tur1"), COL_HOLD, "seat pre-closed safety lid onto carafe rim", o_t1),
            (0.45, _Dyn(p_t_seat, "tur1"), COL_OPEN, "release seated guide collar", o_t1),
            (0.80, p_t_rel, COL_OPEN, "back clear of guide collar", o_t1),
        ],
    )

    # Phase 2: Left arm advances & clamps carafe handle while Right arm lifts clear outboard of brass knob, closes
    # fingertips, encoder-trims 24 mm above knob cup, and presses smoothly to -60 mm bottom stop!
    z_knob0 = float(CARAFE_START[2] + CARAFE_H + KNOB_CUP_LOCAL_Z)
    p_k_clear = np.array([0.205, -0.095, z_knob0 + 0.045])
    p_k_pre, o_k = _press_pose(z_knob0 + 0.024)
    p_k_touch, _ = _press_pose(z_knob0 - 0.002)
    p_k_mid, _ = _press_pose(z_knob0 - 0.042)
    p_k_bot, _ = _press_pose(z_knob0 - 0.082)
    p_k_up, _ = _press_pose(z_knob0 - 0.015)
    p_h_rel, _ = _hnd_pose(back=0.070)

    pl.phase(
        left=[
            (0.85, _Dyn(p_h_hold, "hnd"), HND_OPEN, "advance onto seated carafe handle", o_h),
            (0.55, _Dyn(p_h_hold, "hnd"), HND_HOLD, "brace carafe handle before plunge", o_h),
        ],
        right=[
            (0.70, p_k_clear, KNOB_CLOSED, "lift clear outboard of raised brass knob", o_k),
            (0.70, p_k_pre, KNOB_CLOSED, "close fingers above brass plunger knob", o_k),
        ],
    )
    t_look2 = pl.phase(
        left=[(0.9, _Dyn(p_h_hold, "hnd"), HND_HOLD, "hold carafe for knob encoder trim", o_h)],
        right=[(0.9, _Dyn(p_k_pre, "knb"), KNOB_CLOSED, "encoder PD trim above brass plunger knob", o_k)],
    )
    pl.window(t_look2 + 0.20, t_look2 + 0.9, "knb", lambda env, _p=p_k_pre: _p - _live_tcp(env, "right_robot"))

    pl.phase(
        left=[
            (5.65, _Dyn(p_h_hold, "hnd"), HND_HOLD, "brace carafe against filtration load", o_h),
            (0.50, _Dyn(p_h_hold, "hnd"), HND_OPEN, "release carafe handle", o_h),
            (0.90, p_h_rel, HND_OPEN, "back clear of carafe handle", o_h),
            (1.10, LEFT_PARK, HND_OPEN, "left park clear", o_wait),
        ],
        right=[
            (0.60, _Dyn(p_k_touch, "knb"), KNOB_CLOSED, "engage dished brass plunger knob", o_k),
            (1.90, _Dyn(p_k_mid, "knb"), KNOB_CLOSED, "smooth filtration plunge first half", o_k),
            (1.90, _Dyn(p_k_bot, "knb"), KNOB_CLOSED, "smooth filtration plunge to bottom stop", o_k),
            (0.50, _Dyn(p_k_bot, "knb"), KNOB_CLOSED, "hold plunger at bottom stop", o_k),
            (0.75, _Dyn(p_k_up, "knb"), COL_OPEN, "lift clear above pressed knob", o_k),
            (1.30, RIGHT_PARK, COL_OPEN, "right park clear", o_wait),
        ],
    )

    pl.phase(
        left=[(2.2, LEFT_PARK, HND_OPEN, "left final settle", o_wait)],
        right=[(2.2, RIGHT_PARK, COL_OPEN, "right final settle", o_wait)],
    )
    return {"left": pl.L, "right": pl.R}
