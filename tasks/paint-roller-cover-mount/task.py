"""Slide a 9-inch paint-roller cover onto a loose cranked roller frame.

Modeled 1:1 on a Wooster Sherlock R017-9 9-inch cage paint-roller frame (5/16-inch
90-degree chrome-plated steel shank, free-turning fiberglass-reinforced nylon cage, and
compliant radial quick-release retaining spring) and a standard 9 x 1.5 inch (3/8-inch nap,
57.2 mm OD) cylindrical paint-roller cover.
"""
import json
import math
import os
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401
import labkit
from labkit import Joint, on_reset, on_step, part
from task_metrics import Metric, rx, ry
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS, SHOULDER, WRIST
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 900
SUCCESS_HOLD_S = 2.5
labkit.CONTACT_OFFSET = 0.0005

# -- 1:1 Wooster Sherlock R017-9 + 9-in x 1.5-in (3/8-in nap) roller cover geometry ----------
CAGE_LEN = 0.2286      # 9.0 in
COVER_LEN = 0.2286     # 9.0 in
CORE_R_IN = 0.01905    # 38.1 mm (1.50 in) ID
COVER_R_OUT = 0.02860  # 57.2 mm (2.25 in) OD

C_STEEL = (0.76, 0.78, 0.81)
C_CAGE = (0.16, 0.18, 0.20)
C_SPRING = (0.18, 0.62, 0.28)   # Wooster Sherlock signature green quick-release clip
C_HANDLE = (0.15, 0.17, 0.19)
C_BOLSTER = (0.20, 0.56, 0.26)
C_NAP = (0.94, 0.89, 0.72)      # classic woven golden-cream roller nap

# ROLLER_FRAME local origin = inboard shoulder stop face (y=0.0) on cage centerline (x=0, z=0).
# Cage extends along -Y from y=0.0 to y=-0.2286 m; cranked shank steps -X by 135 mm to ergonomic handle.
FRAME_BOXES = (
    # Ergonomic flat-sided polypropylene grip handle at x = -0.135 m, y in [+0.056, +0.196] m:
    ((-0.1350, +0.0620, 0.0), (0.0360, 0.0120, 0.0400), C_BOLSTER, 0.0),  # front thumb bolster (touches z=-0.020)
    ((-0.1350, +0.1260, 0.0), (0.0320, 0.1160, 0.0340), C_HANDLE, 0.0),   # flat-sided 32 mm grip barrel
    ((-0.1350, +0.1900, 0.0), (0.0360, 0.0120, 0.0400), C_HANDLE, 0.0),   # rear butt flange (touches z=-0.020)
)

FRAME_CYLS = (
    # Inboard 40.0 mm OD nylon shoulder stop collar (y in [0.000, +0.006] m, touches z=-0.020):
    ((0.0, +0.0030, 0.0), 0.0200, 0.0060, C_CAGE, "Y"),
    # 7.9 mm (5/16-in) 90-deg cranked chrome-plated steel rod:
    ((0.0, +0.0190, 0.0), 0.00395, 0.0260, C_STEEL, "Y"),
    ((-0.0675, +0.0320, 0.0), 0.00395, 0.1430, C_STEEL, "X"),
    ((-0.1350, +0.0440, 0.0), 0.00395, 0.0240, C_STEEL, "Y"),
    # Internal spindle shaft inside cage:
    ((0.0, -0.1120, 0.0), 0.00395, 0.2240, C_STEEL, "Y"),
)

# Free-turning fiberglass-reinforced nylon cage (36.0 mm OD inside 38.1 mm cover bore = 1.05 mm radial clearance):
CAGE_BOXES = tuple(
    (
        (0.0155 * math.cos(2.0 * math.pi * k / 5.0), -0.1080, -0.0155 * math.sin(2.0 * math.pi * k / 5.0)),
        (0.0046, 0.1960, 0.0044),
        C_CAGE,
        (0.0, 2.0 * math.pi * k / 5.0),
    )
    for k in range(5)
)

CAGE_CYLS = (
    # Tapered outboard nylon nose cap for smooth coaxial lead-in (30.0 mm -> 33.6 mm -> 35.8 mm OD):
    ((0.0, -0.2256, 0.0), 0.0150, 0.0060, C_CAGE, "Y"),
    ((0.0, -0.2190, 0.0), 0.0168, 0.0072, C_CAGE, "Y"),
    ((0.0, -0.2110, 0.0), 0.0179, 0.0088, C_CAGE, "Y"),
    # Central and inboard nylon support hubs (35.8 mm OD):
    ((0.0, -0.1140, 0.0), 0.0179, 0.0100, C_CAGE, "Y"),
    ((0.0, -0.0100, 0.0), 0.0179, 0.0120, C_CAGE, "Y"),
)

# Compliant outward-sprung quick-release retaining leaf centered 28 mm from inboard shoulder:
# Symmetric 14-deg clipped wedge ramps (outboard [-45.8345, -33.0] mm and inboard [-23.0, -10.1655] mm) meeting
# the 10 mm horizontal crown pad [-33.0, -23.0] mm at an exact 0.000 um seam (max_z = +20.400 mm everywhere).
DETENT_ANCHOR = (0.0, -0.0280, 0.0)
_WEDGE_TRIS = np.array(
    [
        (0, 2, 3), (0, 3, 1),  # -Y face
        (4, 5, 7), (4, 7, 6),  # +Y face
        (0, 1, 5), (0, 5, 4),  # -X face
        (2, 6, 7), (2, 7, 3),  # +X face
        (1, 3, 7), (1, 7, 5),  # +Z top ramp face
        (0, 4, 6), (0, 6, 2),  # -Z bottom face
    ],
    dtype=np.int32,
)
_WEDGE_OUT_PTS = np.array(
    [
        [-0.004, -0.04583449898731471, 0.014520404563690265],
        [-0.004, -0.04583449898731471, 0.0172],
        [+0.004, -0.04583449898731471, 0.014520404563690265],
        [+0.004, -0.04583449898731471, 0.0172],
        [-0.004, -0.033, 0.017720404563690267],
        [-0.004, -0.033, 0.0204],
        [+0.004, -0.033, 0.017720404563690267],
        [+0.004, -0.033, 0.0204],
    ],
    dtype=float,
)
_WEDGE_INB_PTS = np.array(
    [
        [-0.004, -0.023, 0.017720404563690267],
        [-0.004, -0.023, 0.0204],
        [+0.004, -0.023, 0.017720404563690267],
        [+0.004, -0.023, 0.0204],
        [-0.004, -0.01016550101268529, 0.014520404563690265],
        [-0.004, -0.01016550101268529, 0.0172],
        [+0.004, -0.01016550101268529, 0.014520404563690265],
        [+0.004, -0.01016550101268529, 0.0172],
    ],
    dtype=float,
)
_CROWN_Z_BOT = 0.017720404563690267
_CROWN_Z_TOP = 0.0204
DETENT_MESHES = (
    labkit.Mesh(points=_WEDGE_OUT_PTS, tris=_WEDGE_TRIS, color=C_SPRING, collide="hull"),
    labkit.Mesh(points=_WEDGE_INB_PTS, tris=_WEDGE_TRIS, color=C_SPRING, collide="hull"),
)
DETENT_BOXES = (
    (
        (0.0, -0.0280, 0.5 * (_CROWN_Z_BOT + _CROWN_Z_TOP)),
        (0.0080, 0.0100, _CROWN_Z_TOP - _CROWN_Z_BOT),
        C_SPRING,
        0.0,
    ),
)

# 16-stave hollow cylindrical roller cover (ID = 38.1 mm, OD = 57.2 mm, L = 228.6 mm along Y):
_N_STAVES = 16
_WALL_THK = COVER_R_OUT - CORE_R_IN                 # 0.00955 m
_R_MID = 0.5 * (CORE_R_IN + COVER_R_OUT)            # 0.023825 m
_CHORD_W = 2.0 * CORE_R_IN * math.tan(math.pi / _N_STAVES)  # 0.00758 m

COVER_BOXES = tuple(
    (
        (_R_MID * math.cos(2.0 * math.pi * k / _N_STAVES), 0.0, -_R_MID * math.sin(2.0 * math.pi * k / _N_STAVES)),
        (_WALL_THK, COVER_LEN, _CHORD_W),
        C_NAP,
        (0.0, 2.0 * math.pi * k / _N_STAVES),
    )
    for k in range(_N_STAVES)
)

labkit.AXIS_FRICTION.update({
    "cage_spin": (0.002, 0.0015),
})

FRAME_X0, FRAME_Y0, FRAME_Z0 = 0.3750, +0.1150, TOP + 0.0200
COVER_X0, COVER_Y0, COVER_Z0 = 0.2550, -0.2350, TOP + COVER_R_OUT

ROLLER_FRAME = part(
    "roller_frame",
    (FRAME_X0, FRAME_Y0, FRAME_Z0),
    boxes=FRAME_BOXES,
    cylinders=FRAME_CYLS,
    fixed=False,
    mass=0.21,
    self_collide=False,
    friction=0.55,
    joints=(
        Joint(
            name="cage_spin",
            kind="hinge",
            axis="Y",
            anchor=(0.0, 0.0, 0.0),
            boxes=CAGE_BOXES,
            cylinders=CAGE_CYLS,
            lower=-120.0,
            upper=+120.0,
            stiffness=0.0,
            damping=0.001,
            target=0.0,
            initial=0.0,
            friction=0.0,
            mass=0.11,
        ),
        Joint(
            name="detent_z",
            parent="cage_spin",
            kind="slide",
            axis="Z",
            anchor=DETENT_ANCHOR,
            boxes=DETENT_BOXES,
            meshes=DETENT_MESHES,
            lower=-0.0028,
            upper=0.0,
            stiffness=1400.0,
            damping=3.0,
            target=0.0,
            initial=0.0,
            friction=0.0,
            mass=0.02,
        ),
    ),
)

ROLLER_COVER = part(
    "roller_cover",
    (COVER_X0, COVER_Y0, COVER_Z0),
    boxes=COVER_BOXES,
    fixed=False,
    mass=0.085,
    friction=0.55,
)

OBJECTS = [ROLLER_FRAME, ROLLER_COVER]
INSTRUCTION = "Slide the paint-roller cover all the way onto the roller frame's cage until it seats past the retaining clip."

_METRIC = Metric(globals())
_step = 0
_DEBUG = os.environ.get("ROLLER_DEBUG") == "1"
_PROBE = os.environ.get("ROLLER_PROBE", "")

# Live camera-visible frame pose updated EVERY control step, plus grasp/rim offsets latched at steps 100 and 214:
_LIVE_FRM: dict | None = None
_SNAP1: dict | None = None
_SNAP2: dict | None = None


def _align_axis_rot(u_from: np.ndarray, u_to: np.ndarray) -> np.ndarray:
    a = np.asarray(u_from, float)
    b = np.asarray(u_to, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-8:
        return np.eye(3) if c > 0.0 else rx(math.pi)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def make_frame(gx, gz) -> np.ndarray:
    gx = np.asarray(gx, float)
    gz = np.asarray(gz, float)
    gz = gz / np.linalg.norm(gz)
    gx = gx - np.dot(gx, gz) * gz
    gx = gx / np.linalg.norm(gx)
    gy = np.cross(gz, gx)
    return np.column_stack([gx, gy, gz])


def _jacobian_6x6(q: np.ndarray, base_pos: np.ndarray) -> np.ndarray:
    eps = 1e-5
    p0, R0 = ik.tcp_pose(q, base_pos)
    J = np.zeros((6, 6))
    for j in range(6):
        qp = q.copy()
        qp[j] += eps
        pp, Rp = ik.tcp_pose(qp, base_pos)
        J[:3, j] = (pp - p0) / eps
        dR = Rp @ R0.T
        J[3:, j] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) * (0.5 / eps)
    return J


def _inverse_pd_pose(p_eq: np.ndarray, R_eq: np.ndarray, base_pos: np.ndarray, wrench_hold: np.ndarray):
    q_eq, _, _ = ik.solve(p_eq, ik.matrix_to_quat_xyzw(R_eq), np.array(READY, float), base_pos=base_pos, iters=80)
    J = _jacobian_6x6(q_eq, base_pos)
    K = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3, float)
    q_cmd = q_eq + (J.T @ wrench_hold) / K
    return ik.tcp_pose(q_cmd, base_pos)


class Orient(float):
    def __new__(cls, r):
        v = float.__new__(cls, 0.0)
        v._rot = r
        return v

    @property
    def rot(self) -> np.ndarray:
        return np.asarray(self._rot(), float) if callable(self._rot) else np.asarray(self._rot, float)


class SeenPos:
    def __init__(self, fn):
        self._fn = fn

    def __array__(self, dtype=None):
        arr = np.asarray(self._fn(), float)
        return arr.astype(dtype) if dtype is not None else arr


_old_grip = dm.grip_rotation
dm.grip_rotation = lambda yaw=0.0: yaw.rot if hasattr(yaw, "rot") else _old_grip(yaw)


def wp(dur: float, pos, finger: float, label: str, rot) -> Waypoint:
    p_arg = pos if isinstance(pos, SeenPos) else tuple(float(v) for v in pos)
    return Waypoint(dur, p_arg, float(finger), label, Orient(rot))


Q_L_PARK = np.array(READY, float) + np.array([0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
P_L_PARK, ROT_L_PARK = ik.tcp_pose(Q_L_PARK, np.array(LEFT_BASE_POS, float))
Q_R_PARK = np.array(READY, float) + np.array([-0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
P_R_PARK, ROT_R_PARK = ik.tcp_pose(Q_R_PARK, np.array(RIGHT_BASE_POS, float))

# -- Two-bolster benchtop roll-pin brace (Left Hand pins BOTH handle bolsters onto TOP while rolling -11 deg about Y) --
# Rolling -11.0 deg about Y around the bolster +X bottom edge (-0.117 m, y, -0.020 m) on TOP keeps BOTH bolsters
# (128 mm apart along Y) flat on TOP with zero pitch while raising the horizontal cage axis to TOP + 42.0 mm!
ROLL_BRACE_RAD = -math.radians(11.0)
R_FRM_NOM = ry(ROLL_BRACE_RAD)
_BOLSTER_PIV_LOC = np.array([-0.1170, 0.0, -0.0200])
_BOLSTER_PIV_W = np.array([FRAME_X0 - 0.1170, FRAME_Y0, TOP])
P_FRM_NOM = _BOLSTER_PIV_W - R_FRM_NOM @ _BOLSTER_PIV_LOC  # ~ [0.3692, +0.1150, TOP + 0.0420]
Z_CAGE_LIFTED = float(P_FRM_NOM[2])

ROT_HOLD_L = make_frame([1.0, 0.0, 0.0], [+0.22, -0.03, -0.975])
PAD_TCP_L = 0.0020
# Grasp centered between front (+0.062) and rear (+0.190) bolsters, upper half (+0.011 m) so finger hulls stay >10 mm above TOP:
HANDLE_GRASP_LOCAL = np.array([-0.1350, +0.1260, +0.0110])
HANDLE_GRASP_W = np.array([FRAME_X0, FRAME_Y0, FRAME_Z0]) + HANDLE_GRASP_LOCAL

P_L_PRE = HANDLE_GRASP_W - 0.065 * ROT_HOLD_L[:, 2]
P_L_GRASP = HANDLE_GRASP_W - PAD_TCP_L * ROT_HOLD_L[:, 2]

# Geometric equilibrium TCP at -11 deg roll + downward clamping press (-4.5 mm into TOP compliance) + -2.0 deg roll preload:
_P_L_EQ = P_FRM_NOM + R_FRM_NOM @ HANDLE_GRASP_LOCAL - PAD_TCP_L * (R_FRM_NOM @ ROT_HOLD_L)[:, 2]
_R_L_EQ = R_FRM_NOM @ ROT_HOLD_L
P_L_LIFT = _P_L_EQ + np.array([0.0, 0.0, -0.0045])
ROT_LIFT_L = ry(-math.radians(2.0)) @ _R_L_EQ

P_L_SET = np.array([FRAME_X0, FRAME_Y0, TOP + COVER_R_OUT]) + HANDLE_GRASP_LOCAL - PAD_TCP_L * ROT_HOLD_L[:, 2]
P_L_REL = P_L_SET - 0.075 * ROT_HOLD_L[:, 2]

F_OPEN_HANDLE = -0.034
F_CLAMP_HANDLE = -0.0020

# -- Right Hand Stage 1: grasp 57.2 mm OD cover, settle 16 mm in front of cage nose, then track live cage axis at 20 Hz --
ROT_CARRY_R = make_frame([1.0, 0.0, 0.0], [+0.28, +0.06, -0.958])
PAD_TCP_R = 0.0040
GRASP_DY_COV = -0.0620
GRASP_DZ_COV = +0.0040
COV_GRASP_W = np.array([COVER_X0, COVER_Y0 + GRASP_DY_COV, COVER_Z0 + GRASP_DZ_COV])

P_R_PRE = COV_GRASP_W - 0.070 * ROT_CARRY_R[:, 2]
P_R_GRASP = COV_GRASP_W - PAD_TCP_R * ROT_CARRY_R[:, 2]

# Nominal pre-nose cover center (16 mm outboard of the horizontally braced cage nose) + 6D inverse-PD compensation for 0.085 kg cover:
Y_NOSE_WORLD = float(P_FRM_NOM[1] - CAGE_LEN)
Y_COV_CTR_PRE = Y_NOSE_WORLD - 0.0160 - 0.5 * COVER_LEN
Y_COV_CTR_THREAD = Y_NOSE_WORLD + 0.1520 - 0.5 * COVER_LEN

_P_COV_PRENOSE = P_FRM_NOM + np.array([0.0, -CAGE_LEN - 0.0160 - 0.5 * COVER_LEN, 0.0])
_P_R_PRENOSE_EQ = _P_COV_PRENOSE + np.array([0.0, GRASP_DY_COV, GRASP_DZ_COV]) - PAD_TCP_R * ROT_CARRY_R[:, 2]
_F_GRAV_COV = np.array([0.0, 0.0, -0.085 * 9.81])
_WRENCH_HOLD_R = -np.r_[_F_GRAV_COV, np.cross(_P_COV_PRENOSE - _P_R_PRENOSE_EQ, _F_GRAV_COV)]
P_R_PRENOSE, ROT_PRENOSE_R = _inverse_pd_pose(_P_R_PRENOSE_EQ, ROT_CARRY_R, np.array(RIGHT_BASE_POS, float), _WRENCH_HOLD_R)

F_OPEN_COVER = FINGER_OPEN
F_CLAMP_COVER = -0.0175


def _live_frame_pose():
    if _LIVE_FRM is not None:
        return _LIVE_FRM["p_f"], _LIVE_FRM["R_f"]
    return P_FRM_NOM, R_FRM_NOM


def _stage1_target(axial_s_from_nose: float, lift_clear: float = 0.0):
    """Compute (p_cmd, R_cmd) placing ROLLER_COVER coaxially along the LIVE seen ROLLER_FRAME cage axis."""
    p_f, R_f = _live_frame_pose()
    if _SNAP1 is not None:
        R_cov_from_cmd = _SNAP1["R_cov_from_cmd"]
        p_cov_from_cmd = _SNAP1["p_cov_from_cmd"]
        R_v0 = _SNAP1["R_v"]
    else:
        R_v0 = np.eye(3)
        R_cov_from_cmd = ROT_PRENOSE_R.T @ R_v0
        p_cov_from_cmd = ROT_PRENOSE_R.T @ (_P_COV_PRENOSE - P_R_PRENOSE)

    u_cage = R_f[:, 1]  # points toward inboard shoulder (+Y_frame)
    u_cov0 = R_v0[:, 1]
    if float(np.dot(u_cov0, u_cage)) < 0.0:
        u_cov0 = -u_cov0
    R_align = _align_axis_rot(u_cov0, u_cage)
    R_cov_des = R_align @ R_v0

    p_nose = p_f - CAGE_LEN * u_cage
    p_cov_des = p_nose + (axial_s_from_nose - 0.5 * COVER_LEN) * u_cage

    R_cmd = R_cov_des @ R_cov_from_cmd.T
    p_cmd = p_cov_des - R_cmd @ p_cov_from_cmd
    if lift_clear != 0.0:
        p_cmd = p_cmd - lift_clear * R_cmd[:, 2]
    return p_cmd, R_cmd


# -- Right Hand Stage 2: 2-finger symmetric annular end-face push along LIVE +u_cage past 14-deg quick-release ramp --
ROT_PUSH_BASE = make_frame([1.0, 0.0, 0.0], [+0.25, +0.88, -0.405])
TIP_PUSH_OFFSET = 0.0120
F_PUSH_RIM = -0.0220


def _stage2_push_target(mode_tag: str):
    p_f, R_f = _live_frame_pose()
    out_s = _SNAP2["out_s"] if _SNAP2 is not None else (-CAGE_LEN + 0.1520 - COVER_LEN)

    u_cage = R_f[:, 1]
    R_push = _align_axis_rot(np.array([0.0, 1.0, 0.0]), u_cage) @ ROT_PUSH_BASE

    if mode_tag == "pre":
        p_rim = p_f + (out_s - 0.032) * u_cage
    elif mode_tag == "touch":
        p_rim = p_f + (out_s + 0.002) * u_cage
    elif mode_tag == "mid":
        p_rim = p_f + (-COVER_LEN - 0.036) * u_cage
    elif mode_tag == "home":
        # Push outboard rim to -COVER_LEN + 6 mm compliant follow-through so inboard rim snaps past clip flush to shoulder:
        p_rim = p_f + (-COVER_LEN + 0.0060) * u_cage
    else:  # "back"
        p_rim = p_f + (-COVER_LEN - 0.065) * u_cage + np.array([-0.030, 0.0, +0.035])

    return p_rim - TIP_PUSH_OFFSET * R_push[:, 2], R_push


@on_reset
def _reset(env):
    global _step, _LIVE_FRM, _SNAP1, _SNAP2
    _step = 0
    _LIVE_FRM = None
    _SNAP1 = None
    _SNAP2 = None
    _METRIC.reset()


@on_step
def _update(env):
    global _step, _LIVE_FRM, _SNAP1, _SNAP2
    _step += 1
    frm_s = labkit._part_state(env, ROLLER_FRAME)
    cov_s = labkit._part_state(env, ROLLER_COVER)
    _METRIC.update(env, frm_s, cov_s)

    if _PROBE != "unbraced_push":
        p_f = np.asarray(frm_s.pos, float).copy()
        R_f = frm_s.rotation().copy()
        _LIVE_FRM = dict(p_f=p_f, R_f=R_f)

        if _step == 100 and _SNAP1 is None:
            p_v = np.asarray(cov_s.pos, float).copy()
            R_v = cov_s.rotation().copy()
            _SNAP1 = dict(
                R_v=R_v,
                R_cov_from_cmd=ROT_PRENOSE_R.T @ R_v,
                p_cov_from_cmd=ROT_PRENOSE_R.T @ (p_v - P_R_PRENOSE),
            )
            if _DEBUG:
                print(
                    "SNAP1 "
                    + json.dumps(
                        dict(
                            p_f=[round(float(v), 4) for v in p_f],
                            u_cage=[round(float(v), 4) for v in R_f[:, 1]],
                            p_v=[round(float(v), 4) for v in p_v],
                        )
                    ),
                    flush=True,
                )

        if _PROBE == "" and _step == 214 and _SNAP2 is None:
            p_v = np.asarray(cov_s.pos, float).copy()
            R_v = cov_s.rotation().copy()
            u_cage = R_f[:, 1]
            u_cov = R_v[:, 1]
            ea = float(np.dot((p_v + 0.5 * COVER_LEN * u_cov) - p_f, u_cage))
            eb = float(np.dot((p_v - 0.5 * COVER_LEN * u_cov) - p_f, u_cage))
            _SNAP2 = dict(out_s=min(ea, eb))
            if _DEBUG:
                print("SNAP2 " + json.dumps(dict(out_s_mm=round(_SNAP2["out_s"] * 1000.0, 2))), flush=True)

    if _DEBUG and (_step % 10 == 0 or _step <= 5):
        print(
            "ROLDBG "
            + json.dumps(
                dict(
                    step=_step,
                    t=round(_step / 20.0, 2),
                    frm_pos=[round(float(v), 4) for v in frm_s.pos],
                    cov_pos=[round(float(v), 4) for v in cov_s.pos],
                    **_METRIC.detail,
                )
            ),
            flush=True,
        )


def score(objects) -> float:
    return _METRIC.score()


def success(objects) -> bool:
    return score(objects) == 1.0


def demo() -> dict[str, list[Waypoint]]:
    if _PROBE == "unbraced_push":
        p_low_align = np.array([FRAME_X0, Y_COV_CTR_PRE + GRASP_DY_COV, COVER_Z0 + 0.004]) - PAD_TCP_R * ROT_CARRY_R[:, 2]
        p_low_push = np.array([FRAME_X0, Y_COV_CTR_THREAD + GRASP_DY_COV, COVER_Z0 + 0.004]) - PAD_TCP_R * ROT_CARRY_R[:, 2]
        p_low_rel = p_low_push - 0.070 * ROT_CARRY_R[:, 2]
        return {
            "left": [wp(14.5, P_L_PARK, 0.20, "park-l", ROT_L_PARK)],
            "right": [
                wp(1.5, P_R_PRE, F_OPEN_COVER, "above-cover", ROT_CARRY_R),
                wp(0.9, P_R_GRASP, F_OPEN_COVER, "descend-cover", ROT_CARRY_R),
                wp(1.0, P_R_GRASP, F_CLAMP_COVER, "clamp-cover", ROT_CARRY_R),
                wp(1.6, p_low_align, F_CLAMP_COVER, "low-align", ROT_CARRY_R),
                wp(2.2, p_low_push, F_CLAMP_COVER, "shove-unbraced", ROT_CARRY_R),
                wp(0.8, p_low_push, F_OPEN_COVER, "release", ROT_CARRY_R),
                wp(1.2, p_low_rel, F_OPEN_COVER, "lift-away", ROT_CARRY_R),
                wp(1.8, P_R_PARK, 0.20, "retract-r", ROT_R_PARK),
                wp(3.5, P_R_PARK, 0.20, "hold-r", ROT_R_PARK),
            ],
        }

    s1_align_p = SeenPos(lambda: _stage1_target(-0.0080)[0])
    s1_align_r = lambda: _stage1_target(-0.0080)[1]
    s1_entry_p = SeenPos(lambda: _stage1_target(+0.0250)[0])
    s1_entry_r = lambda: _stage1_target(+0.0250)[1]
    s1_f65_p = SeenPos(lambda: _stage1_target(+0.0650)[0])
    s1_f65_r = lambda: _stage1_target(+0.0650)[1]
    s1_f110_p = SeenPos(lambda: _stage1_target(+0.1100)[0])
    s1_f110_r = lambda: _stage1_target(+0.1100)[1]
    s1_thread_p = SeenPos(lambda: _stage1_target(+0.1520)[0])
    s1_thread_r = lambda: _stage1_target(+0.1520)[1]
    s1_clear_p = SeenPos(lambda: _stage1_target(+0.1520, lift_clear=0.068)[0])

    if _PROBE == "half_seated":
        return {
            "left": [
                wp(1.5, P_L_PRE, F_OPEN_HANDLE, "above-handle", ROT_HOLD_L),
                wp(0.9, P_L_GRASP, F_OPEN_HANDLE, "descend-handle", ROT_HOLD_L),
                wp(1.0, P_L_GRASP, F_CLAMP_HANDLE, "clamp-handle", ROT_HOLD_L),
                wp(1.2, P_L_LIFT, F_CLAMP_HANDLE, "roll-pin-bench", ROT_LIFT_L),
                wp(6.6, P_L_LIFT, F_CLAMP_HANDLE, "hold-brace", ROT_LIFT_L),
                wp(1.2, P_L_SET, F_CLAMP_HANDLE, "lower-bench", ROT_HOLD_L),
                wp(0.8, P_L_SET, F_OPEN_HANDLE, "release-handle", ROT_HOLD_L),
                wp(1.1, P_L_REL, F_OPEN_HANDLE, "clear-handle", ROT_HOLD_L),
                wp(1.6, P_L_PARK, 0.20, "retract-l", ROT_L_PARK),
                wp(2.2, P_L_PARK, 0.20, "hold-l", ROT_L_PARK),
            ],
            "right": [
                wp(1.5, P_R_PRE, F_OPEN_COVER, "above-cover", ROT_CARRY_R),
                wp(0.9, P_R_GRASP, F_OPEN_COVER, "descend-cover", ROT_CARRY_R),
                wp(1.0, P_R_GRASP, F_CLAMP_COVER, "clamp-cover", ROT_CARRY_R),
                wp(1.2, P_R_PRENOSE, F_CLAMP_COVER, "carry-prenose", ROT_PRENOSE_R),
                wp(0.5, P_R_PRENOSE, F_CLAMP_COVER, "snap1-dwell", ROT_PRENOSE_R),
                wp(0.9, s1_align_p, F_CLAMP_COVER, "align-nose", s1_align_r),
                wp(0.8, s1_entry_p, F_CLAMP_COVER, "enter-nose", s1_entry_r),
                wp(0.7, s1_f65_p, F_CLAMP_COVER, "feed-65mm", s1_f65_r),
                wp(0.7, s1_f110_p, F_CLAMP_COVER, "feed-110mm", s1_f110_r),
                wp(0.8, s1_thread_p, F_CLAMP_COVER, "feed-152mm", s1_thread_r),
                wp(0.8, s1_thread_p, F_OPEN_COVER, "release-cover", s1_thread_r),
                wp(1.1, s1_clear_p, F_OPEN_COVER, "clear-cover", s1_thread_r),
                wp(1.8, P_R_PARK, 0.20, "retract-r", ROT_R_PARK),
                wp(5.4, P_R_PARK, 0.20, "hold-r", ROT_R_PARK),
            ],
        }

    s2_pre_p = SeenPos(lambda: _stage2_push_target("pre")[0])
    s2_pre_r = lambda: _stage2_push_target("pre")[1]
    s2_touch_p = SeenPos(lambda: _stage2_push_target("touch")[0])
    s2_touch_r = lambda: _stage2_push_target("touch")[1]
    s2_mid_p = SeenPos(lambda: _stage2_push_target("mid")[0])
    s2_mid_r = lambda: _stage2_push_target("mid")[1]
    s2_home_p = SeenPos(lambda: _stage2_push_target("home")[0])
    s2_home_r = lambda: _stage2_push_target("home")[1]
    s2_back_p = SeenPos(lambda: _stage2_push_target("back")[0])
    s2_back_r = lambda: _stage2_push_target("back")[1]

    # Exact timing:
    # t = 4.60 s (step 92): Left finishes roll-pin-bench & Right finishes carry-prenose;
    # t = 4.60..5.10 s: snap1-dwell settles both arms and latches _SNAP1 at step 100 (t = 5.00 s)!
    # t = 5.10..9.00 s: continuous 20 Hz _LIVE_FRM coaxial feed (align-nose -> enter-nose -> 65 -> 110 -> 152 mm)
    # t = 9.00..10.90 s: release-cover [9.0..9.8] -> clear-cover [9.8..10.9]; step 214 (t = 10.70 s) latches _SNAP2!
    # t = 10.90..15.90 s: continuous 20 Hz _LIVE_FRM annular rim push past 14-deg retaining ramp flush to shoulder!
    return {
        "left": [
            wp(1.5, P_L_PRE, F_OPEN_HANDLE, "above-handle", ROT_HOLD_L),
            wp(0.9, P_L_GRASP, F_OPEN_HANDLE, "descend-handle", ROT_HOLD_L),
            wp(1.0, P_L_GRASP, F_CLAMP_HANDLE, "clamp-handle", ROT_HOLD_L),
            wp(1.2, P_L_LIFT, F_CLAMP_HANDLE, "roll-pin-bench", ROT_LIFT_L),
            wp(11.6, P_L_LIFT, F_CLAMP_HANDLE, "brace-frame", ROT_LIFT_L),
            wp(1.2, P_L_SET, F_CLAMP_HANDLE, "lower-assembled", ROT_HOLD_L),
            wp(0.8, P_L_SET, F_OPEN_HANDLE, "release-handle", ROT_HOLD_L),
            wp(1.1, P_L_REL, F_OPEN_HANDLE, "clear-handle", ROT_HOLD_L),
            wp(1.6, P_L_PARK, 0.20, "retract-l", ROT_L_PARK),
            wp(2.2, P_L_PARK, 0.20, "hold-l", ROT_L_PARK),
        ],
        "right": [
            wp(1.5, P_R_PRE, F_OPEN_COVER, "above-cover", ROT_CARRY_R),
            wp(0.9, P_R_GRASP, F_OPEN_COVER, "descend-cover", ROT_CARRY_R),
            wp(1.0, P_R_GRASP, F_CLAMP_COVER, "clamp-cover", ROT_CARRY_R),
            wp(1.2, P_R_PRENOSE, F_CLAMP_COVER, "carry-prenose", ROT_PRENOSE_R),
            wp(0.5, P_R_PRENOSE, F_CLAMP_COVER, "snap1-dwell", ROT_PRENOSE_R),
            wp(0.9, s1_align_p, F_CLAMP_COVER, "align-nose", s1_align_r),
            wp(0.8, s1_entry_p, F_CLAMP_COVER, "enter-nose", s1_entry_r),
            wp(0.7, s1_f65_p, F_CLAMP_COVER, "feed-65mm", s1_f65_r),
            wp(0.7, s1_f110_p, F_CLAMP_COVER, "feed-110mm", s1_f110_r),
            wp(0.8, s1_thread_p, F_CLAMP_COVER, "feed-152mm", s1_thread_r),
            wp(0.8, s1_thread_p, F_OPEN_COVER, "release-cover", s1_thread_r),
            wp(1.1, s1_clear_p, F_OPEN_COVER, "clear-cover", s1_thread_r),
            wp(1.1, s2_pre_p, F_PUSH_RIM, "stage-rim-push", s2_pre_r),
            wp(0.6, s2_touch_p, F_PUSH_RIM, "touch-rim", s2_touch_r),
            wp(0.9, s2_mid_p, F_PUSH_RIM, "push-to-clip", s2_mid_r),
            wp(1.2, s2_home_p, F_PUSH_RIM, "push-past-clip", s2_home_r),
            wp(0.4, s2_home_p, F_PUSH_RIM, "dwell-shoulder", s2_home_r),
            wp(0.8, s2_back_p, F_PUSH_RIM, "step-back-rim", s2_back_r),
            wp(1.6, P_R_PARK, 0.20, "retract-r", ROT_R_PARK),
            wp(5.6, P_R_PARK, 0.20, "hold-r", ROT_R_PARK),
        ],
    }
