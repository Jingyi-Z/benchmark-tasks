"""Load a reusable basket filter into a swivel coffee-maker brew basket, fold its bail handle flat, and close the holder.

Modeled 1:1 on a Hamilton Beach 46310 12-cup programmable drip coffee maker (327.9 x 201.9 x 353.1 mm;
front-hinged Swivel Filter Basket Holder carrying the seated Removable Filter Basket beneath the upper
brew-head showerhead overhang, with the 12-cup carafe on the lower warming plate) and a Cafe Brew BF215
8-12 cup gold-tone permanent basket coffee filter (109.2 mm top OD, 84.0 mm base OD, 53.3 mm height)
with a hinged semicircular bail handle that must fold flat inside the filter rim to clear the brew head.
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
from task_metrics import Metric, rx, ry, rz
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS, SHOULDER, WRIST
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 960
SUCCESS_HOLD_S = 2.5
labkit.CONTACT_OFFSET = 0.0005

# Joint-axis static/dynamic friction (N*m) for realistic swivel-hinge and wire-bail retention:
labkit.AXIS_FRICTION = {
    "swivel_holder": (0.14, 0.11),
    "bail": (0.028, 0.022),
}

# -- 1:1 Hamilton Beach 46310 + Cafe Brew BF215 geometry constants ----------------------------
BREWER_X0 = 0.4350
BREWER_Y0 = -0.0550
BREWER_Z0 = TOP

FILTER_X0 = 0.2650
FILTER_Y0 = -0.2200
FILTER_Z0 = TOP
FILTER_YAW0 = math.radians(-164.0)
FILTER_SEAT_YAW = math.radians(-148.0)

# Swivel hinge on front-left (+Y) pillar; closed basket centered at (-0.080, 0.000) in brewer frame:
SWIVEL_ANCHOR = (-0.0880, +0.0900, 0.2460)
BASKET_CENTER_B = (-0.0800, 0.0000)
Z_SEAT_FLOOR = 0.2280      # internal support floor of Removable Filter Basket
FILTER_HEIGHT = 0.0533     # Cafe Brew BF215 height (53.3 mm -> closed rim at z_b = 0.2813 m)
H_GAP = 0.0120             # 12.0 mm brew-head showerhead clearance above closed filter rim
Z_HEAD_BOTTOM = Z_SEAT_FLOOR + FILTER_HEIGHT + H_GAP  # 0.2933 m exact

# Bail hinge recessed 3.5 mm below filter rim (0.0533 - 0.0035 = 0.0498 m), rotating about +X:
BAIL_ANCHOR = (0.0, 0.0, 0.0498)
R_BAIL = 0.0480

# Colors:
C_BLACK = (0.13, 0.14, 0.16)
C_CHARCOAL = (0.19, 0.20, 0.22)
C_BASKET_INNER = (0.24, 0.25, 0.28)
C_STEEL = (0.76, 0.78, 0.81)
C_GLASS = (0.28, 0.34, 0.38)
C_GOLD_MESH = (0.76, 0.63, 0.28)
C_PLATE = (0.22, 0.22, 0.24)

# -- 1. COFFEE_MAKER static housing + seated carafe + upper brew-head overhang ----------------
BREWER_BOXES = (
    # Lower base platform (327.9 x 201.9 x 22.0 mm):
    ((0.0000, 0.0000, 0.0110), (0.3279, 0.2019, 0.0220), C_BLACK, 0.0),
    # Rear water-reservoir column (x_b in [+0.0100, +0.16395], full 353.1 mm height):
    ((+0.0870, 0.0000, 0.18755), (0.1539, 0.2019, 0.3311), C_BLACK, 0.0),
    # Brushed stainless rear-column side accents & control-panel fascia:
    ((+0.0090, 0.0000, 0.3232), (0.0040, 0.1880, 0.0560), C_STEEL, 0.0),
    ((-0.1620, 0.0000, 0.3232), (0.0040, 0.0700, 0.0480), C_STEEL, 0.0),
    # Left-side (+Y) lower hinge support column (z_b in [0.0220, 0.2020], clear of swivel barrel [0.214, 0.278]):
    ((-0.0440, +0.0910, 0.1120), (0.1080, 0.0190, 0.1800), C_BLACK, 0.0),
    # Lower clevis ear under swivel hinge (z_b in [0.1960, 0.2060]):
    ((-0.0880, +0.0900, 0.2010), (0.0260, 0.0220, 0.0100), C_CHARCOAL, 0.0),
    # Carafe handle on -Y side of lower bay (z_b in [0.052, 0.158]):
    ((-0.0780, -0.0820, 0.1050), (0.0200, 0.0300, 0.1060), C_CHARCOAL, 0.0),
    # Upper brew-head rectangular bridge (z_b in [0.2933, 0.3531] — bottom flush at exact Z_HEAD_BOTTOM = 0.2933 m):
    ((-0.0375, 0.0000, 0.3232), (0.0950, 0.1840, 0.0598), C_BLACK, 0.0),
    # Upper clevis ear above swivel hinge (z_b in [0.2933, 0.3531]):
    ((-0.0880, +0.0900, 0.3232), (0.0280, 0.0220, 0.0598), C_BLACK, 0.0),
)

BREWER_CYLS = (
    # Keep-warm hotplate disk (z_b in [0.0220, 0.0280]):
    ((-0.0780, 0.0000, 0.0250), 0.0720, 0.0060, C_PLATE, "Z"),
    # Seated 12-cup glass carafe body & lid in lower bay (z_b in [0.0280, 0.1820]):
    ((-0.0780, 0.0000, 0.1000), 0.0680, 0.1440, C_GLASS, "Z"),
    ((-0.0780, 0.0000, 0.1770), 0.0660, 0.0100, C_CHARCOAL, "Z"),
    # Upper brew-head circular showerhead hood (z_b in [0.2933, 0.3531] — bottom flush at exact Z_HEAD_BOTTOM = 0.2933 m):
    ((-0.0800, 0.0000, 0.3232), 0.0840, 0.0598, C_BLACK, "Z"),
)

# -- 2. SWIVEL_JOINT: Outer Swivel Filter Basket Holder + Seated Removable Filter Basket ------
# Outer Swivel Filter Basket Holder: 16 vertical staves centered at (-0.080, 0.000), ID=122.0 mm, OD=130.0 mm:
_N_STAVES = 16
_HOLDER_STAVES = tuple(
    (
        (
            BASKET_CENTER_B[0] + 0.0630 * math.cos(2.0 * math.pi * k / _N_STAVES),
            BASKET_CENTER_B[1] + 0.0630 * math.sin(2.0 * math.pi * k / _N_STAVES),
            0.2420,
        ),
        (0.0040, 0.0228, 0.0740),  # z_b in [0.2050, 0.2790]
        C_STEEL if k in (6, 7, 8, 9, 10) else C_BLACK,
        2.0 * math.pi * k / _N_STAVES,
    )
    for k in range(_N_STAVES)
)

# Seated Removable Filter Basket (Manual 840263902 pp. 3-5): 16 tapered staves (13.3 deg outward top flare),
# seat floor at exact Z_SEAT_FLOOR = 0.2280 m (ID = 89.6 mm) flaring to ID = 116.6 mm at rim z_b = 0.2850 m:
_TAPER_RAD = math.radians(13.30)
_INNER_BASKET_STAVES = tuple(
    (
        (
            BASKET_CENTER_B[0] + 0.05285 * math.cos(2.0 * math.pi * k / _N_STAVES),
            BASKET_CENTER_B[1] + 0.05285 * math.sin(2.0 * math.pi * k / _N_STAVES),
            0.2565,
        ),
        (0.0026, 0.0185, 0.0584),
        C_BASKET_INNER,
        (2.0 * math.pi * k / _N_STAVES, +_TAPER_RAD),
    )
    for k in range(_N_STAVES)
)

# Pull-tab center in brewer rest frame (rounded 32 mm horizontal finger ear, 14 mm thick in Z, 61.3 mm below brew head):
PULL_TAB_B = np.array([-0.1680, -0.0320, 0.2320])

SWIVEL_BOXES = (
    _HOLDER_STAVES
    + _INNER_BASKET_STAVES
    + (
        # Hinge arm connecting outer shell (+Y) to SWIVEL_ANCHOR (-0.088, +0.090, 0.246):
        ((-0.0860, +0.0765, 0.2460), (0.0180, 0.0270, 0.0560), C_BLACK, 0.0),
        # Neck connecting outer shell (-0.139, -0.025) to rounded horizontal finger tab (-0.168, -0.032, 0.232):
        ((-0.1510, -0.0285, 0.2320), (0.0280, 0.0240, 0.0140), C_BLACK, 0.0),
        # Two upper rim alignment tabs of Removable Filter Basket seated in Swivel Holder rim keyways (+-Y):
        ((-0.0800, +0.0615, 0.2820), (0.0130, 0.0065, 0.0055), C_BASKET_INNER, 0.0),
        ((-0.0800, -0.0615, 0.2820), (0.0130, 0.0065, 0.0055), C_BASKET_INNER, 0.0),
    )
)

SWIVEL_CYLS = (
    # Hinge barrel at SWIVEL_ANCHOR (z_b in [0.2140, 0.2780]):
    (SWIVEL_ANCHOR, 0.0105, 0.0640, C_BLACK, "Z"),
    # Rounded horizontal pull-tab disk (32.0 mm diameter, 14.0 mm Z thickness at z_b = 0.239..0.253):
    (tuple(PULL_TAB_B), 0.0160, 0.0140, C_CHARCOAL, "Z"),
    # Outer Swivel Holder lower support ring (z_b in [0.2050, 0.2100]):
    ((-0.0800, 0.0000, 0.2075), 0.0520, 0.0050, C_BLACK, "Z"),
    # Seated Removable Filter Basket internal support floor (top face at exact Z_SEAT_FLOOR = 0.2280 m):
    ((-0.0800, 0.0000, 0.2250), 0.0455, 0.0060, C_BASKET_INNER, "Z"),
    # Removable Filter Basket pause-and-serve bottom drip-stop valve boss (z_b in [0.2050, 0.2220]):
    ((-0.0800, 0.0000, 0.2135), 0.0110, 0.0170, C_BASKET_INNER, "Z"),
)

SWIVEL_JOINT = Joint(
    name="swivel_holder",
    kind="hinge",
    axis="Z",
    anchor=SWIVEL_ANCHOR,
    boxes=SWIVEL_BOXES,
    cylinders=SWIVEL_CYLS,
    lower=-2.06,   # -118.0 deg full left-outward service swing
    upper=0.0,     # 0.0 deg closed flush in brewer bay
    stiffness=0.0,
    damping=0.05,
    target=0.0,
    friction=0.08,
    initial=0.0,
    mass=0.195,
)

COFFEE_MAKER = part(
    "coffee_maker",
    pos=(BREWER_X0, BREWER_Y0, BREWER_Z0),
    boxes=BREWER_BOXES,
    cylinders=BREWER_CYLS,
    joints=(SWIVEL_JOINT,),
    mass=3.65,
    friction=0.90,
    self_collide=False,
)

# -- 3. PERM_FILTER: Cafe Brew BF215 8-12 cup gold-tone reusable basket filter + folding bail -
# 16 tapered surgical-mesh staves (84.0 mm base OD -> 109.2 mm top OD, 53.3 mm height) + upper PP collar:
_FILTER_MESH_STAVES = tuple(
    (
        (
            0.0470 * math.cos(2.0 * math.pi * k / _N_STAVES),
            0.0470 * math.sin(2.0 * math.pi * k / _N_STAVES),
            0.0278,
        ),
        (0.0024, 0.0158, 0.0504),
        C_GOLD_MESH,
        (2.0 * math.pi * k / _N_STAVES, +_TAPER_RAD),
    )
    for k in range(_N_STAVES)
)

_FILTER_RIM_SEGS = tuple(
    (
        (
            0.0532 * math.cos(2.0 * math.pi * k / _N_STAVES),
            0.0532 * math.sin(2.0 * math.pi * k / _N_STAVES),
            0.0508,  # z_f in [0.0483, 0.0533] -> crisp rim at exact 53.3 mm, inner radius 51.8 mm > 48.0 mm bail
        ),
        (0.0028, 0.0195, 0.0050),
        C_CHARCOAL,
        2.0 * math.pi * k / _N_STAVES,
    )
    for k in range(_N_STAVES)
)

FILTER_BOXES = _FILTER_MESH_STAVES + _FILTER_RIM_SEGS + (
    # Molded PP rim pivot ears where the coaxial stainless wire bail axles enter the basket rim:
    ((-0.0528, 0.0000, 0.0508), (0.0052, 0.0096, 0.0050), C_CHARCOAL, 0.0),
    ((+0.0528, 0.0000, 0.0508), (0.0052, 0.0096, 0.0050), C_CHARCOAL, 0.0),
)
FILTER_CYLS = (
    # Flat bottom surgical-mesh + PP ring base (82.0 mm OD, z_f in [0.0000, 0.0044]):
    ((0.0, 0.0, 0.0022), 0.0410, 0.0044, C_CHARCOAL, "Z"),
)

# Hinged semicircular wire bail + finger-grip tab rotating about +X at BAIL_ANCHOR = (0, 0, 0.0498):
# At bail = 0.0 rad (folded flat), arch lies inside -Y half of rim (y_f in [-0.0480, 0.0], z_top = 0.0518 m = rim - 1.5 mm).
# At bail = -1.58 rad (-90.5 deg initial), arch stands upright to z_f = 0.0978 m (+44.5 mm above rim), leaning against lower=-1.62 rad.
BAIL_BOXES = (
    # Coaxial X-axis wire pivot axles connecting both bail legs directly into the basket rim pivot ears:
    ((-0.0440, 0.0000, 0.0498), (0.0196, 0.0040, 0.0040), C_STEEL, 0.0),
    ((+0.0440, 0.0000, 0.0498), (0.0196, 0.0040, 0.0040), C_STEEL, 0.0),
    ((-0.0360, -0.0220, 0.0498), (0.0040, 0.0440, 0.0040), C_STEEL, 0.0),
    ((+0.0360, -0.0220, 0.0498), (0.0040, 0.0440, 0.0040), C_STEEL, 0.0),
    ((-0.0280, -0.0445, 0.0498), (0.0180, 0.0040, 0.0040), C_STEEL, 0.0),
    ((+0.0280, -0.0445, 0.0498), (0.0180, 0.0040, 0.0040), C_STEEL, 0.0),
    ((0.0000, -0.0432, 0.0498), (0.0440, 0.0096, 0.0040), C_CHARCOAL, 0.0),
)

BAIL_INIT_RAD = -1.58  # -90.5 deg standing upright on counter

BAIL_JOINT = Joint(
    name="bail",
    kind="hinge",
    axis="X",
    anchor=BAIL_ANCHOR,
    boxes=BAIL_BOXES,
    cylinders=(),
    lower=-1.62,   # -92.8 deg upright ear stop
    upper=0.0,     # 0.0 deg folded flat inside rim
    stiffness=0.0,
    damping=0.015,
    target=BAIL_INIT_RAD,
    friction=0.020,
    initial=BAIL_INIT_RAD,
    mass=0.008,
)

PERM_FILTER = part(
    "permanent_filter",
    pos=(FILTER_X0, FILTER_Y0, FILTER_Z0),
    yaw=FILTER_YAW0,
    boxes=FILTER_BOXES,
    cylinders=FILTER_CYLS,
    joints=(BAIL_JOINT,),
    mass=0.042,
    friction=0.75,
    self_collide=False,
)

OBJECTS = [COFFEE_MAKER, PERM_FILTER]

_METRIC = Metric(globals())
_MODE = os.environ.get("COFFEE_FILTER_MODE", "demo").strip()
_DEBUG = os.environ.get("COFFEE_FILTER_DEBUG", "0") == "1"
_step = 0
_LIVE_BREWER: dict | None = None
_LIVE_FILTER: dict | None = None
_BAIL_SNAP: dict | None = None


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


# -- Safe parked READY poses (>= 130 mm clear of brewer and filter) ---------------------------
Q_L_PARK = np.array(READY, float) + np.array([+0.28, 0.0, 0.0, 0.0, 0.0, 0.0])
P_L_PARK, ROT_L_PARK = ik.tcp_pose(Q_L_PARK, np.array(LEFT_BASE_POS, float))
Q_R_PARK = np.array(READY, float) + np.array([-0.28, 0.0, 0.0, 0.0, 0.0, 0.0])
P_R_PARK, ROT_R_PARK = ik.tcp_pose(Q_R_PARK, np.array(RIGHT_BASE_POS, float))

# Left-arm clear wait pose while Right Arm seats the filter and folds its bail:
P_L_WAIT = np.array([0.1850, +0.2550, TOP + 0.2550])
ROT_L_WAIT = make_frame([0.0, 0.0, -1.0], [+0.88, -0.36, -0.31])

# -- Left Arm: vertical TB clamp on rounded horizontal pull tab along circular swivel arc -----
SWIVEL_OPEN_RAD = -1.94  # -111.2 deg open service angle
F_OPEN_TAB = -0.032
F_CLAMP_TAB = -0.0030    # 6 mm target gap across 14 mm-thick PP pull tab


def _brewer_pose_live():
    if _LIVE_BREWER is not None:
        return _LIVE_BREWER["p_b"], _LIVE_BREWER["R_b"], _LIVE_BREWER["swivel_rad"]
    return np.array([BREWER_X0, BREWER_Y0, BREWER_Z0], float), np.eye(3), 0.0


def _swivel_tab_target(theta_rad: float, radial_back: float = 0.0, dz: float = 0.0):
    """Compute Left-Arm (p_tcp, R_tcp) clamping the rounded 14 mm horizontal pull tab at swivel angle theta_rad."""
    p_b, R_b, _ = _brewer_pose_live()
    anc = np.asarray(SWIVEL_ANCHOR, float)
    v_tab = PULL_TAB_B - anc
    p_tab_b = anc + rz(theta_rad) @ v_tab
    p_tab_w = p_b + R_b @ p_tab_b
    p_bsk_b = anc + rz(theta_rad) @ (np.array([BASKET_CENTER_B[0], BASKET_CENTER_B[1], PULL_TAB_B[2]]) - anc)
    p_bsk_w = p_b + R_b @ p_bsk_b

    # Blend shoulder-to-tab ray with tab-to-basket-center inward ray so jaws stay >25 mm clear of outer shell staves:
    u_base = p_tab_w[:2] - np.array(LEFT_BASE_POS[:2], float)
    u_base = u_base / np.linalg.norm(u_base)
    u_rad = p_bsk_w[:2] - p_tab_w[:2]
    u_rad = u_rad / np.linalg.norm(u_rad)
    u_h = 0.55 * u_base + 0.45 * u_rad
    u_h = u_h / np.linalg.norm(u_h)
    gz = np.array([u_h[0], u_h[1], 0.0], float)
    R_tcp = make_frame([0.0, 0.0, -1.0], gz)

    # Place TCP 8 mm short of tab center so distal flat pads clamp the 32 mm tab cleanly (>24 mm clear of outer shell):
    p_tcp = p_tab_w - (0.0080 + radial_back) * R_tcp[:, 2] + np.array([0.0, 0.0, dz])
    return p_tcp, R_tcp


# -- Right Arm: (A) gentle downward press on flat brewer roof, (B) BF215 bail pickup/seat, (C) bail fold
_U_TOP_R = np.array([0.4300 - RIGHT_BASE_POS[0], -0.0650 - RIGHT_BASE_POS[1]], float)
_U_TOP_R = _U_TOP_R / np.linalg.norm(_U_TOP_R)
ROT_BRACE_R = make_frame([0.0, 0.0, 1.0], [_U_TOP_R[0], _U_TOP_R[1], -0.65])
F_CLOSED_PAD = 0.0


def _brewer_brace_target(dz: float = 0.0):
    # Closed fingertip lowest point sits 9.5 mm below TCP z; TOP + 0.3614 places pad 1.2 mm into the flat 1.1006 m roof (-Z press):
    p_tcp = np.array([0.4300, -0.0650, TOP + 0.3614 + dz], float)
    return p_tcp, ROT_BRACE_R


# Right-Arm local bail grasp frame (filter oriented at -164..-148 deg so bail folds away from showerhead hood;
# jaws pinch across +Y_f on 44.0 x 9.6 x 4.0 mm standing tab; gz points outward & -28.3 deg down):
R_BAIL_LOCAL = make_frame([0.0, +1.0, 0.0], [-0.88, -0.03, -0.474])
ROT_BAIL_CLEAR = rz(FILTER_YAW0) @ R_BAIL_LOCAL
PAD_TCP_BAIL = 0.0035
F_OPEN_BAIL = FINGER_OPEN
F_RELEASE_BAIL = -0.0220
F_CLAMP_BAIL = -0.0006


def _bail_tab_world_from_filter(p_f: np.ndarray, R_f: np.ndarray, bail_rad: float) -> np.ndarray:
    anc = np.asarray(BAIL_ANCHOR, float)
    # Grasp slightly toward +X_f (+6 mm on the 44 mm flat PP tab, away from the upper brew-head hood):
    tab_rest = np.array([+0.0060, -0.0432, 0.0498])
    tab_f = anc + rx(bail_rad) @ (tab_rest - anc)
    return p_f + R_f @ tab_f


# Precompute 50 g carry PD gravity-sag compensation at 4 azimuths across [-164..-148] deg (O(1) lookup at 20 Hz):
_F_GRAV_50G = np.array([0.0, 0.0, -0.050 * 9.81])
_WR_50G = -np.r_[_F_GRAV_50G, np.cross(np.array([0.0, 0.0, -0.090]), _F_GRAV_50G)]
_PD_TABLE = {}
for _deg_k in (-164, -158, -153, -148):
    _yaw_k = math.radians(_deg_k)
    _u_k = (_deg_k - (-164.0)) / 16.0
    _eq_k = (1.0 - _u_k) * np.array([0.2650, -0.2200, TOP + 0.3980]) + _u_k * np.array([0.2590, +0.0550, TOP + 0.3980])
    _R_eq_k = rz(_yaw_k) @ R_BAIL_LOCAL
    _p_cmd_k, _R_cmd_k = _inverse_pd_pose(_eq_k, _R_eq_k, np.array(RIGHT_BASE_POS, float), _WR_50G)
    _PD_TABLE[_deg_k] = (_p_cmd_k - _eq_k, _R_eq_k.T @ _R_cmd_k)


def _apply_carry_pd(p_eq: np.ndarray, R_eq: np.ndarray, yaw_rad: float):
    deg = max(-164.0, min(-148.0, math.degrees(yaw_rad)))
    best_k = min(_PD_TABLE.keys(), key=lambda k: abs(k - deg))
    dp, dR_loc = _PD_TABLE[best_k]
    return p_eq + dp, R_eq @ dR_loc


def _snap_or_default_filter():
    if _LIVE_FILTER is not None and _BAIL_SNAP is None:
        p_f = _LIVE_FILTER["p_f"]
        R_f = _LIVE_FILTER["R_f"]
        bail_rad = _LIVE_FILTER["bail_rad"]
    elif _BAIL_SNAP is not None:
        p_f = _BAIL_SNAP["p_f0"]
        R_f = _BAIL_SNAP["R_f0"]
        bail_rad = _BAIL_SNAP["bail_rad0"]
    else:
        p_f = np.array([FILTER_X0, FILTER_Y0, FILTER_Z0], float)
        R_f = rz(FILTER_YAW0)
        bail_rad = BAIL_INIT_RAD
    yaw0 = math.atan2(float(R_f[1, 0]), float(R_f[0, 0]))
    return p_f, R_f, bail_rad, yaw0


def _filter_pickup_target(mode_tag: str):
    p_f, R_f, bail_rad, yaw0 = _snap_or_default_filter()
    R_grasp = R_f @ R_BAIL_LOCAL
    p_tab = _bail_tab_world_from_filter(p_f, R_f, bail_rad)
    p_grasp = p_tab - PAD_TCP_BAIL * R_grasp[:, 2]
    if mode_tag == "pre":
        return p_grasp - 0.065 * R_grasp[:, 2], R_grasp
    if mode_tag == "grasp":
        return p_grasp, R_grasp
    # Lift straight up so filter bottom clears Z_BASKET_RIM = TOP + 0.2850 m by +19 mm (filter base at TOP + 0.304 m):
    dz_lift = (TOP + 0.3040) - float(p_f[2])
    return _apply_carry_pd(p_grasp + np.array([0.0, 0.0, dz_lift]), R_grasp, yaw0)


def _open_basket_seat_floor_world() -> np.ndarray:
    p_b, R_b, swivel_rad = _brewer_pose_live()
    use_swivel = swivel_rad if abs(swivel_rad) > 0.5 else SWIVEL_OPEN_RAD
    anc = np.asarray(SWIVEL_ANCHOR, float)
    R_loc = rz(use_swivel)
    R_sw = R_b @ R_loc
    p_sw = p_b + R_b @ (anc - R_loc @ anc)
    return p_sw + R_sw @ np.array([BASKET_CENTER_B[0], BASKET_CENTER_B[1], Z_SEAT_FLOOR])


def _tcp_minus_pf_local() -> np.ndarray:
    if _BAIL_SNAP is not None:
        return _BAIL_SNAP["tcp_minus_pf_local"]
    tab_0 = _bail_tab_world_from_filter(np.zeros(3), np.eye(3), BAIL_INIT_RAD)
    return tab_0 - PAD_TCP_BAIL * R_BAIL_LOCAL[:, 2]


def _filter_carry_target(stage_idx: int):
    """Two-corner -X/+Y aisle carry around upper showerhead hood (x=0.182 m, y=+0.088 m > +0.0836 m AABB bound)."""
    p_f0, _, _, yaw0 = _snap_or_default_filter()
    seat_floor_w = _open_basket_seat_floor_world()
    z_carry = float(seat_floor_w[2]) + 0.0780
    if stage_idx == 1:
        p_f_des = np.array([0.1820, -0.1650, z_carry])
        yaw_des = 0.65 * yaw0 + 0.35 * FILTER_SEAT_YAW
    else:
        p_f_des = np.array([0.1820, float(seat_floor_w[1]) + 0.0330, z_carry])
        yaw_des = 0.20 * yaw0 + 0.80 * FILTER_SEAT_YAW
    R_f_des = rz(yaw_des)
    R_tcp_eq = R_f_des @ R_BAIL_LOCAL
    p_tcp_eq = p_f_des + R_f_des @ _tcp_minus_pf_local()
    return _apply_carry_pd(p_tcp_eq, R_tcp_eq, yaw_des)


def _filter_seat_target(stage_tag: str):
    """Carry and seat PERM_FILTER concentrically onto the LIVE seen Removable Filter Basket floor."""
    seat_floor_w = _open_basket_seat_floor_world()
    R_f_seat = rz(FILTER_SEAT_YAW)
    R_tcp_seat = R_f_seat @ R_BAIL_LOCAL
    tcp_loc = _tcp_minus_pf_local()

    if stage_tag == "above":
        p_f_des = seat_floor_w + np.array([0.0, 0.0, +0.0760])  # filter base 76 mm above seat floor (+19 mm above rim)
    elif stage_tag == "mid":
        p_f_des = seat_floor_w + np.array([0.0, 0.0, +0.0300])  # guided funnel entry
    elif stage_tag == "seat":
        p_f_des = seat_floor_w + np.array([0.0, 0.0, +0.0015])  # gentle touchdown on Z_SEAT_FLOOR
    else:  # "clear": slide open 44 mm jaws straight back along -gz (finger axis) off the 4 mm standing tab
        p_tcp_seat = (seat_floor_w + np.array([0.0, 0.0, +0.0015])) + R_f_seat @ tcp_loc
        return p_tcp_seat - 0.0560 * R_tcp_seat[:, 2], R_tcp_seat

    p_tcp_eq = p_f_des + R_f_seat @ tcp_loc
    if stage_tag in ("above", "mid"):
        return _apply_carry_pd(p_tcp_eq, R_tcp_seat, FILTER_SEAT_YAW)
    return p_tcp_eq, R_tcp_seat


P_R_CLEAR_HOOD = np.array([0.2450, -0.1850, TOP + 0.2450])
R_FOLD_LOCAL = make_frame([0.0, +1.0, 0.0], [-0.7248, 0.0, -0.6889])
_RECLAMP_SWIVEL_RAD: float | None = None


def _bail_fold_target(fold_tag: str):
    """Sweep closed Right fingertip onto the 44 mm bail crossbar (108 mm clear of hood, min_z=49.2 mm) to fold bail <=2 deg."""
    if _LIVE_FILTER is not None and float(_LIVE_FILTER["p_f"][2]) > TOP + 0.15:
        p_f = _LIVE_FILTER["p_f"]
        R_f = _LIVE_FILTER["R_f"]
    else:
        p_f = _open_basket_seat_floor_world()
        R_f = rz(FILTER_SEAT_YAW)

    yaw_f = math.atan2(float(R_f[1, 0]), float(R_f[0, 0]))
    R_fold = rz(yaw_f) @ R_FOLD_LOCAL

    # Closed fingertip apex is ~12.5 mm ahead of TCP along R_fold[:, 2]:
    if fold_tag == "stage":
        pt_f = np.array([+0.0040, +0.0160, 0.1040])
    elif fold_tag == "engage":
        pt_f = np.array([+0.0040, -0.0080, 0.0860])
    elif fold_tag == "tip":
        pt_f = np.array([+0.0040, -0.0250, 0.0680])
    elif fold_tag == "down":
        pt_f = np.array([+0.0040, -0.0385, 0.0540])
    else:  # "lift"
        pt_f = np.array([+0.0060, -0.0350, 0.1050])

    p_tip_w = p_f + R_f @ pt_f
    return p_tip_w - 0.0125 * R_fold[:, 2], R_fold


@on_reset
def _reset(env):
    global _step, _LIVE_BREWER, _LIVE_FILTER, _BAIL_SNAP, _RECLAMP_SWIVEL_RAD
    _step = 0
    _LIVE_BREWER = None
    _LIVE_FILTER = None
    _BAIL_SNAP = None
    _RECLAMP_SWIVEL_RAD = None
    _METRIC.reset()


@on_step
def _update(env):
    global _step, _LIVE_BREWER, _LIVE_FILTER, _BAIL_SNAP, _RECLAMP_SWIVEL_RAD
    _step += 1
    br_s = labkit._part_state(env, COFFEE_MAKER)
    fl_s = labkit._part_state(env, PERM_FILTER)
    _METRIC.update(env, br_s, fl_s)

    p_b = np.asarray(br_s.pos, float).copy()
    R_b = br_s.rotation().copy()
    swivel_rad = float(br_s.joints.get("swivel_holder", 0.0))
    _LIVE_BREWER = dict(p_b=p_b, R_b=R_b, swivel_rad=swivel_rad)
    if _step == 350 and _RECLAMP_SWIVEL_RAD is None and abs(swivel_rad) > 0.5:
        _RECLAMP_SWIVEL_RAD = swivel_rad

    p_f = np.asarray(fl_s.pos, float).copy()
    R_f = fl_s.rotation().copy()
    bail_rad = float(fl_s.joints.get("bail", BAIL_INIT_RAD))
    _LIVE_FILTER = dict(p_f=p_f, R_f=R_f, bail_rad=bail_rad)

    # Freeze initial filter pickup geometry right when Right Arm clamps the standing bail on the counter:
    if _step == 180 and _BAIL_SNAP is None:
        R_grasp = R_f @ R_BAIL_LOCAL
        tab_w = _bail_tab_world_from_filter(p_f, R_f, bail_rad)
        tcp_w = tab_w - PAD_TCP_BAIL * R_grasp[:, 2]
        _BAIL_SNAP = dict(
            p_f0=p_f.copy(),
            R_f0=R_f.copy(),
            bail_rad0=bail_rad,
            tcp_minus_pf_local=R_f.T @ (tcp_w - p_f),
        )

    if _DEBUG and (_step % 15 == 0 or _step <= 5):
        print(
            "CFDBG "
            + json.dumps(
                dict(
                    step=_step,
                    t=round(_step / 20.0, 2),
                    br_pos=[round(float(v), 4) for v in p_b],
                    fl_pos=[round(float(v), 4) for v in p_f],
                    **_METRIC.detail,
                )
            ),
            flush=True,
        )


def score(objects) -> float:
    return _METRIC.score()


def task_score(env) -> float:
    return _METRIC.score()


def success(objects) -> bool:
    return score(objects) == 1.0


def demo() -> dict[str, list[Waypoint]]:
    # Left-arm swivel open & close circular arc targets:
    def sw_wp(dur: float, ang_deg: float, finger: float, label: str, radial_back: float = 0.0, dz: float = 0.0):
        rad = math.radians(ang_deg)
        return wp(
            dur,
            SeenPos(lambda r=rad, b=radial_back, z=dz: _swivel_tab_target(r, b, z)[0]),
            finger,
            label,
            lambda r=rad, b=radial_back, z=dz: _swivel_tab_target(r, b, z)[1],
        )

    br_aisle_p = np.array([0.3650, -0.1950, TOP + 0.3980], float)
    br_pre_p, br_pre_r = _brewer_brace_target(+0.038)
    br_touch_p, br_touch_r = _brewer_brace_target(0.0)

    pk_pre_p = SeenPos(lambda: _filter_pickup_target("pre")[0])
    pk_pre_r = lambda: _filter_pickup_target("pre")[1]
    pk_grasp_p = SeenPos(lambda: _filter_pickup_target("grasp")[0])
    pk_grasp_r = lambda: _filter_pickup_target("grasp")[1]
    pk_lift_p = SeenPos(lambda: _filter_pickup_target("lift")[0])
    pk_lift_r = lambda: _filter_pickup_target("lift")[1]

    cy_1_p = SeenPos(lambda: _filter_carry_target(1)[0])
    cy_1_r = lambda: _filter_carry_target(1)[1]
    cy_2_p = SeenPos(lambda: _filter_carry_target(2)[0])
    cy_2_r = lambda: _filter_carry_target(2)[1]

    st_above_p = SeenPos(lambda: _filter_seat_target("above")[0])
    st_above_r = lambda: _filter_seat_target("above")[1]
    st_mid_p = SeenPos(lambda: _filter_seat_target("mid")[0])
    st_mid_r = lambda: _filter_seat_target("mid")[1]
    st_seat_p = SeenPos(lambda: _filter_seat_target("seat")[0])
    st_seat_r = lambda: _filter_seat_target("seat")[1]
    st_clear_p = SeenPos(lambda: _filter_seat_target("clear")[0])
    st_clear_r = lambda: _filter_seat_target("clear")[1]

    fd_stage_p = SeenPos(lambda: _bail_fold_target("stage")[0])
    fd_stage_r = lambda: _bail_fold_target("stage")[1]
    fd_eng_p = SeenPos(lambda: _bail_fold_target("engage")[0])
    fd_eng_r = lambda: _bail_fold_target("engage")[1]
    fd_tip_p = SeenPos(lambda: _bail_fold_target("tip")[0])
    fd_tip_r = lambda: _bail_fold_target("tip")[1]
    fd_down_p = SeenPos(lambda: _bail_fold_target("down")[0])
    fd_down_r = lambda: _bail_fold_target("down")[1]
    fd_lift_p = SeenPos(lambda: _bail_fold_target("lift")[0])
    fd_lift_r = lambda: _bail_fold_target("lift")[1]

    if _MODE == "unseated_close":
        # Negative control: open swivel basket, never load filter, close swivel basket -> must score 0.0:
        return {
            "left": [
                sw_wp(1.4, 0.0, F_OPEN_TAB, "pre-tab", radial_back=0.060),
                sw_wp(0.8, 0.0, F_OPEN_TAB, "on-tab"),
                sw_wp(0.8, 0.0, F_CLAMP_TAB, "clamp-tab"),
                sw_wp(0.9, -37.0, F_CLAMP_TAB, "open-37"),
                sw_wp(0.9, -74.0, F_CLAMP_TAB, "open-74"),
                sw_wp(0.9, -111.0, F_CLAMP_TAB, "open-111"),
                sw_wp(0.9, -74.0, F_CLAMP_TAB, "close-74"),
                sw_wp(0.9, -37.0, F_CLAMP_TAB, "close-37"),
                sw_wp(1.0, 0.0, F_CLAMP_TAB, "close-0"),
                sw_wp(0.7, 0.0, F_OPEN_TAB, "release-tab"),
                sw_wp(0.9, 0.0, F_OPEN_TAB, "back-tab", radial_back=0.070),
                wp(1.6, P_L_PARK, 0.20, "park-l", ROT_L_PARK),
                wp(3.5, P_L_PARK, 0.20, "hold-l", ROT_L_PARK),
            ],
            "right": [
                wp(0.7, br_aisle_p, F_CLOSED_PAD, "aisle-top-open", br_pre_r),
                wp(0.5, br_pre_p, F_CLOSED_PAD, "pre-top-open", br_pre_r),
                wp(0.6, br_touch_p, F_CLOSED_PAD, "press-top-open", br_touch_r),
                wp(7.4, br_touch_p, F_CLOSED_PAD, "hold-top-press", br_touch_r),
                wp(0.4, br_pre_p, F_CLOSED_PAD, "lift-from-top", br_pre_r),
                wp(0.5, br_aisle_p, F_CLOSED_PAD, "aisle-from-top", br_pre_r),
                wp(1.5, P_R_PARK, 0.20, "park-r", ROT_R_PARK),
                wp(3.6, P_R_PARK, 0.20, "hold-park-r", ROT_R_PARK),
            ],
        }

    skip_fold = (_MODE == "upright_bail_close")
    fold_dur = 0.7 if skip_fold else (0.9 + 0.6 + 0.8 + 0.9 + 0.8 + 0.7)  # includes 0.7s outboard swing clear of hood

    # Exact synchronized timeline:
    # t = 0.0..6.4 s: Right braces brewer; Left clamps pull tab & swings open 0 -> -111.2 deg, releases, parks at P_L_WAIT (t = 8.1 s).
    # t = 6.4..16.2 s: Right clears brace, grasps BF215 standing bail on counter (clamped at t = 9.2 s / step 184), lifts to TOP+0.307,
    #                  centers over live seen open Removable Filter Basket, lowers straight to Z_SEAT_FLOOR, releases & clears (t = 16.2 s).
    # t = 16.2..20.2 s (demo only): Right folds bail from -90.5 deg flat to 0.0 deg inside filter rim and lifts clear.
    # Next 7.2 s: Right re-braces brewer while Left re-clamps open pull tab & swings -111.2 -> 0.0 deg flush, then both retract to PARK!
    left_wps = [
        sw_wp(1.4, 0.0, F_OPEN_TAB, "pre-tab", radial_back=0.060),
        sw_wp(0.8, 0.0, F_OPEN_TAB, "on-tab"),
        sw_wp(0.8, 0.0, F_CLAMP_TAB, "clamp-tab"),
        sw_wp(0.8, -28.0, F_CLAMP_TAB, "open-28"),
        sw_wp(0.8, -56.0, F_CLAMP_TAB, "open-56"),
        sw_wp(0.9, -84.0, F_CLAMP_TAB, "open-84"),
        sw_wp(0.9, -114.0, F_CLAMP_TAB, "open-114"),     # t = 6.4 s (settles near -111 deg)
        sw_wp(0.7, -114.0, F_OPEN_TAB, "release-open"),  # t = 7.1 s
        sw_wp(0.8, -114.0, F_OPEN_TAB, "back-open", radial_back=0.060),  # t = 7.9 s
        wp(1.0, P_L_WAIT, 0.20, "wait-park-l", ROT_L_WAIT),              # t = 8.9 s
        wp(8.45 + fold_dur + 1.6, P_L_WAIT, 0.20, "hold-wait-l", ROT_L_WAIT),  # waits until Right re-braces at t = 18.95 + fold_dur
        wp(1.1, SeenPos(lambda: _swivel_tab_target(_RECLAMP_SWIVEL_RAD if _RECLAMP_SWIVEL_RAD is not None else SWIVEL_OPEN_RAD, 0.055)[0]), F_OPEN_TAB, "re-pre-tab", lambda: _swivel_tab_target(_RECLAMP_SWIVEL_RAD if _RECLAMP_SWIVEL_RAD is not None else SWIVEL_OPEN_RAD, 0.055)[1]),
        wp(0.7, SeenPos(lambda: _swivel_tab_target(_RECLAMP_SWIVEL_RAD if _RECLAMP_SWIVEL_RAD is not None else SWIVEL_OPEN_RAD)[0]), F_OPEN_TAB, "re-on-tab", lambda: _swivel_tab_target(_RECLAMP_SWIVEL_RAD if _RECLAMP_SWIVEL_RAD is not None else SWIVEL_OPEN_RAD)[1]),
        wp(0.7, SeenPos(lambda: _swivel_tab_target(_RECLAMP_SWIVEL_RAD if _RECLAMP_SWIVEL_RAD is not None else SWIVEL_OPEN_RAD)[0]), F_CLAMP_TAB, "re-clamp-tab", lambda: _swivel_tab_target(_RECLAMP_SWIVEL_RAD if _RECLAMP_SWIVEL_RAD is not None else SWIVEL_OPEN_RAD)[1]),
        sw_wp(0.8, -84.0, F_CLAMP_TAB, "close-84"),
        *(
            [
                sw_wp(1.1, -58.2, F_CLAMP_TAB, "close-against-bail-stop"),
                sw_wp(0.7, -59.5, F_OPEN_TAB, "release-ajar"),
                sw_wp(0.9, -59.5, F_OPEN_TAB, "back-ajar", radial_back=0.068),
            ]
            if skip_fold
            else [
                sw_wp(0.8, -56.0, F_CLAMP_TAB, "close-56"),
                sw_wp(0.9, -28.0, F_CLAMP_TAB, "close-28"),
                sw_wp(1.0, -1.0, F_CLAMP_TAB, "close-1"),
                sw_wp(0.8, +4.5, F_CLAMP_TAB, "close-flush"),
                sw_wp(0.7, +1.0, F_OPEN_TAB, "release-closed"),
                sw_wp(0.9, 0.0, F_OPEN_TAB, "back-closed", radial_back=0.068),
            ]
        ),
        wp(1.6, P_L_PARK, 0.20, "park-l", ROT_L_PARK),
        wp(3.5, P_L_PARK, 0.20, "hold-l", ROT_L_PARK),
    ]

    right_wps = [
        wp(0.7, br_aisle_p, F_CLOSED_PAD, "aisle-top-brace-open", br_pre_r), # t = 0.7 s (high -Y aisle clear of hood)
        wp(0.5, br_pre_p, F_CLOSED_PAD, "pre-top-brace-open", br_pre_r),     # t = 1.2 s
        wp(0.6, br_touch_p, F_CLOSED_PAD, "press-top-open", br_touch_r),     # t = 1.8 s
        wp(4.6, br_touch_p, F_CLOSED_PAD, "hold-top-open", br_touch_r),      # t = 6.4 s
        wp(0.35, br_pre_p, F_CLOSED_PAD, "lift-from-top-open", br_pre_r),    # t = 6.75 s
        wp(0.45, br_aisle_p, F_OPEN_BAIL, "aisle-from-top-open", br_pre_r),  # t = 7.2 s (exact timing preserved!)
        wp(1.1, pk_pre_p, F_OPEN_BAIL, "pre-bail", pk_pre_r),               # t = 8.3 s
        wp(0.8, pk_grasp_p, F_OPEN_BAIL, "on-bail", pk_grasp_r),            # t = 9.1 s (step 180 snaps filter pose)
        wp(0.9, pk_grasp_p, F_CLAMP_BAIL, "clamp-bail", pk_grasp_r),        # t = 10.0 s
        wp(1.2, pk_lift_p, F_CLAMP_BAIL, "lift-high", pk_lift_r),           # t = 11.2 s
        wp(0.85, cy_1_p, F_CLAMP_BAIL, "carry-arc-1", cy_1_r),              # t = 12.05 s
        wp(0.85, cy_2_p, F_CLAMP_BAIL, "carry-arc-2", cy_2_r),              # t = 12.90 s
        wp(0.85, st_above_p, F_CLAMP_BAIL, "center-above-basket", st_above_r),  # t = 13.75 s
        wp(0.9, st_mid_p, F_CLAMP_BAIL, "funnel-entry", st_mid_r),          # t = 14.65 s
        wp(1.0, st_seat_p, F_CLAMP_BAIL, "seat-floor", st_seat_r),          # t = 15.65 s
        wp(0.8, st_seat_p, F_RELEASE_BAIL, "release-bail", st_seat_r),      # t = 16.45 s
        wp(0.9, st_clear_p, F_RELEASE_BAIL, "clear-bail", st_clear_r),      # t = 17.35 s
    ]

    if not skip_fold:
        right_wps.extend([
            wp(0.9, fd_stage_p, F_CLOSED_PAD, "stage-fold", fd_stage_r),    # t = 18.25 s
            wp(0.6, fd_eng_p, F_CLOSED_PAD, "engage-bail", fd_eng_r),       # t = 18.85 s
            wp(0.8, fd_tip_p, F_CLOSED_PAD, "tip-bail", fd_tip_r),          # t = 19.65 s
            wp(0.9, fd_down_p, F_CLOSED_PAD, "fold-flat", fd_down_r),       # t = 20.55 s
            wp(0.8, fd_lift_p, F_CLOSED_PAD, "lift-fold", fd_lift_r),       # t = 21.35 s
        ])
    right_wps.append(wp(0.7, P_R_CLEAR_HOOD, F_CLOSED_PAD, "swing-outboard-clear-hood", ROT_BAIL_CLEAR))

    right_wps.extend([
        wp(0.55, br_aisle_p, F_CLOSED_PAD, "aisle-top-brace-close", br_pre_r),  # t = 17.90 + fold_dur
        wp(0.45, br_pre_p, F_CLOSED_PAD, "pre-top-brace-close", br_pre_r),      # t = 18.35 + fold_dur
        wp(0.60, br_touch_p, F_CLOSED_PAD, "press-top-close", br_touch_r),      # t = 18.95 + fold_dur
        wp(6.9, br_touch_p, F_CLOSED_PAD, "hold-top-close", br_touch_r),        # t = 25.85 + fold_dur
        wp(0.4, br_pre_p, F_CLOSED_PAD, "lift-from-top-close", br_pre_r),       # t = 26.15 + fold_dur
        wp(0.5, br_aisle_p, F_CLOSED_PAD, "aisle-from-top-close", br_pre_r),    # t = 26.65 + fold_dur
        wp(1.5, P_R_PARK, 0.20, "park-r", ROT_R_PARK),                          # t = 28.15 + fold_dur
        wp(5.2, P_R_PARK, 0.20, "hold-r", ROT_R_PARK),
    ])

    return {"left": left_wps, "right": right_wps}
