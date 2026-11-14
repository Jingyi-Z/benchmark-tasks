"""colander-sink-bridge: pull out both telescoping handles of an over-the-sink colander and bridge it across a sink basin."""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arena_assets_fix  # noqa: F401
import labkit
from labkit import Joint, on_reset, on_step, part

from yam_arena import demo as _demo
from yam_arena import ik as _ik
from yam_arena.constants import FINGER_OPEN, READY as _READY
from yam_arena.demo import ArmPlan, Waypoint
from yam_arena.task_file import TABLE_TOP
from yam_arena.urdf import axis_angle_matrix

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 1000

INSTRUCTION = (
    "Pull out the extendable handles of the colander and rest both handles "
    "across the left and right rims of the sink so the colander hangs suspended over the basin."
)

# -- Colors ------------------------------------------------------------------------------------------------------------
STEEL_BRIGHT = (0.78, 0.80, 0.84)
STEEL_MESH = (0.72, 0.75, 0.79)
SINK_RIM_STEEL = (0.56, 0.59, 0.63)
SINK_BASIN_DARK = (0.30, 0.33, 0.36)
RUBBER_BLACK = (0.12, 0.13, 0.14)
DRAIN_CHROME = (0.48, 0.51, 0.55)

# -- Fixed Countertop Sink-Basin Module (585 x 300 x 125 mm, 455 x 276 mm open basin) ---------------------------------
SINK_XY = np.array([0.435, 0.000], float)
SINK_DX = 0.300
SINK_DY = 0.585
SINK_H = 0.125
BASIN_HALF_Y = 0.2275      # 455.0 mm open sink basin width (> 340 mm collapsed & > 445 mm one-sided extension!)
DECK_HALF_Y = 0.2925       # +-292.5 mm outer edges (65.0 mm-wide left & right stainless sink deck ledges)
BASIN_FLOOR_H = 0.006      # sink floor at TOP + 6.0 mm
WALL_T = 0.012             # 12.0 mm front & rear sink apron walls (276 mm clear front-to-back basin opening > 230 mm basket)

SINK_BOXES = [
    # Basin floor (455 x 252 x 8 mm) + central drain strainer visual ring
    ((0.0, 0.0, BASIN_FLOOR_H / 2), (SINK_DX - 2 * WALL_T, 2 * BASIN_HALF_Y, BASIN_FLOOR_H), SINK_BASIN_DARK),
    # Left & right 65 mm-wide full-height stainless sink deck ledges (y = +227.5..+292.5 mm and -292.5..-227.5 mm)
    *[
        ((0.0, sy * (BASIN_HALF_Y + 0.0325), SINK_H / 2), (SINK_DX, 0.065, SINK_H), SINK_RIM_STEEL)
        for sy in (-1, 1)
    ],
    # Front & rear stainless sink walls (x = +-133 mm, leaving 252 mm clear basin opening along X)
    *[
        ((sx * (SINK_DX / 2 - WALL_T / 2), 0.0, SINK_H / 2), (WALL_T, 2 * BASIN_HALF_Y, SINK_H), SINK_RIM_STEEL)
        for sx in (-1, 1)
    ],
]
SINK_CYLS = [
    ((0.0, 0.0, BASIN_FLOOR_H + 0.0015), 0.042, 0.003, DRAIN_CHROME, "Z"),
]

SINK = part(
    "sink_module",
    pos=(float(SINK_XY[0]), float(SINK_XY[1]), TOP),
    fixed=True,
    friction=0.80,
    boxes=SINK_BOXES,
    cylinders=SINK_CYLS,
)

# -- 1:1 Norpro #2158 Expanding Over-the-Sink Colander (340 x 230 x 120 mm collapsed -> 550 mm, 0.77 kg) --------------
COL_START_XY = np.array([0.155, 0.000], float)
BASKET_DX = 0.230          # 230.0 mm basket width along X (rear at x=0.270 m, +15 mm clear in front of sink x=0.285 m!)
BASKET_DY = 0.236          # 236.0 mm central basket length along Y (+ 2 x 52 mm U-handles = 340.0 mm collapsed span!)
COL_H = 0.120              # 120.0 mm overall collapsed height
SLIDE_MAX = 0.105          # 105.0 mm per side -> 340 + 210 = 550.0 mm max span (Norpro #2158 official spec!)
SLIDE_FRIC = 1.6           # light realistic friction-slide sleeve resistance (not inflated; sequential pulls allowed)
HANDLE_BAR_Y = 0.1655      # thin 9 mm stainless U-handle end bar center (outer face at +-170.0 mm = 340.0 mm collapsed span!)
HANDLE_BAR_DY = 0.009      # authentic 9.0 mm stainless steel U-rod thickness (coplanar with slide rails, zero raised bar!)
HANDLE_BAR_DX = 0.193      # 193.0 mm continuous stainless U-handle end bar joining both x=+-92 mm rails seamlessly!
HANDLE_BAR_Z = 0.112       # exact coplanar Z centerline with the two parallel stainless slide rails (z = 0.1075..0.1165 m)
FOOT_REST_Y = 0.154        # corner rubber rest foot center Y under each U-handle corner (at ext q, sits at +-0.154 +- q)
FOOT_BOTTOM_Z = 0.098      # underside of downward corner rubber feet (+14 mm below z=0.112 m steel bar, +21 mm basin gap!)
GRASP_X_OFF = 0.006        # +-6 mm X-stagger along straight handle bars prevents single-line hinge pitch!

BASKET_BOXES = [
    # Four non-slip black rubber base feet at (x=+-90 mm, y=+-94 mm, z=0..8 mm)
    *[
        ((sx * 0.090, sy * 0.094, 0.004), (0.018, 0.018, 0.008), RUBBER_BLACK)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ],
    # Slatted stainless strainer floor (7 ribs with 10 mm open drain slots so sink basin & drain show through!)
    *[
        ((0.0, ky * 0.028, 0.0115), (0.198, 0.018, 0.007), STEEL_MESH)
        for ky in (-3, -2, -1, 0, 1, 2, 3)
    ],
    *[
        ((sx * 0.094, 0.0, 0.0115), (0.012, 0.204, 0.007), STEEL_BRIGHT)
        for sx in (-1, 1)
    ],
    # Four stainless corner frame uprights
    *[
        ((sx * 0.104, sy * 0.108, 0.056), (0.012, 0.012, 0.092), STEEL_BRIGHT)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ],
    # Slatted stainless strainer sidewalls (3 horizontal bands per wall with open drain gaps!)
    *[
        ((sx * 0.104, 0.0, zb), (0.009, 0.216, 0.018), STEEL_MESH)
        for sx in (-1, 1)
        for zb in (0.026, 0.056, 0.086)
    ],
    *[
        ((0.0, sy * 0.108, zb), (0.208, 0.009, 0.016), STEEL_MESH)
        for sy in (-1, 1)
        for zb in (0.025, 0.054, 0.083)
    ],
    # Upper stainless perimeter rim frame + continuous stainless slide guide sleeves & crimped collars at x = +-0.092 m
    *[
        ((sx * 0.111, 0.0, 0.110), (0.008, BASKET_DY, 0.012), STEEL_BRIGHT)
        for sx in (-1, 1)
    ],
    *[
        ((0.0, sy * 0.114, 0.101), (BASKET_DX, 0.008, 0.008), STEEL_BRIGHT)
        for sy in (-1, 1)
    ],
    *[
        ((sx * 0.092, 0.0, 0.112), (0.013, 0.152, 0.013), STEEL_BRIGHT)
        for sx in (-1, 1)
    ],
    *[
        ((sx * 0.092, sy * 0.098, 0.112), (0.016, 0.044, 0.016), STEEL_BRIGHT)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ],
]


def _handle_boxes(sy: int):
    return [
        # Continuous parallel 9x9 mm stainless slide rails along Y at x = +-0.092 m (y = 0.004..0.1655 m; even at max
        # 105 mm stroke, inner ends remain at y = 0.109 m inside the 0.076..0.120 m crimped collars -> zero gap always!)
        *[
            ((sx * 0.092, sy * 0.08475, HANDLE_BAR_Z), (0.009, 0.1615, 0.009), STEEL_BRIGHT)
            for sx in (-1, 1)
        ],
        # Coplanar 9x9 mm continuous stainless U-handle end bar joining both x = +-0.092 m rails as one piece
        ((0.0, sy * HANDLE_BAR_Y, HANDLE_BAR_Z), (HANDLE_BAR_DX, HANDLE_BAR_DY, 0.009), STEEL_BRIGHT),
        # Compact downward black rubber rest feet only under the two U-handle corners (z = 0.098..0.109 m)
        *[
            ((sx * 0.088, sy * FOOT_REST_Y, 0.1035), (0.018, 0.020, 0.011), RUBBER_BLACK)
            for sx in (-1, 1)
        ],
    ]


ARM_LEFT = Joint(
    "arm_left",
    kind="slide",
    axis="Y",
    anchor=(0.0, 0.0, 0.112),
    lower=0.0,
    upper=SLIDE_MAX,
    stiffness=0.0,
    damping=0.8,
    target=0.0,
    friction=SLIDE_FRIC,
    initial=0.0,
    mass=0.110,
    boxes=_handle_boxes(+1),
)

ARM_RIGHT = Joint(
    "arm_right",
    kind="slide",
    axis="Y",
    anchor=(0.0, 0.0, 0.112),
    lower=-SLIDE_MAX,
    upper=0.0,
    stiffness=0.0,
    damping=0.8,
    target=0.0,
    friction=SLIDE_FRIC,
    initial=0.0,
    mass=0.110,
    boxes=_handle_boxes(-1),
)

COL_START = np.array([COL_START_XY[0], COL_START_XY[1], TOP + 0.0003], float)
COLANDER = part(
    "colander",
    pos=tuple(COL_START),
    mass=0.550,   # 0.550 kg basket + 2 x 0.110 kg slide handles = 0.770 kg (1.7 lb official Norpro #2158 weight!)
    friction=0.78,
    self_collide=False,
    boxes=BASKET_BOXES,
    joints=[ARM_LEFT, ARM_RIGHT],
)

OBJECTS = [SINK, COLANDER]

# -- Rotation helpers & physical-impact / independent-jaw-endpoint / stillness scorer ----------------------------------
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def Rx(a: float) -> np.ndarray:
    return axis_angle_matrix(X, a)


def Ry(a: float) -> np.ndarray:
    return axis_angle_matrix(Y, a)


def Rz(a: float) -> np.ndarray:
    return axis_angle_matrix(Z, a)


HANDS_CLEAR = 0.080
SETTLE_STEPS = 20          # 21 snapshots = 1.00 s at 20 Hz
SETTLE_TOL = 0.001         # < 1.0 mm 3D Euclidean corner-point motion
EXT_PARTIAL_MIN = 0.135    # q_left + |q_right| >= 135 mm (span >= 475 mm > 455 mm basin width)
RIM_OVERLAP_MIN = 0.0175   # outer handle rest foot center >= 227.5 + 17.5 = 245.0 mm (17.5 mm bearing overlap per side)
TILT_MAX_DEG = 5.0
CENTER_X_TOL = 0.040       # basket centered within +-40 mm of x = 0.466 m (well inside 312 mm open basin)
SUSPEND_CLEAR_MIN = 0.015  # basket bottom >= 15 mm clear above TOP + 6 mm sink basin floor

IMPACT_SPEED = 0.50
TABLE_X0, TABLE_X1 = -0.19, 0.91
TABLE_Y_HALF = 0.65
FLOOR_Z = -0.0125
CONTACT_TOL = 0.0008
PENETRATION_FLOOR = -0.004
CONTROL_DT = 0.05

_SINK_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in SINK_BOXES)
_SINK_CYL_SPECS = tuple((np.asarray(c, float), float(r), 0.5 * float(h)) for c, r, h, *_ in SINK_CYLS)
_BASKET_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in BASKET_BOXES)
_HL_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in _handle_boxes(+1))
_HR_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in _handle_boxes(-1))

def _box_vertices(specs) -> tuple[np.ndarray, ...]:
    verts = []
    for c, h in specs:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    verts.append(c + np.array([sx * h[0], sy * h[1], sz * h[2]], float))
    return tuple(verts)


_ACTUAL_BASKET_VERTS = _box_vertices(_BASKET_SPECS)
_ACTUAL_HL_VERTS = _box_vertices(_HL_SPECS)
_ACTUAL_HR_VERTS = _box_vertices(_HR_SPECS)

_state: dict = {}
_hist: list = []


@on_reset
def _reset(env):
    _state.clear()
    _state.update(step=0, viol=set(), settled=False, clear_all=False, prev_pts={}, prev_down={})
    _hist.clear()


def _tips(env):
    """Both pad midpoints and all four independent jaw endpoints (installed URDF FK: joint7 -> -gx, joint8 -> +gx)."""
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
    return min(float(np.linalg.norm(np.maximum(np.abs(ql - c) - h, 0.0))) for c, h in specs) if specs else 1e9


def _min_zcyl_dist(ql: np.ndarray, specs) -> float:
    best = 1e9
    for c, r, hh in specs:
        d = ql - c
        dr = max(math.hypot(float(d[0]), float(d[1])) - r, 0.0)
        dz = max(abs(float(d[2])) - hh, 0.0)
        best = min(best, math.hypot(dr, dz))
    return best


def _surface_dist_all(q_world: np.ndarray, sk, col) -> float:
    q_sk = sk.to_local(q_world)
    d_sk = min(_min_box_dist(q_sk, _SINK_SPECS), _min_zcyl_dist(q_sk, _SINK_CYL_SPECS))

    q_col = col.to_local(q_world)
    d_bsk = _min_box_dist(q_col, _BASKET_SPECS)

    ql = q_col - np.array([0.0, float(col.joints["arm_left"]), 0.0], float)
    qr = q_col - np.array([0.0, float(col.joints["arm_right"]), 0.0], float)
    d_hl = _min_box_dist(ql, _HL_SPECS)
    d_hr = _min_box_dist(qr, _HR_SPECS)
    return min(d_sk, d_bsk, d_hl, d_hr)


def _support_gap(p: np.ndarray, sk) -> float:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    on_table = (TABLE_X0 <= x <= TABLE_X1) and (abs(y) <= TABLE_Y_HALF)
    if not on_table:
        return z - FLOOR_Z
    p_sk = sk.to_local(p)
    in_basin_xy = (abs(float(p_sk[0])) <= (SINK_DX / 2 - WALL_T)) and (abs(float(p_sk[1])) <= BASIN_HALF_Y)
    if in_basin_xy:
        return z - (TOP + BASIN_FLOOR_H)
    in_deck_xy = (abs(float(p_sk[0])) <= SINK_DX / 2) and (BASIN_HALF_Y < abs(float(p_sk[1])) <= DECK_HALF_Y)
    if in_deck_xy and z >= TOP + SINK_H - 0.015:
        return z - (TOP + SINK_H)
    return z - TOP


def _check_impact(key: str, pts: np.ndarray, sk) -> bool:
    prev = _state["prev_pts"].get(key)
    prev_down = _state["prev_down"].get(key)
    _state["prev_pts"][key] = pts
    if prev is None:
        _state["prev_down"][key] = np.zeros(len(pts), float)
        return False
    down = np.maximum(0.0, (prev[:, 2] - pts[:, 2]) / CONTROL_DT)
    _state["prev_down"][key] = down
    if prev_down is None:
        return False
    for i in range(len(pts)):
        g_now = _support_gap(pts[i], sk)
        g_prev = _support_gap(prev[i], sk)
        just_hit = (PENETRATION_FLOOR <= g_now <= CONTACT_TOL) and (g_prev > CONTACT_TOL)
        pre_impact = max(float(down[i]), float(prev_down[i]))
        if just_hit and pre_impact > IMPACT_SPEED:
            return True
    return False


def _tracked_points(col) -> np.ndarray:
    """Actual collider vertices of basket and both slide handles (eliminates empty-AABB-corner false impact)."""
    ql = np.array([0.0, float(col.joints["arm_left"]), 0.0], float)
    qr = np.array([0.0, float(col.joints["arm_right"]), 0.0], float)
    pts = [col.to_world(p) for p in _ACTUAL_BASKET_VERTS]
    pts.extend(col.to_world(p + ql) for p in _ACTUAL_HL_VERTS)
    pts.extend(col.to_world(p + qr) for p in _ACTUAL_HR_VERTS)
    return np.asarray(pts, float)


@on_step
def _watch(env):
    _state["step"] = _state.get("step", 0) + 1
    sk = labkit._part_state(env, labkit._SPECS["sink_module"])
    col = labkit._part_state(env, labkit._SPECS["colander"])
    pts = _tracked_points(col)

    if _state["step"] > 5:
        if _check_impact("colander", pts, sk):
            _state["viol"].add("colander_impact")
        if float(col.pos[2]) < TOP - 0.08 or abs(float(col.pos[1])) > TABLE_Y_HALF or not (TABLE_X0 <= float(col.pos[0]) <= TABLE_X1):
            _state["viol"].add("fell_off_table")

    _hist.append(pts)
    if len(_hist) > SETTLE_STEPS + 1:
        _hist.pop(0)
    if len(_hist) == SETTLE_STEPS + 1:
        arr = np.stack(_hist, axis=0)
        disp = np.linalg.norm(arr[:, None, :, :] - arr[None, :, :, :], axis=-1)
        _state["settled"] = bool(float(np.max(disp)) < SETTLE_TOL)
    else:
        _state["settled"] = False

    tips = _tips(env)
    _state["clear_all"] = all(_surface_dist_all(t, sk, col) >= HANDS_CLEAR for t in tips)


def task_score(env) -> float:
    if _state.get("viol"):
        return 0.0
    if not _state.get("clear_all", False) or not _state.get("settled", False):
        return 0.0

    sk = labkit._part_state(env, labkit._SPECS["sink_module"])
    col = labkit._part_state(env, labkit._SPECS["colander"])

    q_l = max(0.0, float(col.joints["arm_left"]))
    q_r = max(0.0, -float(col.joints["arm_right"]))
    total_ext = q_l + q_r
    if total_ext < EXT_PARTIAL_MIN:
        return 0.0

    # Check whether both extended handle rubber-foot bars bear flat on opposite sink rims with basket suspended clear
    lf_pts = [sk.to_local(col.to_world(np.array([sx * 0.092, +FOOT_REST_Y + q_l, FOOT_BOTTOM_Z], float))) for sx in (-1.0, 1.0)]
    rf_pts = [sk.to_local(col.to_world(np.array([sx * 0.092, -FOOT_REST_Y - q_r, FOOT_BOTTOM_Z], float))) for sx in (-1.0, 1.0)]

    # Support either yaw = 0 deg (left arm on +Y rim, right arm on -Y rim) OR yaw = 180 deg (swapped rims!)
    pos_feet = lf_pts if float(lf_pts[0][1]) >= 0.0 else rf_pts
    neg_feet = rf_pts if float(lf_pts[0][1]) >= 0.0 else lf_pts

    rim_pos_ok = all(
        (BASIN_HALF_Y + RIM_OVERLAP_MIN - 0.009 <= float(p[1]) <= DECK_HALF_Y)
        and (abs(float(p[0])) <= SINK_DX / 2)
        and (abs(float(p[2]) - SINK_H) <= 0.010)
        for p in pos_feet
    )
    rim_neg_ok = all(
        (-DECK_HALF_Y <= float(p[1]) <= -(BASIN_HALF_Y + RIM_OVERLAP_MIN - 0.009))
        and (abs(float(p[0])) <= SINK_DX / 2)
        and (abs(float(p[2]) - SINK_H) <= 0.010)
        for p in neg_feet
    )

    bottom_corners = [col.to_world(np.array([sx * 0.090, sy * 0.102, 0.0], float)) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    min_bottom_z = min(float(p[2]) for p in bottom_corners)
    clear_above_basin = min_bottom_z - (TOP + BASIN_FLOOR_H)
    centered_x = abs(float(col.pos[0]) - float(SINK_XY[0])) <= CENTER_X_TOL
    level_ok = col.tilt_deg() <= TILT_MAX_DEG

    if rim_pos_ok and rim_neg_ok and clear_above_basin >= SUSPEND_CLEAR_MIN and centered_x and level_ok:
        return 1.0
    return 0.40


SUCCESS_HOLD_S = 1.0


def score(objects) -> float:
    e = labkit.env()
    return float(task_score(e)) if e is not None else 0.0


def success(objects) -> bool:
    return score(objects) == 1.0


# -- Arbitrary-3D-orientation Waypoint monkeypatch + closed-loop live TCP trim ----------------------------------------
TIP_ABOVE_TCP = 0.013


class Orient(float):
    """Waypoint yaw carrying a full 3x3 TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_csb", False):
    def _grip_rot(yaw=0.0, _prev=_prev_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rot._csb = True
    _demo.grip_rotation = _grip_rot


def _ik_err(pe, re):
    return pe * 1000.0 + math.degrees(re)


def _endpoints(plan):
    rng = np.random.default_rng(7)
    seeds = [np.array(_READY, float)] + [np.array(_READY, float) + rng.normal(0.0, 0.8, 6) for _ in range(12)]
    plan._q0 = plan.q.copy()
    q_prev, out, switch = plan.q.copy(), [], set()
    for i, w in enumerate(plan.waypoints):
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, q_prev, base_pos=plan.base_pos, iters=200)
        best = (_ik_err(pe, re), q, pe, re)
        if best[0] > 1.0 or float(np.max(np.abs(q - q_prev))) > 1.75:
            for s in seeds:
                qs, ps, rs = _ik.solve(pos, quat, s, base_pos=plan.base_pos, iters=300)
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
    if not hasattr(plan, "_q_end"):
        _endpoints(plan)
    acc = 0.0
    for i, w in enumerate(plan.waypoints):
        if t <= acc + w.duration or i == len(plan.waypoints) - 1:
            u = 1.0 if w.duration <= 0 else min(max((t - acc) / w.duration, 0.0), 1.0)
            s = u * u * (3.0 - 2.0 * u)
            f0 = plan.finger_start if i == 0 else plan.waypoints[i - 1].finger
            finger = (1.0 - s) * f0 + s * w.finger
            grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            if i in plan._switch and not isinstance(w.pos, _Dyn) and (i == 0 or not isinstance(plan.waypoints[i - 1].pos, _Dyn)):
                q0 = plan._q0 if i == 0 else plan._q_end[i - 1]
                plan.q = (1.0 - s) * q0 + s * plan._q_end[i]
                return np.append(plan.q, grip), w.label
            _, rot, finger_t, label = plan.target(t)
            grip = float(np.clip(finger_t / FINGER_OPEN, 0.0, 1.0))
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


_ct = {"trim": {}, "wins": [], "last": {}}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}


def _live_tcp(env, robot_name: str) -> np.ndarray:
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    return _rt.tcp_world(b.scene[robot_name], 0, past_tcp=0.0) - o


@on_step
def _ctrack(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, fn in _ct["wins"]:
        if not t0 <= t < t1:
            continue
        rname = "left_robot" if key.startswith("l") else "right_robot"
        p = _live_tcp(env, rname)
        last = _ct["last"].get(key)
        _ct["last"][key] = p
        if last is None or np.max(np.abs(p - last)) > 0.0025:
            continue
        e = np.asarray(fn(env), float)
        tr = _ct["trim"].get(key, np.zeros(3))
        _ct["trim"][key] = np.clip(tr + np.clip(0.38 * e, -0.0028, 0.0028), -0.035, 0.035)
    if os.environ.get("CSB_DEBUG", "0") == "1" and int(round(t * 20)) % 10 == 0:
        col = labkit._part_state(env, labkit._SPECS["colander"])
        ql = float(col.joints["arm_left"]) * 1000.0
        qr = -float(col.joints["arm_right"]) * 1000.0
        print(
            f"CSBDBG step={_state.get('step',0)} t={t:.2f} ql={ql:.1f} qr={qr:.1f} "
            f"pos={np.round(col.pos,3)} tilt={col.tilt_deg():.1f} viol={sorted(_state.get('viol',()))}",
            flush=True,
        )


class _Dyn:
    def __init__(self, tcp, keys):
        self.tcp = np.asarray(tcp, float)
        self.keys = (keys,) if isinstance(keys, str) else tuple(keys)

    def __array__(self, dtype=None, copy=None):
        off = sum((_ct["trim"].get(k, np.zeros(3)) for k in self.keys), np.zeros(3))
        return np.asarray(self.tcp + off, dtype=dtype or float)


# -- Constant-Branch Y-Jaw Handle Grasps (normal force pulls handles outward, +11 mm rim clearance at seatdown!) ------
DOWN = _prev_grip(0.0)                 # col 0 = [0, -1, 0] (Y), col 2 = [0, 0, -1] (-Z)
R_GRIP = Ry(-math.radians(28.0)) @ DOWN
O_GRIP = Orient(R_GRIP)
HANDLE_OPEN = -0.021                   # 42 mm Y-jaw gap clears 20 mm rubberized handle bar by 11 mm/side!
HANDLE_PAST = 0.001                    # fingertips sit +12 mm above 0.098 m rubber feet -> clears sink rim by +11 mm!
PULL_STROKE = 0.098                    # +98 mm per side -> total ext = 196 mm (536 mm span across 455 mm basin!)
SEAT_CMD_XY = np.array([0.451, +0.005], float)  # centers basket squarely at [0.435, 0.000] over basin drain

LEFT_PARK = np.array([0.13, +0.30, TOP + 0.25])
RIGHT_PARK = np.array([0.13, -0.30, TOP + 0.25])


def _bar_tcp(center_xy, z_col_bottom: float, ext: float, side: str, back: float = 0.0):
    sy = +1.0 if side == "left" else -1.0
    x_off = +GRASP_X_OFF if side == "left" else -GRASP_X_OFF
    p_world = np.array([
        float(center_xy[0]) + x_off,
        float(center_xy[1]) + sy * (HANDLE_BAR_Y + ext),
        float(z_col_bottom) + HANDLE_BAR_Z,
    ], float) - back * _appr(R_GRIP)
    return _tcp(p_world, R_GRIP, past=HANDLE_PAST)


def demo():
    _ct["wins"].clear()
    pl = _Plan()
    z0 = TOP + 0.0003
    # Suspended colander bottom height when corner rubber feet (z_local=0.098 m) rest on z_rim = TOP + 0.125 m:
    z_seated = TOP + SINK_H - FOOT_BOTTOM_Z
    # Commanded bottom Z during carry clears TOP + 0.125 m front sink apron by +38 mm (+22 mm after ~16 mm PD droop!)
    z_carry = TOP + SINK_H + 0.038

    # Phase 1: Align & robot-encoder PD-droop trim onto both rubberized U-handle bars at x = 0.165 m, y = +-0.160 m
    p_l_pre = _bar_tcp(COL_START_XY, z0, 0.0, "left", back=0.055)
    p_r_pre = _bar_tcp(COL_START_XY, z0, 0.0, "right", back=0.055)
    p_l_grasp = _bar_tcp(COL_START_XY, z0, 0.0, "left", back=0.0)
    p_r_grasp = _bar_tcp(COL_START_XY, z0, 0.0, "right", back=0.0)

    pl.phase(
        left=[(1.1, p_l_pre, HANDLE_OPEN, "align above left handle", O_GRIP)],
        right=[(1.1, p_r_pre, HANDLE_OPEN, "align above right handle", O_GRIP)],
    )
    t_obs = pl.phase(
        left=[(0.7, _Dyn(p_l_pre, "l_h"), HANDLE_OPEN, "trim onto left handle", O_GRIP)],
        right=[(0.7, _Dyn(p_r_pre, "r_h"), HANDLE_OPEN, "trim onto right handle", O_GRIP)],
    )
    pl.window(
        t_obs + 0.15,
        t_obs + 0.70,
        "l_h",
        lambda e, _p=p_l_pre: _p - _live_tcp(e, "left_robot"),
    )
    pl.window(
        t_obs + 0.15,
        t_obs + 0.70,
        "r_h",
        lambda e, _p=p_r_pre: _p - _live_tcp(e, "right_robot"),
    )

    pl.phase(
        left=[
            (0.65, _Dyn(p_l_grasp, "l_h"), HANDLE_OPEN, "seat Y-jaws around left handle", O_GRIP),
            (0.45, _Dyn(p_l_grasp, "l_h"), 0.0, "clamp left rubberized handle", O_GRIP),
        ],
        right=[
            (0.65, _Dyn(p_r_grasp, "r_h"), HANDLE_OPEN, "seat Y-jaws around right handle", O_GRIP),
            (0.45, _Dyn(p_r_grasp, "r_h"), 0.0, "clamp right rubberized handle", O_GRIP),
        ],
    )

    # Phase 2: Pull both telescoping stainless slide handles outward (+98 mm/side -> 536 mm total span > 455 mm basin!)
    p_l_ext = _bar_tcp(COL_START_XY, z0, PULL_STROKE, "left", back=0.0)
    p_r_ext = _bar_tcp(COL_START_XY, z0, PULL_STROKE, "right", back=0.0)
    pl.phase(
        left=[(1.3, _Dyn(p_l_ext, "l_h"), 0.0, "pull left slide handle out +98 mm", O_GRIP)],
        right=[(1.3, _Dyn(p_r_ext, "r_h"), 0.0, "pull right slide handle out -98 mm", O_GRIP)],
    )

    # Phase 3: Lift level above sink front wall, carry over x = 0.466 m basin, and seat all 4 downward corner rubber
    # feet gently onto left & right stainless sink deck rims (y = +-248 mm on +-227.5..+-292.5 mm ledges)!
    p_l_lift = _bar_tcp(COL_START_XY, z_carry, PULL_STROKE, "left", back=0.0)
    p_r_lift = _bar_tcp(COL_START_XY, z_carry, PULL_STROKE, "right", back=0.0)
    p_l_over = _bar_tcp(SEAT_CMD_XY, z_carry, PULL_STROKE, "left", back=0.0)
    p_r_over = _bar_tcp(SEAT_CMD_XY, z_carry, PULL_STROKE, "right", back=0.0)
    p_l_seat = _bar_tcp(SEAT_CMD_XY, z_seated - 0.002, PULL_STROKE, "left", back=0.0)
    p_r_seat = _bar_tcp(SEAT_CMD_XY, z_seated - 0.002, PULL_STROKE, "right", back=0.0)

    pl.phase(
        left=[
            (1.1, _Dyn(p_l_lift, "l_h"), 0.0, "lift expanded colander above sink wall", O_GRIP),
            (1.4, _Dyn(p_l_over, "l_h"), 0.0, "carry expanded colander over sink basin", O_GRIP),
            (1.2, _Dyn(p_l_seat, "l_h"), 0.0, "seat left handle rubber feet on left sink rim", O_GRIP),
            (0.45, _Dyn(p_l_seat, "l_h"), 0.0, "settle on sink rims", O_GRIP),
        ],
        right=[
            (1.1, _Dyn(p_r_lift, "r_h"), 0.0, "lift expanded colander above sink wall", O_GRIP),
            (1.4, _Dyn(p_r_over, "r_h"), 0.0, "carry expanded colander over sink basin", O_GRIP),
            (1.2, _Dyn(p_r_seat, "r_h"), 0.0, "seat right handle rubber feet on right sink rim", O_GRIP),
            (0.45, _Dyn(p_r_seat, "r_h"), 0.0, "settle on sink rims", O_GRIP),
        ],
    )

    # Phase 4: Open jaws completely in place, lift straight up +65 mm in +Z clear of handle bars, then park >= 150 mm clear
    p_l_up = _bar_tcp(SEAT_CMD_XY, z_seated + 0.065, PULL_STROKE, "left", back=0.0)
    p_r_up = _bar_tcp(SEAT_CMD_XY, z_seated + 0.065, PULL_STROKE, "right", back=0.0)
    pl.phase(
        left=[
            (0.65, _Dyn(p_l_seat, "l_h"), FINGER_OPEN, "open left jaws", O_GRIP),
            (0.25, _Dyn(p_l_seat, "l_h"), FINGER_OPEN, "dwell open", O_GRIP),
            (0.75, _Dyn(p_l_up, "l_h"), FINGER_OPEN, "lift left clear above handle", O_GRIP),
            (1.10, LEFT_PARK, FINGER_OPEN, "park left clear", O_GRIP),
            (2.00, LEFT_PARK, FINGER_OPEN, "hold settled", O_GRIP),
        ],
        right=[
            (0.65, _Dyn(p_r_seat, "r_h"), FINGER_OPEN, "open right jaws", O_GRIP),
            (0.25, _Dyn(p_r_seat, "r_h"), FINGER_OPEN, "dwell open", O_GRIP),
            (0.75, _Dyn(p_r_up, "r_h"), FINGER_OPEN, "lift right clear above handle", O_GRIP),
            (1.10, RIGHT_PARK, FINGER_OPEN, "park right clear", O_GRIP),
            (2.00, RIGHT_PARK, FINGER_OPEN, "hold settled", O_GRIP),
        ],
    )
    return {"left": pl.L, "right": pl.R}
