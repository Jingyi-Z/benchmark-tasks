"""travel-mug-flip-lock: unlock a Zojirushi SM-KHE48 travel mug's safety switch and press its release button to pop open the spring-loaded flip lid."""

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
RUN_STEPS = 900

# Colours (sRGB)
STAINLESS_BODY = (0.74, 0.76, 0.80)
STAINLESS_BASE = (0.24, 0.25, 0.27)
PP_BLACK = (0.15, 0.16, 0.18)
PP_CHARCOAL = (0.24, 0.26, 0.29)
SPOUT_DARK = (0.20, 0.22, 0.25)
SILICONE_GREY = (0.58, 0.61, 0.65)
BTN_SILVER = (0.78, 0.80, 0.84)
LOCK_RED = (0.82, 0.22, 0.20)

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def Rx(a: float) -> np.ndarray:
    return axis_angle_matrix(X, a)


def Ry(a: float) -> np.ndarray:
    return axis_angle_matrix(Y, a)


def Rz(a: float) -> np.ndarray:
    return axis_angle_matrix(Z, a)


# -- 1:1 Zojirushi SM-KHE48 16 oz (0.48 L) Stainless Mug (67 x 73 x 232 mm, 0.550 kg filled with 300 mL beverage) ----
# Upright on station counter at (0.270, 0.000, TOP).
# Common mug-base frame (z = 0..232 mm):
#   - Double-wall vacuum floor at z = 12.0 mm; 61.0 mm ID inner liner (r = 30.5 mm)
#   - 300 mL fill depth = 300,000 / (pi * 30.5^2) = 102.66 mm -> undisturbed liquid plane at z = 114.7 mm
#   - Open drinking spout lip at z = 208.0 mm, r = 30.5 mm -> first overflow tilt = atan((208.0 - 114.7) / 30.5) = 71.9 deg
MUG_XY = np.array([0.270, 0.000], float)
MUG_R_OUTER = 0.0335        # 67.0 mm OD stainless body
LINER_R_INNER = 0.0305      # 61.0 mm ID liner & drinking spout radius
FLOOR_Z = 0.0120            # 12.0 mm double-wall vacuum bottom floor
FILL_PLANE_Z = 0.1147       # 114.7 mm undisturbed 300 mL fill plane
SPOUT_LIP_Z = 0.2080        # 208.0 mm open drinking spout rim
OVERFLOW_RAD = math.atan2(SPOUT_LIP_Z - FILL_PLANE_Z, LINER_R_INNER)  # 1.2548 rad = 71.896 deg

MUG_CYLS = [
    # Polymer coaster base + stainless double-wall body (z = 0..186 mm) + black PP upper collar (z = 186..204 mm) + spout rim (z = 204..208 mm)
    ((0.0, 0.0, 0.0060), MUG_R_OUTER, 0.0120, STAINLESS_BASE, "Z"),
    ((0.0, 0.0, 0.0990), MUG_R_OUTER, 0.1740, STAINLESS_BODY, "Z"),
    ((0.0, 0.0, 0.1950), MUG_R_OUTER, 0.0180, PP_BLACK, "Z"),
    ((0.0, 0.0, 0.2060), LINER_R_INNER, 0.0040, SPOUT_DARK, "Z"),
]
MUG_BOXES = [
    # Rear hinge knuckle boss (+X reaches +0.0360 m; front lock tab reaches -0.0370 m -> 73.0 mm total depth)
    ((+0.0310, 0.0, 0.2010), (0.0100, 0.0240, 0.0140), PP_BLACK),
]

# Sibling Joint 1: Horizontal safety slide-lock switch (axis="Y", 0.0 m locked -> -0.010 m unlocked)
# All shapes in PART FRAME at rest (q = 0); self_collide=True enables direct rigid contact with push_button:
#   - Outer red thumb tab: x = [-0.0395, -0.0355], y = [-0.0030, +0.0070], z = [0.1873, 0.1943]
#   - Bottom connecting runner: x = [-0.0395, -0.0245], y = [0.0000, +0.0070], z = [0.1850, 0.1872]
#   - Inner blocking bar: x = [-0.0295, -0.0245], y = [0.0000, +0.0070], z = [0.1873, 0.1943]
#     Blocks push_button's downward follower lug (x = [-0.0345, -0.0305], y = [+0.0005, +0.0065]) at q_btn = +1.0 mm
#     whenever q_lock > -0.0065 m, and clears cleanly once slid <= -0.0075 m!
SLIDE_LOCK_ANCHOR = (-0.0340, 0.0, 0.1900)
SLIDE_LOCK_BOXES = [
    ((-0.0375, +0.0020, 0.1908), (0.0040, 0.0100, 0.0070), LOCK_RED),
    ((-0.0320, +0.0035, 0.1861), (0.0150, 0.0070, 0.0022), PP_CHARCOAL),
    ((-0.0270, +0.0035, 0.1908), (0.0050, 0.0070, 0.0070), PP_CHARCOAL),
]
SLIDE_LOCK_JOINT = Joint(
    "slide_lock", kind="slide", axis="Y", anchor=SLIDE_LOCK_ANCHOR,
    boxes=SLIDE_LOCK_BOXES,
    lower=-0.0105, upper=0.0005, initial=0.0, target=0.0,
    stiffness=0.0, damping=1.4, friction=0.18, mass=0.008,
)

# Sibling Joint 2: Inward (+X) spring-returned push-button catch (0.0 m rest -> +0.0060 m full press)
#   - Front silver press face: x = [-0.0375, -0.0325], y = [-0.0085, +0.0085], z = [0.1955, 0.2055]
#   - Downward lock-follower lug: x = [-0.0345, -0.0305], y = [+0.0005, +0.0065], z = [0.1880, 0.1940]
#   - Left/right riser cheeks straddling flip_lid's 9 mm tongue: y = [+-0.0061, +-0.0096], z = [0.2055, 0.2125]
#   - Upper lid-retaining catch hook: x = [-0.0290, -0.0250], y = [-0.0096, +0.0096], z = [0.2097, 0.2125]
#     Overlaps flip_lid's [-0.0297, -0.0255] front catch lip by 3.5 mm at q_btn = 0 (2.5 mm at locked q_btn = +1.0 mm)
#     and clears completely once pressed >= +0.0040 m!
PUSH_BTN_ANCHOR = (-0.0330, 0.0, 0.2005)
PUSH_BTN_BOXES = [
    ((-0.0365, 0.0000, 0.2005), (0.0080, 0.0170, 0.0100), BTN_SILVER),
    ((-0.0325, +0.0035, 0.1910), (0.0040, 0.0060, 0.0060), BTN_SILVER),
    ((-0.0305, +0.00785, 0.2090), (0.0110, 0.0035, 0.0070), BTN_SILVER),
    ((-0.0305, -0.00785, 0.2090), (0.0110, 0.0035, 0.0070), BTN_SILVER),
    ((-0.0270, 0.0000, 0.2111), (0.0040, 0.0192, 0.0028), BTN_SILVER),
]
PUSH_BTN_JOINT = Joint(
    "push_button", kind="slide", axis="X", anchor=PUSH_BTN_ANCHOR,
    boxes=PUSH_BTN_BOXES,
    lower=0.0, upper=0.0065, initial=0.0, target=0.0,
    stiffness=60.0, damping=0.8, friction=0.01, mass=0.010,
)

# Sibling Joint 3: Rear-hinged (+X) torsion-spring flip cap (0.0 rad closed -> 2.36 rad = 135.2 deg open stop)
#   - 14 g molded PP cap + silicone stopper (r_com = 32 mm -> peak gravity torque = 0.0044 N*m; I_hinge ~ 2.0e-5 kg*m^2).
#   - Helical wire hinge spring k = 0.045 N*m/rad gives 0.106 N*m closed torque (24x lid gravity, ~0.22 s pop-open time)
#     and 1.77 N upward preload at the 60 mm front catch lip.
#   - Upper cap cylinder: z = [0.2140, 0.2320], r = 0.0330 m
#   - Inner silicone stopper plug: z = [0.2090, 0.2140], r = 0.0180 m at x = +0.0020 m
#   - Rear hinge arm: x = [+0.0230, +0.0340], y = [-0.0080, +0.0080], z = [0.2070, 0.2160]
#   - Front L-shaped latch tongue (9.0 mm Y-width passing with +1.6 mm side clearance between push_button cheeks):
#       vertical stem: x = [-0.0318, -0.0296], y = [-0.0045, +0.0045], z = [0.2065, 0.2145]
#       +X catch lip:  x = [-0.0297, -0.0255], y = [-0.0045, +0.0045], z = [0.2065, 0.2090]
FLIP_LID_ANCHOR = (+0.0320, 0.0, 0.2080)
FLIP_LID_CYLS = [
    ((0.0, 0.0, 0.2230), 0.0330, 0.0180, PP_BLACK, "Z"),
    ((+0.0020, 0.0, 0.2115), 0.0180, 0.0050, SILICONE_GREY, "Z"),
]
FLIP_LID_BOXES = [
    ((+0.0285, 0.0, 0.2115), (0.0110, 0.0160, 0.0090), PP_BLACK),
    ((-0.0307, 0.0, 0.2105), (0.0022, 0.0090, 0.0080), PP_BLACK),
    ((-0.0276, 0.0, 0.20775), (0.0042, 0.0090, 0.0025), PP_BLACK),
]
FLIP_LID_JOINT = Joint(
    "flip_lid", kind="hinge", axis="Y", anchor=FLIP_LID_ANCHOR,
    boxes=FLIP_LID_BOXES, cylinders=FLIP_LID_CYLS,
    lower=0.0, upper=2.36, initial=0.0, target=2.36,
    stiffness=0.045, damping=0.004, friction=0.001, mass=0.014,
)

TRAVEL_MUG = part(
    "travel_mug",
    pos=(float(MUG_XY[0]), float(MUG_XY[1]), TOP),
    mass=0.518,
    friction=0.55,
    boxes=MUG_BOXES,
    cylinders=MUG_CYLS,
    joints=[SLIDE_LOCK_JOINT, PUSH_BTN_JOINT, FLIP_LID_JOINT],
    self_collide=True,
)

OBJECTS = [TRAVEL_MUG]


# -- Enforce spring targets every physics step (lessons.md 4:09 PM) -----------------------------------------------------
@on_step
def _apply_joint_targets(env):
    art = env.unwrapped.scene["travel_mug"]
    art.set_joint_position_target(art.data.default_joint_pos)


# -- Actual collider-vertex tracking & scoring --------------------------------------------------------------------------
HAND_CLEAR = 0.080
STILL_WINDOW = 21
STILL_MAX_DISP = 0.001
DROP_IMPACT_SPEED = 0.50
UPRIGHT_MAX_DEG = 5.0
LID_UNSEAL_RAD = math.radians(10.0)
LID_OPEN_MIN_RAD = math.radians(110.0)
LOCK_CLEAR_MIN = 0.0075
STATION_X_HALF = 0.60
STATION_Y_HALF = 0.60
CONTACT_TOL = 0.0012


def _box_verts(boxes):
    verts = []
    for item in boxes:
        c, sz = np.asarray(item[0], float), np.asarray(item[1], float)
        hx, hy, hz = 0.5 * sz
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for szz in (-1.0, 1.0):
                    verts.append(c + np.array([sx * hx, sy * hy, szz * hz]))
    return np.asarray(verts, float)


def _cyl_verts(cyls):
    verts = []
    for item in cyls:
        c, r, h = np.asarray(item[0], float), float(item[1]), float(item[2])
        for ang in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            for sz in (-0.5 * h, +0.5 * h):
                verts.append(c + np.array([r * math.cos(ang), r * math.sin(ang), sz]))
    return np.asarray(verts, float)


_ROOT_VERTS = np.vstack([_box_verts(MUG_BOXES), _cyl_verts(MUG_CYLS)])
_LOCK_VERTS = _box_verts(SLIDE_LOCK_BOXES)
_BTN_VERTS = _box_verts(PUSH_BTN_BOXES)
_LID_VERTS = np.vstack([_box_verts(FLIP_LID_BOXES), _cyl_verts(FLIP_LID_CYLS)])
_LID_ANCH = np.asarray(FLIP_LID_ANCHOR, float)


def _mug_all_world_verts(mug) -> np.ndarray:
    R_m, p_m = mug.rotation(), mug.pos
    q_lock = float(mug.joints["slide_lock"])
    q_btn = float(mug.joints["push_button"])
    q_lid = float(mug.joints["flip_lid"])
    v_lock = _LOCK_VERTS + np.array([0.0, q_lock, 0.0])
    v_btn = _BTN_VERTS + np.array([q_btn, 0.0, 0.0])
    v_lid = ((_LID_VERTS - _LID_ANCH) @ Ry(q_lid).T) + _LID_ANCH
    local_all = np.vstack([_ROOT_VERTS, v_lock, v_btn, v_lid])
    return (local_all @ R_m.T) + p_m


_state: dict = {}


@on_reset
def _reset_state(env):
    _state.clear()
    _state.update(step=0, prev_pts=None, prev_vel=None, ring=[], viol=set())


def _check_impact(pts: np.ndarray, prev_pts: np.ndarray | None, prev_vel: np.ndarray | None, dt: float) -> np.ndarray | None:
    if prev_pts is None:
        return None
    vel = (pts - prev_pts) / dt
    if prev_vel is not None:
        pre_vz = prev_vel[:, 2]
        for i in range(len(pts)):
            x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            z_prev = float(prev_pts[i, 2])
            on_table = (abs(x - 0.30) <= STATION_X_HALF) and (abs(y) <= STATION_Y_HALF)
            hit_table = on_table and (z_prev > TOP + CONTACT_TOL) and (z <= TOP + CONTACT_TOL)
            hit_floor = (z_prev > CONTACT_TOL) and (z <= CONTACT_TOL)
            fell_off = z < TOP - 0.06
            if fell_off:
                _state["viol"].add("fell_off_counter")
                break
            if (hit_table or hit_floor) and float(pre_vz[i]) < -DROP_IMPACT_SPEED:
                _state["viol"].add("hard_drop_impact")
                break
    return vel


@on_step
def _track_hazards(env):
    if not _state:
        _reset_state(env)
    _state["step"] += 1
    mug = labkit._part_state(env, labkit._SPECS["travel_mug"])
    pts = _mug_all_world_verts(mug)
    q_lid = float(mug.joints["flip_lid"])
    tilt_rad = math.radians(mug.tilt_deg())
    # Open-lid hydrostatic overflow: only once stopper unseals (q_lid >= 10 deg) AND tilt exceeds 71.9 deg
    if q_lid >= LID_UNSEAL_RAD and tilt_rad > OVERFLOW_RAD:
        _state["viol"].add("open_lid_overflow")
    if _state["step"] >= 8:
        dt = float(env.unwrapped.step_dt)
        _state["prev_vel"] = _check_impact(pts, _state["prev_pts"], _state["prev_vel"], dt)
    _state["prev_pts"] = pts
    _state["ring"].append(pts.copy())
    if len(_state["ring"]) > STILL_WINDOW:
        _state["ring"].pop(0)


_ROOT_BOX_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in MUG_BOXES)
_ROOT_CYL_SPECS = tuple((np.asarray(c, float), float(r), 0.5 * float(h)) for c, r, h, *_ in MUG_CYLS)
_LOCK_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in SLIDE_LOCK_BOXES)
_BTN_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in PUSH_BTN_BOXES)
_LID_BOX_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in FLIP_LID_BOXES)
_LID_CYL_SPECS = tuple((np.asarray(c, float), float(r), 0.5 * float(h)) for c, r, h, *_ in FLIP_LID_CYLS)


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


def _surface_dist_mug(q_world: np.ndarray, mug) -> float:
    q_m = mug.to_local(q_world)
    d_root = min(_min_box_dist(q_m, _ROOT_BOX_SPECS), _min_zcyl_dist(q_m, _ROOT_CYL_SPECS))
    q_lock = float(mug.joints["slide_lock"])
    q_btn = float(mug.joints["push_button"])
    q_lid = float(mug.joints["flip_lid"])
    d_lock = _min_box_dist(q_m - np.array([0.0, q_lock, 0.0]), _LOCK_SPECS)
    d_btn = _min_box_dist(q_m - np.array([q_btn, 0.0, 0.0]), _BTN_SPECS)
    q_lid_frame = _LID_ANCH + Ry(-q_lid) @ (q_m - _LID_ANCH)
    d_lid = min(_min_box_dist(q_lid_frame, _LID_BOX_SPECS), _min_zcyl_dist(q_lid_frame, _LID_CYL_SPECS))
    return min(d_root, d_lock, d_btn, d_lid)


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


def _hands_clear_and_still(env, mug) -> tuple[bool, bool, float, float]:
    d_min = min(_surface_dist_mug(pt, mug) for pt in _tips(env))
    clear_ok = d_min >= HAND_CLEAR
    ring = _state.get("ring", [])
    if len(ring) >= STILL_WINDOW:
        p0 = ring[0]
        max_disp = max(float(np.linalg.norm(p - p0, axis=1).max()) for p in ring[1:])
        still_ok = max_disp < STILL_MAX_DISP
    else:
        max_disp = 1.0
        still_ok = False
    return clear_ok, still_ok, d_min, max_disp


def task_score(env) -> float:
    mug = labkit._part_state(env, labkit._SPECS["travel_mug"])
    q_lock = float(mug.joints["slide_lock"])
    q_btn = float(mug.joints["push_button"])
    q_lid = float(mug.joints["flip_lid"])
    tilt_deg = float(mug.tilt_deg())
    on_counter = (
        abs(float(mug.pos[0]) - 0.30) <= STATION_X_HALF
        and abs(float(mug.pos[1])) <= STATION_Y_HALF
        and abs(float(mug.pos[2]) - TOP) <= 0.008
        and tilt_deg <= UPRIGHT_MAX_DEG
    )
    clear_ok, still_ok, d_min, max_disp = _hands_clear_and_still(env, mug)
    unlocked = (-q_lock) >= LOCK_CLEAR_MIN
    lid_open = q_lid >= LID_OPEN_MIN_RAD

    if _state.get("viol"):
        return 0.0
    if not (on_counter and clear_ok and still_ok):
        return 0.0
    if unlocked and lid_open:
        return 1.0
    if unlocked:
        return 0.40
    return 0.0


SUCCESS_HOLD_S = 1.0


def score(objects) -> float:
    e = labkit.env()
    return float(task_score(e)) if e is not None else 0.0


def success(objects) -> bool:
    return score(objects) == 1.0


# -- Arbitrary-3D-orientation Waypoint monkeypatch + robot-encoder PD trim ----------------------------------------------
TIP_ABOVE_TCP = 0.013


class Orient(float):
    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_tmfl", False):
    def _grip_rot(yaw=0.0, _prev=_prev_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rot._tmfl = True
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
            grip_t = float(np.clip(finger_t / FINGER_OPEN, 0.0, 1.0))
            p0 = np.asarray(plan.waypoints[i - 1].pos, float) if i > 0 else plan.start_pos
            pos = (1.0 - s) * p0 + s * np.asarray(w.pos, float)
            q, pe, re = _ik.solve(pos, _ik.matrix_to_quat_xyzw(rot), plan.q, base_pos=plan.base_pos, iters=80)
            plan.worst_ik_mm = max(plan.worst_ik_mm, pe * 1000.0)
            plan.worst_ik_deg = max(plan.worst_ik_deg, math.degrees(re))
            plan.q = q
            return np.append(q, grip_t), label
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
    if os.environ.get("TMFL_DEBUG", "0") == "1" and int(round(t * 20)) % 10 == 0:
        mug = labkit._part_state(env, labkit._SPECS["travel_mug"])
        ql = float(mug.joints["slide_lock"]) * 1000.0
        qb = float(mug.joints["push_button"]) * 1000.0
        qd = math.degrees(float(mug.joints["flip_lid"]))
        rtip = _live_tcp(env, "right_robot") + TIP_ABOVE_TCP * _appr(R_R_BTN)
        rdx = (float(rtip[0]) - float(mug.pos[0])) * 1000.0
        print(
            f"TMFLDBG step={_state.get('step',0)} t={t:.2f} mug={np.round(mug.pos,3)} "
            f"tilt={mug.tilt_deg():.1f} rdx={rdx:.1f}mm q_lock={ql:.2f}mm q_btn={qb:.2f}mm q_lid={qd:.1f}deg "
            f"viol={sorted(_state.get('viol',()))}",
            flush=True,
        )


class _Dyn:
    def __init__(self, tcp, keys):
        self.tcp = np.asarray(tcp, float)
        self.keys = (keys,) if isinstance(keys, str) else tuple(keys)

    def __array__(self, dtype=None, copy=None):
        off = sum((_ct["trim"].get(k, np.zeros(3)) for k in self.keys), np.zeros(3))
        return np.asarray(self.tcp + off, dtype=dtype or float)


DOWN = _prev_grip(0.0)
# Left hand approaches from +Y left side (-82 deg yaw, -86 deg pitch) so its jaws close along +-X:
# one rigid jaw braces the +X rear wall of the stainless mug directly opposite the +X button press,
# while the left wrist/camera stays at y >= +0.14 m completely clear of the -X front lock/button!
R_L = Rz(-math.radians(82.0)) @ Ry(-math.radians(86.0)) @ DOWN
# Closed YAM fingertips have gx = vertical Z (12.8 mm Z-thin: +-6.4 mm) and gy = horizontal Y (26.4 mm: +-13.2 mm),
# so at z_btn = TOP + 0.2015 m the closed tip spans z = 195.1..207.9 mm (+0.8 mm clear above slide_lock, +6.1 mm clear below flip_lid!)
R_R_LOCK = Rz(+math.radians(16.0)) @ Ry(-0.5 * math.pi) @ Rz(+0.5 * math.pi) @ DOWN
R_R_BTN = Rz(+math.radians(14.0)) @ Ry(-0.5 * math.pi) @ Rz(+0.5 * math.pi) @ DOWN
O_L = Orient(R_L)
O_R_LOCK = Orient(R_R_LOCK)
O_R_BTN = Orient(R_R_BTN)
R_PARK_L = Rz(-0.30) @ Ry(-math.radians(55.0)) @ Rz(+0.5 * math.pi) @ DOWN
R_PARK_R = Rz(+0.30) @ Ry(-math.radians(55.0)) @ Rz(+0.5 * math.pi) @ DOWN
O_PARK_L = Orient(R_PARK_L)
O_PARK_R = Orient(R_PARK_R)

MUG_HOLD_FINGER = -0.024   # 48 mm target gap across 67 mm cylinder -> rigid +-X jaw brace against +X button push!
LEFT_PARK = np.array([0.23, +0.27, TOP + 0.19])
RIGHT_PARK = np.array([0.23, -0.27, TOP + 0.19])


def demo():
    _ct["wins"].clear()
    pl = _Plan()
    mode = os.environ.get("TMFL_MODE", "demo")

    mx, my = float(MUG_XY[0]), float(MUG_XY[1])
    z_hold = TOP + 0.135
    z_lock = TOP + 0.1885
    z_btn = TOP + 0.2015

    # past=0.002 seats the jaw pads cleanly across y=0 while keeping the gripper palm clear of the +33.5 mm +Y mug wall
    p_l_pre = _tcp([mx, my, z_hold], R_L, past=0.002) - 0.080 * _appr(R_L)
    p_l_hold = _tcp([mx, my, z_hold], R_L, past=0.002)

    # Right closed-pad tip (12.8 mm Z-thin x 26.4 mm Y-wide):
    p_r_lock_pre = _tcp([mx - 0.0365, my + 0.022, z_lock], R_R_LOCK, past=0.002) - 0.055 * _appr(R_R_LOCK)
    p_r_lock_tch = _tcp([mx - 0.0365, my + 0.022, z_lock], R_R_LOCK, past=0.002)
    p_r_lock_sld = _tcp([mx - 0.0365, my - 0.004, z_lock], R_R_LOCK, past=0.002)

    # +X press from x = -40.5 mm (front button face) to x = -34.5 mm (+6.0 mm press, +1.2 mm clear of -33.5 mm collar across full 26.4 mm pad)
    p_r_btn_pre = _tcp([mx - 0.0410, my + 0.002, z_btn], R_R_BTN, past=0.000) - 0.035 * _appr(R_R_BTN)
    p_r_btn_prs = _tcp([mx - 0.0345, my + 0.002, z_btn], R_R_BTN, past=0.000)

    r_init_o = O_R_BTN if mode == "locked_press" else O_R_LOCK
    pl.phase(
        left=[(1.2, p_l_pre, FINGER_OPEN, "align left before mug body", O_L)],
        right=[(1.2, p_r_btn_pre if mode == "locked_press" else p_r_lock_pre, 0.0, "align right before collar", r_init_o)],
    )
    t_tr = pl.phase(
        left=[(0.6, _Dyn(p_l_pre, "l0"), FINGER_OPEN, "trim left", O_L)],
        right=[(0.6, _Dyn(p_r_btn_pre if mode == "locked_press" else p_r_lock_pre, "r0"), 0.0, "trim right", r_init_o)],
    )
    pl.window(t_tr + 0.15, t_tr + 0.58, "l0", lambda e, _p=p_l_pre: _p - _live_tcp(e, "left_robot"))
    r_ref = p_r_btn_pre if mode == "locked_press" else p_r_lock_pre
    pl.window(t_tr + 0.15, t_tr + 0.58, "r0", lambda e, _p=r_ref: _p - _live_tcp(e, "right_robot"))

    # Left hand clamps mug body low to steady against overturning moment
    pl.phase(
        left=[
            (0.7, _Dyn(p_l_hold, "l0"), FINGER_OPEN, "seat left jaws around mug body", O_L),
            (0.5, _Dyn(p_l_hold, "l0"), MUG_HOLD_FINGER, "steady mug body", O_L),
        ],
    )

    if mode == "locked_press":
        pl.phase(
            left=[(2.5, _Dyn(p_l_hold, "l0"), MUG_HOLD_FINGER, "steady mug during locked press", O_L)],
            right=[
                (0.8, _Dyn(p_r_btn_prs, "r0"), 0.0, "press button while safety lock engaged", O_R_BTN),
                (1.0, _Dyn(p_r_btn_prs, "r0"), 0.0, "hold locked press", O_R_BTN),
                (0.7, _Dyn(p_r_btn_pre, "r0"), 0.0, "retract from locked button", O_R_BTN),
            ],
        )
        return {"left": pl.L, "right": pl.R}

    # Full demo: slide safety lock -10 mm (-Y), roll horizontal 14 mm blade to button, press +6 mm (+X) so flip lid pops open!
    pl.phase(
        left=[(4.9, _Dyn(p_l_hold, "l0"), MUG_HOLD_FINGER, "steady mug during unlock and press", O_L)],
        right=[
            (0.7, _Dyn(p_r_lock_tch, "r0"), 0.0, "touch +Y side of safety switch", O_R_LOCK),
            (1.0, _Dyn(p_r_lock_sld, "r0"), 0.0, "slide safety lock open -10 mm", O_R_LOCK),
            (0.9, _Dyn(p_r_btn_pre, "r0"), 0.0, "step back and roll horizontal blade to button", O_R_BTN),
            (0.8, _Dyn(p_r_btn_prs, "r0"), 0.0, "press release button inward", O_R_BTN),
            (0.6, _Dyn(p_r_btn_prs, "r0"), 0.0, "hold button while flip lid pops open", O_R_BTN),
            (0.9, _Dyn(p_r_btn_pre - 0.035 * _appr(R_R_BTN), "r0"), 0.0, "retract clear of mug collar", O_R_BTN),
        ],
    )

    pl.phase(
        left=[
            (0.5, _Dyn(p_l_hold, "l0"), FINGER_OPEN, "open left jaws off mug", O_L),
            (0.8, _Dyn(p_l_pre, "l0"), FINGER_OPEN, "back left arm clear", O_L),
            (1.1, LEFT_PARK, FINGER_OPEN, "park left arm", O_PARK_L),
            (2.0, LEFT_PARK, FINGER_OPEN, "hold settled", O_PARK_L),
        ],
        right=[
            (1.2, RIGHT_PARK, FINGER_OPEN, "park right arm", O_PARK_R),
            (3.2, RIGHT_PARK, FINGER_OPEN, "hold settled", O_PARK_R),
        ],
    )
    return {"left": pl.L, "right": pl.R}
