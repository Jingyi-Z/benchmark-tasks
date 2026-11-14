"""Covering and removing a chafing fuel holder from under a commercial buffet stand.

A cold 8-quart stainless buffet chafer stand sits on the bench with its shallow water pan on the upper rim (z=204 mm)
and a loose handled stainless chafing-fuel holder seated inside its lower central burner support ring. The holder's
top-pivoted horizontal snuffer damper starts swung 110 deg open toward -Y, giving the L-shaped assembly a minimum
horizontal caliper width of 168.7 mm across all 0..360 deg yaw angles — wider than the 144 mm clear front leg opening.

Pulling or twisting the open holder forward jams against the front legs, while pushing the snuffer knob one-handed spins
the loose cup inside the burner ring. The robot braces the protruding front handle with its left hand while reaching
through the open right side bay with its right hand to swing the snuffer damper concentric (0 deg, narrowing the width
to 99 mm), then lifts the covered fuel holder out of the burner ring, withdraws it cleanly between the front legs, and
sets it upright inside the stainless bussing pan.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY as _READY  # noqa: E402
from yam_arena import ik as _ik  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 1100

# Colours (sRGB)
STEEL_BRIGHT = (0.78, 0.80, 0.83)
STEEL_MID = (0.60, 0.63, 0.67)
STEEL_DARK = (0.34, 0.37, 0.41)
TIN_GOLD = (0.80, 0.66, 0.28)
WICK_CHARCOAL = (0.20, 0.21, 0.23)
HANDLE_BLACK = (0.18, 0.19, 0.22)
TAB_BRASS = (0.88, 0.72, 0.20)
PAN_SLATE = (0.48, 0.52, 0.57)

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def Rx(a: float) -> np.ndarray:
    return axis_angle_matrix(X, a)


def Ry(a: float) -> np.ndarray:
    return axis_angle_matrix(Y, a)


def Rz(a: float) -> np.ndarray:
    return axis_angle_matrix(Z, a)


# -- 1. Commercial 8-qt stainless chafer stand with upper water pan & lower burner support ring -----------------------
STAND_X = 0.345
RING_FLOOR_H = 0.012
RING_RIM_H = 0.028      # 16 mm retaining lip around 98 x 98 mm inner pocket (+/-4 mm around 90 x 90 mm cup)
RING_INNER_HALF = 0.049
RING_WALL = 0.006
FRONT_LEG_X = -0.098    # world x = 0.247 m
REAR_LEG_X = +0.098     # world x = 0.443 m (188 mm clear side opening x_rel in [-0.094, +0.094])
FRONT_LEG_Y = 0.076     # 8 mm legs -> inner faces at +/-0.072 = exact 144.0 mm clear front opening!
REAR_LEG_Y = 0.098
DECK_CLEAR_Z = 0.204    # bottom of upper water pan (204 mm above table; leaves >112 mm above fuel cup)

STAND_BOXES = [
    # Left & right side base skids (leave front -X corridor y in [-0.072, +0.072] open)
    *[((0.0, sy * 0.087, 0.004), (0.208, 0.026, 0.008), STEEL_DARK) for sy in (-1, 1)],
    # Rear base crossbar & lower ring support spider arms (along +/-Y from side skids to central burner ring)
    ((REAR_LEG_X, 0.0, 0.004), (0.012, 0.160, 0.008), STEEL_DARK),
    *[((0.0, sy * 0.068, 0.006), (0.028, 0.036, 0.012), STEEL_DARK) for sy in (-1, 1)],
    # Central lower burner support ring (98 x 98 mm inner well, floor z=0.012, rim top z=0.028)
    ((0.0, 0.0, RING_FLOOR_H / 2), (2 * (RING_INNER_HALF + RING_WALL), 2 * (RING_INNER_HALF + RING_WALL), RING_FLOOR_H), STEEL_DARK),
    *[((sx * (RING_INNER_HALF + RING_WALL / 2), 0.0, (RING_FLOOR_H + RING_RIM_H) / 2),
       (RING_WALL, 2 * (RING_INNER_HALF + RING_WALL), RING_RIM_H - RING_FLOOR_H), STEEL_MID) for sx in (-1, 1)],
    *[((0.0, sy * (RING_INNER_HALF + RING_WALL / 2), (RING_FLOOR_H + RING_RIM_H) / 2),
       (2 * RING_INNER_HALF, RING_WALL, RING_RIM_H - RING_FLOOR_H), STEEL_MID) for sy in (-1, 1)],
    # Four 8 x 8 mm vertical leg uprights (front pair 144 mm clear inside; rear pair 188 mm clear inside)
    *[((FRONT_LEG_X, sy * FRONT_LEG_Y, 0.106), (0.008, 0.008, 0.196), STEEL_BRIGHT) for sy in (-1, 1)],
    *[((REAR_LEG_X, sy * REAR_LEG_Y, 0.106), (0.008, 0.008, 0.196), STEEL_BRIGHT) for sy in (-1, 1)],
    # Upper 8-qt water pan & rim deck (z_rel 0.204..0.242, 212 x 212 mm pan body + 240 x 228 mm rolled flange)
    ((0.0, 0.0, 0.220), (0.212, 0.212, 0.032), STEEL_MID),
    ((0.0, 0.0, 0.239), (0.240, 0.228, 0.006), STEEL_BRIGHT),
]

CHAFER_STAND = part("chafer_stand", pos=(STAND_X, 0.0, TOP), fixed=True, friction=0.45, boxes=STAND_BOXES)


# -- 2. Loose stainless chafing-fuel holder with 110-deg side-pivoted snuffer damper (253 g total) --------------------
# Local origin (0, 0, 0) is at the bottom center of the 90 x 90 x 60 mm stainless cup:
#   Cup body      : x, y in [-0.045, +0.045], z in [0.000, 0.060]
#   Cold fuel tin : x, y in [-0.038, +0.038], z in [0.006, 0.056], wick collar z in [0.056, 0.060]
#   Front handle  : x in [-0.133, -0.045], y in [-0.015, +0.015], z in [0.038, 0.046] + upright grip block z in [0.046, 0.070]
CUP_W = 0.090
CUP_H = 0.060
HANDLE_GRIP_LOCAL = np.array([-0.114, 0.0, 0.061])

SNUF_ANCHOR = (-0.018, -0.048, 0.062)
SNUF_INIT_RAD = -math.radians(110.0)   # swung 110 deg open out the -Y side bay
SNUF_TAB_LOCAL = np.array([+0.022, -0.018, 0.079])

HOLDER_BOXES = [
    # Cup base & four walls (90 x 90 x 60 mm outer)
    ((0.0, 0.0, 0.004), (CUP_W, CUP_W, 0.008), STEEL_BRIGHT),
    *[((sx * 0.041, 0.0, 0.034), (0.008, CUP_W, 0.052), STEEL_BRIGHT) for sx in (-1, 1)],
    *[((0.0, sy * 0.041, 0.034), (0.074, 0.008, 0.052), STEEL_BRIGHT) for sy in (-1, 1)],
    # Cold spent steel fuel tin & unlit wick boss inside cup
    ((0.0, 0.0, 0.031), (0.074, 0.074, 0.046), TIN_GOLD),
    ((0.0, 0.0, 0.057), (0.022, 0.022, 0.006), WICK_CHARCOAL),
    # Hinge ear on front-right rim (x -0.024..-0.012, y -0.052..-0.045, z 0.046..0.060)
    ((-0.018, -0.0485, 0.053), (0.012, 0.007, 0.014), STEEL_DARK),
    # Protruding flat front handle stem near upper cup rim (x -0.130..-0.045, y -0.015..+0.015, z 0.048..0.056)
    ((-0.0875, 0.0, 0.052), (0.085, 0.030, 0.008), STEEL_MID),
    # Insulated grip neck across 30 mm Y-flanks (x -0.130..-0.098, y -0.015..+0.015, z 0.048..0.074)
    ((-0.114, 0.0, 0.061), (0.032, 0.030, 0.026), HANDLE_BLACK),
    # Flared 42 mm handle end-stop butt (x -0.136..-0.130, y -0.021..+0.021, z 0.047..0.075)
    ((-0.133, 0.0, 0.061), (0.006, 0.042, 0.028), HANDLE_BLACK),
]

SNUFFER_BOXES = [
    # Horizontal 94 x 94 x 6 mm snuffer damper plate (z 0.063..0.069: 3 mm above cup rim so hinge turns cleanly)
    ((0.0, 0.0, 0.066), (0.094, 0.094, 0.006), STEEL_BRIGHT),
]

SNUFFER_CYLS = [
    # Turned cylindrical brass knob stem (dia 18 mm x 49 mm, z 0.069..0.118) + flared dia 28 mm x 6 mm top cap (z 0.118..0.124)
    ((SNUF_TAB_LOCAL[0], SNUF_TAB_LOCAL[1], 0.0935), 0.009, 0.049, TAB_BRASS, "Z"),
    ((SNUF_TAB_LOCAL[0], SNUF_TAB_LOCAL[1], 0.1210), 0.014, 0.006, TAB_BRASS, "Z"),
]

SNUFFER = Joint("snuffer", kind="hinge", axis="Z", anchor=SNUF_ANCHOR, boxes=SNUFFER_BOXES, cylinders=SNUFFER_CYLS,
                lower=-math.radians(118.0), upper=math.radians(1.5),
                stiffness=0.0, target=SNUF_INIT_RAD, damping=0.02, friction=0.010, initial=SNUF_INIT_RAD, mass=0.042)

HOLDER_START = np.array([STAND_X, 0.0, TOP + RING_FLOOR_H + 0.0005])
FUEL_HOLDER = part("fuel_holder", pos=tuple(HOLDER_START), mass=0.215, friction=0.55,
                   boxes=HOLDER_BOXES, joints=[SNUFFER], self_collide=True)


# -- 3. Fixed stainless bussing pan on the left (+Y) ------------------------------------------------------------------
PAN_XY = np.array([0.280, +0.235])
PAN_DX, PAN_DY, PAN_H = 0.220, 0.176, 0.034
PAN_WALL = 0.005

BUSSING_PAN = part(
    "bussing_pan",
    pos=(PAN_XY[0], PAN_XY[1], TOP),
    fixed=True,
    friction=0.55,
    boxes=[
        ((0.0, 0.0, PAN_WALL / 2), (PAN_DX, PAN_DY, PAN_WALL), PAN_SLATE),
        *[((sx * (PAN_DX / 2 - PAN_WALL / 2), 0.0, (PAN_WALL + PAN_H) / 2),
           (PAN_WALL, PAN_DY, PAN_H - PAN_WALL), STEEL_MID) for sx in (-1, 1)],
        *[((0.0, sy * (PAN_DY / 2 - PAN_WALL / 2), (PAN_WALL + PAN_H) / 2),
           (PAN_DX - 2 * PAN_WALL, PAN_WALL, PAN_H - PAN_WALL), STEEL_MID) for sy in (-1, 1)],
    ],
)

OBJECTS = [CHAFER_STAND, FUEL_HOLDER, BUSSING_PAN]


# -- Scoring & continuous safety/validity monitor ---------------------------------------------------------------------
SNUF_SHUT_DEG = 8.0        # snuffer cover within 8 deg of concentric (0 deg)
SNUF_GATE_DEG = 15.0       # holder may not leave burner ring (>25 mm) before snuffer <= 15 deg
RING_RAD_TOL = 0.012       # within 12 mm of ring center for 0.40 partial credit
EARLY_PULL_TOL = 0.025     # >25 mm off ring center while snuffer > 15 deg latches 0
UPRIGHT_TOL_DEG = 5.0      # final upright tolerance
MAX_TILT_DEG = 35.0        # tipping > 35 deg latches 0 (spilling liquid wick fuel)
PAN_INNER_HX = PAN_DX / 2 - PAN_WALL  # 105 mm
PAN_INNER_HY = PAN_DY / 2 - PAN_WALL  # 83 mm
PAN_Z_TOL = 0.008          # lowest corner within 8 mm of bussing pan floor
HANDS_CLEAR = 0.080        # >= 80 mm exact collider-surface clearance from all task objects
SETTLE_STEPS = 20          # 21 samples = 1.0 s at 20 Hz
SETTLE_TOL = 0.001         # < 1 mm 3D Euclidean extremal-point motion

_HOLD_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in HOLDER_BOXES)
_SNUF_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in SNUFFER_BOXES)
_SNUF_CYL_SPECS = tuple((np.asarray(c, float), float(r), 0.5 * float(h)) for c, r, h, *_ in SNUFFER_CYLS)
_STAND_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in STAND_BOXES)
_PAN_SPECS = tuple((np.asarray(c, float), 0.5 * np.asarray(s, float)) for c, s, *_ in BUSSING_PAN.boxes)
_STAND_ORIGIN = np.array([STAND_X, 0.0, TOP], float)
_PAN_ORIGIN = np.array([PAN_XY[0], PAN_XY[1], TOP], float)
_SNUF_ANC_VEC = np.asarray(SNUF_ANCHOR, float)


def _dedup_box_verts(box_list) -> tuple:
    pts, seen = [], set()
    for c, s, *_ in box_list:
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        hx, hy, hz = 0.5 * float(s[0]), 0.5 * float(s[1]), 0.5 * float(s[2])
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    p = (round(cx + sx * hx, 6), round(cy + sy * hy, 6), round(cz + sz * hz, 6))
                    if p not in seen:
                        seen.add(p)
                        pts.append(np.array(p, float))
    return tuple(pts)


_CUP_BASE_CORNERS_LOCAL = (
    np.array([-0.045, -0.045, 0.000], float),
    np.array([-0.045, +0.045, 0.000], float),
    np.array([+0.045, -0.045, 0.000], float),
    np.array([+0.045, +0.045, 0.000], float),
)
_HOLD_CORNERS_LOCAL = _dedup_box_verts(HOLDER_BOXES)
_SNUF_CORNERS_LOCAL = _dedup_box_verts(SNUFFER_BOXES) + (
    np.array([SNUF_TAB_LOCAL[0] - 0.014, SNUF_TAB_LOCAL[1], 0.124], float),
    np.array([SNUF_TAB_LOCAL[0] + 0.014, SNUF_TAB_LOCAL[1], 0.124], float),
    np.array([SNUF_TAB_LOCAL[0], SNUF_TAB_LOCAL[1] - 0.014, 0.124], float),
    np.array([SNUF_TAB_LOCAL[0], SNUF_TAB_LOCAL[1] + 0.014, 0.124], float),
)

_state: dict = {}
_hist: list = []


IMPACT_SPEED = 0.50        # > 0.50 m/s pre-impact downward vertex speed at tabletop/floor contact latches 0
TABLE_X0, TABLE_X1 = -0.19, 0.91
TABLE_Y_HALF = 0.65
FLOOR_Z = -0.0125
CONTACT_TOL = 0.0008
PENETRATION_FLOOR = -0.004
CONTROL_DT = 0.05


@on_reset
def _reset(env):
    _state.clear()
    _state.update(step=0, viol=set(), settled=False, clear_all=False, prev_pts=None)
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


def _min_local_box_dist(ql: np.ndarray, box_specs) -> float:
    return min(float(np.linalg.norm(np.maximum(np.abs(ql - c) - h, 0.0))) for c, h in box_specs)


def _min_local_zcyl_dist(ql: np.ndarray, cyl_specs) -> float:
    return min(
        float(math.hypot(max(math.hypot(float(ql[0] - c[0]), float(ql[1] - c[1])) - r, 0.0),
                         max(abs(float(ql[2] - c[2])) - hh, 0.0)))
        for c, r, hh in cyl_specs
    )


def _min_surf_dist_world(q_world: np.ndarray, fh) -> float:
    q = np.asarray(q_world, float)
    qh = fh.rotation().T @ (q - np.asarray(fh.pos, float))
    d_fh = _min_local_box_dist(qh, _HOLD_SPECS)
    rs = Rz(float(fh.joints["snuffer"]))
    qs = _SNUF_ANC_VEC + rs.T @ (qh - _SNUF_ANC_VEC)
    d_sn = min(_min_local_box_dist(qs, _SNUF_SPECS), _min_local_zcyl_dist(qs, _SNUF_CYL_SPECS))
    d_st = _min_local_box_dist(q - _STAND_ORIGIN, _STAND_SPECS)
    d_pn = _min_local_box_dist(q - _PAN_ORIGIN, _PAN_SPECS)
    return min(d_fh, d_sn, d_st, d_pn)


def _holder_extremal_3d(fh) -> np.ndarray:
    rs = Rz(float(fh.joints["snuffer"]))
    h_pts = [fh.to_world(p) for p in _HOLD_CORNERS_LOCAL]
    s_pts = [fh.to_world(_SNUF_ANC_VEC + rs @ (p - _SNUF_ANC_VEC)) for p in _SNUF_CORNERS_LOCAL]
    return np.asarray(h_pts + s_pts, float)


def _in_burner_ring(pos: np.ndarray) -> bool:
    return float(np.hypot(pos[0] - STAND_X, pos[1])) <= EARLY_PULL_TOL and float(pos[2]) >= TOP + 0.008


def _in_bussing_pan(fh) -> bool:
    pts = _holder_extremal_3d(fh)
    cup_base = np.asarray([fh.to_world(p) for p in _CUP_BASE_CORNERS_LOCAL], float)
    z_min = float(np.min(cup_base[:, 2]))
    wholly_in = bool(np.all(np.abs(pts[:, 0] - PAN_XY[0]) <= PAN_INNER_HX)
                     and np.all(np.abs(pts[:, 1] - PAN_XY[1]) <= PAN_INNER_HY))
    return wholly_in and z_min <= TOP + PAN_WALL + PAN_Z_TOL


@on_step
def _watch(env):
    S = labkit._SPECS
    st = _state
    st["step"] += 1
    dt = float(getattr(getattr(env, "unwrapped", None), "step_dt", CONTROL_DT))
    fh = labkit._part_state(env, S["fuel_holder"])
    p = np.asarray(fh.pos, float)
    snuf_deg = abs(math.degrees(float(fh.joints["snuffer"])))
    pts = _holder_extremal_3d(fh)

    if st["step"] > 5:
        if fh.tilt_deg() > MAX_TILT_DEG:
            st["viol"].add("holder_tilted_spill")
        if float(np.hypot(p[0] - STAND_X, p[1])) > EARLY_PULL_TOL and snuf_deg > SNUF_GATE_DEG:
            st["viol"].add("withdrawn_before_snuffing")
        prev_pts = st.get("prev_pts")
        if prev_pts is not None and dt > 0.0:
            fd_down = (prev_pts[:, 2] - pts[:, 2]) / dt
            vel_pts = np.asarray(fh.lin_vel, float)[None, :] + np.cross(
                np.asarray(fh.ang_vel, float)[None, :], pts - p[None, :]
            )
            down_now = -vel_pts[:, 2]
            prev_down = st.get("prev_down", down_now)
            for i in range(len(pts)):
                x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
                on_table = (
                    (TABLE_X0 <= x <= TABLE_X1)
                    and (abs(y) <= TABLE_Y_HALF)
                    and (PENETRATION_FLOOR <= (z - TOP) <= CONTACT_TOL)
                )
                on_floor = z <= (FLOOR_Z + CONTACT_TOL)
                if on_table or on_floor:
                    z_surf = FLOOR_Z if on_floor else TOP
                    inc = max(float(down_now[i]), float(fd_down[i]))
                    prev_z = float(prev_pts[i, 2])
                    if prev_z > (z_surf + CONTACT_TOL) and (prev_z - float(prev_down[i]) * dt) <= (z_surf + CONTACT_TOL):
                        inc = max(inc, float(prev_down[i]))
                    if inc > IMPACT_SPEED:
                        st["viol"].add("dropped_impact")
                        break
            st["prev_down"] = down_now
        else:
            vel_pts = np.asarray(fh.lin_vel, float)[None, :] + np.cross(
                np.asarray(fh.ang_vel, float)[None, :], pts - p[None, :]
            )
            st["prev_down"] = -vel_pts[:, 2]
    st["prev_pts"] = pts

    st["clear_all"] = all(_min_surf_dist_world(tp, fh) >= HANDS_CLEAR for tp in _tips(env))

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

    fh = labkit._part_state(env, labkit._SPECS["fuel_holder"])
    p = np.asarray(fh.pos, float)
    snuf_deg = abs(math.degrees(float(fh.joints["snuffer"])))
    if snuf_deg > SNUF_SHUT_DEG or fh.tilt_deg() > UPRIGHT_TOL_DEG:
        return 0.0

    if _in_bussing_pan(fh):
        return 1.0

    if float(np.hypot(p[0] - STAND_X, p[1])) <= RING_RAD_TOL and abs(float(p[2]) - (TOP + RING_FLOOR_H)) <= 0.005:
        return 0.40

    return 0.0


SUCCESS_HOLD_S = 1.0


def score(objects) -> float:
    e = labkit.env()
    return float(task_score(e)) if e is not None else 0.0


def success(objects) -> bool:
    return score(objects) == 1.0


# -- Arbitrary-3D-orientation Waypoint monkeypatch + live visual trim -------------------------------------------------
TIP_ABOVE_TCP = 0.013


class Orient(float):
    """Waypoint yaw carrying a full 3x3 TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_cf", False):
    def _grip_rot(yaw=0.0, _prev=_prev_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rot._cf = True
    _demo.grip_rotation = _grip_rot


def _ik_err(pe, re):
    return pe * 1000.0 + math.degrees(re)


def _endpoints(plan):
    rng = np.random.default_rng(7)
    seeds = [np.array(_READY, float)] + [np.array(_READY, float) + rng.normal(0.0, 0.8, 6) for _ in range(10)]
    q_prev, out, switch = plan.q.copy(), [], set()
    for i, w in enumerate(plan.waypoints):
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, q_prev, base_pos=plan.base_pos, iters=300)
        if _ik_err(pe, re) > 1.5:
            switch.add(i)
            best = (_ik_err(pe, re), 0.0, q)
            for s in [q_prev + rng.normal(0.0, 0.3, 6) for _ in range(4)] + seeds:
                qs, ps, rs = _ik.solve(pos, quat, s, base_pos=plan.base_pos, iters=200)
                e, d = _ik_err(ps, rs), float(np.max(np.abs(qs - q_prev)))
                if (e < 1.0 and (best[0] >= 1.0 or d < best[1])) or (best[0] >= 1.0 and e < best[0]):
                    best = (e, d, qs)
            q = best[2]
        elif (float(np.max(np.abs(q[3:5] - q_prev[3:5]))) > math.radians(70.0)
              or float(np.max(np.abs(q - q_prev))) > math.radians(100.0)):
            switch.add(i)
        out.append(q)
        q_prev = q
    return out, switch


def _seg_idx(plan, t):
    acc = 0.0
    for i, w in enumerate(plan.waypoints):
        if t < acc + w.duration:
            return i, _ik.smoothstep((t - acc) / w.duration)
        acc += w.duration
    return len(plan.waypoints) - 1, 1.0


def _action7(self, t):
    _demo._plan_clock = float(t)
    if getattr(self, "_qends", None) is None:
        self._qends, self._switch = _endpoints(self)
    pos, rot, finger, label = self.target(t)
    quat = _ik.matrix_to_quat_xyzw(rot)
    i, s = _seg_idx(self, t)
    q0 = self._qends[i - 1] if i > 0 else self._q_start
    seed = q0 + s * (self._qends[i] - q0)
    if i in self._switch:
        self.q = seed
        grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
        return np.r_[seed, grip], label
    q, pe, re = _ik.solve(pos, quat, self.q, base_pos=self.base_pos, iters=80)
    if _ik_err(pe, re) > 2.0:
        q2, p2, r2 = _ik.solve(pos, quat, seed, base_pos=self.base_pos, iters=80)
        if _ik_err(p2, r2) < _ik_err(pe, re) - 1.0:
            q, pe, re = q2, p2, r2
    self.q = q
    self.worst_ik_mm = max(self.worst_ik_mm, pe * 1000)
    self.worst_ik_deg = max(self.worst_ik_deg, float(np.degrees(re)))
    grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
    return np.r_[q, grip], label


_prev_init = _demo.ArmPlan.__init__
if not getattr(_prev_init, "_cf", False):
    def _init(self, *a, _prev=_prev_init, **k):
        _prev(self, *a, **k)
        self._q_start, self._qends = self.q.copy(), None

    _init._cf = True
    _demo.ArmPlan.__init__ = _init
    _demo.ArmPlan.action7 = _action7


def _appr(R: np.ndarray) -> np.ndarray:
    return R @ _prev_grip(0.0).T @ np.array([0.0, 0.0, -1.0])


def _tcp(tip, R: np.ndarray, past: float = 0.006) -> np.ndarray:
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


def _seen_handle_world(env) -> np.ndarray:
    fh = labkit._part_state(env, labkit._SPECS["fuel_holder"])
    return np.asarray(fh.to_world(HANDLE_GRIP_LOCAL), float)


def _seen_tab_world(env) -> np.ndarray:
    fh = labkit._part_state(env, labkit._SPECS["fuel_holder"])
    rs = Rz(float(fh.joints["snuffer"]))
    anc = np.asarray(SNUF_ANCHOR, float)
    return np.asarray(fh.to_world(anc + rs @ (SNUF_TAB_LOCAL - anc)), float)


def _live_tcp(env, robot_name: str = "right_robot") -> np.ndarray:
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
        rname = "left_robot" if key == "hnd" else "right_robot"
        p = _live_tcp(env, rname)
        last = _ct["last"].get(key)
        _ct["last"][key] = p
        if last is None or np.max(np.abs(p - last)) > 0.0020:
            continue
        e = np.asarray(fn(env), float)
        tr = _ct["trim"].get(key, np.zeros(3))
        _ct["trim"][key] = np.clip(tr + np.clip(0.35 * e, -0.0025, 0.0025), -0.035, 0.035)


class _Dyn:
    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)


# -- Grasp & mechanism pose helpers -----------------------------------------------------------------------------------
DOWN = _prev_grip(0.0)

# 1. Left-arm front handle grasp outside the stand (across 30 mm Y-width at x_world=0.231 m, pitched 22 deg away from stand)
HND_OPEN = -0.030   # 60 mm opening clears 42 mm flared handle end-butt by 9 mm/side
HND_HOLD = 0.0
HND_PAST = 0.020    # 20 mm deep jaw bite across the 32 x 30 x 26 mm insulated grip neck


def _handle_tcp(holder_origin, yaw_deg: float = 0.0, lift: float = 0.0, nose_up_deg: float = 0.0):
    Rz_m = Rz(math.radians(yaw_deg))
    R = Rz_m @ Ry(-math.radians(22.0 + nose_up_deg)) @ DOWN
    p_pad = np.asarray(holder_origin, float) + Rz_m @ HANDLE_GRIP_LOCAL + lift * Z
    return _tcp(p_pad, R, past=HND_PAST), Orient(R)


# 2. Right-arm snuffer knob grasp entering cleanly through the 188 mm -Y side bay (pitched 64 deg so wrist stays <145 mm!)
TAB_OPEN = -0.030   # 60 mm opening clears dia 28 mm brass mushroom cap by 16 mm/side
TAB_HOLD = 0.0
TAB_PAST = 0.016    # centers dia 18 mm cylindrical stem squarely on the 26 mm flat distal pads


def _snuf_tab_world_nominal(deg_rad: float) -> np.ndarray:
    rs = Rz(deg_rad)
    anc = np.asarray(SNUF_ANCHOR, float)
    return HOLDER_START + anc + rs @ (SNUF_TAB_LOCAL - anc)


def _snuf_pose(deg: float, back: float = 0.0, past: float = TAB_PAST):
    """deg in [-110, +14]: right hand enters along +Y through open side bay and sweeps knob stem from -110 deg to 0 deg."""
    rad = math.radians(deg)
    frac = min(max((deg + 110.0) / 110.0, 0.0), 1.15)
    yaw = 82.0 - 14.0 * frac
    R = Rz(math.radians(yaw)) @ Ry(-math.radians(64.0)) @ DOWN
    a = _appr(R)
    p_tab = _snuf_tab_world_nominal(rad) - 0.008 * frac * Z - back * a
    return _tcp(p_tab, R, past=past), Orient(R)


LEFT_WAIT = np.array([0.16, +0.24, TOP + 0.21])
RIGHT_WAIT = np.array([0.18, -0.26, TOP + 0.21])
LEFT_PARK = np.array([0.12, +0.30, TOP + 0.25])
RIGHT_PARK = np.array([0.14, -0.28, TOP + 0.24])


def demo():
    _ct["wins"].clear()
    pl = _Plan()
    o_wait = Orient(Ry(-math.radians(35.0)) @ DOWN)

    # Phase 1: Left arm aligns above front handle, takes closed-loop trim while right arm stages outside -Y bay, then braces!
    p_h_clear = np.array([0.160, +0.150, TOP + 0.195])
    p_h_pre, o_h0 = _handle_tcp(HOLDER_START, lift=0.050)
    p_h_grasp, _ = _handle_tcp(HOLDER_START, lift=0.0)
    p_s_stage = np.array([0.340, -0.230, TOP + 0.135])
    p_s_pre, o_s110 = _snuf_pose(-110.0, back=0.055)

    pl.phase(
        left=[
            (0.8, p_h_clear, HND_OPEN, "clear bussing_pan", o_h0),
            (0.9, p_h_pre, HND_OPEN, "above fuel_holder handle", o_h0),
        ],
        right=[
            (1.7, p_s_stage, TAB_OPEN, "stage outside -Y side bay", o_s110),
        ],
    )

    a_h0 = _appr(o_h0.rot)
    hnd_pre_off = HND_PAST - TIP_ABOVE_TCP
    th = pl.phase(
        left=[(1.1, _Dyn(p_h_pre, "hnd"), HND_OPEN, "visual look at fuel_holder handle", o_h0)],
        right=[(1.1, p_s_pre, TAB_OPEN, "align short of snuffer tab", o_s110)],
    )
    pl.window(th + 0.30, th + 1.1, "hnd", lambda env, _a=a_h0, _k=hnd_pre_off: (_seen_handle_world(env) + 0.050 * Z + _k * _a) - _live_tcp(env, "left_robot"))

    # Phase 1b: Left arm descends onto trimmed handle and braces while right arm takes closed-loop look at snuffer post
    a_110 = _appr(o_s110.rot)
    pre_tcp_off = TAB_PAST - TIP_ABOVE_TCP - 0.055
    tw = pl.phase(
        left=[
            (0.9, _Dyn(p_h_grasp, "hnd"), HND_OPEN, "descend onto fuel_holder handle", o_h0),
            (0.6, _Dyn(p_h_grasp, "hnd"), HND_HOLD, "brace fuel_holder handle", o_h0),
        ],
        right=[(1.5, _Dyn(p_s_pre, "tab"), TAB_OPEN, "visual look at snuffer tab", o_s110)],
    )
    pl.window(tw + 0.35, tw + 1.5, "tab", lambda env, _a=a_110, _k=pre_tcp_off: (_seen_tab_world(env) + _k * _a) - _live_tcp(env, "right_robot"))

    # Phase 2: Right arm grasps snuffer tab deeply and sweeps damper -110 -> 0 deg stop while left arm braces handle!
    p_s110, _ = _snuf_pose(-110.0, back=0.0)
    snuf_arc = []
    for d in (-86.0, -60.0, -34.0, -10.0, +13.0):
        ps, os_d = _snuf_pose(d, back=0.0)
        snuf_arc.append((0.70, _Dyn(ps, "tab"), TAB_HOLD, f"swing snuffer {int(d)} deg", os_d))
    o_s_end = snuf_arc[-1][4]
    p_s0_out, o_s0 = _snuf_pose(0.0, back=0.075)
    p_s0_clear = np.array([0.340, -0.245, TOP + 0.145])

    pl.phase(
        left=[(0.6, _Dyn(p_h_grasp, "hnd"), HND_HOLD, "brace fuel_holder against spin", o_h0)],
        right=[
            (0.9, _Dyn(p_s110, "tab"), TAB_OPEN, "advance onto snuffer tab", o_s110),
            (0.5, _Dyn(p_s110, "tab"), TAB_HOLD, "grasp snuffer tab", o_s110),
            *snuf_arc,
            (0.65, snuf_arc[-1][1], TAB_HOLD, "seat snuffer against concentric stop", o_s_end),
            (0.45, snuf_arc[-1][1], TAB_OPEN, "release closed snuffer tab", o_s_end),
            (0.8, p_s0_out, TAB_OPEN, "withdraw right hand from tab", o_s0),
            (0.8, p_s0_clear, TAB_OPEN, "exit -Y side bay clear of stand leg", o_s0),
            (1.1, RIGHT_PARK, TAB_OPEN, "right park clear", o_wait),
        ],
    )

    # Phase 3: Left arm lifts covered 94 mm fuel holder +48 mm with 10 deg nose-up pitch (~30 mm clear above 16 mm burner ring
    # and >25 mm clear below upper water pan), withdraws -X through 144 mm front legs to open headroom (x=0.172), raises high
    # (+115 mm with 15 deg pitch compensation so cup bottom clears the 34 mm bussing_pan rim by >50 mm!), L-carries over pan, and sets flat!
    p_lift, o_carry0 = _handle_tcp(HOLDER_START + np.array([0.0, 0.0, 0.048]), nose_up_deg=10.0)
    p_exit, o_exit = _handle_tcp(np.array([0.172, 0.0, HOLDER_START[2] + 0.050]), nose_up_deg=10.0)
    p_high_front, o_high = _handle_tcp(np.array([0.178, +0.060, TOP + 0.118]), yaw_deg=4.0, nose_up_deg=15.0)
    pan_target_xy = np.array([PAN_XY[0] + 0.044, PAN_XY[1] + 0.002])
    p_high_pan_front, o_pan_carry = _handle_tcp(np.array([0.215, pan_target_xy[1], TOP + 0.118]), yaw_deg=6.0, nose_up_deg=15.0)
    p_pan_over, _ = _handle_tcp(np.array([pan_target_xy[0], pan_target_xy[1], TOP + 0.115]), yaw_deg=6.0, nose_up_deg=15.0)
    p_pan_set, o_pan_set = _handle_tcp(np.array([pan_target_xy[0], pan_target_xy[1], TOP + PAN_WALL + 0.007]), yaw_deg=6.0, nose_up_deg=8.0)
    p_pan_up, _ = _handle_tcp(np.array([pan_target_xy[0] - 0.020, pan_target_xy[1], TOP + 0.095]), yaw_deg=6.0, nose_up_deg=6.0)

    pl.phase(
        left=[
            (1.1, _Dyn(p_lift, "hnd"), HND_HOLD, "lift covered holder clear of burner ring", o_carry0),
            (1.3, _Dyn(p_exit, "hnd"), HND_HOLD, "withdraw covered holder between 144 mm front legs", o_exit),
            (1.0, _Dyn(p_high_front, "hnd"), HND_HOLD, "raise covered holder above bussing_pan rim height", o_high),
            (1.1, _Dyn(p_high_pan_front, "hnd"), HND_HOLD, "translate laterally in front of bussing_pan", o_pan_carry),
            (1.1, _Dyn(p_pan_over, "hnd"), HND_HOLD, "align covered holder over bussing_pan cavity", o_pan_carry),
            (1.4, _Dyn(p_pan_set, "hnd"), HND_HOLD, "set covered holder upright in bussing_pan", o_pan_set),
            (0.55, _Dyn(p_pan_set, "hnd"), HND_OPEN, "release covered holder", o_pan_set),
            (1.0, p_pan_up, HND_OPEN, "lift clear of fuel_holder handle", o_pan_set),
            (1.3, LEFT_PARK, HND_OPEN, "left park clear", o_wait),
        ],
        right=[(0.6, RIGHT_PARK, TAB_OPEN, "right hold parked", o_wait)],
    )

    pl.phase(
        left=[(2.3, LEFT_PARK, HND_OPEN, "left final settle", o_wait)],
        right=[(2.3, RIGHT_PARK, TAB_OPEN, "right final settle", o_wait)],
    )
    return {"left": pl.L, "right": pl.R}
