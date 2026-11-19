"""Exact 3D geometry, dense frosting-contact latch, and 4-tier scorer for cake-box.

Replaces cake-dome-cover on PR #2318 (TK-QTH3SB):
- Box: 10 x 10 x 5 in (254.0 x 254.0 x 127.0 mm inner) white SBS lock-corner window bakery box
  (BRP Box Shop 245 / Southern Champion Tray 23053 family), modified per tasks/cake-box/fab/ so
  front_panel unfolds flat onto the counter with a declared 38.0 x 36.0 x 1.6 mm +Y lifting ear,
  hinged window lid, and child hinged lid_tab.
- Cake board: 247.6 mm OD x 4.8 mm thick (~55 g) two-ply silver-foil-wrapped corrugated cake circle
  fabricated from 10 in nominal Wilton MFR 191001569 circles using tasks/cake-box/fab/cake_board_template.svg,
  leaving 3.2 mm nominal clearance per side inside the 254.0 mm box.
- Decorated stand-in cake: 8 x 3 in (203.2 x 76.2 mm) round polystyrene foam cake dummy covered in
  2.4 mm pastel rolled fondant (208.0 mm body diameter, 78.6 mm top height) with firm royal-icing
  bottom shell border (108.5 mm outer radius, leaving 15.3 mm exposed board rim), top shell border
  (105.5 mm outer radius, 85.2 mm height), and 8 piped swirl rosettes on top (93.5 mm peak height).
"""
from __future__ import annotations

import math
import pathlib
import struct
import xml.etree.ElementTree as ET
import numpy as np

# 10 x 10 x 5 in lock-corner window bakery box (inner 254.0 x 254.0 x 127.0 mm)
W_IN = 0.2540
D_IN = 0.2540
H_WALL = 0.1270
T_BASE = 0.0010   # 1.0 mm rigid base proxy (with beveled entry lip)
T_WALL = 0.0016   # 1.6 mm rigid panel proxy for folded 24-pt SBS paperboard
TAB_LEN = 0.0350  # 35.0 mm (1.375 in) front tuck tab height
TAB_W_TOP = 0.2440
TAB_W_TIP = 0.2240
WINDOW_SPAN = 0.1740  # 174 x 174 mm clear cellophane window in 254 x 254 mm lid

# Joint anchors in CAKE_BOX local frame (metres)
FRONT_ANCHOR = np.array([-W_IN / 2 - 0.0004, 0.0, 0.0005], dtype=float)             # (-0.1274, 0.0, 0.0005)
LID_ANCHOR = np.array([+W_IN / 2 + T_WALL / 2, 0.0, T_BASE + H_WALL], dtype=float)  # (+0.1278, 0.0, 0.1280)
# Lid front tab crease sits 2.5 mm inside front_panel's inner face (-0.1270 m) when closed
TAB_ANCHOR = np.array([-W_IN / 2 + 0.0025, 0.0, T_BASE + H_WALL], dtype=float)     # (-0.1245, 0.0, 0.1280)

# Initial crease angles (radians)
FRONT_INIT = -math.pi / 2          # -90 deg: folded flat onto the counter toward -X
LID_INIT = math.radians(95.0)      # +95 deg: open, leaning 5 deg back past vertical
TAB_INIT = math.radians(45.0)      # +45 deg: natural pre-scored outward crease angle

# 10 in nominal (9.75 in = 247.6 mm actual) B-flute / double-wall corrugated cake circle (4.8 mm = 3/16 in thick)
R_BOARD = 0.1238
T_BOARD = 0.0048

# 8 x 3 in decorated fondant + royal-icing stand-in cake
R_CAKE_BODY = 0.1040      # 208.0 mm fondant body diameter (104.0 mm radius)
H_CAKE_BODY = 0.0786      # 78.6 mm fondant top height
R_BORDER_BOT = 0.1085     # 108.5 mm bottom royal-icing shell border outer radius (15.3 mm board rim clear)
H_BORDER_BOT = 0.0105     # 10.5 mm bottom shell border height
R_BORDER_TOP = 0.1055     # 105.5 mm top shell border outer radius
H_BORDER_TOP = 0.0852     # 85.2 mm top shell border peak height
N_ROSETTES = 8
R_ROSETTE_RING = 0.0660   # ring radius of the 8 top swirl rosettes
ROSETTE_RAD = 0.0165      # 33 mm diameter swirl rosette
H_ROSETTE_TOP = 0.0935    # 93.5 mm peak height above board top (33.5 mm below closed lid!)

# Scorer thresholds (all numeric, matching discuss/cake-box.md)
FROSTING_CLEAR_M = 0.0010     # < 1.0 mm from any robot hull or box panel latches 0
MAX_CAKE_TILT_DEG = 10.0      # > 10.0 deg cake tilt on board (or absolute) latches 0
MAX_CAKE_SLIDE_M = 0.0100     # > 10.0 mm lateral slide of cake on its board latches 0
MAX_CAKE_LIFT_M = 0.0030      # > 3.0 mm lift of cake off its board latches 0
MAX_BOARD_DROP_M = 0.0200     # > 20.0 mm board lift above counter/base (or below counter) latches 0

# Crushed-box panel stop thresholds (radians)
FRONT_CRUSH_MIN = math.radians(-95.0)
FRONT_CRUSH_MAX = math.radians(10.0)   # pushed inward past vertical stop by > 10 deg
LID_CRUSH_MIN = math.radians(-8.0)     # forced down into cavity past -8 deg
LID_CRUSH_MAX = math.radians(108.0)    # bent backward past +108 deg (lid rear stop is 97 deg)
TAB_CRUSH_MIN = math.radians(-68.0)
TAB_CRUSH_MAX = math.radians(90.0)
MAX_BOX_TILT_DEG = 12.0

# Final-state tier thresholds
BOARD_SEAT_Z_TOL = 0.0035     # board bottom within 3.5 mm of inner box floor
BOARD_IN_MARGIN = 0.0010      # 1.0 mm solver wall-contact / steel-ruler allowance per side (half-extent <= 128.0 mm)
PANEL_CLOSE_DEG = 5.0         # front_panel within 5 deg of vertical; lid within 5 deg of flat
LID_WALL_GAP_TOL = 0.0030     # lid underside within 3.0 mm of the 127.0 mm walls' tops all round
TAB_TUCK_DEPTH_M = 0.0100     # tab free edge >= 10.0 mm behind front_panel's inner face
HAND_CLEAR_M = 0.0800         # all 6 fingertip/pad points >= 80.0 mm from box, board, and cake
STILL_STEPS = 21              # 1.0 s at 20 Hz
STILL_TOL_M = 0.0010          # < 1.0 mm all-pairs 3D movement


def ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rzy(yaw: float, pitch: float) -> np.ndarray:
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    return np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float) @ ry(pitch)


def build_frosting_surface(spacing: float = 0.0032) -> np.ndarray:
    """Dense frosting point cloud in CAKE local frame (bottom resting on board excluded).

    Spaced <= 3.2 mm across the fondant side cylinder, bottom piped shell border,
    top piped shell border, fondant top face, and all 8 piped top rosettes + center medallion.
    """
    pts: list[tuple[float, float, float]] = []

    # 1. Bottom piped royal-icing shell border (z in [0.0015, H_BORDER_BOT], r = R_BORDER_BOT)
    n_bot = int(math.ceil(2.0 * math.pi * R_BORDER_BOT / spacing))
    for z in np.linspace(0.0018, H_BORDER_BOT, 4):
        for a in np.linspace(0.0, 2.0 * math.pi, n_bot, endpoint=False):
            pts.append((R_BORDER_BOT * math.cos(a), R_BORDER_BOT * math.sin(a), float(z)))
    for r in np.linspace(R_CAKE_BODY, R_BORDER_BOT, 3):
        for a in np.linspace(0.0, 2.0 * math.pi, n_bot, endpoint=False):
            pts.append((float(r) * math.cos(a), float(r) * math.sin(a), H_BORDER_BOT))

    # 2. Fondant cylindrical side wall (z in [H_BORDER_BOT, H_CAKE_BODY])
    n_side = int(math.ceil(2.0 * math.pi * R_CAKE_BODY / spacing))
    n_z = max(2, int(math.ceil((H_CAKE_BODY - H_BORDER_BOT) / spacing)) + 1)
    for z in np.linspace(H_BORDER_BOT, H_CAKE_BODY, n_z):
        for a in np.linspace(0.0, 2.0 * math.pi, n_side, endpoint=False):
            pts.append((R_CAKE_BODY * math.cos(a), R_CAKE_BODY * math.sin(a), float(z)))

    # 3. Top piped shell border crown (z in [H_CAKE_BODY, H_BORDER_TOP], r in [0.097, R_BORDER_TOP])
    n_top_b = int(math.ceil(2.0 * math.pi * R_BORDER_TOP / spacing))
    for z in np.linspace(H_CAKE_BODY, H_BORDER_TOP, 3):
        for a in np.linspace(0.0, 2.0 * math.pi, n_top_b, endpoint=False):
            pts.append((R_BORDER_TOP * math.cos(a), R_BORDER_TOP * math.sin(a), float(z)))
    for r in (0.0985, 0.1020, R_BORDER_TOP):
        for a in np.linspace(0.0, 2.0 * math.pi, n_top_b, endpoint=False):
            pts.append((float(r) * math.cos(a), float(r) * math.sin(a), H_BORDER_TOP))

    # 4. Fondant top disc (z = H_CAKE_BODY)
    pts.append((0.0, 0.0, H_CAKE_BODY))
    for r in list(np.arange(spacing, R_CAKE_BODY, spacing)) + [R_CAKE_BODY]:
        m = int(math.ceil(2.0 * math.pi * float(r) / spacing))
        for a in np.linspace(0.0, 2.0 * math.pi, m, endpoint=False):
            pts.append((float(r) * math.cos(a), float(r) * math.sin(a), H_CAKE_BODY))

    # 5. Eight top swirl rosettes + center piped medallion
    rosette_centers = [
        (R_ROSETTE_RING * math.cos(2.0 * math.pi * k / N_ROSETTES),
         R_ROSETTE_RING * math.sin(2.0 * math.pi * k / N_ROSETTES))
        for k in range(N_ROSETTES)
    ] + [(0.0, 0.0)]
    for idx, (cx, cy) in enumerate(rosette_centers):
        rad0 = ROSETTE_RAD if idx < N_ROSETTES else 0.0130
        peak = H_ROSETTE_TOP if idx < N_ROSETTES else (H_CAKE_BODY + 0.0110)
        for frac, z in ((1.0, H_CAKE_BODY + 0.004), (0.78, 0.5 * (H_CAKE_BODY + peak)), (0.42, peak - 0.002)):
            rr = rad0 * frac
            m = max(8, int(math.ceil(2.0 * math.pi * rr / spacing)))
            for a in np.linspace(0.0, 2.0 * math.pi, m, endpoint=False):
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a), float(z)))
        pts.append((cx, cy, float(peak)))

    return np.asarray(pts, dtype=float)


FROSTING_PTS_LOCAL = build_frosting_surface()


def articulated_box_obbs_in_box_frame(q_front: float, q_lid: float, q_tab: float) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (name, center_in_box, rot_in_box, half_extents) for every panel of CAKE_BOX."""
    obbs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    I3 = np.eye(3, dtype=float)

    # Main erected box panels: base, back wall, left wall, right wall
    obbs.append(("box_base", np.array([0.0, 0.0, T_BASE / 2]), I3, np.array([W_IN / 2 + T_WALL, D_IN / 2 + T_WALL, T_BASE / 2])))
    obbs.append(("back_wall", np.array([+W_IN / 2 + T_WALL / 2, 0.0, T_BASE + H_WALL / 2]), I3, np.array([T_WALL / 2, D_IN / 2 + T_WALL, H_WALL / 2])))
    obbs.append(("left_wall", np.array([0.0, +D_IN / 2 + T_WALL / 2, T_BASE + H_WALL / 2]), I3, np.array([W_IN / 2 + T_WALL, T_WALL / 2, H_WALL / 2])))
    obbs.append(("right_wall", np.array([0.0, -D_IN / 2 - T_WALL / 2, T_BASE + H_WALL / 2]), I3, np.array([W_IN / 2 + T_WALL, T_WALL / 2, H_WALL / 2])))

    # Hinged front_panel (rest pose q_front = 0 stands upright at x = -W_IN/2 - T_WALL/2) + outer +Y dust ear
    Rf = ry(q_front)
    c_front_rest = np.array([-W_IN / 2 - T_WALL / 2, 0.0, T_BASE + H_WALL / 2], dtype=float)
    c_front = FRONT_ANCHOR + Rf @ (c_front_rest - FRONT_ANCHOR)
    obbs.append(("front_panel", c_front, Rf, np.array([T_WALL / 2, (D_IN - 0.002) / 2, H_WALL / 2])))
    c_ear_rest = np.array([-W_IN / 2 + 0.0182, +D_IN / 2 + T_WALL + 0.0018, T_BASE + H_WALL - 0.0220], dtype=float)
    c_ear = FRONT_ANCHOR + Rf @ (c_ear_rest - FRONT_ANCHOR)
    obbs.append(("front_ear", c_ear, Rf, np.array([0.0190, 0.0008, 0.0180])))

    # Hinged window lid (rest pose q_lid = 0 lies flat on top of walls at z = T_BASE + H_WALL + T_WALL/2)
    Rl = ry(q_lid)
    c_lid_rest = np.array([0.0, 0.0, T_BASE + H_WALL + T_WALL / 2], dtype=float)
    c_lid = LID_ANCHOR + Rl @ (c_lid_rest - LID_ANCHOR)
    obbs.append(("lid", c_lid, Rl, np.array([(W_IN + 2 * T_WALL) / 2, (D_IN + 2 * T_WALL) / 2, T_WALL / 2])))

    # Child hinged lid_tab (rest pose q_lid = 0, q_tab = 0 hangs straight down -Z from TAB_ANCHOR)
    Rt = ry(q_tab)
    c_tab_rest = np.array([TAB_ANCHOR[0], 0.0, TAB_ANCHOR[2] - TAB_LEN / 2], dtype=float)
    c_tab_lid = TAB_ANCHOR + Rt @ (c_tab_rest - TAB_ANCHOR)
    c_tab = LID_ANCHOR + Rl @ (c_tab_lid - LID_ANCHOR)
    obbs.append(("lid_tab", c_tab, Rl @ Rt, np.array([T_WALL / 2, TAB_W_TOP / 2, TAB_LEN / 2])))

    return obbs


def box_panel_touches_frosting(
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    q_front: float,
    q_lid: float,
    q_tab: float,
    cake_pos: np.ndarray,
    cake_rot: np.ndarray,
    tol: float = FROSTING_CLEAR_M,
) -> str | None:
    """Return panel name if any CAKE_BOX panel comes within `tol` (1.0 mm) of any cake frosting point."""
    surf_w = np.asarray(cake_pos, float) + FROSTING_PTS_LOCAL @ np.asarray(cake_rot, float).T
    surf_box = (surf_w - np.asarray(box_pos, float)) @ np.asarray(box_rot, float)
    for name, c, R, h in articulated_box_obbs_in_box_frame(q_front, q_lid, q_tab):
        if name == "box_base":
            # Cake rests on its 3.2 mm cake_board, which rests on box_base; still check in case cake falls off board!
            pass
        rel = (surf_box - c) @ R
        if np.any(np.all(np.abs(rel) <= (h + tol), axis=1)):
            return name
    return None


# URDF robot hull cache for dense frosting contact checking
_URDF_HULLS: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}


def _read_stl_verts(path: pathlib.Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * n:
            dt = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(float).reshape(-1, 3)
    v = [
        [float(x) for x in ln.split()[1:4]]
        for ln in data.decode("ascii", errors="ignore").splitlines()
        if ln.split()[:1] == ["vertex"]
    ]
    return np.asarray(v, float)


def _rpy_mat(r: list[float]) -> np.ndarray:
    cr, sr = math.cos(r[0]), math.sin(r[0])
    cp, sp = math.cos(r[1]), math.sin(r[1])
    cy, sy = math.cos(r[2]), math.sin(r[2])
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def get_robot_urdf_hulls() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    if not _URDF_HULLS:
        from scipy.spatial import ConvexHull
        from yam_arena import YAM_URDF

        urdf = pathlib.Path(YAM_URDF)
        for link in ET.parse(urdf).getroot().findall("link"):
            for col in link.findall("collision"):
                mesh = col.find("geometry/mesh")
                if mesh is None:
                    continue
                o = col.find("origin")
                R = _rpy_mat([float(x) for x in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()])
                p = np.array([float(x) for x in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()])
                sc = np.array([float(x) for x in (mesh.get("scale") or "1 1 1").split()])
                raw = _read_stl_verts((urdf.parent / mesh.get("filename")).resolve())
                pts = np.unique(np.round((raw * sc) @ R.T + p, 7), axis=0)
                h = ConvexHull(pts)
                _URDF_HULLS.setdefault(link.attrib["name"], []).append((pts[h.vertices], h.equations))
    return _URDF_HULLS


def robot_hulls_touch_frosting(
    world_hulls: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    cake_pos: np.ndarray,
    cake_rot: np.ndarray,
    tol: float = FROSTING_CLEAR_M,
) -> str | None:
    """Return 'arm:link' if any robot collision hull comes within `tol` (1.0 mm) of the cake frosting."""
    p = np.asarray(cake_pos, float)
    R = np.asarray(cake_rot, float)
    surf_w = p + FROSTING_PTS_LOCAL @ R.T
    lo = surf_w.min(axis=0) - tol
    hi = surf_w.max(axis=0) + tol
    for name, verts_w, planes_w, vlo, vhi in world_hulls:
        if np.any(vhi < lo) or np.any(vlo > hi):
            continue
        # 1. Check robot hull vertices against outer frosting cylinder/rosette envelope
        loc = (verts_w - p) @ R
        # Main fondant + border cylinder (r <= R_BORDER_BOT, z in [0.001, H_BORDER_TOP])
        dr_main = np.hypot(loc[:, 0], loc[:, 1]) - R_BORDER_BOT
        dz_main = np.maximum(0.001 - loc[:, 2], loc[:, 2] - H_BORDER_TOP)
        if np.any(np.hypot(np.maximum(dr_main, 0.0), np.maximum(dz_main, 0.0)) <= tol):
            return name
        # Rosette crown cylinder (r <= R_ROSETTE_RING + ROSETTE_RAD, z in [H_CAKE_BODY, H_ROSETTE_TOP])
        dr_ros = np.hypot(loc[:, 0], loc[:, 1]) - (R_ROSETTE_RING + ROSETTE_RAD)
        dz_ros = np.maximum(H_CAKE_BODY - loc[:, 2], loc[:, 2] - H_ROSETTE_TOP)
        if np.any(np.hypot(np.maximum(dr_ros, 0.0), np.maximum(dz_ros, 0.0)) <= tol):
            return name
        # 2. Check dense frosting surface points against convex-hull half-spaces
        if np.any(np.all(surf_w @ planes_w[:, :3].T + planes_w[:, 3] <= tol, axis=1)):
            return name
    return None


def hand_clearance_m(
    pad_pts_world: list[np.ndarray],
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    q_front: float,
    q_lid: float,
    q_tab: float,
    board_pos: np.ndarray,
    board_rot: np.ndarray,
    cake_pos: np.ndarray,
    cake_rot: np.ndarray,
) -> float:
    """Minimum distance (m) from the 6 fingertip/pad points to all box panels, board, and cake."""
    best = float("inf")
    bpos = np.asarray(box_pos, float)
    brot = np.asarray(box_rot, float)
    obbs = articulated_box_obbs_in_box_frame(q_front, q_lid, q_tab)

    pl_pos = np.asarray(board_pos, float)
    pl_rot = np.asarray(board_rot, float)
    ck_pos = np.asarray(cake_pos, float)
    ck_rot = np.asarray(cake_rot, float)

    for pt in pad_pts_world:
        pw = np.asarray(pt, float)
        # Distance to all articulated CAKE_BOX panels
        p_box = brot.T @ (pw - bpos)
        for _, c, R, h in obbs:
            d = np.maximum(np.abs(R.T @ (p_box - c)) - h, 0.0)
            best = min(best, float(np.linalg.norm(d)))
        # Distance to cylindrical CAKE_BOARD
        p_pl = pl_rot.T @ (pw - pl_pos)
        dr_b = max(math.hypot(float(p_pl[0]), float(p_pl[1])) - R_BOARD, 0.0)
        dz_b = max(-float(p_pl[2]), float(p_pl[2]) - T_BOARD, 0.0)
        best = min(best, math.hypot(dr_b, dz_b))
        # Distance to CAKE envelope
        p_ck = ck_rot.T @ (pw - ck_pos)
        dr_c = max(math.hypot(float(p_ck[0]), float(p_ck[1])) - R_BORDER_BOT, 0.0)
        dz_c = max(-float(p_ck[2]), float(p_ck[2]) - H_ROSETTE_TOP, 0.0)
        best = min(best, math.hypot(dr_c, dz_c))
    return best


def tracked_world_points(
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    q_front: float,
    q_lid: float,
    q_tab: float,
    board_pos: np.ndarray,
    board_rot: np.ndarray,
    cake_pos: np.ndarray,
    cake_rot: np.ndarray,
) -> np.ndarray:
    """Non-collinear extremal world points on CAKE_BOX, its 3 hinged panels, CAKE_BOARD, and CAKE."""
    pts: list[np.ndarray] = []
    bpos = np.asarray(box_pos, float)
    brot = np.asarray(box_rot, float)

    # 4 base corners + front_panel top corners + lid front corners + lid_tab tip corners
    for lx, ly in ((-W_IN / 2, -D_IN / 2), (-W_IN / 2, +D_IN / 2), (+W_IN / 2, -D_IN / 2), (+W_IN / 2, +D_IN / 2)):
        pts.append(bpos + brot @ np.array([lx, ly, 0.0]))
    Rf = ry(q_front)
    for ly in (-0.11, +0.11):
        p_rest = np.array([FRONT_ANCHOR[0], ly, T_BASE + H_WALL])
        pts.append(bpos + brot @ (FRONT_ANCHOR + Rf @ (p_rest - FRONT_ANCHOR)))
    Rl = ry(q_lid)
    Rt = ry(q_tab)
    for ly in (-0.10, +0.10):
        p_lid_rest = np.array([TAB_ANCHOR[0], ly, TAB_ANCHOR[2]])
        pts.append(bpos + brot @ (LID_ANCHOR + Rl @ (p_lid_rest - LID_ANCHOR)))
        p_tab_rest = np.array([TAB_ANCHOR[0], ly, TAB_ANCHOR[2] - TAB_LEN])
        p_tab_lid = TAB_ANCHOR + Rt @ (p_tab_rest - TAB_ANCHOR)
        pts.append(bpos + brot @ (LID_ANCHOR + Rl @ (p_tab_lid - LID_ANCHOR)))

    # 3 points on CAKE_BOARD + 4 points on CAKE
    pl_pos = np.asarray(board_pos, float)
    pl_rot = np.asarray(board_rot, float)
    for a in (0.0, 2.094, 4.189):
        pts.append(pl_pos + pl_rot @ np.array([R_BOARD * math.cos(a), R_BOARD * math.sin(a), T_BOARD / 2]))
    ck_pos = np.asarray(cake_pos, float)
    ck_rot = np.asarray(cake_rot, float)
    for a in (0.0, 2.094, 4.189):
        pts.append(ck_pos + ck_rot @ np.array([R_CAKE_BODY * math.cos(a), R_CAKE_BODY * math.sin(a), H_CAKE_BODY]))
    pts.append(ck_pos + ck_rot @ np.array([0.0, 0.0, H_ROSETTE_TOP]))
    return np.asarray(pts, dtype=float)


def step_violation_metrics(
    table_top: float,
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    q_front: float,
    q_lid: float,
    q_tab: float,
    board_pos: np.ndarray,
    board_rot: np.ndarray,
    cake_pos: np.ndarray,
    cake_rot: np.ndarray,
    cake_on_board_init: np.ndarray,
) -> dict:
    """Per-step whole-trial safety checks (cake tilt/slide on board, board drop, box crush)."""
    bpos = np.asarray(box_pos, float)
    brot = np.asarray(box_rot, float)
    pl_pos = np.asarray(board_pos, float)
    pl_rot = np.asarray(board_rot, float)
    ck_pos = np.asarray(cake_pos, float)
    ck_rot = np.asarray(cake_rot, float)

    # 1. Cake relative to board: tilt <= 10 deg and slide <= 10 mm and lift <= 3 mm
    rel_pos = pl_rot.T @ (ck_pos - pl_pos)
    cake_slide_m = float(math.hypot(rel_pos[0] - cake_on_board_init[0], rel_pos[1] - cake_on_board_init[1]))
    cake_lift_m = abs(float(rel_pos[2] - cake_on_board_init[2]))
    cos_rel = max(-1.0, min(1.0, float(np.dot(ck_rot[:, 2], pl_rot[:, 2]))))
    cake_rel_tilt_deg = math.degrees(math.acos(cos_rel))
    cake_abs_tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(ck_rot[2, 2])))))
    cake_disturbed = (
        cake_rel_tilt_deg > MAX_CAKE_TILT_DEG
        or cake_abs_tilt_deg > MAX_CAKE_TILT_DEG
        or cake_slide_m > MAX_CAKE_SLIDE_M
        or cake_lift_m > MAX_CAKE_LIFT_M
    )

    # 2. Board drop check: board bottom must stay within [-20, +20] mm of counter/box-base support
    # Sample 8 perimeter points on bottom of board
    ring_w = [
        pl_pos + pl_rot @ np.array([R_BOARD * math.cos(a), R_BOARD * math.sin(a), 0.0])
        for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    ]
    z_min_board = min(float(p[2]) for p in ring_w)
    z_max_board = max(float(p[2]) for p in ring_w)
    support_top = table_top + T_BASE
    board_dropped = (z_min_board < table_top - MAX_BOARD_DROP_M) or (z_max_board > support_top + MAX_BOARD_DROP_M)

    # 3. Box crushed check: any panel forced past its stop or box knocked/tilted off counter
    box_tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(brot[2, 2])))))
    box_crushed = (
        q_front < FRONT_CRUSH_MIN
        or q_front > FRONT_CRUSH_MAX
        or q_lid < LID_CRUSH_MIN
        or q_lid > LID_CRUSH_MAX
        or q_tab < TAB_CRUSH_MIN
        or q_tab > TAB_CRUSH_MAX
        or box_tilt_deg > MAX_BOX_TILT_DEG
        or float(bpos[2]) < table_top - MAX_BOARD_DROP_M
    )

    return {
        "cake_slide_mm": round(cake_slide_m * 1000.0, 3),
        "cake_rel_tilt_deg": round(cake_rel_tilt_deg, 3),
        "cake_abs_tilt_deg": round(cake_abs_tilt_deg, 3),
        "cake_disturbed": bool(cake_disturbed),
        "board_z_max_above_table_mm": round((z_max_board - table_top) * 1000.0, 3),
        "board_dropped": bool(board_dropped),
        "box_crushed": bool(box_crushed),
    }


def evaluate_final_tier(
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    q_front: float,
    q_lid: float,
    q_tab: float,
    board_pos: np.ndarray,
    board_rot: np.ndarray,
) -> dict:
    """Compute progressive geometric tiers (0.0, 0.25, 0.5, 0.75, 1.0) from live 3D poses."""
    bpos = np.asarray(box_pos, float)
    brot = np.asarray(box_rot, float)
    pl_pos = np.asarray(board_pos, float)
    pl_rot = np.asarray(board_rot, float)

    # Exact analytic 3D projected extent of cylindrical CAKE_BOARD (radius R_BOARD, thickness z in [0, T_BOARD])
    # in CAKE_BOX local frame: eliminates any polygon/perimeter sampling gap at arbitrary yaw or tilt.
    A = brot.T @ pl_rot
    c0 = brot.T @ (pl_pos - bpos)
    c1 = c0 + T_BOARD * A[:, 2]
    r_proj_x = R_BOARD * math.hypot(float(A[0, 0]), float(A[0, 1]))
    r_proj_y = R_BOARD * math.hypot(float(A[1, 0]), float(A[1, 1]))
    r_proj_z = R_BOARD * math.hypot(float(A[2, 0]), float(A[2, 1]))

    max_abs_x = max(abs(float(c0[0])), abs(float(c1[0]))) + r_proj_x
    max_abs_y = max(abs(float(c0[1])), abs(float(c1[1]))) + r_proj_y
    board_z_err = abs(float(c0[2]) - T_BASE) + r_proj_z
    board_inside = (
        max_abs_x <= (W_IN / 2 + BOARD_IN_MARGIN)
        and max_abs_y <= (D_IN / 2 + BOARD_IN_MARGIN)
        and board_z_err <= BOARD_SEAT_Z_TOL
    )

    front_deg_from_vert = abs(math.degrees(q_front))
    front_closed = front_deg_from_vert <= PANEL_CLOSE_DEG

    Rl = ry(q_lid)
    lid_deg_from_flat = abs(math.degrees(q_lid))
    wall_top_z = T_BASE + H_WALL
    lid_corners_box = [
        LID_ANCHOR + Rl @ (np.array([sx * W_IN / 2, sy * D_IN / 2, wall_top_z]) - LID_ANCHOR)
        for sx in (-1.0, +1.0)
        for sy in (-1.0, +1.0)
    ]
    lid_wall_gap_m = max(abs(float(c[2]) - wall_top_z) for c in lid_corners_box)
    lid_closed = (lid_deg_from_flat <= PANEL_CLOSE_DEG) and (lid_wall_gap_m <= LID_WALL_GAP_TOL)

    # Geometric tab-tuck check from actual 3D poses of front_panel, lid, and lid_tab:
    # Front panel inner plane in box frame has point P_in = FRONT_ANCHOR + Rf @ ([+T_WALL/2, 0, H_WALL/2])
    # and inward unit normal n_in = Rf @ [1, 0, 0].
    Rf = ry(q_front)
    p_front_inner = FRONT_ANCHOR + Rf @ np.array([+T_WALL / 2, 0.0, 0.0])
    n_front_inward = Rf @ np.array([1.0, 0.0, 0.0])
    front_top_z = float((FRONT_ANCHOR + Rf @ np.array([+T_WALL / 2, 0.0, H_WALL]))[2])

    Rt = ry(q_tab)
    tab_tips_box = []
    for ly in (-TAB_W_TIP / 2 + 0.010, 0.0, +TAB_W_TIP / 2 - 0.010):
        p_rest = np.array([TAB_ANCHOR[0], ly, TAB_ANCHOR[2] - TAB_LEN])
        p_lid = TAB_ANCHOR + Rt @ (p_rest - TAB_ANCHOR)
        tab_tips_box.append(LID_ANCHOR + Rl @ (p_lid - LID_ANCHOR))

    inward_depths = [float(np.dot(pt - p_front_inner, n_front_inward)) for pt in tab_tips_box]
    min_inward_depth_m = min(inward_depths)
    max_tip_z = max(float(pt[2]) for pt in tab_tips_box)
    max_tip_abs_y = max(abs(float(pt[1])) for pt in tab_tips_box)

    tab_tucked = (
        min_inward_depth_m >= TAB_TUCK_DEPTH_M
        and max_tip_z <= (front_top_z - 0.0080)
        and max_tip_abs_y <= (D_IN / 2 - 0.0020)
    )

    if not board_inside:
        tier = 0.0
    elif not front_closed:
        tier = 0.25
    elif not lid_closed:
        tier = 0.50
    elif not tab_tucked:
        tier = 0.75
    else:
        tier = 1.00

    return {
        "tier": tier,
        "board_inside": bool(board_inside),
        "board_side_margin_x_mm": round((W_IN / 2 - max_abs_x) * 1000.0, 3),
        "board_side_margin_y_mm": round((D_IN / 2 - max_abs_y) * 1000.0, 3),
        "board_gate_margin_x_mm": round((W_IN / 2 + BOARD_IN_MARGIN - max_abs_x) * 1000.0, 3),
        "board_gate_margin_y_mm": round((D_IN / 2 + BOARD_IN_MARGIN - max_abs_y) * 1000.0, 3),
        "board_z_err_mm": round(board_z_err * 1000.0, 3),
        "front_deg_from_vert": round(front_deg_from_vert, 3),
        "front_closed": bool(front_closed),
        "lid_deg_from_flat": round(lid_deg_from_flat, 3),
        "lid_wall_gap_mm": round(lid_wall_gap_m * 1000.0, 3),
        "lid_closed": bool(lid_closed),
        "tab_inward_depth_mm": round(min_inward_depth_m * 1000.0, 3),
        "tab_tucked": bool(tab_tucked),
    }
