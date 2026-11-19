"""Box a decorated cake for a customer without touching the frosting.

Replaces cake-dome-cover on PR #2318 (TK-QTH3SB):
- 10 x 10 x 5 in (254.0 x 254.0 x 127.0 mm inner) white SBS lock-corner window bakery box
  (BRP Box Shop 245 / Southern Champion Tray 23053 family, ~0.146 kg), modified per tasks/cake-box/fab/
  so front_panel unfolds flat onto the counter with a declared 38.0 x 36.0 x 1.6 mm +Y lifting ear,
  hinged window lid (open at the back), and child hinged front tuck tab (lid_tab).
- 247.6 mm OD x 4.8 mm thick (0.055 kg) two-ply silver-foil-wrapped corrugated cake circle fabricated
  from 10 in nominal Wilton MFR 191001569 circles per tasks/cake-box/fab/cake_board_template.svg,
  leaving 3.2 mm nominal wall clearance per side.
- 8 x 3 in (203.2 x 76.2 mm) round polystyrene foam cake dummy covered in pastel rolled fondant
  (208.0 mm body diameter, 78.6 mm height) with firm royal-icing bottom shell border, top shell
  border, and 8 piped swirl rosettes (93.5 mm peak height, 0.300 kg stand-in).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401
import labkit
from labkit import GLASS_COLORS, Joint, MATTE_COLORS, METALLIC_COLORS, Mesh, on_reset, on_step, part
from yam_arena import ik as _ik
import yam_arena.demo as _demo
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, LEFT_BASE_POS, READY as _READY, RIGHT_BASE_POS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

from task_metrics import (
    D_IN,
    FRONT_ANCHOR,
    FRONT_INIT,
    FROSTING_CLEAR_M,
    HAND_CLEAR_M,
    H_BORDER_BOT,
    H_BORDER_TOP,
    H_CAKE_BODY,
    H_ROSETTE_TOP,
    H_WALL,
    LID_ANCHOR,
    LID_INIT,
    N_ROSETTES,
    ROSETTE_RAD,
    R_BOARD,
    R_BORDER_BOT,
    R_BORDER_TOP,
    R_CAKE_BODY,
    R_ROSETTE_RING,
    STILL_STEPS,
    STILL_TOL_M,
    TAB_ANCHOR,
    TAB_INIT,
    TAB_LEN,
    TAB_W_TIP,
    TAB_W_TOP,
    T_BASE,
    T_BOARD,
    T_WALL,
    WINDOW_SPAN,
    W_IN,
    box_panel_touches_frosting,
    evaluate_final_tier,
    get_robot_urdf_hulls,
    hand_clearance_m,
    robot_hulls_touch_frosting,
    step_violation_metrics,
    tracked_world_points,
)

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 1200  # 60 s at 20 Hz
X, Y, Z = np.eye(3)
labkit.CONTACT_OFFSET = 0.0004

# Bakery color palette
BOX_WHITE = (0.96, 0.96, 0.94)
BOX_INNER = (0.92, 0.90, 0.87)
BOX_CREASE = (0.83, 0.81, 0.78)
BOX_GUSSET = (0.90, 0.89, 0.86)
WINDOW_FILM = (0.83, 0.90, 0.94)

BOARD_SILVER = (0.76, 0.79, 0.83)
BOARD_RIM = (0.94, 0.95, 0.96)
BOARD_KRAFT = (0.71, 0.58, 0.42)

FONDANT_BLUSH = (0.91, 0.58, 0.67)
FONDANT_RIBBON = (0.76, 0.34, 0.46)
ICING_IVORY = (0.99, 0.98, 0.95)
ICING_ROSE = (0.84, 0.36, 0.49)
ICING_SAGE = (0.46, 0.73, 0.52)
PEARL_GOLD = (0.95, 0.80, 0.35)

GLASS_COLORS.add(WINDOW_FILM)
METALLIC_COLORS.update({BOARD_SILVER, PEARL_GOLD})
MATTE_COLORS.update({
    BOX_WHITE,
    BOX_INNER,
    BOX_CREASE,
    BOX_GUSSET,
    BOARD_RIM,
    BOARD_KRAFT,
    FONDANT_BLUSH,
    FONDANT_RIBBON,
    ICING_IVORY,
    ICING_ROSE,
    ICING_SAGE,
})

# Fixed starting layout on the bakery counter:
# Open cake box on the left-center angled +28 deg so its folded-down front panel faces toward the robots & counter;
# decorated cake on its 9.75 in corrugated circle waiting unoccluded beside the box on the right half of the counter.
BOX_XY = np.array([0.3550, +0.1550], dtype=float)
BOX_YAW = math.radians(26.0)

UX_BOX = np.array([math.cos(BOX_YAW), math.sin(BOX_YAW)], dtype=float)   # +X_box (slide-in direction)
UY_BOX = np.array([-math.sin(BOX_YAW), math.cos(BOX_YAW)], dtype=float)  # +Y_box (toward left arm)
# Cake starts 280 mm upstream of box center along -X_box and 278 mm right along -Y_box -> (0.2253, -0.2175) m
# (> 27 mm clear of folded-down front_panel, > 65 mm clear of right:base, and pushed purely from outer -Y/-X rims!)
CAKE_START_XY = np.round(BOX_XY - 0.2800 * UX_BOX - 0.2780 * UY_BOX, 4)


def box_local_to_world(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
    xy = BOX_XY + lx * UX_BOX + ly * UY_BOX
    return (float(xy[0]), float(xy[1]), TOP + float(lz))


# -- Procedural high-detail bakery visual & collision mesh builders --------------------------------------------------

def _frustum_ring_mesh(
    z_levels: list[float],
    r_levels: list[float],
    n_seg: int,
    color: tuple[float, float, float],
    center_xy: tuple[float, float] = (0.0, 0.0),
    twist_levels: list[float] | None = None,
    radial_mod=None,
    collide: str = "none",
) -> Mesh:
    cx, cy = center_xy
    pts: list[list[float]] = []
    m_lvl = len(z_levels)
    twists = twist_levels if twist_levels is not None else [0.0] * m_lvl
    for li, (zv, rv, tw) in enumerate(zip(z_levels, r_levels, twists)):
        for si in range(n_seg):
            ang = 2.0 * math.pi * si / n_seg + tw
            rr = rv * (radial_mod(li, si, ang) if radial_mod is not None else 1.0)
            pts.append([cx + rr * math.cos(ang), cy + rr * math.sin(ang), zv])
    bot_idx = len(pts)
    pts.append([cx, cy, z_levels[0]])
    top_idx = len(pts)
    pts.append([cx, cy, z_levels[-1]])

    tris: list[tuple[int, int, int]] = []
    for li in range(m_lvl - 1):
        b0 = li * n_seg
        b1 = (li + 1) * n_seg
        for si in range(n_seg):
            sj = (si + 1) % n_seg
            tris.append((b0 + si, b0 + sj, b1 + sj))
            tris.append((b0 + si, b1 + sj, b1 + si))
    for si in range(n_seg):
        sj = (si + 1) % n_seg
        tris.append((bot_idx, sj, si))
        tris.append((top_idx, (m_lvl - 1) * n_seg + si, (m_lvl - 1) * n_seg + sj))
    return Mesh(points=np.asarray(pts, float), tris=np.asarray(tris, dtype=np.int64), color=color, collide=collide)


def _merge_visual_meshes(meshes: list[Mesh], color: tuple[float, float, float]) -> Mesh:
    all_pts = []
    all_tris = []
    offset = 0
    for m in meshes:
        all_pts.append(m.points)
        all_tris.append(m.tris + offset)
        offset += len(m.points)
    return Mesh(
        points=np.vstack(all_pts),
        tris=np.vstack(all_tris).astype(np.int64),
        color=color,
        collide="none",
    )


def _quad_prism_mesh(verts8: np.ndarray, color: tuple[float, float, float], collide: str = "none") -> Mesh:
    tris = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return Mesh(points=np.asarray(verts8, float), tris=tris, color=color, collide=collide)


def _build_board_meshes() -> tuple[Mesh, ...]:
    # 1.2 mm bottom ski bevel clears the 0.9 mm flat front_panel & 1.0 mm box_base while keeping a tall
    # 3.6 mm vertical kraft wall (z = 1.2..4.8 mm) so +1.3 mm fingertips NEVER ride up onto the board!
    col_mesh = _frustum_ring_mesh(
        z_levels=[0.0, 0.0012, T_BOARD],
        r_levels=[R_BOARD - 0.0030, R_BOARD, R_BOARD],
        n_seg=64,
        color=BOARD_KRAFT,
        collide="hull",
    )
    foil_top = _frustum_ring_mesh(
        z_levels=[T_BOARD - 0.0004, T_BOARD + 0.0001],
        r_levels=[R_BOARD - 0.0012, R_BOARD - 0.0012],
        n_seg=96,
        color=BOARD_SILVER,
        radial_mod=lambda _li, _si, ang: 1.0 - 0.009 * (0.5 + 0.5 * math.cos(36.0 * ang)),
        collide="none",
    )
    doily_ring = _frustum_ring_mesh(
        z_levels=[T_BOARD + 0.0001, T_BOARD + 0.0003],
        r_levels=[R_BORDER_BOT + 0.0055, R_BORDER_BOT + 0.0020],
        n_seg=72,
        color=BOARD_RIM,
        collide="none",
    )
    return (col_mesh, foil_top, doily_ring)


def _piped_shell_puff(cx: float, cy: float, z0: float, yaw: float, length: float, width: float, height: float, color) -> Mesh:
    ca, sa = math.cos(yaw), math.sin(yaw)
    pts = []
    s_vals = [-0.48, -0.25, 0.02, 0.30, 0.56]
    w_scales = [0.55, 1.00, 0.86, 0.52, 0.16]
    h_scales = [0.35, 1.00, 0.88, 0.52, 0.12]
    n_ring = 10
    for sv, ws, hs in zip(s_vals, w_scales, h_scales):
        lx = sv * length
        for k in range(n_ring):
            th = 2.0 * math.pi * k / n_ring
            ly = 0.5 * width * ws * math.cos(th)
            flute = 1.0 + 0.12 * math.cos(5.0 * th)
            lz = z0 + height * hs * (0.5 + 0.5 * math.sin(th)) * flute
            wx = cx + lx * ca - ly * sa
            wy = cy + lx * sa + ly * ca
            pts.append([wx, wy, lz])
    head_i = len(pts)
    pts.append([cx - 0.54 * length * ca, cy - 0.54 * length * sa, z0 + 0.22 * height])
    tail_i = len(pts)
    pts.append([cx + 0.62 * length * ca, cy + 0.62 * length * sa, z0 + 0.05 * height])
    tris = []
    for li in range(len(s_vals) - 1):
        b0, b1 = li * n_ring, (li + 1) * n_ring
        for k in range(n_ring):
            kj = (k + 1) % n_ring
            tris.append((b0 + k, b0 + kj, b1 + kj))
            tris.append((b0 + k, b1 + kj, b1 + k))
    for k in range(n_ring):
        kj = (k + 1) % n_ring
        tris.append((head_i, kj, k))
        tris.append((tail_i, (len(s_vals) - 1) * n_ring + k, (len(s_vals) - 1) * n_ring + kj))
    return Mesh(points=np.asarray(pts, float), tris=np.asarray(tris, dtype=np.int64), color=color, collide="none")


def _swirl_rosette_mesh(cx: float, cy: float, z0: float, rad: float, height: float, color) -> Mesh:
    return _frustum_ring_mesh(
        z_levels=[z0, z0 + 0.28 * height, z0 + 0.58 * height, z0 + 0.84 * height, z0 + height],
        r_levels=[rad * 0.92, rad * 1.00, rad * 0.78, rad * 0.46, rad * 0.12],
        n_seg=24,
        color=color,
        center_xy=(cx, cy),
        twist_levels=[0.0, 0.35, 0.80, 1.35, 1.95],
        radial_mod=lambda _li, _si, ang: 1.0 + 0.11 * math.cos(6.0 * ang),
        collide="none",
    )


def _build_cake_meshes() -> tuple[Mesh, ...]:
    # Keep rigid collider strictly inside the flat fondant cylinder body (up to H_CAKE_BODY - 0.8 mm)
    # so all 36 top crown shells, 8 swirl rosettes, 16 mint leaves, and gold pearls render in full 3D relief!
    # (Scorer checks all 11,650 frosting surface points including every rosette up to H_ROSETTE_TOP = 93.5 mm.)
    collider = _frustum_ring_mesh(
        z_levels=[0.0, H_BORDER_BOT, H_CAKE_BODY - 0.0008],
        r_levels=[R_BORDER_BOT - 0.001, R_CAKE_BODY - 0.0005, R_CAKE_BODY - 0.0045],
        n_seg=64,
        color=FONDANT_BLUSH,
        collide="hull",
    )
    fondant_body = _frustum_ring_mesh(
        z_levels=[0.0002, 0.004, H_CAKE_BODY - 0.0045, H_CAKE_BODY - 0.0012, H_CAKE_BODY],
        r_levels=[R_CAKE_BODY, R_CAKE_BODY, R_CAKE_BODY, R_CAKE_BODY - 0.0016, R_CAKE_BODY - 0.0042],
        n_seg=96,
        color=FONDANT_BLUSH,
        collide="none",
    )
    ribbon_band = _frustum_ring_mesh(
        z_levels=[H_BORDER_BOT - 0.001, H_BORDER_BOT + 0.009],
        r_levels=[R_CAKE_BODY + 0.0005, R_CAKE_BODY + 0.0005],
        n_seg=96,
        color=FONDANT_RIBBON,
        collide="none",
    )

    ivory_parts: list[Mesh] = []
    n_shells = 36
    for k in range(n_shells):
        ang = 2.0 * math.pi * k / n_shells
        tangent = ang + math.pi / 2.0
        r_b = 0.5 * (R_CAKE_BODY + R_BORDER_BOT) - 0.0005
        ivory_parts.append(
            _piped_shell_puff(
                r_b * math.cos(ang), r_b * math.sin(ang), 0.0002,
                tangent, length=0.021, width=0.0105, height=H_BORDER_BOT, color=ICING_IVORY,
            )
        )
        r_t = R_CAKE_BODY - 0.0020
        ivory_parts.append(
            _piped_shell_puff(
                r_t * math.cos(ang), r_t * math.sin(ang), H_CAKE_BODY - 0.0008,
                tangent, length=0.019, width=0.0098, height=(H_BORDER_TOP - H_CAKE_BODY), color=ICING_IVORY,
            )
        )

    rose_parts: list[Mesh] = []
    sage_parts: list[Mesh] = []
    pearl_parts: list[Mesh] = []
    ros_h = H_ROSETTE_TOP - H_CAKE_BODY
    for k in range(N_ROSETTES):
        ang = 2.0 * math.pi * k / N_ROSETTES
        cx = R_ROSETTE_RING * math.cos(ang)
        cy = R_ROSETTE_RING * math.sin(ang)
        if k % 2 == 0:
            ivory_parts.append(_swirl_rosette_mesh(cx, cy, H_CAKE_BODY - 0.0005, ROSETTE_RAD, ros_h, ICING_IVORY))
        else:
            rose_parts.append(_swirl_rosette_mesh(cx, cy, H_CAKE_BODY - 0.0005, ROSETTE_RAD, ros_h, ICING_ROSE))
        for side in (-1.0, +1.0):
            la = ang + side * 0.20
            lx = (R_ROSETTE_RING + 0.013) * math.cos(la)
            ly = (R_ROSETTE_RING + 0.013) * math.sin(la)
            sage_parts.append(
                _piped_shell_puff(lx, ly, H_CAKE_BODY - 0.0003, la + side * 0.9, 0.014, 0.0065, 0.0045, ICING_SAGE)
            )
        pearl_parts.append(
            _frustum_ring_mesh(
                z_levels=[H_ROSETTE_TOP - 0.0015, H_ROSETTE_TOP + 0.0005, H_ROSETTE_TOP + 0.0022],
                r_levels=[0.0018, 0.0028, 0.0012],
                n_seg=12,
                color=PEARL_GOLD,
                center_xy=(cx, cy),
                collide="none",
            )
        )

    rose_parts.append(_swirl_rosette_mesh(0.0, 0.0, H_CAKE_BODY - 0.0005, 0.0135, 0.0110, ICING_ROSE))
    pearl_parts.append(
        _frustum_ring_mesh(
            z_levels=[H_CAKE_BODY + 0.0098, H_CAKE_BODY + 0.0115, H_CAKE_BODY + 0.0130],
            r_levels=[0.0018, 0.0028, 0.0012],
            n_seg=12,
            color=PEARL_GOLD,
            center_xy=(0.0, 0.0),
            collide="none",
        )
    )

    return (
        collider,
        fondant_body,
        ribbon_band,
        _merge_visual_meshes(ivory_parts, ICING_IVORY),
        _merge_visual_meshes(rose_parts, ICING_ROSE),
        _merge_visual_meshes(sage_parts, ICING_SAGE),
        _merge_visual_meshes(pearl_parts, PEARL_GOLD),
    )


def _build_box_main_shapes():
    outer_w = W_IN + 2 * T_WALL
    outer_d = D_IN + 2 * T_WALL
    boxes = [
        ((0.0, 0.0, T_BASE / 2), (W_IN, outer_d, T_BASE), BOX_WHITE),
        ((+W_IN / 2 + T_WALL / 2, 0.0, T_BASE + H_WALL / 2), (T_WALL, outer_d, H_WALL), BOX_WHITE),
        ((0.0, +D_IN / 2 + T_WALL / 2, T_BASE + H_WALL / 2), (outer_w, T_WALL, H_WALL), BOX_WHITE),
        ((0.0, -D_IN / 2 - T_WALL / 2, T_BASE + H_WALL / 2), (outer_w, T_WALL, H_WALL), BOX_WHITE),
    ]

    meshes: list[Mesh] = []
    x0, x1 = -W_IN / 2 - 0.0040, -W_IN / 2 + 0.0010
    y0, y1 = -D_IN / 2, +D_IN / 2
    ramp_v = np.array([
        [x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0],
        [x0, y0, 0.0001], [x1, y0, T_BASE], [x1, y1, T_BASE], [x0, y1, 0.0001],
    ])
    meshes.append(_quad_prism_mesh(ramp_v, BOX_WHITE, collide="hull"))

    liner_v = np.array([
        [-W_IN / 2 + 0.002, -D_IN / 2 + 0.002, T_BASE + 0.0001],
        [+W_IN / 2 - 0.002, -D_IN / 2 + 0.002, T_BASE + 0.0001],
        [+W_IN / 2 - 0.002, +D_IN / 2 - 0.002, T_BASE + 0.0001],
        [-W_IN / 2 + 0.002, +D_IN / 2 - 0.002, T_BASE + 0.0001],
        [-W_IN / 2 + 0.002, -D_IN / 2 + 0.002, T_BASE + 0.0003],
        [+W_IN / 2 - 0.002, -D_IN / 2 + 0.002, T_BASE + 0.0003],
        [+W_IN / 2 - 0.002, +D_IN / 2 - 0.002, T_BASE + 0.0003],
        [-W_IN / 2 + 0.002, +D_IN / 2 - 0.002, T_BASE + 0.0003],
    ])
    meshes.append(_quad_prism_mesh(liner_v, BOX_INNER, collide="none"))

    gusset_parts: list[Mesh] = []
    slot_parts: list[Mesh] = []
    for sy in (-1.0, +1.0):
        yw = sy * (D_IN / 2 + T_WALL + 0.0003)
        yw2 = sy * (D_IN / 2 + T_WALL + 0.0009)
        ya, yb = (yw, yw2) if sy > 0 else (yw2, yw)
        xb = +W_IN / 2
        v_rear = np.array([
            [xb - 0.068, ya, T_BASE + 0.004], [xb, ya, T_BASE + 0.004], [xb, yb, T_BASE + 0.004], [xb - 0.068, yb, T_BASE + 0.004],
            [xb - 0.016, ya, T_BASE + H_WALL - 0.010], [xb, ya, T_BASE + H_WALL - 0.010], [xb, yb, T_BASE + H_WALL - 0.010], [xb - 0.016, yb, T_BASE + H_WALL - 0.010],
        ])
        gusset_parts.append(_quad_prism_mesh(v_rear, BOX_GUSSET, collide="none"))
        v_slot = np.array([
            [xb - 0.052, ya - 0.0001, T_BASE + 0.052], [xb - 0.036, ya - 0.0001, T_BASE + 0.052],
            [xb - 0.036, yb + 0.0001, T_BASE + 0.052], [xb - 0.052, yb + 0.0001, T_BASE + 0.052],
            [xb - 0.052, ya - 0.0001, T_BASE + 0.084], [xb - 0.036, ya - 0.0001, T_BASE + 0.084],
            [xb - 0.036, yb + 0.0001, T_BASE + 0.084], [xb - 0.052, yb + 0.0001, T_BASE + 0.084],
        ])
        slot_parts.append(_quad_prism_mesh(v_slot, BOX_CREASE, collide="none"))
        xf = -W_IN / 2
        v_fslot = np.array([
            [xf + 0.034, ya - 0.0001, T_BASE + 0.052], [xf + 0.050, ya - 0.0001, T_BASE + 0.052],
            [xf + 0.050, yb + 0.0001, T_BASE + 0.052], [xf + 0.034, yb + 0.0001, T_BASE + 0.052],
            [xf + 0.034, ya - 0.0001, T_BASE + 0.084], [xf + 0.050, ya - 0.0001, T_BASE + 0.084],
            [xf + 0.050, yb + 0.0001, T_BASE + 0.084], [xf + 0.034, yb + 0.0001, T_BASE + 0.084],
        ])
        slot_parts.append(_quad_prism_mesh(v_fslot, BOX_CREASE, collide="none"))

    meshes.append(_merge_visual_meshes(gusset_parts, BOX_GUSSET))
    meshes.append(_merge_visual_meshes(slot_parts, BOX_CREASE))
    return boxes, meshes


def _build_front_panel_joint() -> Joint:
    t_fp = 0.0008  # 0.8 mm folded SBS flap proxy (pure UsdGeom.Cube -> 100% GPU-native collision, z <= 0.9 mm when flat!)
    h_fp = H_WALL - 0.0010  # 126.0 mm: scored 1.0 mm below side walls so lid seats flush on all 4 walls
    fp_boxes = [
        ((-W_IN / 2 - t_fp / 2, 0.0, T_BASE + h_fp / 2), (t_fp, D_IN - 0.0030, h_fp), BOX_WHITE),
        # Left (+Y_box) outer lock-corner dust ear (at y = +D_IN/2 + T_WALL + 0.0018 = +0.1304 m, completely outside
        # the 247.6 mm cake board path and hugging 1.0 mm outside left_wall when folded upright; stands 39 mm tall
        # off the counter at q_front = -90 deg so Left hand lifts front_panel by normal push behind the ear):
        ((-W_IN / 2 + 0.0202, +D_IN / 2 + T_WALL + 0.0018, T_BASE + H_WALL - 0.0220), (0.0380, 0.0016, 0.0360), BOX_GUSSET),
    ]
    return Joint(
        name="front_panel",
        kind="hinge",
        axis="Y",
        anchor=tuple(FRONT_ANCHOR),
        boxes=fp_boxes,
        lower=math.radians(-93.5),
        upper=math.radians(0.5),   # wall-contact stop at vertical against left_wall & right_wall
        stiffness=0.0,
        damping=0.008,
        target=FRONT_INIT,
        friction=0.026,   # 0.026 N*m paperboard crease friction (2.6x front_panel 0.010 N*m gravity torque)
        initial=FRONT_INIT,
        mass=0.016,
    )


def _build_lid_and_tab_joints() -> tuple[Joint, Joint]:
    outer_w = W_IN + 2 * T_WALL
    outer_d = D_IN + 2 * T_WALL
    zl = T_BASE + H_WALL + T_WALL / 2
    rail_x = (outer_w - WINDOW_SPAN) / 2
    rail_y = (outer_d - WINDOW_SPAN) / 2

    lid_boxes = [
        ((+outer_w / 2 - rail_x / 2, 0.0, zl), (rail_x, outer_d, T_WALL), BOX_WHITE),
        ((-outer_w / 2 + rail_x / 2, 0.0, zl), (rail_x, outer_d, T_WALL), BOX_WHITE),
        ((0.0, +outer_d / 2 - rail_y / 2, zl), (WINDOW_SPAN, rail_y, T_WALL), BOX_WHITE),
        ((0.0, -outer_d / 2 + rail_y / 2, zl), (WINDOW_SPAN, rail_y, T_WALL), BOX_WHITE),
        ((0.0, 0.0, zl - 0.0002), (WINDOW_SPAN + 0.010, WINDOW_SPAN + 0.010, 0.0006), WINDOW_FILM),
    ]
    lid = Joint(
        name="lid",
        kind="hinge",
        axis="Y",
        anchor=tuple(LID_ANCHOR),
        boxes=lid_boxes,
        lower=math.radians(-0.4),  # wall-contact stop when seated flat on the 127.0 mm walls
        upper=math.radians(97.0),
        stiffness=0.0,
        damping=6.0,
        target=LID_INIT,
        friction=0.082,   # calibrated paperboard crease stiction + 0.085 N*m clamped viscous hysteresis (~0.3 deg/s under gravity)
        initial=LID_INIT,
        mass=0.028,
    )

    xt0 = TAB_ANCHOR[0] - T_WALL / 2
    xt1 = TAB_ANCHOR[0] + T_WALL / 2
    zt_top = TAB_ANCHOR[2]
    zt_tip = TAB_ANCHOR[2] - TAB_LEN
    tab_v_vis = np.array([
        [xt0, -TAB_W_TIP / 2, zt_tip], [xt1, -TAB_W_TIP / 2, zt_tip],
        [xt1, +TAB_W_TIP / 2, zt_tip], [xt0, +TAB_W_TIP / 2, zt_tip],
        [xt0, -TAB_W_TOP / 2, zt_top], [xt1, -TAB_W_TOP / 2, zt_top],
        [xt1, +TAB_W_TOP / 2, zt_top], [xt0, +TAB_W_TOP / 2, zt_top],
    ])
    tab_vis = _quad_prism_mesh(tab_v_vis, BOX_WHITE, collide="none")
    crease_v = np.array([
        [xt0 - 0.0002, -TAB_W_TOP / 2, zt_top - 0.0020], [xt1 + 0.0002, -TAB_W_TOP / 2, zt_top - 0.0020],
        [xt1 + 0.0002, +TAB_W_TOP / 2, zt_top - 0.0020], [xt0 - 0.0002, +TAB_W_TOP / 2, zt_top - 0.0020],
        [xt0 - 0.0002, -TAB_W_TOP / 2, zt_top], [xt1 + 0.0002, -TAB_W_TOP / 2, zt_top],
        [xt1 + 0.0002, +TAB_W_TOP / 2, zt_top], [xt0 - 0.0002, +TAB_W_TOP / 2, zt_top],
    ])
    tab_crease = _quad_prism_mesh(crease_v, BOX_CREASE, collide="none")
    # Native box collider spanning r = 14.5..35.0 mm (14.5 mm hinge relief clears front_panel outer face in lid_no_tuck)
    tab_boxes = [((TAB_ANCHOR[0] - 0.0007, 0.0, TAB_ANCHOR[2] - 0.02475), (0.0012, TAB_W_TIP, 0.0205), BOX_WHITE)]

    lid_tab = Joint(
        name="lid_tab",
        kind="hinge",
        axis="Y",
        parent="lid",
        anchor=tuple(TAB_ANCHOR),
        boxes=tab_boxes,
        meshes=(tab_vis, tab_crease),
        lower=math.radians(-58.0),
        upper=math.radians(85.0),
        stiffness=0.001,
        damping=1.60,
        target=TAB_INIT,
        friction=0.008,   # paperboard crease spring (0.001 N*m/rad) + 0.008 N*m stiction + 0.014 N*m clamped hysteresis
        initial=TAB_INIT,
        mass=0.007,
    )
    return lid, lid_tab


_BOX_MAIN_BOXES, _BOX_MAIN_MESHES = _build_box_main_shapes()
_FRONT_JOINT = _build_front_panel_joint()
_LID_JOINT, _TAB_JOINT = _build_lid_and_tab_joints()

CAKE_BOX = part(
    "cake_box",
    pos=(float(BOX_XY[0]), float(BOX_XY[1]), TOP),
    yaw=BOX_YAW,
    mass=0.095,  # + 0.016 + 0.028 + 0.007 = 0.146 kg total box mass
    friction=0.52,
    color=BOX_WHITE,
    boxes=_BOX_MAIN_BOXES,
    meshes=_BOX_MAIN_MESHES,
    joints=(_FRONT_JOINT, _LID_JOINT, _TAB_JOINT),
    self_collide=True,
)

CAKE_BOARD = part(
    "cake_board",
    pos=(float(CAKE_START_XY[0]), float(CAKE_START_XY[1]), TOP),
    mass=0.055,
    friction=0.20,
    color=BOARD_SILVER,
    meshes=_build_board_meshes(),
)

CAKE = part(
    "cake",
    pos=(float(CAKE_START_XY[0]), float(CAKE_START_XY[1]), TOP + T_BOARD),
    mass=0.300,
    friction=3.20,  # royal-icing dab on foil circle (effective mu ~1.70 vs ~0.35 board-on-counter/liner)
    color=FONDANT_BLUSH,
    meshes=_build_cake_meshes(),
)

OBJECTS = [CAKE_BOX, CAKE_BOARD, CAKE]


@on_reset
def _reset_box_joints(env):
    import torch

    base = env.unwrapped
    asset = base.scene["cake_box"]
    names = list(asset.joint_names)
    djp = asset.data.default_joint_pos
    q = (djp.torch if hasattr(djp, "torch") else djp).clone()
    qd = torch.zeros_like(q)
    for jname, val in (("front_panel_joint", FRONT_INIT), ("lid_joint", LID_INIT), ("lid_tab_joint", TAB_INIT)):
        if jname in names:
            q[:, names.index(jname)] = float(val)
    dev = str(asset.device)
    if "lid_joint" in names and hasattr(asset, "write_joint_effort_limit_to_sim_index"):
        idx_l = names.index("lid_joint")
        asset.write_joint_effort_limit_to_sim_index(
            limits=torch.tensor([[0.085]], device=dev, dtype=torch.float32),
            joint_ids=[idx_l],
        )
    if "lid_tab_joint" in names and hasattr(asset, "write_joint_effort_limit_to_sim_index"):
        idx_tab = names.index("lid_tab_joint")
        asset.write_joint_effort_limit_to_sim_index(
            limits=torch.tensor([[0.014]], device=dev, dtype=torch.float32),
            joint_ids=[idx_tab],
        )
    asset.write_joint_state_to_sim(q, qd)
    asset.set_joint_position_target(q)


# -- IK & state-assisted demo trajectory helpers (frankly disclosed in fidelity per review prompt v24) ----------------

def _frame(d, c):
    z = np.asarray(d, float) / np.linalg.norm(d)
    x = np.asarray(c, float) - np.dot(c, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class Orient(float):
    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_cake_box_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._cake_box_rot = True
    _demo.grip_rotation = _grip


def _ik_err(pe, re):
    return pe * 1000.0 + math.degrees(re)


FREE_MOVES = ("park", "arc around", "reposition", "clear")


def _endpoints(plan):
    rng = np.random.default_rng(11)
    seeds = [np.array(_READY, float)] + [np.array(_READY, float) + rng.normal(0.0, 0.75, 6) for _ in range(12)]
    q_prev, out, switch, forced = plan.q.copy(), [], set(), set()
    for i, w in enumerate(plan.waypoints):
        pos = np.asarray(w.pos, float)
        quat = _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, q_prev, base_pos=plan.base_pos, iters=300)
        if any(m in w.label for m in FREE_MOVES):
            forced.add(i)
        if _ik_err(pe, re) > 1.5:
            switch.add(i)
            best = (_ik_err(pe, re), 0.0, q)
            for s in [q_prev + rng.normal(0.0, 0.3, 6) for _ in range(4)] + seeds:
                qs, ps, rs = _ik.solve(pos, quat, s, base_pos=plan.base_pos, iters=220)
                e, d = _ik_err(ps, rs), float(np.max(np.abs(qs - q_prev)))
                if (e < 1.0 and (best[0] >= 1.0 or d < best[1])) or (best[0] >= 1.0 and e < best[0]):
                    best = (e, d, qs)
            q = best[2]
        elif (float(np.max(np.abs(q[3:5] - q_prev[3:5]))) > math.radians(75.0)
              or float(np.max(np.abs(q - q_prev))) > math.radians(105.0)):
            switch.add(i)
        out.append(q)
        q_prev = q
    return out, switch | forced


def _seg(plan, t):
    acc = 0.0
    for i, w in enumerate(plan.waypoints):
        if t < acc + w.duration:
            return i, _ik.smoothstep((t - acc) / w.duration)
        acc += w.duration
    return len(plan.waypoints) - 1, 1.0


def _action7(self, t):
    _demo._plan_clock = float(t)
    snap_gen = _ct.get("snap_gen", 0)
    if getattr(self, "_qends", None) is None or getattr(self, "_snap_gen", -1) != snap_gen:
        self._qends, self._switch = _endpoints(self)
        self._snap_gen = snap_gen
    pos, rot, finger, label = self.target(t)
    quat = _ik.matrix_to_quat_xyzw(rot)
    i, s = _seg(self, t)
    q0 = self._qends[i - 1] if i > 0 else self._q_start
    seed = q0 + s * (self._qends[i] - q0)
    if i in self._switch:
        self.q = seed
        grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
        return np.r_[seed, grip], label
    q, pe, re = _ik.solve(pos, quat, self.q, base_pos=self.base_pos, iters=90)
    if _ik_err(pe, re) > 2.0:
        q2, p2, r2 = _ik.solve(pos, quat, seed, base_pos=self.base_pos, iters=90)
        if _ik_err(p2, r2) < _ik_err(pe, re) - 1.0:
            q, pe, re = q2, p2, r2
    self.q = q
    self.worst_ik_mm = max(self.worst_ik_mm, pe * 1000.0)
    self.worst_ik_deg = max(self.worst_ik_deg, float(np.degrees(re)))
    grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
    return np.r_[q, grip], label


_prev_init = _demo.ArmPlan.__init__
if not getattr(_prev_init, "_cake_box_fa", False):
    def _init(self, *a, _prev=_prev_init, **k):
        _prev(self, *a, **k)
        self._q_start, self._qends = self.q.copy(), None

    _init._cake_box_fa = True
    _demo.ArmPlan.__init__ = _init
    _demo.ArmPlan.action7 = _action7


_ct = {
    "trim": {},
    "wins": [],
    "box_snap": None,
    "panel_snap": None,
    "snap_ql": math.radians(17.0),
    "snap_wins": [],
    "fired_snaps": set(),
    "snap_gen": 0,
}
TRIM_LIMITS = {
    "s1_steer": (0.0040, 0.036),
    "stage_lat": (0.0006, 0.035),
    "s2_steer": (0.0045, 0.065),
    "push_in": (0.0006, 0.035),
    "r_z_skim": (0.0025, 0.020),
    "r_z_box": (0.0025, 0.020),
}
TRIM_GAIN = {
    "s1_steer": 0.70,
    "stage_lat": 0.18,
    "s2_steer": 0.70,
    "push_in": 0.22,
    "r_z_skim": 0.65,
    "r_z_box": 0.65,
}


@on_reset
def _reset_trims(_env):
    _ct["trim"] = {}
    _ct["box_snap"] = None
    _ct["panel_snap"] = None
    _ct["snap_ql"] = math.radians(17.0)
    _ct["fired_snaps"] = set()
    _ct["snap_gen"] = 0


@on_step
def _update_trims(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    for idx, (ts0, ts1, set_anchor) in enumerate(_ct.get("snap_wins", [])):
        if ts0 <= t < ts1 and idx not in _ct["fired_snaps"]:
            bx = labkit._part_state(env, CAKE_BOX)
            brot = np.asarray(bx.rotation(), float)
            byaw = float(math.atan2(brot[1, 0], brot[0, 0]))
            pose = (np.array([float(bx.pos[0]), float(bx.pos[1])], float), byaw)
            if set_anchor:
                _ct["box_snap"] = pose
            _ct["panel_snap"] = pose
            _ct["snap_ql"] = float(bx.joints.get("lid", math.radians(17.0)))
            _ct["fired_snaps"].add(idx)
            _ct["snap_gen"] = _ct.get("snap_gen", 0) + 1
    for t0, t1, key, fn in _ct["wins"]:
        if not (t0 <= t < t1):
            continue
        err = np.asarray(fn(env), float)
        tr = _ct["trim"].get(key, np.zeros(3))
        step, cap = TRIM_LIMITS.get(key, (0.0008, 0.035))
        _ct["trim"][key] = np.clip(tr + np.clip(TRIM_GAIN.get(key, 0.20) * err, -step, step), -cap, cap)


class SnapOrient(float):
    def __new__(cls, rot0):
        obj = float.__new__(cls, 0.0)
        obj._rot0 = np.asarray(rot0, float)
        return obj

    @property
    def rot(self):
        snap = _ct.get("box_snap")
        if snap is None:
            return self._rot0
        dy = float(snap[1]) - BOX_YAW
        c, s = math.cos(dy), math.sin(dy)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float) @ self._rot0


class _SnapPos:
    def __init__(self, lx: float, ly: float, lz: float):
        self.lx, self.ly, self.lz = float(lx), float(ly), float(lz)

    def __array__(self, dtype=None, copy=None):
        snap = _ct.get("box_snap")
        if snap is None:
            p = box_local_to_world(self.lx, self.ly, self.lz)
        else:
            bxy, byaw = snap
            c, s = math.cos(byaw), math.sin(byaw)
            p = (bxy[0] + self.lx * c - self.ly * s, bxy[1] + self.lx * s + self.ly * c, TOP + self.lz)
        return np.asarray(p, dtype=dtype or float)


class PanelOrient(float):
    def __new__(cls, rot0):
        obj = float.__new__(cls, 0.0)
        obj._rot0 = np.asarray(rot0, float)
        return obj

    @property
    def rot(self):
        ps = _ct.get("panel_snap")
        if ps is None:
            return self._rot0
        dy = float(ps[1]) - BOX_YAW
        c, s = math.cos(dy), math.sin(dy)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float) @ self._rot0


class _PanelPos:
    def __init__(self, lx: float, ly: float, lz: float):
        self.lx, self.ly, self.lz = float(lx), float(ly), float(lz)

    def __array__(self, dtype=None, copy=None):
        ps = _ct.get("panel_snap")
        if ps is None:
            p = box_local_to_world(self.lx, self.ly, self.lz)
        else:
            bxy, byaw = ps
            c, s = math.cos(byaw), math.sin(byaw)
            p = (bxy[0] + self.lx * c - self.ly * s, bxy[1] + self.lx * s + self.ly * c, TOP + self.lz)
        return np.asarray(p, dtype=dtype or float)


class _PanelTabArc:
    """Circular-arc waypoint at radius r_m around lid_tab's live hinge in CAKE_BOX's panel_snap SE(2) frame."""

    def __init__(self, qt_deg: float, r_m: float, ly: float, tip_lead: float = 0.0075):
        self.qt_deg, self.r_m, self.ly, self.tip_lead = float(qt_deg), float(r_m), float(ly), float(tip_lead)

    def __array__(self, dtype=None, copy=None):
        ql = float(_ct.get("snap_ql", math.radians(44.0)))
        c_l, s_l = math.cos(ql), math.sin(ql)
        rx = float(TAB_ANCHOR[0] - LID_ANCHOR[0])  # -0.2523 m
        th = ql + math.radians(self.qt_deg)
        dx = -self.r_m * math.sin(th) - self.tip_lead * 0.720
        dz = -self.r_m * math.cos(th) + self.tip_lead * 0.694
        lx = float(LID_ANCHOR[0]) + rx * c_l + dx
        lz = float(LID_ANCHOR[2]) - rx * s_l + dz
        ps = _ct.get("panel_snap")
        if ps is None:
            p = box_local_to_world(lx, self.ly, lz)
        else:
            bxy, byaw = ps
            c, s = math.cos(byaw), math.sin(byaw)
            p = (bxy[0] + lx * c - self.ly * s, bxy[1] + lx * s + self.ly * c, TOP + lz)
        return np.asarray(p, dtype=dtype or float)

class _Dyn:
    def __init__(self, tcp, *keys):
        self.tcp = np.asarray(tcp, float)
        self.keys = keys

    def __array__(self, dtype=None, copy=None):
        off = sum((_ct["trim"].get(k, np.zeros(3)) for k in self.keys), np.zeros(3))
        return np.asarray(self.tcp + off, dtype=dtype or float)


class _Plan:
    def __init__(self):
        self.L, self.R = [], []
        self.t = 0.0

    def hold(self, plan, dur, label="hold"):
        w = plan[-1]
        plan.append(Waypoint(dur, w.pos, w.finger, label, w.yaw))

    def phase(self, left=(), right=()):
        tl = sum(w[0] for w in left)
        tr = sum(w[0] for w in right)
        dur = max(tl, tr)
        for plan, ws, tot in ((self.L, left, tl), (self.R, right, tr)):
            for w in ws:
                plan.append(Waypoint(*w))
            if dur - tot > 1e-9:
                self.hold(plan, dur - tot)
        t0 = self.t
        self.t += dur
        return t0


LEFT_PARK = (0.120, +0.355, TOP + 0.240)
RIGHT_PARK = (0.120, -0.355, TOP + 0.240)
STAGE_OFFSET_X = -0.2850
STAGE_XY = BOX_XY + STAGE_OFFSET_X * UX_BOX
_V1 = STAGE_XY - CAKE_START_XY
_U1 = _V1 / np.linalg.norm(_V1)
_N1_PERP = np.array([_U1[1], -_U1[0]], dtype=float)


def _s1_steer_err(env):
    """Smooth C^inf Lynch-Mason lateral steering during Move 1a (zero bang-bang step jump across centerline)."""
    pl = labkit._part_state(env, CAKE_BOARD)
    e_lat = float(np.dot(np.asarray(pl.pos, float)[:2] - CAKE_START_XY, _N1_PERP))
    off = float(np.clip(e_lat + 0.0074 * math.tanh(e_lat / 0.0018) + 2.0 * e_lat, -0.034, 0.034))
    target = np.array([off * _N1_PERP[0], off * _N1_PERP[1], 0.0])
    return target - _ct["trim"].get("s1_steer", np.zeros(3))


def _stage_lat_err(env):
    """Forward push alignment (along +uy_live) stopping cleanly once within 0.8 mm of CAKE_BOX's live throat axis."""
    bx = labkit._part_state(env, CAKE_BOX)
    pl = labkit._part_state(env, CAKE_BOARD)
    rot2 = np.asarray(bx.rotation(), float)[:2, :2]
    uy_live = rot2[:, 1]
    lat_err = float(np.dot(np.asarray(bx.pos, float)[:2] - np.asarray(pl.pos, float)[:2], uy_live))
    if lat_err <= 0.0008:
        return np.zeros(3)
    return np.array([lat_err * uy_live[0], lat_err * uy_live[1], 0.0])


def _s2_steer_err(env):
    """Smooth C^inf Lynch-Mason lateral steering during Move 1b relative to CAKE_BOX's live throat centerline."""
    bx = labkit._part_state(env, CAKE_BOX)
    pl = labkit._part_state(env, CAKE_BOARD)
    pl_xy = np.asarray(pl.pos, float)[:2]
    bx_xy = np.asarray(bx.pos, float)[:2]
    uy_live = np.asarray(bx.rotation(), float)[:2, 1]
    d_pl_world = float(np.dot(pl_xy - BOX_XY, UY_BOX))
    e_lat = float(np.dot(pl_xy - bx_xy, uy_live))
    delta = 0.0074 * math.tanh(e_lat / 0.0018) + 2.4 * e_lat
    off = float(np.clip(d_pl_world + delta, -0.060, 0.060))
    target = np.array([off * UY_BOX[0], off * UY_BOX[1], 0.0])
    return target - _ct["trim"].get("s2_steer", np.zeros(3))

def _push_in_err(env):
    """Forward axial seating (along +ux_live) stopping cleanly once within 1.2 mm of CAKE_BOX's live center."""
    bx = labkit._part_state(env, CAKE_BOX)
    pl = labkit._part_state(env, CAKE_BOARD)
    rot2 = np.asarray(bx.rotation(), float)[:2, :2]
    ux_live = rot2[:, 0]
    ax_err = float(np.dot(np.asarray(bx.pos, float)[:2] - np.asarray(pl.pos, float)[:2], ux_live))
    if ax_err <= 0.0018:
        return np.zeros(3)
    return np.array([ax_err * ux_live[0], ax_err * ux_live[1], 0.0])


def _live_arm_tip_z_min(env, arm_name: str = "right_robot") -> float:
    """Minimum world Z (m) of tip_left / tip_right from the robot's own joint encoders."""
    from yam_arena import runtime as _rt

    base = env.unwrapped
    origin = base.scene.env_origins[0].cpu().numpy()
    hulls = get_robot_urdf_hulls()
    poses = _rt.link_poses(base.scene[arm_name])
    z_min = float("inf")
    for link in ("tip_left", "tip_right"):
        T = poses.get(link)
        if T is None:
            continue
        R, p = T[:3, :3], T[:3, 3] - origin
        for v, _ in hulls.get(link, []):
            z_min = min(z_min, float((v @ R.T + p)[:, 2].min()))
    return z_min


def _r_z_skim_err(env):
    """Proprioceptive wrist-droop Z trim keeping right tip bottom at TOP + 1.4 mm (3.4 mm below 4.8 mm board top!)."""
    z_now = _live_arm_tip_z_min(env, "right_robot")
    return np.array([0.0, 0.0, (TOP + 0.0014) - z_now])


def _r_z_box_err(env):
    """Proprioceptive wrist-droop Z trim keeping right tip bottom at TOP + T_BASE + 1.3 mm inside box throat."""
    z_now = _live_arm_tip_z_min(env, "right_robot")
    return np.array([0.0, 0.0, (TOP + T_BASE + 0.0009) - z_now])


def demo():
    """Scripted bakery boxing demo (+ diagnostic controls via CAKE_BOX_MODE)."""
    mode = os.environ.get("CAKE_BOX_MODE", "normal").strip().lower()
    _ct["wins"] = []
    _ct["snap_wins"] = []
    P = _Plan()

    ux3 = np.array([UX_BOX[0], UX_BOX[1], 0.0])
    uy3 = np.array([UY_BOX[0], UY_BOX[1], 0.0])

    R0 = Orient(_prev_grip(0.0))
    R_WALL_CLAMP = Orient(_prev_grip(BOX_YAW))
    R_R_CLAMP = SnapOrient(_frame(0.22 * ux3 - 0.975 * Z, -uy3))

    u1 = _U1
    u1_3 = np.array([u1[0], u1[1], 0.0])
    th28 = math.radians(28.0)
    R_PUSH_STAGE = Orient(_frame(math.sin(th28) * u1_3 - math.cos(th28) * Z, -u1_3))
    R_PUSH_IN = Orient(_frame(math.sin(th28) * ux3 - math.cos(th28) * Z, +ux3))

    OPEN_WIDE = FINGER_OPEN
    PUSH_BLADE = FINGER_CLOSED

    FACE_1A = 0.0087
    FACE_1B = 0.0086
    z_board_s1 = TOP + 0.0143
    z_board_s2 = TOP + T_BASE + 0.0142
    z_high_clear = TOP + 0.1780
    z_clamp_wall = 0.1200
    y_left_loc = +D_IN / 2 + T_WALL / 2
    y_right_loc = -D_IN / 2 - T_WALL / 2

    if mode == "untouched":
        p_l, rot_l = _ik.tcp_pose(_READY, LEFT_BASE_POS)
        p_r, rot_r = _ik.tcp_pose(_READY, RIGHT_BASE_POS)
        P.phase(left=[(3.5, tuple(p_l.tolist()), OPEN_WIDE, "untouched", Orient(rot_l))], right=[(3.5, tuple(p_r.tolist()), OPEN_WIDE, "untouched", Orient(rot_r))])
        return {"left": P.L, "right": P.R}

    if mode == "touch_frosting":
        cx0, cy0 = float(CAKE_START_XY[0]), float(CAKE_START_XY[1])
        P.phase(
            left=[(2.0, LEFT_PARK, OPEN_WIDE, "park", R0)],
            right=[
                (1.8, (cx0, cy0, TOP + H_ROSETTE_TOP + 0.060), FINGER_CLOSED, "above cake", R0),
                (1.4, (cx0, cy0, TOP + T_BOARD + H_CAKE_BODY + 0.004), FINGER_CLOSED, "graze frosting", R0),
                (1.4, (cx0, cy0, TOP + H_ROSETTE_TOP + 0.080), FINGER_CLOSED, "lift", R0),
                (1.8, RIGHT_PARK, OPEN_WIDE, "park", R0),
            ],
        )
        P.phase(left=[(2.0, LEFT_PARK, OPEN_WIDE, "hold", R0)], right=[(2.0, RIGHT_PARK, OPEN_WIDE, "hold", R0)])
        return {"left": P.L, "right": P.R}

    # -----------------------------------------------------------------------------------------------------------------
    # PHASE 1: Exact probe12/probe14 Phase 1 (verified on 5090: board_inside=True, frosting_touch=None, tier=0.25!)
    # -----------------------------------------------------------------------------------------------------------------
    p1_pre = CAKE_START_XY - (R_BOARD + 0.042) * u1
    p1_eng = CAKE_START_XY - (R_BOARD + FACE_1A + 0.010) * u1
    p1_end = STAGE_XY - (R_BOARD + FACE_1A + 0.009) * u1
    p1_off = STAGE_XY - (R_BOARD + 0.048) * u1

    t_s1 = P.phase(
        left=[
            (1.2, box_local_to_world(-0.085, y_left_loc, H_WALL + 0.065), OPEN_WIDE, "reposition open above left wall", R_WALL_CLAMP),
            (0.8, box_local_to_world(-0.085, y_left_loc, z_clamp_wall), OPEN_WIDE, "straddle left wall top", R_WALL_CLAMP),
            (0.6, box_local_to_world(-0.085, y_left_loc, z_clamp_wall), FINGER_CLOSED, "clamp left wall steady", R_WALL_CLAMP),
        ],
        right=[
            (1.7, (p1_pre[0], p1_pre[1], TOP + 0.125), PUSH_BLADE, "reposition high behind -Y rim", R_PUSH_STAGE),
            (0.9, _Dyn((p1_pre[0], p1_pre[1], z_board_s1), "r_z_skim"), PUSH_BLADE, "descend behind -Y rim", R_PUSH_STAGE),
            (0.8, _Dyn((p1_eng[0], p1_eng[1], z_board_s1), "r_z_skim", "s1_steer"), PUSH_BLADE, "engage -Y board rim", R_PUSH_STAGE),
            (4.8, _Dyn((p1_end[0], p1_end[1], z_board_s1), "r_z_skim", "s1_steer"), PUSH_BLADE, "slide board to box apron", R_PUSH_STAGE),
            (1.8, _Dyn((p1_end[0], p1_end[1], z_board_s1), "r_z_skim", "s1_steer", "stage_lat"), PUSH_BLADE, "fine-align throat axis", R_PUSH_STAGE),
            (0.8, (p1_off[0], p1_off[1], z_high_clear), PUSH_BLADE, "lift high clear", R_PUSH_STAGE),
        ],
    )
    _ct["wins"].append((t_s1 + 2.0, t_s1 + 10.0, "r_z_skim", _r_z_skim_err))
    _ct["wins"].append((t_s1 + 2.6, t_s1 + 10.0, "s1_steer", _s1_steer_err))
    _ct["wins"].append((t_s1 + 8.2, t_s1 + 10.0, "stage_lat", _stage_lat_err))

    p2_arc = STAGE_XY - (R_BOARD + 0.072) * (0.72 * UX_BOX - 0.69 * UY_BOX)
    p2_pre = STAGE_XY - (R_BOARD + 0.042) * UX_BOX
    p2_eng = STAGE_XY - (R_BOARD + FACE_1B + 0.010) * UX_BOX
    p2_end = BOX_XY - (R_BOARD + FACE_1B + 0.0135) * UX_BOX
    p2_off = BOX_XY - (R_BOARD + 0.052) * UX_BOX

    t_s2 = P.phase(
        left=[(11.2, box_local_to_world(-0.085, y_left_loc, z_clamp_wall), FINGER_CLOSED, "hold left wall clamped", R_WALL_CLAMP)],
        right=[
            (1.3, (p2_arc[0], p2_arc[1], TOP + 0.155), PUSH_BLADE, "arc around outside rim", R_PUSH_IN),
            (1.0, _Dyn((p2_pre[0], p2_pre[1], z_board_s2), "r_z_box", "s2_steer"), PUSH_BLADE, "behind rear board rim", R_PUSH_IN),
            (0.9, _Dyn((p2_eng[0], p2_eng[1], z_board_s2), "r_z_box", "s2_steer"), PUSH_BLADE, "engage rear board rim", R_PUSH_IN),
            (5.4, _Dyn((p2_end[0], p2_end[1], z_board_s2), "r_z_box", "s2_steer"), PUSH_BLADE, "slide cake straight into box", R_PUSH_IN),
            (1.8, _Dyn((p2_end[0], p2_end[1], z_board_s2), "r_z_box", "s2_steer", "push_in"), PUSH_BLADE, "seat board flush", R_PUSH_IN),
            (0.8, _Dyn((p2_off[0], p2_off[1], TOP + 0.065), "s2_steer"), PUSH_BLADE, "back off throat", R_PUSH_IN),
        ],
    )
    _ct["wins"].append((t_s2 + 1.8, t_s2 + 10.4, "r_z_box", _r_z_box_err))
    _ct["wins"].append((t_s2 + 1.8, t_s2 + 10.4, "s2_steer", _s2_steer_err))
    _ct["wins"].append((t_s2 + 8.6, t_s2 + 10.4, "push_in", _push_in_err))

    t_snap1 = P.phase(
        left=[(0.35, box_local_to_world(-0.085, y_left_loc, z_clamp_wall), FINGER_CLOSED, "hold left wall quiet", R_WALL_CLAMP)],
        right=[(0.35, _Dyn((p2_off[0], p2_off[1], TOP + 0.065), "s2_steer"), PUSH_BLADE, "quiet snapshot 1", R_PUSH_IN)],
    )
    _ct["snap_wins"].append((t_snap1 + 0.08, t_snap1 + 0.28, True))

    # -----------------------------------------------------------------------------------------------------------------
    # PHASE 2: Two-stage front_panel fold (0.25 -> 0.50):
    #   Stage 2A lifts front_ear to ~ -32 deg (proven in probe14!), where 0.030 N*m crease friction holds it parked;
    #   Quiet snapshot 2B refreshes CAKE_BOX's exact pose while Left is in air;
    #   Stage 2B pushes center of front_panel (y = +0.035 m) squarely onto its +0.5 deg vertical stop!
    # -----------------------------------------------------------------------------------------------------------------
    y_ear_push = +0.1330
    R_FLAP_L = PanelOrient(_frame(0.50 * ux3 - 0.866 * Z, +uy3))
    R_LID = PanelOrient(_frame(0.50 * ux3 - 0.25 * uy3 - 0.829 * Z, -uy3))
    R_FLAP_R = PanelOrient(_frame(0.55 * ux3 - 0.835 * Z, -uy3))
    R_L_SNAP = PanelOrient(_prev_grip(BOX_YAW))

    P.phase(
        left=[(2.5, box_local_to_world(-0.085, y_left_loc, z_clamp_wall), FINGER_CLOSED, "hold left wall clamped", R_WALL_CLAMP)],
        right=[
            (0.8, _SnapPos(-0.145, y_right_loc - 0.030, H_WALL + 0.045), OPEN_WIDE, "reposition up outside right wall", R_R_CLAMP),
            (0.6, _SnapPos(-0.090, y_right_loc, H_WALL + 0.042), OPEN_WIDE, "reposition above right wall", R_R_CLAMP),
            (0.6, _SnapPos(-0.090, y_right_loc, z_clamp_wall), OPEN_WIDE, "straddle right wall top", R_R_CLAMP),
            (0.5, _SnapPos(-0.090, y_right_loc, z_clamp_wall), FINGER_CLOSED, "clamp right wall steady", R_R_CLAMP),
        ],
    )

    flap_2a_wps = [
        (0.35, _PanelPos(-0.085, y_left_loc, z_clamp_wall), OPEN_WIDE, "release left wall", R_L_SNAP),
        (0.55, _PanelPos(-0.085, y_left_loc, H_WALL + 0.060), OPEN_WIDE, "lift off left wall", R_L_SNAP),
        (0.8, _PanelPos(-0.270, y_ear_push + 0.012, H_WALL + 0.020), FINGER_CLOSED, "reposition behind +Y front ear", R_FLAP_L),
        (0.7, _PanelPos(-0.270, y_ear_push, 0.018), FINGER_CLOSED, "descend behind upright +Y ear", R_FLAP_L),
        (0.75, _PanelPos(-0.232, y_ear_push, 0.020), FINGER_CLOSED, "engage ear horizontal (-72deg)", R_FLAP_L),
        (0.75, _PanelPos(-0.208, y_ear_push, 0.058), FINGER_CLOSED, "lift ear along arc (-50deg)", R_FLAP_L),
        (0.75, _PanelPos(-0.178, y_ear_push, 0.094), FINGER_CLOSED, "lift ear along arc (-34deg)", R_FLAP_L),
        (0.65, _PanelPos(-0.156, y_ear_push, 0.112), FINGER_CLOSED, "lift ear along arc (-18deg)", R_FLAP_L),
        (0.5, _PanelPos(-0.228, y_ear_push, 0.112), FINGER_CLOSED, "step back from parked ear", R_FLAP_L),
    ]
    P.phase(
        left=flap_2a_wps,
        right=[(sum(w[0] for w in flap_2a_wps), _SnapPos(-0.090, y_right_loc, z_clamp_wall), FINGER_CLOSED, "hold right wall clamped", R_R_CLAMP)],
    )

    t_snap_2b = P.phase(
        left=[(0.3, _PanelPos(-0.228, y_ear_push, 0.105), FINGER_CLOSED, "quiet snapshot 2b", R_FLAP_L)],
        right=[(0.3, _SnapPos(-0.090, y_right_loc, z_clamp_wall), FINGER_CLOSED, "hold right wall quiet", R_R_CLAMP)],
    )
    _ct["snap_wins"].append((t_snap_2b + 0.06, t_snap_2b + 0.24, False))

    flap_2b_wps = [
        (0.65, _PanelPos(-0.242, +0.035, 0.072), FINGER_CLOSED, "reposition behind parked front panel", R_FLAP_L),
        (0.7, _PanelPos(-0.188, +0.035, 0.092), FINGER_CLOSED, "engage front panel center", R_FLAP_L),
        (0.75, _PanelPos(-0.152, +0.035, 0.104), FINGER_CLOSED, "fold front panel -10deg", R_FLAP_L),
        (0.85, _PanelPos(-0.1300, +0.035, 0.104), FINGER_CLOSED, "square front panel vertical", R_FLAP_L),
        (0.35, _PanelPos(-0.1300, +0.035, 0.104), FINGER_CLOSED, "hold front panel upright", R_FLAP_L),
        (0.55, _PanelPos(-0.175, +0.035, 0.132), FINGER_CLOSED, "back off front panel", R_FLAP_L),
    ]
    P.phase(
        left=flap_2b_wps,
        right=[(sum(w[0] for w in flap_2b_wps), _SnapPos(-0.090, y_right_loc, z_clamp_wall), FINGER_CLOSED, "hold right wall clamped", R_R_CLAMP)],
    )

    t_snap2 = P.phase(
        left=[(0.3, _PanelPos(-0.175, +0.035, 0.132), FINGER_CLOSED, "quiet snapshot 2", R_FLAP_L)],
        right=[(0.3, _SnapPos(-0.090, y_right_loc, z_clamp_wall), FINGER_CLOSED, "hold right wall quiet", R_R_CLAMP)],
    )
    _ct["snap_wins"].append((t_snap2 + 0.06, t_snap2 + 0.24, False))

    # -----------------------------------------------------------------------------------------------------------------
    # PHASE 3: Close window lid along y = +0.088 m (39 mm inside support polygon -> 2.4 mm board_z_max in probe14!)
    #          With lid.friction = 0.0092, lid parks rock-solid at ~ +23 deg for tab creasing (0.50 -> 0.75)
    # -----------------------------------------------------------------------------------------------------------------
    ax_l_loc = float(LID_ANCHOR[0])
    az_l_loc = float(LID_ANCHOR[2])
    y_lid_hi = +0.116
    y_lid_lo = +0.088

    def _lid_push_tcp(deg: float, rad: float, preload: float = 0.0035, y_rail: float = y_lid_lo) -> _PanelPos:
        th = math.radians(deg)
        n_out_x, n_out_z = math.sin(th), math.cos(th)
        sx = ax_l_loc - rad * math.cos(th) + (T_WALL - preload) * n_out_x
        sz = az_l_loc + rad * math.sin(th) + (T_WALL - preload) * n_out_z
        return _PanelPos(sx + 0.006, y_rail, sz + 0.0138)

    y_tab_loc = -0.045
    R_TAB = PanelOrient(_frame(0.720 * ux3 - 0.694 * Z, 0.694 * ux3 + 0.720 * Z))

    lid_tip_wps = [
        (0.85, _PanelPos(-0.010, +D_IN / 2 + T_WALL + 0.055, az_l_loc + 0.135), FINGER_CLOSED, "reposition outside +Y lid edge", R_LID),
        (0.85, _PanelPos(ax_l_loc + 0.098, +D_IN / 2 + T_WALL + 0.055, az_l_loc + 0.135), FINGER_CLOSED, "reposition well past +X lid back", R_LID),
        (0.65, _PanelPos(ax_l_loc + 0.098, y_lid_hi, az_l_loc + 0.135), FINGER_CLOSED, "step behind open lid clear of housing", R_LID),
        (0.85, _lid_push_tcp(84.0, rad=0.130, preload=0.0035, y_rail=y_lid_hi), FINGER_CLOSED, "tip lid forward past vertical (84deg)", R_LID),
        (0.85, _lid_push_tcp(58.0, rad=0.140, preload=0.0035, y_rail=y_lid_lo), FINGER_CLOSED, "push lid down to 58deg", R_LID),
    ]
    P.phase(
        left=lid_tip_wps,
        right=[
            (0.40, _SnapPos(-0.090, y_right_loc, z_clamp_wall), OPEN_WIDE, "release right wall while lid at 95deg", R_R_CLAMP),
            (0.55, _SnapPos(-0.090, y_right_loc - 0.035, H_WALL + 0.050), OPEN_WIDE, "lift clear of right wall", R_R_CLAMP),
            (0.85, _PanelPos(-0.180, y_tab_loc, 0.120), FINGER_CLOSED, "reposition in front of front panel", R_FLAP_R),
            (0.55, _PanelPos(-0.1310, y_tab_loc, 0.085), FINGER_CLOSED, "brace front panel low during lid tip", R_FLAP_R),
            (1.70, _PanelPos(-0.1310, y_tab_loc, 0.085), FINGER_CLOSED, "hold front panel braced while lid tips", R_FLAP_R),
        ],
    )

    if mode == "lid_no_tuck":
        P.phase(
            left=[
                (0.85, _lid_push_tcp(34.0, rad=0.150, preload=0.0035), FINGER_CLOSED, "push lid down to 34deg", R_LID),
                (0.85, _lid_push_tcp(14.0, rad=0.158, preload=0.0035), FINGER_CLOSED, "push lid down to 14deg", R_LID),
                (0.85, _lid_push_tcp(3.0, rad=0.163, preload=0.0055), FINGER_CLOSED, "lower lid to 3deg (tab outside)", R_LID),
                (1.1, _lid_push_tcp(0.0, rad=0.168, preload=0.0070), FINGER_CLOSED, "seat lid flush onto walls", R_LID),
                (0.8, _PanelPos(-0.045, y_lid_lo, az_l_loc + 0.075), OPEN_WIDE, "lift off closed lid", R_LID),
                (1.5, LEFT_PARK, OPEN_WIDE, "park", R0),
            ],
            right=[
                (0.55, _PanelPos(-0.195, y_tab_loc, 0.115), OPEN_WIDE, "withdraw cleanly from front panel", R_FLAP_R),
                (1.5, RIGHT_PARK, OPEN_WIDE, "park", R0),
            ],
        )
        P.phase(left=[(2.5, LEFT_PARK, OPEN_WIDE, "settle", R0)], right=[(2.5, RIGHT_PARK, OPEN_WIDE, "settle", R0)])
        return {"left": P.L, "right": P.R}

    P.phase(
        left=[
            (1.65, _lid_push_tcp(48.0, rad=0.144, preload=0.0025), FINGER_CLOSED, "park lid rock-solid at ~44deg", R_LID),
        ],
        right=[
            (0.65, _PanelPos(-0.215, y_tab_loc, 0.135), FINGER_CLOSED, "step -X well clear of box", R_FLAP_R),
            (1.00, _PanelPos(-0.215, y_tab_loc, 0.295), FINGER_CLOSED, "reposition wrist outside tab", R_TAB),
        ],
    )

    t_snap3 = P.phase(
        left=[(0.30, _lid_push_tcp(48.0, rad=0.144, preload=0.0025), FINGER_CLOSED, "quiet snapshot 3", R_LID)],
        right=[(0.30, _PanelPos(-0.215, y_tab_loc, 0.295), FINGER_CLOSED, "hold before tab crease", R_TAB)],
    )
    _ct["snap_wins"].append((t_snap3 + 0.06, t_snap3 + 0.24, False))

    # -----------------------------------------------------------------------------------------------------------------
    # PHASE 4: Left holds lid at ~44 deg (proven 0.0 mm shift in probe22) while Right's thin spatula sweeps along
    #          r = 26..30 mm (+65 -> -61 deg, force aligned through LID_ANCHOR!) onto lid_tab's -58 deg stop!
    # -----------------------------------------------------------------------------------------------------------------
    P.phase(
        left=[(4.15, _lid_push_tcp(48.0, rad=0.144, preload=0.0025), FINGER_CLOSED, "hold lid anvil for tab crease", R_LID)],
        right=[
            (0.55, _PanelTabArc(+65.0, 0.032, y_tab_loc), FINGER_CLOSED, "above outward front tab", R_TAB),
            (0.60, _PanelTabArc(+25.0, 0.026, y_tab_loc), FINGER_CLOSED, "catch outward tab along r=26mm arc", R_TAB),
            (0.60, _PanelTabArc( -8.0, 0.026, y_tab_loc), FINGER_CLOSED, "sweep tab past vertical", R_TAB),
            (0.60, _PanelTabArc(-38.0, 0.026, y_tab_loc), FINGER_CLOSED, "fold tab inward along r=26mm arc", R_TAB),
            (0.60, _PanelTabArc(-61.0, 0.030, y_tab_loc), FINGER_CLOSED, "crease tab inward against -58deg stop", R_TAB),
            (0.30, _PanelTabArc(-61.0, 0.030, y_tab_loc), FINGER_CLOSED, "pause to lock crease stiction", R_TAB),
            (0.45, _PanelTabArc(-25.0, 0.048, y_tab_loc), FINGER_CLOSED, "release radially below creased tab", R_TAB),
            (0.45, _PanelPos(-0.172, y_tab_loc, 0.155), FINGER_CLOSED, "reposition outside front panel", R_FLAP_R),
        ],
    )

    P.phase(
        left=[
            (0.80, _lid_push_tcp(26.0, rad=0.152, preload=0.0035), FINGER_CLOSED, "lower lid to 26deg", R_LID),
            (0.80, _lid_push_tcp(10.0, rad=0.160, preload=0.0035), FINGER_CLOSED, "guide creased tab behind front wall", R_LID),
            (0.80, _lid_push_tcp(3.0, rad=0.164, preload=0.0055), FINGER_CLOSED, "lower lid to 3deg", R_LID),
            (1.00, _lid_push_tcp(0.0, rad=0.168, preload=0.0070), FINGER_CLOSED, "seat lid flat onto walls", R_LID),
            (0.70, _lid_push_tcp(0.0, rad=0.188, preload=0.0070), FINGER_CLOSED, "press front rail flush", R_LID),
            (0.60, _PanelPos(-0.055, y_lid_lo, az_l_loc + 0.092), OPEN_WIDE, "lift off closed box", R_LID),
        ],
        right=[
            (0.70, _PanelPos(-0.1500, y_tab_loc, 0.102), FINGER_CLOSED, "guard outer front panel vertical", R_FLAP_R),
            (2.70, _PanelPos(-0.1500, y_tab_loc, 0.102), FINGER_CLOSED, "hold front panel guard during tuck", R_FLAP_R),
            (0.65, _PanelPos(-0.185, y_tab_loc, 0.145), OPEN_WIDE, "withdraw from front panel", R_FLAP_R),
            (0.65, RIGHT_PARK, OPEN_WIDE, "park right", R0),
        ],
    )

    if mode == "near_hand":
        near_pos = _PanelPos(-0.055, y_lid_lo, az_l_loc + 0.068)
        P.phase(
            left=[(0.8, near_pos, OPEN_WIDE, "hover 66mm near lid", R_LID), (2.2, near_pos, OPEN_WIDE, "hold near", R_LID)],
            right=[(0.8, RIGHT_PARK, OPEN_WIDE, "park right", R0), (2.2, RIGHT_PARK, OPEN_WIDE, "hold right", R0)],
        )
        return {"left": P.L, "right": P.R}

    P.phase(
        left=[(1.0, LEFT_PARK, OPEN_WIDE, "park left", R0), (2.6, LEFT_PARK, OPEN_WIDE, "settled", R0)],
        right=[(1.0, RIGHT_PARK, OPEN_WIDE, "park right", R0), (2.6, RIGHT_PARK, OPEN_WIDE, "settled", R0)],
    )
    return {"left": P.L, "right": P.R}

# -- Live per-step safety latches & 4-tier scorer ---------------------------------------------------------------------

def _robot_tip_and_pad_points(env) -> list[np.ndarray]:
    from yam_arena import runtime as _rt

    base = env.unwrapped
    origin = base.scene.env_origins[0].cpu().numpy()
    pts: list[np.ndarray] = []
    for rname in ("left_robot", "right_robot"):
        robot = base.scene[rname]
        mid = _rt.pad_tips_world(robot) - origin
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        j7, j8 = (abs(float(v)) for v in robot.data.joint_pos.torch[0, ids].cpu().numpy())
        pts.extend([mid - j7 * g[:3, 0], mid + j8 * g[:3, 0], mid])
    return pts


def _live_robot_world_hulls(env) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    from yam_arena import runtime as _rt

    base = env.unwrapped
    origin = base.scene.env_origins[0].cpu().numpy()
    hulls = get_robot_urdf_hulls()
    out = []
    for rname in ("left_robot", "right_robot"):
        for link, T in _rt.link_poses(base.scene[rname]).items():
            for v, eq in hulls.get(link, []):
                R, p = T[:3, :3], T[:3, 3] - origin
                w = v @ R.T + p
                nw = eq[:, :3] @ R.T
                out.append((f"{rname}:{link}", w, np.c_[nw, eq[:, 3] - nw @ p], w.min(0), w.max(0)))
    return out


_STATE = {
    "step": 0,
    "cake_on_board0": None,
    "frosting_touch": None,
    "cake_disturbed": False,
    "board_dropped": False,
    "box_crushed": False,
    "hand_clear_m": 0.0,
    "still_move_m": 1.0,
    "still_ok": False,
    "hist": [],
    "hc_hist": [],
    "last_diag": {},
}


@on_reset
def _reset_watch(_env):
    _STATE.update(
        step=0,
        cake_on_board0=None,
        frosting_touch=None,
        cake_disturbed=False,
        board_dropped=False,
        box_crushed=False,
        hand_clear_m=0.0,
        still_move_m=1.0,
        still_ok=False,
        hist=[],
        hc_hist=[],
        last_diag={},
    )


@on_step
def _step_watch(env):
    _STATE["step"] += 1
    bx, pl, ck = (labkit._part_state(env, p) for p in (CAKE_BOX, CAKE_BOARD, CAKE))
    bpos, brot = np.asarray(bx.pos, float), np.asarray(bx.rotation(), float)
    plpos, plrot = np.asarray(pl.pos, float), np.asarray(pl.rotation(), float)
    ckpos, ckrot = np.asarray(ck.pos, float), np.asarray(ck.rotation(), float)

    qf = float(bx.joints.get("front_panel", FRONT_INIT))
    ql = float(bx.joints.get("lid", LID_INIT))
    qt = float(bx.joints.get("lid_tab", TAB_INIT))

    if _STATE["cake_on_board0"] is None:
        _STATE["cake_on_board0"] = plrot.T @ (ckpos - plpos)

    if _STATE["frosting_touch"] is None:
        panel_hit = box_panel_touches_frosting(bpos, brot, qf, ql, qt, ckpos, ckrot, FROSTING_CLEAR_M)
        if panel_hit is not None:
            _STATE["frosting_touch"] = f"box:{panel_hit}@step{_STATE['step']}"
        else:
            rob_hit = robot_hulls_touch_frosting(_live_robot_world_hulls(env), ckpos, ckrot, FROSTING_CLEAR_M)
            if rob_hit is not None:
                _STATE["frosting_touch"] = f"{rob_hit}@step{_STATE['step']}"

    sv = step_violation_metrics(TOP, bpos, brot, qf, ql, qt, plpos, plrot, ckpos, ckrot, _STATE["cake_on_board0"])
    if sv["cake_disturbed"]:
        _STATE["cake_disturbed"] = True
    if sv["board_dropped"]:
        _STATE["board_dropped"] = True
    if sv["box_crushed"]:
        _STATE["box_crushed"] = True

    pad_pts = _robot_tip_and_pad_points(env)
    hc = hand_clearance_m(pad_pts, bpos, brot, qf, ql, qt, plpos, plrot, ckpos, ckrot)
    _STATE["hand_clear_m"] = hc

    snap = tracked_world_points(bpos, brot, qf, ql, qt, plpos, plrot, ckpos, ckrot)
    hist = _STATE["hist"]
    hc_hist = _STATE["hc_hist"]
    if hc >= HAND_CLEAR_M:
        hist.append(snap)
        hc_hist.append(float(hc))
        del hist[:-STILL_STEPS]
        del hc_hist[:-STILL_STEPS]
    else:
        hist.clear()
        hc_hist.clear()
    if len(hist) == STILL_STEPS:
        H = np.asarray(hist, float)
        mv = float(np.max(np.linalg.norm(H[:, None] - H[None, :], axis=-1)))
        _STATE["still_move_m"] = mv
        _STATE["still_ok"] = (mv < STILL_TOL_M) and (min(hc_hist) >= HAND_CLEAR_M)
    else:
        _STATE["still_move_m"] = 1.0
        _STATE["still_ok"] = False

    tier_info = evaluate_final_tier(bpos, brot, qf, ql, qt, plpos, plrot)
    _STATE["last_diag"] = {
        "step": _STATE["step"],
        "q_front_deg": round(math.degrees(qf), 2),
        "q_lid_deg": round(math.degrees(ql), 2),
        "q_tab_deg": round(math.degrees(qt), 2),
        "frosting_touch": _STATE["frosting_touch"],
        "cake_disturbed": _STATE["cake_disturbed"],
        "board_dropped": _STATE["board_dropped"],
        "box_crushed": _STATE["box_crushed"],
        "hand_clear_mm": round(hc * 1000.0, 2),
        "min_still_hand_clear_mm": round(min(hc_hist) * 1000.0, 2) if hc_hist else 0.0,
        "still_samples_at_80mm": len(hc_hist),
        "still_mm": round(_STATE["still_move_m"] * 1000.0, 3),
        "still_ok": _STATE["still_ok"],
        **sv,
        **tier_info,
    }
    diag_path = os.environ.get("CAKE_BOX_DIAG_JSON")
    if diag_path:
        pathlib.Path(diag_path).write_text(json.dumps(_STATE["last_diag"], indent=2))
    if os.environ.get("CAKE_BOX_TRACE") and (_STATE["step"] == 1 or _STATE["step"] % 20 == 0):
        rz_mm = (_live_arm_tip_z_min(env, "right_robot") - TOP) * 1000.0
        lz_mm = (_live_arm_tip_z_min(env, "left_robot") - TOP) * 1000.0
        pl_box = brot.T @ (plpos - bpos)
        b_sh = (bpos[:2] - BOX_XY) * 1000.0
        b_dyaw = math.degrees(math.atan2(brot[1, 0], brot[0, 0]) - BOX_YAW)
        print(
            f"[trace step={_STATE['step']:4d} t={_STATE['step']/20.0:5.1f}s] "
            f"pl_in_box=({pl_box[0]*1000:+6.1f},{pl_box[1]*1000:+6.1f})mm "
            f"box_sh=({b_sh[0]:+5.1f},{b_sh[1]:+5.1f},{b_dyaw:+4.1f}d) "
            f"q=({math.degrees(qf):+6.1f},{math.degrees(ql):+5.1f},{math.degrees(qt):+6.1f})deg "
            f"tipZ=(L:{lz_mm:+5.1f},R:{rz_mm:+5.1f})mm "
            f"ck_sl={sv['cake_slide_mm']:5.2f}mm touch={_STATE['frosting_touch']} tier={tier_info['tier']}",
            flush=True,
        )


def score(objects) -> float:
    if (
        _STATE["frosting_touch"] is not None
        or _STATE["cake_disturbed"]
        or _STATE["board_dropped"]
        or _STATE["box_crushed"]
    ):
        return 0.0
    if _STATE["hand_clear_m"] < HAND_CLEAR_M or not _STATE["still_ok"]:
        return 0.0

    bx, pl = objects["cake_box"], objects["cake_board"]
    qf = float(bx.joints.get("front_panel", FRONT_INIT))
    ql = float(bx.joints.get("lid", LID_INIT))
    qt = float(bx.joints.get("lid_tab", TAB_INIT))
    info = evaluate_final_tier(
        np.asarray(bx.pos, float),
        np.asarray(bx.rotation(), float),
        qf,
        ql,
        qt,
        np.asarray(pl.pos, float),
        np.asarray(pl.rotation(), float),
    )
    return float(info["tier"])


def success(objects) -> bool:
    return score(objects) >= 1.0


def distance(objects) -> float:
    bx, pl = objects["cake_box"], objects["cake_board"]
    return float(np.linalg.norm(np.asarray(pl.pos, float)[:2] - np.asarray(bx.pos, float)[:2]))
