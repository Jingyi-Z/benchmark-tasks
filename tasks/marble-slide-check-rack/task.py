"""Post an incoming guest check and clear a completed check from a slide check rack.

Station tabletop task (yam_arena): a rear-cantilevered PLA pass riser holds an 18 in anodized aluminum
slide check rack (San Jamar CK6518A) with loose glass spheres on an internal ramp above an open 4.0 mm
bottom throat. Clear the completed pink guest check (NCCO #526) from the left side of the rack into the
quarter-size aluminum out-tray (Winco ALXP-1310H), and post the incoming green guest check (NCCO #525)
from the 1.5 mm clear slot of the input stand into the right side of the rack.

Manufacturer Facts:
  - San Jamar CK6518A 18" Slide Check Rack: 50.8 mm H x 457.2 mm W x 19.05 mm D (2" x 18" x 3/4"),
    anodized aluminum extrusion, screw-on end caps, loose green glass beads, exposed lower rear apron.
  - NCCO #526 Single-Copy Guest Check: 88.9 mm W x 171.5 mm H (3.5" x 6.75") medium cardboard stock;
    shipping case weight 12.0 lb (5,443.1 g) / 2,500 ct = 2.177 g/check GROSS including packaging.
  - Winco ALXP-1310 Quarter-Size Aluminum Sheet Pan (out-tray): 241.3 mm x 330.2 mm x 25.4 mm
    (9.5" x 13.0" x 1.0" catalog outer dimensions; 4.0 mm floor, 4.5 mm rim wall width, and resulting
    232.3 mm x 321.2 mm clear inner bed are modeled engineering estimates).

Modeled Engineering Estimates (Disclosed; Stock Rack Geometry Identical to prototype-m1 / prototype-m2):
  - Net check mass: 2.00 g nominal after ~0.177 g/check (~8%) packaging deduction (MSCR_CHECK_MASS_G).
  - Caliper & stiffness: 0.22 mm nominal caliper (MSCR_THICKNESS_MM) and 3.0 GPa nominal modulus across
    4 horizontal articulated strips (31.5 / 45.0 / 47.5 / 47.5 mm = 171.5 mm, top-strip root, k_theta = E*I/dL,
    MSCR_STIFFNESS_SCALE).
  - Continuous root-strip paper collider (m3 CPU fix): root_boxes[0] centered at z = -0.0020 m (height 27.50 mm,
    spanning [-15.75, +11.75] mm) meets root_boxes[1] header centered at z = +0.01375 m (height 4.00 mm,
    spanning [+11.75, +15.75] mm) with 0.000 mm gap, 0.000 mm overlap, exact root CoM = (0, 0, 0), and
    exact uniform-slab inertia, meeting hinge_01 at -15.75 mm.
  - Internal rack channel: 1.8 mm wall/roof/ramp thickness, 12.50 mm lower rear registration apron,
    4.00 mm bottom throat, 24.5 deg front ramp, 31 loose 12.7 mm (2.68 g) glass spheres.
  - Surface frictions (average combine): mu_bead = 0.11, mu_rack = 0.18, mu_check = 0.30 (MSCR_FRICTION_SCALE).
  - Closer mid-card grasp (exact values): Check B grasped 104.0 mm below top edge (strip_2, el_R = -4.5 deg,
    az_R = 90.0 deg); Check A grasped 102.0 mm below top edge (strip_2, el_L = -8.0 deg, az_L = 88.2 deg).
  - Pair-level authored-collider 3D SAT witness: builds OBBs directly from authored body-local colliders
    (body/box_0, body/box_1, hinge_01/box_0, hinge_12/box_0, hinge_23/box_0) and logs full body poses/quaternions,
    winning collider pairs, contact normals, overlaps, all-corner bounds, and XZ cross-sections.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import xml.etree.ElementTree as _ET

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401
import labkit
from labkit import GLASS_COLORS, Joint, Mesh, on_reset, on_step, part

from goal_scorer import (
    AuthoredColliderBox,
    GoalScorer,
    ScorerSceneGeometry,
    WorldOBB,
    WorldSample,
    card_world_obbs_and_com,
    extract_authored_colliders,
    quat_wxyz_to_mat,
    static_part_world_obbs,
)
from obs_pose_adapter import adapt_articulation_body_poses

import yam_arena.demo as dm
from yam_arena import YAM_URDF as _YAM_URDF, ik
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP
from yam_arena.urdf import axis_angle_matrix as _urdf_aa, parse_joints as _parse_urdf_joints, rpy_matrix as _urdf_rpy

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 1200
SUCCESS_HOLD_S = 1.0

# -- Numerical & Parameter Sensitivity Hooks (read before part() calls) -------------------------
_CONTACT_OFFSET = float(os.environ.get("MSCR_CONTACT_OFFSET", "0.0002"))
_PHYSICS_HZ = float(os.environ.get("MSCR_PHYSICS_HZ", "0"))  # 0 -> default 240 Hz; 480 -> dt = 1/480 s
labkit.CONTACT_OFFSET = _CONTACT_OFFSET
if _PHYSICS_HZ > 0:
    labkit.PHYSICS_DT_OVERRIDE = 1.0 / _PHYSICS_HZ

THICK_MM = float(os.environ.get("MSCR_THICKNESS_MM", "0.22"))
STIFFNESS_SCALE = float(os.environ.get("MSCR_STIFFNESS_SCALE", "1.0"))
FRICTION_SCALE = float(os.environ.get("MSCR_FRICTION_SCALE", "1.0"))
CHECK_MASS_G = float(os.environ.get("MSCR_CHECK_MASS_G", "2.00"))
RAMP_DEG = float(os.environ.get("MSCR_RAMP_DEG", "24.5"))

_MODE = os.environ.get("MSCR_MODE", "full_proto").strip()
_LOG_JSON = os.environ.get("MSCR_LOG_JSON", "").strip()
_ENABLE_SCORER = os.environ.get("MSCR_ENABLE_SCORER", "1").strip() == "1"

# -- Manufacturer Outer Dimensions (Verified) --------------------------------------------------
RACK_OUTER_H = 0.05080  # 50.80 mm (2.00")
RACK_OUTER_W = 0.45720  # 457.20 mm (18.00")
RACK_OUTER_D = 0.01905  # 19.05 mm (0.75")

CHECK_W = 0.08890       # 88.90 mm (3.50")
CHECK_H = 0.17150       # 171.50 mm (6.75")
CHECK_GROSS_MASS_G_REF = 12.0 * 453.59237 / 2500.0  # 2.17724 g/check gross including packaging

# Winco ALXP-1310H Quarter-Size Sheet Pan (9.63" x 13.00" x 1.13" official outer; floor/wall/inner bed estimated):
OUT_TRAY_OUTER_X = 0.24450  # 244.50 mm (9.63" official outer width)
OUT_TRAY_OUTER_Y = 0.33020  # 330.20 mm (13.00" official outer length)
OUT_TRAY_OUTER_Z = 0.02860  # 28.60 mm (1.13" official outer height)
OUT_TRAY_FLOOR_T = 0.00400  # 4.00 mm unmeasured estimated floor thickness
OUT_TRAY_WALL_T = 0.00450   # 4.50 mm unmeasured estimated rim wall width
OUT_TRAY_INNER_X = OUT_TRAY_OUTER_X - 2.0 * OUT_TRAY_WALL_T  # 0.23550 m estimated clear inner X
OUT_TRAY_INNER_Y = OUT_TRAY_OUTER_Y - 2.0 * OUT_TRAY_WALL_T  # 0.32120 m estimated clear inner Y

# -- Modeled Engineering Estimates (Disclosed; Stock Rack Geometry Identical to prototype-m1/m2) -
CHECK_THICK = THICK_MM * 1e-3
CHECK_MASS_KG = CHECK_MASS_G * 1e-3
CHECK_MODULUS_PA = 3.0e9
STRIP_HEIGHTS = (0.03150, 0.04500, 0.04750, 0.04750)  # sum = 0.17150 m exact
_HINGE_DL = 0.04500
_I_AREA = CHECK_W * (CHECK_THICK ** 3) / 12.0
HINGE_K = STIFFNESS_SCALE * (CHECK_MODULUS_PA * _I_AREA / _HINGE_DL)  # ~0.00526 N*m/rad at 0.22 mm
HINGE_D = 0.075 * HINGE_K

WALL_T = 0.00180
ENDCAP_T = 0.00200
Z_APRON_LIP = 0.01250   # 12.50 mm exposed lower rear registration apron below front lip
THROAT_GAP = 0.00400    # 4.00 mm lower throat opening between rear wall and front lip
X_LIP_INNER = -(WALL_T + THROAT_GAP)  # -0.00580 m in rack frame (rear outer face at x=0)

BEAD_RADIUS = 0.00635   # 6.35 mm radius (12.70 mm / 0.50" diameter)
BEAD_MASS = 0.00268     # 2.68 g at 2.52 g/cm^3 soda-lime glass
BEAD_COUNT = 31

MU_BEAD = 0.11 * FRICTION_SCALE
MU_RACK = 0.18 * FRICTION_SCALE
MU_CHECK = 0.30 * FRICTION_SCALE

# -- Scene Placement ---------------------------------------------------------------------------
RISER_CLEAR_H = 0.27000
RACK_X0 = 0.37500
RACK_Y0 = 0.00000
Z_RACK_BOT = TOP + RISER_CLEAR_H  # 1.01750 m
X_WALL_WORLD = RACK_X0 - WALL_T   # 0.37320 m (inner check-contact face of rear wall)
CHECK_RACK_CLEAR_SPAN_Y = RACK_OUTER_W - 2.0 * ENDCAP_T  # 0.45320 m

CHECK_A_Y0 = +0.11500
CHECK_B_INSERT_Y = -0.11000
TARGET_B_ZONE_Y = (-0.20000, -0.02000)

IN_STAND_X0 = 0.25500
IN_STAND_Y0 = -0.05500
OUT_TRAY_X0 = 0.25500
OUT_TRAY_Y0 = +0.18000

# -- Colors ------------------------------------------------------------------------------------
C_ALUM = (0.77, 0.79, 0.82)
C_STEEL = (0.70, 0.72, 0.75)
C_DARK = (0.20, 0.22, 0.25)
C_BEAD = (0.14, 0.62, 0.34)
C_CHECK_A = (0.96, 0.76, 0.80)  # NCCO #526 pink 18-line single-copy cardboard check
C_CHECK_B = (0.76, 0.90, 0.79)  # NCCO #525 green 18-line single-copy cardboard check
C_RULE_A = (0.68, 0.32, 0.40)   # printed crimson/rose ruling lines on #526 pink check
C_RULE_B = (0.26, 0.52, 0.36)   # printed green ruling lines on #525 green check
C_INK = (0.14, 0.18, 0.38)      # dark ballpoint order marks on completed #526 check
C_HEADER = (0.78, 0.26, 0.24)   # printed red check-number header band
GLASS_COLORS.add(C_BEAD)


# -- 1. San Jamar CK6518A Rack Geometry (100% Unchanged Stock Model) ---------------------------
_ALPHA = math.radians(RAMP_DEG)
_CA, _SA = math.cos(_ALPHA), math.sin(_ALPHA)
L_RAMP = (RACK_OUTER_D + X_LIP_INNER - WALL_T * _CA) / _SA
_RAMP_CX = X_LIP_INNER - (0.5 * WALL_T * _CA + 0.5 * L_RAMP * _SA)
_RAMP_CZ = Z_APRON_LIP - (0.5 * WALL_T * _SA - 0.5 * L_RAMP * _CA)
_Z_RAMP_TOP_OUTER = _RAMP_CZ + (-0.5 * WALL_T * _SA + 0.5 * L_RAMP * _CA)
_FASCIA_H = max(0.0040, RACK_OUTER_H - _Z_RAMP_TOP_OUTER)

RACK_BOX_NAMES = (
    "rack/box_0_rear_wall",
    "rack/box_1_top_roof",
    "rack/box_2_front_fascia",
    "rack/box_3_front_ramp",
    "rack/box_4_endcap_pos_y",
    "rack/box_5_endcap_neg_y",
)

RACK_BOXES = (
    # Continuous rear mounting wall + exposed lower 12.50 mm registration apron:
    ((-0.5 * WALL_T, 0.0, 0.5 * RACK_OUTER_H), (WALL_T, CHECK_RACK_CLEAR_SPAN_Y, RACK_OUTER_H), C_ALUM, 0.0),
    # Top channel roof:
    ((-0.5 * RACK_OUTER_D, 0.0, RACK_OUTER_H - 0.5 * WALL_T), (RACK_OUTER_D, CHECK_RACK_CLEAR_SPAN_Y, WALL_T), C_ALUM, 0.0),
    # Upper front fascia capping exact 19.05 mm outer depth:
    ((-RACK_OUTER_D + 0.5 * WALL_T, 0.0, RACK_OUTER_H - 0.5 * _FASCIA_H), (WALL_T, CHECK_RACK_CLEAR_SPAN_Y, _FASCIA_H), C_ALUM, 0.0),
    # Pitched front bead ramp (slope = RAMP_DEG from vertical, 4.00 mm bottom throat):
    ((_RAMP_CX, 0.0, _RAMP_CZ), (WALL_T, CHECK_RACK_CLEAR_SPAN_Y, L_RAMP), C_ALUM, (0.0, -_ALPHA)),
    # Left & right screw-on end caps (exact 457.20 mm total width):
    ((-0.5 * RACK_OUTER_D, +0.5 * (RACK_OUTER_W - ENDCAP_T), 0.5 * RACK_OUTER_H), (RACK_OUTER_D, ENDCAP_T, RACK_OUTER_H), C_DARK, 0.0),
    ((-0.5 * RACK_OUTER_D, -0.5 * (RACK_OUTER_W - ENDCAP_T), 0.5 * RACK_OUTER_H), (RACK_OUTER_D, ENDCAP_T, RACK_OUTER_H), C_DARK, 0.0),
)

CHECK_RACK = part(
    "check_rack",
    pos=(RACK_X0, RACK_Y0, Z_RACK_BOT),
    boxes=RACK_BOXES,
    fixed=True,
    friction=MU_RACK,
)

# -- 2. Open-Underneath Rear-Cantilevered Stainless Pass-Through Riser (Unchanged) -------------
RISER_H_TOTAL = RISER_CLEAR_H + RACK_OUTER_H
RISER_BOX_NAMES = (
    "riser/box_0_mount_beam",
    "riser/box_1_top_canopy",
    "riser/box_2_post_pos_y",
    "riser/box_3_post_neg_y",
    "riser/box_4_foot_pos_y",
    "riser/box_5_foot_neg_y",
)
RISER_BOXES = (
    ((+0.00720, 0.0, RISER_CLEAR_H + 0.5 * RACK_OUTER_H), (0.01400, 0.47000, RACK_OUTER_H), C_STEEL, 0.0),
    ((+0.08020, 0.0, RISER_H_TOTAL - 0.00600), (0.16000, 0.52000, 0.01200), C_STEEL, 0.0),
    ((+0.13500, +0.24500, 0.5 * (RISER_H_TOTAL - 0.01200)), (0.02200, 0.03000, RISER_H_TOTAL - 0.01200), C_STEEL, 0.0),
    ((+0.13500, -0.24500, 0.5 * (RISER_H_TOTAL - 0.01200)), (0.02200, 0.03000, RISER_H_TOTAL - 0.01200), C_STEEL, 0.0),
    ((+0.12500, +0.24500, 0.00400), (0.09000, 0.06000, 0.00800), C_STEEL, 0.0),
    ((+0.12500, -0.24500, 0.00400), (0.09000, 0.06000, 0.00800), C_STEEL, 0.0),
)

PASS_RISER = part(
    "pass_riser",
    pos=(RACK_X0, RACK_Y0, TOP),
    boxes=RISER_BOXES,
    fixed=True,
    friction=0.45,
)


# -- 3. 31 Loose Dynamic Glass Spheres (Unchanged) ---------------------------------------------
def _resting_bead_xz_rel(extra_back_gap: float) -> tuple[float, float]:
    skin = 0.00008
    x_b = -WALL_T - extra_back_gap - BEAD_RADIUS - skin
    z_b = Z_APRON_LIP + ((BEAD_RADIUS + skin) - (x_b - X_LIP_INNER) * _CA) / _SA
    return x_b, z_b


_BEAD_Y_SPAN = 0.42400
BEAD_PARTS = []
for _idx in range(BEAD_COUNT):
    _yb = -0.5 * _BEAD_Y_SPAN + _idx * (_BEAD_Y_SPAN / (BEAD_COUNT - 1))
    _over_a = abs(_yb - CHECK_A_Y0) <= (0.5 * CHECK_W + BEAD_RADIUS + 0.0030)
    _xb_rel, _zb_rel = _resting_bead_xz_rel((CHECK_THICK + 0.00008) if _over_a else 0.0)
    BEAD_PARTS.append(
        part(
            f"bead_{_idx:02d}",
            pos=(RACK_X0 + _xb_rel, RACK_Y0 + _yb, Z_RACK_BOT + _zb_rel),
            spheres=(((0.0, 0.0, 0.0), BEAD_RADIUS, C_BEAD),),
            mass=BEAD_MASS,
            friction=MU_BEAD,
            angular_damping=0.05,
            enable_ccd=True,
        )
    )


# -- 4. Articulated 4-Strip NCCO #526 Guest Check Stock (Continuous Root Collider Fix) ---------
CHECK_TOP_FROM_ROOT = +0.01575
CHECK_BOT_FROM_ROOT = -0.15575
HEADER_H = 0.00400  # 4.00 mm printed red header band


def _strip_print_meshes(
    z_lo: float,
    z_hi: float,
    rule_color: tuple[float, float, float],
    is_root: bool = False,
    is_bottom: bool = False,
    with_order_ink: bool = False,
) -> tuple[Mesh, ...]:
    """Visual-only (`collide='none'`) 18-line guest-check ruling grid and optional order strokes."""
    x_f = 0.5 * CHECK_THICK + 0.000018
    y_margin = 0.5 * CHECK_W - 0.0040  # +-40.45 mm printed frame width
    y_qty = +0.0305                    # left quantity column divider
    y_dlr = -0.0215                    # right dollars column divider
    y_cnt = -0.0345                    # right cents column divider

    rule_rects: list[tuple[float, float, float, float]] = []
    # Outer vertical frame borders + qty/dollars/cents column dividers across this strip:
    z_v_hi = (z_hi - HEADER_H - 0.0010) if is_root else (z_hi - 0.0004)
    z_v_lo = (z_lo + 0.0030) if is_bottom else (z_lo + 0.0004)
    for y_col, w_col in ((+y_margin, 0.0007), (-y_margin, 0.0007), (y_qty, 0.0006), (y_dlr, 0.0006), (y_cnt, 0.0005)):
        rule_rects.append((y_col - 0.5 * w_col, y_col + 0.5 * w_col, z_v_lo, z_v_hi))

    if is_root:
        # Header info box grid (DATE | TABLE | GUESTS | SERVER) + double rule + first 2 order lines:
        for z_r, t_r in ((+0.0102, 0.0008), (+0.0032, 0.0009), (+0.0016, 0.0005), (-0.0062, 0.00055), (-0.0132, 0.00055)):
            rule_rects.append((-y_margin, +y_margin, z_r - 0.5 * t_r, z_r + 0.5 * t_r))
        for y_h in (+0.018, -0.004, -0.022):
            rule_rects.append((y_h - 0.0003, y_h + 0.0003, +0.0032, +0.0102))
    else:
        n_lines = 5
        step_z = (z_hi - z_lo) / (n_lines + 0.4)
        for k in range(1, n_lines + 1):
            z_r = z_hi - k * step_z
            if z_r > z_lo + 0.0025:
                rule_rects.append((-y_margin, +y_margin, z_r - 0.00028, z_r + 0.00028))
        if is_bottom:
            # Double total rule near bottom of check:
            for z_tot in (z_lo + 0.0085, z_lo + 0.0032):
                rule_rects.append((-y_margin, +y_margin, z_tot - 0.00045, z_tot + 0.00045))

    def _rects_to_mesh(rects: list[tuple[float, float, float, float]], col: tuple[float, float, float]) -> Mesh:
        pts = []
        tris = []
        for x_s in (-x_f, +x_f):
            for y0, y1, za, zb in rects:
                base = len(pts)
                pts.extend([(x_s, y0, za), (x_s, y1, za), (x_s, y1, zb), (x_s, y0, zb)])
                tris.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
        return Mesh(points=np.asarray(pts, dtype=float), tris=np.asarray(tris, dtype=np.int32), color=col, collide="none")

    out = [_rects_to_mesh(rule_rects, rule_color)]

    if with_order_ink:
        ink_rects: list[tuple[float, float, float, float]] = []
        if is_root:
            ink_rects.append((-0.014, +0.026, -0.0048, -0.0034))
            ink_rects.append((-0.032, -0.023, -0.0048, -0.0034))
            ink_rects.append((-0.006, +0.027, -0.0118, -0.0104))
            ink_rects.append((-0.031, -0.023, -0.0118, -0.0104))
        elif not is_bottom:
            for frac, y_lo_i in ((0.24, -0.012), (0.48, -0.018), (0.72, -0.005)):
                zi = z_hi - frac * (z_hi - z_lo)
                ink_rects.append((y_lo_i, +0.027, zi, zi + 0.0014))
                ink_rects.append((-0.032, -0.023, zi, zi + 0.0014))
        else:
            ink_rects.append((-0.033, -0.022, z_lo + 0.0050, z_lo + 0.0068))
        if ink_rects:
            out.append(_rects_to_mesh(ink_rects, C_INK))

    return tuple(out)


def make_check(
    name: str,
    pos: tuple[float, float, float],
    card_color: tuple[float, float, float],
    rule_color: tuple[float, float, float],
    with_order_ink: bool = False,
    pitch: float = 0.0,
):
    h0, h1, h2, h3 = STRIP_HEIGHTS  # 31.5, 45.0, 47.5, 47.5 mm
    m_tot = CHECK_MASS_KG
    m0 = m_tot * (h0 / CHECK_H)
    m1 = m_tot * (h1 / CHECK_H)
    m2 = m_tot * (h2 / CHECK_H)
    m3 = m_tot * (h3 / CHECK_H)

    z_h01 = -0.5 * h0            # -0.01575
    z_h12 = z_h01 - h1           # -0.06075
    z_h23 = z_h12 - h2           # -0.10825

    j_01 = Joint(
        name="hinge_01",
        kind="hinge",
        axis="Y",
        anchor=(0.0, 0.0, z_h01),
        boxes=(((0.0, 0.0, z_h01 - 0.5 * h1), (CHECK_THICK, CHECK_W, h1), card_color, 0.0),),
        meshes=_strip_print_meshes(z_h12, z_h01, rule_color, is_root=False, is_bottom=False, with_order_ink=with_order_ink),
        lower=-1.15,
        upper=+1.15,
        stiffness=HINGE_K,
        damping=HINGE_D,
        target=0.0,
        initial=0.0,
        mass=m1,
        parent="",
    )
    j_12 = Joint(
        name="hinge_12",
        kind="hinge",
        axis="Y",
        anchor=(0.0, 0.0, z_h12),
        boxes=(((0.0, 0.0, z_h12 - 0.5 * h2), (CHECK_THICK, CHECK_W, h2), card_color, 0.0),),
        meshes=_strip_print_meshes(z_h23, z_h12, rule_color, is_root=False, is_bottom=False, with_order_ink=False),
        lower=-1.15,
        upper=+1.15,
        stiffness=HINGE_K,
        damping=HINGE_D,
        target=0.0,
        initial=0.0,
        mass=m2,
        parent="hinge_01",
    )
    j_23 = Joint(
        name="hinge_23",
        kind="hinge",
        axis="Y",
        anchor=(0.0, 0.0, z_h23),
        boxes=(((0.0, 0.0, z_h23 - 0.5 * h3), (CHECK_THICK, CHECK_W, h3), card_color, 0.0),),
        meshes=_strip_print_meshes(z_h23 - h3, z_h23, rule_color, is_root=False, is_bottom=True, with_order_ink=with_order_ink),
        lower=-1.15,
        upper=+1.15,
        stiffness=HINGE_K,
        damping=HINGE_D,
        target=0.0,
        initial=0.0,
        mass=m3,
        parent="hinge_12",
    )

    # Continuous root rectangle [-15.75, +15.75] mm with ZERO gap and ZERO self-overlap:
    root_boxes = (
        ((0.0, 0.0, -0.5 * HEADER_H), (CHECK_THICK, CHECK_W, h0 - HEADER_H), card_color, 0.0),
        ((0.0, 0.0, +0.5 * (h0 - HEADER_H)), (CHECK_THICK, CHECK_W, HEADER_H), C_HEADER, 0.0),
    )
    root_meshes = _strip_print_meshes(-0.5 * h0, +0.5 * h0, rule_color, is_root=True, is_bottom=False, with_order_ink=with_order_ink)

    return part(
        name,
        pos=pos,
        pitch=pitch,
        boxes=root_boxes,
        meshes=root_meshes,
        joints=(j_01, j_12, j_23),
        mass=m0,
        friction=MU_CHECK,
        self_collide=False,
        angular_damping=0.4,
        enable_ccd=True,
    )


_CHECK_A_X0 = X_WALL_WORLD - 0.5 * CHECK_THICK - 0.00004
_CHECK_A_Z0 = (Z_RACK_BOT + 0.04100) - CHECK_TOP_FROM_ROOT
CHECK_A = make_check("check_a", (_CHECK_A_X0, CHECK_A_Y0, _CHECK_A_Z0), C_CHECK_A, C_RULE_A, with_order_ink=True, pitch=0.0)

# Input stand: 1.50 mm clear vertical slot between two 5.00 mm walls on 6.50 mm centres (x = -3.25 / +3.25 mm):
IN_STAND_BOXES = (
    ((+0.00200, +0.02000, 0.03100), (0.06000, 0.05600, 0.06200), C_DARK, 0.0),
    ((-0.00325, +0.02000, 0.07300), (0.00500, 0.05200, 0.02200), C_STEEL, 0.0),
    ((+0.00325, +0.02000, 0.07300), (0.00500, 0.05200, 0.02200), C_STEEL, 0.0),
)

IN_STAND = part(
    "in_stand",
    pos=(IN_STAND_X0, IN_STAND_Y0, TOP),
    boxes=IN_STAND_BOXES,
    fixed=True,
    friction=0.22,
)

_CHECK_B_Z0 = (TOP + 0.06240) - CHECK_BOT_FROM_ROOT
CHECK_B = make_check("check_b", (IN_STAND_X0, IN_STAND_Y0, _CHECK_B_Z0), C_CHECK_B, C_RULE_B, with_order_ink=False, pitch=0.0)

# Winco ALXP-1310H Quarter-Size Sheet Pan (244.5 x 330.2 x 28.6 mm official outer; 235.5 x 321.2 mm estimated inner bed):
_WALL_H = OUT_TRAY_OUTER_Z - OUT_TRAY_FLOOR_T  # 0.02140 m
_WALL_CZ = OUT_TRAY_FLOOR_T + 0.5 * _WALL_H    # 0.01470 m
OUT_TRAY_BOXES = (
    ((0.0, 0.0, 0.5 * OUT_TRAY_FLOOR_T), (OUT_TRAY_OUTER_X, OUT_TRAY_OUTER_Y, OUT_TRAY_FLOOR_T), C_ALUM, 0.0),
    ((+0.5 * (OUT_TRAY_OUTER_X - OUT_TRAY_WALL_T), 0.0, _WALL_CZ), (OUT_TRAY_WALL_T, OUT_TRAY_OUTER_Y, _WALL_H), C_ALUM, 0.0),
    ((-0.5 * (OUT_TRAY_OUTER_X - OUT_TRAY_WALL_T), 0.0, _WALL_CZ), (OUT_TRAY_WALL_T, OUT_TRAY_OUTER_Y, _WALL_H), C_ALUM, 0.0),
    ((0.0, +0.5 * (OUT_TRAY_OUTER_Y - OUT_TRAY_WALL_T), _WALL_CZ), (OUT_TRAY_INNER_X, OUT_TRAY_WALL_T, _WALL_H), C_ALUM, 0.0),
    ((0.0, -0.5 * (OUT_TRAY_OUTER_Y - OUT_TRAY_WALL_T), _WALL_CZ), (OUT_TRAY_INNER_X, OUT_TRAY_WALL_T, _WALL_H), C_ALUM, 0.0),
)

OUT_TRAY = part(
    "out_tray",
    pos=(OUT_TRAY_X0, OUT_TRAY_Y0, TOP),
    boxes=OUT_TRAY_BOXES,
    fixed=True,
    friction=0.50,
)

OBJECTS = [PASS_RISER, CHECK_RACK, IN_STAND, OUT_TRAY, CHECK_A, CHECK_B, *BEAD_PARTS]


def check_world_hulls(check_spec):
    """Rest-pose authored box hulls for offline CPU clearance verification."""
    p0 = np.asarray(check_spec.pos, float)
    c_y = math.cos(check_spec.pitch)
    s_y = math.sin(check_spec.pitch)
    R0 = np.array([[c_y, 0.0, s_y], [0.0, 1.0, 0.0], [-s_y, 0.0, c_y]])
    all_b = list(check_spec.boxes) + [x for j in check_spec.joints for x in j.boxes]
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], float)
    lnorms = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    out = []
    for k, b in enumerate(all_b):
        cl = np.asarray(b[0], float)
        h = 0.5 * np.asarray(b[1], float)
        cw = p0 + R0 @ cl
        pts = (signs * h) @ R0.T + cw
        nw = lnorms @ R0.T
        ext = np.array([h[0], h[0], h[1], h[1], h[2], h[2]], float)
        dw = -np.einsum("ij,j->i", nw, cw) - ext
        out.append((f"{check_spec.name}_b{k}", pts, np.column_stack([nw, dw]), pts.min(0), pts.max(0)))
    return out


# -- Authored-Collider Static & Dynamic OBB Tables + Prepared Goal Scorer -----------------------
TRAY_BOX_NAMES = (
    "out_tray/box_0_floor",
    "out_tray/box_1_wall_pos_x",
    "out_tray/box_2_wall_neg_x",
    "out_tray/box_3_wall_pos_y",
    "out_tray/box_4_wall_neg_y",
)
IN_STAND_BOX_NAMES = (
    "in_stand/box_0_base",
    "in_stand/box_1_lip_neg_x",
    "in_stand/box_2_lip_pos_x",
)

RACK_STATIC_OBBS: tuple[WorldOBB, ...] = static_part_world_obbs(CHECK_RACK, RACK_BOX_NAMES)
RISER_STATIC_OBBS: tuple[WorldOBB, ...] = static_part_world_obbs(PASS_RISER, RISER_BOX_NAMES)
TRAY_STATIC_OBBS: tuple[WorldOBB, ...] = static_part_world_obbs(OUT_TRAY, TRAY_BOX_NAMES)
IN_STAND_STATIC_OBBS: tuple[WorldOBB, ...] = static_part_world_obbs(IN_STAND, IN_STAND_BOX_NAMES)
CHECK_A_AUTHORED_COLLIDERS: tuple[AuthoredColliderBox, ...] = extract_authored_colliders(CHECK_A)
CHECK_B_AUTHORED_COLLIDERS: tuple[AuthoredColliderBox, ...] = extract_authored_colliders(CHECK_B)

SCORER_GEOM = ScorerSceneGeometry(
    top_z=TOP,
    rack_x0=RACK_X0,
    rack_y0=RACK_Y0,
    z_rack_bot=Z_RACK_BOT,
    rack_outer_h=RACK_OUTER_H,
    rack_outer_w=RACK_OUTER_W,
    rack_outer_d=RACK_OUTER_D,
    wall_t=WALL_T,
    endcap_t=ENDCAP_T,
    z_apron_lip=Z_APRON_LIP,
    x_lip_inner=X_LIP_INNER,
    ramp_deg=RAMP_DEG,
    bead_radius=BEAD_RADIUS,
    target_b_y_range=TARGET_B_ZONE_Y,
    tray_inner_x_range=(OUT_TRAY_X0 - 0.5 * OUT_TRAY_INNER_X, OUT_TRAY_X0 + 0.5 * OUT_TRAY_INNER_X),
    tray_inner_y_range=(OUT_TRAY_Y0 - 0.5 * OUT_TRAY_INNER_Y, OUT_TRAY_Y0 + 0.5 * OUT_TRAY_INNER_Y),
    tray_floor_top_z=TOP + OUT_TRAY_FLOOR_T,
    tray_rim_top_z=TOP + OUT_TRAY_OUTER_Z,
    rack_obbs=RACK_STATIC_OBBS,
    tray_obbs=TRAY_STATIC_OBBS,
    riser_obbs=RISER_STATIC_OBBS,
    in_stand_obbs=IN_STAND_STATIC_OBBS,
    check_a_colliders=CHECK_A_AUTHORED_COLLIDERS,
    check_b_colliders=CHECK_B_AUTHORED_COLLIDERS,
)
GOAL_SCORER = GoalScorer(SCORER_GEOM)


# -- Exact 3D SAT Oriented-Box Penetration Witness (with Pair, Normal & Axis Classification) ----
def obb_penetration_detail(
    c1: np.ndarray,
    h1: np.ndarray,
    R1: np.ndarray,
    c2: np.ndarray,
    h2: np.ndarray,
    R2: np.ndarray,
) -> tuple[float, np.ndarray | None, str]:
    """Exact 15-axis 3D SAT MTV penetration depth (m), unit MTV normal (pointing 2 -> 1), and axis kind."""
    if np.linalg.norm(c2 - c1) > float(np.linalg.norm(h1) + np.linalg.norm(h2)) + 1e-6:
        return 0.0, None, "disjoint"
    d = c1 - c2  # vector from static box (2) to card box (1)
    candidates: list[tuple[np.ndarray, str]] = [
        (R1[:, 0], "card_face_x"),
        (R1[:, 1], "card_face_y"),
        (R1[:, 2], "card_face_z"),
        (R2[:, 0], "static_face_x"),
        (R2[:, 1], "static_face_y"),
        (R2[:, 2], "static_face_z"),
    ]
    for i, aname in enumerate(("x", "y", "z")):
        ui = R1[:, i]
        for j, bname in enumerate(("x", "y", "z")):
            cr = np.cross(ui, R2[:, j])
            nrm = float(np.linalg.norm(cr))
            if nrm > 1e-8:
                candidates.append((cr / nrm, f"edge_{aname}x{bname}"))
    min_ov = float("inf")
    best_n: np.ndarray | None = None
    best_kind = "disjoint"
    for ax, kind in candidates:
        r1 = float(np.sum(h1 * np.abs(R1.T @ ax)))
        r2 = float(np.sum(h2 * np.abs(R2.T @ ax)))
        proj = float(np.dot(d, ax))
        ov = r1 + r2 - abs(proj)
        if ov <= 0.0:
            return 0.0, None, "disjoint"
        if ov < min_ov:
            min_ov = ov
            best_n = ax if proj >= 0.0 else -ax
            best_kind = kind
    return float(min_ov), best_n, best_kind


def obb_penetration_m(c1: np.ndarray, h1: np.ndarray, R1: np.ndarray, c2: np.ndarray, h2: np.ndarray, R2: np.ndarray) -> float:
    return obb_penetration_detail(c1, h1, R1, c2, h2, R2)[0]


# -- Orientation & IK Helpers (Closer Mid-Card Pinch Frames + Clean Tray Release) ---------------
def make_frame(gx, gz) -> np.ndarray:
    gx = np.asarray(gx, float)
    gz = np.asarray(gz, float)
    gz = gz / np.linalg.norm(gz)
    gx = gx - np.dot(gx, gz) * gz
    gx = gx / np.linalg.norm(gx)
    gy = np.cross(gz, gx)
    return np.column_stack([gx, gy, gz])


class Orient(float):
    def __new__(cls, r):
        v = float.__new__(cls, 0.0)
        v._rot = r
        return v

    @property
    def rot(self) -> np.ndarray:
        return np.asarray(self._rot(), float) if callable(self._rot) else np.asarray(self._rot, float)


_old_grip = dm.grip_rotation
dm.grip_rotation = lambda yaw=0.0: yaw.rot if hasattr(yaw, "rot") else _old_grip(yaw)


def wp(dur: float, pos, finger: float, label: str, rot) -> Waypoint:
    return Waypoint(dur, tuple(float(v) for v in pos), float(finger), label, Orient(rot))


# Closer mid-card coplanar X-pinch:
#   Right arm: el = -4.5 deg, az = 90.0 deg -> grasps Check B 104.0 mm below top edge (0.00 deg yaw)
#   Left arm:  el = -8.0 deg, az = 88.2 deg -> grasps Check A 102.0 mm below top edge
_EL_R_RAD = math.radians(-4.5)
_EL_L_RAD = math.radians(-8.0)
_AZ_R_RAD = math.radians(90.0)
_AZ_L_RAD = math.radians(88.2)
ROT_R_PINCH = make_frame([+1.0, 0.0, 0.0], [math.cos(_EL_R_RAD) * math.cos(_AZ_R_RAD), +math.cos(_EL_R_RAD) * math.sin(_AZ_R_RAD), math.sin(_EL_R_RAD)])
ROT_L_PINCH = make_frame([+1.0, 0.0, 0.0], [math.cos(_EL_L_RAD) * math.cos(_AZ_L_RAD), -math.cos(_EL_L_RAD) * math.sin(_AZ_L_RAD), math.sin(_EL_L_RAD)])

# Near-horizontal (-82 deg, 8 deg from horizontal) tray-release frame with -28 deg downward approach from +Y:
#   - Moves wrist joint6 to ~-2 deg (dead center of [-120, +120] deg!)
#   - Keeps all 289 arm/finger hulls > 20 mm above the 25.4 mm tray rim and > 30 mm above the table
#   - Centers Check A horizontally in x=[0.171, 0.341] m (> 30 mm clear of all 4 tray walls) so it drops flat
#     without touching the -X wall or whipping as an inverted pendulum across a 240 Hz step!
_TRAY_LEAN_RAD = math.radians(-82.0)
_EL_TRAY_RAD = math.radians(-28.0)
ROT_L_TRAY = make_frame(
    [math.cos(_TRAY_LEAN_RAD), 0.0, -math.sin(_TRAY_LEAN_RAD)],
    [math.cos(_EL_TRAY_RAD) * math.cos(_AZ_L_RAD), -math.cos(_EL_TRAY_RAD) * math.sin(_AZ_L_RAD), math.sin(_EL_TRAY_RAD)],
)

Q_R_READY = np.array(READY, float)
P_R_READY, ROT_R_READY = ik.tcp_pose(Q_R_READY, np.array(RIGHT_BASE_POS, float))
Q_L_READY = np.array(READY, float)
P_L_READY, ROT_L_READY = ik.tcp_pose(Q_L_READY, np.array(LEFT_BASE_POS, float))

# Retracted clear park pose (-115 mm in -X from READY so all distal hand/finger hulls sit > 100 mm
# surface-to-surface from the entire 457.2 mm CK6518A rack, both checks, and the 25.4 mm out-tray):
P_R_PARK = (float(P_R_READY[0]) - 0.11500, float(P_R_READY[1]), float(P_R_READY[2]) + 0.02000)
ROT_R_PARK = ROT_R_READY
Q_R_PARK, _pe_rp, _re_rp = ik.solve(np.array(P_R_PARK), ik.matrix_to_quat_xyzw(ROT_R_PARK), Q_R_READY, base_pos=np.array(RIGHT_BASE_POS, float), iters=150)
assert _pe_rp < 1e-3 and _re_rp < 1e-2, f"Bad R clear park IK: {_pe_rp}, {_re_rp}"

P_L_PARK = (float(P_L_READY[0]) - 0.11500, float(P_L_READY[1]), float(P_L_READY[2]) + 0.02000)
ROT_L_PARK = ROT_L_READY
Q_L_PARK, _pe_lp, _re_lp = ik.solve(np.array(P_L_PARK), ik.matrix_to_quat_xyzw(ROT_L_PARK), Q_L_READY, base_pos=np.array(LEFT_BASE_POS, float), iters=150)
assert _pe_lp < 1e-3 and _re_lp < 1e-2, f"Bad L clear park IK: {_pe_lp}, {_re_lp}"

P_R_STAGE = (0.31500, -0.13500, 0.91500)
_Q_R_STAGE, _pe_rs, _re_rs = ik.solve(
    np.array(P_R_STAGE),
    ik.matrix_to_quat_xyzw(ROT_R_PINCH),
    np.array([+0.31, 1.89, 1.02, +0.19, -1.14, -0.64]),
    base_pos=np.array(RIGHT_BASE_POS, float),
    iters=150,
)
assert _pe_rs < 1e-3 and _re_rs < 1e-2, f"Bad R stage IK: {_pe_rs}, {_re_rs}"
_R_BRIDGES = [
    ik.tcp_pose(
        (1.0 - s) * Q_R_PARK + s * _Q_R_STAGE + (4.0 * s * (1.0 - s)) * np.array([-0.22, +0.18, 0.0, 0.0, 0.0, 0.0]),
        np.array(RIGHT_BASE_POS, float),
    )
    for s in (0.25, 0.50, 0.75)
]

P_L_STAGE = (0.34000, +0.14500, 0.91500)
_Q_L_STAGE, _pe_ls, _re_ls = ik.solve(
    np.array(P_L_STAGE),
    ik.matrix_to_quat_xyzw(ROT_L_PINCH),
    np.array([+0.15, 2.33, 1.66, -1.29, +1.28, -1.14]),
    base_pos=np.array(LEFT_BASE_POS, float),
    iters=150,
)
assert _pe_ls < 1e-3 and _re_ls < 1e-2, f"Bad L stage IK: {_pe_ls}, {_re_ls}"
_L_BRIDGES = [
    ik.tcp_pose(
        (1.0 - s) * Q_L_PARK + s * _Q_L_STAGE + (4.0 * s * (1.0 - s)) * np.array([+0.28, +0.16, 0.0, 0.0, 0.0, 0.0]),
        np.array(LEFT_BASE_POS, float),
    )
    for s in (0.25, 0.50, 0.75)
]

F_OPEN = FINGER_OPEN
F_RELEASE = -0.0280
F_PINCH = 0.0


# -- URDF Gravity Compensation & Pair-Level Telemetry -------------------------------------------
_URDF_JOINTS = _parse_urdf_joints(_YAM_URDF)
_URDF_CHILDREN: dict[str, list] = {}
for _uj in _URDF_JOINTS.values():
    _URDF_CHILDREN.setdefault(_uj.parent, []).append(_uj)


def _load_urdf_inertials() -> list[tuple[str, float, np.ndarray]]:
    out = []
    for lk in _ET.parse(_YAM_URDF).getroot().findall("link"):
        m_el = lk.find("inertial/mass")
        o_el = lk.find("inertial/origin")
        if m_el is not None and float(m_el.attrib.get("value", "0")) > 0.0:
            xyz = [float(v) for v in o_el.attrib.get("xyz", "0 0 0").split()] if o_el is not None else [0.0, 0.0, 0.0]
            out.append((lk.attrib["name"], float(m_el.attrib["value"]), np.array(xyz, float)))
    return out


_URDF_INERTIALS = _load_urdf_inertials()


def _urdf_fk_poses(q6: np.ndarray, f_pos: np.ndarray) -> dict[str, np.ndarray]:
    qmap = {f"joint{i + 1}": float(q6[i]) for i in range(6)}
    qmap["joint7"] = float(f_pos[0]) if len(f_pos) > 0 else 0.0
    qmap["joint8"] = float(f_pos[1]) if len(f_pos) > 1 else 0.0
    poses = {"base": np.eye(4)}
    stack = ["base"]
    while stack:
        lname = stack.pop()
        t_parent = poses[lname]
        for j in _URDF_CHILDREN.get(lname, ()):
            t_orig = np.eye(4)
            t_orig[:3, :3] = _urdf_rpy(j.rpy)
            t_orig[:3, 3] = j.xyz
            t_j = np.eye(4)
            val = qmap.get(j.name, 0.0)
            if j.type in ("revolute", "continuous"):
                t_j[:3, :3] = _urdf_aa(j.axis, val)
            elif j.type == "prismatic":
                t_j[:3, 3] = j.axis / np.linalg.norm(j.axis) * val
            poses[j.child] = t_parent @ t_orig @ t_j
            stack.append(j.child)
    return poses


def _urdf_grav_energy(q6: np.ndarray, f_pos: np.ndarray) -> float:
    poses = _urdf_fk_poses(q6, f_pos)
    u = 0.0
    for lname, mass, com_local in _URDF_INERTIALS:
        if lname in poses:
            t_link = poses[lname]
            u += mass * 9.81 * float(t_link[2, 3] + t_link[2, :3] @ com_local)
    return u


def _urdf_grav_torques_6(q6: np.ndarray, f_pos: np.ndarray) -> np.ndarray:
    eps = 1e-5
    u0 = _urdf_grav_energy(q6, f_pos)
    g6 = np.zeros(6)
    for j in range(6):
        qp = q6.copy()
        qp[j] += eps
        g6[j] = (_urdf_grav_energy(qp, f_pos) - u0) / eps
    return g6


_STEP_IDX = 0
_TRACE: list[dict] = []


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


def _arm_telemetry(base, arm_key: str, base_pos: np.ndarray) -> tuple[dict, np.ndarray, tuple[float, float]]:
    rob = base.scene[arm_key]
    arm_ids = rob.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
    f_ids = rob.find_joints(list(FINGER_JOINTS), preserve_order=True)[0]
    q_all = labkit._np(rob.data.joint_pos)[0]
    tau_src = getattr(rob.data, "applied_torque", getattr(rob.data, "computed_torque", None))
    tau_all = labkit._np(tau_src)[0] if tau_src is not None else np.zeros_like(q_all)
    q6 = q_all[arm_ids]
    tau6_raw = tau_all[arm_ids]
    f_pos = q_all[f_ids]
    f_tau = tau_all[f_ids]
    grav6 = _urdf_grav_torques_6(q6, f_pos)
    tau6_net = tau6_raw - grav6
    tcp_p, _ = ik.tcp_pose(q6, base_pos)
    fk = _urdf_fk_poses(q6, f_pos)
    tip_pts = [tcp_p]
    for tname in ("tip_left", "tip_right"):
        if tname in fk:
            tip_pts.append(base_pos + fk[tname][:3, 3])
    J = _jacobian_6x6(q6, base_pos)
    pinv_jt = np.linalg.pinv(J.T, rcond=1e-3)
    wrench_net = pinv_jt @ tau6_net
    wrench_raw = pinv_jt @ tau6_raw
    info = {
        "tcp": [round(float(v), 5) for v in tcp_p],
        "tip_pts": [[round(float(x), 5) for x in pt] for pt in tip_pts],
        "finger_mm": round(float(np.mean(f_pos)) * 1000.0, 3),
        "grip_force_N": round(float(np.mean(np.abs(f_tau))), 3),
        "tcp_force_xyz_N": [round(float(v), 3) for v in wrench_net[:3]],
        "tcp_force_norm_N": round(float(np.linalg.norm(wrench_net[:3])), 3),
        "tcp_force_raw_norm_N": round(float(np.linalg.norm(wrench_raw[:3])), 3),
        "arm_tau_net_norm_Nm": round(float(np.linalg.norm(tau6_net)), 3),
    }
    return info, q6.copy(), (float(f_pos[0]), float(f_pos[1]))


_ADAPTER_ERRORS: list[str] = []


def _check_telemetry(base, origin: np.ndarray, spec, colliders: tuple[AuthoredColliderBox, ...]) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    asset = base.scene[spec.name]
    asset.set_joint_position_target(asset.data.default_joint_pos)
    bposes = labkit._np(asset.data.body_link_pose_w)[0]
    bvels = labkit._np(asset.data.body_link_vel_w)[0]
    bnames = list(asset.body_names)

    body_poses_map, body_poses_serial, max_anchor_err_m = adapt_articulation_body_poses(
        bposes, bnames, origin, invalid_pose_log=_ADAPTER_ERRORS
    )

    p_root = body_poses_map["body"][0]
    p_bot_h = body_poses_map["hinge_23"][0]
    q_j = labkit._np(asset.data.joint_pos)[0]
    jnames = list(asset.joint_names)
    hinges_deg = {
        jn.replace("_joint", ""): round(math.degrees(float(q_j[jnames.index(jn)])), 3)
        for jn in jnames
    }

    # Exact world OBBs, all 40 collider vertices, and mass-weighted CoM from authored colliders:
    card_obbs, all_corners, com_w = card_world_obbs_and_com(colliders, body_poses_map)
    z_top_exact = float(np.max(all_corners[:, 2]))
    z_bot_exact = float(np.min(all_corners[:, 2]))

    # Pair-level SAT overlap against every authored rack & riser box:
    max_rack_pen_m = 0.0
    worst_rack_pair = None
    max_riser_pen_m = 0.0
    worst_riser_pair = None
    for c_obb in card_obbs:
        for r_obb in RACK_STATIC_OBBS:
            pen_m, n_w, kind = obb_penetration_detail(c_obb.center, c_obb.half_extents, c_obb.rot, r_obb.center, r_obb.half_extents, r_obb.rot)
            if pen_m > max_rack_pen_m:
                max_rack_pen_m = pen_m
                worst_rack_pair = {
                    "card_collider": c_obb.label,
                    "rack_collider": r_obb.label,
                    "pen_mm": round(pen_m * 1000.0, 4),
                    "normal_xyz": [round(float(v), 5) for v in n_w] if n_w is not None else None,
                    "axis_kind": kind,
                    "card_collider_center_xyz": [round(float(v), 6) for v in c_obb.center],
                }
        for p_obb in RISER_STATIC_OBBS:
            pen_m, n_w, kind = obb_penetration_detail(c_obb.center, c_obb.half_extents, c_obb.rot, p_obb.center, p_obb.half_extents, p_obb.rot)
            if pen_m > max_riser_pen_m:
                max_riser_pen_m = pen_m
                worst_riser_pair = {
                    "card_collider": c_obb.label,
                    "riser_collider": p_obb.label,
                    "pen_mm": round(pen_m * 1000.0, 4),
                    "normal_xyz": [round(float(v), 5) for v in n_w] if n_w is not None else None,
                    "axis_kind": kind,
                }

    # All-40-corners wall-crossing flags inside the rack's vertical/lateral span:
    in_rack_span = (
        (all_corners[:, 2] >= Z_RACK_BOT)
        & (all_corners[:, 2] <= Z_RACK_BOT + RACK_OUTER_H)
        & (np.abs(all_corners[:, 1]) <= 0.5 * RACK_OUTER_W)
    )
    any_corner_behind_wall_outer = bool(np.any(in_rack_span & (all_corners[:, 0] > RACK_X0)))
    any_corner_past_wall_midplane = bool(np.any(in_rack_span & (all_corners[:, 0] > RACK_X0 - 0.5 * WALL_T)))

    # 4 logical strip centers for backward-compatible plotting/summaries:
    R_root = quat_wxyz_to_mat(*[float(v) for v in body_poses_map["body"][1]])
    strip_centers = [p_root]  # geometric midpoint of continuous [-15.75, +15.75] mm root strip is [0, 0, 0]!
    for b_lbl in ("hinge_01/box_0", "hinge_12/box_0", "hinge_23/box_0"):
        matching = next(o for o in card_obbs if o.label == b_lbl)
        strip_centers.append(matching.center)

    max_lin_speed_mm_s = float(np.max(np.linalg.norm(bvels[:, :3], axis=1))) * 1000.0
    info = {
        "root_xyz": [round(float(v), 5) for v in p_root],
        "top_hinge_xyz": [round(float(v), 5) for v in p_root],
        "bot_hinge_xyz": [round(float(v), 5) for v in p_bot_h],
        "mass_weighted_com_xyz": [round(float(v), 5) for v in com_w],
        "strip_centers_xyz": [[round(float(x), 5) for x in row] for row in strip_centers],
        "body_poses": body_poses_serial,
        "max_chain_anchor_residual_mm": round(max_anchor_err_m * 1000.0, 5),
        "corners_40_xyz": [[round(float(x), 5) for x in row] for row in all_corners],
        "corners_min_xyz": [round(float(v), 5) for v in all_corners.min(axis=0)],
        "corners_max_xyz": [round(float(v), 5) for v in all_corners.max(axis=0)],
        "z_top_est": round(z_top_exact, 5),
        "z_bot_est": round(z_bot_exact, 5),
        "upright_cos_z": round(float(R_root[2, 2]), 6),
        "max_speed_mm_s": round(max_lin_speed_mm_s, 3),
        "hinges_deg": hinges_deg,
        "max_abs_hinge_deg": round(max(abs(v) for v in hinges_deg.values()), 3),
        "obb_rack_pen_mm": round(max_rack_pen_m * 1000.0, 4),
        "worst_rack_pair": worst_rack_pair,
        "obb_riser_pen_mm": round(max_riser_pen_m * 1000.0, 4),
        "worst_riser_pair": worst_riser_pair,
        "behind_wall_flag": any_corner_past_wall_midplane,
        "any_corner_behind_wall_outer": any_corner_behind_wall_outer,
    }
    return info, body_poses_map


def _xz_profile_rel_rack(rec_card: dict) -> dict:
    pts = np.asarray(rec_card["corners_40_xyz"], dtype=float)
    sc = rec_card["strip_centers_xyz"]
    return {
        "hinges_deg": rec_card["hinges_deg"],
        "z_top_rel_mm": round((float(rec_card["z_top_est"]) - Z_RACK_BOT) * 1000.0, 2),
        "corners_x_rel_min_max_mm": [round((float(pts[:, 0].min()) - RACK_X0) * 1000.0, 3), round((float(pts[:, 0].max()) - RACK_X0) * 1000.0, 3)],
        "corners_z_rel_min_max_mm": [round((float(pts[:, 2].min()) - Z_RACK_BOT) * 1000.0, 3), round((float(pts[:, 2].max()) - Z_RACK_BOT) * 1000.0, 3)],
        "strip_centers_xz_rel_mm": [[round((r[0] - RACK_X0) * 1000.0, 2), round((r[2] - Z_RACK_BOT) * 1000.0, 2)] for r in sc],
    }


def _build_peak_contact_audit() -> dict:
    out = {}
    if not _TRACE:
        return out
    for cname in ("check_a", "check_b"):
        idx = max(range(len(_TRACE)), key=lambda i: float(_TRACE[i][cname]["obb_rack_pen_mm"]))
        s_pk = _TRACE[idx]
        s_pre = _TRACE[max(0, idx - 2)]
        s_post = _TRACE[min(len(_TRACE) - 1, idx + 2)]
        c = s_pk[cname]
        out[cname] = {
            "peak_step": int(s_pk["step"]),
            "peak_time_s": float(s_pk["t"]),
            "obb_rack_pen_mm": float(c["obb_rack_pen_mm"]),
            "worst_rack_pair": c["worst_rack_pair"],
            "any_corner_behind_wall_outer": bool(c["any_corner_behind_wall_outer"]),
            "behind_wall_midplane_flag": bool(c["behind_wall_flag"]),
            "body_poses": c["body_poses"],
            "corners_40_xyz": c["corners_40_xyz"],
            "cross_sections_xz_rel_rack_mm": {
                "before_minus_0p10s": _xz_profile_rel_rack(s_pre[cname]),
                "at_peak": _xz_profile_rel_rack(c),
                "after_plus_0p10s": _xz_profile_rel_rack(s_post[cname]),
            },
        }
    return out


def _flush_log() -> None:
    if not _LOG_JSON or not _TRACE:
        return
    scorer_eval = GOAL_SCORER.evaluate() if len(GOAL_SCORER.samples) > 0 else None
    if _ADAPTER_ERRORS and scorer_eval is not None:
        scorer_eval = {**scorer_eval, "score": 0.0, "adapter_errors": list(_ADAPTER_ERRORS)}
    payload = {
        "config": {
            "task_version": "production_final2_alxp1310h_ncco525_526",
            "mode": _MODE,
            "contact_offset_m": _CONTACT_OFFSET,
            "physics_hz": _PHYSICS_HZ if _PHYSICS_HZ > 0 else 240.0,
            "check_thick_mm_est": round(CHECK_THICK * 1000.0, 4),
            "check_mass_g_est": round(CHECK_MASS_KG * 1000.0, 4),
            "check_modulus_gpa_est": round(CHECK_MODULUS_PA * 1e-9, 3),
            "stiffness_scale": STIFFNESS_SCALE,
            "hinge_k_Nm_per_rad": round(HINGE_K, 6),
            "friction_scale": FRICTION_SCALE,
            "mu_bead_est": round(MU_BEAD, 4),
            "mu_rack_est": round(MU_RACK, 4),
            "mu_check_est": round(MU_CHECK, 4),
            "ramp_deg_est": RAMP_DEG,
            "bead_count_est": BEAD_COUNT,
            "bead_diam_mm_est": round(2.0 * BEAD_RADIUS * 1000.0, 2),
            "grasp_d_top_b_mm": 104.0,
            "grasp_d_top_a_mm": 102.0,
            "out_tray_model": "Winco ALXP-1310H (244.5 x 330.2 x 28.6 mm official outer; 235.5 x 321.2 mm estimated inner bed)",
        },
        "peak_contact_audit": _build_peak_contact_audit(),
        "adapter_errors": list(_ADAPTER_ERRORS),
        "goal_scorer_final": scorer_eval,
        "steps": _TRACE,
    }
    p = pathlib.Path(_LOG_JSON)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@on_reset
def _reset(env):
    global _STEP_IDX, _FAST_SCORE_STEP, _FAST_SCORE_VAL
    _flush_log()
    _STEP_IDX = 0
    _FAST_SCORE_STEP = -1
    _FAST_SCORE_VAL = 0.0
    _TRACE.clear()
    _ADAPTER_ERRORS.clear()
    GOAL_SCORER.reset()


@on_step
def _step_cb(env):
    global _STEP_IDX
    _STEP_IDX += 1
    base = env.unwrapped
    origin = labkit._np(base.scene.env_origins[0])
    ca, ca_poses = _check_telemetry(base, origin, CHECK_A, CHECK_A_AUTHORED_COLLIDERS)
    cb, cb_poses = _check_telemetry(base, origin, CHECK_B, CHECK_B_AUTHORED_COLLIDERS)
    ra, rq6, rf = _arm_telemetry(base, "right_robot", np.asarray(RIGHT_BASE_POS, float))
    la, lq6, lf = _arm_telemetry(base, "left_robot", np.asarray(LEFT_BASE_POS, float))

    bead_xyz = np.array([labkit._np(base.scene[bp.name].data.root_link_pose_w)[0, :3] - origin for bp in BEAD_PARTS])
    bead_vel = np.array([labkit._np(base.scene[bp.name].data.root_link_vel_w)[0, :3] for bp in BEAD_PARTS])
    max_bead_speed_mm_s = float(np.max(np.linalg.norm(bead_vel, axis=1))) * 1000.0
    rel_x = bead_xyz[:, 0] - RACK_X0
    rel_z = bead_xyz[:, 2] - Z_RACK_BOT
    pen_wall = np.maximum(0.0, (rel_x + BEAD_RADIUS) - (-WALL_T))
    dist_ramp = (rel_x - X_LIP_INNER) * _CA + (rel_z - Z_APRON_LIP) * _SA
    pen_ramp = np.maximum(0.0, BEAD_RADIUS - dist_ramp)
    max_bead_pen_mm = float(max(pen_wall.max(), pen_ramp.max())) * 1000.0

    near_a = np.abs(bead_xyz[:, 1] - ca["top_hinge_xyz"][1]) <= 0.050
    near_b = np.abs(bead_xyz[:, 1] - cb["top_hinge_xyz"][1]) <= 0.050
    z_beads_a = float(np.mean(rel_z[near_a]) * 1000.0) if np.any(near_a) else float("nan")
    z_beads_b = float(np.mean(rel_z[near_b]) * 1000.0) if np.any(near_b) else float("nan")

    if not (
        np.all(np.isfinite(rq6))
        and np.all(np.isfinite(rf))
        and np.all(np.isfinite(lq6))
        and np.all(np.isfinite(lf))
        and np.all(np.isfinite(bead_xyz))
    ):
        _ADAPTER_ERRORS.append(f"non_finite_arm_or_bead_state@step={_STEP_IDX}")
    t_now = round(_STEP_IDX / 20.0, 3)
    GOAL_SCORER.record(
        WorldSample(
            t=t_now,
            check_a_bodies=ca_poses,
            check_b_bodies=cb_poses,
            bead_centers=bead_xyz.copy(),
            right_q6=rq6,
            right_fingers=rf,
            left_q6=lq6,
            left_fingers=lf,
        )
    )

    all_hand_pts = np.array(ra["tip_pts"] + la["tip_pts"])
    cb_strips = np.array(cb["strip_centers_xyz"])
    ca_strips = np.array(ca["strip_centers_xyz"])
    d_any_hand_cb_mm = float(np.min(np.linalg.norm(cb_strips[:, None, :] - all_hand_pts[None, :, :], axis=2))) * 1000.0
    d_any_hand_ca_mm = float(np.min(np.linalg.norm(ca_strips[:, None, :] - all_hand_pts[None, :, :], axis=2))) * 1000.0

    rec = {
        "step": _STEP_IDX,
        "t": t_now,
        "check_a": ca,
        "check_b": cb,
        "right_arm": {k: v for k, v in ra.items() if k != "tip_pts"},
        "left_arm": {k: v for k, v in la.items() if k != "tip_pts"},
        "beads": {
            "mean_z_rel_near_a_mm": round(z_beads_a, 3) if math.isfinite(z_beads_a) else None,
            "mean_z_rel_near_b_mm": round(z_beads_b, 3) if math.isfinite(z_beads_b) else None,
            "min_z_rel_all_mm": round(float(rel_z.min()) * 1000.0, 3),
            "max_z_rel_all_mm": round(float(rel_z.max()) * 1000.0, 3),
            "max_speed_mm_s": round(max_bead_speed_mm_s, 3),
            "max_pen_mm": round(max_bead_pen_mm, 4),
        },
        "hand_clear_diag_mm": {"min_tip_to_b_strip_center": round(d_any_hand_cb_mm, 1), "min_tip_to_a_strip_center": round(d_any_hand_ca_mm, 1)},
    }
    if _LOG_JSON:
        _TRACE.append(rec)
        if _STEP_IDX == RUN_STEPS:
            _flush_log()


_FAST_SCORE_STEP: int = -1
_FAST_SCORE_VAL: float = 0.0


def _cached_fast_score() -> float:
    global _FAST_SCORE_STEP, _FAST_SCORE_VAL
    if _FAST_SCORE_STEP != _STEP_IDX or _STEP_IDX <= 0:
        _FAST_SCORE_VAL = float(GOAL_SCORER.evaluate(skip_clearance_if_zero=True)["score"])
        _FAST_SCORE_STEP = _STEP_IDX
    return _FAST_SCORE_VAL


# -- Production Goal-Only Scorer Hooks (enabled by default; 1,200-step run limit) ---------------
def score(objects) -> float:
    if _ADAPTER_ERRORS:
        return 0.0
    if _ENABLE_SCORER:
        return _cached_fast_score()
    return 0.0


def task_score(env) -> float:
    _flush_log()
    if _ADAPTER_ERRORS:
        return 0.0
    if _ENABLE_SCORER:
        return float(GOAL_SCORER.evaluate()["score"])
    return 0.0


def success(objects) -> bool:
    if _ADAPTER_ERRORS:
        return False
    if _ENABLE_SCORER:
        return bool(_cached_fast_score() >= 1.0)
    return False


# -- Closer Mid-Card Coupon Trajectories (unchanged stock rack + clean tray release/retract) ----
def demo() -> dict[str, list[Waypoint]]:
    # Check B closer mid-card grasp: 104.0 mm below top edge (67.5 mm above bottom edge, strip_2):
    d_top_b = 0.10400
    dz_grasp_b = CHECK_TOP_FROM_ROOT - d_top_b  # -0.08825 m
    y_off_b = -0.01200
    p_b_pick = (IN_STAND_X0, IN_STAND_Y0 + y_off_b, _CHECK_B_Z0 + dz_grasp_b)
    p_b_pre = (IN_STAND_X0, IN_STAND_Y0 - 0.07500, _CHECK_B_Z0 + dz_grasp_b + 0.00800)
    p_b_lift_up = (IN_STAND_X0, IN_STAND_Y0 + y_off_b, _CHECK_B_Z0 + dz_grasp_b + 0.02500)

    x_win_low = X_WALL_WORLD - 0.00650
    x_win_high = X_WALL_WORLD - 0.00550
    x_apron_reg = X_WALL_WORLD - 0.00240
    x_throat_up = X_WALL_WORLD - 0.00280
    x_release_b = X_WALL_WORLD - 0.00360
    z_tcp_win_low = (Z_RACK_BOT - 0.00600) - CHECK_TOP_FROM_ROOT + dz_grasp_b
    z_tcp_win_high = (Z_RACK_BOT + 0.00700) - CHECK_TOP_FROM_ROOT + dz_grasp_b
    z_tcp_apron_reg = (Z_RACK_BOT + 0.00800) - CHECK_TOP_FROM_ROOT + dz_grasp_b
    z_tcp_seated = (Z_RACK_BOT + 0.04350) - CHECK_TOP_FROM_ROOT + dz_grasp_b

    # Check A closer mid-card grasp: 102.0 mm below top edge (69.5 mm above bottom edge, strip_2):
    d_top_a = 0.10200
    dz_grasp_a = CHECK_TOP_FROM_ROOT - d_top_a  # -0.08625 m
    y_off_a = +0.00800
    z_tcp_a_grasp = _CHECK_A_Z0 + dz_grasp_a
    p_a_pre = (_CHECK_A_X0 - 0.03500, CHECK_A_Y0 + 0.04000, z_tcp_a_grasp + 0.00400)
    p_a_on = (_CHECK_A_X0, CHECK_A_Y0 + y_off_a, z_tcp_a_grasp)

    r_enter = [*(wp(0.45, p, F_OPEN, f"br{k}-r", R) for k, (p, R) in enumerate(_R_BRIDGES, 1)), wp(0.45, P_R_STAGE, F_OPEN, "stage-r", ROT_R_PINCH)]
    r_exit = [wp(0.45, P_R_STAGE, F_OPEN, "unstage-r", ROT_R_PINCH), *(wp(0.45, p, F_OPEN, f"unbr{k}-r", R) for k, (p, R) in enumerate(reversed(_R_BRIDGES), 1)), wp(0.60, P_R_PARK, F_OPEN, "park-r", ROT_R_PARK)]

    l_enter = [*(wp(0.45, p, F_OPEN, f"br{k}-l", R) for k, (p, R) in enumerate(_L_BRIDGES, 1)), wp(0.45, P_L_STAGE, F_OPEN, "stage-l", ROT_L_PINCH)]
    l_exit = [wp(0.45, P_L_STAGE, F_OPEN, "unstage-l", ROT_L_PINCH), *(wp(0.45, p, F_OPEN, f"unbr{k}-l", R) for k, (p, R) in enumerate(reversed(_L_BRIDGES), 1)), wp(0.60, P_L_PARK, F_OPEN, "park-l", ROT_L_PARK)]

    if _MODE == "parked_retention":
        return {
            "right": [wp(10.0, P_R_PARK, F_OPEN, "park-r", ROT_R_PARK)],
            "left": [wp(10.0, P_L_PARK, F_OPEN, "park-l", ROT_L_PARK)],
        }

    if _MODE == "straight_down_control":
        return {
            "right": [wp(12.0, P_R_PARK, F_OPEN, "park-r", ROT_R_PARK)],
            "left": [
                wp(0.5, P_L_PARK, F_OPEN, "hold-init-a", ROT_L_PARK),
                *l_enter,
                wp(0.7, p_a_pre, F_OPEN, "pre-a", ROT_L_PINCH),
                wp(0.7, p_a_on, F_OPEN, "on-a", ROT_L_PINCH),
                wp(1.1, p_a_on, F_PINCH, "pinch-a", ROT_L_PINCH),
                wp(2.1, (_CHECK_A_X0 - 0.00100, CHECK_A_Y0 + y_off_a, z_tcp_a_grasp - 0.05600), F_PINCH, "pull-straight-down", ROT_L_PINCH),
                wp(0.7, (_CHECK_A_X0 - 0.00100, CHECK_A_Y0 + y_off_a, z_tcp_a_grasp - 0.05600), F_OPEN, "release-a", ROT_L_PINCH),
                wp(0.8, (_CHECK_A_X0 - 0.07000, CHECK_A_Y0 + 0.03500, z_tcp_a_grasp - 0.01000), F_OPEN, "retract-a", ROT_L_PINCH),
                *l_exit,
                wp(0.7, P_L_PARK, F_OPEN, "hold-l", ROT_L_PARK),
            ],
        }

    if _MODE == "forward_peel_control":
        return {
            "right": [wp(12.0, P_R_PARK, F_OPEN, "park-r", ROT_R_PARK)],
            "left": [
                wp(0.5, P_L_PARK, F_OPEN, "hold-init-a", ROT_L_PARK),
                *l_enter,
                wp(0.7, p_a_pre, F_OPEN, "pre-a", ROT_L_PINCH),
                wp(0.7, p_a_on, F_OPEN, "on-a", ROT_L_PINCH),
                wp(1.1, p_a_on, F_PINCH, "pinch-a", ROT_L_PINCH),
                wp(1.0, (_CHECK_A_X0 - 0.02600, CHECK_A_Y0 + y_off_a, z_tcp_a_grasp - 0.01400), F_PINCH, "peel-forward", ROT_L_PINCH),
                wp(1.3, (_CHECK_A_X0 - 0.03600, CHECK_A_Y0 + y_off_a, z_tcp_a_grasp - 0.05600), F_PINCH, "peel-down-out", ROT_L_PINCH),
                wp(0.7, (_CHECK_A_X0 - 0.03600, CHECK_A_Y0 + y_off_a, z_tcp_a_grasp - 0.05600), F_OPEN, "release-a", ROT_L_PINCH),
                wp(0.8, (_CHECK_A_X0 - 0.07000, CHECK_A_Y0 + 0.03500, z_tcp_a_grasp - 0.01000), F_OPEN, "retract-a", ROT_L_PINCH),
                *l_exit,
                wp(0.6, P_L_PARK, F_OPEN, "hold-l", ROT_L_PARK),
            ],
        }

    if _MODE == "diagonal_slide_control":
        return {
            "right": [wp(12.0, P_R_PARK, F_OPEN, "park-r", ROT_R_PARK)],
            "left": [
                wp(0.5, P_L_PARK, F_OPEN, "hold-init-a", ROT_L_PARK),
                *l_enter,
                wp(0.7, p_a_pre, F_OPEN, "pre-a", ROT_L_PINCH),
                wp(0.7, p_a_on, F_OPEN, "on-a", ROT_L_PINCH),
                wp(1.1, p_a_on, F_PINCH, "pinch-a", ROT_L_PINCH),
                wp(0.65, (_CHECK_A_X0 - 0.00350, CHECK_A_Y0 + y_off_a + 0.01500, z_tcp_a_grasp + 0.00220), F_PINCH, "unwedge-lift", ROT_L_PINCH),
                wp(1.55, (_CHECK_A_X0 - 0.01400, CHECK_A_Y0 + y_off_a + 0.02800, z_tcp_a_grasp - 0.05600), F_PINCH, "diag-slide-out", ROT_L_PINCH),
                wp(0.7, (_CHECK_A_X0 - 0.01400, CHECK_A_Y0 + y_off_a + 0.02800, z_tcp_a_grasp - 0.05600), F_OPEN, "release-a", ROT_L_PINCH),
                wp(0.8, (_CHECK_A_X0 - 0.07000, CHECK_A_Y0 + 0.03500, z_tcp_a_grasp - 0.01000), F_OPEN, "retract-a", ROT_L_PINCH),
                *l_exit,
                wp(0.6, P_L_PARK, F_OPEN, "hold-l", ROT_L_PARK),
            ],
        }

    right_wps = [
        wp(1.20, P_R_PARK, F_OPEN, "init-hold-r", ROT_R_PARK),                                                  # 1.20 s
        *r_enter,                                                                                               # 3.00 s
        wp(0.60, p_b_pre, F_OPEN, "pre-pick-b", ROT_R_PINCH),                                                   # 3.60 s
        wp(0.60, p_b_pick, F_OPEN, "on-pick-b", ROT_R_PINCH),                                                   # 4.20 s
        wp(1.15, p_b_pick, F_PINCH, "pinch-b", ROT_R_PINCH),                                                    # 5.35 s
        wp(0.55, p_b_lift_up, F_PINCH, "lift-up-stand-b", ROT_R_PINCH),                                         # 5.90 s
        wp(0.40, (0.29500, CHECK_B_INSERT_Y + y_off_b, z_tcp_win_low - 0.02000), F_PINCH, "transit-low-b", ROT_R_PINCH),# 6.30 s
        wp(0.40, (x_win_low, CHECK_B_INSERT_Y + y_off_b, z_tcp_win_low), F_PINCH, "front-window-low-b", ROT_R_PINCH),    # 6.70 s
        wp(0.50, (x_win_high, CHECK_B_INSERT_Y + y_off_b, z_tcp_win_high), F_PINCH, "front-window-high-b", ROT_R_PINCH), # 7.20 s
        wp(0.50, (x_apron_reg, CHECK_B_INSERT_Y + y_off_b, z_tcp_apron_reg), F_PINCH, "kiss-apron-b", ROT_R_PINCH),      # 7.70 s
        wp(1.50, (x_throat_up, CHECK_B_INSERT_Y + y_off_b, z_tcp_seated), F_PINCH, "insert-up-b", ROT_R_PINCH),          # 9.20 s
        wp(0.60, (x_release_b, CHECK_B_INSERT_Y + y_off_b, z_tcp_seated), F_RELEASE, "release-ins-b", ROT_R_PINCH),      # 9.80 s
        wp(0.65, (x_release_b - 0.068, CHECK_B_INSERT_Y - 0.02500, z_tcp_seated - 0.04200), F_OPEN, "withdraw-ins-b", ROT_R_PINCH),# 10.45 s
        *r_exit,                                                                                                # 12.85 s
        wp(15.15, P_R_PARK, F_OPEN, "hold-park-r", ROT_R_PARK),                                                 # 28.00 s
    ]

    # Clean tray placement: lean Check A 42 deg toward -X centered over the 232.3 x 321.2 mm inner bed
    # of Winco ALXP-1310, release gently at TOP + 0.092 m, and lift STRAIGHT UP (+Z) to TOP + 0.215 m
    # BEFORE moving to P_L_STAGE so the open jaws never sweep +X through a standing card!
    left_wps = [
        wp(14.35, P_L_PARK, F_OPEN, "wait-l", ROT_L_PARK),                                                      # 14.35 s
        *l_enter,                                                                                               # 16.15 s
        wp(0.60, p_a_pre, F_OPEN, "pre-a", ROT_L_PINCH),                                                        # 16.75 s
        wp(0.60, p_a_on, F_OPEN, "on-a", ROT_L_PINCH),                                                          # 17.35 s
        wp(1.10, p_a_on, F_PINCH, "pinch-a", ROT_L_PINCH),                                                      # 18.45 s
        wp(1.25, (_CHECK_A_X0 - 0.00180, CHECK_A_Y0 + y_off_a + 0.01000, z_tcp_a_grasp - 0.03450), F_PINCH, "throat-slide-down-a", ROT_L_PINCH),  # 19.70 s (top drops to +6.5 mm < +12.5 mm ramp lip inside 4.0 mm throat!)
        wp(1.05, (_CHECK_A_X0 - 0.03200, CHECK_A_Y0 + y_off_a + 0.01000, z_tcp_a_grasp - 0.05800), F_PINCH, "clear-apron-out-a", ROT_L_PINCH),    # 20.75 s (withdraws -X cleanly below +12.5 mm front-ramp lip!)
        wp(0.75, (0.27200, +0.17500, TOP + 0.10500), F_PINCH, "carry-tray-a", ROT_L_TRAY),                      # 21.50 s (near-horizontal -82 deg centered over tray!)
        wp(0.35, (0.27200, +0.17500, TOP + 0.07400), F_PINCH, "lay-tray-a", ROT_L_TRAY),                        # 21.85 s (lower flat 70 mm above bed, >20 mm above rim!)
        wp(0.35, (0.27200, +0.17500, TOP + 0.07400), F_OPEN, "drop-tray-a", ROT_L_TRAY),                        # 22.20 s (parallel flat release, zero wall whip!)
        wp(0.50, (0.25000, +0.17500, TOP + 0.19500), F_OPEN, "lift-clear-tray-a", ROT_L_PINCH),                 # 22.70 s (lift +Z to +195 mm clear of card!)
        *l_exit,                                                                                                # 25.10 s
        wp(2.90, P_L_PARK, F_OPEN, "hold-park-l", ROT_L_PARK),                                                  # 28.00 s (2.90 s = 58 samples settled!)
    ]
    return {"right": right_wps, "left": left_wps}
