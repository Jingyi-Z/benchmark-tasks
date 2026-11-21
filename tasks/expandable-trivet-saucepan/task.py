"""Expand a coupled scissor pot stand (Joseph Joseph Stretch 70033) to support two stock flat-base pans clear of the worktop.

Scene: Dream's calibrated ``station`` tabletop (`TABLE_TOP = 0.7475 m`).
Objects (1:1 stock commercial hardware, cold & empty, zero custom fixtures or invented latches):
- ``trivet``: Joseph Joseph Stretch(TM) Expandable Silicone-Nylon Pot Stand (SKU 70033, USD 23.00;
  compact 252.0 x 76.0 x 20.0 mm, 0.173 kg; 3-cell / 6-bar coupled scissor pantograph with exact
  PhysX 5 mimic gearing and pure rivet friction, expanding up to 543 mm).
- ``pan_left``, ``pan_right``: two Garcima 26 cm (10.5-in) Enamelled Steel Flat-Bottom Serving / Paella Pans
  (Model 20226, USD 16.95 each; 267.0 mm rim OD, 215.0 mm flat contact base, 38.0 mm wall height,
  0.520 kg, bilateral steel loop side handles spanning 322.0 mm, exact bilateral CoM offset 0.00 mm).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from collections import deque

import numpy as np
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hand_cover_data import (  # noqa: E402
    GRIPPER_D405_COVER_LOCAL,
    HAND_COVER_RADIUS_M,
    N_PTS_PER_HAND,
    TIP_LEFT_COVER_NATIVE,
    TIP_RIGHT_COVER_NATIVE,
)
from labkit import MATTE_COLORS, METALLIC_COLORS, Joint, PartState, on_reset, on_step, part  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
MAX_STEPS = 1200  # <= 1,800 target and <= 2,400 cap (demo finishes in ~36.2 s = 362 steps @ 10 Hz / 725 @ 20 Hz)

# -- Exact 1:1 stock dimensions ---------------------------------------------------------------------------------------

# 1. Joseph Joseph Stretch(TM) Expandable Pot Stand (SKU 70033):
#    Maker compact envelope: 252.0 x 76.0 x 20.0 mm, mass 0.173 kg, silicone-nylon composite.
#    3-cell (6-bar) scissor pantograph: pivot half-span a = 121.51 mm (2a = 243.02 mm), slim composite rib width
#    10.0 mm (matches real 173 g mass at 1.15 g/cm^3; end radius r = 5.0 mm), total height H = 20.0 mm
#    (half-lap 10.0 mm crossing beams + 10.0 mm support/foot pads).
TRIVET_A = 0.12151
TRIVET_BAR_W = 0.0100
TRIVET_R_END = 0.5 * TRIVET_BAR_W
TRIVET_H = 0.0200
TRIVET_HALF_H = 0.5 * TRIVET_H
TRIVET_THETA0 = math.radians(5.1950)  # compact: W = 2*a*cos(th0)+10 = 252.02 mm, L = 6*a*sin(th0)+10 = 76.01 mm
TRIVET_Q_MAX = math.radians(86.0)     # max open: theta = 48.20 deg -> L = 553.2 mm, W = 172.0 mm
TRIVET_Q_DEMO = math.radians(66.0)    # demo open: theta = 38.20 deg -> L = 460.9 mm (>>380 mm), W = 201.0 mm
TRIVET_MASS = 0.173
TRIVET_X = 0.145
TRIVET_Y = 0.000

# 2. Two Garcima 26 cm (10.5-in) Enamelled Steel Flat-Bottom Serving / Paella Pans (Model 20226):
#    Rim OD = 267.0 mm (R_rim = 133.5 mm), flat contact base OD = 215.0 mm (R_base = 107.5 mm),
#    wall height = 38.0 mm, bilateral steel loop side handles at local +-Y (grip center r = 152.0 mm, z = 43.0 mm),
#    overall span across handles = 322.0 mm, empty cold mass = 0.520 kg each.
PAN_R_RIM = 0.1335
PAN_R_BASE = 0.1075
PAN_WALL_H = 0.0380
PAN_BASE_T = 0.0040
PAN_HANDLE_R = 0.1520
PAN_HANDLE_Z = 0.0430
PAN_HANDLE_BAR_LEN = 0.0580
PAN_HANDLE_BAR_THICK = 0.0180
PAN_HANDLE_BAR_H = 0.0140
PAN_MASS = 0.520

PAN_START_X = 0.402
PAN_START_Y = 0.156
PAN_START_YAW = math.radians(24.0)   # equalizes outer-handle (459 mm) and inner-handle (466 mm) reach (<0.15 mm IK)
PAN_GOAL_Y = 0.148                   # placed at (TRIVET_X, +-0.148) -> 296.0 mm center spacing (29.0 mm rim gap, 83 mm inside 462.6 mm ends)
PAN_GOAL_YAW = math.radians(-30.0)   # parallel -30 deg placement yaw separates inner handles by 155.5 mm (+23.8 mm L-hand-to-pan1 clearance!)

# Colours & materials
SILICONE_BLACK = (0.14, 0.14, 0.16)
SILICONE_PAD = (0.20, 0.21, 0.24)
ENAMEL_NAVY = (0.16, 0.22, 0.34)
ENAMEL_BURGUNDY = (0.42, 0.16, 0.18)
STEEL_RIM = (0.76, 0.78, 0.80)
MATTE_COLORS.update({SILICONE_BLACK, SILICONE_PAD})
METALLIC_COLORS.update({STEEL_RIM, ENAMEL_NAVY, ENAMEL_BURGUNDY})

# -- Scoring thresholds (ruler / side-photo / feeler-gauge measurable, zero sub-mm rules) -----------------------------

EXPANDED_LEN_MIN_M = 0.380       # >= 38.0 cm outer trivet length (compact is 7.6 cm; 2-pan eq min is 26.7 cm)
TRIVET_TILT_MAX_DEG = 5.0        # trivet flat on worktop
TRIVET_BASE_GAP_MAX_M = 0.006    # trivet bottom within 6 mm of tabletop
PAN_TABLE_CLEAR_MIN_M = 0.010    # every point of pan base/rim/handles >= 10.0 mm clear above tabletop
PAN_SEAT_Z_MIN_M = 0.014         # pan bottom resting on 20 mm trivet [14, 26] mm (rejects pan-on-pan stacking!)
PAN_SEAT_Z_MAX_M = 0.026
PAN_TILT_MAX_DEG = 5.0           # pan level within 5 deg
PAN_SUPPORT_X_MAX_M = 0.060      # pan center within +-60 mm of trivet longitudinal centerline
PAN_SUPPORT_END_MARGIN_M = 0.030 # pan center at least 30 mm inside trivet +-L/2 ends
PAN_NONOVERLAP_MIN_M = 0.265     # horizontal center distance >= 265 mm (267 mm rims cannot overlap/stack)
HAND_CLEAR_MIN_M = 0.080         # certified 212-pt full-hand surface clearance >= 80.0 mm across ENTIRE 1.0 s window
STILL_SPREAD_MAX_M = 0.005       # < 5.0 mm visible-mark spread across ENTIRE 1.0 s window
HOLD_WINDOW_S = 1.0

# Robot finger validation invariants (matches calibrated yam_arena URDF & native PhysX)
_J_FINGER_ROT0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
_J7_ORIGIN_XYZ = np.array([0.0445119, -0.0239931, 0.0566829], dtype=np.float64)
_J7_AXIS_G = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_J8_ORIGIN_XYZ = np.array([-0.0445119, 0.0239931, 0.0566829], dtype=np.float64)
_J8_AXIS_G = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
_FINGER_ROT_FROB_TOL = 0.05
_FINGER_RAIL_PERP_TOL = 0.003
_FINGER_SLIDE_MIN = -0.0495
_FINGER_SLIDE_MAX = 0.0025


# -- Scene geometry builders ------------------------------------------------------------------------------------------


def _bar_boxes_a(center_xy: tuple[float, float], yaw: float = 0.0, free_end_sign: float = 0.0) -> list[tuple]:
    """Stepped half-lap A-layer bar (continuous lower beam [-10, 0] mm + upper support pads [0, +10] mm).
    If `free_end_sign` is -1 (A_+1) or +1 (A_-1), that outer half never crosses a B-bar pivot and carries
    full-height [0, +10] mm top support all the way to the outer corner tip.
    """
    cx, cy = center_xy
    ca, sa = math.cos(yaw), math.sin(yaw)
    full_len = 2.0 * TRIVET_A + 2.0 * TRIVET_R_END
    out = [
        ((cx, cy, -0.005), (full_len, TRIVET_BAR_W, 0.010), SILICONE_BLACK, yaw),
    ]
    for s_sgn in (-1.0, +1.0):
        if free_end_sign != 0.0 and s_sgn == free_end_sign:
            s_off = s_sgn * (0.62 * TRIVET_A + 0.5 * TRIVET_R_END)
            p_len = 0.76 * TRIVET_A + TRIVET_R_END
        else:
            # Stepped half-lap relief toward the non-free crossing end so z in [10, 20] mm stays open around B_+1 / B_-1
            s_off = s_sgn * (0.36 * TRIVET_A)
            p_len = 0.36 * TRIVET_A
        px = cx + s_off * ca
        py = cy + s_off * sa
        out.append(((px, py, +0.005), (p_len, TRIVET_BAR_W, 0.010), SILICONE_PAD, yaw))
    return out


def _bar_boxes_b(center_xy: tuple[float, float], yaw: float, free_end_sign: float = 0.0) -> list[tuple]:
    """Stepped half-lap B-layer bar (continuous upper beam [0, +10] mm + lower foot pads [-10, 0] mm).
    If `free_end_sign` is +1 (B_+1) or -1 (B_-1), that outer half never crosses an A-bar pivot and carries
    full-height [-10, 0] mm bottom foot & grip thickness all the way to the outer corner tip.
    """
    cx, cy = center_xy
    ca, sa = math.cos(yaw), math.sin(yaw)
    full_len = 2.0 * TRIVET_A + 2.0 * TRIVET_R_END
    out = [
        ((cx, cy, +0.005), (full_len, TRIVET_BAR_W, 0.010), SILICONE_BLACK, yaw),
    ]
    for s_sgn in (-1.0, +1.0):
        if free_end_sign != 0.0 and s_sgn == free_end_sign:
            s_off = s_sgn * (0.62 * TRIVET_A + 0.5 * TRIVET_R_END)
            p_len = 0.76 * TRIVET_A + TRIVET_R_END
        else:
            s_off = s_sgn * (0.36 * TRIVET_A)
            p_len = 0.36 * TRIVET_A
        px = cx + s_off * ca
        py = cy + s_off * sa
        out.append(((px, py, -0.005), (p_len, TRIVET_BAR_W, 0.010), SILICONE_PAD, yaw))
    return out


def _make_trivet():
    """6-bar coupled scissor pantograph (A_0 root + j_b0 leader + 4 PhysX 5 mimic followers).

    IMPORTANT PhysX 5 convention (verified against merged tasks/keyless-chuck-bit-release/task.py):
    PxArticulationMimicJoint enforces ``q_follower + gearing * q_leader + offset = 0``, i.e.,
    ``q_follower = -gearing * q_leader - offset``!
    - Same-sign followers (`j_b_p1 = +q_b0`, `j_b_m1 = +q_b0`) therefore require ``gearing = -1.0``.
    - Opposite-sign followers (`j_a_p1 = -q_b0`, `j_a_m1 = -q_b0`) therefore require ``gearing = +1.0``.
    Follower limits carry a +-0.10 rad buffer outside [0, TRIVET_Q_MAX] so only `j_b0` engages the physical
    hard stop at q = 0 and q = TRIVET_Q_MAX (preventing redundant unilateral stop-limit over-constraint at q = 0).
    """
    yaw_b_rest = 2.0 * TRIVET_THETA0
    c2, s2 = math.cos(yaw_b_rest), math.sin(yaw_b_rest)
    bar_mass = TRIVET_MASS / 6.0
    lim_buf = 0.10

    c_p1 = (-TRIVET_A + TRIVET_A * c2, +TRIVET_A * s2)
    c_m1 = (+TRIVET_A - TRIVET_A * c2, -TRIVET_A * s2)

    j_b0 = Joint(
        "j_b0",
        kind="hinge",
        axis="Z",
        anchor=(0.0, 0.0, 0.0),
        lower=0.0,
        upper=TRIVET_Q_MAX,
        initial=0.0,
        target=0.0,
        stiffness=0.0,
        damping=0.02,
        friction=0.035,  # ~0.6 N pull force (< 1.6 N table static friction) while holding any set extension
        mass=bar_mass,
        boxes=_bar_boxes_b((0.0, 0.0), yaw_b_rest),
    )
    j_b_p1 = Joint(
        "j_b_p1",
        kind="hinge",
        axis="Z",
        anchor=(-TRIVET_A, 0.0, 0.0),
        lower=-lim_buf,
        upper=TRIVET_Q_MAX + lim_buf,
        initial=0.0,
        target=0.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        mass=bar_mass,
        boxes=_bar_boxes_b(c_p1, yaw_b_rest, free_end_sign=+1.0),
    )
    j_a_p1 = Joint(
        "j_a_p1",
        kind="hinge",
        axis="Z",
        parent="j_b_p1",
        anchor=(c_p1[0], c_p1[1], 0.0),
        lower=-TRIVET_Q_MAX - lim_buf,
        upper=+lim_buf,
        initial=0.0,
        target=0.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        mass=bar_mass,
        boxes=_bar_boxes_a(c_p1, 0.0, free_end_sign=-1.0),
    )
    j_b_m1 = Joint(
        "j_b_m1",
        kind="hinge",
        axis="Z",
        anchor=(+TRIVET_A, 0.0, 0.0),
        lower=-lim_buf,
        upper=TRIVET_Q_MAX + lim_buf,
        initial=0.0,
        target=0.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        mass=bar_mass,
        boxes=_bar_boxes_b(c_m1, yaw_b_rest, free_end_sign=-1.0),
    )
    j_a_m1 = Joint(
        "j_a_m1",
        kind="hinge",
        axis="Z",
        parent="j_b_m1",
        anchor=(c_m1[0], c_m1[1], 0.0),
        lower=-TRIVET_Q_MAX - lim_buf,
        upper=+lim_buf,
        initial=0.0,
        target=0.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        mass=bar_mass,
        boxes=_bar_boxes_a(c_m1, 0.0, free_end_sign=+1.0),
    )

    return part(
        "trivet",
        pos=(TRIVET_X, TRIVET_Y, TOP + TRIVET_HALF_H),
        yaw=-TRIVET_THETA0,
        mass=bar_mass,
        fixed=False,
        friction=0.95,
        self_collide=False,
        color=SILICONE_BLACK,
        boxes=_bar_boxes_a((0.0, 0.0), 0.0),
        joints=[j_b0, j_b_p1, j_a_p1, j_b_m1, j_a_m1],
        mimic=[
            ("j_b_p1", "j_b0", -1.0, 0.0),  # PhysX: q_bp1 + (-1.0)*q_b0 = 0 => q_bp1 = +q_b0
            ("j_a_p1", "j_b0", +1.0, 0.0),  # PhysX: q_ap1 + (+1.0)*q_b0 = 0 => q_ap1 = -q_b0
            ("j_b_m1", "j_b0", -1.0, 0.0),  # PhysX: q_bm1 + (-1.0)*q_b0 = 0 => q_bm1 = +q_b0
            ("j_a_m1", "j_b0", +1.0, 0.0),  # PhysX: q_am1 + (+1.0)*q_b0 = 0 => q_am1 = -q_b0
        ],
    )


def _pan_spec_primitives(body_color: tuple[float, float, float]) -> tuple[list[tuple], list[tuple]]:
    """Return local (boxes, cylinders) for one Garcima 26 cm flat-bottom serving/paella pan."""
    n_seg = 16
    r_mid = 0.5 * (PAN_R_BASE + PAN_R_RIM)
    wall_thick = 0.0060
    r_wall_center = PAN_R_RIM - 0.5 * wall_thick
    seg_len = 2.0 * r_wall_center * math.tan(math.pi / n_seg) + 0.002

    boxes = []
    for k in range(n_seg):
        ang = (2.0 * math.pi * k) / n_seg
        wx = r_wall_center * math.cos(ang)
        wy = r_wall_center * math.sin(ang)
        col = STEEL_RIM if k % 4 == 0 else body_color
        boxes.append(((wx, wy, 0.5 * PAN_WALL_H), (wall_thick, seg_len, PAN_WALL_H), col, ang))

    for sgn in (+1.0, -1.0):
        boxes.append((
            (0.0, sgn * PAN_HANDLE_R, PAN_HANDLE_Z),
            (PAN_HANDLE_BAR_LEN, PAN_HANDLE_BAR_THICK, PAN_HANDLE_BAR_H),
            STEEL_RIM,
            0.0,
        ))
        for sx in (-0.025, +0.025):
            boxes.append((
                (sx, sgn * 0.1415, 0.0375),
                (0.008, 0.026, 0.013),
                STEEL_RIM,
                0.0,
            ))

    cylinders = [
        ((0.0, 0.0, 0.5 * PAN_BASE_T), PAN_R_BASE, PAN_BASE_T, body_color, "Z"),
        ((0.0, 0.0, PAN_BASE_T + 0.002), r_mid, 0.004, body_color, "Z"),
    ]
    return boxes, cylinders


def _make_pan(name: str, xy: tuple[float, float], yaw: float, body_color: tuple[float, float, float]):
    boxes, cylinders = _pan_spec_primitives(body_color)
    return part(
        name,
        pos=(xy[0], xy[1], TOP),
        yaw=yaw,
        mass=PAN_MASS,
        fixed=False,
        friction=0.90,
        color=body_color,
        boxes=boxes,
        cylinders=cylinders,
    )


_PAN_BOX_SPECS, _PAN_CYL_SPECS = _pan_spec_primitives(ENAMEL_NAVY)

TRIVET = _make_trivet()
PAN_LEFT = _make_pan("pan_left", (PAN_START_X, +PAN_START_Y), -PAN_START_YAW, ENAMEL_NAVY)
PAN_RIGHT = _make_pan("pan_right", (PAN_START_X, -PAN_START_Y), +PAN_START_YAW, ENAMEL_BURGUNDY)
OBJECTS = [TRIVET, PAN_LEFT, PAN_RIGHT]


# -- Exact analytical point-to-solid collider distances (zero surface-sampling gap!) ----------------------------------


def _rot_z3(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def min_dist_points_to_solids(
    pts_w: np.ndarray,
    boxes_world: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    cyls_world: list[tuple[np.ndarray, np.ndarray, float, float]],
) -> float:
    """Exact continuous Euclidean distance from ``pts_w`` (N, 3) to the union of oriented solid boxes and Z-cylinders."""
    min_d2 = float("inf")
    for c_w, R_w, half_s in boxes_world:
        rel = (pts_w - c_w[None, :]) @ R_w
        q = np.maximum(np.abs(rel) - half_s[None, :], 0.0)
        d2 = float(np.min(np.sum(q * q, axis=1)))
        if d2 < min_d2:
            min_d2 = d2
            if min_d2 == 0.0:
                return 0.0
    for c_w, R_w, radius, half_h in cyls_world:
        rel = (pts_w - c_w[None, :]) @ R_w
        dr = np.maximum(np.hypot(rel[:, 0], rel[:, 1]) - radius, 0.0)
        dz = np.maximum(np.abs(rel[:, 2]) - half_h, 0.0)
        d2 = float(np.min(dr * dr + dz * dz))
        if d2 < min_d2:
            min_d2 = d2
            if min_d2 == 0.0:
                return 0.0
    return math.sqrt(min_d2)


def trivet_kinematics(tr: PartState) -> dict:
    """Compute exact 3D bar solids, outer envelope (L_y, W_x), center, principal frame, and visible marks."""
    q_b0 = float(tr.joints.get("j_b0", 0.0))
    q_bp1 = float(tr.joints.get("j_b_p1", q_b0))
    q_ap1 = float(tr.joints.get("j_a_p1", -q_b0))
    q_bm1 = float(tr.joints.get("j_b_m1", q_b0))
    q_am1 = float(tr.joints.get("j_a_m1", -q_b0))

    if not all(math.isfinite(v) for v in (q_b0, q_bp1, q_ap1, q_bm1, q_am1)):
        raise ValueError("non-finite trivet joint state")

    ang_b0 = 2.0 * TRIVET_THETA0 + q_b0
    ang_bp1 = 2.0 * TRIVET_THETA0 + q_bp1
    ang_ap1 = q_bp1 + q_ap1
    ang_bm1 = 2.0 * TRIVET_THETA0 + q_bm1
    ang_am1 = q_bm1 + q_am1

    u_a0 = np.array([1.0, 0.0], dtype=np.float64)
    u_b0 = np.array([math.cos(ang_b0), math.sin(ang_b0)], dtype=np.float64)
    u_bp1 = np.array([math.cos(ang_bp1), math.sin(ang_bp1)], dtype=np.float64)
    u_ap1 = np.array([math.cos(ang_ap1), math.sin(ang_ap1)], dtype=np.float64)
    u_bm1 = np.array([math.cos(ang_bm1), math.sin(ang_bm1)], dtype=np.float64)
    u_am1 = np.array([math.cos(ang_am1), math.sin(ang_am1)], dtype=np.float64)

    c_a0 = np.zeros(2, dtype=np.float64)
    c_b0 = np.zeros(2, dtype=np.float64)
    c_bp1 = np.array([-TRIVET_A, 0.0]) + TRIVET_A * u_bp1
    c_ap1 = c_bp1.copy()
    c_bm1 = np.array([+TRIVET_A, 0.0]) - TRIVET_A * u_bm1
    c_am1 = c_bm1.copy()

    bars_local = [
        ("A", c_a0, 0.0, u_a0, 0.0),
        ("B", c_b0, ang_b0, u_b0, 0.0),
        ("B", c_bp1, ang_bp1, u_bp1, +1.0),
        ("A", c_ap1, ang_ap1, u_ap1, -1.0),
        ("B", c_bm1, ang_bm1, u_bm1, -1.0),
        ("A", c_am1, ang_am1, u_am1, +1.0),
    ]

    R_w = tr.rotation()
    p_w = np.asarray(tr.pos, dtype=np.float64)
    if not (np.all(np.isfinite(R_w)) and np.all(np.isfinite(p_w))):
        raise ValueError("non-finite trivet pose")

    half_open = TRIVET_THETA0 + 0.5 * q_b0
    x_lat_w = R_w @ np.array([math.cos(half_open), math.sin(half_open), 0.0], dtype=np.float64)
    y_lat_w = R_w @ np.array([-math.sin(half_open), math.cos(half_open), 0.0], dtype=np.float64)

    boxes_world: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    proj_x: list[float] = []
    proj_y: list[float] = []
    min_bottom_z = float("inf")

    for layer, c2, ang, u2, fend in bars_local:
        box_specs = _bar_boxes_a((float(c2[0]), float(c2[1])), ang, fend) if layer == "A" else _bar_boxes_b((float(c2[0]), float(c2[1])), ang, fend)
        for b_c, b_s, _, b_yaw in box_specs:
            c_loc = np.asarray(b_c, dtype=np.float64)
            half_s = 0.5 * np.asarray(b_s, dtype=np.float64)
            R_box = R_w @ _rot_z3(float(b_yaw))
            c_box_w = p_w + R_w @ c_loc
            boxes_world.append((c_box_w, R_box, half_s))
            # Exact lowest z of oriented box: c_box_w[2] - sum(|R_box[2, :]| * half_s)
            z_lo = float(c_box_w[2] - np.sum(np.abs(R_box[2, :]) * half_s))
            if z_lo < min_bottom_z:
                min_bottom_z = z_lo

        v2 = np.array([-u2[1], u2[0]], dtype=np.float64)
        for s_end in (-(TRIVET_A + TRIVET_R_END), +(TRIVET_A + TRIVET_R_END)):
            for w_off in (-TRIVET_R_END, +TRIVET_R_END):
                xy = c2 + s_end * u2 + w_off * v2
                rel = R_w @ np.array([xy[0], xy[1], 0.0], dtype=np.float64)
                proj_x.append(float(np.dot(rel, x_lat_w)))
                proj_y.append(float(np.dot(rel, y_lat_w)))

    length_y = float(max(proj_y) - min(proj_y))
    width_x = float(max(proj_x) - min(proj_x))
    end_p_local = 0.5 * ((c_bp1 + TRIVET_A * u_bp1) + (c_ap1 - TRIVET_A * u_ap1))
    end_m_local = 0.5 * ((c_bm1 - TRIVET_A * u_bm1) + (c_am1 + TRIVET_A * u_am1))
    vis_marks = np.vstack([
        p_w + R_w @ np.array([0.0, 0.0, TRIVET_HALF_H]),
        p_w + R_w @ np.array([end_p_local[0], end_p_local[1], TRIVET_HALF_H]),
        p_w + R_w @ np.array([end_m_local[0], end_m_local[1], TRIVET_HALF_H]),
    ])
    return {
        "q_b0": q_b0,
        "theta_deg": math.degrees(half_open),
        "length_m": length_y,
        "width_m": width_x,
        "center_w": p_w,
        "x_axis_w": x_lat_w,
        "y_axis_w": y_lat_w,
        "tilt_deg": tr.tilt_deg(),
        "min_bottom_z": min_bottom_z,
        "boxes_world": boxes_world,
        "vis_marks": vis_marks,
    }


def pan_kinematics(pan: PartState) -> dict:
    """Compute exact 3D box/cylinder collider solids, analytical extreme heights, and visible marks for one pan."""
    R_w = pan.rotation()
    p_w = np.asarray(pan.pos, dtype=np.float64)
    if not (np.all(np.isfinite(R_w)) and np.all(np.isfinite(p_w))):
        raise ValueError("non-finite pan pose")

    boxes_world: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cyls_world: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    min_any_z = float("inf")

    for b_c, b_s, _, b_yaw in _PAN_BOX_SPECS:
        c_loc = np.asarray(b_c, dtype=np.float64)
        half_s = 0.5 * np.asarray(b_s, dtype=np.float64)
        R_box = R_w @ _rot_z3(float(b_yaw))
        c_box_w = p_w + R_w @ c_loc
        boxes_world.append((c_box_w, R_box, half_s))
        z_lo = float(c_box_w[2] - np.sum(np.abs(R_box[2, :]) * half_s))
        if z_lo < min_any_z:
            min_any_z = z_lo

    for c_c, c_r, c_h, _, _ in _PAN_CYL_SPECS:
        c_loc = np.asarray(c_c, dtype=np.float64)
        half_h = 0.5 * float(c_h)
        radius = float(c_r)
        c_cyl_w = p_w + R_w @ c_loc
        cyls_world.append((c_cyl_w, R_w, radius, half_h))
        # Exact lowest z of oriented Z-cylinder: center_z - |R[2,2]|*half_h - sqrt(R[2,0]^2 + R[2,1]^2)*radius
        sin_tilt = math.hypot(float(R_w[2, 0]), float(R_w[2, 1]))
        z_lo = float(c_cyl_w[2] - abs(float(R_w[2, 2])) * half_h - sin_tilt * radius)
        if z_lo < min_any_z:
            min_any_z = z_lo

    # Exact analytical min/max height of the bottom contact circle (radius PAN_R_BASE at local z = 0)
    sin_tilt = math.hypot(float(R_w[2, 0]), float(R_w[2, 1]))
    min_base_z = float(p_w[2] - sin_tilt * PAN_R_BASE)
    max_base_z = float(p_w[2] + sin_tilt * PAN_R_BASE)

    vis_marks = np.vstack([
        p_w + R_w @ np.array([0.0, 0.0, PAN_BASE_T]),
        p_w + R_w @ np.array([0.0, +PAN_HANDLE_R, PAN_HANDLE_Z]),
        p_w + R_w @ np.array([0.0, -PAN_HANDLE_R, PAN_HANDLE_Z]),
    ])
    return {
        "center_w": p_w,
        "tilt_deg": pan.tilt_deg(),
        "min_any_z": min_any_z,
        "min_base_z": min_base_z,
        "max_base_z": max_base_z,
        "boxes_world": boxes_world,
        "cyls_world": cyls_world,
        "vis_marks": vis_marks,
    }


def certified_hand_clearance_to_objects_m(hand_pts_w: np.ndarray, tk: dict, pk_l: dict, pk_r: dict) -> float:
    """Exact conservative surface separation from 424 hand spheres to all 64 actual trivet+pan collider solids."""
    all_boxes = tk["boxes_world"] + pk_l["boxes_world"] + pk_r["boxes_world"]
    all_cyls = pk_l["cyls_world"] + pk_r["cyls_world"]
    raw_solid_dist_m = min_dist_points_to_solids(hand_pts_w, all_boxes, all_cyls)
    return float(raw_solid_dist_m - HAND_COVER_RADIUS_M)


# -- Live step tracker (requires active observation + 21 consecutive released-and-still samples @ 20 Hz) --------------


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("invalid quaternion shape/NaN")
    n = float(np.linalg.norm(q))
    if abs(n - 1.0) > 0.02:
        raise ValueError(f"non-unit quaternion norm={n:.4f}")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def certified_hand_points_from_link_poses(link_poses_xyzw: dict[str, np.ndarray]) -> np.ndarray:
    """Build (424, 3) certified hand cover points from 6 link 7-poses (x,y,z,qx,qy,qz,qw), validating URDF rails."""
    out = []
    for arm in ("left", "right"):
        g_pose = np.asarray(link_poses_xyzw[f"{arm}.gripper"], dtype=np.float64)
        tl_pose = np.asarray(link_poses_xyzw[f"{arm}.tip_left"], dtype=np.float64)
        tr_pose = np.asarray(link_poses_xyzw[f"{arm}.tip_right"], dtype=np.float64)
        if g_pose.shape != (7,) or tl_pose.shape != (7,) or tr_pose.shape != (7,):
            raise ValueError("link pose must be 7-vector")
        p_g, R_g = g_pose[:3], _quat_xyzw_to_rot(g_pose[3:7])
        p_tl, R_tl = tl_pose[:3], _quat_xyzw_to_rot(tl_pose[3:7])
        p_tr, R_tr = tr_pose[:3], _quat_xyzw_to_rot(tr_pose[3:7])
        if not (np.all(np.isfinite(p_g)) and np.all(np.isfinite(p_tl)) and np.all(np.isfinite(p_tr))):
            raise ValueError("non-finite link position")

        expected_rf = R_g @ _J_FINGER_ROT0
        if float(np.linalg.norm(R_tl - expected_rf, ord="fro")) > _FINGER_ROT_FROB_TOL:
            raise ValueError(f"{arm}.tip_left orientation mismatch")
        if float(np.linalg.norm(R_tr - expected_rf, ord="fro")) > _FINGER_ROT_FROB_TOL:
            raise ValueError(f"{arm}.tip_right orientation mismatch")

        rel_tl = R_g.T @ (p_tl - p_g) - _J7_ORIGIN_XYZ
        q_tl = float(np.dot(rel_tl, _J7_AXIS_G))
        perp_tl = float(np.linalg.norm(rel_tl - q_tl * _J7_AXIS_G))
        if perp_tl > _FINGER_RAIL_PERP_TOL or not (_FINGER_SLIDE_MIN <= q_tl <= _FINGER_SLIDE_MAX):
            raise ValueError(f"{arm}.tip_left rail violation (q={q_tl:.4f}, perp={perp_tl:.4f})")

        rel_tr = R_g.T @ (p_tr - p_g) - _J8_ORIGIN_XYZ
        q_tr = float(np.dot(rel_tr, _J8_AXIS_G))
        perp_tr = float(np.linalg.norm(rel_tr - q_tr * _J8_AXIS_G))
        if perp_tr > _FINGER_RAIL_PERP_TOL or not (_FINGER_SLIDE_MIN <= q_tr <= _FINGER_SLIDE_MAX):
            raise ValueError(f"{arm}.tip_right rail violation (q={q_tr:.4f}, perp={perp_tr:.4f})")

        pts_arm = np.vstack([
            (R_g @ GRIPPER_D405_COVER_LOCAL.T).T + p_g,
            (R_tl @ TIP_LEFT_COVER_NATIVE.T).T + p_tl,
            (R_tr @ TIP_RIGHT_COVER_NATIVE.T).T + p_tr,
        ])
        if pts_arm.shape != (N_PTS_PER_HAND, 3):
            raise ValueError("unexpected cover point count")
        out.append(pts_arm)
    arr = np.vstack(out)
    if arr.shape != (2 * N_PTS_PER_HAND, 3):
        raise ValueError("unexpected total cover shape")
    return arr


class TrivetTrialTracker:
    """Fail-closed live step tracker requiring active observation and a full 1.0 s released-and-still window."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, step_dt: float = 0.05) -> None:
        self.active = False
        self.failed = False
        self.fail_reason = ""
        self.step_dt = float(step_dt)
        self.last_step: int | None = None
        self.hand_pts_w: np.ndarray | None = None
        self.mark_window: deque[np.ndarray] = deque(maxlen=64)
        self.clear_window: deque[float] = deque(maxlen=64)

    def fail(self, reason: str) -> None:
        self.failed = True
        if not self.fail_reason:
            self.fail_reason = reason

    @property
    def required_samples(self) -> int:
        if not math.isfinite(self.step_dt) or self.step_dt <= 0.0 or self.step_dt > 0.25:
            return 999999
        return int(math.ceil(HOLD_WINDOW_S / self.step_dt - 1e-9)) + 1

    def record(
        self,
        sim_step: int,
        step_dt: float,
        link_poses_xyzw: dict[str, np.ndarray],
        objects: dict[str, PartState],
    ) -> None:
        self.active = True
        if self.failed:
            return
        if isinstance(sim_step, bool) or not isinstance(sim_step, int) or sim_step < 0:
            self.fail("invalid_sim_step")
            return
        if isinstance(step_dt, bool) or not isinstance(step_dt, (int, float)) or not math.isfinite(step_dt) or not (0.005 <= float(step_dt) <= 0.20):
            self.fail("invalid_step_dt")
            return
        self.step_dt = float(step_dt)
        if self.last_step is not None and sim_step != self.last_step + 1:
            self.fail(f"non_consecutive_step:{self.last_step}->{sim_step}")
            return
        self.last_step = sim_step

        try:
            self.hand_pts_w = certified_hand_points_from_link_poses(link_poses_xyzw)
            tk = trivet_kinematics(objects["trivet"])
            pk_l = pan_kinematics(objects["pan_left"])
            pk_r = pan_kinematics(objects["pan_right"])
            step_clear_m = certified_hand_clearance_to_objects_m(self.hand_pts_w, tk, pk_l, pk_r)
            marks = np.vstack([tk["vis_marks"], pk_l["vis_marks"], pk_r["vis_marks"]])
        except Exception as exc:
            self.fail(f"step_observation_invalid:{exc}")
            return

        if marks.shape != (9, 3) or not np.all(np.isfinite(marks)) or not math.isfinite(step_clear_m):
            self.fail("invalid_visible_marks_or_clearance")
            return
        self.mark_window.append(marks.copy())
        self.clear_window.append(float(step_clear_m))

    def window_stats(self) -> tuple[bool, float, float]:
        """Return (window_complete, min_certified_clear_over_window_m, max_mark_spread_over_window_m)."""
        req = self.required_samples
        if len(self.mark_window) < req or len(self.clear_window) < req:
            return False, -float("inf"), float("inf")
        recent_marks = np.stack(list(self.mark_window)[-req:], axis=0)
        spread_m = float(np.max(np.linalg.norm(np.ptp(recent_marks, axis=0), axis=1)))
        min_clear_m = float(min(list(self.clear_window)[-req:]))
        return True, min_clear_m, spread_m


_TRACKER = TrivetTrialTracker()
_TRACE_EVERY = int(os.environ.get("TRIVET_TRACE_EVERY", "0") or "0")


def _extract_robot_links_from_env(environment) -> dict[str, np.ndarray]:
    base = environment.unwrapped
    origin = np.asarray(base.scene.env_origins[0].detach().cpu().numpy(), dtype=np.float64)
    poses: dict[str, np.ndarray] = {}
    for arm, key in (("left", "left_robot"), ("right", "right_robot")):
        art = base.scene[key]
        bnames = list(art.body_names)
        bpose = np.asarray(art.data.body_link_pose_w[0].detach().cpu().numpy(), dtype=np.float64)
        for link in ("gripper", "tip_left", "tip_right"):
            idx = bnames.index(link)
            p7 = bpose[idx].copy()
            p7[:3] -= origin
            poses[f"{arm}.{link}"] = p7
    return poses


@on_reset
def _on_env_reset(environment) -> None:
    dt = float(getattr(environment.unwrapped, "step_dt", 0.05))
    _TRACKER.reset(step_dt=dt)
    if _TRACE_EVERY > 0:
        try:
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import PhysxSchema, UsdPhysics

            stage = get_current_stage()
            t_prim = environment.unwrapped.scene["trivet"].cfg.prim_path.replace(".*", "0")
            readback = {}
            for j in TRIVET.joints:
                jp = stage.GetPrimAtPath(f"{t_prim}/{j.name}/{j.joint_name}")
                rj = UsdPhysics.RevoluteJoint(jp)
                info = {
                    "prim": str(jp.GetPath()),
                    "axis": rj.GetAxisAttr().Get(),
                    "lower_deg": rj.GetLowerLimitAttr().Get(),
                    "upper_deg": rj.GetUpperLimitAttr().Get(),
                }
                for tok in ("rotX", "rotY", "rotZ"):
                    if jp.HasAPI(PhysxSchema.PhysxMimicJointAPI, tok):
                        mj = PhysxSchema.PhysxMimicJointAPI(jp, tok)
                        info["mimic_instance"] = tok
                        info["ref_joint"] = [str(p) for p in mj.GetReferenceJointRel().GetTargets()]
                        info["ref_axis"] = mj.GetReferenceJointAxisAttr().Get()
                        info["gearing"] = mj.GetGearingAttr().Get()
                        info["offset"] = mj.GetOffsetAttr().Get()
                if jp.HasAPI(UsdPhysics.DriveAPI, "angular"):
                    drv = UsdPhysics.DriveAPI(jp, "angular")
                    info["drive_stiffness"] = drv.GetStiffnessAttr().Get()
                    info["drive_damping"] = drv.GetDampingAttr().Get()
                if jp.HasAPI(PhysxSchema.PhysxJointAPI):
                    info["joint_friction"] = PhysxSchema.PhysxJointAPI(jp).GetJointFrictionAttr().Get()
                readback[j.name] = info
            print("TRIVET_USD_READBACK " + json.dumps(readback), flush=True)
        except Exception as exc:
            print(f"TRIVET_USD_READBACK_ERR {exc}", flush=True)


@on_step
def _on_env_step(environment) -> None:
    try:
        base = environment.unwrapped
        dt = float(getattr(base, "step_dt", 0.05))
        step_i = int(base.episode_length_buf[0].item())
        from yam_arena import scene as yam_scene

        objs = yam_scene.object_states(environment, ("trivet", "pan_left", "pan_right"))
        links = _extract_robot_links_from_env(environment)
        _TRACKER.record(step_i, dt, links, objs)
        if _TRACE_EVERY > 0 and (step_i % _TRACE_EVERY == 0 or step_i == 1):
            ev = evaluate_trial(objs, _TRACKER)
            print("TRIVET_TRACE " + json.dumps({"step": step_i, "joints": {k: round(math.degrees(v), 2) for k, v in objs["trivet"].joints.items()}, **ev}), flush=True)
    except Exception as exc:
        _TRACKER.active = True
        _TRACKER.fail(f"env_step_error:{exc}")


# -- Outcome-only ruler / side-photo scorer ---------------------------------------------------------------------------


def _pan_valid_on_trivet(pk: dict, tk: dict) -> bool:
    min_any_rel = pk["min_any_z"] - TOP
    min_base_rel = pk["min_base_z"] - TOP
    max_base_rel = pk["max_base_z"] - TOP
    if min_any_rel < PAN_TABLE_CLEAR_MIN_M:
        return False
    if min_base_rel < PAN_SEAT_Z_MIN_M or max_base_rel > PAN_SEAT_Z_MAX_M:
        return False
    if pk["tilt_deg"] > PAN_TILT_MAX_DEG:
        return False
    rel = pk["center_w"] - tk["center_w"]
    dx = abs(float(np.dot(rel, tk["x_axis_w"])))
    dy = abs(float(np.dot(rel, tk["y_axis_w"])))
    if dx > PAN_SUPPORT_X_MAX_M:
        return False
    if dy > 0.5 * tk["length_m"] - PAN_SUPPORT_END_MARGIN_M:
        return False
    return True


def evaluate_trial(objects: dict[str, PartState], tracker: TrivetTrialTracker | None = None) -> dict:
    """Fail-closed ruler/photo evaluation requiring active observation & 21 consecutive released+still frames @ 20 Hz."""
    trk = _TRACKER if tracker is None else tracker
    try:
        tr = objects["trivet"]
        pl = objects["pan_left"]
        pr = objects["pan_right"]
        tk = trivet_kinematics(tr)
        pk_l = pan_kinematics(pl)
        pk_r = pan_kinematics(pr)
    except Exception as exc:
        return {"score": 0.0, "reason": f"malformed_objects:{exc}"}

    if tk["min_bottom_z"] < TOP - 0.040 or pk_l["min_any_z"] < TOP - 0.040 or pk_r["min_any_z"] < TOP - 0.040:
        return {"score": 0.0, "reason": "object_off_table"}

    # Fail closed if tracker is inactive, faulted, or has not yet accumulated a full 1.0 s released-and-still window
    if not trk.active:
        return {"score": 0.0, "reason": "tracker_inactive_no_observations"}
    if trk.failed or trk.hand_pts_w is None:
        return {"score": 0.0, "reason": f"tracker_failed:{trk.fail_reason}"}

    # Verify live env sync if running inside Isaac Lab
    from labkit import env as live_env_fn

    live_env = live_env_fn()
    if live_env is not None and trk is _TRACKER:
        try:
            cur_ep_step = int(live_env.unwrapped.episode_length_buf[0].item())
            if trk.last_step != cur_ep_step:
                return {"score": 0.0, "reason": f"stale_tracker_step:{trk.last_step}!={cur_ep_step}"}
        except Exception as exc:
            return {"score": 0.0, "reason": f"live_env_sync_error:{exc}"}

    win_ok, win_min_clear_m, win_spread_m = trk.window_stats()
    # Also evaluate current-frame exact solid clearance so caller tampering with `objects` after record() cannot pass
    cur_clear_m = certified_hand_clearance_to_objects_m(trk.hand_pts_w, tk, pk_l, pk_r)
    effective_clear_m = min(win_min_clear_m, cur_clear_m)

    if not win_ok:
        return {
            "score": 0.0,
            "reason": "hold_window_incomplete",
            "samples_recorded": len(trk.mark_window),
            "samples_required": trk.required_samples,
        }
    if effective_clear_m < HAND_CLEAR_MIN_M:
        return {
            "score": 0.0,
            "reason": "hand_not_released_80mm_full_window",
            "certified_hand_clear_m": effective_clear_m,
            "cur_certified_hand_clear_m": cur_clear_m,
        }
    if not (win_spread_m < STILL_SPREAD_MAX_M):
        return {
            "score": 0.0,
            "reason": "not_still_1s_5mm",
            "still_spread_m": win_spread_m,
        }

    trivet_flat = (
        tk["tilt_deg"] <= TRIVET_TILT_MAX_DEG
        and (TOP - 0.004) <= tk["min_bottom_z"] <= (TOP + TRIVET_BASE_GAP_MAX_M)
    )
    trivet_expanded = trivet_flat and (tk["length_m"] >= EXPANDED_LEN_MIN_M)
    if not trivet_expanded:
        return {
            "score": 0.0,
            "reason": "trivet_not_flat_and_expanded",
            "trivet_length_m": tk["length_m"],
            "trivet_tilt_deg": tk["tilt_deg"],
        }

    pan_l_ok = _pan_valid_on_trivet(pk_l, tk)
    pan_r_ok = _pan_valid_on_trivet(pk_r, tk)
    horiz_sep_m = float(np.linalg.norm(pk_l["center_w"][:2] - pk_r["center_w"][:2]))
    non_overlapping = horiz_sep_m >= PAN_NONOVERLAP_MIN_M

    if pan_l_ok and pan_r_ok and non_overlapping:
        val = 1.0
        reason = "both_pans_supported_clear_level_released_still"
    elif pan_l_ok or pan_r_ok:
        val = 0.50
        reason = "one_pan_supported_on_expanded_trivet"
    else:
        val = 0.25
        reason = "trivet_expanded_only"

    return {
        "score": val,
        "reason": reason,
        "trivet_length_m": tk["length_m"],
        "trivet_width_m": tk["width_m"],
        "pan_left_ok": pan_l_ok,
        "pan_right_ok": pan_r_ok,
        "pan_horiz_sep_m": horiz_sep_m,
        "certified_hand_clear_m": effective_clear_m,
        "still_spread_m": win_spread_m,
    }


def score(objects: dict[str, PartState]) -> float:
    return float(evaluate_trial(objects)["score"])


def success(objects: dict[str, PartState]) -> bool:
    return score(objects) >= 1.0


# -- Bimanual scripted demonstration (~36.2 s = 362 steps @ 10 Hz / 725 steps @ 20 Hz <= 1,200 cap) -------------------


def _trivet_grasp_pose(theta: float, side: float, outward_bias_m: float = 0.0) -> tuple[tuple[float, float, float], float]:
    """Antipodal opposing-diagonal grasp on B_+1 (+X,+Y for left arm, side=+1) & B_-1 (-X,-Y for right arm, side=-1)
    at s_frac = 0.82 (27 mm inward from bar tip) and z_tcp = TOP + 0.0255 m (pad tips at TOP + 0.0125 m: 7.5 mm blade
    entry depth, +2.5 mm above lower-layer A beams). Includes +2.5 mm Y PD-lag compensation on side=-1 (natively
    validated in int4: both hands pinch at joint7/8 = -0.0049 m, opening lattice to 462.6 mm with 3.0 mm center drift).
    """
    s_frac = 0.82
    n_out_x = -side * math.sin(theta)
    n_out_y = +side * math.cos(theta)
    wx = TRIVET_X + side * (s_frac * TRIVET_A * math.cos(theta)) + outward_bias_m * n_out_x
    wy = TRIVET_Y + side * ((2.0 + s_frac) * TRIVET_A * math.sin(theta)) + outward_bias_m * n_out_y
    if side < 0.0:
        wy += 0.0025
    return (wx, wy, TOP + 0.0255), theta


def _pan_handle_poses(center_xy: tuple[float, float], yaw: float, z_tcp: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cx, cy = center_xy
    s, c = math.sin(yaw), math.cos(yaw)
    p_left = (cx - PAN_HANDLE_R * s, cy + PAN_HANDLE_R * c, z_tcp)
    p_right = (cx + PAN_HANDLE_R * s, cy - PAN_HANDLE_R * c, z_tcp)
    return p_left, p_right


def _demo_direct() -> dict[str, list[Waypoint]]:
    """Coordinated bimanual expansion of trivet + two-hand zero-cantilever placement of both pans + full retreat."""
    z_hover_low = TOP + 0.092
    z_hover_carry = TOP + 0.098
    z_pan_pick = TOP + 0.048
    z_pan_place = TOP + 0.068

    F_OPEN_TRIVET = -0.0095    # 19.0 mm inside gap around 10.0 mm bar => >= +3.10 mm 3D STL clearance to all 18 bars on descent!
    F_RELEASE_TRIVET = -0.0140 # 28.0 mm release gap before lifting clear
    F_PINCH = FINGER_CLOSED

    left_wps: list[Waypoint] = []
    right_wps: list[Waypoint] = []

    # Phase 1: Antipodal X-staggered grasp & smooth circular-arc expansion of coupled scissor trivet (q: 0 -> 66 deg)
    pL0_desc, yaw0 = _trivet_grasp_pose(TRIVET_THETA0, +1.0, outward_bias_m=0.0010)
    pR0_desc, _ = _trivet_grasp_pose(TRIVET_THETA0, -1.0, outward_bias_m=0.0010)
    pL0, _ = _trivet_grasp_pose(TRIVET_THETA0, +1.0, outward_bias_m=0.0)
    pR0, _ = _trivet_grasp_pose(TRIVET_THETA0, -1.0, outward_bias_m=0.0)
    left_wps.extend([
        Waypoint(1.4, (pL0_desc[0], pL0_desc[1], z_hover_low), F_OPEN_TRIVET, "hover_trivet_L", yaw0),
        Waypoint(1.2, pL0_desc, F_OPEN_TRIVET, "descend_trivet_L", yaw0),
        Waypoint(0.9, pL0, F_PINCH, "pinch_trivet_L", yaw0),
    ])
    right_wps.extend([
        Waypoint(1.4, (pR0_desc[0], pR0_desc[1], z_hover_low), F_OPEN_TRIVET, "hover_trivet_R", yaw0),
        Waypoint(1.2, pR0_desc, F_OPEN_TRIVET, "descend_trivet_R", yaw0),
        Waypoint(0.9, pR0, F_PINCH, "pinch_trivet_R", yaw0),
    ])

    n_arc = 5
    for k in range(1, n_arc + 1):
        th_k = TRIVET_THETA0 + (k / n_arc) * (0.5 * TRIVET_Q_DEMO)
        pLk, yawk = _trivet_grasp_pose(th_k, +1.0)
        pRk, _ = _trivet_grasp_pose(th_k, -1.0)
        dur = 1.3 if k == 1 else 1.0  # smooth initial breakaway ease-in
        left_wps.append(Waypoint(dur, pLk, F_PINCH, f"expand_arc_{k}_L", yawk))
        right_wps.append(Waypoint(dur, pRk, F_PINCH, f"expand_arc_{k}_R", yawk))

    pL_end, yaw_end = _trivet_grasp_pose(TRIVET_THETA0 + 0.5 * TRIVET_Q_DEMO, +1.0)
    pR_end, _ = _trivet_grasp_pose(TRIVET_THETA0 + 0.5 * TRIVET_Q_DEMO, -1.0)
    left_wps.extend([
        Waypoint(0.5, pL_end, F_PINCH, "settle_trivet_L", yaw_end),
        Waypoint(0.6, pL_end, F_RELEASE_TRIVET, "release_trivet_L", yaw_end),
        Waypoint(1.2, (pL_end[0], pL_end[1], z_hover_carry), FINGER_OPEN, "lift_from_trivet_L", 0.0),
    ])
    right_wps.extend([
        Waypoint(0.5, pR_end, F_PINCH, "settle_trivet_R", yaw_end),
        Waypoint(0.6, pR_end, F_RELEASE_TRIVET, "release_trivet_R", yaw_end),
        Waypoint(1.2, (pR_end[0], pR_end[1], z_hover_carry), FINGER_OPEN, "lift_from_trivet_R", 0.0),
    ])

    # Phase 2: Bimanual two-handle pick & placement of pan_left onto +Y expanded trivet zone (parallel -30 deg yaw)
    F_RELEASE_PAN = -0.024  # 48 mm jaw opening around 18 mm handle bar (15 mm per-side clearance) before lifting clear
    yaw_pl = -PAN_START_YAW
    yaw_goal = PAN_GOAL_YAW
    pl_pick_L, pl_pick_R = _pan_handle_poses((PAN_START_X, +PAN_START_Y), yaw_pl, z_pan_pick)
    pl_carry_s_L, pl_carry_s_R = _pan_handle_poses((PAN_START_X, +PAN_START_Y), yaw_pl, z_hover_carry)
    pl_carry_g_L, pl_carry_g_R = _pan_handle_poses((TRIVET_X, +PAN_GOAL_Y), yaw_goal, z_hover_carry)
    pl_place_L, pl_place_R = _pan_handle_poses((TRIVET_X, +PAN_GOAL_Y), yaw_goal, z_pan_place)

    left_wps.extend([
        Waypoint(1.8, pl_carry_s_L, FINGER_OPEN, "hover_pan1_L", yaw_pl),
        Waypoint(1.4, pl_pick_L, FINGER_OPEN, "descend_pan1_L", yaw_pl),
        Waypoint(0.8, pl_pick_L, F_PINCH, "grasp_pan1_L", yaw_pl),
        Waypoint(1.4, pl_carry_s_L, F_PINCH, "lift_pan1_L", yaw_pl),
        Waypoint(2.4, pl_carry_g_L, F_PINCH, "carry_pan1_L", yaw_goal),
        Waypoint(1.4, pl_place_L, F_PINCH, "seat_pan1_L", yaw_goal),
        Waypoint(0.7, pl_place_L, F_RELEASE_PAN, "release_pan1_L", yaw_goal),
        Waypoint(1.3, pl_carry_g_L, FINGER_OPEN, "retreat_pan1_L", yaw_goal),
    ])
    right_wps.extend([
        Waypoint(1.8, pl_carry_s_R, FINGER_OPEN, "hover_pan1_R", yaw_pl),
        Waypoint(1.4, pl_pick_R, FINGER_OPEN, "descend_pan1_R", yaw_pl),
        Waypoint(0.8, pl_pick_R, F_PINCH, "grasp_pan1_R", yaw_pl),
        Waypoint(1.4, pl_carry_s_R, F_PINCH, "lift_pan1_R", yaw_pl),
        Waypoint(2.4, pl_carry_g_R, F_PINCH, "carry_pan1_R", yaw_goal),
        Waypoint(1.4, pl_place_R, F_PINCH, "seat_pan1_R", yaw_goal),
        Waypoint(0.7, pl_place_R, F_RELEASE_PAN, "release_pan1_R", yaw_goal),
        Waypoint(1.3, pl_carry_g_R, FINGER_OPEN, "retreat_pan1_R", yaw_goal),
    ])

    # Phase 3: Bimanual two-handle pick & placement of pan_right onto -Y expanded trivet zone (parallel -30 deg yaw)
    yaw_pr = +PAN_START_YAW
    pr_pick_L, pr_pick_L_R = _pan_handle_poses((PAN_START_X, -PAN_START_Y), yaw_pr, z_pan_pick)
    pr_pick_R = pr_pick_L_R
    pr_carry_s_L, pr_carry_s_R = _pan_handle_poses((PAN_START_X, -PAN_START_Y), yaw_pr, z_hover_carry)
    pr_carry_g_L, pr_carry_g_R = _pan_handle_poses((TRIVET_X, -PAN_GOAL_Y), yaw_goal, z_hover_carry)
    pr_place_L, pr_place_R = _pan_handle_poses((TRIVET_X, -PAN_GOAL_Y), yaw_goal, z_pan_place)

    left_wps.extend([
        Waypoint(2.0, pr_carry_s_L, FINGER_OPEN, "hover_pan2_L", yaw_pr),
        Waypoint(1.4, pr_pick_L, FINGER_OPEN, "descend_pan2_L", yaw_pr),
        Waypoint(0.8, pr_pick_L, F_PINCH, "grasp_pan2_L", yaw_pr),
        Waypoint(1.4, pr_carry_s_L, F_PINCH, "lift_pan2_L", yaw_pr),
        Waypoint(2.4, pr_carry_g_L, F_PINCH, "carry_pan2_L", yaw_goal),
        Waypoint(1.4, pr_place_L, F_PINCH, "seat_pan2_L", yaw_goal),
        Waypoint(0.7, pr_place_L, F_RELEASE_PAN, "release_pan2_L", yaw_goal),
        Waypoint(1.3, pr_carry_g_L, FINGER_OPEN, "retreat_pan2_L", yaw_goal),
    ])
    right_wps.extend([
        Waypoint(2.0, pr_carry_s_R, FINGER_OPEN, "hover_pan2_R", yaw_pr),
        Waypoint(1.4, pr_pick_R, FINGER_OPEN, "descend_pan2_R", yaw_pr),
        Waypoint(0.8, pr_pick_R, F_PINCH, "grasp_pan2_R", yaw_pr),
        Waypoint(1.4, pr_carry_s_R, F_PINCH, "lift_pan2_R", yaw_pr),
        Waypoint(2.4, pr_carry_g_R, F_PINCH, "carry_pan2_R", yaw_goal),
        Waypoint(1.4, pr_place_R, F_PINCH, "seat_pan2_R", yaw_goal),
        Waypoint(0.7, pr_place_R, F_RELEASE_PAN, "release_pan2_R", yaw_goal),
        Waypoint(1.3, pr_carry_g_R, FINGER_OPEN, "retreat_pan2_R", yaw_goal),
    ])

    # Phase 4: Retreat both hands > 125 mm clear (>> 80 mm certified threshold) inside 0.150 m straight-down envelope
    retreat_L = (0.245, +0.420, TOP + 0.150)
    retreat_R = (0.245, -0.420, TOP + 0.150)
    left_wps.extend([
        Waypoint(1.8, retreat_L, FINGER_CLOSED, "final_retreat_L", 0.0),
        Waypoint(2.2, retreat_L, FINGER_CLOSED, "final_hold_L", 0.0),
    ])
    right_wps.extend([
        Waypoint(1.8, retreat_R, FINGER_CLOSED, "final_retreat_R", 0.0),
        Waypoint(2.2, retreat_R, FINGER_CLOSED, "final_hold_R", 0.0),
    ])

    return {"left": left_wps, "right": right_wps}


# ==== claude-b50: push-open then regrasp demo (demo_variants.demo_breakaway_regrasp, verbatim; T. prefixes removed) ====
def demo() -> dict[str, list[Waypoint]]:
    base = _demo_direct()
    z_hover_low = TOP + 0.092
    z_hover_carry = TOP + 0.098
    z_push = TOP + 0.0160  # int2/int2b-proven top-press height (~16.5 mm vertical preload onto bar top)
    f_reopen = -0.0145       # 29 mm inside jaw opening inside widened q=15 deg corridor
    f_release_trivet = -0.0140
    f_pinch = FINGER_CLOSED

    q_pre = math.radians(15.0)
    q_goal = TRIVET_Q_DEMO

    pL0, yaw0 = _trivet_grasp_pose(TRIVET_THETA0, +1.0)
    pR0_raw, _ = _trivet_grasp_pose(TRIVET_THETA0, -1.0)
    pR0 = (pR0_raw[0], pR0_raw[1], z_push)
    pL0_push = (pL0[0], pL0[1], z_push)

    # Rotate closed pads by +-90 deg during top-press breakaway so 27.2 mm pad length spans across Y (cannot slip between ribs!)
    yaw_push_L = yaw0 - math.pi / 2
    yaw_push_R = yaw0 + math.pi / 2

    th_pre = TRIVET_THETA0 + 0.5 * q_pre
    pL_pre, yaw_pre = _trivet_grasp_pose(th_pre, +1.0, outward_bias_m=0.0010)
    pR_pre, _ = _trivet_grasp_pose(th_pre, -1.0, outward_bias_m=0.0010)
    pL_pre_pinch, _ = _trivet_grasp_pose(th_pre, +1.0, outward_bias_m=0.0)
    pR_pre_pinch, _ = _trivet_grasp_pose(th_pre, -1.0, outward_bias_m=0.0)
    q_rg = math.radians(5.3)   # claude-b50 brk3: fixed regrasp opening (native push reaches ~5.3 deg in brk1/brk2)
    th_rg = TRIVET_THETA0 + 0.5 * q_rg
    _rg_off = float(os.environ.get("B50_RG_OFFSET_MM", "0")) / 1000.0
    pL_rg, yaw_rg = _trivet_grasp_pose(th_rg, +1.0, outward_bias_m=0.0010 + _rg_off)
    pR_rg, _ = _trivet_grasp_pose(th_rg, -1.0, outward_bias_m=0.0010 + _rg_off)
    pL_rg_pinch, _ = _trivet_grasp_pose(th_rg, +1.0, outward_bias_m=0.0 + _rg_off)
    pR_rg_pinch, _ = _trivet_grasp_pose(th_rg, -1.0, outward_bias_m=0.0 + _rg_off)

    left_p1 = [
        Waypoint(1.3, (pL0[0], pL0[1], z_hover_low), f_pinch, "hover_breakaway_L", yaw_push_L),
        Waypoint(1.0, pL0_push, f_pinch, "press_breakaway_L", yaw_push_L),
        Waypoint(1.8, (pL_pre[0], pL_pre[1], z_push), f_pinch, "push_to_15deg_L", yaw_push_L),
        Waypoint(0.9, (pL_rg[0], pL_rg[1], z_hover_low), f_reopen, "lift_regrasp_L", yaw_rg),
        Waypoint(1.0, pL_rg, f_reopen, "descend_regrasp_L", yaw_rg),
        Waypoint(0.8, pL_rg_pinch, f_pinch, "pinch_regrasp_L", yaw_rg),
    ]
    right_p1 = [
        Waypoint(1.3, (pR0[0], pR0[1], z_hover_low), f_pinch, "hover_breakaway_R", yaw_push_R),
        Waypoint(1.0, pR0, f_pinch, "press_breakaway_R", yaw_push_R),
        Waypoint(1.8, (pR_pre[0], pR_pre[1], z_push), f_pinch, "push_to_15deg_R", yaw_push_R),
        Waypoint(0.9, (pR_rg[0], pR_rg[1], z_hover_low), f_reopen, "lift_regrasp_R", yaw_rg),
        Waypoint(1.0, pR_rg, f_reopen, "descend_regrasp_R", yaw_rg),
        Waypoint(0.8, pR_rg_pinch, f_pinch, "pinch_regrasp_R", yaw_rg),
    ]

    n_arc = 4
    for k in range(1, n_arc + 1):
        q_k = q_rg + (k / n_arc) * (q_goal - q_rg)
        th_k = TRIVET_THETA0 + 0.5 * q_k
        pLk, yawk = _trivet_grasp_pose(th_k, +1.0)
        pRk, _ = _trivet_grasp_pose(th_k, -1.0)
        left_p1.append(Waypoint(1.0, pLk, f_pinch, f"expand_from15_{k}_L", yawk))
        right_p1.append(Waypoint(1.0, pRk, f_pinch, f"expand_from15_{k}_R", yawk))

    pL_end, yaw_end = _trivet_grasp_pose(TRIVET_THETA0 + 0.5 * q_goal, +1.0)
    pR_end, _ = _trivet_grasp_pose(TRIVET_THETA0 + 0.5 * q_goal, -1.0)
    left_p1.extend([
        Waypoint(0.5, pL_end, f_pinch, "settle_trivet_L", yaw_end),
        Waypoint(0.6, pL_end, f_release_trivet, "release_trivet_L", yaw_end),
        Waypoint(1.2, (pL_end[0], pL_end[1], z_hover_carry), FINGER_OPEN, "lift_from_trivet_L", 0.0),
    ])
    right_p1.extend([
        Waypoint(0.5, pR_end, f_pinch, "settle_trivet_R", yaw_end),
        Waypoint(0.6, pR_end, f_release_trivet, "release_trivet_R", yaw_end),
        Waypoint(1.2, (pR_end[0], pR_end[1], z_hover_carry), FINGER_OPEN, "lift_from_trivet_R", 0.0),
    ])

    idx_p2_L = next(i for i, w in enumerate(base["left"]) if w.label == "hover_pan1_L")
    idx_p2_R = next(i for i, w in enumerate(base["right"]) if w.label == "hover_pan1_R")
    return {
        "left": left_p1 + base["left"][idx_p2_L:],
        "right": right_p1 + base["right"][idx_p2_R:],
    }
