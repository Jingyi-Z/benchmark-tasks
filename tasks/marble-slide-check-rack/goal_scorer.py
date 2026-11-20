"""Goal-only, order-independent, fail-closed scorer prepared on CPU for marble-slide-check-rack.

Key properties (addressing codex-m2-review-20261119-1555 and codex-m3-review-20261119-1705):
1. Fail-closed certified surface-to-surface clearance (>= 80.0 mm across all 21 samples):
   - Computes Euclidean distance from all distal robot collision hulls (`link4`, `link5`, `gripper`,
     `tip_left`, `tip_right` on BOTH arms, loaded from `YAM_URDF`) to ALL task surfaces:
     `check_a` (5 OBBs), `check_b` (5 OBBs), `check_rack` (6 OBBs), `out_tray` (5 OBBs),
     `pass_riser` (6 OBBs), `in_stand` (3 OBBs), AND all 31 dynamic glass spheres (`bead_00..30`).
   - Every positive distance must pass BOTH primal feasibility (<= 1e-6 m) AND a separating-hyperplane
     KKT dual lower-bound check (|d_primal - d_dual_sep| <= 1e-4 m), returning min(d_primal, d_dual_sep).
   - Any uncertified optimizer failure, non-finite iterate, primal violation, or KKT gap logs an entry
     in `solver_errors` and FAILS CLOSED by returning `0.0` clearance (never falling back to an upper-bound
     vertex/iterate distance).
2. Solid-material non-embedding & free-channel occupancy (Tier B & Tier A + pass-through guard):
   - Validates `check_b` and `check_a` against all 20 solid static OBBs (`rack`, `tray`, `riser`, `stand`)
     via exact 15-axis 3D SAT overlap (`<= SOLID_CONTACT_ALLOWANCE_M = 0.25 mm` across all 21 settled samples),
     AND verifies every seated `check_b` collider center lies strictly inside the open free channel cross-section
     (in front of `X_WALL_WORLD = 0.37320 m`, on the channel side of the 24.5 deg front ramp inner face, below
     the roof inner face, and between endcap inner faces), rejecting cards embedded in the rear wall or ramp.
   - Monitors every recorded step for midplane wall pass-through without latching zero on harmless contacts.
3. Observable strictly-under-1.0 mm (`< 1.0 mm`) 21-sample all-pairs stillness on visible cards:
   - Exact `max_{0 <= p < q < 21} ||x(p) - x(q)||_2 < 0.0010 m` across all 40 corners of `check_a`
     and all 40 corners of `check_b` (exact 1.0000 mm fails; rack, pan, riser and stand are fixed).
   - Escaped spheres outside the CK6518A channel permanently zero the trial (`sticky_bead_spill`),
     while sub-millimeter sphere creep hidden inside the opaque channel is logged as a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from hand_geometry import read_stl, rotation as _rpy_rot, vector as _urdf_vec
from yam_arena import YAM_URDF
from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS
from yam_arena.urdf import forward_kinematics

SETTLE_SAMPLES = 21  # 1.00 s at 20 Hz (20 intervals -> 21 consecutive samples)
HAND_SURFACE_CLEARANCE_M = 0.080      # >= 80.0 mm certified surface-to-surface clearance
CARD_CORNER_SETTLE_TOL_M = 0.0010     # strictly < 1.0 mm all-pairs corner excursion over 21 samples
BEAD_SETTLE_TOL_M = 0.0010            # strictly < 1.0 mm all-pairs sphere-center excursion over 21 samples
SOLID_CONTACT_ALLOWANCE_M = 0.00025   # <= 0.25 mm numerical contact compliance on settled solid interfaces

PRIMAL_FEAS_TOL_M = 1e-6
KKT_DUAL_GAP_TOL_M = 1e-4

HAND_CLEARANCE_LINKS = ("link4", "link5", "gripper", "tip_left", "tip_right")

_UNIT_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=float,
)


@dataclass(frozen=True)
class AuthoredColliderBox:
    """One authored box collider expressed in its rigid body's local frame."""

    label: str
    body_name: str
    local_center: np.ndarray   # (3,) in body-local frame (part_box_center - body_anchor)
    half_extents: np.ndarray   # (3,) half-size (hx, hy, hz)
    local_rot: np.ndarray      # (3, 3) Rz(yaw) @ Ry(pitch)
    volume_weight: float       # fraction of body's collider volume (sums to 1.0 per body)
    body_mass_kg: float        # mass assigned to owning rigid body


@dataclass(frozen=True)
class WorldOBB:
    label: str
    center: np.ndarray         # (3,)
    half_extents: np.ndarray   # (3,)
    rot: np.ndarray            # (3, 3) columns are world axes of box
    corners: np.ndarray        # (8, 3)


@dataclass(frozen=True)
class DistalHullSpec:
    link: str
    vertices_local: np.ndarray  # (V, 3)
    planes_local: np.ndarray    # (F, 4) outward n.x + d <= 0


@dataclass(frozen=True)
class ScorerSceneGeometry:
    top_z: float
    rack_x0: float
    rack_y0: float
    z_rack_bot: float
    rack_outer_h: float
    rack_outer_w: float
    rack_outer_d: float
    wall_t: float
    endcap_t: float
    z_apron_lip: float
    x_lip_inner: float
    ramp_deg: float
    bead_radius: float
    target_b_y_range: tuple[float, float]
    tray_inner_x_range: tuple[float, float]
    tray_inner_y_range: tuple[float, float]
    tray_floor_top_z: float
    tray_rim_top_z: float
    rack_obbs: tuple[WorldOBB, ...]
    tray_obbs: tuple[WorldOBB, ...] = ()
    riser_obbs: tuple[WorldOBB, ...] = ()
    in_stand_obbs: tuple[WorldOBB, ...] = ()
    check_a_colliders: tuple[AuthoredColliderBox, ...] = ()
    check_b_colliders: tuple[AuthoredColliderBox, ...] = ()

    @property
    def all_static_obbs(self) -> tuple[WorldOBB, ...]:
        return self.rack_obbs + self.tray_obbs + self.riser_obbs + self.in_stand_obbs


@dataclass(frozen=True)
class WorldSample:
    """Complete geometric snapshot at one 20 Hz control step."""

    t: float
    check_a_bodies: dict[str, tuple[np.ndarray, np.ndarray]]
    check_b_bodies: dict[str, tuple[np.ndarray, np.ndarray]]
    bead_centers: np.ndarray   # (N_beads, 3)
    right_q6: np.ndarray       # (6,)
    right_fingers: tuple[float, float]
    left_q6: np.ndarray        # (6,)
    left_fingers: tuple[float, float]


def quat_wxyz_to_mat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    if not (math.isfinite(qw) and math.isfinite(qx) and math.isfinite(qy) and math.isfinite(qz)):
        return np.full((3, 3), float("nan"), dtype=float)
    nrm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(nrm) or nrm <= 1e-12:
        return np.full((3, 3), float("nan"), dtype=float)
    qw, qx, qy, qz = qw / nrm, qx / nrm, qy / nrm, qz / nrm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _box_turn_rot(turn) -> np.ndarray:
    yaw, pitch = (float(turn[0]), float(turn[1])) if isinstance(turn, (tuple, list)) else (float(turn), 0.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    return Rz @ Ry


def obb_penetration_sat_m(
    c1: np.ndarray,
    h1: np.ndarray,
    R1: np.ndarray,
    c2: np.ndarray,
    h2: np.ndarray,
    R2: np.ndarray,
) -> float:
    """Exact 15-axis 3D SAT minimum-translation overlap depth (m) between two OBBs (0.0 if disjoint)."""
    if np.linalg.norm(c2 - c1) > float(np.linalg.norm(h1) + np.linalg.norm(h2)) + 1e-6:
        return 0.0
    d = c1 - c2
    axes = [R1[:, 0], R1[:, 1], R1[:, 2], R2[:, 0], R2[:, 1], R2[:, 2]]
    for i in range(3):
        ui = R1[:, i]
        for j in range(3):
            cr = np.cross(ui, R2[:, j])
            nrm = float(np.linalg.norm(cr))
            if nrm > 1e-8:
                axes.append(cr / nrm)
    min_ov = float("inf")
    for ax in axes:
        r1 = float(np.sum(h1 * np.abs(R1.T @ ax)))
        r2 = float(np.sum(h2 * np.abs(R2.T @ ax)))
        ov = r1 + r2 - abs(float(np.dot(d, ax)))
        if ov <= 0.0:
            return 0.0
        if ov < min_ov:
            min_ov = ov
    return float(min_ov)


def extract_authored_colliders(part_spec) -> tuple[AuthoredColliderBox, ...]:
    """Build exact body-local collider boxes directly from a labkit Part's authored shapes."""
    out: list[AuthoredColliderBox] = []
    root_vols = [float(b[1][0] * b[1][1] * b[1][2]) for b in part_spec.boxes]
    root_vsum = sum(root_vols) if root_vols else 1.0
    m_root = float(part_spec.mass or 0.0)
    for i, b in enumerate(part_spec.boxes):
        c_p = np.asarray(b[0], dtype=float)
        h = 0.5 * np.asarray(b[1], dtype=float)
        R_l = _box_turn_rot(b[3] if len(b) > 3 else 0.0)
        out.append(
            AuthoredColliderBox(
                label=f"body/box_{i}",
                body_name="body",
                local_center=c_p,
                half_extents=h,
                local_rot=R_l,
                volume_weight=root_vols[i] / root_vsum,
                body_mass_kg=m_root,
            )
        )
    for j in part_spec.joints:
        anc = np.asarray(j.anchor, dtype=float)
        j_vols = [float(b[1][0] * b[1][1] * b[1][2]) for b in j.boxes]
        j_vsum = sum(j_vols) if j_vols else 1.0
        for i, b in enumerate(j.boxes):
            c_p = np.asarray(b[0], dtype=float)
            h = 0.5 * np.asarray(b[1], dtype=float)
            R_l = _box_turn_rot(b[3] if len(b) > 3 else 0.0)
            out.append(
                AuthoredColliderBox(
                    label=f"{j.name}/box_{i}",
                    body_name=j.name,
                    local_center=c_p - anc,
                    half_extents=h,
                    local_rot=R_l,
                    volume_weight=j_vols[i] / j_vsum,
                    body_mass_kg=float(j.mass),
                )
            )
    return tuple(out)


def static_part_world_obbs(part_spec, names: tuple[str, ...] | None = None) -> tuple[WorldOBB, ...]:
    p0 = np.asarray(part_spec.pos, dtype=float)
    cy, sy = math.cos(part_spec.yaw), math.sin(part_spec.yaw)
    cp, sp = math.cos(part_spec.pitch), math.sin(part_spec.pitch)
    cr, sr = math.cos(part_spec.roll), math.sin(part_spec.roll)
    R_part = (
        np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    )
    out: list[WorldOBB] = []
    for i, b in enumerate(part_spec.boxes):
        c_l = np.asarray(b[0], dtype=float)
        h = 0.5 * np.asarray(b[1], dtype=float)
        R_b = R_part @ _box_turn_rot(b[3] if len(b) > 3 else 0.0)
        c_w = p0 + R_part @ c_l
        corners = (_UNIT_CORNERS * h) @ R_b.T + c_w
        lbl = names[i] if (names is not None and i < len(names)) else f"{part_spec.name}/box_{i}"
        out.append(WorldOBB(label=lbl, center=c_w, half_extents=h, rot=R_b, corners=corners))
    return tuple(out)


def card_world_obbs_and_com(
    colliders: tuple[AuthoredColliderBox, ...],
    body_poses: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[WorldOBB], np.ndarray, np.ndarray]:
    """Return (world_obbs, all_corners (N*8, 3), mass_weighted_com (3,)) for an articulated card."""
    obbs: list[WorldOBB] = []
    corners_all: list[np.ndarray] = []
    weighted_sum = np.zeros(3, dtype=float)
    mass_sum = 0.0
    for col in colliders:
        pos_b, quat_b = body_poses[col.body_name]
        R_b = quat_wxyz_to_mat(*[float(v) for v in quat_b])
        c_w = np.asarray(pos_b, dtype=float) + R_b @ col.local_center
        R_w = R_b @ col.local_rot
        pts = (_UNIT_CORNERS * col.half_extents) @ R_w.T + c_w
        obbs.append(WorldOBB(label=col.label, center=c_w, half_extents=col.half_extents, rot=R_w, corners=pts))
        corners_all.append(pts)
        m_piece = col.body_mass_kg * col.volume_weight
        weighted_sum += m_piece * c_w
        mass_sum += m_piece
    com_w = weighted_sum / max(mass_sum, 1e-12)
    return obbs, np.vstack(corners_all), com_w


def load_distal_arm_hulls(urdf_path: str | pathlib.Path = YAM_URDF) -> tuple[DistalHullSpec, ...]:
    urdf = pathlib.Path(urdf_path).resolve()
    root = ET.parse(urdf).getroot()
    out: list[DistalHullSpec] = []
    for link in root.findall("link"):
        lname = link.attrib.get("name", "")
        if lname not in HAND_CLEARANCE_LINKS:
            continue
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            stl_path = (urdf.parent / mesh.get("filename")).resolve()
            origin = col.find("origin")
            r = _rpy_rot(_urdf_vec(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]))
            p = _urdf_vec(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
            sc = _urdf_vec(mesh.get("scale"), [1.0, 1.0, 1.0])
            v = (read_stl(stl_path) * sc) @ r.T + p
            v_u = np.unique(np.round(v.reshape(-1, 3), 7), axis=0)
            hull = ConvexHull(v_u)
            verts = v_u[hull.vertices]
            eq = hull.equations.copy()
            eq[:, 3] = -np.max(verts @ eq[:, :3].T, axis=0)
            out.append(DistalHullSpec(link=lname, vertices_local=verts, planes_local=eq))
    if {h.link for h in out} != set(HAND_CLEARANCE_LINKS):
        raise ValueError(f"Missing distal hand clearance hulls; got { {h.link for h in out} }")
    return tuple(out)


def convex_hull_to_obb_distance_m(
    verts_w: np.ndarray,
    planes_local: np.ndarray,
    R_hull_w: np.ndarray,
    p_hull_w: np.ndarray,
    obb: WorldOBB,
    cutoff_m: float | None = None,
    solver_errors: list[str] | None = None,
) -> float:
    """Certified Euclidean surface-to-surface distance between a convex hull and an OBB.

    Fails closed (returns 0.0 and records an entry in `solver_errors`) whenever optimization does not
    converge OR fails primal feasibility (`<= 1e-6 m`) OR fails the separating-hyperplane KKT dual
    lower-bound check (`|d_primal - d_dual_sep| <= 1e-4 m`). Never returns an uncertified vertex or
    iterate distance.
    """
    def _fail_closed(msg: str) -> float:
        if solver_errors is not None:
            solver_errors.append(f"obb({obb.label}): {msg}")
        return 0.0

    if not (
        np.all(np.isfinite(verts_w))
        and np.all(np.isfinite(planes_local))
        and np.all(np.isfinite(R_hull_w))
        and np.all(np.isfinite(p_hull_w))
        and np.all(np.isfinite(obb.center))
        and np.all(np.isfinite(obb.half_extents))
        and np.all(np.isfinite(obb.rot))
    ):
        return _fail_closed("non_finite_input")

    # Express hull vertices in OBB-local coordinates (where OBB is [-h, +h]):
    u_verts = (verts_w - obb.center) @ obb.rot
    h = obb.half_extents
    lo_u = u_verts.min(axis=0)
    hi_u = u_verts.max(axis=0)
    gap_aabb = np.maximum(0.0, np.maximum(lo_u - h, -h - hi_u))
    d_aabb_lower = float(np.linalg.norm(gap_aabb))
    # Safe branch-and-bound pruning: true_dist >= d_aabb_lower is an unconditional algebraic bound.
    if cutoff_m is not None and d_aabb_lower >= cutoff_m:
        return d_aabb_lower

    # Express hull half-spaces n_l . x_l + d_l <= 0 in OBB-local coordinates u:
    M = R_hull_w.T @ obb.rot
    shift = R_hull_w.T @ (obb.center - p_hull_w)
    N_u = planes_local[:, :3] @ M
    b_u = -(planes_local[:, 3] + planes_local[:, :3] @ shift)  # N_u @ u <= b_u

    # Verify supplied vertices are consistent with supplied planes:
    if float(np.max(u_verts @ N_u.T - b_u[None, :])) > 1e-3:
        return _fail_closed("inconsistent_hull_vertices_vs_planes")

    # Construct feasible initial point inside hull (centroid of vertices, projected if needed) & clamped OBB point:
    u_init = u_verts.mean(axis=0)
    clamped_v = np.clip(u_verts, -h, +h)
    idx0 = int(np.argmin(np.linalg.norm(u_verts - clamped_v, axis=1)))
    if float(np.linalg.norm(u_verts[idx0] - clamped_v[idx0])) <= 1e-9:
        return 0.0
    # Blend toward closest vertex while staying strictly interior:
    u0 = 0.1 * u_init + 0.9 * u_verts[idx0]
    w0 = np.clip(u0, -h, +h)
    x0 = np.concatenate([u0, w0])

    def obj(z: np.ndarray) -> tuple[float, np.ndarray]:
        diff = z[:3] - z[3:]
        return 0.5 * float(np.dot(diff, diff)), np.concatenate([diff, -diff])

    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (-float(h[0]), float(h[0])),
        (-float(h[1]), float(h[1])),
        (-float(h[2]), float(h[2])),
    ]
    A_ineq = np.hstack([N_u, np.zeros_like(N_u)])
    cons = {"type": "ineq", "fun": lambda z: b_u - A_ineq @ z, "jac": lambda z: -A_ineq}

    try:
        res = minimize(
            obj,
            x0,
            jac=True,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 100},
        )
    except Exception as exc:
        return _fail_closed(f"qp_exception:{type(exc).__name__}")

    if not getattr(res, "success", False) or not hasattr(res, "x") or len(res.x) != 6 or not np.all(np.isfinite(res.x)):
        return _fail_closed(f"qp_failed:{getattr(res, 'message', 'unconverged')}")

    u_opt = np.asarray(res.x[:3], dtype=float)
    w_opt = np.asarray(res.x[3:], dtype=float)

    # 1. Verify primal feasibility explicitly:
    viol_hull = float(np.max(N_u @ u_opt - b_u))
    viol_box = float(np.max(np.abs(w_opt) - h))
    if viol_hull > PRIMAL_FEAS_TOL_M or viol_box > PRIMAL_FEAS_TOL_M:
        return _fail_closed(f"primal_infeasible:hull={viol_hull:.2e},box={viol_box:.2e}")

    s = u_opt - w_opt
    d_primal = float(np.linalg.norm(s))
    if d_primal <= 1e-7:
        return 0.0

    # 2. Verify separating-hyperplane KKT dual lower bound:
    #    For unit vector n_hat = (u_opt - w_opt) / ||u_opt - w_opt||_2, EVERY u in conv(u_verts) and
    #    EVERY w in [-h, +h] satisfies ||u - w||_2 >= n_hat . (u - w) >= min_v(n_hat . v) - sum_i h_i |n_hat_i|.
    n_hat = s / d_primal
    sup_box = float(np.sum(h * np.abs(n_hat)))
    inf_hull_verts = float(np.min(u_verts @ n_hat))
    d_dual_sep = inf_hull_verts - sup_box
    kkt_gap = abs(d_primal - d_dual_sep)
    if kkt_gap > KKT_DUAL_GAP_TOL_M:
        return _fail_closed(f"kkt_duality_gap:primal={d_primal:.6f},dual={d_dual_sep:.6f},gap={kkt_gap:.6f}")

    d_certified = max(0.0, min(d_primal, d_dual_sep))
    return float(d_certified)


def convex_hull_to_sphere_distance_m(
    verts_w: np.ndarray,
    planes_local: np.ndarray,
    R_hull_w: np.ndarray,
    p_hull_w: np.ndarray,
    sph_center: np.ndarray,
    sph_radius: float,
    cutoff_m: float | None = None,
    solver_errors: list[str] | None = None,
    label: str = "bead",
) -> float:
    """Certified Euclidean surface-to-surface distance between a convex hull and a sphere (fail-closed -> 0.0)."""
    def _fail_closed(msg: str) -> float:
        if solver_errors is not None:
            solver_errors.append(f"sphere({label}): {msg}")
        return 0.0

    sph_center = np.asarray(sph_center, dtype=float)
    if sph_center.shape != (3,) or not (
        np.all(np.isfinite(verts_w))
        and np.all(np.isfinite(planes_local))
        and np.all(np.isfinite(R_hull_w))
        and np.all(np.isfinite(p_hull_w))
        and np.all(np.isfinite(sph_center))
        and math.isfinite(float(sph_radius))
        and float(sph_radius) >= 0.0
    ):
        return _fail_closed("non_finite_or_invalid_sphere_input")
    c_l = R_hull_w.T @ (sph_center - p_hull_w)
    lo_l = verts_w.min(axis=0)
    hi_l = verts_w.max(axis=0)
    d_world_aabb = max(0.0, float(np.linalg.norm(sph_center - np.clip(sph_center, lo_l, hi_l))) - sph_radius)
    if cutoff_m is not None and d_world_aabb >= cutoff_m:
        return d_world_aabb

    N_l = planes_local[:, :3]
    b_l = -planes_local[:, 3]
    verts_l = (verts_w - p_hull_w) @ R_hull_w

    idx0 = int(np.argmin(np.linalg.norm(verts_l - c_l[None, :], axis=1)))
    u0 = 0.1 * verts_l.mean(axis=0) + 0.9 * verts_l[idx0]

    def obj(u: np.ndarray) -> tuple[float, np.ndarray]:
        diff = u - c_l
        return 0.5 * float(np.dot(diff, diff)), diff

    cons = {"type": "ineq", "fun": lambda u: b_l - N_l @ u, "jac": lambda u: -N_l}
    try:
        res = minimize(obj, u0, jac=True, constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 80})
    except Exception as exc:
        return _fail_closed(f"qp_exception:{type(exc).__name__}")

    if not getattr(res, "success", False) or not hasattr(res, "x") or len(res.x) != 3 or not np.all(np.isfinite(res.x)):
        return _fail_closed(f"qp_failed:{getattr(res, 'message', 'unconverged')}")

    u_opt = np.asarray(res.x, dtype=float)
    if float(np.max(N_l @ u_opt - b_l)) > PRIMAL_FEAS_TOL_M:
        return _fail_closed("primal_infeasible")

    s = u_opt - c_l
    d_primal = float(np.linalg.norm(s))
    if d_primal <= sph_radius + 1e-7:
        return 0.0

    n_hat = s / d_primal
    d_dual = float(np.min((verts_l - c_l[None, :]) @ n_hat))
    if abs(d_primal - d_dual) > KKT_DUAL_GAP_TOL_M:
        return _fail_closed(f"kkt_duality_gap:{abs(d_primal - d_dual):.6f}")

    return float(max(0.0, min(d_primal, d_dual) - sph_radius))


def _card_free_channel_and_solid_ok(
    card_obbs: list[WorldOBB],
    card_corners: np.ndarray,
    g: ScorerSceneGeometry,
) -> tuple[bool, float, str]:
    """Check that seated Check B occupies free channel space (not embedded in rear wall, front ramp, roof, or endcaps)."""
    max_solid_pen_m = 0.0
    worst_solid_pair = ""
    for co in card_obbs:
        for so in g.all_static_obbs:
            pen_m = obb_penetration_sat_m(co.center, co.half_extents, co.rot, so.center, so.half_extents, so.rot)
            if pen_m > max_solid_pen_m:
                max_solid_pen_m = pen_m
                worst_solid_pair = f"{co.label}<->{so.label}"

    if max_solid_pen_m > SOLID_CONTACT_ALLOWANCE_M:
        return False, max_solid_pen_m, f"solid_pen_exceeded({worst_solid_pair}:{max_solid_pen_m*1000:.3f}mm)"

    x_wall_inner = g.rack_x0 - g.wall_t
    x_fascia_inner = g.rack_x0 - g.rack_outer_d + g.wall_t
    z_roof_inner = g.z_rack_bot + g.rack_outer_h - g.wall_t
    y_clear_half = 0.5 * g.rack_outer_w - g.endcap_t
    x_lip_w = g.rack_x0 + g.x_lip_inner
    z_lip_w = g.z_rack_bot + g.z_apron_lip
    alpha = math.radians(g.ramp_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)

    has_in_rack_collider = False
    for co in card_obbs:
        c_z_max = float(co.corners[:, 2].max())
        if c_z_max < g.z_rack_bot:
            continue
        has_in_rack_collider = True
        cx, cy, cz = float(co.center[0]), float(co.center[1]), float(co.center[2])
        # If the collider's center is inside the rack vertical span, its center MUST lie in open free channel space:
        if cz >= g.z_rack_bot:
            if not (x_fascia_inner < cx < x_wall_inner):
                return False, max_solid_pen_m, f"center_not_in_channel_x({co.label}:cx={cx:.5f})"
            if not (cz < z_roof_inner):
                return False, max_solid_pen_m, f"center_above_roof_inner({co.label}:cz={cz:.5f})"
            if not (abs(cy - g.rack_y0) < y_clear_half):
                return False, max_solid_pen_m, f"center_in_endcap({co.label}:cy={cy:.5f})"
            if cz >= z_lip_w:
                # Signed inward distance from the 24.5 deg front ramp's inner check-facing plane:
                d_ramp_in = (cx - x_lip_w) * ca + (cz - z_lip_w) * sa
                if d_ramp_in <= 0.0:
                    return False, max_solid_pen_m, f"center_inside_or_past_front_ramp({co.label}:d={d_ramp_in*1000:.3f}mm)"

        # Also check every vertex of co inside the rack vertical span against wall midplanes & free-side requirement:
        in_z = co.corners[:, 2] >= g.z_rack_bot
        pts_in = co.corners[in_z]
        # Free-channel (-X) face must lie strictly in front of x_wall_inner (prevents thin card sitting wholly inside [0.37320, 0.37345]):
        if float(pts_in[:, 0].min()) >= x_wall_inner:
            return False, max_solid_pen_m, f"wholly_behind_wall_inner_face({co.label})"
        # Deepest (+X) corner cannot exceed x_wall_inner + SOLID_CONTACT_ALLOWANCE_M:
        if float(pts_in[:, 0].max()) > x_wall_inner + SOLID_CONTACT_ALLOWANCE_M:
            return False, max_solid_pen_m, f"corner_embedded_in_rear_wall({co.label})"
        # Vertices above z_lip_w cannot penetrate past SOLID_CONTACT_ALLOWANCE_M into the angled front ramp:
        above_lip = pts_in[pts_in[:, 2] >= z_lip_w]
        if len(above_lip) > 0:
            d_pts_ramp = (above_lip[:, 0] - x_lip_w) * ca + (above_lip[:, 2] - z_lip_w) * sa
            if float(d_pts_ramp.max()) <= 0.0:
                return False, max_solid_pen_m, f"wholly_inside_front_ramp({co.label})"
            if float(d_pts_ramp.min()) < -SOLID_CONTACT_ALLOWANCE_M:
                return False, max_solid_pen_m, f"corner_embedded_in_front_ramp({co.label}:{float(d_pts_ramp.min())*1000:.3f}mm)"

    if not has_in_rack_collider:
        return False, max_solid_pen_m, "no_collider_in_rack"

    return True, max_solid_pen_m, "ok"


class GoalScorer:
    """Stateful trial monitor + 21-sample rolling goal-only evaluator."""

    def __init__(self, scene_geom: ScorerSceneGeometry, urdf_path: str | pathlib.Path = YAM_URDF):
        self.geom = scene_geom
        self.urdf_path = pathlib.Path(urdf_path).resolve()
        self.distal_hulls = load_distal_arm_hulls(self.urdf_path)
        self.samples: list[WorldSample] = []
        self.sticky_bead_spill: bool = False
        self.sticky_spill_reason: str = ""
        self.sticky_wall_passthrough: bool = False
        self.sticky_passthrough_reason: str = ""
        self._clearance_memo: dict[tuple, tuple[float, str, tuple[str, ...]]] = {}

    def reset(self) -> None:
        self.samples.clear()
        self.sticky_bead_spill = False
        self.sticky_spill_reason = ""
        self.sticky_wall_passthrough = False
        self.sticky_passthrough_reason = ""
        self._clearance_memo.clear()

    def _check_bead_integrity(self, bead_centers: np.ndarray, t: float) -> None:
        if self.sticky_bead_spill:
            return
        bead_centers = np.asarray(bead_centers, dtype=float)
        if bead_centers.ndim != 2 or bead_centers.shape[1] != 3 or len(bead_centers) == 0 or not np.all(np.isfinite(bead_centers)):
            self.sticky_bead_spill = True
            self.sticky_spill_reason = f"non_finite_or_invalid_beads at t={t:.2f}s"
            return
        g = self.geom
        x_lo = g.rack_x0 - g.rack_outer_d - 0.0020
        x_hi = g.rack_x0 + 0.0020
        y_lo = g.rack_y0 - 0.5 * g.rack_outer_w
        y_hi = g.rack_y0 + 0.5 * g.rack_outer_w
        z_lo = g.z_rack_bot + 0.0040
        z_hi = g.z_rack_bot + g.rack_outer_h + 0.0020
        bad = (
            (bead_centers[:, 0] < x_lo)
            | (bead_centers[:, 0] > x_hi)
            | (bead_centers[:, 1] < y_lo)
            | (bead_centers[:, 1] > y_hi)
            | (bead_centers[:, 2] < z_lo)
            | (bead_centers[:, 2] > z_hi)
        )
        if np.any(bad):
            idx = int(np.argmax(bad))
            self.sticky_bead_spill = True
            self.sticky_spill_reason = f"bead_{idx:02d} left CK6518A channel at t={t:.2f}s: {np.round(bead_centers[idx], 4).tolist()}"

    def _check_wall_passthrough(self, sample: WorldSample) -> None:
        """Flag genuine unphysical tunneling past the midplane of solid rack walls without latching harmless contacts."""
        if self.sticky_wall_passthrough:
            return
        g = self.geom
        x_rear_mid = g.rack_x0 - 0.5 * g.wall_t
        x_lip_w = g.rack_x0 + g.x_lip_inner
        z_lip_w = g.z_rack_bot + g.z_apron_lip
        alpha = math.radians(g.ramp_deg)
        ca, sa = math.cos(alpha), math.sin(alpha)

        for cname, cols, poses in (
            ("check_a", g.check_a_colliders, sample.check_a_bodies),
            ("check_b", g.check_b_colliders, sample.check_b_bodies),
        ):
            if not cols:
                continue
            try:
                obbs, pts_chk, com_chk = card_world_obbs_and_com(cols, poses)
            except Exception:
                self.sticky_wall_passthrough = True
                self.sticky_passthrough_reason = f"{cname} invalid_body_poses at t={sample.t:.2f}s"
                return
            if not (np.all(np.isfinite(pts_chk)) and np.all(np.isfinite(com_chk))):
                self.sticky_wall_passthrough = True
                self.sticky_passthrough_reason = f"{cname} non_finite_card_pose at t={sample.t:.2f}s"
                return
            for co in obbs:
                cx, cy, cz = float(co.center[0]), float(co.center[1]), float(co.center[2])
                if (g.z_rack_bot <= cz <= g.z_rack_bot + g.rack_outer_h) and (abs(cy - g.rack_y0) <= 0.5 * g.rack_outer_w):
                    # Center past rear wall midplane while inside rack footprint [rack_x0 - rack_outer_d, rack_x0 + wall_t]:
                    if x_rear_mid < cx <= g.rack_x0 + g.wall_t:
                        self.sticky_wall_passthrough = True
                        self.sticky_passthrough_reason = f"{cname}/{co.label} center crossed rear-wall midplane at t={sample.t:.2f}s (cx={cx:.5f})"
                        return
                    if cz >= z_lip_w + 0.0020 and cx >= g.rack_x0 - g.rack_outer_d - g.wall_t:
                        d_ramp_in = (cx - x_lip_w) * ca + (cz - z_lip_w) * sa
                        if -g.wall_t <= d_ramp_in < -0.5 * g.wall_t:
                            self.sticky_wall_passthrough = True
                            self.sticky_passthrough_reason = f"{cname}/{co.label} center crossed front-ramp midplane at t={sample.t:.2f}s"
                            return

    def record(self, sample: WorldSample) -> None:
        self._check_bead_integrity(sample.bead_centers, sample.t)
        self._check_wall_passthrough(sample)
        self.samples.append(sample)
        if len(self.samples) > SETTLE_SAMPLES:
            self.samples = self.samples[-SETTLE_SAMPLES:]

    def min_hand_surface_clearance_m(
        self,
        sample: WorldSample,
        target_obbs: list[WorldOBB],
        solver_errors: list[str] | None = None,
    ) -> tuple[float, str]:
        beads_arr = np.asarray(sample.bead_centers, dtype=float)
        if beads_arr.ndim != 2 or beads_arr.shape[1] != 3 or len(beads_arr) == 0 or not np.all(np.isfinite(beads_arr)):
            if solver_errors is not None:
                solver_errors.append("beads:invalid_or_non_finite")
            return 0.0, "invalid_beads"

        best_d = float("inf")
        best_pair = ""
        r_bead = self.geom.bead_radius
        arm_specs = (
            ("right", np.asarray(RIGHT_BASE_POS, float), sample.right_q6, sample.right_fingers),
            ("left", np.asarray(LEFT_BASE_POS, float), sample.left_q6, sample.left_fingers),
        )
        for side, base_p, q6, fingers in arm_specs:
            try:
                q6_arr = np.asarray(q6, dtype=float)
                f_arr = np.asarray(fingers, dtype=float)
            except Exception:
                if solver_errors is not None:
                    solver_errors.append(f"{side}:non_numeric_joints")
                return 0.0, f"{side}/invalid_joints"
            if (
                q6_arr.shape != (6,)
                or f_arr.shape != (2,)
                or not np.all(np.isfinite(base_p))
                or not np.all(np.isfinite(q6_arr))
                or not np.all(np.isfinite(f_arr))
            ):
                if solver_errors is not None:
                    solver_errors.append(f"{side}:invalid_or_non_finite_joints")
                return 0.0, f"{side}/invalid_joints"
            jmap = {f"joint{k + 1}": float(q6_arr[k]) for k in range(6)}
            jmap["joint7"] = float(f_arr[0])
            jmap["joint8"] = float(f_arr[1])
            fk = forward_kinematics(self.urdf_path, jmap)
            for dh in self.distal_hulls:
                tf = fk[dh.link]
                R_w = tf[:3, :3]
                p_w = base_p + tf[:3, 3]
                if not (np.all(np.isfinite(R_w)) and np.all(np.isfinite(p_w))):
                    if solver_errors is not None:
                        solver_errors.append(f"{side}/{dh.link}:non_finite_fk")
                    return 0.0, f"{side}/{dh.link}"
                verts_w = dh.vertices_local @ R_w.T + p_w
                for obb in target_obbs:
                    d = convex_hull_to_obb_distance_m(
                        verts_w,
                        dh.planes_local,
                        R_w,
                        p_w,
                        obb,
                        cutoff_m=best_d,
                        solver_errors=solver_errors,
                    )
                    if not math.isfinite(d):
                        if solver_errors is not None:
                            solver_errors.append(f"{side}/{dh.link}<->{obb.label}:non_finite_dist")
                        return 0.0, f"{side}/{dh.link} <-> {obb.label}"
                    if d < best_d:
                        best_d = d
                        best_pair = f"{side}/{dh.link} <-> {obb.label}"
                        if best_d <= 0.0:
                            return 0.0, best_pair
                for bi in range(len(beads_arr)):
                    d_b = convex_hull_to_sphere_distance_m(
                        verts_w,
                        dh.planes_local,
                        R_w,
                        p_w,
                        beads_arr[bi],
                        r_bead,
                        cutoff_m=best_d,
                        solver_errors=solver_errors,
                        label=f"bead_{bi:02d}",
                    )
                    if not math.isfinite(d_b):
                        if solver_errors is not None:
                            solver_errors.append(f"{side}/{dh.link}<->bead_{bi:02d}:non_finite_dist")
                        return 0.0, f"{side}/{dh.link} <-> bead_{bi:02d}"
                    if d_b < best_d:
                        best_d = d_b
                        best_pair = f"{side}/{dh.link} <-> bead_{bi:02d}"
                        if best_d <= 0.0:
                            return 0.0, best_pair
        if not math.isfinite(best_d):
            if solver_errors is not None:
                solver_errors.append("non_finite_best_d")
            return 0.0, "non_finite_best_d"
        return best_d, best_pair

    def evaluate(self, skip_clearance_if_zero: bool = False) -> dict:
        """Evaluate goal-only score (0.0, 0.5, or 1.0) with full diagnostic breakdown."""
        if self.sticky_bead_spill:
            return {"score": 0.0, "reason": f"sticky_violation: {self.sticky_spill_reason}"}
        if self.sticky_wall_passthrough:
            return {"score": 0.0, "reason": f"sticky_violation: {self.sticky_passthrough_reason}"}
        if len(self.samples) < SETTLE_SAMPLES:
            return {"score": 0.0, "reason": f"insufficient_samples ({len(self.samples)}/{SETTLE_SAMPLES})"}

        win = self.samples[-SETTLE_SAMPLES:]
        g = self.geom
        static_obbs_list = list(g.all_static_obbs)

        ca_corners_seq = []
        cb_corners_seq = []
        beads_seq = []
        per_sample_obbs: list[tuple[WorldSample, list[WorldOBB], np.ndarray, list[WorldOBB], np.ndarray, np.ndarray]] = []

        max_ca_solid_pen_21_m = 0.0
        max_cb_solid_pen_21_m = 0.0
        cb_channel_21_ok = True
        cb_channel_fail_reason = "ok"
        ca_tray_solid_21_ok = True

        final_ca_obbs: list[WorldOBB] = []
        final_ca_corners = np.zeros((0, 3))
        final_ca_com = np.zeros(3)
        final_cb_obbs: list[WorldOBB] = []
        final_cb_corners = np.zeros((0, 3))
        final_cb_com = np.zeros(3)

        expected_beads = len(win[0].bead_centers) if hasattr(win[0].bead_centers, "__len__") else -1
        if expected_beads <= 0:
            return {"score": 0.0, "reason": "invalid_empty_beads"}

        for idx, smp in enumerate(win):
            try:
                ca_obbs, ca_pts, ca_com = card_world_obbs_and_com(g.check_a_colliders, smp.check_a_bodies)
                cb_obbs, cb_pts, cb_com = card_world_obbs_and_com(g.check_b_colliders, smp.check_b_bodies)
                beads_arr = np.asarray(smp.bead_centers, dtype=float)
            except Exception as exc:
                return {"score": 0.0, "reason": f"invalid_sample_geometry:{type(exc).__name__}"}

            if (
                beads_arr.shape != (expected_beads, 3)
                or not np.all(np.isfinite(beads_arr))
                or not np.all(np.isfinite(ca_pts))
                or not np.all(np.isfinite(cb_pts))
                or not np.all(np.isfinite(ca_com))
                or not np.all(np.isfinite(cb_com))
            ):
                return {"score": 0.0, "reason": f"non_finite_sample_geometry@t={smp.t:.2f}s"}

            ca_corners_seq.append(ca_pts)
            cb_corners_seq.append(cb_pts)
            beads_seq.append(beads_arr)
            per_sample_obbs.append((smp, ca_obbs, ca_pts, cb_obbs, cb_pts, beads_arr))

            if idx == len(win) - 1:
                final_ca_obbs, final_ca_corners, final_ca_com = ca_obbs, ca_pts, ca_com
                final_cb_obbs, final_cb_corners, final_cb_com = cb_obbs, cb_pts, cb_com

        ca_stack = np.stack(ca_corners_seq, axis=0)   # (21, 40, 3)
        cb_stack = np.stack(cb_corners_seq, axis=0)   # (21, 40, 3)
        bd_stack = np.stack(beads_seq, axis=0)        # (21, 31, 3)

        ca_allpairs_m = float(np.max(np.linalg.norm(ca_stack[:, None, :, :] - ca_stack[None, :, :, :], axis=-1)))
        cb_allpairs_m = float(np.max(np.linalg.norm(cb_stack[:, None, :, :] - cb_stack[None, :, :, :], axis=-1)))
        bd_allpairs_m = float(np.max(np.linalg.norm(bd_stack[:, None, :, :] - bd_stack[None, :, :, :], axis=-1)))

        # Observable stillness: strictly < 1.0 mm (< 0.0010 m) across all 80 corners of both visible cards
        # over the final 21 samples (the rack, pan, riser and stand are fixed apparatus). Internal sphere creep
        # inside the opaque channel is recorded below as an unscored diagnostic (while escaped spheres latch 0).
        all_settled = bool(
            math.isfinite(ca_allpairs_m)
            and math.isfinite(cb_allpairs_m)
            and (ca_allpairs_m < CARD_CORNER_SETTLE_TOL_M)
            and (cb_allpairs_m < CARD_CORNER_SETTLE_TOL_M)
        )
        beads_settled_diag = bool(math.isfinite(bd_allpairs_m) and (bd_allpairs_m < BEAD_SETTLE_TOL_M))

        # Necessary final-sample geometric goal bounds (O(1) array checks, evaluated before 4,200 SAT / 12,810 QP calls):
        ca_out_of_rack = bool(np.all(final_ca_corners[:, 2] <= g.z_rack_bot - 0.050))
        ca_all_corners_in_tray_xy = bool(
            np.all(final_ca_corners[:, 0] >= g.tray_inner_x_range[0])
            and np.all(final_ca_corners[:, 0] <= g.tray_inner_x_range[1])
            and np.all(final_ca_corners[:, 1] >= g.tray_inner_y_range[0])
            and np.all(final_ca_corners[:, 1] <= g.tray_inner_y_range[1])
        )
        ca_all_corners_in_tray_z = bool(
            np.all(final_ca_corners[:, 2] >= g.tray_floor_top_z - SOLID_CONTACT_ALLOWANCE_M)
            and np.all(final_ca_corners[:, 2] <= g.tray_rim_top_z + 0.0100)
            and (float(final_ca_corners[:, 2].max()) > g.tray_floor_top_z)
        )
        ca_com_supported = bool(
            (g.tray_inner_x_range[0] < float(final_ca_com[0]) < g.tray_inner_x_range[1])
            and (g.tray_inner_y_range[0] < float(final_ca_com[1]) < g.tray_inner_y_range[1])
            and (g.tray_floor_top_z < float(final_ca_com[2]) <= g.tray_rim_top_z)
        )
        ca_pre_ok = bool(ca_out_of_rack and ca_all_corners_in_tray_xy and ca_all_corners_in_tray_z and ca_com_supported)

        _, q_b_root = win[-1].check_b_bodies["body"]
        R_b_root = quat_wxyz_to_mat(*[float(v) for v in q_b_root])
        if not np.all(np.isfinite(R_b_root)):
            return {"score": 0.0, "reason": "non_finite_check_b_root_quat"}
        cb_upright_deg = math.degrees(math.acos(float(np.clip(R_b_root[2, 2], -1.0, 1.0))))
        cb_z_top = float(np.max(final_cb_corners[:, 2]))
        cb_z_bot = float(np.min(final_cb_corners[:, 2]))
        cb_zone_y_ok = bool(
            (g.target_b_y_range[0] <= float(final_cb_com[1]) <= g.target_b_y_range[1])
            and np.all(final_cb_corners[:, 1] >= g.rack_y0 - 0.5 * g.rack_outer_w + g.endcap_t - SOLID_CONTACT_ALLOWANCE_M)
            and np.all(final_cb_corners[:, 1] <= g.rack_y0 + 0.5 * g.rack_outer_w - g.endcap_t + SOLID_CONTACT_ALLOWANCE_M)
        )
        cb_seated_z_ok = bool(g.z_rack_bot + 0.0250 <= cb_z_top <= g.z_rack_bot + g.rack_outer_h - g.wall_t + SOLID_CONTACT_ALLOWANCE_M)
        cb_suspended_ok = bool(g.top_z + 0.0500 <= cb_z_bot <= g.z_rack_bot - 0.1000)
        cb_upright_ok = bool(math.isfinite(cb_upright_deg) and cb_upright_deg <= 15.0)
        cb_pre_ok = bool(cb_zone_y_ok and cb_seated_z_ok and cb_suspended_ok and cb_upright_ok)

        if skip_clearance_if_zero and (not all_settled or (not ca_pre_ok and not cb_pre_ok)):
            return {
                "score": 0.0,
                "raw_goal_tier_before_gates": 0.0,
                "fast_pregate_zero": True,
            }

        for smp, ca_obbs, ca_pts, cb_obbs, cb_pts, _beads_arr in per_sample_obbs:
            for co in ca_obbs:
                if not (
                    g.tray_inner_x_range[0] < float(co.center[0]) < g.tray_inner_x_range[1]
                    and g.tray_inner_y_range[0] < float(co.center[1]) < g.tray_inner_y_range[1]
                    and float(co.center[2]) > g.tray_floor_top_z
                ):
                    ca_tray_solid_21_ok = False
                for so in static_obbs_list:
                    max_ca_solid_pen_21_m = max(
                        max_ca_solid_pen_21_m,
                        obb_penetration_sat_m(co.center, co.half_extents, co.rot, so.center, so.half_extents, so.rot),
                    )
            if max_ca_solid_pen_21_m > SOLID_CONTACT_ALLOWANCE_M:
                ca_tray_solid_21_ok = False

            ch_ok, cb_pen_k, ch_reason = _card_free_channel_and_solid_ok(cb_obbs, cb_pts, g)
            max_cb_solid_pen_21_m = max(max_cb_solid_pen_21_m, cb_pen_k)
            if not ch_ok and cb_channel_21_ok:
                cb_channel_21_ok = False
                cb_channel_fail_reason = f"{ch_reason}@t={smp.t:.2f}s"

        no_solid_embedding_any = bool(
            math.isfinite(max_ca_solid_pen_21_m)
            and math.isfinite(max_cb_solid_pen_21_m)
            and (max_ca_solid_pen_21_m <= SOLID_CONTACT_ALLOWANCE_M)
            and (max_cb_solid_pen_21_m <= SOLID_CONTACT_ALLOWANCE_M)
        )

        tier_a_tray = bool(ca_pre_ok and ca_tray_solid_21_ok)
        tier_b_rack = bool(cb_pre_ok and cb_channel_21_ok)

        raw_tier = (0.5 if tier_a_tray else 0.0) + (0.5 if tier_b_rack else 0.0)

        min_surface_clear_m = float("inf")
        worst_clear_pair = ""
        solver_errors: list[str] = []

        if skip_clearance_if_zero and not (raw_tier > 0.0 and all_settled and no_solid_embedding_any):
            min_surface_clear_m = 0.0
            worst_clear_pair = "skipped_zero_tier_or_unsettled"
            hands_clear = False
        else:
            for smp, ca_obbs, ca_pts, cb_obbs, cb_pts, beads_arr in per_sample_obbs:
                memo_key: tuple | None = None
                try:
                    rq = np.asarray(smp.right_q6, dtype=float)
                    rf = np.asarray(smp.right_fingers, dtype=float)
                    lq = np.asarray(smp.left_q6, dtype=float)
                    lf = np.asarray(smp.left_fingers, dtype=float)
                    if (
                        rq.shape == (6,)
                        and rf.shape == (2,)
                        and lq.shape == (6,)
                        and lf.shape == (2,)
                        and np.all(np.isfinite(rq))
                        and np.all(np.isfinite(rf))
                        and np.all(np.isfinite(lq))
                        and np.all(np.isfinite(lf))
                    ):
                        memo_key = (
                            id(minimize),
                            id(convex_hull_to_obb_distance_m),
                            id(convex_hull_to_sphere_distance_m),
                            rq.tobytes(),
                            rf.tobytes(),
                            lq.tobytes(),
                            lf.tobytes(),
                            beads_arr.tobytes(),
                            ca_pts.tobytes(),
                            cb_pts.tobytes(),
                        )
                except Exception:
                    memo_key = None

                if memo_key is not None and memo_key in self._clearance_memo:
                    d_surf, pair_lbl, smp_errs = self._clearance_memo[memo_key]
                    solver_errors.extend(smp_errs)
                else:
                    smp_err_list: list[str] = []
                    d_surf, pair_lbl = self.min_hand_surface_clearance_m(
                        smp,
                        ca_obbs + cb_obbs + static_obbs_list,
                        solver_errors=smp_err_list,
                    )
                    solver_errors.extend(smp_err_list)
                    if memo_key is not None:
                        if len(self._clearance_memo) >= 128:
                            self._clearance_memo.pop(next(iter(self._clearance_memo)))
                        self._clearance_memo[memo_key] = (d_surf, pair_lbl, tuple(smp_err_list))

                if not math.isfinite(d_surf) or d_surf < min_surface_clear_m:
                    min_surface_clear_m = d_surf if math.isfinite(d_surf) else 0.0
                    worst_clear_pair = f"{pair_lbl} @ t={smp.t:.2f}s"

            hands_clear = bool(
                (len(solver_errors) == 0)
                and math.isfinite(min_surface_clear_m)
                and (min_surface_clear_m >= HAND_SURFACE_CLEARANCE_M)
            )

        final_score = raw_tier if (hands_clear and all_settled and no_solid_embedding_any) else 0.0

        return {
            "score": float(final_score),
            "raw_goal_tier_before_gates": float(raw_tier),
            "gates": {
                "hands_surface_clear_80mm_all_21": hands_clear,
                "min_hand_surface_clearance_mm": round(min_surface_clear_m * 1000.0, 3),
                "closest_hand_surface_pair": worst_clear_pair,
                "solver_error_count": len(solver_errors),
                "solver_errors": solver_errors[:5],
                "all_visible_cards_settled_21_strict_under_1mm": all_settled,
                "no_solid_embedding_any_card_21": no_solid_embedding_any,
                "check_a_corner_allpairs_mm": round(ca_allpairs_m * 1000.0, 4),
                "check_b_corner_allpairs_mm": round(cb_allpairs_m * 1000.0, 4),
                "beads_center_allpairs_mm_diagnostic": round(bd_allpairs_m * 1000.0, 4),
                "beads_diagnostic_under_1mm": beads_settled_diag,
            },
            "tier_a_completed_check_in_tray": {
                "passed": tier_a_tray,
                "out_of_rack": ca_out_of_rack,
                "free_of_solid_embedding_21": ca_tray_solid_21_ok,
                "max_solid_obb_pen_21_mm": round(max_ca_solid_pen_21_m * 1000.0, 4),
                "all_40_corners_in_tray_xy": ca_all_corners_in_tray_xy,
                "all_40_corners_in_tray_z": ca_all_corners_in_tray_z,
                "mass_weighted_com_supported": ca_com_supported,
                "mass_weighted_com_xyz": [round(float(v), 5) for v in final_ca_com],
                "corners_min_xyz": [round(float(v), 5) for v in final_ca_corners.min(axis=0)],
                "corners_max_xyz": [round(float(v), 5) for v in final_ca_corners.max(axis=0)],
            },
            "tier_b_new_check_retained_in_rack": {
                "passed": tier_b_rack,
                "free_channel_and_solid_ok_21": cb_channel_21_ok,
                "free_channel_status": cb_channel_fail_reason,
                "max_solid_obb_pen_21_mm": round(max_cb_solid_pen_21_m * 1000.0, 4),
                "zone_y_ok": cb_zone_y_ok,
                "seated_z_top_ok": cb_seated_z_ok,
                "suspended_free_ok": cb_suspended_ok,
                "upright_within_15deg": cb_upright_ok,
                "upright_tilt_deg": round(cb_upright_deg, 3),
                "z_top_rel_rack_mm": round((cb_z_top - g.z_rack_bot) * 1000.0, 3),
                "z_bot_above_table_mm": round((cb_z_bot - g.top_z) * 1000.0, 3),
                "mass_weighted_com_xyz": [round(float(v), 5) for v in final_cb_com],
            },
        }
