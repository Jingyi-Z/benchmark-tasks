"""Separately hashed observation pose adapter for IsaacLab / PhysX link poses.

Engine validation (pose_peel_coupon1 Part A):
- Both `isaaclab_physx.assets.articulation.articulation_data.ArticulationData.body_link_pose_w`
  and `omni.physics.tensors.api.ArticulationView.get_link_transforms()` emit 7-tuples in
  `(x, y, z, qx, qy, qz, qw)` order (`[0, 0, 0, 1]` at identity; `0.00006 mm` child-anchor
  residual vs `4.34457 mm` under `wxyz` at 3-axis roll=+15 deg, pitch=+28 deg, yaw=-37 deg).
- `goal_scorer.py` (`938cc50c...`, unchanged from `prototype-m3`) expects `quat_wxyz` in
  `WorldSample.check_a_bodies` / `check_b_bodies`.

This adapter converts raw `(qx, qy, qz, qw)` link poses to normalized `(qw, qx, qy, qz)` for
`GoalScorer` and `card_world_obbs_and_com`, explicitly rejects non-finite (`NaN`/`Inf`) poses,
origins, quaternions, norms, residuals, and non-positive/non-finite tolerances, logs invalid
inputs, and enforces a runtime articulated joint-anchor consistency check on every step.
"""
from __future__ import annotations

import math
import numpy as np

from goal_scorer import quat_wxyz_to_mat

# Authored parent->child hinge anchor offsets in parent body-local frame (from make_check in task.py):
CARD_CHAIN_ANCHORS_LOCAL: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("body", "hinge_01", (0.0, 0.0, -0.01575)),
    ("hinge_01", "hinge_12", (0.0, 0.0, -0.04500)),
    ("hinge_12", "hinge_23", (0.0, 0.0, -0.04750)),
)

# Max allowable PhysX reduced-coordinate joint anchor separation (typically < 0.05 mm; 2.0 mm catches any wxyz/xyzw swap):
MAX_JOINT_ANCHOR_RESIDUAL_M = 0.0020


def _validate_positive_finite_tol(tol_m: float) -> float:
    if isinstance(tol_m, bool) or not isinstance(tol_m, (int, float, np.floating)):
        raise ValueError(f"Invalid tolerance type {type(tol_m).__name__}: {tol_m!r}")
    tol_f = float(tol_m)
    if (not math.isfinite(tol_f)) or (tol_f <= 0.0):
        raise ValueError(f"Invalid non-positive or non-finite tolerance tol_m={tol_m!r}")
    return tol_f


def raw_quat_xyzw_to_wxyz(raw_q_xyzw: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert raw `(qx, qy, qz, qw)` quaternion from IsaacLab/PhysX to normalized `(qw, qx, qy, qz)`.

    Explicitly rejects non-finite (`NaN`/`Inf`) components and zero/degenerate norms.
    """
    if len(raw_q_xyzw) != 4:
        raise ValueError(f"Expected 4-element quaternion, got len={len(raw_q_xyzw)}")
    qx, qy, qz, qw = (
        float(raw_q_xyzw[0]),
        float(raw_q_xyzw[1]),
        float(raw_q_xyzw[2]),
        float(raw_q_xyzw[3]),
    )
    if not (math.isfinite(qx) and math.isfinite(qy) and math.isfinite(qz) and math.isfinite(qw)):
        raise ValueError(f"Non-finite raw quaternion components: {(qx, qy, qz, qw)!r}")
    nrm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if (not math.isfinite(nrm)) or (nrm <= 1e-12):
        raise ValueError(f"Degenerate or non-finite quaternion norm={nrm!r}")
    inv = 1.0 / nrm
    out = np.array([qw * inv, qx * inv, qy * inv, qz * inv], dtype=float)
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(f"Non-finite normalized quaternion: {out.tolist()!r}")
    return out


def verify_card_chain_anchors(
    body_poses_wxyz: dict[str, tuple[np.ndarray, np.ndarray]],
    tol_m: float = MAX_JOINT_ANCHOR_RESIDUAL_M,
) -> float:
    """Verify child link origins match `p_parent + R(quat_wxyz_parent) @ anchor_local` within `tol_m`.

    Explicitly rejects non-finite body positions/quaternions, non-positive/non-finite `tol_m`,
    and non-finite residuals; raises ValueError/RuntimeError on any violation.
    """
    tol_f = _validate_positive_finite_tol(tol_m)
    for bname, (p_b, q_b) in body_poses_wxyz.items():
        p_arr = np.asarray(p_b, dtype=float)
        q_arr = np.asarray(q_b, dtype=float)
        if p_arr.shape != (3,) or not bool(np.all(np.isfinite(p_arr))):
            raise RuntimeError(f"Non-finite or invalid position on body {bname!r}: {p_arr.tolist()!r}")
        if q_arr.shape != (4,) or not bool(np.all(np.isfinite(q_arr))):
            raise RuntimeError(f"Non-finite or invalid quaternion on body {bname!r}: {q_arr.tolist()!r}")
        qn = float(np.linalg.norm(q_arr))
        if (not math.isfinite(qn)) or (qn <= 1e-12):
            raise RuntimeError(f"Degenerate quaternion norm on body {bname!r}: {qn!r}")

    max_err_m = 0.0
    for parent_name, child_name, anc_xyz in CARD_CHAIN_ANCHORS_LOCAL:
        p_par, q_par_wxyz = body_poses_wxyz[parent_name]
        p_chi, _ = body_poses_wxyz[child_name]
        R_par = quat_wxyz_to_mat(
            float(q_par_wxyz[0]),
            float(q_par_wxyz[1]),
            float(q_par_wxyz[2]),
            float(q_par_wxyz[3]),
        )
        if not bool(np.all(np.isfinite(R_par))):
            raise RuntimeError(f"Non-finite rotation matrix on {parent_name!r}")
        pred_chi = np.asarray(p_par, dtype=float) + R_par @ np.asarray(anc_xyz, dtype=float)
        err_m = float(np.linalg.norm(pred_chi - np.asarray(p_chi, dtype=float)))
        if (not math.isfinite(err_m)) or (err_m > tol_f):
            raise RuntimeError(
                f"Quaternion/anchor convention mismatch or non-finite residual on {parent_name}->{child_name}: "
                f"residual={err_m * 1000.0:.4f} mm (tol={tol_f * 1000.0:.2f} mm)"
            )
        if err_m > max_err_m:
            max_err_m = err_m
    return max_err_m


def adapt_articulation_body_poses(
    raw_body_link_pose_w: np.ndarray,
    body_names: list[str] | tuple[str, ...],
    origin_xyz: np.ndarray,
    tol_m: float = MAX_JOINT_ANCHOR_RESIDUAL_M,
    invalid_pose_log: list[str] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict], float]:
    """Convert raw `(N_bodies, 7)` `[x, y, z, qx, qy, qz, qw]` into validated `quat_wxyz` map + serializable log.

    Logs any invalid/non-finite input to `invalid_pose_log` (if provided) before raising to fail closed.
    """
    try:
        tol_f = _validate_positive_finite_tol(tol_m)
        poses_arr = np.asarray(raw_body_link_pose_w, dtype=float).reshape(len(body_names), 7)
        origin = np.asarray(origin_xyz, dtype=float).reshape(3)
        if not bool(np.all(np.isfinite(origin))):
            raise RuntimeError(f"Non-finite origin_xyz: {origin.tolist()!r}")
        if not bool(np.all(np.isfinite(poses_arr))):
            raise RuntimeError(f"Non-finite raw_body_link_pose_w: {poses_arr.tolist()!r}")

        body_poses_wxyz: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        body_poses_serial: dict[str, dict] = {}
        for bi, bname in enumerate(body_names):
            pos_rel = poses_arr[bi, :3] - origin
            if not bool(np.all(np.isfinite(pos_rel))):
                raise RuntimeError(f"Non-finite relative position for {bname!r}: {pos_rel.tolist()!r}")
            raw_xyzw = poses_arr[bi, 3:7].copy()
            q_wxyz = raw_quat_xyzw_to_wxyz(raw_xyzw)
            body_poses_wxyz[bname] = (pos_rel.copy(), q_wxyz)
            body_poses_serial[bname] = {
                "pos": [round(float(v), 6) for v in pos_rel],
                "raw_quat_xyzw": [round(float(v), 6) for v in raw_xyzw],
                "quat_wxyz": [round(float(v), 6) for v in q_wxyz],
            }

        max_anchor_err_m = verify_card_chain_anchors(body_poses_wxyz, tol_m=tol_f)
        return body_poses_wxyz, body_poses_serial, max_anchor_err_m
    except Exception as exc:
        if invalid_pose_log is not None:
            invalid_pose_log.append(f"{type(exc).__name__}: {exc}")
        raise
