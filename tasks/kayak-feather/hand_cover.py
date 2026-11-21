"""Certified whole-hand + D405 wrist-camera cover points for both YAM arms (424 points, 14.35 mm radius), copied verbatim
from the expandable-trivet task (claude-cake / antigravity build; hand_cover_data.py 536a04c5). Functions:
  extract_robot_links_from_env(env) -> {arm.link: 7-pose}
  certified_hand_points_from_link_poses(poses) -> (424, 3) world points (raises on rail/orientation violations)"""
from __future__ import annotations
import numpy as np
from hand_cover_data import GRIPPER_D405_COVER_LOCAL, HAND_COVER_RADIUS_M, N_PTS_PER_HAND, TIP_LEFT_COVER_NATIVE, TIP_RIGHT_COVER_NATIVE

_J_FINGER_ROT0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
_J7_ORIGIN_XYZ = np.array([0.0445119, -0.0239931, 0.0566829], dtype=np.float64)
_J7_AXIS_G = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_J8_ORIGIN_XYZ = np.array([-0.0445119, 0.0239931, 0.0566829], dtype=np.float64)
_J8_AXIS_G = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
_FINGER_ROT_FROB_TOL = 0.05
_FINGER_RAIL_PERP_TOL = 0.003
_FINGER_SLIDE_MIN = -0.0495
_FINGER_SLIDE_MAX = 0.0025


def _quat_xyzw_to_rot(q_xyzw):
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


def certified_hand_points_from_link_poses(link_poses_xyzw):
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
        q_tl = float(np.dot(rel_tl, _J7_AXIS_G)); perp_tl = float(np.linalg.norm(rel_tl - q_tl * _J7_AXIS_G))
        if perp_tl > _FINGER_RAIL_PERP_TOL or not (_FINGER_SLIDE_MIN <= q_tl <= _FINGER_SLIDE_MAX):
            raise ValueError(f"{arm}.tip_left rail violation (q={q_tl:.4f}, perp={perp_tl:.4f})")
        rel_tr = R_g.T @ (p_tr - p_g) - _J8_ORIGIN_XYZ
        q_tr = float(np.dot(rel_tr, _J8_AXIS_G)); perp_tr = float(np.linalg.norm(rel_tr - q_tr * _J8_AXIS_G))
        if perp_tr > _FINGER_RAIL_PERP_TOL or not (_FINGER_SLIDE_MIN <= q_tr <= _FINGER_SLIDE_MAX):
            raise ValueError(f"{arm}.tip_right rail violation (q={q_tr:.4f}, perp={perp_tr:.4f})")
        out.append(np.vstack([(R_g @ GRIPPER_D405_COVER_LOCAL.T).T + p_g, (R_tl @ TIP_LEFT_COVER_NATIVE.T).T + p_tl, (R_tr @ TIP_RIGHT_COVER_NATIVE.T).T + p_tr]))
    arr = np.vstack(out)
    if arr.shape != (2 * N_PTS_PER_HAND, 3):
        raise ValueError("unexpected total cover shape")
    return arr


def extract_robot_links_from_env(environment):
    base = environment.unwrapped
    origin = np.asarray(base.scene.env_origins[0].detach().cpu().numpy(), dtype=np.float64)
    poses = {}
    for arm, key in (("left", "left_robot"), ("right", "right_robot")):
        art = base.scene[key]; bnames = list(art.body_names)
        bpose = np.asarray(art.data.body_link_pose_w[0].detach().cpu().numpy(), dtype=np.float64)
        for link in ("gripper", "tip_left", "tip_right"):
            p7 = bpose[bnames.index(link)].copy(); p7[:3] -= origin; poses[f"{arm}.{link}"] = p7
    return poses
