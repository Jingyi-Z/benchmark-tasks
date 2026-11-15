"""Pure 3D world-pose geometric scorer for window-sash-cam-lock."""
import math
import numpy as np


def rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_rot(b) -> np.ndarray:
    turn = b[3] if len(b) > 3 else 0.0
    yaw, pitch = (float(turn[0]), float(turn[1])) if isinstance(turn, (tuple, list)) else (float(turn), 0.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz_m = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rp_m = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rz_m @ rp_m


def cyl_as_boxes(cyls) -> list:
    out = []
    for c in cyls:
        ctr, rad, hgt = c[0], float(c[1]), float(c[2])
        axis = c[4] if len(c) > 4 else "Z"
        if axis == "Z":
            sz = (2.0 * rad, 2.0 * rad, hgt)
        elif axis == "Y":
            sz = (2.0 * rad, hgt, 2.0 * rad)
        else:
            sz = (hgt, 2.0 * rad, 2.0 * rad)
        out.append((ctr, sz, (0.5, 0.5, 0.5), 0.0))
    return out


def part_local_corners(boxes) -> np.ndarray:
    pts = []
    for b in boxes:
        c = np.asarray(b[0], float)
        h = np.asarray(b[1], float) / 2.0
        rb = box_rot(b)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    pts.append(c + rb @ (h * np.array([sx, sy, sz])))
    return np.asarray(pts, float)


def dist_pts_to_boxes(world_pts: np.ndarray, boxes, pos: np.ndarray, R: np.ndarray) -> float:
    rel = (world_pts - pos) @ R
    best = float("inf")
    for b in boxes:
        c = np.asarray(b[0], float)
        h = np.asarray(b[1], float) / 2.0
        rb = box_rot(b)
        q = (rel - c) @ rb
        d = np.maximum(np.abs(q) - h, 0.0)
        best = min(best, float(np.min(np.linalg.norm(d, axis=-1))))
    return best


_AXIS_VEC = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


def dist_pts_to_cyls(world_pts: np.ndarray, cyls, pos: np.ndarray, R: np.ndarray) -> float:
    if not cyls:
        return float("inf")
    rel = (world_pts - pos) @ R
    best = float("inf")
    for c in cyls:
        ctr = np.asarray(c[0], float)
        rad = float(c[1])
        half_h = float(c[2]) / 2.0
        ax = _AXIS_VEC[c[4] if len(c) > 4 else "Z"]
        v = rel - ctr
        along = v @ ax
        perp = v - np.outer(along, ax)
        dr = np.maximum(np.linalg.norm(perp, axis=-1) - rad, 0.0)
        dz = np.maximum(np.abs(along) - half_h, 0.0)
        best = min(best, float(np.min(np.hypot(dr, dz))))
    return best


class Metric:
    def __init__(self, ns: dict):
        self.top = float(ns["TOP"])
        self.sill_z = float(ns["SILL_TOP_Z"])  # 0.0280 m in WINDOW frame
        self.cam_anchor = np.asarray(ns["CAM_ANCHOR"], float)  # (-0.0110, 0.0, 0.3030)
        self.dwell_local = np.asarray(ns["CAM_DWELL_LOCAL"], float)  # local rest center of dwell segment in WINDOW frame

        self.frame_only_boxes = list(ns["FRAME_BOXES"])
        self.frame_cyls = list(ns["FRAME_CYLS"])
        self.frame_boxes = self.frame_only_boxes + cyl_as_boxes(self.frame_cyls)
        self.sash_z_boxes = list(ns["SASH_Z_BOXES"])
        self.sash_x_boxes = list(ns["SASH_X_BOXES"])
        self.cam_only_boxes = list(ns["CAM_BOXES"])
        self.cam_cyls = list(ns["CAM_CYLS"])
        self.cam_boxes = self.cam_only_boxes + cyl_as_boxes(self.cam_cyls)

        self.sash_x_corners = part_local_corners(self.sash_x_boxes)
        self.cam_corners = part_local_corners(self.cam_boxes)
        self.frame_corners = part_local_corners(self.frame_boxes)
        self.lk = ns["labkit"]
        self.reset()

    def reset(self):
        self.hist = []
        self.last_score = 0.0
        self.detail = {}

    def _six_hand_points(self, env) -> np.ndarray:
        from yam_arena import runtime

        base = env.unwrapped
        origin = self.lk._np(base.scene.env_origins[0])
        tips = []
        for name in ("left_robot", "right_robot"):
            arm = base.scene[name]
            mid = runtime.pad_tips_world(arm) - origin
            g = runtime.link_poses(arm)["gripper"]
            ids = arm.find_joints(["joint7", "joint8"], preserve_order=True)[0]
            j7, j8 = np.abs(self.lk._np(arm.data.joint_pos)[0, ids])
            tips.extend([mid, mid - j7 * g[:3, 0], mid + j8 * g[:3, 0]])
        return np.asarray(tips, float)

    def link_poses(self, win_s):
        p_0 = np.asarray(win_s.pos, float)
        R_0 = win_s.rotation()
        dz = float(win_s.joints.get("sash_z", 0.0))
        dx = float(win_s.joints.get("sash_x", 0.0))
        theta = float(win_s.joints.get("cam_lock", 0.0))

        p_z = p_0 + R_0 @ np.array([0.0, 0.0, dz])
        R_z = R_0

        p_s = p_0 + R_0 @ np.array([dx, 0.0, dz])
        R_s = R_0

        R_c = R_s @ rz(theta)
        # labkit specifies joint shapes in part rest frame rotating about cam_anchor on parent sash_x:
        p_c = p_s + R_s @ self.cam_anchor - R_c @ self.cam_anchor
        return p_0, R_0, p_z, R_z, p_s, R_s, p_c, R_c, dz, dx, theta

    def evaluate_states(self, win_s, tip_pts_world: np.ndarray, hist_override=None) -> float:
        p_0, R_0, p_z, R_z, p_s, R_s, p_c, R_c, _, _, _ = self.link_poses(win_s)

        sx_w = self.sash_x_corners @ R_s.T + p_s
        cm_w = self.cam_corners @ R_c.T + p_c
        fr_w = self.frame_corners @ R_0.T + p_0
        moving_pts = np.vstack([sx_w, cm_w])
        all_pts = np.vstack([fr_w, moving_pts])
        off_table = float(np.min(all_pts[:, 2])) < self.top - 0.030

        if hist_override is not None:
            hist = hist_override
        else:
            self.hist.append(moving_pts)
            if len(self.hist) > 21:
                self.hist.pop(0)
            hist = self.hist

        still_disp = 999.0
        still_ok = False
        if len(hist) >= 21:
            h_arr = np.stack(hist[-21:], axis=0)
            disp = h_arr[:, None, :, :] - h_arr[None, :, :, :]
            still_disp = float(np.max(np.linalg.norm(disp, axis=-1)))
            still_ok = still_disp < 0.0010

        hand_clear = min(
            dist_pts_to_boxes(tip_pts_world, self.frame_only_boxes, p_0, R_0),
            dist_pts_to_cyls(tip_pts_world, self.frame_cyls, p_0, R_0),
            dist_pts_to_boxes(tip_pts_world, self.sash_z_boxes, p_z, R_z),
            dist_pts_to_boxes(tip_pts_world, self.sash_x_boxes, p_s, R_s),
            dist_pts_to_boxes(tip_pts_world, self.cam_only_boxes, p_c, R_c),
            dist_pts_to_cyls(tip_pts_world, self.cam_cyls, p_c, R_c),
        )
        hands_ok = hand_clear >= 0.080

        # Pure 3D geometric measurements in WINDOW frame (derived from 3D world poses of sash_x and cam_lock):
        # 1) Underside center of lower sash bottom rail (rest local point (-0.0135, 0.0, 0.0280)) above sill_z = 0.0280:
        rail_btm_w = p_s + R_s @ np.array([-0.0135, 0.0, self.sill_z])
        rail_btm_in_win = R_0.T @ (rail_btm_w - p_0)
        sill_gap = float(rail_btm_in_win[2] - self.sill_z)
        flush_gap = float(-0.0135 - rail_btm_in_win[0])  # 0.0022 m unlocked -> ~0.0002 m locked

        # 2) Crescent cam dwell center in WINDOW frame:
        dwell_w = p_c + R_c @ self.dwell_local
        dwell_in_win = R_0.T @ (dwell_w - p_0)

        closed_half = -0.0015 <= sill_gap <= 0.0060
        closed_full = -0.0015 <= sill_gap <= 0.0035
        keeper_captured = (
            float(dwell_in_win[0]) >= 0.0080
            and abs(float(dwell_in_win[1])) <= 0.0055
            and 0.3035 <= float(dwell_in_win[2]) <= 0.3115
        )
        weatherseal_drawn = -0.0010 <= flush_gap <= 0.0011

        if off_table or not hands_ok or not still_ok:
            val = 0.0
        elif closed_full and keeper_captured and weatherseal_drawn:
            val = 1.0
        elif closed_half:
            val = 0.5
        else:
            val = 0.0

        self.last_score = val
        self.detail = dict(
            score=val,
            off_table=bool(off_table),
            sill_gap_mm=round(sill_gap * 1000.0, 2),
            flush_gap_mm=round(flush_gap * 1000.0, 2),
            dwell_win_mm=[round(float(v) * 1000.0, 2) for v in dwell_in_win],
            keeper_captured=bool(keeper_captured),
            weatherseal_drawn=bool(weatherseal_drawn),
            hand_clear_mm=round(hand_clear * 1000.0, 1),
            still_mm=round(still_disp * 1000.0, 3),
        )
        return val

    def update(self, env, win_s) -> float:
        tips = self._six_hand_points(env)
        return self.evaluate_states(win_s, tips)

    def score(self) -> float:
        return float(self.last_score)
