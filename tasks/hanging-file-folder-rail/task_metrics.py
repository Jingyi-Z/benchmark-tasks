"""Pure 3D geometric scorer for hanging-file-folder-rail (zero joint-angle cheating)."""
import math
import numpy as np


def ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def box_rot(b) -> np.ndarray:
    turn = b[3] if len(b) > 3 else 0.0
    yaw, pitch = (float(turn[0]), float(turn[1])) if isinstance(turn, (tuple, list)) else (float(turn), 0.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rp = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rz @ rp


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


class Metric:
    def __init__(self, ns: dict):
        self.top = float(ns["TOP"])
        self.rail_y = float(ns["RAIL_Y"])            # 0.1588 m
        self.rail_z_top = float(ns["RAIL_Z_TOP"])    # 0.9854 m
        self.rail_x_min = float(ns["RAIL_X_MIN"])    # 0.4250 m
        self.rail_x_max = float(ns["RAIL_X_MAX"])    # 0.7806 m
        self.rod_x_local = float(ns["ROD_SEAT_X"])   # 0.2300 m (underside saddle bearing x_local)
        self.gusset_z = float(ns["GUSSET_Z"])        # 0.0092 m (rear_leaf hinge z_local)
        self.anchor_r = np.array([0.0, 0.0, self.gusset_z], float)
        self.leaf_half_y = float(ns["LEAF_W"]) * 0.5 # 0.14925 m
        self.frame_base_top = float(ns["FRAME_BASE_TOP"])  # TOP + 0.0040 m

        self.front_boxes = ns["FRONT_LEAF_BOXES"]
        self.rear_boxes = ns["REAR_LEAF_BOXES"]
        self.pkt_boxes = ns["PACKET"].boxes
        self.frame_boxes = ns["FILE_FRAME"].boxes
        self.frame_pos = np.asarray(ns["FILE_FRAME"].pos, float)
        self.riser_boxes = ns["INBOX_RISER"].boxes
        self.riser_pos = np.asarray(ns["INBOX_RISER"].pos, float)
        self._I3 = np.eye(3)

        self.front_corners = part_local_corners(self.front_boxes)
        self.rear_corners = part_local_corners(self.rear_boxes)
        self.pkt_corners = part_local_corners(self.pkt_boxes)
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

    def rear_origin_and_rot(self, fld_s):
        p_f = np.asarray(fld_s.pos, float)
        R_f = fld_s.rotation()
        theta = float(fld_s.joints.get("rear_leaf", 0.0))
        R_r = R_f @ ry(theta)
        # labkit specifies joint boxes in part frame at rest (theta=0), rotating around anchor_r:
        # world_pt = p_f + R_f @ anchor_r + R_r @ (q - anchor_r) = p_r + R_r @ q
        p_r = p_f + R_f @ self.anchor_r - R_r @ self.anchor_r
        return p_f, R_f, p_r, R_r, theta

    def evaluate_states(self, fld_s, pkt_s, tip_pts_world: np.ndarray, hist_override=None) -> float:
        p_f, R_f, p_r, R_r, theta = self.rear_origin_and_rot(fld_s)

        p_p = np.asarray(pkt_s.pos, float)
        R_p = pkt_s.rotation()

        fc_w = self.front_corners @ R_f.T + p_f
        rc_w = self.rear_corners @ R_r.T + p_r
        pc_w = self.pkt_corners @ R_p.T + p_p
        all_pts = np.vstack([fc_w, rc_w, pc_w])
        off_table = float(np.min(all_pts[:, 2])) < self.top - 0.030

        if hist_override is not None:
            hist = hist_override
        else:
            self.hist.append(all_pts)
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
            dist_pts_to_boxes(tip_pts_world, self.front_boxes, p_f, R_f),
            dist_pts_to_boxes(tip_pts_world, self.rear_boxes, p_r, R_r),
            dist_pts_to_boxes(tip_pts_world, self.pkt_boxes, p_p, R_p),
            dist_pts_to_boxes(tip_pts_world, self.frame_boxes, self.frame_pos, self._I3),
            dist_pts_to_boxes(tip_pts_world, self.riser_boxes, self.riser_pos, self._I3),
        )
        hands_ok = hand_clear >= 0.080

        # 4 underside rail-hook saddle points in world coordinates:
        notch_fl_local = np.array([self.rod_x_local, +self.rail_y, 0.00125])
        notch_fr_local = np.array([self.rod_x_local, -self.rail_y, 0.00125])
        notch_rl_local = np.array([self.rod_x_local, +self.rail_y, self.gusset_z + 0.00125])
        notch_rr_local = np.array([self.rod_x_local, -self.rail_y, self.gusset_z + 0.00125])
        fl_w = p_f + R_f @ notch_fl_local
        fr_w = p_f + R_f @ notch_fr_local
        rl_w = p_r + R_r @ notch_rl_local
        rr_w = p_r + R_r @ notch_rr_local

        notches = [(fl_w, +self.rail_y), (rl_w, +self.rail_y), (fr_w, -self.rail_y), (rr_w, -self.rail_y)]
        max_dz = max(abs(float(pt[2] - self.rail_z_top)) for pt, _ in notches)
        max_dy = max(abs(float(pt[1] - ry_t)) for pt, ry_t in notches)
        x_vals = [float(pt[0]) for pt, _ in notches]
        x_in_rails = min(x_vals) >= self.rail_x_min + 0.015 and max(x_vals) <= self.rail_x_max - 0.015

        # Crease suspended above desk (8.0 mm nominal when hanging on 238 mm rail vs 0.5 mm flat on desk):
        crease_pts = np.array([[0.0, -0.145, 0.0], [0.0, 0.0, 0.0], [0.0, +0.145, 0.0]]) @ R_f.T + p_f
        crease_min_z = float(np.min(crease_pts[:, 2]))
        crease_clear = crease_min_z - self.top

        hooks_ok = (
            max_dz <= 0.0045
            and max_dy <= 0.0060
            and x_in_rails
            and crease_clear >= 0.0050
        )

        # Tier 1.0: Document packet retained upright inside the suspended V-folder:
        pkt_min_z = float(np.min(pc_w[:, 2]))
        pkt_clear = pkt_min_z - self.top
        rel_f = R_f.T @ (p_p - p_f)
        rel_r = R_r.T @ (p_p - p_r)
        # Front leaf inner face sits at z_f = 0.0025; rear leaf inner face sits at z_r = GUSSET_Z (0.0092):
        between_leaves = (
            float(rel_f[2]) >= 0.0015
            and float(rel_r[2]) <= self.gusset_z + 0.0010
            and 0.020 <= float(rel_f[0]) <= 0.220
            and abs(float(rel_f[1])) <= self.leaf_half_y - 0.005
        )
        pkt_y_span_ok = float(np.max(np.abs(pc_w[:, 1] - p_f[1]))) <= self.leaf_half_y + 0.010
        packet_ok = between_leaves and pkt_y_span_ok and pkt_clear >= 0.0050

        sc = 0.0
        if not off_table and hands_ok and still_ok and hooks_ok:
            sc = 1.0 if packet_ok else 0.5

        self.last_score = sc
        self.detail = dict(
            score=sc,
            off_table=off_table,
            hand_clear_mm=round(hand_clear * 1000.0, 2),
            still_mm=round(still_disp * 1000.0, 3),
            max_notch_dz_mm=round(max_dz * 1000.0, 2),
            max_notch_dy_mm=round(max_dy * 1000.0, 2),
            x_min=round(min(x_vals), 4),
            x_max=round(max(x_vals), 4),
            crease_clear_mm=round(crease_clear * 1000.0, 2),
            pkt_clear_mm=round(pkt_clear * 1000.0, 2),
            between_leaves=between_leaves,
            theta_deg=round(math.degrees(theta), 2),
        )
        return sc

    def update(self, env, fld_s, pkt_s) -> float:
        tips = self._six_hand_points(env)
        return self.evaluate_states(fld_s, pkt_s, tips)

    def score(self) -> float:
        return float(self.last_score)
