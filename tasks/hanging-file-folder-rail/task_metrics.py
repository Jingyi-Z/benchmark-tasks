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


class _DefaultPose:
    def __init__(self, pos):
        self.pos = np.asarray(pos, float)
        self._R = np.eye(3)

    def rotation(self) -> np.ndarray:
        return self._R


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
        self.rear_folder_specs = list(ns.get("REAR_FOLDERS", ()))
        self._I3 = np.eye(3)

        self.front_corners = part_local_corners(self.front_boxes)
        self.rear_corners = part_local_corners(self.rear_boxes)
        self.pkt_corners = part_local_corners(self.pkt_boxes)
        self.rear_folder_corners = [part_local_corners(rf.boxes) for rf in self.rear_folder_specs]
        self.lk = ns["labkit"]
        # Derive each void from its two actual retaining lugs, not a point tolerance.
        self.saddles = []
        for boxes, rear in ((self.front_boxes, False), (self.rear_boxes, True)):
            for sign in (1.0, -1.0):
                lugs = [b for b in boxes
                        if abs(float(b[0][0]) - (self.rod_x_local - 0.00225)) < 1e-7
                        and abs(float(b[0][1]) - sign * self.rail_y) < 0.012]
                assert len(lugs) == 2, "Expected two physical saddle retaining lugs"
                lugs.sort(key=lambda b: b[0][1])
                lo, hi = lugs
                self.saddles.append(dict(
                    rear=rear, rail_y=sign * self.rail_y,
                    ylo=float(lo[0][1] + lo[1][1] / 2),
                    yhi=float(hi[0][1] - hi[1][1] / 2),
                    xlo=max(float(b[0][0] - b[1][0] / 2) for b in lugs),
                    bearing=min(float(b[0][0] + b[1][0] / 2) for b in lugs),
                    zlo=max(float(b[0][2] - b[1][2] / 2) for b in lugs),
                    zhi=min(float(b[0][2] + b[1][2] / 2) for b in lugs)))
        self.rail_half_width = float(self.frame_boxes[0][1][1]) / 2
        self.reset()

    def rail_in_saddle(self, origin, rotation, saddle):
        """Intersect the real rail with both finite hook faces.

        Both sides of the 3.6mm rail must be inside the 7.2mm void,
        including the upper 2mm of its height. The rail top must meet
        the rod's bearing underside (0.50mm separation / 0.10mm solver
        penetration allowance). No sideways clearance is added.
        """
        normal = rotation[:, 2]
        if abs(float(normal[0])) < 0.25:
            return False, -999.0, 999.0
        points = []
        tops = []
        for local_z in (saddle['zlo'], saddle['zhi']):
            for y in (saddle['rail_y'] - self.rail_half_width,
                      saddle['rail_y'] + self.rail_half_width):
                for z in (self.rail_z_top, self.rail_z_top - 0.002):
                    # Dot(n, world-origin)=local_z, solving for rail-longitudinal X.
                    x = float(origin[0] + (local_z - normal[1] * (y-origin[1])
                                          - normal[2] * (z-origin[2])) / normal[0])
                    if not self.rail_x_min <= x <= self.rail_x_max:
                        return False, -999.0, 999.0
                    q = rotation.T @ (np.array([x, y, z]) - origin)
                    points.append(q)
                    if z == self.rail_z_top:
                        tops.append(q)
        points = np.asarray(points)
        tops = np.asarray(tops)
        side_margin = min(float(np.min(points[:, 1]) - saddle['ylo']),
                          float(saddle['yhi'] - np.max(points[:, 1])))
        bearing_gap = saddle['bearing'] - tops[:, 0]
        supported = bool(np.all(bearing_gap >= -0.00010)
                         and np.all(bearing_gap <= 0.00050))
        nested = bool(np.min(points[:, 0]) >= saddle['xlo'] - 1e-7
                      and np.max(points[:, 0]) <= saddle['bearing'] + 0.00010)
        return (side_margin >= -1e-7 and nested and supported,
                side_margin * 1000, float(np.max(bearing_gap)) * 1000)

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

    def all_world_corners(self, fld_s, pkt_s, rear_states=None) -> np.ndarray:
        p_f, R_f, p_r, R_r, _ = self.rear_origin_and_rot(fld_s)
        p_p = np.asarray(pkt_s.pos, float)
        R_p = pkt_s.rotation()
        blocks = [
            self.front_corners @ R_f.T + p_f,
            self.rear_corners @ R_r.T + p_r,
            self.pkt_corners @ R_p.T + p_p,
        ]
        if rear_states is None:
            rear_states = [_DefaultPose(rf.pos) for rf in self.rear_folder_specs]
        for corners_rf, rf_s in zip(self.rear_folder_corners, rear_states):
            blocks.append(corners_rf @ rf_s.rotation().T + np.asarray(rf_s.pos, float))
        return np.vstack(blocks)

    def evaluate_states(self, fld_s, pkt_s, tip_pts_world: np.ndarray, rear_states=None, hist_override=None) -> float:
        p_f, R_f, p_r, R_r, theta = self.rear_origin_and_rot(fld_s)
        if rear_states is None:
            rear_states = [_DefaultPose(rf.pos) for rf in self.rear_folder_specs]

        p_p = np.asarray(pkt_s.pos, float)
        R_p = pkt_s.rotation()

        fc_w = self.front_corners @ R_f.T + p_f
        rc_w = self.rear_corners @ R_r.T + p_r
        pc_w = self.pkt_corners @ R_p.T + p_p
        rf_world_blocks = [
            corners_rf @ rf_s.rotation().T + np.asarray(rf_s.pos, float)
            for corners_rf, rf_s in zip(self.rear_folder_corners, rear_states)
        ]
        all_pts = np.vstack([fc_w, rc_w, pc_w] + rf_world_blocks)
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

        clear_dists = [
            dist_pts_to_boxes(tip_pts_world, self.front_boxes, p_f, R_f),
            dist_pts_to_boxes(tip_pts_world, self.rear_boxes, p_r, R_r),
            dist_pts_to_boxes(tip_pts_world, self.pkt_boxes, p_p, R_p),
            dist_pts_to_boxes(tip_pts_world, self.frame_boxes, self.frame_pos, self._I3),
            dist_pts_to_boxes(tip_pts_world, self.riser_boxes, self.riser_pos, self._I3),
        ]
        for rf_spec, rf_s in zip(self.rear_folder_specs, rear_states):
            clear_dists.append(
                dist_pts_to_boxes(tip_pts_world, rf_spec.boxes, np.asarray(rf_s.pos, float), rf_s.rotation())
            )
        hand_clear = min(clear_dists)
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

        # Crease suspended above desk (8.0 mm nominal when hanging on 238 mm rail vs 0.5 mm flat on desk):
        crease_pts = np.array([[0.0, -0.145, 0.0], [0.0, 0.0, 0.0], [0.0, +0.145, 0.0]]) @ R_f.T + p_f
        crease_min_z = float(np.min(crease_pts[:, 2]))
        crease_clear = crease_min_z - self.top

        engagement = [self.rail_in_saddle(p_r if s['rear'] else p_f,
                                         R_r if s['rear'] else R_f, s)
                      for s in self.saddles]
        hooks_ok = all(e[0] for e in engagement) and crease_clear >= 0.0050

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
            rail_engaged=[bool(e[0]) for e in engagement],
            min_saddle_side_margin_mm=round(min(e[1] for e in engagement), 3),
            max_bearing_gap_mm=round(max(e[2] for e in engagement), 3),
        )
        return sc

    def update(self, env, fld_s, pkt_s, rear_states=None) -> float:
        tips = self._six_hand_points(env)
        return self.evaluate_states(fld_s, pkt_s, tips, rear_states=rear_states)

    def score(self) -> float:
        return float(self.last_score)
