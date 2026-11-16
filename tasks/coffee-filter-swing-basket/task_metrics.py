"""Pure 3D world-pose geometric scorer for coffee-filter-swing-basket."""
import math
import numpy as np


def rx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_rot(b) -> np.ndarray:
    turn = b[3] if len(b) > 3 else 0.0
    yaw, pitch = (float(turn[0]), float(turn[1])) if isinstance(turn, (tuple, list)) else (float(turn), 0.0)
    return rz(yaw) @ ry(pitch)


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
    if not boxes:
        return float("inf")
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
        self.swivel_anchor = np.asarray(ns["SWIVEL_ANCHOR"], float)
        self.basket_center_b = np.asarray(ns["BASKET_CENTER_B"], float)  # (-0.080, 0.000) in brewer rest frame
        self.z_seat_floor = float(ns["Z_SEAT_FLOOR"])                    # 0.2280 m in brewer rest frame
        self.filter_height = float(ns["FILTER_HEIGHT"])                  # 0.0533 m
        self.bail_anchor = np.asarray(ns["BAIL_ANCHOR"], float)          # (0.0, 0.0, 0.0498) in filter rest frame

        self.brewer_boxes = list(ns["BREWER_BOXES"])
        self.brewer_cyls = list(ns["BREWER_CYLS"])
        self.swivel_boxes = list(ns["SWIVEL_BOXES"])
        self.swivel_cyls = list(ns["SWIVEL_CYLS"])
        self.filter_boxes = list(ns["FILTER_BOXES"])
        self.filter_cyls = list(ns["FILTER_CYLS"])
        self.bail_boxes = list(ns["BAIL_BOXES"])

        self.brewer_corners = part_local_corners(self.brewer_boxes + cyl_as_boxes(self.brewer_cyls))
        self.swivel_corners = part_local_corners(self.swivel_boxes + cyl_as_boxes(self.swivel_cyls))
        self.filter_corners = part_local_corners(self.filter_boxes + cyl_as_boxes(self.filter_cyls))
        self.bail_corners = part_local_corners(self.bail_boxes)
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

    def brewer_link_poses(self, brewer_s):
        """World pose (p_b, R_b) of brewer main body and (p_sw, R_sw) of swivel_holder rest frame."""
        p_b = np.asarray(brewer_s.pos, float)
        R_b = brewer_s.rotation()
        swivel_rad = float(brewer_s.joints.get("swivel_holder", 0.0))
        # Shapes on swivel_holder are defined in brewer rest frame and rotate about SWIVEL_ANCHOR along +Z:
        R_loc = rz(swivel_rad)
        R_sw = R_b @ R_loc
        p_sw = p_b + R_b @ (self.swivel_anchor - R_loc @ self.swivel_anchor)
        return p_b, R_b, p_sw, R_sw, swivel_rad

    def filter_link_poses(self, filter_s):
        """World pose (p_f, R_f) of permanent filter body and (p_bl, R_bl) of bail rest frame."""
        p_f = np.asarray(filter_s.pos, float)
        R_f = filter_s.rotation()
        bail_rad = float(filter_s.joints.get("bail", 0.0))
        # Shapes on bail are defined in filter rest frame and rotate about BAIL_ANCHOR along +X:
        R_loc = rx(bail_rad)
        R_bl = R_f @ R_loc
        p_bl = p_f + R_f @ (self.bail_anchor - R_loc @ self.bail_anchor)
        return p_f, R_f, p_bl, R_bl, bail_rad

    def evaluate_states(self, brewer_s, filter_s, tip_pts_world: np.ndarray, hist_override=None) -> float:
        p_b, R_b, p_sw, R_sw, swivel_rad = self.brewer_link_poses(brewer_s)
        p_f, R_f, p_bl, R_bl, bail_rad = self.filter_link_poses(filter_s)

        br_w = self.brewer_corners @ R_b.T + p_b
        sw_w = self.swivel_corners @ R_sw.T + p_sw
        fl_w = self.filter_corners @ R_f.T + p_f
        bl_w = self.bail_corners @ R_bl.T + p_bl

        all_pts = np.vstack([br_w, sw_w, fl_w, bl_w])
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
            dist_pts_to_boxes(tip_pts_world, self.brewer_boxes, p_b, R_b),
            dist_pts_to_cyls(tip_pts_world, self.brewer_cyls, p_b, R_b),
            dist_pts_to_boxes(tip_pts_world, self.swivel_boxes, p_sw, R_sw),
            dist_pts_to_cyls(tip_pts_world, self.swivel_cyls, p_sw, R_sw),
            dist_pts_to_boxes(tip_pts_world, self.filter_boxes, p_f, R_f),
            dist_pts_to_cyls(tip_pts_world, self.filter_cyls, p_f, R_f),
            dist_pts_to_boxes(tip_pts_world, self.bail_boxes, p_bl, R_bl),
        )
        hands_ok = hand_clear >= 0.080

        # 1. Coffee maker upright within 8.0 deg:
        brewer_tilt_deg = float(math.degrees(math.acos(float(np.clip(R_b[2, 2], -1.0, 1.0)))))
        brewer_upright_ok = brewer_tilt_deg <= 8.0

        # 2. Permanent filter seated upright inside the removable filter basket (carried on swivel_holder):
        seat_floor_local_sw = np.array([self.basket_center_b[0], self.basket_center_b[1], self.z_seat_floor])
        seat_rim_local_sw = np.array([
            self.basket_center_b[0],
            self.basket_center_b[1],
            self.z_seat_floor + self.filter_height,
        ])
        seat_floor_w = p_sw + R_sw @ seat_floor_local_sw
        seat_rim_w = p_sw + R_sw @ seat_rim_local_sw

        filter_base_w = p_f
        filter_rim_w = p_f + R_f @ np.array([0.0, 0.0, self.filter_height])

        u_sw_z = R_sw[:, 2]
        u_f_z = R_f[:, 2]
        filter_axis_deg = float(math.degrees(math.acos(float(np.clip(np.dot(u_sw_z, u_f_z), -1.0, 1.0)))))

        # Radial offset of filter rim center from basket axis, and axial base offset from Z_SEAT_FLOOR:
        rim_delta_w = filter_rim_w - seat_rim_w
        rim_radial_m = float(np.linalg.norm(rim_delta_w - np.dot(rim_delta_w, u_sw_z) * u_sw_z))
        base_axial_m = float(np.dot(filter_base_w - seat_floor_w, u_sw_z))

        seated_ok = (
            filter_axis_deg <= 10.0
            and rim_radial_m <= 0.0080
            and -0.0040 <= base_axial_m <= 0.0060
        )

        # 3. Bail folded flat inside the filter rim (<= 14.0 deg from flat AND <= 10.2 mm above filter rim plane):
        bail_deg = abs(float(math.degrees(bail_rad)))
        bail_rel_rim = (bl_w - filter_rim_w) @ u_f_z
        bail_above_rim_m = float(np.max(bail_rel_rim))
        bail_folded_ok = (bail_deg <= 14.0) and (bail_above_rim_m <= 0.0102)

        # 4. Swivel basket holder closed flush into brewer bay (<= 5.0 deg from closed):
        swivel_deg = abs(float(math.degrees(swivel_rad)))
        swivel_closed_ok = swivel_deg <= 5.0

        valid_posture = (not off_table) and brewer_upright_ok and hands_ok and still_ok

        if valid_posture and seated_ok and bail_folded_ok and swivel_closed_ok:
            sc = 1.0
        elif valid_posture and seated_ok and (not swivel_closed_ok):
            sc = 0.5
        else:
            sc = 0.0

        self.detail = {
            "score": sc,
            "off_table": off_table,
            "brewer_tilt_deg": round(brewer_tilt_deg, 3),
            "brewer_upright_ok": brewer_upright_ok,
            "seated_ok": seated_ok,
            "filter_axis_deg": round(filter_axis_deg, 3),
            "rim_radial_mm": round(rim_radial_m * 1000.0, 2),
            "base_axial_mm": round(base_axial_m * 1000.0, 2),
            "bail_deg": round(bail_deg, 2),
            "bail_above_rim_mm": round(bail_above_rim_m * 1000.0, 2),
            "bail_folded_ok": bail_folded_ok,
            "swivel_deg": round(swivel_deg, 2),
            "swivel_closed_ok": swivel_closed_ok,
            "hand_clear_mm": round(hand_clear * 1000.0, 2),
            "hands_ok": hands_ok,
            "still_disp_mm": round(still_disp * 1000.0, 3),
            "still_ok": still_ok,
        }
        self.last_score = sc
        return sc

    def update(self, env, brewer_s, filter_s) -> float:
        tips = self._six_hand_points(env)
        return self.evaluate_states(brewer_s, filter_s, tips)

    def score(self) -> float:
        return float(self.last_score)

    def step(self, env) -> float:
        brewer_s = self.lk._part_state(env, self.lk._SPEC_BY_NAME["coffee_maker"])
        filter_s = self.lk._part_state(env, self.lk._SPEC_BY_NAME["permanent_filter"])
        return self.update(env, brewer_s, filter_s)
