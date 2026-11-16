"""Pure 3D world-pose geometric scorer for power-strip-keyhole-mount (Eaton Tripp Lite TLP606)."""
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


def _axis_rot(axis: str, rad: float) -> np.ndarray:
    a = axis.upper()
    if a == "X":
        return rx(rad)
    if a == "Y":
        return ry(rad)
    return rz(rad)


def box_rot(b) -> np.ndarray:
    turn = b[3] if len(b) > 3 else 0.0
    yaw, pitch = (float(turn[0]), float(turn[1])) if isinstance(turn, (tuple, list)) else (float(turn), 0.0)
    return rz(yaw) @ ry(pitch)


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


class Metric:
    def __init__(self, ns: dict):
        self.top = float(ns["TOP"])
        self.riser_face_x = float(ns["RISER_FACE_X"])
        self.riser_half_y = float(ns["RISER_HALF_Y"])
        self.stud_world_pts = np.asarray(ns["STUD_HEAD_CENTERS_W"], float)
        self.keyhole_centers_xy = np.asarray(ns["KEYHOLE_BULB_XY_S"], float)

        self.riser_boxes = list(ns["RISER_BOXES"])
        self.strip_boxes = list(ns["STRIP_BOXES"])
        # Track the 8 outer corners of the 47.0 x 276.0 x 33.0 mm TLP606 housing for stillness & bounds:
        self.strip_corners = np.array(
            [[sx * 0.0235, sy * 0.1380, z] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (0.0, 0.0330)],
            float,
        )
        self.strip_joints = list(ns["POWER_STRIP"].joints) if "POWER_STRIP" in ns else []
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

    def _dist_to_articulated_strip_links(self, world_pts: np.ndarray, ps: np.ndarray, Rs: np.ndarray, joints_map: dict) -> float:
        best = float("inf")
        frames = {"": (ps, Rs, np.zeros(3))}
        for j in self.strip_joints:
            p_par, R_par, a_par = frames[j.parent]
            a_j = np.asarray(j.anchor, float)
            q_j = float(joints_map.get(j.name, j.target))
            p_j = p_par + R_par @ (a_j - a_par)
            R_j = R_par @ _axis_rot(j.axis, q_j)
            frames[j.name] = (p_j, R_j, a_j)
            if j.boxes:
                shifted = [((np.asarray(b[0], float) - a_j).tolist(), b[1], b[2], b[3]) for b in j.boxes]
                best = min(best, dist_pts_to_boxes(world_pts, shifted, p_j, R_j))
        return best

    def evaluate_states(self, riser_s, strip_s, tips_6: np.ndarray, hist_override=None) -> float:
        pr = np.asarray(riser_s.pos, float)
        Rr = riser_s.rotation()
        ps = np.asarray(strip_s.pos, float)
        Rs = strip_s.rotation()

        strip_w = self.strip_corners @ Rs.T + ps
        min_z = float(np.min(strip_w[:, 2]))
        off_table = min_z < (self.top - 0.030)
        suspended_ok = min_z >= (self.top + 0.060)

        if hist_override is not None:
            hist = hist_override
        else:
            self.hist.append(strip_w)
            if len(self.hist) > 21:
                self.hist.pop(0)
            hist = self.hist

        still_disp_mm = 999000.0
        still_ok = False
        if len(hist) >= 21:
            h_arr = np.stack(hist[-21:], axis=0)
            disp = h_arr[:, None, :, :] - h_arr[None, :, :, :]
            still_disp_mm = float(np.max(np.linalg.norm(disp, axis=-1)) * 1000.0)
            still_ok = still_disp_mm < 1.0

        backplate_normal_w = -Rs[:, 2]
        flush_normal_ok = float(backplate_normal_w[0]) >= 0.965
        rear_x_max = float(np.max(strip_w[:, 0]))
        rear_gap_m = self.riser_face_x - float(ps[0])
        within_board_y = (float(np.min(strip_w[:, 1])) >= float(pr[1]) - self.riser_half_y) and (
            float(np.max(strip_w[:, 1])) <= float(pr[1]) + self.riser_half_y
        )

        stud_s = (self.stud_world_pts - ps) @ Rs
        matched_slots_05 = set()
        matched_slots_10 = set()
        slide_travels_mm = []
        for s_idx in range(2):
            xs, ys, zs = float(stud_s[s_idx, 0]), float(stud_s[s_idx, 1]), float(stud_s[s_idx, 2])
            in_cavity_z = (0.0030 <= zs <= 0.0108)
            best_travel = -999.0
            for k_idx, (xk, yk) in enumerate(self.keyhole_centers_xy):
                dx_k = abs(xs - float(xk))
                travel_m = float(yk) - ys
                if in_cavity_z and (dx_k <= 0.0068) and (-0.0065 <= travel_m <= 0.0165):
                    matched_slots_05.add((s_idx, k_idx))
                    best_travel = max(best_travel, travel_m * 1000.0)
                if in_cavity_z and (dx_k <= 0.0035) and (0.0085 <= travel_m <= 0.0160):
                    matched_slots_10.add((s_idx, k_idx))
                    best_travel = max(best_travel, travel_m * 1000.0)
            slide_travels_mm.append(round(best_travel, 2))

        engaged_05 = (
            any(s == 0 for s, _ in matched_slots_05)
            and any(s == 1 for s, _ in matched_slots_05)
            and len({k for _, k in matched_slots_05}) >= 2
        )
        locked_10 = (
            any(s == 0 for s, _ in matched_slots_10)
            and any(s == 1 for s, _ in matched_slots_10)
            and len({k for _, k in matched_slots_10}) >= 2
        )

        j_map = getattr(strip_s, "joints", None) or {}
        d_min = min(
            dist_pts_to_boxes(tips_6, self.riser_boxes, pr, Rr),
            dist_pts_to_boxes(tips_6, self.strip_boxes, ps, Rs),
            self._dist_to_articulated_strip_links(tips_6, ps, Rs, j_map),
        )
        hand_clear_mm = float(d_min * 1000.0)
        hands_ok = hand_clear_mm >= 80.0

        valid_posture = (
            (not off_table)
            and suspended_ok
            and within_board_y
            and flush_normal_ok
            and (rear_x_max <= self.riser_face_x + 0.0020)
            and (-0.0015 <= rear_gap_m <= 0.0045)
            and hands_ok
            and still_ok
        )

        if valid_posture and locked_10:
            sc = 1.0
        elif valid_posture and engaged_05:
            sc = 0.5
        else:
            sc = 0.0

        self.last_score = sc
        self.detail = {
            "score": sc,
            "off_table": off_table,
            "min_z_above_top_mm": round((min_z - self.top) * 1000.0, 2),
            "rear_gap_mm": round(rear_gap_m * 1000.0, 2),
            "back_nx": round(float(backplate_normal_w[0]), 4),
            "slide_travels_mm": slide_travels_mm,
            "engaged_05": engaged_05,
            "locked_10": locked_10,
            "hand_clear_mm": round(hand_clear_mm, 2),
            "hands_ok": hands_ok,
            "still_disp_mm": round(still_disp_mm, 3),
            "still_ok": still_ok,
        }
        return sc

    def update(self, env, riser_s, strip_s) -> float:
        tips_6 = self._six_hand_points(env)
        return self.evaluate_states(riser_s, strip_s, tips_6)

    def score(self) -> float:
        return float(self.last_score)
