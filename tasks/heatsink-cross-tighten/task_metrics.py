"""Pure final-state 3D pose & fastener-travel scorer for heatsink-cross-tighten (Alphacool Eisblock XPX Pro 1U P/N 12936)."""
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
        self.ihs_face_x = float(ns["IHS_FACE_X"])
        self.sock_z_w = float(ns["SOCK_Z_W"])
        self.cp_start_x = float(ns["CP_START_X"])
        self.cp_seated_x = float(ns["CP_SEATED_X"])
        self.seated_travel = float(ns["SEATED_TRAVEL"])
        self.corner_yz = tuple((float(y), float(z)) for y, z in ns["CORNER_YZ"])

        self.fixture_boxes = list(ns["FIXTURE_BOXES"])
        for j in ns["FIXTURE_JOINTS"]:
            self.fixture_boxes.extend(list(j.boxes))
        self.cp_boxes = list(ns["CP_BOXES"])

        # 4 Square-ILM bracket probe points on the 80.0 x 80.0 mm M4 stud square (y = +-0.040, z = +-0.040, x_c = -0.0139):
        self.ear_pts_local = np.array(
            [
                [-0.0139, +0.0400, +0.0400],
                [-0.0139, -0.0400, -0.0400],
                [-0.0139, +0.0400, -0.0400],
                [-0.0139, -0.0400, +0.0400],
            ],
            float,
        )
        # 8 outer corners of COLD_PLATE (96.0 x 96.0 mm bracket outline):
        self.cp_corners = np.array(
            [[x, sy * 0.0480, sz * 0.0480] for x in (-0.0155, +0.0120) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            float,
        )
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

    def moving_surface_points(self, pf: np.ndarray, Rf: np.ndarray, pc: np.ndarray, Rc: np.ndarray, j_map: dict) -> np.ndarray:
        """World-frame points on ALL manipulated/moving bodies: 8 cold-plate corners + 8 rotating nut-rib tips + 4 advancing collars."""
        cp_w = self.cp_corners @ Rc.T + pc
        nut_and_cap_local = []
        for i, (yc, zc) in enumerate(self.corner_yz):
            q_rot = float(j_map.get(f"nut_rot_{i}", 0.0))
            q_cap = float(j_map.get(f"nut_cap_{i}", 0.0))
            r_x = rx(q_rot)
            anchor = np.array([-0.0545, yc, zc], float)
            for dy, dz in ((+0.0092, 0.0), (-0.0092, 0.0)):
                nut_and_cap_local.append(anchor + r_x @ np.array([0.0, dy, dz], float))
            nut_and_cap_local.append(np.array([-0.0412 + q_cap, yc, zc + 0.0048], float))
        nc_w = np.asarray(nut_and_cap_local, float) @ Rf.T + pf
        return np.vstack([cp_w, nc_w])

    def evaluate_states(self, sb_s, cp_s, tips_6: np.ndarray, hist_override=None) -> float:
        pf = np.asarray(sb_s.pos, float)
        Rf = sb_s.rotation()
        pc = np.asarray(cp_s.pos, float)
        Rc = cp_s.rotation()
        j_map = getattr(sb_s, "joints", None) or {}

        cp_w = self.cp_corners @ Rc.T + pc
        off_table = float(np.min(cp_w[:, 2])) < (self.top - 0.030)

        all_pts_w = self.moving_surface_points(pf, Rf, pc, Rc, j_map)
        if hist_override is not None:
            hist = hist_override
        else:
            self.hist.append(all_pts_w)
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

        cap_adv_mm = [float(j_map.get(f"nut_cap_{i}", 0.0)) * 1000.0 for i in range(4)]
        rot_turns = [float(j_map.get(f"nut_rot_{i}", 0.0)) / (2.0 * math.pi) for i in range(4)]

        ears_w = self.ear_pts_local @ Rc.T + pc
        ear_x_mm = [float(v) * 1000.0 for v in ears_w[:, 0]]
        corner_spread_mm = float(max(ear_x_mm) - min(ear_x_mm))

        # Cold-plate copper base center (+0.0120 in local X) gap to IHS front (-X) plane:
        base_center_w = pc + Rc @ np.array([+0.0120, 0.0, 0.0])
        ihs_gap_mm = float((self.ihs_face_x - base_center_w[0]) * 1000.0)
        yz_err_mm = float(math.hypot(base_center_w[1], base_center_w[2] - self.sock_z_w) * 1000.0)
        plate_nx = float(Rc[0, 0])  # +1.0 when parallel to IHS

        # Also transform dynamically rotated nut ribs and shifted collars for exact 6-point hand clearance:
        dyn_fix_boxes = list(self.fixture_boxes)
        for i, (yc, zc) in enumerate(self.corner_yz):
            q_cap = float(j_map.get(f"nut_cap_{i}", 0.0))
            dyn_fix_boxes.append(((-0.0412 + q_cap, yc, zc + 0.0048), (0.0024, 0.0130, 0.0030), (0, 0, 0), 0.0))
            dyn_fix_boxes.append(((-0.0412 + q_cap, yc, zc - 0.0048), (0.0024, 0.0130, 0.0030), (0, 0, 0), 0.0))

        d_min = min(
            dist_pts_to_boxes(tips_6, dyn_fix_boxes, pf, Rf),
            dist_pts_to_boxes(tips_6, self.cp_boxes, pc, Rc),
        )
        hand_clear_mm = float(d_min * 1000.0)
        hands_ok = hand_clear_mm >= 80.0

        valid_common = (
            (not off_table)
            and (yz_err_mm <= 6.0)
            and (plate_nx >= 0.990)
            and hands_ok
            and still_ok
        )

        # Pure final-state seating check:
        # 1.0 requires actual copper-to-IHS contact (ihs_gap_mm in [-0.20, +0.24] mm, matching PhysX's ~0.172 mm contact-skin
        # equilibrium at seated spring preload while rejecting a separated 0.40 mm air gap), all 4 nut collars advanced >= 2.20 mm
        # (within 0.20 mm of 2.40 mm full travel), and 80.0 x 80.0 mm Square-ILM four-corner spread <= 0.45 mm.
        all_seated = all(c >= 2.20 for c in cap_adv_mm)
        full_10 = (
            valid_common
            and all_seated
            and (-0.20 <= ihs_gap_mm <= 0.24)
            and (corner_spread_mm <= 0.45)
        )

        # 0.5: partial uniform compression (>= 2 nuts advanced >= 0.90 mm, gap in [-0.20, +1.65] mm, 80x80 mm spread <= 0.90 mm).
        n_half = sum(1 for c in cap_adv_mm if c >= 0.90)
        partial_05 = (
            valid_common
            and (n_half >= 2)
            and (-0.20 <= ihs_gap_mm <= 1.65)
            and (corner_spread_mm <= 0.90)
        )

        if full_10:
            sc = 1.0
        elif partial_05:
            sc = 0.5
        else:
            sc = 0.0

        self.last_score = sc
        self.detail = {
            "score": sc,
            "ihs_gap_mm": round(ihs_gap_mm, 3),
            "corner_spread_mm": round(corner_spread_mm, 3),
            "cap_adv_mm": [round(c, 2) for c in cap_adv_mm],
            "rot_turns": [round(t, 2) for t in rot_turns],
            "yz_err_mm": round(yz_err_mm, 2),
            "hand_clear_mm": round(hand_clear_mm, 2),
            "hands_ok": hands_ok,
            "still_disp_mm": round(still_disp_mm, 3),
            "still_ok": still_ok,
        }
        return sc

    def update(self, env, sb_s, cp_s) -> float:
        tips_6 = self._six_hand_points(env)
        return self.evaluate_states(sb_s, cp_s, tips_6)

    def score(self) -> float:
        return float(self.last_score)
