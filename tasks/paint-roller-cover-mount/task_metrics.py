"""Pure 3D world-pose geometric scorer for paint-roller-cover-mount."""
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


def _pt_seg_dist_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom < 1e-14 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    d = p - (a + t * ab)
    return float(np.dot(d, d))


def _pt_tri_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    nn = float(np.dot(n, n))
    if nn > 1e-16:
        # Project onto triangle plane and test barycentric sign:
        q = p - (float(np.dot(p - a, n)) / nn) * n
        c0 = float(np.dot(np.cross(b - a, q - a), n))
        c1 = float(np.dot(np.cross(c - b, q - b), n))
        c2 = float(np.dot(np.cross(a - c, q - c), n))
        if c0 >= -1e-12 and c1 >= -1e-12 and c2 >= -1e-12:
            return float(np.linalg.norm(p - q))
    return math.sqrt(min(_pt_seg_dist_sq(p, a, b), _pt_seg_dist_sq(p, b, c), _pt_seg_dist_sq(p, c, a)))


def dist_pts_to_meshes(world_pts: np.ndarray, meshes, pos: np.ndarray, R: np.ndarray) -> float:
    if not meshes:
        return float("inf")
    rel = (world_pts - pos) @ R
    best = float("inf")
    for m in meshes:
        pts = np.asarray(m.points, float)
        tris = np.asarray(m.tris, int)
        for p in rel:
            for ia, ib, ic in tris:
                best = min(best, _pt_tri_dist(p, pts[ia], pts[ib], pts[ic]))
    return best


class Metric:
    def __init__(self, ns: dict):
        self.top = float(ns["TOP"])
        self.cage_len = float(ns["CAGE_LEN"])      # 0.2286 m (9.0 in)
        self.cover_len = float(ns["COVER_LEN"])    # 0.2286 m (9.0 in)
        self.detent_anchor = np.asarray(ns["DETENT_ANCHOR"], float)

        self.frame_only_boxes = list(ns["FRAME_BOXES"])
        self.frame_cyls = list(ns["FRAME_CYLS"])
        self.cage_only_boxes = list(ns["CAGE_BOXES"])
        self.cage_cyls = list(ns["CAGE_CYLS"])
        self.detent_only_boxes = list(ns["DETENT_BOXES"])
        self.detent_meshes = list(ns.get("DETENT_MESHES", ()))
        self.cover_only_boxes = list(ns["COVER_BOXES"])

        self.frame_corners = part_local_corners(self.frame_only_boxes + cyl_as_boxes(self.frame_cyls))
        self.cage_corners = part_local_corners(self.cage_only_boxes + cyl_as_boxes(self.cage_cyls))
        det_pts = [part_local_corners(self.detent_only_boxes)] if self.detent_only_boxes else []
        det_pts.extend(np.asarray(m.points, float) for m in self.detent_meshes)
        self.detent_corners = np.vstack(det_pts)
        self.cover_corners = part_local_corners(self.cover_only_boxes)
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

    def link_poses(self, frame_s):
        p_f = np.asarray(frame_s.pos, float)
        R_f = frame_s.rotation()
        spin = float(frame_s.joints.get("cage_spin", 0.0))
        dz_det = float(frame_s.joints.get("detent_z", 0.0))

        # cage_spin rotates about Y at anchor (0, 0, 0):
        R_c = R_f @ ry(spin)
        p_c = p_f

        # detent_z slides along Z of parent cage_spin:
        R_d = R_c
        p_d = p_c + R_c @ np.array([0.0, 0.0, dz_det])
        return p_f, R_f, p_c, R_c, p_d, R_d, spin, dz_det

    def evaluate_states(self, frame_s, cover_s, tip_pts_world: np.ndarray, hist_override=None) -> float:
        p_f, R_f, p_c, R_c, p_d, R_d, _, dz_det = self.link_poses(frame_s)
        p_v = np.asarray(cover_s.pos, float)
        R_v = cover_s.rotation()

        fr_w = self.frame_corners @ R_f.T + p_f
        cg_w = self.cage_corners @ R_c.T + p_c
        dt_w = self.detent_corners @ R_d.T + p_d
        cv_w = self.cover_corners @ R_v.T + p_v

        all_pts = np.vstack([fr_w, cg_w, dt_w, cv_w])
        off_table = float(np.min(all_pts[:, 2])) < self.top - 0.030

        # Track non-spinning frame root + cover corners for 1.0 s stillness (plus cage corners):
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
            dist_pts_to_boxes(tip_pts_world, self.frame_only_boxes, p_f, R_f),
            dist_pts_to_cyls(tip_pts_world, self.frame_cyls, p_f, R_f),
            dist_pts_to_boxes(tip_pts_world, self.cage_only_boxes, p_c, R_c),
            dist_pts_to_cyls(tip_pts_world, self.cage_cyls, p_c, R_c),
            dist_pts_to_boxes(tip_pts_world, self.detent_only_boxes, p_d, R_d),
            dist_pts_to_meshes(tip_pts_world, self.detent_meshes, p_d, R_d),
            dist_pts_to_boxes(tip_pts_world, self.cover_only_boxes, p_v, R_v),
        )
        hands_ok = hand_clear >= 0.080

        # Pure 3D coaxial measurements between ROLLER_FRAME cage axis and ROLLER_COVER bore axis:
        # In ROLLER_FRAME local coords: shoulder stop face is at y=0.0, cage spans y in [-0.2286, 0.0] along +Y/-Y.
        u_cage = R_f[:, 1]   # unit vector pointing toward inboard shoulder (+Y_frame)
        u_cov = R_v[:, 1]    # cover cylinder axis (symmetric end-for-end: either end may enter first!)
        cos_ax = abs(float(np.clip(np.dot(u_cage, u_cov), -1.0, 1.0)))
        axis_angle_deg = float(math.degrees(math.acos(cos_ax)))

        # Express cover center and both end-centers in ROLLER_FRAME local coordinates:
        half_L = 0.5 * self.cover_len
        end_a_w = p_v + half_L * u_cov
        end_b_w = p_v - half_L * u_cov
        end_a_f = R_f.T @ (end_a_w - p_f)
        end_b_f = R_f.T @ (end_b_w - p_f)
        ctr_f = R_f.T @ (p_v - p_f)

        # Inboard end of cover is whichever end has larger y_f (closer to shoulder y_f = 0.0):
        if end_a_f[1] >= end_b_f[1]:
            inb_f, out_f = end_a_f, end_b_f
        else:
            inb_f, out_f = end_b_f, end_a_f

        # Radial offset = max perpendicular distance (in X_f, Z_f) of both cover ends from cage centerline (0, y, 0):
        rad_inb = float(math.hypot(inb_f[0], inb_f[2]))
        rad_out = float(math.hypot(out_f[0], out_f[2]))
        radial_offset = max(rad_inb, rad_out)

        # Overlap of cover [out_f[1], inb_f[1]] with cage span [-cage_len, 0.0]:
        ov_lo = max(-self.cage_len, float(out_f[1]))
        ov_hi = min(0.0020, float(inb_f[1]))
        axial_overlap = max(0.0, ov_hi - ov_lo)

        # Signed gap from cover inboard rim to shoulder stop face (y_f = 0.0):
        shoulder_gap = float(-inb_f[1])

        half_seated = (
            axial_overlap >= 0.1150
            and radial_offset <= 0.0050
            and axis_angle_deg <= 5.0
        )
        full_seated = (
            axial_overlap >= 0.2226
            and -0.0020 <= shoulder_gap <= 0.0060
            and radial_offset <= 0.0040
            and axis_angle_deg <= 4.0
        )

        if off_table or not hands_ok or not still_ok:
            sc = 0.0
        elif full_seated:
            sc = 1.0
        elif half_seated:
            sc = 0.5
        else:
            sc = 0.0

        self.last_score = sc
        self.detail = dict(
            score=sc,
            overlap_mm=round(axial_overlap * 1000.0, 2),
            shoulder_gap_mm=round(shoulder_gap * 1000.0, 2),
            radial_mm=round(radial_offset * 1000.0, 2),
            angle_deg=round(axis_angle_deg, 2),
            detent_z_mm=round(dz_det * 1000.0, 2),
            ctr_f_mm=[round(float(v) * 1000.0, 2) for v in ctr_f],
            hand_clear_mm=round(hand_clear * 1000.0, 2),
            still_mm=round(still_disp * 1000.0, 3),
            off_table=off_table,
        )
        return sc

    def update(self, env, frame_s, cover_s) -> float:
        tips = self._six_hand_points(env)
        return self.evaluate_states(frame_s, cover_s, tips)

    def score(self) -> float:
        return float(self.last_score)
