"""Geometric end-state scoring for rj45-switch-unplug."""
import math
from itertools import product
import numpy as np

SIGNS = np.asarray(list(product([-1.0, 1.0], repeat=3)))


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def box_sample(box, n: int = 5) -> np.ndarray:
    assert box[3] == (0.0, 0.0), "Rotated boxes require explicit transform"
    c = np.asarray(box[0], float)
    h = np.asarray(box[1], float) / 2.0
    g = np.linspace(-1.0, 1.0, n)
    pts = [c + SIGNS * h]
    for ax in range(3):
        u, v = [i for i in range(3) if i != ax]
        for side in (-1.0, 1.0):
            for a in g:
                for b in g:
                    q = np.zeros(3)
                    q[ax], q[u], q[v] = side * h[ax], a * h[u], b * h[v]
                    pts.append(c + q[None, :])
    return np.vstack(pts)


def cyl_sample(cyl) -> np.ndarray:
    c, r, h, _, axis = cyl
    c = np.asarray(c, float)
    assert axis == "Y", "Expected Y-axis strain-relief cylinder"
    angs = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    ys = np.linspace(-h / 2.0, h / 2.0, 5)
    return np.array([c + [r * math.cos(a), y, r * math.sin(a)] for a in angs for y in ys])


def spec_corners(sp) -> np.ndarray:
    chunks = [np.asarray(x[0], float) + SIGNS * (np.asarray(x[1], float) / 2.0) for x in sp.boxes]
    for x in sp.cylinders:
        chunks.append(cyl_sample(x))
    return np.vstack(chunks)


def spec_surface(sp) -> np.ndarray:
    chunks = [box_sample(x, n=4) for x in sp.boxes]
    for x in sp.cylinders:
        chunks.append(cyl_sample(x))
    return np.vstack(chunks)


def boxdist(p: np.ndarray, box) -> float:
    return float(np.linalg.norm(np.maximum(np.abs(p - np.asarray(box[0], float)) - np.asarray(box[1], float) / 2.0, 0.0)))


def boxdist_many(pts: np.ndarray, box) -> np.ndarray:
    d = np.maximum(np.abs(pts - np.asarray(box[0], float)) - np.asarray(box[1], float) / 2.0, 0.0)
    return np.linalg.norm(d, axis=-1)


def cyldist_y(p: np.ndarray, cyl) -> float:
    c, r, h, _, axis = cyl
    assert axis == "Y"
    q = p - np.asarray(c, float)
    return math.hypot(max(math.hypot(q[0], q[2]) - r, 0.0), max(abs(q[1]) - h / 2.0, 0.0))


def cyldist_y_many(pts: np.ndarray, cyl) -> np.ndarray:
    c, r, h, _, axis = cyl
    assert axis == "Y"
    q = pts - np.asarray(c, float)
    rad = np.maximum(np.hypot(q[:, 0], q[:, 2]) - r, 0.0)
    ax = np.maximum(np.abs(q[:, 1]) - h / 2.0, 0.0)
    return np.hypot(rad, ax)


class Metric:
    def __init__(self, ns):
        self.lk = ns["labkit"]
        self.top = float(ns["TOP"])
        self.sw_spec = ns["SWITCH"]
        self.pl_spec = ns["PLUG"]
        self.bezel_y = float(ns["BEZEL_LOCAL_Y"])
        self.tab_joint = next(j for j in self.pl_spec.joints if j.name == "tab")
        self.tab_anchor = np.asarray(self.tab_joint.anchor, float)

        self.sw_corners = spec_corners(self.sw_spec)
        self.sw_surf = spec_surface(self.sw_spec)
        self.pl_body_corners = spec_corners(self.pl_spec)
        self.pl_body_surf = spec_surface(self.pl_spec)
        self.pl_tab_corners = spec_corners(self.tab_joint) - self.tab_anchor
        self.pl_tab_surf = spec_surface(self.tab_joint) - self.tab_anchor
        self.reset()

    def reset(self):
        self.hist = []
        self.value = 0.0
        self.detail = {}

    def _plug_world_points(self, pl_state, surface: bool = False) -> np.ndarray:
        q_tab = float(pl_state.joints.get("tab", 0.0))
        rt = rx(q_tab)
        body_pts = self.pl_body_surf if surface else self.pl_body_corners
        tab_pts = (self.pl_tab_surf if surface else self.pl_tab_corners) @ rt.T + self.tab_anchor
        local = np.vstack([body_pts, tab_pts])
        return local @ pl_state.rotation().T + np.asarray(pl_state.pos, float)

    def _dist_to_switch(self, world_pts: np.ndarray, sw_state) -> float:
        q = (world_pts - np.asarray(sw_state.pos, float)) @ sw_state.rotation()
        d_boxes = [boxdist_many(q, b) for b in self.sw_spec.boxes]
        return float(np.min(np.stack(d_boxes, axis=0)))

    def _dist_to_plug(self, world_pts: np.ndarray, pl_state) -> float:
        q = (world_pts - np.asarray(pl_state.pos, float)) @ pl_state.rotation()
        dists = [boxdist_many(q, b) for b in self.pl_spec.boxes]
        dists += [cyldist_y_many(q, c) for c in self.pl_spec.cylinders]
        q_tab = float(pl_state.joints.get("tab", 0.0))
        qt = (q - self.tab_anchor) @ rx(q_tab) + self.tab_anchor
        dists += [boxdist_many(qt, b) for b in self.tab_joint.boxes]
        return float(np.min(np.stack(dists, axis=0)))

    def evaluate_states(self, sw_state, pl_state, finger_tips_world: np.ndarray, hist_override=None):
        sw_world_corners = self.sw_corners @ sw_state.rotation().T + np.asarray(sw_state.pos, float)
        pl_world_corners = self._plug_world_points(pl_state, surface=False)
        all_corners = np.vstack([sw_world_corners, pl_world_corners])

        if hist_override is not None:
            hist = hist_override
        else:
            self.hist = (self.hist + [all_corners])[-21:]
            hist = self.hist

        still = False
        max_move_mm = 999.0
        if len(hist) == 21:
            h = np.stack(hist, axis=0)
            disp = h[:, None, :, :] - h[None, :, :, :]
            max_move = float(np.max(np.linalg.norm(disp, axis=-1)))
            max_move_mm = max_move * 1000.0
            still = max_move < 0.001

        sw_min_z = float(sw_world_corners[:, 2].min())
        sw_supported = abs(sw_min_z - self.top) <= 0.003
        sw_tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(sw_state.rotation()[2, 2])))))
        sw_upright = sw_tilt_deg <= 5.0

        pl_surf_world = self._plug_world_points(pl_state, surface=True)
        pl_min_z = float(pl_surf_world[:, 2].min())
        pl_supported = abs(pl_min_z - self.top) <= 0.003

        # Express plug surface points in switch local frame to check axial port clearance and 3D separation
        pl_in_sw = (pl_surf_world - np.asarray(sw_state.pos, float)) @ sw_state.rotation()
        outside_bezel_m = self.bezel_y - float(pl_in_sw[:, 1].max())

        sw_surf_world = self.sw_surf @ sw_state.rotation().T + np.asarray(sw_state.pos, float)
        sep_m = min(
            self._dist_to_switch(pl_surf_world, sw_state),
            self._dist_to_plug(sw_surf_world, pl_state),
        )

        tips = np.asarray(finger_tips_world, float)
        hand_clear_m = min(
            self._dist_to_switch(tips, sw_state),
            self._dist_to_plug(tips, pl_state),
        )

        q_tab = float(pl_state.joints.get("tab", 0.0))
        tab_in_bounds = -0.02 <= q_tab <= 0.22

        common = (
            still
            and sw_supported
            and sw_upright
            and pl_supported
            and hand_clear_m >= 0.080
            and tab_in_bounds
            and outside_bezel_m >= 0.010
        )
        if not common:
            val = 0.0
        elif sep_m >= 0.035:
            val = 1.0
        else:
            val = 0.5

        self.value = val
        self.detail = dict(
            score=val,
            outside_bezel_mm=round(outside_bezel_m * 1000.0, 2),
            sep_mm=round(sep_m * 1000.0, 2),
            tab_deg=round(math.degrees(q_tab), 2),
            sw_tilt_deg=round(sw_tilt_deg, 2),
            sw_z_mm=round((sw_min_z - self.top) * 1000.0, 2),
            pl_z_mm=round((pl_min_z - self.top) * 1000.0, 2),
            hands_mm=round(hand_clear_m * 1000.0, 2),
            max_move_mm=round(max_move_mm, 3),
            still=bool(still),
            sw_supported=bool(sw_supported),
            sw_upright=bool(sw_upright),
            pl_supported=bool(pl_supported),
        )
        return val

    def update(self, env, sw_state, pl_state):
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
        return self.evaluate_states(sw_state, pl_state, np.asarray(tips, float))

    def score(self) -> float:
        return float(self.value)
