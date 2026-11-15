"""Geometric end-state scoring for qsfp-bail-unlatch."""
import math
from itertools import product
import numpy as np

SIGNS = np.asarray(list(product([-1.0, 1.0], repeat=3)))


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def box_rot(box) -> np.ndarray:
    yaw, pitch = box[3]
    R = np.eye(3)
    if abs(pitch) > 1e-12:
        R = ry(pitch) @ R
    if abs(yaw) > 1e-12:
        R = rz(yaw) @ R
    return R


def box_sample(box, n: int = 4) -> np.ndarray:
    c = np.asarray(box[0], float)
    h = np.asarray(box[1], float) / 2.0
    R = box_rot(box)
    g = np.linspace(-1.0, 1.0, n)
    pts = [SIGNS * h]
    for ax in range(3):
        u, v = [i for i in range(3) if i != ax]
        for side in (-1.0, 1.0):
            for a in g:
                for b in g:
                    q = np.zeros(3)
                    q[ax], q[u], q[v] = side * h[ax], a * h[u], b * h[v]
                    pts.append(q[None, :])
    return np.vstack(pts) @ R.T + c


def box_corners(box) -> np.ndarray:
    c = np.asarray(box[0], float)
    h = np.asarray(box[1], float) / 2.0
    R = box_rot(box)
    return (SIGNS * h) @ R.T + c


def spec_corners(sp) -> np.ndarray:
    return np.vstack([box_corners(x) for x in sp.boxes])


def spec_surface(sp) -> np.ndarray:
    return np.vstack([box_sample(x, n=4) for x in sp.boxes])


def boxdist_many(pts: np.ndarray, box) -> np.ndarray:
    c = np.asarray(box[0], float)
    h = np.asarray(box[1], float) / 2.0
    R = box_rot(box)
    local = (pts - c) @ R
    d = np.maximum(np.abs(local) - h, 0.0)
    return np.linalg.norm(d, axis=-1)


class Metric:
    def __init__(self, ns):
        self.lk = ns["labkit"]
        self.top = float(ns["TOP"])
        self.sw_spec = ns["SWITCH_RACK"]
        self.tx_spec = ns["TRANSCEIVER"]
        self.cage_front_local_x = float(ns["CAGE_FRONT_LOCAL_X"])

        self.lp_joint = next(j for j in self.sw_spec.joints if j.name == "left_pawl")
        self.rp_joint = next(j for j in self.sw_spec.joints if j.name == "right_pawl")
        self.tab_joint = next(j for j in self.tx_spec.joints if j.name == "pull_tab")

        self.sw_corners = spec_corners(self.sw_spec)
        self.sw_surf = spec_surface(self.sw_spec)
        self.lp_corners = spec_corners(self.lp_joint)
        self.lp_surf = spec_surface(self.lp_joint)
        self.rp_corners = spec_corners(self.rp_joint)
        self.rp_surf = spec_surface(self.rp_joint)

        self.tx_body_corners = spec_corners(self.tx_spec)
        self.tx_body_surf = spec_surface(self.tx_spec)
        self.tab_corners = spec_corners(self.tab_joint)
        self.tab_surf = spec_surface(self.tab_joint)
        self.reset()

    def reset(self):
        self.hist = []
        self.value = 0.0
        self.detail = {}

    def _switch_world_points(self, sw_state, surface: bool = False) -> np.ndarray:
        ql = float(sw_state.joints.get("left_pawl", 0.0))
        qr = float(sw_state.joints.get("right_pawl", 0.0))
        b = self.sw_surf if surface else self.sw_corners
        lp = (self.lp_surf if surface else self.lp_corners) + np.array([0.0, ql, 0.0])
        rp = (self.rp_surf if surface else self.rp_corners) + np.array([0.0, qr, 0.0])
        return np.vstack([b, lp, rp]) @ sw_state.rotation().T + np.asarray(sw_state.pos, float)

    def _tx_world_points(self, tx_state, surface: bool = False, body_only: bool = False) -> np.ndarray:
        qt = float(tx_state.joints.get("pull_tab", 0.0))
        b = self.tx_body_surf if surface else self.tx_body_corners
        if body_only:
            local = b
        else:
            tb = (self.tab_surf if surface else self.tab_corners) + np.array([qt, 0.0, 0.0])
            local = np.vstack([b, tb])
        return local @ tx_state.rotation().T + np.asarray(tx_state.pos, float)

    def _dist_to_switch(self, world_pts: np.ndarray, sw_state) -> float:
        q = (world_pts - np.asarray(sw_state.pos, float)) @ sw_state.rotation()
        dists = [boxdist_many(q, b) for b in self.sw_spec.boxes]
        ql = float(sw_state.joints.get("left_pawl", 0.0))
        dists += [boxdist_many(q - np.array([0.0, ql, 0.0]), b) for b in self.lp_joint.boxes]
        qr = float(sw_state.joints.get("right_pawl", 0.0))
        dists += [boxdist_many(q - np.array([0.0, qr, 0.0]), b) for b in self.rp_joint.boxes]
        return float(np.min(np.stack(dists, axis=0)))

    def _dist_to_tx(self, world_pts: np.ndarray, tx_state) -> float:
        q = (world_pts - np.asarray(tx_state.pos, float)) @ tx_state.rotation()
        dists = [boxdist_many(q, b) for b in self.tx_spec.boxes]
        qt = float(tx_state.joints.get("pull_tab", 0.0))
        dists += [boxdist_many(q - np.array([qt, 0.0, 0.0]), b) for b in self.tab_joint.boxes]
        return float(np.min(np.stack(dists, axis=0)))

    def evaluate_states(self, sw_state, tx_state, finger_tips_world: np.ndarray, hist_override=None):
        all_corners = np.vstack([
            self._switch_world_points(sw_state, surface=False),
            self._tx_world_points(tx_state, surface=False),
        ])
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

        sw_surf = self._switch_world_points(sw_state, surface=True)
        sw_min_z = float(sw_surf[:, 2].min())
        sw_tilt_deg = math.degrees(math.acos(np.clip(float(sw_state.rotation()[2, 2]), -1.0, 1.0)))
        sw_upright = abs(sw_min_z - self.top) <= 0.003 and sw_tilt_deg <= 5.0

        tx_surf = self._tx_world_points(tx_state, surface=True)
        tx_min_z = float(tx_surf[:, 2].min())
        tx_on_counter = abs(tx_min_z - self.top) <= 0.003

        # Rear-most point of transceiver zinc body in switch local X relative to cage front face:
        # Initially rear end sits at x_mod_max = +0.036 -> +50.0 mm inside cage behind cage front!
        # Extracted >= 20 mm past cage opening means entire zinc body is <= cage_front_local_x - 0.020 m (or withdrawn >= 20 mm from initial seated depth).
        tx_body_world = self._tx_world_points(tx_state, surface=True, body_only=True)
        tx_in_sw = (tx_body_world - np.asarray(sw_state.pos, float)) @ sw_state.rotation()
        tx_rear_rel_cage_m = float(tx_in_sw[:, 0].max()) - self.cage_front_local_x
        # Initial seated rear end is +0.0600 m behind cage front; withdrawn distance along -X:
        withdrawn_m = 0.0600 - tx_rear_rel_cage_m

        sw_tx_clear_m = min(
            self._dist_to_switch(tx_surf, sw_state),
            self._dist_to_tx(sw_surf, tx_state),
        )

        tips = np.asarray(finger_tips_world, float)
        hand_clear_m = min(
            self._dist_to_switch(tips, sw_state),
            self._dist_to_tx(tips, tx_state),
        )

        qt = float(tx_state.joints.get("pull_tab", 0.0))
        common = still and hand_clear_m >= 0.080 and sw_upright
        if not common:
            val = 0.0
        elif tx_on_counter and sw_tx_clear_m >= 0.035 and tx_rear_rel_cage_m <= -0.020:
            val = 1.0
        elif withdrawn_m >= 0.025:
            val = 0.5
        else:
            val = 0.0

        self.value = val
        self.detail = dict(
            score=val,
            tab_mm=round(-qt * 1000.0, 2),
            withdrawn_mm=round(withdrawn_m * 1000.0, 2),
            rear_rel_cage_mm=round(tx_rear_rel_cage_m * 1000.0, 2),
            sw_tx_clear_mm=round(sw_tx_clear_m * 1000.0, 2),
            tx_z_mm=round((tx_min_z - self.top) * 1000.0, 2),
            sw_tilt_deg=round(sw_tilt_deg, 2),
            hands_mm=round(hand_clear_m * 1000.0, 2),
            max_move_mm=round(max_move_mm, 3),
            still=bool(still),
        )
        return val

    def update(self, env, sw_state, tx_state):
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
        return self.evaluate_states(sw_state, tx_state, np.asarray(tips, float))

    def score(self) -> float:
        return float(self.value)
