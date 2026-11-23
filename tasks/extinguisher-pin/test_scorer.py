"""Scorer tests for extinguisher-pin (pure Python; the simulator modules are stubbed so the PRODUCTION functions
task._reset, task._track, task._clearance and task.score run unchanged on synthetic state).
Run:  python test_scorer.py"""
import math
import pathlib
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _stub():
    mods = {}
    lk = types.ModuleType("labkit")
    lk.Joint = lambda *a, **k: types.SimpleNamespace(a=a, k=k, name=a[0] if a else k.get("name"))
    lk.Mesh = lambda p, t, c, col="none": types.SimpleNamespace(points=np.asarray(p, float), tris=t, color=c, collide=col)
    lk.part = lambda *a, **k: types.SimpleNamespace(a=a, k=k, name=a[0])
    lk.on_reset = lambda f: f
    lk.on_step = lambda f: f
    lk._np = np.asarray
    lk.TIES = {}
    lk.SURFACES = {}
    lk.SMOOTH_COLORS = set()
    mods["labkit"] = lk
    mods["arena_assets_fix"] = types.ModuleType("arena_assets_fix")
    sp = types.ModuleType("scipy"); sps = types.ModuleType("scipy.spatial")
    sps.ConvexHull = lambda p: types.SimpleNamespace(simplices=np.zeros((1, 3), int))
    mods["scipy"], mods["scipy.spatial"] = sp, sps
    ya = types.ModuleType("yam_arena")
    d = types.ModuleType("yam_arena.demo"); d.Waypoint = lambda *a, **k: (a, k); d.grip_rotation = lambda y=0.0: np.eye(3)
    c = types.ModuleType("yam_arena.constants"); c.FINGER_OPEN, c.FINGER_CLOSED = -0.047, 0.0
    t = types.ModuleType("yam_arena.task_file"); t.TABLE_TOP = {"station": 0.7475}
    mods.update({"yam_arena": ya, "yam_arena.demo": d, "yam_arena.constants": c, "yam_arena.task_file": t})
    for k, v in mods.items():
        sys.modules.setdefault(k, v)


_stub()
import task as T  # noqa: E402


def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Sim:
    """synthetic state fed to the production _track through the stubbed labkit._part_state and task._pads"""

    def __init__(self):
        self.ex = np.array([T.EX_X, T.EX_Y, T.TOP])
        self.R = np.eye(3)
        self.lever = 0.0
        self.pin = np.array([T.EX_X, T.EX_Y - 0.004, T.TOP + T.Z_PIN])   # seated
        self.Rpin = np.eye(3)
        self.pads = [np.array([[0.21, 0.28, 0.93]]), np.array([[0.21, -0.28, 0.93]])]   # parked
        T.labkit._part_state = self._state
        T._pads = lambda env: self.pads
        T._reset(None)
        T._BRK.clear(); T._BRK["sub"] = None

    def _state(self, env, part):
        if part is T.EXT:
            return types.SimpleNamespace(pos=self.ex.copy(), joints={"lever": self.lever}, rotation=lambda: self.R)
        return types.SimpleNamespace(pos=self.pin.copy(), joints={}, rotation=lambda: self.Rpin)

    def step(self, n=1):
        for _ in range(n):
            T._track(None)
        return T.score(None)

    def pull_out_to_bench(self):
        """the pin slides out along -y, then lies on the bench beside the extinguisher"""
        for k in range(40):
            self.pin = self.pin + np.array([0.0, -0.0025, 0.0])
            self.step()
        self.pin = np.array([T.EX_X, T.EX_Y - 0.15, T.TOP + T.PIN_R])
        self.Rpin = np.eye(3)


class T1(unittest.TestCase):
    def test_untouched_is_zero(self):
        s = Sim()
        self.assertEqual(s.step(200), 0.0)

    def test_clean_extraction_is_one(self):
        s = Sim(); s.step(20); T._BRK["event_step"] = 21; s.pull_out_to_bench()
        self.assertEqual(s.step(30), 1.0)

    def test_lever_squeeze_latches_even_if_released(self):
        s = Sim(); s.step(20)
        s.lever = math.radians(2.0); s.step(5); s.lever = 0.0
        s.pull_out_to_bench()
        self.assertEqual(s.step(30), 0.0)
        self.assertEqual(T._S["latch"], "lever squeezed")

    def test_shove_and_return_is_zero(self):
        s = Sim(); s.step(20)
        start = s.ex.copy()
        for k in range(20):
            s.ex = start + np.array([0.0, -0.002 * (k + 1), 0.0]); s.step()     # 40 mm shove
        s.ex = start.copy()                                                      # and back
        s.pull_out_to_bench()
        self.assertEqual(s.step(30), 0.0)

    def test_small_slide_is_allowed(self):
        s = Sim(); s.step(20); T._BRK["event_step"] = 21
        s.ex = s.ex + np.array([0.0, -0.005, 0.0]); s.step()
        s.pull_out_to_bench()
        self.assertEqual(s.step(30), 1.0)

    def test_lean_and_return_is_zero(self):
        s = Sim(); s.step(20)
        s.R = _Rx(math.radians(8.0)); s.step(10); s.R = np.eye(3)
        s.pull_out_to_bench()
        self.assertEqual(s.step(30), 0.0)

    def test_tip_over_latches(self):
        s = Sim(); s.step(20)
        s.R = _Rx(math.radians(30.0)); s.step()
        self.assertEqual(T._S["latch"], "extinguisher tipped")

    def test_pin_held_in_air_is_zero(self):
        s = Sim(); s.step(20)
        for k in range(40):
            s.pin = s.pin + np.array([0.0, -0.0025, 0.0]); s.step()
        self.assertEqual(s.step(30), 0.0)        # out but not on the bench

    def test_pin_started_with_seal_broken_is_half(self):
        s = Sim(); s.step(20)
        T._BRK["event_step"] = 21                              # PhysX reported the tie broken
        s.pin = s.pin + np.array([0.0, -0.010, 0.0])
        self.assertEqual(s.step(40), 0.5)
        self.assertTrue(T._S["seal"])

    def test_withdraw_then_reseat_is_zero(self):
        s = Sim(); s.step(20)
        T._BRK["event_step"] = 21
        s.pin = s.pin + np.array([0.0, -0.006, 0.0]); s.step(10)
        s.pin = s.pin + np.array([0.0, 0.006, 0.0])            # pushed fully back in
        self.assertEqual(s.step(40), 0.0)

    def test_pin_moved_without_break_event_is_zero(self):
        s = Sim(); s.step(20)
        s.pin = s.pin + np.array([0.0, -0.010, 0.0])           # moved, but no seal-break event
        self.assertEqual(s.step(40), 0.0)
        self.assertFalse(T._S["seal"])

    def test_rotating_stationary_pin_is_zero(self):
        s = Sim(); s.step(20)
        T._BRK["event_step"] = 21
        c, s_ = math.cos(0.5), math.sin(0.5)
        s.Rpin = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])   # turned about its axis, not withdrawn
        self.assertEqual(s.step(40), 0.0)

    def test_pin_barely_moved_is_zero(self):
        s = Sim(); s.step(20)
        s.pin = s.pin + np.array([0.0, -0.003, 0.0])          # seal broken but under 5 mm of travel
        self.assertEqual(s.step(40), 0.0)

    def test_half_credit_lost_if_lever_squeezed(self):
        s = Sim(); s.step(20)
        s.pin = s.pin + np.array([0.0, -0.010, 0.0]); s.lever = math.radians(3.0); s.step(); s.lever = 0.0
        self.assertEqual(s.step(40), 0.0)

    def test_pin_off_bench_latches(self):
        s = Sim(); s.step(20); s.pull_out_to_bench()
        s.pin = np.array([T.EX_X, T.EX_Y - 0.15, T.TOP - 0.05]); s.step()
        self.assertEqual(T._S["latch"], "pin dropped off the bench")

    def test_hand_near_cylinder_surface_is_zero(self):
        """a pad 100 mm from the cylinder axis is only about 59 mm from its surface: not clear"""
        s = Sim(); s.step(20); s.pull_out_to_bench()
        s.pads = [np.array([[T.EX_X, T.EX_Y + 0.100, T.TOP + 0.15]]), np.array([[0.21, -0.28, 0.93]])]
        self.assertEqual(s.step(30), 0.0)

    def test_clearance_is_to_surfaces(self):
        p_ex, R = np.array([0.0, 0.0, 0.0]), np.eye(3)
        far_pin = np.array([5.0, 5.0, 5.0])
        d = T._clearance(np.array([0.0, 0.100, 0.10]), p_ex, R, far_pin, np.eye(3))
        self.assertAlmostEqual(d, 0.100 - T.R_BODY, places=6)

    def _ring_point(self, q):
        """world centre of the ring for hinge angle q, pin seated at the origin (as task._clearance computes it)"""
        c, s_ = math.cos(q), math.sin(q)
        return np.array([0.0, -T.PIN_L / 2 - 0.0016 + (T.RING_R + 0.0016) * s_, -(T.RING_R + 0.0016) * c])

    def test_ring_clearance_hanging(self):
        T._S["ring_q"] = 0.0
        far = np.array([5.0, 5.0, 5.0])
        d = T._clearance(self._ring_point(0.0), far, np.eye(3), np.zeros(3), np.eye(3))
        self.assertLess(d, 0.002)                    # a pad at the hanging ring's centre is on the ring

    def test_ring_clearance_raised(self):
        far = np.array([5.0, 5.0, 5.0])
        T._S["ring_q"] = -math.pi                    # raised above the pin head
        hang = self._ring_point(0.0)
        self.assertGreater(T._clearance(hang, far, np.eye(3), np.zeros(3), np.eye(3)), 0.010)
        self.assertLess(T._clearance(self._ring_point(-math.pi), far, np.eye(3), np.zeros(3), np.eye(3)), 0.002)
        T._S["ring_q"] = 0.0

    def test_motion_at_end_is_zero(self):
        s = Sim(); s.step(20); s.pull_out_to_bench()
        for k in range(30):
            s.pin = s.pin + np.array([0.0005 * (-1) ** k, 0.0, 0.0]) * 10; s.step()
        self.assertEqual(T.score(None), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
