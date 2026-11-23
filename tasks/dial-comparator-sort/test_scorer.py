"""Scorer tests for the dial comparator sort (pure Python: simulator modules and scipy are stubbed; the production
evaluate(), under_contact(), tray_of(), zero_ok() and reading_mm() in task.py are called unchanged).
Run:  python test_scorer.py"""
import math
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class _Obj:
    def __init__(self, *a, **k):
        self.__dict__.update(k)


def _part(name, **k):
    return types.SimpleNamespace(name=name, **k)


_stub("arena_assets_fix")
_stub("labkit", Joint=_Obj, Mesh=lambda *a, **k: None, part=_part, on_reset=lambda f: f, on_step=lambda f: f,
      SURFACES={}, _np=np.asarray, _part_state=None, CONTACT_OFFSET=0.0005)
_stub("scipy"); _stub("scipy.spatial", ConvexHull=lambda p: types.SimpleNamespace(simplices=np.zeros((1, 3))))
_stub("yam_arena")
_stub("yam_arena.demo", Waypoint=lambda *a, **k: (a, k), grip_rotation=lambda yaw=0.0: np.eye(3))
_stub("yam_arena.task_file", TABLE_TOP={"station": 0.7475})
_stub("yam_arena.constants", FINGER_OPEN=-0.04695, FINGER_CLOSED=0.0)
sys.path.insert(0, str(HERE))
import task as T  # noqa: E402


class St:                                       # a rigid pose like labkit._part_state's
    def __init__(self, x, y, z, yaw=0.0):
        self.pos = np.array([x, y, z]); self._yaw = yaw

    def rotation(self):
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


BZ0 = 2 * math.pi * T.ZERO_GAP * 1e3          # bezel angle that zeroes the dial on the master (arm clamped at 0)


def J(arm=0.0, spindle=None, bezel=BZ0):
    spindle = T.FREE - T.ZERO_GAP if spindle is None else spindle   # contact resting on the master
    return {"arm": arm, "spindle": spindle, "bezel": bezel, "needle": 0.0}


def parts_at(under=None, on_anvil_h=None):
    """gauge block and spacers on the bench row; optionally one item under the contact at the right height"""
    out = {"gb": (St(0.33, T.YP, T.TOP), 0.025, (0.009, 0.035))}
    for i, k in enumerate(T.SPACER_H):
        out[k] = (St(0.38 + 0.05 * i, T.YP, T.TOP), T.SPACER_H[k], (0.010, T.SL))
    if under:
        st, h, dims = out[under]
        out[under] = (St(T.XS, T.YS - 0.012, T.ANV), h, dims)
    return out


def STATE(**over):
    s = dict(zero_ok=True, zero_on="gb", measured={"s1", "s2", "s3"},
             trays={"gb": None, "s1": "rework", "s2": "pass", "s3": "rework"}, latch=None, clear=0.30, still=True,
             gb_home=True)
    s.update(over)
    return s


CASES = []


def case(name, expect):
    def deco(f):
        CASES.append((name, expect, f))
        return f
    return deco


@case("all three sorted right, zero set on the gauge block, measured, clear, still", 1.0)
def _():
    return T.evaluate(STATE())


@case("good spacer put in REWORK", 0.5)
def _():
    return T.evaluate(STATE(trays={"gb": None, "s1": "rework", "s2": "rework", "s3": "rework"}))


@case("everything in PASS (sorting without reading): only s2 right", 0.0)
def _():
    return T.evaluate(STATE(trays={"gb": None, "s1": "pass", "s2": "pass", "s3": "pass"}))


@case("trays right but s1 never measured", 0.5)
def _():
    return T.evaluate(STATE(measured={"s2", "s3"}))


@case("trays right but nothing measured (memorised sort)", 0.0)
def _():
    return T.evaluate(STATE(measured=set()))


@case("zero set while a spacer was under the contact", 0.0)
def _():
    return T.evaluate(STATE(zero_on="s2"))


@case("zero never set (still 0.2 mm off)", 0.0)
def _():
    return T.evaluate(STATE(zero_ok=False))


@case("gauge block left in a tray", 0.0)
def _():
    return T.evaluate(STATE(trays={"gb": "pass", "s1": "rework", "s2": "pass", "s3": "rework"}))


@case("a part fell off the bench (latch)", 0.0)
def _():
    return T.evaluate(STATE(latch="s3 fell off the bench"))


@case("hand 60 mm from a part", 0.0)
def _():
    return T.evaluate(STATE(clear=0.06))


@case("not still", 0.0)
def _():
    return T.evaluate(STATE(still=False))


@case("zero_ok with the bezel turned to put 0 under the needle on the master; not 1.2 divisions off", True)
def _():
    return T.zero_ok(J()) and not T.zero_ok(J(bezel=BZ0 + 2 * math.pi * 0.012)) and not T.zero_ok(J(bezel=0.0))


@case("dial reads +0.040 mm on the 25.04 spacer after zeroing", True)
def _():
    j = J(spindle=T.FREE - T.ZERO_GAP + 0.00004)
    return abs(T.reading_mm(j) - 0.040) < 1e-9


@case("under_contact: gauge block seated under the tip, contact on its top", True)
def _():
    j = J()
    P = parts_at(under="gb")
    tip_top = T.TIP0 + j["arm"] + j["spindle"]
    st, h, d = P["gb"]
    P["gb"] = (St(T.XS, T.YS - 0.012, tip_top - h), h, d)
    return T.under_contact(P, j) == "gb"


@case("under_contact: nothing under the tip", True)
def _():
    return T.under_contact(parts_at(), J()) is None


@case("under_contact: block shoved 20 mm aside is not 'under'", True)
def _():
    j = J()
    P = parts_at()
    tip_top = T.TIP0 + j["arm"] + j["spindle"]
    P["gb"] = (St(T.XS + 0.020, T.YS - 0.012, tip_top - 0.025), 0.025, (0.009, 0.035))
    return T.under_contact(P, j) is None


@case("tray_of: spacer resting in the PASS tray / on the bench / lifted above the tray", True)
def _():
    tx, ty = T.TRAYS["pass"]
    return (T.tray_of(St(tx, ty, T.TOP + 0.003)) == "pass" and T.tray_of(St(0.40, -0.13, T.TOP)) is None
            and T.tray_of(St(tx, ty, T.TOP + 0.05)) is None)


@case("spacer truth: s2 good, s1 and s3 rework (25.00 +- 0.02 mm)", True)
def _():
    return T.GOOD == {"s1": False, "s2": True, "s3": False}


@case("blind sorters across the 8 seeds: any fixed tray assignment is fully right on exactly 1 seed (1/8)", True)
def _():
    import itertools
    hits = []
    for assign in itertools.product(("pass", "rework"), repeat=3):
        n = 0
        for seed in range(8):
            h = [T.GOOD_H[i] if seed >> i & 1 else T.BAD_H[i] for i in range(3)]
            good = [abs(x - 0.025) <= T.TOL for x in h]
            n += all((a == "pass") == g for a, g in zip(assign, good))
        hits.append(n)
    return hits == [1] * 8


@case("seed heights: every good height within 25.00 +- 0.02, every bad one outside", True)
def _():
    return all(abs(x - 0.025) <= T.TOL for x in T.GOOD_H) and all(abs(x - 0.025) > T.TOL for x in T.BAD_H)


@case("master left on the bare bench (not in the parts tray): 0", 0.0)
def _():
    return T.evaluate(STATE(gb_home=False))


@case("missing hand data counts as a hand at the parts (clearance 0): 0", 0.0)
def _():
    return T.evaluate(STATE(clear=0.0))


@case("under_contact: tip 0.30 mm above the master's top (air gap) is not contact", True)
def _():
    j = J()
    P = parts_at()
    tip_top = T.TIP0 + j["arm"] + j["spindle"]
    P["gb"] = (St(T.XS, T.YS - 0.012, tip_top - 0.025 - 0.0003), 0.025, (0.009, 0.035))
    return T.under_contact(P, j) is None


@case("under_contact: lever held down (-0.18 rad) is not a resting contact", True)
def _():
    j = dict(J(), lever=-0.18)
    P = parts_at()
    tip_top = T.TIP0 + j["arm"] + j["spindle"]
    P["gb"] = (St(T.XS, T.YS - 0.012, tip_top - 0.025), 0.025, (0.009, 0.035))
    return T.under_contact(P, j) is None


@case("reading on the seated master before zeroing (bezel untouched) shows -0.060 mm", True)
def _():
    return abs(T.reading_mm(J(bezel=0.0)) + T.ZERO_GAP * 1e3) < 1e-9 and abs(T.ZERO_GAP - 0.00006) < 1e-12


@case("measured: 10 alternating contact/miss ticks do not count; reads before a valid zero do not count", True)
def _():
    S = {}
    for i in range(10):
        T.update_measured(S, "s2" if i % 2 == 0 else None, True, False)
    a = "s2" not in S.get("measured", set())
    S = {}
    for i in range(20):
        T.update_measured(S, "s2", False, False)
    b = "s2" not in S.get("measured", set())
    S = {}
    for i in range(10):
        T.update_measured(S, "s2", True, False)
    c = "s2" in S["measured"]
    T.update_measured(S, None, True, True)                    # recalibration wipes readings
    return a and b and c and not S["measured"]


@case("tray_of: spacer turned 90 deg still inside; one 5 mm over the rim is not", True)
def _():
    tx, ty = T.TRAYS["pass"]
    inside = T.tray_of(St(tx, ty, T.TOP + 0.003, yaw=math.pi / 2)) == "pass"
    over = T.tray_of(St(tx + 0.045, ty, T.TOP + 0.003)) is None
    return inside and over


@case("in_parts_tray: master back on the tray floor yes; on the bench at (0.50, 0.10) no", True)
def _():
    return T.in_parts_tray(St(0.33, T.YP, T.TOP + T.PT_FLOOR)) and not T.in_parts_tray(St(0.50, 0.10, T.TOP))


@case("under_contact: tip 0.04 mm above the part (inside the old 0.05 mm window) is not contact", True)
def _():
    j = J()
    P = parts_at()
    tip_top = T.TIP0 + j["arm"] + j["spindle"]
    P["s2"] = (St(T.XS, T.YS - 0.012, tip_top - T.SPACER_H["s2"] - 0.00004), T.SPACER_H["s2"], (0.010, T.SL))
    return T.under_contact(P, j) is None


@case("two calibrations: the second wipes the first's readings (read cache and buffers too)", True)
def _():
    S = {}
    for i in range(10):
        T.update_measured(S, "s1", True, False)
    S["read"] = {"s1": 0.040}; S["rdbuf"] = {"s1": [0.04] * 10}
    T.update_measured(S, None, True, True)
    wiped = not S["measured"] and not S["read"] and not S["rdbuf"]
    for i in range(10):
        T.update_measured(S, "s1", True, False)
    return wiped and "s1" in S["measured"]


@case("master home only in its own pocket: in a spacer pocket (x 0.38) it is not home", True)
def _():
    return T.in_parts_tray(St(0.33, T.YP, T.TOP + T.PT_FLOOR)) and not T.in_parts_tray(St(0.38, T.YP, T.TOP + T.PT_FLOOR))


@case("master bump latch: arms at rest even with a hand near; a 326 -> 335.6 mm shove latches; 1 mm does not; 4 deg tilt does", True)
def _():
    S = {}
    for _ in range(10):
        T.check_master_bump(S, True, (0.326, -0.168), 0.0)     # returned, at rest (hand may be near)
    armed = S.get("gb_rest") is not None
    shove = T.check_master_bump(dict(S), True, (0.3356, -0.168), 0.0) is not None
    small = T.check_master_bump(dict(S), True, (0.327, -0.168), 0.0) is None
    tilt = T.check_master_bump(dict(S), True, (0.326, -0.168), 4.0) is not None
    return armed and shove and small and tilt


@case("zero judged by the needle on the master: +0.0095 read under the contact counts; +0.011 does not; none read = not set", True)
def _():
    return (T.zero_ok(J(), {"zero_rd": 0.0095}) and not T.zero_ok(J(), {"zero_rd": 0.011})
            and not T.zero_ok(J(), {"zero_rd": None}) and not T.zero_ok(J(), {}))


@case("a zero 1 division off cannot flip any sort: every good height reads within 0.02, every bad one outside, at +-0.01 offset", True)
def _():
    ok = True
    for off in (-0.010, 0.0, 0.010):
        ok &= all(abs((h - 0.025) * 1e3 + off) <= T.TOL * 1e3 + 1e-9 for h in T.GOOD_H)
        ok &= all(abs((h - 0.025) * 1e3 + off) > T.TOL * 1e3 for h in T.BAD_H)
    return ok


@case("held zero: the master under the contact but a hand still on the bezel does not set the zero", True)
def _():
    S = {}
    T.zero_update(S, "gb", False, False, 0.0, 0.3)
    held = S.get("zero_rd") is None
    T.zero_update(S, "gb", False, True, 0.0, 0.3)
    return held and S["zero_rd"] == 0.0


@case("slow drift: 0.2 deg of bezel turned in tiny steps after the master left voids the zero", True)
def _():
    S = {}
    T.zero_update(S, "gb", False, True, 0.0, 0.3)
    voided = False
    for k in range(1, 41):                                   # 40 ticks of 0.005 deg each: never a per-tick "move"
        voided |= T.zero_update(S, None, False, True, 0.0, 0.3 + math.radians(0.005 * k))
    return voided and S["zero_rd"] is None


@case("lifted master: raised 5 mm and set back in the same XY after its return latches the bump", True)
def _():
    S = {}
    for _ in range(10):
        T.check_master_bump(S, True, (0.326, -0.168, 0.749), 0.0)
    return T.check_master_bump(dict(S), True, (0.326, -0.168, 0.754), 0.0) is not None


def main():
    bad = 0
    for name, expect, f in CASES:
        got = f()
        ok = (got == expect) if isinstance(expect, bool) else abs(float(got) - expect) < 1e-9
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"{name}: got {got}, expected {expect}")
    print(f"{len(CASES) - bad}/{len(CASES)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
