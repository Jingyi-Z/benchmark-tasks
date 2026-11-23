"""Scorer tests (numpy-free, no simulator): python test_scorer.py -> ALL_PASS."""
import math
import prism_scorer as S

D = math.radians
FIX = {"laser": (0.15, 0, 0.88, 0.0), "screen": (0.79, 0, 0.88, 0.0), "post_A": (0.25, 0, 0.8, 0.0), "post_B": (0.29, 0, 0.8, 0.0)}


START = (D(30.0), D(-30.0))


def best_single(which):
    """closest approach with one ring free and the other left at its fixed start"""
    best = (9.0, None)
    for i in range(0, 7200):
        x = D(i * 0.05)
        m = S.miss_m(x, START[1]) if which == "A" else S.miss_m(START[0], x)
        best = min(best, (m, x))
    return best


def best_pair():
    best = (9.0, None, None)
    for i in range(0, 720):
        for j in range(0, 720):
            a, b = D(i * 0.5), D(j * 0.5)
            m = S.miss_m(a, b)
            if m < best[0]:
                best = (m, a, b)
    return best


def run(a, b, fixed=None, hands=0.3, beam=0.05, n=25, jitter=0.0, moved=None, beam_until=None, glass_at=None):
    T = S.Tracker()
    for i in range(n):
        bm = 0.004 if (beam_until is not None and i < beam_until) else beam
        f = dict(FIX if fixed is None else fixed)
        if moved and i > 5:
            k, dx = moved
            x, y, z, t = f[k]; f[k] = (x + dx, y, z, t)
        T.step(dict(a=a + (D(jitter) if i % 2 else 0.0), b=b, fixed=f, hands_clear_m=hands, beam_clear_m=bm,
                    glass_touch=(glass_at is not None and i == glass_at)))
    return T.score(), T.latch


mA, aA = best_single("A")
mB, bB = best_single("B")
m0, a0, b0 = best_pair()


def near(miss_mm):
    """a pair of angles giving a miss close to miss_mm (search around the best pair)"""
    best = None
    for i in range(0, 720):
        a = a0 + D(i * 0.05 - 18)
        m = S.miss_m(a, b0) * 1e3
        if best is None or abs(m - miss_mm) < abs(best[0] - miss_mm):
            best = (m, a)
    return best


m35, a35 = near(2.5)
m8, a8 = near(8.0)
CASES = [
    (f"both prisms set (miss {m0 * 1e3:.2f} mm)", run(a0, b0), 1.0),
    (f"A alone at its best, B at its start (miss {mA * 1e3:.2f} mm)", run(aA, START[1]), 0.0),
    (f"B alone at its best, A at its start (miss {mB * 1e3:.2f} mm)", run(START[0], bB), 0.0),
    (f"start pose (miss {S.miss_m(*START) * 1e3:.1f} mm)", run(*START), 0.0),
    (f"miss {m35:.2f} mm (half band)", run(a35, b0), 0.5),
    (f"miss {m8:.2f} mm", run(a8, b0), 0.0),
    ("laser nudged 3 mm", run(a0, b0, moved=("laser", 0.003)), 0.0),
    ("screen moved 3 mm", run(a0, b0, moved=("screen", 0.003)), 0.0),
    ("hand 50 mm from a mount", run(a0, b0, hands=0.05), 0.0),
    ("hand in the beam", run(a0, b0, beam=0.004), 0.0),
    ("no hand observation", run(a0, b0, hands=None), 0.0),
    ("ring still turning", run(a0, b0, jitter=0.2), 0.0),
    ("hand in the beam until the last sample", run(a0, b0, beam_until=24), 0.0),
    ("a pad touched the glass once, early, then everything right (latched)", run(a0, b0, glass_at=2), 0.0),
    ("hand on the bent final segment, clear of the straight axis",
     run(a0, b0, beam=S.beam_clear([(S.beam_offset(0.30, a0, b0)[0], 0.30, S.beam_offset(0.30, a0, b0)[1] + 0.003)], a0, b0)), 0.0),
    ("hand on the straight axis but clear of the bent beam",
     run(a0, b0, beam=S.beam_clear([(0.0, 0.30, 0.0)], a0, b0)), 1.0),
    ("too few samples for 1 s still", run(a0, b0, n=10), 0.0),
]

if __name__ == "__main__":
    bad = 0
    for name, (got, latch), want in CASES:
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} got {got} want {want} latch={latch}  {name}")
    print(f"single-prism best: A {mA * 1e3:.2f} mm, B {mB * 1e3:.2f} mm (must exceed 1.5 x the 3.0 mm half band = 4.5 mm)")
    print("ALL_PASS" if bad == 0 else f"{bad} FAILED")
