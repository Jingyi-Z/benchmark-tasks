"""Wedge-prism spot scorer (pure Python; task.py feeds one observation per step; test_scorer.py drives it directly).

Spot model (stated in fidelity): thin-prism deviation. Each prism deviates the beam by DEV toward its base direction
(its ring angle; 0 = base up, +z; positive = right-handed about the beam axis +y, which turns +z toward +x). The second
prism acts on the beam already deviated by the first, at the real 50 mm spacing (positions along the beam axis):
    spot(a, b) = dA * (SB - SA) + (dA + dB) * (SS - SB),   dX = DEV * (sin x, cos x) in (u = x, v = z) on the screen.
Layout v2: L = SS - SB = 0.35 m, so everything is scaled by 0.7 from the 500 mm layout.

Score, from the final state (checkable with a ruler on a photo of the screen):
  1.0  spot centre within 2.0 mm of the target centre (inside the 4 mm printed ring)
  0.5  within 3.0 mm (still needs both prisms: from the fixed start either one alone cannot get closer than the
       one-prism bests printed by test_scorer.py, both well over 4.5 mm)
  0    otherwise
  and, for any credit, on every sample of the last 1 s: both hands at least 80 mm from both mounts and nothing in the
  bent beam (no pad tip within 8 mm of it), both rings still (under 0.05 deg change over the last 1 s).
Latched 0: any pad tip inside either mount clear aperture (on the glass); the laser, screen, target card or either
mount post moved more than 2 mm or tilted more than 1 deg from its
start (only the two rings may move).
"""
import math

DEV = math.radians(2.0)
SA, SB, SS = -0.05, 0.0, 0.35   # on the breadboard's 25 mm hole grid
TARGET = (0.0, 0.01225)
R_FULL, R_HALF = 0.0020, 0.0030
CLEAR_M, BEAM_M = 0.080, 0.008
STILL_N, STILL_DEG = 21, 0.05
MOVE_M, TILT_DEG = 0.002, 1.0


def spot_yz(a, b):
    dA = (DEV * math.sin(a), DEV * math.cos(a))
    dB = (DEV * math.sin(b), DEV * math.cos(b))
    return (dA[0] * (SB - SA) + (dA[0] + dB[0]) * (SS - SB), dA[1] * (SB - SA) + (dA[1] + dB[1]) * (SS - SB))


def beam_offset(y, a, b):
    """(u, v) offset of the bent beam from the straight axis at position y along the axis (y measured like SA/SB/SS):
    straight up to A, deviated by A between A and B, deviated by A + B after B"""
    dAu, dAv = DEV * math.sin(a), DEV * math.cos(a)
    if y <= SA:
        return 0.0, 0.0
    if y <= SB:
        return dAu * (y - SA), dAv * (y - SA)
    du, dv = dAu + DEV * math.sin(b), dAv + DEV * math.cos(b)
    return dAu * (SB - SA) + du * (y - SB), dAv * (SB - SA) + dv * (y - SB)


def beam_clear(points, a, b, y0=-0.13):
    """smallest distance (m) from any point (u, y, v) -- u, v relative to the axis -- to the bent beam between the laser
    aperture (y0) and the screen"""
    out = 9.0
    for u, y, v in points:
        if y0 < y < SS:
            bu, bv = beam_offset(y, a, b)
            out = min(out, math.hypot(u - bu, v - bv))
    return out


def miss_m(a, b):
    y, z = spot_yz(a, b)
    return math.hypot(y - TARGET[0], z - TARGET[1])


class Tracker:
    def __init__(self):
        self.start = None
        self.latch = None
        self.hist = []

    def step(self, ob):
        """ob: dict(a, b (rad), fixed: {name: (x, y, z, tilt_deg)}, hands_clear_m, beam_clear_m) (None = not observed)"""
        if self.latch:
            return
        if ob.get("glass_touch"):
            self.latch = "pad on the glass (inside a mount clear aperture)"
            return
        if self.start is None:
            self.start = dict(ob["fixed"])
        for k, (x, y, z, t) in ob["fixed"].items():
            x0, y0, z0, t0 = self.start[k]
            if math.dist((x, y, z), (x0, y0, z0)) > MOVE_M or abs(t - t0) > TILT_DEG:
                self.latch = f"{k} moved"
                return
        self.hist.append(ob)
        self.hist = self.hist[-STILL_N:]

    def score(self):
        if self.latch or len(self.hist) < STILL_N:
            return 0.0
        ob = self.hist[-1]
        a_s = [math.degrees(o["a"]) for o in self.hist]; b_s = [math.degrees(o["b"]) for o in self.hist]
        if max(a_s) - min(a_s) > STILL_DEG or max(b_s) - min(b_s) > STILL_DEG:
            return 0.0
        for o in self.hist:                     # hands clear and beam unblocked on EVERY sample of the 1 s window
            hc, bc = o["hands_clear_m"], o["beam_clear_m"]
            if hc is None or bc is None or hc < CLEAR_M or bc < BEAM_M:
                return 0.0
        m = miss_m(ob["a"], ob["b"])
        return 1.0 if m <= R_FULL else (0.5 if m <= R_HALF else 0.0)
