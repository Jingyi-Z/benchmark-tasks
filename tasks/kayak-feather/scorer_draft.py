"""kayak-feather end-state scorer draft (claude-ebeam). Pure numpy; no sim. Inputs are the settled world poses of the two
paddle halves (origin at each half's ferrule face, local +y along the shaft toward that half's blade, local +x the blade face
normal) and whether the halves are still joined (axial gap at the ferrule).

Feather angle: the signed rotation of the RIGHT blade's face normal relative to the LEFT blade's, about the shaft axis
pointing toward the right blade. Right-hand control = positive (the right blade turned 'top toward the paddler').
The convention is a model choice to be checked against the maker's marking before trials.

Rule (end state, every frame of a 21-frame still window):
  seated : axial gap at the ferrule face <= 1.0 mm and the two shaft axes within 2 deg (straight)
  angle  : feather within +-5 deg of +30 (index positions are 15 deg apart, so +-5 deg never accepts a neighbour)
  free   : both halves resting on the saddles, no hand within 80 mm (from the shared hand cover, as in drawer-peg)
  score  : 1.0 seated + angle + free; 0.5 seated + free at another legal index other than the 0 deg start (re-indexed, wrong setting);
           0 otherwise (unseated, separated, dropped, held, or a non-index angle)."""
import math

import numpy as np

GAP_MAX = 1.0e-3
STRAIGHT_DEG = 2.0
TARGET, TOL = 30.0, 5.0
INDEX = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
START = 0          # the paddle starts assembled at 0 deg (unfeathered)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("degenerate vector")
    return v / n


def feather_deg(R_left, R_right):
    """Signed angle (deg) of the right blade normal about the right half's shaft axis, relative to the left blade normal."""
    a = _unit(np.asarray(R_right)[:, 1])            # shaft axis toward the right blade
    nL = np.asarray(R_left)[:, 0]
    nR = np.asarray(R_right)[:, 0]
    nL = _unit(nL - a * (nL @ a))
    nR = _unit(nR - a * (nR @ a))
    return math.degrees(math.atan2(float(np.cross(nL, nR) @ a), float(nL @ nR)))


def score_frame(R_left, R_right, gap_m, hands_clear_m):
    try:
        RL, RR = np.asarray(R_left, float).reshape(3, 3), np.asarray(R_right, float).reshape(3, 3)
        if not (np.all(np.isfinite(RL)) and np.all(np.isfinite(RR)) and math.isfinite(gap_m) and math.isfinite(hands_clear_m)):
            raise ValueError("nonfinite")
        straight = math.degrees(math.acos(max(-1.0, min(1.0, float(-RL[:, 1] @ RR[:, 1])))))
        ang = feather_deg(RL, RR)
    except ValueError as e:
        return 0.0, "invalid input: %s" % e
    if hands_clear_m < 0.080:
        return 0.0, "held: a hand within 80 mm"
    if gap_m > GAP_MAX or straight > STRAIGHT_DEG:
        return 0.0, "not seated: gap %.1f mm, bend %.1f deg" % (gap_m * 1e3, straight)
    if abs(ang - TARGET) <= TOL:
        return 1.0, "seated at %.1f deg right-hand feather" % ang
    near = min(INDEX, key=lambda k: abs(ang - k))
    if near == START and abs(ang - near) <= TOL:
        return 0.0, "seated at the starting index %+d deg: unchanged" % near
    if abs(ang - near) <= TOL:
        return 0.5, "seated at the wrong index %+d deg (%.1f)" % (near, ang)
    return 0.0, "seated off any index (%.1f deg): not a legal Posi-Lok position" % ang


if __name__ == "__main__":
    def half(feather, left):
        """left half: shaft axis -y, blade normal +z; right half: axis +y, normal = +z turned by `feather` about +y."""
        if left:
            n, y = np.array([0, 0, 1.0]), np.array([0, -1.0, 0])
        else:
            t = math.radians(feather)
            R_about_y = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
            n, y = R_about_y @ np.array([0, 0, 1.0]), np.array([0, 1.0, 0])
        return np.column_stack([n, y, np.cross(n, y)])
    L = half(0, True)
    cases = [
        ("K1 seated at +30 (target)", half(30, False), 0.0002, 0.15, 1.0),
        ("K2 seated at 0 (start, untouched)", half(0, False), 0.0002, 0.15, 0.0),
        ("K3 seated at -30 (left-hand: wrong sign)", half(-30, False), 0.0002, 0.15, 0.5),
        ("K4 seated at +45 (overshoot one index)", half(45, False), 0.0002, 0.15, 0.5),
        ("K5 at +30 but UNLOCKED: 6 mm gap", half(30, False), 0.006, 0.15, 0.0),
        ("K6 at +30, seated, still held (hand 40 mm)", half(30, False), 0.0002, 0.040, 0.0),
        ("K7 seated off index at +22 (forced between teeth)", half(22, False), 0.0002, 0.15, 0.0),
        ("K8 nonfinite rotation", np.full((3, 3), np.nan), 0.0002, 0.15, 0.0),
    ]
    ok = True
    for name, R, gap, clear, exp in cases:
        s, why = score_frame(L, R, gap, clear)
        ok &= s == exp
        print("%s %-52s score %.2f (expect %.2f): %s" % ("PASS" if s == exp else "FAIL", name, s, exp, why))
    print("feather(+30) reads %.2f deg" % feather_deg(L, half(30, False)))
    print("ALL PASS" if ok else "FAILURES")
