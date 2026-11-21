"""codex 0704 sweep (claude-ebeam): every stock index and every halfway point, extraction stroke 0..14 mm, three button
states, at both inferred bounds; the separated 0..30 deg turn; re-engagement at the seat. Points sampled once per
bound (0.4 mm grid), then rigidly moved, so the sweep is exact to the same sampling as audit_overlap.py."""
import math
import sys

import numpy as np

import audit_overlap as A
import ferrule_proto as FP


def prep(d):
    sh = FP.shapes(d, 0.0)
    L, R = sh["left"], sh["right"]
    pts = [A.sample_box(b[0], b[1], b[3] if len(b) > 3 else 0.0, 0.0004) for b in R["boxes"]]
    pts += [A.sample_cyl(c, r, h, a, 0.0004, (-0.07, 0.07)) for c, r, h, _, a in R["cylinders"]]
    P0 = np.vstack([p for p in pts if len(p)])
    return L, P0


def hits(L, P0, d, f_deg, dy, state):
    t = math.radians(f_deg)
    P = P0 @ A.Ry(t).T + np.array([0, dy, 0])
    out = {}
    for k, b in enumerate(L["boxes"]):
        n = int(A.box_inside(P, b[0], b[1], b[3] if len(b) > 3 else 0.0, 0.0001).sum())
        if n:
            out["left key %d" % k] = n
    for c, r, h, _, a in L["cylinders"]:
        n = int(A.cyl_inside(P, c, r, h, a, 0.0001).sum())
        if n:
            out["left %s r%.1f" % ("collar" if r > FP.SHAFT_R else "shaft", r * 1e3)] = n
    press = {"none": (0, 0), "one": (d["protrude"], 0), "both": (d["protrude"], d["protrude"])}[state]
    for j, st in zip(L["joints"], press):
        sx = 1 if j["name"].endswith("px") else -1
        for b in j["boxes"]:
            n = int(A.box_inside(P, np.asarray(b[0]) + np.array([-sx * st, 0, 0]), b[1], 0.0, 0.0001).sum())
            if n:
                out["%s hook/arm" % j["name"]] = n
    return out


ok = True
for which in ("lo", "hi"):
    d = FP.dims(which)
    L, P0 = prep(d)
    e = d["engage"]
    print("== %s bound (engage %.0f mm, hook %.1f mm)" % (which, e * 1e3, d["hook"] * 1e3))
    strokes = [s / 1000 for s in range(0, 15)]
    for f in FP.INDEX + [k + 7.5 for k in FP.INDEX[:-1]]:
        on_index = f in FP.INDEX
        row = []
        for state in ("none", "one", "both"):
            first_block, pair = None, None
            for s in strokes:
                h = hits(L, P0, d, f, s, state)
                hook = {k: v for k, v in h.items() if "hook" in k}
                cog = {k: v for k, v in h.items() if "key" in k}
                if s == 0 and on_index and h:
                    first_block, pair = 0.0, h
                    break
                if hook and first_block is None:
                    first_block, pair = s, hook
                    break
                if not on_index and cog and s < e:
                    first_block, pair = s, cog
                    break
            row.append((state, first_block, pair))
        # expectations
        if on_index:
            good = (row[0][1] is not None and 0 < row[0][1] <= 0.002 and row[1][1] is not None and 0 < row[1][1] <= 0.002
                    and row[2][1] is None)
        else:
            good = all(r[1] is not None and r[1] < e for r in row)     # off-index: cannot even sit there
        ok &= good
        print("  %s f %+6.1f  %s" % ("PASS" if good else "FAIL", f, "  |  ".join(
            "%s: %s" % (st, ("blocked at %.0f mm by %s" % (fb * 1e3, list(p)[:2])) if fb is not None else "clear 0..14 mm")
            for st, fb, p in row)))
    # separated turn: both pressed, pulled engage + 1 mm, 0 -> 30 deg in 1 deg steps
    bad = [f for f in range(0, 31) if hits(L, P0, d, float(f), e + 0.001, "both")]
    good = not bad
    ok &= good
    print("  %s separated 0..30 deg turn at %.0f mm pull, both pressed: collisions at %s" % ("PASS" if good else "FAIL", (e + 0.001) * 1e3, bad or "none"))
    # re-engagement at +30: tabs released, seated -> no overlap (hooks in groove); 1..2 mm out -> hooks hit the lip
    seat = hits(L, P0, d, 30.0, 0.0, "none")
    out1 = {k: v for k, v in hits(L, P0, d, 30.0, 0.0015, "none").items() if "hook" in k}
    good = (not seat) and bool(out1) and ("tab_px hook/arm" in out1) and ("tab_nx hook/arm" in out1)
    ok &= good
    print("  %s re-engaged at +30: seated clean=%s; 1.5 mm pull blocked by BOTH hooks=%s" % ("PASS" if good else "FAIL", not seat, sorted(out1)))
print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
