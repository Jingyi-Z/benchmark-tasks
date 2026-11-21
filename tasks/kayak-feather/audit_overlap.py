"""CPU collider overlap audit for ferrule_proto shapes (claude-ebeam). Samples points inside every RIGHT-half primitive
near the ferrule (0.4 mm grid), places the right half at (feather angle about +y, axial offset dy), and counts points
inside any LEFT-half primitive (main body + catch joints at a given stroke). Box rotation = USD RotateY(pitch) after
RotateZ(yaw), as labkit authors it. Expected:
  seated at a legal index (0, 15, 30, -60): no overlap (keys in slots; hooks in the groove behind the clip lip)
  seated off-index (7.5 deg): overlap (keys hit slot walls) => cannot seat there
  pulled out by engage + 2 mm: no overlap at ANY angle (free rotation)
  pulled by 3 mm with hooks at rest, or with ONE tab pressed: the lip overlaps a hook => locked
  pulled by 3 mm with BOTH tabs pressed flush: no hook overlap => released (v2 mechanism, US 8,801,483 B2)"""
import math
import sys

import numpy as np

import ferrule_proto as FP


def Ry(p):
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def box_inside(P, c, size, turn, tol=0.0):
    yaw, pitch = turn if isinstance(turn, (tuple, list)) else (turn, 0.0)
    R = Rz(yaw) @ Ry(pitch)
    q = (P - np.asarray(c)) @ R          # into the box frame
    return np.all(np.abs(q) < np.asarray(size) / 2 - tol, axis=1)


def cyl_inside(P, c, r, h, axis, tol=0.0):
    i = "XYZ".index(axis)
    q = P - np.asarray(c)
    rad = np.sqrt(np.sum(np.delete(q, i, axis=1) ** 2, axis=1))
    return (rad < r - tol) & (np.abs(q[:, i]) < h / 2 - tol)


def sample_box(c, size, turn, step):
    yaw, pitch = turn if isinstance(turn, (tuple, list)) else (turn, 0.0)
    ax = [np.arange(-s / 2 + step / 2, s / 2, step) if s > step else np.array([0.0]) for s in size]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return G @ (Rz(yaw) @ Ry(pitch)).T + np.asarray(c)


def sample_cyl(c, r, h, axis, step, ylim):
    i = "XYZ".index(axis)
    lo, hi = c[i] - h / 2, c[i] + h / 2
    lo, hi = max(lo, ylim[0]), min(hi, ylim[1])
    if hi <= lo:
        return np.zeros((0, 3))
    g = np.arange(-r, r + 1e-9, step)
    U, V = np.meshgrid(g, g)
    m = U ** 2 + V ** 2 < (r - step / 2) ** 2
    U, V = U[m], V[m]
    out = []
    for a in np.arange(lo + step / 2, hi, step):
        P = np.zeros((len(U), 3))
        P[:, i] = a
        j, k = [n for n in range(3) if n != i]
        P[:, j] = c[j] + U
        P[:, k] = c[k] + V
        out.append(P)
    return np.vstack(out)


def audit(d, feather, dy, stroke=(0.0, 0.0), step=0.0004, tol=0.0001):
    sh = FP.shapes(d, feather)
    L, R = sh["left"], sh["right"]
    ylim = (-0.07, 0.07)
    pts = [sample_box(c, s, t, step) for c, s, _, *rest in [(b[0], b[1], b[2], b[3] if len(b) > 3 else 0.0) for b in R["boxes"]] for t in [rest[0]]]
    pts += [sample_cyl(c, r, h, a, step, ylim) for c, r, h, _, a in R["cylinders"]]
    P = np.vstack([p for p in pts if len(p)])
    P = P + np.array([0.0, dy, 0.0])
    hits = {}
    lb = [(b[0], b[1], b[3] if len(b) > 3 else 0.0) for b in L["boxes"]]
    for k, (c, s, t) in enumerate(lb):
        n = int(box_inside(P, c, s, t, tol).sum())
        if n:
            hits["sleeve/cog box %d" % k] = n
    for c, r, h, _, a in L["cylinders"]:
        n = int(cyl_inside(P, c, r, h, a, tol).sum())
        if n:
            hits["left cyl %s" % (c,)] = n
    for j, st in zip(L["joints"], stroke):
        sx = 1 if j["name"].endswith("px") else -1
        off = np.array([-sx * st, 0, 0])            # pressing moves the plunger inward (toward the axis)
        for c, s, _ in [(b[0], b[1], b[2]) for b in j["boxes"]]:
            n = int(box_inside(P, np.asarray(c) + off, s, 0.0, tol).sum())
            if n:
                hits["%s hook/arm" % j["name"]] = n
    return hits


if __name__ == "__main__":
    ok = True
    for which in ("lo", "hi"):
        d = FP.dims(which)
        e = d["engage"]
        cases = [
            ("seated 0 deg", 0.0, 0.0, (0, 0), False),
            ("seated +15 deg", 15.0, 0.0, (0, 0), False),
            ("seated +30 deg (target)", 30.0, 0.0, (0, 0), False),
            ("seated -60 deg (last index)", -60.0, 0.0, (0, 0), False),
            ("seated +7.5 deg (off index)", 7.5, 0.0, (0, 0), True),
            ("seated +75 deg (past the index set)", 75.0, 0.0, (0, 0), True),
            ("pulled engage+2 mm, at 7.5 deg (free turn)", 7.5, e + 0.002, (d["protrude"], d["protrude"]), False),
            ("pulled 3 mm, catches at rest (locked)", 0.0, 0.003, (0, 0), True),
            ("pulled 3 mm, ONE catch pressed (still locked)", 0.0, 0.003, (d["protrude"], 0), True),
            ("pulled 3 mm, BOTH pressed (catches clear)", 0.0, 0.003, (d["protrude"], d["protrude"]), False),
        ]
        print("== %s bound" % which)
        for name, f, dy, st, expect_hit in cases:
            h = audit(d, f, dy, st)
            # for the lock cases only the catch-tooth hits count; for index cases only sleeve/cog hits
            if "pulled 3 mm" in name:
                hit = any("hook" in k for k in h)
            else:
                hit = bool(h)
            good = hit == expect_hit
            ok &= good
            print("  %s %-46s overlap=%s %s" % ("PASS" if good else "FAIL", name, hit, dict(list(h.items())[:4])))
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
