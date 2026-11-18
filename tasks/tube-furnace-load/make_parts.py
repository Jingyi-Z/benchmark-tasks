"""Print files (STL, mm) and a mass check for this task's cold-trial stand-ins: python make_parts.py [out_dir].

Real trials are cold and carry no powder. The three boats, their captive coloured loads and the push rod are printed
in PETG at the simulation's sizes; the substrate is a 10 x 10 x 0.5 mm stainless square. Sizes are the ones in task.py
(checked against it when task.py can be imported). Each part is written as closed solids that a slicer merges; the
rod's bar and T-handle plate are hollow, with closed 1.0 mm walls, so their printed mass is set by geometry, not by an
infill setting. Print everything solid (100 % infill), 0.2 mm layers, 0.4 mm nozzle.

Writes: boat_precursor.stl (x2), load_yellow.stl, load_white.stl, boat_substrate.stl, push_rod.stl, and prints the
mass table (PETG 1.27 g/cm3, 304 stainless 7.93 g/cm3).
"""
import math
import os
import struct
import sys

import numpy as np

# -- sizes (m), as in task.py --------------------------------------------------------------------------------------
BOAT_W, BOAT_H, BOAT_T = 0.020, 0.012, 0.002   # outside width, height, wall
BOAT_L = {"precursor": 0.060, "substrate": 0.035}
BEVEL = 0.004                                    # underside end edges bevelled 1.2 mm up over 4 mm (task.py: floor 0.8 mm thick at the very end)
TONGUE_L, TONGUE_W, TONGUE_Z = 0.020, 0.005, 0.004
TONGUE_TURN = math.radians(-55.0)               # the tongue's square turned on its own axis
LOAD_GAP = 0.0002                                # the load's clearance inside the walls, each side
LOAD_H = 0.004                                   # the load: 4 mm deep, on the boat's floor
ROD_L, GRIP_W = 0.293, 0.008                     # the bar: 8 mm square; its back end 293 mm behind the paddle's face
RING_T, TH_UP, TH_W = 0.005, 0.045, 0.026        # the T-handle plate: 5 mm thick, 45 mm up, 26 mm wide
WALL = 0.0010                                    # printed wall of the hollow bar and handle plate
SUB, SUB_T = 0.010, 0.0005                       # the substrate square
BALLAST = (0.018, 0.010, 0.0008)                 # 304 stainless plate sealed in the substrate boat's floor
PETG, STEEL = 1270.0, 7930.0                     # kg/m3
SIM_MASS_G = {"boat_precursor+load": 12.0, "boat_substrate": 6.0, "push_rod": 15.0, "substrate": 0.4}


def _box(c, s, rot=None, inward=False):
    """A box as 12 triangles (c centre, s full sizes, rot a 3x3 matrix); inward=True flips it into a cavity."""
    c, h = np.asarray(c, float), np.asarray(s, float) / 2
    R = np.eye(3) if rot is None else rot
    v = [c + R @ (h * np.array([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return _hexa(v, inward)


def _hexa(v, inward=False):
    """A convex hexahedron from 8 corners ordered as (x, y, z) in {-1, 1}^3 (x slowest): 12 triangles, outward."""
    v = [np.asarray(p, float) for p in v]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    cen = sum(v) / 8.0
    tris = []
    for a, b, c, d in quads:
        for t in ((v[a], v[b], v[c]), (v[a], v[c], v[d])):
            n = np.cross(t[1] - t[0], t[2] - t[0])
            out = np.dot(n, (t[0] + t[1] + t[2]) / 3 - cen) > 0
            if out == inward:
                t = (t[0], t[2], t[1])
            tris.append(t)
    return tris


def _volume(tris):
    return sum(float(np.dot(a, np.cross(b, c))) for a, b, c in tris) / 6.0


def _ry(a):
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def boat(L, ballast=False):
    """The boat in its own frame (centre of its underside, length along y, tongue out of the back at -y). ballast=True
    leaves a closed pocket in the floor (18 x 10 x 0.8 mm, 0.6 to 1.4 mm up) for the steel ballast plate: pause the
    print after the 1.4 mm layer, drop the plate in, resume; the plate ends sealed inside the floor."""
    T, W, H = BOAT_T, BOAT_W, BOAT_H
    tris = _box((0.0, 0.0, T / 2), (W, L - 2 * BEVEL, T))                                  # floor (between bevels)
    if ballast:
        tris += _box((0.0, 0.0, 0.001), BALLAST, inward=True)
    for e in (-1, 1):
        tris += _box((e * (W - T) / 2, 0.0, (H + T) / 2), (T, L, H - T))                  # side walls
        tris += _box((0.0, e * (L - T) / 2, (H + T) / 2), (W - 2 * T, T, H - T))          # end walls
        # bevelled floor ends: full 2 mm floor 4 mm in, 0.8 mm thick (underside 1.2 mm up) at the boat's end, as task.py
        tris += _hexa([(x, e * y, z) for x in (-W / 2, W / 2) for y, z0 in ((L / 2 - BEVEL, 0.0), (L / 2, T - 0.0008))
                       for z in (z0, T)])
    tris += _box((0.0, -L / 2 - TONGUE_L / 2 + 0.001, TONGUE_Z), (TONGUE_W, TONGUE_L + 0.002, TONGUE_W),
                 _ry(TONGUE_TURN))                                                         # tongue (1 mm into the wall)
    return tris


def load(L):
    """The captive load: a 4 mm slab filling the boat's floor inside the walls, 0.2 mm clearance each side."""
    return _box((0.0, 0.0, BOAT_T + LOAD_H / 2), (BOAT_W - 2 * BOAT_T - 2 * LOAD_GAP, L - 2 * BOAT_T - 2 * LOAD_GAP,
                                                  LOAD_H))


def push_rod():
    """The rod in its own frame: paddle face at y = 0, the bar back along -y, the T-handle plate over its end."""
    tris = []
    # the bar: 8 mm square, hollow (1.0 mm walls, closed ends), from 4 mm behind the face to 293 mm
    y0, y1 = -0.004, -ROD_L
    tris += _box((0.0, (y0 + y1) / 2, 0.0), (GRIP_W, y0 - y1, GRIP_W))
    tris += _box((0.0, (y0 + y1) / 2, 0.0), (GRIP_W - 2 * WALL, y0 - y1 - 2 * WALL, GRIP_W - 2 * WALL), inward=True)
    # the paddle: solid, 4 mm thick, 14 mm wide at its foot and 20 mm at its top, foot bevelled up at the front
    tris += _hexa([(sx * (0.007 if z < 0 else 0.010), y, (z if z > 0 else (-0.003 if y == 0.0 else -0.0055)))
                   for sx in (-1, 1) for y in (0.0, -0.004) for z in (-0.006, 0.006)])
    # the T-handle plate: 26 x 5 x 45 mm over the bar's end (2 mm above its axis), hollow with 1.0 mm walls
    yc, zc = -(ROD_L + RING_T / 2), 0.002 + TH_UP / 2
    tris += _box((0.0, yc, zc), (TH_W, RING_T, TH_UP))
    tris += _box((0.0, yc, zc), (TH_W - 2 * WALL, RING_T - 2 * WALL, TH_UP - 2 * WALL), inward=True)
    return tris


def _stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"make_parts.py".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *(a * 1000), *(b * 1000), *(c * 1000), 0))


def _check_against_task():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import task as T  # noqa: F401  (needs the simulator's Python packages)
    except Exception as e:  # noqa: BLE001
        return f"task.py not importable here ({type(e).__name__}); sizes not cross-checked"
    pairs = [(BOAT_W, T.BOAT_W), (BOAT_H, T.BOAT_H), (BOAT_T, T.BOAT_T), (TONGUE_L, T.TONGUE_L),
             (TONGUE_W, T.TONGUE_W), (TONGUE_Z, T.TONGUE_Z), (TONGUE_TURN, T.TONGUE_TURN), (ROD_L, T.ROD_L),
             (GRIP_W, T.GRIP_W), (RING_T, T.RING_T), (TH_UP, T.TH_UP), (TH_W, T.TH_W), (SUB, T.CHIP),
             (SUB_T, T.CHIP_T), (BOAT_L["precursor"], T.BOATS["s_boat"][0]),
             (BOAT_L["substrate"], T.BOATS["chip_boat"][0])]
    bad = [p for p in pairs if abs(p[0] - p[1]) > 1e-9]
    return "sizes match task.py" if not bad else f"MISMATCH with task.py: {bad}"


def main(out="parts"):
    os.makedirs(out, exist_ok=True)
    parts = {"boat_precursor": boat(BOAT_L["precursor"]), "load_yellow": load(BOAT_L["precursor"]),
             "load_white": load(BOAT_L["precursor"]), "boat_substrate": boat(BOAT_L["substrate"], ballast=True),
             "push_rod": push_rod()}
    g = {}
    for name, tris in parts.items():
        _stl(os.path.join(out, name + ".stl"), tris)
        g[name] = _volume(tris) * PETG * 1000.0
    g["substrate"] = SUB * SUB * SUB_T * STEEL * 1000.0
    g["ballast"] = BALLAST[0] * BALLAST[1] * BALLAST[2] * STEEL * 1000.0
    rows = [("boat_precursor+load", g["boat_precursor"] + g["load_yellow"]),
            ("boat_substrate", g["boat_substrate"] + g["ballast"]), ("push_rod", g["push_rod"]),
            ("substrate", g["substrate"])]
    print(_check_against_task())
    print(f"{'part':22s} {'printed g':>10s} {'sim g':>7s} {'diff':>7s}")
    for n, m in rows:
        s = SIM_MASS_G[n]
        print(f"{n:22s} {m:10.2f} {s:7.2f} {100 * (m - s) / s:+6.1f}%")
    print(f"(boat_precursor alone {g['boat_precursor']:.2f} g, load {g['load_yellow']:.2f} g; boat_substrate printed "
          f"{g['boat_substrate']:.2f} g + ballast plate {g['ballast']:.2f} g)")
    print("wrote", ", ".join(sorted(parts)), "to", os.path.abspath(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "parts")
