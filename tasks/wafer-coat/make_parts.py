"""Write the real-trial parts for this task: STL files to 3D print and a DXF to laser-cut the lid.

    pip install numpy trimesh manifold3d shapely
    python make_parts.py [--wafer-d 50.80] [--out parts]

The aligner, the wafer stand, the tweezers' cup, the bowl and the chuck are built from the same shapes as the
simulation: this script reads them from task.py (no simulator needed). Units in the files: millimetres. The mock
coater adds what a real build needs (an open-bottom housing in four quarters, hinge brackets with the lid's 74 degree
stop, lid knuckles and the handle), and three centring gauges for scoring (see the README, Real-trial build).
"""

import argparse
import math
import pathlib
import sys
import types

import numpy as np
import trimesh
from shapely.geometry import Polygon

HERE = pathlib.Path(__file__).resolve().parent


# -- read the scene's shapes from task.py, with the simulator's modules stubbed out ----------------------------------
def _load_shapes():
    parts = {}

    class Joint:
        def __init__(self, name, **kw):
            self.name, self.kw = name, kw

    def part(name, pos, boxes=(), cylinders=(), **kw):
        parts[name] = dict(pos=pos, boxes=list(boxes), cylinders=list(cylinders), **kw)
        return name

    lk = types.ModuleType("labkit")
    lk.Joint, lk.part = Joint, part
    lk.on_reset = lk.on_step = lambda f: f
    lk.TORSIONAL_PATCH, lk.FRICTION_COMBINE, lk.JOINT_FRAME_ROT = {}, {}, {}
    lk.GLASS_COLORS = set()
    lk.Mesh = lambda **kw: kw
    stubs = {"labkit": lk}
    for m in ("yam_arena", "yam_arena.demo", "yam_arena.constants", "yam_arena.task_file", "yam_arena.urdf"):
        stubs[m] = types.ModuleType(m)
    stubs["yam_arena.demo"].Waypoint = object
    stubs["yam_arena.constants"].FINGER_OPEN = 0.0
    stubs["yam_arena.task_file"].TABLE_TOP = {"station": 0.0}
    stubs["yam_arena.urdf"].axis_angle_matrix = None
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        src = (HERE / "task.py").read_text()
        src = src[:src.index("OBJECTS = [")]  # the scene only: no scoring or demo code
        ns = {"__file__": str(HERE / "task.py"), "__name__": "task_shapes"}
        exec(compile(src, "task.py", "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return parts, ns


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def _move(c):
    t = np.eye(4)
    t[:3, 3] = c
    return t


def box(c, size, yaw=0.0, pitch=0.0):
    """A box in mm, centred at c, turned as labkit turns it: pitch about y, then yaw about z."""
    return trimesh.creation.box(extents=size, transform=_move(c) @ _rz(yaw) @ _ry(pitch))


def cyl(c, r, h, axis="Z", sections=64):
    t = {"Z": np.eye(4), "Y": trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]),
         "X": trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])}[axis]
    return trimesh.creation.cylinder(radius=r, height=h, sections=sections, transform=_move(c) @ t)


def union(ms):
    return trimesh.boolean.union(ms, engine="manifold")


def diff(a, bs):
    return trimesh.boolean.difference([a, *bs], engine="manifold")


def from_part(p, skip_cyl=lambda c: False):
    """A part's own boxes and cylinders, in mm, in the part's frame."""
    ms = []
    for b in p["boxes"]:
        turn = b[3] if len(b) > 3 else 0.0
        yaw, pitch = (turn[0], turn[1]) if isinstance(turn, (tuple, list)) else (turn, 0.0)
        ms.append(box(np.multiply(b[0], 1000), np.multiply(b[1], 1000), yaw, pitch))
    for c in p["cylinders"]:
        if skip_cyl(c):
            continue
        ms.append(cyl(np.multiply(c[0], 1000), c[1] * 1000, c[2] * 1000, c[4] if len(c) > 4 else "Z"))
    return union(ms)


def dxf(path, layers):
    """A minimal DXF (R12): closed polylines and circles, in mm."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for kind, data in layers:
        if kind == "circle":
            (x, y), r = data
            out += ["0", "CIRCLE", "8", "CUT", "10", f"{x:.3f}", "20", f"{y:.3f}", "30", "0", "40", f"{r:.3f}"]
        else:
            out += ["0", "POLYLINE", "8", "CUT", "66", "1", "70", "1"]
            for x, y in data:
                out += ["0", "VERTEX", "8", "CUT", "10", f"{x:.3f}", "20", f"{y:.3f}", "30", "0"]
            out += ["0", "SEQEND"]
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wafer-d", type=float, default=50.80, help="the trial wafers' measured diameter, mm")
    ap.add_argument("--out", default="parts")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    parts, ns = _load_shapes()
    mm = lambda v: v * 1000.0  # noqa: E731
    HW, HOUSING_H, RIM = mm(ns["HW"]), mm(ns["HOUSING_H"]), mm(ns["RIM"])
    BOWL_IN, RIB_R, CHUCK_R, CHUCK_TOP = mm(ns["BOWL_IN"]), mm(ns["RIB_R"]), mm(ns["CHUCK_R"]), mm(ns["CHUCK_TOP"])
    HX, HZ = mm(ns["HINGE"][0]), mm(ns["HINGE"][2])  # the lid's hinge axis (along y), coater frame
    LID_T, HOLE = mm(ns["LID_T"]), mm(ns["HOLE"])
    LID_MAX = ns["LID_MAX"]
    written = []

    def save(m, name):
        m.export(out / f"{name}.stl")
        assert trimesh.load(out / f"{name}.stl").is_volume, name  # closed and manifold as written
        written.append((name, m.extents.round(1).tolist()))

    # 1. Aligner, wafer stand and tweezers' cup: the simulation's own shapes. Print the aligner in PETG at 0.1 mm layers
    #    (or SLA resin), plate down.
    save(from_part(parts["aligner"]), "aligner")
    save(from_part(parts["stand"]), "wafer_stand")
    save(from_part(parts["tweezer_cup"]), "tweezer_cup")

    # 2. The bowl: a 4 mm floor disc carrying the ribbed wall up to the rim, one print (204 mm across), and a hole for
    #    the chuck's pin; four M3 holes screw it into the housing's recess. The floor's top is the sim's housing top.
    floor_r = BOWL_IN + 2 * RIB_R
    ribs = [c for c in parts["coater"]["cylinders"] if abs(mm(c[1]) - RIB_R) < 1e-6]
    bowl = union([cyl((0, 0, HOUSING_H - 2.0), floor_r, 4.0, sections=180),
                  *[cyl(np.multiply(c[0], 1000), RIB_R, mm(c[2]), sections=24) for c in ribs]])
    bowl = diff(bowl, [cyl((0, 0, HOUSING_H - 2.0), 4.1, 6.0),
                       *[cyl((60 * math.cos(t), 60 * math.sin(t), HOUSING_H - 2.0), 1.7, 6.0, sections=24)
                         for t in np.radians([45, 135, 225, 315])]])
    save(bowl, "bowl")

    # 3. The chuck: 24 mm across, its top 52 mm above the table, with an 8 mm pin pressed into the bowl's floor.
    #    Print it top face down (on the bed), so that face is flat.
    chuck = union([cyl((0, 0, (HOUSING_H + CHUCK_TOP) / 2), CHUCK_R, CHUCK_TOP - HOUSING_H),
                   cyl((0, 0, HOUSING_H - 2.0), 4.0, 4.0)])
    save(chuck, "chuck")

    # 4. The housing: an open-bottom box 220 x 220 x 40 mm, 3 mm walls and a 7 mm deck with a recess for the bowl's
    #    floor, in four quarters that fit any printer's bed; glue them on a flat bench, then screw the bowl in.
    deck0 = HOUSING_H - 7.0
    housing = diff(box((0, 0, HOUSING_H / 2), (2 * HW, 2 * HW, HOUSING_H)),
                   [box((0, 0, deck0 / 2 - 0.5), (2 * HW - 6, 2 * HW - 6, deck0 + 1)),
                    cyl((0, 0, HOUSING_H - 2.0 + 0.5), floor_r + 0.3, 5.0, sections=180),
                    *[cyl((60 * math.cos(t), 60 * math.sin(t), HOUSING_H - 4), 1.25, 8.0, sections=24)
                      for t in np.radians([45, 135, 225, 315])],
                    *[cyl((HW - 1.5, e * 70.0, 25.0), 1.25, 8.0, "X", sections=24) for e in (-1, 1)]])
    for qx in (-1, 1):
        for qy in (-1, 1):
            q = trimesh.intersections.slice_mesh_plane(housing, [qx, 0, 0], [0, 0, 0], cap=True)
            q = trimesh.intersections.slice_mesh_plane(q, [0, qy, 0], [0, 0, 0], cap=True)
            save(q, f"housing_quarter_{'front' if qx < 0 else 'back'}_{'left' if qy > 0 else 'right'}")

    # 5. Hinge brackets (two, at y = +-70 mm): they stand in the lid's back notches, hold its 3 mm hinge pin on the
    #    sim's axis, and stop it at 74 degrees (LID_MAX): their front face above the shut lid lies where the lid's top
    #    face lies at 74 degrees, so the lid rests on it, short of upright, and falls shut when let go.
    nrm = np.array([math.sin(LID_MAX), math.cos(LID_MAX)])  # the lid's top-face normal at 74 degrees (x, z)
    d = np.array([-math.cos(LID_MAX), math.sin(LID_MAX)])  # along the lid, from the hinge to its front
    p0 = np.array([HX, HZ]) + nrm * LID_T / 2
    at = lambda z: p0 + d * (z - p0[1]) / d[1]  # noqa: E731
    z_top = HZ + 3.5
    prof = [(HX - 8, HOUSING_H), (HX, HOUSING_H), (HX, 10.0), (HX + 8, 10.0), (HX + 8, 112.0), tuple(at(112.0)),
            tuple(at(z_top)), (HX - 8, z_top)]
    poly = trimesh.creation.extrude_polygon(Polygon(prof), 12.0)
    poly.apply_transform(_move((0, -6.0, 0)) @ trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    poly.apply_translation((0, 12.0, 0))
    bracket = diff(poly, [cyl((HX, 0, HZ), 1.65, 20, "Y", sections=32), cyl((HX + 4, 0, 25.0), 1.7, 20, "X")])
    save(bracket, "hinge_bracket")

    # 6. Lid knuckles (two): clip on the lid's back edge beside each bracket, a 3 mm steel pin pressed in on the axis,
    #    an M3 screw through knuckle and lid. The lid is cut 4 mm short at the back (216 mm) to clear the pin.
    lid_back = HX - 4.0
    knuckle = diff(box((HX - 7.5, 0, HZ), (19.0, 8.0, 10.0)),
                   [box((lid_back - 10 + 0.1, 0, HZ), (20.2, 10, LID_T + 0.2)), cyl((HX, 0, HZ), 1.5, 10, "Y"),
                    cyl((HX - 11.0, 0, HZ), 1.65, 12)])
    save(knuckle, "lid_knuckle")

    # 7. The lid's handle bar: 50 mm long, 10 mm across, on two 10 x 10 x 20 mm posts, screwed on from under the lid.
    hp = [b for b in parts["coater"]["joints"][0].kw["boxes"] if abs(b[1][2] - 0.020) < 1e-9]
    hc = parts["coater"]["joints"][0].kw["cylinders"][0]
    z_lid = mm(hp[0][0][2]) - 10.0  # the lid's top face
    handle = union([*[box(np.multiply(b[0], 1000), np.multiply(b[1], 1000)) for b in hp],
                    cyl(np.multiply(hc[0], 1000), 5.0, 50.0, "Y")])
    handle = diff(handle, [cyl((mm(b[0][0]), mm(b[0][1]), z_lid + 6), 1.25, 12.2) for b in hp])
    save(handle, "lid_handle")

    # 8. Centring gauges: a plate resting on the bowl's rim, centred by a spigot inside its ribs, with a tube hanging
    #    down round the chuck. A gauge seats only if the wafer's edge passes inside its tube: tube bore = wafer diameter
    #    + 2 x the tolerance. The bore is open at the top, so the wafer can be seen and its depth measured.
    for tol, name in ((0.5, "gauge_0p5mm"), (1.5, "gauge_1p5mm"), (12.0, "gauge_12mm")):
        bore = a.wafer_d / 2 + tol
        g = union([cyl((0, 0, RIM + 1.5), 103.0, 3.0, sections=180),
                   cyl((0, 0, RIM - 2.0), BOWL_IN - 0.2, 4.0, sections=180),
                   cyl((0, 0, (RIM + 46.0) / 2), bore + 2.0, RIM - 46.0, sections=180)])
        g = diff(g, [cyl((0, 0, RIM - 2.5), BOWL_IN - 3.2, 5.2, sections=180),
                     cyl((0, 0, (RIM + 46.0) / 2 + 1.0), bore, RIM - 46.0 + 8.0, sections=180),
                     *[cyl((70 * math.cos(t), 70 * math.sin(t), RIM + 1.5), 18.0, 3.4, sections=64)
                       for t in np.radians(range(0, 360, 60))]])
        save(g, name)

    # 9. Edge template: a black ring laid on the wafer after a trial, centred on its edge by a shallow recess; under
    #    UV light, any tool mark seen through its 38.8 mm window lies more than 6 mm inside the rim.
    t = diff(cyl((0, 0, 0.75), 30.0, 1.5), [cyl((0, 0, 0.75), a.wafer_d / 2 - 6.0, 2.0, sections=180),
                                             cyl((0, 0, 0.25), a.wafer_d / 2 + 0.4, 0.6, sections=180)])
    save(t, "edge_template")

    # 10. The lid, to laser-cut from 5 mm clear cast acrylic: 216 x 220 mm (front edge at x = -110), the 12 mm square
    #     dispense hole, two holes for the handle's screws, two back notches for the brackets, two for the knuckles' screws.
    x0, x1, yh = -HW, lid_back, HW
    nx = HX - 11.0  # the notches reach forward to here: the lid clears the brackets as it opens
    outline = [(x0, -yh), (x1, -yh)]
    for yc in (-70.0, 70.0):
        outline += [(x1, yc - 7), (nx, yc - 7), (nx, yc + 7), (x1, yc + 7)]
    outline += [(x1, yh), (x0, yh)]
    ents = [("poly", outline), ("poly", [(-HOLE / 2, -HOLE / 2), (HOLE / 2, -HOLE / 2), (HOLE / 2, HOLE / 2),
                                         (-HOLE / 2, HOLE / 2)])]
    ents += [("circle", ((mm(b[0][0]), mm(b[0][1])), 1.65)) for b in hp]
    ents += [("circle", ((HX - 11.0, e * 81.0), 1.65)) for e in (-1, 1)]
    dxf(out / "lid_5mm_acrylic.dxf", ents)

    for n, e in written:
        print(f"{n}.stl  {e[0]} x {e[1]} x {e[2]} mm")
    print("lid_5mm_acrylic.dxf  216 x 220 mm")


if __name__ == "__main__":
    main()
