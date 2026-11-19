"""Verification of knob_adapter.py (exits 1 on any failure; geometry errors are raised, not counted as no overlap).

1. Every printed part is one solid.
2. The assembled adapter (printed parts and the belts, from knob_adapter.assembly()) with stand-ins for what it sits
   among, taken from the simulated scene (task.py) in the adapter's frame: the plate carrying the control, the upper
   stage plates swept over their whole travel (50 mm front to back, 76 mm left to right) with the leaf clips on the
   top plate, the focus housing and the right coarse knob with its fine disc, the coaxial stage knobs on their shaft, the
   5 mm shafts and the bearings. Every pair that must not touch is checked for overlap.
3. Belt geometry: each belt's centre distance stays fixed (the control and the bracket are on the same plate).
4. Section renders through every connection (bearing seats, shaft bores, sleeves on the stage knobs, the wall on the
   plate's edge, both belt planes) in out/sections.png.

    python check_adapter.py
"""
import itertools
import sys

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import knob_adapter as A  # noqa: E402


def box(u0, u1, v0, v1, z0, z1):
    return cq.Workplane("XY").box(u1 - u0, v1 - v0, z1 - z0, centered=False).translate((u0, v0, z0))


def cyl(u, v, r, z0, z1):
    """A vertical cylinder at plan point (u, v) from z0 to z1."""
    return cq.Workplane("XY").center(u, v).circle(r).extrude(z1 - z0).translate((0, 0, z0))


def cyl_v(u, z, r, v0, v1):
    """A cylinder along v (the focus knobs' shaft runs left to right) centred at (u, z), from v0 to v1."""
    c = cq.Workplane("XY").circle(r).extrude(v1 - v0)  # along z, then turned so z becomes +v
    return c.rotate((0, 0, 0), (1, 0, 0), -90).translate((u, v0, z))


def scene():
    """Stand-ins in the adapter frame (u forward = -x of the scene, v outboard = -y, z from the stage top at the start),
    from task.py: lower plate 120 x 150 mm, z -16 to -11; y plate 110 x 140, z -11 to -6, +-38 mm left to right; x plate
    100 x 130, z -6 to 0, +-25 mm front to back, carrying the two leaf clips (posts and leaves up to 6 mm above it); the
    coaxial axis 5 mm behind the lower plate's front edge and 15 mm inboard of its right edge; the focus housing and
    the right coarse knob (60 mm across, 22.6 mm wide) on the shaft 70 mm above the table (z -143 here)."""
    S = {}
    S["carrying_plate"] = box(A.FRONT_U - A.PLATE_FRONT_TO_BACK, A.FRONT_U, A.EDGE_V - 150.0, A.EDGE_V, A.PLATE_BOT_Z, A.PLATE_TOP_Z)
    # upper plates swept over their whole travel (u from the scene's x, v from its y): y plate u -110 to 0, v -168 to
    # 48; x plate (and the clips on it) u -130 to 20, v -163 to 43
    S["y_plate_swept"] = box(-110.0, 0.0, -168.0, 48.0, -11.0, -6.0)
    S["x_plate_swept"] = box(-130.0, 20.0, -163.0, 43.0, -6.0, 6.0)
    # focus mechanism: housing 50 mm deep, 200 mm wide, 30 to 100 mm above the table; right coarse knob and the fine
    # disc in its face
    S["focus_housing"] = box(-70.0, -20.0, -160.0, 40.0, -183.0, -113.0)
    S["coarse_knob_r"] = cyl_v(-45.0, -143.0, 30.0, 40.0, 62.6)
    S["fine_knob_r"] = cyl_v(-45.0, -143.0, 25.0, 62.6, 66.6)
    # the coaxial control: both knobs on one shaft from the plate's underside down past the lower knob
    S["coax_shaft"] = cyl(0, 0, 3.0, A.Y_KNOB_TOP - A.Y_KNOB_H - 2.0, A.PLATE_BOT_Z)
    S["knob_x"] = cyl(0, 0, A.X_KNOB_D / 2, A.X_KNOB_TOP - A.X_KNOB_H, A.X_KNOB_TOP)
    S["knob_y"] = cyl(0, 0, A.Y_KNOB_D / 2, A.Y_KNOB_TOP - A.Y_KNOB_H, A.Y_KNOB_TOP)
    for tag, (u, v), zb in (("x", A.X_SHAFT, A.X_BELT_Z), ("y", A.Y_SHAFT, A.Y_BELT_Z)):
        bot = zb + A.RING_Z0 - 6.0
        S[f"shaft_{tag}"] = cyl(u, v, A.SHAFT_D / 2, bot, A.KNOB_Z0 + A.TOPKNOB_H - 3.0)
        seat = A.ARM_TOP + A.BOSS_H - 2 * A.BEARING_W
        S[f"bearings_{tag}"] = (cq.Workplane("XY").center(u, v).circle(A.BEARING_D / 2).circle(A.SHAFT_D / 2)
                                .extrude(2 * A.BEARING_W).translate((0, 0, seat)))
    return S


MEANT = {frozenset(p) for p in [
    ("sleeve_x", "knob_x"), ("sleeve_y", "knob_y"), ("knob_x", "coax_shaft"), ("knob_y", "coax_shaft"),
    ("knob_x", "carrying_plate"),
    ("bearings_x", "bracket"), ("bearings_y", "bracket"), ("bearings_x", "shaft_x"), ("bearings_y", "shaft_y"),
    ("pulley_x", "shaft_x"), ("pulley_y", "shaft_y"), ("top_knob_x", "shaft_x"), ("top_knob_y", "shaft_y"),
    ("belt_x", "sleeve_x"), ("belt_x", "pulley_x"), ("belt_y", "sleeve_y"), ("belt_y", "pulley_y"),
    ("coax_shaft", "carrying_plate"), ("y_plate_swept", "x_plate_swept"), ("y_plate_swept", "carrying_plate"),
    ("coarse_knob_r", "focus_housing"), ("fine_knob_r", "coarse_knob_r"),
    ("focus_housing", "carrying_plate")]}
# stand-ins among themselves are not the adapter's business
STANDINS = {"carrying_plate", "y_plate_swept", "x_plate_swept", "focus_housing", "coarse_knob_r",
            "fine_knob_r", "coax_shaft", "knob_x", "knob_y"}


def interference(parts):
    rows = []
    for (a, pa), (b, pb) in itertools.combinations(parts.items(), 2):
        if a in STANDINS and b in STANDINS:
            continue
        bb_a, bb_b = pa.val().BoundingBox(), pb.val().BoundingBox()
        if (bb_a.xmax < bb_b.xmin or bb_b.xmax < bb_a.xmin or bb_a.ymax < bb_b.ymin or bb_b.ymax < bb_a.ymin
                or bb_a.zmax < bb_b.zmin or bb_b.zmax < bb_a.zmin):
            continue
        inter = pa.intersect(pb)  # (errors propagate: a failed Boolean is not "no overlap")
        v = sum(s.Volume() for s in inter.solids().vals())
        if v > 1e-3:
            rows.append((a, b, v, frozenset((a, b)) in MEANT))
    return rows


def section(parts, ax, plane, title):
    import math
    for i, (name, p) in enumerate(parts.items()):
        if plane[0] == "z":
            slab = cq.Workplane("XY").box(600, 600, 0.2).translate((0, 0, plane[1]))
            cut = p.intersect(slab)
            proj = lambda q: (q.x, q.y)  # noqa: E731
        else:
            (u0, v0), (u1, v1) = plane[1], plane[2]
            ang = math.degrees(math.atan2(v1 - v0, u1 - u0))
            slab = cq.Workplane("XY").box(600, 0.2, 300).rotate((0, 0, 0), (0, 0, 1), ang).translate((u0, v0, 0))
            cut = p.intersect(slab)
            c, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            proj = lambda q, c=c, s_=s_, u0=u0, v0=v0: ((q.x - u0) * c + (q.y - v0) * s_, q.z)  # noqa: E731
        for e in cut.edges().vals():
            pts = [proj(e.positionAt(t / 24)) for t in range(25)]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], lw=0.6, color=f"C{i % 10}")
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def main():
    P, Asm = A.main()
    for name, p in P.items():
        A.check_single(name, p)
    parts = dict(Asm)
    parts.update(scene())
    rows = interference(parts)
    bad = [r for r in rows if not r[3]]
    print("\nOVERLAPS (part, part, volume mm3):")
    for r in rows:
        print(f"  {r[0]:15s} {r[1]:15s} {r[2]:9.1f}  {'fit' if r[3] else 'CLASH'}")
    import math
    print(f"\nbelt centre distances: x {math.dist((0, 0), A.X_SHAFT):.2f} mm (belt {A.BELT_X_L:.0f}), "
          f"y {math.dist((0, 0), A.Y_SHAFT):.2f} mm (belt {A.BELT_Y_L:.0f}); fixed over the stage's travel and the focus "
          f"stroke because the control and the bracket are on the same plate")
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    section(parts, axs[0, 0], ("line", (0, 0), A.X_SHAFT), "vertical section: coaxial axis to the X shaft")
    section(parts, axs[0, 1], ("line", (0, 0), A.Y_SHAFT), "vertical section: coaxial axis to the Y shaft")
    section(parts, axs[1, 0], ("z", A.X_BELT_Z), f"plan at the x belt, z = {A.X_BELT_Z:.0f} mm")
    section(parts, axs[1, 1], ("z", A.ARM_TOP + A.BOSS_H - A.BEARING_W), "plan through the bearing seats")
    fig.tight_layout()
    fig.savefig(A.OUT / "sections.png", dpi=110)
    print("wrote", A.OUT / "sections.png")
    print(f"\n{len(bad)} unintended overlaps")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
