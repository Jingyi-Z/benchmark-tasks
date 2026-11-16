"""Generate watertight binary STLs for the 80.0 x 80.0 mm Intel LGA2011 Square-ILM spring-pocket bolster.

Usage:
    python generate_fixture_stl.py [output_dir]

Writes three watertight binary STL files (units: millimetres) matching ``task.py`` to 0.001 mm:
  1. ``square_ilm_bolster_plate.stl``: 94.0 x 94.0 x 5.0 mm Square-ILM bolster base (z in [0, 5] mm),
     central 38.0 x 38.0 x 2.865 mm bare CPU IHS boss (z in [5.000, 7.865] mm, so adding the 0.100 mm
     Kapton insulator + 0.035 mm copper contact pads places the instrumented IHS contact datum at exact
     z = 8.000 mm = +3.000 mm above the bolster plane), and four corner 10.50 x 10.50 x 27.30 mm M4
     standoffs on exact 80.00 x 80.00 mm centers (z in [5.000, 32.300] mm -> mouth rim +24.300 mm above
     the instrumented IHS datum, matching 0.4020 - 0.3777 = 0.02430 m in task.py). Each standoff has an
     8.90 mm lower M4 bore section (4.20 x 4.20 mm stud clearance) and an 18.40 mm-deep coaxial
     7.00 x 7.00 mm spring counterbore pocket (z in [13.900, 32.300] mm) housing a Lee Spring LC 016C 12 S
     stainless return spring (OD 6.10 mm, ID 5.28 mm, free length 23.83 mm, solid height 3.29 mm).
  2. ``square_ilm_spring_standoff_cup.stl`` (print 4x for Option A ASUS X99-A motherboard): standalone
     10.50 x 10.50 x 24.30 mm M4-bored standoff cup with the same 18.40 mm-deep 7.00 x 7.00 mm spring
     counterbore pocket (5.90 mm lower M4 base + 18.40 mm pocket) to mount on four M4x0.7x45 mm studs.
  3. ``spring_follower_collar.stl`` (print 4x): 6.50 x 6.50 x 2.00 mm sliding follower collar with a
     4.30 x 4.30 mm M4 stud bore that rides on top of each LC 016C 12 S coil inside the 7.00 x 7.00 mm
     pocket, projecting 0.40 mm at reset (L_reset = 18.80 mm) and retracting flush at full 2.40 mm
     seating (L_seated = 16.40 mm >> 3.29 mm solid height).
"""
from __future__ import annotations

import pathlib
import struct
import sys


def _box_tris(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> list[tuple]:
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    v = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    faces = [
        ((0, 0, -1), (0, 2, 1), (0, 3, 2)),
        ((0, 0, +1), (4, 5, 6), (4, 6, 7)),
        ((0, -1, 0), (0, 1, 5), (0, 5, 4)),
        ((0, +1, 0), (3, 7, 6), (3, 6, 2)),
        ((-1, 0, 0), (0, 4, 7), (0, 7, 3)),
        ((+1, 0, 0), (1, 2, 6), (1, 6, 5)),
    ]
    out = []
    for n, f1, f2 in faces:
        out.append((n, v[f1[0]], v[f1[1]], v[f1[2]]))
        out.append((n, v[f2[0]], v[f2[1]], v[f2[2]]))
    return out


def _hollow_rect_tube(
    cx: float, cy: float, z0: float, z1: float, outer_w: float, inner_w: float
) -> list[tuple]:
    """Four disjoint wall slabs forming a rectangular tube with outer_w and square bore inner_w."""
    wall = 0.5 * (outer_w - inner_w)
    zc = 0.5 * (z0 + z1)
    zh = z1 - z0
    off = 0.5 * (inner_w + wall)
    tris = []
    # +Y and -Y full-width slabs:
    tris += _box_tris(cx, cy + off, zc, outer_w, wall, zh)
    tris += _box_tris(cx, cy - off, zc, outer_w, wall, zh)
    # +X and -X inner-span slabs:
    tris += _box_tris(cx + off, cy, zc, wall, inner_w, zh)
    tris += _box_tris(cx - off, cy, zc, wall, inner_w, zh)
    return tris


def _write_stl(path: pathlib.Path, header: str, tris: list[tuple]) -> None:
    buf = bytearray(header.encode("ascii")[:80].ljust(80, b" "))
    buf += struct.pack("<I", len(tris))
    for n, p0, p1, p2 in tris:
        buf += struct.pack("<12fH", *n, *p0, *p1, *p2, 0)
    path.write_bytes(bytes(buf))
    print(f"Wrote {path.resolve()} ({len(tris)} triangles)")


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "cad_stl")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Complete benchtop Square-ILM bolster plate with counterbored spring-pocket standoffs:
    bolster_tris = []
    bolster_tris += _box_tris(0.0, 0.0, 2.5, 94.0, 94.0, 5.0)
    # Bare IHS boss 2.865 mm tall -> + 0.135 mm (0.100 mm Kapton + 0.035 mm copper pad) = z 8.000 mm datum:
    bolster_tris += _box_tris(0.0, 0.0, 5.0 + 1.4325, 38.0, 38.0, 2.865)
    for sy in (-40.0, +40.0):
        for sz in (-40.0, +40.0):
            # Lower 8.90 mm solid standoff base with 4.20 mm M4 stud bore (z in [5.00, 13.90] mm):
            bolster_tris += _hollow_rect_tube(sy, sz, 5.0, 13.9, 10.5, 4.2)
            # Upper 18.40 mm coaxial 7.00 x 7.00 mm spring counterbore pocket (z in [13.90, 32.30] mm):
            bolster_tris += _hollow_rect_tube(sy, sz, 13.9, 32.3, 10.5, 7.0)
    _write_stl(
        out_dir / "square_ilm_bolster_plate.stl",
        "heatsink-cross-tighten Square-ILM spring-pocket bolster (mm)",
        bolster_tris,
    )

    # 2. Standalone 24.30 mm M4 spring-pocket standoff cup (print 4x for ASUS X99-A motherboard option):
    cup_tris = []
    cup_tris += _hollow_rect_tube(0.0, 0.0, 0.0, 5.9, 10.5, 4.2)
    cup_tris += _hollow_rect_tube(0.0, 0.0, 5.9, 24.3, 10.5, 7.0)
    _write_stl(
        out_dir / "square_ilm_spring_standoff_cup.stl",
        "heatsink-cross-tighten 24.3mm M4 spring-pocket standoff cup (mm)",
        cup_tris,
    )

    # 3. Sliding spring-follower collar (print 4x: 6.50 x 6.50 x 2.00 mm, 4.30 mm M4 bore):
    collar_tris = _hollow_rect_tube(0.0, 0.0, 0.0, 2.0, 6.5, 4.3)
    _write_stl(
        out_dir / "spring_follower_collar.stl",
        "heatsink-cross-tighten 6.5x6.5x2.0mm M4 spring follower collar (mm)",
        collar_tris,
    )


if __name__ == "__main__":
    main()
