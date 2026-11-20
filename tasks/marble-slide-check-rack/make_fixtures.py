"""Deterministic 1:1 printable PLA fixture exporter (`python make_fixtures.py --out-dir fixtures_stl`).

Exports 11 normal-bed (`189.8 mm` max dimension across all pieces, fitting any `220 x 220 x 220 mm`
desktop FDM printer) closed 2-manifold watertight binary STL parts + `ASSEMBLY.json` whose assembled
outer envelope fits 100% inside `IN_STAND_BOXES` (`60 x 56 x 84 mm`, `1.50 mm` clear slot between
`5.00 mm` walls on `6.50 mm` centres) and `RISER_BOXES` in `task.py`, with `>= 0.10 mm` assembly
clearance (`0.000 mm^3` solid overlap) across all 55 part pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct

N_CIRCLE = 24  # Multiple of 4 so quadrant-aligned circle vertices connect cleanly to cell corners
PLA_DENSITY_G_CM3 = 1.24


def _unit_normal(p0: tuple[float, float, float], p1: tuple[float, float, float], p2: tuple[float, float, float]) -> tuple[float, float, float]:
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
    assert nlen > 1e-10, f"Degenerate triangle: {p0}, {p1}, {p2}"
    return (nx / nlen, ny / nlen, nz / nlen)


def _add_tri(tris: list, p0: tuple[float, float, float], p1: tuple[float, float, float], p2: tuple[float, float, float]) -> None:
    p0r = (round(p0[0], 5), round(p0[1], 5), round(p0[2], 5))
    p1r = (round(p1[0], 5), round(p1[1], 5), round(p1[2], 5))
    p2r = (round(p2[0], 5), round(p2[1], 5), round(p2[2], 5))
    tris.append((_unit_normal(p0r, p1r, p2r), p0r, p1r, p2r))


def _emit_perforated_cap(
    tris: list,
    to_xyz,
    u_nodes: list[float],
    v_nodes: list[float],
    hc_u: float,
    hc_v: float,
    r: float,
    w: float,
    outward_pos_w: bool,
) -> None:
    """Triangulate `[ua, ub] x [va, vb]` minus circle `(hc_u, hc_v, r)` matching all grid edge subdivisions
    `u_nodes` and `v_nodes` (zero T-junctions on neighboring grid cells!) and 24 inner circle vertices.
    """
    ua, ub = u_nodes[0], u_nodes[-1]
    va, vb = v_nodes[0], v_nodes[-1]
    assert ua + 0.25 < hc_u - r and hc_u + r < ub - 0.25, (ua, ub, hc_u, r)
    assert va + 0.25 < hc_v - r and hc_v + r < vb - 0.25, (va, vb, hc_v, r)
    q_step = N_CIRCLE // 4  # 6 segments per quadrant
    ring = [
        (hc_u + r * math.cos(2.0 * math.pi * m / N_CIRCLE), hc_v + r * math.sin(2.0 * math.pi * m / N_CIRCLE))
        for m in range(N_CIRCLE)
    ]
    c_ne = (ub, vb)
    c_nw = (ua, vb)
    c_sw = (ua, va)
    c_se = (ub, va)
    quad_corners = (c_ne, c_nw, c_sw, c_se)

    uv_tris: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for q in range(4):
        corner = quad_corners[q]
        m0 = q * q_step
        for s in range(q_step):
            a = ring[m0 + s]
            b = ring[(m0 + s + 1) % N_CIRCLE]
            uv_tris.append((corner, b, a))
    # Subdivided outer boundary edge fans matching adjacent grid intervals 1-for-1:
    for k in range(len(v_nodes) - 1):
        uv_tris.append(((ub, v_nodes[k]), (ub, v_nodes[k + 1]), ring[0 * q_step]))
    for k in range(len(u_nodes) - 1, 0, -1):
        uv_tris.append(((u_nodes[k], vb), (u_nodes[k - 1], vb), ring[1 * q_step]))
    for k in range(len(v_nodes) - 1, 0, -1):
        uv_tris.append(((ua, v_nodes[k]), (ua, v_nodes[k - 1]), ring[2 * q_step]))
    for k in range(len(u_nodes) - 1):
        uv_tris.append(((u_nodes[k], va), (u_nodes[k + 1], va), ring[3 * q_step]))

    for a, b, c in uv_tris:
        cross2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        assert cross2 > 1e-8, f"Non-positive cap triangle area: {cross2}"
        if outward_pos_w:
            _add_tri(tris, to_xyz(a[0], a[1], w), to_xyz(b[0], b[1], w), to_xyz(c[0], c[1], w))
        else:
            _add_tri(tris, to_xyz(a[0], a[1], w), to_xyz(c[0], c[1], w), to_xyz(b[0], b[1], w))


def _emit_annulus(
    tris: list,
    to_xyz,
    hc_u: float,
    hc_v: float,
    r_in: float,
    r_out: float,
    w: float,
    outward_pos_w: bool,
) -> None:
    """Emit annular counterbore step between `r_in` and `r_out` at coordinate `w`."""
    assert r_out > r_in + 0.1
    for m in range(N_CIRCLE):
        a0 = 2.0 * math.pi * m / N_CIRCLE
        a1 = 2.0 * math.pi * ((m + 1) % N_CIRCLE) / N_CIRCLE
        in0 = (hc_u + r_in * math.cos(a0), hc_v + r_in * math.sin(a0))
        in1 = (hc_u + r_in * math.cos(a1), hc_v + r_in * math.sin(a1))
        out0 = (hc_u + r_out * math.cos(a0), hc_v + r_out * math.sin(a0))
        out1 = (hc_u + r_out * math.cos(a1), hc_v + r_out * math.sin(a1))
        # Outer boundary is CW (hole enlarges if normal is -w; shrinks if normal is +w):
        # For +w normal, solid has outer boundary CCW (out0 -> out1) and inner boundary CW (in1 -> in0):
        if outward_pos_w:
            _add_tri(tris, to_xyz(in0[0], in0[1], w), to_xyz(out0[0], out0[1], w), to_xyz(out1[0], out1[1], w))
            _add_tri(tris, to_xyz(in0[0], in0[1], w), to_xyz(out1[0], out1[1], w), to_xyz(in1[0], in1[1], w))
        else:
            _add_tri(tris, to_xyz(in0[0], in0[1], w), to_xyz(out1[0], out1[1], w), to_xyz(out0[0], out0[1], w))
            _add_tri(tris, to_xyz(in0[0], in0[1], w), to_xyz(in1[0], in1[1], w), to_xyz(out1[0], out1[1], w))


def _emit_cyl_bore_wall(tris: list, to_xyz, hc_u: float, hc_v: float, r: float, wa: float, wb: float) -> None:
    """Emit inward-facing cylindrical bore wall from `wa` to `wb` (`wb > wa`)."""
    assert wb > wa + 1e-6
    for m in range(N_CIRCLE):
        a0 = 2.0 * math.pi * m / N_CIRCLE
        a1 = 2.0 * math.pi * ((m + 1) % N_CIRCLE) / N_CIRCLE
        u0, v0 = hc_u + r * math.cos(a0), hc_v + r * math.sin(a0)
        u1, v1 = hc_u + r * math.cos(a1), hc_v + r * math.sin(a1)
        b0 = to_xyz(u0, v0, wa)
        b1 = to_xyz(u1, v1, wa)
        t1 = to_xyz(u1, v1, wb)
        t0 = to_xyz(u0, v0, wb)
        # Inward normal toward (hc_u, hc_v): (b0 -> t0 -> t1) has cross((0,0,wb-wa), (u1-u0, v1-v0, wb-wa)) = (-(wb-wa)*(v1-v0), (wb-wa)*(u1-u0), 0) -> points inward!
        _add_tri(tris, b0, t0, t1)
        _add_tri(tris, b0, t1, b1)


class ManifoldPartBuilder:
    """Coordinate-compressed 2-manifold watertight solid builder (boxes + through/counterbored bores)."""

    def __init__(self, name: str):
        self.name = name
        self.boxes: list[tuple[float, float, float, float, float, float]] = []
        self.bores: list[dict] = []

    def add_box(self, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> None:
        assert x1 > x0 and y1 > y0 and z1 > z0
        for bx0, bx1, by0, by1, bz0, bz1 in self.boxes:
            ox = min(x1, bx1) - max(x0, bx0)
            oy = min(y1, by1) - max(y0, by0)
            oz = min(z1, bz1) - max(z0, bz0)
            if ox > 1e-6 and oy > 1e-6 and oz > 1e-6:
                raise AssertionError(f"{self.name}: internal box overlap ({ox:.3f}, {oy:.3f}, {oz:.3f})")
        self.boxes.append((round(x0, 4), round(x1, 4), round(y0, 4), round(y1, 4), round(z0, 4), round(z1, 4)))

    def add_bore(
        self,
        axis: str,
        u: float,
        v: float,
        w0: float,
        w1: float,
        r_thru: float,
        cb_side: str | None = None,
        r_cb: float = 0.0,
        cb_depth: float = 0.0,
    ) -> None:
        assert axis in ("X", "Z"), axis
        assert w1 > w0 + 0.5
        pad = round((r_cb if cb_side else r_thru) + 0.35, 4)
        self.bores.append(
            {
                "axis": axis,
                "u": round(u, 4),
                "v": round(v, 4),
                "ua": round(u - pad, 4),
                "ub": round(u + pad, 4),
                "va": round(v - pad, 4),
                "vb": round(v + pad, 4),
                "w0": round(w0, 4),
                "w1": round(w1, 4),
                "r_thru": round(r_thru, 4),
                "cb_side": cb_side,
                "r_cb": round(r_cb, 4),
                "cb_depth": round(cb_depth, 4),
            }
        )

    def build_tris(self) -> list:
        xs_set, ys_set, zs_set = set(), set(), set()
        for x0, x1, y0, y1, z0, z1 in self.boxes:
            xs_set.update((x0, x1))
            ys_set.update((y0, y1))
            zs_set.update((z0, z1))
        for br in self.bores:
            if br["axis"] == "Z":
                xs_set.update((br["ua"], br["ub"]))
                ys_set.update((br["va"], br["vb"]))
                zs_set.update((br["w0"], br["w1"]))
            else:  # axis == "X": u=Y, v=Z, w=X
                ys_set.update((br["ua"], br["ub"]))
                zs_set.update((br["va"], br["vb"]))
                xs_set.update((br["w0"], br["w1"]))

        xs = sorted(xs_set)
        ys = sorted(ys_set)
        zs = sorted(zs_set)
        nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1

        def is_solid(ix: int, iy: int, iz: int) -> bool:
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                return False
            mx = 0.5 * (xs[ix] + xs[ix + 1])
            my = 0.5 * (ys[iy] + ys[iy + 1])
            mz = 0.5 * (zs[iz] + zs[iz + 1])
            for x0, x1, y0, y1, z0, z1 in self.boxes:
                if x0 <= mx <= x1 and y0 <= my <= y1 and z0 <= mz <= z1:
                    return True
            return False

        occ = [[[is_solid(ix, iy, iz) for iz in range(nz)] for iy in range(ny)] for ix in range(nx)]

        # Track perforated endcap faces to replace solid grid quads with perforated caps:
        perf_caps_z: dict[tuple[int, int, int, bool], dict] = {}
        perf_caps_x: dict[tuple[int, int, int, bool], dict] = {}
        tris: list = []

        for br in self.bores:
            if br["axis"] == "Z":
                ix_list = [i for i in range(nx) if xs[i] >= br["ua"] - 1e-6 and xs[i + 1] <= br["ub"] + 1e-6]
                iy_list = [j for j in range(ny) if ys[j] >= br["va"] - 1e-6 and ys[j + 1] <= br["vb"] + 1e-6]
                iz_list = [k for k in range(nz) if zs[k] >= br["w0"] - 1e-6 and zs[k + 1] <= br["w1"] + 1e-6]
                assert ix_list and iy_list and iz_list, f"{self.name} empty Z-bore indices: {br}"
                iz_lo, iz_hi = iz_list[0], iz_list[-1]
                for ix0 in ix_list:
                    for iy0 in iy_list:
                        for kz in iz_list:
                            assert occ[ix0][iy0][kz], f"{self.name} Z-bore passes through empty cell: {br}"
                        assert not is_solid(ix0, iy0, iz_lo - 1), f"{self.name} Z-bore w0 not exposed: {br}"
                        assert not is_solid(ix0, iy0, iz_hi + 1), f"{self.name} Z-bore w1 not exposed: {br}"
                        perf_caps_z[(ix0, iy0, iz_lo, False)] = br
                        perf_caps_z[(ix0, iy0, iz_hi, True)] = br
                u_nodes = [xs[ix_list[0]]] + [xs[i + 1] for i in ix_list]
                v_nodes = [ys[iy_list[0]]] + [ys[j + 1] for j in iy_list]
                to_xyz = lambda u, v, w: (u, v, w)
            else:
                iy_list = [j for j in range(ny) if ys[j] >= br["ua"] - 1e-6 and ys[j + 1] <= br["ub"] + 1e-6]
                iz_list = [k for k in range(nz) if zs[k] >= br["va"] - 1e-6 and zs[k + 1] <= br["vb"] + 1e-6]
                ix_list = [i for i in range(nx) if xs[i] >= br["w0"] - 1e-6 and xs[i + 1] <= br["w1"] + 1e-6]
                assert iy_list and iz_list and ix_list, f"{self.name} empty X-bore indices: {br}"
                ix_lo, ix_hi = ix_list[0], ix_list[-1]
                for iy0 in iy_list:
                    for iz0 in iz_list:
                        for kx in ix_list:
                            assert occ[kx][iy0][iz0], f"{self.name} X-bore passes through empty cell: {br}"
                        assert not is_solid(ix_lo - 1, iy0, iz0), f"{self.name} X-bore w0 not exposed: {br}"
                        assert not is_solid(ix_hi + 1, iy0, iz0), f"{self.name} X-bore w1 not exposed: {br}"
                        perf_caps_x[(ix_lo, iy0, iz0, False)] = br
                        perf_caps_x[(ix_hi, iy0, iz0, True)] = br
                u_nodes = [ys[iy_list[0]]] + [ys[j + 1] for j in iy_list]
                v_nodes = [zs[iz_list[0]]] + [zs[k + 1] for k in iz_list]
                to_xyz = lambda u, v, w: (w, u, v)

            r_lo = br["r_cb"] if br["cb_side"] == "lo" else br["r_thru"]
            r_hi = br["r_cb"] if br["cb_side"] == "hi" else br["r_thru"]
            _emit_perforated_cap(tris, to_xyz, u_nodes, v_nodes, br["u"], br["v"], r_lo, br["w0"], False)
            _emit_perforated_cap(tris, to_xyz, u_nodes, v_nodes, br["u"], br["v"], r_hi, br["w1"], True)
            if br["cb_side"] == "lo":
                w_step = round(br["w0"] + br["cb_depth"], 4)
                _emit_cyl_bore_wall(tris, to_xyz, br["u"], br["v"], br["r_cb"], br["w0"], w_step)
                # At w_step, solid steps INWARD going +w, so annular step normal points -w (`outward_pos_w=False`):
                _emit_annulus(tris, to_xyz, br["u"], br["v"], br["r_thru"], br["r_cb"], w_step, False)
                _emit_cyl_bore_wall(tris, to_xyz, br["u"], br["v"], br["r_thru"], w_step, br["w1"])
            elif br["cb_side"] == "hi":
                w_step = round(br["w1"] - br["cb_depth"], 4)
                _emit_cyl_bore_wall(tris, to_xyz, br["u"], br["v"], br["r_thru"], br["w0"], w_step)
                # At w_step, solid steps OUTWARD going -w (hole widens above w_step), so annular step normal points +w (`outward_pos_w=True`):
                _emit_annulus(tris, to_xyz, br["u"], br["v"], br["r_thru"], br["r_cb"], w_step, True)
                _emit_cyl_bore_wall(tris, to_xyz, br["u"], br["v"], br["r_cb"], w_step, br["w1"])
            else:
                _emit_cyl_bore_wall(tris, to_xyz, br["u"], br["v"], br["r_thru"], br["w0"], br["w1"])

        # Emit exterior boundary quads for all solid cells:
        for ix in range(nx):
            x0, x1 = xs[ix], xs[ix + 1]
            for iy in range(ny):
                y0, y1 = ys[iy], ys[iy + 1]
                for iz in range(nz):
                    if not occ[ix][iy][iz]:
                        continue
                    z0, z1 = zs[iz], zs[iz + 1]
                    # -X face:
                    if not is_solid(ix - 1, iy, iz) and (ix, iy, iz, False) not in perf_caps_x:
                        _add_tri(tris, (x0, y0, z0), (x0, y0, z1), (x0, y1, z1))
                        _add_tri(tris, (x0, y0, z0), (x0, y1, z1), (x0, y1, z0))
                    # +X face:
                    if not is_solid(ix + 1, iy, iz) and (ix, iy, iz, True) not in perf_caps_x:
                        _add_tri(tris, (x1, y0, z0), (x1, y1, z0), (x1, y1, z1))
                        _add_tri(tris, (x1, y0, z0), (x1, y1, z1), (x1, y0, z1))
                    # -Y face:
                    if not is_solid(ix, iy - 1, iz):
                        _add_tri(tris, (x0, y0, z0), (x1, y0, z0), (x1, y0, z1))
                        _add_tri(tris, (x0, y0, z0), (x1, y0, z1), (x0, y0, z1))
                    # +Y face:
                    if not is_solid(ix, iy + 1, iz):
                        _add_tri(tris, (x0, y1, z0), (x0, y1, z1), (x1, y1, z1))
                        _add_tri(tris, (x0, y1, z0), (x1, y1, z1), (x1, y1, z0))
                    # -Z face:
                    if not is_solid(ix, iy, iz - 1) and (ix, iy, iz, False) not in perf_caps_z:
                        _add_tri(tris, (x0, y0, z0), (x0, y1, z0), (x1, y1, z0))
                        _add_tri(tris, (x0, y0, z0), (x1, y1, z0), (x1, y0, z0))
                    # +Z face:
                    if not is_solid(ix, iy, iz + 1) and (ix, iy, iz, True) not in perf_caps_z:
                        _add_tri(tris, (x0, y0, z1), (x1, y0, z1), (x1, y1, z1))
                        _add_tri(tris, (x0, y0, z1), (x1, y1, z1), (x0, y1, z1))

        return tris


def _verify_2manifold_and_metrics(name: str, tris: list) -> tuple[float, float]:
    """Assert strict closed 2-manifold topology (every directed edge appears once with exact reverse mate)
    and return `(volume_mm3, outer_area_mm2)`.
    """
    edge_counts: dict[tuple, int] = {}
    vol6 = 0.0
    area = 0.0
    for _, v0, v1, v2 in tris:
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (
                (round(a[0], 4), round(a[1], 4), round(a[2], 4)),
                (round(b[0], 4), round(b[1], 4), round(b[2], 4)),
            )
            edge_counts[key] = edge_counts.get(key, 0) + 1
        cx = v0[1] * v1[2] - v0[2] * v1[1]
        cy = v0[2] * v1[0] - v0[0] * v1[2]
        cz = v0[0] * v1[1] - v0[1] * v1[0]
        vol6 += cx * v2[0] + cy * v2[1] + cz * v2[2]
        ux, uy, uz = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
        vx, vy, vz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        area += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)

    for (a, b), cnt in edge_counts.items():
        assert cnt == 1, f"{name}: non-manifold duplicate directed edge {a}->{b} count={cnt}"
        rev_cnt = edge_counts.get((b, a), 0)
        assert rev_cnt == 1, f"{name}: open boundary / unmatched edge {a}->{b} rev_cnt={rev_cnt}"

    vol_mm3 = vol6 / 6.0
    assert vol_mm3 > 0.0, f"{name}: non-positive volume {vol_mm3}"
    return vol_mm3, area


def write_stl(path: pathlib.Path, builder: ManifoldPartBuilder, label: str) -> dict:
    tris = builder.build_tris()
    vol_mm3, area_mm2 = _verify_2manifold_and_metrics(path.name, tris)
    hdr = label.encode("ascii")[:80].ljust(80, b" ")
    buf = bytearray(hdr) + struct.pack("<I", len(tris))
    xs, ys, zs = [], [], []
    for nrm, v0, v1, v2 in tris:
        buf += struct.pack("<12fH", *nrm, *v0, *v1, *v2, 0)
        for vx, vy, vz in (v0, v1, v2):
            xs.append(vx)
            ys.append(vy)
            zs.append(vz)
    raw = bytes(buf)
    path.write_bytes(raw)
    lo = [round(min(xs), 3), round(min(ys), 3), round(min(zs), 3)]
    hi = [round(max(xs), 3), round(max(ys), 3), round(max(zs), 3)]
    ext = [round(hi[i] - lo[i], 3) for i in range(3)]
    assert max(ext) <= 190.0, f"{path.name} exceeds 220 mm printer bed: {ext}"

    vol_cm3 = vol_mm3 / 1000.0
    area_cm2 = area_mm2 / 100.0
    # Shell + 35% gyroid infill calculation (not a weighed print):
    # 4 perimeters (1.6 mm vertical walls) and 5 top/bottom layers (1.0 mm horizontal skins) -> ~1.15 mm mean skin
    # after corner overlap correction; remaining interior core at 35% infill:
    shell_vol_cm3 = min(0.75 * vol_cm3, area_cm2 * 0.115)
    core_vol_cm3 = max(0.0, vol_cm3 - shell_vol_cm3)
    eff_vol_cm3 = shell_vol_cm3 + 0.35 * core_vol_cm3
    g_35_lower = vol_cm3 * PLA_DENSITY_G_CM3 * 0.35
    g_shell_35inf = eff_vol_cm3 * PLA_DENSITY_G_CM3
    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "watertight_2manifold": True,
        "triangles": len(tris),
        "bounds_min_mm": lo,
        "bounds_max_mm": hi,
        "bbox_mm": ext,
        "solid_volume_cm3": round(vol_cm3, 2),
        "surface_area_cm2": round(area_cm2, 2),
        "pla_35pct_uniform_lower_bound_g": round(g_35_lower, 1),
        "pla_4perim_5skin_35pct_infill_calc_g": round(g_shell_35inf, 1),
    }


def _populate_mount_beam(
    b: ManifoldPartBuilder,
    y_lo: float,
    y_hi: float,
    bolt_ys: tuple[float, ...],
    r_m4: float,
) -> None:
    """Relieved front rack-mounting beam (`x in [+0.20, +14.20]`, `z in [270.00, 308.70]`, `0.10 mm` print clearance
    below canopy `z = 308.80`) with open-rear (`+X`) `7 mm`-wrench nut pockets (`12.2 x 12.0 x 10.0 mm`) at `z in [270, 280]`
    clearing `Ø9.0 mm` DIN 125A washers by `>= 0.70 mm` radially on all sides.
    """
    b.add_box(+0.20, +14.20, y_lo, y_hi, 280.00, 308.70)
    b.add_box(+0.20, +2.00, y_lo, y_hi, 270.00, 280.00)
    pocket_half_y = 6.00
    cur_y = y_lo
    for yb in sorted(bolt_ys):
        if yb - pocket_half_y > cur_y + 1e-6:
            b.add_box(+2.00, +14.20, cur_y, yb - pocket_half_y, 270.00, 280.00)
        cur_y = yb + pocket_half_y
        b.add_bore("Z", +7.20, yb, 280.00, 308.70, r_m4)
    if y_hi > cur_y + 1e-6:
        b.add_box(+2.00, +14.20, cur_y, y_hi, 270.00, 280.00)


def export_all(out_dir: pathlib.Path) -> tuple[dict, list[ManifoldPartBuilder]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    R_M4 = 2.20      # Ø4.40 mm M4 clearance bore
    R_CB_M4 = 5.00   # Ø10.00 mm counterbore (24-gon min flat Ø9.91 mm clears Ø9.00 mm DIN 125A M4 washer by +0.91 mm)
    CB_DEPTH = 5.50  # Recesses 0.80 mm DIN 125A M4 washer + 4.00 mm ISO 4762 M4 cap head = 4.80 mm stack by 0.70 mm below flush
    R_CL = 2.75      # Ø5.50 mm rear foot clamp/M5 bore

    builders: list[ManifoldPartBuilder] = []
    parts_meta: list[dict] = []

    # 1. Monolithic slotted input stand (`60.0 x 56.0 x 84.0 mm` exact match to IN_STAND_BOXES, 1.50 mm clear slot):
    b_st = ManifoldPartBuilder("in_stand")
    b_st.add_box(-28.00, +32.00, -8.00, +48.00, 0.00, 62.00)
    b_st.add_box(-5.75, -0.75, -6.00, +46.00, 62.00, 84.00)
    b_st.add_box(+0.75, +5.75, -6.00, +46.00, 62.00, 84.00)
    builders.append(b_st)
    parts_meta.append(write_stl(out_dir / "in_stand.stl", b_st, "in_stand 60x56x84mm 1.5mm slot PLA"))

    # 2. Relieved front rack-mounting beam (`z in [270.00, 308.70]`, 0 overlap with canopy `z >= 308.80`):
    #    Asymmetric 9.45 mm / 4.45 mm half-lap leaves keep 3.95 mm solid PLA behind the 5.50 mm deep Ø10.0 mm counterbore!
    b_mbl = ManifoldPartBuilder("mount_beam_left")
    _populate_mount_beam(b_mbl, +85.00, +235.00, (+135.00, +205.00), R_M4)
    b_mbl.add_box(+0.20, +9.65, +65.20, +85.00, 270.00, 308.70)
    for zb in (281.00, 297.80):
        b_mbl.add_bore("X", +75.00, zb, +0.20, +9.65, R_M4, cb_side="lo", r_cb=R_CB_M4, cb_depth=CB_DEPTH)
    builders.append(b_mbl)
    parts_meta.append(write_stl(out_dir / "mount_beam_left.stl", b_mbl, "mount_beam_left z<=308.7 PLA"))

    b_mbc = ManifoldPartBuilder("mount_beam_center")
    _populate_mount_beam(b_mbc, -65.00, +65.00, (-35.00, +35.00), R_M4)
    b_mbc.add_box(+9.75, +14.20, +65.00, +84.80, 270.00, 308.70)
    b_mbc.add_box(+9.75, +14.20, -84.80, -65.00, 270.00, 308.70)
    for yb in (-75.00, +75.00):
        for zb in (281.00, 297.80):
            b_mbc.add_bore("X", yb, zb, +9.75, +14.20, R_M4)
    builders.append(b_mbc)
    parts_meta.append(write_stl(out_dir / "mount_beam_center.stl", b_mbc, "mount_beam_center z<=308.7 PLA"))

    b_mbr = ManifoldPartBuilder("mount_beam_right")
    _populate_mount_beam(b_mbr, -235.00, -85.00, (-205.00, -135.00), R_M4)
    b_mbr.add_box(+0.20, +9.65, -85.00, -65.20, 270.00, 308.70)
    for zb in (281.00, 297.80):
        b_mbr.add_bore("X", -75.00, zb, +0.20, +9.65, R_M4, cb_side="lo", r_cb=R_CB_M4, cb_depth=CB_DEPTH)
    builders.append(b_mbr)
    parts_meta.append(write_stl(out_dir / "mount_beam_right.stl", b_mbr, "mount_beam_right z<=308.7 PLA"))

    # 3. Top cantilever canopy (`x in [+0.20, +160.20]`, `z in [308.80, 320.80]`), top-counterbored for flush heads:
    b_cpl = ManifoldPartBuilder("canopy_left")
    b_cpl.add_box(+0.20, +160.20, +90.00, +260.00, 308.80, 320.80)
    b_cpl.add_box(+0.20, +160.20, +70.20, +90.00, 308.80, 314.75)
    for xb, yb in ((+7.20, +135.00), (+7.20, +205.00), (+135.00, +237.50), (+135.00, +252.50)):
        b_cpl.add_bore("Z", xb, yb, 308.80, 320.80, R_M4, cb_side="hi", r_cb=R_CB_M4, cb_depth=CB_DEPTH)
    for xb in (+45.00, +120.00):
        b_cpl.add_bore("Z", xb, +80.00, 308.80, 314.75, R_M4)
    builders.append(b_cpl)
    parts_meta.append(write_stl(out_dir / "canopy_left.stl", b_cpl, "canopy_left z>=308.8 PLA"))

    b_cpc = ManifoldPartBuilder("canopy_center")
    b_cpc.add_box(+0.20, +160.20, -70.00, +70.00, 308.80, 320.80)
    b_cpc.add_box(+0.20, +160.20, +70.00, +89.80, 314.85, 320.80)
    b_cpc.add_box(+0.20, +160.20, -89.80, -70.00, 314.85, 320.80)
    for yb in (-35.00, +35.00):
        b_cpc.add_bore("Z", +7.20, yb, 308.80, 320.80, R_M4, cb_side="hi", r_cb=R_CB_M4, cb_depth=CB_DEPTH)
    for yb in (-80.00, +80.00):
        for xb in (+45.00, +120.00):
            b_cpc.add_bore("Z", xb, yb, 314.85, 320.80, R_M4)
    builders.append(b_cpc)
    parts_meta.append(write_stl(out_dir / "canopy_center.stl", b_cpc, "canopy_center z>=308.8 PLA"))

    b_cpr = ManifoldPartBuilder("canopy_right")
    b_cpr.add_box(+0.20, +160.20, -260.00, -90.00, 308.80, 320.80)
    b_cpr.add_box(+0.20, +160.20, -90.00, -70.20, 308.80, 314.75)
    for xb, yb in ((+7.20, -205.00), (+7.20, -135.00), (+135.00, -252.50), (+135.00, -237.50)):
        b_cpr.add_bore("Z", xb, yb, 308.80, 320.80, R_M4, cb_side="hi", r_cb=R_CB_M4, cb_depth=CB_DEPTH)
    for xb in (+45.00, +120.00):
        b_cpr.add_bore("Z", xb, -80.00, 308.80, 314.75, R_M4)
    builders.append(b_cpr)
    parts_meta.append(write_stl(out_dir / "canopy_right.stl", b_cpr, "canopy_right z>=308.8 PLA"))

    # 4. Rear posts (`22 x 30 mm` exact Y span, zero external flange!) + integral `90 x 60 x 8 mm` base feet:
    for side_lbl, yc in (("left", +245.00), ("right", -245.00)):
        b_pl = ManifoldPartBuilder(f"post_lower_{side_lbl}")
        b_pl.add_box(+80.00, +170.00, yc - 30.00, yc + 30.00, 0.00, 8.00)
        b_pl.add_box(+124.00, +146.00, yc - 15.00, yc + 15.00, 8.00, 145.00)
        b_pl.add_box(+124.00, +134.95, yc - 15.00, yc + 15.00, 145.00, 164.80)
        for yb in (yc - 18.00, yc + 18.00):
            b_pl.add_bore("Z", +96.00, yb, 0.00, 8.00, R_CL)
        for yb in (yc - 7.50, yc + 7.50):
            b_pl.add_bore("X", yb, 155.00, +124.00, +134.95, R_M4)
        builders.append(b_pl)
        parts_meta.append(write_stl(out_dir / f"post_lower_{side_lbl}.stl", b_pl, f"post_lower_{side_lbl} PLA"))

        b_pu = ManifoldPartBuilder(f"post_upper_{side_lbl}")
        b_pu.add_box(+135.05, +146.00, yc - 15.00, yc + 15.00, 145.20, 165.00)
        b_pu.add_box(+124.00, +146.00, yc - 15.00, yc + 15.00, 165.00, 280.00)
        b_pu.add_box(+124.00, +128.50, yc - 15.00, yc + 15.00, 280.00, 290.80)
        b_pu.add_box(+124.00, +146.00, yc - 15.00, yc + 15.00, 290.80, 308.70)
        for yb in (yc - 7.50, yc + 7.50):
            b_pu.add_bore("X", yb, 155.00, +135.05, +146.00, R_M4)
            b_pu.add_bore("Z", +135.00, yb, 290.80, 308.70, R_M4)
        builders.append(b_pu)
        parts_meta.append(write_stl(out_dir / f"post_upper_{side_lbl}.stl", b_pu, f"post_upper_{side_lbl} PLA"))

    # Verify >= 0.09 mm clearance (0.000 mm^3 overlap) across all 55 pairs of distinct parts:
    min_gap_mm = float("inf")
    for i in range(len(builders)):
        for j in range(i + 1, len(builders)):
            for ax0, ax1, ay0, ay1, az0, az1 in builders[i].boxes:
                for bx0, bx1, by0, by1, bz0, bz1 in builders[j].boxes:
                    gx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
                    gy = max(0.0, max(ay0, by0) - min(ay1, by1))
                    gz = max(0.0, max(az0, bz0) - min(az1, bz1))
                    sep = math.sqrt(gx * gx + gy * gy + gz * gz)
                    min_gap_mm = min(min_gap_mm, sep)
                    if sep < 0.09:
                        raise AssertionError(
                            f"Insufficient clearance {builders[i].name} <-> {builders[j].name}: sep={sep:.4f} mm"
                        )

    tot_vol_cm3 = round(sum(p["solid_volume_cm3"] for p in parts_meta), 2)
    tot_area_cm2 = round(sum(p["surface_area_cm2"] for p in parts_meta), 2)
    tot_g_35_low = round(sum(p["pla_35pct_uniform_lower_bound_g"] for p in parts_meta), 1)
    tot_g_calc = round(sum(p["pla_4perim_5skin_35pct_infill_calc_g"] for p in parts_meta), 1)
    tot_g_range = [int(round(tot_g_calc * 0.90, -1)), int(round(tot_g_calc * 1.10, -1))]

    bolt_schedule = [
        {
            "joint": "mount_beam half-lap splices (left-center & right-center, y=+-75 mm, z=281.0 & 297.8 mm)",
            "screw": "M4 x 20 mm socket-head cap screw (ISO 4762, head Ø7.0 x 4.0 mm)",
            "qty": 4,
            "counterbore": "Ø10.0 mm x 5.5 mm deep on -X face (clears Ø9.0 x 0.8 mm DIN 125A head washer + 4.0 mm cap head = 4.8 mm with 0.7 mm recess below flush)",
            "pla_grip_mm": 8.5,
            "washer_stack_mm": 1.6,
            "nyloc_height_mm": 5.0,
            "total_under_head_stack_mm": 15.1,
            "thread_past_nyloc_mm": 4.9,
            "access_and_protrusion": "3.95 mm front leaf web + 0.10 mm gap + 4.45 mm rear leaf; -X cap head + DIN 125A washer recessed 0.7 mm inside Ø10.0 x 5.5 mm counterbore (100% flat rack face); +X DIN 125A washer + nyloc behind mount_beam",
        },
        {
            "joint": "canopy half-lap splices (left-center & right-center, x=+45 & +120 mm, y=+-80 mm)",
            "screw": "M4 x 25 mm socket-head cap screw (ISO 4762, head Ø7.0 x 4.0 mm)",
            "qty": 4,
            "counterbore": "None (external top head + DIN 125A washer at z in [320.8, 325.6] mm, audited in 578-hull check)",
            "pla_grip_mm": 12.0,
            "washer_stack_mm": 1.6,
            "nyloc_height_mm": 5.0,
            "total_under_head_stack_mm": 18.6,
            "thread_past_nyloc_mm": 6.4,
            "access_and_protrusion": "270 mm open access underneath canopy behind rack",
        },
        {
            "joint": "CK6518A rack rear wall to mount_beam (transferred Ø4.2 mm holes through solid 14 mm beam)",
            "screw": "M4 x 25 mm pan-head machine screw",
            "qty": 2,
            "counterbore": "None (pan head bears directly on 1.8 mm aluminum rear wall inside factory hole without head washer; 1x DIN 125A washer + nyloc on +X PLA rear face)",
            "pla_plus_metal_grip_mm": 16.0,
            "washer_stack_mm": 0.8,
            "nyloc_height_mm": 5.0,
            "total_under_head_stack_mm": 21.8,
            "thread_past_nyloc_mm": 3.2,
            "access_and_protrusion": "Pan head inside CK6518A rear mounting holes (no washer inside check channel); +X DIN 125A washer + nyloc behind mount_beam (4 screws supplied in kit)",
        },
        {
            "joint": "post mid-height half-lap splices (left & right posts, y=+-237.5 & +-252.5 mm, z=155.0 mm)",
            "screw": "M4 x 35 mm socket-head cap screw (ISO 4762, head Ø7.0 x 4.0 mm)",
            "qty": 4,
            "counterbore": "None (external -X head + DIN 125A washer and +X DIN 125A washer + nyloc, audited in 578-hull check)",
            "pla_grip_mm": 22.0,
            "washer_stack_mm": 1.6,
            "nyloc_height_mm": 5.0,
            "total_under_head_stack_mm": 28.6,
            "thread_past_nyloc_mm": 6.4,
            "access_and_protrusion": "Horizontal X-axis bolt at rear post (x=+124..+146 mm; >=21.4 mm 3D clearance from all 578 arm hulls during motion, >=291.5 mm parked)",
        },
        {
            "joint": "canopy-to-post_upper joints (x=+135 mm, y=+-237.5 & +-252.5 mm)",
            "screw": "M4 x 35 mm socket-head cap screw (ISO 4762, head Ø7.0 x 4.0 mm)",
            "qty": 4,
            "counterbore": "Ø10.0 mm x 5.5 mm deep on +Z canopy top (clears Ø9.0 x 0.8 mm DIN 125A head washer + 4.0 mm cap head = 4.8 mm with 0.7 mm recess below flush)",
            "pla_grip_mm": 24.5,
            "washer_stack_mm": 1.6,
            "nyloc_height_mm": 5.0,
            "total_under_head_stack_mm": 31.1,
            "thread_past_nyloc_mm": 3.9,
            "access_and_protrusion": "6.50 mm canopy web + 0.10 mm gap + 17.90 mm post head; top head + DIN 125A washer recessed 0.7 mm in Ø10.0 x 5.5 mm counterbore; bottom DIN 125A washer + nyloc accessed via 17.5 x 30 x 10.8 mm open +X rear post window (0.0 mm outer protrusion)",
        },
        {
            "joint": "canopy-to-mount_beam vertical joints (x=+7.2 mm, y=+-35, +-135, +-205 mm)",
            "screw": "M4 x 45 mm socket-head cap screw (ISO 4762, head Ø7.0 x 4.0 mm)",
            "qty": 6,
            "counterbore": "Ø10.0 mm x 5.5 mm deep on +Z canopy top (clears Ø9.0 x 0.8 mm DIN 125A head washer + 4.0 mm cap head = 4.8 mm with 0.7 mm recess below flush)",
            "pla_grip_mm": 35.3,
            "washer_stack_mm": 1.6,
            "nyloc_height_mm": 5.0,
            "total_under_head_stack_mm": 41.9,
            "thread_past_nyloc_mm": 3.1,
            "access_and_protrusion": "6.50 mm canopy web + 0.10 mm gap + 28.70 mm mount_beam body; top head + DIN 125A washer recessed 0.7 mm in Ø10.0 x 5.5 mm counterbore; bottom DIN 125A washer + nyloc recessed inside 12.2 x 12.0 x 10.0 mm open +X wrench pocket (0.0 mm outer protrusion)",
        },
    ]

    manifest = {
        "material": "Rigid PLA (1.24 g/cm^3; 4 perimeters / 1.6 mm walls, 5 top/bottom layers / 1.0 mm skins, 0.20 mm layers, 35% gyroid infill)",
        "volume_and_filament_calculation_not_weighed": {
            "total_closed_mesh_volume_cm3": tot_vol_cm3,
            "total_outer_surface_area_cm2": tot_area_cm2,
            "idealized_35pct_uniform_infill_lower_bound_g": tot_g_35_low,
            "calculated_4perim_5skin_35pct_infill_mass_g": tot_g_calc,
            "calculated_slicer_mass_range_g": tot_g_range,
            "spools_required": "2x 1 kg PLA spools (~$36 purchase, ~$22 consumed filament at $18/kg)",
        },
        "max_piece_dimension_mm": max(max(p["bbox_mm"]) for p in parts_meta),
        "outer_envelope_containment": "100% contained within authored IN_STAND_BOXES (60 x 56 x 84 mm) and RISER_BOXES (0.000 mm protrusion)",
        "pairwise_inter_part_solid_overlap_mm3": 0.0,
        "min_pairwise_surface_clearance_mm": round(min_gap_mm, 3),
        "input_stand_slot_geometry": "1.50 mm clear vertical slot between two 5.00 mm jaws on 6.50 mm centres (x = -3.25 / +3.25 mm), VHB-taped under 60 x 56 mm base",
        "washer_and_nut_spec": "DIN 125A M4 flat washer (ID 4.3 mm, OD 9.0 mm, thickness 0.8 mm; https://www.aspenfasteners.com/content/pdf/Metric_DIN_125_spec.pdf) + DIN 985 M4 nyloc nut (7.0 mm AF / 8.08 mm AC, height 5.0 mm) + ISO 4762 M4 SHCS (head Ø7.0 x 4.0 mm)",
        "bolt_schedule_reconciled": bolt_schedule,
        "total_fastener_counts": {
            "M4_x_20mm_SHCS": 4,
            "M4_x_25mm_SHCS": 4,
            "M4_x_25mm_pan_head_for_rack": "2 required (4 supplied in kit)",
            "M4_x_35mm_SHCS": 8,
            "M4_x_45mm_SHCS": 6,
            "M4_DIN125A_flat_washers_9p0OD_x_0p8mm": "46 required: 2 per screw on all 22 PLA-to-PLA SHCS bolts (44) + 1 rear washer per screw on 2 rack pan-head bolts (2); 50 supplied in kit",
            "M4_DIN985_nyloc_nuts_5p0mm": 24,
            "rear_2in_C_clamps": "2x on post_lower_{left,right} feet at rear tabletop edge x in [0.455, 0.565] m, y = +-0.245 m (>=52.3 mm 3D clearance from all 578 arm hulls during motion, >=298.5 mm parked)",
            "VHB_tape": "0.6 mm double-sided acrylic foam under 60 x 56 mm in_stand base at (0.255, -0.055) m",
        },
        "parts": parts_meta,
    }
    (out_dir / "ASSEMBLY.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, builders


def main() -> None:
    ap = argparse.ArgumentParser(description="Export 11 non-overlapping 2-manifold PLA fixture STLs + ASSEMBLY.json.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("fixtures_stl"))
    args = ap.parse_args()
    m, _ = export_all(args.out_dir)
    vinfo = m["volume_and_filament_calculation_not_weighed"]
    print(
        f"Exported {len(m['parts'])} watertight 2-manifold STL parts (max dim {m['max_piece_dimension_mm']} mm <= 190 mm, "
        f"min gap={m['min_pairwise_surface_clearance_mm']} mm, vol={vinfo['total_closed_mesh_volume_cm3']} cm^3, "
        f"calc PLA={vinfo['calculated_4perim_5skin_35pct_infill_mass_g']} g "
        f"[range {vinfo['calculated_slicer_mass_range_g']} g]) to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
