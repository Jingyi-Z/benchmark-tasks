"""578-hull installed-URDF ArmPlan.action7(t) + 3D swept bail-vs-brew-head + 16 scorer counterexamples."""
import json
import os
import math
import pathlib
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import task
from hand_geometry import read_stl, rotation, vector, world_planes
from task_metrics import box_rot, cyl_as_boxes, part_local_corners, rx, ry, rz
from yam_arena import YAM_URDF, ik
from yam_arena.constants import LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import ArmPlan
from yam_arena.urdf import forward_kinematics


class FakeState:
    def __init__(self, pos, R=None, joints=None):
        self.pos = np.asarray(pos, float)
        self._R = np.eye(3) if R is None else np.asarray(R, float)
        self.joints = {} if joints is None else dict(joints)

    def rotation(self):
        return self._R


def _polyhedra_intersect_exact(va, pa, vb, pb):
    if np.any(np.min(vb @ pa[:, :3].T + pa[:, 3], axis=0) > 1e-6):
        return False
    if np.any(np.min(va @ pb[:, :3].T + pb[:, 3], axis=0) > 1e-6):
        return False
    P = np.vstack([pa, pb])
    res = linprog(np.zeros(3), A_ub=P[:, :3], b_ub=-P[:, 3] - 1e-6, bounds=(None, None), method="highs")
    return bool(res.success)


_UNIT_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    float,
)
_BOX_LOCAL_NORMALS = np.array(
    [
        [+1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, +1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, +1.0],
        [0.0, 0.0, -1.0],
    ],
    float,
)


def _boxes_to_hulls(prefix: str, boxes, pos: np.ndarray, R: np.ndarray = None):
    if R is None:
        R = np.eye(3)
    out = []
    for k, b in enumerate(boxes):
        c_l = np.asarray(b[0], float)
        h = 0.5 * np.asarray(b[1], float)
        U = R @ box_rot(b)
        c_w = pos + R @ c_l
        pts_w = (_UNIT_SIGNS * h) @ U.T + c_w
        n_w = _BOX_LOCAL_NORMALS @ U.T
        ext = np.array([h[0], h[0], h[1], h[1], h[2], h[2]], float)
        d_w = -np.einsum("ij,j->i", n_w, c_w) - ext
        eq = np.column_stack([n_w, d_w])
        out.append((f"{prefix}_{k}", pts_w, eq, pts_w.min(0), pts_w.max(0)))
    return out


def _cyl_z_to_hull(name: str, cyl, pos: np.ndarray, R: np.ndarray = None, n_sides: int = 24):
    if R is None:
        R = np.eye(3)
    ctr = np.asarray(cyl[0], float)
    rad = float(cyl[1])
    hh = 0.5 * float(cyl[2])
    pts_l = []
    for sz in (-hh, +hh):
        for k in range(n_sides):
            ang = 2.0 * math.pi * k / n_sides
            pts_l.append(ctr + np.array([rad * math.cos(ang), rad * math.sin(ang), sz]))
    pts_w = np.asarray(pts_l, float) @ R.T + pos
    hull = ConvexHull(pts_w)
    return (name, pts_w[hull.vertices], hull.equations.copy(), pts_w.min(0), pts_w.max(0))


def check_metadata_and_blind_review():
    tj = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    allowed = {
        "Impact on jobs",
        "Impact on AI timelines",
        "Safety and alignment",
        "Societal relatability",
        "Other",
    }
    assert tj["category"] in allowed, f"Bad category: {tj['category']}"
    fid_len = len(tj["fidelity"])
    assert fid_len < 500, f"fidelity too long: {fid_len}"

    banned = tuple(t.strip().lower() for t in os.environ.get("BLIND_TERMS", "").split(",") if t.strip())  # team terms, kept out of the repo
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in (".py", ".json", ".md") and p.name != "check_path.py":
            low = p.read_text(encoding="utf-8", errors="replace").lower()
            for b in banned:
                assert b not in low, f"Banned identifier '{b}' found in {p.name}"
    print(f"[PASS] task.json category='{tj['category']}', len(fidelity)={fid_len} < 500, 0 identifiers")


def check_nested_fit_and_bail_sweep():
    I3 = np.eye(3)
    p_b = np.array([task.BREWER_X0, task.BREWER_Y0, task.BREWER_Z0])
    anc_sw = np.asarray(task.SWIVEL_ANCHOR, float)
    anc_bl = np.asarray(task.BAIL_ANCHOR, float)

    brewer_hulls = _boxes_to_hulls("br_b", task.BREWER_BOXES, p_b, I3) + [
        _cyl_z_to_hull(f"br_c_{i}", c, p_b, I3) for i, c in enumerate(task.BREWER_CYLS)
    ]

    # 1. Swivel holder + seated Removable Filter Basket vs static brewer across entire [-115, 0] deg arc:
    sw_brewer_hits = []
    for ang_deg in np.linspace(-115.0, 0.0, 47):
        R_loc = rz(math.radians(ang_deg))
        R_sw = R_loc
        p_sw = p_b + (anc_sw - R_loc @ anc_sw)
        sw_hulls = _boxes_to_hulls("sw_b", task.SWIVEL_BOXES, p_sw, R_sw) + [
            _cyl_z_to_hull(f"sw_c_{i}", c, p_sw, R_sw) for i, c in enumerate(task.SWIVEL_CYLS)
        ]
        for sname, sv, sp, slo, shi in sw_hulls:
            for bname, bv, bp, blo, bhi in brewer_hulls:
                if np.all(np.minimum(shi, bhi) - np.maximum(slo, blo) > 1e-6):
                    if _polyhedra_intersect_exact(sv, sp, bv, bp):
                        sw_brewer_hits.append((round(float(ang_deg), 1), sname, bname))
    assert not sw_brewer_hits, f"Swivel vs brewer collisions: {sw_brewer_hits[:6]}"
    print("[PASS] Swivel holder + Removable Filter Basket clear static brewer & carafe across [-115, 0] deg (0 hits)")

    # 2. Concentrically seated PERM_FILTER vs Removable Filter Basket staves & floor (at 0.3 mm above Z_SEAT_FLOOR):
    p_f_seated = p_b + np.array([task.BASKET_CENTER_B[0], task.BASKET_CENTER_B[1], task.Z_SEAT_FLOOR + 0.0003])
    fl_hulls = _boxes_to_hulls("fl_b", task.FILTER_BOXES, p_f_seated, I3) + [
        _cyl_z_to_hull(f"fl_c_{i}", c, p_f_seated, I3) for i, c in enumerate(task.FILTER_CYLS)
    ]
    sw_closed_hulls = _boxes_to_hulls("sw0_b", task.SWIVEL_BOXES, p_b, I3) + [
        _cyl_z_to_hull(f"sw0_c_{i}", c, p_b, I3) for i, c in enumerate(task.SWIVEL_CYLS)
    ]
    nested_hits = []
    for fname, fv, fp, flo, fhi in fl_hulls:
        for sname, sv, sp, slo, shi in sw_closed_hulls:
            if np.all(np.minimum(fhi, shi) - np.maximum(flo, slo) > 1e-6):
                if _polyhedra_intersect_exact(fv, fp, sv, sp):
                    nested_hits.append((fname, sname))
    assert not nested_hits, f"Seated filter overlaps Removable Filter Basket: {nested_hits[:6]}"
    print("[PASS] Seated Cafe Brew BF215 clears Removable Filter Basket 16-stave funnel & rim (0 hits)")

    # 3. 3D swept bail-vs-brew-head overhang collision audit across swivel in [-112, 0] deg x bail angles:
    head_hulls = [brewer_hulls[7], brewer_hulls[8], brewer_hulls[len(task.BREWER_BOXES) + 3]]
    summary = {}
    for bail_deg in (0.0, 6.0, 12.0, 16.0, 20.0, 35.0, 60.0, 90.5):
        bail_rad = -math.radians(bail_deg)
        first_hit_deg = None
        min_vert_clear_mm = 999.0
        for sw_deg in np.linspace(-112.0, 0.0, 113):
            R_sw = rz(math.radians(sw_deg))
            p_sw = p_b + (anc_sw - R_sw @ anc_sw)
            p_f = p_sw + R_sw @ np.array([task.BASKET_CENTER_B[0], task.BASKET_CENTER_B[1], task.Z_SEAT_FLOOR])
            # Filter sits inside open basket at R_f = rz(FILTER_SEAT_YAW) when loaded at -111.2 deg and co-rotates by (sw_deg - (-111.2)):
            R_f = rz(task.FILTER_SEAT_YAW + math.radians(sw_deg - (-111.2)))
            R_loc_bl = rx(bail_rad)
            R_bl = R_f @ R_loc_bl
            p_bl = p_f + R_f @ (anc_bl - R_loc_bl @ anc_bl)
            bl_hulls = _boxes_to_hulls("bl", task.BAIL_BOXES, p_bl, R_bl)
            z_top_bail = max(float(h[4][2]) for h in bl_hulls)
            min_vert_clear_mm = min(min_vert_clear_mm, (float(p_b[2]) + task.Z_HEAD_BOTTOM - z_top_bail) * 1000.0)
            hit_here = False
            for _, bv, bp, blo, bhi in bl_hulls:
                for _, hv, hp, hlo, hhi in head_hulls:
                    if np.all(np.minimum(bhi, hhi) - np.maximum(blo, hlo) > 1e-6):
                        if _polyhedra_intersect_exact(bv, bp, hv, hp):
                            hit_here = True
                            break
                if hit_here:
                    break
            if hit_here and first_hit_deg is None:
                first_hit_deg = round(float(sw_deg), 1)
        summary[bail_deg] = (round(min_vert_clear_mm, 2), first_hit_deg)

    print(f"[INFO] 3D swept bail-vs-brew-head table (bail_deg: (vert_clear_mm, first_hit_swivel_deg)): {summary}")
    for safe_deg in (0.0, 6.0, 12.0):
        vc_mm, hit_sw = summary[safe_deg]
        assert hit_sw is None and vc_mm >= 3.40, f"Bail at {safe_deg} deg failed clearance: vc={vc_mm}mm, hit={hit_sw}"
    for block_deg in (20.0, 35.0, 60.0, 90.5):
        _, hit_sw = summary[block_deg]
        assert hit_sw is not None and abs(hit_sw) >= 15.0, f"Bail at {block_deg} deg did not physically block swivel: {hit_sw}"
    ajar_90 = abs(summary[90.5][1])
    assert 35.0 <= ajar_90 <= 75.0, f"Unexpected upright-bail ajar stop angle: {ajar_90} deg"
    print(
        f"[PASS] 3D bail-to-brew-head interlock verified: <=12.0 deg clears full sweep by >={summary[12.0][0]:.2f}mm; "
        f"90.5 deg upright bail physically blocks swivel ajar at {ajar_90:.1f} deg"
    )


def check_scorer_counterexamples():
    m = task.Metric(vars(task))
    far_tips = np.tile(np.array([0.0, +0.48, task.TOP + 0.44]), (6, 1))
    I3 = np.eye(3)
    p_b = np.array([task.BREWER_X0, task.BREWER_Y0, task.BREWER_Z0])
    anc_sw = np.asarray(task.SWIVEL_ANCHOR, float)

    def seated_filter_pos(swivel_deg: float, dz: float = 0.0, dx: float = 0.0, dy: float = 0.0):
        R_sw = rz(math.radians(swivel_deg))
        p_sw = p_b + (anc_sw - R_sw @ anc_sw)
        return p_sw + R_sw @ np.array([task.BASKET_CENTER_B[0] + dx, task.BASKET_CENTER_B[1] + dy, task.Z_SEAT_FLOOR + dz])

    def eval_case(p_br, R_br, sw_deg, p_fl, R_fl, bail_deg, tips=far_tips, drift=0.0):
        bs = FakeState(p_br, R=R_br, joints={"swivel_holder": math.radians(sw_deg)})
        fs = FakeState(p_fl, R=R_fl, joints={"bail": -math.radians(bail_deg)})
        _, _, p_sw, R_sw, _ = m.brewer_link_poses(bs)
        _, _, p_bl, R_bl, _ = m.filter_link_poses(fs)
        pts0 = np.vstack([
            m.brewer_corners @ R_br.T + p_br,
            m.swivel_corners @ R_sw.T + p_sw,
            m.filter_corners @ R_fl.T + p_fl,
            m.bail_corners @ R_bl.T + p_bl,
        ])
        hist = [pts0.copy() for _ in range(20)] + [pts0 + np.array([drift, 0.0, 0.0])]
        return m.evaluate_states(bs, fs, tips, hist_override=hist)

    p_f_counter = np.array([task.FILTER_X0, task.FILTER_Y0, task.FILTER_Z0])
    # 1. Start layout -> 0.0
    assert eval_case(p_b, I3, 0.0, p_f_counter, I3, 90.5) == 0.0
    # 2. Counter filter with bail folded -> 0.0
    assert eval_case(p_b, I3, 0.0, p_f_counter, I3, 0.0) == 0.0
    # 3. Full valid goal (bail folded 0 deg, swivel closed 0 deg) -> 1.0
    assert eval_case(p_b, I3, 0.0, seated_filter_pos(0.0), I3, 0.0) == 1.0
    # 4. Full valid goal with filter yawed 180 deg & bail at 10.5 deg & swivel at -3.2 deg -> 1.0
    assert eval_case(p_b, I3, -3.2, seated_filter_pos(-3.2), rz(math.pi), 10.5) == 1.0
    # 5. Upright bail blocking swivel ajar at -54 deg -> 0.5
    assert eval_case(p_b, I3, -54.0, seated_filter_pos(-54.0), I3, 90.5) == 0.5
    # 6. Seated + bail folded, but swivel left open at -110 deg -> 0.5
    assert eval_case(p_b, I3, -110.0, seated_filter_pos(-110.0), I3, 0.0) == 0.5
    # 7. Impossible tunneled state (swivel closed 0 deg while bail upright 45 deg) -> 0.0
    assert eval_case(p_b, I3, 0.0, seated_filter_pos(0.0), I3, 45.0) == 0.0
    # 8. Bail just over 12.0 deg threshold (15.0 deg) with closed swivel -> 0.0
    assert eval_case(p_b, I3, 0.0, seated_filter_pos(0.0), I3, 15.0) == 0.0
    # 9. Off-center filter (+12 mm radial offset) -> 0.0
    assert eval_case(p_b, I3, 0.0, seated_filter_pos(0.0, dx=0.012), I3, 0.0) == 0.0
    # 10. High/cocked filter (+10 mm above seat floor) -> 0.0
    assert eval_case(p_b, I3, -55.0, seated_filter_pos(-55.0, dz=0.010), I3, 0.0) == 0.0
    # 11. Cocked filter axis (14 deg tilt) -> 0.0
    assert eval_case(p_b, I3, 0.0, seated_filter_pos(0.0), rx(math.radians(14.0)), 0.0) == 0.0
    # 12. Tilted coffee maker (11 deg tilt) -> 0.0
    R_tilt = ry(math.radians(11.0))
    assert eval_case(p_b, R_tilt, 0.0, p_b + R_tilt @ np.array([-0.080, 0.0, task.Z_SEAT_FLOOR]), R_tilt, 0.0) == 0.0
    # 13. Dropped filter below countertop -> 0.0
    assert eval_case(p_b, I3, 0.0, np.array([0.30, -0.20, task.TOP - 0.045]), I3, 0.0) == 0.0
    # 14. Hand 65 mm from brewer housing -> 0.0
    near_br = far_tips.copy()
    near_br[0] = p_b + np.array([+0.080, +0.101 + 0.065, 0.180])
    assert eval_case(p_b, I3, 0.0, seated_filter_pos(0.0), I3, 0.0, tips=near_br) == 0.0
    # 15. Hand 65 mm from open swivel holder -> 0.0
    near_sw = far_tips.copy()
    near_sw[3] = seated_filter_pos(-54.0) + np.array([-0.065 - 0.055, 0.0, 0.030])
    assert eval_case(p_b, I3, -54.0, seated_filter_pos(-54.0), I3, 90.5, tips=near_sw) == 0.0
    # 16. Stillness violation (1.3 mm drift over 21 samples) -> 0.0
    assert eval_case(p_b, I3, 0.0, seated_filter_pos(0.0), I3, 0.0, drift=0.0013) == 0.0
    print("[PASS] 16/16 CPU scorer counterexamples verified")


def _expected_part_poses(mode: str, t: float):
    I3 = np.eye(3)
    p_b = np.array([task.BREWER_X0, task.BREWER_Y0, task.BREWER_Z0])
    anc_sw = np.asarray(task.SWIVEL_ANCHOR, float)

    if mode == "unseated_close":
        if t <= 3.0:
            sw_deg = 0.0
        elif t <= 5.7:
            sw_deg = -111.0 * ((t - 3.0) / 2.7)
        elif t <= 8.5:
            sw_deg = -111.0 * (1.0 - (t - 5.7) / 2.8)
        else:
            sw_deg = 0.0
        p_f = np.array([task.FILTER_X0, task.FILTER_Y0, task.FILTER_Z0])
        return p_b, I3, sw_deg, p_f, rz(task.FILTER_YAW0), 90.5

    fold_dur = 0.7 if mode == "upright_bail_close" else 4.7
    t_close_start = 18.95 + fold_dur + 2.5  # after re-pre-tab (1.1) + re-on-tab (0.7) + re-clamp-tab (0.7) = 2.5 s
    # Exact piecewise swivel arc matching left_wps:
    if mode == "upright_bail_close":
        close_knots = [(0.0, -111.2), (0.8, -84.0), (1.9, -59.5)]
        end_sw = -59.5
    else:
        close_knots = [
            (0.0, -111.2),
            (0.8, -84.0),
            (1.6, -56.0),
            (2.5, -28.0),
            (3.5, -2.0),
            (4.2, 0.0),
        ]
        end_sw = 0.0

    if t <= 3.0:
        sw_deg = 0.0
    elif t <= 6.4:
        open_knots = [(3.0, 0.0), (3.8, -28.0), (4.6, -56.0), (5.5, -84.0), (6.4, -111.2)]
        sw_deg = -111.2
        for (ta, aa), (tb, ab) in zip(open_knots, open_knots[1:]):
            if t <= tb:
                u = (t - ta) / (tb - ta)
                s = u * u * (3.0 - 2.0 * u)
                sw_deg = aa + s * (ab - aa)
                break
    elif t <= t_close_start:
        sw_deg = -111.2
    else:
        dt_c = t - t_close_start
        sw_deg = end_sw
        for (ta, aa), (tb, ab) in zip(close_knots, close_knots[1:]):
            if dt_c <= tb:
                u = (dt_c - ta) / (tb - ta)
                s = u * u * (3.0 - 2.0 * u)
                sw_deg = aa + s * (ab - aa)
                break

    R_sw = rz(math.radians(sw_deg))
    p_sw = p_b + (anc_sw - R_sw @ anc_sw)
    seat_w = p_sw + R_sw @ np.array([task.BASKET_CENTER_B[0], task.BASKET_CENTER_B[1], task.Z_SEAT_FLOOR])

    p_f0 = np.array([task.FILTER_X0, task.FILTER_Y0, task.FILTER_Z0])
    p_f_lift = np.array([task.FILTER_X0, task.FILTER_Y0, task.TOP + 0.3040])
    p_f_cy1 = np.array([0.1820, -0.1650, float(seat_w[2]) + 0.0780])
    p_f_cy2 = np.array([0.1820, float(seat_w[1]) + 0.0330, float(seat_w[2]) + 0.0780])
    p_f_above = seat_w + np.array([0.0, 0.0, +0.0760])
    yaw_cy1 = 0.65 * task.FILTER_YAW0 + 0.35 * task.FILTER_SEAT_YAW
    yaw_cy2 = 0.20 * task.FILTER_YAW0 + 0.80 * task.FILTER_SEAT_YAW

    if t <= 10.0:
        p_f = p_f0
        R_f = rz(task.FILTER_YAW0)
    elif t <= 11.2:
        u = (t - 10.0) / 1.2
        p_f = (1.0 - u) * p_f0 + u * p_f_lift
        R_f = rz(task.FILTER_YAW0)
    elif t <= 12.05:
        u = (t - 11.2) / 0.85
        p_f = (1.0 - u) * p_f_lift + u * p_f_cy1
        R_f = rz((1.0 - u) * task.FILTER_YAW0 + u * yaw_cy1)
    elif t <= 12.90:
        u = (t - 12.05) / 0.85
        p_f = (1.0 - u) * p_f_cy1 + u * p_f_cy2
        R_f = rz((1.0 - u) * yaw_cy1 + u * yaw_cy2)
    elif t <= 13.75:
        u = (t - 12.90) / 0.85
        p_f = (1.0 - u) * p_f_cy2 + u * p_f_above
        R_f = rz((1.0 - u) * yaw_cy2 + u * task.FILTER_SEAT_YAW)
    elif t <= 15.65:
        u = (t - 13.75) / 1.90
        p_f = (1.0 - u) * p_f_above + u * (seat_w + np.array([0.0, 0.0, 0.0015]))
        R_f = rz(task.FILTER_SEAT_YAW)
    else:
        p_f = seat_w + np.array([0.0, 0.0, 0.0005])
        R_f = rz(task.FILTER_SEAT_YAW + (math.radians(sw_deg - (-111.2)) if t > t_close_start else 0.0))

    if mode == "upright_bail_close" or t <= 18.85:
        bail_deg = 90.5
    elif t <= 20.55:
        bail_deg = 90.5 * (1.0 - (t - 18.85) / 1.70)
    else:
        bail_deg = 0.0

    return p_b, I3, sw_deg, p_f, R_f, bail_deg


def check_full_trajectory():
    urdf = pathlib.Path(YAM_URDF)
    root = ET.parse(urdf).getroot()
    shapes = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        if name in ("base", "world"):
            continue
        for c in link.findall("collision"):
            mesh = c.find("geometry/mesh")
            if mesh is None:
                continue
            path = (urdf.parent / mesh.get("filename")).resolve()
            o = c.find("origin")
            r = rotation(vector(o.get("rpy") if o is not None else None, [0, 0, 0]))
            p = vector(o.get("xyz") if o is not None else None, [0, 0, 0])
            v = (read_stl(path) * vector(mesh.get("scale"), [1, 1, 1])) @ r.T + p
            v = np.unique(np.round(v.reshape(-1, 3), 7), axis=0)
            hull = ConvexHull(v)
            shapes.append((name, v[hull.vertices], hull.equations.copy()))
    print(f"[INFO] Loaded {len(shapes)} collision hulls from YAM_URDF (2 arms = {2 * len(shapes)} hulls)")

    bases = [np.array(RIGHT_BASE_POS, float), np.array(LEFT_BASE_POS, float)]
    anc_sw = np.asarray(task.SWIVEL_ANCHOR, float)
    anc_bl = np.asarray(task.BAIL_ANCHOR, float)

    def placed(i, q, f):
        joints = {f"joint{k + 1}": float(q[k]) for k in range(6)}
        joints.update(joint7=f, joint8=f)
        fk = forward_kinematics(urdf, joints)
        out = []
        for name, v, planes in shapes:
            tf = fk[name].copy()
            tf[:3, 3] += bases[i]
            vv = v @ tf[:3, :3].T + tf[:3, 3]
            out.append((name, vv, planes, tf[:3, :3], tf[:3, 3], vv.min(0), vv.max(0)))
        return out

    p_b0 = np.array([task.BREWER_X0, task.BREWER_Y0, task.BREWER_Z0])
    brewer_static_hulls = _boxes_to_hulls("br", task.BREWER_BOXES, p_b0, np.eye(3)) + [
        _cyl_z_to_hull(f"br_c_{i}", c, p_b0, np.eye(3)) for i, c in enumerate(task.BREWER_CYLS)
    ]

    for mode in ("demo", "upright_bail_close", "unseated_close"):
        task._MODE = mode
        task._LIVE_BREWER = None
        task._LIVE_FILTER = None
        task._BAIL_SNAP = None

        plan = task.demo()
        aps = [ArmPlan(plan["right"], RIGHT_BASE_POS), ArmPlan(plan["left"], LEFT_BASE_POS)]
        duration = max(sum(w.duration for w in v) for v in plan.values())
        worst_pos = worst_rot = 0.0
        max_carry_dq = 0.0
        max_carry_q3 = 0.0
        inter_hits = []
        below_hits = []
        unintended_hits = []
        min_inter_aabb_mm = 999.0
        min_top_mm = 999.0
        dt = 0.05 if mode == "demo" else 0.10
        q_prev_r = aps[0].q.copy()

        fold_dur = 0.7 if mode == "upright_bail_close" else 4.7
        t_close_start = 18.95 + fold_dur + 2.5

        for t in np.arange(0.0, duration + 0.001, dt):
            p_b, R_b, sw_deg, p_f, R_f, bail_deg = _expected_part_poses(mode, float(t))
            task._LIVE_BREWER = dict(p_b=p_b, R_b=R_b, swivel_rad=math.radians(sw_deg))
            task._LIVE_FILTER = dict(p_f=p_f, R_f=R_f, bail_rad=-math.radians(bail_deg))

            bodies = []
            for i in range(2):
                tcp, rot, _, _ = aps[i].target(float(t))
                act7, _ = aps[i].action7(float(t))
                q = np.asarray(act7[:6], float)
                if i == 0:
                    if mode != "unseated_close" and 9.5 <= t <= 17.4:
                        max_carry_dq = max(max_carry_dq, float(np.max(np.abs(q - q_prev_r))) * (0.05 / dt))
                        max_carry_q3 = max(max_carry_q3, float(q[2]))
                    q_prev_r = q.copy()
                tcp_act, rot_act = ik.tcp_pose(q, bases[i])
                pe = float(np.linalg.norm(tcp_act - tcp))
                cos_ang = float(np.clip((np.trace(rot_act.T @ rot) - 1.0) * 0.5, -1.0, 1.0))
                re = math.acos(cos_ang)
                if pe * 1000.0 > 0.5 and len(inter_hits) == 0:
                    print(f"  [IK>0.5mm] arm={i} t={t:.2f}s pe={pe*1000:.2f}mm re={np.degrees(re):.2f}deg tcp={np.round(tcp, 3)}")
                worst_pos = max(worst_pos, pe * 1000.0)
                worst_rot = max(worst_rot, float(np.degrees(re)))
                bodies.append(placed(i, q, float(act7[6]) * float(task.FINGER_OPEN)))

            alo = np.array([v[5] for v in bodies[0]])
            ahi = np.array([v[6] for v in bodies[0]])
            blo = np.array([v[5] for v in bodies[1]])
            bhi = np.array([v[6] for v in bodies[1]])
            gap = np.maximum(0.0, np.maximum(alo[:, None, :] - bhi[None, :, :], blo[None, :, :] - ahi[:, None, :]))
            min_inter_aabb_mm = min(min_inter_aabb_mm, float(np.min(np.linalg.norm(gap, axis=-1)) * 1000.0))
            cands = np.all(np.minimum(ahi[:, None], bhi[None, :]) - np.maximum(alo[:, None], blo[None, :]) > 1e-6, axis=-1)
            for ia, ib in np.argwhere(cands):
                aa, bb = bodies[0][ia], bodies[1][ib]
                if _polyhedra_intersect_exact(aa[1], world_planes(aa[2], aa[3], aa[4]), bb[1], world_planes(bb[2], bb[3], bb[4])):
                    inter_hits.append((round(float(t), 2), aa[0], bb[0]))

            for side, arm_b in (("right", bodies[0]), ("left", bodies[1])):
                for lname, vv, planes, R_l, p_l, lo, hi in arm_b:
                    if lname in ("link1", "link2"):
                        continue
                    dz_top = float(lo[2] - task.TOP) * 1000.0
                    min_top_mm = min(min_top_mm, dz_top)
                    if lo[2] < task.TOP + 0.0020:
                        below_hits.append((round(float(t), 2), side, lname, round(dz_top, 2)))

            # Scene hulls at current time t:
            R_sw = rz(math.radians(sw_deg))
            p_sw = p_b + (anc_sw - R_sw @ anc_sw)
            sw_hulls = _boxes_to_hulls("sw", task.SWIVEL_BOXES, p_sw, R_sw) + [
                _cyl_z_to_hull(f"sw_c_{i}", c, p_sw, R_sw) for i, c in enumerate(task.SWIVEL_CYLS)
            ]
            fl_hulls = _boxes_to_hulls("fl", task.FILTER_BOXES, p_f, R_f) + [
                _cyl_z_to_hull(f"fl_c_{i}", c, p_f, R_f) for i, c in enumerate(task.FILTER_CYLS)
            ]
            R_loc_bl = rx(-math.radians(bail_deg))
            R_bl = R_f @ R_loc_bl
            p_bl = p_f + R_f @ (anc_bl - R_loc_bl @ anc_bl)
            bl_hulls = _boxes_to_hulls("bl", task.BAIL_BOXES, p_bl, R_bl)

            # Carried filter + bail vs static brewer housing/hood (and vs open swivel basket before final seating),
            # including +-5.0 deg carry-tilt perturbations (rx/ry):
            tilt_perturbs = [np.eye(3)]
            if mode != "unseated_close" and 10.2 <= t <= 14.80:
                tilt_perturbs.extend([rx(math.radians(5.0)), rx(math.radians(-5.0)), ry(math.radians(5.0)), ry(math.radians(-5.0))])
            piv_f_w = p_f + R_f @ np.array([0.0, 0.0, 0.0930])
            for R_pt in tilt_perturbs:
                if R_pt is tilt_perturbs[0]:
                    cand_hulls = fl_hulls + bl_hulls
                else:
                    R_f_t = R_pt @ R_f
                    p_f_t = piv_f_w - R_pt @ (piv_f_w - p_f)
                    R_bl_t = R_f_t @ R_loc_bl
                    p_bl_t = p_f_t + R_f_t @ (anc_bl - R_loc_bl @ anc_bl)
                    cand_hulls = (
                        _boxes_to_hulls("fl_t", task.FILTER_BOXES, p_f_t, R_f_t)
                        + [_cyl_z_to_hull(f"fl_tc_{i}", c, p_f_t, R_f_t) for i, c in enumerate(task.FILTER_CYLS)]
                        + _boxes_to_hulls("bl_t", task.BAIL_BOXES, p_bl_t, R_bl_t)
                    )
                for fname, fpts, feq, flo, fhi in cand_hulls:
                    for hname, hpts, heq, hlo, hhi in brewer_static_hulls:
                        if mode == "upright_bail_close" and t > (t_close_start + 1.6) and fname.startswith("bl"):
                            continue
                        if np.all(np.minimum(fhi, hhi) - np.maximum(flo, hlo) > 1e-6):
                            if _polyhedra_intersect_exact(fpts, feq, hpts, heq):
                                unintended_hits.append((round(float(t), 2), "filter_carry", fname, hname))
                    if mode != "unseated_close" and t <= 15.50 and R_pt is tilt_perturbs[0]:
                        for sname, spts, seq_p, slo, shi in sw_hulls:
                            if np.all(np.minimum(fhi, shi) - np.maximum(flo, slo) > 1e-6):
                                if _polyhedra_intersect_exact(fpts, feq, spts, seq_p):
                                    unintended_hits.append((round(float(t), 2), "filter_carry", fname, sname))

            # Right-arm unintended collision screen:
            # Right fingertips intentionally press straight down onto the flat brewer roof (br_1, br_2, br_6, br_c_3 at z >= TOP + 0.348)
            # during opening and closing, and contact the bail during pickup/fold:
            if mode == "unseated_close":
                r_brace_win = 1.3 <= t <= 9.6
            else:
                r_brace_win = (1.3 <= t <= 6.7) or ((18.45 + fold_dur) <= t <= (26.20 + fold_dur))
            r_bail_win = False if mode == "unseated_close" else (8.0 <= t <= (17.55 + fold_dur))

            for lname, vv, planes, R_l, p_l, lo, hi in bodies[0]:
                is_tip = lname in ("tip_left", "tip_right")
                pl_arm = None
                for hname, hpts, heq, hlo, hhi in brewer_static_hulls:
                    if is_tip and r_brace_win and (lo[2] >= task.TOP + 0.347) and hname in ("br_1", "br_2", "br_6", "br_7", "br_8", "br_c_3"):
                        continue
                    if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                        if pl_arm is None:
                            pl_arm = world_planes(planes, R_l, p_l)
                        if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                            unintended_hits.append((round(float(t), 2), "right", lname, hname))
                for hname, hpts, heq, hlo, hhi in sw_hulls + fl_hulls:
                    if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                        if pl_arm is None:
                            pl_arm = world_planes(planes, R_l, p_l)
                        if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                            unintended_hits.append((round(float(t), 2), "right", lname, hname))
                if not (is_tip and r_bail_win):
                    for hname, hpts, heq, hlo, hhi in bl_hulls:
                        if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                            if pl_arm is None:
                                pl_arm = world_planes(planes, R_l, p_l)
                            if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                                unintended_hits.append((round(float(t), 2), "right", lname, hname))

            # Left-arm unintended collision screen:
            # Left tips intentionally clamp only the pull tab (sw_33 neck + sw_37 cyl) during open/close windows:
            if mode == "unseated_close":
                l_tab_win = 1.2 <= t <= 10.2
            else:
                l_tab_win = (1.2 <= t <= 8.0) or ((18.75 + fold_dur) <= t <= (28.35 + fold_dur))
            tab_allowed = {"sw_33", "sw_c_1"}

            for lname, vv, planes, R_l, p_l, lo, hi in bodies[1]:
                is_tip = lname in ("tip_left", "tip_right")
                pl_arm = None
                for hname, hpts, heq, hlo, hhi in brewer_static_hulls + fl_hulls + bl_hulls:
                    if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                        if pl_arm is None:
                            pl_arm = world_planes(planes, R_l, p_l)
                        if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                            unintended_hits.append((round(float(t), 2), "left", lname, hname))
                for hname, hpts, heq, hlo, hhi in sw_hulls:
                    if is_tip and l_tab_win and hname in tab_allowed:
                        continue
                    if np.all(np.minimum(hi, hhi) - np.maximum(lo, hlo) > 1e-6):
                        if pl_arm is None:
                            pl_arm = world_planes(planes, R_l, p_l)
                        if _polyhedra_intersect_exact(vv, pl_arm, hpts, heq):
                            unintended_hits.append((round(float(t), 2), "left", lname, hname))

        print(
            f"[INFO] mode={mode}: dur={duration:.2f}s worst_pos={worst_pos:.4f}mm worst_rot={worst_rot:.4f}deg "
            f"max_carry_dq={max_carry_dq:.4f}rad max_carry_q3={max_carry_q3:.2f}rad "
            f"inter_hits={len(inter_hits)} below_top={len(below_hits)} unintended_hits={len(unintended_hits)} "
            f"min_inter_aabb={min_inter_aabb_mm:.2f}mm min_top={min_top_mm:.2f}mm"
        )
        if unintended_hits:
            from collections import defaultdict
            by_pair = defaultdict(list)
            for th, sd, ln, hn in unintended_hits:
                by_pair[(sd, ln, hn)].append(th)
            for k, ts in sorted(by_pair.items(), key=lambda x: min(x[1]))[:20]:
                print(f"  [UNINTENDED] {k}: count={len(ts)}, t=[{min(ts):.2f}..{max(ts):.2f}]")
        assert worst_pos < 1.0, f"IK pos err {worst_pos:.3f}mm in {mode}"
        assert worst_rot < 1.0, f"IK rot err {worst_rot:.3f}deg in {mode}"
        assert max_carry_dq < 0.16, f"Carry joint step {max_carry_dq:.3f}rad >= 0.16 in {mode}"
        assert max_carry_q3 < 2.25, f"Carry elbow q3 {max_carry_q3:.2f}rad >= 2.25 in {mode}"
        assert not inter_hits, f"Inter-arm hits in {mode}: {inter_hits[:6]}"
        assert not below_hits, f"Below-table hits in {mode}: {below_hits[:6]}"
        assert not unintended_hits, f"Unintended part hits in {mode}: {unintended_hits[:10]}"

    task._MODE = "demo"
    task._LIVE_BREWER = None
    task._LIVE_FILTER = None
    task._BAIL_SNAP = None
    print("[PASS] Full 578-hull trajectory + swept brewer/swivel/filter/bail checks verified across all 3 modes")


if __name__ == "__main__":
    check_metadata_and_blind_review()
    check_nested_fit_and_bail_sweep()
    check_scorer_counterexamples()
    check_full_trajectory()
    print("ALL PREFLIGHT CHECKS PASSED")
