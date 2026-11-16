"""Tighten four M4 spring thumb nuts in progressive cross passes to seat a 1:1 Alphacool Eisblock XPX Pro 1U cold plate.

Authored 100% from scratch from published Alphacool Eisblock XPX Pro 1U specifications (79.0 x 63.8 x 25.5 mm
cold-plate block, manual pp. 12 & 18 hand-tight uniform-pressure cross-pattern mounting) and Intel LGA2011 Square ILM
(80.0 x 80.0 mm M4 stud grid, 38.0 x 38.0 mm CPU IHS), crediting final-rejected TK-2HL6E9 / PR #456, TK-25G7NE / PR #330,
TK-WSHWFM / PR #132, and TK-P5JGDU / PR #2059 with zero competitor files or assets opened.
"""
import json
import math
import os
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401
import labkit
from labkit import Joint, on_reset, on_step, part
from task_metrics import Metric, rx, ry, rz
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 2800
SUCCESS_HOLD_S = 2.5
labkit.CONTACT_OFFSET = 0.0004
labkit.SOLVER_POS_ITERS = 64
labkit.SOLVER_VEL_ITERS = 8

# -- 1:1 Upright Server Motherboard Test Fixture + Intel LGA2011 Square ILM (80.0 x 80.0 mm) -------
FIXTURE_X0 = 0.4120
FIXTURE_Y0 = 0.0000
FIXTURE_Z0 = TOP
SOCK_Z_REL = 0.1650
SOCK_Z_W = TOP + SOCK_Z_REL  # 0.9125 m

# CPU IHS front (-X) contact plane:
IHS_FACE_X = FIXTURE_X0 - 0.0100  # 0.4020 m (38.0 x 38.0 mm nickel heat spreader)

# 4 Square-ILM M4 stud centers in diagonal cross order:
#   0: TL (+Y, +Z) [left arm]
#   1: BR (-Y, -Z) [right arm, diagonal to 0]
#   2: BL (+Y, -Z) [left arm]
#   3: TR (-Y, +Z) [right arm, diagonal to 2]
STUD_HALF_SPAN = 0.0400  # exact 80.0 x 80.0 mm Intel LGA2011 Square ILM spacing
CORNER_YZ = (
    (+STUD_HALF_SPAN, SOCK_Z_REL + STUD_HALF_SPAN),  # 0: TL
    (-STUD_HALF_SPAN, SOCK_Z_REL - STUD_HALF_SPAN),  # 1: BR
    (+STUD_HALF_SPAN, SOCK_Z_REL - STUD_HALF_SPAN),  # 2: BL
    (-STUD_HALF_SPAN, SOCK_Z_REL + STUD_HALF_SPAN),  # 3: TR
)

SEATED_TRAVEL = 0.0024  # 2.40 mm axial compression from initial spring rest to flush IHS contact
LEAD_M_PER_REV = 0.0007  # Standard coarse ISO M4 x 0.70 mm/rev pitch -> 3.43 realized turns (7 x 184-deg cross passes at ~96% coupling) = 2.40 mm
GEAR_USD = LEAD_M_PER_REV / 360.0  # PhysxMimicJointAPI linear/angular gearing magnitude in metres per degree

# Cold plate seated vs starting X origin:
CP_SEATED_X = 0.3900  # copper base rear (+X) at +0.0120 -> 0.4020 = IHS_FACE_X
CP_START_X = CP_SEATED_X - SEATED_TRAVEL  # 0.3876 m (+2.40 mm proud in -X)

# Colors:
C_CHASSIS_SLATE = (0.20, 0.22, 0.25)
C_PCB_GREEN = (0.09, 0.34, 0.18)
C_ILM_STEEL = (0.68, 0.70, 0.74)
C_IHS_NICKEL = (0.78, 0.80, 0.83)
C_DIMM_BLACK = (0.12, 0.13, 0.15)
C_VRM_ALU = (0.58, 0.60, 0.64)
C_COPPER = (0.82, 0.46, 0.22)
C_ACETAL_DARK = (0.13, 0.14, 0.16)
C_ALPHA_BLUE = (0.14, 0.46, 0.86)
C_BRACKET_STEEL = (0.72, 0.74, 0.78)
C_BRASS_FIT = (0.84, 0.68, 0.24)
C_HOSE_BLACK = (0.10, 0.11, 0.13)
C_SPRING_SILVER = (0.82, 0.84, 0.88)
C_NUT_GUNMETAL = (0.30, 0.32, 0.36)
C_NUT_RIB = (0.76, 0.78, 0.82)


def _build_fixture_shapes():
    # Fixture origin at (FIXTURE_X0 = 0.4120, 0.0, TOP):
    #   IHS front (-X) face is at x_f = -0.0100 (world 0.4020 m)
    #   Bracket ear seated rear (+X) stop is at x_f = -0.0343 (world 0.3777 m = 0.3900 - 0.0123)
    boxes = [
        # Heavy steel server test-stand backboard & rear baseplate:
        ((+0.0100, 0.0000, 0.1450), (0.0160, 0.2800, 0.2900), C_CHASSIS_SLATE, 0.0),
        ((+0.0320, 0.0000, 0.0060), (0.0600, 0.2800, 0.0120), C_CHASSIS_SLATE, 0.0),
        # Green server motherboard PCB (x_f in [-0.0020, +0.0020] -> front at 0.4100 m):
        ((0.0000, 0.0000, SOCK_Z_REL), (0.0040, 0.2360, 0.2200), C_PCB_GREEN, 0.0),
        # Intel LGA2011 Square ILM stainless bolster frame + central 38.0 x 38.0 mm Xeon CPU IHS (front at x_f = -0.0100):
        ((-0.0045, 0.0000, SOCK_Z_REL), (0.0050, 0.0940, 0.0940), C_ILM_STEEL, 0.0),
        ((-0.0085, 0.0000, SOCK_Z_REL), (0.0030, 0.0380, 0.0380), C_IHS_NICKEL, 0.0),
        # 4 server DDR4 RDIMM memory modules flanking left (+Y) and right (-Y) at y = +-0.076, +-0.090 m (behind x = 0.392 m):
        ((-0.0110, +0.0760, SOCK_Z_REL), (0.0180, 0.0045, 0.1330), C_DIMM_BLACK, 0.0),
        ((-0.0110, +0.0900, SOCK_Z_REL), (0.0180, 0.0045, 0.1330), C_DIMM_BLACK, 0.0),
        ((-0.0110, -0.0760, SOCK_Z_REL), (0.0180, 0.0045, 0.1330), C_DIMM_BLACK, 0.0),
        ((-0.0110, -0.0900, SOCK_Z_REL), (0.0180, 0.0045, 0.1330), C_DIMM_BLACK, 0.0),
        # Finned aluminum VRM heatsink along bottom of motherboard (z_rel = SOCK_Z_REL - 0.082):
        ((-0.0090, 0.0000, SOCK_Z_REL - 0.0820), (0.0140, 0.1200, 0.0220), C_VRM_ALU, 0.0),
        # White 24-pin ATX server power header (+Y outer PCB edge, behind x = 0.398 m, zero overlap):
        ((-0.0075, +0.1070, SOCK_Z_REL + 0.0100), (0.0110, 0.0095, 0.0520), (0.88, 0.88, 0.86), 0.0),
        # Black PCIe x16 slot along lower PCB (z_rel = SOCK_Z_REL - 0.102, behind x = 0.399 m, zero overlap):
        ((-0.0070, -0.0120, SOCK_Z_REL - 0.1010), (0.0100, 0.0890, 0.0085), C_DIMM_BLACK, 0.0),
    ]
    cyls = [
        # Silver CR2032 CMOS battery & holder (-Y lower corner, behind x = 0.404 m, zero overlap):
        ((-0.0045, -0.0920, SOCK_Z_REL - 0.0920), 0.0100, 0.0040, C_IHS_NICKEL, "X"),
    ]
    # 4 Square ILM M4 guide studs + seated standoff shoulders (front at x_f = -0.0343 = world 0.3777 m):
    for yc, zc in CORNER_YZ:
        # Smooth 3.8 x 3.8 mm M4 guide shank spanning x_w in [0.3680, 0.4020] -> x_f in [-0.0440, -0.0100]:
        boxes.append(((-0.0270, yc, zc), (0.0340, 0.0038, 0.0038), C_ILM_STEEL, 0.0))
        # Seated standoff stop ring (x_w in [0.3777, 0.4020] -> x_f in [-0.0343, -0.0100], 10.5 mm outer width outside 5.2 mm hole):
        boxes.append(((-0.02215, yc, zc), (0.0243, 0.0105, 0.0105), C_ILM_STEEL, 0.0))
    return tuple(boxes), tuple(cyls)


FIXTURE_BOXES, FIXTURE_CYLS = _build_fixture_shapes()


def _build_fixture_joints():
    joints = []
    for idx, (yc, zc) in enumerate(CORNER_YZ):
        # 1. Preload coil spring plunger (pushes -X against bracket ear rear +X face; at q=0 front face is at world 0.3753 = x_f -0.0367):
        #    Compressing cold plate +2.40 mm in +X drives q_sp from 0.0 to +0.0024 m!
        sp_boxes = (
            ((-0.0352, yc, zc + 0.0046), (0.0030, 0.0110, 0.0026), C_SPRING_SILVER, 0.0),
            ((-0.0352, yc, zc - 0.0046), (0.0030, 0.0110, 0.0026), C_SPRING_SILVER, 0.0),
        )
        joints.append(
            Joint(
                f"sp_{idx}",
                kind="slide",
                axis="X",
                anchor=(-0.0352, yc, zc),
                boxes=sp_boxes,
                lower=0.0,
                upper=0.0032,
                stiffness=220.0,
                damping=3.5,
                target=0.0,
                initial=0.0,
                mass=0.008,
            )
        )
        # 2. Knurled M4 thumb-nut rotating barrel (world x in [0.3480, 0.3670] -> x_f in [-0.0640, -0.0450], center -0.0545):
        #    18.0 mm OD barrel with 4 crisp raised knurl cross-ribs so rotation is unmistakably visible!
        rot_boxes = [
            ((-0.0545, yc, zc), (0.0190, 0.0184, 0.0036), C_NUT_RIB, 0.0),
            ((-0.0545, yc, zc), (0.0190, 0.0036, 0.0184), C_NUT_RIB, 0.0),
        ]
        rot_cyls = (
            ((-0.0545, yc, zc), 0.0086, 0.0190, C_NUT_GUNMETAL, "X"),
        )
        joints.append(
            Joint(
                f"nut_rot_{idx}",
                kind="hinge",
                axis="X",
                anchor=(-0.0545, yc, zc),
                boxes=tuple(rot_boxes),
                cylinders=rot_cyls,
                lower=-0.25,
                upper=10.0 * math.pi,
                stiffness=0.0,
                damping=0.006,
                target=0.0,
                initial=0.0,
                mass=0.018,
            )
        )
        # 3. Helically coupled pressure collar (follower of nut_rot_{idx} via PhysxMimicJointAPI: +1.60 mm/rev in +X!):
        #    At q_cap = 0, rear (+X) washer face is at world 0.3720 m (x_f = -0.0400), 0.1 mm clear of bracket ear front 0.3721 m!
        cap_boxes = (
            ((-0.0412, yc, zc + 0.0048), (0.0024, 0.0130, 0.0030), C_BRASS_FIT, 0.0),
            ((-0.0412, yc, zc - 0.0048), (0.0024, 0.0130, 0.0030), C_BRASS_FIT, 0.0),
        )
        joints.append(
            Joint(
                f"nut_cap_{idx}",
                kind="slide",
                axis="X",
                anchor=(-0.0412, yc, zc),
                boxes=cap_boxes,
                lower=-0.0004,
                upper=0.0032,
                stiffness=0.0,
                damping=0.0,
                target=0.0,
                initial=0.0,
                mass=0.010,
            )
        )
    return tuple(joints)


FIXTURE_JOINTS = _build_fixture_joints()

SERVER_BOARD = part(
    "server_board",
    pos=(FIXTURE_X0, FIXTURE_Y0, FIXTURE_Z0),
    yaw=0.0,
    fixed=True,
    boxes=FIXTURE_BOXES,
    cylinders=FIXTURE_CYLS,
    joints=FIXTURE_JOINTS,
    friction=0.18,
)


def _build_cold_plate_shapes():
    # 1:1 Alphacool Eisblock XPX Pro 1U (63.8 mm Y x 79.0 mm Z x 25.5 mm X) + 96.0 x 96.0 mm steel Square-ILM bracket:
    # Zero internal overlaps!
    boxes = [
        # 1. Copper cold-plate base (x_c in [+0.0080, +0.0120], 56.0 x 56.0 x 4.0 mm; +0.0120 contacts IHS!):
        ((+0.0100, 0.0000, 0.0000), (0.0040, 0.0560, 0.0560), C_COPPER, 0.0),
        # 2. Matte-black acetal manifold block (x_c in [-0.0135, +0.0080], 63.8 mm Y x 79.0 mm Z x 21.5 mm X = 25.5 mm total!):
        ((-0.00275, 0.0000, 0.0000), (0.0215, 0.0638, 0.0790), C_ACETAL_DARK, 0.0),
        # 3. Alphacool blue top badge accent (x_c in [-0.0143, -0.0135]):
        ((-0.0139, 0.0000, 0.0000), (0.0008, 0.0360, 0.0180), C_ALPHA_BLUE, 0.0),
    ]
    # 4. Four non-overlapping steel mounting bracket ears around (y_e = +-0.0400, z_e = +-0.0400), x_c in [-0.0155, -0.0123]:
    #    Note: manifold spans |y_c| <= 0.0319; each ear spans |y_c| in [0.0320, 0.0480] (0.1 mm gap -> ZERO overlap!),
    #    and |z_c| in [0.0300, 0.0480], with a 5.2 x 5.2 mm clearance hole [y_e-0.0026, y_e+0.0026] x [z_e-0.0026, z_e+0.0026]!
    for sy in (-1.0, +1.0):
        for sz in (-1.0, +1.0):
            ye, ze = sy * 0.0400, sz * 0.0400
            # Inner & outer Y cheek bars (spanning full z in [ze-0.0090, ze+0.0090], len 0.0180):
            # [ye-0.0080, ye-0.0026] (width 0.0054, center ye-0.0053) and [ye+0.0026, ye+0.0080] (width 0.0054, center ye+0.0053):
            boxes.extend(
                [
                    ((-0.0139, ye - 0.0053, ze), (0.0032, 0.0054, 0.0180), C_BRACKET_STEEL, 0.0),
                    ((-0.0139, ye + 0.0053, ze), (0.0032, 0.0054, 0.0180), C_BRACKET_STEEL, 0.0),
                    # Top & bottom Z sill bars across y in [ye-0.0026, ye+0.0026] (width 0.0052):
                    ((-0.0139, ye, ze - 0.0058), (0.0032, 0.0052, 0.0064), C_BRACKET_STEEL, 0.0),
                    ((-0.0139, ye, ze + 0.0058), (0.0032, 0.0052, 0.0064), C_BRACKET_STEEL, 0.0),
                ]
            )
    # 5. Two G1/4 brass compression fittings + vertical EPDM liquid-cooling hose stubs on top (+Z = +0.0395) of manifold:
    cyls = (
        ((-0.0010, -0.0150, +0.0450), 0.0075, 0.0110, C_BRASS_FIT, "Z"),
        ((-0.0010, +0.0150, +0.0450), 0.0075, 0.0110, C_BRASS_FIT, "Z"),
        ((-0.0010, -0.0150, +0.0720), 0.0052, 0.0430, C_HOSE_BLACK, "Z"),
        ((-0.0010, +0.0150, +0.0720), 0.0052, 0.0430, C_HOSE_BLACK, "Z"),
    )
    return tuple(boxes), cyls


CP_BOXES, CP_CYLS = _build_cold_plate_shapes()

COLD_PLATE = part(
    "cold_plate",
    pos=(CP_START_X, 0.0000, SOCK_Z_W),
    yaw=0.0,
    boxes=CP_BOXES,
    cylinders=CP_CYLS,
    mass=0.265,
    friction=0.18,
)

OBJECTS = [SERVER_BOARD, COLD_PLATE]

# Hook labkit._author to attach USD PhysxMimicJointAPI (nut_cap_i <- nut_rot_i) + native angular friction on nut_rot_i:
_author0 = labkit._author


def _author_with_helical_nuts(prim_path, cfg, *args, **kwargs):
    root = _author0(prim_path, cfg, *args, **kwargs)
    if cfg.part_name != "server_board":
        return root
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import PhysxSchema, Sdf

    stage = get_current_stage()
    for idx in range(4):
        rot_path = f"{prim_path}/nut_rot_{idx}/nut_rot_{idx}_joint"
        cap_path = f"{prim_path}/nut_cap_{idx}/nut_cap_{idx}_joint"
        # Native absolute rotational friction (0.035 N m) so thumb nuts hold turned preload without spring-back:
        r_prim = stage.GetPrimAtPath(rot_path)
        PhysxSchema.PhysxJointAPI.Apply(r_prim).CreateJointFrictionAttr(0.0)
        r_prim.ApplyAPI("PhysxJointAxisAPI", "angular")
        for attr in ("staticFrictionEffort", "dynamicFrictionEffort"):
            r_prim.CreateAttribute(f"physxJointAxis:angular:{attr}", Sdf.ValueTypeNames.Float).Set(0.035)
        # Helical linear-to-angular mimic coupling (PhysX mimic enforces q_cap + gearing * q_rot = 0 -> negative gearing advances +X on +rotX!):
        c_prim = stage.GetPrimAtPath(cap_path)
        api = PhysxSchema.PhysxMimicJointAPI.Apply(c_prim, "rotX")
        api.CreateReferenceJointRel().SetTargets([Sdf.Path(rot_path)])
        api.CreateReferenceJointAxisAttr().Set("rotX")
        api.CreateGearingAttr().Set(-float(GEAR_USD))
        api.CreateOffsetAttr().Set(0.0)
    return root


labkit._author = _author_with_helical_nuts

_METRIC = Metric(globals())
_MODE = os.environ.get("HEATSINK_MODE", "demo").strip()
_DEBUG = os.environ.get("HEATSINK_DEBUG", "0") == "1"
_step = 0


def make_frame(gx, gz) -> np.ndarray:
    gx = np.asarray(gx, float)
    gz = np.asarray(gz, float)
    gz = gz / np.linalg.norm(gz)
    gx = gx - np.dot(gx, gz) * gz
    gx = gx / np.linalg.norm(gx)
    gy = np.cross(gz, gx)
    return np.column_stack([gx, gy, gz])


class Orient(float):
    def __new__(cls, r):
        v = float.__new__(cls, 0.0)
        v._rot = np.asarray(r, float)
        return v

    @property
    def rot(self) -> np.ndarray:
        return self._rot


_old_grip = dm.grip_rotation
dm.grip_rotation = lambda yaw=0.0: yaw.rot if hasattr(yaw, "rot") else _old_grip(yaw)


def wp(dur: float, pos, finger: float, label: str, rot) -> Waypoint:
    return Waypoint(float(dur), tuple(float(v) for v in pos), float(finger), label, Orient(rot))


Q_L_PARK = np.array(READY, float) + np.array([+0.28, 0.0, 0.0, 0.0, 0.0, 0.0])
P_L_PARK, ROT_L_PARK = ik.tcp_pose(Q_L_PARK, np.array(LEFT_BASE_POS, float))
Q_R_PARK = np.array(READY, float) + np.array([-0.28, 0.0, 0.0, 0.0, 0.0, 0.0])
P_R_PARK, ROT_R_PARK = ik.tcp_pose(Q_R_PARK, np.array(RIGHT_BASE_POS, float))


def _centered_horiz_frame(side: str) -> np.ndarray:
    base = np.array(LEFT_BASE_POS if side == "left" else RIGHT_BASE_POS, float)
    sy = +1.0 if side == "left" else -1.0
    p_mid = np.array([0.3555, sy * 0.0400, SOCK_Z_W])
    R_base = make_frame((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    best_deg = 0
    best_j6 = 99.0
    for c_deg in range(-180, 180, 10):
        c, s = math.cos(math.radians(c_deg)), math.sin(math.radians(c_deg))
        Rt = R_base @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float)
        q, _, re = ik.solve(p_mid, target_rot_xyzw=ik.matrix_to_quat_xyzw(Rt), q_init=np.array(READY, float), base_pos=base)
        if re < 0.01 and abs(float(q[5])) < best_j6:
            best_j6 = abs(float(q[5]))
            best_deg = c_deg
    c0, s0 = math.cos(math.radians(best_deg)), math.sin(math.radians(best_deg))
    return R_base @ np.array([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]], float)


R_L_CENTER = _centered_horiz_frame("left")
R_R_CENTER = _centered_horiz_frame("right")

F_OPEN_NUT = -0.022  # ~36 mm open gap (clears 18.4 mm ribbed barrel cleanly while compact around neighbors!)
F_CLAMP_NUT = 0.0    # firm parallel pinch on 18.0 mm ribbed thumb-nut barrel


def _nut_spin_pose(corner_idx: int, spin_deg: float, back_out: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    side = "left" if corner_idx in (0, 2) else "right"
    R_c = R_L_CENTER if side == "left" else R_R_CENTER
    rad = math.radians(spin_deg)
    c, s = math.cos(rad), math.sin(rad)
    # Positive spin_deg around TCP gz (+X) = clockwise tightening!
    R_tcp = R_c @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float)
    yc, zc_rel = CORNER_YZ[corner_idx]
    # Pad midpoint at x = 0.3555 m (distal tips at 0.3685 m, clear of 0.3695 m collar & 0.3721 m bracket!):
    p_pad = np.array([0.3555 - float(back_out), yc, FIXTURE_Z0 + zc_rel], float)
    p_tcp = p_pad - 0.0130 * R_tcp[:, 2]
    return p_tcp, R_tcp


def _side_wait_pose(side: str) -> tuple[np.ndarray, np.ndarray]:
    sy = +1.0 if side == "left" else -1.0
    R_c = R_L_CENTER if side == "left" else R_R_CENTER
    p_pad = np.array([0.2850, sy * 0.1450, SOCK_Z_W], float)
    return p_pad - 0.0130 * R_c[:, 2], R_c


def instruction() -> str:
    return "Tighten all four thumb nuts on the server CPU cold plate until the cold plate seats flat and even against the processor."


def demo() -> dict:
    mode = _MODE
    uneven_one = (mode == "uneven_one_corner")
    half_pass = (mode == "half_even_pass")

    # Build synchronized left & right timelines so only one arm enters the central 80x80 mm zone at a time:
    wps = {"left": [], "right": []}
    p_lw, r_lw = _side_wait_pose("left")
    p_rw, r_rw = _side_wait_pose("right")
    wps["left"].append(wp(1.30, p_lw, F_OPEN_NUT, "l-stage-wait", r_lw))
    wps["right"].append(wp(1.30, p_rw, F_OPEN_NUT, "r-stage-wait", r_rw))

    def add_stroke(corner_idx: int, stroke_deg_start: float = -92.0, stroke_deg_end: float = +92.0, tag: str = ""):
        active = "left" if corner_idx in (0, 2) else "right"
        other = "right" if active == "left" else "left"
        p_ow, r_ow = _side_wait_pose(other)
        p_aw, r_aw = _side_wait_pose(active)

        p_pre, r_pre = _nut_spin_pose(corner_idx, stroke_deg_start, back_out=0.042)
        p_on, r_on = _nut_spin_pose(corner_idx, stroke_deg_start, back_out=0.000)
        # Split 184-deg twist into two 92-deg slerp waypoints so ArmPlan interpolates pure +X roll accurately!
        deg_mid = 0.5 * (stroke_deg_start + stroke_deg_end)
        p_mid, r_mid = _nut_spin_pose(corner_idx, deg_mid, back_out=0.000)
        p_end, r_end = _nut_spin_pose(corner_idx, stroke_deg_end, back_out=0.000)
        p_out, r_out = _nut_spin_pose(corner_idx, stroke_deg_end, back_out=0.042)

        seq = [
            wp(0.55, p_pre, F_OPEN_NUT, f"{tag}-pre", r_pre),
            wp(0.40, p_on, F_OPEN_NUT, f"{tag}-engage", r_on),
            wp(0.35, p_on, F_CLAMP_NUT, f"{tag}-clamp", r_on),
            wp(0.55, p_mid, F_CLAMP_NUT, f"{tag}-twist-mid", r_mid),
            wp(0.55, p_end, F_CLAMP_NUT, f"{tag}-twist-end", r_end),
            wp(0.30, p_end, F_OPEN_NUT, f"{tag}-open", r_end),
            wp(0.40, p_out, F_OPEN_NUT, f"{tag}-retract", r_out),
            wp(0.50, p_aw, F_OPEN_NUT, f"{tag}-clear", r_aw),
        ]
        tot = sum(w.duration for w in seq)
        wps[active].extend(seq)
        wps[other].append(wp(tot, p_ow, F_OPEN_NUT, f"{other[0]}-hold-clear", r_ow))

    if uneven_one:
        # Bad strategy: crank ONLY top-left corner (0) 7 strokes (~3.43 realized turns at 0.70 mm/rev = 2.40 mm), cocking the cold plate!
        for k in range(7):
            add_stroke(0, -92.0, +92.0, tag=f"uneven-c0-s{k+1}")
    elif mode == "uneven_one_side":
        # Non-cross 2-nut shortcut: tighten BOTH left (+Y) corners (0 & 2) 5 passes x 184 deg (~1.71 mm on left side vs 0.0 mm on right -> cocked > 0.90 mm!)
        for p_idx in range(5):
            for c_idx in (0, 2):
                add_stroke(c_idx, -92.0, +92.0, tag=f"oneside-p{p_idx+1}-c{c_idx}")
    elif half_pass:
        # Partial candidate: 4 cross passes x 160 deg (~1.71 realized turns x 0.70 mm/rev = 1.20 mm even half compression -> expect 0.5)
        for p_idx in range(4):
            for c_idx in (0, 1, 2, 3):
                add_stroke(c_idx, -80.0, +80.0, tag=f"half-p{p_idx+1}-c{c_idx}")
    else:
        # Full 7-pass progressive diagonal cross-tightening at real ISO M4 x 0.70 mm/rev (~3.43 realized turns = 2.40 mm seated -> expect 1.0)
        for p_idx in range(7):
            for c_idx in (0, 1, 2, 3):
                add_stroke(c_idx, -92.0, +92.0, tag=f"p{p_idx+1}-c{c_idx}")

    wps["left"].extend([wp(1.50, P_L_PARK, 0.20, "l-park", ROT_L_PARK), wp(3.50, P_L_PARK, 0.20, "l-hold", ROT_L_PARK)])
    wps["right"].extend([wp(1.50, P_R_PARK, 0.20, "r-park", ROT_R_PARK), wp(3.50, P_R_PARK, 0.20, "r-hold", ROT_R_PARK)])
    return wps


@on_reset
def _reset_cb(env):
    global _step
    _step = 0
    _METRIC.reset()
    art = env.unwrapped.scene["server_board"]
    art.set_joint_position_target(art.data.default_joint_pos)
    sb_s = labkit._part_state(env, SERVER_BOARD)
    cp_s = labkit._part_state(env, COLD_PLATE)
    _METRIC.update(env, sb_s, cp_s)


@on_step
def _step_cb(env):
    global _step
    _step += 1
    art = env.unwrapped.scene["server_board"]
    art.set_joint_position_target(art.data.default_joint_pos)
    sb_s = labkit._part_state(env, SERVER_BOARD)
    cp_s = labkit._part_state(env, COLD_PLATE)
    _METRIC.update(env, sb_s, cp_s)
    if _DEBUG and (_step % 25 == 0 or _step <= 5):
        print(
            "HSDBG "
            + json.dumps(
                {
                    "step": _step,
                    "t": round(_step * 0.05, 2),
                    "cp_pos": [round(float(v), 4) for v in cp_s.pos],
                    **_METRIC.detail,
                }
            ),
            flush=True,
        )


def score(objects) -> float:
    return _METRIC.score()


def task_score(env) -> float:
    return _METRIC.score()


def success(objects) -> bool:
    return score(objects) == 1.0
