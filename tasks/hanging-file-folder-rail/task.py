"""Clamp a V-hinged hanging file folder around a loose packet and hang it on frame rails.

Modeled 1:1 on an Officemate 91990 letter-size steel hanging file frame set to its 14 in
(355.6 mm) depth (317.6 mm center-to-center rail span, 238 mm rail height), two loose
already-filed Pendaflex 4152 letter-size hanging folders near the rear of the frame, and a
target Pendaflex 4152 hanging folder (328 mm top hanger-rod span, 298.5 x 230 mm pressboard
leaves, 1/5-cut index tab, 0.16 kg) enclosing a loose ~60-sheet printed US-letter document
packet in a Smead 10330 1/3-cut manila folder cover (280 x 195 x 6 mm core stack with an
exposed 62 x 180 x 4 mm printed top-sheet header and 12 x 88 x 2 mm 1/3-cut manila tab, 0.20 kg).
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
from labkit import Joint, MATTE_COLORS, METALLIC_COLORS, Mesh, on_reset, on_step, part
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 900
SUCCESS_HOLD_S = 2.5
labkit.CONTACT_OFFSET = 0.0004

# Frame & hardware colors
STEEL_DARK = (0.20, 0.22, 0.25)
STEEL_RAIL = (0.76, 0.79, 0.83)
ROD_WHITE = (0.91, 0.92, 0.93)

# Natural butcher-block oak pickup riser colors (matte wood, distinct from metal & paper)
WOOD_OAK = (0.71, 0.55, 0.36)
WOOD_LIGHT = (0.77, 0.62, 0.42)
WOOD_DARK = (0.63, 0.47, 0.30)
WOOD_BASE = (0.28, 0.23, 0.18)
MARK_DARK = (0.16, 0.15, 0.14)

# Matte Pendaflex 4152 pressboard/paper-stock & Smead manila/paper document colors
FOLDER_GREEN = (0.21, 0.41, 0.28)
FOLDER_INNER = (0.61, 0.69, 0.55)
FOLDER_REINF = (0.15, 0.31, 0.21)
MANILA = (0.87, 0.76, 0.50)
PAPER_WHITE = (0.96, 0.95, 0.92)
PAPER_CREAM = (0.92, 0.90, 0.86)
PAPER_EDGE = (0.84, 0.82, 0.78)
LABEL_WHITE = (0.97, 0.97, 0.95)
INK_HEADER = (0.17, 0.24, 0.35)
INK_LINE = (0.27, 0.29, 0.32)

# Staggered 1/5-cut clip-on plastic index tabs
TAB_BLUE = (0.23, 0.47, 0.73)
TAB_AMBER = (0.78, 0.52, 0.18)
TAB_CLEAR = (0.76, 0.81, 0.85)

METALLIC_COLORS.add(STEEL_RAIL)
MATTE_COLORS.update({
    STEEL_DARK,
    WOOD_OAK,
    WOOD_LIGHT,
    WOOD_DARK,
    WOOD_BASE,
    MARK_DARK,
    FOLDER_GREEN,
    FOLDER_INNER,
    FOLDER_REINF,
    MANILA,
    PAPER_WHITE,
    PAPER_CREAM,
    PAPER_EDGE,
    LABEL_WHITE,
    INK_HEADER,
    INK_LINE,
})

_BOX_TRIS = np.array(
    [
        [0, 2, 1], [1, 2, 3],  # -X
        [4, 5, 6], [5, 7, 6],  # +X
        [0, 1, 4], [1, 5, 4],  # -Y
        [2, 6, 3], [3, 6, 7],  # +Y
        [0, 4, 2], [2, 4, 6],  # -Z
        [1, 3, 5], [3, 7, 5],  # +Z
    ],
    dtype=np.int32,
)


def _box_mesh(center, size, color, pitch: float = 0.0) -> Mesh:
    """Visual-only box mesh (collide='none') for crisp paper edges, printed lines, labels, and wood grain."""
    c = np.asarray(center, float)
    hx, hy, hz = 0.5 * np.asarray(size, float)
    pts = np.array(
        [[sx * hx, sy * hy, sz * hz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        float,
    )
    if pitch:
        cp, sp = math.cos(pitch), math.sin(pitch)
        rp = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        pts = pts @ rp.T
    return Mesh(points=pts + c, tris=_BOX_TRIS, color=color, collide="none")


# --- 0. Fixed wooden pickup block (200 x 130 x 32 mm) supporting the flat folder leaves ---
# Narrow 130 mm Y-width (y in [-0.065, +0.065]) leaves |y| in [0.065, 0.164] m overhanging 32 mm above TOP.
RISER_H = 0.0320
RISER_MESHES = [
    _box_mesh((0.0, 0.0, 0.0018), (0.1996, 0.1296, 0.0032), WOOD_BASE),
    _box_mesh((0.0, -0.0455, 0.5 * RISER_H + 0.0002), (0.1996, 0.0360, RISER_H - 0.0002), WOOD_LIGHT),
    _box_mesh((0.0, 0.0000, 0.5 * RISER_H + 0.0002), (0.1996, 0.0360, RISER_H - 0.0002), WOOD_DARK),
    _box_mesh((0.0, +0.0455, 0.5 * RISER_H + 0.0002), (0.1996, 0.0360, RISER_H - 0.0002), WOOD_LIGHT),
    # Reset alignment corner L-marks on top of the oak block:
    _box_mesh((-0.0940, -0.0560, RISER_H + 0.0001), (0.0030, 0.0140, 0.0002), MARK_DARK),
    _box_mesh((-0.0885, -0.0615, RISER_H + 0.0001), (0.0140, 0.0030, 0.0002), MARK_DARK),
    _box_mesh((-0.0940, +0.0560, RISER_H + 0.0001), (0.0030, 0.0140, 0.0002), MARK_DARK),
    _box_mesh((-0.0885, +0.0615, RISER_H + 0.0001), (0.0140, 0.0030, 0.0002), MARK_DARK),
]

INBOX_RISER = part(
    "inbox_riser",
    pos=(0.1750, 0.0, TOP),
    fixed=True,
    friction=0.55,
    boxes=[((0.0, 0.0, 0.5 * RISER_H), (0.2000, 0.1300, RISER_H), WOOD_OAK)],
    meshes=RISER_MESHES,
)

# --- 1. Fixed Officemate 91990 letter-size steel hanging file frame (14 in = 355.6 mm D) ---
RAIL_X_MIN = 0.4250
RAIL_X_MAX = 0.7806
RAIL_LEN = RAIL_X_MAX - RAIL_X_MIN  # 0.3556 m
RAIL_X_MID = 0.5 * (RAIL_X_MIN + RAIL_X_MAX)
RAIL_Y = 0.1588                      # 317.6 mm center-to-center rail span
RAIL_Z_TOP = TOP + 0.2380            # 0.9854 m
FRAME_BASE_H = 0.0040
FRAME_BASE_TOP = TOP + FRAME_BASE_H

FRAME_BOXES = [
    # Parallel horizontal top steel blade rails (3.6 mm wide x 8.0 mm tall, z in [0.2300, 0.2380]):
    ((RAIL_X_MID, +RAIL_Y, 0.2340), (RAIL_LEN, 0.0036, 0.0080), STEEL_RAIL),
    ((RAIL_X_MID, -RAIL_Y, 0.2340), (RAIL_LEN, 0.0036, 0.0080), STEEL_RAIL),
    # Four corner vertical steel posts (6 x 6 mm, z in [0.0040, 0.2300]):
    ((RAIL_X_MIN + 0.0030, +RAIL_Y, 0.1170), (0.0060, 0.0060, 0.2260), STEEL_DARK),
    ((RAIL_X_MIN + 0.0030, -RAIL_Y, 0.1170), (0.0060, 0.0060, 0.2260), STEEL_DARK),
    ((RAIL_X_MAX - 0.0030, +RAIL_Y, 0.1170), (0.0060, 0.0060, 0.2260), STEEL_DARK),
    ((RAIL_X_MAX - 0.0030, -RAIL_Y, 0.1170), (0.0060, 0.0060, 0.2260), STEEL_DARK),
    # Side base runners + low 4 mm front/rear U-saddle crossbars (open 311.6 x 226 mm front throat):
    ((RAIL_X_MID, +RAIL_Y, 0.0020), (RAIL_LEN, 0.0080, FRAME_BASE_H), STEEL_DARK),
    ((RAIL_X_MID, -RAIL_Y, 0.0020), (RAIL_LEN, 0.0080, FRAME_BASE_H), STEEL_DARK),
    ((RAIL_X_MIN + 0.0040, 0.0, 0.0020), (0.0080, 2.0 * RAIL_Y + 0.0080, FRAME_BASE_H), STEEL_DARK),
    ((RAIL_X_MAX - 0.0040, 0.0, 0.0020), (0.0080, 2.0 * RAIL_Y + 0.0080, FRAME_BASE_H), STEEL_DARK),
]

FILE_FRAME = part(
    "file_frame",
    pos=(0.0, 0.0, TOP),
    fixed=True,
    friction=0.45,
    boxes=FRAME_BOXES,
)

# --- 2. Loose V-hinged Pendaflex 4152 letter-size hanging file folder (0.16 kg total) ---
LEAF_W = 0.2985
ROD_SPAN = 0.3280
ROD_SEAT_X = 0.2300
GUSSET_Z = 0.0092


def _rod_and_hook_boxes(z_center: float):
    """328 mm coated-steel hanger rod + underside rail-hook saddle lugs centered at y = +-0.1588 m."""
    boxes = [
        # Main hanger rod spanning y in [-0.1640, +0.1640], x_local in [0.2300, 0.2350]:
        ((0.2325, 0.0, z_center), (0.0050, ROD_SPAN, 0.0030), ROD_WHITE),
    ]
    for sgn in (+1.0, -1.0):
        # Inner lug |y| in [0.1510, 0.1552] and outer hook tip |y| in [0.1624, 0.1660] -> 7.2 mm saddle around 3.6 mm rail:
        boxes.append(((0.22775, sgn * 0.1531, z_center), (0.0045, 0.0042, 0.0030), ROD_WHITE))
        boxes.append(((0.22775, sgn * 0.1642, z_center), (0.0045, 0.0036, 0.0030), ROD_WHITE))
    return boxes


FRONT_LEAF_BOXES = [
    # Lower/front pressboard leaf (x in [0.0000, 0.2300], z in [0.0000, 0.0025]):
    ((0.1150, 0.0, 0.00125), (0.2300, LEAF_W, 0.0025), FOLDER_GREEN),
    # 9.2 mm bottom V-gusset spine wall at x in [0.0000, 0.0025], z in [0.0025, 0.0092]:
    ((0.00125, 0.0, 0.00585), (0.0025, LEAF_W, 0.0067), FOLDER_GREEN),
] + _rod_and_hook_boxes(0.00125)

FRONT_LEAF_MESHES = [
    # Lighter two-tone anti-misfile kraft-sage interior lining + polylaminate top/gusset reinforcement strips:
    _box_mesh((0.1160, 0.0, 0.00255), (0.2140, LEAF_W - 0.0020, 0.0002), FOLDER_INNER),
    _box_mesh((0.2240, 0.0, -0.00005), (0.0110, LEAF_W - 0.0010, 0.0002), FOLDER_REINF),
    _box_mesh((0.0080, 0.0, -0.00005), (0.0150, LEAF_W - 0.0010, 0.0002), FOLDER_REINF),
    # Exterior subject label on front leaf (visible when hung upright facing -X toward top_cam):
    _box_mesh((0.1620, +0.0520, -0.00010), (0.0360, 0.0880, 0.0002), LABEL_WHITE),
    _box_mesh((0.1745, +0.0520, -0.00018), (0.0055, 0.0820, 0.0002), INK_HEADER),
    _box_mesh((0.1635, +0.0460, -0.00018), (0.0022, 0.0680, 0.0002), INK_LINE),
    _box_mesh((0.1555, +0.0490, -0.00018), (0.0022, 0.0740, 0.0002), INK_LINE),
]

# Rear/upper leaf is hinged at anchor (0.0, 0.0, GUSSET_Z); its boxes are specified in FOLDER part frame:
# Includes a clip-on 1/5-cut plastic index tab at y in [+0.0180, +0.0700] m (52 mm clear of GRASP_Y = 0.1220 m).
REAR_LEAF_BOXES_JOINT = [
    ((0.1150, 0.0, GUSSET_Z + 0.00125), (0.2300, LEAF_W, 0.0025), FOLDER_GREEN),
    ((0.24125, +0.0440, GUSSET_Z + 0.00125), (0.0125, 0.0520, 0.0018), TAB_BLUE),
] + _rod_and_hook_boxes(GUSSET_Z + 0.00125)

REAR_LEAF_MESHES_JOINT = [
    # Lighter kraft-sage interior lining + polylaminate reinforcement & scored V-gusset fold lines:
    _box_mesh((0.1160, 0.0, GUSSET_Z - 0.00005), (0.2140, LEAF_W - 0.0020, 0.0002), FOLDER_INNER),
    _box_mesh((0.2240, 0.0, GUSSET_Z + 0.00255), (0.0110, LEAF_W - 0.0010, 0.0002), FOLDER_REINF),
    _box_mesh((0.0080, 0.0, GUSSET_Z + 0.00255), (0.0150, LEAF_W - 0.0010, 0.0002), FOLDER_REINF),
    _box_mesh((0.0045, 0.0, GUSSET_Z + 0.00265), (0.0012, LEAF_W - 0.0020, 0.0002), MARK_DARK),
    _box_mesh((0.0105, 0.0, GUSSET_Z + 0.00265), (0.0012, LEAF_W - 0.0020, 0.0002), MARK_DARK),
    # White paper insert inside the 1/5-cut index tab:
    _box_mesh((0.2415, +0.0440, GUSSET_Z + 0.00125), (0.0082, 0.0440, 0.0020), LABEL_WHITE),
    _box_mesh((0.2415, +0.0420, GUSSET_Z + 0.00125), (0.0022, 0.0340, 0.0022), INK_HEADER),
    # Ruled exterior file-index label card on upper leaf (visible in setup.png when flat on pickup block):
    _box_mesh((0.1480, +0.0520, GUSSET_Z + 0.00260), (0.0400, 0.0900, 0.0002), LABEL_WHITE),
    _box_mesh((0.1620, +0.0520, GUSSET_Z + 0.00268), (0.0060, 0.0840, 0.0002), INK_HEADER),
    _box_mesh((0.1500, +0.0460, GUSSET_Z + 0.00268), (0.0022, 0.0700, 0.0002), INK_LINE),
    _box_mesh((0.1415, +0.0490, GUSSET_Z + 0.00268), (0.0022, 0.0760, 0.0002), INK_LINE),
    _box_mesh((0.1330, +0.0420, GUSSET_Z + 0.00268), (0.0022, 0.0620, 0.0002), INK_LINE),
]

# Equivalent boxes wrt folder origin when rear_leaf = 0 (used by geometric scorer):
REAR_LEAF_BOXES = REAR_LEAF_BOXES_JOINT

REAR_LEAF = Joint(
    "rear_leaf",
    kind="hinge",
    axis="Y",
    anchor=(0.0, 0.0, GUSSET_Z),
    lower=-1.25,
    upper=0.04,
    stiffness=0.0,
    damping=0.02,
    target=0.0,
    initial=0.0,
    mass=0.08,
    boxes=REAR_LEAF_BOXES_JOINT,
    meshes=REAR_LEAF_MESHES_JOINT,
)

FOLDER_X0 = 0.0720
FOLDER_Z0 = TOP + RISER_H + 0.0005

FOLDER = part(
    "folder",
    pos=(FOLDER_X0, 0.0, FOLDER_Z0),
    fixed=False,
    mass=0.08,
    friction=0.58,
    self_collide=False,
    boxes=FRONT_LEAF_BOXES,
    meshes=FRONT_LEAF_MESHES,
    joints=[REAR_LEAF],
)

# --- 3. Loose ~60-sheet printed document packet in a Smead 10330 1/3-cut manila cover (0.20 kg) ---
# Core 280 x 195 x 6.0 mm stack sits inside the V-folder, while a 62 x 180 x 4.0 mm printed top-sheet
# header and 12 x 88 x 2.0 mm 1/3-cut manila tab protrude cleanly between the two hanger rods (y in [-0.090, +0.090]).
PKT_H = 0.1950
PKT_W = 0.2800
PKT_T = 0.0060

PACKET_BOXES = [
    ((0.0, 0.0, 0.0), (PKT_H, PKT_W, PKT_T), PAPER_WHITE),
    ((+0.1285, 0.0, 0.0), (0.0620, 0.1800, 0.0040), PAPER_WHITE),
    ((+0.1655, -0.0460, -0.0010), (0.0120, 0.0880, 0.0020), MANILA),
]

PACKET_MESHES = [
    # Manila outer folder cover backing + side margins framing the white 60-sheet paper stack:
    _box_mesh((0.0, 0.0, -0.00285), (PKT_H - 0.0004, PKT_W - 0.0004, 0.0004), MANILA),
    _box_mesh((+0.1285, 0.0, -0.00190), (0.0618, 0.1798, 0.0004), MANILA),
    _box_mesh((+0.1285, -0.0880, 0.0), (0.0618, 0.0040, 0.0041), MANILA),
    _box_mesh((+0.1285, +0.0880, 0.0), (0.0618, 0.0040, 0.0041), MANILA),
    # Printed label on the 1/3-cut manila tab:
    _box_mesh((+0.1655, -0.0460, +0.00010), (0.0075, 0.0720, 0.0003), LABEL_WHITE),
    _box_mesh((+0.1655, -0.0480, +0.00022), (0.0022, 0.0560, 0.0003), INK_HEADER),
    # Stepped white/cream page-edge layers & side sheet striations (60-sheet stack cues):
    _box_mesh((+0.1540, 0.0, +0.0014), (0.0100, 0.1700, 0.0012), PAPER_CREAM),
    _box_mesh((+0.1570, +0.0020, +0.0004), (0.0050, 0.1680, 0.0010), PAPER_EDGE),
    _box_mesh((0.0, 0.0, -0.0012), (PKT_H + 0.0002, PKT_W + 0.0002, 0.0008), PAPER_EDGE),
    _box_mesh((0.0, 0.0, +0.0008), (PKT_H + 0.0002, PKT_W + 0.0002, 0.0008), PAPER_EDGE),
    # Printed report header banner + text lines on +Z face (faces top_cam both at t=0 and when hung!):
    _box_mesh((+0.1475, -0.0040, +0.00208), (0.0065, 0.1480, 0.0002), INK_HEADER),
    _box_mesh((+0.1385, -0.0220, +0.00208), (0.0026, 0.1120, 0.0002), INK_LINE),
    _box_mesh((+0.1325, -0.0080, +0.00208), (0.0022, 0.1380, 0.0002), INK_LINE),
    _box_mesh((+0.1270, -0.0120, +0.00208), (0.0022, 0.1300, 0.0002), INK_LINE),
    _box_mesh((+0.1215, -0.0050, +0.00208), (0.0022, 0.1440, 0.0002), INK_LINE),
    _box_mesh((+0.1160, -0.0280, +0.00208), (0.0022, 0.0980, 0.0002), INK_LINE),
]

PACKET = part(
    "packet",
    pos=(FOLDER_X0 + 0.1050, 0.0, FOLDER_Z0 + 0.0028 + 0.5 * PKT_T),
    fixed=False,
    mass=0.20,
    friction=0.55,
    boxes=PACKET_BOXES,
    meshes=PACKET_MESHES,
)


# --- 4. Two loose pre-filed Pendaflex 4152 hanging folders seated on the rear half of FILE_FRAME ---
def _make_rear_filed_folder(name: str, x_slot: float, tab_y: float, tab_color, manila_y: float) -> labkit.Part:
    """Loose rail-supported Pendaflex 4152 folder with 4 real hook saddles, V-leaves, and visible filed documents."""
    boxes = [
        # Front & rear 328 mm coated-steel hanger rods (underside bearing at z_local = 0.0000 on RAIL_Z_TOP):
        ((-0.0100, 0.0, +0.0025), (0.0030, ROD_SPAN, 0.0050), ROD_WHITE),
        ((+0.0100, 0.0, +0.0025), (0.0030, ROD_SPAN, 0.0050), ROD_WHITE),
        # V-angled front & rear pressboard leaves meeting at a 9.2 mm bottom crease 8 mm above the desk:
        ((-0.0064, 0.0, -0.1140), (0.0025, LEAF_W, 0.2280), FOLDER_GREEN, (0.0, -0.025)),
        ((+0.0064, 0.0, -0.1140), (0.0025, LEAF_W, 0.2280), FOLDER_GREEN, (0.0, +0.025)),
        ((0.0, 0.0, -0.2288), (0.0092, LEAF_W, 0.0025), FOLDER_GREEN),
        # Filed white letter pages + Smead 1/3-cut manila tab peeking out the V-mouth:
        ((0.0, 0.0, -0.0980), (0.0055, 0.2780, 0.2200), PAPER_WHITE),
        ((+0.0018, manila_y, +0.0155), (0.0020, 0.0860, 0.0090), MANILA),
        # Clip-on 1/5-cut plastic index tab on the rear rod:
        ((+0.0100, tab_y, +0.01125), (0.0018, 0.0520, 0.0125), tab_color),
    ]
    for x_rod in (-0.0100, +0.0100):
        for sgn in (+1.0, -1.0):
            boxes.append(((x_rod, sgn * 0.1531, -0.00225), (0.0030, 0.0042, 0.0045), ROD_WHITE))
            boxes.append(((x_rod, sgn * 0.1642, -0.00225), (0.0030, 0.0036, 0.0045), ROD_WHITE))
    meshes = [
        # Two-tone kraft-sage interior lining visible down the 20 mm open V-mouth:
        _box_mesh((-0.0050, 0.0, -0.1140), (0.0002, LEAF_W - 0.0020, 0.2260), FOLDER_INNER, pitch=-0.025),
        _box_mesh((+0.0050, 0.0, -0.1140), (0.0002, LEAF_W - 0.0020, 0.2260), FOLDER_INNER, pitch=+0.025),
        # White paper insert inside the 1/5-cut index tab + printed header lines on the exposed document top:
        _box_mesh((+0.0100, tab_y, +0.01125), (0.0020, 0.0440, 0.0082), LABEL_WHITE),
        _box_mesh((+0.0100, tab_y - 0.0020, +0.01125), (0.0022, 0.0340, 0.0022), INK_HEADER),
        _box_mesh((-0.0029, -0.0100, +0.0085), (0.0003, 0.1400, 0.0035), INK_HEADER),
        _box_mesh((0.0, +0.0030, +0.0122), (0.0040, 0.2680, 0.0008), PAPER_EDGE),
    ]
    return part(
        name,
        pos=(x_slot, 0.0, RAIL_Z_TOP + 0.0003),
        fixed=False,
        mass=0.22,
        friction=0.52,
        boxes=boxes,
        meshes=meshes,
    )


REAR_FOLDER_1 = _make_rear_filed_folder("rear_folder_1", 0.6350, -0.0620, TAB_AMBER, +0.0460)
REAR_FOLDER_2 = _make_rear_filed_folder("rear_folder_2", 0.7100, +0.0920, TAB_CLEAR, -0.0460)
REAR_FOLDERS = (REAR_FOLDER_1, REAR_FOLDER_2)

OBJECTS = [INBOX_RISER, FILE_FRAME, FOLDER, PACKET, REAR_FOLDER_1, REAR_FOLDER_2]

from task_metrics import Metric, ry  # noqa: E402

_METRIC = Metric(globals())
_step = 0
_DEBUG = os.environ.get("FILE_DEBUG") == "1"
_PROBE = os.environ.get("FILE_PROBE", "")


@on_reset
def _reset(env):
    global _step
    _step = 0
    _METRIC.reset()


@on_step
def _update(env):
    global _step
    _step += 1
    fld_s = labkit._part_state(env, FOLDER)
    pkt_s = labkit._part_state(env, PACKET)
    rear_s = [labkit._part_state(env, rf) for rf in REAR_FOLDERS]
    _METRIC.update(env, fld_s, pkt_s, rear_s)
    if _DEBUG and (_step % 10 == 0 or _step <= 5):
        from yam_arena import runtime as _rt

        b = env.unwrapped
        orig = b.scene.env_origins[0].cpu().numpy()
        g_l = _rt.link_poses(b.scene["left_robot"])["gripper"]
        g_r = _rt.link_poses(b.scene["right_robot"])["gripper"]
        tip_l = _rt.pad_tips_world(b.scene["left_robot"]) - orig
        tip_r = _rt.pad_tips_world(b.scene["right_robot"]) - orig
        q_l = np.asarray(fld_s.to_world((0.2325, +GRASP_Y, 0.00585)), float)
        q_r = np.asarray(fld_s.to_world((0.2325, -GRASP_Y, 0.00585)), float)
        rod_l_mm = [round(float(v) * 1000.0, 2) for v in (g_l[:3, :3].T @ (q_l - tip_l))]
        rod_r_mm = [round(float(v) * 1000.0, 2) for v in (g_r[:3, :3].T @ (q_r - tip_r))]
        print(
            "FILEDBG "
            + json.dumps(
                dict(
                    step=_step,
                    t=round(_step / 20.0, 2),
                    fld_pos=[round(float(v), 4) for v in fld_s.pos],
                    pkt_pos=[round(float(v), 4) for v in pkt_s.pos],
                    rf1_pos=[round(float(v), 4) for v in rear_s[0].pos],
                    rf2_pos=[round(float(v), 4) for v in rear_s[1].pos],
                    rod_l_mm=rod_l_mm,
                    rod_r_mm=rod_r_mm,
                    **_METRIC.detail,
                )
            ),
            flush=True,
        )


def score(objects) -> float:
    return _METRIC.score()


def success(objects) -> bool:
    return score(objects) == 1.0


class Orient(float):
    def __new__(cls, r):
        v = float.__new__(cls, 0.0)
        v.rot = np.asarray(r, float)
        return v


_old_grip = dm.grip_rotation
dm.grip_rotation = lambda yaw=0.0: yaw.rot if hasattr(yaw, "rot") else _old_grip(yaw)


def make_frame(gx, gz) -> np.ndarray:
    gx = np.asarray(gx, float)
    gz = np.asarray(gz, float)
    gz = gz / np.linalg.norm(gz)
    gx = gx - np.dot(gx, gz) * gz
    gx = gx / np.linalg.norm(gx)
    gy = np.cross(gz, gx)
    return np.column_stack([gx, gy, gz])


def wp(dur: float, pos, finger: float, label: str, rot: np.ndarray) -> Waypoint:
    return Waypoint(dur, tuple(float(v) for v in pos), float(finger), label, Orient(rot))


Q_L_PARK = np.array(READY, float) + np.array([0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
P_L_PARK, ROT_L_PARK = ik.tcp_pose(Q_L_PARK, np.array(LEFT_BASE_POS, float))
Q_R_PARK = np.array(READY, float) + np.array([-0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
P_R_PARK, ROT_R_PARK = ik.tcp_pose(Q_R_PARK, np.array(RIGHT_BASE_POS, float))

# Exact rigid-body co-rotation from flat (theta=0) -> mid (-45 deg) -> vertical hanging (-90 deg):
# ROT_MID = ry(-pi/4) @ ROT_FLAT and ROT_HANG = ry(-pi/2) @ ROT_FLAT keep 0.00 deg relative pad-to-rod twist!
GRASP_Y = 0.1220
GRASP_X_LOCAL = 0.2295
PAD_OFFSET = 0.0040
ROT_FLAT_L = make_frame([0.0, 0.0, +1.0], [-0.35, -0.86, -0.36])
ROT_FLAT_R = make_frame([0.0, 0.0, -1.0], [-0.35, +0.86, -0.36])

ROT_MID_L = ry(-0.25 * math.pi) @ ROT_FLAT_L
ROT_MID_R = ry(-0.25 * math.pi) @ ROT_FLAT_R

R_HANG = ry(-0.5 * math.pi)
ROT_HANG_L = R_HANG @ ROT_FLAT_L
ROT_HANG_R = R_HANG @ ROT_FLAT_R

P_ROD_FLAT_L = np.array([FOLDER_X0 + GRASP_X_LOCAL, +GRASP_Y, FOLDER_Z0 + 0.00585]) - PAD_OFFSET * ROT_FLAT_L[:, 2]
P_ROD_FLAT_R = np.array([FOLDER_X0 + GRASP_X_LOCAL, -GRASP_Y, FOLDER_Z0 + 0.00585]) - PAD_OFFSET * ROT_FLAT_R[:, 2]
P_PRE_FLAT_L = P_ROD_FLAT_L - 0.052 * ROT_FLAT_L[:, 2]
P_PRE_FLAT_R = P_ROD_FLAT_R - 0.052 * ROT_FLAT_R[:, 2]
P_VLIFT_L = P_ROD_FLAT_L + np.array([0.0, 0.0, +0.0650])
P_VLIFT_R = P_ROD_FLAT_R + np.array([0.0, 0.0, +0.0650])

# Lift straight +Z first, pitch -45 deg high, then finish -90 deg pitch at x = 0.380 m ABOVE rail height BEFORE crossing RAIL_X_MIN = 0.425 m:
X_HANG = 0.4650
Z_ABOVE_RAIL = RAIL_Z_TOP + 0.0400
Z_SEAT_RAIL = RAIL_Z_TOP + 0.0140

P_MID_L = np.array([0.3350, +GRASP_Y, TOP + 0.2650]) - PAD_OFFSET * ROT_MID_L[:, 2]
P_MID_R = np.array([0.3350, -GRASP_Y, TOP + 0.2650]) - PAD_OFFSET * ROT_MID_R[:, 2]
P_PRE_ENTRY_L = np.array([0.3800, +GRASP_Y, Z_ABOVE_RAIL]) - PAD_OFFSET * ROT_HANG_L[:, 2]
P_PRE_ENTRY_R = np.array([0.3800, -GRASP_Y, Z_ABOVE_RAIL]) - PAD_OFFSET * ROT_HANG_R[:, 2]
P_ABOVE_L = np.array([X_HANG, +GRASP_Y, Z_ABOVE_RAIL]) - PAD_OFFSET * ROT_HANG_L[:, 2]
P_ABOVE_R = np.array([X_HANG, -GRASP_Y, Z_ABOVE_RAIL]) - PAD_OFFSET * ROT_HANG_R[:, 2]
P_SEAT_L = np.array([X_HANG, +GRASP_Y, Z_SEAT_RAIL]) - PAD_OFFSET * ROT_HANG_L[:, 2]
P_SEAT_R = np.array([X_HANG, -GRASP_Y, Z_SEAT_RAIL]) - PAD_OFFSET * ROT_HANG_R[:, 2]
P_RETRACT_L = P_SEAT_L - 0.085 * ROT_HANG_L[:, 2] + np.array([-0.025, 0.0, +0.040])
P_RETRACT_R = P_SEAT_R - 0.085 * ROT_HANG_R[:, 2] + np.array([-0.025, 0.0, +0.040])

F_OPEN_RODS = -0.024
F_RELEASE_SEAT = -0.015
F_CLAMP_RODS = -0.0020


def demo():
    if _PROBE == "one_hand_only":
        # Negative control 1: single-arm grasp on left side only while right arm stays parked:
        left_only = [
            wp(1.5, P_PRE_FLAT_L, F_OPEN_RODS, "stage at left rod tips", ROT_FLAT_L),
            wp(1.0, P_ROD_FLAT_L, F_OPEN_RODS, "align around both left rods", ROT_FLAT_L),
            wp(1.2, P_ROD_FLAT_L, F_CLAMP_RODS, "clamp front+rear rods shut", ROT_FLAT_L),
            wp(1.0, P_VLIFT_L, F_CLAMP_RODS, "lift straight up off riser", ROT_FLAT_L),
            wp(1.4, P_MID_L, F_CLAMP_RODS, "lift and pitch folder 45 deg", ROT_MID_L),
            wp(1.3, P_PRE_ENTRY_L, F_CLAMP_RODS, "pitch 90 deg above rail height", ROT_HANG_L),
            wp(1.3, P_ABOVE_L, F_CLAMP_RODS, "enter frame above top rails", ROT_HANG_L),
            wp(1.3, P_SEAT_L, F_CLAMP_RODS, "seat left rod hooks onto rail", ROT_HANG_L),
            wp(0.8, P_SEAT_L, F_RELEASE_SEAT, "release left rods", ROT_HANG_L),
            wp(1.2, P_RETRACT_L, FINGER_OPEN, "retract clear of rail", ROT_HANG_L),
            wp(2.6, P_L_PARK, FINGER_OPEN, "park left", ROT_L_PARK),
            wp(2.2, P_L_PARK, FINGER_OPEN, "hold still", ROT_L_PARK),
        ]
        right_parked = [wp(16.8, P_R_PARK, FINGER_OPEN, "stay parked", ROT_R_PARK)]
        return {"left": left_only, "right": right_parked}
    if _PROBE == "unclamped_lift":
        # Negative control 2: bimanual scoop/lift with 44 mm open jaw gap (-0.022) that never clamps the rods/V-pocket:
        f_loose = -0.022
        left_u = [
            wp(1.5, P_PRE_FLAT_L, F_OPEN_RODS, "stage at left rod tips", ROT_FLAT_L),
            wp(1.0, P_ROD_FLAT_L, F_OPEN_RODS, "align around both left rods", ROT_FLAT_L),
            wp(1.2, P_ROD_FLAT_L, f_loose, "leave V-pocket unclamped", ROT_FLAT_L),
            wp(1.0, P_VLIFT_L, f_loose, "lift straight up unclamped", ROT_FLAT_L),
            wp(1.4, P_MID_L, f_loose, "lift unclamped 45 deg", ROT_MID_L),
            wp(1.3, P_PRE_ENTRY_L, f_loose, "pitch 90 deg unclamped", ROT_HANG_L),
            wp(1.3, P_ABOVE_L, f_loose, "enter frame unclamped", ROT_HANG_L),
            wp(1.3, P_SEAT_L, f_loose, "seat unclamped", ROT_HANG_L),
            wp(0.8, P_SEAT_L, F_RELEASE_SEAT, "release", ROT_HANG_L),
            wp(1.2, P_RETRACT_L, FINGER_OPEN, "retract", ROT_HANG_L),
            wp(2.6, P_L_PARK, FINGER_OPEN, "park left", ROT_L_PARK),
            wp(2.2, P_L_PARK, FINGER_OPEN, "hold still", ROT_L_PARK),
        ]
        right_u = [
            wp(1.5, P_PRE_FLAT_R, F_OPEN_RODS, "stage at right rod tips", ROT_FLAT_R),
            wp(1.0, P_ROD_FLAT_R, F_OPEN_RODS, "align around both right rods", ROT_FLAT_R),
            wp(1.2, P_ROD_FLAT_R, f_loose, "leave V-pocket unclamped", ROT_FLAT_R),
            wp(1.0, P_VLIFT_R, f_loose, "lift straight up unclamped", ROT_FLAT_R),
            wp(1.4, P_MID_R, f_loose, "lift unclamped 45 deg", ROT_MID_R),
            wp(1.3, P_PRE_ENTRY_R, f_loose, "pitch 90 deg unclamped", ROT_HANG_R),
            wp(1.3, P_ABOVE_R, f_loose, "enter frame unclamped", ROT_HANG_R),
            wp(1.3, P_SEAT_R, f_loose, "seat unclamped", ROT_HANG_R),
            wp(0.8, P_SEAT_R, F_RELEASE_SEAT, "release", ROT_HANG_R),
            wp(1.2, P_RETRACT_R, FINGER_OPEN, "retract", ROT_HANG_R),
            wp(2.6, P_R_PARK, FINGER_OPEN, "park right", ROT_R_PARK),
            wp(2.2, P_R_PARK, FINGER_OPEN, "hold still", ROT_R_PARK),
        ]
        return {"left": left_u, "right": right_u}
    left = [
        wp(1.5, P_PRE_FLAT_L, F_OPEN_RODS, "stage at left rod tips", ROT_FLAT_L),
        wp(1.0, P_ROD_FLAT_L, F_OPEN_RODS, "align around both left rods", ROT_FLAT_L),
        wp(1.2, P_ROD_FLAT_L, F_CLAMP_RODS, "clamp front+rear rods shut", ROT_FLAT_L),
        wp(1.0, P_VLIFT_L, F_CLAMP_RODS, "lift straight up off riser", ROT_FLAT_L),
        wp(1.4, P_MID_L, F_CLAMP_RODS, "lift and pitch folder 45 deg", ROT_MID_L),
        wp(1.3, P_PRE_ENTRY_L, F_CLAMP_RODS, "pitch 90 deg above rail height", ROT_HANG_L),
        wp(1.3, P_ABOVE_L, F_CLAMP_RODS, "enter frame above top rails", ROT_HANG_L),
        wp(1.3, P_SEAT_L, F_CLAMP_RODS, "seat left rod hooks onto rail", ROT_HANG_L),
        wp(0.8, P_SEAT_L, F_RELEASE_SEAT, "release left rods", ROT_HANG_L),
        wp(1.2, P_RETRACT_L, FINGER_OPEN, "retract clear of rail", ROT_HANG_L),
        wp(2.6, P_L_PARK, FINGER_OPEN, "park left", ROT_L_PARK),
        wp(2.2, P_L_PARK, FINGER_OPEN, "hold still", ROT_L_PARK),
    ]
    right = [
        wp(1.5, P_PRE_FLAT_R, F_OPEN_RODS, "stage at right rod tips", ROT_FLAT_R),
        wp(1.0, P_ROD_FLAT_R, F_OPEN_RODS, "align around both right rods", ROT_FLAT_R),
        wp(1.2, P_ROD_FLAT_R, F_CLAMP_RODS, "clamp front+rear rods shut", ROT_FLAT_R),
        wp(1.0, P_VLIFT_R, F_CLAMP_RODS, "lift straight up off riser", ROT_FLAT_R),
        wp(1.4, P_MID_R, F_CLAMP_RODS, "lift and pitch folder 45 deg", ROT_MID_R),
        wp(1.3, P_PRE_ENTRY_R, F_CLAMP_RODS, "pitch 90 deg above rail height", ROT_HANG_R),
        wp(1.3, P_ABOVE_R, F_CLAMP_RODS, "enter frame above top rails", ROT_HANG_R),
        wp(1.3, P_SEAT_R, F_CLAMP_RODS, "seat right rod hooks onto rail", ROT_HANG_R),
        wp(0.8, P_SEAT_R, F_RELEASE_SEAT, "release right rods", ROT_HANG_R),
        wp(1.2, P_RETRACT_R, FINGER_OPEN, "retract clear of rail", ROT_HANG_R),
        wp(2.6, P_R_PARK, FINGER_OPEN, "park right", ROT_R_PARK),
        wp(2.2, P_R_PARK, FINGER_OPEN, "hold still", ROT_R_PARK),
    ]
    return {"left": left, "right": right}
