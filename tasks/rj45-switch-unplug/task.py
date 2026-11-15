"""Unplug a spring-latched IEC 60603-7 RJ45 connector from a loose Gigabit cable modem.

Exterior housing dimensions follow the Motorola MB8600 DOCSIS 3.1 4-port Gigabit cable
modem (57.0 mm W x 184.0 mm D x 200.0 mm H, 0.523 kg) with its vertical rear stack of
four Gigabit RJ45 LAN ports below the coaxial F-barrel. Plug/boot outer dimensions follow
IEC 60603-7 (11.68 x 6.60 mm mating tongue) and Stewart Connector SS-39200-085 (36.07 mm
plug-and-boot length) plus a 14 x 6 mm strain-relief collar. Internal latch stiffness,
catch overlap, exact port heights, and omitted flexible cable tail are estimates.
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
from labkit import Joint, Mesh, on_reset, on_step, part
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 900
SUCCESS_HOLD_S = 1.0
labkit.CONTACT_OFFSET = 0.0002

CHARCOAL = (0.16, 0.18, 0.21)
SLATE = (0.24, 0.27, 0.31)
SILVER = (0.74, 0.77, 0.80)
BRASS = (0.76, 0.68, 0.36)
AMBER_PC = (0.86, 0.82, 0.68)
CABLE_BLUE = (0.16, 0.44, 0.78)
LED_GREEN = (0.18, 0.78, 0.32)
YELLOW_LAN = (0.86, 0.72, 0.18)


def visual_box(center, size, color):
    c, h = np.asarray(center, float), np.asarray(size, float) / 2.0
    pts = np.array([c + h * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = np.array([
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
    ])
    return Mesh(pts, tris, color, "none")


# --- 1. Motorola MB8600 4-port Gigabit Cable Modem envelope (57 x 184 x 200 mm, 0.523 kg) ---
# Centered at (0.280, +0.0420, TOP); rear port bezel is at local y = -0.0920 m (world y = -0.0500 m).
SW_X, SW_Y = 0.280, 0.0420
BEZEL_LOCAL_Y = -0.0920  # world y = -0.0500 m
PORT_Z = 0.1180          # LAN 1 RJ45 port centerline 118.0 mm above counter

# LAN 1 cavity: x in [-0.00625, +0.00625] (12.50 mm wide around 11.68 mm plug),
# y in [-0.0920, -0.0785] (13.5 mm deep), floor z = 0.11435 m, pocket roof z = 0.12450 m,
# rear-panel upper locking lip y in [-0.0920, -0.0902] (1.8 mm thick) with underside at z = 0.12180 m (+3.80 mm).
SWITCH_BOXES = [
    # Snap-on stability base foot (74 x 184 x 7 mm)
    ((0.0, 0.0, 0.0035), (0.0740, 0.1840, 0.0070), CHARCOAL),
    # Lower body below LAN 1 port floor (z = 0.0070 .. 0.11435 m): single uninterrupted port floor face
    ((0.0, 0.0, 0.060675), (0.0570, 0.1840, 0.10735), SLATE),
    # Main body forward (+Y) of LAN 1 back stop (y = -0.0785 .. +0.0920 m, z = 0.11435 .. 0.2000 m)
    ((0.0, +0.00675, 0.157175), (0.0570, 0.1705, 0.08565), SLATE),
    # Left and right cheeks flanking the LAN 1 cavity (x = +- [0.00625 .. 0.0285] m, y = -0.0920 .. -0.0785 m)
    ((-0.017375, -0.08525, 0.157175), (0.02225, 0.0135, 0.08565), SLATE),
    ((+0.017375, -0.08525, 0.157175), (0.02225, 0.0135, 0.08565), SLATE),
    # Upper spine above LAN 1 latch pocket (z = 0.12450 .. 0.2000 m)
    ((0.0, -0.08525, 0.16225), (0.0125, 0.0135, 0.07550), SLATE),
    # LAN 1 upper bezel locking lip (y = -0.0920 .. -0.0902 m, z = 0.12180 .. 0.12450 m)
    ((0.0, -0.0911, 0.12315), (0.0125, 0.0018, 0.0027), SILVER),
]

SWITCH_VISUALS = [
    # Coaxial F-connector visual trim above LAN 1 (z = 0.156 m)
    visual_box((0.0, -0.0945, 0.1560), (0.0095, 0.0050, 0.0095), BRASS),
    # LAN 1 metallic shield + link LED
    visual_box((0.0, -0.0921, 0.1180), (0.0160, 0.0004, 0.0135), SILVER),
    visual_box((-0.0055, -0.0922, 0.1265), (0.0025, 0.0004, 0.0018), LED_GREEN),
    # LAN 2, LAN 3, LAN 4 Gigabit Ethernet port trims below LAN 1
    visual_box((0.0, -0.0921, 0.0980), (0.0155, 0.0004, 0.0130), YELLOW_LAN),
    visual_box((0.0, -0.0921, 0.0780), (0.0155, 0.0004, 0.0130), YELLOW_LAN),
    visual_box((0.0, -0.0921, 0.0580), (0.0155, 0.0004, 0.0130), YELLOW_LAN),
    # DC power jack trim near bottom
    visual_box((0.0, -0.0921, 0.0280), (0.0090, 0.0004, 0.0090), CHARCOAL),
]

SWITCH = part(
    "switch",
    pos=(SW_X, SW_Y, TOP),
    fixed=False,
    mass=0.523,
    friction=0.55,
    self_collide=False,
    boxes=SWITCH_BOXES,
    meshes=SWITCH_VISUALS,
)

# --- 2. IEC 60603-7 RJ45 plug + Stewart SS-39200-085 boot (36.07 mm) + 14 mm collar ---
# Seated origin at (0.280, -0.0545, TOP + 0.1180):
#   Nose at y_local = +0.0175 m -> world y = -0.0370 m (13.0 mm inside the -0.0500 m bezel).
#   Boot rear at y_local = -0.01857 m -> world y = -0.07307 m (23.07 mm outside bezel).
#   Collar rear at y_local = -0.03257 m -> world y = -0.08707 m.
PLUG_Y0 = -0.0545
TAB_JOINT = Joint(
    "tab",
    kind="hinge",
    axis="X",
    anchor=(0.0, +0.0175, +0.0033),
    lower=0.0,
    upper=0.170,
    initial=0.0,
    target=0.0,
    stiffness=0.12,
    damping=0.004,
    friction=0.0,
    mass=0.004,
    boxes=[
        ((0.0, +0.01065, +0.00375), (0.0074, 0.0123, 0.0009), AMBER_PC),
        ((0.0, +0.00760, +0.00415), (0.0074, 0.0018, 0.0016), AMBER_PC),
        ((0.0, -0.00200, +0.00710), (0.0102, 0.0130, 0.0014), CABLE_BLUE),
    ],
)

PLUG = part(
    "plug",
    pos=(SW_X, PLUG_Y0, TOP + PORT_Z),
    fixed=False,
    mass=0.022,
    friction=0.75,
    self_collide=False,
    boxes=[
        ((0.0, +0.01100, 0.0), (0.01168, 0.0130, 0.00660), AMBER_PC),
        ((0.0, -0.00200, -0.00060), (0.01168, 0.0130, 0.00540), AMBER_PC),
        ((0.0, -0.013535, -0.00010), (0.01220, 0.01007, 0.00640), CABLE_BLUE),
    ],
    cylinders=[
        ((0.0, -0.02557, 0.0), 0.0030, 0.0140, CABLE_BLUE, "Y"),
    ],
    joints=[TAB_JOINT],
)

OBJECTS = [SWITCH, PLUG]

from task_metrics import Metric  # noqa: E402

_METRIC = Metric(globals())
_step = 0
_DEBUG = os.environ.get("RJ45_DEBUG") == "1"
_PROBE = os.environ.get("RJ45_PROBE", "")


@on_reset
def _reset(env):
    global _step
    _step = 0
    _METRIC.reset()


@on_step
def _update(env):
    global _step
    _step += 1
    plug_art = env.unwrapped.scene["plug"]
    plug_art.set_joint_position_target(plug_art.data.default_joint_pos)

    sw_s = labkit._part_state(env, SWITCH)
    pl_s = labkit._part_state(env, PLUG)
    _METRIC.update(env, sw_s, pl_s)
    if _DEBUG and (_step % 10 == 0 or _step <= 5):
        print(
            "RJ45DBG "
            + json.dumps(
                dict(
                    step=_step,
                    t=round(_step / 20.0, 2),
                    sw_pos=[round(float(v), 4) for v in sw_s.pos],
                    pl_pos=[round(float(v), 4) for v in pl_s.pos],
                    **_METRIC.detail,
                )
            ),
            flush=True,
        )


def score(objects) -> float:
    return _METRIC.score()


def success(objects) -> bool:
    return score(objects) == 1.0


# --- Fixed open-loop two-arm trajectory (zero simulator object-pose reads) ---
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


# Left hand: top-down pitched 20 deg from +Y, pinching 57 mm modem width along X
ROT_L = make_frame([1.0, 0.0, 0.0], [0.0, -math.sin(math.radians(20.0)), -math.cos(math.radians(20.0))])

# Right hand extraction: +Y approach with slight 8 deg downward pitch (cos 8 = 0.990 axial), gx vertical pinch on top tab + belly
_A_PULL = math.radians(8.0)
ROT_R_PULL = make_frame([0.0, -math.sin(_A_PULL), -math.cos(_A_PULL)], [0.0, math.cos(_A_PULL), -math.sin(_A_PULL)])

# Right hand gentle counter set-down: roll 90 deg so jaws open horizontally +-X (zero scoop/hook)
# and pitch 38 deg downward so plug lays flat on its side 2 mm above counter before horizontal release.
ROT_R_PLACE = make_frame(
    [1.0, 0.0, 0.0],
    [0.0, math.cos(math.radians(38.0)), -math.sin(math.radians(38.0))],
)

# Right hand side-pinch negative control: gx = [1, 0, 0] (pinches left/right sides ONLY, never touches top tab)
ROT_R_SIDE = make_frame([1.0, 0.0, 0.0], [0.0, math.cos(math.radians(55.0)), -math.sin(math.radians(55.0))])


def wp(dur: float, pos, finger: float, label: str, rot: np.ndarray) -> Waypoint:
    return Waypoint(dur, tuple(float(v) for v in pos), float(finger), label, Orient(rot))


def demo():
    hl, rl = ik.tcp_pose(np.array(READY, float), np.array(LEFT_BASE_POS, float))
    hr, rr = ik.tcp_pose(np.array(READY, float), np.array(RIGHT_BASE_POS, float))

    # Left hand braces upper +Y half of 57 mm wide MB8600 housing (x=0.280, y=+0.085, z=TOP+0.180)
    p_brace = np.array([SW_X, 0.085, TOP + 0.180]) - 0.006 * ROT_L[:, 2]
    p_brace_pre = p_brace - 0.070 * ROT_L[:, 2]

    # Right hand pinch TCP at y = -0.0685 puts jaw front ~5.5 mm clear of -0.0500 bezel
    p_pinch = np.array([SW_X, -0.0685, TOP + PORT_Z + 0.0020])
    p_pinch_pre = np.array([SW_X, -0.1280, TOP + PORT_Z + 0.0020])
    p_pulled = np.array([SW_X, -0.1400, TOP + PORT_Z + 0.0020])

    p_place_hover = np.array([SW_X, -0.175, TOP + 0.080])
    p_place_down = np.array([SW_X, -0.175, TOP + 0.021])
    p_place_back = p_place_down - 0.085 * ROT_R_PLACE[:, 2]

    if _PROBE == "unpressed_unbraced":
        p_side = np.array([SW_X, -0.0760, TOP + PORT_Z]) - 0.006 * ROT_R_SIDE[:, 2]
        return {
            "left": [wp(14.0, hl, FINGER_OPEN, "park left", rl)],
            "right": [
                wp(2.5, p_side - 0.060 * ROT_R_SIDE[:, 2], -0.025, "approach rear boot sides", ROT_R_SIDE),
                wp(1.2, p_side, -0.025, "straddle boot sides", ROT_R_SIDE),
                wp(1.0, p_side, -0.002, "pinch sides without touching top tab", ROT_R_SIDE),
                wp(3.0, p_side + np.array([0.0, -0.065, 0.0]), -0.002, "pull -Y without depressing tab", ROT_R_SIDE),
                wp(1.0, p_side + np.array([0.0, -0.065, 0.0]), -0.025, "release", ROT_R_SIDE),
                wp(2.5, hr, FINGER_OPEN, "park right", rr),
                wp(2.8, hr, FINGER_OPEN, "settle", rr),
            ],
        }

    if _PROBE == "unpressed_braced":
        p_side = np.array([SW_X, -0.0740, TOP + PORT_Z]) - 0.006 * ROT_R_SIDE[:, 2]
        return {
            "left": [
                wp(2.2, p_brace_pre, -0.045, "approach modem", ROT_L),
                wp(1.2, p_brace, -0.045, "descend onto modem", ROT_L),
                wp(0.8, p_brace, -0.0235, "brace modem", ROT_L),
                wp(6.5, p_brace, -0.0235, "hold modem steady", ROT_L),
                wp(0.8, p_brace, -0.045, "release modem", ROT_L),
                wp(1.2, p_brace_pre, -0.045, "lift clear", ROT_L),
                wp(2.5, hl, FINGER_OPEN, "park left", rl),
            ],
            "right": [
                wp(3.5, p_side - 0.060 * ROT_R_SIDE[:, 2], -0.025, "wait for brace", ROT_R_SIDE),
                wp(1.2, p_side, -0.025, "straddle boot sides", ROT_R_SIDE),
                wp(1.0, p_side, -0.0015, "pinch sides only (tab unpressed)", ROT_R_SIDE),
                wp(3.0, p_side + np.array([0.0, -0.065, 0.0]), -0.0015, "pull -Y against locked barb", ROT_R_SIDE),
                wp(1.0, p_side + np.array([0.0, -0.065, 0.0]), -0.025, "release", ROT_R_SIDE),
                wp(2.5, hr, FINGER_OPEN, "park right", rr),
                wp(3.0, hr, FINGER_OPEN, "settle", rr),
            ],
        }

    if _PROBE == "early_release":
        p_side = np.array([SW_X, -0.0750, TOP + PORT_Z]) - 0.006 * ROT_R_SIDE[:, 2]
        return {
            "left": [
                wp(2.2, p_brace_pre, -0.045, "approach modem", ROT_L),
                wp(1.2, p_brace, -0.045, "descend onto modem", ROT_L),
                wp(0.8, p_brace, -0.0235, "brace modem", ROT_L),
                wp(11.0, p_brace, -0.0235, "hold modem steady", ROT_L),
                wp(0.8, p_brace, -0.045, "release modem", ROT_L),
                wp(1.2, p_brace_pre, -0.045, "lift clear", ROT_L),
                wp(2.5, hl, FINGER_OPEN, "park left", rl),
            ],
            "right": [
                wp(3.5, p_pinch_pre, -0.020, "align behind RJ45 plug", ROT_R_PULL),
                wp(1.5, p_pinch, -0.020, "advance around tab and belly", ROT_R_PULL),
                wp(1.0, p_pinch, -0.0018, "depress locking tab", ROT_R_PULL),
                wp(1.0, p_pinch + np.array([0.0, -0.0010, 0.0]), -0.0018, "pull 1 mm before lip", ROT_R_PULL),
                wp(1.0, p_pinch + np.array([0.0, -0.0010, 0.0]), -0.020, "release tab early", ROT_R_PULL),
                wp(1.5, p_pinch_pre, -0.025, "back off", ROT_R_PULL),
                wp(2.0, p_side, -0.025, "re-approach sides only", ROT_R_SIDE),
                wp(1.0, p_side, -0.0015, "pinch sides without tab", ROT_R_SIDE),
                wp(2.5, p_side + np.array([0.0, -0.060, 0.0]), -0.0015, "pull against re-engaged barb", ROT_R_SIDE),
                wp(1.0, p_side + np.array([0.0, -0.060, 0.0]), -0.025, "open", ROT_R_SIDE),
                wp(2.5, hr, FINGER_OPEN, "park right", rr),
                wp(2.0, hr, FINGER_OPEN, "settle", rr),
            ],
        }

    return {
        "left": [
            wp(2.2, p_brace_pre, -0.045, "approach modem top front", ROT_L),
            wp(1.2, p_brace, -0.045, "seat around 57 mm modem body", ROT_L),
            wp(0.8, p_brace, -0.0235, "counterhold loose modem", ROT_L),
            wp(6.3, p_brace, -0.0235, "hold modem steady during RJ45 withdrawal", ROT_L),
            wp(0.8, p_brace, -0.045, "release modem gently", ROT_L),
            wp(1.2, p_brace_pre, -0.045, "lift clear of modem", ROT_L),
            wp(2.5, hl, FINGER_OPEN, "park left arm", rl),
            wp(6.0, hl, FINGER_OPEN, "hold park", rl),
        ],
        "right": [
            wp(3.0, p_pinch_pre, -0.020, "align horizontally behind RJ45 plug", ROT_R_PULL),
            wp(1.5, p_pinch, -0.020, "advance pads over locking tab and belly", ROT_R_PULL),
            wp(1.2, p_pinch, -0.0018, "pinch to depress locking tab past bezel lip", ROT_R_PULL),
            wp(3.0, p_pulled, -0.0018, "withdraw RJ45 plug axially -Y while holding tab", ROT_R_PULL),
            wp(2.2, p_place_hover, -0.0018, "roll 90 deg over counter clear of modem", ROT_R_PLACE),
            wp(1.8, p_place_down, -0.0018, "lower RJ45 plug gently onto its side", ROT_R_PLACE),
            wp(1.0, p_place_down, FINGER_OPEN, "open jaws 94 mm horizontally along +-X", ROT_R_PLACE),
            wp(1.6, p_place_back, FINGER_OPEN, "retract axially -gz clear of resting plug", ROT_R_PLACE),
            wp(2.6, hr, FINGER_OPEN, "park right arm", rr),
            wp(2.8, hr, FINGER_OPEN, "hold still for evaluation", rr),
        ],
    }
