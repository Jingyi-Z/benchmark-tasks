"""Unlatch and remove a pull-tab QSFP28 optical transceiver from a 1U rack switch.

Exterior transceiver dimensions follow SFF-8661 / Coherent FTLC9558REPM Type 2B pull-tab
QSFP28 (18.35 mm W x 8.50 mm H x 72.0 mm L die-cast housing with a 28.0 mm protruding
release pull-tab) seated inside a 1U 44.0 mm leaf-switch faceplate mounted at U3 on a
loose EIA-310 19-inch (482.6 mm) two-post desktop test rack stand. Dual spring-loaded
cage side-latching pawls engage flat 90-degree shoulders on the transceiver body and cam
open when the sliding pull-tab translates 6.0 mm forward; ramp angles, spring rates,
bounded contact patches, and the omitted fibre cord are disclosed engineering estimates.
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
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 900
SUCCESS_HOLD_S = 1.0
labkit.CONTACT_OFFSET = 0.0001

RACK_BLACK = (0.16, 0.18, 0.21)
SWITCH_SLATE = (0.28, 0.31, 0.36)
CAGE_SILVER = (0.70, 0.73, 0.77)
ZINC_BODY = (0.62, 0.65, 0.70)
PULLTAB_BLUE = (0.14, 0.48, 0.82)

# --- 1. Loose EIA-310 19-inch two-post desktop test rack (2.4 kg) + full-depth 1U switch (439 x 44 x 406 mm, 8.8 kg = 11.2 kg total) ---
# Origin at (RX = 0.430, RY = 0.000, TOP); 1U switch front faceplate at x_local = -0.0900 (world x = 0.3400 m), rear at +0.3160 (406 mm deep).
RX, RY = 0.430, 0.000
CAGE_FRONT_LOCAL_X = -0.0900
PORT_Y = -0.0750
PORT_Z = 0.1250

SWITCH_RACK_BOXES = [
    # Left (+Y) and right (-Y) steel desktop rack base skids (380 mm deep along X: [-0.095, +0.285] supporting full 406 mm switch CoM!)
    ((+0.0950, +0.2350, 0.0040), (0.3800, 0.0320, 0.0080), RACK_BLACK),
    ((+0.0950, -0.2350, 0.0040), (0.3800, 0.0320, 0.0080), RACK_BLACK),
    # Rear base tie-bar connecting the two desktop rack skids (482.6 mm span)
    ((+0.2700, 0.0000, 0.0040), (0.0300, 0.4380, 0.0080), RACK_BLACK),
    # Two vertical EIA-310 19-inch 5U desktop rack posts (250 mm tall, outer span 482.6 mm -> post centers y = +-0.2293 m)
    ((-0.0780, +0.2293, 0.1250), (0.0240, 0.0240, 0.2500), RACK_BLACK),
    ((-0.0780, -0.2293, 0.1250), (0.0240, 0.0240, 0.2500), RACK_BLACK),
    # Top rigid 5U rack cross-tie at z = 0.244 m (leaves U4/U5 open above the U3 1U switch)
    ((-0.0780, 0.0000, 0.2440), (0.0240, 0.4346, 0.0120), RACK_BLACK),
    # Full-size 1U leaf switch chassis at U3 (439 mm W x 44.0 mm H x 406.0 mm D: x_local in [-0.0900, +0.3160], z in [0.1030, 0.1470]):
    # Bottom tray & top cover across full 406 mm depth
    ((+0.1130, 0.0000, 0.1090), (0.4060, 0.4390, 0.0120), SWITCH_SLATE),
    ((+0.1130, 0.0000, 0.1410), (0.4060, 0.4390, 0.0120), SWITCH_SLATE),
    # Switch chassis right (-Y of port) and left (+Y of port) solid sections across full 406 mm depth flanking cage bay at PORT_Y = -0.075
    ((+0.1130, -0.1535, 0.1250), (0.4060, 0.1320, 0.0200), SWITCH_SLATE),
    ((+0.1130, +0.0835, 0.1250), (0.4060, 0.2720, 0.0200), SWITCH_SLATE),
    # Rear chassis body behind cage bay (x_local in [-0.0160, +0.3160] = 332 mm)
    ((+0.1500, PORT_Y, 0.1250), (0.3320, 0.0350, 0.0200), SWITCH_SLATE),
    # Mounting ears bolted to the 19-inch posts (482.6 mm total span)
    ((-0.0880, +0.2293, 0.1250), (0.0040, 0.0240, 0.0440), SWITCH_SLATE),
    ((-0.0880, -0.2293, 0.1250), (0.0040, 0.0240, 0.0440), SWITCH_SLATE),
    # Original SFF-8661 QSFP28 EMI cage around (PORT_Y = -0.0750, PORT_Z = 0.1250):
    # Inner guide tunnel: y_rel in [-0.0098, +0.0098] (19.6 mm), z_rel in [-0.0048, +0.0048] (9.6 mm)
    ((-0.0550, PORT_Y, PORT_Z - 0.0064), (0.0700, 0.0240, 0.0032), CAGE_SILVER),  # cage floor
    ((-0.0550, PORT_Y, PORT_Z + 0.0064), (0.0700, 0.0240, 0.0032), CAGE_SILVER),  # cage ceiling
    # Front bezel side guides (x_sw in [-0.0900, -0.0835], inner |y_rel| = 0.0098, leaving [-0.0835, -0.0705] open for side pawls)
    ((-0.08675, PORT_Y + 0.0109, PORT_Z), (0.0065, 0.0022, 0.0096), CAGE_SILVER),
    ((-0.08675, PORT_Y - 0.0109, PORT_Z), (0.0065, 0.0022, 0.0096), CAGE_SILVER),
    # Rear cage side guides (x_sw in [-0.0705, -0.0200], inner |y_rel| = 0.0098) and rear connector backstop
    ((-0.04525, PORT_Y + 0.0109, PORT_Z), (0.0505, 0.0022, 0.0096), CAGE_SILVER),
    ((-0.04525, PORT_Y - 0.0109, PORT_Z), (0.0505, 0.0022, 0.0096), CAGE_SILVER),
    ((-0.0180, PORT_Y, PORT_Z), (0.0040, 0.0240, 0.0128), CAGE_SILVER),
]

# Dual spring-loaded side-latching pawls at x_sw = -0.0785 (size_x = 0.0030 -> [-0.0800, -0.0770]):
# Upper tooth (z = PORT_Z + 0.0025) engages flat 90-deg module shoulder; lower lug (z = PORT_Z - 0.0025) rides pull-tab cam!
LEFT_PAWL_JOINT = Joint(
    "left_pawl",
    kind="slide",
    axis="Y",
    anchor=(-0.0785, PORT_Y + 0.0105, PORT_Z),
    lower=0.0,
    upper=0.0040,
    initial=0.0,
    target=0.0,
    stiffness=18.0,
    damping=0.55,
    friction=0.0,
    mass=0.008,
    boxes=[
        # Outer carrier block outside cage
        ((-0.0785, PORT_Y + 0.0128, PORT_Z), (0.0050, 0.0030, 0.0088), CAGE_SILVER),
        # Upper flat 90-deg retention tooth (inner face at y_rel = +0.0073 -> 1.88 mm per side = 3.75 mm total lock on 18.35 mm module!)
        ((-0.0785, PORT_Y + 0.0094, PORT_Z + 0.0025), (0.0030, 0.0042, 0.0030), CAGE_SILVER),
        # Lower cam follower lug (inner face at y_rel = +0.0065 -> +2.90 mm dwell opens upper tooth to 0.0102 > 0.009175 by 1.02 mm/side!)
        ((-0.0785, PORT_Y + 0.0086, PORT_Z - 0.0025), (0.0030, 0.0042, 0.0030), CAGE_SILVER),
    ],
)

RIGHT_PAWL_JOINT = Joint(
    "right_pawl",
    kind="slide",
    axis="Y",
    anchor=(-0.0785, PORT_Y - 0.0105, PORT_Z),
    lower=-0.0040,
    upper=0.0,
    initial=0.0,
    target=0.0,
    stiffness=18.0,
    damping=0.55,
    friction=0.0,
    mass=0.008,
    boxes=[
        ((-0.0785, PORT_Y - 0.0128, PORT_Z), (0.0050, 0.0030, 0.0088), CAGE_SILVER),
        ((-0.0785, PORT_Y - 0.0094, PORT_Z + 0.0025), (0.0030, 0.0042, 0.0030), CAGE_SILVER),
        ((-0.0785, PORT_Y - 0.0086, PORT_Z - 0.0025), (0.0030, 0.0042, 0.0030), CAGE_SILVER),
    ],
)

SWITCH_RACK = part(
    "switch_rack",
    pos=(RX, RY, TOP),
    fixed=False,
    mass=11.20,
    friction=0.38,
    self_collide=False,
    boxes=SWITCH_RACK_BOXES,
    joints=[LEFT_PAWL_JOINT, RIGHT_PAWL_JOINT],
)

# --- 2. SFF-8661 Type 2B pull-tab QSFP28 optical transceiver (18.35 x 8.50 x 72.0 mm) ---
# Seated at TX0 = 0.3640 m -> front zinc face x_world = 0.3280 (12.0 mm outside 0.3400 cage front),
# rear face x_world = 0.4000 (60.0 mm inside cage). Pawl window [-0.0800, -0.0770] maps to x_mod in [-0.0140, -0.0110].
TX0, TY0, TZ0 = 0.3640, PORT_Y, TOP + PORT_Z

TRANSCEIVER_BOXES = [
    # Upper half (z_mod in [+0.0008, +0.00425]):
    # Front zinc collar (x_mod in [-0.0360, -0.0155], full 18.35 mm width — exposed 12 mm outside cage for housing_pull probe!)
    ((-0.02575, 0.0, +0.002525), (0.0205, 0.01835, 0.00345), ZINC_BODY),
    # Narrowed notch neck (x_mod in [-0.0155, -0.0095], 12.4 mm width -> creates flat 90-deg side notches around x_mod = -0.0125!)
    ((-0.01250, 0.0, +0.002525), (0.0060, 0.01240, 0.00345), ZINC_BODY),
    # Rear upper body (x_mod in [-0.0095, +0.0360], full 18.35 mm width -> flat -X shoulder at -0.0095 locks against pawls!)
    ((+0.01325, 0.0, +0.002525), (0.0455, 0.01835, 0.00345), ZINC_BODY),
    # Lower half (z_mod in [-0.00425, -0.0008]): 8.0 mm core leaving clean side channels for pull-tab cam bars!
    ((0.00000, 0.0, -0.002525), (0.0720, 0.00800, 0.00345), ZINC_BODY),
]

# Sliding release pull-tab (translates 0 -> -8.5 mm along -X relative to transceiver):
# Protrudes 28.0 mm ahead of front zinc face (-0.0360 -> -0.0640 m, i.e., world x = 0.3000..0.3280 m!).
# Note on USD RotateZ(+yaw): +yaw rotates +X toward +Y, so +28 deg on +Y side puts the NARROW nose (|y_outer| = 0.00575 < 0.00650)
# at the -X leading end (x_tab = -0.0108 > -0.0110) and the WIDE ramp end (|y_outer| = 0.00948 < 0.00980 cage wall) at x_tab = -0.0052,
# seamlessly feeding onto the 10.0 mm parallel dwell land [-0.0055, +0.0045] at |y_outer| = 0.00950!
PULL_TAB_JOINT = Joint(
    "pull_tab",
    kind="slide",
    axis="X",
    anchor=(-0.0360, 0.0, 0.0),
    lower=-0.0105,
    upper=0.0,
    initial=0.0,
    target=0.0,
    stiffness=4.0,
    damping=0.25,
    friction=0.0,
    mass=0.010,
    boxes=[
        # External blue elastomer pull-tab grip strap (24.0 mm long x 11.0 mm wide x 4.6 mm thick at x_mod in [-0.0640, -0.0400],
        # narrower than the 18.35 mm zinc collar so side-pinching the housing collar never catches the pull-tab strap!)
        ((-0.0520, 0.0, -0.0010), (0.0240, 0.0110, 0.0046), PULLTAB_BLUE),
        # Side connecting runner strips into lower cam channels (x_mod in [-0.0420, -0.0095], |y| in [0.0048, 0.0058] < 0.00650 follower!)
        ((-0.02575, +0.00530, -0.002525), (0.0325, 0.0010, 0.0028), PULLTAB_BLUE),
        ((-0.02575, -0.00530, -0.002525), (0.0325, 0.0010, 0.0028), PULLTAB_BLUE),
        # Angled 28-deg cam wedges: leading -X nose at x_tab = -0.01055 > -0.0110, |y_outer| = 0.00610 < 0.00650 follower;
        # trailing +X peak at (x_tab = -0.00420, |y_outer| = 0.00948), 0.32 mm (> 2 * CONTACT_OFFSET = 0.20 mm) inside original 0.00980 cage wall!
        ((-0.0070, +0.00708, -0.002525), (0.0072, 0.0016, 0.0028), PULLTAB_BLUE, +math.radians(28.0)),
        ((-0.0070, -0.00708, -0.002525), (0.0072, 0.0016, 0.0028), PULLTAB_BLUE, -math.radians(28.0)),
        # Parallel cam dwell lands starting at x_tab = -0.00410 (strictly behind -0.00420 ramp peak!) with |y_outer| = 0.00935 < 0.00948
        # (recessed 0.13 mm inside ramp peak so ZERO flat -X step, opening upper 7.30 mm tooth to 10.15 mm > 9.175 mm module half-width!):
        ((+0.0014, +0.00805, -0.002525), (0.0110, 0.0026, 0.0028), PULLTAB_BLUE),
        ((+0.0014, -0.00805, -0.002525), (0.0110, 0.0026, 0.0028), PULLTAB_BLUE),
    ],
)

TRANSCEIVER = part(
    "transceiver",
    pos=(TX0, TY0, TZ0),
    fixed=False,
    mass=0.035,
    friction=0.45,
    self_collide=False,
    boxes=TRANSCEIVER_BOXES,
    joints=[PULL_TAB_JOINT],
)

OBJECTS = [SWITCH_RACK, TRANSCEIVER]

from task_metrics import Metric  # noqa: E402

_METRIC = Metric(globals())
_step = 0
_DEBUG = os.environ.get("QSFP_DEBUG") == "1"
_PROBE = os.environ.get("QSFP_PROBE", "")


@on_reset
def _reset(env):
    global _step
    _step = 0
    _METRIC.reset()
    for name in ("switch_rack", "transceiver"):
        art = env.unwrapped.scene[name]
        q0 = art.data.joint_pos.clone()
        v0 = art.data.joint_vel.clone()
        q0.zero_()
        v0.zero_()
        art.write_joint_state_to_sim(q0, v0)
        art.set_joint_position_target(art.data.default_joint_pos)


@on_step
def _update(env):
    global _step
    _step += 1
    for name in ("switch_rack", "transceiver"):
        art = env.unwrapped.scene[name]
        art.set_joint_position_target(art.data.default_joint_pos)

    sw_s = labkit._part_state(env, SWITCH_RACK)
    tx_s = labkit._part_state(env, TRANSCEIVER)
    _METRIC.update(env, sw_s, tx_s)
    if _DEBUG and (_step % 10 == 0 or _step <= 5):
        print(
            "QSDBG "
            + json.dumps(
                dict(
                    step=_step,
                    t=round(_step / 20.0, 2),
                    sw_x=round(float(sw_s.pos[0]), 4),
                    tx_pos=[round(float(v), 4) for v in tx_s.pos],
                    lp_mm=round(float(sw_s.joints.get("left_pawl", 0.0)) * 1000.0, 2),
                    rp_mm=round(float(sw_s.joints.get("right_pawl", 0.0)) * 1000.0, 2),
                    **_METRIC.detail,
                )
            ),
            flush=True,
        )


def score(objects) -> float:
    return _METRIC.score()


def success(objects) -> bool:
    return score(objects) == 1.0


# --- Fixed open-loop two-arm trajectory (zero simulator object-pose feedback) ---
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


# Left Hand steadies the 1U switch front faceplate (x_world = 0.3400 m, z in [TOP + 0.103, TOP + 0.147]) in +X
# at y = +0.075 m (150 mm clear of PORT_Y = -0.075 -> >= 40 mm inter-arm AABB):
ROT_L_BRACE = make_frame([0.0, -1.0, 0.0], [0.985, 0.0, -0.174])
P_L_HOVER = np.array([0.255, +0.075, TOP + 0.139])
P_L_BRACE = np.array([0.3280, +0.075, TOP + 0.127])
Q_L_PARK = np.array(READY, float) + np.array([0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
P_L_PARK, ROT_L_PARK = ik.tcp_pose(Q_L_PARK, np.array(LEFT_BASE_POS, float))

# Right Hand pinches the horizontal 4.6 mm-thick blue pull-tab paddle (world x in [0.300, 0.324], y = -0.080, z = TOP + 0.124)
# with jaws closing vertically (gx = [0, 0, 1]) and gz pointing along +X, placing [0.310, 0.324] squarely on the [-3, +11] mm flat pad:
ROT_R_TAB = make_frame([0.0, 0.0, 1.0], [0.990, 0.0, -0.141])
P_TAB_GRASP = np.array([0.313, PORT_Y, TOP + 0.1252])
P_TAB_PRE = np.array([0.235, PORT_Y, TOP + 0.136])
# Continuous linear -X pull: first 10.5 mm cams open both side pawls onto recessed parallel dwell, next 95 mm withdraws 72 mm module cleanly out of 0.340 cage!
P_TAB_CAM = np.array([0.2995, PORT_Y, TOP + 0.1245])
P_TAB_OUT = np.array([0.202, PORT_Y, TOP + 0.1245])

# Gentle counter set-down at (x = 0.215, y = -0.145) >= 120 mm clear of rack skids (x >= 0.330):
ROT_R_PLACE = make_frame([0.0, 0.0, 1.0], [0.75, 0.0, -0.66])
P_R_CARRY = np.array([0.215, -0.145, TOP + 0.095])
P_R_PLACE = np.array([0.215, -0.145, TOP + 0.032])
P_R_BACK = P_R_PLACE - 0.085 * ROT_R_PLACE[:, 2] + np.array([0.0, 0.0, +0.035])
Q_R_PARK = np.array(READY, float) + np.array([-0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
P_R_PARK, ROT_R_PARK = ik.tcp_pose(Q_R_PARK, np.array(RIGHT_BASE_POS, float))

FR_PRE = -0.015
FR_HOLD = 0.0

# Housing-pull negative control: pinch across Y (gx = [0, -1, 0]) on the 18.35 mm exposed zinc collar (world x in [0.328, 0.340])
# from slightly above so jaws grab ONLY the zinc collar without pulling the lower release pull-tab!
ROT_R_COLLAR = make_frame([0.0, -1.0, 0.0], [0.82, 0.0, -0.57])
P_COLLAR = np.array([0.331, PORT_Y, TOP + 0.130])


def demo():
    if _PROBE == "housing_pull":
        left = [
            wp(1.4, P_L_HOVER, 0.0, "approach switch faceplate", ROT_L_BRACE),
            wp(1.0, P_L_BRACE, 0.0, "brace switch faceplate in +X", ROT_L_BRACE),
            wp(5.8, P_L_BRACE, 0.0, "hold faceplate while pulling housing", ROT_L_BRACE),
            wp(1.2, P_L_HOVER, FINGER_OPEN, "retract brace", ROT_L_BRACE),
            wp(1.8, P_L_PARK, FINGER_OPEN, "park left", ROT_L_PARK),
        ]
        right = [
            wp(1.6, P_COLLAR - 0.075 * ROT_R_COLLAR[:, 2], -0.015, "approach zinc collar", ROT_R_COLLAR),
            wp(1.2, P_COLLAR, -0.015, "seat on zinc collar only", ROT_R_COLLAR),
            wp(0.8, P_COLLAR, -0.0078, "pinch 18.35 mm zinc collar (15.6 mm > 11.0 mm pull-tab!)", ROT_R_COLLAR),
            wp(2.2, P_COLLAR + np.array([-0.065, 0.0, 0.0]), -0.0078, "pull housing without tab -> locked!", ROT_R_COLLAR),
            wp(0.8, P_COLLAR + np.array([-0.065, 0.0, 0.0]), -0.016, "release", ROT_R_COLLAR),
            wp(1.8, P_R_PARK, FINGER_OPEN, "park right", ROT_R_PARK),
            wp(2.8, P_R_PARK, FINGER_OPEN, "hold", ROT_R_PARK),
        ]
        return {"left": left, "right": right}

    if _PROBE == "unbraced_pull":
        left = [wp(14.0, P_L_PARK, FINGER_OPEN, "left parked", ROT_L_PARK)]
        right = [
            wp(1.6, P_TAB_PRE, FR_PRE, "approach pull-tab", ROT_R_TAB),
            wp(1.2, P_TAB_GRASP, FR_PRE, "align on pull-tab", ROT_R_TAB),
            wp(0.8, P_TAB_GRASP, FR_HOLD, "pinch pull-tab", ROT_R_TAB),
            wp(2.6, P_TAB_OUT, FR_HOLD, "pull unbraced", ROT_R_TAB),
            wp(0.8, P_TAB_OUT, FR_PRE, "release", ROT_R_TAB),
            wp(2.0, P_R_PARK, FINGER_OPEN, "park right", ROT_R_PARK),
            wp(5.0, P_R_PARK, FINGER_OPEN, "hold", ROT_R_PARK),
        ]
        return {"left": left, "right": right}

    left = [
        wp(1.4, P_L_HOVER, 0.0, "approach 1U switch faceplate", ROT_L_BRACE),
        wp(1.0, P_L_BRACE, 0.0, "brace switch faceplate in +X", ROT_L_BRACE),              # t = 2.4 s
        wp(6.6, P_L_BRACE, 0.0, "steady switch during pull-tab extraction", ROT_L_BRACE),  # t = 9.0 s
        wp(1.2, P_L_HOVER, FINGER_OPEN, "retract brace cleanly", ROT_L_BRACE),             # t = 10.2 s
        wp(1.8, P_L_PARK, FINGER_OPEN, "park left clear", ROT_L_PARK),                     # t = 12.0 s
        wp(7.6, P_L_PARK, FINGER_OPEN, "hold still", ROT_L_PARK),                          # t = 19.6 s
    ]
    right = [
        wp(1.6, P_TAB_PRE, FR_PRE, "approach pull-tab", ROT_R_TAB),
        wp(1.2, P_TAB_GRASP, FR_PRE, "align on pull-tab", ROT_R_TAB),
        wp(0.8, P_TAB_GRASP, FR_HOLD, "pinch pull-tab firmly", ROT_R_TAB),                # t = 3.6 s
        wp(1.2, P_TAB_CAM, FR_HOLD, "slide pull-tab 8.5 mm to cam open pawls", ROT_R_TAB),  # t = 4.8 s
        wp(2.6, P_TAB_OUT, FR_HOLD, "withdraw unlatched QSFP28 from cage", ROT_R_TAB),    # t = 7.4 s
        wp(1.6, P_R_CARRY, FR_HOLD, "carry clear over counter", ROT_R_PLACE),             # t = 9.0 s
        wp(1.6, P_R_PLACE, FR_HOLD, "lower gently to counter", ROT_R_PLACE),              # t = 10.6 s
        wp(0.8, P_R_PLACE, FR_PRE, "release pull-tab gently", ROT_R_PLACE),               # t = 11.4 s
        wp(1.4, P_R_BACK, FINGER_OPEN, "retract cleanly along -gz", ROT_R_PLACE),         # t = 12.8 s
        wp(1.8, P_R_PARK, FINGER_OPEN, "park right clear", ROT_R_PARK),                   # t = 14.6 s
        wp(5.0, P_R_PARK, FINGER_OPEN, "hold still", ROT_R_PARK),                         # t = 19.6 s
    ]
    return {"left": left, "right": right}
