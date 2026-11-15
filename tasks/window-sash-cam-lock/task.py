"""Close a single-hung vinyl window sash and turn its cam lock into the keeper.

Modeled 1:1 on an 18 x 24 in nominal (17-1/2 x 23-1/2 in actual outer width/height,
Tafco VSH1824B single-hung vinyl utility window; 2-1/2 in frame depth estimated) and a
Prime-Line F 2588 die-cast cam-action sash lock with a 5/16 in keeper.
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
from task_metrics import Metric, rz
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

# -- 1:1 18x24 in single-hung vinyl window + Prime-Line F 2588 geometry -----------------------
WIN_X = 0.4200
WIN_W = 0.4445   # 17-1/2 in actual outer width
WIN_H = 0.5969   # 23-1/2 in actual outer height
WIN_D = 0.0635   # 2-1/2 in actual outer frame depth
SILL_TOP_Z = 0.0280
Z_MEET = 0.3000  # horizontal meeting-rail flush plane when lower sash is closed (dz = 0)

C_FRAME = (0.93, 0.94, 0.95)
C_UPPER = (0.85, 0.88, 0.91)
C_LOWER = (0.97, 0.98, 0.99)
C_BRACKET = (0.28, 0.30, 0.33)
C_KEEPER = (0.22, 0.24, 0.28)
C_LOCK_BASE = (0.24, 0.26, 0.30)
C_LEVER = (0.80, 0.63, 0.24)  # brushed architectural brass finish for clear lock/keeper contrast
C_SEAL = (0.18, 0.20, 0.22)

# Fixed outer frame + fixed exterior upper sash + fixed Prime-Line F 2588 C-keeper:
FRAME_BOXES = (
    # Outer white uPVC window frame (444.5 x 596.9 x 63.5 mm):
    ((0.0, 0.0, 0.0140), (WIN_D, WIN_W, 0.0280), C_FRAME, 0.0),                  # bottom sill
    ((0.0, 0.0, 0.5829), (WIN_D, WIN_W, 0.0280), C_FRAME, 0.0),                  # top head jamb
    ((0.0, +0.20825, 0.29845), (WIN_D, 0.0280, 0.5409), C_FRAME, 0.0),           # left side jamb
    ((0.0, -0.20825, 0.29845), (WIN_D, 0.0280, 0.5409), C_FRAME, 0.0),           # right side jamb
    # Rear steel bench-mounting brackets (+X exterior side):
    ((+0.05175, +0.1850, 0.0060), (0.0400, 0.0440, 0.0120), C_BRACKET, 0.0),
    ((+0.05175, -0.1850, 0.0060), (0.0400, 0.0440, 0.0120), C_BRACKET, 0.0),
    # Fixed exterior upper sash (384 x 297 x 26 mm, open glazing aperture):
    ((+0.0150, 0.0, 0.2850), (0.0260, 0.3840, 0.0300), C_UPPER, 0.0),            # bottom meeting rail (z in [0.270, 0.300])
    ((+0.0150, 0.0, 0.5530), (0.0260, 0.3840, 0.0280), C_UPPER, 0.0),            # top rail
    ((+0.0150, +0.1780, 0.4195), (0.0260, 0.0280, 0.2390), C_UPPER, 0.0),        # left stile
    ((+0.0150, -0.1780, 0.4195), (0.0260, 0.0280, 0.2390), C_UPPER, 0.0),        # right stile
    # Fixed Prime-Line F 2588 keeper base, side risers, and top hood (throat z in [0.3025, 0.3125]):
    ((+0.0080, 0.0, 0.30125), (0.0130, 0.0640, 0.0025), C_KEEPER, 0.0),          # keeper base (2-1/16 in screw span)
    ((+0.0120, +0.0220, 0.3075), (0.0100, 0.0060, 0.0100), C_KEEPER, 0.0),       # left keeper riser
    ((+0.0120, -0.0220, 0.3075), (0.0100, 0.0060, 0.0100), C_KEEPER, 0.0),       # right keeper riser
    ((+0.00925, 0.0, 0.31425), (0.0155, 0.0500, 0.0035), C_KEEPER, 0.0),         # keeper top roof (z in [0.3125, 0.3160])
)

# Smooth cylindrical keeper catch pin (6.0 mm OD at (+0.0045, 0.0), +X face at +0.0075 m):
FRAME_CYLS = (
    ((+0.0045, 0.0, 0.3075), 0.0030, 0.0100, C_KEEPER, "Z"),
)

# Jamb-balance carrier shoes inside left/right jamb tracks (sash_z):
SASH_Z_BOXES = (
    ((-0.0135, +0.1980, 0.1640), (0.0120, 0.0060, 0.0400), C_SEAL, 0.0),
    ((-0.0135, -0.1980, 0.1640), (0.0120, 0.0060, 0.0400), C_SEAL, 0.0),
)

# Interior vertical-sliding lower sash + lock housing base (rides on sash_x -> sash_z):
SASH_X_BOXES = (
    ((-0.0135, 0.0, 0.0430), (0.0260, 0.3840, 0.0300), C_LOWER, 0.0),            # bottom sill rail (z in [0.028, 0.058])
    ((-0.0135, 0.0, 0.2850), (0.0260, 0.3840, 0.0300), C_LOWER, 0.0),            # top meeting rail (z in [0.270, 0.300])
    ((-0.0135, +0.1780, 0.1640), (0.0260, 0.0280, 0.2120), C_LOWER, 0.0),        # left stile
    ((-0.0135, -0.1780, 0.1640), (0.0260, 0.0280, 0.2120), C_LOWER, 0.0),        # right stile
    ((+0.0005, 0.0, 0.2850), (0.0020, 0.3800, 0.0100), C_SEAL, 0.0),             # meeting-rail bulb weatherseal
    ((-0.0110, 0.0, 0.3015), (0.0180, 0.0640, 0.0030), C_LOCK_BASE, 0.0),        # F 2588 lock base housing
)

# Eccentric J-hook crescent cam + flared thumb-turn lever (rides on cam_lock -> sash_x -> sash_z):
CAM_ANCHOR = (-0.0110, 0.0, 0.3030)
LEVER_PHI_DEG = 115.0  # unlocked lever points left-front (+Y, tilted 25 deg toward -X so +Y->-Y approach drives +theta CCW)


def _polar_box(r_c: float, phi_deg: float, z_c: float, r_thk: float, ch_len: float, hgt: float, color, yaw_ex_deg: float = 0.0):
    phi = math.radians(phi_deg)
    cx = CAM_ANCHOR[0] + r_c * math.cos(phi)
    cy = CAM_ANCHOR[1] + r_c * math.sin(phi)
    yaw = phi + math.radians(yaw_ex_deg)
    return ((cx, cy, z_c), (r_thk, ch_len, hgt), color, yaw)


CAM_DWELL_BOX = _polar_box(0.0203, -165.0, 0.3075, 0.0032, 0.0080, 0.0050, C_LEVER, 0.0)
CAM_DWELL_LOCAL = CAM_DWELL_BOX[0]

CAM_BOXES = (
    # 5-segment eccentric J-hook crescent cam wafer in z in [0.3050, 0.3100] (5.0 mm thick inside 10.0 mm keeper throat):
    _polar_box(0.0110, +152.0, 0.3075, 0.0140, 0.0036, 0.0050, C_LEVER, 0.0),    # radial neck
    _polar_box(0.0203, +173.0, 0.3075, 0.0032, 0.0082, 0.0050, C_LEVER, 0.0),    # trailing heel
    CAM_DWELL_BOX,                                                                # locked dwell (world +X at theta=+165 deg)
    _polar_box(0.0213, -148.0, 0.3075, 0.0030, 0.0080, 0.0050, C_LEVER, -6.0),   # eccentric draw-in ramp
    _polar_box(0.0225, -132.0, 0.3075, 0.0028, 0.0074, 0.0050, C_LEVER, -5.0),   # leading entry beak
    # Flared thumb-turn paddle in z in [0.3135, 0.3395] (sweeps +Y -> -X -> -Y on open room side):
    _polar_box(0.0205, LEVER_PHI_DEG, 0.3265, 0.0250, 0.0065, 0.0260, C_LEVER, 0.0),  # radial paddle blade (r=8..33 mm)
    _polar_box(0.0335, LEVER_PHI_DEG, 0.3265, 0.0040, 0.0150, 0.0260, C_LEVER, 0.0),  # outer T-tip finger retainer
)

CAM_CYLS = (
    ((-0.0110, 0.0, 0.3085), 0.0075, 0.0110, C_LEVER, "Z"),                      # central die-cast lock hub
)

SASH_Z_INIT = 0.0550   # starts 55 mm open above sill
SASH_X_INIT = -0.0022  # 2.2 mm compliant bulb-weatherseal preload toward -X

# Absolute joint-axis friction (PhysxJointAxisAPI: N for linear slides, N m for revolute hinge):
# Moving mass = 0.10 + 1.25 + 0.10 = 1.45 kg (14.22 N gravity); 17.5 N static / 16.0 N dynamic holds sash motionless at +55 mm.
labkit.AXIS_FRICTION.update({
    "sash_z": (17.5, 16.0),
    "cam_lock": (0.06, 0.05),
})

WINDOW = part(
    "window",
    (WIN_X, 0.0, TOP),
    boxes=FRAME_BOXES,
    cylinders=FRAME_CYLS,
    fixed=True,
    self_collide=True,
    friction=0.45,
    joints=(
        Joint(
            name="sash_z",
            kind="slide",
            axis="Z",
            anchor=(0.0, 0.0, SILL_TOP_Z),
            boxes=SASH_Z_BOXES,
            lower=0.0,
            upper=0.0800,
            stiffness=0.0,
            damping=12.0,
            target=SASH_Z_INIT,
            initial=SASH_Z_INIT,
            friction=0.0,
            mass=0.10,
        ),
        Joint(
            name="sash_x",
            parent="sash_z",
            kind="slide",
            axis="X",
            anchor=(0.0, 0.0, SILL_TOP_Z),
            boxes=SASH_X_BOXES,
            lower=-0.0030,
            upper=0.0006,
            stiffness=450.0,
            damping=8.0,
            target=SASH_X_INIT,
            initial=SASH_X_INIT,
            friction=0.0,
            mass=1.25,
        ),
        Joint(
            name="cam_lock",
            parent="sash_x",
            kind="hinge",
            axis="Z",
            anchor=CAM_ANCHOR,
            boxes=CAM_BOXES,
            cylinders=CAM_CYLS,
            lower=0.0,
            upper=math.radians(171.0),
            stiffness=0.0,
            damping=0.02,
            target=0.0,
            initial=0.0,
            friction=0.0,
            mass=0.10,
        ),
    ),
)

OBJECTS = [WINDOW]
INSTRUCTION = (
    "Push the raised lower window sash all the way down onto the sill and turn the "
    "meeting-rail cam lock into its keeper to pull the sash shut."
)

_METRIC = Metric(globals())
_step = 0
_DEBUG = os.environ.get("SASH_DEBUG") == "1"
_PROBE = os.environ.get("SASH_PROBE", "")


@on_reset
def _reset(env):
    global _step
    _step = 0
    _METRIC.reset()


@on_step
def _update(env):
    global _step
    _step += 1
    win_s = labkit._part_state(env, WINDOW)
    _METRIC.update(env, win_s)
    if _DEBUG and (_step % 10 == 0 or _step <= 5):
        tgt_dx_mm = None
        try:
            w_asset = env.unwrapped.scene["window"]
            jidx = next(i for i, n in enumerate(w_asset.joint_names) if "sash_x" in n)
            jt = w_asset.data.joint_pos_target
            tgt_dx_mm = round(float((jt.torch if hasattr(jt, "torch") else jt)[0, jidx]) * 1000.0, 2)
        except Exception:
            pass
        print(
            "SASHDBG "
            + json.dumps(
                dict(
                    step=_step,
                    t=round(_step / 20.0, 2),
                    dz_mm=round(float(win_s.joints.get("sash_z", 0.0)) * 1000.0, 2),
                    dx_mm=round(float(win_s.joints.get("sash_x", 0.0)) * 1000.0, 2),
                    tgt_dx_mm=tgt_dx_mm,
                    cam_deg=round(math.degrees(float(win_s.joints.get("cam_lock", 0.0))), 2),
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

# Left arm orientation for pressing the left half of the lower sash meeting rail straight down:
ROT_PUSH_L = make_frame([0.0, 0.0, 1.0], [+0.92, -0.18, -0.35])
TIP_PAST_TCP = 0.0130


def _tcp_for_tip(tip_world, rot: np.ndarray) -> np.ndarray:
    return np.asarray(tip_world, float) - TIP_PAST_TCP * rot[:, 2]


# Push point on lower sash top meeting rail (x_rail = 0.420 - 0.0135 - 0.0022 = 0.4043, y = +0.1150):
X_PUSH_L = WIN_X - 0.0155
Y_PUSH_L = +0.1150
Z_RAIL_OPEN = TOP + Z_MEET + SASH_Z_INIT  # TOP + 0.3550 m
Z_RAIL_CLOSED = TOP + Z_MEET              # TOP + 0.3000 m

P_L_ABOVE = _tcp_for_tip((X_PUSH_L - 0.025, Y_PUSH_L, Z_RAIL_OPEN + 0.036), ROT_PUSH_L)
P_L_CONTACT = _tcp_for_tip((X_PUSH_L + 0.002, Y_PUSH_L, Z_RAIL_OPEN + 0.006), ROT_PUSH_L)
P_L_SEATED = _tcp_for_tip((X_PUSH_L + 0.002, Y_PUSH_L, Z_RAIL_CLOSED - 0.012), ROT_PUSH_L)
P_L_CLEAR = _tcp_for_tip((X_PUSH_L - 0.045, Y_PUSH_L + 0.020, Z_RAIL_CLOSED + 0.048), ROT_PUSH_L)

# Right arm orientation for sweeping the flared thumb-turn paddle +168 deg around CAM_ANCHOR:
ROT_SWEEP_R = make_frame([0.0, 0.0, 1.0], [+0.92, +0.12, -0.37])


def _sweep_tip(dx: float, dy: float, z_sash: float = 0.0, dz_extra: float = 0.0) -> np.ndarray:
    # At tz=0.3335, tip_right min z is TOP+0.3167 (0.7mm above 0.3160 keeper roof) with 22.8mm vertical overlap on [0.3135, 0.3395] paddle.
    return _tcp_for_tip((dx, dy, TOP + 0.3335 + z_sash + dz_extra), ROT_SWEEP_R)


def _right_ccw_sweep_waypoints(z_sash: float, hold_end_s: float) -> list[Waypoint]:
    # 2-stroke tangential CCW sweep (verified across th0=70..95 deg x +-6mm Y perturbation against YAM_URDF finger polygon):
    # Stroke A sweeps cam_deg 0 -> ~82 deg (alpha ~ 197 deg, pointing along -X where u_theta is -Y).
    # Re-indexes through open air at x=0.350 -> (0.393, +0.024) on the +Y broad side of the mid-blade (trapped inside the r=31.5mm T-flange pocket),
    # drives -Y through (0.390, -0.020) past dead-center, then sweeps +X through (0.408, -0.024) -> (0.424, -0.016) into +170 deg keeper dwell:
    return [
        wp(1.2, _sweep_tip(0.375, +0.056, z_sash, +0.018), 0.01, "approach-left", ROT_SWEEP_R),
        wp(0.5, _sweep_tip(0.412, +0.054, z_sash, 0.0), 0.01, "hook-behind", ROT_SWEEP_R),
        wp(0.6, _sweep_tip(0.402, +0.024, z_sash, 0.0), 0.01, "sweep-35", ROT_SWEEP_R),
        wp(0.6, _sweep_tip(0.386, -0.004, z_sash, 0.0), 0.01, "sweep-70", ROT_SWEEP_R),
        wp(0.5, _sweep_tip(0.386, -0.024, z_sash, 0.0), 0.01, "sweep-78", ROT_SWEEP_R),
        wp(0.45, _sweep_tip(0.350, -0.024, z_sash, 0.0), 0.01, "step-back", ROT_SWEEP_R),
        wp(0.50, _sweep_tip(0.350, +0.026, z_sash, 0.0), 0.01, "step-plus-y", ROT_SWEEP_R),
        wp(0.55, _sweep_tip(0.393, +0.024, z_sash, 0.0), 0.01, "inboard-mid", ROT_SWEEP_R),
        wp(0.75, _sweep_tip(0.390, -0.020, z_sash, 0.0), 0.01, "drive-y", ROT_SWEEP_R),
        wp(0.75, _sweep_tip(0.408, -0.024, z_sash, 0.0), 0.01, "drive-xy", ROT_SWEEP_R),
        wp(0.85, _sweep_tip(0.424, -0.016, z_sash, 0.0), 0.01, "seat-170", ROT_SWEEP_R),
        wp(0.45, _sweep_tip(0.424, -0.016, z_sash, 0.0), 0.01, "dwell-170", ROT_SWEEP_R),
        wp(0.80, _sweep_tip(0.360, -0.036, z_sash, +0.022), 0.01, "exit-away", ROT_SWEEP_R),
        wp(1.30, P_R_PARK, 0.20, "retract-r", ROT_R_PARK),
        wp(hold_end_s, P_R_PARK, 0.20, "hold-r", ROT_R_PARK),
    ]


def demo() -> dict[str, list[Waypoint]]:
    if _PROBE == "always0":
        return {
            "left": [wp(5.0, P_L_PARK, 0.20, "park", ROT_L_PARK)],
            "right": [wp(5.0, P_R_PARK, 0.20, "park", ROT_R_PARK)],
        }

    if _PROBE == "close_without_lock":
        # Negative control 1: lower sash onto sill (scores 0.5), leave cam lock unturned:
        return {
            "left": [
                wp(1.6, P_L_ABOVE, 0.02, "above-rail", ROT_PUSH_L),
                wp(0.8, P_L_CONTACT, 0.02, "touch-rail", ROT_PUSH_L),
                wp(2.2, P_L_SEATED, 0.02, "push-down", ROT_PUSH_L),
                wp(0.4, P_L_SEATED, 0.02, "hold-sill", ROT_PUSH_L),
                wp(1.0, P_L_CLEAR, 0.02, "lift-clear", ROT_PUSH_L),
                wp(1.6, P_L_PARK, 0.20, "retract", ROT_L_PARK),
                wp(2.2, P_L_PARK, 0.20, "hold", ROT_L_PARK),
            ],
            "right": [
                wp(9.8, P_R_PARK, 0.20, "park", ROT_R_PARK),
            ],
        }

    if _PROBE == "turn_before_close":
        # Negative control 2: sweep cam lock +168 deg FIRST while sash is raised 55 mm, then push sash down (cam hits keeper roof -> 0.0):
        return {
            "left": [
                wp(9.1, P_L_PARK, 0.20, "wait-turn", ROT_L_PARK),
                wp(1.4, P_L_ABOVE, 0.02, "above-rail", ROT_PUSH_L),
                wp(0.7, P_L_CONTACT, 0.02, "touch-rail", ROT_PUSH_L),
                wp(2.2, P_L_SEATED, 0.02, "push-blocked", ROT_PUSH_L),
                wp(1.0, P_L_CLEAR, 0.02, "lift-clear", ROT_PUSH_L),
                wp(1.5, P_L_PARK, 0.20, "retract", ROT_L_PARK),
                wp(2.2, P_L_PARK, 0.20, "hold", ROT_L_PARK),
            ],
            "right": _right_ccw_sweep_waypoints(z_sash=SASH_Z_INIT, hold_end_s=8.65),
        }

    if _PROBE in ("locked_pull", "unlocked_pull"):
        # Retention verification: close sash (and lock if locked_pull), then hook BOTH hands symmetrically
        # underneath the lower sash meeting rail inside the open glazing aperture (y = +-0.090 m, 24 mm clear
        # of the +-0.164 m stiles; hull support peak centered at x = 0.4043 m under the [0.3913, 0.4173] rail).
        # Analytical inverse-PD compliance (K = [80,80,80,10,10,10] Nm/rad) pre-compensates wrist/shoulder sag
        # so the loaded pad stays nose-up (+8 deg) dead on x = 0.4043 m through 34.4 N breakaway (> 31.7 N).
        from yam_arena.constants import SHOULDER, WRIST

        k_gains = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3, float)

        def _lift_seq(base_pos, y_sgn: float):
            base = np.asarray(base_pos, float)
            q_prev = np.array(READY, float)
            out = []
            stages = (
                ("under-pre",    1.1,  0.0, 0.255, 12.7, 0.3350),
                ("under-rail",   0.6,  0.0, 0.258, 12.7, 0.4043),
                ("seat-under",   0.4,  2.5, 0.272, 11.5, 0.4043),
                ("preload-half", 0.5,  9.5, 0.273,  9.5, 0.4043),
                ("breakaway",    0.6, 17.5, 0.279,  8.5, 0.4043),
                ("slide-26mm",   0.6, 16.2, 0.304,  8.5, 0.4043),
                ("slide-52mm",   0.7, 16.2, 0.330,  8.5, 0.4043),
                ("unload-half",  0.4,  5.0, 0.268, 11.0, 0.4000),
                ("drop-clear",   0.4,  0.0, 0.252, 12.7, 0.3944),
                ("back-out",     0.6,  0.0, 0.252, 12.7, 0.3350),
            )
            for lbl, dur, fz, z_sup, p_deg, x_sup in stages:
                pr = math.radians(p_deg)
                gz = [math.cos(pr) * 0.985, -0.15 * y_sgn, math.sin(pr)]
                rot_eq = make_frame([0.0, 0.0, 1.0], gz)
                tcp_eq = np.array([x_sup - 0.0099, y_sgn * 0.090, TOP + z_sup - 0.0062])
                q_eq, _, _ = ik.solve(tcp_eq, ik.matrix_to_quat_xyzw(rot_eq), q_prev, base_pos=base, iters=100)
                q_prev = q_eq.copy()
                if fz > 0.0:
                    fp0, fr0 = ik.tcp_pose(q_eq, base)
                    ps0 = fp0 + 0.010 * fr0[:, 2]
                    jac = np.zeros((3, 6))
                    eps = 1e-5
                    for j in range(6):
                        dq = q_eq.copy()
                        dq[j] += eps
                        fp1, fr1 = ik.tcp_pose(dq, base)
                        jac[:, j] = ((fp1 + 0.010 * fr1[:, 2]) - ps0) / eps
                    q_cmd = q_eq + (fz * jac[2, :]) / k_gains
                    p_cmd, r_cmd = ik.tcp_pose(q_cmd, base)
                else:
                    p_cmd, r_cmd = tcp_eq, rot_eq
                out.append(wp(dur, p_cmd, 0.0, lbl, r_cmd))
            return out

        seq_l = _lift_seq(LEFT_BASE_POS, +1.0)
        seq_r = _lift_seq(RIGHT_BASE_POS, -1.0)
        do_lock = _PROBE == "locked_pull"
        return {
            "left": [
                wp(1.5, P_L_ABOVE, 0.02, "above-rail", ROT_PUSH_L),
                wp(0.7, P_L_CONTACT, 0.02, "touch-rail", ROT_PUSH_L),
                wp(2.2, P_L_SEATED, 0.02, "push-down", ROT_PUSH_L),
                wp(0.4, P_L_SEATED, 0.02, "hold-sill", ROT_PUSH_L),
                wp(0.9, P_L_CLEAR, 0.02, "lift-clear", ROT_PUSH_L),
                wp(1.5, P_L_PARK, 0.20, "retract-l", ROT_L_PARK),
                wp(8.6 if do_lock else 0.4, P_L_PARK, 0.20, "wait-lock", ROT_L_PARK),
                *seq_l,
                wp(1.4, P_L_PARK, 0.20, "retract-end", ROT_L_PARK),
                wp(1.8, P_L_PARK, 0.20, "hold-end", ROT_L_PARK),
            ],
            "right": (
                [
                    wp(5.4, P_R_PARK, 0.20, "wait-close", ROT_R_PARK),
                    *_right_ccw_sweep_waypoints(z_sash=0.0, hold_end_s=0.60),
                    *seq_r,
                    wp(1.4, P_R_PARK, 0.20, "retract-end", ROT_R_PARK),
                    wp(1.8, P_R_PARK, 0.20, "hold-end", ROT_R_PARK),
                ]
                if do_lock
                else [
                    wp(7.6, P_R_PARK, 0.20, "wait-close", ROT_R_PARK),
                    *seq_r,
                    wp(1.4, P_R_PARK, 0.20, "retract-end", ROT_R_PARK),
                    wp(1.8, P_R_PARK, 0.20, "hold-end", ROT_R_PARK),
                ]
            ),
        }

    # Nominal full demo: left arm lowers sash onto sill & retracts -> right arm sweeps cam lock +168 deg into keeper & retracts:
    return {
        "left": [
            wp(1.5, P_L_ABOVE, 0.02, "above-rail", ROT_PUSH_L),
            wp(0.7, P_L_CONTACT, 0.02, "touch-rail", ROT_PUSH_L),
            wp(2.2, P_L_SEATED, 0.02, "push-down", ROT_PUSH_L),
            wp(0.4, P_L_SEATED, 0.02, "hold-sill", ROT_PUSH_L),
            wp(0.9, P_L_CLEAR, 0.02, "lift-clear", ROT_PUSH_L),
            wp(1.5, P_L_PARK, 0.20, "retract-l", ROT_L_PARK),
            wp(9.85, P_L_PARK, 0.20, "hold-l", ROT_L_PARK),
        ],
        "right": [
            wp(5.4, P_R_PARK, 0.20, "wait-close", ROT_R_PARK),
            *_right_ccw_sweep_waypoints(z_sash=0.0, hold_end_s=1.8),
        ],
    }
