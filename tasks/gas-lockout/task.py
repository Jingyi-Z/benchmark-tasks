"""On the station table: a yellow gas line with a quarter-turn ball valve (open), a lockout tab under the valve's
handle, and, downstream of it, a dial gauge and a small bleed valve; a lockout padlock with a danger tag, open, standing
on a stand.

Locking out the line, as a lockout procedure has it: shut the isolating valve (its handle a quarter turn, across the
pipe, against its stop); lock it shut with the padlock, its shackle through the hole in the handle and the hole in the
lockout tab under it, pressed down until it latches; then release the gas trapped downstream through the bleed valve
and check the gauge reads zero. Opening the bleed while the isolating valve is still open vents the live supply.

The padlock: a laminated steel body (40 x 20 x 35 mm) with a red cover and a hardened steel shackle (6 mm rod, legs
24 mm apart). Open, its shackle is raised 28 mm: its short leg's end stands 20 mm above the body, so the handle and the
tab (13 mm stacked) slide in between, and pressing the shackle down puts the short leg through both holes into its
socket, where it latches. It does not open again without its key.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

STEEL = (0.62, 0.63, 0.66)
YELLOW = (0.95, 0.78, 0.10)
BRASS = (0.78, 0.62, 0.28)
RED = (0.85, 0.12, 0.12)
DARK = (0.18, 0.19, 0.21)
BLUE = (0.15, 0.30, 0.75)
WHITE = (0.95, 0.95, 0.93)
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

# -- the gas line (one fixed part; frame origin on the table under the isolating valve's centre) ------------------------
# The supply comes in from -y; downstream (+y) are the gauge and the bleed valve.
VX, VY = 0.30, -0.02
PIPE_Z, PIPE_R, PIPE_L = 0.060, 0.016, 0.40  # the pipe runs along y, from y -0.20 to +0.20 in this frame
PIPE_TOP = PIPE_Z + PIPE_R
BODY = 0.060  # brass ball-valve body, a 6 cm cube round the pipe
STEM_TOP = 0.105
HANDLE_Z = 0.1075  # handle mid-plane; it is 5 mm thick: underside 0.105, top 0.110
HANDLE_T, HANDLE_W, HANDLE_L = 0.005, 0.022, 0.130
HOLE = 0.014  # square lock holes in the handle and the tab, 14 mm
HOLE_R = 0.045  # the handle's lock hole is 4.5 cm from the pivot
GRIP_R0, GRIP_R1 = 0.075, 0.130  # red grip sleeve along the handle's outer part
SLEEVE_W, SLEEVE_T = 0.026, 0.012
TAB_Z0, TAB_T, TAB = 0.097, 0.004, 0.036  # the lockout tab: a 3.6 cm plate, 4 mm thick, 4 mm under the closed handle
CLOSED = math.pi / 2  # the handle's stop: turned from +y (along the pipe) to -x (across it, towards the robots)


def _frame(cx, cy, z, t, outer, hole, colour):
    """A square plate with a square hole in the middle, as four boxes: (centre, size, colour)."""
    side = (outer - hole) / 2
    off = (outer + hole) / 4
    return [((cx, cy + off, z), (outer, side, t), colour), ((cx, cy - off, z), (outer, side, t), colour),
            ((cx + off, cy, z), (side, hole, t), colour), ((cx - off, cy, z), (side, hole, t), colour)]


_rail = (HANDLE_W - HOLE) / 2
HANDLE = Joint(
    "handle", kind="hinge", axis="Z", anchor=(0.0, 0.0, HANDLE_Z), lower=0.0, upper=CLOSED, initial=0.0,
    stiffness=0.0, damping=0.5, target=0.0, mass=0.12,
    boxes=[((0.0, (HOLE_R - HOLE / 2) / 2, HANDLE_Z), (HANDLE_W, HOLE_R - HOLE / 2, HANDLE_T), STEEL),
           ((HOLE / 2 + _rail / 2, HOLE_R, HANDLE_Z), (_rail, HOLE, HANDLE_T), STEEL),
           ((-HOLE / 2 - _rail / 2, HOLE_R, HANDLE_Z), (_rail, HOLE, HANDLE_T), STEEL),
           ((0.0, (HOLE_R + HOLE / 2 + HANDLE_L) / 2, HANDLE_Z), (HANDLE_W, HANDLE_L - HOLE_R - HOLE / 2, HANDLE_T),
            STEEL),
           ((0.0, (GRIP_R0 + GRIP_R1) / 2, HANDLE_Z), (SLEEVE_W, GRIP_R1 - GRIP_R0, SLEEVE_T), RED)],
    cylinders=[((0.0, 0.0, HANDLE_Z), 0.012, HANDLE_T, STEEL)],
)
TAB_X = -HOLE_R  # the tab sits under the closed handle's hole

# The gauge: a 50 mm dial facing up on a short stem off the pipe's top, 7.5 cm downstream of the valve, low enough that
# the open handle passes over it. 0 to 250 kPa over 270 degrees.
G_Y = 0.075
G_R, G_Z = 0.025, 0.095  # dial radius; the dial's face height
NEEDLE_PER_KPA = math.radians(270.0) / 250.0
P_LINE = 200.0  # kPa (gauge): the supply pressure, and the line's at the start
NEEDLE = Joint(
    "needle", kind="hinge", axis="Z", anchor=(0.0, G_Y, G_Z + 0.001), lower=-4.8, upper=0.1,
    initial=-P_LINE * NEEDLE_PER_KPA, target=-P_LINE * NEEDLE_PER_KPA, stiffness=0.02, damping=0.0008, mass=0.001,
    boxes=[((0.0, G_Y + 0.009, G_Z + 0.0012), (0.0025, 0.020, 0.0008), RED)],
)
_ticks = []
for _k in range(26):  # a tick every 10 kPa from 0 (pointing +y) round to 250, red from 0 to 10 (the zero band)
    _a = -_k * 10 * NEEDLE_PER_KPA
    _ticks.append(((-0.020 * math.sin(_a), G_Y + 0.020 * math.cos(_a), G_Z + 0.0003),
                   (0.0015, 0.004 if _k % 5 else 0.007, 0.0004), RED if _k <= 1 else DARK, _a))

# The bleed valve: a small quarter-turn valve on a tee on the pipe's top 14 cm downstream, venting through an elbow
# pointing away from the robots. Its blue lever points at the robots (-x) when shut; a quarter turn clockwise seen from
# above (to +y, along the pipe) opens it.
B_Y = 0.140
B_BODY_Z = PIPE_TOP + 0.016  # the bleed valve's body (a 16 mm cube) centre
B_LEVER_Z = B_BODY_Z + 0.008 + 0.006  # the lever's mid-plane, on a 4 mm stem
B_LEVER_L, B_LEVER_W, B_LEVER_T = 0.050, 0.010, 0.004
B_OPEN = -math.pi / 2
BLEED = Joint(
    "bleed", kind="hinge", axis="Z", anchor=(0.0, B_Y, B_LEVER_Z), lower=B_OPEN, upper=0.0, initial=0.0,
    stiffness=0.0, damping=0.05, target=0.0, mass=0.02,
    boxes=[((-B_LEVER_L / 2, B_Y, B_LEVER_Z), (B_LEVER_L, B_LEVER_W, B_LEVER_T), BLUE)],
    cylinders=[((0.0, B_Y, B_LEVER_Z), 0.007, B_LEVER_T + 0.002, STEEL)],
)

LINE = part(
    "gas_line", pos=(VX, VY, TOP), fixed=True, friction=0.8,
    boxes=[
        ((0.0, 0.0, PIPE_Z), (BODY, BODY, BODY), BRASS),
        *[((0.0, e * (PIPE_L / 2 - 0.02), (PIPE_Z - PIPE_R) / 2), (0.04, 0.03, PIPE_Z - PIPE_R), DARK) for e in (-1, 1)],
        ((-BODY / 2 + 0.001, 0.0, (PIPE_Z + BODY / 2 + TAB_Z0) / 2), (0.006, TAB, TAB_Z0 - PIPE_Z - BODY / 2), YELLOW),
        *_frame(TAB_X, 0.0, TAB_Z0 + TAB_T / 2, TAB_T, TAB, HOLE, YELLOW),
        ((0.0, B_Y, B_BODY_Z), (0.016, 0.016, 0.016), BRASS),  # bleed valve body
        ((0.020, B_Y, B_BODY_Z), (0.024, 0.008, 0.008), BRASS),  # its vent elbow, away from the robots
        ((0.033, B_Y, B_BODY_Z - 0.008), (0.008, 0.008, 0.016), BRASS),
        *_ticks,
    ],
    cylinders=[
        ((0.0, 0.0, PIPE_Z), PIPE_R, PIPE_L, YELLOW, "Y"),
        ((0.0, 0.0, (PIPE_Z + BODY / 2 + STEM_TOP) / 2), 0.007, STEM_TOP - PIPE_Z - BODY / 2, STEEL),
        ((0.0, B_Y, (PIPE_TOP + B_BODY_Z - 0.008) / 2 + 0.0005), 0.006, B_BODY_Z - 0.008 - PIPE_TOP + 0.002, BRASS),
        ((0.0, G_Y, (PIPE_TOP + G_Z - 0.008) / 2), 0.004, G_Z - 0.008 - PIPE_TOP, BRASS),  # the gauge's stem
        ((0.0, G_Y, G_Z - 0.004), G_R, 0.008, DARK),  # its case
        ((0.0, G_Y, G_Z + 0.0001), G_R - 0.002, 0.0002, WHITE),  # its face
    ],
    joints=[HANDLE, NEEDLE, BLEED],
)
HOLE_WORLD = np.array([VX + TAB_X, VY])  # both holes, the handle shut

# -- the padlock (loose) and its stand ----------------------------------------------------------------------------------
# Frame: origin at the middle of the body's top face. The shackle's legs lie along y, 24 mm apart: the long leg (always
# in the body) at y = -12 mm, the short leg (into its socket when shut) at y = +12 mm. Shut, the crossbar's underside is
# 17 mm above the body; open, the shackle is raised 28 mm (spring-loaded up).
LK_BODY = (0.020, 0.040, 0.035)  # x (thickness), y (width), z (height)
LEG_R, LEG_DY = 0.003, 0.012
INNER_H = 0.017  # shut: the crossbar's underside above the body's top
SOCKET = 0.008  # the short leg goes this far into the body, shut
OPEN_RAISE = 0.028
LATCH_AT = 0.0015  # pressed to within 1.5 mm of shut, it latches
_crossbar_z = INNER_H + LEG_R
SHACKLE = Joint(
    "shackle", kind="slide", axis="Z", anchor=(0.0, 0.0, 0.0), lower=0.0, upper=OPEN_RAISE, initial=OPEN_RAISE,
    stiffness=80.0, damping=1.0, target=OPEN_RAISE, mass=0.02,
    cylinders=[((0.0, -LEG_DY, (_crossbar_z - 0.030) / 2), LEG_R, _crossbar_z + 0.030, STEEL),  # long leg, deep in
               ((0.0, LEG_DY, (_crossbar_z - SOCKET) / 2), LEG_R, _crossbar_z + SOCKET, STEEL),  # short leg
               ((0.0, 0.0, _crossbar_z), LEG_R, 2 * LEG_DY + 2 * LEG_R, STEEL, "Y")],  # crossbar
)
STAND_XY, STAND_H = (0.17, -0.20), 0.050  # the padlock stands on a block 5 cm tall
PADLOCK = part(
    "padlock", pos=(*STAND_XY, TOP + STAND_H + LK_BODY[2]), mass=0.12, friction=0.9,
    boxes=[((0.0, 0.0, -LK_BODY[2] / 2), LK_BODY, STEEL),
           ((-LK_BODY[0] / 2 - 0.0005, 0.0, -LK_BODY[2] / 2 - 0.002), (0.001, LK_BODY[1] - 0.004, 0.022), RED),
           # the danger tag, drawn flat on the body's back (a red band over white): carried by a tie in a real lock
           ((LK_BODY[0] / 2 + 0.0005, 0.0, -LK_BODY[2] / 2), (0.001, 0.030, 0.012), RED)],
    joints=[SHACKLE],
)
STAND = part("padlock_stand", pos=(*STAND_XY, TOP), fixed=True, friction=0.9,
             boxes=[((0.0, 0.0, STAND_H / 2), (0.036, 0.056, STAND_H), DARK)])

OBJECTS = [LINE, STAND, PADLOCK]

# -- the gas, the gauge and the lock's latch ----------------------------------------------------------------------------
BLEED_K = 1.5  # 1/s: the trapped gas bleeds off through the fully open bleed valve (a few seconds to zero)
SHUT_TOL = math.radians(5.0)
OPEN_BLEED = math.radians(20.0)  # the bleed counts as open past 20 degrees
ZERO = 2.0  # kPa: the gauge reads zero (within its first division)
_st = {}


def _jid(asset, j):
    return [list(asset.joint_names).index(j.joint_name)]


@on_reset
def _reset(env):
    _st.clear()
    _st.update(p=P_LINE, vented_live=False, latched=False, armed=False, latch_t=None, bled_after=False, t=0.0, set_hard=False,
               settled=False, far=True, hist=[])


@on_step
def _gas(env):
    import torch

    if "p" not in _st:
        _reset(env)
    b = env.unwrapped
    dt = float(getattr(b, "step_dt", 0.05))
    _st["t"] += dt
    S = labkit._SPECS
    line = labkit._part_state(env, S["gas_line"])
    lock = labkit._part_state(env, S["padlock"])
    shut = line.joints["handle"] >= CLOSED - SHUT_TOL
    bleed = -line.joints["bleed"]
    f = float(np.clip(bleed / (math.pi / 2), 0.0, 1.0))
    if bleed > OPEN_BLEED and not shut:
        _st["vented_live"] = True  # the bleed opened on the live supply
    p = _st["p"]
    p = P_LINE if not shut else p - BLEED_K * f * p * dt
    _st["p"] = max(0.0, p)
    if bleed > OPEN_BLEED and _st["latched"]:
        _st["bled_after"] = True
    a = b.scene["gas_line"]
    a.set_joint_position_target(torch.tensor([[-_st["p"] * NEEDLE_PER_KPA]], device=a.device),
                                joint_ids=_jid(a, NEEDLE))
    # the latch: pressed shut, the shackle stays shut (a pawl): its spring's target goes to shut and its limit closes
    if not _st["latched"]:  # the shackle's spring holds it open (its drive target is not set at reset otherwise)
        pl = b.scene["padlock"]
        pl.set_joint_position_target(torch.full((1, 1), OPEN_RAISE, device=pl.device), joint_ids=_jid(pl, SHACKLE))
    if lock.joints["shackle"] >= OPEN_RAISE / 2:
        _st["armed"] = True  # the latch only catches a shackle pressed down from open (not its state before the reset)
    if not _st["latched"] and _st["armed"] and lock.joints["shackle"] <= LATCH_AT:
        _st["latched"] = True
    if _st["latched"] and not _st["set_hard"]:
        pl = b.scene["padlock"]
        ids = _jid(pl, SHACKLE)
        pl.set_joint_position_target(torch.zeros((1, 1), device=pl.device), joint_ids=ids)
        for name, val in (("write_joint_stiffness_to_sim", torch.full((1, 1), 20000.0, device=pl.device)),):
            fn = getattr(pl, name, None)
            if fn is not None:
                try:
                    fn(val, joint_ids=ids)
                except Exception:
                    pass
        lim = torch.tensor([[[0.0, 0.0005]]], device=pl.device)
        for name in ("write_joint_position_limit_to_sim", "write_joint_limits_to_sim"):
            fn = getattr(pl, name, None)
            if fn is not None:
                try:
                    fn(lim, joint_ids=ids)
                    break
                except Exception:
                    pass
        _st["set_hard"] = True
    tips = _tips(env)
    lever = line.to_world((-0.040 * math.cos(line.joints["bleed"]), B_Y - 0.040 * math.sin(line.joints["bleed"]),
                           B_LEVER_Z))
    _st["far"] = all(min(np.linalg.norm(np.asarray(t) - lock.to_world((0.0, 0.0, 0.0))),
                         np.linalg.norm(np.asarray(t) - lever)) >= HANDS_FAR for t in tips)
    h = _st["hist"]
    h.append(np.concatenate([np.asarray(lock.pos, float), [line.joints["handle"], line.joints["bleed"]]]))
    del h[:-(SETTLE_STEPS + 1)]
    _st["settled"] = len(h) > SETTLE_STEPS and float(np.max(np.abs(np.asarray(h) - h[-1]))) < SETTLE_TOL


def _tips(env):
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7"], preserve_order=True)[0]
        half = abs(float(robot.data.joint_pos.torch[0, ids].cpu().numpy()[0]))
        out += [mid + half * g[:3, 0], mid - half * g[:3, 0]]
    return out


def gauge_kpa() -> float:
    return float(_st.get("p", P_LINE))


# -- scoring ----------------------------------------------------------------------------------------------------------
HANDS_FAR = 0.08
SETTLE_STEPS, SETTLE_TOL = 20, 0.001
THROUGH_TOL = 0.005  # the short leg's axis within 5 mm of the holes' centre, where it crosses them
SUCCESS_HOLD_S = 1.0


def _shut(objects) -> bool:
    return objects["gas_line"].joints["handle"] >= CLOSED - SHUT_TOL


def locked_through(objects) -> bool:
    """Latched, with the short leg through both holes: its axis within 5 mm of the holes' centre at the tab's and the
    handle's heights, and the body below the tab."""
    lk = objects["padlock"]
    if not _st.get("latched"):
        return False
    q = lk.joints["shackle"]
    for z in (TOP + TAB_Z0 + TAB_T / 2, TOP + HANDLE_Z):
        # the short leg's axis point at height z (the leg runs along the lock's z axis)
        a, b = lk.to_world((0.0, LEG_DY, q - SOCKET)), lk.to_world((0.0, LEG_DY, q + INNER_H))
        if abs(b[2] - a[2]) < 1e-6:
            return False
        u = (z - a[2]) / (b[2] - a[2])
        if not 0.0 <= u <= 1.0:
            return False
        pt = a + u * (b - a)
        if math.hypot(pt[0] - HOLE_WORLD[0], pt[1] - HOLE_WORLD[1]) > THROUGH_TOL:
            return False
    return lk.to_world((0.0, 0.0, 0.0))[2] < TOP + TAB_Z0


def score(objects):
    """0 if the bleed valve was ever opened (past 20 degrees) while the isolating valve was open (venting the live
    supply), or the valves and padlock have not been still for the last second. Otherwise: 0.25 if the isolating
    valve is shut (its handle within 5 degrees of its stop); 0.5 if also locked with the padlock: latched, its short
    leg through both the handle's and the tab's holes; 0.75 if also the bleed was opened after the lock latched and
    the gauge reads zero (under 2 kPa); 1.0 if also every fingertip is 8 cm from the padlock and the bleed lever's end.
    0 if the isolating valve is not shut."""
    if _st.get("vented_live") or not _st.get("settled", False):
        return 0.0
    if not _shut(objects):
        return 0.0
    if not locked_through(objects):
        return 0.25
    if not (_st.get("bled_after") and gauge_kpa() < ZERO):
        return 0.5
    return 1.0 if _st.get("far", True) else 0.75


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return max(0.0, CLOSED - objects["gas_line"].joints["handle"])


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left hand: grips the valve handle's red sleeve from above and carries it round a quarter circle to its stop, lets go.
# Right hand: takes the open padlock off its stand by the ends of its body (in from the robots' side, level), brings it
# under the shut handle beside the tab, and slides it sideways so the handle and the tab go in between its raised
# shackle and its body, the short leg over the holes. Left hand presses the shackle's crossbar down until it latches;
# the right lets go. Then the left turns the bleed valve's lever a quarter turn open, holds it while the gauge falls to
# zero, lets go, and both hands go back. The demo reads only what a camera sees: the gauge's dial.


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot", False):
    def _grip(yaw=0.0):
        return yaw.rot if hasattr(yaw, "rot") else _prev_grip(yaw)

    _grip._rot = True
    _demo.grip_rotation = _grip

TIP_ABOVE_TCP = 0.013
GRIP_R = 0.105
SLEEVE_GRIP_Z = TOP + HANDLE_Z - SLEEVE_T / 2 + 0.003 + TIP_ABOVE_TCP
SLEEVE_HOVER_Z = TOP + HANDLE_Z + 0.06
FINGER_WIDE = -0.03
ARC_STEPS = 6
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.15, -0.25, TOP + 0.16)
# the right hand on the padlock: in from the robots' right (+y), tipped 40 degrees down, the fingers closing along x on
# the body's two broad faces (clear of the valve's handle and the tab, which are over the body's other half)
LK_TILT = math.radians(40.0)
A_LK = np.array([0.0, math.cos(LK_TILT), -math.sin(LK_TILT)])  # the approach
ROT_LK = np.column_stack([X, np.cross(A_LK, X), A_LK])  # columns: closing axis, the third axis, the approach
O_LK = Orient(ROT_LK)
LK_OPEN, LK_HOLD = -0.022, -0.0075  # (the body is 20 mm thick: a 2.5 mm squeeze each side)
LK_GRIP_DZ = -LK_BODY[2] / 2  # the fingertips at the body's mid-height
LK_TIP_DY = 0.004  # the fingertips 4 mm past the body's middle (from its -y end)
BODY_TOP_Z = TOP + TAB_Z0 - 0.002  # placed: the body's top 2 mm under the tab
L_FINAL = np.array([HOLE_WORLD[0], HOLE_WORLD[1] - LEG_DY, BODY_TOP_Z])  # the lock's frame: short leg over the holes
L_PRE = L_FINAL - [0.0, 0.045, 0.0]  # beside the tab, everything clear of it
L_START = np.array([*STAND_XY, TOP + STAND_H + LK_BODY[2]])
PRESS_DOWN = OPEN_RAISE + 0.002  # (pressed 2 mm past shut: against the stop)


def _lk_tcp(lock_origin, back=0.0):
    """The right hand's TCP holding the lock whose frame is at ``lock_origin`` (``back`` further back along its
    approach)."""
    tip = np.asarray(lock_origin, float) + [0.0, LK_TIP_DY, LK_GRIP_DZ]
    return tuple(tip - A_LK * (TIP_ABOVE_TCP + back))


def _grip_point(theta):
    return (VX - GRIP_R * math.sin(theta), VY + GRIP_R * math.cos(theta))


def _bleed_end(theta, r=0.040):
    """The bleed lever's grip point (``r`` out from its pivot) turned ``theta`` (0 shut, -pi/2 open)."""
    return (VX - r * math.cos(theta), VY + B_Y - r * math.sin(theta))


def _dur(wps):
    return sum(w.duration for w in wps)


def _pad(wps, t, pos, finger, label, yaw=0.0):
    """Hold ``pos`` until the list reaches time ``t``."""
    if t - _dur(wps) > 1e-6:
        wps.append(Waypoint(t - _dur(wps), pos, finger, label, yaw))


def demo():
    left, right = [], []
    # -- left: shut the valve ---------------------------------------------------------------------------------------
    p0, y0 = _grip_point(0.0), -math.pi / 2
    left += [Waypoint(2.0, (*p0, SLEEVE_HOVER_Z), FINGER_WIDE, "above the red sleeve", y0),
             Waypoint(1.5, (*p0, SLEEVE_GRIP_Z), FINGER_WIDE, "down round it", y0),
             Waypoint(0.8, (*p0, SLEEVE_GRIP_Z), FINGER_CLOSED, "grip", y0)]
    for i in range(1, ARC_STEPS + 1):
        th = CLOSED * i / ARC_STEPS
        left.append(Waypoint(0.9, (*_grip_point(th), SLEEVE_GRIP_Z), FINGER_CLOSED, "turn the handle shut", y0 + th))
    p1, y1 = _grip_point(CLOSED), y0 + CLOSED
    left += [Waypoint(0.8, (*p1, SLEEVE_GRIP_Z), FINGER_WIDE, "let go", y1),
             Waypoint(1.0, (*p1, SLEEVE_HOVER_Z), FINGER_WIDE, "up", y1)]
    t_shut = _dur(left)
    # -- right: take the padlock, bring it beside the tab -------------------------------------------------------------
    up = np.array([0.0, 0.0, 0.02])
    right += [Waypoint(2.0, tuple(np.array(_lk_tcp(L_START, 0.035)) + up), LK_OPEN, "hand tipped, behind the padlock",
                       O_LK),
              Waypoint(1.2, _lk_tcp(L_START, 0.035), LK_OPEN, "down beside it", O_LK),
              Waypoint(1.2, _lk_tcp(L_START), LK_OPEN, "in round the body's ends", O_LK),
              Waypoint(0.8, _lk_tcp(L_START), LK_HOLD, "hold the body", O_LK),
              Waypoint(1.0, tuple(np.array(_lk_tcp(L_START)) + up), LK_HOLD, "lift it off its stand", O_LK),
              Waypoint(2.0, tuple(np.array(_lk_tcp(L_PRE)) + [-0.03, 0.0, 0.0]), LK_HOLD, "towards the valve", O_LK)]
    _pad(right, t_shut + 0.5, tuple(np.array(_lk_tcp(L_PRE)) + [-0.03, 0.0, 0.0]), LK_HOLD,
         "wait for the valve to be shut", O_LK)
    right += [Waypoint(1.5, _lk_tcp(L_PRE), LK_HOLD, "beside the tab, under the handle's level", O_LK),
              Waypoint(0.8, _lk_tcp(L_PRE), LK_HOLD, "steady", O_LK),
              Waypoint(2.5, _lk_tcp(L_FINAL), LK_HOLD, "slide the handle and tab in: short leg over the holes", O_LK),
              Waypoint(1.0, _lk_tcp(L_FINAL), LK_HOLD, "steady", O_LK)]
    t_ready = _dur(right)
    # -- left: press the shackle shut ---------------------------------------------------------------------------------
    cb = L_FINAL + [0.0, 0.0, OPEN_RAISE + INNER_H + 2 * LEG_R]  # the crossbar's top, open
    press_xy = (cb[0], cb[1])
    _pad(left, t_ready - 2.0, (*p1, SLEEVE_HOVER_Z), FINGER_WIDE, "wait", y1)
    left += [Waypoint(2.0, (*press_xy, cb[2] + 0.008 + TIP_ABOVE_TCP), FINGER_CLOSED, "above the shackle", 0.0),
             Waypoint(1.0, (*press_xy, cb[2] + 0.002 + TIP_ABOVE_TCP), FINGER_CLOSED, "on its crossbar", 0.0),
             Waypoint(2.5, (*press_xy, cb[2] - PRESS_DOWN + TIP_ABOVE_TCP), FINGER_CLOSED,
                      "press it down through the holes until it latches", 0.0),
             Waypoint(0.8, (*press_xy, cb[2] - PRESS_DOWN + TIP_ABOVE_TCP), FINGER_CLOSED, "hold", 0.0),
             Waypoint(1.0, (*press_xy, cb[2] + 0.008 + TIP_ABOVE_TCP), FINGER_CLOSED, "up", 0.0)]
    t_latched = _dur(left)
    # -- right lets go and goes back; left opens the bleed ------------------------------------------------------------
    _pad(right, t_latched - 1.0, _lk_tcp(L_FINAL), LK_HOLD, "hold the lock while it is pressed shut", O_LK)
    right += [Waypoint(0.8, _lk_tcp(L_FINAL), LK_OPEN, "let go", O_LK),
              Waypoint(1.2, _lk_tcp(L_FINAL, 0.05), LK_OPEN, "back out", O_LK),
              Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "park")]
    lz_hover = TOP + B_LEVER_Z + 0.05 + TIP_ABOVE_TCP
    lz_grip = TOP + B_LEVER_Z + 0.001 + TIP_ABOVE_TCP - 0.004
    e0 = _bleed_end(0.0)
    left += [Waypoint(2.0, (*e0, lz_hover), FINGER_WIDE, "above the bleed lever's end", 0.0),
             Waypoint(1.2, (*e0, lz_grip), FINGER_WIDE, "down round it", 0.0),
             Waypoint(0.8, (*e0, lz_grip), FINGER_CLOSED, "grip", 0.0)]
    for i in range(1, 5):
        th = B_OPEN * i / 4
        left.append(Waypoint(0.8, (*_bleed_end(th), lz_grip), FINGER_CLOSED, "turn the bleed open", th))
    e1 = _bleed_end(B_OPEN)
    left += [Waypoint(4.0, (*e1, lz_grip), FINGER_CLOSED, "hold it open: the gauge falls to zero", B_OPEN),
             Waypoint(0.8, (*e1, lz_grip), FINGER_WIDE, "let go", B_OPEN),
             Waypoint(1.0, (*e1, lz_hover), FINGER_WIDE, "up", B_OPEN),
             Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "park")]
    t_end = max(_dur(left), _dur(right)) + 2.0
    _pad(left, t_end, LEFT_PARK, FINGER_OPEN, "rest")
    _pad(right, t_end, RIGHT_PARK, FINGER_OPEN, "rest")
    return {"left": left, "right": right}
