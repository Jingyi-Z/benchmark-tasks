"""Pop a phone's SIM tray out with its eject pin and stand the pin back in its holder (yam_arena task file, built
with labkit).

A phone the size of an iPhone 12 stands in a loose desk dock with its SIM tray flush in its left edge; an eject tool
stands pin-down in a foam block beside it. The tray is latched by a sprung pawl that only the pin, pushed through the
1 mm hole in the tray's cap, can lift; a sprung ejector then pops the tray a few millimetres out. The tool then goes
back in its block. Scoring in score(); the scripted demo reads only what a camera sees (part poses and fingertips)."""
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
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
X, Y, Z = np.eye(3)
BLACK = (0.12, 0.12, 0.13)

# Phone (iPhone 12 size: 146.7 x 71.5 x 7.4 mm, 164 g), standing upright in a desk dock's groove, screen towards the
# robot. Part frame (yaw -90 deg on the counter): u = part x across its width (world -y), v = part y through its
# thickness (world +x), z up from its bottom edge. Its nano-SIM tray sits in its left-hand edge, as on that phone
# (u = EDGE, world +y, facing the left arm) and slides out along -u. The eject hole goes through the tray's own end cap near its top end. Inside (estimates): a sprung pawl drops
# its nose into a notch on the tray's top edge and holds it; a pin pushed through the hole meets the pawl's 45 degree
# underside and lifts it, and a sprung ejector plate behind the tray then pushes it 3 mm out.
PH_T, PH_W, PH_H = 0.0074, 0.0715, 0.1467
EDGE = -PH_W / 2
WALL = 0.0015  # the frame wall at the tray edge, and the cap's depth
CAV = (0.020, 0.0025, 0.050, 0.074)  # the cavity behind it: depth (u), half-thickness (v), z from, z to
TRAY_Z = 0.060  # the cap's centre height (phone frame)
CAP_HV, CAP_HZ = 0.00165, 0.0077  # tray end cap half-sizes (v, z)
BODY_Z, BODY_HZ, BODY_HV, BODY_L = 0.0584, 0.0056, 0.0006, 0.014  # the tray's body behind the cap (centre z, ...)
NOTCH = (0.0055, 0.0067, 0.0010)  # the notch on the body's top edge: u' from, to (tray frame), depth
HOLE_Z, HOLE_H, CHAMFER = 0.0655, 0.0005, 0.0003  # eject hole through the cap: centre z, half-size, mouth chamfer
PAWL_LIFT = 0.0025
EJECT = 0.003  # the ejector's stroke (how far the tray pops)
PHONE_BLACK, FRAME_ALU, TRAY_STEEL, STAND_GREY = (0.08, 0.08, 0.09), (0.70, 0.71, 0.73), (0.62, 0.63, 0.66), (0.30, 0.31, 0.33)
_e, _cu, _cv, _cz0, _cz1 = EDGE, CAV[0], CAV[1], CAV[2], CAV[3]


def _b(u0, u1, v0, v1, z0, z1, c=PHONE_BLACK):
    return (((u0 + u1) / 2, (v0 + v1) / 2, (z0 + z1) / 2), (u1 - u0, v1 - v0, z1 - z0), c)


_PHONE = [_b(_e + _cu, -_e, -PH_T / 2, PH_T / 2, 0.0, PH_H),  # everything beyond the cavity
          _b(_e, _e + _cu, -PH_T / 2, PH_T / 2, 0.0, _cz0),  # below it
          _b(_e, _e + _cu, -PH_T / 2, PH_T / 2, _cz1, PH_H),  # above it
          _b(_e, _e + _cu, -PH_T / 2, -_cv, _cz0, _cz1),  # glass faces over it
          _b(_e, _e + _cu, _cv, PH_T / 2, _cz0, _cz1)]
_cap0, _cap1 = TRAY_Z - CAP_HZ - 0.00005, TRAY_Z + CAP_HZ + 0.00005
_w1 = _e + WALL
_PHONE += [_b(_e, _w1, -_cv, _cv, _cz0, _cap0, FRAME_ALU),  # the edge wall round the cap's opening
           _b(_e, _w1, -_cv, -CAP_HV - 0.00005, _cap0, _cap1, FRAME_ALU),
           _b(_e, _w1, CAP_HV + 0.00005, _cv, _cap0, _cap1, FRAME_ALU),
           _b(_e, _w1, -_cv, _cv, _cap1, _cz1, FRAME_ALU)]
_bz0, _bz1 = BODY_Z - BODY_HZ, BODY_Z + BODY_HZ
_PHONE += [_b(_w1, _e + _cu, -_cv, _cv, _cz0, _bz0 - 0.0001),  # tray guides: floor, top (clear of the pawl), sides
           _b(_e + 0.0070, _e + _cu, -_cv, _cv, _bz1 + 0.0001, _bz1 + 0.0008),
           _b(_w1, _e + _cu, -_cv, -BODY_HV - 0.0002, _bz0 - 0.0001, _bz1 + 0.0001),
           _b(_w1, _e + _cu, BODY_HV + 0.0002, _cv, _bz0 - 0.0001, _bz1 + 0.0001)]
# the pawl: its nose sits in the tray's notch; above it a plate tilted 30 deg puts a sloped underside across the
# hole's line, about 2 mm behind the cap (a 30 deg slope: the pin lifts the pawl 0.58 mm per mm it goes in)
RAMP_DEG = 30.0
_RAMP_U = _e + 0.0034  # where the slope crosses the hole's centre line
_rl, _rh, _ra = 0.0034, 0.0015, math.radians(RAMP_DEG)
_rn = np.array([-math.sin(_ra), -math.cos(_ra)])  # the underside's normal (u, z)
_pc = np.array([_RAMP_U, HOLE_Z]) - (_rh / 2) * _rn
_rx = np.array([math.cos(_ra), -math.sin(_ra)])
_corners = [_pc + a * (_rl / 2) * _rx + b * (_rh / 2) * (-_rn) for a in (-1, 1) for b in (-1, 1)]
assert min(c[0] for c in _corners) > _e + WALL + 0.0003, "pawl ramp too close to the tray's cap"
_NOSE_Z0 = _bz1 - NOTCH[2] + 0.0001
_PAWL_BOXES = [((_e + 0.0061, 0.0, (_NOSE_Z0 + _bz1 + 0.001) / 2), (0.0010, 0.0012, _bz1 + 0.001 - _NOSE_Z0), FRAME_ALU),
               ((_pc[0], 0.0, _pc[1]), (_rl, 0.0012, _rh), FRAME_ALU, (0.0, _ra)),
               ((_e + 0.0053, 0.0, 0.0672), (0.0026, 0.0012, 0.0012), FRAME_ALU),  # bridge over to the nose, above the pin
               ((_e + 0.0061, 0.0, 0.0661), (0.0010, 0.0012, 0.0022), FRAME_ALU)]
PAWL = Joint("pawl", kind="slide", axis="Z", anchor=(_e + 0.0036, 0.0, HOLE_Z), boxes=_PAWL_BOXES, lower=0.0,
             upper=PAWL_LIFT, stiffness=80.0, damping=0.05, target=0.0, initial=0.0, mass=0.001)
_EJ_U = _e + WALL + BODY_L + 0.0001
EJECTOR = Joint("ejector", kind="slide", axis="X", anchor=(_EJ_U + 0.001, 0.0, BODY_Z),
                boxes=[((_EJ_U + 0.001, 0.0, BODY_Z), (0.002, 0.0012, 0.010), FRAME_ALU)],
                lower=-EJECT, upper=0.0, stiffness=150.0, damping=0.3, target=-EJECT, initial=0.0, mass=0.001)

STAND_XY = np.array([0.38, 0.0])
FLOOR = 0.004  # the dock's groove floor above the counter
PH_YAW = -math.pi / 2
PHONE = part("phone", pos=(STAND_XY[0], STAND_XY[1], TOP + FLOOR), yaw=PH_YAW, mass=0.164, friction=0.5,
             boxes=_PHONE, joints=[PAWL, EJECTOR])

# Tray (steel, 4 g): frame origin at its end cap's outer face, at the cap's centre; u' along the phone's u.
_n0, _n1, _nd = NOTCH
_hz = HOLE_Z - TRAY_Z  # the hole's height on the cap
_bzc = BODY_Z - TRAY_Z
_TRAY = []
for _u0, _u1, _a in ((0.0, CHAMFER, HOLE_H + CHAMFER), (CHAMFER, WALL, HOLE_H)):  # the cap: a 1.6 mm mouth, then 1.0 mm
    _du, _uc = _u1 - _u0, (_u0 + _u1) / 2
    _TRAY += [((_uc, 0.0, (-CAP_HZ + _hz - _a) / 2), (_du, 2 * CAP_HV, CAP_HZ + _hz - _a), TRAY_STEEL),
              ((_uc, 0.0, (CAP_HZ + _hz + _a) / 2), (_du, 2 * CAP_HV, CAP_HZ - _hz - _a), TRAY_STEEL),
              ((_uc, (CAP_HV + _a) / 2, _hz), (_du, CAP_HV - _a, 2 * _a), TRAY_STEEL),
              ((_uc, -(CAP_HV + _a) / 2, _hz), (_du, CAP_HV - _a, 2 * _a), TRAY_STEEL)]
# the hole's mouth: a 0.3 mm 45 degree chamfer on each of its four sides (square bars turned 45 deg)
_cs, _cc = CHAMFER * math.sqrt(2), (CHAMFER, HOLE_H + CHAMFER)
_TRAY += [((_cc[0], 0.0, _hz + s * _cc[1]), (_cs, 2 * HOLE_H, _cs), TRAY_STEEL, (0.0, math.radians(45.0))) for s in (1, -1)]
_TRAY += [((_cc[0], s * _cc[1], _hz), (_cs, _cs, 2 * HOLE_H), TRAY_STEEL, math.radians(45.0)) for s in (1, -1)]
_TRAY += [((WALL + BODY_L / 2, 0.0, _bzc - _nd / 2), (BODY_L, 2 * BODY_HV, 2 * BODY_HZ - _nd), TRAY_STEEL),
          (((WALL + _n0) / 2, 0.0, _bzc + BODY_HZ - _nd / 2), (_n0 - WALL, 2 * BODY_HV, _nd), TRAY_STEEL),
          (((_n1 + WALL + BODY_L) / 2, 0.0, _bzc + BODY_HZ - _nd / 2), (WALL + BODY_L - _n1, 2 * BODY_HV, _nd), TRAY_STEEL)]
TRAY = part("tray", pos=(STAND_XY[0], STAND_XY[1] - EDGE, TOP + FLOOR + TRAY_Z), yaw=PH_YAW, mass=0.004,
            friction=0.25, boxes=_TRAY)

# Desk dock (loose, weighted, 300 g): a base plate, a 10 mm front lip and a 70 mm back rest either side of a groove
# along y, 0.6 mm wider than the phone each side.
_g = PH_T / 2 + 0.0006
STAND = part("stand", pos=(STAND_XY[0], STAND_XY[1], TOP), mass=0.300, friction=0.6, color=STAND_GREY,
             boxes=[((0.0, 0.0, FLOOR / 2), (0.070, 0.110, FLOOR), STAND_GREY),
                    ((-_g - 0.004, 0.0, FLOOR + 0.005), (0.008, 0.110, 0.010), STAND_GREY),
                    ((_g + 0.004, 0.0, FLOOR + 0.035), (0.008, 0.050, 0.070), STAND_GREY)])

# Eject tool (steel, 3 g): frame origin at the pin's tip, z up its axis; an 18 x 10 x 0.8 mm paddle (width along
# its x) on a 0.8 mm round pin 8 mm long. It stands pin-down in a slit in a loose foam block.
PIN_L, PAD_L, PAD_W, PAD_T, PIN_R = 0.008, 0.018, 0.010, 0.0008, 0.0004
TOOL_XY = np.array([0.30, 0.15])
BLOCK_H, SLIT_D = 0.020, 0.006
TOOL = part("tool", pos=(TOOL_XY[0], TOOL_XY[1], TOP + BLOCK_H - SLIT_D - PIN_L + 0.0002), yaw=-math.pi / 2,
            mass=0.003, friction=1.0,
            cylinders=[((0.0, 0.0, PIN_L / 2), PIN_R, PIN_L, TRAY_STEEL, "Z")],
            boxes=[((0.0, 0.0, PIN_L + PAD_L / 2), (PAD_W, PAD_T, PAD_L), TRAY_STEEL)])
# foam block 40 x 40 x 20 mm: a 2.0 x 13 mm slit 6 mm deep for the paddle, a 2 mm square hole below it for the pin
_BLK = [((0.0, 0.0, (BLOCK_H - SLIT_D - PIN_L) / 2), (0.040, 0.040, BLOCK_H - SLIT_D - PIN_L), BLACK)]
for z0, z1, hx, hy in ((BLOCK_H - SLIT_D - PIN_L, BLOCK_H - SLIT_D, 0.001, 0.001), (BLOCK_H - SLIT_D, BLOCK_H, 0.001, 0.0065)):
    zc, dz = (z0 + z1) / 2, z1 - z0
    _BLK += [(((0.020 + hx) / 2, 0.0, zc), (0.020 - hx, 0.040, dz), BLACK),
             ((-(0.020 + hx) / 2, 0.0, zc), (0.020 - hx, 0.040, dz), BLACK),
             ((0.0, (0.020 + hy) / 2, zc), (2 * hx, 0.020 - hy, dz), BLACK),
             ((0.0, -(0.020 + hy) / 2, zc), (2 * hx, 0.020 - hy, dz), BLACK)]
BLOCK = part("block", pos=(TOOL_XY[0], TOOL_XY[1], TOP), mass=0.030, friction=0.8, color=BLACK, boxes=_BLK)
OBJECTS = [STAND, PHONE, TRAY, BLOCK, TOOL]
ARTICULATED = ["phone"]


@on_step
def _spring_targets(env):
    # labkit leaves Isaac Lab's joint position targets at 0; aim each spring at its joint's default (its target).
    for n in ARTICULATED:
        art = env.unwrapped.scene[n]
        art.set_joint_position_target(art.data.default_joint_pos)


def score(objects):
    return 0.0


def success(objects):
    return False


def distance(objects):
    return 0.0


from yam_arena import ik as _ik  # noqa: E402
from yam_arena.constants import READY as _READY  # noqa: E402


def _tips(env):
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        j7, j8 = (abs(float(v)) for v in robot.data.joint_pos.torch[0, ids].cpu().numpy())
        out += [mid - j7 * g[:3, 0], mid + j8 * g[:3, 0]]  # (joint7 moves towards -x of the gripper, joint8 towards +x)
    return out


def tilt_deg(R, down=False):
    c = float(R[2, 2]) * (-1.0 if down else 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))

def _ik_err(pe, re):
    return pe * 1000.0 + math.degrees(re)


FREE_MOVES = ("wrist turned", "hand turned", "up and clear")


def _endpoints(plan):
    """Joint angles for every waypoint's end, chained: warm-started from the previous one, and, where that solve misses,
    the closest of several seeded solves that reach it (so the wrist turns the way the waypoints mean)."""
    rng = np.random.default_rng(7)
    seeds = [np.array(_READY, float)] + [np.array(_READY, float) + rng.normal(0.0, 0.8, 6) for _ in range(10)]
    q_prev, out, switch, forced = plan.q.copy(), [], set(), set()
    for i, w in enumerate(plan.waypoints):
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, q_prev, base_pos=plan.base_pos, iters=300)
        if any(m in w.label for m in FREE_MOVES):
            forced.add(i)  # a free-space move: always a joint-space blend
        if _ik_err(pe, re) > 1.5:
            switch.add(i)
            best = (_ik_err(pe, re), 0.0, q)
            for s in [q_prev + rng.normal(0.0, 0.3, 6) for _ in range(4)] + seeds:
                qs, ps, rs = _ik.solve(pos, quat, s, base_pos=plan.base_pos, iters=200)
                e, d = _ik_err(ps, rs), float(np.max(np.abs(qs - q_prev)))
                if (e < 1.0 and (best[0] >= 1.0 or d < best[1])) or (best[0] >= 1.0 and e < best[0]):
                    best = (e, d, qs)
            q = best[2]
        elif (float(np.max(np.abs(q[3:5] - q_prev[3:5]))) > math.radians(70.0)
              or float(np.max(np.abs(q - q_prev))) > math.radians(100.0)):
            switch.add(i)  # reached, but by turning the wrist over
        out.append(q)
        q_prev = q
    # then backwards: where the arm changes its wrist's branch between two waypoints, move the change earlier if the
    # earlier waypoint can be reached in the later branch (so it happens in free space, before an approach)
    for i in range(len(out) - 1, 0, -1):
        if i not in switch or i in forced:
            continue
        w = plan.waypoints[i - 1]
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, out[i], base_pos=plan.base_pos, iters=300)
        if (_ik_err(pe, re) < 1.5 and float(np.max(np.abs(q - out[i])))
                < float(np.max(np.abs(out[i - 1] - out[i]))) - math.radians(10.0)):
            out[i - 1] = q
            switch.discard(i)
            switch.add(i - 1)
    return out, switch | forced


def _seg(plan, t):
    acc = 0.0
    for i, w in enumerate(plan.waypoints):
        if t < acc + w.duration:
            return i, _ik.smoothstep((t - acc) / w.duration)
        acc += w.duration
    return len(plan.waypoints) - 1, 1.0


def _action7(self, t):
    """[j1..j6, grip] at time t: IK warm-started from the previous step, or, where that misses, from the joint-space
    blend of the segment's end solutions. It also sets the plan clock the trims run on."""
    _demo._plan_clock = float(t)
    if getattr(self, "_qends", None) is None:
        self._qends, self._switch = _endpoints(self)
    pos, rot, finger, label = self.target(t)
    quat = _ik.matrix_to_quat_xyzw(rot)
    i, s = _seg(self, t)
    q0 = self._qends[i - 1] if i > 0 else self._q_start
    seed = q0 + s * (self._qends[i] - q0)
    if i in self._switch:
        # the wrist changes branch on this (free-space) segment: follow the joint-space blend
        self.q = seed
        grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
        return np.r_[seed, grip], label
    q, pe, re = _ik.solve(pos, quat, self.q, base_pos=self.base_pos, iters=80)
    if _ik_err(pe, re) > 2.0:
        q2, p2, r2 = _ik.solve(pos, quat, seed, base_pos=self.base_pos, iters=80)
        if _ik_err(p2, r2) < _ik_err(pe, re) - 1.0:
            q, pe, re = q2, p2, r2
    self.q = q
    self.worst_ik_mm = max(self.worst_ik_mm, pe * 1000)
    self.worst_ik_deg = max(self.worst_ik_deg, float(np.degrees(re)))
    grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0)) if FINGER_OPEN != FINGER_CLOSED else 1.0
    return np.r_[q, grip], label


_prev_init = _demo.ArmPlan.__init__
if not getattr(_prev_init, "_fa", False):
    def _init(self, *a, _prev=_prev_init, **k):
        _prev(self, *a, **k)
        self._q_start, self._qends = self.q.copy(), None

    _init._fa = True
    _demo.ArmPlan.__init__ = _init
    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


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

class _Plan:
    """Two arms' waypoints, built in phases that start together; the clock is the phases' running time."""

    def __init__(self):
        self.L, self.R = [], []
        self.t = 0.0

    def hold(self, plan, dur, label="hold"):
        w = plan[-1]
        plan.append(Waypoint(dur, w.pos, w.finger, label, w.yaw))

    def phase(self, right=(), left=()):
        """Each list: (duration, pos, finger, label, yaw) tuples. The shorter one is padded with a hold."""
        tr, tl = sum(w[0] for w in right), sum(w[0] for w in left)
        dur = max(tr, tl)
        for plan, ws, tot in ((self.R, right, tr), (self.L, left, tl)):
            for w in ws:
                plan.append(Waypoint(*w))
            if dur - tot > 1e-9:
                self.hold(plan, dur - tot)
        t0 = self.t
        self.t += dur
        return t0

    def window(self, t0, t1, key, fn, watch):
        _ct["wins"].append((t0, t1, key, fn, watch))

    def gate(self, t0, t1, key, cond):
        _ct["gates"].append((t0, t1, key, cond))


# trims: each a 3-vector added to the TCPs of _Dyn waypoints with its key; integrated in windows from seen positions
_ct = {"trim": {}, "wins": [], "last": {}, "gates": [], "gate": {}}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}
    _ct["gate"] = {}


@on_step
def _gtrack(env):
    """Gates: from t0 on, latch ``key`` once ``cond`` sees what it waits for (at t1 at the latest)."""
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, cond in _ct["gates"]:
        if t0 <= t and not _ct["gate"].get(key) and (t >= t1 or cond(env)):
            _ct["gate"][key] = t


class _Gate:
    """A TCP that is ``a`` until the gate ``key`` latches, then ``b``."""

    def __init__(self, a, b, key):
        self.a, self.b, self.key = np.asarray(a, float), np.asarray(b, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.b if _ct["gate"].get(self.key) else self.a, dtype=dtype or float)


def _seen(env, name):
    return np.asarray(labkit._part_state(env, labkit._SPECS[name]).pos, float)


@on_step
def _ctrack(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, fn, watch in _ct["wins"]:
        if not t0 <= t < t1:
            continue
        p = _seen(env, watch)
        last = _ct["last"].get(key)
        _ct["last"][key] = p
        if last is None or np.max(np.abs(p - last)) > 0.0005:
            continue  # only while what is watched holds still
        e = np.asarray(fn(env), float)
        tr = _ct["trim"].get(key, np.zeros(3))
        step, cap = TRIM_LIMITS.get(key, (0.0003, 0.020))
        _ct["trim"][key] = np.clip(tr + np.clip(TRIM_GAIN.get(key, 0.2) * e, -step, step), -cap, cap)


class _Dyn:
    """A TCP plus the live trim ``key``."""

    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)




def _frame(d, c):
    """Gripper rotation pointing along d with the fingers closing along c (made square to d)."""
    z = np.asarray(d, float) / np.linalg.norm(d)
    x = np.asarray(c, float) - np.dot(c, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


R0 = _prev_grip(0.0)
RIGHT_PARK = np.array([0.15, -0.32, TOP + 0.20])
LEFT_PARK = np.array([0.15, 0.32, TOP + 0.20])
TRIM_LIMITS = {"aim": (0.0002, 0.015), "pick": (0.0005, 0.010), "ret": (0.0003, 0.010)}
TRIM_GAIN = {"aim": 0.04, "ret": 0.15}  # slow: the arm lags a few steps, and a faster aim loop rings by about 1 mm
R_DOWN = _frame(-Z, -X)  # (left hand) pointing down, closing along x (across the paddle's thickness)
R_PIN = _frame(-Y, -X)  # (left hand) pointing -y (into the tray edge), closing along x
R_BRACE = _frame(Y, -Z)  # (right hand) pointing +y onto the phone's far edge
R_PULL = _frame(X, -Z)  # (left hand) pointing +x from the screen side, closing along z (onto the cap's two ends)
TOOL_TOP = TOP + BLOCK_H - SLIT_D - PIN_L + 0.0002 + PIN_L + PAD_L  # the standing tool's paddle top
GRIP_DOWN = 0.006  # the pad tips this far below the paddle's top
PIN_REACH = PIN_L + PAD_L - GRIP_DOWN  # the pin's tip this far beyond the pad tips, along the hand
HOLE_W = np.array([STAND_XY[0], STAND_XY[1] - EDGE, TOP + FLOOR + HOLE_Z])  # the hole's mouth, as set up
EDGE_FAR = STAND_XY[1] + EDGE  # the phone's far (-y) edge
DEPTH = 0.0050  # the pin's tip pushed this far past the hole's mouth (the slope is met at 2.6 mm; 1.4 mm of lift)
PULL_OPEN = -0.011  # 22 mm between the pads, round the 15.4 mm cap
POP = EJECT  # the tray's cap stands this far out once popped


def _pick(dz=0.0):
    tip = np.array([TOOL_XY[0], TOOL_XY[1], TOOL_TOP - GRIP_DOWN + dz])
    return tip - TIP_ABOVE_TCP * (-Z)


def _pin(dy, dx=0.0, dz=0.0):
    """TCP with the held pin's tip at the hole's mouth moved dy in along -y (negative: outside)."""
    tip = HOLE_W + np.array([dx, -dy, dz])
    return tip + (PIN_REACH + TIP_ABOVE_TCP) * Y


def _brace(dy=0.0):
    """Left: closed fingertips 0.5 mm off the phone's far edge at the hole's height (the push meets them)."""
    tip = np.array([STAND_XY[0], EDGE_FAR - 0.0005 - dy, TOP + FLOOR + HOLE_Z])
    return tip - TIP_ABOVE_TCP * Y


def _pull(dy=0.0, dx=0.0, dz=0.0):
    """Right: pads round the popped cap's two ends, coming in from the screen side (cap 4 to 8 mm past the TCP)."""
    c = np.array([STAND_XY[0] - 0.006 + dx, STAND_XY[1] - EDGE + POP - 0.0002 - dy, TOP + FLOOR + TRAY_Z + dz])
    return c


def _tip(env):
    return np.asarray(labkit._part_state(env, TOOL).to_world((0.0, 0.0, 0.0)), float)


def _hole(env):
    return np.asarray(labkit._part_state(env, TRAY).to_world((0.0, 0.0, HOLE_Z - TRAY_Z)), float)


def _aim_err(standoff):
    def f(env):
        return _hole(env) + np.array([0.0, standoff, 0.0]) - _tip(env)
    return f


def _ret_err(env):
    """Seen offset (x, y) of the foam block's pin hole (its centre) from the held tool's pin tip."""
    blk = np.asarray(labkit._part_state(env, BLOCK).pos, float)
    e = blk - _tip(env)
    return np.array([e[0], e[1], 0.0])


def _cap_err(env):
    """Seen offset of the popped cap's middle from where the pull pinch was planned."""
    c = np.asarray(labkit._part_state(env, TRAY).to_world((0.00075, 0.0, 0.0)), float)
    plan = np.array([STAND_XY[0], STAND_XY[1] - EDGE + POP - 0.00075, TOP + FLOOR + TRAY_Z])
    return c - plan - _ct["trim"].get("pick", np.zeros(3))  # (the trim settles on the seen offset)


def demo():
    P = _Plan()
    rd, rp, lb, rq, lp, rr = Orient(R_DOWN), Orient(R_PIN), Orient(R_BRACE), Orient(R_PULL), Orient(R0), Orient(R0)
    # 1. the left hand takes the eject tool by its paddle out of the foam block and lifts it; the right goes to the
    #    phone's far edge
    P.phase(left=[(2.5, tuple(_pick(0.05)), FINGER_OPEN * 0.4, "above the tool", rd),
                   (1.5, tuple(_pick()), FINGER_OPEN * 0.4, "down round the paddle", rd),
                   (1.0, tuple(_pick()), FINGER_CLOSED, "pinch the paddle", rd),
                   (1.5, tuple(_pick(0.06)), FINGER_CLOSED, "lift it out", rd)],
            right=[(3.0, tuple(_brace(0.04)), FINGER_CLOSED, "out beside the phone's far edge", lb),
                  (2.0, tuple(_brace()), FINGER_CLOSED, "fingertips at the far edge", lb)])
    # 2. turn the tool's pin to point along -y and bring it 10 mm outside the hole
    k = "aim"
    P.phase(left=[(2.5, _Dyn(_pin(-0.030, 0.0, 0.03), k), FINGER_CLOSED, "wrist turned: pin level, facing the edge", rp),
                   (1.5, _Dyn(_pin(-0.010), k), FINGER_CLOSED, "pin 10 mm outside the hole", rp)],
            right=[(0.1, tuple(_brace()), FINGER_CLOSED, "hold at the edge", lb)])
    t_a = P.t
    # 3. line the pin up on the hole as seen (10 mm out, then 1 mm out), push it in: the tray pops; draw the pin out
    P.phase(left=[(3.5, _Dyn(_pin(-0.010), k), FINGER_CLOSED, "hold still: line the pin up on the hole, as seen", rp),
                   (1.0, _Dyn(_pin(-0.001), k), FINGER_CLOSED, "up to the hole's mouth", rp),
                   (3.5, _Dyn(_pin(-0.001), k), FINGER_CLOSED, "hold still: line up again, as seen", rp),
                   (2.0, _Dyn(_pin(DEPTH), k), FINGER_CLOSED, "push the pin in: the tray pops", rp),
                   (1.0, _Dyn(_pin(DEPTH), k), FINGER_CLOSED, "hold", rp),
                   (1.5, _Dyn(_pin(-0.015), k), FINGER_CLOSED, "draw the pin out", rp)],
            right=[(0.1, tuple(_brace()), FINGER_CLOSED, "brace the far edge", lb)])
    P.window(t_a + 0.3, t_a + 3.5, k, _aim_err(0.010), "tool")
    P.window(t_a + 4.8, t_a + 8.0, k, _aim_err(0.001), "tool")
    # 4. stand the tool back pin-down in its foam block, lined up on the seen slit just above it; the right hand
    #    leaves the edge
    kr = "ret"
    P.phase(left=[(2.5, _Dyn(_pick(0.04), kr), FINGER_CLOSED, "wrist turned: the tool upright over its block", rd),
                  (1.0, _Dyn(_pick(0.020), kr), FINGER_CLOSED, "down to just above the slit", rd)],
            right=[(1.5, tuple(_brace(0.04)), FINGER_CLOSED, "off the edge", lb),
                   (2.0, tuple(RIGHT_PARK), FINGER_OPEN, "away, to park", lp)])
    t_r = P.t
    P.phase(left=[(3.5, _Dyn(_pick(0.020), kr), FINGER_CLOSED, "hold still: line the pin up over the slit, as seen", rd),
                  (2.0, _Dyn(_pick(0.001), kr), FINGER_CLOSED, "lower it into the slit", rd),
                  (0.6, _Dyn(_pick(0.001), kr), FINGER_OPEN * 0.4, "let go", rd),
                  (1.0, _Dyn(_pick(0.06), kr), FINGER_OPEN * 0.4, "up and clear", rd)])
    P.window(t_r + 0.3, t_r + 3.5, kr, _ret_err, "tool")
    # 5. the left hand parks too
    P.phase(left=[(2.5, tuple(LEFT_PARK), FINGER_OPEN, "away, to park", lp)])
    P.phase(right=[(3.0, tuple(RIGHT_PARK), FINGER_OPEN, "parked", lp)],
            left=[(3.0, tuple(LEFT_PARK), FINGER_OPEN, "parked", rr)])
    return {"left": P.L, "right": P.R}



# -- scoring ----------------------------------------------------------------------------------------------------------
# Final state only. 0.5: the tray popped (its cap at least 2 mm out of the phone's edge); 1.0: also the eject tool
# standing back pin-down in its foam block. Every tier needs the phone standing in its dock (tilt under 10 degrees, its
# bottom within 3 mm of the groove floor and its middle within 3 mm of the groove's), all four fingertips at least
# 80 mm from the phone, tray, tool and dock, and the phone and tray still (each tracked corner moved under 1 mm over
# the last 21 samples).
STILL_STEPS, STILL_TOL = 21, 0.001
RELEASE = 0.080
SLOT_LEN = WALL + BODY_L  # the tray's full length
POPPED = 0.002  # popped: the tray's cap face at least this far out of the phone's edge
_PH_BOX = [(np.array([0.0, 0.0, PH_H / 2]), np.array([PH_W / 2, PH_T / 2, PH_H / 2]))]
_TR_BOX = [(np.array([SLOT_LEN / 2, 0.0, 0.0]), np.array([SLOT_LEN / 2, CAP_HV, CAP_HZ]))]
_ST_BOX = [(np.array([0.0, 0.0, 0.037]), np.array([0.035, 0.055, 0.037]))]
_TL_BOX = [(np.array([0.0, 0.0, (PIN_L + PAD_L) / 2]), np.array([PAD_W / 2, PAD_T / 2, (PIN_L + PAD_L) / 2]))]
_PH_PTS = [np.array([u, 0.0, z]) for u in (-PH_W / 2, PH_W / 2) for z in (0.0, PH_H)]
_TR_PTS = [np.array([u, 0.0, z]) for u in (0.0, SLOT_LEN) for z in (-CAP_HZ, CAP_HZ)]


def _box_dist(q, c, h):
    d = np.maximum(np.abs(np.asarray(q, float) - c) - h, 0.0)
    return float(np.linalg.norm(d))


def _clearance(parts, tips):
    best = float("inf")
    for p, boxes in parts:
        for t in tips:
            q = np.asarray(p.to_local(np.asarray(t, float)), float)
            best = min(best, min(_box_dist(q, c, h) for c, h in boxes))
    return best


def _popped(phone, tray):
    """How far (m) the tray's cap face stands out of the phone's edge, along the tray's slot."""
    return -(float(np.asarray(phone.to_local(tray.pos), float)[0]) - EDGE)


TOOL_HOME_XY, TOOL_HOME_Z = 0.003, 0.004  # the tool back in its block: pin tip this near the hole's centre and floor


def _tool_home(tool, block):
    """The eject tool stands back pin-down in its foam block (pin tip near the bottom of the block's pin hole)."""
    q = np.asarray(block.to_local(tool.to_world((0.0, 0.0, 0.0))), float)
    floor = BLOCK_H - SLIT_D - PIN_L
    return math.hypot(q[0], q[1]) <= TOOL_HOME_XY and abs(q[2] - floor) <= TOOL_HOME_Z and tool.tilt_deg() < 15.0


def _docked(phone, stand):
    q = np.asarray(stand.to_local(phone.pos), float)
    return phone.tilt_deg() < 10.0 and abs(q[2] - FLOOR) < 0.003 and abs(q[0]) < 0.003


_w = {"hist": [], "still": False, "far": False}


@on_reset
def _w_reset(env):
    _w["hist"], _w["still"], _w["far"] = [], False, False


@on_step
def _watch(env):
    ph, tr = labkit._part_state(env, PHONE), labkit._part_state(env, TRAY)
    st, tl = labkit._part_state(env, STAND), labkit._part_state(env, TOOL)
    _w["far"] = _clearance([(ph, _PH_BOX), (tr, _TR_BOX), (st, _ST_BOX), (tl, _TL_BOX)], _tips(env)) >= RELEASE
    snap = np.array([np.asarray(ph.to_world(tuple(p)), float) for p in _PH_PTS]
                    + [np.asarray(tr.to_world(tuple(p)), float) for p in _TR_PTS])
    h = _w["hist"]
    h.append(snap)
    del h[:-STILL_STEPS]
    _w["still"] = (len(h) == STILL_STEPS
                   and max(float(np.max(np.linalg.norm(p - h[-1], axis=1))) for p in h) < STILL_TOL)


def score(objects):
    phone, tray = objects["phone"], objects["tray"]
    if not (_w["far"] and _w["still"] and _docked(phone, objects["stand"])):
        return 0.0
    if _popped(phone, tray) < POPPED:
        return 0.0
    return 1.0 if _tool_home(objects["tool"], objects["block"]) else 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return max(0.0, POPPED - _popped(objects["phone"], objects["tray"]))
