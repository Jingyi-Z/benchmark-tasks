"""Reset a tripped breaker in a home load center (yam_arena task file, built with labkit)."""
import math
import pathlib
import sys

import os as _os

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

# A Square D QO-style home load center standing on the bench, its dead front facing the robots (-x), with six QO120
# single-pole breakers in two columns of three (0.75 in = 19.05 mm pitch; each breaker 3.0 in = 76 mm long across the
# column and 2.91 in deep). Each breaker's handle swings sideways about a vertical pin: ON with its tip towards -y,
# OFF towards +y in the left column and the mirror image in the right (both columns switch ON towards the centre
# bus), a tripped handle resting midway (Schneider FAQ FA353467). Five are ON; one has tripped and shows a red flag.
#
# Trip-free reset, made of contacts (internal geometry estimated; the QO's own is not published): the handle is held
# where it is left by its pivot friction. Behind the dead front it carries a lever and, in a layer above, a finger.
# A sliding arming block sits against the lever's end with a 45 degree ramp: only pushing the handle all the way to
# OFF drives the ramp, sliding the block about 10 mm sideways (friction keeps it there; this is the reset 'click').
# Until then an arm on the same block stops the finger, so the handle cannot get more than about 2 degrees past
# midway towards ON. Once the block has moved, the arm is out of the finger's way and the handle goes to ON.
BLACK, DARK, GREY, STEEL = (0.08, 0.08, 0.09), (0.18, 0.18, 0.2), (0.55, 0.56, 0.58), (0.72, 0.73, 0.75)
WHITE, RED = (0.92, 0.92, 0.9), (0.85, 0.12, 0.1)
TH = 0.44  # handle travel each side of midway (rad, about 25 degrees; estimate)
FACE_X = 0.400  # the breakers' face plane (world x)
PANEL_Z = TOP + 0.190  # middle row's handle pivots
PITCH = 0.01905
COL_Y = (0.040, -0.040)  # column centres: left (+y), right (-y)
ARM_Q = 0.0097  # the arming block's travel when the handle is pushed fully to OFF (from the ramp geometry)
ARMED = 0.008  # the block counts as moved (armed) beyond this
_S45 = math.sqrt(0.5)


def _breaker(name, y, z, tripped, m):
    """m = +1: ON with the handle's tip towards -y (left column); m = -1: the mirror image (right column). Both columns
    switch ON towards the panel's centre line, as on a two-column QO load center."""
    on = m * (TH - 0.01)
    handle = Joint("handle", kind="hinge", axis="Z", anchor=(0.0, 0.0, 0.0), lower=-TH, upper=TH,
                   friction=0.002, damping=0.0005, target=0.0 if tripped else on,
                   initial=0.0 if tripped else on, mass=0.006,
                   boxes=[((-0.0095, 0.0, 0.0), (0.013, 0.007, 0.008), BLACK),  # the handle out front
                          ((-0.01615, 0.0, 0.0), (0.0005, 0.004, 0.0025), WHITE),  # white mark on its end
                          ((0.0175, 0.0, 0.0), (0.031, 0.004, 0.006), DARK),  # lever behind the dead front
                          ((0.036, 0.0, 0.0065), (0.004, 0.004, 0.004), DARK)])  # interlock finger (upper layer)
    rc = (0.031 + 0.003 * _S45 - 0.006 * _S45, m * (-0.0045 - 0.003 * _S45 - 0.006 * _S45), 0.0)
    block = Joint("arming", kind="slide", axis="X", anchor=(0.03, 0.0, 0.0), lower=-0.0005, upper=0.014,
                  friction=0.5, damping=0.5, target=0.0 if tripped else ARM_Q + 0.002,
                  initial=0.0 if tripped else ARM_Q + 0.002, mass=0.004,
                  boxes=[(rc, (0.006, 0.030, 0.006), GREY, -m * math.pi / 4),  # 45 degree ramp at the lever's end
                         ((0.036, m * 0.0055, 0.0065), (0.010, 0.005, 0.004), GREY),  # interlock arm (upper layer)
                         ((-0.0035, m * 0.026, 0.0), (0.002, 0.008, 0.005), RED)])  # trip flag: proud of the face
    # until the block slides back (reset), then hidden behind it
    face = [((-0.0015, s * 0.0235, 0.0), (0.003, 0.029, 0.019), BLACK) for s in (1, -1)]
    face += [((-0.0015, 0.0, s * 0.0075), (0.003, 0.018, 0.004), BLACK) for s in (1, -1)]
    face += [((0.0595, 0.0, 0.0), (0.029, 0.076, 0.019), DARK)]  # the breaker's body behind the mechanism
    return part(name, pos=(FACE_X, y, z), fixed=True, boxes=face, joints=[handle, block], self_collide=True,
                friction=0.2)  # greased mechanism (estimate)


ROWS = (PANEL_Z + PITCH, PANEL_Z, PANEL_Z - PITCH)
TRIPPED = (0, 1)  # left column, middle row
ON_SIGN = (1, -1)  # left column ON towards -y, right column towards +y (both towards the centre bus)
BREAKERS = [_breaker(f"brk{c}{r}", COL_Y[c], ROWS[r], (c, r) == TRIPPED, ON_SIGN[c]) for c in range(2) for r in range(3)]
SIGN = {f"brk{c}{r}": ON_SIGN[c] for c in range(2) for r in range(3)}
TRIP_NAME = f"brk{TRIPPED[0]}{TRIPPED[1]}"

# Load-center dead front and tub (steel), open behind the breakers: a frame round the two columns, a back plate.
_W, _H = 0.160, 0.058  # the cut-out round the breakers
_PANEL = [((FACE_X - 0.001, 0.0, PANEL_Z - TOP + s * (_H / 2 + 0.060)), (0.002, 0.300, 0.120), STEEL) for s in (1, -1)]
_PANEL += [((FACE_X - 0.001, s * (_W / 2 + 0.035), PANEL_Z - TOP), (0.002, 0.070, _H), STEEL) for s in (1, -1)]
_PANEL += [((FACE_X + 0.090, 0.0, PANEL_Z - TOP), (0.006, 0.300, 0.300), GREY)]
_PANEL += [((FACE_X + 0.046, s * 0.147, PANEL_Z - TOP), (0.086, 0.006, 0.300), GREY) for s in (1, -1)]
_PANEL += [((FACE_X + 0.046, 0.0, PANEL_Z - TOP + s * 0.147), (0.086, 0.300, 0.006), GREY) for s in (1, -1)]  # top and bottom
_PANEL += [((FACE_X - 0.003, 0.0, PANEL_Z - TOP + s * (_H / 2 + 0.004)), (0.002, _W + 0.016, 0.008), GREY) for s in (1, -1)]  # trim
_PANEL += [((FACE_X - 0.003, s * (_W / 2 + 0.004), PANEL_Z - TOP), (0.002, 0.008, _H), GREY) for s in (1, -1)]
_PANEL += [((FACE_X + 0.046, 0.0, (PANEL_Z - TOP - 0.150) / 2), (0.086, 0.200, PANEL_Z - TOP - 0.150), DARK)]  # stand
PANEL = part("panel", pos=(0.0, 0.0, TOP), fixed=True, color=STEEL, boxes=_PANEL)
OBJECTS = [PANEL] + BREAKERS

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




R0 = _prev_grip(0.0)
RIGHT_PARK = np.array([0.15, -0.32, TOP + 0.20])
LEFT_PARK = np.array([0.15, 0.32, TOP + 0.20])
TRIM_LIMITS = {}
TRIM_GAIN = {}


def _frame(d, c):
    z = np.asarray(d, float) / np.linalg.norm(d)
    x = np.asarray(c, float) - np.dot(c, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


R_PUSH = _frame(X, Z)  # pointing into the panel, fingers shut one above the other: their side pushes the handle
FT = 0.005  # half the shut fingers' thickness across the push (y), an estimate checked in the debug print
HW = 0.0035  # half the handle's width
TIP_X = FACE_X - 0.011  # how deep the fingertips go: 3 mm short of the handle's root
YC, ZC = COL_Y[TRIPPED[0]], ROWS[TRIPPED[1]]
PUSH_TO_ON_ONLY = False  # True: the control run, pushing the tripped handle straight towards ON


def _hy(q):
    """Handle tip's sideways offset (y) at handle angle q, where the fingers touch it."""
    return -0.011 * math.sin(q)


def _tcp(y, back=0.0):
    return np.array([TIP_X - back, YC + y, ZC]) - TIP_ABOVE_TCP * X


def demo():
    P = _Plan()
    rp, lp, rr = Orient(R_PUSH), Orient(R0), Orient(R0)
    left = []
    if not PUSH_TO_ON_ONLY:
        # 1. fingers beside the tripped handle on its ON side; push it all the way to OFF (the reset)
        y_on_side = _hy(0.0) - HW - FT - 0.002
        y_off_push = _hy(-TH) - HW * math.cos(TH) - FT + 0.004  # 4 mm past touching at the OFF stop (the arm lags)
        left += [(3.0, tuple(_tcp(y_on_side, 0.05)), FINGER_CLOSED, "in front of the tripped breaker", rp),
                 (1.5, tuple(_tcp(y_on_side)), FINGER_CLOSED, "fingers beside its handle", rp),
                 (3.5, tuple(_tcp(y_off_push)), FINGER_CLOSED, "push it all the way to OFF", rp),
                 (2.5, tuple(_tcp(y_off_push)), FINGER_CLOSED, "hold at the OFF stop", rp),
                 (1.0, tuple(_tcp(y_off_push, 0.04)), FINGER_CLOSED, "back off", rp)]
        q_from = -TH
    else:
        q_from = 0.0
    # 2. fingers to the handle's other side; push it to ON
    y_off_side = _hy(q_from) + HW * math.cos(q_from) + FT + 0.002
    y_on_push = _hy(TH) + HW * math.cos(TH) + FT - 0.004
    left += [(2.0, tuple(_tcp(y_off_side, 0.04)), FINGER_CLOSED, "round to the OFF side of the handle", rp),
             (1.5, tuple(_tcp(y_off_side)), FINGER_CLOSED, "fingers beside it", rp),
             (3.5, tuple(_tcp(y_on_push)), FINGER_CLOSED, "push it to ON", rp),
             (2.5, tuple(_tcp(y_on_push)), FINGER_CLOSED, "hold at ON", rp),
             (1.0, tuple(_tcp(y_on_push, 0.05)), FINGER_CLOSED, "back off", rp),
             (2.5, tuple(LEFT_PARK), FINGER_OPEN, "away, to park", lp),
             (3.0, tuple(LEFT_PARK), FINGER_OPEN, "parked", lp)]
    P.phase(left=left, right=[(3.0, tuple(RIGHT_PARK), FINGER_OPEN, "parked", rr)])
    return {"left": P.L, "right": P.R}




# -- scoring ----------------------------------------------------------------------------------------------------------
# 0 for the whole trial once any of the five working breakers has been switched off (its handle more than 0.15 rad,
# about 9 degrees, back from ON at any moment). Otherwise from the final state of the tripped breaker: 0.5 reset (its
# handle pushed all the way to OFF, so the trip-free mechanism has re-armed: arming block moved 8 mm or more) and its
# handle now at OFF (within 0.09 rad of the OFF stop); 1.0 reset and its handle at ON (within 0.09 rad of the ON stop).
# A handle pushed towards ON without the reset stays near midway and scores 0. Every positive tier needs the four pad
# tips and each hand's pad midpoint at least 80 mm from the breakers' faces, and every handle and arming block still
# over the last 21 samples (1 s at 20 Hz: no handle moving 0.01 rad, no block 0.5 mm).
OFF_TOL, ON_TOL, KNOCK = 0.09, 0.09, 0.15
STILL_STEPS, STILL_ANG, STILL_LIN, RELEASE = 21, 0.01, 0.0005, 0.080
_FACE_BOX = (np.array([FACE_X - 0.008, 0.0, PANEL_Z]), np.array([0.008, 0.080, 1.5 * PITCH + 0.010]))


def _box_dist(q, c, h):
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(q, float) - c) - h, 0.0)))


_w = {"hist": [], "still": False, "far": False, "bad": False}


@on_reset
def _w_reset(env):
    _w.update(hist=[], still=False, far=False, bad=False)


@on_step
def _watch(env):
    js = {b.name: labkit._part_state(env, b).joints for b in BREAKERS}
    for name, j in js.items():
        if name != TRIP_NAME and SIGN[name] * float(j["handle"]) < TH - KNOCK:
            _w["bad"] = True
    tips = _tips(env)
    pts = tips + [(tips[0] + tips[1]) / 2, (tips[2] + tips[3]) / 2]
    _w["far"] = min(_box_dist(p, *_FACE_BOX) for p in pts) >= RELEASE
    snap = np.array([[float(j["handle"]), float(j["arming"])] for j in js.values()])
    h = _w["hist"]
    h.append(snap)
    del h[:-STILL_STEPS]
    H = np.array(h)
    span = H.max(0) - H.min(0) if len(h) == STILL_STEPS else None
    _w["still"] = span is not None and bool(np.all(span[:, 0] < STILL_ANG) and np.all(span[:, 1] < STILL_LIN))


def _tier(j):
    if float(j["arming"]) < ARMED:
        return 0.0
    q = SIGN[TRIP_NAME] * float(j["handle"])
    if q >= TH - ON_TOL:
        return 1.0
    if q <= -TH + OFF_TOL:
        return 0.5
    return 0.0


def score(objects):
    if _w["bad"] or not (_w["far"] and _w["still"]):
        return 0.0
    return _tier(objects[TRIP_NAME].joints)


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return 0.0 if _tier(objects[TRIP_NAME].joints) == 1.0 else 1.0
