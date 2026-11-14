"""Recapping a used dental needle with the one-handed scoop technique.

On a stainless instrument tray: a dental aspirating syringe (stainless barrel, finger-grip bar, thumb ring, a 27 gauge
short needle on a plastic hub) lies on a two-block rest, and the needle's cap lies loose on the tray floor beside it,
open end towards the robots. Dentistry is one of the few places where a used needle is recapped (it is used again on
the same patient), and US rules (OSHA 1910.1030) allow that only one-handed or with a device: the free hand never
comes near the needle, and the cap is not held while the needle goes in. The syringe hand slides the needle into the
lying cap, lifts it on, pushes the needle home until the hub meets the cap, and lays the syringe back on its rest.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

STAINLESS = (0.62, 0.63, 0.66)
STEEL = (0.78, 0.79, 0.81)
HUB_C = (0.55, 0.56, 0.60)  # 27 G hubs are grey
CAP_C = (0.15, 0.35, 0.80)
WHITE = (0.95, 0.95, 0.95)
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

# -- the tray and the syringe rest ------------------------------------------------------------------------------------
TX, TY = 0.33, -0.14  # tray centre
TRAY_X, TRAY_Y = 0.24, 0.34  # outside, 24 x 34 cm (a dental setup tray)
FLOOR_T = 0.002
RIM_H, RIM_T = 0.015, 0.0015
FZ = TOP + FLOOR_T  # the tray floor's top (world)

YS = -0.08  # the syringe lies along x at this y, barrel centre at x = XS
XS = 0.30
BARREL_R, BARREL_L = 0.0055, 0.080
AXZ = 0.025  # barrel axis height above the tray floor, on the rest
REST_X = (-0.028, 0.030)  # the rest's two blocks, along the barrel from its centre
REST_L = 0.008  # each block 8 mm along the barrel
REST_TOP = AXZ - BARREL_R  # the barrel lies on the blocks' tops
LIP_H, LIP_T = 0.003, 0.003
LIP_GAP = 0.012  # the lips' inner faces 12 mm apart: the barrel cannot roll off


def _tray_boxes():
    hx, hy = TRAY_X / 2, TRAY_Y / 2
    b = [((0.0, 0.0, FLOOR_T / 2), (TRAY_X, TRAY_Y, FLOOR_T), STAINLESS),
         ((hx - RIM_T / 2, 0.0, FLOOR_T + RIM_H / 2), (RIM_T, TRAY_Y, RIM_H), STAINLESS),
         ((-hx + RIM_T / 2, 0.0, FLOOR_T + RIM_H / 2), (RIM_T, TRAY_Y, RIM_H), STAINLESS),
         ((0.0, hy - RIM_T / 2, FLOOR_T + RIM_H / 2), (TRAY_X, RIM_T, RIM_H), STAINLESS),
         ((0.0, -hy + RIM_T / 2, FLOOR_T + RIM_H / 2), (TRAY_X, RIM_T, RIM_H), STAINLESS)]
    for rx in REST_X:  # the rest: a block with two low lips
        cx, cy = XS + rx - TX, YS - TY
        b.append(((cx, cy, FLOOR_T + REST_TOP / 2), (REST_L, LIP_GAP + 2 * LIP_T, REST_TOP), WHITE))
        for s in (-1, 1):
            b.append(((cx, cy + s * (LIP_GAP / 2 + LIP_T / 2), FLOOR_T + REST_TOP + LIP_H / 2),
                      (REST_L, LIP_T, LIP_H), WHITE))
    return b


TRAY = part("tray", pos=(TX, TY, TOP), fixed=True, boxes=_tray_boxes())

# -- the syringe ------------------------------------------------------------------------------------------------------
# Part frame: origin on the barrel's axis at its middle, +x towards the needle.
NOSE_R, NOSE_L = 0.004, 0.010  # the threaded nose the needle's hub screws onto, 8 mm across
HUB_R, HUB_L = 0.003, 0.010
HUB_X0 = BARREL_L / 2 + NOSE_L  # the hub's back face, on the nose's front end
HUB_FACE = HUB_X0 + HUB_L  # the hub's front face, where the cap's mouth seats
NEEDLE_R, NEEDLE_L = 0.0002, 0.025  # 27 gauge (0.4 mm), short (25 mm)
TIP_X = HUB_FACE + NEEDLE_L
RING_X = -BARREL_L / 2 - 0.017  # thumb ring centre
RING_O, RING_W = 0.022, 0.004  # ring 22 mm across, 4 mm bars
SYRINGE = part(
    "syringe", pos=(XS, YS, FZ + AXZ), mass=0.060, friction=1.0,
    cylinders=[((0.0, 0.0, 0.0), BARREL_R, BARREL_L, STAINLESS, "X"),
               ((BARREL_L / 2 + NOSE_L / 2, 0.0, 0.0), NOSE_R, NOSE_L, STAINLESS, "X"),
               ((HUB_X0 + HUB_L / 2, 0.0, 0.0), HUB_R, HUB_L, HUB_C, "X"),
               ((HUB_FACE + NEEDLE_L / 2, 0.0, 0.0), NEEDLE_R, NEEDLE_L, STEEL, "X"),
               ((-BARREL_L / 2 - 0.003, 0.0, 0.0), 0.002, 0.006, STEEL, "X")],  # plunger rod to the ring
    boxes=[((-BARREL_L / 2 + 0.004, 0.0, 0.0), (0.004, 0.034, 0.004), STAINLESS),  # finger-grip bar
           ((RING_X, 0.0, (RING_O - RING_W) / 2), (RING_O, 0.004, RING_W), STAINLESS),  # thumb ring, upright
           ((RING_X, 0.0, -(RING_O - RING_W) / 2), (RING_O, 0.004, RING_W), STAINLESS),
           ((RING_X - (RING_O - RING_W) / 2, 0.0, 0.0), (RING_W, 0.004, RING_O), STAINLESS),
           ((RING_X + (RING_O - RING_W) / 2, 0.0, 0.0), (RING_W, 0.004, RING_O), STAINLESS)],
)

# -- the needle's cap -------------------------------------------------------------------------------------------------
# Part frame: origin at the cap's middle, +x from its open end (the mouth) to its closed end.
CAP_O = 0.007  # 7 mm square outside (a flat-sided cap: it lies still)
BORE = 0.0026  # the needle's bore, 2.6 mm square
BORE1, WIDE_L = 0.0035, 0.010  # but 3.5 mm for its first 10 mm (caps flare at the mouth)
FUN_L, FUN_IN = 0.005, 0.0050  # the mouth: a 5 mm funnel from 5.0 mm across down to the wide bore
END_T = 0.002
CAP_L = FUN_L + NEEDLE_L + 0.003 + END_T  # 35 mm: the needle's tip stops 3 mm short of the closed end
CAP_X0 = -CAP_L / 2  # the mouth's face
_wall = (CAP_O - BORE) / 2
_wall1 = (CAP_O - BORE1) / 2
_run = CAP_L - FUN_L - WIDE_L - END_T  # the narrow bore's length
_rx = CAP_X0 + FUN_L + WIDE_L + _run / 2
_wx = CAP_X0 + FUN_L + WIDE_L / 2
_fa = math.atan2((FUN_IN - BORE1) / 2, FUN_L)  # the funnel walls' slope
_ft = 0.0012  # funnel walls 1.2 mm thick
_fc = CAP_X0 + FUN_L / 2
_fo = (FUN_IN + BORE1) / 4 + _ft / 2 / math.cos(_fa)  # the funnel walls' centres off the axis
_fl = FUN_L / math.cos(_fa) + 0.0004
_sh = 0.0006  # the funnel section's outer shell
CAP_BOXES = [
    ((_rx, 0.0, (BORE + _wall) / 2), (_run, CAP_O, _wall), CAP_C),  # straight bore: top, bottom and sides
    ((_rx, 0.0, -(BORE + _wall) / 2), (_run, CAP_O, _wall), CAP_C),
    ((_rx, (BORE + _wall) / 2, 0.0), (_run, _wall, BORE), CAP_C),
    ((_rx, -(BORE + _wall) / 2, 0.0), (_run, _wall, BORE), CAP_C),
    ((_wx, 0.0, (BORE1 + _wall1) / 2), (WIDE_L, CAP_O, _wall1), CAP_C),  # wide bore: top, bottom and sides
    ((_wx, 0.0, -(BORE1 + _wall1) / 2), (WIDE_L, CAP_O, _wall1), CAP_C),
    ((_wx, (BORE1 + _wall1) / 2, 0.0), (WIDE_L, _wall1, BORE1), CAP_C),
    ((_wx, -(BORE1 + _wall1) / 2, 0.0), (WIDE_L, _wall1, BORE1), CAP_C),
    ((CAP_L / 2 - END_T / 2, 0.0, 0.0), (END_T, CAP_O, CAP_O), CAP_C),  # closed end
    # the funnel: four sloped walls (pitched about y, or turned about z), inside a thin square shell
    ((_fc, 0.0, _fo), (_fl, FUN_IN + 2 * _ft, _ft), CAP_C, (0.0, _fa)),
    ((_fc, 0.0, -_fo), (_fl, FUN_IN + 2 * _ft, _ft), CAP_C, (0.0, -_fa)),
    ((_fc, _fo, 0.0), (_fl, _ft, FUN_IN + 2 * _ft), CAP_C, (-_fa, 0.0)),
    ((_fc, -_fo, 0.0), (_fl, _ft, FUN_IN + 2 * _ft), CAP_C, (_fa, 0.0)),
    ((_fc, 0.0, (CAP_O - _sh) / 2), (FUN_L, CAP_O, _sh), CAP_C),
    ((_fc, 0.0, -(CAP_O - _sh) / 2), (FUN_L, CAP_O, _sh), CAP_C),
    ((_fc, (CAP_O - _sh) / 2, 0.0), (FUN_L, _sh, CAP_O - 2 * _sh), CAP_C),
    ((_fc, -(CAP_O - _sh) / 2, 0.0), (FUN_L, _sh, CAP_O - 2 * _sh), CAP_C),
]
YC = -0.18  # the cap lies along x at this y, its mouth towards the robots
MOUTH_X = 0.360  # the cap's mouth face (world x)
CAP_MASS = 0.0006  # 0.6 g of polypropylene
CAP = part("needle_cap", pos=(MOUTH_X + CAP_L / 2, YC, FZ + CAP_O / 2), mass=CAP_MASS, friction=0.5, boxes=CAP_BOXES)

OBJECTS = [TRAY, SYRINGE, CAP]

# -- scoring ----------------------------------------------------------------------------------------------------------
IN_DEPTH = 0.010  # the needle's tip at least 10 mm inside the cap's mouth, within the bore
BORE_TOL = 0.0015  # the tip within 1.5 mm of the cap's axis (inside the bore)
SEAT_TOL = 0.002  # seated: the hub's front face within 2 mm of the cap's mouth
AXIS_TOL_DEG = 10.0
REST_POS_TOL, REST_AX_DEG = 0.005, 5.0  # back on the rest: barrel centre within 5 mm of its place, axis within 5 deg
FREE_HAND = 0.10  # never both hands' fingertips within 10 cm of the needle's tip at once
CAP_TOUCH = 0.003  # a fingertip within 3 mm of the cap's surface touches it
FLOOR_TOUCH = 0.0015  # the needle's tip within 1.5 mm of the tray floor touches it
RELEASE = 0.08  # let go: every fingertip at least 8 cm from the syringe's barrel axis and the cap
STILL_STEPS, STILL_TOL = 20, 0.001
_w = {"bad": "", "hist": [], "still": False, "far": False}


def _hand_tips(env):
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
        out.append([mid + half * g[:3, 0], mid - half * g[:3, 0]])
    return out  # [[left tip a, left tip b], [right tip a, right tip b]]


def _cap_gap(cap, p):
    """Distance (m) from world point p to the cap's outside (0 inside it)."""
    q = np.abs(cap.to_local(p)) - np.array([CAP_L / 2, CAP_O / 2, CAP_O / 2])
    return float(np.linalg.norm(np.maximum(q, 0.0)))


def _seg_dist(p, a, b):
    d = b - a
    u = min(1.0, max(0.0, float((p - a) @ d / (d @ d))))
    return float(np.linalg.norm(p - (a + u * d)))


def needle(objects):
    """(tip depth inside the cap's mouth, tip distance from the cap's axis, hub face to mouth gap, axis angle deg)."""
    s, c = objects["syringe"], objects["needle_cap"]
    tip = s.to_world((TIP_X, 0.0, 0.0))
    hub = s.to_world((HUB_FACE, 0.0, 0.0))
    lt, lh = c.to_local(tip), c.to_local(hub)
    depth = float(lt[0] - CAP_X0)
    off = float(math.hypot(lt[1], lt[2]))
    gap = float(CAP_X0 - lh[0]) if math.hypot(lh[1], lh[2]) <= BORE_TOL else 1.0
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(s.rotation()[:, 0] @ c.rotation()[:, 0])))))
    return depth, off, gap, ang


def _in_cap(objects):
    """The tip at least IN_DEPTH inside the mouth and inside the bore's square section there (the wide bore for the
    first FUN_L + WIDE_L), the two axes within AXIS_TOL_DEG."""
    depth, _, _, ang = needle(objects)
    s, c = objects["syringe"], objects["needle_cap"]
    lt = c.to_local(s.to_world((TIP_X, 0.0, 0.0)))
    half = (BORE1 if depth < FUN_L + WIDE_L else BORE) / 2 - NEEDLE_R + 0.0002
    return depth >= IN_DEPTH and max(abs(float(lt[1])), abs(float(lt[2]))) <= half and ang <= AXIS_TOL_DEG


def _seated(objects):
    return _in_cap(objects) and needle(objects)[2] <= SEAT_TOL


def _on_rest(s):
    ax = s.rotation()[:, 0]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(ax[0])))))
    return (float(np.linalg.norm(np.asarray(s.pos, float) - [XS, YS, FZ + AXZ])) <= REST_POS_TOL
            and ang <= REST_AX_DEG)


@on_reset
def _reset_watch(env):
    _w["bad"], _w["hist"], _w["still"], _w["far"] = "", [], False, False


@on_step
def _watch(env):
    """Latches a violation for the rest of the trial: both hands near the needle at once, the cap touched before the
    needle is seated in it, or the needle's tip on the tray floor. Also tracks release and stillness."""
    objs = {n: labkit._part_state(env, labkit._SPECS[n]) for n in ("syringe", "needle_cap")}
    s, c = objs["syringe"], objs["needle_cap"]
    tip = np.asarray(s.to_world((TIP_X, 0.0, 0.0)), float)
    hands = _hand_tips(env)
    near = [min(float(np.linalg.norm(t - tip)) for t in h) < FREE_HAND for h in hands]
    if not _w["bad"]:
        if all(near):
            _w["bad"] = "both hands near the needle"
        elif not _seated(objs) and min(_cap_gap(c, t) for h in hands for t in h) < CAP_TOUCH:
            _w["bad"] = "cap touched"
        elif tip[2] < FZ + FLOOR_TOUCH:
            _w["bad"] = "needle on the tray"
    a = np.asarray(s.to_world((-BARREL_L / 2, 0.0, 0.0)), float)
    b = np.asarray(s.to_world((BARREL_L / 2, 0.0, 0.0)), float)
    _w["far"] = all(_seg_dist(t, a, b) >= RELEASE and _cap_gap(c, t) >= RELEASE for h in hands for t in h)
    h = _w["hist"]
    h.append(np.r_[np.asarray(s.pos, float), np.asarray(c.pos, float)])
    del h[:-STILL_STEPS]
    _w["still"] = len(h) == STILL_STEPS and max(float(np.max(np.abs(p - h[-1]))) for p in h) < STILL_TOL


def score(objects):
    """0 after any violation (latched for the whole trial): both hands' fingertips within 10 cm of the needle's tip at
    the same moment; a fingertip touching the cap (within 3 mm of it) before the needle is seated in it; the needle's
    tip within 1.5 mm of the tray floor. Otherwise, at the end: 0.25 if the needle's tip is at least 10 mm inside the
    cap's mouth, within 1.5 mm of its axis, the two axes within 10 deg. 0.5 if also seated: the hub's front face within
    2 mm of the cap's mouth. 1.0 if also the syringe lies back on its rest (barrel centre within 5 mm of its starting
    place, axis within 5 deg), let go (every fingertip 8 cm or more from the barrel's axis and from the cap), and both
    are at rest (moved under 1 mm over the last 1 s)."""
    if _w["bad"] or not _in_cap(objects):
        return 0.0
    if not _seated(objects):
        return 0.25
    if not (_on_rest(objects["syringe"]) and _w["far"] and _w["still"]):
        return 0.5
    return 1.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """How far (m) the needle still has to go: the hub's front face to the cap's mouth, along the cap."""
    s, c = objects["syringe"], objects["needle_cap"]
    return float(np.linalg.norm(np.asarray(s.to_world((HUB_FACE, 0.0, 0.0))) - np.asarray(c.to_world((CAP_X0, 0.0, 0.0)))))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The right hand takes the syringe off its rest by the barrel, tips it 15 degrees nose down, slides the needle into the
# lying cap's funnel, lifts a little so the cap's mouth rides up onto the needle, pushes on until the hub meets the cap,
# then lifts, levels the syringe and lays it back on its rest. The left hand stays parked, far from the needle.
# Everything it aims at is seen: the syringe and the cap.

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
        ids = robot.find_joints(["joint7"], preserve_order=True)[0]
        half = abs(float(robot.data.joint_pos.torch[0, ids].cpu().numpy()[0]))
        out += [mid + half * g[:3, 0], mid - half * g[:3, 0]]
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
        _ct["trim"][key] = np.clip(tr + np.clip(0.2 * e, -0.0003, 0.0003), -0.020, 0.020)


class _Dyn:
    """A TCP plus the live trim ``key``."""

    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)


GRIP_OPEN = -0.012  # 24 mm between the pads for the 11 mm barrel
PITCH = math.radians(8.0)  # the syringe held nose down by this while it scoops (the barrel and fingers then clear the floor)
R0 = _prev_grip(0.0)  # straight down, the fingers closing along y: across the barrel, which lies along x
RT = axis_angle_matrix(Y, PITCH) @ R0  # the same, tipped nose down
O0, OT = Orient(R0), Orient(RT)
D = RT @ R0.T @ X  # the needle's direction while tipped
GRIP_TIPS = -0.007  # the pad tips 7 mm below the barrel's axis (the barrel held on the pads' flat faces, not their tips)
TCP_REST = np.array([XS, YS, FZ + AXZ + GRIP_TIPS + TIP_ABOVE_TCP])  # holding the barrel's middle, on its rest
TIP_OFF = np.array([XS + TIP_X, YS, FZ + AXZ]) - TCP_REST  # the needle's tip from the TCP, held level


def _tcp_for_tip(tip, R=RT):
    """TCP that puts the held needle's tip at ``tip`` with the hand turned to R."""
    return np.asarray(tip, float) - R @ R0.T @ TIP_OFF


# The cap as it lies, and as it will lie tipped up onto the needle (pivoting on its closed end's bottom edge).
MOUTH0 = np.array([MOUTH_X, YC, FZ + CAP_O / 2])
PIVOT = np.array([MOUTH_X + CAP_L, YC, FZ])
_Ry = axis_angle_matrix(Y, PITCH)
MOUTH_T = PIVOT + _Ry @ np.array([-CAP_L, 0.0, CAP_O / 2])
ENTRY = MOUTH0 + 0.0009 * Z  # aimed 0.9 mm high: the tipped needle drops 1.7 mm over 12 mm, inside the wide bore
DEPTH1 = 0.012  # first in this far, into the wide bore (the lying cap stops a deeper tipped needle)
RIDE = 0.0010  # once lifted, the tip rides this far above the cap's axis: the cap's weight rests on the needle
TRIM_GAIN = 0.25  # (the trim then moves 5% of the seen error a step: slow enough that the arm keeps up)
LINE_UP = 0.008  # held this far short of the mouth while the tip is lined up on it (as seen)
SEAT_PUSH = 0.001  # pushed on 1 mm past the hub meeting the cap's mouth
RIGHT_PARK = np.array([0.15, -0.32, TOP + 0.20])


def _line_up(env):
    """Seen error of the needle's tip from its line-up point, LINE_UP short of the seen cap mouth along the needle."""
    s = labkit._part_state(env, labkit._SPECS["syringe"])
    c = labkit._part_state(env, labkit._SPECS["needle_cap"])
    mouth = np.asarray(c.to_world((CAP_X0, 0.0, 0.0)), float)
    tip = np.asarray(s.to_world((TIP_X, 0.0, 0.0)), float)
    return TRIM_GAIN * ((mouth + (ENTRY - MOUTH0) - LINE_UP * D) - tip)


def _ride(env):
    """Seen error of the needle's tip from DEPTH1 inside the seen cap, RIDE above its axis (the cap tipped up on it)."""
    s = labkit._part_state(env, labkit._SPECS["syringe"])
    c = labkit._part_state(env, labkit._SPECS["needle_cap"])
    want = np.asarray(c.to_world((CAP_X0 + DEPTH1, 0.0, 0.0)), float) + RIDE * Z
    e = want - np.asarray(s.to_world((TIP_X, 0.0, 0.0)), float)
    return TRIM_GAIN * (e - (e @ D) * D)  # (across the needle only: chasing depth would just push the loose cap away)


def demo():
    from yam_arena.constants import LEFT_BASE_POS

    up = Z
    right = [
        Waypoint(2.0, tuple(TCP_REST + 0.06 * up), GRIP_OPEN, "above the barrel", O0),
        Waypoint(1.5, tuple(TCP_REST), GRIP_OPEN, "down round the barrel", O0),
        Waypoint(0.8, tuple(TCP_REST), FINGER_CLOSED, "close on the barrel", O0),
        Waypoint(1.2, tuple(TCP_REST + 0.04 * up), FINGER_CLOSED, "lift off the rest", O0),
        Waypoint(2.5, tuple(_tcp_for_tip(ENTRY - 0.015 * D + 0.02 * up)), FINGER_CLOSED,
                 "over to the cap, nose down", OT),
    ]
    k = "nt"
    T = lambda tip: _Dyn(_tcp_for_tip(tip), k)  # (every waypoint from here on carries the seen trim)
    right.append(Waypoint(1.5, T(ENTRY - LINE_UP * D), FINGER_CLOSED, "the needle in line with the cap's mouth", OT))
    t_line = sum(w.duration for w in right)
    lifted = PIVOT + _Ry @ (MOUTH0 + DEPTH1 * X - PIVOT) + RIDE * Z  # DEPTH1 in, the cap tipped up by PITCH round its far edge
    seat = lifted + (NEEDLE_L + SEAT_PUSH - DEPTH1) * D
    rim_in = TX + TRAY_X / 2 - RIM_T  # the tray's far rim, inner face (world x)
    at_rim = lifted + (rim_in - 0.001 - (MOUTH_X + CAP_L)) * X  # the cap's closed end 1 mm short of the rim
    pressed = at_rim + (NEEDLE_L + SEAT_PUSH - DEPTH1 + 0.001) * X  # then on until the hub meets the cap's mouth
    right += [
        Waypoint(4.0, T(ENTRY - LINE_UP * D), FINGER_CLOSED, "hold still: the tip lined up on the mouth, as seen", OT),
        Waypoint(2.0, T(ENTRY + DEPTH1 * D), FINGER_CLOSED, "into the funnel", OT),
        Waypoint(2.0, T(lifted), FINGER_CLOSED, "lift: the cap tips up on the needle, its far end on the floor", OT),
        Waypoint(3.0, T(lifted), FINGER_CLOSED, "hold still: the tip kept on the tipped cap's axis, as seen", OT),
        Waypoint(3.0, T(at_rim), FINGER_CLOSED, "carry the riding cap on until its closed end meets the tray's far rim", OT),
        Waypoint(2.5, T(pressed), FINGER_CLOSED, "press it home against the rim: the hub meets the cap", OT),
        Waypoint(0.8, T(pressed), FINGER_CLOSED, "hold", OT),
        Waypoint(1.5, _Dyn(_tcp_for_tip(pressed - 0.004 * X) + 0.04 * up, k), FINGER_CLOSED, "back off the rim and lift, capped", O0),
        Waypoint(2.5, tuple(TCP_REST + 0.03 * up), FINGER_CLOSED, "level, over the rest", O0),
        Waypoint(1.5, tuple(TCP_REST + 0.001 * up), FINGER_CLOSED, "lay it on the rest", O0),
        Waypoint(0.8, tuple(TCP_REST + 0.001 * up), GRIP_OPEN, "let go", O0),
        Waypoint(1.2, tuple(TCP_REST + 0.06 * up), GRIP_OPEN, "up", O0),
        Waypoint(2.0, tuple(RIGHT_PARK), FINGER_OPEN, "away, to park", O0),
        Waypoint(2.0, tuple(RIGHT_PARK), FINGER_OPEN, "hold", O0),
    ]
    t_ride = t_line + 4.0 + 2.0 + 2.0  # (the lifted hold starts here)
    _ct["wins"][:] = [(t_line + 0.3, t_line + 3.8, k, _line_up, "syringe"),  # (only while the syringe is seen still)
                      (t_ride + 0.5, t_ride + 2.8, k, _ride, "needle_cap")]  # (and the cap)
    p_l, r_l = _ik.tcp_pose(np.array(_READY, float), np.array(LEFT_BASE_POS))
    t = sum(w.duration for w in right)
    left = [Waypoint(t, tuple(p_l), FINGER_OPEN, "stays parked, far from the needle", Orient(r_l))]
    return {"left": left, "right": right}

