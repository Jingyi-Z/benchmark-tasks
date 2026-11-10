"""Loading chips into a 2 inch tube furnace for an anneal (2D materials, after a CVD tube-furnace SOP).

A quartz tube runs through a split-tube furnace, its open end, with a KF flange, towards the robots. Just inside the
mouth stands a quartz boat with three chips in it, and behind it, lying in two cradles in line with the tube, a stainless
hook rod, its pusher just short of the tube's end. The boat must go to the furnace's hot zone: the rod carries a red
mark that meets the tube's end when the boat is centred on the thermocouple. A rod as long as the push cannot be pushed
from its end, so it is fed hand over hand; the boat is hidden inside the furnace, so the mark is all one can see.

The job: push the boat to the hot zone with the rod, without jolting the chips off their places (a chip that slides or
flips is a lost sample), pull the rod back out and lay it in its cradles. Then seal the tube (centering ring with its
O-ring on the flange, blank cap, clamp).

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
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
DT = 0.05  # one control step at 20 Hz

QUARTZ = (0.86, 0.90, 0.92)
STAINLESS = (0.62, 0.63, 0.66)
STEEL = (0.72, 0.73, 0.75)
DARK = (0.20, 0.21, 0.23)
FURNACE_C = (0.86, 0.86, 0.84)
INSUL = (0.93, 0.92, 0.88)
CHIP_C = (0.42, 0.33, 0.58)
RED = (0.80, 0.12, 0.10)
WHITE = (0.96, 0.96, 0.96)
labkit.GLASS_COLORS.update({QUARTZ})
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
FONT = {}


def Rz(a):
    return axis_angle_matrix(Z, a)


def _boxes_mesh(boxes, colour):
    """One drawn-only mesh (no collision) from many boxes: (centre, size[, yaw]) tuples."""
    pts, tris = [], []
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    for box in boxes:
        c, h = np.asarray(box[0], float), np.asarray(box[1], float) / 2
        rot = Rz(box[2]) if len(box) > 2 else np.eye(3)
        b = len(pts)
        pts += [c + rot @ (h * [sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        tris += [(b + i, b + j, b + k) for i, j, k in faces]
    return labkit.Mesh(points=np.array(pts), tris=np.array(tris), color=colour, collide="none")


def _glyphs(text, pix):
    w = len(text) * 6 * pix - pix
    out = []
    for n, ch in enumerate(text):
        for r, row in enumerate(FONT[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    out.append((-w / 2 + (n * 6 + c + 0.5) * pix, (3 - r) * pix))
    return out


def _text(line, pix, cx, cy, z):
    """Letters printed on a flat face, read from the robots' side: the line runs along -y, letters' tops towards +x."""
    return [((cx + v, cy - u, z), (pix * 0.96, pix * 0.96, 0.0006)) for u, v in _glyphs(line, pix)]


def _prism(outline, z0, z1, colour):
    n = len(outline)
    pts = [(*p, z1) for p in outline] + [(*p, z0) for p in outline] + [(0.0, 0.0, z1), (0.0, 0.0, z0)]
    ct, cb = 2 * n, 2 * n + 1
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(ct, i, j), (cb, n + j, n + i), (i, n + i, n + j), (i, n + j, j)]
    return labkit.Mesh(points=np.array(pts, float), tris=np.array(tris), color=colour, collide="none")


def _disc(r, n=96):
    return np.array([(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)])


def _arc_boxes(r0, r1, z0, z1, degs, width, colour):
    """Boxes along a ring between radii r0 and r1, one per angle in degs (tangential size ``width``)."""
    out = []
    for d in degs:
        a = math.radians(d)
        rm = (r0 + r1) / 2
        out.append(((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (r1 - r0, width, z1 - z0), colour, a))
    return out


def _tube_mesh(x0, x1, r_out, r_in, colour, n=48):
    """A drawn-only hollow cylinder along x from x0 to x1 (part frame, axis at y = 0, z = AXZ)."""
    pts, tris = [], []
    for x in (x0, x1):
        for r in (r_out, r_in):
            pts += [(x, r * math.cos(2 * math.pi * i / n), AXZ + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    o0, i0, o1, i1 = 0, n, 2 * n, 3 * n
    for i in range(n):
        j = (i + 1) % n
        tris += [(o0 + i, o1 + i, o1 + j), (o0 + i, o1 + j, o0 + j)]  # outside
        tris += [(i0 + i, i1 + j, i1 + i), (i0 + i, i0 + j, i1 + j)]  # inside
        tris += [(o0 + i, o0 + j, i0 + j), (o0 + i, i0 + j, i0 + i)]  # end at x0
        tris += [(o1 + i, i1 + j, o1 + j), (o1 + i, i1 + i, i1 + j)]  # end at x1
    return labkit.Mesh(points=np.array(pts, float), tris=np.array(tris), color=colour, collide="none")


# -- the furnace (fixed): a split-tube furnace 300 mm long with a 2 inch quartz tube through it, the tube's open end towards
# the robots, a KF flange on it. Frame: on the bench, the tube's axis at y = 0, AXZ above the bench, along +x. ---------------
AXZ = 0.150  # the tube's axis above the bench (low: the arms reach the rod only a little higher)
TUBE_R, TUBE_IN = 0.0254, 0.0230  # 2 inch quartz tube: 50.8 mm outside, 46 mm inside
MOUTH_X, TUBE_X1 = 0.330, 0.860  # the flange's face (clear of the hands' strokes), and the tube's far end
FL_X0, FL_R = 0.315, 0.0375  # the KF flange (stainless, 75 mm across) from FL_X0 to the mouth
FURN_X0, FURN_X1, FURN_Y, FURN_Z1 = 0.470, 0.770, 0.130, 0.330
HOT_X = 0.620  # the hot zone's centre, at the thermocouple
FLOOR = AXZ - 0.0195  # the tube's floor under the boat: the 46 mm bore 12 mm either side of the bottom
_L = TUBE_X1 - FL_X0
_mid = (TUBE_X1 + FL_X0) / 2
_hole = TUBE_R + 0.002
FURNACE_BOXES = [
    # inside the tube: its floor and the walls either side, in steps that follow the round bore
    ((_mid, 0.0, FLOOR - 0.002), (_L, 0.024, 0.004), QUARTZ),
    *[((_mid, e * 0.0145, AXZ - 0.0165 - 0.002), (_L, 0.005, 0.004), QUARTZ) for e in (-1, 1)],
    *[((_mid, e * 0.0195, AXZ - 0.010 - 0.004), (_L, 0.004, 0.008), QUARTZ) for e in (-1, 1)],
    *[((_mid, e * 0.0215, AXZ), (_L, 0.003, 0.014), QUARTZ) for e in (-1, 1)],
    ((_mid, 0.0, AXZ + 0.0215), (_L, 0.030, 0.003), QUARTZ),
    # the flange: a square of four bars round the bore, inside its round outline
    ((FL_X0 + 0.0075, 0.0, AXZ + 0.0285), (0.015, 0.034, 0.011), STAINLESS),
    ((FL_X0 + 0.0075, 0.0, AXZ - 0.0285), (0.015, 0.034, 0.011), STAINLESS),
    *[((FL_X0 + 0.0075, e * 0.0285, AXZ), (0.015, 0.011, 0.034), STAINLESS) for e in (-1, 1)],
    # the tube's stand, just behind the flange
    ((0.375, 0.0, (AXZ - TUBE_R) / 2), (0.030, 0.050, AXZ - TUBE_R), DARK),
    # the furnace body round the tube (a square hole for it), on its feet
    (((FURN_X0 + FURN_X1) / 2, 0.0, (0.030 + AXZ - _hole) / 2), (FURN_X1 - FURN_X0, 2 * FURN_Y, AXZ - _hole - 0.030),
     FURNACE_C),
    (((FURN_X0 + FURN_X1) / 2, 0.0, (AXZ + _hole + FURN_Z1) / 2), (FURN_X1 - FURN_X0, 2 * FURN_Y, FURN_Z1 - AXZ - _hole),
     FURNACE_C),
    *[(((FURN_X0 + FURN_X1) / 2, e * (FURN_Y + _hole) / 2, AXZ), (FURN_X1 - FURN_X0, FURN_Y - _hole, 2 * _hole),
       FURNACE_C) for e in (-1, 1)],
    *[((x, e * (FURN_Y - 0.015), 0.015), (0.030, 0.030, 0.030), DARK) for x in (FURN_X0 + 0.02, FURN_X1 - 0.02)
      for e in (-1, 1)],
    # the tube's far end rests on a block
    ((0.825, 0.0, (AXZ - TUBE_R) / 2), (0.030, 0.050, AXZ - TUBE_R), DARK),
    # the hot zone's mark on the furnace's lid, in line with the thermocouple
    ((HOT_X, 0.0, FURN_Z1 + 0.0005), (0.004, 0.120, 0.001), RED),
]
FURNACE = part("tube_furnace", pos=(0.0, 0.0, TOP), fixed=True, friction=0.4, boxes=FURNACE_BOXES,
               meshes=[_tube_mesh(MOUTH_X - 0.001, TUBE_X1, TUBE_R, TUBE_IN, QUARTZ),
                       _tube_mesh(FL_X0, MOUTH_X, FL_R, TUBE_IN + 0.0005, STAINLESS)])

# -- the boat (loose): a fused-quartz boat 80 x 20 x 8 mm, open on top, its back end raised 17 mm. Frame: origin at the
# centre of its underside. ---------------------------------------------------------------------------------------------
BOAT_L, BOAT_W, BOAT_H, BOAT_T = 0.080, 0.020, 0.008, 0.0015
BACK_H = 0.017  # its back end is raised into a tab, which the rod's pusher meets
BOAT_START = np.array([MOUTH_X + 0.004 + BOAT_L / 2, 0.0, TOP + FLOOR])
BOAT = part("boat", pos=tuple(BOAT_START), mass=0.012, friction=0.4,
            boxes=[((0.0, 0.0, BOAT_T / 2), (BOAT_L, BOAT_W, BOAT_T), QUARTZ),
                   *[((0.0, e * (BOAT_W - BOAT_T) / 2, BOAT_H / 2), (BOAT_L, BOAT_T, BOAT_H), QUARTZ) for e in (-1, 1)],
                   ((-(BOAT_L - BOAT_T) / 2, 0.0, BACK_H / 2), (BOAT_T, BOAT_W - 2 * BOAT_T, BACK_H), QUARTZ),
                   (((BOAT_L - BOAT_T) / 2, 0.0, BOAT_H / 2), (BOAT_T, BOAT_W - 2 * BOAT_T, BOAT_H), QUARTZ)])

# -- the chips (loose): three 10 x 10 x 0.5 mm silicon chips with 2D flakes on them, lying in the boat. -------------------
CHIP, CHIP_T = 0.010, 0.0005
CHIP_DX = (-0.024, 0.0, 0.024)
CHIP_START = [BOAT_START + [dx, 0.0, BOAT_T + CHIP_T / 2] for dx in CHIP_DX]
CHIPS = [part(f"chip{i}", pos=tuple(CHIP_START[i]), mass=0.00012, friction=0.8,
              boxes=[((0.0, 0.0, 0.0), (CHIP, CHIP, CHIP_T), CHIP_C)]) for i in range(3)]

# -- the hook rod (loose): a 6 mm stainless rod 520 mm long, a 16 mm pusher disc at its front end for the boat's raised
# back, and under it a skid that slides on the tube's floor (so the rod's front stays level),
# and a red mark 250 mm back from the hook: at the tube's end when the boat is centred in the hot zone. Frame: origin on
# the rod's axis at the hook's front face, the rod along -x. ------------------------------------------------------------
ROD_L, ROD_R = 0.520, 0.003
HANDLE_X0, HANDLE_X1, HANDLE_W = 0.180, 0.400, 0.008  # its square 8 mm grip, this far back from the front
PUSHER_R = 0.008  # the pusher disc at its front: it meets the boat's back wall whichever way the rod is turned
MARK_AT = HOT_X - BOAT_L / 2 - MOUTH_X  # 0.25 m behind the hook
ROD_START = np.array([MOUTH_X - 0.010, 0.0, TOP + AXZ])  # its pusher 10 mm short of the tube's end
ROD = part("hook_rod", pos=tuple(ROD_START), mass=0.115, friction=0.7,
           cylinders=[((-(HANDLE_X0 + 0.0) / 2, 0.0, 0.0), ROD_R, HANDLE_X0, STEEL, "X"),
                      ((-(HANDLE_X1 + ROD_L) / 2, 0.0, 0.0), ROD_R, ROD_L - HANDLE_X1, STEEL, "X"),
                      ((-0.0015, 0.0, 0.0), PUSHER_R, 0.003, STEEL, "X"),
                      ((-MARK_AT, 0.0, 0.0), ROD_R + 0.0003, 0.006, RED, "X")],
           boxes=[((-(HANDLE_X0 + HANDLE_X1) / 2, 0.0, 0.0), (HANDLE_X1 - HANDLE_X0, HANDLE_W, HANDLE_W), DARK),
                  ((-0.0055, 0.0, -(AXZ - FLOOR) / 2), (0.003, 0.005, AXZ - FLOOR - 0.002), STEEL),
                  ((-0.0095, 0.0, -(AXZ - FLOOR) + 0.001), (0.015, 0.012, 0.002), STEEL)])

# -- the rod's cradles (fixed): two posts in line with the tube, each with an 11 mm slot the rod lies in. --------------------
REST_X = (-0.040, 0.075)  # its back end stays on the front one even with the boat in the hot zone
_FLARE = math.radians(30.0)
REST_Z = AXZ - HANDLE_W / 2 - 0.0015  # the cradles' tops, 1.5 mm under the rod's grip: its ends must not catch on them
REST = part("rod_cradles", pos=(0.0, 0.0, TOP), fixed=True, friction=0.5,
            boxes=[*[((REST_X[1] + 0.007 + 0.009 * math.cos(_FLARE), e * (0.0095 + 0.009 * math.sin(_FLARE)),
                       REST_Z + 0.004), (0.018, 0.004, 0.008), DARK, e * _FLARE) for e in (-1, 1)],
                   *[((REST_X[1] + 0.007 + 0.009 * math.cos(_FLARE), e * (0.0095 + 0.009 * math.sin(_FLARE)),
                       (REST_Z) / 2), (0.018, 0.004, REST_Z), DARK, e * _FLARE) for e in (-1, 1)],
                   *[b for x in REST_X for b in (
                ((x, 0.0, (REST_Z) / 2), (0.014, 0.020, REST_Z), DARK),
                *[((x, e * 0.0075, REST_Z + 0.004), (0.014, 0.004, 0.008), DARK) for e in (-1, 1)])]])

OBJECTS = [FURNACE, BOAT, *CHIPS, ROD, REST]
PLANCHECK_BOXES = [
    ("furnace", ((FURN_X0 + FURN_X1) / 2, 0.0, TOP + FURN_Z1 / 2), (FURN_X1 - FURN_X0, 2 * FURN_Y, FURN_Z1)),
    ("tube", ((MOUTH_X + TUBE_X1) / 2, 0.0, TOP + AXZ), (TUBE_X1 - MOUTH_X, 2 * TUBE_R, 2 * TUBE_R)),
    ("flange", ((FL_X0 + MOUTH_X) / 2, 0.0, TOP + AXZ), (MOUTH_X - FL_X0, 2 * FL_R, 2 * FL_R)),
]

# -- scoring ----------------------------------------------------------------------------------------------------------
CHIP_SLIDE = 0.003  # a chip that moves over 3 mm on the boat has slid (lost or scratched)
CHIP_FLIP = 30.0  # or tilted over 30 degrees
BOAT_TILT = 20.0
HOT_TOL = 0.010  # the boat centred on the hot zone within 10 mm
RACK_TOL = 0.015  # the rod back in its cradles: within 15 mm of where it lay, hook out of the tube
HANDS_FAR = 0.08
SETTLE_STEPS, SETTLE_TOL = 20, 0.001

_state = {}
_hist = []


@on_reset
def _reset_state(env):
    _state.clear()
    _state.update(step=0, viol=set(), settled=False, far=True, centred=False, racked=False, boat_x=0.0)
    _hist.clear()


def chip_offsets(b, chips):
    """Each chip's place and tilt on the boat: (offset from its starting place on the boat (m), tilt (degrees))."""
    Rb = b.rotation()
    out = []
    for i, c in enumerate(chips):
        q = Rb.T @ (np.asarray(c.pos, float) - np.asarray(b.pos, float))
        q0 = np.array([CHIP_DX[i], 0.0, BOAT_T + CHIP_T / 2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float((Rb.T @ c.rotation())[2, 2])))))
        out.append((float(np.linalg.norm(q - q0)), tilt))
    return out


def rod_racked(r):
    p = np.asarray(r.pos, float)
    return (float(np.linalg.norm(p - ROD_START)) <= RACK_TOL and p[0] < MOUTH_X - 0.001
            and labkit_tilt(r) <= 5.0)


def labkit_tilt(r):
    """How far the rod's axis is off level and off the tube's line (degrees)."""
    a = r.rotation()[:, 0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a[0])))))


@on_step
def _watch(env):
    """Latch the violations (a chip slid or flipped on the boat, the boat tipped) and follow the end state."""
    S = labkit._SPECS
    st = _state
    st["step"] += 1
    b, r = labkit._part_state(env, S["boat"]), labkit._part_state(env, S["hook_rod"])
    chips = [labkit._part_state(env, S[f"chip{i}"]) for i in range(3)]
    for d, tilt in chip_offsets(b, chips):
        if d > CHIP_SLIDE:
            st["viol"].add("a chip slid on the boat")
        if tilt > CHIP_FLIP:
            st["viol"].add("a chip flipped")
    if b.tilt_deg() > BOAT_TILT:
        st["viol"].add("the boat tipped")
    bp = np.asarray(b.pos, float)
    st["boat_x"] = float(bp[0])
    st["centred"] = abs(bp[0] - HOT_X) <= HOT_TOL and abs(bp[1]) <= 0.005
    st["racked"] = rod_racked(r)
    tips = _tips(env)
    st["far"] = all(np.linalg.norm(np.asarray(t) - np.array([MOUTH_X, 0.0, TOP + AXZ])) >= HANDS_FAR
                    and np.linalg.norm(np.asarray(t) - np.asarray(r.pos)) >= HANDS_FAR for t in tips)
    _hist.append(np.concatenate([bp, np.asarray(r.pos, float), *[np.asarray(c.pos, float) for c in chips]]))
    del _hist[:-(SETTLE_STEPS + 1)]
    st["settled"] = (len(_hist) > SETTLE_STEPS
                     and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 after any violation (a chip slid over 3 mm on the boat or tilted over 30 degrees, the boat tipped over 20
    degrees), or unless the boat, chips and rod have been still (under 1 mm) for the last second. Otherwise 0 unless the
    boat is centred on the hot zone (within 10 mm); 0.5 for that; 1.0 if also the rod lies back in its cradles (within
    15 mm of where it started, its hook out of the tube) and every fingertip is 8 cm from the rod and the tube's mouth."""
    st = _state
    if st.get("viol") or not st.get("settled", False) or not st.get("centred"):
        return 0.0
    if not st.get("racked"):
        return 0.5
    return 1.0 if st.get("far", True) else 0.75


SUCCESS_HOLD_S = 2.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return float(abs(np.asarray(objects["boat"].pos, float)[0] - HOT_X))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The hands feed the rod in turn: right takes it near its back, pushes it 12 cm in and lets go (it rests in its cradles
# and on its hook); left takes it near its back and pushes on until the rod's red mark meets the tube's end (seen).
# Then back out the same way: left pulls it one stroke and lets go, right pulls it home into its cradles; both away.
# Aims are trimmed only by what a camera sees: the rod's position and its mark.
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


LEFT_PARK = (0.20, 0.22, TOP + 0.22)
RIGHT_PARK = (0.20, -0.22, TOP + 0.22)
ROD_OPEN, ROD_HOLD = -0.012, -0.0030  # round the rod's 8 mm square grip, squeezed firmly
GRIP_DZ = -0.008  # the fingertips' ends this far below the rod's axis (well past it, or a squeeze pops it out)
FEED_X0, FEED_X1 = 0.110, 0.2433  # each stroke, the hand moves along the rod's line from here to here
H_IN = HOT_X - BOAT_L / 2  # the hook at the hot zone (the boat's back wall)
X_AT = {"right": -0.0, "left": 0.0}

# trims: each a 3-vector added to the TCPs of _Dyn waypoints with its key; integrated in windows from seen positions
_ct = {"trim": {}, "wins": [], "last": {}}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}


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
        _ct["trim"][key] = np.clip(tr + np.clip(0.2 * e, -0.0003, 0.0003), -0.012, 0.012)


class _Dyn:
    """A TCP plus the live trim ``key``."""

    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)


def _down(p, yaw):
    """TCP and rotation for straight-down fingertips ending at point p, the fingers closing along Rz(yaw) y."""
    return tuple(np.asarray(p, float) + [0.0, 0.0, TIP_ABOVE_TCP]), Orient(_prev_grip(yaw))


def _on_rod(x, up=0.0):
    """TCP and rotation for a hand over the rod's line at x, fingers either side of the rod, ``up`` above the grip."""
    return _down((x, 0.0, TOP + AXZ + GRIP_DZ + up), 0.0)


def demo():
    _ct["trim"], _ct["wins"] = {}, []
    pl = _Plan()
    O = _on_rod(FEED_X0)[1]
    park = {"right": RIGHT_PARK, "left": LEFT_PARK}
    stroke = FEED_X1 - FEED_X0

    def take(side, x, key):
        """Over the rod at x, down round it, close."""
        return [(2.0, _Dyn(_on_rod(x, 0.015)[0], key), ROD_OPEN, "high over the rod", O),
                (1.5, _Dyn(_on_rod(x, 0.0)[0], key), ROD_OPEN, "down round the rod", O),
                (0.6, _Dyn(_on_rod(x, 0.0)[0], key), ROD_HOLD, "close on the rod", O)]

    def leave(side, x, key):
        return [(0.6, _Dyn(_on_rod(x, 0.0)[0], key), ROD_OPEN, "let go of the rod", O),
                (1.0, _Dyn(_on_rod(x, 0.015)[0], key), ROD_OPEN, "up off the rod", O),
                (1.5, park[side], ROD_OPEN, "aside", O)]

    def move(x0, x1, key, label, dur=6.0):
        return [(dur, _Dyn(_on_rod(x1, 0.0)[0], key), ROD_HOLD, label, O)]

    # -- 1. right takes the rod near its back and pushes it in one stroke (slowly: the chips must not slide), then lets go:
    # the rod rests in its cradles and on its hook in the tube -------------------------------------------------------------
    pl.phase(right=take("right", FEED_X0, "r1"), left=[(2.0, LEFT_PARK, ROD_OPEN, "wait", O)])
    pl.phase(right=move(FEED_X0, FEED_X1, "r1", "push the rod in"))
    pl.phase(right=leave("right", FEED_X1, "r1"))
    # -- 2. left takes it near its back and pushes on until the red mark meets the tube's end (seen) ----------------------
    pl.phase(left=take("left", FEED_X0, "l1"))
    tw = pl.phase(left=move(FEED_X0, FEED_X1, "l1", "push on until the red mark meets the tube's end"))
    pl.phase(left=[(2.0, _Dyn(_on_rod(FEED_X1, 0.0)[0], "l1"), ROD_HOLD, "hold still: the mark seen at the tube's end",
                    O)])
    pl.window(tw + 6.0, tw + 8.0, "l1",
              lambda env: np.array([MOUTH_X - (_seen(env, "hook_rod")[0] - MARK_AT), -_seen(env, "hook_rod")[1], 0.0]),
              "hook_rod")
    pl.phase(left=[(1.0, _Dyn(_on_rod(FEED_X1, 0.0)[0], "l1"), ROD_HOLD, "the boat in the hot zone", O)])
    # -- 3. back out: left pulls the rod out one stroke and lets go; right takes it and pulls it home into its cradles ------
    pl.phase(left=move(FEED_X1, FEED_X0, "l1", "pull the rod back out"))
    pl.phase(left=leave("left", FEED_X0, "l1"))
    pl.phase(right=take("right", FEED_X1, "r2"))
    tw = pl.phase(right=[(1.5, _Dyn(_on_rod(FEED_X1, 0.0)[0], "r2"), ROD_HOLD, "hold still: the rod seen in line with its "
                          "cradles", O)])
    pl.window(tw, tw + 1.5, "r2", lambda env: np.array([0.0, -_seen(env, "hook_rod")[1], 0.0]), "hook_rod")
    pl.phase(right=move(FEED_X1, FEED_X0, "r2", "pull it back into its cradles"))
    pl.phase(right=leave("right", FEED_X0, "r2"))
    pl.phase(right=[(2.0, RIGHT_PARK, FINGER_CLOSED, "rest", 0.0)], left=[(2.0, LEFT_PARK, FINGER_CLOSED, "rest", 0.0)])
    return {"left": pl.L, "right": pl.R}
