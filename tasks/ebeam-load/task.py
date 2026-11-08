"""Loading an e-beam evaporator's load lock. On a loading stand lies the evaporator's aluminium sample chuck, face up,
with a patterned 100 mm wafer on it (pattern up) and six swing clips round it, all parked open. The wafer's primary flat
faces one of the clips. Also on the chuck: a 10 mm chip taped down by its corners (Kapton), and a 10 mm chip mounted on
a 3 mm glass block. Beside the stand, the load lock's transfer fork waits with its lid open, and a label card reads LOAD
LOCK / MAX 1.5 MM: nothing on the chuck may stand more than 1.5 mm proud, or it jams during the transfer.

The task, as in a Lesker PVD load-lock SOP: take off the chip on the glass block (too tall), clip the wafer by its edge
(the clips must bite on the round edge, not on the flat, and enough of them to hold it face down), then turn the chuck
over so the samples face down (towards the sources) and set it on the transfer fork.

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

SI_BACK = (0.58, 0.59, 0.62)
DIE = (0.20, 0.22, 0.42)
STREET = (0.62, 0.60, 0.48)
ALU = (0.74, 0.75, 0.77)
STAND_C = (0.42, 0.43, 0.46)
STEEL = (0.72, 0.73, 0.75)
DARK = (0.20, 0.21, 0.23)
KAPTON = (0.78, 0.50, 0.12)
GLASS = (0.80, 0.90, 0.86)
CARD = (0.98, 0.98, 0.96)
INK = (0.03, 0.03, 0.04)
PTFE = (0.93, 0.93, 0.90)
labkit.GLASS_COLORS.update({GLASS})
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

FONT = {  # 5 x 7, rows top to bottom
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    " ": ["00000"] * 7,
}


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


def _card_text(lines, pix, z):
    """Letters printed on a card lying flat, read from the robots' side: lines run along -y, letters' tops towards +x."""
    out = []
    for k, line in enumerate(lines):
        v0 = (len(lines) - 1) / 2 * 8 * pix - k * 8 * pix
        out += [((v0 + v, -u, z), (pix * 0.96, pix * 0.96, 0.0006)) for u, v in _glyphs(line, pix)]
    return out


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


# -- the wafer (loose): 100 mm silicon, drawn 1 mm thick, pattern up, with its 32.5 mm primary flat. Frame: origin at the
# mid-plane centre, +z out of the pattern side, the flat's middle on +x. ------------------------------------------------
W_R, W_T = 0.0508, 0.0010
PRI_L = 0.0325
FLAT_D = math.sqrt(W_R ** 2 - (PRI_L / 2) ** 2)  # the flat's distance from the centre
EDGE_EXCL = 0.005


def _outline():
    pts = [p for p in _disc(W_R, 144) if p[0] <= FLAT_D + 1e-9]
    pts += [(FLAT_D, PRI_L / 2), (FLAT_D, -PRI_L / 2)]
    pts.sort(key=lambda p: math.atan2(p[1], p[0]))
    return np.array(pts, float)


def wafer_inside(p):
    """How far the point p (wafer frame, x and y) lies inside the wafer's outline (negative: outside)."""
    return min(W_R - math.hypot(p[0], p[1]), FLAT_D - p[0])


def _pattern():
    r = W_R - EDGE_EXCL
    out = [_prism(_disc(r), W_T / 2, W_T / 2 + 0.00008, DIE)]
    lines = []
    for k in range(-4, 5):
        u = 0.010 * k + 0.005
        half = math.sqrt(max(r * r - u * u, 0.0)) - 0.0005
        if half > 0.002:
            lines += [((u, 0.0, W_T / 2 + 0.00012), (0.0006, 2 * half, 0.00004)),
                      ((0.0, u, W_T / 2 + 0.00012), (2 * half, 0.0006, 0.00004))]
    out.append(_boxes_mesh(lines, STREET))
    return out


def _strips():
    """The wafer collides as 16 thin strips across its centre, each end 1 mm inside the outline (inside the flat too):
    a thin convex mesh sinks through thin supports, and strip faces give the clips a contact patch."""
    out = []
    for k in range(16):
        a = math.pi * k / 16
        d = np.array([math.cos(a), math.sin(a)])
        ends = []
        for s in (1, -1):
            t = W_R
            while wafer_inside(s * t * d) < 0.001:
                t -= 0.0002
            ends.append(t)
        mid = (ends[0] - ends[1]) / 2
        out.append(((mid * d[0], mid * d[1], 0.0), (ends[0] + ends[1], 0.010, W_T * 0.9), SI_BACK, a))
    return out


# -- the chuck (loose): an aluminium sample chuck 150 mm across and 3 mm thick, face up on the loading stand. Round its
# 100 mm wafer seat stand six swing clips every 60 degrees (pivots 59.5 mm out): each is a 12 mm steel tab on a screw
# post, parked along the rim (tangential, joint 0) and swung in about the post (to 90 degrees, its stop) to lay its tip
# 2.3 mm over a round edge. On the chuck also: a 10 mm chip with Kapton tape on its four corners, and a 10 mm chip on a 3
# mm glass block (loose). Frame: origin at the centre of the chuck's back face, +z out of its sample face. -------------
CH_R, CH_T = 0.075, 0.003
CLIP_P, CLIP_L, CLIP_W, CLIP_T = 0.0595, 0.012, 0.004, 0.0010
TAB_Z = CH_T + W_T + 0.0003 + CLIP_T / 2  # the tabs' mid-plane: 0.3 mm over the wafer
KNOB_AT, KNOB_R, KNOB_H = 0.007, 0.0020, 0.008
CLIP_DEG = (0, 60, 120, 180, 240, 300)
FLAT_AT = 60.0  # the wafer's flat faces the 60 degree clip
TAPED_AT, BLOCK_AT, ON_R = 150.0, 210.0, 0.067


def clip_frame(k):
    """Pivot, parked (tangential) direction and inward direction of clip k, in the chuck frame (x, y)."""
    f = math.radians(CLIP_DEG[k])
    P = CLIP_P * np.array([math.cos(f), math.sin(f)])
    return P, np.array([-math.sin(f), math.cos(f)]), -np.array([math.cos(f), math.sin(f)])


def clip_tip(k, th):
    P, t, n = clip_frame(k)
    return P + CLIP_L * (math.cos(th) * t + math.sin(th) * n)


def _clip_joint(k):
    P, t, _ = clip_frame(k)
    yaw = math.atan2(t[1], t[0])
    c = P + t * CLIP_L / 2
    kb = P + t * KNOB_AT
    return Joint(f"clip{k}", kind="hinge", axis="Z", anchor=(P[0], P[1], TAB_Z), lower=0.0, upper=math.pi / 2,
                 initial=0.0, friction=0.0015, damping=0.001, mass=0.002,
                 boxes=[((c[0], c[1], TAB_Z), (CLIP_L + 0.002, CLIP_W, CLIP_T), STEEL, yaw)],
                 cylinders=[((kb[0], kb[1], TAB_Z + CLIP_T / 2 + KNOB_H / 2), KNOB_R, KNOB_H, DARK)])


def _on(deg, r=ON_R):
    a = math.radians(deg)
    return np.array([r * math.cos(a), r * math.sin(a)])


_tp = _on(TAPED_AT)
_tape = [((_tp[0] + sx * 0.0055, _tp[1] + sy * 0.0055, CH_T + 0.00055), (0.004, 0.004, 0.0001), math.radians(45))
         for sx in (-1, 1) for sy in (-1, 1)]
STAND_XY, STAND_H = (0.20, 0.0), 0.130
BOSS_R, BOSS_H = 0.020, 0.003  # a locating boss under the chuck's centre, which sits in the stand's socket
CHUCK_START = np.array([*STAND_XY, TOP + STAND_H])
CHUCK = part(
    "chuck", pos=tuple(CHUCK_START), mass=0.14, friction=0.9,
    cylinders=[((0.0, 0.0, CH_T / 2), CH_R, CH_T, ALU), ((0.0, 0.0, -BOSS_H / 2), BOSS_R, BOSS_H, ALU),
               *[((*clip_frame(k)[0], (CH_T + TAB_Z) / 2), 0.0022, TAB_Z - CH_T, STEEL) for k in range(6)],
               *[((*clip_frame(k)[0], TAB_Z + CLIP_T / 2 + 0.0006), 0.0025, 0.0012, STEEL) for k in range(6)]],
    boxes=[((*_tp, CH_T + 0.00025), (0.010, 0.010, 0.0005), SI_BACK, math.radians(TAPED_AT))],
    meshes=[_boxes_mesh(_tape, KAPTON)],
    joints=[_clip_joint(k) for k in range(6)],
)
WAFER_START = CHUCK_START + [0.0, 0.0, CH_T + W_T / 2]
WAFER = part("wafer", pos=tuple(WAFER_START), yaw=math.radians(FLAT_AT), mass=0.010, friction=1.0,
             boxes=_strips(), meshes=[_prism(_outline(), -W_T / 2, W_T / 2, SI_BACK), *_pattern()])
_bp = _on(BLOCK_AT)
BLOCK_H = 0.003
BLOCK = part("chip_block", pos=(CHUCK_START[0] + _bp[0], CHUCK_START[1] + _bp[1], CHUCK_START[2] + CH_T),
             yaw=math.radians(BLOCK_AT), mass=0.0012, friction=1.0,
             boxes=[((0.0, 0.0, BLOCK_H / 2), (0.010, 0.010, BLOCK_H), GLASS),
                    ((0.0, 0.0, BLOCK_H + 0.00025), (0.010, 0.010, 0.0005), SI_BACK)])
STAND = part("loading_stand", pos=(*STAND_XY, TOP), fixed=True, friction=0.9,
             boxes=[((0.0, 0.0, (STAND_H - 0.009) / 2), (0.050, 0.050, STAND_H - 0.009), STAND_C),
                    ((0.0, 0.0, STAND_H - BOSS_H - 0.003), (0.120, 0.100, 0.006), STAND_C),
                    # four socket blocks round the chuck's boss: the chuck's back face rests on them
                    *[((0.0285 * math.cos(a), 0.0285 * math.sin(a), STAND_H - BOSS_H / 2), (0.016, 0.016, BOSS_H),
                       STAND_C, a) for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]])

# -- the tray (fixed): a PTFE tray for samples that are not going in, front right. --------------------------------------
TRAY_XY = (0.13, -0.17)
TRAY = part("sample_tray", pos=(*TRAY_XY, TOP), fixed=True, friction=0.9,
            boxes=[((0.0, 0.0, 0.002), (0.080, 0.080, 0.004), PTFE)])

# -- the load lock (fixed): its open bay, simplified to a base plate, the transfer fork (two rails 141 mm apart, their
# tops 130 mm above the bench, that carry a chuck face down by its rim) and the chamber wall behind; with a label card
# on the bench beside it. -------------------------------------------------------------------------------------------
LL_XY = (0.37, 0.0)
RAIL_TOP, RAIL_X = 0.130, 0.0705
LOADLOCK = part(
    "load_lock", pos=(*LL_XY, TOP), fixed=True, friction=0.8,
    boxes=[((0.0, 0.0, 0.003), (0.200, 0.260, 0.006), DARK),
           *[((e * RAIL_X, 0.0, RAIL_TOP - 0.003), (0.009, 0.080, 0.006), STEEL) for e in (-1, 1)],
           *[((e * RAIL_X, 0.0, (RAIL_TOP - 0.006) / 2 + 0.003), (0.009, 0.020, RAIL_TOP - 0.012), STEEL)
             for e in (-1, 1)],
           ((0.125, 0.0, 0.100), (0.040, 0.260, 0.200), STEEL)],
)
CARD_XY, CARD_SIZE, CARD_PIX = (0.37, -0.20), (0.075, 0.150, 0.002), 0.0026
LABEL = part("height_label", pos=(*CARD_XY, TOP), fixed=True,
             boxes=[((0.0, 0.0, CARD_SIZE[2] / 2), CARD_SIZE, CARD)],
             meshes=[_boxes_mesh(_card_text(("LOAD LOCK", "MAX 1.5 MM"), CARD_PIX, CARD_SIZE[2] + 0.0003), INK)])
MAX_PROUD = 0.0015

OBJECTS = [STAND, CHUCK, WAFER, BLOCK, TRAY, LOADLOCK, LABEL]
PLANCHECK_BOXES = [
    ("load_lock_wall", (LL_XY[0] + 0.125, 0.0, TOP + 0.100), (0.040, 0.260, 0.200)),
    *[(f"rail{e}", (LL_XY[0] + e * RAIL_X, 0.0, TOP + RAIL_TOP / 2), (0.009, 0.080, RAIL_TOP)) for e in (-1, 1)],
    ("stand_plate", (*STAND_XY, TOP + STAND_H - BOSS_H - 0.003), (0.120, 0.100, 0.006)),
]

# -- scoring ----------------------------------------------------------------------------------------------------------
CLIP_IN = math.radians(80.0)  # a clip swung in past 80 degrees
BITE = 0.0015  # and its tip at least 1.5 mm inside the wafer's outline: it holds the wafer
GAP_MAX = 150.0  # the holding clips' largest angular gap round the wafer (degrees): under 150, the wafer cannot drop
SEAT_XY, SEAT_Z, SEAT_DEG = 0.004, 0.003, 5.0
TOUCH = 0.004
HANDS_FAR = 0.08
SETTLE_STEPS, SETTLE_TOL = 20, 0.001
_state = {}
_hist = []


@on_reset
def _reset_state(env):
    _state.update(dropped=False, block_fell=False, touched=False, far=True, settled=False, block_out=False,
                  gap=360.0, on_fork=False)
    _hist.clear()


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


def wafer_in_chuck(c, w):
    """The wafer's centre in the chuck's frame."""
    return c.rotation().T @ (np.asarray(w.pos, float) - np.asarray(c.pos, float))


def wafer_on(c, w):
    q = wafer_in_chuck(c, w)
    return math.hypot(q[0], q[1]) <= 0.006 and CH_T - 0.001 <= q[2] <= CH_T + 0.004


def holding_clips(c, w):
    """The clips swung in whose tips bite at least 1.5 mm inside the wafer's outline (chuck-frame angles, degrees)."""
    Rc, Rw = c.rotation(), w.rotation()
    out = []
    for k in range(6):
        th = float(c.joints.get(f"clip{k}", 0.0))
        if th < CLIP_IN:
            continue
        tip = clip_tip(k, th)
        world = np.asarray(c.pos, float) + Rc @ np.array([tip[0], tip[1], TAB_Z])
        p = Rw.T @ (world - np.asarray(w.pos, float))
        if wafer_inside(p) >= BITE:
            out.append(CLIP_DEG[k])
    return out


def largest_gap(degs):
    if not degs:
        return 360.0
    s = sorted(degs)
    return max((s[(i + 1) % len(s)] - s[i]) % 360 or 360.0 for i in range(len(s)))


def on_fork(c):
    R, p = c.rotation(), np.asarray(c.pos, float)
    return (tilt_deg(R, down=True) <= SEAT_DEG and math.hypot(p[0] - LL_XY[0], p[1] - LL_XY[1]) <= SEAT_XY
            and abs(p[2] - (TOP + RAIL_TOP + CH_T)) <= SEAT_Z)


def block_out(b):
    """The tall chip is off the chuck and out of the load lock, lying on the bench or the tray."""
    p = np.asarray(b.pos, float)
    off_chuck = math.hypot(p[0] - STAND_XY[0], p[1] - STAND_XY[1]) > CH_R + 0.010
    off_ll = not (abs(p[0] - LL_XY[0]) < 0.12 and abs(p[1] - LL_XY[1]) < 0.14)
    return off_chuck and off_ll and p[2] < TOP + 0.02 and tilt_deg(b.rotation()) < 10.0


@on_step
def _watch(env):
    """Latch the violations: the wafer off its seat on the chuck (dropped), the glass block tipped over (fell), a
    fingertip at the wafer's pattern."""
    S = labkit._SPECS
    c, w, b = (labkit._part_state(env, S[n]) for n in ("chuck", "wafer", "chip_block"))
    if not wafer_on(c, w):
        _state["dropped"] = True
    if tilt_deg(b.rotation()) > 45.0:
        _state["block_fell"] = True
    tips = _tips(env)
    Rw, pw = w.rotation(), np.asarray(w.pos, float)
    for tip in tips:
        q = Rw.T @ (np.asarray(tip, float) - pw)
        if abs(q[2] - W_T / 2) < TOUCH and math.hypot(q[0], q[1]) < W_R - EDGE_EXCL:
            _state["touched"] = True
    _state["block_out"] = block_out(b)
    _state["gap"] = largest_gap(holding_clips(c, w))
    _state["on_fork"] = on_fork(c)
    cp = np.asarray(c.pos, float)
    _state["far"] = all(np.linalg.norm(np.asarray(t) - cp) >= HANDS_FAR for t in tips)
    _hist.append(np.concatenate([cp, pw, np.asarray(b.pos, float)]))
    del _hist[:-(SETTLE_STEPS + 1)]
    _state["settled"] = (len(_hist) > SETTLE_STEPS
                         and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 if the wafer ever left its seat on the chuck, the chip's glass block tipped over, a fingertip came within 4 mm
    of the wafer's pattern, or the chuck, wafer and block have not been still (under 1 mm) for the last second. Otherwise:
    0.25 if the chip on the glass block is off the chuck, out of the load lock, lying on the bench or tray; 0.5 if also
    the clips holding the wafer (swung in past 80 degrees, tips at least 1.5 mm inside its outline) leave no gap over
    150 degrees round it; 0.75 if also the chuck lies face down on the transfer fork (within 5 degrees, centre within 4
    mm); 1.0 if also every fingertip is 8 cm from the chuck's centre."""
    if any(_state.get(k) for k in ("dropped", "block_fell", "touched")):
        return 0.0
    if not _state.get("settled", False):
        return 0.0
    if not _state.get("block_out"):
        return 0.0
    if _state.get("gap", 360.0) >= GAP_MAX:
        return 0.25
    if not _state.get("on_fork"):
        return 0.5
    return 1.0 if _state.get("far", True) else 0.75


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    p = np.asarray(objects["chuck"].pos, float)
    return float(math.hypot(p[0] - LL_XY[0], p[1] - LL_XY[1]))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right hand: lifts the chip on its glass block off the chuck and sets it on the tray, then swings the 240 degree clip in.
# Left hand: swings the 0 and 120 degree clips in (the 60 degree one faces the wafer's flat and could not hold it; the
# three at 120 degree spacing can). Then both hands pinch the chuck's side edges, lift it, roll it over together (samples
# face down), carry it over the transfer fork, trim their aim by the chuck's seen position, set it down, let go and back
# out. The demo reads only what a camera sees: the chuck's pose.
_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)
        return _prev(self, t)

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
LEFT_PARK = (0.15, 0.25, TOP + 0.16)
RIGHT_PARK = (0.15, -0.25, TOP + 0.16)
OPEN_W = -0.012
PINCH = -0.0008  # squeezing the 3 mm chuck
BLOCK_OPEN, BLOCK_HOLD = -0.012, -0.0040  # round the 10 mm block
PAD_IN = 0.008  # the pad tips this far in from the chuck's edge
CLIPS_USED = {"right": (4,), "left": (0, 2)}  # 240, 0 and 120 degrees
FLIP = {"right": True, "left": False}  # each hand's grasp (found by an IK search for the mask roll) that can follow
ROLL_SIGN = -1.0
CLIP0_DYAW = 0.0
LIFT_Z = 0.22  # the chuck's centre this high (above the bench) while it is rolled over

_ct = {"trim": np.zeros(3), "wins": [], "last": None}


@on_reset
def _reset_ct(env):
    _ct["trim"] = np.zeros(3)
    _ct["last"] = None


@on_step
def _ctrack(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    win = next((w for w in _ct["wins"] if w[0] <= t < w[1]), None)
    p = np.asarray(labkit._part_state(env, labkit._SPECS["chuck"]).pos, float)
    last, _ct["last"] = _ct["last"], p
    if win is None or last is None or np.max(np.abs(p - last)) > 0.0005:
        return
    e = np.asarray(win[2], float) - p
    e[2] = 0.0
    _ct["trim"] = np.clip(_ct["trim"] + np.clip(0.2 * e, -0.0003, 0.0003), -0.015, 0.015)


class _Dyn:
    """A TCP plus the live trim."""

    def __init__(self, tcp):
        self.tcp = np.asarray(tcp, float)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"], dtype=dtype or float)


def _down(p, yaw):
    """TCP and rotation for straight-down fingertips at point p, the fingers closing along Rz(yaw) y."""
    return tuple(np.asarray(p, float) + [0.0, 0.0, TIP_ABOVE_TCP]), Orient(_prev_grip(yaw))


def _grasp(side, flip):
    a = Y if side == "right" else -Y
    c0 = -Z if flip else Z
    return np.column_stack([c0, np.cross(a, c0), a])


def _hand(side, c, R=np.eye(3), out=0.0):
    """TCP and rotation for ``side``'s hand pinching the chuck's side edge, the chuck's mid-plane centre at ``c`` and
    its attitude R (rolled about y); ``out`` backs the hand out sideways."""
    Rh = R @ _grasp(side, FLIP[side])
    a = Rh[:, 2]
    tip = np.asarray(c, float) - a * (CH_R - PAD_IN + out)
    return tuple(tip - a * TIP_ABOVE_TCP), Orient(Rh)


def _w(p):
    return np.array([STAND_XY[0] + p[0], STAND_XY[1] + p[1]])


def demo():
    L, R = [], []
    _ct["trim"], _ct["wins"] = np.zeros(3), []
    clock = [0.0]

    def ph(dur, r=None, l=None, see=None):
        for plan, w in ((R, r), (L, l)):
            plan.append(Waypoint(dur, *w) if w else Waypoint(dur, plan[-1].pos, plan[-1].finger, "hold", plan[-1].yaw))
        if see is not None:
            _ct["wins"].append((clock[0], clock[0] + dur, np.asarray(see, float)))
        clock[0] += dur

    top = TOP + STAND_H + CH_T  # the chuck's sample face
    knob_z = top + (TAB_Z - CH_T) + CLIP_T / 2 + 0.0008  # the fingertips just over the tab, low on the knob

    def knob(k, th, back=None):
        """The fingertips' point behind clip k's knob (against its swing) at swing angle th: the closed fingertips are 16
        mm wide across and 8 mm thick along their closing axis, which lies along the clip's parked direction."""
        if back is None:
            back = KNOB_R + 0.0015 + 0.0078 * abs(math.cos(th)) + 0.004 * abs(math.sin(th))
        P, t, n = clip_frame(k)
        kp = P + KNOB_AT * (math.cos(th) * t + math.sin(th) * n)
        d = -math.sin(th) * t + math.cos(th) * n
        return np.array([*_w(kp - back * d), knob_z])

    def push(k, side_label, dyaw=0.0):
        yaw = math.radians(CLIP_DEG[k] + dyaw)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        if abs(yaw) > math.pi / 2:
            yaw -= math.copysign(math.pi, yaw)
        s = [_down(knob(k, 0.0) + [0, 0, 0.03], yaw), _down(knob(k, 0.0), yaw)]
        s += [_down(knob(k, math.radians(a)), yaw) for a in (35.0, 70.0, 100.0)]
        return [(1.5, (s[0][0], FINGER_CLOSED, f"above the {side_label} clip, outside its knob", s[0][1])),
                (1.0, (s[1][0], FINGER_CLOSED, "fingertips down beside the knob", s[1][1])),
                (1.0, (s[2][0], FINGER_CLOSED, "swing the clip in", s[2][1])),
                (1.0, (s[3][0], FINGER_CLOSED, "swing the clip in", s[3][1])),
                (1.0, (s[4][0], FINGER_CLOSED, "against its stop: the tip over the wafer's edge", s[4][1])),
                (0.8, (tuple(np.asarray(s[4][0]) + [0, 0, 0.03]), FINGER_CLOSED, "up", s[4][1]))]

    # the block: right hand, fingers closing across it along the rim (tangential)
    bp = np.array([*_w(_bp), top + BLOCK_H / 2])
    byaw = math.radians(BLOCK_AT) - math.pi
    bt, bo = _down(bp, byaw)
    tray = np.array([*TRAY_XY, TOP + 0.004 + BLOCK_H / 2])
    tt, _ = _down(tray, byaw)
    block_seq = [
        (2.0, (tuple(np.asarray(bt) + [0, 0, 0.04]), BLOCK_OPEN, "above the chip on its glass block", bo)),
        (1.2, (bt, BLOCK_OPEN, "fingers either side of the block", bo)),
        (0.8, (bt, BLOCK_HOLD, "hold it", bo)),
        (1.0, (tuple(np.asarray(bt) + [0, 0, 0.04]), BLOCK_HOLD, "lift it off the chuck: too tall to go in", bo)),
        (2.0, (tuple(np.asarray(tt) + [0, 0, 0.04]), BLOCK_HOLD, "over the tray", bo)),
        (1.0, (tt, BLOCK_HOLD, "set it down", bo)),
        (0.6, (tt, BLOCK_OPEN, "let go", bo)),
        (0.8, (tuple(np.asarray(tt) + [0, 0, 0.05]), BLOCK_OPEN, "up", bo)),
    ]
    right_seq = block_seq + push(4, "240 degree") + [
        (1.5, (_down((0.20, -0.17, TOP + 0.20), 0.0)[0], FINGER_CLOSED, "aside, out of the left hand's way",
               _down((0.2, -0.17, 0.0), 0.0)[1]))]
    p0 = push(0, "0 degree", CLIP0_DYAW)
    # the left hand waits, up, while the right hand works in front of the chuck, then swings the 120 degree clip in
    aside = (1.5, (_down((0.20, 0.17, TOP + 0.20), 0.0)[0], FINGER_CLOSED, "aside, out of the right hand's way",
                   _down((0.2, 0.17, 0.0), 0.0)[1]))
    # one hand at a time over the chuck: left 0 degree clip; right block and 240 degree clip; left 120 degree clip
    left_seq = p0 + [aside] + [(w[0], None) for w in right_seq] + push(2, "120 degree")
    wait_r = (p0[0][0], (RIGHT_PARK, FINGER_CLOSED, "wait, out of the left hand's way", 0.0))
    right_seq = [wait_r] + [(w[0], None) for w in p0[1:]] + [(1.5, None)] + right_seq
    n = max(len(left_seq), len(right_seq))
    for i in range(n):
        lw = left_seq[i] if i < len(left_seq) else (0.0, None)
        rw = right_seq[i] if i < len(right_seq) else (0.0, None)
        ph(max(lw[0], rw[0]), rw[1], lw[1])

    c0 = np.array([*STAND_XY, TOP + STAND_H + CH_T / 2])
    lift = np.array([*STAND_XY, TOP + LIFT_Z])

    def both(dur, c, finger, label, Rm=np.eye(3), out=0.0, trim=False, see=None):
        lt, lo = _hand("left", c, Rm, out)
        rt, ro = _hand("right", c, Rm, out)
        if trim:
            lt, rt = _Dyn(lt), _Dyn(rt)
        ph(dur, (rt, finger, label, ro), (lt, finger, label, lo), see=see)

    both(4.0, c0 + [0, 0, 0.03], OPEN_W, "beside the chuck's side edges, above", out=0.04)
    both(1.5, c0, OPEN_W, "level with its edges", out=0.04)
    both(1.5, c0, OPEN_W, "pads round its side edges, where they overhang the stand")
    both(0.8, c0, PINCH, "pinch them")
    both(1.5, c0 + [0, 0, 0.03], PINCH, "lift the chuck off the stand")
    both(2.0, lift, PINCH, "up, clear of everything")
    for k in range(1, 9):
        both(0.8, lift, PINCH, "roll it over together: samples face down", axis_angle_matrix(Y, ROLL_SIGN * math.pi * k / 8))
    Rf = axis_angle_matrix(Y, ROLL_SIGN * math.pi)
    seat = np.array([*LL_XY, TOP + RAIL_TOP + CH_T / 2])
    above = seat + [0, 0, 0.035]
    both(3.0, above, PINCH, "carry it over the transfer fork", Rf)
    both(2.0, above, PINCH, "hold it still, seen over the fork", Rf, trim=True,
         see=above + [0, 0, CH_T / 2])
    both(2.0, seat + [0, 0, 0.002], PINCH, "set it down on the fork's rails", Rf, trim=True)
    both(0.8, seat + [0, 0, 0.0005], OPEN_W, "let go", Rf, trim=True)
    both(1.2, seat, OPEN_W, "back out sideways", Rf, out=0.05, trim=True)
    both(1.5, seat + [0, 0, 0.08], OPEN_W, "up and away", Rf, out=0.05, trim=True)
    ph(2.0)
    return {"left": L, "right": R}


# --- debug (not for submission) ---
_dbgr = {"n": 0}
@on_step
def _dbg_eb(env):
    _dbgr["n"] += 1
    if _dbgr["n"] % 10:
        return
    S = labkit._SPECS
    c, w, b = (labkit._part_state(env, S[n]) for n in ("chuck", "wafer", "chip_block"))
    q = wafer_in_chuck(c, w)
    tp = _tips(env)
    tl = np.round((0.5 * (tp[0] + tp[1]) - [0, 0, TOP]) * 1000, 0).tolist()
    tr_ = np.round((0.5 * (tp[2] + tp[3]) - [0, 0, TOP]) * 1000, 0).tolist()
    cl = [round(math.degrees(float(c.joints.get(f"clip{k}", 0.0)))) for k in range(6)]
    fl = "".join(k[0].upper() if _state.get(k) else "." for k in ("dropped", "block_fell", "touched", "block_out", "on_fork", "far", "settled"))
    open("/home/ubuntu/dbg_eb.txt", "a").write(
        f"{_dbgr['n']} t={getattr(_demo, '_plan_clock', 0.0):.1f} ch={np.round((np.asarray(c.pos) - [0, 0, TOP]) * 1000, 1).tolist()} "
        f"cdn={tilt_deg(c.rotation(), True):.1f} wq={np.round(q * 1000, 1).tolist()} blk={np.round((np.asarray(b.pos) - [0, 0, TOP]) * 1000, 1).tolist()} "
        f"bt={tilt_deg(b.rotation()):.0f} L={tl} R={tr_} clips={cl} hold={holding_clips(c, w)} gap={_state.get('gap')} tr={np.round(_ct['trim'] * 1000, 1).tolist()} {fl} s={score(None)}\n")
