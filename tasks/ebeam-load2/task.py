"""Loading an e-beam evaporator's load lock (after a Lesker PVD load-lock SOP).

The load lock is a closed chamber under vacuum, with a door at its front that slides up. Inside, its holder carries the
ring holder. On a loading stand beside it lies the sample plate (stainless, 200 mm across, for wafers up to 6 inch), face
up, with a 100 mm wafer on it, pattern up and about 5 mm off centre. Round the wafer, in the plate's 4 inch ring of tapped
holes, stand eight swing clips, all parked open with their screws loose. The wafer's primary flat faces one of them, and
one clip has a screw too long for its hole: its head stands 6 mm proud, and tightened, its tip would stick out of the
plate's back and jam the transfer. A screwdriver stands in a holder, and a control box has VENT and PUMP buttons.

The job: vent the load lock and wait for it to reach air before opening the door; take out the ring holder and set it on
the plate; centre the wafer; swing in clips that can hold its round edge, enough of them and not the one with the long
screw, and tighten their screws; turn the plate and ring over together (samples face down, the ring now carrying the
plate) and set them on the load lock's holder; close the door, and hold it shut while pressing PUMP.

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

SI_BACK = (0.58, 0.59, 0.62)
DIE = (0.20, 0.22, 0.42)
STREET = (0.62, 0.60, 0.48)
STAINLESS = (0.60, 0.61, 0.64)
ALU = (0.80, 0.81, 0.83)
STAND_C = (0.42, 0.43, 0.46)
STEEL = (0.72, 0.73, 0.75)
DARK = (0.20, 0.21, 0.23)
HOLE = (0.12, 0.12, 0.13)
CHAMBER = (0.55, 0.57, 0.60)
VIEWPORT = (0.82, 0.90, 0.92)
BOXC = (0.30, 0.32, 0.36)
LABEL_C = (0.97, 0.97, 0.95)
INK = (0.03, 0.03, 0.04)
VENT_C = (0.95, 0.72, 0.08)
PUMP_C = (0.12, 0.62, 0.24)
RED = (0.75, 0.15, 0.12)
labkit.GLASS_COLORS.update({VIEWPORT})
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

FONT = {  # 5 x 7, rows top to bottom
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
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


def _on(deg, r):
    a = math.radians(deg)
    return np.array([r * math.cos(a), r * math.sin(a)])


# -- the wafer (loose): 100 mm silicon, drawn 1 mm thick, pattern up, with its 32.5 mm primary flat. Frame: origin at the
# mid-plane centre, +z out of the pattern side, the flat's middle on +x. ------------------------------------------------
W_R, W_T = 0.0508, 0.0010
PRI_L = 0.0325
FLAT_D = math.sqrt(W_R ** 2 - (PRI_L / 2) ** 2)  # the flat's distance from the centre
EDGE_EXCL = 0.005
STRIP_IN = 0.001  # the wafer's collision strips end this far inside its outline


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
            while wafer_inside(s * t * d) < STRIP_IN:
                t -= 0.0002
            ends.append(t)
        mid = (ends[0] - ends[1]) / 2
        out.append(((mid * d[0], mid * d[1], 0.0), (ends[0] + ends[1], 0.010, W_T * 0.9), SI_BACK, a))
    return out


# -- the sample plate (loose): stainless, 200 mm across, 1.5 mm thick, face up on the loading stand, with rings of tapped
# holes for 2, 3, 4 and 6 inch wafers. In the 4 inch ring (59.5 mm out) stand eight swing clips every 45 degrees. Each is
# a 13 mm steel tab on its screw: parked along the rim (joint 0) it clears the wafer; swung in 90 degrees clockwise to its
# stop, its tip lies 3.3 mm over a round edge, and a lip under it, 8.2 mm from the screw, stops the wafer's edge 0.5 mm
# outside a centred wafer. Each clip's screw (slotted head, 7 mm) turns on the clip: clockwise tightens it. A boss under
# the plate sits in the stand's socket. Frame: origin at the centre of the plate's back face, +z out of its sample face.
PL_R, PL_T = 0.100, 0.0015
CLIP_P, CLIP_L, CLIP_W, CLIP_T = 0.0595, 0.013, 0.004, 0.0010
TAB_BOT = PL_T + W_T + 0.0003  # the tabs' underside: 0.3 mm over the wafer
TAB_Z = TAB_BOT + CLIP_T / 2
TAB_TOP = TAB_BOT + CLIP_T
LIP_D0, LIP_D1 = 0.0024, CLIP_P - (W_R + 0.0005)  # the lip under each tab, from the screw out along the tab
KNOB_AT, KNOB_R, KNOB_H = 0.007, 0.0020, 0.008
HEAD_R, HEAD_H, SLOT, RIDGE = 0.0035, 0.0025, 0.0018, (0.0011, 0.0060, 0.0015)
CLIP_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
FLAT_AT = 90.0  # the wafer's flat faces the 90 degree clip
LONG_AT = 225  # the clip whose screw is too long
LONG_UP = 0.006  # its head stands this far proud
# each screw's slot direction (plate frame, degrees, joint 0, clip parked); the clips are swung in by -90 degrees
SLOT_DEG = {0: 145.0, 45: 20.0, 90: 160.0, 135: 145.0, 180: 75.0, 225: 60.0, 270: 145.0, 315: 110.0}
W_OFF = 0.005 * np.array([math.cos(math.radians(202.5)), math.sin(math.radians(202.5))])  # the wafer starts off centre
SCREW_TIGHT = math.radians(300.0)  # tightened: turned at least 300 degrees clockwise
LONG_TURN = math.radians(90.0)


def clip_frame(k):
    """Screw (pivot), parked direction and inward direction of clip k, in the plate frame (x, y)."""
    f = math.radians(CLIP_DEG[k])
    P = CLIP_P * np.array([math.cos(f), math.sin(f)])
    return P, np.array([math.sin(f), -math.cos(f)]), -np.array([math.cos(f), math.sin(f)])


def clip_dir(k, th):
    """The tab's direction when clip k is swung in by th (clockwise, radians)."""
    _, t, n = clip_frame(k)
    return math.cos(th) * t + math.sin(th) * n


def clip_tip(k, th):
    return clip_frame(k)[0] + CLIP_L * clip_dir(k, th)


def _clip_joints(k):
    P, t, _ = clip_frame(k)
    yaw = math.atan2(t[1], t[0])
    c = P + t * CLIP_L / 2
    lc = P + t * (LIP_D0 + LIP_D1) / 2
    kb = P + t * KNOB_AT
    clip = Joint(f"clip{k}", kind="hinge", axis="Z", anchor=(P[0], P[1], TAB_Z), lower=-math.pi / 2, upper=0.0,
                 initial=0.0, friction=0.008, damping=0.001, mass=0.002,
                 boxes=[((c[0], c[1], TAB_Z), (CLIP_L + 0.002, CLIP_W, CLIP_T), STEEL, yaw),
                        ((lc[0], lc[1], (PL_T + 0.0001 + TAB_BOT) / 2), (LIP_D1 - LIP_D0, CLIP_W, TAB_BOT - PL_T - 0.0001),
                         STEEL, yaw)],
                 cylinders=[((kb[0], kb[1], TAB_TOP + KNOB_H / 2), KNOB_R, KNOB_H, DARK)])
    up = LONG_UP if CLIP_DEG[k] == LONG_AT else 0.0
    s = math.radians(SLOT_DEG[CLIP_DEG[k]])
    nrm = np.array([-math.sin(s), math.cos(s)])  # across the slot
    hz = TAB_TOP + up + HEAD_H / 2
    rz = TAB_TOP + up + HEAD_H + RIDGE[2] / 2
    cyl = [((P[0], P[1], hz), HEAD_R, HEAD_H, STEEL)]
    if up:
        cyl.append(((P[0], P[1], TAB_TOP + up / 2), 0.0019, up, STEEL))
    screw = Joint(f"screw{k}", kind="hinge", axis="Z", anchor=(P[0], P[1], TAB_Z), lower=-7.0, upper=0.05,
                  initial=0.0, friction=0.004, damping=0.0005, mass=0.002, parent=f"clip{k}", cylinders=cyl,
                  boxes=[((P[0] + e * (SLOT / 2 + RIDGE[0] / 2) * nrm[0], P[1] + e * (SLOT / 2 + RIDGE[0] / 2) * nrm[1], rz),
                          RIDGE, STEEL, s - math.pi / 2) for e in (-1, 1)])
    return [clip, screw]


_holes = [((*_on(d, r), PL_T + 0.00005), (0.0042, 0.0042, 0.0001), math.radians(45 + d))
          for r, n, off in ((0.0340, 6, 30.0), (0.0470, 6, 0.0), (0.0850, 8, 22.5)) for d in
          (off + 360.0 * i / n for i in range(n))]
ST_XY, ST_H = (0.18, 0.0), 0.130
BOSS_R, BOSS_H = 0.020, 0.003
PLATE_START = np.array([*ST_XY, TOP + ST_H])
PLATE = part(
    "sample_plate", pos=tuple(PLATE_START), mass=0.36, friction=0.9,
    cylinders=[((0.0, 0.0, PL_T / 2), PL_R, PL_T, STAINLESS), ((0.0, 0.0, -BOSS_H / 2), BOSS_R, BOSS_H, STAINLESS),
               *[((*clip_frame(k)[0], (PL_T + TAB_BOT) / 2), 0.0022, TAB_BOT - PL_T, STEEL) for k in range(8)]],
    meshes=[_boxes_mesh(_holes, HOLE)],
    joints=[j for k in range(8) for j in _clip_joints(k)],
)
WAFER_START = PLATE_START + [*W_OFF, PL_T + W_T / 2]
WAFER = part("wafer", pos=tuple(WAFER_START), yaw=math.radians(FLAT_AT), mass=0.010, friction=0.6,
             boxes=_strips(), meshes=[_prism(_outline(), -W_T / 2, W_T / 2, SI_BACK), *_pattern()])
STAND = part("loading_stand", pos=(*ST_XY, TOP), fixed=True, friction=0.9,
             boxes=[((0.0, 0.0, (ST_H - 0.009) / 2), (0.050, 0.050, ST_H - 0.009), STAND_C),
                    ((0.0, 0.0, ST_H - BOSS_H - 0.003), (0.120, 0.120, 0.006), STAND_C),
                    *[((0.0285 * math.cos(a), 0.0285 * math.sin(a), ST_H - BOSS_H / 2), (0.016, 0.016, BOSS_H),
                       STAND_C, a) for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]])

# -- the ring holder (loose): aluminium, 220 mm across, 180 mm inside, 4 mm thick, with a 1.5 mm recess under its rim
# (its skirt, 101.5 mm in radius) into which the sample plate fits. It starts on the load lock's holder. Frame: origin
# at the centre of its underside (the face that rests on the plate), +z towards its top. -----------------------------
RG_IN, RG_OUT, RG_T, SK_IN = 0.090, 0.110, 0.004, 0.1015
_segs = [5.0 + 10.0 * i for i in range(36)]
RING_BOXES = (_arc_boxes(RG_IN, RG_OUT, 0.0, RG_T, _segs, 0.0200, ALU)
              + _arc_boxes(SK_IN, RG_OUT, -PL_T, 0.0, _segs, 0.0205, ALU))

# -- the load lock (fixed): a closed chamber, 0.76 m wide, its front at x = 0.36 m, with a viewport on top. Its front
# opening (0.75 x 0.20 m, 100 to 300 mm above the bench) is closed by a door that slides up. Inside, on a centre shaft,
# its holder: two arcs (front and back) that carry the ring holder 170 mm above the bench, between low fences. ----------
LL_X0, LL_X1, LL_Y, LL_Z = 0.360, 0.630, 0.380, 0.336
OPEN_Z0, OPEN_Z1 = 0.100, 0.300
HOLD_XY, HOLD_Z = (0.475, 0.0), 0.170
WALL = 0.006
RING_START = np.array([*HOLD_XY, TOP + HOLD_Z])
RING = part("ring_holder", pos=tuple(RING_START), mass=0.14, friction=0.9, boxes=RING_BOXES)
_arcs = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
_cx = LL_X0 + (LL_X1 - LL_X0) / 2
_hold = [((HOLD_XY[0] - _cx, 0.0, (HOLD_Z - 0.030) / 2), (0.030, 0.030, HOLD_Z - 0.030), STEEL),
         ((HOLD_XY[0] - _cx, 0.0, HOLD_Z - 0.027), (0.200, 0.024, 0.006), STEEL)]
for side in (0.0, 180.0):
    for d in _arcs:
        a = math.radians(side + d)
        p = _on(side + d, 0.093)
        h = HOLD_Z - 0.024
        _hold.append(((HOLD_XY[0] - _cx + p[0], p[1], h + 0.012), (0.012, 0.0175, 0.024), STEEL, a))
    for d in (-15.0, 0.0, 15.0):
        a = math.radians(side + d)
        p = _on(side + d, 0.1155)
        _hold.append(((HOLD_XY[0] - _cx + p[0], p[1], (HOLD_Z + 0.004) / 2), (0.004, 0.018, HOLD_Z + 0.004), STEEL, a))
_ll_w = LL_X1 - LL_X0
LL_BOXES = [
    ((-_ll_w / 2 + WALL / 2, 0.0, OPEN_Z0 / 2), (WALL, 2 * LL_Y, OPEN_Z0), CHAMBER),  # front, under the opening
    ((-_ll_w / 2 + WALL / 2, 0.0, (OPEN_Z1 + LL_Z) / 2), (WALL, 2 * LL_Y, LL_Z - OPEN_Z1), CHAMBER),  # over it
    *[((WALL / 2, e * (LL_Y - WALL / 2), LL_Z / 2), (_ll_w, WALL, LL_Z), CHAMBER) for e in (-1, 1)],  # sides
    ((_ll_w / 2 - WALL / 2, 0.0, LL_Z / 2), (WALL, 2 * LL_Y, LL_Z), CHAMBER),  # back
]
LOADLOCK = part("load_lock", pos=(_cx, 0.0, TOP), fixed=True, friction=0.8,
                boxes=[*LL_BOXES, *_hold,
                       ((0.0, 0.0, LL_Z - WALL / 2), (_ll_w - 2 * WALL, 2 * LL_Y - 2 * WALL, WALL), VIEWPORT)])

# the door (fixed frame, a slide joint up): an aluminium plate 6 mm thick in front of the opening, with a 12 mm handle bar
# 25 mm in front of it. Friction holds it where it is left.
DOOR_X, DOOR_Z0, DOOR_Z1 = LL_X0 - 0.005, 0.095, 0.305
HANDLE_X, HANDLE_Z, HANDLE_W = DOOR_X - 0.025, 0.130, 0.012
DOOR_UP = 0.215  # fully open
DOOR_M = 0.25  # kg, counterbalanced: it stays where it is left
DOOR = part(
    "door", pos=(DOOR_X, 0.0, TOP), fixed=True, friction=1.0,
    boxes=[((0.0, 0.0, 0.003), (0.004, 0.004, 0.006), DARK)],
    joints=[Joint("door", kind="slide", axis="Z", anchor=(0.0, 0.0, DOOR_Z0), lower=0.0, upper=DOOR_UP + 0.005,
                  initial=0.0, damping=6.0, mass=DOOR_M,
                  boxes=[((0.0, 0.0, (DOOR_Z0 + DOOR_Z1) / 2), (0.006, 2 * LL_Y, DOOR_Z1 - DOOR_Z0), ALU),
                         ((HANDLE_X - DOOR_X, 0.0, HANDLE_Z), (HANDLE_W, 0.140, HANDLE_W), DARK),
                         *[((-0.0145, e * 0.064, HANDLE_Z), (0.023, 0.010, HANDLE_W), DARK) for e in (-1, 1)]])],
)

# -- the control box (fixed): two push buttons, VENT and PUMP, with their names printed in 21 mm letters below them. ----
BOX_XY, BOX_H = (0.140, 0.220), 0.050
BTN_R, BTN_H, BTN_TRAVEL = 0.012, 0.012, 0.004
BTN = {"vent": (0.165, 0.260), "pump": (0.165, 0.180)}
BTN_PRESSED = 0.0025
_lab = (_text("VENT", 0.003, 0.115 - BOX_XY[0], BTN["vent"][1] - BOX_XY[1], BOX_H + 0.0004)
        + _text("PUMP", 0.003, 0.115 - BOX_XY[0], BTN["pump"][1] - BOX_XY[1], BOX_H + 0.0004))
CONTROLS = part(
    "controls", pos=(*BOX_XY, TOP), fixed=True, friction=0.8,
    boxes=[((0.0, 0.0, BOX_H / 2), (0.120, 0.160, BOX_H), BOXC),
           ((0.115 - BOX_XY[0], 0.0, BOX_H + 0.0001), (0.028, 0.150, 0.0002), LABEL_C)],
    meshes=[_boxes_mesh(_lab, INK)],
    joints=[Joint(n, kind="slide", axis="Z", anchor=(x - BOX_XY[0], y - BOX_XY[1], BOX_H), lower=-BTN_TRAVEL, upper=0.0,
                  initial=0.0, stiffness=400.0, damping=3.0, target=0.0, mass=0.01,
                  cylinders=[((x - BOX_XY[0], y - BOX_XY[1], BOX_H + BTN_H / 2), BTN_R, BTN_H,
                              VENT_C if n == "vent" else PUMP_C)])
            for n, (x, y) in BTN.items()],
)

# -- the screwdriver (loose) in its holder (fixed): a 0.8 x 5 mm blade on a 5 mm square shaft 60 mm long under a 12 mm
# square handle 35 mm long. Frame: origin at the blade's tip, +z up the shaft, the blade's width along y. It stands
# blade down in its holder, a block with a 7 mm square socket 10 mm deep, the tip 50 mm above the bench. ---------------
BLADE = (0.0008, 0.0050, 0.0040)
SH_W, SH_L, H_W, H_L = 0.0050, 0.060, 0.012, 0.035
SD_XY = (0.120, -0.170)
SOCK_Z, SOCK_D, SOCK_W = 0.060, 0.010, 0.007  # the holder's top above the bench, the socket's depth and width
SD_START = np.array([*SD_XY, TOP + SOCK_Z - SOCK_D])
SCREWDRIVER = part(
    "screwdriver", pos=tuple(SD_START), mass=0.035, friction=1.0,
    boxes=[((0.0, 0.0, BLADE[2] / 2), BLADE, STEEL),
           ((0.0, 0.0, BLADE[2] + (SH_L - BLADE[2]) / 2), (SH_W, SH_W, SH_L - BLADE[2]), STEEL),
           ((0.0, 0.0, SH_L + H_L / 2), (H_W, H_W, H_L), RED)],
)
_hb = (0.030 - SOCK_W) / 2  # the socket's walls
HOLDER = part(
    "sd_holder", pos=(*SD_XY, TOP), fixed=True, friction=0.6,
    boxes=[((0.0, 0.0, (SOCK_Z - SOCK_D) / 2), (0.030, 0.030, SOCK_Z - SOCK_D), DARK),
           *[((e * (SOCK_W + _hb) / 2, 0.0, SOCK_Z - SOCK_D / 2), (_hb, 0.030, SOCK_D), DARK) for e in (-1, 1)],
           *[((0.0, e * (SOCK_W + _hb) / 2, SOCK_Z - SOCK_D / 2), (SOCK_W, _hb, SOCK_D), DARK) for e in (-1, 1)]],
)

OBJECTS = [STAND, PLATE, WAFER, RING, LOADLOCK, DOOR, CONTROLS, SCREWDRIVER, HOLDER]
PLANCHECK_BOXES = [
    *[(f"ll{i}", (_cx + c[0], c[1], TOP + c[2]), s) for i, (c, s, _) in enumerate(LL_BOXES)],
    ("viewport", (_cx, 0.0, TOP + LL_Z - WALL / 2), (_ll_w, 2 * LL_Y, WALL)),
    ("door", (DOOR_X, 0.0, TOP + (DOOR_Z0 + DOOR_Z1) / 2), (0.006, 2 * LL_Y, DOOR_Z1 - DOOR_Z0)),
    ("holder_back", (HOLD_XY[0] + 0.093, 0.0, TOP + HOLD_Z / 2), (0.014, 0.060, HOLD_Z)),
    ("stand_plate", (*ST_XY, TOP + ST_H - BOSS_H - 0.003), (0.120, 0.120, 0.006)),
    ("controls", (*BOX_XY, TOP + BOX_H / 2 + 0.006), (0.120, 0.160, BOX_H + 0.012)),
    ("sd_holder", (*SD_XY, TOP + SOCK_Z / 2), (0.030, 0.030, SOCK_Z)),
    ("door_up", (DOOR_X, 0.0, TOP + (DOOR_Z0 + DOOR_Z1) / 2 + DOOR_UP), (0.006, 2 * LL_Y, DOOR_Z1 - DOOR_Z0)),
]

# -- scoring ----------------------------------------------------------------------------------------------------------
VENT_S = 10.0  # the load lock takes 10 s to reach air after VENT
DOOR_SHUT = 0.003  # the door counts as closed within 3 mm of its seat
HOLD_S, HOLD_NEAR = 2.0, 0.030  # after PUMP, a fingertip within 3 cm of the handle and the door shut for 2 s
CLIP_IN = math.radians(80.0)
BITE = 0.0020  # a clip's tip at least 2 mm inside the wafer's outline holds it
GAP_MAX = 150.0
CENTRED = 0.002
SEAT_XY, SEAT_Z, SEAT_DEG, PLATE_ON = 0.004, 0.003, 5.0, 0.003
TOUCH = 0.004
HANDS_FAR = 0.05
SETTLE_STEPS, SETTLE_TOL = 20, 0.001
_state = {}
_hist = []


@on_reset
def _reset_state(env):
    _state.clear()
    _state.update(step=0, pstate="vac", vent_step=None, seal_step=None, seal_ok=False, sealed=False, viol=set(),
                  pressed={"vent": False, "pump": False}, tight={}, settled=False, far=True, gap=360.0,
                  centred=False, stack=False, door=0.0, sd_in=False)
    _hist.clear()
    _counterbalance(env)


def _counterbalance(env):
    """The door hangs on a counterweight, like a sash window: an upward force equal to its weight."""
    import torch

    door = env.unwrapped.scene["door"]
    ids = door.find_bodies("door")[0]
    f = torch.zeros((1, len(ids), 3), device=door.device)
    f[..., 2] = DOOR_M * 9.81
    door.set_external_force_and_torque(f, torch.zeros_like(f), body_ids=ids, is_global=True)


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


def in_plate(p, x):
    """The point x (world) in the plate's frame."""
    return p.rotation().T @ (np.asarray(x, float) - np.asarray(p.pos, float))


def wafer_seated(p, w):
    q = in_plate(p, w.pos)
    return math.hypot(q[0], q[1]) <= 0.008 and PL_T - 0.0005 <= q[2] <= PL_T + W_T + 0.003


def holding_clips(p, w):
    """Plate-frame angles of the clips that hold the wafer: swung in past 80 degrees, screw tightened after it was
    swung in (and not swung since), tip at least 2 mm inside the wafer's outline."""
    Rp, Rw = p.rotation(), w.rotation()
    out = []
    for k in range(8):
        th = -float(p.joints.get(f"clip{k}", 0.0))
        lock = _state["tight"].get(k)
        if th < CLIP_IN or lock is None or abs(th - lock) > math.radians(10.0):
            continue
        tip = clip_tip(k, th)
        world = np.asarray(p.pos, float) + Rp @ np.array([tip[0], tip[1], TAB_Z])
        q = Rw.T @ (world - np.asarray(w.pos, float))
        if wafer_inside(q) >= BITE:
            out.append(CLIP_DEG[k])
    return out


def largest_gap(degs):
    if not degs:
        return 360.0
    s = sorted(degs)
    return max((s[(i + 1) % len(s)] - s[i]) % 360 or 360.0 for i in range(len(s)))


def stack_loaded(p, r):
    """The ring holder face down on the load lock's holder, carrying the plate face down and centred on it."""
    rp, pp = np.asarray(r.pos, float), np.asarray(p.pos, float)
    ring_ok = (tilt_deg(r.rotation(), down=True) <= SEAT_DEG
               and math.hypot(rp[0] - HOLD_XY[0], rp[1] - HOLD_XY[1]) <= SEAT_XY
               and abs(rp[2] - (TOP + HOLD_Z + RG_T)) <= SEAT_Z)
    plate_ok = (tilt_deg(p.rotation(), down=True) <= SEAT_DEG
                and math.hypot(pp[0] - rp[0], pp[1] - rp[1]) <= PLATE_ON
                and abs(pp[2] - (TOP + HOLD_Z + RG_T + PL_T)) <= SEAT_Z)
    return ring_ok and plate_ok


def handle_pos(door):
    return np.array([HANDLE_X, 0.0, TOP + HANDLE_Z + door])


def sd_inside(s):
    x = np.asarray(s.pos, float)
    return LL_X0 < x[0] < LL_X1 and abs(x[1]) < LL_Y and x[2] < TOP + LL_Z


@on_step
def _watch(env):
    """Latch the violations (the door opened before the load lock reached air, PUMP pressed with the door open, the long
    screw tightened, the wafer off its seat on the plate, the plate or ring dropped, a fingertip at the wafer's pattern)
    and follow the load lock's pressure, the clips' screws and the end state."""
    S = labkit._SPECS
    st = _state
    st["step"] += 1
    n = st["step"]
    p, w, r, d, c, s = (labkit._part_state(env, S[k]) for k in
                        ("sample_plate", "wafer", "ring_holder", "door", "controls", "screwdriver"))
    door = float(d.joints.get("door", 0.0))
    st["door"] = door
    tips = _tips(env)
    hp = handle_pos(door)
    # buttons, on their press
    for b in ("vent", "pump"):
        down = float(c.joints.get(b, 0.0)) <= -BTN_PRESSED
        if down and not st["pressed"][b]:
            if b == "vent" and st["pstate"] == "vac":
                st["pstate"], st["vent_step"], st["sealed"], st["seal_step"] = "venting", n, False, None
            if b == "pump":
                if door > DOOR_SHUT:
                    st["viol"].add("pumped with the door open")
                else:
                    st["pstate"], st["seal_step"], st["seal_ok"], st["sealed"] = "vac", n, True, False
        st["pressed"][b] = down
    if st["pstate"] == "venting" and (n - st["vent_step"]) * DT >= VENT_S:
        st["pstate"] = "air"
    if door > DOOR_SHUT and st["pstate"] != "air":
        st["viol"].add("door opened before the load lock reached air")
    if st["seal_step"] is not None and not st["sealed"]:
        near = min(np.linalg.norm(np.asarray(t) - hp) for t in tips) <= HOLD_NEAR
        if door > DOOR_SHUT or not near:
            st["seal_ok"] = False
        if (n - st["seal_step"]) * DT >= HOLD_S:
            st["sealed"], st["seal_step"] = st["seal_ok"], None
    # screws
    for k in range(8):
        sa = -float(p.joints.get(f"screw{k}", 0.0))
        if CLIP_DEG[k] == LONG_AT:
            if sa >= LONG_TURN:
                st["viol"].add("the long screw tightened")
            continue
        if sa >= SCREW_TIGHT and k not in st["tight"]:
            th = -float(p.joints.get(f"clip{k}", 0.0))
            if th >= CLIP_IN:
                st["tight"][k] = th
    # the wafer, the plate and the ring
    if not wafer_seated(p, w):
        st["viol"].add("wafer off its seat on the plate")
    if min(float(p.pos[2]), float(r.pos[2])) < TOP + 0.10:
        st["viol"].add("plate or ring dropped")
    Rw, pw = w.rotation(), np.asarray(w.pos, float)
    for tip in tips:
        q = Rw.T @ (np.asarray(tip, float) - pw)
        if abs(q[2]) < TOUCH and math.hypot(q[0], q[1]) < W_R - EDGE_EXCL:
            st["viol"].add("fingertip at the wafer's pattern")
    q = in_plate(p, w.pos)
    st["centred"] = math.hypot(q[0], q[1]) <= CENTRED
    st["gap"] = largest_gap(holding_clips(p, w))
    st["stack"] = stack_loaded(p, r)
    st["sd_in"] = sd_inside(s)
    st["far"] = all(np.linalg.norm(np.asarray(t) - hp) >= HANDS_FAR for t in tips)
    _hist.append(np.concatenate([np.asarray(p.pos, float), pw, np.asarray(r.pos, float), [door]]))
    del _hist[:-(SETTLE_STEPS + 1)]
    st["settled"] = (len(_hist) > SETTLE_STEPS
                     and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 after any violation (the door opened less than 10 s after VENT or after PUMP, PUMP pressed with the door open,
    the long screw turned over 90 degrees, the wafer off its seat on the plate, the plate or ring dropped, a fingertip
    within 4 mm of the wafer's pattern), if the screwdriver is in the load lock, or unless the plate, wafer, ring and door
    have been still (under 1 mm) for the last second. Otherwise: 0.25 if the wafer is centred on the plate (within 2 mm)
    and held by clips (swung in past 80 degrees, then their screws tightened 300 degrees, tips at least 2 mm inside the
    wafer's outline) that leave no gap over 150 degrees round it; 0.5 if also the ring holder lies face down on the load
    lock's holder (within 5 degrees and 4 mm) carrying the plate face down (centred on it within 3 mm); 0.75 if also the
    door is shut and PUMP was last pressed with it shut and a fingertip within 3 cm of its handle for 2 s; 1.0 if also
    every fingertip is 5 cm from the handle."""
    st = _state
    if st.get("viol") or not st.get("settled", False) or st.get("sd_in"):
        return 0.0
    if not st.get("centred") or st.get("gap", 360.0) >= GAP_MAX:
        return 0.0
    if not st.get("stack"):
        return 0.25
    if not (st.get("door", 1.0) <= DOOR_SHUT and st.get("sealed") and st.get("pstate") == "vac"):
        return 0.5
    return 1.0 if st.get("far", True) else 0.75


SUCCESS_HOLD_S = 2.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    p = np.asarray(objects["sample_plate"].pos, float)
    return float(math.hypot(p[0] - HOLD_XY[0], p[1] - HOLD_XY[1]))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left: presses VENT. Right: pushes the wafer to the plate's centre by its edge, waits for the vent, lifts the door.
# Both: take the ring holder off the load lock's holder and set it on the plate. Left swings in the 45 and 180 degree
# clips, right the 315 degree one (the 90 degree one faces the flat; the 225 degree one has the long screw). Right takes
# the screwdriver and tightens the three screws, two strokes each, then puts it back. Both pinch the plate and ring by
# their side edges, lift them, roll them over together, carry them into the load lock and set them on its holder. Right
# pulls the door down and holds it; left presses PUMP; both let go. Aims are trimmed only by what a camera sees: the
# wafer's, plate's, ring's and screwdriver tip's positions.
from yam_arena import ik as _ik  # noqa: E402
from yam_arena.constants import READY as _READY  # noqa: E402


def _ik_err(pe, re):
    return pe * 1000.0 + math.degrees(re)


def _endpoints(plan):
    """Joint angles for every waypoint's end, chained: warm-started from the previous one, and, where that solve misses,
    the closest of several seeded solves that reach it (so the wrist turns the way the waypoints mean)."""
    rng = np.random.default_rng(7)
    seeds = [np.array(_READY, float)] + [np.array(_READY, float) + rng.normal(0.0, 0.8, 6) for _ in range(10)]
    q_prev, out, switch = plan.q.copy(), [], set()
    for i, w in enumerate(plan.waypoints):
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, q_prev, base_pos=plan.base_pos, iters=300)
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
        if i not in switch:
            continue
        w = plan.waypoints[i - 1]
        pos, quat = np.asarray(w.pos, float), _ik.matrix_to_quat_xyzw(_demo.grip_rotation(w.yaw))
        q, pe, re = _ik.solve(pos, quat, out[i], base_pos=plan.base_pos, iters=300)
        if (_ik_err(pe, re) < 1.5 and float(np.max(np.abs(q - out[i])))
                < float(np.max(np.abs(out[i - 1] - out[i]))) - math.radians(10.0)):
            out[i - 1] = q
            switch.discard(i)
            switch.add(i - 1)
    return out, switch


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
if not getattr(_prev_init, "_eb2", False):
    def _init(self, *a, _prev=_prev_init, **k):
        _prev(self, *a, **k)
        self._q_start, self._qends = self.q.copy(), None

    _init._eb2 = True
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
LEFT_PARK = (0.20, 0.22, TOP + 0.22)
RIGHT_PARK = (0.20, -0.22, TOP + 0.22)
OPEN_W = -0.014
PINCH = -0.0020  # squeezing 5.5 mm: plate and ring, or the ring's rim and skirt
SD_OPEN, SD_HOLD = -0.012, -0.0012  # round the 5 mm shaft
BAR_OPEN, BAR_HOLD = -0.016, -0.0045  # round the 12 mm handle bar
SD_GRIP = 0.018  # the fingertips' ends this far above the blade's tip
CLIPS_USED = {"left": (0, 3), "right": (6,)}  # 0, 135 and 270 degrees
STROKE_TOP, STROKE = 145.0, 190.0  # hand yaw at the start of each tightening stroke, and how far it turns
FLIP = {"right": True, "left": False}
ROLL_SIGN = -1.0
LIFT_Z = 0.27  # the stack's centre this high (above the bench) while it is rolled over
STACK_IN = 0.092  # pad tips this far out from the centre when pinching the plate and ring
RING_IN = 0.104  # and when pinching the ring alone (its rim and skirt)
PUSH_DIR = -W_OFF / np.linalg.norm(W_OFF)

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
            continue
        e = np.asarray(fn(env), float)
        e[2] = 0.0
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


def _grasp(side, flip):
    a = Y if side == "right" else -Y
    c0 = -Z if flip else Z
    return np.column_stack([c0, np.cross(a, c0), a])


def _pinch(side, c, R, r_tip, out=0.0):
    """TCP and rotation for ``side``'s hand pinching an edge at its side, the grip line's centre at ``c``, the stack's
    attitude R (rolled about y); pad tips ``r_tip`` out from the centre; ``out`` backs the hand out sideways."""
    Rh = R @ _grasp(side, FLIP[side])
    a = Rh[:, 2]
    tip = np.asarray(c, float) - a * (r_tip + out)
    return tuple(tip - a * TIP_ABOVE_TCP), Orient(Rh)


def _bar(z, pitch=60.0, back=0.0):
    """TCP and rotation for the right hand round the door's handle bar at height z (above the bench), pitched down
    towards the door; ``back`` backs it off along its approach."""
    R = axis_angle_matrix(Y, math.radians(-pitch)) @ _prev_grip(math.pi / 2)
    a = R[:, 2]
    c = np.array([HANDLE_X, 0.0, TOP + z])
    return tuple(c - a * (0.007 + back)), Orient(R)


def _pw(q):
    """A plate-frame point (x, y, z above the plate's face) on the stand, in the world."""
    return np.array([PLATE_START[0] + q[0], PLATE_START[1] + q[1], PLATE_START[2] + PL_T + q[2]])


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


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


def demo():
    _ct["trim"], _ct["wins"] = {}, []
    pl = _Plan()
    face = PL_T  # the plate's face, plate frame

    # -- 0. start: both up and clear; left presses VENT; right pushes the wafer to the centre ---------------------------
    vx, vy = BTN["vent"]
    vt, vo = _down((vx, vy, TOP + BOX_H + BTN_H - BTN_TRAVEL - 0.0015), 0.0)  # pressed to the bottom
    vent = [(2.5, tuple(np.asarray(vt) + [0, 0, 0.04]), FINGER_CLOSED, "above VENT", vo),
            (1.0, tuple(np.asarray(vt) + [0, 0, 0.004]), FINGER_CLOSED, "fingertips on VENT", vo),
            (0.8, vt, FINGER_CLOSED, "press VENT", vo),
            (0.8, tuple(np.asarray(vt) + [0, 0, 0.04]), FINGER_CLOSED, "up: the load lock vents", vo)]
    u = PUSH_DIR
    pyaw = _wrap(math.atan2(u[1], u[0]) + math.pi / 2)  # fingers closing along the push: its 16 mm face leads
    tipz = -0.0002  # fingertips aimed 0.2 mm into the plate's face: they ride about 0.4 mm over it
    w0 = W_OFF - u * (W_R - STRIP_IN + 0.004)  # the fingertips' centre touching the wafer's edge
    w1 = -u * (W_R - STRIP_IN + 0.004)  # and where it is when the wafer is centred
    p_start = _down(_pw((*(w0 - u * 0.012), tipz)), pyaw)
    p_touch = _down(_pw((*(w0 - u * 0.001), tipz)), pyaw)
    p_near = _down(_pw((*(w1 - u * 0.0008), tipz)), pyaw)
    push = [(2.5, tuple(np.asarray(p_start[0]) + [0, 0, 0.03]), FINGER_CLOSED, "above the plate, beside the wafer's "
             "edge", p_start[1]),
            (1.2, p_start[0], FINGER_CLOSED, "fingertips down on the plate, outside the wafer", p_start[1]),
            (1.0, p_touch[0], FINGER_CLOSED, "against the wafer's edge", p_start[1]),
            (3.0, p_near[0], FINGER_CLOSED, "push the wafer towards the plate's centre", p_start[1])]
    t0 = pl.phase(right=push, left=vent)
    seen_off = lambda env: (u * float(np.dot(u, (_seen(env, "sample_plate") - _seen(env, "wafer"))[:2]))).tolist() + [0]
    pl.phase(right=[(3.0, _Dyn(p_near[0], "push"), FINGER_CLOSED, "hold still, then push on until the wafer is "
                     "centred", p_start[1])])
    pl.window(pl.t - 3.0, pl.t, "push", lambda env: np.array([*seen_off(env)[:2], 0.0]), "wafer")
    back = tuple(np.asarray(p_near[0]) - [*(u * 0.010), 0] + [0, 0, 0.0])
    pl.phase(right=[(0.8, _Dyn(back, "push"), FINGER_CLOSED, "back off the wafer's edge", p_start[1]),
                    (1.0, _Dyn(tuple(np.asarray(back) + [0, 0, 0.04]), "push"), FINGER_CLOSED, "up", p_start[1])])
    vent_at = t0 + 4.3  # VENT pressed (end of its press)

    # -- 1. right lifts the door, once the load lock has had 10 s to vent ---------------------------------------------------
    b0, bo = _bar(HANDLE_Z, back=0.05)
    b1, _ = _bar(HANDLE_Z)
    b2, _ = _bar(HANDLE_Z + DOOR_UP - 0.003)
    b3, _ = _bar(HANDLE_Z + DOOR_UP - 0.003, back=0.06)
    wait = max(0.0, vent_at + VENT_S + 1.0 - pl.t - 3.7)
    door_open = [(2.0 + wait, b0, BAR_OPEN, "in front of the door's handle", bo),
                 (1.2, b1, BAR_OPEN, "round the handle", bo),
                 (0.5, b1, BAR_HOLD, "hold it (10 s after VENT: the load lock is at air)", bo),
                 (3.0, b2, BAR_HOLD, "lift the door", bo),
                 (0.5, b2, BAR_OPEN, "let go: it stays up", bo),
                 (1.0, b3, BAR_OPEN, "back off", bo)]
    pl.phase(right=door_open, left=[(2.0, LEFT_PARK, FINGER_CLOSED, "wait, up and clear", 0.0)])

    # -- 2. both take the ring holder out and set it on the plate -------------------------------------------------------
    rc = RING_START + [0, 0, (RG_T - PL_T) / 2]  # its grip line: between the rim's top and the skirt's underside
    over = np.array([*ST_XY, PLATE_START[2] + PL_T + (RG_T - PL_T) / 2])  # the same, the ring resting on the plate
    I = np.eye(3)

    def both(dur, c, finger, label, R=I, r_tip=RING_IN, out=0.0, key=None):
        rt, ro = _pinch("right", c, R, r_tip, out)
        lt, lo = _pinch("left", c, R, r_tip, out)
        if key:
            rt, lt = _Dyn(rt, key), _Dyn(lt, key)
        return [(dur, rt, finger, label, ro)], [(dur, lt, finger, label, lo)]

    def ph(*a, **k):
        r, l_ = both(*a, **k)
        return pl.phase(right=r, left=l_)

    ph(3.0, rc + [-0.225, 0, 0.02], OPEN_W, "in front of the open door, wrist turned", out=0.02)
    ph(2.5, rc + [-0.03, 0, 0.0], OPEN_W, "into the load lock, beside the ring holder", out=0.02)
    ph(2.0, rc, OPEN_W, "level with its rim", out=0.02)
    ph(1.5, rc, OPEN_W, "fingers over its rim and under its skirt")
    ph(0.8, rc, PINCH, "pinch it")
    ph(1.2, rc + [0, 0, 0.012], PINCH, "lift it off the holder")
    hi = PLATE_START[2] + 0.075
    ph(3.0, np.array([*ST_XY, rc[2] + 0.012]) + [0.05, 0, 0], PINCH, "out of the load lock")
    ph(2.0, np.array([*ST_XY, over[2] + 0.008]), PINCH, "over the plate")
    tw = pl.t
    ph(2.0, np.array([*ST_XY, over[2] + 0.004]), PINCH, "hold still, its skirt seen round the plate", key="ring")
    pl.window(tw, pl.t, "ring", lambda env: _seen(env, "sample_plate") - _seen(env, "ring_holder"), "ring_holder")
    ph(1.2, over + [0, 0, 0.0005], PINCH, "set it on the plate", key="ring")
    ph(0.6, over + [0, 0, 0.0005], OPEN_W, "let go", key="ring")
    ph(1.2, over, OPEN_W, "back out sideways", out=0.012, key="ring")
    ph(1.2, over + [0, 0, hi - over[2]], OPEN_W, "up", out=0.012, key="ring")

    # -- 3. clips: left 0 and 135 degrees, right 270 ------------------------------------------------------------------
    knob_z = face + (TAB_TOP - PL_T) + 0.0008  # the fingertips just over the tab, low on the knob

    def knob(k, th, back=None):
        """The fingertips' point behind clip k's knob (against its swing) at swing angle th: the closed fingertips are 16
        mm wide across and 8 mm thick along their closing axis, which lies along the clip's parked direction."""
        if back is None:  # pressing: the fingertips' face 1 mm into the knob
            back = KNOB_R - 0.001 + 0.0078 * abs(math.cos(th)) + 0.004 * abs(math.sin(th))
        P, t, n = clip_frame(k)
        kp = P + KNOB_AT * clip_dir(k, th)
        d = -math.sin(th) * t + math.cos(th) * n
        return _pw((*(kp - back * d), knob_z - face))

    def swing(k, label):
        yaw = _wrap(math.radians(CLIP_DEG[k]))
        if abs(yaw) > math.pi / 2:
            yaw -= math.copysign(math.pi, yaw)
        clear = KNOB_R + 0.0015 + 0.0078
        s = [_down(knob(k, 0.0, clear) + [0, 0, 0.03], yaw), _down(knob(k, 0.0, clear), yaw)]
        s += [_down(knob(k, math.radians(a)), yaw) for a in (25.0, 50.0, 75.0)]
        P, t, n = clip_frame(k)
        k90 = P + KNOB_AT * n
        s.append(_down(_pw((*(k90 + t * (KNOB_R + 0.004 - 0.0015)), knob_z - face)), yaw))
        return [(1.5, s[0][0], FINGER_CLOSED, f"above the {label} clip, outside its knob", s[0][1]),
                (1.0, s[1][0], FINGER_CLOSED, "fingertips down beside the knob", s[1][1]),
                (0.8, s[2][0], FINGER_CLOSED, "swing the clip in", s[2][1]),
                (0.8, s[3][0], FINGER_CLOSED, "swing the clip in", s[3][1]),
                (0.8, s[4][0], FINGER_CLOSED, "swing the clip in", s[4][1]),
                (1.0, s[5][0], FINGER_CLOSED, "against its stop: the tip over the wafer's edge", s[5][1]),
                (0.8, tuple(np.asarray(s[5][0]) + [0, 0, 0.03]), FINGER_CLOSED, "up", s[5][1])]

    lp = [(2.0, LEFT_PARK, FINGER_CLOSED, "up and clear", 0.0)] + swing(0, "0 degree") + swing(3, "135 degree")
    rp = [(2.0 + sum(w[0] for w in swing(0, "") + swing(3, "")), RIGHT_PARK, FINGER_CLOSED, "wait, out of the left hand's way", 0.0)]
    rp += swing(6, "270 degree")
    pl.phase(right=rp, left=lp + [(1.5, LEFT_PARK, FINGER_CLOSED, "aside, out of the right hand's way", 0.0)])

    # -- 4. right: the screwdriver, three screws ----------------------------------------------------------------------
    def sd_tcp(tip, z):
        return (tip[0], tip[1], z + SD_GRIP + TIP_ABOVE_TCP)

    z0 = SD_START[2]
    up = z0 + 0.04  # its tip clear of the socket
    yo = Orient(_prev_grip(0.0))
    pl.phase(right=[(2.5, sd_tcp(SD_XY, z0 + 0.04), SD_OPEN, "above the screwdriver's shaft", yo),
                    (1.2, sd_tcp(SD_XY, z0), SD_OPEN, "fingers round its shaft", yo),
                    (0.8, sd_tcp(SD_XY, z0), SD_HOLD, "hold it", yo),
                    (1.8, sd_tcp(SD_XY, up), SD_HOLD, "lift it out of its holder", yo)])
    head = PLATE_START[2] + TAB_TOP + HEAD_H  # a screw head's top (world z)
    for k in (6, 0, 3):  # 270, 0, 135 degrees
        P = clip_frame(k)[0]
        xy = _pw((*P, 0.0))[:2]
        key = f"sd{k}"
        y0 = math.radians(STROKE_TOP)
        yaws = [Orient(_prev_grip(y0 - math.radians(STROKE) * i / 6)) for i in range(7)]
        mid = Orient(_prev_grip(math.radians(STROKE_TOP - STROKE / 2)))
        seq = [(1.5, sd_tcp(xy, head + 0.008), SD_HOLD, f"over the {CLIP_DEG[k]} degree clip's screw", mid),
               (1.0, sd_tcp(xy, head + 0.008), SD_HOLD, "turn the blade to the slot", yaws[0])]
        tw0 = pl.t + 2.5
        seq += [(2.0, _Dyn(sd_tcp(xy, head + 0.004), key), SD_HOLD, "hold still, the blade's tip seen over the slot",
                 yaws[0]),
                (1.0, _Dyn(sd_tcp(xy, head + 0.0002), key), SD_HOLD, "blade into the slot", yaws[0])]
        for stroke in range(2):
            seq += [(0.6, _Dyn(sd_tcp(xy, head - 0.0004), key), SD_HOLD, "turn it clockwise", yaws[i])
                    for i in range(1, 7)]
            seq.append((0.6, _Dyn(sd_tcp(xy, head + 0.003), key), SD_HOLD, "lift the blade out", yaws[6]))
            if stroke == 0:
                seq += [(0.8, _Dyn(sd_tcp(xy, head + 0.003), key), SD_HOLD, "turn back", yaws[i]) for i in (4, 2, 0)]
                seq.append((0.8, _Dyn(sd_tcp(xy, head + 0.0002), key), SD_HOLD, "blade back in the slot", yaws[0]))
        seq.append((1.2, _Dyn(sd_tcp(xy, head + 0.008), key), SD_HOLD, "up", yaws[6]))
        pl.phase(right=seq)
        pl.window(tw0 + 0.8, tw0 + 2.0, key, lambda env, P=P: np.array([*(_pw((*P, 0.0))[:2]
                                                                            - _seen(env, "screwdriver")[:2]), 0.0]),
                  "screwdriver")
    pl.phase(right=[(1.5, sd_tcp((0.17, -0.10), head + 0.03), SD_HOLD, "up, off the plate", yo),
                    (2.0, sd_tcp(SD_XY, up), SD_HOLD, "back over its holder", yo),
                    (1.8, sd_tcp(SD_XY, z0 + 0.001), SD_HOLD, "into its holder", yo),
                    (0.6, sd_tcp(SD_XY, z0 + 0.001), SD_OPEN, "let go", yo),
                    (1.2, sd_tcp(SD_XY, z0 + 0.06), SD_OPEN, "up", yo)])

    # -- 5. both: roll the plate and ring over together and set them on the load lock's holder ---------------------------
    c0 = np.array([*ST_XY, PLATE_START[2] + (PL_T + RG_T) / 2])
    lift = np.array([*ST_XY, TOP + LIFT_Z])
    ph(3.0, c0 + [0, 0, 0.12], OPEN_W, "high beside the plate and ring, wrist turned", r_tip=STACK_IN, out=0.03)
    ph(2.0, c0 + [0, 0, 0.03], OPEN_W, "beside the plate's and ring's side edges, above", r_tip=STACK_IN, out=0.03)
    ph(1.5, c0, OPEN_W, "level with them", r_tip=STACK_IN, out=0.03)
    ph(1.5, c0, OPEN_W, "fingers over the ring and under the plate", r_tip=STACK_IN)
    ph(0.8, c0, PINCH, "pinch them together", r_tip=STACK_IN)
    ph(1.5, c0 + [0, 0, 0.03], PINCH, "lift them off the stand", r_tip=STACK_IN)
    ph(2.5, lift, PINCH, "up, clear of everything", r_tip=STACK_IN)
    for k in range(1, 9):
        ph(0.8, lift, PINCH, "roll them over together: samples face down", axis_angle_matrix(Y, ROLL_SIGN * math.pi * k / 8),
           r_tip=STACK_IN)
    Rf = axis_angle_matrix(Y, ROLL_SIGN * math.pi)
    seat = np.array([*HOLD_XY, TOP + HOLD_Z + (PL_T + RG_T) / 2])
    above = seat + [0, 0, 0.012]
    ph(2.0, np.array([*ST_XY, above[2]]) + [0.06, 0, 0], PINCH, "down to the door's opening", Rf, r_tip=STACK_IN)
    ph(3.0, above, PINCH, "into the load lock, over its holder", Rf, r_tip=STACK_IN)
    tw = pl.t
    ph(2.0, above, PINCH, "hold still, seen over the holder", Rf, r_tip=STACK_IN, key="stack")
    pl.window(tw, pl.t, "stack", lambda env: np.array([*HOLD_XY, 0.0]) - _seen(env, "ring_holder"), "ring_holder")
    ph(1.5, seat + [0, 0, 0.0015], PINCH, "set the ring holder on the holder", Rf, r_tip=STACK_IN, key="stack")
    ph(0.6, seat + [0, 0, 0.0005], OPEN_W, "let go", Rf, r_tip=STACK_IN, key="stack")
    ph(1.2, seat, OPEN_W, "back out sideways", Rf, r_tip=STACK_IN, out=0.03, key="stack")
    ph(2.5, seat + [-0.17, 0, 0.0], OPEN_W, "out of the load lock", Rf, r_tip=STACK_IN, out=0.03)

    # -- 6. right closes the door and holds it; left presses PUMP; both let go -------------------------------------------
    pt, po = _down((*BTN["pump"], TOP + BOX_H + BTN_H - BTN_TRAVEL - 0.0015), 0.0)
    pl.phase(right=[(3.0, _bar(HANDLE_Z + DOOR_UP - 0.003, back=0.10)[0], BAR_OPEN, "back from the raised door, hand turned", bo),
                    (2.0, b3, BAR_OPEN, "in front of the raised door's handle", bo),
                    (1.2, b2, BAR_OPEN, "round the handle", bo),
                    (0.5, b2, BAR_HOLD, "hold it", bo),
                    (3.0, _bar(HANDLE_Z - 0.002)[0], BAR_HOLD, "pull the door down onto its seat", bo)],
             left=[(1.5, _pinch("left", seat + [-0.20, 0.0, 0.03], Rf, STACK_IN, 0.03)[0], OPEN_W, "up and out",
                    _pinch("left", seat, Rf, STACK_IN)[1]),
                   (3.0, LEFT_PARK, FINGER_CLOSED, "up and clear", 0.0),
                   (2.5, tuple(np.asarray(pt) + [0, 0, 0.04]), FINGER_CLOSED, "above PUMP", po)])
    pl.phase(right=[(0.6, _bar(HANDLE_Z - 0.002)[0], BAR_HOLD, "hold the door shut", bo)],
             left=[(0.6, tuple(np.asarray(pt) + [0, 0, 0.004]), FINGER_CLOSED, "fingertips on PUMP", po)])
    pl.phase(right=[(0.8, _bar(HANDLE_Z - 0.002)[0], BAR_HOLD, "hold the door shut", bo)],
             left=[(0.8, pt, FINGER_CLOSED, "press PUMP", po)])
    pl.phase(right=[(2.8, _bar(HANDLE_Z - 0.002)[0], BAR_HOLD, "hold the door shut while it pumps down", bo)],
             left=[(0.8, tuple(np.asarray(pt) + [0, 0, 0.04]), FINGER_CLOSED, "up", po),
                   (2.0, LEFT_PARK, FINGER_CLOSED, "away", 0.0)])
    pl.phase(right=[(0.6, _bar(HANDLE_Z - 0.001)[0], BAR_OPEN, "let go", bo),
                    (1.2, _bar(HANDLE_Z, back=0.07)[0], BAR_OPEN, "back off", bo),
                    (1.5, RIGHT_PARK, FINGER_CLOSED, "away", 0.0)])
    pl.phase(right=[(2.5, RIGHT_PARK, FINGER_CLOSED, "rest", 0.0)], left=[(2.5, LEFT_PARK, FINGER_CLOSED, "rest", 0.0)])
    return {"left": pl.L, "right": pl.R}
