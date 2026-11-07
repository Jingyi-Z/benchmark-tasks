"""On the station table, at a contact mask aligner's mask-loading station: the aligner's mask holder, pulled out and
lying upright on its loading stand, still holds the LAYER 1 chrome mask from the last exposure, chrome side down, held
by the MASK VACUUM (its toggle is ON). Behind it stand two mask rests: an empty one for LAYER 1 (left) and one with
the LAYER 2 mask (right), lying chrome side UP.

The masks are 5 inch soda-lime glass plates (127 x 127 x 2.3 mm) with a chrome pattern on one side. They look alike;
the only cue to which side carries the chrome is the label, written in chrome: it reads correctly only from the chrome
side and appears mirrored through the glass.

The task: swap the masks. Switch the mask vacuum off, put LAYER 1 on its rest, set LAYER 2 on the holder chrome side
DOWN (the chrome faces the wafer in contact printing) and pushed against the holder's two back stop pins, and switch the
mask vacuum on. Masks are held by their edges only; nothing touches the patterned area. A mask lifted while the vacuum
holds it, or dropped, scores 0.

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

GLASS = (0.86, 0.90, 0.88)  # soda-lime glass
labkit.GLASS_COLORS.update({GLASS})
CHROME = (0.30, 0.30, 0.33)
ALU = (0.74, 0.75, 0.77)
STAND_C = (0.42, 0.43, 0.46)
STEEL = (0.72, 0.73, 0.75)
BLACK = (0.06, 0.06, 0.07)
GREEN = (0.10, 0.55, 0.20)
WHITE_PAD = (0.90, 0.90, 0.88)

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

# -- the masks (loose): 5 inch soda-lime plates. Frame: origin at the plate's centre, mid-plane. The chrome is on the
# +z face (s=+1) or the -z face (s=-1): a 9 x 9 array of chrome dies inside the central 100 mm square (the patterned
# area) and the label, in chrome letters 9 mm tall, along the plate's +x margin. Read from the chrome side, the label's
# letters stand up towards +x and read along -y (left to right from the robots' side). -------------------------------
M_HALF, M_T = 0.0635, 0.0023
PATTERN_HALF = 0.050
PIX = 0.0013  # the label's font pixel
FONT = {  # 5 x 7, rows top to bottom
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    " ": ["00000"] * 7,
}


def _boxes_mesh(boxes, colour):
    """One drawn-only mesh (no collision) from many boxes: (centre, size) pairs."""
    pts, tris = [], []
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    for c, size in boxes:
        c, h = np.asarray(c, float), np.asarray(size, float) / 2
        b = len(pts)
        pts += [c + h * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        tris += [(b + i, b + j, b + k) for i, j, k in faces]
    return labkit.Mesh(points=np.array(pts), tris=np.array(tris), color=colour, collide="none")


def _chrome(text, s):
    """The chrome on face s (+1: +z, -1: -z): dies and the label, laid out to read correctly seen from that face."""
    z = s * (M_T / 2 + 0.00005)
    t = 0.0001
    dies = [((-PATTERN_HALF + 0.0055 + 0.011 * i, -PATTERN_HALF + 0.0055 + 0.011 * j, z), (0.008, 0.008, t))
            for i in range(9) for j in range(9)]
    label = []
    w = len(text) * 6 * PIX
    for n, ch in enumerate(text):
        for r, row in enumerate(FONT[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    u = -w / 2 + (n * 6 + c + 0.5) * PIX  # along the reading direction
                    v = 0.0565 + (3 - r) * PIX  # up the letters, towards +x
                    label.append(((v, -u * s, z), (PIX * 0.95, PIX * 0.95, t)))  # mirrored when read through
    return [_boxes_mesh(dies, CHROME), _boxes_mesh(label, CHROME)]


def _mask(name, text, s, xyz, yaw=0.0):
    m = part(name, pos=xyz, yaw=yaw, mass=0.093, friction=0.7,
             boxes=[((0.0, 0.0, 0.0), (2 * M_HALF, 2 * M_HALF, M_T), GLASS)], meshes=_chrome(text, s))
    return m


# -- the mask holder on its loading stand (fixed): an aluminium plate 160 x 100 mm (narrower than the mask, so its two
# side edges overhang by 13.5 mm), with a window under the patterned area, two stop pins at its back edge and a white
# vacuum pad strip round the window; on a narrow column, its top 110 mm above the bench. Frame: origin on the bench
# under the holder's centre. -------------------------------------------------------------------------------------------
H_XY = (0.22, 0.0)
H_TOP = 0.110  # the holder plate's top above the bench
H_PLATE = (0.160, 0.100, 0.006)
STOP_X = M_HALF  # the stop pins' front faces: the mask's back edge goes against them
STOP_Y = 0.035
STOP_R = 0.003
_hz = H_TOP - H_PLATE[2] / 2
_win = (0.104, 0.060)  # the window's x and y size
HOLDER = part(
    "mask_holder", pos=(*H_XY, TOP), fixed=True, friction=0.5,
    boxes=[
        ((0.0, 0.0, (H_TOP - H_PLATE[2]) / 2), (0.060, 0.050, H_TOP - H_PLATE[2]), STAND_C),  # the column
        # the plate round its window
        (((H_PLATE[0] + _win[0]) / 4, 0.0, _hz), ((H_PLATE[0] - _win[0]) / 2, H_PLATE[1], H_PLATE[2]), ALU),
        ((-(H_PLATE[0] + _win[0]) / 4, 0.0, _hz), ((H_PLATE[0] - _win[0]) / 2, H_PLATE[1], H_PLATE[2]), ALU),
        ((0.0, (H_PLATE[1] + _win[1]) / 4, _hz), (_win[0], (H_PLATE[1] - _win[1]) / 2, H_PLATE[2]), ALU),
        ((0.0, -(H_PLATE[1] + _win[1]) / 4, _hz), (_win[0], (H_PLATE[1] - _win[1]) / 2, H_PLATE[2]), ALU),
    ],
    cylinders=[((STOP_X + STOP_R, e * STOP_Y, H_TOP + 0.002), STOP_R, 0.004, STEEL) for e in (-1, 1)],
    meshes=[_boxes_mesh([((0.0, e * (_win[1] / 2 + 0.004), H_TOP + 0.00005), (_win[0] + 0.008, 0.002, 0.0001))
                         for e in (-1, 1)] + [((e * (_win[0] / 2 + 0.004), 0.0, H_TOP + 0.00005),
                                               (0.002, _win[1] + 0.008, 0.0001)) for e in (-1, 1)], WHITE_PAD)],
)
H_C = np.array([*H_XY, TOP + H_TOP])
SEAT_C = H_C + [0.0, 0.0, M_T / 2]  # a mask's centre lying on the holder, against the stops, centred across

# -- the mask rack (fixed), behind the holder: two padded shelves, 100 x 90 mm, cantilevered from a post at the back:
# the lower one (110 mm above the bench) holds the LAYER 2 mask, the upper one (210 mm) is LAYER 1's, empty. -----------
RACK_XY = (0.365, 0.0)
R_TOP = 0.110
R1_TOP = 0.210
REST_L1_XY = REST_L2_XY = RACK_XY
_post_x = 0.078  # behind the shelved masks' back edges


def _shelf(h):
    return [((0.0, 0.0, h - 0.003), (0.110, 0.090, 0.006), STAND_C),
            ((0.0, 0.0, h - 0.0003), (0.100, 0.090, 0.0006), WHITE_PAD)]


RACK = part("mask_rack", pos=(*RACK_XY, TOP), fixed=True, friction=0.8,
            boxes=[((_post_x, 0.0, R1_TOP / 2), (0.020, 0.050, R1_TOP), STAND_C),
                   ((0.0, 0.0, (R_TOP - 0.006) / 2), (0.060, 0.050, R_TOP - 0.006), STAND_C),
                   *_shelf(R_TOP), *_shelf(R1_TOP),
                   ((_post_x / 2 + 0.03, 0.0, R1_TOP - 0.003), (_post_x - 0.04, 0.030, 0.006), STAND_C)])
MASK_L1 = _mask("mask_layer1", "LAYER 1", -1, (*SEAT_C[:2], SEAT_C[2]))  # on the holder, chrome down
MASK_L2 = _mask("mask_layer2", "LAYER 2", +1, (*REST_L2_XY, TOP + R_TOP + M_T / 2))  # on its rest, chrome up
CHROME_SIDE = {"mask_layer1": -1, "mask_layer2": +1}  # which local face carries the chrome

# -- the MASK VACUUM toggle (fixed), on a small box at the front right: its lever tilted forward (-x) is ON, back (+x)
# is OFF. It starts ON: the vacuum holds the LAYER 1 mask on the holder. ----------------------------------------------
V_XY = (0.12, -0.12)
V_H = 0.030
VAC = Joint("vacuum", kind="hinge", axis="Y", anchor=(0.0, 0.0, V_H), lower=-0.45, upper=0.45, initial=-0.42,
            target=0.0, friction=0.012, damping=0.02, mass=0.005,
            boxes=[((0.0, 0.0, V_H + 0.012), (0.004, 0.004, 0.024), STEEL)],
            cylinders=[((0.0, 0.0, V_H + 0.025), 0.0035, 0.004, BLACK)])
VAC_SWITCH = part("vacuum_switch", pos=(*V_XY, TOP), fixed=True, friction=0.6,
                  boxes=[((0.0, 0.0, V_H / 2), (0.030, 0.024, V_H), BLACK),
                         ((-0.019, 0.0, 0.0003), (0.005, 0.014, 0.0006), GREEN)],  # the ON mark, in front
                  joints=[VAC])
VAC_ON = -0.20  # ON when tilted forward past this
VAC_OFF = 0.20  # OFF when tilted back past this

OBJECTS = [HOLDER, RACK, MASK_L1, MASK_L2, VAC_SWITCH]
MASKS = ("mask_layer1", "mask_layer2")

# -- scoring ----------------------------------------------------------------------------------------------------------
FLAT_DEG = 2.0
SQUARE_DEG = 1.5  # a mask's edges within 1.5 degrees of the holder's
STOP_TOL = 0.001  # its back edge within 1 mm of both stop pins
SIDE_TOL = 0.0015  # centred across the holder within 1.5 mm
REST_TOL = 0.015  # LAYER 1 back on its rest: centre within 15 mm of the rest's centre, lying flat on it
TOUCH = 0.004  # a pad tip within 4 mm of either face of a mask over its patterned area: the pattern touched
HANDS_FAR = 0.08  # both hands' pad tips this far from the holder's centre at the end
PULL = 0.001  # the LAYER 1 mask moved off its seat while the vacuum is ON: pulled off under vacuum

_state = {}
SETTLE_STEPS, SETTLE_TOL = 20, 0.001
_hist = []


@on_reset
def _reset_state(env):
    _state.update(touched=False, dropped=False, pulled=False, far=True, settled=False)
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


def chrome_normal(name, m) -> np.ndarray:
    return m.rotation() @ (Z * CHROME_SIDE[name])


def square_err_deg(m) -> float:
    v = m.rotation()[:, 0]
    a = math.degrees(math.atan2(v[1], v[0])) % 90.0
    return min(a, 90.0 - a)


def on_holder(name, m) -> bool:
    """Lying flat on the holder, chrome side down, square, centred across it and against both stop pins."""
    p = np.asarray(m.pos, float)
    return (chrome_normal(name, m)[2] < -math.cos(math.radians(FLAT_DEG))
            and abs(p[2] - SEAT_C[2]) <= 0.001 and square_err_deg(m) <= SQUARE_DEG
            and abs(p[1] - SEAT_C[1]) <= SIDE_TOL and abs(p[0] - SEAT_C[0]) <= STOP_TOL)


def on_holder_loose(m) -> bool:
    p = np.asarray(m.pos, float)
    return np.hypot(*(p[:2] - SEAT_C[:2])) < 0.03 and abs(p[2] - SEAT_C[2]) <= 0.002


def on_rest(m, xy, h) -> bool:
    p = np.asarray(m.pos, float)
    return (np.hypot(p[0] - xy[0], p[1] - xy[1]) <= REST_TOL and abs(p[2] - (TOP + h + M_T / 2)) <= 0.001
            and abs(m.rotation()[2, 2]) > math.cos(math.radians(FLAT_DEG)))


@on_step
def _watch(env):
    """Latch the violations: a mask's patterned area touched by the gripper, a mask dropped (below its supports), the
    LAYER 1 mask pulled off the holder while the vacuum held it."""
    S = labkit._SPECS
    tips = _tips(env)
    vac = labkit._part_state(env, S["vacuum_switch"]).joints["vacuum"]
    track = []
    for n in MASKS:
        m = labkit._part_state(env, S[n])
        p = np.asarray(m.pos, float)
        R = m.rotation()
        if p[2] < TOP + min(R_TOP, H_TOP) - 0.004:
            _state["dropped"] = True
        for tip in tips:
            q = R.T @ (np.asarray(tip, float) - p)
            if abs(q[0]) < PATTERN_HALF and abs(q[1]) < PATTERN_HALF and abs(q[2]) < M_T / 2 + TOUCH:
                _state["touched"] = True
        track.append(p)
    l1 = np.asarray(labkit._part_state(env, S["mask_layer1"]).pos, float)
    if vac < VAC_OFF and np.linalg.norm(l1 - SEAT_C) > PULL and not _state.get("released"):
        _state["pulled"] = True
    if np.linalg.norm(l1 - SEAT_C) > 0.02:
        _state["released"] = True  # once it is off the holder, the vacuum no longer bears on it
    _state["far"] = all(np.hypot(*(t[:2] - H_C[:2])) >= HANDS_FAR for t in tips)
    _hist.append(np.concatenate([np.concatenate(track), [vac * 0.01]]))
    del _hist[:-(SETTLE_STEPS + 1)]
    _state["settled"] = (len(_hist) > SETTLE_STEPS
                         and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def score(objects):
    """0 if a mask's patterned area was ever touched by the gripper, a mask was dropped, the LAYER 1 mask was lifted
    while the vacuum held it, or the masks have not both been still (under 1 mm of movement) for the last second.
    Otherwise 0.25 for LAYER 1 lying flat on its rest; with that, 0.5 if LAYER 2 also lies on the holder chrome side
    down; 0.75 if it is square (1.5 degrees), centred across the holder (1.5 mm) and against both stop pins (1 mm);
    1.0 if, in addition, the mask vacuum is ON and both hands are 8 cm from the holder's centre. 0 if LAYER 1 is not on
    its rest."""
    if _state.get("touched") or _state.get("dropped") or _state.get("pulled"):
        return 0.0
    if not _state.get("settled", False):
        return 0.0
    l1, l2 = objects["mask_layer1"], objects["mask_layer2"]
    if not on_rest(l1, REST_L1_XY, R1_TOP):
        return 0.0
    if not (on_holder_loose(l2) and chrome_normal("mask_layer2", l2)[2] < -math.cos(math.radians(FLAT_DEG))):
        return 0.25
    if not on_holder("mask_layer2", l2):
        return 0.5
    if objects["vacuum_switch"].joints["vacuum"] > VAC_ON or not _state.get("far", True):
        return 0.75
    return 1.0


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return float(np.linalg.norm(np.asarray(objects["mask_layer2"].pos, float) - SEAT_C))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right hand: pushes the vacuum toggle back to OFF. Both hands: pinch the LAYER 1 mask's two side edges where they
# overhang the holder, lift it, carry it to its rest and set it down. Then pinch LAYER 2's side edges on its rest, lift
# it, roll it over together (half a turn about the line between the hands), carry it over the holder, lower it just
# in front of the stop pins, slide it back against them and let go. Right hand: pushes the toggle forward to ON. The
# demo reads only what the cameras see: the masks' poses.
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
PAD_IN = 0.008  # the pad tips this far in from a mask's side edge
OPEN_W = -0.012  # pads 24 mm apart, round the 2.3 mm plate
PINCH = -0.0008  # squeezing the plate


def _grasp(side, flip):
    """Hand rotation pinching a mask's side edge: approach across the mask (+y for the right hand, -y for the left),
    fingers closing vertically (``flip`` turns the hand half a turn about its approach)."""
    a = Y if side == "right" else -Y
    c0 = -Z if flip else Z
    return np.column_stack([c0, np.cross(a, c0), a])


FLIP = {"right": True, "left": False}  # each hand's grasp (found by an IK search) that can follow the roll
ROLL_SIGN = -1.0  # the roll's sense about +y


def _hand(side, c, R_mask=np.eye(3)):
    """TCP and rotation for ``side``'s hand pinching the side edge of a mask centred at ``c`` with attitude R_mask
    (relative to how it lay: rolled about y, or turned about z)."""
    R = R_mask @ _grasp(side, FLIP[side])
    a = R[:, 2]
    tip = np.asarray(c, float) - a * (M_HALF - PAD_IN)
    return tuple(tip - a * TIP_ABOVE_TCP), Orient(R)


def _pre(side, c, R_mask=np.eye(3), out=0.03):
    """The same, backed out sideways ``out`` from the edge."""
    tcp, rot = _hand(side, c, R_mask)
    return tuple(np.asarray(tcp) - rot.rot[:, 2] * out), rot


def _wp2(dur, c, finger, label, R_mask=np.eye(3), out=0.0):
    """A waypoint pair (left, right) for both hands at the mask's side edges."""
    out_l = _pre("left", c, R_mask, out) if out else _hand("left", c, R_mask)
    out_r = _pre("right", c, R_mask, out) if out else _hand("right", c, R_mask)
    return (Waypoint(dur, out_l[0], finger, label, out_l[1]), Waypoint(dur, out_r[0], finger, label, out_r[1]))


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    left, right = [], []

    def both(*pairs):
        for lw, rw in pairs:
            left.append(lw)
            right.append(rw)

    dn = Orient(_prev_grip(0.0))
    tcp = lambda p: tuple(np.asarray(p, float) + [0.0, 0.0, TIP_ABOVE_TCP])  # noqa: E731
    knob = lambda a: np.array([V_XY[0] + 0.014 * math.sin(a), V_XY[1], TOP + V_H + 0.014 * math.cos(a)])  # noqa: E731
    # (a point 14 mm up the lever: pushed there, mid-rod, the fingertips do not ride over its knob)
    # right: vacuum OFF (push the lever back from in front of its knob)
    right += [
        Waypoint(2.5, tcp(knob(-0.42) + [-0.012, 0.0, 0.03]), FINGER_CLOSED, "above, in front of the vacuum toggle", dn),
        Waypoint(1.2, tcp(knob(-0.42) + [-0.012, 0.0, -0.002]), FINGER_CLOSED, "fingertips down in front of its knob", dn),
        Waypoint(1.5, tcp(knob(0.42) + [0.006, 0.0, -0.004]), FINGER_CLOSED, "push it back: MASK VACUUM off", dn),
        Waypoint(1.0, tcp(knob(0.42) + [0.0, 0.0, 0.05]), FINGER_CLOSED, "up", dn),
    ]
    left.append(Waypoint(_dur(right), LEFT_PARK, FINGER_OPEN, "wait"))
    c1 = SEAT_C
    r1 = np.array([*REST_L1_XY, TOP + R1_TOP + M_T / 2])
    both(
        _wp2(3.0, c1 + [0, 0, 0.03], OPEN_W, "beside the LAYER 1 mask's side edges, above", out=0.04),
        _wp2(1.5, c1, OPEN_W, "level with its edges", out=0.04),
        _wp2(1.5, c1, OPEN_W, "pads round its side edges, where they overhang the holder"),
        _wp2(0.8, c1, PINCH, "pinch them"),
        _wp2(1.5, c1 + [0, 0, 0.03], PINCH, "lift it off the holder"),
        _wp2(2.0, np.array([*c1[:2], r1[2] + 0.03]), PINCH, "up, above the rack's upper shelf"),
        _wp2(2.5, r1 + [0, 0, 0.03], PINCH, "carry it over its shelf"),
        _wp2(1.5, r1 + [0, 0, 0.001], PINCH, "lower it onto the rest"),
        _wp2(0.8, r1 + [0, 0, 0.001], OPEN_W, "let go"),
        _wp2(1.2, r1, OPEN_W, "back out sideways", out=0.04),
    )
    c2 = np.array([*REST_L2_XY, TOP + R_TOP + M_T / 2])
    lift = np.array([*H_XY, TOP + 0.20])  # above the holder, clear of the rack
    both(
        _wp2(2.5, c2 + [0, 0, 0.03], OPEN_W, "beside the LAYER 2 mask's side edges, above", out=0.04),
        _wp2(1.5, c2, OPEN_W, "level with its edges", out=0.04),
        _wp2(1.5, c2, OPEN_W, "pads round its side edges"),
        _wp2(0.8, c2, PINCH, "pinch them"),
        _wp2(1.5, c2 + [0, 0, 0.012], PINCH, "lift it off its shelf"),
        _wp2(2.5, np.array([*H_XY, c2[2] + 0.012]), PINCH, "draw it out from under the upper shelf"),
        _wp2(2.0, lift, PINCH, "up, clear of everything"),
    )
    for k in range(1, 9):
        R = axis_angle_matrix(Y, ROLL_SIGN * math.pi * k / 8)
        both(_wp2(0.8, lift, PINCH, "roll it over together: chrome side down", R))
    Rf = axis_angle_matrix(Y, ROLL_SIGN * math.pi)
    pre = SEAT_C + [-0.003, 0.0, 0.0]
    both(
        _wp2(3.0, pre + [0, 0, 0.04], PINCH, "carry it over the holder", Rf),
        _wp2(2.0, pre + [0, 0, 0.004], PINCH, "down, just in front of the stop pins", Rf),
        _wp2(1.0, pre + [0, 0, 0.0004], PINCH, "set it on the holder", Rf),
        _wp2(1.5, SEAT_C + [0.0006, 0.0, 0.0004], PINCH, "slide it back against both stop pins", Rf),
        _wp2(0.8, SEAT_C + [0.0006, 0.0, 0.0004], OPEN_W, "let go", Rf),
        _wp2(1.2, SEAT_C, OPEN_W, "back out sideways", Rf, out=0.05),
    )
    lo = _pre("left", SEAT_C, Rf, 0.05)
    ro = _pre("right", SEAT_C, Rf, 0.05)
    l_rest = tuple(np.asarray(lo[0]) + [0, 0, 0.06])
    left += [Waypoint(1.5, l_rest, OPEN_W, "up, and stay clear", lo[1])]
    right += [Waypoint(1.5, tuple(np.asarray(ro[0]) + [0, 0, 0.06]), OPEN_W, "up", ro[1])]
    right += [
        Waypoint(2.5, tcp(knob(0.42) + [0.012, 0.0, 0.03]), FINGER_CLOSED, "above, behind the vacuum toggle", dn),
        Waypoint(1.2, tcp(knob(0.42) + [0.012, 0.0, -0.002]), FINGER_CLOSED, "fingertips down behind its knob", dn),
        Waypoint(1.5, tcp(knob(-0.42) + [-0.006, 0.0, -0.004]), FINGER_CLOSED, "push it forward: MASK VACUUM on", dn),
        Waypoint(1.0, tcp(knob(-0.42) + [0.0, 0.0, 0.05]), FINGER_CLOSED, "up", dn),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), l_rest, OPEN_W, "rest", lo[1]))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
