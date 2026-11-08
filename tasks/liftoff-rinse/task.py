"""On a wet bench: a small PFA wafer cassette holding two 2 inch wafers that have just come out of gold evaporation
(gold over patterned photoresist), standing on a PTFE tray; a row of four 600 ml glass beakers, each with a printed
label card on the bench in front of it and a label on its side: ACE (acetone for ordinary resist strip), LIFT-OFF
(acetone kept for gold lift-off only), IPA and DI WATER; a PTFE drip dish on its stand; and the bench's 4 L solvent
waste bottle with a funnel. The two acetone beakers look the same and hold the same liquid. The wafers lie in the
cassette pattern side down, so what lifts off falls away from the pattern; the cassette must stay upright.

The task: lift the gold off the wafers and rinse them. Gold flakes from lift-off must only ever go into the dedicated
LIFT-OFF beaker, so the wet cassette never goes into (or drips over) the ACE beaker. It soaks in LIFT-OFF, then goes into
IPA before the acetone can dry on it (drying redeposits flakes), then into DI water, and is left standing in the DI
water. Whenever the wet cassette is out of the liquid and not over a beaker's mouth, it drips: the drip dish must be
under it. The dish is then emptied into the waste bottle's funnel (all solvent and rinse waste goes in that one bottle)
and put back on its stand.

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
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

GLASS = (0.86, 0.90, 0.88)  # borosilicate glass
LIQUID = (0.80, 0.88, 0.95)  # the solvents and water: clear, all alike
labkit.GLASS_COLORS.update({GLASS, LIQUID})
PFA = (0.93, 0.93, 0.90)  # PFA / PTFE: milky white
HDPE = (0.88, 0.89, 0.84)
PP = (0.80, 0.82, 0.78)
GOLD = (0.85, 0.68, 0.22)
STREET = (0.62, 0.50, 0.25)
CARD = (0.98, 0.98, 0.96)
INK = (0.03, 0.03, 0.04)
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

FONT = {  # 5 x 7, rows top to bottom
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
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


def _ring_mesh(r0, z0, r1, z1, colour, n=40):
    """A drawn-only surface of revolution from radius r0 at z0 to r1 at z1 (a cylinder or a cone), both ends open."""
    pts = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for r, z in ((r1, z1), (r0, z0))
           for i in range(n)]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, n + i, n + j), (i, n + j, j), (i, n + j, n + i), (i, j, n + j)]  # both faces
    return labkit.Mesh(points=np.array(pts, float), tris=np.array(tris), color=colour, collide="none")


def _cyl_mesh(r, z0, z1, colour, n=40):
    """A drawn-only closed vertical cylinder."""
    ring = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    pts = [(x, y, z1) for x, y in ring] + [(x, y, z0) for x, y in ring] + [(0.0, 0.0, z1), (0.0, 0.0, z0)]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(2 * n, i, j), (2 * n + 1, n + j, n + i), (i, n + i, n + j), (i, n + j, j)]
    return labkit.Mesh(points=np.array(pts, float), tris=np.array(tris), color=colour, collide="none")


def _glyphs(text, pix):
    """(u, v) centres of a line's lit pixels: u along the line from its middle, v up from its middle."""
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


def _wrap_text(text, pix, r, z):
    """Letters on a label wrapped round a vertical cylinder of radius r, centred on its -x side (towards the robots),
    read left to right from in front."""
    out = []
    for u, v in _glyphs(text, pix):
        a = math.pi - u / r
        out.append(((r * math.cos(a), r * math.sin(a), z + v), (0.0006, pix * 0.96, pix * 0.96), a))
    return out


def _wrap_band(width, height, r, z):
    n = max(int(width / 0.004), 1)
    seg = width / n
    out = []
    for i in range(n):
        a = math.pi - (-width / 2 + (i + 0.5) * seg) / r
        out.append(((r * math.cos(a), r * math.sin(a), z), (0.0006, seg * 1.05, height), a))
    return out


# -- the beakers (fixed): 600 ml borosilicate glass, 90 mm across, 125 mm tall, filled to 60 mm (about 350 ml); each
# with a white label round its front (towards the robots). Frame: origin on the table under the beaker's centre. ------
B_OUT, B_WALL, B_H, FILL = 0.045, 0.0025, 0.125, 0.060
B_FLOOR = 0.003
N_SEG = 36
ROW_X = 0.33


def _beaker(name, xy, tag):
    r = B_OUT - B_WALL / 2
    seg = 2 * math.pi * r / N_SEG * 1.06
    walls = [((r * math.cos(a), r * math.sin(a), B_H / 2), (B_WALL, seg, B_H), GLASS, a)
             for a in (2 * math.pi * (i + 0.5) / N_SEG for i in range(N_SEG))]
    pix = 0.0015
    w = len(tag) * 6 * pix + 0.008
    return part(name, pos=(*xy, TOP), fixed=True, friction=0.6,
                boxes=[*walls, ((0.0, 0.0, B_FLOOR / 2), (2 * (B_OUT - B_WALL), 2 * (B_OUT - B_WALL), B_FLOOR), GLASS)],
                meshes=[_cyl_mesh(B_OUT - B_WALL - 0.0005, B_FLOOR, FILL, LIQUID),
                        _boxes_mesh(_wrap_band(w, 0.016, B_OUT + 0.0005, 0.095), CARD),
                        _boxes_mesh(_wrap_text(tag, pix, B_OUT + 0.0012, 0.095), INK)])


BEAKERS = {  # name: (xy, side label, card lines)
    "beaker_acetone": ((ROW_X, -0.265), "ACETONE", ("ACE",)),
    "beaker_liftoff": ((ROW_X, -0.17), "LIFT-OFF", ("LIFT", "OFF")),
    "beaker_ipa": ((ROW_X, -0.075), "IPA", ("IPA",)),
    "beaker_di": ((ROW_X, 0.02), "DI WATER", ("DI",)),
}
BEAKER_PARTS = [_beaker(n, xy, t) for n, (xy, t, _) in BEAKERS.items()]
ORDER = ("beaker_liftoff", "beaker_ipa", "beaker_di")

# -- the label cards (fixed): laminated white cards, 90 x 70 mm, lying on the bench in front of each beaker with its
# contents printed in 25 mm letters. ----------------------------------------------------------------------------------
CARD_SIZE = (0.070, 0.090, 0.002)
CARD_PIX = 0.0036
CARD_X = ROW_X - B_OUT - 0.005 - CARD_SIZE[0] / 2
CARDS = [part(f"card_{n.split('_')[1]}", pos=(CARD_X, xy[1], TOP), fixed=True,
              boxes=[((0.0, 0.0, CARD_SIZE[2] / 2), CARD_SIZE, CARD)],
              meshes=[_boxes_mesh(_card_text(lines, CARD_PIX, CARD_SIZE[2] + 0.0003), INK)])
         for n, (xy, _, lines) in BEAKERS.items()]

# -- the cassette (loose): a small PFA lift-off cassette for 2 inch wafers: four corner posts with slot tabs holding two
# wafers flat, PATTERN SIDE DOWN (gold over resist underneath, so what lifts off falls away from the pattern; the bare
# silicon backs face up), and a round handle bar on top (along y). Frame: origin on the bottom of its feet. ------------
C_XY, C_H = 0.054, 0.040  # its square footprint (corners 4 mm inside a beaker) and its frame height
W_R, W_T = 0.0254, 0.0006
SLOT_Z = (0.012, 0.026)  # the wafers' mid-planes above the feet
HANDLE_H = 0.040  # the handle bar's top above the frame's top
BAR_R, BAR_L = 0.004, 0.050
SI = (0.55, 0.56, 0.60)
_pc = C_XY / 2 - 0.004  # the posts' centres from the middle


def _wafer_mesh(z):
    """A wafer lying flat at height z: grey silicon back on top, gold film with darker streets on its underside."""
    out = [_cyl_mesh(W_R, z, z + W_T / 2, SI), _cyl_mesh(W_R, z - W_T / 2, z, GOLD)]
    lines = []
    for k in range(-4, 5):
        u = 0.0045 * k
        half = math.sqrt(max((W_R - 0.003) ** 2 - u * u, 0.0))
        lines += [((u, 0.0, z - W_T / 2 - 0.0001), (0.0005, 2 * half, 0.0001)),
                  ((0.0, u, z - W_T / 2 - 0.0001), (2 * half, 0.0005, 0.0001))]
    out.append(_boxes_mesh(lines, STREET))
    return out


GRIP_Z = C_H + HANDLE_H - BAR_R  # the handle bar's axis above the cassette's feet
TRAY_XY = (0.15, -0.20)
CASSETTE_START = np.array([*TRAY_XY, TOP + 0.004])
_posts = [(sx * _pc, sy * _pc) for sx in (-1, 1) for sy in (-1, 1)]
CASSETTE = part(
    "cassette", pos=tuple(CASSETTE_START), mass=0.15, friction=0.8,
    boxes=[
        *[((px, py, C_H / 2), (0.006, 0.006, C_H), PFA) for px, py in _posts],
        # the frame: bottom and top rings of bars between the posts
        *[((0.0, e * _pc, z), (2 * _pc, 0.004, 0.004), PFA) for e in (-1, 1) for z in (0.002, C_H - 0.002)],
        *[((e * _pc, 0.0, z), (0.004, 2 * _pc, 0.004), PFA) for e in (-1, 1) for z in (0.002, C_H - 0.002)],
        # the slot tabs the wafers rest on, reaching in from each post under the wafers' rims
        *[((px * 0.80, py * 0.80, sz - W_T / 2 - 0.001), (0.010, 0.010, 0.002), PFA) for px, py in _posts
          for sz in SLOT_Z],
        # the handle: two posts on a cross bar
        *[((0.0, e * (BAR_L / 2 - 0.003), C_H + HANDLE_H / 2), (0.006, 0.006, HANDLE_H), PFA) for e in (-1, 1)],
        ((0.0, 0.0, C_H - 0.002), (0.010, BAR_L, 0.004), PFA),
    ],
    cylinders=[((0.0, 0.0, GRIP_Z), BAR_R, BAR_L, PFA, "Y")],
    meshes=[m for sz in SLOT_Z for m in _wafer_mesh(sz)],
)

TRAY = part("tray", pos=(*TRAY_XY, TOP), fixed=True, friction=0.8,
            boxes=[((0.0, 0.0, 0.002), (0.080, 0.090, 0.004), PFA)])

# -- the drip dish (loose): a PTFE dish 100 mm across and 15 mm deep with a flat handle tab (45 x 24 x 8 mm) pointing
# towards the robots, resting on a PTFE stand (60 mm across, 130 mm tall). Frame: origin under the dish's centre. ------
D_R, D_H, D_WALL = 0.050, 0.015, 0.003
TAB = ((-(D_R - 0.0025) - 0.0225, 0.0, 0.011), (0.045, 0.024, 0.008))
TAB_TIPS = np.array([-0.072, 0.0, 0.011])  # where the fingertips sit on the tab, 20 mm in from its end
STAND_XY, STAND_H = (ROW_X, 0.13), 0.130
_dn = 24
DISH = part(
    "drip_dish", pos=(*STAND_XY, TOP + STAND_H), mass=0.06, friction=0.9,
    boxes=[*[(((D_R - D_WALL / 2) * math.cos(a), (D_R - D_WALL / 2) * math.sin(a), D_H / 2),
               (D_WALL, 2 * math.pi * D_R / _dn * 1.08, D_H), PFA, a)
             for a in (2 * math.pi * (i + 0.5) / _dn for i in range(_dn))],
           (*TAB, PFA)],
    cylinders=[((0.0, 0.0, 0.0015), D_R - D_WALL, 0.003, PFA)],
)
STAND = part("dish_stand", pos=(*STAND_XY, TOP), fixed=True, friction=0.9,
             cylinders=[((0.0, 0.0, STAND_H / 2), 0.030, STAND_H, PFA)])

# -- the waste bottle (fixed): a 4 L HDPE solvent waste bottle (170 mm across, 170 mm body) with a polypropylene funnel
# (120 mm mouth, rim 235 mm above the bench) and a WASTE label. ----------------------------------------------------
JUG_XY = (0.42, 0.29)
JUG_R, JUG_H = 0.085, 0.170
FUN_R, FUN_TOP = 0.060, 0.235
_fn = 30
JUG = part(
    "waste_bottle", pos=(*JUG_XY, TOP), fixed=True, friction=0.6,
    cylinders=[((0.0, 0.0, JUG_H / 2), JUG_R, JUG_H, HDPE), ((0.0, 0.0, JUG_H + 0.010), 0.050, 0.020, HDPE),
               ((0.0, 0.0, JUG_H + 0.025), 0.020, 0.010, HDPE)],
    boxes=[(((FUN_R - 0.002) * math.cos(a), (FUN_R - 0.002) * math.sin(a), FUN_TOP - 0.005),
            (0.003, 2 * math.pi * FUN_R / _fn * 1.08, 0.010), PP, a)
           for a in (2 * math.pi * (i + 0.5) / _fn for i in range(_fn))],
    meshes=[_ring_mesh(0.012, JUG_H + 0.025, FUN_R, FUN_TOP, PP),
            _boxes_mesh(_wrap_band(0.090, 0.024, JUG_R + 0.0005, 0.110), CARD),
            _boxes_mesh(_wrap_text("WASTE", 0.0026, JUG_R + 0.0012, 0.110), INK)],
)

OBJECTS = [*BEAKER_PARTS, *CARDS, TRAY, CASSETTE, STAND, DISH, JUG]
PLANCHECK_BOXES = [  # for the offline plan check only: world-frame, axis-aligned
    *[(n, (xy[0], xy[1], TOP + B_H / 2), (2 * B_OUT, 2 * B_OUT, B_H)) for n, (xy, _, _) in BEAKERS.items()],
    ("dish_stand", (*STAND_XY, TOP + STAND_H / 2), (0.06, 0.06, STAND_H)),
    ("waste_bottle", (*JUG_XY, TOP + FUN_TOP / 2), (2 * JUG_R, 2 * JUG_R, FUN_TOP)),
]

# -- scoring ----------------------------------------------------------------------------------------------------------
SOAK = {"beaker_liftoff": 12.0, "beaker_ipa": 5.0, "beaker_di": 5.0}  # minimum soaks, s (shortened for the 2 min run)
DRY_LIMIT = 10.0  # s: out of liquid between LIFT-OFF and IPA, or the acetone dries on the wafers
IN_XY = 0.030  # the cassette counts as in a beaker with its centre within 30 mm of the beaker's axis
IN_Z = 0.030  # and its feet within 30 mm of the beaker's floor (the wafers then under the liquid)
MOUTH = 0.040  # out of the liquid, the cassette drips into a beaker while its centre is within 40 mm of its axis
CATCH = 0.035  # or into the drip dish while its centre is within 35 mm of the dish's, above the dish's rim
TILT_MAX = 30.0  # degrees: tipped (the patterns no longer face down)
TOUCH = 0.004  # a pad tip within 4 mm of a wafer: touched
DISH_SPILL = 30.0  # degrees: a wet dish tipped further away from the funnel spills
POUR_TILT, POUR_XY = 60.0, 0.050  # emptied into the funnel: tipped past 60 degrees, centre within 50 mm of its axis
HANDS_FAR = 0.08  # both hands' pad tips this far from the handle bar and the dish's tab at the end
DT = 0.05
SETTLE_STEPS, SETTLE_TOL = 20, 0.001
_state = {}
_hist = []


@on_reset
def _reset_state(env):
    _state.update(contaminated=False, tipped=False, touched=False, spilled=False, wet=False, dish_wet=False,
                  poured=False, far=True, settled=False, dish_home=False, visits=[], cur=None, t=0.0)
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


def which_beaker(pos):
    for n, (xy, _, _) in BEAKERS.items():
        if math.hypot(pos[0] - xy[0], pos[1] - xy[1]) <= IN_XY and pos[2] <= TOP + B_FLOOR + IN_Z:
            return n
    return None


def over_mouth(pos):
    for n, (xy, _, _) in BEAKERS.items():
        if math.hypot(pos[0] - xy[0], pos[1] - xy[1]) <= MOUTH and pos[2] >= TOP + B_FLOOR:
            return n
    return None


def tilt_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))


@on_step
def _watch(env):
    """Record the cassette's baths (name, time in, time out) and latch the violations: wet cassette in or over the ACE
    beaker, dripping where nothing catches it, a wet dish spilled, the cassette tipped, a wafer touched."""
    S = labkit._SPECS
    c = labkit._part_state(env, S["cassette"])
    d = labkit._part_state(env, S["drip_dish"])
    p, R = np.asarray(c.pos, float), c.rotation()
    dp, dR = np.asarray(d.pos, float), d.rotation()
    _state["t"] += DT
    t = _state["t"]
    b = which_beaker(p)
    if b != _state["cur"]:
        if _state["cur"] is not None:
            _state["visits"][-1][2] = t
        if b is not None:
            _state["visits"].append([b, t, None])
        _state["cur"] = b
    if b is not None:
        _state["wet"] = True
    if b == "beaker_acetone":
        _state["contaminated"] = True
    if _state["wet"] and b is None:
        m = over_mouth(p)
        if m == "beaker_acetone":
            _state["contaminated"] = True
        if m is None:
            caught = (tilt_deg(dR) < 15.0 and math.hypot(p[0] - dp[0], p[1] - dp[1]) <= CATCH
                      and dp[2] + D_H <= p[2] + 0.002)
            if caught:
                _state["dish_wet"] = True
            else:
                _state["spilled"] = True
    dt = tilt_deg(dR)
    over_funnel = math.hypot(dp[0] - JUG_XY[0], dp[1] - JUG_XY[1]) <= POUR_XY and dp[2] > TOP + FUN_TOP - 0.02
    if _state["dish_wet"]:
        if over_funnel and dt > POUR_TILT:
            _state["dish_wet"] = False
            _state["poured"] = True
        elif not over_funnel and dt > DISH_SPILL:
            _state["spilled"] = True
    if tilt_deg(R) > TILT_MAX:
        _state["tipped"] = True
    tips = _tips(env)
    for sz in SLOT_Z:
        wc = p + R @ np.array([0.0, 0.0, sz])
        for tip in tips:
            q = R.T @ (np.asarray(tip, float) - wc)
            if abs(q[2]) < TOUCH and math.hypot(q[0], q[1]) < W_R:
                _state["touched"] = True
    bar = p + R @ np.array([0.0, 0.0, GRIP_Z])
    tab = dp + dR @ TAB_TIPS
    _state["far"] = all(min(np.linalg.norm(np.asarray(tip) - bar), np.linalg.norm(np.asarray(tip) - tab)) >= HANDS_FAR
                        for tip in tips)
    _state["dish_home"] = (math.hypot(dp[0] - STAND_XY[0], dp[1] - STAND_XY[1]) <= 0.015
                           and abs(dp[2] - (TOP + STAND_H)) <= 0.005 and dt < 10.0)
    _hist.append(np.concatenate([p, dp]))
    del _hist[:-(SETTLE_STEPS + 1)]
    _state["settled"] = (len(_hist) > SETTLE_STEPS
                         and float(np.max(np.abs(np.asarray(_hist) - _hist[-1]))) < SETTLE_TOL)


def progress():
    """How far the bath sequence went right: 0 nothing; 1 soaked in LIFT-OFF first; 2 then into IPA within the dry
    limit and soaked; 3 then into DI WATER and soaked, and still in it."""
    v = _state.get("visits", [])
    t_now = _state.get("t", 0.0)
    done = 0
    for k, name in enumerate(ORDER):
        if len(v) <= k or v[k][0] != name:
            break
        t_out = v[k][2] if v[k][2] is not None else t_now
        if t_out - v[k][1] < SOAK[name]:
            break
        if k == 1 and v[1][1] - (v[0][2] or t_now) > DRY_LIMIT:
            break
        done = k + 1
    if done == 3 and (len(v) != 3 or _state.get("cur") != "beaker_di"):
        done = 2
    return done


def score(objects):
    """0 if the wet cassette was ever in or over the ACE beaker, ever dripped where nothing caught it (out of the liquid,
    not over a beaker's mouth and not over the drip dish), the wet dish was tipped past 30 degrees anywhere but over the
    funnel, the cassette tipped over 30 degrees, a wafer was touched by the gripper, or the cassette and dish have not
    been still (under 1 mm) for the last second. Otherwise by how far the baths went right, in order: 0.25 for a first
    soak of at least 12 s in LIFT-OFF; 0.5 if it then went into IPA within 10 s of leaving LIFT-OFF and soaked at least
    5 s; 0.75 if it then went into DI WATER, soaked at least 5 s and stands in it at the end, with no other bath visited;
    1.0 if, in addition, the dish was emptied into the funnel (tipped past 60 degrees over it) after its last catch, is
    back on its stand, and both hands' fingertips are 8 cm from the cassette's handle bar and the dish's tab."""
    if any(_state.get(k) for k in ("contaminated", "tipped", "touched", "spilled")):
        return 0.0
    if not _state.get("settled", False):
        return 0.0
    k = progress()
    if k < 3:
        return 0.25 * k
    full = (not _state.get("dish_wet") and _state.get("poured") and _state.get("dish_home")
            and _state.get("far", True))
    return 1.0 if full else 0.75


SUCCESS_HOLD_S = 3.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    xy = BEAKERS["beaker_di"][0]
    p = np.asarray(objects["cassette"].pos, float)
    return float(math.hypot(p[0] - xy[0], p[1] - xy[1]))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right hand: takes the cassette by its handle bar (hand tilted 30 degrees back), lifts it clear of the rims, soaks it
# in LIFT-OFF, then carries it to IPA and to DI WATER, lets go and moves away. Left hand: takes the drip dish by its
# tab and keeps it under the cassette wherever the cassette is out of the liquid between two beakers (sliding it along
# just above the rims, out of the way of each descent), then pours it into the waste funnel and puts it back. The demo
# follows a fixed timeline; it reads nothing but the scene layout.
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
OPEN_W = -0.016
HOLD = -0.0025  # a hand holding: the right squeezes the 8 mm bar, the left the 8 mm tab, firmly
R_HOLD, L_HOLD = -0.0015, -0.0012
R_ROT = axis_angle_matrix(Y, math.radians(-30.0)) @ _prev_grip(math.pi / 2)  # down, tilted 30 deg back, across y
_la = X
L_ROT = np.column_stack([Z, np.cross(_la, Z), _la])  # pointing +x, fingers closing vertically on the tab
CARRY = B_H + 0.050  # the cassette's feet this far above the bench while carried (the dish passes 20 mm under it)
DISH_Z = B_H + 0.015  # the dish's underside this far up while it works over the beakers (10 mm over its stand)
POUR_ROLL = math.radians(100.0)


# what the demo sees of the cassette: while the right hand holds it still over a beaker, its aim is trimmed by how far
# the cassette hangs from where it should (grip offset and the arm's sag under load), so it goes down the beaker's middle
_ct = {"trim": np.zeros(3), "wins": [], "last": None}


@on_reset
def _reset_ct(env):
    _ct["trim"] = np.zeros(3)
    _ct["last"] = None


@on_step
def _ctrack(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    win = next((w for w in _ct["wins"] if w[0] <= t < w[1]), None)
    p = np.asarray(labkit._part_state(env, labkit._SPECS["cassette"]).pos, float)
    last, _ct["last"] = _ct["last"], p
    if win is None or last is None or np.max(np.abs(p - last)) > 0.0005:
        return
    e = np.asarray(win[2], float) - p
    _ct["trim"] = np.clip(_ct["trim"] + np.clip(0.2 * e, -0.0003, 0.0003), -0.025, 0.025)


class _Dyn:
    """A TCP plus the live cassette trim."""

    def __init__(self, tcp):
        self.tcp = np.asarray(tcp, float)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"], dtype=dtype or float)


def _rtcp(cas_origin, back=0.0):
    a = R_ROT[:, 2]
    bar = np.asarray(cas_origin, float) + [0.0, 0.0, GRIP_Z]
    return tuple(bar - a * (TIP_ABOVE_TCP - BAR_R + back))


def _ltcp(dish_origin, back=0.0):
    return tuple(np.asarray(dish_origin, float) + TAB_TIPS - _la * (TIP_ABOVE_TCP + back))


def _cas(xy, z):
    return np.array([xy[0], xy[1], TOP + z])


def demo():
    rr, lr = Orient(R_ROT), Orient(L_ROT)
    lift, ipa, di = (BEAKERS[n][0] for n in ORDER)
    seat = B_FLOOR + 0.002
    R, L = [], []
    _ct["trim"], _ct["wins"] = np.zeros(3), []
    clock = [0.0]

    def ph(dur, r=None, l=None, see=None):
        """One step of the shared timeline: each hand moves to its new target or holds where it is. With ``see`` (the
        cassette's origin as it should be), the right hand's aim is trimmed by the cassette's seen position meanwhile."""
        for plan, w, hold in ((R, r, R_HOLD), (L, l, L_HOLD)):
            if w:
                pos, finger = w[0], (hold if w[1] == HOLD else w[1])
                if plan is R and finger == R_HOLD and len(R) > 3:
                    pos = _Dyn(pos)
                plan.append(Waypoint(dur, pos, finger, *w[2:]))
            else:
                plan.append(Waypoint(dur, plan[-1].pos, plan[-1].finger, "hold", plan[-1].yaw))
        if see is not None:
            _ct["wins"].append((clock[0], clock[0] + dur, np.asarray(see, float)))
        clock[0] += dur

    d0 = np.array([*STAND_XY, TOP + STAND_H])
    dish = lambda xy, z=DISH_Z: np.array([xy[0], xy[1], TOP + z])  # noqa: E731
    pre_g1, g1, over_di, g2, out = (ROW_X, -0.08), (ROW_X, -0.1225), (ROW_X, 0.15), (ROW_X, -0.0275), (ROW_X, 0.15)
    pour_tcp = np.array([JUG_XY[0] + TAB_TIPS[0] - TIP_ABOVE_TCP, JUG_XY[1], TOP + 0.30])
    s = CASSETTE_START
    up = np.array([0.0, 0.0, 0.07])
    ph(2.0, (_rtcp(s, 0.08), OPEN_W, "behind the cassette's handle bar", rr),
       (tuple(np.array(_ltcp(d0, 0.03)) + up), OPEN_W, "above and behind the dish's tab", lr))
    ph(1.2, (_rtcp(s, 0.03), OPEN_W, "closer", rr), (_ltcp(d0, 0.03), OPEN_W, "behind the tab", lr))
    ph(1.3, (_rtcp(s), OPEN_W, "fingers either side of the bar", rr), (_ltcp(d0), OPEN_W, "fingers over and under the tab", lr))
    ph(0.8, (_rtcp(s), HOLD, "hold the bar", rr), (_ltcp(d0), HOLD, "hold the tab", lr))
    ph(2.0, (_rtcp(_cas(TRAY_XY, CARRY)), HOLD, "lift it clear of the rims", rr),
       (_ltcp(dish(STAND_XY)), HOLD, "lift the dish off its stand", lr))
    ph(3.0, (_rtcp(_cas(lift, CARRY)), HOLD, "over the LIFT-OFF beaker", rr),
       (_ltcp(dish(pre_g1)), HOLD, "wait beside the LIFT-OFF beaker's mouth", lr))
    ph(2.5, None, None, see=_cas(lift, CARRY))
    ph(2.0, (_rtcp(_cas(lift, seat)), HOLD, "down into LIFT-OFF", rr))
    ph(SOAK["beaker_liftoff"] + 1.0, (_rtcp(_cas(lift, seat - 0.001)), HOLD, "soak", rr))
    ph(2.0, (_rtcp(_cas(lift, CARRY)), HOLD, "lift it out over the mouth", rr))
    ph(1.2, None, (_ltcp(dish(g1)), HOLD, "dish under the gap to IPA", lr))
    ph(2.0, (_rtcp(_cas(ipa, CARRY)), HOLD, "across to IPA over the dish", rr))
    ph(1.5, None, (_ltcp(dish(over_di)), HOLD, "dish on along, out of the way", lr), see=_cas(ipa, CARRY))
    ph(1.8, (_rtcp(_cas(ipa, seat)), HOLD, "down into IPA", rr))
    ph(SOAK["beaker_ipa"] + 1.0, (_rtcp(_cas(ipa, seat - 0.001)), HOLD, "soak", rr))
    ph(2.0, (_rtcp(_cas(ipa, CARRY)), HOLD, "lift it out", rr))
    ph(1.2, None, (_ltcp(dish(g2)), HOLD, "dish under the gap to DI", lr))
    ph(2.0, (_rtcp(_cas(di, CARRY)), HOLD, "across to DI over the dish", rr))
    ph(1.5, None, (_ltcp(dish(out)), HOLD, "dish on along, over its stand", lr), see=_cas(di, CARRY))
    ph(2.0, (_rtcp(_cas(di, seat)), HOLD, "down into DI", rr))
    ph(1.5, (_rtcp(_cas(di, seat - 0.001)), HOLD, "rinse", rr),
       (_ltcp(dish((ROW_X, 0.19), 0.22)), HOLD, "up and away from the right hand", lr))
    ph(2.0, None, (tuple(pour_tcp), HOLD, "over the funnel", lr))
    ph(2.0, None, (tuple(pour_tcp), HOLD, "pour", Orient(axis_angle_matrix(X, POUR_ROLL) @ L_ROT)))
    ph(1.5)
    ph(2.0, None, (tuple(pour_tcp), HOLD, "level again", lr))
    ph(2.0, (_rtcp(_cas(di, seat - 0.001)), OPEN_W, "let go: it stands in the DI water", rr),
       (_ltcp(dish((ROW_X, 0.19), 0.22)), HOLD, "wait aside", lr))
    ph(1.5, (tuple(np.array(_rtcp(_cas(di, seat))) + [0, 0, 0.08]), OPEN_W, "straight up", rr))
    ph(1.5, (RIGHT_PARK, FINGER_OPEN, "park", 0.0))
    ph(1.5, None, (_ltcp(dish(STAND_XY, STAND_H + 0.03)), HOLD, "over its stand", lr))
    ph(1.2, None, (_ltcp(dish(STAND_XY, STAND_H + 0.001)), HOLD, "on its stand", lr))
    ph(0.8, None, (_ltcp(dish(STAND_XY, STAND_H)), OPEN_W, "let go", lr))
    ph(1.0, None, (_ltcp(dish(STAND_XY, STAND_H), 0.03), OPEN_W, "back off the tab", lr))
    ph(1.2, None, (tuple(np.array(_ltcp(dish(STAND_XY, STAND_H), 0.03)) + up + [0, 0, 0.03]), OPEN_W, "up and away", lr))
    ph(2.0, None, (tuple(np.array(_ltcp(dish(STAND_XY, STAND_H), 0.08)) + up + [0, 0, 0.05]), FINGER_OPEN, "park", lr))
    ph(2.0)
    return {"left": L, "right": R}


# --- debug (not for submission) ---
_dbgr = {"n": 0}
@on_step
def _dbg_lo(env):
    _dbgr["n"] += 1
    if _dbgr["n"] % 10:
        return
    S = labkit._SPECS
    c = labkit._part_state(env, S["cassette"]); d = labkit._part_state(env, S["drip_dish"])
    fl = "".join(k[0].upper() if _state.get(k) else "." for k in
                 ("contaminated", "tipped", "touched", "spilled", "wet", "dish_wet", "poured", "dish_home", "far", "settled"))
    open("/home/ubuntu/dbg_lo.txt", "a").write(
        f"{_dbgr['n']} t={_state['t']:.1f} cas={np.round((np.asarray(c.pos) - [0, 0, TOP]) * 1000, 1).tolist()} "
        f"ct={tilt_deg(c.rotation()):.1f} dish={np.round((np.asarray(d.pos) - [0, 0, TOP]) * 1000, 1).tolist()} "
        f"dt={tilt_deg(d.rotation()):.1f} tr={np.round(_ct['trim']*1000,1).tolist()} cur={_state['cur']} {fl} k={progress()} s={score(None)}\n")
