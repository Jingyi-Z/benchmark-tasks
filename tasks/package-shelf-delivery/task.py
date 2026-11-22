"""Doorstep (apartment parcel-shelf) delivery, STEP A baseline. The courier's cart deck is the bench. In front of it stands
an apartment corridor's parcel shelf for unit 12 (top 0.80 m above the floor, its front edge carrying a "12" unit label).
A small shipping box (Uline S-20491 6 x 4 x 2 in corrugated, outside 162 x 67 x 111 mm as stood, 0.8 kg packed, barcode label on top) sits on the cart deck; the courier's phone
(a 6.1 in phone, 147.6 x 71.6 x 7.8 mm, 171 g) stands upside down in its dock on the deck.

The task: set the box on the unit-12 shelf, label up, then pick up the phone, frame the delivered box and the unit label,
and press its volume-up button (the phone's camera shutter) to take the photo; put the phone back in its dock.

Everything the scorer reads is camera-visible: rigid poses of the box, phone and fixtures, and the pad tips. The proof
photo is the phone camera's view at the shutter tap (a pinhole model of the 26 mm main camera, portrait 69.4 x 54.4 deg).

Run:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
        --instruction "<see task.json>" --scorer task_score --max-steps 2400 --store-frames --log-dir logs
"""

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401,E402
import cart_scene_fix  # noqa: F401,E402
import labkit  # noqa: E402
from labkit import on_reset, on_step, part  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]          # cart deck (bench) surface
DEBUG = bool(os.environ.get("PKG_DEBUG"))

KRAFT, WHITE, BLACK, WOOD, GREY, SCREEN = (0.72, 0.55, 0.36), (0.95, 0.95, 0.93), (0.05, 0.05, 0.05), (0.55, 0.42, 0.28), (0.35, 0.35, 0.37), (0.08, 0.10, 0.14)
PINE, PINE_EDGE, WALL, SKIRT = (0.86, 0.72, 0.50), (0.70, 0.55, 0.34), (0.88, 0.87, 0.83), (0.30, 0.30, 0.32)
STEEL, CART_GREY, RUBBER = (0.62, 0.63, 0.66), (0.20, 0.21, 0.23), (0.06, 0.06, 0.07)
TITANIUM, ISLAND, TAPE = (0.56, 0.56, 0.58), (0.30, 0.30, 0.32), (0.62, 0.45, 0.25)       # phone frame and back, camera island

SHELF_TOP = 0.80
SH_X0, SH_X1, SH_Y = 0.24, 0.44, 0.40          # shelf top spans x [0.24, 0.44], y [-0.20, 0.20]
SH_T = 0.018
LABEL_Y = float(os.environ.get("PKG_LABEL_Y", "-0.105"))   # plaques sit 105 mm right of each bay centre, over the delivery spot
PLAQUE_Z = float(os.environ.get("PKG_PLAQUE_Z", "1.000"))   # unit plaque centre height on the wall
LABEL_W, LABEL_H, DIGIT_H = 0.100, 0.060, 0.050


def _digit_segments(cx, cy, cz, d):
    """'1' and '2' as seven-segment bars (black, 6 mm stroke) on the label face (x = cx), digit height d."""
    s, w = 0.006, d * 0.55
    segs = []
    x = cx - 0.001
    # the label faces -x, so a reader's left is +y. '1' on the left: right-hand vertical stroke, full height
    y1 = cy + 0.020
    segs.append(((x, y1 - w / 2 + s / 2, cz), (0.001, s, d), BLACK))
    # '2' on the right: top, upper-right (-y), middle, lower-left (+y), bottom
    y2 = cy - 0.022
    segs += [((x, y2, cz + d / 2 - s / 2), (0.001, w, s), BLACK), ((x, y2 - w / 2 + s / 2, cz + d / 4), (0.001, s, d / 2), BLACK),
             ((x, y2, cz), (0.001, w, s), BLACK), ((x, y2 + w / 2 - s / 2, cz - d / 4), (0.001, s, d / 2), BLACK),
             ((x, y2, cz - d / 2 + s / 2), (0.001, w, s), BLACK)]
    return segs


# -- apartment corridor: three wall-mounted parcel shelves, units 11, 12 and 13 from left (+y) to right (-y) as seen from
# the cart. Each is an 18 mm pine board, top 0.80 m above the floor, x 0.24..0.44, 400 mm wide, on two steel brackets from
# the wall, open air to the floor. Unit 12 (the delivery target) is the middle bay; its plaque hangs on the wall above it.
FLOOR = -0.0125                                 # station floor top
WALL_X = SH_X1 + 0.010                          # wall face, 10 mm behind the shelves' back edges
BAY_PITCH = SH_Y + 0.018
BAYS = {"11": BAY_PITCH, "12": 0.0, "13": -BAY_PITCH}
LABEL_C = np.array([WALL_X - 0.0015, LABEL_Y, PLAQUE_Z])   # unit-12 plaque centre, on the wall, facing -x
SHELF_BOXES = [
    (((SH_X0 + SH_X1) / 2, 0.0, SHELF_TOP - SH_T / 2), (SH_X1 - SH_X0, SH_Y, SH_T), PINE),                 # unit-12 board
    (tuple(LABEL_C), (0.003, LABEL_W, LABEL_H), WHITE),                                                    # unit-12 plaque
] + _digit_segments(LABEL_C[0] - 0.0015, LABEL_C[1], LABEL_C[2], DIGIT_H)


_SEG = {"1": "bc", "2": "abged", "3": "abgcd"}


def _digits(text, cx, cy, cz, d):
    """seven-segment digits on a wall plaque facing -x (a reader's right is -y); '1' is one full-height stroke."""
    s, w, x = 0.006, d * 0.55, cx - 0.001
    out = []
    for k, ch in enumerate(text):
        yc = cy + (0.021 if k == 0 else -0.021)
        if ch == "1":
            out.append(((x, yc - w / 2 + s / 2, cz), (0.001, s, d), BLACK)); continue
        yr, yl = yc - w / 2 + s / 2, yc + w / 2 - s / 2
        seg = {"a": ((x, yc, cz + d / 2 - s / 2), (0.001, w, s)), "g": ((x, yc, cz), (0.001, w, s)),
               "d": ((x, yc, cz - d / 2 + s / 2), (0.001, w, s)),
               "b": ((x, yr, cz + d / 4), (0.001, s, d / 2)), "c": ((x, yr, cz - d / 4), (0.001, s, d / 2)),
               "f": ((x, yl, cz + d / 4), (0.001, s, d / 2)), "e": ((x, yl, cz - d / 4), (0.001, s, d / 2))}
        out += [(seg[q][0], seg[q][1], BLACK) for q in _SEG[ch]]
    return out


def _brackets(yc):
    out = []
    for yb in (yc - 0.14, yc + 0.14):
        out += [((WALL_X - 0.003, yb, SHELF_TOP - SH_T - 0.045), (0.006, 0.025, 0.090), STEEL),        # wall plate
                (((SH_X0 + 0.02 + WALL_X) / 2, yb, SHELF_TOP - SH_T - 0.005), (WALL_X - SH_X0 - 0.02, 0.025, 0.010), STEEL),  # arm
                ((WALL_X - 0.045, yb, SHELF_TOP - SH_T - 0.040), (0.070, 0.006, 0.060), STEEL)]        # gusset
    return out


# solid (collider) scenery: the other two boards, all brackets, the wall
WALL_Y = (-BAY_PITCH - SH_Y / 2 - 0.15, BAY_PITCH + SH_Y / 2 + 0.15)
EXTRA_COLLIDERS = [
    (((SH_X0 + SH_X1) / 2, BAYS["11"], SHELF_TOP - SH_T / 2), (SH_X1 - SH_X0, SH_Y, SH_T), PINE),
    (((SH_X0 + SH_X1) / 2, BAYS["13"], SHELF_TOP - SH_T / 2), (SH_X1 - SH_X0, SH_Y, SH_T), PINE),
    ((WALL_X + 0.010, sum(WALL_Y) / 2, (FLOOR + 1.25) / 2), (0.020, WALL_Y[1] - WALL_Y[0], 1.25 - FLOOR), WALL),
] + _brackets(BAYS["11"]) + _brackets(0.0) + _brackets(BAYS["13"])

# render-only: plaques 11 and 13, skirting, and the courier cart (rim on the deck edge, legs, lower tray, wheels with
# brake tabs, push handle at the left end). The deck itself replaces the station table (cart_scene_fix.py).
CART_X, CART_Y, CART_TOP = (-0.32, 0.21), (-0.45, 0.45), TOP
_leg_h = CART_TOP - 0.04 - FLOOR - 0.10
VISUAL_BOXES = [
    ((WALL_X - 0.001, sum(WALL_Y) / 2, FLOOR + 0.050), (0.002, WALL_Y[1] - WALL_Y[0], 0.100), SKIRT),       # skirting
    ((WALL_X - 0.0015, BAYS["11"] + LABEL_Y, PLAQUE_Z), (0.003, LABEL_W, LABEL_H), WHITE),
    ((WALL_X - 0.0015, BAYS["13"] + LABEL_Y, PLAQUE_Z), (0.003, LABEL_W, LABEL_H), WHITE),
] + _digits("11", WALL_X - 0.003, BAYS["11"] + LABEL_Y, PLAQUE_Z, DIGIT_H) + _digits("13", WALL_X - 0.003, BAYS["13"] + LABEL_Y, PLAQUE_Z, DIGIT_H) + [
    ((CART_X[1] - 0.006, 0.0, CART_TOP + 0.006), (0.012, CART_Y[1] - CART_Y[0], 0.012), STEEL),                  # deck rim, front
    ((sum(CART_X) / 2, CART_Y[0] + 0.006, CART_TOP + 0.006), (CART_X[1] - CART_X[0], 0.012, 0.012), STEEL),      # rim, right
    ((sum(CART_X) / 2, CART_Y[1] - 0.006, CART_TOP + 0.006), (CART_X[1] - CART_X[0], 0.012, 0.012), STEEL),      # rim, left
] + [
    ((x, y, FLOOR + 0.10 + _leg_h / 2), (0.030, 0.030, _leg_h), STEEL)                                           # cart legs
    for x in (CART_X[0] + 0.02, CART_X[1] - 0.02) for y in (CART_Y[0] + 0.02, CART_Y[1] - 0.02)] + [
    ((sum(CART_X) / 2, 0.0, FLOOR + 0.12), (CART_X[1] - CART_X[0], CART_Y[1] - CART_Y[0], 0.020), CART_GREY),   # lower tray
] + [((x, y, FLOOR + 0.045), (0.090, 0.035, 0.090), RUBBER)                                                   # wheels
     for x in (CART_X[0] + 0.03, CART_X[1] - 0.03) for y in (CART_Y[0] + 0.03, CART_Y[1] - 0.03)] + [
    ((x + 0.05, y, FLOOR + 0.075), (0.030, 0.020, 0.010), (0.75, 0.12, 0.10))                                    # brake tabs (red)
    for x in (CART_X[0] + 0.03, CART_X[1] - 0.03) for y in (CART_Y[0] + 0.03, CART_Y[1] - 0.03)] + [
    ((x, CART_Y[1] + 0.015, (CART_TOP + 1.00) / 2), (0.025, 0.025, 1.00 - CART_TOP), STEEL)                       # push handle posts
    for x in (-0.28, -0.16)] + [
    ((-0.22, CART_Y[1] + 0.015, 1.00), (0.150, 0.030, 0.030), RUBBER)]                                            # push handle grip


def _box_mesh(c, sz, rgb):
    c, h = np.asarray(c, float), np.asarray(sz, float) / 2
    pts = np.array([c + h * np.array([a, b, d]) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])
    return labkit.Mesh(points=pts, tris=tris, color=tuple(float(v) for v in rgb), collide="none")


SHELF = part("unit12_shelf", pos=(0.0, 0.0, 0.0), fixed=True, friction=0.6, boxes=SHELF_BOXES + EXTRA_COLLIDERS,
             meshes=[_box_mesh(c, sz, rgb) for c, sz, rgb in VISUAL_BOXES])

# -- the parcel (free). Part frame: origin at its bottom centre; long side along y.
BOX = np.array([0.1619, 0.0667, 0.1111])   # Uline S-20491 6 x 4 x 2 in stock carton, OUTSIDE 6 3/8 x 4 3/8 x 2 5/8 in, stood on its long narrow face
PKG_X0 = float(os.environ.get("PKG_X0", "0.120"))
PKG_Y = float(os.environ.get("PKG_Y", "-0.09"))      # same y on the deck and on the shelf
PKG_XS = 0.33                                         # where the demo sets it on the shelf
PKG_MU = float(os.environ.get("PKG_MU", "0.6"))
PARCEL_BOXES = [
    ((0.0, 0.0, BOX[2] / 2), tuple(BOX), KRAFT),
    ((0.0, 0.0, BOX[2] + 0.0003), (BOX[0] + 0.002, 0.048, 0.0006), TAPE),       # packing tape over the top seam
    ((0.0, 0.0, BOX[2] + 0.0009), (0.10, 0.06, 0.0006), WHITE),                # fictional shipping label
] + [((0.020, dy, BOX[2] + 0.0013), (0.030, 0.003, 0.0002), BLACK) for dy in (-0.018, -0.008, 0.002)] + [   # address lines (no text)
    ((-0.035 + 0.005 * i, 0.0, BOX[2] + 0.0013), (0.002, 0.045, 0.0002), BLACK) for i in range(8)]          # barcode bars
PARCEL = part("parcel", pos=(PKG_X0, PKG_Y, TOP), mass=0.8, friction=PKG_MU, boxes=PARCEL_BOXES)

# -- the phone (free), standing upside down in its dock; part frame at its centre; thickness x, width y, length z.
PH = np.array([0.0078, 0.0716, 0.1476])
CAM_LOCAL = np.array([+PH[0] / 2, 0.020, -0.060])       # rear camera, near the (lower, upside-down) top end, on the +x back
VOLUP_LOCAL = np.array([0.0, -PH[1] / 2, -0.029])         # volume-up button on the +y edge, 45 mm from the (lower, upside-down) top edge; the camera shutter
SHUTTER_LOCAL = VOLUP_LOCAL
DOCK = np.array([float(os.environ.get("PKG_DOCK_X", "0.12")), float(os.environ.get("PKG_DOCK_Y", "0.28")), TOP])
BTN_PROUD, BTN_TRAVEL, BTN_PRESSED = 0.0010, 0.0008, 0.0006      # volume button 1 mm proud, 0.8 mm travel, shutter at 0.6 mm
BTN_K = 3000.0                                                      # N/m: ~2.4 N at full travel (assumed; stated in fidelity)
VOLUP = labkit.Joint("volup", kind="slide", axis="Y", anchor=(0.0, -PH[1] / 2, VOLUP_LOCAL[2]), lower=0.0, upper=BTN_TRAVEL,
                     initial=0.0, target=0.0, stiffness=BTN_K, damping=2.0, mass=0.002,
                     boxes=[((0.0, -PH[1] / 2, VOLUP_LOCAL[2]), (0.004, 2 * BTN_PROUD, 0.012), WHITE)])
# the button sits in a 2 mm deep, 14 mm long recess in the -y edge, so its 0.8 mm travel never meets the phone body
_RZ0, _RZ1 = VOLUP_LOCAL[2] - 0.007, VOLUP_LOCAL[2] + 0.007
PHONE = part("phone", pos=(DOCK[0], DOCK[1], TOP + PH[2] / 2 + 0.004), mass=0.171, friction=0.9, joints=[VOLUP], boxes=[
    ((0.0, 0.001, 0.0), (PH[0], PH[1] - 0.002, PH[2]), TITANIUM),
    ((0.0, -PH[1] / 2 + 0.001, (-PH[2] / 2 + _RZ0) / 2), (PH[0], 0.002, _RZ0 + PH[2] / 2), TITANIUM),
    ((0.0, -PH[1] / 2 + 0.001, (PH[2] / 2 + _RZ1) / 2), (PH[0], 0.002, PH[2] / 2 - _RZ1), TITANIUM),
    ((-PH[0] / 2 - 0.0002, 0.0, 0.0), (0.0004, PH[1] - 0.006, PH[2] - 0.012), SCREEN),
    ((PH[0] / 2 + 0.0005, 0.012, -0.052), (0.001, 0.030, 0.030), ISLAND),                            # raised camera island
    ((PH[0] / 2 + 0.0014, CAM_LOCAL[1], CAM_LOCAL[2]), (0.0008, 0.010, 0.010), BLACK)])            # lens
DOCK_PART = part("phone_dock", pos=(0.0, 0.0, 0.0), fixed=True, friction=0.8, boxes=[
    ((DOCK[0], DOCK[1], TOP + 0.002), (0.040, 0.090, 0.004), BLACK),
    ((DOCK[0] - 0.012, DOCK[1], TOP + 0.0145), (0.004, 0.090, 0.025), BLACK),
    ((DOCK[0] + 0.012, DOCK[1], TOP + 0.0145), (0.004, 0.090, 0.025), BLACK)])

OBJECTS = [SHELF, PARCEL, PHONE, DOCK_PART]


# -- scoring --------------------------------------------------------------------------------------------------------------
# The proof photograph decides full credit. At each shutter click the phone camera's picture is rendered from the
# simulation (render_proof.cast: pinhole, portrait 480 x 640, 69.4 x 54.4 deg) with every solid that could be in it: cart,
# both shelves and their brackets, wall, plaques, the parcel, both robot arms and grippers. The picture is saved and the
# photo is judged from its pixels alone.
VFOV, HFOV = math.radians(69.4), math.radians(54.4)
IMG_W, IMG_H = 480, 640
MARGIN_X, MARGIN_Y = 24, 32          # 5 % of the width / height on each edge
PARCEL_VISIBLE_MIN = 0.90            # visible share of the parcel's own silhouette from that viewpoint
PLAQUE_VISIBLE_MIN, STROKE_VISIBLE_MIN = 0.98, 0.95
DIGIT_PX_MIN = 20                    # measured height of the '1' in the image, px
HAND_CLEAR, STILL_MM, STILL_N = 0.080, 0.001, 21
RELEASE_MM, SETTLE_N = 0.020, 11     # at the click: every pad >= 20 mm from the parcel, parcel still for 0.5 s
MOVE_MM, MOVE_DEG = 0.005, 2.0       # a photo is spoiled once the parcel leaves its photographed pose by more
DROP_Z = 0.05
CRADLE_TILT = 15.0
ID_BG, ID_PARCEL, ID_PLAQUE, ID_SCENE, ID_ROBOT = 0, 1, 2, 9, 10      # digit strokes are ids 3..8
_S: dict = {}


def _reset_state():
    _S.clear()
    _S.update(latch=None, photos=[], hist=[], clear=0.0, unsupported_since=None, max_free_drop=0.0, n=0,
              btn_prev=False, trim={"btn": np.zeros(3), "dock": np.zeros(3)})


def _corners(st):
    R, p = st.rotation(), np.asarray(st.pos, float)
    return np.array([p + R @ np.array([sx * BOX[0] / 2, sy * BOX[1] / 2, sz * BOX[2]]) for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])


def _ph_corners(st):
    R, p = st.rotation(), np.asarray(st.pos, float)
    return np.array([p + R @ (PH / 2 * np.array([a, b, c])) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])


def placed(st) -> bool:
    """All four bottom corners on the unit-12 shelf top (x 0.24..0.44, y -0.20..0.20, within 5 mm of 0.80 m), label up (tilt <= 10)."""
    c = _corners(st)[0::2]
    on = all(SH_X0 <= q[0] <= SH_X1 and abs(q[1]) <= SH_Y / 2 and abs(q[2] - SHELF_TOP) <= 0.005 for q in c)
    return bool(on and st.tilt_deg() <= 10.0)


def docked(st) -> bool:
    """Phone secure in its cradle: its lower end inside the 20 mm slot (between the walls, over the 90 mm base, below the
    25 mm wall tops) and within 15 deg of upright, either face forward and either end down."""
    R, p = st.rotation(), np.asarray(st.pos, float)
    ends = [p + R @ np.array([0.0, 0.0, s * PH[2] / 2]) for s in (-1, 1)]
    low = min(ends, key=lambda e: e[2])
    up_tilt = math.degrees(math.acos(min(1.0, abs(float(R[2, 2])))))
    return bool(abs(low[0] - DOCK[0]) <= 0.010 and abs(low[1] - DOCK[1]) <= 0.045 and TOP + 0.002 <= low[2] <= TOP + 0.0265
                and up_tilt <= CRADLE_TILT)


def camera_view(ph):
    """camera centre and (forward, up, right) unit vectors in the world from the phone pose (portrait, gravity-rotated)."""
    R, p = ph.rotation(), np.asarray(ph.pos, float)
    cam = p + R @ CAM_LOCAL
    f = R @ np.array([1.0, 0.0, 0.0]); up = R @ np.array([0.0, 0.0, 1.0])
    up = up if up[2] >= 0 else -up          # portrait, auto-rotated: image up is the phone's long axis nearer world up
    r = np.cross(f, up)
    return cam, f, up, r


TABLE_BOX = (((CART_X[0] + CART_X[1]) / 2, 0.0, CART_TOP - 0.020), (CART_X[1] - CART_X[0], CART_Y[1] - CART_Y[0], 0.040))   # cart deck


def _aabb(c, sz, rgb, oid):
    return (np.asarray(c, float), np.eye(3), np.asarray(sz, float) / 2, rgb, oid)


def _scene_boxes():
    """the fixed solids in the picture: plaque 12 (id 2) and its strokes (ids 3..8), everything else id 9."""
    out = [_aabb(*TABLE_BOX, CART_GREY, ID_SCENE)]
    stroke = 3
    for c, sz, rgb in SHELF_BOXES:
        if tuple(rgb) == BLACK:
            out.append(_aabb(c, sz, rgb, stroke)); stroke += 1
        elif tuple(rgb) == WHITE:
            out.append(_aabb(c, sz, rgb, ID_PLAQUE))
        else:
            out.append(_aabb(c, sz, rgb, ID_SCENE))
    for c, sz, rgb in EXTRA_COLLIDERS + VISUAL_BOXES:
        oid = ID_SCENE
        if tuple(rgb) == WHITE and abs(c[0] - (WALL_X - 0.0015)) < 1e-6:      # neighbour plaques: 11 (+y), 13 (-y)
            oid = 11 if c[1] > 0 else 13
        out.append(_aabb(c, sz, rgb, oid))
    return out


SCENE_BOXES = _scene_boxes()


def _parcel_boxes(pkg):
    R, p = pkg.rotation(), np.asarray(pkg.pos, float)
    return [(p + R @ np.asarray(c, float), R, np.asarray(sz, float) / 2, rgb, ID_PARCEL) for c, sz, rgb in PARCEL_BOXES]


def _segment_box(a, b, thick, rgb, oid):
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = float(np.linalg.norm(d))
    z = d / L; x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0.0, 0.0]); x /= np.linalg.norm(x)
    return ((a + b) / 2, np.column_stack([x, np.cross(z, x), z]), np.array([thick / 2, thick / 2, L / 2 + thick / 2]), rgb, oid)


ARM_CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")


def _robot_boxes(links):
    """each arm as 70 mm boxes along its link chain, plus the gripper's fingers and palm (110 x 45 x 110 mm)."""
    out = []
    for lk in links:
        pts = [lk[n][:3, 3] for n in ARM_CHAIN if n in lk]
        out += [_segment_box(a, b, 0.07, (0.92, 0.92, 0.92), ID_ROBOT) for a, b in zip(pts, pts[1:]) if np.linalg.norm(b - a) > 0.02]
        if "gripper" in lk:
            g = lk["gripper"]
            out.append((g[:3, 3] + g[:3, :3] @ np.array([0.0, 0.0, 0.045]), g[:3, :3], np.array([0.055, 0.0225, 0.055]), (0.1, 0.1, 0.1), ID_ROBOT))
    return out


def judge_photo(pkg, ph, links=(), save=None):
    """render the proof photo and judge it from its pixels. Valid: (a) the parcel's visible pixels keep the 5 % margin and
    show >= 90 % of its own silhouette; (b) the plaque keeps the margin, >= 98 % of its pixels visible and >= 95 % of every
    digit stroke; (c) of the unit plaques in the picture, 12's is the nearest to the parcel (image columns); (d) the '1'
    measures >= 20 px tall."""
    cam, f, up, r = camera_view(ph)
    pb, rb = _parcel_boxes(pkg), _robot_boxes(links)
    img, ids = _cast(cam, f, up, r, SCENE_BOXES + pb + rb)
    _S["png_written"] = False
    if save:
        import render_proof
        try:
            render_proof.write_png(save, img)
            _S["png_written"] = True           # this call wrote the picture (an older file at that path does not count)
        except OSError:
            _S["png_written"] = False
    _, own = _cast(cam, f, up, r, pb)
    plq = [b for b in SCENE_BOXES if b[4] == ID_PLAQUE or 3 <= b[4] <= 8]      # plaque 12 and its strokes only
    _, plq_ids = _cast(cam, f, up, r, plq)
    band = np.zeros(ids.shape, bool); band[:MARGIN_Y] = band[-MARGIN_Y:] = True; band[:, :MARGIN_X] = band[:, -MARGIN_X:] = True
    n_own = int((own == ID_PARCEL).sum()); n_vis = int((ids == ID_PARCEL).sum())
    if n_own == 0 or (band & (own == ID_PARCEL)).any():
        return False, "parcel_cut_off"
    if n_vis < PARCEL_VISIBLE_MIN * n_own:
        return False, f"parcel_hidden:{n_vis / n_own:.2f}"
    full = plq_ids > 0; n_full = int(full.sum())
    if n_full == 0 or (band & full).any():
        return False, "plaque_cut_off"
    if int((full & (ids == plq_ids)).sum()) < PLAQUE_VISIBLE_MIN * n_full:
        return False, "plaque_hidden"
    for sid in range(3, 9):
        m = plq_ids == sid
        if m.sum() == 0 or int((m & (ids == sid)).sum()) < STROKE_VISIBLE_MIN * int(m.sum()):
            return False, f"digit_stroke_hidden:{sid}"
    # the parcel must belong to unit 12 in the picture: of the plaques visible, 12's is the nearest to the parcel (image columns)
    cols = np.where(ids == ID_PARCEL)[1]; pc = float(cols.mean())
    d12 = abs(float(np.where(full)[1].mean()) - pc)
    for nid in (11, 13):
        m = ids == nid
        if m.sum() > 0 and abs(float(np.where(m)[1].mean()) - pc) < d12:
            return False, f"parcel_nearer_unit_{nid}"
    rows = np.where((ids == 3).any(axis=1))[0]
    px = int(rows[-1] - rows[0] + 1) if len(rows) else 0
    if px < DIGIT_PX_MIN:
        return False, f"illegible:{px}px"
    return True, f"ok:{px}px"


def _cast(cam, f, up, r, boxes):
    from render_proof import cast
    return cast(cam, f, up, r, boxes, VFOV, HFOV, IMG_W, IMG_H)


def _tips(env):
    from yam_arena import runtime as _rt
    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = {"links": []}
    for n, k in (("left_robot", "L"), ("right_robot", "R")):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        lk = {name: m.copy() for name, m in _rt.link_poses(robot).items()}
        for m in lk.values():
            m[:3, 3] -= o
        g = lk["gripper"]
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        q7, q8 = (abs(float(v)) for v in robot.data.joint_pos.torch[0, ids].cpu().numpy())
        out[k] = [mid - q7 * g[:3, 0], mid + q8 * g[:3, 0], mid]
        out[k + "_g"] = g
        out["links"].append(lk)
    return out


def _box_dist_local(p_local, c, s):
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(p_local) - np.asarray(c)) - np.asarray(s) / 2, 0.0)))


def _dist_to(st, pts, centre, size):
    R, p = st.rotation(), np.asarray(st.pos, float)
    return min(_box_dist_local(R.T @ (np.asarray(q) - p), centre, size) for q in pts)


def _gripped_by(ph, tips):
    """the hand whose two pads both touch the phone (within 3 mm) with the jaw closed on it (pad gap = phone width or
    thickness, +-5 mm); None if no hand grips it."""
    for k in ("L", "R"):
        a, b = tips[k][0], tips[k][1]
        gap = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        touching = all(_dist_to(ph, [t], (0, 0, 0), PH) < 0.003 for t in (a, b))
        if touching and min(abs(gap - PH[1]), abs(gap - PH[0])) <= 0.005:
            return k
    return None


def _pose_dev(pkg, pos, R0):
    dp = float(np.linalg.norm(np.asarray(pkg.pos, float) - pos))
    da = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ pkg.rotation()) - 1) / 2))))
    return dp, da


@on_reset
def _reset(env):
    _reset_state()


@on_step
def _watch(env):
    _S["n"] += 1
    pkg = labkit._part_state(env, PARCEL); ph = labkit._part_state(env, PHONE)
    tips = _tips(env); allp = tips["L"][:2] + tips["R"][:2]
    d_pkg = _dist_to(pkg, allp, (0, 0, BOX[2] / 2), BOX)
    d_ph = _dist_to(ph, allp, (0, 0, 0), PH)
    # latches: only a failed delivery or a drop
    if _S["latch"] is None:
        if float(ph.pos[2]) < TOP - 0.005 or float(np.min(_corners(pkg)[:, 2])) < TOP - 0.005:
            _S["latch"] = "fell_toward_the_floor"
        else:
            z = float(pkg.pos[2]); held = d_pkg < 0.010
            if not held:
                if _S["unsupported_since"] is None:
                    _S["unsupported_since"] = z
                _S["max_free_drop"] = max(_S["max_free_drop"], _S["unsupported_since"] - z)
                if _S["max_free_drop"] > DROP_Z:
                    _S["latch"] = f"parcel_dropped_{_S['max_free_drop'] * 1000:.0f}mm"
            else:
                _S["unsupported_since"] = None; _S["max_free_drop"] = 0.0
            phone_free = _gripped_by(ph, tips) is None and min(_dist_to(ph, tips[k][:2], (0, 0, 0), PH) for k in ("L", "R")) > 0.010
            if phone_free and not docked(ph) and ph.tilt_deg() > 60.0:
                _S["latch"] = "phone_dropped"
    # every photo taken so far is spoiled once the parcel leaves its photographed pose (tracked every step)
    for pho in _S["photos"]:
        if pho["ok"] and not pho["spoiled"]:
            dp, da = _pose_dev(pkg, pho["pkg_pos"], pho["pkg_R"])
            if dp > MOVE_MM or da > MOVE_DEG:
                pho["spoiled"] = True
    # shutter: a fresh press of the volume-up button (released on the previous step, clicked now: >= 0.6 mm of its 0.8 mm
    # travel) while a hand already grips the phone at least 30 mm above its dock
    pressed = float(ph.joints.get("volup", 0.0)) >= BTN_PRESSED
    holder = _gripped_by(ph, tips)
    high = float(ph.pos[2]) > TOP + PH[2] / 2 + 0.030
    settled = len(_S["hist"]) >= SETTLE_N and max(float(np.max(np.linalg.norm(a[:8] - b[:8], axis=1)))
                                                   for a in _S["hist"][-SETTLE_N:] for b in _S["hist"][-SETTLE_N:]) < 0.001
    shot = shutter(pressed, holder, high, pkg, ph, tips["links"], d_pkg, settled)
    if shot is not None:
        _proof_dump(pkg, ph, tips, shot["ok"], shot["why"])
        if DEBUG:
            print(f"[pkg] shutter at step {_S['n']}: {shot['why']} (held by {holder}) png={shot['png']}", file=sys.stderr, flush=True)
    _trims(ph, tips)
    _S["clear"] = min(d_pkg, d_ph)
    _S["hist"] = (_S["hist"] + [np.vstack([_corners(pkg), _ph_corners(ph)])])[-STILL_N:]
    held = [holder] if holder else []
    dense = DEBUG and _WIN and _WIN["btn"][1] - 0.5 <= _S["n"] / 20.0 <= _WIN["btn"][1] + 2.0
    if dense and _S["n"] % 2 == 0:
        print(f"[btn] t={_S['n'] / 20:5.2f} btn={float(ph.joints.get('volup', 0.0)) * 1000:.3f}mm held={held} "
              f"ph={np.round(np.asarray(ph.pos) * 1000, 1).tolist()}", file=sys.stderr, flush=True)
    if DEBUG and _S["n"] % 20 == 0:
        last = _S["photos"][-1]["why"] if _S["photos"] else None
        print(f"[pkg] t={_S['n'] / 20:5.1f}s placed={placed(pkg)} tilt={pkg.tilt_deg():.1f} docked={docked(ph)} "
              f"clear={_S['clear'] * 1000:.0f}mm latch={_S['latch']} photo={last} "
              f"pkg={np.round(np.asarray(pkg.pos) * 1000).astype(int).tolist()} ph={np.round(np.asarray(ph.pos) * 1000).astype(int).tolist()} "
              f"ph_tilt={ph.tilt_deg():.0f} btn={float(ph.joints.get('volup', 0.0)) * 1000:.2f}mm held={held} dpkg={d_pkg * 1000:.0f} "
              f"L={np.round(tips['L'][2] * 1000).astype(int).tolist()} R={np.round(tips['R'][2] * 1000).astype(int).tolist()} "
              f"trim_btn={np.round(_S['trim']['btn'] * 1000, 1).tolist()} trim_dock={np.round(_S['trim']['dock'] * 1000, 1).tolist()}",
              file=sys.stderr, flush=True)


def shutter(pressed, holder, high, pkg, ph, links, d_pkg, settled):
    """one step of the phone's shutter. A photo is taken only on a rising edge of the button (released on the previous
    step, pressed on this one) while the phone is already gripped and >= 30 mm above its dock; a press that began low or
    before the grip never fires however long it is held. The photo counts only if its image file was written."""
    rising = pressed and not _S.get("btn_prev", False)
    _S["btn_prev"] = pressed
    if not (rising and holder and high):
        return None
    out_dir = os.environ.get("PKG_PROOF_DIR", str(pathlib.Path.cwd() / "proof_photos"))
    save = str(pathlib.Path(out_dir) / f"proof_step{_S['n']}.png")
    try:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    ok, why = judge_photo(pkg, ph, links, save)
    written = bool(_S.get("png_written")) and pathlib.Path(save).is_file() and pathlib.Path(save).stat().st_size > 0
    if not written:
        ok, why, save = False, "photo_not_saved", None
    elif not placed(pkg):
        ok, why = False, "parcel_not_placed_at_shutter"
    elif d_pkg < RELEASE_MM:
        ok, why = False, "parcel_still_held_at_shutter"
    elif not settled:
        ok, why = False, "parcel_moving_at_shutter"
    shot = {"step": _S["n"], "ok": ok, "why": why, "spoiled": False, "png": save,
            "pkg_pos": np.asarray(pkg.pos, float).copy(), "pkg_R": pkg.rotation().copy()}
    _S["photos"].append(shot)
    return shot


def _proof_dump(pkg, ph, tips, ok, why):
    """the photo's inputs (camera and every box in the picture) as JSON next to the PNG, for review."""
    out_dir = os.environ.get("PKG_PROOF_DIR")
    if not out_dir:
        return
    import json
    cam, f, up, r = camera_view(ph)
    boxes = []
    for c, ax, hs, rgb, oid in SCENE_BOXES + _parcel_boxes(pkg) + _robot_boxes(tips["links"]):
        pts = [(np.asarray(c) + np.asarray(ax) @ (np.asarray(hs) * np.array([a, b, d]))).tolist() for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]
        boxes.append({"c": pts, "rgb": list(rgb), "id": int(oid)})
    rec = {"step": _S["n"], "ok": bool(ok), "why": why, "cam": cam.tolist(), "f": f.tolist(), "up": up.tolist(), "r": r.tolist(),
           "vfov_deg": math.degrees(VFOV), "hfov_deg": math.degrees(HFOV), "img": [IMG_W, IMG_H], "boxes": boxes}
    try:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(out_dir) / f"proof_step{_S['n']}.json").write_text(json.dumps(rec))
    except OSError:
        pass                                  # an optional diagnostic; never aborts a trial


def _still() -> bool:
    h = _S.get("hist", [])
    return len(h) >= STILL_N and max(float(np.max(np.linalg.norm(a - b, axis=1))) for i, a in enumerate(h) for b in h[i + 1:]) < STILL_MM


def retained_photo(pkg):
    """a valid proof photo that was never spoiled and whose parcel pose is the final one (any earlier valid photo counts;
    a bad retake never erases it)."""
    for pho in _S.get("photos", []):
        if pho["ok"] and not pho["spoiled"]:
            dp, da = _pose_dev(pkg, pho["pkg_pos"], pho["pkg_R"])
            if dp <= MOVE_MM and da <= MOVE_DEG:
                return pho
    return None


def score(objects) -> float:
    """0 for a failed delivery or a drop: the parcel not placed on the unit-12 shelf (all 4 bottom corners on its top, within
    10 deg of upright), the parcel falling more than 5 cm unsupported, the parcel or phone going over the cart or shelf edge
    toward the floor, or the phone dropped (free, out of its cradle, tilted over 60 deg); also 0 while a gripper pad is within
    80 mm of the parcel or phone, or anything (parcel and phone corners) moved 1 mm or more over the last 1 s.
    1.0: placed, a retained valid proof photo (judge_photo, taken after the parcel was released and still, never spoiled by
    a later move of the parcel) and the phone secure in its cradle. 0.5: placed safely without that."""
    pkg, ph = objects["parcel"], objects["phone"]
    if _S.get("latch") or not placed(pkg) or _S.get("clear", 0.0) < HAND_CLEAR or not _still():
        return 0.0
    return 1.0 if (retained_photo(pkg) is not None and docked(ph)) else 0.5


def success(objects) -> bool:
    return score(objects) == 1.0


# -- scripted demo (open loop from the fixed layout; reads no hidden state) ------------------------------------------------
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402


class Orient(float):
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


def frame_rot(approach, closing):
    a = np.asarray(approach, float); a = a / np.linalg.norm(a)
    c = np.asarray(closing, float); c = c - a * (c @ a); c = c / np.linalg.norm(c)
    return Orient(np.column_stack([c, np.cross(a, c), a]))


PAD_PAST_TCP, TIP_PAST_TCP = 0.0035, 0.0128
GRIP_BELOW_TOP = 0.030                                   # R pinches the parcel's 75 mm width 30 mm below its top
CARRY_LIFT = float(os.environ.get("PKG_LIFT", "0.035"))  # parcel bottom this far above the shelf top on the carry
PRESS_IN = float(os.environ.get("PKG_PRESS_IN", "0.020"))
PRESS_JAW = float(os.environ.get("PKG_PRESS_JAW", "0.0"))   # aim one closed fingertip's centre, not the seam, at the button   # fingertip target past the button face
PHOTO_CAM = np.array([float(v) for v in os.environ.get("PKG_PHOTO_CAM", "0.0,0.165,0.84").split(",")])
LEFT_PARK, RIGHT_PARK = (0.15, 0.42, TOP + 0.16), (0.15, -0.42, TOP + 0.16)
M_PH = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # L TCP axes in the phone frame (x=-width, y=-normal, z=down);
# the 180 deg wrist turn of the symmetric gripper keeps L off its joint-5 limit (IK sweep: 56 mm miss -> 0.03 mm)
DOCK_BOTTOM = np.array([DOCK[0], DOCK[1], TOP + 0.004 + 0.003])     # phone's lower end 3 mm over the dock floor at release
_WIN: dict = {}       # demo windows (s) in which the watcher updates the live trims


class _Dyn:
    """a TCP target plus a live trim the watcher derives from camera-visible poses (the phone and the L pad tips)."""
    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        tr = _S.get("trim", {}).get(self.key, np.zeros(3))
        return np.asarray(self.tcp + tr, dtype=dtype or float)


def _trims(ph, tips):
    """btn: how far the held phone's volume button sits from where the plan put it (the phone slips in the pinch);
    frozen once the press starts. dock: where L's pads must go so the phone's lower end lands over the dock floor, from the
    phone's offset in the pinch, measured while L hovers over the dock; frozen once it descends. Low-pass 0.2 per step."""
    t = _S["n"] / 20.0
    if not _WIN:
        return
    R, p = ph.rotation(), np.asarray(ph.pos, float)
    if _WIN["btn"][0] <= t < _WIN["btn"][1]:
        s_obs = p + R @ _WIN["btn_local"]
        _S["trim"]["btn"] = _S["trim"]["btn"] + 0.2 * (np.clip(s_obs - _WIN["btn_nom"], -0.03, 0.03) - _S["trim"]["btn"])
    if _WIN["dock"][0] <= t < _WIN["dock"][1]:
        bottom = p + R @ np.array([0.0, 0.0, -PH[2] / 2])
        pad = np.asarray(tips["L"][2], float)
        want = DOCK_BOTTOM - (bottom - pad) - _WIN["dock_pad_nom"]
        _S["trim"]["dock"] = _S["trim"]["dock"] + 0.2 * (np.clip(want, -0.03, 0.03) - _S["trim"]["dock"])
    elif _WIN["dock_fb"][0] <= t < _WIN["dock_fb"][1]:      # while descending: steer the phone's lower end over the slot (x, y)
        bottom = p + R @ np.array([0.0, 0.0, -PH[2] / 2])
        e = np.clip(0.1 * (DOCK_BOTTOM[:2] - bottom[:2]), -0.0003, 0.0003)
        _S["trim"]["dock"][:2] = np.clip(_S["trim"]["dock"][:2] + e, -0.03, 0.03)


def _w(t, pos, finger, label, rot):
    if not isinstance(pos, _Dyn):
        pos = tuple(float(v) for v in np.asarray(pos, float))
    return Waypoint(t, pos, finger, label, rot)


def _dur(wps):
    return sum(w.duration for w in wps)


def photo_phone_pose():
    """phone pose that puts the camera at PHOTO_CAM aimed at the parcel-on-shelf + label centroid, image upright."""
    pkg_c = np.array([PKG_XS, PKG_Y, SHELF_TOP + BOX[2] / 2])
    tgt = (pkg_c * 8 + LABEL_C * 4) / 12
    f = tgt - PHOTO_CAM; f /= np.linalg.norm(f)
    zl = np.array([0.0, 0.0, 1.0]) - f * f[2]; zl /= np.linalg.norm(zl)          # phone +z (local) points up
    Rph = np.column_stack([f, np.cross(zl, f), zl])
    pos = PHOTO_CAM - Rph @ CAM_LOCAL
    return pos, Rph


def phone_tcp(pos, Rph, z_local=0.0):
    R = Rph @ M_PH
    return pos + Rph @ np.array([0.0, 0.0, z_local]) - R[:, 2] * PAD_PAST_TCP, Orient(R)


def demo():
    x0, xs, y = PKG_X0, PKG_XS, PKG_Y
    gz = lambda bottom: bottom + BOX[2] - GRIP_BELOW_TOP + PAD_PAST_TCP * 0   # TCP height for a pinch 30 mm below the top
    # R: one-hand pinch across the parcel's 75 mm width at its centre, carry it to the shelf, set it down, let go
    R = [_w(2.0, (x0, y, TOP + BOX[2] + 0.05), FINGER_OPEN, "open hand over the parcel", 0.0),
         _w(1.2, (x0, y, gz(TOP)), FINGER_OPEN, "fingers either side of the parcel", 0.0),
         _w(0.8, (x0, y, gz(TOP)), FINGER_CLOSED, "pinch it", 0.0),
         _w(1.5, (x0, y, gz(SHELF_TOP + CARRY_LIFT)), FINGER_CLOSED, "lift it clear of the shelf edge", 0.0),
         _w(2.0, (xs, y, gz(SHELF_TOP + CARRY_LIFT)), FINGER_CLOSED, "carry it over the unit-12 shelf", 0.0),
         _w(1.5, (xs, y, gz(SHELF_TOP + 0.001)), FINGER_CLOSED, "set it down, label up", 0.0),
         _w(0.8, (xs, y, gz(SHELF_TOP + 0.001)), FINGER_OPEN, "let go", 0.0),
         _w(1.0, (xs, y, gz(SHELF_TOP + 0.001) + 0.03), FINGER_OPEN, "lift the hand a little", 0.0),
         _w(1.2, (0.19, y, gz(SHELF_TOP + 0.001) + 0.03), FINGER_OPEN, "slide the open hand back off the parcel", 0.0)]
    if os.environ.get("PKG_PHONE_ONLY"):
        R = [_w(2.0, RIGHT_PARK, FINGER_CLOSED, "park", 0.0)]
    t_placed = _dur(R)
    # L: phone from its dock (pinched at its middle across the long edges) to the photo pose, while R carries
    ph0 = np.array([DOCK[0], DOCK[1], TOP + PH[2] / 2 + 0.004])
    tcp_d, rot_d = phone_tcp(ph0, np.eye(3))
    L = [_w(2.0, tcp_d + np.array([0, 0, 0.10]), FINGER_OPEN, "over the phone in its dock", rot_d),
         _w(1.5, tcp_d, FINGER_OPEN, "fingers either side of the phone", rot_d),
         _w(0.8, tcp_d, FINGER_CLOSED, "pinch it", rot_d),
         _w(1.2, tcp_d + np.array([0, 0, 0.09]), FINGER_CLOSED, "lift it out", rot_d)]
    pos_p, Rph = photo_phone_pose()
    tcp_p, rot_p = phone_tcp(pos_p, Rph)
    L.append(_w(2.5, tcp_p, FINGER_CLOSED, "frame the parcel and the unit label", rot_p))
    t_framed = _dur(L)
    # R: from the shelf to the volume button on the phone's -y edge (facing R), press it, back off
    btn_n = Rph[:, 1]                                                  # into the phone at its -y edge
    btn_local = np.array([0.0, -PH[1] / 2 - BTN_PROUD, VOLUP_LOCAL[2] + PRESS_JAW])   # aim point on the button face
    surf = pos_p + Rph @ btn_local
    rot_b = frame_rot(btn_n, Rph[:, 2])
    t_appr = max(t_placed, t_framed) + 1.0
    R.append(_w(t_appr + 2.5 - _dur(R), _Dyn(surf - btn_n * (0.030 + TIP_PAST_TCP), "btn"), FINGER_CLOSED,
                "fingertips facing the volume button", rot_b))
    t_press = _dur(R)
    R += [_w(1.2, _Dyn(surf + btn_n * (PRESS_IN - TIP_PAST_TCP), "btn"), FINGER_CLOSED, "press the volume button: take the photo", rot_b),
          _w(0.6, _Dyn(surf + btn_n * (PRESS_IN - TIP_PAST_TCP), "btn"), FINGER_CLOSED, "hold the press", rot_b),
          _w(0.6, _Dyn(surf - btn_n * (0.030 + TIP_PAST_TCP), "btn"), FINGER_CLOSED, "let it go", rot_b),
          _w(2.0, RIGHT_PARK, FINGER_CLOSED, "park", 0.0)]
    t_done = t_press + 2.4
    L.append(_w(t_done + 0.5 - _dur(L), tcp_p, FINGER_CLOSED, "hold still for the photo", rot_p))
    # L: phone back in its dock (the target corrected for where the phone sits in the pinch), let go, park
    L.append(_w(2.5, tcp_d + np.array([0, 0, 0.09]), FINGER_CLOSED, "over the dock", rot_d))
    t_hover = _dur(L)
    L += [_w(1.0, tcp_d + np.array([0, 0, 0.09]), FINGER_CLOSED, "steady over the dock", rot_d),
          _w(3.0, _Dyn(tcp_d, "dock"), FINGER_CLOSED, "phone back in its dock", rot_d),
          _w(0.8, _Dyn(tcp_d, "dock"), FINGER_OPEN, "let go", rot_d),
          _w(1.2, tcp_d + np.array([0, 0, 0.10]), FINGER_OPEN, "up and away", rot_d),
          _w(2.0, LEFT_PARK, FINGER_CLOSED, "park", 0.0)]
    t_end = max(_dur(L), _dur(R)) + 2.0
    L.append(_w(t_end - _dur(L), LEFT_PARK, FINGER_CLOSED, "rest", 0.0))
    R.append(_w(t_end - _dur(R), RIGHT_PARK, FINGER_CLOSED, "rest", 0.0))
    _WIN.clear()
    _WIN.update(btn=(t_framed + 0.5, t_press), btn_local=btn_local, btn_nom=surf,
                dock=(t_hover + 0.2, t_hover + 1.0), dock_fb=(t_hover + 1.0, t_hover + 4.0), dock_pad_nom=tcp_d + rot_d.rot[:, 2] * PAD_PAST_TCP)
    return {"left": L, "right": R}
