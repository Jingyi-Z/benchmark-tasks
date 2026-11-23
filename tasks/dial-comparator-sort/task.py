"""Dial comparator go/no-go: zero a dial indicator on a 25.000 mm master block by turning its bezel, put the master back
on its MASTER area, then measure three ground steel spacers under the dial and sort each into the ACCEPT bin
(25.00 +- 0.02 mm) or the REJECT bin.

Bench (station table, top 0.7475 m). Cast iron dial comparator stand (KBC 1-807-450 type: 3-7/8 x 6 in base, raised
serrated anvil 60 mm across, 8 in tall; its fine-adjust knob, column clamp and stem clamp are modelled where the maker
photo shows them and are not used) holding a 0.01 mm dial indicator (Mitutoyo 2046A type, lug back: 57 mm dial, knurled
bezel about 60 mm across that turns the face, its 0 mark and two red tolerance markers; 10 mm travel; flat contact) with
a spindle lifting lever (21EZA198 type) at the spindle top. The needle is driven only by the spindle's real travel
(1 mm per turn). Parts on a blue tray: the master (yellow band, "MASTER 25.000") and three spacers (40 x 10 mm
footprint) whose heights depend on the seed (DC_SEED 0..7). Two bins.

The scorer uses the simulated state. The scripted demo reads the dial only from the right arm camera image
(dial_read.py) for its bezel turns, its zero exit and each bin choice.

Run:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
        -E cam_width=640 -E cam_height=352 --instruction "<see task.json>" --scorer task_score --max-steps 3200
"""
import json
import math
import os
import pathlib
import sys

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401,E402
import labkit  # noqa: E402
from labkit import Joint, Mesh, on_reset, on_step, part  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
DEBUG = bool(os.environ.get("DC_DEBUG"))
LOGALL = bool(os.environ.get("DC_LOGALL"))
NEEDLE_TEST = float(os.environ["DC_NEEDLE_TEST"]) if os.environ.get("DC_NEEDLE_TEST") else None
labkit.CONTACT_OFFSET = 0.0002

GRANITE, STEEL, CHROME, BLACK, WHITE = (0.30, 0.33, 0.36), (0.62, 0.63, 0.66), (0.80, 0.81, 0.83), (0.06, 0.06, 0.07), (0.96, 0.96, 0.94)
RED, GREEN, TRAYC = (0.75, 0.12, 0.10), (0.12, 0.55, 0.20), (0.30, 0.31, 0.33)
labkit.SURFACES.update({tuple(round(c, 4) for c in STEEL): {"roughness": 0.35, "metallic": 0.85},
                        tuple(round(c, 4) for c in CHROME): {"roughness": 0.15, "metallic": 0.95},
                        tuple(round(c, 4) for c in GRANITE): {"roughness": 0.5, "metallic": 0.0}})

try:                                       # visual-only meshes (collide "none"); colliders stay the boxes below
    import dcs_visuals as _V  # noqa: E402
except ImportError:  # noqa: BLE001
    _V = None


def _vis(key):
    return list(_V.get(key, globals())) if _V is not None and hasattr(_V, "get") else []


# ---- stand ----------------------------------------------------------------------------------------------------------
XS, YS = 0.42, float(os.environ.get("DC_YS", "-0.05"))   # granite centre
GB = 0.1524                               # 6 in
GBX, GBY = 0.0984, 0.1524                  # KBC 1-807-450 cast iron base, 3-7/8 x 6 in (long side front to back, along y)
SPECKLE = False                            # enamelled cast iron, not granite
GH = 0.0508                               # measuring height above the bench (ESTIMATE; the listing gives 8 in overall height)
ANV_R, ANV_H = 0.030, 0.010               # raised serrated anvil, 60 mm across (ABM 2-3/8 in class), 10 mm proud (ESTIMATE)
GHB = GH - ANV_H                          # cast base slab under it
ANV = TOP + GH                            # measuring surface: the anvil top
XC, YCOL = XS, float(os.environ.get("DC_YS", "-0.05")) + 0.065   # column behind the dial (+y), on the granite
COL_H = 0.205                              # column top just above the arm clamp, under the lifting lever
ZERO_GAP = float(os.environ.get("DC_ZGAP", "0.00006"))  # start: the seated master reads -0.06 mm (6 divisions off zero)
FREE = 0.0010                              # spindle pre-travel on a 25 mm block at true zero (1 mm into the 10 mm)
TIP0 = ANV + 0.025 - FREE + ZERO_GAP       # free spindle tip height (arm joint 0)
DIAL_Z = TIP0 + float(os.environ.get("DC_DIALH", "0.085"))   # dial centre height above the free tip (contact point extension)
DIAL_R, DIAL_T = 0.0285, 0.022
BZ_R = DIAL_R + 0.0015                     # knurled bezel radius (about 60 mm across)
BZ_GAIN = float(os.environ.get("DC_BZ_GAIN", "0.0"))
BZ_CYCLES = int(os.environ.get("DC_BZ_CYCLES", "3"))   # turn-look cycles available; unused ones become looks
ZERO_DONE_MM, ZERO_DONE_N = 0.005, 10          # demo exit: master reads within 0.6 division for 0.5 s with the hand clear (scorer: 1 division)
BZ_FF = float(os.environ.get("DC_BZ_FF", "1.0"))       # first pinch: this share of the owed turn (the bezel coasts a little)
BZ_FF2 = float(os.environ.get("DC_BZ_FF2", "1.0"))     # second pinch, after the look: this share of what is left   # slip-correction gain per settled control step while pinched
K_SPRING = 150.0                           # spindle return spring, about 1.5 N at 1 mm (ESTIMATE)


def _ring_mesh(r_in, r_out, t, n=48):
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        for r in (r_in, r_out):
            for x in (-t / 2, t / 2):
                pts.append((x, r * math.cos(a), r * math.sin(a)))
    return np.array(pts)


needle_pts = np.array([(x, y, z) for x in (-0.0008, 0.0008) for y in (-0.0006, 0.0006) for z in (-0.004, DIAL_R * 0.85)])
arm = Joint("arm", kind="slide", axis="Z", anchor=(XS, YS, DIAL_Z), lower=-0.004, upper=0.004, initial=0.0, target=0.0, meshes=_vis("arm"),
            stiffness=2.0e6, damping=2000.0, mass=0.30, boxes=[
                ((XS, (YS + YCOL) / 2 + 0.004, DIAL_Z + 0.030), (0.016, YCOL - YS - 0.008, 0.014), CHROME),   # arm to column
                ((XC, YCOL, DIAL_Z + 0.030), (0.036, 0.036, 0.020), CHROME),                      # clamp on the column
                ((XS, YS + 0.012, DIAL_Z), (0.010, 0.010, 0.030), CHROME)],                         # dial back stem
            cylinders=[((XS, YS, DIAL_Z), DIAL_R, DIAL_T, CHROME, "Y"),                            # dial case (faces -y)
                       ((XS, YS, DIAL_Z - DIAL_R - 0.006), 0.004, 0.012, CHROME, "Z")]
            )                                              # stem below; the face and its marks turn with the bezel
spindle = Joint("spindle", kind="slide", axis="Z", anchor=(XS, YS, TIP0), lower=0.0, upper=0.010, initial=0.0, meshes=_vis("spindle"),
                target=0.0, stiffness=K_SPRING, damping=3.0, mass=0.02, parent="arm",
                cylinders=[((XS, YS, TIP0 + 0.010), 0.003, 0.020, CHROME, "Z"),                     # lower spindle + flat tip
                           ((XS, YS, TIP0 + 0.0015), 0.003, 0.003, STEEL, "Z"),
                           ((XS, YS, DIAL_Z + DIAL_R + 0.012), 0.002, 0.018, CHROME, "Z"),          # upper spindle
                           ((XS, YS, DIAL_Z + DIAL_R + 0.024), 0.006, 0.008, BLACK, "Z")])          # lift cap
needle = Joint("needle", kind="hinge", axis="Y", anchor=(XS, YS - DIAL_T / 2 - 0.0012, DIAL_Z), lower=-200.0, upper=200.0,
               initial=0.0, target=0.0, stiffness=5.0, damping=0.05, mass=0.002, parent="arm",
               meshes=[Mesh(needle_pts + np.array([XS, YS - DIAL_T / 2 - 0.0012, DIAL_Z]),
                            ConvexHull(needle_pts).simplices.astype(np.int64), BLACK, "none")] + _vis("needle"))
LEV_L = float(os.environ.get("DC_LEVL", "0.040"))   # lifting lever, pivot to pad: stock Mitutoyo 21EZA198 spindle lifting lever (length ESTIMATE, unverified)
LEV_Z = DIAL_Z + DIAL_R + 0.024              # at the spindle top, beside the lift cap, as the 21EZA198 mounts
LEV_Y0 = YS + 0.008                        # pivot lug just +y of the spindle; the lever runs +y
LIFT_PER_RAD = 0.050                       # spindle lift per lever radian (about 4 mm at 4.6 deg of press)
lever = Joint("lever", kind="hinge", axis="X", anchor=(XS, LEV_Y0, LEV_Z), lower=-0.40, upper=0.05, meshes=_vis("lever"),
              initial=0.0, target=0.0, stiffness=0.6, damping=0.02, mass=0.01, parent="arm",
              boxes=[((XS, LEV_Y0 + LEV_L / 2, LEV_Z), (0.006, LEV_L, 0.004), BLACK),
                     ((XS, LEV_Y0 + LEV_L - 0.006, LEV_Z + 0.003), (0.014, 0.014, 0.004), BLACK)])  # pad
def _bezel_ring(n=24, r_in=DIAL_R - 0.0020, r_out=BZ_R, y0=YS - DIAL_T / 2 - 0.0015, y1=YS - DIAL_T / 2 + 0.0035):
    """the knurled bezel as an open ring of convex segments (collides), so the white face shows through it"""
    out = []
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pts = np.array([(XS + r * math.cos(a), y, DIAL_Z + r * math.sin(a)) for r in (r_in, r_out) for a in (a0, a1)
                        for y in (y0, y1)])
        out.append(Mesh(pts, ConvexHull(pts).simplices.astype(np.int64), CHROME, "hull"))
    return out


bezel = Joint("bezel", kind="hinge", axis="Y", anchor=(XS, YS - DIAL_T / 2, DIAL_Z), lower=-7.0, upper=7.0, initial=0.0,
              target=0.0, friction=0.02, mass=0.02, parent="arm", meshes=_bezel_ring() + _vis("bezel"))   # knurled ring
STAND = part("comparator", pos=(0, 0, 0), fixed=True, friction=0.3, meshes=_vis("stand"), boxes=(
    ((XS, YS, TOP + GHB / 2), (GBX, GBY, GHB), GRANITE),
    ((XS - GBX / 2 + 0.004, YS, TOP + GHB - 0.010), (0.0005, 0.030, 0.006), WHITE)),         # base label plate
    cylinders=(((XS, YS, TOP + GHB + ANV_H / 2), ANV_R, ANV_H, STEEL, "Z"),                      # raised serrated anvil
               ((XC, YCOL, TOP + GHB + (COL_H - GHB) / 2), 0.0127, COL_H - GHB, CHROME, "Z"),     # 1 in round steel column
               ((XC, YCOL, TOP + GHB + 0.004), 0.020, 0.008, CHROME, "Z")),                      # column foot
    joints=[arm, spindle, needle, lever, bezel])

# ---- gauge block, spacers, trays -------------------------------------------------------------------------------------
YP = float(os.environ.get("DC_YP", "-0.17"))   # parts row
PT_FLOOR = 0.0015                          # parts tray floor (the parts rest on it)
PT_X0, PT_X1, PT_W = 0.316, 0.525, 0.070
MASTER_POCKET = (0.3355, 0.033, 0.050)      # centre x, size x, size y of the master's own (yellow) pocket: rim to first rib
SP_X0 = float(os.environ.get("DC_SPX0", "0.39"))   # first spacer: 10 mm further from the master than before (0016/0026)
PART_XS = (0.33, SP_X0, SP_X0 + 0.05, SP_X0 + 0.10)
PTC = (0.20, 0.36, 0.62)                  # blue plastic parts tray  # parts tray x span and y width (rims clear the open jaws)


def _parts_tray():
    t, h = 0.002, 0.006
    cx, lx = (PT_X0 + PT_X1) / 2, PT_X1 - PT_X0
    return part("parts_tray", pos=(cx, YP, TOP), fixed=True, friction=0.4, meshes=_vis("parts_tray"), boxes=(
        ((0, 0, PT_FLOOR / 2), (lx, PT_W, PT_FLOOR), PTC),
        ((0, PT_W / 2 - t / 2, h / 2), (lx, t, h), PTC), ((0, -PT_W / 2 + t / 2, h / 2), (lx, t, h), PTC),
        ((lx / 2 - t / 2, 0, h / 2), (t, PT_W, h), PTC), ((-lx / 2 + t / 2, 0, h / 2), (t, PT_W, h), PTC)))


PARTS_TRAY = _parts_tray()
GBLOCK = part("gauge_block", pos=(0.33, YP, TOP + PT_FLOOR), mass=0.061, friction=0.25, meshes=_vis("gauge_block"),
              boxes=(((0, 0, 0.0125), (0.009, 0.035, 0.025), CHROME),
                     ((0, -0.0177, 0.0125), (0.006, 0.0004, 0.010), BLACK)))              # long axis along y; engraved end
# trial seed: which spacers are in tolerance. Each of the 8 seeds is one good/bad pattern (bit i = spacer i+1 good),
# so a sorter that ignores the dial gets a given pattern right on 1 seed in 8. Heights look identical.
GOOD_H, BAD_H = (0.02501, 0.02500, 0.02499), (0.02504, 0.02504, 0.02496)   # good within 0.01, bad at least 0.04 off: a zero 1 division off cannot flip a sort
SEED = int(os.environ.get("DC_SEED", "2")) % 8         # seed 2: spacer 2 good, 1 and 3 out (the demo layout)
SPACER_H = {f"s{i + 1}": (GOOD_H[i] if SEED >> i & 1 else BAD_H[i]) for i in range(3)}
SL = 0.040                                 # ground spacer length (x)
SPACERS = {k: part("spacer_" + k, pos=(SP_X0 + 0.05 * i, YP, TOP + PT_FLOOR), mass=0.078, friction=0.25, meshes=_vis("spacer_" + k),
                   boxes=(((0, 0, h / 2), (0.010, SL, h), STEEL),)                      # long axis along y
                   + tuple(((0, -SL / 2 - 0.0002, h * (0.3 + 0.2 * d)), (0.004, 0.0004, 0.003), BLACK)
                           for d in range(i + 1)))                                       # 1/2/3 marks on the -y end
           for i, (k, h) in enumerate(SPACER_H.items())}
TOL = 0.00002
GOOD = {k: abs(h - 0.025) <= TOL for k, h in SPACER_H.items()}


def _tray(name, x, y, color):
    w, d, h, t = 0.10, 0.08, 0.015, 0.003
    return part(name, pos=(x, y, TOP), fixed=True, friction=0.4, meshes=_vis(name), boxes=(
        ((0, 0, t / 2), (w, d, t), color),
        ((0, d / 2 - t / 2, h / 2), (w, t, h), color), ((0, -d / 2 + t / 2, h / 2), (w, t, h), color),
        ((w / 2 - t / 2, 0, h / 2), (t, d, h), color), ((-w / 2 + t / 2, 0, h / 2), (t, d, h), color)))


_FONT = {"A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
         "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
         "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
         "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
         "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
         "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
         "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"]}


def _label(name, text, x, y, color, px=0.0022):
    """plain-text label printed flat on the bench: letters read along -y (left to right from the robots' side), tops toward +x"""
    n = len(text) * 6 - 1
    bx = []
    for i, ch in enumerate(text):
        for r, row in enumerate(_FONT[ch]):
            for c, v in enumerate(row):
                if v == "1":
                    bx.append(((3 * px - r * px, -((i * 6 + c) * px - n * px / 2 + px / 2), 0.0002), (px, px, 0.0004), color))
    bx.append(((0, 0, 0.0001), (9 * px, (n + 2) * px, 0.0002), WHITE))
    return part(name, pos=(x, y, TOP), fixed=True, boxes=tuple(bx[-1:] + bx[:-1]))


TRAY_PASS = _tray("tray_pass", 0.26, -0.20, GREEN)
TRAY_REWORK = _tray("tray_rework", 0.26, -0.29, RED)
LABEL_PASS = _label("label_accept", "ACCEPT", 0.196, -0.20, BLACK)
LABEL_REWORK = _label("label_reject", "REJECT", 0.196, -0.29, BLACK)
OBJECTS = [STAND, PARTS_TRAY, GBLOCK, TRAY_PASS, TRAY_REWORK, LABEL_PASS, LABEL_REWORK] + list(SPACERS.values())

# ---- live coupling: knob turns move the arm; spindle travel drives the needle ----------------------------------------
_S: dict = {}


@on_reset
def _reset(env):
    _S.clear()
    _S.update(step=0, score=0.0, latch=None, hist=[])


def _joints(env):
    art = env.unwrapped.scene["comparator"]
    names = [n.replace("_joint", "") for n in art.joint_names]
    q = labkit._np(art.data.joint_pos)[0]
    return {n: float(q[i]) for i, n in enumerate(names)}, art, names


def reading_mm(j):
    """what the dial shows (mm, + = contact pushed up): the needle's position against the bezel's 0 mark. The needle is
    driven only by the spindle's travel (1 mm per turn); turning the bezel by b moves the 0 mark by b/(2 pi) mm.
    After a correct zero the master reads 0.000."""
    return (j["spindle"] - FREE) * 1e3 + j.get("bezel", 0.0) / (2 * math.pi)


def master_reading_mm(j):
    """what the dial would show with the master seated (its tip height is fixed by the arm): the zero check"""
    return (-ZERO_GAP - j["arm"]) * 1e3 + j.get("bezel", 0.0) / (2 * math.pi)


HAND_CLEAR, STILL_MM, STILL_N = 0.080, 0.001, 21
ZERO_TOL = 0.010                             # zero: the master reads within one 0.01 mm division of 0 (about 2 px at 224)
CONTACT_TOL = 0.00001                        # contact tip on a part's top face (m): 0.01 mm = one dial division, half the pass band
LEVER_FREE = 0.005                           # lever within 5 mrad of rest = released (spindle on its spring alone)
KNOB_MOVE = math.radians(0.05)               # bezel turn per tick that counts as re-setting the zero (0.14 um)
MEASURE_N = 10                               # consecutive valid ticks (0.5 s) for a reading
TRAYS = {"pass": (0.26, -0.20), "rework": (0.26, -0.29)}
TRAY_IN = (0.094 / 2, 0.074 / 2)             # inner half sizes (x, y)


def _parts(env):
    out = {"gb": (labkit._part_state(env, GBLOCK), 0.025, (0.009, 0.035))}
    for k, pt in SPACERS.items():
        out[k] = (labkit._part_state(env, pt), SPACER_H[k], (0.010, SL))
    return out


def under_contact(parts, j):
    """name of the part whose top face the contact tip rests on (tip inside its footprint, at its top), else None"""
    if j.get("lever", 0.0) < -LEVER_FREE:      # lever held: the contact is not resting under its spring
        return None
    tip_z = TIP0 + j["arm"] + j["spindle"]
    for nm, (st, h, (lx, ly)) in parts.items():
        R = st.rotation(); p = np.asarray(st.pos, float)
        d = R.T @ (np.array([XS, YS, tip_z]) - p)
        if abs(d[0]) <= lx / 2 - 0.001 and abs(d[1]) <= ly / 2 - 0.001 and abs(d[2] - h) <= CONTACT_TOL:
            return nm
    return None


def _inside(st, dims, cx, cy, hx, hy, floor_z):
    """the part's whole footprint (oriented corners) inside the box (cx, cy, +-hx, +-hy) and its bottom on the floor"""
    R = st.rotation(); p = np.asarray(st.pos, float)
    lx, ly = dims
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = p + R @ np.array([sx * lx / 2, sy * ly / 2, 0.0])
            if abs(c[0] - cx) > hx or abs(c[1] - cy) > hy or abs(c[2] - floor_z) > 0.002:
                return False
    return True


def tray_of(st, dims=(0.010, 0.040)):
    for nm, (tx, ty) in TRAYS.items():
        if _inside(st, dims, tx, ty, TRAY_IN[0], TRAY_IN[1], TOP + 0.003):
            return nm
    return None




def in_parts_tray(st, dims=(0.009, 0.035)):
    """the master is home: its whole footprint inside its own pocket, resting on the tray floor"""
    x, lx, ly = MASTER_POCKET
    return _inside(st, dims, x, YP, lx / 2, ly / 2, TOP + PT_FLOOR)


BUMP_MM, BUMP_DEG = 2.0, 3.0                 # master moved (its bottom-centre, any direction) or tilted after its return
REST_N, REST_MM = 10, 0.2                    # the reference arms once the returned master has been at rest 10 ticks


def check_master_bump(S, home, gb_xy, tilt_deg):
    """reference = the returned master's pose once it has rested REST_N ticks (hand near or not); any later move over
    BUMP_MM or tilt over BUMP_DEG latches a violation"""
    if "gb_rest" not in S:
        if not home:
            S["gb_hist"] = []
            return None
        h = S.setdefault("gb_hist", [])
        h.append(tuple(gb_xy))
        if len(h) >= REST_N and max(math.dist(h[-REST_N], q) for q in h[-REST_N:]) * 1e3 <= REST_MM:
            S["gb_rest"] = h[-1]
        return None
    if math.dist(S["gb_rest"], gb_xy) * 1e3 > BUMP_MM or tilt_deg > BUMP_DEG:
        return "master bumped after it was put back"
    return None


BZ_CLEAR = 0.036                             # hands clear of the bezel: every pad tip > 36 mm from its centre (rim at 30 mm)
BZ_DRIFT = math.radians(0.1)                 # bezel turned more than 0.1 deg (0.03 division) since the zero = zero voided


def zero_update(S, under, moved, bz_clear, reading, bezel):
    """the scorer's zero: taken only with the master under the released contact, the bezel still and both hands clear
    of it; voided by any cumulative turn of the bezel beyond BZ_DRIFT from where it was when taken. Returns True when
    the zero was voided this tick (readings must then be wiped)."""
    if moved:
        S["zero_rd"], S["bz_at_zero"] = None, None
        return True
    if under == "gb" and bz_clear:
        S["zero_rd"], S["bz_at_zero"] = reading, bezel
        return False
    if S.get("bz_at_zero") is not None and abs(bezel - S["bz_at_zero"]) > BZ_DRIFT:
        S["zero_rd"], S["bz_at_zero"] = None, None
        return True
    return False


def update_measured(S, under, zero_valid, arm_moved):
    """reading bookkeeping: a spacer is measured after MEASURE_N consecutive ticks resting under the released contact
    while a valid zero (set on the master) stands; moving the arm (recalibration) wipes all readings"""
    if arm_moved:                                   # recalibration: every reading taken against the old zero is void
        S["rest"], S["measured"], S["read"], S["rdbuf"], S["demo_slot"] = {}, set(), {}, {}, {}
    rest = S.setdefault("rest", {})
    for k in list(rest):
        if k != under:
            rest[k] = 0
    if under in SPACER_H and zero_valid:
        rest[under] = rest.get(under, 0) + 1
        if rest[under] >= MEASURE_N:
            S.setdefault("measured", set()).add(under)
    elif under in SPACER_H:
        rest[under] = 0


def zero_ok(j, S=None) -> bool:
    """zero is set when, with the master resting under the released contact, the needle read within ZERO_TOL of the
    bezel's 0 mark, and the bezel has not been turned since (S carries that last reading; without S: the geometric check)"""
    if S is not None:
        r = S.get("zero_rd")
        return r is not None and abs(r) <= ZERO_TOL
    return abs(master_reading_mm(j)) <= ZERO_TOL


def evaluate(state) -> float:
    """pure scoring from a state dict: zero_ok, zero_on (who was under the contact when the arm last moved), measured
    (set of spacers that the contact rested on while released), trays {name: tray or None}, latch, clear, still."""
    if state.get("latch") or state["clear"] < HAND_CLEAR or not state["still"]:
        return 0.0
    if not state["zero_ok"] or state.get("zero_on") != "gb":
        return 0.0
    right = sum(1 for k in SPACER_H if state["trays"].get(k) == ("pass" if GOOD[k] else "rework") and k in state["measured"])
    if state["trays"].get("gb") is not None or not state.get("gb_home"):
        return 0.0
    return {3: 1.0, 2: 0.5}.get(right, 0.0)


@on_step
def _track(env):
    _S["step"] += 1
    j, art, names = _joints(env)
    tgt = art.data.default_joint_pos.clone()
    tgt[:, names.index("arm")] = 0.0                                                # arm clamped (zeroing is by the bezel)
    lift = LIFT_PER_RAD * max(0.0, -j["lever"])                                     # pad pressed down -> spindle lifted
    tgt[:, names.index("spindle")] = lift                                           # spring toward full extension, or lever lift
    tgt[:, names.index("lever")] = 0.0                                              # lever return spring
    tgt[:, names.index("needle")] = -2 * math.pi * j["spindle"] / 0.001            # 1 mm per needle revolution
    if NEEDLE_TEST is not None:                  # RENDER-ONLY readability test (never in a trial): needle N divisions off 0
        tgt[:, names.index("needle")] = -2 * math.pi * (FREE + NEEDLE_TEST * 0.00001) / 0.001
    tgt[:, names.index("bezel")] = j["bezel"]                                      # the bezel stays where it is turned
    art.set_joint_position_target(tgt)
    _S["j"] = j
    _see(env, j)                                                                     # camera read of the dial (demo)
    if len(_S.get("hist", [])) and _S.get("padsR_near_bz"):            # bezel pinched: turn it by what is still owed
        b_tgt = 2 * math.pi * (ZERO_GAP + j["arm"]) * 1e3
        if "bz_b0" not in _S:                                            # turn by what the CAMERA last read on the master
            _S["bz_ep"] = _S.get("bz_ep", 0) + 1
            _S["bz_b0"], _S["bz_i"], _S["bz_t"] = j["bezel"], 0.0, 0
            seen = _S.get("seen", {}).get("gb")                          # divisions; None = not read yet: no turn
            _S["bz_ff"] = 0.0 if seen is None else -(BZ_FF if _S["bz_ep"] == 1 else BZ_FF2) * 2 * math.pi * seen * 0.01
            _S.get("seen", {}).pop("gb", None); _S.get("seen_buf", {}).pop("gb", None)   # read again after this turn
        _S["bz_t"] += 1
        moving = abs(j["bezel"] - _S.get("bz_prev", j["bezel"])) > 0.0005
        if not moving and _S["bz_t"] > 50:                               # settled after the turn: trim any slip left
            _S["bz_i"] += BZ_GAIN * (b_tgt - j["bezel"])
        _S["bz_cmd"] = _S["bz_ff"] + _S["bz_i"]
    elif "bz_b0" in _S and _S.get("padsR_far_bz"):                    # hand well away: the next pinch starts afresh
        _S.pop("bz_b0"); _S["bz_cmd"] = 0.0
    _S["bz_prev"] = j["bezel"]
    parts = _parts(env)
    under = under_contact(parts, j)
    moved = abs(j["bezel"] - _S.get("knob_prev", j["bezel"])) > KNOB_MOVE         # the bezel was turned: the zero moved
    if moved:                                                                        # remember what it was set on
        _S["zero_on"] = under
    _S["knob_prev"] = j["bezel"]
    voided = zero_update(_S, under, moved, _S.get("bz_clear", False), reading_mm(j), j["bezel"])
    moved = moved or voided
    valid = zero_ok(j, _S) and _S.get("zero_on") == "gb"
    zr = _S.get("seen", {}).get("gb")                                                # demo exit: 0 SEEN by the camera
    if zr is not None and abs(zr) * 0.01 <= ZERO_DONE_MM and not _S.get("padsR_near_bz", True) and not moved:
        _S["zd_n"] = _S.get("zd_n", 0) + 1
        if _S["zd_n"] >= ZERO_DONE_N:
            _S["zero_done"] = True
    else:
        _S["zd_n"] = 0
    update_measured(_S, under, valid, moved)
    buf = _S.setdefault("rdbuf", {})                     # the dial readings behind each measurement (scorer and log only; the demo never uses them)
    for k in SPACER_H:
        if k == under and valid:
            buf.setdefault(k, []).append(reading_mm(j))
        elif _S.get("rest", {}).get(k, 0) == 0:
            buf[k] = []
        if k in _S.get("measured", set()) and k not in _S.setdefault("read", {}) and len(buf.get(k, [])) >= MEASURE_N:
            _S["read"][k] = float(np.mean(buf[k][-MEASURE_N:]))
    for nm, (st, h, _) in parts.items():
        if float(st.pos[2]) < TOP - 0.02 and not _S["latch"]:
            _S["latch"] = f"{nm} fell off the bench"
    pads = _pads(env)
    _S["under_prev"] = under
    bzc = np.array([XS, YS - DIAL_T / 2, DIAL_Z])
    _S["bz_clear"] = bool(len(pads) == 2 and min(float(np.linalg.norm(p - bzc)) for P in pads for p in P) > BZ_CLEAR)
    if len(pads) == 2:
        pr0 = pads[1].mean(axis=0)
        _S["look_ok"] = bool(np.linalg.norm(pr0 - _S.get("padsR_prev", pr0)) < 0.0005)   # hand settled (not moving)
        _S["padsR_prev"] = pr0
        _S["padsL_z"] = float(pads[0][:, 2].mean())
        pr = pads[1].mean(axis=0)
        _S["padsR_near_bz"] = (abs(pr[0] - XS) < 0.012 and abs(pr[1] - (YS - DIAL_T / 2)) < 0.012
                               and abs(pr[2] - DIAL_Z) < 0.012 and float(np.ptp(pads[1][:, 2])) < 0.0645)
        _S["padsR_far_bz"] = float(np.linalg.norm(pr - np.array([XS, YS - DIAL_T / 2, DIAL_Z]))) > 0.05
    pts = [np.asarray(st.pos, float) + st.rotation() @ np.array([sx * d[0] / 2, sy * d[1] / 2, sz * h])   # part corners
           for st, h, d in parts.values() for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)]
    pts += [np.array([XS + DIAL_R * math.sin(a), YS, DIAL_Z + DIAL_R * math.cos(a)]) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    clear_now = min((float(np.linalg.norm(a - b)) for P in pads for a in P for b in pts), default=0.0) if len(pads) == 2 else 0.0
    _S.setdefault("clear_hist", []).append(clear_now)
    del _S["clear_hist"][:-STILL_N]
    clear = min(_S["clear_hist"]) if len(_S["clear_hist"]) >= STILL_N else 0.0          # hands clear over the whole 1 s
    _S["hist"].append(np.concatenate([np.asarray(st.pos, float) for st, _, _ in parts.values()]
                                     + [0.02 * st.rotation().reshape(-1) for st, _, _ in parts.values()]   # 20 mm lever arm
                                     + [[j["arm"], j["spindle"], 0.030 * j["bezel"], 0.025 * j["needle"]]]))   # rim / needle-tip travel
    _S["hist"] = _S["hist"][-STILL_N:]
    H = np.array(_S["hist"])
    still = len(H) >= STILL_N and float(np.ptp(H, axis=0).max()) <= STILL_MM
    state = dict(zero_ok=zero_ok(j, _S), zero_on=_S.get("zero_on"), measured=_S.get("measured", set()),
                 trays={nm: tray_of(st, d) for nm, (st, _, d) in parts.items()}, latch=_S["latch"], clear=clear, still=still,
                 gb_home=in_parts_tray(parts["gb"][0], parts["gb"][2]))
    Rg = parts["gb"][0].rotation()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(Rg[2, 2])))))
    returned = state["gb_home"] and state["zero_ok"] and state.get("zero_on") == "gb"    # back home after the zero
    bump = check_master_bump(_S, returned, np.asarray(parts["gb"][0].pos, float), tilt)   # 3D: a lift counts too
    _S["gb_armed_step"] = _S.get("gb_armed_step") or (_S["step"] if "gb_rest" in _S else None)
    if bump and not _S["latch"]:
        _S["latch"] = bump
    state["latch"] = _S["latch"]
    _S["score"] = evaluate(state)
    if LOGALL:                                   # one line per control step for the close-up panel (closeup.py)
        print("DCL %d %.7f %.7f %s %d %s %s %.3f %.1f" % (_S["step"], j["spindle"], j["arm"], under or "-", int(state["zero_ok"]),
              state.get("zero_on") or "-", ",".join(f"{k}:{v or '-'}" for k, v in state["trays"].items()),
              reading_mm(j), _S["score"]), flush=True)
    if DEBUG and (_S["step"] % 10 == 0 or _S["step"] <= 2):
        rec = dict(step=_S["step"], knob_deg=round(math.degrees(j["bezel"]), 2), master_rd=round(master_reading_mm(j), 4), arm_mm=round(j["arm"] * 1e3, 4),
                   spindle_mm=round(j["spindle"] * 1e3, 4), reading_mm=round(reading_mm(j), 4), under=under,
                   zero_ok=state["zero_ok"], zero_on=state["zero_on"], measured=sorted(state["measured"]),
                   trays=state["trays"], clear=round(clear, 3), still=still, score=_S["score"], latch=_S["latch"])
        for nm, (st, h, _) in parts.items():
            rec[nm] = [round(float(v), 4) for v in st.pos]
        rec["gb_armed"] = _S.get("gb_armed_step")
        rec["padsL"] = [round(float(v), 3) for v in pads[0].mean(axis=0)] if pads else None
        rec["padsR"] = [round(float(v), 3) for v in pads[1].mean(axis=0)] if pads else None
        print("DCDBG " + json.dumps(rec), flush=True)


SEE_EVERY, SEE_N = 5, 5                    # read the right wrist image every 5 ticks; decide on the median of the last 5


def _see(env, j):
    """the demo's eye: at the look pose, with a part under the released contact, read the dial from the right wrist
    camera image (dial_read.py). Readings are kept per part; the scorer never uses them."""
    if _S["step"] % SEE_EVERY or _S.get("padsR_near_bz") or not _S.get("look_ok"):
        return
    try:
        import dial_read
        sc = env.unwrapped.scene
        cam = None
        for key in ("right_cam", "right_wrist_cam", "camera_right_cam"):
            try:
                cam = sc[key]
                break
            except Exception:  # noqa: BLE001
                continue
        if cam is None:
            keys = list(getattr(sc, "sensors", {}).keys())
            cam = sc[next(k for k in keys if "right" in k and "cam" in k)]
        out = cam.data.output["rgb"]
        img = np.asarray(out[0].detach().cpu().numpy() if hasattr(out, "detach") else out[0])
    except Exception as e:  # noqa: BLE001
        if not _S.get("see_err"):
            _S["see_err"] = 1
            print("DCSEEN_ERR %r sensors=%r" % (e, list(getattr(env.unwrapped.scene, "sensors", {}).keys())), flush=True)
        return
    ph = _S.get("phase")                     # which part the demo is COMMANDED to read now (its own plan, not the sim)
    if not ph or ph[0] is None or ph[1] < _S["step"] - 1:
        return
    under = ph[0]
    r = dial_read.read_dial(img)
    true = reading_mm(j) / 0.01                  # printed for checking only; never used by the demo
    print("DCSEEN %d %s %s %.2f" % (_S["step"], under, "none" if r is None else "%.2f" % r, true), flush=True)
    if r is None:
        return
    buf = _S.setdefault("seen_buf", {}).setdefault(under, [])
    buf.append(r)
    del buf[:-SEE_N]
    if len(buf) >= 3:
        _S.setdefault("seen", {})[under] = float(np.median(buf))


def _pads(env):
    try:
        from yam_arena import runtime as _rt
        o = labkit._np(env.unwrapped.scene.env_origins)[0]
        return [np.asarray(_rt.pad_tips_world(env.unwrapped.scene[n]), float).reshape(-1, 3) - o
                for n in ("left_robot", "right_robot")]
    except Exception:  # noqa: BLE001
        return []


def score(objects) -> float:
    return float(_S.get("score", 0.0))


def success(objects) -> bool:
    return score(objects) >= 1.0


from yam_arena.constants import FINGER_OPEN, FINGER_CLOSED  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402


class Orient(float):                          # a full grip rotation passed through the planner's yaw slot
    def __new__(cls, rot):
        o = float.__new__(cls, 0.0)
        o.rot = np.asarray(rot, float)
        return o


class _ReadWP(Waypoint if isinstance(Waypoint, type) else object):
    """a waypoint during which the demo is commanded to read the dial for one part. The planner reads .label only for
    the waypoint it is executing, so that read stamps the phase; the eye (_see) files its readings under it."""
    def __getattribute__(self, n):
        if n == "label":
            _S["phase"] = (object.__getattribute__(self, "read_key"), _S.get("step", 0))
        return object.__getattribute__(self, n)


def _rw(w, key):
    if not isinstance(Waypoint, type):           # scorer tests stub the planner
        return w
    r = _ReadWP(w.duration, w.pos, w.finger, w.label, w.yaw)
    r.read_key = key
    return r


_prev_gr = _demo.grip_rotation
if not getattr(_prev_gr, "_dc", False):
    def _gr(yaw=0.0, _p=_prev_gr):
        return yaw.rot if hasattr(yaw, "rot") else _p(yaw)
    _gr._dc = True
    _demo.grip_rotation = _gr


def _pitched(deg, yaw=0.0):
    a = math.radians(deg)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return Orient(Ry @ _prev_gr(yaw))


CAP_GRIP = _pitched(float(os.environ.get("DC_CAP_PITCH", "-45")))
BZ_MODE = os.environ.get("DC_BZ_MODE", "pinch")
BZ_SIGN = float(os.environ.get("DC_BZ_SIGN", "1"))     # wrist roll direction that turns the pinched bezel +
_RX90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # fingers pointing +y (face-on), jaws closing along z


class BezelOrient(float):
    """face-on bezel grip; its wrist roll is read live from the turn command (_S['bz_cmd']) unless a fixed roll is given.
    A wrist roll of +t turns the pinched bezel by -t about +y, so the command is the negative of the bezel turn owed."""
    def __new__(cls, fixed=None):
        o = float.__new__(cls, 0.0)
        o.fixed = fixed
        return o

    @property
    def rot(self):
        t = self.fixed if self.fixed is not None else -BZ_SIGN * _S.get("bz_cmd", 0.0)
        return _RX90 @ _prev_gr(t)
PICK = FINGER_OPEN * float(os.environ.get("DC_PICK", "0.45"))   # pick opening in the parts tray: clear of the next pocket
R_TILT = float(os.environ.get("DC_RTILT", "20"))   # right hand: fingers lean toward +y, keeps joints 3/4 off their limits on the slide


def _tilted(deg, yaw):
    a = math.radians(deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    o = Orient(Rx @ _prev_gr(yaw))
    o.dpos = (0.0, -0.013 * math.sin(a), 0.013 * (1 - math.cos(a)))   # keep the pads (13 mm below the TCP) on target
    return o


def _w(d, pos, f, lab, yaw=0.0):
    dp = getattr(yaw, "dpos", (0.0, 0.0, 0.0))
    return Waypoint(d, tuple(float(v) + dv for v, dv in zip(pos, dp)), f, lab, yaw)


def demo():
    parked_l, parked_r = (0.21, 0.28, 0.93), (0.21, -0.28, 0.93)
    if os.environ.get("DC_STILL"):              # setup-image stills only (never a trial): right wrist camera posed over
        down = Orient(_prev_gr(math.pi / 2))      # the parts tray, then over the bins; left arm parked
        return dict(left=[_w(30.0, parked_l, FINGER_CLOSED, "parked")],
                    right=[_w(3.0, (0.40, -0.215, 0.99), FINGER_OPEN, "still: over the parts tray", down),
                           _w(3.0, (0.40, -0.215, 0.99), FINGER_OPEN, "hold for the tray still", down),
                           _w(3.0, (0.20, -0.30, 0.99), FINGER_OPEN, "still: over the bins", down),
                           _w(3.0, (0.20, -0.30, 0.99), FINGER_OPEN, "hold for the bins still", down)])
    L, R = [], []

    def dur(A):
        return sum(w.duration for w in A)

    def wait(A, B, lab):                      # A holds its last pose until B's timeline catches up
        last = A[-1] if A else _w(0.1, parked_l if A is L else parked_r, 0.0, "start")
        if dur(B) > dur(A):
            A.append(Waypoint(dur(B) - dur(A), last.pos, last.finger, lab, last.yaw))

    zc = DIAL_Z + DIAL_R + 0.024 + 0.012        # TCP for the cap: pads sit about 13 mm below the TCP
    LIFT = 0.004
    BZ_IN = float(os.environ.get("DC_BZ_IN", "0.0008"))         # fingertips pressed this far into the knurled rim
    btx = XS - BZ_R + BZ_IN - 0.013             # TCP: closed fingertips (13 mm ahead of it) on the bezel's left rim
    bty = YS - DIAL_T / 2 + 0.0010              # mid-depth of the bezel ring
    bz0 = DIAL_Z - 0.006                        # start a little below the 9 o'clock point; rolling upward turns it +
    BG = _pitched(-90.0, math.pi / 2)           # fingers horizontal along +x, closed jaws stacked vertically
    gz = 0.0255                                 # TCP above the part base: pads at mid-height (13 mm below the TCP)
    carry = ANV + 0.045

    cg = CAP_GRIP
    cp = math.radians(float(os.environ.get("DC_CAP_PITCH", "-45")))
    cxo = float(os.environ.get("DC_CAP_DX", str(-0.012 * abs(math.sin(cp)))))   # pads lean along x with the pitch (-30 tuned at -0.006)

    lpx, lpy, lpz = XS, LEV_Y0 + LEV_L - 0.006, LEV_Z + 0.005      # lever pad top
    PRESS = float(os.environ.get("DC_PRESS", "0.0065"))
    lz = lpz + 0.013 * math.cos(cp) + 0.0017   # TCP with the closed pads just above the lever pad (1.7 mm, as tuned at -30)

    def cap_lift(lab):                         # press the lifting lever down with closed fingers: the contact rises
        return [_w(1.4, (lpx + cxo, lpy, lz + 0.04), FINGER_CLOSED, "over the lifting lever", cg),
                _w(0.8, (lpx + cxo, lpy, lz), FINGER_CLOSED, "fingers on the lever pad", cg),
                _w(0.8, (lpx + cxo, lpy, lz - PRESS), FINGER_CLOSED, lab, cg)]

    def cap_lower():                           # ease off the lever: the contact comes down on the part
        return [_w(0.8, (lpx + cxo, lpy, lz + 0.002), FINGER_CLOSED, "ease off the lever: contact down", cg),
                _w(0.8, (lpx + cxo, lpy, lz + 0.06), FINGER_CLOSED, "clear of the lever", cg)]

    GY = _tilted(R_TILT, float(os.environ.get("DC_RYAW", str(-math.pi / 2))))   # handling grip (load, unload, bins): jaws along x; wrist turned half a turn so its camera stays in front of the dial
    MID = _tilted(R_TILT, 0.0)                  # quarter-turn via between the look pose and the half-turned grip: each wrist
                                                # turn is then about 90 deg, so it never goes the long way round past the dial

    lt = math.radians(float(os.environ.get("DC_LOOK_TILT", "65")))
    look_rot = np.array([[1, 0, 0], [0, math.cos(lt), -math.sin(lt)], [0, math.sin(lt), math.cos(lt)]]) @ _prev_gr(math.pi / 2)
    lky, lkz = float(os.environ.get("DC_LOOK_Y", "-0.11")), float(os.environ.get("DC_LOOK_Z", "0.925"))
    PRE = FINGER_OPEN * float(os.environ.get("DC_PRE", "0.69"))   # jaws held at about 65 mm (bezel 60 mm + margin) while looking
    def look(key=None):                         # key: the part this look is commanded to read (None: just watch)
        W = ([_w(1.0, (XS, lky, lkz), PRE, "back off and turn the wrist camera onto the dial face", Orient(look_rot)),
              _w(1.2, (XS, lky, lkz), PRE, "watch the dial", Orient(look_rot))]
             if os.environ.get("DC_LOOK", "1") == "1" else [])
        return [_rw(w, key) for w in W] if key else W

    SET_BACK = float(os.environ.get("DC_SETBACK", "0.016"))   # set the part down this far short of its measuring spot:
                                                              # on the anvil, front edge 2 mm clear of the lifted tip

    def slide_in(px, py, h, grip_dy, lab):     # pick a part (long axis along y) and slide it +y under the lifted contact
        gy = py + grip_dy
        ty = YS - 0.012 + grip_dy                # contact lands 12 mm toward +y from the part's centre
        return [_w(1.6, (px, gy, TOP + PT_FLOOR + gz + 0.06), PICK, f"over the {lab}, jaws only part open (neighbours close)", GY),
                _w(1.0, (px, gy, TOP + PT_FLOOR + gz), PICK, f"jaws around the {lab}", GY),
                _w(1.2, (px, gy, TOP + PT_FLOOR + gz), FINGER_CLOSED, f"pinch the {lab}", GY),
                _w(1.2, (px, gy, carry + gz), FINGER_CLOSED, f"lift the {lab}", GY),
                _w(1.6, (XS, ty - SET_BACK, carry + gz), FINGER_CLOSED, f"carry the {lab} over the anvil, short of the contact", GY),
                _w(1.0, (XS, ty - SET_BACK, ANV + gz + 0.0008), FINGER_CLOSED, "set it down on the anvil, front edge clear of the tip", GY),
                _w(1.4, (XS, ty, ANV + gz + 0.0008), FINGER_CLOSED, "slide it on the anvil in under the lifted contact", GY),
                _w(0.6, (XS, ty, ANV + gz + 0.0008), FINGER_OPEN, "let go", GY),
                _w(0.8, (XS, ty - 0.02, ANV + gz + 0.05), FINGER_OPEN, "hand up and back, clear of the dial", GY),
                _w(0.6, (XS, -0.15, ANV + gz + 0.09), FINGER_OPEN, "turn the wrist a quarter, well in front of the dial", MID)] + look(), ty

    def slide_out(ty, lab):
        return [_w(0.6, (XS, -0.15, ANV + gz + 0.09), FINGER_OPEN, "turn the wrist a quarter, well in front of the dial", MID),
                _w(1.0, (XS, -0.16, ANV + gz + 0.05), FINGER_OPEN, "hand down and back, well in front of the dial", GY),
                _w(0.8, (XS, ty - 0.04, ANV + gz), FINGER_OPEN, f"low, behind the {lab}", GY),
                _w(1.0, (XS, ty, ANV + gz), FINGER_OPEN, f"jaws around the {lab}", GY),
                _w(1.2, (XS, ty, ANV + gz), FINGER_CLOSED, f"pinch the {lab}", GY),
                _w(1.2, (XS, ty - SET_BACK, ANV + gz + 0.0008), FINGER_CLOSED, "slide it back on the anvil, out from under the contact", GY),
                _w(1.0, (XS, ty - SET_BACK, carry + gz), FINGER_CLOSED, f"lift the {lab} off the anvil", GY)]

    # 1) zero on the gauge block: L lifts the contact, R slides the block in, L lowers it, L turns the zero knob
    L += cap_lift("lift the contact for the gauge block")
    wait(R, L, "wait")
    seq, tx = slide_in(0.33, YP, 0.025, -0.014, "master")           # grip its far end: the fingers stay clear of the dial
    R += seq
    wait(L, R, "hold the contact up")
    L += cap_lower()
    class _RollPos:                              # the roll goes on until the dial reads 0 on the master (live reading)
        def __init__(self, dx):
            self.dx = dx

        def __array__(self, dtype=None, copy=None):
            j, pz = _S.get("j"), _S.get("padsL_z")
            z = bz0
            if j is not None and pz is not None:
                b_tgt = 2 * math.pi * (ZERO_GAP + j["arm"]) * 1e3        # bezel angle that puts 0 under the needle
                z = min(max(pz + BZ_R * (b_tgt - j["bezel"]), bz0), bz0 + 0.030)   # fingertips now + the arc still owed
            return np.array([btx + self.dx, bty, z], dtype=dtype or float)

        def __iter__(self):
            return iter(self.__array__())

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return self.__array__()[i]

    if BZ_MODE == "roll":
        L += [_w(1.2, (btx - 0.05, bty, bz0 + 0.04), FINGER_CLOSED, "jaws closed, over to the bezel's left side", BG),
              _w(1.0, (btx - 0.03, bty, bz0), FINGER_CLOSED, "level with the knurled rim", BG),
              _w(0.8, (btx, bty, bz0), FINGER_CLOSED, "fingertips on the knurled bezel rim", BG),
              Waypoint(2.4, _RollPos(0.0), FINGER_CLOSED, "roll the bezel until 0 sits under the needle", BG),
              Waypoint(1.0, _RollPos(0.0), FINGER_CLOSED, "hold: check the needle on 0", BG),
              Waypoint(0.6, _RollPos(-0.015), FINGER_CLOSED, "fingertips off the rim", BG),
              _w(0.8, (btx - 0.05, bty, bz0 + 0.06), FINGER_CLOSED, "hand back and up, clear of the dial", BG)]
    else:                                        # pinch the knurled bezel across its diameter, roll the wrist
        wait(R, L, "hold the dial view")
        R += [_rw(_w(1.0, (XS, lky, lkz), PRE, "read the dial on the master", Orient(look_rot)), "gb")]   # lever released
        pty, pdy = YS - DIAL_T / 2 + 0.0010 - 0.013, 0.025   # TCP: pads 13 mm ahead, on the bezel ring
        bo = BezelOrient()
        via = Orient(np.array([[1, 0, 0], [0, math.cos(math.pi / 4), -math.sin(math.pi / 4)],
                               [0, math.sin(math.pi / 4), math.cos(math.pi / 4)]]) @ _prev_gr(0.0))
        look_p = (XS, lky, lkz)

        class _Gate:                             # a pinch-cycle waypoint that becomes "keep looking" once the dial reads 0
            def __init__(self, pos):
                self.pos = pos

            def __array__(self, dtype=None, copy=None):
                return np.asarray(look_p if _S.get("zero_done") else self.pos, dtype=dtype or float)

            def __iter__(self):
                return iter(self.__array__())

            def __len__(self):
                return 3

            def __getitem__(self, i):
                return self.__array__()[i]

        class _GateO(float):
            def __new__(cls, o):
                x = float.__new__(cls, 0.0)
                x.o = o
                return x

            @property
            def rot(self):
                return look_rot if _S.get("zero_done") else self.o.rot

        class _GateF:                            # finger command: once zeroed, hold the pre-close width (no empty closes)
            def __init__(self, f):
                self.f = f

            def __float__(self):
                return float(PRE if _S.get("zero_done") else self.f)

            def __add__(self, o): return float(self) + float(o)
            def __radd__(self, o): return float(o) + float(self)
            def __sub__(self, o): return float(self) - float(o)
            def __rsub__(self, o): return float(o) - float(self)
            def __mul__(self, o): return float(self) * float(o)
            def __rmul__(self, o): return float(o) * float(self)
            def __truediv__(self, o): return float(self) / float(o)
            def __rtruediv__(self, o): return float(o) / float(self)
            def __neg__(self): return -float(self)

        def gw(d, pos, f, lab, o):
            return Waypoint(d, _Gate(pos), _GateF(f), lab, _GateO(o))

        def pinch_turn(lab):
            return [gw(0.8, (XS, -0.16, 0.94), FINGER_OPEN, "wrist turns over, well in front of the dial", via),   # IK branch with joint 4 clear of its limit
                    gw(0.8, (XS, pty - pdy - 0.03, DIAL_Z), FINGER_OPEN, "in front of the dial, jaws wide open past the bezel", BezelOrient(0.0)),
                    gw(0.8, (XS, pty, DIAL_Z), FINGER_OPEN, "jaws around the knurled bezel, clear of the crystal", BezelOrient(0.0)),
                    gw(0.6, (XS, pty, DIAL_Z), FINGER_CLOSED, "pinch the bezel rim top and bottom", BezelOrient(0.0)),
                    gw(1.5, (XS, pty, DIAL_Z), FINGER_CLOSED, lab, bo),
                    gw(1.0, (XS, pty, DIAL_Z), FINGER_CLOSED, "hold", bo),
                    gw(0.8, (XS, pty, DIAL_Z), FINGER_OPEN, "open the jaws slowly, straight apart", bo),
                    gw(0.6, (XS, pty - 0.04, DIAL_Z), FINGER_OPEN, "hand straight back along the dial axis", bo),
                    gw(0.8, (XS, -0.16, 0.94), FINGER_OPEN, "back out, well in front of the dial", via)]
        for c in range(BZ_CYCLES):                 # turn, let go, look; repeat until the camera shows 0 under the needle
            R += pinch_turn("turn the bezel by what the dial shows" if c == 0 else "trim the bezel by what the dial shows") + look("gb")
        wait(L, R, "wait for the zero")
    L += cap_lift("lift the contact off the gauge block")
    wait(R, L, "wait for the zero")
    R += slide_out(tx, "gauge block")
    R += [_w(1.4, (0.33, YP - 0.014, carry + gz), FINGER_CLOSED, "carry the block back", GY),
          _w(1.0, (0.33, YP - 0.014, TOP + PT_FLOOR + gz + 0.001), FINGER_CLOSED, "set it down", GY),
          _w(0.6, (0.33, YP - 0.014, TOP + PT_FLOOR + gz + 0.001), FINGER_OPEN * 0.45, "let go (jaws only part open)", GY),
          _w(0.8, (0.33, YP - 0.014, TOP + gz + 0.06), FINGER_OPEN * 0.45, "straight up and clear", GY)]
    # 2) each spacer: slide in, read it with the camera, slide out, bin it by that reading
    class _TrayPos:                              # resolved while the run plays: from the dial reading THIS run took
        def __init__(self, k, dz, use_dx, dp):
            self.k, self.dz, self.use_dx, self.dp = k, dz, use_dx, dp

        def __array__(self, dtype=None, copy=None):
            r = _S.get("seen", {}).get(self.k)                       # what the camera read (divisions)
            r = None if r is None else r * 0.01                      # -> mm
            if r is None and self.k not in _S.get("demo_slot", {}):   # no camera reading yet: hold the part where it is
                return np.array([XS + self.dp[0], YS - 0.071 + self.dp[1], self.dz + self.dp[2]], dtype=dtype or float)
            name = "pass" if (r is not None and abs(r) <= TOL * 1e3) else "rework"
            got = _S.setdefault("demo_slot", {})
            if self.k not in got and r is not None:
                n = sum(1 for v in got.values() if v[0] == name)
                got[self.k] = (name, (-0.025, 0.025, 0.0)[n])
            name, dx = got.get(self.k, (name, 0.0))
            tx, ty = TRAYS[name]
            return np.array([tx + (dx if self.use_dx else 0.0) + self.dp[0], ty + self.dp[1], self.dz + self.dp[2]], dtype=dtype or float)

        def __iter__(self):
            return iter(self.__array__())

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return self.__array__()[i]

    dpg = getattr(GY, "dpos", (0.0, 0.0, 0.0))
    slot = {}                                                                  # spread parts that share a tray along x
    REL = FINGER_OPEN * float(os.environ.get("DC_REL", "0.45"))               # release opening in a shared bin
    for i, k in enumerate(("s1", "s2", "s3")):                               # names only: the heights are hidden from the demo
        px = SP_X0 + 0.05 * i
        seq, tx = slide_in(px, YP, 0.025, -0.014, f"spacer {i + 1}")             # nominal height for the grasp
        wait(R, L, "wait")
        R += seq
        wait(L, R, "hold the contact up")
        L += [_w(0.8, (lpx + cxo, lpy, lz + 0.002), FINGER_CLOSED, "ease off the lever onto the spacer", cg),
              _rw(_w(1.5, (lpx + cxo, lpy, lz + 0.006), FINGER_CLOSED, "read the dial", cg), k)]
        L += [_w(0.8, (lpx + cxo, lpy, lz - PRESS), FINGER_CLOSED, "press the lever: contact off the spacer", cg)]
        wait(R, L, "wait for the reading")
        R += slide_out(tx, f"spacer {i + 1}")
        R += [Waypoint(1.6, _TrayPos(k, carry + gz, False, dpg), FINGER_CLOSED, "carry it to the tray its reading calls for", GY),
              Waypoint(1.0, _TrayPos(k, TOP + 0.003 + gz + 0.002, True, dpg), FINGER_CLOSED, "lower it into the tray", GY),
              Waypoint(0.6, _TrayPos(k, TOP + 0.003 + gz + 0.002, True, dpg), REL, "let go (jaws only part open, clear of its neighbour)", GY),
              Waypoint(0.8, _TrayPos(k, carry + gz + 0.04, True, dpg), REL, "straight up and clear", GY)]
    wait(L, R, "hold the contact up")
    L += cap_lower()
    L.append(_w(2.0, parked_l, FINGER_CLOSED, "park (fingertips stay closed, as on the lever)"))
    R.append(_w(2.0, parked_r, REL, "park (jaws stay as they let go: no empty close)"))
    wait(L, R, "parked"); wait(R, L, "parked")
    L.append(_w(2.0, parked_l, FINGER_CLOSED, "hold still")); R.append(_w(2.0, parked_r, REL, "hold still"))
    return dict(left=L, right=R)
