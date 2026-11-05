"""On the station table: an aluminium base plate lying on a printed gluing stand, a printed cover plate of the same size
lying on a mat to the left, and a small spring clamp clipped on the front edge of a clamp rack to the right.

The plates are 60 mm long (along x, away from the robots), 40 mm wide and 6 mm thick. The base plate lies flat on the
grey gluing stand, 45 mm above the bench, its front edge flush with the stand's front face; white lines on the stand's
top, 1 mm outside the plate on its other three sides, mark where it belongs. Under the middle of the plate's front edge
the stand has a notch, 22 mm wide and 26 mm deep, open at the front and marked by two red marks on the front face: the
clamp relief. The base plate's top has been glued, so the cover slides on it easily (wet glue: friction 0.08 between
the plates). The cover lies flat on the mat.

The clamp is a small sprung C-clamp, 66 mm long, 12 mm wide: an orange nylon frame with a fixed lower jaw, a top
grip and a back spine, and a black upper jaw on a slide in the spine, sprung down onto the lower jaw (about 6 N on a
12 mm stack). Under the top grip is a trigger on the same slide: squeezing trigger and grip together lifts the upper
jaw, but only to 14 mm open, 2 mm more than the stack. Under the upper jaw is a soft black foam pad that stands 2.5 mm
proud of it, with a rounded nose 10 mm across at the jaw tip; it squashes flat at 0.3 N. So even fully open, the gap
between the lower jaw and the pad (11.5 mm) is less than the stack: as the clamp goes on, the pad's nose rides up onto
the cover and slides along its top, pressing on it, and that drags the cover along on the wet glue unless the cover is
pressed down. Grip and trigger are ribbed and slope up towards their ends by 10 degrees. The clamp hangs on the rack's
10 mm thick front edge with its jaws closed on it and pointing away from the robots, grip and trigger towards them.

Clamping means: set the cover on the base with all four edges flush and hold it there, pressing straight down on it;
squeeze the clamp open, take it off the rack, slide its open jaws over the stacked plates' front edge through the
notch, lower jaw under the base plate and upper jaw over the cover, and let go so the jaws close on both plates; then
take the holding hand away. On the glue the cover slides easily, and the clamp's pad drags it as the jaws go on, so it
is held, pressed down, while the clamp goes on: the press adds to the glue's grip on the cover, which on its own is
less than the pad's drag.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

ALU = (0.78, 0.79, 0.81)  # the base plate
COVER_C = (0.16, 0.17, 0.19)  # the dark grey printed cover
STAND_C = (0.55, 0.56, 0.58)  # the grey printed gluing stand
WHITE = (0.96, 0.96, 0.96)
RED = (0.82, 0.10, 0.08)
NYLON = (0.95, 0.50, 0.10)  # the clamp's orange frame
BLACK = (0.06, 0.06, 0.07)  # its jaw slide, trigger and rubber jaw faces
RACK_C = (0.40, 0.41, 0.44)
MAT_C = (0.18, 0.36, 0.30)  # green cutting mat

# -- the plates (loose). Frame: origin at the middle of the plate's underside; length along x. ----------------------
PL, PW, PT = 0.060, 0.040, 0.006  # 60 x 40 x 6 mm
SX, SY = 0.31, -0.04  # the base plate's centre on the stand, world
STAND_H = 0.045  # the stand's top: the base plate's underside, 45 mm above the bench
EDGE_X = SX - PL / 2  # the front edge (towards the robots): the one that is clamped
MAT_XY, MAT = (0.30, 0.12), (0.080, 0.060, 0.003)
# Friction: PhysX averages the two materials of a contact. Wet glue between the plates is modelled as low friction,
# no viscosity or adhesion: the plates combine to 0.08. The base is 0.56 on the stand (1.0), the cover 0.42 on the mat
# (0.8) and 0.27 on the robot's fingers (0.5), the clamp (0.66) 0.35 on the cover and 0.39 on the base.
BASE_MU, COVER_MU, CLAMP_MU = 0.12, 0.04, 0.66
GLUE_MU = (BASE_MU + COVER_MU) / 2  # 0.08

BASE = part("base_plate", pos=(SX, SY, TOP + STAND_H), mass=0.039, friction=BASE_MU,
            boxes=[((0.0, 0.0, PT / 2), (PL, PW, PT), ALU)])
COVER = part("cover_plate", pos=(*MAT_XY, TOP + MAT[2]), mass=0.018, friction=COVER_MU,
             boxes=[((0.0, 0.0, PT / 2), (PL, PW, PT), COVER_C)])
MAT_PART = part("mat", pos=(*MAT_XY, TOP), fixed=True, friction=0.8, boxes=[((0.0, 0.0, MAT[2] / 2), MAT, MAT_C)])

# -- the gluing stand (fixed), with the base plate's outline and the clamp relief ------------------------------------
NOTCH_W, NOTCH_D = 0.022, 0.026
STAND_X1 = PL / 2 + 0.005  # 5 mm of top behind and beside the plate
STAND_HW = PW / 2 + 0.005
LINE_W, LINE_T, LINE_GAP = 0.002, 0.0003, 0.001
_nx = -PL / 2 + NOTCH_D  # the back of the notch, from the plate's centre
_cheek_w = STAND_HW - NOTCH_W / 2
_lx = PL / 2 + LINE_GAP + LINE_W / 2
_ly = PW / 2 + LINE_GAP + LINE_W / 2
STAND = part(
    "gluing_stand", pos=(SX, SY, TOP), fixed=True, friction=1.0,
    boxes=[
        (((_nx + STAND_X1) / 2, 0.0, STAND_H / 2), (STAND_X1 - _nx, 2 * STAND_HW, STAND_H), STAND_C),
        *[(((-PL / 2 + _nx) / 2, s * (NOTCH_W / 2 + _cheek_w / 2), STAND_H / 2), (NOTCH_D, _cheek_w, STAND_H), STAND_C)
          for s in (1, -1)],
        # the outline: white lines 2 mm wide, 1 mm outside the plate's back and side edges
        ((_lx, 0.0, STAND_H + LINE_T / 2), (LINE_W, 2 * _ly + LINE_W, LINE_T), WHITE),
        *[(((-PL / 2 + _lx + LINE_W / 2) / 2, s * _ly, STAND_H + LINE_T / 2), (_lx + LINE_W / 2 + PL / 2, LINE_W,
                                                                               LINE_T), WHITE) for s in (1, -1)],
        # red marks on the front face either side of the notch
        *[((-PL / 2 - 0.0002, s * 0.016, STAND_H - 0.006), (0.0004, 0.004, 0.010), RED) for s in (1, -1)],
    ],
)

# -- the spring clamp (loose, one slide). Frame: origin at the bottom of the lower jaw, under the spine's front face;
# x along the jaws (away from the robots), z up. It is built where it starts: on the rack, jaws 10 mm apart. ---------
JAW_L, JAW_T, HALF_W = 0.022, 0.005, 0.006  # jaws 22 mm long, 5 mm thick, 12 mm wide
SPINE_T = 0.006  # the spine from x -6 to 0 mm
GRIP_X0, GRIP_X1 = -0.044, -0.006  # grip and trigger: 38 mm long behind the spine
O_RACK = 0.010  # the jaw opening on the rack (its edge is 10 mm thick): the slide's zero
O_OPEN = 0.014  # squeezed fully open: the slide's stop, only 2 mm more than the stack
O_STACK = 2 * PT  # clamped on the two plates
CLAMP_K, CLAMP_C = 500.0, 5.0  # N/m: 5 N on the rack, 6 N on the plates, 7 N squeezed open
# The foam pad under the upper jaw: a slide on the jaw, 2.5 mm proud when free, squashed flat (into the jaw) under the
# clamp's spring. Its spring has no preload: 0 N free, 0.3 N squashed flat. Fully open, the gap from the lower jaw to
# the pad is 11.5 mm, less than the 12 mm stack, so the pad always presses on the cover as the jaws go on.
PAD_OUT = 0.0025
PAD_K, PAD_C = 120.0, 1.0  # N/m, N s/m
PAD_R = 0.005  # its nose: a rounded end 10 mm across at the jaw tip, so it rides up onto the cover's edge
PAD_L = JAW_L - PAD_R  # the flat of the pad, from the spine to the middle of the nose
PITCH = math.radians(10.0)  # grip and trigger slope up 10 degrees towards their ends, square to a hand tilted so
_s, _c = math.sin(PITCH), math.cos(PITCH)
# The hand's centre between its fingers, squeezing the clamp fully open, in the clamp's frame; grip and trigger faces
# are the planes 6 mm either side of it along the closing line (sin, 0, cos) of a hand pointing 10 degrees down.
GRIP_U, GRIP_Z = -0.024, 0.023
FACE = 0.006
RIB_R = 0.0015
RIB_U = (-0.010, -0.016, -0.022, -0.028, -0.034, -0.040)
CORE1 = (0.0225, 0.025)  # the grip's core, bottom and top
CORE2 = (0.020, 0.022)  # the trigger's core, squeezed open (built 8 mm lower)


def _rib_z(u: float, side: int) -> float:
    """Height of a rib's axis at u whose crest lies on the grip's face (side +1) or the trigger's (side -1), open."""
    return GRIP_Z + (side * (FACE - RIB_R) - _s * (u - GRIP_U)) / _c


_dz = O_RACK - O_OPEN  # the slide built 8 mm below its fully open place
_grip_ribs, _trig_ribs, _grip_posts, _trig_posts = [], [], [], []
for _u in RIB_U:
    z1 = _rib_z(_u, 1)
    _grip_ribs.append(((_u, 0.0, z1), RIB_R, 2 * HALF_W, NYLON, "Y"))
    if z1 > CORE1[1]:
        _grip_posts.append(((_u, 0.0, (CORE1[1] + z1) / 2), (2 * RIB_R, 2 * HALF_W, z1 - CORE1[1] + 0.0002), NYLON))
    z2 = _rib_z(_u, -1)
    _trig_ribs.append(((_u, 0.0, z2 + _dz), RIB_R, 2 * HALF_W, BLACK, "Y"))
    if z2 < CORE2[0]:
        _trig_posts.append(((_u, 0.0, (CORE2[0] + z2) / 2 + _dz), (2 * RIB_R, 2 * HALF_W, CORE2[0] - z2 + 0.0002),
                            BLACK))

_gl = GRIP_X1 - GRIP_X0
JAW = Joint(
    "jaw", kind="slide", axis="Z", anchor=(JAW_L / 2, 0.0, JAW_T + O_RACK + JAW_T / 2),
    lower=-O_RACK, upper=O_OPEN - O_RACK, initial=0.0, stiffness=CLAMP_K, damping=CLAMP_C, target=0.0, mass=0.010,
    boxes=[
        ((JAW_L / 2, 0.0, JAW_T + O_RACK + JAW_T / 2), (JAW_L, 2 * HALF_W, JAW_T), BLACK),  # upper jaw
        ((-SPINE_T / 2, 0.0, (CORE2[0] + _dz + JAW_T + O_RACK + JAW_T) / 2),
         (SPINE_T, 0.005, JAW_T + O_RACK + JAW_T - CORE2[0] - _dz), BLACK),  # the slide in the spine's slot
        (((GRIP_X0 + GRIP_X1) / 2, 0.0, (CORE2[0] + CORE2[1]) / 2 + _dz), (_gl, 2 * HALF_W, CORE2[1] - CORE2[0]),
         BLACK),  # trigger core
        *_trig_posts,
    ],
    cylinders=_trig_ribs,
)

# The pad, built where it starts: squashed flat on the rack, its underside flush with the upper jaw's face. Its slide
# runs from -2.5 mm (free, proud of the jaw) to 0 (flat).
_jaw_face = JAW_T + O_RACK
PAD = Joint(
    "pad", kind="slide", axis="Z", parent="jaw", anchor=(PAD_L / 2, 0.0, _jaw_face + PAD_OUT / 2),
    lower=-PAD_OUT, upper=0.0, initial=0.0, stiffness=PAD_K, damping=PAD_C, target=0.0, mass=0.003,
    boxes=[((PAD_L / 2, 0.0, _jaw_face + PAD_OUT / 2), (PAD_L, 2 * HALF_W - 0.001, PAD_OUT), BLACK)],
    cylinders=[((PAD_L, 0.0, _jaw_face + PAD_R), PAD_R, 2 * HALF_W - 0.001, BLACK, "Y")],
)

CLAMP_START = (EDGE_X - 0.002, SY - 0.09, TOP + 0.036)  # on the rack: jaws 20 mm over its edge
_cheek = HALF_W - 0.0025
CLAMP = part(
    "clamp", pos=CLAMP_START, mass=0.025, friction=CLAMP_MU,
    boxes=[
        (((JAW_L - SPINE_T) / 2, 0.0, JAW_T / 2), (JAW_L + SPINE_T, 2 * HALF_W, JAW_T), NYLON),  # lower jaw
        ((JAW_L / 2, 0.0, JAW_T - 0.0004), (JAW_L, 2 * HALF_W - 0.001, 0.0008), BLACK),  # its rubber face
        *[((-SPINE_T / 2, s * (0.0025 + _cheek / 2), (JAW_T + CORE1[1]) / 2), (SPINE_T, _cheek, CORE1[1] - JAW_T),
           NYLON) for s in (1, -1)],  # the spine's two cheeks round the slide
        (((GRIP_X0 + GRIP_X1) / 2, 0.0, (CORE1[0] + CORE1[1]) / 2), (_gl, 2 * HALF_W, CORE1[1] - CORE1[0]),
         NYLON),  # grip core
        *_grip_posts,
    ],
    cylinders=_grip_ribs,
    joints=[JAW, PAD],
)

# -- the clamp rack (fixed): a steel bar 10 mm thick, its front edge level with the stand's, on a block at its back --
RACK_Y = CLAMP_START[1]
RACK_Z0 = CLAMP_START[2] + JAW_T - TOP  # the bar's underside, 41 mm above the bench
RACK_T, RACK_L, RACK_W = O_RACK, 0.050, 0.040
RACK_BLOCK = 0.025
RACK = part(
    "clamp_rack", pos=(EDGE_X, RACK_Y, TOP), fixed=True, friction=0.8,
    boxes=[
        ((RACK_L / 2, 0.0, RACK_Z0 + RACK_T / 2), (RACK_L, RACK_W, RACK_T), RACK_C),
        ((RACK_L - RACK_BLOCK / 2, 0.0, RACK_Z0 / 2), (RACK_BLOCK, RACK_W, RACK_Z0), RACK_C),
    ],
)

OBJECTS = [BASE, COVER, MAT_PART, STAND, CLAMP, RACK]


# -- the spring --------------------------------------------------------------------------------------------------
# The slide is built at its start (jaws 10 mm apart on the rack, slide at 0, drive target 0), and so is the pad
# (squashed flat, at 0); from the first control step the slide's drive target is the closed jaws, so the spring
# presses the jaws shut on whatever is between them, and the pad's is free (2.5 mm proud, no preload), so the pad
# stands proud whenever nothing presses on it.
@on_step
def _spring(env):
    import torch

    asset = env.unwrapped.scene["clamp"]
    names = list(asset.joint_names)
    ids = [names.index(JAW.joint_name), names.index(PAD.joint_name)]
    asset.set_joint_position_target(torch.tensor([[-O_RACK, -PAD_OUT]], device=asset.device), joint_ids=ids)


# -- scoring -------------------------------------------------------------------------------------------------------
CONTACT = 0.001  # each jaw face within 1 mm of its plate's face
OPEN_TOL = 0.001  # the jaws open by the stack's 12 mm, within 1 mm
OVER = 0.008  # jaw tips at least 8 mm in from the front edge
FLUSH = 0.0015  # the cover's edges within 1.5 mm of the base's
TWIST = 3.0  # degrees
MOVE_MAX = 0.010  # the base plate no corner more than 10 mm from its outline
TILT_MAX = 10.0
CORNERS = [(sx * PL / 2, sy * PW / 2, 0.0) for sx in (1, -1) for sy in (1, -1)]
SUCCESS_HOLD_S = 11.0  # end a trial early only once it has stayed clamped and flush for 11 s: in the demo the jaws
#                        close from 26.3 s and the plan ends at 35.2 s with both hands parked, so the video plays to its
#                        end


def _yaw(rot) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def base_moved(base) -> float:
    """The largest horizontal distance of a base plate corner from where it started (m)."""
    start = np.array([SX, SY])
    return max(float(np.linalg.norm(base.to_world(c)[:2] - (start + np.array(c[:2])))) for c in CORNERS)


def jaw_opening(clamp) -> float:
    return float(clamp.joints["jaw"] + O_RACK)


def clamp_fit(base, cover, clamp):
    """(lower jaw face from the base's underside, upper jaw face from the cover's top, opening minus 12 mm, how far the
    jaw tips are in from the front edge, the larger of the two; how far off the plates' centre line, sideways),
    measured at the jaw tips, in m."""
    o = jaw_opening(clamp)
    lo = base.to_local(clamp.to_world((JAW_L, 0.0, JAW_T)))
    up = cover.to_local(clamp.to_world((JAW_L, 0.0, JAW_T + o)))
    over = min(lo[0], up[0]) + PL / 2
    side = max(abs(lo[1]), abs(up[1]))
    return float(lo[2]), float(up[2] - PT), o - O_STACK, float(over), float(side)


def clamped(base, cover, clamp) -> bool:
    """Both jaws on the stack: each jaw face within 1 mm of its plate's outer face, the jaws open 11 to 13 mm, the jaw
    tips at least 8 mm in from the front edge of both plates and fully on them sideways, the clamp tilted no more than
    10 degrees from the base."""
    dz_lo, dz_up, do, over, side = clamp_fit(base, cover, clamp)
    tilt = math.degrees(math.acos(float(np.clip((base.rotation().T @ clamp.rotation())[2, 2], -1.0, 1.0))))
    return (abs(dz_lo) <= CONTACT and abs(dz_up) <= CONTACT and abs(do) <= OPEN_TOL and over >= OVER
            and side <= PW / 2 - HALF_W and tilt <= TILT_MAX)


def cover_offset(base, cover):
    """(the largest offset of a cover edge from the base's, along x or y, at any corner, in m; the twist in degrees),
    in the base's frame."""
    off = 0.0
    for c in CORNERS:
        p = base.to_local(cover.to_world(c))
        off = max(off, abs(p[0] - c[0]), abs(p[1] - c[1]))
    twist = math.degrees(_wrap(_yaw(base.rotation().T @ cover.rotation())))
    return float(off), float(twist)


def score(objects):
    """0 if the base plate has moved more than 10 mm off its outline (any corner) or tilted more than 10 degrees. 0
    unless the clamp grips both plates (see clamped). 0.5 clamped with the cover misaligned: any edge more than 1.5 mm
    from the base's or twisted more than 3 degrees. 1.0 clamped with the cover flush within 1.5 mm and 3 degrees."""
    base, cover, clamp = objects["base_plate"], objects["cover_plate"], objects["clamp"]
    if base_moved(base) > MOVE_MAX or base.tilt_deg() > TILT_MAX:
        return 0.0
    if not clamped(base, cover, clamp):
        return 0.0
    off, twist = cover_offset(base, cover)
    return 1.0 if off <= FLUSH and abs(twist) <= TWIST else 0.5


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """The cover's worst edge offset plus the jaw tips' distance from their clamped place, 16 mm in from the front edge
    and 3 mm off each plate (m)."""
    base, cover, clamp = objects["base_plate"], objects["cover_plate"], objects["clamp"]
    off, _ = cover_offset(base, cover)
    tip = base.to_local(clamp.to_world((JAW_L, 0.0, JAW_T)))
    return off + float(np.linalg.norm(tip - np.array([-PL / 2 + 0.016, 0.0, 0.0])))


# -- scripted demo -------------------------------------------------------------------------------------------------
# Left arm: picks the cover off the mat by its long sides (pad tips 1 mm above its underside, fingers commanded 3 mm
# inside it each side: firm, not crushing), carries it over the base and, closed loop on the live plates (see
# ``_track``), lines its centre and its yaw up on the base's, then sets it down, opens wide, waits, lifts and comes down
# with its fingers closed on the cover's back half, 18 mm behind its centre, pressing straight down (about 3 N, leaned
# against the arm's compliance so the force is vertical). It holds the cover there until the clamp is on and the right
# hand is away, then lifts straight off and parks.
# Right arm: points forward and 10 degrees down, fingers closing in the vertical plane, the closing line square to the
# clamp's sloped grip and trigger. It comes in behind the clamp on the rack with its fingers 40 mm apart, top finger
# over the grip and bottom finger under the trigger, and squeezes them together to the slide's stop (commanded 4 mm
# past it each side, about 20 N against the spring's 7 N): the jaws open to 14 mm, the lower jaw and the pad 0.75 mm
# clear of the rack's edge. It draws the clamp off the rack, carries it beside the stand and waits there until the
# cover is held. Then it brings the open jaws 30 mm in front of the stack, trims the clamp's measured position onto the
# stack's (closed loop), freezes the trim, slides the jaws in through the notch (lower jaw 1 mm under the base, upper
# jaw 1 mm over the cover, tips 16 mm in) and lets go: it opens its fingers 64 mm apart, waits, draws back and parks.
# On the way in the pad, 1.5 mm lower than the cover's top, rides up onto it on its rounded nose and slides along it,
# pressing about 0.15 to 0.2 N (its spring in series with the right arm's give): that drags the cover 0.05 to 0.07 N,
# twice what the wet glue alone holds it with (0.03 N: 0.08 of its weight and the pad's force), and a quarter of what
# the glue holds it with under the left hand's 3 N press (0.27 N), so the pressed cover stays put. The pad's push while
# its nose rides up the cover's edge peaks at 0.11 to 0.15 N, still well under the 0.27 N. Unheld, the cover goes along
# with the jaws, about 18 mm.
TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.16, -0.20, TOP + 0.12)
GRIP_UP = 0.001  # the left pad tips 1 mm above the cover's underside
COVER_WIDE = -0.028  # pads 56 mm apart round the 40 mm cover
COVER_HOLD = -0.017  # commanded 34 mm apart: 3 mm inside it each side
PICK_Z = TOP + MAT[2] + GRIP_UP + TIP_ABOVE_TCP
PICK_HOVER = TOP + 0.07
SQUEEZE_WIDE = -0.020  # the right fingers 40 mm apart round grip and trigger (16 mm across them on the rack)
SQUEEZE = -0.002  # past the stop: grip and trigger are 12 mm across when the jaws are fully open
RELEASE_OPEN = -0.032  # let go 64 mm apart, 26 mm clear of grip and trigger: fully open, the bottom finger would
#                        reach down to the table
SET_GAP = 0.0003

C_LEFT = np.array([1.12, -0.73, 2.16]) / 1000.0  # m/N, the left arm's compliance at the pads, as on the servo rig
HOLD_F = 5.0
PRESS_DX = 0.018  # the press point behind the cover's centre


def cover_z(underside: float) -> float:
    """Left TCP height, gripping the cover, that puts its underside ``underside`` above the base plate's top."""
    return TOP + STAND_H + PT + underside + GRIP_UP + TIP_ABOVE_TCP


CARRY_Z = cover_z(0.025)
ENTRY_Z = cover_z(0.002)
SET_Z = cover_z(SET_GAP)
OPEN_UP_Z = cover_z(PT + 0.010)  # open pad tips 10 mm over the cover's top
TOUCH_Z = TOP + STAND_H + 2 * PT + TIP_ABOVE_TCP  # closed tips on the cover's top

# The clamp in the right hand, fully open: where its frame is relative to the base plate when its jaws are in place.
LOW_GAP = 0.001  # the lower jaw under the base, and so the upper jaw over the cover (14 - 12 - 1 mm)
CLAMP_IN = np.array([-PL / 2 - 0.006, 0.0, -LOW_GAP - JAW_T])  # spine 6 mm in front of the edge, lower jaw 1 mm under
BACK = 0.030  # drawn back 30 mm for the line-up
HAND_IN_CLAMP = np.array([GRIP_U, 0.0, GRIP_Z])

_prev_grip = _demo.grip_rotation
if not getattr(_prev_grip, "_rot_duck", False):
    def _grip_rotation(yaw=0.0, _prev=_prev_grip):
        return yaw.rot if hasattr(yaw, "rot") else _prev(yaw)

    _grip_rotation._rot_duck = True
    _demo.grip_rotation = _grip_rotation

_prev_action7 = _demo.ArmPlan.action7
if not getattr(_prev_action7, "_clock", False):
    def _action7(self, t, _prev=_prev_action7):
        _demo._plan_clock = float(t)  # the plan's time, for waypoints that move with the trial
        return _prev(self, t)

    _action7._clock = True
    _demo.ArmPlan.action7 = _action7


def _clock() -> float:
    return float(getattr(_demo, "_plan_clock", 0.0))


def _rz(a: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), a)


# The right hand: pointing along +x and 10 degrees down, fingers closing along (sin, 0, cos).
FORWARD = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -(math.pi / 2 - PITCH)) @ _prev_grip(math.pi / 2)


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation, fixed or read live (``fn``)."""

    def __new__(cls, rot=None, fn=None):
        obj = float.__new__(cls, 0.0)
        obj._rot, obj._fn = (None if rot is None else np.asarray(rot, float)), fn
        return obj

    @property
    def rot(self):
        return self._fn() if self._fn is not None else self._rot


TRACK_GAIN, TRACK_STEP, TRACK_MAX = 0.15, 0.0004, 0.004  # per step: 15% of the offset, at most 0.4 mm; 4 mm in all
YAW_GAIN, YAW_STEP, YAW_MAX = 0.15, math.radians(0.5), math.radians(8.0)
_live = {"base": np.array([SX, SY, TOP + STAND_H]), "psi": 0.0, "ccorr": np.zeros(2), "cyaw": 0.0,
         "kcorr": np.zeros(3), "base_frozen": False}
_plan = {"c_trim0": 1e9, "c_trim1": 1e9, "k_trim0": 1e9, "k_trim1": 1e9}


def _live_reset():
    _live.update(base=np.array([SX, SY, TOP + STAND_H]), psi=0.0, ccorr=np.zeros(2), cyaw=0.0, kcorr=np.zeros(3),
                 base_frozen=False)
    _demo._plan_clock = 0.0


@on_reset
def _reset_live(env):
    _live_reset()


def _base_to_world(local) -> np.ndarray:
    return _live["base"] + _rz(_live["psi"]) @ np.asarray(local, float)


@on_step
def _track(env):
    """The base plate's pose, live until the clamp's trim is frozen. While the left hand holds the cover over the base:
    its trim, integrated from the cover's offset and yaw from the base's. While the right hand holds the open clamp in
    front of the stack: its trim, integrated from the clamp frame's offset from where it should be."""
    import labkit

    t = _clock()
    base = labkit._part_state(env, labkit._SPECS["base_plate"])
    if not _live["base_frozen"]:
        _live.update(base=base.pos.copy(), psi=_yaw(base.rotation()))
        if t >= _plan["k_trim1"]:
            _live["base_frozen"] = True
    if _plan["c_trim0"] <= t < _plan["c_trim1"]:
        cover = labkit._part_state(env, labkit._SPECS["cover_plate"])
        err = base.pos[:2] - cover.pos[:2]
        corr = _live["ccorr"] + np.clip(TRACK_GAIN * err, -TRACK_STEP, TRACK_STEP)
        _live["ccorr"] = np.clip(corr, -TRACK_MAX, TRACK_MAX)
        e = _wrap(_yaw(base.rotation()) - _yaw(cover.rotation()))
        _live["cyaw"] = float(np.clip(_live["cyaw"] + np.clip(YAW_GAIN * e, -YAW_STEP, YAW_STEP), -YAW_MAX, YAW_MAX))
    if _plan["k_trim0"] <= t < _plan["k_trim1"]:
        clamp = labkit._part_state(env, labkit._SPECS["clamp"])
        want = _base_to_world(CLAMP_IN - np.array([BACK, 0.0, 0.0]))
        err = want - clamp.pos
        corr = _live["kcorr"] + np.clip(TRACK_GAIN * err, -TRACK_STEP, TRACK_STEP)
        _live["kcorr"] = np.clip(corr, -TRACK_MAX, TRACK_MAX)


class _CoverAt:
    """The left TCP over the base plate (where it is now), plus the cover trim, at height ``z``; ``dx`` along the
    base's length."""

    def __init__(self, z, dx=0.0, trim=True):
        self.z, self.dx, self.trim = z, dx, trim

    def __array__(self, dtype=None, copy=None):
        p = _base_to_world((self.dx, 0.0, 0.0))[:2] + (_live["ccorr"] if self.trim else 0.0)
        return np.asarray((p[0], p[1], self.z), dtype=dtype or float)


class _Press(_CoverAt):
    """Closed fingers pressing on the cover's back half: commanded HOLD_F of compliance into it, leaned so the force is
    vertical."""

    def __array__(self, dtype=None, copy=None):
        p = _base_to_world((PRESS_DX, 0.0, 0.0))
        return np.asarray((p[0] - HOLD_F * C_LEFT[0], p[1] - HOLD_F * C_LEFT[1], TOUCH_Z - HOLD_F * C_LEFT[2]),
                          dtype=dtype or float)


class _ClampAt:
    """The right TCP holding the open clamp in front of the stack: the clamp's frame ``back`` behind its clamped place,
    turned with the base, plus the clamp trim."""

    def __init__(self, back):
        self.back = back

    def __array__(self, dtype=None, copy=None):
        local = CLAMP_IN + np.array([-self.back, 0.0, 0.0]) + HAND_IN_CLAMP
        p = _base_to_world(local) + _live["kcorr"]
        return np.asarray(p, dtype=dtype or float)


_COVER_YAW = Orient(fn=lambda: _prev_grip(_live["psi"] + _live["cyaw"]))
_FORWARD_LIVE = Orient(fn=lambda: _rz(_live["psi"]) @ FORWARD)
_FORWARD = Orient(FORWARD)
_PRESS_YAW = Orient(fn=lambda: _prev_grip(_live["psi"]))  # the left hand's closed fingers on the cover

# Where the right hand squeezes the clamp on the rack: the clamp frame 0.75 mm below its start (the rack's edge centred
# between the lower jaw and the free pad, 11.5 mm apart), and the same with the jaws drawn 30 mm off it.
_k0 = np.array(CLAMP_START) + np.array([0.0, 0.0, -(O_OPEN - PAD_OUT - O_RACK) / 2])
RACK_GRIP = tuple(_k0 + HAND_IN_CLAMP)
RACK_BEHIND = tuple(_k0 + HAND_IN_CLAMP + np.array([-0.030, 0.0, 0.0]))
RACK_OFF = tuple(_k0 + HAND_IN_CLAMP + np.array([-0.035, 0.0, 0.0]))
APPROACH = (RACK_BEHIND[0] - 0.04, RACK_BEHIND[1], RACK_BEHIND[2] + 0.04)
WAIT = (RACK_OFF[0], SY - 0.06, RACK_OFF[2])  # beside the stand, clear of the left hand


def W(x, y, z):
    """The base plate's frame (plate at its start) to world."""
    return (SX + x, SY + y, TOP + STAND_H + z)


PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("stand", (SX + (-PL / 2 + STAND_X1) / 2, SY, TOP + STAND_H / 2), (STAND_X1 + PL / 2, 2 * STAND_HW, STAND_H)),
    ("plates", W(0.0, 0.0, PT), (PL, PW, 2 * PT)),
    ("mat", (*MAT_XY, TOP + MAT[2] / 2), MAT),
    ("rack", (EDGE_X + RACK_L / 2, RACK_Y, TOP + (RACK_Z0 + RACK_T) / 2), (RACK_L, RACK_W, RACK_Z0 + RACK_T)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    _live_reset()
    cx, cy = MAT_XY
    cyaw = _COVER_YAW
    left = [
        Waypoint(2.5, (cx, cy, PICK_HOVER), COVER_WIDE, "above the cover"),
        Waypoint(1.5, (cx, cy, PICK_Z), COVER_WIDE, "pads either side of the cover"),
        Waypoint(0.8, (cx, cy, PICK_Z), COVER_HOLD, "grip the cover by its long sides"),
        Waypoint(1.2, (cx, cy, PICK_HOVER), COVER_HOLD, "lift"),
        Waypoint(2.5, _CoverAt(CARRY_Z), COVER_HOLD, "over the base plate", cyaw),
        Waypoint(1.5, _CoverAt(ENTRY_Z), COVER_HOLD, "cover 2 mm above the base", cyaw),
    ]
    _plan["c_trim0"] = _dur(left)
    left.append(Waypoint(2.0, _CoverAt(ENTRY_Z), COVER_HOLD, "line the cover up on the base", cyaw))
    _plan["c_trim1"] = _dur(left)
    left += [
        Waypoint(1.2, _CoverAt(SET_Z), COVER_HOLD, "set the cover down on the base", cyaw),
        Waypoint(0.5, _CoverAt(SET_Z), COVER_HOLD, "settle", cyaw),
        Waypoint(1.0, _CoverAt(SET_Z), FINGER_OPEN, "let go: open the fingers wide", cyaw),
        Waypoint(0.6, _CoverAt(SET_Z), FINGER_OPEN, "wait for the fingers to clear the cover", cyaw),
        Waypoint(1.0, _CoverAt(OPEN_UP_Z), FINGER_OPEN, "up off the cover", cyaw),
        Waypoint(0.8, _CoverAt(OPEN_UP_Z, PRESS_DX, trim=False), FINGER_CLOSED, "close the fingers over its back half",
                 _PRESS_YAW),
        Waypoint(1.0, _CoverAt(TOUCH_Z + 0.002, PRESS_DX, trim=False), FINGER_CLOSED, "down to just above the cover",
                 _PRESS_YAW),
        Waypoint(1.2, _Press(TOUCH_Z), FINGER_CLOSED, "press the cover straight down", _PRESS_YAW),
        Waypoint(0.5, _Press(TOUCH_Z), FINGER_CLOSED, "settle", _PRESS_YAW),
    ]
    t_held = _dur(left)

    fw = _FORWARD
    right = [
        Waypoint(2.5, APPROACH, SQUEEZE_WIDE, "turn the hand forward, behind the clamp", fw),
        Waypoint(1.5, RACK_BEHIND, SQUEEZE_WIDE, "fingers level with grip and trigger", fw),
        Waypoint(1.5, RACK_GRIP, SQUEEZE_WIDE, "top finger over the grip, bottom finger under the trigger", fw),
        Waypoint(1.0, RACK_GRIP, SQUEEZE, "squeeze the clamp fully open", fw),
        Waypoint(0.5, RACK_GRIP, SQUEEZE, "settle", fw),
        Waypoint(1.5, RACK_OFF, SQUEEZE, "draw the open jaws off the rack", fw),
        Waypoint(2.0, WAIT, SQUEEZE, "carry it beside the stand", fw),
    ]
    right.append(Waypoint(max(0.5, t_held - _dur(right)), WAIT, SQUEEZE, "wait for the cover to be held", fw))
    fl = _FORWARD_LIVE
    right.append(Waypoint(2.0, _ClampAt(BACK), SQUEEZE, "open jaws 30 mm in front of the stack", fl))
    _plan["k_trim0"] = _dur(right)
    right.append(Waypoint(2.0, _ClampAt(BACK), SQUEEZE, "line the jaws up on the stack", fl))
    _plan["k_trim1"] = _dur(right)
    right += [
        Waypoint(3.5, _ClampAt(0.0), SQUEEZE, "slide the jaws over the stack through the notch, slowly", fl),
        Waypoint(0.5, _ClampAt(0.0), SQUEEZE, "settle", fl),
        Waypoint(1.0, _ClampAt(0.0), RELEASE_OPEN, "let go: open the fingers wide", fl),
        Waypoint(0.6, _ClampAt(0.0), RELEASE_OPEN, "wait for the fingers to clear grip and trigger", fl),
        Waypoint(1.5, _ClampAt(0.045), RELEASE_OPEN, "draw the hand straight back", fl),
    ]
    _plan["t_clamped"] = _dur(right) - 2.1
    t_away = _dur(right)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park", fw))
    left.append(Waypoint(t_away - _dur(left), _Press(TOUCH_Z), FINGER_CLOSED,
                         "hold the cover down while the clamp goes on", _PRESS_YAW))
    left += [
        Waypoint(0.8, _CoverAt(TOUCH_Z + 0.003, PRESS_DX, trim=False), FINGER_CLOSED, "lift straight off the cover",
                 _PRESS_YAW),
        Waypoint(1.0, _CoverAt(TOUCH_Z + 0.03, PRESS_DX, trim=False), FINGER_CLOSED, "up", _PRESS_YAW),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold", fw))
    _plan["t_end"] = t_end
    return {"left": left, "right": right}

