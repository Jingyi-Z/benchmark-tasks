"""On the station table: an upright light microscope, its controls as the measured Zeiss Axio Imager.A2m trial stand
(mechanical stage with two leaf clips and a coaxial x/y drive under its front right, turned through a printed top-knob
adapter; coaxial coarse and fine focus), a calibration slide already under the clips, and a camera monitor. The stage
starts 12.5 mm towards the robots and 25.5 mm left of centred, 1 mm below focus.

The task, in the order a microscopist works:
1. Bring the scales' crossing under the objective with the stage's x and y drive knobs (through the adapter).
2. Focus: raise the stage with the right coarse focus knob to near focus, never letting the slide come within 0.5 mm of
   the objective, then finish with the left fine focus knob until the scales are sharp on the camera monitor (a tablet
   on the table) (within 5 um of focus).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
"""

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import VISUAL_ONLY, Joint, on_reset, on_step, part, stl_mesh  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

STAND = (0.90, 0.90, 0.88)
DARK = (0.16, 0.17, 0.19)
GREY = (0.35, 0.36, 0.38)
BLACK = (0.06, 0.06, 0.07)
STEEL = (0.70, 0.71, 0.73)
GLASS = (0.80, 0.90, 0.95)
INK = (0.05, 0.05, 0.06)  # the slide's crossed scales (drawn only: too thin to touch)
SPECIMEN = (0.55, 0.20, 0.45)
SHARP, SOFT, BLUR = (0.04, 0.03, 0.05), (0.22, 0.12, 0.25), (0.40, 0.22, 0.38)  # the mark on the monitor, by focus
VIEW_CHIP, VIEW_GLASS = (0.95, 0.95, 0.90), (0.92, 0.92, 0.86)  # the monitor's bright transmitted-light view: over the
#   scales, or over plain glass
SCREEN = (0.03, 0.03, 0.04)
WHITE = (0.97, 0.97, 0.97)
RED = (0.85, 0.12, 0.12)
LIGHT = (1.0, 0.98, 0.80)  # the illuminator's spot of light
ADAPTER_GREY = (0.24, 0.25, 0.27)  # the printed adapter (dark grey PETG)
ADAPTER_DIR = pathlib.Path(__file__).resolve().parent / "adapter" / "out"
VISUAL_ONLY.add(LIGHT)
for _c in (INK, SHARP, SOFT, BLUR, VIEW_CHIP, VIEW_GLASS):
    VISUAL_ONLY.add(_c)
SPOT_R = 0.012  # the light spot: 24 mm across, on the objective's axis, so it shows round the objective from above

# -- the microscope (fixed). Frame: origin on the table under the stand, x away from the robots, y to their left. -----
MX, MY = 0.34, 0.0  # centred between the robots, so each hand reaches its own focus knob from the side
BASE = (0.26, 0.16, 0.03)
HEAD_TOP = 0.332  # the arm's top: its underside carries the nosepiece, 45 mm above the objective's front lens
PILLAR = (0.07, 0.11, HEAD_TOP - 0.03)
PILLAR_X = 0.085
HEAD = (0.18, 0.06, 0.05)  # the arm over the stage (6 cm wide)
STAGE_X = -0.03  # objective axis (local x); the stage is centred under it when its carriages are at zero
OBJ_R, OBJ_TIP = 0.012, 0.225  # the 10x objective: front lens 22.5 cm above the table (measured on the trial stand)
WD = 0.009  # working distance: in focus when the slide's top is 9.0 mm below the front lens (measured: EC
#   Epiplan-Neofluar 10x/0.25 HD on the trial stand)
START_OFF = 0.001  # the stage starts 1 mm below focus (a quarter turn of the coarse knob)
SLIDE_W, SLIDE_L, SLIDE_T = 0.026, 0.076, 0.002  # x by y by thickness (drawn 2 mm thick; a real slide is 1 mm)
SPEC_Y = 0.0  # the crossed scales' crossing (the alignment mark) is at the slide's centre; the label on the left end
PLATE_TOP0 = OBJ_TIP - WD - START_OFF - SLIDE_T  # top of the stage (x carriage) at the start
STAGE_MAX = START_OFF + WD - 0.0002  # the focus stops just short of the slide touching the objective

# Focus drive housing under the stage, with one knob shaft through it along y.
HOUSING = (0.05, 0.16, 0.07)  # the focus mechanism, 3 to 10 cm above the table (50 mm deep, 16 cm wide: an
#   assumption; the coarse knobs sit on its sides)
FKNOB_X, FKNOB_Z = -0.04, 0.07  # the focus shaft's axis, 7 cm above the table (measured)
FKNOB_R, FKNOB_W = 0.030, 0.0226  # coarse knobs 60 mm across, 22.6 mm wide (measured)
FKNOB_Y0 = HOUSING[1] / 2
F_LEAD = 0.004  # coarse focus: m of stage travel per turn (measured: 4 mm per turn; the right knob turned clockwise,
#   seen from the right, lowers the stage)
F_MIN = -2 * math.pi * STAGE_MAX / F_LEAD
# Fine focus on the same shaft: the left one a knob 35 mm across and 25 mm long outboard of the left coarse knob (the
# robot's left hand turns it), the right one a 50 mm graduated disc set in the right coarse knob's face.
FINE_LEAD = 0.004 / 12  # m per turn (measured: one fine turn moves the coarse knob 30 degrees, 0.333 mm)
FINE_R, FINE_W = 0.0175, 0.025
DISC_R, DISC_W = 0.025, 0.004
FINE_Y0 = FKNOB_Y0 + FKNOB_W + 0.001  # inner face of the left fine knob (y)
FINE_LIM = 4 * math.pi

# Stage travel: the x carriage starts 85 mm towards the robots (loading position), the y carriage centred.
QX0, QY0 = -0.0125, 0.025  # the stage starts 12.5 mm towards the robots and 25 mm left of centred, the slide loaded
QX_FOCUS, QY_FOCUS = 0.0, -SPEC_Y  # carriage positions that put the specimen on the objective axis
X_LEAD, Y_LEAD = 0.050 / 1.5, 0.076 / 1.75  # stage drive, m per knob turn, measured on the trial stand: the upper knob
#   (x, front to back) 540 degrees for the full 50 mm travel (33.3 mm/turn), the lower knob (y, left to right) 630 degrees
#   for the full 76 mm (43.4 mm/turn). Turned clockwise seen from above, the upper knob moves the stage towards the user
#   (the robots, -x) and the lower knob moves it to the left (+y). Backlash: almost none.
# The stage knobs are the top knobs of the printed belt adapter (adapter/knob_adapter.py, the one geometry for both the
# real adapter and this scene). Its frame: the stand's coaxial stage control's axis, 5 mm behind the lower stage plate's
# front edge and 15 mm inboard of its right edge, at the stage's top surface; u towards the robots (-x), v outboard (-y).
ADAPTER_AXIS = (STAGE_X - 0.060 + 0.005, -0.075 + 0.015)
ADAPTER_SHAFTS = ((0.050, 0.05724), (0.050, 0.12648))  # (u, v) of the x and y knobs' shafts (64-tooth pulleys, belts
#   280 and 400 mm: centre distances 76 and 136 mm)


def _adapter_xy(u, v):
    return (ADAPTER_AXIS[0] - u, ADAPTER_AXIS[1] - v)


XKNOB = _adapter_xy(*ADAPTER_SHAFTS[0])  # x (inner) and y (outer) stage knobs, 69.2 mm apart (local xy)
YKNOB = _adapter_xy(*ADAPTER_SHAFTS[1])
DKNOB_R, DKNOB_H = 0.017, 0.016
DKNOB_TOP = PLATE_TOP0 - 0.020  # the adapter's knob tops, 20 mm BELOW the stage top (they rise with the stage): a
#   straight-down grip reaches them there (yaw 0 to 90 degrees, yam_arena IK) but not above the stage top
LIM = 0.10
LIM_X, LIM_Y = (-0.025, 0.025), 0.038  # the stage's travel: 50 mm front to back, 76 mm left to right (measured)

# Slide retention on the x carriage: two leaf clips (spring steel, 57 x 20.3 x 0.25 mm) on posts 5.8 mm across whose tops
# are 5.1 mm above the plate, as on the trial stand, placed behind the slide so that each leaf lies over one end of it
# (the model's plate is smaller than the stand's 16 x 28 cm, so the posts sit closer in). The slide is already loaded.
PUSHED_DY = 0.0  # the loaded slide's centre is the holder centre
CLIP_L, CLIP_W, CLIP_T = 0.057, 0.0203, 0.00025
POST_R, POST_TOP = 0.0029, 0.0051
CLIP_Y = 0.030  # the leaves' centre lines, 8 mm inside each end of the 76 mm slide
CLIP_POST_X = STAGE_X + SLIDE_W / 2 + 0.032  # the posts, 32 mm behind the slide's back edge
CLIP_TIP_X = CLIP_POST_X - CLIP_L  # each leaf's free end, 1 mm short of the slide's front edge
CLIP_TIP_Z = PLATE_TOP0 + SLIDE_T + 0.0001  # a leaf's underside at its tip: 0.1 mm over the slide's top
CLIP_PITCH = -math.atan2(POST_TOP - (SLIDE_T + 0.0001 + CLIP_T), CLIP_L)  # the leaf rises to its post


def _ribs(cx, cy, z0, r, h, n=12):
    return [((cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n), z0 + h / 2), 0.0025, h,
             BLACK) for i in range(n)]


FOCUS = Joint(
    "focus", kind="hinge", axis="Y", anchor=(FKNOB_X, 0.0, FKNOB_Z), lower=F_MIN - 0.3, upper=0.3, initial=0.0,
    stiffness=0.0, damping=0.3, target=0.0, mass=0.05,
    cylinders=[
        ((FKNOB_X, 0.0, FKNOB_Z), 0.006, 2 * FKNOB_Y0 + 2 * FKNOB_W - 0.002, STEEL, "Y"),
        ((FKNOB_X, FKNOB_Y0 + FKNOB_W / 2, FKNOB_Z), FKNOB_R, FKNOB_W, BLACK, "Y"),
        ((FKNOB_X, -(FKNOB_Y0 + FKNOB_W / 2), FKNOB_Z), FKNOB_R, FKNOB_W, BLACK, "Y"),
        *[((FKNOB_X + FKNOB_R * math.cos(i * math.pi / 6), s * (FKNOB_Y0 + FKNOB_W / 2),
            FKNOB_Z + FKNOB_R * math.sin(i * math.pi / 6)), 0.0025, FKNOB_W, BLACK, "Y")
          for s in (1, -1) for i in range(12)],
    ],
    boxes=[((FKNOB_X, s * (FKNOB_Y0 + FKNOB_W + 0.0005), FKNOB_Z + FKNOB_R / 2), (0.003, 0.001, FKNOB_R), WHITE)
           for s in (1, -1)],
)
FINE = Joint(
    "fine", kind="hinge", axis="Y", anchor=(FKNOB_X, 0.0, FKNOB_Z), lower=-FINE_LIM, upper=FINE_LIM, initial=0.0,
    stiffness=0.0, damping=0.05, target=0.0, mass=0.02,
    cylinders=[
        ((FKNOB_X, FINE_Y0 + FINE_W / 2, FKNOB_Z), FINE_R, FINE_W, BLACK, "Y"),  # the left fine knob
        *[((FKNOB_X + FINE_R * math.cos(i * math.pi / 8), FINE_Y0 + FINE_W / 2,
            FKNOB_Z + FINE_R * math.sin(i * math.pi / 8)), 0.0015, FINE_W, BLACK, "Y") for i in range(16)],
        # the right fine disc, in the right coarse knob's face (4 mm proud of it)
        ((FKNOB_X, -(FKNOB_Y0 + FKNOB_W + DISC_W / 2 + 0.0005), FKNOB_Z), DISC_R, DISC_W, GREY, "Y"),
    ],
    # graduations: a white index line on the left fine knob's face and on the right disc
    boxes=[((FKNOB_X, FINE_Y0 + FINE_W + 0.0005, FKNOB_Z + FINE_R / 2), (0.002, 0.001, FINE_R), WHITE),
           ((FKNOB_X, -(FKNOB_Y0 + FKNOB_W + DISC_W + 0.001), FKNOB_Z + DISC_R / 2), (0.002, 0.001, DISC_R), WHITE)],
)
STAGE = Joint(  # the focus slide: carries the stage, the drive bracket and knobs
    "stage", kind="slide", axis="Z", anchor=(STAGE_X, 0.0, PLATE_TOP0), lower=0.0, upper=STAGE_MAX, initial=0.0,
    stiffness=20000.0, damping=400.0, target=0.0, mass=0.05,
    boxes=[
        ((STAGE_X + 0.075, 0.0, PLATE_TOP0 - 0.03), (0.03, 0.07, 0.03), DARK),  # carrier, into the pillar
        ((STAGE_X, 0.0, PLATE_TOP0 - 0.0135), (0.12, 0.15, 0.005), DARK),  # fixed lower plate
        # the knob adapter (colliding stand-ins for its drawn mesh): the arm, z -53 to -49 mm under the stage top, as a
        # strip from behind the coaxial knobs to the shafts' line (u -75 to 66 mm, v 20 to 34) and a bar along that
        # line (u 34 to 66, v 20 to 142.5); the wall on the lower plate's right edge face (u -75 to 5, v 16 to 20,
        # z -17 to -11.5); the drop web from the wall down to the arm (u -75 to -65, v 16 to 34)
        ((ADAPTER_AXIS[0] + 0.0045, ADAPTER_AXIS[1] - 0.027, PLATE_TOP0 - 0.051), (0.141, 0.014, 0.004), ADAPTER_GREY),
        ((ADAPTER_AXIS[0] - 0.050, ADAPTER_AXIS[1] - 0.08125, PLATE_TOP0 - 0.051), (0.032, 0.1225, 0.004), ADAPTER_GREY),
        ((ADAPTER_AXIS[0] + 0.035, ADAPTER_AXIS[1] - 0.018, PLATE_TOP0 - 0.01425), (0.080, 0.004, 0.0055), ADAPTER_GREY),
        ((ADAPTER_AXIS[0] + 0.070, ADAPTER_AXIS[1] - 0.025, PLATE_TOP0 - 0.03225), (0.010, 0.018, 0.0415), ADAPTER_GREY),
    ],
    # The illuminator's light, from below through the stage, shows as a spot on the slide on the objective's axis. It
    # rises with the stage (the condenser does) and is only drawn: nothing collides with it.
    # It is drawn just over the glass and under the specimen square's top, so the square shows on the light.
    cylinders=[((STAGE_X, 0.0, PLATE_TOP0 + SLIDE_T + 0.0001), SPOT_R, 0.0001, LIGHT),
               # the adapter's bearing bosses, from the arm's top (49 mm under the stage top) up to the knobs
               *[((kx, ky, (PLATE_TOP0 - 0.049 + DKNOB_TOP - DKNOB_H) / 2), 0.012,
                  DKNOB_TOP - DKNOB_H - (PLATE_TOP0 - 0.049), ADAPTER_GREY) for kx, ky in (XKNOB, YKNOB)]],
    # the adapter's own parts as drawn (adapter/out/asm_*.stl, in its frame: millimetres, u and v turned to -x and -y)
    meshes=[stl_mesh(ADAPTER_DIR / f"asm_{n}.stl", c, pos=(*ADAPTER_AXIS, PLATE_TOP0), rpy=(0.0, 0.0, math.pi))
            for n, c in (("bracket", ADAPTER_GREY), ("sleeve_x", ADAPTER_GREY), ("sleeve_y", ADAPTER_GREY),
                         ("pulley_x", ADAPTER_GREY), ("pulley_y", ADAPTER_GREY), ("belt_x", BLACK), ("belt_y", BLACK))],
)
STAGE_Y = Joint(
    "stage_y", kind="slide", axis="Y", anchor=(STAGE_X, 0.0, PLATE_TOP0), lower=-LIM_Y, upper=LIM_Y, initial=QY0,
    stiffness=20000.0, damping=400.0, target=QY0, mass=0.05, parent="stage",
    boxes=[((STAGE_X, 0.0, PLATE_TOP0 - 0.0085), (0.11, 0.14, 0.005), GREY)],
)
STAGE_X_J = Joint(
    "stage_x", kind="slide", axis="X", anchor=(STAGE_X, 0.0, PLATE_TOP0), lower=LIM_X[0], upper=LIM_X[1], initial=QX0,
    stiffness=20000.0, damping=400.0, target=QX0, mass=0.05, parent="stage_y",
    boxes=[
        ((STAGE_X, 0.0, PLATE_TOP0 - 0.003), (0.10, 0.13, 0.006), DARK),  # top plate
        # the two leaf clips, each rising from just over the slide's top at its free end to its post's top
        *[((0.5 * (CLIP_TIP_X + CLIP_POST_X), s * CLIP_Y, 0.5 * (CLIP_TIP_Z + PLATE_TOP0 + POST_TOP)),
           (CLIP_L, CLIP_W, CLIP_T), STEEL, (0.0, CLIP_PITCH)) for s in (1, -1)],
        ((STAGE_X - 0.02, 0.0, PLATE_TOP0 + 0.0002), (0.002, 0.03, 0.0004), WHITE),  # holder index line
    ],
    cylinders=[((CLIP_POST_X, s * CLIP_Y, PLATE_TOP0 + POST_TOP / 2), POST_R, POST_TOP, STEEL) for s in (1, -1)],
)


def _drive_knob(name, xy):
    cx, cy = xy
    return Joint(
        name, kind="hinge", axis="Z", anchor=(cx, cy, DKNOB_TOP - DKNOB_H), lower=-4 * math.pi, upper=4 * math.pi,
        initial=0.0, stiffness=0.0, damping=0.1, target=0.0, mass=0.02, parent="stage",
        cylinders=[((cx, cy, DKNOB_TOP - DKNOB_H / 2), DKNOB_R, DKNOB_H, BLACK),
                   *_ribs(cx, cy, DKNOB_TOP - DKNOB_H, DKNOB_R, DKNOB_H)],
        boxes=[((cx - DKNOB_R / 2, cy, DKNOB_TOP + 0.0004), (DKNOB_R, 0.002, 0.0008), WHITE)],
    )


XDRIVE, YDRIVE = _drive_knob("xdrive", XKNOB), _drive_knob("ydrive", YKNOB)

MICROSCOPE = part(
    "microscope", pos=(MX, MY, TOP), fixed=True, friction=1.0,
    boxes=[
        ((0.0, 0.0, BASE[2] / 2), BASE, STAND),
        ((PILLAR_X, 0.0, BASE[2] + PILLAR[2] / 2), PILLAR, STAND),
        ((PILLAR_X + PILLAR[0] / 2 - HEAD[0] / 2, 0.0, HEAD_TOP - HEAD[2] / 2), HEAD, STAND),
        ((FKNOB_X, 0.0, 0.03 + HOUSING[2] / 2), HOUSING, STAND),
        # the binocular head on the arm over the objective, swivelled to face the back (away from the robots, whose right
        # arm reaches past it to the stage knobs), its eyepiece tubes inclined 45 degrees
        ((STAGE_X + 0.005, 0.0, HEAD_TOP + 0.020), (0.070, 0.060, 0.040), STAND),
        *[((STAGE_X + 0.0577, sy * 0.032, HEAD_TOP + 0.0577), (0.050, 0.022, 0.022), DARK, (0.0, -math.pi / 4))
          for sy in (1, -1)],
        ((STAGE_X + 0.0812, -0.032, HEAD_TOP + 0.0812), (0.012, 0.026, 0.026), BLACK, (0.0, -math.pi / 4)),  # eyecup
        ((STAGE_X + 0.0812, 0.032, HEAD_TOP + 0.0812), (0.012, 0.026, 0.026), BLACK, (0.0, -math.pi / 4)),  # eyecup
        # the camera port on top of the head and the Zeiss Axiocam 506 color on its 0.5x C-mount adapter
        ((STAGE_X + 0.005, 0.0, HEAD_TOP + 0.050), (0.030, 0.030, 0.020), STAND),
        ((STAGE_X + 0.005, 0.0, HEAD_TOP + 0.090), (0.068, 0.068, 0.060), DARK),
    ],
    cylinders=[
        ((STAGE_X, 0.0, HEAD_TOP - HEAD[2] - 0.006), 0.032, 0.012, STEEL),  # nosepiece
        ((STAGE_X, 0.0, (OBJ_TIP + HEAD_TOP - HEAD[2] - 0.012) / 2), OBJ_R, HEAD_TOP - HEAD[2] - 0.012 - OBJ_TIP, STEEL),
        ((STAGE_X, 0.0, OBJ_TIP + 0.004), OBJ_R + 0.0005, 0.002, RED),  # the objective's colour ring
    ],
    joints=[FOCUS, FINE, STAGE, STAGE_Y, STAGE_X_J, XDRIVE, YDRIVE],
)

# -- the slide (loose) on its rest -----------------------------------------------------------------------------------
SLIDE = part(
    "slide", pos=(MX + STAGE_X + QX0, MY + QY0 + PUSHED_DY, TOP + PLATE_TOP0), mass=0.006, friction=0.6,
    boxes=[
        ((0.0, 0.0, SLIDE_T / 2), (SLIDE_W, SLIDE_L, SLIDE_T), GLASS),
        # the crossed scales (MAGUS CS5): two 1 mm scales crossing at the slide's centre, each a 10 um line with 100
        # divisions of 0.01 mm (drawn: the two lines, 20 um wide, without their ticks)
        ((0.0, SPEC_Y, SLIDE_T + 0.00005), (0.001, 0.00002, 0.0001), INK),
        ((0.0, SPEC_Y, SLIDE_T + 0.00005), (0.00002, 0.001, 0.0001), INK),
        ((0.0, 0.025, SLIDE_T + 0.0002), (0.024, 0.020, 0.0004), WHITE),  # label
    ],
)

# -- the camera monitor (fixed): a tablet lying flat on the table in front of the microscope, between the robots, its
# screen up, showing the microscope camera's live 10x view, 2.50 x 2.00 mm of the slide (the Axiocam 506's 5:4 image),
# over a 12.75 x 10.2 cm picture in the middle of its screen (51 mm on the screen per mm of the slide). The CS5's two
# 1 mm scales show in full: two crossing 1 mm lines (51 mm on the screen) with their divisions, moved with the stage
# and drawn sharp, soft or blurred by how far the slide is from focus (within 5 um, 0.1 mm, 1 mm; beyond that they are
# lost in the blur). The screen's centre is the objective's axis; away from the robots on the slide is away on the
# screen (its far edge is up in the robots' view), and to their left is to the left.
MON_XY = (0.09, 0.0)  # the screen's centre on the table
MON_W, MON_H, MON_T = 0.18, 0.102, 0.012  # the tablet's screen: across (y), along (x), thickness
VIEW_W = MON_H * 2.50 / 2.00  # the camera picture's width across (y): 12.75 cm, the image's 5:4
SCR_PER_M = VIEW_W / 0.0025  # 51 mm on the screen per mm of the slide
MARK_LEVELS = ((SHARP, 0.0015), (SOFT, 0.004), (BLUR, 0.010))  # colour, line width on the screen
_face = MON_T + 0.0010  # the screen's face, on top of the tablet
MONITOR_BOXES = [
    ((0.0, 0.0, MON_T / 2), (MON_H + 0.012, MON_W + 0.012, MON_T), DARK),  # tablet body and bezel
    ((0.0, 0.0, MON_T + 0.0005), (MON_H, MON_W, 0.001), SCREEN),
    # layers, bottom to top: the chip view, the marks, then the bare-glass view on top, so that the start state (bare
    # glass) is what shows before the first step hides the layers not in use
    ((0.0, 0.0, _face + 0.0010), (MON_H, VIEW_W, 0.0003), VIEW_GLASS),  # the view over bare glass
    ((0.0, 0.0, _face + 0.0002), (MON_H, VIEW_W, 0.0003), VIEW_CHIP),  # the view over the scales
]
VIEW_BOX0 = 2
# Each scale line is drawn as SEG_N short segments (0.05 mm of the slide each), so that the part off the picture can be
# hidden; all are built at the screen's centre (under the top view layer) and placed when drawn.
SEG = 0.00005 * SCR_PER_M
SEG_K = range(-10, 10)  # segment k covers k to k + 1 times 0.05 mm from the crossing
MARK_BOX0 = len(MONITOR_BOXES)
SEGS = []  # (segment k, on the line along "y" or "x"), in box order within a level
for _col, _w in MARK_LEVELS:
    SEGS = []
    for _k in SEG_K:
        MONITOR_BOXES += [((0.0, 0.0, _face + 0.0006), (_w, SEG + 0.0002, 0.0003), _col),   # the line along y
                          ((0.0, 0.0, _face + 0.0006), (SEG + 0.0002, _w, 0.0003), _col)]   # the line along x
        SEGS += [(_k, "y"), (_k, "x")]
MARK_PER_LEVEL = 2 * len(SEG_K)
TICK = 0.00001 * SCR_PER_M  # one 0.01 mm division on the screen (0.51 mm)
# Divisions drawn: every tenth one (0.1 mm, 5.1 mm on the screen), longer at the ends and the centre. The 0.01 mm
# divisions themselves are 0.51 mm apart on the screen, finer than the drawn line; the real camera resolves them
# (0.9 um of slide per pixel).
TICK_K = range(-50, 51, 10)
TICK_BOX0 = len(MONITOR_BOXES)  # then, per level, a tick across each line at each drawn division
TICKS = []  # (division k, on the line along "y" or "x"), in box order within a level
for _col, _w in MARK_LEVELS:
    TICKS = []
    for _k in TICK_K:
        _len = 0.006 if _k % 50 == 0 else 0.0035
        MONITOR_BOXES += [((0.0, 0.0, _face + 0.0006), (_len, _w, 0.0003), _col),   # on the line along y
                          ((0.0, 0.0, _face + 0.0006), (_w, _len, 0.0003), _col)]   # on the line along x
        TICKS += [(_k, "y"), (_k, "x")]
TICK_PER_LEVEL = 2 * len(TICK_K)
MONITOR = part("monitor", pos=(*MON_XY, TOP), fixed=True, boxes=MONITOR_BOXES)

OBJECTS = [MICROSCOPE, SLIDE, MONITOR]


# -- drives: the stage follows its knobs ------------------------------------------------------------------------------
def carriages(j: dict) -> tuple[float, float, float]:
    """(x, y, z) stage positions the knobs ask for: z from the coarse and fine focus knobs, x and y from the drive
    knobs."""
    qz = float(np.clip(-(F_LEAD * j["focus"] + FINE_LEAD * j["fine"]) / (2 * math.pi), 0.0, STAGE_MAX))
    # (a knob angle is positive counterclockwise seen from above: clockwise moves x towards the robots, y to the left)
    qx = float(np.clip(QX0 + X_LEAD * j["xdrive"] / (2 * math.pi), *LIM_X))
    qy = float(np.clip(QY0 - Y_LEAD * j["ydrive"] / (2 * math.pi), -LIM_Y, LIM_Y))
    return qx, qy, qz


def stage_gap(qz: float) -> float:
    """Clearance (m) from a loaded slide's top to the objective's front lens."""
    return OBJ_TIP - (PLATE_TOP0 + SLIDE_T + qz)


_trial = {"min_gap": None}
_scr = {"prims": None, "shown": None}


@on_reset
def _clear(env):
    _trial["min_gap"] = None
    _trial["n"] = 0
    _trial["far"], _trial["still"], _trial["hist"] = False, False, []
    _scr["shown"] = None
    # the start state on the monitor: bare glass, no mark (the slide is on its rest, not under the objective); without
    # this the screen shows every layer until the first step redraws it, which put the chip view in the setup image
    _show_screen(env, ("glass", None, (0.0, 0.0)))


def _screen_prims(env):
    """The monitor's two view layers and mark boxes as USD prims, with each mark box's resting translation."""
    if _scr["prims"] is None:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        path = env.unwrapped.scene["monitor"].cfg.prim_path.replace("env_[^/]+", "env_0").replace("env_.*", "env_0")
        body = f"{path}/body"
        marks = []
        for lv in range(len(MARK_LEVELS)):
            row = []
            for k in range(MARK_PER_LEVEL):
                prim = stage.GetPrimAtPath(f"{body}/box_{MARK_BOX0 + lv * MARK_PER_LEVEL + k}")
                op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
                row.append((prim, op, op.Get()))
            marks.append(row)
        ticks = []
        for lv in range(len(MARK_LEVELS)):
            row = []
            for k in range(TICK_PER_LEVEL):
                prim = stage.GetPrimAtPath(f"{body}/box_{TICK_BOX0 + lv * TICK_PER_LEVEL + k}")
                op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
                row.append((prim, op, op.Get()))
            ticks.append(row)
        _scr["prims"] = {"glass": stage.GetPrimAtPath(f"{body}/box_{VIEW_BOX0}"),
                         "chip": stage.GetPrimAtPath(f"{body}/box_{VIEW_BOX0 + 1}"),
                         "marks": marks, "ticks": ticks}
    return _scr["prims"]


def screen_view(j: dict, slide):
    """What the monitor shows: (field: 'chip' or 'glass', mark level 0 sharp / 1 soft / 2 blurred / None, mark offset
    on the screen (along x, across y) in m)."""
    if not _loaded(j, slide):
        return "glass", None, (0.0, 0.0)
    p = slide.to_world((0.0, SPEC_Y, 0.0))
    dx, dy = p[0] - OBJ_AXIS[0], p[1] - OBJ_AXIS[1]
    field = "chip" if mark_in_view(dx, dy) else "glass"
    if field != "chip" or not mark_in_view(dx, dy):
        return field, None, (0.0, 0.0)
    e = focus_error(j)
    level = 0 if e <= FOCUS_TOL else 1 if e <= NEAR_TOL else 2 if e <= 0.001 else None
    return field, level, (round(dx * SCR_PER_M, 3), round(dy * SCR_PER_M, 3))


def _show_screen(env, view):
    """Draw the monitor's view (only when it changes): the field's colour, and one level of the mark, moved."""
    if view == _scr["shown"]:
        return
    _scr["shown"] = view
    try:
        from pxr import Gf, UsdGeom

        pr = _screen_prims(env)
        field, level, (ox, oy) = view
        for name in ("glass", "chip"):
            img = UsdGeom.Imageable(pr[name])
            img.MakeVisible() if name == field else img.MakeInvisible()
        for lv, row in enumerate(pr["marks"]):
            for (prim, op, base), (k, line) in zip(row, SEGS):
                img = UsdGeom.Imageable(prim)
                c = (k + 0.5) * SEG
                x, y = base[0] + ox + (c if line == "x" else 0.0), base[1] + oy + (c if line == "y" else 0.0)
                if lv == level and abs(x) <= MON_H / 2 - SEG / 2 and abs(y) <= VIEW_W / 2 - SEG / 2:
                    op.Set(Gf.Vec3d(x, y, base[2]))
                    img.MakeVisible()
                else:
                    img.MakeInvisible()
        for lv, row in enumerate(pr["ticks"]):
            for (prim, op, base), (k, line) in zip(row, TICKS):
                img = UsdGeom.Imageable(prim)
                x, y = base[0] + ox + (k * TICK if line == "x" else 0.0), base[1] + oy + (k * TICK if line == "y" else 0.0)
                if lv == level and abs(x) <= MON_H / 2 - 0.003 and abs(y) <= VIEW_W / 2 - 0.003:
                    op.Set(Gf.Vec3d(x, y, base[2]))
                    img.MakeVisible()
                else:
                    img.MakeInvisible()
    except Exception:
        pass


# -- released hands and stillness (every positive score) --------------------------------------------------------------
RELEASE = 0.080  # every jaw point and pad midpoint at least this far from every task surface
STILL_N, STILL_TOL = 21, 0.001  # and the slide and stage moved under 1 mm over the last 21 control steps (1 s at 20 Hz)


def _tips(env):
    """Both hands' pad midpoints (the runtime's own pad datum) and their two jaw points (world, env frame)."""
    from yam_arena import runtime as _rt

    b = env.unwrapped
    o = b.scene.env_origins[0].cpu().numpy()
    out = []
    for n in ("left_robot", "right_robot"):
        robot = b.scene[n]
        mid = _rt.pad_tips_world(robot) - o
        g = _rt.link_poses(robot)["gripper"]
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        q7, q8 = (abs(float(v)) for v in robot.data.joint_pos.torch[0, ids].cpu().numpy())
        out += [mid, mid - q7 * g[:3, 0], mid + q8 * g[:3, 0]]
    return out


def _fingers(env) -> str:
    """Debug: each hand's two finger joints (m), left then right."""
    out = []
    for n in ("left_robot", "right_robot"):
        robot = env.unwrapped.scene[n]
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        out.append("/".join(f"{float(v):.3f}" for v in robot.data.joint_pos.torch[0, ids].cpu().numpy()))
    return " ".join(out)


def _task_boxes(j: dict, slide):
    """World axis-aligned boxes (centre, half sizes) of every task surface: the microscope's fixed body and cylinders,
    its stage (with the leaf clips) and knobs where the joints now put them, the slide and the monitor."""
    org = np.array([MX, MY, TOP])
    qx, qy, qz = j["stage_x"], j["stage_y"], j["stage"]
    out = []
    for c, sz, *_ in MICROSCOPE.boxes:
        out.append((org + np.array(c, float), np.array(sz, float) / 2))
    for c, r, h, *rest in MICROSCOPE.cylinders:
        ax = rest[1] if len(rest) > 1 else "Z"
        half = [r, r, h / 2] if ax == "Z" else ([r, h / 2, r] if ax == "Y" else [h / 2, r, r])
        out.append((org + np.array(c, float), np.array(half, float)))
    moves = {"stage": (0.0, 0.0, qz), "stage_y": (0.0, qy, qz), "stage_x": (qx, qy, qz),
             "xdrive": (0.0, 0.0, qz), "ydrive": (0.0, 0.0, qz), "focus": (0.0, 0.0, 0.0), "fine": (0.0, 0.0, 0.0)}
    for J in MICROSCOPE.joints:
        d = np.array(moves.get(J.name, (0.0, 0.0, 0.0)), float) - (np.array([QX0, QY0, 0.0]) if J.name == "stage_x" else np.array([0.0, QY0, 0.0]) if J.name == "stage_y" else 0.0)
        for c, sz, *_ in (J.boxes or []):
            out.append((org + np.array(c, float) + d, np.array(sz, float) / 2))
        for c, r, h, *rest in (J.cylinders or []):
            ax = rest[1] if len(rest) > 1 else "Z"
            half = [r, r, h / 2] if ax == "Z" else ([r, h / 2, r] if ax == "Y" else [h / 2, r, r])
            out.append((org + np.array(c, float) + d, np.array(half, float)))
    out.append((np.asarray(slide.to_world((0.0, 0.0, SLIDE_T / 2)), float), np.array([SLIDE_W, SLIDE_L, SLIDE_T]) / 2))
    out.append((np.array([MON_XY[0], MON_XY[1], TOP + MON_T / 2]), np.array([MON_H + 0.012, MON_W + 0.012, MON_T]) / 2))
    return out


def _moving_points(j: dict, slide) -> np.ndarray:
    """Material points on every moving task part, in 3-D (m): three on the slide, one on the stage's x carriage (it
    carries the y and focus motions too; the leaf clips ride on it), and a rim point on each knob (coarse, fine and both
    stage knobs), each at its knob's own radius and angle."""
    pts = [np.asarray(slide.to_world(p), float) for p in ((0.0, -SLIDE_L / 2, 0.0), (0.0, SLIDE_L / 2, 0.0),
                                                           (SLIDE_W / 2, 0.0, 0.0))]
    qx, qy, qz = j["stage_x"], j["stage_y"], j["stage"]
    pts.append(np.array([qx, qy, qz]))
    for name, r, lift in (("focus", FKNOB_R, 0.0), ("fine", FINE_R, 0.0), ("xdrive", DKNOB_R, qz), ("ydrive", DKNOB_R, qz)):
        a = j[name]
        pts.append(np.array([r * math.cos(a), r * math.sin(a), lift]))
    return np.array(pts)


def _max_travel(hist) -> float:
    """The largest distance any one tracked point moved between any two of the recorded samples (m)."""
    h = np.asarray(hist)  # samples x points x 3
    d = np.linalg.norm(h[:, None, :, :] - h[None, :, :, :], axis=3)
    return float(d.max())


def _clearance(j: dict, slide, tips) -> float:
    best = 9.0
    for c, h in _task_boxes(j, slide):
        for p in tips:
            d = np.maximum(np.abs(np.asarray(p, float) - c) - h, 0.0)
            best = min(best, float(np.linalg.norm(d)))
    return best


@on_step
def _drive(env):
    import torch

    import labkit

    s = labkit._part_state(env, labkit._SPECS["microscope"])
    sl = labkit._part_state(env, labkit._SPECS["slide"])
    g = stage_gap(s.joints["stage"])
    if _loaded(s.joints, sl):
        _trial["min_gap"] = g if _trial["min_gap"] is None else min(_trial["min_gap"], g)
    if os.environ.get("MICRO_DEBUG"):
        _trial["n"] = _trial.get("n", 0) + 1
        if _trial["n"] % 20 == 0:
            dx, dy = specimen_xy(sl)
            print(f"[micro] t={_trial['n'] / 20:5.1f}s loaded={_loaded(s.joints, sl)} mark=({dx * 1000:.3f}, "
                  f"{dy * 1000:.3f}) mm focus_err={focus_error(s.joints) * 1e6:.1f} um "
                  f"min_gap={(_trial['min_gap'] or 0) * 1000:.2f} mm knobs x={s.joints['xdrive']:.3f} "
                  f"y={s.joints['ydrive']:.3f} coarse={s.joints['focus']:.3f} fine={s.joints['fine']:.3f} "
                  f"fingers {_fingers(env)}", file=sys.stderr, flush=True)
    asset = env.unwrapped.scene["microscope"]
    names = list(asset.joint_names)
    ids = [names.index(j.joint_name) for j in (STAGE_X_J, STAGE_Y, STAGE)]
    qx, qy, qz = carriages(s.joints)
    asset.set_joint_position_target(torch.tensor([[qx, qy, qz]], device=asset.device), joint_ids=ids)
    _show_screen(env, screen_view(s.joints, sl))
    # the release and stillness gate's state, each control step
    clear = _clearance(s.joints, sl, _tips(env))
    _trial["far"] = clear >= RELEASE
    hist = _trial.setdefault("hist", [])
    hist.append(_moving_points(s.joints, sl))
    del hist[:-STILL_N]
    _trial["still"] = len(hist) == STILL_N and _max_travel(hist) < STILL_TOL
    if os.environ.get("MICRO_DEBUG") and _trial.get("n", 0) % 20 == 0 and len(hist) == STILL_N:
        h = np.asarray(hist)
        per = np.linalg.norm(h[:, None] - h[None, :], axis=3).max(axis=(0, 1)) * 1000
        print(f"[gate] clear={clear * 1000:.1f} mm far={_trial['far']} still={_trial['still']} travel_mm="
              + ",".join(f"{v:.2f}" for v in per),
              file=sys.stderr, flush=True)


# -- scoring ----------------------------------------------------------------------------------------------------------
OBJ_AXIS = (MX + STAGE_X, MY)
HOLD_TOL, HOLD_Z, FLAT_DEG = 0.003, 0.001, 3.0
# The camera's field of view is what the monitor shows: the Zeiss Axiocam 506 color's 12.5 x 10.0 mm sensor through the
# 10x objective and a 0.5x camera adapter (5x onto the sensor), 2.50 mm left to right (y) by 2.00 mm towards and away
# from the robots (x) of the slide, centred on the objective's axis. The crossing is in view when it is on the screen.
FIELD_X, FIELD_Y, MARK_HALF = 0.0010, 0.00125, 0.0
CENTRE_TOL = 0.00005  # full credit: the crossing within 0.05 mm of the objective's axis (the image centre)


def mark_in_view(dx: float, dy: float) -> bool:
    return abs(dx) <= FIELD_X - MARK_HALF and abs(dy) <= FIELD_Y - MARK_HALF
CRASH_GAP = 0.0005
FOCUS_TOL, NEAR_TOL = 0.000005, 0.0001  # depth of field of a 10x / 0.25 objective: about 10 um (+-5 um)


def holder_centre(j: dict) -> np.ndarray:
    """World position of the holder centre (the middle of a loaded slide's underside)."""
    return np.array([MX + STAGE_X + j["stage_x"], MY + PUSHED_DY + j["stage_y"], TOP + PLATE_TOP0 + j["stage"]])


def _loaded(j: dict, slide) -> bool:
    d = slide.pos - holder_centre(j)
    return (math.hypot(d[0], d[1]) <= HOLD_TOL and abs(d[2]) <= HOLD_Z and slide.tilt_deg() <= FLAT_DEG
            and slide.to_world((0, 1, 0))[1] - slide.pos[1] > 0.99)


def specimen_xy(slide):
    p = slide.to_world((0.0, SPEC_Y, 0.0))
    return p[0] - OBJ_AXIS[0], p[1] - OBJ_AXIS[1]


def specimen_offset(slide) -> float:
    return math.hypot(*specimen_xy(slide))


def focus_error(j: dict) -> float:
    """Distance (m) of a loaded slide's top from the objective's focal plane, from the stage's height."""
    return abs(stage_gap(j["stage"]) - WD)


def score(objects):
    """0 unless the slide is still under its two clips (flat, within 3 mm of the holder centre, label to the left), it
    never came within 0.5 mm of the objective, both hands are released (every jaw point and pad midpoint 80 mm or more
    from every task surface) and every moving part moved under 1 mm over the last 21 control steps. Then: 0 unless the
    scales' crossing is in the camera's 2.50 x 2.00 mm view; 0.5 if it is; 0.75 if the slide's top is also within
    0.1 mm of focus (9.0 mm below the front lens); 1.0 if within 5 um of focus with the crossing within 0.05 mm of the
    objective's axis (the image centre)."""
    m, sl = objects["microscope"], objects["slide"]
    if not _loaded(m.joints, sl):
        return 0.0
    if _trial["min_gap"] is not None and _trial["min_gap"] < CRASH_GAP:
        return 0.0
    if not (_trial.get("far") and _trial.get("still")):
        return 0.0
    if not mark_in_view(*specimen_xy(sl)):
        return 0.0
    e = focus_error(m.joints)
    if e > NEAR_TOL:
        return 0.5
    return 1.0 if e <= FOCUS_TOL and specimen_offset(sl) <= CENTRE_TOL else 0.75


SUCCESS_HOLD_S = 15.0  # end a trial early only once it has stayed in focus for 15 s (time to let go of the knob and park)


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return specimen_offset(objects["slide"]) + focus_error(objects["microscope"].joints)


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Right arm: turns the adapter's y knob and x knob (straight-down grips, counterclockwise: on the trial stand clockwise
# moves the stage towards the robots and to the left) in an open quarter turn where needed and then closed loops on
# where the crossing is seen on the monitor; then grips the right coarse focus knob from the side and turns it a
# quarter turn, top towards the robots, and parks. Left arm: then grips the left fine focus knob from the side and turns
# it until the scales are sharp, and parks. The task file lets a Waypoint's ``yaw`` carry a full gripper rotation
# (``Orient``). MICRO_CONTROL=untouched (both arms stay parked) and MICRO_CONTROL=crash (the coarse knob is turned on
# until the slide reaches the objective) are the negative controls; both must score 0.


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_orient", False):
    def _grip_rotation(yaw=0.0):
        return yaw.rot if isinstance(yaw, Orient) else _down_grip(yaw)

    _grip_rotation._orient = True
    _demo.grip_rotation = _grip_rotation

SIDE = axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.pi / 2) @ _down_grip(0.0)  # level, pointing +y


def side(roll: float) -> Orient:
    return Orient(axis_angle_matrix(np.array([0.0, 1.0, 0.0]), roll) @ SIDE)


class Servo(Orient):
    """A Waypoint ``yaw`` whose gripper rotation turns a knob until the knob reaches ``target`` (closed loop): an
    ``on_step`` hook moves the commanded wrist angle by the remaining knob error, a little each step, between ``t0``
    and ``t1`` (seconds from the start of the demo). ``rot_of`` maps the wrist angle to a gripper rotation. ``target``
    may be a function of the knob angles (read off the knobs' index marks)."""

    def __new__(cls, joint, target, start, rot_of):
        obj = float.__new__(cls, 0.0)
        obj.joint, obj.target, obj.start, obj.rot_of = joint, target, start, rot_of
        obj.cmd, obj.t0, obj.t1, obj.a0 = start, 0.0, 0.0, None
        obj.slip = 0.0  # rad: aim this far counterclockwise of the target (the knob turns back when let go)
        return obj

    @property
    def rot(self):
        return self.rot_of(self.cmd)


_SERVOS: list = []
_clock = {"t": 0.0}
SERVO_GAIN, SERVO_MAX = 0.2, 0.02  # rad of wrist per step per rad of knob error; at most 0.02 rad (0.4 rad/s)


@on_reset
def _servo_reset(env):
    _clock["t"] = 0.0
    for sv in _SERVOS:
        sv.cmd, sv.a0 = sv.start, None


@on_step
def _servo_step(env):
    import labkit

    _clock["t"] += float(getattr(env.unwrapped, "step_dt", 0.05))
    active = [sv for sv in _SERVOS if sv.t0 <= _clock["t"] <= sv.t1]
    if not active:
        return
    j = labkit._part_state(env, labkit._SPECS["microscope"]).joints
    sl = labkit._part_state(env, labkit._SPECS["slide"])
    for sv in active:
        tgt = sv.target(j, sl) if callable(sv.target) else sv.target
        if sv.a0 is None:  # the turn's direction, fixed when the closed loop starts
            sv.a0 = 1.0 if tgt >= j[sv.joint] else -1.0
        tgt += sv.slip
        err = tgt - j[sv.joint]
        sv.cmd += float(np.clip(SERVO_GAIN * err, -SERVO_MAX, SERVO_MAX))


def _servo(wps, duration, xyz, finger, label, joint, target, start, rot_of):
    """Append a closed-loop knob turn of ``duration`` seconds to ``wps``: a 0.05 s step that lands on the servo's
    wrist angle (equal to ``start``, the previous waypoint's), then a hold on the same servo, so the commanded wrist
    angle is used as it is, not interpolated."""
    sv = Servo(joint, target, start, rot_of)
    wps.append(Waypoint(0.05, xyz, finger, label, sv))
    sv.t0 = sum(w.duration for w in wps)
    sv.t1 = sv.t0 + duration
    _SERVOS.append(sv)
    wps.append(Waypoint(duration, xyz, finger, label, sv))
    return sv


def _down(yaw):
    return _down_grip(yaw)


SIDE_L = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.pi) @ SIDE  # level, pointing -y (the left hand)


def side_l(roll: float) -> Orient:
    return Orient(axis_angle_matrix(np.array([0.0, 1.0, 0.0]), roll) @ SIDE_L)


def side_lv(roll: float) -> Orient:
    """The left hand's knob grip: pointing -y, fingers closing front and back (the side grip rolled 90 degrees)."""
    return side_l(math.pi / 2 + roll)


TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.12, -0.30, TOP + 0.18)  # well over 80 mm from the adapter's y knob (release needs 80 mm)


def W(x, y, z):
    """Local microscope coordinates to world."""
    return (MX + x, MY + y, TOP + z)


DK_GRIP_Z = TOP + DKNOB_TOP - DKNOB_H / 2 + TIP_ABOVE_TCP  # fingertips at the knob's mid-height
DK_IN = 0.05  # each knob grip comes in from the front (towards the robots) at grip height: a hover above the knob is
#   out of the arm's reach (the stage top is 0.96 m)
F_GRIP = W(FKNOB_X, -(FKNOB_Y0 + FKNOB_W) + 0.009 - TIP_ABOVE_TCP, FKNOB_Z)  # fingertips 9 mm onto the coarse knob
F_OUT = (F_GRIP[0], F_GRIP[1] - 0.025, F_GRIP[2])  # 25 mm off the knob along its axis (further is out of reach)
F_OUT_UP = (F_GRIP[0] - 0.06, F_GRIP[1] - 0.025, F_GRIP[2] + 0.02)
# the way from the stage knobs to the coarse knob: the adapter's belts and pulleys hang down to 0.855 m, so the hand goes
# out in front of them, down to 0.84 m, and round outboard of the y shaft (y -0.187 m)
F_FRONT = (MX + ADAPTER_AXIS[0] - ADAPTER_SHAFTS[0][0] - 0.07, MY + ADAPTER_AXIS[1] - 0.10, TOP + 0.093)
F_SIDE = (F_GRIP[0] - 0.06, MY + ADAPTER_AXIS[1] - ADAPTER_SHAFTS[1][1] - 0.035, TOP + 0.090)
FF_GRIP = W(FKNOB_X, FINE_Y0 + FINE_W - 0.007 + TIP_ABOVE_TCP, FKNOB_Z)  # left hand: fingertips 7 mm onto the fine knob
FF_OUT = (FF_GRIP[0], FF_GRIP[1] + 0.015, FF_GRIP[2])  # 15 mm off the knob (the fingertips are 7 mm onto it)
FF_FAR = (FF_GRIP[0] - 0.08, FF_GRIP[1] + 0.03, FF_GRIP[2])  # then out to the front, clear of the stage
FF_FAR_UP = (FF_GRIP[0] - 0.08, FF_GRIP[1] + 0.03, FF_GRIP[2] + 0.04)
FINE_WIDE = -0.024  # part-open round the 35 mm fine knob
# The left hand does both focus knobs, from the left: the coarse knob's rim, the fine knob (outboard of it, 35 mm
# across) between its fingers, then the fine knob itself. (From the right, the arm's way from the stage knobs to the
# right coarse knob ran through the adapter's belts and pulleys: r1, r2.)
LF_GRIP = W(FKNOB_X, FKNOB_Y0 + FKNOB_W - 0.009 + TIP_ABOVE_TCP, FKNOB_Z)  # fingertips 9 mm onto the left coarse knob



# Letting go of a stage knob turns it about 0.02 rad clockwise, whichever way it was being turned (r4: y -0.020,
# -0.020; x -0.021, -0.016 rad as the fingers opened). Each closed loop on a stage knob aims that far counterclockwise
# of its target. At 33.3 and 43.4 mm per turn the 0.05 mm target is 0.0094 and 0.0072 rad of knob.
SPRING_BACK = {"xdrive": 0.019, "ydrive": 0.020}  # rad of knob
FINE_RELEASE = 0.0  # rad: with the fine turn small, letting go moves the fine knob under 0.02 rad, about 1 um (r7; the
#   0.31 rad seen in r5 and r6 was lifting off with the fingers rolled after a large fine turn)


def x_target(j: dict, slide) -> float:
    """x drive angle that puts the crossing on the objective's axis, from where it is seen now (on the monitor); a
    counterclockwise (positive) turn moves the stage away from the robots."""
    p = slide.to_world((0.0, SPEC_Y, 0.0))
    return j["xdrive"] + 2 * math.pi * (OBJ_AXIS[0] - p[0]) / X_LEAD


def y_target(j: dict, slide) -> float:
    """y drive angle that puts the crossing on the objective's axis line, from where it is seen now (on the monitor);
    a counterclockwise (positive) turn moves the stage to the right."""
    p = slide.to_world((0.0, SPEC_Y, 0.0))
    return j["ydrive"] + 2 * math.pi * (p[1] - OBJ_AXIS[1]) / Y_LEAD


def fine_target(j: dict, slide=None) -> float:
    """Fine knob angle that brings the slide into focus, from how far the stage now sits from focus (seen as image
    sharpness; turning the fine knob negative raises the stage)."""
    return j["fine"] - 2 * math.pi * (START_OFF - j["stage"]) / FINE_LEAD


F_TURN = -2 * math.pi * START_OFF / F_LEAD  # a quarter turn: 1 mm at 4 mm per turn
F_SAG = 0.0001  # the stage sits about 0.1 mm low on its focus drive once it is off its lower stop (r5)
F_TARGET = F_TURN - 2 * math.pi * F_SAG / F_LEAD  # the coarse knob's aim, near focus: what is left for the fine knob is
#   a few degrees of it, so the left hand's fingers are still front and back when it lifts off (r5: a 1.7 rad fine turn
#   left them top and bottom, and lifting dragged the knob back 0.12 mm)
X_TARGET = 2 * math.pi * (QX_FOCUS - QX0) / X_LEAD  # drive knob angles that centre the crossing (counterclockwise)
Y_TARGET = -2 * math.pi * (QY_FOCUS - PUSHED_DY - QY0) / Y_LEAD
X_QUARTERS = max(0, int(abs(X_TARGET) // (math.pi / 2)) - 1)  # open quarter turns before a closed loop finishes
Y_QUARTERS = max(0, int(abs(Y_TARGET) // (math.pi / 2)) - 1)
FOCUS_WIDE = -0.038  # part-open round the 60 mm coarse knob
KNOB_WIDE = -0.030  # part-open round a 39 mm ribbed drive knob
KNOB_GRIP = -0.004  # closed on a drive knob
KNOB_RELEASE = -0.022  # opened just off the knob
KNOB_UNROLL = -0.040  # wide open (80 mm) to unroll round a released knob: 20 mm off its ribs, 10 mm off the other knob

PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("base", W(0.0, 0.0, BASE[2] / 2), BASE),
    ("housing", W(FKNOB_X, 0.0, 0.03 + HOUSING[2] / 2), HOUSING),
    ("pillar", W(PILLAR_X, 0.0, BASE[2] + PILLAR[2] / 2), PILLAR),
    ("objective", W(STAGE_X, 0.0, (OBJ_TIP + HEAD_TOP - HEAD[2]) / 2), (0.024, 0.024, HEAD_TOP - HEAD[2] - OBJ_TIP)),
    ("head", W(PILLAR_X + PILLAR[0] / 2 - HEAD[0] / 2, 0.0, HEAD_TOP - HEAD[2] / 2), HEAD),
    ("stage at load", W(STAGE_X + QX0, QY0, PLATE_TOP0 - 0.003), (0.10, 0.13, 0.006)),
    ("monitor", (MON_XY[0], MON_XY[1], TOP + MON_T / 2), (MON_H + 0.012, MON_W + 0.012, MON_T)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def _knob_turn(wps, xy, label, joint, target, servo):
    """A counterclockwise turn of a top-turn drive knob: in from the front at grip height with the fingers across the
    knob (yaw 0), grip, and turn; an open quarter turn to yaw 90 degrees, or (``servo``) a closed-loop turn that stops
    at the knob angle ``target``. Then let go, unroll the wrist round the released knob, and back out to the front."""
    x, y = xy
    g, h = (x, y, DK_GRIP_Z), (x - DK_IN, y, DK_GRIP_Z)
    wps += [
        Waypoint(1.5, h, KNOB_WIDE, f"in front of the {label}", 0.0),
        Waypoint(1.0, g, KNOB_WIDE, f"round the {label}", 0.0),
        Waypoint(0.8, g, KNOB_GRIP, "grip", 0.0),
    ]
    if servo:
        end = _servo(wps, 5.0, g, KNOB_GRIP, f"turn the {label} counterclockwise to its mark", joint, target, 0.0,
                     _down)
        end.slip = SPRING_BACK[joint]
    else:
        wps.append(Waypoint(4.0, g, KNOB_GRIP, f"turn the {label} a quarter turn counterclockwise", math.pi / 2))
        end = math.pi / 2
    wps += [
        Waypoint(0.6, g, KNOB_RELEASE, "let go", end),
        # wide open before the wrist unrolls round the released knob (at 2.5 mm off its ribs the fingertips dragged
        # it back about a degree, 0.1 mm of stage: r1)
        Waypoint(0.5, g, KNOB_UNROLL, "open wide", end),
        Waypoint(0.8, g, KNOB_UNROLL, "unroll the wrist round the knob", 0.0),
        Waypoint(0.8, h, KNOB_UNROLL, "back out to the front", 0.0),
    ]


def _coarse_grip(wps, from_knobs=False):
    if from_knobs:  # from the stage knobs: out to the front, down, and round outboard of the adapter's y shaft
        wps += [Waypoint(1.5, F_FRONT, KNOB_WIDE, "out in front of the adapter, down to the focus knobs' height"),
                Waypoint(1.5, F_SIDE, KNOB_WIDE, "outboard of the adapter, beside the right coarse knob")]
    wps += [
        Waypoint(2.0, F_OUT_UP, KNOB_WIDE, "above, beside the right coarse knob"),
        Waypoint(1.5, F_OUT, FOCUS_WIDE, "level the gripper", side(0.0)),
        Waypoint(1.5, F_GRIP, FOCUS_WIDE, "onto the coarse knob", side(0.0)),
        Waypoint(0.8, F_GRIP, FINGER_CLOSED, "grip the rim", side(0.0)),
    ]


def demo():
    _SERVOS.clear()
    mode = os.environ.get("MICRO_CONTROL", "")
    if mode == "untouched":  # negative control: nothing is turned
        return {"left": [Waypoint(20.0, LEFT_PARK, FINGER_OPEN, "rest")],
                "right": [Waypoint(20.0, RIGHT_PARK, FINGER_OPEN, "rest")]}
    yk, xk = W(*YKNOB, 0.0)[:2], W(*XKNOB, 0.0)[:2]
    right = [Waypoint(2.5, (yk[0] - DK_IN, yk[1], DK_GRIP_Z + 0.01), KNOB_WIDE, "towards the stage knobs", 0.0)]
    left = []
    if mode == "crash":  # negative control: the coarse knob is turned on, a quarter turn a grip, into the objective
        for _ in range(11):
            _coarse_grip(right)
            right += [Waypoint(2.0, F_GRIP, FINGER_CLOSED, "coarse: a quarter turn up", side(-math.pi / 2)),
                      Waypoint(0.8, F_GRIP, FOCUS_WIDE, "let go", side(-math.pi / 2)),
                      Waypoint(1.0, F_OUT, FOCUS_WIDE, "back off", side(-math.pi / 2)),
                      Waypoint(1.2, F_OUT, FOCUS_WIDE, "unroll", side(0.0))]
        right += [Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"), Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "hold")]
        left.append(Waypoint(max(0.5, _dur(right)), LEFT_PARK, FINGER_OPEN, "rest"))
        return {"left": left, "right": right}
    # Right arm: the stage drives, y first, then x, then a second and third closed loop on each (at 43.4 and 33.3 mm per
    # turn the 0.05 mm target is under a degree of knob; letting go can leave a degree or two)
    for _ in range(Y_QUARTERS):
        _knob_turn(right, yk, "y knob", "ydrive", None, servo=False)
    _knob_turn(right, yk, "y knob", "ydrive", y_target, servo=True)
    for _ in range(X_QUARTERS):
        _knob_turn(right, xk, "x knob", "xdrive", None, servo=False)
    _knob_turn(right, xk, "x knob", "xdrive", x_target, servo=True)
    for _ in range(2):
        _knob_turn(right, yk, "y knob", "ydrive", y_target, servo=True)
        _knob_turn(right, xk, "x knob", "xdrive", x_target, servo=True)
    right += [Waypoint(2.0, (yk[0] - DK_IN, yk[1] - 0.05, DK_GRIP_Z + 0.01), KNOB_WIDE, "out of the way", 0.0),
              Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park")]
    t_right = _dur(right)  # the right hand is parked from here
    # Left arm, at the same time: the left coarse knob a quarter turn, top towards the robots (most of it open loop,
    # then a closed loop to near focus), then the fine knob until sharp. Its fingers close front and back (the grip
    # rolled 90 degrees), so the hand comes straight down with the fine knob between them, slides in onto the coarse
    # knob's rim, and lifts straight off at the end (fingers closing top and bottom met the fine knob: r4).
    ff_up = (FF_GRIP[0], FF_GRIP[1], FF_GRIP[2] + 0.06)
    left = [Waypoint(2.0, ff_up, FOCUS_WIDE, "above the left focus knobs, fingers front and back", side_lv(0.0)),
            Waypoint(1.5, FF_GRIP, FOCUS_WIDE, "down round the fine knob", side_lv(0.0)),
            Waypoint(1.2, LF_GRIP, FOCUS_WIDE, "in onto the coarse knob's rim", side_lv(0.0)),
            Waypoint(0.8, LF_GRIP, FINGER_CLOSED, "grip the coarse knob", side_lv(0.0))]
    for k in range(1, 4):
        left.append(Waypoint(1.2, LF_GRIP, FINGER_CLOSED, "coarse focus: turn, top towards the robots",
                             side_lv(0.9 * F_TARGET * k / 3)))
    fsv = _servo(left, 5.0, LF_GRIP, FINGER_CLOSED, "coarse focus: finish the turn to near focus", "focus",
                 F_TARGET, 0.9 * F_TARGET, lambda r: side_lv(r).rot)
    left += [
        Waypoint(0.6, LF_GRIP, FINGER_CLOSED, "settle", fsv),
        Waypoint(0.8, LF_GRIP, FOCUS_WIDE, "let go", fsv),
        Waypoint(1.0, FF_GRIP, FOCUS_WIDE, "out along the shaft, round the fine knob", fsv),
        Waypoint(1.2, FF_GRIP, FOCUS_WIDE, "unroll round the fine knob", side_lv(0.0)),
        # the fine knob only once the right hand has let go of the stage knobs and parked: pressing on them moves the
        # stage up to 15 um (r6)
        Waypoint(max(0.5, t_right - _dur(left) - 5.4), FF_GRIP, FOCUS_WIDE, "wait, round the fine knob", side_lv(0.0)),
        Waypoint(0.8, FF_GRIP, FINE_WIDE, "close in on the fine knob", side_lv(0.0)),
        Waypoint(0.8, FF_GRIP, FINGER_CLOSED, "grip it", side_lv(0.0)),
    ]
    nsv = _servo(left, 8.0, FF_GRIP, FINGER_CLOSED, "fine focus: turn until sharp", "fine", fine_target, 0.0,
                 lambda r: side_lv(r).rot)
    nsv.slip = FINE_RELEASE
    left += [
        Waypoint(0.6, FF_GRIP, FINGER_CLOSED, "settle", nsv),
        Waypoint(0.8, FF_GRIP, FOCUS_WIDE, "let go", nsv),
        Waypoint(1.2, ff_up, FOCUS_WIDE, "straight up off the fine knob", nsv),
        Waypoint(1.0, ff_up, FOCUS_WIDE, "unroll", side_lv(0.0)),
        Waypoint(1.5, FF_FAR_UP, KNOB_WIDE, "out to the front, pointing down again"),
        Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    end = max(_dur(left), _dur(right)) + 2.0
    left.append(Waypoint(end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    return {"left": left, "right": right}
