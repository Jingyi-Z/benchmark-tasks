"""On the station table: an upright light microscope (modelled on a research stand: mechanical stage with a spring slide
holder and x/y drive knobs, focus knobs on both sides) and a glass slide on a rest beside it. The stage is pulled
towards the robots, away from the objective, for loading, and sits 1 mm below focus.

The task, in the order a microscopist works:
1. Load the slide: hold the slide holder's spring arm open and lay the slide in the holder against its stops, label to
   the left; let the arm close over the slide's right end. (Two hands: the arm springs shut when let go.)
2. Bring the specimen under the objective with the stage's x and y drive knobs.
3. Focus: raise the stage with a coarse focus knob to near focus, never letting the slide come within 0.5 mm of the
   objective, then finish with the fine focus knob until the chip's alignment mark is sharp on the camera monitor (a tablet on the table)
   (within the 10x objective's depth of field, 10 um).

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
MX, MY = 0.34, 0.06
BASE = (0.26, 0.16, 0.03)
PILLAR = (0.07, 0.11, 0.185)
PILLAR_X = 0.085
HEAD = (0.18, 0.06, 0.05)  # the arm over the stage (6 cm wide)
HEAD_TOP = 0.215
STAGE_X = -0.03  # objective axis (local x); the stage is centred under it when its carriages are at zero
OBJ_R, OBJ_TIP = 0.012, 0.120  # the 10x objective: 24 mm across, front lens 12 cm above the table
WD = 0.0052  # working distance: in focus when the slide's top is 5.2 mm below the front lens
START_OFF = 0.001  # the stage starts 1 mm below focus
SLIDE_W, SLIDE_L, SLIDE_T = 0.026, 0.076, 0.002  # x by y by thickness (drawn 2 mm thick; a real slide is 1 mm)
SPEC_Y = 0.0  # the crossed scales' crossing (the alignment mark) is at the slide's centre; the label on the left end
PLATE_TOP0 = OBJ_TIP - WD - START_OFF - SLIDE_T  # top of the stage (x carriage) at the start
STAGE_MAX = START_OFF + WD - 0.0002  # the focus stops just short of the slide touching the objective

# Focus drive housing under the stage, with one knob shaft through it along y.
HOUSING = (0.05, 0.20, 0.052)  # the focus mechanism under the stage (50 mm deep, behind the stage-knob adapter)
FKNOB_X, FKNOB_Z = -0.04, 0.07
FKNOB_R, FKNOB_W = 0.021, 0.020
FKNOB_Y0 = HOUSING[1] / 2
F_LEAD = 0.002  # coarse focus: m of stage travel per turn (the stand's spec: 2 mm per revolution)
F_MIN = -2 * math.pi * STAGE_MAX / F_LEAD
# Fine focus: a smaller knob on the same axis, outboard of each coarse knob.
FINE_LEAD = 0.0002  # m of stage travel per turn (the stand's spec: 0.2 mm per revolution, 2 um divisions)
FINE_R, FINE_W = 0.013, 0.012
FINE_Y0 = FKNOB_Y0 + FKNOB_W + 0.001  # inner face of each fine knob (|y|)
FINE_LIM = 4 * math.pi

# Stage travel: the x carriage starts 85 mm towards the robots (loading position), the y carriage centred.
QX0, QY0 = -0.0125, 0.025  # the stage starts at its front-left limits, the slide already in its holder
QX_FOCUS, QY_FOCUS = 0.0, -SPEC_Y  # carriage positions that put the specimen on the objective axis
X_LEAD, Y_LEAD = 0.015, 0.015  # stage drive: m per knob turn (turning a knob clockwise seen from above moves the stage
#   away from the robots for x, to the right for y); measured on the reference stand: about 15 mm per 360-degree turn,
#   confirmed on the stage's scales before trials
# The stage knobs are the top knobs of the printed belt adapter (adapter/knob_adapter.py, the one geometry for both the
# real adapter and this scene). Its frame: the stand's coaxial stage control's axis, 5 mm behind the lower stage plate's
# front edge and 15 mm inboard of its right edge, at the stage's top surface; u towards the robots (-x), v outboard (-y).
ADAPTER_AXIS = (STAGE_X - 0.060 + 0.005, -0.075 + 0.015)
ADAPTER_SHAFTS = ((0.050, 0.0461), (0.050, 0.1435))  # (u, v) of the x and y knobs' shafts (belts 232 and 400 mm)


def _adapter_xy(u, v):
    return (ADAPTER_AXIS[0] - u, ADAPTER_AXIS[1] - v)


XKNOB = _adapter_xy(*ADAPTER_SHAFTS[0])  # x (inner) and y (outer) stage knobs, 97.4 mm apart (local xy)
YKNOB = _adapter_xy(*ADAPTER_SHAFTS[1])
DKNOB_R, DKNOB_H = 0.017, 0.016
DKNOB_TOP = PLATE_TOP0 + 0.0175  # the adapter's knob tops, 17.5 mm above the stage top (they rise with the stage)
LIM = 0.10
LIM_X, LIM_Y = (-0.0125, 0.0125), 0.025  # the stage's travel: 25 mm front to back, 50 mm left to right

# Slide holder on the x carriage: the slide lies with its centre at the holder centre (STAGE_X, 0) against two back
# posts and a left stop; a spring lever pivoting behind its right end presses a jaw on that end, against the stop. Pulling the lever's
# handle (out to the right) back swings the jaw off the slide's end.
STOP_IN = SLIDE_L / 2 + 0.0005  # the left stop's inner face (y): a loaded slide is pushed against it
PUSHED_DY = STOP_IN - SLIDE_L / 2  # so a loaded slide's centre sits this far left of the holder centre
PIVOT = (STAGE_X + SLIDE_W / 2 + 0.012, STOP_IN - SLIDE_L + 0.0008 - 0.003)  # the closed jaw's face presses 0.8 mm
ARM_L = 0.044  # the clamp jaw, across the slide's right end
HANDLE_L = 0.060  # the lever's handle, out to the right, with the post at its end
ARM_Z = PLATE_TOP0 + 0.0003  # the jaw's underside, just clear of the plate: it pushes the slide's right end
POST_R, POST_H = 0.005, 0.020
ARM_OPEN = 1.2


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
        ((FKNOB_X, s * (FINE_Y0 + FINE_W / 2), FKNOB_Z), FINE_R, FINE_W, BLACK, "Y") for s in (1, -1)
    ] + [
        ((FKNOB_X + FINE_R * math.cos(i * math.pi / 8), s * (FINE_Y0 + FINE_W / 2),
          FKNOB_Z + FINE_R * math.sin(i * math.pi / 8)), 0.0015, FINE_W, BLACK, "Y")
        for s in (1, -1) for i in range(16)
    ],
    # graduations: a white index line on each fine knob's face
    boxes=[((FKNOB_X, s * (FINE_Y0 + FINE_W + 0.0005), FKNOB_Z + FINE_R / 2), (0.002, 0.001, FINE_R), WHITE)
           for s in (1, -1)],
)
STAGE = Joint(  # the focus slide: carries the stage, the drive bracket and knobs
    "stage", kind="slide", axis="Z", anchor=(STAGE_X, 0.0, PLATE_TOP0), lower=0.0, upper=STAGE_MAX, initial=0.0,
    stiffness=20000.0, damping=400.0, target=0.0, mass=0.05,
    boxes=[
        ((STAGE_X + 0.075, 0.0, PLATE_TOP0 - 0.03), (0.03, 0.07, 0.03), DARK),  # carrier, into the pillar
        ((STAGE_X, 0.0, PLATE_TOP0 - 0.0135), (0.12, 0.15, 0.005), DARK),  # fixed lower plate
        # the knob adapter's L-shaped arm and locating wall (colliding stand-ins for its drawn mesh), z -15.5 to -11.5 mm
        # under the stage top: a strip along the wall (u -5 to 66 mm, v 16 to 30) and a bar along the shafts' line
        # (u 34 to 66, v 16 to 159.5); the wall along the lower plate's right edge face, z -17 to -11.5 mm
        ((ADAPTER_AXIS[0] - 0.0305, ADAPTER_AXIS[1] - 0.023, PLATE_TOP0 - 0.0135), (0.071, 0.014, 0.004), ADAPTER_GREY),
        ((ADAPTER_AXIS[0] - 0.050, ADAPTER_AXIS[1] - 0.08775, PLATE_TOP0 - 0.0135), (0.032, 0.1435, 0.004), ADAPTER_GREY),
        ((ADAPTER_AXIS[0] - (0.005 - 0.075) / 2, ADAPTER_AXIS[1] - 0.018, PLATE_TOP0 - 0.01425),
         (0.080, 0.004, 0.0055), ADAPTER_GREY),
    ],
    # The illuminator's light, from below through the stage, shows as a spot on the slide on the objective's axis. It
    # rises with the stage (the condenser does) and is only drawn: nothing collides with it.
    # It is drawn just over the glass and under the specimen square's top, so the square shows on the light.
    cylinders=[((STAGE_X, 0.0, PLATE_TOP0 + SLIDE_T + 0.0001), SPOT_R, 0.0001, LIGHT),
               # the adapter's bearing bosses, from the arm's top (11.5 mm under the stage top) up to the knobs
               *[((kx, ky, (PLATE_TOP0 - 0.0115 + DKNOB_TOP - DKNOB_H) / 2), 0.012,
                  DKNOB_TOP - DKNOB_H - (PLATE_TOP0 - 0.0115), ADAPTER_GREY) for kx, ky in (XKNOB, YKNOB)]],
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
        *[((STAGE_X + SLIDE_W / 2 + 0.004, y, PLATE_TOP0 + 0.0004), (0.004, 0.006, 0.0008), STEEL)
          for y in (-0.020, 0.005)],  # low back posts, 2 mm behind the slide's back edge
        ((STAGE_X, STOP_IN + 0.002, PLATE_TOP0 + 0.0015), (0.012, 0.004, 0.003), STEEL),  # left stop
        ((STAGE_X - 0.02, 0.0, PLATE_TOP0 + 0.0002), (0.002, 0.03, 0.0004), WHITE),  # holder index line
    ],
)
HOLDER = Joint(
    "holder", kind="hinge", axis="Z", anchor=(*PIVOT, ARM_Z), lower=0.0, upper=ARM_OPEN, initial=0.0,
    stiffness=0.25, damping=0.03, target=0.0, mass=0.02, parent="stage_x",
    boxes=[((PIVOT[0] - ARM_L / 2, PIVOT[1], ARM_Z + 0.00135), (ARM_L, 0.006, 0.0027), STEEL),  # jaw
           ((PIVOT[0], PIVOT[1] - HANDLE_L / 2, ARM_Z + 0.0015), (0.006, HANDLE_L, 0.003), STEEL)],  # handle
    cylinders=[((PIVOT[0], PIVOT[1], ARM_Z + 0.0015), 0.005, 0.003, STEEL),
               ((PIVOT[0], PIVOT[1] - HANDLE_L + 0.003, ARM_Z + 0.003 + POST_H / 2), POST_R, POST_H, BLACK)],
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
        # the 1 MP eyepiece camera (as AmScope's MD100) in the other eyepiece tube
        ((STAGE_X + 0.0971, 0.032, HEAD_TOP + 0.0971), (0.045, 0.030, 0.030), BLACK, (0.0, -math.pi / 4)),
    ],
    cylinders=[
        ((STAGE_X, 0.0, HEAD_TOP - HEAD[2] - 0.006), 0.032, 0.012, STEEL),  # nosepiece
        ((STAGE_X, 0.0, (OBJ_TIP + HEAD_TOP - HEAD[2] - 0.012) / 2), OBJ_R, HEAD_TOP - HEAD[2] - 0.012 - OBJ_TIP, STEEL),
        ((STAGE_X, 0.0, OBJ_TIP + 0.004), OBJ_R + 0.0005, 0.002, RED),  # the objective's colour ring
    ],
    joints=[FOCUS, FINE, STAGE, STAGE_Y, STAGE_X_J, HOLDER, XDRIVE, YDRIVE],
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
# screen up, showing the microscope camera's live 10x view over the whole screen (18 x 12 cm, 6 x 4 mm of the slide:
# 30 mm on the screen per mm). The screen's colour shows whether the chip is under the objective; the alignment mark
# moves with the stage and is drawn sharp, soft or blurred by how far the slide is from focus (within 10 um, 0.1 mm,
# 1 mm; beyond that it is lost in the blur). The screen's centre is the objective's axis; away from the robots on the
# slide is away on the screen (its far edge is up in the robots' view), and to their left is to the left.
MON_XY = (0.09, 0.0)  # the screen's centre on the table
MON_W, MON_H, MON_T = 0.18, 0.102, 0.012  # across (y), along (x), thickness
SCR_PER_M = 0.18 / 0.000384  # 469 mm on the screen per mm of the slide
MARK_LEVELS = ((SHARP, 0.005), (SOFT, 0.012), (BLUR, 0.025))  # colour, line width on the screen
_face = MON_T + 0.0010  # the screen's face, on top of the tablet
MONITOR_BOXES = [
    ((0.0, 0.0, MON_T / 2), (MON_H + 0.012, MON_W + 0.012, MON_T), DARK),  # tablet body and bezel
    ((0.0, 0.0, MON_T + 0.0005), (MON_H, MON_W, 0.001), SCREEN),
    # layers, bottom to top: the chip view, the marks, then the bare-glass view on top, so that the start state (bare
    # glass) is what shows before the first step hides the layers not in use
    ((0.0, 0.0, _face + 0.0010), (MON_H, MON_W, 0.0003), VIEW_GLASS),  # the view over bare glass
    ((0.0, 0.0, _face + 0.0002), (MON_H, MON_W, 0.0003), VIEW_CHIP),  # the view over the chip
]
VIEW_BOX0 = 2
MARK_BOX0 = len(MONITOR_BOXES)  # then, per level, the cross's two lines, each across the whole screen (the 1 mm scales
# are longer than the field): the one across (along y) moves only along x, the one along x moves only across
for _col, _w in MARK_LEVELS:
    MONITOR_BOXES += [((0.0, 0.0, _face + 0.0006), (_w, MON_W, 0.0003), _col),
                      ((0.0, 0.0, _face + 0.0006), (MON_H, _w, 0.0003), _col)]
MARK_PER_LEVEL = 2
TICK = 0.00001 * SCR_PER_M  # the scales' 0.01 mm divisions on the screen (4.7 mm apart)
TICK_K = range(-19, 20)
TICK_BOX0 = len(MONITOR_BOXES)  # then, per level, a tick across each line at each division: all built at the screen's
# centre (under the top view layer, so none shows outside the screen before the first draw) and placed when drawn,
# hidden where it would fall off the screen
TICKS = []  # (division k, on the line along "y" or "x"), in box order within a level
for _col, _w in MARK_LEVELS:
    TICKS = []
    for _k in TICK_K:
        _len = 0.007 if _k % 10 == 0 else 0.004 if _k % 5 == 0 else 0.0022
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
    qx = float(np.clip(QX0 - X_LEAD * j["xdrive"] / (2 * math.pi), *LIM_X))
    qy = float(np.clip(QY0 + Y_LEAD * j["ydrive"] / (2 * math.pi), -LIM_Y, LIM_Y))
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
    _trial["far"], _trial["still"], _trial["hist"], _trial["hist_holder"] = False, False, [], []
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
    field = "chip" if max(abs(dx), abs(dy)) < 0.0005 else "glass"
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
            for prim, op, base in row:
                img = UsdGeom.Imageable(prim)
                if lv == level:
                    k = row.index((prim, op, base))
                    op.Set(Gf.Vec3d(base[0] + (ox if k == 0 else 0.0), base[1] + (oy if k == 1 else 0.0), base[2]))
                    img.MakeVisible()
                else:
                    img.MakeInvisible()
        for lv, row in enumerate(pr["ticks"]):
            for (prim, op, base), (k, line) in zip(row, TICKS):
                img = UsdGeom.Imageable(prim)
                x, y = base[0] + ox + (k * TICK if line == "x" else 0.0), base[1] + oy + (k * TICK if line == "y" else 0.0)
                if lv == level and abs(x) <= MON_H / 2 - 0.004 and abs(y) <= MON_W / 2 - 0.004:
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


def _task_boxes(j: dict, slide):
    """World axis-aligned boxes (centre, half sizes) of every task surface: the microscope's fixed body and cylinders,
    its stage, holder and knobs where the joints now put them, the slide and the monitor."""
    org = np.array([MX, MY, TOP])
    qx, qy, qz = j["stage_x"], j["stage_y"], j["stage"]
    out = []
    for c, sz, *_ in MICROSCOPE.boxes:
        out.append((org + np.array(c, float), np.array(sz, float) / 2))
    for c, r, h, *rest in MICROSCOPE.cylinders:
        ax = rest[1] if len(rest) > 1 else "Z"
        half = [r, r, h / 2] if ax == "Z" else ([r, h / 2, r] if ax == "Y" else [h / 2, r, r])
        out.append((org + np.array(c, float), np.array(half, float)))
    moves = {"stage": (0.0, 0.0, qz), "stage_y": (0.0, qy, qz), "stage_x": (qx, qy, qz), "holder": (qx, qy, qz),
             "xdrive": (0.0, 0.0, qz), "ydrive": (0.0, 0.0, qz), "focus": (0.0, 0.0, 0.0), "fine": (0.0, 0.0, 0.0)}
    for J in MICROSCOPE.joints:
        d = np.array(moves.get(J.name, (0.0, 0.0, 0.0)), float) - (np.array([QX0, QY0, 0.0]) if J.name in ("stage_x", "holder") else np.array([0.0, QY0, 0.0]) if J.name == "stage_y" else 0.0)
        for c, sz, *_ in (J.boxes or []):
            pad = 0.05 if J.name == "holder" else 0.0  # (the lever swings: a generous box)
            out.append((org + np.array(c, float) + d, np.array(sz, float) / 2 + pad))
        for c, r, h, *rest in (J.cylinders or []):
            ax = rest[1] if len(rest) > 1 else "Z"
            half = [r, r, h / 2] if ax == "Z" else ([r, h / 2, r] if ax == "Y" else [h / 2, r, r])
            out.append((org + np.array(c, float) + d, np.array(half, float)))
    out.append((np.asarray(slide.to_world((0.0, 0.0, SLIDE_T / 2)), float), np.array([SLIDE_W, SLIDE_L, SLIDE_T]) / 2))
    out.append((np.array([MON_XY[0], MON_XY[1], TOP + MON_T / 2]), np.array([MON_H + 0.012, MON_W + 0.012, MON_T]) / 2))
    return out


def _moving_points(j: dict, slide) -> np.ndarray:
    """Material points on every moving task part, in 3-D (m): three on the slide, one on the stage's x carriage (it
    carries the y and focus motions too), the holder jaw's tip, and a rim point on each knob (coarse, fine and both
    stage knobs), each at its knob's own radius and angle."""
    pts = [np.asarray(slide.to_world(p), float) for p in ((0.0, -SLIDE_L / 2, 0.0), (0.0, SLIDE_L / 2, 0.0),
                                                           (SLIDE_W / 2, 0.0, 0.0))]
    qx, qy, qz = j["stage_x"], j["stage_y"], j["stage"]
    pts.append(np.array([qx, qy, qz]))
    th = j["holder"]
    pts.append(np.array([PIVOT[0] - ARM_L * math.cos(th) + qx, PIVOT[1] - ARM_L * math.sin(th) + qy, qz]))
    for name, r, lift in (("focus", FKNOB_R, 0.0), ("fine", FINE_R, 0.0), ("xdrive", DKNOB_R, qz), ("ydrive", DKNOB_R, qz)):
        a = j[name]
        pts.append(np.array([r * math.cos(a), r * math.sin(a), lift]))
    return np.array(pts)


def _max_travel(hist) -> float:
    """The largest distance any one tracked point moved between any two of the recorded samples (m)."""
    h = np.asarray(hist)  # samples x points x 3
    d = np.linalg.norm(h[:, None, :, :] - h[None, :, :, :], axis=3)
    return float(d.max())


HOLDER_R = 0.062  # the holder's farthest surface from its pivot: the grasp post (centre 57 mm out, radius 5 mm)


def _holder_travel(hist) -> float:
    """The largest movement of any point of the holder lever between any two samples (m), as a conservative bound
    from its pivot's translation and its turn: |d pivot| + 2 * HOLDER_R * |sin(d angle / 2)|."""
    h = np.asarray(hist)  # samples x (x, y, z, angle)
    dt = np.linalg.norm(h[:, None, :3] - h[None, :, :3], axis=2)
    da = np.abs(np.sin((h[:, None, 3] - h[None, :, 3]) / 2.0))
    return float((dt + 2.0 * HOLDER_R * da).max())


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
                  f"min_gap={(_trial['min_gap'] or 0) * 1000:.2f} mm", file=sys.stderr, flush=True)
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
    qx, qy, qz = (s.joints["stage_x"], s.joints["stage_y"], s.joints["stage"])
    hh = _trial.setdefault("hist_holder", [])
    hh.append((PIVOT[0] + qx, PIVOT[1] + qy, qz, s.joints["holder"]))  # the lever's pivot (it rides on the carriage)
    del hh[:-STILL_N]
    _trial["still"] = (len(hist) == STILL_N and _max_travel(hist) < STILL_TOL and len(hh) == STILL_N
                       and _holder_travel(hh) < STILL_TOL)
    if os.environ.get("MICRO_DEBUG") and _trial.get("n", 0) % 20 == 0 and len(hist) == STILL_N:
        h = np.asarray(hist)
        per = np.linalg.norm(h[:, None] - h[None, :], axis=3).max(axis=(0, 1)) * 1000
        print(f"[gate] clear={clear * 1000:.1f} mm far={_trial['far']} still={_trial['still']} travel_mm="
              + ",".join(f"{v:.2f}" for v in per) + f" holder_mm={_holder_travel(hh) * 1000:.2f}",
              file=sys.stderr, flush=True)


# -- scoring ----------------------------------------------------------------------------------------------------------
OBJ_AXIS = (MX + STAGE_X, MY)
HOLD_TOL, HOLD_Z, FLAT_DEG, CLOSED = 0.003, 0.001, 3.0, 0.10
# The camera's field of view is what the monitor shows: 4 mm along x by 6 mm across y of the slide, centred on the
# objective's axis. The mark (1.6 mm across) is in view when all of it is on the screen.
FIELD_X, FIELD_Y, MARK_HALF = 0.000108, 0.000192, 0.0
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
            and slide.to_world((0, 1, 0))[1] - slide.pos[1] > 0.99 and j["holder"] <= CLOSED)


def specimen_xy(slide):
    p = slide.to_world((0.0, SPEC_Y, 0.0))
    return p[0] - OBJ_AXIS[0], p[1] - OBJ_AXIS[1]


def specimen_offset(slide) -> float:
    return math.hypot(*specimen_xy(slide))


def focus_error(j: dict) -> float:
    """Distance (m) of a loaded slide's top from the objective's focal plane, from the stage's height."""
    return abs(stage_gap(j["stage"]) - WD)


def score(objects):
    """0 unless the slide is still in the holder (flat, within 3 mm of its centre, label to the left, the holder's jaw
    closed), it never came within 0.5 mm of the objective, both hands are released (every jaw point and pad midpoint
    80 mm or more from every task surface) and the slide and stage moved under 1 mm over the last 21 control steps.
    Then: 0 unless the scales' crossing is in the camera's 0.38 x 0.22 mm view; 0.5 if it is; 0.75 if the slide's top
    is also within 0.1 mm of focus (5.2 mm below the front lens); 1.0 if within 5 um of focus with the crossing within
    0.05 mm of the objective's axis (the image centre)."""
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
# Left arm: picks the slide off its rest by the label end and lays it in the holder while the right arm holds the
# holder's arm open by its post; then parks. Right arm: lets the holder arm close, turns the y knob a quarter turn and
# the x knob half a turn (two quarter turns) clockwise, then grips the right focus knob from the side and turns it half
# a turn, top towards the robots. The task file lets a Waypoint's ``yaw`` carry a full gripper rotation (``Orient``).


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


TIP_ABOVE_TCP = 0.013
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.12, -0.30, TOP + 0.18)  # 104 mm from the adapter's y knob (release needs 80 mm)


def W(x, y, z):
    """Local microscope coordinates to world."""
    return (MX + x, MY + y, TOP + z)


DK_GRIP_Z = TOP + DKNOB_TOP - DKNOB_H / 2 + TIP_ABOVE_TCP
DK_HOVER = DK_GRIP_Z + 0.05
F_GRIP = W(FKNOB_X, -(FKNOB_Y0 + FKNOB_W) + 0.009 - TIP_ABOVE_TCP, FKNOB_Z)
F_OUT = (F_GRIP[0], F_GRIP[1] - 0.04, F_GRIP[2])
F_OUT_UP = (F_GRIP[0], F_GRIP[1] - 0.04, F_GRIP[2] + 0.04)
FF_GRIP = W(FKNOB_X, -(FINE_Y0 + FINE_W) + 0.007 - TIP_ABOVE_TCP, FKNOB_Z)  # fingertips 7 mm onto the fine knob
FF_OUT = (FF_GRIP[0], FF_GRIP[1] - 0.04, FF_GRIP[2])
FF_FAR = (FF_GRIP[0], FF_GRIP[1] - 0.10, FF_GRIP[2])
FF_FAR_UP = (FF_GRIP[0], FF_GRIP[1] - 0.10, FF_GRIP[2] + 0.04)
FINE_WIDE = -0.02  # part-open round the 26 mm fine knob


# Letting go of a stage knob turns it back about a degree (r13: the cross moved +0.078 mm in x and -0.081 mm in y after
# each closed-loop turn that had put it on the axis), so each closed loop aims that far past the axis.
RELEASE_SLIP = (0.000078, -0.000081)


def x_target(j: dict, slide) -> float:
    """x drive angle that puts the alignment mark on the objective's axis, from where the mark is seen now (on the monitor)."""
    p = slide.to_world((0.0, SPEC_Y, 0.0))
    return j["xdrive"] + 2 * math.pi * (p[0] - (OBJ_AXIS[0] - RELEASE_SLIP[0])) / X_LEAD


def y_target(j: dict, slide) -> float:
    """y drive angle that puts the alignment mark on the objective's axis line, from where the mark is seen now (on the monitor)."""
    p = slide.to_world((0.0, SPEC_Y, 0.0))
    return j["ydrive"] + 2 * math.pi * ((OBJ_AXIS[1] - RELEASE_SLIP[1]) - p[1]) / Y_LEAD


def fine_target(j: dict, slide=None) -> float:
    """Fine knob angle that brings the slide into focus, from how far the stage now sits from focus (seen as image
    sharpness; turning the fine knob negative raises the stage)."""
    return j["fine"] - 2 * math.pi * (START_OFF - j["stage"]) / FINE_LEAD
F_TURN = -2 * math.pi * START_OFF / F_LEAD
F_TARGET = F_TURN  # focus knob angle for the working distance
X_TARGET = -2 * math.pi * (QX_FOCUS - QX0) / X_LEAD  # drive knob angles that centre the specimen
Y_TARGET = 2 * math.pi * (QY_FOCUS - PUSHED_DY - QY0) / Y_LEAD
X_QUARTERS = max(0, int(abs(X_TARGET) // (math.pi / 2)) - 1)  # open quarter turns before a closed loop finishes
Y_QUARTERS = max(0, int(abs(Y_TARGET) // (math.pi / 2)) - 1)
WIDE = -0.025  # fingers part-open round the post or the slide
FOCUS_WIDE = -0.03  # part-open round the 47 mm focus knob, clear of the table
KNOB_WIDE = -0.030  # part-open round a 39 mm ribbed drive knob
KNOB_GRIP = -0.004  # closed on a drive knob
KNOB_RELEASE = -0.022  # opened just off the knob, clear of its neighbour

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
    """A clockwise turn of a top-down drive knob: grip at yaw 90 degrees (fingers closing along x) and turn; an open
    quarter turn to yaw 0, or (``servo``) a closed-loop turn that stops at the knob angle ``target``. Then let go and
    lift."""
    z = DK_GRIP_Z
    q = math.pi / 2
    wps += [
        Waypoint(1.5, (*xy, DK_HOVER), KNOB_WIDE, f"above the {label}", q),
        Waypoint(1.0, (*xy, z), KNOB_WIDE, f"round the {label}", q),
        Waypoint(0.8, (*xy, z), KNOB_GRIP, "grip", q),
    ]
    if servo:
        sv = _servo(wps, 5.0, (*xy, z), KNOB_GRIP, f"turn the {label} clockwise to its mark", joint, target, q, _down)
        end = sv
    else:
        wps.append(Waypoint(4.0, (*xy, z), KNOB_GRIP, f"turn the {label} a quarter turn clockwise", 0.0))
        end = 0.0
    wps += [
        Waypoint(0.6, (*xy, z), KNOB_RELEASE, "let go", end),
        Waypoint(0.8, (*xy, DK_HOVER), KNOB_RELEASE, "up", end),
        Waypoint(0.8, (*xy, DK_HOVER), KNOB_RELEASE, "wrist back", q),
    ]


def demo():
    _SERVOS.clear()
    right = [Waypoint(2.5, (*W(*YKNOB, 0.0)[:2], DK_HOVER + 0.03), KNOB_WIDE, "towards the stage knobs")]
    left = []
    # Right arm: the stage drives (under the stage's right side), once the jaw has closed over the slide: the y knob
    # about a turn in quarter turns, then closed loops on where the cross is seen; then the x knob half a turn the same
    # way, and a second closed loop on each
    for _ in range(Y_QUARTERS):
        _knob_turn(right, W(*YKNOB, 0.0)[:2], "y knob", "ydrive", None, servo=False)
    _knob_turn(right, W(*YKNOB, 0.0)[:2], "y knob", "ydrive", y_target, servo=True)
    for _ in range(X_QUARTERS):
        _knob_turn(right, W(*XKNOB, 0.0)[:2], "x knob", "xdrive", None, servo=False)
    _knob_turn(right, W(*XKNOB, 0.0)[:2], "x knob", "xdrive", x_target, servo=True)
    _knob_turn(right, W(*YKNOB, 0.0)[:2], "y knob", "ydrive", y_target, servo=True)
    _knob_turn(right, W(*XKNOB, 0.0)[:2], "x knob", "xdrive", x_target, servo=True)
    # a third closed loop on each (15 mm per turn: 0.05 mm is 1.2 deg of knob; letting go can leave a degree)
    _knob_turn(right, W(*YKNOB, 0.0)[:2], "y knob", "ydrive", y_target, servo=True)
    _knob_turn(right, W(*XKNOB, 0.0)[:2], "x knob", "xdrive", x_target, servo=True)
    right += [
        Waypoint(2.0, F_OUT_UP, KNOB_WIDE, "above, beside the right focus knob"),
        Waypoint(1.5, F_OUT, FOCUS_WIDE, "level the gripper", side(0.0)),
        Waypoint(1.5, F_GRIP, FOCUS_WIDE, "onto the focus knob", side(0.0)),
        Waypoint(0.8, F_GRIP, FINGER_CLOSED, "grip the rim", side(0.0)),
    ]
    for k in range(1, 6):  # most of the way open loop, in 1.2 s steps of 0.18 turn
        right.append(Waypoint(1.2, F_GRIP, FINGER_CLOSED, "focus: turn, top towards the robots",
                              side(0.9 * F_TARGET * k / 5)))
    fsv = _servo(right, 5.0, F_GRIP, FINGER_CLOSED, "coarse focus: finish the turn to near focus", "focus",
                 F_TARGET, 0.9 * F_TARGET, lambda r: side(r).rot)
    right += [
        Waypoint(0.6, F_GRIP, FINGER_CLOSED, "settle", fsv),
        Waypoint(0.8, F_GRIP, FOCUS_WIDE, "let go", fsv),
        Waypoint(1.0, F_OUT, FOCUS_WIDE, "back off", fsv),
        Waypoint(1.2, F_OUT, FOCUS_WIDE, "unroll", side(F_TURN / 2)),
        Waypoint(1.2, F_OUT, FOCUS_WIDE, "unroll", side(0.0)),
        Waypoint(1.2, FF_OUT, FINE_WIDE, "out to the fine knob", side(0.0)),
        Waypoint(1.2, FF_GRIP, FINE_WIDE, "onto the fine knob", side(0.0)),
        Waypoint(0.8, FF_GRIP, FINGER_CLOSED, "grip it", side(0.0)),
    ]
    nsv = _servo(right, 6.0, FF_GRIP, FINGER_CLOSED, "fine focus: turn until sharp", "fine", fine_target, 0.0,
                 lambda r: side(r).rot)
    right += [
        Waypoint(0.6, FF_GRIP, FINGER_CLOSED, "settle", nsv),
        Waypoint(0.8, FF_GRIP, FINE_WIDE, "let go", nsv),
        Waypoint(1.0, FF_OUT, FINE_WIDE, "back off", nsv),
        # a second grip on the fine knob (0.2 mm per turn: one grip turns it about a third of a turn, 0.06 mm)
        Waypoint(1.0, FF_FAR, FINE_WIDE, "out to the side, clear of the stage", nsv),
        Waypoint(1.0, FF_FAR, FINE_WIDE, "unroll", side(0.0)),
        Waypoint(1.0, FF_OUT, FINE_WIDE, "back in to the fine knob", side(0.0)),
        Waypoint(1.2, FF_GRIP, FINE_WIDE, "onto the fine knob again", side(0.0)),
        Waypoint(0.8, FF_GRIP, FINGER_CLOSED, "grip it", side(0.0)),
    ]
    nsv2 = _servo(right, 8.0, FF_GRIP, FINGER_CLOSED, "fine focus: turn until sharp", "fine", fine_target, 0.0,
                  lambda r: side(r).rot)
    right += [
        Waypoint(0.6, FF_GRIP, FINGER_CLOSED, "settle", nsv2),
        Waypoint(0.8, FF_GRIP, FINE_WIDE, "let go", nsv2),
        Waypoint(1.0, FF_OUT, FINE_WIDE, "back off", nsv2),
        # well clear of the stage and the holder lever's handle before the wrist turns back (turning beside the knob
        # swept the fingers into the lever and nudged the slide 0.2 mm)
        Waypoint(1.0, FF_FAR, FINE_WIDE, "out to the side, clear of the stage", nsv2),
        Waypoint(1.0, FF_FAR, FINE_WIDE, "unroll", side(0.0)),
        Waypoint(1.5, FF_FAR_UP, KNOB_WIDE, "point down again"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"),
        Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "hold"),
    ]
    left.append(Waypoint(max(0.5, _dur(right)), LEFT_PARK, FINGER_OPEN, "rest"))
    return {"left": left, "right": right}
