"""Replacing the passenger-side cabin air filter element behind a 2002-2006 Honda CR-V's glove box.

A mock-up of the CR-V's passenger dash stands on the table, its glove box unlatched and resting open on its two
stoppers. The filter's passenger-side element sits in its bay behind it, reachable only once the box swings down. Each
stopper is a separate clip on the outside of a side wall whose catch hooks on the dash: squeezed by its tab and pushed
into the box it clears and stays in, and with both in the box drops. The old element slides out by its cartridge's
pull tab and the new one slides in. Pulling the box down past a stopper that is not pushed in breaks it.

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
X, Y, Z = np.eye(3)

# -- the dash mock-up (fixed) ----------------------------------------------------------------------------------------
# Part frame: origin on the table under the glove box's hinge, x into the dash (away from the robots), y to the left,
# z up. The box is shown closed here (its door vertical); it starts swung open onto its stops.
DASH_X = 0.32  # the hinge line's world x
GREY, DARK, BLACK = (0.55, 0.56, 0.58), (0.30, 0.31, 0.33), (0.12, 0.12, 0.13)
HZ = 0.030  # the hinge's height above the table
W = 0.175  # the box's half width (side walls' outer faces)
DEPTH, HEIGHT = 0.120, 0.160  # the box: front to back, and its side and back walls' height
DOOR_H = 0.270  # the door is taller than the box: closed, it covers the filter slot behind it
WALL, DOOR_T = 0.0025, 0.010  # wall and door thickness
CHEEK_IN = 0.183  # the dash cheeks' inner faces (y), 8 mm clear of the box's sides
CHEEK_T, DASH_D, DASH_H = 0.012, 0.200, 0.400
CHEEK_H = DASH_H  # the cheeks run up to the dash's top: the filter is reached only through the glove box's opening
LUG = (0.115, 0.090)  # each stopper's catch: its centre on a side wall, from the hinge, back (x) and up (z)
LUG_S = (0.010, 0.006, 0.010)  # catch size: along x, out from the wall (y), along z
STOP_ANG = math.radians(45.0)  # the box rests open this far, its stoppers' catches on the dash's stops
# Each stopper (CR-V 77547-S9A-G01) is a separate clip on the outside of a side wall: a strip carrying the catch, with a
# tab standing above the wall's top edge. Pushed 10 mm into the box (its tabs squeezed), the catch clears the dash's stop
# and the clip stays in (it is modelled as latching there); with both in, the box drops.
STRIP_Z0, TAB_H, STRIP_T, STRIP_W = 0.075, 0.035, 0.003, 0.020  # strip from this height up to TAB_H above the wall's
#   top, 3 mm thick (set into the wall's outer face, so only the catch reaches the dash's stop), 20 mm wide
SQUEEZE = 0.010  # how far a stopper is pushed in to clear (and then stays)
LATCH_AT = 0.007  # pushed in this far, a stopper is in and stays
PANEL_K = 300.0  # N/m: 3 N to push a stopper fully in


def _rot_y(a, p):
    """p (x, z from the hinge) turned by a (rad) about the hinge's +y axis."""
    c, s = math.cos(a), math.sin(a)
    return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1]])


# Each stop: a block on a cheek's inner face, just ahead (towards the robots) of where the lug sits at STOP_ANG.
_lug = _rot_y(-STOP_ANG, LUG)
_lead = _lug[0] - (LUG_S[0] + LUG_S[2]) / 2 * math.cos(math.pi / 4)  # the turned lug's front edge
STOP_X1 = _lead - 0.0005
STOP_Z = HZ + _lug[1]
STOP_Y0 = W + 0.0015  # the stop's inner face: 1.5 mm clear of the side wall, so the lug (6 mm proud) catches 4.5 mm

# The filter housing behind the box (CR-V): two trays side by side, each holding one element 112 mm wide and 226 mm deep
# into the housing. Its access opening (the slot) is in front of the passenger-side tray, centred at y = BAY_Y (the
# robots' right); the driver-side tray, with its element, sits closed behind the back wall and is not part of the task.
# The slot's floor is SLOT_Z0 up.
F_L, F_W, F_T = 0.226, 0.112, 0.029  # along the tray (into the housing), across, thick: FRAM CF10135 / Honda 80292-S5D-416
BAY_Y = -0.058
SLOT_X, SLOT_Z0, SLOT_H, SLOT_W = 0.190, 0.230, 0.035, F_W + 0.006  # (3 mm of play a side)
HOUSING_D = F_L + 0.0025  # the tray's floor, roof and walls, from 10 mm inside the slot's face
BACK_STOP = 0.002 + F_L + 0.0005  # the tray's back stop (its face, from the slot's face): a seated element's far end
#   0.5 mm short of it, its near end 2 mm inside the slot (t26b: with the stop 10.5 mm deeper, home was not the seat)
_ys0, _ys1 = BAY_Y - SLOT_W / 2, BAY_Y + SLOT_W / 2  # the slot's sides
DASH_BOXES = [
    ((DASH_D / 2, CHEEK_IN + CHEEK_T / 2, CHEEK_H / 2), (DASH_D, CHEEK_T, CHEEK_H), GREY),  # cheeks
    ((DASH_D / 2, -CHEEK_IN - CHEEK_T / 2, CHEEK_H / 2), (DASH_D, CHEEK_T, CHEEK_H), GREY),
    ((DASH_D / 2, 0.0, DASH_H + 0.005), (DASH_D, 2 * (CHEEK_IN + CHEEK_T), 0.010), GREY),  # top
    # back wall round the slot (below, above, left and right of it)
    ((SLOT_X + 0.005, 0.0, (SLOT_Z0 - 0.008) / 2), (0.010, 2 * CHEEK_IN, SLOT_Z0 - 0.008), DARK),
    ((SLOT_X + 0.005, 0.0, (SLOT_Z0 + SLOT_H + 0.008 + DASH_H) / 2), (0.010, 2 * CHEEK_IN, DASH_H - SLOT_Z0 - SLOT_H - 0.008), DARK),
    ((SLOT_X + 0.005, (_ys1 + CHEEK_IN) / 2, SLOT_Z0 + SLOT_H / 2), (0.010, CHEEK_IN - _ys1, SLOT_H + 0.016), DARK),
    ((SLOT_X + 0.005, (_ys0 - CHEEK_IN) / 2, SLOT_Z0 + SLOT_H / 2), (0.010, CHEEK_IN + _ys0, SLOT_H + 0.016), DARK),
    # the housing behind: floor and roof over both trays, outer sides, the divider and the back stop
    ((SLOT_X + 0.010 + HOUSING_D / 2, -BAY_Y / 2 + BAY_Y / 2, SLOT_Z0 - 0.004), (HOUSING_D, 2 * SLOT_W + 0.030, 0.008), BLACK),
    ((SLOT_X + 0.010 + HOUSING_D / 2, 0.0, SLOT_Z0 + SLOT_H + 0.004), (HOUSING_D, 2 * SLOT_W + 0.030, 0.008), BLACK),
    ((SLOT_X + 0.010 + HOUSING_D / 2, _ys0 - 0.004, SLOT_Z0 + SLOT_H / 2), (HOUSING_D, 0.008, SLOT_H), BLACK),
    ((SLOT_X + 0.010 + HOUSING_D / 2, _ys1 + 0.0035, SLOT_Z0 + SLOT_H / 2), (HOUSING_D, 0.007, SLOT_H), BLACK),  # divider
    ((SLOT_X + 0.010 + HOUSING_D / 2, -BAY_Y + SLOT_W / 2 + 0.011, SLOT_Z0 + SLOT_H / 2), (HOUSING_D, 0.008, SLOT_H), BLACK),
    ((SLOT_X + BACK_STOP + 0.004, 0.0, SLOT_Z0 + SLOT_H / 2), (0.008, 2 * SLOT_W + 0.030, SLOT_H + 0.016), BLACK),
    # the driver-side element in its tray (fixed; left in place)
    ((SLOT_X + 0.002 + F_L / 2, -BAY_Y + 0.007, SLOT_Z0 + F_T / 2 + 0.0005), (F_L, F_W, F_T), (0.52, 0.47, 0.40)),
]
for _s in (1, -1):  # the two stops
    DASH_BOXES.append((((0.0 + STOP_X1) / 2, _s * (STOP_Y0 + CHEEK_IN) / 2, STOP_Z),
                       (STOP_X1, CHEEK_IN - STOP_Y0, 0.024), BLACK))

# The box: door, floor, back and the side walls' lower halves turn on the hinge; the upper halves (with the lugs)
# ride on it and slide in when squeezed, sprung back out.
TAN = (0.80, 0.76, 0.68)
BOX_BOXES = [
    ((-DOOR_T / 2, 0.0, HZ + DOOR_H / 2), (DOOR_T, 2 * W, DOOR_H), TAN),  # door
    ((-DOOR_T - 0.004, 0.0, HZ + DOOR_H - 0.040), (0.008, 0.110, 0.016), DARK),  # handle
    ((DEPTH / 2, 0.0, HZ + WALL / 2), (DEPTH, 2 * W, WALL), TAN),  # floor
    ((DEPTH - WALL / 2, 0.0, HZ + HEIGHT / 2), (WALL, 2 * W, HEIGHT), TAN),  # back
    ((DEPTH / 2, W - WALL / 2, HZ + HEIGHT / 2), (DEPTH, WALL, HEIGHT), TAN),  # side walls
    ((DEPTH / 2, -W + WALL / 2, HZ + HEIGHT / 2), (DEPTH, WALL, HEIGHT), TAN),
]


def _stopper(name, s):
    """A side wall's stopper (s = +1 left, -1 right): strip, tab and catch on the wall's outside. Pushed in = towards
    the box's middle."""
    yw = s * (W - STRIP_T / 2)
    zc = HZ + (STRIP_Z0 + HEIGHT + TAB_H) / 2
    boxes = [((LUG[0], yw, zc), (STRIP_W, STRIP_T, HEIGHT + TAB_H - STRIP_Z0), DARK),
             ((LUG[0], s * (W + LUG_S[1] / 2), HZ + LUG[1]), LUG_S, DARK)]
    lo, hi = (-SQUEEZE, 0.0) if s > 0 else (0.0, SQUEEZE)
    return Joint(name, kind="slide", axis="Y", anchor=(LUG[0], yw, zc), boxes=boxes,
                 lower=lo, upper=hi, stiffness=PANEL_K, damping=4.0, target=0.0, initial=0.0, mass=0.010,
                 parent="box")


STOPPER_L, STOPPER_R = _stopper("panel_left", 1), _stopper("panel_right", -1)


OPEN_LIMIT = math.radians(92.0)  # how far the box swings once past its stops (it hangs on its hinge)
BOX = Joint("box", kind="hinge", axis="Y", anchor=(0.0, 0.0, HZ), boxes=BOX_BOXES, lower=-OPEN_LIMIT, upper=0.0,
            damping=0.02, initial=-(STOP_ANG - math.radians(0.4)), mass=0.55)
DASH = part("dash", pos=(DASH_X, 0.0, TOP), fixed=True, boxes=DASH_BOXES,
            joints=[BOX, STOPPER_L, STOPPER_R], friction=0.5,
            self_collide=True)  # (the stoppers' catches catch on the stops, on the dash)

# -- the cabin air filter (loose) ------------------------------------------------------------------------------------
# Part frame: origin at the element's middle; x along its length (into the housing), z up. One CR-V element (F_L x F_W x
# F_T above), pleated paper in its plastic case, with the case's long thin pull tab on the end that faces out.
TAB_L, TAB_W, TAB_T = 0.020, 0.040, 0.0025
F_MASS = 0.060  # an element in its cartridge (estimate)
WHITE, PLEAT = (0.93, 0.93, 0.90), (0.80, 0.80, 0.74)
DIRTY, DIRTY_PLEAT = (0.52, 0.47, 0.40), (0.38, 0.34, 0.29)
ARROW = (0.15, 0.35, 0.75)


def _filter_boxes(frame, pleat):
    out = [((0.0, 0.0, 0.0), (F_L, F_W, F_T), frame),
           ((-F_L / 2 - TAB_L / 2, 0.0, 0.0), (TAB_L, TAB_W, TAB_T), frame)]
    out += [((0.0, -F_W / 2 + 0.011 + 0.015 * i, F_T / 2 + 0.0005), (F_L - 0.010, 0.006, 0.001), pleat) for i in range(7)]  # (within the element)
    out += [((-F_L / 2 + 0.030, F_W / 2 + 0.0005, 0.0), (0.030, 0.001, 0.004), ARROW)]  # airflow mark, left edge
    return out


F_SEAT_X = SLOT_X + F_L / 2 + 0.002  # seated: 2 mm inside the slot's face
F_Z = SLOT_Z0 + F_T / 2 + 0.0005
OLD = part("old_filter", pos=(DASH_X + F_SEAT_X, BAY_Y, TOP + F_Z), mass=F_MASS, friction=0.5,
           boxes=_filter_boxes(DIRTY, DIRTY_PLEAT))
# The new filter waits on a stand left of the dash, at the slot's height, its tab towards the robots.
STAND_XY = np.array([0.090 + F_L / 2, -0.470])  # right of the dash, all of the element in front of its face
#   (its far end 4 mm short of it), clear of the right hand at its stopper
STAND_H = 0.189  # (the element on it 40 mm below the slot)
STAND = part("filter_stand", pos=(STAND_XY[0], STAND_XY[1], TOP), fixed=True,
            boxes=[((0.0, 0.0, STAND_H / 2), (0.120, 0.060, STAND_H), GREY)])  # (the element overhangs it)
NEW = part("new_filter", pos=(STAND_XY[0], STAND_XY[1], TOP + STAND_H + F_T / 2 + 0.0005), mass=F_MASS, friction=0.5,
           boxes=_filter_boxes(WHITE, PLEAT))

OBJECTS = [DASH, OLD, STAND, NEW]

# -- scoring --------------------------------------------------------------------------------------------------------
UP_AGAIN = math.radians(46.0)  # back up past its stops: open no more than this (resting on them, or closed)
FORCE_WIN = (math.radians(46.5), math.radians(54.0))  # in here each lug overlaps its stop by over 1 mm
LUG_CLEAR = 0.0045  # a panel pushed in less than this there means its lug was forced past the stop (breaking it)
OUT_GAP = 0.010  # the old filter is out once all of it is at least this far in front of the slot's face
SEAT_TOL, SIDE_TOL, YAW_TOL_DEG = 0.005, 0.003, 3.0  # the new one seated: within 5 mm of home, 3 mm up, 3 deg
ACROSS_TOL = (SLOT_W - F_W) / 2 + 0.001  # and across: the tray's play (3 mm a side) plus 1 mm
RELEASE = 0.08  # every fingertip at least this far from the nearest surface of either filter or the box
STILL_STEPS, STILL_TOL = 21, 0.001  # 21 samples (1 s at 20 Hz): every tracked surface point moved under 1 mm
PASSED = FORCE_WIN[1]  # the box has passed its stops once it is open more than this
MOVED = 0.005  # the old filter has been drawn from its housing once its centre is this far from where it started


def _in_housing(dash, filt):
    """Any of the filter past the slot's face, within the slot's width and height (it has gone into the housing)."""
    for p in ((F_L / 2, 0.0, 0.0), (F_L / 2, F_W / 2, 0.0), (F_L / 2, -F_W / 2, 0.0)):
        q = dash.to_local(filt.to_world(p))
        if q[0] > SLOT_X and abs(float(q[1]) - BAY_Y) < SLOT_W / 2 + 0.005 and SLOT_Z0 - 0.005 < q[2] < SLOT_Z0 + SLOT_H + 0.005:
            return True
    return False


PART_IN = 0.020  # partly in: both far corners at least this far into the slot


def _partly_in(dash, filt):
    """The new filter started into its slot and resting there: both far corners at least PART_IN past the slot's face,
    inside its width and height, the filter top side up and level within YAW_TOL_DEG + 2 deg (it lies on the slot's
    floor, not hanging off a lip). With the hands clear and everything still, it is supported."""
    for sy in (1, -1):
        q = dash.to_local(filt.to_world((F_L / 2, sy * F_W / 2, 0.0)))
        if not (q[0] >= SLOT_X + PART_IN and abs(float(q[1]) - BAY_Y) <= SLOT_W / 2 and SLOT_Z0 <= q[2] <= SLOT_Z0 + SLOT_H):
            return False
    rel = dash.rotation().T @ filt.rotation()
    return float(rel[2, 2]) >= math.cos(math.radians(YAW_TOL_DEG + 2.0))


def _box_angle(dash):
    """How far the box is open (rad, positive)."""
    return -float(dash.joints["box"])


def _squeezed(dash):
    return abs(float(dash.joints["panel_left"])), abs(float(dash.joints["panel_right"]))


# A filter's two boxes in its own frame: the frame (body) and the pull tab, as (centre, half sizes).
_F_BOXES = [(np.zeros(3), np.array([F_L / 2, F_W / 2, F_T / 2])),
            (np.array([-F_L / 2 - TAB_L / 2, 0.0, 0.0]), np.array([TAB_L / 2, TAB_W / 2, TAB_T / 2]))]
_F_CORNERS = [c + h * np.array([sx, sy, sz]) for c, h in _F_BOXES for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def _filter_out(dash, filt):
    """All of the filter, its tab's corners included, at least OUT_GAP in front of the slot's face."""
    hi_x = max(float(dash.to_local(filt.to_world(tuple(p)))[0]) for p in _F_CORNERS)
    return hi_x <= SLOT_X - OUT_GAP


_BOX_PTS = ([np.array([x, y, HZ + z]) for x in (0.0, DEPTH / 2, DEPTH) for y in (-W, 0.0, W) for z in (0.0, HEIGHT / 2, HEIGHT)]
            + [np.array([0.0, y, HZ + z]) for y in (-W, 0.0, W) for z in (0.21, DOOR_H)])


def _box_world(dash):
    a = float(dash.joints["box"])
    c, s = math.cos(a), math.sin(a)
    out = []
    for p in _BOX_PTS:
        x, z = p[0], p[2] - HZ
        out.append(dash.to_world((c * x + s * z, p[1], HZ - s * x + c * z)))
    return out


def _seated(dash, filt):
    """The new filter home in its housing: within SEAT_TOL of its seat along the slot, ACROSS_TOL across, SIDE_TOL up,
    square to the slot and top side up (its airflow mark then points the right way)."""
    c = dash.to_local(filt.pos)
    rel = dash.rotation().T @ filt.rotation()
    yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
    return (abs(float(c[0]) - F_SEAT_X) <= SEAT_TOL and abs(float(c[1]) - BAY_Y) <= ACROSS_TOL
            and abs(float(c[2]) - F_Z) <= SIDE_TOL and abs(yaw) <= YAW_TOL_DEG
            and float(rel[2, 2]) >= math.cos(math.radians(YAW_TOL_DEG)))


def _filter_pts(filt):
    """The filter's box corners, tab included (world): tracked for stillness."""
    return [np.asarray(filt.to_world(tuple(p)), float) for p in _F_CORNERS]


def _box_dist(q, c, h):
    """Distance from point q to the box centred at c with half sizes h (all in the box's frame); 0 inside."""
    d = np.maximum(np.abs(np.asarray(q, float) - c) - h, 0.0)
    return float(np.linalg.norm(d))


# The glove box's own boxes, in the dash frame with the box shut (hinge at (0, 0, HZ)); the side walls' upper halves
# are taken where they rest (pushed in at most 10 mm when squeezed, so this errs on the near side).
_GB_BOXES = [(np.asarray(b[0], float), np.asarray(b[1], float) / 2) for b in BOX_BOXES] + [
    (np.array([LUG[0], s * (W - STRIP_T / 2), HZ + (STRIP_Z0 + HEIGHT + TAB_H) / 2]),
     np.array([STRIP_W / 2, STRIP_T / 2, (HEIGHT + TAB_H - STRIP_Z0) / 2])) for s in (1, -1)]  # (the stoppers)
_HINGE = np.array([0.0, 0.0, HZ])


def _clearance(dash, filters, tips):
    """The nearest distance (m) from any fingertip to the surface of either filter (frame and tab) or the glove box."""
    a = float(dash.joints["box"])
    c, s = math.cos(a), math.sin(a)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])  # the box's turn about its hinge (as _box_world)
    best = float("inf")
    for t in tips:
        t = np.asarray(t, float)
        for f in filters:
            q = np.asarray(f.to_local(t), float)
            best = min(best, min(_box_dist(q, cb, hb) for cb, hb in _F_BOXES))
        q = Ry.T @ (np.asarray(dash.to_local(t), float) - _HINGE) + _HINGE
        best = min(best, min(_box_dist(q, cb, hb) for cb, hb in _GB_BOXES))
    return best


_w = {"bad": "", "hist": [], "still": False, "far": False, "passed": False, "home": None, "in": set(), "ids": None}


def _hold_stoppers(env, dash):
    """A stopper pushed in past LATCH_AT stays in: from then on its spring aims at fully in, every step."""
    import torch

    for jt, sgn in ((STOPPER_L, -1.0), (STOPPER_R, 1.0)):
        if abs(float(dash.joints[jt.name])) >= LATCH_AT:
            _w["in"].add(jt.name)
    if not _w["in"]:
        return
    art = env.unwrapped.scene["dash"]
    if _w["ids"] is None:
        _w["ids"] = {jt.name: list(art.joint_names).index(jt.joint_name) for jt in (STOPPER_L, STOPPER_R)}
    for jt, sgn in ((STOPPER_L, -1.0), (STOPPER_R, 1.0)):
        if jt.name in _w["in"]:
            art.set_joint_position_target(torch.full((1, 1), sgn * SQUEEZE, device=art.device),
                                          joint_ids=[_w["ids"][jt.name]])


@on_reset
def _w_reset(env):
    _w["bad"], _w["hist"], _w["still"], _w["far"] = "", [], False, False
    _w["passed"], _w["home"], _w["in"] = False, None, set()


@on_step
def _watch(env):
    """Latches a stop forced (the box opened past a stop without that wall squeezed in), latches the box having
    passed its stops, latches the old filter drawn from its housing or the new one put into it before that (picking
    the new one up off its stand is fine), and tracks release and stillness."""
    dash = labkit._part_state(env, DASH)
    old = labkit._part_state(env, OLD)
    new = labkit._part_state(env, NEW)
    _hold_stoppers(env, dash)
    a = _box_angle(dash)
    if FORCE_WIN[0] < a < FORCE_WIN[1] and min(_squeezed(dash)) < LUG_CLEAR and not _w["bad"]:
        _w["bad"] = "a stop forced"
    if _w["home"] is None:
        _w["home"] = np.asarray(old.pos, float).copy()
    if a > PASSED and not _w["bad"]:
        _w["passed"] = True
    if not _w["passed"] and not _w["bad"]:
        if float(np.linalg.norm(np.asarray(old.pos, float) - _w["home"])) > MOVED:
            _w["bad"] = "the old filter was drawn from its housing before the box passed its stops"
        elif _in_housing(dash, new):
            _w["bad"] = "the new filter went into the housing before the box passed its stops"
    _w["far"] = _clearance(dash, (old, new), _tips(env)) >= RELEASE
    snap = np.array(_filter_pts(new) + _filter_pts(old) + [np.asarray(p, float) for p in _box_world(dash)])
    h = _w["hist"]
    h.append(snap)
    del h[:-STILL_STEPS]
    _w["still"] = (len(h) == STILL_STEPS
                   and max(float(np.max(np.linalg.norm(p - h[-1], axis=1))) for p in h) < STILL_TOL)


def score(objects):
    if _w["bad"] or not _w["passed"]:
        return 0.0
    dash, old, new = objects["dash"], objects["old_filter"], objects["new_filter"]
    if not (_w["far"] and _w["still"]):
        return 0.0  # every positive score: hands clear of both filters and the box, and everything at rest
    if not _filter_out(dash, old):
        return 0.0
    if _seated(dash, new):
        return 1.0
    return 0.5 if _partly_in(dash, new) else 0.25


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    """How far (m) the new filter is from its seat."""
    dash, new = objects["dash"], objects["new_filter"]
    return float(np.linalg.norm(dash.to_local(new.pos) - np.array([F_SEAT_X, BAY_Y, F_Z])))


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Both hands pinch the two side walls' top edges, squeeze them in together and carry the box down past its stops, then
# let it swing down. The right hand turns to point into the filter slot, pinches the pull tab and draws the filter out.
# Everything it aims at is fixed scenery or seen: the dash and the box's and filter's start.


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
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        q7, q8 = (abs(float(v)) for v in robot.data.joint_pos.torch[0, ids].cpu().numpy())
        out += [mid - q7 * g[:3, 0], mid + q8 * g[:3, 0]]  # each jaw on its own: joint7's pad towards -x, joint8's +x
    return out + [(out[0] + out[1]) / 2, (out[2] + out[3]) / 2]


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

    def window(self, t0, t1, key, fn, watch, still=True):
        _ct["wins"].append((t0, t1, key, fn, watch if still else None))

    def gate(self, t0, t1, key, cond):
        _ct["gates"].append((t0, t1, key, cond))


# trims: each a 3-vector added to the TCPs of _Dyn waypoints with its key; integrated in windows from seen positions
_ct = {"trim": {}, "wins": [], "last": {}, "gates": [], "gate": {}}


@on_reset
def _reset_ct(env):
    _ct["trim"] = {}
    _ct["last"] = {}
    _ct["gate"] = {}


@on_step
def _gtrack(env):
    """Gates: from t0 on, latch ``key`` once ``cond`` sees what it waits for (at t1 at the latest)."""
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, cond in _ct["gates"]:
        if t0 <= t and not _ct["gate"].get(key) and (t >= t1 or cond(env)):
            _ct["gate"][key] = t


class _Gate:
    """A TCP that is ``a`` until the gate ``key`` latches, then ``b``."""

    def __init__(self, a, b, key):
        self.a, self.b, self.key = np.asarray(a, float), np.asarray(b, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.b if _ct["gate"].get(self.key) else self.a, dtype=dtype or float)


def _seen(env, name):
    return np.asarray(labkit._part_state(env, labkit._SPECS[name]).pos, float)


@on_step
def _ctrack(env):
    t = getattr(_demo, "_plan_clock", 0.0)
    for t0, t1, key, fn, watch in _ct["wins"]:
        if not t0 <= t < t1:
            continue
        if watch is not None:
            p = _seen(env, watch)
            last = _ct["last"].get(key)
            _ct["last"][key] = p
            if last is None or np.max(np.abs(p - last)) > 0.0005:
                continue  # only while what is watched holds still (a window with no watch also runs while it moves)
        e = np.asarray(fn(env), float)
        tr = _ct["trim"].get(key, np.zeros(3))
        step, lim = TRIM_LIMITS.get(key, (0.0003, 0.020))
        _ct["trim"][key] = np.clip(tr + np.clip(0.2 * e, -step, step), -lim, lim)


# per-key (most per step, most in all), m: the tab is taken where the camera sees it, which after the far-end-in release
#    can be 40 mm lower than planned (t24: the element tipped 10 deg about the slot's lip once let go)
TRIM_LIMITS = {"tab": (0.0008, 0.060)}


class _Dyn:
    """A TCP plus the live trim ``key``."""

    def __init__(self, tcp, key):
        self.tcp, self.key = np.asarray(tcp, float), key

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.tcp + _ct["trim"].get(self.key, np.zeros(3)), dtype=dtype or float)


GRIP_OPEN = -0.012  # 24 mm between the pads
R0 = _prev_grip(0.0)  # straight down, the fingers closing along y
HORIZ = axis_angle_matrix(X, math.radians(90.0)) @ axis_angle_matrix(Y, math.radians(-90.0)) @ R0  # pointing +x, closing along z
HINGE = np.array([DASH_X, 0.0, TOP + HZ])
FRONT = axis_angle_matrix(Y, math.radians(-90.0)) @ R0  # pointing +x, closing along y (across a stopper's tab)
TAB_IN = 0.012  # the pads' tips this far past the tab's front edge
SQUEEZE_CMD = 0.020  # each hand moves in this far (the stopper stops at 10 mm; the arm gives a little)
SWING_TO = math.radians(62.0)  # the hands carry the squeezed box this far open before letting go

def _turn(a):
    """The box's rotation when open by a (rad)."""
    return axis_angle_matrix(Y, -a)


def _tab_point(side, a):
    """World point where the pads' tips pinch a stopper's tab (mid-height, just past its front edge), box open by a."""
    x, z = _rot_y(-a, (LUG[0] - STRIP_W / 2 + TAB_IN, HEIGHT + 0.022))
    return np.array([DASH_X + x, side * (W - STRIP_T / 2), TOP + HZ + z])


_TIP0 = {side: _tab_point(side, STOP_ANG) for side in (1, -1)}


def _pinch(side, a, inward=0.0):
    """TCP and hand rotation pinching a stopper's tab at the box's opening a, moved in by ``inward``."""
    rel = _turn(a - STOP_ANG)
    tip = HINGE + rel @ (_TIP0[side] - HINGE) - side * inward * Y
    return tip - rel @ (TIP_ABOVE_TCP * X), Orient(rel @ FRONT)


def _tab_path(side, a, inward=0.0):
    """TCP following a stopper's tab as the box opens to a, the hand's turn held as it was at the stops."""
    rel = _turn(a - STOP_ANG)
    tip = HINGE + rel @ (_TIP0[side] - HINGE) - side * inward * Y
    return tip - TIP_ABOVE_TCP * X


CORNER_OPEN = -0.023  # 46 mm between the pads, round an element's 29 mm frame
SEAT_NEAR = DASH_X + SLOT_X + 0.002  # world x of a seated element's near end (2 mm inside the slot)
LEDGE_NEAR = DASH_X + SLOT_X + 0.013 - F_L / 2  # drawn this far, an element still rests in its tray on its own (its
#   centre 13 mm inside the slot), so the hand can take it again
DROP_NEAR = 0.150  # the old element is let go with its near end here, its centre over the hanging box's back (its far
#   end over the cavity, well in front of the slot), after being lowered DROP_DZ: it drops flat onto the box
DROP_DZ = -0.030
TAB_TIP = 0.006  # a tab pinch: the pads' tips this far in front of the element's near end (on its 20 mm tab)
NOSE_YAW = math.radians(30.0)  # a nose pinch on the tab turns the hand this far towards the other arm's side (its
#   forearm then clears the dash's walls and it reaches close to the robots)
PICK_YAW = math.radians(25.0)  # the right hand's pinch on the new element's inner edge turns out this far from the
#   element's own axis (its gripper then stays outside the element)
EDGE_AT, EDGE_IN = 0.150, 0.006  # that pinch: this far behind the element's near end; the pads' centre 6 mm inside
#   its edge (the pads overlap it; turned 25 deg, the gripper body stays outside)
PICK_OPEN = -0.035  # 70 mm between the pads for that pinch (open round the 29 mm edge with room to spare)
NEW_NEAR = 0.085  # the new element is carried across with its near end here: its far end 9 mm in front of the dash
CROSS_TIP_Y = BAY_Y - 0.025  # carried across until its centre is 25 mm right of the tray's line (all of it within the
#   opening), then in and over together
LINE_NEAR = 0.250  # the right hand lines the new element up with its near end here (its far end 34 mm short of the
#   slot), its far end's centre corrected onto the tray's line from where the camera sees it
RATCHET = 0.060  # each step along the edge (two of them)
SLIDE_OPEN = -0.019  # the pads 38 mm apart round the 29 mm edge while they slide back along it
HAND_NEAR = 0.330  # then pushes its far end 46 mm into the tray (the tray's walls square it in the pinch, a free pivot),
#   lets go of the edge and takes it again by its tab
LEFT_REST = np.array([0.220, 0.130, TOP + 0.310])  # (pointing into the dash, in front of it, clear of the elements)
LEFT_AWAY = np.array([0.180, 0.250, TOP + 0.280])  # out to the left, once the new element rests in its tray: the right
#   arm's wrist swung into the left hand at LEFT_REST while turning to the tab (final1, about 88 s)
RIGHT_WAIT = np.array([0.220, -0.130, TOP + 0.310])  # (pointing into the dash, up and clear of the old element)
STAGE_X = 0.200  # a hand lines up in front of the dash's face here (its tips' x) before going straight in
NEAR2 = HAND_NEAR + 2 * RATCHET  # where its near end is after the two steps: its centre 40 mm inside the slot, resting



def _yawed(hand, yaw):
    """Hand rotation pointing into the dash, turned yaw towards the other arm's side (hand +1 left, -1 right), closing
    along z."""
    return axis_angle_matrix(Z, -hand * yaw) @ HORIZ


def _dir(hand, yaw):
    return np.array([math.cos(yaw), -hand * math.sin(yaw), 0.0])


def _tab(near, dz=0.0):
    """TCP pinching an element's tab (hand pointing straight into the slot), its near end at x = near."""
    tip = np.array([near - TAB_TIP, BAY_Y, TOP + F_Z + dz])
    return tip - TIP_ABOVE_TCP * X


def _far_error(env):
    """The tray's line minus where the camera sees the new element's far end (its centre, across only): the hand moves
    the element by this much (t21 and t22 had it the other way round, which walked the far end off the line)."""
    st = labkit._part_state(env, NEW)
    seen = np.asarray(st.to_world((F_L / 2, 0.0, 0.0)), float)
    return np.array([0.0, BAY_Y - seen[1], 0.0])


YAW_K = 10.0  # the "yaw" trim is kept in tenths: a trim of 0.020 turns the hand 0.2 rad (11 deg)


def _yaw_error(env):
    """How far the camera sees the new element turned from the tray's line (rad, anticlockwise), in the trim's units.
    Half the error per step (the pinch lags the hand)."""
    st = labkit._part_state(env, NEW)
    ax = np.asarray(st.to_world((1.0, 0.0, 0.0)), float) - np.asarray(st.to_world((0.0, 0.0, 0.0)), float)
    return np.array([0.5 * math.atan2(ax[1], ax[0]) / YAW_K, 0.0, 0.0])


class _DynO(Orient):
    """The right hand's rotation holding the new element at psi (deg), less the live "yaw" trim."""

    def __new__(cls, psi):
        obj = Orient.__new__(cls, np.eye(3))
        obj.psi = psi
        return obj

    @property
    def rot(self):
        c = math.radians(self.psi) - YAW_K * float(_ct["trim"].get("yaw", np.zeros(3))[0])
        return _yawed(-1, -(PICK_YAW - c))

    @rot.setter
    def rot(self, v):
        pass


def _tab_error(env):
    """Where the camera sees the new element's tab (the point the pads' tips aim at) minus where the plan expects it."""
    st = labkit._part_state(env, NEW)
    seen = np.asarray(st.to_world((-F_L / 2 - TAB_TIP, 0.0, 0.0)), float)
    return seen - np.array([HAND_NEAR - TAB_TIP, BAY_Y, TOP + F_Z])


def _nose(near, hand, dz=0.0):
    """TCP pinching an element's tab with the hand turned NOSE_YAW towards the other arm's side."""
    tip = np.array([near - TAB_TIP - 0.002, BAY_Y, TOP + F_Z + dz])
    return tip - TIP_ABOVE_TCP * _dir(hand, NOSE_YAW)


def _carry(near, cy, psi, dz=0.0):
    """TCP and hand rotation of the right hand holding the new element by its inner (left) edge, the element's near end
    at x = near, its centre at y = cy, turned psi (deg, anticlockwise from straight into the dash) and raised dz from the
    slot's height. The hand turns with the element: PICK_YAW out from the element's axis."""
    c = math.radians(psi)
    rz = np.array([[math.cos(c), -math.sin(c)], [math.sin(c), math.cos(c)]])
    ctr = np.array([near + F_L / 2 * math.cos(c), cy])
    tip = ctr + rz @ np.array([EDGE_AT - F_L / 2, F_W / 2 - EDGE_IN])
    yaw = PICK_YAW - c  # (outward from straight into the dash)
    rot = _yawed(-1, -yaw)
    return np.array([tip[0], tip[1], TOP + F_Z + dz]) - TIP_ABOVE_TCP * _dir(-1, -yaw), Orient(rot)


def demo():
    from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS

    P = _Plan()
    up = Z
    side_of = {"left": 1, "right": -1}
    a0 = STOP_ANG

    def both(dur, fn, finger, label):
        ws = {}
        for arm, s in side_of.items():
            tcp, o = fn(s)
            ws[arm] = [(dur, tuple(tcp), finger, label, o)]
        P.phase(right=ws["right"], left=ws["left"])

    # 1. the left hand pinches the left stopper's tab and pushes the stopper into the box (it stays in; the box still hangs
    #    on the right one), lets go and rests; the right hand does the same with the right stopper and, the box now
    #    free, carries it down past the stops and lets it swing down
    o_rest = Orient(HORIZ)

    def push(s):
        return [(2.0, (0.200, s * 0.260, TOP + 0.300), GRIP_OPEN, "hand turned, in front of the dash", _pinch(s, a0)[1]),
                (1.5, tuple(_pinch(s, a0)[0] - 0.060 * X + 0.030 * up), GRIP_OPEN, "in line with the stopper's tab",
                 _pinch(s, a0)[1]),
                (1.2, tuple(_pinch(s, a0)[0] - 0.040 * X), GRIP_OPEN, "in front of the stopper's tab", _pinch(s, a0)[1]),
                (1.2, tuple(_pinch(s, a0)[0]), GRIP_OPEN, "pads either side of the tab", _pinch(s, a0)[1]),
                (0.8, tuple(_pinch(s, a0)[0]), FINGER_CLOSED, "squeeze the stopper's tab", _pinch(s, a0)[1]),
                (1.5, tuple(_pinch(s, a0, SQUEEZE_CMD)[0]), FINGER_CLOSED, "push the stopper into the box",
                 _pinch(s, a0)[1]),
                (0.5, tuple(_pinch(s, a0, SQUEEZE_CMD)[0]), FINGER_CLOSED, "hold it in", _pinch(s, a0)[1])]

    P.phase(left=[*push(1),
                  (0.6, tuple(_pinch(1, a0, SQUEEZE_CMD)[0]), GRIP_OPEN, "let go: the stopper stays in", _pinch(1, a0)[1]),
                  (1.0, tuple(_pinch(1, a0, SQUEEZE_CMD)[0] - 0.060 * X), GRIP_OPEN, "back off", _pinch(1, a0)[1]),
                  (2.5, tuple(LEFT_REST), FINGER_OPEN, "up and clear, to rest", o_rest)],
            right=push(-1)[:2])
    P.phase(right=[*push(-1)[2:],
                   *[(1.0, tuple(_tab_path(-1, a0 + (SWING_TO - a0) * k / 3, SQUEEZE_CMD)), FINGER_CLOSED,
                      "carry the box down past its stops", _pinch(-1, a0)[1]) for k in (1, 2, 3)],
                   (0.6, tuple(_tab_path(-1, SWING_TO, SQUEEZE_CMD)), GRIP_OPEN, "let go: the box swings down",
                    _pinch(-1, a0)[1]),
                   (1.0, tuple(_tab_path(-1, SWING_TO, SQUEEZE_CMD) - 0.060 * X + 0.040 * up), GRIP_OPEN,
                    "back and up, off the tab", _pinch(-1, a0)[1])])
    # 2. the right hand pinches the old element's tab and draws it out onto the rails; takes the tab again, turned 45 deg
    #    towards the left, draws it out of the opening and lowers it onto the open glove box, where it lets it drop; then
    #    it waits up and clear. The left hand waits at its rest.
    o_h = Orient(HORIZ)
    o_n = Orient(_yawed(-1, NOSE_YAW))
    d_n = _dir(-1, NOSE_YAW)
    n0 = SEAT_NEAR
    stage = lambda t: t * np.array([0.0, 1.0, 1.0]) + (STAGE_X - TIP_ABOVE_TCP) * X  # (the same TCP, in front of the dash)
    P.phase(right=[(2.5, tuple(stage(_tab(n0)) + 0.030 * up), GRIP_OPEN, "hand turned to point into the slot", o_h),
                   (1.0, tuple(stage(_tab(n0))), GRIP_OPEN, "down to the tab's height", o_h),
                   (1.5, tuple(_tab(n0) - 0.030 * X), GRIP_OPEN, "straight in, in front of the tab", o_h),
                   (1.0, tuple(_tab(n0)), GRIP_OPEN, "pads above and below the tab", o_h),
                   (0.8, tuple(_tab(n0)), FINGER_CLOSED, "pinch the old element's tab", o_h),
                   *[(1.2, tuple(_tab(n0 + (LEDGE_NEAR - n0) * k / 2)), FINGER_CLOSED,
                      "draw it out until it rests in its tray on its own", o_h) for k in (1, 2)],
                   (0.6, tuple(_tab(LEDGE_NEAR)), GRIP_OPEN, "let go of the tab", o_h),
                   (1.0, tuple(_tab(LEDGE_NEAR) - 0.050 * X), GRIP_OPEN, "back off", o_h),
                   (2.0, tuple(_nose(LEDGE_NEAR, -1) - 0.120 * d_n + 0.040 * up), GRIP_OPEN,
                    "hand turned towards the left, in front of the opening", o_n),
                   (1.0, tuple(_nose(LEDGE_NEAR, -1) - 0.120 * d_n), GRIP_OPEN, "down to the tab's height", o_n),
                   (1.8, tuple(_nose(LEDGE_NEAR, -1)), GRIP_OPEN, "along its line, pads round the tab", o_n),
                   (0.8, tuple(_nose(LEDGE_NEAR, -1)), FINGER_CLOSED, "pinch the tab again", o_n),
                   *[(1.2, tuple(_nose(LEDGE_NEAR + (DROP_NEAR - LEDGE_NEAR) * k / 3, -1)), FINGER_CLOSED,
                      "draw it out of the opening", o_n) for k in (1, 2, 3)],
                   (1.0, tuple(_nose(DROP_NEAR, -1, DROP_DZ)), FINGER_CLOSED, "lower it over the open glove box", o_n),
                   (0.6, tuple(_nose(DROP_NEAR, -1, DROP_DZ)), GRIP_OPEN, "let go: it drops onto the box", o_n),
                   (1.2, tuple(_nose(DROP_NEAR, -1, DROP_DZ) - 0.050 * d_n + 0.060 * up), GRIP_OPEN, "back and up", o_n),
                   (2.0, tuple(RIGHT_WAIT), GRIP_OPEN, "to wait, up and clear", o_h)],
            left=[(2.5, tuple(LEFT_REST), FINGER_OPEN, "up and clear, to rest", o_h)])
    # 3. the right hand takes the new element by its inner edge (palm outside it, the forearm beside it), lifts it,
    #    carries it across in front of the dash turning it up to 30 deg so its far end rounds the opening's corner first,
    #    then straightens it in line with the tray as its far end goes into the slot; the left hand pinches its tab;
    #    the right lets go and waits; the left pushes it home by the tab
    c_s, n_s = STAND_XY[1], STAND_XY[0] - F_L / 2
    sdz = STAND_H + 0.001 - SLOT_Z0  # the element on its stand lies this far below the slot's height
    t_s, o_s = _carry(n_s, c_s, 0.0, sdz)
    path = [((n_s, c_s, 0.0, sdz + 0.030), "lift it off the stand"),
            ((NEW_NEAR, c_s, 0.0, sdz + 0.030), "back, all of it in front of the dash"),
            ((NEW_NEAR, -0.400, 0.0, -0.020), "across in front of the dash"),
            ((NEW_NEAR, -0.330, 10.0, -0.010), "across, turning its far end towards the opening"),
            ((NEW_NEAR, -0.290, 20.0, -0.005), "across, turning its far end towards the opening"),
            ((NEW_NEAR, -0.250, 30.0, 0.000), "its far end rounds the opening's corner"),
            ((NEW_NEAR, -0.190, 30.0, 0.004), "across, its far corner just inside the opening's edge"),
            ((0.120, -0.170, 30.0, 0.004), "into the opening, far end first"),
            ((0.150, -0.130, 30.0, 0.004), "into the opening, far end first"),
            ((0.200, -0.090, 20.0, 0.004), "in, straightening it towards the tray"),
            ((0.250, -0.065, 10.0, 0.004), "in, straightening it towards the tray"),
            ((LINE_NEAR, BAY_Y, 0.0, 0.004), "in line with the tray, its far end short of the slot")]
    P.phase(right=[(2.0, tuple(t_s + 0.025 * Y + 0.070 * up), PICK_OPEN, "above the new element's inner edge", o_s),
                   (1.2, tuple(t_s + 0.025 * Y), PICK_OPEN, "beside its inner edge", o_s),
                   (1.2, tuple(t_s), PICK_OPEN, "pads above and below the edge", o_s),
                   (0.8, tuple(t_s), FINGER_CLOSED, "pinch the new element's inner edge", o_s),
                   *[(1.2, tuple(_carry(*q)[0]), FINGER_CLOSED, lab, _carry(*q)[1]) for q, lab in path]])
    # hold still while the camera sees where the far end is, then push it in with that correction
    t_line = P.t
    t_l, o_l2 = _carry(LINE_NEAR, BAY_Y, 0.0, 0.004)
    # (the edge pinch lags the hand by a few degrees: t21 held it 3.9 deg turned and its far corner caught the slot's
    #    face, so the camera's view of its turn also trims the hand's turn)
    o_y = _DynO(0.0)
    P.window(t_line, t_line + 2.5, "far", _far_error, "new_filter")
    P.window(t_line, t_line + 2.5, "yaw", _yaw_error, "new_filter")
    # and while it pushes (t23: pushed from behind the pivot, the element turned 3.6 deg and jammed 3 mm in), the
    #    camera keeps both trims going, at the push's slower pace
    P.window(t_line + 2.5, t_line + 2.5 + 6.0, "far", _far_error, "new_filter", still=False)
    P.window(t_line + 2.5, t_line + 2.5 + 6.0, "yaw", _yaw_error, "new_filter", still=False)
    P.phase(right=[(2.5, _Dyn(t_l, "far"), FINGER_CLOSED, "hold still: line its far end up with the tray", o_y),
                   *[(1.5, _Dyn(_carry(LINE_NEAR + (HAND_NEAR - LINE_NEAR) * k / 4, BAY_Y, 0.0, 0.004)[0], "far"),
                      FINGER_CLOSED, "push its far end into the tray", o_y) for k in (1, 2, 3, 4)]])
    # then step along its edge without letting it drop (t24: let go with only its far end 44 mm in, it tipped 10 deg
    #    about the slot's lip and the tab regrip missed): open the pads a little, slide them 60 mm back along the edge,
    #    pinch again and push it 60 mm further, twice, so its centre of mass ends inside the tray and it rests there
    t_r = P.t
    back, fwd = _carry(HAND_NEAR - RATCHET, BAY_Y, 0.0, 0.004)[0], _carry(HAND_NEAR, BAY_Y, 0.0, 0.004)[0]
    one = [(0.6, fwd, SLIDE_OPEN, "open the pads a little"),
           (1.5, back, SLIDE_OPEN, "slide them back along its edge"),
           (0.6, back, FINGER_CLOSED, "pinch its edge again"),
           (1.5, back + (fwd - back) / 2, FINGER_CLOSED, "push it further in"),
           (1.5, fwd, FINGER_CLOSED, "push it further in")]
    rw = [(d, _Dyn(tcp, "far"), f, lab, o_y) for d, tcp, f, lab in one + one]
    P.window(t_r, t_r + sum(w[0] for w in rw), "far", _far_error, "new_filter", still=False)
    P.window(t_r, t_r + sum(w[0] for w in rw), "yaw", _yaw_error, "new_filter", still=False)
    P.phase(right=rw)
    # the right hand lets go of the edge and backs out; takes the tab where the camera sees it (the element may have turned
    #    a little in the edge pinch, which is a pivot), moves it onto the tray's line (the element turns about its far
    #    end, at the slot's mouth) and pushes it home by the tab. The left hand stays at its rest.
    t_h, o_hd = _carry(HAND_NEAR, BAY_Y, 0.0, 0.004)
    d_h = _dir(-1, -PICK_YAW)
    P.phase(right=[(0.6, _Dyn(t_h, "far"), PICK_OPEN, "let go of its edge", o_y)])
    P.phase(right=[(2.5, _Dyn(t_h, "far"), PICK_OPEN, "hold still, open, while the left arm moves out of the way", o_y)],
            left=[(2.5, tuple(LEFT_AWAY), FINGER_OPEN, "out to the left, clear of the right arm's turn", o_h)])
    P.phase(right=[(1.5, _Dyn(t_h - 0.150 * d_h, "far"), PICK_OPEN, "back out along its line", o_y),
                   (1.0, tuple(t_h - 0.150 * d_h + 0.040 * up), PICK_OPEN, "up", o_hd)])
    t_tab = P.t
    tab_w = [(2.0, stage(_tab(NEAR2)) + 0.030 * up, GRIP_OPEN, "hand turned to point into the slot, in line with the tab"),
             (1.0, stage(_tab(NEAR2)), GRIP_OPEN, "down to the tab's height"),
             (1.5, _tab(NEAR2) - 0.030 * X, GRIP_OPEN, "straight in, in front of the tab"),
             (1.0, _tab(NEAR2), GRIP_OPEN, "pads above and below the tab")]
    P.window(t_tab, t_tab + sum(w[0] for w in tab_w), "tab", _tab_error, "new_filter")
    P.phase(right=[*[(d, _Dyn(tcp, "tab"), f, lab, o_h) for d, tcp, f, lab in tab_w],
                   (0.8, _Dyn(_tab(NEAR2), "tab"), FINGER_CLOSED, "pinch the new element's tab", o_h),
                   (1.5, tuple(_tab(NEAR2)), FINGER_CLOSED, "move the tab onto the tray's line", o_h),
                   *[(1.2, tuple(_tab(NEAR2 + (SEAT_NEAR + 0.003 - NEAR2) * k / 4)), FINGER_CLOSED,
                      "push it home by its tab", o_h) for k in (1, 2, 3, 4)],
                   (0.6, tuple(_tab(SEAT_NEAR + 0.003)), GRIP_OPEN, "let go of the tab", o_h),
                   (1.0, tuple(_tab(SEAT_NEAR + 0.003) - 0.040 * X), GRIP_OPEN, "back off", o_h),
                   (2.0, tuple(stage(_tab(SEAT_NEAR)) + 0.030 * up), GRIP_OPEN, "straight back out of the opening", o_h),
                   (2.0, tuple(RIGHT_WAIT), GRIP_OPEN, "to wait, up and clear", o_h)])
    # 4. both hands rest, clear of the box and the elements, until the recording ends
    P.phase(right=[(4.0, tuple(RIGHT_WAIT), GRIP_OPEN, "hold", o_h)],
            left=[(4.0, tuple(LEFT_AWAY), FINGER_OPEN, "hold", o_h)])
    return {"left": P.L, "right": P.R}


O0 = Orient(R0)


import os as _os  # noqa: E402

_DBG = {"n": 0}


@on_reset
def _dbg_reset(env):
    _DBG["n"] = 0


@on_step
def _dbg(env):
    """GLOVE_DEBUG=1: every 2 s, the box's opening, both filters' poses and the score's latches."""
    if not _os.environ.get("GLOVE_DEBUG"):
        return
    _DBG["n"] += 1
    if _os.environ.get("GLOVE_JOINTS"):  # both arms' actual joint angles every step, for an offline collision replay
        b = env.unwrapped
        if _DBG["n"] == 1:
            with open(_os.environ["GLOVE_JOINTS"], "a") as fh:
                fh.write("# " + " ".join(b.scene["left_robot"].joint_names[:8]) + chr(10))
        row = [round(_DBG["n"] / 20, 2)]
        for nm in ("left_robot", "right_robot"):
            row += [round(float(v), 5) for v in b.scene[nm].data.joint_pos.torch[0, :8].cpu().numpy()]
        with open(_os.environ["GLOVE_JOINTS"], "a") as fh:
            fh.write(" ".join(str(v) for v in row) + chr(10))
    if _DBG["n"] % (10 if 1480 <= _DBG["n"] <= 1900 else 40):
        return
    dash = labkit._part_state(env, DASH)
    out = [f"GLOVE t={_DBG['n'] / 20:.1f} box={math.degrees(_box_angle(dash)):.1f}"
           f" sq={tuple(round(v * 1000, 1) for v in _squeezed(dash))}"]
    for nm, spec in (("old", OLD), ("new", NEW)):
        f = labkit._part_state(env, spec)
        c = dash.to_local(f.pos)
        rel = dash.rotation().T @ f.rotation()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rel[2, 2])))))
        yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
        out.append(f"{nm}=({c[0]*1000:.0f},{c[1]*1000:.0f},{c[2]*1000:.0f})mm tilt={tilt:.1f} yaw={yaw:.1f}")
    out.append(f"bad='{_w['bad']}' passed={_w['passed']} far={_w['far']} still={_w['still']}")
    tps = _tips(env)
    out.append("tipsL=" + str(np.round(np.asarray(tps[4]) * 1000).astype(int).tolist())
               + " tipsR=" + str(np.round(np.asarray(tps[5]) * 1000).astype(int).tolist())
               + " jawR=" + str(round(float(np.linalg.norm(np.asarray(tps[2]) - np.asarray(tps[3]))) * 1000)))
    print(" ".join(out), flush=True)
