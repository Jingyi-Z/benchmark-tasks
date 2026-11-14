"""A cold oyster lies cup side down on a wooden shucking board, its hinge toward the right arm; an oyster knife lies on
the board beside it, and a bowl for the empty top shells stands on the bench.

Task: shuck the oyster. Work the knife's tip into the hinge and twist it to pop the hinge (prying at the lip does
nothing); slide the blade in under the top shell to cut the muscle that holds it; lift the top shell off into the shell
bowl; put the knife back down on the board. All the while the other hand holds the cup level, so the liquor stays in,
and keeps clear of the knife's tip.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --instruction "..." --scorer task_score ...

The two shells are rigid bodies held together by two breakable PhysX joints (built with bodykit.py, which this folder
reuses unchanged from the accepted ice-cube tray task, TK-YTH6M4): the hinge ligament, a hinge along the hinge end
that breaks past 6 N, which a blade twisted in the 4 mm hinge gap wedges against but a straight push doesn't; and the
muscle, a ball joint where a small muscle body hangs from the top shell, which breaks past 1 N, which the blade
reaches by sweeping through the muscle body. Neither is switched off by the task file.
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bodykit  # noqa: E402
from bodykit import (Body, BoxShape, CylShape, allow_matrix_grips, as_numpy, ball, hinge, on_reset,  # noqa: E402
                     on_step, plate, rotation)
from yam_arena.constants import FINGER_CLOSED, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- the shucking board: end-grain hardwood ------------------------------------------------------------------------
BOARD_C = (0.300, -0.060)
BOARD = (0.200, 0.300, 0.030)
BT = TOP + BOARD[2]  # the board's top

# -- the oyster: a Pacific oyster, about 90 mm long; hinge toward -y (the right arm), lip toward +y --------------------
OX, OY = 0.300, 0.000
CUP = (0.056, 0.084, 0.020)  # the cup (bottom shell) outside: x, y, height to its rim
WALL = 0.004
CUP_MASS = 0.070  # with the meat and liquor
LID = (0.064, 0.088, 0.005)  # the flat top shell
LID_MASS = 0.025
RIM = BT + CUP[2]  # the rim the top shell rests on
NOTCH_W, NOTCH_H = 0.024, 0.004  # the hinge gap: a 2 mm blade slides in flat; its 8 mm tip turned 30 deg wedges 1.7 mm
HINGE_Y = OY - CUP[1] / 2  # the hinge end
ADDUCTOR = np.array([OX + 0.010, OY + 0.002, RIM])  # where the muscle meets the top shell
MUSCLE_R, MUSCLE_H = 0.005, 0.012  # the muscle, hanging from the top shell into the cup: what the blade must cut
POP_N = 6.0  # N: the hinge ligament breaks (pops) under this force: a blade twisted in the gap wedges past it
POP_NM = 0.03  # N m: or under this twisting load about the hinge line (the wedge mostly turns the shell there)
CUT_N = 1.0  # N: the muscle lets go of the top shell under this force: the blade swept through it
POP_GAP = 0.0015  # the top shell's hinge end lifted this far off the cup: the hinge has popped (latched)
OFF = 0.025  # the top shell's centre this far above its seat, or twice this to the side: it is off the cup
SEATED = 0.003  # the top shell's centre within this of its seat: still seated (a pop lifts only the hinge end)
TORN = 0.005  # the top shell lifted this far before the pop and the cut: torn off, which no shucker can do (score 0)

# -- the oyster knife: stout 65 mm blade, 2 mm thick, dull edges; plastic handle --------------------------------------
HANDLE = (0.024, 0.100, 0.024)  # x, y (along the knife), z
BLADE = (0.010, 0.065, 0.002)
TIP_W, TIP_L = 0.008, 0.010  # the blade narrows to an 8 mm tip for its last 10 mm
KNIFE_MASS = 0.090
KX, KY = 0.232, -0.170  # the handle's centre, lying on the board along y, blade toward +y
KZ = BT + HANDLE[2] / 2
TIP_LOCAL = np.array([0.0, HANDLE[1] / 2 + BLADE[1], 0.0])  # the blade's tip, from the handle's centre

# -- the shell bowl: stainless, open top ------------------------------------------------------------------------------
BOWL = np.array([0.400, 0.215])  # on the left hand's side: that hand takes the top shell off
BOWL_IN = (0.110, 0.110, 0.040)
BOWL_W = 0.003

WOOD, SHELL, NACRE, MEAT, STEEL, HANDLE_C, BOWL_C = ((0.62, 0.45, 0.28), (0.52, 0.50, 0.46), (0.86, 0.85, 0.82),
                                                     (0.78, 0.74, 0.62), (0.75, 0.76, 0.78), (0.10, 0.25, 0.55),
                                                     (0.70, 0.71, 0.73))

# -- scoring ----------------------------------------------------------------------------------------------------------
SPILL_DEG = 20.0  # the cup tipped more than this at any time: the liquor spilled
LEVEL_DEG = 10.0
GUARD = 0.030  # the knife's tip this close to a fingertip of the holding hand: score 0
CLEAR = 0.08
STILL_MM, STILL_STEPS = 1.0, 20


def _board() -> Body:
    return Body.build("board", pos=(BOARD_C[0], BOARD_C[1], TOP), mass=1.5, kinematic=True, color=WOOD,
                      friction=(0.7, 0.6), shapes=[BoxShape((0.0, 0.0, BOARD[2] / 2), BOARD, WOOD)])


def _bowl() -> Body:
    lx, ly, h = BOWL_IN
    w = BOWL_W
    shapes = [BoxShape((0.0, 0.0, w / 2), (lx + 2 * w, ly + 2 * w, w), BOWL_C)]
    for s in (-1.0, 1.0):
        shapes.append(BoxShape((s * (lx / 2 + w / 2), 0.0, w + h / 2), (w, ly + 2 * w, h), BOWL_C))
        shapes.append(BoxShape((0.0, s * (ly / 2 + w / 2), w + h / 2), (lx, w, h), BOWL_C))
    return Body.build("shell_bowl", pos=(float(BOWL[0]), float(BOWL[1]), TOP), mass=0.3, kinematic=True,
                      color=BOWL_C, friction=(0.5, 0.4), shapes=shapes)


def _outline(a: float, b: float, n: int = 24) -> list[np.ndarray]:
    """An oyster's outline in plan, as points round it: an oval, narrowed toward its pointed hinge end (-y)."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n - math.pi / 2  # the first point at the hinge
        taper = 1.0 - 0.30 * max(0.0, -math.sin(t)) ** 2
        pts.append(np.array([a * math.cos(t) * taper, b * math.sin(t)]))
    return pts


def _cup() -> Body:
    """The bottom shell, origin at its base's centre: a floor, a wall round an oyster-shaped outline (lower for 18 mm
    at the hinge, the gap the knife goes into), and the meat (visual only)."""
    cx, cy, h = CUP
    pts = _outline(cx / 2 - WALL / 2, cy / 2 - WALL / 2)
    shapes = []
    for k in range(5):  # the floor, in strips across the outline
        y0, y1 = -cy / 2 + k * cy / 5, -cy / 2 + (k + 1) * cy / 5
        ym = (y0 + y1) / 2
        half = max(abs(float(p[0])) for p in pts if y0 - 0.004 <= float(p[1]) <= y1 + 0.004)
        shapes.append(BoxShape((0.0, ym, WALL / 2), (2 * half, y1 - y0, WALL), SHELL))
    for p, q in zip(pts, pts[1:] + pts[:1]):
        mid = (p + q) / 2
        out = np.array([mid[0], mid[1] * (cx / cy) ** 2, 0.0])  # roughly outward
        notch = float(mid[1]) < 0 and abs(float(mid[0])) < NOTCH_W / 2
        hh = h - NOTCH_H if notch else h
        shapes.append(plate((float(p[0]), float(p[1]), hh / 2), (float(q[0]), float(q[1]), hh / 2), hh, WALL,
                            out, SHELL))
    shapes.append(BoxShape((0.0, 0.006, WALL + 0.004), (cx - 2 * WALL - 0.012, cy - 2 * WALL - 0.024, 0.008), MEAT,
                           collide=False))
    # the cupped shell's frilled outer edge, wider than the top shell and showing round it (drawn only)
    for z, scale in ((0.004, 1.30), (0.010, 1.22), (0.016, 1.14)):
        shapes += _frill(cx / 2 * scale, cy / 2 * scale * 0.95, z, (0.009, 0.006, 0.006), (0.40, 0.38, 0.35))
    return Body.build("cup", pos=(OX, OY, BT), mass=CUP_MASS, shapes=shapes, color=SHELL, friction=(0.8, 0.7),
                      com=(0.0, 0.0, 0.006))


def _half_width(y: float, a: float, b: float) -> float:
    """The outline's half width (x) at ``y`` (the same oval, narrowed toward the hinge end, as _outline)."""
    s = max(-1.0, min(1.0, y / b))
    return a * math.sqrt(max(0.0, 1.0 - s * s)) * (1.0 - 0.30 * max(0.0, -s) ** 2)


def _bands(a: float, b: float, n: int, z: float, thick: float, color, collide: bool) -> list[BoxShape]:
    """The outline filled with ``n`` bands across y, each as wide as the outline at its middle."""
    out = []
    for k in range(n):
        y0, y1 = -b + 2 * b * k / n, -b + 2 * b * (k + 1) / n
        half = _half_width((y0 + y1) / 2, a, b)
        if half > 0.0005:
            out.append(BoxShape((0.0, (y0 + y1) / 2, z), (2 * half, y1 - y0, thick), color, collide=collide))
    return out


def _frill(a: float, b: float, z: float, size, color, n: int = 36) -> list[BoxShape]:
    """A rough, scalloped edge round the outline (drawn only): small plates along it, alternately in and out."""
    pts = _outline(a, b, n)
    out = []
    for i, (p, q) in enumerate(zip(pts, pts[1:] + pts[:1])):
        mid, d = (p + q) / 2, q - p
        r = mid / max(float(np.linalg.norm(mid)), 1e-6) * (0.0012 if i % 2 else -0.0008)
        out.append(BoxShape((float(mid[0] + r[0]), float(mid[1] + r[1]), z), size, color,
                            yaw=math.atan2(float(d[1]), float(d[0])), collide=False))
    return out


def _lid() -> Body:
    """The flat top shell: an oyster-shaped plate (oval, narrowing to the hinge), overhanging the cup's wall by about
    4 mm, with its growth rings and scalloped edge drawn on top (visual only; the collision shape is the plate)."""
    a, b = LID[0] / 2, LID[1] / 2
    shapes = _bands(a, b, 14, 0.0, LID[2], SHELL, True)
    for k, (scale, color) in enumerate(((0.82, (0.46, 0.44, 0.40)), (0.62, (0.58, 0.56, 0.51)),
                                        (0.40, (0.43, 0.41, 0.37)))):
        shapes += _bands(a * scale, b * scale, 10, LID[2] / 2 + 0.0003 * (k + 1), 0.0004, color, False)
    shapes += _frill(a, b, LID[2] / 2 - 0.0005, (0.007, 0.0025, 0.0015), (0.48, 0.46, 0.42))
    m = ADDUCTOR - np.array([OX, OY, RIM + LID[2] / 2])
    shapes.append(CylShape((float(m[0]), float(m[1]), -LID[2] / 2 - MUSCLE_H / 2), MUSCLE_R, MUSCLE_H, MEAT))
    return Body.build("lid", pos=(OX, OY, RIM + LID[2] / 2), mass=LID_MASS, shapes=shapes, color=SHELL,
                      friction=(0.8, 0.7), com=(0.0, 0.004, 0.0))


def _knife() -> Body:
    shapes = [BoxShape((0.0, 0.0, 0.0), HANDLE, HANDLE_C),
              BoxShape((0.0, HANDLE[1] / 2 + 0.004, 0.0), (0.032, 0.008, 0.030), HANDLE_C),  # the guard
              BoxShape((0.0, HANDLE[1] / 2 + (BLADE[1] - TIP_L) / 2, 0.0), (BLADE[0], BLADE[1] - TIP_L, BLADE[2]),
                       STEEL),
              BoxShape((0.0, HANDLE[1] / 2 + BLADE[1] - TIP_L / 2, 0.0), (TIP_W, TIP_L, BLADE[2]), STEEL)]  # tip
    return Body.build("knife", pos=(KX, KY, KZ), mass=KNIFE_MASS, shapes=shapes, color=HANDLE_C,
                      friction=(1.5, 1.4), com=(0.0, -0.010, 0.0))  # a ridged handle


OBJECTS = [_board(), _bowl(), _cup(), _lid(), _knife()]
# The hinge ligament: a hinge along the hinge end that PhysX breaks once its force passes POP_N; with the muscle it
# holds the top shell rigid. The muscle: a ball joint at its top that PhysX breaks past CUT_N. Both physical: the
# task file never switches them off.
hinge("ligament", "cup", "lid", anchor=(OX, HINGE_Y + 0.004, RIM), axis="X", limits_deg=(-0.5, 0.5),
      break_force=POP_N, break_torque=POP_NM)
ball("muscle", "cup", "lid", anchor=tuple(float(v) for v in ADDUCTOR))


def _author_hinges_colliding(stage, prim_path: str, spec, _author=bodykit._author_hinges) -> None:
    """bodykit's hinge, with the two shells still colliding (a USD joint turns contact between its bodies off by
    default, which would let the top shell sink into the cup once it can move)."""
    from pxr import UsdPhysics

    _author(stage, prim_path, spec)
    for path in bodykit._HINGE_PATHS.values():
        if path.startswith(prim_path + "/"):
            UsdPhysics.Joint(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(True)


def _author_balls_breakable(stage, prim_path: str, spec, _author=bodykit._author_balls) -> None:
    """bodykit's ball joint, breakable past CUT_N and re-armed at every reset like a breakable hinge, contact kept."""
    from pxr import UsdPhysics

    _author(stage, prim_path, spec)
    path = f"{prim_path}/muscle"
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        joint = UsdPhysics.Joint(prim)
        joint.CreateCollisionEnabledAttr(True)
        joint.CreateBreakForceAttr(float(CUT_N))
        if path not in bodykit._BREAKABLE_PATHS:
            bodykit._BREAKABLE_PATHS.append(path)


bodykit._author_hinges = _author_hinges_colliding
bodykit._author_balls = _author_balls_breakable
LID_HINGE_END = np.array([0.0, -LID[1] / 2 + 0.006, -LID[2] / 2])  # on the top shell's underside, at the hinge end
SEAT = RIM + LID[2] / 2  # the top shell's centre height when seated
START = {o.name: np.array(o.pos, float) for o in OBJECTS}


# -- followed after every control step --------------------------------------------------------------------------------

# The demo's one look: once the knife is picked up and held still, where it sits in the hand (from the knife's seen
# pose and the hand's own pose). ``t`` is set by demo(); the rest is cleared at every reset.
_CAL = {"t": None, "R": None, "p": None, "steps": 0, "looks": [], "done": 0, "trim": None}


def _fresh() -> dict:
    return {"popped": False, "through": False, "cut": False, "torn": False, "spilled": False, "guard": False,
            "dropped": False, "roll0": None,
            "s": {}, "hist": deque(maxlen=STILL_STEPS + 1)}


_EP = _fresh()
_DEBUG = {"on": bool(os.environ.get("OYSTER_DEBUG")), "steps": 0}


@on_reset
def _new_episode() -> None:
    _EP.clear()
    _EP.update(_fresh())
    _DEBUG["steps"] = 0
    _CAL.update({"R": None, "p": None, "steps": 0, "done": 0, "trim": None})


def _hands(base, origin) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each hand's TCP position and rotation (left, right)."""
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        i = next(j for j, n in enumerate(robot.body_names) if n == "gripper" or n.endswith("/gripper"))
        g = as_numpy(robot.data.body_link_pose_w)[0][i]
        r = rotation(g[3:7])
        out.append((g[:3] - origin + r @ np.array(TCP_OFFSET), r))
    return out


def _tilt(r) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def _roll(r_knife, r_cup) -> float:
    """The knife's turn about its own length (its y axis), against the cup: 0 with the blade flat."""
    rel = r_cup.T @ r_knife
    return math.degrees(math.atan2(float(rel[2, 0]), float(rel[2, 2])))


def _box_dist(p, c, r, half) -> float:
    q = r.T @ (np.asarray(p, float) - c)
    return float(np.linalg.norm(np.maximum(np.abs(q) - half, 0.0)))


def _in_bowl(p) -> bool:
    return (abs(float(p[0]) - BOWL[0]) <= BOWL_IN[0] / 2 and abs(float(p[1]) - BOWL[1]) <= BOWL_IN[1] / 2
            and float(p[2]) < TOP + BOWL_W + BOWL_IN[2])


def _on_board(p, z_max) -> bool:
    return (abs(float(p[0]) - BOARD_C[0]) <= BOARD[0] / 2 and abs(float(p[1]) - BOARD_C[1]) <= BOARD[1] / 2
            and float(p[2]) < z_max)


@on_step
def _follow(env) -> None:
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    st = {}
    for n in ("cup", "lid", "knife"):
        a = as_numpy(base.scene[n].data.root_link_pose_w)[0]
        st[n] = (a[:3] - origin, rotation(a[3:7]))
    hands = _hands(base, origin)
    tcps = [h[0] for h in hands]
    (cp, cr), (lp, lr), (kp, kr) = st["cup"], st["lid"], st["knife"]
    tip = kp + kr @ TIP_LOCAL
    _CAL["steps"] += 1
    if _CAL["t"] is not None and _CAL["R"] is None and _CAL["steps"] / 20.0 >= _CAL["t"]:
        pg, rg = hands[1]  # the knife in the right hand, as seen: kept for the rest of the demo
        _CAL["R"], _CAL["p"] = rg.T @ kr, rg.T @ (kp - pg)
    looks = _CAL["looks"]
    if _CAL["done"] < len(looks) and _CAL["steps"] / 20.0 >= looks[_CAL["done"]][0]:
        # a look at the tip, held still: the hand doesn't quite reach the turn it's told (a few degrees, varying with
        # the pose), which moves the tip 115 mm away; shift the hand by the tip's seen miss, from here on
        trim = _CAL["trim"] if _CAL["trim"] is not None else np.zeros(3)
        _CAL["trim"] = np.clip(trim + np.asarray(looks[_CAL["done"]][1], float) - tip, -0.025, 0.025)
        _CAL["done"] += 1
        if _DEBUG["on"]:
            print(f"[oyster] AIM {_CAL['done']} at step {_CAL['steps']}: tip={np.round(tip * 1000, 1).tolist()} "
                  f"trim={np.round(_CAL['trim'] * 1000, 1).tolist()}", file=sys.stderr, flush=True)
    tip_c = cr.T @ (tip - cp)  # the tip in the cup's frame (origin at the base's centre)
    roll = _roll(kr, cr)
    # popped: the top shell's hinge end lifted off the cup's rim while the shell's centre stays seated (levered open
    # at the hinge, which needs the ligament broken), not the whole shell lifted
    gap = float((cr.T @ (lp + lr @ LID_HINGE_END - cp))[2]) - CUP[2]
    lid_c = cr.T @ (lp - cp)
    rise = float(lid_c[2]) - (CUP[2] + LID[2] / 2)  # the top shell's centre above its seat
    if gap >= POP_GAP and rise <= SEATED:
        _EP["popped"] = True
    # through: after the pop, the knife's tip has been in the muscle under the top shell (the cut)
    m_c = ADDUCTOR - np.array([OX, OY, BT])
    if _EP["popped"] and float(np.linalg.norm((tip_c - m_c)[:2])) <= MUSCLE_R + 0.002 \
            and CUP[2] - MUSCLE_H <= float(tip_c[2]) <= CUP[2]:
        _EP["through"] = True
    # torn: the top shell lifted before both, which only force (no shucker) does
    if rise > TORN and not (_EP["popped"] and _EP["through"]):
        _EP["torn"] = True
    # off: the top shell is clear of the cup (only possible once the muscle has let go too)
    off = rise >= OFF or float(np.linalg.norm(lid_c[:2])) >= 2 * OFF
    if off and not _EP["torn"]:
        _EP["cut"] = True
    if _tilt(cr) > SPILL_DEG:
        _EP["spilled"] = True
    if float(np.linalg.norm(tip - tcps[0])) < GUARD:
        _EP["guard"] = True
    if float(cp[2]) < BT - 0.010 or (float(lp[2]) < TOP + 0.020 and not _in_bowl(lp)) or float(kp[2]) < BT - 0.010:
        _EP["dropped"] = True
    _EP["hist"].append(np.concatenate([cp, lp, kp]))
    hist = np.array(_EP["hist"])
    still = len(hist) > STILL_STEPS and float(np.abs(hist - hist[-1]).max()) * 1000 < STILL_MM
    cup_back = (float(np.linalg.norm(cp[:2] - START["cup"][:2])) <= 0.030 and _tilt(cr) <= LEVEL_DEG
                and abs(float(cp[2]) - BT) <= 0.005)
    knife_down = _on_board(kp, BT + HANDLE[2])
    shapes = [(cp, cr, np.array(CUP) / 2 + [0, 0, 0]), (lp, lr, np.array(LID) / 2), (kp, kr, np.array(HANDLE) / 2)]
    clear = all(_box_dist(t, c, r, h) >= CLEAR for t in tcps for c, r, h in shapes)
    _EP["s"] = {"lid_in_bowl": _in_bowl(lp), "cup_back": cup_back, "knife_down": knife_down, "clear": clear,
                "still": still}
    if _DEBUG["on"]:
        _DEBUG["steps"] += 1
        if _DEBUG["steps"] % 10 == 0:
            print(f"[oyster] t={_DEBUG['steps'] / 20:5.1f}s tip_c={np.round(tip_c * 1000).astype(int).tolist()} "
                  f"gap={gap * 1000:5.1f} "
                  f"roll={roll:6.1f} rise={rise * 1000:5.1f} popped={_EP['popped']} through={_EP['through']} "
                  f"off={_EP['cut']} torn={_EP['torn']} "
                  f"cup_tilt={_tilt(cr):4.1f} spilled={_EP['spilled']} guard={_EP['guard']} dropped={_EP['dropped']} "
                  f"lid={np.round(lp * 1000).astype(int).tolist()} {_EP['s']} score={_points()}",
                  file=sys.stderr, flush=True)


# -- scoring --------------------------------------------------------------------------------------------------------

def _points() -> float:
    e, s = _EP, _EP["s"]
    if e["spilled"] or e["guard"] or e["dropped"] or e["torn"] or not s.get("still"):
        return 0.0
    if not e["popped"]:
        return 0.0
    if not e["cut"]:
        return 0.25
    if not s["lid_in_bowl"]:
        return 0.5
    if not (s["cup_back"] and s["knife_down"]):
        return 0.75
    return 1.0 if s["clear"] else 0.75


def score(objects) -> float:
    """0 if, at any time, the cup tipped more than 20 degrees (the liquor spilled), the knife's tip came within 3 cm of
    a fingertip of the hand holding the cup, the cup, the top shell or the knife fell off the board (the top shell may
    go into the shell bowl), or the top shell was lifted 5 mm before the hinge had popped and the knife's tip had been
    in the muscle (torn off); 0 unless the cup, the top shell and the knife moved under 1 mm in the last second.
    Otherwise 0.25 once the hinge has popped (the top shell's hinge end lifted 1.5 mm or more off the cup's rim while
    its centre stayed seated); 0.5 once the top shell is off the cup (its centre 25 mm or more above its seat, or 50 mm
    or more to the side); 0.75 with the top shell also in the shell bowl; 1.0 with the cup also level on the board
    (within 10 degrees, within 3 cm of where it started), the knife lying on the board, and every fingertip 8 cm or
    more from the cup, the top shell and the knife."""
    return _points()


def success(objects) -> bool:
    return score(objects) >= 1.0


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left hand comes down on the top shell's lip end, rolled outward, fingers closed, and presses the oyster on the
# board so it stays level and still while the knife works. The right hand picks the knife up by its handle (from
# above, blade pointing at the hinge), tips it 15 degrees tip down, works the tip into the hinge gap, twists it 30
# degrees to pop the hinge, levels it, slides the tip in under the top shell to the muscle and sweeps across it, draws
# the knife out and lays it back where it was. Then the left hand lifts off and takes the top shell, by its sides, to
# the shell bowl. The demo looks twice, at what a camera could see: where the knife sits in the hand once picked up,
# and where its tip is when held still 3 cm short of the hinge (the hand's small turn error moves the tip ~10 mm).


def _rot(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


DOWN_R = grip_rotation(math.pi / 2)  # right hand: straight down, fingers closing across x
DOWN_L = grip_rotation(-math.pi / 2)  # left hand: straight down, fingers closing across x
HOLD_ROLL = -35.0  # the holding hand rolled outward (wrist toward +y): two straight-down hands collide within ~210 mm
FINGERTIP = 0.008  # the closed fingertips this far past the TCP along the fingers
PRESS = 0.002  # pressed this far into the top shell
OPEN_CUP = -0.040
OPEN_KNIFE = -0.022
OPEN_LID = -0.040
TILT = 15.0
TWIST = -30.0  # the right wrist reaches this way, not the other
LEAD_S, LEAD_STEPS = 3.0, 15


def _good_ik(arm: str, pos, rot) -> np.ndarray:
    from yam_arena import ik, station

    best = None
    for seed in itertools.product((-0.5, 0.0, 0.5), (0.8, 1.3), (0.4, 1.0), (-0.8, 0.0, 0.8), (-1.4, 0.0, 1.4),
                                  (-1.5, 0.0, 1.5)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.array(seed),
                             base_pos=station.BASES[arm], iters=200)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    if best is None:
        raise ValueError(f"no IK solution for the {arm} grip at {pos}")
    return best[0]


def _joint_path(arm: str, q0, q1, label: str, finger) -> list[Waypoint]:
    from yam_arena import ik, station

    out = []
    for k in range(1, LEAD_STEPS + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / LEAD_STEPS, station.BASES[arm])
        out.append(Waypoint(LEAD_S / LEAD_STEPS, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, pos, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, rot)


def _knife_pose(tip, tilt: float, twist: float):
    """The TCP pose that puts the knife's tip at ``tip``, the knife tipped ``tilt`` degrees tip down and turned
    ``twist`` degrees about its length, given how the knife sits in the hand: as measured once it was picked up
    (``_CAL``), or, before that, as it was meant to be picked up (handle's centre at the TCP, along the fingers' y)."""
    p = _rot([1.0, 0.0, 0.0], -tilt)
    along = p @ np.array([0.0, 1.0, 0.0])
    r_k = _rot(along, twist) @ p  # the knife's rotation wanted
    r_hk = _CAL["R"] if _CAL["R"] is not None else DOWN_R.T  # the knife in the hand
    p_hk = _CAL["p"] if _CAL["p"] is not None else np.zeros(3)
    r_g = r_k @ r_hk.T
    trim = _CAL["trim"] if _CAL["trim"] is not None else np.zeros(3)
    return np.asarray(tip, float) - r_k @ TIP_LOCAL - r_g @ p_hk + trim, r_g


class _KnifePos:
    """A waypoint TCP position that puts the knife's tip at ``tip`` (re-evaluated every step, so it follows the knife's
    seen place in the hand once that is measured)."""

    def __init__(self, tip, tilt: float, twist: float):
        self.args = (np.asarray(tip, float), tilt, twist)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(_knife_pose(*self.args)[0], dtype=dtype or float)


class _KnifeRot(_KnifePos):
    """The matching TCP rotation (a 3x3, as bodykit's matrix grips allow)."""

    def __array__(self, dtype=None, copy=None):
        return np.asarray(_knife_pose(*self.args)[1], dtype=dtype or float)


def _wk(t, tip, tilt, twist, label) -> Waypoint:
    return Waypoint(t, _KnifePos(tip, tilt, twist), FINGER_CLOSED, label, _KnifeRot(tip, tilt, twist))


def demo():
    """The scripted demo. OYSTER_SHORTCUT=push runs it with no twist (the knife pushed straight in, the rest the same);
    OYSTER_SHORTCUT=pull leaves the knife alone and has the left hand pull the top shell off: both must score 0."""
    from yam_arena import ik, station
    from yam_arena.constants import READY

    shortcut = os.environ.get("OYSTER_SHORTCUT", "")
    twist = 0.0 if shortcut == "push" else TWIST
    # -- left: press the oyster down on the board by its lip end, fingers closed (as a towelled hand does) -------------
    hold_r = _rot([1.0, 0.0, 0.0], HOLD_ROLL) @ DOWN_L
    approach = hold_r @ np.array([0.0, 0.0, 1.0])  # the fingers' pointing line, down and in
    press_at = np.array([OX, OY + 0.032, RIM + LID[2]])  # on the top shell, near the lip: 34 mm or more from the tip
    hold = press_at - approach * (FINGERTIP - PRESS)
    above_hold = hold - 0.080 * approach
    left = _joint_path("left", np.array(READY, float), _good_ik("left", above_hold, hold_r), "over the lip end",
                       FINGER_CLOSED)
    left += [_w(1.5, hold, FINGER_CLOSED, "press the oyster down on the board: it stays level and still", hold_r)]

    # -- right: pick the knife up ----------------------------------------------------------------------------------
    handle = np.array([KX, KY, KZ])
    over_knife = handle + [0.0, 0.0, 0.080]
    right = _joint_path("right", np.array(READY, float), _good_ik("right", over_knife, DOWN_R), "over the knife",
                        OPEN_KNIFE)
    right += [_w(1.2, handle, OPEN_KNIFE, "down round the handle", DOWN_R),
              _w(0.6, handle, FINGER_CLOSED, "take hold of the handle", DOWN_R),
              _w(1.0, over_knife, FINGER_CLOSED, "lift the knife", DOWN_R),
              _w(0.8, over_knife, FINGER_CLOSED, "hold still: see where the knife sits in the hand", DOWN_R)]
    _CAL["t"] = sum(w.duration for w in right) - 0.2  # the one look, at the end of that hold
    # -- into the hinge (from here every pose puts the knife's tip where it's wanted, as the knife sits in the hand) ---
    notch = np.array([OX, HINGE_Y, RIM - NOTCH_H / 2 + 0.0005])  # the gap's middle, 0.5 mm high: clear of its floor
    out_tip = notch + [0.0, -0.030, 0.012]
    in_tip = notch + [0.0, WALL + 0.010, 0.0]  # aimed 4 mm deeper than wanted: the arm stops ~4 mm short along y
    near_tip = notch + [0.0, -0.008, 0.0]  # 8 mm short of the hinge end, at the gap's height
    looks = []
    right += [_wk(2.0, out_tip, TILT, 0.0, "tip lined up with the hinge, 3 cm back, tipped down"),
              _wk(0.8, out_tip, TILT, 0.0, "hold still: see where the tip is")]
    looks.append((sum(w.duration for w in right) - 0.2, out_tip))
    right += [_wk(0.6, out_tip, TILT, 0.0, "tip on the line"),
              _wk(1.2, near_tip, TILT, 0.0, "tip 8 mm short of the hinge"),
              _wk(0.8, near_tip, TILT, 0.0, "hold still: see where the tip is, close up")]
    looks.append((sum(w.duration for w in right) - 0.2, near_tip))
    _CAL["looks"] = looks
    right += [_wk(0.6, near_tip, TILT, 0.0, "tip on the line, close up"),
              _wk(1.5, in_tip, TILT, 0.0, "work the tip into the hinge")]
    right += [_wk(0.3, in_tip, TILT, twist * k / 5, "twist the knife: the hinge pops") for k in range(1, 6)]
    right += [_wk(0.6, in_tip, TILT, twist, "hold the twist"),
              _wk(1.0, in_tip, TILT, 0.0, "untwist")]
    # -- cut the top muscle -----------------------------------------------------------------------------------------
    under = in_tip + [0.0, 0.0, 0.001]
    at_muscle = np.array([ADDUCTOR[0] - 0.008, ADDUCTOR[1], RIM - NOTCH_H / 2])
    right += [_wk(1.0, under, 0.0, 0.0, "level the blade under the top shell"),
              _wk(2.0, at_muscle, 0.0, 0.0, "slide in flat under the top shell to the muscle"),
              _wk(1.5, at_muscle + [0.020, 0.0, 0.0], 0.0, 0.0, "sweep across the muscle"),
              _wk(1.2, at_muscle, 0.0, 0.0, "sweep back across it"),
              _wk(1.2, at_muscle + [0.020, 0.0, 0.0], 0.0, 0.0, "and across again: it's cut"),
              _wk(2.0, under, 0.0, 0.0, "draw the blade back to the hinge"),
              _wk(1.5, notch + [0.0, -0.030, 0.0], 0.0, 0.0, "straight back out of the hinge, level"),
              _wk(0.8, out_tip, 0.0, 0.0, "up, clear of the oyster")]
    # -- knife down -------------------------------------------------------------------------------------------------
    right += [_w(2.0, over_knife, FINGER_CLOSED, "over its place on the board", DOWN_R),
              _w(1.2, handle + [0.0, 0.0, 0.002], FINGER_CLOSED, "lay it down", DOWN_R),
              _w(0.6, handle + [0.0, 0.0, 0.002], OPEN_KNIFE, "let go", DOWN_R),
              _w(1.0, over_knife, OPEN_KNIFE, "up", DOWN_R)]
    right += _joint_path("right", _good_ik("right", over_knife, DOWN_R), np.array(READY, float),
                         "back to where it started", OPEN_KNIFE)
    if shortcut == "pull":  # no knife work: the right hand stays where it started
        p0, r0 = ik.tcp_pose(np.array(READY, float), station.BASES["right"])
        right = [Waypoint(2.0, tuple(float(v) for v in p0), OPEN_KNIFE, "stay put (pull-off check)", r0)]
    # -- left: once the knife is down and the right hand is away, let go of the cup and take the top shell off ---------
    t_left = sum(w.duration for w in left)
    left.append(Waypoint(max(sum(w.duration for w in right) - t_left, 0.5), tuple(float(v) for v in hold),
                         FINGER_CLOSED, "keep holding the cup", hold_r))
    left += [_w(1.0, above_hold, FINGER_CLOSED, "lift the hand off the oyster", hold_r)]
    lid = np.array([OX, OY, RIM + LID[2] / 2])
    over_lid = lid + [0.0, 0.0, 0.060]
    over_bowl = np.array([BOWL[0], BOWL[1], TOP + 0.120])
    left += _joint_path("left", _good_ik("left", above_hold, hold_r), _good_ik("left", over_lid, DOWN_L),
                        "over the top shell, pointing down", OPEN_LID)
    left += [_w(1.2, lid, OPEN_LID, "down round its sides", DOWN_L),
             _w(0.6, lid, FINGER_CLOSED, "take hold of it", DOWN_L),
             _w(2.0, over_lid, FINGER_CLOSED, "lift it off, slowly", DOWN_L),
             _w(2.0, over_bowl, FINGER_CLOSED, "over the shell bowl", DOWN_L),
             _w(0.6, over_bowl, OPEN_LID, "drop it in", DOWN_L),
             _w(1.0, over_bowl + [-0.050, 0.0, 0.0], OPEN_LID, "back", DOWN_L)]
    left += _joint_path("left", _good_ik("left", over_bowl + [-0.050, 0.0, 0.0], DOWN_L), np.array(READY, float),
                        "back to where it started", OPEN_LID)
    total = max(sum(w.duration for w in left), sum(w.duration for w in right)) + 3.0
    for side in (left, right):
        t = sum(w.duration for w in side)
        side.append(Waypoint(max(total - t, 0.5), side[-1].pos, side[-1].finger,
                             "hold still: the trial ends once the full score has held 1 s", side[-1].yaw))
    return {"left": left, "right": right}





