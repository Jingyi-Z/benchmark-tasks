"""Prototype (not a submission): can a gripper-held screwdriver turn a slotted screw head in this sim?

A screwdriver stands blade down in a holder. A screw head (a hinge about z on a fixed block) has a slot between two
ridges. The right hand takes the screwdriver by its handle, sets the blade in the slot and turns it two half turns,
lifting and turning back between them. Scored by how far the screw turned.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
STEEL = (0.72, 0.73, 0.75)
DARK = (0.20, 0.21, 0.23)
RED = (0.75, 0.15, 0.12)
ALU = (0.74, 0.75, 0.77)

# the screw: head 7 mm across, 2.5 mm tall, two ridges 1.5 mm tall either side of a 1.2 mm slot (along y at joint 0)
S_XY, S_TOP = (0.22, -0.10), 0.100
HEAD_R, HEAD_H, SLOT, RIDGE = 0.0035, 0.0025, 0.0012, (0.0012, 0.007, 0.0015)
SCREW = part(
    "screw_block", pos=(*S_XY, TOP), fixed=True, friction=0.6,
    boxes=[((0.0, 0.0, S_TOP / 2), (0.040, 0.040, S_TOP), ALU)],
    joints=[Joint("screw", kind="hinge", axis="Z", anchor=(0.0, 0.0, S_TOP), lower=-60.0, upper=60.0,
                  friction=0.004, damping=0.0005, mass=0.003,
                  cylinders=[((0.0, 0.0, S_TOP + HEAD_H / 2), HEAD_R, HEAD_H, STEEL)],
                  boxes=[((e * (SLOT / 2 + RIDGE[0] / 2), 0.0, S_TOP + HEAD_H + RIDGE[2] / 2), RIDGE, STEEL)
                         for e in (-1, 1)])],
)

# the screwdriver (loose): a 14 mm handle 50 mm long over a 5 mm shaft 60 mm long, with a 0.8 x 5 mm blade at its tip;
# frame: origin at the blade's tip, +z up the shaft. It stands in a holder: a collar round its shaft under the handle.
H_R, H_L, SH_R, SH_L = 0.007, 0.050, 0.0025, 0.060
BLADE = (0.0008, 0.005, 0.004)  # thickness along x, width along y, height
SD_XY = (0.15, -0.20)
COLLAR_Z = 0.045  # the collar's top above the bench: the handle's bottom rests on it
SD_START = np.array([*SD_XY, TOP + COLLAR_Z - SH_L])
SCREWDRIVER = part(
    "screwdriver", pos=tuple(SD_START), mass=0.04, friction=1.0,
    cylinders=[((0.0, 0.0, SH_L + H_L / 2), H_R, H_L, RED), ((0.0, 0.0, BLADE[2] + (SH_L - BLADE[2]) / 2), SH_R,
                                                             SH_L - BLADE[2], STEEL)],
    boxes=[((0.0, 0.0, BLADE[2] / 2), BLADE, STEEL)],
)
_cn = 10
HOLDER = part(
    "sd_holder", pos=(*SD_XY, TOP), fixed=True, friction=0.6,
    boxes=[((0.0, 0.0, 0.003), (0.040, 0.040, 0.006), DARK),
           ((0.016, 0.0, COLLAR_Z / 2), (0.006, 0.020, COLLAR_Z), DARK),
           *[((0.0045 * math.cos(a), 0.0045 * math.sin(a), COLLAR_Z - 0.004), (0.003, 0.004, 0.008), DARK, a)
             for a in (2 * math.pi * k / _cn for k in range(_cn))],
           *[((0.010 * math.cos(a), 0.010 * math.sin(a), COLLAR_Z - 0.004), (0.010, 0.004, 0.008), DARK, a)
             for a in (2 * math.pi * k / _cn for k in range(_cn))]],
)
OBJECTS = [SCREW, SCREWDRIVER, HOLDER]

_state = {}


@on_reset
def _r(env):
    _state.update(angle=0.0)


@on_step
def _w(env):
    s = labkit._part_state(env, labkit._SPECS["screw_block"])
    _state["angle"] = float(s.joints.get("screw", 0.0))


def score(objects):
    """1.0 if the screw turned at least 300 degrees (either way), 0.5 past 120 degrees, else 0."""
    a = abs(math.degrees(_state.get("angle", 0.0)))
    return 1.0 if a >= 300 else 0.5 if a >= 120 else 0.0


SUCCESS_HOLD_S = 1.0


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    return 0.0


# -- demo -------------------------------------------------------------------------------------------------------------
TIP_ABOVE_TCP = 0.013
OPEN_W, HOLD = -0.018, -0.0055  # round the 14 mm handle
GRIP_UP = SH_L + 0.030  # the fingertips this far above the blade's tip (on the handle's lower part)


def _tcp(tip_xy, tip_z):
    """TCP for a straight-down hand holding the screwdriver upright with its blade tip at (xy, z)."""
    return (tip_xy[0], tip_xy[1], tip_z + GRIP_UP + TIP_ABOVE_TCP)


def demo():
    r = []
    z0 = SD_START[2]
    hz = S_TOP + HEAD_H + 0.0002  # the blade's tip just above the head's top, between the ridges
    a0 = math.radians(-45.0)  # the hand's yaw at pick-up; the blade then lies along y, the slot's direction at joint 0
    r += [
        Waypoint(2.5, _tcp(SD_XY, z0 + 0.04), OPEN_W, "above the screwdriver's handle", a0),
        Waypoint(1.2, _tcp(SD_XY, z0), OPEN_W, "fingers round the handle", a0),
        Waypoint(0.8, _tcp(SD_XY, z0), HOLD, "hold", a0),
        Waypoint(1.5, _tcp(SD_XY, z0 + 0.08), HOLD, "lift it out of its holder", a0),
        Waypoint(2.5, _tcp(S_XY, TOP + hz + 0.02), HOLD, "over the screw, blade along the slot", a0),
        Waypoint(1.2, _tcp(S_XY, TOP + hz), HOLD, "blade into the slot", a0),
    ]
    for stroke in range(2):
        for k in range(1, 7):
            r.append(Waypoint(0.6, _tcp(S_XY, TOP + hz - 0.0004), HOLD, "turn", a0 + math.pi * k / 6))
        r.append(Waypoint(0.6, _tcp(S_XY, TOP + hz + 0.006), HOLD, "lift the blade out", a0 + math.pi))
        for k in (4, 2, 0):
            r.append(Waypoint(0.8, _tcp(S_XY, TOP + hz + 0.006), HOLD, "turn back", a0 + math.pi * k / 6))
        r.append(Waypoint(0.8, _tcp(S_XY, TOP + hz), HOLD, "blade back in the slot", a0))
    r.append(Waypoint(1.5, _tcp(S_XY, TOP + hz + 0.02), HOLD, "up", a0))
    r.append(Waypoint(2.0, _tcp(S_XY, TOP + hz + 0.02), HOLD, "rest", a0))
    left = [Waypoint(sum(w.duration for w in r), (0.15, 0.25, TOP + 0.16), FINGER_OPEN, "stay clear")]
    return {"left": left, "right": r}


# --- debug (not for submission) ---
_dbgr = {"n": 0}
@on_step
def _dbg_sd(env):
    _dbgr["n"] += 1
    if _dbgr["n"] % 5:
        return
    sd = labkit._part_state(env, labkit._SPECS["screwdriver"])
    R = sd.rotation()
    yaw = math.degrees(math.atan2(R[1, 1], R[0, 1]))
    open("/home/ubuntu/dbg_sd.txt", "a").write(
        f"{_dbgr['n']} t={getattr(_demo, '_plan_clock', 0.0):.1f} screw={math.degrees(_state.get('angle', 0.0)):.0f} "
        f"sd={np.round((np.asarray(sd.pos) - [0, 0, TOP]) * 1000, 1).tolist()} tilt={math.degrees(math.acos(max(-1, min(1, R[2, 2])))):.1f} yaw={yaw:.0f}\n")
