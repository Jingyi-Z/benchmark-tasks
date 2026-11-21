"""claude-b50: witnesses for drawer_plates_live.Latches on synthetic 20 Hz plate trajectories (task_v4 geometry)."""
import math
import numpy as np
import drawer_plates_live as L

FLOOR = 0.5095; FAR = [[0.32, 0.30, 1.0], [0.32, -0.30, 1.0]]; ZC = 0.750; BT = 0.5254; INTERIOR = (0.1065, 0.6145, -0.580, 0.130); COUNTER = (0.120, 0.440, 0.150, 0.400); WALL_TOP = 0.700
I = np.eye(3)
def Ry(d): a = math.radians(d); return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def run(traj, hands=None):
    """traj: list of steps, each [(pos, R), (pos, R)]; hands: hand-cover points per step (default: both hands far away)"""
    lt = L.Latches(ZC, COUNTER, BT, INTERIOR, WALL_TOP, FLOOR)
    for n, pl in enumerate(traj): lt.step(n, pl, FAR if hands is None else hands)
    return lt


def fall(z0, z1, xy, other):
    """free fall from z0 to z1 at 20 Hz, then rest"""
    out = []; z = z0; v = 0.0
    while z > z1:
        out.append([((xy[0], xy[1], z), I), other]); v += 9.81 * 0.05; z = max(z1, z - v * 0.05)
    out += [[((xy[0], xy[1], z1), I), other]] * 10
    return out


C = (0.265, -0.200); rest_other = ((0.265, 0.320, ZC), I)
cases = []
def case(name, traj, exp_drop, exp_hard, hands=None):
    lt = run(traj, hands); ok = (lt.dropped, lt.hard_hit) == (exp_drop, exp_hard); cases.append(ok)
    print('%-40s %s dropped=%s hard=%s %s' % (name, 'OK ' if ok else 'BAD', lt.dropped, lt.hard_hit, lt.events[:2]))


case('rest on counter', [[((0.265, 0.32, ZC), I), ((0.265, 0.32, ZC + 0.0125), I)]] * 30, False, False)
case('gentle place, 2 mm release', fall(BT + 0.002, BT, C, rest_other), False, False)
case('3 mm release', fall(BT + 0.003, BT, C, rest_other), False, False)
case('15 mm drop onto the board', fall(BT + 0.015, BT, C, rest_other), False, True)
case('40 mm drop onto the board', fall(BT + 0.040, BT, C, rest_other), False, True)
# fast carry in free air (0.5 m/s, abrupt start/stop far from any support): no latch
tr = [[((0.265, 0.32 - 0.025 * max(0, min(n - 5, 20)), 0.80), I), rest_other] for n in range(40)]
case('abrupt 0.5 m/s carry in free air', tr, False, False)
# lateral slam into the -y wall at 0.4 m/s inside the drawer
y0 = INTERIOR[2] + L.R_RIM + 0.04
tr = [[((0.30, y0 - 0.02 * min(n, 2), BT), I), rest_other] for n in range(12)]
case('0.4 m/s slam into the -y wall', tr, False, True)
# slow 0.1 m/s push against the wall: under the threshold
tr = [[((0.30, y0 - 0.005 * min(n, 8), BT), I), rest_other] for n in range(20)]
case('0.1 m/s nudge to the wall', tr, False, False)
# plate falls off the +y counter edge to the floor bay
tr = [[((0.265, 0.32, ZC), I), rest_other]] * 3 + [[((0.265, 0.45, ZC - 0.01 * n), I), rest_other] for n in range(1, 6)]
case('falls off the counter', tr, True, False)
case('held tilt 50 deg in free air (controlled)', [[((0.265, 0.10, 0.85), Ry(50)), rest_other]] * 10, False, False,
     hands=[[0.265 - 0.13, 0.10, 0.85 + 0.11]])
# codex 1146: held 50 deg tilt beside the +x wall, 26.6 mm over the board, no descent, a hand on the rim -> no DROP
R50 = Ry(50); W50 = (R50 @ L.PTS.T).T + np.array([0.506410, -0.200, 0.650]); hpt = W50[np.argmax(W50[:, 2])] + np.array([0, 0, 0.016])
case('codex held 50 deg tilt by the +x wall', [[((0.506410, -0.200, 0.650), R50), rest_other]] * 10, False, False, hands=[hpt.tolist()])
# codex 1146: flat plate at board height x 0.260 -> 0.210 in 50 ms, outer edge crosses the -x wall face (x 0.1065)
tr = [[((0.260, -0.200, BT), I), rest_other]] * 3 + [[((0.210, -0.200, BT), I), rest_other]] * 5
case('codex wall crossing x 0.260 to 0.210', tr, False, True)
# vertical pass above the -x wall: plate edge over the wall slab, descending 3 mm/step from 60 to 30 mm over the wall top
tr = [[((0.1065 + 0.120, -0.200, WALL_TOP + 0.060 - 0.003 * min(n, 10)), I), rest_other] for n in range(14)]
case('vertical pass above the -x wall', tr, False, False)
# low-speed slide along the -y wall (1 mm from it, 0.1 m/s along x) on the board
yb = INTERIOR[2] + L.R_RIM + 0.001
tr = [[((0.250 + 0.005 * n, yb, BT), I), rest_other] for n in range(16)]
case('slow slide along the -y wall', tr, False, False)
# tipped and resting on the board: lowest posed point on the board top
def low_of(R, z): return min(((R @ L.PTS.T).T[:, 2])) + z
zt = BT - low_of(Ry(50), 0.0)
case('tipped 50 deg, resting, released', [[((0.265, -0.2, zt), Ry(50)), rest_other]] * 5, True, False)
W50b = (Ry(50) @ L.PTS.T).T + np.array([0.265, -0.2, zt]); hpb = W50b[np.argmax(W50b[:, 2])] + np.array([0, 0, 0.016])
case('tipped 50 deg on the board, held (controlled)', [[((0.265, -0.2, zt), Ry(50)), rest_other]] * 5, False, False, hands=[hpb.tolist()])
# codex 1136 replay: flat plate x 0.265 -> 0.281, z 0.5304 -> 0.5283 in one step over the board (lateral, 0.042 m/s normal)
tr = [[((0.265, -0.2, 0.5304), I), rest_other]] * 3 + [[((0.281, -0.2, 0.5283), I), rest_other]] * 5
case('codex lateral descent replay', tr, False, False)
# benign lateral descent: 0.3 m/s sideways while settling 1 mm per step onto the board
tr = [[((0.20 + 0.015 * n, -0.2, BT + max(0.0, 0.006 - 0.001 * n)), I), rest_other] for n in range(10)]
case('benign lateral descent 0.3 m/s', tr, False, False)
# edge-led impact: tilted 20 deg, falls 20 mm (lowest edge) onto the board
R20 = Ry(20); zb = BT - low_of(R20, 0.0); out = []; z = zb + 0.020; v = 0.0
while z > zb:
    out.append([((C[0], C[1], z), R20), rest_other]); v += 9.81 * 0.05; z = max(zb, z - v * 0.05)
out += [[((C[0], C[1], zb), R20), rest_other]] * 5
case('edge-led 20 deg drop of 20 mm', out, False, True)
# gentle edge-led settle: tilted 5 deg, edge lowered 1 mm per step onto the board
R5 = Ry(5); zb = BT - low_of(R5, 0.0)
case('edge-led gentle settle 1 mm/step', [[((C[0], C[1], zb + 0.006 - 0.001 * min(n, 6)), R5), rest_other] for n in range(10)], False, False)
# upper plate dropped 20 mm onto the lower plate in the drawer
low = ((C[0], C[1], BT), I)
out = []; z = BT + 0.0125 + 0.020; v = 0.0
while z > BT + 0.0125:
    out.append([low, ((C[0], C[1], z), I)]); v += 9.81 * 0.05; z = max(BT + 0.0125, z - v * 0.05)
out += [[low, ((C[0], C[1], BT + 0.0125), I)]] * 10
case('upper plate dropped 20 mm onto the lower', out, False, True)
case('nan pose', [[((0.265, 0.32, ZC), I), rest_other]] * 3 + [[((float('nan'), 0.32, ZC), I), rest_other]], True, False)
print('witnesses', sum(cases), '/', len(cases))
