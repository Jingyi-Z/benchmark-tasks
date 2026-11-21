"""ONE-PLATE (option B) witnesses for drawer_plates_score_b.score_window (n_plates=1). claude-b50: witnesses for drawer_plates_score.score_window on synthetic frames in the task_v4 geometry
(0, partial and 1.0 end states, plus fail-closed cases)."""
import math
import numpy as np
import drawer_plates_score_b as S

P = 0.0635
HX = [0.1698 + P * i for i in range(-1, 6)]; HY = [-0.3270 + P * j for j in range(-3, 6)]
HOLES = [(x, y) for x in HX for y in HY]
def hidx(x, y): return min(range(len(HOLES)), key=lambda k: math.hypot(HOLES[k][0] - x, HOLES[k][1] - y))
START = [(0.1698, -0.3270), (0.3603, -0.3270), (0.1698, -0.1365), (0.3603, -0.1365)] + [(0.4873, -0.3270 + P * j) for j in range(-1, 4)]
TGT = {2: (0.1698, -0.0730), 3: (0.3603, -0.0730)}
start_hole = [hidx(*xy) for xy in START]; target_hole = {j: hidx(*xy) for j, xy in TGT.items()}
FLOOR = 0.5095; BT = 0.5254; RING = (0.265, -0.200); INTERIOR = (0.1065, 0.6145, -0.580, 0.130)
I = np.eye(3)
def Ry(d): a = math.radians(d); return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
FAR = np.array([[0.32, 0.30, 1.0], [0.32, -0.30, 1.0]])


def frame(pegs_moved=(2, 3), plates_in=True):
    pegs = [((*(TGT[j] if j in pegs_moved else START[j]), FLOOR), I) for j in range(9)]
    if plates_in: pl = [((RING[0], RING[1], BT), I)]
    else: pl = [((0.265, 0.32, 0.75), I)]
    return dict(plates=pl, pegs=pegs, hand_pts=FAR.copy())


def win(**kw): return [frame(**kw) for _ in range(21)]
def sc(frames, lat=None):
    return S.score_window(frames, lat or dict(dropped=False, hard_hit=False), FLOOR, BT, HOLES, start_hole, target_hole, RING, INTERIOR, n_plates=1)


cases = []
def case(name, frames, expect, lat=None):
    s, d = sc(frames, lat); ok = abs(s - expect) < 1e-9; cases.append(ok)
    print('%-30s %s score %.2f (expect %.2f) %s' % (name, 'OK ' if ok else 'BAD', s, expect, d.get('why')))


case('full', win(), 1.0)
case('pegs only, plates on counter', win(plates_in=False), 0.5)
case('one peg, plates on counter', win(pegs_moved=(2,), plates_in=False), 0.25)
case('start state', win(pegs_moved=(), plates_in=False), 0.0)
w = win(); [f['pegs'].__setitem__(3, ((0.4238, -0.0730, FLOOR), I)) for f in w]; case('peg 3 in wrong hole', w, 0.0)
w = win(); [f['pegs'].__setitem__(2, ((TGT[2][0], TGT[2][1], FLOOR + 0.006), I)) for f in w]; case('peg 2 unseated 6 mm', w, 0.0)
case('drop latched', win(), 0.0, dict(dropped=True, hard_hit=False))
case('hard hit latched', win(), 0.0, dict(dropped=False, hard_hit=True))
w = win(); w[-1]['hand_pts'] = np.array([[0.265, -0.07, 0.56]]); case('hand still near', w, 0.0)
w = win(); w[0]['plates'][0] = ((RING[0] + 0.008, RING[1], BT), I); case('plate moved 8 mm in window', w, 0.0)
w = win(); [f['plates'].__setitem__(0, ((RING[0], RING[1], BT), Ry(6))) for f in w]; case('plate tilted 6 deg', w, 0.5)
w = win(); [f['plates'].__setitem__(0, ((RING[0] + 0.012, RING[1], BT), I)) for f in w]
case('plates 12 mm off (inside tol)', w, 1.0)
w = win(); [f['plates'].__setitem__(0, ((RING[0] + 0.020, RING[1], BT), I)) for f in w]
case('plates 20 mm off the ring', w, 0.5)
w = win(); [f['plates'].__setitem__(0, ((RING[0], RING[1], BT + 0.010), I)) for f in w]; case('plate 10 mm above the board', w, 0.5)
w = win(); [f['plates'].append(((RING[0], RING[1], BT + 0.0125), I)) for f in w]; case('a second plate row (wrong count)', w, 0.0)
w = win(pegs_moved=(2,)); case('plates in, peg 3 not moved', w, 0.25)
case('window incomplete', win()[:20], 0.0)
w = win(); w[5]['plates'][0] = ((float('nan'), RING[1], BT), I); case('nan pose', w, 0.0)
w = win(); w[3]['pegs'] = w[3]['pegs'][:8]; case('missing peg row', w, 0.0)
w = win(); w[-1]['hand_pts'] = None; case('hand cover error', w, 0.0)
print('witnesses', sum(cases), '/', len(cases))
