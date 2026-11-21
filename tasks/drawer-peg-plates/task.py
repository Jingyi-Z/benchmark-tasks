"""task_v4p16 = INTEGRATED drawer B (one plate plus both peg moves): task_v4p15 (p17 native PASS: roll 0, lip 30 / open -0.021, aim 125 mm, radius gate, post band 128 +-3, 11-sample 2.0 deg gate) run in full: pegs moved first, then pick, rotating carry to place az 0, place, withdraw, homes; live drawer_plates_score_b + latches. task_v4p15 = task_v4p14 with the pinch AIMED 3 mm deeper (V4_RPIN default 0.125; p16 closed +4 mm outward to 132) and the pre-close radius gate re-centred on the aim (aim +-2 mm); post-close band stays absolute 128 +-3 mm (task and checker, P12PLAN.r_pin_mm = 128). task_v4p14 = task_v4p13 with jaw ROLL 0 on both arms (P7's level-carry setting; roll -/+15 leaned the jaw squeeze 15 deg along the rim tangent, p15 closed 6 mm outward and tilted 19 deg) and post_ok tightened to radii within 128 +-3 mm (the checker band). task_v4p13 = task_v4p12 + p14 fixes (codex 1533): single-plate entry at lip 30.0 mm / open -0.021 (p13's native-proven entry radius; rim boxes untouched along the slide, upper finger 12.4 mm over the rim top, blade 3 mm under the rim underside and 3.1 mm over the counter) and a MEASURED-RADIUS GATE before closing: if either TCP sits outside 126..130 mm of the seen plate centre at the end of centring, both hands close backed out at the approach radius (in air) and the plate is never pinched. task_v4p12 = OPTION B (offline prep, not approved): task_v4p11 with ONE dinner plate (V4_NPLATES default 1, P7 single-plate lip 24.5 mm / open -0.013) and drawer_plates_score_b (plate count parameter). task_v4p11 = INTEGRATED: task_v4p10 (post-close grasp check, centring + loaded-lift trims, 11-sample gate) ahead of the rotating carry, V4_LIFT_ONLY default 0, original peg START, peg yaw 135, and v4p7's measured-link scorer + contact latches (codex 1313). task_v4p10 = task_v4p9 + POST-CLOSE grasp-line check used by the gate, and V4_LIFT_ONLY (default 1): lift, gate, then always set the pair back down on the counter and back out (codex 1234). task_v4p9 = task_v4p8 + visible loaded-lift trim (codex 1213: p12 arms sagged 17-20 mm under the pair). task_v4p8 = task_v4p6 + visible pre-close centring trim and slow lift onset (codex 1210; p12 swung about a pinch line 4.4 mm off the pair CoM). drawer-peg v2 / v3 scene (claude-b50; Jingyi 0810, layout 0806): make room for dinner plates in a pegboard dish drawer.
Run with --embodiment yam_floor_bay (pedestal-mounted YAM arms over a floor, no table). NOT a submission yet.
Positions in m; x forward from the arm-base line, y left; z from the floor bay's world frame (floor at FLOOR_Z).
  * Kitchen base-cabinet run (fixed): countertop top at 0.750 m set back behind the drawer (x 0.590..1.100) and a counter
    section on the robot's +y side (x 0.120..0.440, y 0.035..0.400) holding the plates.
  * Deep drawer (fixed, pulled fully out toward the robot on two full-extension slides): 18 x 18 in interior (457 mm),
    8 in (203.2 mm) tall box, 5/8 in (15.9 mm) sides/front/back, 1/2 in (12.7 mm) bottom; interior x 0.1065..0.5635,
    y -0.457..0.000; box top edge z 0.700; drawer face panel 19 mm thick on the robot side.
  * Pegboard insert (Quik Drawers style, cut to the interior): 5/8 in birch, round holes (27.0 mm, octagon) on the stock
    2-1/2 in grid; holes x = 0.1698 + 63.5 i, y = -0.3270 + 63.5 j inside a 30 mm edge margin.
  * Nine birch dowel pegs, 25.4 mm diameter, 152.4 mm above the board (6-5/8 in overall with the board), 51 g, seated in holes.
    START (salad-plate ring, 3 x 3 holes): (0.1698,-0.3270) (0.3603,-0.3270) (0.1698,-0.1365) (0.3603,-0.1365);
    five spares in the back row x = 0.4873.
  * Two 27 cm glazed ceramic dinner plates, 0.55 kg each, stacked on the +y counter at (0.265, +0.210).
GOAL ring (3 x 4 holes): the two +y pegs one hole +y, to (0.1698,-0.0730) (0.3603,-0.0730); plates flat at the ring centre
C = (0.265, -0.200), lowered in from above (peg tops z 0.678, drawer wall top 0.700). Skeleton: arms parked, settle logs."""
import sys, pathlib, math, json, os
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit
from labkit import part, on_step, on_reset
from yam_arena.demo import Waypoint
from yam_arena.station import FLOOR_Z
SCENE = 'station'; RUN_STEPS = 420
labkit.CONTACT_OFFSET = .0005
BIRCH = (.80, .68, .50); PEGC = (.86, .74, .56); MAPLE = (.72, .56, .38); WHITE = (.94, .94, .92)
CAB = (.90, .89, .86); STONE = (.55, .55, .56); STEELC = (.62, .63, .65)
P = 0.0635; H = 0.0270; T = 0.0159; PEG_D = 0.0254; PEG_L = 0.1524 + T; DB = 0.0127; WALL = 0.0159; BOX_H = 0.2032
Z_CT = 0.750; Z_DT = 0.700; Z_DB = Z_DT - BOX_H            # drawer box bottom underside
IX0, IX1, IY0, IY1 = 0.1065, 0.6145, -0.580, 0.130   # v4: 33 in base-cabinet dish drawer class, interior about 28 x 20 in (711 x 508 mm); dims to be source-checked
Z_IN = Z_DB + DB; BT = Z_IN + T                            # interior floor, board top
C = (0.265, -0.200)
HX = [x for x in (0.1698 + P * i for i in range(-1, 7)) if IX0 + 0.03 < x < IX1 - 0.03]
HY = [y for y in (-0.3270 + P * j for j in range(-3, 6)) if IY0 + 0.03 < y < IY1 - 0.03]


def furniture_boxes():
    b = []
    # base cabinet run behind the drawer, with countertop (front face at x 0.590)
    b.append(((0.895, -0.035, (FLOOR_Z + Z_CT - 0.03) / 2), (0.51, 1.15, Z_CT - 0.03 - FLOOR_Z), CAB))
    b.append(((0.895, -0.035, Z_CT - 0.015), (0.51, 1.15, 0.03), STONE))
    # +y counter section (cabinet + top) beside the drawer
    b.append(((0.280, 0.3475, (FLOOR_Z + Z_CT - 0.03) / 2), (0.32, 0.365, Z_CT - 0.03 - FLOOR_Z), CAB))
    b.append(((0.280, 0.3475, Z_CT - 0.015), (0.32, 0.365, 0.03), STONE))
    # drawer box
    cx, cy = (IX0 + IX1) / 2, (IY0 + IY1) / 2; wx, wy = IX1 - IX0, IY1 - IY0
    b.append(((cx, cy, Z_DB + DB / 2), (wx + 2 * WALL, wy + 2 * WALL, DB), MAPLE))
    hz = Z_DB + DB + (BOX_H - DB) / 2; hh = BOX_H - DB
    b.append(((IX0 - WALL / 2, cy, hz), (WALL, wy + 2 * WALL, hh), MAPLE))
    b.append(((IX1 + WALL / 2, cy, hz), (WALL, wy + 2 * WALL, hh), MAPLE))
    b.append(((cx, IY0 - WALL / 2, hz), (wx, WALL, hh), MAPLE))
    b.append(((cx, IY1 + WALL / 2, hz), (wx, WALL, hh), MAPLE))
    # drawer face panel (19 mm, overlay) and two full-extension slides into the cabinet
    b.append(((IX0 - WALL - 0.0095, cy, Z_DB + 0.125), (0.019, wy + 2 * WALL + 0.04, 0.25), CAB))
    for ys in (IY0 - WALL - 0.007, IY1 + WALL + 0.007):
        b.append((((IX0 + 0.67) / 2, ys, Z_DB + 0.06), (0.67 - IX0, 0.012, 0.035), STEELC))
    return b


USED_HOLES = [(0.1698, -0.3270), (0.3603, -0.3270), (0.1698, -0.1365), (0.3603, -0.1365), (0.1698, -0.0730), (0.3603, -0.0730)] + [(0.4873, -0.3270 + 0.0635 * j) for j in range(5)]


def board_boxes():
    out = []; zc = Z_IN + T / 2
    xs = [IX0] + [v for x in HX for v in (x - H / 2, x + H / 2)] + [IX1]
    ys = [IY0] + [v for y in HY for v in (y - H / 2, y + H / 2)] + [IY1]
    for a, b in zip(xs[0::2], xs[1::2]):
        if b - a > 1e-4: out.append((((a + b) / 2, (IY0 + IY1) / 2, zc), (b - a, IY1 - IY0, T), BIRCH))
    for x in HX:
        for a, b in zip(ys[0::2], ys[1::2]):
            if b - a > 1e-4: out.append(((x, (a + b) / 2, zc), (H, b - a, T), BIRCH))
        for y in HY:
            if not any(abs(x - hx) < 1e-3 and abs(y - hy) < 1e-3 for hx, hy in USED_HOLES): continue
            for sx in (1, -1):
                for sy in (1, -1):
                    d = H / 2 + 0.125 * H
                    out.append(((x + sx * d / math.sqrt(2), y + sy * d / math.sqrt(2), zc), (0.25 * H, H, T), BIRCH, math.radians(45 * sx * sy)))
    return out


KITCHEN = part('kitchen', pos=(0.0, 0.0, 0.0), fixed=True, friction=0.8, boxes=furniture_boxes() + board_boxes())
START = [(0.1698, -0.3270), (0.3603, -0.3270), (0.1698, -0.1365), (0.3603, -0.1365)] + [(0.4873, y) for y in HY[-5:]]   # v4p11 integrated: salad ring + 5 spares (as v4p7)
GOAL_MOVES = {2: (0.1698, -0.0730), 3: (0.3603, -0.0730)}
PEGS = [part(f'peg{k}', pos=(x, y, Z_IN + 0.0005), mass=math.pi * (PEG_D / 2) ** 2 * PEG_L * 600, friction=0.6,
             cylinders=[((0, 0, PEG_L / 2), PEG_D / 2, PEG_L, PEGC, 'Z')]) for k, (x, y) in enumerate(START)]


def plate_prims():
    """270 mm dinner plate (plate frame, z up from the foot): foot ring, well disk, sloped shoulder, and a solid rim ring
    115..135 mm radius, 12..24 mm high (16 segments) that the two hands pinch radially from above."""
    boxes, cyls = [], []
    for k in range(16):      # foot ring (r 0.044..0.050, 4 mm) and well (r 0.050..0.095, 3 mm) as flat box segments (p2: cylinder foot crept)
        a = 2 * math.pi * k / 16
        boxes.append(((0.047 * math.cos(a), 0.047 * math.sin(a), 0.002), (0.006, 2 * 0.050 * math.tan(math.pi / 16) + 0.002, 0.004), WHITE, a))
        boxes.append(((0.0725 * math.cos(a), 0.0725 * math.sin(a), 0.0055), (0.045, 2 * 0.095 * math.tan(math.pi / 16) + 0.002, 0.003), WHITE, a))
    boxes.append(((0, 0, 0.0055), (0.072, 0.072, 0.003), WHITE))   # well centre
    for r_in, r_out, z0, z1 in ((0.095, 0.115, 0.004, 0.012), (0.115, 0.135, 0.012, 0.024)):
        rm = (r_in + r_out) / 2; seg = 2 * r_out * math.tan(math.pi / 16) + 0.002
        for k in range(16):
            a = 2 * math.pi * k / 16
            boxes.append(((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (r_out - r_in, seg, z1 - z0), WHITE, a))
    return boxes, cyls


PB, PC = plate_prims()
PLATE_XY = (0.265, 0.320); PLATE_DZ = 0.0125   # nested stack: the upper plate's foot sits on the lower plate's well
PLATES = [part(f'plate{k}', pos=(PLATE_XY[0], PLATE_XY[1], Z_CT + 0.0005 + k * PLATE_DZ), mass=0.55, friction=0.8, color=WHITE,
               boxes=PB, cylinders=PC) for k in range(int(os.environ.get('V4_NPLATES', '1')))]   # v4p12 option B: one dinner plate   # v4p6: the nested pair (codex 1044 keeps two plates)
OBJECTS = [KITCHEN] + PEGS + PLATES
_step = 0


@on_reset
def reset(env):
    global _step
    _step = 0


@on_step
def observe(env):
    global _step
    _step += 1
    if _step % 20 and _step != 2:
        return
    rec = dict(step=_step, pegs=[], plates=[])
    for p in PEGS:
        s = labkit._part_state(env, p); f = np.asarray(s.pos); R = s.rotation()
        rec['pegs'].append([round(float(f[0]), 4), round(float(f[1]), 4), round((float(f[2]) - Z_IN) * 1e3, 1), round(math.degrees(math.acos(max(-1, min(1, float(R[2, 2]))))), 2)])
    for p in PLATES:
        s = labkit._part_state(env, p); f = np.asarray(s.pos); R = s.rotation()
        rec['plates'].append([round(float(f[0]), 4), round(float(f[1]), 4), round((float(f[2]) - Z_CT) * 1e3, 1), round(math.degrees(math.acos(max(-1, min(1, float(R[2, 2]))))), 2)])
    print('V3LOG ' + json.dumps(rec), flush=True)


# ===== claude-b50 v3 p1: peg-move coupon (pg3 / full1 method: latched seen-foot offset, seen-foot trim) =====
MOVES = [(2, START[2], GOAL_MOVES[2]), (3, START[3], GOAL_MOVES[3])]
SEG = [1.5, 1.0, 0.8, 1.2, 1.5, 1.5, 1.8, 0.8, 1.0, 0.8, 0.8, 0.8]; T0 = 1.5; MOVE_S = sum(SEG)   # + press-home: over the top, press 4 mm, rise
def _win(k):
    c = np.cumsum([0] + SEG) + T0 + k * MOVE_S
    return dict(latch=(int(round((c[4] - 0.6) * 20)), int(round(c[4] * 20))), trim=(int(round(c[5] * 20)), int(round(c[7] * 20))))
WIN = [_win(k) for k in range(len(MOVES))]
_off = [[0.0, 0.0] for _ in MOVES]; _acc = [[] for _ in MOVES]; _n = [0]


@on_reset
def _p1_reset(env):
    _n[0] = 0; _GATE['ok'] = None; _CTR.update(off={1: [0.0, 0.0], -1: [0.0, 0.0]}, ok=None, post_ok=None, prev=None, log={}); _RG.update(abort=False, log={}); _LZ.update(dz=0.0, low=None)
    for k in range(len(MOVES)): _off[k][:] = [0.0, 0.0]; _acc[k].clear()


@on_step
def _p1_step(env):
    _n[0] += 1; n = _n[0]
    for k, ((pi, src, dst), w) in enumerate(zip(MOVES, WIN)):
        if w['latch'][0] <= n <= w['trim'][1]:
            f = np.asarray(labkit._part_state(env, PEGS[pi]).pos)
            if n <= w['latch'][1]:
                _acc[k].append((src[0] - float(f[0]), src[1] - float(f[1])))
                if n == w['latch'][1]: _off[k][:] = list(np.mean(_acc[k], axis=0))
            elif n >= w['trim'][0] and f[2] > BT + 0.001:
                for a_, e in enumerate((dst[0] - float(f[0]), dst[1] - float(f[1]))):
                    _off[k][a_] += float(np.clip(0.2 * e, -3e-4, 3e-4))
    w0, w1 = _CTR['win']
    if w0 <= n <= w1:
        # v4p8 visible centring (codex 1210): move each open hand so the two contacts sit diametric about the plate centre
        # seen in the camera; gain 0.2 per step, clip 0.3 mm per step, only while the plate is still (< 0.5 mm per step)
        from yam_arena import runtime as _rt4
        from yam_arena.constants import TCP_OFFSET as _TO4
        o_ = labkit._np(env.unwrapped.scene.env_origins)[0]; c_ = np.asarray(labkit._part_state(env, PLATES[0]).pos, float)
        still = _CTR['prev'] is None or float(np.linalg.norm(c_[:2] - _CTR['prev'])) < 0.0005; _CTR['prev'] = c_[:2].copy()
        tcp = {}
        for sg_, key in ((1, 'left_robot'), (-1, 'right_robot')):
            g_ = _rt4.link_poses(env.unwrapped.scene[key])['gripper']; tcp[sg_] = (g_[:3, :3] @ np.asarray(_TO4, float) + g_[:3, 3] - o_)[:2]
            a_ = math.radians(90 * sg_ + _CTR['azp']); want = c_[:2] + _CTR['r'] * np.array([math.cos(a_), math.sin(a_)])
            if still and n < w1:
                for i_ in (0, 1): _CTR['off'][sg_][i_] += float(np.clip(0.2 * (want[i_] - tcp[sg_][i_]), -3e-4, 3e-4))
        if n == w1:
            d_ = tcp[1] - tcp[-1]; u_ = d_ / max(float(np.linalg.norm(d_)), 1e-9)
            v_ = c_[:2] - tcp[-1]; line_off = abs(float(u_[0] * v_[1] - u_[1] * v_[0]))
            rL, rR = float(np.linalg.norm(tcp[1] - c_[:2])), float(np.linalg.norm(tcp[-1] - c_[:2]))
            _CTR['ok'] = line_off <= 0.0010 and abs(rL - rR) <= 0.0010
            _RG['abort'] = not (abs(rL - _CTR['r']) <= 0.002 and abs(rR - _CTR['r']) <= 0.002)   # v4p13: measured slide-in reached the pinch radius?
            if _RG['abort']: _CTR['ok'] = False
            _RG['log'] = dict(step=n, r_left_mm=rL * 1e3, r_right_mm=rR * 1e3, band_mm=[_CTR['r'] * 1e3 - 2, _CTR['r'] * 1e3 + 2], abort=_RG['abort'])
            print('P13RG ' + json.dumps(_RG['log']), flush=True)
            _CTR['log'] = dict(step=n, line_offset_mm=line_off * 1e3, r_left_mm=rL * 1e3, r_right_mm=rR * 1e3,
                               off_left_mm=[v * 1e3 for v in _CTR['off'][1]], off_right_mm=[v * 1e3 for v in _CTR['off'][-1]], ok=_CTR['ok'])
            print('P8CTR ' + json.dumps(_CTR['log']), flush=True)
    if n == _CTR.get('pc', -1):
        # v4p10 (codex 1234): the grasp line AFTER the 0.8 s close, before the first rise: both TCPs vs the camera-seen plate centre
        from yam_arena import runtime as _rt5
        from yam_arena.constants import TCP_OFFSET as _TO5
        o5 = labkit._np(env.unwrapped.scene.env_origins)[0]; c5 = np.asarray(labkit._part_state(env, PLATES[0]).pos, float)[:2]; tp = {}
        for sg_, key in ((1, 'left_robot'), (-1, 'right_robot')):
            g_ = _rt5.link_poses(env.unwrapped.scene[key])['gripper']; tp[sg_] = (g_[:3, :3] @ np.asarray(_TO5, float) + g_[:3, 3] - o5)[:2]
        d_ = tp[1] - tp[-1]; u_ = d_ / max(float(np.linalg.norm(d_)), 1e-9); v_ = c5 - tp[-1]
        lo5 = abs(float(u_[0] * v_[1] - u_[1] * v_[0])); rL5, rR5 = float(np.linalg.norm(tp[1] - c5)), float(np.linalg.norm(tp[-1] - c5))
        _CTR['post_ok'] = lo5 <= 0.0010 and abs(rL5 - rR5) <= 0.0010 and abs(rL5 - 0.128) <= 0.003 and abs(rR5 - 0.128) <= 0.003   # v4p15: absolute 128 +-3 mm (checker band)
        print('P10CLOSE ' + json.dumps(dict(step=n, line_offset_mm=lo5 * 1e3, r_left_mm=rL5 * 1e3, r_right_mm=rR5 * 1e3, ok=_CTR['post_ok'],
              pre_close_ok=_CTR['ok'])), flush=True)
    if _LZ['win'][0] <= n <= _LZ['win'][1]:
        # v4p9: raise both hands together until the seen lowest rim point (pair, 16-gon corner 138 mm) is >= 15 mm over the
        # counter; +<= 0.5 mm per step, capped at 30 mm; frozen before the gate samples (gate-10..gate)
        st_ = labkit._part_state(env, PLATES[0]); t_ = math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(st_.rotation())[2, 2])))))
        low_ = float(st_.pos[2]) - Z_CT - 0.138 * math.sin(math.radians(t_)); _LZ['low'] = low_
        _LZ['dz'] = min(0.030, _LZ['dz'] + float(np.clip(0.2 * (0.015 - low_), 0.0, 5e-4)))
        if n == _LZ['win'][1]: print('P9LIFT ' + json.dumps(dict(step=n, dz_mm=_LZ['dz'] * 1e3, lowest_rim_mm=low_ * 1e3, tilt_deg=t_)), flush=True)
    if _GATE['window'] - 10 <= n <= _GATE['window']:
        # v4p6 gate (codex 1044): all 11 samples of the last 0.5 s, every plate: lowest rim point >= 10 mm over the counter
        # throughout, max tilt <= 2 deg, max - min tilt <= 0.3 deg, and the plates' axes within 1 deg of each other (pair moves as one)
        if n == _GATE['window'] - 10: _GATE['samples'] = []
        st = [labkit._part_state(env, p) for p in PLATES]
        zs = [np.asarray(s_.rotation())[:, 2] for s_ in st]
        tg = [math.degrees(math.acos(max(-1.0, min(1.0, float(z_[2]))))) for z_ in zs]
        low = [float(s_.pos[2]) - Z_CT - 0.138 * math.sin(math.radians(t_)) for s_, t_ in zip(st, tg)]   # 16-gon corner radius 137.6 mm
        rel = max([math.degrees(math.acos(max(-1.0, min(1.0, float(zs[0] @ z_))))) for z_ in zs] + [0.0])
        _GATE['samples'].append((low, tg, rel))
        if n == _GATE['window']:
            S = _GATE['samples']; nP = len(PLATES)
            lo_min = min(min(x[0]) for x in S); t_max = max(max(x[1]) for x in S)
            t_rng = max(max(x[1][k] for x in S) - min(x[1][k] for x in S) for k in range(nP)); r_max = max(x[2] for x in S)
            _GATE['ok'] = len(S) == 11 and lo_min >= 0.010 and t_max <= 2.0 and t_rng <= 0.3 and r_max <= 1.0 and _CTR.get('post_ok') is True   # v4p10: the POST-close grasp line, not the stale pre-close trim
            print('P7GATE ' + json.dumps(dict(step=n, samples=len(S), lowest_rim_min_mm=round(lo_min * 1e3, 1), tilt_max_deg=round(t_max, 2),
                  tilt_range_deg=round(t_rng, 2), plate_rel_tilt_max_deg=round(r_max, 2), centring_pre_ok=_CTR['ok'], post_close_ok=_CTR.get('post_ok'), ok=_GATE['ok'])), flush=True)
    if n == 1: print('P12PLAN ' + json.dumps(_PLAN), flush=True)
    # v4p6 (codex 1054): unrounded per-step witness: plate poses + axes, plate 1 relative to plate 0, goal pegs, both TCPs
    _st = [labkit._part_state(env, p) for p in PLATES]; _o = labkit._np(env.unwrapped.scene.env_origins)[0]
    _tcp = {}
    from yam_arena import runtime as _rt2
    from yam_arena.constants import TCP_OFFSET as _TO2
    for side, key in (('L', 'left_robot'), ('R', 'right_robot')):
        g_ = _rt2.link_poses(env.unwrapped.scene[key])['gripper']
        _tcp[side] = [float(v) for v in g_[:3, :3] @ np.asarray(_TO2, float) + g_[:3, 3] - _o]
    _pg = [labkit._part_state(env, PEGS[i]) for i in (2, 3)]
    print('P12LOG ' + json.dumps(dict(step=n, plates=[[float(v) for v in s_.pos] for s_ in _st],
          plate_axis=[[float(v) for v in np.asarray(s_.rotation())[:, 2]] for s_ in _st],
          rel=[float(v) for v in (np.asarray(_st[1].pos) - np.asarray(_st[0].pos))] if len(_st) > 1 else None,
          pegs=[[float(v) for v in s_.pos] + [float(np.asarray(s_.rotation())[2, 2])] for s_ in _pg], tcp=_tcp)), flush=True)
    if n % 5 == 0:
        pl = [labkit._part_state(env, p) for p in PLATES]
        tl = [round(math.degrees(math.acos(max(-1, min(1, float(s_.rotation()[2, 2]))))), 2) for s_ in pl]
        pg = [labkit._part_state(env, PEGS[i]) for i in (2, 3)]
        jaws = {}
        for side, key in (('L', 'left_robot'), ('R', 'right_robot')):
            rob = env.unwrapped.scene[key]; q = rob.data.joint_pos[0]
            jaws[side] = [round(float(q[6]) * 1e3, 1), round(float(q[7]) * 1e3, 1)]
            from yam_arena import runtime as _rt
            from yam_arena.constants import TCP_OFFSET as _TO
            g = _rt.link_poses(rob)['gripper']; o = labkit._np(env.unwrapped.scene.env_origins)[0]
            pt = g[:3, :3] @ np.asarray(_TO, float) + g[:3, 3] - o
            jaws[side + 'tcp'] = [round(float(v), 4) for v in pt]
        print('P5JAW ' + json.dumps(dict(step=n, jaws=jaws)), flush=True)
        print('P1LOG ' + json.dumps(dict(step=n, plates=[[round(float(v), 4) for v in s_.pos] for s_ in pl], plate_tilt=tl,
              moved=[[round(float(s_.pos[0]), 4), round(float(s_.pos[1]), 4), round((float(s_.pos[2]) - Z_IN) * 1e3, 1),
                      round(math.degrees(math.acos(max(-1, min(1, float(s_.rotation()[2, 2]))))), 2)] for s_ in pg],
              off=[[round(v * 1e3, 2) for v in o] for o in _off])), flush=True)


_GATE = dict(window=10 ** 9, ok=None)
_PLAN = {}   # planned step counts, printed once (P12PLAN) for the coupon checker


class _Gate:
    """carry target if the visible lift gate passed, else hold the lift pose over the source counter (never drag)"""
    def __init__(self, go, hold): self.go, self.hold = tuple(go), tuple(hold)
    def __array__(self, dtype=None, copy=None): return np.array(self.go if _GATE['ok'] else self.hold, dtype=dtype or float)
    def __len__(self): return 3
    def __getitem__(self, i): return self.__array__()[i]
    def __iter__(self): return iter(self.__array__())


_CTR = dict(win=(10 ** 9, 10 ** 9), off={1: [0.0, 0.0], -1: [0.0, 0.0]}, ok=None, azp=0.0, r=0.128, prev=None, log={})
_LZ = dict(win=(10 ** 9, 10 ** 9), dz=0.0, low=None)   # v4p9 loaded-lift trim: common z added to both hands' lift..carry targets


_RG = dict(abort=False, vec={1: np.zeros(3), -1: np.zeros(3)}, log={})   # v4p13 measured-radius gate


class _Ct:
    """plate-phase target + the arm's latched visible centring offset (xy)"""
    def __init__(self, sg, inner, lift=False): self.sg, self.inner, self.lift = sg, inner, lift
    def __array__(self, dtype=None, copy=None):
        v = np.array(np.asarray(self.inner, float), dtype=float); o = _CTR['off'][self.sg]; v[0] += o[0]; v[1] += o[1]
        if _RG['abort']: v = v + _RG['vec'][self.sg]   # radius gate failed: every later plate-phase target stays backed out
        if self.lift: v[2] += _LZ['dz']   # lift/carry/hold only (both gate outcomes, no jump); never on set-down, lower or release targets
        return v.astype(dtype or float)
    def __len__(self): return 3
    def __getitem__(self, i): return self.__array__()[i]
    def __iter__(self): return iter(self.__array__())


class _GRot(float):
    """grip rotation that follows the carry only if the lift gate passed, else stays at the pick rotation (no twist on abort)"""
    def __new__(c, go, hold): v = float.__new__(c, 0.0); v.go, v.hold = np.asarray(go), np.asarray(hold); return v
    @property
    def rot(self): return self.go if _GATE['ok'] else self.hold


class _Tr:
    def __init__(self, k, xy, z): self.k, self.xy, self.z = k, xy, z
    def __array__(self, dtype=None, copy=None): return np.array([self.xy[0] + _off[self.k][0], self.xy[1] + _off[self.k][1], self.z], dtype=dtype or float)
    def __len__(self): return 3
    def __getitem__(self, i): return self.__array__()[i]
    def __iter__(self): return iter(self.__array__())


# ===== v4p7 integrated scorer (codex 1116): drawer_plates_score on the actual scene geometry, every control step =====
import drawer_plates_score_b as DPS, drawer_plates_live as DPLV, hand_cover_data as HC
from collections import deque
HOLES = [(x, y) for x in HX for y in HY]
def _hidx(xy): return min(range(len(HOLES)), key=lambda k: math.hypot(HOLES[k][0] - xy[0], HOLES[k][1] - xy[1]))
START_HOLE = [_hidx(xy) for xy in START]; TARGET_HOLE = {j: _hidx(xy) for j, xy in GOAL_MOVES.items()}
_cb = furniture_boxes()[3]; COUNTER_XY = (_cb[0][0] - _cb[1][0] / 2, _cb[0][0] + _cb[1][0] / 2, _cb[0][1] - _cb[1][1] / 2, _cb[0][1] + _cb[1][1] / 2)
INTERIOR = (IX0, IX1, IY0, IY1)
_LAT = DPLV.Latches(Z_CT, COUNTER_XY, BT, INTERIOR, Z_DT, Z_IN)
_FR = deque(maxlen=DPS.WIN); _SC = dict(s=0.0, det={}, n=0)


def _hand_pts(env):
    """certified hand-link cover (424 points per arm) from the measured link poses of both arms"""
    from yam_arena import runtime as _rt3
    o = labkit._np(env.unwrapped.scene.env_origins)[0]; pts = []
    for key in ('left_robot', 'right_robot'):
        lp = _rt3.link_poses(env.unwrapped.scene[key])
        for link, loc in (('gripper', HC.GRIPPER_D405_COVER_LOCAL), ('tip_left', HC.TIP_LEFT_COVER_NATIVE), ('tip_right', HC.TIP_RIGHT_COVER_NATIVE)):
            g = np.asarray(lp[link]); pts.append((g[:3, :3] @ np.asarray(loc).T).T + g[:3, 3] - o)
    return np.vstack(pts)


@on_reset
def _sc_reset(env):
    _LAT.reset(); _FR.clear(); _SC.update(s=0.0, det={}, n=0)


@on_step
def _sc_step(env):
    _SC['n'] += 1; n = _SC['n']
    try:
        pl = [(np.asarray(s_.pos, float), np.asarray(s_.rotation(), float)) for s_ in (labkit._part_state(env, p) for p in PLATES)]
        pg = [(np.asarray(s_.pos, float), np.asarray(s_.rotation(), float)) for s_ in (labkit._part_state(env, p) for p in PEGS)]
        hp = _hand_pts(env)
    except Exception as e:   # fail closed: an unreadable frame makes the window invalid
        pl, pg, hp = [], [], None; _SC['det'] = dict(why='frame error %r' % (e,))
    if len(pl) == len(PLATES): _LAT.step(n, pl, hp)
    _FR.append(dict(plates=pl, pegs=pg, hand_pts=hp))
    s_, det = DPS.score_window(list(_FR), dict(dropped=_LAT.dropped, hard_hit=_LAT.hard_hit), Z_IN, BT, HOLES, START_HOLE,
                               TARGET_HOLE, (C[0], C[1]), INTERIOR, n_plates=len(PLATES))
    _SC.update(s=s_, det=det)
    if n % 20 == 0 or n >= _PLAN.get('total', 10 ** 9) - 21 or (_LAT.events and _LAT.events[-1][0] == n):   # every step of the final window too
        print('P7SCORE ' + json.dumps(dict(step=n, score=s_, why=det.get('why'), dropped=_LAT.dropped, hard_hit=_LAT.hard_hit,
              events=_LAT.events[:3], clear_mm=det.get('min_hand_clear_mm'), still_mm=det.get('still_mm'))), flush=True)


def score(objects): return float(_SC['s'])
def success(objects): return False


def demo():
    from yam_arena import ik
    from yam_arena.constants import READY, LEFT_BASE_POS, RIGHT_BASE_POS, FINGER_OPEN, FINGER_CLOSED, DOWN_XYZW
    from yam_arena.urdf import quat_xyzw_to_matrix
    hL, rL = ik.tcp_pose(np.asarray(READY, float), np.asarray(LEFT_BASE_POS, float))
    hR, rR = ik.tcp_pose(np.asarray(READY, float), np.asarray(RIGHT_BASE_POS, float))
    class O(float):
        def __new__(c, r): v = float.__new__(c, 0.); v.rot = np.asarray(r); return v
    import yam_arena.demo as dm
    if not getattr(dm.grip_rotation, '_o', False):
        _old = dm.grip_rotation
        def g(yaw=0.): return yaw.rot if hasattr(yaw, 'rot') else _old(yaw)
        g._o = True; dm.grip_rotation = g
    a = math.radians(float(os.environ.get('V3_PEG_YAW', '135')))   # v4p11 (as v4p7): 45 puts right joint 6 on its limit at m0 above
    Rd = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]]) @ quat_xyzw_to_matrix(np.asarray(DOWN_XYZW, float))
    w = lambda du, p, f, lab: Waypoint(du, p, f, lab, O(Rd))
    zg = BT + 0.076; lift = 0.045; d = SEG
    right = [Waypoint(T0, hR, FINGER_OPEN, 'start', O(rR))]
    for k, (pi, (x0, y0), (x1, y1)) in enumerate([] if os.environ.get('V3_PLATES_ONLY') else MOVES):
        right += [w(d[0], (x0, y0, zg + 0.06), FINGER_OPEN, f'm{k} above'), w(d[1], (x0, y0, zg), FINGER_OPEN, f'm{k} descend'),
                  w(d[2], (x0, y0, zg), FINGER_CLOSED, f'm{k} grip'), w(d[3], (x0, y0, zg + lift), FINGER_CLOSED, f'm{k} lift'),
                  w(d[4], _Tr(k, (x1, y1), zg + lift), FINGER_CLOSED, f'm{k} carry'), w(d[5], _Tr(k, (x1, y1), zg + 0.030), FINGER_CLOSED, f'm{k} settle'),
                  w(d[6], _Tr(k, (x1, y1), zg + 0.002), FINGER_CLOSED, f'm{k} lower'), w(d[7], _Tr(k, (x1, y1), zg + 0.002), FINGER_OPEN, f'm{k} release'),
                  w(d[8], _Tr(k, (x1, y1), zg + 0.09), FINGER_OPEN, f'm{k} clear'),
                  w(d[9], (x1, y1, BT + 0.1524 + 0.030), FINGER_CLOSED, f'm{k} over top closed'),
                  w(d[10], (x1, y1, BT + 0.1524 + 0.013 - 0.004), FINGER_CLOSED, f'm{k} press home 4 mm'),
                  w(d[11], (x1, y1, BT + 0.1524 + 0.060), FINGER_CLOSED, f'm{k} rise')]
    # ---- plates (p7, codex 0858 option A): outside under-rim pinch. Fingers tilted TILT deg down, approach radially inward
    # from outside the rim; the lower 6.2 mm blade enters the 12 mm gap under the rim lip, the jaws close across the 12 mm
    # lip thickness (half-opening 9 mm). Visible lift gate: carry only if the plate bottom is >= 17 mm over the counter.
    TILT = float(os.environ.get('V3_TILT', '40')); r_pin = float(os.environ.get('V4_RPIN', '0.125')); z_lip = 0.0245; P_OPEN = -0.013   # at 40 deg: lower blade ~1.4 mm over the counter, upper pad ~2 mm over the 24 mm lip top on entry
    def Rx_(d):
        a_ = math.radians(d); c_, s_ = math.cos(a_), math.sin(a_); return np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]])
    Dn = quat_xyzw_to_matrix(np.asarray(DOWN_XYZW, float))
    t_pegs = sum(x.duration for x in right)
    left = [Waypoint(t_pegs + 2.0, hL, FINGER_OPEN, 'hold during pegs', O(rL))]
    right2 = [Waypoint(2.0, hR, FINGER_OPEN, 'home between pegs and plates', O(rR))]
    z_lift = Z_CT + z_lip + 0.026
    for i, (zb_pick, zb_place) in enumerate(((Z_CT, BT),)):   # v4p6: one pick of the whole stack (one plate or the nested pair)
        for arm_list, sg in ((left, +1), (right2, -1)):
            # v4p1 (codex 0952 screen): per-arm grasp from grasp_sweep (tilt, rim azimuth, roll), approach radius 0.16, withdraw 0.165
            # v4p4: both az +3 so the two pinches are diametrically opposite (p10: az +10/-10 put the pinch line 22 mm off the plate CoM,
            # the plate swung 8.6 deg about it); v4p3 (right mirrored) / v4p2: per-arm lip height and half-opening from rim_fit_mesh.py (upper finger over the rim >= 26.5 mm, blade under
            # the 12 mm rim underside, >= 3.4 mm over the counter); left tilt 50 needs the wider -0.017 opening (p8 stalled at -0.013)
            # v4p5 (codex 1025 option 2, offline-screened by rotate_sweep.py): P7's proven tilt-40 grasp (jaw axis 40 deg, carried at
            # 0.42 deg in P7) with opposite pinches, picked at rim azimuth +25 (left reach) and placed at 0 (peg clearance). The plate is
            # round, so both hands turn it 25 deg about its own vertical axis during the carry (pinch span stays 256 mm +-0.1 mm).
            _t, _azp, _azd, _ro, _zl, _po, O0 = [float(v) for v in os.environ.get('V4_L' if sg > 0 else 'V4_R', '40,25,0,0,0.030,-0.021,0.145' if sg > 0 else '40,25,0,0,0.030,-0.021,0.16').split(',')]
            if len(PLATES) == 2:   # nested pair as one lip (box_pair.py): blade under the lower rim, upper pad over the upper rim
                _zl, _po = [float(v) for v in os.environ.get('V4_PAIR', '0.030,-0.021').split(',')]
            _zf = Z_CT + _zl + 0.026; _zp = zb_pick + _zl; _zd = zb_place + _zl + 0.002; O1 = 0.165
            def Rz_(d):
                a_ = math.radians(d); c_, s_ = math.cos(a_), math.sin(a_); return np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
            def Rot(az, sg=sg, _t=_t, _ro=_ro): return Rz_(az) @ Rx_(-sg * (90 - _t)) @ Dn @ np.diag([-1.0, -1.0, 1.0]) @ Rz_(_ro)
            def at(c, rad, z, az, sg=sg): a_ = math.radians(90 * sg + az); return (c[0] + rad * math.cos(a_), c[1] + rad * math.sin(a_), z)
            def W(du, p, f, lab, az, ct=False, lift=False): return Waypoint(du, _Ct(sg, p, lift) if ct else p, f, lab, O(Rot(az)))
            def WG(du, go, hold, f_go, f_hold, lab, az, azh, lift=False):   # gated: carry/place if the gate passed, else the abort pose
                assert f_go == f_hold   # both branches use the same finger command
                return Waypoint(du, _Ct(sg, _Gate(go, hold), lift), f_go, lab, _GRot(Rot(az), Rot(azh)))
            P0 = PLATE_XY
            _a0 = math.radians(90 * sg + _azp); _RG['vec'][sg] = np.array([math.cos(_a0), math.sin(_a0), 0.0]) * (O0 - r_pin)
            _GATE['window'] = int(round((t_pegs + 2.0 + 2.5 + 1.2 + 1.5 + 1.5 + 0.8 + 1.0 + 1.5 + 0.8) * 20))
            _LZ['win'] = (int(round((t_pegs + 2.0 + 2.5 + 1.2 + 1.5 + 1.5 + 0.8) * 20)) + 1, _GATE['window'] - 11)
            _CTR['win'] = (int(round((t_pegs + 2.0 + 2.5 + 1.2 + 1.5) * 20)) + 1, int(round((t_pegs + 2.0 + 2.5 + 1.2 + 1.5 + 1.5) * 20)) - 2)
            _CTR['pc'] = int(round((t_pegs + 2.0 + 2.5 + 1.2 + 1.5 + 1.5 + 0.8) * 20))   # last step of the close, before the first rise
            arm_list += [W(2.5, at(P0, O0, _zp + 0.06, _azp), _po, f'p{i} high outside', _azp),
                         W(1.2, at(P0, O0, _zp, _azp), _po, f'p{i} down outside rim', _azp),
                         W(1.5, at(P0, r_pin, _zp, _azp), _po, f'p{i} slide blade under lip', _azp),
                         W(1.5, at(P0, r_pin, _zp, _azp), _po, f'p{i} centre trim (visible)', _azp, ct=True),
                         W(0.8, at(P0, r_pin, _zp, _azp), FINGER_CLOSED, f'p{i} pinch lip', _azp, ct=True),
                         W(1.0, at(P0, r_pin, _zp + 0.005, _azp), FINGER_CLOSED, f'p{i} lift onset 5 mm', _azp, ct=True, lift=True),
                         W(1.5, at(P0, r_pin, _zf, _azp), FINGER_CLOSED, f'p{i} lift 26 mm', _azp, ct=True, lift=True),
                         W(0.8, at(P0, r_pin, _zf, _azp), FINGER_CLOSED, f'p{i} lift gate hold', _azp, ct=True, lift=True)]
            _CTR['azp'] = _azp; _CTR['r'] = r_pin
            if os.environ.get('V4_LIFT_ONLY', '0') == '1':   # shortest witness: always set the pair back down where it was picked, open, back out, clear, settle
                arm_list += [W(1.5, at(P0, r_pin, _zp, _azp), FINGER_CLOSED, f'p{i} set down on counter', _azp, ct=True),
                             W(0.8, at(P0, r_pin, _zp, _azp), _po, f'p{i} release on counter', _azp, ct=True),
                             W(1.5, at(P0, O0, _zp, _azp), _po, f'p{i} back out', _azp, ct=True),
                             W(1.2, at(P0, O0, _zp + 0.06, _azp), _po, f'p{i} rise 60 mm', _azp, ct=True),
                             Waypoint(2.0, hL if sg > 0 else hR, FINGER_OPEN, f'p{i} up out of drawer (home, clear of the pair)', O(rL if sg > 0 else rR)),
                             Waypoint(1.2, hL if sg > 0 else hR, FINGER_OPEN, f'p{i} settle at home (still window)', O(rL if sg > 0 else rR))]
                continue
            for j, f_ in enumerate([q / 8 for q in range(1, 9)]):
                c_ = (P0[0] + (C[0] - P0[0]) * f_, P0[1] + (C[1] - P0[1]) * f_); az_ = _azp + (_azd - _azp) * f_
                arm_list.append(WG(0.8, at(c_, r_pin, _zf, az_), at(P0, r_pin, _zf, _azp), FINGER_CLOSED, FINGER_CLOSED, f'p{i} carry leg {j}', az_, _azp, lift=True))
            # abort branch (gate false): lower back onto the source counter, open, back out and up; never carry or twist a tilted plate
            arm_list += [WG(1.0, at(C, r_pin, _zf, _azd), at(P0, r_pin, _zf, _azp), FINGER_CLOSED, FINGER_CLOSED, f'p{i} over ring', _azd, _azp, lift=True),
                         WG(2.0, at(C, r_pin, _zd, _azd), at(P0, r_pin, _zp, _azp), FINGER_CLOSED, FINGER_CLOSED, f'p{i} lower into ring', _azd, _azp),
                         WG(0.8, at(C, r_pin, _zd, _azd), at(P0, r_pin, _zp, _azp), _po, _po, f'p{i} release', _azd, _azp),
                         WG(1.5, at(C, O1, _zd, _azd), at(P0, O0, _zp, _azp), _po, _po, f'p{i} withdraw blade', _azd, _azp),
                         WG(1.2, at(C, O1, Z_CT + 0.06, _azd), at(P0, O0, _zp + 0.06, _azp), _po, _po, f'p{i} up out of drawer', _azd, _azp)]
    right = right + right2
    _t_acc = 0.0
    for w_ in right:   # planned step at which the last plate withdrawal ('up out of drawer') ends
        _t_acc += w_.duration
        if 'up out of drawer' in w_.label: _PLAN['withdraw_end'] = int(round(_t_acc * 20))
    _PLAN['gate'] = _GATE['window']; _PLAN['close'] = _CTR['pc']; _PLAN['r_pin_mm'] = 128.0; _PLAN['r_aim_mm'] = _CTR['r'] * 1e3   # checker band centre stays 128
    if not (os.environ.get('V4_LIFT_ONLY', '0') == '1'):
        left += [Waypoint(2.5, hL, FINGER_OPEN, 'left home', O(rL)), Waypoint(3.0, hL, FINGER_OPEN, 'left hold', O(rL))]
        right += [Waypoint(2.5, hR, FINGER_OPEN, 'right home', O(rR)), Waypoint(3.0, hR, FINGER_OPEN, 'right hold', O(rR))]
    _PLAN['total'] = int(round(sum(w_.duration for w_ in right) * 20))
    return dict(left=left, right=right)
