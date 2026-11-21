"""OPTION B COPY (one dinner plate; plate count is a parameter, default 1). Same rules as drawer_plates_score.py, with "both plates" read as "every plate". Drawer-plates (drawer-peg v2) end-state scorer, claude-b50 owner. Pure functions; the task's on_step feeds one frame per
control step and keeps the latches. Rules from Jingyi 0810 (relayed by claude-work), every check over the FINAL 21-frame
window (1.0 s at 20 Hz) and fail-closed on missing or non-finite data:
  frame = dict(plates=[(pos(3), R(3x3)) x2], pegs=[(foot(3), R(3x3)) x9], hand_pts=(N,3) certified cover points or None)
  latches = dict(dropped=bool, hard_hit=bool)   set by on_step and never cleared (a plate below the counter outside the
                                                drawer, or a plate speed over 0.5 m/s at contact)
  Z  any latch, window incomplete or invalid frame                                         -> 0
  W  released and still: every frame's certified hand clearance >= 80 mm (cover minus 14.35 mm) to both plate envelopes
     and every peg shaft; every mark (plate centres, two rim marks per plate, every peg foot and top) moves < 5 mm, else 0
  H  every peg seated in EVERY frame (foot within 3 mm of the drawer floor, tilt <= 8 deg, foot within 1.3 mm of a hole
     centre) and only in its own start hole or, for a target peg, its target hole; any other hole or an unseated peg -> 0
  pegs   +0.25 per target peg in its target hole in every frame (up to 2)
  plates +0.5 if in every frame both plates are flat (tilt <= 5 deg), centred in the ring (centre within 15 mm of the ring
     centre), resting (lower plate base within 3 mm of the board top, upper within 3 mm of board top + 12.5 mm nest),
     clear of every drawer wall by >= 2 mm (16-gon corner radius 138 mm) and clear of every peg by its radius
  score = 0.25 x pegs + 0.5 x plates (1.0 = both pegs moved and both plates placed)"""
import math
import numpy as np

WIN = 21; CLEAR_MIN = 0.080; COVER_R = 0.01435; STILL_MAX = 0.005
SEAT_TOL = 0.003; PEG_TILT = 8.0; HOLE_TOL = (0.0270 - 0.0254) / 2 + 0.0005; PEG_R = 0.0127; PEG_L = 0.1683
PLATE_R = 0.138; PLATE_ENV_H = 0.040; PLATE_TILT = 5.0; RING_TOL = 0.015; REST_TOL = 0.003; NEST_DZ = 0.0125; WALL_GAP = 0.002


def _tilt(R): return math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(R)[2, 2])))))


def hand_clearance(hand_pts, plates, pegs):
    """conservative lower bound (m): cover points to each plate envelope (cylinder r 138 mm, 0..40 mm along its axis) and
    to every peg capsule, minus the cover radius; -1 on a cover error"""
    if hand_pts is None: return -1.0
    H = np.asarray(hand_pts, float); best = 9.0
    for p, R in plates:
        a = np.asarray(R)[:, 2]; rel = H - np.asarray(p); z = rel @ a; rho = np.linalg.norm(rel - np.outer(z, a), axis=1)
        d = np.hypot(np.maximum(rho - PLATE_R, 0.0), np.maximum(np.maximum(-z, z - PLATE_ENV_H), 0.0)); best = min(best, float(d.min()))
    for f, R in pegs:
        ax = np.asarray(R)[:, 2]; rel = H - np.asarray(f); t = np.clip(rel @ ax, 0.0, PEG_L)
        best = min(best, float((np.linalg.norm(rel - np.outer(t, ax), axis=1) - PEG_R).min()))
    return best - COVER_R


def marks(fr):
    m = []
    for p, R in fr['plates']:
        p = np.asarray(p, float); R = np.asarray(R, float)
        m += [p, p + R @ np.array([0.128, 0, 0.02]), p + R @ np.array([0, 0.128, 0.02])]
    for f, R in fr['pegs']: m += [np.asarray(f, float), np.asarray(f, float) + np.asarray(R, float)[:, 2] * PEG_L]
    return np.vstack(m)


def _hole_of(f, R, floor_z, holes_xy):
    f = np.asarray(f, float)
    if abs(float(f[2]) - floor_z) > SEAT_TOL or _tilt(R) > PEG_TILT: return None
    d, k = min((math.hypot(float(f[0]) - hx, float(f[1]) - hy), k) for k, (hx, hy) in enumerate(holes_xy))
    return k if d <= HOLE_TOL else None


def _plates_ok(fr, board_top, ring_xy, interior, peg_xy):
    ps = sorted(fr['plates'], key=lambda q: float(np.asarray(q[0])[2]))
    x0, x1, y0, y1 = interior
    for j, (p, R) in enumerate(ps):
        p = np.asarray(p, float)
        if _tilt(R) > PLATE_TILT or math.hypot(p[0] - ring_xy[0], p[1] - ring_xy[1]) > RING_TOL: return False
        if abs(p[2] - (board_top + j * NEST_DZ)) > REST_TOL: return False
        if min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]) - PLATE_R < WALL_GAP: return False
        if any(math.hypot(p[0] - x, p[1] - y) < PLATE_R + PEG_R for x, y in peg_xy): return False
    return True


def score_window(frames, latches, floor_z, board_top, holes_xy, start_hole, target_hole, ring_xy, interior, n_plates=1):
    """frames: the last WIN frames, oldest first. start_hole[j]: peg j's start hole index; target_hole: {peg j: hole index}.
    Returns (score, detail)."""
    if latches.get('dropped') or latches.get('hard_hit'):
        return 0.0, dict(why='latched: %s' % sorted(k for k, v in latches.items() if v))
    if len(frames) < WIN: return 0.0, dict(why='window incomplete (%d of %d)' % (len(frames), WIN))
    frames = frames[-WIN:]
    try:
        V = []
        for fr in frames:
            if len(fr['plates']) != n_plates or len(fr['pegs']) != len(start_hole): raise ValueError('wrong object count')
            cl = hand_clearance(fr['hand_pts'], fr['plates'], fr['pegs'])
            M = marks(fr)
            if not (math.isfinite(cl) and np.all(np.isfinite(M))): raise ValueError('nonfinite')
            V.append(dict(clear=cl, marks=M, holes=[_hole_of(f, R, floor_z, holes_xy) for f, R in fr['pegs']],
                          plates=_plates_ok(fr, board_top, ring_xy, interior, [np.asarray(f, float)[:2] for f, _ in fr['pegs']])))
        S = np.stack([v['marks'] for v in V])
        still = max(float(np.linalg.norm(S[:, k][:, None] - S[:, k][None], axis=2).max()) for k in range(S.shape[1]))
    except (KeyError, ValueError, TypeError, IndexError) as e:
        return 0.0, dict(why='invalid frame: %s' % e)
    det = dict(min_hand_clear_mm=round(min(v['clear'] for v in V) * 1e3, 1), still_mm=round(still * 1e3, 3))
    if min(v['clear'] for v in V) < CLEAR_MIN or still >= STILL_MAX:
        det['why'] = 'not released/still'; return 0.0, det
    for v in V:
        for j, h in enumerate(v['holes']):
            if h is None or h not in (start_hole[j], target_hole.get(j, -1)):
                det['why'] = 'peg %d %s' % (j, 'not seated' if h is None else 'in wrong hole %d' % h); return 0.0, det
    n_t = sum(1 for j, h in target_hole.items() if all(v['holes'][j] == h for v in V))
    pl = all(v['plates'] for v in V)
    det.update(target_pegs=n_t, plates_placed=pl, why='pegs %d/2, plates %s' % (n_t, 'placed' if pl else 'not placed'))
    return 0.25 * n_t + (0.5 if pl else 0.0), det
