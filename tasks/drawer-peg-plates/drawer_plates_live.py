"""Drawer-plates live latches (claude-b50; codex 1116, 1136): sticky DROP and HARD-CONTACT diagnostics from the observed
plate poses at every control step (20 Hz). These are diagnostic definitions from 20 Hz poses: they do not measure a force or
an exact impact speed, and the threshold is not ceramic failure data.
  plate geometry (plate frame, origin at the base centre, z up): 24 points on the foot ring (r 115 mm, z 0), 32 on the
    rim underside edge and 32 on the rim top edge (16-gon corner radius 138 mm, z 12 and 24 mm), posed with the FULL
    rotation every step, so a tilted plate's leading edge is the contacting point.
  supports, each with a signed normal gap (min over the posed points along the support's outward normal):
    horizontal (vertical support): 'counter' (z - counter top, points over the counter box), 'board' (z - board top,
      points inside the drawer), 'nest' (z - (other plate base + 12.5 mm), points over the other plate when it is below),
      'top-x0' .. 'top-y1' (z - wall top, points over each wall slab's footprint);
    walls (finite faces): 'wall-x0' .. 'wall-y1': for every point inside the face's tangential bounds (the other horizontal
      axis over the slab length incl. its thickness, z from the drawer floor to the wall top) the signed distance to the
      inner face, NEGATIVE once a point is past it (a crossing in one step closes the gap from positive to negative).
  HARD: for some support, previous gap > PROX and current gap <= PROX (it ARRIVES at, or crosses, that same support) and
    the normal closing speed (previous gap - current gap) / 50 ms > V_HARD = 0.25 m/s (diagnostic: the 20 Hz closing speed
    of about a 3.2 mm free fall). Motion along a support does not count; a gentle release (<= 3 mm) and the planned
    lowering stay under it. A 20 Hz geometric diagnostic only: no measured impact, force or damage.
  DROP: a plate centre more than 20 mm below the counter top outside the drawer (fell off), its lowest point more than 5 mm
    under the board top (through the floor), a non-finite pose, or a tilt over 45 deg that is RELEASED (no certified
    hand-cover point within 5 mm of the plate envelope; hand points are passed in each step) AND either SUPPORTED (gap
    <= PROX to a horizontal support; wall proximity never counts) or FALLING (lowest point descending faster than V_HARD).
    A held tilt, a controlled reorientation, or a plate near a wall is not a drop.
Latches never clear within an episode."""
import math
import numpy as np

PROX = 0.003; WALL_T = 0.0159; HELD_GAP = 0.005; COVER_R = 0.01435; ENV_H = 0.040; V_HARD = 0.25; NEST_DZ = 0.0125; DT = 0.05; R_FOOT = 0.115; R_RIM = 0.138; Z_RIM_LO = 0.012; Z_RIM_HI = 0.024


def _ring(r, z, n): a = np.linspace(0, 2 * math.pi, n, endpoint=False); return np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]


PTS = np.vstack([_ring(R_FOOT, 0.0, 24), _ring(R_RIM, Z_RIM_LO, 32), _ring(R_RIM, Z_RIM_HI, 32)])


def _tilt(R): return math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(R)[2, 2])))))


class Latches:
    def __init__(self, z_counter, counter_xy, board_top, interior, wall_top, floor_z):
        self.zc, self.cxy, self.bt, self.inr, self.wt, self.fz = z_counter, counter_xy, board_top, interior, wall_top, floor_z
        self.reset()

    def reset(self):
        self.dropped = False; self.hard_hit = False; self.prev_gap = None; self.prev_low = None; self.events = []

    def _gaps(self, k, plates):
        """{support: signed normal gap (m)} for plate k; a support appears only if some posed point lies over/inside it"""
        p, R = plates[k]; W = (np.asarray(R, float) @ PTS.T).T + np.asarray(p, float)
        x0, x1, y0, y1 = self.inr; g = {}
        inC = (W[:, 0] >= self.cxy[0]) & (W[:, 0] <= self.cxy[1]) & (W[:, 1] >= self.cxy[2]) & (W[:, 1] <= self.cxy[3])
        inD = (W[:, 0] >= x0) & (W[:, 0] <= x1) & (W[:, 1] >= y0) & (W[:, 1] <= y1)
        if inC.any(): g['counter'] = float(W[inC, 2].min() - self.zc)
        if inD.any(): g['board'] = float(W[inD, 2].min() - self.bt)
        t = WALL_T; zin = (W[:, 2] >= self.fz) & (W[:, 2] <= self.wt)
        along_y = (W[:, 1] >= y0 - t) & (W[:, 1] <= y1 + t); along_x = (W[:, 0] >= x0 - t) & (W[:, 0] <= x1 + t)
        for nm, sel, d, top in (('x0', along_y, W[:, 0] - x0, (W[:, 0] >= x0 - t) & (W[:, 0] <= x0)),
                                ('x1', along_y, x1 - W[:, 0], (W[:, 0] >= x1) & (W[:, 0] <= x1 + t)),
                                ('y0', along_x, W[:, 1] - y0, (W[:, 1] >= y0 - t) & (W[:, 1] <= y0)),
                                ('y1', along_x, y1 - W[:, 1], (W[:, 1] >= y1) & (W[:, 1] <= y1 + t))):
            m = sel & zin
            if m.any(): g['wall-' + nm] = float(d[m].min())          # signed: negative past the inner face
            m = sel & top
            if m.any(): g['top-' + nm] = float(W[m, 2].min() - self.wt)
        for j, (q, Rq) in enumerate(plates):
            q = np.asarray(q, float)
            if j != k and q[2] < float(np.asarray(p, float)[2]):
                inN = np.hypot(W[:, 0] - q[0], W[:, 1] - q[1]) <= R_RIM
                if inN.any(): g['nest'] = float(W[inN, 2].min() - (q[2] + NEST_DZ))
        return g, float(W[:, 2].min())

    @staticmethod
    def _held(p, R, hand_pts):
        if hand_pts is None: return True   # unknown: never infer a release
        H = np.asarray(hand_pts, float); a = np.asarray(R, float)[:, 2]; rel = H - np.asarray(p, float); z = rel @ a
        rho = np.linalg.norm(rel - np.outer(z, a), axis=1)
        d = np.hypot(np.maximum(rho - R_RIM, 0.0), np.maximum(np.maximum(-z, z - ENV_H), 0.0)) - COVER_R
        return bool(d.min() <= HELD_GAP)

    def step(self, n, plates, hand_pts=None):
        """plates: [(pos, R)] for this step; hand_pts: certified hand-cover points (None = unknown, treated as held).
        Updates and returns (dropped, hard_hit)."""
        try:
            P = np.array([np.asarray(p, float) for p, _ in plates]); Rs = [np.asarray(R, float) for _, R in plates]
            ok = np.all(np.isfinite(P)) and all(np.all(np.isfinite(R)) for R in Rs)
        except Exception:
            ok = False
        if not ok:
            if not self.dropped: self.events.append((n, 'nonfinite plate pose'))
            self.dropped = True; return self.dropped, self.hard_hit
        G = [self._gaps(k, plates) for k in range(len(plates))]
        for k, (p, R) in enumerate(plates):
            p = np.asarray(p, float); gaps, low = G[k]
            falling = self.prev_low is not None and (self.prev_low[k] - low) / DT > V_HARD
            supported = any(v <= PROX for s_, v in gaps.items() if not s_.startswith('wall-'))
            released = not self._held(p, R, hand_pts)
            why = None
            x0, x1, y0, y1 = self.inr
            if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) and p[2] < self.zc - 0.020: why = 'fell off the counter'
            elif low < self.bt - 0.005: why = 'through the floor'
            elif _tilt(R) > 45.0 and released and (supported or falling): why = 'tipped over %.0f deg, released and %s' % (_tilt(R), 'supported' if supported else 'falling')
            if why:
                if not self.dropped: self.events.append((n, 'drop plate %d: %s' % (k, why)))
                self.dropped = True
            if self.prev_gap is not None:
                for s, gnow in gaps.items():
                    gprev = self.prev_gap[k].get(s)
                    if gprev is None or not (gprev > PROX >= gnow): continue
                    v = (gprev - gnow) / DT
                    if v > V_HARD:
                        if not self.hard_hit: self.events.append((n, 'hard contact plate %d at %s, normal closing %.2f m/s (20 Hz diagnostic)' % (k, s, v)))
                        self.hard_hit = True
        self.prev_gap = [g for g, _ in G]; self.prev_low = [lo for _, lo in G]
        return self.dropped, self.hard_hit
