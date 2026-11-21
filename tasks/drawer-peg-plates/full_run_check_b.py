"""OPTION B COPY: N plates (argv[2], default 1; the count on every P12LOG row must equal it; rel must be null with one plate). claude-b50 full-run reader for the integrated task_v4p11 (codex 1333). Usage: full_run_check.py RUN_DIR (run.log, done.txt).
Reads only the task's own telemetry: P12PLAN, the unrounded per-step P12LOG (plates, plate axes, goal pegs 2/3, both TCPs),
P10CLOSE, P7GATE and P7SCORE (the measured-link scorer and its sticky contact latches). Fail-closed; always prints every verdict.
  RUNNER  done.txt 'sim_exit_code 0'; exactly one well-formed P12PLAN (close, gate, withdraw_end, total), P10CLOSE and P7GATE,
          each summary at its planned step; P12LOG steps 1..last each once, last >= total, all finite.
  BRANCH  'carry' if P7GATE ok is true, else 'set-down' (the abort puts the pair back on the source counter).
  RELEASE last 21 consecutive observed samples, all after withdraw_end, SETTLED (every plate centre, goal peg and TCP moves
          <= 1.0 mm) and at the branch's place: carry -> both plates' xy within 15 mm of the ring centre (0.265, -0.200), lower
          base within 3 mm of the board top 0.5254, upper within 3 mm of + 12.5 mm, tilt <= 5 deg; set-down -> both plates'
          xy within 10 mm and height within 3 mm of their step-1 pose, tilt <= 1 deg. Both TCPs >= 80 mm from both rims.
  PEGS    goal pegs 2 and 3 end (whole window) within 1.3 mm of their target holes (0.1698/0.3603, -0.0730), seated (height
          within 3 mm of step 1) and tilted <= 8 deg.
  HAZARD  the LIVE latch first: any P7SCORE row with dropped or hard_hit true (it owns tilt, release and contact). Raw checks,
          kept separate (codex 1343): (a) geometry: a plate centre > 20 mm under the counter top outside the drawer footprint;
          (b) tilt only with the live latch's own conditions reconstructed from the trace: tilt > 45 deg AND released (both
          TCPs > 170 mm from that plate's centre; TCP-centre estimate, a pinched TCP sits ~128 mm out) AND (supported: its
          lowest rim point, 138 mm corner, within 3 mm of the counter top or board top; or falling: lowest point descending
          > 12.5 mm in one step). A held, transient tilt is not a hazard. A hazard forces the expected score to 0.
  SCORE   final P7SCORE rows cover every step of the end window and the last one equals the expected score:
          hazard -> 0; carry + RELEASE + PEGS -> 1.0; set-down + RELEASE + PEGS -> 0.5 (valid pegs, plates safely back);
          anything else -> 0 unless the scorer's own 'why' names it. A scorer row disagreeing with the expectation fails.
  VERDICT  'FULL 1.0', 'PEGS-ONLY 0.5 (plates safely back)', 'HAZARD 0', or 'FAIL' with the first failing check."""
import json, sys, math, os
D = sys.argv[1]; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1; Z_CT = 0.750; RIM = 0.138; BT = 0.5254; DZ = 0.0125; RING = (0.265, -0.200)
TGT = [(0.1698, -0.0730), (0.3603, -0.0730)]; INTERIOR = (0.1065, 0.6145, -0.580, 0.130)
def fin(x):
    if isinstance(x, (list, tuple)): return all(fin(v) for v in x)
    if isinstance(x, dict): return all(fin(v) for v in x.values())
    return isinstance(x, (int, float)) and math.isfinite(x)
def tilt(ax): return math.degrees(math.acos(max(-1.0, min(1.0, ax[2]))))
out = {}
try:
    done = open(os.path.join(D, 'done.txt')).read(); L = open(os.path.join(D, 'run.log')).read().splitlines()
except Exception as e:
    done, L = '', []; out['io'] = str(e)
def rows(tag):
    r = []
    for l in L:
        if l.startswith(tag + ' '):
            try: r.append(json.loads(l.split(' ', 1)[1]))
            except Exception: r.append(None)
    return r
plan, pcs, g7s, scs = rows('P12PLAN'), rows('P10CLOSE'), rows('P7GATE'), rows('P7SCORE')
rec = {}; bad = 0
for r in rows('P12LOG'):
    try:
        relok = (r['rel'] is None) if NP == 1 else (isinstance(r['rel'], list) and len(r['rel']) == 3 and fin(r['rel']))
        ok = (fin([r['plates'], r['plate_axis'], r['pegs'], r['tcp']]) and relok and len(r['plates']) == NP and len(r['plate_axis']) == NP
              and len(r['pegs']) == 2 and set(r['tcp']) == {'L', 'R'})
        if not ok or r['step'] in rec: bad += 1
        else: rec[r['step']] = r
    except Exception:
        bad += 1
P = plan[0] if len(plan) == 1 and isinstance(plan[0], dict) else {}
tot, gate, wend, cstep = P.get('total'), P.get('gate'), P.get('withdraw_end'), P.get('close')
last = max(rec) if rec else 0
summ = (len(pcs) == 1 and isinstance(pcs[0], dict) and pcs[0].get('step') == cstep and len(g7s) == 1 and isinstance(g7s[0], dict)
        and g7s[0].get('step') == gate and all(isinstance(s, dict) and fin([s.get('step'), s.get('score')]) for s in scs) and len(scs) > 0)
runner = ('sim_exit_code 0 ' in done and None not in (tot, gate, wend, cstep) and bad == 0 and summ and last >= tot
          and all(s in rec for s in range(1, last + 1)))
out['runner'] = dict(exit0='sim_exit_code 0 ' in done, plan=P, records=len(rec), last_step=last, malformed_or_dup=bad, summaries_ok=summ)
branch = ('carry' if g7s[0].get('ok') is True else 'set-down') if runner else None
release = pegs = False; hazard = None; score_ok = False; first_fail = None if runner else 'RUNNER'
if runner:
    hz = [s for s in scs if s.get('dropped') or s.get('hard_hit')]
    raw = []; prev_low = {}
    for st in sorted(rec):
        r = rec[st]
        for k in range(NP):
            p = r['plates'][k]; tk = tilt(r['plate_axis'][k]); low = p[2] - RIM * math.sin(math.radians(tk))
            inD = INTERIOR[0] <= p[0] <= INTERIOR[1] and INTERIOR[2] <= p[1] <= INTERIOR[3]
            if not inD and p[2] < Z_CT - 0.020: raw.append((st, k, 'off the counter')); break
            released = min(math.dist(r['tcp'][h], p) for h in 'LR') > 0.170
            supported = abs(low - Z_CT) <= 0.003 or (inD and abs(low - BT) <= 0.003)
            falling = (k in prev_low) and (prev_low[k] - low) > 0.0125
            prev_low[k] = low
            if tk > 45.0 and released and (supported or falling): raw.append((st, k, 'tipped %.0f deg, released and %s' % (tk, 'supported' if supported else 'falling'))); break
        if raw: break
    hazard = bool(hz or raw); out['hazard'] = dict(scorer_rows=len(hz), first_scorer=hz[0] if hz else None, first_raw=raw[0] if raw else None)
    if last - 20 > wend:
        E = [rec[s] for s in range(last - 20, last + 1)]; r1 = rec[1]; w = {}
        def upd(k, v): w[k] = max(w.get(k, -1e9), v)
        marks = lambda r: list(r['plates']) + [r['tcp']['L'], r['tcp']['R']] + [p_[:3] for p_ in r['pegs']]
        w['settle_mm'] = max(math.dist(a[i], b[i]) * 1e3 for a in map(marks, E) for b in map(marks, E) for i in range(NP + 4))
        for r in E:
            pl = sorted(range(NP), key=lambda k: r['plates'][k][2])
            for k in range(NP):
                p = r['plates'][k]; upd('tilt', tilt(r['plate_axis'][k]))
                if branch == 'carry': upd('xy_mm', math.hypot(p[0] - RING[0], p[1] - RING[1]) * 1e3)
                else: upd('xy_mm', math.hypot(p[0] - r1['plates'][k][0], p[1] - r1['plates'][k][1]) * 1e3); upd('dz_mm', abs(p[2] - r1['plates'][k][2]) * 1e3)
            if branch == 'carry':
                for i_, k_ in enumerate(pl): upd('dz_mm', abs(r['plates'][k_][2] - BT - i_ * DZ) * 1e3)
            for j in (0, 1):
                g = r['pegs'][j]; upd('peg_xy_mm', math.hypot(g[0] - TGT[j][0], g[1] - TGT[j][1]) * 1e3)
                upd('peg_dz_mm', abs(g[2] - r1['pegs'][j][2]) * 1e3); upd('peg_tilt', math.degrees(math.acos(max(-1.0, min(1.0, g[3])))))
            upd('hand_near_mm', -min(math.dist(r['tcp'][h], r['plates'][k]) - RIM for h in 'LR' for k in range(NP)) * 1e3)
        w['hand_clear_min_mm'] = -w.pop('hand_near_mm')
        lim = (15.0, 5.0) if branch == 'carry' else (10.0, 1.0)
        release = (w['settle_mm'] <= 1.0 and w['xy_mm'] <= lim[0] and w['tilt'] <= lim[1] and w['dz_mm'] <= 3.0 and w['hand_clear_min_mm'] >= 80.0)
        pegs = w['peg_xy_mm'] <= 1.3 and w['peg_dz_mm'] <= 3.0 and w['peg_tilt'] <= 8.0
        out['end'] = w
        expect = 0.0 if hazard else (1.0 if (branch == 'carry' and release and pegs) else (0.5 if (branch == 'set-down' and release and pegs) else 0.0))
        endrows = {s['step']: s for s in scs if last - 20 <= s['step'] <= last}
        cover = all(s in endrows for s in range(last - 20, last + 1))
        fin_sc = endrows.get(last, {}).get('score') if cover else None
        score_ok = cover and fin_sc is not None and abs(fin_sc - expect) < 1e-9
        out['score'] = dict(expected=expect, scorer_final=fin_sc, scorer_why=endrows.get(last, {}).get('why'), end_rows_cover_window=cover)
    else:
        first_fail = 'END WINDOW (last - 20 <= withdraw_end)'
print('RUNNER', 'PASS' if runner else 'FAIL', out.get('runner'), out.get('io', ''))
print('BRANCH', branch)
print('RELEASE', 'PASS' if release else 'FAIL', out.get('end'))
print('PEGS', 'PASS' if pegs else 'FAIL')
print('HAZARD', 'YES' if hazard else ('NO' if hazard is not None else 'UNKNOWN'), out.get('hazard'))
print('SCORE', 'PASS' if score_ok else 'FAIL', out.get('score'))
if not (runner and score_ok): verdict = 'FAIL (%s)' % (first_fail or ('SCORE: scorer disagrees with the raw end state' if runner else 'RUNNER'))
elif hazard: verdict = 'HAZARD 0'
elif out['score']['expected'] == 1.0: verdict = 'FULL 1.0'
elif out['score']['expected'] == 0.5: verdict = 'PEGS-ONLY 0.5 (plates safely back on the counter)'
else: verdict = 'ZERO 0 (no hazard; %s)' % ('pegs not valid' if not pegs else 'plates not released at the branch place')
print('VERDICT', verdict)
sys.exit(0 if (runner and score_ok) else 1)
