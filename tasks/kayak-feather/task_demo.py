"""kayak-feather MECHANISM coupon (claude-ebeam, codex 0654). NOT a submission. Full 210 cm two-half paddle (the two
outer blade joints are rigid here: the 4-piece's blade joints are separate snap buttons, codex 0654) on two stock-style
canoe blocks, centre Posi-Lok ferrule from ferrule_proto v2 at the LO or HI inferred bound (env KF_BOUND).

World layout: the paddle lies along world y. Ferrule frame F (ferrule_proto docstring) is turned yaw pi, so F +y (the
ferrule / 'right' half) points to world -y, held by the RIGHT arm; the tab shaft (F -y) lies to world +y, pressed by
the LEFT arm. F origin = ferrule mouth at world (X0, YF, ZS).

KF_MODE:
  both : left pads squeeze both tabs flush; right pulls 12 mm (world -y), turns the ferrule half +30 deg about F +y,
         pushes home; left releases, then right. Expect: seated at +30, feather reads +30.
  none : nobody presses; right pulls and tries the turn. Expect: still seated near 0 (hooks on the lip, keys in slots).
  one  : the px tab is driven flush by its own joint target (a fingertip stand-in), the nx tab free; right pulls and
         tries the turn. Expect: still seated near 0.
Logged every 10 steps (KFM json): axial gap (mm), feather (deg, scorer_draft sign), tab joint positions, part tilts."""
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ferrule_proto as FP  # noqa: E402
import labkit  # noqa: E402
import scorer_draft as SD  # noqa: E402
import kayak_motion_waypoints as KM  # noqa: E402  (claude-cake roll-robust motion, KF_MODE=roll)
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, DOWN_XYZW  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402


class Orient(float):
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

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_STEPS = 700
labkit.SOLVER_POS_ITERS = 64
MODE = os.environ.get("KF_MODE", "both")
PULL_N = float(os.environ.get("KF_PULL_N", "15"))
PULL_STEPS = int(os.environ.get("KF_PULL_STEPS", "5"))
if os.environ.get("KF_CO"):
    labkit.CONTACT_OFFSET = float(os.environ["KF_CO"])   # paired contact-offset diagnostic (codex 0715)   # vel mode: sustained pull on the ferrule half (N), body frame
D = FP.dims(os.environ.get("KF_BOUND", "lo"))
X0, YF = 0.35, 0.025
BLOCK_H, CH_D, CH_W = 0.1016, 0.012, 0.0381          # NRS block 4 in tall; channel 38.1 mm wide (sourced), 12 mm deep (inferred)
ZS = TOP + BLOCK_H - CH_D + FP.SHAFT_R               # shaft axis height resting in the channel
HALF_M = 0.865 / 2                                   # sourced total 865 g, split evenly (model)
BLADE = (0.16, 0.46, 0.006)                          # sourced 16 x 46 cm; 6 mm thick (inferred)
foam = (0.15, 0.15, 0.17)
blade_c = (0.85, 0.85, 0.2)


def _jl(j):
    return Joint(j["name"], kind=j["kind"], axis=j["axis"], anchor=j["anchor"], lower=j["lower"], upper=j["upper"],
                 stiffness=j["stiffness"], target=j["target"], damping=2.0, mass=0.01,
                 boxes=j.get("boxes", ()), cylinders=j.get("cylinders", ()))


sh = FP.shapes(D, 0.0)
L, R = sh["left"], sh["right"]
# blades at the outer ends (F y = -1.05 + 0.23 for the tab half, +1.05 - 0.23 for the ferrule half), flat (normal +z)
L_boxes = list(L["boxes"]) + [((0, -1.05 + BLADE[1] / 2, 0), (BLADE[0], BLADE[1], BLADE[2]), blade_c)]
R_boxes = list(R["boxes"]) + [((0, 1.05 - BLADE[1] / 2, 0), (BLADE[0], BLADE[1], BLADE[2]), blade_c)]
L_cyl = [((0, 0.5 * (FP.INS - 0.59), 0), FP.SHAFT_R, 0.59 + FP.INS, (0.08, 0.08, 0.09), "Y")] + [c for c in L["cylinders"] if c[1] > FP.SHAFT_R]
R_cyl = [c for c in R["cylinders"] if abs(c[0][1] - (FP.FER_L + 0.5)) > 1e-6] + [((0, FP.FER_L + 0.265, 0), FP.SHAFT_R, 0.53, (0.08, 0.08, 0.09), "Y")]
PADDLE_L = part("paddle_l", pos=(X0, YF, ZS), mass=HALF_M, yaw=math.pi, boxes=L_boxes, cylinders=L_cyl,
                joints=[_jl(j) for j in L["joints"]], friction=0.6)
SPAWN_DY = float(os.environ.get("KF_SPAWN_DY", "0"))
PADDLE_R = part("paddle_r", pos=(X0, YF - SPAWN_DY, ZS), mass=HALF_M, yaw=math.pi, boxes=R_boxes, cylinders=R_cyl, friction=0.6)


def _block(name, y):
    """NRS-style 4 x 4 x 7 in block, flat face up (channel down), under a blade: the blade lies flat on it, which
    holds the paddle's 0 deg roll (a round shaft alone rolls freely in the 38.1 mm channel: mech1 / mech1b)."""
    h = ZS - BLADE[2] / 2 - TOP
    return part(name, pos=(X0, y, TOP), fixed=True, friction=0.9, boxes=[((0, 0, h / 2), (0.178, 0.1016, h), foam)])


def _channel_block(name, y):
    """STOCK demo support (codex 0723): NRS 4 x 4 x 7 in block, channel up (38.1 mm, sourced; 12 mm deep, inferred), the
    shaft resting in it, blades free"""
    w = (0.178 - CH_W) / 2
    return part(name, pos=(X0, y, TOP), fixed=True, friction=0.9, boxes=[
        ((0, 0, (BLOCK_H - CH_D) / 2), (0.178, 0.1016, BLOCK_H - CH_D), foam),
        ((-(CH_W + w) / 2, 0, BLOCK_H - CH_D / 2), (w, 0.1016, CH_D), foam),
        (((CH_W + w) / 2, 0, BLOCK_H - CH_D / 2), (w, 0.1016, CH_D), foam)])


LAYOUT = os.environ.get("KF_LAYOUT", "diag")       # "diag": blades flat on blocks (lock diagnostic only); "channel": stock demo
if LAYOUT == "channel":
    BLOCK_L = _channel_block("block_l", YF + 0.45)   # under the tab half's shaft
    BLOCK_R = _channel_block("block_r", YF - 0.50)   # under the ferrule half's shaft
else:
    BLOCK_L = _block("block_l", YF + 0.82)           # under the tab half's blade (F y = -0.82)
    BLOCK_R = _block("block_r", YF - 0.82)           # under the ferrule half's blade (F y = +0.82)
MID = os.environ.get("KF_MID", "0") == "1"


def _pad(name, y):
    """flat support under a shaft beside the joint (lock diagnostic only): takes the bending moment off the ferrule"""
    h = ZS - FP.SHAFT_R - TOP
    return part(name, pos=(X0, y, TOP), fixed=True, friction=0.3, boxes=[((0, 0, h / 2), (0.06, 0.04, h), foam)])


OBJECTS = [BLOCK_L, BLOCK_R, PADDLE_L, PADDLE_R] + ([_pad("pad_l", YF + 0.10), _pad("pad_r", YF - 0.20)] if MID else [])   # pad_r clear of the ferrule at a 55 mm spawn
_step = [0]


@on_reset
def _reset(env):
    _step[0] = 0
    KM.reset_roll()
    _WIN.clear()
    _MARKS.clear()
    _LAST.update(score=0.0, why="window incomplete")


def _halves(env):
    sl = labkit._part_state(env, labkit._SPECS["paddle_l"])
    sr = labkit._part_state(env, labkit._SPECS["paddle_r"])
    Rl, Rr = sl.rotation(), sr.rotation()
    Hl = np.column_stack([Rl[:, 2], -Rl[:, 1], np.cross(Rl[:, 2], -Rl[:, 1])])   # blade normal, axis toward its blade
    Hr = np.column_stack([Rr[:, 2], Rr[:, 1], np.cross(Rr[:, 2], Rr[:, 1])])
    gap = float((sr.pos - sl.pos) @ Rl[:, 1])
    return sl, sr, Hl, Hr, gap


@on_step
def _watch(env):
    _step[0] += 1
    if MODE == "roll":          # camera-visible roll: the tab half's blade normal (world), latched by KM at step 90
        _sl = labkit._part_state(env, labkit._SPECS["paddle_l"])
        KM.observe_roll(_sl.rotation()[:, 2], _step[0])
        try:   # camera-visible relative blade angle (both blade normals) for the closed-loop turn
            _sl2, _sr2, _Hl, _Hr, _g = _halves(env)
            _f = SD.feather_deg(_Hl, _Hr)
            KM.observe_feather(_f)
            KM.AXIAL["engage"] = D["engage"]
            KM.observe_axial(_g, _f, _step[0])   # the joint gap is visible in the top view (collar to ferrule mouth)
        except Exception:
            pass
        if _step[0] % 10 == 0:
            print("KFM roll " + json.dumps(dict(step=_step[0], roll_deg=round(KM.ROLL["deg"], 2), latched=KM.ROLL["latched"],
                                                feather_corr=round(KM.FEATHER["corr"], 2), axial_mm=round(KM.AXIAL["y"] * 1e3, 1),
                                                axial_phase=KM.AXIAL["phase"])), flush=True)
    if MODE in ("ctrl", "bothf", "onef", "ctrl_none"):
        # single fresh-reset force test (codex 0723): tabs driven from step 10; 15 N pull on the ferrule half for steps 40..49
        press = {"ctrl_none": (0.0, 0.0), "ctrl": (-D["protrude"], D["protrude"]), "bothf": (-D["protrude"], D["protrude"]), "onef": (-D["protrude"], 0.0)}[MODE]
        try:
            art = env.unwrapped.scene["paddle_l"]
            names = list(art.joint_names)
            tgt = art.data.joint_pos_target.clone()
            if _step[0] >= (1 if MODE == "ctrl" else 10):
                tgt[0, names.index("tab_px_joint")] = press[0]
                tgt[0, names.index("tab_nx_joint")] = press[1]
            art.set_joint_position_target(tgt)
            pr_ = env.unwrapped.scene["paddle_r"]
            import torch
            fz = torch.zeros((1, 1, 3), device=pr_.device)
            if 40 <= _step[0] < 40 + PULL_STEPS:
                fz[0, 0, 1] = PULL_N
            pr_.set_external_force_and_torque(fz, torch.zeros_like(fz))
        except Exception as e:
            print("KFM force error %r" % (e,), flush=True)
    if MODE == "vel":
        # hands-free lock test: phases of 60 steps; tabs driven by their own joint targets; the ferrule half is pushed
        # out along F +y (world -y) by a sustained PULL_N force for steps 20..39 of each phase. none -> one -> both.
        ph, k = divmod(_step[0], 60)
        press = {0: (0.0, 0.0), 1: (-D["protrude"], 0.0), 2: (-D["protrude"], D["protrude"])}.get(ph, (0.0, 0.0))
        try:
            art = env.unwrapped.scene["paddle_l"]
            names = list(art.joint_names)
            tgt = art.data.joint_pos_target.clone()
            tgt[0, names.index("tab_px_joint")] = press[0]
            tgt[0, names.index("tab_nx_joint")] = press[1]
            art.set_joint_position_target(tgt)
            pr_ = env.unwrapped.scene["paddle_r"]
            import torch
            fz = torch.zeros((1, 1, 3), device=pr_.device)
            if 20 <= k < 30 and ph <= 2:
                fz[0, 0, 1] = PULL_N                      # body frame: +y = F +y (out of the ferrule)
            pr_.set_external_force_and_torque(fz, torch.zeros_like(fz))
        except Exception as e:
            print("KFM vel error %r" % (e,), flush=True)
    if MODE == "one":
        try:
            art = env.unwrapped.scene["paddle_l"]
            names = list(art.joint_names)
            tgt = art.data.joint_pos_target.clone()
            tgt[0, names.index("tab_px_joint")] = -D["protrude"]
            art.set_joint_position_target(tgt)
        except Exception as e:  # a failed drive must be visible
            print("KFM drive error %r" % (e,), flush=True)
    try:
        _score_step(env)
    except Exception as e:
        _LAST.update(score=0.0, why="scorer error %r" % (e,))
    if _step[0] % 10 and _step[0] > 12 and not (MODE in ("ctrl", "bothf", "onef", "ctrl_none") and 35 <= _step[0] <= 80):
        return
    sl, sr, Hl, Hr, gap = _halves(env)
    try:
        f = SD.feather_deg(Hl, Hr)
    except ValueError:
        f = float("nan")
    diag = {}
    for side in ("left", "right"):
        try:
            rob = env.unwrapped.scene[side + "_robot"]
            nm = list(rob.joint_names)
            q = labkit._np(rob.data.joint_pos)[0]
            diag[side[0] + "_fing"] = [round(float(q[nm.index(j)]) * 1e3, 1) for j in ("joint7", "joint8") if j in nm]
        except Exception as e:
            diag[side[0] + "_fing"] = repr(e)[:60]
    diag.update(pl=[round(float(v) * 1e3, 1) for v in sl.pos], pr=[round(float(v) * 1e3, 1) for v in sr.pos],
                zl_axis=[round(float(v), 3) for v in sl.rotation()[:, 2]], yl_axis=[round(float(v), 3) for v in sl.rotation()[:, 1]])
    print("KFM " + json.dumps(dict(step=_step[0], mode=MODE, gap_mm=round(gap * 1e3, 2), feather=round(f, 2), score=_LAST["score"], why=_LAST["why"], **diag,
                                   tabs={k: round(v * 1e3, 2) for k, v in sl.joints.items()},
                                   tilt_l=round(sl.tilt_deg(), 2), zl=round(float(sl.pos[2]) * 1e3, 1),
                                   zr=round(float(sr.pos[2]) * 1e3, 1))), flush=True)


_WIN = []          # last 21 frames: (score, why) from scorer_draft.score_frame
_LAST = {"score": 0.0, "why": "window incomplete"}
SHAFT_COVER_R = 0.0195 + 0.002   # capsule radius covering the ferrule OD (hi bound 39 mm) + 2 mm


def _hand_clear(env, sl, sr):
    """certified hand-cover points (shared drawer-peg hand_cover) to both halves' shaft capsules, minus the cover radius"""
    try:
        import hand_cover as HC
        P = np.asarray(HC.certified_hand_points_from_link_poses(HC.extract_robot_links_from_env(env)), float)
    except Exception:
        return -1.0
    best = 9.0
    half = np.array([BLADE[0] / 2, BLADE[1] / 2, BLADE[2] / 2])
    for st, sgn in ((sl, -1.0), (sr, 1.0)):
        R = st.rotation(); a = R[:, 1] * sgn; o = np.asarray(st.pos, float)
        rel = P - o; t = np.clip(rel @ a, -0.01, 0.60)
        d = np.linalg.norm(rel - np.outer(t, a), axis=1) - SHAFT_COVER_R
        best = min(best, float(d.min()))
        # codex 0756: the blade (OBB, centre on the axis 1.05 - BLADE[1]/2 from the ferrule mouth) is covered too
        bc = o + a * (1.05 - BLADE[1] / 2)
        q = np.abs((P - bc) @ R) - half
        best = min(best, float(np.min(np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0))))
    return best - 0.01435


STILL_MAX = 0.005
_MARKS = []        # last 21 frames of marks: both blade tips, both blade edge points, both ferrule-mouth points


def _marks(sl, sr):
    out = []
    for st, sgn in ((sl, -1.0), (sr, 1.0)):
        R = st.rotation(); a = R[:, 1] * sgn; o = np.asarray(st.pos, float)
        tip, bc = o + a * 1.05, o + a * (1.05 - BLADE[1] / 2)
        out += [o, tip, bc + R[:, 0] * BLADE[0] / 2, bc - R[:, 0] * BLADE[0] / 2]
    return np.array(out)


COAX_MAX = 0.0005          # m: right root's offset from the left axis, perpendicular to the shaft
HOOK_MIN = 0.0002           # m: each hook must overlap the clip lip radially by at least this
CRADLE_Z = 0.003           # m: shaft axis height at its block vs the seated height


def _lock_gates(sl, sr, gap):
    """codex 0818: every positive tier needs a coaxial seat, both hooks returned behind the lip, and both shafts in their
    cradles. Returns '' when all hold, else the first failing reason."""
    Rl = sl.rotation(); a = Rl[:, 1]
    d = np.asarray(sr.pos, float) - np.asarray(sl.pos, float)
    off = float(np.linalg.norm(d - a * (d @ a)))
    if off > COAX_MAX:
        return "not coaxial: halves offset %.1f mm across the shaft" % (off * 1e3)
    tabs = getattr(sl, "joints", {}) or {}
    if len(tabs) < 2:
        return "catch not engaged: tab states missing"
    # codex 0828: per-hook authored radial overlap against the actual clip lip, worst side of any coaxial offset, and
    # axial capture of the hook body behind the lip (in the groove band of ferrule_proto)
    for name, q in tabs.items():
        overlap = FP.SHAFT_R + D["hook"] - abs(q) - FP.BORE_R - off
        if overlap < HOOK_MIN:
            return "catch not engaged: %s hook overlaps the lip by %.2f mm (needs %.1f)" % (name, overlap * 1e3, HOOK_MIN * 1e3)
    g0 = FP.HOOK_Y - FP.HOOK_L / 2 - 0.0005                 # lip end = groove start (ferrule_proto shapes)
    capture = (FP.HOOK_Y - FP.HOOK_L / 2) - (g0 + gap)       # hook back face past the lip end
    if capture < 0.0:
        return "catch not engaged: hooks %.2f mm short of the groove (gap %.2f mm)" % (-capture * 1e3, gap * 1e3)
    if LAYOUT == "channel":
        for st, sgn, blk in ((sl, -1.0, BLOCK_L), (sr, 1.0, BLOCK_R)):
            R = st.rotation(); ax = R[:, 1] * sgn; o = np.asarray(st.pos, float)
            if abs(ax[1]) < 0.5:
                return "not cradled: shaft not along the cradle line"
            t = (blk.pos[1] - o[1]) / ax[1]
            p = o + ax * t
            if abs(p[0] - blk.pos[0]) > CH_W / 2 - FP.SHAFT_R + 0.001 or abs(p[2] - ZS) > CRADLE_Z:
                return "not cradled: %s shaft %.1f mm across / %.1f mm off its channel seat" % (
                    blk.name, (p[0] - blk.pos[0]) * 1e3, (p[2] - ZS) * 1e3)
    return ""


def _score_step(env):
    sl, sr, Hl, Hr, gap = _halves(env)
    s, why = SD.score_frame(Hl, Hr, max(gap, 0.0) if gap > -0.002 else 9.9, _hand_clear(env, sl, sr))
    if s > 0.0:
        g = _lock_gates(sl, sr, gap)
        if g:
            s, why = 0.0, g
    _WIN.append((s, why))
    _MARKS.append(_marks(sl, sr))
    del _WIN[:-21]
    del _MARKS[:-21]
    if len(_WIN) < 21:
        _LAST.update(score=0.0, why="window incomplete")
        return
    M = np.stack(_MARKS)                                   # (21, 8, 3)
    still = max(float(np.max(np.linalg.norm(M[:, k][:, None, :] - M[:, k][None, :, :], axis=2))) for k in range(M.shape[1]))
    sc = min(x[0] for x in _WIN)
    why = _WIN[-1][1]
    if still >= STILL_MAX:       # codex 0756: blade tips, blade edges and shaft marks still < 5 mm over the last 1 s
        sc, why = 0.0, "moving: a blade/shaft mark moved %.1f mm in the last 1 s" % (still * 1e3)
    _LAST.update(score=sc, why=why, still_mm=round(still * 1e3, 2))


def score(objects):
    return float(_LAST["score"])


def success(objects):
    return _LAST["score"] >= 1.0


Rd = quat_xyzw_to_matrix(np.asarray(DOWN_XYZW, float))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


G = _rz(math.pi / 2) @ Rd                     # down grip, jaw along world x


def demo():
    if MODE == "roll":
        return KM.motion_waypoints_roll(X0, YF, ZS, TOP, Rd, Waypoint, FINGER_OPEN, FINGER_CLOSED)
    tab = (X0, YF - FP.TAB_Y, ZS)               # F y = TAB_Y maps to world y = YF - TAB_Y (yaw pi)
    hold = (X0, YF - 0.175, ZS)                 # F y = +0.175 on the ferrule half's shaft
    pull = 0.012
    press = FINGER_CLOSED                       # full close: the pads stop on the 30 mm shaft with both tabs flush
    turn = _ry(-math.radians(30.0)) @ G         # +30 deg about F +y = -30 deg about world +y
    lp = [Waypoint(1.0, (0.32, 0.15, TOP + 0.15), FINGER_OPEN, "start", Orient(G)),
          Waypoint(1.5, (tab[0], tab[1], ZS + 0.05), FINGER_OPEN, "above the tabs", Orient(G)),
          Waypoint(1.0, tab, FINGER_OPEN, "down around the tabs", Orient(G))]
    if MODE in ("vel", "ctrl", "bothf", "onef", "ctrl_none"):
        return {"left": [Waypoint(12.0, (0.20, 0.22, TOP + 0.22), FINGER_OPEN, "park well clear", 0.0)],
                "right": [Waypoint(12.0, (0.20, -0.22, TOP + 0.22), FINGER_OPEN, "park well clear", 0.0)]}
    if MODE == "both":
        lp += [Waypoint(1.0, tab, press, "squeeze both tabs flush", Orient(G)),
               Waypoint(8.0, tab, press, "hold both tabs while the right re-indexes", Orient(G)),
               Waypoint(1.0, tab, FINGER_OPEN, "release the tabs: hooks snap behind the lip", Orient(G))]
    else:
        lp += [Waypoint(10.0, tab, FINGER_OPEN, "brace only, no press", Orient(G))]
    lp += [Waypoint(1.0, (tab[0], tab[1], ZS + 0.05), FINGER_OPEN, "lift clear", Orient(G)),
           Waypoint(4.0, (0.32, 0.15, TOP + 0.15), FINGER_OPEN, "park", Orient(G))]
    rp = [Waypoint(1.0, (0.32, -0.15, TOP + 0.15), FINGER_OPEN, "start", Orient(G)),
          Waypoint(1.5, (hold[0], hold[1], ZS + 0.05), FINGER_OPEN, "above the shaft", Orient(G)),
          Waypoint(1.0, hold, FINGER_OPEN, "down on the shaft", Orient(G)),
          Waypoint(1.0, hold, FINGER_CLOSED, "grip the shaft", Orient(G)),
          Waypoint(1.0, hold, FINGER_CLOSED, "wait for the tabs to be pressed", Orient(G)),
          Waypoint(1.5, (hold[0], hold[1] - pull, ZS), FINGER_CLOSED, "pull 12 mm out (world -y)", Orient(G)),
          Waypoint(2.0, (hold[0], hold[1] - pull, ZS), FINGER_CLOSED, "turn +30 deg about the shaft", Orient(turn)),
          Waypoint(1.5, hold, FINGER_CLOSED, "push home", Orient(turn)),
          Waypoint(2.0, hold, FINGER_CLOSED, "hold while the left releases", Orient(turn)),
          Waypoint(1.0, hold, FINGER_OPEN, "release", Orient(turn)),
          Waypoint(1.0, (hold[0], hold[1], ZS + 0.05), FINGER_OPEN, "lift clear", Orient(turn)),
          Waypoint(4.0, (0.32, -0.15, TOP + 0.15), FINGER_OPEN, "park", Orient(G))]
    return {"left": lp, "right": rp}
