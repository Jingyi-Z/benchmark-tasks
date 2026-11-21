"""kayak-feather two-arm motion block (claude-cake, codex 0655), mergeable into claude-ebeam's task_mech.py.

World layout = task_mech.py: paddle along world y, ferrule frame F turned yaw pi (world y = YF - F_y), F origin at the
ferrule mouth (X0, YF, ZS). LEFT arm braces the tab half and squeezes both tabs; RIGHT arm holds the ferrule half,
pulls it axially (world -y) only after both catches are pressed, turns +30 deg about F +y (= -30 deg about world +y),
pushes home, and the left releases only after the collar is home.

Two changes from task_mech.demo(), from the offline checks in reach_motion.py / witness_motion.py:
1. LEFT pads centred at F y = PAD_FY = -0.040 (task_mech: -0.035, on the tabs). Measured from yam.urdf's tip_left.stl in
   the gripper frame: the fingertip is 27.4 mm wide along the shaft (+-13.7 mm) over its last ~10 mm, widening to about
   +-16 mm by 12 mm above the TCP (the height up to which the key collar, r 18.0 lo / 19.5 hi, reaches the 15 mm finger
   face). At -0.035 the upper finger edge (-0.019) overlaps the hi-bound collar (from -0.020) by ~1 mm; at -0.040 it
   clears it by 4 mm (lo: 8 mm) and the 13.7 mm tip still covers the whole tab head at both bounds (F y -0.041..-0.029).
   (An earlier 07:18 version used -0.058 from a 68 mm pad width; 68 mm is the finger BASE, so -0.058 would miss the tabs.)
2. Approach / lift heights ZS + APPROACH (0.05). Straight down at x 0.35 the left arm misses ZS + 0.08 by 11 mm and
   ZS + 0.10 by 35 mm (IK, 4 seeds, both wrist senses); ZS + 0.03..0.05 solve to <= 0.1 mm for both hands.
Everything else (12 mm pull, 30 deg turn, 8 s hold, release order) is unchanged from task_mech.demo().
"""
import math

import numpy as np

PAD_FY = -0.040
APPROACH = 0.05
HOLD_FY = 0.175
PULL = 0.012
TURN_DEG = 30.0


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def motion_waypoints(X0, YF, ZS, TOP, Rd, Orient, Waypoint, FINGER_OPEN, FINGER_CLOSED, mode="both",
                     pad_fy=PAD_FY, approach=APPROACH, pull=PULL, turn_deg=TURN_DEG, lift=None):
    """Return {"left": [...], "right": [...]} for yam_scripted. Rd = DOWN_XYZW as a matrix; Orient / Waypoint as in task_mech.
    lift=(LL, LR), for the blades-on-blocks layout (mech2): after the press both hands raise the joint (left +LL, right +LR,
    in proportion to their distance from the LEFT blade's block, so the shaft stays straight while the paddle pivots
    there) before the pull and turn, and lower it after the push home. (0.030, 0.038) lifts the right blade's inner end
    ~53 mm, clear of the 40 mm a 160 mm blade drops at 30 deg; reach <= 0.1 mm at (0.025, 0.030) and (0.035, 0.042)."""
    G = _rz(math.pi / 2) @ Rd                             # down grip, jaw along world x
    T = _ry(-math.radians(turn_deg)) @ G                   # +turn about F +y = -turn about world +y
    pad = (X0, YF - pad_fy, ZS)                           # world y = YF + 0.040
    hold = (X0, YF - HOLD_FY, ZS)
    up = lambda p: (p[0], p[1], ZS + approach)
    lp = [Waypoint(1.0, (0.32, 0.15, TOP + 0.15), FINGER_OPEN, "start", Orient(G)),
          Waypoint(1.5, up(pad), FINGER_OPEN, "above the tabs (pads 5 mm outboard, clear of the key collar)", Orient(G)),
          Waypoint(1.0, pad, FINGER_OPEN, "down around the tabs", Orient(G))]
    if mode == "both" and lift:
        padu = (pad[0], pad[1], ZS + lift[0])
        lp += [Waypoint(1.0, pad, FINGER_CLOSED, "squeeze both tabs flush", Orient(G)),
               Waypoint(1.0, padu, FINGER_CLOSED, "lift the joint with the right (tabs held)", Orient(G)),
               Waypoint(5.0, padu, FINGER_CLOSED, "hold both tabs while the right re-indexes", Orient(G)),
               Waypoint(1.0, pad, FINGER_CLOSED, "lower the joint with the right", Orient(G)),
               Waypoint(1.0, pad, FINGER_CLOSED, "settle on the blocks", Orient(G)),
               Waypoint(1.0, pad, FINGER_OPEN, "release the tabs: hooks snap behind the lip", Orient(G))]
    elif mode == "both":
        lp += [Waypoint(1.0, pad, FINGER_CLOSED, "squeeze both tabs flush", Orient(G)),
               Waypoint(8.0, pad, FINGER_CLOSED, "hold both tabs while the right re-indexes", Orient(G)),
               Waypoint(1.0, pad, FINGER_OPEN, "release the tabs: hooks snap behind the lip", Orient(G))]
    else:
        lp += [Waypoint(10.0, pad, FINGER_OPEN, "brace only, no press", Orient(G))]
    lp += [Waypoint(1.0, up(pad), FINGER_OPEN, "lift clear", Orient(G)),
           Waypoint(4.0, (0.32, 0.15, TOP + 0.15), FINGER_OPEN, "park", Orient(G))]
    rp = [Waypoint(1.0, (0.32, -0.15, TOP + 0.15), FINGER_OPEN, "start", Orient(G)),
          Waypoint(1.5, up(hold), FINGER_OPEN, "above the shaft", Orient(G)),
          Waypoint(1.0, hold, FINGER_OPEN, "down on the shaft", Orient(G)),
          Waypoint(1.0, hold, FINGER_CLOSED, "grip the shaft", Orient(G)),
          ] + ([
          Waypoint(1.0, (hold[0], hold[1], ZS + lift[1]), FINGER_CLOSED, "lift the joint with the left", Orient(G)),
          Waypoint(1.5, (hold[0], hold[1] - pull, ZS + lift[1]), FINGER_CLOSED, "pull out (world -y)", Orient(G)),
          Waypoint(2.0, (hold[0], hold[1] - pull, ZS + lift[1]), FINGER_CLOSED, "turn about the shaft", Orient(T)),
          Waypoint(1.5, (hold[0], hold[1], ZS + lift[1]), FINGER_CLOSED, "push home", Orient(T)),
          Waypoint(1.0, hold, FINGER_CLOSED, "lower the joint with the left", Orient(T)),
          Waypoint(2.0, hold, FINGER_CLOSED, "hold until the left has released (left opens 12.5..13.5 s)", Orient(T)),
          ] if lift else [
          Waypoint(1.0, hold, FINGER_CLOSED, "wait for the tabs to be pressed", Orient(G)),
          Waypoint(1.5, (hold[0], hold[1] - pull, ZS), FINGER_CLOSED, "pull out (world -y)", Orient(G)),
          Waypoint(2.0, (hold[0], hold[1] - pull, ZS), FINGER_CLOSED, "turn about the shaft", Orient(T)),
          Waypoint(1.5, hold, FINGER_CLOSED, "push home", Orient(T)),
          Waypoint(3.0, hold, FINGER_CLOSED, "hold until the left has released (left opens 12.5..13.5 s)", Orient(T)),
          ]) + [

          Waypoint(1.0, hold, FINGER_OPEN, "release", Orient(T)),
          Waypoint(1.0, up(hold), FINGER_OPEN, "lift clear", Orient(T)),
          Waypoint(4.0, (0.32, -0.15, TOP + 0.15), FINGER_OPEN, "park", Orient(G))]
    return {"left": lp, "right": rp}


# ---------------------------------------------------------------------------------------------------------------------
# ROLL-ROBUST version for the stock shaft-in-channel layout (codex 0723): the round shaft rolls freely in the channels
# (mech1b: 15 deg by step 10, 30 deg by the end). The demo reads the common roll ONLY from what a camera sees, the tab
# half's blade normal n (alpha = atan2(n_x, n_z), roll about world +y), and latches it once the right hand has closed on
# the shaft (which stops the roll). Then:
#   alpha <= 0 : the left tab grip tilts to the rolled tab line, Ry(alpha) @ G (reach <= 0.1 mm for -30..0);
#   alpha  > 0 : the right hand first squares the locked paddle back to 0 roll by turning Ry(-alpha) (reach <= 0.1 mm for
#                +10..+30); a tilted left grip is out of reach there (55 to 107 mm, both wrist senses).
# The right hand's feather turn is always a further -30 deg about world y from where it holds, so the score (relative
# feather) is independent of the common roll. Order: the right grips first; the left presses before the right pulls.
ROLL = {"deg": 0.0, "latched": False, "step": 0}
LATCH_STEP = 90                                        # 4.5 s at 20 Hz: end of the right's "grip the shaft"


def observe_roll(blade_normal_world, step):
    """Call from the task's on_step with the tab half's blade normal (world). Camera-visible only."""
    n = np.asarray(blade_normal_world, float)
    ROLL["step"] = step
    if not ROLL["latched"]:
        ROLL["deg"] = math.degrees(math.atan2(n[0], n[2]))
        if step >= LATCH_STEP:
            ROLL["latched"] = True
            print(f"KFM roll latched at step {step}: {ROLL['deg']:+.2f} deg", flush=True)


def reset_roll():
    ROLL.update(deg=0.0, latched=False, step=0)
    AXIAL.update(y=0.0, phase="idle", ok=0)
    FEATHER.update(deg=0.0, corr=0.0)          # on/off steps are kept: they come from the plan, not the trial


# claude-ebeam (after kd2): closed-loop feather. kd2 commanded -30 deg and the blades read 39 to 42 deg (about 10 deg
# overshoot), so the keys met a slot wall and the push stopped at a 6.2 mm gap. The demo now trims its turn from the
# camera-visible relative blade angle (both blade normals), integrating only once the turn is under way (|f| > 5 deg).
FEATHER = {"deg": 0.0, "corr": 0.0, "on_step": 10 ** 9, "off_step": 10 ** 9}
FEATHER_GAIN, FEATHER_CLIP = 0.15, 20.0


def observe_feather(f_deg, turn_deg=30.0):
    """trim only between the end of the commanded turn and the start of the push home (planner steps set below)"""
    FEATHER["deg"] = float(f_deg)
    if FEATHER["on_step"] <= ROLL["step"] < FEATHER["off_step"] and abs(f_deg) > 5.0:
        c = FEATHER["corr"] + FEATHER_GAIN * (turn_deg - f_deg)
        FEATHER["corr"] = max(-FEATHER_CLIP, min(FEATHER_CLIP, c))


# claude-ebeam (after kd7): closed-loop AXIAL motion. The right pinch slips by a varying amount (a 32 mm commanded pull
# opened 19 mm in kd7_hi but 6 mm in kd7_lo), so the right hand's axial offset is now driven by the visible joint gap:
# pull until gap >= engage + 4 mm, hold through turn and trim, enter only after the visible feather is inside the
# geometric entry band (+-0.9 deg, codex 0809) for 5 steps, then press 4 mm past home. Steps are capped at 1 mm (20 mm/s).
AXIAL = {"y": 0.0, "phase": "idle", "ok": 0, "pull_on": 10 ** 9, "push_on": 10 ** 9, "off": 10 ** 9, "engage": 0.006}
AX_GAIN, AX_STEP, ENTRY_BAND = 0.5, 0.001, 0.9


def observe_axial(gap_m, f_deg, step, turn_deg=30.0):
    e = AXIAL["engage"]
    if step < AXIAL["pull_on"] or step >= AXIAL["off"]:
        return
    if step < AXIAL["push_on"]:
        target, AXIAL["phase"] = e + 0.004, "pull/turn/trim"
    else:
        inside = abs(f_deg - turn_deg) <= ENTRY_BAND
        AXIAL["ok"] = AXIAL["ok"] + 1 if inside else 0
        if gap_m < e - 0.001 or AXIAL["ok"] >= 5 or AXIAL["phase"] == "enter":
            target, AXIAL["phase"] = -0.004, "enter"
        else:
            target, AXIAL["phase"] = e + 0.002, "wait at the mouth"
    AXIAL["y"] += max(-AX_STEP, min(AX_STEP, AX_GAIN * (target - gap_m)))
    AXIAL["y"] = max(-0.03, min(0.06, AXIAL["y"]))


class LivePos:
    """waypoint position whose world y is shifted by -AXIAL['y'] (F +y = world -y) each time the planner reads it"""
    def __init__(self, p):
        self.p = tuple(float(v) for v in p)

    def __array__(self, dtype=None, copy=None):
        return np.array([self.p[0], self.p[1] - AXIAL["y"], self.p[2]], dtype=dtype or float)

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return self.__array__()[i]

    def __iter__(self):
        return iter(self.__array__())


class LiveRot(float):
    """yaw stand-in whose .rot is evaluated each planner step from the latched roll (grip hooks test hasattr(yaw, 'rot'))."""
    def __new__(cls, fn):
        o = float.__new__(cls, 0.0)
        o._fn = fn
        return o

    @property
    def rot(self):
        return np.asarray(self._fn(ROLL["deg"]), float)


def motion_waypoints_roll(X0, YF, ZS, TOP, Rd, Waypoint, FINGER_OPEN, FINGER_CLOSED,
                          pad_fy=PAD_FY, approach=APPROACH, pull=0.032, push_over=0.016, t_axial=2.5, turn_deg=TURN_DEG,
                          park_l=(0.25, 0.30, None), park_r=(0.25, -0.30, None)):
    """kd2 (after kd1): the right's shaft pinch slips ~60 % axially (12 mm commanded opened the joint 4.9 mm, under the
    6 mm key depth, so the turn could not index), hence a 22 mm commanded pull at 8.8 mm/s and a push 4 mm past home;
    park far clear of the scorer's 80 mm hand-clearance window (kd1 parked ~41 mm above the shafts)."""
    G = _rz(math.pi / 2) @ Rd
    sq = lambda a: math.radians(-a) if a > 0 else 0.0                      # right's squaring turn
    left_R = lambda a: (_ry(math.radians(a)) @ G) if a <= 0 else G          # left tab grip
    R_hold = LiveRot(lambda a: _ry(sq(a)) @ G)
    R_turn = LiveRot(lambda a: _ry(sq(a) - math.radians(turn_deg + FEATHER["corr"])) @ G)
    L_grip = LiveRot(left_R)
    Gs = LiveRot(lambda a: G)
    pad = (X0, YF - pad_fy, ZS)
    hold = (X0, YF - HOLD_FY, ZS)
    up = lambda p: (p[0], p[1], ZS + approach)
    # claude-ebeam: the old (0.25, +-0.30, TOP+0.28) park at the yaw-90 grip failed IK by 77.8 mm+deg (ik_check), so the arm
    # never reached it. Use the reachable default-grip park (0.20, +-0.22, TOP+0.22): 189 mm from the shaft axis.
    pk_l = (0.20, 0.22, TOP + 0.22)
    pk_r = (0.20, -0.22, TOP + 0.22)
    rp = [Waypoint(1.0, (0.32, -0.15, TOP + 0.15), FINGER_OPEN, "start", Gs),
          Waypoint(1.5, up(hold), FINGER_OPEN, "above the shaft", Gs),
          Waypoint(1.0, hold, FINGER_OPEN, "down on the shaft", Gs),
          Waypoint(1.0, hold, FINGER_CLOSED, "grip the shaft (stops the free roll; roll latched at 4.5 s)", Gs),
          Waypoint(1.0, hold, FINGER_CLOSED, "square the paddle to 0 roll if it rolled positive", R_hold),
          Waypoint(3.5, hold, FINGER_CLOSED, "hold square while the left comes down and presses", R_hold),
          Waypoint(t_axial, LivePos(hold), FINGER_CLOSED, "pull out (world -y)", R_hold),
          Waypoint(2.0, LivePos(hold), FINGER_CLOSED, "turn the feather", R_turn),
          Waypoint(1.5, LivePos(hold), FINGER_CLOSED, "trim the feather on the visible blades", R_turn),
          Waypoint(t_axial, LivePos(hold), FINGER_CLOSED, "push home (4 mm past, slip)", R_turn),
          Waypoint(4.0, LivePos(hold), FINGER_CLOSED, "hold until the left has released (left opens 19..20 s)", R_turn),
          Waypoint(1.0, LivePos(hold), FINGER_OPEN, "release", R_turn),
          Waypoint(1.0, up(hold), FINGER_OPEN, "lift clear", R_turn),
          Waypoint(4.0, pk_r, FINGER_OPEN, "park far clear (default grip)", 0.0)]
    _t, _labs = 0.0, {}
    for w in rp:
        _t += w.duration
        _labs[w.label] = _t
    FEATHER["on_step"] = int(round(_labs["turn the feather"] * 20))
    # kd3: the angle held 30.6 at the end of the trim, then drifted to 38-39 deg DURING the push (wrist drift under the
    # axial load) before the keys reached the slots. Keep trimming through the push until the collar is home.
    FEATHER["off_step"] = int(round(_labs["push home (4 mm past, slip)"] * 20))
    AXIAL["pull_on"] = int(round(_labs["hold square while the left comes down and presses"] * 20))
    AXIAL["push_on"] = int(round(_labs["trim the feather on the visible blades"] * 20))
    AXIAL["off"] = int(round(_labs["hold until the left has released (left opens 19..20 s)"] * 20))
    lp = [Waypoint(5.5, (0.32, 0.15, TOP + 0.15), FINGER_OPEN, "wait for the right's grip and squaring", Gs),
          Waypoint(1.5, up(pad), FINGER_OPEN, "above the tabs (pads 5 mm outboard of the tab centre)", L_grip),
          Waypoint(1.0, pad, FINGER_OPEN, "down around the tabs", L_grip),
          Waypoint(1.0, pad, FINGER_CLOSED, "squeeze both tabs flush", L_grip),
          Waypoint(6.5 + 2 * t_axial, pad, FINGER_CLOSED, "hold both tabs through pull, turn, trim and push", L_grip),
          Waypoint(1.0, pad, FINGER_OPEN, "release the tabs: hooks snap behind the lip", L_grip),
          Waypoint(1.0, up(pad), FINGER_OPEN, "lift clear", L_grip),
          Waypoint(4.0, pk_l, FINGER_OPEN, "park far clear (default grip)", 0.0)]
    return {"left": lp, "right": rp}
