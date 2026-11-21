"""kayak-feather: bounded Posi-Lok centre-ferrule geometry prototype v2 (claude-ebeam, codex 0654). Pure python; builds
labkit-style shape lists and runs CPU geometry gates at both inferred dimension bounds. No sim.

Mechanism, after US 8,801,483 B2: the LEFT centre shaft (the patent's shaft 11) carries two opposed push tabs through
holes in its wall, each on a resilient arm with a catch hook. The RIGHT centre shaft (shaft 12) carries the external
ferrule whose bushing has a full-circle clip lip. Seated, both hooks sit behind the lip, so the joint is axially locked at
ANY angle; the angle itself is fixed by the exterior mating cogs (left keys in right cog slots, 15 deg index set).
Pressing ONE tab moves only its own hook in; the other hook still holds the lip. Pressing BOTH frees the pull-out; once
the keys clear the cogs (pull > engage) the halves turn freely; re-insert at the new index and release.

FRAME F (shared with claude-cake's motion block): origin on the shaft axis at the ferrule MOUTH face (right half);
+y toward the RIGHT blade; tabs on +x and -x at y = TAB_Y (outside the ferrule, so a straight-down grip with the jaw
along world x presses both); +z up. 0 deg: both blade normals +z. Right-hand 30 deg = right half turned +30 deg about +y.

SOURCED: shaft OD 30.0 mm (Aqua Bound FAQ); dual tabs, opposed arms, catches on a clip lip, exterior cogs, the
press-pull-rotate-slide sequence (US 8,801,483 B2); 0/15/30/45/60 L/R index set (Aqua Bound).
INFERRED (labelled, LO..HI bounds; verified before trials): everything in BOUNDS. Real spring force unmeasured.
"""
import math

SHAFT_R = 0.0150                      # sourced (30.0 mm OD)
BORE_R = SHAFT_R + 0.0003             # ferrule bore = the clip lip's inner radius (0.3 mm running clearance)
TAB_Y = -0.035                        # tabs 35 mm outside the ferrule mouth
HOOK_Y, HOOK_L = 0.028, 0.004         # hook centre / axial length inside the ferrule
FER_L, INS = 0.060, 0.045             # ferrule length; inserted length of the left shaft when seated
COLLAR_L = 0.010                      # left collar carrying the two keys, just outside the cog ring
BOUNDS = {
    "ferrule_od": (0.036, 0.039),
    "engage":     (0.006, 0.010),     # key depth in the cog slots = pull-out before rotation
    "tab_d":      (0.008, 0.012),
    "protrude":   (0.0025, 0.0040),   # tab head above the 30 mm shaft at rest
    "hook":       (0.0010, 0.0015),   # hook reach beyond the lip (catch overlap)
    "key_w":      (0.0016, 0.0020),
}
INDEX = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
PAD_GAP_OPEN = 2 * 0.04695            # finger joint travel (yam_arena.constants); verify on the pad colliders
PAD_W = 0.020                         # assumed pad face width; verify on the pad colliders


def dims(which):
    k = 0 if which == "lo" else 1
    return {n: v[k] for n, v in BOUNDS.items()}


def ring_r(d):
    """cog ring centre radius; its 3 mm radial band clears the left shaft by 0.5 mm"""
    return max(d["ferrule_od"] / 2 - 0.0010, SHAFT_R + 0.0005 + 0.0015)


def groove_r(d):
    return BORE_R + d["hook"] + 0.0005


def gates(d):
    out = []
    span = 2 * (SHAFT_R + d["protrude"])
    r_o = d["ferrule_od"] / 2
    out.append(("G1 one tab pressed: the other hook still overlaps the lip (mm)", (SHAFT_R + d["hook"] - BORE_R) * 1e3,
                SHAFT_R + d["hook"] - BORE_R >= 0.0006))
    out.append(("G2 both pressed to flush: hook clears the lip by (mm)", (d["protrude"] - d["hook"]) * 1e3,
                d["protrude"] - d["hook"] >= 0.0008))
    out.append(("G3 jaw open gap - tab span (mm)", (PAD_GAP_OPEN - span) * 1e3, PAD_GAP_OPEN - span > 0.010))
    out.append(("G3b pad width covers the tab head (mm)", (PAD_W - d["tab_d"]) * 1e3, PAD_W >= d["tab_d"]))
    out.append(("G4 ferrule wall outside the hook groove (mm)", (r_o - groove_r(d)) * 1e3, r_o - groove_r(d) >= 0.0010))
    pitch = ring_r(d) * math.radians(15)
    worst = BOUNDS["key_w"][1] + 0.0006
    out.append(("G5 slot wall between 15 deg slots, worst key (mm)", (pitch - worst) * 1e3, pitch - worst >= 0.0015))
    out.append(("G6 pull-out before any turn (mm, = engage)", d["engage"] * 1e3, d["engage"] >= 0.005))
    out.append(("G7 hook still inside the ferrule after the pull (mm)", (HOOK_Y - HOOK_L / 2 - d["engage"] - 0.002) * 1e3,
                HOOK_Y - HOOK_L / 2 - d["engage"] - 0.002 > 0.0))
    return out, span


def _wall_boxes(d, color):
    """right cog ring on the ferrule mouth (y in [-engage, 0]): one box per wall arc between key slots. Slots sit at
    -k for every index k (and +180), so turning the right half by +k brings a slot under the left key at 0 / 180."""
    r = ring_r(d)
    slot = (BOUNDS["key_w"][1] + 0.0006) / r                     # slot angular width (rad), sized for the widest key
    centres = sorted({(-k + s) % 360 for k in INDEX for s in (0, 180)})
    edges = [(math.radians(c) - slot / 2, math.radians(c) + slot / 2) for c in centres]
    out = []
    for (a0, a1), (b0, b1) in zip(edges, edges[1:] + [(edges[0][0] + 2 * math.pi, 0)]):
        lo, hi = a1, b0
        n = max(1, int(math.ceil((hi - lo) / math.radians(8))))      # long arcs split into <= 8 deg chords
        for i in range(n):
            t0, t1 = lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n
            tm = 0.5 * (t0 + t1)
            chord = 2 * (r - 0.0015) * math.sin((t1 - t0) / 2)       # inner-edge chord: never pokes into a slot
            out.append(((r * math.sin(tm), -d["engage"] / 2, r * math.cos(tm)), (chord, d["engage"], 0.0030), color, (0.0, tm)))
    return out


def shapes(d, feather_deg=0.0):
    r_o = d["ferrule_od"] / 2
    grey, black, red = (0.25, 0.25, 0.27), (0.08, 0.08, 0.09), (0.75, 0.12, 0.10)
    e = d["engage"]
    # LEFT: shaft (inserted end at y = +INS), collar + 2 keys, 2 tab/arm/hook slides
    L = dict(cylinders=[((0, 0.5 * (INS - 1.0), 0), SHAFT_R, 1.0 + INS, black, "Y"),
                        ((0, -e - COLLAR_L / 2, 0), r_o, COLLAR_L, grey, "Y")], boxes=[], joints=[])
    r = ring_r(d)
    for s in (0.0, math.pi):
        L["boxes"].append(((r * math.sin(s), -e / 2, r * math.cos(s)), (d["key_w"], e, 0.0030), grey, (0.0, s)))
    for nm, sx in (("tab_px", 1), ("tab_nx", -1)):
        L["joints"].append(dict(
            name=nm, kind="slide", axis="X", anchor=(sx * SHAFT_R, TAB_Y, 0),
            lower=-d["protrude"] if sx > 0 else 0.0, upper=0.0 if sx > 0 else d["protrude"], stiffness=400.0, target=0.0,
            cylinders=[((sx * (SHAFT_R + d["protrude"] / 2 - 0.001), TAB_Y, 0), d["tab_d"] / 2, d["protrude"] + 0.002, red, "X")],
            boxes=[((sx * (SHAFT_R - 0.002), 0.5 * (TAB_Y + HOOK_Y), 0), (0.002, HOOK_Y - TAB_Y, 0.004), red),      # arm (inside)
                   ((sx * (SHAFT_R + d["hook"] / 2 - 0.0005), HOOK_Y, 0), (d["hook"] + 0.001, HOOK_L, 0.006), red)]))  # hook
    # RIGHT: ferrule tube (bore BORE_R) with the hook groove, cog ring on its mouth, then its own shaft
    tf = math.radians(feather_deg)
    n_seg = 24
    g0, g1 = HOOK_Y - HOOK_L / 2 - 0.0005, HOOK_Y + HOOK_L / 2 + 0.003
    R = dict(cylinders=[((0, FER_L + 0.5, 0), SHAFT_R, 1.0, black, "Y"),
                        ((0, 0.5 * (INS + 0.001 + FER_L), 0), BORE_R - 0.0001, FER_L - INS - 0.001, grey, "Y")],   # bushing end
             boxes=_wall_boxes(d, grey))
    for y0, y1, r_in in ((0.0, g0, BORE_R), (g0, g1, groove_r(d)), (g1, INS + 0.001, BORE_R)):
        seg = 2 * r_o * math.tan(math.pi / n_seg) + 0.0004
        for k in range(n_seg):
            t = 2 * math.pi * k / n_seg
            rc = 0.5 * (r_o + r_in)
            R["boxes"].append(((rc * math.sin(t), 0.5 * (y0 + y1), rc * math.cos(t)), (seg, y1 - y0, r_o - r_in), grey, (0.0, t)))
    # turn the right half's shapes by the feather angle about +y (authoring check only; in sim the body turns)
    R["boxes"] = [((c[0] * math.cos(tf) + c[2] * math.sin(tf), c[1], -c[0] * math.sin(tf) + c[2] * math.cos(tf)), s, k, (0.0, t[1] + tf))
                  for c, s, k, t in R["boxes"]]
    return dict(left=L, right=R)


if __name__ == "__main__":
    ok = True
    for which in ("lo", "hi"):
        d = dims(which)
        rows, span = gates(d)
        sh = shapes(d, 30.0)
        print("== %s bound: ferrule OD %.1f, tab span %.1f mm, ring r %.1f, groove r %.2f mm, %d right boxes" % (
            which, d["ferrule_od"] * 1e3, span * 1e3, ring_r(d) * 1e3, groove_r(d) * 1e3, len(sh["right"]["boxes"])))
        for name, val, good in rows:
            ok &= good
            print("  %s %-62s %6.2f" % ("PASS" if good else "FAIL", name, val))
    print("ALL PASS" if ok else "FAILURES")
