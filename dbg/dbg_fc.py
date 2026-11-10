

# -- debug (appended on Brev only) --
_dfb = {"n": 0}


@on_step
def _dbg_fb(env):
    _dfb["n"] += 1
    if _dfb["n"] % 10 and _dfb["n"] > 40:
        return
    S = labkit._SPECS
    b, r, c = (labkit._part_state(env, S[n]) for n in ("boat", "hook_rod", "end_cap"))
    t, f = labkit._part_state(env, S["quartz_tube"]), labkit._part_state(env, S["tube_furnace"])
    chips = [labkit._part_state(env, S[f"chip{i}"]) for i in range(3)]
    co = [round(d * 1000, 1) for d, _ in chip_offsets(b, chips)]
    mm = lambda v: np.round((np.asarray(v, float) - [0, 0, TOP]) * 1000, 1).tolist()
    tips = [np.round((np.asarray(x) - [0, 0, TOP]) * 1000).astype(int).tolist() for x in _tips(env)]
    t0 = _state.get("tube0")
    dt = round(float(np.linalg.norm(np.asarray(t.pos) - t0[0])) * 1000, 1) if t0 else None
    open("/home/ubuntu/dbg_fc.txt", "a").write(
        f"{_dfb['n']} t={getattr(_demo, '_plan_clock', 0.0):.1f} P={_gas.get('p', 0):.2f} valve={math.degrees(t.joints.get('vent_valve', 0)):.0f} "
        f"needle={math.degrees(t.joints.get('needle', 0)):.0f} lid={math.degrees(f.joints.get('lid', 0)):.1f} capout={cap_out(c) * 1000:.1f} "
        f"cap={mm(c.pos)} tube_d={dt} boat={mm(b.pos)} chips={co} rod={mm(r.pos)} viol={sorted(_state.get('viol', []))} "
        f"cent={_state.get('centred')} rack={_state.get('racked')} lidok={_state.get('lid')} seal={_state.get('sealed')} "
        f"far={_state.get('far')} settled={_state.get('settled')} tips={tips}\n")


import os as _os
if _os.path.exists("/home/ubuntu/fa_noforce"):
    def _push_cap(env, fy):  # noqa: F811
        return None
if _os.path.exists("/home/ubuntu/fa_noneedle"):
    def _show_pressure(env, p):  # noqa: F811
        return None

_orig_push = _push_cap


def _push_cap(env, fy):  # noqa: F811
    n = _dfb["n"]
    if n < 80:
        c = labkit._part_state(env, labkit._SPECS["end_cap"])
        open("/home/ubuntu/dbg_force.txt", "a").write(f"{n} fy={fy:.3f} d={cap_out(c)*1000:.2f} cap={np.round(np.asarray(c.pos)*1000,1).tolist()} v={np.round(np.asarray(c.lin_vel),3).tolist()}\n")
    if not _os.path.exists("/home/ubuntu/fa_noforce"):
        _orig_push(env, fy)
