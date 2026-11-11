

# -- debug (appended on Brev only) --
_dfb = {"n": 0}


@on_step
def _dbg_f3(env):
    _dfb["n"] += 1
    if _dfb["n"] % 10 and _dfb["n"] > 40:
        return
    S = labkit._SPECS
    b, r, c = (labkit._part_state(env, S[n]) for n in ("boat", "hook_rod", "end_plate"))
    t, f, g = (labkit._part_state(env, S[n]) for n in ("quartz_tube", "tube_furnace", "gas_lines"))
    chips = [labkit._part_state(env, S[f"chip{i}"]) for i in range(3)]
    co = [round(d * 1000, 1) for d, _ in chip_offsets(b, chips)]
    mm = lambda v: np.round((np.asarray(v, float) - [0, 0, TOP]) * 1000, 1).tolist()
    tips = [np.round((np.asarray(x) - [0, 0, TOP]) * 1000).astype(int).tolist() for x in _tips(env)]
    t0 = _state.get("tube0")
    dt = round(float(np.linalg.norm(_tube_ref(t)[0] - t0[0])) * 1000, 1) if t0 else None
    dg = lambda part, j: round(math.degrees(part.joints.get(j, 0)), 1)
    open("/home/ubuntu/dbg_fd.txt", "a").write(
        f"{_dfb['n']} t={getattr(_demo, '_plan_clock', 0.0):.1f} P={_gas.get('p', 0):.2f} argon={dg(g, 'argon_valve')} "
        f"bleed={dg(g, 'bleed_valve')} needle={dg(t, 'needle')} clamp={dg(t, 'clamp')} lid={dg(f, 'lid')} "
        f"capout={plate_out(c) * 1000:.1f} ptilt={axis_off(c):.1f} roll={math.degrees(math.asin(float(t.rotation()[0, 2]))):.1f} cap={mm(c.pos)} tube_d={dt} boat={mm(b.pos)} chips={co} rod={mm(r.pos)} "
        f"viol={sorted(_state.get('viol', []))} cent={_state.get('centred')} rack={_state.get('racked')} "
        f"lidok={_state.get('lid')} seal={_state.get('sealed')} far={_state.get('far')} settled={_state.get('settled')} "
        f"tips={tips}\n")


@on_reset
def _dbg_f3_reset(env):
    S = labkit._SPECS
    t = labkit._part_state(env, S["quartz_tube"])
    open("/home/ubuntu/dbg_fd.txt", "a").write(f"RESET tpos={np.round((np.asarray(t.pos) - [0, 0, TOP]) * 1000, 1).tolist()} tR={np.round(t.rotation()[:, 1:3].T, 3).tolist()} "
                                              f"tips={[np.round((np.asarray(x) - [0, 0, TOP]) * 1000).astype(int).tolist() for x in _tips(env)]}\n")
