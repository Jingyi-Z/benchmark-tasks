"""Ready a small fire extinguisher: pull the ring pin without squeezing the lever (claude-b50, scene v0).
A stock 2.5 lb ABC home extinguisher (Kidde FA110 class: 3.25 in diameter, 13.75 in tall, 3.9 lb) stands on the bench.
Its operating lever sits just above the ring pin that passes through the valve; squeezing the lever presses it onto the
pin, which then binds (and in a real trial would discharge). One hand steadies the cylinder, the other pulls the pin out
sideways by its ring and lays it on the bench. Scored: pin fully out and resting on the bench, the lever never moved,
the extinguisher upright where it stood, hands clear, all still."""
import json
import math
import os
import pathlib
import sys

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: F401,E402
import labkit  # noqa: E402
from labkit import Joint, Mesh, on_reset, on_step, part  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.constants import FINGER_OPEN, FINGER_CLOSED  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
labkit.CONTACT_OFFSET = 0.0004
RED = (0.72, 0.06, 0.05)
BLACK = (0.07, 0.07, 0.08)
CHROME = (0.80, 0.81, 0.83)          # chrome valve (HOME1 photos; also the colliders' drawn colour)
STEEL = (0.62, 0.63, 0.66)
YEL = (0.85, 0.72, 0.10)

EX_X, EX_Y = float(os.environ.get("EX_X", "0.42")), float(os.environ.get("EX_Y", "-0.02"))
R_BODY = 0.0413                     # 3.25 in diameter (maker)
H_BODY = 0.283                      # First Alert HOME1: 14.3 in (363 mm) overall; cylinder to the shoulder (estimate)
M_TOTAL = float(os.environ.get("EX_MASS", "2.45"))   # HOME1 unit weight 5.4 lb as listed by retailers (not a maker sheet)
Z_PIN = 0.339                       # pin axis above the bench (estimate, scaled from the valve layout)
PIN_R, PIN_L = 0.0016, 0.050        # 3.2 mm steel pin (estimate), 50 mm long
CHEEK_Y = 0.016                     # the valve's two cheeks, either side of the lever, carry the pin holes
LEVER_GAP = 0.0006                  # lever underside above the pin top when at rest


def _hull(pts, color, collide="hull"):
    p = np.asarray(pts, float)
    return Mesh(p, ConvexHull(p).simplices.astype(np.int64), color, collide)


def _cheek(y):
    """a 4 mm cheek plate at y with a round-ish 3.8 mm hole for the pin at (0, Z_PIN): four blocks around the hole"""
    hr = PIN_R + 0.0003
    b = []
    for (cx, cz, sx, sz) in ((0.0, Z_PIN + hr + 0.006, 0.024, 0.012), (0.0, Z_PIN - hr - 0.008, 0.024, 0.016),
                             (hr + 0.0055, Z_PIN, 0.011, 2 * hr), (-hr - 0.0055, Z_PIN, 0.011, 2 * hr)):
        b.append(((cx, y, cz), (sx, 0.004, sz), CHROME))
    return b


# ---- visual look (drawn only; the colliders above do the physics) ------------------------------------------------------
PAINT = (0.70, 0.03, 0.03)
CREAM = (0.93, 0.91, 0.84)
ALU = (0.80, 0.81, 0.83)
GREEN = (0.10, 0.62, 0.20)
labkit.SURFACES[tuple(round(c, 4) for c in PAINT)] = dict(roughness=0.25, metallic=0.0)
labkit.SURFACES[tuple(round(c, 4) for c in ALU)] = dict(roughness=0.18, metallic=0.95)


def _lathe(profile, color, n=72, a0=0.0, a1=2 * math.pi, c=(0.0, 0.0)):
    """revolve an (r, z) profile about the vertical axis at c: a smooth body of n segments (drawn only)"""
    full = abs(a1 - a0 - 2 * math.pi) < 1e-9
    m = n if full else n + 1
    pts = []
    for k in range(m):
        a = a0 + (a1 - a0) * k / n
        pts += [(c[0] + r * math.cos(a), c[1] + r * math.sin(a), z) for r, z in profile]
    P = len(profile)
    tris = []
    for k in range(n):
        k2 = (k + 1) % m
        for i in range(P - 1):
            a, b, cc, d = k * P + i, k2 * P + i, k2 * P + i + 1, k * P + i + 1
            tris += [(a, b, cc), (a, cc, d)]
    return Mesh(np.asarray(pts, float), np.asarray(tris, np.int64), color, "none")


def _rbox(c, s, color, bev=0.0015):
    """a box with bevelled edges (convex hull of its corners pulled in on each face), drawn only"""
    hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts += [(c[0] + sx * (hx - bev), c[1] + sy * (hy - bev), c[2] + sz * hz),
                        (c[0] + sx * (hx - bev), c[1] + sy * hy, c[2] + sz * (hz - bev)),
                        (c[0] + sx * hx, c[1] + sy * (hy - bev), c[2] + sz * (hz - bev))]
    p = np.asarray(pts, float)
    return Mesh(p, ConvexHull(p).simplices.astype(np.int64), color, "none")


STEEL_BLK = (0.10, 0.10, 0.11)          # painted pressed-steel lever and carry handle
RING_Y = (0.95, 0.80, 0.10)
for _c in (PAINT, STEEL_BLK, RING_Y, ALU):
    labkit.SMOOTH_COLORS.add(tuple(round(c, 4) for c in _c))


def _sweep(x0, x1, y, z0, w, h, color, bend=0.004, n=24, m=16):
    """a pressed-steel strip: rounded (superelliptic) cross-section w x h swept along x from x0 to x1, curving up by
    'bend' at the far end; smooth shaded"""
    pts, tris = [], []
    for i in range(n + 1):
        u = i / n
        x = x0 + (x1 - x0) * u
        zc = z0 + bend * u * u
        for j in range(m):
            a = 2 * math.pi * j / m
            ca, sa = math.cos(a), math.sin(a)
            pts.append((x, y + w / 2 * math.copysign(abs(ca) ** 0.4, ca), zc + h / 2 * math.copysign(abs(sa) ** 0.4, sa)))
    for i in range(n):
        for j in range(m):
            a, b, c, d = i * m + j, (i + 1) * m + j, (i + 1) * m + (j + 1) % m, i * m + (j + 1) % m
            tris += [(a, b, c), (a, c, d)]
    for end, i in ((0, 0), (1, n)):                      # end caps
        ctr = len(pts)
        pts.append((x0 if end == 0 else x1, y, z0 + (0 if end == 0 else bend)))
        for j in range(m):
            tris.append((ctr, i * m + j, i * m + (j + 1) % m) if end else (ctr, i * m + (j + 1) % m, i * m + j))
    return Mesh(np.asarray(pts, float), np.asarray(tris, np.int64), color, "none")


def _torus(c, R, r, color, n=40, m=12):
    """a round wire ring in the x-z plane centred at c (smooth)"""
    pts, tris = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        for j in range(m):
            b = 2 * math.pi * j / m
            rr = R + r * math.cos(b)
            pts.append((c[0] + rr * math.cos(a), c[1] + r * math.sin(b), c[2] + rr * math.sin(a)))
    for i in range(n):
        for j in range(m):
            p0, p1 = i * m + j, ((i + 1) % n) * m + j
            p2, p3 = ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m
            tris += [(p0, p1, p2), (p0, p2, p3)]
    return Mesh(np.asarray(pts, float), np.asarray(tris, np.int64), color, "none")


Z_TAPER = 0.215                       # HOME1 photos: straight wall to about 60 % of the height, then a long taper
_taper = [(0.016 + (0.0418 - 0.016) * (0.5 + 0.5 * math.cos(math.pi * u)), Z_TAPER + (H_BODY - Z_TAPER) * u)
          for u in np.linspace(0, 1, 20)]
BODY_PROFILE = ([(0.0, 0.0005), (0.0392, 0.0005), (0.0404, 0.0018), (0.0412, 0.0040), (0.0418, 0.0070)]
                + _taper + [(0.0160, H_BODY + 0.012), (0.0, H_BODY + 0.012)])
LOOK = [_lathe([(r + 0.0004, z) for r, z in BODY_PROFILE], PAINT, n=128),
        # wraparound label (lower half, as on the HOME1): white band, red instruction panel, blue A/B/C pictograms
        _lathe([(0.0424, 0.036), (0.0424, 0.182)], CREAM, n=96, a0=math.pi * 0.30, a1=math.pi * 1.70),
        _lathe([(0.0427, 0.090), (0.0427, 0.168)], (0.80, 0.12, 0.10), n=48, a0=math.pi * 0.62, a1=math.pi * 1.38),
        _lathe([(0.0427, 0.050), (0.0427, 0.080)], (0.10, 0.55, 0.85), n=48, a0=math.pi * 0.62, a1=math.pi * 1.38),
        _rbox((-0.0432, -0.018, 0.065), (0.0012, 0.014, 0.022), CREAM, 0.0004),   # pictogram tiles
        _rbox((-0.0432, 0.000, 0.065), (0.0012, 0.014, 0.022), CREAM, 0.0004),
        _rbox((-0.0432, 0.018, 0.065), (0.0012, 0.014, 0.022), CREAM, 0.0004),
        _rbox((-0.0432, 0.0, 0.155), (0.0012, 0.040, 0.004), CREAM, 0.0003),     # "INSTRUCTIONS" bar
        _rbox((-0.0432, 0.0, 0.135), (0.0012, 0.050, 0.002), CREAM, 0.0003),     # step lines
        _rbox((-0.0432, 0.0, 0.118), (0.0012, 0.050, 0.002), CREAM, 0.0003),
        _rbox((-0.0432, 0.0, 0.101), (0.0012, 0.050, 0.002), CREAM, 0.0003),
        # valve head: aluminium body with bevels, the fixed carry handle, the nozzle horn and the gauge
        _rbox((0.0, 0.0, Z_PIN - 0.024), (0.030, 0.030, 0.034), ALU, 0.006),
        _sweep(-0.012, 0.092, 0.0, Z_PIN - 0.030, 0.025, 0.0075, STEEL_BLK, bend=-0.003),
        _sweep(0.086, 0.100, 0.0, Z_PIN - 0.030, 0.025, 0.0075, STEEL_BLK, bend=-0.030),   # handle tail curls down
        ]


def _place(mesh, R, t):
    return Mesh(mesh.points @ np.asarray(R, float).T + np.asarray(t, float), mesh.tris, mesh.color, mesh.collide)


def _rx(a):
    c_, s_ = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]])


def _ry(a):
    c_, s_ = math.cos(a), math.sin(a)
    return np.array([[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]])


# chrome nozzle tube: straight out on the side opposite the lever and handle (HOME1 photos)
LOOK += [_place(_lathe([(0.0, 0.0), (0.0058, 0.0), (0.0058, 0.044), (0.0035, 0.044), (0.0035, 0.040), (0.0, 0.040)],
                       ALU, n=48), _ry(-math.pi / 2), (-0.013, 0.0, Z_PIN - 0.020))]
# large round gauge on the valve's side face: chrome bezel, red face, green zone at the top, needle in the green
G0 = np.array([0.0, 0.026, Z_PIN - 0.010])
LOOK += [_place(_lathe([(0.0, -0.004), (0.0150, -0.004), (0.0158, -0.002), (0.0150, 0.0), (0.0, 0.0)], ALU, n=64),
                _rx(-math.pi / 2), G0),
         _place(_lathe([(0.0, 0.0), (0.0132, 0.0)], (0.78, 0.10, 0.09), n=64), _rx(-math.pi / 2), G0 + [0, 0.0002, 0]),
         _place(_lathe([(0.0, 0.0), (0.0118, 0.0)], GREEN, n=12, a0=math.radians(70), a1=math.radians(110)),
                _rx(-math.pi / 2), G0 + [0, 0.0004, 0]),
         _rbox(tuple(G0 + [0.0, 0.0006, 0.005]), (0.0012, 0.0004, 0.010), BLACK, 0.0001)]


# ---- the extinguisher (free articulated part: shell + valve + carrying handle; the lever is a sprung hinge) -------
def _tang(y):
    """a 3 mm lever side plate at y with its own hole round the pin (0.6 mm clearance all round at rest): squeezing the
    lever brings the top of this hole down onto the pin, which then bears on the bottoms of the cheek holes and binds"""
    hr = PIN_R + LEVER_GAP
    return [((0.0, y, Z_PIN + hr + 0.002), (2 * hr + 0.008, 0.003, 0.004), BLACK),
            ((0.0, y, Z_PIN - hr - 0.002), (2 * hr + 0.008, 0.003, 0.004), BLACK),
            ((hr + 0.002, y, Z_PIN), (0.004, 0.003, 2 * hr), BLACK),
            ((-hr - 0.002, y, Z_PIN), (0.004, 0.003, 2 * hr), BLACK)]


SQUEEZE = float(os.environ.get("EX_SQUEEZE", "0"))   # diagnostic only: a lever target in rad stands in for a squeezing hand
LEVER = Joint("lever", kind="hinge", axis="Y", anchor=(-0.012, 0.0, Z_PIN + PIN_R + LEVER_GAP + 0.003), lower=-0.02,
              upper=0.40, initial=0.0, target=0.0, stiffness=float(os.environ.get("EX_LEVER_K", "20.0")),
              damping=0.05, mass=0.06,
              boxes=[((0.046, 0.0, Z_PIN + PIN_R + LEVER_GAP + 0.006), (0.092, 0.022, 0.006), BLACK),
                     ((0.090, 0.0, Z_PIN + PIN_R + LEVER_GAP + 0.013), (0.004, 0.022, 0.012), BLACK)]
              + _tang(0.0095) + _tang(-0.0095),
              meshes=[_sweep(-0.004, 0.094, 0.0, Z_PIN + PIN_R + LEVER_GAP + 0.006, 0.026, 0.0080, STEEL_BLK, bend=0.006)]
              if os.environ.get("EX_LOOK", "1") == "1" else [])
EXT = part("extinguisher", pos=(EX_X, EX_Y, TOP), mass=M_TOTAL - 0.06, friction=0.6, joints=[LEVER],
           cylinders=[((0, 0, Z_TAPER / 2), R_BODY, Z_TAPER, RED, "Z"),                     # shell (to the taper)
                      ((0, 0, (Z_TAPER + H_BODY) / 2), 0.0155, H_BODY - Z_TAPER, RED, "Z"),   # shoulder core
                      ((0, 0, H_BODY + 0.010), 0.020, 0.020, CHROME, "Z"),                  # neck (chrome valve collar)
                      ((-0.012, 0.0, Z_PIN + 0.012), 0.004, 0.030, CHROME, "Y")],          # lever pivot pin
           boxes=[((0.0, 0.0, Z_PIN - 0.024), (0.026, 0.026, 0.030), CHROME),             # valve body
                  ((0.040, 0.0, Z_PIN - 0.030), (0.100, 0.022, 0.006), BLACK),            # carrying handle (fixed)
                  ((-0.035, 0.0, Z_PIN - 0.020), (0.040, 0.010, 0.010), CHROME),          # nozzle (opposite the handle)
                  ((0.0, 0.0, 0.002), (0.056, 0.056, 0.004), BLACK)]                     # base footprint (inside the foot ring)
           + _cheek(CHEEK_Y) + _cheek(-CHEEK_Y),
           meshes=(LOOK + [_hull([(x, y, z) for x in (-0.0012, 0.0012) for (y, z) in
                                   ((-0.010, Z_PIN - 0.027), (-0.008, Z_PIN - 0.026), (-0.024, Z_PIN - 0.012),
                                    (-0.022, Z_PIN - 0.011))], YEL, "none")])     # tamper tie, handle half
           if os.environ.get("EX_LOOK", "1") == "1" else [])
# ---- the ring pin (loose rigid part through both cheek holes, ring on the -y side) -------------------------------------
RING_R = 0.012
_ring = []
for k in range(10):
    a0, a1 = 2 * math.pi * k / 10, 2 * math.pi * (k + 1) / 10
    pts = [((RING_R + dr) * math.cos(a), dy, (RING_R + dr) * math.sin(a))
           for a in (a0, a1) for dr in (-0.0012, 0.0012) for dy in (-0.0012, 0.0012)]
    _ring.append(_hull([(x, y - PIN_L / 2 - 0.0016, z - RING_R - 0.0016) for x, y, z in pts], YEL))
# the ring passes through the eye of the pin head and swings freely about it (axis x), hanging down at rest; lifting it
# up gives room and a lever arm for the twist
RING = Joint("ring", kind="hinge", axis="X", anchor=(0.0, -PIN_L / 2 - 0.0016, 0.0), lower=-3.3, upper=0.15,
             initial=0.0, target=0.0, stiffness=0.0, damping=0.0005, mass=0.004,
             meshes=_ring + ([_torus((0.0, -PIN_L / 2 - 0.0016, -RING_R - 0.0016), RING_R, 0.0018, RING_Y)]
                             if os.environ.get("EX_LOOK", "1") == "1" else []))
PIN = part("pin", pos=(EX_X, EX_Y - 0.004, TOP + Z_PIN), mass=0.006, friction=0.5, joints=[RING],
           cylinders=[((0, 0, 0), PIN_R, PIN_L, STEEL, "Y"),
                      # shoulder (the pin's bent head on a real unit): seats against the outer valve cheek, so the pin
                      # cannot be pushed further in; it can only leave toward the ring side
                      ((0.0, -CHEEK_Y - 0.002 - 0.0011 + 0.004, 0.0), 0.0030, 0.0020, STEEL, "Y"),
                      ((0.0, -PIN_L / 2 - 0.0016, 0.0), 0.0035, 0.0032, STEEL, "Y")],     # flattened eye head
           meshes=[_hull([(x, y, z) for x in (-0.0012, 0.0012) for (y, z) in
                          ((-PIN_L / 2 + 0.0005, -0.0025), (-PIN_L / 2 + 0.0025, -0.0035),
                           (-0.020 + 0.004, -0.012), (-0.018 + 0.004, -0.013))], YEL, "none")])   # tamper tie, ring half
OBJECTS = [EXT, PIN]
# tamper seal: a plastic tie from the ring round the carry handle. UL 299 requires it to break at 65 N (15 lbf) or less;
# we use the upper limit. Twisting the ring loads the tie at the ring's radius, so it breaks at 65 N x 12 mm = 0.78 N m.
SEAL_F = float(os.environ.get("EX_SEAL_F", "65.0"))
SEAL_T = float(os.environ.get("EX_SEAL_T", str(65.0 * (2 * RING_R + 0.0016))))   # raised ring: 25.6 mm arm
SEAL_TURN, SEAL_SHIFT = 10.0, 0.003
# a real tie through the ring goes slack if the pin is pushed in, so the seal is a one-way slide along the pin axis: free
# inward (up to the shoulder stop), locked outward and in twist; it breaks only from an outward pull or a twist
labkit.TIES["pin"] = [("extinguisher", SEAL_F, SEAL_T, (0.0, -0.004, Z_PIN),
                       ("Y", -0.0001, 0.003) if os.environ.get("EX_TIE", "slide") == "slide" else None)]

LEVER_LATCH = math.radians(1.0)     # any lever travel past 1 deg: squeezed (would discharge)
HAND_CLEAR, STILL_N = 0.080, 21
MAX_SHIFT, MAX_LEAN = 0.010, 5.0   # largest base slide (m) and lean (deg) allowed at any time
_S: dict = {}


@on_reset
def _reset(env):
    _S.clear()
    _S.update(step=0, latch=None, start=None, hist=[], score=0.0, max_moved=0.0, max_tilt=0.0, seal=False, rel0=None,
              travel=0.0)


HEAD_LO = np.array([-0.025, -0.022, H_BODY])            # valve head, lever and carry handle envelope (extinguisher frame)
HEAD_HI = np.array([0.095, 0.022, Z_PIN + PIN_R + LEVER_GAP + 0.020])


def _clearance(a, p_ex, R, p_pin, R_pin):
    """distance (m) from a pad point to the nearest SURFACE of the cylinder, the valve-head envelope, the pin shaft or
    its ring (not to their centres)"""
    q = R.T @ (np.asarray(a, float) - p_ex)
    rad = math.hypot(q[0], q[1])
    dr, dz = max(0.0, rad - R_BODY), max(0.0, -q[2], q[2] - H_BODY)
    d_cyl = math.hypot(dr, dz)
    d_head = float(np.linalg.norm(np.maximum(0.0, np.maximum(HEAD_LO - q, q - HEAD_HI))))
    u = R_pin[:, 1]                                          # pin axis (pin frame y)
    v = np.asarray(a, float) - p_pin
    t = float(np.clip(v @ u, -PIN_L / 2, PIN_L / 2))
    d_pin = max(0.0, float(np.linalg.norm(v - t * u)) - PIN_R)
    q = _S.get("ring_q", 0.0)                         # the ring's hinge angle about the eye (raised ring is above)
    c_, s_ = math.cos(q), math.sin(q)
    off = np.array([0.0, 0.0 * c_ - (-(RING_R + 0.0016)) * s_, (-(RING_R + 0.0016)) * c_])
    ring_c = p_pin + R_pin @ (np.array([0.0, -PIN_L / 2 - 0.0016, 0.0]) + off)
    d_ring = max(0.0, float(np.linalg.norm(np.asarray(a, float) - ring_c)) - RING_R - 0.0012)
    return min(d_cyl, d_head, d_pin, d_ring)


def _pads(env):
    try:
        from yam_arena import runtime as _rt
        o = labkit._np(env.unwrapped.scene.env_origins)[0]
        return [np.asarray(_rt.pad_tips_world(env.unwrapped.scene[n]), float).reshape(-1, 3) - o
                for n in ("left_robot", "right_robot")]
    except Exception:  # noqa: BLE001
        return []


_BRK: dict = {}


def _watch_breaks():
    """subscribe to PhysX JOINT_BREAK events once; the seal tie's break is recorded with its control step"""
    if "sub" in _BRK:
        return
    try:
        from omni.physx import get_physx_interface
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools

        def cb(e):
            if e.type == int(SimulationEvent.JOINT_BREAK):
                p = str(PhysicsSchemaTools.decodeSdfPath(e.payload["jointPath"][0], e.payload["jointPath"][1]))
                if "tie_extinguisher" in p and "event_step" not in _BRK:
                    _BRK["event_step"] = _S.get("step", -1)
                    print(f"EXBREAK joint {p} broke at step {_BRK['event_step']}", flush=True)
        _BRK["sub"] = get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(cb)
    except Exception as e:  # noqa: BLE001
        _BRK["sub"] = None
        print("EXBREAK subscribe failed", repr(e), flush=True)


@on_step
def _track(env):
    _watch_breaks()
    _S["step"] += 1
    st = _S["step"]
    if os.environ.get("EX_PUSH"):   # hold proof (diagnostic only): the twist torque and a pull, applied to the unit
        try:
            import torch
            art = env.unwrapped.scene[EXT.name]
            on = int(os.environ.get("EX_PUSH0", "200")) <= st < int(os.environ.get("EX_PUSH1", "320"))
            f = torch.zeros((1, art.num_bodies, 3), device=art.device)
            t = torch.zeros((1, art.num_bodies, 3), device=art.device)
            if on:
                f[0, 0, 1] = -float(os.environ.get("EX_PUSH_F", "30"))
                t[0, 0, 1] = float(os.environ.get("EX_PUSH_T", "1.66"))
            art.set_external_force_and_torque(f, t)
            if st in (200, 201):
                print("EXPUSH on", on, flush=True)
        except Exception as e:  # noqa: BLE001
            if st == 1:
                print("EXPUSH failed", repr(e), flush=True)
    if SQUEEZE > 0.0:  # diagnostic squeeze: labkit only seeds the default pose, so drive the lever target every step
        try:
            import torch
            art = env.unwrapped.scene[EXT.name]
            ids = [i for i, n in enumerate(art.joint_names) if "lever" in n]
            tgt = art.data.joint_pos_target.clone()
            tgt[:, ids] = SQUEEZE
            art.set_joint_position_target(tgt[:, ids], joint_ids=ids)
            if st in (1, 50, 200):
                print("SQZ", art.joint_names, [round(float(v), 4) for v in art.data.joint_pos[0]], flush=True)
        except Exception as e:  # noqa: BLE001
            if st == 1:
                print("SQZ failed", repr(e), flush=True)
    ex = labkit._part_state(env, EXT)
    pn = labkit._part_state(env, PIN)
    lever = float(ex.joints.get("lever", 0.0))
    R = ex.rotation()
    tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
    p_ex = np.asarray(ex.pos, float)
    p_pin = np.asarray(pn.pos, float)
    if _S["start"] is None:
        _S["start"] = p_ex.copy()
    if lever > LEVER_LATCH and not _S["latch"]:
        _S["latch"] = "lever squeezed"
    if tilt > 20.0 and not _S["latch"]:
        _S["latch"] = "extinguisher tipped"
    if p_pin[2] < TOP - 0.02 and not _S["latch"]:
        _S["latch"] = "pin dropped off the bench"
    # pin out: its inner end has cleared the outer cheek (in the extinguisher frame)
    loc = R.T @ (p_pin - p_ex)
    _S["ring_q"] = float(pn.joints.get("ring", 0.0))
    Rp = pn.rotation()
    rel_R = R.T @ Rp
    if _S["rel0"] is None:
        _S["rel0"] = (loc.copy(), rel_R.copy())
    l0, r0 = _S["rel0"]
    dang = math.degrees(math.acos(float(np.clip((np.trace(r0.T @ rel_R) - 1) / 2, -1, 1))))
    _S["max_turn"] = max(_S.get("max_turn", 0.0), dang)
    dR = r0.T @ rel_R                    # shaft twist about its own axis (pin frame y), separate from tilt
    _S["axial_twist"] = math.degrees(math.atan2(dR[0, 2] - dR[2, 0], dR[0, 0] + dR[2, 2]))
    _S["axial_mm"] = float(loc[1] - l0[1]) * 1e3          # signed: + is inward (into the valve), - is outward
    _S["max_shift"] = max(_S.get("max_shift", 0.0), float(np.linalg.norm(loc - l0)))
    if not _S["seal"] and (np.linalg.norm(loc - l0) > SEAL_SHIFT or dang > SEAL_TURN):
        # an intact tie lets the pin give only slightly under load (never-break control: about 2 deg, 0.3 mm); past
        # 10 deg of turn or 3 mm of travel in the valve the tie has gone
        _S["pose_hint"] = st
        print(f"EXSEAL (pose cross-check only) at step {st}: shift {np.linalg.norm(loc - l0) * 1e3:.2f} mm, turn {dang:.1f} deg", flush=True)
    if "event_step" in _BRK:
        _S["seal"] = True
    _S["travel"] = max(_S["travel"], float(l0[1] - loc[1]))
    pin_out = bool(loc[1] + PIN_L / 2 < -CHEEK_Y - 0.004 or np.linalg.norm(loc[[0, 2]] - np.array([0.0, Z_PIN])) > 0.02)
    on_bench = bool(p_pin[2] < TOP + 0.02)
    pads = _pads(env)
    clear = min((_clearance(a, p_ex, R, p_pin, pn.rotation()) for P in pads for a in P), default=0.0)         if len(pads) == 2 else 0.0
    lhold = -1.0
    if len(pads) == 2:
        lo = np.array([-0.010, -0.011, Z_PIN - 0.033]); hi = np.array([0.090, 0.011, Z_PIN - 0.027])
        ds = [float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - q, q - hi))))
              for q in (R.T @ (np.asarray(a, float) - p_ex) for a in pads[0])]
        lhold = min(ds)
        _S["lhold_each"] = [round(d * 1e3, 1) for d in ds]
    _S["hist"].append((p_ex.copy(), p_pin.copy(), clear))
    _S["hist"] = _S["hist"][-STILL_N:]
    still = (len(_S["hist"]) == STILL_N and
             max(float(np.ptp(np.array([h[0] for h in _S["hist"]]), axis=0).max()),
                 float(np.ptp(np.array([h[1] for h in _S["hist"]]), axis=0).max())) < 0.002
             and min(h[2] for h in _S["hist"]) >= HAND_CLEAR)
    moved = float(np.linalg.norm((p_ex - _S["start"])[:2]))
    _S["max_moved"] = max(_S["max_moved"], moved)
    _S["max_tilt"] = max(_S["max_tilt"], tilt)
    # the steadying hand must actually steady: at no time may the cylinder slide more than MAX_SHIFT or lean more than
    # MAX_LEAN (a shove that is later undone still fails)
    ok_base = (_S["latch"] is None and _S["max_tilt"] < MAX_LEAN and _S["max_moved"] < MAX_SHIFT and still)
    if ok_base and _S["seal"] and pin_out and on_bench:
        _S["score"] = 1.0
    elif ok_base and _S["seal"] and float(l0[1] - loc[1]) > 0.005 and not pin_out:
        _S["score"] = 0.5        # seal broken and pin started out of the valve, no violation
    else:
        _S["score"] = 0.0
    if st % 10 == 0 or st <= 2:
        print("EX " + json.dumps(dict(step=st, lever_deg=round(math.degrees(lever), 2), tilt=round(tilt, 2),
                                      moved_mm=round(moved * 1e3, 1), max_moved_mm=round(_S["max_moved"] * 1e3, 1), max_tilt=round(_S["max_tilt"], 2), pin=[round(float(v), 4) for v in p_pin],
                                      pin_local_y=round(float(loc[1]), 4), pin_out=pin_out, on_bench=on_bench, seal_broken=_S["seal"], travel_mm=round(_S["travel"] * 1e3, 1),
                                      pin_turn=round(_S.get("max_turn", 0.0), 1), axial_twist=round(_S.get("axial_twist", 0.0), 1),
                                      axial_mm=round(_S.get("axial_mm", 0.0), 1), break_event=_BRK.get("event_step"),
                                      pull_dot_axis=round(float(-Rp[1, 1] * -1.0), 3), ring_deg=round(math.degrees(float(pn.joints.get("ring", 0.0))), 1),
                                      lhold_mm=round(lhold * 1e3, 1), lpads_mm=_S.get("lhold_each"),
                                      clear=round(clear, 3), still=still, latch=_S["latch"], score=_S["score"])), flush=True)


def score(objects) -> float:
    """1.0: the seal tie is broken (PhysX joint-break event), the pin is fully out of the valve and resting on the bench,
    the lever never moved more than 1 deg, at no time did the extinguisher slide more than 10 mm or lean more than 5 deg,
    both pad tips end at least 80 mm from the cylinder, valve head, pin and ring surfaces, and extinguisher and pin each
    move under 2 mm for 1 s. 0.5: the same, but the pin is currently withdrawn more than 5 mm and not yet out. 0
    otherwise; 0 for the rest of the trial once the lever moves past 1 deg, the extinguisher tips past 20 deg, or the
    pin falls off the bench."""
    return float(_S.get("score", 0.0))


def success(objects) -> bool:
    return score(objects) >= 1.0


# ---- scripted demo -------------------------------------------------------------------------------------------------
class Orient(float):
    def __new__(cls, rot):
        o = float.__new__(cls, 0.0)
        o.rot = np.asarray(rot, float)
        return o


_prev = _demo.grip_rotation
if not getattr(_prev, "_alt", False):
    def _gr(yaw=0.0, _p=_prev):
        return yaw.rot if hasattr(yaw, "rot") else _p(yaw)
    _gr._alt = True
    _demo.grip_rotation = _gr

def _Ry(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def _Rx(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def _Rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


PARK_L, PARK_R = (0.21, 0.28, 0.93), (0.21, -0.28, 0.93)


def demo():
    if os.environ.get("EX_LOOKPOSE"):   # look stills: each wrist camera about 20 cm from the unit, from two sides
        pt = float(os.environ.get("EX_LOOKP", "0.25"))
        oL = Orient(_Rz(-math.pi / 2) @ _Ry(-math.pi / 2 + pt) @ _prev(0.0))
        oR = Orient(_Rz(math.pi / 2) @ _Ry(-math.pi / 2 + pt) @ _prev(0.0))
        z = TOP + float(os.environ.get("EX_LOOKZ", "0.30"))
        dd = float(os.environ.get("EX_LOOKD", "0.22"))
        return dict(left=[Waypoint(4.0, (EX_X - 0.02, EX_Y + dd, z), FINGER_OPEN, "look", oL),
                          Waypoint(26.0, (EX_X - 0.02, EX_Y + dd, z), FINGER_OPEN, "look", oL)],
                    right=[Waypoint(4.0, (EX_X - 0.02, EX_Y - dd, z), FINGER_OPEN, "look", oR),
                           Waypoint(26.0, (EX_X - 0.02, EX_Y - dd, z), FINGER_OPEN, "look", oR)])
    if os.environ.get("EX_PARK"):
        return dict(left=[Waypoint(30.0, PARK_L, 0.0, "parked")], right=[Waypoint(30.0, PARK_R, 0.0, "parked")])
    # right: pinch the ring across its sides (jaws closing along x), pull straight out along -y, lay the pin down
    yaw = float(os.environ.get("EX_RYAW", str(math.pi / 2)))
    g = os.environ.get("EX_GRIP", "lift")
    if g == "side":     # fingers pointing +x (level), jaws closing along y across the ring's flat
        yaw = Orient(_Rz(math.radians(float(os.environ.get("EX_GYAW", "0")))) @ _Ry(-math.pi / 2) @ _prev(0.0))
    elif g == "pitch":  # pitched 45 deg down from vertical toward +x, jaws closing along y
        yaw = Orient(_Ry(-math.pi / 4) @ _prev(0.0))
    ry = EX_Y - 0.004 - PIN_L / 2
    rx = EX_X + float(os.environ.get("EX_RDX", "-0.006"))
    rz = TOP + Z_PIN - float(os.environ.get("EX_RZOFF", str(RING_R)))
    tp = float(os.environ.get("EX_PULL_T", "2.0"))
    FP = FINGER_CLOSED + float(os.environ.get("EX_FOPEN", "0.3")) * (FINGER_OPEN - FINGER_CLOSED)
    if g == "lift":
        # Jingyi: one finger THROUGH the ring, one outside, clamp the wire at the bottom; lift, rotate, pull.
        # Fingers point along the ring's axis (+y, toward the valve); jaws close vertically. The upper finger threads the
        # ring's 21.6 mm opening (fingertip 15.5 x 4-6 mm over its last 14 mm), the lower one stays outside below; the
        # pads clamp the bottom wire. Lifting swings the ring up about the pin eye (the wire rolls in the clamp); turning
        # the wrist about the pin axis turns the ring by the finger inside it, breaking the seal; then the pin is drawn
        # out along its axis, the inside finger hooking the ring.
        Rt = _Rz(math.pi / 2) @ _Ry(-math.pi / 2) @ _prev(math.pi / 2)     # fingers +y, jaws along z
        E = np.array([EX_X, EX_Y - 0.004 - PIN_L / 2 - 0.0016, TOP + Z_PIN])
        Lr = 2 * RING_R + 0.0016
        dx = float(os.environ.get("EX_LDX", "0.0"))                          # clamp point along the bottom wire
        dep = float(os.environ.get("EX_DEPTH", "0.004"))                     # fingertips past the ring's plane
        lrot = os.environ.get("EX_LROT", "0") == "1"                          # hand turns with the ring while lifting
        FP = FINGER_CLOSED + float(os.environ.get("EX_FOPEN3", "0.2")) * (FINGER_OPEN - FINGER_CLOSED)
        FSLOT = FINGER_CLOSED + float(os.environ.get("EX_FSLOT", "0.06")) * (FINGER_OPEN - FINGER_CLOSED)

        def ori(deg):
            return Orient(_Rx(-math.radians(deg)) @ Rt) if lrot else Orient(Rt)

        def arc(deg):
            a = math.radians(deg)
            return E + np.array([dx, -Lr * math.sin(a) + dep, -Lr * math.cos(a)])
        tw = float(os.environ.get("EX_TWIST", "30"))
        sgn = float(os.environ.get("EX_TSIGN", "1"))
        top = float(os.environ.get("EX_LIFT", "165"))

        def twist(deg):
            Rw = _Ry(math.radians(sgn * deg))
            b = arc(top)
            a_ = np.array([EX_X, b[1], TOP + Z_PIN])
            return tuple(a_ + Rw @ (b - a_)), Orient(Rw @ ori(top).rot)
        o = Orient(Rt)
        b0 = arc(0)
        R = [Waypoint(2.5, tuple(b0 + [0.0, -0.09, 0.0]), FP, "out along the ring's axis, level with its bottom", o),
             Waypoint(1.5, tuple(b0 + [0.0, -0.04, 0.0]), FP, "pre-closed jaws lined up: top finger on the opening", o),
             Waypoint(1.5, tuple(b0), FP, "top finger through the ring, bottom finger under it", o),
             Waypoint(0.8, tuple(b0), FSLOT, "close to a slot just wider than the wire", o)]
        # lift with a loose slot (wire rolls, ring swings about its eye); a full clamp would drag the pin out instead
        for deg in (30, 60, 90, 120, 150, top):
            R.append(Waypoint(0.8, tuple(arc(deg)), FSLOT, f"lift the ring up ({deg:.0f} deg)", ori(deg)))
        R.append(Waypoint(0.8, tuple(arc(top)), FINGER_CLOSED, "clamp the raised ring's wire", ori(top)))
        for k in range(1, 4):
            pk, ok_ = twist(tw * k / 3)
            R.append(Waypoint(0.8, pk, FINGER_CLOSED, f"turn the ring {tw * k / 3:.0f} deg to break the seal", ok_))
        pe, oe = twist(tw)
        R += [Waypoint(tp, (pe[0], pe[1] - 0.07, pe[2]), FINGER_CLOSED, "pull the pin straight out along its axis", oe),
              Waypoint(1.5, (pe[0], pe[1] - 0.12, pe[2] - 0.06), FINGER_CLOSED, "carry the pin out over the bench", oe),
              Waypoint(0.6, (pe[0], pe[1] - 0.12, pe[2] - 0.06), FP, "let it drop onto the bench", oe),
              Waypoint(1.0, (pe[0], pe[1] - 0.16, pe[2] - 0.02), FP, "back away", oe),
              Waypoint(2.0, PARK_R, FINGER_OPEN, "park"),
              Waypoint(3.0, PARK_R, FINGER_OPEN, "hold still")]
    elif g == "twist":
        # fingers pointing +y (toward the valve, along the pin), jaws closing along x on the ring's two sides. Twisting
        # the wrist turns the ring about the pin's axis (the ring hangs below it), which loads the tamper tie; then the
        # pin is drawn straight out along -y.
        F2 = FINGER_CLOSED + float(os.environ.get("EX_FOPEN2", "0.45")) * (FINGER_OPEN - FINGER_CLOSED)
        Rp0 = _Rz(math.pi / 2) @ _Ry(-math.pi / 2) @ _prev(0.0)
        ax = np.array([EX_X, EX_Y - 0.004, TOP + Z_PIN])              # pin axis point (world)
        rc = np.array([EX_X, ry + float(os.environ.get("EX_RDY", "0.0")), TOP + Z_PIN - RING_R])
        tw = float(os.environ.get("EX_TWIST", "90"))
        sgn = float(os.environ.get("EX_TSIGN", "1"))

        def at(deg, dy=0.0):
            Rw = _Ry(math.radians(sgn * deg))
            return tuple(ax + Rw @ (rc - ax) + np.array([0.0, dy, 0.0])), Orient(Rw @ Rp0)
        p0, o0 = at(0)
        R = [Waypoint(2.5, (p0[0], p0[1] - 0.09, p0[2]), F2, "out along the pin axis, beside the ring", o0),
             Waypoint(1.5, (p0[0], p0[1] - 0.05, p0[2]), F2, "level with the ring", o0),
             Waypoint(1.5, p0, F2, "jaws either side of the ring", o0),
             Waypoint(0.8, p0, FINGER_CLOSED, "pinch the ring by its sides", o0)]
        if os.environ.get("EX_MODE", "twist") == "twist":
            for k in range(1, 5):
                pk, ok_ = at(tw * k / 4)
                R.append(Waypoint(1.0, pk, FINGER_CLOSED, f"twist the ring {tw * k / 4:.0f} deg to break the seal", ok_))
        pe, oe = at(tw if os.environ.get("EX_MODE", "twist") == "twist" else 0)
        R += [Waypoint(tp, (pe[0], pe[1] - 0.07, pe[2]), FINGER_CLOSED, "pull the pin straight out along its axis", oe),
              Waypoint(1.5, (pe[0], pe[1] - 0.10, pe[2] - float(os.environ.get("EX_DROPDZ", "0.06"))), FINGER_CLOSED,
                       "carry the pin out over the bench", oe),
              Waypoint(0.6, (pe[0], pe[1] - 0.10, pe[2] - float(os.environ.get("EX_DROPDZ", "0.06"))), F2,
                       "let it drop onto the bench", oe),
              Waypoint(1.0, (pe[0], pe[1] - 0.12, pe[2] - float(os.environ.get("EX_DROPDZ", "0.06")) + 0.04), F2,
                       "back away", oe),
              Waypoint(2.0, PARK_R, FINGER_OPEN, "park"),
              Waypoint(3.0, PARK_R, FINGER_OPEN, "hold still")]
    else:
      R = [Waypoint(2.5, (rx - 0.08, ry, rz + 0.04), FP, "high, beside the ring", yaw),
           Waypoint(1.5, (rx - 0.05, ry, rz), FP, "level with the ring", yaw),
           Waypoint(1.2, (rx, ry, rz), FP, "jaws either side of the ring", yaw),
           Waypoint(0.8, (rx, ry, rz), FINGER_CLOSED, "pinch the ring", yaw),
           Waypoint(tp, (rx, ry - 0.07, rz), FINGER_CLOSED, "pull the pin straight out sideways", yaw),
           Waypoint(1.5, (rx, ry - 0.14, TOP + float(os.environ.get("EX_DROPZ", "0.12"))), FINGER_CLOSED, "carry the pin out over the bench", yaw),
           Waypoint(0.6, (rx, ry - 0.14, TOP + float(os.environ.get("EX_DROPZ", "0.12"))), FP, "let it drop onto the bench", yaw),
           Waypoint(1.0, (rx - 0.06, ry - 0.14, TOP + float(os.environ.get("EX_DROPZ", "0.12")) + 0.05), FP, "back away", yaw),
           Waypoint(2.0, PARK_R, FINGER_OPEN, "park"),
           Waypoint(3.0, PARK_R, FINGER_OPEN, "hold still")]
    if os.environ.get("EX_LEFT", "1") == "1":
        # left: hold the fixed carry handle (below the lever), fingers pointing -y, jaws closing along z, only partly
        # open so the upper finger stays well below the lever
        lg = Orient(_Rz(-math.pi / 2) @ _Ry(-math.pi / 2) @ _prev(math.pi / 2))
        hx = EX_X + float(os.environ.get("EX_HX", "0.06"))
        hz = TOP + Z_PIN - 0.030
        hy = EX_Y + float(os.environ.get("EX_HY", "0.0"))
        L = [Waypoint(2.5, (hx, hy + 0.10, hz + 0.03), FP, "beside the carry handle", lg),
             Waypoint(1.5, (hx, hy + 0.06, hz), FP, "level with the carry handle", lg),
             Waypoint(1.5, (hx, hy, hz), FP, "jaws over and under the carry handle", lg),
             Waypoint(0.8, (hx, hy, hz), FINGER_CLOSED, "hold the carry handle", lg)]
        tl = sum(w.duration for w in L)
        R = [Waypoint(tl, PARK_R, FINGER_OPEN, "wait for the steadying hand")] + R
        t = sum(w.duration for w in R)
        L += [Waypoint(max(0.5, t - 6.0 - tl), (hx, hy, hz), FINGER_CLOSED, "keep holding the carry handle", lg),
              Waypoint(0.6, (hx, hy, hz), FP, "let go of the carry handle", lg),
              Waypoint(1.2, (hx, hy + 0.08, hz), FP, "back away", lg),
              Waypoint(2.0, PARK_L, FINGER_OPEN, "park"),
              Waypoint(3.0, PARK_L, FINGER_OPEN, "hold still")]
        R += [Waypoint(max(0.5, sum(w.duration for w in L) - t), PARK_R, FINGER_OPEN, "hold still")]
    else:
        t = sum(w.duration for w in R)
        L = [Waypoint(t, PARK_L, FINGER_OPEN, "parked")]
    if os.environ.get("EX_MODE") == "hold":   # hold test: the left hand grips the carry handle; the right stays parked
        R = [Waypoint(sum(w.duration for w in L), PARK_R, FINGER_OPEN, "parked")]
    return dict(left=L, right=R)
