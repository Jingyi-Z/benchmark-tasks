"""Top-turn knob adapter for the Zeiss Axio Imager.A2m's coaxial stage drive (microscope-focus real trials).

The stand's two stage knobs share one vertical shaft hanging under the stage's front right (upper knob, inner shaft:
front to back; lower knob: left to right), where a robot gripper is too bulky to turn them. This adapter carries two
top-turn knobs on vertical 5 mm shafts outboard of the stage; each shaft turns one stage knob 1:1 through a 6 mm GT2
belt between two identical 64-tooth pulleys (a printed sleeve clamped on the stage knob, a printed pulley on the
shaft). So each top knob moves the stage exactly as the stage knob under it does.

The top knobs sit 20 mm BELOW the stage top: the robot's straight-down grip reaches about 0.95 m above the floor of its
base frame but not 0.98 m (yam_arena IK, 8 to 60 mm short), and the stage top is 0.96 m.

Mounting: the bracket's locating wall is bonded with 1 mm VHB tape (3M 4910) to the right edge face of the stage plate
that carries the coaxial control (assumed the lower plate, which moves only with focus). Nothing of the adapter rises
above that plate's top face inboard of its edge, so the upper plates can travel over its edge without touching it,
and because the control and the bracket are on the same plate, the belt centres never change over the stage's travel
or the focus stroke. A drop web behind the coaxial knobs carries the arm down to its level under the plate. If the
real control rides on a moving plate instead, bond the wall to that plate: the adapter then travels with it and the
belt centres are still fixed.

Run with cadquery 2.x:  python knob_adapter.py  ->  STEP and STL files in ./out/

MEASURED on the stand (Nov 18 2026): the two coaxial knobs' diameters, heights and tops below the stage top surface.
ASSUMED (measure before printing; set them below):
  PLATE_TOP_Z, PLATE_BOT_Z                  the carrying plate's top and bottom faces below the stage's top surface
  AXIS_IN, AXIS_BACK                        the coaxial shaft's distance inboard of the plate's right edge and behind
                                            its front edge
Frame: origin on the coaxial shaft's axis at the stage's top surface; u towards the user (front), v to the right
(outboard), z up. Print orientation: every printed part flat on the bed as generated (z up); no supports.
"""
import math
import pathlib

import cadquery as cq

# ---------------- parameters (mm) ----------------
X_KNOB_D, X_KNOB_H, X_KNOB_TOP = 32.8, 34.1, -40.0   # upper coaxial knob (front to back): diameter, height, top z (measured)
Y_KNOB_D, Y_KNOB_H, Y_KNOB_TOP = 27.8, 30.6, -80.0   # lower coaxial knob (left to right) (measured)
PLATE_TOP_Z, PLATE_BOT_Z = -11.0, -16.0              # the plate carrying the control (the model's lower stage plate)
AXIS_IN, AXIS_BACK = 15.0, 5.0                       # coaxial axis: inboard of the plate's right edge, behind its front edge
PLATE_FRONT_TO_BACK = 120.0                          # the carrying plate's depth (for the wall's length)
FIT = 0.4              # diametral clearance on the sleeve bores (free fit; 0.2 for snug)
BEARING_D, BEARING_W = 16.0, 5.0   # 625ZZ (5 x 16 x 5)
BEARING_FIT = 0.1      # press fit: bore = 16.1 mm (tune 0.0 to 0.2 for your printer)
SHAFT_D = 5.0
TEETH, PITCH = 64, 2.0             # GT2, 64 teeth: pitch diameter 40.74 mm (a 48-tooth band is too small for the
#                                    32.8 mm knob's sleeve)
BELT_W, BELT_T = 6.0, 1.4          # GT2-6 belt width and thickness
BELT_X_L, BELT_Y_L = 280.0, 400.0  # closed-loop GT2-6 belt lengths
TOPKNOB_D, TOPKNOB_H = 34.0, 16.0  # the top-turn knobs, as in the model
TOPKNOB_TOP = -20.0    # the top knobs' tops below the stage top surface (inside the robot's reach)
STRIP_V0 = 20.0        # the arm's strip starts here in v, clear of the 32.8 mm upper knob (radius 16.4) on the axis
ARM_T = 4.0            # bracket arm thickness
BOSS_D, BOSS_H = 24.0, 12.0        # bearing boss on the arm (two bearings)
VHB = 1.0              # tape thickness between the wall and the plate's edge face
SHAFT_U = 50.0         # both shafts this far in front of the coaxial axis (the arm to them stays clear of the
#                        robot's hand turning the right focus knob)

PD = TEETH * PITCH / math.pi       # pitch diameter
OD = PD - 0.508                    # GT2 outside diameter
FLANGE_R = OD / 2 + 1.5
C_X = (BELT_X_L - TEETH * PITCH) / 2   # centre distance for 1:1 equal pulleys
C_Y = (BELT_Y_L - TEETH * PITCH) / 2
X_SHAFT = (SHAFT_U, math.sqrt(C_X ** 2 - SHAFT_U ** 2))
Y_SHAFT = (SHAFT_U, math.sqrt(C_Y ** 2 - SHAFT_U ** 2))
EDGE_V = AXIS_IN                   # the carrying plate's right edge face, in v
FRONT_U = AXIS_BACK                # its front edge, in u
WALL_V0 = EDGE_V + VHB             # the wall's inner face
KNOB_Z0 = TOPKNOB_TOP - TOPKNOB_H  # the top knobs' undersides
ARM_TOP = KNOB_Z0 - 1.0 - BOSS_H   # the arm's top face (1 mm washer and the boss under each knob)
ARM_Z0 = ARM_TOP - ARM_T
ARM_V0 = STRIP_V0                  # the arm starts outboard of the plate edge and of the upper knob
WALL_TOP = PLATE_TOP_Z - 0.5       # the wall's top: 0.5 mm under the plate's top face, so plates above clear it
X_BELT_Z = X_KNOB_TOP - X_KNOB_H + 8.0  # belt mid-heights: near the bottom of the upper knob (the shaft pulley must
Y_BELT_Z = Y_KNOB_TOP - Y_KNOB_H / 2    # clear the arm above it) and at the middle of the lower knob
RING_H = BELT_W + 1.0              # toothed band height (the belt plus 1 mm)
RING_Z0 = -(RING_H + 2.0) / 2      # a pulley's or sleeve's bottom, relative to its belt mid-height
OUT = pathlib.Path(__file__).resolve().parent / "out"


def gt2_ring(bore):
    """A GT2 pulley body (TEETH teeth): 1 mm flat flange at the bottom, the toothed band, a 45 degree cone flange on top
    (no flat ledge printed in the air); z from 0 to RING_H + 3.25, bore through."""
    body = cq.Workplane("XY").circle(OD / 2).extrude(RING_H)
    for i in range(TEETH):
        a = 2 * math.pi * i / TEETH
        r = OD / 2 + 0.555 - 0.75
        body = body.cut(cq.Workplane("XY").center(r * math.cos(a), r * math.sin(a)).circle(0.555).extrude(RING_H))
    flange = cq.Workplane("XY").circle(FLANGE_R).extrude(1.0)
    top = cq.Solid.makeCone(OD / 2 - 0.75, FLANGE_R, 2.25, cq.Vector(0, 0, RING_H + 1.0))
    ring = flange.union(body.translate((0, 0, 1.0))).union(cq.Workplane("XY").add(top))
    return ring.cut(cq.Workplane("XY").circle(bore / 2).extrude(RING_H + 5.0))


def set_screw(part, z, length=20.0):
    """An M3 set-screw hole (2.5 mm tap drill) radially through one side at height z."""
    return part.cut(cq.Workplane("YZ").center(0, z).circle(1.25).extrude(length))


def sleeve(knob_d):
    """Clamps onto a coaxial stage knob (bore = knob + FIT); its toothed band carries the belt; one M3 set screw."""
    return set_screw(gt2_ring(knob_d + FIT), 1.0 + RING_H / 2)


def shaft_pulley():
    """A TEETH-tooth pulley with its hub below the band (so its top stays clear of the arm above it)."""
    s = gt2_ring(SHAFT_D + 0.2)
    hub = cq.Workplane("XY").circle(7.0).circle((SHAFT_D + 0.2) / 2).extrude(6.0).translate((0, 0, -6.0))
    s = s.union(hub)
    return set_screw(s, -3.0)


def top_knob():
    """34 mm ribbed knob with a 5 mm blind bore and an M3 set screw; prints upside down (rim on the bed)."""
    k = cq.Workplane("XY").circle(TOPKNOB_D / 2 - 1.5).extrude(TOPKNOB_H)
    for i in range(12):
        a = 2 * math.pi * i / 12
        k = k.union(cq.Workplane("XY").center((TOPKNOB_D / 2 - 1.5) * math.cos(a), (TOPKNOB_D / 2 - 1.5) * math.sin(a))
                    .circle(2.5).extrude(TOPKNOB_H))
    k = k.cut(cq.Workplane("XY").circle((SHAFT_D + 0.2) / 2).extrude(TOPKNOB_H - 3.0))
    k = k.cut(cq.Workplane("XY").center(TOPKNOB_D / 2 - 6, 0).rect(6, 2).extrude(0.6).translate((0, 0, TOPKNOB_H - 0.6)))
    return set_screw(k, 6.0)


def bracket():
    """The locating wall bonded to the carrying plate's right edge face (its top 0.5 mm under the plate's top face), a
    drop web behind the coaxial knobs down to the arm, the arm (outboard of the plate and of the upper knob) and two
    bearing bosses standing on it under the top knobs."""
    u0, u1 = FRONT_U - 80.0, SHAFT_U + 16.0
    v1 = Y_SHAFT[1] + 16.0
    # an L at arm level: a strip from behind the knobs out to the shafts' line, then a bar along that line out past the
    # y shaft, kept in front of the right focus knob (where the robot's hand turns it)
    strip = cq.Workplane("XY").box(u1 - u0, 14.0, ARM_T, centered=False).translate((u0, ARM_V0, ARM_Z0))
    bar = cq.Workplane("XY").box(2 * 16.0, v1 - ARM_V0, ARM_T, centered=False).translate((SHAFT_U - 16.0, ARM_V0, ARM_Z0))
    arm = strip.union(bar)
    wall = cq.Workplane("XY").box(FRONT_U - u0, 4.0, WALL_TOP - (PLATE_BOT_Z - 1.0), centered=False).translate(
        (u0, WALL_V0, PLATE_BOT_Z - 1.0))
    web = cq.Workplane("XY").box(10.0, ARM_V0 + 14.0 - WALL_V0, WALL_TOP - ARM_Z0, centered=False).translate(
        (u0, WALL_V0, ARM_Z0))  # 75 mm behind the coaxial axis, clear of both knobs
    b = arm.union(wall).union(web)
    for (u, v) in (X_SHAFT, Y_SHAFT):
        boss = cq.Workplane("XY").center(u, v).circle(BOSS_D / 2).extrude(BOSS_H).translate((0, 0, ARM_TOP))
        b = b.union(boss)
        seat = cq.Workplane("XY").center(u, v).circle((BEARING_D + BEARING_FIT) / 2).extrude(2 * BEARING_W + 0.01)
        b = b.cut(seat.translate((0, 0, ARM_TOP + BOSS_H - 2 * BEARING_W)))
        b = b.cut(cq.Workplane("XY").center(u, v).circle(SHAFT_D / 2 + 1.0).extrude(ARM_T + BOSS_H + 2)
                  .translate((0, 0, ARM_Z0 - 1)))
    return b


def belt(p0, p1, z_mid):
    """A closed belt's envelope around two equal pulleys at plan points p0, p1, BELT_W tall about z_mid."""
    (u0, v0), (u1, v1) = p0, p1
    c = math.dist(p0, p1)
    ang = math.degrees(math.atan2(v1 - v0, u1 - u0))

    def stadium(r):
        s = cq.Workplane("XY").box(c, 2 * r, BELT_W).translate((c / 2, 0, 0))
        s = s.union(cq.Workplane("XY").circle(r).extrude(BELT_W).translate((0, 0, -BELT_W / 2)))
        s = s.union(cq.Workplane("XY").circle(r).extrude(BELT_W).translate((c, 0, -BELT_W / 2)))
        return s

    loop = stadium(PD / 2 + BELT_T - 0.75).cut(stadium(PD / 2 - 0.75))
    return loop.rotate((0, 0, 0), (0, 0, 1), ang).translate((u0, v0, z_mid))


def assembly():
    """Every printed part and the belts where they sit in the frame (origin: coaxial axis at the stage top)."""
    P = {"bracket": bracket(), "sleeve_x": sleeve(X_KNOB_D), "sleeve_y": sleeve(Y_KNOB_D),
         "shaft_pulley": shaft_pulley(), "top_knob": top_knob()}
    A = {"bracket": P["bracket"],
         "sleeve_x": P["sleeve_x"].translate((0, 0, X_BELT_Z + RING_Z0)),
         "sleeve_y": P["sleeve_y"].translate((0, 0, Y_BELT_Z + RING_Z0)),
         "belt_x": belt((0, 0), X_SHAFT, X_BELT_Z), "belt_y": belt((0, 0), Y_SHAFT, Y_BELT_Z)}
    for tag, (u, v), zb in (("x", X_SHAFT, X_BELT_Z), ("y", Y_SHAFT, Y_BELT_Z)):
        A[f"pulley_{tag}"] = P["shaft_pulley"].translate((u, v, zb + RING_Z0))
        A[f"top_knob_{tag}"] = P["top_knob"].translate((u, v, KNOB_Z0))
    return P, A


def check_single(name, part):
    n = len(part.val().Solids())
    assert n == 1, f"{name}: {n} solids"


def main():
    OUT.mkdir(exist_ok=True)
    P, A = assembly()
    for name, p in P.items():
        check_single(name, p)
        cq.exporters.export(p, str(OUT / f"{name}.step"))
        bb = p.val().BoundingBox()
        print(f"{name}: 1 solid, {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm, volume {p.val().Volume() / 1000:.1f} cm3")
    for name, p in A.items():  # the assembled parts as STL in the frame, for the simulated scene
        cq.exporters.export(p, str(OUT / f"asm_{name}.stl"), tolerance=0.05, angularTolerance=0.2)
    print(f"pitch diameter {PD:.2f} mm; centre distances X {C_X:.1f}, Y {C_Y:.1f} mm; X shaft (u {X_SHAFT[0]:.1f}, "
          f"v {X_SHAFT[1]:.1f}), Y shaft (u {Y_SHAFT[0]:.1f}, v {Y_SHAFT[1]:.1f}); top knobs "
          f"{math.dist(X_SHAFT, Y_SHAFT):.1f} mm apart, tops {KNOB_Z0 + TOPKNOB_H:.1f} mm from the stage top")
    print(f"shaft lengths: X {KNOB_Z0 + TOPKNOB_H - 3.0 - (X_BELT_Z + RING_Z0 - 6.0):.1f} mm, "
          f"Y {KNOB_Z0 + TOPKNOB_H - 3.0 - (Y_BELT_Z + RING_Z0 - 6.0):.1f} mm")
    return P, A


if __name__ == "__main__":
    main()
