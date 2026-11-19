"""Generate 1:1 SVG/DXF fabrication templates and dimensioned closure cross-section for cake-box.

Outputs in tasks/cake-box/fab/ (all LF-normalized, derived directly from task_metrics.py):
  - cake_board_template.svg       1:1 cutting, beveling & centering template for 247.6 x 4.8 mm foil board
  - front_lifting_ear_template.svg 1:1 cutting/scoring template & carton-prep diagram for +Y front ear
  - closure_cross_section.svg     Dimensioned XZ cross-section of tab / front_panel / wall / relief / force paths
  - templates.dxf                 1:1 R12 ASCII DXF (CUT / SCORE / GUIDE layers) for laser or knife plotter
"""
from __future__ import annotations

import ast
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def _load_metric_scalars(metrics_py: pathlib.Path) -> dict[str, float]:
    tree = ast.parse(metrics_py.read_text(encoding="utf-8"))
    vals: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                vals[name] = float(node.value.value)
    return vals


_M = _load_metric_scalars(HERE.parent / "task_metrics.py")
W_IN = _M["W_IN"]
H_WALL = _M["H_WALL"]
T_BASE = _M["T_BASE"]
R_BOARD = _M["R_BOARD"]
T_BOARD = _M["T_BOARD"]
R_CAKE_BODY = _M["R_CAKE_BODY"]
R_BORDER_BOT = _M["R_BORDER_BOT"]
H_ROSETTE_TOP = _M["H_ROSETTE_TOP"]
BOARD_IN_MARGIN = _M["BOARD_IN_MARGIN"]
TAB_TUCK_DEPTH_M = _M["TAB_TUCK_DEPTH_M"]
HAND_CLEAR_M = _M["HAND_CLEAR_M"]
assert math.isclose(HAND_CLEAR_M, 0.0800), HAND_CLEAR_M


def _write_lf(path: pathlib.Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def build_cake_board_svg() -> str:
    d_stock = 254.0
    d_board = R_BOARD * 2000.0        # 247.6 mm
    d_bevel_in = (R_BOARD - 0.0030) * 2000.0  # 241.6 mm (3.0 mm x 1.2 mm bottom ski bevel)
    d_border = R_BORDER_BOT * 2000.0  # 217.0 mm (leaves 15.3 mm exposed rim)
    d_cake = R_CAKE_BODY * 2000.0     # 208.0 mm
    rim_mm = (R_BOARD - R_BORDER_BOT) * 1000.0  # 15.3 mm
    t_mm = T_BOARD * 1000.0           # 4.8 mm

    w_mm, h_mm = 270.0, 305.0
    cx, cy = 135.0, 140.0
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.1f}mm" height="{h_mm:.1f}mm" viewBox="0 0 {w_mm:.1f} {h_mm:.1f}">
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1a1a1a; }}
    .cut {{ fill: none; stroke: #cc0000; stroke-width: 0.45; }}
    .stock {{ fill: none; stroke: #888888; stroke-width: 0.35; stroke-dasharray: 3,2; }}
    .guide {{ fill: none; stroke: #0055aa; stroke-width: 0.35; stroke-dasharray: 2,1.5; }}
    .dim {{ stroke: #222222; stroke-width: 0.30; fill: none; }}
  </style>
  <rect x="0" y="0" width="{w_mm:.1f}" height="{h_mm:.1f}" fill="#ffffff"/>
  <text x="135" y="9" font-size="4.4" font-weight="bold" text-anchor="middle">1:1 FABRICATION TEMPLATE: 247.6 x {t_mm:.1f} mm TWO-PLY FOIL CORRUGATED CAKE CIRCLE (~55 g)</text>

  <!-- 10 in nominal stock outline (254.0 mm) -->
  <circle cx="{cx}" cy="{cy}" r="{d_stock/2:.2f}" class="stock"/>
  <!-- Finished outer cut line: 247.6 mm OD (leaves 3.2 mm nominal clearance per side in 254.0 mm carton) -->
  <circle cx="{cx}" cy="{cy}" r="{d_board/2:.2f}" class="cut"/>
  <!-- Bottom ski-bevel inner guide: 241.6 mm OD (sand/chamfer bottom kraft edge 3.0 mm radial x 1.2 mm vertical) -->
  <circle cx="{cx}" cy="{cy}" r="{d_bevel_in/2:.2f}" class="guide"/>
  <!-- Bottom royal-icing border outer limit: 217.0 mm OD (108.5 mm radius -> 15.3 mm exposed silver pushing rim) -->
  <circle cx="{cx}" cy="{cy}" r="{d_border/2:.2f}" class="guide"/>
  <!-- Fondant dummy body centering circle: 208.0 mm OD (104.0 mm radius) -->
  <circle cx="{cx}" cy="{cy}" r="{d_cake/2:.2f}" class="guide"/>
  <line x1="{cx-6}" y1="{cy}" x2="{cx+6}" y2="{cy}" class="dim"/>
  <line x1="{cx}" y1="{cy-6}" x2="{cx}" y2="{cy+6}" class="dim"/>

  <text x="{cx}" y="{cy - d_board/2 + 4.5:.1f}" font-size="3.1" fill="#cc0000" text-anchor="middle">SOLID RED = CUT TO 247.6 +/- 0.5 mm OD (R = 123.8 mm)</text>
  <text x="{cx}" y="{cy - d_border/2 + 4.5:.1f}" font-size="3.0" fill="#0055aa" text-anchor="middle">MAX PIPED BORDER = 217.0 mm OD (LEAVES {rim_mm:.1f} mm SILVER PUSHING RIM)</text>
  <text x="{cx}" y="{cy - d_cake/2 + 5.0:.1f}" font-size="3.0" fill="#0055aa" text-anchor="middle">FONDANT CAKE BODY = 208.0 mm OD (CENTERED)</text>
  <text x="{cx}" y="{cy + 6.0:.1f}" font-size="3.0" text-anchor="middle">Stock: 2x Wilton 10 in Corrugated White Cake Circles (MFR 191001569)</text>

  <!-- 100.0 mm print verification scale bar -->
  <line x1="85" y1="274" x2="185" y2="274" stroke="#000000" stroke-width="0.6"/>
  <line x1="85" y1="271.5" x2="85" y2="276.5" stroke="#000000" stroke-width="0.6"/>
  <line x1="185" y1="271.5" x2="185" y2="276.5" stroke="#000000" stroke-width="0.6"/>
  <text x="135" y="272.5" font-size="3.2" font-weight="bold" text-anchor="middle">PRINT SCALE CHECK: 100.0 mm EXACT (PRINT AT 100% / ACTUAL SIZE)</text>

  <text x="10" y="282.5" font-size="2.9">1. Laminate two 10 in Wilton MFR 191001569 single-wall B-flute circles (~2.4 mm/ply) at 90 deg crossed flutes with thin PVA glue; press flat ({t_mm:.1f} +/- 0.4 mm).</text>
  <text x="10" y="287.0" font-size="2.9">2. Wrap top face smoothly with 0.018 mm household silver aluminum foil (tucking only onto top/side or flush before final trim; keep kraft underside exposed).</text>
  <text x="10" y="291.5" font-size="2.9">3. Trim around SOLID RED circle to 247.6 +/- 0.5 mm OD; lightly chamfer/sand bottom kraft perimeter 3.0 mm inward x 1.2 mm high so board glides onto 1.0 mm box base.</text>
  <text x="10" y="296.0" font-size="2.9">4. Pre-trial check: caliper thickness {t_mm:.1f} +/- 0.4 mm, steel-ruler OD 247.6 +/- 0.5 mm (3.2 mm nominal clearance/side in 254.0 mm box), 1 g scale mass 55 +/- 6 g.</text>
</svg>
"""


def build_front_ear_svg() -> str:
    w_mm, h_mm = 210.0, 235.0
    # Blank sits at (20, 28): two-ply 116.0 x 36.0 mm blank -> folded in half along x = 20+58 = 78 mm
    # yielding a 58.0 x 36.0 x 1.2-1.6 mm two-ply piece, then scored 20.0 mm from left end (x = 40 mm)
    # to form a 20.0 x 36.0 mm mounting flange + 38.0 x 36.0 mm perpendicular lifting ear.
    bx, by = 22.0, 28.0
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.1f}mm" height="{h_mm:.1f}mm" viewBox="0 0 {w_mm:.1f} {h_mm:.1f}">
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1a1a1a; }}
    .cut {{ fill: #fdfcf9; stroke: #cc0000; stroke-width: 0.45; }}
    .fold {{ fill: none; stroke: #0066cc; stroke-width: 0.40; stroke-dasharray: 3,1.8; }}
    .box {{ fill: #f7f6f2; stroke: #222222; stroke-width: 0.40; }}
    .ear {{ fill: #ebe6dc; stroke: #cc0000; stroke-width: 0.45; }}
  </style>
  <rect x="0" y="0" width="{w_mm:.1f}" height="{h_mm:.1f}" fill="#ffffff"/>
  <text x="105" y="10" font-size="4.2" font-weight="bold" text-anchor="middle">1:1 TEMPLATE &amp; CARTON PREP: +Y FRONT-PANEL LIFTING EAR (BRP #245 / SCT #23053)</text>

  <!-- 1:1 24-pt (0.6 mm) SBS cardstock blank: 116.0 x 36.0 mm -->
  <text x="{bx}" y="23" font-size="3.3" font-weight="bold">A. 1:1 Cut &amp; Score Dieline (cut from 24-pt / 0.6 mm white SBS cardstock or removed front gusset flap):</text>
  <rect x="{bx}" y="{by}" width="116.0" height="36.0" class="cut"/>
  <!-- Center 180-deg lamination fold at +58.0 mm -->
  <line x1="{bx+58.0}" y1="{by}" x2="{bx+58.0}" y2="{by+36.0}" class="fold"/>
  <!-- 90-deg flange score at +20.0 mm (and mirror +96.0 mm so fold works cleanly) -->
  <line x1="{bx+20.0}" y1="{by}" x2="{bx+20.0}" y2="{by+36.0}" class="fold"/>
  <line x1="{bx+96.0}" y1="{by}" x2="{bx+96.0}" y2="{by+36.0}" class="fold"/>

  <text x="{bx+10.0}" y="{by+16.5}" font-size="2.9" text-anchor="middle">20.0 mm</text>
  <text x="{bx+10.0}" y="{by+21.0}" font-size="2.7" text-anchor="middle">FLANGE</text>
  <text x="{bx+39.0}" y="{by+16.5}" font-size="3.0" font-weight="bold" text-anchor="middle">38.0 x 36.0 mm</text>
  <text x="{bx+39.0}" y="{by+21.0}" font-size="2.8" text-anchor="middle">LIFTING EAR (PLY 1)</text>
  <text x="{bx+77.0}" y="{by+16.5}" font-size="3.0" font-weight="bold" text-anchor="middle">38.0 x 36.0 mm</text>
  <text x="{bx+77.0}" y="{by+21.0}" font-size="2.8" text-anchor="middle">LIFTING EAR (PLY 2)</text>
  <text x="{bx+106.0}" y="{by+16.5}" font-size="2.9" text-anchor="middle">20.0 mm</text>
  <text x="{bx+106.0}" y="{by+21.0}" font-size="2.7" text-anchor="middle">FLANGE</text>

  <text x="{bx}" y="{by+42.0}" font-size="2.9">Fold 180 deg at x = 58.0 mm &amp; glue plies flat -> 58.0 x 36.0 x 1.2-1.6 mm (~1.3 g); bend 20.0 mm flange 90 deg along score line.</text>

  <!-- 50.0 mm scale bar -->
  <line x1="146" y1="46" x2="196" y2="46" stroke="#000000" stroke-width="0.6"/>
  <line x1="146" y1="43.5" x2="146" y2="48.5" stroke="#000000" stroke-width="0.6"/>
  <line x1="196" y1="43.5" x2="196" y2="48.5" stroke="#000000" stroke-width="0.6"/>
  <text x="171" y="42.5" font-size="3.0" font-weight="bold" text-anchor="middle">50.0 mm EXACT</text>

  <!-- Schematic placement & lock-corner carton preparation -->
  <text x="22" y="84" font-size="3.3" font-weight="bold">B. Declared Preparation of 10 x 10 x 5 in Lock-Corner Window Carton (BRP #245 / SCT #23053):</text>
  <!-- Top-view schematic (not 1:1, labeled in mm) -->
  <rect x="26" y="92" width="76" height="76" class="box"/>
  <text x="64" y="131" font-size="3.1" text-anchor="middle">BOX BASE 254.0 x 254.0 mm</text>
  <text x="64" y="136" font-size="2.8" text-anchor="middle">(+X back wall &amp; +/-Y side walls erected)</text>
  <!-- Unfolded front_panel (-X) -->
  <rect x="26" y="168" width="76" height="36" fill="#fcfbf7" stroke="#222222" stroke-width="0.4" stroke-dasharray="2,1"/>
  <text x="64" y="185" font-size="2.9" text-anchor="middle">front_panel (251.0 x 126.0 mm)</text>
  <text x="64" y="190" font-size="2.7" text-anchor="middle">unfolded -90 deg flat onto counter</text>
  <!-- Mounted +Y lifting ear -->
  <rect x="102" y="191" width="4.5" height="11" class="ear"/>
  <text x="110" y="196" font-size="2.8" fill="#aa0000">+Y Lifting Ear (38.0 x 36.0 x 1.6 mm)</text>
  <text x="110" y="200.5" font-size="2.7">stands 39 mm upright off counter at y_box = +130.4 mm</text>
  <text x="110" y="205.0" font-size="2.7">(3.4 mm outside 254.0 mm throat; clear of 247.6 mm board)</text>

  <text x="110" y="98" font-size="2.85">1. Rear (+X) corners: hook factory lock tabs into rear side-wall</text>
  <text x="114" y="102.5" font-size="2.85">slots &amp; secure outer rear corners with 19 x 60 mm clear tape</text>
  <text x="114" y="107.0" font-size="2.85">so back_wall, left_wall, and right_wall stay rigid at 90 deg.</text>
  <text x="110" y="114.0" font-size="2.85">2. Front (-X) corners: unhook front lock tabs and slit the two</text>
  <text x="114" y="118.5" font-size="2.85">45 deg front diagonal gusset webs flush along the front vertical</text>
  <text x="114" y="123.0" font-size="2.85">edges of left_wall &amp; right_wall so front_panel unfolds -90 deg</text>
  <text x="114" y="127.5" font-size="2.85">flat onto the counter and seats against the side walls at 0 deg.</text>
  <text x="110" y="134.5" font-size="2.85">3. Tape the 20.0 x 36.0 mm ear flange (19 mm double-sided tape)</text>
  <text x="114" y="139.0" font-size="2.85">onto front_panel's outer face at its +Y corner, 4.0 mm below</text>
  <text x="114" y="143.5" font-size="2.85">its top edge (z = 88.0..124.0 mm), with the 38.0 x 36.0 mm ear</text>
  <text x="114" y="148.0" font-size="2.85">projecting +38.0 mm along +X_box at y_box = +130.4 mm.</text>
  <text x="110" y="155.0" font-size="2.85">4. Closure retention: folding front_panel to 0 deg, closing the</text>
  <text x="114" y="159.5" font-size="2.85">257.2 mm window lid onto the walls, and tucking the 35.0 mm</text>
  <text x="114" y="164.0" font-size="2.85">front tab &gt;= 10.0 mm behind front_panel's inner face traps</text>
  <text x="114" y="168.5" font-size="2.85">front_panel between the side walls and tucked tab.</text>
</svg>
"""


def build_cross_section_svg() -> str:
    # Dimensioned XZ cross-section near front wall (-X_box) and full box cavity
    # Scale s = 0.62 mm_svg per mm_real for full-box XZ view + 2.6x zoom inset on tab/front_panel/hinge relief!
    w_mm, h_mm = 275.0, 195.0
    # Zoom inset centered on (-127.0, +128.0) mm at 2.4 px/mm:
    # Real x in [-160, -85] mm, real z in [80, 150] mm -> mapped to x_svg = 30 + (x + 160)*2.2, y_svg = 152 - (z - 75)*1.8
    def zx(x_real_mm: float) -> float:
        return 22.0 + (x_real_mm + 162.0) * 1.95

    def zz(z_real_mm: float) -> float:
        return 154.0 - (z_real_mm - 76.0) * 1.75

    # Front panel upright: inner face x = -127.0, outer face x = -127.8, z = 1.0..127.0 mm
    fp_x0, fp_x1 = zx(-127.8), zx(-127.0)
    fp_z0, fp_z1 = zz(127.0), zz(78.0)
    # Lid flat (q_lid = 0): z = 128.0..129.6 mm, x = -128.6..-85.0 mm
    lid_x0, lid_x1 = zx(-128.6), zx(-86.0)
    lid_z0, lid_z1 = zz(129.6), zz(128.0)
    # TAB_ANCHOR = (-124.5, 128.0) mm
    ta_x, ta_z = zx(-124.5), zz(128.0)

    # Tucked tab (q_tab = -57.8 deg: tip at (-124.5 + 35*sin(57.8), 128.0 - 35*cos(57.8)) = (-94.88, 109.35) mm)
    a_tuck = math.radians(57.84)
    t_r14_x, t_r14_z = zx(-124.5 + 14.5 * math.sin(a_tuck)), zz(128.0 - 14.5 * math.cos(a_tuck))
    t_tip_x, t_tip_z = zx(-124.5 + 35.0 * math.sin(a_tuck)), zz(128.0 - 35.0 * math.cos(a_tuck))

    # Untucked tab (0.75 control: q_tab = +11.45 deg outward: tip at (-124.5 - 35*sin(11.45), 128.0 - 35*cos(11.45)) = (-131.45, 93.70) mm)
    a_out = math.radians(11.45)
    u_r14_x, u_r14_z = zx(-124.5 - 14.5 * math.sin(a_out)), zz(128.0 - 14.5 * math.cos(a_out))
    u_tip_x, u_tip_z = zx(-124.5 - 35.0 * math.sin(a_out)), zz(128.0 - 35.0 * math.cos(a_out))

    # Cake rosette peak height = T_BASE + T_BOARD + H_ROSETTE_TOP = 1.0 + 4.8 + 93.5 = 99.3 mm (28.7 mm below z=128.0 lid!)
    ck_top_z = zz((T_BASE + T_BOARD + H_ROSETTE_TOP) * 1000.0)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.1f}mm" height="{h_mm:.1f}mm" viewBox="0 0 {w_mm:.1f} {h_mm:.1f}">
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1a1a1a; }}
  </style>
  <rect x="0" y="0" width="{w_mm:.1f}" height="{h_mm:.1f}" fill="#ffffff"/>
  <text x="137.5" y="9.5" font-size="4.1" font-weight="bold" text-anchor="middle">DIMENSIONED XZ CLOSURE CROSS-SECTION: TAB TUCK, 14.5 mm HINGE RELIEF &amp; FORCE PATHS</text>

  <!-- Zoomed front-throat cross-section box -->
  <rect x="14" y="15" width="166" height="145" fill="#fafbfc" stroke="#cccccc" stroke-width="0.35"/>
  <text x="18" y="21.5" font-size="3.2" font-weight="bold">Front-Throat XZ Detail (x_box in [-160, -86] mm, z in [78, 148] mm):</text>

  <!-- Side wall top reference line at z = 128.0 mm (T_BASE 1.0 + H_WALL 127.0 mm) -->
  <line x1="{zx(-155):.1f}" y1="{zz(128.0):.1f}" x2="{zx(-86):.1f}" y2="{zz(128.0):.1f}" stroke="#888888" stroke-width="0.3" stroke-dasharray="2,2"/>
  <text x="{zx(-154):.1f}" y="{zz(128.0)-1.5:.1f}" font-size="2.6" fill="#555555">Side-wall top z = 128.0 mm (lid seats &lt;= 3.0 mm gap)</text>

  <!-- Front panel upright (q_front = 0 deg): x = -127.8..-127.0 mm, top at z = 127.0 mm -->
  <rect x="{fp_x0:.1f}" y="{fp_z0:.1f}" width="{max(1.6, fp_x1-fp_x0):.1f}" height="{fp_z1-fp_z0:.1f}" fill="#dcd9d0" stroke="#222222" stroke-width="0.45"/>
  <text x="{fp_x0-2.0:.1f}" y="{zz(103.0):.1f}" font-size="2.7" text-anchor="end">front_panel (0.8 mm proxy,</text>
  <text x="{fp_x0-2.0:.1f}" y="{zz(100.5):.1f}" font-size="2.7" text-anchor="end">inner face x = -127.0 mm,</text>
  <text x="{fp_x0-2.0:.1f}" y="{zz(98.0):.1f}" font-size="2.7" text-anchor="end">top edge z = 127.0 mm)</text>

  <!-- Seated window lid (q_lid = 0 deg): z = 128.0..129.6 mm -->
  <rect x="{lid_x0:.1f}" y="{lid_z0:.1f}" width="{lid_x1-lid_x0:.1f}" height="{lid_z1-lid_z0:.1f}" fill="#eceae4" stroke="#222222" stroke-width="0.45"/>
  <text x="{zx(-105):.1f}" y="{lid_z0-2.0:.1f}" font-size="2.8" text-anchor="middle">Seated window lid (1.6 mm proxy; LID_ANCHOR at x = +127.8, z = 128.0 mm)</text>

  <!-- TAB_ANCHOR pivot dot -->
  <circle cx="{ta_x:.1f}" cy="{ta_z:.1f}" r="1.3" fill="#cc0000"/>
  <text x="{ta_x+2.0:.1f}" y="{ta_z-4.5:.1f}" font-size="2.65" fill="#aa0000">TAB_ANCHOR (-124.5, 128.0) mm (2.5 mm inside front_panel inner face)</text>

  <!-- TUCKED TAB (1.0 tier, q_tab = -57.8 deg): dashed 0..14.5 mm hinge-root relief + solid 14.5..35.0 mm collider -->
  <line x1="{ta_x:.1f}" y1="{ta_z:.1f}" x2="{t_r14_x:.1f}" y2="{t_r14_z:.1f}" stroke="#007733" stroke-width="0.8" stroke-dasharray="2.2,1.4"/>
  <line x1="{t_r14_x:.1f}" y1="{t_r14_z:.1f}" x2="{t_tip_x:.1f}" y2="{t_tip_z:.1f}" stroke="#007733" stroke-width="2.1"/>
  <text x="{t_tip_x+2.0:.1f}" y="{t_tip_z:.1f}" font-size="2.75" fill="#006622" font-weight="bold">1.0 TUCKED (q_tab = -57.8 deg):</text>
  <text x="{t_tip_x+2.0:.1f}" y="{t_tip_z+3.6:.1f}" font-size="2.65" fill="#006622">tip 30.9 mm behind x = -127.0 mm (&gt;= 10.0 mm gate)</text>
  <text x="{t_r14_x+2.5:.1f}" y="{t_r14_z-1.5:.1f}" font-size="2.55" fill="#006622">r = 0..14.5 mm collider relief (dashed)</text>

  <!-- UNTUCKED TAB (0.75 control, q_tab = +11.5 deg outward): passes cleanly outside front_panel -->
  <line x1="{ta_x:.1f}" y1="{ta_z:.1f}" x2="{u_r14_x:.1f}" y2="{u_r14_z:.1f}" stroke="#bb5500" stroke-width="0.8" stroke-dasharray="2.2,1.4"/>
  <line x1="{u_r14_x:.1f}" y1="{u_r14_z:.1f}" x2="{u_tip_x:.1f}" y2="{u_tip_z:.1f}" stroke="#bb5500" stroke-width="2.0"/>
  <text x="{u_tip_x-2.0:.1f}" y="{u_tip_z+4.5:.1f}" font-size="2.7" fill="#994400" text-anchor="end">0.75 UNTUCKED (+11.5 deg): distal collider</text>
  <text x="{u_tip_x-2.0:.1f}" y="{u_tip_z+8.0:.1f}" font-size="2.65" fill="#994400" text-anchor="end">r = 14.5..35.0 mm hangs outside front_panel</text>

  <!-- Cake peak height clearance line -->
  <line x1="{zx(-118):.1f}" y1="{ck_top_z:.1f}" x2="{zx(-86):.1f}" y2="{ck_top_z:.1f}" stroke="#cc4477" stroke-width="0.45" stroke-dasharray="2.5,1.5"/>
  <text x="{zx(-102):.1f}" y="{ck_top_z+3.8:.1f}" font-size="2.65" fill="#aa2255" text-anchor="middle">Cake rosette peak z = 99.3 mm (28.7 mm clear below closed lid &amp; tucked tab!)</text>

  <!-- Right-hand engineering summary panel -->
  <rect x="184" y="15" width="85" height="172" fill="#f8f9fb" stroke="#cccccc" stroke-width="0.35"/>
  <text x="188" y="22" font-size="3.1" font-weight="bold">Reproducible Mechanics &amp; Force Paths:</text>
  <text x="188" y="28.5" font-size="2.65">1. Real vs Sim Tab Hinge Relief (0..14.5 mm):</text>
  <text x="190" y="33.0" font-size="2.55">In real 0.6 mm SBS board, an uncreased outward</text>
  <text x="190" y="37.0" font-size="2.55">tab bows ~2.5 mm near the lid score line so the</text>
  <text x="190" y="41.0" font-size="2.55">lid still closes flat with the tab outside (0.75).</text>
  <text x="190" y="45.0" font-size="2.55">Relieving rigid tab_boxes on r = 0..14.5 mm</text>
  <text x="190" y="49.0" font-size="2.55">(collider on r = 14.5..35.0 mm) reproduces both</text>
  <text x="190" y="53.0" font-size="2.55">0.75 (tab outside) and 1.0 (tucked &gt;= 10 mm).</text>

  <text x="188" y="60.5" font-size="2.65">2. Estimated Crease Parameters (disclosed):</text>
  <text x="190" y="65.0" font-size="2.55">- front_panel: friction 0.026 N*m, damp 0.008</text>
  <text x="190" y="69.0" font-size="2.55">- lid: friction 0.082 N*m, hysteresis 0.085 N*m</text>
  <text x="190" y="73.0" font-size="2.55">- lid_tab: spring 0.001 N*m/rad around +45 deg,</text>
  <text x="190" y="77.0" font-size="2.55">  stiction 0.008 N*m, hysteresis 0.014 N*m.</text>

  <text x="188" y="84.5" font-size="2.65">3. Force Paths by Phase:</text>
  <text x="190" y="89.0" font-size="2.55">- Slide-in: Left clamps left_wall (load -> counter</text>
  <text x="190" y="93.0" font-size="2.55">  friction mu=0.52); Right pushes 15.3 mm board</text>
  <text x="190" y="97.0" font-size="2.55">  rim (~1.1 N &lt;&lt; ~5.0 N cake-on-board shear).</text>
  <text x="190" y="101.5" font-size="2.55">- Flap fold: Right clamps right_wall; Left pushes</text>
  <text x="190" y="105.5" font-size="2.55">  +Y ear arc (-90 -&gt; -32 deg) then center to 0 deg.</text>
  <text x="190" y="110.0" font-size="2.55">- Tab crease: Left braces lid at ~44 deg while</text>
  <text x="190" y="114.0" font-size="2.55">  Right sweeps r = 26..30 mm (+65 -&gt; -61 deg,</text>
  <text x="190" y="118.0" font-size="2.55">  ~0.85 N aligned through LID_ANCHOR).</text>
  <text x="190" y="122.5" font-size="2.55">- Lid seat: Right guards front_panel outer face;</text>
  <text x="190" y="126.5" font-size="2.55">  Left seats lid onto all four 127.0 mm walls.</text>

  <text x="188" y="134.0" font-size="2.65">4. Board Boundary &amp; Release Gate:</text>
  <text x="190" y="138.5" font-size="2.55">- Analytic 3D cylinder projection &lt;= {W_IN*500 + BOARD_IN_MARGIN*1000:.1f} mm</text>
  <text x="190" y="142.5" font-size="2.55">  (127.0 mm inner wall + 1.0 mm wall/ruler tol).</text>
  <text x="190" y="146.5" font-size="2.55">- All 6 pad/jaw points &gt;= 80.0 mm clear for full</text>
  <text x="190" y="150.5" font-size="2.55">  21-sample (1.0 s) &lt; 1.0 mm 3D stillness window.</text>
  <text x="190" y="155.0" font-size="2.55">- Tab free edge &gt;= {TAB_TUCK_DEPTH_M*1000:.1f} mm behind x = -127.0 mm.</text>

  <text x="14" y="169" font-size="2.8">Prospective real-trial retention check (procedure, not a pre-claimed measurement): after 1.0 s stillness with &gt;= 80 mm ruler/gauge hand clearance,</text>
  <text x="14" y="173.5" font-size="2.8">lift the closed carton 50 mm by its side walls and tilt 30 +/- 2 deg forward against a 30 deg wedge for 3 s; front_panel and lid_tab must stay tucked.</text>
</svg>
"""


def build_dxf() -> str:
    """Minimal clean R12 ASCII DXF in millimetres (1:1) with CUT, SCORE, and GUIDE layers."""
    entities: list[str] = []

    def add_circle(layer: str, cx: float, cy: float, r: float) -> None:
        entities.extend([
            "0", "CIRCLE", "8", layer,
            "10", f"{cx:.4f}", "20", f"{cy:.4f}", "30", "0.0",
            "40", f"{r:.4f}",
        ])

    def add_line(layer: str, x1: float, y1: float, x2: float, y2: float) -> None:
        entities.extend([
            "0", "LINE", "8", layer,
            "10", f"{x1:.4f}", "20", f"{y1:.4f}", "30", "0.0",
            "11", f"{x2:.4f}", "21", f"{y2:.4f}", "31", "0.0",
        ])

    def add_rect(layer: str, x0: float, y0: float, w: float, h: float) -> None:
        add_line(layer, x0, y0, x0 + w, y0)
        add_line(layer, x0 + w, y0, x0 + w, y0 + h)
        add_line(layer, x0 + w, y0 + h, x0, y0 + h)
        add_line(layer, x0, y0 + h, x0, y0)

    # 1. Cake board centered at (0, 0) mm
    add_circle("GUIDE", 0.0, 0.0, 127.0000)                  # 254.0 mm 10-in stock circle
    add_circle("CUT", 0.0, 0.0, R_BOARD * 1000.0)            # 123.8000 mm (247.6 mm OD finished cut)
    add_circle("GUIDE", 0.0, 0.0, (R_BOARD - 0.0030) * 1000.0)  # 120.8000 mm bottom ski bevel inner guide
    add_circle("GUIDE", 0.0, 0.0, R_BORDER_BOT * 1000.0)     # 108.5000 mm bottom shell border outer limit
    add_circle("GUIDE", 0.0, 0.0, R_CAKE_BODY * 1000.0)      # 104.0000 mm fondant body centering circle

    # 2. Front +Y lifting-ear 116.0 x 36.0 mm two-ply blank placed at x = [150, 266], y = [-18, +18] mm
    x0, y0 = 150.0, -18.0
    add_rect("CUT", x0, y0, 116.0, 36.0)
    add_line("SCORE", x0 + 20.0, y0, x0 + 20.0, y0 + 36.0)   # 90-deg mounting flange score (ply 1)
    add_line("SCORE", x0 + 58.0, y0, x0 + 58.0, y0 + 36.0)   # 180-deg two-ply lamination fold
    add_line("SCORE", x0 + 96.0, y0, x0 + 96.0, y0 + 36.0)   # 90-deg mounting flange score (ply 2)

    lines = [
        "0", "SECTION", "2", "HEADER",
        "9", "$INSUNITS", "70", "4",  # 4 = millimetres
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
        *entities,
        "0", "ENDSEC",
        "0", "EOF",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    _write_lf(HERE / "cake_board_template.svg", build_cake_board_svg())
    _write_lf(HERE / "front_lifting_ear_template.svg", build_front_ear_svg())
    _write_lf(HERE / "closure_cross_section.svg", build_cross_section_svg())
    _write_lf(HERE / "templates.dxf", build_dxf())
    print("Generated 4 fabrication & cross-section files in", HERE)


if __name__ == "__main__":
    main()
