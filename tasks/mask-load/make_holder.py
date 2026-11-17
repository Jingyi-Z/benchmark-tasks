"""Write DXF cutting files (mm) for the real-trial mask holder: python make_holder.py [out_dir].

The holder is a 160 x 100 mm aluminium plate, 6 mm thick in all, built from two laser-cut layers bonded together:
a 5 mm base with the window, the two stop-pin holes and the vacuum port, and a 1 mm top layer cut as an outer and an
inner frame that leave a closed 2 mm-wide vacuum groove between them, under the mask's flat border. Plain DXF R12.
"""
import os
import sys

def _dxf(entities):
    out = ["0", "SECTION", "2", "ENTITIES"]
    for e in entities:
        if e[0] == "C":
            _, x, y, r = e
            out += ["0", "CIRCLE", "8", "0", "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0", "40", f"{r:.4f}"]
        elif e[0] == "L":
            _, x0, y0, x1, y1 = e
            out += ["0", "LINE", "8", "0", "10", f"{x0:.4f}", "20", f"{y0:.4f}", "30", "0",
                    "11", f"{x1:.4f}", "21", f"{y1:.4f}", "31", "0"]
        elif e[0] == "A":
            _, x, y, r, a0, a1 = e
            out += ["0", "ARC", "8", "0", "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0", "40", f"{r:.4f}",
                    "50", f"{a0:.4f}", "51", f"{a1:.4f}"]
    out += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"



def _rect(x0, y0, x1, y1):
    return [("L", x0, y0, x1, y0), ("L", x1, y0, x1, y1), ("L", x1, y1, x0, y1), ("L", x0, y1, x0, y0)]


# holder frame: origin at the plate's centre, +x towards the stop pins (back), y across
PLATE = (80.0, 50.0)       # half sizes of the 160 x 100 mm plate
WINDOW = (52.0, 30.0)      # half sizes of the 104 x 60 mm window
GROOVE_IN = (56.0, 38.0)   # inner edge of the vacuum groove (half sizes)
GROOVE_W = 2.0             # groove width: the gap between the top layer's two frames
PIN_X, PIN_Y, PIN_D = 66.5, 35.0, 6.0   # 6 mm stop pins; their front faces at x = 63.5 mm (the mask's back edge)
PORT = (-GROOVE_IN[0] - GROOVE_W / 2, 0.0, 2.0)   # 2 mm hole from the groove down through the base


def base():
    """6061 aluminium, 5 mm: plate, window, two 6 mm pin holes (press-fit dowel pins) and a 2 mm port hole into the
    groove (from below, drill 4.2 mm to 4 mm depth and tap M5 for a barb fitting)."""
    return (_rect(-PLATE[0], -PLATE[1], PLATE[0], PLATE[1]) + _rect(-WINDOW[0], -WINDOW[1], WINDOW[0], WINDOW[1])
            + [("C", PIN_X, PIN_Y, PIN_D / 2), ("C", PIN_X, -PIN_Y, PIN_D / 2), ("C", PORT[0], PORT[1], PORT[2] / 2)])


def top_outer():
    """6061 aluminium, 1 mm: outer frame, from the plate's edge in to the groove's outer edge."""
    gx, gy = GROOVE_IN[0] + GROOVE_W, GROOVE_IN[1] + GROOVE_W
    return (_rect(-PLATE[0], -PLATE[1], PLATE[0], PLATE[1]) + _rect(-gx, -gy, gx, gy)
            + [("C", PIN_X, PIN_Y, PIN_D / 2), ("C", PIN_X, -PIN_Y, PIN_D / 2)])


def top_inner():
    """6061 aluminium, 1 mm: inner frame, from the groove's inner edge in to the window."""
    return (_rect(-GROOVE_IN[0], -GROOVE_IN[1], GROOVE_IN[0], GROOVE_IN[1])
            + _rect(-WINDOW[0], -WINDOW[1], WINDOW[0], WINDOW[1]))


def main(out):
    os.makedirs(out, exist_ok=True)
    for name, ents in (("holder_base_5mm_al6061", base()), ("holder_top_outer_1mm_al6061", top_outer()),
                       ("holder_top_inner_1mm_al6061", top_inner())):
        with open(os.path.join(out, name + ".dxf"), "w") as f:
            f.write(_dxf(ents))
        print("wrote", name + ".dxf")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "parts")
