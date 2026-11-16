// Force-sensing pin for real trials of caliper-measure: the two aluminium half-sleeves (see README.md).
// Units: mm. Render half A and half B side by side with their flats up; export STL or a drawing from OpenSCAD.
// Machine from 1 in 6061 bar; the drawing gives the finished shapes (the outer diameter is held to +-0.01 mm).

D = 23.40;          // finished pin diameter (both halves together, flats 0.60 mm apart)
L = 40.0;           // pin length
GAP = 0.60;         // gap between the flats
H = D / 2 - GAP / 2; // each half: flat to top of the arc, 11.40 mm
SENSOR_D = 8.5;     // pocket for the SingleTact S8-10N (8 mm disc)
SENSOR_T = 0.30;    // head thickness: 0.30 mm on the product page, 0.35 mm on the maker's drawing; measure the actual
                    //   sensor's head with a micrometer and enter it here (BOSS follows)
POCKET = 0.15;      // pocket depth in half A
BOSS_D = 8.0;       // boss on half B that presses the sensor
// The halves touch each other only through the sensor, so the stack sets the gap: pocket floor to boss face is the
// sensor. Boss height = gap - (sensor - pocket) = 0.60 - 0.15 = 0.45 mm. Machine it 0.02 mm proud and lap it down
// until the assembled diameter reads 23.40 mm at 0.05 N (README, step 6).
BOSS = GAP - (SENSOR_T - POCKET);
TAIL_W = 4.0;       // channel for the sensor's flat tail (maker's drawing: up to 3.5 mm wide, 0.20 mm thick), from the
TAIL_D = 0.5;       //   pocket to the top end; with the 0.60 mm gap the tail has 1.1 mm of depth and is never pinched
SCREW_AT = 8.0;     // the two loose M2 x 16 nylon screws, this far from each end
SCREW_Y = 5.0;      // and 5 mm off the axis (half A's holes at y = +5; half B's at y = -5 before it is turned over)
M2_TAP = 1.6;       // M2 tapping hole in half A
TAP_DEPTH = 8.0;    // half A is 10.28 mm thick at y = 5, so 2.3 mm of wall remains above the hole
M2_CLEAR = 3.0;     // clearance hole in half B (so the screws hold but do not clamp)
SEAT_D = 5.5;       // flat seat for the head and two 0.5 mm nylon washers (5 mm OD) on half B's curved outside
SEAT_Z = 8.0;       // seat height above half B's flat: the washer's outer edge (y = 7.5) is still on metal (8.68 mm)
// Screw engagement: 16 - SEAT_Z (8.0) - GAP (0.6) - washers (1.0) = 6.4 mm of thread in half A, inside its 8 mm hole.

module half_body() {
    // a D section: the part of the cylinder above a flat H below its top, the flat at z = 0
    intersection() {
        translate([0, 0, H - D / 2]) rotate([0, 90, 0]) cylinder(d = D, h = L, $fn = 180);
        translate([0, -D / 2, 0]) cube([L, D, H]);
    }
}

module half_a() {
    difference() {
        half_body();
        translate([L / 2, 0, -0.01]) cylinder(d = SENSOR_D, h = POCKET + 0.01, $fn = 90);    // sensor pocket
        translate([L / 2, -TAIL_W / 2, -0.01]) cube([L / 2 + 0.01, TAIL_W, TAIL_D + 0.01]); // tail channel
        for (x = [SCREW_AT, L - SCREW_AT]) translate([x, SCREW_Y, -0.01]) cylinder(d = M2_TAP, h = TAP_DEPTH + 0.01, $fn = 30);
        translate([L - 6, -5, H / 2]) rotate([0, 90, 0]) cylinder(d = M2_TAP, h = 6.01, $fn = 30); // lead terminal, 6 mm deep from the top end
    }
}

module half_b() {
    difference() {
        union() {
            half_body();
            translate([L / 2, 0, -BOSS]) cylinder(d = BOSS_D, h = BOSS + 0.01, $fn = 90);         // sensor boss
        }
        for (x = [SCREW_AT, L - SCREW_AT]) {
            translate([x, -SCREW_Y, -BOSS - 0.01]) cylinder(d = M2_CLEAR, h = H + 1, $fn = 30);
            translate([x, -SCREW_Y, SEAT_Z]) cylinder(d = SEAT_D, h = H, $fn = 60);  // flat seat for the head
        }
        translate([L - 6, 5, H / 2]) rotate([0, 90, 0]) cylinder(d = M2_TAP, h = 6.01, $fn = 30);  // lead terminal, 6 mm deep from the top end
    }
}

// Assembly: half B turned over about the pin's axis (x) onto half A, its flat GAP below half A's flat. Turning it over
// maps y to -y, so half B's screw holes at y = -5 land on half A's at y = +5, and its terminal at y = +5 lands at -5,
// below half A's terminal (they are 2 x H/2 = 11.4 mm apart in z and never touch).
module assembled() {
    half_a();
    translate([0, 0, -GAP]) rotate([180, 0, 0]) half_b();
    %translate([L / 2, 0, POCKET - SENSOR_T]) cylinder(d = 8.0, h = SENSOR_T, $fn = 90);   // the sensor (transparent)
}

ASSEMBLED = false;  // true: show the assembled pin; false: both halves flat up, for machining
if (ASSEMBLED) assembled();
else {
    half_a();
    translate([0, D + 10, 0]) half_b();  // half B as machined, flat up (no mirroring)
}
