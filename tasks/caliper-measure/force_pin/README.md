# Force-sensing pin for real trials

The simulated pin is a plain aluminium cylinder, 23.40 mm across and 40 mm long. For real trials the same pin is built
in two halves with a thin force sensor between them, so a trial can check what the simulation checks: both caliper
jaws on the pin, and how hard they squeeze it.

- **Force:** the squeeze across the pin's diameter goes through a SingleTact S8-10N sensor between the halves.
- **Both jaws touching:** the two halves are insulated from each other. A continuity tester on the halves beeps
  only when both jaws touch, because the circuit closes through one jaw, the caliper's steel and the other jaw.

## Parts (prices from the makers' or sellers' pages, Nov 2026)

| Part | Use | Part no. | Price | Source |
|---|---|---|---|---|
| SingleTact S8-10N force sensor | measures the squeeze (8 mm disc, head 0.30 to 0.35 mm thick, tail 0.20 mm, 10 N full scale; maker's figures: resolution under 0.2%, repeatability under 1%, hysteresis under 4% of full scale, i.e. 0.02, 0.10 and 0.40 N; drift 2% in 1 min at 50% load) | S8-10N | $37.50 | https://www.singletact.com/micro-force-sensors/standard-sensors/p/s8-10n |
| SingleTact Standard Electronics board | reads the sensor, I2C or analog out | (see page) | $43.95 | https://www.singletact.com/micro-force-sensor/singletact-electronics/p/electronics-board |
| Arduino Nano | reads the board over I2C and prints the force over USB | A000005 | $17.99 to $25.70 | https://store-usa.arduino.cc/products/arduino-nano |
| 6061-T6511 aluminium round bar, 1 in x 12 in | the two half-sleeves | (any 1 in 6061 bar) | about $7 | e.g. https://www.zoro.com/zoro-select-aluminum-rod-12-in-l-1-in-od-1090120/i/G602645002 |
| Digital caliper, 0 to 150 mm, stainless steel | the caliper the robot uses | iGaging 100-700-06 (or Mitutoyo 500-196-30) | $83.95 ($169.28) | https://penntoolco.com/igaging-6-ip54-digital-caliper-absolute-origin-100-700-06 |
| Multimeter with continuity beep | the both-jaws check | SparkFun TOL-12966 | $21.50 | https://www.sparkfun.com/digital-multimeter-basic.html |
| Calibration weights, 5 g to 200 g | calibration over 0.05 to 2 N | any 1 to 200 g set (class M1 or better) | $20 to $350 by class | e.g. Adam Equipment 1-200 g M1 set |
| M2 x 16 nylon pan-head screws (2), 0.5 mm nylon washers 5 mm OD (4), M2 x 4 brass screws (2), two M2 ring terminals, thin wire | assembly and the tester's leads | | under $10 | any hardware supplier |
| Dial test indicator, 0.002 mm divisions, with a magnetic stand | the assembled-diameter checks (step 6) | any | about $60 to $150 | any metrology supplier |

Total, excluding the caliper and weights: about $130.

## Construction (see `force_pin.scad` for the drawing)

1. **Turn the bar.** Turn 40 mm of the 1 in bar to 23.40 mm (±0.01 mm) and face both ends square.
2. **Split it.** Cut it lengthwise through its axis. Then face-mill each half's flat until the flat is 0.30 mm short
   of the axis: each half is 11.40 mm from flat to the top of its arc. With the two flats 0.60 mm apart, the outer
   circle is whole again, 23.40 mm across the split (the jaws measure across it).
3. **Pocket and boss: the sensor is the only thing between the halves.** Measure the sensor's thickness t with a
   micrometer across its 8 mm head (0.30 mm on SingleTact's product page, 0.35 mm on its mechanical drawing). In half
   A's flat, at mid-length, mill an 8.5 mm diameter pocket 0.15 mm deep, plus a 4.0 mm wide, 0.5 mm deep channel from
   the pocket to the top end for the sensor's flat tail (the drawing gives the tail 0.20 mm thick and up to 3.5 mm
   wide): with the 0.60 mm gap the tail has 1.1 mm of depth and 0.25 mm a side, so it is never pinched. In half B's
   flat, leave an 8.0 mm diameter boss standing 0.60 - (t - 0.15) mm proud (0.45 mm for t = 0.30, 0.40 mm for
   t = 0.35), plus 0.02 mm to lap off in step 6. Assembled cross-section across the split, from half A's flat downward:

       half A flat ............ z =  0.00   pocket floor at z = +0.15 (inside half A)
       sensor ................. z = +0.15 to -0.15   (drawn for t = 0.30, seated on the pocket floor)
       boss face (half B) ..... z = -0.15   boss 0.45 tall, so half B's flat is at z = -0.60
       gap between the flats .. 0.60 mm everywhere else; outer circle 11.40 + 0.60 + 11.40 = 23.40 mm

   So the halves bear on each other only through the sensor, with no free travel: the jaws start loading the sensor
   at the pin's full 23.40 mm diameter. There is no spacer or other contact across the gap that could carry part of
   the squeeze.
4. **Hold the halves together loosely.** Two M2 x 16 nylon screws, 8 mm from each end and 5 mm off the axis, pass
   through 3.0 mm clearance holes in half B into 8 mm deep M2 threads in half A. Each head sits on a 5.5 mm flat seat
   milled 8.0 mm above half B's flat (so the washers bear on metal all round) with two 0.5 mm nylon washers: that
   leaves 16 - 8.0 - 0.6 - 1.0 = 6.4 mm of thread engaged in half A. Run each screw in until its head just touches,
   then back it off a quarter turn (0.1 mm of free play at the 0.4 mm pitch): the screws keep the halves together but
   cannot clamp, so they carry none of the squeeze. Check: with the pin on its side (calibration step 1), a 20 g
   weight reads the same within 0.005 N with the screws in and with them out.
5. **Leads for the continuity tester.** Tap an M2 hole 6 mm deep into the top end of each half, 5 mm off the axis on
   the opposite side from that half's screws and clear of the sensor's channel, and fix a ring terminal with a brass
   screw; run a thin wire from each. The halves touch each other nowhere else (the gap, the nylon screws and the
   sensor's polyimide insulate them). Check with the meter: no beep between the two leads.
6. **Diameter checks (acceptance).** Stand the assembled pin on a surface plate with the split vertical and a dial
   test indicator on the outside of half B, opposite the sensor, its contact force included in the reading by zeroing
   the force with the indicator touching.
   - Zero load: the diameter across the split, measured with the caliper used in the trials closed gently, is
     23.40 mm within 0.01 mm; if it is over, lap the boss face and reassemble.
   - Loaded: hang or press 5 g, 50 g and 200 g onto half B over the sensor (on its side, as in calibration step 1)
     and record the indicator: the pin is accepted if half B moves under 0.005 mm between 0.05 N and 2 N, so the
     sensor's compression changes the diameter by under a tenth of the 0.05 mm tier. Record the reading in the trial
     log.
7. **Mass.** The finished pin is about 45 g, as the simulated one (46 g). Cable mass is negligible if the sensor's tail
   and the two leads leave from the top end and run loosely up and away.

## Wiring

- Sensor tail to the SingleTact Standard Electronics board's sensor connector.
- Board to the Arduino Nano: power and ground, and I2C (SDA to A4, SCL to A5), following SingleTact's pinout for the
  board. Read it with SingleTact's own Arduino example sketch (from SingleTact's site), which prints the force over
  USB; a laptop's serial monitor shows it at 10 readings a second or more.
- The two half-sleeve leads go to the multimeter's probes, set to continuity (beep).

## Calibration and measured uncertainty over the scored range (0.05 N to 2 N)

The task scores at 0.05 N (both jaws loaded), 0.3 N and 2 N. The maker's figures are upper limits, not this pin's
performance (hysteresis up to 0.40 N would swamp the 0.3 N limit), so each built pin's own uncertainty is measured
here, and the trial uses it as a guard band. A jaw squeeze only loads the sensor (the jaws close, then hold), so the
pin is calibrated and used on its loading branch.

1. Lay the pin on its side on a flat surface with the split horizontal, half A underneath, so a weight on half B
   loads the sensor exactly as a jaw squeeze does. Put a light flat cradle (about 5 g, any flat plastic piece) on half
   B over the sensor and zero the reading with the cradle on.
2. **Loading runs.** Add weights onto the cradle in this order: 5, 10, 20, 30, 50, 100, 200 and 250 g (0.049 to
   2.45 N), recording the reading 3 s after each. Take them all off, re-zero, and repeat: five runs in all.
3. **Fit.** Fit one curve to all loading readings (a quadratic through zero, by least squares), so the sensor's
   non-linearity is calibrated out.
4. **Hold.** Re-zero, put 30 g on (0.29 N) and record the reading every 10 s for 2 minutes (a whole trial); then the
   same with 200 g (1.96 N).
5. **The pin's uncertainty U at each scored force** (0.05, 0.3 and 2 N): the largest difference between a fitted
   reading and the applied force, over all five loading runs at that force (or the two weights either side of it),
   plus the largest change over the 2-minute hold at that force (step 4; at 0.05 N use the 5 g run's spread).
   Record U in the trial log.
6. **Acceptance.** The pin is accepted if U is at most 0.03 N at 0.05 N, 0.05 N at 0.3 N and 0.15 N at 2 N. If it is
   not, rebuild it with the boss lapped flat and the sensor re-seated, or use a sensor with stated figures at this
   range (for example FUTEK LLB130, 9.5 mm button, 10 N: non-repeatability 0.01 N, hysteresis 0.05 N; about EUR 970).
7. **Guard bands in a trial.** A measurement counts as both jaws loaded only if the reading is at least 0.05 N + U(0.05)
   and the continuity tester beeps; it meets the 0.3 N limit only if the reading is at most 0.3 N - U(0.3), and the
   2 N limit only if at most 2 N - U(2). A reading inside a guard band scores as if the limit were missed. So an
   uncertain pin can only lower a score, never raise it.
8. **Before each trial.** Zero the reading with no load, and check one weight (30 g): if it reads off its fitted
   value by more than U(0.3), recalibrate.

## Checking the caliper closes the circuit through both jaws

The continuity check needs the caliper's two jaws joined electrically through its beam and slider, which is not
stated by the makers.

1. With the caliper open about 10 mm, touch one meter probe to the fixed jaw's measuring face and the other to the
   sliding jaw's measuring face. It must beep. Repeat at 25 mm and 50 mm, and while sliding the jaw slowly.
2. With the meter on the two half-sleeve leads, close the caliper on the pin: it must beep only when both jaws touch,
   and stop when either jaw is lifted off. Lift each jaw in turn to check.
3. If step 1 does not beep at every opening, use another stainless caliper that passes it.

## In a trial

Zero the force reading (calibration step 8), set the pin upright at its marked spot with the split vertical and square to the line between
the jaws (so the jaws close across the split, as they measure the diameter), and start the trial. The measurement counts when the photographed display, the force reading and the
beep agree: both jaws touching (beep) with at least 0.05 N, and the force at that moment for the 0.3 N and 2 N limits,
each with the pin's guard band (calibration step 7).
