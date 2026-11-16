# Mount the n-type (100) wafer face down on a dicing-tape mounter

Four unlabelled 100 mm silicon wafers lie pattern side up in open single-wafer carriers. They look alike: only their
flats tell them apart (SEMI convention, seen from the pattern side). Every wafer has a primary flat; p-type (111) has
no second flat, n-type (111) has one 45 degrees counterclockwise from the primary, p-type (100) one 90 degrees
counterclockwise, and n-type (100) one opposite it. Beside them stands a manual tape-frame mounter: a chuck plate with
centring and alignment grooves and a tweezer recess, a steel film frame already on its three locating pins, and a WAFER
VACUUM toggle. Two pairs of angled wafer tweezers stand in cups.

The task follows a standard tape-station SOP: the wafer goes on the chuck face down (the tape goes on its back),
centred and aligned, then the vacuum is switched on before taping. Here the primary flat must face the frame's flat side.
Taping is not part of the task.

## Files

- `task.py`: scene, scoring and scripted demo (yam_arena task file). The carriers, cups, mounter and toggle are fixed;
  the wafers and tweezers are loose. Each pair of tweezers is two halves, one on a sprung labkit slide, so squeezing their
  arms closes the blades. An `on_step` hook latches the violations (a fingertip at a wafer's pattern or rim; a wafer
  dropped) and checks that everything has been still for the last second.
- `labkit.py`: parts built from several shapes, with hinges and slides. It hooks `yam_arena.scene` and the embodiment
  from the task's import, so no change to yam_arena is needed.
- `arena_assets_fix.py`: imported by `labkit.py`; points Arena library assets at NVIDIA's production bucket. This task
  uses no library assets; every object is built in `task.py`.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (the Robocurve Brev launchable):

    inspect-robots run --embodiment yam_arena --policy dream \
        -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')" \
        --scorer task_score --max-steps 2400 --store-frames --log-dir logs

Scripted demo: the same command with `--policy yam_scripted`. The right tweezers take the wafer out of its carrier and
stand it on edge; the left tweezers take its top edge from behind and turn it over, so it hangs pattern side down from
blades that are above it, trim their aim by the wafer's seen position over the chuck and lay it on the chuck through
the recess. The left hand stands its tweezers back in their cup and switches the vacuum on; the right hand keeps its
tweezers, out of the way. A wafer cannot be turned face down in one pair of tweezers and still be laid down by them, so
the hand-over is needed (found with an IK search: one arm cannot turn the held wafer over on its own). The demo reads
only what the cameras see: the wafer's pose and the tweezers' tips. Run one trial per process.

The score is `task_score` in the log, computed by `score()` in `task.py`.

## Build the real rig

The simulated mounter follows a manual wafer/frame film mounter such as the UH114
(https://www.microworld.eu/manual-wafer-frame-film-mounter-uh114/, vacuum chuck from 2 to 12 inch). A low-cost build
that matches the scene (all sizes from `task.py`):

- Housing: any box 240 x 240 mm, 34 mm tall (plywood or printed).
- Chuck plate: 6 mm aluminium (6061) or printed PLA, 240 x 240 mm, on the housing so its top is 40 mm above the bench.
  Mark 0.8 mm lines flush with the surface (printed, painted or laser-marked, not cut, so they cannot leak vacuum):
  circles of radius 25.4, 38.1, 50.8 and 63.5 mm (2 to 5 inch) about the centre, two 140 mm straight lines crossing at
  it, and a 70 mm line along the 100 mm wafer's primary-flat chord. Cut a tweezer
  recess 20 mm wide and 4 mm deep, running from 30 to 78 mm out from the centre on its left (+y) side, as in the scene.
- Film frame: stainless or aluminium sheet 1.2 mm thick, 150 mm inside and 200 mm outside diameter, with one flat
  93 mm from the centre; or a standard steel film frame of that size. It sits on three 6 mm dowel pins, 6 mm tall, at
  103 mm from the centre, 120 degrees apart.
- Working vacuum chuck (required): the chuck must really hold the wafer when the lever is ON.
  - Chuck path, all inside unbroken land (the tweezer recess starts 30 mm out, so nothing may reach it): a 1 mm hole
    through the chuck plate at its centre, a 0.5 mm deep, 1 mm wide vacuum groove ring of 20 mm radius on the top
    face, and one 0.5 mm deep radial groove from the hole to that ring, pointing away from the recess (-y). The wafer
    lies on it pattern side down, as on a real tape-frame mounter's chuck, and covers the whole circuit. Under the
    plate, a 1/4 in. barb fitting sealed over the hole (thread sealant or an O-ring). A printed plate must be sealed
    (epoxy coat on top) so it does not leak through its layers.
  - Pump and tubing: a small oil-free diaphragm vacuum pump reaching at least 60 kPa below atmosphere, through 6 mm
    (1/4 in.) vacuum tubing with hose clamps at every barb.
  - WAFER VACUUM valve: a vacuum-rated quarter-inch three-port, two-position lever valve (an L-port three-way ball
    valve), mounted on a 22 x 16 mm block at the housing's corner. Common port to the chuck's barb; one port to the
    pump; the third port open to the room through a small inline air filter. ON (lever 24 mm long, tilted forward,
    green mark in front): pump to chuck, the wafer is held. OFF (tilted 24 degrees back): pump isolated and the chuck
    vented to the room, so the wafer comes free. The scorer reads the lever's position (ON past 11 degrees forward);
    on the rig, ON must also mean the wafer is held and OFF that it is released.
  - Acceptance check before each session (the design above has not been built and tested by us; this check is what
    qualifies a rig): pump on, lever ON, a 100 mm wafer centred on the chuck; tilt the whole mounter until the chuck
    faces sideways (vertical): the wafer must stay put for 10 s. Lever OFF: within 2 s it must lift off freely with
    tweezers. If it slides with the lever ON, find and seal the leak; if it sticks with the lever OFF, check the vent
    port and filter. Only run trials on a rig that passes both.

Pattern contact in a real trial: judged from the trial video and by looking at each wafer's pattern side afterwards
under a low-angle desk lamp for tweezer marks or scratches more than 10 mm inside the rim. The simulation's 0.3 mm
(tweezer) and 4 mm (fingertip) distances in `task.py` stand in for that contact.

## Runtime and seeds

- 2,400 control steps (2 minutes of simulated time) per trial.
- One fixed starting layout: run with `-E object_jitter=0`. No other randomness.
