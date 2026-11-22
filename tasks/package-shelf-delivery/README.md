# Photograph a delivered parcel on its unit shelf

A courier sets a parcel on the unit 12 parcel shelf, then takes the proof-of-delivery photo with a hand-held phone: one
hand aims the phone at the parcel and the unit's plaque, the other presses the phone's volume-up button (the shutter). The
phone goes back in its dock.

## Files

- `task.py`: scene, scorer and scripted demo (yam_arena + labkit).
- `labkit.py`, `arena_assets_fix.py`: shared helpers.
- `cart_scene_fix.py`: replaces the simulator station's table with the courier cart's deck (same height, ends short of the shelves).
- `test_scorer.py`: teleport tests of every score branch and shortcut (no policy).
- `render_proof.py`: draws the proof photo from the record saved at the shutter.

## Run and score a trial

    inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py -E object_jitter=0 -E control_hz=20 \
        --instruction "<instruction from task.json>" --scorer task_score --max-steps 1200 --store-frames --log-dir logs

The score is `task_score` in `logs/*.json`. The scripted demo is `--policy yam_scripted` (about 30 s, 600 steps).
No randomness: one fixed layout (`object_jitter=0`), no seeds.

Runtime per trial: about 4 to 6 minutes on one L40S, mostly simulator start-up.

## Proof photo

Each time the volume-up button clicks while a hand grips the phone, the scorer renders the phone camera's picture
(`render_proof.cast`: pinhole, portrait 480 x 640, 69.4 x 54.4 deg) of the cart, shelves, wall, plaques, parcel and both
arms, saves it as `proof_step<N>.png` (in `PKG_PROOF_DIR`, default `./proof_photos`) and judges the photo from those pixels:
parcel and unit-12 plaque inside a 5 % margin and unobstructed, every digit stroke visible, the '1' at least 20 px tall.
With `PKG_PROOF_DIR` set, a JSON record of the camera and boxes is saved too; `python render_proof.py <json> out.png`
redraws it. The picture is a render of the modelled boxes, not a camera sensor image.

## Reproducing on real hardware

Everything is stock, off-the-shelf equipment; no fabrication or special facility. Prices are from product pages opened
on 2026-11-22; lines marked "price not verified" could not be opened (store blocked automated access) and are not estimated.

| Item | Stock product | Price (page opened 2026-11-22) |
|---|---|---|
| Courier cart (deck the arms stand on) | No purchase needed: use the robot's existing table, parked with its front edge 30 mm from the shelf fronts, its top at about 0.75 m (the scene's deck height). If the table has no wheels, it simply stands in for the parked cart. Cheapest cart checked, for the parcel and phone cradle only: Uline H-12681BL 2-shelf compact utility cart, 27 x 18 x 33 in, two locking casters | $95 to $105 each, https://www.uline.com/Product/Detail/H-12681BL/Utility-Carts/Uline-2-Shelf-Compact-Utility-Cart-with-Flat-Shelves-27-x-18-x-33-Black |
| Parcel | Uline S-20491 corrugated box, 6 x 4 x 2 in inside (6 3/8 x 4 3/8 x 2 5/8 in outside), 200 lb test, packed to 0.8 kg, packing tape, a printed fictional label | $0.50 per box (bundle of 25), https://www.uline.com/Product/Detail/S-20491/Corrugated-Boxes-200-Test/6-x-4-x-2-Corrugated-Boxes |
| Three shelf boards | 1 x 8 in common pine or softwood board, cut to 400 mm, three pieces (one 4 ft board gives three) | price not verified (Home Depot item 204852097, page blocked) |
| Six shelf brackets | Everbilt 8 x 10 in white steel shelf bracket, screwed to a wall or a plywood backer | price not verified (Home Depot item 202034282, page blocked) |
| Unit numbers | 2 in self-adhesive vinyl numbers "1", "1", "1", "2", "1", "3" on white card plaques, 100 x 60 mm | price not verified (Home Depot item 100147740, page blocked) |
| Phone | Any 6.1 in phone already on hand, about 148 x 72 x 8 mm, whose volume button takes a photo; a cheaper or older phone works as well, no purchase needed | none if a phone is on hand |
| Phone cradle | any desk phone stand with a slot about 20 mm wide, screwed or taped to the cart | price not verified |

The only verified price is the cartons: $0.50 each, $12.50 for a bundle of 25 (cartons only). With the robot's own table and a phone on hand nothing else from this list needs buying except the shelving; add $95 to $105 for the optional compact cart. Boards, brackets, numbers and the cradle are extra and not priced here. Consumed per trial: nothing (the same carton is reused). Scoring tools: the
phone's own saved photo, a tape measure for the shelf placement, and a look at the cradle. Estimates, not measured:
about 1 hour to mount the brackets and shelves and lay out the cart; about 1 minute to reset between trials (carton back
on the cart, phone back in its cradle).

## Why it matters

Light truck drivers held about 1.1 million US jobs in 2025 (BLS Occupational Outlook Handbook,
https://www.bls.gov/ooh/transportation-and-material-moving/delivery-truck-drivers-and-driver-sales-workers.htm). Carriers
photograph parcels where they leave them: FedEx picture proof of delivery, available for US and Canada residential
deliveries (https://www.fedex.com/en-us/tracking/picture-proof-delivery.html).

## Tests

    python test_scorer.py      # needs the yam_arena environment; prints ALL_PASS and saves the control photos in test_photos/
