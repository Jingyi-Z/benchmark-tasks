"""Teleport tests of the scorer (no policy). Loads task.py, sets part states and the watcher's record directly, and checks
every score branch, the image-based photo verdict and its controls. Run in the yam_arena environment; prints ALL_PASS."""
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task as T  # noqa: E402


class St:
    def __init__(self, pos, R=np.eye(3)):
        self.pos = np.asarray(pos, float); self._R = np.asarray(R, float)

    def rotation(self):
        return self._R

    def tilt_deg(self):
        return float(np.degrees(np.arccos(np.clip(self._R[2, 2], -1, 1))))


def Ry(a):
    a = math.radians(a); c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    a = math.radians(a); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


ON_SHELF = St((0.33, T.PKG_Y, T.SHELF_TOP))
ON_CART = St((T.PKG_X0, T.PKG_Y, T.TOP))
ON_11 = St((0.33, T.BAYS["11"], T.SHELF_TOP))
ON_13 = St((0.33, T.BAYS["13"], T.SHELF_TOP))
DOCKED = St((T.DOCK[0], T.DOCK[1], T.TOP + T.PH[2] / 2 + 0.004))
DOCKED_TURNED = St((T.DOCK[0], T.DOCK[1], T.TOP + T.PH[2] / 2 + 0.004), Rz(180))       # other face forward
DOCKED_ENDUP = St((T.DOCK[0], T.DOCK[1], T.TOP + T.PH[2] / 2 + 0.004), Ry(180) @ Rz(180))  # other end down
OUT = St((0.15, 0.30, 0.87))


def photo(ok=True, pkg=ON_SHELF, why="ok", spoiled=False):
    return {"step": 900, "ok": ok, "why": why, "spoiled": spoiled, "png": None,
            "pkg_pos": np.asarray(pkg.pos, float).copy(), "pkg_R": pkg.rotation().copy()}


def state(pkg, ph, *, latch=None, photos=(), clear=0.2, still=True):
    T._reset_state()
    T._S.update(latch=latch, photos=list(photos), clear=clear)
    pts = np.vstack([T._corners(pkg), T._ph_corners(ph)])
    T._S["hist"] = [pts.copy() for _ in range(T.STILL_N)] if still else [pts + 0.002 * i for i in range(T.STILL_N)]
    return {"parcel": pkg, "phone": ph}


CASES = [
    ("start state", lambda: state(ON_CART, DOCKED), 0.0),
    ("goal: placed, valid photo, phone in its cradle, clear, still", lambda: state(ON_SHELF, DOCKED, photos=[photo()]), 1.0),
    ("phone back in the cradle facing the other way", lambda: state(ON_SHELF, DOCKED_TURNED, photos=[photo()]), 1.0),
    ("phone back in the cradle the other end down", lambda: state(ON_SHELF, DOCKED_ENDUP, photos=[photo()]), 1.0),
    ("placed, no photo", lambda: state(ON_SHELF, DOCKED), 0.5),
    ("placed, photo invalid (cut off / hidden / illegible)", lambda: state(ON_SHELF, DOCKED, photos=[photo(False, why="plaque_hidden")]), 0.5),
    ("placed, valid photo, phone left out of its cradle", lambda: state(ON_SHELF, OUT, photos=[photo()]), 0.5),
    ("photo before placing (recorded invalid)", lambda: state(ON_SHELF, DOCKED, photos=[photo(False, why="parcel_not_placed_at_shutter")]), 0.5),
    ("parcel moved 8 mm after the photo", lambda: state(St((0.338, T.PKG_Y, T.SHELF_TOP)), DOCKED, photos=[photo()]), 0.5),
    ("parcel moved after the photo and put back (spoiled by tracking)", lambda: state(ON_SHELF, DOCKED, photos=[photo(spoiled=True)]), 0.5),
    ("valid photo, then a bad retake: the valid one is kept", lambda: state(ON_SHELF, DOCKED, photos=[photo(), photo(False, why="parcel_hidden:0.40")]), 1.0),
    ("wrong unit: placed on shelf 11", lambda: state(ON_11, DOCKED, photos=[photo(pkg=ON_11)]), 0.0),
    ("wrong unit: placed on shelf 13", lambda: state(ON_13, DOCKED, photos=[photo(pkg=ON_13)]), 0.0),
    ("parcel tilted 20 deg on the shelf (label not up)", lambda: state(St((0.33, T.PKG_Y, T.SHELF_TOP + 0.02), Ry(20)), DOCKED, photos=[photo()]), 0.0),
    ("parcel overhanging the shelf front", lambda: state(St((0.27, T.PKG_Y, T.SHELF_TOP)), DOCKED, photos=[photo()]), 0.0),
    ("parcel left on the cart", lambda: state(ON_CART, DOCKED), 0.0),
    ("dropped from > 5 cm (latched)", lambda: state(ON_SHELF, DOCKED, photos=[photo()], latch="parcel_dropped_80mm"), 0.0),
    ("phone fell toward the floor (latched)", lambda: state(ON_SHELF, DOCKED, photos=[photo()], latch="fell_toward_the_floor"), 0.0),
    ("phone dropped (latched)", lambda: state(ON_SHELF, DOCKED, photos=[photo()], latch="phone_dropped"), 0.0),
    ("hand still within 80 mm", lambda: state(ON_SHELF, DOCKED, photos=[photo()], clear=0.05), 0.0),
    ("not still over the last 1 s", lambda: state(ON_SHELF, DOCKED, photos=[photo()], still=False), 0.0),
]

bad = 0
for name, make, want in CASES:
    got = T.score(make())
    ok = abs(got - want) < 1e-9
    bad += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {got:.2f} (want {want:.2f})  {name}")


def check(name, ok, why, want_ok, want_why=""):
    global bad
    good = (ok == want_ok) and (want_why in why)
    bad += 0 if good else 1
    print(f"{'PASS' if good else 'FAIL'}  judge_photo {why:28s} {name}")


# image verdicts at the demo's photo pose, judged from the rendered pixels
pos_p, Rph = T.photo_phone_pose()
cam_ph = St(pos_p, Rph)
out = HERE / "test_photos"; out.mkdir(exist_ok=True)
ok, why = T.judge_photo(ON_SHELF, cam_ph, (), str(out / "demo_pose.png"))
check("demo photo pose: whole parcel, plaque 12 readable", ok, why, True, "ok:")


def finger_links(point, axis):
    """a fake 'gripper' link whose finger box (110 x 45 x 110 mm, 45 mm down its z) sits at point, z along axis."""
    z = np.asarray(axis, float); z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    g = np.eye(4); g[:3, :3] = np.column_stack([x, np.cross(z, x), z]); g[:3, 3] = np.asarray(point) - z * 0.045
    return [{"gripper": g}]


cam = T.camera_view(cam_ph)[0]
# a gripper across the middle of the digits (between the camera and the plaque centre)
mid = cam + 0.75 * (T.LABEL_C - cam)
ok, why = T.judge_photo(ON_SHELF, cam_ph, finger_links(mid, T.LABEL_C - cam), str(out / "finger_on_digits.png"))
check("a gripper across the digits", ok, why, False, "hidden")
# the parcel shot from too low: the shelf board hides its lower part
cam_low = np.array([0.05, 0.0, 0.76]); tgt = (np.array([0.33, T.PKG_Y, 0.85]) * 8 + T.LABEL_C * 4) / 12
f = (tgt - cam_low) / np.linalg.norm(tgt - cam_low); zl = np.array([0, 0, 1.0]) - f * f[2]; zl /= np.linalg.norm(zl)
R4 = np.column_stack([f, np.cross(zl, f), zl])
ok, why = T.judge_photo(ON_SHELF, St(cam_low - R4 @ T.CAM_LOCAL, R4), (), str(out / "low_view.png"))
check("camera just above the deck (parcel hidden or plaque cut)", ok, why, False, "")
# the camera aimed at bay 13 instead: plaque 12 leaves the frame
tgt13 = np.array([T.WALL_X, T.BAYS["13"] + T.LABEL_Y, T.PLAQUE_Z])
f = (tgt13 - cam) / np.linalg.norm(tgt13 - cam); zl = np.array([0, 0, 1.0]) - f * f[2]; zl /= np.linalg.norm(zl)
R5 = np.column_stack([f, np.cross(zl, f), zl])
ok, why = T.judge_photo(ON_SHELF, St(cam - R5 @ T.CAM_LOCAL, R5), (), str(out / "aimed_at_13.png"))
check("camera aimed at unit 13's plaque", ok, why, False, "")
# pointing 60 deg down: nothing framed
ok, why = T.judge_photo(ON_SHELF, St(pos_p, Rph @ Ry(60)), (), None)
check("phone pointing 60 deg down", ok, why, False, "")


# -- matched wrong-number controls: the parcel stays whole in frame; only the unit identification changes
def scored(pkg, ph_cam, links, png, want, name):
    """judge a photo, record it as the watcher would, then score the end state (phone docked, still, clear)."""
    global bad
    okp, whyp = T.judge_photo(pkg, ph_cam, links, str(out / png))
    rec = photo(okp, pkg=pkg, why=whyp)
    got = T.score(state(pkg, DOCKED, photos=[rec]))
    good = abs(got - want) < 1e-9
    bad += 0 if good else 1
    print(f"{'PASS' if good else 'FAIL'}  {got:.2f} (want {want:.2f})  photo {whyp:24s} {name}  [{png}]")


def blocker_links(c, size_hint=None):
    """a fake gripper whose 110 x 45 x 110 mm finger box sits centred at c, its z axis pointing from the camera."""
    z = c - cam; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    g = np.eye(4); g[:3, :3] = np.column_stack([x, np.cross(z, x), z]); g[:3, 3] = c - z * 0.045
    return [{"gripper": g}]


# (1) positive, for comparison: parcel on 12, plaque 12 fully readable
scored(ON_SHELF, cam_ph, (), "ctrl_positive_12.png", 1.0, "parcel on 12, plaque 12 readable")
# (2) parcel on 12 and fully framed, but plaque 12 covered (a hand in front of it, 70 % of the way from the lens)
cover = cam + 0.70 * (T.LABEL_C - cam)
scored(ON_SHELF, cam_ph, blocker_links(cover), "ctrl_12_covered.png", 0.5, "parcel on 12 whole, plaque 12 covered")
# (3) parcel on 12 and fully framed, only the '2' lower-left stroke hidden (12 no longer distinguishable from 13)
e_seg = [c for c, sz, rgb in T.SHELF_BOXES if tuple(rgb) == T.BLACK][4]          # strokes: '1', a, b, g, e, d
Re = np.eye(3); tiny = (np.asarray(e_seg) + (cam - np.asarray(e_seg)) * 0.04)
blk = [(tiny, Re, np.array([0.004, 0.006, 0.014]), (0.1, 0.1, 0.1), T.ID_ROBOT)]
orig = T._robot_boxes
T._robot_boxes = lambda links: blk if links == "mask_e" else orig(links)
scored(ON_SHELF, cam_ph, "mask_e", "ctrl_12_stroke_e_hidden.png", 0.5, "parcel on 12 whole, the '2' lower-left stroke hidden")
T._robot_boxes = orig
# (4) mirror: parcel delivered to 13, photographed well framed with plaque 13 -> wrong delivery, 0
p13 = np.array([0.33, T.BAYS["13"], 0.855]); tgt = (p13 * 8 + np.array([T.WALL_X, T.BAYS["13"] + T.LABEL_Y, T.PLAQUE_Z]) * 4) / 12
cam13 = cam + (p13 - np.array([0.33, T.PKG_Y, 0.855]))
f = (tgt - cam13) / np.linalg.norm(tgt - cam13); zl = np.array([0, 0, 1.0]) - f * f[2]; zl /= np.linalg.norm(zl)
R6 = np.column_stack([f, np.cross(zl, f), zl])
ON_13b = St((0.33, T.BAYS["13"] + T.PKG_Y, T.SHELF_TOP))
scored(ON_13b, St(cam13 - R6 @ T.CAM_LOCAL, R6), (), "ctrl_on_13_photo_13.png", 0.0, "parcel on 13, framed with plaque 13")
# (5) mirror: parcel delivered to 11, photographed well framed with plaque 11 -> 0
p11 = np.array([0.33, T.BAYS["11"] + T.PKG_Y, 0.855]); tgt = (p11 * 8 + np.array([T.WALL_X, T.BAYS["11"] + T.LABEL_Y, T.PLAQUE_Z]) * 4) / 12
cam11 = cam + (p11 - np.array([0.33, T.PKG_Y, 0.855]))
f = (tgt - cam11) / np.linalg.norm(tgt - cam11); zl = np.array([0, 0, 1.0]) - f * f[2]; zl /= np.linalg.norm(zl)
R7 = np.column_stack([f, np.cross(zl, f), zl])
ON_11b = St((0.33, T.BAYS["11"] + T.PKG_Y, T.SHELF_TOP))
scored(ON_11b, St(cam11 - R7 @ T.CAM_LOCAL, R7), (), "ctrl_on_11_photo_11.png", 0.0, "parcel on 11, framed with plaque 11")
okx, whyx = T.judge_photo(ON_11b, St(cam11 - R7 @ T.CAM_LOCAL, R7), (), None)
check("parcel on 11 with 12 also in frame: the picture ties the parcel to 11", okx, whyx, False, "nearer_unit_11")


# -- shutter controls: a fresh press while gripped and high, and a saved image file
import os  # noqa: E402


def run_presses(seq, proof_dir):
    """feed (pressed, gripped, high) steps to the shutter; return the photos it took."""
    os.environ["PKG_PROOF_DIR"] = proof_dir
    T._reset_state()
    for i, (pr, gr, hi) in enumerate(seq):
        T._S["n"] = 1000 + i
        T.shutter(pr, "L" if gr else None, hi, ON_SHELF, cam_ph, (), 0.2, True)
    return list(T._S["photos"])


def shutter_case(name, seq, proof_dir, want_score, want_n, want_why=""):
    global bad
    ph = run_presses(seq, proof_dir)
    got = T.score(state(ON_SHELF, DOCKED, photos=ph))
    why = ph[-1]["why"] if ph else "no photo"
    good = abs(got - want_score) < 1e-9 and len(ph) == want_n and want_why in why
    bad += 0 if good else 1
    print(f"{'PASS' if good else 'FAIL'}  {got:.2f} (want {want_score:.2f})  photos={len(ph)} ({why})  {name}")


ok_dir = str(out / "shutter_ok")
shutter_case("fresh press while gripped and 30 mm up: photo saved and valid",
             [(False, True, True), (True, True, True), (True, True, True)], ok_dir, 1.0, 1, "ok:")
shutter_case("button pressed below 30 mm and held while the phone is lifted through 30 mm: no photo",
             [(True, True, False), (True, True, False), (True, True, True), (True, True, True), (True, True, True)], ok_dir, 0.5, 0)
shutter_case("button pressed before the grip and held while gripped: no photo",
             [(True, False, True), (True, True, True), (True, True, True)], ok_dir, 0.5, 0)
blocker = out / "not_a_folder"
blocker.write_text("x")
shutter_case("fresh valid press but the photo folder cannot be written: photo_not_saved",
             [(False, True, True), (True, True, True)], str(blocker / "photos"), 0.5, 1, "photo_not_saved")
# a picture of the same name left from an earlier trial, and this shutter's write fails: not saved
import render_proof  # noqa: E402
stale_dir = out / "shutter_stale"; stale_dir.mkdir(exist_ok=True)
(stale_dir / "proof_step1001.png").write_text("old file")
_real_write = render_proof.write_png


def _failing_write(path, img):
    raise OSError("disk full")


render_proof.write_png = _failing_write
shutter_case("fresh valid press, write fails, an old file of that name exists: photo_not_saved",
             [(False, True, True), (True, True, True)], str(stale_dir), 0.5, 1, "photo_not_saved")
render_proof.write_png = _real_write
os.environ.pop("PKG_PROOF_DIR", None)
print("ALL_PASS" if bad == 0 else f"{bad} FAILED")
sys.exit(1 if bad else 0)
