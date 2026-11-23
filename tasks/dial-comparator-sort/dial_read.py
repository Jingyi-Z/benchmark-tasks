"""Read the dial indicator from a camera image (numpy RGB array): the reading in 0.01 mm divisions, or None.
1. face: the white, unsaturated pixels of the dial face (the bench is beige, so saturation separates them); the
   largest blob near the image centre; its second moments give the face ellipse, which is rectified to a unit circle.
2. zero: the two red tolerance markers sit on the bezel at +-2 divisions about its 0 mark; their circular mean angle
   in the rectified face is the 0 direction.
3. needle: dark pixels between the hub and the numerals (rectified radius 0.15 to 0.48); their circular mean angle.
Reading = (needle - zero) / 3.6 deg, signed so a taller part reads +. None when the face, both markers or the needle
are not found clearly (the caller then holds and looks again)."""
import math

import numpy as np

SIGN, OFFSET = -1.0, -1.9                    # calibrated on 9 + 5 frames with known readings (224 and 640 x 352 views)


def _blob(mask, seed):
    """the 4-connected blob of mask that contains (or is nearest to) seed; plain numpy flood fill"""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    k = int(np.argmin((ys - seed[0]) ** 2 + (xs - seed[1]) ** 2))
    start = (int(ys[k]), int(xs[k]))
    seen = np.zeros_like(mask, dtype=bool)
    stack = [start]
    seen[start] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True
                stack.append((yy, xx))
    return seen


def read_dial(rgb, debug=None):
    a = np.asarray(rgb)[..., :3].astype(np.float32)
    h, w = a.shape[:2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    face = (mn > 150) & (mx - mn < 28)
    blob = _blob(face, (h * 0.5, w * 0.5))
    if blob is None or blob.sum() < 0.002 * h * w:
        return None
    ys, xs = np.nonzero(blob)
    mu = np.array([ys.mean(), xs.mean()])
    C = np.cov(np.stack([ys - mu[0], xs - mu[1]]))
    ev, V = np.linalg.eigh(C)
    if ev.min() <= 0:
        return None
    Wm = V @ np.diag(1.0 / np.sqrt(4.0 * ev)) @ V.T      # uniform disc: var = r^2 / 4 -> rectified radius 1
    yy, xx = np.mgrid[0:h, 0:w]
    P = np.stack([yy - mu[0], xx - mu[1]], axis=-1) @ Wm.T
    r = np.hypot(P[..., 0], P[..., 1])
    ang = np.arctan2(P[..., 1], -P[..., 0])              # 0 = image up, + = clockwise in the image

    red = (a[..., 0] > 140) & (a[..., 1] < 90) & (a[..., 2] < 90) & (r > 0.7) & (r < 1.45)
    if red.sum() < 4:
        return None
    # the two markers must both be seen: split the red pixels at the largest angular gap into two clusters,
    # each with at least 4 pixels and 15% of the red (one marker can sit in shadow), about 4 divisions (14.4 deg) apart; otherwise no read
    ra = np.sort(np.mod(ang[red], 2 * math.pi))
    gaps = np.diff(np.concatenate([ra, ra[:1] + 2 * math.pi]))
    order = np.argsort(gaps)[::-1]
    i1, i2 = sorted(order[:2])                           # the two largest gaps bound the two clusters
    c1 = ra[i1 + 1:i2 + 1]
    c2 = np.concatenate([ra[i2 + 1:], ra[:i1 + 1]])
    if min(len(c1), len(c2)) < max(4, 0.15 * len(ra)):
        return None
    m1 = math.atan2(np.sin(c1).mean(), np.cos(c1).mean())
    m2 = math.atan2(np.sin(c2).mean(), np.cos(c2).mean())
    sep = abs(math.degrees(np.angle(np.exp(1j * (m1 - m2)))))
    if not 8.0 <= sep <= 22.0:
        return None
    zero = math.atan2(math.sin(m1) + math.sin(m2), math.cos(m1) + math.cos(m2))
    dark = (mx < 80) & (r > 0.15) & (r < 0.48)
    if dark.sum() < 2:
        return None
    rel = np.angle(np.exp(1j * (ang[dark] - zero)))
    # the needle is one dark streak; take the densest 30-degree sector of dark pixels, then its mean
    hist, edges = np.histogram(rel, bins=24, range=(-math.pi, math.pi))
    k = int(np.argmax(hist))
    lo, hi = edges[max(k - 1, 0)], edges[min(k + 2, 24)]
    sel = rel[(rel >= lo) & (rel <= hi)]
    if len(sel) < 2 or hist[k] < 0.5 * len(rel) / 6:
        return None
    needle = math.atan2(np.sin(sel).mean(), np.cos(sel).mean())
    div = SIGN * math.degrees(needle) / 3.6 + OFFSET
    if debug is not None:
        debug.update(face_px=int(blob.sum()), red_px=int(red.sum()), dark_px=int(dark.sum()), zero_deg=math.degrees(zero),
                     needle_rel_deg=math.degrees(needle), marker_sep_deg=sep)
    return div
