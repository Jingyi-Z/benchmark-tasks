"""The phone's proof photograph: a pinhole ray cast of oriented boxes (portrait 480 x 640, 69.4 x 54.4 deg), returning the
colour image and a per-pixel object id. task.py calls cast() at the shutter and judges the photo from these pixels;
`python render_proof.py proof_stepN.json out.png` redraws a saved record."""
import json
import struct
import sys
import zlib

import numpy as np

W, H = 480, 640


def write_png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def rays(f, up, r, vfov, hfov, w=W, h=H):
    tv, th = np.tan(vfov / 2), np.tan(hfov / 2)
    ys, xs = np.mgrid[0:h, 0:w]
    v = (1 - 2 * (ys + 0.5) / h) * tv
    u = (2 * (xs + 0.5) / w - 1) * th
    d = f[None, None] + u[..., None] * r[None, None] + v[..., None] * up[None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def cast(cam, f, up, r, boxes, vfov, hfov, w=W, h=H):
    """boxes: (centre (3,), axes (3, 3) columns, half sizes (3,), rgb, id). Returns (uint8 image (h, w, 3), int ids (h, w))."""
    cam, f, up, r = (np.asarray(a, float) for a in (cam, f, up, r))
    d = rays(f, up, r, vfov, hfov, w, h)
    depth = np.full((h, w), np.inf)
    img = np.empty((h, w, 3)); img[:] = (0.78, 0.78, 0.76)
    ids = np.zeros((h, w), np.int32)
    light = np.array([0.3, 0.4, 0.85]); light /= np.linalg.norm(light)
    for c, ax, hs, rgb, oid in boxes:
        c, ax, hs = np.asarray(c, float), np.asarray(ax, float), np.asarray(hs, float)
        tmin = np.full((h, w), -np.inf); tmax = np.full((h, w), np.inf); nidx = np.zeros((h, w), np.int8)
        for i in range(3):
            a = ax[:, i]; e = float((c - cam) @ a); fd = d @ a
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (e - hs[i]) / fd; t2 = (e + hs[i]) / fd
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            par = np.abs(fd) < 1e-12
            inside = abs(e) <= hs[i]
            lo = np.where(par, -np.inf if inside else np.inf, lo); hi = np.where(par, np.inf if inside else -np.inf, hi)
            nidx = np.where(lo > tmin, i, nidx)
            tmin = np.maximum(tmin, lo); tmax = np.minimum(tmax, hi)
        hit = (tmin <= tmax) & (tmin > 0.005) & (tmin < depth)
        if not hit.any():
            continue
        shade = 0.55 + 0.45 * np.abs(np.choose(nidx, [ax[:, i] @ light for i in range(3)]))
        img = np.where(hit[..., None], np.asarray(rgb, float)[None, None] * shade[..., None], img)
        ids = np.where(hit, oid, ids); depth = np.where(hit, tmin, depth)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), ids


def boxes_from_record(rec):
    out = []
    for b in rec["boxes"]:
        c = np.asarray(b["c"], float)                       # 8 corners, index = 4a + 2b + d over (-1, 1)^3
        e = [c[4] - c[0], c[2] - c[0], c[1] - c[0]]
        hs = np.array([np.linalg.norm(v) / 2 for v in e])
        ax = np.column_stack([v / (2 * h) if h > 0 else np.eye(3)[i] for i, (v, h) in enumerate(zip(e, hs))])
        out.append((c.mean(0), ax, hs, b["rgb"], b.get("id", 9)))
    return out


if __name__ == "__main__":
    rec = json.load(open(sys.argv[1]))
    img, _ = cast(rec["cam"], rec["f"], rec["up"], rec["r"], boxes_from_record(rec), np.radians(rec["vfov_deg"]), np.radians(rec["hfov_deg"]))
    write_png(sys.argv[2], img)
    print("wrote", sys.argv[2], "step", rec["step"], rec["why"])
