"""cart_scene_fix: the station's table becomes the courier cart's deck.

The scene is an apartment corridor: the two arms, the phone dock and the parcel ride on a courier cart parked in
front of wall-mounted parcel shelves. yam_arena's station builds one static box named "table" (1.10 x 1.30 m, top
0.7475 m) that would run under the shelves. Importing this module from a task file wraps
``yam_arena.scene._station_room`` so that box becomes the cart deck instead: the same top height and the same
static-box class, but only x -0.32..0.21 m, y -0.45..0.45 m, dark grey. Everything the station places on the
table (arm bases, top-camera stand foot) still stands on the deck. Floor, lights and cameras are unchanged. The cart's
frame, wheels and handle are render-only meshes in the task file. Changes nothing on disk; no Isaac imports of its own.

    import cart_scene_fix  # noqa: F401   (before yam_arena.scene is imported)
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys

TARGET = "yam_arena.scene"
DECK_X = (-0.32, 0.21)
DECK_Y = (-0.45, 0.45)
DECK_THICK = 0.04
DECK_COLOR = (0.16, 0.17, 0.19)
_done = False


def _patch(scene) -> None:
    global _done
    if _done or not hasattr(scene, "_station_room"):
        return
    inner = scene._station_room

    def _station_room():
        out = []
        for a in inner():
            if getattr(a, "name", None) == "table":
                top = scene.station.TABLE_TOP
                size = (DECK_X[1] - DECK_X[0], DECK_Y[1] - DECK_Y[0], DECK_THICK)
                centre = ((DECK_X[0] + DECK_X[1]) / 2, (DECK_Y[0] + DECK_Y[1]) / 2, top - DECK_THICK / 2)
                a = scene._box("table", size, centre, DECK_COLOR, cls=scene._Table)
            out.append(a)
        return out

    scene._station_room = _station_room
    _done = True
    print("[cart_scene_fix] station table replaced by the courier cart deck", file=sys.stderr)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
