"""OUTPUT-ONLY review camera wrapper for the frozen kayak task (claude-ebeam, codex 0858 / 0907).

Imports the frozen, scored task_demo unchanged, and adds one passive camera, 'review_cam', at the calibrated top camera's
exact pose, with an ideal pinhole at REVIEW_FOCAL_SCALE x the top lens focal length (codex projected the blade boxes:
about 0.63x shows both blade tips at 640 x 352). The three policy cameras (top_cam, left_wrist_cam, right_wrist_cam), the
dynamics, the controller and the scorer are untouched: the extra sensor is never in the observation (inspect_embodiment
_observation reads only DEFAULT_CAMERAS). Each control step's review_cam RGB is written as a JPEG to $KF_REVIEW_DIR.
Run with -E task=review_task.py and the same env/command as the scored run; compare its KFM trace to the scored run's."""
import importlib.abc
import importlib.util
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from task_demo import *  # noqa: E402,F401,F403  (the frozen scored task, unchanged)
import task_demo as _T  # noqa: E402
from labkit import on_step  # noqa: E402

REVIEW_FOCAL_SCALE = float(os.environ.get("KF_REVIEW_FOCAL", "0.63"))
REVIEW_DIR = os.environ.get("KF_REVIEW_DIR", "")


def _patch_embodiment_module(E):
    cls = E.BimanualYamEmbodiment
    if getattr(cls, "_review_cam", False):
        return

    def _camera_cfg(self, height, width, overview):
        top = self.rig.top_cam
        fields = [("top_cam", E.CameraCfg, E.CameraCfg(prim_path="{ENV_REGEX_NS}/TopCam", height=height, width=width,
                                                       data_types=["rgb"], spawn=E._pinhole(top, height, width, near=0.05),
                                                       offset=E._offset(top), renderer_cfg=E.RENDERER))]
        for side in self.arms:
            fields.append((E.SIDES[side]["cam"], E.CameraCfg, self._wrist_cam_cfg(side, height, width)))
        lens = top.lens.resized(width, height)
        fields.append(("review_cam", E.CameraCfg, E.CameraCfg(
            prim_path="{ENV_REGEX_NS}/ReviewCam", height=height, width=width, data_types=["rgb"],
            spawn=E.sim_utils.PinholeCameraCfg(focal_length=lens.focal_mm * REVIEW_FOCAL_SCALE, clipping_range=(0.05, 20.0)),
            offset=E._offset(top), renderer_cfg=E.RENDERER)))
        return E.make_configclass("YamCameraCfg", fields, bases=(E.ArenaCameraCfg,))()

    cls._camera_cfg = _camera_cfg
    cls._review_cam = True


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch_embodiment_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.embodiment":
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


if "yam_arena.embodiment" in sys.modules:
    _patch_embodiment_module(sys.modules["yam_arena.embodiment"])
else:
    sys.meta_path.insert(0, _Finder())

_N = [0]


@on_step
def _review_frame(env):
    if not REVIEW_DIR:
        return
    try:
        import cv2
        out = env.unwrapped.scene["review_cam"].data.output["rgb"]
        out = out.torch if hasattr(out, "torch") else out
        rgb = out[0, ..., :3].cpu().numpy()
        os.makedirs(REVIEW_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(REVIEW_DIR, "review_%05d.jpg" % _N[0]), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        _N[0] += 1
    except Exception as e:
        if _N[0] == 0:
            print("REVIEW_CAM error %r" % (e,), flush=True)
            _N[0] = -10 ** 9
