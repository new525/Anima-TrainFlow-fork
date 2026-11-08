"""Shared test helpers.

Every test runs the real app.py, executed in a fresh temporary folder (so its training/ and settings.json live there),
with torch replaced by a stub that reports a GPU and U2Net replaced by a plain resize. Nothing needs a GPU or the
models. Each test script prints PASS/FAIL lines and exits non-zero if anything failed; run_all.py runs them all.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
APP = REPO / "app.py"
DATA = TESTS / "data"

_failed = []


def check(name, cond, detail=""):
    if not cond:
        _failed.append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""), flush=True)
    return bool(cond)


def finish():
    print("\nALL PASS" if not _failed else f"\n{len(_failed)} FAILED: " + "; ".join(_failed))
    sys.exit(1 if _failed else 0)


def stub_torch():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    sys.modules["torch"] = torch


def load_app(fake_cropper=True, seen_shapes=None, src_transform=None):
    """Execute app.py (without launching the UI) in a fresh temp folder; returns its globals."""
    stub_torch()
    work = Path(tempfile.mkdtemp(prefix="tf_app_"))
    shutil.copy(APP, work / "app.py")
    shutil.copy(REPO / "embedded_tags.py", work / "embedded_tags.py")   # imported by app.py
    src = APP.read_text(encoding="utf-8").replace('if __name__ == "__main__":', "if False:")
    if src_transform:
        src = src_transform(src)
    g = {"__name__": "app", "__file__": str(work / "app.py")}
    exec(compile(src, str(APP), "exec"), g)
    if fake_cropper:
        class FakeCropper(g["SmartCropper"]):
            def load_model(self):
                return "FAKE"

            def process_image(self, img, tw, th):
                import cv2
                if seen_shapes is not None:
                    seen_shapes.append(img.shape[:2])
                return cv2.resize(img, (tw, th))
        g["smart_cropper"] = FakeCropper()
    g["_work"] = work
    return g


def drain(g, gen, clear=True):
    """Run a UI generator to the end; returns the last console text it yielded."""
    if clear:
        g["console"].clear()
    out = ""
    for out in gen:
        pass
    return out


def bucket(g, d, side_min=512, side_max=768, convert_other=True, set_aside=True, clear=True):
    return drain(g, g["run_smart_crop_ui"](str(d), side_min, side_max, convert_other, set_aside), clear)


def tree(d):
    d = Path(d)
    return sorted(str(p.relative_to(d)).replace(os.sep, "/") for p in d.rglob("*") if p.is_file())


def listing(d):
    return sorted(p.name for p in Path(d).iterdir() if p.is_file())


def tmpdir(name="ds"):
    d = Path(tempfile.mkdtemp(prefix="tf_")) / name
    d.mkdir(parents=True)
    return d


def sample_images():
    """Folder of real photos (scikit-image's bundled sample data: astronaut, coffee, chelsea, motorcycle pair…)."""
    try:
        import skimage.data
    except ImportError:
        sys.exit("scikit-image is needed for this test: pip install scikit-image")
    return Path(skimage.data.__file__).parent


_media = None


def media_dir():
    """Test videos/GIF/AVIF, generated once per process with imageio-ffmpeg (see make_media.py)."""
    global _media
    if _media is None:
        _media = Path(tempfile.mkdtemp(prefix="tf_media_"))
        subprocess.run([sys.executable, str(TESTS / "make_media.py"), str(_media)], check=True, capture_output=True)
    return _media
