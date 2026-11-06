"""Frame extraction: middle frame of each video codec/container and GIF, AVIF decode, phone-video rotation,
broken files, missing frame count, and non-ASCII file names that OpenCV on Windows refuses."""
import shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import imageio_ffmpeg
from _harness import check, finish, load_app, media_dir

g = load_app(fake_cropper=False)
SRC = media_dir()
FF = imageio_ffmpeg.get_ffmpeg_exe()

mid = lambda rgb: int(np.median(rgb[40:, 60:, 1]))   # gray level away from the red marker
for n in ["clip_h264.mp4", "clip_vp9.webm", "clip_h264.mkv", "clip_mjpeg.avi", "clip.mov", "anim.gif"]:
    rgb, what = g["extract_media_frame"](SRC / n)
    check(f"{n:16} middle frame", what == "frame 31 of 61" and abs(mid(rgb) - 130) <= 3 and rgb.shape == (180, 320, 3), f"{what}, gray {mid(rgb)}, {rgb.shape}")
rgb, what = g["extract_media_frame"](SRC / "still.gif")
check("still.gif single frame", what == "converted" and abs(mid(rgb) - 10) <= 2, f"{what}, gray {mid(rgb)}")
rgb, what = g["extract_media_frame"](SRC / "photo.avif")
check("photo.avif decoded, transparency on white", rgb.shape == (300, 200, 3) and rgb[150, 10].tolist() == [255, 255, 255]
      and all(abs(a - b) <= 6 for a, b in zip(rgb[150, 150].tolist(), (20, 120, 220))), f"{rgb.shape} left={rgb[150,10].tolist()} right={rgb[150,150].tolist()}")
# phone video: compare with ffmpeg's own decode of the same frame (ffmpeg applies the rotation flag by default)
rgb, what = g["extract_media_frame"](SRC / "phone_rotated.mp4")
oracle = Path(tempfile.mkdtemp(prefix="tf_")) / "oracle_phone.png"
subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(SRC / "phone_rotated.mp4"), "-vf", "select=eq(n\\,30)", "-frames:v", "1", str(oracle)], check=True)
ref = np.asarray(Image.open(oracle).convert("RGB"))
red = lambda a: tuple(int(v) for v in np.argwhere((a[..., 0] > 200) & (a[..., 1] < 60)).mean(0).round())
check("phone video rotated like ffmpeg does it", rgb.shape == ref.shape and red(rgb) == red(ref) and abs(mid(rgb.transpose(1, 0, 2)) - 130) <= 3,
      f"ours {rgb.shape} red@{red(rgb)} | ffmpeg {ref.shape} red@{red(ref)}")
try:
    g["extract_media_frame"](SRC / "broken.mp4"); check("broken.mp4 raises", False)
except Exception as e:
    check("broken.mp4 raises a readable error", "open" in str(e), str(e))
# frame-count fallback: pretend the container reports no frame count
cv2 = g["cv2"]; real = cv2.VideoCapture
class NoCount:
    def __init__(self, *a): self.c = real(*a)
    def get(self, p): return 0 if p == cv2.CAP_PROP_FRAME_COUNT else self.c.get(p)
    def __getattr__(self, k): return getattr(self.c, k)
g["cv2"].VideoCapture = NoCount
try:
    rgb, what = g["extract_media_frame"](SRC / "clip_vp9.webm")
    check("no frame count -> decode-and-count fallback", what == "frame 31 of 61" and abs(mid(rgb) - 130) <= 3, f"{what}, gray {mid(rgb)}")
finally:
    g["cv2"].VideoCapture = real

# Windows OpenCV can't open some non-ASCII paths: simulate it refusing them; the temp-copy fallback must kick in
uni = Path(tempfile.mkdtemp(prefix="tf_")) / "unicode_case"; uni.mkdir()
src = uni / "猫の動画.mp4"; shutil.copy(SRC / "clip_h264.mp4", src)
cv2 = g["cv2"]; real = cv2.VideoCapture; opened = []


def picky(p, *a):
    opened.append(p)
    return real("/nonexistent/x.mp4") if not str(p).isascii() else real(p, *a)


cv2.VideoCapture = picky
try:
    rgb, what = g["extract_media_frame"](src)
    tmp_used = [p for p in opened if "trainflow_" in str(p)]
    check("non-ASCII name OpenCV refuses: read via an ASCII temp copy, which is cleaned up",
          what == "frame 31 of 61" and bool(tmp_used) and not Path(tmp_used[0]).parent.exists(), (what, tmp_used[:1]))
finally:
    cv2.VideoCapture = real

finish()
