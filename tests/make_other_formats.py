import subprocess, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pillow_jxl, imageio_ffmpeg
out = Path(sys.argv[1]); src = Path(sys.argv[2]); shutil.rmtree(out, ignore_errors=True); out.mkdir()
FF = imageio_ffmpeg.get_ffmpeg_exe()
gray = lambda v, w=300, h=200: Image.fromarray(np.full((h, w, 3), v, np.uint8))
a = np.zeros((300, 200, 4), np.uint8); a[..., :3] = (20, 120, 220); a[..., 3] = 255; a[:, :50, 3] = 0
Image.fromarray(a, "RGBA").save(out / "photo.jxl", quality=95)
gray(60).save(out / "scan.tiff", save_all=True, append_images=[gray(200)])          # page 1 = 60, page 2 = 200
gray(90).save(out / "frame.tga"); gray(110).save(out / "browser.jfif", format="JPEG", quality=95)
gray(150).save(out / "pic.jpg_large", format="JPEG", quality=95)
# layered PSD: background layer 40, top layer 230 covering the left half -> composite has both
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
psd = PSDImage.new("RGB", (300, 200))
psd.append(PixelLayer.frompil(gray(40), psd, "bg"))
top = PixelLayer.frompil(gray(230, 150, 200), psd, "top"); psd.append(top)
psd.save(out / "layered.psd")
clip = src / "clip_h264.mp4"
subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(clip), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "3gp", str(out / "phone.3gp")], check=True)
subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(clip), "-c", "copy", "-f", "mpegts", str(out / "recording.ts")], check=True)
shutil.copy(clip, out / "clip_noext")
subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "pcm_s16le", str(out / "voice.mka")], check=True)
(out / "notes.dat").write_bytes(bytes(range(256)) * 40)
(out / "cache.npz").write_bytes(b"PK\x03\x04junk"); (out / "meta.json").write_text("{}"); (out / "Thumbs.db").write_bytes(b"\xd0\xcf\x11\xe0junk")
print(sorted(p.name for p in out.iterdir()))
