"""Formats beyond the usual five: JXL, multi-page TIFF, TGA, JFIF, PSD composite, odd video containers and a file
with no extension are converted; audio and junk are left alone."""
import shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from _harness import TESTS, bucket, check, finish, load_app, media_dir

g = load_app()
SRC = Path(tempfile.mkdtemp(prefix="tf_other_")) / "src"
subprocess.run([sys.executable, str(TESTS / "make_other_formats.py"), str(SRC), str(media_dir())], check=True, capture_output=True)


def run(d, convert_other=True):
    return bucket(g, d, convert_other=convert_other, set_aside=False)



fresh = lambda: (lambda d: (shutil.copytree(SRC, d, dirs_exist_ok=True), d)[1])(Path(tempfile.mkdtemp()))
px = lambda p, xy=(150, 100): Image.open(p).convert("RGB").getpixel(xy)

d = fresh(); log = run(d); O = d / "original_images"
root = sorted(p.name for p in d.iterdir() if p.is_file())
check("converted: jxl tiff tga jfif jpg_large psd 3gp ts no-ext",
      all((d / f"{s}.png").exists() for s in ["photo", "scan", "frame", "browser", "pic", "layered", "phone", "recording", "clip_noext"]), root)
check("left alone: audio-only .mka, random .dat, .npz, .json, Thumbs.db",
      all((d / n).exists() for n in ["voice.mka", "notes.dat", "cache.npz", "meta.json", "Thumbs.db"]), root)
check("only the two undecodable files are mentioned", "Left 2 file(s)" in log and "notes.dat" in log and "voice.mka" in log
      and "cache.npz" not in log and "Thumbs.db" not in log, [l for l in log.splitlines() if "Left" in l])
check("JXL: decoded, transparency on white", px(O / "photo.png", (10, 150)) == (255, 255, 255) and
      all(abs(a - b) <= 6 for a, b in zip(px(O / "photo.png", (150, 150)), (20, 120, 220))), (px(O / "photo.png", (10, 150)), px(O / "photo.png", (150, 150))))
check("TIFF: first page, not the middle", abs(px(O / "scan.png")[0] - 60) <= 2 and "first page of 2" in log, px(O / "scan.png"))
check("PSD: flattened composite, not a single layer", px(O / "layered.psd".replace(".psd", ".png"), (10, 100))[0] > 200
      and px(O / "layered.png", (290, 100))[0] < 60, (px(O / "layered.png", (10, 100)), px(O / "layered.png", (290, 100))))
check(".jfif / .jpg_large / .tga decoded", [abs(px(O / f"{s}.png")[0] - v) <= 3 for s, v in [("browser", 110), ("pic", 150), ("frame", 90)]] == [True] * 3)
for s in ["phone", "recording", "clip_noext"]:
    check(f"video {s}: middle frame", abs(px(O / f"{s}.png", (200, 120))[1] - 130) <= 3 and f"video {s}" in log, px(O / f"{s}.png", (200, 120)))
check("labels use the detected format", "JXL photo.jxl" in log and "TIFF scan.tiff" in log and "PSD layered.psd → converted (flattened image, not the separate layers)" in log and "JPEG pic.jpg_large" in log,
      [l.strip() for l in log.splitlines() if "→" in l][:5])
check("sources kept in original_images/", all((O / n).exists() for n in ["photo.jxl", "scan.tiff", "layered.psd", "pic.jpg_large", "recording.ts", "clip_noext"]))
log2 = run(d)
check("re-run: nothing more to convert", "Converting" not in log2 and "Left 2 file(s)" in log2)
shutil.rmtree(d)

d = fresh(); log = run(d, convert_other=False)
check("toggle off: only JXL converted, others untouched", (d / "photo.png").exists() and (d / "scan.tiff").exists() and not (d / "scan.png").exists()
      and (d / "recording.ts").exists() and "Converting other formats is off" in log)
shutil.rmtree(d)

d = Path(tempfile.mkdtemp()); shutil.copy(SRC / "photo.jxl", d)
g["JXL_PLUGIN"] = False; log = run(d); g["JXL_PLUGIN"] = True
check("JXL without the plugin: clear fix in the message", "pillow-jxl-plugin: re-run Install_Requirements.bat" in log and (d / "photo.jxl").exists(),
      [l for l in log.splitlines() if "jxl" in l.lower()])
shutil.rmtree(d)

d = Path(tempfile.mkdtemp()); shutil.copy(SRC / "scan.tiff", d / "a.tiff"); Image.new("RGB", (900, 600)).save(d / "a.jpg")
log = run(d); check("clash: a.tiff + a.jpg aborts", "aborted" in log and (d / "a.tiff").exists() and not (d / "original_images").exists())
shutil.rmtree(d)
# trainer reads .jxl itself when the plugin is present -> pre-flight shouldn't call it skipped
check("pre-flight: .jxl not counted as skipped when the plugin is installed", g["trainer_skips"](Path("x.jxl")) is False and g["trainer_skips"](Path("x.tiff")))

finish()
