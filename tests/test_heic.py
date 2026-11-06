"""HEIC/HEIF/HIF photos: converted upright (EXIF rotation applied once), main image of multi-image files."""
import shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pillow_heif
from _harness import bucket, check, finish, load_app

pillow_heif.register_heif_opener()
g = load_app()


def run(d):
    return bucket(g, d, set_aside=False)


d = Path(tempfile.mkdtemp())
a = np.full((200, 300, 3), 90, np.uint8); a[:30, :40] = (255, 0, 0)
Image.fromarray(a).save(d / "landscape.heic", quality=90)
ex = Image.Exif(); ex[0x0112] = 6
Image.fromarray(a).save(d / "iphone_portrait.heic", exif=ex.tobytes(), quality=90)
Image.fromarray(np.full((200, 300, 3), 160, np.uint8)).save(d / "other_ext.heif", quality=90)
shutil.copy(d / "landscape.heic", d / "camera.hif")
imgs = [Image.fromarray(np.full((200, 300, 3), v, np.uint8)) for v in (30, 130, 230)]
imgs[0].save(d / "multi.heic", save_all=True, append_images=imgs[1:], primary_index=1, quality=90)
log = run(d); O = d / "original_images"
px = lambda p, xy: Image.open(p).convert("RGB").getpixel(xy)
red = lambda p: tuple(int(v) for v in np.argwhere((np.asarray(Image.open(p).convert("RGB"))[..., 0] > 200) & (np.asarray(Image.open(p).convert("RGB"))[..., 1] < 80)).mean(0).round())
check("all four extensions converted", all((d / f"{s}.png").exists() for s in ["landscape", "iphone_portrait", "other_ext", "camera", "multi"]),
      sorted(p.name for p in d.iterdir() if p.is_file()))
check("landscape stays landscape", Image.open(O / "landscape.png").size == (300, 200))
check("iPhone-style portrait: rotated exactly once (portrait, marker top-right)",
      Image.open(O / "iphone_portrait.png").size == (200, 300) and red(O / "iphone_portrait.png") == (20, 184), (Image.open(O / "iphone_portrait.png").size, red(O / "iphone_portrait.png")))
check("portrait gets a portrait bucket", Image.open(d / "iphone_portrait.png").size[1] > Image.open(d / "iphone_portrait.png").size[0], Image.open(d / "iphone_portrait.png").size)
v = px(O / "multi.png", (150, 100))[0]
check("multi-image HEIC: the main image (2nd of 3), not the first", abs(v - 130) <= 4 and "main image of 3" in log, (v, [l.strip() for l in log.splitlines() if "multi" in l]))
check("labels follow the extension", "HEIC landscape.heic → converted" in log and "HEIF other_ext.heif" in log and "HIF camera.hif" in log)
check("sources kept", all((O / n).exists() for n in ["landscape.heic", "iphone_portrait.heic", "other_ext.heif", "camera.hif", "multi.heic"]))
shutil.rmtree(d)
d = Path(tempfile.mkdtemp()); Image.fromarray(a).save(d / "x.heic")
g["HEIF_PLUGIN"] = False; log = run(d); g["HEIF_PLUGIN"] = True
check("without the plugin: clear fix, file left in place", "HEIC/HEIF needs pillow-heif: re-run Install_Requirements.bat" in log and (d / "x.heic").exists())
check("pre-flight counts HEIC as skipped by the trainer", g["trainer_skips"](Path("x.heic")) is True)
shutil.rmtree(d)

finish()
