"""duplicates.log: written in report-only and set-aside mode, points at the originals, isn't appended twice,
and the other tools ignore it."""
import re, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from _harness import bucket, check, finish, load_app, sample_images
g = load_app()
REAL = sample_images()


def run(d, aside):
    return bucket(g, d, set_aside=aside)


def build(d):
    d.mkdir(parents=True)
    a = Image.open(REAL / "astronaut.png").convert("RGB"); a.save(d / "astro.png"); a.save(d / "astro_resave.jpg", quality=85); a.resize((256, 256)).save(d / "astro_small.png")
    c = Image.open(REAL / "chelsea.png").convert("RGB"); c.save(d / "cat.png"); ImageOps.mirror(c).save(d / "cat_flipped.png")
    Image.open(REAL / "coffee.png").convert("RGB").save(d / "coffee.png")
d = Path(tempfile.mkdtemp()) / "ds"; build(d)
log = run(d, False); L = d / "duplicates.log"; text = L.read_text(encoding="utf-8") if L.exists() else ""
print("---- duplicates.log (report-only) ----\n" + text + "\n--------------------------------------")
check("report-only: log written and announced", L.exists() and "Full list in duplicates.log" in log)
check("report-only: every copy listed with reason and path", all(s in text for s in ["KEEP  astro.png  512x512", "COPY  astro_resave.jpg  512x512  same picture  original_images/astro_resave.jpg",
      "COPY  astro_small.png  256x256  same picture", "COPY  cat_flipped.png  451x300  mirrored copy"]) and "not moved" in text)
check("report-only: nothing moved", not (d / "duplicates").exists())
before = L.read_text(encoding="utf-8"); run(d, False)
check("report-only again, same result: not appended twice", L.read_text(encoding="utf-8") == before)
log = run(d, True); text = L.read_text(encoding="utf-8")
print("---- appended after set-aside ----\n" + text[len(before):] + "\n----------------------------------")
check("set-aside: every copy points at its original in duplicates/original_images/", text.count("=== ") == 2 and all(f"-> duplicates/original_images/{n}" in text for n in ["astro_resave.jpg", "astro_small.png", "cat_flipped.png"]))
after = L.read_text(encoding="utf-8"); log = run(d, True)
check("later check finds none: log untouched", L.read_text(encoding="utf-8") == after and "none found" in log)
# the log is invisible to the other tools
g["console"].clear(); ilog = "".join(x for x in g["inspect_dataset"](str(d), "kw")); itxt = g["console"].text()
check("inspector: no orphan-caption / format warnings about the log", "duplicates.log" not in itxt, [l for l in itxt.splitlines() if "duplicates" in l])
check("bucketing: log isn't probed as a file to convert", "duplicates.log" not in run(d, True))
g["console"].clear(); [x for x in g["rename_dataset_sequential"](str(d))]
check("rename: log left alone", (d / "duplicates.log").exists() and (d / "001.png").exists())
# >30 groups: console truncates, file has all
d2 = Path(tempfile.mkdtemp()) / "ds"; d2.mkdir(); rng = np.random.default_rng(3)
for i in range(35):
    im = Image.fromarray(rng.integers(0, 255, (24, 16, 3), dtype=np.uint8)).resize((600, 900), Image.BICUBIC)
    im.save(d2 / f"p{i:02d}.png"); im.save(d2 / f"p{i:02d}_copy.jpg", quality=90)
log = run(d2, False); text = (d2 / "duplicates.log").read_text(encoding="utf-8")
check("35 groups: console says '… and 5 more', file lists all 35", "and 5 more group(s)" in log and len(re.findall(r"^KEEP ", text, re.M)) == 35)
# no duplicates -> no file
d3 = Path(tempfile.mkdtemp()) / "ds"; d3.mkdir(); Image.open(REAL / "coffee.png").save(d3 / "a.png"); Image.open(REAL / "chelsea.png").save(d3 / "b.png")
run(d3, True); check("no duplicates: no log file", not (d3 / "duplicates.log").exists())

finish()
