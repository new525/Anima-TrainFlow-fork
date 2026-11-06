"""Duplicate detection during bucketing: exact copies, re-saves, resized and mirrored copies are found; colour
variants, same-pose sketches, a stereo pair and near-identical solid colours are not."""
import json, random, shutil, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from _harness import bucket, check, finish, load_app, sample_images, tmpdir, tree
g = load_app()
REAL = sample_images()


def run(d, aside=True):
    return bucket(g, d, set_aside=aside)


def flat_art(seed):
    r = random.Random(seed); im = Image.new("RGB", (600, 900), (250, 250, 250)); d = ImageDraw.Draw(im)
    hair = tuple(r.randrange(256) for _ in range(3)); cloth = tuple(r.randrange(256) for _ in range(3)); cx = 300 + r.randint(-40, 40)
    d.ellipse([cx - 110, 120, cx + 110, 360], fill=hair); d.ellipse([cx - 80, 170, cx + 80, 350], fill=(250, 220, 200))
    d.rectangle([cx - 120, 350, cx + 120, 800], fill=cloth); return im
def sketch(seed):
    r = random.Random(seed); im = Image.new("RGB", (1000, 1500), (255, 255, 255)); dr = ImageDraw.Draw(im); cx = 500 + r.randint(-30, 30)
    dr.ellipse([cx - 160, 200, cx + 160, 540], outline=(0, 0, 0), width=4)
    dr.line([(cx - 180, 560), (cx - 220, 1350)], fill=0, width=4); dr.line([(cx + 180, 560), (cx + 220, 1350)], fill=0, width=4)
    for _ in range(r.randint(6, 14)):
        x0, y0 = cx + r.randint(-200, 200), r.randint(150, 1300); dr.line([(x0, y0), (x0 + r.randint(-120, 120), y0 + r.randint(40, 220))], fill=0, width=4)
    return im
def build(d):
    d.mkdir(parents=True)
    astro = Image.open(REAL / "astronaut.png").convert("RGB")       # 512x512
    coffee = Image.open(REAL / "coffee.png").convert("RGB")
    chelsea = Image.open(REAL / "chelsea.png").convert("RGB")
    astro.save(d / "astro.png"); (d / "astro.txt").write_text("astro caption")
    astro.save(d / "astro_resave.jpg", quality=85)                   # re-save
    astro.resize((256, 256), Image.LANCZOS).save(d / "astro_small.png")   # sample-size copy
    coffee.save(d / "coffee.jpg", quality=92); shutil.copy(d / "coffee.jpg", d / "coffee_copy.jpg")   # identical bytes
    chelsea.save(d / "chelsea.png"); ImageOps.mirror(chelsea).save(d / "chelsea_mirror.png")          # mirrored
    for side in ("left", "right"): Image.open(REAL / f"motorcycle_{side}.png").convert("RGB").save(d / f"moto_{side}.png")  # stereo pair
    for i in range(40): flat_art(i).save(d / f"art{i:02d}.png")      # colour variants, same layout
    for i in range(30): sketch(i).save(d / f"sketch{i:02d}.png")     # same-pose line art, different details
    for i, c in enumerate([(200, 60, 60), (210, 60, 60), (200, 75, 60)]): Image.new("RGB", (900, 600), c).save(d / f"solid{i}.png")
    sketch(99).save(d / "line_orig.png"); sketch(99).resize((500, 750), Image.LANCZOS).save(d / "line_small.jpg", quality=80)
d = Path(tempfile.mkdtemp()) / "ds"; build(d)
t = time.time(); log = run(d); t1 = time.time() - t
dup = d / "duplicates"
aside = sorted(p.name for p in dup.iterdir() if p.is_file()) if dup.exists() else []
check("found exactly the 4 groups (astro x3, coffee x2, chelsea x2, line art x2)", "4 group(s), 5 extra copies" in log, [l for l in log.splitlines() if "Duplicate check" in l])
check("kept the largest astro (with caption), set aside the re-save and the small copy",
      (d / "astro.png").exists() and (d / "astro.txt").read_text() == "astro caption"
      and "astro_resave.png" not in [p.name for p in d.iterdir()] and "astro_small.png" not in [p.name for p in d.iterdir()], aside)
check("identical copy and mirrored copy set aside", sum(1 for n in ["coffee_copy", "chelsea_mirror"] if not (d / f"{n}.png").exists()) == 2
      and "identical file" in log and "mirrored copy" in log)
check("stereo pair kept (not duplicates)", (d / "moto_left.png").exists() and (d / "moto_right.png").exists())
check("all 40 colour-variant artworks kept", all((d / f"art{i:02d}.png").exists() for i in range(40)))
check("all 30 same-pose sketches kept", all((d / f"sketch{i:02d}.png").exists() for i in range(30)))
check("solid colours 10-15 levels apart kept", all((d / f"solid{i}.png").exists() for i in range(3)))
check("re-saved half-size sketch set aside", (d / "line_orig.png").exists() and not (d / "line_small.png").exists())
check("set-aside originals kept in duplicates/original_images/", (dup / "original_images" / "astro_resave.jpg").exists() and (dup / "original_images" / "coffee_copy.jpg").exists(),
      sorted(p.name for p in (dup / "original_images").iterdir()) if (dup / "original_images").exists() else None)
man = json.loads((d / "original_images" / "bucketing_manifest.json").read_text(encoding="utf-8"))
check("manifest has no entries for set-aside images", not any(k.startswith(("astro_resave", "astro_small", "coffee_copy", "chelsea_mirror")) for k in man))
t = time.time(); log2 = run(d); t2 = time.time() - t
check("re-run: no duplicates left, nothing regenerated", "Duplicate check: none found" in log2 and "already bucketed correctly" in log2,
      [l for l in log2.splitlines() if "Duplicate" in l or "✅" in l])
print(f"   timing: first run {t1:.2f}s, cached re-run {t2:.2f}s ({len(list(d.glob('*.png')))} images)")

# a new download that duplicates an image bucketed earlier: the larger copy wins, wherever it is
big = Image.open(REAL / "coffee.png").convert("RGB").resize((1200, 800), Image.LANCZOS); big.save(d / "coffee_hires.jpg", quality=95)
log3 = run(d)
check("new larger copy kept, earlier bucketed one (output+original) set aside", (d / "coffee_hires.png").exists() and not (d / "coffee.png").exists()
      and (dup / "coffee.png").exists() and (dup / "original_images" / "coffee.jpg").exists(), [l.strip() for l in log3.splitlines() if "kept" in l])
Image.open(REAL / "chelsea.png").convert("RGB").resize((150, 100)).save(d / "chelsea_tiny.jpg")
log4 = run(d)
check("new smaller copy set aside, bucketed one kept", (d / "chelsea.png").exists() and not (d / "chelsea_tiny.png").exists() and (dup / "original_images" / "chelsea_tiny.jpg").exists())

# report-only
d2 = Path(tempfile.mkdtemp()) / "ds"; build(d2)
log = run(d2, aside=False)
check("report-only: groups listed, nothing moved", "4 group(s)" in log and "Nothing moved" in log and not (d2 / "duplicates").exists()
      and (d2 / "astro_small.png").exists())
log = "".join(x for x in g["inspect_dataset"](str(d2), "kw"))
check("inspector reports the groups", "4 group(s) of duplicate images" in g["console"].text(), [l for l in g["console"].text().splitlines() if "duplicate" in l][-1:])
# rename still works afterwards and leaves duplicates/ alone
g["console"].clear(); [x for x in g["rename_dataset_sequential"](str(d))]
check("rename after set-aside: dataset renamed, duplicates/ untouched", (d / "001.png").exists() and (dup / "original_images" / "coffee.jpg").exists())

# the inspector only reads: no duplicate-check cache appears in a folder it looked at
d3 = tmpdir()
a = Image.open(REAL / "astronaut.png").convert("RGB")
a.save(d3 / "one.png"); a.save(d3 / "two.jpg", quality=90)
before = tree(d3)
g["console"].clear(); [x for x in g["inspect_dataset"](str(d3), "kw")]
check("inspector finds the pair without writing anything", "1 group(s) of duplicate images" in g["console"].text() and tree(d3) == before, tree(d3))

# the kept (largest) copy has no caption, a copy being set aside does: the caption comes along
d4 = tmpdir()
a.save(d4 / "big.png")
a.resize((300, 300)).save(d4 / "small.jpg"); (d4 / "small.txt").write_text("my hand-written caption", encoding="utf-8")
a.resize((200, 200)).save(d4 / "tiny.jpg"); (d4 / "tiny.txt").write_text("a worse caption", encoding="utf-8")
log = run(d4)
check("kept copy gets the caption of the largest captioned copy", (d4 / "big.txt").exists()
      and (d4 / "big.txt").read_text(encoding="utf-8") == "my hand-written caption", tree(d4))
check("…the set-aside copies keep theirs in duplicates/", (d4 / "duplicates" / "small.txt").exists() and (d4 / "duplicates" / "tiny.txt").exists())
check("…and it's said in the console and in duplicates.log", "had no caption: copied small.txt to big.txt" in log
      and "(caption copied from small.txt)" in (d4 / "duplicates.log").read_text(encoding="utf-8"))
d5 = tmpdir()
a.save(d5 / "big.png"); (d5 / "big.txt").write_text("keeper's own caption", encoding="utf-8")
a.resize((300, 300)).save(d5 / "small.jpg"); (d5 / "small.txt").write_text("other caption", encoding="utf-8")
run(d5)
check("a kept copy that has a caption keeps its own", (d5 / "big.txt").read_text(encoding="utf-8") == "keeper's own caption")
d6 = tmpdir()
a.save(d6 / "big.png"); a.resize((300, 300)).save(d6 / "small.jpg"); (d6 / "small.txt").write_text("c", encoding="utf-8")
run(d6, aside=False)
check("report-only mode copies nothing", not (d6 / "big.txt").exists())

# Windows: a caption whose name differs in letter case from its image goes along when the image is set aside
d7 = tmpdir()
a.save(d7 / "keep.png"); (d7 / "keep.txt").write_text("k", encoding="utf-8")
a.resize((300, 300)).save(d7 / "copy.jpg"); (d7 / "COPY.txt").write_text("c", encoding="utf-8")
run(d7)
check("set-aside takes Copy.TXT-style captions too", not (d7 / "COPY.txt").exists() and (d7 / "duplicates" / "COPY.txt").exists(), tree(d7))

finish()
