"""Bucketing: originals preserved, re-runs, phone photos, name clashes (incl. letter case), edited outputs, side sizes."""
import json
import shutil

import numpy as np
import toml
from PIL import Image

from _harness import bucket, check, finish, listing, load_app, tmpdir, tree

seen_shapes = []
g = load_app(seen_shapes=seen_shapes)


def sizes(folder):
    return {p.name: Image.open(p).size for p in sorted(folder.iterdir()) if p.suffix.lower() in (".png", ".jpg")}


def solid(w, h, v):
    return Image.fromarray(np.full((h, w, 3), v, np.uint8))


# --- re-run with different sizes must not touch originals
d = tmpdir()
solid(1200, 1600, 10).save(d / "a.png")
solid(1000, 1500, 200).save(d / "b.jpg")
bucket(g, d, 512, 768)
orig_after_1 = sizes(d / "original_images")
bucket(g, d, 640, 1024)
orig_after_2 = sizes(d / "original_images")
check("re-run keeps originals intact", orig_after_1 == orig_after_2 == {"a.png": (1200, 1600), "b.jpg": (1000, 1500)}, orig_after_2)
out2 = sizes(d)
check("re-run regenerates outputs at new buckets", all(min(v) == 640 for v in out2.values()), out2)
manifest = json.loads((d / "original_images" / "bucketing_manifest.json").read_text(encoding="utf-8"))
check("manifest records outputs", set(manifest) == {"a.png", "b.png"})

# --- EXIF-rotated phone photo gets a portrait bucket and is processed upright
d = tmpdir()
im = solid(900, 600, 0)
exif = Image.Exif()
exif[0x0112] = 6
im.save(d / "phone.jpg", exif=exif)
seen_shapes.clear()
bucket(g, d)
out = sizes(d)
check("phone photo bucket is portrait", out.get("phone.png", (0, 0))[1] > out.get("phone.png", (1, 0))[0], out)
check("cropper received upright pixels", seen_shapes == [(900, 600)], seen_shapes)

# --- non-ASCII file name + unreadable file are reported, not silently dropped
d = tmpdir()
solid(600, 900, 50).save(d / "猫の写真.jpg")
(d / "broken.jpg").write_bytes(b"not a jpeg")
log = bucket(g, d)
check("non-ASCII name processed", "猫の写真.png" in sizes(d), sizes(d))
check("broken file reported", "broken.jpg" in log, [line for line in log.splitlines() if "broken" in line])
check("broken original still preserved", (d / "original_images" / "broken.jpg").exists())

# --- name clashes abort with nothing changed
for files in (("cat.jpg", "cat.png"), ("Cat.png", "cat.jpg"), ("IMG_1.JPG", "img_1.webp")):
    d = tmpdir()
    for i, n in enumerate(files):
        solid(600, 900, 10 + 40 * i).save(d / n)
    before = listing(d)
    log = bucket(g, d)
    check(f"clash {' + '.join(files)} aborts, nothing changed",
          "aborted" in log and not (d / "original_images").exists() and listing(d) == before, listing(d))

d = tmpdir()
solid(600, 900, 10).save(d / "Cat.png")
solid(600, 900, 200).save(d / "cat.gif")
log = bucket(g, d)
check("Cat.png + cat.gif (letter case only) aborts: on Windows the GIF frame would overwrite Cat.png",
      "aborted" in log and sorted(listing(d)) == ["Cat.png", "cat.gif"] and np.asarray(Image.open(d / "Cat.png"))[0, 0, 0] == 10)

# a new file whose name matches an existing original (any extension, any letter case) is refused too
d = tmpdir()
solid(600, 900, 10).save(d / "dog.jpg")
bucket(g, d)
solid(600, 900, 99).save(d / "Dog.webp")
log = bucket(g, d)
check("new Dog.webp next to original dog.jpg aborts", "aborted" in log and "Dog.webp" in log
      and (d / "Dog.webp").exists() and listing(d / "original_images") == ["bucketing_manifest.json", "dog.jpg"], listing(d))

# originals that already collide (e.g. bucketed by an older version) are reported instead of overwriting each other
d = tmpdir()
(d / "original_images").mkdir()
solid(600, 900, 10).save(d / "original_images" / "Bird.png")
solid(600, 900, 99).save(d / "original_images" / "bird.jpg")
log = bucket(g, d)
check("colliding originals already in original_images abort", "aborted" in log and "original_images/Bird.png" in log, log.splitlines()[-3:])

# --- an output the user edited after bucketing is kept, not overwritten
d = tmpdir()
solid(1000, 1500, 9).save(d / "e.jpg")
bucket(g, d)
solid(512, 768, 123).save(d / "e.png")
log = bucket(g, d, 640, 1024)
px = Image.open(d / "e.png").getpixel((0, 0))
check("edited output kept", px == (123, 123, 123) and "Kept 1" in log, px)

# --- legacy folder (bucketed by the old code, no manifest) re-runs safely
d = tmpdir()
(d / "original_images").mkdir()
solid(1200, 1600, 10).save(d / "original_images" / "a.png")
solid(512, 704, 10).save(d / "a.png")  # old 512/768 output
bucket(g, d, 640, 1024)
check("legacy: original untouched", sizes(d / "original_images") == {"a.png": (1200, 1600)}, sizes(d / "original_images"))
check("legacy: output regenerated", min(sizes(d)["a.png"]) == 640, sizes(d))

# --- Min/Max Side: multiples of 64, Min <= Max
d = tmpdir()
solid(1000, 1500, 9).save(d / "s.jpg")
log = bucket(g, d, 500, 800)
check("sides snapped to 64 (Min up, Max down) and reported", "using Min Side 512 and Max Side 768" in log, log.splitlines()[:2])
check("outputs stay on the 64 grid within the entered range", all(v % 64 == 0 and 512 <= v <= 768 for v in sizes(d)["s.png"]), sizes(d))
d = tmpdir()
solid(1000, 1500, 9).save(d / "s.jpg")
log = bucket(g, d, 1024, 768)
check("Min > Max refused, nothing changed", "larger than Max Side" in log and listing(d) == ["s.jpg"], log.splitlines()[-1])
d = tmpdir()
solid(1000, 1500, 9).save(d / "s.jpg")
log = bucket(g, d, None, 768)
check("empty Min Side refused", "must be numbers" in log and listing(d) == ["s.jpg"], log.splitlines()[-1])

# --- trigger prefix as sd-scripts assembles it (library/train_util.py: caption_prefix + " " + caption)
d = tmpdir()
toml_path = g["create_dataset_toml"]("p", str(d), "mytrigger", 640, 768, d, 100, 1, 1)
prefix = toml.load(toml_path)["datasets"][0]["subsets"][0]["caption_prefix"]
check("caption becomes 'mytrigger, 1girl'", prefix + " " + "1girl" == "mytrigger, 1girl", repr(prefix + " " + "1girl"))

# --- keyword folder names
pf = g["project_folder_name"]
check("non-Latin keyword keeps its own folder", pf("ねこ") == "ねこ" and pf("いぬ") != pf("ねこ"), (pf("ねこ"), pf("いぬ")))
check("forbidden characters replaced", pf('a/b:c*?') == "a_b_c", pf('a/b:c*?'))
check("reserved name guarded", pf("con") == "_con", pf("con"))
check("empty -> untitled", pf("  ") == "untitled")

shutil.rmtree(g["_work"], ignore_errors=True)
finish()
