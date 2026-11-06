"""Videos, GIF and AVIF in the dataset: middle frame / decoded image saved as a PNG original, source kept, re-runs
don't reconvert, name clashes abort, a source that can't be moved leaves nothing behind."""
import shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from _harness import bucket, check, finish, listing, load_app, media_dir

g = load_app()
SRC = media_dir()


def run(d, mn=512, mx=768):
    return bucket(g, d, mn, mx, set_aside=False, clear=False)



d = Path(tempfile.mkdtemp())
for n in ["clip_h264.mp4", "clip_vp9.webm", "anim.gif", "photo.avif", "phone_rotated.mp4", "broken.mp4"]:
    shutil.copy(SRC / n, d / n)
Image.fromarray(np.full((1500, 1000, 3), 77, np.uint8)).save(d / "plain.jpg")
for stem in ["clip_h264", "anim", "plain"]:
    (d / f"{stem}.txt").write_text(f"caption for {stem}")
log = run(d)
root, orig = listing(d), listing(d / "original_images")
check("outputs: one bucketed PNG per source, broken file left in place",
      root == ["anim.png", "anim.txt", "broken.mp4", "clip_h264.png", "clip_h264.txt", "clip_vp9.png", "duplicates.log", "phone_rotated.png",
               "photo.png", "plain.png", "plain.txt"], root)  # duplicates.log: this fixture holds one clip in three formats
check("originals: full-size frame PNGs plus every source file",
      all(n in orig for n in ["clip_h264.png", "clip_h264.mp4", "clip_vp9.png", "clip_vp9.webm", "anim.png", "anim.gif",
                              "photo.png", "photo.avif", "phone_rotated.png", "phone_rotated.mp4", "plain.jpg"]), orig)
check("frame original is full resolution", Image.open(d / "original_images" / "clip_h264.png").size == (320, 180))
check("captions untouched and pair with the new PNGs", (d / "clip_h264.txt").read_text() == "caption for clip_h264")
sizes = {n: Image.open(d / n).size for n in ["clip_h264.png", "phone_rotated.png", "photo.png"]}
check("landscape clip -> landscape bucket, phone clip -> portrait bucket", sizes["clip_h264.png"][0] > sizes["clip_h264.png"][1]
      and sizes["phone_rotated.png"][1] > sizes["phone_rotated.png"][0], sizes)
check("log names each conversion and the failure", "video clip_h264.mp4 → frame 31 of 61, 320×180" in log
      and "GIF anim.gif → frame 31 of 61" in log and "AVIF photo.avif → converted" in log
      and "broken.mp4:" in log and "Converted 5, ❌ failed 1" in log, [l for l in log.splitlines() if "🎞️" in l or "⚠️" in l])

snapshot = {p.name: p.stat().st_mtime_ns for p in d.rglob("*") if p.is_file()}
log2 = run(d)
after = {p.name: p.stat().st_mtime_ns for p in d.rglob("*") if p.is_file()}
check("re-run: nothing reconverted or rewritten (only the broken file is retried)", snapshot == after and "Converted 0, ❌ failed 1" in log2,
      [l for l in log2.splitlines() if "Converted" in l])
log3 = run(d, 640, 1024)
check("re-run with new sizes re-crops from the frame originals",
      Image.open(d / "clip_h264.png").size[1] == 640 and Image.open(d / "original_images" / "clip_h264.png").size == (320, 180),
      Image.open(d / "clip_h264.png").size)
shutil.rmtree(d)

# clashes: abort before changing anything
for label, files in [("video + image with the same name", ["clip.mp4", "clip.jpg"]),
                     ("video + GIF with the same name", ["clip.mp4", "clip.gif"])]:
    d = Path(tempfile.mkdtemp())
    for n in files:
        if n.endswith(".mp4"): shutil.copy(SRC / "clip_h264.mp4", d / n)
        elif n.endswith(".gif"): shutil.copy(SRC / "anim.gif", d / n)
        else: Image.fromarray(np.full((900, 600, 3), 5, np.uint8)).save(d / n)
    before = listing(d); log = run(d)
    check(f"clash ({label}) aborts with nothing changed", "aborted" in log and listing(d) == before and not (d / "original_images").exists(), before)
    shutil.rmtree(d)
d = Path(tempfile.mkdtemp())
Image.fromarray(np.full((900, 600, 3), 5, np.uint8)).save(d / "cat.jpg"); run(d)
shutil.copy(SRC / "clip_h264.mp4", d / "cat.mp4"); before = sorted(str(p.relative_to(d)) for p in d.rglob("*"))
log = run(d)
check("new video named like an existing original aborts", "cat.mp4 (an original named 'cat.jpg' is already in original_images/)" in log
      and sorted(str(p.relative_to(d)) for p in d.rglob("*")) == before)
shutil.rmtree(d)

# the source can't be moved into original_images/ (file locked): its PNG original is removed again, so the folder is
# exactly as before and the next run simply converts it
d = Path(tempfile.mkdtemp())
shutil.copy(SRC / "anim.gif", d / "clip.gif")
real_move = shutil.move


def locked(src, dst, *a, **k):
    if str(src).endswith("clip.gif"):
        raise PermissionError(13, "The process cannot access the file because it is being used by another process", str(src))
    return real_move(src, dst, *a, **k)


shutil.move = locked
try:
    log = run(d)
finally:
    shutil.move = real_move
check("locked source: reported, nothing left behind", "clip.gif:" in log and "left in the dataset folder, unchanged" in log
      and not (d / "original_images" / "clip.png").exists() and not (d / "clip.png").exists(),
      sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()))
log = run(d)
check("…and the next run converts it normally", "GIF clip.gif → frame 31 of 61" in log and (d / "clip.png").exists()
      and (d / "original_images" / "clip.gif").exists(), [l for l in log.splitlines() if "clip" in l or "abort" in l])
shutil.rmtree(d)

finish()
