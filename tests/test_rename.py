"""Rename to Numbers: rolls back when a rename fails part-way, pairs files regardless of letter case, refuses
names that differ only in case, and only refuses during training when training uses this folder."""
import os
from pathlib import Path

import toml
from PIL import Image

from _harness import check, drain, finish, listing, load_app, tmpdir

g = load_app()
rename = lambda d: drain(g, g["rename_dataset_sequential"](str(d)))


def dataset(names):
    d = tmpdir()
    for i, n in enumerate(names):
        Image.new("RGB", (64, 64), (i * 40, 0, 0)).save(d / f"{n}.png")
        (d / f"{n}.txt").write_text(f"caption {n}", encoding="utf-8")
    return d


def failing_replace(fail_on_call, lock_files=0):
    """os.replace that raises like a locked file on its Nth call. With lock_files=k, the first k files the undo then
    tries to move stay locked for good (every later attempt on them fails too)."""
    real = os.replace
    state = {"n": 0, "failed": False, "locked": set()}

    def fake(src, dst, *a, **k):
        state["n"] += 1
        if state["n"] == fail_on_call:
            state["failed"] = True
            raise PermissionError(13, "The process cannot access the file because it is being used by another process", str(src))
        if state["failed"]:
            if str(src) not in state["locked"] and len(state["locked"]) < lock_files:
                state["locked"].add(str(src))
            if str(src) in state["locked"]:
                raise PermissionError(13, "still locked", str(src))
        return real(src, dst, *a, **k)
    return real, fake


for label, call in (("while moving to temporary names", 4), ("while giving the final names", 13)):
    d = dataset(["b", "c", "a", "d", "e"])
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    real, fake = failing_replace(call)
    os.replace = fake
    try:
        log = rename(d)
    finally:
        os.replace = real
    after = {p.name: p.read_bytes() for p in d.iterdir()}
    check(f"failure {label}: every file back under its old name, contents intact", after == before, sorted(after))
    check(f"failure {label}: reported as undone, no rename map left", "was undone, nothing changed" in log and not list(d.glob("rename_map_*")),
          log.splitlines()[-2:])

d = dataset(["b", "c", "a", "d", "e"])
real, fake = failing_replace(13, lock_files=1)
os.replace = fake
try:
    log = rename(d)
finally:
    os.replace = real
check("undo that can't finish lists what to rename by hand and keeps the map",
      "could not be put back" in log and "→" in log and len(list(d.glob("rename_map_*"))) == 1, log.splitlines()[-4:])
hand = [line.strip()[2:].split(" → ") for line in log.splitlines() if line.strip().startswith("• ")]
for now, was in hand:
    os.replace(d / now, d / was)
check("…and following those instructions restores the folder exactly", {p.name: p.read_bytes() for p in d.iterdir()
      if not p.name.startswith("rename_map")} == before, hand)

# swapped names (002 -> 001 and 001 -> 002) with an undo that fails: nothing may be overwritten
d = tmpdir()
for n, c in [("002", "two"), ("1", "one"), ("b", "bee")]:
    Image.new("RGB", (64, 64)).save(d / f"{n}.jpg"); (d / f"{n}.txt").write_text(c, encoding="utf-8")
before = {p.name: p.read_bytes() for p in d.iterdir()}
for fail_at in range(1, 13):
    for undo_fail in (0, 1, 2, 3):
        real, fake = failing_replace(fail_at, lock_files=undo_fail)
        os.replace = fake
        try:
            log = rename(d)
        finally:
            os.replace = real
        contents = sorted(p.read_bytes() for p in d.iterdir() if not p.name.startswith("rename_map"))
        if contents != sorted(before.values()):
            check(f"swap rename, failure at call {fail_at}, {undo_fail} failed undo(s): no file lost", False, sorted(p.name for p in d.iterdir()))
            break
        for line in log.splitlines():  # put things back by hand, as the message says
            if line.strip().startswith("• "):
                now, was = line.strip()[2:].split(" → ")
                os.replace(d / now, d / was)
        for m in d.glob("rename_map_*"):
            m.unlink()
        if {p.name: p.read_bytes() for p in d.iterdir()} != before:
            check(f"swap rename, failure at call {fail_at}, {undo_fail} failed undo(s): restorable", False, sorted(p.name for p in d.iterdir()))
            break
    else:
        continue
    break
else:
    check("swapped names: every failure point x 0-3 locked files loses nothing and the listed steps restore it", True)

# Windows pairs IMG_0001.JPG with img_0001.txt; the rename must keep them together
d = tmpdir()
Image.new("RGB", (64, 64), "red").save(d / "IMG_0001.JPG"); (d / "img_0001.txt").write_text("red one", encoding="utf-8")
Image.new("RGB", (64, 64), "blue").save(d / "IMG_0002.JPG"); (d / "IMG_0002.txt").write_text("blue one", encoding="utf-8")
(d / "img_0002.JPG.XMP").write_text("<x/>", encoding="utf-8")
rename(d)
names = [n for n in listing(d) if not n.startswith("rename_map")]
check("caption and sidecar with different letter case move with their image",
      names == ["001.JPG", "001.txt", "002.JPG", "002.JPG.XMP", "002.txt"]
      and (d / "001.txt").read_text(encoding="utf-8") == "red one", names)

d = tmpdir()
Image.new("RGB", (64, 64)).save(d / "Cat.png"); Image.new("RGB", (64, 64)).save(d / "cat.jpg")
before = listing(d)
log = rename(d)
check("images whose names differ only in letter case: refused, nothing changed",
      "differ only in letter case" in log and listing(d) == before, log.splitlines()[-2:])

# training interlock: refused only when the running trainer uses this folder
d = dataset(["x", "y"])
other = tmpdir("other")
run_dir = tmpdir("run")
(run_dir / "configs").mkdir()
tm = g["training_manager"]
real_running, real_snapshot = tm.is_running, tm.snapshot


def training_on(folder):
    toml.dump({"datasets": [{"subsets": [{"image_dir": Path(folder).resolve().as_posix()}]}]},
              open(run_dir / "configs" / "kw_dataset.toml", "w", encoding="utf-8"))
    tm.is_running = lambda: True
    tm.snapshot = lambda: {"state": "running", "step": None, "run_dir": run_dir, "output_name": "kw"}


try:
    training_on(d)
    log = rename(d)
    check("refused while training uses this dataset", "training is running on this dataset" in log and (d / "x.png").exists())
    training_on(other)
    log = rename(d)
    check("allowed while training uses another dataset", (d / "001.png").exists(), log.splitlines()[-1])
finally:
    tm.is_running, tm.snapshot = real_running, real_snapshot

finish()
