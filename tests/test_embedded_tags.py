"""Captions from tags embedded by Grabber/exiftool (IPTC, XMP, Windows keywords, sidecars, video/GIF sources),
formatted the way Anima expects; WD tagger only for images without tags; rename keeps everything linked.
The fixtures in data/grabber were written with exiftool exactly as Grabber writes them (see data/README.md)."""
import json, shutil, tempfile
from pathlib import Path
from PIL import Image
from _harness import DATA, check, drain, finish, load_app

g = load_app()
wd_calls = []
g["wd_tagger"].load_model = lambda: (wd_calls.append("load"), "FAKE-WD")[1]
g["wd_tagger"].predict = lambda img, a, b: "wd, caption"
g["wd_tagger"].unload_model = lambda: None
last = lambda gen: drain(g, gen)
caption = lambda d, s: (d / f"{s}.txt").read_text(encoding="utf-8") if (d / f"{s}.txt").exists() else None
_last_copy = None


def grabber_copy():
    global _last_copy
    _last_copy = Path(tempfile.mkdtemp(prefix="tf_")) / "ds"
    shutil.copytree(DATA / "grabber", _last_copy)
    return _last_copy


EXPECT = {
 "a01": "1girl, long hair, hatsune miku, vocaloid, smile",            # user's comma style, IPTC
 "a02": "1girl, long hair, smile",                                   # both fields: not duplicated
 "a03": "highres, 1girl, hatsune miku, vocaloid, @kantoku, long hair, smile",   # namespaced -> Anima order + @artist
 "a04": "1girl, long hair, hatsune miku, artist name",               # default %all%: artist_name stays a tag
 "a05": "1girl, hatsune miku, vocaloid, @kantoku, artist name",      # %all:includenamespace% underscore form
 "a07": "1girl, 初音ミク, smile",
 "a08": "1girl, webp sidecar", "a09": "1girl, noext sidecar",
 "a10": "1girl, smile, score_9",                                     # Windows keywords; score tag keeps underscore
 "a12": "1girl, ^_^, @_@",                                           # kaomoji kept, @_@ isn't an artist
 "a13": "2girls, multiple girls, yuri",                              # count tags first
 "a15": "1girl, hatsune miku (cosplay)",                             # parentheses not escaped
}
# 1. captioning straight on Grabber's downloads (before bucketing)
d = grabber_copy()
log = last(g["run_auto_tagging"](str(d), 0.35, 0.85, False, True))
for s, exp in EXPECT.items():
    check(f"{s}: {exp}", caption(d, s) == exp, caption(d, s))
check("a11 (no metadata) went to the WD tagger", caption(d, "a11") == "wd, caption" and wd_calls == ["load"], (caption(d, "a11"), wd_calls))
check("summary counts sources", "Captioned 12 image(s) from embedded tags" in log and "XMP sidecar: 2" in log and "Windows keywords: 1" in log,
      [l for l in log.splitlines() if "Captioned" in l or "WD tagger" in l])
# 2. bucket first (PNGs re-encoded, originals moved), then caption: tags come from originals / sources / sidecars
d2 = grabber_copy()
blog = last(g["run_smart_crop_ui"](str(d2), 512, 768, True))
check("bucketing doesn't list .xmp sidecars as unreadable", "Left" not in blog, [l for l in blog.splitlines() if "Left" in l])
wd_calls.clear(); log = last(g["run_auto_tagging"](str(d2), 0.35, 0.85, False, True))
for s, exp in EXPECT.items():
    check(f"after bucketing {s}", caption(d2, s) == exp, caption(d2, s))
check("GIF source (XMP pdf:Keywords) after bucketing", caption(d2, "a06") == "1girl, solo, animated", caption(d2, "a06"))
check("MP4 source after bucketing", caption(d2, "a14") == "1girl, video frame", caption(d2, "a14"))
check("sources reported as (original)", "XMP (original)" in log and "IPTC (original)" in log)
# 3. every image tagged -> WD never loads
d3 = grabber_copy(); (d3 / "a11.jpg").unlink()
wd_calls.clear(); log = last(g["run_auto_tagging"](str(d3), 0.35, 0.85, False, True))
check("all embedded: WD model never loaded", wd_calls == [] and "WD tagger not needed" in log, wd_calls)
# 4. checkbox off -> WD for everything
d4 = grabber_copy()
log = last(g["run_auto_tagging"](str(d4), 0.35, 0.85, False, False))
check("embedded off: everything from WD", all(caption(d4, s) == "wd, caption" for s in EXPECT) and "embedded" not in log.lower())
# 5. existing captions kept unless Overwrite
(d4 / "a01.txt").write_text("my edit"); log = last(g["run_auto_tagging"](str(d4), 0.35, 0.85, False, True))
check("existing .txt kept without Overwrite", caption(d4, "a01") == "my edit")
# 6. inspector
log = last(g["inspect_dataset"](str(d2), "kw"))
check("inspector reports embedded tags", "carry embedded tags" in log, [l for l in log.splitlines() if "embedded" in l])

# 7. rename to numbers after bucketing + captioning: everything stays linked
before = {s: caption(d2, s) for s in [p.stem for p in d2.glob("*.png")]}
log = last(g["rename_dataset_sequential"](str(d2)))
O = d2 / "original_images"
pngs = sorted(p.stem for p in d2.glob("*.png"))
check("renamed to 001..015", pngs == [f"{i:03d}" for i in range(1, 16)], pngs)
order = sorted(before, key=str.lower)
check("each caption moved with its image", all(caption(d2, f"{i:03d}") == before[s] for i, s in enumerate(order, 1)))
check("originals, sources and sidecars renamed", (O / "006.gif").exists() and (O / "014.mp4").exists() and (O / "001.jpg").exists()
      and (d2 / "008.webp.xmp").exists() and (d2 / "009.xmp").exists() and not list(d2.glob("a0*")) and not list(O.glob("a0*")),
      sorted(p.name for p in O.iterdir())[:8])
man = json.loads((O / "bucketing_manifest.json").read_text(encoding="utf-8"))
check("manifest updated", set(man) == {f"{i:03d}.png" for i in range(1, 16)} and man["006.png"]["source"] == "006.png" and man["001.png"]["source"] == "001.jpg", (len(man), man.get("006.png"), man.get("001.png")))
check("rename map written", len(list(d2.glob("rename_map_*.json"))) == 1)
blog = last(g["run_smart_crop_ui"](str(d2), 512, 768, True))
check("re-bucketing after rename: nothing new, nothing regenerated", "All 15 images are already bucketed correctly" in blog and "Moved" not in blog,
      [l for l in blog.splitlines() if "✅" in l or "Moved" in l])
wd_calls.clear(); log = last(g["run_auto_tagging"](str(d2), 0.35, 0.85, True, True))
check("embedded tags still found after rename (overwrite on)", caption(d2, "003") == EXPECT["a03"] and caption(d2, "014") == "1girl, video frame",
      (caption(d2, "003"), caption(d2, "014")))
log = last(g["rename_dataset_sequential"](str(d2)))
check("second rename: already numbered", "already numbered" in log)
# 8. collision with an unrelated file aborts with nothing changed
d5 = Path(tempfile.mkdtemp()) / "ds"; d5.mkdir()
for n in ["b.jpg", "a.jpg"]: Image.new("RGB", (64, 64)).save(d5 / n)
(d5 / "002.txt").write_text("orphan caption")
snap = sorted(p.name for p in d5.iterdir()); log = last(g["rename_dataset_sequential"](str(d5)))
check("orphan caption with a target name aborts, nothing changed", "aborted" in log and "002.txt" in log and sorted(p.name for p in d5.iterdir()) == snap, sorted(p.name for p in d5.iterdir()))
(d5 / "002.txt").unlink(); (d5 / "002.jpg.xmp").write_text("x")
log = last(g["rename_dataset_sequential"](str(d5)))
check("orphan sidecar with a target name also aborts", "aborted" in log and (d5 / "b.jpg").exists())
# 9. swap-style renames don't collide (002 -> 001, 001 -> 002 situations)
d6 = Path(tempfile.mkdtemp()) / "ds"; d6.mkdir()
for n, c in [("002.jpg", "two"), ("1.jpg", "one")]:
    Image.new("RGB", (64, 64)).save(d6 / n); (d6 / n).with_suffix(".txt").write_text(c)
log = last(g["rename_dataset_sequential"](str(d6)))
check("names that overlap the targets are handled", caption(d6, "001") == "two" and caption(d6, "002") == "one", (caption(d6, "001"), caption(d6, "002")))

finish()
