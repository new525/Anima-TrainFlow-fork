"""Inspector, tagger and the training pre-flight explain what to do with videos/GIFs that aren't converted yet."""
import shutil, tempfile
from pathlib import Path
from PIL import Image
from _harness import check, drain, finish, load_app, media_dir

g = load_app()
SRC = media_dir()
work = g["_work"]
for f in ("dit", "qwen", "vae"):
    (work / f"{f}.safetensors").touch()
g["training_manager"].start = lambda *a, **k: True   # don't launch anything
last = lambda gen: drain(g, gen, clear=False)

last = lambda gen: [x for x in gen][-1]
d = Path(tempfile.mkdtemp())
shutil.copy(SRC / "clip_h264.mp4", d / "clip.mp4"); (d / "clip.txt").write_text("1girl")
log = last(g["inspect_dataset"](str(d), "kw"))
check("inspector: flags the video, doesn't call clip.txt an orphan", "1 video/GIF/other-format file(s) not converted yet" in log and "orphan" not in log.lower() and "no image" not in log,
      [l for l in log.splitlines() if "video" in l or "image" in l][:3])
log = last(g["run_auto_tagging"](str(d), 0.35, 0.85, False))
check("tagger: explains why there are no images", "1 video/GIF/other-format file(s) found: run ✂️ Start Bucketing first" in log, log.splitlines()[-1])
args = lambda: ["kw", str(d), str(work/"dit.safetensors"), str(work/"qwen.safetensors"), str(work/"vae.safetensors"), 32, "1.0", "Prodigy",
                100, 50, 50, "", "neg", 1024, 1024, 30, 4.0, 42, 42, 1, 1, True, 0]
log = g["start_training"](*args())
check("pre-flight: videos only -> error with the fix", "Only videos/GIFs/other formats were found: run ✂️ Start Bucketing first." in log, log.splitlines()[-1])
Image.new("RGB", (512, 768)).save(d / "img.png"); (d / "img.txt").write_text("1girl")
g["console"].clear(); log = g["start_training"](*args())
check("pre-flight: mixed folder -> trains with a warning", "1 video/GIF/other-format file(s) won't be trained on" in log and "Run folder" in log,
      [l for l in log.splitlines() if "won't" in l])
shutil.rmtree(d)

finish()
