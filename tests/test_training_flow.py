"""Training entry points and small app behaviours: input validation before a run folder exists, the interlock between
training and the dataset tools, Train More rolling back when the trainer can't launch, settings.json safety, preview
prompt trigger handling, WD tagger caption format, inspector checks, launch settings."""
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import toml
from PIL import Image

from _harness import REPO, check, drain, finish, load_app, tmpdir

g = load_app()
console, tm, tools = g["console"], g["training_manager"], g["tool_runner"]
work = g["_work"]
models = work / "models_for_test"
models.mkdir()
for f in ("dit", "qwen", "vae"):
    (models / f"{f}.safetensors").touch()


def dataset():
    d = tmpdir()
    for i in range(3):
        Image.new("RGB", (512, 768)).save(d / f"{i}.png")
        (d / f"{i}.txt").write_text("1girl", encoding="utf-8")
    return d


def args(kw, ds, **over):
    a = dict(trigger_word=kw, dataset_path=str(ds), dit_p=str(models / "dit.safetensors"), qwen_p=str(models / "qwen.safetensors"),
             vae_p=str(models / "vae.safetensors"), rank=32, lr="1.0", optimizer="Prodigy", t_steps=300, save_steps=100,
             sample_steps=50, pos="p", neg="n", w=1024, h=1024, s_steps_gen=30, s_cfg=4.0, s_seed=42, train_seed=42,
             batch_size=1, grad_acc=1, save_resume_state=True, blocks_to_swap=0, network_alpha=0)
    a.update(over)
    return a


launches = []
real_start = tm.start
tm.start = lambda *a, **k: (launches.append(a), True)[1]   # record instead of launching a trainer

# --- invalid inputs are refused before anything is written
ds = dataset()
for field, value, expect in [("rank", 0, "Network Rank must be at least 1"), ("rank", None, "Network Rank must be a whole number"),
                             ("t_steps", 0, "Training Steps must be at least 1"), ("batch_size", 1.5, "Batch Size must be a whole number"),
                             ("grad_acc", 0, "Gradient Accumulation must be at least 1"), ("lr", "abc", "Learning Rate must be a number"),
                             ("lr", "0", "Learning Rate must be a number above 0"), ("w", 32, "Preview Width must be at least 64"),
                             ("save_steps", -5, "Save Every n Steps must be at least 0")]:
    kw = f"bad_{field}"
    console.clear()
    out = g["start_training"](**args(kw, ds, **{field: value}))
    check(f"{field}={value!r} refused with '{expect}', no run folder", expect in out and not (g["OUTPUT_BASE"] / kw).exists(), out.splitlines()[-1])
console.clear()
out = g["start_training"](**args("good", ds))
check("valid inputs launch", len(launches) == 1 and "Run folder" in out, out.splitlines()[-1])
console.clear()
out = g["start_training"](**args("alpha", ds, rank=16, network_alpha=8))
check("rank/alpha line shows the LoRA scale", "Network rank 16, alpha 8 → LoRA output scale 0.5" in out)


# --- interlock: a dataset tool working on the folder blocks training; another folder only warns about VRAM
def hold(name, folder, seconds=1.5):
    def job():
        yield ""
        time.sleep(seconds)
    msg = tools.start(name, job, folder=str(folder))
    assert msg == "", msg
    time.sleep(0.05)


def wait_tools():
    for _ in range(100):
        if tools.busy() is None:
            return
        time.sleep(0.05)


launches.clear()
hold("Bucketing", ds)
console.clear()
out = g["start_training"](**args("blocked", ds))
check("Start Training refused while Bucketing runs on this dataset", "Bucketing is still running on this dataset" in out
      and not launches and not (g["OUTPUT_BASE"] / "blocked").exists(), out.splitlines()[-1])
wait_tools()
other = dataset()
hold("Bucketing", other)
console.clear()
out = g["start_training"](**args("allowed", ds))
check("…allowed (with a VRAM warning) while Bucketing runs on another dataset",
      len(launches) == 1 and "running on another dataset and shares the GPU" in out, out.splitlines()[:2])
wait_tools()
hold("Dataset inspection", ds)
launches.clear()
console.clear()
g["start_training"](**args("inspect_ok", ds))
check("…and the read-only inspector never blocks training", len(launches) == 1)
wait_tools()

# the other direction: Bucketing refuses the folder a running trainer uses (another folder is fine)
run_dir = sorted((g["OUTPUT_BASE"] / "good").iterdir())[0]
real_running, real_snapshot = tm.is_running, tm.snapshot
tm.is_running = lambda: True
tm.snapshot = lambda: {"state": "running", "step": None, "run_dir": run_dir, "output_name": "good"}
try:
    out = drain(g, g["run_smart_crop_ui"](str(ds), 512, 768, True, True))
    check("Bucketing refused on the dataset being trained", "training is running on this dataset" in out and not (ds / "original_images").exists())
    out = drain(g, g["run_smart_crop_ui"](str(other), 512, 768, True, True))
    check("Bucketing runs on another dataset during training", "Total Dataset Size" in out, out.splitlines()[-1])
    console.clear()
    g["start_bucketing"](str(other), 512, 768, True, True)
    wait_tools()
    check("…with the VRAM warning", "will share the GPU" in console.text())
    console.clear()
    g["start_rename"](str(other))
    wait_tools()
    check("Rename on another dataset: no VRAM warning (it doesn't use the GPU)", "share the GPU" not in console.text() and (other / "001.png").exists())
finally:
    tm.is_running, tm.snapshot = real_running, real_snapshot

# --- Train More: if the trainer can't be launched, the finished run is put back exactly as it was
tm.start = real_start   # the real one fails here: there is no sd-scripts folder in this test tree
kwd = g["OUTPUT_BASE"] / "tm" / "20260101_000000"
(kwd / "configs").mkdir(parents=True)
cfg = {"output_name": "tm", "max_train_steps": 600, "pretrained_model_name_or_path": str(models / "dit.safetensors"),
       "qwen3": str(models / "qwen.safetensors"), "vae": str(models / "vae.safetensors"), "lr_scheduler": "constant"}
toml.dump(cfg, open(kwd / "configs" / "tm_training.toml", "w", encoding="utf-8"))
toml.dump({"datasets": [{"subsets": [{"image_dir": ds.as_posix()}]}]}, open(kwd / "configs" / "tm_dataset.toml", "w", encoding="utf-8"))
(kwd / "tm.safetensors").write_bytes(b"final")
(kwd / "run.json").write_text('{"keyword": "tm", "max_train_steps": 600}', encoding="utf-8")
st = kwd / "tm-state"; st.mkdir()
(st / "train_state.json").write_text('{"current_step": 600}'); (st / "random_states_0.pkl").touch()
snap = {str(p.relative_to(kwd)): p.read_bytes() for p in kwd.rglob("*") if p.is_file()}
console.clear()
out = g["extend_training"]("tm", 700)
now = {str(p.relative_to(kwd)): p.read_bytes() for p in kwd.rglob("*") if p.is_file() and "logs" not in p.parts}
check("Train More launch failure: run restored (final LoRA, config, run.json, no stray backup)", now == snap,
      sorted(set(now) ^ set(snap)))
check("…and says so", "put back as it was" in out, out.splitlines()[-1])
out = g["extend_training"]("tm", 700)
check("…so Train More is still offered for it (not 'use Resume Run')", "isn't finished" not in out, out.splitlines()[-1])

# --- Resume Run: only the keyword's newest started run counts; a newer run with no resume point isn't skipped over
def fake_run(kw, name, state_step=None, final=False, configs=True):
    rd = g["OUTPUT_BASE"] / kw / name
    rd.mkdir(parents=True)
    if configs:
        (rd / "configs").mkdir()
        toml.dump(dict(cfg, output_name=kw, max_train_steps=2400), open(rd / "configs" / f"{kw}_training.toml", "w", encoding="utf-8"))
        toml.dump({"datasets": [{"subsets": [{"image_dir": ds.as_posix()}]}]}, open(rd / "configs" / f"{kw}_dataset.toml", "w", encoding="utf-8"))
    if state_step is not None:
        sd = rd / f"{kw}-step{state_step:08d}-state"
        sd.mkdir()
        (sd / "train_state.json").write_text(json.dumps({"current_step": state_step})); (sd / "random_states_0.pkl").touch()
    if final:
        (rd / f"{kw}.safetensors").write_bytes(b"final")
    return rd


def resume_point(kw):
    point = g["find_resume_point"](g["OUTPUT_BASE"] / kw)
    return point and (point["run_dir"].name, point["step"])


fake_run("rs1", "20260101_000000", state_step=600)
check("resume point: the halted run, at its saved step", resume_point("rs1") == ("20260101_000000", 600), resume_point("rs1"))
fake_run("rs1", "20260102_000000", configs=False)      # a start that failed before its configs were written
check("…a newer folder that never started doesn't hide it", resume_point("rs1") == ("20260101_000000", 600), resume_point("rs1"))
fake_run("rs1", "20260103_000000", state_step=200)
check("…a newer halted run is the one resumed", resume_point("rs1") == ("20260103_000000", 200), resume_point("rs1"))
fake_run("rs2", "20260101_000000", state_step=600)
fake_run("rs2", "20260102_000000")                     # newer run, stopped before its first resume point
check("a newer run with no resume point: the older halted run is not resumed in its place", resume_point("rs2") is None, resume_point("rs2"))
launches.clear()
tm.start = lambda *a, **k: (launches.append(a), True)[1]
console.clear()
out = g["resume_training"]("rs2")
check("…↩ Resume Run says there's nothing to resume and launches nothing", "Nothing to resume for 'rs2'" in out and not launches, out.splitlines()[-1])
console.clear()
out = g["resume_training"]("rs1")
check("↩ Resume Run launches the newest halted run from its state", len(launches) == 1 and "run 20260103_000000 from step 200/2400" in out
      and "--resume" in launches[0][0], out.splitlines()[:2])
fake_run("rs3", "20260101_000000", state_step=600)
fake_run("rs3", "20260102_000000", state_step=2400, final=True)
check("a newer finished run: nothing to resume", resume_point("rs3") is None, resume_point("rs3"))
tm.start = real_start

# --- settings.json: atomic save; an unreadable file is kept and reported, not silently replaced by defaults
sf = g["SETTINGS_FILE"]
good = dict(g["DEFAULT_SETTINGS"], dit_path="D:/models/anima.safetensors", trigger_word="mychar")
g["save_settings"](good)
check("save_settings leaves no temp file", not list(sf.parent.glob("settings.json.tmp")) and json.loads(sf.read_text(encoding="utf-8"))["trigger_word"] == "mychar")
errors = []


def writer(i):
    try:
        for _ in range(30):
            g["save_settings"](dict(good, trigger_word="x" * (i * 7)))
    except Exception as e:
        errors.append(e)


threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
[t.start() for t in threads]
[t.join() for t in threads]
check("concurrent saves (all fields change at once on page load) leave valid JSON",
      not errors and isinstance(json.loads(sf.read_text(encoding="utf-8")), dict), errors[:1])
sf.write_text(json.dumps(good)[:120], encoding="utf-8")   # truncated, e.g. power cut mid-write in an old version
console.clear()
s = g["load_settings"]()
kept = list(sf.parent.glob("settings.unreadable_*.json"))
check("unreadable settings.json: defaults loaded, file kept aside, warning shown",
      s["trigger_word"] == g["DEFAULT_SETTINGS"]["trigger_word"] and len(kept) == 1 and not sf.exists()
      and "could not be read" in console.text() and kept[0].read_text(encoding="utf-8") == json.dumps(good)[:120],
      (len(kept), console.text().splitlines()[-1:]))

# --- preview prompts: the trigger is matched as a tag, not as a text prefix
pd = tmpdir("prompts")
p = g["create_sample_prompts"]("cat", "cat", "catgirl, smile\ncat, sitting\n1girl, Cat, smile\na cat on a sofa", "bad", 1024, 1024, 30, 4.0, 42, pd)
lines = [line.split(" --n ")[0] for line in Path(p).read_text(encoding="utf-8").splitlines()]
check("trigger added unless the prompt has it as a tag",
      lines == ["cat, catgirl, smile", "cat, sitting", "1girl, Cat, smile", "cat, a cat on a sofa"], lines)

# --- WD tagger: tags with parentheses are written plain (sd-scripts reads captions literally)
wd = g["WDTagger"]()
wd.tag_names = ["1girl", "hatsune_miku_(cosplay)".replace("_", " "), "smile", "holding (object)"]
wd.general_indexes, wd.character_indexes = [0, 2, 3], [1]
wd.model = type("M", (), {"get_inputs": lambda self: [type("I", (), {"name": "x"})()],
                          "run": lambda self, a, b: [np.array([[0.9, 0.95, 0.8, 0.7]])]})()
caption = wd.predict(Image.new("RGB", (64, 64)), 0.35, 0.85)
check("WD tagger caption has plain parentheses", caption == "hatsune miku (cosplay), 1girl, smile, holding (object)", caption)

# --- inspector: case-only caption names, case-only image clashes, escaped parentheses
d = tmpdir()
Image.new("RGB", (600, 900)).save(d / "IMG_1.png"); (d / "img_1.txt").write_text("1girl", encoding="utf-8")
Image.new("RGB", (600, 900)).save(d / "Dog.png"); Image.new("RGB", (600, 900)).save(d / "dog.jpg")
(d / "Dog.txt").write_text("hatsune miku \\(cosplay\\), smile", encoding="utf-8")
Image.new("RGB", (600, 900)).save(d / "two.png"); (d / "two.txt").write_text("1girl, smile\nA girl smiling.\n", encoding="utf-8")
out = drain(g, g["inspect_dataset"](str(d), "kw"))
check("inspector: caption named in another letter case is pointed out, not called an orphan",
      "named with different letter case than their image" in out and "img_1.txt" in out and "with no image" not in out,
      [line for line in out.splitlines() if "letter case" in line or "no image" in line])
check("inspector: Dog.png + dog.jpg reported as a clash", "Dog.png + dog.jpg" in out or "dog.jpg + Dog.png" in out)
check("inspector: escaped parentheses flagged", "contain escaped parentheses" in out and "Dog.txt" in out)
check("inspector: multi-line captions flagged (only the first line is trained)", "have more than one line" in out and "two.txt" in out
      and "1 caption(s) have more than one line" in out)

# --- launch / config details
src = (REPO / "app.py").read_text(encoding="utf-8")
check("ui.launch allows serving files from training/output (previews, grid)", re.search(r"ui\.launch\(.*allowed_paths=\[str\(OUTPUT_BASE", src) is not None)
check("weighting_scheme set once", src.count('"weighting_scheme":') == 2 and g["HIDDEN_SETTINGS"]["weighting_scheme"] == "logit_normal")
req = (REPO / "training" / "sd-scripts" / "requirements.txt").read_text(encoding="utf-8").split()
check("requirements list the GUI's direct imports", "psutil" in req and "pandas" in req)

finish()
