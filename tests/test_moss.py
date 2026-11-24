"""MOSS-Image in the app (no GPU, no trainer packages): caption sources and the trigger word, the files a run writes,
model-file checks, memory options, per-model preview settings, resume/Train More choosing the right trainer, settings
from before MOSS-Image support, and the controls shown for each model. What sd-scripts makes of these files is checked
against its real code in test_trainer_moss.py."""
import json
import struct
from pathlib import Path

import toml
from PIL import Image

from _harness import check, finish, load_app, tmpdir

g = load_app()
console, tm = g["console"], g["training_manager"]
work = g["_work"]
NL, TAGS, BOTH = g["CAPTIONS_NL"], g["CAPTIONS_TAGS"], g["CAPTIONS_BOTH"]
ANIMA, MOSS = g["MODEL_ANIMA"], g["MODEL_MOSS"]
OUT = g["OUTPUT_BASE"]


def safetensors_file(path, keys):
    header = json.dumps({k: {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]} for i, k in enumerate(keys)}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * len(keys))
    return path


models = work / "models_for_test"
models.mkdir()
for f in ("dit", "qwen", "vae"):
    (models / f"{f}.safetensors").touch()
moss_dit = safetensors_file(models / "moss_image_base_bf16.safetensors", ["layers.0.proj.q.weight"])
moss_dit_int8 = safetensors_file(models / "moss_image_base_int8_convrot.safetensors", ["layers.0.proj.q.weight", "layers.0.proj.q.comfy_quant"])
moss_te = safetensors_file(models / "qwen3.5_9b_int8_convrot.safetensors", ["layers.0.mlp.gate_proj.weight", "layers.0.mlp.gate_proj.comfy_quant"])
moss_te_bf16 = safetensors_file(models / "qwen3.5_9b_bf16.safetensors", ["layers.0.mlp.gate_proj.weight"])
moss_vae = safetensors_file(models / "flux2-vae.safetensors", ["encoder.conv_in.weight"])


def dataset():
    """a: both captions; b: both, and its natural-language caption names the trigger; c: tags only; d: natural language only."""
    d = tmpdir()
    Image.new("RGB", (512, 768)).save(d / "a.png")
    (d / "a.txt").write_text("1girl, smile\nthis second line is not part of the caption", encoding="utf-8")
    (d / "a.toriigate.txt").write_text("A girl smiles.\n\nSecond paragraph.\n", encoding="utf-8")
    Image.new("RGB", (768, 512)).save(d / "b.jpg")
    (d / "b.txt").write_text("1girl, trig, standing", encoding="utf-8")
    (d / "b.toriigate.txt").write_text("Trig stands in the rain; a striking image.", encoding="utf-8")
    Image.new("RGB", (640, 640)).save(d / "c.png")
    (d / "c.txt").write_text("1girl, sitting", encoding="utf-8")
    Image.new("RGB", (512, 512)).save(d / "d.webp")
    (d / "d.toriigate.txt").write_text("A cat sleeps on a windowsill, striking a pose.", encoding="utf-8")
    return d


def captions(metadata):
    return {Path(k).name: v["caption"] for k, v in metadata.items()}


# --- caption sources ---------------------------------------------------------------------------------------------------
ds = dataset()
meta, stats = g["build_moss_metadata"](str(ds), "trig", BOTH)
caps = captions(meta)
check("both: an image with both captions gets two candidates, natural language first, line breaks kept, only the first "
      "line of the tag file", caps["a.png"] == ["trig, A girl smiles.\n\nSecond paragraph.", "trig, 1girl, smile"], caps["a.png"])
check("both: the trigger is added only where it's missing (word match in sentences, any letter case; tag match in tags)",
      caps["b.jpg"] == ["Trig stands in the rain; a striking image.", "1girl, trig, standing"], caps["b.jpg"])
check("both: images with one kind of caption train on that one", caps["c.png"] == "trig, 1girl, sitting"
      and caps["d.webp"] == "trig, A cat sleeps on a windowsill, striking a pose.", (caps["c.png"], caps["d.webp"]))
check("a trigger inside another word doesn't count as present ('trig' in 'striking')", caps["d.webp"].startswith("trig, "))
check("both: the report names the images that have only one kind", stats["both"] == 2 and stats["only_tags"] == ["c.png"]
      and stats["only_nl"] == ["d.webp"] and stats["missing"] == [] and stats["candidates"] == 6 and stats["prefixed"] == 4, stats)
check("image sizes are recorded as stored (width, height)", meta[(ds / "b.jpg").resolve().as_posix()]["image_size"] == [768, 512])

meta, stats = g["build_moss_metadata"](str(ds), "trig", NL)
check("natural language only: images without a .toriigate.txt are reported, the rest carry one caption",
      stats["missing"] == ["c.png"] and captions(meta)["a.png"] == "trig, A girl smiles.\n\nSecond paragraph." and len(meta) == 3, stats["missing"])
meta, stats = g["build_moss_metadata"](str(ds), "trig", TAGS)
check("tags only: images without a .txt are reported", stats["missing"] == ["d.webp"] and captions(meta)["a.png"] == "trig, 1girl, smile", stats["missing"])
meta, stats = g["build_moss_metadata"](str(ds), "", BOTH)
check("no trigger word: captions are used as written", captions(meta)["a.png"] == ["A girl smiles.\n\nSecond paragraph.", "1girl, smile"]
      and stats["prefixed"] == 0, captions(meta)["a.png"])
(ds / "c.toriigate.txt").write_text("   \n", encoding="utf-8")
meta, stats = g["build_moss_metadata"](str(ds), "trig", NL)
check("an empty caption file counts as missing", stats["missing"] == ["c.png"], stats["missing"])
(ds / "c.toriigate.txt").write_text("A girl sits.\n---===--- not a divider I meant\nMore.", encoding="utf-8")
try:
    g["build_moss_metadata"](str(ds), "trig", NL)
    refused = ""
except ValueError as e:
    refused = str(e)
check("a caption line that sd-scripts would read as a candidate divider is refused by name", "c.png" in refused and "---===---" in refused, refused)
(ds / "c.toriigate.txt").unlink()

# --- starting a run -----------------------------------------------------------------------------------------------------
launches = []
tm.start = lambda *a, **k: (launches.append((a, k)), True)[1]   # record instead of launching a trainer


def args(kw, d, **over):
    a = dict(trigger_word=kw, dataset_path=str(d), dit_p=str(models / "dit.safetensors"), qwen_p=str(models / "qwen.safetensors"),
             vae_p=str(models / "vae.safetensors"), rank=16, lr="1.0", optimizer="Prodigy", t_steps=300, save_steps=100,
             sample_steps=50, pos="a portrait\ntrig in the rain", neg="anima negative", w=1024, h=1024, s_steps_gen=30, s_cfg=4.0,
             s_seed=42, train_seed=42, batch_size=1, grad_acc=1, save_resume_state=True, blocks_to_swap=7, network_alpha=0,
             model_family=MOSS, moss_dit_p=str(moss_dit), moss_te_p=str(moss_te), moss_vae_p=str(moss_vae), moss_tokenizer_dir="",
             moss_caption_source=BOTH, moss_convrot_int8=True, moss_convrot_int8_te=True, moss_blocks_to_swap=20,
             moss_h2d_only=True, moss_te_blocks_to_swap=0, moss_neg="", moss_s_steps_gen=50, moss_s_cfg=3.0)
    a.update(over)
    return a


def start(kw, d=None, **over):
    console.clear()
    return g["start_training"](**args(kw, d or ds, **over))


def run_files(kw):
    run = sorted((OUT / kw).iterdir())[-1]
    cfg = toml.load(str(next((run / "configs").glob("*_training.toml"))))
    sub = toml.load(str(next((run / "configs").glob("*_dataset.toml"))))["datasets"][0]["subsets"][0]
    return run, cfg, sub


for field, value, expect in [("moss_blocks_to_swap", 41, "Block Swap can be at most 40"), ("moss_te_blocks_to_swap", 33, "Text Encoder Block Swap can be at most 32"),
                             ("moss_caption_source", "nonsense", "Unknown caption source"), ("moss_s_steps_gen", 0, "Preview Steps must be at least 1"),
                             ("model_family", "SDXL", "Unknown model")]:
    out = start(f"bad_{field}", **{field: value})
    check(f"{field}={value!r} refused with '{expect}', no run folder", expect in out and not (OUT / f"bad_{field}").exists(), out.splitlines()[-1])

out = start("nomodel", moss_dit_p=str(models / "nope.safetensors"), moss_te_p="")
check("missing MOSS-Image model files are named, pointing at the MOSS-Image paths section", "MOSS-Image DiT file not found" in out
      and "Qwen3.5-9B text encoder file not found" in out and "Paths to MOSS-Image Models" in out and not (OUT / "nomodel").exists(), out.splitlines()[:2])
out = start("w4a8", moss_te_p=str(safetensors_file(models / "qwen3.5_9b_w4a8_gptq.safetensors", ["x"])))
check("the W4A8 GPTQ text encoder, which sd-scripts can't load, is refused", "W4A8" in out and not (OUT / "w4a8").exists(), out.splitlines()[0])
shard2 = safetensors_file(models / "diffusion_pytorch_model-00002-of-00003.safetensors", ["x"])
out = start("shard", moss_dit_p=str(shard2))
check("a later part of a split checkpoint is refused, naming the first part", "give the first part" in out
      and "diffusion_pytorch_model-00001-of-00003.safetensors" in out, out.splitlines()[0])
shard1 = safetensors_file(models / "diffusion_pytorch_model-00001-of-00003.safetensors", ["x"])
out = start("shard", moss_dit_p=str(shard1))
check("a split checkpoint with parts missing is refused, naming them", "split checkpoint is incomplete" in out
      and "diffusion_pytorch_model-00003-of-00003.safetensors" in out and not (OUT / "shard").exists(), out.splitlines()[0])
out = start("nocap", moss_caption_source=NL)
check("natural-language source: an image without a .toriigate.txt stops the run and points at the captioner",
      "c.png" in out and "start_captioner.bat" in out and not (OUT / "nocap").exists(), out.splitlines()[-1])
check("nothing was launched by any refused start", launches == [], len(launches))

out = start("trig")
run, cfg, sub = run_files("trig")
check("a valid MOSS-Image run launches the MOSS-Image trainer", len(launches) == 1
      and launches[0][0][0][launches[0][0][0].index("--config_file") - 1].endswith("moss_image_train_network.py"), launches[0][0][0][:7] if launches else None)
check("training config: MOSS-Image's model files and network, none of Anima's keys",
      cfg["pretrained_model_name_or_path"] == moss_dit.resolve().as_posix() and cfg["text_encoder"] == moss_te.resolve().as_posix()
      and cfg["vae"] == moss_vae.resolve().as_posix() and cfg["network_module"] == "networks.lora_generic"
      and not {"qwen3", "sigmoid_scale", "weighting_scheme", "vae_chunk_size", "vae_disable_cache"} & set(cfg), sorted(cfg))
check("MOSS-Image's block swap is used, not Anima's slider (20, not 7), with H2D-only", cfg["blocks_to_swap"] == 20 and cfg["block_swap_h2d_only"] is True, cfg.get("blocks_to_swap"))
check("bf16 DiT + INT8 ticked: quantized while loading; pre-quantized text encoder: no option needed",
      cfg.get("convrot_int8") is True and "convrot_int8_text_encoder" not in cfg, {k: v for k, v in cfg.items() if "int8" in k})
check("the log says which precision each model runs in", "DiT: ConvRot INT8 (quantized while loading) · Text encoder: ConvRot INT8 (pre-quantized file)" in out,
      [l for l in out.splitlines() if "DiT:" in l])
check("dataset config: a metadata subset of the dataset folder, multi-line captions on, no caption prefix",
      Path(sub["image_dir"]) == ds.resolve() and Path(sub["metadata_file"]).parent == run / "configs" and sub["enable_multiline_caption"] is True
      and "caption_prefix" not in sub and "caption_extension" not in sub and sub["num_repeats"] == 75, sub)
check("the run's dataset folder is found from its config (training ⇄ tool interlock)", g["run_dataset_dir"](run) == ds.resolve(), g["run_dataset_dir"](run))
check("the log reports the caption mix", "2 image(s) have both captions" in out and "only a natural-language caption" in out and "d.webp" in out
      and "only a tag caption" in out and "c.png" in out, [l for l in out.splitlines() if "caption" in l])
prompts = next((run / "configs").glob("*_prompts.txt")).read_text(encoding="utf-8").splitlines()
check("previews: MOSS-Image's own steps/CFG, no negative prompt when its field is empty (Anima's negative isn't used)",
      prompts == ["trig, a portrait --w 1024 --h 1024 --l 3.0 --s 50 --d 42", "trig in the rain --w 1024 --h 1024 --l 3.0 --s 50 --d 42"], prompts)
info = json.loads((run / "run.json").read_text(encoding="utf-8"))
check("run.json records the model and the caption source", info["model_family"] == MOSS and info["caption_source"] == BOTH, info)

out = start("opts", moss_dit_p=str(moss_dit_int8), moss_te_p=str(moss_te_bf16), moss_convrot_int8=False, moss_convrot_int8_te=False,
            moss_blocks_to_swap=0, moss_te_blocks_to_swap=28, moss_neg="blurry", moss_s_steps_gen=20, moss_s_cfg=4.5,
            moss_tokenizer_dir=str(models), moss_caption_source=TAGS, dataset_path=str(ds), optimizer="AdamW8bit", lr="1e-4")
check("tags source refuses the image without a .txt", "d.webp" in out and not (OUT / "opts").exists(), out.splitlines()[-1])
(ds / "d.txt").write_text("cat, sleeping", encoding="utf-8")
out = start("opts", moss_dit_p=str(moss_dit_int8), moss_te_p=str(moss_te_bf16), moss_convrot_int8=False, moss_convrot_int8_te=False,
            moss_blocks_to_swap=0, moss_te_blocks_to_swap=28, moss_neg="blurry", moss_s_steps_gen=20, moss_s_cfg=4.5,
            moss_tokenizer_dir=str(models), moss_caption_source=TAGS, optimizer="AdamW8bit", lr="1e-4")
run2, cfg2, sub2 = run_files("opts")
check("no block swap: neither swap option written; text-encoder swap and tokenizer folder are", "blocks_to_swap" not in cfg2
      and "block_swap_h2d_only" not in cfg2 and cfg2["text_encoder_blocks_to_swap"] == 28 and cfg2["tokenizer_cache_dir"] == models.resolve().as_posix(),
      {k: v for k, v in cfg2.items() if "swap" in k or "tokenizer" in k})
check("INT8 unticked + bf16 text encoder: no quantization option; a pre-quantized DiT is still reported as INT8",
      "convrot_int8" not in cfg2 and "convrot_int8_text_encoder" not in cfg2
      and "DiT: ConvRot INT8 (pre-quantized file) · Text encoder: bf16" in out, [l for l in out.splitlines() if "DiT:" in l])
check("optimizer and learning rate are the shared ones", cfg2["optimizer_type"] == "AdamW8bit" and cfg2["learning_rate"] == 1e-4 and cfg2["lr_scheduler"] == "cosine")
prompts2 = next((run2 / "configs").glob("*_prompts.txt")).read_text(encoding="utf-8").splitlines()
check("a MOSS-Image negative prompt and preview settings are written when given", prompts2[0] == "opts, a portrait --n blurry --w 1024 --h 1024 --l 4.5 --s 20 --d 42", prompts2)
meta2 = json.loads(Path(sub2["metadata_file"]).read_text(encoding="utf-8"))
check("tags source: one caption per image", all(isinstance(v["caption"], str) for v in meta2.values()) and len(meta2) == 4, captions(meta2))
(ds / "d.txt").unlink()

# --- Anima is untouched ----------------------------------------------------------------------------------------------------
ads = tmpdir()
for i in range(3):
    Image.new("RGB", (512, 768)).save(ads / f"{i}.png")
    (ads / f"{i}.txt").write_text("1girl", encoding="utf-8")
n = len(launches)
out = start("ani", ads, model_family=ANIMA)
run_a, cfg_a, sub_a = run_files("ani")
check("an Anima run launches the Anima trainer with Anima's files, block swap (7) and preview settings",
      len(launches) == n + 1 and launches[-1][0][0][launches[-1][0][0].index("--config_file") - 1].endswith("anima_train_network.py")
      and cfg_a["qwen3"].endswith("qwen.safetensors") and cfg_a["network_module"] == "networks.lora_anima" and cfg_a["blocks_to_swap"] == 7
      and "text_encoder" not in cfg_a and "convrot_int8" not in cfg_a and sub_a["caption_extension"] == ".txt" and "metadata_file" not in sub_a,
      {k: cfg_a.get(k) for k in ("network_module", "blocks_to_swap")})
check("…and Anima's negative prompt, steps and CFG", next((run_a / "configs").glob("*_prompts.txt")).read_text(encoding="utf-8").splitlines()[0]
      == "ani, a portrait --n anima negative --w 1024 --h 1024 --l 4.0 --s 30 --d 42")
out = g["start_training"](**{k: v for k, v in args("ani2", ads).items() if not k.startswith("moss_") and k != "model_family"})
check("start_training called the pre-MOSS way (no model given) trains Anima", toml.load(str(next((sorted((OUT / "ani2").iterdir())[-1] / "configs").glob("*_training.toml"))))["network_module"] == "networks.lora_anima")

# --- which trainer continues a run ----------------------------------------------------------------------------------------
check("a run's model is read from run.json", g["run_model_family"](run) == MOSS and g["run_model_family"](run_a) == ANIMA)
(run_a / "run.json").write_text(json.dumps({"keyword": "ani"}), encoding="utf-8")       # a run from before MOSS-Image support
check("a run without the marker is Anima; a MOSS-Image config is recognised even without run.json",
      g["run_model_family"](run_a, cfg_a) == ANIMA and g["run_model_family"](work / "nowhere", cfg) == MOSS)
check("each model's files are looked up under its own config keys", g["model_file_keys"](MOSS) == ("pretrained_model_name_or_path", "text_encoder", "vae")
      and g["model_file_keys"](ANIMA) == ("pretrained_model_name_or_path", "qwen3", "vae"))


def fake_state(run_dir, name, step):
    st = run_dir / name
    st.mkdir()
    (st / "train_state.json").write_text(json.dumps({"current_epoch": 1, "current_step": step}), encoding="utf-8")
    (st / "random_states_0.pkl").write_bytes(b"rng")


fake_state(run, "trig-step00000100-state", 100)
(ds / "c.toriigate.txt").write_text("A girl sits by a window.", encoding="utf-8")          # captioned since the run started
n = len(launches)
console.clear()
out = g["resume_training"]("trig")
cmd = launches[-1][0][0]
meta = json.loads(Path(sub["metadata_file"]).read_text(encoding="utf-8"))
check("Resume Run continues a MOSS-Image run with the MOSS-Image trainer from its state", len(launches) == n + 1
      and cmd[cmd.index("--config_file") - 1].endswith("moss_image_train_network.py") and cmd[cmd.index("--resume") + 1].endswith("trig-step00000100-state")
      and "(MOSS-Image)" in out, out.splitlines()[0])
check("…reading the captions as they are now", captions(meta)["c.png"] == ["trig, A girl sits by a window.", "trig, 1girl, sitting"], captions(meta)["c.png"])
(ds / "a.txt").unlink(); (ds / "a.toriigate.txt").unlink()
n = len(launches)
console.clear()
out = g["resume_training"]("trig")
check("Resume Run refuses when an image has lost its captions, instead of training without it", len(launches) == n and "a.png" in out
      and "no caption for this run's caption source" in out, out.splitlines()[-1])
(ds / "a.txt").write_text("1girl, smile", encoding="utf-8")
Path(cfg["text_encoder"]).rename(models / "moved_te.safetensors")
console.clear()
out = g["resume_training"]("trig")
check("Resume Run names a MOSS-Image model file that has gone", len(launches) == n and "Model file not found" in out and "qwen3.5_9b_int8_convrot" in out, out.splitlines()[0])
(models / "moved_te.safetensors").rename(cfg["text_encoder"])

(run / "trig.safetensors").write_bytes(b"final")
fake_state(run, "trig-state", 300)
n = len(launches)
console.clear()
out = g["extend_training"]("trig", 400)
cmd = launches[-1][0][0]
check("Train More extends a finished MOSS-Image run with the MOSS-Image trainer", len(launches) == n + 1
      and cmd[cmd.index("--config_file") - 1].endswith("moss_image_train_network.py") and cmd[cmd.index("--resume") + 1].endswith("trig-state")
      and toml.load(str(next((run / "configs").glob("*_training.toml"))))["max_train_steps"] == 400, out.splitlines()[0])

# --- model file helpers -------------------------------------------------------------------------------------------------------
check("pre-quantized INT8 files are recognised from their header alone; others, empty and missing files are not",
      g["is_convrot_int8_file"](moss_dit_int8) and g["is_convrot_int8_file"](moss_te) and not g["is_convrot_int8_file"](moss_dit)
      and not g["is_convrot_int8_file"](models / "dit.safetensors") and not g["is_convrot_int8_file"](models / "nope.safetensors"))
for part in ("00002", "00003"):
    safetensors_file(models / f"diffusion_pytorch_model-{part}-of-00003.safetensors", ["x"])
check("a complete split checkpoint is accepted through its first part", g["missing_shards"](shard1) == [] and g["missing_shards"](moss_dit) == []
      and g["moss_model_errors"](str(shard1), str(moss_te), str(moss_vae)) == [])
check("a tokenizer folder that doesn't exist is reported", "Tokenizer folder not found" in " ".join(g["moss_model_errors"](str(moss_dit), str(moss_te), str(moss_vae), str(models / "no_tok"))))

# --- settings and the UI ------------------------------------------------------------------------------------------------------
defaults = g["DEFAULT_SETTINGS"]
keys = list(defaults)
check("MOSS-Image settings come after every earlier key (settings are matched to UI components by position)",
      keys.index("model_family") == keys.index("network_alpha") + 1 and all(k.startswith("moss_") for k in keys[keys.index("model_family") + 1:]), keys[-15:])
check("defaults: Anima stays the model; MOSS-Image starts with INT8 files, block swap 20 and the official preview settings",
      defaults["model_family"] == ANIMA and defaults["moss_convrot_int8"] and defaults["moss_convrot_int8_te"] and defaults["moss_blocks_to_swap"] == 20
      and defaults["moss_dit_path"].endswith("moss_image_base_int8_convrot.safetensors") and defaults["moss_te_path"].endswith("qwen3.5_9b_int8_convrot.safetensors")
      and (defaults["moss_neg_prompt"], defaults["moss_sample_steps_gen"], defaults["moss_sample_cfg"]) == ("", 50, 3.0))
old = {k: v for k, v in defaults.items() if k != "model_family" and not k.startswith("moss_")}
old["trigger_word"] = "from_1_13"
g["SETTINGS_FILE"].write_text(json.dumps(old), encoding="utf-8")
loaded = g["load_settings"]()
check("a settings.json from before MOSS-Image support loads: its values kept, MOSS-Image defaults filled in", loaded["trigger_word"] == "from_1_13"
      and loaded["model_family"] == ANIMA and loaded["moss_blocks_to_swap"] == 20 and list(loaded) == keys)
g["auto_save_state"](*[("MOSS-Image" if k == "model_family" else loaded[k]) for k in keys])
check("the model choice is saved with the other settings", json.loads(g["SETTINGS_FILE"].read_text(encoding="utf-8"))["model_family"] == MOSS)


def shown(updates):
    return [u.get("visible") for u in updates]


H = "hidden"   # single controls stay in the page while hidden; the three groups are switched with True/False
check("choosing MOSS-Image shows its paths, options and preview fields and hides Anima's; choosing Anima does the reverse",
      shown(g["model_family_visibility"](MOSS)) == [False, True, H, True, H, H, H, True, True, True]
      and shown(g["model_family_visibility"](ANIMA)) == [True, False, True, False, True, True, True, H, H, H],
      shown(g["model_family_visibility"](MOSS)))
finish()
