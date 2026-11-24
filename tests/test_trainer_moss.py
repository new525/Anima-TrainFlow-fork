"""MOSS-Image against sd-scripts' real code, without a GPU or model files.

The configs TrainFlow writes for a MOSS-Image run (and, for comparison, an Anima run) are read by the real trainers'
argument parsers: every key must be an option that trainer knows (sd-scripts ignores unknown keys in a config file
silently, so a misspelt option would otherwise just not happen), and every value must be one the option accepts. The
MOSS-Image trainer's own start-up checks then run on the dataset TrainFlow describes, and its real caching code runs
with stand-in encoders: both caption candidates of an image are encoded with the trigger word, a sample picks between
them, edited captions and edited images are encoded again, and the caches of the two models don't collide.

Needs the trainer's packages (torch, transformers, diffusers…). On Windows this script re-runs itself with
python_embeded\\python.exe when the current Python doesn't have them; otherwise it reports SKIP."""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
SD = REPO / "training" / "sd-scripts"
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import diffusers  # noqa: F401
except ImportError:
    embedded = REPO / "python_embeded" / ("python.exe" if os.name == "nt" else "bin/python")
    if embedded.exists() and Path(sys.executable).resolve() != embedded.resolve() and not os.environ.get("TF_REEXEC"):
        sys.exit(subprocess.run([str(embedded), str(Path(__file__).resolve())], env=dict(os.environ, TF_REEXEC="1")).returncode)
    print("SKIP: needs the trainer's packages (torch etc.): run with python_embeded\\python.exe, or pip install the "
          "packages from training/sd-scripts/requirements.txt")
    sys.exit(0)

import numpy as np
import toml
import torch
from PIL import Image

sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SD))
from _harness import check, finish  # noqa: E402

# TrainFlow's own config writers (app.py executed with the real torch)
work = Path(tempfile.mkdtemp(prefix="tf_moss_app_"))
shutil.copy(REPO / "app.py", work / "app.py")
shutil.copy(REPO / "embedded_tags.py", work / "embedded_tags.py")
app = {"__name__": "app", "__file__": str(work / "app.py")}
exec(compile((REPO / "app.py").read_text(encoding="utf-8").replace('if __name__ == "__main__":', "if False:"), str(REPO / "app.py"), "exec"), app)
NL, TAGS, BOTH = app["CAPTIONS_NL"], app["CAPTIONS_TAGS"], app["CAPTIONS_BOTH"]

import anima_train_network  # noqa: E402
import moss_image_train_network  # noqa: E402
import library.args as args_util  # noqa: E402
from library import config_util, strategy_anima, strategy_base, strategy_moss_image  # noqa: E402

# --- a dataset with both kinds of caption --------------------------------------------------------------------------------
root = Path(tempfile.mkdtemp(prefix="tf_moss_"))
ds, cfg_dir, models = root / "ds", root / "cfg", root / "models"
for d in (ds, cfg_dir, models):
    d.mkdir()
Image.new("RGB", (512, 768), (200, 30, 30)).save(ds / "a.png")
(ds / "a.txt").write_text("1girl, smile\nsecond line is ignored", encoding="utf-8")
(ds / "a.toriigate.txt").write_text("A girl smiles at the viewer.\n\nSoft daylight falls from the left.", encoding="utf-8")
Image.new("RGB", (768, 512), (30, 30, 200)).save(ds / "b.png")
(ds / "b.txt").write_text("1girl, standing", encoding="utf-8")
(ds / "b.toriigate.txt").write_text("mytrigger stands in the rain.", encoding="utf-8")
Image.new("RGB", (512, 512), (30, 200, 30)).save(ds / "c.png")
(ds / "c.txt").write_text("mytrigger, 1girl, sitting", encoding="utf-8")       # no natural-language caption


def safetensors_file(path, keys):
    header = json.dumps({k: {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]} for i, k in enumerate(keys)}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * len(keys))
    return path


dit = safetensors_file(models / "moss_image_base_bf16.safetensors", ["layers.0.proj.q.weight"])
dit_int8 = safetensors_file(models / "moss_image_base_int8_convrot.safetensors",
                            ["layers.0.proj.q.weight", "layers.0.proj.q.weight_scale", "layers.0.proj.q.comfy_quant"])
te = safetensors_file(models / "model-00001-of-00002.safetensors", ["model.language_model.layers.0.x"])
safetensors_file(models / "model-00002-of-00002.safetensors", ["model.language_model.layers.1.x"])
vae = safetensors_file(models / "flux2-vae.safetensors", ["encoder.conv_in.weight"])
anima = {k: str(safetensors_file(models / f"anima_{k}.safetensors", ["w"])) for k in ("dit_path", "qwen_path", "vae_path")}


def write_moss_configs(source=BOTH, trigger="mytrigger", dit_path=dit, swap=20, optimizer="Prodigy", lr="1.0", **moss):
    metadata, stats = app["build_moss_metadata"](str(ds), trigger, source)
    meta_path = cfg_dir / "kw_metadata.json"
    app["write_moss_metadata"](metadata, meta_path)
    ds_toml = app["create_moss_dataset_toml"]("kw", str(ds), meta_path, len(metadata), 640, 768, cfg_dir, 100, 1, 1)
    prompts = app["create_sample_prompts"]("kw", trigger, "a portrait\nmytrigger in the rain", "", 1024, 1024, 50, 3.0, 42, cfg_dir,
                                           model_family=app["MODEL_MOSS"])
    options = {"convrot_int8": True, "convrot_int8_te": True, "h2d_only": True, "te_blocks_to_swap": 28, "tokenizer_dir": "", **moss}
    tr_toml = app["create_training_toml"]("kw", cfg_dir, root / "out", 16, lr, optimizer, 100, 50, 50,
                                          {"dit_path": str(dit_path), "te_path": str(te), "vae_path": str(vae)}, prompts, 42, 1, 1,
                                          root / "out" / ".halt_request", True, swap, 0, model_family=app["MODEL_MOSS"], moss=options)
    return tr_toml, ds_toml, metadata, stats


def read_like_the_trainer(module, tr_toml, ds_toml):
    """What `<trainer>.py --config_file … --dataset_config …` does before it trains."""
    parser = module.setup_parser()
    argv, sys.argv = sys.argv, [module.__file__, "--config_file", str(tr_toml), "--dataset_config", str(ds_toml)]
    try:   # read_config_from_file parses the process's own command line a second time, on top of the config file
        args = parser.parse_args()
        args_util.verify_command_line_training_args(args)
        return parser, args_util.read_config_from_file(args, parser)
    finally:
        sys.argv = argv


def config_problems(parser, tr_toml):
    """Keys the trainer doesn't have, and values outside an option's choices."""
    actions = {a.dest: a for a in parser._actions}
    problems = []
    for key, value in toml.load(str(tr_toml)).items():
        if key not in actions:
            problems.append(f"unknown option: {key}")
        elif actions[key].choices is not None and value not in actions[key].choices:
            problems.append(f"{key}={value!r} not in {list(actions[key].choices)}")
    return problems


# --- the configs are ones the real trainers accept ----------------------------------------------------------------------
tr_toml, ds_toml, metadata, stats = write_moss_configs()
parser, args = read_like_the_trainer(moss_image_train_network, tr_toml, ds_toml)
check("MOSS-Image config: every key is an option of moss_image_train_network.py, every value an accepted one",
      config_problems(parser, tr_toml) == [], config_problems(parser, tr_toml))
check("…and the same config is NOT valid for the Anima trainer (the two are kept apart)",
      any("text_encoder" in p_ for p_ in config_problems(anima_train_network.setup_parser(), tr_toml)))
check("MOSS-Image options reach the trainer", args.network_module == "networks.lora_generic" and args.text_encoder == te.as_posix()
      and args.convrot_int8 is True and args.convrot_int8_text_encoder is True and args.blocks_to_swap == 20
      and args.block_swap_h2d_only is True and args.text_encoder_blocks_to_swap == 28 and args.gradient_checkpointing is True
      and args.network_train_unet_only is True and args.cache_text_encoder_outputs is True and args.mixed_precision == "bf16",
      {k: getattr(args, k) for k in ("network_module", "convrot_int8", "blocks_to_swap", "block_swap_h2d_only", "text_encoder_blocks_to_swap")})
check("the pre-training timestep distribution is requested", (args.timestep_sampling, args.logit_mean, args.logit_std, args.discrete_flow_shift)
      == ("truncated_logit_normal", -1.09, 0.95, 1.0), (args.timestep_sampling, args.logit_mean, args.logit_std, args.discrete_flow_shift))
check("the halt-request hook is an option of the MOSS-Image trainer too", args.halt_request_file.endswith(".halt_request"))
check("resume points are requested the same way as for Anima", args.save_state is True and args.save_every_n_steps == 50
      and args.save_last_n_steps_state == 50)

a_prompts = app["create_sample_prompts"]("kw", "mytrigger", "x", "bad", 1024, 1024, 30, 4.0, 42, cfg_dir)
a_ds = app["create_dataset_toml"]("kwa", str(ds), "mytrigger", 640, 768, cfg_dir, 100, 1, 1)
a_tr = app["create_training_toml"]("kwa", cfg_dir, root / "out", 32, "1.0", "Prodigy", 100, 50, 50, anima, a_prompts, 42, 1, 1,
                                   root / "out" / ".halt_request", True, 4, 0)
a_parser, a_args = read_like_the_trainer(anima_train_network, a_tr, a_ds)
check("Anima config: still made of options anima_train_network.py has, after the sd-scripts update",
      config_problems(a_parser, a_tr) == [], config_problems(a_parser, a_tr))
check("Anima options reach the trainer", a_args.network_module == "networks.lora_anima" and a_args.qwen3 == Path(anima["qwen_path"]).as_posix()
      and a_args.blocks_to_swap == 4 and a_args.halt_request_file.endswith(".halt_request"))

# pre-quantized files need no quantization option; other optimizers and no swap produce valid configs as well
tr2, _, _, _ = write_moss_configs(dit_path=dit_int8, swap=0, optimizer="AdamW8bit", lr="1e-4", te_blocks_to_swap=0, tokenizer_dir=str(models))
p2, args2 = read_like_the_trainer(moss_image_train_network, tr2, ds_toml)
check("a pre-quantized INT8 DiT is left to sd-scripts' own detection (no --convrot_int8), nothing swapped when swap is 0",
      config_problems(p2, tr2) == [] and args2.convrot_int8 is False and args2.blocks_to_swap is None
      and args2.block_swap_h2d_only is False and args2.text_encoder_blocks_to_swap is None, (args2.convrot_int8, args2.blocks_to_swap))
check("a tokenizer folder is passed on for offline use", args2.tokenizer_cache_dir == models.as_posix(), args2.tokenizer_cache_dir)
tr_toml, ds_toml, metadata, stats = write_moss_configs()   # back to the first configuration
parser, args = read_like_the_trainer(moss_image_train_network, tr_toml, ds_toml)

# --- the real dataset, the trainer's own start-up checks ----------------------------------------------------------------
encoded_text, encoded_images = [], []


class Tok(strategy_base.TokenizeStrategy):
    def tokenize(self, text):
        texts = text if isinstance(text, list) else [text]
        encoded_text.extend(texts)
        return [texts]


class Enc(strategy_moss_image.MossImageTextEncodingStrategy):
    """Like the real Qwen3.5 strategy: one embedding per caption, padded to the longest, with its mask."""
    def encode_tokens(self, tokenize_strategy, models, tokens):
        texts = tokens[0]
        lengths = [3 + len(t) % 5 for t in texts]
        embeds = torch.zeros(len(texts), max(lengths), 8, dtype=torch.bfloat16)
        mask = torch.zeros(len(texts), max(lengths), dtype=torch.int32)
        for i, (t, n) in enumerate(zip(texts, lengths)):
            embeds[i, :n] = float(len(t))      # the caption's length marks its embedding
            mask[i, :n] = 1
        return [embeds, mask]


class FakeVAE:
    device, dtype = torch.device("cpu"), torch.float32

    def encode(self, img):                        # FLUX.2 VAE stand-in: 32 channels at 1/8 resolution
        encoded_images.append(img.shape[0])
        return torch.nn.functional.avg_pool2d(img, 8).mean(1, keepdim=True).repeat(1, 32, 1, 1)


trainer = moss_image_train_network.MossImageNetworkTrainer()
strategy_base.TokenizeStrategy.set_strategy(Tok())
strategy_base.TextEncodingStrategy.set_strategy(Enc())
strategy_base.LatentsCachingStrategy.set_strategy(trainer.get_latents_caching_strategy(args))
strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(trainer.get_text_encoder_outputs_caching_strategy(args))
acc = types.SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, wait_for_everyone=lambda: None,
                            device=torch.device("cpu"))


def build_group(arguments):
    gen = config_util.BlueprintGenerator(config_util.ConfigSanitizer(True, True, arguments.masked_loss, True))
    group, _ = config_util.generate_dataset_group_by_blueprint(gen.generate(config_util.load_user_config(arguments.dataset_config), arguments).dataset_group)
    group.set_current_strategies()
    return group


def training_run(source=BOTH, trigger="mytrigger"):
    """What one 🚀 Start Training does to the caches: write the configs, build the dataset, cache latents and text."""
    global tr_toml, ds_toml, metadata, stats, parser, args
    tr_toml, ds_toml, metadata, stats = write_moss_configs(source=source, trigger=trigger)
    parser, args = read_like_the_trainer(moss_image_train_network, tr_toml, ds_toml)
    group = build_group(args)
    encoded_text.clear(); encoded_images.clear()
    group.new_cache_latents(FakeVAE(), acc)
    group.new_cache_text_encoder_outputs([None], acc)
    return group


def te_npz(stem):
    return ds / "cache_text_encoder" / f"{stem}_moss_te.npz"


def latent_npz(stem):
    return next((ds / "latent_cache").glob(f"{stem}_*_moss.npz"))


group = build_group(args)
dataset = group.datasets[0]
check("sd-scripts reads TrainFlow's metadata file: three images, in the dataset folder", type(dataset).__name__ == "FineTuningDataset"
      and sorted(Path(i.absolute_path).name for i in dataset.image_data.values()) == ["a.png", "b.png", "c.png"], type(dataset).__name__)
try:
    trainer.assert_extra_args(args, group, None)
    problem = None
except AssertionError as e:
    problem = str(e)
check("the MOSS-Image trainer's own start-up checks pass on this dataset and these options (caching with caption "
      "candidates and dropout, bucket sizes in multiples of 16, block swap with gradient checkpointing)", problem is None, problem)
check("the run trains the DiT only: the text encoder stays frozen and its outputs are cached",
      trainer.is_train_text_encoder(args) is False and trainer.is_text_encoder_not_needed_for_training(args) is True)
bucket_sizes = {tuple(i.bucket_reso) for i in dataset.image_data.values()}
check("buckets are multiples of 16", all(w % 16 == 0 and h % 16 == 0 for w, h in bucket_sizes), bucket_sizes)

# --- caching, with the real MOSS-Image strategies -----------------------------------------------------------------------
group = training_run()
want = sorted(["", "mytrigger, A girl smiles at the viewer.\n\nSoft daylight falls from the left.", "mytrigger, 1girl, smile",
               "mytrigger stands in the rain.", "mytrigger, 1girl, standing", "mytrigger, 1girl, sitting"])
check("every caption candidate is encoded once, with the trigger word where it was missing, plus the empty prompt for "
      "caption dropout", sorted(encoded_text) == want, encoded_text)
npz = np.load(te_npz("a"))
check("an image with both captions has both in one cache file, natural language first", int(npz["num_candidates"]) == 2
      and str(npz["caption_0"]).startswith("mytrigger, A girl smiles") and "\n\n" in str(npz["caption_0"])
      and str(npz["caption_1"]) == "mytrigger, 1girl, smile", [str(npz[k]) for k in ("caption_0", "caption_1")])
check("an image with only tags has one candidate", int(np.load(te_npz("c"))["num_candidates"]) == 1)
check("caption dropout rate and the unconditional embedding are in the cache",
      abs(float(npz["caption_dropout_rate"]) - 0.05) < 1e-6 and "uncond_embeds" in npz)
check("first run encodes the three images", sum(encoded_images) == 3, encoded_images)
check("latents: FLUX.2's 32 channels, cached under latent_cache/ with the image's fingerprint",
      any(np.load(latent_npz("a"))[k].shape[0] == 32 for k in np.load(latent_npz("a")).files if k.startswith("latents_"))
      and "image_fingerprint" in np.load(latent_npz("a")))

# a training sample picks one of the candidates
dataset = group.datasets[0]
dataset.set_current_epoch(1) if hasattr(dataset, "set_current_epoch") else None
key_a = next(k for k, i in dataset.image_data.items() if Path(i.absolute_path).name == "a.png")
seen = set()
strategy = strategy_base.TextEncoderOutputsCachingStrategy.get_strategy()
for k in range(2):
    embeds = strategy.load_outputs_npz(str(te_npz("a")), candidate_index=k)[0]
    seen.add(float(embeds[0, 0]))
check("the two candidates load as different embeddings", len(seen) == 2, seen)
try:
    batch = dataset[0]
    outs = batch.get("text_encoder_outputs_list")
    ok = batch.get("latents") is not None and outs is not None and len(outs) == 5
    detail = (None if batch.get("latents") is None else tuple(batch["latents"].shape), None if outs is None else len(outs))
except Exception as e:  # noqa: BLE001
    ok, detail = False, repr(e)
check("the dataset serves a batch from these caches (latents + text-encoder outputs with the unconditional pair and "
      "dropout rate)", ok, detail)

# --- nothing changed: caches reused --------------------------------------------------------------------------------------
training_run()
check("unchanged dataset: no caption re-encoded", encoded_text == [], encoded_text)
check("unchanged dataset: no image re-encoded", encoded_images == [], encoded_images)

# --- edited natural-language caption --------------------------------------------------------------------------------------
(ds / "a.toriigate.txt").write_text("A girl laughs.", encoding="utf-8")
training_run()
check("an edited caption re-encodes that image's candidates only (the empty prompt is encoded once per launch)",
      sorted(encoded_text) == ["mytrigger, 1girl, smile", "mytrigger, A girl laughs."], encoded_text)

# --- another caption source: the candidates of each image change ---------------------------------------------------------
training_run(source=TAGS)
check("switching to tags only: images that had two candidates are cached again with one",
      sorted(encoded_text) == ["mytrigger, 1girl, smile", "mytrigger, 1girl, standing"] and int(np.load(te_npz("a"))["num_candidates"]) == 1,
      encoded_text)
check("…and images are not re-encoded for a caption change", encoded_images == [], encoded_images)

# --- changed trigger word ------------------------------------------------------------------------------------------------
training_run(source=TAGS, trigger="other")
check("changed trigger word: every caption encoded again", sorted(encoded_text) ==
      ["other, 1girl, smile", "other, 1girl, standing", "other, mytrigger, 1girl, sitting"], encoded_text)

# --- image edited in place, same name and size ---------------------------------------------------------------------------
time.sleep(0.02)
Image.new("RGB", (512, 768), (10, 220, 10)).save(ds / "a.png")
training_run(source=TAGS, trigger="other")
check("edited image is encoded again (only that one), captions untouched", sum(encoded_images) == 1 and encoded_text == [], (encoded_images, encoded_text))

# --- Anima and MOSS-Image caches share the two folders without colliding ------------------------------------------------
class ATok(strategy_base.TokenizeStrategy):
    def tokenize(self, text):
        n = len(text) if isinstance(text, list) else 1
        return [torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4), torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4)]


class AEnc(strategy_base.TextEncodingStrategy):
    def encode_tokens(self, tokenize_strategy, models_, tokens_and_masks):
        n = tokens_and_masks[0].shape[0]
        return torch.zeros(n, 4, 8), torch.ones(n, 4), torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4)


class AVAE:
    device, dtype = torch.device("cpu"), torch.float32

    def encode_pixels_to_latents(self, img):
        return torch.nn.functional.avg_pool2d(img, 8).mean(1, keepdim=True).repeat(1, 16, 1, 1)


for cls in (strategy_base.TokenizeStrategy, strategy_base.TextEncodingStrategy, strategy_base.TextEncoderOutputsCachingStrategy,
            strategy_base.LatentsCachingStrategy):
    cls._strategy = None      # one trainer process holds one set of strategies; this test plays two trainers in turn
strategy_base.TokenizeStrategy.set_strategy(ATok())
strategy_base.TextEncodingStrategy.set_strategy(AEnc())
strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(strategy_anima.AnimaTextEncoderOutputsCachingStrategy(True, 8, False))
strategy_base.LatentsCachingStrategy.set_strategy(strategy_anima.AnimaLatentsCachingStrategy(True, 4, False))
a_group = build_group(a_args)
a_group.new_cache_latents(AVAE(), acc)
a_group.new_cache_text_encoder_outputs([None, None], acc)
names = sorted(p_.name for p_ in (ds / "cache_text_encoder").iterdir()) + sorted(p_.name for p_ in (ds / "latent_cache").iterdir())
check("an Anima run on the same folder adds its own cache files next to MOSS-Image's",
      sum(n.endswith("_anima_te.npz") for n in names) == 3 and sum(n.endswith("_moss_te.npz") for n in names) == 3
      and sum(n.endswith("_anima.npz") for n in names) == 3 and sum(n.endswith("_moss.npz") for n in names) == 3, names)
check("the dataset folder itself holds only images and captions (caches are in the two subfolders)",
      not list(ds.glob("*.npz")), [p_.name for p_ in ds.glob("*.npz")])

# --- ConvRot INT8 without the Triton kernels: what a GPU older than the RTX 30 series trains with --------------------
import logging  # noqa: E402

from library import convrot_int8_kernels as ck  # noqa: E402
from library import convrot_int8_utils as cu  # noqa: E402

check("the Triton kernels are for compute capability 8.0 and newer: a 2080 Ti (7.5) is given the reason, newer GPUs none",
      "7.5" in (ck.triton_capability_problem((7, 5)) or "") and "RTX 30" in ck.triton_capability_problem((7, 5))
      and [ck.triton_capability_problem(c) for c in ((8, 0), (8, 6), (8, 9), (12, 0))] == [None] * 4, ck.triton_capability_problem((7, 5)))
check("without a GPU the kernels are off and their entry points are absent", ck.HAS_TRITON is False and ck.int8_linear is None)


class Heard(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


heard = Heard()
cu.logger.addHandler(heard)
saved_reason = cu.TRITON_UNUSABLE_REASON
try:
    cu.TRITON_UNUSABLE_REASON = None
    cu._validate_convrot_bwd_mode("bf16")
    plain = list(heard.lines)
    del heard.lines[:]
    cu.TRITON_UNUSABLE_REASON = ck.triton_capability_problem((7, 5))
    cu._validate_convrot_bwd_mode("bf16")
    old_gpu = list(heard.lines)
    try:
        cu._validate_convrot_bwd_mode("int8")
        refused = ""
    except ValueError as e:
        refused = str(e)
finally:
    cu.TRITON_UNUSABLE_REASON = saved_reason
    cu.logger.removeHandler(heard)
check("no Triton: the log suggests installing it", len(plain) == 1 and "Install triton" in plain[0], plain)
check("Triton there but the GPU too old: the log gives that reason and doesn't suggest installing anything",
      len(old_gpu) == 1 and "compute capability is 7.5" in old_gpu[0] and "Install" not in old_gpu[0] and "falls back" in old_gpu[0], old_gpu)
check("…and the INT8 backward mode is refused for the same reason", "7.5" in refused and "Install" not in refused, refused)

torch.manual_seed(0)
lin_w = torch.randn(48, 256) * 0.05
lin_x = torch.randn(2, 5, 256)
wq, w_scale = ck.quantize_int8_convrot_weight(lin_w, 256)
got = cu.convrot_int8_linear(lin_x, wq, w_scale, None, 256, True, "bf16")
want = torch.nn.functional.linear(lin_x, lin_w)
rel = ((got - want).norm() / want.norm()).item()
check("the route such a GPU takes computes the layer from the INT8 weight (within quantization error of the original)",
      wq.dtype == torch.int8 and got.shape == want.shape and rel < 0.05, rel)
lin_x.requires_grad_(True)
cu.convrot_int8_linear(lin_x, wq, w_scale, None, 256, True, "bf16").sum().backward()
want_grad = lin_w.sum(0).expand_as(lin_x)
rel_g = ((lin_x.grad - want_grad).norm() / want_grad.norm()).item()
check("…and its backward pass gives the gradient the original layer would", rel_g < 0.05, rel_g)

shutil.rmtree(root, ignore_errors=True)
finish()
