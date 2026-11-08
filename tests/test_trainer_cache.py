"""sd-scripts' caches, run for real: the dataset is built from a dataset.toml written by TrainFlow exactly as
train_network.py builds it, then the real text-encoder and latent caching code runs with stand-in encoders (no GPU,
no model files). Checks that the trigger word reaches the cached text, and that edited captions and edited images are
encoded again instead of reusing old cache files.

Needs the trainer's packages (torch, transformers, diffusers…). On Windows this script re-runs itself with
python_embeded\\python.exe when the current Python doesn't have them; otherwise it reports SKIP."""
import os
import shutil
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
import torch
from PIL import Image

sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SD))
from _harness import check, finish  # noqa: E402

# TrainFlow's own dataset.toml writer (app.py executed with the real torch)
work = Path(tempfile.mkdtemp(prefix="tf_cache_app_"))
shutil.copy(REPO / "app.py", work / "app.py")
app = {"__name__": "app", "__file__": str(work / "app.py")}
exec(compile((REPO / "app.py").read_text(encoding="utf-8").replace('if __name__ == "__main__":', "if False:"), str(REPO / "app.py"), "exec"), app)

import anima_train_network  # noqa: E402
from library import config_util, strategy_anima, strategy_base  # noqa: E402

# --- stand-in encoders that record what they're given ----------------------------------------------------------------
encoded_text, encoded_images = [], []


class Tok(strategy_base.TokenizeStrategy):
    def tokenize(self, text):
        texts = text if isinstance(text, list) else [text]
        encoded_text.extend(texts)
        n = len(texts)
        return [torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4), torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4)]


class Enc(strategy_base.TextEncodingStrategy):
    def encode_tokens(self, tokenize_strategy, models, tokens_and_masks):
        n = tokens_and_masks[0].shape[0]
        return torch.zeros(n, 4, 8), torch.ones(n, 4), torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4)


class FakeVAE:
    device, dtype = torch.device("cpu"), torch.float32

    def encode_pixels_to_latents(self, img):          # content-dependent stand-in for the Qwen Image VAE
        encoded_images.append(img.shape[0])
        return torch.nn.functional.avg_pool2d(img, 8).mean(1, keepdim=True).repeat(1, 16, 1, 1)


strategy_base.TokenizeStrategy.set_strategy(Tok())
strategy_base.TextEncodingStrategy.set_strategy(Enc())
strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
    strategy_anima.AnimaTextEncoderOutputsCachingStrategy(cache_to_disk=True, batch_size=8, skip_disk_cache_validity_check=False))
strategy_base.LatentsCachingStrategy.set_strategy(
    strategy_anima.AnimaLatentsCachingStrategy(cache_to_disk=True, batch_size=4, skip_disk_cache_validity_check=False))
acc = types.SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, wait_for_everyone=lambda: None,
                            device=torch.device("cpu"))

ds = Path(tempfile.mkdtemp(prefix="tf_cache_")) / "ds"
ds.mkdir(parents=True)
Image.new("RGB", (512, 768), (200, 30, 30)).save(ds / "a.png"); (ds / "a.txt").write_text("1girl, smile", encoding="utf-8")
Image.new("RGB", (512, 768), (30, 30, 200)).save(ds / "b.png")
(ds / "b.txt").write_text("1girl, standing\nA girl standing in the rain.", encoding="utf-8")
cfg_dir = Path(tempfile.mkdtemp(prefix="tf_cache_cfg_"))


def training_run(trigger="mytrigger"):
    """What one 🚀 Start Training does to the caches: write dataset.toml, build the dataset, cache text and latents."""
    with_trigger, n_caps = app["trigger_caption_counts"](str(ds), trigger)          # same decision as start_training
    add_prefix = not (n_caps > 0 and with_trigger == n_caps)
    toml_path = app["create_dataset_toml"]("kw", str(ds), trigger, 512, 768, cfg_dir, 100, 1, 1, add_prefix=add_prefix)
    args = anima_train_network.setup_parser().parse_args(["--dataset_config", str(toml_path), "--output_dir", str(cfg_dir)])
    gen = config_util.BlueprintGenerator(config_util.ConfigSanitizer(True, True, args.masked_loss, True))
    group, _ = config_util.generate_dataset_group_by_blueprint(gen.generate(config_util.load_user_config(args.dataset_config), args).dataset_group)
    group.set_current_strategies()
    encoded_text.clear(); encoded_images.clear()
    group.new_cache_latents(FakeVAE(), acc)
    group.new_cache_text_encoder_outputs([None, None], acc)
    return group


def te_npz(stem):
    return ds / "cache_text_encoder" / f"{stem}_anima_te.npz"


def latent_npz(stem):
    return next((ds / "latent_cache").glob(f"{stem}_*_anima.npz"))


# --- the trigger word reaches the cached text ---------------------------------------------------------------------------
group = training_run()
check("cached text includes the trigger prefix", sorted(encoded_text) == ["mytrigger, 1girl, smile", "mytrigger, 1girl, standing"], encoded_text)
check("…identical to what uncached training would encode (minus dropout)",
      sorted(encoded_text) == sorted(group.datasets[0].caption_for_cache(group.datasets[0].subsets[0], i.caption)
                                     for i in group.datasets[0].image_data.values()))
check("…and stored in the cache file", str(np.load(te_npz("a"))["caption_text"]) == "mytrigger, 1girl, smile")
check("caption dropout still applies at train time (rate kept in the cache)",
      abs(float(np.load(te_npz("a"))["caption_dropout_rate"]) - 0.05) < 1e-6)
check("first run encodes both images", sum(encoded_images) == 2, encoded_images)

# --- nothing changed: caches reused --------------------------------------------------------------------------------------
training_run()
check("unchanged dataset: no caption re-encoded", encoded_text == [], encoded_text)
check("unchanged dataset: no image re-encoded", encoded_images == [], encoded_images)

# --- edited caption ------------------------------------------------------------------------------------------------------
(ds / "a.txt").write_text("1girl, smile, long hair", encoding="utf-8")
training_run()
check("edited caption is encoded again (only that one)", encoded_text == ["mytrigger, 1girl, smile, long hair"], encoded_text)

# --- changed trigger word ------------------------------------------------------------------------------------------------
training_run(trigger="othertrigger")
check("changed trigger word: every caption encoded again", sorted(encoded_text) ==
      ["othertrigger, 1girl, smile, long hair", "othertrigger, 1girl, standing"], encoded_text)

# --- caption already contains the trigger: TrainFlow adds no prefix, the text is cached as written ----------------------
(ds / "a.txt").write_text("othertrigger, 1girl, smile", encoding="utf-8")
(ds / "b.txt").write_text("1girl, othertrigger, standing", encoding="utf-8")
training_run(trigger="othertrigger")
check("captions that already have the trigger are cached as written", sorted(encoded_text) ==
      ["1girl, othertrigger, standing", "othertrigger, 1girl, smile"], encoded_text)

# --- caches from before this fix (no caption text / no image fingerprint) are rebuilt once ------------------------------
for path, key in ((te_npz("a"), "caption_text"), (latent_npz("a"), "image_fingerprint")):
    data = dict(np.load(path))
    data.pop(key)
    np.savez(path, **data)
training_run(trigger="othertrigger")
check("old-format caption cache is rebuilt", encoded_text == ["othertrigger, 1girl, smile"], encoded_text)
check("old-format latent cache is rebuilt", sum(encoded_images) == 1, encoded_images)

# --- image edited in place, same name and size ---------------------------------------------------------------------------
before = np.load(latent_npz("a"))
before_latents = {k: before[k] for k in before.files if k.startswith("latents_")}
time.sleep(0.02)
Image.new("RGB", (512, 768), (10, 220, 10)).save(ds / "a.png")        # e.g. a watermark painted out, same size
training_run(trigger="othertrigger")
after = np.load(latent_npz("a"))
check("edited image is encoded again (only that one)", sum(encoded_images) == 1, encoded_images)
check("…and its cache holds the new latents", any(not np.array_equal(after[k], v) for k, v in before_latents.items()))
check("…with the new file's fingerprint", np.array_equal(after["image_fingerprint"],
                                                         strategy_base.LatentsCachingStrategy.image_fingerprint(str(ds / "a.png"))))
check("…and no captions re-encoded for it", encoded_text == [], encoded_text)

# --- 🔢 Rename to Numbers shifts names onto other pictures (a new image sorts first): no cache may follow the old name
for _ in app["rename_dataset_sequential"](str(ds)):
    pass
training_run(trigger="othertrigger")                       # caches for 001/002
Image.new("RGB", (512, 768), (240, 240, 0)).save(ds / "000_new.png")
(ds / "000_new.txt").write_text("othertrigger, 1girl, yellow background", encoding="utf-8")
for _ in app["rename_dataset_sequential"](str(ds)):         # 000_new -> 001, old 001 -> 002, old 002 -> 003
    pass
check("rename shifted the names", (ds / "003.png").exists() and (ds / "001.txt").read_text(encoding="utf-8") == "othertrigger, 1girl, yellow background")
training_run(trigger="othertrigger")
check("after the shift every image whose name now means another picture is encoded again", sum(encoded_images) == 3, encoded_images)
check("…and so is every caption", sorted(encoded_text) == ["1girl, othertrigger, standing", "othertrigger, 1girl, smile",
                                                           "othertrigger, 1girl, yellow background"], encoded_text)
check("…and 001's cached text is the new picture's caption", str(np.load(te_npz("001"))["caption_text"]) == "othertrigger, 1girl, yellow background")

shutil.rmtree(ds.parent, ignore_errors=True)
finish()
