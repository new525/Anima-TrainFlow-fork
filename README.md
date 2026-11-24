# Anima TrainFlow — Fork

A one-page GUI for training LoRAs on the **Anima 2B** anime model, tuned for NVIDIA GPUs with as little as **6 GB of VRAM**, and on **MOSS-Image** (OpenMOSS's 5B model), which needs more: see [Training MOSS-Image](#training-moss-image).

This is a fork of [ThetaCursed/Anima-TrainFlow](https://github.com/ThetaCursed/Anima-TrainFlow), which created the app. The fork keeps the same workflow and focuses on making long training runs hard to lose and checkpoints easy to choose. It is not affiliated with or endorsed by the upstream project.

`Raw Images` ➔ `Smart Resize & Crop` ➔ `Auto-Tag` ➔ `Inspect` ➔ `Train` ➔ `Compare Checkpoints` ➔ `Ready LoRA`

> **Current version: 1.16** — see [CHANGELOG.md](CHANGELOG.md) and [Releases](https://github.com/new525/Anima-TrainFlow-fork/releases).
> Fork versions are numbered independently of upstream. Fork 1.1 was built on upstream's v1.1.0 code.

---

## How this fork differs from upstream

| | Upstream (v1.1.0) | This fork (1.16) |
|---|---|---|
| **Models** | Anima 2B | Anima 2B and MOSS-Image |
| **Refreshing the page during training** | The trainer hangs on a full output pipe, keeping its VRAM, and Stop can no longer reach it | Training carries on; any tab shows the live run |
| **Closing or restarting the app during training** | Training ends | Training carries on; the restarted app reattaches to it |
| **Interrupting a run** | Stop only (progress lost) | ⏸ Pause, 💾 Halt & Save, ↩ Resume Run, crash recovery from resume points |
| **Continuing a finished run** | — | ➕ Train More extends it to a higher step count |
| **Network alpha** | Fixed to the rank | Settable (default: same as rank) |
| **Low VRAM** | No block swapping | Block Swap slider (up to 26 of 28 DiT blocks in system RAM) |
| **Output folders** | One folder per keyword, shared by every run (earlier checkpoints overwritten); keywords with no Latin letters or digits all go to `untitled` | One folder per run: `output/<keyword>/<date_time>/`; any keyword |
| **Picking a checkpoint** | One preview prompt, unlabelled gallery | Several prompts, step × prompt grid, live loss chart, labelled previews |
| **Duplicate images** | Not checked | Identical files and re-saved, resized or mirrored copies found during bucketing and set aside, keeping the largest |
| **Re-running bucketing with new sizes** | Can overwrite your original images with crops | Non-destructive; originals are never touched |
| **Phone photos** | Bucket chosen from unrotated size: cropped the wrong way | Rotated upright first |
| **Captions** | WD tagger only | Tags already embedded by Grabber/exiftool first, written in Anima's caption format; WD tagger for the rest |
| **Natural-language captions** | — | A separate ToriiGate 0.5 captioner writes `<name>.toriigate.txt` next to each image. MOSS-Image runs can train on them, on the tags, or on both; Anima runs read the tag captions |
| **Dataset formats** | PNG, JPG, WebP, BMP (the tagger skipped BMP) | Also videos and GIFs (their middle frame), AVIF, JPEG XL, HEIC, and anything else Pillow or OpenCV can read |
| **Dataset checks** | Pre-flight (its oversized-image check never fired) | Working pre-flight plus 🔎 Inspect Dataset |
| **Bucketing / tagging during a refresh** | Interrupted with the tab | Run in the background |
| **Trigger word prefix** | `trigger,  tags` (double space); added even if captions already contain it; never actually trained, because the cached captions left it out | `trigger, tags`, and trained; skipped when every caption has it |
| **Caption and image caches** | Reused after a caption or image is edited, or after renaming shifts names onto other pictures | Encoded again whenever the caption text or the image file changed |

The complete list, including smaller fixes, is in [CHANGELOG.md](CHANGELOG.md).

![Anima TrainFlow interface](preview.png)
*The fork's interface during a training run, with the Loss tab open (demo run).*

---

## Installation (Windows)

> **Don't use upstream's 3 GB portable download for this fork.** It contains upstream's code, so you would get none of the changes above. This fork already bundles its own Python, so no portable build is needed.

**Requirements**
* Windows 10/11.
* An NVIDIA GPU with a driver recent enough for CUDA 12.8; the installer uses PyTorch's CUDA 12.8 build.
  * **Anima:** 6 GB of VRAM or more.
  * **MOSS-Image:** about 7 to 8 GB of VRAM with the default low-memory settings and about 14 GB of free system RAM, going by sd-scripts' measurements; see [Training MOSS-Image](#training-moss-image).
* Free disk space for PyTorch, the model files, and your runs. Resume points include optimizer state and are much larger than a LoRA.

### 1. Get the fork
* **Easiest:** download the zip from [Releases](https://github.com/new525/Anima-TrainFlow-fork/releases) and extract it to a folder of your choice.
* **Or with git:**
  ```bash
  git clone https://github.com/new525/Anima-TrainFlow-fork
  cd Anima-TrainFlow-fork
  ```

Either way you get `python_embeded/`, an embedded Python 3.10 with pip. There's no separate Python to install.

### 2. Install dependencies
Run **`Install_Requirements.bat`**. It installs PyTorch (CUDA 12.8) and the training dependencies into `python_embeded/`.

**Only for MOSS-Image:** then run **`Install_Triton_for_MOSS.bat`**. It's optional, and it makes the low-memory ConvRot INT8 option run at full speed. It finishes by trying one of the trainer's INT8 kernels on your GPU and says whether that worked; if it didn't, MOSS-Image still trains, more slowly. If it reports a problem with PyTorch instead, it names the Python it ran with and shows the error. Triton compiles its kernels against the bundled Python, which carries its own headers and import library for that since 1.16 (`python_embeded\Include`, `python_embeded\libs`).

### 3. Download the models
Put each file in the folder shown; these are the paths the app uses by default. If you already have the files elsewhere, for example in ComfyUI, you can point the app at them instead (step 4).

This table is for Anima, plus the two dataset tools both models use (the WD tagger and U2Net). For MOSS-Image, download the three files listed under [Training MOSS-Image](#training-moss-image) instead of the first three rows.

| What | File | Put it in | Download |
|---|---|---|---|
| Anima DiT (base) | `anima-base-v1.0.safetensors` | `models/anima/dit/` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors) |
| Text encoder (Qwen3 0.6B) | `qwen_3_06b_base.safetensors` | `models/anima/text_encoder/` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/text_encoders/qwen_3_06b_base.safetensors) |
| VAE (Qwen-Image) | `qwen_image_vae.safetensors` | `models/anima/vae/` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/vae/qwen_image_vae.safetensors) |
| WD tagger (auto-captioning) | `model.onnx` + `selected_tags.csv` | `models/wd-eva02-large-tagger-v3/` | [model.onnx](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx) (1.26 GB), [selected_tags.csv](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv) |
| U2Net (smart cropping) | `u2net.onnx` | `models/u2net/` | [u2net.onnx](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx) |

* The Anima model card says LoRAs should be trained on the **base** model, which is why it's listed here.
* Anima is released under the CircleStone Labs Non-Commercial License; read its terms on the model page.
* For the tagger, download just the two files rather than cloning the repo. A clone also pulls two other formats of the same model, 3.78 GB in total.

### 4. Launch
Run **`start_trainer.bat`**; the UI opens in your browser. Pick the model under **Model to train a LoRA for** (Anima or MOSS-Image); the choice is remembered. If your model files aren't in the default folders, set their paths once under **🔧 Paths to Models** (each model has its own).

### Updating
* **git:** run `git pull`.
* **Zip:** extract the new release, then copy the *contents* of its top folder over your existing install. The zip doesn't contain your settings (`training/settings.json`), runs (`training/output/`) or installed packages, so they are kept.
* Re-run `Install_Requirements.bat` when the changelog says dependencies changed.
  * **1.2.1** (GitHub release 1.3) updated Gradio.
  * **1.4** adds `pillow-jxl-plugin` for JPEG XL and `pillow-heif` for HEIC.
  * **1.5** through **1.13** add no dependencies. (1.9 lists `pandas` and `psutil` explicitly, but they were already installed.)
  * **1.14** replaces the bundled sd-scripts with v0.13.1, which needs newer `transformers`, `diffusers`, `accelerate`, `safetensors` and `huggingface-hub`. **Re-run `Install_Requirements.bat`**; training won't start on the old packages.
  * **1.15** and **1.16** add no dependencies. 1.16 adds two folders to `python_embeded` (`Include`, `libs`) that Triton needs.

---

## Using it

### Prepare the dataset
Put your images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) in one folder and set it as **Dataset Path**. Other formats can go in too:
* **Videos:** `.mp4`, `.m4v`, `.mov`, `.webm`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.mpg`, `.mpeg`
* **GIF, AVIF, JPEG XL (`.jxl`) and HEIC/HEIF** (iPhone photos) images
* **Anything else Pillow or OpenCV can read**, such as TIFF, PSD, TGA, `.jfif`, a saved `.jpg_large`, or `.3gp` and `.ts` videos. This is on by default; untick *Also convert any other format…* to limit conversion to the formats above.

Then:

1. **✂️ Start Bucketing** resizes and crops images to training buckets.
   * First it turns each video or animation into an image of its middle frame, and converts the other formats to PNG. Training can't read these formats directly. Phone videos come out upright.
   * A multi-page TIFF gives its first page, a layered PSD its flattened image, and a multi-image HEIC its main image.
   * Files that nothing can decode are left where they are and listed in the log. This includes audio, documents and caches.
   * Your originals, including the videos, GIFs and AVIFs, are moved to `original_images/` and never modified.
   * Running it again with different sizes regenerates the crops from those originals.
   * **Min Side** and **Max Side** are rounded to multiples of 64 (Min up, Max down), the bucket step training uses.
   * It refuses, changing nothing, files that would end up with the same name: `cat.jpg` + `cat.png`, `cat.mp4` + `cat.gif`, or names that differ only in letter case like `Cat.png` + `cat.jpg`, which Windows treats as one file.
   * A caption already written for a video (`clip.txt` next to `clip.mp4`) pairs with the image made from it (`clip.png`).
   * **Duplicates are checked** across the whole dataset, including images bucketed in earlier runs.
     * **What counts:** identical files, plus re-saved, resized, recompressed, watermarked or mirrored copies — the typical result of downloading one post from two sites, or a sample alongside the full size.
     * **What happens:** the copy with the most pixels is kept; on a tie, the one with a caption. *Set duplicates aside* (on by default) moves the other copies to `duplicates/`, together with their captions, originals and sidecars. Nothing is deleted; move a file back to restore it.
     * **Captions:** if the kept copy has no caption but a copy being set aside does, the kept copy gets a copy of it.
     * **Untick it** to only list the groups.
     * **`duplicates.log`:** every check that finds duplicates appends the full list to this file in the dataset folder, whether or not they were moved.
       * Each group shows the kept file and every copy, with its size and why it matched.
       * Copies are listed with their path, or where they were moved.
       * A report-only result identical to the previous one isn't appended again.
       * It's a `.log` rather than a `.txt` so it isn't mistaken for a caption.
     * **What isn't a duplicate:** colour variants of the same drawing, different sketches with the same pose, and brightness-edited or cropped versions.
2. **Create .txt captions** writes a `.txt` for every image.
   * **Embedded tags come first.** Tags that a downloader such as Grabber saved inside the image are used as they are, and the WD tagger only handles images without any.
     * **Where it looks:** XMP Subject/Keywords, IPTC Keywords, Windows keywords, or an `.xmp` sidecar. It reads the original in `original_images/` too, so it works after bucketing. It also reads tags saved in GIFs and videos.
     * **Format:** tags are rewritten to Anima's caption format — lowercase, spaces instead of underscores, `@` before artists, and the model card's order: meta, 1girl/1boy…, character, series, artist, general.
     * **Parentheses stay plain**, e.g. `hatsune miku (cosplay)`, in embedded and WD-tagger captions alike. Training reads captions literally, while Forge and ComfyUI strip a prompt's `\(` back to `(` before the text encoder sees it. So the backslash belongs in your prompts, not in captions.
     * **Turn it off** with *Use tags embedded in the images first*.
   * **Recommended Grabber setup.** In Grabber's Exiftool metadata settings, set the key to `Subject` and the value to `%all:includenamespace,unsafe,separator=;%`.
     * The namespaces tell the app which tags are artists, characters and series, so it can add the `@` and order them.
     * `Subject` is stored as XMP, which keeps non-English tags. `Keywords` is stored as IPTC in JPG and PNG, which turns them into `????` when the file is saved.
     * Grabber's docs write keys as `XMP.Subject`; exiftool rejects that form, so use `Subject` or `XMP:Subject`.
3. **🔎 Inspect Dataset** checks the result before you spend an hour training:
   * missing, empty or orphaned captions, and captions named in a different letter case than their image
   * escaped parentheses `\(` `\)` in captions
   * duplicate images
   * videos, GIFs or other formats that haven't been converted yet
   * oversized, small, unreadable or rotated images
   * repeated tags, and the trigger word already in some captions
   * captions with more than one line (training reads only the first)
   * the most frequent tags, and tags present in every caption

4. **🔢 Rename to Numbers** (optional) renames the images to `001`, `002`, … in filename order. Short names avoid Windows' path-length limit once cache suffixes are added.
   * Everything tied to an image is renamed together: caption, original and source in `original_images/`, `.xmp` sidecars, and bucketing's records. Bucketing and captioning keep working afterwards.
   * The old → new names are saved to `rename_map_<time>.json`.
   * It changes nothing if another file already uses one of the new names — for example a leftover `002.txt` that would otherwise become an image's caption.
   * Files pair with their image regardless of letter case (`IMG_0001.JPG` and `img_0001.txt`).
   * If a file is locked by another program part-way, everything already renamed is put back.

These tools and training don't work on the same folder at once. 🚀 Start Training waits until a tool working on its dataset has finished, and Bucketing and Rename refuse a dataset that's being trained. Tools on another folder run normally.

The tools run in the background; their progress shows in the log.

### Natural-language captions (optional)
`start_captioner.bat` opens a separate captioner that writes a natural-language caption for every image with [ToriiGate 0.5](https://huggingface.co/Minthy/ToriiGate-0.5).
* It runs the model in llama.cpp. Both are downloaded on first use: about 5.2 GB for the model.
* The image's Grabber tags or tag caption are passed to the model as hints, so characters get their right names. **🔎 Check hints** shows, without loading the model, where each image's hints come from and why some have none.
* The prompt can be edited, starting from the model's own. Each image's tags and names are still filled in.
* Each caption is saved as `<name>.toriigate.txt`; your tag captions aren't touched. The dataset tools above treat it as part of its image.
* **MOSS-Image runs can train on these captions** (Caption Source). Anima runs read only `<name>.txt`.
* There's a command line too. Details are in [tools/toriigate-captioner/README.md](tools/toriigate-captioner/README.md).

### Train
Set a **Trigger Word**, which also names the run's folder, then press **🚀 Start Training**. The status line above the buttons always shows what's running. This section describes Anima runs; what differs for MOSS-Image is under [Training MOSS-Image](#training-moss-image).

The trigger word is put at the front of every caption that doesn't already have it. Captions and images are encoded once and cached next to the dataset (`cache_text_encoder/`, `latent_cache/`). Each run checks those caches against the current caption text and image files, and encodes again whatever changed, so editing, re-captioning or renaming needs no manual cleanup.

| Button | What it does |
|---|---|
| ⏸ **Pause / ▶ Continue** | Freezes training instantly and continues exactly where it stopped. VRAM stays allocated while paused. |
| 💾 **Halt & Save** | Finishes the current step, saves a LoRA and a full resume point, then exits and frees the VRAM. |
| 🛑 **Stop** | Ends training immediately, without saving. |
| ↩ **Resume Run** | Continues the keyword's newest unfinished run (halted or crashed) from its newest resume point, using the settings it was started with. |
| ➕ **Train More** | Extends the keyword's newest *finished* run up to the current **Training Steps** value. The previous final LoRA is kept. |

More options:
* **Network Alpha:** the LoRA's output is scaled by alpha ÷ rank.
  * **0** (the default) means "same as rank", giving a scale of 1.0, which is how the app always trained before this setting existed.
  * **With AdamW:** Adam's steps don't depend on how large the gradients are, so the scale acts as a multiplier on the effective learning rate. Alpha 16 at rank 32 is roughly half the effective learning rate.
  * **With Prodigy:** Prodigy estimates its own step size, so it should largely compensate for a different alpha. Leaving it at 0 is the safe choice.
  * The log shows the alpha and scale each run uses.
* **Block Swap (Low VRAM):** keeps that many of the 28 DiT blocks in system RAM. This lowers VRAM use at the cost of slower steps. Use it if training runs out of memory.
* **Save resume points with each checkpoint** (on by default): saves a resume point with every checkpoint and keeps the newest two. A crash then costs at most one save interval, and ➕ Train More becomes possible. A 💾 Halt & Save resume point is removed once a newer one exists, or when the run finishes.
* You can refresh the page, close it, or even close the console window and restart the app. Training keeps going, and the restarted app reconnects to it.

### Training MOSS-Image
**In short:**
1. Run `Install_Triton_for_MOSS.bat` once (optional, see *Memory*).
2. Download the three model files below.
3. Caption the dataset: `start_captioner.bat` for natural-language captions, *Create .txt captions* for tags, or both.
4. Choose **MOSS-Image**, check **Caption Source**, set a Trigger Word, press **🚀 Start Training**.

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) is a 5B diffusion transformer with a Qwen3.5-9B text encoder and the FLUX.2 VAE. Choose **MOSS-Image** under *Model to train a LoRA for*; the page then shows that model's own paths, memory options and preview settings. Everything else works as for Anima: the dataset tools, Pause, Halt & Save, Resume Run, Train More, the checkpoint grid and the loss chart. The two models share nothing, so a dataset folder can be trained on both; each keeps its own cache files.

**Model files.** The pre-quantized ConvRot INT8 files from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) are the defaults: they are the smallest downloads and need the least memory.

| What | File | Put it in | Where it is in the Comfy-Org repository |
|---|---|---|---|
| DiT (INT8, 5.41 GB) | `moss_image_base_int8_convrot.safetensors` | `models/moss_image/dit/` | `diffusion_models/` |
| Text encoder (INT8, 7.34 GB) | `qwen3.5_9b_int8_convrot.safetensors` | `models/moss_image/text_encoder/` | `text_encoders/` |
| VAE (336 MB) | `flux2-vae.safetensors` | `models/moss_image/vae/` | `vae/` |

* Also accepted, as sd-scripts documents them: the bf16 single file `moss_image_base_bf16.safetensors` (10.16 GB), the SFT variants, and the official split files from [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base). For split files give the first part (`…-00001-of-00003.safetensors` for the DiT, `model-00001-of-00004.safetensors` for the text encoder); the rest are loaded from the same folder.
* **Not accepted:** `qwen3.5_9b_w4a8_gptq.safetensors`. sd-scripts can't load it, and the app refuses it.
* The tokenizer (about 20 MB) is downloaded from Hugging Face the first time a run starts. To train offline, put its files in a folder and set **Tokenizer folder**.
* MOSS-Image is released under Apache 2.0; read the terms on the model pages.

**Captions.** MOSS-Image was trained on long natural-language captions, so **Caption Source** offers three choices:

| Caption Source | A run reads | An image without it |
|---|---|---|
| Natural language (the default) | `<name>.toriigate.txt`, whole, line breaks included | stops the run before it starts |
| Tags | the first line of `<name>.txt`, as Anima does | stops the run before it starts |
| Both | both files: two captions per image, one picked at random at every step | trains on the one it has; the log lists those images |

* The trigger word is added to each caption that doesn't contain it yet, so no caption gets it twice. In sentences it's matched as a whole word, in any letter case; in tags, as a tag.
* With *Both*, each caption is encoded and cached, so caching takes twice as long and twice the disk space.
* Editing a caption, or switching the Caption Source, re-encodes just the captions that changed. ↩ Resume Run and ➕ Train More read the captions as they are when you press the button.
* The captions a run used are in its folder: `configs/<keyword>_metadata.json`.

**Memory.** sd-scripts' own measurements, at 1024×1024, batch size 1, rank 16, on Windows ([its guide](training/sd-scripts/docs/moss_image_train_network.md) has the full table):

| DiT | Text encoder | Block Swap | VRAM while caching captions | VRAM while training | System RAM |
|---|---|---|---|---|---|
| bf16 | bf16 | 0 | 15.3 GB | 13.2 GB | 17.2 GB |
| INT8 | INT8 | 0 | 9.1 GB | 9.1 GB | 9.9 GB |
| INT8 | INT8 | 20 | 7.9 GB | 6.9 GB | 14.0 GB |
| INT8 | INT8 | 36 | 7.9 GB | 5.2 GB | 15.6 GB |

* **The defaults are the third row** (INT8 for both, Block Swap 20), which leaves room on an 11 or 12 GB card on paper. sd-scripts measured these on an RTX PRO 6000; on another card, watch the first run.
* **ConvRot INT8 DiT / text encoder:** quantizes a bf16 file while loading. For the pre-quantized files the boxes change nothing, because sd-scripts detects those files itself.
* **INT8 needs Triton to run at full speed.** Run **`Install_Triton_for_MOSS.bat`** once, after `Install_Requirements.bat`. It installs the `triton-windows` version your PyTorch names, then compiles and runs one of sd-scripts' INT8 kernels on your GPU and tells you whether that worked. Without Triton, INT8 still saves the VRAM but runs through a fallback that sd-scripts describes as slower than bf16. The app's log warns at the start of a run when Triton isn't installed.
* **Block Swap** keeps that many of the 42 DiT blocks in system RAM (at most 40). **H2D-only** is sd-scripts' faster variant for LoRA training; it uses slightly more VRAM for the same count.
* **Two notices while captions are cached are harmless:** `causal_conv1d is not installed` and `flash-linear-attention is not installed`. They come from the text encoder falling back to a slower, equally correct routine. With cached captions the text encoder runs once per caption and not at all during training steps, so only the caching step is slower.
* **Text Encoder Block Swap** streams that many of the text encoder's 32 layers from RAM while captions are cached, for cards the caching step doesn't fit on. sd-scripts gives about 5 GB at 28 and 3.5 GB at 32 for the bf16 text encoder, and 2.2 GB at 32 with INT8. It's slower, and only matters during caching.
* On Windows the page file has to be large enough: sd-scripts notes a commit charge of up to about 38 GB for its bf16 runs.

**Training settings.**
* Rank, alpha, learning rate, optimizer, steps and batch size are the same controls as for Anima, with the values you last set. They aren't changed when you switch model.
* The example in sd-scripts' MOSS-Image guide uses rank 16, AdamW8bit at 1e-4 with a constant schedule and alpha 1. In this app an alpha of 0 means "same as rank", which scales the LoRA 16 times stronger than alpha 1 at rank 16, so that learning rate doesn't carry over unchanged. Prodigy picks its own step size.
* Fixed for MOSS-Image runs: the timestep distribution of the model's own pre-training (`truncated_logit_normal`, mean -1.09, std 0.95), bf16 mixed precision, gradient checkpointing, DiT-only training with the text encoder frozen (the model authors' recommendation), caption dropout 5%.

**Previews.** MOSS-Image has its own Negative Prompt, Steps and CFG, starting at the official pipeline's defaults: empty, 50 and 3. Width, height, seed and the prompts are shared with Anima. Fifty steps per preview is slow with block swap; lower Steps or raise *Preview Every n Steps* if previews hold the run up.

**Using the LoRA.** sd-scripts states that ComfyUI loads these LoRAs with its standard LoRA loader, no conversion needed, in a ComfyUI version that supports MOSS-Image.

### Choose the best checkpoint
* **Preview Prompts:** write one prompt per line. Each is rendered at every preview step, with the trigger word added.
* **Checkpoint Grid** tab: one image of every step × prompt, with steps that saved a LoRA marked.
* **Loss** tab: the moving-average loss over the run.

### Where things go
```
training/output/<keyword>/<YYYYmmdd_HHMMSS>/   one folder per run
├── <keyword>.safetensors                      final LoRA
├── <keyword>-step00000300.safetensors         checkpoints
├── <keyword>-step00000300-state/              resume points
├── sample/                                    previews
├── checkpoint_grid.png
├── configs/  logs/  run.json
```

---

## Status

* **Tested:** the fork's changes are covered by the test suite in [`tests/`](tests/README.md). It runs the real app on Python 3.10, the bundled version, end to end through its API, with a stand-in trainer that mimics the real one's files and output. No GPU is needed.
* **Tested against sd-scripts' real code, without a GPU:** the configs the app writes for both models are read by the real trainers' option parsers, the MOSS-Image trainer's own start-up checks run on the app's dataset, and the real caching code runs with stand-in encoders.
* **Browser-tested:** the interface has been exercised in headless Chromium. That covers switching tabs mid-run, the loss chart updating, previews, and settings surviving a page refresh.
* **Not yet confirmed:**
  * **A real MOSS-Image training run.** Nothing in 1.14 has been run on a GPU: not loading the models, not the INT8 kernels, not the memory figures above, not a finished LoRA.
  * **Anima on the updated sd-scripts (v0.13.1) on a real GPU.** 1.14 replaces the trainer under Anima too. Keep your 1.13 folder until a 1.14 Anima run has worked for you.
  * `Install_Triton_for_MOSS.bat` on Windows, and Triton with the bundled embedded Python.
  * RTX 20-series cards: speed of bf16 mixed precision and of the INT8 kernels. sd-scripts publishes no measurements for them.
  * On a real GPU: halt/resume through the bundled trainer (including the 1.9 fix for resuming with gradient accumulation), and block swap's VRAM savings.
  * On Windows: training surviving a closed console window.
  * The ToriiGate captioner with the real model on a GPU. Its llama.cpp handling was tested with a tiny model of the same architecture on the CPU.

See *Known limitations* in [CHANGELOG.md](CHANGELOG.md), and please open an issue if something misbehaves.

## Technical details
* **Core:** kohya-ss `sd-scripts` v0.13.1 with this fork's patches, each marked `[TrainFlow]` in the source: a halt-request hook, resuming with gradient accumulation fixed (the step count, and the epoch it resumes in), Halt & Save resume points removed once they're superseded, cached Anima captions and cached latents of both models encoded again when the caption or image changed, and caches kept in `cache_text_encoder/` and `latent_cache/`. `anima_train_network.py` trains Anima and `moss_image_train_network.py` MOSS-Image; both run the same patched `train_network.py`.
* **UI:** Gradio 6.17.3, kept at that version because of the tab-switching freeze fixed in 1.2.1. The trainer runs as a separate process launched with `accelerate launch`, and the UI follows its log file.
* **Defaults:** BF16, latent and text-encoder caching, and the Prodigy optimizer with a constant learning rate; AdamW uses a cosine schedule.
* **MOSS-Image datasets** are sd-scripts metadata datasets: the app writes the captions of a run into a JSON file, which is how one image can carry two captions.
* **Settings** are auto-saved to `training/settings.json`.

## Credits & license
Anima TrainFlow was created by **ThetaCursed** ([upstream repository](https://github.com/ThetaCursed/Anima-TrainFlow)). This fork's changes are released under the same [MIT License](LICENSE). Anima model weights are © CircleStone Labs under their own license. MOSS-Image is by OpenMOSS (Apache 2.0). The bundled `sd-scripts` is by kohya-ss. The captioner's `prompts.py` is ToriiGate 0.5's own, by Minthy (MIT); see [its third-party notes](tools/toriigate-captioner/THIRD_PARTY.md).
