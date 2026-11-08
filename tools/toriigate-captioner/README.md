# ToriiGate 0.5 Captioner

Writes a natural-language caption for every image in a folder with [ToriiGate 0.5](https://huggingface.co/Minthy/ToriiGate-0.5), a 4B vision model for anime art. The model runs in [llama.cpp](https://github.com/ggml-org/llama.cpp).

- Each caption goes next to its image as `<name>.toriigate.txt`.
- Your tag captions (`<name>.txt`) are never touched.
- The tool runs on its own; TrainFlow doesn't need to be open.

It comes with a web UI and a command line. Both use the same code.

> **TrainFlow doesn't train on these captions yet.** Training reads only `<name>.txt`. TrainFlow's dataset tools already treat `.toriigate.txt` as part of the image:
> - 🔢 Rename to Numbers renames it with its image.
> - The duplicate check sets it aside with its image, and copies it to the copy that's kept.
> - 🔎 Inspect Dataset counts these files and leaves them out of the tag checks.

## Start

**Web UI:** double-click `start_captioner.bat` in the TrainFlow folder. The page opens in your browser.

- Captioning runs in the background. Reloading or closing the page doesn't stop it, and the page shows the progress again when reopened.
- To stop it, press ⏹ Stop or close the console window.

**Command line** (from this folder):

```
toriigate_cli.bat "C:\Images\MyDataset"
toriigate_cli.bat "C:\Images\MyDataset" --format long --characters "my_trigger"
toriigate_cli.bat "C:\Images\MyDataset" --dry-run --limit 3      (prints the prompts, loads nothing)
toriigate_cli.bat --help
```

### What the first run downloads

These are stored in this folder:

- **`runtime\llama.cpp\`:** llama.cpp release `b12211`, the Windows CUDA 12.4 build, together with its CUDA runtime. That build runs on RTX 20-series cards and newer.
- **`models\`:** the model from [SleepVeryHard/ToriiGate-0.5_GGUF](https://huggingface.co/SleepVeryHard/ToriiGate-0.5_GGUF). Two files:
  - `ToriiGate-0.5_Q8_0.gguf`, 5.16 GB
  - the vision projector `mmproj_BF16.gguf`

Downloads resume if they're interrupted. ⬇ Download now fetches everything up front. If a Hugging Face download is refused, set `HF_TOKEN` to your token; it's sent when one is set.

## Settings

### Caption format

These are the model's own formats. Their instructions come from its `prompts.py`, which is bundled unmodified.

| Format | What you get |
|---|---|
| `short` | A few sentences covering the main things. The best fit for training captions. |
| `long` | 2 to 5 vivid paragraphs |
| `chroma-style` | Summary, numbered parts, a Midjourney-style summary, a commission request |
| `json` | JSON: each character, background, effects, texts, atmosphere |
| `min_structured_json` | Short JSON with tag-like phrases |
| `min_structured_md` | Markdown: thoughts about characters, key details, structured description (needs names) |
| `long_thoughts` | Markdown in 6 parts, starting with reasoning about who is who (needs names) |
| `long_thoughts_v2` | Reasoning, key details, a long description, then each character (needs names) |
| `json_comic` | JSON for comics, frame by frame |
| `md_comic` | Markdown for comics, frame by frame (needs names) |

**Length matters for training:** Anima's text encoder reads at most 512 tokens of a caption. That's sd-scripts' `qwen3_max_token_length`, and TrainFlow doesn't change it. A `long` caption can go past that.

### Hints: tags and names

The model card says captions get names right when the model is given the image's tags and characters. For each image the tool looks, in this order, at:

1. **Tags embedded by Grabber** (XMP, IPTC, Windows keywords, `.xmp` sidecars). These are read the same way as TrainFlow's 📥 Extract Embedded Tags.
   - With namespaces (`character:`, `artist:`, `copyright:`…), the tool can tell characters from other tags.
   - Without them, every tag is passed as a plain tag.
2. **The image's tag caption** (`<name>.txt`), first line only. TrainFlow's `@artist` tags are left out.
3. **Names you type** under *Names to use for every image*. These replace any found in the tags; for example, your trigger word.

Some tags are never passed:
- Artist tags, unless you tick *Include artist tags*.
- `score_*` tags.

**Character names:**
- **known** (default): the model uses names that were given or found, and is told not to guess when there are none.
- **recognize:** with no names known, the model is asked to recognise the characters.
- **none:** the model is told not to name anyone.

The formats marked *needs names* only work with names, so with **none**, or with no names found, those formats ask the model to recognise characters. The log says when that happens.

**Tag style:**
- **plain** (default): underscores become spaces and tags are comma-separated, e.g. `long hair, smile`.
- **booru:** `long_hair smile`.

Both styles are ones the model's prompt code produces itself.

### Generation

The defaults are the model's own:
- temperature 0.5
- up to 2048 new tokens
- each image scaled to at most 1 megapixel and sent as a JPEG at quality 95

Seed `-1` picks a random one each time.

### Model and server

- **llama.cpp, downloaded automatically (Windows):** the default.
- **my llama-server:** the path to a `llama-server` you installed, e.g. on Linux.
- **a server that's already running:** a URL. The tool talks to `/v1/chat/completions` and `/health` the way llama-server provides them.

**Picking a smaller model file:** 🔄 List files shows every `.gguf` in the repo with its size. Pick a smaller quant if the model doesn't fit next to other things on the GPU. A local `.gguf` path also works.

**GPU layers:**
- **all** (default): the whole model runs on the GPU.
- A number puts only that many layers there.
- **0** runs on the CPU, which is much slower.

**Keep the model loaded** keeps llama-server running between runs and 🔍 Try one image, so trying several settings doesn't reload it each time. ⏏ Unload model frees the VRAM.

🔍 **Try one image** captions a single image without saving it, and shows the caption next to the prompt that was sent.
- It uses the file name you enter, or the first image without a caption.
- Use it to check a format or names before a whole run.

## Files it writes

| Where | What |
|---|---|
| next to each image | `<name>.toriigate.txt`, the caption |
| the image folder | `toriigate_captions.log`: one line per image with the format, time, tokens and where the hints came from. Failures are logged too. |
| this folder | `settings.json` (UI settings), `runtime\` (llama.cpp and its log), `models\` |

Images that already have a `.toriigate.txt` are skipped unless *Overwrite* is on, so an interrupted run picks up where it stopped.

## What has been checked, and what hasn't

**Checked:**
- The prompts are the model's own `prompts.py` (checksum test).
- Requests have the shape of the model's reference script.
- Against a real llama.cpp `b12211` llama-server, using a tiny model with ToriiGate's architecture (Qwen3.5 vision-language) on the CPU:
  - the server starts with the flags this tool passes
  - the image reaches the model, about one token per 32×32 pixels
  - answers come back without a thinking block, and captions are written
- The UI, downloads, stop/resume, and TrainFlow's handling of the files are covered by tests: `tests/test_toriigate_captioner.py` and `tests/e2e/test_captioner_ui.py`.

**Not checked yet:**
- **Caption quality:** the real model hasn't been run on a GPU.
- **VRAM use** on an 11 GB card.
- **Whether the BF16 vision projector runs at full speed on RTX 20-series cards.** They have no native BF16.

To run the real-model test yourself:

```
set TG_LLAMA_SERVER=...\runtime\llama.cpp\llama-server.exe
set TG_MODEL=...\models\SleepVeryHard__ToriiGate-0.5_GGUF\ToriiGate-0.5_Q8_0.gguf
set TG_MMPROJ=...\models\SleepVeryHard__ToriiGate-0.5_GGUF\mmproj_BF16.gguf
set TG_REAL_MODEL=1
python_embeded\python.exe tests\test_toriigate_real_llama.py
```

Third-party parts and their licences are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
