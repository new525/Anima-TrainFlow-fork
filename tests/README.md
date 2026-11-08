# Tests

These tests run the real `app.py` without a GPU and without the models. torch is replaced by a stub that reports a GPU, U2Net by a plain resize, and the WD tagger by a fixed caption where needed. The end-to-end test replaces sd-scripts with `e2e/fake_trainer.py`, which writes the same files and prints the same progress lines.

## Running

Use a separate Python 3.10+ environment. The app's `python_embeded` works too, but this keeps the test packages out of it.

```
python -m venv .venv
.venv\Scripts\pip install -r tests\requirements.txt        (Linux/macOS: .venv/bin/pip)
.venv\Scripts\python tests\run_all.py
```

`run_all.py` runs every script in its own process and prints one line per script. It exits with code 1 if anything failed. To run only some scripts, pass part of their name, e.g. `run_all.py rename e2e`. Any single script also runs on its own, e.g. `python tests\test_rename.py`. It prints a `PASS`/`FAIL` line for each check.

| Script | Covers |
|---|---|
| `test_bucketing.py` | Originals preserved on re-runs, phone-photo rotation, name clashes (including names that differ only in letter case), edited outputs kept, Min/Max Side validation |
| `test_duplicates.py` | Duplicate detection (true copies found, look-alikes kept), caption carried over to the kept copy, read-only inspector |
| `test_duplicates_log.py` | `duplicates.log` in report-only and set-aside mode |
| `test_embedded_tags.py` | Grabber/exiftool tags → captions; WD tagger only for untagged images; renaming keeps everything linked |
| `test_heic.py`, `test_media_*.py`, `test_other_formats.py` | Video/GIF/AVIF/JXL/HEIC/PSD/TIFF… conversion, frame extraction, messages |
| `test_rename.py` | Rename to Numbers: undo when a file is locked part-way, letter-case pairing, training interlock |
| `test_trainer_patch.py` | The sd-scripts patch: resume/Train More with gradient accumulation, halt-state cleanup (runs the real trainer code, extracted) |
| `test_trainer_cache.py` | sd-scripts' real dataset and caching code: the trigger word reaches the cached captions; edited captions, edited images and renamed datasets are encoded again. Needs the trainer's packages: on Windows it runs itself with `python_embeded\python.exe`, elsewhere it reports SKIP |
| `test_training_flow.py` | Input validation, training ⇄ tool interlock, Train More rollback, settings.json safety, preview prompts, WD caption format, inspector checks |
| `test_toriigate_captioner.py` | The ToriiGate captioner (`tools/toriigate-captioner`), against `e2e/fake_llama_server.py`, which stands in for llama-server. Covers:<br>• the bundled `prompts.py` is the official file<br>• hints from Grabber tags and `.txt` captions, and name modes<br>• image preparation<br>• the llama-server lifecycle<br>• captioning, skip, overwrite, stop<br>• error handling<br>• resumable downloads and the llama.cpp zip<br>• the CLI<br>• TrainFlow's handling of `.toriigate.txt` |
| `test_toriigate_real_llama.py` | Optional: the captioner against a real llama-server and model. Skipped unless `TG_LLAMA_SERVER`, `TG_MODEL` and `TG_MMPROJ` are set; `TG_REAL_MODEL=1` adds caption-content checks for the real ToriiGate model |
| `e2e/test_e2e.py` | The app served by Gradio and driven through its API: the trainer outliving the app, reattach, pause/halt/resume, grid, loss, Train More, background tools |
| `e2e/test_captioner_ui.py` | The captioner's web UI served by Gradio and driven through its API. Covers:<br>• caption folder, try one image<br>• a page reload during a run, and stop<br>• keeping the model loaded, and unload<br>• settings surviving a restart |
| `e2e/test_browser.py` | Optional: the UI in a headless browser (tab switching mid-run, loss chart, settings restored after reload) |

## Browser test (optional)

`e2e/test_browser.py` is skipped unless Node.js is installed and `npm install` has been run in `tests/browser`. On Windows or macOS, set `CHROME_PATH` to an installed Chrome or Edge. For example:

```
set CHROME_PATH=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
```

On Linux it uses `@sparticuz/chromium` instead.

## Fixtures

`data/grabber/` holds images tagged the way Grabber writes them. `data/make_grabber_fixtures.sh` rebuilds them and needs exiftool. Videos and the other formats are generated on each run by `make_media.py` and `make_other_formats.py`. The real photos used by the duplicate tests come from scikit-image's bundled sample data.
