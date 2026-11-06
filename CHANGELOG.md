# Changelog

All notable changes to this fork of [ThetaCursed/Anima-TrainFlow](https://github.com/ThetaCursed/Anima-TrainFlow) are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.9] - 2026-11-05

Fixes from a full code review. Every item below was reproduced first, then fixed and covered by a test.

### Fixed
- **↩ Resume Run and ➕ Train More with Gradient Accumulation 2 or more could finish without training.**
  - **Cause:** the bundled sd-scripts worked out which epoch to resume in by dividing micro-batches by optimizer steps. With accumulation ≥ 2 that overshot by the accumulation factor.
  - **When it hit:** with the datasets the app builds (one epoch per run), a run resumed past roughly 1 ÷ accumulation of its length skipped the whole training loop. Train More with accumulation ≥ 2 always did.
  - **What you saw:** the trainer immediately saved a "final" LoRA at the step it resumed from, and the app reported ✅ Finished.
  - **Fix:** one line in `train_network.py` (`epoch_to_start = initial_step // len(train_dataloader)`). The test runs the trainer's own resume code for accumulation 1, 2 and 4 and checks the run reaches its last step.
- **Names that differ only in letter case could destroy an original on Windows.** Windows treats `Cat.png` and `cat.png` as the same file; bucketing's clash check didn't.
  - `Cat.png` + `cat.gif`: the GIF frame's crop overwrote your `Cat.png`, and no copy survived anywhere.
  - `Cat.png` + `cat.jpg`: one of the two silently dropped out of the dataset.
  - Bucketing now compares names ignoring letter case and refuses such pairs before changing anything. That includes pairs already sitting in `original_images/` from an earlier version.
  - It also refuses a new file of any extension named like an existing original, e.g. a new `dog.webp` next to an earlier `dog.jpg`. Before, such a file was skipped and trained on without being bucketed.
- **🔢 Rename to Numbers left captions behind when their letter case differed** (`IMG_0001.JPG` with `img_0001.txt` became `001.JPG` with no caption). Captions, sidecars and originals now pair with their image regardless of case. Images whose names differ only in case are refused, since on Windows they share one caption.
- **Rename to Numbers could leave a dataset half-renamed.** A file locked by another program (image viewer, antivirus, OneDrive) stopped it part-way, with hidden `.tf_rename_*` files left behind.
  - It now tracks every file and puts everything back. It never overwrites a file while undoing.
  - If a file stays locked, it lists the exact renames to do by hand, in an order that's safe to follow.
- **Training and the dataset tools could work on the same folder at once.**
  - 🚀 Start Training, ↩ Resume Run and ➕ Train More now wait for Bucketing, Rename or Captioning on that dataset to finish.
  - ✂️ Start Bucketing and 🔢 Rename refuse the dataset of a running trainer.
  - Tools on *another* folder still run. GPU tools only warn that they share VRAM. Rename no longer shows that warning, since it doesn't use the GPU.
  - 🔎 Inspect Dataset never blocks anything.
- **💾 Halt & Save resume points were never deleted.** sd-scripts only cleans up resume points at multiples of *Save Every n Steps*.
  - A halt's resume point is now removed once a newer one exists, or when the run finishes.
  - With periodic resume points off, a second halt replaces the first.
- **Duplicates:** when only a copy being set aside had a caption, your caption went to `duplicates/` and the kept image had none. Now the kept copy gets a copy of it (from the largest captioned copy). This shows in the log and in `duplicates.log`. A caption named in a different letter case is now set aside with its image too.
- **WD tagger captions had backslashes before parentheses** (`hatsune miku \(cosplay\)`). Now they're written plain (`hatsune miku (cosplay)`), like embedded tags.
  - sd-scripts passes captions to the text encoder as they are.
  - Forge Neo removes a prompt's backslashes before encoding (checked in its Anima text engine), as A1111 and ComfyUI do.
  - So an escaped caption trained on backslashes that never appear at generation time. 🔎 Inspect Dataset flags captions that still contain them.
- **Network Rank 0 or empty** crashed Start Training (division by zero in the alpha log line) and left an empty run folder.
  - All numeric fields are now checked before anything is written: rank, steps, save/preview intervals, batch size, accumulation, preview size/steps/seed/CFG and learning rate.
  - Each problem gets a message naming the field.
- **Min Side / Max Side** were used as typed: Min 500 produced buckets up to 820 px, above Max and off the 64-px grid, and Min > Max cropped everything square. Now Min is rounded up and Max down to multiples of 64, with a note, and Min > Max is refused.
- **A video/GIF whose source couldn't be moved** (file locked) left its PNG original behind, despite the "unchanged" message, and every later bucketing run then aborted with a name clash. The PNG is removed again, so the next run simply converts it.
- **➕ Train More when the trainer can't be launched** had already renamed the final LoRA and raised the step target. The run is now put back exactly as it was.
- **`settings.json`:**
  - It's written to a temporary file and swapped in, so closing the app mid-write can't truncate it. Concurrent saves are serialised.
  - An unreadable file is kept as `settings.unreadable_<time>.json` with a warning. Before, it was silently replaced by defaults, losing your model paths.
- **Previews and the checkpoint grid failed to load when `app.py` was started from another folder.** Gradio only served files under the working folder or temp. The launch now allows `training/output`. `start_trainer.bat` was never affected.
- **Preview prompts:** the trigger is matched as a tag, so trigger `cat` is now added to `catgirl, smile`. Before, the text-prefix check matched it and skipped the trigger.
- 🔎 **Inspect Dataset:**
  - Reports images whose names differ only in letter case, and captions named in a different letter case than their image (only Windows pairs those) instead of calling them orphans.
  - No longer writes the duplicate-check cache into the folder it inspects.

### Changed
- `requirements.txt` lists `pandas` and `psutil`, which the app imports directly. Both were already installed as dependencies of Gradio and accelerate, so re-running `Install_Requirements.bat` isn't required.
- The duplicate `weighting_scheme` entry in the hidden settings is gone. The value that took effect, `logit_normal`, is unchanged, and Anima's loss weighting treats it the same as `none`.

### Added
- **Test suite in `tests/`.** Until now it only existed outside the repository.
  - 14 scripts covering about 260 checks: bucketing, duplicates, media conversion, embedded tags, rename, the trainer patch, the training flow, an end-to-end run through the Gradio API with a stand-in trainer, and an optional headless-browser UI check.
  - No GPU or models are needed. See [tests/README.md](tests/README.md).

### Verified
- `tests/run_all.py` passes in a fresh Python 3.10 environment built from `tests/requirements.txt`. The browser check passes in headless Chromium.
- **Letter case:** the clash and rename fixes were also checked on a case-insensitive filesystem that behaves like NTFS.
- **Rename undo:** checked at every failure point, with up to three files staying locked.
- **Not yet confirmed:** the gradient-accumulation resume fix on a real GPU run. The test executes the trainer's real resume code with the app's dataset sizes, but not the full trainer.

## [1.8] - 2026-11-05

### Added
- **`duplicates.log`.** Each duplicate check that finds something appends a timestamped section to `<dataset>/duplicates.log`, whether or not *Set duplicates aside* is on.
  - **Contents:** every group, with the `KEEP` line, then a `COPY` line per extra copy giving its size, the reason (`identical file`, `same picture`, `mirrored copy`) and its path.
  - **Set-aside copies** point to where the original went, e.g. `-> duplicates/original_images/x.jpg`.
  - **Complete:** the on-screen log stops at 30 groups; the file doesn't.
  - **No repeats:** a report-only result identical to the previous section isn't appended again, so re-running bucketing without moving anything doesn't fill the file.
  - **Nothing found, nothing written.**
  - It's a `.log` rather than a `.txt`, so the inspector, captioning and training don't treat it as a caption; format probing and 🔢 Rename to Numbers ignore it too.

### Verified
- Report-only and set-aside sections, the unchanged report-only re-run, 35 groups written in full, and no file when nothing is found.
- The inspector, bucketing and the rename tool all leave the log alone.
- Report-only mode in the browser.
- All earlier test suites pass.

## [1.7] - 2026-11-04

### Added
- **Network Alpha** field next to Network Rank, saved with your settings.
  - Before this, the app always wrote `network_alpha = network_rank`, so the LoRA output scale, alpha ÷ rank, was fixed at 1.0.
  - The default, **0**, means "same as rank" and keeps exactly that behaviour, so existing setups train as before.
  - Any positive value is passed through, fractional or above the rank.
  - The start of each run logs the rank, the alpha and the resulting scale.
  - ↩ Resume Run and ➕ Train More keep the alpha the run was started with, because they use that run's own config.

### Verified
- The generated config for alpha left at 0, empty, 8 at rank 32, 0.5, and 64 above the rank.
- In the browser, the field keeps its value across a page refresh, and tab switching has no loop errors.
- All earlier test suites pass.

## [1.6] - 2026-11-04

### Added
- **Duplicate check in ✂️ Start Bucketing.** It covers every image in the dataset, new and from earlier runs.
  - **Exact duplicates:** the same SHA-256.
  - **Likely duplicates** — re-saved, resized, JPEG/WebP-recompressed, watermarked or mirrored copies — must pass every test:
    - a 64-bit difference hash within 10 bits, and the same aspect ratio within 2%;
    - **colour:** the 90th percentile of per-pixel colour difference between 16×16 thumbnails ≤ 18/255. This tells colour variants apart (same drawing, different hair colour), which a grayscale hash can't;
    - **structure:** the correlation of contrast-normalised 64×64 grayscale ≥ 0.93. This tells different line-art sketches apart, which thumbnails can't, because thin lines vanish when shrunk;
    - images with no structure at all (solid fills) only match within 4 levels of colour.
  - **Keeping:** the copy with the most pixels is kept; on a tie, the one with a caption, then the larger file. When a new download is larger than an image bucketed earlier, the new one is kept.
  - **Set duplicates aside** (on by default, saved with your settings) moves the other copies to `duplicates/`, with their dataset files, captions, originals, sources and `.xmp` sidecars, and drops them from bucketing's records. Nothing is deleted. Unticked, the groups are only listed.
  - Results are cached in `original_images/.duplicate_check_cache.json`, keyed by file size and modification time.
- 🔎 Inspect Dataset lists duplicate groups.

### Fixed
- The bucketing summary's "Total Dataset Size" left out PNGs that were already bucket-sized, which bucketing leaves in place.

### Verified
- **No false positives** in 12,346 pairs of different images, including hash-similar pairs and mirrored comparisons:
  - real photos, including a stereo pair (`motorcycle_left`/`right`);
  - 120 flat-colour artworks with near-identical layouts;
  - 80 same-pose line-art sketches;
  - solid colours.
- **Caught:** half-size, JPEG 60, WebP 70, half-size + JPEG 80, watermarked and mirrored copies of every test image.
  - Misses: 2 of 186 quarter-size + JPEG 70 copies, and a 1-pixel chessboard pattern that aliases when halved.
  - Brightness edits and crops are mostly not flagged, by design.
- **Design history:** a first version that used only a grayscale hash and colour thumbnails flagged every pair of different sketches (3,213 of 3,213). That's why the structure test exists.
- **In the bucketing flow:** a new copy larger than an image bucketed earlier, and a smaller one, the report-only mode, re-runs (nothing more found, nothing regenerated, cache untouched), Rename to Numbers afterwards, the inspector, and the browser UI.
- **Speed:** 600 full-size JPEGs take 2.1 s on the first check and 0.03 s cached.

## [1.5] - 2026-11-04

### Added
- **Captions from embedded tags.** *Create .txt captions* first uses tags already saved in the image, e.g. by Grabber, and runs the WD tagger only for images without any. When every image has them, the WD model isn't loaded at all.
  - **Read from**, matching what Grabber writes through exiftool (`-sep ";"`) or the Windows Property System:
    - XMP `dc:subject` (`Subject`), or XMP `pdf:Keywords` (`Keywords` on GIF/MP4)
    - IPTC Keywords (`Keywords` on JPG/PNG; exiftool hex-encodes it inside PNG)
    - EXIF `XPKeywords` (Windows `System.Keywords`)
    - `.xmp` sidecars named `image.jpg.xmp` or `image.xmp`
  - **The original is read too.** After bucketing, the caption comes from the original or source in `original_images/`, so tags saved in GIFs and MP4s are used for their extracted frame.
  - **One source per image.** The `exiftool -Keywords -Subject -w txt` approach writes both fields, so a file with both gets every tag twice; here only one is used.
  - **Splitting:** comma, semicolon and space-separated lists are all handled, which covers Grabber's default `%all%` and `%all:separator=,%` and replaces a comma-fixing pass.
  - **Anima caption format**, from the model card:
    - Lowercase, with spaces instead of underscores; score tags and kaomoji keep their underscores.
    - Artists get an `@` prefix.
    - Order: meta, 1girl/1boy/1other, character, series, artist, general. Count tags are moved to the front even without namespaces.
    - Parentheses aren't escaped (`hatsune miku (cosplay)`).
  - **Namespaces:** Grabber's `%all:includenamespace,unsafe%` form (`artist:name`) is understood, mapped from Grabber's own categories (artist/circle → artist, character/model, copyright → series, meta, general/species/…).
    - The underscore form without `unsafe` (`artist_name`) is only treated as namespaced when every tag carries one, because real tags such as `artist name` look the same.
  - Checkbox *Use tags embedded in the images first*, on by default and saved with your settings.
  - 🔎 Inspect Dataset reports how many images carry embedded tags.
- **🔢 Rename to Numbers.** Renames images to `001`, `002`, … in filename order, like a pair-renaming script, but keeps everything linked.
  - Captions, originals and sources in `original_images/`, `.xmp` sidecars and the bucketing manifest are renamed together, so re-bucketing finds nothing new and captioning still finds embedded tags.
  - It goes through temporary names, so overlapping names (`002` → `001` while `1` → `002`) can't collide.
  - It writes `rename_map_<time>.json` first.
  - It changes nothing when another file already uses a new name, such as an orphan `002.txt` that would otherwise become an image's caption.
  - It refuses to run during training.

### Changed
- `.xmp` sidecars are no longer probed as images by bucketing.

### Verified
- **Test files:** written by exiftool 13.60 with Grabber's exact arguments.
  - Your comma-separated `Keywords`, both fields on one file, and the namespaced colon and underscore forms.
  - A Japanese tag via XMP, sidecars in both naming styles, Windows keywords with a score tag, kaomoji, IPTC inside PNG, parentheses.
  - GIF and MP4 sources read after bucketing.
- **Captioning paths:** all embedded (WD never loads), the checkbox off, and existing captions kept.
- **Rename:** it keeps re-bucketing and captioning working, and handles overlapping names. An orphan caption or sidecar with a target name aborts it.
- **UI:** run end to end in the browser.
- **Confirmed:** exiftool stores non-ASCII IPTC keywords as `?`, and rejects the dotted key names from Grabber's docs.

## [1.4] - 2026-11-02

**Updating:** re-run `Install_Requirements.bat`, or run `python_embeded\python.exe -m pip install pillow-jxl-plugin==1.3.8 pillow-heif==1.8.1`, for JPEG XL and HEIC support.

### Added
- **✂️ Start Bucketing converts videos, GIFs, AVIF, JPEG XL and HEIC/HEIF images, and by default any other format Pillow or OpenCV can read.** Before cropping, each one becomes a full-resolution PNG original in `original_images/`, and the crop step treats it like any other original.
  - **What gets saved:** the middle frame of a video or animation, or the decoded image otherwise.
  - **Other multi-frame files:** a multi-page TIFF gives its first page, and a layered PSD its flattened image rather than a single layer.
  - **Source files** are moved to `original_images/` too and never modified or deleted.
  - **Captions:** a `clip.txt` written for `clip.mp4` pairs with the resulting `clip.png`.
  - **Videos:** `.mp4`, `.m4v`, `.mov`, `.webm`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.mpg`, `.mpeg`, decoded by the FFmpeg bundled with the pinned `opencv-python` 4.10.0.84.
    - Phone videos with a rotation flag come out upright.
    - If a container reports no frame count, the frames are counted by decoding.
    - On Windows, if OpenCV can't open a non-ASCII path, the 8.3 short path and then a temporary ASCII-named copy are tried.
  - **GIF and AVIF** are decoded by Pillow 12.3.0, whose Windows wheel includes native AVIF support. Animated files give their middle frame; transparency is flattened onto white.
  - **JPEG XL** is decoded through **`pillow-jxl-plugin` 1.3.8, a new dependency**. It has a prebuilt Windows wheel for Python 3.10, its only dependency is `packaging`, and it leaves Pillow at 12.3.0.
    - With the plugin installed, sd-scripts also reads `.jxl` itself, so pre-flight doesn't count JXL files as skipped.
    - Without the plugin, a `.jxl` is reported with the fix rather than failing silently.
  - **HEIC/HEIF** (`.heic`, `.heif`, `.hif`, e.g. iPhone photos) is decoded through **`pillow-heif` 1.8.1, a new dependency**.
    - Its Windows wheel for Python 3.10 bundles libheif with its decoder. It requires `pillow>=11.1.0`, adds no other packages, and leaves Pillow at 12.3.0.
    - **Rotation is applied once.** iPhone photos carry both the container's rotation and EXIF orientation 6; pillow-heif applies the rotation and resets the EXIF tag, so the app doesn't rotate them a second time.
    - A multi-image HEIC gives its main (primary) image.
    - Without the plugin, a `.heic` is reported with the fix.
  - **Any other file** is probed, unless *Also convert any other format…* is unticked (on by default).
    - Pillow is tried first, then OpenCV for video.
    - Examples: TIFF, PSD, TGA, `.jfif`, `.jpg_large`, and `.3gp`, `.ts` or extensionless videos.
    - The log names the detected format (`TIFF scan.tiff → converted (first page of 2)`).
    - Files nothing can decode stay where they are, with one log line. Audio-only files aren't mistaken for video.
    - Caches, captions, configs and archives aren't probed at all. Neither are playlist and subtitle files, which FFmpeg would try to follow.

### Changed
- **Name clashes:** the bucketing check now covers these files, and aborts before changing anything.
  - Two sources with the same name (`clip.mp4` + `clip.gif`, or `clip.mp4` + `clip.jpg`).
  - A new file named like an existing original.
- **Failed conversions:** an unreadable file is reported and left in the dataset folder unchanged, and retried on the next run.
- **Unconverted files are pointed out everywhere they matter.** sd-scripts ignores all of these formats except `.jxl` with the plugin; it only reads AVIF with the `pillow_avif` plugin, which isn't installed.
  - 🔎 Inspect Dataset reports them, and no longer calls their captions orphans.
  - Pre-flight warns that they won't be trained on.
  - The tagger explains why a folder of videos has "no images".

### Verified
- On Python 3.10 with the bundled versions (Pillow 12.3.0, opencv 4.10.0.84, Gradio 6.17.3).
- **Formats:** H.264 MP4, MKV and MOV, VP9 WebM, MJPEG AVI, animated and single-frame GIF, and AVIF with transparency.
  - Each test frame encodes its index in its brightness, so the exact frame could be checked: frame 31 of 61 every time, including a non-keyframe seek.
- **A phone clip with a 90° rotation flag** decodes identically to ffmpeg's own output.
- **Other formats:**
  - JXL with transparency.
  - A 2-page TIFF, checked by page brightness.
  - A two-layer PSD, checked by pixel values that the composite was saved.
  - TGA, `.jfif` and `.jpg_large`.
  - `.3gp`, `.ts` and extensionless videos, all giving frame 31 of 61.
  - Left alone: an audio-only `.mka` and a random binary.
  - Not probed at all: `.npz`, `.json` and `Thumbs.db`.
  - Checked both with the checkbox on and off, and with the JXL plugin missing.
- **HEIC:**
  - A landscape photo stays landscape.
  - An iPhone-layout portrait (container rotation plus EXIF orientation 6) is rotated exactly once and gets a portrait bucket.
  - A three-image file whose main image is the second gives that image.
  - `.heif` and `.hif` work, and a missing plugin is reported.
- **Not tested:** animated JXL, since the plugin can't write a test file. Reading one uses the same middle-frame code as GIF. HEIC test files were written by pillow-heif rather than taken from an iPhone.
- **Speed:** a 3-minute 1080p H.264 file takes about 0.5 s.
- **Behaviour:** re-running doesn't reconvert anything, and re-cropping at new sizes works from the frame originals.
- **Clashes and failures:** both clash cases abort with nothing changed, and a damaged file is reported and left in place.
- **UI:** run end to end in the browser.
- **Not tested on Windows:** the non-ASCII path fallbacks. They were only exercised on Linux by making OpenCV refuse the path.

## [1.2.1] - 2026-10-16

Published as GitHub release `1.3`.

**Updating from 1.2:** re-run `Install_Requirements.bat`, or run `python_embeded\python.exe -m pip install gradio==6.17.3`, so the new Gradio version is installed.

### Fixed
- **Switching tabs (Previews / Checkpoint Grid / Loss) froze, then crashed, the browser tab.**
  - On Gradio 6.14.0 this app's page went into an endless Svelte update loop (`effect_update_depth_exceeded`, over 1,000 errors in seconds) on any tab switch, whether training or idle.
  - The trigger was the page-load event that restores saved settings combined with the tabs. A trivial two-tab app doesn't show it.
  - Reproduced in headless Chromium: the same app also fails on Gradio 6.15.2 and works on 6.17.3 and 6.30.0.
- Gradio is now pinned to **6.17.3**. From 6.18.0 on, Gradio requires `huggingface-hub>=1.2`, which conflicts with sd-scripts' pinned `huggingface-hub==0.34.3`.

### Documentation
- The README screenshot now shows the fork's own interface, mid-run on the Loss tab (it showed upstream's).
- README Status lists the browser testing.

### Verified on 6.17.3
- On Python 3.10:
  - the end-to-end and bucketing test suites;
  - in the browser: switching between all three tabs mid-run;
  - the loss chart drawing and updating;
  - previews;
  - settings still restored after a page refresh.

## [1.2] - 2026-10-16

### Documentation
- README rewritten for the fork:
  - a comparison table of what differs from upstream
  - install steps that use this fork's own release zip or git clone and its bundled Python 3.10, instead of upstream's portable build, which contains upstream's code
  - direct model download links (official `circlestone-labs/Anima` files; just the two tagger files instead of a 3.78 GB repo clone)
  - a usage guide for the new controls and the output layout
- Changelog versions renumbered to match the release tags (1.1, 1.2).

### Added
- **➕ Train More** — extends the keyword's newest finished run to the current *Training Steps* value, resuming from sd-scripts' end-of-run state (`<name>-state`).
  - The previous final LoRA is kept as `<name>-step<N>[-final].safetensors`.
  - The finished config is backed up as `configs/<name>_training.before_extend_<N>.toml`.
  - With Prodigy (constant LR) training simply continues. With AdamW the cosine schedule is rebuilt for the new length (torch restores the schedule's position, not its length), so the LR jumps back up to where that schedule stands at the old step, then decays to 0 again.
- **Block Swap (Low VRAM)** slider (0–26) — writes `blocks_to_swap`, which the Anima trainer already supported; the app defined the setting but never passed it. Anima 2B has 28 blocks and sd-scripts allows swapping up to `num_blocks - 2`.
- **Multiple preview prompts** — one per line in the prompt box, each prefixed with the trigger. Previously newlines were joined into a single prompt.
- **Checkpoint Grid tab** — composites every preview into a steps × prompts image (`<run>/checkpoint_grid.png`) and marks steps that saved a LoRA.
- **Loss tab** — live chart of the moving-average loss parsed from the trainer's progress line. History is rebuilt from the run's logs, so it continues across halts, resumes and app restarts.
- **🔎 Inspect Dataset**, which reports:
  - missing, empty and orphaned captions
  - unreadable, ≥2048px, <512px and EXIF-rotated images
  - same-name clashes and repeated tags
  - captions already containing the trigger
  - the 20 most frequent tags, and tags present in every caption
- Preview gallery captions show step and prompt.

### Changed
- **The trainer outlives the app.** It is launched in its own process group, with its own hidden console on Windows, and writes directly to `<run>/logs/train_*.log`, which the app tails.
  - Closing the console window, a crash, or restarting `app.py` no longer ends a run.
  - `<run>/trainer.json` records the process; on startup the app reattaches to a trainer that is still running. Reattaching checks the process start time as well as the PID, so a reused PID isn't mistaken for the trainer.
  - Pause state, halt requests and loss history survive the reattach.
- Bucketing, auto-captioning and inspection run as background jobs, so a page refresh doesn't interrupt them. One job runs at a time, and the status line shows it. Starting one while training warns that the GPU is shared.
- If every caption already contains the trigger tag, no `caption_prefix` is added. When only some captions contain it, the app warns.
- **↩ Resume Run** also considers the end-of-run resume point, so an interrupted Train More can be resumed.

### Fixed
- `.bmp` images: the tagger and the resolution scan skipped them, while pre-flight required captions for them. All tools now share sd-scripts' extension list.
- The default DiT path now points at `models/anima/dit/anima-base-v1.0.safetensors`; the official model card says LoRAs should be trained on the base model. Saved settings are unaffected.

### Known limitations
- After reattaching, the exit code may be unavailable. The outcome is then read from the log and the final LoRA's presence instead.
- Two app instances started at once would both reattach to the same trainer.

## [1.1] - 2026-10-16

First fork release. Based on upstream commit `cf13c7b` (upstream portable release v1.1.0).

### Added
- **⏸ Pause / ▶ Continue** — freezes the trainer process tree in place and continues exactly where it stopped. VRAM stays allocated while paused; a pause does not survive closing the app.
- **💾 Halt & Save** — the trainer finishes its current optimizer step, saves a step LoRA plus a full resume point (LoRA weights, optimizer state, LR scheduler, RNG, step count) and exits, freeing VRAM. Implemented as a halt-request file polled once per step (`--halt_request_file` in the bundled sd-scripts).
- **↩ Resume Run** — continues the newest unfinished run of the current keyword from its newest complete resume point (`--resume` + `--skip_until_initial_step`), using the configs that run was started with rather than the current UI values. Finished runs are not resumable.
- **Save resume points with each checkpoint** (on by default) — writes a resume point alongside every periodic checkpoint and keeps the two newest, so a crash, power loss or closed console can be resumed from the last checkpoint.
- **Status line** showing state, step and run folder; it survives page refreshes like the log.
- **Per-run log file** at `<run>/logs/train_<timestamp>.log` with the full trainer output.
- **Pre-flight warning** for images carrying an EXIF rotation flag, which sd-scripts would otherwise train on sideways or mirrored.

### Changed
- **Training is decoupled from the browser tab.** The trainer subprocess is owned by the app; a background thread drains its output; tabs poll the log, previews and status once a second. Refreshing, closing or opening extra tabs no longer affects a run.
- **Output layout:** one folder per run at `training/output/<keyword>/<YYYYmmdd_HHMMSS>/` containing the LoRAs, `sample/`, `configs/`, `logs/`, resume points and `run.json`. Re-running a keyword never overwrites an earlier run, and the preview gallery only shows the current run's samples.
- **Keyword folder names keep non-Latin text** (e.g. Japanese); only characters Windows forbids are replaced. Previously any keyword without Latin letters or digits collapsed into a shared `untitled` folder.
- **📁 Checkpoint Folder** opens the active (or newest) run folder of the keyword.
- **Bucketing decodes images with PIL** instead of `cv2.imread`: EXIF orientation is applied, non-ASCII file paths work on Windows, transparency is flattened onto white (previously whatever colour sat behind transparent pixels, often black), and the subject-detection model (U2Net) now receives RGB input as rembg feeds it (previously BGR). Crops may differ slightly from earlier versions.
- Bucketing writes outputs atomically (temporary `.part` file, then replace).
- The tagger tags the EXIF-upright image.
- A Save or Preview interval of 0 now disables that interval instead of making sd-scripts divide by zero.

### Fixed
- **Refreshing the page killed training.** The disconnected tab's output reader was closed and the app dropped its handle on the trainer, which then blocked on a full output pipe (still holding VRAM), could not be stopped, and did not stop a second Start from launching on top of it.
- **Re-running bucketing destroyed originals.** Previous outputs in the dataset folder were moved into `original_images/`, overwriting PNG originals with their low-resolution crops and adding crops of JPG originals as extra "originals". Outputs are now regenerated from the originals; a manifest (`original_images/bucketing_manifest.json`) records what bucketing wrote so hand-edited outputs are kept, and same-name originals (`cat.jpg` + `cat.png`) abort before any file is moved.
- **Unreadable images vanished during bucketing** — they were counted as processed after being moved out of the dataset. They are now reported and their originals left in `original_images/`.
- **Phone photos were cropped in the wrong orientation** — the bucket was chosen from the stored (unrotated) size while the pixels were decoded rotated.
- **The log gained one line per training step**; consecutive refreshes of any progress bar now update a single line.
- **Trigger prefix produced `trigger,  tags`** (double space); now `trigger, tags`.
- **Pre-flight oversized-image check never fired** — it read a non-existent `img.name` attribute inside a bare `except`.
- **Run end was always reported as "finished or stopped"**; it now reports finished, stopped by user, halted, or failed with the exit code.
- **sd-scripts: resuming with gradient accumulation > 1 stopped short of the step target** (e.g. 2100 of 2400 steps at accumulation 2 after a halt at step 300), because the step counter was restored in micro-batches. It now resumes at the saved optimizer step, and the progress bar shows the absolute step.

### Known limitations
- A resumed run sees the dataset in a different order than an uninterrupted run would.
- Resume points include the optimizer state and are considerably larger than a LoRA. sd-scripts also saves one when a run finishes, and a Halt resume point between save intervals is never pruned automatically; both are safe to delete.
- A Halt pressed during latent caching takes effect after the first training step.
- Bucketing and auto-captioning still run attached to the browser tab (fixed in 1.2).
