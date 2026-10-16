# Changelog

All notable changes to this fork of [ThetaCursed/Anima-TrainFlow](https://github.com/ThetaCursed/Anima-TrainFlow) are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v1.1.0-fork.1] - 2026-10-16

Based on upstream commit `cf13c7b` (portable release v1.1.0).

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
- The bundled sd-scripts changes (halt request, resume step count) have not yet been run on a GPU with the real model; they were compiled, simulated and tested against a stand-in trainer. Check with a short run: halt around step 30 of 100, resume, and confirm the log continues at `31/100`.
- A resumed run sees the dataset in a different order than an uninterrupted run would.
- Resume points include the optimizer state and are considerably larger than a LoRA. sd-scripts also saves one when a run finishes, and a Halt resume point between save intervals is never pruned automatically; both are safe to delete.
- A Halt pressed during latent caching takes effect after the first training step.
- Bucketing and auto-captioning still run attached to the browser tab.
