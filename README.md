# Anima TrainFlow

Anima TrainFlow is a streamlined, one-page GUI for training LoRA on the **Anima 2B** model. Optimized to run on hardware with as little as **6GB of VRAM**, it eliminates technical overhead by focusing on the essential settings that impact training results the most.

### ⚡ Everything you need in one tool
**A complete, ready-to-use workflow from start to finish.**

`Raw Images` ➔ `Smart Resize & Crop*` ➔ `Auto-Tag` ➔ `One-Click Train**` ➔ `Ready LoRA`

* **Smart Prep:** In most cases, it only performs resizing. Intelligent cropping is triggered only for extreme aspect ratios that don't fit into training buckets.
* **Fast Results:** Average training time is **~1 hour** (benchmarked on RTX 3060 12GB).

![Anima TrainFlow Interface Preview](preview.png)

## Quick Start (Portable)
1. [**Download Portable Version (3GB)**](https://huggingface.co/datasets/ThetaCursed/Anima-TrainFlow-Portable/resolve/main/Anima-TrainFlow-v1.1.0-Portable.7z?download=true)
2. **Extract the archive using [7-Zip](https://7-zip.org/) or WinRAR**.
3. Run `start_trainer.bat`.
4. Open the `🔧 Paths to Models <- Set Once` accordion and specify the paths to your model files.
5. Specify your **Dataset Path** (images + .txt files) and **Trigger Word**, then click **Start**.

## Manual Installation
If you prefer to set up the environment manually instead of using the portable version, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ThetaCursed/Anima-TrainFlow
   cd Anima-TrainFlow
   ```

2. **Install dependencies:**  
   Run `Install_Requirements.bat`

3. **Download Required Models:**
   Run the following commands from the root folder:

   *   **WD Tagger** (used for auto-captioning):
       ```bash
       git clone https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3 models/wd-eva02-large-tagger-v3
       ```
   *   **U2Net Model** (used for Smart Cropping):
       ```bash
       curl -L -o models/u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
       ```

4. **Launch the Trainer:** 
   Run `start_trainer.bat`

## Key Features
* **Zero-Tab Interface:** All critical parameters (Trigger Word, Rank, LR, Steps) are accessible on a single screen.
* **Live Training Previews:** Watch your LoRA improve in real-time. The built-in gallery automatically updates whenever a new sample is generated.
* **AI-Powered Smart Cropping:** Integrated U2Net model automatically performs subject-aware, head-first cropping and resizes images to optimal aspect-ratio buckets via multi-threading.
* **Built-in Auto-Captioning:** Integrated WD14 Tagger (EVA02 v3) automatically generates multi-threaded `.txt` tags for your dataset with customizable general and character thresholds.
* **Pre-Flight Validation:** Automatically scans the dataset for missing captions, oversized images (>=2048px), and missing model paths to prevent crashes before training starts.
* **Persistent Sessions:** All UI inputs, paths, and slider positions are instantly auto-saved and restored on the next launch.
* **Portable Edition:** Includes an embedded Python environment to avoid installation or complex setup.
* **Low VRAM Friendly:** Specifically tuned for 6GB+ NVIDIA GPUs with aggressive RAM/VRAM clearing between tasks.
* **Optimized Defaults:** Pre-configured for BF16 precision and latent caching to ensure maximum performance and stability.
* **Prodigy Native:** Intelligent Learning Rate handling and optimized defaults for the Prodigy optimizer.

## Fork Changes

### Training controls
* **Training survives the app:** The trainer runs as its own process and writes to its own log file. Refreshing or closing the browser tab, or closing and restarting the app itself, doesn't stop a run. A restarted app finds the running trainer and reattaches: log, previews, loss chart and status all continue.
* **⏸ Pause / ▶ Continue:** Freezes the trainer in place and continues exactly where it stopped. VRAM stays allocated while paused, and a pause does not survive closing the app.
* **💾 Halt & Save:** Finishes the current step, then saves a LoRA and a full resume point (weights, optimizer, scheduler, RNG, step) and exits, freeing the VRAM.
* **↩ Resume Run:** Continues the newest unfinished run of the current keyword from its newest resume point, using the settings that run was started with.
* **Resume points:** With *Save resume points with each checkpoint* on, a resume point is also saved with every checkpoint and the two newest are kept, so a crash costs at most one save interval.
* **➕ Train More:** Continues the keyword's newest finished run up to the current *Training Steps* value, from its end-of-run resume point. The previous final LoRA is kept, renamed to its step.
* **Block Swap (Low VRAM):** Keeps up to 26 of the DiT's 28 transformer blocks in system RAM. Each step is slower, but the model fits in less VRAM.

### Choosing a checkpoint
* **Multiple preview prompts:** Write one prompt per line; the trigger word is added to each.
* **Checkpoint Grid tab:** Builds one image of every preview step × prompt, marking steps that saved a LoRA.
* **Loss tab:** Live chart of the moving-average loss. It continues across halts, resumes and app restarts.
* Previews are captioned with their step and prompt.

### Outputs
* **One folder per run:** Each run gets `training/output/<keyword>/<date_time>/`, holding its LoRAs, `sample/`, `configs/`, `logs/` and resume points. Re-running a keyword never overwrites an earlier run. Non-Latin keywords keep their own folder.

### Dataset fixes
* **Bucketing is non-destructive:**
  * Re-running it with different sizes regenerates the dataset PNGs from `original_images/` instead of overwriting the originals.
  * Edited outputs are left alone.
  * Name clashes (`cat.jpg` + `cat.png`) abort before anything moves.
  * Unreadable files are reported instead of silently vanishing.
* **Phone photos:** EXIF orientation is applied when bucketing and tagging, so portrait photos are cropped and tagged upright. Training warns about EXIF-rotated images it would otherwise learn sideways.

### Dataset tools
* **🔎 Inspect Dataset** reports problems before training:
  * missing, empty or orphaned captions
  * unreadable, oversized, small or EXIF-rotated images
  * same-name clashes and repeated tags
  * captions that already contain the trigger
  * the 20 most frequent tags, and tags present in every caption
* If every caption already contains the trigger, training doesn't add it again.
* Bucketing, auto-captioning and inspection run in the background like training, so a refresh doesn't interrupt them. Progress shows in the log and the status line.
* `.bmp` is handled consistently: the tagger used to skip it while training required a caption for it.

### Smaller fixes
* The trigger word is prefixed as `trigger, tags` (was `trigger,  tags`).
* The progress bar updates in place instead of adding a log line every step.

## Dataset Preparation
Place all your training images (.png, .jpg, .webp) in a single folder. Every image must have a matching text file with the same name containing its tags/captions (e.g., `image1.png` and `image1.txt`). You can easily generate these text files using the built-in **Auto-Caption Dataset** tool.

## System Requirements
* **OS:** Windows 10/11.
* **GPU:** NVIDIA GPU (6GB+ VRAM recommended for Anima 2B training).
* **Storage:** ~6.5GB of free space (SSD recommended).

## Technical Details
* **Core:** Based on a modified version of `sd-scripts` for Anima 2B architecture.
* **UI:** Built with Gradio featuring a customized dark theme.
* **Backend:** Utilizes `accelerate launch` for optimized execution.
* **Auto-Save:** All paths and configurations are automatically saved to `settings.json`.