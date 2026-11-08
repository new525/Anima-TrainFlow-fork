"""Web UI for the ToriiGate 0.5 captioner (python captioner_ui.py). The work runs in a background thread, so
refreshing or closing the page doesn't stop it; the page polls for progress."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # the portable Python's ._pth leaves this folder off sys.path
import toriigate_captioner as tc  # noqa: E402  (puts the TrainFlow folder on sys.path too, for embedded_tags)
import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

SETTINGS = Path(os.environ.get("TG_CAPTIONER_SETTINGS") or tc.HERE / "settings.json")
OPTION_KEYS = [f.name for f in fields(tc.Options) if f.name != "timeout"]
SERVER_KEYS = [f.name for f in fields(tc.ServerConfig)]
UI_KEYS = ["folder", "try_image"] + OPTION_KEYS + SERVER_KEYS
DEFAULTS = {"folder": "", "try_image": "", **asdict(tc.Options()), **asdict(tc.ServerConfig())}


def load_settings() -> dict:
    s = dict(DEFAULTS)
    try:
        s.update({k: v for k, v in json.loads(SETTINGS.read_text(encoding="utf-8")).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def save_settings(values: dict) -> None:
    tmp = SETTINGS.with_name(SETTINGS.name + ".tmp")
    try:
        tmp.write_text(json.dumps(values, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS)
    except OSError:
        pass


class Job:
    """One background job at a time; keeps the llama-server session between jobs when asked to."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lines: list = []
        self.version = 0
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.status = "Idle."
        self.progress = ""
        self.preview: Image.Image | None = None
        self.caption = ""
        self.query = ""
        self.session: tc.Session | None = None
        self.session_cfg: dict | None = None

    # --- console
    def log(self, msg: str) -> None:
        with self.lock:
            self.lines.extend(str(msg).splitlines() or [""])
            del self.lines[:-400]
            self.version += 1

    def emit(self, e: tc.Event) -> None:
        if e.kind == "progress":
            with self.lock:
                self.progress = e.message
                self.version += 1
        elif e.kind == "caption":
            self.log(e.message)
            self.show(Path(e.image), e.caption, e.query)
        else:
            self.log(e.message)

    def show(self, image: Path, caption: str, query: str) -> None:
        try:
            with Image.open(image) as im:
                im.thumbnail((768, 768))
                thumb = im.convert("RGB")
        except OSError:
            thumb = None
        with self.lock:
            self.preview, self.caption, self.query = thumb, caption, query
            self.version += 1

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def get_session(self, cfg: tc.ServerConfig) -> tc.Session:
        wanted = asdict(cfg)
        same = self.session is not None and {k: v for k, v in wanted.items() if k != "keep_running"} == \
            {k: v for k, v in (self.session_cfg or {}).items() if k != "keep_running"}
        if not same:
            if self.session is not None:
                self.session.close(force=True)
            self.session = tc.Session(cfg, self.emit)
            self.session_cfg = wanted
        self.session.cfg.keep_running = cfg.keep_running
        return self.session

    def start(self, label: str, fn) -> str:
        if self.busy():
            return "⚠️ Still working — wait or press ⏹ Stop."
        self.stop_event = threading.Event()

        def run():
            with self.lock:
                self.status, self.progress = f"⏳ {label}…", ""
            t0 = time.time()
            try:
                fn()
                status = f"✅ {label} finished ({time.time() - t0:.0f} s)."
                if self.stop_event.is_set():
                    status = f"⏹ {label} stopped."
            except tc.SetupError as e:
                self.log(f"❌ {e}")
                status = f"❌ {label} failed: {str(e).splitlines()[0]}"
            except Exception as e:  # keep the UI alive whatever happens
                self.log(f"❌ {type(e).__name__}: {e}")
                status = f"❌ {label} failed: {e}"
            finally:
                if self.session is not None and not self.session.cfg.keep_running:
                    self.session.close()
            with self.lock:
                self.status = status
                self.progress = self.progress.split(" · ")[0]   # no "about N min left" once it has ended
                self.version += 1
        self.thread = threading.Thread(target=run, daemon=True, name="captioner")
        self.thread.start()
        return ""


job = Job()
atexit.register(lambda: job.session.close(force=True) if job.session else None)


def split_values(values) -> tuple:
    v = dict(zip(UI_KEYS, values))
    opts = tc.Options(**{k: v[k] for k in OPTION_KEYS})
    opts.max_tokens, opts.seed = int(opts.max_tokens or tc.MAX_TOKENS), int(-1 if opts.seed is None else opts.seed)
    cfg = tc.ServerConfig(**{k: v[k] for k in SERVER_KEYS})
    cfg.ctx = int(cfg.ctx or 8192)
    return v, opts, cfg


def start_captioning(*values):
    v, opts, cfg = split_values(values)
    save_settings(v)
    folder = Path(str(v["folder"]).strip().strip('"'))
    if not folder.is_dir():
        job.log(f"❌ Not a folder: {folder}")
        return
    session = job.get_session(cfg)
    msg = job.start("Captioning", lambda: session.caption_folder(folder, opts, job.stop_event))
    if msg:
        job.log(msg)


def try_one(*values):
    v, opts, cfg = split_values(values)
    save_settings(v)
    folder = Path(str(v["folder"]).strip().strip('"'))
    if not folder.is_dir():
        job.log(f"❌ Not a folder: {folder}")
        return
    images = tc.list_images(folder)
    name = str(v["try_image"] or "").strip()
    if name:
        image = next((im for im in images if im.name.lower() == name.lower()), None)
        if image is None:
            job.log(f"❌ {name} isn't an image in this folder.")
            return
    else:
        image = next((im for im in images if not tc.caption_path(im).exists()), images[0] if images else None)
    if image is None:
        job.log("❌ No images in this folder.")
        return
    session = job.get_session(cfg)
    if not cfg.keep_running:
        job.log("ℹ️ Tip: tick 'Keep the model loaded' to try several settings without reloading it each time.")

    def run():
        session.prepare(job.stop_event)
        index = tc.EmbeddedTagIndex(folder) if tc.EmbeddedTagIndex is not None else None
        res = session.caption_one(image, opts, index, write=False)
        job.log(f"🔍 {image.name} ({res['seconds']:.1f} s, hints: {res['source'] or 'none'}) — not saved"
                + (f"\n   {res['note']}" if res["note"] else ""))
        job.show(image, res["text"], res["query"])
    msg = job.start(f"Trying {image.name}", run)
    if msg:
        job.log(msg)


def download_now(*values):
    v, opts, cfg = split_values(values)
    save_settings(v)
    session = job.get_session(cfg)

    def run():
        if cfg.mode == "external":
            session.log("ℹ️ Nothing to download: the model runs in the server at the URL you gave.")
            return
        if cfg.mode == "auto" and tc.find_llama_server() is None:
            session.log(f"⬇ Downloading llama.cpp {tc.LLAMA_CPP_TAG}…")
            tc.ensure_llama_server(session._progress)
        for f in (cfg.model, cfg.mmproj):
            session.log(f"📦 {tc.model_file(cfg.repo, f, session._progress)}")
    msg = job.start("Download", run)
    if msg:
        job.log(msg)


def stop():
    if job.busy():
        job.stop_event.set()
        job.log("⏹ Stopping after the current image…")


def unload():
    if job.busy():
        job.log("⚠️ Still working — stop first.")
    elif job.session is not None and job.session.server is not None:
        job.session.close(force=True)
    else:
        job.log("No model loaded.")


def list_files(repo):
    try:
        files = tc.list_repo_files(repo)
    except Exception as e:
        job.log(f"❌ {e}")
        return gr.update(), gr.update()
    models = [(f"{n}  ({sz / 2**30:.2f} GB)", n) for n, sz in files if "mmproj" not in n.lower()]
    mmprojs = [(f"{n}  ({sz / 2**30:.2f} GB)", n) for n, sz in files if "mmproj" in n.lower()]
    job.log(f"📃 {repo}: {len(models)} model file(s), {len(mmprojs)} vision projector(s)")
    return gr.update(choices=models), gr.update(choices=mmprojs)


def poll(seen):
    with job.lock:
        if seen == job.version:
            return [gr.skip()] * 5 + [seen]
        text = "\n".join(job.lines)
        status = f"**{job.status}** {job.progress}"
        return [text, status, job.preview, job.caption, job.query, job.version]


CSS = (".tg-status p { margin: 0 } #tg-log textarea { font-family: ui-monospace, monospace; font-size: 12px }"
       " .tg-checks .block { min-width: min(200px, 100%) !important }")   # wrap on narrow screens


def build_ui():
    s = load_settings()
    with gr.Blocks(title="ToriiGate Captioner") as ui:
        gr.Markdown("## ToriiGate 0.5 Captioner\nNatural-language captions for a folder of images, written next to each "
                    f"image as `<name>{tc.NL_CAPTION_SUFFIX}`. Tag captions (`.txt`) are never changed.")
        with gr.Row():
            with gr.Column(scale=3):
                folder = gr.Textbox(label="Image folder", value=s["folder"], placeholder="C:/Images/MyDataset")
                fmt = gr.Dropdown(label="Caption format", value=s["format"],
                                  choices=[(f"{k} — {tc.FORMAT_NOTES.get(k, '')}", k) for k in tc.FORMATS])
                names = gr.Radio(label="Character names", value=s["names"],
                                 choices=[(v, k) for k, v in tc.NAMES_MODES.items()])
                characters = gr.Textbox(label="Names to use for every image (optional, comma-separated)", value=s["characters"],
                                        placeholder="e.g. your trigger word, or 'hatsune miku'. Empty: Grabber character tags")
                with gr.Row(elem_classes="tg-checks"):
                    use_tags = gr.Checkbox(label="Give the image's tags as hints", value=s["use_tags"])
                    include_artists = gr.Checkbox(label="Include artist tags", value=s["include_artists"])
                    overwrite = gr.Checkbox(label=f"Overwrite existing {tc.NL_CAPTION_SUFFIX}", value=s["overwrite"])
                tag_style = gr.Radio(label="Tag style in the prompt", value=s["tag_style"],
                                     choices=[("plain: long hair, smile", "plain"), ("booru: long_hair smile", "booru")])
                with gr.Accordion("Generation", open=False):
                    with gr.Row():
                        temperature = gr.Slider(0.0, 1.5, value=s["temperature"], step=0.05, label="Temperature (model default 0.5)")
                        max_tokens = gr.Number(label="Max new tokens", value=s["max_tokens"], precision=0, minimum=16)
                        seed = gr.Number(label="Seed (-1 = random)", value=s["seed"], precision=0)
                with gr.Accordion("Model and server", open=False):
                    mode = gr.Radio(label="Run the model with", value=s["mode"], choices=[
                        ("llama.cpp, downloaded automatically (Windows)", "auto"), ("my llama-server", "local"),
                        ("a server that's already running", "external")])
                    with gr.Row():
                        llama_server = gr.Textbox(label="llama-server path (for 'my llama-server')", value=s["llama_server"])
                        url = gr.Textbox(label="Server URL (for 'already running')", value=s["url"], placeholder="http://127.0.0.1:8080")
                    with gr.Row(equal_height=True):
                        repo = gr.Textbox(label="Hugging Face repo", value=s["repo"], scale=3)
                        list_btn = gr.Button("🔄 List files", scale=1)
                    with gr.Row():
                        model = gr.Dropdown(label="Model file (or a local .gguf path)", value=s["model"], choices=[s["model"]],
                                            allow_custom_value=True)
                        mmproj = gr.Dropdown(label="Vision projector (mmproj)", value=s["mmproj"], choices=[s["mmproj"]],
                                             allow_custom_value=True)
                    with gr.Row(equal_height=True):
                        gpu_layers = gr.Textbox(label="GPU layers", value=s["gpu_layers"])
                        ctx = gr.Number(label="Context size", value=s["ctx"], precision=0, minimum=2048)
                        keep_running = gr.Checkbox(label="Keep the model loaded between runs", value=s["keep_running"])
                        download_btn = gr.Button("⬇ Download now")
                try_image = gr.Textbox(label="Image to try (file name; empty = the first without a caption)", value=s["try_image"])
                with gr.Row():
                    start_btn = gr.Button("▶ Caption folder", variant="primary")
                    try_btn = gr.Button("🔍 Try one image")
                    stop_btn = gr.Button("⏹ Stop", variant="stop")
                    unload_btn = gr.Button("⏏ Unload model")
                status = gr.Markdown("**Idle.**", elem_classes="tg-status")
                log = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True, interactive=False, elem_id="tg-log")
            with gr.Column(scale=2):
                preview = gr.Image(label="Last image", type="pil", interactive=False, height=420)
                caption = gr.Textbox(label="Caption", lines=10, interactive=False)
                with gr.Accordion("Prompt sent to the model", open=False):
                    query = gr.Textbox(show_label=False, lines=12, interactive=False)
        inputs = [folder, try_image, fmt, names, characters, use_tags, tag_style, include_artists, overwrite,
                  temperature, max_tokens, seed, mode, llama_server, url, repo, model, mmproj, gpu_layers, ctx, keep_running]
        assert len(inputs) == len(UI_KEYS)
        seen = gr.State(-1)
        start_btn.click(start_captioning, inputs, None, api_name="start")
        try_btn.click(try_one, inputs, None, api_name="try_one")
        download_btn.click(download_now, inputs, None, api_name="download")
        stop_btn.click(stop, None, None, api_name="stop")
        unload_btn.click(unload, None, None, api_name="unload")
        list_btn.click(list_files, [repo], [model, mmproj], api_name="list_files")
        outputs = [log, status, preview, caption, query, seen]
        gr.Timer(1.0).tick(poll, [seen], outputs, show_progress="hidden", concurrency_limit=None, api_name="poll")
        ui.load(poll, [seen], outputs, show_progress="hidden")
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Web UI for the ToriiGate 0.5 captioner")
    ap.add_argument("--port", type=int, default=None, help="default: the first free port from 7860")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it to your network (no password!)")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = ap.parse_args()
    build_ui().launch(server_name=a.host, server_port=a.port, inbrowser=not a.no_browser, theme=gr.themes.Soft(), css=CSS)
