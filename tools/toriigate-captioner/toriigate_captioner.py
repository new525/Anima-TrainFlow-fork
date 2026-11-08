"""ToriiGate 0.5 captioner: natural-language captions for a folder of images.

The model (Minthy/ToriiGate-0.5, a Qwen3.5-4B fine-tune for anime art, MIT) runs in llama.cpp's llama-server from a
GGUF build. Prompts come from the model's own scripts/prompts.py (bundled unmodified as prompts.py), and requests
follow its reference script (scripts/caption_distributed.py): image scaled to at most 1 megapixel and sent as JPEG
quality 95, the image before the text, temperature 0.5, up to 2048 new tokens.

Booru tags and character names are given to the model as grounding, which its card recommends for correct names:
from tags embedded by Grabber (character:/artist:/copyright: namespaces tell which is which), otherwise from the
image's tag caption (.txt), and/or names you type in. Captions are written next to each image as
<name>.toriigate.txt; the tag captions (.txt) are never touched.

Command line:  python toriigate_captioner.py <folder> [options]      (python toriigate_captioner.py --help)
Web UI:        python captioner_ui.py
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# The portable Python's ._pth file keeps the script's folder off sys.path.
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import prompts as tg  # noqa: E402  (ToriiGate's official prompts, unmodified)

try:
    from embedded_tags import KAOMOJIS, NL_CAPTION_SUFFIX, EmbeddedTagIndex, namespaced_tags  # TrainFlow's reader
except ImportError:  # the captioner copied out of the TrainFlow folder: no embedded tags, .txt captions still work
    EmbeddedTagIndex = None
    KAOMOJIS = []
    NL_CAPTION_SUFFIX = ".toriigate.txt"

    def namespaced_tags(items):
        return [("general", t) for t in items]
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
try:
    import pillow_jxl  # noqa: F401
except ImportError:
    pass

VERSION = "1.0"
# --- llama.cpp: a pinned release (its flags are what this file uses), Windows CUDA 12.4 build (runs on Turing+)
LLAMA_CPP_TAG = "b12211"
LLAMA_CPP_ASSETS = (f"llama-{LLAMA_CPP_TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip")
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"
# --- model: GGUF build listed on the model card (llama.cpp needs the text model and the vision projector, mmproj)
MODEL_REPO = "SleepVeryHard/ToriiGate-0.5_GGUF"
MODEL_FILE = "ToriiGate-0.5_Q8_0.gguf"
MMPROJ_FILE = "mmproj_BF16.gguf"
HF_URL = "https://huggingface.co"
RUNTIME = HERE / "runtime"
MODELS = HERE / "models"
# --- request settings from the model's reference script
MAX_PIXELS = 1_000_000
JPEG_QUALITY = 95
TEMPERATURE = 0.5
MAX_TOKENS = 2048

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif", ".jxl", ".heic", ".heif", ".hif", ".tif", ".tiff"}
LOG_NAME = "toriigate_captions.log"
FORMATS = list(tg.prompts_b)
FORMAT_NOTES = {
    "short": "a few sentences covering the main things — the best fit for training captions",
    "long": "2 to 5 vivid paragraphs (long: with tags it can pass Anima's 512-token caption limit)",
    "chroma-style": "summary, numbered parts, Midjourney-style summary, commission request",
    "json": "JSON: each character, background, effects, texts, atmosphere",
    "min_structured_json": "short JSON with tag-like phrases",
    "min_structured_md": "markdown: thoughts about characters, key details, structured description (needs names)",
    "long_thoughts": "markdown in 6 parts, starting with reasoning about who is who (needs names)",
    "long_thoughts_v2": "reasoning, key details, long description, per character (needs names)",
    "json_comic": "JSON for comics: frame by frame",
    "md_comic": "markdown for comics: frame by frame (needs names)",
}
NAMES_MODES = {
    "known": "use the names given or found in Grabber tags; otherwise don't guess names",
    "recognize": "use the names given or found; otherwise let the model recognise characters",
    "none": "never name characters",
}
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_SCORE_TAG = re.compile(r"^score_\d+(?:_up)?$", re.I)


class CaptionError(RuntimeError):
    pass


class SetupError(RuntimeError):
    pass


@dataclass
class Options:
    format: str = "short"
    names: str = "known"          # see NAMES_MODES
    characters: str = ""          # comma-separated names used for every image (e.g. your trigger word)
    use_tags: bool = True         # give the image's booru tags as grounding
    tag_style: str = "plain"      # plain: "long hair, smile" / booru: "long_hair smile" (both supported by prompts.py)
    include_artists: bool = False
    overwrite: bool = False
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    seed: int = -1                # -1: random each time
    timeout: float = 600.0


@dataclass
class ServerConfig:
    mode: str = "auto"            # auto: download + run llama.cpp / local: run llama_server / external: use url
    llama_server: str = ""
    url: str = ""
    repo: str = MODEL_REPO
    model: str = MODEL_FILE       # a file in repo, or a path to a local .gguf
    mmproj: str = MMPROJ_FILE
    gpu_layers: str = "all"
    ctx: int = 8192
    keep_running: bool = False


# ======================================================================================================================
# Grounding and prompts
# ======================================================================================================================
def caption_path(image: Path) -> Path:
    return image.with_name(image.stem + NL_CAPTION_SUFFIX)


def list_images(folder: Path) -> List[Path]:
    return sorted((f for f in Path(folder).iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS and not f.name.startswith((".", "~"))),
                  key=lambda p: (p.name.lower(), p.name))


def _split_caption(text: str) -> List[str]:
    first = next((line for line in text.splitlines() if line.strip()), "")
    return [t.strip() for t in first.split(",") if t.strip()]


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def grounding(image: Path, index, opts: Options) -> Tuple[dict, str]:
    """The item dict prompts.make_user_query expects, and where its tags came from."""
    tags: List[str] = []
    chars: List[str] = []
    source = ""
    if index is not None:
        items, src = index.items_for(image)
        if items:
            for section, t in namespaced_tags(items):
                if section == "character":
                    chars.append(t)
                elif section == "artist":
                    if opts.include_artists:
                        tags.append(t)
                else:
                    tags.append(t)
            source = f"embedded tags ({src})"
    if not tags and not chars:
        txt = image.with_suffix(".txt")
        if txt.is_file():
            for t in _split_caption(txt.read_text(encoding="utf-8", errors="replace")):
                if t.startswith("@"):
                    if opts.include_artists:
                        tags.append(t[1:].strip())
                else:
                    tags.append(t)
            source = txt.name
    tags = [t for t in tags if not _SCORE_TAG.match(t)]
    given = [c.strip() for c in re.split(r"[,;\n]", opts.characters or "") if c.strip()]
    if given:
        chars = given
        source = (source + " + " if source else "") + "names given"

    def styled(t: str) -> str:
        t = re.sub(r"\s+", " ", t.strip())
        if t in KAOMOJIS or opts.tag_style != "booru":
            return t
        return t.replace(" ", "_")
    item = {"tags": _dedupe(styled(t) for t in tags), "characters": _dedupe(styled(c) for c in chars),
            # prompts.py reads these even when unused; its fallback is a string, so they must be real dicts
            "char_p_tags": {"chars": {}, "skins": {}}, "char_descr": {"chars": {}, "skins": {}}}
    return item, source


def name_settings(item: dict, opts: Options) -> Tuple[bool, bool, str]:
    """(use_names, add_characters, note) for prompts.make_user_query."""
    names_only = tg.prompts_names_only.get(opts.format, False)
    has_names = bool(item["characters"])
    if opts.names == "none" and not names_only:
        return False, False, ""
    if has_names:
        return True, True, ("" if opts.names != "none" else
                            f"'{opts.format}' only works with named characters, so the names are used anyway")
    if opts.names == "recognize" or names_only:
        return True, False, ("" if opts.names == "recognize" else
                             f"'{opts.format}' needs named characters; none known, so the model is asked to recognise them")
    return False, False, ""


def build_query(item: dict, opts: Options, key: str = "") -> Tuple[str, str]:
    """(user prompt, note). prompts.make_user_query shuffles the tags; seeding by the image's name keeps the prompt
    for an image the same from run to run."""
    if opts.format not in tg.prompts_b:
        raise CaptionError(f"unknown format '{opts.format}' (choose from: {', '.join(FORMATS)})")
    use_names, add_chars, note = name_settings(item, opts)
    state = random.getstate()
    random.seed(key)
    try:
        query = tg.make_user_query(copy.deepcopy(item), opts.format, use_names, opts.use_tags and bool(item["tags"]),
                                   add_chars, False, False, underscores_replace=(opts.tag_style != "booru"))
    finally:
        random.setstate(state)
    return query, note


def encode_image(path: Path) -> Tuple[str, Tuple[int, int]]:
    """Upright RGB (transparency on white), at most MAX_PIXELS, JPEG -> base64, as the reference script sends it."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            im = Image.new("RGB", rgba.size, (255, 255, 255))
            im.paste(rgba, mask=rgba.getchannel("A"))
        else:
            im = im.convert("RGB")
        w, h = im.size
        if w * h > MAX_PIXELS:
            scale = (MAX_PIXELS / (w * h)) ** 0.5
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("ascii"), im.size


def request_caption(client: httpx.Client, base_url: str, image_b64: str, query: str, opts: Options) -> dict:
    payload = {
        "model": "toriigate",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": tg.system_prompt}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                                         {"type": "text", "text": query}]},
        ],
        "max_tokens": int(opts.max_tokens), "temperature": float(opts.temperature), "stream": False,
    }
    if int(opts.seed) >= 0:
        payload["seed"] = int(opts.seed)
    try:
        r = client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=opts.timeout)
    except httpx.TimeoutException as e:
        raise CaptionError(f"no answer within {opts.timeout:.0f} s") from e
    if r.status_code != 200:
        raise CaptionError(f"server answered {r.status_code}: {r.text[:300]}")
    data = r.json()
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    if not text.strip() and (msg.get("reasoning_content") or "").strip():
        text = msg["reasoning_content"]   # a chat template that puts the whole answer in the thinking part
    text = _THINK.sub("", text).strip()
    if not text:
        raise CaptionError("the model returned an empty caption")
    return {"text": text, "finish_reason": choice.get("finish_reason"), "usage": data.get("usage") or {}}


def write_caption(image: Path, text: str) -> Path:
    dest = caption_path(image)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, dest)
    return dest


# ======================================================================================================================
# Downloads
# ======================================================================================================================
def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download(url: str, dest: Path, progress: Optional[Callable[[str, int, int], None]] = None,
             headers: Optional[dict] = None) -> Path:
    """Stream url to dest through dest.part, resuming a partial download."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    hdrs = dict(headers or {})
    if have:
        hdrs["Range"] = f"bytes={have}-"
    with httpx.stream("GET", url, headers=hdrs, follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as r:
        if r.status_code == 416:            # already complete
            os.replace(part, dest)
            return dest
        if r.status_code not in (200, 206):
            raise SetupError(f"download failed ({r.status_code}): {url}")
        if r.status_code == 200:
            have = 0                         # server ignored the range: start over
        total = have + int(r.headers.get("content-length") or 0)
        with open(part, "ab" if have else "wb") as f:
            done, last = have, 0.0
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress and time.time() - last > 0.5:
                    progress(dest.name, done, total)
                    last = time.time()
    if progress:
        progress(dest.name, done, total)
    os.replace(part, dest)
    return dest


def list_repo_files(repo: str = MODEL_REPO) -> List[Tuple[str, int]]:
    """(file name, size in bytes) of the .gguf files in a Hugging Face model repo."""
    r = httpx.get(f"{HF_URL}/api/models/{repo}", params={"blobs": "true"}, headers=_hf_headers(),
                  follow_redirects=True, timeout=30)
    if r.status_code != 200:
        raise SetupError(f"couldn't list {repo} on Hugging Face ({r.status_code})")
    return sorted((s["rfilename"], int(s.get("size") or 0)) for s in r.json().get("siblings", [])
                  if s["rfilename"].lower().endswith(".gguf"))


def model_file(repo: str, name: str, progress=None) -> Path:
    """A local .gguf path: name itself if it's an existing file, else downloaded from repo into models/."""
    p = Path(name).expanduser()
    if p.is_file():
        return p
    dest = MODELS / repo.replace("/", "__") / name
    if dest.is_file():
        return dest
    return download(f"{HF_URL}/{repo}/resolve/main/{name}", dest, progress, _hf_headers())


def find_llama_server(folder: Path = RUNTIME / "llama.cpp", platform: Optional[str] = None) -> Optional[Path]:
    if not folder.is_dir():
        return None
    exe = "llama-server.exe" if (platform or os.name) == "nt" else "llama-server"
    return next(iter(sorted(folder.rglob(exe))), None)


def ensure_llama_server(progress=None, platform: Optional[str] = None, url_template: str = LLAMA_CPP_URL,
                        folder: Path = RUNTIME / "llama.cpp") -> Path:
    found = find_llama_server(folder, platform)
    if found:
        return found
    if (platform or os.name) != "nt":
        raise SetupError("The automatic llama.cpp download is the Windows CUDA build. Elsewhere, install llama.cpp "
                         "and choose 'local' with the path to llama-server.")
    downloads = folder.parent / "downloads"
    for asset in LLAMA_CPP_ASSETS:
        zpath = download(url_template.format(tag=LLAMA_CPP_TAG, asset=asset), downloads / asset, progress)
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():   # no paths outside the folder
                target = (folder / info.filename).resolve()
                if folder.resolve() not in target.parents and target != folder.resolve():
                    raise SetupError(f"unexpected path in {asset}: {info.filename}")
            z.extractall(folder)
    found = find_llama_server(folder, platform)
    if not found:
        raise SetupError(f"llama-server wasn't in the llama.cpp {LLAMA_CPP_TAG} download")
    return found


# ======================================================================================================================
# llama-server
# ======================================================================================================================
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    """A llama-server process serving one model + vision projector on 127.0.0.1."""

    def __init__(self, exe, model: Path, mmproj: Path, gpu_layers: str = "all", ctx: int = 8192,
                 port: Optional[int] = None, log_path: Path = RUNTIME / "llama-server.log", extra: Sequence[str] = ()):
        self.cmd_prefix = list(exe) if isinstance(exe, (list, tuple)) else [str(exe)]
        self.model, self.mmproj = Path(model), Path(mmproj)
        self.gpu_layers, self.ctx, self.port = str(gpu_layers), int(ctx), port or free_port()
        self.log_path, self.extra = Path(log_path), list(extra)
        self.proc: Optional[subprocess.Popen] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> List[str]:
        return self.cmd_prefix + ["-m", str(self.model), "--mmproj", str(self.mmproj), "-ngl", self.gpu_layers,
                                  "-c", str(self.ctx), "--host", "127.0.0.1", "--port", str(self.port),
                                  "--reasoning", "off",     # the answer goes to content, not a thinking block
                                  "--no-webui", "--alias", "toriigate", "-np", "1"] + self.extra

    def log_tail(self, n: int = 15) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    def start(self, timeout: float = 900.0, stop_event: Optional[threading.Event] = None) -> str:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with open(self.log_path, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         creationflags=flags)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SetupError(f"llama-server exited (code {self.proc.returncode}) while loading:\n{self.log_tail()}")
            if stop_event is not None and stop_event.is_set():
                self.stop()
                raise SetupError("stopped while the model was loading")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return self.url
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise SetupError(f"llama-server wasn't ready after {timeout:.0f} s:\n{self.log_tail()}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        self.proc = None


# ======================================================================================================================
# A captioning session: server set-up + folder loop (used by the CLI and the web UI)
# ======================================================================================================================
@dataclass
class Event:
    kind: str                     # "log" | "progress" | "caption" | "done"
    message: str = ""
    done: int = 0
    total: int = 0
    image: str = ""
    caption: str = ""
    query: str = ""


class Session:
    def __init__(self, server: ServerConfig, emit: Callable[[Event], None] = lambda e: None):
        self.cfg = server
        self.emit = emit
        self.server: Optional[LlamaServer] = None
        self.base_url: Optional[str] = None
        self.client = httpx.Client()

    def log(self, msg: str) -> None:
        self.emit(Event("log", msg))

    def _progress(self, name: str, done: int, total: int) -> None:
        pct = f"{100 * done / total:.0f}%" if total else ""
        self.emit(Event("progress", f"⬇ {name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MB {pct}", done, total))

    def prepare(self, stop_event: Optional[threading.Event] = None) -> str:
        """Download what's missing, start llama-server (or check the external one); returns its URL."""
        if self.base_url and (self.server is None or self.server.alive()):
            return self.base_url
        cfg = self.cfg
        if cfg.mode == "external":
            url = cfg.url.strip().rstrip("/")
            if not url:
                raise SetupError("no server URL given")
            try:
                ok = httpx.get(f"{url}/health", timeout=10).status_code == 200
            except httpx.HTTPError as e:
                raise SetupError(f"can't reach {url}: {e}") from e
            if not ok:
                raise SetupError(f"{url} isn't ready (its /health didn't answer 200)")
            self.base_url = url
            self.log(f"🔌 Using the server at {url}")
            return url
        if cfg.mode == "local":
            exe = Path(cfg.llama_server.strip().strip('"')).expanduser()
            if not exe.is_file():
                raise SetupError(f"llama-server not found: {exe}")
        else:
            exe = find_llama_server()
            if exe is None:
                self.log(f"⬇ Downloading llama.cpp {LLAMA_CPP_TAG} (Windows, CUDA 12.4)…")
                exe = ensure_llama_server(self._progress)
        self.log("📦 Model files…")
        model = model_file(cfg.repo, cfg.model, self._progress)
        mmproj = model_file(cfg.repo, cfg.mmproj, self._progress)
        self.server = LlamaServer(exe, model, mmproj, cfg.gpu_layers, cfg.ctx)
        self.log(f"🚀 Loading {model.name} + {mmproj.name} in llama-server (port {self.server.port})…")
        t0 = time.time()
        self.base_url = self.server.start(stop_event=stop_event)
        self.log(f"✅ Model ready in {time.time() - t0:.0f} s")
        return self.base_url

    def close(self, force: bool = False) -> None:
        if self.server is not None and (force or not self.cfg.keep_running):
            self.server.stop()
            self.server, self.base_url = None, None
            self.log("⏏ Model unloaded")

    def caption_one(self, image: Path, opts: Options, index=None, write: bool = True) -> dict:
        item, source = grounding(image, index, opts)
        query, note = build_query(item, opts, image.name)
        b64, size = encode_image(image)
        t0 = time.time()
        res = request_caption(self.client, self.base_url, b64, query, opts)
        res.update(seconds=time.time() - t0, query=query, note=note, source=source, item=item, size=size)
        if write:
            res["path"] = write_caption(image, res["text"])
        return res

    def caption_folder(self, folder: Path, opts: Options, stop_event: Optional[threading.Event] = None,
                       limit: Optional[int] = None) -> dict:
        folder = Path(folder)
        images = list_images(folder)
        todo = [im for im in images if opts.overwrite or not caption_path(im).exists()]
        skipped = len(images) - len(todo)
        if limit and len(todo) > limit:
            todo = todo[:limit]
        summary = {"images": len(images), "captioned": 0, "skipped": skipped, "failed": 0, "stopped": False}
        if not images:
            self.log("❌ No images in this folder.")
            return summary
        self.log(f"🖼 {len(images)} image(s): {len(todo)} to caption"
                 + (f" (limit {limit})" if limit and len(todo) == limit else "")
                 + (f", {skipped} already have {NL_CAPTION_SUFFIX} (overwrite is off)" if skipped else ""))
        if not todo:
            return summary
        stop_event = stop_event or threading.Event()
        self.prepare(stop_event)
        index = EmbeddedTagIndex(folder) if EmbeddedTagIndex is not None else None
        log_file = folder / LOG_NAME
        t_start = time.time()
        for n, image in enumerate(todo, start=1):
            if stop_event.is_set():
                summary["stopped"] = True
                self.log("⏹ Stopped.")
                break
            try:
                res = self.caption_one(image, opts, index)
            except (CaptionError, OSError, ValueError, KeyError, IndexError, TypeError) as e:
                summary["failed"] += 1
                self.log(f"❌ {image.name}: {e}")
                self._append_log(log_file, f"{image.name}  FAILED  {e}")
                if self.server is not None and not self.server.alive():
                    self.log(f"❌ llama-server stopped:\n{self.server.log_tail()}")
                    break
                continue
            except httpx.HTTPError as e:   # server gone
                summary["failed"] += 1
                self.log(f"❌ {image.name}: lost the server ({e}).")
                break
            summary["captioned"] += 1
            tokens = res["usage"].get("completion_tokens")
            warn = " ⚠️ hit the token limit (caption may be cut off)" if res["finish_reason"] == "length" else ""
            self._append_log(log_file, f"{image.name}  {opts.format}  {res['seconds']:.1f}s"
                             + (f"  {tokens} tokens" if tokens else "") + f"  hints: {res['source'] or 'none'}"
                             + (f"  ({res['note']})" if res["note"] else "") + warn)
            elapsed = time.time() - t_start
            eta = elapsed / n * (len(todo) - n)
            self.emit(Event("caption", f"✏️ {image.name} ({res['seconds']:.1f} s){warn}", n, len(todo),
                            str(image), res["text"], res["query"]))
            self.emit(Event("progress", f"{n}/{len(todo)} · about {eta / 60:.0f} min left" if n < len(todo) else
                            f"{n}/{len(todo)}", n, len(todo)))
        self.log(f"🏁 Captioned {summary['captioned']}, skipped {summary['skipped']}, failed {summary['failed']}."
                 + f" Log: {log_file.name}")
        return summary

    @staticmethod
    def _append_log(path: Path, line: str) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + line + "\n")
        except OSError:
            pass


# ======================================================================================================================
# Command line
# ======================================================================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="toriigate_captioner",
        description="Natural-language captions with ToriiGate 0.5 (writes <image>.toriigate.txt next to each image).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="formats:\n" + "\n".join(f"  {k:20s} {v}" for k, v in FORMAT_NOTES.items()))
    ap.add_argument("folder", nargs="?", help="folder of images")
    g = ap.add_argument_group("captions")
    g.add_argument("--format", default="short", choices=FORMATS)
    g.add_argument("--names", default="known", choices=list(NAMES_MODES), help="; ".join(f"{k}: {v}" for k, v in NAMES_MODES.items()))
    g.add_argument("--characters", default="", help="comma-separated names to use for every image (e.g. your trigger word)")
    g.add_argument("--no-tags", action="store_true", help="don't give the image's booru tags as hints")
    g.add_argument("--tag-style", default="plain", choices=["plain", "booru"])
    g.add_argument("--include-artists", action="store_true", help="also pass artist tags as hints")
    g.add_argument("--overwrite", action="store_true", help="caption images that already have a .toriigate.txt")
    g.add_argument("--limit", type=int, default=0, help="caption at most this many images")
    g.add_argument("--temperature", type=float, default=TEMPERATURE)
    g.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    g.add_argument("--seed", type=int, default=-1)
    g.add_argument("--dry-run", action="store_true", help="print the prompt for each image; load nothing")
    s = ap.add_argument_group("model and server")
    s.add_argument("--server-url", default="", help="use an already running llama-server (or compatible) at this URL")
    s.add_argument("--llama-server", default="", help="path to llama-server to run (default: download llama.cpp on Windows)")
    s.add_argument("--repo", default=MODEL_REPO)
    s.add_argument("--model", default=MODEL_FILE, help="GGUF file in --repo, or a local path")
    s.add_argument("--mmproj", default=MMPROJ_FILE, help="vision projector file in --repo, or a local path")
    s.add_argument("--gpu-layers", default="all")
    s.add_argument("--ctx", type=int, default=8192)
    s.add_argument("--download-only", action="store_true", help="download llama.cpp and the model, then exit")
    s.add_argument("--list-files", action="store_true", help="list the .gguf files in --repo and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):   # a Windows pipe/redirect defaults to the ANSI code page: no emoji
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = parse_args(argv)
    opts = Options(format=a.format, names=a.names, characters=a.characters, use_tags=not a.no_tags,
                   tag_style=a.tag_style, include_artists=a.include_artists, overwrite=a.overwrite,
                   temperature=a.temperature, max_tokens=a.max_tokens, seed=a.seed)
    cfg = ServerConfig(mode="external" if a.server_url else ("local" if a.llama_server else "auto"),
                       llama_server=a.llama_server, url=a.server_url, repo=a.repo, model=a.model, mmproj=a.mmproj,
                       gpu_layers=a.gpu_layers, ctx=a.ctx)

    def emit(e: Event):
        if e.kind == "progress" and e.message.startswith("⬇"):
            print("\r" + e.message, end="", flush=True)
        elif e.kind == "caption":
            print(f"[{e.done}/{e.total}] {Path(e.image).name}: {e.caption[:120].replace(chr(10), ' ')}…" if len(e.caption) > 120
                  else f"[{e.done}/{e.total}] {Path(e.image).name}: {e.caption}", flush=True)
        elif e.kind == "log":
            print(e.message, flush=True)

    if a.list_files:
        for name, size in list_repo_files(a.repo):
            print(f"{size / 2**30:6.2f} GB  {name}")
        return 0
    session = Session(cfg, emit)
    try:
        if a.download_only:
            if cfg.mode == "auto":
                ensure_llama_server(session._progress)
            for f in (a.model, a.mmproj):
                print("\n" + str(model_file(a.repo, f, session._progress)))
            return 0
        if not a.folder:
            print("give a folder of images (see --help)")
            return 2
        folder = Path(a.folder)
        if not folder.is_dir():
            print(f"not a folder: {folder}")
            return 2
        if a.dry_run:
            index = EmbeddedTagIndex(folder) if EmbeddedTagIndex is not None else None
            for image in list_images(folder)[: a.limit or None]:
                item, source = grounding(image, index, opts)
                query, note = build_query(item, opts, image.name)
                print(f"===== {image.name}   (hints: {source or 'none'}){'   ' + note if note else ''}\n{query}")
            return 0
        stop = threading.Event()
        try:
            summary = session.caption_folder(folder, opts, stop, a.limit or None)
        except KeyboardInterrupt:
            stop.set()
            print("\ninterrupted")
            return 130
        return 0 if summary["failed"] == 0 else 1
    except SetupError as e:
        print(f"\n❌ {e}")
        return 3
    finally:
        session.close(force=True)


if __name__ == "__main__":
    sys.exit(main())
