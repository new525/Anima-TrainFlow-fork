import os
import re
import json
import platform
import subprocess
import toml
import glob
import math
from pathlib import Path
import sys
import gradio as gr
import psutil
from PIL import Image, ImageOps
import torch
import numpy as np
import onnxruntime as rt
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import shutil
from typing import Generator, List, Tuple

CSS = """
.gradio-container {
    position: relative !important;
}

#main-header {
    margin-bottom: -10px !important;
}

.attribution {
    position: absolute;
    right: 25px;
    top: 5px;
    z-index: 99999;
    pointer-events: none;
}

.author-text {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(187, 154, 247, 0.6);
    letter-spacing: 0.5px;
}

#log-container textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 14px !important;
    line-height: 1.3 !important;
    white-space: pre-wrap !important;
    overflow-x: hidden !important;
    height: 385px !important;
    border: 1px solid #2f334d !important;
}

*::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
*::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1) !important;
}
*::-webkit-scrollbar-thumb {
    background: #7aa2f7 !important;
    border-radius: 10px !important;
}
*::-webkit-scrollbar-thumb:hover {
    background: #bb9af7 !important;
}

footer {
    display: none !important;
}

"""

JS_SCROLL = """
function() {
    setTimeout(() => {
        const el = document.querySelector('#log-container textarea');
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    }, 50);
}
"""


# Image types sd-scripts trains on (library/train_util.py IMAGE_EXTENSIONS); every tool uses the same list.
VALID_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

LOG_BLACKLIST = [
    "triton not found",
    "flop counting will not work",
    "Lib\\site-packages\\torch\\utils\\flop_counter.py"
]


LOG_BOX__MAX_LINES = 16
POLL_INTERVAL = 1.0  # seconds between UI refreshes of the log and preview gallery
ANIMA_SWAPPABLE_BLOCKS = 26  # Anima 2B has 28 DiT blocks; sd-scripts allows swapping at most num_blocks - 2
GALLERY_HEIGHT = 440
MAX_LOG_LINES = 500


ROOT = Path(__file__).resolve().parent
PORTABLE_PYTHON = ROOT / "python_embeded" / "python.exe"

TRAIN_BASE = ROOT / "training"
OUTPUT_BASE = TRAIN_BASE / "output"
SETTINGS_FILE = TRAIN_BASE / "settings.json"

TRAIN_DIR = TRAIN_BASE / "sd-scripts" 
TRAIN_SCRIPT = TRAIN_DIR / "anima_train_network.py"

for d in [TRAIN_BASE, OUTPUT_BASE]:
    d.mkdir(parents=True, exist_ok=True)


# ==========================================
# SERVER-SIDE CONSOLE
# The log lives here, not in the browser. Every browser tab just displays it,
# so a page refresh (or a second tab) shows the same log instead of losing it.
# ==========================================
TQDM_RE = re.compile(r"^(?P<desc>.*?)\s*\d+%\|")


class Console:
    def __init__(self, max_lines: int = MAX_LOG_LINES) -> None:
        self._lines: List[str] = []
        self._lock = threading.Lock()
        self._max_lines = max_lines
        self.version = 0

    def _trim(self) -> None:
        if len(self._lines) > self._max_lines:
            del self._lines[:-self._max_lines]

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self.version += 1

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            self._trim()
            self.version += 1

    def extend(self, lines: List[str]) -> None:
        with self._lock:
            self._lines.extend(lines)
            self._trim()
            self.version += 1

    def replace_last_if(self, line: str, marker: str) -> None:
        """Overwrite the last line if it contains `marker`, otherwise append (used for in-place progress)."""
        with self._lock:
            if self._lines and marker in self._lines[-1]:
                self._lines[-1] = line
            else:
                self._lines.append(line)
                self._trim()
            self.version += 1

    def add_process_line(self, line: str) -> None:
        """Append a subprocess line, collapsing successive refreshes of the same tqdm bar into one line."""
        m = TQDM_RE.match(line)
        with self._lock:
            if m and self._lines:
                prev = TQDM_RE.match(self._lines[-1])
                if prev and prev.group("desc") == m.group("desc"):
                    self._lines[-1] = line
                    self.version += 1
                    return
            self._lines.append(line)
            self._trim()
            self.version += 1

    def text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def snapshot(self) -> Tuple[int, str]:
        with self._lock:
            return self.version, "\n".join(self._lines)


console = Console()


# ==========================================
# TRAINING MANAGER
# The trainer runs as an independent process: its own process group (no console
# window on Windows), output written straight to <run>/logs/train_*.log. The app only
# tails that file. So the trainer survives page refreshes AND closing or restarting the
# app; <run>/trainer.json lets a restarted app find the running trainer and reattach.
#
# Pause  = suspend the trainer's process tree in place (instant; VRAM stays allocated).
# Halt   = ask the trainer to save a LoRA + resumable state after the current step, then exit.
# Resume = relaunch a halted (or crashed) run from its newest saved state.
# ==========================================
STEP_RE = re.compile(r"^steps:.*?\|\s*(\d+)/(\d+)")
LOSS_RE = re.compile(r"avr_loss=([-+0-9.eE]+|nan|inf)")
HALTED_RE = re.compile(r"HALTED at step (\d+)")
TRAINER_MARKER = "trainer.json"

if platform.system() == "Windows":
    # Own process group (Ctrl+C in the app console doesn't reach it) and its own hidden console
    # (closing the app console doesn't kill it). accelerate's child inherits that hidden console
    # and the log-file handles.
    DETACH_KWARGS = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                                      | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
else:
    DETACH_KWARGS = {"start_new_session": True}


def iter_log_lines(chunks):
    """Split raw trainer output into lines on \\n and \\r (tqdm redraws with \\r)."""
    buf = b""
    for chunk in chunks:
        buf += chunk
        buf = buf.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        *lines, buf = buf.split(b"\n")
        for raw in lines:
            yield raw.decode("utf-8", errors="ignore").strip()
    if buf:
        yield buf.decode("utf-8", errors="ignore").strip()


def parse_losses(log_paths: List[Path]) -> List[Tuple[int, float]]:
    """(step, moving-average loss) pairs from trainer logs, one per step (last value wins)."""
    by_step: dict = {}
    for lp in log_paths:
        try:
            with open(lp, "rb") as f:
                for line in iter_log_lines(iter(lambda: f.read(1 << 20), b"")):
                    m, lm = STEP_RE.match(line), LOSS_RE.search(line)
                    if m and lm:
                        try:
                            by_step[int(m.group(1))] = float(lm.group(1))
                        except ValueError:
                            pass
        except OSError:
            continue
    return sorted(by_step.items())


class TrainingManager:
    def __init__(self, console: Console) -> None:
        self.console = console
        self._lock = threading.Lock()
        self._popen: subprocess.Popen | None = None  # set when this app instance launched the trainer
        self._ps: psutil.Process | None = None        # set for launched and reattached trainers
        self._stop_requested = False
        self._halt_requested = False
        self._saw_halted = False
        self.state = "idle"  # idle | running | paused | halting | halted | stopping | stopped | finished | failed
        self.run_dir: Path | None = None
        self.sample_dir: Path | None = None
        self.gallery_since = 0.0
        self.log_path: Path | None = None
        self.halt_file: Path | None = None
        self.output_name = ""
        self.step: Tuple[int, int] | None = None
        self.losses: List[Tuple[int, float]] = []
        self.loss_version = 0

    # ---------- process helpers ----------
    def _alive_locked(self) -> bool:
        if self._popen is not None:
            return self._popen.poll() is None
        if self._ps is not None:
            try:
                return self._ps.is_running() and self._ps.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False
        return False

    def _process_tree(self) -> List[psutil.Process]:
        if self._ps is None or not self._alive_locked():
            return []
        try:
            return [self._ps] + self._ps.children(recursive=True)
        except psutil.NoSuchProcess:
            return []

    def _write_marker_locked(self) -> None:
        if self.run_dir is None or self._ps is None:
            return
        data = {"pid": self._ps.pid, "create_time": self._ps.create_time(), "log_path": str(self.log_path),
                "halt_file": str(self.halt_file), "gallery_since": self.gallery_since, "output_name": self.output_name,
                "paused": self.state == "paused", "halt_requested": self._halt_requested}
        try:
            tmp = self.run_dir / (TRAINER_MARKER + ".part")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self.run_dir / TRAINER_MARKER)
        except OSError:
            pass

    def is_running(self) -> bool:
        with self._lock:
            return self._alive_locked()

    def gallery_source(self) -> Tuple[Path | None, float]:
        with self._lock:
            return self.sample_dir, self.gallery_since

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "step": self.step, "run_dir": self.run_dir, "output_name": self.output_name}

    def loss_snapshot(self) -> Tuple[int, List[Tuple[int, float]]]:
        with self._lock:
            return self.loss_version, list(self.losses)

    # ---------- launch / reattach ----------
    def _begin_locked(self, run_dir: Path, log_path: Path, halt_file: Path, gallery_since: float, output_name: str) -> None:
        self._stop_requested = False
        self._halt_requested = False
        self._saw_halted = False
        self.run_dir = run_dir
        self.sample_dir = run_dir / "sample"  # sd-scripts writes samples to <output_dir>/sample
        self.gallery_since = gallery_since
        self.log_path = log_path
        self.halt_file = halt_file
        self.output_name = output_name
        self.step = None
        # loss history of this run so far (earlier logs, e.g. before a halt) so the chart continues
        self.losses = parse_losses(sorted((run_dir / "logs").glob("*.log")))
        self.loss_version += 1

    def start(self, cmd: List[str], cwd: Path, env: dict, run_dir: Path, log_path: Path, halt_file: Path,
              gallery_since: float, output_name: str = "") -> bool:
        with self._lock:
            if self._alive_locked():
                return False
            try:
                halt_file.unlink()  # stale request from an earlier run must not halt this one at step 1
            except FileNotFoundError:
                pass
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = dict(env, PYTHONUNBUFFERED="1")  # the trainer writes to a file now: flush every line
            with open(log_path, "ab") as log_handle:
                popen = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         cwd=str(cwd), env=env, **DETACH_KWARGS)
            # Only record the run once the process actually launched.
            self._begin_locked(run_dir, log_path, halt_file, gallery_since, output_name)
            self._popen = popen
            self._ps = psutil.Process(popen.pid)
            self.state = "running"
            self._write_marker_locked()
        threading.Thread(target=self._tail, args=(log_path, 0), daemon=True, name="train-log-tail").start()
        return True

    def reattach(self, output_base: Path) -> bool:
        """Find a trainer that outlived a previous app instance and follow it again."""
        for marker in sorted(output_base.glob(f"*/*/{TRAINER_MARKER}"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
                ps = psutil.Process(int(data["pid"]))
                same = abs(ps.create_time() - float(data["create_time"])) < 1.0  # guard against PID reuse
                alive = same and ps.is_running() and ps.status() != psutil.STATUS_ZOMBIE
            except (OSError, ValueError, KeyError, psutil.Error):
                alive = False
            if not alive:
                try: marker.unlink()
                except OSError: pass
                continue
            run_dir = marker.parent
            log_path = Path(data["log_path"])
            with self._lock:
                if self._alive_locked():
                    return False
                self._begin_locked(run_dir, log_path, Path(data["halt_file"]), float(data.get("gallery_since", 0.0)),
                                   data.get("output_name", ""))
                self._popen = None
                self._ps = ps
                self._halt_requested = bool(data.get("halt_requested"))
                self.state = "paused" if data.get("paused") else ("halting" if self._halt_requested else "running")
            self.console.clear()
            self.console.append(f"🔌 Reattached to a training run that kept going while the app was closed: "
                                f"{run_dir.parent.name}/{run_dir.name}")
            # show recent history, then follow the file from its current end
            try:
                size = log_path.stat().st_size
                with open(log_path, "rb") as f:
                    f.seek(max(0, size - 256 * 1024))
                    recent = f.read(size - f.tell())
                for line in iter_log_lines([recent]):
                    self._handle_line(line)
            except OSError:
                size = 0
            threading.Thread(target=self._tail, args=(log_path, size), daemon=True, name="train-log-tail").start()
            return True
        return False

    # ---------- output following ----------
    def _handle_line(self, line: str) -> None:
        if not line:
            return
        m = STEP_RE.match(line)
        if m:
            step = (int(m.group(1)), int(m.group(2)))
            lm = LOSS_RE.search(line)
            with self._lock:
                self.step = step
                if lm:
                    try:
                        loss = float(lm.group(1))
                        last = self.losses[-1][0] if self.losses else -1
                        if step[0] == last:      # tqdm redraws a step several times; keep the latest value
                            self.losses[-1] = (step[0], loss)
                            self.loss_version += 1
                        elif step[0] > last:     # older steps (replayed after reattach) are already in the history
                            self.losses.append((step[0], loss))
                            self.loss_version += 1
                    except ValueError:
                        pass
        if HALTED_RE.search(line):
            with self._lock:
                self._saw_halted = True
        if any(skip_word in line for skip_word in LOG_BLACKLIST):
            return
        self.console.add_process_line(line)

    def _tail(self, log_path: Path, offset: int) -> None:
        def chunks():
            try:
                f = open(log_path, "rb")
            except OSError as e:
                self.console.append(f"⚠️ Could not open trainer log {log_path}: {e}")
                return
            with f:
                f.seek(offset)
                while True:
                    data = f.read(65536)
                    if data:
                        yield data
                        continue
                    with self._lock:
                        alive = self._alive_locked()
                    if not alive:
                        rest = f.read()  # output written just before exit
                        if rest:
                            yield rest
                        return
                    time.sleep(0.2)
        try:
            for line in iter_log_lines(chunks()):
                self._handle_line(line)
        except Exception as e:
            self.console.append(f"⚠️ Log reader error: {e}")
        finally:
            self._finish(log_path)

    def _finish(self, log_path: Path) -> None:
        rc = None
        with self._lock:
            popen, ps = self._popen, self._ps
        try:
            if popen is not None:
                rc = popen.wait()
            elif ps is not None:
                rc = ps.wait(timeout=5)  # exit code when the OS provides it for a non-child; else None
        except Exception:
            rc = None
        with self._lock:
            final_lora = None
            if self.run_dir is not None and self.output_name:
                final_lora = self.run_dir / f"{self.output_name}.safetensors"
            step_txt = f"{self.step[0]}/{self.step[1]}" if self.step else "?"
            if self._stop_requested:
                self.state, msg = "stopped", "🛑 Training stopped by user."
            elif self._saw_halted and rc in (0, None):
                self.state, msg = "halted", f"💾 Halted at step {step_txt}. LoRA and resume state saved — press ↩ Resume Run to continue."
            elif rc == 0 or (rc is None and final_lora is not None and final_lora.exists()):
                self.state, msg = "finished", "✅ Training finished."
            elif rc is None:
                self.state, msg = "failed", f"❌ Trainer ended without a final LoRA (exit code unavailable). Full log: {log_path}"
            else:
                self.state, msg = "failed", f"❌ Trainer exited with code {rc}. Full log: {log_path}"
            if self.run_dir is not None:
                try: (self.run_dir / TRAINER_MARKER).unlink()
                except OSError: pass
            self._popen = None
            self._ps = None
        self.console.append(msg)

    # ---------- controls ----------
    def pause(self) -> str:
        with self._lock:
            if self.state != "running":
                return "ℹ️ Nothing to pause." if self.state not in ("paused", "halting") else "ℹ️ Already paused or halting."
            tree = self._process_tree()
            if not tree:
                return "ℹ️ Not running."
            for p in tree:  # trainer, accelerate launcher and dataloader workers
                try: p.suspend()
                except psutil.NoSuchProcess: pass
            self.state = "paused"
            self._write_marker_locked()
        return "⏸ Paused. GPU work is frozen but VRAM stays allocated. Press ▶ Continue to carry on."

    def _unpause_locked(self) -> None:
        for p in self._process_tree():
            try: p.resume()
            except psutil.NoSuchProcess: pass

    def unpause(self) -> str:
        with self._lock:
            if self.state != "paused":
                return "ℹ️ Not paused."
            self._unpause_locked()
            self.state = "running"
            self._write_marker_locked()
        return "▶ Continuing."

    def halt(self) -> str:
        with self._lock:
            if self.state == "halting":
                return "ℹ️ Halt already requested."
            if self.state not in ("running", "paused") or not self._process_tree():
                return "ℹ️ Not running."
            if self.state == "paused":
                self._unpause_locked()
            try:
                self.halt_file.parent.mkdir(parents=True, exist_ok=True)
                self.halt_file.write_text("halt", encoding="utf-8")
            except OSError as e:
                return f"⚠️ Could not request halt: {e}"
            self._halt_requested = True
            self.state = "halting"
            self._write_marker_locked()
        return "💾 Halt requested: finishing the current step, then saving the LoRA and resume state..."

    def stop(self) -> str:
        with self._lock:
            if not self._alive_locked() or self._ps is None:
                return "ℹ️ Not running."
            if self.state == "paused":
                self._unpause_locked()  # a suspended process can't act on terminate
            self._stop_requested = True
            self.state = "stopping"
            parent = self._ps
        try:
            children = parent.children(recursive=True)
            for child in children:
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            _, alive = psutil.wait_procs(children, timeout=3)
            for survivor in alive:
                try: survivor.kill()
                except psutil.NoSuchProcess: pass
            parent.terminate()
            try:
                parent.wait(timeout=3)
            except psutil.TimeoutExpired:
                parent.kill()
            return "🛑 Stopping training... (Clearing VRAM)"
        except psutil.NoSuchProcess:
            return "ℹ️ Process already finished."
        except Exception as e:
            return f"⚠️ Error during stop: {str(e)}"


training_manager = TrainingManager(console)


# ==========================================
# BACKGROUND TOOL JOBS
# Bucketing, tagging and dataset inspection run in a background thread, like training,
# so a page refresh doesn't cut them off. Their progress goes to the shared console.
# ==========================================
class ToolRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current: str | None = None

    def start(self, name: str, gen_fn, *args) -> str:
        with self._lock:
            if self.current is not None:
                return f"⚠️ {self.current} is still running; wait for it to finish."
            self.current = name

        def run():
            try:
                for _ in gen_fn(*args):
                    pass
            except Exception as e:
                console.append(f"❌ {name} crashed: {e}")
            finally:
                with self._lock:
                    self.current = None

        threading.Thread(target=run, daemon=True, name=f"tool-{name}").start()
        return ""

    def busy(self) -> str | None:
        with self._lock:
            return self.current


tool_runner = ToolRunner()


def start_tool(name: str, gen_fn, *args) -> str:
    if training_manager.is_running():
        console.append(f"⚠️ Training is running: {name} will share the GPU with it and may run out of VRAM.")
    msg = tool_runner.start(name, gen_fn, *args)
    if msg:
        console.append(msg)
    return console.text()


# ==========================================
# OUTPUT LAYOUT
# training/output/<keyword>/<YYYYmmdd_HHMMSS>/  -> one folder per run:
#   <name>-step00000300.safetensors, <name>.safetensors, <name>-step00000300-state/,
#   sample/, configs/, logs/, run.json
# ==========================================
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def project_folder_name(trigger_word: str) -> str:
    """Filesystem-safe folder/file name for a keyword. Keeps non-Latin text (only characters Windows
    forbids are replaced), so e.g. a Japanese keyword gets its own folder instead of 'untitled'."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', (trigger_word or "").strip())
    name = re.sub(r'\s+', '_', name).strip(' ._')
    if not name:
        return "untitled"
    if name.split('.')[0].upper() in _WIN_RESERVED:
        name = "_" + name
    return name[:100]


def new_run_dir(project_dir: Path) -> Path:
    base = time.strftime("%Y%m%d_%H%M%S")
    run_dir, n = project_dir / base, 2
    while run_dir.exists():
        run_dir, n = project_dir / f"{base}_{n}", n + 1
    return run_dir


def read_state_step(state_dir: Path) -> int | None:
    """Step stored in an sd-scripts state folder, or None if the folder is incomplete. accelerate writes
    the RNG file last, so its presence means the save finished."""
    ts = state_dir / "train_state.json"
    if not ts.is_file() or not any(state_dir.glob("random_states_*.pkl")):
        return None
    try:
        with open(ts, "r", encoding="utf-8") as f:
            return int(json.load(f)["current_step"])
    except Exception:
        return None


def find_resume_point(project_dir: Path) -> dict | None:
    """Newest unfinished run of this keyword that has a complete saved state, with its newest state.
    Stops at the newest finished run: older runs behind it are treated as abandoned, not resumed silently."""
    if not project_dir.is_dir():
        return None
    for run_dir in sorted((d for d in project_dir.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True):
        configs = run_dir / "configs"
        training_tomls = sorted(configs.glob("*_training.toml"))
        dataset_tomls = sorted(configs.glob("*_dataset.toml"))
        if not training_tomls or not dataset_tomls:
            continue
        try:
            cfg = toml.load(str(training_tomls[0]))
            max_steps = int(cfg.get("max_train_steps", 0))
        except Exception:
            continue
        # sd-scripts writes <output_name>.safetensors only when a run completes
        if (run_dir / f"{cfg.get('output_name', '')}.{cfg.get('save_model_as', 'safetensors')}").exists():
            return None
        best = None
        for state_dir in run_dir.glob("*-state"):  # step states and the end-of-run <name>-state
            step = read_state_step(state_dir)
            if step is not None and step < max_steps and (best is None or step > best[0]):
                best = (step, state_dir)
        if best:
            return {"run_dir": run_dir, "state_dir": best[1], "step": best[0], "max_steps": max_steps,
                    "training_toml": training_tomls[0], "dataset_toml": dataset_tomls[0]}
    return None


def build_train_command(training_toml: Path, dataset_toml: Path, extra_args: Tuple[str, ...] = ()) -> List[str]:
    return [
        str(PORTABLE_PYTHON.resolve()), "-m", "accelerate.commands.launch", "--num_processes=1", "--mixed_precision=bf16", "--dynamo_backend=no",
        TRAIN_SCRIPT.resolve().as_posix(),
        "--config_file", Path(training_toml).resolve().as_posix(),
        "--dataset_config", Path(dataset_toml).resolve().as_posix(),
        *extra_args,
    ]


def build_train_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRAIN_DIR.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    env["KMP_WARNINGS"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["ACCELERATE_USE_CPU"] = "False"
    return env


CUDA_ERROR = (
    "❌ ERROR: NVIDIA GPU not detected or CUDA drivers are not installed!\n\n"
    "Technical details:\n"
    "- PyTorch cannot initialize CUDA.\n"
    "- Training on CPU is extremely slow and is not supported by this script.\n\n"
    "Please update your NVIDIA drivers and restart the app."
)


DEFAULT_SETTINGS = {
    "trigger_word": "",
    "dataset_path": "",
    "dit_path": str(ROOT / "models" / "anima" / "dit" / "anima-base-v1.0.safetensors"),  # the model card says to train LoRAs on base
    "qwen_path": str(ROOT / "models" / "anima" / "text_encoder" / "qwen_3_06b_base.safetensors"),
    "vae_path": str(ROOT / "models" / "anima" / "vae" / "qwen_image_vae.safetensors"),
    "network_rank": 32,
    "learning_rate": "1.0", # Prodigy default
    "optimizer": "Prodigy", # Prodigy default
    "training_steps": 2400,
    "save_steps": 300,
    "sample_steps": 300,
    "pos_prompt": "",
    "neg_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name",
    "width": 1024,
    "height": 1024,
    "sample_steps_gen": 30,
    "sample_cfg": 4.0,
    "sample_seed": 42,
    "train_seed": 42,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "side_min": 512,
    "side_max": 768,
    "tagger_gen_thresh": 0.35,
    "tagger_char_thresh": 0.85,
    "tagger_overwrite": False,
    "save_resume_state": True,
    "blocks_to_swap": 0,  # keep new keys at the end: settings are zipped with the UI components in order
}
def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings

def save_settings(settings_dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4)

def auto_save_state(*args):
    keys = list(DEFAULT_SETTINGS.keys())
    current_state = dict(zip(keys, args))
    save_settings(current_state)


HIDDEN_SETTINGS = {
    "lr_scheduler": "cosine",
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "gradient_checkpointing": True,
    "network_module": "networks.lora_anima",
    "network_train_unet_only": True,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "weighting_scheme": "none",
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs_to_disk": True,
    "cache_text_encoder_outputs": True,
    "sdpa": True,
    "weighting_scheme": "logit_normal",
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": True,
    "max_grad_norm": 1.0,
    "vae_batch_size": 1,
    "sigmoid_scale": 1.3,
}


def analyze_dataset_resolution(dataset_path: str) -> Tuple[int, int]:
    path = Path(dataset_path)
    if not path.exists(): return 512, 768

    valid_exts = VALID_IMAGE_EXTS
    image_files = [f for f in path.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    if not image_files: return 512, 768

    max_area = 0
    max_side = 0

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                max_area = max(max_area, w * h)
                max_side = max(max_side, w, h)
        except Exception: continue

    # Calculate the side of the square that covers the area of the largest bucket
    # If the largest is 512x768, this yields 640.
    base_res = int(math.ceil(math.sqrt(max_area) / 64.0) * 64)
    max_bucket = int(math.ceil(max_side / 64.0) * 64)

    return base_res, max_bucket


def create_sample_prompts(project_name, trigger_word, pos_prompt, neg_prompt, width, height, steps_gen, cfg, seed, out_dir):
    prompt_path = out_dir / f"{project_name}_prompts.txt"
    trigger = trigger_word.strip()
    # One preview prompt per line (sd-scripts reads the file line by line; '#' lines are comments).
    user_prompts = [line.strip() for line in (pos_prompt or "").splitlines() if line.strip()] or [""]

    actual_neg = neg_prompt.strip().replace("\n", " ")
    lines = []
    for user_prompt in user_prompts:
        if trigger and not user_prompt.startswith(trigger):
            actual_pos = f"{trigger}, {user_prompt}" if user_prompt else trigger
        else:
            actual_pos = user_prompt if user_prompt else trigger
        lines.append(f"{actual_pos} --n {actual_neg} --w {int(width)} --h {int(height)} --l {float(cfg)} --s {int(steps_gen)} --d {int(seed)}")
    with open(prompt_path, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
    return str(prompt_path)

def create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, out_dir, max_steps, batch_size, grad_acc, add_prefix=True):
    config_path = out_dir / f"{project_name}_dataset.toml"
    # sd-scripts builds `caption_prefix + " " + caption`, so no trailing space here ("trigger, tags", not "trigger,  tags")
    prefix = f"{trigger_word.strip()}," if trigger_word.strip() and add_prefix else None

    path = Path(dataset_path)
    valid_exts = VALID_IMAGE_EXTS
    num_images = len([f for f in path.glob('*') if f.is_file() and f.suffix.lower() in valid_exts])

    if num_images == 0: num_images = 1

    effective_batch = int(batch_size) * int(grad_acc)
    calculated_repeats = (int(max_steps) * effective_batch) / num_images
    final_repeats = max(1, math.ceil(calculated_repeats))

    dataset_config = {
        "general": {
            "enable_bucket": True, 
            "min_bucket_reso": 256, 
            "max_bucket_reso": max_bucket, 
            "bucket_reso_steps": 64, 
            "bucket_no_upscale": True
        },
        "datasets": [{
            "resolution": base_res, 
            "subsets": [{
                "image_dir": Path(dataset_path).resolve().as_posix(), 
                "caption_extension": ".txt", 
                "num_repeats": final_repeats,
                "caption_prefix": prefix, 
                "keep_tokens": 1, 
                "caption_dropout_rate": 0.05
            }]
        }]
    }
    with open(config_path, "w", encoding="utf-8") as f: 
        toml.dump(dataset_config, f)
    return str(config_path)

def create_training_toml(project_name, config_save_dir, actual_output_dir, rank, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, halt_request_file, save_resume_state, blocks_to_swap=0):
    config_path = config_save_dir / f"{project_name}_training.toml"

    network_rank = int(rank)
    network_alpha = network_rank
    
    opt_args = ["weight_decay=0.01"]
    if optimizer == "Prodigy":
        scheduler = "constant"
        opt_args = ["decouple=True", "weight_decay=0.01", "d_coef=1.0", "use_bias_correction=True", "safeguard_warmup=True", "betas=0.9,0.99"]
    else:
        scheduler = "cosine"

    training_config = {
        "pretrained_model_name_or_path": Path(models["dit_path"]).resolve().as_posix(),
        "qwen3": Path(models["qwen_path"]).resolve().as_posix(),
        "vae": Path(models["vae_path"]).resolve().as_posix(),
        "network_module": HIDDEN_SETTINGS["network_module"],
        "network_dim": int(rank),
        "network_alpha": network_alpha,
        "network_train_unet_only": HIDDEN_SETTINGS["network_train_unet_only"],
        "gradient_checkpointing": HIDDEN_SETTINGS["gradient_checkpointing"],
        "max_grad_norm": 1.0,
        "learning_rate": float(lr),
        "optimizer_type": optimizer,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        "max_train_steps": int(max_steps),
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "sample_prompts": Path(prompt_path).resolve().as_posix(),
        "sample_sampler": "euler",
        "timestep_sampling": HIDDEN_SETTINGS["timestep_sampling"],
        "discrete_flow_shift": HIDDEN_SETTINGS["discrete_flow_shift"],
        "sigmoid_scale": HIDDEN_SETTINGS["sigmoid_scale"],
        "weighting_scheme": HIDDEN_SETTINGS["weighting_scheme"],
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "attn_mode": "sdpa",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "max_data_loader_n_workers": 4,
        "vae_chunk_size": 32,
        "vae_disable_cache": True,
        "seed": int(train_seed),
        # Halt & save: the trainer polls for this file once per step (see train_network.py, [TrainFlow]).
        "halt_request_file": Path(halt_request_file).resolve().as_posix(),
    }
    # 0 (or less) disables the interval; sd-scripts would otherwise divide by zero.
    if int(save_steps) > 0:
        training_config["save_every_n_steps"] = int(save_steps)
        if save_resume_state:
            # Resumable state (weights + optimizer + scheduler + RNG + step) next to each checkpoint;
            # older states are pruned so only the two most recent remain.
            training_config["save_state"] = True
            training_config["save_last_n_steps_state"] = int(save_steps)
    if int(sample_steps) > 0:
        training_config["sample_every_n_steps"] = int(sample_steps)
    if int(blocks_to_swap or 0) > 0:
        # Low VRAM: keep this many of the DiT's transformer blocks in system RAM and stream them in (slower).
        training_config["blocks_to_swap"] = int(blocks_to_swap)
    with open(config_path, "w", encoding="utf-8") as f: toml.dump(training_config, f)
    return str(config_path)


def get_latest_images(sample_dir, since: float = 0.0):
    """Sample images in sample_dir, newest first. `since` hides images left over from earlier runs."""
    if sample_dir is None or not sample_dir.exists(): return []
    images = glob.glob(str(sample_dir / "*.png")) + glob.glob(str(sample_dir / "*.jpg")) + glob.glob(str(sample_dir / "*.webp"))
    stamped = []
    for img in images:
        try:
            mtime = os.path.getmtime(img)
        except OSError:
            continue  # deleted between glob and stat
        if mtime >= since - 1.0:
            stamped.append((mtime, img))
    stamped.sort(reverse=True)

    def caption(img):
        m = SAMPLE_NAME_RE.search(Path(img).name)
        return f"step {int(m.group(1))} · prompt {int(m.group(2)) + 1}" if m else Path(img).name
    return [(img, caption(img)) for _, img in stamped]


# ==========================================
# CROPPER CLASS (U2Net Head-First + Buckets)
# ==========================================
class SmartCropper:
    def __init__(self) -> None:
        self.session: rt.InferenceSession | None = None
        self.model_path: Path = ROOT / "models" / "u2net" / "u2net.onnx"

    def load_model(self) -> str:
        if self.session is not None: 
            return "Already loaded"
        providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        self.session = rt.InferenceSession(str(self.model_path), providers=providers)
        return "GPU" if "CUDA" in self.session.get_providers()[0] else "CPU"

    def get_valid_buckets(self, side_min: int, side_max: int) -> List[Tuple[int, int]]:
        s_min, s_max = int(side_min), int(side_max)
        buckets = {(s_min, s_min)} 
        for s in range(s_min + 64, s_max + 64, 64):
            buckets.add((s_min, s))
            buckets.add((s, s_min))
        return sorted(list(buckets), key=lambda x: (x[0] * x[1]))

    def get_best_bucket(self, w: int, h: int, buckets: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], str]:
        orig_ar = w / h
        log_orig = math.log(orig_ar)
        best_b = buckets[0]
        min_diff = float('inf')
        
        for b in buckets:
            b_ar = b[0] / b[1]
            diff = abs(math.log(b_ar) - log_orig)
            if diff < min_diff:
                min_diff = diff
                best_b = b
        
        reason = f"AR: {orig_ar:.2f} -> {best_b[0]/best_b[1]:.2f}"
        return best_b, reason

    def process_image(self, original_img: np.ndarray, tw: int, th: int) -> np.ndarray:
        """Crop/resize an upright RGB image to (tw, th), keeping the U2Net-detected subject's top in frame."""
        h_orig, w_orig = original_img.shape[:2]

        if abs((w_orig / h_orig) - (tw / th)) < 0.01:
            return cv2.resize(original_img, (tw, th), interpolation=cv2.INTER_AREA)

        low_res_scale = 1024 / max(h_orig, w_orig)
        img_sm = cv2.resize(original_img, (int(w_orig*low_res_scale), int(h_orig*low_res_scale)), interpolation=cv2.INTER_AREA)

        input_size = 320
        img_inp = cv2.resize(img_sm, (input_size, input_size), interpolation=cv2.INTER_AREA)
        img_inp = img_inp.astype(np.float32) / 255.0
        # ImageNet mean/std in RGB order, as rembg feeds U2Net (the input used to be BGR from cv2.imread)
        img_inp -= [0.485, 0.456, 0.406]
        img_inp /= [0.229, 0.224, 0.225]
        input_tensor = np.expand_dims(np.transpose(img_inp, (2, 0, 1)), 0)

        mask = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0][0][0]
        mask = cv2.resize(mask, (img_sm.shape[1], img_sm.shape[0]))

        y_idx, x_idx = np.where(mask > 0.15)
        if len(y_idx) > 0:
            top_y = int(np.min(y_idx) / low_res_scale)
            center_x = int(np.mean(x_idx) / low_res_scale)
        else:
            top_y, center_x = h_orig // 4, w_orig // 2

        scale = max(tw / w_orig, th / h_orig)
        cw, ch = int(tw / scale), int(th / scale)
        y1 = max(0, min(top_y - int(ch * 0.05), h_orig - ch))
        x1 = max(0, min(center_x - cw // 2, w_orig - cw))

        return cv2.resize(original_img[y1:y1+ch, x1:x1+cw], (tw, th), interpolation=cv2.INTER_AREA)

    def unload_model(self) -> None:
        self.session = None

smart_cropper = SmartCropper()

def generate_bucket_summary(bucket_counts: dict) -> List[str]:
    """Generates a nicely formatted text report on image distribution."""
    if not bucket_counts:
        return []
        
    lines = ["", "📊 Bucket Distribution Summary:"]
    total = 0
    # Sort keys (WxH tuples) by area for pretty printing
    for bkt in sorted(bucket_counts.keys(), key=lambda x: x[0] * x[1]):
        count = bucket_counts[bkt]
        total += count
        lines.append(f"  ├─ {bkt[0]}x{bkt[1]}: {count} images")
    lines.append(f"  └─ Total Dataset Size: {total} images")
    lines.append("")
    return lines

EXIF_ORIENTATION_TAG = 0x0112
BUCKET_MANIFEST = "bucketing_manifest.json"


def exif_orientation(img: Image.Image) -> int:
    try:
        return int(img.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    except Exception:
        return 1


def oriented_size(img: Image.Image) -> Tuple[int, int]:
    """(width, height) as the image is meant to be displayed. EXIF orientations 5-8 are rotated by 90°,
    which is how phones store portrait photos (landscape pixels + a rotate flag)."""
    w, h = img.size
    return (h, w) if exif_orientation(img) in (5, 6, 7, 8) else (w, h)


def load_rgb_upright(img_path: Path) -> np.ndarray:
    """Decode with PIL (non-ASCII paths work on Windows, unlike cv2.imread), apply the EXIF orientation,
    and flatten any transparency onto white. Returns an RGB uint8 array."""
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
            canvas.alpha_composite(im)
            im = canvas
        return np.asarray(im.convert("RGB"))


def _file_fingerprint(p: Path) -> dict:
    st = p.stat()
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _load_manifest(backup_dir: Path) -> dict:
    mf = backup_dir / BUCKET_MANIFEST
    if mf.exists():
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_manifest(backup_dir: Path, manifest: dict) -> None:
    mf = backup_dir / BUCKET_MANIFEST
    tmp = mf.with_name(mf.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    os.replace(tmp, mf)


def run_smart_crop_ui(dataset_dir: str, side_min: float, side_max: float) -> Generator[str, None, None]:
    """Bucket the dataset. Originals live in <dataset>/original_images and are never overwritten; the
    dataset folder holds one <stem>.png per original, generated from it. Re-running with different sizes
    regenerates those PNGs from the originals instead of treating them as new originals."""
    path = Path(dataset_dir) if dataset_dir else None

    if path is None or not path.is_dir():
        console.append("❌ Error: Dataset path invalid!")
        yield console.text()
        return

    backup_dir = path / "original_images"
    valid_exts = VALID_IMAGE_EXTS

    available_buckets = smart_cropper.get_valid_buckets(int(side_min), int(side_max))
    console.append(f"📋 Buckets configured: {len(available_buckets)}")
    yield console.text()

    manifest = _load_manifest(backup_dir)
    backup_stems = set()
    if backup_dir.is_dir():
        backup_stems = {f.stem for f in backup_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts}

    # 1. Classify what is in the dataset folder.
    #    A <stem> that already has an original in original_images/ means the file here was produced by
    #    bucketing. If it no longer matches what bucketing wrote (per the manifest), someone edited or
    #    replaced it: leave it alone rather than overwrite it.
    new_originals: List[Path] = []
    edited: set = set()
    for f in sorted(path.iterdir()):
        if not f.is_file() or f.suffix.lower() not in valid_exts:
            continue
        if f.stem in backup_stems:
            rec = manifest.get(f.name)
            if rec is not None and {"bytes": rec.get("bytes"), "mtime_ns": rec.get("mtime_ns")} != _file_fingerprint(f):
                edited.add(f.name)
            continue  # produced by an earlier run (or pre-manifest output): regenerate from the original if needed
        new_originals.append(f)

    # Two new originals with the same stem (cat.jpg + cat.png) would both become cat.png.
    stems: dict = {}
    for f in new_originals:
        stems.setdefault(f.stem, []).append(f.name)
    clashes = [names for names in stems.values() if len(names) > 1]
    if clashes:
        console.append("❌ Bucketing aborted, nothing was changed. These files share a name and would overwrite each other:")
        for names in clashes:
            console.append(f"  • {', '.join(names)}")
        console.append("  Rename them (and their .txt captions) and run again.")
        yield console.text()
        return

    # 2. Move new originals aside. A PNG that is already upright and exactly bucket-sized is kept in place.
    moved = 0
    for f in new_originals:
        try:
            if f.suffix.lower() == '.png':
                with Image.open(f) as img:
                    if img.size in available_buckets and exif_orientation(img) == 1:
                        continue
        except Exception:
            pass
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / f.name
        if dest.exists():  # unreachable given the stem check above; never overwrite an original
            console.append(f"⚠️ {f.name}: already exists in original_images, left untouched.")
            continue
        shutil.move(str(f), str(dest))
        moved += 1
    if moved:
        console.append(f"📦 Moved {moved} new originals to original_images/")

    # 3. Task preparation (Idempotency + Statistics collection)
    originals = sorted(f for f in backup_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts) if backup_dir.is_dir() else []
    tasks = []
    bucket_counts = {}
    skipped = 0
    kept_edited = []
    unreadable = []

    for orig in originals:
        try:
            with Image.open(orig) as img:
                w, h = oriented_size(img)
            (tw, th), reason = smart_cropper.get_best_bucket(w, h, available_buckets)
            out_p = path / f"{orig.stem}.png"

            if out_p.name in edited:
                kept_edited.append(out_p.name)
                continue
            if out_p.exists():
                with Image.open(out_p) as check:
                    if check.size == (tw, th):
                        skipped += 1
                        bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                        continue
            tasks.append((orig, out_p, tw, th))
        except Exception as e:
            unreadable.append(f"{orig.name}: {e}")

    for name in unreadable:
        console.append(f"⚠️ Could not read original {name}")
    if kept_edited:
        console.append(f"✋ Kept {len(kept_edited)} image(s) unchanged because they were modified after bucketing: {', '.join(kept_edited[:10])}{' …' if len(kept_edited) > 10 else ''}")
        console.append("   Delete one to have it regenerated from its original.")

    if not tasks:
        console.append(f"✅ All {skipped} images are already bucketed correctly.")
        console.extend(generate_bucket_summary(bucket_counts))
        yield console.text()
        return

    # 4. Multi-threaded inference
    processed = 0
    failed = 0
    try:
        console.append(f"🚀 Model: {smart_cropper.load_model()}")
        console.append(f"⚙️ Processing {len(tasks)} images (Skipped: {skipped})...")
        yield console.text()

        lock = threading.Lock()

        def worker(task_data: Tuple[Path, Path, int, int]) -> str | None:
            in_p, out_p, tw, th = task_data
            nonlocal processed, failed
            try:
                img = load_rgb_upright(in_p)
                res = smart_cropper.process_image(img, tw, th)
                tmp = out_p.with_name(out_p.name + ".part")
                Image.fromarray(res).save(tmp, format="PNG", compress_level=4)
                os.replace(tmp, out_p)  # never leave a half-written PNG in the dataset
                fp = _file_fingerprint(out_p)
                with lock:
                    processed += 1
                    bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                    manifest[out_p.name] = {"source": in_p.name, "size": [tw, th], **fp}
                return None
            except Exception as e:
                with lock:
                    failed += 1
                return f"⚠️ {in_p.name}: {str(e)}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, t) for t in tasks]

            for future in as_completed(futures):
                err = future.result()
                if err: console.append(err)

                done = processed + failed
                if done % 5 == 0 or done == len(tasks):
                    msg = f"🔄 Progress: {done}/{len(tasks)}"
                    console.replace_last_if(msg, "Progress:")
                    yield console.text()

        console.append(f"✅ Done! Processed: {processed}" + (f" | ❌ Failed: {failed} (their originals are untouched in original_images/)" if failed else ""))

    except Exception as e:
        console.append(f"❌ Error: {str(e)}")
    finally:
        smart_cropper.unload_model()
        try:
            if backup_dir.is_dir():
                _save_manifest(backup_dir, manifest)
        except Exception as e:
            console.append(f"⚠️ Could not save {BUCKET_MANIFEST}: {e}")
        console.extend(generate_bucket_summary(bucket_counts))
        yield console.text()


# ==========================================
# TAGGER CLASS (WD14)
# ==========================================
class WDTagger:
    def __init__(self):
        self.model = None
        self.tag_names = []
        self.general_indexes = []
        self.character_indexes = []
        self.target_size = 448
        self.model_dir = ROOT / "models" / "wd-eva02-large-tagger-v3"
        self.model_path = self.model_dir / "model.onnx"
        self.csv_path = self.model_dir / "selected_tags.csv"
        self.kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]

    def load_model(self):
        if self.model is not None:
            return "Already loaded"
            
        if not self.model_path.exists() or not self.csv_path.exists():
            raise FileNotFoundError(f"Model or CSV not found in {self.model_dir}.")

        df = pd.read_csv(self.csv_path)
        name_series = df["name"].map(lambda x: str(x).replace("_", " ") if pd.notna(x) and str(x) not in self.kaomojis else str(x))
        self.tag_names = name_series.tolist()
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        
        self.model = rt.InferenceSession(str(self.model_path), providers=providers)
        current_provider = self.model.get_providers()[0]
        return "GPU (CUDA)" if "CUDA" in current_provider else "CPU (Slow Mode)"

    def preprocess(self, image):
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        image = canvas.convert("RGB")
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        if max_dim != self.target_size:
            padded_image = padded_image.resize((self.target_size, self.target_size), Image.BICUBIC)
        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1] # BGR
        return np.expand_dims(image_array, axis=0)

    def predict(self, image, gen_thresh, char_thresh):
        image_array = self.preprocess(image)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: image_array})
        preds = outputs[0][0]
        
        general_tags = [self.tag_names[i] for i in self.general_indexes if i < len(preds) and preds[i] > gen_thresh]
        char_tags = [self.tag_names[i] for i in self.character_indexes if i < len(preds) and preds[i] > char_thresh]
        
        char_tags = [char.replace("(", r"\(").replace(")", r"\)") for char in char_tags]
        general_tags = sorted(general_tags, key=lambda x: preds[self.tag_names.index(x)], reverse=True)
        
        final_list = []
        if char_tags: final_list.append(", ".join(char_tags))
        if general_tags: final_list.append(", ".join(general_tags).replace("(", r"\(").replace(")", r"\)"))
        return ", ".join(final_list)

    def unload_model(self):
        import gc
        import torch
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()    

wd_tagger = WDTagger()

# ==========================================
# GUI EXECUTOR FUNCTIONS
# ==========================================

def run_auto_tagging(dataset_dir, gen_thresh, char_thresh, overwrite):
    
    if not dataset_dir or not os.path.exists(dataset_dir):
        console.append("❌ Tagger Error: Dataset path invalid!")
        yield console.text()
        return

    valid_exts = VALID_IMAGE_EXTS
    all_files = [f for f in Path(dataset_dir).glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    
    image_files = []
    skipped = 0
    for f in all_files:
        if f.with_suffix('.txt').exists() and not overwrite:
            skipped += 1
        else:
            image_files.append(f)

    if not image_files and skipped == 0:
        console.append("❌ Tagger Error: No images found!")
        yield console.text()
        return
        
    if not image_files and skipped > 0:
        console.append(f"ℹ️ All images already have captions. Skipped: {skipped}")
        yield console.text()
        return

    console.append(f"🏷️ Initializing WD-Tagger (Multi-threaded)...")
    yield console.text()
    
    try:
        mode = wd_tagger.load_model()
        console.append(f"🚀 Model loaded using: {mode}")
        console.append(f"⚙️ Processing {len(image_files)} images in 4 threads... (Skipped: {skipped})")
        yield console.text()

        processed_count = 0
        total_to_process = len(image_files)
        lock = threading.Lock()

        def process_single_image(img_path):
            nonlocal processed_count
            try:
                with Image.open(img_path) as img:
                    upright = ImageOps.exif_transpose(img)  # tag what the photo actually shows
                    tags = wd_tagger.predict(upright, gen_thresh, char_thresh)
                with open(img_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(tags)
                with lock:
                    processed_count += 1
                return None
            except Exception as e:
                return f"⚠️ Error {img_path.name}: {str(e)}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_image, path) for path in image_files]
            
            for future in as_completed(futures):
                error_msg = future.result()
                if error_msg: console.append(error_msg)
                
                if processed_count % 5 == 0 or processed_count == total_to_process:
                    msg = f"🔄 Tag Progress: {processed_count}/{total_to_process} images."
                    console.replace_last_if(msg, "Tag Progress:")
                    yield console.text()

        console.append(f"✅ Tagging complete! Processed: {processed_count} | Skipped: {skipped}")
        
    except Exception as e:
        console.append(f"❌ Critical Tagger Error: {str(e)}")
        
    finally:
        wd_tagger.unload_model()
        console.append("🧹 VRAM cleared.")
        yield console.text()


def start_bucketing(dataset_dir, side_min, side_max):
    return start_tool("Bucketing", run_smart_crop_ui, dataset_dir, side_min, side_max)


def start_tagging(dataset_dir, gen_thresh, char_thresh, overwrite):
    return start_tool("Auto-captioning", run_auto_tagging, dataset_dir, gen_thresh, char_thresh, overwrite)


def start_inspect(dataset_dir, trigger_word):
    msg = tool_runner.start("Dataset inspection", inspect_dataset, dataset_dir, trigger_word)
    if msg:
        console.append(msg)
    return console.text()


def open_dataset_folder_ui(dataset_dir):
    if not dataset_dir or not os.path.exists(dataset_dir):
        console.append("❌ Error: Dataset path invalid or empty!")
        return console.text()
    
    target_dir = Path(dataset_dir)
    if platform.system() == "Windows": os.startfile(target_dir)
    else: subprocess.Popen(["xdg-open", str(target_dir)])
    
    console.append(f"📁 Opened Dataset folder.")
    return console.text()


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

# ==========================================
# CAPTIONS / DATASET INSPECTOR
# ==========================================
def split_tags(caption: str) -> List[str]:
    return [t.strip() for t in caption.replace("\n", ",").split(",") if t.strip()]


def trigger_caption_counts(dataset_dir: str, trigger_word: str) -> Tuple[int, int]:
    """(captions that already contain the trigger as a tag, captions read)."""
    trig = trigger_word.strip().lower()
    path = Path(dataset_dir) if dataset_dir else None
    if not trig or path is None or not path.is_dir():
        return 0, 0
    with_trigger = total = 0
    for img in path.iterdir():
        if img.is_file() and img.suffix.lower() in VALID_IMAGE_EXTS:
            txt = img.with_suffix(".txt")
            if txt.is_file():
                total += 1
                try:
                    tags = [t.lower() for t in split_tags(txt.read_text(encoding="utf-8", errors="ignore"))]
                except OSError:
                    continue
                if trig in tags:
                    with_trigger += 1
    return with_trigger, total


def _examples(names: List[str], n: int = 5) -> str:
    return ", ".join(names[:n]) + (f" … (+{len(names) - n})" if len(names) > n else "")


def inspect_dataset(dataset_dir: str, trigger_word: str) -> Generator[str, None, None]:
    path = Path(dataset_dir) if dataset_dir else None
    if path is None or not path.is_dir():
        console.append("❌ Inspect: Dataset path invalid!")
        yield console.text()
        return
    console.append(f"🔎 Inspecting {path} ...")
    yield console.text()

    images = sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS)
    stems = {f.stem for f in images}
    txts = sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".txt")
    missing = [f.name for f in images if not f.with_suffix(".txt").is_file()]
    orphans = [f.name for f in txts if f.stem not in stems]
    stem_clash = sorted({f.stem for f in images if sum(1 for g in images if g.stem == f.stem) > 1})

    trig = trigger_word.strip().lower()
    tag_counts: dict = {}
    empty, dup_tags, with_trigger, n_caps, tags_total = [], [], 0, 0, 0
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.is_file():
            continue
        try:
            tags = split_tags(txt.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        n_caps += 1
        tags_total += len(tags)
        if not tags:
            empty.append(txt.name)
        low = [t.lower() for t in tags]
        if len(set(low)) != len(low):
            dup_tags.append(txt.name)
        if trig and trig in low:
            with_trigger += 1
        for t in set(low):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    sizes, rotated, oversized, small, unreadable = [], [], [], [], []
    for img in images:
        try:
            with Image.open(img) as im:
                w, h = oriented_size(im)
                if exif_orientation(im) != 1:
                    rotated.append(img.name)
                if im.size[0] >= 2048 or im.size[1] >= 2048:
                    oversized.append(img.name)
            sizes.append((w, h))
            if min(w, h) < 512:
                small.append(img.name)
        except Exception:
            unreadable.append(img.name)

    out = [f"📋 Images: {len(images)} | Captions: {n_caps} | Avg tags per caption: {tags_total / n_caps:.1f}" if n_caps
           else f"📋 Images: {len(images)} | Captions: 0"]
    if sizes:
        out.append(f"📐 Short side {min(min(s) for s in sizes)}–{max(min(s) for s in sizes)}px, "
                   f"long side {min(max(s) for s in sizes)}–{max(max(s) for s in sizes)}px")
    problems = 0
    def flag(cond, line):
        nonlocal problems
        if cond:
            problems += 1
            out.append(line)
    flag(missing, f"❌ {len(missing)} image(s) without a caption: {_examples(missing)}")
    flag(unreadable, f"❌ {len(unreadable)} image(s) can't be opened: {_examples(unreadable)}")
    flag(stem_clash, f"❌ Same name with different extensions (captions are shared, bucketing refuses): {_examples(stem_clash)}")
    flag(oversized, f"❌ {len(oversized)} image(s) are 2048px or larger (training refuses them): {_examples(oversized)}")
    flag(empty, f"⚠️ {len(empty)} empty caption(s): {_examples(empty)}")
    flag(orphans, f"⚠️ {len(orphans)} caption file(s) with no image: {_examples(orphans)}")
    flag(rotated, f"⚠️ {len(rotated)} image(s) have an EXIF rotation sd-scripts ignores — bucket them first: {_examples(rotated)}")
    flag(dup_tags, f"⚠️ {len(dup_tags)} caption(s) repeat a tag: {_examples(dup_tags)}")
    flag(small, f"ℹ️ {len(small)} image(s) have a short side under 512px (trained at their own lower resolution): {_examples(small)}")
    if trig and n_caps:
        if with_trigger == n_caps:
            out.append(f"ℹ️ Every caption already contains '{trigger_word.strip()}': training won't add it again.")
        elif with_trigger:
            problems += 1
            out.append(f"⚠️ {with_trigger}/{n_caps} captions already contain '{trigger_word.strip()}' and will get it twice "
                       f"(training adds it to every caption). Remove it from those, or add it to all of them.")
    if tag_counts:
        top = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        everywhere = [t for t, c in top if c == n_caps and t != trig]
        out.append(f"🏷️ {len(tag_counts)} distinct tags. Most frequent:")
        for t, c in top[:20]:
            out.append(f"   {c:>4} ({c * 100 // n_caps:>3}%)  {t}")
        if everywhere:
            out.append(f"ℹ️ In every caption: {_examples(everywhere, 10)} — tags present everywhere tend to get "
                       f"absorbed into the trigger rather than stay promptable.")
    out.append("✅ No problems found." if problems == 0 else f"🔎 Inspection done: {problems} thing(s) to look at.")
    console.extend(out)
    yield console.text()


# ==========================================
# CHECKPOINT COMPARISON
# sd-scripts names previews <output_name>_<step:06d>_<prompt:02d>_<YYYYmmddHHMMSS>[_seed].png
# ==========================================
SAMPLE_NAME_RE = re.compile(r"_(\d{6})_(\d{2})_\d{14}(?:_\d+)?\.(?:png|jpg|webp)$", re.IGNORECASE)


def collect_samples(sample_dir: Path | None) -> dict:
    """{(step, prompt_index): newest image path}."""
    found: dict = {}
    if sample_dir is None or not sample_dir.is_dir():
        return found
    for p in sample_dir.iterdir():
        m = SAMPLE_NAME_RE.search(p.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if key not in found or mt > found[key][0]:
            found[key] = (mt, p)
    return {k: v[1] for k, v in found.items()}


def run_dir_for_keyword(trigger_word: str) -> Path | None:
    """The active run if it belongs to this keyword, else the keyword's newest run."""
    project_dir = OUTPUT_BASE / project_folder_name(trigger_word)
    snap = training_manager.snapshot()
    if snap["run_dir"] is not None and snap["run_dir"].parent == project_dir and snap["run_dir"].exists():
        return snap["run_dir"]
    if project_dir.is_dir():
        runs = sorted((d for d in project_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
        if runs:
            return runs[-1]
    return None


def build_checkpoint_grid(run_dir: Path, max_rows: int = 40) -> Tuple[Path | None, str]:
    samples = collect_samples(run_dir / "sample")
    if not samples:
        return None, f"ℹ️ No previews in {run_dir / 'sample'} yet."
    steps = sorted({s for s, _ in samples})[-max_rows:]
    prompts = sorted({i for _, i in samples})
    try:
        cfg = toml.load(str(sorted((run_dir / "configs").glob("*_training.toml"))[0]))
        name = cfg.get("output_name", "")
    except Exception:
        name = ""
    with Image.open(next(iter(samples.values()))) as first:
        aspect = first.height / first.width
    cell_w = 256
    cell_h = max(64, int(cell_w * aspect))
    label_w, header_h, pad = 150, 32, 6
    W = label_w + len(prompts) * (cell_w + pad) + pad
    H = header_h + len(steps) * (cell_h + pad) + pad
    grid = Image.new("RGB", (W, H), (24, 26, 33))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    try:
        font, small_font = ImageFont.load_default(size=18), ImageFont.load_default(size=13)
    except TypeError:  # Pillow < 10.1
        font = small_font = ImageFont.load_default()
    for c, idx in enumerate(prompts):
        draw.text((label_w + pad + c * (cell_w + pad), 8), f"prompt {idx + 1}", fill=(220, 220, 230), font=font)
    for r, step in enumerate(steps):
        y = header_h + pad + r * (cell_h + pad)
        draw.text((10, y + 6), f"step {step}", fill=(235, 235, 245), font=font)
        if name and (run_dir / f"{name}-step{step:08d}.safetensors").exists():
            draw.text((10, y + 30), "LoRA saved", fill=(158, 206, 106), font=small_font)
        for c, idx in enumerate(prompts):
            p = samples.get((step, idx))
            if p is None:
                continue
            try:
                with Image.open(p) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (cell_w, cell_h))
                grid.paste(thumb, (label_w + pad + c * (cell_w + pad), y))
            except Exception:
                continue
    out = run_dir / "checkpoint_grid.png"
    grid.save(out)
    return out, f"🧮 Checkpoint grid: {len(steps)} step(s) × {len(prompts)} prompt(s) → {out}"


def make_checkpoint_grid(trigger_word):
    run_dir = run_dir_for_keyword(trigger_word)
    if run_dir is None:
        console.append(f"ℹ️ No runs for '{project_folder_name(trigger_word)}' yet.")
        return gr.skip(), console.text()
    try:
        path, msg = build_checkpoint_grid(run_dir)
    except Exception as e:
        path, msg = None, f"❌ Could not build the grid: {e}"
    console.append(msg)
    return (str(path) if path else gr.skip()), console.text()


def loss_frame(points: List[Tuple[int, float]], max_points: int = 600) -> pd.DataFrame:
    if len(points) > max_points:
        stride = math.ceil(len(points) / max_points)
        points = points[::stride] + ([points[-1]] if (len(points) - 1) % stride else [])
    return pd.DataFrame(points, columns=["step", "avg_loss"])


# ==========================================
# TRAIN MORE (extend a finished run)
# ==========================================
def find_extend_point(project_dir: Path) -> dict:
    runs = sorted((d for d in project_dir.iterdir() if d.is_dir()), key=lambda d: d.name) if project_dir.is_dir() else []
    if not runs:
        return {"error": f"no runs for '{project_dir.name}' yet"}
    run_dir = runs[-1]
    tomls = sorted((run_dir / "configs").glob("*_training.toml"))
    datasets = sorted((run_dir / "configs").glob("*_dataset.toml"))
    if not tomls or not datasets:
        return {"error": f"{run_dir.name} has no configs"}
    try:
        cfg = toml.load(str(tomls[0]))
    except Exception as e:
        return {"error": f"could not read {tomls[0].name}: {e}"}
    name = cfg.get("output_name", "")
    final = run_dir / f"{name}.safetensors"
    if not final.exists():
        return {"error": f"the newest run ({run_dir.name}) isn't finished — use ↩ Resume Run for it"}
    state_dir = run_dir / f"{name}-state"  # sd-scripts' end-of-run state (save_state)
    step = read_state_step(state_dir)
    if step is None:
        return {"error": f"{run_dir.name} has no end-of-run resume point (it was trained with resume points off)"}
    return {"run_dir": run_dir, "state_dir": state_dir, "step": step, "max_steps": int(cfg.get("max_train_steps", step)),
            "training_toml": tomls[0], "dataset_toml": datasets[0], "cfg": cfg, "name": name, "final": final}


def extend_training(trigger_word, new_total):
    """Continue the newest finished run of this keyword up to `new_total` steps, from its end-of-run state.
    The previous final LoRA is kept (renamed to its step). Prodigy (constant LR) simply continues; with AdamW
    the cosine schedule is re-evaluated over the new length, so the LR rises again and decays to 0 by the end."""
    if training_manager.is_running():
        console.append("⚠️ Training is already running!")
        return console.text()
    project_name = project_folder_name(trigger_word)
    point = find_extend_point(OUTPUT_BASE / project_name)
    if "error" in point:
        console.append(f"ℹ️ Can't train more: {point['error']}.")
        return console.text()
    try:
        new_total = int(new_total)
    except (TypeError, ValueError):
        new_total = 0
    if new_total <= point["step"]:
        console.append(f"ℹ️ Set Training Steps above {point['step']} (what {point['run_dir'].name} already trained), then press ➕ Train More.")
        return console.text()
    cfg = point["cfg"]
    missing = [f"❌ Model file not found: {cfg.get(k)}" for k in ("pretrained_model_name_or_path", "qwen3", "vae") if not cfg.get(k) or not os.path.isfile(cfg.get(k))]
    if missing:
        console.extend(missing)
        return console.text()
    if not torch.cuda.is_available():
        console.extend(CUDA_ERROR.split("\n"))
        return console.text()

    run_dir, name, old = point["run_dir"], point["name"], point["step"]
    try:
        # keep the config this run finished with, then raise the step target
        backup = point["training_toml"].with_name(f"{name}_training.before_extend_{old}.toml")
        if not backup.exists():
            shutil.copy2(point["training_toml"], backup)
        cfg["max_train_steps"] = new_total
        with open(point["training_toml"], "w", encoding="utf-8") as f:
            toml.dump(cfg, f)
        # keep the old final LoRA; sd-scripts would overwrite <name>.safetensors at the new end
        kept = run_dir / f"{name}-step{old:08d}.safetensors"
        if kept.exists():
            kept = run_dir / f"{name}-step{old:08d}-final.safetensors"
        os.replace(point["final"], kept)
        meta = run_dir / "run.json"
        try:
            info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        except Exception:
            info = {}
        info.setdefault("extensions", []).append({"from": old, "to": new_total, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        info["max_train_steps"] = new_total
        meta.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        console.append(f"❌ Could not prepare the run for more training: {e}")
        return console.text()

    halt_file = Path(cfg.get("halt_request_file") or (run_dir / ".halt_request"))
    log_path = run_dir / "logs" / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = build_train_command(point["training_toml"], point["dataset_toml"],
                              ("--resume", point["state_dir"].resolve().as_posix(), "--skip_until_initial_step"))
    console.clear()
    console.append(f"➕ Training {project_name} — run {run_dir.name} further: step {old} → {new_total}")
    console.append(f"   Previous final LoRA kept as {kept.name}")
    if str(cfg.get("lr_scheduler", "")) == "cosine":
        console.append(f"   Cosine LR: the schedule is rebuilt for {new_total} steps, so the LR jumps back up to where that schedule is at step {old}, then decays to 0 again.")
    console.append(f"📁 Run folder: {run_dir}")
    try:
        if not training_manager.start(cmd, TRAIN_DIR.resolve(), build_train_env(), run_dir, log_path, halt_file,
                                      gallery_since=0.0, output_name=name):
            console.append("⚠️ Training is already running!")
    except Exception as e:
        console.append(f"❌ Error: {str(e)}")
    return console.text()

def start_training(trigger_word, dataset_path, dit_p, qwen_p, vae_p, rank, lr, optimizer, t_steps, save_steps, sample_steps, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, train_seed, batch_size, grad_acc, save_resume_state, blocks_to_swap=0):
    """Validate, write configs into a fresh run folder and hand the run to training_manager. Returns
    immediately; the trainer keeps running regardless of what happens to the browser tab."""
    if training_manager.is_running():
        console.append("⚠️ Training is already running!")
        return console.text()

    model_errors = []
    if not dit_p or not os.path.isfile(dit_p): model_errors.append(f"❌ DiT file not found: {dit_p}")
    if not qwen_p or not os.path.isfile(qwen_p): model_errors.append(f"❌ Qwen3 file not found: {qwen_p}")
    if not vae_p or not os.path.isfile(vae_p): model_errors.append(f"❌ VAE file not found: {vae_p}")
    
    dataset_errors = []
    preflight_warnings = []
    if not dataset_path or not os.path.exists(dataset_path): 
        dataset_errors.append(f"❌ Dataset path not found: {dataset_path}")
    else:
        valid_exts = VALID_IMAGE_EXTS
        dataset_path_obj = Path(dataset_path)
        image_files = [f for f in dataset_path_obj.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
        
        if not image_files:
            dataset_errors.append(f"❌ No valid images found in the dataset path.")
        else:
            # Check for missing text files (captions)
            missing_captions = [img.name for img in image_files if not img.with_suffix('.txt').exists()]
            if missing_captions:
                dataset_errors.append(f"❌ Error: Found {len(missing_captions)} images without .txt captions. Please use the 'Auto-Caption Dataset' tool first.")
            
            # Check for oversized images (and EXIF-rotated ones, which sd-scripts would train on unrotated)
            oversized_images = []
            rotated_images = []
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        w_img, h_img = img.size
                        if w_img >= 2048 or h_img >= 2048:
                            oversized_images.append(img_path.name)
                        if exif_orientation(img) != 1:
                            rotated_images.append(img_path.name)
                except Exception:
                    pass
            
            if rotated_images:
                preflight_warnings.append(f"⚠️ {len(rotated_images)} image(s) carry an EXIF rotation flag (typical for phone photos), which sd-scripts ignores: they would be trained sideways or mirrored. Run ✂️ Start Bucketing to save them upright, e.g. {', '.join(rotated_images[:3])}")

            if oversized_images:
                dataset_errors.append(f"❌ Error: Found {len(oversized_images)} images that are too large (>= 2048px). Please use the 'Smart Aspect Ratio Bucketing' tool to resize them before training.")

    if model_errors or dataset_errors:
        full_error = ""
        if model_errors:
            full_error += "\n".join(model_errors)
            full_error += "\n\n⚠️ ERROR: Please check and set the correct model paths in the section:\n'🔧 Paths to Models <- Set Once'\n\n"
        if dataset_errors:
            full_error += "\n".join(dataset_errors)
            
        console.extend(full_error.strip().split("\n"))
        return console.text()

    if not torch.cuda.is_available():
        console.extend(CUDA_ERROR.split("\n"))
        return console.text()
    
    project_name = project_folder_name(trigger_word)
    project_dir = OUTPUT_BASE / project_name
    run_dir = new_run_dir(project_dir)
    sample_dir = run_dir / "sample"
    project_configs_dir = run_dir / "configs"

    for d in [run_dir, sample_dir, project_configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    console.clear()
    console.append(f"🚀 Preparing: {project_name} — run {run_dir.name}")
    for warning in preflight_warnings:
        console.append(warning)
    console.append(f"🔍 Analyzing dataset images...")
    base_res, max_bucket = analyze_dataset_resolution(dataset_path)
    console.append(f"📐 Auto-Resolution Set: Base {base_res}px, Max Bucket {max_bucket}px")

    with_trigger, n_caps = trigger_caption_counts(dataset_path, trigger_word)
    add_prefix = not (n_caps > 0 and with_trigger == n_caps)
    if not add_prefix:
        console.append(f"ℹ️ Every caption already contains '{trigger_word.strip()}', so it isn't added a second time.")
    elif with_trigger:
        console.append(f"⚠️ {with_trigger}/{n_caps} captions already contain '{trigger_word.strip()}' and will get it twice. 🔎 Inspect Dataset lists them.")
    if int(blocks_to_swap or 0) > 0:
        console.append(f"🧊 Block swap: {int(blocks_to_swap)} of 28 blocks kept in system RAM (less VRAM, slower steps).")

    halt_file = run_dir / ".halt_request"
    try:
        models = {"dit_path": dit_p, "qwen_path": qwen_p, "vae_path": vae_p}
        prompt_path = create_sample_prompts(project_name, trigger_word, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, project_configs_dir)
        dataset_toml = create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, project_configs_dir, t_steps, batch_size, grad_acc, add_prefix=add_prefix)
        training_toml = create_training_toml(project_name, project_configs_dir, run_dir, rank, lr, optimizer, t_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, halt_file, save_resume_state, blocks_to_swap)
        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump({"keyword": trigger_word.strip(), "output_name": project_name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "dataset_path": str(dataset_path), "max_train_steps": int(t_steps)}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        console.append(f"❌ Could not write training config: {e}")
        return console.text()

    if not save_resume_state or int(save_steps) <= 0:
        console.append("ℹ️ Periodic resume points are off: only 💾 Halt & Save creates one.")

    log_path = run_dir / "logs" / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    console.append(f"📁 Run folder: {run_dir}")
    try:
        if not training_manager.start(build_train_command(training_toml, dataset_toml), TRAIN_DIR.resolve(), build_train_env(),
                                      run_dir, log_path, halt_file, gallery_since=time.time(), output_name=project_name):
            console.append("⚠️ Training is already running!")
    except Exception as e:
        console.append(f"❌ Error: {str(e)}")
    return console.text()


def resume_training(trigger_word):
    """Continue the newest unfinished run of this keyword from its newest saved state, with the settings
    that run was started with (its own configs), not the current UI values."""
    if training_manager.is_running():
        console.append("⚠️ Training is already running!")
        return console.text()

    project_name = project_folder_name(trigger_word)
    point = find_resume_point(OUTPUT_BASE / project_name)
    if point is None:
        console.append(f"ℹ️ Nothing to resume for '{project_name}': its newest run either finished or has no saved state yet ({OUTPUT_BASE / project_name}).")
        return console.text()

    try:
        cfg = toml.load(str(point["training_toml"]))
    except Exception as e:
        console.append(f"❌ Could not read {point['training_toml']}: {e}")
        return console.text()
    missing = [f"❌ Model file not found: {cfg.get(k)}" for k in ("pretrained_model_name_or_path", "qwen3", "vae") if not cfg.get(k) or not os.path.isfile(cfg.get(k))]
    if missing:
        console.extend(missing + ["   This run was configured with these paths; restore the files to resume it."])
        return console.text()
    if not torch.cuda.is_available():
        console.extend(CUDA_ERROR.split("\n"))
        return console.text()

    run_dir = point["run_dir"]
    halt_file = Path(cfg.get("halt_request_file") or (run_dir / ".halt_request"))
    log_path = run_dir / "logs" / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = build_train_command(point["training_toml"], point["dataset_toml"],
                              ("--resume", point["state_dir"].resolve().as_posix(), "--skip_until_initial_step"))

    console.clear()
    console.append(f"↩ Resuming {project_name} — run {run_dir.name} from step {point['step']}/{point['max_steps']}")
    console.append(f"   State: {point['state_dir'].name}")
    console.append(f"📁 Run folder: {run_dir}")
    try:
        if not training_manager.start(cmd, TRAIN_DIR.resolve(), build_train_env(), run_dir, log_path, halt_file, gallery_since=0.0,
                                      output_name=cfg.get("output_name", "")):
            console.append("⚠️ Training is already running!")
    except Exception as e:
        console.append(f"❌ Error: {str(e)}")
    return console.text()


def toggle_pause():
    state = training_manager.snapshot()["state"]
    console.append(training_manager.unpause() if state == "paused" else training_manager.pause())
    return console.text()


def halt_training():
    console.append(training_manager.halt())
    return console.text()


def stop_training():
    console.append(training_manager.stop())
    return console.text()


def open_output_folder(trigger_word):
    project_dir = OUTPUT_BASE / project_folder_name(trigger_word)
    snap = training_manager.snapshot()
    if snap["run_dir"] is not None and snap["run_dir"].parent == project_dir and snap["run_dir"].exists():
        target_dir = snap["run_dir"]
    elif project_dir.is_dir():
        runs = sorted((d for d in project_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
        target_dir = runs[-1] if runs else project_dir
    else:
        target_dir = OUTPUT_BASE
    if platform.system() == "Windows": os.startfile(target_dir)
    else: subprocess.Popen(["xdg-open", str(target_dir)])
    console.append(f"📁 Opened {target_dir}")
    return console.text()


STATUS_TEXT = {
    "idle": "⚪ Idle",
    "running": "🟢 Training",
    "paused": "⏸ Paused — VRAM still allocated",
    "halting": "💾 Halting — finishing the step and saving",
    "halted": "💾 Halted — ↩ Resume Run continues it",
    "stopping": "🛑 Stopping",
    "stopped": "🛑 Stopped",
    "finished": "✅ Finished",
    "failed": "❌ Failed — see log",
}


def status_line(snap: dict) -> str:
    text = STATUS_TEXT.get(snap["state"], snap["state"])
    if snap["step"]:
        text += f" · step {snap['step'][0]}/{snap['step'][1]}"
    if snap["run_dir"] is not None:
        text += f" · {snap['run_dir'].parent.name}/{snap['run_dir'].name}"
    job = tool_runner.busy()
    if job:
        text += f" · ⚙️ {job} running"
    return f"**Status:** {text}"


def poll_ui(seen):
    """Timer callback: send the log / gallery / status only when they changed since this tab last saw them."""
    seen = seen or {}
    version, text = console.snapshot()
    sample_dir, since = training_manager.gallery_source()
    images = get_latest_images(sample_dir, since) if sample_dir is not None else []
    gallery_sig = tuple(img for img, _ in images)
    snap = training_manager.snapshot()
    status = status_line(snap)
    pause_label = "▶ Continue" if snap["state"] == "paused" else "⏸ Pause"

    loss_version, losses = training_manager.loss_snapshot()

    log_out = text if seen.get("log") != version else gr.skip()
    gallery_out = images if seen.get("gallery") != gallery_sig else gr.skip()
    status_out = status if seen.get("status") != status else gr.skip()
    pause_out = gr.update(value=pause_label) if seen.get("pause") != pause_label else gr.skip()
    loss_out = loss_frame(losses) if seen.get("loss") != loss_version else gr.skip()
    return log_out, gallery_out, status_out, pause_out, loss_out, {"log": version, "gallery": gallery_sig, "status": status,
                                                                   "pause": pause_label, "loss": loss_version}

def handle_optimizer_change(opt, current_lr, saved_adam_lr):
    if opt == "Prodigy": return "1.0", current_lr
    return (saved_adam_lr if current_lr == "1.0" else current_lr), saved_adam_lr


# ==========================================
# UI BUILDER
# ==========================================
cs = load_settings()

# A trainer that kept running while the app was closed is picked up again here.
training_manager.reattach(OUTPUT_BASE)

with gr.Blocks(title="Anima TrainFlow: Easy LoRA Trainer for Anima 2B") as ui:
    gr.Markdown(
        "# Anima TrainFlow\n"
        '<div class="attribution"><span class="author-text">Created by ThetaCursed</span></div>',
        elem_id="main-header"
    )
    
    saved_adam_lr = gr.State(value="0.00005")

    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    trigger_word = gr.Textbox(label="Trigger Word / Project Name", value=cs.get("trigger_word", ""), placeholder="e.g., unique_style")
                    dataset_path = gr.Textbox(label="Dataset Path (Images + .txt)", value=cs.get("dataset_path", ""), placeholder="C:/Images/MyDataset")
                with gr.Accordion("🔧 Paths to Models <- Set Once", open=False):
                    dit_input = gr.Textbox(label="DiT", value=cs.get("dit_path", ""))
                    qwen_input = gr.Textbox(label="Qwen3", value=cs.get("qwen_path", ""))
                    vae_input = gr.Textbox(label="VAE", value=cs.get("vae_path", ""))
                status_md = gr.Markdown("**Status:** ⚪ Idle")
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    pause_btn = gr.Button("⏸ Pause", variant="secondary")
                    halt_btn = gr.Button("💾 Halt & Save", variant="secondary")
                    stop_btn = gr.Button("🛑 Stop", variant="stop")
                with gr.Row():
                    resume_btn = gr.Button("↩ Resume Run", variant="secondary")
                    extend_btn = gr.Button("➕ Train More (to Training Steps)", variant="secondary")
                    folder_btn = gr.Button("📁 Checkpoint Folder", variant="secondary")
            with gr.Column(scale=1):
                with gr.Row():
                    rank_input = gr.Number(label="Network Rank", value=cs.get("network_rank", 16), precision=0)
                    lr_input = gr.Textbox(label="Learning Rate", value=cs.get("learning_rate", "1.0"))
                    optimizer_input = gr.Dropdown(label="Optimizer", choices=["Prodigy", "AdamW8bit", "AdamW"], value=cs.get("optimizer", "Prodigy"))
                    train_seed_val = gr.Number(value=cs.get("train_seed", 42), visible=False)
                    batch_size_input = gr.Number(label="Batch Size", value=cs.get("train_batch_size", 1), precision=0)
                with gr.Row():
                    steps_input = gr.Number(label="Training Steps", value=cs.get("training_steps", 2400), precision=0)
                    save_steps_input = gr.Number(label="Save Every n Steps", value=cs.get("save_steps", 300), precision=0)
                    sample_steps_input = gr.Number(label="Preview Every n Steps", value=cs.get("sample_steps", 300), precision=0)
                    grad_acc_input = gr.Number(label="Gradient Accumulation", value=cs.get("gradient_accumulation_steps", 1), precision=0)
                with gr.Row():
                    blocks_swap_input = gr.Slider(
                        0, ANIMA_SWAPPABLE_BLOCKS, value=cs.get("blocks_to_swap", 0), step=1,
                        label="Block Swap (Low VRAM): DiT blocks kept in system RAM — less VRAM, slower steps (0 = off)",
                    )
                    save_resume_state_input = gr.Checkbox(
                        label="Save resume points with each checkpoint (crash recovery, ➕ Train More; uses extra disk)",
                        value=cs.get("save_resume_state", True),
                    )

    with gr.Row():
        with gr.Column(scale=1):
            output_log = gr.Textbox(label="Logs", lines=LOG_BOX__MAX_LINES, max_lines=LOG_BOX__MAX_LINES, interactive=False, autoscroll=True, elem_id="log-container")
            
            with gr.Group():
                gr.Markdown("### Smart Aspect Ratio Bucketing")
                with gr.Row():
                    side_min_input = gr.Number(label="Min Side (Base)", value=cs.get("side_min", 512), precision=0)
                    side_max_input = gr.Number(label="Max Side (Limit)", value=cs.get("side_max", 768), precision=0)
                with gr.Row():
                    crop_btn = gr.Button("✂️ Start Bucketing", variant="secondary")
                    open_ds_btn = gr.Button("📂 Open Dataset", variant="secondary")
                    
            # --- AUTO CAPTION ---
            with gr.Group():
                gr.Markdown("### Auto-Caption Dataset")
                with gr.Row():
                    tagger_gen_thresh = gr.Slider(0.0, 1.0, value=cs.get("tagger_gen_thresh", 0.35), step=0.01, label="General Tags Threshold")
                    tagger_char_thresh = gr.Slider(0.0, 1.0, value=cs.get("tagger_char_thresh", 0.85), step=0.01, label="Character Tags Threshold")
                with gr.Row():
                    tagger_btn = gr.Button("Create .txt captions", variant="secondary")        
                    tagger_overwrite = gr.Checkbox(label="Overwrite existing .txt", value=cs.get("tagger_overwrite", False)) 
                with gr.Row():
                    inspect_btn = gr.Button("🔎 Inspect Dataset", variant="secondary")
                
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Previews"):
                    preview_gallery = gr.Gallery(label="Previews", columns=2, rows=2, height=GALLERY_HEIGHT, object_fit="contain")
                with gr.Tab("Checkpoint Grid"):
                    grid_btn = gr.Button("🧮 Build grid for this keyword's run (steps × prompts)", variant="secondary")
                    grid_image = gr.Image(label="Checkpoint Grid", type="filepath", height=GALLERY_HEIGHT - 50, interactive=False)
                with gr.Tab("Loss"):
                    loss_plot = gr.LinePlot(value=loss_frame([]), x="step", y="avg_loss", title="Moving-average training loss",
                                            x_title="step", y_title="avg loss", height=GALLERY_HEIGHT - 40)
            with gr.Group():
                pos_prompt = gr.Textbox(label="Preview Prompts (one per line; trigger word added automatically)", lines=2, value=cs.get("pos_prompt", ""))
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=cs.get("neg_prompt", ""))
                with gr.Row():
                    width_input = gr.Number(label="Width", value=cs.get("width", 1024), precision=0, min_width=80)
                    height_input = gr.Number(label="Height", value=cs.get("height", 1024), precision=0, min_width=80)
                    sample_steps_gen_input = gr.Number(label="Steps", value=cs.get("sample_steps_gen", 30), precision=0, min_width=80)
                    sample_cfg_input = gr.Number(label="CFG", value=cs.get("sample_cfg", 4.0), min_width=80)
                    sample_seed_input = gr.Number(label="Seed", value=cs.get("sample_seed", 42), precision=0, min_width=80)

    training_inputs = [
        trigger_word, dataset_path, dit_input, qwen_input, vae_input,
        rank_input, lr_input, optimizer_input, 
        steps_input, save_steps_input, sample_steps_input,
        pos_prompt, neg_prompt, width_input, height_input,
        sample_steps_gen_input, sample_cfg_input, sample_seed_input, train_seed_val, batch_size_input, grad_acc_input
    ]

    # Order must match DEFAULT_SETTINGS (auto_save_state zips them together).
    all_settings_list = training_inputs + [
        side_min_input, side_max_input, tagger_gen_thresh, tagger_char_thresh, tagger_overwrite, save_resume_state_input,
        blocks_swap_input,
    ]
    assert len(all_settings_list) == len(DEFAULT_SETTINGS), "settings keys and UI components are out of sync"

    def load_state_on_refresh():
        current_settings = load_settings()
        return [current_settings.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS.keys()]

    ui.load(fn=load_state_on_refresh, inputs=None, outputs=all_settings_list)    

    # Log, previews, status and loss are pulled from the server every second rather than pushed by
    # the button handlers, so a refreshed (or newly opened) tab picks up whatever is running.
    poll_seen = gr.State(value=None)
    poll_timer = gr.Timer(POLL_INTERVAL)
    poll_outputs = [output_log, preview_gallery, status_md, pause_btn, loss_plot, poll_seen]
    ui.load(fn=poll_ui, inputs=[poll_seen], outputs=poll_outputs)
    poll_timer.tick(fn=poll_ui, inputs=[poll_seen], outputs=poll_outputs, show_progress="hidden", concurrency_limit=None)

    output_log.change(None, None, None, js=JS_SCROLL)
    optimizer_input.change(fn=handle_optimizer_change, inputs=[optimizer_input, lr_input, saved_adam_lr], outputs=[lr_input, saved_adam_lr])
    
    for comp in all_settings_list: 
        comp.change(fn=auto_save_state, inputs=all_settings_list)

    # Bucketing, tagging and inspection run as background jobs (refresh-safe); progress shows in the log.
    crop_btn.click(fn=start_bucketing, inputs=[dataset_path, side_min_input, side_max_input], outputs=[output_log])
    open_ds_btn.click(fn=open_dataset_folder_ui, inputs=[dataset_path], outputs=[output_log])
    tagger_btn.click(fn=start_tagging, inputs=[dataset_path, tagger_gen_thresh, tagger_char_thresh, tagger_overwrite], outputs=[output_log])
    inspect_btn.click(fn=start_inspect, inputs=[dataset_path, trigger_word], outputs=[output_log])
    
    start_btn.click(fn=start_training, inputs=training_inputs + [save_resume_state_input, blocks_swap_input], outputs=[output_log])
    pause_btn.click(fn=toggle_pause, outputs=output_log)
    halt_btn.click(fn=halt_training, outputs=output_log)
    stop_btn.click(fn=stop_training, outputs=output_log)
    resume_btn.click(fn=resume_training, inputs=[trigger_word], outputs=output_log)
    extend_btn.click(fn=extend_training, inputs=[trigger_word, steps_input], outputs=output_log)
    folder_btn.click(fn=open_output_folder, inputs=[trigger_word], outputs=output_log)
    grid_btn.click(fn=make_checkpoint_grid, inputs=[trigger_word], outputs=[grid_image, output_log])

if __name__ == "__main__":
     ui.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
