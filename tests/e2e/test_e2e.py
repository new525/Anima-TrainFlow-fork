"""End to end: the real app.py served by Gradio, driven through its API, with fake_trainer.py standing in for
sd-scripts. Covers the trainer outliving the app (kill + restart + reattach), pause/halt/resume across restarts, block
swap and multi-prompt previews in the config, checkpoint grid, loss history, Train More, background tools, the
inspector, the trigger prefix, and the training/tool interlock. The app tree lives in tests/e2e/_work (outside the
server's working folder and the temp folder, so preview/grid serving depends on allowed_paths like a real install)."""
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import psutil
import toml
from PIL import Image

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent))
from _harness import APP as APP_SRC, check, finish  # noqa: E402

WORK = H / "_work"
TREE = WORK / "apptree"; M = TREE / "models"; OUT = TREE / "training" / "output"; DS = WORK / "ds"
KW = "kw"
shutil.rmtree(WORK, ignore_errors=True)
for d in (M, TREE / "training" / "sd-scripts", DS):
    d.mkdir(parents=True)
shutil.copy(APP_SRC, TREE / "app.py")
for f in ("dit", "qwen", "vae"):
    (M / f"{f}.safetensors").touch()
for i in range(3):
    Image.new("RGB", (512, 768)).save(DS / f"img{i}.png"); (DS / f"img{i}.txt").write_text("1girl, solo, smile")
with socket.socket() as s_:
    s_.bind(("127.0.0.1", 0)); PORT = s_.getsockname()[1]
os.environ["TF_TEST_PORT"] = str(PORT)
SERVER_CWD = Path(tempfile.mkdtemp(prefix="tf_cwd_"))
server = None


def start_server():
    global server
    env = dict(os.environ, FAKE_SIDE_LOG=str(WORK / "side.log"))
    server = subprocess.Popen([sys.executable, str(H / "run_app.py"), str(TREE / "app.py"), str(PORT)], cwd=SERVER_CWD, env=env,
                              stdout=open(WORK / "server.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            httpx.get(f"http://127.0.0.1:{PORT}/config", timeout=2); break
        except Exception:
            time.sleep(0.5)
    import client
    importlib.reload(client)
    return client.call


def kill_server():
    server.kill(); server.wait()   # like the console window being closed / the app crashing


def args(steps=300, blocks=0, prompts="first prompt\nsecond prompt", kw=KW):
    return [kw, str(DS), str(M / "dit.safetensors"), str(M / "qwen.safetensors"), str(M / "vae.safetensors"),
            32, "1.0", "Prodigy", steps, 100, 50, prompts, "neg", 1024, 1024, 30, 4.0, 42, 42, 1, 1, True, blocks, 0]


def steps_written():
    p = WORK / "side.log"
    return len(p.read_text().splitlines()) if p.exists() else 0


def poll(call):
    _, p = call("poll_ui", [None])
    return p


def wait_status(call, word, t=30):
    for _ in range(int(t / 0.5)):
        p = poll(call)
        if word in p[2]:
            return p
        time.sleep(0.5)
    return p


try:
    call = start_server()
    # --- launch with block swap + 2 prompts
    call("start_training", args(steps=600, blocks=4)); time.sleep(2)
    run_dir = sorted((OUT / KW).iterdir())[0]
    import toml
    cfg = toml.load(str(next((run_dir / "configs").glob("*_training.toml"))))
    check("block swap written to config", cfg.get("blocks_to_swap") == 4, cfg.get("blocks_to_swap"))
    prompt_lines = next((run_dir / "configs").glob("*_prompts.txt")).read_text(encoding="utf-8").splitlines()
    check("one preview prompt per line, trigger added", len(prompt_lines) == 2 and all(l.startswith("kw, ") for l in prompt_lines), prompt_lines)

    # --- app dies mid-run; trainer must keep going
    time.sleep(2)
    kill_server()
    a = steps_written(); time.sleep(3); b = steps_written()
    check("trainer keeps training with the app dead", b > a + 20, (a, b))
    check("trainer.json marker present", (run_dir / "trainer.json").exists())

    # --- app comes back and reattaches
    call = start_server()
    p = poll(call)
    check("restarted app reattaches", "Training" in p[2] and "kw/" in p[2] and "Reattached" in p[0], p[2])
    s1 = int(p[2].split("step ")[1].split("/")[0]); time.sleep(2)
    p = poll(call); s2 = int(p[2].split("step ")[1].split("/")[0])
    check("reattached status follows new steps", s2 > s1, (s1, s2))
    df = p[4]
    n_loss = len(df["value"]["data"]) if isinstance(df, dict) and "value" in df else (len(df["data"]) if isinstance(df, dict) and "data" in df else -1)
    check("loss chart has history from before the restart", n_loss >= s1 - 5, (n_loss, s1))

    # --- pause, app dies while paused, restart shows paused, continue works
    call("toggle_pause", []); time.sleep(0.5)
    kill_server()
    a = steps_written(); time.sleep(2); b = steps_written()
    check("paused trainer stays paused with app dead", a == b, (a, b))
    call = start_server()
    p = poll(call)
    check("reattach restores paused state", "Paused" in p[2] and p[3]["value"] == "▶ Continue", (p[2], p[3]))
    call("toggle_pause", []); time.sleep(2)
    check("continue after reattach", steps_written() > b)

    # --- halt from the reattached app, resume, finish
    call("halt_training", [])
    p = wait_status(call, "Halted")
    check("halt works on a reattached trainer", "Halted" in p[2], p[2])
    check("marker removed after exit", not (run_dir / "trainer.json").exists())
    call("resume_training", [KW])
    p = wait_status(call, "Finished", 40)
    check("resumed run finishes", "Finished" in p[2], p[2])

    # --- checkpoint grid
    _, out = call("make_checkpoint_grid", [KW])
    grid_path = run_dir / "checkpoint_grid.png"
    if grid_path.exists():
        g = Image.open(grid_path); print("   grid size", g.size)
    check("grid built: 12 steps x 2 prompts", grid_path.exists() and "12 step(s) × 2 prompt(s)" in out[1], out[1].splitlines()[-1])
    check("grid image served (the app tree is outside the server's working folder and temp)",
          isinstance(out[0], dict) and "/gradio_api/file=" in (out[0].get("url") or ""), out[0])
    p = poll(call)
    caps = [e.get("caption") or "" for e in p[1]] if isinstance(p[1], list) else []
    check("gallery captions name step and prompt", any(c.startswith("step 600 · prompt 2") for c in caps), caps[:3])

    # --- Train More 600 -> 700
    _, out = call("extend_training", [KW, 550]); check("Train More refuses fewer steps", "above 600" in out[0].splitlines()[-1], out[0].splitlines()[-1])
    _, out = call("extend_training", [KW, 700])
    print("   " + " | ".join(out[0].splitlines()[:2]))
    for _ in range(40):
        time.sleep(0.25)
        argv = json.loads((WORK / "last_argv.json").read_text())
        if "--resume" in argv and argv[argv.index("--resume") + 1].endswith("kw-state"): break
    check("Train More resumes from end-of-run state", argv[argv.index("--resume") + 1].endswith("kw-state"), argv[-3:])
    p = wait_status(call, "Finished", 40)
    check("extended run finishes at 700", "Finished" in p[2] and "700/700" in p[2], p[2])
    check("old final LoRA kept", (run_dir / "kw-step00000600-final.safetensors").exists(), sorted(x.name for x in run_dir.glob("*.safetensors")))
    check("config backup kept", (run_dir / "configs" / "kw_training.before_extend_600.toml").exists())

    # --- background tools + inspector
    (DS / "img0.txt").write_text("kw, 1girl, solo, solo")    # trigger in 1 of 3 captions + duplicate tag
    (DS / "orphan.txt").write_text("x")
    Image.new("RGB", (300, 300)).save(DS / "small.bmp")       # no caption, short side < 512, .bmp
    t0 = time.time(); _, out = call("start_inspect", [str(DS), KW]); dt = time.time() - t0
    time.sleep(1.5); p = poll(call); log = p[0]
    check("inspector reports missing caption (.bmp counted)", "without a caption: small.bmp" in log)
    check("inspector reports orphan caption", "caption file(s) with no image: orphan.txt" in log)
    check("inspector reports partial trigger", "1/3 captions already contain 'kw'" in log)
    check("inspector reports repeated tag", "repeat a tag: img0.txt" in log)
    check("inspector reports tag frequency", "(100%)  1girl" in log or "100%)  1girl" in log)
    (DS / "orphan.txt").unlink(); (DS / "small.bmp").unlink()
    for i in range(3): (DS / f"img{i}.txt").write_text("kw2, 1girl, smile")
    call("start_training", args(steps=100, kw="kw2")); time.sleep(1)
    run2 = sorted((OUT / "kw2").iterdir())[0]
    ds_cfg = toml.load(str(next((run2 / "configs").glob("*_dataset.toml"))))
    check("no prefix when every caption already has the trigger",
          "caption_prefix" not in ds_cfg["datasets"][0]["subsets"][0], ds_cfg["datasets"][0]["subsets"][0].get("caption_prefix"))
    call("stop_training", []); time.sleep(1)

    big = Image.fromarray(np.full((1500, 1000, 3), 90, np.uint8))
    for i in range(30): big.save(DS / f"big{i}.jpg")
    t0 = time.time(); _, out = call("start_bucketing", [str(DS), 512, 768, True, True]); dt = time.time() - t0
    p = poll(call)
    check("bucketing returns immediately (background job)", dt < 1.5, f"{dt:.2f}s")
    _, out = call("start_training", args(steps=100, kw="kw3"))
    check("training refused while bucketing works on its dataset", "still running on this dataset" in out[0].splitlines()[-1]
          and not (OUT / "kw3").exists(), out[0].splitlines()[-1])
    second = call("start_tagging", [str(DS), 0.35, 0.85, False, True])[1][0].splitlines()[-1]
    check("second tool refused while one runs", "still running" in second, second)
    for _ in range(60):
        time.sleep(0.5); p = poll(call)
        if "Total Dataset Size" in p[0]: break
    check("bucketing completes in background", "Total Dataset Size" in p[0], p[0].splitlines()[-1])
finally:
    try:
        kill_server()
    except Exception:
        pass
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if any("fake_trainer.py" in part for part in (proc.info["cmdline"] or [])):
                proc.kill()
        except psutil.Error:
            pass
finish()
