"""End to end: the ToriiGate captioner's web UI (tools/toriigate-captioner/captioner_ui.py) served by Gradio and driven
through its API, with fake_llama_server.py standing in for llama-server — as a running server ('already running'
mode) and, on Linux/macOS, as the program the UI starts itself ('my llama-server' mode). Covers captioning a folder,
trying one image without saving, work carrying on when the page is reloaded, stop, keeping the model loaded between
runs, unloading it, and the settings surviving a restart."""
import importlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent))
from _harness import DATA, REPO, check, finish  # noqa: E402

TOOL = REPO / "tools" / "toriigate-captioner"
sys.path.insert(0, str(TOOL))
import captioner_ui  # noqa: E402  (only for its list of inputs; the UI under test runs in its own process)

FAKE = H / "fake_llama_server.py"
WORK = Path(tempfile.mkdtemp(prefix="tf_tgui_"))
DS = WORK / "ds"
shutil.copytree(DATA / "grabber", DS)
N_IMAGES = len([p for p in DS.iterdir() if p.suffix.lower() in captioner_ui.tc.IMAGE_EXTS])
SETTINGS = WORK / "settings.json"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
os.environ["TF_TEST_PORT"] = str(PORT)
ui_proc = None


def start_ui():
    global ui_proc
    env = dict(os.environ, TG_CAPTIONER_SETTINGS=str(SETTINGS))
    ui_proc = subprocess.Popen([sys.executable, str(TOOL / "captioner_ui.py"), "--port", str(PORT), "--no-browser"],
                               cwd=WORK, env=env, stdout=open(WORK / "ui.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            httpx.get(f"http://127.0.0.1:{PORT}/config", timeout=2)
            break
        except httpx.HTTPError:
            time.sleep(0.5)
    import client
    importlib.reload(client)
    return client


def fake_server(delay=0.0):
    port = free_port()
    rec = WORK / f"requests_{port}.jsonl"
    env = dict(os.environ, FAKE_LLAMA_LOAD="0", FAKE_LLAMA_DELAY=str(delay), FAKE_LLAMA_RECORD=str(rec))
    proc = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port)], env=env, stdout=subprocess.DEVNULL)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    return proc, f"http://127.0.0.1:{port}"


def values(**kw):
    v = dict(captioner_ui.DEFAULTS, folder=str(DS))
    v.update(kw)
    return [v[k] for k in captioner_ui.UI_KEYS]


def poll(client, session=None):
    _, out = client.call("poll", [-1], session)
    return {"log": out[0], "status": out[1], "preview": out[2], "caption": out[3], "query": out[4]}


def wait_idle(client, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = poll(client)
        if "⏳" not in p["status"]:
            return p
        time.sleep(0.3)
    return poll(client)


try:
    client = start_ui()
    cfg = client.cfg
    check("captioner UI is up", ui_proc.poll() is None and "dependencies" in cfg)
    apis = {d.get("api_name") for d in cfg["dependencies"]}
    check("API: start, try_one, download, stop, unload, list_files, poll",
          {"start", "try_one", "download", "stop", "unload", "list_files", "poll"} <= apis, sorted(a for a in apis if a))

    srv, url = fake_server()
    try:
        client.call("start", values(mode="external", url=url))
        p = wait_idle(client)
        written = sorted(DS.glob("*.toriigate.txt"))
        check("▶ Caption folder: every image captioned", len(written) == N_IMAGES and "Captioning finished" in p["status"]
              and f"🏁 Captioned {N_IMAGES}," in p["log"], (len(written), p["status"], p["log"][-300:]))
        check("…the last caption, its prompt and its image are shown", p["caption"].startswith("A caption for a")
              and p["query"].startswith("# Captioning format:") and p["preview"] is not None, (p["caption"], p["query"][:40]))
        saved = json.loads(SETTINGS.read_text(encoding="utf-8"))
        check("settings saved when a job starts", saved["folder"] == str(DS) and saved["url"] == url and saved["mode"] == "external")

        before = (DS / "a03.toriigate.txt").read_text(encoding="utf-8")
        (DS / "a03.toriigate.txt").unlink()
        client.call("try_one", values(mode="external", url=url, try_image="A03.JPG", characters="mychar"))
        p = wait_idle(client)
        check("🔍 Try one image: named image (any letter case), result shown, nothing written",
              p["log"].splitlines()[-1].startswith("🔍 a03.jpg") and "not saved" in p["log"].splitlines()[-1] and p["caption"].startswith("A caption")
              and "make sure to use them: [mychar]" in p["query"] and not (DS / "a03.toriigate.txt").exists(), p["log"][-300:])
        client.call("try_one", values(mode="external", url=url))
        p = wait_idle(client)
        check("…with no name: the first image without a caption", "🔍 a03.jpg" in p["log"], p["log"][-200:])
        (DS / "a03.toriigate.txt").write_text(before, encoding="utf-8")
        client.call("try_one", values(mode="external", url=url, try_image="nope.png"))
        time.sleep(0.5)
        check("…an image that isn't there is reported", "nope.png isn't an image in this folder" in poll(client)["log"])
        client.call("start", values(mode="external", url=url))
        p = wait_idle(client)
        check("re-run with overwrite off skips everything", f"0 to caption, {N_IMAGES} already have .toriigate.txt" in p["log"], p["log"][-200:])
    finally:
        srv.kill()

    srv, url = fake_server(delay=0.6)
    try:
        client.call("start", values(mode="external", url=url, overwrite=True))
        time.sleep(1.5)
        client.call("start", values(mode="external", url=url, overwrite=True))
        p = poll(client, session="another-tab")      # a reloaded page is a new session
        check("a second start while working is refused", "Still working" in p["log"].splitlines()[-1], p["log"].splitlines()[-1])
        check("reloading the page: the run carries on and shows its progress", "⏳ Captioning" in p["status"]
              and f"/{N_IMAGES}" in p["status"], p["status"])
        client.call("stop", [])
        p = wait_idle(client)
        n_done = int(p["log"].rsplit("🏁 Captioned ", 1)[1].split(",")[0])
        check("⏹ Stop: ends after the current image", "Captioning stopped" in p["status"] and 1 <= n_done < N_IMAGES
              and "⏹ Stopped." in p["log"] and "min left" not in p["status"], (p["status"], n_done))
    finally:
        srv.kill()

    client.call("start", values(mode="external", url=f"http://127.0.0.1:{free_port()}", overwrite=True))
    p = wait_idle(client)
    check("unreachable server: the run fails with the reason", "Captioning failed: can't reach" in p["status"], p["status"])

    if os.name != "nt":
        # 'my llama-server': the UI starts the program itself. A wrapper notes each start, then becomes the fake server.
        starts = WORK / "starts.txt"
        wrapper = WORK / "llama-server"
        wrapper.write_text(f'#!/bin/sh\necho started >> "{starts}"\nFAKE_LLAMA_LOAD=0.5 exec "{sys.executable}" "{FAKE}" "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        for f in ("m.gguf", "mm.gguf"):
            (WORK / f).write_bytes(b"GGUF")
        local = dict(mode="local", llama_server=str(wrapper), model=str(WORK / "m.gguf"), mmproj=str(WORK / "mm.gguf"))

        def n_starts():
            return len(starts.read_text().splitlines()) if starts.exists() else 0

        def server_procs():
            import psutil
            return [p for p in psutil.process_iter(["cmdline"]) if str(FAKE) in " ".join(p.info["cmdline"] or [])
                    and "-m" in (p.info["cmdline"] or [])]

        client.call("try_one", values(**local, keep_running=True, try_image="a01.jpg"))
        p = wait_idle(client)
        client.call("try_one", values(**local, keep_running=True, try_image="a02.jpg", format="long"))
        p = wait_idle(client)
        check("keep the model loaded: two tries, one llama-server start", n_starts() == 1 and "🔍 a02.jpg" in p["log"]
              and len(server_procs()) == 1, (n_starts(), p["log"][-300:]))
        check("…started with the model, projector and port it was given",
              any(str(WORK / "m.gguf") in " ".join(pr.info["cmdline"]) and "--mmproj" in pr.info["cmdline"] for pr in server_procs()))
        client.call("unload", [])
        time.sleep(1)
        check("⏏ Unload model: the server is stopped", "⏏ Model unloaded" in poll(client)["log"] and not server_procs())
        client.call("start", values(**local, keep_running=False, overwrite=True))
        p = wait_idle(client)
        check("without keep-loaded: loads for the run, unloads after", n_starts() == 2 and "Captioning finished" in p["status"]
              and p["log"].rstrip().endswith("⏏ Model unloaded") and not server_procs(), (n_starts(), p["log"][-200:]))
        client.call("start", values(**dict(local, llama_server=str(WORK / "missing")), keep_running=False, overwrite=True))
        p = wait_idle(client)
        check("a llama-server path that doesn't exist is reported", "llama-server not found" in p["status"], p["status"])

    ui_proc.kill()
    ui_proc.wait()
    client = start_ui()
    comp = {c.get("props", {}).get("label"): c.get("props", {}).get("value") for c in client.cfg["components"]}
    check("settings come back after a restart", comp.get("Image folder") == str(DS), comp.get("Image folder"))
finally:
    if ui_proc is not None:
        ui_proc.kill()

finish()
