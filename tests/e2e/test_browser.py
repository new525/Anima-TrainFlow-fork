"""Optional UI check in a headless browser (see tests/browser): starts the app with a fake training run in progress,
then runs ui_check.mjs against it. Skipped when Node.js or the browser packages aren't installed."""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import psutil
from PIL import Image

H = Path(__file__).resolve().parent
BROWSER = H.parent / "browser"
sys.path.insert(0, str(H.parent))
from _harness import APP, check, finish  # noqa: E402

if not shutil.which("node") or not (BROWSER / "node_modules" / "puppeteer-core").exists():
    print("SKIP: needs Node.js and `npm install` in tests/browser (plus CHROME_PATH on Windows/macOS)")
    sys.exit(0)

work = Path(tempfile.mkdtemp(prefix="tf_ui_"))
tree, ds = work / "apptree", work / "ds"
for d in (tree / "models", tree / "training" / "sd-scripts", ds):
    d.mkdir(parents=True)
shutil.copy(APP, tree / "app.py")
for f in ("dit", "qwen", "vae"):
    (tree / "models" / f"{f}.safetensors").touch()
for i in range(3):
    Image.new("RGB", (512, 768), (40 * i, 80, 120)).save(ds / f"img{i}.png"); (ds / f"img{i}.txt").write_text("1girl")
with socket.socket() as s_:
    s_.bind(("127.0.0.1", 0)); port = s_.getsockname()[1]
os.environ["TF_TEST_PORT"] = str(port)
env = dict(os.environ, FAKE_SIDE_LOG=str(work / "side.log"), FAKE_DELAY="0.4")
server = subprocess.Popen([sys.executable, str(H / "run_app.py"), str(tree / "app.py"), str(port)], cwd=work, env=env,
                          stdout=open(work / "server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        try:
            httpx.get(f"http://127.0.0.1:{port}/config", timeout=2); break
        except Exception:
            time.sleep(0.5)
    import client
    m = tree / "models"
    client.call("start_training", ["kw", str(ds), str(m / "dit.safetensors"), str(m / "qwen.safetensors"), str(m / "vae.safetensors"),
                                   32, "1.0", "Prodigy", 2000, 100, 5, "a\nb", "neg", 1024, 1024, 30, 4.0, 42, 42, 1, 1, True, 0, 0])
    time.sleep(8)   # some steps, previews and loss points
    r = subprocess.run(["node", str(BROWSER / "ui_check.mjs"), f"http://127.0.0.1:{port}/", str(work / "ui.png")],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout.strip())
    if r.returncode not in (0, 1):
        print(r.stderr[-2000:])
    check("browser UI check", r.returncode == 0, f"screenshot {work / 'ui.png'}")
finally:
    server.kill(); server.wait()
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if any("fake_trainer.py" in part for part in (proc.info["cmdline"] or [])):
                proc.kill()
        except psutil.Error:
            pass
finish()
