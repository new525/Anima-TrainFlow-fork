"""Optional: the ToriiGate captioner against a real llama.cpp llama-server. Runs only when these are set:
  TG_LLAMA_SERVER  path to llama-server(.exe)
  TG_MODEL         a ToriiGate-0.5 .gguf (or any Qwen3.5-VL-architecture model, e.g. a tiny one for a quick check)
  TG_MMPROJ        its vision projector (mmproj .gguf)
  TG_GPU_LAYERS    optional (default "all"; "0" for CPU only)
With the real model it also checks the caption reads like one (TG_REAL_MODEL=1)."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

from _harness import DATA, REPO, check, finish, sample_images

need = [k for k in ("TG_LLAMA_SERVER", "TG_MODEL", "TG_MMPROJ") if not os.environ.get(k)]
if need:
    print(f"SKIP needs {', '.join(need)} (a real llama-server and model)")
    sys.exit(0)
sys.path.insert(0, str(REPO / "tools" / "toriigate-captioner"))
import toriigate_captioner as tc  # noqa: E402

d = Path(tempfile.mkdtemp(prefix="tf_tgreal_")) / "ds"
shutil.copytree(DATA / "grabber", d)
Image.open(sample_images() / "astronaut.png").convert("RGB").resize((1600, 1200)).save(d / "big.png")
log = []
cfg = tc.ServerConfig(mode="local", llama_server=os.environ["TG_LLAMA_SERVER"], model=os.environ["TG_MODEL"],
                      mmproj=os.environ["TG_MMPROJ"], gpu_layers=os.environ.get("TG_GPU_LAYERS", "all"), ctx=4096)
session = tc.Session(cfg, lambda e: log.append(e.message))
real = os.environ.get("TG_REAL_MODEL") == "1"
opts = tc.Options(max_tokens=512 if real else 24, seed=1)
try:
    url = session.prepare()
    check("llama-server starts with the flags the captioner passes and becomes ready", url.startswith("http://127.0.0.1:")
          and session.server.alive(), log[-3:])
    index = tc.EmbeddedTagIndex(d)
    res = session.caption_one(d / "big.png", opts, index, write=False)
    w, h = res["size"]
    # Qwen3.5's vision encoder: 16 px patches merged 2x2 -> one token per 32x32 px
    check("the image reaches the model (prompt tokens include ~1 per 32x32 px of the 1 MP image)",
          res["usage"].get("prompt_tokens", 0) > (w // 32) * (h // 32) * 0.8, (res["usage"], res["size"]))
    check("reasoning off: an answer in content, no thinking block", res["text"] and "<think>" not in res["text"]
          and res["finish_reason"] in ("stop", "length"), (res["text"][:200], res["finish_reason"]))
    summary = session.caption_folder(d, opts, limit=3)
    written = sorted(d.glob("*" + tc.NL_CAPTION_SUFFIX))
    check("folder run writes the captions", summary["captioned"] == 3 and len(written) == 3
          and all(p.read_text(encoding="utf-8").strip() for p in written), summary)
    if real:
        text = res["text"]
        check("real model: an English description of the astronaut photo", len(text.split()) > 15
              and any(w in text.lower() for w in ("astronaut", "space", "suit", "flag")), text[:300])
        q = session.caption_one(d / "a03.jpg", opts, index, write=False)
        check("real model: uses the character name it was given", "miku" in q["text"].lower(), q["text"][:300])
finally:
    proc = session.server.proc if session.server else None
    session.close(force=True)
check("llama-server stopped afterwards", proc is None or proc.poll() is not None)
finish()
