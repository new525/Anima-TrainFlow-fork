"""Stand-in for llama.cpp's llama-server (the parts the ToriiGate captioner uses): /health answers 503 while
"loading", then 200; /v1/chat/completions checks the request has the shape llama-server's multimodal API takes and
answers with a caption describing what it received. Every request is appended to FAKE_LLAMA_RECORD as JSON.

FAKE_LLAMA_MODE: normal | reasoning (answer only in reasoning_content) | think (answer wrapped in <think>…</think>)
                 | length (finish_reason "length") | error (HTTP 500) | crash_after:N (exit after N requests)
                 | die_loading (exit while loading)
FAKE_LLAMA_LOAD: seconds of "loading" before /health says 200 (default 1).
FAKE_LLAMA_DELAY: seconds each caption takes (default 0)."""
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else 8080
MODE = os.environ.get("FAKE_LLAMA_MODE", "normal")
RECORD = os.environ.get("FAKE_LLAMA_RECORD")
READY_AT = time.time() + float(os.environ.get("FAKE_LLAMA_LOAD", "1"))
DELAY = float(os.environ.get("FAKE_LLAMA_DELAY", "0"))
print("fake llama-server argv:", json.dumps(args), flush=True)
if MODE == "die_loading":
    print("error: failed to load model (fake)", flush=True)
    sys.exit(1)
served = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            if time.time() < READY_AT:
                return self.reply(503, {"error": {"message": "Loading model"}})
            return self.reply(200, {"status": "ok"})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self.reply(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = req["messages"]
        part_img, part_txt = user["content"]
        assert system["role"] == "system" and system["content"][0]["type"] == "text"
        assert part_img["type"] == "image_url" and part_txt["type"] == "text"
        url = part_img["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        info = {"system": system["content"][0]["text"], "query": part_txt["text"], "image_size": im.size,
                "image_format": im.format, "image_mode": im.mode, "pixel00": im.convert("RGB").getpixel((0, 0)),
                "temperature": req.get("temperature"), "max_tokens": req.get("max_tokens"), "seed": req.get("seed")}
        if RECORD:
            with open(RECORD, "a", encoding="utf-8") as f:
                f.write(json.dumps(info) + "\n")
        served["n"] += 1
        if MODE.startswith("crash_after:") and served["n"] > int(MODE.split(":")[1]):
            os._exit(3)
        time.sleep(DELAY)
        if MODE == "error":
            return self.reply(500, {"error": {"message": "fake failure"}})
        text = f"A caption for a {im.size[0]}x{im.size[1]} image."
        msg = {"role": "assistant", "content": text}
        if MODE == "reasoning":
            msg = {"role": "assistant", "content": "", "reasoning_content": text}
        elif MODE == "think":
            msg = {"role": "assistant", "content": f"<think>pondering</think>\n{text}"}
        finish = "length" if MODE == "length" else "stop"
        self.reply(200, {"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                         "usage": {"prompt_tokens": 1100, "completion_tokens": 12, "total_tokens": 1112}})


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
