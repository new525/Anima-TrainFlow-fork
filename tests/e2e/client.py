"""Minimal client for the app's Gradio queue API (one call = join + read the event stream to completion)."""
import json
import os
import uuid

import httpx

URL = f"http://127.0.0.1:{os.environ.get('TF_TEST_PORT', '7864')}"
cfg = httpx.get(f"{URL}/config").json()


def fn_index(name):
    return next(i for i, d in enumerate(cfg["dependencies"]) if d.get("api_name") == name)


def call(name, data, session=None):
    session = session or uuid.uuid4().hex[:10]
    r = httpx.post(f"{URL}/gradio_api/queue/join", json={"data": data, "fn_index": fn_index(name), "session_hash": session,
                                                          "event_data": None, "trigger_id": None})
    r.raise_for_status()
    with httpx.stream("GET", f"{URL}/gradio_api/queue/data", params={"session_hash": session}, timeout=60) as s:
        for line in s.iter_lines():
            if not line.startswith("data:"):
                continue
            msg = json.loads(line[5:])
            if msg.get("msg") == "process_completed":
                if not msg.get("success", True):
                    raise RuntimeError(f"{name} failed: {msg.get('output')}")
                return session, msg.get("output", {}).get("data")
