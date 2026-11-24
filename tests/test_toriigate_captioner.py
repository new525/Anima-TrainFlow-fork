"""tools/toriigate-captioner: prompts (the model's own prompts.py), grounding from Grabber tags and .txt captions,
image preparation, the llama-server lifecycle and captioning loop against a stand-in server, downloads, the CLI, and
how TrainFlow's tools treat <name>.toriigate.txt."""
import hashlib
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import httpx
from PIL import Image

from _harness import DATA, REPO, TESTS, bucket, check, drain, finish, load_app, sample_images, tmpdir

TOOL = REPO / "tools" / "toriigate-captioner"
sys.path.insert(0, str(TOOL))
import toriigate_captioner as tc  # noqa: E402

FAKE = TESTS / "e2e" / "fake_llama_server.py"
OFFICIAL_PROMPTS_SHA256 = "7980d5c839e80c54bf32"  # first 20 hex digits of the file as published in Minthy/ToriiGate-0.5


def grabber():
    d = Path(tempfile.mkdtemp(prefix="tf_tg_")) / "ds"
    shutil.copytree(DATA / "grabber", d)
    return d


# --- the bundled prompts are the model's own, unmodified ---------------------------------------------------------------
check("prompts.py is the official file, unmodified",
      hashlib.sha256((TOOL / "prompts.py").read_bytes()).hexdigest().startswith(OFFICIAL_PROMPTS_SHA256))
check("every official format is offered", tc.FORMATS == list(tc.tg.prompts_b) and set(tc.FORMAT_NOTES) == set(tc.FORMATS))

# --- grounding + prompts -----------------------------------------------------------------------------------------------
d = grabber()
index = tc.EmbeddedTagIndex(d)
opts = tc.Options()
item, src = tc.grounding(d / "a03.jpg", index, opts)
check("namespaced Grabber tags: character found, artist left out, series and general kept",
      item["characters"] == ["hatsune_miku"] and "kantoku" not in " ".join(item["tags"]) and "vocaloid" in item["tags"]
      and "1girl" in item["tags"] and src.startswith("embedded tags"), (item, src))
q, note = tc.build_query(item, opts, "a03.jpg")
check("…and the prompt names the character the official way",
      "Here are names/tags for characters from the picture, make sure to use them: [hatsune miku]." in q, q)
check("…tags passed plain (underscores to spaces, comma-separated)", "# Booru tags for the image\n[" in q and "long hair" in q)
check("same image, same prompt (prompts.py shuffles tags; seeded by file name)",
      all(tc.build_query(item, opts, "a03.jpg")[0] == q for _ in range(5)))
item_b, _ = tc.grounding(d / "a03.jpg", index, tc.Options(tag_style="booru"))
qb, _ = tc.build_query(item_b, tc.Options(tag_style="booru"), "a03.jpg")
check("booru style: underscores kept, space-separated, as prompts.py does without underscores_replace",
      "long_hair" in qb and ", " not in qb.split("# Booru tags for the image\n[")[1].split("]")[0]
      and "[hatsune_miku]" in qb, qb)
item_a, _ = tc.grounding(d / "a03.jpg", index, tc.Options(include_artists=True))
check("artist tags only on request", "kantoku" in item_a["tags"])
item10, _ = tc.grounding(d / "a10.jpg", index, opts)
check("score tags aren't passed as hints", item10["tags"] == ["1girl", "smile"], item10)
item12, _ = tc.grounding(d / "a12.jpg", index, tc.Options(tag_style="booru"))
check("kaomoji kept as they are", "^_^" in item12["tags"] and "@_@" in item12["tags"], item12)

q1, _ = tc.build_query(tc.grounding(d / "a01.jpg", index, opts)[0], opts, "a01.jpg")
check("no names known, mode 'known': the model is told not to guess", "Avoid to guess names for characters." in q1, q1)
q2, _ = tc.build_query(tc.grounding(d / "a01.jpg", index, tc.Options(names="recognize"))[0], tc.Options(names="recognize"), "a01.jpg")
check("mode 'recognize': the model is asked to recognise them", "Try to recognize the characters in the picture" in q2)
q3, _ = tc.build_query(item, tc.Options(names="none"), "a03.jpg")
check("mode 'none': names withheld even when known", "Avoid to guess names" in q3 and "hatsune" not in q3.split("# Characters")[1])
q4, note4 = tc.build_query(tc.grounding(d / "a01.jpg", index, tc.Options(format="long_thoughts", names="none"))[0],
                           tc.Options(format="long_thoughts", names="none"), "a01.jpg")
check("names-only format with names off: falls back to recognising, with a note (model card: those formats need names)",
      "Try to recognize" in q4 and "needs named characters" in note4, note4)
given = tc.Options(characters="mychar, Other Person")
item_g, src_g = tc.grounding(d / "a03.jpg", index, given)
check("names typed in replace the found ones", item_g["characters"] == ["mychar", "Other Person"] and "names given" in src_g)
check("…booru style turns their spaces into underscores", tc.grounding(d / "a03.jpg", index, tc.Options(characters="Other Person", tag_style="booru"))[0]["characters"] == ["Other_Person"])
item11, src11 = tc.grounding(d / "a11.jpg", index, opts)
q11, _ = tc.build_query(item11, opts, "a11.jpg")
check("image without tags: no empty tag section", item11["tags"] == [] and "# Booru tags" not in q11 and src11 == "", q11)
(d / "a11.txt").write_text("1girl, @some artist, blue sky, score_9\nsecond line", encoding="utf-8")
item11, src11 = tc.grounding(d / "a11.jpg", index, opts)
check("falls back to the image's .txt caption (first line; @artist and score tags left out)",
      item11["tags"] == ["1girl", "blue sky"] and src11 == "a11.txt", (item11, src11))
check("query text matches prompts.make_user_query for the same inputs", q.startswith("# Captioning format:\n" + tc.tg.prompts_b["short"]))

# --- when an image has no hints, the tool says what it looked at ----------------------------------------------------------
from PIL import PngImagePlugin  # noqa: E402

XMP = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       '<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">{}</rdf:Description></rdf:RDF></x:xmpmeta>')
SUBJECT = "<dc:subject><rdf:Bag><rdf:li>character:hatsune_miku</rdf:li><rdf:li>general:1girl</rdf:li></rdf:Bag></dc:subject>"


def png(path, xmp=None, compressed=False, **text_chunks):
    info = PngImagePlugin.PngInfo()
    if xmp is not None:
        info.add_itxt("XML:com.adobe.xmp", xmp, zip=compressed)
    for key, text in text_chunks.items():
        info.add_text(key, text)
    Image.new("RGB", (64, 64), (len(path.name) * 9 % 255, 60, 90)).save(path, pnginfo=info)


nohints = tmpdir()
png(nohints / "zipped.png", xmp=XMP.format(SUBJECT), compressed=True)
png(nohints / "described.png", xmp=XMP.format("<dc:description>1girl, hatsune_miku</dc:description>"))
png(nohints / "generated.png", parameters="masterpiece, 1girl\nSteps: 20")
png(nohints / "bare.png")
png(nohints / "captioned.png")
(nohints / "captioned.txt").write_text("1girl, smile", encoding="utf-8")
(nohints / "original_images").mkdir()
png(nohints / "bucketed.png")
Image.new("RGB", (64, 64)).save(nohints / "original_images" / "bucketed.jpg")
nindex = tc.EmbeddedTagIndex(nohints)
item_z, src_z = tc.grounding(nohints / "zipped.png", nindex, tc.Options())
check("XMP stored compressed inside a PNG is read (the byte scan alone can't see it)",
      item_z["characters"] == ["hatsune_miku"] and item_z["tags"] == ["1girl"] and src_z == "embedded tags (XMP)", (item_z, src_z))
why = {f: tc.why_no_hints(nohints / f, nindex) for f in ("bare.png", "described.png", "generated.png", "bucketed.png")}
check("no hints, no metadata: said so, with the files looked at", "bare.png: no metadata" in why["bare.png"]
      and "no other file of that name here or in original_images/" in why["bare.png"] and "no .xmp sidecar" in why["bare.png"]
      and "no bare.txt" in why["bare.png"], why["bare.png"])
check("…the original in original_images/ is named when there is one", "original_images/bucketed.jpg: no metadata" in why["bucketed.png"], why["bucketed.png"])
check("no hints, but metadata of another kind: the fields it does have are named",
      "XMP: dc:description" in why["described.png"] and "PNG text: parameters" in why["generated.png"], (why["described.png"], why["generated.png"]))
check("…and nothing is read from those fields (a description or generation settings aren't a tag list)",
      nindex.items_for(nohints / "described.png") == ([], "") and nindex.items_for(nohints / "generated.png") == ([], ""))
report = "\n".join(tc.hint_report(nohints, tc.Options()))
check("hint report: counts by where the hints come from", "6 image(s)" in report and "1 from embedded tags, 1 from .txt captions, 4 with none" in report, report.splitlines()[0])
check("…images without metadata and images with other metadata are listed apart, with what was found",
      "2 with no metadata at all" in report and "bare.png, bucketed.png" in report and "2 with metadata, but no tags where they are read from" in report
      and "described.png: XMP: dc:description" in report and "generated.png: PNG text: parameters" in report, report)
check("…and it says where tags are read from", "XMP Subject/Keywords, IPTC Keywords or Windows keywords" in report)
untagged = tmpdir()
for i in range(3):
    png(untagged / f"{i}.png")
report_u = "\n".join(tc.hint_report(untagged, tc.Options()))
check("a folder with no metadata anywhere: the report points at the downloader (exiftool, extension list)",
      "0 from embedded tags, 0 from .txt captions, 3 with none" in report_u and "exiftool" in report_u and "jpg jpeg png gif mp4" in report_u, report_u)
check("…a tagged folder gets no such advice", "exiftool" not in "\n".join(tc.hint_report(grabber(), tc.Options())))
check("names typed in are counted for images that have nothing else", "still get the names you typed in" in "\n".join(tc.hint_report(untagged, tc.Options(characters="mychar"))))
missing_reader, tc.EmbeddedTagIndex = tc.EmbeddedTagIndex, None
try:
    check("without embedded_tags.py (captioner copied out of TrainFlow) the report says tags can't be read",
          "embedded_tags.py isn't next to the captioner" in "\n".join(tc.hint_report(grabber(), tc.Options())))
finally:
    tc.EmbeddedTagIndex = missing_reader

# --- your own prompt -----------------------------------------------------------------------------------------------------
n = differ = 0
for fmt in tc.FORMATS:
    for names in tc.NAMES_MODES:
        for style in ("plain", "booru"):
            for use_tags in (True, False):
                for chars in ("", "mychar"):
                    for im in tc.list_images(d):
                        o = tc.Options(format=fmt, names=names, tag_style=style, use_tags=use_tags, characters=chars)
                        it, _ = tc.grounding(im, index, o)
                        official, _ = tc.build_query(it, o, im.name)
                        o.prompt = tc.default_template(fmt)
                        n += 1
                        differ += tc.build_query(it, o, im.name)[0] != official
check(f"the model's prompt loaded as a template fills in to exactly what prompts.py makes ({n} cases: every format, "
      "names mode, tag style, with/without tags and typed names, every fixture image)", n > 3000 and differ == 0, differ)
mine = tc.default_template("short").replace("Cover main objects and details.", "Cover main objects. Mention the lighting.")
qm, _ = tc.build_query(item, tc.Options(prompt=mine), "a03.jpg")
check("your prompt: your text, with this image's tag and character sections filled in where {tags} and {characters} were",
      "Mention the lighting." in qm and "# Booru tags for the image\n[" in qm and "make sure to use them: [hatsune miku]" in qm
      and "{tags}" not in qm and "{characters}" not in qm, qm)
check("…the filled-in sections are the ones the model's own prompt has for that image",
      qm.replace("Cover main objects. Mention the lighting.", "Cover main objects and details.") == q)
qr, _ = tc.build_query(item, tc.Options(prompt="{characters}One sentence, please.\n{tags}"), "a03.jpg")
check("…placeholders can go anywhere, in any order",
      qr.index("# Characters on picture:") < qr.index("One sentence") < qr.index("# Booru tags"), qr)
bare = "# Captioning format:\nOne sentence.\n"
check("…left out, that part isn't sent", tc.build_query(item, tc.Options(prompt=bare), "a03.jpg")[0] == bare)
w = tc.prompt_warnings(tc.Options(prompt=bare))
check("…and that's said before a run", len(w) == 2 and "{tags}" in w[0] and "{characters}" in w[1], w)
check("no such notes for the model's own prompt, or a template with both placeholders",
      tc.prompt_warnings(tc.Options()) == [] and tc.prompt_warnings(tc.Options(prompt=mine)) == [])
check("…nor about {tags} when tag hints are turned off", len(tc.prompt_warnings(tc.Options(prompt="{characters}", use_tags=False))) == 0)
qj, _ = tc.build_query(item, tc.Options(prompt='Answer as {"caption": "…"} {unknown} {tags}{characters}'), "a03.jpg")
check("other braces in your prompt are kept as they are (JSON examples, unknown names)",
      qj.startswith('Answer as {"caption": "…"} {unknown} # Booru tags'), qj[:80])
qn, note = tc.build_query(tc.grounding(d / "a01.jpg", index, tc.Options())[0],
                          tc.Options(format="long_thoughts", names="none", prompt="Think first.\n{tags}{characters}"), "a01.jpg")
check("with your prompt, the chosen format still decides names (a names-only format still asks to recognise them)",
      "Try to recognize the characters" in qn and "needs named characters" in note, qn)

# --- image preparation -------------------------------------------------------------------------------------------------
REAL = sample_images()
big = Image.open(REAL / "astronaut.png").convert("RGB").resize((2000, 1500))
p = Path(tempfile.mkdtemp()) / "big.png"
big.save(p)
b64, size = tc.encode_image(p)
im = Image.open(io.BytesIO(__import__("base64").b64decode(b64)))
check("downscaled to at most 1 megapixel, aspect kept, JPEG", size[0] * size[1] <= 1_000_000 and abs(size[0] / size[1] - 4 / 3) < 0.01
      and im.format == "JPEG" and im.size == size, size)
small = Path(tempfile.mkdtemp()) / "small.png"
big.resize((640, 480)).save(small)
check("small images aren't upscaled", tc.encode_image(small)[1] == (640, 480))
rot = Path(tempfile.mkdtemp()) / "phone.jpg"
exif = Image.Exif(); exif[0x0112] = 6
Image.new("RGB", (900, 600), "red").save(rot, exif=exif)
check("EXIF rotation applied (phone photo sent upright)", tc.encode_image(rot)[1] == (600, 900))
alpha = Path(tempfile.mkdtemp()) / "alpha.png"
Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(alpha)
px = Image.open(io.BytesIO(__import__("base64").b64decode(tc.encode_image(alpha)[0]))).getpixel((10, 10))
check("transparency on white", all(c > 245 for c in px), px)

# --- llama-server lifecycle (stand-in server, real process handling) ------------------------------------------------------
os.environ["FAKE_LLAMA_LOAD"] = "1.5"
srv = tc.LlamaServer([sys.executable, str(FAKE)], Path("m.gguf"), Path("mm.gguf"), log_path=Path(tempfile.mkdtemp()) / "srv.log")
cmd = srv.command()
check("llama-server command: model, projector, GPU layers, context, local host, reasoning off",
      all(x in cmd for x in ("-m", "--mmproj", "-ngl", "-c", "--host", "127.0.0.1", "--port", "--reasoning", "off", "--no-webui"))
      and cmd[cmd.index("--reasoning") + 1] == "off", cmd)
t0 = time.time()
url = srv.start(timeout=30)
check("start waits until /health says ready", time.time() - t0 >= 1.4 and httpx.get(url + "/health").status_code == 200)
proc = srv.proc
srv.stop()
check("stop ends the process", srv.proc is None and proc.poll() is not None and not srv.alive())
os.environ["FAKE_LLAMA_MODE"] = "die_loading"
srv = tc.LlamaServer([sys.executable, str(FAKE)], Path("m.gguf"), Path("mm.gguf"), log_path=Path(tempfile.mkdtemp()) / "srv.log")
try:
    srv.start(timeout=30)
    check("a server that dies while loading is reported", False)
except tc.SetupError as e:
    check("a server that dies while loading is reported with its log", "failed to load model (fake)" in str(e), str(e)[:120])
os.environ["FAKE_LLAMA_MODE"] = "normal"
os.environ["FAKE_LLAMA_LOAD"] = "60"
srv = tc.LlamaServer([sys.executable, str(FAKE)], Path("m.gguf"), Path("mm.gguf"), log_path=Path(tempfile.mkdtemp()) / "srv.log")
try:
    srv.start(timeout=2)
    check("load timeout", False)
except tc.SetupError as e:
    check("load timeout reported, process cleaned up", "wasn't ready after 2 s" in str(e) and srv.proc is None)


# --- captioning a folder through a running server ------------------------------------------------------------------------
def fake_server(mode="normal"):
    port = tc.free_port()
    rec = Path(tempfile.mkdtemp()) / "requests.jsonl"
    env = dict(os.environ, FAKE_LLAMA_MODE=mode, FAKE_LLAMA_LOAD="0", FAKE_LLAMA_RECORD=str(rec))
    proc = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port)], env=env, stdout=subprocess.DEVNULL)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    return proc, f"http://127.0.0.1:{port}", rec


def records(rec):
    return [json.loads(line) for line in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []


events = []
proc, url, rec = fake_server()
try:
    d = grabber()
    Image.open(REAL / "astronaut.png").convert("RGB").resize((1600, 1200)).save(d / "big.png")
    session = tc.Session(tc.ServerConfig(mode="external", url=url), events.append)
    summary = session.caption_folder(d, tc.Options())
    images = tc.list_images(d)
    written = sorted(p.name for p in d.glob("*" + tc.NL_CAPTION_SUFFIX))
    check("every image captioned (videos skipped), one .toriigate.txt each",
          summary["captioned"] == len(images) == 15 and len(written) == 15 and "a14.toriigate.txt" not in written, (summary, written))
    check("tag captions untouched", not list(d.glob("a0[1-9].txt")))
    r = records(rec)
    by_q = {x["query"]: x for x in r}
    big_rec = next(x for x in r if x["image_size"][0] > 1000)
    check("requests as the reference script sends them: system prompt, JPEG, image first, temperature 0.5, 2048 tokens",
          all(x["system"] == tc.tg.system_prompt and x["image_format"] == "JPEG" and x["temperature"] == 0.5
              and x["max_tokens"] == 2048 and x["seed"] is None for x in r), r[0])
    check("…image at most 1 MP", big_rec["image_size"][0] * big_rec["image_size"][1] <= 1_000_000, big_rec["image_size"])
    w, h = tc.encode_image(d / "big.png")[1]
    check("the caption file holds the model's answer", (d / "big.toriigate.txt").read_text(encoding="utf-8") == f"A caption for a {w}x{h} image.\n")
    check("grounded prompt reached the server", any("make sure to use them: [hatsune miku]" in q for q in by_q))
    log = (d / tc.LOG_NAME).read_text(encoding="utf-8")
    check("per-image log line with format, time and where the hints came from", "a03.jpg  short" in log and "hints: embedded tags (XMP)" in log, log.splitlines()[0])
    said = "\n".join(e.message for e in events if e.kind == "log")
    check("the run ends with where the hints came from", summary["hints"] == {"embedded tags": 13, ".txt": 0, "none": 2}
          and "Tag hints: 13 from embedded tags, 0 from .txt, 2 none." in said, summary.get("hints"))
    check("…and a tagged folder gets no warning about missing hints", "No tag hints for" not in said)
    events.clear()
    summary_u = session.caption_folder(untagged, tc.Options())
    said_u = [e.message for e in events if e.kind == "log"]
    check("a folder whose first images have no hints: warned before the first caption, with what the first file carries",
          summary_u["captioned"] == 3 and any("No tag hints for any of the first 3 images" in m for m in said_u)
          and any("0.png: no metadata" in m for m in said_u) and summary_u["hints"]["none"] == 3, said_u[:4])
    events.clear()
    n_before = len(records(rec))
    summary = session.caption_folder(d, tc.Options())
    check("re-run skips images that already have a caption", summary["captioned"] == 0 and summary["skipped"] == 15 and len(records(rec)) == n_before)
    summary = session.caption_folder(d, tc.Options(overwrite=True, seed=7, format="long", temperature=0.3, max_tokens=300), limit=2)
    r2 = records(rec)[n_before:]
    check("overwrite + limit + generation settings passed through", summary["captioned"] == 2 and len(r2) == 2
          and all(x["seed"] == 7 and x["temperature"] == 0.3 and x["max_tokens"] == 300 for x in r2)
          and r2[0]["query"].startswith("# Captioning format:\n" + tc.tg.prompts_b["long"]), r2[:1])
    stop = threading.Event()
    seen = []

    def emit(e):
        seen.append(e)
        if e.kind == "caption" and e.done == 3:
            stop.set()
    s2 = tc.Session(tc.ServerConfig(mode="external", url=url), emit)
    summary = s2.caption_folder(d, tc.Options(overwrite=True), stop)
    check("stop takes effect after the current image", summary["captioned"] == 3 and summary["stopped"], summary)
    check("caption events carry the image, caption and prompt", all(e.image and e.caption and e.query for e in seen if e.kind == "caption"))
finally:
    proc.kill()

proc, url, rec = fake_server()
try:
    d = grabber()
    s = tc.Session(tc.ServerConfig(mode="external", url=url)).caption_folder(
        d, tc.Options(prompt=mine, system_prompt="You are terse."), limit=3)
    r = records(rec)
    check("folder run: your prompt and system prompt are what the server gets", len(r) == 3
          and all("Mention the lighting." in x["query"] and x["system"] == "You are terse." for x in r), r[:1])
    check("…and the log says so", "short (custom prompt + custom system prompt)" in (d / tc.LOG_NAME).read_text(encoding="utf-8"))
finally:
    proc.kill()

for mode, check_fn, label in [
    ("reasoning", lambda d, s: (d / "a01.toriigate.txt").read_text(encoding="utf-8").startswith("A caption"), "answer only in reasoning_content is used"),
    ("think", lambda d, s: "pondering" not in (d / "a01.toriigate.txt").read_text(encoding="utf-8"), "<think> blocks stripped"),
    ("length", lambda d, s: "hit the token limit" in (d / tc.LOG_NAME).read_text(encoding="utf-8"), "token-limit cut-off is flagged"),
    ("error", lambda d, s: s["failed"] == 2 and s["captioned"] == 0 and not list(d.glob("*.toriigate.txt")), "HTTP errors: image counted as failed, nothing written"),
    ("crash_after:1", lambda d, s: s["captioned"] == 1 and s["failed"] == 1, "server gone mid-run: stops instead of failing every image"),
]:
    proc, url, rec = fake_server(mode)
    try:
        d = grabber()
        s = tc.Session(tc.ServerConfig(mode="external", url=url)).caption_folder(d, tc.Options(), limit=2 if mode != "crash_after:1" else 5)
        check(label, check_fn(d, s), s)
    finally:
        proc.kill()

try:
    tc.Session(tc.ServerConfig(mode="external", url=f"http://127.0.0.1:{tc.free_port()}")).prepare()
    check("unreachable external server reported", False)
except tc.SetupError as e:
    check("unreachable external server reported", "can't reach" in str(e))


# --- web UI: a button pressed during a run must not stop the server that run is using ------------------------------------
os.environ["TG_CAPTIONER_SETTINGS"] = str(Path(tempfile.mkdtemp(prefix="tf_tgset_")) / "settings.json")   # not the real one
import captioner_ui as cui  # noqa: E402

os.environ.update(FAKE_LLAMA_LOAD="0", FAKE_LLAMA_DELAY="0.4")
d = grabber()


def ui_values(**kw):
    v = dict(cui.DEFAULTS, folder=str(d), keep_running=True)
    v.update(kw)
    return [v[k] for k in cui.UI_KEYS]


check("UI: 📋 Load gives the model's prompt for the chosen format, and its system prompt",
      cui.load_prompt("long") == (tc.default_template("long"), tc.tg.system_prompt))
check("UI: ↺ empties both, back to the model's own", cui.clear_prompt() == ("", ""))
check("UI: the prompt boxes reach the captioning options",
      (lambda o: o.prompt == "P {tags}" and o.system_prompt == "S")(cui.split_values(ui_values(prompt="P {tags}", system_prompt="S"))[1]))
session = cui.job.get_session(cui.split_values(ui_values())[2])   # the UI's session, its model already loaded
srv = session.server = tc.LlamaServer([sys.executable, str(FAKE)], Path("m.gguf"), Path("mm.gguf"),
                                      log_path=Path(tempfile.mkdtemp()) / "srv.log")
try:
    session.base_url = srv.start(timeout=30)
    cui.start_captioning(*ui_values())
    time.sleep(1.0)
    check("UI: a folder run is in progress", cui.job.busy(), cui.job.status)
    for label, press in [("🔍 Try one image", cui.try_one), ("▶ Caption folder", cui.start_captioning),
                         ("⬇ Download now", cui.download_now)]:
        press(*ui_values(ctx=4096, gpu_layers="0"))
        check(f"UI: {label} with changed server settings during a run is refused; the run keeps its server",
              srv.alive() and cui.job.session is session and cui.job.lines[-1] == cui.BUSY, cui.job.lines[-2:])
    cui.job.thread.join(120)
    done = len(list(d.glob("*" + tc.NL_CAPTION_SUFFIX)))
    check("…and the run captions every image", done == len(tc.list_images(d)) and "Captioning finished" in cui.job.status
          and not any("lost the server" in line for line in cui.job.lines), (done, cui.job.status, cui.job.lines[-3:]))
    cui.try_one(*ui_values(mode="external", url=f"http://127.0.0.1:{tc.free_port()}"))
    cui.job.thread.join(60)
    check("UI: once idle, changed server settings do unload the old model", not srv.alive() and cui.job.session is not session,
          cui.job.status)
finally:
    srv.stop()
    del os.environ["FAKE_LLAMA_DELAY"]


# --- downloads ----------------------------------------------------------------------------------------------------------
serve_dir = Path(tempfile.mkdtemp(prefix="tf_dl_"))
payload = os.urandom(3 * 2**20 + 123)
(serve_dir / "model.gguf").write_bytes(payload)
for asset in tc.LLAMA_CPP_ASSETS:
    with zipfile.ZipFile(serve_dir / asset, "w") as z:
        if asset.startswith("llama-"):
            z.writestr("llama-server.exe", b"MZ fake")
            z.writestr("ggml-cuda.dll", b"dll")
        else:
            z.writestr("cudart64_12.dll", b"dll")
ranges = []


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(serve_dir), **k)

    def log_message(self, *a):
        pass

    def send_head(self):
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if rng and path.is_file():
            ranges.append(rng)
            start = int(rng.split("=")[1].split("-")[0])
            data = path.read_bytes()[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return io.BytesIO(data)
        return super().send_head()


httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{httpd.server_address[1]}"
dest = Path(tempfile.mkdtemp()) / "model.gguf"
dest.with_name("model.gguf.part").write_bytes(payload[:1_000_000])         # an interrupted earlier download
prog = []
tc.download(f"{base}/model.gguf", dest, lambda n, a, b: prog.append((a, b)))
check("download resumes a partial file and completes it exactly", dest.read_bytes() == payload and ranges == ["bytes=1000000-"]
      and prog[-1] == (len(payload), len(payload)) and not dest.with_name("model.gguf.part").exists(), (ranges, prog[-1:]))
rt = Path(tempfile.mkdtemp()) / "llama.cpp"
exe = tc.ensure_llama_server(platform="nt", url_template=base + "/{asset}", folder=rt)
check("llama.cpp download (Windows build): both zips extracted into one folder, llama-server.exe found",
      exe == rt / "llama-server.exe" and (rt / "cudart64_12.dll").exists() and (rt / "ggml-cuda.dll").exists()
      and (rt.parent / "downloads" / tc.LLAMA_CPP_ASSETS[1]).exists(), exe)
check("…and found again without downloading", tc.ensure_llama_server(platform="nt", url_template="http://127.0.0.1:9/{asset}", folder=rt) == exe)
with zipfile.ZipFile(serve_dir / tc.LLAMA_CPP_ASSETS[0], "w") as z:
    z.writestr("../evil.exe", b"x")
try:
    tc.ensure_llama_server(platform="nt", url_template=base + "/{asset}", folder=Path(tempfile.mkdtemp()) / "llama.cpp")
    check("zip with paths outside the folder refused", False)
except tc.SetupError as e:
    check("zip with paths outside the folder refused", "unexpected path" in str(e))
try:
    tc.ensure_llama_server(platform="posix", folder=Path(tempfile.mkdtemp()) / "none")
    check("non-Windows: asks for a llama-server path", False)
except tc.SetupError as e:
    check("non-Windows: asks for a llama-server path", "path to llama-server" in str(e))
httpd.shutdown()

# --- command line --------------------------------------------------------------------------------------------------------
d = grabber()
r = subprocess.run([sys.executable, str(TOOL / "toriigate_captioner.py"), str(d), "--dry-run", "--limit", "3", "--characters", "mychar"],
                   capture_output=True, text=True, encoding="utf-8")
check("--dry-run prints the prompts without loading anything", r.returncode == 0 and r.stdout.count("=====") == 3
      and "make sure to use them: [mychar]" in r.stdout and not list(d.glob("*.toriigate.txt")), r.stdout[:200] + r.stderr[-300:])
r = subprocess.run([sys.executable, str(TOOL / "toriigate_captioner.py"), str(nohints), "--dry-run"], capture_output=True, text=True, encoding="utf-8")
check("--dry-run says what was looked at for an image with no hints", "bare.png   (hints: none)" in r.stdout and "[bare.png: no metadata" in r.stdout, r.stdout[:300])
r = subprocess.run([sys.executable, str(TOOL / "toriigate_captioner.py"), str(nohints), "--check-hints"], capture_output=True, text=True, encoding="utf-8")
check("--check-hints prints the hint report and loads nothing", r.returncode == 0 and "1 from embedded tags, 1 from .txt captions, 4 with none" in r.stdout
      and not list(nohints.glob("*.toriigate.txt")), r.stdout[:300] + r.stderr[-300:])
r = subprocess.run([sys.executable, str(TOOL / "toriigate_captioner.py"), "--show-prompt", "--format", "long"],
                   capture_output=True, text=True, encoding="utf-8")
check("--show-prompt prints the model's prompt for the format as a template, and the system prompt",
      r.returncode == 0 and tc.default_template("long") in r.stdout and tc.tg.system_prompt in r.stdout, r.stdout[:200])
pf, sf = Path(tempfile.mkdtemp()) / "prompt.txt", Path(tempfile.mkdtemp()) / "system.txt"
pf.write_text("\ufeff" + mine.replace("\n", "\r\n"), encoding="utf-8")   # as Notepad saves it: BOM, CRLF
sf.write_text("You are terse.", encoding="utf-8")
r = subprocess.run([sys.executable, str(TOOL / "toriigate_captioner.py"), str(d), "--dry-run", "--limit", "1",
                    "--prompt-file", str(pf), "--system-prompt-file", str(sf)], capture_output=True, text=True, encoding="utf-8")
check("--prompt-file / --system-prompt-file (a Notepad file: BOM and CRLF are fine)", r.returncode == 0
      and "Mention the lighting." in r.stdout and "You are terse." in r.stdout and "\ufeff" not in r.stdout
      and "\r" not in r.stdout and "[" in r.stdout.split("# Booru tags for the image")[1], r.stdout[:400] + r.stderr[-300:])
r = subprocess.run([sys.executable, str(TOOL / "toriigate_captioner.py"), str(d), "--dry-run", "--prompt-file", "nope.txt"],
                   capture_output=True, text=True, encoding="utf-8")
check("…a missing file is reported", r.returncode == 2 and "can't read nope.txt" in r.stdout, r.stdout + r.stderr[-200:])
proc, url, rec = fake_server()
try:
    r = subprocess.run([sys.executable, str(TOOL / "toriigate_captioner.py"), str(d), "--server-url", url, "--limit", "2", "--format", "json"],
                       capture_output=True, text=True, encoding="utf-8")
    check("CLI captions through a server URL", r.returncode == 0 and len(list(d.glob("*.toriigate.txt"))) == 2
          and "Captioned 2" in r.stdout, r.stdout[-300:] + r.stderr[-300:])
finally:
    proc.kill()

# --- TrainFlow's tools and .toriigate.txt -------------------------------------------------------------------------------
g = load_app()
d = tmpdir()
for n in ("b", "a"):
    Image.new("RGB", (600, 900), (10 if n == "a" else 200, 0, 0)).save(d / f"{n}.png")
    (d / f"{n}.txt").write_text("1girl", encoding="utf-8")
    (d / f"{n}.toriigate.txt").write_text(f"A picture called {n}.\n\nSecond paragraph.", encoding="utf-8")
out = drain(g, g["inspect_dataset"](str(d), "kw"))
check("inspector: counts natural-language captions, doesn't call them orphans or multi-line tag captions",
      "2/2 image(s) have a natural-language caption" in out and "with no image" not in out and "more than one line" not in out, out)
check("inspector: tag statistics ignore them", "Captions: 2 " in out or "Captions: 2 |" in out, out.splitlines()[:2])
drain(g, g["rename_dataset_sequential"](str(d)))
check("Rename to Numbers renames the natural-language caption with its image",
      (d / "001.toriigate.txt").read_text(encoding="utf-8").startswith("A picture called a") and not (d / "a.toriigate.txt").exists())
(d / "zz_orphan.toriigate.txt").write_text("x", encoding="utf-8")
out = drain(g, g["inspect_dataset"](str(d), "kw"))
check("inspector: a natural-language caption without its image is flagged", "natural-language caption(s) with no image: zz_orphan.toriigate.txt" in out)
d = tmpdir()
a = Image.open(REAL / "astronaut.png").convert("RGB")
a.save(d / "big.png")
a.resize((300, 300)).save(d / "small.jpg")
(d / "small.toriigate.txt").write_text("A caption written for the small copy.", encoding="utf-8")
bucket(g, d)
check("duplicate set aside: its natural-language caption goes too, and the kept copy gets a copy of it",
      (d / "duplicates" / "small.toriigate.txt").exists() and (d / "big.toriigate.txt").exists()
      and not (d / "small.toriigate.txt").exists(), sorted(p.name for p in d.rglob("*.toriigate.txt")))

finish()
