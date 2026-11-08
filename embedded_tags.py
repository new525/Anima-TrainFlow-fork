"""Tags embedded in images by Grabber / exiftool / Windows (shared by TrainFlow's app.py and the ToriiGate captioner).

Grabber stores tags via exiftool (-sep ";") or the Windows Property System. They end up in XMP dc:subject, XMP
pdf:Keywords (GIF/MP4 "Keywords"), IPTC Keywords, EXIF XPKeywords, or an .xmp sidecar. Bucketing re-encodes the
dataset PNGs, so the original in original_images/ (and any sidecar) is read too.
Captions follow Anima's model card: lowercase, spaces instead of underscores (score tags keep them), artists prefixed
with @, order [meta] [1girl/1boy...] [character] [series] [artist] [general].
"""
import re
from pathlib import Path
from typing import List, Tuple

from PIL import Image

BUCKET_MANIFEST = "bucketing_manifest.json"  # TrainFlow's bucketing record in original_images/ (not a tag source)
NL_CAPTION_SUFFIX = ".toriigate.txt"          # natural-language captions written by tools/toriigate-captioner


KAOMOJIS = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]
# Grabber's namespace names (tags/tag.cpp) -> Anima caption section
TAG_NAMESPACES = {
    "artist": "artist", "circle": "artist",
    "character": "character", "model": "character",
    "copyright": "series", "series": "series",
    "meta": "meta",
    "general": "general", "species": "general", "photo_set": "general", "unknown": "general", "lore": "general", "invalid": "general",
}
CAPTION_SECTIONS = ["meta", "count", "character", "series", "artist", "general"]
_COUNT_TAG = re.compile(r"^(?:\d+\+?|multiple )(?:girl|boy|other)s?$")
_SCORE_TAG = re.compile(r"^score_\d+(?:_up)?$")
_XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)
_XMP_SUBJECT = re.compile(r"<dc:subject\b[^>]*>(.*?)</dc:subject>", re.S)
_XMP_LI = re.compile(r"<rdf:li\b[^>]*>(.*?)</rdf:li>", re.S)
_XMP_PDF_KW = re.compile(r"<pdf:Keywords\b[^>]*>(.*?)</pdf:Keywords>|\bpdf:Keywords\s*=\s*(['\"])(.*?)\2", re.S)
_XMP_SCAN_WHOLE = 64 * 1024 * 1024  # bigger files (videos): scan the first and last 32 MB, where containers keep metadata


def _decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")  # exiftool writes IPTC as Latin unless told otherwise


def _xmp_items_from_text(text: str) -> List[str]:
    import html
    for m in _XMP_SUBJECT.finditer(text):
        items = [html.unescape(li).strip() for li in _XMP_LI.findall(m.group(1))]
        if any(items):
            return items
    for m in _XMP_PDF_KW.finditer(text):
        value = m.group(1) if m.group(1) is not None else m.group(3)
        if value and value.strip():
            return [html.unescape(value).strip()]
    return []


def _xmp_items_from_file(path: Path) -> List[str]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size <= _XMP_SCAN_WHOLE:
                blobs = [f.read()]
            else:
                head = f.read(_XMP_SCAN_WHOLE // 2)
                f.seek(size - _XMP_SCAN_WHOLE // 2)
                blobs = [head, f.read()]
    except OSError:
        return []
    for blob in blobs:
        for packet in _XMP_PACKET.findall(blob):
            items = _xmp_items_from_text(packet.decode("utf-8", errors="replace"))
            if items:
                return items
    return []


def _iim_keywords(blob: bytes) -> List[str]:
    """IPTC IIM record 2, dataset 25 (Keywords)."""
    out, i = [], 0
    while True:
        i = blob.find(b"\x1c\x02\x19", i)
        if i < 0 or i + 5 > len(blob):
            return out
        n = int.from_bytes(blob[i + 3:i + 5], "big")
        start = i + 5
        if n & 0x8000:  # extended dataset length
            k = n & 0x7FFF
            n, start = int.from_bytes(blob[start:start + k], "big"), start + k
        out.append(_decode_text(blob[start:start + n]).strip())
        i = start + n


def _pil_keyword_items(path: Path) -> Tuple[List[str], str]:
    """IPTC Keywords and EXIF XPKeywords (Windows keywords) through Pillow."""
    from PIL import IptcImagePlugin
    try:
        with Image.open(path) as im:
            items: List[str] = []
            if im.format in ("JPEG", "TIFF", "MPO"):
                info = IptcImagePlugin.getiptcinfo(im) or {}
                v = info.get((2, 25))
                if v:
                    items = [_decode_text(x).strip() for x in (v if isinstance(v, list) else [v])]
            elif "Raw profile type iptc" in im.info:  # exiftool writes PNG IPTC as a hex text chunk
                lines = str(im.info["Raw profile type iptc"]).strip().splitlines()
                items = _iim_keywords(bytes.fromhex("".join(lines[2:])))
            if any(items):
                return items, "IPTC"
            xp = im.getexif().get(0x9C9E)
            if xp:
                text = (xp if isinstance(xp, bytes) else bytes(xp)).decode("utf-16-le", errors="ignore").rstrip("\x00").strip()
                if text:
                    return [text], "Windows keywords"
    except Exception:
        pass
    return [], ""


def _split_tag_items(items: List[str]) -> List[str]:
    items = [s.strip() for s in items if s and s.strip()]
    out: List[str] = []
    for s in items:
        if re.search(r"[,;\n]", s):
            out += re.split(r"[,;\n]+", s)          # one keyword holding a comma/semicolon list
        elif len(items) == 1:
            out += s.split()                          # Grabber's default %all%: space-separated booru tags
        else:
            out.append(s)                             # one tag per keyword
    return [t.strip() for t in out if t.strip()]


def _split_namespaces(tags: List[str]) -> List[Tuple[str | None, str]]:
    def colon(t):
        ns, sep, rest = t.partition(":")
        return (ns.lower(), rest) if sep and rest and ns.lower() in TAG_NAMESPACES else (None, t)
    parsed = [colon(t) for t in tags]
    if any(ns for ns, _ in parsed):
        return parsed                                 # %all:includenamespace,unsafe% -> artist:name
    # %all:includenamespace% without unsafe gives artist_name. Real tags look like that too (artist_name,
    # character_doll), so this form only counts when every tag carries a namespace.
    known = sorted(TAG_NAMESPACES, key=len, reverse=True)
    def underscore(t):
        for ns in known:
            if t.lower().startswith(ns + "_") and len(t) > len(ns) + 1:
                return ns, t[len(ns) + 1:]
        return None
    us = [underscore(t) for t in tags]
    if tags and all(us):
        return us
    return [(None, t) for t in tags]


def format_embedded_tags(items: List[str]) -> str:
    """Raw keyword values -> Anima-style caption."""
    sections: dict = {s: [] for s in CAPTION_SECTIONS}
    seen = set()
    for ns, tag in _split_namespaces(_split_tag_items(items)):
        t = tag.strip()
        if t in KAOMOJIS:
            section, text = "general", t
        else:
            text = t.lower()
            if not _SCORE_TAG.match(text):
                text = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
            section = TAG_NAMESPACES.get(ns, "general") if ns else "general"
            if section == "artist" or text.startswith("@"):
                section, text = "artist", "@" + text.lstrip("@").strip()
            elif section == "general" and _COUNT_TAG.match(text):
                section = "count"
        if text and text != "@" and text not in seen:
            seen.add(text)
            sections[section].append(text)
    return ", ".join(t for s in CAPTION_SECTIONS for t in sections[s])


class EmbeddedTagIndex:
    """Finds the embedded tags for a dataset image: in the file itself, its original in original_images/, or an
    .xmp sidecar (image.jpg.xmp or image.xmp) next to either."""
    def __init__(self, dataset_dir: Path):
        self.root = dataset_dir
        self.backup = dataset_dir / "original_images"
        self.by_stem: dict = {}
        self.sidecars: dict = {}
        for folder in (self.root, self.backup):
            if not folder.is_dir():
                continue
            for f in folder.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() == ".xmp":
                    base = f.name[:-4]
                    for key in {base.lower(), Path(base).stem.lower()}:
                        self.sidecars.setdefault(key, []).append(f)
                elif f.name != BUCKET_MANIFEST and f.suffix.lower() not in (".txt", ".part"):
                    self.by_stem.setdefault(f.stem.lower(), []).append(f)

    def items_for(self, image: Path) -> Tuple[List[str], str]:
        key = image.stem.lower()
        candidates = [image] + sorted((f for f in self.by_stem.get(key, []) if f != image), key=lambda p: (p.parent != self.backup, p.name))
        for c in candidates:
            items = _xmp_items_from_file(c)
            if items:
                return items, "XMP" if c == image else "XMP (original)"
            items, src = _pil_keyword_items(c)
            if items:
                return items, src if c == image else f"{src} (original)"
        for sc in sorted(set(self.sidecars.get(key, []) + [s for c in candidates for s in self.sidecars.get(c.name.lower(), [])])):
            try:
                items = _xmp_items_from_text(sc.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                items = []
            if items:
                return items, "XMP sidecar"
        return [], ""

    def caption_for(self, image: Path) -> Tuple[str, str]:
        items, src = self.items_for(image)
        return (format_embedded_tags(items), src) if items else ("", "")


def namespaced_tags(items: List[str]) -> List[Tuple[str, str]]:
    """Raw keyword values -> [(section, tag)] with section one of CAPTION_SECTIONS' sources: artist, character,
    series, meta or general (tags without a namespace are general). Tags keep their original spelling."""
    out = []
    for ns, tag in _split_namespaces(_split_tag_items(items)):
        t = tag.strip()
        if t:
            out.append((TAG_NAMESPACES.get(ns, "general") if ns else "general", t))
    return out
