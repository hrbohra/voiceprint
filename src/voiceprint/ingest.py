"""Loaders: every supported format becomes a `Corpus` of `Turn`s.

Supported: JSON Lines / JSON arrays of turns, CSV, plain text and Markdown, Project Gutenberg
books, WhatsApp exports, Instagram and Messenger JSON exports, and the Customer Support on Twitter
dataset. `load()` picks the loader from the extension and content unless told explicitly.
"""

from __future__ import annotations

import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from .schema import Corpus, Turn


def latin_share(text: str) -> float:
    """Share of letters in Latin script. The feature models are English; this is how non-English
    turns are kept out (a brand account can answer in Japanese in one thread and English in the next)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c.isascii() or "À" <= c <= "ɏ" for c in letters) / len(letters)


# Very common English words that are rare in other Latin-script languages ("die", "an", "so" are
# excluded because German uses them). Short chat turns rarely contain none of these.
_EN_MARKERS = frozenset(
    "the you your we our is are was were be been have has had do does did not this that with for "
    "to of and it it's i'm we're you're can could would will please thanks thank sorry help hi hey "
    "what how when where why there here just get know let ok okay great yes no sure glad happy good "
    "my me from on at by an in a".split()
)


def is_english(text: str, min_latin: float = 0.9) -> bool:
    """Latin script and enough English marker words (any for short turns, 15% for longer). Cheap and
    dependency-free; good enough to drop the German, Spanish and Japanese turns in brand corpora."""
    if latin_share(text) < min_latin:
        return False
    ws = re.findall(r"[a-z']+", text.lower())
    hits = sum(w in _EN_MARKERS for w in ws)
    if len(ws) < 6:
        return hits > 0 or len(ws) < 2
    return hits / len(ws) >= 0.15  # English prose runs ~40-50%; a stray "hi" in German is ~6%


# Agent sign-off (support accounts) at the end: marker + initials/first name, optionally followed by "2/2" and/or a link
SIG = re.compile(r"\s*[\^*/~\-–—]\s?[A-Z][A-Za-z]{0,14}\.?(?:\s+\d/\d)?(?:\s*<URL>)?\s*$")
def strip_signature(text: str) -> str:
    prev = None
    while prev != text:
        prev, text = text, SIG.sub("", text)
    return text.strip()


def without_signatures(corpus: Corpus) -> Corpus:
    """Ablation helper: the same corpus with agent sign-offs ("^TN", "-SLM", "*Kellen") removed."""
    return Corpus([Turn(t.doc_id, t.turn_id, t.speaker, strip_signature(t.text), t.timestamp, t.reply_to, t.meta)
                   for t in corpus.turns], corpus.name)


def english_only(corpus: Corpus, min_latin: float = 0.9) -> Corpus:
    return Corpus([t for t in corpus.turns if is_english(t.text, min_latin)], corpus.name)


def _parse_time(v) -> datetime | None:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 1e12 else v)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%a %b %d %H:%M:%S %z %Y", "%d/%m/%Y, %H:%M", "%d/%m/%Y, %H:%M:%S", "%m/%d/%y, %H:%M"):
        try:
            return datetime.strptime(str(v), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


# ── generic structured formats ─────────────────────────────────────────────


def load_jsonl(path: Path) -> Corpus:
    """One JSON object per line, or a JSON array. Fields: text (required), speaker, doc_id,
    turn_id, timestamp, reply_to. Unknown fields go to meta."""
    raw = path.read_text(encoding="utf-8")
    rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    turns, counters = [], {}
    for r in rows:
        doc = str(r.get("doc_id", r.get("conversation_id", "doc")))
        tid = int(r["turn_id"]) if "turn_id" in r else counters.get(doc, 0)
        counters[doc] = tid + 1
        known = {"text", "speaker", "doc_id", "conversation_id", "turn_id", "timestamp", "reply_to"}
        turns.append(Turn(doc, tid, str(r.get("speaker", "unknown")), str(r.get("text", "")), _parse_time(r.get("timestamp")),
                          int(r["reply_to"]) if r.get("reply_to") not in (None, "") else None, {k: v for k, v in r.items() if k not in known}))
    return Corpus(turns, path.stem)


def load_csv(path: Path, text_col: str = "text", speaker_col: str = "speaker", doc_col: str = "doc_id") -> Corpus:
    turns, counters = [], {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            doc = str(r.get(doc_col) or "doc")
            tid = counters.get(doc, 0)
            counters[doc] = tid + 1
            turns.append(Turn(doc, tid, str(r.get(speaker_col) or "unknown"), r.get(text_col) or "", _parse_time(r.get("timestamp"))))
    return Corpus(turns, path.stem)


# ── prose ──────────────────────────────────────────────────────────────────

GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
QUOTE = re.compile(r"[\"“]([^\"”]{2,})[\"”]")


def load_prose(path: Path, speaker: str | None = None, split_dialogue: bool = True) -> Corpus:
    """Plain text, Markdown or a Gutenberg book. Paragraphs are turns. With split_dialogue, quoted
    speech inside a paragraph is separated from narration, so an author's narrative voice can be
    studied apart from their characters' lines."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
        e = GUTENBERG_END.search(text)
        text = text[: e.start()] if e else text
    who = speaker or path.stem
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p) > 1 and not re.fullmatch(r"(chapter|book|part)\s+[\w.]+\.?", p, re.I)]
    turns: list[Turn] = []
    for i, p in enumerate(paras):
        if split_dialogue:
            speech = " ".join(QUOTE.findall(p)).strip()
            narration = QUOTE.sub(" ", p)
            narration = re.sub(r"\s+", " ", narration).strip(" ,.;")
            if narration and len(narration.split()) >= 3:
                turns.append(Turn(path.stem, len(turns), f"{who}:narration", narration, meta={"para": i}))
            if speech:
                turns.append(Turn(path.stem, len(turns), f"{who}:dialogue", speech, meta={"para": i}))
        else:
            turns.append(Turn(path.stem, len(turns), who, p, meta={"para": i}))
    return Corpus(turns, path.stem)


def load_prose_dir(path: Path, block: int = 25, narration_only: bool = True) -> Corpus:
    """A folder of books named `<author>__<id>.txt` (e.g. from Project Gutenberg). The speaker is the
    author. Each book is cut into blocks of `block` paragraphs that serve as documents, so held-out
    splits take whole passages rather than scattered paragraphs."""
    turns: list[Turn] = []
    for f in sorted(path.glob("*.txt")):
        author = f.stem.split("__")[0]
        book = load_prose(f, speaker=author)
        paras = [t for t in book.turns if not narration_only or t.speaker.endswith(":narration")]
        for i, t in enumerate(paras):
            doc = f"{f.stem}#{i // block}"
            turns.append(Turn(doc, i % block, author, t.text, meta={"book": f.stem}))
    return Corpus(turns, path.name)


# ── chat exports ───────────────────────────────────────────────────────────

WHATSAPP = re.compile(r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),? (\d{1,2}:\d{2}(?::\d{2})?)(?:\s?[ap]m)?\]? (?:- )?([^:]+?): (.*)$", re.I)


def load_whatsapp(path: Path) -> Corpus:
    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = WHATSAPP.match(line.strip("‎"))
        if m:
            d, t, who, msg = m.groups()
            if msg.strip() in ("<Media omitted>", "This message was deleted"):
                continue
            turns.append(Turn(path.stem, len(turns), who.strip(), msg, _parse_time(f"{d}, {t}")))
        elif turns and line.strip():
            turns[-1].text += "\n" + line  # continuation of a multi-line message
    return Corpus(turns, path.stem)


def load_meta_export(path: Path) -> Corpus:
    """Instagram / Messenger 'Download your information' JSON: {participants, messages:[{sender_name,
    timestamp_ms, content}]}. Accepts a file or a folder of conversation folders."""
    files = [path] if path.is_file() else sorted(path.rglob("message_*.json"))
    turns: list[Turn] = []
    for f in files:
        j = json.loads(f.read_text(encoding="utf-8"))
        doc = f.parent.name if path.is_dir() else f.stem
        msgs = sorted(j.get("messages", []), key=lambda m: m.get("timestamp_ms", 0))
        for m in msgs:
            content = m.get("content")
            if not content:
                continue
            # Meta exports encode UTF-8 as latin-1 escapes; undo that so emoji and accents survive.
            try:
                content = content.encode("latin-1").decode("utf-8")
                who = m.get("sender_name", "unknown").encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                who = m.get("sender_name", "unknown")
            turns.append(Turn(doc, len([t for t in turns if t.doc_id == doc]), who, content, _parse_time(m.get("timestamp_ms"))))
    return Corpus(turns, path.stem)


def load_twitter_support(path: Path, max_rows: int | None = None) -> Corpus:
    """Customer Support on Twitter (twcs.csv): tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id. Rebuilds threads so each conversation is a doc and
    brand replies point at the customer tweet they answer."""
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if max_rows and i >= max_rows:
                break
            rows[r["tweet_id"]] = r
    root: dict[str, str] = {}

    def find_root(tid: str) -> str:
        seen = []
        cur = tid
        while True:
            if cur in root:
                r = root[cur]
                break
            parent = rows.get(cur, {}).get("in_response_to_tweet_id") or ""
            if not parent or parent not in rows or cur in seen:
                r = cur
                break
            seen.append(cur)
            cur = parent
        for s in seen + [tid]:
            root[s] = r
        return r

    by_doc: dict[str, list[dict]] = {}
    for tid, r in rows.items():
        by_doc.setdefault(find_root(tid), []).append(r)
    turns: list[Turn] = []
    for doc, rs in by_doc.items():
        rs.sort(key=lambda r: int(r["tweet_id"]))
        ids = {r["tweet_id"]: i for i, r in enumerate(rs)}
        for i, r in enumerate(rs):
            inbound = r["inbound"].strip().lower() == "true"
            text = html.unescape(re.sub(r"^(@\w+\s+)+", "", r["text"])).strip()  # leading @mentions are addressing, not voice
            text = re.sub(r"https?://t\.co/\w+", "<URL>", text)  # shortened links: one placeholder, not noise
            parent = r.get("in_response_to_tweet_id") or ""
            turns.append(Turn(doc, i, "customer" if inbound else r["author_id"], text, _parse_time(r["created_at"]),
                              ids.get(parent), {"inbound": inbound, "brand": None if inbound else r["author_id"]}))
    return Corpus(turns, path.stem)


LOADERS: dict[str, Callable[..., Corpus]] = {
    "jsonl": load_jsonl, "json": load_jsonl, "csv": load_csv, "txt": load_prose, "md": load_prose,
    "gutenberg": load_prose, "prose-dir": load_prose_dir, "whatsapp": load_whatsapp, "meta": load_meta_export, "twitter-support": load_twitter_support,
}


def detect(path: Path) -> str:
    if path.is_dir():
        return "prose-dir" if any(path.glob("*__*.txt")) else "meta"
    ext = path.suffix.lower().lstrip(".")
    if ext == "csv":
        head = path.open(encoding="utf-8", errors="replace").readline()
        return "twitter-support" if "in_response_to_tweet_id" in head else "csv"
    if ext == "json":
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
        return "meta" if '"sender_name"' in head else "jsonl"
    if ext == "txt":
        head = path.read_text(encoding="utf-8", errors="replace")[:6000]
        if "PROJECT GUTENBERG" in head.upper():
            return "gutenberg"
        if WHATSAPP.match(head.splitlines()[0].strip("‎") if head else ""):
            return "whatsapp"
    return ext if ext in LOADERS else "txt"


def load(path: str | Path, fmt: str | None = None, **kw) -> Corpus:
    p = Path(path)
    return LOADERS[fmt or detect(p)](p, **kw)
