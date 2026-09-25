"""Surface features: computed from the text alone, exactly, with no model. Per turn.

Every feature is a number per turn (a rate, a count or a flag), so the aggregator can report its
distribution, not just its mean. Rates are per word or per sentence as named."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import emoji as emojilib
import textstat

from .lexicons import BOOSTERS, CONTRACTIONS_RE, DISCOURSE_MARKERS, EXPANDABLE, HEDGES, INTENSIFIERS, PROFANITY, TEXTESE, UK_US

WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d+(?:[.,]\d+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9\"“'(<])|\n+")
CONTRACTION = re.compile(CONTRACTIONS_RE, re.I)
EMOTICON = re.compile(r"(?:[:;=8xX][-o^']?[)(\]\[dDpP/\\|*@3]|<3|\^_\^|:'\(|¯\\_\(ツ\)_/¯)")
PLACEHOLDER = re.compile(r"<[A-Z_]+(?:_\d+)?>")
URLISH = re.compile(r"https?://\S+|www\.\S+")


def _phrase_re(items: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(i) for i in sorted(items, key=len, reverse=True)) + r")\b", re.I)


HEDGE_RE, BOOST_RE, INTENS_RE, DM_RE = _phrase_re(HEDGES), _phrase_re(BOOSTERS), _phrase_re(INTENSIFIERS), _phrase_re(DISCOURSE_MARKERS)
EXPAND_RE, TEXTESE_RE, PROF_RE = _phrase_re(EXPANDABLE), _phrase_re(TEXTESE), _phrase_re(PROFANITY)
UK_RE, US_RE = _phrase_re([u for u, _ in UK_US]), _phrase_re([a for _, a in UK_US])
PRON = {
    "first_singular": {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"},
    "first_plural": {"we", "us", "our", "ours", "ourselves", "we're", "we've", "we'll", "we'd"},
    "second": {"you", "your", "yours", "yourself", "yourselves", "you're", "you've", "you'll", "you'd", "u", "ur"},
    "third": {"he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs"},
}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]


def words(text: str) -> list[str]:
    return WORD.findall(PLACEHOLDER.sub(" NAME ", URLISH.sub(" URL ", text)))


def emoji_info(text: str) -> tuple[int, str]:
    """(count, position) where position is lead / end / inline / standalone / none."""
    found = emojilib.emoji_list(text)
    if not found:
        return 0, "none"
    stripped = text.strip()
    if emojilib.replace_emoji(stripped, "").strip() == "":
        return len(found), "standalone"
    first, last = found[0]["match_start"], found[-1]["match_end"]
    if first <= len(text) - len(text.lstrip()) + 1:
        return len(found), "lead"
    if last >= len(stripped) - 1 or not text[last:].strip(" .!?"):
        return len(found), "end"
    return len(found), "inline"


def _only_names_capitalised(text: str) -> bool:
    """True when every capital letter belongs to a vocative or mid-sentence proper noun, i.e. the
    writer lowercases everything they are not forced to capitalise."""
    caps = re.findall(r"\b[A-Z][a-z']*", text)
    return len(caps) <= 1 and not re.match(r"\W*[A-Z][a-z']*\s+[a-z]", text)


def _strip_tail(text: str) -> str:
    """Text without trailing emoji, emoticons and spaces, so "need? 🙂" still ends with a question."""
    t = emojilib.replace_emoji(text, "").rstrip()
    return EMOTICON.sub("", t[-4:]).rstrip() if EMOTICON.search(t[-4:]) else t


def casing(text: str) -> str:
    text = PLACEHOLDER.sub(" ", text)  # "<PERSON_1>" is not the writer's capitalisation
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "none"
    if all(c.islower() for c in letters):
        return "all_lower"
    if sum(c.isupper() for c in letters) / len(letters) > 0.7 and len(letters) > 6:
        return "shouting"
    # Sentence starts, skipping a vocative name ("Sam! the room is free"): a capitalised first word
    # directly followed by "!" or "," is addressing someone, not the writer's casing habit.
    starts = []
    for s in sentences(text):
        m = re.match(r"\W*([A-Za-z][\w']*)([!,])?\s*(\w)?", s)
        if not m:
            continue
        word, punct, nxt = m.groups()
        starts.append(nxt if (punct and word[0].isupper() and nxt) else word[0])
    if starts and all(c.islower() for c in starts if c.isalpha()):
        if _only_names_capitalised(text):
            return "all_lower"
        return "lower_starts"
    return "standard"


def turn_features(text: str) -> dict[str, float]:
    ws = words(text)
    nw = max(len(ws), 1)
    lw = [w.lower() for w in ws]
    sents = sentences(text) or [text]
    ns = len(sents)
    chars = len(text)
    n_emoji, epos = emoji_info(text)
    punct = Counter(c for c in text if unicodedata.category(c).startswith("P") or c in "!?…")
    f: dict[str, float] = {
        # size and rhythm
        "words": len(ws),
        "chars": chars,
        "sentences": ns,
        "words_per_sentence": len(ws) / ns,
        "mean_word_length": sum(len(w) for w in ws) / nw,
        "long_word_rate": sum(len(w) >= 7 for w in ws) / nw,
        "line_breaks": text.count("\n"),
        "is_fragment": float(len(ws) <= 3 and not re.search(r"[.!?]$", text.strip())),
        # punctuation per sentence
        "exclamations_per_sentence": text.count("!") / ns,
        "questions_per_sentence": text.count("?") / ns,
        "ends_with_question": float(_strip_tail(text).endswith("?")),
        "ends_with_exclamation": float(_strip_tail(text).endswith("!")),
        "ends_without_punctuation": float(bool(_strip_tail(text)) and _strip_tail(text)[-1].isalnum()),
        "ellipses_per_sentence": (text.count("...") + text.count("…")) / ns,
        "multi_exclamation": float(bool(re.search(r"!{2,}", text))),
        "multi_question": float(bool(re.search(r"\?{2,}", text))),
        "commas_per_sentence": punct.get(",", 0) / ns,
        "dashes_per_sentence": (text.count(" - ") + text.count("—") + text.count("–")) / ns,
        "parentheses": float("(" in text),
        "semicolons_colons": (punct.get(";", 0) + punct.get(":", 0)) / ns,
        "quotes": float(bool(re.search(r"[\"“”]", text))),
        # emoji and emoticons
        "emoji_count": n_emoji,
        "has_emoji": float(n_emoji > 0),
        "emoji_lead": float(epos == "lead"),
        "emoji_end": float(epos == "end"),
        "emoji_inline": float(epos == "inline"),
        "emoji_standalone": float(epos == "standalone"),
        "emoticons": len(EMOTICON.findall(text)),
        # casing
        "all_lowercase": float(casing(text) == "all_lower"),
        "lowercase_sentence_starts": float(casing(text) == "lower_starts"),
        "shouting": float(casing(text) == "shouting"),
        "capitalised_words_rate": sum(w.isupper() and len(w) > 1 for w in ws) / nw,
        # words and register
        "contraction_rate": len(CONTRACTION.findall(text)) / nw,
        "uncontracted_forms": len(EXPAND_RE.findall(text)) / nw,
        "hedge_rate": len(HEDGE_RE.findall(text)) / nw,
        "booster_rate": len(BOOST_RE.findall(text)) / nw,
        "intensifier_rate": len(INTENS_RE.findall(text)) / nw,
        "discourse_marker_rate": len(DM_RE.findall(text)) / nw,
        "textese_rate": len(TEXTESE_RE.findall(text)) / nw,
        "profanity_rate": len(PROF_RE.findall(text)) / nw,
        "uk_spelling": len(UK_RE.findall(text)),
        "us_spelling": len(US_RE.findall(text)),
        "numbers_rate": sum(w[0].isdigit() for w in ws) / nw,
        "placeholder_names": len(re.findall(r"<PERSON_\d+>", text)),
        "urls": len(URLISH.findall(text)),
        # address
        **{f"pronoun_{k}": sum(w in v for w in lw) / nw for k, v in PRON.items()},
        "addresses_by_name_first": float(bool(re.match(r"^\W*(hi|hey|hello|dear|thanks|thank you)?\W*<PERSON_\d+>", text, re.I))),
    }
    # readability: defined on text with at least one full sentence of a few words
    if len(ws) >= 5:
        f["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        f["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        f["gunning_fog"] = textstat.gunning_fog(text)
    return f


# ── lexical richness over the speaker's whole text (length-robust measures) ──


def mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010), averaged forwards and back."""
    def one_pass(toks: list[str]) -> float:
        factors, types, count = 0.0, set(), 0
        for t in toks:
            types.add(t)
            count += 1
            if len(types) / count <= ttr_threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - ttr_threshold) if ttr < 1 else 0
        return len(toks) / factors if factors else float(len(toks))

    if len(tokens) < 50:
        return float("nan")
    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def hdd(tokens: list[str], sample: int = 42) -> float:
    """HD-D (McCarthy & Jarvis 2007): expected type contribution in random 42-token samples."""
    n = len(tokens)
    if n < sample:
        return float("nan")
    counts = Counter(tokens)

    def comb(a: int, b: int) -> float:
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1) if 0 <= b <= a else float("-inf")

    total = 0.0
    for c in counts.values():
        p0 = math.exp(comb(n - c, sample) - comb(n, sample)) if n - c >= sample else 0.0
        total += (1 - p0) / sample
    return total * sample


def richness(texts: list[str]) -> dict[str, float]:
    toks = [w.lower() for t in texts for w in words(t)]
    return {"mtld": mtld(toks), "hdd": hdd(toks), "tokens": float(len(toks)), "types": float(len(set(toks)))}
