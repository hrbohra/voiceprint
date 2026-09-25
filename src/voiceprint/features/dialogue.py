"""Dialogue features: how a speaker behaves relative to the turn they answer.

Grounded in conversation-analysis and chatbot-evaluation practice:
  - length ratio and accommodation (Giles' Communication Accommodation Theory);
  - Language Style Matching over function-word categories (Ireland & Pennebaker 2010);
  - openers and closers (sequence organisation: greeting / pre-closing / closing);
  - act transitions (adjacency pairs: question -> answer, thanks -> acknowledge);
  - timing: reply latency and message bursts (multi-message turns, typical of DMs).
"""

from __future__ import annotations

import re
from collections import Counter

from ..schema import Corpus, Turn
from .surface import words

# LIWC-style function-word categories used for LSM (public approximations, not the LIWC lexicon).
LSM_CATS = {
    "personal_pronouns": {"i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "she", "him", "her", "they", "them", "their"},
    "impersonal_pronouns": {"it", "its", "this", "that", "these", "those", "anything", "something", "everything", "nothing"},
    "articles": {"a", "an", "the"},
    "prepositions": {"to", "of", "in", "for", "on", "with", "at", "by", "from", "about", "into", "over", "after", "under"},
    "auxiliary_verbs": {"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must"},
    "high_frequency_adverbs": {"very", "really", "so", "just", "then", "now", "also", "too", "quite", "still", "even"},
    "conjunctions": {"and", "but", "or", "because", "so", "if", "while", "although", "though", "yet"},
    "negations": {"no", "not", "never", "none", "nothing", "don't", "can't", "won't", "isn't", "didn't", "doesn't"},
    "quantifiers": {"all", "some", "any", "many", "much", "few", "more", "most", "lots", "every", "each"},
}


def _cat_rates(text: str) -> dict[str, float]:
    ws = [w.lower() for w in words(text)]
    n = max(len(ws), 1)
    return {c: sum(w in s for w in ws) / n for c, s in LSM_CATS.items()}


def lsm(a: str, b: str) -> float:
    """Language Style Matching in [0, 1]: 1 - |a-b|/(a+b) averaged over categories."""
    ra, rb = _cat_rates(a), _cat_rates(b)
    vals = [1 - abs(ra[c] - rb[c]) / (ra[c] + rb[c] + 1e-4) for c in LSM_CATS]
    return sum(vals) / len(vals)


def opener(text: str) -> str:
    """First two lowercased words, placeholders kept (so '<PERSON_1>' counts as naming the reader)."""
    ws = re.findall(r"<[A-Z_]+(?:_\d+)?>|[A-Za-z']+", text)
    return " ".join(w if w.startswith("<") else w.lower() for w in ws[:2]) or "∅"


def closer(text: str) -> str:
    ws = re.findall(r"<[A-Z_]+(?:_\d+)?>|[A-Za-z']+", text)
    return " ".join(w if w.startswith("<") else w.lower() for w in ws[-2:]) or "∅"


def turn_dialogue_features(turn: Turn, prev: Turn | None) -> dict[str, float]:
    f: dict[str, float] = {"has_context": float(prev is not None)}
    if prev is None:
        return f
    nw, pw = len(words(turn.text)), len(words(prev.text))
    f["length_ratio"] = (nw + 1) / (pw + 1)
    f["lsm"] = lsm(turn.text, prev.text)
    f["answers_question"] = float("?" in prev.text)
    f["asks_back"] = float("?" in prev.text and "?" in turn.text)
    pl = {w.lower() for w in words(prev.text) if len(w) > 3}
    tl = [w.lower() for w in words(turn.text) if len(w) > 3]
    f["lexical_echo"] = sum(w in pl for w in tl) / max(len(tl), 1)
    if turn.timestamp and prev.timestamp:
        f["reply_latency_s"] = max((turn.timestamp - prev.timestamp).total_seconds(), 0.0)
    return f


def speaker_dialogue_summary(corpus: Corpus, speakers: set[str]) -> dict:
    """Conversation-level behaviour: openers, closers, burstiness, who starts and ends."""
    openers, closers = Counter(), Counter()
    bursts, starts, ends, docs = [], 0, 0, 0
    for turns in corpus.docs().values():
        mine = [t for t in turns if t.speaker in speakers]
        if not mine:
            continue
        docs += 1
        starts += turns[0].speaker in speakers
        ends += turns[-1].speaker in speakers
        # a burst: consecutive turns by the target without anyone else speaking
        run = 0
        for t in turns:
            if t.speaker in speakers:
                run += 1
            elif run:
                bursts.append(run)
                run = 0
        if run:
            bursts.append(run)
        # openers and closers of the target's first and last turn in each conversation
        openers[opener(mine[0].text)] += 1
        closers[closer(mine[-1].text)] += 1
    return {
        "conversations": docs,
        "starts_conversation_rate": starts / max(docs, 1),
        "ends_conversation_rate": ends / max(docs, 1),
        "mean_burst": sum(bursts) / max(len(bursts), 1),
        "multi_message_turn_rate": sum(b > 1 for b in bursts) / max(len(bursts), 1),
        "top_openers": openers.most_common(12),
        "top_closers": closers.most_common(12),
    }


def act_transitions(prev_acts: list[str | None], acts: list[str]) -> dict[str, dict[str, float]]:
    """P(target act | previous act): the speaker's adjacency-pair habits."""
    table: dict[str, Counter] = {}
    for p, a in zip(prev_acts, acts):
        if p is None:
            continue
        table.setdefault(p, Counter())[a] += 1
    return {p: {a: c / sum(cnt.values()) for a, c in cnt.most_common()} for p, cnt in table.items()}
