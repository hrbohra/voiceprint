"""Per-turn feature extraction: one row per turn, every family of features, one stage at a time.

Stages run in a fixed order and release their models between stages, so peak GPU memory stays at
one model. Stage outputs are merged into a pandas DataFrame indexed by turn key.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import models
from ..schema import Corpus, Turn
from . import dialogue, neural, surface, syntax

log = logging.getLogger("voiceprint")

BLOCK = 512  # turns per checkpointed block in the model stages

FAMILIES = ["surface", "syntax", "affect", "acts", "social", "dialogue"]


def _family_of(col: str) -> str:
    if col.startswith(("pos_", "lexical_density", "formality_fscore", "tree_", "clauses", "subordination", "coordination", "passive", "imperative", "interrogative", "exclamative", "modal", "negation", "past_tense", "starts_with_conj", "named_entities")):
        return "syntax"
    if col.startswith(("sentiment_", "emotion_", "formality_model")):
        return "affect"
    if col.startswith("act_"):
        return "acts"
    if col.startswith("social_"):
        return "social"
    if col in ("has_context", "length_ratio", "lsm", "answers_question", "asks_back", "lexical_echo", "reply_latency_s"):
        return "dialogue"
    return "surface"


def family_map(columns) -> dict[str, str]:
    return {c: _family_of(c) for c in columns}


def _code_version() -> str:
    """Hash of the feature code itself, so editing a feature invalidates cached features."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def extract(corpus: Corpus, turns: list[Turn], *, neural_on: bool = True, research_models: bool = False,
            progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Features for `turns` (a subset of `corpus`, which supplies conversation context)."""
    say = progress or (lambda m: log.info(m))
    texts = [t.text for t in turns]
    prev_map = corpus.previous_turn()
    prevs = [prev_map.get(t.key) for t in turns]
    # Content-addressed cache: same turns, same context, same model pins -> same features.
    h = hashlib.sha256(repr((neural_on, research_models, sorted(models.MODELS.items()), _code_version())).encode())
    for t, p in zip(turns, prevs):
        h.update("\x00".join((t.key, t.speaker, t.text, p.text if p else "")).encode() + b"\x01")
    cache = Path(os.environ.get("VOICEPRINT_FEATURE_CACHE", ".voiceprint/features")) / f"{h.hexdigest()[:24]}.pkl"
    if cache.exists():
        say(f"features from cache ({cache.name})")
        return pd.read_pickle(cache)
    rows: list[dict] = [{} for _ in turns]
    # Resumable: every stage, and every block of BLOCK turns inside the neural stages, is
    # checkpointed next to the cache file. A run killed after an hour of NLI resumes where it stopped.
    ckpt = cache.with_suffix("")
    ckpt.mkdir(parents=True, exist_ok=True)

    def staged(name: str, fn: Callable[[list[int]], list[dict]], block: int | None = None) -> None:
        idx = list(range(len(turns)))
        blocks = [idx[i : i + block] for i in range(0, len(idx), block)] if block else [idx]
        done = 0
        for b, ids in enumerate(blocks):
            f = ckpt / f"{name}.{b}.pkl"
            if f.exists():
                part = pd.read_pickle(f)
                done += 1
            else:
                part = fn(ids)
                pd.to_pickle(part, f)
            for i, feats in zip(ids, part):
                rows[i].update(feats)
        say(f"{name}" + (f" ({done}/{len(blocks)} blocks from checkpoint)" if done else ""))

    staged("surface", lambda ids: [surface.turn_features(texts[i]) for i in ids])
    staged("syntax", lambda ids: [syntax.doc_features(d) for d in syntax.parse([texts[i] for i in ids], batch_size=models.batch_size(64, 16))], BLOCK)
    staged("dialogue", lambda ids: [dialogue.turn_dialogue_features(turns[i], prevs[i]) for i in ids])
    if neural_on:
        stages = [("sentiment", lambda ids: neural.sentiment([texts[i] for i in ids])),
                  ("emotions", lambda ids: neural.emotions([texts[i] for i in ids]))]
        if research_models:
            stages.append(("formality", lambda ids: neural.formality_model([texts[i] for i in ids])))
        stages += [("dialogue_acts", lambda ids: neural.dialogue_acts([texts[i] for i in ids], [prevs[i].text if prevs[i] else "" for i in ids])),
                   ("social_acts", lambda ids: neural.social_acts([texts[i] for i in ids]))]
        for name, fn in stages:
            staged(name, fn, BLOCK)
            models.release()

    df = pd.DataFrame(rows, index=[t.key for t in turns])
    df.insert(0, "speaker", [t.speaker for t in turns])
    df.insert(1, "doc_id", [t.doc_id for t in turns])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    import shutil

    shutil.rmtree(ckpt, ignore_errors=True)  # the full cache supersedes the checkpoints
    return df


def top_act(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("act_") and c != "act_confidence"]
    if not cols:
        return pd.Series(index=df.index, dtype=object)
    return df[cols].idxmax(axis=1).str.removeprefix("act_")
