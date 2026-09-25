"""Authorship attribution on held-out conversations: if the extracted features capture voice, they
should tell speakers apart on text the model never saw.

For N speakers with enough turns: features for every turn (one pass), split by conversation into
train/test, then three classifiers compared on the same split:
  - surface   : the portable scorer's features only (what Kiki's TS port sees);
  - full      : every measured feature (surface + syntax + affect + acts + dialogue);
  - style     : nearest style-embedding centroid (StyleDistance), no training.
Accuracy is reported per turn and per 10-turn chunk (one message is a small sample; a voice shows
over several), against the majority-class and chance baselines.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .. import features, models, scorer
from ..pipeline import split_docs
from ..schema import Corpus


def _chunks(idx: np.ndarray, size: int, rng: random.Random) -> list[np.ndarray]:
    idx = list(idx)
    rng.shuffle(idx)
    return [np.array(idx[i : i + size]) for i in range(0, len(idx) - size + 1, size)]


def run(corpus: Corpus, out: Path, n_speakers: int = 10, min_turns: int = 150, max_turns: int = 400, public: bool = True,
        seed: int = 7, progress=print, exclude: tuple[str, ...] = ("customer", "unknown")) -> dict:
    """`exclude` removes pseudo-speakers that pool many writers (all customers in a support corpus):
    telling "a brand" from "everyone else" is not authorship attribution, and including them inflated
    an earlier run (decision D-49)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = random.Random(seed)
    spk = [s for s, n in corpus.speakers().items() if n >= min_turns and s not in exclude][:n_speakers]
    if len(spk) < 2:
        raise ValueError(f"need at least 2 speakers with >= {min_turns} turns")
    train, test = [], []
    for s in spk:
        ts = [t for t in corpus.turns if t.speaker == s and len(t.text.split()) >= 2]
        ts = rng.sample(ts, min(max_turns, len(ts)))
        a, b = split_docs(ts, 0.3, seed)
        train += a
        test += b
    progress(f"attribution: {len(spk)} speakers, {len(train)} train / {len(test)} test turns")
    df = features.extract(corpus, train + test, progress=lambda m: progress(f"    {m}"))
    y_tr = np.array([t.speaker for t in train])
    y_te = np.array([t.speaker for t in test])
    tr_keys, te_keys = [t.key for t in train], [t.key for t in test]
    num = [c for c in df.columns if c not in ("speaker", "doc_id") and df[c].dtype != object]
    surf = [c for c in scorer.PORTABLE if c in df.columns]

    def fit_eval(cols):
        Xtr = df.loc[tr_keys, cols].astype(float).fillna(0).to_numpy()
        Xte = df.loc[te_keys, cols].astype(float).fillna(0).to_numpy()
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), y_tr)
        return m.predict_log_proba(sc.transform(Xte)), list(m.classes_)

    res = {}
    for name, cols in (("surface", surf), ("full", num)):
        lp, classes = fit_eval(cols)
        res[name] = _accuracy(lp, classes, y_te, rng)
    # style centroids
    from ..situations import embed

    etr = embed([t.text for t in train], "style")
    ete = embed([t.text for t in test], "style")
    models.release()
    classes = sorted(set(y_tr))
    cents = np.stack([etr[y_tr == c].mean(0) for c in classes])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    res["style"] = _accuracy(ete @ cents.T * 20, classes, y_te, rng)
    maj = max((y_te == c).mean() for c in classes)
    summary = {"speakers": len(spk), "test_turns": len(test), "chance": round(1 / len(spk), 3), "majority": round(float(maj), 3), **res}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "speakers": spk}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"summary": summary}


def _accuracy(scores: np.ndarray, classes: list[str], y: np.ndarray, rng: random.Random, chunk: int = 10) -> dict:
    pred = np.array(classes)[scores.argmax(1)]
    per_turn = float((pred == y).mean())
    hits, n = 0, 0
    for c in classes:
        idx = np.where(y == c)[0]
        for ch in _chunks(idx, chunk, rng):
            n += 1
            hits += classes[int(scores[ch].sum(0).argmax())] == c
    return {"per_turn": round(per_turn, 3), f"per_{chunk}_turns": round(hits / max(n, 1), 3), "chunks": n}
