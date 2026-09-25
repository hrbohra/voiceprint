"""Situations: what the speaker is responding to, so rules can be conditional ("when the reader is
upset, ...") instead of averaged across very different moments.

A situation is clustered from the *incoming* turn (what the speaker faces), falling back to the
turn itself when there is no context (prose, first messages). Embeddings: bge-large; clustering:
scikit-learn HDBSCAN (density-based, no k to guess, leaves outliers unassigned), with a KMeans +
silhouette fallback for small corpora where density clustering finds nothing. Clusters are
described with class-based TF-IDF (the BERTopic idea, reimplemented in ~20 lines to avoid its
dependency tree; decision D-03) and optionally named by the LLM from those keywords only.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import models
from .features.surface import words
from .features.lexicons import FUNCTION_WORDS

STOP = set(FUNCTION_WORDS) | {"name", "url", "im", "dont", "thats", "its", "ive", "ill", "hi", "hey", "thanks", "ok", "okay", "yes", "no"}


@dataclass
class Situation:
    id: int
    size: int
    keywords: list[str]
    name: str = ""
    members: list[str] = field(default_factory=list)  # turn keys


def embed(texts: list[str], key: str = "semantic") -> np.ndarray:
    m = models.encoder(key)
    return np.asarray(m.encode(texts, batch_size=models.batch_size(128, 32), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def _ctfidf(docs_by_cluster: dict[int, list[str]], top: int = 8) -> dict[int, list[str]]:
    tf = {c: Counter(w.lower() for d in ds for w in words(d) if len(w) > 2 and w.lower() not in STOP) for c, ds in docs_by_cluster.items()}
    avg = sum(sum(c.values()) for c in tf.values()) / max(len(tf), 1)
    total = Counter()
    for c in tf.values():
        total.update(c)
    out = {}
    for c, cnt in tf.items():
        n = sum(cnt.values()) or 1
        scores = {w: (v / n) * math.log(1 + avg / total[w]) for w, v in cnt.items() if v >= 2 or len(cnt) < 30}
        out[c] = [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top]]
    return out


def cluster(vecs: np.ndarray, seed: int = 7) -> np.ndarray:
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import silhouette_score

    n = len(vecs)
    if n < 12:
        return np.zeros(n, dtype=int)
    labels = HDBSCAN(min_cluster_size=max(5, n // 60), min_samples=3, metric="euclidean").fit_predict(vecs)
    n_clusters = len(set(labels) - {-1})
    if n_clusters >= 2 and (labels == -1).mean() < 0.6:
        return labels
    best, best_s = np.zeros(n, dtype=int), -1.0
    for k in range(2, min(9, n // 5) + 1):
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(vecs)
        s = silhouette_score(vecs, lab)
        if s > best_s:
            best, best_s = lab, s
    return best


def find(keys: list[str], situation_texts: list[str], seed: int = 7) -> tuple[list[Situation], dict[str, int]]:
    vecs = embed(situation_texts)
    labels = cluster(vecs, seed)
    models.release()
    by = {}
    for k, t, lab in zip(keys, situation_texts, labels):
        by.setdefault(int(lab), []).append((k, t))
    kw = _ctfidf({c: [t for _, t in v] for c, v in by.items() if c != -1})
    sits = [Situation(c, len(v), kw.get(c, []), members=[k for k, _ in v]) for c, v in sorted(by.items()) if c != -1]
    sits.sort(key=lambda s: -s.size)
    assign = {k: int(lab) for k, lab in zip(keys, labels)}
    return sits, assign


NAME_SYSTEM = "You name clusters of customer or conversation messages. Reply with names only, as JSON."


def name_with_llm(sits: list[Situation], llm) -> None:
    """Name situations from keywords only (no raw text leaves the machine for this step)."""
    from pydantic import BaseModel

    class Names(BaseModel):
        names: list[str]

    if not sits:
        return
    listing = "\n".join(f"{i}. {', '.join(s.keywords)}" for i, s in enumerate(sits))
    prompt = (f"Each line is the top keywords of a cluster of messages that a speaker was replying to. Give each a short "
              f"situation name (2-5 words, e.g. 'late check-in request', 'thanks after a stay'). Return exactly {len(sits)} names in order.\n\n{listing}")
    parsed, _ = llm.json(NAME_SYSTEM, prompt, Names, max_tokens=800)
    for s, n in zip(sits, parsed.names):
        s.name = n.strip()
