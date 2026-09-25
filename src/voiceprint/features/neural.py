"""Neural per-turn features: sentiment, emotions, dialogue acts, social acts, optional formality.

All run through `_classify`, which batches on the GPU under fp16 autocast, and splits any text longer than the
model window into sentence-aligned chunks whose probabilities are length-weighted, so nothing is
ever truncated (decision D-07). Social acts combine zero-shot entailment with high-precision cue
phrases (decision D-04)."""

from __future__ import annotations

import re

import numpy as np

from .. import models
from .lexicons import SOCIAL_ACT_CUES, SOCIAL_ACT_HYPOTHESES
from .surface import sentences

# Acts whose cue phrases are formulaic enough to decide alone: NLI scores bare "hello," or "I'm afraid
# not" low because they are short or indirect, but the cue lists for these are strict (lexicons.py).
CUE_SUFFICIENT = {"greet", "thank", "apologise", "refuse"}

EMOTION_THRESHOLD = 0.3

MAX_CHARS = 1200  # ~ 300 tokens: safely inside every model's 512-token window, with the pair/hypothesis


def _chunks(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text or "."]
    out, cur = [], ""
    for s in sentences(text):
        while len(s) > MAX_CHARS:  # a single enormous "sentence": split on spaces, never mid-word
            cut = s.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            out.append(s[:cut])
            s = s[cut:].strip()
        if len(cur) + len(s) + 1 > MAX_CHARS and cur:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out or [text[:MAX_CHARS]]


def _classify(key: str, texts: list[str], pairs: list[str] | None = None, activation: str = "softmax") -> np.ndarray:
    """Probabilities [n_texts, n_labels]. `pairs` gives a second segment per text (context or hypothesis)."""
    import torch

    tok, model = models.classifier(key)
    flat, owner, weight = [], [], []
    for i, t in enumerate(texts):
        for c in _chunks(t):
            flat.append((c, pairs[i] if pairs else None))
            owner.append(i)
            weight.append(max(len(c), 1))
    bs = models.batch_size(64, 16)
    probs = []
    half = models.device() == "cuda" and key not in models.FP32_ONLY
    autocast = torch.autocast("cuda", dtype=torch.float16) if half else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), autocast:
        for s in range(0, len(flat), bs):
            part = flat[s : s + bs]
            if pairs is not None:
                enc = tok([p for _, p in part], [c for c, _ in part], truncation="only_first", max_length=512, padding=True, return_tensors="pt")
            else:
                enc = tok([c for c, _ in part], truncation=True, max_length=512, padding=True, return_tensors="pt")
            logits = model(**enc.to(model.device)).logits.float()
            p = torch.sigmoid(logits) if activation == "sigmoid" else torch.softmax(logits, dim=-1)
            probs.append(p.cpu().numpy())
    arr = np.concatenate(probs) if probs else np.zeros((0, model.config.num_labels))
    out = np.zeros((len(texts), arr.shape[1]))
    wsum = np.zeros(len(texts))
    for row, o, w in zip(arr, owner, weight):
        out[o] += row * w
        wsum[o] += w
    return out / np.maximum(wsum, 1)[:, None]


def sentiment(texts: list[str]) -> list[dict[str, float]]:
    p = _classify("sentiment", texts)  # labels: negative, neutral, positive
    return [{"sentiment_negative": a, "sentiment_neutral": b, "sentiment_positive": c, "sentiment_score": c - a} for a, b, c in p]


def emotions(texts: list[str]) -> list[dict[str, float]]:
    _, model = models.classifier("emotion")
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    p = _classify("emotion", texts, activation="sigmoid")
    # Presence at p >= 0.3 (the usual GoEmotions operating point) is the feature; the raw probability is
    # kept as a `_p` diagnostic. Means of raw probabilities (0.004 vs 0.0007) give large effect sizes
    # with no practical meaning, which the planted-rule eval surfaced as false rules.
    out = []
    for row in p:
        d = {}
        for l, v in zip(labels, row):
            d[f"emotion_{l}"] = float(v >= EMOTION_THRESHOLD)
            d[f"emotion_{l}_p"] = float(v)
        out.append(d)
    return out


def formality_model(texts: list[str]) -> list[dict[str, float]]:
    """Research-only (CC BY-NC-SA). Probability the text is formal."""
    p = _classify("formality", texts)
    return [{"formality_model": float(row[0])} for row in p]


def dialogue_acts(texts: list[str], previous: list[str]) -> list[dict[str, float]]:
    """Structural acts from the SILICONE-trained DeBERTa, conditioned on the previous utterance."""
    p = _classify("dialogue_act", texts, pairs=[x or "" for x in previous])
    out = []
    for row in p:
        top = int(row.argmax())
        d = {f"act_{l}": 0.0 for l in models.DIALOGUE_ACT_LABELS}
        d[f"act_{models.DIALOGUE_ACT_LABELS[top]}"] = 1.0
        d["act_confidence"] = float(row[top])
        out.append(d)
    return out


def _nli_entailment(premises: list[str], hypotheses: list[str]) -> np.ndarray:
    """P(entailment) from the full three-way softmax. The common multi-label trick of renormalising
    entailment against contradiction alone discards "neutral" and over-fires on short chat turns
    (measured in the smoke test: a refusal was labelled greet and close), so it is not used."""
    _, model = models.classifier("nli")
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    p = _classify("nli", premises, pairs=hypotheses)
    return p[:, label2id.get("entailment", 0)]


def social_acts(texts: list[str], threshold: float = 0.6, cue_threshold: float = 0.25) -> list[dict[str, float]]:
    """Acts live in sentences, not messages: "Sorry, forgot to mention it. Let me know when you land"
    apologises and requests. Each sentence is scored against each act hypothesis and the message
    takes the max. A matching cue phrase lowers the threshold (cue as prior, model as judge)."""
    acts = list(SOCIAL_ACT_HYPOTHESES)
    sents = [(sentences(t) or [t])[:12] for t in texts]
    owner = [i for i, ss in enumerate(sents) for _ in ss]
    flat = [s for ss in sents for s in ss]
    premises = [s for s in flat for _ in acts]
    hyps = [SOCIAL_ACT_HYPOTHESES[a] for _ in flat for a in acts]
    ent = _nli_entailment(premises, hyps).reshape(len(flat), len(acts)) if flat else np.zeros((0, len(acts)))
    best = np.zeros((len(texts), len(acts)))
    for o, row in zip(owner, ent):
        best[o] = np.maximum(best[o], row)
    cue_res = {a: [re.compile(p, re.I) for p in SOCIAL_ACT_CUES.get(a, [])] for a in acts}
    out = []
    for t, row in zip(texts, best):
        d = {}
        for a, prob in zip(acts, row):
            cue = any(r.search(t) for r in cue_res[a])
            present = prob >= threshold or (cue and (a in CUE_SUFFICIENT or prob >= cue_threshold))
            d[f"social_{a}"] = float(present)
            d[f"social_{a}_p"] = float(prob)
        out.append(d)
    return out
