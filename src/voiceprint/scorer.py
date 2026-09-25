"""Voice scorer: "how much does this text sound like the target?" as a calibrated probability.

Two signals:
  1. A logistic regression over the *surface* features (exact, model-free, so it runs anywhere,
     including the TypeScript port used in Kiki). Trained target-vs-reference with class balance
     and L2, evaluated by grouped cross-validation (whole conversations held out). Its weights,
     scaler and feature list are stored in voice.json, and the TS port reproduces it exactly.
  2. Style-embedding similarity (StyleDistance, trained to encode style and ignore content):
     cosine to the target centroid minus cosine to the reference centroid. Python only.

The combined score is what the eval uses to measure generation fidelity without an LLM judge, and
what Kiki can use to rank or gate drafts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import surface

# Surface features the scorer may use: exact, cheap, and portable to TypeScript.
PORTABLE = [
    "words", "words_per_sentence", "mean_word_length", "long_word_rate", "is_fragment", "exclamations_per_sentence",
    "questions_per_sentence", "ends_with_question", "ends_with_exclamation", "ends_without_punctuation", "ellipses_per_sentence",
    "multi_exclamation", "commas_per_sentence", "dashes_per_sentence", "has_emoji", "emoji_end", "emoji_lead", "emoticons",
    "all_lowercase", "lowercase_sentence_starts", "capitalised_words_rate", "contraction_rate", "uncontracted_forms",
    "hedge_rate", "booster_rate", "intensifier_rate", "discourse_marker_rate", "textese_rate", "pronoun_first_singular",
    "pronoun_first_plural", "pronoun_second", "addresses_by_name_first", "line_breaks",
]


def _transform(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df.reindex(columns=cols).astype(float).fillna(0.0).to_numpy()
    # counts are heavy-tailed: log1p keeps one 400-word message from dominating the fit
    for j, c in enumerate(cols):
        if c in ("words", "line_breaks", "emoticons"):
            x[:, j] = np.log1p(x[:, j])
    return x


def fit(target: pd.DataFrame, reference: pd.DataFrame, seed: int = 7) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    cols = [c for c in PORTABLE if c in target.columns and c in reference.columns]
    X = np.vstack([_transform(target, cols), _transform(reference, cols)])
    y = np.r_[np.ones(len(target)), np.zeros(len(reference))]
    groups = np.r_[target["doc_id"].to_numpy(), reference["doc_id"].to_numpy()]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    aucs = []
    n_groups = len(set(groups))
    if n_groups >= 4 and len(set(y)) == 2:
        for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(Z, y, groups):
            if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
                continue
            m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed).fit(Z[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(Z[te])[:, 1]))
    m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed).fit(Z, y)
    coef = dict(zip(cols, m.coef_[0].round(5).tolist()))
    return {
        "type": "logistic-surface-v1", "features": cols, "log1p": ["words", "line_breaks", "emoticons"],
        "mean": mu.round(6).tolist(), "scale": sd.round(6).tolist(), "coef": m.coef_[0].round(6).tolist(),
        "intercept": float(round(m.intercept_[0], 6)), "cv_auc": float(np.mean(aucs)) if aucs else None,
        "cv_folds": len(aucs), "top_weights": sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:12],
    }


def score_surface(model: dict, texts: list[str]) -> np.ndarray:
    df = pd.DataFrame([surface.turn_features(t) for t in texts])
    X = _transform(df, model["features"])
    Z = (X - np.array(model["mean"])) / np.array(model["scale"])
    logit = Z @ np.array(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-logit))


def style_centroids(target_texts: list[str], reference_texts: list[str]) -> dict:
    from .situations import embed

    t = embed(target_texts, "style").mean(0)
    r = embed(reference_texts, "style").mean(0) if reference_texts else np.zeros_like(t)
    return {"target": t.tolist(), "reference": r.tolist()}


def score_style(centroids: dict, texts: list[str]) -> np.ndarray:
    from .situations import embed

    v = embed(texts, "style")
    t, r = np.array(centroids["target"]), np.array(centroids["reference"])
    t, r = t / (np.linalg.norm(t) + 1e-9), r / (np.linalg.norm(r) + 1e-9)
    return v @ t - (v @ r if np.any(r) else 0)


def combined(model: dict, centroids: dict | None, texts: list[str]) -> np.ndarray:
    s = score_surface(model, texts)
    if not centroids:
        return s
    st = score_style(centroids, texts)
    # the style margin is ~[-0.2, 0.2]; map to a probability-like scale and average with the surface score
    return 0.5 * s + 0.5 * (1 / (1 + np.exp(-st * 20)))


def to_typescript(model: dict | None) -> str:
    """The dependency-free TS scorer (ts/voiceScorer.ts) with this run's lexicons and weights
    injected. Kiki's @kiki/voice imports the generated file; tests/test_ts_parity.py checks that it
    reproduces score_surface."""
    import json
    from importlib.resources import files

    from .features import lexicons, surface as surf

    template = (files("voiceprint") / "ts" / "voiceScorer.ts").read_text(encoding="utf-8")
    lex = {
        "hedges": lexicons.HEDGES, "boosters": lexicons.BOOSTERS, "intensifiers": lexicons.INTENSIFIERS,
        "discourse_markers": lexicons.DISCOURSE_MARKERS, "expandable": lexicons.EXPANDABLE, "textese": lexicons.TEXTESE,
        "contractions_re": lexicons.CONTRACTIONS_RE, "pronouns": {k: sorted(v) for k, v in surf.PRON.items()},
    }
    if model is None:  # library mode: the model is passed at runtime (e.g. read from voice.json)
        model_ts = "export const VOICE_SCORER: VoiceprintModel | null = null;"
    else:
        m = {k: model[k] for k in ("type", "features", "log1p", "mean", "scale", "coef", "intercept", "cv_auc")}
        model_ts = "export const VOICE_SCORER: VoiceprintModel = " + json.dumps(m) + ";"
    return (template
            .replace("/* __LEXICONS__ */", "const LEXICONS = " + json.dumps(lex, ensure_ascii=False) + " as const;")
            .replace("/* __MODEL__ */", model_ts))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
