"""From per-turn features to a voice profile: distributions, confidence, and contrast.

A voice is not "the speaker's mean". It is where the speaker *differs* from comparable writers, how
consistently, and in which situations. So for every feature this module reports:

  - the distribution for the target (mean, median, p10/p90, share of turns where the feature is
    present), with a bootstrap 95% CI resampled *by conversation* (turns in one chat are not
    independent, so resampling turns would overstate confidence);
  - the same for the reference (peers in the corpus, or a built-in reference set);
  - effect size (Cohen's d, Hedges-corrected) and Cliff's delta for skewed features;
  - a distinctiveness ranking combining effect size with its lower confidence bound.

Lexical keyness uses the Monroe, Colaresi & Quinn (2008) log-odds ratio with an informative
Dirichlet prior: stable on small corpora, where raw frequency ratios explode on rare words.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from .features.surface import words

META_COLS = {"speaker", "doc_id"}


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _boot_ci(values: np.ndarray, groups: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of the mean, resampling whole conversations."""
    ok = ~np.isnan(values)
    values, groups = values[ok], groups[ok]
    if len(values) < 3 or n <= 0:
        return (float("nan"), float("nan"))
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    sums = np.array([values[idx[g]].sum() for g in uniq])
    counts = np.array([len(idx[g]) for g in uniq])
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        means.append(sums[pick].sum() / max(counts[pick].sum(), 1))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0 if a.mean() == b.mean() else math.copysign(3.0, a.mean() - b.mean())
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def cliffs_delta(a: np.ndarray, b: np.ndarray, cap: int = 3000) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan")
    rng = np.random.default_rng(0)
    if len(a) > cap:
        a = rng.choice(a, cap, replace=False)
    if len(b) > cap:
        b = rng.choice(b, cap, replace=False)
    bs = np.sort(b)
    gt = np.searchsorted(bs, a, side="left").sum()
    lt = (len(bs) - np.searchsorted(bs, a, side="right")).sum()
    return float((gt - lt) / (len(a) * len(b)))


def describe(df: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    groups = df["doc_id"].to_numpy()
    out = {}
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        vv = v[~np.isnan(v)]
        if not len(vv):
            continue
        lo, hi = _boot_ci(v, groups, n_boot, rng)
        out[c] = {
            "mean": float(vv.mean()), "median": float(np.median(vv)), "p10": float(np.percentile(vv, 10)),
            "p90": float(np.percentile(vv, 90)), "present": float((vv != 0).mean()), "n": int(len(vv)),
            "ci95": [lo, hi],
        }
    return out


def contrast(target: pd.DataFrame, reference: pd.DataFrame, n_boot: int = 400, seed: int = 7) -> pd.DataFrame:
    """One row per feature: target vs reference, effect sizes, and a distinctiveness score."""
    cols = [c for c in numeric_cols(target) if c in reference.columns]
    td, rd = describe(target, cols, n_boot, seed), describe(reference, cols, max(n_boot // 4, 50), seed)
    rows = []
    for c in cols:
        if c not in td or c not in rd:
            continue
        a, b = target[c].to_numpy(dtype=float), reference[c].to_numpy(dtype=float)
        g, cd = hedges_g(a, b), cliffs_delta(a, b)
        t, r = td[c], rd[c]
        # CI of the target mean fully outside the reference mean: the difference is not sampling noise
        separated = not (t["ci95"][0] <= r["mean"] <= t["ci95"][1]) if not math.isnan(t["ci95"][0]) else False
        rows.append({
            "feature": c, "target_mean": t["mean"], "target_ci_lo": t["ci95"][0], "target_ci_hi": t["ci95"][1],
            "target_present": t["present"], "reference_mean": r["mean"], "reference_present": r["present"],
            "hedges_g": g, "cliffs_delta": cd, "separated": separated, "n_target": t["n"], "n_reference": r["n"],
        })
    out = pd.DataFrame(rows).set_index("feature")
    if out.empty:
        return out
    # distinctiveness: robust effect size, zeroed when the difference is inside sampling noise
    out["distinctiveness"] = (out["hedges_g"].abs().fillna(0).clip(upper=3) * 0.6 + out["cliffs_delta"].abs().fillna(0) * 2 * 0.4) * out["separated"].astype(float)
    return out.sort_values("distinctiveness", ascending=False)


def keyness(target_texts: list[str], reference_texts: list[str], top: int = 40, min_count: int = 3) -> dict[str, list[dict]]:
    """Monroe et al. log-odds with informative Dirichlet prior. Returns words and bigrams over- and
    under-used by the target, with z-scores. Placeholders are never reported."""
    def grams(texts):
        c = Counter()
        for t in texts:
            ws = [w.lower() for w in words(t)]
            c.update(ws)
            c.update(f"{a} {b}" for a, b in zip(ws, ws[1:]))
        return c

    ct, cr = grams(target_texts), grams(reference_texts)
    prior = ct + cr
    a0 = sum(prior.values()) or 1
    nt, nr = sum(ct.values()) or 1, sum(cr.values()) or 1
    scale = 500 / a0  # prior strength: comparable to a 500-token pseudo-corpus
    res = []
    for w, pc in prior.items():
        if ct[w] + cr[w] < min_count or w in ("name", "url"):
            continue
        aw = pc * scale
        a_sum = a0 * scale
        lt = math.log((ct[w] + aw) / (nt + a_sum - ct[w] - aw))
        lr = math.log((cr[w] + aw) / (nr + a_sum - cr[w] - aw))
        var = 1 / (ct[w] + aw) + 1 / (cr[w] + aw)
        res.append((w, (lt - lr) / math.sqrt(var), ct[w], cr[w]))
    res.sort(key=lambda x: -x[1])
    fmt = lambda xs: [{"term": w, "z": round(z, 2), "target": a, "reference": b} for w, z, a, b in xs]
    return {"over": fmt([r for r in res if r[1] > 1.96][:top]), "under": fmt([r for r in res[::-1] if r[1] < -1.96][:top])}


def act_profile(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith("_p") and c != "act_confidence"]
    return {c.removeprefix(prefix): float(df[c].mean()) for c in cols}
