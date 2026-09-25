"""Data efficiency: how many turns does a voice need?

Features for the full target sample are computed once; then for growing prefixes of the target's
conversations we recompute the contrast and measure, against the full-data profile:
  - rank agreement of the top distinctive features (Spearman over the full top-30);
  - overlap of the statistical rule set (Jaccard);
  - held-out AUC of the surface scorer trained on the prefix (same held-out set throughout).
The claim "works on a small corpus, better with volume" is this curve.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .. import features, profile, rules, scorer
from ..pipeline import split_docs
from ..schema import Corpus


def run(corpus: Corpus, speaker: str, out: Path, sizes: list[int], public: bool = True, seed: int = 7, progress=print) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    tgt = [t for t in corpus.turns if t.speaker == speaker and t.text.strip()]
    # only as many target turns as the largest size needs (plus a test share), sampled by conversation
    rng0 = random.Random(seed)
    if len(tgt) > int(max(sizes) * 1.3):
        docs = sorted({t.doc_id for t in tgt})
        rng0.shuffle(docs)
        want, keep, n = int(max(sizes) * 1.3), set(), 0
        by = {}
        for t in tgt:
            by.setdefault(t.doc_id, 0)
            by[t.doc_id] += 1
        for d in docs:
            if n >= want:
                break
            keep.add(d)
            n += by[d]
        tgt = [t for t in tgt if t.doc_id in keep]
    ref = [t for t in corpus.turns if t.speaker != speaker and t.text.strip()]
    rng = random.Random(seed)
    ref = rng.sample(ref, min(len(ref), max(sizes) * 2, 4000))
    t_tr, t_te = split_docs(tgt, 0.2, seed)
    r_tr, r_te = split_docs(ref, 0.2, seed + 1)
    progress(f"curve: {speaker}: {len(t_tr)} train, {len(t_te)} test; reference {len(r_tr)} / {len(r_te)}")
    df = features.extract(corpus, t_tr + t_te + r_tr + r_te, progress=lambda m: progress(f"    {m}"))
    k = lambda ts: [t.key for t in ts]
    rf, th, rh = df.loc[k(r_tr)], df.loc[k(t_te)], df.loc[k(r_te)]
    full_tf = df.loc[k(t_tr)]
    full = profile.contrast(full_tf, rf, 200, seed)
    full_top = list(full.index[:30])
    full_rules = {r.statement for r in rules.statistical_rules(full, frame=pd.concat([full_tf.assign(_group=1), rf.assign(_group=0)]))}
    rng.shuffle(t_tr)
    rows = []
    y = np.r_[np.ones(len(th)), np.zeros(len(rh))]
    for n in [s for s in sizes if s <= len(t_tr)] + [len(t_tr)]:
        sub = df.loc[k(t_tr[:n])]
        c = profile.contrast(sub, rf, 200, seed)
        common = [f for f in full_top if f in c.index]
        rho = spearmanr([full.loc[f, "hedges_g"] for f in common], [c.loc[f, "hedges_g"] for f in common]).statistic if len(common) > 3 else float("nan")
        sr = {r.statement for r in rules.statistical_rules(c, frame=pd.concat([sub.assign(_group=1), rf.assign(_group=0)]))}
        jac = len(sr & full_rules) / max(len(sr | full_rules), 1)
        m = scorer.fit(sub, rf, seed)
        auc = roc_auc_score(y, scorer.score_surface(m, [t.text for t in t_te] + [t.text for t in r_te]))
        rows.append({"turns": n, "effect_rank_spearman": round(float(rho), 3), "rule_jaccard_vs_full": round(jac, 3), "heldout_auc": round(float(auc), 3), "rules": len(sr)})
        progress(f"  n={n}: rho={rows[-1]['effect_rank_spearman']} jaccard={rows[-1]['rule_jaccard_vs_full']} auc={rows[-1]['heldout_auc']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"summary": rows, "speaker": speaker}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"summary": rows}
