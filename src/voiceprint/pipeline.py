"""The extraction run, stage by stage. Each stage is logged, timed and checkpointed in the manifest.

    load → anonymise → split (by conversation) → features → contrast + keyness + dialogue
         → situations → statistical rules → [LLM: induce → merge → verify] → scorer
         → exemplars + SFT/DPO → memorisation check → write

Held-out conversations (20% by default) are never shown to the inducer; verification and the
scorer's reported AUC are measured on them only.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import features, models, profile, rules as rules_mod, scorer, situations
from .config import Config, Tier
from .features import dialogue
from .privacy import Anonymiser, PrivacyReport, RareSpans, memorisation_check
from .schema import Corpus, Turn

log = logging.getLogger("voiceprint")


@dataclass
class RunResult:
    data: dict
    exemplars: list[dict]
    sft: list[dict]
    dpo: list[dict]
    target_features: pd.DataFrame
    reference_features: pd.DataFrame
    heldout: dict = field(default_factory=dict)


def split_docs(turns: list[Turn], frac: float, seed: int) -> tuple[list[Turn], list[Turn]]:
    docs = sorted({t.doc_id for t in turns})
    rng = random.Random(seed)
    rng.shuffle(docs)
    if len(docs) < 5:  # one long document (a book): split its turns in contiguous blocks instead
        n = len(turns)
        blocks = [turns[i : i + max(n // 10, 1)] for i in range(0, n, max(n // 10, 1))]
        rng.shuffle(blocks)
        k = max(1, round(len(blocks) * frac))
        return [t for b in blocks[k:] for t in b], [t for b in blocks[:k] for t in b]
    held = set(docs[: max(1, round(len(docs) * frac))])
    return [t for t in turns if t.doc_id not in held], [t for t in turns if t.doc_id in held]


def cap_by_doc(turns: list[Turn], n: int, seed: int) -> list[Turn]:
    if len(turns) <= n:
        return turns
    rng = random.Random(seed)
    return sorted(rng.sample(turns, n), key=lambda t: (t.doc_id, t.turn_id))


def brief(name: str, table: pd.DataFrame, kw: dict, dial: dict, acts: dict, social: dict, sits: list, top: int = 30) -> str:
    L = [f"Target: {name}. Positive effect = target uses MORE than reference."]
    shown = table[[not f.endswith(rules_mod.DIAGNOSTIC_SUFFIXES) for f in table.index]] if len(table) else table
    for f, r in shown.head(top).iterrows():
        L.append(f"- {f}: target {r['target_mean']:.3g} (present in {r['target_present']:.0%} of turns) vs reference {r['reference_mean']:.3g}; g={r['hedges_g']:.2f}")
    if kw.get("over"):
        L.append("Over-used terms: " + ", ".join(x["term"] for x in kw["over"][:25]))
    if kw.get("under"):
        L.append("Under-used terms: " + ", ".join(x["term"] for x in kw["under"][:15]))
    if dial:
        L.append(f"Openers: {dial['top_openers'][:6]}; closers: {dial['top_closers'][:6]}; multi-message turns {dial['multi_message_turn_rate']:.0%}")
    L.append("Dialogue acts share: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(acts.items(), key=lambda kv: -kv[1]) if v > 0.02))
    L.append("Social acts share: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(social.items(), key=lambda kv: -kv[1]) if v > 0.02))
    if sits:
        L.append("Situations: " + "; ".join(f"{s.name or s.id} ({s.size} turns: {', '.join(s.keywords[:5])})" for s in sits[:10]))
    return "\n".join(L)


def run(corpus: Corpus, cfg: Config, *, targets: list[str] | None = None, reference_speakers: list[str] | None = None, use_llm: bool = True, public: bool = False,
        heldout_frac: float = 0.2, max_reference: int = 20000, progress=print) -> RunResult:
    t0 = time.time()
    timings: dict[str, float] = {}

    def stage(name: str):
        timings[name] = time.time()
        progress(f"▸ {name}")

    def done(name: str):
        timings[name] = round(time.time() - timings[name], 2)

    models.configure(cfg.use_gpu)
    speakers = corpus.speakers()
    targets = targets or cfg.target_speakers or [next(iter(speakers))]
    missing = [t for t in targets if t not in speakers]
    if missing:
        raise ValueError(f"target speaker(s) not in corpus: {missing}. Speakers: {list(speakers)[:20]}")

    # ── privacy ──
    prep = PrivacyReport(turns_in=len(corpus), turns_out=len(corpus))
    if cfg.privacy.enabled and not public:
        stage("anonymise")
        corpus, prep = Anonymiser().run(corpus)
        done("anonymise")
    rare = RareSpans(corpus, n=cfg.privacy.min_ngram, k=cfg.privacy.k_anonymity)
    rare_mem = RareSpans(corpus, n=cfg.privacy.memorisation_ngram, k=cfg.privacy.k_anonymity)

    tset = set(targets)
    target_all = [t for t in corpus.turns if t.speaker in tset and t.text.strip()]
    # reference: named peers when given (e.g. other hosts, not the guests they talk to), else everyone else
    rset = set(reference_speakers) if reference_speakers else None
    ref_all = [t for t in corpus.turns if t.speaker not in tset and t.text.strip() and (rset is None or t.speaker in rset)]
    t_train, t_held = split_docs(target_all, heldout_frac, cfg.seed)
    r_train, r_held = split_docs(ref_all, heldout_frac, cfg.seed + 1) if ref_all else ([], [])
    r_train = cap_by_doc(r_train, max(len(t_train) * 3, 500) if max_reference else len(r_train), cfg.seed)
    r_train = r_train[:max_reference]
    r_held = cap_by_doc(r_held, max(len(t_held) * 2, 200), cfg.seed)
    name = ", ".join(targets)
    progress(f"  target {name}: {len(t_train)} train / {len(t_held)} held-out turns; reference: {len(r_train)} / {len(r_held)}")

    # ── features ──
    stage("features")
    all_turns = t_train + t_held + r_train + r_held
    df = features.extract(corpus, all_turns, research_models=cfg.research_models, progress=lambda m: progress(f"    {m}"))
    keys = lambda ts: [t.key for t in ts]
    tf, th, rf, rh = df.loc[keys(t_train)], df.loc[keys(t_held)], df.loc[keys(r_train)], df.loc[keys(r_held)]
    done("features")

    # ── contrast ──
    stage("contrast")
    has_ref = len(rf) > 0
    table = profile.contrast(tf, rf, cfg.bootstrap, cfg.seed) if has_ref else pd.DataFrame()
    kw = profile.keyness([t.text for t in t_train], [t.text for t in r_train]) if has_ref else {}
    dial = dialogue.speaker_dialogue_summary(corpus, tset)
    prev_map = corpus.previous_turn()
    top_acts = features.top_act(df)
    trans = dialogue.act_transitions([top_acts.get(prev_map[t.key].key) if prev_map.get(t.key) and prev_map[t.key].key in top_acts.index else None for t in t_train], [top_acts[t.key] for t in t_train])
    acts, social = profile.act_profile(tf, "act_"), profile.act_profile(tf, "social_")
    rich = {"target": dialogue_rich(t_train), "reference": dialogue_rich(r_train) if has_ref else {}}
    done("contrast")

    # ── situations ──
    stage("situations")
    sit_texts = [(prev_map[t.key].text if prev_map.get(t.key) else t.text) for t in t_train]
    sits, assign = situations.find(keys(t_train), sit_texts, cfg.seed)
    done("situations")

    router = None
    llm_label = "none"
    if use_llm:
        from .llm.router import Router

        router = Router(cfg, corpus_public=public)
        if not router.any_available():
            progress("  no LLM provider configured: statistical rules only")
            router = None
    if router and sits:
        situations.name_with_llm(sits, router.stage("merge"))

    # ── rules ──
    stage("rules")
    stat_rules = rules_mod.statistical_rules(table, frame=pd.concat([tf.assign(_group=1), rf.assign(_group=0)])) if has_ref else []
    # held-out check for statistical rules: the direction must hold on unseen conversations
    for r in stat_rules:
        if r.feature in th.columns and r.feature in rh.columns and len(th) and len(rh):
            want_more = "more" if table.loc[r.feature, "target_mean"] > table.loc[r.feature, "reference_mean"] else "less"
            got = th[r.feature].mean() > rh[r.feature].mean()
            r.kept = bool(got == (want_more == "more"))
            r.verification.update({"heldout_target_mean": round(float(th[r.feature].mean()), 4), "heldout_reference_mean": round(float(rh[r.feature].mean()), 4)})
    llm_rules: list[rules_mod.Rule] = []
    if router:
        sit_name = {s.id: (s.name or str(s.id)) for s in sits}
        by_sit: dict[int, list[Turn]] = {}
        for t in t_train:
            by_sit.setdefault(assign.get(t.key, -1), []).append(t)
        rng = random.Random(cfg.seed)
        per = max(cfg.sample_for_rules // max(len(by_sit), 1), 5)
        sample = [t for ts in by_sit.values() for t in rng.sample(ts, min(per, len(ts)))][: cfg.sample_for_rules]
        pairs = [((prev_map[t.key].text if prev_map.get(t.key) else None), t.text) for t in sample]
        b = brief(name, table, kw, dial, acts, social, sits)
        inducer = router.stage("induce")
        cands = rules_mod.induce(inducer, b, pairs)
        llm_rules = rules_mod.merge(router.stage("merge"), b, cands)
        judge = router.stage("judge")
        held_t = [t.text for t in t_held] or [t.text for t in t_train[-40:]]
        rules_mod.verify(judge, llm_rules, held_t, [t.text for t in r_held])
        llm_label = f"induce {inducer.label}, judge {judge.label}"
        del sit_name
    all_rules = sorted(llm_rules, key=lambda r: not r.kept) + stat_rules
    done("rules")

    # ── scorer ──
    stage("scorer")
    sc, sc_eval, cents = None, {}, None
    if has_ref:
        sc = scorer.fit(tf, rf, cfg.seed)
        if len(th) and len(rh):
            from sklearn.metrics import roc_auc_score

            p = scorer.score_surface(sc, [t.text for t in t_held] + [t.text for t in r_held])
            y = np.r_[np.ones(len(t_held)), np.zeros(len(r_held))]
            sc_eval["heldout_auc_surface"] = float(roc_auc_score(y, p))
            cents = scorer.style_centroids([t.text for t in t_train][:4000], [t.text for t in r_train][:4000])
            ps = scorer.combined(sc, cents, [t.text for t in t_held] + [t.text for t in r_held])
            sc_eval["heldout_auc_combined"] = float(roc_auc_score(y, ps))
            models.release()
    done("scorer")

    # ── exemplars, SFT, DPO ──
    stage("exemplars")
    exemplars = []
    quotable = [t for t in t_train if 3 <= len(t.text.split()) <= 90 and rare.is_quotable(t.text)
                and (not prev_map.get(t.key) or rare.is_quotable(prev_map[t.key].text))]
    ranked = quotable
    if sc and quotable:
        s = scorer.score_surface(sc, [t.text for t in quotable])
        ranked = [quotable[i] for i in np.argsort(-s)]
    per_sit: dict[int, int] = {}
    for t in ranked:
        sid = assign.get(t.key, -1)
        if per_sit.get(sid, 0) >= cfg.exemplars_per_situation:
            continue
        per_sit[sid] = per_sit.get(sid, 0) + 1
        prev = prev_map.get(t.key)
        exemplars.append({"situation": next((x.name or x.id for x in sits if x.id == sid), None), "context": prev.text if prev else None, "text": t.text})
    sft = [{"messages": ([{"role": "user", "content": prev_map[t.key].text}] if prev_map.get(t.key) else []) + [{"role": "assistant", "content": t.text}], "doc_id": t.doc_id}
           for t in t_train]
    dpo = build_dpo(t_train, r_train, prev_map, cfg.seed) if has_ref and any(prev_map.get(t.key) for t in t_train) else []
    done("exemplars")

    manifest = {
        "tier": cfg.tier.value if isinstance(cfg.tier, Tier) else cfg.tier, "targets": targets, "public_corpus": public,
        "seed": cfg.seed, "models": models.manifest(), "llm": llm_label, "privacy": cfg.privacy.model_dump(),
        "substitutions": router.substitutions if router else [], "cost": {"gbp": router.ledger.spent_gbp if router else 0.0, **(router.ledger.report() if router else {})},
        "timings_s": timings, "total_s": round(time.time() - t0, 1),
    }
    data = {
        "name": name,
        "corpus": {"name": corpus.name, "turns": len(corpus), "speakers": len(speakers), "target_turns": len(target_all), "reference_turns": len(ref_all),
                   "train_target": len(t_train), "heldout_target": len(t_held)},
        "rules": [r.model_dump() for r in all_rules],
        "contrast": table.round(5).to_dict(orient="index") if has_ref else {},
        "target_profile": profile.describe(tf, profile.numeric_cols(tf), 0, cfg.seed) if len(tf) else {},
        "keyness": kw, "dialogue": dial, "act_transitions": trans, "acts": acts, "social_acts": social, "richness": rich,
        "situations": [{"id": s.id, "name": s.name, "size": s.size, "keywords": s.keywords} for s in sits],
        "scorer": sc, "scorer_eval": sc_eval, "privacy": prep.as_dict(), "manifest": manifest,
    }
    # ── memorisation check over every text we publish ──
    from .export import prompt_pack, voice_md

    pack = prompt_pack(name, all_rules, dial, exemplars)
    allowed = {e["text"] for e in exemplars} | {e["context"] for e in exemplars if e["context"]}
    leaks = memorisation_check(list(pack.values()) + [voice_md(name, data)], rare_mem, allowed)
    manifest["memorisation_leaks"] = len(leaks)
    if leaks and not public:
        raise RuntimeError(f"memorisation check failed: {len(leaks)} rare corpus spans in outputs, e.g. {leaks[:3]}")
    return RunResult(data, exemplars, sft, dpo, tf, rf, {"t": th, "r": rh})


def dialogue_rich(turns: list[Turn]) -> dict:
    from .features.surface import richness

    return richness([t.text for t in turns])


def build_dpo(t_train: list[Turn], r_train: list[Turn], prev_map: dict, seed: int, limit: int = 3000) -> list[dict]:
    """chosen = the target's reply; rejected = a reference speaker's reply to the most similar incoming
    message. Same kind of situation, different voice: the pair isolates style."""
    tt = [t for t in t_train if prev_map.get(t.key)][:limit]
    rr = [t for t in r_train if prev_map.get(t.key)]
    if not tt or not rr:
        return []
    ev = situations.embed([prev_map[t.key].text for t in tt])
    rv = situations.embed([prev_map[t.key].text for t in rr])
    models.release()
    best = (ev @ rv.T).argmax(1)
    return [{"prompt": prev_map[t.key].text, "chosen": t.text, "rejected": rr[j].text, "doc_id": t.doc_id} for t, j in zip(tt, best)]


def write(result: RunResult, out: str | Path) -> Path:
    from .export import write_all

    out = Path(out)
    write_all(out, result.data["name"], result.data, result.exemplars, result.sft, result.dpo)
    if result.data.get("scorer"):
        (out / "voiceScorer.ts").write_text(scorer.to_typescript(result.data["scorer"]), encoding="utf-8", newline="\n")
    return out
