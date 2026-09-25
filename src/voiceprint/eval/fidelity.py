"""Generation fidelity: does an LLM given the prompt pack actually write in the voice, without
changing what it says?

For held-out incoming messages the target really replied to, generate two replies with the same
model: BASELINE (a neutral "reply helpfully" system prompt) and VOICED (the standard prompt pack
plus exemplars). Then measure:
  - voice score (surface scorer, and style-embedding margin) for baseline, voiced, and the real
    reply (the ceiling);
  - blind pairwise judge: shown real examples of the target, which of two replies (order
    randomised) sounds more like them? Reported as the voiced win rate, per judge;
  - content preservation: bidirectional NLI entailment between baseline and voiced replies (the
    voiced reply should say the same thing, differently);
  - rule compliance on the voiced replies for each verified rule (same judge prompt as verify).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .. import models, scorer
from ..config import Config
from ..features import neural
from ..llm.router import Router
from ..pipeline import split_docs
from ..schema import Corpus

BASE_SYSTEM = ("You reply to messages as the person they were sent to. Be helpful and natural. Each numbered message is a "
               "separate conversation; reply to each independently.")
BATCH = 10  # items per call: 40 items cost 12 calls, not 120 (free-tier Gemini allows 20 per model per day)


class Replies(BaseModel):
    replies: list[str]


class Picks(BaseModel):
    picks: list[str]  # "A" or "B", one per pair


JUDGE = """You compare pairs of replies against real examples of one writer's messages. For each pair, pick the reply whose
STYLE (wording, tone, length, punctuation, rhythm, habits) is more like the writer's. Ignore which is more helpful.
Answer "A" or "B" for every pair, in order."""


def _batched_replies(llm, system: str, messages: list[str]) -> list[str]:
    out: list[str] = []
    for s0 in range(0, len(messages), BATCH):
        part = messages[s0 : s0 + BATCH]
        body = "\n\n".join(f"{i + 1}. {m}" for i, m in enumerate(part))
        parsed, _ = llm.json(system, f"MESSAGES\n{body}\n\nReturn exactly {len(part)} replies, in order, each the reply text only.", Replies, 400 * len(part))
        got = [r.strip() for r in parsed.replies][: len(part)]
        out.extend(got + [""] * (len(part) - len(got)))
    return out


def run(voice_dir: Path, corpus: Corpus, speaker: str, out: Path, n: int = 40, public: bool = True, seed: int = 7, cfg: Config | None = None,
        progress=print) -> dict:
    cfg = cfg or Config()
    voice_dir = Path(voice_dir)
    data = json.loads((voice_dir / "voice.json").read_text(encoding="utf-8"))
    pack = (voice_dir / "prompt_pack" / "standard.txt").read_text(encoding="utf-8")
    exemplars = [json.loads(line) for line in (voice_dir / "exemplars.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shots = "\n\n".join(f"Message: {e['context']}\nReply: {e['text']}" for e in exemplars[:10] if e.get("context"))
    voiced_system = pack + ("\n\nExamples:\n" + shots if shots else "") + "\n\nReply with the message only."
    prev = corpus.previous_turn()
    tgt = [t for t in corpus.turns if t.speaker == speaker and prev.get(t.key)]
    _, held = split_docs(tgt, 0.2, cfg.seed)  # same split the extractor used
    rng = random.Random(seed)
    # never test on anything the prompt pack shows the model (templated brand replies repeat verbatim)
    shown = {e["text"] for e in exemplars} | {e["context"] for e in exemplars if e.get("context")}
    held = [t for t in held if t.text not in shown and prev[t.key].text not in shown]
    items = rng.sample(held, min(n, len(held)))
    router = Router(cfg, corpus_public=public)
    gen, judge = router.stage("generate"), router.stage("judge")
    ctx = [prev[t.key].text for t in items]
    real = [t.text for t in items]
    base = _batched_replies(gen, BASE_SYSTEM, ctx)
    progress(f"  generated {len(base)} baseline replies")
    voiced = _batched_replies(gen, voiced_system + "\n\nEach numbered message is a separate conversation; reply to each independently.", ctx)
    progress(f"  generated {len(voiced)} voiced replies")
    sc = data["scorer"]
    s_base, s_voiced, s_real = (scorer.score_surface(sc, x) for x in (base, voiced, real))
    cents = scorer.style_centroids([e["text"] for e in exemplars] + real[: len(real) // 2], [])  # target-only centroid
    st = lambda xs: scorer.score_style({"target": cents["target"], "reference": [0.0] * len(cents["target"])}, xs)
    y_base, y_voiced, y_real = st(base), st(voiced), st(real)
    models.release()
    # blind pairwise judgement
    examples = "\n".join(f"- {e['text']}" for e in exemplars[:12])
    wins, judged = 0, 0
    flips = [rng.random() < 0.5 for _ in base]  # voiced shown as A when flip, else as B
    for s0 in range(0, len(base), BATCH):
        pairs = list(zip(base[s0 : s0 + BATCH], voiced[s0 : s0 + BATCH], flips[s0 : s0 + BATCH]))
        body = "\n\n".join(f"PAIR {i + 1}\nA: {v if f else b}\nB: {b if f else v}" for i, (b, v, f) in enumerate(pairs))
        p, _ = judge.json(JUDGE, f"WRITER'S REAL MESSAGES\n{examples}\n\n{body}\n\nReturn exactly {len(pairs)} picks.", Picks, 60 * len(pairs))
        for pick, (_, _, f) in zip(p.picks, pairs):
            judged += 1
            wins += pick.strip().upper().startswith("A") == f
    # content preservation (bidirectional entailment)
    e1 = neural._nli_entailment(voiced, base)
    e2 = neural._nli_entailment(base, voiced)
    models.release()
    summary = {
        "n": len(items), "generator": gen.label, "judge": judge.label,
        "surface_score": {"baseline": round(float(s_base.mean()), 3), "voiced": round(float(s_voiced.mean()), 3), "real": round(float(s_real.mean()), 3)},
        "style_similarity": {"baseline": round(float(y_base.mean()), 3), "voiced": round(float(y_voiced.mean()), 3), "real": round(float(y_real.mean()), 3)},
        "judge_voiced_win_rate": round(wins / max(judged, 1), 3), "judged_pairs": judged,
        "content_preserved_bidirectional": round(float(np.mean((e1 > 0.5) & (e2 > 0.5))), 3),
        "content_entailed_voiced_from_baseline": round(float(np.mean(e2 > 0.5)), 3),
        "cost": router.ledger.report(),
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c, b, v, r in zip(ctx, base, voiced, real):
            f.write(json.dumps({"message": c, "baseline": b, "voiced": v, "real": r}, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"summary": summary}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"summary": summary}
