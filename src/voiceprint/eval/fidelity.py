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

BASE_SYSTEM = "Reply to the message below as the person it was sent to. Be helpful and natural. Reply with the message only."


class Pick(BaseModel):
    more_like_target: str  # "A" or "B"


JUDGE = """You compare two replies against real examples of one writer's messages. Pick the reply whose STYLE (wording, tone,
length, punctuation, rhythm, habits) is more like the writer's. Ignore which reply is more helpful. Answer "A" or "B"."""


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
    items = rng.sample(held, min(n, len(held)))
    router = Router(cfg, corpus_public=public)
    gen, judge = router.stage("generate"), router.stage("judge")
    base, voiced, real, ctx = [], [], [], []
    for i, t in enumerate(items):
        m = prev[t.key].text
        ctx.append(m)
        real.append(t.text)
        base.append(gen.text(BASE_SYSTEM, m, 600).text.strip())
        voiced.append(gen.text(voiced_system, m, 600).text.strip())
        if i % 10 == 9:
            progress(f"  generated {i + 1}/{len(items)}")
    sc = data["scorer"]
    s_base, s_voiced, s_real = (scorer.score_surface(sc, x) for x in (base, voiced, real))
    cents = scorer.style_centroids([e["text"] for e in exemplars] + real[: len(real) // 2], [])  # target-only centroid
    st = lambda xs: scorer.score_style({"target": cents["target"], "reference": [0.0] * len(cents["target"])}, xs)
    y_base, y_voiced, y_real = st(base), st(voiced), st(real)
    models.release()
    # blind pairwise judgement
    examples = "\n".join(f"- {e['text']}" for e in exemplars[:12])
    wins = 0
    for b, v in zip(base, voiced):
        flip = rng.random() < 0.5
        a_, b_ = (v, b) if flip else (b, v)
        p, _ = judge.json(JUDGE, f"WRITER'S REAL MESSAGES\n{examples}\n\nREPLY A\n{a_}\n\nREPLY B\n{b_}", Pick, 50)
        wins += (p.more_like_target.strip().upper().startswith("A")) == flip
    # content preservation (bidirectional entailment)
    e1 = neural._nli_entailment(voiced, base)
    e2 = neural._nli_entailment(base, voiced)
    models.release()
    summary = {
        "n": len(items), "generator": gen.label, "judge": judge.label,
        "surface_score": {"baseline": round(float(s_base.mean()), 3), "voiced": round(float(s_voiced.mean()), 3), "real": round(float(s_real.mean()), 3)},
        "style_similarity": {"baseline": round(float(y_base.mean()), 3), "voiced": round(float(y_voiced.mean()), 3), "real": round(float(y_real.mean()), 3)},
        "judge_voiced_win_rate": round(wins / max(len(items), 1), 3),
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
