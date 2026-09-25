"""Reproduce the published evaluation on public corpora.

    python scripts/public_evals.py [twcs|books|curve|extract|fidelity|all]

Corpora (not redistributed; fetch them yourself, see RESULTS.md):
  data/public/twcs.csv          Customer Support on Twitter (ThoughtVector, CC BY-NC-SA 4.0), HF mirror
                                SunidhiSriram/twcs @ b03fa0a7. First 700k rows (memory: 16 GB machine).
  data/public/gutenberg/        23 public-domain novels by 8 authors, named <author>__<gutenberg id>.txt
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from voiceprint.config import Config
from voiceprint.eval import attribution, curve, fidelity
from voiceprint.ingest import english_only, load, load_twitter_support, without_signatures
from voiceprint.schema import Corpus

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data" / "public", ROOT / "results"
TWCS_ROWS = 700_000
BRAND = "AmazonHelp"
# Test runs: frontier stages are answered in the Claude Code session (configs/session.yaml) unless overridden.
CFG = Config.load(os.environ.get("VOICEPRINT_CONFIG", str(ROOT / "configs" / "session.yaml")))


def twcs() -> Corpus:
    c = english_only(load_twitter_support(DATA / "twcs.csv", max_rows=TWCS_ROWS))
    # tweets that are only a link or a mention carry no voice
    return Corpus([t for t in c.turns if len(t.text.split()) >= 2], "twcs")


def main(what: str) -> None:
    out = {}
    if what in ("twcs", "all"):
        out["twcs_attribution"] = attribution.run(twcs(), RESULTS / "attribution_twcs", n_speakers=10, min_turns=400, max_turns=400)["summary"]
    if what in ("twcs_nosig", "all"):  # ablation: does brand attribution survive removing agent sign-offs?
        out["twcs_attribution_no_signatures"] = attribution.run(without_signatures(twcs()), RESULTS / "attribution_twcs_nosig",
                                                                n_speakers=10, min_turns=400, max_turns=400)["summary"]
    if what in ("books", "all"):
        out["books_attribution"] = attribution.run(load(DATA / "gutenberg"), RESULTS / "attribution_books", n_speakers=8, min_turns=400, max_turns=400)["summary"]
    if what in ("curve", "all"):
        out["curve"] = curve.run(twcs(), BRAND, RESULTS / "curve_twcs", [25, 50, 100, 200, 400, 800, 1600])["summary"]
    if what in ("extract", "all"):
        from voiceprint import pipeline

        c = twcs()
        brands = [s for s, n in c.speakers().items() if s != "customer"][:12]
        res = pipeline.run(c, CFG, targets=[BRAND], reference_speakers=[b for b in brands if b != BRAND], max_target=2000, public=True, max_reference=6000)
        pipeline.write(res, RESULTS / f"voice_{BRAND}")
        out["extract"] = {"rules_kept": sum(r["kept"] for r in res.data["rules"]), "scorer_eval": res.data["scorer_eval"], "cost": res.data["manifest"]["cost"]}
    if what in ("fidelity", "all"):
        out["fidelity"] = fidelity.run(RESULTS / f"voice_{BRAND}", twcs(), BRAND, RESULTS / "fidelity_twcs", n=40, public=True, cfg=CFG)["summary"]
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
