"""The generated TypeScript scorer must reproduce the Python scorer: same features, same score.

Runs the generated file with Node's built-in type stripping (Node >= 22.6). Skipped if Node is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from voiceprint import scorer
from voiceprint.features import surface

CASES = [
    "hey <PERSON_1>! yes that's fine 🙂 anything else?",
    "Yes. That is fine.",
    "Sam! the room's free. anything else you need? 🙂",
    "I think we could maybe do Tuesday... not sure tbh",
    "Thank you so much!! We'll definitely be back :)",
    "🙂 morning! kettle's on",
    "Unfortunately we can't take pets, sorry about that.",
    "the flat is near Camden, see you soon x",
    "OK\nsee you at 7",
    "Honestly it was AMAZING, really lovely stay 👍🏽",
    "Could you let me know when you land? Cheers",
    "Just so you know - the wifi password is on the fridge; it's easy.",
]


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(3)
    t = [f"hey <PERSON_1>! {x} 🙂" for x in ["yes that's fine", "sure thing", "no worries at all", "lovely, see you soon"] * 15]
    r = [x for x in ["Yes. That is fine.", "Thank you for your message.", "We will confirm shortly.", "Please see the details below."] * 15]
    tf = pd.DataFrame([surface.turn_features(x) for x in t]).assign(doc_id=[f"a{i // 3}" for i in range(len(t))])
    rf = pd.DataFrame([surface.turn_features(x) for x in r]).assign(doc_id=[f"b{i // 3}" for i in range(len(r))])
    # jitter so no feature has zero variance in only one class
    return scorer.fit(tf, rf, seed=int(rng.integers(100)))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_typescript_matches_python(model, tmp_path):
    (tmp_path / "voiceScorer.ts").write_text(scorer.to_typescript(model), encoding="utf-8")
    (tmp_path / "run.ts").write_text(
        "import { features, scoreVoice } from './voiceScorer.ts';\n"
        f"const cases: string[] = {json.dumps(CASES, ensure_ascii=False)};\n"
        "console.log(JSON.stringify(cases.map((c) => ({ f: features(c), s: scoreVoice(c) }))));\n",
        encoding="utf-8",
    )
    out = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", "run.ts"], cwd=tmp_path,
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    ts = json.loads(out)
    py_scores = scorer.score_surface(model, CASES)
    for case, got, want in zip(CASES, ts, py_scores):
        pyf = surface.turn_features(case)
        for name in model["features"]:
            assert got["f"][name] == pytest.approx(pyf.get(name, 0.0), abs=1e-9), (case, name)
        assert got["s"] == pytest.approx(want, abs=1e-6), case
