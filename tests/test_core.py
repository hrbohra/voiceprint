"""Fast tests: no neural models, no network. Run with `pytest -m "not models and not llm"`."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from voiceprint import profile, rules, scorer
from voiceprint.eval.planted import PLANTED, make_corpus
from voiceprint.features import dialogue, surface
from voiceprint.ingest import load
from voiceprint.pipeline import split_docs
from voiceprint.privacy import RareSpans, memorisation_check
from voiceprint.schema import Corpus, Turn


# ── surface ──

def test_surface_basics():
    f = surface.turn_features("Hey <PERSON_1>! I'm so happy you're coming 🙂")
    assert f["addresses_by_name_first"] == 1.0
    assert f["emoji_end"] == 1.0
    assert f["contraction_rate"] > 0
    assert f["exclamations_per_sentence"] > 0


def test_casing_and_questions():
    f = surface.turn_features("sure, anything else you need?")
    assert f["all_lowercase"] == 1.0
    assert f["ends_with_question"] == 1.0


def test_mtld_needs_length_and_is_higher_for_richer_text():
    assert math.isnan(surface.mtld(["a"] * 10))
    poor = ("the cat sat on the mat " * 20).split()
    rich = " ".join(f"word{i}" for i in range(120)).split()
    assert surface.mtld(rich) > surface.mtld(poor)


def test_hdd_bounded():
    toks = [f"w{i % 30}" for i in range(200)]
    assert 0 < surface.hdd(toks) <= 42


# ── dialogue ──

def test_lsm_identity_is_max():
    a = "I think we could go to the park and then maybe eat."
    assert dialogue.lsm(a, a) == pytest.approx(1.0, abs=1e-3)
    assert dialogue.lsm(a, "Buy milk.") < 0.9


def test_openers_keep_placeholders():
    assert dialogue.opener("<PERSON_1>! thanks") == "<PERSON_1> thanks"


def test_previous_turn_uses_other_speaker():
    c = Corpus([Turn("d", 0, "a", "hi"), Turn("d", 1, "b", "hello"), Turn("d", 2, "b", "how are you"), Turn("d", 3, "a", "good")])
    prev = c.previous_turn()
    assert prev["d#0"] is None
    assert prev["d#1"].text == "hi"
    assert prev["d#3"].text == "how are you"


# ── profile ──

def _frame(values, speaker, docs):
    return pd.DataFrame({"speaker": speaker, "doc_id": docs, "x": values})


def test_contrast_detects_real_difference_and_ignores_noise():
    rng = np.random.default_rng(0)
    t = _frame(rng.normal(1.0, 1, 300), "t", [f"t{i // 3}" for i in range(300)])
    r = _frame(rng.normal(0.0, 1, 600), "r", [f"r{i // 3}" for i in range(600)])
    t["noise"] = rng.normal(0, 1, 300)
    r["noise"] = rng.normal(0, 1, 600)
    c = profile.contrast(t, r, n_boot=200)
    assert c.loc["x", "separated"] and c.loc["x", "hedges_g"] > 0.7
    assert c.loc["x", "distinctiveness"] > c.loc["noise", "distinctiveness"]


def test_keyness_finds_overused_word():
    t = ["cheers mate that is great"] * 30
    r = ["thank you that is great"] * 30
    k = profile.keyness(t, r)
    assert "cheers" in [x["term"] for x in k["over"]]
    assert "thank" in [x["term"] for x in k["under"]]


def test_cliffs_delta_range():
    a, b = np.arange(10.0), np.arange(10.0) + 100
    assert profile.cliffs_delta(a, b) == -1.0


# ── rules ──

def test_statistical_rules_from_contrast():
    table = pd.DataFrame([{"feature": "has_emoji", "target_mean": 0.8, "target_ci_lo": 0.7, "target_ci_hi": 0.9, "target_present": 0.8,
                           "reference_mean": 0.1, "reference_present": 0.1, "hedges_g": 2.0, "cliffs_delta": 0.7, "separated": True,
                           "n_target": 100, "n_reference": 100, "distinctiveness": 2}]).set_index("feature")
    rs = rules.statistical_rules(table)
    assert rs and rs[0].statement == "Include an emoji" and rs[0].strength == "usually"


# ── scorer ──

def test_scorer_separates_styles_and_exports_ts():
    t = [f"hey <PERSON_1>! yes that's fine 🙂 anything else?" for _ in range(60)]
    r = [f"Yes. That is fine." for _ in range(60)]
    tf = pd.DataFrame([surface.turn_features(x) for x in t]).assign(doc_id=[f"a{i // 3}" for i in range(60)])
    rf = pd.DataFrame([surface.turn_features(x) for x in r]).assign(doc_id=[f"b{i // 3}" for i in range(60)])
    m = scorer.fit(tf, rf)
    p = scorer.score_surface(m, ["hey <PERSON_2>! sure 🙂 anything else?", "Yes. That is correct."])
    assert p[0] > 0.8 > 0.2 > p[1]
    assert "VOICE_SCORER" in scorer.to_typescript(m)


# ── privacy ──

def test_rare_spans_and_memorisation():
    c = Corpus([Turn(f"d{i}", 0, "a", "thanks so much for letting me know") for i in range(5)]
               + [Turn("dx", 0, "a", "my flat is at the corner of elm street and oak road")])
    rs = RareSpans(c, n=5, k=3)
    assert rs.is_quotable("thanks so much for letting me know")
    assert not rs.is_quotable("the corner of elm street and oak")
    leaks = memorisation_check(["the corner of elm street and oak road"], rs)
    assert leaks


# ── split, planted, ingest ──

def test_split_docs_keeps_conversations_whole():
    ts = [Turn(f"d{i}", j, "a", "x") for i in range(20) for j in range(3)]
    tr, te = split_docs(ts, 0.2, 7)
    assert not ({t.doc_id for t in tr} & {t.doc_id for t in te})
    assert len(te) == 12


def test_planted_corpus_has_target_and_rules():
    c = make_corpus(30)
    assert "target_host" in c.speakers()
    assert len(PLANTED) == 6


def test_load_jsonl_and_whatsapp(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps({"doc_id": "x", "speaker": s, "text": t}) for s, t in [("a", "hi"), ("b", "yo")]), encoding="utf-8")
    c = load(p)
    assert [t.turn_id for t in c.turns] == [0, 1]
    w = tmp_path / "chat.txt"
    w.write_text("12/03/2025, 10:01 - Sam: hi there\ncontinued\n12/03/2025, 10:02 - Ella: hey!\n", encoding="utf-8")
    c = load(w, "whatsapp")
    assert c.speakers() == {"Sam": 1, "Ella": 1}
    assert "continued" in c.turns[0].text


@pytest.mark.parametrize("text,expected", [
    ("Sam! the room's free. anything else?", "all_lower"),
    ("the flat is near Camden, see you soon", "all_lower"),
    ("hey there. Really good", "standard"),
    ("Yes. That is fine.", "standard"),
    ("hi <PERSON_1>, all good", "all_lower"),
])
def test_casing_vocative_and_names(text, expected):
    assert surface.casing(text) == expected


def test_question_before_trailing_emoji():
    assert surface.turn_features("anything else you need? 🙂")["ends_with_question"] == 1.0


def test_redundancy_uses_within_group_correlation():
    rng = np.random.default_rng(1)
    n = 200
    # two independent target-only traits: pooled correlation is high, within-group it is ~0
    t = pd.DataFrame({"a": rng.random(n) < 0.8, "b": rng.random(n) < 0.8, "_group": 1}).astype(float)
    r = pd.DataFrame({"a": np.zeros(n), "b": np.zeros(n), "_group": 0.0})
    frame = pd.concat([t, r])
    table = pd.DataFrame({"separated": [True, True]}, index=["a", "b"])
    assert rules.prune_redundant(table, frame) == {}
    # a duplicate measure of the same trait is folded
    frame["c"] = frame["a"]
    table = pd.DataFrame({"separated": [True, True, True]}, index=["a", "b", "c"])
    assert rules.prune_redundant(table, frame) == {"c": "a"}


def test_cli_imports_and_lists_commands():
    from typer.testing import CliRunner

    from voiceprint.cli import app

    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("extract", "score", "eval", "ts-lib", "doctor"):
        assert cmd in res.output


def test_provider_reads_retry_delay_and_quota():
    from voiceprint.llm.base import Provider

    err = Exception("429 RESOURCE_EXHAUSTED {'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '5'}, {'retryDelay': '32s'}")
    p = Provider("m")
    assert p.retry_after(err) == 32.0
    assert p.quota_rpm(err) == 5
    assert p.retry_after(Exception("boom")) is None


def test_batched_verify_keeps_distinctive_rules_only():
    import re as _re

    class FakeJudge:
        label = "fake"
        calls = 0

        def json(self, system, user, schema, max_tokens):
            FakeJudge.calls += 1
            msgs = _re.findall(r"^\d+\. (.*)$", user.split("MESSAGES")[1], _re.M)
            rules_ = _re.findall(r"^R(\d+)\. (.*)$", user, _re.M)
            out = []
            for num, stmt in rules_:
                if "exclamation" in stmt:
                    out.append({"rule": int(num), "follows": ["!" in m for m in msgs]})
                else:  # "Be polite": everyone complies, so it is not distinctive
                    out.append({"rule": int(num), "follows": [True] * len(msgs)})
            return schema.model_validate({"judgements": out}), None

    target = [f"great news {i}!" for i in range(30)]
    ref = [f"noted {i}." for i in range(30)]
    rs = [rules.Rule(statement="Use exclamation marks"), rules.Rule(statement="Be polite")]
    rules.verify(FakeJudge(), rs, target, ref)
    assert FakeJudge.calls == 1
    assert rs[0].kept and rs[0].verification["target_compliance"] == 1.0 and rs[0].verification["reference_compliance"] == 0.0
    assert not rs[1].kept  # true of the target but not distinctive


def test_verify_recalibrates_overstated_strength():
    class Judge:
        label = "fake"

        def json(self, system, user, schema, max_tokens):
            import re as _re

            msgs = _re.findall(r"^\d+\. (.*)$", user.split("MESSAGES")[1], _re.M)
            return schema.model_validate({"judgements": [{"rule": 1, "follows": ["🙂" in m for m in msgs]}]}), None

    target = [f"ok {i} 🙂" if i % 10 < 6 else f"ok {i}" for i in range(40)]  # 60% of target turns
    ref = [f"fine {i}." for i in range(40)]
    rule = rules.Rule(statement="End on 🙂", strength="usually")
    rules.verify(Judge(), [rule], target, ref)
    assert rule.kept and rule.strength in ("often", "usually")
