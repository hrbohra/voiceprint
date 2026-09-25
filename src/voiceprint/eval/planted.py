"""Planted-rule recovery: the test where the right answer is known.

We generate short host/guest conversations from a shared pool of content. The TARGET host's replies
go through style transforms (the planted rules), each applied with a stated probability. REFERENCE
hosts reply with the same content in a plain style. Every turn records which planted rules actually
changed it, so the measured presence can be checked against the true presence.

Two runs:
  planted  the target has the planted rules. Metric: recall, meaning the share of planted rules
           recovered as kept statistical rules (and as verified LLM rules when an LLM is configured).
  null     the same generator, but nothing planted: the target writes exactly like the reference.
           Every rule found here is a false discovery. This is the honest false-positive test; a
           hand-made list of "expected side effects" would not be.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..config import Config
from ..schema import Corpus, Turn

GUEST = [
    "Hi, is the room free from the {d1} to the {d2}?", "Hello! Could I check in a bit late, around {t}?", "Is there a washing machine I can use?",
    "Do you mind if I bring my bike?", "What's the wifi like? I need to work from the flat.", "Thanks so much for having me, it was lovely.",
    "Could you recommend somewhere for breakfast nearby?", "Is it quiet at night? I'm a light sleeper.", "Can I leave my bags after check-out?",
    "Sorry, my train is delayed, I'll be there around {t}.", "Is the neighbourhood easy to walk around?", "Do you have a spare towel?",
]
HOST = [
    "yes the room is free on those dates and you are welcome to stay", "that is fine, I will leave the key in the lockbox for you",
    "there is a washing machine in the kitchen and you are free to use it any time", "you can bring the bike, there is space in the hallway",
    "the wifi is fast and there is a desk in the room", "thank you for staying, it was a pleasure to host you",
    "the cafe on the corner does a very good breakfast and it is open early", "it is a quiet street and the windows are double glazed",
    "you can leave your bags in the hallway until the evening", "no problem, I will be at home all evening so come whenever suits",
    "it is very walkable and the station is ten minutes away", "there are spare towels in the cupboard by the bathroom",
]
NAMES = ["Sam", "Priya", "Tom", "Aisha", "Leo", "Maya", "Jon", "Nia", "Ravi", "Ella"]
CONTRACT = [("you are", "you're"), ("it is", "it's"), ("I will", "I'll"), ("there is", "there's"), ("that is", "that's"), ("there are", "there're")]


def _contract(s: str) -> str:
    for a, b in CONTRACT:
        s = re.sub(rf"\b{a}\b", b, s, flags=re.I)
    return s


def _lower_except(s: str, name: str) -> str:
    return re.sub(rf"\b{name.lower()}\b", name, s.lower())


# (id, description, feature(s) that measure it, probability). Applied in the order that keeps each
# transform intact: content edits first, then the question, then the emoji at the very end, then
# lowercasing (which spares the name so the name rule stays detectable).
PLANTED = [
    ("P4", "Uses contractions", ["contraction_rate"], 0.9),
    ("P5", "Hedges with 'I think'", ["hedge_rate"], 0.6),
    ("P1", "Opens by addressing the reader by name", ["opens_with_name", "addresses_by_name_first"], 0.85),
    ("P6", "Ends with a question offering more help", ["ends_with_question", "questions_per_sentence", "act_ask", "asks_back", "social_offer"], 0.5),
    ("P2", "Ends with an emoji", ["emoji_end", "has_emoji", "emoji_count"], 0.7),
    ("P3", "Writes in lowercase", ["all_lowercase", "lowercase_sentence_starts", "capitalised_words_rate"], 0.8),
]


def _apply(pid: str, s: str, name: str) -> str:
    if pid == "P4":
        return _contract(s)
    if pid == "P5":
        return "I think " + s[0].lower() + s[1:]
    if pid == "P1":
        return f"{name}! {s}"
    if pid == "P6":
        return s.rstrip(".") + ". Anything else you need?"
    if pid == "P2":
        return s.rstrip(".") + " 🙂"
    if pid == "P3":
        return _lower_except(s, name)
    raise KeyError(pid)


def make_corpus(n_docs: int = 120, seed: int = 7, plant: bool = True) -> Corpus:
    rng = random.Random(seed)
    turns = []
    for d in range(n_docs):
        host = "target_host" if d % 3 == 0 else f"host_{d % 5}"
        guest = f"guest_{d}"
        name = rng.choice(NAMES)
        for i in range(rng.randint(2, 4)):
            q = rng.randrange(len(GUEST))
            gtxt = GUEST[q].format(d1=f"{rng.randint(1, 14)}th", d2=f"{rng.randint(15, 28)}th", t=f"{rng.randint(7, 11)}pm")
            turns.append(Turn(f"d{d}", 2 * i, guest, gtxt))
            txt = HOST[q][0].upper() + HOST[q][1:] + "."
            applied = []
            if host == "target_host" and plant:
                for pid, _, _, p in PLANTED:
                    if rng.random() < p:
                        new = _apply(pid, txt, name)
                        if new != txt:
                            applied.append(pid)
                        txt = new
            elif rng.random() < 0.15:  # a little natural variation in every host
                txt = txt.rstrip(".") + "!"
            turns.append(Turn(f"d{d}", 2 * i + 1, host, txt, meta={"planted": applied}))
    return Corpus(turns, "planted" if plant else "null")


def _run_one(corpus: Corpus, cfg: Config, use_llm: bool, progress):
    from .. import pipeline

    others = sorted({t.speaker for t in corpus.turns if t.speaker.startswith("host_")})
    return pipeline.run(corpus, cfg, targets=["target_host"], reference_speakers=others, use_llm=use_llm, public=True,
                        progress=progress, max_reference=100000)


def _llm_match(desc: str, statement: str) -> bool:
    keys = {"P1": ["name"], "P2": ["emoji"], "P3": ["lowercase", "lower-case", "lower case"], "P4": ["contraction", "contract"],
            "P5": ["hedg", "i think", "soften"], "P6": ["question", "anything else", "offer"]}
    return any(k in statement.lower() for k in keys[desc])


def run(out: Path, n_docs: int = 120, use_llm: bool = True, seed: int = 7, progress=print) -> dict:
    from .. import pipeline

    cfg = Config(bootstrap=300, seed=seed)
    corpus = make_corpus(n_docs, seed, plant=True)
    res = _run_one(corpus, cfg, use_llm, progress)
    kept_stats = [r for r in res.data["rules"] if r["source"] == "stats" and r["kept"]]
    found = {r["feature"] for r in kept_stats}
    per_rule = {}
    target_turns = [t for t in corpus.turns if t.speaker == "target_host"]
    for pid, desc, feats, p in PLANTED:
        true_rate = sum(pid in t.meta["planted"] for t in target_turns) / len(target_turns)
        hit = next((f for f in feats if f in found), None)
        c = res.data["contrast"].get(feats[0], {})
        per_rule[pid] = {"rule": desc, "true_presence": round(true_rate, 3), "measured": feats[0],
                         "measured_target": c.get("target_mean"), "measured_reference": c.get("reference_mean"), "recovered_as": hit}
    llm_rules = [r for r in res.data["rules"] if r["source"] != "stats"]
    if llm_rules:
        for pid in per_rule:
            m = [r["statement"] for r in llm_rules if r["kept"] and _llm_match(pid, r["statement"])]
            per_rule[pid]["llm_rule"] = m[0] if m else None
    progress("▸ null control (nothing planted)")
    null = _run_one(make_corpus(n_docs, seed, plant=False), cfg, use_llm, progress)
    null_rules = [r["statement"] for r in null.data["rules"] if r["kept"]]
    summary = {
        "planted": len(PLANTED),
        "recall_statistical": round(sum(v["recovered_as"] is not None for v in per_rule.values()) / len(PLANTED), 3),
        "statistical_rules_total": len(kept_stats),
        "recall_llm": round(sum(bool(v.get("llm_rule")) for v in per_rule.values()) / len(PLANTED), 3) if llm_rules else None,
        "null_false_discoveries": len(null_rules),
        "null_rules": null_rules,
        "scorer_heldout_auc": res.data.get("scorer_eval", {}).get("heldout_auc_surface"),
        "null_scorer_heldout_auc": null.data.get("scorer_eval", {}).get("heldout_auc_surface"),
    }
    out.mkdir(parents=True, exist_ok=True)
    pipeline.write(res, out / "run")
    pipeline.write(null, out / "null")
    (out / "summary.json").write_text(json.dumps({"summary": summary, "per_rule": per_rule,
                                                  "statistical_rules": [(r["feature"], r["statement"]) for r in kept_stats]},
                                                 indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return {"summary": summary, "per_rule": per_rule}
