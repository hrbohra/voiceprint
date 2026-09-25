"""Voice rules: plain-language statements a writer (human or LLM) can follow, each one verified.

Three sources, one format:

1. Statistical rules come straight from the contrast table (no LLM). Every feature whose target CI
   is separated from the reference with a meaningful effect becomes a rule with its numbers.
2. Induced rules come from the LLM reading a *compressed brief*: the statistics above, lexical
   keyness, openers and closers, act habits, situations, and a small stratified sample of
   anonymised, k-anonymous turns. Map over chunks, then one merge call reconciles duplicates and
   conflicts against the statistics (map-reduce; decision D-08).
3. Verification on held-out turns the inducer never saw: a judge labels whether each held-out turn
   by the target, and by the reference, follows the rule. A rule is kept only if the target follows
   it clearly more often than the reference (it is *distinctive*) and often enough for its stated
   strength (it is *true*). The numbers are stored on the rule, so every rule carries its evidence.
"""

from __future__ import annotations

import json
import random
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

Strength = Literal["always", "usually", "often", "sometimes", "rarely", "never"]
STRENGTH_MIN = {"always": 0.9, "usually": 0.7, "often": 0.45, "sometimes": 0.2, "rarely": 0.0, "never": 0.0}
STRENGTH_MAX = {"rarely": 0.2, "never": 0.05}


class RuleDraft(BaseModel):
    """What the LLM writes. Bookkeeping (id, source, verification, kept) is deliberately absent, so a
    model can never fill in its own verification result, and the schema has no free-form objects
    (which some providers' structured-output modes reject)."""

    statement: str = Field(description="One imperative sentence a writer can follow, e.g. 'Open with the reader's first name.'")
    category: Literal["lexical", "syntax", "punctuation", "emoji", "register", "affect", "pragmatics", "dialogue", "structure", "other"] = "other"
    situation: str = Field(default="always", description="'always' or the situation name it applies to")
    strength: Strength = "usually"
    rationale: str = ""
    example: str = Field(default="", description="A short illustrative line in the voice, written fresh, never quoted from data")
    counter_example: str = Field(default="", description="The same content written against the rule")
    feature: str | None = Field(default=None, description="Measured feature this rule relates to, if any")


class DraftList(BaseModel):
    rules: list[RuleDraft]


class Rule(BaseModel):
    id: str = ""
    statement: str = Field(description="One imperative sentence a writer can follow, e.g. 'Open with the reader's first name.'")
    category: Literal["lexical", "syntax", "punctuation", "emoji", "register", "affect", "pragmatics", "dialogue", "structure", "other"] = "other"
    situation: str = Field(default="always", description="'always' or the situation name it applies to")
    strength: Strength = "usually"
    rationale: str = ""
    example: str = Field(default="", description="A short illustrative line in the voice, written fresh, never quoted from data")
    counter_example: str = Field(default="", description="The same content written against the rule")
    feature: str | None = Field(default=None, description="Measured feature this rule relates to, if any")
    source: Literal["stats", "llm", "merged"] = "llm"
    verification: dict = Field(default_factory=dict)
    kept: bool = True


class RuleList(BaseModel):
    rules: list[Rule]


class Judgement(BaseModel):
    follows: list[bool] = Field(description="One entry per numbered message, in order: does it follow the rule?")


# ── 1. statistical rules ─────────────────────────────────────────────────────

FEATURE_TEXT: dict[str, tuple[str, str, str]] = {
    # feature: (category, "more" phrasing, "less" phrasing)
    "words": ("structure", "Write longer messages than typical", "Keep messages short"),
    "words_per_sentence": ("syntax", "Use longer sentences", "Use short sentences"),
    "sentences": ("structure", "Write several sentences per message", "Say it in one or two sentences"),
    "exclamations_per_sentence": ("punctuation", "Use exclamation marks freely", "Avoid exclamation marks"),
    "questions_per_sentence": ("pragmatics", "Ask questions", "Rarely ask questions"),
    "ends_with_question": ("pragmatics", "End messages with a question", "Do not end on a question"),
    "ellipses_per_sentence": ("punctuation", "Trail off with ellipses", "Avoid ellipses"),
    "commas_per_sentence": ("punctuation", "Use comma-rich, flowing sentences", "Keep commas to a minimum"),
    "dashes_per_sentence": ("punctuation", "Use dashes for asides", "Avoid dashes"),
    "has_emoji": ("emoji", "Include an emoji", "Do not use emoji"),
    "emoji_end": ("emoji", "Put emoji at the end of the message", "Do not end with an emoji"),
    "emoji_lead": ("emoji", "Lead with an emoji", "Do not open with an emoji"),
    "all_lowercase": ("register", "Write in all lowercase", "Use standard capitalisation"),
    "contraction_rate": ("register", "Use contractions (I'm, we'll, don't)", "Avoid contractions"),
    "uncontracted_forms": ("register", "Prefer full forms (I am, do not)", "Contract where natural"),
    "hedge_rate": ("pragmatics", "Soften claims with hedges (maybe, I think)", "State things directly without hedging"),
    "booster_rate": ("pragmatics", "Use emphatic boosters (definitely, really)", "Avoid boosters"),
    "intensifier_rate": ("affect", "Intensify (so, super, really)", "Avoid intensifiers"),
    "discourse_marker_rate": ("dialogue", "Use conversational markers (so, well, anyway)", "Avoid filler markers"),
    "textese_rate": ("register", "Use casual chat forms (haha, yeah, cool)", "Avoid chat slang"),
    "pronoun_first_singular": ("register", "Speak as 'I'", "Avoid 'I'"),
    "pronoun_first_plural": ("register", "Speak as 'we'", "Avoid 'we'"),
    "pronoun_second": ("register", "Address the reader as 'you' often", "Use 'you' sparingly"),
    "addresses_by_name_first": ("dialogue", "Open by addressing the reader by name", "Do not open with the reader's name"),
    "opens_with_name": ("dialogue", "Open by addressing the reader by name", "Do not open with the reader's name"),
    "emoji_count": ("emoji", "Use several emoji", "Use emoji sparingly"),
    "lowercase_sentence_starts": ("register", "Start sentences in lowercase", "Capitalise sentence starts"),
    "social_offer": ("pragmatics", "Offer further help", "Do not offer further help unprompted"),
    "social_thank": ("pragmatics", "Thank the other person", "Rarely say thanks"),
    "social_apologise": ("pragmatics", "Apologise readily", "Rarely apologise"),
    "social_greet": ("pragmatics", "Greet the other person", "Skip greetings"),
    "social_reassure": ("pragmatics", "Reassure the other person", "Rarely reassure"),
    "social_empathise": ("pragmatics", "Acknowledge how the other person feels", "Stay matter-of-fact about feelings"),
    "social_close": ("pragmatics", "Close with a sign-off", "Skip sign-offs"),
    "social_request": ("pragmatics", "Ask the other person to do things", "Rarely make requests"),
    "social_refuse": ("pragmatics", "Say no plainly when needed", "Rarely refuse"),
    "formality_fscore": ("register", "Write formally (noun- and preposition-heavy)", "Write informally (verb- and pronoun-heavy)"),
    "sentiment_score": ("affect", "Keep the tone warm and positive", "Keep the tone neutral"),
    "lexical_density": ("syntax", "Pack sentences with content words", "Use light, easy sentences"),
    "imperative_sentence_rate": ("syntax", "Use direct instructions", "Avoid commands"),
    "passive_sentence_rate": ("syntax", "Use the passive voice", "Use the active voice"),
    "length_ratio": ("dialogue", "Reply at greater length than the message received", "Reply more briefly than the message received"),
    "lsm": ("dialogue", "Mirror the other person's style", "Keep your own style regardless of the other person"),
    "asks_back": ("dialogue", "Answer a question with a question back", "Answer without asking back"),
    "uk_spelling": ("lexical", "Use British spelling", "Avoid British spelling"),
}


DIAGNOSTIC_SUFFIXES = ("_p", "_confidence")


def prune_redundant(table: pd.DataFrame, frame: pd.DataFrame | None, threshold: float = 0.8) -> dict[str, str]:
    """Features measuring the same fact (ends_with_question, questions_per_sentence, act_ask) would
    each become a rule. Walk features in distinctiveness order; a feature whose per-turn values
    correlate at |r| >= threshold with an already-kept one is folded into it. Returns
    {folded_feature: kept_feature}.

    Correlation is computed *within* groups (each feature centred on its target and reference means)
    when `frame` has a `_group` column. Pooled correlation is dominated by group membership: any two
    traits the reference never shows look perfectly correlated, which folded "opens with a name"
    into "writes in lowercase" in the planted-rule eval."""
    if frame is None or frame.empty:
        return {}
    cols = [c for c in table.index if c in frame.columns]
    x = frame[cols].astype(float).fillna(0)
    if "_group" in frame.columns:
        x = x - x.groupby(frame["_group"].to_numpy()).transform("mean")
    corr = x.corr().abs().fillna(0)
    kept: list[str] = []
    folded: dict[str, str] = {}
    for c in cols:
        match = next((k for k in kept if corr.loc[c, k] >= threshold), None)
        if match:
            folded[c] = match
        else:
            kept.append(c)
    return folded


ACT_TEXT = {
    "ask": ("Ask questions of the other person", "Rarely ask questions"),
    "ask_yes_no": ("Ask yes/no questions", "Avoid yes/no questions"),
    "answer": ("Answer directly", "Respond with more than a bare answer"),
    "acknowledge": ("Acknowledge what was said before moving on", "Move on without a separate acknowledgement"),
    "backchannel": ("Use short backchannels (ok, got it)", "Avoid bare backchannels"),
    "exclaim": ("Use exclamations", "Avoid exclamations"),
    "say": ("Make plain statements", "Avoid flat statements"),
    "reply_yes": ("Say yes readily", "Rarely give a bare yes"),
    "reply_no": ("Say no plainly", "Rarely give a bare no"),
    "hold": ("Buy time before answering (let me check)", "Answer without holding"),
    "intent": ("State what you will do next", "Rarely announce next steps"),
}


def phrase(feat: str, more: bool) -> tuple[str, str] | None:
    """(category, statement) for a feature and direction, or None if the feature is a measurement
    with no followable phrasing (e.g. the proper-noun rate)."""
    if feat in FEATURE_TEXT:
        cat, up, down = FEATURE_TEXT[feat]
        return cat, up if more else down
    if feat == "emotion_neutral":
        return "affect", "Keep an even, neutral tone" if more else "Let feeling show rather than staying neutral"
    if feat.startswith("emotion_"):
        emo = feat.removeprefix("emotion_")
        return "affect", f"Express {emo}" if more else f"Rarely express {emo}"
    act = feat.removeprefix("act_")
    if feat.startswith("act_") and act in ACT_TEXT:
        up, down = ACT_TEXT[act]
        return "dialogue", up if more else down
    return None


def statistical_rules(table: pd.DataFrame, min_effect: float = 0.3, top: int = 25, frame: pd.DataFrame | None = None,
                      min_abs_diff: float = 0.05) -> list[Rule]:
    """Rules from the contrast table. `frame` (target + reference per-turn features, with a `_group`
    column) enables redundancy pruning. `min_abs_diff` applies to features in [0, 1] (flags,
    shares): a difference under five points is not a voice trait however consistent it is.

    Order matters: candidates are first filtered to features that are significant, large enough and
    *phrasable*, and only then pruned for redundancy. Pruning before the phrasable filter let an
    unphrasable measurement (proper-noun rate) absorb "opens with the reader's name" and drop it."""
    cands = []
    for feat, r in table.iterrows():
        if feat.endswith(DIAGNOSTIC_SUFFIXES) or not r["separated"]:
            continue
        if abs(r["hedges_g"]) < min_effect and abs(r["cliffs_delta"]) < 0.15:
            continue
        bounded = 0 <= r["target_mean"] <= 1 and 0 <= r["reference_mean"] <= 1
        if bounded and abs(r["target_mean"] - r["reference_mean"]) < min_abs_diff and not feat.endswith("_rate"):
            continue
        more = bool(r["target_mean"] > r["reference_mean"])
        ph = phrase(feat, more)
        if ph:
            cands.append((feat, r, more, ph))
    folded = prune_redundant(table.loc[[c[0] for c in cands]], frame)
    also: dict[str, list[str]] = {}
    for f, k in folded.items():
        also.setdefault(k, []).append(f)
    out = []
    for feat, r, more, (cat, stmt) in cands:
        if feat in folded:
            continue
        present = r["target_present"]
        strength: Strength = "usually" if present >= 0.7 else "often" if present >= 0.45 else "sometimes" if present >= 0.2 else "rarely"
        out.append(Rule(
            statement=stmt, category=cat, strength=strength if more else "rarely" if present < 0.2 else strength,
            feature=feat, source="stats",
            rationale=f"target {r['target_mean']:.3g} (95% CI {r['target_ci_lo']:.3g}–{r['target_ci_hi']:.3g}) vs reference {r['reference_mean']:.3g}; Hedges g {r['hedges_g']:.2f}, Cliff's δ {r['cliffs_delta']:.2f}"
                      + (f"; also measured by {', '.join(also[feat])}" if feat in also else ""),
            verification={"statistical": True, "hedges_g": round(float(r["hedges_g"]), 3), "cliffs_delta": round(float(r["cliffs_delta"]), 3)},
        ))
        if len(out) >= top:
            break
    for i, r in enumerate(out):
        r.id = f"S{i + 1:02d}"
    return out


# ── 2. LLM induction and merge ──────────────────────────────────────────────

INDUCE_SYSTEM = """You are a computational linguist who extracts a writer's voice as rules another writer can follow.
You are given measured statistics comparing the TARGET speaker to a REFERENCE, and anonymised example turns
(placeholders like <PERSON_1> replace names; keep them as placeholders). Propose rules that are:
- specific and followable ("Open with the reader's first name, then a one-line thank-you"), never vague ("be friendly");
- about HOW the target writes (form, register, rhythm, pragmatics, dialogue moves), not WHAT facts they state;
- consistent with the statistics; where a rule matches a measured feature, name it in `feature`;
- conditional where the evidence is conditional (set `situation`);
- illustrated with a freshly written example and counter-example. Never copy text from the examples."""

MERGE_SYSTEM = """You merge candidate voice rules into one clean rulebook. Combine duplicates, resolve contradictions using the
statistics (the statistics win), drop rules that are vague, content-specific or unsupported, and keep at most the requested number,
most distinctive first. Keep examples freshly written; never copy data."""


def _fmt_turn(i: int, prev: str | None, text: str) -> str:
    ctx = f"  [replying to: {prev[:220]}]\n" if prev else ""
    return f"{ctx}  ({i}) {text[:600]}"


def induce(llm, brief: str, samples: list[tuple[str | None, str]], chunk: int = 60, max_rules: int = 30) -> list[Rule]:
    candidates: list[Rule] = []
    for s in range(0, len(samples), chunk):
        part = samples[s : s + chunk]
        body = "\n".join(_fmt_turn(i + 1, p, t) for i, (p, t) in enumerate(part))
        user = f"STATISTICS\n{brief}\n\nTARGET TURNS\n{body}\n\nPropose up to 15 rules."
        parsed, _ = llm.json(INDUCE_SYSTEM, user, DraftList, max_tokens=6000)
        candidates.extend(Rule(**d.model_dump(), source="llm") for d in parsed.rules)
    return candidates


def merge(llm, brief: str, candidates: list[Rule], max_rules: int = 30) -> list[Rule]:
    if not candidates:
        return []
    listing = json.dumps([r.model_dump(include={"statement", "category", "situation", "strength", "feature", "example", "counter_example"}) for r in candidates], ensure_ascii=False)
    user = f"STATISTICS\n{brief}\n\nCANDIDATES ({len(candidates)})\n{listing}\n\nReturn at most {max_rules} rules."
    parsed, _ = llm.json(MERGE_SYSTEM, user, DraftList, max_tokens=12000)
    return [Rule(**d.model_dump(), id=f"R{i + 1:02d}", source="merged") for i, d in enumerate(parsed.rules)]


# ── 3. verification on held-out turns ───────────────────────────────────────

JUDGE_SYSTEM = """You check whether messages follow a writing rule. Judge only the rule, not quality. A message the rule's
situation does not apply to counts as NOT following it. Answer with one boolean per message, in order."""


def verify(llm, rules: list[Rule], target_heldout: list[str], reference_heldout: list[str], n: int = 24, seed: int = 7,
           min_margin: float = 0.15) -> list[Rule]:
    """Label n held-out target turns and n reference turns per rule in one shuffled batch (the judge
    cannot tell which is which). Keep a rule if target compliance beats reference by min_margin and,
    for positive strengths, reaches the strength's floor (with 10 points of slack)."""
    rng = random.Random(seed)
    for rule in rules:
        t = rng.sample(target_heldout, min(n, len(target_heldout)))
        r = rng.sample(reference_heldout, min(n, len(reference_heldout))) if reference_heldout else []
        items = [(x, "t") for x in t] + [(x, "r") for x in r]
        rng.shuffle(items)
        body = "\n".join(f"{i + 1}. {x[:500]}" for i, (x, _) in enumerate(items))
        scope = "" if rule.situation == "always" else f" (applies in the situation: {rule.situation})"
        parsed, _ = llm.json(JUDGE_SYSTEM, f"RULE: {rule.statement}{scope}\n\nMESSAGES\n{body}\n\nReturn exactly {len(items)} booleans.", Judgement, max_tokens=1500)
        flags = (parsed.follows + [False] * len(items))[: len(items)]
        tc = [f for f, (_, g) in zip(flags, items) if g == "t"]
        rc = [f for f, (_, g) in zip(flags, items) if g == "r"]
        t_rate = sum(tc) / max(len(tc), 1)
        r_rate = sum(rc) / max(len(rc), 1) if rc else None
        negative = rule.strength in ("rarely", "never")
        if negative:
            ok = t_rate <= STRENGTH_MAX[rule.strength] + 0.1 and (r_rate is None or r_rate - t_rate >= min_margin)
        else:
            ok = t_rate >= STRENGTH_MIN[rule.strength] - 0.1 and (r_rate is None or t_rate - r_rate >= min_margin)
        rule.verification.update({"target_compliance": round(t_rate, 3), "reference_compliance": None if r_rate is None else round(r_rate, 3),
                                  "n_target": len(tc), "n_reference": len(rc), "judge": getattr(llm, "label", "")})
        rule.kept = bool(ok)
    return rules
