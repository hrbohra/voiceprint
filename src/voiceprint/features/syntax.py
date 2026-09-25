"""Syntactic features from the spaCy transformer parse. Per turn.

Includes the Heylighen & Dewaele (2002) formality F-score, computed from part-of-speech rates.
It is the licence-free primary formality measure (decision D-06), so the non-commercial formality
model is only an optional extra."""

from __future__ import annotations

import re
from typing import Iterable

POS_KEYS = ["NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET", "ADP", "INTJ", "CCONJ", "SCONJ", "NUM", "PART"]
CLAUSE_DEPS = {"ccomp", "xcomp", "advcl", "relcl", "acl", "csubj", "csubjpass", "parataxis"}
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def _depth(tok) -> int:
    d = 0
    while tok.head.i != tok.i:
        tok = tok.head
        d += 1
    return d


def doc_features(doc) -> dict[str, float]:
    toks = [t for t in doc if not t.is_space and not t.is_punct]
    n = max(len(toks), 1)
    pos = {k: sum(t.pos_ == k for t in toks) / n for k in POS_KEYS}
    sents = list(doc.sents) or [doc[:]]
    ns = len(sents)
    depths = [max((_depth(t) for t in s), default=0) for s in sents]
    clauses = [1 + sum(t.dep_ in CLAUSE_DEPS or (t.dep_ == "conj" and t.pos_ in ("VERB", "AUX")) for t in s) for s in sents]
    passive = sum(any(t.dep_ in ("nsubjpass", "auxpass", "nsubj:pass", "aux:pass") for t in s) for s in sents)
    imperative = 0
    for s in sents:
        root = s.root
        has_subj = any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in root.children)
        if root.tag_ == "VB" and not has_subj and not s.text.strip().endswith("?"):
            imperative += 1
    interrogative = sum(s.text.strip().endswith("?") for s in sents)
    exclamative = sum(s.text.strip().endswith("!") for s in sents)
    past = sum(t.tag_ in ("VBD", "VBN") for t in toks)
    present = sum(t.tag_ in ("VBP", "VBZ") for t in toks)
    # Heylighen & Dewaele F-score, on percentages; articles approximated by DET, prepositions by ADP.
    p = {k: v * 100 for k, v in pos.items()}
    fscore = ((p["NOUN"] + p["PROPN"]) + p["ADJ"] + p["ADP"] + p["DET"] - p["PRON"] - p["VERB"] - p["ADV"] - p["INTJ"] + 100) / 2
    # Opens by addressing someone by name: a PERSON entity or a proper noun in the first three tokens,
    # followed by punctuation ("Sam! ...", "Hi Sam, ..."). Complements the placeholder check in
    # surface.py, which only fires on anonymised text.
    head = [t for t in doc[:4]]
    opens_name = any(
        (t.ent_type_ == "PERSON" or t.pos_ == "PROPN") and t.i + 1 < len(doc) and doc[t.i + 1].is_punct
        for t in head if t.i <= 2
    ) or bool(re.match(r"^\W*(hi|hey|hello|dear|thanks|thank you)?\W*<PERSON_\d+>", doc.text, re.I))
    f = {f"pos_{k.lower()}": v for k, v in pos.items()}
    f["opens_with_name"] = float(opens_name)
    f.update({
        "lexical_density": sum(t.pos_ in CONTENT for t in toks) / n,
        "formality_fscore": fscore,
        "tree_depth_mean": sum(depths) / ns,
        "tree_depth_max": float(max(depths) if depths else 0),
        "clauses_per_sentence": sum(clauses) / ns,
        "subordination_rate": sum(t.dep_ == "mark" or t.pos_ == "SCONJ" for t in toks) / n,
        "coordination_rate": sum(t.dep_ == "cc" for t in toks) / n,
        "passive_sentence_rate": passive / ns,
        "imperative_sentence_rate": imperative / ns,
        "interrogative_sentence_rate": interrogative / ns,
        "exclamative_sentence_rate": exclamative / ns,
        "modal_rate": sum(t.tag_ == "MD" for t in toks) / n,
        "negation_rate": sum(t.dep_ == "neg" for t in toks) / n,
        "past_tense_share": past / max(past + present, 1),
        "starts_with_conjunction": float(bool(toks) and toks[0].pos_ in ("CCONJ", "SCONJ")),
        "named_entities_rate": len(doc.ents) / n,
    })
    return f


def parse(texts: Iterable[str], batch_size: int = 32):
    from .. import models

    nlp = models.spacy_nlp()
    if not models.low_power():
        return nlp.pipe(texts, batch_size=batch_size)

    def rested():  # low-power mode: parse a batch, rest, repeat (GPU duty cycle, D-51)
        texts_l = list(texts)
        for i in range(0, len(texts_l), batch_size):
            yield from nlp.pipe(texts_l[i : i + batch_size], batch_size=batch_size)
            models.gpu_rest()

    return rested()
