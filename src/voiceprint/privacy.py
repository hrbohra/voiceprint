"""Privacy runs first. Nothing downstream, and nothing sent to any LLM, sees un-anonymised text
unless the frontier-direct tier is explicitly acknowledged.

1. PII replacement with Presidio on the spaCy transformer NER, plus recognisers Presidio lacks
   (social handles, UK phones and postcodes, order and booking references). Placeholders are
   typed and consistent within a conversation: "Sarah" becomes <PERSON_1> every time it appears,
   so "Hi <PERSON_1>!" still teaches that the speaker opens with a name.
2. Residual-risk quarantine: a turn still carrying a high-risk pattern after replacement is dropped
   and counted.
3. k-anonymity for quotes: any n-gram of >= min_ngram tokens seen in fewer than k conversations is
   "rare", and no output may contain one.
4. Memorisation check: every output text is scanned for rare corpus spans before it is written.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .schema import Corpus, Turn

# Patterns Presidio's defaults miss or handle poorly for DMs and UK data.
CUSTOM = {
    "HANDLE": r"(?<![\w@])@[A-Za-z0-9_.]{2,30}\b",
    "UK_PHONE": r"(?:(?:\+44\s?|0044\s?|\(?0)(?:\d\s?){9,10})",
    "UK_POSTCODE": r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
    "REFERENCE": r"\b(?:order|booking|ref|reference|ticket|case)\s*(?:no\.?|number|#|:)?\s*[A-Z0-9-]{5,}\b",
    "LONG_NUMBER": r"\b\d[\d\s-]{7,}\d\b",
}
PRESIDIO_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "LOCATION", "NRP", "DATE_TIME"]
# Kept (not replaced): DATE_TIME and LOCATION are replaced only when specific (a street, a full date);
# generic mentions ("London", "next week") carry register, not identity. See decision D-05.
HIGH_RISK_RESIDUAL = [re.compile(p, re.I) for p in (r"[\w.+-]+@[\w-]+\.[\w.]+", r"\b\d{10,}\b", r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b")]
PLACEHOLDER = re.compile(r"<([A-Z_]+)_(\d+)>|<([A-Z_]+)>")


@dataclass
class PrivacyReport:
    turns_in: int = 0
    turns_out: int = 0
    quarantined: int = 0
    replaced: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict:
        return {"turns_in": self.turns_in, "turns_out": self.turns_out, "quarantined": self.quarantined, "replaced": dict(self.replaced)}


class Anonymiser:
    def __init__(self):
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        from . import models

        name = "en_core_web_trf"
        try:
            models.spacy_nlp()  # warms GPU and confirms which pipeline is installed
            name = models.spacy_nlp().meta.get("lang", "en") + "_" + models.spacy_nlp().meta.get("name", "core_web_trf")
        except Exception:  # noqa: BLE001
            name = "en_core_web_sm"
        engine = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": name}]}).create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=engine)
        for ent, pat in CUSTOM.items():
            registry.add_recognizer(PatternRecognizer(supported_entity=ent, patterns=[Pattern(ent, pat, 0.6)]))
        self.analyzer = AnalyzerEngine(nlp_engine=engine, registry=registry, supported_languages=["en"])
        self.entities = PRESIDIO_ENTITIES + list(CUSTOM)

    @staticmethod
    def _specific(ent: str, span: str) -> bool:
        """Only replace locations and dates when they could identify someone."""
        if ent == "LOCATION":
            return bool(re.search(r"\d|\b(street|st|road|rd|avenue|ave|lane|flat|apt|close|drive)\b", span, re.I))
        if ent == "DATE_TIME":
            return bool(re.search(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b(19|20)\d{2}\b.*\b\d{1,2}\b", span))
        return True

    def run(self, corpus: Corpus) -> tuple[Corpus, PrivacyReport]:
        rep = PrivacyReport(turns_in=len(corpus))
        out: list[Turn] = []
        for doc_id, turns in corpus.docs().items():
            mapping: dict[tuple[str, str], str] = {}
            counters: dict[str, int] = defaultdict(int)
            for t in turns:
                results = self.analyzer.analyze(text=t.text, language="en", entities=self.entities, score_threshold=0.45)
                spans = sorted((r for r in results if self._specific(r.entity_type, t.text[r.start:r.end])), key=lambda r: (r.start, -(r.end - r.start)))
                merged, last_end = [], -1
                for r in spans:  # drop overlaps, keep the earliest-longest
                    if r.start >= last_end:
                        merged.append(r)
                        last_end = r.end
                text = t.text
                for r in reversed(merged):
                    ent = "PERSON" if r.entity_type == "HANDLE" else ("PHONE" if r.entity_type in ("PHONE_NUMBER", "UK_PHONE") else r.entity_type)
                    raw = t.text[r.start:r.end].strip().lower()
                    if ent in ("PERSON",):
                        key = (ent, raw.lstrip("@"))
                        if key not in mapping:
                            counters[ent] += 1
                            mapping[key] = f"<{ent}_{counters[ent]}>"
                        ph = mapping[key]
                    else:
                        ph = f"<{ent}>"
                    text = text[: r.start] + ph + text[r.end:]
                    rep.replaced[ent] += 1
                if any(p.search(text) for p in HIGH_RISK_RESIDUAL):
                    rep.quarantined += 1
                    continue
                out.append(Turn(t.doc_id, t.turn_id, t.speaker, text, t.timestamp, t.reply_to, t.meta))
        rep.turns_out = len(out)
        return Corpus(out, corpus.name), rep


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+|<[a-z_0-9]+>", text.lower())


class RareSpans:
    """Index of n-grams and how many distinct conversations contain each. An n-gram in fewer than
    k conversations is rare: quoting it could identify someone or reproduce private content."""

    def __init__(self, corpus: Corpus, n: int = 5, k: int = 3):
        self.n, self.k = n, k
        docs_per: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for t in corpus.turns:
            toks = _tokens(t.text)
            for i in range(len(toks) - n + 1):
                docs_per[tuple(toks[i : i + n])].add(t.doc_id)
        self.rare = {g for g, d in docs_per.items() if len(d) < k and not all(PLACEHOLDER.fullmatch(x) or len(x) < 3 for x in g)}

    def violations(self, text: str) -> list[str]:
        toks = _tokens(text)
        return [" ".join(toks[i : i + self.n]) for i in range(len(toks) - self.n + 1) if tuple(toks[i : i + self.n]) in self.rare]

    def is_quotable(self, text: str) -> bool:
        return not self.violations(text)


def memorisation_check(outputs: list[str], rare: RareSpans, allowed: set[str] | None = None) -> list[tuple[int, str]]:
    """Every output checked against rare corpus spans. Returns (index, span) for each leak; the run
    fails if this is non-empty. `allowed` exempts exemplars that were deliberately cleared."""
    leaks = []
    for i, o in enumerate(outputs):
        if allowed and o in allowed:
            continue
        for a in sorted(allowed or (), key=len, reverse=True):  # cleared exemplars may be embedded in a larger text
            if a and a in o:
                o = o.replace(a, " ")
        for v in rare.violations(o):
            leaks.append((i, v))
    return leaks
