"""Builds the LLM for each pipeline stage from config, with three safety rules applied here, once:

1. Provider fallback: if the configured provider has no credentials, use the next available one in
   the order anthropic > openai > gemini > ollama, and record the substitution in the manifest.
2. Refusal fallback: a refused Claude call is retried once on the prior Opus model.
3. Data-flow guards: raw (non-anonymised) text may only go to a provider on the frontier-direct
   tier with explicit acknowledgement, and private corpora never go to a Gemini key that cannot be
   confirmed as paid tier (the free tier may use submitted content to improve products).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config, StageModel, Tier
from .base import LLM, Cache, CallResult, Ledger, LLMError, Refused, Unavailable
from .providers import available, make_provider

ORDER = ["anthropic", "openai", "gemini", "ollama"]
DEFAULT_MODEL = {"anthropic": "claude-opus-5", "openai": "gpt-5", "gemini": "gemini-3.8-flash", "ollama": "qwen2.5:7b-instruct"}
REFUSAL_FALLBACK = {"claude-opus-5": "claude-opus-4-8", "claude-sonnet-5": "claude-opus-4-8"}
# Sustained overload (every retry a 429/5xx) moves to a sibling model of the same provider, so data
# never crosses to a provider the config did not choose. Each hop is recorded in the manifest.
OVERLOAD_FALLBACK = {
    "claude-opus-5": ["claude-opus-4-8"],
    "claude-haiku-4-5": ["claude-sonnet-5"],
    "gemini-3.8-flash": ["gemini-3.7-flash", "gemini-3.6-flash"],
    "gemini-3.7-flash": ["gemini-3.8-flash", "gemini-3.6-flash"],
    "gemini-3.6-flash": ["gemini-3.8-flash", "gemini-3.7-flash"],
}


@dataclass
class Router:
    cfg: Config
    corpus_public: bool = False
    substitutions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cache = Cache(self.cfg.llm.cache_dir)
        self.ledger = Ledger(self.cfg.llm.budget_gbp)
        self.avail = available()

    def any_available(self) -> bool:
        return any(self.avail.values())

    def _resolve(self, sm: StageModel) -> tuple[str, str]:
        if sm.provider == "none":
            raise LLMError("LLM stages are disabled in this config")
        if self.avail.get(sm.provider):
            return sm.provider, sm.model
        for p in ORDER:
            if self.avail.get(p):
                self.substitutions.append(f"{sm.provider}/{sm.model} unavailable; using {p}/{DEFAULT_MODEL[p]}")
                return p, DEFAULT_MODEL[p]
        raise LLMError("no LLM provider is configured (set ANTHROPIC_API_KEY, or run with --no-llm)")

    def stage(self, name: str, *, raw_text: bool = False) -> "StageLLM":
        sm: StageModel = getattr(self.cfg.llm, name)
        provider, model = self._resolve(sm)
        if raw_text and not (self.cfg.tier == Tier.frontier_direct and self.cfg.privacy.raw_to_provider_acknowledged):
            raise LLMError("refusing to send non-anonymised text to a provider outside the frontier-direct tier")
        if provider == "gemini" and not self.corpus_public and not self.cfg.privacy.allow_unverified_gemini_tier:
            raise LLMError(
                "refusing to send a private corpus to Gemini: the free tier may use submitted content. "
                "Use a paid key and set privacy.allow_unverified_gemini_tier: true, or use another provider."
            )
        llm = LLM(make_provider(provider, model, self.cfg.llm.timeout_s), self.cache, self.ledger, name)
        return StageLLM(llm, self)


@dataclass
class StageLLM:
    llm: LLM
    router: Router

    @property
    def label(self) -> str:
        return f"{self.llm.p.name}/{self.llm.model}"

    def _sibling(self, model: str, why: str) -> LLM:
        self.router.substitutions.append(f"{why} on {self.llm.p.name}/{self.llm.model}; using {model}")
        return LLM(make_provider(self.llm.p.name, model, self.router.cfg.llm.timeout_s), self.router.cache, self.router.ledger, self.llm.stage)

    def _call(self, method: str, *args):
        try:
            return getattr(self.llm, method)(*args)
        except Refused:
            alt = REFUSAL_FALLBACK.get(self.llm.model) if self.llm.p.name == "anthropic" else None
            if not alt:
                raise
            return getattr(self._sibling(alt, "refusal"), method)(*args)
        except Unavailable:
            for alt in OVERLOAD_FALLBACK.get(self.llm.model, []):
                try:
                    fb = self._sibling(alt, "overloaded")
                    out = getattr(fb, method)(*args)
                    self.llm = fb  # stay on the sibling for the rest of this stage
                    return out
                except LLMError:  # overloaded too, or retired for this key (404): try the next sibling
                    continue
            raise

    def text(self, system: str, user: str, max_tokens: int = 2000) -> CallResult:
        return self._call("text", system, user, max_tokens)

    def json(self, system: str, user: str, schema, max_tokens: int = 8000):
        return self._call("json", system, user, schema, max_tokens)
