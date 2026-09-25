"""Run configuration. One YAML file (or defaults) decides the processing tier, which provider and
model each LLM stage uses, the privacy thresholds and the budget. Every run writes the resolved
config into its manifest, so results are reproducible and the data flow is auditable.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Tier(str, Enum):
    local_first = "local-first"  # local anonymise + measure; frontier reads a compressed brief
    hybrid = "hybrid"  # local anonymise; frontier labels every anonymised turn
    frontier_direct = "frontier-direct"  # frontier on raw text; explicit opt-in only


class StageModel(BaseModel):
    provider: Literal["anthropic", "openai", "gemini", "ollama", "none"] = "anthropic"
    model: str = "claude-opus-5"


class LLMConfig(BaseModel):
    """Model per stage: a fast model for high-volume steps, the strongest for merging and judging."""

    # Bulk per-turn labelling (hybrid tier only) is the one high-volume stage on a cheaper model;
    # every reasoning, merging, judging and generating stage uses the strongest model.
    label: StageModel = StageModel(provider="anthropic", model="claude-haiku-4-5")
    induce: StageModel = StageModel(provider="anthropic", model="claude-opus-5")
    merge: StageModel = StageModel(provider="anthropic", model="claude-opus-5")
    judge: StageModel = StageModel(provider="anthropic", model="claude-opus-5")
    generate: StageModel = StageModel(provider="anthropic", model="claude-opus-5")
    extra_judges: list[StageModel] = Field(default_factory=list)  # cross-vendor judging
    budget_gbp: float = 20.0
    cache_dir: str = ".voiceprint/cache"
    max_concurrency: int = 4
    timeout_s: float = 90.0


class PrivacyConfig(BaseModel):
    enabled: bool = True
    k_anonymity: int = 3  # an n-gram of >= min_ngram tokens must appear in >= k conversations to be quotable
    min_ngram: int = 5
    memorisation_ngram: int = 8  # any output sharing an 8-gram with a rare corpus span fails the run
    raw_to_provider_acknowledged: bool = False
    allow_unverified_gemini_tier: bool = False


class Config(BaseModel):
    tier: Tier = Tier.local_first
    target_speakers: list[str] = Field(default_factory=list)
    reference: Literal["peers", "builtin", "file"] = "peers"
    reference_path: str | None = None
    llm: LLMConfig = LLMConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    use_gpu: bool = True
    seed: int = 7
    bootstrap: int = 400
    sample_for_rules: int = 360  # turns the frontier model reads in local-first mode
    exemplars_per_situation: int = 8
    research_models: bool = False  # enables non-commercially-licensed models (e.g. the formality ranker)

    @model_validator(mode="after")
    def _guard_tier(self) -> "Config":
        if self.tier == Tier.frontier_direct and not self.privacy.raw_to_provider_acknowledged:
            raise ValueError(
                "tier 'frontier-direct' sends raw text to an LLM provider. Set "
                "privacy.raw_to_provider_acknowledged: true to confirm you have a zero-retention "
                "agreement and a lawful basis for it."
            )
        return self

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if not path:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
