"""Model manager: every neural model the pipeline uses, pinned, loaded lazily, one stage at a time.

Stability rules (see BUILD_LOG.md, decision D-07):
  - every Hugging Face model is pinned to an exact revision hash, so an upstream change can never
    silently alter results;
  - models load for the stage that needs them and are released afterwards (`release()`), keeping
    peak GPU memory well under 8 GB;
  - GPU is used when available and falls back to CPU with identical outputs (only slower);
  - text longer than a model's window is split on sentence boundaries by the caller, never cut.
"""

from __future__ import annotations

import gc
import os
from functools import lru_cache

MODELS: dict[str, tuple[str, str]] = {
    # key: (repo, revision)
    "sentiment": ("cardiffnlp/twitter-roberta-base-sentiment-latest", "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"),
    "emotion": ("SamLowe/roberta-base-go_emotions", "d75048347613a25d77de8cf6412eaae9fa7b26be"),
    "dialogue_act": ("diwank/silicone-deberta-pair", "405127a73ef60674b1780e053cd52e538bf7ba7d"),
    "nli": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"),
    "style": ("StyleDistance/styledistance", "b7df5f0b0480773c097ba3121d83ca32b71015ca"),
    "semantic": ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09"),
    # research-only (CC BY-NC-SA): enabled with config.research_models
    "formality": ("s-nlp/roberta-base-formality-ranker", "db9388f420a835ad49c45840adc9850a52240627"),
}

# DeBERTa v1's attention mask fill overflows fp16 in transformers 4.5x; base-size, so fp32 is cheap.
FP32_ONLY = {"dialogue_act"}

DIALOGUE_ACT_LABELS = ["acknowledge", "answer", "backchannel", "reply_yes", "exclaim", "say", "reply_no", "hold", "ask", "intent", "ask_yes_no"]

_state = {"use_gpu": True}


def configure(use_gpu: bool = True) -> None:
    _state["use_gpu"] = use_gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def device() -> str:
    try:
        import torch

        return "cuda" if _state["use_gpu"] and torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache(maxsize=1)
def spacy_nlp():
    """en_core_web_trf on GPU when available; en_core_web_sm otherwise (same outputs, lower accuracy).
    Which one ran is recorded in the manifest via `spacy_name()`."""
    import spacy

    if device() == "cuda":
        try:
            spacy.require_gpu()
            return spacy.load("en_core_web_trf")
        except Exception:  # noqa: BLE001 - fall back cleanly
            pass
    try:
        return spacy.load("en_core_web_trf")
    except OSError:
        return spacy.load("en_core_web_sm")


def spacy_name() -> str:
    nlp = spacy_nlp()
    return f"{nlp.meta.get('lang')}_{nlp.meta.get('name')}-{nlp.meta.get('version')}"


@lru_cache(maxsize=None)
def classifier(key: str):
    """(tokenizer, model) for a sequence-classification model in eval mode. Weights stay fp32; the
    caller runs inference under CUDA autocast, which keeps numerically sensitive ops (softmax,
    layer norm, DeBERTa's relative attention) in fp32. Casting the weights to fp16 broke DeBERTa v1."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo, rev = MODELS[key]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev).to(device()).eval()
    return tok, model


@lru_cache(maxsize=None)
def encoder(key: str):
    from sentence_transformers import SentenceTransformer

    repo, rev = MODELS[key]
    m = SentenceTransformer(repo, revision=rev, device=device())
    if device() == "cuda":
        m = m.half()
    return m


def batch_size(default_gpu: int = 64, default_cpu: int = 16) -> int:
    return default_gpu if device() == "cuda" else default_cpu


def release() -> None:
    """Drop every cached neural model and free GPU memory. Called between stages."""
    classifier.cache_clear()
    encoder.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def manifest() -> dict:
    return {"device": device(), "spacy": spacy_name(), "models": {k: {"repo": r, "revision": v} for k, (r, v) in MODELS.items()}}
