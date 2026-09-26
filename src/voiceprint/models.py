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


def low_power() -> bool:
    """VOICEPRINT_LOW_POWER=1: share the machine politely (decision D-51). Idle CPU priority, 2 CPU
    threads, small GPU batches and a rest after each batch, so a laptop stays usable and cool while a
    long run continues. Slower per batch, but no thermal throttling and no fight with other apps."""
    return os.environ.get("VOICEPRINT_LOW_POWER") == "1"


_thermal = {"checked": 0.0, "temp": None, "rest_read": 0.0, "rest": None, "hot_c": 84, "pause_c": 88}


def _gpu_temp() -> int | None:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001 - no nvidia-smi: no thermal guard, base rest only
        return None


def gpu_rest() -> None:
    """Pause after a GPU batch in low-power mode (D-51, D-53).

    Settings come from `.voiceprint/throttle.json` {"rest_s", "hot_c", "pause_c"} (re-read every 20 s, so speed can
    be tuned mid-run without losing work), else VOICEPRINT_GPU_REST_S, else 0.1 s. A thermal guard
    reads the GPU temperature every 15 s: at >= hot_c (84 °C) the rest grows to 0.5 s, and at >= pause_c (88 °C) the
    run pauses 20 s to cool. The run goes as fast as the temperature allows, never faster."""
    if not low_power():
        return
    import json
    import time

    now = time.monotonic()
    if now - _thermal["rest_read"] > 20:
        _thermal["rest_read"] = now
        try:
            cfg = json.loads(open(".voiceprint/throttle.json", encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            cfg = {}
        _thermal["rest"] = float(cfg.get("rest_s", os.environ.get("VOICEPRINT_GPU_REST_S", "0.1")))
        # Defaults sit just under this laptop GPU's 87 °C target, above which the driver cuts clocks.
        _thermal["hot_c"] = int(cfg.get("hot_c", 84))
        _thermal["pause_c"] = int(cfg.get("pause_c", 88))
    if now - _thermal["checked"] > 15:
        _thermal["checked"] = now
        _thermal["temp"] = _gpu_temp()
    rest, t = _thermal["rest"], _thermal["temp"]
    if t is not None and t >= _thermal["pause_c"]:
        time.sleep(20)
        _thermal["checked"] = 0.0  # re-check straight after cooling
    elif t is not None and t >= _thermal["hot_c"]:
        rest = max(rest, 0.5)
    time.sleep(rest)


def configure(use_gpu: bool = True) -> None:
    _state["use_gpu"] = use_gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if low_power():
        threads = int(os.environ.get("VOICEPRINT_THREADS", "4"))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
        try:
            import torch

            torch.set_num_threads(threads)
        except ImportError:
            pass
        if os.name == "nt":  # IDLE_PRIORITY_CLASS: every interactive app is scheduled first
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE  # untyped, the 64-bit pseudo-handle is truncated and the call fails silently
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x40)
        else:
            os.nice(19)


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
    if low_power():
        return 32 if device() == "cuda" else 8
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
