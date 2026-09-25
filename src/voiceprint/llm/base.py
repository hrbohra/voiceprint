"""One interface for every LLM provider.

Every call in Voiceprint goes through `LLM.text()` or `LLM.json()`, which add, uniformly:
  - a content-addressed cache (provider, model, system, user, schema) so reruns are free and
    reproducible;
  - a cost ledger with a hard budget: a call that would push spend past the budget is refused;
  - retries with backoff on transient errors, and a clean `LLMError` on permanent ones;
  - schema validation: a JSON response that does not validate is retried, then reported.
Adapters only implement `_call()`; they never see the cache or the budget.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output). Anthropic rates from the current price list; others are
# defaults a team should confirm and can override in config. Unknown models fall back to a
# deliberately high rate so the budget errs on the side of stopping early.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    # Gemini 3.x flash list prices were not confirmed at build time; priced at 2.5 Pro so the budget
    # guard errs high rather than low.
    "gemini-3.8-flash": (1.25, 10.0),
    "claude-opus-5.5": (0.0, 0.0),  # session provider: answered in a Claude Code session, not billed per token
    "gemini-3.7-flash": (1.25, 10.0),
    "gemini-3.6-flash": (1.25, 10.0),
    "gemini-3.5-flash": (1.25, 10.0),
}
UNKNOWN_PRICE = (10.0, 50.0)
USD_TO_GBP = 0.79


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


class Refused(LLMError):
    pass


class Unavailable(LLMError):
    """Every retry hit a transient error (overload, rate limit, network): the model, not the request,
    is the problem, so the router may try a sibling model."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        pin, pout = PRICES.get(model, UNKNOWN_PRICE)
        return (self.input_tokens * pin + self.output_tokens * pout) / 1_000_000


@dataclass
class CallResult:
    text: str
    usage: Usage
    model: str
    provider: str
    cached: bool = False


class Ledger:
    """Spend so far, per stage, with a hard budget in GBP. Thread-safe."""

    def __init__(self, budget_gbp: float):
        self.budget_gbp = budget_gbp
        self.spent_gbp = 0.0
        self.by_stage: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def check(self, estimate_gbp: float) -> None:
        with self._lock:
            if self.spent_gbp + estimate_gbp > self.budget_gbp:
                raise BudgetExceeded(
                    f"budget £{self.budget_gbp:.2f} would be exceeded (spent £{self.spent_gbp:.2f}, next call ~£{estimate_gbp:.3f})"
                )

    def add(self, stage: str, model: str, usage: Usage) -> None:
        gbp = usage.cost_usd(model) * USD_TO_GBP
        with self._lock:
            self.spent_gbp += gbp
            s = self.by_stage.setdefault(stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "gbp": 0.0})
            s["calls"] += 1
            s["input_tokens"] += usage.input_tokens
            s["output_tokens"] += usage.output_tokens
            s["gbp"] += gbp

    def report(self) -> dict[str, Any]:
        return {"budget_gbp": self.budget_gbp, "spent_gbp": round(self.spent_gbp, 4), "by_stage": self.by_stage}


class Cache:
    """SQLite, content-addressed. Keys never contain secrets (API keys are not part of the key)."""

    def __init__(self, directory: str | Path):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / "llm_cache.sqlite"
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as db:
            db.execute("create table if not exists c (k text primary key, v text, created real)")

    @staticmethod
    def key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, k: str) -> dict[str, Any] | None:
        with self._lock, sqlite3.connect(self.path) as db:
            row = db.execute("select v from c where k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict[str, Any]) -> None:
        with self._lock, sqlite3.connect(self.path) as db:
            db.execute("insert or replace into c values (?, ?, ?)", (k, json.dumps(v), time.time()))


class Provider:
    """Adapter base. Subclasses implement `_call` and set `name`."""

    name = "base"

    def __init__(self, model: str, timeout_s: float = 90.0):
        self.model = model
        self.timeout_s = timeout_s

    def _call(self, system: str, user: str, max_tokens: int, schema: type[BaseModel] | None) -> CallResult:  # pragma: no cover
        raise NotImplementedError

    def transient(self, err: Exception) -> bool:
        """Whether an error is worth retrying (rate limits, 5xx, network)."""
        return False

    def retry_after(self, err: Exception) -> float | None:
        """Seconds the server asked us to wait, if it said (Retry-After, RetryInfo)."""
        m = re.search(r"retryDelay'?:\s*'?(\d+(?:\.\d+)?)s", str(err)) or re.search(r"retry[- ]after[^\d]{0,5}(\d+(?:\.\d+)?)", str(err), re.I)
        return float(m.group(1)) if m else None

    def quota_rpm(self, err: Exception) -> int | None:
        """Requests-per-minute quota reported with a 429, if any (e.g. Gemini free tier: 5)."""
        if "PerMinute" not in str(err):
            return None
        m = re.search(r"quotaValue'?:\s*'?(\d+)", str(err))
        return int(m.group(1)) if m else None


class Pacer:
    """Minimum spacing between calls per provider/model, learned from the quota a 429 reports. Shared
    by every stage in the process, so five stages cannot each spend the same 5-per-minute quota."""

    _interval: dict[str, float] = {}
    _last: dict[str, float] = {}
    _lock = threading.Lock()

    @classmethod
    def wait(cls, key: str) -> None:
        with cls._lock:
            gap = cls._interval.get(key, 0.0)
            now = time.monotonic()
            ready = cls._last.get(key, 0.0) + gap
            cls._last[key] = max(now, ready)
        if ready > now:
            time.sleep(ready - now)

    @classmethod
    def learn(cls, key: str, rpm: int) -> None:
        with cls._lock:
            cls._interval[key] = max(cls._interval.get(key, 0.0), 60.0 / max(rpm, 1) * 1.05)


class LLM:
    """What the pipeline stages use: a provider plus cache, ledger, retries and validation."""

    def __init__(self, provider: Provider, cache: Cache, ledger: Ledger, stage: str, retries: int = 6):
        self.p = provider
        self.cache = cache
        self.ledger = ledger
        self.stage = stage
        self.retries = retries

    @property
    def model(self) -> str:
        return self.p.model

    def _estimate_gbp(self, system: str, user: str, max_tokens: int) -> float:
        approx_in = (len(system) + len(user)) / 3.5
        return Usage(int(approx_in), max_tokens // 4).cost_usd(self.p.model) * USD_TO_GBP

    def _run(self, system: str, user: str, max_tokens: int, schema: type[BaseModel] | None) -> CallResult:
        schema_sig = json.dumps(schema.model_json_schema(), sort_keys=True) if schema else ""
        k = self.cache.key(self.p.name, self.p.model, system, user, schema_sig, str(max_tokens))
        hit = self.cache.get(k)
        if hit:
            return CallResult(hit["text"], Usage(**hit["usage"]), self.p.model, self.p.name, cached=True)
        self.ledger.check(self._estimate_gbp(system, user, max_tokens))
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            pace_key = f"{self.p.name}/{self.p.model}"
            Pacer.wait(pace_key)
            try:
                res = self.p._call(system, user, max_tokens, schema)
                if schema is not None:
                    schema.model_validate_json(res.text)  # raise now, so a bad answer is retried, not cached
                self.ledger.add(self.stage, self.p.model, res.usage)
                self.cache.put(k, {"text": res.text, "usage": res.usage.__dict__})
                return res
            except (ValidationError, json.JSONDecodeError) as e:
                last = e
            except Refused:
                raise
            except Exception as e:  # noqa: BLE001 - classified by the adapter
                last = e
                if not self.p.transient(e):
                    raise LLMError(f"{self.p.name}/{self.p.model}: {type(e).__name__}: {e}") from e
                if "PerDay" in str(e):  # a daily quota will not recover by waiting: let the router fail over now
                    raise Unavailable(f"{self.p.name}/{self.p.model}: daily quota exhausted") from e
                rpm = self.p.quota_rpm(e)
                if rpm:
                    Pacer.learn(pace_key, rpm)
                server_wait = self.p.retry_after(e)
                if server_wait is not None:
                    time.sleep(min(server_wait + 1.0, 120.0))
                    continue
            time.sleep(min(30.0, 1.5 * 2**attempt))
        msg = f"{self.p.name}/{self.p.model}: gave up after {self.retries + 1} attempts: {last}"
        if isinstance(last, (ValidationError, json.JSONDecodeError)):
            raise LLMError(msg)
        raise Unavailable(msg)

    def text(self, system: str, user: str, max_tokens: int = 2000) -> CallResult:
        return self._run(system, user, max_tokens, None)

    def json(self, system: str, user: str, schema: type[T], max_tokens: int = 8000) -> tuple[T, CallResult]:
        res = self._run(system, user, max_tokens, schema)
        return schema.model_validate_json(res.text), res
