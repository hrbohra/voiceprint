"""The `session` provider: an interactive frontier model answers each LLM call through files.

Each call writes `<dir>/requests/<id>.json` ({system, user, schema, max_tokens}) and waits for
`<dir>/responses/<id>.json` ({"text": ...}). An operator, in practice a Claude model driving a
Claude Code session, reads the request and writes the response. Everything else is unchanged:
same prompts, same JSON schema validation, same cache and manifest. A response that fails
validation is consumed and the request is re-issued, so a wrong answer is corrected, never cached.

Why it exists: for testing without an API key, or when API quotas are exhausted, the frontier stages
are still run by a frontier model, not by a local stand-in. The manifest records the provider as
`session/<model>` so no result can pass for an API run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .base import CallResult, LLMError, Provider, Usage


class SessionProvider(Provider):
    name = "session"

    def __init__(self, model: str = "claude-opus-5.5", timeout_s: float = 90.0, directory: str | None = None,
                 wait_s: float | None = None):
        super().__init__(model, timeout_s)
        self.dir = Path(directory or os.environ.get("VOICEPRINT_SESSION_DIR", ".voiceprint/session"))
        (self.dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.dir / "responses").mkdir(parents=True, exist_ok=True)
        (self.dir / "answered").mkdir(parents=True, exist_ok=True)
        self.wait_s = wait_s if wait_s is not None else float(os.environ.get("VOICEPRINT_SESSION_WAIT_S", 6 * 3600))

    def _call(self, system, user, max_tokens, schema):
        from .base import Cache

        rid = Cache.key(self.model, system, user, json.dumps(schema.model_json_schema(), sort_keys=True) if schema else "")[:16]
        req, resp = self.dir / "requests" / f"{rid}.json", self.dir / "responses" / f"{rid}.json"
        req.write_text(json.dumps({"id": rid, "system": system, "user": user, "max_tokens": max_tokens,
                                   "schema": schema.model_json_schema() if schema else None}, ensure_ascii=False, indent=1),
                       encoding="utf-8", newline="\n")
        deadline = time.monotonic() + self.wait_s
        while not resp.exists():
            if time.monotonic() > deadline:
                raise LLMError(f"session: no response to {rid} within {self.wait_s:.0f}s")
            time.sleep(2.0)
        time.sleep(0.2)  # let the writer finish
        text = json.loads(resp.read_text(encoding="utf-8"))["text"]
        resp.replace(self.dir / "answered" / f"{rid}.json")  # consumed: a re-issued request needs a fresh answer
        req.unlink(missing_ok=True)
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        # not billed through an API: usage is recorded as tokens for transparency, cost 0 (PRICES entry)
        return CallResult(text, Usage(int((len(system) + len(user)) / 3.5), int(len(text) / 3.5)), self.model, self.name)
