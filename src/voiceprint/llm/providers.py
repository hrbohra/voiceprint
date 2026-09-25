"""Provider adapters: Anthropic (default), OpenAI, Gemini, Ollama. Each turns (system, user,
schema) into a CallResult and classifies its own errors as transient or permanent. Keys are read
from the environment by each SDK and never touch the cache, the ledger or any output."""

from __future__ import annotations

import json
import os

import httpx
from pydantic import BaseModel

from .base import CallResult, LLMError, Provider, Refused, Usage


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", timeout_s: float = 90.0):
        super().__init__(model, timeout_s)
        import anthropic

        self._a = anthropic
        # SDK retries are off: the LLM wrapper owns retries so they share one backoff and budget.
        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)

    def transient(self, err: Exception) -> bool:
        a = self._a
        if isinstance(err, (a.RateLimitError, a.APIConnectionError, a.APITimeoutError)):
            return True
        return isinstance(err, a.APIStatusError) and err.status_code >= 500

    def _call(self, system, user, max_tokens, schema):
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        if schema is not None:
            resp = self.client.messages.parse(output_format=schema, **kwargs)
            self._check(resp)
            text = resp.parsed_output.model_dump_json() if resp.parsed_output is not None else ""
        else:
            resp = self.client.messages.create(**kwargs)
            self._check(resp)
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return CallResult(text, Usage(resp.usage.input_tokens, resp.usage.output_tokens), self.model, self.name)

    @staticmethod
    def _check(resp) -> None:
        if resp.stop_reason == "refusal":
            raise Refused(f"refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise LLMError("hit max_tokens; raise max_tokens for this stage")


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-5", timeout_s: float = 90.0):
        super().__init__(model, timeout_s)
        import openai

        self._o = openai
        self.client = openai.OpenAI(timeout=timeout_s, max_retries=0)

    def transient(self, err: Exception) -> bool:
        o = self._o
        if isinstance(err, (o.RateLimitError, o.APIConnectionError, o.APITimeoutError)):
            return True
        return isinstance(err, o.APIStatusError) and err.status_code >= 500

    def _call(self, system, user, max_tokens, schema):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if schema is not None:
            resp = self.client.chat.completions.parse(model=self.model, messages=messages, response_format=schema, max_completion_tokens=max_tokens)
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                raise Refused(msg.refusal)
            text = msg.parsed.model_dump_json() if msg.parsed is not None else ""
        else:
            resp = self.client.chat.completions.create(model=self.model, messages=messages, max_completion_tokens=max_tokens)
            text = resp.choices[0].message.content or ""
        u = resp.usage
        return CallResult(text, Usage(u.prompt_tokens, u.completion_tokens), self.model, self.name)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str = "gemini-3.8-flash", timeout_s: float = 90.0):
        super().__init__(model, timeout_s)
        from google import genai
        from google.genai import types

        self._types = types
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def transient(self, err: Exception) -> bool:
        code = getattr(err, "code", None) or getattr(err, "status_code", None)
        return code in (429, 500, 502, 503, 504) or isinstance(err, (httpx.TimeoutException, httpx.NetworkError))

    def _call(self, system, user, max_tokens, schema):
        t = self._types
        # Gemini 2.5+/3.x "think" by default and thinking tokens count against max_output_tokens, so a
        # 1,500-token answer budget could be spent before the answer starts. `max_tokens` here means
        # the answer; thinking gets its own explicit budget on top.
        thinking = 0 if self.model.startswith(("gemini-1", "gemini-2.0")) else min(8192, max(1024, max_tokens))
        cfg = dict(system_instruction=system, max_output_tokens=max_tokens + thinking)
        if thinking:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=thinking)
        if schema is not None:
            cfg.update(response_mime_type="application/json", response_schema=schema)
        resp = self.client.models.generate_content(model=self.model, contents=user, config=t.GenerateContentConfig(**cfg))
        cand = (resp.candidates or [None])[0]
        reason = str(getattr(cand, "finish_reason", "") or "")
        if "SAFETY" in reason or "PROHIBITED" in reason:
            raise Refused(f"gemini finish_reason {reason}")
        if "MAX_TOKENS" in reason:
            raise LLMError("gemini hit max_output_tokens; raise max_tokens for this stage")
        text = resp.text or ""
        um = resp.usage_metadata
        billed_out = (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)  # thinking is billed
        return CallResult(text, Usage(getattr(um, "prompt_token_count", 0) or 0, billed_out), self.model, self.name)


class OllamaProvider(Provider):
    """Local models through Ollama. Offline fallback only; never the default."""

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b-instruct", timeout_s: float = 300.0, host: str | None = None):
        super().__init__(model, timeout_s)
        self.host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    def transient(self, err: Exception) -> bool:
        return isinstance(err, (httpx.TimeoutException, httpx.NetworkError))

    def _call(self, system, user, max_tokens, schema):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        if schema is not None:
            body["format"] = schema.model_json_schema()
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        return CallResult(j["message"]["content"], Usage(j.get("prompt_eval_count", 0), j.get("eval_count", 0)), self.model, self.name)


def make_provider(provider: str, model: str, timeout_s: float = 90.0) -> Provider:
    match provider:
        case "anthropic":
            return AnthropicProvider(model, timeout_s)
        case "openai":
            return OpenAIProvider(model, timeout_s)
        case "gemini":
            return GeminiProvider(model, timeout_s)
        case "ollama":
            return OllamaProvider(model, timeout_s)
        case "session":
            from .session import SessionProvider

            return SessionProvider(model, timeout_s)
    raise LLMError(f"unknown provider {provider!r}")


def available() -> dict[str, bool]:
    """Which providers have credentials in this environment (never prints the keys)."""
    ollama_up = False
    try:
        ollama_up = httpx.get(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") + "/api/tags", timeout=1.5).status_code == 200
    except Exception:  # noqa: BLE001
        pass
    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
        "ollama": ollama_up,
        "session": True,  # no credentials needed; only used when a stage names it (never in the default fallback list)
    }


__all__ = ["make_provider", "available", "AnthropicProvider", "OpenAIProvider", "GeminiProvider", "OllamaProvider", "json"]
