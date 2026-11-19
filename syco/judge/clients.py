"""Judge model clients: Claude through the Anthropic API, and a mock for free tests."""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..io import sha12
from ..models.base import ModelError


@dataclass
class JudgeReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ClaudeJudge:
    def __init__(self, spec: dict):
        import anthropic

        self._anthropic = anthropic
        key_env = spec.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(f"Set {key_env} to use the Claude judge (see .env.example).")
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0,
                                          timeout=float(spec.get("timeout_s", 120)))
        self.model = spec["model"]
        self.thinking = spec.get("thinking")    # e.g. {type: disabled}; see config.yaml

    def request(self, system: str | None, user: str, max_tokens: int) -> dict:
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": user}]}
        if system:
            kwargs["system"] = system
        if self.thinking:
            kwargs["thinking"] = self.thinking
        return kwargs

    def complete(self, system: str | None, user: str, max_tokens: int) -> JudgeReply:
        an = self._anthropic
        try:
            resp = self.client.messages.create(**self.request(system, user, max_tokens))
        except (an.RateLimitError, an.APITimeoutError, an.APIConnectionError, an.InternalServerError) as e:
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from e
        except an.APIStatusError as e:
            retryable = e.status_code in (408, 409, 429, 529) or e.status_code >= 500
            raise ModelError(f"{type(e).__name__} {e.status_code}: {e}", retryable=retryable) from e
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return JudgeReply(text, resp.usage.input_tokens, resp.usage.output_tokens)


class MockJudge:
    """Deterministic fake labels, so the judge and analyze steps can be tried for free."""

    model = "mock"

    def __init__(self, spec: dict):
        self.spec = spec

    def complete(self, system: str | None, user: str, max_tokens: int) -> JudgeReply:
        agreeable = "absolutely right" in user          # the mock model's baseline replies
        label = 1 if agreeable or int(sha12(user), 16) % 4 == 0 else 0
        return JudgeReply(str(label), 0, 1)


def get_judge(spec: dict):
    if spec["provider"] == "anthropic":
        return ClaudeJudge(spec)
    if spec["provider"] == "mock":
        return MockJudge(spec)
    raise SystemExit(f"Unknown judge provider '{spec['provider']}'")
