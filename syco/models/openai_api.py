"""OpenAI and DeepSeek chat models (owner: Bruce).

DeepSeek's API is OpenAI-compatible, so both use the `openai` package:
    OpenAI     default base URL              key: OPENAI_API_KEY
    DeepSeek   https://api.deepseek.com      key: DEEPSEEK_API_KEY
"""
from __future__ import annotations

import os
import time

from .base import ChatModel, Generation, ModelError


class OpenAIChat(ChatModel):
    provider = "openai"
    default_key_env = "OPENAI_API_KEY"
    default_base_url: str | None = None
    max_tokens_field = "max_completion_tokens"   # OpenAI's name; it also covers reasoning tokens

    def __init__(self, spec: dict, gen_cfg: dict):
        super().__init__(spec, gen_cfg)
        import openai

        self._openai = openai
        key_env = spec.get("api_key_env", self.default_key_env)
        api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(f"Set {key_env} to run '{self.name}' (see .env.example).")
        # Retries are done by call_with_retries, so every failure is visible and logged.
        self.client = openai.OpenAI(api_key=api_key, base_url=spec.get("base_url", self.default_base_url),
                                    max_retries=0, timeout=self.timeout)

    def provider_params(self) -> dict:
        """`params` from config.yaml, translated for this API. OpenAI: passed through as is
        (for example reasoning_effort)."""
        return dict(self.params)

    def request(self, system: str | None, user: str) -> dict:
        """Keyword arguments for chat.completions.create (separate so tests can check them)."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kwargs = {"model": self.model, "messages": messages, self.max_tokens_field: self.max_output_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        kwargs.update(self.provider_params())
        return kwargs

    def generate(self, system, user) -> Generation:
        oa = self._openai
        start = time.time()
        try:
            resp = self.client.chat.completions.create(**self.request(system, user))
        except (oa.RateLimitError, oa.APITimeoutError, oa.APIConnectionError, oa.InternalServerError) as e:
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from e
        except oa.APIStatusError as e:
            retryable = e.status_code in (408, 409, 429) or e.status_code >= 500
            raise ModelError(f"{type(e).__name__} {e.status_code}: {e}", retryable=retryable) from e
        choice = resp.choices[0]
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        return Generation(
            text=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
            served_model=getattr(resp, "model", None),
            latency_s=round(time.time() - start, 2),
        )


class DeepSeekChat(OpenAIChat):
    provider = "deepseek"
    default_key_env = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com"
    max_tokens_field = "max_tokens"

    def provider_params(self) -> dict:
        """`thinking: disabled` (or enabled) becomes extra_body={"thinking": {"type": ...}}.
        DeepSeek turns thinking ON by default (api-docs.deepseek.com/guides/thinking_mode)."""
        params = dict(self.params)
        extra = dict(params.pop("extra_body", None) or {})
        if "thinking" in params:
            value = params.pop("thinking")
            off = value in (False, None, "disabled", "off", "false", "no")   # YAML reads a bare off as False
            extra["thinking"] = {"type": "disabled" if off else "enabled"}
        if extra:
            params["extra_body"] = extra
        return params
