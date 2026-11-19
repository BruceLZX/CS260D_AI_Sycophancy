"""The interface every model backend implements, and the shared retry loop."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class Generation:
    """What a backend returns for one call."""
    text: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None      # includes reasoning tokens where the API counts them there
    reasoning_tokens: int | None = None
    served_model: str | None = None       # the model/version the API says it actually used
    latency_s: float | None = None
    extra: dict = field(default_factory=dict)


class ModelError(Exception):
    """A failed call. retryable=True for rate limits, timeouts and server errors."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ChatModel:
    """One configured model (an entry under `models:` in config.yaml).

    To add a backend: subclass this, implement `generate`, and register the provider name in
    syco/models/__init__.py (and in MODEL_PROVIDERS in syco/config.py).
    """

    provider = "base"

    def __init__(self, spec: dict, gen_cfg: dict):
        self.spec = spec
        self.name = spec["name"]
        self.model = spec["model"]
        self.params = dict(spec.get("params") or {})
        self.max_output_tokens = int(spec.get("max_output_tokens") or gen_cfg.get("max_output_tokens") or 1024)
        temperature = spec.get("temperature", gen_cfg.get("temperature"))
        self.temperature = None if temperature is None else float(temperature)
        self.timeout = float(spec.get("timeout_s", 300))

    def generate(self, system: str | None, user: str) -> Generation:
        """One conversation: an optional system message and one user message."""
        raise NotImplementedError

    def settings(self) -> dict:
        """Saved with every response so runs can be compared and reproduced."""
        return {"provider": self.provider, "model": self.model, "params": self.params,
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature}


def call_with_retries(fn, retries: int = 6, base_delay: float = 2.0, max_delay: float = 60.0):
    """Run fn(). On a retryable ModelError wait 2s, 4s, 8s, ... (plus jitter) and try again."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except ModelError as e:
            if not e.retryable or attempt > retries:
                raise
            time.sleep(min(max_delay, base_delay * 2 ** (attempt - 1)) + random.uniform(0, 1))
