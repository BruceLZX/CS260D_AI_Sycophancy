"""Local open-weight models through Ollama (owner: Cecilia).

Uses Ollama's native /api/chat endpoint, not its OpenAI-compatible one, because only the
native endpoint takes `think` (Qwen thinks by default; turn it off) and `options.num_ctx`
(the context window; Ollama's default is too small for a long post plus the policy prompt).

Setup: install Ollama (https://ollama.com), then for example `ollama pull qwen3.5:9b` and
`ollama pull llama3.1:8b`.

Status: a starting point. It is unit-tested against a fake server only; check it on your
machine with `python -m syco smoke --models local` before the main run.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import ChatModel, Generation, ModelError

REQUEST_FIELDS = ("think", "keep_alive")   # sent at the top level; every other param goes into options


class OllamaChat(ChatModel):
    provider = "ollama"

    def __init__(self, spec: dict, gen_cfg: dict):
        super().__init__(spec, gen_cfg)
        host = spec.get("host") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        self.host = host.rstrip("/")
        # Talk to Ollama directly: a system proxy or VPN setting must not reroute local traffic.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, system: str | None, user: str) -> dict:
        """JSON body for POST /api/chat (separate so tests can check it)."""
        params = dict(self.params)
        body = {"model": self.model, "stream": False,
                "messages": ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": user}]}
        for key in REQUEST_FIELDS:
            if key in params:
                body[key] = params.pop(key)
        options = dict(params.pop("options", None) or {})
        options.update(params)                       # e.g. num_ctx, seed, top_p
        options["num_predict"] = self.max_output_tokens
        if self.temperature is not None:
            options["temperature"] = self.temperature
        body["options"] = options
        return body

    def generate(self, system, user) -> Generation:
        data = json.dumps(self.request(system, user)).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/chat", data=data,
                                     headers={"Content-Type": "application/json"})
        start = time.time()
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise ModelError(f"Ollama HTTP {e.code}: {detail}", retryable=e.code == 429 or e.code >= 500) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ModelError(f"Ollama timed out after {self.timeout:.0f}s", retryable=True) from e
            raise ModelError(f"Cannot reach Ollama at {self.host} ({e.reason}). "
                             "Is the Ollama app (or `ollama serve`) running?", retryable=False) from e
        except TimeoutError as e:
            raise ModelError(f"Ollama timed out after {self.timeout:.0f}s", retryable=True) from e
        message = payload.get("message") or {}
        return Generation(
            text=(message.get("content") or "").strip(),
            finish_reason=payload.get("done_reason"),
            input_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
            served_model=payload.get("model"),
            latency_s=round(time.time() - start, 2),
            # Should stay 0 when thinking is off; `python -m syco status` reports it.
            extra={"thinking_chars": len(message.get("thinking") or "")},
        )
