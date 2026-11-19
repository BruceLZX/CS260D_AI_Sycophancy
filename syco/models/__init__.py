"""Model backends. `get_model(spec, gen_cfg)` builds the backend named by spec['provider']."""
from __future__ import annotations

from .base import ChatModel, Generation, ModelError, call_with_retries


def get_model(spec: dict, gen_cfg: dict) -> ChatModel:
    provider = spec["provider"]
    if provider == "openai":
        from .openai_api import OpenAIChat as cls
    elif provider == "deepseek":
        from .openai_api import DeepSeekChat as cls
    elif provider == "ollama":
        from .ollama_local import OllamaChat as cls
    elif provider == "mock":
        from .mock import MockChat as cls
    else:
        raise SystemExit(f"Unknown provider '{provider}' for model '{spec.get('name')}'.")
    return cls(spec, gen_cfg)


__all__ = ["ChatModel", "Generation", "ModelError", "call_with_retries", "get_model"]
