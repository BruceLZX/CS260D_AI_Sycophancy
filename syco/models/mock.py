"""A fake model for trying the whole pipeline without API keys or cost."""
from __future__ import annotations

from ..io import sha12
from .base import ChatModel, Generation, ModelError


class MockChat(ChatModel):
    provider = "mock"

    def generate(self, system, user) -> Generation:
        h = int(sha12((system or "") + "\n" + user), 16)
        fail_percent = int(self.params.get("fail_percent", 0))
        if fail_percent and h % 100 < fail_percent:
            raise ModelError("mock failure (params.fail_percent)", retryable=False)
        if system:
            text = f"[mock reply {h % 1000}] I can see why this is hard. Here is another way to look at it."
        else:
            text = f"[mock reply {h % 1000}] You're absolutely right, and your feelings are completely valid."
        return Generation(text=text, finish_reason="stop", input_tokens=len((system or "") + user) // 4,
                          output_tokens=len(text) // 4, served_model="mock", latency_s=0.0)
