"""Offline Hugging Face inference on a Slurm GPU (owner: Cecilia)."""
from __future__ import annotations

import time
from threading import Lock

from .base import ChatModel, Generation, ModelError
from ..io import sha12


class CeciliaChat(ChatModel):
    provider = "cecilia"

    def __init__(self, spec: dict, gen_cfg: dict):
        super().__init__(spec, gen_cfg)
        self.revision = self.params["revision"]
        self.template_date = self.params["date_string"]
        if self.temperature not in (None, 0):
            raise ModelError("The Cecilia backend uses greedy decoding. Set temperature to 0.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ModelError("The Cecilia backend needs a CUDA allocation with native BF16 support.")
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, revision=self.revision, local_files_only=True,
            )
            self.engine = AutoModelForCausalLM.from_pretrained(
                self.model, revision=self.revision, local_files_only=True,
                use_safetensors=True, dtype=torch.bfloat16, attn_implementation="sdpa",
            ).to("cuda").eval()
        except (ImportError, OSError) as exc:
            raise ModelError(f"Could not load {self.model}@{self.revision}. "
                             f"Check the CUDA environment and stage the pinned snapshot. {exc}") from exc

        config = self.engine.generation_config
        stop_ids = config.eos_token_id
        self.stop_ids = {stop_ids} if isinstance(stop_ids, int) else set(stop_ids or ())
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.lock = Lock()

    def settings(self) -> dict:
        return {**super().settings(), "dtype": "bfloat16", "decoding": "greedy", "attention": "sdpa",
                "chat_template_sha": sha12(self.tokenizer.chat_template),
                "gpu": self.torch.cuda.get_device_name(0), "torch": self.torch.__version__}

    def generate(self, system: str | None, user: str) -> Generation:
        messages = [{"role": "user", "content": user}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, date_string=self.template_date,
        )
        if user.strip() not in rendered or (system and system.strip() not in rendered):
            raise ModelError("The chat template did not preserve the input messages.")
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]
        if input_tokens + self.max_output_tokens > self.engine.config.max_position_embeddings:
            raise ModelError("The full prompt and response budget exceed the model context window.")

        with self.lock, self.torch.inference_mode():
            start = time.monotonic()
            generated = self.engine.generate(
                **inputs.to(self.engine.device), max_new_tokens=self.max_output_tokens,
                do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.pad_id,
            )[0, input_tokens:].tolist()
            latency = round(time.monotonic() - start, 2)
        end = next((i for i, token in enumerate(generated) if token in self.stop_ids), len(generated))
        return Generation(
            text=self.tokenizer.decode(generated[:end], skip_special_tokens=True).strip(),
            finish_reason="length" if end == self.max_output_tokens else "stop",
            input_tokens=input_tokens, output_tokens=end, reasoning_tokens=0,
            served_model=f"{self.model}@{self.revision}", latency_s=latency,
            extra={"thinking_chars": 0},
        )
