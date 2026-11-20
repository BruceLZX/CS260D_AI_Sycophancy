"""The Cecilia backend uses the shared experiment configuration."""
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from syco.config import load_config, model_spec
from syco.models import get_model


class CeciliaTest(unittest.TestCase):
    def test_profile_uses_the_shared_experiment(self):
        shared = load_config()
        config = load_config(model_config="config.cecilia.yaml")
        self.assertEqual(config["conditions"], shared["conditions"])
        self.assertEqual(config["data"], shared["data"])
        self.assertEqual(config["models"][:len(shared["models"])], shared["models"])
        self.assertEqual(model_spec(config, "qwen2.5-3b")["provider"], "cecilia")

    def test_native_template_and_stop_token_are_used(self):
        cfg = load_config(model_config="config.cecilia.yaml")
        spec = model_spec(cfg, "llama3.2-3b")

        class Encoding(dict):
            def to(self, device):
                return self

        tokenizer = Mock(return_value=Encoding(input_ids=SimpleNamespace(shape=(1, 4))))
        tokenizer.apply_chat_template.return_value = "SYSTEM POLICY\nUSER ADVICE"
        tokenizer.decode.return_value = " An answer. "
        tokenizer.pad_token_id = 99
        engine = MagicMock()
        engine.device = "cuda"
        engine.config.max_position_embeddings = 4096
        engine.generation_config.eos_token_id = [99, 100]
        engine.to.return_value.eval.return_value = engine
        engine.generate.return_value[0, 4:].tolist.return_value = [1, 2, 100]
        torch = SimpleNamespace(cuda=Mock(), bfloat16="bf16", inference_mode=nullcontext)
        transformers = SimpleNamespace(AutoTokenizer=Mock(), AutoModelForCausalLM=Mock())
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        transformers.AutoModelForCausalLM.from_pretrained.return_value = engine
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            result = get_model(spec, cfg["generation"]).generate("SYSTEM POLICY", "USER ADVICE")
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "system", "content": "SYSTEM POLICY"}, {"role": "user", "content": "USER ADVICE"}],
            tokenize=False, add_generation_prompt=True, date_string=spec["params"]["date_string"],
        )
        tokenizer.decode.assert_called_once_with([1, 2], skip_special_tokens=True)
        self.assertEqual((result.finish_reason, result.input_tokens, result.output_tokens), ("stop", 4, 2))
