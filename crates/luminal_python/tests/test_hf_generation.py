"""Hugging Face text-generation smoke tests.

These tests intentionally download real Hugging Face checkpoints, configs,
and tokenizers. They compare eager PyTorch output against torch.compile
with the luminal backend to verify numerical equivalence.
"""

from __future__ import annotations

import huggingface_hub
import pytest
import torch
import torch._dynamo
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging

from luminal import luminal_backend

MODELS = [
    "NousResearch/Llama-3.2-1B",
    "Qwen/Qwen3-8B",
]


def _load_model_and_tokenizer(
    model_id: str, device: torch.device
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a pretrained HF causal LM and its tokenizer, ready for generation."""
    transformers_logging.disable_progress_bar()
    huggingface_hub.utils.disable_progress_bars()

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=dtype)
        .eval()
        .to(device)
    )

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return model, tokenizer


def _encode(
    tokenizer: AutoTokenizer, prompt: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """Tokenize a prompt and move tensors to device."""
    encoded = tokenizer(prompt, return_tensors="pt")
    result = {"input_ids": encoded["input_ids"].to(device)}
    if encoded.get("attention_mask") is not None:
        result["attention_mask"] = encoded["attention_mask"].to(device)
    return result


def _generate_kwargs(
    tokenizer: AutoTokenizer,
    encoded: dict[str, torch.Tensor],
    max_new_tokens: int = 6,
) -> dict:
    """Build kwargs dict for model.generate()."""
    return dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


@pytest.mark.slow
class TestHFGeneration:
    """End-to-end tests comparing eager PyTorch against torch.compile with luminal."""

    @pytest.mark.parametrize("model_id", MODELS)
    def test_capital_of_france(self, model_id: str, device: torch.device):
        """Basic greedy generation -- the original smoke test."""
        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        encoded = _encode(tokenizer, "What is the capital of France ", device)
        kwargs = _generate_kwargs(tokenizer, encoded)

        with torch.no_grad():
            eager_output = model.generate(**kwargs)

        compiled_model = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            compiled_output = compiled_model.generate(**kwargs)

        torch.testing.assert_close(compiled_output, eager_output)

    @pytest.mark.parametrize("model_id", MODELS)
    def test_forward_logits(self, model_id: str, device: torch.device):
        """Forward pass only -- compare raw logits, not generated tokens."""
        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        encoded = _encode(tokenizer, "The quick brown fox", device)

        with torch.no_grad():
            eager_out = model(**encoded)

        compiled_model = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            compiled_out = compiled_model(**encoded)

        dtype = next(model.parameters()).dtype
        atol = 1e-2 if dtype == torch.float16 else 1e-3
        torch.testing.assert_close(
            compiled_out.logits, eager_out.logits, atol=atol, rtol=1e-3
        )

    @pytest.mark.parametrize("model_id", MODELS[:1])
    @pytest.mark.parametrize(
        "prompt",
        ["Hi", "What is the capital of France", "Explain the theory of general relativity in simple terms that a high school student could understand"],
        ids=["short", "medium", "long"],
    )
    def test_variable_length_prompts(self, model_id: str, prompt: str, device: torch.device):
        """Generate with prompts of different lengths -- tests dynamic shape handling."""
        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        encoded = _encode(tokenizer, prompt, device)
        kwargs = _generate_kwargs(tokenizer, encoded, max_new_tokens=4)

        with torch.no_grad():
            eager_output = model.generate(**kwargs)

        compiled_model = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            compiled_output = compiled_model.generate(**kwargs)

        torch.testing.assert_close(compiled_output, eager_output)

    @pytest.mark.parametrize("model_id", MODELS[:1])
    def test_chat_template_generation(self, model_id: str, device: torch.device):
        """Generate using chat-templated input with special tokens."""
        model, tokenizer = _load_model_and_tokenizer(model_id, device)

        if tokenizer.chat_template is None:
            pytest.skip(f"{model_id} has no chat template")

        messages = [{"role": "user", "content": "What is 2+2?"}]
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        kwargs = _generate_kwargs(tokenizer, encoded)

        with torch.no_grad():
            eager_output = model.generate(**kwargs)

        compiled_model = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            compiled_output = compiled_model.generate(**kwargs)

        torch.testing.assert_close(compiled_output, eager_output)

    @pytest.mark.parametrize("model_id", MODELS[:1])
    @pytest.mark.parametrize("max_new_tokens", [20, 50])
    def test_longer_generation(self, model_id: str, max_new_tokens: int, device: torch.device):
        """Generate many tokens to stress KV cache over extended decode loop."""
        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        encoded = _encode(tokenizer, "Once upon a time", device)
        kwargs = _generate_kwargs(tokenizer, encoded, max_new_tokens=max_new_tokens)

        with torch.no_grad():
            eager_output = model.generate(**kwargs)

        compiled_model = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            compiled_output = compiled_model.generate(**kwargs)

        torch.testing.assert_close(compiled_output, eager_output)

    @pytest.mark.parametrize("model_id", MODELS[:1])
    def test_greedy_determinism(self, model_id: str, device: torch.device):
        """Greedy generation produces identical results on repeated calls."""
        torch._dynamo.config.cache_size_limit = 4

        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        encoded = _encode(tokenizer, "The meaning of life is", device)
        kwargs = _generate_kwargs(tokenizer, encoded, max_new_tokens=10)

        compiled_model = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            output_1 = compiled_model.generate(**kwargs)
            output_2 = compiled_model.generate(**kwargs)

        torch.testing.assert_close(output_1, output_2)

    @pytest.mark.parametrize("model_id", MODELS[:1])
    def test_reuse_compiled_model(self, model_id: str, device: torch.device):
        """Call the same compiled model multiple times with different prompts."""
        torch._dynamo.config.cache_size_limit = 8

        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        compiled_model = torch.compile(model, backend=luminal_backend)

        prompts = [
            "The capital of France is",
            "Water boils at",
            "The largest planet in our solar system is",
        ]

        for prompt in prompts:
            encoded = _encode(tokenizer, prompt, device)
            kwargs = _generate_kwargs(tokenizer, encoded, max_new_tokens=4)

            with torch.no_grad():
                eager_output = model.generate(**kwargs)
                compiled_output = compiled_model.generate(**kwargs)

            torch.testing.assert_close(compiled_output, eager_output)

    @pytest.mark.parametrize("model_id", MODELS[:1])
    def test_batched_inference(self, model_id: str, device: torch.device):
        """Batched generation with multiple prompts and left-padding."""
        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        tokenizer.padding_side = "left"

        prompts = ["Hello", "What is the capital of France"]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        kwargs = _generate_kwargs(tokenizer, encoded, max_new_tokens=4)

        with torch.no_grad():
            eager_output = model.generate(**kwargs)

        compiled_model = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            compiled_output = compiled_model.generate(**kwargs)

        torch.testing.assert_close(compiled_output, eager_output)
