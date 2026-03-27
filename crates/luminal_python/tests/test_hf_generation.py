"""Hugging Face text-generation smoke tests.

These tests intentionally download real Hugging Face checkpoints, configs,
and tokenizers. They compare eager PyTorch output against torch.compile
with the luminal backend to verify numerical equivalence.
"""

from __future__ import annotations

import huggingface_hub
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging

from luminal import luminal_backend

MODELS = [
    "NousResearch/Llama-3.2-1B",
    "Qwen/Qwen3-8B",
]

PROMPT = "What is the capital of France "


@pytest.mark.slow
@pytest.mark.parametrize("model_id", MODELS)
def test_capital_of_france(model_id: str, device: torch.device):
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

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Clear sampling params baked into the model config so generate()
    # doesn't warn about flags that conflict with do_sample=False.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=6,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Eager baseline
    with torch.no_grad():
        eager_output = model.generate(**generate_kwargs)

    # Compiled with luminal backend
    compiled_model = torch.compile(model, backend=luminal_backend)
    with torch.no_grad():
        compiled_output = compiled_model.generate(**generate_kwargs)

    # Outputs must match
    torch.testing.assert_close(compiled_output, eager_output)
