"""Hugging Face causal-LM config option support tests.

These tests verify that luminal matches eager Hugging Face execution across
supported causal-LM config options using tiny public model definitions loaded
through AutoConfig.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import huggingface_hub
import pytest
import torch
import torch._dynamo
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers import logging as transformers_logging
from transformers.generation.configuration_utils import ContinuousBatchingConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from luminal import luminal_backend

# Attention implementations that require optional packages.
_ATTN_REQUIRES_PACKAGE: dict[str, str] = {
    "flash_attention_2": "flash_attn",
    "flash_attention_3": "flash_attn_3",
    "flash_attention_4": "cutlass",
    "paged|flash_attention_4": "cutlass",
}

# Attention implementations known to be incompatible with tiny random models
# (e.g. head_dim < 16 or missing scaffolding).
_ATTN_SKIP_TINY_MODEL: set[str] = {"flex_attention", "paged_attention"}

_PAGED_ATTN_IMPLEMENTATIONS = tuple(
    k for k in ALL_ATTENTION_FUNCTIONS.valid_keys() if k.startswith("paged|")
)


@dataclass(frozen=True)
class _CausalLMConfigCase:
    case_id: str
    model_id: str
    input_ids: tuple[int, ...]
    atol: float
    rtol: float


_MODEL_CASES = [
    _CausalLMConfigCase(
        case_id="llama_3.2_1B",
        model_id="meta-llama/Llama-3.2-1B",
        input_ids=(1, 2, 3, 4),
        atol=1e-5,
        rtol=1e-5,
    )
]

_ATTN_IMPLEMENTATIONS = tuple(
    dict.fromkeys([None, "eager", *ALL_ATTENTION_FUNCTIONS.valid_keys()])
)

transformers_logging.disable_progress_bar()
huggingface_hub.utils.disable_progress_bars()


def _attn_id(attn_impl: str | None) -> str:
    return "default" if attn_impl is None else attn_impl


def _compare_past_key_values(lhs, rhs, *, atol: float, rtol: float) -> None:
    assert lhs is not None
    assert rhs is not None
    assert hasattr(lhs, "layers")
    assert hasattr(rhs, "layers")
    assert len(lhs.layers) == len(rhs.layers)

    for lhs_layer, rhs_layer in zip(lhs.layers, rhs.layers):
        torch.testing.assert_close(lhs_layer.keys, rhs_layer.keys, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            lhs_layer.values, rhs_layer.values, atol=atol, rtol=rtol
        )


def _instantiate_model(
    model_id: str, *, use_cache: bool, attn_impl: str | None, device: torch.device
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(model_id)
    config.use_cache = use_cache
    if attn_impl is not None:
        config._attn_implementation = attn_impl
    return AutoModelForCausalLM.from_config(config).eval().to(device)


@pytest.mark.parametrize("model_case", _MODEL_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("use_cache", [False, True], ids=["no_cache", "cache"])
@pytest.mark.parametrize("attn_impl", _ATTN_IMPLEMENTATIONS, ids=_attn_id)
def test_hf_causal_lm_config_options_match_eager(
    model_case: _CausalLMConfigCase,
    use_cache: bool,
    attn_impl: str | None,
    device: torch.device,
):
    """Compare luminal against eager HF across causal-LM config options."""
    # flash_attention_2 is very slow; skip unless explicitly requested.
    if attn_impl == "flash_attention_2":
        pytest.skip("flash_attention_2 is very slow")

    # Skip paged variants -- they require the continuous batching API.
    if attn_impl is not None and attn_impl.startswith("paged|"):
        pytest.skip(f"{attn_impl} requires continuous batching API")

    # Skip attention implementations that need missing optional packages.
    if attn_impl in _ATTN_REQUIRES_PACKAGE:
        pkg = _ATTN_REQUIRES_PACKAGE[attn_impl]
        if importlib.util.find_spec(pkg) is None:
            pytest.skip(f"{attn_impl} requires package '{pkg}' which is not installed")

    # Skip implementations incompatible with tiny random models.
    if attn_impl in _ATTN_SKIP_TINY_MODEL:
        pytest.skip(f"{attn_impl} is incompatible with tiny random test models")

    if use_cache:
        torch._dynamo.config.cache_size_limit = 2

    model = _instantiate_model(
        model_case.model_id,
        use_cache=use_cache,
        attn_impl=attn_impl,
        device=device,
    )
    input_ids = torch.tensor([model_case.input_ids], device=device)

    with torch.no_grad():
        eager_prefill = model(input_ids)

    compiled_model = torch.compile(model, backend=luminal_backend)

    with torch.no_grad():
        compiled_prefill = compiled_model(input_ids)

    torch.testing.assert_close(
        compiled_prefill.logits,
        eager_prefill.logits,
        atol=model_case.atol,
        rtol=model_case.rtol,
    )

    if not use_cache:
        assert eager_prefill.past_key_values is None
        assert compiled_prefill.past_key_values is None
        return

    assert eager_prefill.past_key_values is not None
    assert compiled_prefill.past_key_values is not None
    _compare_past_key_values(
        compiled_prefill.past_key_values,
        eager_prefill.past_key_values,
        atol=model_case.atol,
        rtol=model_case.rtol,
    )

    next_token = eager_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    with torch.no_grad():
        eager_decode = model(next_token, past_key_values=eager_prefill.past_key_values)
        compiled_decode = compiled_model(
            next_token,
            past_key_values=compiled_prefill.past_key_values,
        )

    torch.testing.assert_close(
        compiled_decode.logits,
        eager_decode.logits,
        atol=model_case.atol,
        rtol=model_case.rtol,
    )
    assert eager_decode.past_key_values is not None
    assert compiled_decode.past_key_values is not None
    _compare_past_key_values(
        compiled_decode.past_key_values,
        eager_decode.past_key_values,
        atol=model_case.atol,
        rtol=model_case.rtol,
    )


@pytest.mark.parametrize("model_case", _MODEL_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("attn_impl", _ATTN_IMPLEMENTATIONS, ids=_attn_id)
def test_hf_generate_batch(
    model_case: _CausalLMConfigCase,
    attn_impl: str | None,
    device: torch.device,
):
    """Compare generate_batch output for each attention variant against eager baseline."""
    if attn_impl is not None and "flash_attention_2" in attn_impl:
        pytest.skip("flash_attention_2 is very slow")

    if attn_impl in _ATTN_REQUIRES_PACKAGE:
        pkg = _ATTN_REQUIRES_PACKAGE[attn_impl]
        if importlib.util.find_spec(pkg) is None:
            pytest.skip(f"{attn_impl} requires package '{pkg}' which is not installed")

    if attn_impl in _ATTN_SKIP_TINY_MODEL:
        pytest.skip(f"{attn_impl} is incompatible with tiny random test models")

    if device.type != "cuda":
        pytest.skip("generate_batch requires CUDA")

    config = AutoConfig.from_pretrained(model_case.model_id)
    config.use_cache = True
    model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval().to(device)

    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=5,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    cb_config = ContinuousBatchingConfig(
        block_size=256,
        use_cuda_graph=False,
    )

    inputs = [list(model_case.input_ids)]

    # Baseline: eager generate_batch.
    model.set_attn_implementation("eager")
    eager_outputs = model.generate_batch(
        inputs,
        generation_config=gen_config,
        continuous_batching_config=cb_config,
        progress_bar=False,
        warmup=False,
    )

    # Variant under test.
    if attn_impl is not None:
        model.set_attn_implementation(attn_impl)
    variant_outputs = model.generate_batch(
        inputs,
        generation_config=gen_config,
        continuous_batching_config=cb_config,
        progress_bar=False,
        warmup=False,
    )

    assert len(eager_outputs) == len(variant_outputs)
    eager_out = next(iter(eager_outputs.values()))
    variant_out = next(iter(variant_outputs.values()))
    assert eager_out.error is None, f"Eager baseline failed: {eager_out.error}"
    assert variant_out.error is None, f"Variant {attn_impl} failed: {variant_out.error}"
    assert eager_out.generated_tokens == variant_out.generated_tokens, (
        f"Token mismatch for {attn_impl}: eager={eager_out.generated_tokens} "
        f"vs variant={variant_out.generated_tokens}"
    )
