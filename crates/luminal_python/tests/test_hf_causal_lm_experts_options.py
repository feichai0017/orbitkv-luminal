"""Hugging Face causal-LM experts backend smoke tests.

These tests load a real pretrained text-only MoE model and compare eager
PyTorch against torch.compile with the luminal backend across the standardized
`experts_implementation` backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import huggingface_hub
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import logging as transformers_logging

from luminal import luminal_backend


@dataclass(frozen=True)
class _HFMoeCase:
    case_id: str
    model_id: str
    input_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _HFMoeBundle:
    case: _HFMoeCase
    model: AutoModelForCausalLM
    device: torch.device
    dtype: torch.dtype


_MODEL_CASES = [
    _HFMoeCase(
        case_id="qwen15_moe_a27b",
        model_id="Qwen/Qwen1.5-MoE-A2.7B",
        input_ids=(
            (1, 2, 3, 4, 5, 6, 7, 8),
            (8, 7, 6, 5, 4, 3, 2, 1),
        ),
    ),
]

_EXPERTS_IMPLEMENTATIONS = ("eager", "batched_mm", "grouped_mm")


def _model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _output_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 5e-2
    if dtype == torch.float16:
        return 1e-2
    return 1e-3


def _compare_router_logits(lhs, rhs, *, atol: float, rtol: float) -> None:
    assert lhs is not None
    assert rhs is not None

    if isinstance(lhs, torch.Tensor):
        torch.testing.assert_close(lhs, rhs, atol=atol, rtol=rtol)
        return

    assert len(lhs) == len(rhs)
    for lhs_layer, rhs_layer in zip(lhs, rhs):
        torch.testing.assert_close(lhs_layer, rhs_layer, atol=atol, rtol=rtol)


@pytest.fixture(scope="module", params=_MODEL_CASES, ids=lambda case: case.case_id)
def hf_moe_bundle(request: pytest.FixtureRequest) -> _HFMoeBundle:
    backend = os.getenv("LUMINAL_BACKEND", "native").lower()
    if backend != "cuda" or not torch.cuda.is_available():
        pytest.skip("HF MoE experts backend tests require the CUDA backend")

    transformers_logging.disable_progress_bar()
    huggingface_hub.utils.disable_progress_bars()

    case: _HFMoeCase = request.param
    device = torch.device("cuda")
    dtype = _model_dtype(device)

    config = AutoConfig.from_pretrained(case.model_id)
    config.use_cache = False
    config.output_router_logits = True
    # Keep attention fixed so this test isolates experts backends.
    config._attn_implementation = "eager"

    model = (
        AutoModelForCausalLM.from_pretrained(
            case.model_id,
            config=config,
            dtype=dtype,
        )
        .eval()
        .to(device)
    )
    return _HFMoeBundle(case=case, model=model, device=device, dtype=dtype)


@pytest.mark.slow
@pytest.mark.parametrize(
    "experts_implementation",
    _EXPERTS_IMPLEMENTATIONS,
    ids=list(_EXPERTS_IMPLEMENTATIONS),
)
def test_hf_causal_lm_experts_implementation_matches_eager(
    hf_moe_bundle: _HFMoeBundle, experts_implementation: str
):
    model = hf_moe_bundle.model
    model.set_experts_implementation(experts_implementation)
    assert model.config._experts_implementation == experts_implementation

    input_ids = torch.tensor(hf_moe_bundle.case.input_ids, device=hf_moe_bundle.device)
    kwargs = {
        "input_ids": input_ids,
        "use_cache": False,
        "output_router_logits": True,
        "logits_to_keep": 1,
    }

    with torch.no_grad():
        eager_output = model(**kwargs)

    compiled_model = torch.compile(model, backend=luminal_backend)
    with torch.no_grad():
        compiled_output = compiled_model(**kwargs)

    atol = _output_tolerance(hf_moe_bundle.dtype)
    rtol = 1e-3

    torch.testing.assert_close(
        compiled_output.logits,
        eager_output.logits,
        atol=atol,
        rtol=rtol,
    )
    _compare_router_logits(
        compiled_output.router_logits,
        eager_output.router_logits,
        atol=atol,
        rtol=rtol,
    )

    if eager_output.aux_loss is not None or compiled_output.aux_loss is not None:
        assert eager_output.aux_loss is not None
        assert compiled_output.aux_loss is not None
        torch.testing.assert_close(
            compiled_output.aux_loss,
            eager_output.aux_loss,
            atol=atol,
            rtol=rtol,
        )
