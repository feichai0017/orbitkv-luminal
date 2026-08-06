"""End-to-end coverage for the fused Qwen3-30B-A3B sparse-MoE HostOp.

These tests require the separately built CuTeDSL shared object. They exercise
the real path from a Hugging Face block through torch.compile/PT2 and egglog to
the `Qwen3Moe` operation selected by luminal_cuda_lite.

Set ``LUMINAL_QWEN3_MOE_TEST_TOKENS=16,256,1024,1025,2048`` to exercise
multiple dynamic sizes with one compilation per dtype, including both sides
of the persistent-prefill crossover. Token 1 has a distinct static Dynamo
topology and must be tested by itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from luminal import luminal_backend


def _configured_shared_object() -> Path | None:
    value = os.getenv("LUMINAL_QWEN3_MOE_LIBRARY")
    return Path(value).expanduser() if value else None


_SHARED_OBJECT = _configured_shared_object()
_QWEN3_MOE_DISABLED = bool(os.getenv("LUMINAL_DISABLE_QWEN3_MOE"))
_TOKEN_COUNTS = tuple(
    int(value.strip())
    for value in os.getenv("LUMINAL_QWEN3_MOE_TEST_TOKENS", "16").split(",")
    if value.strip()
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not _QWEN3_MOE_DISABLED
        and (_SHARED_OBJECT is None or not _SHARED_OBJECT.is_file()),
        reason="set LUMINAL_QWEN3_MOE_LIBRARY to the built libqwen3_moe.so",
    ),
]


def _make_production_block(dtype: torch.dtype) -> tuple[torch.nn.Module, int]:
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    config = Qwen3MoeConfig(
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        hidden_act="silu",
    )
    # Full Qwen models set this before constructing their layers. Direct block
    # construction must do the same so PT2 sees the exportable grouped-mm graph
    # rather than the eager, data-dependent Python expert loop.
    config._experts_implementation = "grouped_mm"
    block = Qwen3MoeSparseMoeBlock(config).eval().to(device="cuda", dtype=dtype)
    return block, config.hidden_size


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hf_qwen3_moe_block_selects_fused_hostop_and_matches_eager(dtype):
    if not _TOKEN_COUNTS or any(tokens < 1 for tokens in _TOKEN_COUNTS):
        pytest.fail("LUMINAL_QWEN3_MOE_TEST_TOKENS must contain positive integers")
    if len(_TOKEN_COUNTS) > 1 and 1 in _TOKEN_COUNTS:
        pytest.fail("token 1 has a static topology and must be tested separately")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    block, hidden_size = _make_production_block(dtype)
    first_hidden_states = torch.randn(
        1,
        _TOKEN_COUNTS[0],
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    # Only the flattened token count varies for this production block. Using
    # torch.compile(dynamic=True) also symbolizes the fixed hidden dimension,
    # which loses the equality between the activation's 2048 columns and the
    # expert weights' literal 2048 columns during PT2 translation.
    torch._dynamo.mark_dynamic(first_hidden_states, 1)

    captured = []

    def capture_backend(gm, example_inputs, options=None):
        compiled = luminal_backend(gm, example_inputs, options)
        captured.append(compiled)
        return compiled

    compiled_block = torch.compile(
        block,
        backend=capture_backend,
        fullgraph=True,
    )

    with torch.no_grad():
        for index, tokens in enumerate(_TOKEN_COUNTS):
            hidden_states = (
                first_hidden_states
                if index == 0
                else torch.randn(
                    1,
                    tokens,
                    hidden_size,
                    device="cuda",
                    dtype=dtype,
                )
            )
            expected = block(hidden_states)
            actual = compiled_block(hidden_states)

            if dtype is torch.bfloat16:
                torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
            else:
                torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)

    assert len(captured) == 1, f"expected one compiled graph, got {len(captured)}"
    # Dynamic PT2 compilations are deferred until the first invocation so
    # torch.export does not mutate Dynamo's ShapeEnv while Dynamo is still
    # installing guards. Inspect the realized CompiledModel, not that lazy
    # wrapper.
    compiled_model = captured[0]
    if hasattr(compiled_model, "_compiled"):
        assert compiled_model._compiled is not None
        compiled_model = compiled_model._compiled
    selected_host_ops = compiled_model._graph.selected_host_ops
    if _QWEN3_MOE_DISABLED:
        assert "Qwen3Moe" not in selected_host_ops
    else:
        assert "Qwen3Moe" in selected_host_ops, (
            "the PT2 graph executed correctly but search did not select the fused "
            "Qwen3Moe HostOp"
        )
