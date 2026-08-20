from __future__ import annotations

import pytest
import torch
from torch import fx

import luminal.region_compile as region_compile_module
from luminal.region_compile import compile_region
from luminal.region_export import export_region


def _add_graph() -> fx.GraphModule:
    graph = fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    result = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    graph.output((result,))
    return fx.GraphModule(torch.nn.Module(), graph)


def test_compile_region_preserves_runtime_input_indices(monkeypatch) -> None:
    region = export_region(
        _add_graph(),
        [torch.randn(2, 4), torch.randn(2, 4)],
    )
    sentinel = object()
    factory = object()
    received = {}

    monkeypatch.setattr(
        "luminal.luminal._cuda_lite_factory_capsule", lambda: factory, raising=False
    )

    def fake_save_and_compile(program, capsule, iterations, **kwargs):
        received.update(
            program=program,
            capsule=capsule,
            iterations=iterations,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(
        region_compile_module, "_save_and_compile", fake_save_and_compile
    )

    assert compile_region(region, search_iterations=3) is sentinel
    assert received == {
        "program": region.program,
        "capsule": factory,
        "iterations": 3,
        "user_indices": region.input_indices,
        "input_device_ptrs": None,
    }


def _cuda_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    try:
        from luminal.luminal import _cuda_lite_factory_capsule

        _cuda_lite_factory_capsule()
    except (ImportError, AttributeError, RuntimeError) as error:
        return f"luminal_python was not built with CUDA support: {error}"
    return None


_CUDA_SKIP_REASON = _cuda_skip_reason()


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_compile_region_from_fake_cuda_metadata() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_inputs = [
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
        ]
        region = export_region(_add_graph(), fake_inputs)

    compiled = compile_region(region, search_iterations=1)

    real_inputs = [
        torch.randn((2, 4), device="cuda", dtype=torch.float16),
        torch.randn((2, 4), device="cuda", dtype=torch.float16),
    ]
    (actual,) = compiled(*real_inputs)
    torch.testing.assert_close(actual, real_inputs[0] + real_inputs[1])
