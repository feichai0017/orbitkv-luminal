"""Tests for backend-free PT2 to Luminal HLIR graph inspection."""

from pathlib import Path

import torch

from luminal import translate_pt2_to_dot, translate_pt2_to_egglog


class AddMul(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x + y) * y


class CumsumWithOutputDtype(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x, 0, dtype=torch.int32)


def test_translate_pt2_to_dot_does_not_require_backend(tmp_path: Path):
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    exported = torch.export.export(AddMul(), inputs, strict=False)
    pt2_path = tmp_path / "add_mul.pt2"
    torch.export.save(exported, pt2_path)

    dot = translate_pt2_to_dot(str(pt2_path))

    assert dot.startswith("digraph")
    assert "Add" in dot
    assert "Mul" in dot


def test_translate_pt2_to_egglog_does_not_require_backend(tmp_path: Path):
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    exported = torch.export.export(AddMul(), inputs, strict=False)
    pt2_path = tmp_path / "add_mul.pt2"
    torch.export.save(exported, pt2_path)

    program, root = translate_pt2_to_egglog(str(pt2_path))

    assert program.startswith("(let t0 ")
    assert "Add" in program
    assert "Mul" in program
    assert root.startswith("t")
    assert f"(let {root} " in program


def test_translate_cumsum_honors_output_dtype(tmp_path: Path):
    # Qwen's grouped-mm expert path computes float histograms and converts
    # their cumulative offsets to int32 through cumsum's dtype argument.
    exported = torch.export.export(
        CumsumWithOutputDtype(),
        (torch.tensor([1.0, 2.0, 3.0]),),
        strict=False,
    )
    pt2_path = tmp_path / "cumsum_int32.pt2"
    torch.export.save(exported, pt2_path)

    dot = translate_pt2_to_dot(str(pt2_path))

    assert dot.startswith("digraph")
    # GraphTensor lowers cumsum into gather/mask/reduce primitives. The cast
    # must precede those primitives so grouped-mm receives integer offsets.
    assert "Cast(3, Int)" in dot
    assert "SumReduce" in dot
