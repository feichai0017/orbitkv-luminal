"""Torture tests for scalar (rank-0) tensor handling.

Most tests in this suite use ``torch.allclose`` which is silent about shape.
That hides discrepancies between PyTorch (where ``x.sum()`` is shape ``()``)
and luminal (where the same op may produce shape ``(1,)``). These tests
assert dtype, shape, AND values, so any rank-0 vs rank-1 drift fails loudly.

The tests cover:
 - Per-op rank-0 production (sum, max, mean, min, prod, indexing, constants)
 - Sequences of unsqueeze/squeeze/expand/reshape that round-trip through scalar
 - Scalars participating in arithmetic, comparisons, mod, where
 - Models that return scalars as their final output

Each test compiles a small ``nn.Module`` with the luminal backend and compares
to PyTorch eager.
"""

from typing import Callable

import pytest
import torch
import torch._dynamo

from luminal import luminal_backend


# ---------------------------------------------------------------------------
# Strict comparison helper: catches shape / dtype divergence in addition to
# value differences. This is the rigor that ``torch.allclose`` lacks.
# ---------------------------------------------------------------------------


def _strict_match(
    output: torch.Tensor,
    original: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    assert output.dtype == original.dtype, (
        f"dtype mismatch: luminal={output.dtype} vs eager={original.dtype}"
    )
    assert tuple(output.shape) == tuple(original.shape), (
        f"shape mismatch: luminal={tuple(output.shape)} vs "
        f"eager={tuple(original.shape)} (rank {output.dim()} vs {original.dim()})"
    )
    if output.numel() == 0:
        return
    if output.dtype.is_floating_point:
        assert torch.allclose(output, original, atol=atol, rtol=rtol), (
            f"value mismatch (max abs err: "
            f"{(output - original).abs().max().item()})"
        )
    else:
        assert torch.equal(output, original), (
            f"value mismatch: luminal={output} vs eager={original}"
        )


def _run(
    model: torch.nn.Module, *inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (eager_output, compiled_output) for matched comparison."""
    compiled: Callable = torch.compile(model, backend=luminal_backend)
    eager = model(*inputs)
    compiled_out = compiled(*inputs)
    return eager, compiled_out


# ---------------------------------------------------------------------------
# Section 1: Full reductions produce a rank-0 scalar.
# ---------------------------------------------------------------------------


class _SumAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _MaxAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max()


class _MinAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.min()


class _MeanAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class _ProdAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.prod()


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)])
def test_sum_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_SumAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_max_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MaxAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_min_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MinAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_mean_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MeanAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_prod_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    # Use values close to 1.0 so the product stays well-conditioned.
    x = torch.rand(shape, device=device) * 0.5 + 0.75
    eager, compiled = _run(_ProdAll(), x)
    _strict_match(compiled, eager, atol=1e-4)


# ---------------------------------------------------------------------------
# Section 2: insert-dim / remove-dim sequences round-trip through scalar.
# ---------------------------------------------------------------------------


class _SumUnsqueezeSqueeze(torch.nn.Module):
    """sum -> () -> unsqueeze(0) -> (1,) -> squeeze(0) -> ()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().unsqueeze(0).squeeze(0)


class _SumDoubleUnsqueezeDoubleSqueeze(torch.nn.Module):
    """sum -> [u(0), u(0), s(0), s(0)] back to scalar."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().unsqueeze(0).unsqueeze(0).squeeze(0).squeeze(0)


class _UnsqueezeNegativeAxis(torch.nn.Module):
    """sum -> unsqueeze(-1) -> squeeze(-1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().unsqueeze(-1).squeeze(-1)


class _NestedUnsqueezeSqueezeAll(torch.nn.Module):
    """sum -> u(0)*3 -> squeeze() (squeeze with no dim removes ALL size-1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        s = s.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return s.squeeze()


class _AlternatingUnsqueezeSqueeze(torch.nn.Module):
    """Insert and remove dims in alternation, ending at scalar."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        s = s.unsqueeze(0)  # (1,)
        s = s.squeeze(0)  # ()
        s = s.unsqueeze(0).unsqueeze(0)  # (1, 1)
        s = s.squeeze(-1).squeeze(-1)  # ()
        return s


class _ReduceKeepDimThenSqueeze(torch.nn.Module):
    """sum(keepdim=True) -> (1, 1) -> squeeze() -> ()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(0, 1), keepdim=True).squeeze()


class _ReshapeToFromScalar(torch.nn.Module):
    """sum -> reshape(()) -> reshape((1,)) -> reshape(()) — explicit scalar reshapes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        return s.reshape(()).reshape((1,)).reshape(())


class _UnsqueezeExpandSumBack(torch.nn.Module):
    """() -> (1,) -> (5,) (expand) -> sum back to ()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        return s.unsqueeze(0).expand(5).sum()


def test_unsqueeze_squeeze_roundtrip(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_SumUnsqueezeSqueeze(), x)
    _strict_match(compiled, eager)


def test_double_unsqueeze_double_squeeze(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_SumDoubleUnsqueezeDoubleSqueeze(), x)
    _strict_match(compiled, eager)


def test_unsqueeze_negative_axis(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_UnsqueezeNegativeAxis(), x)
    _strict_match(compiled, eager)


def test_nested_unsqueeze_squeeze_all(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_NestedUnsqueezeSqueezeAll(), x)
    _strict_match(compiled, eager)


def test_alternating_unsqueeze_squeeze(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_AlternatingUnsqueezeSqueeze(), x)
    _strict_match(compiled, eager)


def test_reduce_keepdim_then_squeeze_to_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReduceKeepDimThenSqueeze(), x)
    _strict_match(compiled, eager)


def test_reshape_to_and_from_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReshapeToFromScalar(), x)
    _strict_match(compiled, eager)


@pytest.mark.xfail(
    reason=(
        "Full reduction (x.sum()) returns shape [1] instead of () in "
        "translator/reduction.rs; downstream unsqueeze(0).expand(5) then "
        "tries to contract rank-2 (1,1) to rank-1, panicking with "
        "'Cannot expand from 2 dims to 1 dims'."
    ),
    strict=False,
)
def test_unsqueeze_expand_sum_back(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_UnsqueezeExpandSumBack(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 3: Scalars broadcast against rank-N tensors in arithmetic chains.
# ---------------------------------------------------------------------------


class _NormalizeBySum(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.sum()


class _CenterByMean(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean()


class _MinMaxScale(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min() + 1e-6)


class _DoubleScalarBroadcast(torch.nn.Module):
    """Two independent scalar reductions multiplied, then broadcast onto x."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * (y.sum() * x.mean())


class _ScalarChainedArithmetic(torch.nn.Module):
    """Long chain of scalar ops: ((s+1) * 2 - 0.5) used to scale x."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        s = (s + 1.0) * 2.0 - 0.5
        return x * s


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_normalize_by_sum(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) + 0.1
    eager, compiled = _run(_NormalizeBySum(), x)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_center_by_mean(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_CenterByMean(), x)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_minmax_scale(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MinMaxScale(), x)
    _strict_match(compiled, eager, atol=1e-4)


def test_double_scalar_broadcast(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    y = torch.rand((2, 5), device=device)
    eager, compiled = _run(_DoubleScalarBroadcast(), x, y)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_scalar_chained_arithmetic(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_ScalarChainedArithmetic(), x)
    _strict_match(compiled, eager, atol=1e-4)


# ---------------------------------------------------------------------------
# Section 4: 0-d tensor constants in the graph.
# ---------------------------------------------------------------------------


class _AddScalarTensorConst(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tensor(2.5).to(x.device)


class _MulScalarTensorConst(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tensor(0.5).to(x.device)


class _ClampWithScalarTensors(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(-0.25).to(x.device)
        hi = torch.tensor(0.75).to(x.device)
        return torch.clamp(x, lo, hi)


class _WhereWithScalarBranches(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.tensor(0.0).to(x.device)
        one = torch.tensor(1.0).to(x.device)
        return torch.where(x > 0.5, one, zero)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_add_scalar_tensor_constant(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_AddScalarTensorConst(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_mul_scalar_tensor_constant(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MulScalarTensorConst(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_clamp_with_scalar_tensors(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 2.0 - 1.0  # in [-1, 1]
    eager, compiled = _run(_ClampWithScalarTensors(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_where_with_scalar_branches(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_WhereWithScalarBranches(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 5: Comparisons that produce or consume scalars.
# ---------------------------------------------------------------------------


class _ScalarLtScalar(torch.nn.Module):
    """Compare two scalar reductions; result is a 0-d bool, cast to float."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x.sum() < y.sum()).float()


class _ThresholdByMean(torch.nn.Module):
    """tensor > scalar — scalar broadcasts in comparison."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > x.mean()).float()


class _MaskedByScalarThreshold(torch.nn.Module):
    """Use a scalar comparison as a mask back into the tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x > x.mean()).to(x.dtype)


def test_scalar_lt_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    y = torch.rand((2, 5), device=device)
    eager, compiled = _run(_ScalarLtScalar(), x, y)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_threshold_by_mean(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_ThresholdByMean(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_masked_by_scalar_threshold(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MaskedByScalarThreshold(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 6: Mod with scalar RHS — exercises broadcasting through luminal Rem.
# ---------------------------------------------------------------------------


class _ModByScalarTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x % torch.tensor(3.0).to(x.device)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_mod_by_scalar_tensor(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 10.0
    eager, compiled = _run(_ModByScalarTensor(), x)
    _strict_match(compiled, eager, atol=1e-4)


# ---------------------------------------------------------------------------
# Section 7: Indexing / select that produces a 0-d scalar.
# ---------------------------------------------------------------------------


class _Index1DToScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[0]


class _IndexAllDimsToScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[1, 2, 3]


class _IndexThenAddScalarConst(torch.nn.Module):
    """Indexed scalar enters arithmetic with a scalar constant."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[1, 2] + torch.tensor(7.0).to(x.device)


def test_index_1d_produces_scalar(device: torch.device) -> None:
    x = torch.rand((5,), device=device)
    eager, compiled = _run(_Index1DToScalar(), x)
    _strict_match(compiled, eager)


def test_index_all_dims_produces_scalar(device: torch.device) -> None:
    x = torch.rand((4, 5, 6), device=device)
    eager, compiled = _run(_IndexAllDimsToScalar(), x)
    _strict_match(compiled, eager)


def test_index_then_add_scalar_const(device: torch.device) -> None:
    x = torch.rand((4, 5), device=device)
    eager, compiled = _run(_IndexThenAddScalarConst(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 8: Models whose final output is a scalar.
# ---------------------------------------------------------------------------


class _ReturnScalarSum(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _ReturnScalarFromIndex(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[0, 0]


class _ReturnDerivedScalar(torch.nn.Module):
    """Return scalar built from constant and reduction — no input shape leak."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(3.14).to(x.device) + 0.0 * x.sum()


def test_model_returns_scalar_sum(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReturnScalarSum(), x)
    _strict_match(compiled, eager)


def test_model_returns_scalar_from_index(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReturnScalarFromIndex(), x)
    _strict_match(compiled, eager)


def test_model_returns_derived_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReturnDerivedScalar(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 9: Mixed dtype scalars.
# ---------------------------------------------------------------------------


class _IntSumProducesIntScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _FloatScalarTimesIntTensor(torch.nn.Module):
    """Scalar float constant + int tensor — exercises promotion rules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * torch.tensor(0.5).to(x.device)


def test_int_sum_produces_int_scalar(device: torch.device) -> None:
    x = torch.randint(0, 10, (3, 4), device=device, dtype=torch.int64)
    eager, compiled = _run(_IntSumProducesIntScalar(), x)
    _strict_match(compiled, eager)


def test_float_scalar_times_int_tensor(device: torch.device) -> None:
    x = torch.randint(0, 10, (3, 4), device=device, dtype=torch.int64)
    eager, compiled = _run(_FloatScalarTimesIntTensor(), x)
    _strict_match(compiled, eager)
