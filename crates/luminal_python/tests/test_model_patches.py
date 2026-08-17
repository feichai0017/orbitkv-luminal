"""Tests for `luminal.model_patches`.

The equivalence test here is load-bearing rather than routine. `patch_model`
substitutes our implementation for an upstream one, so if transformers changes
the *math* (not just the spelling) we would silently compute the old thing.
Test 2 imports transformers' real function and compares against it, so a
dependency bump fails here rather than in the field.
"""

import pytest
import torch
import torch.nn.functional as F

from luminal.model_patches import (
    LuminalPatchedModule,
    _make_chunk_gated_delta_rule,
    _unit_lower_inverse,
    patch_model,
    unpatch_model,
)

qwen3_5 = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5",
    reason="transformers without qwen3_5 support",
)


def _forward_substitution(a):
    """The upstream loop, verbatim -- the reference `_unit_lower_inverse` replaces."""
    out = a.clone()
    for i in range(1, out.shape[-1]):
        row = out[..., i, :i].clone()
        sub = out[..., :i, :i].clone()
        out[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return out


def test_unit_lower_inverse_matches_forward_substitution() -> None:
    """Operand shaped like the real one: a strictly-lower-triangular Gram matrix
    of *correlated* unit vectors, which is what `k_beta @ key.T` on l2-normalized
    keys produces.

    The correlation is the point. Random keys in 64 dimensions are
    near-orthogonal, so the Gram matrix is diagonally dominant and every
    implementation agrees to fp32 epsilon. Measured: a Neumann-series inverse
    (mathematically identical, numerically unstable) passes on random input at
    |a|max = 0.98, and on a constant `tril` whose powers peak at 1e16 -- but
    fails here by 7.6e+07 against 2.0e-07 for block inversion. Replacing this
    with random input silently removes the guard.
    """
    torch.manual_seed(0)
    base = torch.randn(1, 1, 1, 16)
    keys = 0.7 * base + 0.3 * torch.randn(1, 1, 64, 16)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    a = (-(keys @ keys.transpose(-1, -2))).tril(-1)

    expected = _forward_substitution(a.double()) + torch.eye(64, dtype=torch.float64)
    actual = _unit_lower_inverse(a).double()
    rel = ((expected - actual).abs().max() / expected.abs().max()).item()
    assert rel < 1e-5, f"rel={rel:.3e}"


def _realistic_inputs(seq_len: int, batch: int = 1, heads: int = 4, dim: int = 64):
    """Inputs shaped like the ones the model actually produces.

    `beta = b.sigmoid()` and `g = -exp(A_log) * softplus(a + dt_bias)` come from
    Qwen3_5GatedDeltaNet.forward. Random g/beta are NOT a valid substitute: the
    decay term can then exceed 1, the recurrence explodes to ~1e30 and NaNs out
    by seq 128 -- in transformers' own implementation too.
    """
    torch.manual_seed(0)
    query = torch.randn(batch, seq_len, heads, dim)
    key = torch.randn(batch, seq_len, heads, dim)
    value = torch.randn(batch, seq_len, heads, dim)
    beta = torch.randn(batch, seq_len, heads).sigmoid()
    a_log, dt_bias = torch.randn(heads), torch.randn(heads)
    g = -a_log.exp() * F.softplus(torch.randn(batch, seq_len, heads) + dt_bias)
    return query, key, value, g, beta


# 16/128/320/1024 -> 1/2/5/16 chunks, so the chunk loop we deliberately kept is
# exercised alongside the row loop we replaced.
@pytest.mark.parametrize("seq_len", [16, 128, 320, 1024])
def test_matches_transformers_implementation(seq_len: int) -> None:
    ours = _make_chunk_gated_delta_rule(qwen3_5.l2norm)
    query, key, value, g, beta = _realistic_inputs(seq_len)
    kwargs = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)

    expected, expected_state = qwen3_5.torch_chunk_gated_delta_rule(
        query, key, value, g, beta, **kwargs
    )
    actual, actual_state = ours(query, key, value, g, beta, **kwargs)

    assert torch.isfinite(expected).all(), "reference produced non-finite values"
    denom = expected.abs().max().clamp_min(1e-30)
    assert ((expected - actual).abs().max() / denom).item() < 1e-5
    state_denom = expected_state.abs().max().clamp_min(1e-30)
    assert ((expected_state - actual_state).abs().max() / state_denom).item() < 1e-5


def _tiny_gated_delta_net():
    """One real GatedDeltaNet, built small so the test stays cheap.

    Built from `Qwen3_5TextConfig` directly: the linear-attention dimensions
    live there, and passing them to the top-level `Qwen3_5Config` leaves the
    nested text config at its (much larger) defaults.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    module = qwen3_5.Qwen3_5GatedDeltaNet(config, layer_idx=0)
    assert module.hidden_size == 64, "config did not take effect"
    return module


def test_patch_is_idempotent_and_reversible() -> None:
    model = torch.nn.Sequential(_tiny_gated_delta_net())
    original_cls = type(model[0])
    original_rule = model[0].chunk_gated_delta_rule

    assert patch_model(model) == 1
    assert patch_model(model) == 0, "second patch must be a no-op"
    assert isinstance(model[0], LuminalPatchedModule)
    assert isinstance(model[0], original_cls), "patched class must still be a subclass"
    assert model[0].chunk_gated_delta_rule is not original_rule

    assert unpatch_model(model) == 1
    assert type(model[0]) is original_cls
    assert model[0].chunk_gated_delta_rule is original_rule
    assert unpatch_model(model) == 0


def test_patch_does_not_reallocate_parameters() -> None:
    """The class swap must not touch storage -- the compile path aliases weights
    by `data_ptr`, and a silent reallocation would break that binding."""
    model = torch.nn.Sequential(_tiny_gated_delta_net())
    before = {name: p.data_ptr() for name, p in model.named_parameters()}
    patch_model(model)
    after = {name: p.data_ptr() for name, p in model.named_parameters()}
    assert before == after


def test_patch_model_ignores_unrelated_models() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    assert patch_model(model) == 0
    assert unpatch_model(model) == 0


def test_patched_module_output_matches_original() -> None:
    """End to end through the real module: patching must not change its output."""
    module = _tiny_gated_delta_net().eval()
    hidden = torch.randn(1, 8, 64)
    with torch.no_grad():
        expected = module(hidden)
        patch_model(torch.nn.Sequential(module))
        actual = module(hidden)
    denom = expected.abs().max().clamp_min(1e-30)
    assert ((expected - actual).abs().max() / denom).item() < 1e-4
