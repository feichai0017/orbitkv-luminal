"""Swap compile-hostile model code for equivalent, traceable spellings.

Some upstream implementations are written in a way that is fine to execute but
pathological to trace. The motivating case is Hugging Face's gated delta-net
(`Qwen3_5GatedDeltaNet` and its siblings), whose chunked delta rule inverts a
unit lower-triangular matrix with a 63-iteration Python loop:

    for i in range(1, chunk_size):              # chunk_size is a constant 64
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

Dynamo unrolls that to 21 ATen nodes per iteration -- 1,323 per
linear-attention layer, 31,752 across a 24-layer model, ~79% of the whole
graph -- and the trip count does not depend on sequence length, so every
forward pays it. Translating the result needed 161 GB of host memory.

It cannot be repaired after tracing. Each iteration slices `attn[..., i, :i]`,
so the widths ramp 1..63 and no two iterations share a shape; re-rolling needs
isomorphic bodies, and the parameterization that made them "the same" existed
only in the Python source. Hence a source-level swap, applied before tracing.

`patch_model` is the entry point. It is opt-in for `torch.compile` users --
by the time a backend is invoked, tracing has already happened -- and applied
automatically by `luminal.pt2.compile`, which still holds the module.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

__all__ = ["patch_model", "unpatch_model"]

_ATTR = "chunk_gated_delta_rule"
_SAVED = "_luminal_original_delta_rule"


class LuminalPatchedModule:
    """Marker base so patched modules are identifiable and re-patching is a no-op."""


def _unit_lower_inverse(a):
    """`(I - a)^-1` for batched strictly-lower-triangular `a`, in log2(n) levels.

    Replaces the upstream n-1 step forward substitution. Each level merges
    adjacent diagonal blocks pairwise; written on full matrices that merge is

        inv <- inv - inv @ lower @ inv

    where `lower` masks in only this level's sub-diagonal blocks, which is
    exactly

        [[m11, 0  ]^-1  =  [[ i11,           0  ]
         [m21, m22]]        [ -i22 m21 i11, i22 ]]

    Every level is two matmuls on uniformly shaped operands, so the traced form
    stays small and its iterations are isomorphic.

    NOT the Neumann series `sum_k a^k` by repeated squaring. The two are
    mathematically identical and the latter is unusable in fp32: on real
    Qwen3.5 activations |a|max ~ 0.96 while |a^16|max ~ 3e7, seven orders of
    magnitude above the answer, which must then cancel back down. Measured 380%
    error, and cosine 0.9910 with 76% token agreement end to end -- a wrong
    answer that still compiles. This formulation never forms a power of `a` and
    matches forward substitution to ~4e-07 in fp32.
    """
    n = a.shape[-1]
    off_diagonal = -a  # the off-diagonal blocks of (I - a); its diagonal is 1
    index = torch.arange(n, device=a.device)
    inv = torch.eye(n, dtype=a.dtype, device=a.device).expand_as(a)
    block = 1
    while block < n:
        row_block = (index // block).unsqueeze(-1)
        col_block = (index // block).unsqueeze(0)
        # keep block (2j+1, 2j) for every j: odd row-block, paired even col-block
        keep = ((row_block % 2 == 1) & (col_block == row_block - 1)).to(a.dtype)
        inv = inv - inv @ (off_diagonal * keep) @ inv
        block *= 2
    return inv


def _make_chunk_gated_delta_rule(l2norm):
    """Build the replacement, closing over the source module's own `l2norm`.

    Each model family (qwen3_5, qwen3_5_moe, qwen3_next, olmo_hybrid) defines
    its own copy, so bind the one belonging to the class being patched rather
    than importing from a single hard-coded module.
    """

    def chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        """Line-for-line the upstream `torch_chunk_gated_delta_rule`, except the
        forward-substitution loop, which becomes `_unit_lower_inverse`."""
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=0,
        )

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

        # Upstream runs 63 forward-substitution steps here, leaving
        # sum_{j>=1} attn^j, then adds I -- i.e. (I - attn)^-1. Compute that
        # inverse directly; the trailing `+ eye` is subsumed.
        attn = _unit_lower_inverse(attn)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(
                batch_size,
                num_heads,
                k_head_dim,
                v_head_dim,
                dtype=value.dtype,
                device=value.device,
            )
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        )

        # The chunk loop stays: its iterations are uniformly shaped, so unlike
        # the row loop above they re-roll and cost only what they compute.
        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    return chunk_gated_delta_rule


_patched_classes: dict[type, type] = {}


def _patched_class(cls):
    """Return (and cache) the Luminal subclass of `cls`."""
    if cls in _patched_classes:
        return _patched_classes[cls]

    source = sys.modules.get(cls.__module__)
    l2norm = getattr(source, "l2norm", None)
    if l2norm is None:
        raise AttributeError(
            f"{cls.__module__} has no `l2norm`; the delta-rule replacement cannot "
            "be built against this version of transformers"
        )
    rule = _make_chunk_gated_delta_rule(l2norm)

    def __init__(self, *args, **kwargs):
        super(patched, self).__init__(*args, **kwargs)
        setattr(self, _ATTR, rule)

    patched = type(
        f"Luminal{cls.__name__}",
        (cls, LuminalPatchedModule),
        {
            "__doc__": (
                f"{cls.__name__} with a traceable chunked delta rule. "
                "See luminal.model_patches."
            ),
            "__init__": __init__,
            "_luminal_chunk_rule": staticmethod(rule),
        },
    )
    _patched_classes[cls] = patched
    return patched


def patch_model(model) -> int:
    """Replace compile-hostile submodule implementations in `model`, in place.

    Returns the number of modules patched. Idempotent. Safe to call on a model
    with nothing to patch, and on a machine without transformers installed.

    Nothing is reallocated: the module object, its parameters and their storage
    are untouched, so `data_ptr()`s are stable across the call.
    """
    patched = 0
    for module in model.modules():
        if isinstance(module, LuminalPatchedModule):
            continue
        original = getattr(module, _ATTR, None)
        if original is None:
            continue
        cls = type(module)
        # Both steps are required. Reassigning __class__ does not re-run
        # __init__, and the upstream __init__ stores the rule as an *instance*
        # attribute, which would shadow anything the class provides. The class
        # swap makes the change identifiable; the attribute makes it take effect.
        module.__class__ = _patched_class(cls)
        module.__dict__[_SAVED] = (cls, original)
        setattr(module, _ATTR, type(module)._luminal_chunk_rule)
        patched += 1
    return patched


def unpatch_model(model) -> int:
    """Undo `patch_model`, restoring the original class and implementation."""
    restored = 0
    for module in model.modules():
        saved = module.__dict__.pop(_SAVED, None)
        if saved is None:
            continue
        cls, original = saved
        module.__class__ = cls
        setattr(module, _ATTR, original)
        restored += 1
    return restored
