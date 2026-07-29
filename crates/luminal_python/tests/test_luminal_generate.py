"""luminal_generate — compile-once HF generation over a retained artifact.

Decode (and every prefill after the first) re-executes the same compiled
graph, which the single-execute reference runtime cannot do — those tests are
gated to CUDA runs (see requires_persistent_backend). Validation and
error-path tests run everywhere.
"""

import os

import pytest
import torch
from luminal import BackendHooks, luminal_backend, luminal_generate
from test_writeback_mutations import tiny_llama

# The reference (CPU) runtime consumes its input buffers after execute() —
# including compile-time-loaded PT2 constants — so a compiled graph is
# single-execute there. The decode loop re-executes the retained graph every
# step, which needs a backend with persistent buffers (cuda_lite keeps inputs
# device-resident).
requires_persistent_backend = pytest.mark.skipif(
    os.getenv("LUMINAL_TEST_DEVICE", "cpu").lower() != "cuda",
    reason="ReferenceRuntime is single-execute (input buffers consumed after "
    "run); re-executing the retained graph needs a persistent backend — "
    "run with LUMINAL_TEST_DEVICE=cuda",
)


def _eager_greedy(model, input_ids, max_new_tokens):
    return model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=0,
    )


@requires_persistent_backend
def test_generate_matches_eager_greedy(device):
    """model.generate(custom_generate=luminal_generate) is token-for-token
    identical to stock eager greedy generation."""
    config, model = tiny_llama()
    model = model.to(device)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

    ref = _eager_greedy(model, input_ids, max_new_tokens=8)
    out = model.generate(
        input_ids,
        custom_generate=luminal_generate,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=0,
        return_dict_in_generate=True,
    )
    assert torch.equal(out.sequences.cpu(), ref.cpu())
    assert out.luminal_ttft_ms > 0 and out.luminal_tpot_ms > 0


@requires_persistent_backend
def test_compile_once_and_reuse(device):
    """A second generate() on the same model reuses the retained artifact —
    the backend is invoked exactly once — and still matches eager."""
    config, model = tiny_llama()
    model = model.to(device)
    invocations = []

    def counting_backend(gm, example_inputs, options=None):
        invocations.append(gm)
        return luminal_backend(gm, example_inputs)

    prompts = [
        torch.tensor([[1, 2, 3, 4, 5]], device=device),
        torch.tensor([[9, 8, 7]], device=device),
    ]
    refs = [_eager_greedy(model, p, max_new_tokens=6) for p in prompts]
    outs = [
        model.generate(
            p,
            custom_generate=luminal_generate,
            backend=counting_backend,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=0,
            max_cache_length=32,
            return_dict_in_generate=True,
        )
        for p in prompts
    ]
    assert len(invocations) == 1
    for out, ref in zip(outs, refs):
        assert torch.equal(out.sequences, ref)


@requires_persistent_backend
def test_cache_start_pos_prefill_reuse(device):
    """Continuing the persisted cache with cache_start_pos matches eager
    generation over the concatenated sequence."""
    config, model = tiny_llama()
    model = model.to(device)
    part_a = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    part_b = torch.tensor([[11, 12, 13]], device=device)

    out_a = model.generate(
        part_a,
        custom_generate=luminal_generate,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=0,
        max_cache_length=32,
        return_dict_in_generate=True,
    )
    # Continue: the device cache holds part_a + its 4 generated tokens.
    consumed = part_a.shape[1] + len(out_a.luminal_generated_ids)
    out_b = model.generate(
        part_b,
        custom_generate=luminal_generate,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=0,
        cache_start_pos=consumed,
        return_dict_in_generate=True,
    )
    full_prompt = torch.cat([out_a.sequences, part_b], dim=-1)
    ref = _eager_greedy(model, full_prompt, max_new_tokens=4)
    assert torch.equal(out_b.sequences, ref[:, -out_b.sequences.shape[1] :])


class _SpyHooks(BackendHooks):
    def __init__(self):
        self.before_compile_calls = []
        self.after_compile_calls = []
        self.after_warmup_calls = []
        self.step_positions = []

    def before_compile(self, ctx):
        self.before_compile_calls.append(ctx)

    def after_compile(self, compiled, example_inputs):
        self.after_compile_calls.append(compiled)

    def after_warmup(self, model, compiled):
        self.after_warmup_calls.append(compiled)

    def on_step(self, position):
        self.step_positions.append(position)


@requires_persistent_backend
def test_hooks_spy(device):
    """The hooks protocol fires at the documented points with the documented
    payloads."""
    config, model = tiny_llama()
    model = model.to(device)
    spy = _SpyHooks()
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)

    out = model.generate(
        input_ids,
        custom_generate=luminal_generate,
        hooks=spy,
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=0,
        max_cache_length=16,
        return_dict_in_generate=True,
    )
    assert len(spy.before_compile_calls) == 1
    ctx = spy.before_compile_calls[0]
    assert ctx.max_cache_len == 16 and ctx.prompt_len == 4 and ctx.max_new_tokens == 3
    assert len(spy.after_compile_calls) == 1
    assert len(spy.after_warmup_calls) == 1
    # Traced prefill at 0, then one decode per generated token after the first.
    n_decode = len(out.luminal_generated_ids) - 1
    assert spy.step_positions == [0] + [4 + i for i in range(n_decode)]


def test_entry_point_lookup():
    """torch.compile(backend="luminal") resolves through the entry point."""
    from torch._dynamo.backends.registry import lookup_backend

    try:
        resolved = lookup_backend("luminal")
    except Exception:
        pytest.skip("luminal entry-point metadata not installed in this env")
    assert resolved is not None


def test_errors(device):
    config, model = tiny_llama()
    model = model.to(device)
    ids = torch.tensor([[1, 2, 3]], device=device)

    with pytest.raises(RuntimeError, match="greedy"):
        model.generate(
            ids,
            custom_generate=luminal_generate,
            max_new_tokens=2,
            do_sample=True,
            pad_token_id=0,
        )
    with pytest.raises(RuntimeError, match="batch-1"):
        model.generate(
            torch.tensor([[1, 2], [3, 4]], device=device),
            custom_generate=luminal_generate,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=0,
        )
    with pytest.raises(RuntimeError, match="cache_start_pos requires a prior"):
        model.generate(
            ids,
            custom_generate=luminal_generate,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=0,
            cache_start_pos=4,
        )
    # Compile, then exceed the cache geometry on reuse. max_new_tokens=1
    # keeps this to a single execute (the traced prefill), so it runs on the
    # single-execute reference backend too.
    model.generate(
        ids,
        custom_generate=luminal_generate,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=0,
        max_cache_length=8,
    )
    with pytest.raises(RuntimeError, match="outside the compiled cache length"):
        model.generate(
            ids,
            custom_generate=luminal_generate,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=0,
        )
