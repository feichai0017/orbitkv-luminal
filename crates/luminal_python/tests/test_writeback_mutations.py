"""In-place input mutations (HF StaticCache) through the compiled path.

transformers' StaticLayer.update mutates the cache tensors (and its
cumulative_length counter) in place; torch.export functionalizes those
mutations into extra outputs declared by the graph signature. CompiledModel
applies them back to the caller's tensors and returns only the user outputs,
matching eager semantics and torch.compile's calling contract.
"""

import pytest
import torch
from luminal import luminal_backend

TINY_LLAMA_KWARGS = dict(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=64,
    use_cache=True,
    attn_implementation="eager",
)


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(**TINY_LLAMA_KWARGS)
    return config, LlamaForCausalLM(config).eval()


def make_static_cache(config, max_cache_len, device):
    from transformers.cache_utils import StaticCache

    cache = StaticCache(config=config, max_cache_len=max_cache_len)
    cache.early_initialization(
        batch_size=1,
        num_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.float32,
        device=device,
    )
    return cache


def static_cache_forward(model, cache, input_ids):
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    return model(
        input_ids,
        past_key_values=cache,
        cache_position=cache_position,
        position_ids=cache_position.unsqueeze(0),
        logits_to_keep=1,
        use_cache=True,
        return_dict=True,
    )


def test_static_cache_prefill_smoke(device):
    """One StaticCache forward through the compiled path matches eager —
    logits, cache contents, and the in-place mutation semantics."""
    config, model = tiny_llama()
    model = model.to(device)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)

    ref_cache = make_static_cache(config, 16, device)
    lum_cache = make_static_cache(config, 16, device)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    with torch.inference_mode():
        ref = static_cache_forward(model, ref_cache, input_ids)
        lum = static_cache_forward(compiled, lum_cache, input_ids)
    assert torch.allclose(lum.logits, ref.logits, atol=1e-4)
    for ref_layer, lum_layer in zip(ref_cache.layers, lum_cache.layers):
        assert torch.allclose(lum_layer.keys, ref_layer.keys, atol=1e-4)
        assert torch.allclose(lum_layer.values, ref_layer.values, atol=1e-4)


def test_cuda_writeback_preserves_input_allocation(device):
    """A CUDA input mutation writes directly into its original allocation.

    Persistent state must not be read through the CPU or replaced by a fresh
    output tensor between invocations; captured library kernels rely on the
    input pointer remaining stable.
    """
    if device.type != "cuda":
        pytest.skip("direct device-pointer writeback is CUDA-only")

    class UpdateState(torch.nn.Module):
        def forward(self, state, positions, values):
            state.index_put_((positions,), values)
            # Returning the mutated tensor creates both a mutation output and
            # a user output with the same exported tensor name. Both must stay
            # bound to the original persistent-state allocation.
            return state, state.sum()

    compiled = torch.compile(
        UpdateState().to(device), backend=luminal_backend, fullgraph=True
    )
    state = torch.zeros(8, 4, device=device)
    original_ptr = state.data_ptr()

    with torch.inference_mode():
        for position in (1, 5):
            positions = torch.tensor([position], dtype=torch.int64, device=device)
            values = torch.full((1, 4), float(position), device=device)
            expected = state.clone()
            expected.index_put_((positions,), values)
            actual_state, actual_sum = compiled(state, positions, values)
            torch.testing.assert_close(actual_sum, expected.sum())
            torch.testing.assert_close(actual_state, expected)
            torch.testing.assert_close(state, expected)
            assert actual_state.data_ptr() == original_ptr
            assert state.data_ptr() == original_ptr


def test_cuda_int32_output_never_reads_back_through_host(device):
    """Native-width integer outputs use the same direct CUDA path as floats.

    LuminalPagedCache exposes one growing int32 positions output per layer. A
    call to get_output_i32() here would synchronize the stream, copy to CPU,
    construct a CPU tensor, and copy it back to CUDA. Trap that getter so this
    test proves correctness without allowing the old round-trip fallback.
    """
    if device.type != "cuda":
        pytest.skip("direct device-pointer outputs are CUDA-only")

    class IncrementPositions(torch.nn.Module):
        def forward(self, positions):
            return positions + 1

    class NoInt32HostReadback:
        def __init__(self, graph):
            self._graph = graph

        def __getattr__(self, name):
            return getattr(self._graph, name)

        def get_output_i32(self, name):
            raise AssertionError(
                f"int32 CUDA output {name!r} was copied through the host"
            )

        def copy_output_to_device_ptr(self, name, device_ptr, n_bytes):
            raise AssertionError(
                f"int32 CUDA output {name!r} was not written directly"
            )

    artifacts = []

    def trapping_backend(gm, example_inputs, options=None):
        artifact = luminal_backend(gm, example_inputs, options=options)
        # This graph is static, so the backend returns CompiledModel directly
        # rather than the lazy dynamic-shape wrapper.
        artifact._graph = NoInt32HostReadback(artifact._graph)
        artifacts.append(artifact)
        return artifact

    compiled = torch.compile(
        IncrementPositions().to(device), backend=trapping_backend, fullgraph=True
    )
    positions = torch.arange(7, dtype=torch.int32, device=device)

    with torch.inference_mode():
        actual = compiled(positions)

    assert artifacts
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual, positions + 1)


def test_writeback_metadata_exposed(device):
    """The compiled artifact names its write-back outputs and their inputs."""
    config, model = tiny_llama()
    model = model.to(device)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)

    captured = []

    def capturing_backend(gm, example_inputs, options=None):
        compiled = luminal_backend(gm, example_inputs)
        captured.append(compiled)
        return compiled

    cache = make_static_cache(config, 16, device)
    compiled_model = torch.compile(model, backend=capturing_backend, fullgraph=True)
    with torch.inference_mode():
        static_cache_forward(compiled_model, cache, input_ids)

    (compiled,) = captured
    writebacks = compiled.writeback_inputs
    # 2 layers x (keys, values) index_put mutations + 2 cumulative_length
    # counters = 6 write-back outputs, each bound to a distinct input.
    assert len(writebacks) == 2 * config.num_hidden_layers + config.num_hidden_layers
    assert len(set(writebacks.values())) == len(writebacks)
