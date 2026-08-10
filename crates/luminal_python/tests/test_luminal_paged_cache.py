import json
from types import SimpleNamespace

import pytest
import torch

from luminal.cache_utils import _register_cache_serialization
from luminal.transformers_cache import LuminalPagedCache


def _config(num_layers=2, num_kv_heads=2, head_dim=4):
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=4,
        hidden_size=4 * head_dim,
        head_dim=head_dim,
        sliding_window=None,
    )


def _coordinate_states(start: int, tokens: int, heads: int, dim: int):
    values = torch.empty(1, heads, tokens, dim)
    for head in range(heads):
        for token in range(tokens):
            for column in range(dim):
                values[0, head, token, column] = (
                    head * 10_000 + (start + token) * 100 + column
                )
    return values


def _update(cache, keys, values, positions):
    return cache.update(
        keys,
        values,
        0,
        {"cache_position": torch.tensor(positions, dtype=torch.int32)},
    )


def test_paged_cache_requires_page_aligned_capacity():
    with pytest.raises(ValueError, match="multiple of page_size"):
        LuminalPagedCache(
            _config(num_layers=1),
            max_cache_len=10,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
        )


def test_paged_cache_uses_real_fixed_pages_and_hf_view():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    layer = cache.layers[0]
    keys = _coordinate_states(0, 3, heads=2, dim=4)
    values = keys + 50_000

    actual_keys, actual_values = _update(cache, keys, values, [0, 1, 2])

    assert layer.keys.shape == (2, 4, 2, 4)
    assert layer.values.shape == (2, 4, 2, 4)
    assert layer.block_table.tolist() == [0, 1]
    assert layer.block_table.dtype == torch.int32
    assert layer.sequence_length.tolist() == [3]
    torch.testing.assert_close(actual_keys[:, :, :3], keys)
    torch.testing.assert_close(actual_values[:, :, :3], values)
    torch.testing.assert_close(actual_keys[:, :, 3:], torch.zeros_like(actual_keys[:, :, 3:]))


def test_paged_cache_crosses_page_boundary_without_reallocation():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    layer = cache.layers[0]
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            layer.keys,
            layer.values,
            layer.block_table,
            layer.sequence_length,
        )
    )
    first_keys = _coordinate_states(0, 3, heads=2, dim=4)
    first_values = first_keys + 50_000
    _update(cache, first_keys, first_values, [0, 1, 2])
    next_keys = _coordinate_states(3, 3, heads=2, dim=4)
    next_values = next_keys + 50_000

    actual_keys, actual_values = _update(cache, next_keys, next_values, [3, 4, 5])

    torch.testing.assert_close(actual_keys[:, :, :6], torch.cat((first_keys, next_keys), dim=2))
    torch.testing.assert_close(
        actual_values[:, :, :6], torch.cat((first_values, next_values), dim=2)
    )
    assert int(cache.get_seq_length()) == 6
    assert tuple(
        tensor.data_ptr()
        for tensor in (
            layer.keys,
            layer.values,
            layer.block_table,
            layer.sequence_length,
        )
    ) == pointers


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_cache_preserves_native_16bit_storage(dtype):
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        page_size=4,
        dtype=dtype,
        device="cpu",
    )
    keys = _coordinate_states(0, 2, heads=2, dim=4).to(dtype)
    values = (keys.float() + 50).to(dtype)

    actual_keys, actual_values = _update(cache, keys, values, [0, 1])

    assert cache.layers[0].keys.dtype == dtype
    assert cache.layers[0].values.dtype == dtype
    torch.testing.assert_close(actual_keys[:, :, :2], keys)
    torch.testing.assert_close(actual_values[:, :, :2], values)


def test_paged_cache_reports_capacity_overflow_before_writing():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    keys = _coordinate_states(0, 2, heads=2, dim=4)

    with pytest.raises(IndexError, match="capacity exceeded"):
        _update(cache, keys, keys, [7, 8])


def test_paged_cache_obeys_noncontiguous_block_table():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    layer = cache.layers[0]
    layer.block_table.copy_(torch.tensor([1, 0], dtype=torch.int32))
    keys = _coordinate_states(0, 6, heads=2, dim=4)
    values = keys + 50_000

    actual_keys, actual_values = _update(cache, keys, values, list(range(6)))

    torch.testing.assert_close(actual_keys[:, :, :6], keys)
    torch.testing.assert_close(actual_values[:, :, :6], values)
    # Logical page zero was deliberately assigned to physical page one.
    expected_first_page = keys[:, :, :4].permute(0, 2, 1, 3).squeeze(0)
    torch.testing.assert_close(layer.keys[1], expected_first_page)


def test_paged_cache_reset_only_mutates_device_length():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    layer = cache.layers[0]
    keys = _coordinate_states(0, 3, heads=2, dim=4)
    _update(cache, keys, keys + 1, [0, 1, 2])
    pointers = (layer.keys.data_ptr(), layer.values.data_ptr())
    old_keys = layer.keys.clone()
    old_values = layer.values.clone()

    cache.reset()

    assert int(cache.get_seq_length()) == 0
    assert (layer.keys.data_ptr(), layer.values.data_ptr()) == pointers
    torch.testing.assert_close(layer.keys, old_keys)
    torch.testing.assert_close(layer.values, old_values)


def test_paged_cache_pytree_roundtrip_preserves_all_state_pointers():
    _register_cache_serialization()
    cache = LuminalPagedCache(
        _config(),
        max_cache_len=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    states = _coordinate_states(0, 2, heads=2, dim=4)
    _update(cache, states, states + 1, [0, 1])

    leaves, spec = torch.utils._pytree.tree_flatten(cache)
    rebuilt = torch.utils._pytree.tree_unflatten(leaves, spec)

    assert isinstance(rebuilt, LuminalPagedCache)
    assert len(rebuilt.layers) == 2
    for original, restored in zip(cache.layers, rebuilt.layers):
        assert restored.page_size == 4
        assert restored.max_cache_len == 8
        for lhs, rhs in (
            (original.keys, restored.keys),
            (original.values, restored.values),
            (original.block_table, restored.block_table),
            (original.sequence_length, restored.sequence_length),
        ):
            assert lhs.data_ptr() == rhs.data_ptr()


def _tiny_qwen():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attn_implementation="eager",
        use_cache=True,
    )
    config._experts_implementation = "grouped_mm"
    return config, Qwen3MoeForCausalLM(config).eval()


def test_tiny_hf_qwen_paged_cache_matches_dynamic_cache_across_pages():
    from transformers import DynamicCache

    torch.manual_seed(0)
    config, model = _tiny_qwen()
    reference_cache = DynamicCache(config=config)
    paged_cache = LuminalPagedCache(
        config,
        max_cache_len=8,
        page_size=2,
        dtype=torch.float32,
        device="cpu",
    )

    with torch.no_grad():
        for input_ids in (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4]]),
            torch.tensor([[5]]),
        ):
            expected = model(
                input_ids=input_ids,
                past_key_values=reference_cache,
                use_cache=True,
            )
            actual = model(
                input_ids=input_ids,
                past_key_values=paged_cache,
                use_cache=True,
            )
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=1e-6)

    assert int(paged_cache.get_seq_length()) == reference_cache.get_seq_length() == 5


def test_tiny_hf_qwen_export_records_only_fixed_state_mutations():
    from torch.export.graph_signature import OutputKind

    config, model = _tiny_qwen()
    cache = LuminalPagedCache(
        config,
        max_cache_len=8,
        page_size=2,
        dtype=torch.float32,
        device="cpu",
    )

    exported = torch.export.export(
        model,
        args=(),
        kwargs={
            "input_ids": torch.tensor([[1]]),
            "past_key_values": cache,
            "use_cache": True,
        },
        strict=False,
    ).run_decompositions()

    mutations = [
        output
        for output in exported.graph_signature.output_specs
        if output.kind == OutputKind.USER_INPUT_MUTATION
    ]
    # One layer mutates its K pages, V pages, and device sequence length.  Its
    # fixed block table is read-only and therefore must not become a writeback.
    assert len(mutations) == 3
    mutation_targets = {
        output.target for output in mutations if isinstance(output.target, str)
    }
    assert any("key_pages" in target for target in mutation_targets)
    assert any("value_pages" in target for target in mutation_targets)
    assert any("sequence_lengths" in target for target in mutation_targets)
    assert not any("block_tables" in target for target in mutation_targets)

    targets = {str(node.target) for node in exported.graph_module.graph.nodes}
    # Functionalization lowers the in-place page writes to pure index_put
    # values paired with the USER_INPUT_MUTATION outputs above.
    assert "aten.index_put.default" in targets
    assert "aten.index_select.default" in targets

    def placeholder_ancestors(node):
        pending = list(node.all_input_nodes)
        visited = set()
        placeholders = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current.op == "placeholder":
                placeholders.add(current.name)
            else:
                pending.extend(current.all_input_nodes)
        return placeholders

    # Qwen's RoPE implementation legitimately contains concatenations.  What
    # must be absent is DynamicCache-style concatenation of prior cache state.
    for node in exported.graph_module.graph.nodes:
        if "cat" in str(node.target):
            assert not any(
                "past_key_values_key_pages" in name
                or "past_key_values_value_pages" in name
                for name in placeholder_ancestors(node)
            )


def test_index_select_translation_matches_reference_runtime(tmp_path):
    """The page-table primitive is correct below the cache abstraction."""
    from luminal import CompiledModel, process_pt2
    from luminal.luminal import _reference_factory_capsule

    class SelectRows(torch.nn.Module):
        def forward(self, pages, block_table):
            return torch.index_select(pages, 0, block_table.to(torch.int64))

    pages = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    block_table = torch.tensor([2, 0, 1], dtype=torch.int32)
    exported = torch.export.export(
        SelectRows(), (pages, block_table), strict=False
    ).run_decompositions()
    pt2_path = tmp_path / "index_select.pt2"
    torch.export.save(exported, pt2_path)

    compiled = CompiledModel(
        process_pt2(str(pt2_path), "", 0, _reference_factory_capsule())
    )
    (actual,) = compiled(pages, block_table)
    torch.testing.assert_close(actual, torch.index_select(pages, 0, block_table.long()))


def test_tiny_hf_qwen_paged_cache_translates_with_fixed_writebacks(tmp_path):
    """The real PT2 path accepts the cache and preserves its mutation ABI."""
    from luminal import process_pt2
    from luminal.luminal import _reference_factory_capsule

    config, model = _tiny_qwen()
    cache = LuminalPagedCache(
        config,
        max_cache_len=8,
        page_size=2,
        dtype=torch.float32,
        device="cpu",
    )
    exported = torch.export.export(
        model,
        args=(),
        kwargs={
            "input_ids": torch.tensor([[1]]),
            "past_key_values": cache,
            "use_cache": True,
        },
        strict=False,
    ).run_decompositions()
    pt2_path = tmp_path / "qwen_paged_cache.pt2"
    torch.export.save(exported, pt2_path)

    compiled = process_pt2(str(pt2_path), "", 0, _reference_factory_capsule())
    writebacks = dict(compiled.writeback_outputs)
    assert len(writebacks) == 3
    assert any("key_pages" in name for name in writebacks.values())
    assert any("value_pages" in name for name in writebacks.values())
    assert any("sequence_lengths" in name for name in writebacks.values())


def test_cuda_hf_qwen_paged_attention_replays_without_materialization(
    device, tmp_path, monkeypatch
):
    """Real HF/PT2/egglog/CUDA integration across a physical page boundary.

    The first decode invocation may materialize its specialization. Subsequent
    calls reuse the same token/cache addresses and must launch without walking
    or patching CUDA graph nodes.
    """
    if device.type != "cuda":
        pytest.skip("CUDA integration test")

    from transformers import Qwen3Config, Qwen3ForCausalLM

    from luminal import luminal_backend

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        attn_implementation="eager",
        use_cache=True,
    )
    model = Qwen3ForCausalLM(config).eval().to(device=device, dtype=torch.bfloat16)
    reference_cache = LuminalPagedCache(
        config,
        max_cache_len=8,
        page_size=2,
        dtype=torch.bfloat16,
        device=device,
    )
    compiled_cache = LuminalPagedCache(
        config,
        max_cache_len=8,
        page_size=2,
        dtype=torch.bfloat16,
        device=device,
    )

    trace = tmp_path / "paged_decode.jsonl"
    monkeypatch.setenv("LUMINAL_PROFILE_JSONL", str(trace))
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 8

    captured = []

    def backend(gm, example_inputs, options=None):
        artifact = luminal_backend(
            gm,
            example_inputs,
            options={"search_iterations": 10},
        )
        captured.append(artifact)
        return artifact

    compiled = torch.compile(model, backend=backend, fullgraph=True)
    prefill = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=device)
    stable_decode_input = torch.empty((1, 1), dtype=torch.int32, device=device)

    with torch.inference_mode():
        expected = model(
            input_ids=prefill,
            past_key_values=reference_cache,
            use_cache=True,
        )
        actual = compiled(
            input_ids=prefill,
            past_key_values=compiled_cache,
            use_cache=True,
        )
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0.02, atol=0.03)

        # Four fixed-address decode calls cross cache page boundaries at
        # logical positions 4 and 6.
        for token in (4, 5, 6, 7):
            stable_decode_input.fill_(token)
            expected = model(
                input_ids=stable_decode_input,
                past_key_values=reference_cache,
                use_cache=True,
            )
            actual = compiled(
                input_ids=stable_decode_input,
                past_key_values=compiled_cache,
                use_cache=True,
            )
            torch.testing.assert_close(
                actual.logits, expected.logits, rtol=0.02, atol=0.03
            )

    assert captured, "Luminal backend was not invoked"
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    decode_records = [record for record in records if record.get("phase") == "decode"]
    assert len(decode_records) >= 4
    assert any(
        record["runtime"]["selected_host_ops"].get("FlashInferAttention") == 1
        for record in decode_records
    ), "paged HF attention did not select FlashInferAttention"
    for record in decode_records[-2:]:
        runtime = record["runtime"]
        assert runtime["materialization_fast_path"] is True
        assert runtime["counts"]["graph_nodes_inspected"] == 0
        assert runtime["counts"]["graph_nodes_updated"] == 0
