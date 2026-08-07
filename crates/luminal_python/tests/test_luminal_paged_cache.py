import re
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


def test_luminal_paged_cache_is_nhd_and_returns_hf_view():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        dtype=torch.float32,
        device="cpu",
    )
    keys = _coordinate_states(0, 3, heads=2, dim=4)
    values = keys + 50_000

    actual_keys, actual_values = cache.update(
        keys,
        values,
        0,
        {"cache_position": torch.arange(3)},
    )

    torch.testing.assert_close(actual_keys, keys)
    torch.testing.assert_close(actual_values, values)
    expected_pool = keys.permute(0, 2, 1, 3).reshape(3, 8)
    torch.testing.assert_close(cache.layers[0].keys[:3], expected_pool)
    assert cache.layers[0].keys.is_contiguous()
    assert cache.layers[0].keys.shape == (8, 8)
    assert cache.layers[0].positions.dtype == torch.int32
    assert actual_keys.shape == (1, 2, 3, 4)
    assert actual_keys.stride() == (8, 4, 8, 1)


def test_luminal_paged_cache_append_matches_dense_cache_semantics():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        dtype=torch.float32,
        device="cpu",
    )
    first_keys = _coordinate_states(0, 3, heads=2, dim=4)
    first_values = first_keys + 50_000
    key_ptr = cache.layers[0].keys.data_ptr()
    value_ptr = cache.layers[0].values.data_ptr()
    cache.update(
        first_keys,
        first_values,
        0,
        {"cache_position": torch.arange(3)},
    )
    next_keys = _coordinate_states(3, 2, heads=2, dim=4)
    next_values = next_keys + 50_000

    actual_keys, actual_values = cache.update(
        next_keys,
        next_values,
        0,
        {"cache_position": torch.arange(3, 5)},
    )

    torch.testing.assert_close(actual_keys, torch.cat((first_keys, next_keys), dim=2))
    torch.testing.assert_close(
        actual_values, torch.cat((first_values, next_values), dim=2)
    )
    assert cache.get_seq_length() == 5
    assert cache.get_mask_sizes(2, 0) == (7, 0)
    assert cache.layers[0].keys.data_ptr() == key_ptr
    assert cache.layers[0].values.data_ptr() == value_ptr


def test_luminal_paged_cache_reset_reuses_backing_pools():
    cache = LuminalPagedCache(
        _config(num_layers=1),
        max_cache_len=8,
        dtype=torch.float32,
        device="cpu",
    )
    layer = cache.layers[0]
    first_keys = _coordinate_states(0, 3, heads=2, dim=4)
    first_values = first_keys + 50_000
    cache.update(
        first_keys,
        first_values,
        0,
        {"cache_position": torch.arange(3)},
    )
    key_ptr = layer.keys.data_ptr()
    value_ptr = layer.values.data_ptr()
    old_keys = layer.keys.clone()
    old_values = layer.values.clone()

    cache.reset()

    assert cache.get_seq_length() == 0
    assert layer.keys.data_ptr() == key_ptr
    assert layer.values.data_ptr() == value_ptr
    # Resetting logical state must not spend time clearing the large pools.
    torch.testing.assert_close(layer.keys, old_keys)
    torch.testing.assert_close(layer.values, old_values)

    replacement_keys = _coordinate_states(20, 2, heads=2, dim=4)
    replacement_values = replacement_keys + 70_000
    actual_keys, actual_values = cache.update(
        replacement_keys,
        replacement_values,
        0,
        {"cache_position": torch.arange(2)},
    )

    torch.testing.assert_close(actual_keys, replacement_keys)
    torch.testing.assert_close(actual_values, replacement_values)
    assert layer.keys.data_ptr() == key_ptr
    assert layer.values.data_ptr() == value_ptr


def test_luminal_paged_cache_pytree_roundtrip_preserves_state():
    _register_cache_serialization()
    cache = LuminalPagedCache(
        _config(),
        max_cache_len=8,
        dtype=torch.float32,
        device="cpu",
    )
    states = _coordinate_states(0, 2, heads=2, dim=4)
    cache.update(states, states + 1, 0, {"cache_position": torch.arange(2)})

    leaves, spec = torch.utils._pytree.tree_flatten(cache)
    rebuilt = torch.utils._pytree.tree_unflatten(leaves, spec)

    assert isinstance(rebuilt, LuminalPagedCache)
    assert len(rebuilt.layers) == 2
    torch.testing.assert_close(rebuilt.layers[0].keys, cache.layers[0].keys)
    torch.testing.assert_close(
        rebuilt.layers[0].positions, cache.layers[0].positions
    )


def test_index_select_translation_matches_reference_runtime(tmp_path):
    from luminal import CompiledModel, process_pt2, translate_pt2_to_egglog
    from luminal.luminal import _reference_factory_capsule

    class SelectRows(torch.nn.Module):
        def forward(self, data, indices):
            # Matches LuminalPagedCache: compact state is int32, while the
            # PyTorch indexing API requires a temporary int64 widening.
            return torch.index_select(data, 0, indices.to(torch.int64))

    data = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    indices = torch.tensor([4, 1, 4], dtype=torch.int32)
    exported = torch.export.export(SelectRows(), (data, indices), strict=False)
    exported = exported.run_decompositions()
    exported._example_inputs = None
    pt2_path = tmp_path / "index_select.pt2"
    torch.export.save(exported, pt2_path)
    compiled = CompiledModel(
        process_pt2(str(pt2_path), "", 0, _reference_factory_capsule())
    )

    (actual,) = compiled(data, indices)
    torch.testing.assert_close(
        actual, torch.index_select(data, 0, indices.to(torch.int64))
    )

    egglog, _ = translate_pt2_to_egglog(str(pt2_path))
    assert '(Input 1 "indices" (Int))' in egglog
    assert "(Int64)" not in egglog


def test_index_only_widen_elision_does_not_remove_observable_i64(tmp_path):
    from luminal import translate_pt2_to_egglog

    class ReturnWidenedIndices(torch.nn.Module):
        def forward(self, data, indices):
            widened = indices.to(torch.int64)
            return widened, torch.index_select(data, 0, widened)

    data = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    indices = torch.tensor([4, 1, 4], dtype=torch.int32)
    exported = torch.export.export(
        ReturnWidenedIndices(), (data, indices), strict=False
    ).run_decompositions()
    exported._example_inputs = None
    pt2_path = tmp_path / "observable_i64_index.pt2"
    torch.export.save(exported, pt2_path)

    egglog, _ = translate_pt2_to_egglog(str(pt2_path))
    assert "(Int64)" in egglog


def test_tiny_hf_qwen_matches_dynamic_cache_across_prefill_and_decode():
    from transformers import DynamicCache, Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
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
    model = Qwen3MoeForCausalLM(config).eval()
    reference_cache = DynamicCache(config=config)
    paged_cache = LuminalPagedCache(
        config,
        max_cache_len=16,
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
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)

    assert paged_cache.get_seq_length() == reference_cache.get_seq_length() == 5


def test_tiny_hf_qwen_export_exposes_paged_cache_boundary(tmp_path):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from luminal import translate_pt2_to_egglog
    from luminal.pt2 import _decomp_table

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
    model = Qwen3MoeForCausalLM(config).eval()
    cache = LuminalPagedCache(
        config,
        max_cache_len=16,
        dtype=torch.float32,
        device="cpu",
    )

    exported = torch.export.export(
        model,
        args=(),
        kwargs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "past_key_values": cache,
            "use_cache": True,
        },
        strict=False,
    )
    decomposed = exported.run_decompositions(_decomp_table())
    targets = [str(node.target) for node in decomposed.graph_module.graph.nodes]
    assert targets.count("aten.index_select.default") == 2
    # Depending on the PyTorch version/decomposition table, an explicit dtype
    # conversion is represented as either aten.to.dtype or aten._to_copy.
    assert sum(
        target in ("aten.to.dtype", "aten._to_copy.default") for target in targets
    ) >= 2
    assert (
        sum(
            output.kind.name == "USER_OUTPUT"
            for output in decomposed.graph_signature.output_specs
        )
        == 4
    )
    assert (
        sum(
            output.kind.name == "USER_INPUT_MUTATION"
            for output in decomposed.graph_signature.output_specs
        )
        == 2
    )

    decomposed._example_inputs = None
    pt2_path = tmp_path / "qwen_paged_cache.pt2"
    torch.export.save(decomposed, pt2_path)
    egglog, _ = translate_pt2_to_egglog(str(pt2_path))

    assert egglog.count("(Op (Scatter ") >= 2
    assert egglog.count("(Op (Gather ") >= 2

    # Verify the translated cache ABI, not merely the presence of unrelated
    # Scatter/Gather operations elsewhere in Qwen's MoE graph. The named K/V
    # pool inputs must feed separate row scatters, and those exact scatter
    # results must feed the paired active-row gathers used by attention.
    lines = egglog.splitlines()

    def input_term(label: str, dtype: str) -> str:
        pattern = re.compile(
            rf'^\(let (t\d+) \(Input \d+ "{re.escape(label)}" \({dtype}\)\)\)$'
        )
        return next(pattern.match(line).group(1) for line in lines if pattern.match(line))

    def producer_term(op: str, input_fragment: str) -> str:
        prefix = re.compile(rf"^\(let (t\d+) \(Op \({op} ")
        return next(
            prefix.match(line).group(1)
            for line in lines
            if prefix.match(line) and input_fragment in line
        )

    key_pool = input_term("past_key_values_key_cache_0", "F32")
    value_pool = input_term("past_key_values_value_cache_0", "F32")
    input_term("past_key_values_positions_0", "Int")
    key_scatter = producer_term("Scatter", f"(ICons {key_pool} (ICons ")
    value_scatter = producer_term("Scatter", f"(ICons {value_pool} (ICons ")
    producer_term("Gather", f"(ICons {key_scatter} (INil))")
    producer_term("Gather", f"(ICons {value_scatter} (INil))")


def test_luminal_paged_cache_rejects_unsupported_batch_and_sliding_window():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        LuminalPagedCache(
            _config(),
            max_cache_len=8,
            batch_size=2,
            dtype=torch.float32,
            device="cpu",
        )
    config = _config()
    config.sliding_window = 4
    with pytest.raises(NotImplementedError, match="sliding-window"):
        LuminalPagedCache(
            config,
            max_cache_len=8,
            dtype=torch.float32,
            device="cpu",
        )
