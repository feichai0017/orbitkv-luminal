"""Transformers cache adapters backed by Luminal-native persistent state.

The first cache intentionally targets the narrow execution contract needed to
make single-stream decode structurally static:

* batch size one;
* append-only, full attention;
* fixed-size NHD pages allocated before tracing;
* a fixed-capacity int32 block table;
* a device-resident int32 sequence length; and
* in-place mutation of every state tensor.

The public Hugging Face view remains BHND.  The physical representation exposed
to PT2 is paged NHD, which is the representation a paged-attention backend can
consume without replacing cache tensors as the context grows.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


def _cache_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Accept both Transformers ``Cache.update(..., cache_kwargs)`` spellings."""

    cache_kwargs = kwargs.get("cache_kwargs")
    if cache_kwargs is None and args and isinstance(args[0], dict):
        cache_kwargs = args[0]
    return cache_kwargs or {}


class LuminalPagedLayer(CacheLayerMixin):
    """One append-only KV layer stored in fixed-size physical NHD pages."""

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        max_cache_len: int,
        page_size: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        if max_cache_len % page_size != 0:
            raise ValueError(
                "max_cache_len must be a multiple of page_size so the fixed "
                "physical capacity survives PT2 pytree roundtrips exactly"
            )
        if batch_size != 1:
            raise NotImplementedError(
                "LuminalPagedCache currently supports batch_size=1"
            )
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be positive")

        self.max_cache_len = int(max_cache_len)
        self.page_size = int(page_size)
        self.max_batch_size = int(batch_size)
        self.num_heads = int(num_heads)
        self.k_head_dim = int(head_dim)
        self.v_head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)

        self.num_pages = self.max_cache_len // self.page_size
        # Physical ABI: [page, token-within-page, kv-head, head-dim].
        self.keys = torch.zeros(
            (self.num_pages, self.page_size, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros_like(self.keys)
        # The narrow first implementation reserves every page up front.  The
        # table is nevertheless explicit so the compiler and backend see the
        # final paged ABI rather than relying on contiguous page assignment.
        self.block_table = torch.arange(
            self.num_pages, dtype=torch.int32, device=self.device
        )
        self.sequence_length = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self.is_initialized = True

        # These tensors are state, not ordinary outputs.  Marking their
        # addresses static helps Dynamo retain the same input identity outside
        # tracing; PT2 still records their in-place mutations explicitly.
        if not torch.compiler.is_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.block_table)
            torch._dynamo.mark_static_address(self.sequence_length)

    @classmethod
    def from_tensors(
        cls,
        keys: torch.Tensor,
        values: torch.Tensor,
        block_table: torch.Tensor,
        sequence_length: torch.Tensor,
    ) -> "LuminalPagedLayer":
        """Rebuild a layer during Dynamo/Export pytree unflattening."""

        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError(
                "paged K/V pools must be equal [pages, page_size, heads, dim] tensors"
            )
        if block_table.ndim != 1 or block_table.dtype != torch.int32:
            raise ValueError("block_table must be a one-dimensional int32 tensor")
        if sequence_length.shape != (1,) or sequence_length.dtype != torch.int32:
            raise ValueError("sequence_length must be an int32 tensor shaped [1]")

        layer = cls.__new__(cls)
        CacheLayerMixin.__init__(layer)
        layer.num_pages = int(keys.shape[0])
        layer.page_size = int(keys.shape[1])
        layer.max_cache_len = layer.num_pages * layer.page_size
        layer.max_batch_size = 1
        layer.num_heads = int(keys.shape[2])
        layer.k_head_dim = int(keys.shape[3])
        layer.v_head_dim = int(values.shape[3])
        layer.dtype = keys.dtype
        layer.device = keys.device
        layer.keys = keys
        layer.values = values
        layer.block_table = block_table
        layer.sequence_length = sequence_length
        layer.is_initialized = True
        return layer

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        if not self.is_initialized:
            raise RuntimeError("LuminalPagedLayer must be initialized before tracing")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("expected K/V tensors shaped [batch, heads, tokens, dim]")
        if key_states.shape[0] != 1 or value_states.shape[0] != 1:
            raise NotImplementedError(
                "LuminalPagedCache currently supports batch_size=1"
            )
        if key_states.shape[:3] != value_states.shape[:3]:
            raise ValueError("K/V batch, head, and token dimensions must match")
        if key_states.shape[1] != self.num_heads:
            raise ValueError("K/V head count does not match the cache")
        if key_states.shape[-1] != self.k_head_dim:
            raise ValueError("K head dimension does not match the cache")
        if value_states.shape[-1] != self.v_head_dim:
            raise ValueError("V head dimension does not match the cache")

        cache_kwargs = _cache_kwargs(args, kwargs)
        cache_position = cache_kwargs.get("cache_position")
        token_count = key_states.shape[-2]
        if cache_position is None:
            cache_position = torch.arange(
                token_count, dtype=torch.int32, device=key_states.device
            ) + self.sequence_length[0]
        else:
            cache_position = cache_position.to(
                device=key_states.device, dtype=torch.int32
            ).reshape(-1)

        if cache_position.shape[0] != token_count:
            raise ValueError("cache_position length must equal the number of new tokens")
        if not torch.compiler.is_compiling():
            first = int(cache_position.min().item())
            last = int(cache_position.max().item())
            if first < 0 or last >= self.max_cache_len:
                raise IndexError(
                    "LuminalPagedCache capacity exceeded: positions "
                    f"[{first}, {last}] are outside [0, {self.max_cache_len})"
                )

        # Map logical positions through the stable block table.  PyTorch index
        # operators require int64 indices, but int32 remains the authoritative
        # metadata representation consumed by the backend.
        logical_page = torch.div(
            cache_position, self.page_size, rounding_mode="floor"
        )
        page_offset = torch.remainder(cache_position, self.page_size)
        physical_page = torch.index_select(
            self.block_table, 0, logical_page.to(torch.int64)
        )
        physical_slot = physical_page * self.page_size + page_offset

        new_keys = key_states.permute(0, 2, 1, 3).reshape(
            token_count, self.num_heads, self.k_head_dim
        )
        new_values = value_states.permute(0, 2, 1, 3).reshape(
            token_count, self.num_heads, self.v_head_dim
        )
        self.keys.view(-1, self.num_heads, self.k_head_dim).index_copy_(
            0, physical_slot.to(torch.int64), new_keys
        )
        self.values.view(-1, self.num_heads, self.v_head_dim).index_copy_(
            0, physical_slot.to(torch.int64), new_values
        )

        # This remains a tensor mutation.  No Python integer or changing-shape
        # positions tensor participates in the compiled decode boundary.
        next_length = torch.maximum(
            self.sequence_length[0], cache_position[-1] + 1
        ).reshape_as(self.sequence_length)
        self.sequence_length.copy_(next_length)

        # Present the full fixed logical capacity to Hugging Face attention.
        # Its causal/static-cache mask excludes unwritten slots.  The block
        # table makes this correct even when physical pages are not contiguous.
        logical_keys = torch.index_select(
            self.keys, 0, self.block_table.to(torch.int64)
        ).reshape(-1, self.num_heads, self.k_head_dim)[: self.max_cache_len]
        logical_values = torch.index_select(
            self.values, 0, self.block_table.to(torch.int64)
        ).reshape(-1, self.num_heads, self.v_head_dim)[: self.max_cache_len]
        return (
            logical_keys.permute(1, 0, 2).unsqueeze(0),
            logical_values.permute(1, 0, 2).unsqueeze(0),
        )

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.max_cache_len, 0

    def get_seq_length(self) -> torch.Tensor | int:
        return self.sequence_length[0] if self.is_initialized else 0

    def get_max_length(self) -> int:
        return self.max_cache_len

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if beam_idx.numel() != 1 or int(beam_idx[0]) != 0:
            raise NotImplementedError(
                "LuminalPagedCache does not yet support beam-cache reordering"
            )

    def reset(self) -> None:
        # Every subsequently active slot is overwritten before it is read, so
        # reset only the logical extent and preserve every large allocation.
        self.sequence_length.zero_()


class LuminalPagedCache(Cache):
    """Fixed-capacity paged Transformers cache for static Luminal decode."""

    def __init__(
        self,
        config: Any,
        max_cache_len: int,
        *,
        page_size: int = 16,
        batch_size: int = 1,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        text_config = (
            config.get_text_config(decoder=True)
            if hasattr(config, "get_text_config")
            else config
        )
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is not None and any(
            layer_type not in ("full_attention", "moe")
            for layer_type in layer_types
        ):
            raise NotImplementedError(
                "LuminalPagedCache currently supports full-attention layers only"
            )
        if getattr(text_config, "sliding_window", None) is not None:
            raise NotImplementedError(
                "LuminalPagedCache does not yet support sliding-window attention"
            )

        num_layers = int(text_config.num_hidden_layers)
        num_heads = int(text_config.num_key_value_heads)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            head_dim = text_config.hidden_size // text_config.num_attention_heads
        layers = [
            LuminalPagedLayer(
                max_cache_len,
                page_size,
                batch_size,
                num_heads,
                int(head_dim),
                dtype,
                device,
            )
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)

    @classmethod
    def from_tensors(
        cls,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        block_tables: list[torch.Tensor],
        sequence_lengths: list[torch.Tensor],
    ) -> "LuminalPagedCache":
        count = len(keys)
        if not (
            len(values)
            == len(block_tables)
            == len(sequence_lengths)
            == count
        ):
            raise ValueError("paged cache tensor lists must have equal lengths")
        cache = cls.__new__(cls)
        layers = [
            LuminalPagedLayer.from_tensors(
                key,
                value,
                block_table,
                sequence_length,
            )
            for key, value, block_table, sequence_length in zip(
                keys,
                values,
                block_tables,
                sequence_lengths,
            )
        ]
        Cache.__init__(cache, layers=layers)
        return cache
