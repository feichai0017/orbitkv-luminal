"""Transformers cache adapters whose storage is native to Luminal backends.

``LuminalPagedCache`` keeps each layer's keys and values in token-major NHD
storage.  Hugging Face attention still receives its usual BHND logical view,
but PT2 sees the cache update as indexed writes into a fixed two-dimensional
pool followed by indexed reads of the active slots.  That is the cache ABI
used by luminal_cuda_lite's page-size-one FlashInfer implementation.

The first implementation intentionally supports append-only, batch-one,
full-attention generation.  Unsupported cache semantics fail explicitly
instead of silently producing a graph with the wrong positional behavior.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


def _cache_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Accept both Cache.update(..., cache_kwargs) calling conventions."""
    cache_kwargs = kwargs.get("cache_kwargs")
    if cache_kwargs is None and args and isinstance(args[0], dict):
        cache_kwargs = args[0]
    return cache_kwargs or {}


class LuminalPagedLayer(CacheLayerMixin):
    """One append-only KV layer backed by page-size-one NHD pools."""

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        max_cache_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        if batch_size != 1:
            raise NotImplementedError(
                "LuminalPagedCache currently supports batch_size=1"
            )
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be positive")

        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size
        self.num_heads = num_heads
        self.k_head_dim = head_dim
        self.v_head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.keys = torch.zeros(
            (max_cache_len, num_heads * head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.values = torch.zeros_like(self.keys)
        # FlashInfer's compact page-table ABI uses int32 slot ids.  PyTorch's
        # indexed tensor operators require int64, so update() casts this state
        # only at those operator boundaries.  Keeping the authoritative state
        # int32 prevents the backend from reinterpreting an int64 buffer.
        self.positions = torch.empty(0, dtype=torch.int32, device=self.device)
        self.is_initialized = True

    @classmethod
    def from_tensors(
        cls,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> "LuminalPagedLayer":
        """Rebuild a layer during Dynamo/Export pytree unflattening."""
        if keys.ndim != 2 or values.ndim != 2 or keys.shape != values.shape:
            raise ValueError("paged K/V pools must be equal two-dimensional tensors")
        layer = cls.__new__(cls)
        CacheLayerMixin.__init__(layer)
        layer.max_cache_len = keys.shape[0]
        layer.max_batch_size = 1
        # Only their product is recoverable from a flattened pool.  These two
        # values are replaced by key_states' authoritative shape in update().
        layer.num_heads = 0
        layer.k_head_dim = 0
        layer.v_head_dim = 0
        layer.dtype = keys.dtype
        layer.device = keys.device
        layer.keys = keys
        layer.values = values
        layer.positions = positions
        layer.is_initialized = True
        return layer

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        # Pools are deliberately initialized before tracing so they cross the
        # compiled boundary as cache inputs.
        if not self.is_initialized:
            raise RuntimeError("LuminalPagedLayer must be initialized eagerly")

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
        if key_states.shape[-1] != value_states.shape[-1]:
            raise ValueError("different K/V head dimensions are not yet supported")

        num_heads = key_states.shape[1]
        head_dim = key_states.shape[-1]
        kv_dim = num_heads * head_dim
        if self.keys.shape[1] != kv_dim or self.values.shape[1] != kv_dim:
            raise ValueError(
                "cache pool width does not match the model's KV head geometry"
            )

        cache_kwargs = _cache_kwargs(args, kwargs)
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            start = self.positions.shape[0]
            cache_position = torch.arange(
                start,
                start + key_states.shape[-2],
                dtype=torch.int32,
                device=key_states.device,
            )
        else:
            cache_position = cache_position.to(
                device=key_states.device, dtype=torch.int32
            ).reshape(-1)

        if cache_position.shape[0] != key_states.shape[-2]:
            raise ValueError("cache_position length must equal the number of new tokens")

        # HF supplies BHND.  The pools are NHD flattened to [slot, H*D].
        new_keys = key_states.permute(0, 2, 1, 3).reshape(-1, kv_dim)
        new_values = value_states.permute(0, 2, 1, 3).reshape(-1, kv_dim)
        write_positions = cache_position.to(dtype=torch.int64)
        self.keys = torch.index_put(self.keys, (write_positions,), new_keys)
        self.values = torch.index_put(self.values, (write_positions,), new_values)
        self.positions = torch.cat((self.positions, cache_position))

        # Read only active pages.  Returning a strided BHND view preserves the
        # public Transformers cache contract without changing physical NHD
        # storage.
        read_positions = self.positions.to(dtype=torch.int64)
        active_keys = torch.index_select(self.keys, 0, read_positions)
        active_values = torch.index_select(self.values, 0, read_positions)
        context = self.positions.shape[0]
        keys_hf = active_keys.reshape(context, num_heads, head_dim).permute(1, 0, 2)
        values_hf = active_values.reshape(context, num_heads, head_dim).permute(
            1, 0, 2
        )
        self.num_heads = num_heads
        self.k_head_dim = head_dim
        self.v_head_dim = head_dim
        return keys_hf.unsqueeze(0), values_hf.unsqueeze(0)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.positions.shape[0] + query_length, 0

    def get_seq_length(self) -> int:
        return self.positions.shape[0]

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if beam_idx.numel() != 1 or int(beam_idx[0]) != 0:
            raise NotImplementedError(
                "LuminalPagedCache does not yet support beam-cache reordering"
            )

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.positions = self.positions[:0]


class LuminalPagedCache(Cache):
    """A Transformers Cache with fixed, token-major page-size-one storage."""

    def __init__(
        self,
        config: Any,
        max_cache_len: int,
        *,
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
            layer_type not in ("full_attention", "moe") for layer_type in layer_types
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
        positions: list[torch.Tensor],
    ) -> "LuminalPagedCache":
        if not (len(keys) == len(values) == len(positions)):
            raise ValueError("paged cache tensor lists must have equal lengths")
        cache = cls.__new__(cls)
        layers = [
            LuminalPagedLayer.from_tensors(key, value, position)
            for key, value, position in zip(keys, values, positions)
        ]
        Cache.__init__(cache, layers=layers)
        return cache
