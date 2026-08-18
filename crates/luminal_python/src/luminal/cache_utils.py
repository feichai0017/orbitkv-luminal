"""Transformers cache integration for PyTorch capture."""

import torch


class _DynamicCacheAdapter:
    @staticmethod
    def as_dict(cache):
        return {
            "key_cache": [
                layer.keys for layer in cache.layers if layer.keys is not None
            ],
            "value_cache": [
                layer.values for layer in cache.layers if layer.values is not None
            ],
        }

    @classmethod
    def flatten(cls, cache):
        return torch.utils._pytree._dict_flatten(cls.as_dict(cache))

    @classmethod
    def flatten_with_keys(cls, cache):
        return torch.utils._pytree._dict_flatten_with_keys(cls.as_dict(cache))

    @staticmethod
    def unflatten(values, context):
        from transformers.cache_utils import DynamicCache

        tensors = torch.utils._pytree._dict_unflatten(values, context)
        cache = DynamicCache()
        keys = tensors.get("key_cache", [])
        values = tensors.get("value_cache", [])
        for index in range(max(len(keys), len(values))):
            key = keys[index] if index < len(keys) else None
            value = values[index] if index < len(values) else None
            cache.update(key, value, index)
        return cache

    @classmethod
    def flatten_spec(cls, cache, spec):
        return torch.fx._pytree._dict_flatten_spec(cls.as_dict(cache), spec)


class _StaticCacheAdapter:
    @staticmethod
    def as_dict(cache):
        from transformers.cache_utils import StaticLayer

        unsupported = [
            type(layer).__name__
            for layer in cache.layers
            if type(layer) is not StaticLayer
        ]
        if unsupported:
            raise TypeError(
                "Luminal direct inference currently supports full-attention "
                f"StaticCache layers, got {unsupported}"
            )
        return {
            "key_cache": [layer.keys for layer in cache.layers],
            "value_cache": [layer.values for layer in cache.layers],
            "cumulative_length": [layer.cumulative_length for layer in cache.layers],
        }

    @classmethod
    def flatten(cls, cache):
        values, context = torch.utils._pytree._dict_flatten(cls.as_dict(cache))
        lengths = tuple(layer.max_cache_len for layer in cache.layers)
        return values, (context, lengths)

    @classmethod
    def flatten_with_keys(cls, cache):
        values, context = torch.utils._pytree._dict_flatten_with_keys(
            cls.as_dict(cache)
        )
        lengths = tuple(layer.max_cache_len for layer in cache.layers)
        return values, (context, lengths)

    @staticmethod
    def unflatten(values, context):
        from transformers.cache_utils import Cache, StaticCache, StaticLayer

        dictionary_context, lengths = context
        tensors = torch.utils._pytree._dict_unflatten(values, dictionary_context)
        layers = []
        for keys, values, cumulative_length, max_cache_len in zip(
            tensors["key_cache"],
            tensors["value_cache"],
            tensors["cumulative_length"],
            lengths,
            strict=True,
        ):
            layer = StaticLayer(max_cache_len=max_cache_len)
            layer.keys = keys
            layer.values = values
            layer.cumulative_length = cumulative_length
            layer.dtype = keys.dtype
            layer.device = keys.device
            layer.max_batch_size, layer.num_heads = keys.shape[:2]
            layer.k_head_dim = keys.shape[-1]
            layer.v_head_dim = values.shape[-1]
            layer.is_initialized = True
            layers.append(layer)

        cache = StaticCache.__new__(StaticCache)
        Cache.__init__(cache, layers=layers)
        return cache

    @classmethod
    def flatten_spec(cls, cache, _spec):
        return cls.flatten(cache)[0]


def _register(cache_type, adapter):
    if cache_type in torch.utils._pytree.SUPPORTED_NODES:
        return
    torch.utils._pytree.register_pytree_node(
        cache_type,
        adapter.flatten,
        adapter.unflatten,
        serialized_type_name=f"{cache_type.__module__}.{cache_type.__name__}",
        flatten_with_keys_fn=adapter.flatten_with_keys,
    )
    torch.fx._pytree.register_pytree_flatten_spec(cache_type, adapter.flatten_spec)


def register_transformers_caches():
    """Register available Transformers cache containers, idempotently."""
    try:
        from transformers import cache_utils
    except ImportError:
        return

    for name, adapter in (
        ("DynamicCache", _DynamicCacheAdapter),
        ("StaticCache", _StaticCacheAdapter),
    ):
        cache_type = getattr(cache_utils, name, None)
        if cache_type is not None:
            _register(cache_type, adapter)
