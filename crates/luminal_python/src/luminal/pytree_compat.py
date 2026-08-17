"""Optional third-party PyTree compatibility for PyTorch capture."""


def _cache_dict(cache):
    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [
            layer.values for layer in cache.layers if layer.values is not None
        ],
    }


def _flatten_dynamic_cache(cache):
    import torch

    return torch.utils._pytree._dict_flatten(_cache_dict(cache))


def _flatten_with_keys_dynamic_cache(cache):
    import torch

    return torch.utils._pytree._dict_flatten_with_keys(_cache_dict(cache))


def _unflatten_dynamic_cache(values, context):
    import torch
    from transformers.cache_utils import DynamicCache

    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    key_list = dictionary.get("key_cache", [])
    value_list = dictionary.get("value_cache", [])
    for index in range(max(len(key_list), len(value_list))):
        key = key_list[index] if index < len(key_list) else None
        value = value_list[index] if index < len(value_list) else None
        cache.update(key, value, index)
    return cache


def register_optional_pytrees():
    """Register supported optional-library containers, if installed.

    Registration is idempotent and importing Luminal does not require the
    optional Transformers package.
    """
    try:
        import torch
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
        return

    torch.utils._pytree.register_pytree_node(
        DynamicCache,
        _flatten_dynamic_cache,
        _unflatten_dynamic_cache,
        serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
    )
    torch.fx._pytree.register_pytree_flatten_spec(
        DynamicCache,
        lambda cache, spec: torch.fx._pytree._dict_flatten_spec(
            _cache_dict(cache), spec
        ),
    )
