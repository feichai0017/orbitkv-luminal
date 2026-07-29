"""Luminal Python bindings - PyTorch backend using Luminal."""

# Import Python components
# Register DynamicCache pytree serialization once at import time
from .backend_hooks import BackendHooks, register_backend_hooks
from .cache_utils import _register_cache_serialization
from .compiled_model import CompiledModel
from .generation import luminal_generate

# Import Rust extension components (built by maturin)
from .luminal import CompiledGraph, process_pt2
from .main import luminal_backend, register_backend

_register_cache_serialization()

# Re-export everything for clean package interface
__all__ = [
    "BackendHooks",
    "CompiledModel",
    "luminal_backend",
    "luminal_generate",
    "register_backend",
    "register_backend_hooks",
    "CompiledGraph",
    "process_pt2",
]
