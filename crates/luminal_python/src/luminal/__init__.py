"""Luminal's public Python frontend."""

from .api import compile
from .backend import backend, luminal_backend, make_backend, register_backend
from .compiled_model import CompiledModel as CompiledModel
from .luminal import CompiledGraph as CompiledGraph
from .luminal import process_pt2 as process_pt2
from .pytree_compat import register_optional_pytrees

register_optional_pytrees()

__all__ = [
    "backend",
    "compile",
    "make_backend",
    # Compatibility aliases. Prefer the names above in new code.
    "luminal_backend",
    "register_backend",
]
