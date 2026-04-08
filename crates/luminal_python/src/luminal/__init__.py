"""Luminal Python bindings - PyTorch backend using Luminal."""

import logging
import os
import tempfile

from torch._logging import _internal as _torch_log_internal
from torch._logging._internal import register_log, register_artifact

# Register with PyTorch's logging system BEFORE importing the Rust extension.
# This ensures Python loggers are configured before pyo3-log queries their levels.
register_log("luminal", ["luminal"])
register_artifact(
    "luminal_hello_world",
    "Writes a hello-world file to demonstrate the artifact system",
    visible=True,
    off_by_default=True,
)

# Import Python components
from .compiled_model import CompiledModel

# Import Rust extension components (built by maturin)
# These are available directly in the package namespace
from .luminal import CompiledGraph, process_onnx, process_pt2
from .main import luminal_backend

# Register DynamicCache pytree serialization once at import time
from .cache_utils import _register_cache_serialization

_register_cache_serialization()


def _build_artifact_config():
    """Query torch._logging for enabled luminal artifacts and build config dict.

    Called at compilation time (not import time), so it respects runtime
    changes made via torch._logging.set_logs().

    Returns:
        dict[str, dict[str, str]]: Artifact name -> key-value params.
        Empty dict if no artifacts are enabled.
    """
    config = {}

    # Check luminal_hello_world artifact via torch._logging's authoritative state.
    # We use log_state.is_artifact_enabled() rather than checking logger.isEnabledFor()
    # because the latter walks the parent logger chain and gives false positives
    # for off_by_default artifacts when the parent "luminal" logger is set to DEBUG.
    if _torch_log_internal.log_state.is_artifact_enabled("luminal_hello_world"):
        config["luminal_hello_world"] = {
            "enabled": "true",
            "output_path": os.environ.get(
                "LUMINAL_HELLO_WORLD_PATH",
                os.path.join(tempfile.gettempdir(), "luminal_hello.txt"),
            ),
        }

    return config


# Re-export everything for clean package interface
__all__ = [
    "CompiledModel",
    "luminal_backend",
    "process_onnx",
    "CompiledGraph",
    "process_pt2",
    "_build_artifact_config",
]
