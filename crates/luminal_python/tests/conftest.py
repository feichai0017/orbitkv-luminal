"""Test configuration."""

import logging
from pathlib import Path

# Enable automatic Rust rebuilds during test development
try:
    import maturin_import_hook

    maturin_import_hook.install()
    logging.getLogger("maturin_import_hook").disabled = True
    logging.getLogger("maturin_import_hook.project_importer").disabled = True
except ImportError:
    pass  # Hook not available, rebuilds will be manual

import os

import pytest
import torch
import torch._dynamo
from _llama38b_artifacts import ensure_onnx_bundle, ensure_pt2_bundle


@pytest.fixture
def device() -> torch.device:
    backend = os.getenv("LUMINAL_BACKEND", "native").lower()
    return torch.device("cuda") if backend == "cuda" else torch.device("cpu")


@pytest.fixture(scope="session")
def _llama38b_cache_dir(pytestconfig: pytest.Config) -> Path:
    return pytestconfig.cache.mkdir("luminal_llama38b_artifacts_v1")


@pytest.fixture(scope="session")
def _llama38b_onnx_bundle(
    pytestconfig: pytest.Config, _llama38b_cache_dir: Path
):
    return ensure_onnx_bundle(pytestconfig.cache, _llama38b_cache_dir)


@pytest.fixture(scope="session")
def _llama38b_pt2_bundle(
    pytestconfig: pytest.Config, _llama38b_cache_dir: Path
):
    return ensure_pt2_bundle(pytestconfig.cache, _llama38b_cache_dir)


@pytest.fixture(scope="session")
def llama38b_ref_logits(request: pytest.FixtureRequest) -> torch.Tensor:
    fixturenames = set(request.fixturenames)
    uses_onnx = "llama38b_onnx_path" in fixturenames
    uses_pt2 = bool({"llama38b_pt2_path", "llama38b_weights_path"} & fixturenames)

    if uses_onnx and uses_pt2:
        raise pytest.UsageError(
            "llama38b_ref_logits cannot be requested with both ONNX and PT2 "
            "artifact fixtures in the same test"
        )
    if uses_onnx:
        bundle = request.getfixturevalue("_llama38b_onnx_bundle")
    elif uses_pt2:
        bundle = request.getfixturevalue("_llama38b_pt2_bundle")
    else:
        raise pytest.UsageError(
            "llama38b_ref_logits must be requested alongside llama38b_onnx_path "
            "or llama38b_pt2_path/llama38b_weights_path"
        )

    return torch.load(bundle.ref_logits_path, weights_only=True)


@pytest.fixture(scope="session")
def llama38b_onnx_path(_llama38b_onnx_bundle) -> Path:
    assert _llama38b_onnx_bundle.onnx_path is not None
    return _llama38b_onnx_bundle.onnx_path


@pytest.fixture(scope="session")
def llama38b_pt2_path(_llama38b_pt2_bundle) -> Path:
    assert _llama38b_pt2_bundle.pt2_path is not None
    return _llama38b_pt2_bundle.pt2_path


@pytest.fixture(scope="session")
def llama38b_weights_path(_llama38b_pt2_bundle) -> Path:
    assert _llama38b_pt2_bundle.weights_path is not None
    return _llama38b_pt2_bundle.weights_path


@pytest.fixture(autouse=True, scope="function")
def reset_torch_dynamo():
    # We need this for two reasons
    # 1. Some of our casts tests use the same model, but those graph have some state to them
    # and the cache will return old models
    # 2. The cache adds a large preformace hit to the test suite
    torch._dynamo.config.cache_size_limit = 1
    # Disable silent fallback to eager mode so backend errors surface as test failures
    torch._dynamo.config.suppress_errors = False
    """Reset PyTorch Dynamo state after each test to prevent state leakage.

    This fixture automatically runs after every test function to clear
    torch._dynamo's compilation cache, ensuring test isolation.
    """
    yield  # Test runs here
    torch._dynamo.reset()
