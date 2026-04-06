"""Test configuration."""
# ruff: noqa: E402

import logging
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import warnings

try:
    import huggingface_hub
    from transformers import logging as transformers_logging
except ImportError:  # pragma: no cover - optional for non-HF test environments
    huggingface_hub = None
    transformers_logging = None

# Enable automatic Rust rebuilds during test development
import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings
from maturin_import_hook.project_importer import DefaultProjectFileSearcher

backend = os.getenv("LUMINAL_BACKEND", "native").lower()
settings = MaturinSettings(
    release=(backend == "cuda"),
    features=["cuda"] if backend == "cuda" else None,
    skip_install=True,
)
searcher = DefaultProjectFileSearcher(
    source_excluded_dir_names=(
        DefaultProjectFileSearcher.DEFAULT_SOURCE_EXCLUDED_DIR_NAMES
        | {".claude", "docs", ".github", "examples"}
    ),
)
maturin_import_hook.install(
    settings=settings,
    enable_automatic_installation=True,
    file_searcher=searcher,
)
logging.getLogger("maturin_import_hook").disabled = True
logging.getLogger("maturin_import_hook.project_importer").disabled = True

# Silence noisy ONNX / onnxscript / httpx logging
for _logger_name in (
    "onnxscript",
    "onnx_ir",
    "torch.onnx",
    "httpx",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# Suppress torch.onnx diagnostics/progress output and torchvision warnings
os.environ.setdefault("TORCH_ONNX_VERBOSE", "0")
os.environ.setdefault("TORCH_ONNX_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", module="torch.onnx")

import pytest
import torch
import torch._dynamo
from _llama38b_artifacts import ensure_onnx_bundle, ensure_pt2_bundle

torch.set_float32_matmul_precision("highest")


@pytest.fixture
def device() -> torch.device:
    backend = os.getenv("LUMINAL_BACKEND", "native").lower()
    return torch.device("cuda") if backend == "cuda" else torch.device("cpu")


@pytest.fixture(scope="session", autouse=True)
def configure_hf_test_output() -> None:
    if transformers_logging is not None:
        transformers_logging.disable_progress_bar()
    if huggingface_hub is not None:
        huggingface_hub.utils.disable_progress_bars()


@pytest.fixture
def configure_dynamo():
    original_cache_size_limit = torch._dynamo.config.cache_size_limit
    original_suppress_errors = torch._dynamo.config.suppress_errors

    def _configure(
        *, cache_size_limit: int | None = None, suppress_errors: bool | None = None
    ) -> None:
        if cache_size_limit is not None:
            torch._dynamo.config.cache_size_limit = cache_size_limit
        if suppress_errors is not None:
            torch._dynamo.config.suppress_errors = suppress_errors

    yield _configure

    torch._dynamo.config.cache_size_limit = original_cache_size_limit
    torch._dynamo.config.suppress_errors = original_suppress_errors


@pytest.fixture(scope="session")
def _llama38b_cache_dir(pytestconfig: pytest.Config) -> Path:
    return pytestconfig.cache.mkdir("luminal_llama38b_artifacts_v1")


@pytest.fixture(scope="session")
def _hf_multimodal_cache_dir(pytestconfig: pytest.Config) -> Path:
    return pytestconfig.cache.mkdir("luminal_hf_multimodal_v1")


@pytest.fixture(scope="session")
def hf_multimodal_image_path(
    pytestconfig: pytest.Config, _hf_multimodal_cache_dir: Path
) -> Path:
    image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/bee.jpg"
    )
    image_path = _hf_multimodal_cache_dir / "bee.jpg"
    metadata_key = "luminal_python/hf_multimodal_image_v1"
    metadata = {
        "schema_version": 1,
        "url": image_url,
        "filename": image_path.name,
    }

    needs_download = pytestconfig.cache.get(metadata_key, None) != metadata or not (
        image_path.is_file()
    )
    if not needs_download:
        return image_path

    image_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with urlopen(image_url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=image_path.parent, delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                while chunk := response.read(1024 * 1024):
                    tmp_file.write(chunk)

        assert tmp_path is not None
        tmp_path.replace(image_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    pytestconfig.cache.set(metadata_key, metadata)
    return image_path


@pytest.fixture(scope="session")
def _llama38b_onnx_bundle(pytestconfig: pytest.Config, _llama38b_cache_dir: Path):
    return ensure_onnx_bundle(pytestconfig.cache, _llama38b_cache_dir)


@pytest.fixture(scope="session")
def _llama38b_pt2_bundle(pytestconfig: pytest.Config, _llama38b_cache_dir: Path):
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
