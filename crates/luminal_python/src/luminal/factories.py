"""Backend factory selection and native tensor binding helpers."""

import torch

from .dtype_util import torch_dtype_code as _torch_dtype_code


def detect_factory(example_inputs):
    """Select the built-in backend factory for the first tensor input."""
    first_tensor = next(
        (value for value in example_inputs or () if torch.is_tensor(value)), None
    )
    device = first_tensor.device if first_tensor is not None else torch.device("cpu")
    if device.type == "cuda":
        try:
            from .luminal import _cuda_lite_factory_capsule

            return _cuda_lite_factory_capsule()
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "CUDA input was provided, but luminal_python was not built with "
                "the cuda feature. Rebuild with `maturin develop --features cuda` "
                "or run through `run_tests_cuda.sh`/the Modal CUDA test runner."
            ) from exc

    from .luminal import _reference_factory_capsule

    return _reference_factory_capsule()


def collect_weight_pointers(weights):
    """Partition weight tensors into CUDA and CPU pointer bindings."""
    keep_alive = []
    device_ptrs = {}
    cpu_ptrs = {}
    for name, tensor in weights.items():
        contiguous = tensor.detach().contiguous()
        n_bytes = contiguous.numel() * contiguous.element_size()
        keep_alive.append(contiguous)
        if contiguous.is_cuda:
            device_ptrs[name] = (contiguous.data_ptr(), n_bytes)
        else:
            cpu_ptrs[name] = (
                contiguous.data_ptr(),
                n_bytes,
                _torch_dtype_code(contiguous.dtype),
            )
    return keep_alive, device_ptrs, cpu_ptrs


def load_cpu_weights(compiled_graph, cpu_weights):
    """Load CPU weight data into a compiled graph."""
    for name, (pointer, n_bytes, dtype_code) in cpu_weights.items():
        compiled_graph.set_weight_from_ptr(name, pointer, n_bytes, dtype_code)
