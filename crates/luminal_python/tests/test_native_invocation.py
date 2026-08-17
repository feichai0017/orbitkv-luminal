"""Contracts for generic and explicitly bound native invocation."""

import pytest
import torch

from luminal import backend


class Affine(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0


class PythonScalarOutput(torch.nn.Module):
    def forward(self, x):
        return x.sum().item()


class UpdateState(torch.nn.Module):
    def forward(self, state, positions, values):
        state[positions] = values
        return state * 2.0


class TokenOutput(torch.nn.Module):
    def forward(self, logits):
        return logits.argmax()


def _capture_artifact(model, example):
    artifacts = []

    def capturing_backend(graph_module, example_inputs, options=None):
        artifact = backend(graph_module, example_inputs, options)
        artifacts.append(artifact)
        return artifact

    compiled = torch.compile(model, backend=capturing_backend, fullgraph=True)
    examples = example if isinstance(example, tuple) else (example,)
    compiled(*examples)
    assert len(artifacts) == 1
    return artifacts[0]


def test_generic_invoke_observes_replaced_storage(device):
    model = Affine().to(device)
    first = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    artifact = _capture_artifact(model, first)

    replacement = torch.arange(12, 24, dtype=torch.float32, device=device).reshape(3, 4)
    torch.testing.assert_close(artifact(replacement)[0], model(replacement))

    # Preserve Python object identity while replacing its underlying storage.
    rebound = replacement.clone()
    rebound.set_(torch.full_like(replacement, 7.0))
    torch.testing.assert_close(artifact(rebound)[0], model(rebound))


def test_generic_invoke_python_metadata_fallback(device, monkeypatch):
    monkeypatch.setenv("LUMINAL_DISABLE_DLPACK_C_EXCHANGE", "1")
    model = Affine().to(device)
    example = torch.randn(3, 4, device=device)
    artifact = _capture_artifact(model, example)

    replacement = torch.randn(3, 4, device=device)
    torch.testing.assert_close(artifact(replacement)[0], model(replacement))


def test_generic_invoke_uses_native_tensor_observation(device, monkeypatch):
    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        pytest.skip("PyTorch does not expose the DLPack C exchange API")

    model = Affine().to(device)
    example = torch.randn(3, 4, device=device)
    artifact = _capture_artifact(model, example)
    replacement = torch.randn(3, 4, device=device)
    expected = model(replacement)

    def python_metadata_was_used(*_args, **_kwargs):
        raise AssertionError("native observation fell back to Python tensor methods")

    with monkeypatch.context() as patch:
        for method in ("numel", "element_size", "data_ptr", "is_contiguous"):
            patch.setattr(torch.Tensor, method, python_metadata_was_used)
        actual = artifact(replacement)[0]

    torch.testing.assert_close(actual, expected)


def test_generic_invoke_materializes_noncontiguous_input(device):
    model = Affine().to(device)
    example = torch.randn(3, 4, device=device)
    artifact = _capture_artifact(model, example)
    noncontiguous = torch.randn(4, 3, device=device).transpose(0, 1)
    assert not noncontiguous.is_contiguous()
    torch.testing.assert_close(artifact(noncontiguous)[0], model(noncontiguous))


def test_generic_invoke_observes_contiguous_storage_offset(device):
    model = Affine().to(device)
    example = torch.randn(3, 4, device=device)
    artifact = _capture_artifact(model, example)
    base = torch.arange(20, dtype=torch.float32, device=device)
    offset_view = base[4:16].reshape(3, 4)
    assert offset_view.is_contiguous()
    assert offset_view.storage_offset() == 4

    torch.testing.assert_close(artifact(offset_view)[0], model(offset_view))


def test_generic_invoke_rejects_dtype_mismatch(device):
    model = Affine().to(device)
    example = torch.randn(3, 4, dtype=torch.float32, device=device)
    artifact = _capture_artifact(model, example)

    with pytest.raises(TypeError, match="expects torch.float32 but got torch.int32"):
        artifact(torch.ones(3, 4, dtype=torch.int32, device=device))


def test_generic_invoke_rejects_wrong_argument_count(device):
    model = Affine().to(device)
    example = torch.randn(3, 4, device=device)
    artifact = _capture_artifact(model, example)

    with pytest.raises(ValueError, match="Expected 1 inputs, got 0"):
        artifact()

    with pytest.raises(ValueError, match="Expected 1 inputs, got 2"):
        artifact(example, example)


def test_generic_invoke_validates_static_shape_plan(device):
    model = Affine().to(device)
    example = torch.randn(3, 4, device=device)
    artifact = _capture_artifact(model, example)

    with pytest.raises(ValueError, match="dimension 1 expected size 4, got 5"):
        artifact(torch.randn(3, 5, device=device))

    with pytest.raises(ValueError, match="expected rank 2, got 1"):
        artifact(torch.randn(12, device=device))


def test_bound_execution_is_explicit_and_exclusive(device):
    model = Affine().to(device)
    input_buffer = torch.randn(3, 4, device=device)
    artifact = _capture_artifact(model, input_buffer)

    if device.type != "cuda":
        with pytest.raises(
            NotImplementedError,
            match="bound execution currently requires a CUDA backend",
        ):
            artifact.bind(input_buffer)
        return

    bound = artifact.bind(input_buffer)
    first_output = bound.replay()[0]
    torch.testing.assert_close(first_output, model(input_buffer))

    input_buffer.copy_(torch.full_like(input_buffer, 3.0))
    second_output = bound.replay()[0]
    assert second_output is first_output
    torch.testing.assert_close(second_output, model(input_buffer))

    with pytest.raises(RuntimeError, match="consumed by bound execution"):
        artifact(input_buffer)


def test_bound_replay_keeps_python_scalar_output_on_device(device, monkeypatch):
    if device.type != "cuda":
        pytest.skip("bound execution requires CUDA")

    model = PythonScalarOutput().to(device)
    input_buffer = torch.randn(3, 4, device=device)
    expected = model(input_buffer)
    artifact = _capture_artifact(model, input_buffer)
    bound = artifact.bind(input_buffer)

    def item_must_not_run(*_args, **_kwargs):
        raise AssertionError("bound replay must not call Tensor.item()")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", item_must_not_run)
        output = bound.replay()[0]

    assert isinstance(output, torch.Tensor)
    assert output.is_cuda
    assert output.ndim == 0
    assert output.dtype == torch.float64
    torch.testing.assert_close(
        output,
        torch.scalar_tensor(expected, dtype=output.dtype, device=device),
    )


def test_bound_replay_commits_writebacks(device):
    if device.type != "cuda":
        pytest.skip("bound execution requires CUDA")

    model = UpdateState().to(device)
    state_buffer = torch.empty(4, device=device)
    positions = torch.tensor([1, 3], device=device)
    values = torch.randn(2, device=device)
    artifact = _capture_artifact(model, (state_buffer, positions, values))
    bound = artifact.bind(state_buffer, positions, values)

    state_buffer.zero_()
    values.copy_(torch.randn_like(values))
    expected_state = torch.zeros_like(state_buffer)
    expected_state[positions] = values
    output = bound.replay()[0]

    torch.testing.assert_close(state_buffer, expected_state)
    torch.testing.assert_close(output, expected_state * 2.0)


def test_bound_replay_keeps_integer_token_on_device(device):
    if device.type != "cuda":
        pytest.skip("bound execution requires CUDA")

    model = TokenOutput().to(device)
    logits = torch.randn(32, device=device)
    artifact = _capture_artifact(model, logits)
    bound = artifact.bind(logits)

    token = bound.replay()[0]

    assert token.device == logits.device
    assert token.dtype == torch.int64
    assert token.ndim == 0
    torch.testing.assert_close(token, model(logits))
