"""Direct ``luminal.compile`` fixed-resource inference contract."""

import pytest
import torch

import luminal
import luminal.inference


class Affine(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0


class TokenOutput(torch.nn.Module):
    def forward(self, logits):
        return logits.argmax()


def test_compile_flattens_structured_examples_once(monkeypatch):
    tensor_a = torch.randn(2)
    tensor_b = torch.randn(3)
    captured = {}

    class Bound:
        def replay(self):
            return (tensor_a,)

    class Artifact:
        has_dynamic_dims = False
        dim_params = []

        def bind(self, *inputs):
            captured["inputs"] = inputs
            return Bound()

    def fake_compile_artifact(model, example_input, **kwargs):
        captured["model"] = model
        captured["example_input"] = example_input
        captured["kwargs"] = kwargs
        return Artifact()

    monkeypatch.setattr(luminal.inference, "compile_artifact", fake_compile_artifact)
    model = Affine()
    compiled = luminal.compile(
        model,
        tensor_a,
        export_kwargs={"state": {"cache": [tensor_b]}, "use_cache": True},
    )

    assert captured["inputs"] == (tensor_a, tensor_b)
    assert compiled() == (tensor_a,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
    ],
)
def test_sampling_params_validate_request_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        luminal.SamplingParams(**kwargs)


def test_sample_logits_supports_per_request_batch_params():
    logits = torch.tensor([[1.0, 4.0, 2.0], [5.0, 1.0, 3.0]])
    params = [
        luminal.SamplingParams(),
        luminal.SamplingParams(temperature=1.0, top_k=1),
    ]

    torch.testing.assert_close(
        luminal.sample_logits(logits, params), torch.tensor([1, 0])
    )


def test_sample_logits_validates_batch_size():
    with pytest.raises(ValueError, match="expected 2 SamplingParams"):
        luminal.sample_logits(torch.randn(2, 8), [luminal.SamplingParams()])


def test_sample_logits_applies_top_p():
    logits = torch.tensor([[8.0, 1.0, 0.0]])
    params = luminal.SamplingParams(temperature=1.0, top_p=0.1)

    torch.testing.assert_close(luminal.sample_logits(logits, params), torch.tensor([0]))


def test_causal_lm_prefill_writes_stable_token_buffer(monkeypatch):
    input_ids = torch.tensor([[1, 2]])
    cache = object()
    token_buffer = torch.empty((1, 1), dtype=torch.int64)
    captured = {}

    class Output:
        logits = torch.randn(1, 2, 8)

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            captured["forward_kwargs"] = kwargs
            return Output()

    def fake_compile(step, example_input, **kwargs):
        captured["example_input"] = example_input
        captured["compile_kwargs"] = kwargs
        return lambda: (step(*example_input),)

    monkeypatch.setattr(luminal.inference, "compile", fake_compile)
    compiled = luminal.compile_causal_lm_forward(
        Model(), input_ids, cache, token_buffer, search_iterations=7
    )
    result = compiled(luminal.SamplingParams(temperature=1.0, top_k=1))

    assert result is token_buffer
    torch.testing.assert_close(
        token_buffer.reshape(-1), Output.logits[:, -1, :].argmax(dim=-1)
    )
    assert captured["example_input"] == (input_ids, cache)
    assert captured["compile_kwargs"]["search_iterations"] == 7
    assert captured["forward_kwargs"] == {
        "input_ids": input_ids,
        "past_key_values": cache,
        "use_cache": True,
        "logits_to_keep": 1,
    }


def test_causal_lm_decode_self_feeds_token_buffer(monkeypatch):
    token_buffer = torch.tensor([[2]])
    cache = object()

    class Output:
        logits = torch.randn(1, 1, 8)

    class Model(torch.nn.Module):
        def forward(self, **_kwargs):
            return Output()

    def fake_compile(step, example_input, **_kwargs):
        assert example_input == (token_buffer, cache)
        return lambda: (step(*example_input),)

    monkeypatch.setattr(luminal.inference, "compile", fake_compile)
    compiled = luminal.compile_causal_lm_forward(
        Model(), token_buffer, cache, token_buffer
    )
    result = compiled()

    assert result is token_buffer
    torch.testing.assert_close(
        token_buffer.reshape(-1), Output.logits[:, -1, :].argmax(dim=-1)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="bound execution requires CUDA"
)
def test_compile_binds_examples_and_replays_without_inputs(monkeypatch):
    device = torch.device("cuda")
    model = Affine().to(device)
    input_buffer = torch.randn(3, 4, device=device)

    def torch_compile_must_not_run(*_args, **_kwargs):
        raise AssertionError("luminal.compile must not call torch.compile")

    monkeypatch.setattr(torch, "compile", torch_compile_must_not_run)
    compiled = luminal.compile(model, input_buffer, search_iterations=1)

    first_outputs = compiled()
    first_output = first_outputs[0]
    torch.testing.assert_close(first_output, model(input_buffer))

    input_buffer.copy_(torch.full_like(input_buffer, 3.0))
    second_outputs = compiled.replay()
    second_output = second_outputs[0]
    assert second_outputs is first_outputs
    assert second_output is first_output
    torch.testing.assert_close(second_output, model(input_buffer))

    with pytest.raises(TypeError):
        compiled(input_buffer)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="bound execution requires CUDA"
)
def test_compile_keeps_sampled_token_on_device(monkeypatch):
    device = torch.device("cuda")
    logits = torch.randn(128, device=device)
    expected = logits.argmax()
    compiled = luminal.compile(TokenOutput().to(device), logits, search_iterations=1)

    def item_must_not_run(*_args, **_kwargs):
        raise AssertionError("direct inference replay must not call Tensor.item()")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", item_must_not_run)
        token = compiled()[0]

    assert token.device == logits.device
    assert token.dtype == torch.int64
    assert token.ndim == 0
    torch.testing.assert_close(token, expected)

