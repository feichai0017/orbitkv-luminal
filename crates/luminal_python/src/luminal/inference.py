"""Direct, stable-resource inference frontend.

Unlike :mod:`luminal.backend`, this API does not preserve arbitrary PyTorch
tensor-replacement semantics. The tensor leaves in ``example_input`` and
``export_kwargs`` are bound for the lifetime of the returned executable;
callers update their contents in place and replay without passing arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .pt2 import compile_artifact


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")


class CompiledInferenceModel:
    """A directly compiled model bound to stable CUDA storage."""

    def __init__(self, artifact, bound):
        # The artifact retains model-weight tensors whose pointers are borrowed
        # by the native runtime. BoundExecutable separately retains user inputs.
        self._artifact = artifact
        self._bound = bound

    def replay(self):
        """Execute once using the buffers supplied to :func:`compile`."""
        return self._bound.replay()

    def __call__(self):
        """Alias for :meth:`replay`; replay intentionally accepts no inputs."""
        return self.replay()

    @property
    def has_dynamic_dims(self):
        return self._artifact.has_dynamic_dims

    @property
    def dim_params(self):
        return self._artifact.dim_params


class CompiledCausalLMStep:
    """A bound logits graph followed by ordinary PyTorch sampling."""

    def __init__(self, model, token_buffer):
        self._model = model
        self._token_buffer = token_buffer
        self._default_sp = SamplingParams()

    def __call__(
        self,
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    ):
        logits = self._model()[0]
        params = self._default_sp if sampling_params is None else sampling_params
        token = sample_logits(logits, params)
        self._token_buffer.copy_(token.reshape_as(self._token_buffer))
        return self._token_buffer


def _example_args(example_input):
    if isinstance(example_input, (list, tuple)):
        return tuple(example_input)
    return (example_input,)


def _tensor_leaves(args, kwargs):
    leaves, _ = torch.utils._pytree.tree_flatten((args, kwargs))
    return tuple(leaf for leaf in leaves if isinstance(leaf, torch.Tensor))


def compile(
    model,
    example_input,
    search_iterations=25,
    factory=None,
    export_kwargs=None,
    dynamic_dim=None,
    dynamic_shapes=None,
):
    """Compile and bind a model for fixed-resource inference.

    Tensor leaves from the example arguments are retained and their addresses
    become part of the executable's lifetime contract. Change values in those
    tensors in place, then call the returned object with no arguments.

    ``torch.export`` is used once during compilation. Runtime replay does not
    enter Dynamo, flatten pytrees, inspect tensors, or update bindings.
    """
    args = _example_args(example_input)
    kwargs = dict(export_kwargs or {})
    artifact = compile_artifact(
        model,
        args,
        search_iterations=search_iterations,
        factory=factory,
        export_kwargs=kwargs,
        dynamic_dim=dynamic_dim,
        dynamic_shapes=dynamic_shapes,
    )
    bound = artifact.bind(*_tensor_leaves(args, kwargs))
    return CompiledInferenceModel(artifact, bound)


def _sample_logits(logits, params):
    if params.temperature == 0:
        return logits.argmax(dim=-1)

    scores = logits.float() / params.temperature
    if params.top_k:
        k = min(params.top_k, scores.shape[-1])
        threshold = scores.topk(k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)

    if params.top_p < 1:
        scores, indices = scores.sort(dim=-1, descending=True)
        probabilities = scores.softmax(dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities > params.top_p
        probabilities = scores.masked_fill(remove, -torch.inf).softmax(dim=-1)
        sampled_rank = torch.multinomial(probabilities, 1)
        return indices.gather(-1, sampled_rank).squeeze(-1)

    return torch.multinomial(scores.softmax(dim=-1), 1).squeeze(-1)


def sample_logits(
    logits,
    sampling_params: SamplingParams | Sequence[SamplingParams],
):
    """Sample token IDs from GPU logits using ordinary PyTorch operations."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [vocab] or [batch, vocab]")

    if isinstance(sampling_params, SamplingParams):
        return _sample_logits(logits, sampling_params)

    params = tuple(sampling_params)
    if len(params) != logits.shape[0]:
        raise ValueError(
            f"expected {logits.shape[0]} SamplingParams, got {len(params)}"
        )
    return torch.stack(
        [
            _sample_logits(row.unsqueeze(0), param)[0]
            for row, param in zip(logits, params)
        ]
    )


class _CausalLMLogits(torch.nn.Module):
    """Return only the final-token logits needed by the sampling step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache):
        logits = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits
        return logits[:, -1, :]


def compile_causal_lm_forward(
    model,
    input_ids,
    cache,
    token_buffer,
    *,
    search_iterations=25,
    factory=None,
):
    """Compile one bound causal-LM step with eager on-device sampling.

    Cache mutations remain exported durable writebacks, but the cache object is
    not redundantly returned as hundreds of ordinary output tensors and full
    prompt logits never cross the compiled boundary. Sampling uses regular
    PyTorch CUDA operations on the retained final-token logits output.
    """
    compiled = compile(
        _CausalLMLogits(model).eval(),
        (input_ids, cache),
        search_iterations=search_iterations,
        factory=factory,
    )
    return CompiledCausalLMStep(compiled, token_buffer)
