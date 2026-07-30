"""Compile-once greedy generation over a retained luminal artifact.

`luminal_generate` is a transformers `custom_generate` callable: pass it to
`model.generate(..., custom_generate=luminal_generate)` and Hugging Face runs
its usual preparation (resolved GenerationConfig, device placement, special
tokens), then hands the decoding loop to us. It can equally be called directly
with the same arguments.

The first call per model compiles once: a `StaticCache` is allocated up front
so its tensors trace as graph inputs, the prefill forward is captured with a
dynamic sequence dimension, and the compiled artifact plus its argument layout
are retained on the model. Every later step — decode, and the prefill of any
subsequent call — invokes the artifact directly, so Dynamo never re-enters and
recompiles are structurally impossible. In-place cache updates flow through
CompiledModel's write-back handling, so the `StaticCache` behaves exactly as
in eager mode.

Scope: batch-1, greedy decoding. `attention_mask` is ignored (batch-1 has no
padding; the causal mask is built from `cache_position`). Note that
transformers' callable-`custom_generate` dispatch does not forward `streamer=`
from `model.generate` — pass a streamer by calling `luminal_generate`
directly.
"""

import time

import torch

from .backend_hooks import GenerationCompileContext, get_backend_hooks


class _GenerationState:
    """Retained compile artifact + bindings, stored on the model object."""

    __slots__ = ("compiled", "example_inputs", "step_slots", "cache", "max_cache_len")

    def __init__(self, compiled, example_inputs, step_slots, cache, max_cache_len):
        self.compiled = compiled
        self.example_inputs = example_inputs
        self.step_slots = step_slots
        self.cache = cache
        self.max_cache_len = max_cache_len


def _row(token_ids, like):
    """[[tokens]] with the dtype/device of `like`."""
    return torch.tensor([token_ids], dtype=like.dtype, device=like.device)


def _resolve_backend(backend):
    if callable(backend):
        return backend
    from torch._dynamo.backends.registry import lookup_backend

    try:
        return lookup_backend(backend)
    except Exception:
        if backend == "luminal":  # dev tree without installed entry-point metadata
            from .main import luminal_backend

            return luminal_backend
        raise


def _match_step_slots(markers, example_inputs):
    """Map marker names to Dynamo argv positions by tensor identity — Dynamo
    lifts the very objects we passed. A marker with no slot either wasn't
    traced into the graph (fine: cache_position on recent transformers) or
    trips the required-slots check in _compile_and_prefill."""
    step_slots = {}
    for index, value in enumerate(example_inputs):
        for name, marker in markers.items():
            if value is marker:
                step_slots[name] = index
    return step_slots


def _materialize_skeleton(model):
    """Make a disk-mapped model traceable. Stock ``from_pretrained(...,
    device_map={"": "disk"})`` downloads and resolves without materializing —
    parameters land on the meta device. Allocate storage (``to_empty``:
    virtual pages; nothing writes the parameters, whose bytes the backend
    streams from the checkpoint) and restore config-computed buffers (rotary
    inv_freq) by re-running the model's own ``_init_weights`` on
    buffer-owning modules only — the same repair HF applies to its own
    meta-built loads. No-op for normally loaded models."""
    if not any(p.is_meta for p in model.parameters()):
        return
    model.to_empty(device="cpu")
    with torch.no_grad():
        for module in model.modules():
            if next(module.buffers(recurse=False), None) is not None:
                model._init_weights(module)


def _reset_cache_position(cache, position):
    """Point the cache's write cursor at `position` (transformers versions
    that track it; older ones take the position from `cache_position`)."""
    for layer in cache.layers:
        counter = getattr(layer, "cumulative_length", None)
        if torch.is_tensor(counter):
            counter.fill_(position)


def luminal_generate(
    model,
    input_ids,
    logits_processor=None,
    stopping_criteria=None,
    generation_config=None,
    streamer=None,
    backend="luminal",
    max_cache_length=None,
    cache_start_pos=0,
    hooks=None,
    **model_kwargs,
):
    """Greedy batch-1 generation; compiles on the first call per model.

    Args (beyond the transformers decoding-loop contract):
        backend: torch.compile backend name or callable (default "luminal").
        max_cache_length: KV capacity; default prompt + max_new_tokens.
        cache_start_pos: prefill the prompt at this cache position instead of
            0, continuing the retained cache (conversation/prefill reuse).
            Only valid after the first (compiling) call.
        hooks: BackendHooks override; default resolved from `backend`.
    """
    hooks = hooks or get_backend_hooks(backend)
    if generation_config is None:
        generation_config = getattr(model, "generation_config", None)

    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("[luminal] luminal_generate: batch-1 2-D input_ids required")
    if (
        getattr(generation_config, "do_sample", False)
        or (getattr(generation_config, "num_beams", 1) or 1) != 1
    ):
        raise RuntimeError(
            "[luminal] luminal_generate: greedy decoding only "
            "(do_sample=False, num_beams=1)"
        )
    prompt_len = int(input_ids.shape[1])
    start_pos = int(cache_start_pos or 0)
    max_new_tokens = model_kwargs.pop("max_new_tokens", None) or getattr(
        generation_config, "max_new_tokens", None
    )
    if max_new_tokens is None and getattr(generation_config, "max_length", None):
        max_new_tokens = int(generation_config.max_length) - (start_pos + prompt_len)
    if not max_new_tokens or int(max_new_tokens) < 1:
        raise RuntimeError("[luminal] luminal_generate: max_new_tokens required")
    max_new_tokens = int(max_new_tokens)
    eos_value = getattr(generation_config, "eos_token_id", None)
    if torch.is_tensor(eos_value):
        eos_value = eos_value.reshape(-1).tolist()
    if eos_value is None:
        eos_ids = set()
    elif isinstance(eos_value, (list, tuple, set)):
        eos_ids = {int(v) for v in eos_value}
    else:
        eos_ids = {int(eos_value)}

    state = getattr(model, "_luminal_generation_state", None)
    if state is None:
        if start_pos:
            raise RuntimeError(
                "[luminal] luminal_generate: cache_start_pos requires a prior "
                "call (the first call compiles and prefills at position 0)"
            )
        state, first_logits, ttft_ms = _compile_and_prefill(
            model,
            input_ids,
            prompt_len,
            max_new_tokens,
            max_cache_length,
            backend,
            hooks,
            model_kwargs,
        )
    else:
        if (
            prompt_len < 2
            or start_pos + prompt_len + max_new_tokens > state.max_cache_len
        ):
            raise RuntimeError(
                f"[luminal] luminal_generate: start_pos={start_pos} + "
                f"prompt_len={prompt_len} + max_new_tokens={max_new_tokens} "
                f"outside the compiled cache length {state.max_cache_len} "
                "(2 <= prompt, start+prompt+new <= cache)"
            )
        _reset_cache_position(state.cache, start_pos)
        positions = torch.arange(
            start_pos, start_pos + prompt_len, device=input_ids.device
        )
        started = time.perf_counter()
        first_logits = _run_step(
            state, hooks, input_ids, positions, positions.unsqueeze(0), start_pos
        )
        ttft_ms = (time.perf_counter() - started) * 1e3

    # ---- greedy decode over the retained artifact ----
    def pick_token(logits, sequences):
        row = logits.reshape(-1, logits.shape[-1])[-1:]
        if logits_processor is not None and len(logits_processor):
            row = logits_processor(sequences, row)
        return int(torch.argmax(row))

    generated = [pick_token(first_logits, input_ids)]
    if streamer is not None:
        streamer.put(torch.tensor([[generated[-1]]]))
    decode_started = time.perf_counter()
    pos = start_pos + prompt_len
    while len(generated) < max_new_tokens and generated[-1] not in eos_ids:
        step_ids = torch.tensor(
            [[generated[-1]]], dtype=input_ids.dtype, device=input_ids.device
        )
        step_pos = torch.tensor([pos], dtype=torch.long, device=input_ids.device)
        logits = _run_step(state, hooks, step_ids, step_pos, step_pos.unsqueeze(0), pos)
        sequences = torch.cat([input_ids, _row(generated, input_ids)], dim=-1)
        generated.append(pick_token(logits, sequences))
        pos += 1
        if streamer is not None:
            streamer.put(torch.tensor([[generated[-1]]]))
    if streamer is not None and hasattr(streamer, "end"):
        streamer.end()
    tpot_ms = (time.perf_counter() - decode_started) * 1e3 / max(len(generated) - 1, 1)

    sequences = torch.cat([input_ids, _row(generated, input_ids)], dim=-1)
    if not getattr(generation_config, "return_dict_in_generate", False):
        return sequences
    from transformers.generation import GenerateDecoderOnlyOutput

    out = GenerateDecoderOnlyOutput(sequences=sequences, past_key_values=state.cache)
    out.luminal_generated_ids = generated
    out.luminal_ttft_ms = ttft_ms
    out.luminal_tpot_ms = tpot_ms
    return out


def _compile_and_prefill(
    model,
    input_ids,
    prompt_len,
    max_new_tokens,
    max_cache_length,
    backend,
    hooks,
    model_kwargs,
):
    """First call: build the StaticCache, compile once on the symbolic
    prefill, retain the artifact + argument bindings on the model, and return
    the prefill logits (the traced call IS the prefill)."""
    from transformers.cache_utils import StaticCache

    _materialize_skeleton(model)
    max_cache_len = max(int(max_cache_length or 0), prompt_len + max_new_tokens)
    # Our own StaticCache replaces whatever HF prepared: it must be allocated
    # BEFORE tracing (so the tensors become graph inputs, not graph-internal
    # allocations) and sized to our geometry.
    model_kwargs.pop("past_key_values", None)
    config = model.config.get_text_config(decoder=True)
    cache = StaticCache(config=config, max_cache_len=max_cache_len)
    cache.early_initialization(
        batch_size=1,
        num_heads=int(config.num_key_value_heads),
        head_dim=int(
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        ),
        dtype=model.dtype,
        device=input_ids.device,
    )
    hooks.before_compile(
        GenerationCompileContext(
            model, max_cache_len, prompt_len, max_new_tokens, cache
        )
    )

    cache_position = torch.arange(prompt_len, device=input_ids.device)
    position_ids = cache_position.unsqueeze(0)
    # Declare the SAME dynamic range on every compile: the declaration is a
    # guard contract, and 2 is the floor because Dynamo specializes size-0/1
    # dims (decode's s=1 runs through the retained artifact, never Dynamo).
    # input_ids must carry the same mark — its static seq dim would otherwise
    # specialize the shared symbol when broadcast against the rotary
    # embeddings derived from position_ids.
    torch._dynamo.mark_dynamic(input_ids, 1, min=2, max=max_cache_len)
    torch._dynamo.mark_dynamic(cache_position, 0, min=2, max=max_cache_len)
    torch._dynamo.mark_dynamic(position_ids, 1, min=2, max=max_cache_len)

    markers = {
        "input_ids": input_ids,
        "cache_position": cache_position,
        "position_ids": position_ids,
    }
    for i, layer in enumerate(cache.layers):
        markers[f"k_cache_{i}"] = layer.keys
        markers[f"v_cache_{i}"] = layer.values

    record = {}
    inner_backend = _resolve_backend(backend)

    def recording_backend(gm, example_inputs, options=None):
        compiled = inner_backend(gm, example_inputs)
        record["compiled"] = compiled
        record["example_inputs"] = list(example_inputs)
        record["step_slots"] = _match_step_slots(markers, example_inputs)
        hooks.after_compile(compiled, example_inputs)
        return compiled

    compiled_forward = torch.compile(model, backend=recording_backend, fullgraph=True)
    hooks.on_step(0)
    started = time.perf_counter()
    with torch.inference_mode():
        out = compiled_forward(
            input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            position_ids=position_ids,
            logits_to_keep=1,
            use_cache=True,
            return_dict=True,
        )
    ttft_ms = (time.perf_counter() - started) * 1e3

    if "compiled" not in record:
        raise RuntimeError(
            "[luminal] luminal_generate: nothing retained after compile — the "
            "backend was not invoked"
        )
    # cache_position is intentionally absent from the required set: recent
    # transformers derive cache write positions from the cache's own
    # cumulative_length counter and never read the kwarg, so Dynamo drops it
    # from the graph inputs. When a slot exists (older versions), _run_step
    # feeds it.
    missing = {"input_ids", "position_ids"} - set(record["step_slots"])
    if missing:
        raise RuntimeError(
            f"[luminal] luminal_generate: could not locate {sorted(missing)} in "
            "the compiled argument list"
        )
    state = _GenerationState(
        record["compiled"],
        record["example_inputs"],
        record["step_slots"],
        cache,
        max_cache_len,
    )
    model._luminal_generation_state = state
    hooks.after_warmup(model, state.compiled)
    return state, out.logits, ttft_ms


def _run_step(state, hooks, input_ids, cache_position, position_ids, position):
    """One direct invocation of the retained artifact (no Dynamo). Works for
    both prefill (seq >= 2, dynamic dim) and decode (seq == 1): the compiled
    graph derives the sequence length from the input shapes."""
    argv = list(state.example_inputs)
    step = {
        "input_ids": input_ids,
        "cache_position": cache_position,
        "position_ids": position_ids,
    }
    for name, idx in state.step_slots.items():
        if name in step:
            argv[idx] = step[name]
    hooks.on_step(int(position))
    with torch.inference_mode():
        outs = state.compiled(*argv)
    outs = [
        t
        for t in (outs if isinstance(outs, (list, tuple)) else [outs])
        if torch.is_tensor(t)
    ]
    if len(outs) != 1:
        raise RuntimeError(
            f"[luminal] luminal_generate: expected exactly the logits output, "
            f"got {len(outs)} tensors"
        )
    return outs[0]
