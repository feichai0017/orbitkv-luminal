"""Backend-specific configuration hooks for the compile-once generation runner.

The Rust backend-factory ABI (``BackendCompileArgs``) is frozen and carries no
backend-specific options. Backends that need out-of-band configuration — cache
geometry before compile, a per-step KV write position, weight sourcing —
register a :class:`BackendHooks` object under their torch.compile backend
name. Pure Python; no Rust changes required to add a new backend knob.

A plugin's dynamo entry point needs no helper from this module: the existing
`register_backend(capsule)` already returns a torch.compile-compatible
callable, and `luminal_backend` is one.
"""

from dataclasses import dataclass


@dataclass
class GenerationCompileContext:
    """Everything a backend may want to know before the generation compile."""

    model: object  # the HF model being compiled
    max_cache_len: int
    prompt_len: int
    max_new_tokens: int
    cache: object  # the StaticCache whose tensors become graph inputs


class BackendHooks:
    """No-op defaults; backends subclass and override what they need."""

    def before_compile(self, ctx: GenerationCompileContext) -> None:
        """Called once, before torch.compile traces the model."""

    def after_compile(self, compiled, example_inputs) -> None:
        """Called from inside the dynamo backend, right after the luminal
        artifact is built. Extension point for weight sourcing etc."""

    def after_warmup(self, model, compiled) -> None:
        """Called once, after the traced prefill call has executed."""

    def on_step(self, position: int) -> None:
        """Called before every direct execute with the KV write position."""


_HOOKS_REGISTRY: dict = {}
_NO_OP_HOOKS = BackendHooks()


def register_backend_hooks(name: str, hooks: BackendHooks) -> None:
    """Register hooks for the torch.compile backend registered under `name`."""
    _HOOKS_REGISTRY[str(name)] = hooks


def get_backend_hooks(backend) -> BackendHooks:
    """Resolve hooks for a torch.compile backend name; no-op default.

    Callable backends have no name to key on — pass hooks explicitly to
    `luminal_generate(hooks=...)` instead.
    """
    if isinstance(backend, str):
        return _HOOKS_REGISTRY.get(backend, _NO_OP_HOOKS)
    return _NO_OP_HOOKS
