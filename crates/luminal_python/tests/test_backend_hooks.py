"""Unit tests: the backend hooks registry."""

from luminal import BackendHooks, register_backend_hooks
from luminal.backend_hooks import get_backend_hooks


def test_registry_and_defaults():
    hooks = BackendHooks()
    register_backend_hooks("test-backend", hooks)
    assert get_backend_hooks("test-backend") is hooks
    # Unknown names and callables resolve to a shared no-op (callable
    # backends pass hooks explicitly via luminal_generate(hooks=...)).
    for backend in ("nowhere", lambda gm, example_inputs, options=None: gm):
        noop = get_backend_hooks(backend)
        assert isinstance(noop, BackendHooks)
    # No-op methods are callable and return None.
    noop.before_compile(None)
    noop.on_step(3)
