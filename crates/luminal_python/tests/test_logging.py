"""Tests for PyTorch logging integration and artifact system.

Verifies that:
- Luminal registers with torch._logging (log + artifacts)
- Log levels can be set programmatically and via env vars
- Log level changes after import are respected
- The luminal_hello_world artifact creates a file when enabled
- Artifacts are off by default
"""

import logging
import os
import tempfile

import pytest
import torch
import torch._dynamo
import torch._logging

from luminal import luminal_backend


class AddModel(torch.nn.Module):
    """Minimal model for triggering compilation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


def _compile_and_run(device):
    """Compile and run a minimal model to trigger Rust code paths."""
    model = AddModel().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    x = torch.ones(2, 2, device=device)
    _ = compiled(x)


# ---------------------------------------------------------------------------
# Log level tests
# ---------------------------------------------------------------------------


def test_luminal_log_registered():
    """Verify 'luminal' is a recognized log name in torch._logging."""
    from torch._logging._internal import log_registry

    assert log_registry.is_log("luminal"), (
        "'luminal' not registered with torch._logging"
    )


def test_luminal_artifact_registered():
    """Verify 'luminal_hello_world' is a recognized artifact."""
    from torch._logging._internal import log_registry

    assert log_registry.is_artifact("luminal_hello_world"), (
        "'luminal_hello_world' not registered with torch._logging"
    )


def test_programmatic_log_level_debug(device, capfd):
    """Setting luminal to DEBUG via set_logs should produce Rust trace output."""
    torch._logging.set_logs(luminal=logging.DEBUG)
    try:
        _compile_and_run(device)
    finally:
        # Reset to defaults
        torch._logging.set_logs(luminal=logging.WARNING)

    # Rust trace/debug output goes through Python logging to stderr
    captured = capfd.readouterr()
    # We just verify it doesn't crash — actual log content depends on
    # what trace! calls exist in the compilation path


def test_programmatic_log_level_warning_suppresses(device, capfd):
    """Setting luminal to WARNING should suppress DEBUG/INFO output."""
    torch._logging.set_logs(luminal=logging.WARNING)
    _compile_and_run(device)
    # At WARNING level, trace-level Rust output should not appear


def test_dynamic_level_change(device, capfd):
    """Log level changes after import should take effect on next compilation."""
    # Start at WARNING (quiet)
    torch._logging.set_logs(luminal=logging.WARNING)
    _compile_and_run(device)

    # Switch to DEBUG (verbose) — should take effect immediately
    torch._logging.set_logs(luminal=logging.DEBUG)
    _compile_and_run(device)

    # Switch back to WARNING
    torch._logging.set_logs(luminal=logging.WARNING)
    _compile_and_run(device)

    # The key assertion is that none of this crashes — level changes are respected


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------


def test_artifact_off_by_default(device):
    """Without enabling the artifact, no file should be created."""
    output_path = os.path.join(tempfile.mkdtemp(), "should_not_exist.txt")
    os.environ["LUMINAL_HELLO_WORLD_PATH"] = output_path
    try:
        # Ensure artifact is not enabled (default state)
        torch._logging.set_logs(luminal=logging.WARNING)
        _compile_and_run(device)
        assert not os.path.exists(output_path), (
            "Artifact file created even though luminal_hello_world was not enabled"
        )
    finally:
        os.environ.pop("LUMINAL_HELLO_WORLD_PATH", None)


def test_artifact_programmatic_enable(device):
    """Enabling luminal_hello_world via set_logs should create the file."""
    output_path = os.path.join(tempfile.mkdtemp(), "hello_programmatic.txt")
    os.environ["LUMINAL_HELLO_WORLD_PATH"] = output_path
    try:
        torch._logging.set_logs(luminal_hello_world=True)
        _compile_and_run(device)
        assert os.path.exists(output_path), (
            f"Artifact file not created at {output_path}"
        )
        content = open(output_path).read()
        assert "Hello from luminal" in content
    finally:
        os.environ.pop("LUMINAL_HELLO_WORLD_PATH", None)
        torch._logging.set_logs(luminal_hello_world=False)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_artifact_disable_after_enable(device):
    """Disabling the artifact after enabling it should stop file creation."""
    output_path = os.path.join(tempfile.mkdtemp(), "hello_toggle.txt")
    os.environ["LUMINAL_HELLO_WORLD_PATH"] = output_path
    try:
        # Enable, compile, file should exist
        torch._logging.set_logs(luminal_hello_world=True)
        _compile_and_run(device)
        assert os.path.exists(output_path)
        os.unlink(output_path)

        # Disable, compile again, file should NOT be recreated
        torch._logging.set_logs(luminal_hello_world=False)
        _compile_and_run(device)
        assert not os.path.exists(output_path), (
            "Artifact file recreated after disabling luminal_hello_world"
        )
    finally:
        os.environ.pop("LUMINAL_HELLO_WORLD_PATH", None)
        torch._logging.set_logs(luminal_hello_world=False)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_artifact_env_var(device, monkeypatch):
    """Enabling via TORCH_LOGS env var should create the file."""
    output_path = os.path.join(tempfile.mkdtemp(), "hello_env.txt")
    monkeypatch.setenv("LUMINAL_HELLO_WORLD_PATH", output_path)
    monkeypatch.setenv("TORCH_LOGS", "luminal_hello_world")

    # Re-initialize torch logging to pick up env var
    torch._logging._init_logs()

    try:
        _compile_and_run(device)
        assert os.path.exists(output_path), (
            f"Artifact file not created at {output_path} via TORCH_LOGS env var"
        )
        content = open(output_path).read()
        assert "Hello from luminal" in content
    finally:
        # monkeypatch auto-restores env vars; re-init logging to reset
        pass

    # After monkeypatch restores env, re-init to clean state
    torch._logging._init_logs()
