"""Contracts for Luminal's two public frontend entry points."""

import importlib

import luminal
from luminal.pt2 import compile as compile_pt2


def test_backend_entry_points_route_factory_and_options(monkeypatch):
    backend_module = importlib.import_module("luminal.backend")
    calls = []
    sentinel = object()

    def fake_compile(graph, inputs, factory, *, search_iterations=None):
        calls.append((graph, inputs, factory, search_iterations))
        return sentinel

    monkeypatch.setattr(backend_module, "_compile_graph", fake_compile)
    selected_factory, explicit_factory = object(), object()
    monkeypatch.setattr(
        backend_module, "detect_factory", lambda inputs: selected_factory
    )
    graph, inputs = object(), [object()]

    assert luminal.backend(graph, inputs, {"search_iterations": 11}) is sentinel
    assert luminal.make_backend(explicit_factory)(
        graph, inputs, {"search_iterations": 13}
    ) is sentinel
    assert calls == [
        (graph, inputs, selected_factory, 11),
        (graph, inputs, explicit_factory, 13),
    ]
    assert luminal.luminal_backend is luminal.backend
    assert luminal.register_backend is luminal.make_backend
    assert luminal.compile is compile_pt2
