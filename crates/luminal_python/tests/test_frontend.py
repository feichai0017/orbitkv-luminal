"""Contract tests for Luminal's public Python frontend."""

import importlib

import luminal


def test_public_frontend_exports_are_intentional():
    assert set(luminal.__all__) == {
        "backend",
        "compile",
        "luminal_backend",
        "make_backend",
        "register_backend",
    }


def test_legacy_backend_names_alias_clear_names():
    assert luminal.luminal_backend is luminal.backend
    assert luminal.register_backend is luminal.make_backend


def test_compile_routes_to_direct_pt2_path(monkeypatch):
    sentinel = object()
    received = {}

    def fake_compile(model, example_input, **kwargs):
        received.update(model=model, example_input=example_input, **kwargs)
        return sentinel

    monkeypatch.setattr("luminal.pt2.compile", fake_compile)
    model = object()
    example_input = object()

    result = luminal.compile(
        model,
        example_input,
        search_iterations=7,
        dynamic_dim="auto",
    )

    assert result is sentinel
    assert received == {
        "model": model,
        "example_input": example_input,
        "search_iterations": 7,
        "factory": None,
        "export_kwargs": None,
        "dynamic_dim": "auto",
        "dynamic_shapes": None,
    }


def test_make_backend_routes_options_to_pt2(monkeypatch):
    sentinel = object()
    received = {}

    def fake_compile_graph(
        graph_module, example_inputs, factory, *, search_iterations=None
    ):
        received.update(
            graph_module=graph_module,
            example_inputs=example_inputs,
            factory=factory,
            search_iterations=search_iterations,
        )
        return sentinel

    backend_module = importlib.import_module("luminal.backend")
    monkeypatch.setattr(backend_module, "_compile_graph", fake_compile_graph)
    factory = object()
    graph_module = object()
    example_inputs = [object()]

    result = luminal.make_backend(factory)(
        graph_module,
        example_inputs,
        options={"search_iterations": 11},
    )

    assert result is sentinel
    assert received == {
        "graph_module": graph_module,
        "example_inputs": example_inputs,
        "factory": factory,
        "search_iterations": 11,
    }


def test_backend_selects_factory_and_routes_options(monkeypatch):
    backend_module = importlib.import_module("luminal.backend")
    factory = object()
    sentinel = object()
    received = {}

    monkeypatch.setattr(backend_module, "detect_factory", lambda inputs: factory)

    def fake_compile_graph(
        graph_module, example_inputs, selected_factory, *, search_iterations=None
    ):
        received.update(
            graph_module=graph_module,
            example_inputs=example_inputs,
            factory=selected_factory,
            search_iterations=search_iterations,
        )
        return sentinel

    monkeypatch.setattr(backend_module, "_compile_graph", fake_compile_graph)
    graph_module = object()
    example_inputs = [object()]

    result = luminal.backend(
        graph_module,
        example_inputs,
        options={"search_iterations": 13},
    )

    assert result is sentinel
    assert received == {
        "graph_module": graph_module,
        "example_inputs": example_inputs,
        "factory": factory,
        "search_iterations": 13,
    }
