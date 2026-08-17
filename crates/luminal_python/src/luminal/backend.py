"""Public ``torch.compile`` backend entry points."""

from .factories import detect_factory


def make_backend(factory):
    """Create a ``torch.compile`` backend using a specific native factory."""

    def configured_backend(graph_module, example_inputs, options=None):
        return _compile_graph(
            graph_module,
            example_inputs,
            factory,
            search_iterations=(options or {}).get("search_iterations"),
        )

    return configured_backend


def backend(graph_module, example_inputs, options=None):
    """Compile a Dynamo graph with the backend selected from its inputs.

    This is Luminal's generic PyTorch entry point. Runtime tensor storage is
    treated as caller-owned and may change between invocations.
    """
    factory = detect_factory(example_inputs)
    return _compile_graph(
        graph_module,
        example_inputs,
        factory,
        search_iterations=(options or {}).get("search_iterations"),
    )


def _compile_graph(graph_module, example_inputs, factory, *, search_iterations=None):
    # Keep the capture and translation pipeline lazy at package import time.
    from .pt2 import pt2_backend

    return pt2_backend(
        graph_module,
        example_inputs,
        factory=factory,
        search_iterations=search_iterations,
    )


# Compatibility names used by existing callers. New code should use
# ``backend`` and ``make_backend`` because no global registration occurs here.
luminal_backend = backend
register_backend = make_backend
