"""High-level Luminal APIs."""


def compile(
    model,
    example_input,
    search_iterations=25,
    factory=None,
    export_kwargs=None,
    dynamic_dim=None,
    dynamic_shapes=None,
):
    """Compile a model through Luminal's inference-oriented direct path.

    This frontend is intentionally separate from :func:`luminal.backend`, the
    fully generic ``torch.compile`` integration. For now it delegates to the
    existing PT2 direct compiler without changing compilation or invocation
    semantics; later serving preparation can evolve behind this API.
    """
    from .pt2 import compile as compile_pt2

    return compile_pt2(
        model,
        example_input,
        search_iterations=search_iterations,
        factory=factory,
        export_kwargs=export_kwargs,
        dynamic_dim=dynamic_dim,
        dynamic_shapes=dynamic_shapes,
    )
