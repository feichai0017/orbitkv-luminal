"""Thin Python facade over Luminal's native execution artifacts."""


class CompiledModel:
    """Thin facade over a native ``CompiledGraph`` execution artifact."""

    def __init__(
        self,
        graph_result,
        weight_refs=None,
        input_names=None,
        user_indices=None,
        scalar_output_positions=(),
    ):
        """Initialize with a compiled CompiledGraph from Rust.

        Args:
            graph_result: The CompiledGraph from luminal_python.process_pt2()
            weight_refs: List of PyTorch tensors to keep alive (prevents GC of shared weights)
            input_names: Override for user input names. If None, uses graph_result.input_names.
            user_indices: When torch.compile lifts model parameters into extra args,
                this tells __call__ which arg positions are actual user inputs.
                None means all args are user inputs (PT2 path).
        """
        self._graph = graph_result
        self._input_names = input_names or graph_result.input_names
        self._output_names = graph_result.output_names
        # {output position: mutated input name} for the write-back outputs
        # torch.export's functionalization appends for in-place input
        # mutations. Keyed by position, not name: a model that mutates an
        # input and also returns it yields two same-named outputs.
        self._writeback_by_pos = dict(graph_result.writeback_outputs)
        self._has_dynamic_dims = getattr(graph_result, "has_dynamic_dims", False)
        self._weight_refs = weight_refs or []
        input_dtype_codes = graph_result.input_dtypes
        if len(input_dtype_codes) != len(self._input_names):
            raise RuntimeError(
                f"CompiledGraph returned {len(input_dtype_codes)} input dtype "
                f"codes for {len(self._input_names)} declared inputs "
                f"({self._input_names!r}) — every declared input needs a "
                f"matching dtype."
            )
        self._graph.configure_invocation(
            list(self._input_names),
            list(user_indices) if user_indices is not None else None,
            list(scalar_output_positions),
        )

    def set_dim(self, param_name: str, value: int) -> None:
        """Set a dynamic dimension value by its param name."""
        self._graph.set_dim(param_name, value)

    @property
    def writeback_inputs(self) -> dict:
        """{output name: input name it writes back to} for the in-place input
        mutations `__call__` applies to the caller's tensors."""
        return {
            self._output_names[pos]: input_name
            for pos, input_name in self._writeback_by_pos.items()
        }

    @property
    def has_dynamic_dims(self) -> bool:
        return self._has_dynamic_dims

    @property
    def dim_params(self) -> list[str]:
        return self._graph.dim_params

    def __call__(self, *inputs):
        """Execute with fully generic PyTorch tensor-binding semantics."""
        return self._graph.invoke(inputs)

    def bind(self, *inputs):
        """Bind stable CUDA resources and return an exclusive executable.

        The returned object owns strong references to its inputs and outputs;
        callers update buffer contents in place and call ``replay()`` repeatedly.
        Replay outputs remain device tensors, including rank-zero outputs.
        """
        return self._graph.bind(inputs)
