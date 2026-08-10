"""CompiledModel wrapper for the Rust CompiledGraph."""

import os
import time
from typing import List

import torch

from .dtype_util import code_to_torch_dtype
from .dtype_util import torch_dtype_code as _torch_dtype_code


class DTypeBoundaryError(TypeError):
    """Raised when the caller passes an input whose dtype does not match the
    compiled graph's declared input dtype.

    The previous behaviour cast silently at every call, which (a) hid real
    precision bugs (e.g. f64 → f32 truncation on values outside the f32
    range) and (b) burnt CPU/GPU on a per-call allocation+copy that the
    user couldn't see in their profile. The contract is now strict:
    `model(x)` requires `x.dtype == model.input_dtypes[i]` for every
    positional input. Convert at the call site with
    `x.to(model.input_dtypes[i])` if you need a different dtype.
    """


class CompiledModel:
    """Wrapper around CompiledGraph that handles PyTorch tensor conversion."""

    def __init__(
        self, graph_result, weight_refs=None, input_names=None, user_indices=None
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
        self._output_shapes = graph_result.output_shapes
        self._has_dynamic_dims = getattr(graph_result, "has_dynamic_dims", False)
        self._weight_refs = weight_refs or []
        self._user_indices = user_indices
        self.skip_input_names = frozenset()
        self._is_gpu = getattr(graph_result, "device_type", "cpu") != "cpu"
        self._supports_device_ptrs = getattr(
            graph_result, "supports_device_ptrs", False
        )
        self._profile_call_index = 0
        # Expected input dtypes from graph. Every declared input MUST
        # have a dtype code — refuse to silently default to float32 if
        # the Rust side returned a shorter list than `input_names`.
        input_dtype_codes = graph_result.input_dtypes
        if len(input_dtype_codes) != len(self._input_names):
            raise RuntimeError(
                f"CompiledGraph returned {len(input_dtype_codes)} input dtype "
                f"codes for {len(self._input_names)} declared inputs "
                f"({self._input_names!r}) — every declared input needs a "
                f"matching dtype."
            )
        self._input_dtypes = [code_to_torch_dtype(c) for c in input_dtype_codes]

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
    def dim_params(self) -> List[str]:
        return self._graph.dim_params

    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Execute the compiled model with PyTorch tensor inputs.

        Args:
            *inputs: PyTorch tensors. When torch.compile lifts model parameters,
                this includes both weights and user inputs. user_indices filters
                to just the user inputs.

        Returns:
            Tuple of PyTorch tensors containing the model outputs
        """
        profile_boundary = os.getenv("LUMINAL_PROFILE_COMPILED_MODEL") == "1"
        profile_t0 = time.perf_counter_ns() if profile_boundary else 0

        # Drop stripped SymInt args, if any.
        if self._user_indices is not None:
            user_inputs = [inputs[i] for i in self._user_indices]
        else:
            user_inputs = inputs
        # Positional binding against input_names: never zip-truncate silently.
        if len(user_inputs) != len(self._input_names):
            raise ValueError(
                f"Expected {len(self._input_names)} inputs, got {len(user_inputs)}"
            )

        # Device for outputs: prefer any CUDA input — inputs include lifted
        # weights, and user_inputs[0] may be a CPU-resident weight (offloaded
        # models) while activations live on the GPU.
        input_device = next(
            (t.device for t in user_inputs if t.is_cuda),
            user_inputs[0].device if user_inputs else torch.device("cpu"),
        )

        # Auto-detect dynamic dims from input shapes
        if self._has_dynamic_dims:
            input_shapes = [list(t.shape) for t in user_inputs]
            self._graph.auto_set_dims_from_input_shapes(input_shapes)
        profile_setup_end = time.perf_counter_ns() if profile_boundary else 0

        # Set user input data via pointer.
        # Convert to the graph's expected dtype so bytes match the Input node's dtype tag.
        # For CUDA inputs, keep references alive so the caching allocator doesn't
        # recycle GPU memory before run() reads the pointers.
        _input_refs = []
        bound_cuda_inputs = {}
        for name, tensor, expected_dtype in zip(
            self._input_names, user_inputs, self._input_dtypes
        ):
            if name in self.skip_input_names:
                continue
            if tensor.dtype != expected_dtype:
                raise DTypeBoundaryError(
                    f"Luminal compiled input '{name}' expects "
                    f"{expected_dtype} but got {tensor.dtype}. "
                    "Convert at the call site with "
                    f"`x.to({expected_dtype})` — the boundary used to silently "
                    "cast (and warn) on every call, which masked precision "
                    "bugs and burnt cycles on per-call allocation+copy."
                )
            if self._supports_device_ptrs and tensor.is_cuda:
                t = tensor.detach().contiguous()
                n_bytes = t.numel() * t.element_size()
                self._graph.set_input_device_ptr(name, t.data_ptr(), n_bytes)
                _input_refs.append(t)
                bound_cuda_inputs[name] = t
            else:
                t = tensor.detach().cpu().contiguous()
                n_bytes = t.numel() * t.element_size()
                dtype_code = _torch_dtype_code(t.dtype)
                self._graph.set_input_from_ptr(name, t.data_ptr(), n_bytes, dtype_code)
        profile_input_bind_end = time.perf_counter_ns() if profile_boundary else 0

        # Raw CUDA pointers do not communicate PyTorch's producer-stream
        # dependency to Luminal. Record/wait at the backend boundary after all
        # detach/contiguous work above so Luminal cannot race asynchronous
        # tensor creation or mutation on PyTorch's current stream.
        if self._supports_device_ptrs and _input_refs:
            producer_stream = torch.cuda.current_stream(input_device)
            self._graph.wait_for_external_cuda_stream(producer_stream.cuda_stream)
        profile_stream_handoff_end = time.perf_counter_ns() if profile_boundary else 0

        # Resolve output shapes before run() (needed for pre-allocation).
        if self._has_dynamic_dims:
            output_shapes = self._graph.resolve_output_shapes()
        else:
            output_shapes = self._output_shapes

        # Every declared output MUST have a dtype code; refuse to default
        # to float32 the way we used to if the Rust side returned fewer
        # codes than declared outputs.
        output_dtype_codes = self._graph.output_dtypes
        if len(output_dtype_codes) != len(self._output_names):
            raise RuntimeError(
                f"CompiledGraph returned {len(output_dtype_codes)} output "
                f"dtype codes for {len(self._output_names)} declared outputs "
                f"({self._output_names!r}) — every declared output needs a "
                f"matching dtype."
            )
        output_torch_dtypes = [code_to_torch_dtype(c) for c in output_dtype_codes]
        profile_output_metadata_end = time.perf_counter_ns() if profile_boundary else 0

        # Per-dtype dispatch table mapping `torch_dtype` → the typed
        # `_graph` getter for that dtype. Every supported dtype has an
        # explicit native-width getter; anything not listed raises
        # `NotImplementedError` from `_read_typed_output`. There is no
        # open-ended fallback — a missing entry means we don't know how
        # to read that dtype yet, and we'd rather fail loudly than
        # silently reinterpret bytes.
        #
        # `float16` / `bfloat16` getters return `uint16` bit patterns
        # (Python has no native `f16` / `bf16`); the helper below
        # bit-casts them back to the declared dtype via
        # `torch.frombuffer`. That's a reinterpret, not a numeric
        # cast — no precision change.
        #
        # Narrow ints (`int8` / `int16` / `uint8`) are intentionally
        # absent — luminal's IR refuses them at the FFI boundary (cf.
        # `pt2_util::torch_dtype_int_to_luminal`,
        # `typed_data::from_pytorch_bytes`), so a graph can never
        # declare a narrow-int output that reaches this dispatch.
        # Output registration is byte-oriented in the CUDA runtime, so native
        # integer tensors can be written directly just like floating-point
        # tensors. In particular, LuminalPagedCache returns one int32 positions
        # tensor per layer. Routing those outputs through get_output_i32() would
        # otherwise force a DtoH sync followed by an HtoD copy on every layer
        # and every decode step.
        _zero_copy_cuda_dtypes = (
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
        )
        _output_readers = {
            torch.float32: ("get_output", torch.float32),
            torch.float64: ("get_output_f64", torch.float64),
            torch.float16: ("get_output_f16", torch.float16),
            torch.bfloat16: ("get_output_bf16", torch.bfloat16),
            torch.int64: ("get_output_i64", torch.int64),
            torch.int32: ("get_output_i32", torch.int32),
            torch.bool: ("get_output_bool", torch.bool),
        }

        def _read_typed_output(name: str, shape, out_dtype) -> torch.Tensor:
            """Pull one output back from the runtime at the right dtype.

            Strict: any `out_dtype` not in `_output_readers` raises
            `NotImplementedError`. The previous code's open-ended
            fallback read the buffer as f32 and `.to(out_dtype)`'d
            back, which silently aliased dtypes we don't really
            support; refusing surfaces the gap.

            For `float16` / `bfloat16` the typed getter returns
            `uint16` bit patterns (Python has no native half-precision
            float type); we bit-cast via `torch.tensor(..., uint16)`
            and `.view(half)` so the conversion is a reinterpret of the
            bytes, not a numeric cast.
            """
            entry = _output_readers.get(out_dtype)
            if entry is None:
                raise NotImplementedError(
                    f"Output '{name}' declared dtype {out_dtype} isn't "
                    f"supported by the luminal read boundary. Add a typed "
                    f"getter for this dtype (see `_output_readers`) or cast "
                    f"the output to a supported dtype upstream."
                )
            getter_name, read_dtype = entry
            data = getattr(self._graph, getter_name)(name)
            if len(data) == 0 and all(d != 0 for d in shape):
                return None
            if out_dtype in (torch.float16, torch.bfloat16):
                # Getter returned an immutable `bytes` from Rust; wrap in
                # `bytearray` to make the storage writable (suppresses
                # the "non-writable buffer" warning), then bit-cast via
                # `frombuffer` — no numeric conversion.
                tensor = torch.frombuffer(bytearray(data), dtype=out_dtype)
            else:
                tensor = torch.tensor(data, dtype=read_dtype)
            tensor = tensor.reshape(tuple(shape))
            return tensor.to(input_device)

        # Pre-allocation is GPU-only: the CUDA kernel needs the
        # output's device pointer registered *before* `_graph.run()`
        # so the final kernel writes directly into PyTorch's buffer.
        # Dtypes in `_zero_copy_cuda_dtypes` take the direct-output path;
        # remaining dtypes read back via `_read_typed_output` after `run()`.
        # CPU never zero-copies — there is no separate device buffer to
        # register against.
        _use_zero_copy = self._supports_device_ptrs
        output_tensors = []
        direct_writeback_positions = {}
        direct_writeback_by_name = {}
        direct_state_output_positions = set()
        output_pointer_registrations = 0
        output_allocations = 0
        if _use_zero_copy:
            for position, input_name in self._writeback_by_pos.items():
                name = self._output_names[position]
                target = user_inputs[self._input_names.index(input_name)]
                bound_target = bound_cuda_inputs.get(input_name)
                if (
                    bound_target is not None
                    and bound_target.dtype == output_torch_dtypes[position]
                    and tuple(bound_target.shape) == tuple(output_shapes[position])
                ):
                    direct_writeback_by_name[name] = (target, bound_target)
        if _use_zero_copy:
            for i, (name, shape) in enumerate(zip(self._output_names, output_shapes)):
                out_dtype = output_torch_dtypes[i]
                if i in self._writeback_by_pos:
                    # Bind CUDA mutation outputs directly to the caller's
                    # original input allocation. If the selected LLIR output
                    # aliases that input (for example KernelScatterNoCopy), the
                    # runtime performs no copy. A materializing candidate still
                    # honors the registration with a device-to-device epilogue.
                    # Either way the persistent state's data_ptr stays stable.
                    state = direct_writeback_by_name.get(name)
                    if state is not None:
                        target, bound_target = state
                        self._graph.set_output_device_ptr(
                            name,
                            bound_target.data_ptr(),
                            bound_target.numel() * bound_target.element_size(),
                        )
                        output_pointer_registrations += 1
                        direct_writeback_positions[i] = (target, bound_target)
                    output_tensors.append(None)
                    continue
                if name in direct_writeback_by_name:
                    # Export may expose the same mutated tensor again as a user
                    # output. Keep that duplicate on the state allocation too;
                    # registering a fresh output here would overwrite the
                    # write-back registration (registrations are keyed by name).
                    target, bound_target = direct_writeback_by_name[name]
                    self._graph.set_output_device_ptr(
                        name,
                        bound_target.data_ptr(),
                        bound_target.numel() * bound_target.element_size(),
                    )
                    output_pointer_registrations += 1
                    output_tensors.append(target)
                    direct_state_output_positions.add(i)
                    continue
                out = torch.empty(shape, dtype=out_dtype, device=input_device)
                output_allocations += 1
                if out_dtype in _zero_copy_cuda_dtypes:
                    self._graph.set_output_device_ptr(
                        name, out.data_ptr(), out.numel() * out.element_size()
                    )
                    output_pointer_registrations += 1
                output_tensors.append(out)

        profile_output_plan_end = time.perf_counter_ns() if profile_boundary else 0
        self._graph.run()
        profile_graph_run_end = time.perf_counter_ns() if profile_boundary else 0

        outputs = []
        for i, (name, shape) in enumerate(zip(self._output_names, output_shapes)):
            out_dtype = output_torch_dtypes[i]
            if i in self._writeback_by_pos:
                # In-place input mutation: copy the computed state back into
                # the caller's tensor (the same object the model would have
                # mutated eagerly); it is not part of the returned tuple.
                # CUDA mutations registered directly on that tensor before
                # run() are already complete and require no readback or copy.
                if i in direct_writeback_positions:
                    target, bound_target = direct_writeback_positions[i]
                    if target.data_ptr() != bound_target.data_ptr():
                        target.copy_(bound_target)
                    continue
                input_name = self._writeback_by_pos[i]
                target = user_inputs[self._input_names.index(input_name)]
                target.copy_(_read_typed_output(name, shape, out_dtype))
                continue
            if i in direct_state_output_positions:
                out = output_tensors[i]
            elif _use_zero_copy and out_dtype in _zero_copy_cuda_dtypes:
                out = output_tensors[i]
                if not self._graph.output_is_zero_copy(name):
                    self._graph.copy_output_to_device_ptr(
                        name, out.data_ptr(), out.numel() * out.element_size()
                    )
            else:
                out = _read_typed_output(name, shape, out_dtype)
            outputs.append(out)

            if (
                os.getenv("LUMINAL_DEBUG_CACHE_POSITIONS")
                and out_dtype == torch.int32
                and out.ndim == 1
            ):
                print(
                    "LUMINAL_CACHE_POSITIONS"
                    f" name={name} shape={tuple(out.shape)}"
                    f" values={out.detach().cpu().tolist()}",
                    flush=True,
                )

        if profile_boundary:
            profile_end = time.perf_counter_ns()

            def to_ms(duration_ns):
                return duration_ns / 1_000_000.0

            call_index = self._profile_call_index
            self._profile_call_index += 1
            print(
                "LUMINAL_COMPILED_MODEL_PROFILE"
                f" call={call_index}"
                f" inputs={len(user_inputs)}"
                f" cuda_inputs={len(_input_refs)}"
                f" outputs={len(self._output_names)}"
                f" output_allocations={output_allocations}"
                f" output_registrations={output_pointer_registrations}"
                f" writebacks={len(self._writeback_by_pos)}"
                f" total_ms={to_ms(profile_end - profile_t0):.3f}"
                f" setup_ms={to_ms(profile_setup_end - profile_t0):.3f}"
                f" input_bind_ms={to_ms(profile_input_bind_end - profile_setup_end):.3f}"
                f" stream_handoff_ms={to_ms(profile_stream_handoff_end - profile_input_bind_end):.3f}"
                f" output_metadata_ms={to_ms(profile_output_metadata_end - profile_stream_handoff_end):.3f}"
                f" output_plan_ms={to_ms(profile_output_plan_end - profile_output_metadata_end):.3f}"
                f" graph_run_ms={to_ms(profile_graph_run_end - profile_output_plan_end):.3f}"
                f" output_finalize_ms={to_ms(profile_end - profile_graph_run_end):.3f}",
                flush=True,
            )

        return tuple(outputs)
