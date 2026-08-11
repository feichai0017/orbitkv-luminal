"""CompiledModel wrapper for the Rust CompiledGraph."""

import itertools
import json
import math
import os
import threading
import time
from pathlib import Path

import torch

from .dtype_util import code_to_torch_dtype
from .dtype_util import torch_dtype_code as _torch_dtype_code

_PROFILE_GRAPH_IDS = itertools.count(1)
_PROFILE_INVOCATION_IDS = itertools.count(1)
_PROFILE_WRITE_LOCK = threading.Lock()


def _cuda_input_binding_signature(tensor, n_bytes: int) -> tuple:
    """Return the metadata that determines an external CUDA binding.

    Tensor contents are deliberately absent from the signature. A producer may
    update the same allocation between invocations without changing anything
    about the compiled graph's pointer binding.
    """
    return (tensor.device, tensor.data_ptr(), n_bytes, tensor.dtype)


class _DisabledProfileStage:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


class _EnabledProfileStage:
    __slots__ = ("label", "timings", "name", "use_nvtx", "start")

    def __init__(self, label: str, timings: dict, name: str):
        self.label = label
        self.timings = timings
        self.name = name

    def __enter__(self):
        self.use_nvtx = torch.cuda.is_available()
        if self.use_nvtx:
            torch.cuda.nvtx.range_push(self.label)
        self.start = time.perf_counter_ns()
        return None

    def __exit__(self, _exc_type, _exc, _traceback):
        self.timings[self.name] = (time.perf_counter_ns() - self.start) / 1_000.0
        if self.use_nvtx:
            torch.cuda.nvtx.range_pop()
        return False


_DISABLED_PROFILE_STAGE = _DisabledProfileStage()


def _profile_stage(enabled: bool, label: str, timings: dict, name: str):
    if not enabled:
        return _DISABLED_PROFILE_STAGE
    return _EnabledProfileStage(label, timings, name)


def _append_profile_record(path: str, record: dict) -> None:
    """Append one complete JSON record using a single OS write.

    O_APPEND prevents independent benchmark workers from sharing a mutable file
    position. The process-local lock additionally keeps Python threads ordered.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    with _PROFILE_WRITE_LOCK:
        fd = os.open(destination, flags, 0o644)
        try:
            written = os.write(fd, payload)
            if written != len(payload):
                raise OSError(
                    f"short write to {destination}: wrote {written} of "
                    f"{len(payload)} bytes"
                )
        finally:
            os.close(fd)


def _infer_logical_tokens(inputs) -> int | None:
    """Infer sequence width from integer [batch, sequence] input metadata.

    This is diagnostic-only and never reads tensor contents. It deliberately
    returns None instead of guessing when the signature has no such tensor.
    """
    candidates = [
        int(t.shape[1])
        for t in inputs
        if isinstance(t, torch.Tensor)
        and t.dtype in (torch.int32, torch.int64)
        and t.ndim == 2
        and int(t.shape[0]) >= 1
        and int(t.shape[1]) >= 1
    ]
    return candidates[0] if candidates else None


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
        self,
        graph_result,
        weight_refs=None,
        input_names=None,
        user_indices=None,
        frozen_parameter_names=None,
    ):
        """Initialize with a compiled CompiledGraph from Rust.

        Args:
            graph_result: The CompiledGraph from luminal_python.process_pt2()
            weight_refs: List of PyTorch tensors to keep alive (prevents GC of shared weights)
            input_names: Override for user input names. If None, uses graph_result.input_names.
            user_indices: When torch.compile lifts model parameters into extra args,
                this tells __call__ which arg positions are actual user inputs.
                None means all args are user inputs (PT2 path).
            frozen_parameter_names: Input names proven to originate from
                ``nn.Parameter`` objects during PT2 export.
        """
        self._graph = graph_result
        self._input_names = input_names or graph_result.input_names
        self._output_names = graph_result.output_names
        # {output position: mutated input name} for the write-back outputs
        # torch.export's functionalization appends for in-place input
        # mutations. Keyed by position, not name: a model that mutates an
        # input and also returns it yields two same-named outputs.
        self._writeback_by_pos = dict(graph_result.writeback_outputs)
        input_positions = {name: pos for pos, name in enumerate(self._input_names)}
        self._writeback_input_pos = {
            output_pos: input_positions[input_name]
            for output_pos, input_name in self._writeback_by_pos.items()
        }
        self._output_shapes = graph_result.output_shapes
        self._has_dynamic_dims = getattr(graph_result, "has_dynamic_dims", False)
        self._weight_refs = weight_refs or []
        self._frozen_parameter_names = frozenset(frozen_parameter_names or ())
        self._user_indices = user_indices
        self._is_bound = False
        self.skip_input_names = frozenset()
        self._is_gpu = getattr(graph_result, "device_type", "cpu") != "cpu"
        self._supports_device_ptrs = getattr(
            graph_result, "supports_device_ptrs", False
        )
        # name -> (device, pointer, required bytes, dtype, strong tensor ref).
        # CUDA bindings are persistent in the runtime; only changed metadata
        # needs to cross PyO3 on subsequent calls.
        self._cuda_input_bindings = {}
        # output position -> (device, pointer, required bytes, dtype, strong tensor ref).
        # Functionalized mutation outputs normally target long-lived state
        # tensors, so their durable registrations cross PyO3 only when the
        # actual storage changes.
        self._cuda_writeback_bindings = {}
        self._profile_graph_id = next(_PROFILE_GRAPH_IDS)
        self._profile_call_index = 0
        # Profiling is a property of this compiled instance. Resolve the
        # opt-in once instead of consulting the process environment on every
        # invocation of the model's hot path.
        self._profile_path = os.getenv("LUMINAL_PROFILE_JSONL")
        self._profile_enabled = bool(self._profile_path)
        if hasattr(self._graph, "set_structured_profiling"):
            self._graph.set_structured_profiling(self._profile_enabled)
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
        if self._is_bound:
            raise RuntimeError("Cannot change dimensions after binding this model")
        self._graph.set_dim(param_name, value)

    def _select_user_inputs(self, inputs):
        if self._user_indices is not None:
            return tuple(inputs[i] for i in self._user_indices)
        return tuple(inputs)

    def bind(self, *inputs: torch.Tensor):
        """Bind fixed CUDA storage once for scan-free serving replay.

        Binding is exclusive for this CompiledModel. The returned executable
        retains every tensor whose device pointer Rust uses; callers update
        invocation-buffer contents in place and call ``run()`` with no args.
        """
        if self._is_bound:
            raise RuntimeError("This CompiledModel already has a bound executable")
        if not self._supports_device_ptrs:
            raise RuntimeError("Bound execution requires a device-pointer backend")
        if self.skip_input_names:
            raise RuntimeError("Bound execution does not support skipped inputs")

        user_inputs = self._select_user_inputs(inputs)
        if len(user_inputs) != len(self._input_names):
            raise ValueError(
                f"Expected {len(self._input_names)} inputs, got {len(user_inputs)}"
            )

        if self._has_dynamic_dims:
            self._graph.auto_set_dims_from_input_shapes(
                [list(tensor.shape) for tensor in user_inputs]
            )
            output_shapes = self._graph.resolve_output_shapes()
        else:
            output_shapes = self._output_shapes

        output_dtype_codes = self._graph.output_dtypes
        if len(output_dtype_codes) != len(self._output_names):
            raise RuntimeError(
                f"CompiledGraph returned {len(output_dtype_codes)} output dtype "
                f"codes for {len(self._output_names)} declared outputs"
            )
        output_dtypes = [code_to_torch_dtype(code) for code in output_dtype_codes]
        native_output_dtypes = (torch.float32, torch.float16, torch.bfloat16)

        input_refs = []
        input_signatures = []
        for position, (name, tensor, expected_dtype) in enumerate(
            zip(self._input_names, user_inputs, self._input_dtypes)
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Bound input '{name}' is not a tensor")
            if not tensor.is_cuda:
                raise ValueError(f"Bound input '{name}' must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"Bound input '{name}' must be contiguous")
            if tensor.dtype != expected_dtype:
                raise DTypeBoundaryError(
                    f"Luminal compiled input '{name}' expects {expected_dtype} "
                    f"but got {tensor.dtype}"
                )
            n_bytes = tensor.numel() * tensor.element_size()
            self._graph.set_input_device_ptr(name, tensor.data_ptr(), n_bytes)
            self._cuda_input_bindings[name] = (
                *_cuda_input_binding_signature(tensor, n_bytes),
                tensor,
            )
            input_refs.append(tensor)
            input_signatures.append(
                (position, name, tuple(tensor.shape), tuple(tensor.stride()), n_bytes)
            )

        persistent_outputs = []
        output_destinations = []
        durable_input_positions = set(self._writeback_input_pos.values())
        for position, (name, shape, dtype) in enumerate(
            zip(self._output_names, output_shapes, output_dtypes)
        ):
            if position in self._writeback_by_pos:
                target = user_inputs[self._writeback_input_pos[position]]
                expected_numel = math.prod(shape)
                if target.numel() != expected_numel or target.dtype != dtype:
                    raise ValueError(
                        f"Writeback output '{name}' does not match its bound target"
                    )
            else:
                if dtype not in native_output_dtypes:
                    raise NotImplementedError(
                        f"Bound output '{name}' has unsupported dtype {dtype}; "
                        "ordinary bound outputs currently require a native "
                        "CUDA float dtype"
                    )
                target = torch.empty(shape, dtype=dtype, device=user_inputs[0].device)
                persistent_outputs.append(target)
            n_bytes = target.numel() * target.element_size()
            self._graph.set_output_device_ptr_at(
                position, target.data_ptr(), n_bytes
            )
            output_destinations.append((position, target, n_bytes))

        frozen_positions = tuple(
            position
            for position, name in enumerate(self._input_names)
            if name in self._frozen_parameter_names
        )
        overlap = durable_input_positions.intersection(frozen_positions)
        if overlap:
            raise ValueError(
                "Inputs cannot be both frozen parameters and durable state: "
                f"{sorted(overlap)}"
            )

        bound = BoundCompiledModel(
            parent=self,
            input_refs=tuple(input_refs),
            input_signatures=tuple(input_signatures),
            outputs=tuple(persistent_outputs),
            output_destinations=tuple(output_destinations),
            frozen_count=len(frozen_positions),
            durable_count=len(durable_input_positions),
            invocation_count=(
                len(user_inputs) - len(frozen_positions) - len(durable_input_positions)
            ),
        )
        self._is_bound = True
        return bound

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

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        """Execute the compiled model with PyTorch tensor inputs.

        Args:
            *inputs: PyTorch tensors. When torch.compile lifts model parameters,
                this includes both weights and user inputs. user_indices filters
                to just the user inputs.

        Returns:
            Tuple of PyTorch tensors containing the model outputs
        """
        if self._is_bound:
            raise RuntimeError(
                "This CompiledModel is bound; call BoundCompiledModel.run()"
            )
        profile_path = self._profile_path
        profile_enabled = self._profile_enabled
        profile_total_start = time.perf_counter_ns() if profile_enabled else None
        profile_timings = {}
        invocation_id = next(_PROFILE_INVOCATION_IDS) if profile_enabled else None
        call_index = self._profile_call_index
        if profile_enabled:
            self._profile_call_index += 1

        with _profile_stage(
            profile_enabled,
            f"luminal.compiled_model.setup.{invocation_id}",
            profile_timings,
            "setup",
        ):
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
            logical_tokens = (
                _infer_logical_tokens(user_inputs) if profile_enabled else None
            )

        # Auto-detect dynamic dims from input shapes
        with _profile_stage(
            profile_enabled,
            f"luminal.compiled_model.dynamic_dims.{invocation_id}",
            profile_timings,
            "dynamic_dims",
        ):
            if self._has_dynamic_dims:
                input_shapes = [list(t.shape) for t in user_inputs]
                self._graph.auto_set_dims_from_input_shapes(input_shapes)

        # Set user input data via pointer.
        # Convert to the graph's expected dtype so bytes match the Input node's dtype tag.
        # For CUDA inputs, keep references alive so the caching allocator doesn't
        # recycle GPU memory before run() reads the pointers.
        _input_refs = []
        input_bindings = 0
        changed_input_bindings = 0
        with _profile_stage(
            profile_enabled,
            f"luminal.compiled_model.input_bind.{invocation_id}",
            profile_timings,
            "input_bind",
        ):
            for name, tensor, expected_dtype in zip(
                self._input_names, user_inputs, self._input_dtypes
            ):
                if name in self.skip_input_names:
                    continue
                input_bindings += 1
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
                    # A contiguous caller tensor is already a valid read-only
                    # boundary input; making a detached view for every lifted
                    # weight adds hundreds of Python objects per invocation.
                    t = (
                        tensor
                        if tensor.is_contiguous()
                        else tensor.detach().contiguous()
                    )
                    n_bytes = t.numel() * t.element_size()
                    signature = _cuda_input_binding_signature(t, n_bytes)
                    previous = self._cuda_input_bindings.get(name)
                    if previous is None or previous[:4] != signature:
                        self._graph.set_input_device_ptr(name, t.data_ptr(), n_bytes)
                        changed_input_bindings += 1
                    # Commit only after a changed registration succeeds. Retaining
                    # the tensor prevents allocator reuse while Rust holds its
                    # non-owning pointer; `previous` keeps the old allocation alive
                    # until the replacement has crossed the boundary.
                    self._cuda_input_bindings[name] = (*signature, t)
                    _input_refs.append(t)
                else:
                    t = tensor.detach().cpu().contiguous()
                    n_bytes = t.numel() * t.element_size()
                    dtype_code = _torch_dtype_code(t.dtype)
                    self._graph.set_input_from_ptr(
                        name, t.data_ptr(), n_bytes, dtype_code
                    )
                    changed_input_bindings += 1

        # Resolve output shapes before run() (needed for pre-allocation).
        with _profile_stage(
            profile_enabled,
            f"luminal.compiled_model.output_metadata.{invocation_id}",
            profile_timings,
            "output_metadata",
        ):
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
        _zero_copy_native_floats = (torch.float32, torch.float16, torch.bfloat16)
        _output_readers = {
            torch.float32: ("get_output", torch.float32),
            torch.float64: ("get_output_f64", torch.float64),
            torch.float16: ("get_output_f16", torch.float16),
            torch.bfloat16: ("get_output_bf16", torch.bfloat16),
            torch.int64: ("get_output_i64", torch.int64),
            torch.int32: ("get_output_i32", torch.int32),
            torch.bool: ("get_output_bool", torch.bool),
        }

        def _read_typed_output(
            position: int, name: str, shape, out_dtype
        ) -> torch.Tensor:
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
            data = getattr(self._graph, f"{getter_name}_at")(position)
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
        # Only the float dtypes luminal natively writes
        # (`_zero_copy_native_floats`) take the zero-copy path; other
        # dtypes (int*, bool, f64) read back via `_read_typed_output`
        # after `run()` and so don't need a pre-allocated tensor at
        # this layer. CPU never zero-copies — there's no separate
        # device buffer to register against.
        _use_zero_copy = self._supports_device_ptrs
        output_tensors = []
        output_allocations = 0
        output_registrations = 0
        direct_writebacks = set()
        with _profile_stage(
            profile_enabled,
            f"luminal.compiled_model.output_plan.{invocation_id}",
            profile_timings,
            "output_plan",
        ):
            if _use_zero_copy:
                for i, (name, shape) in enumerate(
                    zip(self._output_names, output_shapes)
                ):
                    out_dtype = output_torch_dtypes[i]
                    if i in self._writeback_by_pos:
                        # Point functionalized mutation outputs at the caller's
                        # state tensor up front. The CUDA runtime either writes
                        # there directly or schedules its required epilogue D2D
                        # copy on the graph stream before the one terminal wait.
                        target = user_inputs[self._writeback_input_pos[i]]
                        expected_numel = math.prod(shape)
                        if (
                            target.is_cuda
                            and target.is_contiguous()
                            and target.dtype == out_dtype
                            and target.numel() == expected_numel
                        ):
                            n_bytes = target.numel() * target.element_size()
                            signature = _cuda_input_binding_signature(target, n_bytes)
                            previous = self._cuda_writeback_bindings.get(i)
                            if previous is None or previous[:4] != signature:
                                self._graph.set_output_device_ptr_at(
                                    i, target.data_ptr(), n_bytes
                                )
                                output_registrations += 1
                            self._cuda_writeback_bindings[i] = (*signature, target)
                            direct_writebacks.add(i)
                        elif i in self._cuda_writeback_bindings:
                            self._graph.clear_output_device_ptr_at(i)
                            del self._cuda_writeback_bindings[i]
                        output_tensors.append(None)
                        continue
                    out = torch.empty(shape, dtype=out_dtype, device=input_device)
                    output_allocations += 1
                    if out_dtype in _zero_copy_native_floats:
                        self._graph.set_output_device_ptr_at(
                            i, out.data_ptr(), out.numel() * out.element_size()
                        )
                        output_registrations += 1
                    output_tensors.append(out)

        with _profile_stage(
            profile_enabled,
            f"luminal.compiled_model.graph_run.{invocation_id}",
            profile_timings,
            "graph_run",
        ):
            if profile_enabled:
                self._graph.set_profile_invocation_id(invocation_id)
            self._graph.run()
        runtime_profile_json = (
            self._graph.take_last_execution_profile_json() if profile_enabled else None
        )

        outputs = []
        gpu_writebacks = []
        cpu_writebacks = 0
        with _profile_stage(
            profile_enabled,
            f"luminal.compiled_model.output_finalize.{invocation_id}",
            profile_timings,
            "output_finalize",
        ):
            for i, (name, shape) in enumerate(zip(self._output_names, output_shapes)):
                out_dtype = output_torch_dtypes[i]
                if i in self._writeback_by_pos:
                    # In-place input mutation: copy the computed state back into
                    # the caller's tensor (the same object the model would have
                    # mutated eagerly); it is not part of the returned tuple.
                    target = user_inputs[self._writeback_input_pos[i]]
                    if i in direct_writebacks:
                        continue
                    expected_numel = math.prod(shape)
                    can_copy_on_device = (
                        self._supports_device_ptrs
                        and hasattr(self._graph, "copy_outputs_to_device_ptrs_at")
                        and target.is_cuda
                        and target.is_contiguous()
                        and target.dtype == out_dtype
                        and target.numel() == expected_numel
                    )
                    if can_copy_on_device:
                        gpu_writebacks.append(
                            (
                                i,
                                target.data_ptr(),
                                target.numel() * target.element_size(),
                            )
                        )
                    else:
                        target.copy_(_read_typed_output(i, name, shape, out_dtype))
                        cpu_writebacks += 1
                    continue
                if _use_zero_copy and out_dtype in _zero_copy_native_floats:
                    out = output_tensors[i]
                    if not self._graph.output_is_zero_copy_at(i):
                        self._graph.copy_output_to_device_ptr_at(
                            i, out.data_ptr(), out.numel() * out.element_size()
                        )
                else:
                    out = _read_typed_output(i, name, shape, out_dtype)
                outputs.append(out)

            if gpu_writebacks:
                self._graph.copy_outputs_to_device_ptrs_at(gpu_writebacks)

        if profile_enabled:
            total_us = (time.perf_counter_ns() - profile_total_start) / 1_000.0
            named_us = sum(profile_timings.values())
            runtime_profile = (
                json.loads(runtime_profile_json) if runtime_profile_json else None
            )
            runtime_boundary_us = max(
                0.0,
                profile_timings.get("graph_run", 0.0)
                - (
                    runtime_profile.get("timings_us", {}).get("total", 0.0)
                    if runtime_profile
                    else profile_timings.get("graph_run", 0.0)
                ),
            )
            record = {
                "schema_version": 1,
                "kind": "luminal_invocation",
                "pid": os.getpid(),
                "thread": threading.get_ident(),
                "graph": self._profile_graph_id,
                "invocation": invocation_id,
                "call_index": call_index,
                "phase": (
                    "decode"
                    if logical_tokens == 1
                    else "prefill"
                    if logical_tokens is not None and logical_tokens > 1
                    else "unknown"
                ),
                "logical_tokens": logical_tokens,
                "execution_mode": "generic",
                "compiled_model": {
                    "timings_us": {
                        "total": total_us,
                        **profile_timings,
                        "runtime_boundary": runtime_boundary_us,
                        "stream_handoff": 0.0,
                        "unattributed": max(0.0, total_us - named_us),
                    },
                    "counts": {
                        "inputs": len(user_inputs),
                        "input_bindings": input_bindings,
                        "bindings_inspected": input_bindings,
                        "changed_input_bindings": changed_input_bindings,
                        "frozen_bindings": len(self._frozen_parameter_names),
                        "durable_bindings": len(
                            set(self._writeback_input_pos.values())
                        ),
                        "invocation_buffers": (
                            len(user_inputs)
                            - len(self._frozen_parameter_names)
                            - len(set(self._writeback_input_pos.values()))
                        ),
                        "outputs": len(self._output_names),
                        "ordinary_outputs_planned": (
                            len(self._output_names) - len(self._writeback_by_pos)
                        ),
                        "durable_outputs_registered": len(direct_writebacks),
                        "output_allocations": output_allocations,
                        "output_registrations": output_registrations,
                        "writebacks": len(self._writeback_by_pos),
                        "gpu_writebacks": len(gpu_writebacks),
                        "cpu_writebacks": cpu_writebacks,
                        "writeback_batches": int(bool(gpu_writebacks)),
                        "direct_writebacks": len(direct_writebacks),
                    },
                },
                "runtime": runtime_profile,
            }
            _append_profile_record(profile_path, record)

        return tuple(outputs)


class BoundCompiledModel:
    """A fixed-storage serving executable with a scan-free steady hot path.

    Inputs and outputs are registered by :meth:`CompiledModel.bind`. ``run``
    accepts no tensors: callers mutate invocation-buffer contents in place.
    Returned output tensors are persistent and overwritten by the next replay.
    """

    def __init__(
        self,
        *,
        parent,
        input_refs,
        input_signatures,
        outputs,
        output_destinations,
        frozen_count,
        durable_count,
        invocation_count,
    ):
        self._parent = parent
        self._graph = parent._graph
        self._input_refs = input_refs
        self._input_signatures = input_signatures
        self._outputs = outputs
        self._output_destinations = output_destinations
        self._fallback_output_copies = None
        self._frozen_count = frozen_count
        self._durable_count = durable_count
        self._invocation_count = invocation_count

    @property
    def outputs(self):
        """Persistent ordinary-output tensors, in graph return order."""
        return self._outputs

    def _finish_output_binding(self):
        """Resolve lazy CUDA output aliases after the first bound execution."""
        fallback = []
        for position, target, n_bytes in self._output_destinations:
            if not self._graph.output_is_zero_copy_at(position):
                fallback.append((position, target.data_ptr(), n_bytes))
        self._fallback_output_copies = tuple(fallback)
        self._output_destinations = ()

    def run(self):
        """Replay against already-bound storage and return persistent outputs."""
        parent = self._parent
        profile_enabled = parent._profile_enabled
        profile_path = parent._profile_path
        profile_total_start = time.perf_counter_ns() if profile_enabled else None
        invocation_id = next(_PROFILE_INVOCATION_IDS) if profile_enabled else None
        call_index = parent._profile_call_index
        if profile_enabled:
            parent._profile_call_index += 1
        profile_timings = {}

        with _profile_stage(
            profile_enabled,
            f"luminal.bound_model.graph_run.{invocation_id}",
            profile_timings,
            "graph_run",
        ):
            if profile_enabled:
                self._graph.set_profile_invocation_id(invocation_id)
            self._graph.run()

        first_run = self._fallback_output_copies is None
        if first_run:
            self._finish_output_binding()
        if self._fallback_output_copies:
            with _profile_stage(
                profile_enabled,
                f"luminal.bound_model.output_copy.{invocation_id}",
                profile_timings,
                "output_copy",
            ):
                self._graph.copy_outputs_to_device_ptrs_at(
                    self._fallback_output_copies
                )

        runtime_profile_json = (
            self._graph.take_last_execution_profile_json()
            if profile_enabled
            else None
        )
        if profile_enabled:
            total_us = (time.perf_counter_ns() - profile_total_start) / 1_000.0
            runtime_profile = (
                json.loads(runtime_profile_json) if runtime_profile_json else None
            )
            record = {
                "schema_version": 1,
                "kind": "luminal_invocation",
                "pid": os.getpid(),
                "thread": threading.get_ident(),
                "graph": parent._profile_graph_id,
                "invocation": invocation_id,
                "call_index": call_index,
                "phase": "decode",
                "logical_tokens": 1,
                "execution_mode": "bound",
                "compiled_model": {
                    "timings_us": {"total": total_us, **profile_timings},
                    "counts": {
                        "inputs": len(self._input_refs),
                        "input_bindings": 0,
                        "bindings_inspected": 0,
                        "changed_input_bindings": 0,
                        "frozen_bindings": self._frozen_count,
                        "durable_bindings": self._durable_count,
                        "invocation_buffers": self._invocation_count,
                        "outputs": len(parent._output_names),
                        "ordinary_outputs_planned": 0,
                        "durable_outputs_registered": 0,
                        "fallback_output_copies": len(
                            self._fallback_output_copies
                        ),
                        "first_bound_run": int(first_run),
                    },
                },
                "runtime": runtime_profile,
            }
            _append_profile_record(profile_path, record)

        return self._outputs
