//! Generic PyTorch invocation for a compiled Luminal graph.
//!
//! This path implements the normal `torch.compile` contract: every runtime
//! tensor is inspected on every invocation and may be backed by new storage.

use luminal::{prelude::*, shape::Expression};
use pyo3::{
    prelude::*,
    types::{PyAny, PyTuple, PyTupleMethods},
};

use crate::{
    compiled_graph::{CompiledGraph, SingleVarDimSolver, copy_host_bytes},
    tensor_bridge::{TensorObservation, TorchApi, is_zero_copy_output_dtype},
    torch_dtype::TorchDType,
    typed_data::TypedData,
};

/// One external tensor argument consumed by the compiled graph.
///
/// This includes ordinary user inputs and Dynamo-lifted parameters or buffers,
/// but never graph outputs or weights already baked into the runtime. Its
/// dimension plans are applied immediately while that input is inspected.
#[derive(Clone)]
pub(crate) struct InputPlan {
    pub name: String,
    pub argument_index: usize,
    pub node: NodeIndex,
    pub dtype_code: u32,
    pub expected_rank: usize,
    pub static_dims: Vec<StaticDimPlan>,
    pub dynamic_dims: Vec<DynamicDimPlan>,
}

/// A dimension of an external input that can set one symbolic graph dimension.
///
/// The affine solver is compiled once; invocation only reads the indexed
/// concrete input dimension and applies the solver.
#[derive(Clone)]
pub(crate) struct DynamicDimPlan {
    pub dimension_position: usize,
    expression: Expression,
    solver: SingleVarDimSolver,
}

impl DynamicDimPlan {
    pub(crate) fn from_expression(
        dimension_position: usize,
        expression: Expression,
    ) -> Option<Self> {
        let solver = SingleVarDimSolver::from_expression(expression)?;
        Some(Self {
            dimension_position,
            expression,
            solver,
        })
    }

    fn apply(&self, graph: &mut CompiledGraph, input_shape: &[usize]) -> PyResult<()> {
        let observed = input_shape
            .get(self.dimension_position)
            .copied()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "input has no dimension {} required by dynamic shape plan",
                    self.dimension_position
                ))
            })?;
        let (variable, value) = self.solver.solve(observed).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "input dimension {} with size {observed} does not satisfy dynamic expression {:?}",
                self.dimension_position, self.expression
            ))
        })?;
        graph.graph.set_dim(variable, value);
        Ok(())
    }
}

/// A dimension of an external input whose compiled size must remain fixed.
///
/// Static and dynamic plans describe dimensions, so one input tensor may be
/// referenced by both kinds of plan.
#[derive(Clone)]
pub(crate) struct StaticDimPlan {
    pub dimension_position: usize,
    pub expected: usize,
}

/// One graph output returned to the Python caller.
///
/// Writeback-only outputs use `WritebackPlan` instead and are not returned.
#[derive(Clone)]
pub(crate) struct OutputPlan {
    pub position: usize,
    pub name: String,
    pub node: NodeIndex,
    pub dtype_code: u32,
    pub scalar: bool,
}

/// A graph output that updates an external input buffer, such as a KV cache.
///
/// `input_position` indexes `ExecutionPlan::inputs`; `output` describes the
/// graph node whose result must be written there.
#[derive(Clone)]
pub(crate) struct WritebackPlan {
    pub output: OutputPlan,
    pub input_position: usize,
}

/// Immutable, artifact-local description of all work needed around execution.
///
/// It pre-resolves graph nodes and groups input binding, shape processing,
/// returned outputs, and state writebacks so invocation does not rediscover
/// those roles on every call.
#[derive(Clone)]
pub(crate) struct ExecutionPlan {
    pub inputs: Vec<InputPlan>,
    pub exact_argument_count: Option<usize>,
    pub returned_outputs: Vec<OutputPlan>,
    pub writebacks: Vec<WritebackPlan>,
    pub has_dynamic_dims: bool,
    pub static_output_shapes: Vec<Vec<usize>>,
}

impl ExecutionPlan {
    pub(crate) fn apply_input_shape(
        &self,
        graph: &mut CompiledGraph,
        input_position: usize,
        input_shape: &[usize],
    ) -> PyResult<()> {
        let input = &self.inputs[input_position];
        for static_dim in &input.static_dims {
            let observed = input_shape
                .get(static_dim.dimension_position)
                .copied()
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "input {} has no dimension {} required by its compiled shape",
                        input_position, static_dim.dimension_position
                    ))
                })?;
            if observed != static_dim.expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "input {} dimension {} expected size {}, got {observed}",
                    input_position, static_dim.dimension_position, static_dim.expected
                )));
            }
        }
        for dynamic_dim in &input.dynamic_dims {
            dynamic_dim.apply(graph, input_shape)?;
        }
        Ok(())
    }
}

pub(crate) struct CachedBinding {
    pub device_type: i32,
    pub device_id: i32,
    pub pointer: u64,
    pub n_bytes: usize,
    pub dtype_code: u32,
    pub _tensor: Py<PyAny>,
}

struct PreparedOutput {
    tensor: Py<PyAny>,
    data_ptr: u64,
    n_bytes: usize,
}

#[derive(Default)]
pub(crate) struct TorchInvocationState {
    pub plan: Option<ExecutionPlan>,
    torch_api: Option<TorchApi>,
    input_bindings: Vec<Option<CachedBinding>>,
    writeback_bindings: Vec<Option<CachedBinding>>,
    prepared_outputs: Vec<Option<PreparedOutput>>,
    direct_writebacks: Vec<bool>,
    gpu_output_copies: Vec<(NodeIndex, u64, usize)>,
    external_outputs_registered: bool,
}

impl TorchInvocationState {
    pub(crate) fn configure(&mut self, plan: ExecutionPlan) {
        self.input_bindings = (0..plan.inputs.len()).map(|_| None).collect();
        self.writeback_bindings = (0..plan.writebacks.len()).map(|_| None).collect();
        self.prepared_outputs = Vec::with_capacity(plan.returned_outputs.len());
        self.direct_writebacks = vec![false; plan.writebacks.len()];
        self.gpu_output_copies =
            Vec::with_capacity(plan.writebacks.len() + plan.returned_outputs.len());
        self.external_outputs_registered = false;
        self.plan = Some(plan);
    }
}

pub(crate) fn invoke(
    graph: &mut CompiledGraph,
    py: Python<'_>,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let mut state = std::mem::take(&mut graph.torch_invocation);
    let result = invoke_with_state(graph, &mut state, py, inputs);
    graph.torch_invocation = state;
    result
}

fn invoke_with_state(
    graph: &mut CompiledGraph,
    state: &mut TorchInvocationState,
    py: Python<'_>,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    if state.torch_api.is_none() {
        state.torch_api = Some(TorchApi::new(py)?);
    }
    let api = state.torch_api.as_ref().expect("TorchApi was initialized");
    let plan = state.plan.as_ref().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "CompiledGraph invocation was not configured by CompiledModel",
        )
    })?;
    if let Some(expected) = plan.exact_argument_count
        && inputs.len() != expected
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {expected} inputs, got {}",
            inputs.len()
        )));
    }

    let supports_external_buffers = graph.runtime.supports_device_ptrs();
    let mut input_device = None;
    let mut fallback_device = None;
    for (position, input_plan) in plan.inputs.iter().enumerate() {
        let input = inputs.get_item(input_plan.argument_index)?;
        let observation = api.observe(py, &input)?;
        let borrow_device_buffer = supports_external_buffers && observation.is_cuda();
        let (tensor, metadata) = if borrow_device_buffer && observation.is_contiguous {
            (input, observation)
        } else if borrow_device_buffer {
            let tensor = api.make_contiguous(&input)?;
            let metadata = api.observe(py, &tensor)?;
            (tensor, metadata)
        } else if observation.is_cpu() && observation.is_contiguous {
            (input, observation)
        } else {
            let tensor = api.make_cpu_contiguous(&input)?;
            let metadata = api.observe(py, &tensor)?;
            (tensor, metadata)
        };
        if metadata.shape.len() != input_plan.expected_rank {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "input '{}' expected rank {}, got {}",
                input_plan.name,
                input_plan.expected_rank,
                metadata.shape.len()
            )));
        }
        plan.apply_input_shape(graph, position, &metadata.shape)?;
        if fallback_device.is_none() {
            fallback_device = Some(api.device(&tensor)?.unbind());
        }
        if metadata.is_cuda() && input_device.is_none() {
            input_device = Some(api.device(&tensor)?.unbind());
        }
        if metadata.dtype_code != input_plan.dtype_code {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Luminal compiled input '{}' expects {} but got {}. Convert at the call site.",
                input_plan.name,
                api.dtype(py, input_plan.dtype_code)?.str()?,
                api.dtype(py, metadata.dtype_code)?.str()?
            )));
        }
        let n_bytes = metadata.n_bytes();
        if borrow_device_buffer {
            let changed = state.input_bindings[position]
                .as_ref()
                .is_none_or(|previous| binding_changed(previous, &metadata));
            if changed {
                unsafe {
                    graph
                        .runtime
                        .set_device_ptr(input_plan.node, metadata.data_ptr, n_bytes)
                };
            }
            state.input_bindings[position] = Some(CachedBinding {
                device_type: metadata.device_type,
                device_id: metadata.device_id,
                pointer: metadata.data_ptr,
                n_bytes,
                dtype_code: metadata.dtype_code,
                _tensor: tensor.unbind(),
            });
        } else {
            state.input_bindings[position] = None;
            let bytes = copy_host_bytes(metadata.data_ptr, n_bytes, "input")?;
            let typed = TypedData::from_pytorch_bytes(bytes, metadata.dtype_code);
            graph
                .runtime
                .set_data_bytes(input_plan.node, typed.bytes, typed.dtype);
        }
    }
    let use_zero_copy = supports_external_buffers && input_device.is_some();
    let input_device = input_device
        .or(fallback_device)
        .map(|device| device.into_bound(py))
        .unwrap_or_else(|| api.cpu_device(py));

    let dynamic_output_shapes;
    let output_shapes = if plan.has_dynamic_dims {
        dynamic_output_shapes = graph.resolve_output_shapes_native()?;
        &dynamic_output_shapes
    } else {
        &plan.static_output_shapes
    };

    state.prepared_outputs.clear();
    state.direct_writebacks.fill(false);
    if use_zero_copy {
        state.external_outputs_registered = false;
        for (writeback_position, writeback) in plan.writebacks.iter().enumerate() {
            let output_plan = &writeback.output;
            let position = output_plan.position;
            let shape = &output_shapes[position];
            let dtype_code = output_plan.dtype_code;
            let output_node = output_plan.node;
            let target = inputs.get_item(plan.inputs[writeback.input_position].argument_index)?;
            let metadata = api.observe(py, &target)?;
            let expected_numel = shape.iter().product::<usize>();
            if metadata.is_cuda()
                && metadata.is_contiguous
                && metadata.dtype_code == dtype_code
                && metadata.numel == expected_numel
            {
                let changed = state.writeback_bindings[writeback_position]
                    .as_ref()
                    .is_none_or(|previous| binding_changed(previous, &metadata));
                if changed {
                    unsafe {
                        graph.runtime.set_output_device_ptr(
                            output_node,
                            metadata.data_ptr,
                            metadata.n_bytes(),
                        )
                    };
                }
                state.external_outputs_registered = true;
                let n_bytes = metadata.n_bytes();
                state.writeback_bindings[writeback_position] = Some(CachedBinding {
                    device_type: metadata.device_type,
                    device_id: metadata.device_id,
                    pointer: metadata.data_ptr,
                    n_bytes,
                    dtype_code: metadata.dtype_code,
                    _tensor: target.unbind(),
                });
                state.direct_writebacks[writeback_position] = true;
            } else if state.writeback_bindings[writeback_position]
                .take()
                .is_some()
            {
                graph.runtime.clear_output_device_ptr(output_node);
            }
        }

        for output_plan in &plan.returned_outputs {
            let shape = &output_shapes[output_plan.position];
            let dtype_code = output_plan.dtype_code;
            if is_zero_copy_output_dtype(dtype_code) {
                let output = api.empty(py, shape, dtype_code, &input_device)?;
                let metadata = api.observe(py, &output)?;
                state.prepared_outputs.push(Some(PreparedOutput {
                    tensor: output.unbind(),
                    data_ptr: metadata.data_ptr,
                    n_bytes: metadata.n_bytes(),
                }));
            } else {
                state.prepared_outputs.push(None);
            }
        }
    } else if supports_external_buffers && state.external_outputs_registered {
        for output_plan in &plan.returned_outputs {
            graph.runtime.clear_output_device_ptr(output_plan.node);
        }
        for writeback in &plan.writebacks {
            graph.runtime.clear_output_device_ptr(writeback.output.node);
        }
        state
            .writeback_bindings
            .iter_mut()
            .for_each(|slot| *slot = None);
        state.external_outputs_registered = false;
    }

    graph.execute_runtime();

    let mut outputs = Vec::with_capacity(plan.returned_outputs.len());
    state.gpu_output_copies.clear();
    for (writeback_position, writeback) in plan.writebacks.iter().enumerate() {
        let output_plan = &writeback.output;
        let position = output_plan.position;
        let shape = &output_shapes[position];
        let dtype_code = output_plan.dtype_code;
        let output_node = output_plan.node;
        if state.direct_writebacks[writeback_position] {
            continue;
        }
        let target = inputs.get_item(plan.inputs[writeback.input_position].argument_index)?;
        let metadata = api.observe(py, &target)?;
        let expected_numel = shape.iter().product::<usize>();
        if use_zero_copy
            && metadata.is_cuda()
            && metadata.is_contiguous
            && metadata.dtype_code == dtype_code
            && metadata.numel == expected_numel
        {
            state
                .gpu_output_copies
                .push((output_node, metadata.data_ptr, metadata.n_bytes()));
        } else {
            let value = read_output(
                graph,
                api,
                output_node,
                &output_plan.name,
                shape,
                dtype_code,
                &input_device,
            )?;
            target.call_method1("copy_", (value,))?;
        }
    }

    for (return_position, output_plan) in plan.returned_outputs.iter().enumerate() {
        let position = output_plan.position;
        let shape = &output_shapes[position];
        let dtype_code = output_plan.dtype_code;
        let output_node = output_plan.node;
        let output = if use_zero_copy && is_zero_copy_output_dtype(dtype_code) {
            let output = state.prepared_outputs[return_position]
                .take()
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "missing preallocated output at position {position}"
                    ))
                })?;
            state
                .gpu_output_copies
                .push((output_node, output.data_ptr, output.n_bytes));
            output.tensor
        } else {
            read_output(
                graph,
                api,
                output_node,
                &output_plan.name,
                shape,
                dtype_code,
                &input_device,
            )?
            .unbind()
        };
        outputs.push((output, output_plan.scalar));
    }

    if !state.gpu_output_copies.is_empty() {
        unsafe {
            graph
                .runtime
                .copy_outputs_to_device_ptrs(&state.gpu_output_copies)
        };
    }

    let outputs = outputs
        .into_iter()
        .map(|(output, scalar)| {
            if scalar {
                output.call_method0(py, "item")
            } else {
                Ok(output)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, outputs)?.into_any().unbind())
}

fn binding_changed(previous: &CachedBinding, current: &TensorObservation) -> bool {
    previous.pointer != current.data_ptr
        || previous.n_bytes != current.n_bytes()
        || previous.dtype_code != current.dtype_code
        || previous.device_type != current.device_type
        || previous.device_id != current.device_id
}

fn read_output<'py>(
    graph: &CompiledGraph,
    api: &TorchApi,
    node: NodeIndex,
    name: &str,
    shape: &[usize],
    dtype_code: u32,
    device: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = device.py();
    macro_rules! values {
        ($getter:ident) => {{
            let data = graph.runtime.$getter(node);
            if data.is_empty() {
                return empty_output(api, py, shape, dtype_code, device);
            }
            api.tensor_from_values(py, data, dtype_code)?
        }};
    }
    let tensor = match TorchDType::from_code(dtype_code) {
        Ok(TorchDType::Float) => values!(get_output_f32),
        Ok(TorchDType::Double) => values!(get_output_f64),
        Ok(TorchDType::Long) => values!(get_output_i64),
        Ok(TorchDType::Int) => values!(get_output_i32),
        Ok(TorchDType::Short) => values!(get_output_i16),
        Ok(TorchDType::Char) => values!(get_output_i8),
        Ok(TorchDType::Bool) => values!(get_output_bool),
        Ok(TorchDType::Byte) => {
            let data = graph.runtime.get_output_u8(node);
            if data.is_empty() {
                return empty_output(api, py, shape, dtype_code, device);
            }
            api.tensor_from_bytes(py, &data, dtype_code)?
        }
        Ok(TorchDType::Half) => {
            let data = graph.runtime.get_output_f16(node);
            if data.is_empty() {
                return empty_output(api, py, shape, dtype_code, device);
            }
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
            api.tensor_from_bytes(py, bytes, dtype_code)?
        }
        Ok(TorchDType::BFloat16) => {
            let data = graph.runtime.get_output_bf16(node);
            if data.is_empty() {
                return empty_output(api, py, shape, dtype_code, device);
            }
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
            api.tensor_from_bytes(py, bytes, dtype_code)?
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "Output '{name}' declared PT2 dtype code {dtype_code}, which is not supported"
            )));
        }
    };

    api.reshape_to_device(tensor, shape, device)
}

fn empty_output<'py>(
    api: &TorchApi,
    py: Python<'py>,
    shape: &[usize],
    dtype_code: u32,
    device: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if shape.iter().all(|&dimension| dimension != 0) {
        Ok(py.None().into_bound(py))
    } else {
        api.empty(py, shape, dtype_code, device)
    }
}
