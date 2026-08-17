//! Explicit stable-resource execution for inference serving.

use pyo3::{
    prelude::*,
    types::{PyAny, PyTuple, PyTupleMethods},
};

use crate::{compiled_graph::CompiledGraph, tensor_bridge::TorchApi};

struct BoundOutput {
    tensor: Py<PyAny>,
}

struct BoundDestination {
    node: luminal::prelude::NodeIndex,
    data_ptr: u64,
    n_bytes: usize,
    always_copy: bool,
}

#[pyclass(unsendable)]
pub struct BoundExecutable {
    graph: Py<CompiledGraph>,
    // These references are the storage-lifetime contract. The runtime borrows
    // their pointers until this executable is dropped.
    input_refs: Vec<Py<PyAny>>,
    outputs: Vec<BoundOutput>,
    destinations: Vec<BoundDestination>,
    // The selected runtime graph is stable after its first execution, so its
    // non-zero-copy outputs only need to be discovered once.
    output_copies: Option<Vec<(luminal::prelude::NodeIndex, u64, usize)>>,
}

pub(crate) fn bind(
    graph_object: Py<CompiledGraph>,
    py: Python<'_>,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<BoundExecutable> {
    let api = TorchApi::new(py)?;
    let mut graph = graph_object.borrow_mut(py);
    if graph.is_bound {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "this CompiledGraph is already bound",
        ));
    }
    if !graph.runtime.supports_device_ptrs() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "bound execution currently requires a CUDA backend",
        ));
    }
    let plan = graph.torch_invocation.plan.clone().ok_or_else(|| {
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

    let mut user_inputs = Vec::with_capacity(plan.inputs.len());
    let mut input_refs = Vec::with_capacity(plan.inputs.len());
    let mut device = None;
    for (position, input_plan) in plan.inputs.iter().enumerate() {
        let input = inputs.get_item(input_plan.argument_index)?;
        let metadata = api.observe(py, &input)?;
        if !metadata.is_cuda() || !metadata.is_contiguous {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bound input {position} ('{}') must be a contiguous CUDA tensor",
                input_plan.name
            )));
        }
        if metadata.shape.len() != input_plan.expected_rank {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bound input '{}' expected rank {}, got {}",
                input_plan.name,
                input_plan.expected_rank,
                metadata.shape.len()
            )));
        }
        if metadata.dtype_code != input_plan.dtype_code {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "bound input '{}' has {}, expected {}",
                input_plan.name,
                api.dtype(py, metadata.dtype_code)?.str()?,
                api.dtype(py, input_plan.dtype_code)?.str()?
            )));
        }
        if device.is_none() {
            device = Some(api.device(&input)?.unbind());
        }
        plan.apply_input_shape(&mut graph, position, &metadata.shape)?;
        unsafe {
            graph
                .runtime
                .set_device_ptr(input_plan.node, metadata.data_ptr, metadata.n_bytes())
        };
        input_refs.push(input.clone().unbind());
        user_inputs.push(input);
    }
    let output_shapes = if plan.has_dynamic_dims {
        graph.resolve_output_shapes_native()?
    } else {
        plan.static_output_shapes.clone()
    };
    let device = device
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "bound execution requires at least one CUDA tensor input",
            )
        })?
        .into_bound(py);

    let mut destinations = Vec::with_capacity(plan.writebacks.len() + plan.returned_outputs.len());
    for writeback in &plan.writebacks {
        let output_plan = &writeback.output;
        let position = output_plan.position;
        let shape = &output_shapes[position];
        let dtype_code = output_plan.dtype_code;
        let output_node = output_plan.node;
        let target = &user_inputs[writeback.input_position];
        let metadata = api.observe(py, target)?;
        let expected_numel = shape.iter().product::<usize>();
        if !metadata.is_cuda()
            || !metadata.is_contiguous
            || metadata.dtype_code != dtype_code
            || metadata.numel != expected_numel
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "writeback output {position} cannot bind directly to input {}",
                writeback.input_position
            )));
        }
        unsafe {
            graph
                .runtime
                .set_output_device_ptr(output_node, metadata.data_ptr, metadata.n_bytes())
        };
        destinations.push(BoundDestination {
            node: output_node,
            data_ptr: metadata.data_ptr,
            n_bytes: metadata.n_bytes(),
            always_copy: false,
        });
    }

    let mut outputs = Vec::with_capacity(plan.returned_outputs.len());
    for output_plan in &plan.returned_outputs {
        let position = output_plan.position;
        let shape = &output_shapes[position];
        let dtype_code = output_plan.dtype_code;
        let output_node = output_plan.node;
        let output = api.empty(py, shape, dtype_code, &device)?;
        let metadata = api.observe(py, &output)?;
        // Generic invocation may have left a prior caller-owned destination
        // registered. Returned bound outputs currently execute into runtime
        // storage and are copied into their retained tensors after replay;
        // writebacks above remain directly registered.
        if !plan
            .writebacks
            .iter()
            .any(|writeback| writeback.output.node == output_node)
        {
            graph.runtime.clear_output_device_ptr(output_node);
        }
        destinations.push(BoundDestination {
            node: output_node,
            data_ptr: metadata.data_ptr,
            n_bytes: metadata.n_bytes(),
            always_copy: true,
        });
        outputs.push(BoundOutput {
            tensor: output.unbind(),
        });
    }

    graph.is_bound = true;
    drop(graph);
    Ok(BoundExecutable {
        graph: graph_object,
        input_refs,
        outputs,
        destinations,
        output_copies: None,
    })
}

#[pymethods]
impl BoundExecutable {
    /// Replay using the CUDA resources validated and retained by `bind()`.
    ///
    /// Outputs always remain tensors, including rank-zero tensors that the
    /// generic torch.compile path converts to Python scalars with `.item()`.
    fn replay(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut graph = self.graph.borrow_mut(py);
        graph.execute_runtime();

        let output_copies = self.output_copies.get_or_insert_with(|| {
            self.destinations
                .iter()
                .filter(|output| {
                    output.always_copy || !graph.runtime.output_is_zero_copy(output.node)
                })
                .map(|output| (output.node, output.data_ptr, output.n_bytes))
                .collect()
        });
        if !output_copies.is_empty() {
            unsafe {
                graph.runtime.copy_outputs_to_device_ptrs(output_copies);
            }
        }
        drop(graph);
        Ok(PyTuple::new(
            py,
            self.outputs
                .iter()
                .map(|output| output.tensor.clone_ref(py)),
        )?
        .into_any()
        .unbind())
    }

    /// Compatibility alias for callers using the initial bound API.
    fn run(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.replay(py)
    }

    fn __call__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.replay(py)
    }

    #[getter]
    fn inputs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(
            PyTuple::new(py, self.input_refs.iter().map(|value| value.clone_ref(py)))?
                .into_any()
                .unbind(),
        )
    }

    #[getter]
    fn outputs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(
            py,
            self.outputs
                .iter()
                .map(|output| output.tensor.clone_ref(py)),
        )?
        .into_any()
        .unbind())
    }
}
