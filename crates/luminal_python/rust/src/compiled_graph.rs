use luminal::{
    dyn_backend::{BackendCompileArgs, BackendFactory, DynBackend},
    prelude::*,
    shape::Expression,
    visualization::ToDot,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

use crate::typed_data::TypedData;
use crate::{
    bound_execution::BoundExecutable,
    torch_invocation::{
        DynamicDimPlan, ExecutionPlan, InputPlan, OutputPlan, StaticDimPlan, TorchInvocationState,
        WritebackPlan,
    },
};

/// Copy a CPU buffer into Rust-owned storage.
///
/// PyTorch legitimately reports `data_ptr() == 0` for an empty tensor, so a
/// null pointer is valid exactly when there are no bytes to read.  Keeping
/// this check in one helper gives inputs and weights the same boundary
/// contract and prevents `from_raw_parts` from ever receiving a null pointer.
pub(crate) fn copy_host_bytes(ptr: u64, n_bytes: usize, buffer_kind: &str) -> PyResult<Vec<u8>> {
    if n_bytes == 0 {
        return Ok(Vec::new());
    }
    if ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{buffer_kind} pointer is null for a non-empty buffer of {n_bytes} bytes"
        )));
    }

    // SAFETY: the caller guarantees that a non-null pointer addresses at
    // least `n_bytes` readable bytes for the duration of this call.  We copy
    // immediately, so the resulting Vec does not borrow the source buffer.
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, n_bytes).to_vec() })
}

/// Maps symbolic dimension parameter names (e.g. "seq_len") to their dim symbol.
pub type DimParamMap = HashMap<String, Symbol>;

#[derive(Clone)]
pub(crate) struct SingleVarDimSolver {
    variable: Symbol,
    expression: Expression,
    slope: i64,
    intercept: i64,
}

impl SingleVarDimSolver {
    /// Compile a safely invertible single-variable shape expression once.
    pub(crate) fn from_expression(expression: Expression) -> Option<Self> {
        use luminal::shape::Term;
        let terms = expression.terms.read();

        let mut variable = None;
        for term in terms.iter() {
            if let Term::Var(candidate) = term {
                match variable {
                    None => variable = Some(*candidate),
                    Some(existing) if existing == *candidate => {}
                    Some(_) => return None,
                }
            }
        }
        let variable = variable?;

        if terms.len() == 1 {
            return Some(Self {
                variable,
                expression,
                slope: 1,
                intercept: 0,
            });
        }

        // Probe two points to compile f(x) = slope*x + intercept. Calls still
        // round-trip through the expression so a non-affine form that happens
        // to be collinear at the probe points cannot produce a wrong binding.
        drop(terms);
        let f2 = expression.exec_single_var_checked(2)? as i64;
        let f3 = expression.exec_single_var_checked(3)? as i64;
        let slope = f3 - f2;
        if slope == 0 {
            return None;
        }
        Some(Self {
            variable,
            expression,
            slope,
            intercept: f2 - 2 * slope,
        })
    }

    pub(crate) fn solve(&self, observed: usize) -> Option<(Symbol, usize)> {
        let target = observed as i128 - self.intercept as i128;
        let slope = self.slope as i128;
        if target % slope != 0 {
            return None;
        }
        let value = usize::try_from(target / slope).ok()?;
        (self.expression.exec_single_var_checked(value)? == observed)
            .then_some((self.variable, value))
    }
}

/// Recover a single-variable dim's variable value from an observed runtime size.
pub(crate) fn solve_single_var_dim(expr: &Expression, dim_val: usize) -> Option<(Symbol, usize)> {
    SingleVarDimSolver::from_expression(*expr)?.solve(dim_val)
}

/// Convert luminal `DType` to a PT2 dtype code via `TorchDType`. Panics
/// for luminal-specific dtypes that have no PyTorch counterpart (`I4`,
/// `U4`, the F6 / F4 families, ...).
pub(crate) fn luminal_dtype_to_pt2_code(dtype: DType) -> u32 {
    crate::torch_dtype::TorchDType::try_from(dtype)
        .map(|t| t.code())
        .unwrap_or_else(|d| panic!("luminal_dtype_to_pt2_code: unsupported dtype {d:?}"))
}

/// Common intermediate result from translating a model graph.
pub struct GraphTranslation {
    pub graph: Graph,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    /// Output node identities in exactly the same order as `output_names`.
    /// Names are not unique when a functionalized mutation is also returned.
    pub output_ids: Vec<NodeIndex>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    /// Output dtypes as PT2 dtype codes (e.g. 5 = int64, 7 = float32).
    /// Stored as PT2 codes (rather than luminal `DType`) so we can preserve
    /// distinctions luminal collapses internally — notably int64 vs int32,
    /// both of which map to `DType::Int` in luminal but must be reported
    /// back to PyTorch with their original precision.
    pub output_dtypes: Vec<u32>,
    pub input_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
    /// (output position, user-input name) for outputs that write back into a
    /// user input's buffer (in-place state updates like HF StaticCache k/v).
    pub writeback_outputs: Vec<(usize, String)>,
}

/// Pre-loaded weight data from any model format (dtype-aware).
pub struct WeightData {
    /// (Input node label, typed data) for weights and constants.
    pub weights: Vec<(String, TypedData)>,
    /// label → element count for ALL Input nodes (for CUDA dummy data sizing).
    pub tensor_sizes: HashMap<String, usize>,
    /// label → (device_ptr, n_bytes) for zero-copy CUDA weight sharing.
    pub device_ptrs: HashMap<String, (u64, usize)>,
}

#[pyclass(unsendable)]
pub struct CompiledGraph {
    pub graph: Graph,
    pub runtime: Box<dyn DynBackend>,
    pub tensor_ids: HashMap<String, NodeIndex>,
    /// Cached label → NodeIndex map for O(1) lookups in set_weight_* methods.
    label_map: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_ids: Vec<NodeIndex>,
    pub output_shapes: Vec<Vec<usize>>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    /// Output dtypes as PT2 dtype codes (preserves int64 / int32 distinction
    /// that luminal collapses to `DType::Int` internally).
    pub output_dtypes: Vec<u32>,
    pub input_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
    /// See [`GraphTranslation::writeback_outputs`].
    pub writeback_outputs: Vec<(usize, String)>,
    pub(crate) torch_invocation: TorchInvocationState,
    pub(crate) is_bound: bool,
}

impl CompiledGraph {
    /// Compilation pipeline for PT2/FX graphs.
    ///
    /// Takes a `GraphTranslation` (produced by `translate_pt2`) and `WeightData`,
    /// builds the backend via the global registry, loads weights, and
    /// returns a ready-to-execute `CompiledGraph`.
    pub fn parse_graph(
        translation: GraphTranslation,
        weight_data: WeightData,
        factory: BackendFactory,
        search_iters: usize,
    ) -> Result<CompiledGraph, String> {
        let GraphTranslation {
            mut graph,
            tensor_ids,
            input_names,
            output_names,
            output_ids,
            output_shape_exprs,
            output_dtypes,
            input_shape_exprs,
            dim_param_map,
            writeback_outputs,
        } = translation;
        let WeightData {
            weights,
            tensor_sizes,
            device_ptrs,
        } = weight_data;

        // Build compile args from WeightData.
        let compile_args = BackendCompileArgs {
            search_iters,
            weights: weights
                .iter()
                .map(|(label, td)| (label.clone(), td.bytes.clone(), td.dtype))
                .collect(),
            tensor_sizes,
            device_ptrs,
        };

        // Create backend via the factory directly
        let rt =
            luminal::dyn_backend::compile_backend_from_factory(factory, &mut graph, compile_args)?;

        // Resolve concrete output shapes from expressions
        let output_shapes: Vec<Vec<usize>> = output_shape_exprs
            .iter()
            .map(|exprs| exprs.iter().map(|e| e.to_usize().unwrap_or(1)).collect())
            .collect();

        let label_map = luminal::dyn_backend::build_label_map(&graph);

        Ok(CompiledGraph {
            graph,
            runtime: rt,
            tensor_ids,
            label_map,
            input_names,
            output_names,
            output_ids,
            output_shapes,
            output_shape_exprs,
            output_dtypes,
            input_shape_exprs,
            dim_param_map,
            writeback_outputs,
            torch_invocation: TorchInvocationState::default(),
            is_bound: false,
        })
    }

    pub(crate) fn input_dtype_codes(&self) -> Vec<u32> {
        self.input_names
            .iter()
            .map(|name| {
                if let Some(&node_id) = self.tensor_ids.get(name)
                    && let Some(input) = (*self.graph.graph[node_id])
                        .as_any()
                        .downcast_ref::<luminal::hlir::Input>()
                {
                    return luminal_dtype_to_pt2_code(input.dtype);
                }
                7
            })
            .collect()
    }

    pub(crate) fn resolve_output_shapes_native(&self) -> PyResult<Vec<Vec<usize>>> {
        let dyn_map = &self.graph.dyn_map;
        self.output_shape_exprs
            .iter()
            .map(|shape_exprs| {
                shape_exprs
                    .iter()
                    .map(|expression| {
                        expression.exec(dyn_map).ok_or_else(|| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "Cannot resolve dimension expression {expression:?}. Set all dynamic dims first."
                            ))
                        })
                    })
                    .collect()
            })
            .collect()
    }

    pub(crate) fn auto_set_dims_native(&mut self, input_shapes: &[Vec<usize>]) {
        for (shape_exprs, shape) in self.input_shape_exprs.iter().zip(input_shapes) {
            for (dim_expr, &dim_value) in shape_exprs.iter().zip(shape) {
                if let Some((variable, value)) = solve_single_var_dim(dim_expr, dim_value) {
                    self.graph.set_dim(variable, value);
                }
            }
        }
    }

    pub(crate) fn execute_runtime(&mut self) {
        self.runtime.execute(&self.graph.dyn_map);
    }

    fn output_node_at(&self, position: usize) -> PyResult<NodeIndex> {
        self.output_ids.get(position).copied().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "output position {position} is out of range for {} outputs",
                self.output_ids.len()
            ))
        })
    }

    fn output_node_by_name(&self, name: &str) -> PyResult<NodeIndex> {
        let mut matches = self
            .output_names
            .iter()
            .zip(self.output_ids.iter().copied())
            .filter_map(|(candidate, node)| (candidate == name).then_some(node));
        let Some(node) = matches.next() else {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Unknown output tensor: {name}"
            )));
        };
        if matches.next().is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Output tensor name '{name}' is ambiguous; use the positional output API"
            )));
        }
        Ok(node)
    }
}

#[pymethods]
impl CompiledGraph {
    /// Get the list of input tensor names.
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Get the PT2 dtype codes for all inputs (in order of input_names).
    #[getter]
    fn input_dtypes(&self) -> Vec<u32> {
        self.input_dtype_codes()
    }

    /// Get the list of output tensor names.
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.output_names.clone()
    }

    /// Get the output shapes.
    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.output_shapes.clone()
    }

    /// Get all tensor names in the graph.
    #[getter]
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_ids.keys().cloned().collect()
    }

    /// (output position, input name) pairs for outputs that write back into a
    /// user input's buffer (in-place state updates like HF StaticCache k/v).
    #[getter]
    fn writeback_outputs(&self) -> Vec<(usize, String)> {
        self.writeback_outputs.clone()
    }

    /// Get the name of the active backend.
    #[getter]
    fn backend(&self) -> &str {
        self.runtime.name()
    }

    /// The device type this backend operates on (e.g. "cpu", "cuda").
    #[getter]
    fn device_type(&self) -> &str {
        self.runtime.device_type()
    }

    /// Whether the active backend supports device pointer operations (zero-copy GPU I/O).
    #[getter]
    fn supports_device_ptrs(&self) -> bool {
        self.runtime.supports_device_ptrs()
    }

    /// Whether this graph has dynamic (symbolic) dimensions.
    #[getter]
    fn has_dynamic_dims(&self) -> bool {
        !self.dim_param_map.is_empty()
    }

    /// Get the dynamic dimension parameter names (e.g. ["seq_len"]).
    #[getter]
    fn dim_params(&self) -> Vec<String> {
        self.dim_param_map.keys().cloned().collect()
    }

    /// Set a dynamic dimension value by its param name (e.g. "seq_len").
    fn set_dim(&mut self, param_name: &str, value: usize) -> PyResult<()> {
        let ch = self.dim_param_map.get(param_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown dim param '{}'. Available: {:?}",
                param_name,
                self.dim_param_map.keys().collect::<Vec<_>>()
            ))
        })?;
        self.graph.set_dim(*ch, value);
        Ok(())
    }

    /// Auto-detect and set dynamic dimensions from input tensor shapes.
    ///
    /// For each user input we walk the symbolic shape expressions side-by-side
    /// with the concrete sizes Dynamo handed us at runtime and try to recover
    /// each unbound variable's value. Two cases are handled:
    ///
    ///   * Bare-variable dim (`s`): set directly from the size.
    ///   * Single-variable affine dim (`a*s + b`): solve `s = (size - b)/a`
    ///     by sampling the expression at two probe points to extract the
    ///     slope, recovering the intercept, and verifying that plugging the
    ///     recovered value back through `exec_single_var_checked` reproduces
    ///     the observed size. The verification step rejects everything
    ///     non-affine (`s*s`, `min(s, 8)`, etc.) without committing a wrong
    ///     guess to `dyn_map`.
    ///
    /// Multi-variable dims are skipped here; another input's shape — or an
    /// explicit `set_dim` call — is expected to bind those.
    fn auto_set_dims_from_input_shapes(&mut self, input_shapes: Vec<Vec<usize>>) {
        self.auto_set_dims_native(&input_shapes);
    }

    /// Resolve output shapes using current dynamic dimension values.
    /// Returns concrete shapes after substituting all symbolic dims.
    fn resolve_output_shapes(&self) -> PyResult<Vec<Vec<usize>>> {
        self.resolve_output_shapes_native()
    }

    /// Configure the immutable positional plan used by native invocation.
    fn configure_invocation(
        &mut self,
        input_names: Vec<String>,
        user_indices: Option<Vec<usize>>,
        scalar_output_positions: Vec<usize>,
    ) -> PyResult<()> {
        if input_names.len() != self.input_names.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invocation plan has {} input names for {} compiled inputs",
                input_names.len(),
                self.input_names.len()
            )));
        }
        let exact_argument_count = user_indices.is_none().then_some(input_names.len());
        let argument_indices = match user_indices {
            Some(indices) => {
                if indices.len() != input_names.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "invocation plan has {} user indices for {} compiled inputs",
                        indices.len(),
                        input_names.len()
                    )));
                }
                indices
            }
            None => (0..input_names.len()).collect(),
        };
        let input_plans = input_names
            .into_iter()
            .zip(argument_indices)
            .zip(self.input_dtype_codes())
            .zip(&self.input_shape_exprs)
            .map(|(((name, argument_index), dtype_code), shape)| {
                let mut dynamic_dims = Vec::new();
                let mut static_dims = Vec::new();
                for (dimension_position, &expression) in shape.iter().enumerate() {
                    if let Some(dynamic) =
                        DynamicDimPlan::from_expression(dimension_position, expression)
                    {
                        dynamic_dims.push(dynamic);
                    } else if let Some(expected) = expression.to_usize() {
                        static_dims.push(StaticDimPlan {
                            dimension_position,
                            expected,
                        });
                    }
                }
                self.tensor_ids
                    .get(&name)
                    .copied()
                    .map(|node| InputPlan {
                        name: name.clone(),
                        argument_index,
                        node,
                        dtype_code,
                        expected_rank: shape.len(),
                        static_dims,
                        dynamic_dims,
                    })
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Unknown input tensor: {name}"
                        ))
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;

        if self.output_names.len() != self.output_ids.len()
            || self.output_names.len() != self.output_dtypes.len()
            || self.output_names.len() != self.output_shape_exprs.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compiled output metadata lengths disagree: names={}, nodes={}, dtypes={}, shapes={}",
                self.output_names.len(),
                self.output_ids.len(),
                self.output_dtypes.len(),
                self.output_shape_exprs.len()
            )));
        }
        let input_positions: HashMap<&str, usize> = input_plans
            .iter()
            .enumerate()
            .map(|(position, input)| (input.name.as_str(), position))
            .collect();
        let mut writeback_inputs = vec![None; self.output_names.len()];
        for (output_position, input_name) in &self.writeback_outputs {
            let destination = writeback_inputs.get_mut(*output_position).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "writeback output position {output_position} is out of range"
                ))
            })?;
            let input_position = input_positions
                .get(input_name.as_str())
                .copied()
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "writeback output {output_position} refers to unknown input '{input_name}'"
                    ))
                })?;
            *destination = Some(input_position);
        }
        let mut scalar_outputs = vec![false; self.output_names.len()];
        for position in scalar_output_positions {
            let scalar = scalar_outputs.get_mut(position).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "scalar output position {position} is out of range"
                ))
            })?;
            *scalar = true;
        }
        let mut returned_outputs = Vec::new();
        let mut writebacks = Vec::new();
        for position in 0..self.output_names.len() {
            let output = OutputPlan {
                position,
                name: self.output_names[position].clone(),
                node: self.output_ids[position],
                dtype_code: self.output_dtypes[position],
                scalar: scalar_outputs[position],
            };
            if let Some(input_position) = writeback_inputs[position] {
                writebacks.push(WritebackPlan {
                    output,
                    input_position,
                });
            } else {
                returned_outputs.push(output);
            }
        }
        self.torch_invocation.configure(ExecutionPlan {
            inputs: input_plans,
            exact_argument_count,
            returned_outputs,
            writebacks,
            has_dynamic_dims: !self.dim_param_map.is_empty(),
            static_output_shapes: self.output_shapes.clone(),
        });
        Ok(())
    }

    /// Fully generic PyTorch invocation. Every external tensor binding is
    /// observed on every call; only changed runtime bindings are reinstalled.
    fn invoke(
        &mut self,
        py: Python<'_>,
        inputs: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        if self.is_bound {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this CompiledGraph has been consumed by bound execution",
            ));
        }
        crate::torch_invocation::invoke(self, py, inputs)
    }

    /// Bind stable CUDA resources once and return an exclusive executable.
    fn bind(
        slf: Py<Self>,
        py: Python<'_>,
        inputs: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<BoundExecutable> {
        crate::bound_execution::bind(slf, py, inputs)
    }

    /// Set input tensor data by name (f32, for backward compatibility).
    fn set_input(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        self.runtime.set_data_f32(*node_id, data);
        Ok(())
    }

    /// Set input tensor data from a CPU host memory pointer (dtype-aware).
    /// The pointer must point to contiguous data. It may be null only when
    /// `n_bytes == 0`; otherwise it must address at least `n_bytes` readable bytes.
    /// `dtype_code` uses PT2 numbering (7=f32, 6=f16, 13=bf16, etc.).
    /// Preserves the source dtype and width in luminal's typed input buffer.
    fn set_input_from_ptr(
        &mut self,
        name: &str,
        ptr: u64,
        n_bytes: usize,
        dtype_code: u32,
    ) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        let raw_bytes = copy_host_bytes(ptr, n_bytes, "input")?;
        let typed = TypedData::from_pytorch_bytes(raw_bytes, dtype_code);
        self.runtime
            .set_data_bytes(*node_id, typed.bytes, typed.dtype);
        Ok(())
    }

    /// Set input from a device pointer. Zero-copy on device.
    /// The pointer must be a valid device allocation with at least n_bytes bytes.
    /// Requires a GPU backend (e.g. CUDA).
    fn set_input_device_ptr(
        &mut self,
        name: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_input_device_ptr requires a GPU backend",
            ));
        }
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        unsafe { self.runtime.set_device_ptr(*node_id, device_ptr, n_bytes) };
        Ok(())
    }

    /// Register a weight from a device pointer (e.g. "fc1.weight"). Zero-copy on device.
    /// Requires a GPU backend.
    fn set_weight_device_ptr(
        &mut self,
        label: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_weight_device_ptr requires a GPU backend",
            ));
        }
        let &node_id = self.label_map.get(label).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("No Input node with label: {}", label))
        })?;
        unsafe { self.runtime.set_device_ptr(node_id, device_ptr, n_bytes) };
        Ok(())
    }

    /// Register an external device pointer for an output tensor (zero-copy output).
    /// Call before run() — the runtime will write kernel results directly into this buffer.
    /// For aliased outputs (in-place ops), falls back to DtoD copy; check output_is_zero_copy() after run().
    /// Requires a GPU backend.
    fn set_output_device_ptr(
        &mut self,
        name: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_output_device_ptr requires a GPU backend",
            ));
        }
        let node_id = self.output_node_by_name(name)?;
        unsafe {
            self.runtime
                .set_output_device_ptr(node_id, device_ptr, n_bytes)
        };
        Ok(())
    }

    /// Positional variant of `set_output_device_ptr`; unlike output names,
    /// output positions are unique for functionalized mutation outputs.
    fn set_output_device_ptr_at(
        &mut self,
        position: usize,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_output_device_ptr_at requires a GPU backend",
            ));
        }
        let node_id = self.output_node_at(position)?;
        unsafe {
            self.runtime
                .set_output_device_ptr(node_id, device_ptr, n_bytes)
        };
        Ok(())
    }

    fn clear_output_device_ptr_at(&mut self, position: usize) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "clear_output_device_ptr_at requires a GPU backend",
            ));
        }
        let node_id = self.output_node_at(position)?;
        self.runtime.clear_output_device_ptr(node_id);
        Ok(())
    }

    /// Check whether an output tensor was zero-copied (written directly to the registered pointer).
    /// Returns false for aliased outputs that need a fallback DtoD copy, or if no GPU backend.
    /// Must be called after run().
    fn output_is_zero_copy(&self, name: &str) -> PyResult<bool> {
        let node_id = self.output_node_by_name(name)?;
        Ok(self.runtime.output_is_zero_copy(node_id))
    }

    fn output_is_zero_copy_at(&self, position: usize) -> PyResult<bool> {
        let node_id = self.output_node_at(position)?;
        Ok(self.runtime.output_is_zero_copy(node_id))
    }

    /// Register a weight tensor from a CPU host pointer, matching by Input node label (dtype-aware).
    /// `ptr` may be null only when `n_bytes == 0`; otherwise it must address at
    /// least `n_bytes` readable bytes. `dtype_code` uses PT2 numbering
    /// (7=f32, 6=f16, 13=bf16, etc.).
    fn set_weight_from_ptr(
        &mut self,
        label: &str,
        ptr: u64,
        n_bytes: usize,
        dtype_code: u32,
    ) -> PyResult<()> {
        let &node_id = self.label_map.get(label).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("No Input node with label: {}", label))
        })?;
        let bytes = copy_host_bytes(ptr, n_bytes, "weight")?;
        let typed = TypedData::from_pytorch_bytes(bytes, dtype_code);
        self.runtime
            .set_data_bytes(node_id, typed.bytes, typed.dtype);
        Ok(())
    }

    /// Execute the graph.
    fn run(&mut self) {
        self.execute_runtime();
    }

    /// Return the HLIR graph as a DOT string for visualization.
    fn to_dot(&self) -> PyResult<String> {
        self.graph.graph.to_dot().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("DOT generation failed: {e}"))
        })
    }

    /// Get the PT2 dtype codes for all outputs (in order).
    #[getter]
    fn output_dtypes(&self) -> Vec<u32> {
        self.output_dtypes.clone()
    }

    /// Get output tensor data by name as f32 (copies to host).
    fn get_output(&self, name: &str) -> PyResult<Vec<f32>> {
        Ok(self.runtime.get_output_f32(self.output_node_by_name(name)?))
    }

    fn get_output_at(&self, position: usize) -> PyResult<Vec<f32>> {
        Ok(self.runtime.get_output_f32(self.output_node_at(position)?))
    }

    /// Get output tensor data by name as i32 (copies to host).
    fn get_output_i32(&self, name: &str) -> PyResult<Vec<i32>> {
        Ok(self.runtime.get_output_i32(self.output_node_by_name(name)?))
    }

    fn get_output_i32_at(&self, position: usize) -> PyResult<Vec<i32>> {
        Ok(self.runtime.get_output_i32(self.output_node_at(position)?))
    }

    /// Read an output as f16 (returned as raw little-endian bytes —
    /// Python has no native f16, so the caller bit-casts via
    /// `torch.frombuffer(..., dtype=torch.float16)`). Strict: the
    /// producer node must already be `DType::F16`; no widening at
    /// the read boundary.
    fn get_output_f16<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.runtime.get_output_f16(self.output_node_by_name(name)?);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        Ok(PyBytes::new(py, bytes))
    }

    fn get_output_f16_at<'py>(
        &self,
        py: Python<'py>,
        position: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.runtime.get_output_f16(self.output_node_at(position)?);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        Ok(PyBytes::new(py, bytes))
    }

    /// Read an output as bf16 (returned as raw little-endian bytes —
    /// caller bit-casts via `torch.frombuffer(..., dtype=torch.
    /// bfloat16)`). Strict: the producer node must already be
    /// `DType::Bf16`; no widening at the read boundary.
    fn get_output_bf16<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let data = self
            .runtime
            .get_output_bf16(self.output_node_by_name(name)?);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        Ok(PyBytes::new(py, bytes))
    }

    fn get_output_bf16_at<'py>(
        &self,
        py: Python<'py>,
        position: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.runtime.get_output_bf16(self.output_node_at(position)?);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        Ok(PyBytes::new(py, bytes))
    }

    /// Read an output as i64. Strict: the producer node must already
    /// be `DType::I64`; no widening at the read boundary.
    fn get_output_i64(&self, name: &str) -> PyResult<Vec<i64>> {
        Ok(self.runtime.get_output_i64(self.output_node_by_name(name)?))
    }

    fn get_output_i64_at(&self, position: usize) -> PyResult<Vec<i64>> {
        Ok(self.runtime.get_output_i64(self.output_node_at(position)?))
    }

    /// Read an output as i8 without widening.
    fn get_output_i8(&self, name: &str) -> PyResult<Vec<i8>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_i8(*node_id))
    }

    /// Read an output as u8 without widening.
    fn get_output_u8(&self, name: &str) -> PyResult<Vec<u8>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_u8(*node_id))
    }

    /// Read an output as i16 without widening.
    fn get_output_i16(&self, name: &str) -> PyResult<Vec<i16>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_i16(*node_id))
    }

    /// Read an output as f64. Strict: the producer node must already
    /// be `DType::F64`; no widening at the read boundary.
    fn get_output_f64(&self, name: &str) -> PyResult<Vec<f64>> {
        Ok(self.runtime.get_output_f64(self.output_node_by_name(name)?))
    }

    fn get_output_f64_at(&self, position: usize) -> PyResult<Vec<f64>> {
        Ok(self.runtime.get_output_f64(self.output_node_at(position)?))
    }

    /// Get output tensor data by name as bool (copies to host).
    fn get_output_bool(&self, name: &str) -> PyResult<Vec<bool>> {
        Ok(self
            .runtime
            .get_output_bool(self.output_node_by_name(name)?))
    }

    fn get_output_bool_at(&self, position: usize) -> PyResult<Vec<bool>> {
        Ok(self.runtime.get_output_bool(self.output_node_at(position)?))
    }

    /// Copy output tensor data directly to a device pointer (DtoD).
    /// Avoids the DtoH + HtoD round-trip of get_output() + .to(device).
    /// Requires a GPU backend.
    fn copy_output_to_device_ptr(&self, name: &str, dest_ptr: u64, n_bytes: usize) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "copy_output_to_device_ptr requires a GPU backend",
            ));
        }
        let node_id = self.output_node_by_name(name)?;
        unsafe {
            self.runtime
                .copy_output_to_device_ptr(node_id, dest_ptr, n_bytes)
        };
        Ok(())
    }

    fn copy_output_to_device_ptr_at(
        &self,
        position: usize,
        dest_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "copy_output_to_device_ptr_at requires a GPU backend",
            ));
        }
        let node_id = self.output_node_at(position)?;
        unsafe {
            self.runtime
                .copy_output_to_device_ptr(node_id, dest_ptr, n_bytes)
        };
        Ok(())
    }

    /// Copy several outputs directly to CUDA device pointers, synchronizing
    /// only after the entire batch has been enqueued.
    fn copy_outputs_to_device_ptrs(&self, copies: Vec<(String, u64, usize)>) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "copy_outputs_to_device_ptrs requires a GPU backend",
            ));
        }
        let resolved = copies
            .into_iter()
            .map(|(name, dest_ptr, n_bytes)| {
                self.output_node_by_name(&name)
                    .map(|node_id| (node_id, dest_ptr, n_bytes))
            })
            .collect::<PyResult<Vec<_>>>()?;
        unsafe { self.runtime.copy_outputs_to_device_ptrs(&resolved) };
        Ok(())
    }

    fn copy_outputs_to_device_ptrs_at(&self, copies: Vec<(usize, u64, usize)>) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "copy_outputs_to_device_ptrs_at requires a GPU backend",
            ));
        }
        let resolved = copies
            .into_iter()
            .map(|(position, dest_ptr, n_bytes)| {
                self.output_node_at(position)
                    .map(|node_id| (node_id, dest_ptr, n_bytes))
            })
            .collect::<PyResult<Vec<_>>>()?;
        unsafe { self.runtime.copy_outputs_to_device_ptrs(&resolved) };
        Ok(())
    }
}
