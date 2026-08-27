//! The CUDA-lite runtime: the reference ladder
//! (`load → bind_* → search → set_data → execute → get_*`) with the
//! search claiming only this backend's codegen inventory and execution
//! delegated to the `device` module.
//!
//! Everything up to `execute` is device-free and runs anywhere: load
//! accumulates the native program parts, bind_* appends bounds seeds,
//! search assembles + saturates + runs the genetic search with OUR
//! allow list (candidate profiling stays on the reference host
//! executor in CL-1 — a documented cost proxy). Only `execute`
//! requires the `device` feature and a CUDA device.

use anyhow::{anyhow, bail, Context, Result};
use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::BufferIrGraph;
use luminal::graph;
use luminal::implementation_search::{ImplementationSearchOptions, SearchOutcome};
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal::shape;

/// The accumulated pre-search program parts (the reference runtime's
/// NativeSpec is private; this is the same accumulation rebuilt from
/// the public `bound_parts` seam).
struct NativeParts {
    pre_schedule: String,
    input_slots: Vec<graph::InputSlot>,
    output_slots: Vec<graph::OutputSlot>,
    post_checks: String,
    labeled_checks: Vec<(String, String)>,
    binding_seeds: String,
}

#[derive(Default)]
pub struct CudaRuntime {
    native: Option<NativeParts>,
    plan: Option<BufferIrGraph>,
    /// Host-staged input payloads by BufferLit id, H2D'd at execute.
    staged: FxHashMap<i64, TypedBuffer>,
    /// Host copies of output buffers, filled by execute (D2H).
    outputs_host: FxHashMap<i64, TypedBuffer>,
    input_buffers: FxHashMap<NodeIndex, i64>,
    output_buffers: FxHashMap<NodeIndex, i64>,
}

impl CudaRuntime {
    /// Record the graph's native program. Saturation happens in
    /// [`CudaRuntime::search`].
    pub fn load(graph: &graph::Graph) -> Result<Self> {
        let (pre_schedule, input_slots, output_slots, post_checks, labeled_checks) =
            graph.logical.bound_parts(&crate::bindings::CudaBindings).map_err(|e| anyhow!(e))?;
        Ok(Self {
            native: Some(NativeParts {
                pre_schedule,
                input_slots,
                output_slots,
                post_checks,
                labeled_checks,
                binding_seeds: String::new(),
            }),
            ..Self::default()
        })
    }

    /// Seed interval bounds for a dynamic dimension (facts, never pins:
    /// `[n, n]` is how a caller pins).
    pub fn bind_dyn_range(
        &mut self,
        var: impl Into<shape::Symbol>,
        lower: u64,
        upper: u64,
    ) -> Result<()> {
        let native = self.native.as_mut().ok_or_else(|| anyhow!("load before bind"))?;
        let name = var.into();
        native.binding_seeds.push_str(&format!(
            "(set (lower-bound-of (IntVar \"{name}\")) (bigint {lower}))\n\
             (set (upper-bound-of (IntVar \"{name}\")) (bigint {upper}))\n"
        ));
        Ok(())
    }

    /// The ops this runtime claims: the CUDA analogue of
    /// `reference_allow_list()` — matcher constructors whose label has
    /// a codegen row.
    pub fn allow_list() -> Vec<&'static str> {
        let labels: Vec<&'static str> =
            crate::kernels::cuda_kernels().iter().map(|k| k.label).collect();
        crate::ops::cuda_matchers()
            .iter()
            .map(|m| m.egglog_constructor())
            .filter(|ctor| {
                let stripped = ctor.trim_start_matches("LayoutTensorOp");
                labels.iter().any(|label| {
                    stripped == *label
                        || stripped.trim_end_matches("Generic") == *label
                })
            })
            .collect()
    }

    /// Assemble, saturate, and search — with THIS backend's allow list.
    /// On saturation failure the labeled post-checks are re-run in
    /// isolation to name the door, mirroring the reference runtime.
    pub fn search(
        &mut self,
        input_data: &FxHashMap<NodeIndex, TypedBuffer>,
        options: &ImplementationSearchOptions,
    ) -> Result<SearchOutcome> {
        let native = self.native.as_ref().ok_or_else(|| anyhow!("load before search"))?;
        let program = graph::LogicalProgram {
            text: format!(
                "{}{}{}{}",
                native.pre_schedule,
                native.binding_seeds,
                crate::bindings::CudaBindings::SCHEDULE,
                native.post_checks
            ),
            input_slots: native.input_slots.clone(),
            output_slots: native.output_slots.clone(),
        };
        let full = format!(
            "{}\n\n{}",
            luminal::egglog_snippet::assembled_program_for(&crate::ops::cuda_matchers()),
            program.text
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
        if let Err(err) = egraph.parse_and_run_program(None, &full) {
            // Name the door: re-saturate without checks, then probe each
            // labeled check alone.
            let mut doors = Vec::new();
            let unchecked = format!(
                "{}\n\n{}\n{}\n{}",
                luminal::egglog_snippet::assembled_program_for(&crate::ops::cuda_matchers()),
                native.pre_schedule,
                native.binding_seeds,
                crate::bindings::CudaBindings::SCHEDULE
            );
            let mut probe = luminal::egglog_snippet::new_egraph();
            if probe.parse_and_run_program(None, &unchecked).is_ok() {
                for (label, text) in &native.labeled_checks {
                    if probe.parse_and_run_program(None, text).is_err() {
                        doors.push(label.clone());
                    }
                }
            }
            if doors.is_empty() {
                return Err(err).context("cuda-lite saturation failed");
            }
            bail!("shape contracts failed:\n  - {}", doors.join("\n  - "));
        }
        let serialized = egraph.serialize(luminal::prelude::egglog::SerializeConfig::default());

        // Own matchers, own allow list, and a profiler that never
        // touches another runtime: candidates rank by the heuristic
        // byte-move estimate (device profiling arrives with CL-3).
        let outcome = luminal::implementation_search::search_implementations_with_runtime(
            &serialized.egraph,
            &program,
            input_data,
            options,
            Some(Self::allow_list()),
            crate::ops::cuda_matchers(),
            &mut luminal::implementation_search::StaticProfiler,
        )?;

        self.input_buffers = native
            .input_slots
            .iter()
            .map(|slot| (slot.tensor, slot.buffer))
            .collect();
        self.output_buffers = native
            .output_slots
            .iter()
            .map(|slot| (slot.tensor, slot.buffer))
            .collect();
        self.plan = Some(outcome.best_plan.clone());
        Ok(outcome)
    }

    /// Stage input payload for a bound tensor (host side; H2D happens
    /// inside execute).
    pub fn set_data(&mut self, tensor: NodeIndex, data: impl Into<TypedBuffer>) {
        let Some(&buffer) = self.input_buffers.get(&tensor) else {
            panic!("set_data on a tensor with no input binding");
        };
        self.staged.insert(buffer, data.into());
    }

    /// Run the plan on the CUDA device. Requires the `device` feature
    /// and an available device; refuses loudly otherwise.
    pub fn execute(&mut self) -> Result<()> {
        let plan = self.plan.as_ref().ok_or_else(|| anyhow!("search before execute"))?;
        #[cfg(feature = "device")]
        {
            let outputs = crate::device::execute_plan(plan, &self.staged)?;
            self.outputs_host = outputs;
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = plan;
            bail!(
                "cuda-lite built without the `device` feature: plans can be \
                 searched and inspected but not executed on this host"
            )
        }
    }

    /// Read back an output tensor's f32 payload (already D2H'd by
    /// execute).
    pub fn get_f32(&self, tensor: NodeIndex) -> Result<&Vec<f32>> {
        let buffer = self
            .output_buffers
            .get(&tensor)
            .ok_or_else(|| anyhow!("tensor has no output binding"))?;
        match self.outputs_host.get(buffer) {
            Some(TypedBuffer::F32(values)) => Ok(values),
            Some(other) => bail!("output is {}, not f32", other.type_name()),
            None => bail!("execute before get_f32"),
        }
    }

    /// The searched plan, for inspection and tests.
    pub fn plan(&self) -> Option<&BufferIrGraph> {
        self.plan.as_ref()
    }
}
