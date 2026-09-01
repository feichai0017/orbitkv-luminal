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

use crate::layouts::CudaLayout;
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
    /// Train 3: assemble/search with the cuBLASLt marker vocabulary.
    /// OFF by default — the unconditional splice detonates the
    /// `view-arity-lock` tripwire on real model graphs (see
    /// [`crate::ops::cuda_registry_with_cublaslt`]); enabled through
    /// [`CudaRuntime::load_with_cublaslt`].
    cublaslt: bool,
    plan: Option<BufferIrGraph<CudaLayout>>,
    /// Host-staged input payloads by BufferLit id, H2D'd at execute.
    staged: FxHashMap<i64, TypedBuffer>,
    /// Host copies of each output slot's BACKING buffer plus its elected
    /// layout, filled by execute (D2H) — the escape-and-disclose fetch,
    /// keyed by slot index (an escaped slot's backing buffer is a minted
    /// allocation with no BufferLit, so slot order is the stable key).
    outputs_host: FxHashMap<usize, (TypedBuffer, luminal::bufferize::OutputBinding<CudaLayout>)>,
    input_buffers: FxHashMap<NodeIndex, i64>,
    /// Bound output tensor → its slot index (program slot order).
    output_index: FxHashMap<NodeIndex, usize>,
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

    /// [`CudaRuntime::load`] with the cuBLASLt marker vocabulary
    /// enabled: search assembles the marker's egg snippets and may
    /// elect the four host-call contracts. EXPLICIT OPT-IN (Train 3):
    /// on real model graphs the marker's canonicalization rewrites
    /// currently detonate the `view-arity-lock` coherence tripwire at
    /// saturation (measured on all seven Train-2 minis); callers get a
    /// loud saturation error, never a wrong plan. The 2D canonical
    /// matmul form searches and elects green.
    pub fn load_with_cublaslt(graph: &graph::Graph) -> Result<Self> {
        let mut rt = Self::load(graph)?;
        rt.cublaslt = true;
        Ok(rt)
    }

    /// The matcher vocabulary this instance assembles/searches with.
    fn matchers(&self) -> Vec<Box<dyn luminal::layout_ir::OpMatcher>> {
        if self.cublaslt {
            crate::ops::cuda_matchers_with_cublaslt()
        } else {
            crate::ops::cuda_matchers()
        }
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
    /// `reference_allow_list()` — three classes, all derived, never
    /// name-listed (M4 Phase 5 + Train 3):
    ///
    ///  * KERNEL-BEARING: matcher constructors whose label has a
    ///    codegen row — claimable because the device can execute them.
    ///  * PLAN-TRANSPARENT: constructors whose registered PROTOTYPE's
    ///    declared effects prove the planner folds them before any
    ///    kernel is needed (see [`crate::plan_transparent`]) —
    ///    claimable because nothing ever executes.
    ///  * HOST-CALL DISPATCHABLE (Train 3): constructors whose
    ///    prototype the executor dispatches as a host library call
    ///    (`cublasLtMatmul`) — claimable because the device runs them
    ///    without any NVRTC kernel (see
    ///    [`crate::ops::cublaslt::host_dispatchable`]).
    pub fn allow_list() -> Vec<&'static str> {
        Self::allow_list_over(&crate::ops::cuda_registry())
    }

    /// [`CudaRuntime::allow_list`] over the marker-enabled registry —
    /// the claim set a [`CudaRuntime::load_with_cublaslt`] search uses.
    pub fn allow_list_with_cublaslt() -> Vec<&'static str> {
        Self::allow_list_over(&crate::ops::cuda_registry_with_cublaslt())
    }

    fn allow_list_over(registry: &[crate::ops::RegisteredOp]) -> Vec<&'static str> {
        let labels: Vec<&'static str> =
            crate::kernels::cuda_kernels().iter().map(|k| k.label).collect();
        registry
            .iter()
            .filter(|entry| {
                let ctor = entry.matcher.egglog_constructor();
                let stripped = ctor.trim_start_matches("LayoutTensorOp");
                let kernel_bearing = labels.iter().any(|label| {
                    stripped == *label
                        || stripped.trim_end_matches("Generic") == *label
                });
                kernel_bearing
                    || crate::plan_transparent(entry.prototype.as_ref())
                    || crate::ops::cublaslt::host_dispatchable(entry.prototype.as_ref())
            })
            .map(|entry| entry.matcher.egglog_constructor())
            .collect()
    }

    /// Assemble, saturate, and search — with THIS backend's allow list.
    /// On saturation failure the labeled post-checks are re-run in
    /// isolation to name the door, mirroring the reference runtime.
    pub fn search(
        &mut self,
        input_data: &FxHashMap<NodeIndex, TypedBuffer>,
        options: &ImplementationSearchOptions,
    ) -> Result<SearchOutcome<CudaLayout>> {
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
            luminal::egglog_snippet::assembled_program_for(&self.matchers()),
            program.text
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
        if let Err(err) = egraph.parse_and_run_program(None, &full) {
            // Name the door: re-saturate without checks, then probe each
            // labeled check alone.
            let mut doors = Vec::new();
            let unchecked = format!(
                "{}\n\n{}\n{}\n{}",
                luminal::egglog_snippet::assembled_program_for(&self.matchers()),
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
            Some(if self.cublaslt {
                Self::allow_list_with_cublaslt()
            } else {
                Self::allow_list()
            }),
            self.matchers(),
            &crate::layouts::CudaLayoutRenderer,
            &mut luminal::implementation_search::StaticProfiler,
        )?;

        self.input_buffers = native
            .input_slots
            .iter()
            .map(|slot| (slot.tensor, slot.buffer))
            .collect();
        self.output_index = native
            .output_slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (slot.tensor, index))
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

    /// Read back a DENSE output tensor's f32 payload (already
    /// D2H'd by execute). Loud on a view-elected (escaped) output: its
    /// backing bytes are parent-laid-out — indistinguishable by length
    /// from row-major on a same-numel weld (e.g. a transpose) — so the
    /// legacy dense-shaped signature must never hand them over silently.
    /// Escaped outputs go through [`Self::fetch`] and interpret under
    /// [`Self::output_layout`] (the escape-and-disclose contract; the
    /// reader is [`crate::layouts::dense_f32`], this runtime evaluating
    /// its own layout vocabulary).
    pub fn get_f32(&self, tensor: NodeIndex) -> Result<&Vec<f32>> {
        // The row-major question is asked of the HELD LAYOUT's FUNCTION
        // — literally the codegen read path's own simplifier
        // ([`crate::kernels::reads_identity`]): element `k` of this
        // value is at flat index `k` of the backing, so the dense
        // `Vec<f32>` IS the value. Asked of the function, not the
        // spelling: a class the renderer happens to hand back as a
        // dense strided chain answers yes, exactly as its right-major
        // spelling would.
        let is_dense = |binding: &luminal::bufferize::OutputBinding<CudaLayout>| {
            match binding.layout.mirror.literal_extents() {
                Some(dims) => crate::kernels::reads_identity(&binding.layout, &dims),
                None => false,
            }
        };
        match self.fetch(tensor)? {
            (_, binding) if !is_dense(binding) => bail!(
                "get_f32 on a view-elected (escaped) output: the backing \
                 bytes are not row-major over the value's dims — use fetch() \
                 and interpret under the disclosed layout"
            ),
            (TypedBuffer::F32(values), _) => Ok(values),
            (other, _) => bail!("output is {}, not f32", other.type_name()),
        }
    }

    /// The universal escape-and-disclose fetch: the output slot's backing
    /// bytes plus its [`luminal::bufferize::OutputBinding`] (the elected
    /// layout).
    pub fn fetch(
        &self,
        tensor: NodeIndex,
    ) -> Result<(&TypedBuffer, &luminal::bufferize::OutputBinding<CudaLayout>)> {
        let index = self
            .output_index
            .get(&tensor)
            .ok_or_else(|| anyhow!("tensor has no output binding"))?;
        match self.outputs_host.get(index) {
            Some((data, binding)) => Ok((data, binding)),
            None => bail!("execute before fetch"),
        }
    }

    /// The slot's elected layout alone (see [`Self::fetch`]).
    pub fn output_layout(
        &self,
        tensor: NodeIndex,
    ) -> Result<&luminal::bufferize::OutputBinding<CudaLayout>> {
        Ok(self.fetch(tensor)?.1)
    }

    /// The searched plan, for inspection and tests.
    pub fn plan(&self) -> Option<&BufferIrGraph<CudaLayout>> {
        self.plan.as_ref()
    }
}
