//! The reference-flavored implementation search: core's runtime-owned
//! search entry (`search_implementations_with_runtime`) defaulted to THIS
//! crate's registry (matchers + allow list) and host profiler — the
//! historical `search_implementations(_with_ops)` surface, relocated in
//! Step B (ruling 2026-08-17).

use std::time::Instant;

use anyhow::Result;
use rustc_hash::FxHashMap;

use luminal::graph::LogicalProgram;
use luminal::implementation_search::{
    search_implementations_with_runtime, ImplementationSearchOptions, PlanProfiler, SearchOutcome,
};

use crate::runtime::{reference_allow_list, ReferenceRuntime};

/// The historical profiler: execute on the reference host runtime.
#[derive(Default)]
pub struct ReferenceProfiler;

impl PlanProfiler for ReferenceProfiler {
    fn profile(
        &mut self,
        plan: &luminal::bufferize::BufferIrGraph,
        input_data: &FxHashMap<i64, luminal::buffer_tensor_ir::TypedBuffer>,
        trials: usize,
        _heuristic_cost: u64,
    ) -> Result<u128> {
        let mut runtime = ReferenceRuntime::default();
        runtime.load_plan(plan.clone());
        for (id, data) in input_data {
            runtime.set_data_buffer(*id, data.clone());
        }
        runtime.execute()?; // warmup + validity
        let mut best_nanos = u128::MAX;
        for _ in 0..trials.max(1) {
            let start = Instant::now();
            runtime.execute()?;
            best_nanos = best_nanos.min(start.elapsed().as_nanos());
        }
        Ok(best_nanos)
    }
}

/// [`search_implementations`] with the runtime's ALLOWABLE-OPS inventory
/// made explicit (M3 Step 2: per-runtime, unstandardized). `None` keeps
/// the reference runtime's own allow list — the historical default.
pub fn search_implementations_with_ops(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, luminal::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
    allow_override: Option<Vec<&'static str>>,
) -> Result<SearchOutcome> {
    let allow = allow_override.or_else(|| Some(reference_allow_list()));
    search_implementations_with_runtime(
        egraph,
        program,
        input_data,
        options,
        allow,
        crate::ops::built_in_matchers(),
        &mut ReferenceProfiler,
    )
}

/// Search the saturated e-graph for the fastest executable plan on the
/// reference runtime, profiling with the given caller data. Deterministic
/// for a fixed seed.
pub fn search_implementations(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, luminal::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
) -> Result<SearchOutcome> {
    search_implementations_with_ops(egraph, program, input_data, options, None)
}
