pub mod dtype;
pub mod dyn_backend;
pub mod egglog_utils;
pub mod frontend;
pub mod graph;
pub mod hlir;
pub mod mask_events;
pub mod op;
pub mod shape;
pub mod visualization;

// The logical-SSA layout compiler (the egglog_layout_trial graft, M0: vendored
// unwired). Egglog program assembly + registries live beside the ops; the
// fixture suite runs as core tests. See src/egglog/checkpoint_5/ for the core
// preamble and fixtures.
pub mod buffer_tensor_ir;
pub mod bufferize;
pub mod dps;
pub mod egglog_snippet;
pub mod extractor;
pub mod hlir_to_logical;
pub mod layout_ir;
pub mod logical_op;
pub mod ssa_reference;
#[cfg(test)]
pub mod test_support;

#[cfg(test)]
pub mod tests;

pub mod prelude {
    pub use crate::dtype::DType;
    pub use crate::egglog_utils::SerializedEGraph;
    pub use crate::frontend::binary::F32Pow;
    pub use crate::frontend::*;
    pub use crate::graph::*;
    pub use crate::hlir::ReferenceRuntime;
    pub use crate::op::Runtime;
    pub use crate::shape::*;
    pub use crate::visualization::{display_graph, display_graph_to_file};
    pub use anyhow;
    pub use egglog;
    pub use egglog::ast as egglog_ast;
    pub use egraph_serialize::NodeId as ENodeId;
    pub use half::{bf16, f16};
    pub use petgraph;
    pub use petgraph::stable_graph::NodeIndex;
    pub use rustc_hash::{FxHashMap, FxHashSet};
    pub use tinyvec;
    pub use tracing;
}

pub use paste::paste;
