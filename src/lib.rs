pub mod dtype;
pub mod egglog_utils;
pub mod frontend;
pub mod graph;
pub mod reference_binding;
pub mod mask_events;
pub mod shape;

// The logical-SSA layout compiler (the egglog_layout_trial graft, M0: vendored
// unwired). Egglog program assembly + registries live beside the ops; the
// fixture suite runs as core tests. See src/egglog/checkpoint_5/ for the core
// preamble and fixtures.
pub mod buffer_tensor_ir;
pub mod bufferize;
pub mod dps;
pub mod egglog_snippet;
pub mod extractor;
pub mod layout_ir;
pub mod logical_op;
pub mod implementation_search;
pub mod ssa_reference;
pub mod test_support;

#[cfg(test)]
pub mod tests;

pub mod prelude {
    pub use crate::dtype::DType;
    pub use crate::frontend::binary::F32Pow;
    pub use crate::frontend::*;
    pub use crate::graph::*;
    pub use crate::shape::*;
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
