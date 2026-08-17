//! CUDA-lite's OWN op registry (ruling 2026-08-17: every runtime owns
//! its executable ops — structs, matchers, snippets, and codegen live
//! per-op here; the shared crate supplies only the IR traits and the
//! search machinery).
//!
//! One module per op, mirroring the reference layout: adding an op =
//! writing its module and adding its two lines here (mod + matcher).
//! FUNCTIONAL forms only in CL-1 — the runtime is out-of-place by
//! design, so the mutating/alias-safe family is deliberately absent
//! from this assembly (it arrives with the in-place ties in CL-4).

pub mod add;
pub mod cast;
pub mod constant;
pub mod div;
pub mod exp;
pub mod exp2;
pub mod gather;
pub mod index_map_apply_materialize;
pub mod iota;
pub mod less_than;
pub mod log2;
pub mod materialize_layout_copy;
pub mod modulo;
pub mod mul;
pub mod recip;
pub mod reduce_max;
pub mod reduce_sum;
pub mod scatter;
pub mod sin;
pub mod sqrt;
pub mod trunc_div;
pub mod trunc_rem;

use luminal::layout_ir::OpMatcher;

/// The matcher set this runtime assembles and extracts with.
pub fn cuda_matchers() -> Vec<Box<dyn OpMatcher>> {
    vec![
        Box::new(add::AddFunctionalMatcher),
        Box::new(materialize_layout_copy::MaterializeLayoutCopyMatcher),
        Box::new(sqrt::SqrtFunctionalMatcher),
        Box::new(exp::ExpFunctionalMatcher),
        Box::new(mul::MulFunctionalMatcher),
        Box::new(div::DivFunctionalMatcher),
        Box::new(trunc_div::TruncDivFunctionalMatcher),
        Box::new(trunc_rem::TruncRemFunctionalMatcher),
        Box::new(reduce_sum::ReduceSumMatcher),
        Box::new(reduce_max::ReduceMaxMatcher),
        Box::new(iota::IotaMatcher),
        Box::new(gather::GatherMatcher),
        Box::new(constant::ConstantMatcher),
        Box::new(scatter::ScatterFunctionalMatcher),
        Box::new(exp2::Exp2FunctionalMatcher),
        Box::new(log2::Log2FunctionalMatcher),
        Box::new(sin::SinFunctionalMatcher),
        Box::new(recip::RecipFunctionalMatcher),
        Box::new(modulo::ModFunctionalMatcher),
        Box::new(less_than::LessThanMatcher),
        Box::new(cast::CastMatcher),
        Box::new(index_map_apply_materialize::IndexMapApplyMaterializeMatcher),
    ]
}
