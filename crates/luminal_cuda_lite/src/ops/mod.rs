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
pub mod index_map_apply_view;
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

use luminal::layout_ir::{LayoutIrOp, OpMatcher};

/// One registered op: the matcher plus a PROTOTYPE instance of the op
/// it extracts. The prototype exists so claim derivation can read the
/// op's DECLARED EFFECTS (memory-effect predicates, alias contract,
/// DPS story) without an e-graph in hand — the allow list's
/// plan-transparent class is derived from these trait answers, never
/// from a name list. Prototype metadata fields (entries, ranks, axes)
/// take their cheapest value: the effect predicates of every op here
/// are metadata-independent.
pub struct RegisteredOp {
    pub matcher: Box<dyn OpMatcher>,
    pub prototype: Box<dyn LayoutIrOp>,
}

/// The registry this runtime assembles, extracts, and derives claims
/// with: every matcher paired with a prototype of the (functional-form)
/// op its `extract` produces.
pub fn cuda_registry() -> Vec<RegisteredOp> {
    fn reg(
        matcher: impl OpMatcher + 'static,
        prototype: impl LayoutIrOp + 'static,
    ) -> RegisteredOp {
        RegisteredOp { matcher: Box::new(matcher), prototype: Box::new(prototype) }
    }
    vec![
        reg(add::AddFunctionalMatcher, add::AddFunctional),
        reg(
            materialize_layout_copy::MaterializeLayoutCopyMatcher,
            materialize_layout_copy::MaterializeLayoutCopy,
        ),
        reg(sqrt::SqrtFunctionalMatcher, sqrt::SqrtFunctional),
        reg(exp::ExpFunctionalMatcher, exp::ExpFunctional),
        reg(mul::MulFunctionalMatcher, mul::MulFunctional),
        reg(div::DivFunctionalMatcher, div::DivFunctional),
        reg(trunc_div::TruncDivFunctionalMatcher, trunc_div::TruncDivFunctional),
        reg(trunc_rem::TruncRemFunctionalMatcher, trunc_rem::TruncRemFunctional),
        reg(reduce_sum::ReduceSumMatcher, reduce_sum::ReduceSum { axis: 0 }),
        reg(reduce_max::ReduceMaxMatcher, reduce_max::ReduceMax { axis: 0 }),
        reg(iota::IotaMatcher, iota::Iota { expr: None }),
        reg(gather::GatherMatcher, gather::Gather { rank: 1 }),
        reg(constant::ConstantMatcher, constant::Constant { value: 0.0 }),
        reg(scatter::ScatterFunctionalMatcher, scatter::ScatterFunctional { rank: 1 }),
        reg(exp2::Exp2FunctionalMatcher, exp2::Exp2Functional),
        reg(log2::Log2FunctionalMatcher, log2::Log2Functional),
        reg(sin::SinFunctionalMatcher, sin::SinFunctional),
        reg(recip::RecipFunctionalMatcher, recip::RecipFunctional),
        reg(modulo::ModFunctionalMatcher, modulo::ModFunctional),
        reg(less_than::LessThanMatcher, less_than::LessThan),
        reg(cast::CastMatcher, cast::Cast),
        reg(
            index_map_apply_materialize::IndexMapApplyMaterializeMatcher,
            index_map_apply_materialize::IndexMapApplyMaterialize { entries: None },
        ),
        // M4 Phase 5: the view op is ELECTABLE on this runtime — no
        // kernel, claimed through the plan-transparent class its
        // declared effects prove (see `crate::plan_transparent`).
        reg(
            index_map_apply_view::IndexMapApplyViewMatcher,
            index_map_apply_view::IndexMapApplyView { entries: None },
        ),
    ]
}

/// The matcher set this runtime assembles and extracts with — the
/// registry's matcher column.
pub fn cuda_matchers() -> Vec<Box<dyn OpMatcher>> {
    cuda_registry().into_iter().map(|entry| entry.matcher).collect()
}
