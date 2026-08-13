//! The built-in LayoutIR op set, one file per op pair (functional form and
//! its destination-passing form, defined side by side — the pairing is part
//! of each op's definition).
//!
//! Everything here is deliberately hand-written and verbose: no macro, no
//! shared index arithmetic. Each op spells out its operand signature slot by
//! slot — which operands are read, and which destination operand is tied to
//! which result — the way a function signature spells out its returns.
//!
//! Label policy: an op's label is its egglog constructor name with only the
//! `LayoutTensorOp` namespace prefix removed — nothing else is edited (the
//! `Generic` suffix stays, and NO suffix is ever appended). The IR name is the
//! op's identity: future op sets will encode calling convention in the name
//! itself (e.g. FunctionalAdd vs InPlaceAdd), so tooling must never do name
//! surgery. The DPS form of an op therefore displays the SAME label; DPS-ness
//! is visible in the operand list (trailing destinations), not the name.

// The instance re-exports are the op-authoring surface: since the extractor
// went registry-driven it constructs instances only through matchers, so the
// non-test build no longer names these types — their remaining consumers are
// tests and the upcoming reference-backend/DPS work (same situation as the
// `#![allow(dead_code)]` note in `layout_ir.rs`).
#![allow(unused_imports)]

mod add;
mod add_mul_fused;
mod cast;
mod constant;
mod div;
mod trunc_div;
mod trunc_rem;
mod exp;
mod exp2;
mod gather;
mod index_map_apply_materialize;
mod index_map_apply_view;
mod iota;
mod less_than;
mod log2;
mod materialize_layout_copy;
mod matmul_fused;
mod modulo;
mod mul;
mod poison;
mod recip;
mod reduce_max;
mod reduce_sum;
mod scatter;
mod sin;
mod sqrt;

// The functional forms, the mutating forms, and Poison are exported: a DPS
// form is an implementation detail of its op pair, entering the world solely
// as a `Box<dyn LayoutIrOp>` through `to_dps()`. (A mutating form is NOT a
// DPS detail — it is its own extractable implementation.) If a pass ever
// needs a concrete DPS type, re-export it explicitly here.
//
// Instance structs deliberately do NOT derive `Default`: an instance is
// created by its matcher's `extract` (or derived via `to_dps` / synthesized
// by the DPS pass, for Poison) — a hygiene convention, not a privacy fence.
pub use add::{AddFunctional, AddMutating, AddMutatingInputAliasSafe};
pub use add_mul_fused::AddMulFused;
pub use cast::Cast;
pub use constant::Constant;
pub use div::{DivFunctional, DivMutating};
pub use exp::{ExpFunctional, ExpMutating};
pub use exp2::{Exp2Functional, Exp2Mutating};
pub use gather::Gather;
pub use index_map_apply_materialize::IndexMapApplyMaterialize;
pub use index_map_apply_view::IndexMapApplyView;
pub use iota::Iota;
pub use less_than::LessThan;
pub use log2::{Log2Functional, Log2Mutating};
pub use modulo::{ModFunctional, ModMutating};
pub use materialize_layout_copy::MaterializeLayoutCopy;
pub use mul::{MulFunctional, MulMutating};
pub use poison::Poison;
pub use recip::{RecipFunctional, RecipMutating};
pub use reduce_max::ReduceMax;
pub use reduce_sum::ReduceSum;
pub use scatter::{ScatterFunctional, ScatterMutating};
pub use sin::{SinFunctional, SinMutating};
pub use sqrt::{SqrtFunctional, SqrtMutating};

// DPS forms, re-exported for the reference runtime's kernel registry
// (reference::kernels downcasts plan ops to these concrete types —
// ops carry no execution of their own, ruling 2026-08-06).
pub use add::AddFunctionalDps;
pub use add_mul_fused::AddMulFusedDps;
pub use cast::CastDps;
pub use constant::ConstantDps;
pub use div::DivFunctionalDps;
pub use exp::ExpFunctionalDps;
pub use exp2::Exp2FunctionalDps;
pub use gather::GatherDps;
pub use index_map_apply_materialize::IndexMapApplyMaterializeDps;
pub use iota::IotaDps;
pub use less_than::LessThanDps;
pub use log2::Log2FunctionalDps;
pub use materialize_layout_copy::MaterializeLayoutCopyDps;
pub use matmul_fused::MatMulFusedDps;
pub use modulo::ModFunctionalDps;
pub use mul::MulFunctionalDps;
pub use recip::RecipFunctionalDps;
pub use reduce_max::ReduceMaxDps;
pub use reduce_sum::ReduceSumDps;
pub use scatter::ScatterFunctionalDps;
pub use sin::SinFunctionalDps;
pub use sqrt::SqrtFunctionalDps;

pub use add::{AddFunctionalMatcher, AddMutatingInputAliasSafeMatcher, AddMutatingMatcher};
pub use add_mul_fused::AddMulFusedMatcher;
pub use cast::CastMatcher;
pub use constant::ConstantMatcher;
pub use div::{DivFunctionalMatcher, DivMutatingMatcher};
pub use trunc_div::{TruncDivFunctional, TruncDivFunctionalDps, TruncDivFunctionalMatcher};
pub use trunc_rem::{TruncRemFunctional, TruncRemFunctionalDps, TruncRemFunctionalMatcher};
pub use exp::{ExpFunctionalMatcher, ExpMutatingMatcher};
pub use exp2::{Exp2FunctionalMatcher, Exp2MutatingMatcher};
pub use gather::GatherMatcher;
pub use index_map_apply_materialize::IndexMapApplyMaterializeMatcher;
pub use index_map_apply_view::IndexMapApplyViewMatcher;
pub use iota::IotaMatcher;
pub use less_than::LessThanMatcher;
pub use log2::{Log2FunctionalMatcher, Log2MutatingMatcher};
pub use modulo::{ModFunctionalMatcher, ModMutatingMatcher};
pub use materialize_layout_copy::MaterializeLayoutCopyMatcher;
pub use matmul_fused::MatMulFusedMatcher;
pub use mul::{MulFunctionalMatcher, MulMutatingMatcher};
pub use recip::{RecipFunctionalMatcher, RecipMutatingMatcher};
pub use reduce_max::ReduceMaxMatcher;
pub use reduce_sum::ReduceSumMatcher;
pub use scatter::{ScatterFunctionalMatcher, ScatterMutatingMatcher};
pub use sin::{SinFunctionalMatcher, SinMutatingMatcher};
pub use sqrt::{SqrtFunctionalMatcher, SqrtMutatingMatcher};

/// THE registration list: every built-in matcher, one line per registered
/// implementation. Adding an op = writing its module (instances + traits +
/// matcher, all in one file) and adding its line here; nothing else in the
/// tree changes. The extractor builds its constructor-name registry from this
/// list — there is no dispatch table anywhere else. (Poison has no matcher:
/// it is synthesized by the DPS pass, never extracted from the e-graph.)
pub fn built_in_matchers() -> Vec<Box<dyn crate::layout_ir::OpMatcher>> {
    vec![
        Box::new(MaterializeLayoutCopyMatcher),
        Box::new(SqrtFunctionalMatcher),
        Box::new(ExpFunctionalMatcher),
        Box::new(AddFunctionalMatcher),
        Box::new(MulFunctionalMatcher),
        Box::new(DivFunctionalMatcher),
        Box::new(TruncDivFunctionalMatcher),
        Box::new(TruncRemFunctionalMatcher),
        Box::new(SqrtMutatingMatcher),
        Box::new(ExpMutatingMatcher),
        Box::new(AddMutatingMatcher),
        Box::new(AddMutatingInputAliasSafeMatcher),
        Box::new(MulMutatingMatcher),
        Box::new(DivMutatingMatcher),
        Box::new(ReduceSumMatcher),
        Box::new(ReduceMaxMatcher),
        Box::new(IotaMatcher),
        Box::new(GatherMatcher),
        Box::new(ConstantMatcher),
        Box::new(ScatterFunctionalMatcher),
        Box::new(ScatterMutatingMatcher),
        Box::new(Exp2FunctionalMatcher),
        Box::new(Exp2MutatingMatcher),
        Box::new(Log2FunctionalMatcher),
        Box::new(Log2MutatingMatcher),
        Box::new(SinFunctionalMatcher),
        Box::new(SinMutatingMatcher),
        Box::new(RecipFunctionalMatcher),
        Box::new(RecipMutatingMatcher),
        Box::new(ModFunctionalMatcher),
        Box::new(ModMutatingMatcher),
        Box::new(LessThanMatcher),
        Box::new(CastMatcher),
        Box::new(IndexMapApplyMaterializeMatcher),
        Box::new(IndexMapApplyViewMatcher),
        Box::new(AddMulFusedMatcher),
        Box::new(MatMulFusedMatcher),
    ]
}
