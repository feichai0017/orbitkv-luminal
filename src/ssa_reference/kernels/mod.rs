//! The REFERENCE RUNTIME's kernel inventory — the single home of every
//! reference implementation (ruling 2026-08-06). Ops in `ssa_reference::ops`
//! carry NO execution: they are semantics + matching + bufferization
//! contracts. A kernel exists in this folder or the op is simply not
//! implemented on this runtime — and because [`super::reference_allow_list`]
//! is DERIVED from this table, the runtime can neither over-claim (an
//! allow-listed op with no kernel) nor under-claim (a kernel the search
//! is not offered).
//!
//! BUNDLE-LEVEL CLAIMS, TYPE-LEVEL DISPATCH (ruling 2026-08-06): the
//! runtime's CLAIM is per op FAMILY — one matcher constructor covers
//! every rank, layout, and instance its matcher can match; nothing is
//! ever enumerated per variant. One registry row = one executable FORM
//! of a family (today always the DPS form), and instance flexibility
//! (rank, axis, expression, map entries) flows into the kernel through
//! the downcast. The row key is the concrete op TYPE (TypeId), not the
//! label, because labels are shared: DPS forms keep their functional
//! form's IR name, and only DPS forms are executable (plans are
//! post-`dps_rewrite`) — TypeId dispatch turns that from an assumption
//! into a checked invariant. Any op type absent from the table refuses
//! loudly at execution ("no reference kernel for ..."). A future
//! layout-specialized kernel changes nothing here: the family still
//! claims once; its kernel branches internally on instance data.

mod data_movement;
mod elementwise;
mod matmul;
mod plan_infra;
mod reduce;
mod source;

use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx};
use std::any::TypeId;

pub struct ReferenceKernel {
    /// The op's IR label (DPS forms keep the functional form's label);
    /// the allow-list derivation matches this against matcher constructors.
    pub label: &'static str,
    /// The concrete op type this kernel downcasts to.
    pub op_type: TypeId,
    pub execute: fn(&dyn BufferTensorIrOp, &mut ReferenceKernelCtx) -> anyhow::Result<()>,
}

fn entry<T: 'static>(
    label: &'static str,
    execute: fn(&dyn BufferTensorIrOp, &mut ReferenceKernelCtx) -> anyhow::Result<()>,
) -> ReferenceKernel {
    ReferenceKernel { label, op_type: TypeId::of::<T>(), execute }
}

/// The full kernel table. One entry per executable op type; labels repeat
/// only if two executable types legitimately share one (none do today —
/// functional forms are not executable and have no entries).
pub fn reference_kernels() -> &'static [ReferenceKernel] {
    use crate::ssa_reference::ops::*;
    static KERNELS: std::sync::OnceLock<Vec<ReferenceKernel>> = std::sync::OnceLock::new();
    KERNELS.get_or_init(|| {
        vec![
            // ── elementwise ──
            entry::<AddFunctionalDps>("AddFunctionalGeneric", elementwise::add),
            entry::<MulFunctionalDps>("MulFunctionalGeneric", elementwise::mul),
            entry::<DivFunctionalDps>("DivFunctionalGeneric", elementwise::div),
            entry::<TruncDivFunctionalDps>("TruncDivFunctionalGeneric", elementwise::trunc_div),
            entry::<TruncRemFunctionalDps>("TruncRemFunctionalGeneric", elementwise::trunc_rem),
            entry::<StrictAddFunctionalDps>("StrictAddFunctionalGeneric", elementwise::strict_add),
            entry::<StrictMulFunctionalDps>("StrictMulFunctionalGeneric", elementwise::strict_mul),
            entry::<StrictTruncDivFunctionalDps>("StrictTruncDivFunctionalGeneric", elementwise::trunc_div),
            entry::<StrictTruncRemFunctionalDps>("StrictTruncRemFunctionalGeneric", elementwise::trunc_rem),
            entry::<ModFunctionalDps>("ModFunctionalGeneric", elementwise::modulo),
            entry::<SqrtFunctionalDps>("SqrtFunctionalGeneric", elementwise::sqrt),
            entry::<ExpFunctionalDps>("ExpFunctionalGeneric", elementwise::exp),
            entry::<Exp2FunctionalDps>("Exp2FunctionalGeneric", elementwise::exp2),
            entry::<Log2FunctionalDps>("Log2FunctionalGeneric", elementwise::log2),
            entry::<SinFunctionalDps>("SinFunctionalGeneric", elementwise::sin),
            entry::<RecipFunctionalDps>("RecipFunctionalGeneric", elementwise::recip),
            entry::<LessThanDps>("LessThanGeneric", elementwise::less_than),
            entry::<CastDps>("CastGeneric", elementwise::cast),
            entry::<AddMulFusedDps>("AddMulFusedGeneric", elementwise::add_mul_fused),
            // ── sources ──
            entry::<ConstantDps>("ConstantGeneric", source::constant),
            entry::<IotaDps>("IotaGeneric", source::iota),
            // ── reductions ──
            entry::<ReduceSumDps>("ReduceSumGeneric", reduce::sum),
            entry::<ReduceMaxDps>("ReduceMaxGeneric", reduce::max),
            // ── matmul ──
            entry::<MatMulFusedDps>("MatMulFusedGeneric", matmul::matmul_fused),
            // ── data movement ──
            entry::<GatherDps>("GatherGeneric", data_movement::gather),
            entry::<ScatterFunctionalDps>("ScatterFunctionalGeneric", data_movement::scatter_checked),
            entry::<IndexMapApplyMaterializeDps>("IndexMapApplyMaterialize", data_movement::materialize),
            entry::<MaterializeLayoutCopyDps>("CopyGeneric", data_movement::copy),
            // ── plan infrastructure (bufferizer-synthesized, no matchers) ──
            entry::<crate::buffer_tensor_ir::BufferAlloc>("BufferAlloc", plan_infra::alloc),
            entry::<crate::buffer_tensor_ir::BufferFree>("BufferFree", plan_infra::free),
        ]
    })
}

/// Look up the kernel for a plan op by its concrete type.
pub fn kernel_for(op: &dyn BufferTensorIrOp) -> Option<&'static ReferenceKernel> {
    let op_type = op.as_any().type_id();
    reference_kernels().iter().find(|kernel| kernel.op_type == op_type)
}

/// Downcast the dispatched op to the kernel's concrete type — a mismatch
/// means the registry row and the kernel disagree, which refuses loudly.
pub(crate) fn expect_op<T: 'static>(op: &dyn BufferTensorIrOp) -> anyhow::Result<&T> {
    op.as_any().downcast_ref::<T>().ok_or_else(|| {
        anyhow::anyhow!(
            "kernel dispatched to a different op type than its registry row declares (registry drift)"
        )
    })
}
