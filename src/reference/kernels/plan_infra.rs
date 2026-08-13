//! Kernels for bufferizer-synthesized plan infrastructure. These are not
//! LayoutTensor ops (no matchers, never in the allow list); they exist so
//! the executor's dispatch stays uniform.

use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx};

/// No computation: storage is pre-materialized; contents start undefined
/// (zeros here).
pub(super) fn alloc(
    _op: &dyn BufferTensorIrOp,
    _ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    Ok(())
}

/// No computation: liveness bookkeeping only; the reference executor
/// keeps storage.
pub(super) fn free(_op: &dyn BufferTensorIrOp, _ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    Ok(())
}
