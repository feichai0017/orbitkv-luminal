//! Axis-reduction kernels. Bodies relocated from the op modules
//! (ruling 2026-08-06).

use super::expect_op;
use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx};
use crate::ssa_reference::ops::{ReduceMaxDps, ReduceSumDps};

pub(super) fn sum(op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let op = expect_op::<ReduceSumDps>(op)?;
    ctx.reduce_axis(op.axis, 0.0, |acc, x| acc + x)
}

pub(super) fn max(op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let op = expect_op::<ReduceMaxDps>(op)?;
    ctx.reduce_axis(op.axis, f32::NEG_INFINITY, |acc, x| acc.max(x))
}
