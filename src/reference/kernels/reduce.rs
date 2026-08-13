//! Axis-reduction kernels. Bodies relocated from the op modules
//! (ruling 2026-08-06). Typed arms 2026-08-11: Int sums are CHECKED
//! (non-wrapping ruling — an accumulator overflow is a loud kernel
//! error); Int max needs no check (max never leaves the operand range).

use super::expect_op;
use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx, TypedBuffer};
use crate::reference::ops::{ReduceMaxDps, ReduceSumDps};

pub(super) fn sum(op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let op = expect_op::<ReduceSumDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.reduce_axis(op.axis, 0.0, |acc, x| acc + x),
        TypedBuffer::I32(_) => ctx.reduce_axis_i32(op.axis, 0, |acc, x| {
            acc.checked_add(x).ok_or_else(|| {
                anyhow::anyhow!("i32 reduce-sum overflow (ints are non-wrapping)")
            })
        }),
        other => anyhow::bail!("reduce-sum has no {} arm", other.type_name()),
    }
}

pub(super) fn max(op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let op = expect_op::<ReduceMaxDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.reduce_axis(op.axis, f32::NEG_INFINITY, |acc, x| acc.max(x)),
        TypedBuffer::I32(_) => ctx.reduce_axis_i32(op.axis, i32::MIN, |acc, x| Ok(acc.max(x))),
        other => anyhow::bail!("reduce-max has no {} arm", other.type_name()),
    }
}
