//! Zero-input source kernels: constant fill and the iota coordinate
//! generator. Bodies relocated from the op modules (ruling 2026-08-06).

use super::expect_op;
use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx};
use crate::ssa_reference::ops::{ConstantDps, IotaDps};

pub(super) fn constant(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<ConstantDps>(op)?;
    ctx.dests[0].as_f32_mut()?.fill(op.value as f32);
    Ok(())
}

pub(super) fn iota(op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let op = expect_op::<IotaDps>(op)?;
    let Some(expr) = &op.expr else {
        anyhow::bail!("iota reference kernel supports Lit/Coord/Add/Mul expressions only");
    };
    let out_dims = ctx.operand_dims.last().cloned().unwrap_or_default();
    let rank = out_dims.len();
    let dest = ctx.dests[0].as_f32_mut()?;
    for flat in 0..dest.len() {
        let mut remainder = flat;
        let mut coords = vec![0usize; rank];
        for axis in (0..rank).rev() {
            coords[axis] = remainder % out_dims[axis];
            remainder /= out_dims[axis];
        }
        dest[flat] = expr.eval(&coords) as f32;
    }
    Ok(())
}
