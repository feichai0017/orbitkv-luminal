//! Zero-input source kernels: constant fill and the iota coordinate
//! generator. Bodies relocated from the op modules (ruling 2026-08-06);
//! typed dests 2026-08-11 — the iota's exact i64 evaluation now lands
//! in NATIVE integer storage (the old `as f32` store was the producer
//! side of the Int-in-f32 smuggling and its 2^24 exactness cliff).

use super::expect_op;
use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx, TypedBuffer};
use crate::ssa_reference::ops::{ConstantDps, IotaDps};

pub(super) fn constant(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<ConstantDps>(op)?;
    match &mut ctx.dests[0] {
        TypedBuffer::F32(dest) => dest.fill(op.value as f32),
        // LogicalConstant is F32 by its dtype rule; integer dests would
        // mean the plan annotated something the op cannot mean.
        other => anyhow::bail!(
            "constant fill has no {} arm (LogicalConstant is F32)",
            other.type_name()
        ),
    }
    Ok(())
}

pub(super) fn iota(op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let op = expect_op::<IotaDps>(op)?;
    let Some(expr) = &op.expr else {
        anyhow::bail!("iota reference kernel supports Lit/Coord/Add/Mul expressions only");
    };
    let out_dims = ctx.operand_dims.last().cloned().unwrap_or_default();
    let rank = out_dims.len();
    let numel = ctx.dests[0].len();
    let mut coords = vec![0usize; rank];
    let mut eval_at = |flat: usize, coords: &mut Vec<usize>| {
        let mut remainder = flat;
        for axis in (0..rank).rev() {
            coords[axis] = remainder % out_dims[axis];
            remainder /= out_dims[axis];
        }
        expr.eval(coords)
    };
    match &mut ctx.dests[0] {
        // Iota is Int by its dtype rule; the i64 evaluation lands
        // checked in i32 (loud on overflow — non-wrapping ruling).
        TypedBuffer::I32(dest) => {
            for flat in 0..numel {
                let value = eval_at(flat, &mut coords);
                dest[flat] = i32::try_from(value).map_err(|_| {
                    anyhow::anyhow!("iota value {value} overflows i32 (ints are non-wrapping)")
                })?;
            }
        }
        TypedBuffer::I64(dest) => {
            for flat in 0..numel {
                dest[flat] = eval_at(flat, &mut coords);
            }
        }
        other => anyhow::bail!(
            "iota has no {} arm (iota output is Int by its dtype rule)",
            other.type_name()
        ),
    }
    Ok(())
}
