//! Matrix-multiply kernels. Body relocated from the op module
//! (ruling 2026-08-06).

use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx};

pub(super) fn matmul_fused(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let lhs_dims = &ctx.operand_dims[0];
    let rhs_dims = &ctx.operand_dims[1];
    anyhow::ensure!(
        lhs_dims.len() == 2 && rhs_dims.len() == 2 && lhs_dims[1] == rhs_dims[0],
        "matmul kernel geometry mismatch: {lhs_dims:?} x {rhs_dims:?}"
    );
    let (m, k, n) = (lhs_dims[0], lhs_dims[1], rhs_dims[1]);
    let lhs = ctx.operands[0].as_f32()?.clone();
    let rhs = ctx.operands[1].as_f32()?.clone();
    let dest = ctx.dests[0].as_f32_mut()?;
    anyhow::ensure!(dest.len() == m * n, "matmul dest length mismatch");
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += lhs[i * k + kk] * rhs[kk * n + j];
            }
            dest[i * n + j] = acc;
        }
    }
    Ok(())
}
