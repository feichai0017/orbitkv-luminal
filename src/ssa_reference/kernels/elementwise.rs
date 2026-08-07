//! Elementwise reference kernels (unary, binary, comparison, cast, and
//! the fused add/mul pair). Bodies relocated verbatim from the op modules
//! (ruling 2026-08-06: execution lives only in the reference runtime).

use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx, TypedBuffer};

pub(super) fn add(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.binary_elementwise(|a, b| a + b)
}

pub(super) fn mul(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.binary_elementwise(|a, b| a * b)
}

/// IEEE f32 division — a zero divisor yields inf/nan exactly as their
/// runtime would; no special-casing.
pub(super) fn div(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.binary_elementwise(|a, b| a / b)
}

pub(super) fn modulo(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.binary_elementwise(|a, b| a % b)
}

pub(super) fn sqrt(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.unary_elementwise(|x| x.sqrt())
}

pub(super) fn exp(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.unary_elementwise(|x| x.exp())
}

pub(super) fn exp2(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.unary_elementwise(|x| x.exp2())
}

pub(super) fn log2(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.unary_elementwise(|x| x.log2())
}

pub(super) fn sin(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.unary_elementwise(|x| x.sin())
}

pub(super) fn recip(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    ctx.unary_elementwise(|x| x.recip())
}

pub(super) fn less_than(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    // The comparison is the one op whose OUTPUT dtype differs from its
    // inputs by construction: f32 operands, a boolean result stored as
    // Bool8 codes (exact 0x00/0x01 — the writer side of the Bool8
    // invariant; never a partial-bit write).
    let lhs = ctx.operands[0].as_f32()?;
    let rhs = ctx.operands[1].as_f32()?;
    let dest = ctx.dests[0].as_bool8_mut()?;
    anyhow::ensure!(
        lhs.len() == rhs.len() && lhs.len() == dest.len(),
        "less-than kernel length mismatch"
    );
    for (index, out) in dest.iter_mut().enumerate() {
        *out = u8::from(lhs[index] < rhs[index]);
    }
    Ok(())
}

pub(super) fn cast(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    // The conversion is driven by the BUFFER types the plan annotated —
    // the op needs no dtype field of its own. Covered pairs only;
    // anything else refuses loudly (never a silent reinterpretation).
    match (&ctx.operands[0], &mut ctx.dests[0]) {
        // Same-type: value-preserving copy (their Int-iota → F32 path
        // stores integer VALUES in f32, so this stays exact).
        (TypedBuffer::F32(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        (TypedBuffer::Bool8(input), TypedBuffer::Bool8(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        // The indicator bridge: bool -> float is exactly 0.0 / 1.0.
        (TypedBuffer::Bool8(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, code) in dest.iter_mut().zip(input) {
                // The Bool8 invariant, enforced at the read: only the
                // two legal codes exist; anything else is ill-formed
                // data, not a truthy byte.
                anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                *out = f32::from(*code);
            }
        }
        (TypedBuffer::F32(_), TypedBuffer::Bool8(_)) => {
            anyhow::bail!(
                "cast f32 -> Bool8 is not a reinterpretation: the != 0 \
                 reading is a PROJECTION and must appear as an explicit \
                 comparison in the model (LessThan), never as a cast"
            );
        }
    }
    Ok(())
}

pub(super) fn add_mul_fused(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let lhs = ctx.operands[0].as_f32()?.clone();
    let rhs = ctx.operands[1].as_f32()?.clone();
    let (sum_dest, product_rest) = ctx.dests.split_at_mut(1);
    let sum_dest = sum_dest[0].as_f32_mut()?;
    let product_dest = product_rest[0].as_f32_mut()?;
    anyhow::ensure!(
        lhs.len() == rhs.len()
            && sum_dest.len() == lhs.len()
            && product_dest.len() == lhs.len(),
        "fused add/mul kernel length mismatch"
    );
    for index in 0..lhs.len() {
        sum_dest[index] = lhs[index] + rhs[index];
        product_dest[index] = lhs[index] * rhs[index];
    }
    Ok(())
}
