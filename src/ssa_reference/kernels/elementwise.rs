//! Elementwise reference kernels (unary, binary, comparison, cast, and
//! the fused add/mul pair). Bodies relocated verbatim from the op modules
//! (ruling 2026-08-06: execution lives only in the reference runtime).

use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx, TypedBuffer};

/// Add/Mul dispatch on the operand variants. Int arithmetic is CHECKED
/// (non-wrapping by ruling 2026-08-11): an overflow is a loud kernel
/// error, never a wrapped value — until the landing-D bounds proofs
/// gate Int ops statically, this dynamic check is the soundness floor.
pub(super) fn add(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.binary_elementwise(|a, b| a + b),
        TypedBuffer::I32(_) => ctx.binary_elementwise_i32(|a, b| {
            a.checked_add(b)
                .ok_or_else(|| anyhow::anyhow!("i32 add overflow: {a} + {b} (ints are non-wrapping)"))
        }),
        TypedBuffer::I64(_) => ctx.binary_elementwise_i64(|a, b| {
            a.checked_add(b)
                .ok_or_else(|| anyhow::anyhow!("i64 add overflow: {a} + {b} (ints are non-wrapping)"))
        }),
        other => anyhow::bail!("add has no {} arm", other.type_name()),
    }
}

pub(super) fn mul(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.binary_elementwise(|a, b| a * b),
        TypedBuffer::I32(_) => ctx.binary_elementwise_i32(|a, b| {
            a.checked_mul(b)
                .ok_or_else(|| anyhow::anyhow!("i32 mul overflow: {a} * {b} (ints are non-wrapping)"))
        }),
        TypedBuffer::I64(_) => ctx.binary_elementwise_i64(|a, b| {
            a.checked_mul(b)
                .ok_or_else(|| anyhow::anyhow!("i64 mul overflow: {a} * {b} (ints are non-wrapping)"))
        }),
        other => anyhow::bail!("mul has no {} arm", other.type_name()),
    }
}

/// IEEE f32 division — a zero divisor yields inf/nan exactly as their
/// runtime would; no special-casing. Int operands REFUSE here: integer
/// division is a DIFFERENT mathematical function (truncation toward
/// zero) and gets its own operators, LogicalTruncDiv/LogicalTruncRem
/// (ruling 2026-08-11, landing D) — Div on Int would be a silent
/// semantics substitution.
pub(super) fn div(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.binary_elementwise(|a, b| a / b),
        other => anyhow::bail!(
            "Div has no {} arm: integer division is TruncDiv, not Div",
            other.type_name()
        ),
    }
}

/// Same story as [`div`]: f32 `%` only; integer remainder is TruncRem.
pub(super) fn modulo(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.binary_elementwise(|a, b| a % b),
        other => anyhow::bail!(
            "Mod has no {} arm: integer remainder is TruncRem, not Mod",
            other.type_name()
        ),
    }
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
    // inputs by construction: same-typed operands, a boolean result
    // stored as Bool8 codes (exact 0x00/0x01 — the writer side of the
    // Bool8 invariant; never a partial-bit write). Int comparisons are
    // native and exact — no f32 detour (typed buffers 2026-08-11).
    match (&ctx.operands[0], &ctx.operands[1]) {
        (TypedBuffer::F32(lhs), TypedBuffer::F32(rhs)) => {
            let (lhs, rhs) = (lhs.clone(), rhs.clone());
            let dest = ctx.dests[0].as_bool8_mut()?;
            anyhow::ensure!(
                lhs.len() == rhs.len() && lhs.len() == dest.len(),
                "less-than kernel length mismatch"
            );
            for (index, out) in dest.iter_mut().enumerate() {
                *out = u8::from(lhs[index] < rhs[index]);
            }
        }
        (TypedBuffer::I32(lhs), TypedBuffer::I32(rhs)) => {
            let (lhs, rhs) = (lhs.clone(), rhs.clone());
            let dest = ctx.dests[0].as_bool8_mut()?;
            anyhow::ensure!(
                lhs.len() == rhs.len() && lhs.len() == dest.len(),
                "less-than kernel length mismatch"
            );
            for (index, out) in dest.iter_mut().enumerate() {
                *out = u8::from(lhs[index] < rhs[index]);
            }
        }
        (TypedBuffer::I64(lhs), TypedBuffer::I64(rhs)) => {
            let (lhs, rhs) = (lhs.clone(), rhs.clone());
            let dest = ctx.dests[0].as_bool8_mut()?;
            anyhow::ensure!(
                lhs.len() == rhs.len() && lhs.len() == dest.len(),
                "less-than kernel length mismatch"
            );
            for (index, out) in dest.iter_mut().enumerate() {
                *out = u8::from(lhs[index] < rhs[index]);
            }
        }
        (lhs, rhs) => anyhow::bail!(
            "less-than has no ({}, {}) arm (operands must share one dtype)",
            lhs.type_name(),
            rhs.type_name()
        ),
    }
    Ok(())
}

pub(super) fn cast(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    // The conversion is driven by the BUFFER types the plan annotated —
    // the op needs no dtype field of its own. Covered pairs only;
    // anything else refuses loudly (never a silent reinterpretation).
    // The 2026-08-11 cast policy: int -> float is CHECKED-EXACT
    // (conservative |v| <= 2^24, loud outside it); float -> int is a
    // REFUSAL (a lossy read is an explicit op, never a cast — the
    // Bool8 projection rule generalized); bool -> number is the exact
    // 0/1 indicator bridge; number -> bool is always the refusal.
    const F32_EXACT_INT: i64 = 1 << 24;
    match (&ctx.operands[0], &mut ctx.dests[0]) {
        // Same-type: value-preserving copies.
        (TypedBuffer::F32(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        (TypedBuffer::I32(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        (TypedBuffer::I64(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        (TypedBuffer::Bool8(input), TypedBuffer::Bool8(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        // The indicator bridges: bool -> number is exactly 0 / 1.
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
        (TypedBuffer::Bool8(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, code) in dest.iter_mut().zip(input) {
                anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                *out = i32::from(*code);
            }
        }
        (TypedBuffer::Bool8(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, code) in dest.iter_mut().zip(input) {
                anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                *out = i64::from(*code);
            }
        }
        // Int -> float: checked-exact (conservative), loud outside it.
        (TypedBuffer::I32(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                anyhow::ensure!(
                    (i64::from(*value)).abs() <= F32_EXACT_INT,
                    "cast i32 -> f32 loses exactness at value {value} \
                     (|v| <= 2^24 by the conservative-exact ruling)"
                );
                *out = *value as f32;
            }
        }
        (TypedBuffer::I64(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                anyhow::ensure!(
                    value.abs() <= F32_EXACT_INT,
                    "cast i64 -> f32 loses exactness at value {value} \
                     (|v| <= 2^24 by the conservative-exact ruling)"
                );
                *out = *value as f32;
            }
        }
        // Int widenings/narrowings: exact or loud.
        (TypedBuffer::I32(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                *out = i64::from(*value);
            }
        }
        (TypedBuffer::I64(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                *out = i32::try_from(*value).map_err(|_| {
                    anyhow::anyhow!("cast i64 -> i32 out of range at value {value}")
                })?;
            }
        }
        // Float -> int is a REFUSAL: rounding/truncation is a lossy
        // read and must be an explicit op with ruled semantics, never
        // a cast (the F32 -> Bool8 projection rule generalized).
        (TypedBuffer::F32(_), TypedBuffer::I32(_) | TypedBuffer::I64(_)) => {
            anyhow::bail!(
                "cast f32 -> int is not a reinterpretation: a rounding \
                 or truncation is a lossy read and must appear as an \
                 explicit op in the model, never as a cast"
            );
        }
        (TypedBuffer::F32(_) | TypedBuffer::I32(_) | TypedBuffer::I64(_), TypedBuffer::Bool8(_)) => {
            anyhow::bail!(
                "cast number -> Bool8 is not a reinterpretation: the != 0 \
                 reading is a PROJECTION and must appear as an explicit \
                 comparison in the model (LessThan), never as a cast"
            );
        }
        (input, dest) => anyhow::bail!(
            "cast has no ({} -> {}) arm",
            input.type_name(),
            dest.type_name()
        ),
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
