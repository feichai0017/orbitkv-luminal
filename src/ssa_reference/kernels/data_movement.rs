//! Data-movement kernels: coordinate gather, the CHECKED coordinate
//! scatter, index-map materialization, and the dense layout copy. Bodies
//! relocated from the op modules (ruling 2026-08-06); the UNCHECKED
//! scatter body was deleted with the ScatterMutating kernel — the
//! reference runtime is out-of-place only and duplicate scatter targets
//! are a deterministic runtime panic here, never last-write-wins.
//!
//! TYPED (2026-08-11): coordinates are read from NATIVE i32 buffers —
//! the old `as_f32` read plus `as i64` truncation was the consumer side
//! of the Int-in-f32 smuggling (a 4.9999995 truncating to 4 while
//! passing the bounds check). Payload movement is variant-generic:
//! the index computation is dtype-independent, and the element moves
//! dispatch once on the (source, dest) variant pair.

use super::expect_op;
use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx, TypedBuffer};
use crate::ssa_reference::ops::{GatherDps, IndexMapApplyMaterializeDps, ScatterFunctionalDps};

/// Read a rank of coordinate operands as native i32, promoted to i64
/// for index arithmetic.
fn coordinate_columns(
    operands: &[TypedBuffer],
) -> anyhow::Result<Vec<Vec<i64>>> {
    operands
        .iter()
        .map(|operand| {
            Ok(operand
                .as_i32()?
                .iter()
                .map(|value| i64::from(*value))
                .collect())
        })
        .collect()
}

/// dest[flat] = source[index[flat]] for every flat, whatever the payload
/// dtype — variants must match (the plan annotated both sides).
fn move_gathered(
    source: &TypedBuffer,
    dest: &mut TypedBuffer,
    index_of: &[usize],
) -> anyhow::Result<()> {
    match (source, dest) {
        (TypedBuffer::F32(data), TypedBuffer::F32(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::I32(data), TypedBuffer::I32(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::I64(data), TypedBuffer::I64(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::Bool8(data), TypedBuffer::Bool8(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::F8E4M3(data), TypedBuffer::F8E4M3(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (source, dest) => anyhow::bail!(
            "payload move between {} and {} buffers",
            source.type_name(),
            dest.type_name()
        ),
    }
    Ok(())
}

pub(super) fn gather(op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let op = expect_op::<GatherDps>(op)?;
    let rank = op.rank;
    let data_dims = &ctx.operand_dims[0];
    anyhow::ensure!(
        data_dims.len() == rank,
        "gather kernel: data rank {} vs op rank {}",
        data_dims.len(),
        rank
    );
    let mut data_strides = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        data_strides[k] = data_strides[k + 1] * data_dims[k + 1];
    }
    let coord_operands = coordinate_columns(&ctx.operands[1..1 + rank])?;
    let numel = ctx.dests[0].len();
    let mut index_of = vec![0usize; numel];
    for (flat, slot) in index_of.iter_mut().enumerate() {
        let mut data_flat = 0usize;
        for axis in 0..rank {
            let coord = coord_operands[axis][flat];
            anyhow::ensure!(
                coord >= 0 && (coord as usize) < data_dims[axis],
                "gather coordinate {coord} out of bounds for axis {axis} (extent {}) — \
                 UB per the scatter/gather ruling, surfaced loudly",
                data_dims[axis]
            );
            data_flat += coord as usize * data_strides[axis];
        }
        *slot = data_flat;
    }
    let source = ctx.operands[0].clone();
    move_gathered(&source, &mut ctx.dests[0], &index_of)
}

/// The CHECKED scatter kernel (ruling 2026-08-06): dest starts as a copy
/// of init, then dest[coords(i)] = src[i] over the src iteration space.
/// Out-of-bounds coordinates are UB by ruling — surfaced LOUDLY. Two src
/// elements landing on one destination element is a runtime panic, not
/// last-write-wins: write conflicts mean the coordinate tensors are not
/// injective and the program's meaning is coordinate-order-dependent,
/// which we refuse to silently define. (User-provided coordinate data may
/// eventually carry an asserted-injective bit instead; until then the
/// reference checks.)
pub(super) fn scatter_checked(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<ScatterFunctionalDps>(op)?;
    let rank = op.rank;
    let init_dims = ctx.operand_dims[0].clone();
    anyhow::ensure!(
        init_dims.len() == rank,
        "scatter kernel: init rank {} vs op rank {rank}",
        init_dims.len()
    );
    let mut strides = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * init_dims[k + 1];
    }
    let coord_operands = coordinate_columns(&ctx.operands[2..2 + rank])?;
    let src_len = ctx.operands[1].len();
    let dest_len = ctx.dests[0].len();
    let mut target_of = vec![0usize; src_len];
    let mut written = vec![false; dest_len];
    for (i, target) in target_of.iter_mut().enumerate() {
        let mut flat = 0usize;
        for axis in 0..rank {
            let coord = coord_operands[axis][i];
            anyhow::ensure!(
                coord >= 0 && (coord as usize) < init_dims[axis],
                "scatter coordinate {coord} out of bounds for axis {axis} (extent {}) — \
                 UB per ruling, surfaced loudly",
                init_dims[axis]
            );
            flat += coord as usize * strides[axis];
        }
        anyhow::ensure!(
            !written[flat],
            "conflicting scatter writes: src element {i} targets destination element \
             {flat}, which an earlier src element already wrote — coordinates must be \
             injective (checked-scatter ruling 2026-08-06)"
        );
        written[flat] = true;
        *target = flat;
    }
    // Payload move: dest = copy(init); dest[target[i]] = src[i].
    match (&ctx.operands[0], &ctx.operands[1], &mut ctx.dests[0]) {
        (TypedBuffer::F32(init), TypedBuffer::F32(src), TypedBuffer::F32(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::I32(init), TypedBuffer::I32(src), TypedBuffer::I32(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::I64(init), TypedBuffer::I64(src), TypedBuffer::I64(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::Bool8(init), TypedBuffer::Bool8(src), TypedBuffer::Bool8(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::F8E4M3(init), TypedBuffer::F8E4M3(src), TypedBuffer::F8E4M3(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (init, src, dest) => anyhow::bail!(
            "scatter payload dtypes disagree: init {}, src {}, dest {}",
            init.type_name(),
            src.type_name(),
            dest.type_name()
        ),
    }
    Ok(())
}

pub(super) fn materialize(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<IndexMapApplyMaterializeDps>(op)?;
    let Some(entries) = &op.entries else {
        anyhow::bail!(
            "materialize reference kernel: index map beyond the parsed expression subset"
        );
    };
    let parent_dims = &ctx.operand_dims[0];
    let out_dims = &ctx.operand_dims[1];
    anyhow::ensure!(
        entries.len() == parent_dims.len(),
        "index map arity {} vs parent rank {}",
        entries.len(),
        parent_dims.len()
    );
    let mut parent_strides = vec![1usize; parent_dims.len()];
    for k in (0..parent_dims.len().saturating_sub(1)).rev() {
        parent_strides[k] = parent_strides[k + 1] * parent_dims[k + 1];
    }
    let out_rank = out_dims.len();
    let numel = ctx.dests[0].len();
    let mut index_of = vec![0usize; numel];
    for (flat, slot) in index_of.iter_mut().enumerate() {
        // Decompose the flat OUT index into row-major coordinates.
        let mut remainder = flat;
        let mut coords = vec![0usize; out_rank];
        for axis in (0..out_rank).rev() {
            coords[axis] = remainder % out_dims[axis];
            remainder /= out_dims[axis];
        }
        let mut parent_flat = 0usize;
        for (k, entry) in entries.iter().enumerate() {
            let index = entry.eval(&coords);
            anyhow::ensure!(
                index >= 0 && (index as usize) < parent_dims[k],
                "materialize index {index} out of bounds for parent axis {k} (extent {})",
                parent_dims[k]
            );
            parent_flat += index as usize * parent_strides[k];
        }
        *slot = parent_flat;
    }
    let parent = ctx.operands[0].clone();
    move_gathered(&parent, &mut ctx.dests[0], &index_of)
}

/// Dense same-geometry copy (CopyGeneric / MaterializeLayoutCopy). The
/// reference runtime holds every buffer dense row-major (no view reads in
/// its allow list), so materializing a copy IS an element copy — but only
/// under identical geometry, which is checked loudly rather than assumed.
pub(super) fn copy(_op: &dyn BufferTensorIrOp, ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    anyhow::ensure!(
        ctx.operand_dims[0] == ctx.operand_dims[1],
        "copy kernel: input geometry {:?} vs dest geometry {:?} — a shape-changing \
         copy is not a dense copy and has no reference lowering",
        ctx.operand_dims[0],
        ctx.operand_dims[1]
    );
    anyhow::ensure!(
        ctx.operands[0].len() == ctx.dests[0].len(),
        "copy kernel length mismatch"
    );
    let index_of: Vec<usize> = (0..ctx.dests[0].len()).collect();
    let source = ctx.operands[0].clone();
    move_gathered(&source, &mut ctx.dests[0], &index_of)
}
