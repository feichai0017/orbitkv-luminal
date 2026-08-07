//! Data-movement kernels: coordinate gather, the CHECKED coordinate
//! scatter, index-map materialization, and the dense layout copy. Bodies
//! relocated from the op modules (ruling 2026-08-06); the UNCHECKED
//! scatter body was deleted with the ScatterMutating kernel — the
//! reference runtime is out-of-place only and duplicate scatter targets
//! are a deterministic runtime panic here, never last-write-wins.

use super::expect_op;
use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx};
use crate::ssa_reference::ops::{GatherDps, IndexMapApplyMaterializeDps, ScatterFunctionalDps};

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
    let data = ctx.operands[0].as_f32()?.clone();
    let coord_operands: Vec<&Vec<f32>> = ctx.operands[1..1 + rank]
        .iter()
        .map(|operand| operand.as_f32())
        .collect::<anyhow::Result<_>>()?;
    let dest = ctx.dests[0].as_f32_mut()?;
    for flat in 0..dest.len() {
        let mut data_flat = 0usize;
        for axis in 0..rank {
            let coord = coord_operands[axis][flat];
            let coord = coord as i64;
            anyhow::ensure!(
                coord >= 0 && (coord as usize) < data_dims[axis],
                "gather coordinate {coord} out of bounds for axis {axis} (extent {}) — \
                 UB per the scatter/gather ruling, surfaced loudly",
                data_dims[axis]
            );
            data_flat += coord as usize * data_strides[axis];
        }
        dest[flat] = data[data_flat];
    }
    Ok(())
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
    let init = ctx.operands[0].as_f32()?.clone();
    let src = ctx.operands[1].as_f32()?.clone();
    let coord_operands: Vec<Vec<f32>> = ctx.operands[2..2 + rank]
        .iter()
        .map(|operand| operand.as_f32().cloned())
        .collect::<anyhow::Result<_>>()?;
    let dest = ctx.dests[0].as_f32_mut()?;
    dest.copy_from_slice(&init);
    let mut written = vec![false; dest.len()];
    for i in 0..src.len() {
        let mut flat = 0usize;
        for axis in 0..rank {
            let coord = coord_operands[axis][i] as i64;
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
        dest[flat] = src[i];
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
    let parent = ctx.operands[0].as_f32()?.clone();
    let dest = ctx.dests[0].as_f32_mut()?;
    for flat in 0..dest.len() {
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
        dest[flat] = parent[parent_flat];
    }
    Ok(())
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
    let input = ctx.operands[0].as_f32()?;
    let dest = ctx.dests[0].as_f32_mut()?;
    anyhow::ensure!(input.len() == dest.len(), "copy kernel length mismatch");
    dest.copy_from_slice(input);
    Ok(())
}
