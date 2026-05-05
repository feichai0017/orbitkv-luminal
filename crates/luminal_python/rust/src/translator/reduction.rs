use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

/// Compute total element count, returning an error if any dimension is symbolic.
fn concrete_numel(a: &GraphTensor) -> Result<usize> {
    a.dims().iter().try_fold(1usize, |acc, d| {
        d.to_usize().map(|v| acc * v).ok_or_else(|| {
            anyhow::anyhow!("Full reduction requires concrete dimensions, got symbolic dim")
        })
    })
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_reduction(
        &mut self,
        node: &Node,
        op: ReductionOp,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;

        // Try to get dims arg; if missing or empty, fall back to full reduce
        let dims_result = self.get_ints_arg(node, 1);
        let (axes, keepdim) = match dims_result {
            Ok(ref dims) if !dims.is_empty() => {
                let ndim = a.shape.len();
                let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
                let keepdim = if node.inputs.len() > 2 {
                    self.get_bool_arg(node, 2).unwrap_or(false)
                } else {
                    false
                };
                (axes, keepdim)
            }
            _ => {
                // Full reduce: reduce over every axis, leaving a rank-0 (scalar) tensor.
                // PyTorch eager returns shape () for `x.sum()` etc., and downstream ops
                // (e.g. unsqueeze(0).expand(N)) rely on this rank.
                let ndim = a.shape.len();
                if ndim == 0 {
                    // Already rank-0 — reducing over no axes is a no-op for sum/max/min/prod,
                    // and mean of a scalar is just the scalar.
                    return Ok(a);
                }
                let total = concrete_numel(&a)?;
                let axes: Vec<usize> = (0..ndim).collect();
                let result = match op {
                    ReductionOp::Sum => a.sum(axes),
                    // Note: the luminal `mean` helper divides by the product of the
                    // axis dims, but we already require concrete dims here so we
                    // divide by the cached `total` to avoid recomputing.
                    ReductionOp::Mean => a.sum(axes) / total as f32,
                    ReductionOp::Max => a.max(axes),
                    ReductionOp::Min => a.min(axes),
                    ReductionOp::Prod => a.prod(axes),
                };
                return Ok(result);
            }
        };

        let mut result = match op {
            ReductionOp::Sum => a.sum(axes.clone()),
            ReductionOp::Mean => a.mean(axes.clone()),
            ReductionOp::Max => a.max(axes.clone()),
            ReductionOp::Min => a.min(axes.clone()),
            ReductionOp::Prod => a.prod(axes.clone()),
        };

        if keepdim {
            let mut sorted_axes = axes.clone();
            sorted_axes.sort();
            for &ax in &sorted_axes {
                result = result.unsqueeze(ax);
            }
        }

        Ok(result)
    }
}
