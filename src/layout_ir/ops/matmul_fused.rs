//! Fused matrix multiply: reduce-sum over an elementwise product of two
//! broadcast views, executed as ONE kernel that never materializes the
//! expanded product. The chain-fusion exemplar: its input list holds the
//! ORIGINAL operand tensors and its output list only the contraction result,
//! so the broadcast views and the [M, N, K] product are absent from the op's
//! I/O — a plan that selects this op never demands them (dead rows, no
//! intermediate buffer).

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// `MatMulFusedGeneric(lhs, rhs) -> out` where `out = sum_k lhs[i,k] * rhs[k,j]`.
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (both operands read, the result freshly allocated). NOT elementwise: every
/// output element reads a full row of `lhs` and column of `rhs`, so no
/// in-place/aliasing claims are made anywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulFused;

impl OpSlotNames for MatMulFused {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatMulFused {
    fn label(&self) -> &str {
        "MatMulFusedGeneric"
    }
}

impl Bufferizable for MatMulFused {}

impl ToDps for MatMulFused {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MatMulFusedDps))
    }
}

impl LayoutIrOp for MatMulFused {}

/// Destination-passing form of [`MatMulFused`]:
///
/// ```text
/// MatMulFusedGeneric(lhs: read, rhs: read, dest: write-only ↔ out) -> out
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulFusedDps;

impl OpSlotNames for MatMulFusedDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatMulFusedDps {
    fn reference_execute(
        &self,
        ctx: &mut crate::buffer_tensor_ir::ReferenceKernelCtx,
    ) -> anyhow::Result<()> {
        let lhs_dims = &ctx.operand_dims[0];
        let rhs_dims = &ctx.operand_dims[1];
        anyhow::ensure!(
            lhs_dims.len() == 2 && rhs_dims.len() == 2 && lhs_dims[1] == rhs_dims[0],
            "matmul kernel geometry mismatch: {lhs_dims:?} x {rhs_dims:?}"
        );
        let (m, k, n) = (lhs_dims[0], lhs_dims[1], rhs_dims[1]);
        anyhow::ensure!(ctx.dests[0].len() == m * n, "matmul dest length mismatch");
        let (lhs, rhs) = (&ctx.operands[0], &ctx.operands[1]);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += lhs[i * k + kk] * rhs[kk * n + j];
                }
                ctx.dests[0][i * n + j] = acc;
            }
        }
        Ok(())
    }

    fn label(&self) -> &str {
        "MatMulFusedGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // lhs
            1 => true,  // rhs
            2 => false, // dest: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for MatMulFusedDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 2, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for MatMulFusedDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for MatMulFusedDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpMatMulFusedGeneric` enodes and produces
/// [`MatMulFused`] instances. Metadata children: `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatMulFusedMatcher;

impl OpMatcher for MatMulFusedMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMatMulFusedGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("matmul_fused/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("matmul_fused/match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MatMulFused)
    }
}
