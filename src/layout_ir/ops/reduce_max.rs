//! Max reduction along one axis (the axis is op metadata, not an operand).

use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};
use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};

/// `ReduceMaxGeneric(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMax {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceMax {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMax {
    fn label(&self) -> &str {
        "ReduceMaxGeneric"
    }
}

impl Bufferizable for ReduceMax {}

impl ToDps for ReduceMax {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ReduceMaxDps { axis: self.axis }))
    }
}

impl LayoutIrOp for ReduceMax {}

/// Destination-passing form of [`ReduceMax`], signature spelled slot by slot:
///
/// ```text
/// ReduceMaxGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMaxDps {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceMaxDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMaxDps {
    fn reference_execute(
        &self,
        ctx: &mut crate::buffer_tensor_ir::ReferenceKernelCtx,
    ) -> anyhow::Result<()> {
        ctx.reduce_axis(self.axis, f32::NEG_INFINITY, |acc, x| acc.max(x))
    }

    fn label(&self) -> &str {
        "ReduceMaxGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for ReduceMaxDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 1, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for ReduceMaxDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for ReduceMaxDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpReduceMaxGeneric` enodes and produces
/// [`ReduceMax`] instances. Metadata children: `axis` at child 1, `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReduceMaxMatcher;

impl OpMatcher for ReduceMaxMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpReduceMaxGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("reduce_max/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("reduce_max/match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("axis", 1), ("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ReduceMax { axis: site.child_i64(1) })
    }
}
