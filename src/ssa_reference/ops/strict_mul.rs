//! StrictMul: see the match rules for the gating story.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// `StrictMulFunctionalGeneric(lhs, rhs) -> out` — functional form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictMulFunctional;

impl OpSlotNames for StrictMulFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictMulFunctional {
    fn label(&self) -> &str {
        "StrictMulFunctionalGeneric"
    }
}

impl Bufferizable for StrictMulFunctional {}

impl ToDps for StrictMulFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(StrictMulFunctionalDps))
    }
}

impl LayoutIrOp for StrictMulFunctional {}

/// Destination-passing form of [`StrictMulFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictMulFunctionalDps;

impl OpSlotNames for StrictMulFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictMulFunctionalDps {
    fn label(&self) -> &str {
        "StrictMulFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 | 1 => true,
            2 => false, // dest0: write-only destination
            _ => true,
        }
    }
}

impl Bufferizable for StrictMulFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 2, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for StrictMulFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for StrictMulFunctionalDps {}

/// Matches `LayoutTensorOpStrictMulFunctionalGeneric` enodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrictMulFunctionalMatcher;

impl OpMatcher for StrictMulFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpStrictMulFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("strict_mul/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("strict_mul/match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(StrictMulFunctional)
    }
}
