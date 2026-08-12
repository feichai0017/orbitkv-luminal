//! StrictTruncDiv: see the match rules for the gating story.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// `StrictTruncDivFunctionalGeneric(numerator, denominator) -> out` — functional form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictTruncDivFunctional;

impl OpSlotNames for StrictTruncDivFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictTruncDivFunctional {
    fn label(&self) -> &str {
        "StrictTruncDivFunctionalGeneric"
    }
}

impl Bufferizable for StrictTruncDivFunctional {}

impl ToDps for StrictTruncDivFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(StrictTruncDivFunctionalDps))
    }
}

impl LayoutIrOp for StrictTruncDivFunctional {}

/// Destination-passing form of [`StrictTruncDivFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictTruncDivFunctionalDps;

impl OpSlotNames for StrictTruncDivFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictTruncDivFunctionalDps {
    fn label(&self) -> &str {
        "StrictTruncDivFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 | 1 => true,
            2 => false, // dest0: write-only destination
            _ => true,
        }
    }
}

impl Bufferizable for StrictTruncDivFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 2, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for StrictTruncDivFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for StrictTruncDivFunctionalDps {}

/// Matches `LayoutTensorOpStrictTruncDivFunctionalGeneric` enodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrictTruncDivFunctionalMatcher;

impl OpMatcher for StrictTruncDivFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpStrictTruncDivFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("strict_trunc_div/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("strict_trunc_div/match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(StrictTruncDivFunctional)
    }
}
