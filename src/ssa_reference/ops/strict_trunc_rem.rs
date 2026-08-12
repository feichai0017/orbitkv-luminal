//! StrictTruncRem: see the match rules for the gating story.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// `StrictTruncRemFunctionalGeneric(numerator, denominator) -> out` — functional form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictTruncRemFunctional;

impl OpSlotNames for StrictTruncRemFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictTruncRemFunctional {
    fn label(&self) -> &str {
        "StrictTruncRemFunctionalGeneric"
    }
}

impl Bufferizable for StrictTruncRemFunctional {}

impl ToDps for StrictTruncRemFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(StrictTruncRemFunctionalDps))
    }
}

impl LayoutIrOp for StrictTruncRemFunctional {}

/// Destination-passing form of [`StrictTruncRemFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictTruncRemFunctionalDps;

impl OpSlotNames for StrictTruncRemFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictTruncRemFunctionalDps {
    fn label(&self) -> &str {
        "StrictTruncRemFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 | 1 => true,
            2 => false, // dest0: write-only destination
            _ => true,
        }
    }
}

impl Bufferizable for StrictTruncRemFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 2, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for StrictTruncRemFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for StrictTruncRemFunctionalDps {}

/// Matches `LayoutTensorOpStrictTruncRemFunctionalGeneric` enodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrictTruncRemFunctionalMatcher;

impl OpMatcher for StrictTruncRemFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpStrictTruncRemFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("strict_trunc_rem/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("strict_trunc_rem/match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(StrictTruncRemFunctional)
    }
}
