//! StrictAdd: see the match rules for the gating story.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// `StrictAddFunctionalGeneric(lhs, rhs) -> out` — functional form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictAddFunctional;

impl OpSlotNames for StrictAddFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictAddFunctional {
    fn label(&self) -> &str {
        "StrictAddFunctionalGeneric"
    }
}

impl Bufferizable for StrictAddFunctional {}

impl ToDps for StrictAddFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(StrictAddFunctionalDps))
    }
}

impl LayoutIrOp for StrictAddFunctional {}

/// Destination-passing form of [`StrictAddFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrictAddFunctionalDps;

impl OpSlotNames for StrictAddFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for StrictAddFunctionalDps {
    fn label(&self) -> &str {
        "StrictAddFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 | 1 => true,
            2 => false, // dest0: write-only destination
            _ => true,
        }
    }
}

impl Bufferizable for StrictAddFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 2, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for StrictAddFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for StrictAddFunctionalDps {}

/// Matches `LayoutTensorOpStrictAddFunctionalGeneric` enodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrictAddFunctionalMatcher;

impl OpMatcher for StrictAddFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpStrictAddFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("strict_add/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("strict_add/match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(StrictAddFunctional)
    }
}
