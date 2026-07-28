//! Materialize a view: apply an index map to the input and write the gathered
//! elements densely (the index map and output shape are op metadata, not
//! operands).

use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};
use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};

/// `IndexMapApplyMaterialize(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Note the label: the
/// egglog name has no `Generic` suffix, so neither does the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexMapApplyMaterialize;

impl OpSlotNames for IndexMapApplyMaterialize {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IndexMapApplyMaterialize {
    fn label(&self) -> &str {
        "IndexMapApplyMaterialize"
    }
}

impl Bufferizable for IndexMapApplyMaterialize {}

impl ToDps for IndexMapApplyMaterialize {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(IndexMapApplyMaterializeDps))
    }
}

impl LayoutIrOp for IndexMapApplyMaterialize {}

/// Destination-passing form of [`IndexMapApplyMaterialize`], signature spelled
/// slot by slot:
///
/// ```text
/// IndexMapApplyMaterialize(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexMapApplyMaterializeDps;

impl OpSlotNames for IndexMapApplyMaterializeDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IndexMapApplyMaterializeDps {
    fn label(&self) -> &str {
        "IndexMapApplyMaterialize" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for IndexMapApplyMaterializeDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 1, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for IndexMapApplyMaterializeDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for IndexMapApplyMaterializeDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpIndexMapApplyMaterialize` enodes and produces
/// [`IndexMapApplyMaterialize`] instances. Metadata children: `index_map` at child 1, `shape` at child 2, `out_layout` at child 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexMapApplyMaterializeMatcher;

impl OpMatcher for IndexMapApplyMaterializeMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIndexMapApplyMaterialize"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("index_map_apply_materialize/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("index_map_apply_materialize/match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("index_map", 1), ("shape", 2), ("out_layout", 3)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(IndexMapApplyMaterialize)
    }
}
