//! A materialize/copy between two distinct layouts of the same logical value,
//! chosen by extraction. An ordinary op (no special-casing): its cost falls
//! out of the tensor sizes it reads and writes.

use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};
use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};

/// `CopyGeneric(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Elementwise in the
/// LOGICAL index; physical safety is enforced by the engine's
/// layout-equivalence gate. The Rust type keeps its descriptive name; the
/// label follows the label policy (egglog: `LayoutTensorOpCopyGeneric`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializeLayoutCopy;

impl OpSlotNames for MaterializeLayoutCopy {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MaterializeLayoutCopy {
    fn label(&self) -> &str {
        "CopyGeneric"
    }
}

impl Bufferizable for MaterializeLayoutCopy {}

impl ToDps for MaterializeLayoutCopy {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MaterializeLayoutCopyDps))
    }
}

impl LayoutIrOp for MaterializeLayoutCopy {}

/// Destination-passing form of [`MaterializeLayoutCopy`], signature spelled
/// slot by slot:
///
/// ```text
/// CopyGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializeLayoutCopyDps;

impl OpSlotNames for MaterializeLayoutCopyDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MaterializeLayoutCopyDps {
    fn label(&self) -> &str {
        "CopyGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for MaterializeLayoutCopyDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 1, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for MaterializeLayoutCopyDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for MaterializeLayoutCopyDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpCopyGeneric` enodes and produces
/// [`MaterializeLayoutCopy`] instances. Metadata children: `layout` at child 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaterializeLayoutCopyMatcher;

impl OpMatcher for MaterializeLayoutCopyMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCopyGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("layout", 1)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MaterializeLayoutCopy)
    }
}
