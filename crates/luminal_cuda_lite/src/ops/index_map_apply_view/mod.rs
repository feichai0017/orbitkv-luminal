//! CUDA-lite's OWN view op (op-ownership ruling 2026-08-17: every
//! runtime owns its executable — and here, its PLAN-TRANSPARENT — ops;
//! the reference crate's copy stays untouched and the reference runtime
//! stays permanently materialize-only per ruling aff22598).
//!
//! Reinterpret the parent's storage through a composed offset function:
//! the index map, output shape, and composed layout are op metadata,
//! not operands. No bytes move — the planner binds the result into its
//! parent's buffer (the Must tie) and lowering folds the op to a
//! producer redirect. On THIS runtime the fold is what M4 Phases 1-4
//! built toward: the consumer's operand descriptor carries the composed
//! access and the CUDA kernel reads straight through it.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `IndexMapApplyViewGeneric(input) -> out`
///
/// The metadata-view form of index-map application. The egglog match
/// admits it only where the output layout IS the composed layout (the
/// parent's offset function precomposed with the index map), so the
/// result is the operand's own bytes by construction: the operand is
/// never read (no bytes observed) and the result is never written (no
/// bytes produced). A rejected tie is repairable — a view over a copy
/// of the parent's buffer is a faithful lowering — so Must is a
/// requirement on WHERE the shared storage lives, never a hard error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyView {
    /// The index map, numerically — one full expression tree per PARENT
    /// axis (outermost inward), evaluated at the OUT coordinates; parsed
    /// at extraction exactly like the materialize ops' entries. The
    /// bufferizer reads this through `view_index_map` when it folds the
    /// op, recording the composed access on consumer slot descriptors
    /// (M4 Phase 3). `None` = entries beyond the parsed expression
    /// subset: the fold records a fail-closed hop and numeric consumers
    /// refuse loudly.
    pub entries: Option<Vec<luminal::index_expr::IotaExpr>>,
}

impl OpSlotNames for IndexMapApplyView {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IndexMapApplyView {
    fn label(&self) -> &str {
        "IndexMapApplyViewGeneric"
    }

    fn operand_reads_memory(&self, _operand: usize) -> bool {
        false // metadata op: no bytes observed
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // metadata op: no bytes produced
    }

    fn view_index_map(&self, _result: usize) -> Option<Vec<luminal::index_expr::IotaExpr>> {
        self.entries.clone()
    }
}

impl Bufferizable for IndexMapApplyView {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for IndexMapApplyView {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // nothing is written: there is no destination to pass
    }
}

impl LayoutIrOp for IndexMapApplyView {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpIndexMapApplyViewGeneric` enodes and produces
/// [`IndexMapApplyView`] instances. Metadata children: `index_map` at child 1, `shape` at child 2, `out_layout` at child 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexMapApplyViewMatcher;

impl OpMatcher for IndexMapApplyViewMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIndexMapApplyViewGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("index_map", 1), ("shape", 2), ("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        // Metadata children: index_map at 1, OUT shape at 2 (the
        // owner-shape guard) — the same walk the materialize matcher does.
        Box::new(IndexMapApplyView {
            entries: luminal::index_expr::parse_index_map_entries(site, 1, 2),
        })
    }
}
