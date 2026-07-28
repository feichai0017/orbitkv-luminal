//! The index-expression generator: a ZERO-INPUT source op that writes
//! `expr(c0..ck)` at every output coordinate — the tensor form of a single
//! `IntExpr` over the output shape's coordinate variables.
//!
//! Iota is the first op with no tensor operands, which makes it the proof
//! case for the zero-input path through extraction (empty operand list),
//! DPS rewriting (the appended destination is the ONLY operand), and
//! bufferization (nothing to alias, one fresh write). There is no mutating
//! form — with no input there is nothing to mutate.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// `IotaGeneric() -> out`
///
/// Functional source form: no operands, one freshly-written result.
/// Reference semantics: `out[c0..ck] = expr(c0..ck)` evaluated in the
/// canonical right-major order, Int element type (the value IS an index
/// expression, never a float; the dtype rule pins this in egglog).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Iota;

impl OpSlotNames for Iota {}

impl BufferTensorIrOp for Iota {
    fn label(&self) -> &str {
        "IotaGeneric"
    }
}

impl Bufferizable for Iota {}

impl ToDps for Iota {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(IotaDps))
    }
}

impl LayoutIrOp for Iota {}

/// Destination-passing form of [`Iota`], signature spelled slot by slot:
///
/// ```text
/// IotaGeneric(dest0: write-only ↔ out0) -> out0
/// ```
///
/// The destination is the op's ONLY operand — the zero-input source's DPS
/// form is pure write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IotaDps;

impl OpSlotNames for IotaDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IotaDps {
    fn label(&self) -> &str {
        "IotaGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for IotaDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 0, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for IotaDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for IotaDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpIotaGeneric` enodes and produces [`Iota`]
/// instances. Metadata children: `expr` at child 0, `shape` at child 1,
/// `out_layout` at child 2 — the whole constructor is metadata; there are
/// no tensor operands.
#[derive(Debug, Clone, Copy, Default)]
pub struct IotaMatcher;

impl OpMatcher for IotaMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIotaGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("iota/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("iota/match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("expr", 0), ("shape", 1), ("out_layout", 2)]
    }

    /// Pure structure — the bounds story lives in egglog (user ruling
    /// 2026-07-23): the iota-int32-certified gate on the op match makes a
    /// missing-bounds iota unimplementable (fail-open), and the
    /// fixpoint-invariants stratum panics on a PROVEN violation. An enode
    /// reaching this matcher is certified by construction.
    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Iota)
    }
}
