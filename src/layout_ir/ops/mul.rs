//! Elementwise multiplication.

use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};
use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};

/// `MulFunctionalGeneric(lhs, rhs) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Elementwise: element
/// `i` of each input is read before element `i` of `out` is written (op-level
/// all-pairs claim — see the NOTE on `bufferizes_to_elementwise_access`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulFunctional;

impl OpSlotNames for MulFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MulFunctional {
    fn label(&self) -> &str {
        "MulFunctionalGeneric"
    }
}

impl Bufferizable for MulFunctional {}

impl ToDps for MulFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MulFunctionalDps))
    }
}

impl LayoutIrOp for MulFunctional {}

/// Destination-passing form of [`MulFunctional`], signature spelled slot by slot:
///
/// ```text
/// MulGeneric(lhs: read, rhs: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulFunctionalDps;

impl OpSlotNames for MulFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MulFunctionalDps {
    fn label(&self) -> &str {
        "MulFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // lhs
            1 => true,  // rhs
            2 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for MulFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 2, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for MulFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for MulFunctionalDps {}

/// `MulMutatingGeneric(lhs: read+write, rhs: read) -> out`
///
/// Mutating form: the kernel reads and overwrites ONE storage — its
/// first operand's. Matched in egglog only when the output layout equals
/// that operand's layout AND the written tensor is provably injective, so an
/// admitted tie is descriptor-exact by construction. The tie is `May` in the
/// relocation sense: a rejected mutation relocates the operand into the tied
/// result's fresh buffer (copy-in) and mutates there — the kernel's
/// one-buffer contract is invariant under relocation, never a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulMutating;

impl OpSlotNames for MulMutating {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MulMutating {
    fn label(&self) -> &str {
        "MulMutatingGeneric"
    }
}

impl Bufferizable for MulMutating {

    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 0, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for MulMutating {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for MulMutating {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpMulFunctionalGeneric` enodes and produces
/// [`MulFunctional`] instances. Metadata children: `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulFunctionalMatcher;

impl OpMatcher for MulFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMulFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("mul/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("mul/match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MulFunctional)
    }
}

/// Matches `LayoutTensorOpMulMutatingGeneric` enodes and produces
/// [`MulMutating`] instances. No metadata children: the output layout IS
/// the mutated operand's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulMutatingMatcher;

impl OpMatcher for MulMutatingMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMulMutatingGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("mul/match_mutating_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("mul/match_mutating.egg"),
            },
        ]
    }


    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MulMutating)
    }
}
