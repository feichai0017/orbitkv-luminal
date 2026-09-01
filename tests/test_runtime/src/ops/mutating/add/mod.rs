//! Mutating elementwise add — plain, and the input-alias-safe spelling.
//!
//! Rehomed from `luminal_reference::ops::add` — the reference runtime is
//! functional and out-of-place, and its kernel table has no row for this
//! spelling, so it was registered-but-never-selectable there. The op and
//! its egglog rewrites move together, unchanged.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `AddMutatingGeneric(lhs: read+write, rhs: read) -> out`
///
/// Mutating form: the kernel reads and overwrites ONE storage — its
/// first operand's. Matched in egglog only when the output layout equals
/// that operand's layout AND the written tensor is provably injective, so an
/// admitted tie is descriptor-exact by construction. The tie is `May` in the
/// relocation sense: a rejected mutation relocates the operand into the tied
/// result's fresh buffer (copy-in) and mutates there — the kernel's
/// one-buffer contract is invariant under relocation, never a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddMutating;

impl OpSlotNames for AddMutating {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddMutating {
    fn label(&self) -> &str {
        "AddMutatingGeneric"
    }
}

impl Bufferizable for AddMutating {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for AddMutating {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for AddMutating {}

/// `AddMutatingInputAliasSafeGeneric(lhs: read+write, rhs: read) -> out`
///
/// Mutating form whose egglog match requires BOTH inputs and the output to
/// share one layout — which is exactly what makes it safe for `rhs` to alias
/// the mutated storage: element `i` coincides everywhere by construction,
/// and an elementwise kernel reads element `i` before writing it. The op
/// therefore declares the may-share permit for `rhs` against the mutated
/// result. The permit is unconditional here because its precondition was
/// discharged at match time; the engine trusts it and checks no layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddMutatingInputAliasSafe;

impl OpSlotNames for AddMutatingInputAliasSafe {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddMutatingInputAliasSafe {
    fn label(&self) -> &str {
        "AddMutatingInputAliasSafeGeneric"
    }
}

impl Bufferizable for AddMutatingInputAliasSafe {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![
            AliasInfo {
                operand: 0,
                result: 0,
                sharing: Sharing::Must,
            },
            AliasInfo {
                operand: 1,
                result: 0,
                sharing: Sharing::May,
            },
        ]
    }
}

impl ToDps for AddMutatingInputAliasSafe {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for AddMutatingInputAliasSafe {}

/// Matches `LayoutTensorOpAddMutatingGeneric` enodes and produces
/// [`AddMutating`] instances. No metadata children: the output layout IS
/// the mutated operand's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddMutatingMatcher;

impl OpMatcher for AddMutatingMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAddMutatingGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_mutating_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_mutating.egg"),
            },
        ]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(AddMutating)
    }
}

/// Matches `LayoutTensorOpAddMutatingInputAliasSafeGeneric` enodes and produces
/// [`AddMutatingInputAliasSafe`] instances. No metadata children: the output layout IS
/// the mutated operand's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddMutatingInputAliasSafeMatcher;

impl OpMatcher for AddMutatingInputAliasSafeMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAddMutatingInputAliasSafeGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_mutating_alias_safe_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_mutating_alias_safe.egg"),
            },
        ]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(AddMutatingInputAliasSafe)
    }
}
