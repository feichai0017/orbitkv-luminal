//! Mutating spelling of `exp`.
//!
//! Rehomed from `luminal_reference::ops::exp` — the reference runtime is
//! functional and out-of-place, and its kernel table has no row for this
//! spelling, so it was registered-but-never-selectable there. The op and
//! its egglog rewrites move together, unchanged.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `ExpMutatingGeneric(x: read+write) -> out`
///
/// Mutating form: the kernel reads and overwrites ONE storage — its
/// single operand's. Matched in egglog only when the output layout equals
/// that operand's layout AND the written tensor is provably injective, so an
/// admitted tie is descriptor-exact by construction. The tie is `May` in the
/// relocation sense: a rejected mutation relocates the operand into the tied
/// result's fresh buffer (copy-in) and mutates there — the kernel's
/// one-buffer contract is invariant under relocation, never a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpMutating;

impl OpSlotNames for ExpMutating {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ExpMutating {
    fn label(&self) -> &str {
        "ExpMutatingGeneric"
    }
}

impl Bufferizable for ExpMutating {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ExpMutating {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for ExpMutating {}

/// Matches `LayoutTensorOpExpMutatingGeneric` enodes and produces
/// [`ExpMutating`] instances. No metadata children: the output layout IS
/// the mutated operand's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpMutatingMatcher;

impl OpMatcher for ExpMutatingMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpExpMutatingGeneric"
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
        Box::new(ExpMutating)
    }
}
