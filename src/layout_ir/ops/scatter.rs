//! Coordinate-form scatter (R8 dual):
//! `out[coord_0[c], .., coord_{r-1}[c]] = src[c]` for `c` over src's shape;
//! everywhere else `out = init`. One Int coordinate tensor per INIT axis,
//! all sharing SRC's shape. Duplicate coordinates and out-of-bounds
//! coordinate VALUES are undefined behavior (user ruling 2026-07-22).
//!
//! Operand 0 is named `init` — it provides the result's initial contents
//! (MLIR's term) — NOT `dest`, deliberately: the DPS rewrite appends a
//! trailing `dest0` destination to the functional form, and two slots both
//! named "dest" in one plan would be unreadable. In-place intent is a
//! BOUNDARY statement (result and init sharing one BufferId); whether it
//! executes in place is decided by which implementation extraction picks:
//! the functional form (fresh result) or the mutating form (writes into
//! init's storage, Must-tied, matched only when the output layout IS
//! init's).
//!
//! Like gather, scatter is variable-arity and its instances carry data:
//! `rank` = init's rank = the coordinate count, read out of the e-graph by
//! the matchers. Operand order: init, src, coord0..coord{r-1} — fixed
//! slots first, the variable tail last.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// Walk the LayoutTensorCons spine at `child` counting elements — the
/// shared rank reader for both scatter matchers (same class-resolving walk
/// as gather's; see the OpMatcher validity contract for the panics).
fn coordinate_rank(site: &ExtractionSite<'_>, child: usize) -> usize {
    let mut rank = 0usize;
    let mut class = site.child_class(child);
    loop {
        let spine = site
            .egraph
            .nodes
            .values()
            .find(|node| {
                node.eclass == class
                    && (node.op == "LayoutTensorCons" || node.op == "LayoutTensorNil")
            })
            .unwrap_or_else(|| {
                panic!(
                    "schema drift: coordinate-list class {class} under enode {} has no \
                     LayoutTensorCons/LayoutTensorNil constructor",
                    site.node_id
                )
            });
        if spine.op == "LayoutTensorNil" {
            break;
        }
        rank += 1;
        let tail_id = spine.children.get(1).unwrap_or_else(|| {
            panic!("schema drift: a LayoutTensorCons in class {class} has no tail child")
        });
        class = site
            .egraph
            .nodes
            .get(tail_id)
            .unwrap_or_else(|| panic!("dangling list tail node {tail_id}"))
            .eclass
            .clone();
    }
    rank
}

/// `ScatterFunctionalGeneric(init, src, coord0, .., coord{r-1}) -> out`
///
/// Functional form: pure dataflow — every operand read (init supplies the
/// unwritten regions), the result freshly allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterFunctional {
    pub rank: usize,
}

impl OpSlotNames for ScatterFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "init".to_string(),
            1 => "src".to_string(),
            n if n < 2 + self.rank => format!("coord{}", n - 2),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScatterFunctional {
    fn label(&self) -> &str {
        "ScatterFunctionalGeneric"
    }
}

impl Bufferizable for ScatterFunctional {}

impl ToDps for ScatterFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ScatterFunctionalDps { rank: self.rank }))
    }
}

impl LayoutIrOp for ScatterFunctional {}

/// Destination-passing form of [`ScatterFunctional`]:
///
/// ```text
/// ScatterFunctionalGeneric(init: read, src: read, coord0..: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterFunctionalDps {
    pub rank: usize,
}

impl ScatterFunctionalDps {
    fn dest_index(&self) -> usize {
        self.rank + 2
    }
}

impl OpSlotNames for ScatterFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        if operand == 0 {
            "init".to_string()
        } else if operand == 1 {
            "src".to_string()
        } else if operand < self.dest_index() {
            format!("coord{}", operand - 2)
        } else if operand == self.dest_index() {
            "dest0".to_string()
        } else {
            format!("in{operand}")
        }
    }
}

impl BufferTensorIrOp for ScatterFunctionalDps {
    fn label(&self) -> &str {
        "ScatterFunctionalGeneric" // DPS forms keep the IR name
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != self.dest_index() // dest0 is write-only; everything else reads
    }
}

impl Bufferizable for ScatterFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: self.dest_index(), result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for ScatterFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for ScatterFunctionalDps {}

/// `ScatterMutatingGeneric(init: read+write, src: read, coord0..: read) -> out`
///
/// Mutating form: the kernel reads and overwrites ONE storage — init's.
/// Matched in egglog only when the output layout IS init's layout (the
/// precondition, discharged at match time). The Must tie is relocatable as
/// always: a rejected mutation copies init into the tied result's fresh
/// buffer and mutates there. NOTE: no write-map injectivity gate exists or
/// can — scatter's writes are data-dependent, and duplicate coordinates
/// are UB by ruling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterMutating {
    pub rank: usize,
}

impl OpSlotNames for ScatterMutating {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "init".to_string(),
            1 => "src".to_string(),
            n if n < 2 + self.rank => format!("coord{}", n - 2),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScatterMutating {
    fn label(&self) -> &str {
        "ScatterMutatingGeneric"
    }
}

impl Bufferizable for ScatterMutating {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 0, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for ScatterMutating {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for ScatterMutating {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpScatterFunctionalGeneric` enodes and produces
/// [`ScatterFunctional`] instances. Metadata children: `out_layout` at
/// child 3 (children 0-2 — init, src, the coordinate list — are OPERANDS,
/// discovered through the `LayoutTensorOpLit` union partner). `rank` is
/// the coordinate list's length, walked out of the e-graph.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScatterFunctionalMatcher;

impl OpMatcher for ScatterFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpScatterFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("scatter/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("scatter/match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ScatterFunctional { rank: coordinate_rank(site, 2) })
    }
}

/// Matches `LayoutTensorOpScatterMutatingGeneric` enodes and produces
/// [`ScatterMutating`] instances. No metadata children: the output layout
/// IS init's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScatterMutatingMatcher;

impl OpMatcher for ScatterMutatingMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpScatterMutatingGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("scatter/match_mutating_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("scatter/match_mutating.egg"),
            },
        ]
    }


    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ScatterMutating { rank: coordinate_rank(site, 2) })
    }
}
