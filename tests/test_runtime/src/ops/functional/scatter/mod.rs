//! FORKED from `luminal_reference::ops::scatter` — the TestRuntime owns its
//! whole op set outright and depends on no other runtime.
//!
//! The KERNEL is deliberately not carried over. This runtime is
//! plan-level: it asserts on `ExtractedGraph`s and `BufferIrGraph`s and
//! never executes, so a kernel here would be dead code demanding a
//! dispatch table to sit in. What it needs is the matcher, the instance
//! and the DPS form — the declarations the bufferizer reads.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

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
        vec![AliasInfo {
            operand: self.dest_index(),
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ScatterFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for ScatterFunctionalDps {}

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
        &[("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ScatterFunctional {
            rank: coordinate_rank(site, 2),
        })
    }
}
