//! FORKED from `luminal_reference::ops::gather` — the TestRuntime owns its
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

/// `GatherGeneric(data, coord0, .., coord{r-1}) -> out`
///
/// Functional form: pure dataflow — every operand read, the result freshly
/// allocated. `rank` is the DATA tensor's rank = the number of coordinate
/// operands; total operands = 1 + rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gather {
    pub rank: usize,
}

impl OpSlotNames for Gather {
    fn operand_name(&self, operand: usize) -> String {
        if operand == 0 {
            "data".to_string()
        } else if operand <= self.rank {
            format!("coord{}", operand - 1)
        } else {
            format!("in{operand}")
        }
    }
}

impl BufferTensorIrOp for Gather {
    fn label(&self) -> &str {
        "GatherGeneric"
    }
}

impl Bufferizable for Gather {}

impl ToDps for Gather {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(GatherDps { rank: self.rank }))
    }
}

impl LayoutIrOp for Gather {}

/// Destination-passing form of [`Gather`], signature spelled slot by slot:
///
/// ```text
/// GatherGeneric(data: read, coord0..coord{r-1}: read, dest0: write-only ↔ out0) -> out0
/// ```
///
/// The destination is the trailing operand at index `rank + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherDps {
    pub rank: usize,
}

impl GatherDps {
    fn dest_index(&self) -> usize {
        self.rank + 1
    }
}

impl OpSlotNames for GatherDps {
    fn operand_name(&self, operand: usize) -> String {
        if operand == 0 {
            "data".to_string()
        } else if operand <= self.rank {
            format!("coord{}", operand - 1)
        } else if operand == self.dest_index() {
            "dest0".to_string()
        } else {
            format!("in{operand}")
        }
    }
}

impl BufferTensorIrOp for GatherDps {
    fn label(&self) -> &str {
        "GatherGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != self.dest_index() // dest0 is write-only; everything else reads
    }
}

impl Bufferizable for GatherDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: self.dest_index(),
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for GatherDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for GatherDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpGatherGeneric` enodes and produces [`Gather`]
/// instances. Metadata children: `out_layout` at child 2 (children 0 and 1
/// — the data layout tensor and the coordinate layout-tensor list — are
/// OPERANDS, discovered through the `LayoutTensorOpLit` union partner, not
/// metadata). The instance's `rank` is the coordinate list's length, walked
/// out of the serialized e-graph here.
#[derive(Debug, Clone, Copy, Default)]
pub struct GatherMatcher;

impl OpMatcher for GatherMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpGatherGeneric"
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
        &[("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        // Walk the LayoutTensorCons spine at child 1 counting elements. Each
        // hop resolves BY E-CLASS: a list class can also hold non-structural
        // nodes (functions whose output is the list), and the serializer's
        // chosen child node may be one of those — so every step searches the
        // class for its cons/nil CONSTRUCTOR. A class with neither is schema
        // drift and panics (see the OpMatcher validity contract).
        //
        // NO validity checking happens here (user ruling 2026-07-23):
        // coordinate-shape agreement is a POSITIVE PREMISE of the egglog
        // rules — a gather whose coordinate shapes were never proven equal
        // derives no shape, matches no op, and never reaches this matcher.
        // Extraction reads structure; it does not re-litigate validity.
        let mut rank = 0usize;
        let mut class = site.child_class(1);
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
        Box::new(Gather { rank })
    }
}
