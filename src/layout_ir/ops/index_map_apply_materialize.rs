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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyMaterialize {
    /// The index map, numerically — one entry per PARENT axis (outermost
    /// inward). `None` = entries beyond the affine CoordVar/IntLit forms
    /// (arithmetic maps): extraction stays infallible, and the reference
    /// KERNEL refuses loudly instead.
    pub entries: Option<Vec<MapEntry>>,
}

/// One numeric index-map entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapEntry {
    /// A CoordVar over the OUT coordinates (axis zero-based from the END).
    Coord(usize),
    /// A constant index.
    Lit(i64),
}

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
        Some(Box::new(IndexMapApplyMaterializeDps { entries: self.entries.clone() }))
    }
}

impl LayoutIrOp for IndexMapApplyMaterialize {}

/// Destination-passing form of [`IndexMapApplyMaterialize`], signature spelled
/// slot by slot:
///
/// ```text
/// IndexMapApplyMaterialize(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyMaterializeDps {
    /// See [`IndexMapApplyMaterialize::entries`].
    pub entries: Option<Vec<MapEntry>>,
}

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
    fn reference_execute(
        &self,
        ctx: &mut crate::buffer_tensor_ir::ReferenceKernelCtx,
    ) -> anyhow::Result<()> {
        let Some(entries) = &self.entries else {
            anyhow::bail!(
                "materialize reference kernel supports affine CoordVar/IntLit maps only"
            );
        };
        let parent_dims = &ctx.operand_dims[0];
        let out_dims = &ctx.operand_dims[1];
        anyhow::ensure!(
            entries.len() == parent_dims.len(),
            "index map arity {} vs parent rank {}",
            entries.len(),
            parent_dims.len()
        );
        let mut parent_strides = vec![1usize; parent_dims.len()];
        for k in (0..parent_dims.len().saturating_sub(1)).rev() {
            parent_strides[k] = parent_strides[k + 1] * parent_dims[k + 1];
        }
        let out_rank = out_dims.len();
        let parent = &ctx.operands[0];
        for flat in 0..ctx.dests[0].len() {
            // Decompose the flat OUT index into row-major coordinates.
            let mut remainder = flat;
            let mut coords = vec![0usize; out_rank];
            for axis in (0..out_rank).rev() {
                coords[axis] = remainder % out_dims[axis];
                remainder /= out_dims[axis];
            }
            let mut parent_flat = 0usize;
            for (k, entry) in entries.iter().enumerate() {
                let index = match entry {
                    MapEntry::Coord(axis_from_end) => coords[out_rank - 1 - axis_from_end],
                    MapEntry::Lit(value) => *value as usize,
                };
                parent_flat += index * parent_strides[k];
            }
            ctx.dests[0][flat] = parent[parent_flat];
        }
        Ok(())
    }

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

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(IndexMapApplyMaterialize { entries: parse_map_entries(site) })
    }
}

/// Walk the matched term's index-map metadata (child 1) into numeric
/// entries: `IndexMapLit` → cons spine BY E-CLASS → per element a CoordVar
/// (axis primitive at child 0) or an IntLit. Anything else (arithmetic
/// entries) yields `None` — the kernel's loud refusal carries the burden,
/// extraction never fails.
fn parse_map_entries(site: &ExtractionSite<'_>) -> Option<Vec<MapEntry>> {
    let map_class = site.child_class(1);
    let map_node = site.node_in_class(&map_class, "IndexMapLit")?;
    let mut current = site.class_of_child(map_node, 0)?;
    let mut entries = Vec::new();
    loop {
        if let Some(cons) = site.node_in_class(&current, "IntExprCons") {
            let element = site.class_of_child(cons, 0)?;
            let tail = site.class_of_child(cons, 1)?;
            if let Some(coord) = site.node_in_class(&element, "CoordVar") {
                let axis_class = site.class_of_child(coord, 0)?;
                let axis = site
                    .node_in_class_parse_i64(&axis_class)?;
                entries.push(MapEntry::Coord(axis as usize));
            } else if let Some(lit) = site.node_in_class(&element, "IntLit") {
                let value_class = site.class_of_child(lit, 0)?;
                entries.push(MapEntry::Lit(site.node_in_class_parse_i64(&value_class)?));
            } else {
                return None;
            }
            current = tail;
        } else if site.node_in_class(&current, "IntExprNil").is_some() {
            return Some(entries);
        } else {
            return None;
        }
    }
}
