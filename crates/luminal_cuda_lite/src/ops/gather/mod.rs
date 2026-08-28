//! Coordinate-form gather (R8):
//! `out[c] = data[coord_0[c], .., coord_{r-1}[c]]` — CUDA-lite's OWN
//! op (ruling 2026-08-17: every runtime owns its executable ops; the
//! shared crate supplies only the IR traits). Same egglog constructor
//! and label as the reference runtime's gather — assemblies are
//! per-runtime, labels are IR identity — but the structs, matcher,
//! snippets, and codegen all live here. Variable-arity: `rank` is the
//! DATA tensor's rank = the coordinate operand count, walked out of
//! the e-graph by the matcher and baked into the instance.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    composed_read_index, composed_read_index_pref, coord_prelude, cuda_type, numel, strides_of,
    CodegenCtx, KernelSource,
};
use anyhow::{bail, Result};

/// `GatherGeneric(data, coord0, .., coord{r-1}) -> out` — pure
/// dataflow form; total operands = 1 + rank.
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

/// Destination-passing form: `Gather(data: read, coord0..: read,
/// dest0: write ↔ out0)` — the destination is the trailing operand at
/// index `rank + 1`.
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
        "GatherGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != self.dest_index() // dest0 is write-only; everything else reads
    }
}

impl Bufferizable for GatherDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: self.dest_index(), result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for GatherDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for GatherDps {}

/// The CUDA lowering, colocated with its op. Train-2B: every READ
/// operand (data AND coordinates) may carry a [`ComposedAccess`] — the
/// hop chain composes into that operand's read index exactly as
/// [`crate::kernels::composed_read_index`] does for the elementwise
/// templates. The WRITE side (dest0) stays fail-closed (CL-4b).
pub(crate) fn codegen(
    op: &dyn BufferTensorIrOp,
    ctx: &CodegenCtx,
) -> Result<Vec<KernelSource>> {
    let Some(gather) = op.as_any().downcast_ref::<GatherDps>() else {
        bail!("gather codegen reached with a non-Gather op");
    };
    let rank = gather.rank;
    if ctx.composed_access.get(gather.dest_index()).is_some_and(Option::is_some) {
        bail!(
            "dest operand slot {} carries a composed access: strided writes \
             are not lowered (dests stay dense out-of-place; CL-4b)",
            gather.dest_index()
        );
    }
    let data_dims = &ctx.operand_dims[0];
    if data_dims.len() != rank {
        bail!("gather data rank {} vs op rank {rank}", data_dims.len());
    }
    let t = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let out_dims = &ctx.dest_dims[0];
    let n = numel(out_dims);
    let strides = strides_of(data_dims);
    let mut sig = format!("const {t}* data");
    for axis in 0..rank {
        sig.push_str(&format!(", const int* coord{axis}"));
    }
    if ctx.composed_access.iter().all(Option::is_none) {
        // The flat fast path, byte-identical to pre-Train-2B codegen.
        let mut body = String::from("    long long flat = 0;\n    long long coord;\n");
        for axis in 0..rank {
            body.push_str(&format!(
                "    coord = (long long)coord{axis}[i];\n    if (coord < 0 || coord >= {ext}LL) __trap();\n    flat += coord * {stride}LL;\n",
                ext = data_dims[axis],
                stride = strides[axis]
            ));
        }
        let source = format!(
            r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{body}    out[i] = data[flat];
}}"#
        );
        return Ok(vec![KernelSource::plain(source, n)]);
    }
    // The strided branch. A COORDINATE operand's value spans the out
    // iteration space (`coord{k}[i]` in the flat kernel), so its chain
    // is evaluated at the OUT coordinates — the coordinate prelude is
    // needed iff some coordinate operand carries a fold. The DATA
    // operand's value coordinates are the gathered coordinate values
    // themselves: they are bound as `data_c{axis}` and the data chain
    // is evaluated at THOSE (the gather's own indirection composes ON
    // TOP of the folded chain).
    let coord_folded =
        (1..=rank).any(|slot| ctx.composed_access.get(slot).is_some_and(Option::is_some));
    let data_access = ctx.composed_access[0].as_ref();
    let mut body = String::new();
    if coord_folded {
        body.push_str(&coord_prelude(out_dims));
    }
    if data_access.is_none() {
        body.push_str("    long long flat = 0;\n");
    }
    body.push_str("    long long coord;\n");
    for axis in 0..rank {
        if let Some(access) = ctx.composed_access.get(axis + 1).and_then(|a| a.as_ref()) {
            // The coordinate value's own extents must be the out
            // extents for `c*` to be its coordinates: refuse a
            // mismatch, never reinterpret (the elementwise contract).
            if &ctx.operand_dims[axis + 1] != out_dims {
                bail!(
                    "operand coord{axis} value extents {:?} differ from dest extents {:?} \
                     under composed access — the gather iterates the dest",
                    ctx.operand_dims[axis + 1],
                    out_dims
                );
            }
            let name = format!("coord{axis}");
            let (chain, idx) = composed_read_index(&name, access, out_dims.len())?;
            body.push_str(&chain);
            body.push_str(&format!("    coord = (long long){name}[{idx}];\n"));
        } else {
            body.push_str(&format!("    coord = (long long)coord{axis}[i];\n"));
        }
        // The gather's own checked contract: coordinates are bounded by
        // the data VALUE's extents, composed access or not.
        body.push_str(&format!(
            "    if (coord < 0 || coord >= {ext}LL) __trap();\n",
            ext = data_dims[axis]
        ));
        if data_access.is_some() {
            body.push_str(&format!("    long long data_c{axis} = coord;\n"));
        } else {
            body.push_str(&format!("    flat += coord * {stride}LL;\n", stride = strides[axis]));
        }
    }
    let read = if let Some(access) = data_access {
        let (chain, idx) = composed_read_index_pref("data", access, rank, "data_c")?;
        body.push_str(&chain);
        format!("data[{idx}]")
    } else {
        "data[flat]".to_string()
    };
    let source = format!(
        r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{body}    out[i] = {read};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Matches `LayoutTensorOpGatherGeneric` and produces this runtime's
/// [`Gather`]. Metadata children: `out_layout` at child 2 (children 0
/// and 1 — the data layout tensor and the coordinate layout-tensor
/// list — are OPERANDS). The instance's `rank` is the coordinate
/// list's length, walked out of the serialized e-graph here.
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
