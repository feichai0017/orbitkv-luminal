//! The CUDA codegen table: one row per executable op type, keyed by the
//! concrete DPS struct's `TypeId` exactly like the reference kernel
//! registry (labels repeat across functional/DPS forms; types do not).
//!
//! A row's `codegen` turns (op instance, buffer geometry) into a
//! self-contained CUDA source string — dense row-major, one thread per
//! output element, geometry baked as literals. Generation is pure and
//! host-side (snapshot-testable without a device); NVRTC compilation
//! and launch live in the `device` module. The codegen BODIES live in
//! the op modules under `crate::ops` (op-ownership ruling 2026-08-17);
//! this file keeps the table and the shared lowering helpers.
//!
//! CL-1 coverage: the elementwise family + constant + cast + copy +
//! axis reductions + the expression-carrying ops (iota, materialize,
//! gather, scatter). The allow list stays honest by construction —
//! search can only elect what this table generates.

use anyhow::{bail, Result};
use luminal::buffer_tensor_ir::BufferTensorIrOp;
use crate::layouts::CudaLayout;
use luminal::bufferize::SlotDescriptor;
use luminal::dtype::PlanDtype;
use luminal::index_expr::IotaExpr;
use std::any::TypeId;

/// Geometry + typing for one compute node, in plan order: operands
/// (destination-last, the DPS convention), then destinations again as
/// the write set. EVERYTHING here derives from the node's own
/// [`SlotDescriptor`] layouts — this runtime's carried `CudaLayout` per
/// slot: dims are the layout's literal domain extents, dtypes its
/// carried dtype fact, and every non-direct read lowers the layout's
/// own offset expression ([`layout_read_index`]). The hop-chain
/// machinery is fully retired (corrected contract, 2026-08-31): the
/// e-graph mints every view's composed layout at view creation, and the
/// runtime's rendered `L` IS the read path.
#[derive(Debug)]
pub struct CodegenCtx {
    pub operand_dims: Vec<Vec<usize>>,
    pub operand_dtypes: Vec<PlanDtype>,
    pub dest_dims: Vec<Vec<usize>>,
    pub dest_dtypes: Vec<PlanDtype>,
    /// Per-operand slot layouts, parallel to `operand_dims` — each
    /// operand's OWN elected layout as the runtime's renderer minted it
    /// (for a folded operand, the view's COMPOSED layout, addressing
    /// the residence's bytes directly).
    pub operand_layouts: Vec<CudaLayout>,
}

impl CodegenCtx {
    /// Build codegen geometry from the compute node's own slot
    /// descriptors — never the shared buffer table. Dims and dtypes come
    /// from each slot's carried layout (the layout's DOMAIN is the
    /// value's shape); loud on symbolic extents or a missing dtype fact,
    /// never a guess.
    pub fn from_descriptors(
        label: &str,
        operand_info: &[SlotDescriptor<CudaLayout>],
        result_info: &[SlotDescriptor<CudaLayout>],
    ) -> Result<Self> {
        let dims_of = |slot: &SlotDescriptor<CudaLayout>, role: &str| -> Result<Vec<usize>> {
            slot.layout.mirror.literal_extents().ok_or_else(|| {
                anyhow::anyhow!("{label} {role} has symbolic layout extents (no numeric codegen)")
            })
        };
        let dtype_of = |slot: &SlotDescriptor<CudaLayout>, role: &str| -> Result<PlanDtype> {
            slot.layout
                .dtype
                .ok_or_else(|| anyhow::anyhow!("{label} {role} carries no dtype fact"))
        };
        let dest_dims: Vec<Vec<usize>> = result_info
            .iter()
            .map(|s| dims_of(s, "dest"))
            .collect::<Result<_>>()?;
        // Strided WRITES are not lowered (CL-4b territory): destinations
        // stay dense out-of-place, so a result slot whose layout is not
        // the direct row-major form over its domain refuses loudly at
        // the single codegen entry point — a CAPABILITY refusal (this
        // backend lowers no strided write), never an e-graph re-check.
        for (k, (slot, dims)) in result_info.iter().zip(&dest_dims).enumerate() {
            if !layout_is_direct(&slot.layout, dims) {
                bail!(
                    "{label} result {k} carries a non-direct layout: strided writes \
                     are not lowered (dests stay dense out-of-place; CL-4b)"
                );
            }
        }
        Ok(CodegenCtx {
            operand_dims: operand_info
                .iter()
                .map(|s| dims_of(s, "operand"))
                .collect::<Result<_>>()?,
            operand_dtypes: operand_info
                .iter()
                .map(|s| dtype_of(s, "operand"))
                .collect::<Result<_>>()?,
            dest_dims,
            dest_dtypes: result_info
                .iter()
                .map(|s| dtype_of(s, "dest"))
                .collect::<Result<_>>()?,
            operand_layouts: operand_info.iter().map(|s| s.layout.clone()).collect(),
        })
    }

    /// The slot's layout when it is NOT the direct read for its dims —
    /// the expression-read discriminator every family keys on (`None` =
    /// the flat `name[i]` fast path holds).
    pub fn non_direct_operand(&self, slot: usize) -> Option<&CudaLayout> {
        let layout = &self.operand_layouts[slot];
        (!layout_is_direct(layout, &self.operand_dims[slot])).then_some(layout)
    }
}

// ===========================================================================
// PROTOTYPE (Option B): reading operands through their SLOT LAYOUTS.
//
// The slot's own elected layout (`SlotDescriptor::layout`, the runtime's
// rendered `MirrorLayout`) is the ONE vocabulary for how a value
// addresses its residence — for a folded operand it is the view's
// COMPOSED layout, which the e-graph already minted (preamble view
// BitOffset composition / native strided chains). The elementwise family
// below lowers that layout's offset expression DIRECTLY, retiring the
// per-slot hop chain for this family.
//
// BOUNDS HONESTY (pinned in `codegen_identity::strided`): the hop chain
// trapped EVERY intermediate index against its hop's parent extents; a
// composed expression has no intermediate parents, so the trap surface
// shrinks to ONE final check of the flat element index — against the
// layout's own SPAN where the layout discloses one (the packed ladder:
// right-major / left-major / strided), and against NOTHING but
// non-negativity for the offset-expression forms, which deliberately do
// not disclose their reach (`SpanExpr` is unimplemented there). That is
// the cost of the composed read: out-of-bounds inside the expression is
// caught only if the final index escapes the span (packed) or goes
// negative (offset forms).
// ===========================================================================

/// Is this layout the DIRECT read for a value of `dims` — row-major,
/// packed, value-shaped? (The flat `a[i]` fast path; also the CL-4b
/// write fence.) Rank ≤ 1 left-major is the same function but the
/// renderer prefers the right-major spelling when present, so we key on
/// right-major alone — a dense class rendering otherwise takes the
/// (correct, slower) expression read and the byte-identity pin flags it.
pub fn layout_is_direct(layout: &CudaLayout, dims: &[usize]) -> bool {
    match &layout.mirror {
        luminal::layouts::MirrorLayout::RightMajor(_) => {
            layout.mirror.literal_extents().as_deref() == Some(dims)
        }
        _ => false,
    }
}

/// Lower a mirror-layout [`IntExprTerm`] to a C expression over
/// `long long`, coordinates spelled `{prefix}{front_index}`
/// (`Coord{axis_from_end}` reads `{prefix}{rank-1-axis_from_end}`).
/// Symbolic vars bail loudly (no numeric codegen for symbolic layouts).
fn lower_layout_term(
    expr: &luminal::layouts::IntExprTerm,
    rank: usize,
    prefix: &str,
) -> Result<String> {
    use luminal::layouts::IntExprTerm as T;
    let rec = |e: &T| lower_layout_term(e, rank, prefix);
    Ok(match expr {
        T::Lit(v) => format!("{v}LL"),
        T::Var(name) => bail!("layout read: symbolic dim `{name}` has no numeric codegen"),
        T::Coord { axis_from_end } => {
            let axis = usize::try_from(*axis_from_end)
                .ok()
                .filter(|&a| a < rank)
                .ok_or_else(|| {
                    anyhow::anyhow!("layout read: coordinate axis {axis_from_end} out of rank {rank}")
                })?;
            format!("{prefix}{}", rank - 1 - axis)
        }
        T::Add(a, b) => format!("({} + {})", rec(a)?, rec(b)?),
        T::Mul(a, b) => format!("({} * {})", rec(a)?, rec(b)?),
        T::TruncDiv(a, b) => format!("({} / {})", rec(a)?, rec(b)?),
        T::TruncRem(a, b) => format!("({} % {})", rec(a)?, rec(b)?),
        T::CeilDiv(a, b) => {
            // PROTOTYPE: minted layouts have not needed CeilDiv in a
            // lowered read yet; refuse rather than guess a negative-
            // operand convention.
            let (_, _) = (rec(a)?, rec(b)?);
            bail!("layout read: IntCeilDiv lowering not implemented (fail-closed)")
        }
        T::Min(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) < ({b}) ? ({a}) : ({b}))")
        }
        T::Max(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) > ({b}) ? ({a}) : ({b}))")
        }
        T::LessThanCast(a, b) => {
            format!("(({}) < ({}) ? 1LL : 0LL)", rec(a)?, rec(b)?)
        }
    })
}

/// Lower one operand's SLOT LAYOUT to C statements computing its flat
/// element read index at the current coordinates
/// `{in_prefix}0..{in_prefix}{rank-1}` (front-indexed). Returns
/// `(code, index_var)`; the statements bind `{operand}_idx` plus the
/// single final bounds trap described in the module note above. The
/// layout's own domain (its shape) must be LITERAL and equal the slot's
/// value dims — a foreign-domain layout is a planner/renderer
/// incoherence and refuses loudly.
pub fn layout_read_index(
    operand: &str,
    layout: &CudaLayout,
    slot_dims: &[usize],
    in_prefix: &str,
) -> Result<(String, String)> {
    use luminal::layouts::{MirrorLayout, SpanExpr};
    let rank = slot_dims.len();
    let idx = format!("{operand}_idx");
    let check_domain = |shape: &luminal::layouts::ShapeTerm| -> Result<()> {
        let extents: Option<Vec<usize>> = shape
            .0
            .iter()
            .map(|e| e.eval_literal().and_then(|v| usize::try_from(v).ok()))
            .collect();
        let Some(extents) = extents else {
            bail!("operand {operand}: layout has symbolic extents (no numeric codegen)");
        };
        if extents != slot_dims {
            bail!(
                "operand {operand}: layout domain {extents:?} differs from the slot's \
                 value extents {slot_dims:?} — refuse, never reinterpret"
            );
        }
        Ok(())
    };
    // (code lines, offset expr, span bound: Some(packed reach) / None)
    let (offset, span): (String, Option<String>) = match &layout.mirror {
        MirrorLayout::RightMajor(rm) => {
            check_domain(&rm.shape)?;
            let strides = strides_of(slot_dims);
            let flat = if rank == 0 {
                "0LL".to_string()
            } else {
                (0..rank)
                    .map(|axis| format!("{in_prefix}{axis} * {}LL", strides[axis]))
                    .collect::<Vec<_>>()
                    .join(" + ")
            };
            (flat, Some(format!("{}LL", numel(slot_dims))))
        }
        MirrorLayout::LeftMajor(lm) => {
            check_domain(&lm.shape)?;
            let mut strides = vec![1usize; rank];
            for axis in 1..rank {
                strides[axis] = strides[axis - 1] * slot_dims[axis - 1];
            }
            let flat = if rank == 0 {
                "0LL".to_string()
            } else {
                (0..rank)
                    .map(|axis| format!("{in_prefix}{axis} * {}LL", strides[axis]))
                    .collect::<Vec<_>>()
                    .join(" + ")
            };
            (flat, Some(format!("{}LL", numel(slot_dims))))
        }
        MirrorLayout::Strided(st) => {
            check_domain(&st.shape)?;
            let summands = st
                .chain
                .iter()
                .map(|s| lower_layout_term(s, rank, in_prefix))
                .collect::<Result<Vec<_>>>()?;
            let flat = if summands.is_empty() {
                "0LL".to_string()
            } else {
                summands.join(" + ")
            };
            // The strided span IS disclosed (SpanExpr): 1 + Σ summand at
            // the last coordinate of each axis — a literal expression
            // here (rank 0: no coordinates survive the substitution).
            let span = lower_layout_term(&st.span(), 0, in_prefix)?;
            (flat, Some(span))
        }
        MirrorLayout::ElementOffset(eo) => {
            check_domain(&eo.shape)?;
            // NO DISCLOSED REACH: an offset function alone does not say
            // how far it points (SpanExpr deliberately unimplemented) —
            // the only honest trap left is non-negativity.
            (lower_layout_term(&eo.offset, rank, in_prefix)?, None)
        }
        MirrorLayout::BitOffset(bo) => {
            check_domain(&bo.shape)?;
            let bits = lower_layout_term(&bo.offset, rank, in_prefix)?;
            let width = bo.width.0;
            // Bit form: element index = bit offset / width, with a
            // divisibility trap (a mid-element bit offset has no element
            // read). Same undisclosed-reach story as ElementOffset.
            let bits_var = format!("{operand}_bits");
            let code = format!(
                "    long long {bits_var} = {bits};\n    if ({bits_var} < 0 || ({bits_var} % {width}LL) != 0) __trap();\n    long long {idx} = {bits_var} / {width}LL;\n"
            );
            return Ok((code, idx));
        }
    };
    let mut code = format!("    long long {idx} = {offset};\n");
    match span {
        Some(span) => code.push_str(&format!(
            "    if ({idx} < 0 || {idx} >= ({span})) __trap();\n"
        )),
        None => code.push_str(&format!("    if ({idx} < 0) __trap();\n")),
    }
    Ok((code, idx))
}

/// One generated launch: entry name is always `k`; `n` is the launch
/// size (one thread per index). `scratch_bytes > 0` asks the executor
/// for a zero-initialized device scratch buffer passed as the
/// second-to-last argument (before `out`, `n`) — scatter's injectivity
/// flags use this.
#[derive(Debug)]
pub struct KernelSource {
    pub source: String,
    pub n: usize,
    pub scratch_bytes: usize,
}

impl KernelSource {
    pub(crate) fn plain(source: String, n: usize) -> Self {
        Self { source, n, scratch_bytes: 0 }
    }
}

/// An op lowers to an ordered launch SEQUENCE on one stream (stream
/// order makes multi-phase ops race-free: scatter = init-copy then
/// scattered writes).
pub struct CudaKernel {
    pub label: &'static str,
    pub op_type: TypeId,
    pub codegen: fn(&dyn BufferTensorIrOp, &CodegenCtx) -> Result<Vec<KernelSource>>,
}

fn row<T: 'static>(
    label: &'static str,
    codegen: fn(&dyn BufferTensorIrOp, &CodegenCtx) -> Result<Vec<KernelSource>>,
) -> CudaKernel {
    CudaKernel { label, op_type: TypeId::of::<T>(), codegen }
}

/// CUDA scalar type for a plan dtype. CL-1 covers the reference
/// executor's own executable set; everything else refuses loudly.
pub(crate) fn cuda_type(dtype: PlanDtype) -> Result<&'static str> {
    Ok(match dtype {
        PlanDtype::F32 => "float",
        PlanDtype::Int => "int",
        PlanDtype::Int64 => "long long",
        PlanDtype::Bool | PlanDtype::Bool8 => "unsigned char",
        other => bail!("cuda-lite CL-1 has no device type for {other:?}"),
    })
}

pub(crate) fn numel(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// `out[i] = <expr of a[i], b[i]>` over the destination's numel.
/// PROTOTYPE (Option B): each operand is read through its SLOT LAYOUT —
/// a direct (row-major, value-shaped) layout keeps the flat `a[i]`
/// (byte-identical fast path when every slot is direct); any other
/// layout switches `a[i]` to `a[a_idx]` with the layout's own offset
/// expression lowered by [`layout_read_index`]. The hop chain is NOT
/// consulted in this family.
pub(crate) fn binary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let [a, b, _dest] = ctx.operand_dtypes.as_slice() else {
        bail!("binary op expects two operands + dest, got {}", ctx.operand_dtypes.len());
    };
    let (ta, tb) = (cuda_type(*a)?, cuda_type(*b)?);
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    if all_operands_direct(ctx) {
        // The flat fast path, byte-identical to pre-Phase-4 codegen.
        let source = format!(
            r#"extern "C" __global__ void k(const {ta}* a, const {tb}* b, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = {expr};
}}"#
        );
        return Ok(vec![KernelSource::plain(source, n)]);
    }
    let sig = format!("const {ta}* a, const {tb}* b");
    strided_elementwise(ctx, expr, &["a", "b"], &sig, to)
}

/// Every operand slot's layout is the direct read for its dims (the
/// flat-fast-path / write-fence discriminator — Option B keys this on
/// the LAYOUT, never on hop presence).
fn all_operands_direct(ctx: &CodegenCtx) -> bool {
    ctx.operand_layouts
        .iter()
        .zip(&ctx.operand_dims)
        .all(|(layout, dims)| layout_is_direct(layout, dims))
}

/// `out[i] = <expr of a[i]>` over the destination's numel. A non-direct
/// slot layout switches `a[i]` to the expression read `a[a_idx]` (see
/// [`binary`]).
pub(crate) fn unary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    if all_operands_direct(ctx) {
        // The flat fast path, byte-identical to pre-Phase-4 codegen.
        let source = format!(
            r#"extern "C" __global__ void k(const {ta}* a, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = {expr};
}}"#
        );
        return Ok(vec![KernelSource::plain(source, n)]);
    }
    let sig = format!("const {ta}* a");
    strided_elementwise(ctx, expr, &["a"], &sig, to)
}

// RULING 2026-08-27: the Phase-5 `copy_through_fold` lowering is DELETED —
// a BufferCopy is only ever a dumb whole-buffer memcpy. A copy
// materialized into a specific layout is a LayoutTensor candidate in the
// e-graph (the materialize kernel), discovered via search, never a copy
// mode.

/// The strided elementwise form — PROTOTYPE (Option B): identical
/// launch geometry to the flat template (one thread per OUT element),
/// but every operand whose SLOT LAYOUT is not the direct read is read
/// at `name[f(out_coords)]`, where `f` is the layout's own offset
/// expression lowered by [`layout_read_index`] over the out-coordinate
/// prelude — the hop chain is dead in this family. Contract with the
/// op-module exprs: the template expr reads operand `name` exactly as
/// the literal token `name[i]`, which is rewritten here to
/// `name[{name}_idx]`.
///
/// The DPS dest slot (the operand slot after the named reads) must stay
/// direct: strided WRITES are CL-4b and refuse loudly.
fn strided_elementwise(
    ctx: &CodegenCtx,
    expr: &str,
    names: &[&str],
    sig: &str,
    to: &str,
) -> Result<Vec<KernelSource>> {
    let out_dims = &ctx.dest_dims[0];
    let n = numel(out_dims);
    for (k, layout) in ctx.operand_layouts.iter().enumerate() {
        if k >= names.len() && !layout_is_direct(layout, &ctx.operand_dims[k]) {
            bail!(
                "dest operand slot {k} carries a non-direct layout: strided writes \
                 are not lowered (dests stay dense out-of-place; CL-4b)"
            );
        }
    }
    let mut chains = String::new();
    let mut rendered = expr.to_string();
    for (k, name) in names.iter().enumerate() {
        if layout_is_direct(&ctx.operand_layouts[k], &ctx.operand_dims[k]) {
            continue;
        }
        // An elementwise operand VALUE spans the out iteration space —
        // its layout's domain is the slot's own coordinates, which must
        // therefore be the out coordinates. A mismatch means the elected
        // layout has a different geometry than this template iterates:
        // refuse, never reinterpret.
        if &ctx.operand_dims[k] != out_dims {
            bail!(
                "operand {name} value extents {:?} differ from dest extents {:?} \
                 under a non-direct layout — elementwise templates iterate the dest",
                ctx.operand_dims[k],
                out_dims
            );
        }
        let (code, idx) =
            layout_read_index(name, &ctx.operand_layouts[k], out_dims, "c")?;
        chains.push_str(&code);
        let flat = format!("{name}[i]");
        if !rendered.contains(&flat) {
            bail!("template expr `{expr}` has no `{flat}` token to rewrite for a composed operand");
        }
        rendered = rendered.replace(&flat, &format!("{name}[{idx}]"));
    }
    let prelude = coord_prelude(out_dims);
    let source = format!(
        r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}{chains}    out[i] = {rendered};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Loud refusal for codegen bodies that do not lower expression reads
/// (iota, constant — dest-only signatures): an operand arriving with a
/// non-direct layout through a kernel that would index it flat is
/// silent mistranslation — bail instead.
pub(crate) fn require_flat_operands(label: &str, ctx: &CodegenCtx) -> Result<()> {
    for k in 0..ctx.operand_layouts.len() {
        if ctx.non_direct_operand(k).is_some() {
            bail!(
                "{label}: operand {k} carries a non-direct layout this kernel does \
                 not lower (fail-closed, never identity)"
            );
        }
    }
    Ok(())
}

/// Axis reduction, axis zero-based FROM THE END (the DPS convention).
/// One thread per output element; the reduced extent is looped.
pub(crate) fn reduce(
    ctx: &CodegenCtx,
    axis_from_end: usize,
    init: &str,
    fold: &str,
) -> Result<Vec<KernelSource>> {
    let in_dims = &ctx.operand_dims[0];
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    if axis_from_end >= in_dims.len() {
        bail!("reduce axis {axis_from_end} out of rank {}", in_dims.len());
    }
    let axis = in_dims.len() - 1 - axis_from_end;
    let extent = in_dims[axis];
    // Row-major strides of the input; the output walks the same dims
    // with the reduced axis removed.
    let inner: usize = in_dims[axis + 1..].iter().product();
    let outer: usize = in_dims[..axis].iter().product();
    let n = outer * inner;
    if (0..ctx.operand_layouts.len()).all(|k| ctx.non_direct_operand(k).is_none()) {
        // The flat fast path, byte-identical to pre-Phase-4 codegen.
        let source = format!(
            r#"extern "C" __global__ void k(const {ta}* a, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long outer = i / {inner}ULL;
    unsigned long long inner = i % {inner}ULL;
    {ta} acc = {init};
    for (unsigned long long r = 0; r < {extent}ULL; ++r) {{
        {ta} v = a[outer * {extent}ULL * {inner}ULL + r * {inner}ULL + inner];
        acc = {fold};
    }}
    out[i] = acc;
}}"#
        );
        return Ok(vec![KernelSource::plain(source, n)]);
    }
    // Strided read: the input slot's layout is not the direct form, so
    // its flat address is replaced by the layout's own offset expression
    // evaluated at the INPUT VALUE's coordinates `c0..c{rank-1}` —
    // rebuilt here from the outer/inner decomposition plus the loop's
    // own `r` at the reduced axis. The dest slot must stay direct
    // (CL-4b, no strided writes).
    for k in 1..ctx.operand_layouts.len() {
        if ctx.non_direct_operand(k).is_some() {
            bail!(
                "dest operand slot {k} carries a non-direct layout: strided writes \
                 are not lowered (dests stay dense out-of-place; CL-4b)"
            );
        }
    }
    let layout = ctx
        .non_direct_operand(0)
        .expect("reduce strided path entered with a non-direct input layout");
    // Coordinates OUTSIDE the reduced axis are loop-invariant: decompose
    // `inner` then `outer` (row-major, innermost axis first) before the
    // loop; `c{axis}` is the loop variable.
    let mut coords = String::from("    unsigned long long rem = inner;\n");
    for ax in ((axis + 1)..in_dims.len()).rev() {
        coords.push_str(&format!(
            "    long long c{ax} = (long long)(rem % {d}ULL); rem /= {d}ULL;\n",
            d = in_dims[ax]
        ));
    }
    coords.push_str("    rem = outer;\n");
    for ax in (0..axis).rev() {
        coords.push_str(&format!(
            "    long long c{ax} = (long long)(rem % {d}ULL); rem /= {d}ULL;\n",
            d = in_dims[ax]
        ));
    }
    let (chain, idx) = layout_read_index("a", layout, in_dims, "c")?;
    // Re-indent the chain into the loop body.
    let chain = chain.replace("    ", "        ");
    let source = format!(
        r#"extern "C" __global__ void k(const {ta}* a, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long outer = i / {inner}ULL;
    unsigned long long inner = i % {inner}ULL;
{coords}    {ta} acc = {init};
    for (unsigned long long r = 0; r < {extent}ULL; ++r) {{
        long long c{axis} = (long long)r;
{chain}        {ta} v = a[{idx}];
        acc = {fold};
    }}
    out[i] = acc;
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Lower an [`IotaExpr`] to a C expression over `long long`, with OUT
/// coordinates available as `c0..c{rank-1}` (front-indexed, matching
/// the reference evaluator: `Coord(axis_from_end)` reads
/// `c[rank-1-axis_from_end]`).
pub(crate) fn lower_expr(expr: &IotaExpr, rank: usize) -> Result<String> {
    lower_expr_pref(expr, rank, "c")
}

/// [`lower_expr`] with a caller-chosen coordinate variable prefix:
/// `Coord(axis_from_end)` reads `{prefix}{rank-1-axis_from_end}`. The
/// composed-access chain uses this to evaluate hop `k+1`'s entries at
/// hop `k`'s outputs (`{operand}_h{k}_{m}`) instead of `c{m}`.
pub(crate) fn lower_expr_pref(expr: &IotaExpr, rank: usize, prefix: &str) -> Result<String> {
    let rec = |e: &IotaExpr| lower_expr_pref(e, rank, prefix);
    Ok(match expr {
        IotaExpr::Lit(v) => format!("{v}LL"),
        IotaExpr::Coord(axis_from_end) => {
            if *axis_from_end >= rank {
                bail!("coordinate axis {axis_from_end} out of rank {rank}");
            }
            format!("{prefix}{}", rank - 1 - axis_from_end)
        }
        IotaExpr::Add(a, b) => format!("({} + {})", rec(a)?, rec(b)?),
        IotaExpr::Mul(a, b) => format!("({} * {})", rec(a)?, rec(b)?),
        IotaExpr::TruncDiv(a, b) => {
            format!("({} / {})", rec(a)?, rec(b)?)
        }
        IotaExpr::TruncRem(a, b) => {
            format!("({} % {})", rec(a)?, rec(b)?)
        }
        IotaExpr::Min(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) < ({b}) ? ({a}) : ({b}))")
        }
        IotaExpr::Max(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) > ({b}) ? ({a}) : ({b}))")
        }
        IotaExpr::LessThanCast(a, b) => {
            format!("(({}) < ({}) ? 1LL : 0LL)", rec(a)?, rec(b)?)
        }
    })
}

/// The row-major coordinate prelude: decompose flat `i` into
/// `c0..c{rank-1}` over `dims` (front-indexed).
pub(crate) fn coord_prelude(dims: &[usize]) -> String {
    let mut out = String::from("    unsigned long long rem = i;\n");
    for axis in (0..dims.len()).rev() {
        out.push_str(&format!(
            "    long long c{axis} = (long long)(rem % {}ULL); rem /= {}ULL;\n",
            dims[axis], dims[axis]
        ));
    }
    out
}

/// Row-major strides for dims.
pub(crate) fn strides_of(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * dims[k + 1];
    }
    strides
}

/// The table. Alloc/free are handled structurally by the executor
/// (real device alloc/free), not by codegen rows.
pub fn cuda_kernels() -> &'static [CudaKernel] {
    use crate::ops;
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<CudaKernel>> = OnceLock::new();
    TABLE.get_or_init(|| {
        vec![
            row::<ops::add::AddFunctionalDps>("AddFunctional", ops::add::codegen),
            row::<ops::mul::MulFunctionalDps>("MulFunctional", ops::mul::codegen),
            row::<ops::div::DivFunctionalDps>("DivFunctional", ops::div::codegen),
            row::<ops::trunc_div::TruncDivFunctionalDps>(
                "TruncDivFunctional",
                ops::trunc_div::codegen,
            ),
            row::<ops::trunc_rem::TruncRemFunctionalDps>(
                "TruncRemFunctional",
                ops::trunc_rem::codegen,
            ),
            row::<ops::modulo::ModFunctionalDps>("ModFunctional", ops::modulo::codegen),
            row::<ops::less_than::LessThanDps>("LessThan", ops::less_than::codegen),
            row::<ops::sqrt::SqrtFunctionalDps>("SqrtFunctional", ops::sqrt::codegen),
            row::<ops::exp::ExpFunctionalDps>("ExpFunctional", ops::exp::codegen),
            row::<ops::exp2::Exp2FunctionalDps>("Exp2Functional", ops::exp2::codegen),
            row::<ops::log2::Log2FunctionalDps>("Log2Functional", ops::log2::codegen),
            row::<ops::sin::SinFunctionalDps>("SinFunctional", ops::sin::codegen),
            row::<ops::recip::RecipFunctionalDps>("RecipFunctional", ops::recip::codegen),
            row::<ops::cast::CastDps>("Cast", ops::cast::codegen),
            row::<ops::constant::ConstantDps>("Constant", ops::constant::codegen),
            row::<ops::materialize_layout_copy::MaterializeLayoutCopyDps>(
                "Copy",
                ops::materialize_layout_copy::codegen,
            ),
            row::<ops::reduce_sum::ReduceSumDps>("ReduceSum", ops::reduce_sum::codegen),
            row::<ops::reduce_max::ReduceMaxDps>("ReduceMax", ops::reduce_max::codegen),
            row::<ops::iota::IotaDps>("Iota", ops::iota::codegen),
            row::<ops::index_map_apply_materialize::IndexMapApplyMaterializeDps>(
                "IndexMapApplyMaterialize",
                ops::index_map_apply_materialize::codegen,
            ),
            row::<ops::gather::GatherDps>("Gather", ops::gather::codegen),
            row::<ops::scatter::ScatterFunctionalDps>(
                "ScatterFunctional",
                ops::scatter::codegen,
            ),
        ]
    })
}

/// Codegen lookup by concrete op type.
pub fn codegen_for(op: &dyn BufferTensorIrOp) -> Option<&'static CudaKernel> {
    let ty = op.as_any().type_id();
    cuda_kernels().iter().find(|k| k.op_type == ty)
}
