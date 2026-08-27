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
use luminal::bufferize::{ComposedAccess, SlotDescriptor};
use luminal::dtype::PlanDtype;
use luminal::index_expr::IotaExpr;
use std::any::TypeId;

/// Geometry + typing for one compute node, in plan order: operands
/// (destination-last, the DPS convention), then destinations again as
/// the write set. Dims come from the node's own [`SlotDescriptor`]s (M4
/// Phase 3) — per-slot VALUE geometry, which equals the shared buffer
/// table's numbers while no view is electable on this backend (the
/// string-identity pin in `tests/codegen_identity.rs`).
#[derive(Debug)]
pub struct CodegenCtx {
    pub operand_dims: Vec<Vec<usize>>,
    pub operand_dtypes: Vec<PlanDtype>,
    pub dest_dims: Vec<Vec<usize>>,
    pub dest_dtypes: Vec<PlanDtype>,
    /// Per-operand composed view access, parallel to `operand_dims` —
    /// `Some` iff folded views stand between the slot's value and its
    /// buffer. Phase 4: the elementwise/reduce templates lower a `Some`
    /// operand to `parent[f(out_coords)]` (see [`composed_read_index`]);
    /// every other codegen body refuses loudly via
    /// [`require_flat_operands`]. On this backend it is all-`None` on
    /// real plans today (views are not electable on real backends), so
    /// real-plan codegen strings are unchanged — the flat `a[i]` fast
    /// path is byte-identical (pinned in `tests/codegen_identity.rs`).
    pub composed_access: Vec<Option<ComposedAccess>>,
}

impl CodegenCtx {
    /// Build codegen geometry from the compute node's own slot
    /// descriptors — never the shared buffer table (Phase 3 pin). Loud
    /// on missing numerics, mirroring the executor's None-dims bail.
    pub fn from_descriptors(
        label: &str,
        operand_info: &[SlotDescriptor],
        result_info: &[SlotDescriptor],
    ) -> Result<Self> {
        // Strided WRITES are not lowered (CL-4b territory): destinations
        // stay dense out-of-place, so a composed access on a RESULT slot
        // is a loud refusal at the single codegen entry point — never a
        // silently dense write through a view.
        for (k, slot) in result_info.iter().enumerate() {
            if slot.composed_access.is_some() {
                bail!(
                    "{label} result {k} carries a composed access: strided writes \
                     are not lowered (dests stay dense out-of-place; CL-4b)"
                );
            }
        }
        let dims_of = |slot: &SlotDescriptor, role: &str| -> Result<Vec<usize>> {
            let dims = slot
                .dims
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("{label} {role} lacks geometry"))?;
            Ok(dims.iter().map(|&d| usize::try_from(d).unwrap_or(0)).collect())
        };
        let dtype_of = |slot: &SlotDescriptor, role: &str| -> Result<PlanDtype> {
            slot.dtype
                .ok_or_else(|| anyhow::anyhow!("{label} {role} lacks dtype"))
        };
        Ok(CodegenCtx {
            operand_dims: operand_info
                .iter()
                .map(|s| dims_of(s, "operand"))
                .collect::<Result<_>>()?,
            operand_dtypes: operand_info
                .iter()
                .map(|s| dtype_of(s, "operand"))
                .collect::<Result<_>>()?,
            dest_dims: result_info
                .iter()
                .map(|s| dims_of(s, "dest"))
                .collect::<Result<_>>()?,
            dest_dtypes: result_info
                .iter()
                .map(|s| dtype_of(s, "dest"))
                .collect::<Result<_>>()?,
            composed_access: operand_info.iter().map(|s| s.composed_access.clone()).collect(),
        })
    }
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

/// `out[i] = <expr of a[i], b[i]>` over the destination's numel. An
/// operand carrying a [`ComposedAccess`] is read through its folded-view
/// chain instead: `a[i]` becomes `a[a_idx]` with the chain lowered by
/// [`composed_read_index`] (Phase 4 strided reads).
pub(crate) fn binary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let [a, b, _dest] = ctx.operand_dtypes.as_slice() else {
        bail!("binary op expects two operands + dest, got {}", ctx.operand_dtypes.len());
    };
    let (ta, tb) = (cuda_type(*a)?, cuda_type(*b)?);
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    if ctx.composed_access.iter().all(Option::is_none) {
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

/// `out[i] = <expr of a[i]>` over the destination's numel. A composed
/// access on the operand switches `a[i]` to the strided read `a[a_idx]`
/// (see [`binary`]).
pub(crate) fn unary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    if ctx.composed_access.iter().all(Option::is_none) {
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

/// M4 Phase 5: the FOLDED-COPY lowering. A `BufferCopy` whose value
/// resides in `src` through folded views is a materializing read, not a
/// byte-move: `out[i] = src[chain(coords(i))]` over the VALUE's own
/// extents (dst geometry, per the writer-identity dims join). Reuses
/// the unary template — the copy is exactly `a[i]` with a composed
/// operand — so index lowering and the per-axis bounds traps are the
/// same code the compute kernels use.
pub fn copy_through_fold(
    value_dims: &[usize],
    dtype: PlanDtype,
    access: &ComposedAccess,
) -> Result<Vec<KernelSource>> {
    let ctx = CodegenCtx {
        // Mirror the DPS slot shape (operand + dest) the templates expect.
        operand_dims: vec![value_dims.to_vec(), value_dims.to_vec()],
        operand_dtypes: vec![dtype, dtype],
        dest_dims: vec![value_dims.to_vec()],
        dest_dtypes: vec![dtype],
        composed_access: vec![Some(access.clone()), None],
    };
    unary(&ctx, "a[i]")
}

/// The strided elementwise form (Phase 4): identical launch geometry to
/// the flat template — one thread per OUT element — but every operand
/// carrying a [`ComposedAccess`] is read at `name[f(out_coords)]`,
/// where `f` is its hop chain lowered by [`composed_read_index`] over
/// the out-coordinate prelude. Contract with the op-module exprs: the
/// template expr reads operand `name` exactly as the literal token
/// `name[i]`, which is rewritten here to `name[{name}_idx]`.
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
    for (k, access) in ctx.composed_access.iter().enumerate() {
        if k >= names.len() && access.is_some() {
            bail!(
                "dest operand slot {k} carries a composed access: strided writes \
                 are not lowered (dests stay dense out-of-place; CL-4b)"
            );
        }
    }
    let mut chains = String::new();
    let mut rendered = expr.to_string();
    for (k, name) in names.iter().enumerate() {
        let Some(access) = ctx.composed_access.get(k).and_then(|a| a.as_ref()) else {
            continue;
        };
        // An elementwise operand VALUE spans the out iteration space —
        // hop 0's entries are functions of the slot's own coordinates,
        // which must therefore be the out coordinates. A mismatch means
        // the fold recorded a different geometry than this template
        // iterates: refuse, never reinterpret.
        if &ctx.operand_dims[k] != out_dims {
            bail!(
                "operand {name} value extents {:?} differ from dest extents {:?} \
                 under composed access — elementwise templates iterate the dest",
                ctx.operand_dims[k],
                out_dims
            );
        }
        let (code, idx) = composed_read_index(name, access, out_dims.len())?;
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

/// Lower one operand's [`ComposedAccess`] chain to C statements
/// computing its flat read index at the CURRENT coordinates
/// `c0..c{coord_rank-1}` (front-indexed, from [`coord_prelude`] or the
/// reduce template's coordinate rebuild). Returns `(code, index_var)`:
/// the statements bind `{operand}_h{k}_{m}` per hop `k` / parent axis
/// `m` — hop 0 evaluated at the slot's own coordinates, each hop's
/// outputs feeding the next hop's [`IotaExpr::Coord`]s (codegen-time
/// composition) — with a per-axis bounds `__trap()` exactly like the
/// materialize kernel, and finally `{operand}_idx`, the row-major flat
/// offset into the LAST hop's parent (the residence actually read).
///
/// Fail-closed at every hop: `entries: None` (map beyond the parsed
/// subset) and symbolic/non-positive parent extents bail loudly —
/// never treated as identity.
pub(crate) fn composed_read_index(
    operand: &str,
    access: &ComposedAccess,
    coord_rank: usize,
) -> Result<(String, String)> {
    if access.hops.is_empty() {
        bail!("operand {operand}: composed access with zero hops");
    }
    let mut code = String::new();
    let mut in_rank = coord_rank;
    let mut in_prefix = "c".to_string();
    let mut last_parent: Vec<usize> = Vec::new();
    for (h, hop) in access.hops.iter().enumerate() {
        let Some(entries) = &hop.entries else {
            bail!(
                "operand {operand} hop {h}: index map beyond the parsed expression \
                 subset (fail-closed, never identity)"
            );
        };
        let Some(parent_dims) = &hop.parent_dims else {
            bail!("operand {operand} hop {h}: symbolic parent extents");
        };
        if entries.len() != parent_dims.len() {
            bail!(
                "operand {operand} hop {h}: {} map entries vs parent rank {}",
                entries.len(),
                parent_dims.len()
            );
        }
        let parent: Vec<usize> = parent_dims
            .iter()
            .map(|&d| {
                usize::try_from(d).ok().filter(|&d| d > 0).ok_or_else(|| {
                    anyhow::anyhow!("operand {operand} hop {h}: non-positive parent extent {d}")
                })
            })
            .collect::<Result<_>>()?;
        for (m, entry) in entries.iter().enumerate() {
            let value = lower_expr_pref(entry, in_rank, &in_prefix)?;
            let var = format!("{operand}_h{h}_{m}");
            code.push_str(&format!(
                "    long long {var} = {value};\n    if ({var} < 0 || {var} >= {ext}LL) __trap();\n",
                ext = parent[m]
            ));
        }
        in_rank = parent.len();
        in_prefix = format!("{operand}_h{h}_");
        last_parent = parent;
    }
    let strides = strides_of(&last_parent);
    let last_hop = access.hops.len() - 1;
    let idx = format!("{operand}_idx");
    let flat = if last_parent.is_empty() {
        "0".to_string()
    } else {
        (0..last_parent.len())
            .map(|m| format!("{operand}_h{last_hop}_{m} * {}LL", strides[m]))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    code.push_str(&format!("    long long {idx} = {flat};\n"));
    Ok((code, idx))
}

/// Loud refusal for codegen bodies that do not lower composed access
/// (gather, scatter, iota, constant, materialize): an operand arriving
/// with folded-view addressing through a kernel that would index it
/// flat is silent mistranslation — bail instead.
pub(crate) fn require_flat_operands(label: &str, ctx: &CodegenCtx) -> Result<()> {
    for (k, access) in ctx.composed_access.iter().enumerate() {
        if access.is_some() {
            bail!(
                "{label}: operand {k} carries a composed access this kernel does \
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
    if ctx.composed_access.iter().all(Option::is_none) {
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
    // Phase 4 strided read: the input slot carries a ComposedAccess, so
    // its flat address is replaced by the hop chain evaluated at the
    // INPUT VALUE's coordinates `c0..c{rank-1}` — rebuilt here from the
    // outer/inner decomposition plus the loop's own `r` at the reduced
    // axis. The dest slot must stay direct (CL-4b, no strided writes).
    for (k, access) in ctx.composed_access.iter().enumerate() {
        if k >= 1 && access.is_some() {
            bail!(
                "dest operand slot {k} carries a composed access: strided writes \
                 are not lowered (dests stay dense out-of-place; CL-4b)"
            );
        }
    }
    let access = ctx.composed_access[0]
        .as_ref()
        .expect("reduce strided path entered with a composed input access");
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
    let (chain, idx) = composed_read_index("a", access, in_dims.len())?;
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
