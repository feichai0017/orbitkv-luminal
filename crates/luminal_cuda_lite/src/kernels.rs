//! The CUDA codegen table: one row per executable op type, keyed by the
//! concrete DPS struct's `TypeId` exactly like the reference kernel
//! registry (labels repeat across functional/DPS forms; types do not).
//!
//! A row's `codegen` turns (op instance, buffer geometry) into a
//! self-contained CUDA source string — dense row-major, one thread per
//! output element, geometry baked as literals. Generation is pure and
//! host-side (snapshot-testable without a device); NVRTC compilation
//! and launch live in the `device` module.
//!
//! CL-1 coverage: the elementwise family + constant + cast + copy +
//! axis reductions. Expression-carrying ops (iota, materialize,
//! gather, scatter) need the IotaExpr-to-CUDA lowering and join the
//! table in CL-1b; the allow list stays honest by construction —
//! search can only elect what this table generates.

use anyhow::{bail, Context, Result};
use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::dtype::PlanDtype;
use luminal::reference::ops as rops;
use std::any::TypeId;

/// Geometry + typing for one compute node, in plan order: operands
/// (destination-last, the DPS convention), then destinations again as
/// the write set. Dims come from the plan's buffer annotations — the
/// same numbers the reference executor sizes with.
pub struct CodegenCtx {
    pub operand_dims: Vec<Vec<usize>>,
    pub operand_dtypes: Vec<PlanDtype>,
    pub dest_dims: Vec<Vec<usize>>,
    pub dest_dtypes: Vec<PlanDtype>,
}

/// One generated launch: entry name is always `k`; `n` is the launch
/// size (one thread per index). `scratch_bytes > 0` asks the executor
/// for a zero-initialized device scratch buffer passed as the
/// second-to-last argument (before `out`, `n`) — scatter's injectivity
/// flags use this.
pub struct KernelSource {
    pub source: String,
    pub n: usize,
    pub scratch_bytes: usize,
}

impl KernelSource {
    fn plain(source: String, n: usize) -> Self {
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
fn cuda_type(dtype: PlanDtype) -> Result<&'static str> {
    Ok(match dtype {
        PlanDtype::F32 => "float",
        PlanDtype::Int => "int",
        PlanDtype::Int64 => "long long",
        PlanDtype::Bool | PlanDtype::Bool8 => "unsigned char",
        other => bail!("cuda-lite CL-1 has no device type for {other:?}"),
    })
}

fn numel(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// `out[i] = <expr of a[i], b[i]>` over the destination's numel.
fn binary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let [a, b, _dest] = ctx.operand_dtypes.as_slice() else {
        bail!("binary op expects two operands + dest, got {}", ctx.operand_dtypes.len());
    };
    let (ta, tb) = (cuda_type(*a)?, cuda_type(*b)?);
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    let source = format!(
        r#"extern "C" __global__ void k(const {ta}* a, const {tb}* b, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = {expr};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// `out[i] = <expr of a[i]>` over the destination's numel.
fn unary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    let source = format!(
        r#"extern "C" __global__ void k(const {ta}* a, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = {expr};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

fn gen_add(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "a[i] + b[i]")
}
fn gen_mul(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "a[i] * b[i]")
}
fn gen_div(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "a[i] / b[i]")
}
fn gen_trunc_div(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "a[i] / b[i]") // integer division in C truncates toward zero
}
fn gen_trunc_rem(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "a[i] % b[i]") // C remainder carries the dividend's sign
}
fn gen_mod(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    // Float mod mirrors the reference kernel's fmodf semantics.
    binary(ctx, "fmodf(a[i], b[i])")
}
fn gen_less_than(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    binary(ctx, "(a[i] < b[i]) ? 1 : 0")
}
fn gen_sqrt(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "sqrtf(a[i])")
}
fn gen_exp(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "expf(a[i])")
}
fn gen_exp2(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "exp2f(a[i])")
}
fn gen_log2(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "log2f(a[i])")
}
fn gen_sin(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "sinf(a[i])")
}
fn gen_recip(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "1.0f / a[i]")
}
fn gen_copy(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    unary(ctx, "a[i]")
}

fn gen_cast(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let to = cuda_type(ctx.dest_dtypes[0])?;
    unary(ctx, &format!("({to})a[i]"))
}

fn gen_constant(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(constant) = op.as_any().downcast_ref::<rops::ConstantDps>() else {
        bail!("constant codegen reached with a non-Constant op");
    };
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    let value = constant.value;
    let source = format!(
        r#"extern "C" __global__ void k({to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ({to}){value};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Axis reduction, axis zero-based FROM THE END (the DPS convention).
/// One thread per output element; the reduced extent is looped.
fn reduce(ctx: &CodegenCtx, axis_from_end: usize, init: &str, fold: &str) -> Result<Vec<KernelSource>> {
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
    Ok(vec![KernelSource::plain(source, n)])
}

fn gen_reduce_sum(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(r) = op.as_any().downcast_ref::<rops::ReduceSumDps>() else {
        bail!("reduce_sum codegen reached with a non-ReduceSum op");
    };
    let axis = usize::try_from(r.axis).context("negative reduce axis")?;
    reduce(ctx, axis, "0", "acc + v")
}

fn gen_reduce_max(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(r) = op.as_any().downcast_ref::<rops::ReduceMaxDps>() else {
        bail!("reduce_max codegen reached with a non-ReduceMax op");
    };
    let axis = usize::try_from(r.axis).context("negative reduce axis")?;
    reduce(ctx, axis, "-INFINITY", "v > acc ? v : acc")
}

/// The table. Alloc/free are handled structurally by the executor
/// (real device alloc/free), not by codegen rows.
pub fn cuda_kernels() -> &'static [CudaKernel] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<CudaKernel>> = OnceLock::new();
    TABLE.get_or_init(|| {
        vec![
            row::<rops::AddFunctionalDps>("AddFunctional", gen_add),
            row::<rops::MulFunctionalDps>("MulFunctional", gen_mul),
            row::<rops::DivFunctionalDps>("DivFunctional", gen_div),
            row::<rops::TruncDivFunctionalDps>("TruncDivFunctional", gen_trunc_div),
            row::<rops::TruncRemFunctionalDps>("TruncRemFunctional", gen_trunc_rem),
            row::<rops::ModFunctionalDps>("ModFunctional", gen_mod),
            row::<rops::LessThanDps>("LessThan", gen_less_than),
            row::<rops::SqrtFunctionalDps>("SqrtFunctional", gen_sqrt),
            row::<rops::ExpFunctionalDps>("ExpFunctional", gen_exp),
            row::<rops::Exp2FunctionalDps>("Exp2Functional", gen_exp2),
            row::<rops::Log2FunctionalDps>("Log2Functional", gen_log2),
            row::<rops::SinFunctionalDps>("SinFunctional", gen_sin),
            row::<rops::RecipFunctionalDps>("RecipFunctional", gen_recip),
            row::<rops::CastDps>("Cast", gen_cast),
            row::<rops::ConstantDps>("Constant", gen_constant),
            row::<rops::MaterializeLayoutCopyDps>("Copy", gen_copy),
            row::<rops::ReduceSumDps>("ReduceSum", gen_reduce_sum),
            row::<rops::ReduceMaxDps>("ReduceMax", gen_reduce_max),
            row::<rops::IotaDps>("Iota", gen_iota),
            row::<rops::IndexMapApplyMaterializeDps>(
                "IndexMapApplyMaterialize",
                gen_materialize,
            ),
            row::<rops::GatherDps>("Gather", gen_gather),
            row::<rops::ScatterFunctionalDps>("ScatterFunctional", gen_scatter),
        ]
    })
}

/// Codegen lookup by concrete op type.
pub fn codegen_for(op: &dyn BufferTensorIrOp) -> Option<&'static CudaKernel> {
    let ty = op.as_any().type_id();
    cuda_kernels().iter().find(|k| k.op_type == ty)
}

// ===================== CL-1b: expression-carrying ops =====================

use luminal::reference::ops::IotaExpr;

/// Lower an [`IotaExpr`] to a C expression over `long long`, with OUT
/// coordinates available as `c0..c{rank-1}` (front-indexed, matching
/// the reference evaluator: `Coord(axis_from_end)` reads
/// `c[rank-1-axis_from_end]`).
fn lower_expr(expr: &IotaExpr, rank: usize) -> Result<String> {
    Ok(match expr {
        IotaExpr::Lit(v) => format!("{v}LL"),
        IotaExpr::Coord(axis_from_end) => {
            if *axis_from_end >= rank {
                bail!("coordinate axis {axis_from_end} out of rank {rank}");
            }
            format!("c{}", rank - 1 - axis_from_end)
        }
        IotaExpr::Add(a, b) => format!("({} + {})", lower_expr(a, rank)?, lower_expr(b, rank)?),
        IotaExpr::Mul(a, b) => format!("({} * {})", lower_expr(a, rank)?, lower_expr(b, rank)?),
        IotaExpr::TruncDiv(a, b) => {
            format!("({} / {})", lower_expr(a, rank)?, lower_expr(b, rank)?)
        }
        IotaExpr::TruncRem(a, b) => {
            format!("({} % {})", lower_expr(a, rank)?, lower_expr(b, rank)?)
        }
        IotaExpr::Min(a, b) => {
            let (a, b) = (lower_expr(a, rank)?, lower_expr(b, rank)?);
            format!("(({a}) < ({b}) ? ({a}) : ({b}))")
        }
        IotaExpr::Max(a, b) => {
            let (a, b) = (lower_expr(a, rank)?, lower_expr(b, rank)?);
            format!("(({a}) > ({b}) ? ({a}) : ({b}))")
        }
        IotaExpr::LessThanCast(a, b) => {
            format!("(({}) < ({}) ? 1LL : 0LL)", lower_expr(a, rank)?, lower_expr(b, rank)?)
        }
    })
}

/// The row-major coordinate prelude: decompose flat `i` into
/// `c0..c{rank-1}` over `dims` (front-indexed).
fn coord_prelude(dims: &[usize]) -> String {
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
fn strides_of(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * dims[k + 1];
    }
    strides
}

fn gen_iota(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(iota) = op.as_any().downcast_ref::<rops::IotaDps>() else {
        bail!("iota codegen reached with a non-Iota op");
    };
    let Some(expr) = &iota.expr else {
        bail!("iota beyond the parsed expression subset (fail-closed, as the reference)");
    };
    let out_dims = &ctx.dest_dims[0];
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(out_dims);
    let prelude = coord_prelude(out_dims);
    let value = lower_expr(expr, out_dims.len())?;
    let source = format!(
        r#"extern "C" __global__ void k({to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}    out[i] = ({to})({value});
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

fn gen_materialize(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(mat) = op.as_any().downcast_ref::<rops::IndexMapApplyMaterializeDps>() else {
        bail!("materialize codegen reached with a non-Materialize op");
    };
    let Some(entries) = &mat.entries else {
        bail!("index map beyond the parsed expression subset (fail-closed, as the reference)");
    };
    let parent_dims = &ctx.operand_dims[0];
    let out_dims = &ctx.operand_dims[1];
    if entries.len() != parent_dims.len() {
        bail!("index map arity {} vs parent rank {}", entries.len(), parent_dims.len());
    }
    let t = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(out_dims);
    let prelude = coord_prelude(out_dims);
    let parent_strides = strides_of(parent_dims);
    let mut body = String::from("    long long pflat = 0;\n    long long idx;\n");
    for (k, entry) in entries.iter().enumerate() {
        let value = lower_expr(entry, out_dims.len())?;
        body.push_str(&format!(
            "    idx = {value};\n    if (idx < 0 || idx >= {ext}LL) __trap();\n    pflat += idx * {stride}LL;\n",
            ext = parent_dims[k],
            stride = parent_strides[k]
        ));
    }
    let source = format!(
        r#"extern "C" __global__ void k(const {t}* parent, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}{body}    out[i] = parent[pflat];
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

fn gen_gather(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(gather) = op.as_any().downcast_ref::<rops::GatherDps>() else {
        bail!("gather codegen reached with a non-Gather op");
    };
    let rank = gather.rank;
    let data_dims = &ctx.operand_dims[0];
    if data_dims.len() != rank {
        bail!("gather data rank {} vs op rank {rank}", data_dims.len());
    }
    let t = cuda_type(ctx.operand_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    let strides = strides_of(data_dims);
    // Coordinate operands are int columns indexed by the OUT flat index.
    let mut sig = format!("const {t}* data");
    let mut body = String::from("    long long flat = 0;\n    long long coord;\n");
    for axis in 0..rank {
        sig.push_str(&format!(", const int* coord{axis}"));
        body.push_str(&format!(
            "    coord = (long long)coord{axis}[i];\n    if (coord < 0 || coord >= {ext}LL) __trap();\n    flat += coord * {stride}LL;\n",
            ext = data_dims[axis],
            stride = strides[axis]
        ));
    }
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let source = format!(
        r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{body}    out[i] = data[flat];
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

fn gen_scatter(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(scatter) = op.as_any().downcast_ref::<rops::ScatterFunctionalDps>() else {
        bail!("scatter codegen reached with a non-Scatter op");
    };
    let rank = scatter.rank;
    let init_dims = &ctx.operand_dims[0];
    if init_dims.len() != rank {
        bail!("scatter init rank {} vs op rank {rank}", init_dims.len());
    }
    let t = cuda_type(ctx.operand_dtypes[0])?;
    let dest_n = numel(&ctx.dest_dims[0]);
    let src_n = numel(&ctx.operand_dims[1]);
    let strides = strides_of(init_dims);
    // Every launch in the sequence shares the op's full signature so
    // the executor pushes one uniform argument list.
    let mut sig = format!("const {t}* init, const {t}* src");
    for axis in 0..rank {
        sig.push_str(&format!(", const int* coord{axis}"));
    }
    // Launch 1: dest = copy(init), over dest numel.
    let copy_src = format!(
        r#"extern "C" __global__ void k({sig}, unsigned int* flags, {t}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = init[i];
}}"#
    );
    // Launch 2: scattered writes over src numel, with the injectivity
    // check (checked-scatter ruling): an already-set flag is a
    // conflicting write and traps loudly.
    let mut body = String::from("    long long flat = 0;\n    long long coord;\n");
    for axis in 0..rank {
        body.push_str(&format!(
            "    coord = (long long)coord{axis}[i];\n    if (coord < 0 || coord >= {ext}LL) __trap();\n    flat += coord * {stride}LL;\n",
            ext = init_dims[axis],
            stride = strides[axis]
        ));
    }
    let scatter_src = format!(
        r#"extern "C" __global__ void k({sig}, unsigned int* flags, {t}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{body}    if (atomicExch(&flags[flat], 1u) != 0u) __trap();
    out[flat] = src[i];
}}"#
    );
    let flags_bytes = dest_n * std::mem::size_of::<u32>();
    Ok(vec![
        KernelSource { source: copy_src, n: dest_n, scratch_bytes: flags_bytes },
        KernelSource { source: scatter_src, n: src_n, scratch_bytes: flags_bytes },
    ])
}
