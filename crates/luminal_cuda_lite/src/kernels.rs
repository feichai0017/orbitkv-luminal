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

/// A generated kernel: entry name is always `k`; `n` is the launch
/// size (one thread per output element).
pub struct KernelSource {
    pub source: String,
    pub n: usize,
}

pub struct CudaKernel {
    pub label: &'static str,
    pub op_type: TypeId,
    pub codegen: fn(&dyn BufferTensorIrOp, &CodegenCtx) -> Result<KernelSource>,
}

fn row<T: 'static>(
    label: &'static str,
    codegen: fn(&dyn BufferTensorIrOp, &CodegenCtx) -> Result<KernelSource>,
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
fn binary(ctx: &CodegenCtx, expr: &str) -> Result<KernelSource> {
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
    Ok(KernelSource { source, n })
}

/// `out[i] = <expr of a[i]>` over the destination's numel.
fn unary(ctx: &CodegenCtx, expr: &str) -> Result<KernelSource> {
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(&ctx.dest_dims[0]);
    let source = format!(
        r#"extern "C" __global__ void k(const {ta}* a, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = {expr};
}}"#
    );
    Ok(KernelSource { source, n })
}

fn gen_add(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    binary(ctx, "a[i] + b[i]")
}
fn gen_mul(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    binary(ctx, "a[i] * b[i]")
}
fn gen_div(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    binary(ctx, "a[i] / b[i]")
}
fn gen_trunc_div(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    binary(ctx, "a[i] / b[i]") // integer division in C truncates toward zero
}
fn gen_trunc_rem(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    binary(ctx, "a[i] % b[i]") // C remainder carries the dividend's sign
}
fn gen_mod(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    // Float mod mirrors the reference kernel's fmodf semantics.
    binary(ctx, "fmodf(a[i], b[i])")
}
fn gen_less_than(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    binary(ctx, "(a[i] < b[i]) ? 1 : 0")
}
fn gen_sqrt(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    unary(ctx, "sqrtf(a[i])")
}
fn gen_exp(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    unary(ctx, "expf(a[i])")
}
fn gen_exp2(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    unary(ctx, "exp2f(a[i])")
}
fn gen_log2(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    unary(ctx, "log2f(a[i])")
}
fn gen_sin(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    unary(ctx, "sinf(a[i])")
}
fn gen_recip(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    unary(ctx, "1.0f / a[i]")
}
fn gen_copy(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    unary(ctx, "a[i]")
}

fn gen_cast(_op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    let to = cuda_type(ctx.dest_dtypes[0])?;
    unary(ctx, &format!("({to})a[i]"))
}

fn gen_constant(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
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
    Ok(KernelSource { source, n })
}

/// Axis reduction, axis zero-based FROM THE END (the DPS convention).
/// One thread per output element; the reduced extent is looped.
fn reduce(ctx: &CodegenCtx, axis_from_end: usize, init: &str, fold: &str) -> Result<KernelSource> {
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
    Ok(KernelSource { source, n })
}

fn gen_reduce_sum(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
    let Some(r) = op.as_any().downcast_ref::<rops::ReduceSumDps>() else {
        bail!("reduce_sum codegen reached with a non-ReduceSum op");
    };
    let axis = usize::try_from(r.axis).context("negative reduce axis")?;
    reduce(ctx, axis, "0", "acc + v")
}

fn gen_reduce_max(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<KernelSource> {
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
        ]
    })
}

/// Codegen lookup by concrete op type.
pub fn codegen_for(op: &dyn BufferTensorIrOp) -> Option<&'static CudaKernel> {
    let ty = op.as_any().type_id();
    cuda_kernels().iter().find(|k| k.op_type == ty)
}
