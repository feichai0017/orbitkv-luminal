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
use luminal::dtype::PlanDtype;
use luminal::index_expr::IotaExpr;
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
pub(crate) fn binary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
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
pub(crate) fn unary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
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

/// Lower an [`IotaExpr`] to a C expression over `long long`, with OUT
/// coordinates available as `c0..c{rank-1}` (front-indexed, matching
/// the reference evaluator: `Coord(axis_from_end)` reads
/// `c[rank-1-axis_from_end]`).
pub(crate) fn lower_expr(expr: &IotaExpr, rank: usize) -> Result<String> {
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
