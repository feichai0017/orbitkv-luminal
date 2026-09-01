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
/// carried dtype fact, and EVERY read goes through the layout's own
/// offset expression ([`layout_read_index`]) — the ones that simplify
/// to the identity ([`reads_identity`]) collapse to `a[i]` and the rest
/// are lowered. The hop-chain
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
        // stay dense out-of-place. Every kernel here writes `out[i]`, so
        // a result slot is only writable when its layout's read function
        // IS the identity over that same `i` — decided by the simplifier
        // ([`reads_identity`]), never by the mirror constructor. A
        // CAPABILITY refusal (this backend lowers no strided write),
        // never an e-graph re-check.
        for (k, (slot, dims)) in result_info.iter().zip(&dest_dims).enumerate() {
            if !reads_identity(&slot.layout, dims) {
                bail!(
                    "{label} result {k} carries a layout that does not reduce to the \
                     identity index: strided writes are not lowered (dests stay dense \
                     out-of-place; CL-4b)"
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

    /// The slot's layout when its read does NOT simplify away — i.e.
    /// when the layout expression must actually be lowered and
    /// evaluated at materialized coordinates. `None` means the
    /// simplifier reduced the read to the identity, so the operand is
    /// spelled `name[i]` and no coordinate is materialized for it.
    /// There is no second path: this is the same expression pathway,
    /// simplified.
    pub fn expression_operand(&self, slot: usize) -> Option<&CudaLayout> {
        let layout = &self.operand_layouts[slot];
        (!reads_identity(layout, &self.operand_dims[slot])).then_some(layout)
    }

    /// Every operand's read simplified to the identity, so the lowered
    /// kernel materializes no coordinates at all — the coordinate
    /// prelude would be dead code and the whole body collapses to the
    /// flat form.
    pub fn all_reads_simplify_to_flat(&self) -> bool {
        (0..self.operand_layouts.len()).all(|k| self.expression_operand(k).is_none())
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
// NO RUNTIME BOUNDS TRAPS (ruling 2026-08-31). This runtime emits NO
// `__trap()`. It once did, and the question was considered rather than
// forgotten — this note is the record.
//
// What was checked, and where:
//   * here in `layout_read_index`, per read: the flat element index
//     against the layout's disclosed SPAN for the packed ladder
//     (right-major / left-major / strided), and non-negativity alone for
//     the offset-expression forms (`ElementOffset` / `BitOffset`, whose
//     `SpanExpr` is deliberately unimplemented — an offset function does
//     not say how far it reaches);
//   * in the `BitOffset` arm: that the bit offset divides evenly by the
//     element width, a mid-element bit offset having no element read;
//   * in `ops::index_map_apply_materialize`: each mapped coordinate
//     against the parent's extent;
//   * in `ops::gather` / `ops::scatter`: each gathered/scattered
//     coordinate against the indexed axis extent — these read from an
//     index BUFFER, so they were the only DATA-derived checks;
//   * in `ops::scatter`: injectivity, `atomicExch(&flags[flat],1u)!=0u`
//     over a zeroed scratch buffer, catching two sources writing one
//     destination element.
//
// Why they went. Austin, 2026-08-31: "in the cuda runtime, we should
// have no traps. We can put them back in, later, with a flag or
// something but for now lets get all __trap out of the cuda codegen."
// On the individual checks: the span check is "legitimately useless"; a
// negative offset "would be someone violating their contract, which is
// ub and we shouldn't test for it in runtime"; the bit-divisibility
// check "would also be indicative of a bug somewhere in our compiler,
// which we would have to solve directly, vs having this test at runtime
// in every kernel." The data-derived index checks went with them: an
// out-of-range index in a user tensor is UB at this layer, not a
// diagnosed error.
//
// The consequence, stated plainly: an out-of-range index — from a
// mis-composed layout, a compiler bug, or an out-of-range value in a
// user index tensor — is now an out-of-bounds device access, i.e.
// undefined behaviour, not a diagnosed fault. Debug it with
// `compute-sanitizer`, which sees what these checks used to.
//
// Restoring them belongs behind a feature flag (a `checked` cargo
// feature gating the emission), not behind a runtime branch in every
// thread of every kernel.
// ===========================================================================

// ===========================================================================
// THE READ SIMPLIFIER — the only decision on the read path.
//
// RULING (Austin, 2026-08-31): "It always needs to emit a strided
// expression? That strided expression might just simplify to a[i]. But
// there should be no special casing. it should always go through the
// expression pathway and should never be special cased."
//
// There is ONE read path: materialize the value's coordinates from the
// flat thread index `i`, then evaluate the slot layout's own offset
// expression at those coordinates. For a DENSE layout that whole
// round trip is the identity — the coordinate decomposition and the
// index recomposition cancel — and the kernel may read `a[i]` with no
// coordinates materialized at all. That is a SIMPLIFICATION of the
// expression, not a fork in front of it.
//
// What died here: `layout_is_direct`, which answered the same question
// by matching the mirror CONSTRUCTOR (`RightMajor` and nothing else).
// Its own doc comment admitted the hole — "a dense class rendering
// otherwise takes the (correct, slower) expression read" — and a
// decision made on a SPELLING is exactly what the e-graph is entitled
// to break: every spelling in a layout class denotes ONE function, and
// the renderer hands us whichever it finds. `reads_identity` below
// decides on the FUNCTION.
// ===========================================================================

/// An AFFINE form over a value's coordinates: `constant + Σ coeffs[axis]
/// * c{axis}`, `coeffs` FRONT-indexed and exactly `rank` long. This is a
/// canonical form, not a spelling: two layouts denoting the same affine
/// function reduce to the same `Affine` however they are written.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Affine {
    constant: i64,
    coeffs: Vec<i64>,
}

impl Affine {
    fn zero(rank: usize) -> Self {
        Affine { constant: 0, coeffs: vec![0; rank] }
    }

    fn constant(v: i64, rank: usize) -> Self {
        Affine { constant: v, coeffs: vec![0; rank] }
    }

    /// The unit form for one FRONT axis: `c{axis}`.
    fn coord(axis: usize, rank: usize) -> Self {
        let mut coeffs = vec![0; rank];
        coeffs[axis] = 1;
        Affine { constant: 0, coeffs }
    }

    /// `constant + Σ strides[axis] * c{axis}` with `constant = 0`.
    fn from_strides(strides: &[usize]) -> Option<Self> {
        Some(Affine {
            constant: 0,
            coeffs: strides.iter().map(|&s| i64::try_from(s).ok()).collect::<Option<_>>()?,
        })
    }

    /// The whole form, if it is coordinate-independent.
    fn as_constant(&self) -> Option<i64> {
        self.coeffs.iter().all(|&c| c == 0).then_some(self.constant)
    }

    /// Overflow is treated as "not analyzable" (`None`) — the caller
    /// then takes the general expression read, which is always correct.
    fn add(self, other: Self) -> Option<Self> {
        Some(Affine {
            constant: self.constant.checked_add(other.constant)?,
            coeffs: self
                .coeffs
                .iter()
                .zip(&other.coeffs)
                .map(|(a, b)| a.checked_add(*b))
                .collect::<Option<_>>()?,
        })
    }

    fn scale(self, k: i64) -> Option<Self> {
        Some(Affine {
            constant: self.constant.checked_mul(k)?,
            coeffs: self.coeffs.iter().map(|c| c.checked_mul(k)).collect::<Option<_>>()?,
        })
    }

    /// Division that is EXACT on every term — the only division an
    /// affine form survives. `(6*c0 + 2*c1) / 2` is `3*c0 + c1`;
    /// `(3*c0) / 2` is not affine and gives `None`. Exactness is what
    /// makes this sound for truncating division at any sign: if every
    /// term divides evenly then the whole value is `k * (affine)` and
    /// truncation never rounds.
    fn exact_div(self, k: i64) -> Option<Self> {
        if k == 0 || self.constant % k != 0 || self.coeffs.iter().any(|c| c % k != 0) {
            return None;
        }
        Some(Affine {
            constant: self.constant.checked_div(k)?,
            coeffs: self.coeffs.iter().map(|c| c.checked_div(k)).collect::<Option<_>>()?,
        })
    }
}

/// Reduce one mirror term to an affine form over the value's `rank`
/// coordinates. `None` means "not an affine function of the
/// coordinates" — symbolic vars, coordinate-dependent products,
/// inexact division, remainder, min/max, the bool bridge — and a `None`
/// anywhere simply means the general expression read is emitted, which
/// is always correct. Nothing here inspects a layout constructor.
fn affine_of_term(expr: &luminal::layouts::IntExprTerm, rank: usize) -> Option<Affine> {
    use luminal::layouts::IntExprTerm as T;
    match expr {
        T::Lit(v) => Some(Affine::constant(*v, rank)),
        T::Var(_) => None,
        T::Coord { axis_from_end } => {
            let axis = usize::try_from(*axis_from_end).ok().filter(|&a| a < rank)?;
            Some(Affine::coord(rank - 1 - axis, rank))
        }
        T::Add(a, b) => affine_of_term(a, rank)?.add(affine_of_term(b, rank)?),
        T::Mul(a, b) => {
            let (a, b) = (affine_of_term(a, rank)?, affine_of_term(b, rank)?);
            match (a.as_constant(), b.as_constant()) {
                (Some(k), _) => b.scale(k),
                (_, Some(k)) => a.scale(k),
                // A product of two coordinate-dependent forms is not
                // affine — no simplification, take the expression read.
                _ => None,
            }
        }
        T::TruncDiv(a, b) => {
            let k = affine_of_term(b, rank)?.as_constant()?;
            affine_of_term(a, rank)?.exact_div(k)
        }
        T::TruncRem(_, _) | T::CeilDiv(_, _) | T::Min(_, _) | T::Max(_, _)
        | T::LessThanCast(_, _) => None,
    }
}

/// The slot layout's READ FUNCTION, reduced to an affine form over the
/// value's coordinates — one `Affine` per mirror spelling, never a
/// classification of the spelling itself. The layout's own domain must
/// be literal and equal `dims` (its domain IS the value's shape); a
/// foreign domain is a planner/renderer incoherence and is refused
/// downstream, so it yields `None` here rather than a read.
fn read_affine(layout: &CudaLayout, dims: &[usize]) -> Option<Affine> {
    use luminal::layouts::MirrorLayout as M;
    let rank = dims.len();
    if layout.mirror.literal_extents().as_deref() != Some(dims) {
        return None;
    }
    match &layout.mirror {
        // The packed ladder states its strides structurally.
        M::RightMajor(_) => Affine::from_strides(&strides_of(dims)),
        M::LeftMajor(_) => {
            let mut strides = vec![1usize; rank];
            for axis in 1..rank {
                strides[axis] = strides[axis - 1] * dims[axis - 1];
            }
            Affine::from_strides(&strides)
        }
        // The expression forms state it as a term.
        M::Strided(st) => st
            .chain
            .iter()
            .try_fold(Affine::zero(rank), |acc, s| acc.add(affine_of_term(s, rank)?)),
        M::ElementOffset(eo) => affine_of_term(&eo.offset, rank),
        M::BitOffset(bo) => affine_of_term(&bo.offset, rank)?.exact_div(bo.width.0),
    }
}

/// THE SIMPLIFIER'S VERDICT: does this layout's read function reduce to
/// the IDENTITY over the flat index the coordinates were derived from?
///
/// The read path decomposes flat `i` into row-major coordinates
/// `c0..c{rank-1}` over `dims` and evaluates the layout at them. Writing
/// `s = strides_of(dims)`, that decomposition satisfies `i = Σ s[axis] *
/// c{axis}` exactly. So the round trip is the identity precisely when
/// the layout's affine form is that same sum with no constant term —
/// and then the kernel reads `a[i]` with no coordinate materialized.
///
/// SPELLING-INDEPENDENT by construction: right-major, a strided chain
/// with dense strides, an offset expression that folds to the same, and
/// a bit-offset form whose bits divide out all reduce to one `Affine`.
///
/// An EXTENT-1 axis contributes `c{axis} ∈ {0}`, so its coefficient is
/// unobservable and any value of it is accepted — the coefficient
/// collision is a fact about the function, not a licence.
///
/// This is also the CL-4b WRITE FENCE (a destination is written at
/// `out[i]`, so it must land where the identity says it does) and the
/// escaped-output question in `CudaRuntime::get_f32` (is the backing
/// dense over the value's dims?). One function, three callers.
pub fn reads_identity(layout: &CudaLayout, dims: &[usize]) -> bool {
    let Some(affine) = read_affine(layout, dims) else {
        return false;
    };
    let strides = strides_of(dims);
    affine.constant == 0
        && (0..dims.len()).all(|axis| {
            dims[axis] == 1 || i64::try_from(strides[axis]) == Ok(affine.coeffs[axis])
        })
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
/// `(code, index_var)`; the statements bind `{operand}_idx` and nothing
/// else — no bounds check is emitted, deliberately (see the NO RUNTIME
/// BOUNDS TRAPS note above for what used to be here and why). The
/// layout's own domain (its shape) must be LITERAL and equal the slot's
/// value dims — a foreign-domain layout is a planner/renderer
/// incoherence and refuses loudly.
pub fn layout_read_index(
    operand: &str,
    layout: &CudaLayout,
    slot_dims: &[usize],
    in_prefix: &str,
) -> Result<(String, String)> {
    use luminal::layouts::MirrorLayout;
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
    // The flat element offset expression. No bound travels with it: see
    // the NO RUNTIME BOUNDS TRAPS note above.
    let offset: String = match &layout.mirror {
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
            flat
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
            flat
        }
        MirrorLayout::Strided(st) => {
            check_domain(&st.shape)?;
            let summands = st
                .chain
                .iter()
                .map(|s| lower_layout_term(s, rank, in_prefix))
                .collect::<Result<Vec<_>>>()?;
            if summands.is_empty() {
                "0LL".to_string()
            } else {
                summands.join(" + ")
            }
        }
        MirrorLayout::ElementOffset(eo) => {
            check_domain(&eo.shape)?;
            lower_layout_term(&eo.offset, rank, in_prefix)?
        }
        MirrorLayout::BitOffset(bo) => {
            check_domain(&bo.shape)?;
            let bits = lower_layout_term(&bo.offset, rank, in_prefix)?;
            let width = bo.width.0;
            // Bit form: element index = bit offset / width. The bit
            // offset's divisibility by the element width is a COMPILER
            // invariant (a mid-element bit offset has no element read) —
            // it used to be re-derived at runtime in every thread; see
            // the NO RUNTIME BOUNDS TRAPS note above for why it is not.
            let bits_var = format!("{operand}_bits");
            let code = format!(
                "    long long {bits_var} = {bits};\n    long long {idx} = {bits_var} / {width}LL;\n"
            );
            return Ok((code, idx));
        }
    };
    Ok((format!("    long long {idx} = {offset};\n"), idx))
}

/// One generated launch: entry name is always `k`; `n` is the launch
/// size (one thread per index). Every launch takes the same argument
/// list — the op's inputs, then `out`, then `n`.
///
/// There was once a `scratch_bytes` field asking the executor for a
/// zero-initialized device scratch buffer (passed before `out`), with
/// exactly one user: scatter's injectivity `flags`. That check went with
/// the rest of the traps (2026-08-31), leaving the scratch facility with
/// no caller, so it went too rather than sit unexercised — restore it
/// alongside the check.
#[derive(Debug)]
pub struct KernelSource {
    pub source: String,
    pub n: usize,
}

impl KernelSource {
    pub(crate) fn plain(source: String, n: usize) -> Self {
        Self { source, n }
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

/// `out[i] = <expr of a[i], b[i]>` over the destination's numel. Both
/// operands go through the ONE read path ([`elementwise`]); a read whose
/// layout expression simplifies to the identity stays the literal
/// `a[i]`.
pub(crate) fn binary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let [a, b, _dest] = ctx.operand_dtypes.as_slice() else {
        bail!("binary op expects two operands + dest, got {}", ctx.operand_dtypes.len());
    };
    let (ta, tb) = (cuda_type(*a)?, cuda_type(*b)?);
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let sig = format!("const {ta}* a, const {tb}* b");
    elementwise(ctx, expr, &["a", "b"], &sig, to)
}

/// `out[i] = <expr of a[i]>` over the destination's numel — the same
/// one read path as [`binary`], one operand.
pub(crate) fn unary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let sig = format!("const {ta}* a");
    elementwise(ctx, expr, &["a"], &sig, to)
}

// RULING 2026-08-27: the Phase-5 `copy_through_fold` lowering is DELETED —
// a BufferCopy is only ever a dumb whole-buffer memcpy. A copy
// materialized into a specific layout is a LayoutTensor candidate in the
// e-graph (the materialize kernel), discovered via search, never a copy
// mode.

/// THE elementwise read path — there is no other one. One thread per
/// OUT element; every named operand is read at
/// `name[f(out_coords)]`, where `f` is that operand's own slot layout
/// lowered by [`layout_read_index`] over the out-coordinate prelude.
///
/// THE SIMPLIFIER (ruling 2026-08-31). Before lowering, each operand's
/// read function is reduced ([`reads_identity`]): a read that is the
/// identity over the very `i` the coordinates were decomposed from
/// needs no expression at all and stays the literal `name[i]`. If EVERY
/// read simplifies that way, no coordinate is ever used, the prelude is
/// dead, and the body collapses to the flat one-liner — which is what
/// the byte-identity pin observes. That collapse is a simplification of
/// this path, not a fast path in front of it: nothing here looks at a
/// mirror constructor.
///
/// Contract with the op-module exprs: the template expr reads operand
/// `name` exactly as the literal token `name[i]`, rewritten here to
/// `name[{name}_idx]` when the read does not simplify.
///
/// The DPS dest slot (the operand slot after the named reads) must
/// write at the identity index: strided WRITES are CL-4b and refuse
/// loudly.
fn elementwise(
    ctx: &CodegenCtx,
    expr: &str,
    names: &[&str],
    sig: &str,
    to: &str,
) -> Result<Vec<KernelSource>> {
    let out_dims = &ctx.dest_dims[0];
    let n = numel(out_dims);
    for k in names.len()..ctx.operand_layouts.len() {
        if ctx.expression_operand(k).is_some() {
            bail!(
                "dest operand slot {k} carries a layout that does not reduce to the \
                 identity index: strided writes are not lowered (dests stay dense \
                 out-of-place; CL-4b)"
            );
        }
    }
    // An elementwise operand VALUE spans the out iteration space, so its
    // own extents must be the dest's — asked of EVERY named operand,
    // whatever its layout spells. (It used to be asked only of operands
    // that took the expression read, which made a coherence check
    // spelling-dependent: a dense-but-strided operand answered it and a
    // right-major one of the same wrong shape did not.)
    for (k, name) in names.iter().enumerate() {
        if &ctx.operand_dims[k] != out_dims {
            bail!(
                "operand {name} value extents {:?} differ from dest extents {:?} — \
                 elementwise templates iterate the dest; refuse, never reinterpret",
                ctx.operand_dims[k],
                out_dims
            );
        }
    }
    let mut chains = String::new();
    let mut rendered = expr.to_string();
    for (k, name) in names.iter().enumerate() {
        let Some(layout) = ctx.expression_operand(k) else {
            // The read simplified to the identity: `name[i]` already.
            continue;
        };
        let (code, idx) = layout_read_index(name, layout, out_dims, "c")?;
        chains.push_str(&code);
        let flat = format!("{name}[i]");
        if !rendered.contains(&flat) {
            bail!("template expr `{expr}` has no `{flat}` token to rewrite for a composed operand");
        }
        rendered = rendered.replace(&flat, &format!("{name}[{idx}]"));
    }
    if chains.is_empty() {
        // Every read simplified away — no coordinate is referenced, so
        // the prelude and the coordinate-indexed body vanish with them.
        let source = format!(
            r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = {rendered};
}}"#
        );
        return Ok(vec![KernelSource::plain(source, n)]);
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
/// (iota, constant — dest-only signatures): an operand whose read does
/// NOT simplify to the identity, arriving at a kernel that would index
/// it flat, is silent mistranslation — bail instead.
pub(crate) fn require_flat_operands(label: &str, ctx: &CodegenCtx) -> Result<()> {
    for k in 0..ctx.operand_layouts.len() {
        if ctx.expression_operand(k).is_some() {
            bail!(
                "{label}: operand {k} carries a layout expression this kernel does \
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
    if ctx.all_reads_simplify_to_flat() {
        // Every read simplified to the identity, so the input needs no
        // coordinates: the reduction walks the row-major addresses
        // directly. Same collapse as the elementwise family.
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
    // Expression read: the input slot's read did NOT simplify, so its
    // flat address is the layout's own offset expression evaluated at
    // the INPUT VALUE's coordinates `c0..c{rank-1}` — rebuilt here from
    // the outer/inner decomposition plus the loop's own `r` at the
    // reduced axis. The dest slot must still write at the identity
    // index (CL-4b, no strided writes).
    for k in 1..ctx.operand_layouts.len() {
        if ctx.expression_operand(k).is_some() {
            bail!(
                "dest operand slot {k} carries a layout that does not reduce to the \
                 identity index: strided writes are not lowered (dests stay dense \
                 out-of-place; CL-4b)"
            );
        }
    }
    let layout = ctx
        .expression_operand(0)
        .expect("reduce expression path entered with an unsimplified input layout");
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
