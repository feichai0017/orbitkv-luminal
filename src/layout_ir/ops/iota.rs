//! The index-expression generator: a ZERO-INPUT source op that writes
//! `expr(c0..ck)` at every output coordinate — the tensor form of a single
//! `IntExpr` over the output shape's coordinate variables.
//!
//! Iota is the first op with no tensor operands, which makes it the proof
//! case for the zero-input path through extraction (empty operand list),
//! DPS rewriting (the appended destination is the ONLY operand), and
//! bufferization (nothing to alias, one fresh write). There is no mutating
//! form — with no input there is nothing to mutate.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps};

/// `IotaGeneric() -> out`
///
/// Functional source form: no operands, one freshly-written result.
/// Reference semantics: `out[c0..ck] = expr(c0..ck)` evaluated in the
/// canonical right-major order, Int element type (the value IS an index
/// expression, never a float; the dtype rule pins this in egglog).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iota {
    /// The window expression, numerically — `None` for forms beyond the
    /// parsed subset (extraction stays infallible; the reference kernel
    /// refuses loudly instead).
    pub expr: Option<IotaExpr>,
}

/// A numeric IntExpr tree for reference evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IotaExpr {
    Lit(i64),
    /// CoordVar axis, zero-based from the END over the OUT coordinates.
    Coord(usize),
    Add(Box<IotaExpr>, Box<IotaExpr>),
    Mul(Box<IotaExpr>, Box<IotaExpr>),
    /// Truncated (toward-zero) division — the preamble's IntTruncDiv.
    TruncDiv(Box<IotaExpr>, Box<IotaExpr>),
    /// Truncated remainder — the preamble's IntTruncRem.
    TruncRem(Box<IotaExpr>, Box<IotaExpr>),
    Min(Box<IotaExpr>, Box<IotaExpr>),
    Max(Box<IotaExpr>, Box<IotaExpr>),
    /// The bool bridge's indicator: `(a < b) as i64` — IntCastFromBool
    /// over BoolLessThanInt.
    LessThanCast(Box<IotaExpr>, Box<IotaExpr>),
}

impl IotaExpr {
    /// Evaluate at the given OUT coordinates (front-indexed).
    pub fn eval(&self, coords: &[usize]) -> i64 {
        match self {
            IotaExpr::Lit(value) => *value,
            IotaExpr::Coord(axis_from_end) => coords[coords.len() - 1 - axis_from_end] as i64,
            IotaExpr::Add(a, b) => a.eval(coords) + b.eval(coords),
            IotaExpr::Mul(a, b) => a.eval(coords) * b.eval(coords),
            // Divisors are literal strides/extents (>= 1 by construction);
            // a zero here is a translator bug and deserves the loud panic.
            IotaExpr::TruncDiv(a, b) => a.eval(coords) / b.eval(coords),
            IotaExpr::TruncRem(a, b) => a.eval(coords) % b.eval(coords),
            IotaExpr::Min(a, b) => a.eval(coords).min(b.eval(coords)),
            IotaExpr::Max(a, b) => a.eval(coords).max(b.eval(coords)),
            IotaExpr::LessThanCast(a, b) => (a.eval(coords) < b.eval(coords)) as i64,
        }
    }
}

/// Parse one IntExpr class into an [`IotaExpr`], preferring folded literals;
/// depth-guarded (saturated classes hold many equal representations — any
/// one denotes the same function). `None` = unsupported constructors.
pub(crate) fn parse_int_expr(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
) -> Option<IotaExpr> {
    if depth == 0 {
        return None;
    }
    if let Some(lit) = site.node_in_class(class, "IntLit") {
        let value_class = site.class_of_child(lit, 0)?;
        return Some(IotaExpr::Lit(site.node_in_class_parse_i64(&value_class)?));
    }
    if let Some(coord) = site.node_in_class(class, "CoordVar") {
        // Scoped coordinates: child 0 is the owner Shape, child 1 the axis.
        let axis_class = site.class_of_child(coord, 1)?;
        return Some(IotaExpr::Coord(
            site.node_in_class_parse_i64(&axis_class)? as usize,
        ));
    }
    // Binary kinds: BACKTRACK across every representation in the class —
    // a saturated class holds many equal spellings, and the first node of
    // a kind may have children outside the parsed subset while a sibling
    // spelling parses fine.
    let binary_kinds: [(&str, fn(Box<IotaExpr>, Box<IotaExpr>) -> IotaExpr); 6] = [
        ("IntAdd", |a, b| IotaExpr::Add(a, b)),
        ("IntMul", |a, b| IotaExpr::Mul(a, b)),
        ("IntTruncDiv", |a, b| IotaExpr::TruncDiv(a, b)),
        ("IntTruncRem", |a, b| IotaExpr::TruncRem(a, b)),
        ("IntMin", |a, b| IotaExpr::Min(a, b)),
        ("IntMax", |a, b| IotaExpr::Max(a, b)),
    ];
    for (kind, build) in binary_kinds {
        for node in site.nodes_in_class(class, kind) {
            let Some(lhs_class) = site.class_of_child(node, 0) else { continue };
            let Some(rhs_class) = site.class_of_child(node, 1) else { continue };
            let Some(lhs) = parse_int_expr(site, &lhs_class, depth - 1) else { continue };
            let Some(rhs) = parse_int_expr(site, &rhs_class, depth - 1) else { continue };
            return Some(build(Box::new(lhs), Box::new(rhs)));
        }
    }
    for cast in site.nodes_in_class(class, "IntCastFromBool") {
        let Some(bool_class) = site.class_of_child(cast, 0) else { continue };
        let Some(less_than) = site.node_in_class(&bool_class, "BoolLessThanInt") else {
            continue;
        };
        let Some(lhs_class) = site.class_of_child(less_than, 0) else { continue };
        let Some(rhs_class) = site.class_of_child(less_than, 1) else { continue };
        let Some(lhs) = parse_int_expr(site, &lhs_class, depth - 1) else { continue };
        let Some(rhs) = parse_int_expr(site, &rhs_class, depth - 1) else { continue };
        return Some(IotaExpr::LessThanCast(Box::new(lhs), Box::new(rhs)));
    }
    None
}

impl OpSlotNames for Iota {}

impl BufferTensorIrOp for Iota {
    fn label(&self) -> &str {
        "IotaGeneric"
    }
}

impl Bufferizable for Iota {}

impl ToDps for Iota {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(IotaDps { expr: self.expr.clone() }))
    }
}

impl LayoutIrOp for Iota {}

/// Destination-passing form of [`Iota`], signature spelled slot by slot:
///
/// ```text
/// IotaGeneric(dest0: write-only ↔ out0) -> out0
/// ```
///
/// The destination is the op's ONLY operand — the zero-input source's DPS
/// form is pure write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IotaDps {
    /// See [`Iota::expr`].
    pub expr: Option<IotaExpr>,
}

impl OpSlotNames for IotaDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IotaDps {
    fn reference_execute(
        &self,
        ctx: &mut crate::buffer_tensor_ir::ReferenceKernelCtx,
    ) -> anyhow::Result<()> {
        let Some(expr) = &self.expr else {
            anyhow::bail!("iota reference kernel supports Lit/Coord/Add/Mul expressions only");
        };
        let out_dims = ctx.operand_dims.last().cloned().unwrap_or_default();
        let rank = out_dims.len();
        let dest = ctx.dests[0].as_f32_mut()?;
        for flat in 0..dest.len() {
            let mut remainder = flat;
            let mut coords = vec![0usize; rank];
            for axis in (0..rank).rev() {
                coords[axis] = remainder % out_dims[axis];
                remainder /= out_dims[axis];
            }
            dest[flat] = expr.eval(&coords) as f32;
        }
        Ok(())
    }

    fn label(&self) -> &str {
        "IotaGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for IotaDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 0, result: 0, sharing: Sharing::Must }]
    }
}

impl ToDps for IotaDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for IotaDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpIotaGeneric` enodes and produces [`Iota`]
/// instances. Metadata children: `expr` at child 0, `shape` at child 1,
/// `out_layout` at child 2 — the whole constructor is metadata; there are
/// no tensor operands.
#[derive(Debug, Clone, Copy, Default)]
pub struct IotaMatcher;

impl OpMatcher for IotaMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIotaGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("iota/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("iota/match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("expr", 0), ("shape", 1), ("out_layout", 2)]
    }

    /// Pure structure — the bounds story lives in egglog (user ruling
    /// 2026-07-23): the iota-int32-certified gate on the op match makes a
    /// missing-bounds iota unimplementable (fail-open), and the
    /// fixpoint-invariants stratum panics on a PROVEN violation. An enode
    /// reaching this matcher is certified by construction.
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Iota { expr: parse_int_expr(site, &site.child_class(0), 64) })
    }
}
