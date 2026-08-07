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
/// `expected_shape` is the OWNER-SHAPE GUARD (Design A fold-in,
/// 2026-08-06): a CoordVar spelling counts only if its owner shape class
/// IS the consumer's out shape — a foreign shape's coordinate must never
/// be silently read as an out-coordinate; the kernel's loud refusal
/// carries the burden instead.
pub(crate) fn parse_int_expr(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
) -> Option<IotaExpr> {
    parse_int_expr_memo(site, class, depth, expected_shape, &mut std::collections::HashMap::new())
}

/// Memo entry: a finished parse, or the in-progress cycle guard.
#[derive(Clone)]
pub(crate) enum ParseMemo {
    InProgress,
    Done(Option<IotaExpr>),
}

/// Memoized worker: each class parses at most once — the subsumed-
/// spelling fallback widens the branching factor enough that the naive
/// backtracking walk goes exponential on fat saturated classes.
///
/// CYCLE-TAINT RULE (2026-08-06, found by the MoE map parse): hitting an
/// in-progress class fails THAT spelling, but the failure is contextual —
/// the same class can parse fine once its ancestor resolves. So a `None`
/// outcome whose walk touched the cycle guard is NOT cached (a later
/// query retries from a clean stack), while an untainted `None` — every
/// spelling genuinely outside the subset — caches as before, keeping the
/// exponential-blowup protection.
pub(crate) fn parse_int_expr_memo(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, ParseMemo>,
) -> Option<IotaExpr> {
    parse_int_expr_tainting(site, class, depth, expected_shape, memo, &mut false)
}

fn parse_int_expr_tainting(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, ParseMemo>,
    tainted: &mut bool,
) -> Option<IotaExpr> {
    match memo.get(class) {
        Some(ParseMemo::Done(cached)) => return cached.clone(),
        Some(ParseMemo::InProgress) => {
            *tainted = true;
            return None;
        }
        None => {}
    }
    memo.insert(class.clone(), ParseMemo::InProgress);
    let mut local_taint = false;
    let parsed = parse_int_expr_uncached(site, class, depth, expected_shape, memo, &mut local_taint);
    if parsed.is_none() && local_taint {
        memo.remove(class);
        *tainted = true;
    } else {
        memo.insert(class.clone(), ParseMemo::Done(parsed.clone()));
    }
    parsed
}

fn parse_int_expr_uncached(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, ParseMemo>,
    tainted: &mut bool,
) -> Option<IotaExpr> {
    if depth == 0 {
        return None;
    }
    if let Some(lit) = site.nodes_in_class_value(class, "IntLit").next() {
        let value_class = site.class_of_child(lit, 0)?;
        return Some(IotaExpr::Lit(site.node_in_class_parse_i64(&value_class)?));
    }
    for coord in site.nodes_in_class_value(class, "CoordVar") {
        // Scoped coordinates: child 0 is the owner Shape, child 1 the
        // axis. The owner-shape guard: when the caller names its out
        // shape, a CoordVar owned by any OTHER shape is not an
        // out-coordinate and cannot parse (loud kernel refusal instead
        // of a silently misread axis).
        if let Some(expected) = expected_shape {
            let Some(owner_class) = site.class_of_child(coord, 0) else { continue };
            if owner_class != *expected {
                continue;
            }
        }
        let Some(axis_class) = site.class_of_child(coord, 1) else { continue };
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
        for node in site.nodes_in_class_value(class, kind) {
            let Some(lhs_class) = site.class_of_child(node, 0) else { continue };
            let Some(rhs_class) = site.class_of_child(node, 1) else { continue };
            let Some(lhs) = parse_int_expr_tainting(site, &lhs_class, depth - 1, expected_shape, memo, tainted) else { continue };
            let Some(rhs) = parse_int_expr_tainting(site, &rhs_class, depth - 1, expected_shape, memo, tainted) else { continue };
            return Some(build(Box::new(lhs), Box::new(rhs)));
        }
    }
    for cast in site.nodes_in_class_value(class, "IntCastFromBool") {
        let Some(bool_class) = site.class_of_child(cast, 0) else { continue };
        let Some(less_than) = site.nodes_in_class_value(&bool_class, "BoolLessThanInt").next() else {
            continue;
        };
        let Some(lhs_class) = site.class_of_child(less_than, 0) else { continue };
        let Some(rhs_class) = site.class_of_child(less_than, 1) else { continue };
        let Some(lhs) = parse_int_expr_tainting(site, &lhs_class, depth - 1, expected_shape, memo, tainted) else { continue };
        let Some(rhs) = parse_int_expr_tainting(site, &rhs_class, depth - 1, expected_shape, memo, tainted) else { continue };
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
        Box::new(Iota { expr: parse_int_expr(site, &site.child_class(0), 64, Some(&site.child_class(1))) })
    }
}
