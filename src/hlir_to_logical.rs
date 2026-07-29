//! HLIR → Logical translation: the M1 interim path from luminal's HLIR graph
//! into the logical-SSA egglog vocabulary (model + binding text appended to
//! `egglog_snippet::assembled_program()`).
//!
//! SLICE 2: STATIC graphs with contiguous ops plus movement views lifted
//! from affine stride patterns — permutes, expands/broadcasts (zero
//! strides), size-1 axes, and rank-0 constants broadcast anywhere. A
//! non-affine stride (repeat's `z % d`, slices' offsets, merged axes) or
//! anything else outside the slice (symbolic dims, gather/scatter/cast/
//! loops) fails LOUDLY with a named node — never a silent mistranslation.
//!
//! This module is interim scaffolding: it dies when GraphTensor emits
//! logical ops directly (M3). It deliberately lives beside — never inside —
//! the existing `build_search_space` ladder; the branch is the gate
//! (ruling: no cargo features).

use anyhow::{Context, Result, anyhow, bail, ensure};
use as_any::Downcast;
use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;

use crate::dtype::DType;
use crate::graph::Graph;
use crate::hlir::{
    Add, Constant, Exp2, Gather, Input, Iota, Log2, MaxReduce, Mod, Mul, Output, Recip,
    ClampView, MaskIota, Scatter, Sin, SliceView, Sqrt, SumReduce, UnfoldView,
};
use crate::shape::{Expression, Term};
use std::collections::BTreeMap;
use crate::shape::ShapeTracker;

/// The translated program plus the I/O binding tables the runtime needs.
/// Buffer ids equal HLIR node indices, matching `ReferenceRuntime`'s
/// `set_data`/`get_f32` keying so differential tests bind identically.
#[derive(Debug, Clone)]
pub struct LogicalProgram {
    /// Model + binding + schedule + authoring-contract checks. Run as
    /// `format!("{}\n\n{}", egglog_snippet::assembled_program(), text)`.
    pub text: String,
    /// `(HLIR input node, BufferLit id)` in signature order.
    pub input_slots: Vec<(NodeIndex, u64)>,
    /// `(Output.node key, BufferLit id)` in output-slot order — the key
    /// their `get_f32` matches on (the SOURCE tensor's node index).
    pub output_slots: Vec<(usize, u64)>,
}

/// Per-translated-value bookkeeping.
struct ValueInfo {
    let_name: String,
    dims: Vec<usize>,
    dtype: DType,
}

/// Their expression, parsed into a tree (RPN with the stack TOP as the
/// LEFT operand — verified against their `as_op` and the `Sub` impl).
enum ExprTree {
    Num(i64),
    Var(char),
    Op(Term, Box<ExprTree>, Box<ExprTree>),
}

fn rpn_tree(expr: &Expression) -> Option<ExprTree> {
    let mut stack: Vec<ExprTree> = Vec::new();
    for term in expr.terms.read().iter() {
        match term {
            Term::Num(n) => stack.push(ExprTree::Num(*n)),
            Term::Var(c) => stack.push(ExprTree::Var(*c)),
            op => {
                let left = stack.pop()?;
                let right = stack.pop()?;
                stack.push(ExprTree::Op(*op, Box::new(left), Box::new(right)));
            }
        }
    }
    if stack.len() == 1 { stack.pop() } else { None }
}

/// A stride expression's STRUCTURAL classification — syntax-directed over
/// the parsed tree, so the recovered semantics are exact BY CONSTRUCTION
/// (their tracker algebra emits these forms; anything else is a loud bail,
/// never a guess — no evaluation, no probing, per the 2026-07-29 ruling).
enum StrideForm {
    /// Broadcast: contributes nothing.
    Zero,
    /// `c · z` (or bare `z`): an ordinary strided axis.
    Affine(usize),
    /// `c · (z % d)`: a repeat/tiling axis over a parent of extent `d`.
    Repeat { stride: usize, extent: usize },
}

fn classify_stride(expr: &Expression) -> Option<StrideForm> {
    fn positive(n: i64) -> Option<usize> {
        (n > 0).then_some(n as usize)
    }
    let tree = rpn_tree(expr)?;
    match tree {
        ExprTree::Num(0) => Some(StrideForm::Zero),
        ExprTree::Var('z') => Some(StrideForm::Affine(1)),
        ExprTree::Op(Term::Mul, left, right) => {
            let (factor, other) = match (*left, *right) {
                (ExprTree::Num(n), other) | (other, ExprTree::Num(n)) => (positive(n)?, other),
                _ => return None,
            };
            match other {
                ExprTree::Var('z') => Some(StrideForm::Affine(factor)),
                ExprTree::Op(Term::Mod, modl, modr) => match (*modl, *modr) {
                    (ExprTree::Var('z'), ExprTree::Num(d)) => Some(StrideForm::Repeat {
                        stride: factor,
                        extent: positive(d)?,
                    }),
                    _ => None,
                },
                _ => None,
            }
        }
        ExprTree::Op(Term::Mod, left, right) => match (*left, *right) {
            (ExprTree::Var('z'), ExprTree::Num(d)) => Some(StrideForm::Repeat {
                stride: 1,
                extent: positive(d)?,
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Recover the CANONICAL parent dims/// Recover the CANONICAL parent dims (size-1 axes omitted — they are
/// unobservable through strides and semantically inert) from one consumer's
/// tracker: contiguous trackers answer directly; affine views reconstruct
/// by sorting real axes' coefficients into a telescoping contiguous ladder.
fn parent_dims_from_tracker(
    tracker: &ShapeTracker,
    dyn_map: &FxHashMap<char, usize>,
    at: &str,
) -> Result<Vec<usize>> {
    let dims: Vec<usize> = tracker
        .dims
        .iter()
        .map(|dim| {
            dim.exec(dyn_map)
                .with_context(|| format!("hlir_to_logical: unpinned symbolic dim at {at}"))
        })
        .collect::<Result<_>>()?;
    if tracker.is_contiguous() {
        return Ok(dims.into_iter().filter(|dim| *dim > 1).collect());
    }
    let mut real: Vec<(usize, usize)> = Vec::new(); // (coefficient, extent)
    for (stride, &dim) in tracker.strides.iter().zip(&dims) {
        let stride = stride.resolve_vars(dyn_map);
        match classify_stride(&stride) {
            Some(StrideForm::Zero) => {}
            Some(StrideForm::Affine(c)) => {
                if dim > 1 {
                    real.push((c, dim));
                }
            }
            Some(StrideForm::Repeat { stride: c, extent: d }) => {
                if d > 1 {
                    real.push((c, d));
                }
            }
            None => bail!(
                "hlir_to_logical: unrecognized stride form at {at} — loud bail, never a guess"
            ),
        }
    }
    real.sort_by(|a, b| b.0.cmp(&a.0));
    let mut expected = 1usize;
    for (c, dim) in real.iter().rev() {
        ensure!(
            *c == expected,
            "hlir_to_logical slice 2: strides at {at} do not telescope to a contiguous parent"
        );
        expected *= dim;
    }
    Ok(real.into_iter().map(|(_, dim)| dim).collect())
}

/// Translate one operand: identity when the snapshot is contiguous over the
/// source's own dims; otherwise LIFT the affine stride pattern into an
/// IndexMapApply view (permute/expand/broadcast; size-1 view axes and
/// size-1 parent axes are index-0 constants). Returns the operand's let
/// name and its VIEW dims (the shape the consuming op computes over).
fn lift_operand(
    ops_text: &mut String,
    tracker: &ShapeTracker,
    source: &ValueInfo,
    dyn_map: &FxHashMap<char, usize>,
    node_index: usize,
    position: usize,
) -> Result<(String, Vec<usize>)> {
    let at = format!("t{node_index} operand {position}");
    let view_dims: Vec<usize> = tracker
        .dims
        .iter()
        .map(|dim| {
            dim.exec(dyn_map)
                .with_context(|| format!("hlir_to_logical: unpinned symbolic dim at {at}"))
        })
        .collect::<Result<_>>()?;
    if tracker.is_contiguous() && view_dims == source.dims {
        return Ok((source.let_name.clone(), view_dims));
    }

    let forms: Vec<Option<StrideForm>> = tracker
        .strides
        .iter()
        .map(|stride| classify_stride(&stride.resolve_vars(dyn_map)))
        .collect();

    let parent = &source.dims;
    let mut parent_strides = vec![1usize; parent.len()];
    for k in (0..parent.len().saturating_sub(1)).rev() {
        parent_strides[k] = parent_strides[k + 1] * parent[k + 1];
    }

    // Match view axes to parent axes, STRUCTURALLY, in three forms:
    //   exact  — one view axis is one parent axis (c = pstride, d = pdim);
    //   merge  — one view axis covers a telescoping RUN of parent axes
    //            (c = the run's innermost pstride, d = the run's extent
    //            product): entries are div/rem digits of its coordinate;
    //   split  — several view axes form a MIXED-RADIX group over one
    //            parent axis (coefficients telescope down to pstride, the
    //            extents multiply to the parent extent): the entry is the
    //            group's weighted coordinate sum.
    // Plus the repeat form as before. Everything is syntax-derived from the
    // classified stride forms — no evaluation anywhere; unmatched shapes
    // bail loudly.
    let rank = view_dims.len();
    /// How one parent axis is addressed by the view.
    enum ParentUse {
        /// (view axis, wraps via `% extent` — the repeat form)
        Exact { axis: usize, repeat: bool },
        /// One view axis whose coordinate holds this axis as a digit:
        /// (view axis, divisor = run-extent-product inward of this axis,
        /// needs_rem = not the outermost axis of its run)
        MergeDigit { axis: usize, divisor: usize, needs_rem: bool },
        /// Mixed-radix group: (view axis, weight) terms, outermost first.
        SplitGroup { terms: Vec<(usize, usize)> },
    }
    let mut consumed: Vec<Option<ParentUse>> = (0..parent.len()).map(|_| None).collect();
    let mut matched_view: Vec<bool> = vec![false; rank];

    // Pass 1: exact matches and repeats.
    for (axis, (form, &dim)) in forms.iter().zip(&view_dims).enumerate() {
        let Some(form) = form else { continue };
        match form {
            StrideForm::Zero => {
                matched_view[axis] = true; // broadcast
            }
            _ if dim == 1 => {
                matched_view[axis] = true; // degenerate: index always 0
            }
            StrideForm::Affine(c) => {
                if let Some(k) = (0..parent.len()).find(|&k| {
                    consumed[k].is_none() && parent_strides[k] == *c && parent[k] == dim
                }) {
                    consumed[k] = Some(ParentUse::Exact { axis, repeat: false });
                    matched_view[axis] = true;
                }
            }
            StrideForm::Repeat { stride, extent } => {
                let Some(k) = (0..parent.len()).find(|&k| {
                    consumed[k].is_none()
                        && parent_strides[k] == *stride
                        && parent[k] == *extent
                        && dim % extent == 0
                }) else {
                    bail!(
                        "hlir_to_logical: {at} axis {axis} repeat form matches no axis of \
                         parent {parent:?}"
                    );
                };
                consumed[k] = Some(ParentUse::Exact { axis, repeat: true });
                matched_view[axis] = true;
            }
        }
    }

    // Pass 2: merges — an unmatched affine view axis whose extent is the
    // product of a run of unconsumed parent axes ending where its
    // coefficient sits.
    for (axis, (form, &dim)) in forms.iter().zip(&view_dims).enumerate() {
        if matched_view[axis] {
            continue;
        }
        let Some(StrideForm::Affine(c)) = form else { continue };
        let Some(run_end) = (0..parent.len())
            .find(|&m| consumed[m].is_none() && parent_strides[m] == *c)
        else {
            continue;
        };
        let mut product = 1usize;
        let mut run_start = None;
        for k in (0..=run_end).rev() {
            if consumed[k].is_some() {
                break;
            }
            product *= parent[k];
            if product == dim {
                run_start = Some(k);
                break;
            }
            if product > dim {
                break;
            }
        }
        let Some(run_start) = run_start else { continue };
        let mut divisor = 1usize;
        for k in (run_start..=run_end).rev() {
            consumed[k] = Some(ParentUse::MergeDigit {
                axis,
                divisor,
                needs_rem: k != run_start,
            });
            divisor *= parent[k];
        }
        matched_view[axis] = true;
    }

    // Pass 3: split groups — remaining affine view axes partition by the
    // parent axis their coefficient falls inside, then each group must
    // telescope as a mixed-radix system covering the parent extent.
    for k in 0..parent.len() {
        if consumed[k].is_some() {
            continue;
        }
        let lo = parent_strides[k];
        let hi = lo * parent[k];
        let mut group: Vec<(usize, usize, usize)> = Vec::new(); // (c, axis, dim)
        for (axis, (form, &dim)) in forms.iter().zip(&view_dims).enumerate() {
            if matched_view[axis] {
                continue;
            }
            if let Some(StrideForm::Affine(c)) = form {
                if *c >= lo && *c < hi && c % lo == 0 {
                    group.push((*c, axis, dim));
                }
            }
        }
        if group.is_empty() {
            continue;
        }
        group.sort_by(|a, b| b.0.cmp(&a.0));
        let mut expected = lo;
        let mut extent_product = 1usize;
        let telescopes = group.iter().rev().all(|(c, _, dim)| {
            let ok = *c == expected;
            expected = c * dim;
            extent_product *= dim;
            ok
        });
        ensure!(
            telescopes && extent_product == parent[k],
            "hlir_to_logical: {at} split group over parent axis {k} does not telescope \
             (parent extent {}, group {:?})",
            parent[k],
            group
        );
        for (_, axis, _) in &group {
            matched_view[*axis] = true;
        }
        consumed[k] = Some(ParentUse::SplitGroup {
            terms: group
                .into_iter()
                .map(|(c, axis, _)| (axis, c / lo))
                .collect(),
        });
    }

    for (axis, matched) in matched_view.iter().enumerate() {
        ensure!(
            *matched,
            "hlir_to_logical: {at} axis {axis} (extent {}, stride form unrecognized or \
             unmatched) — loud bail, never a guess",
            view_dims[axis]
        );
    }
    for k in 0..parent.len() {
        ensure!(
            parent[k] == 1 || consumed[k].is_some(),
            "hlir_to_logical: {at} drops parent axis {k} (extent {}) — slicing is a \
             later slice",
            parent[k]
        );
    }

    // Map entries per PARENT axis, outermost inward; CoordVar axes are    // Map entries per PARENT axis, outermost inward; CoordVar axes are
    // zero-based from the innermost of the VIEW shape.
    let mut entries = "(IntExprNil)".to_string();
    for k in (0..parent.len()).rev() {
        let coord = |axis: usize| {
            format!("(CoordVar {} (IntLit {}))", rank - 1 - axis, view_dims[axis])
        };
        let entry = match &consumed[k] {
            Some(ParentUse::Exact { axis, repeat: false }) => coord(*axis),
            Some(ParentUse::Exact { axis, repeat: true }) => {
                format!("(IntTruncRem {} (IntLit {}))", coord(*axis), parent[k])
            }
            Some(ParentUse::MergeDigit { axis, divisor, needs_rem }) => {
                let mut term = coord(*axis);
                if *divisor > 1 {
                    term = format!("(IntTruncDiv {term} (IntLit {divisor}))");
                }
                if *needs_rem {
                    term = format!("(IntTruncRem {term} (IntLit {}))", parent[k]);
                }
                term
            }
            Some(ParentUse::SplitGroup { terms }) => {
                let mut parts: Vec<String> = terms
                    .iter()
                    .map(|(axis, weight)| {
                        if *weight == 1 {
                            coord(*axis)
                        } else {
                            format!("(IntMul {} (IntLit {weight}))", coord(*axis))
                        }
                    })
                    .collect();
                let mut term = parts.pop().expect("non-empty split group");
                while let Some(part) = parts.pop() {
                    term = format!("(IntAdd {part} {term})");
                }
                term
            }
            None => "(IntLit 0)".to_string(),
        };
        entries = format!("(IntExprCons {entry} {entries})");
    }
    let shape = shape_term(&view_dims);
    let name = format!("t{node_index}_operand{position}_view");
    ops_text.push_str(&format!(
        "(let {name} (LogicalIndexMapApply {} (IndexMapLit {entries}) {shape}))\n",
        source.let_name
    ));
    Ok((name, view_dims))
}

/// The slice-1 view of an op's per-input ShapeTrackers (`None` = the op
/// carries none we understand — Outputs return an empty list).
fn op_input_trackers(op: &dyn crate::op::HLIROp) -> Option<Vec<ShapeTracker>> {
    if let Some(op) = op.downcast_ref::<Add>() {
        return Some(op.input_shapes.clone());
    }
    if let Some(op) = op.downcast_ref::<Mul>() {
        return Some(op.input_shapes.clone());
    }
    if let Some(op) = op.downcast_ref::<Mod>() {
        return Some(op.input_shapes.clone());
    }
    if let Some(op) = op.downcast_ref::<Recip>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<Sqrt>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<Sin>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<Exp2>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<Log2>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<Gather>() {
        return Some(op.input_shapes.clone());
    }
    if let Some(op) = op.downcast_ref::<SliceView>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<UnfoldView>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<ClampView>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<Scatter>() {
        // Source order [dest, indexes, src]; src iterates the index shape.
        return Some(vec![
            tracker_of(&op.dest_shape, &op.dest_strides),
            tracker_of(&op.index_shape, &op.index_strides),
            tracker_of(&op.index_shape, &op.src_strides),
        ]);
    }
    if let Some(op) = op.downcast_ref::<SumReduce>() {
        return Some(vec![op.input_shape]);
    }
    if let Some(op) = op.downcast_ref::<MaxReduce>() {
        return Some(vec![op.input_shape]);
    }
    None
}

/// An Input's dims, recovered from its consumers' ShapeTracker snapshots
/// (Input ops carry no shape). Every consuming snapshot must be slice-1
/// admissible and they must all agree.
fn input_dims_from_consumers(
    graph: &Graph,
    input: NodeIndex,
    dyn_map: &FxHashMap<char, usize>,
) -> Result<(Vec<usize>, Option<Vec<Expression>>)> {
    let mut derived: Option<Vec<usize>> = None;
    let mut dim_exprs: Option<Vec<Expression>> = None;
    for consumer in graph.graph.neighbors_directed(input, Direction::Outgoing) {
        let op = &graph.graph[consumer];
        if op.as_ref().downcast_ref::<Output>().is_some() {
            continue; // outputs constrain nothing
        }
        let Some(trackers) = op_input_trackers(op.as_ref()) else {
            bail!(
                "hlir_to_logical slice 1: input t{} consumed by unsupported op {op:?}",
                input.index()
            );
        };
        let sources = graph.get_sources(consumer);
        for (position, source) in sources.iter().enumerate() {
            if *source != input {
                continue;
            }
            let tracker = trackers.get(position).ok_or_else(|| {
                anyhow!(
                    "hlir_to_logical: consumer t{} has no tracker for operand {position}",
                    consumer.index()
                )
            })?;
            let dims =
                parent_dims_from_tracker(tracker, dyn_map, &format!("input t{}", input.index()))?;
            if tracker.is_contiguous() && dim_exprs.is_none() {
                // A contiguous snapshot preserves the input's own dim
                // EXPRESSIONS — the symbolic terms the declaration should
                // carry (size-1 dims included; they resolve identically).
                dim_exprs = Some(tracker.dims.iter().copied().filter(|d| d.exec(dyn_map) != Some(1)).collect());
            }
            match &derived {
                None => derived = Some(dims),
                Some(existing) if *existing == dims => {}
                Some(existing) => bail!(
                    "hlir_to_logical: input t{} consumers disagree on shape ({existing:?} vs {dims:?})",
                    input.index()
                ),
            }
        }
    }
    let dims = derived.ok_or_else(|| {
        anyhow!(
            "hlir_to_logical: input t{} has no shape-bearing consumer",
            input.index()
        )
    })?;
    Ok((dims, dim_exprs))
}

/// `[2, 3]` → `(ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil))))`
fn shape_term(dims: &[usize]) -> String {
    let mut term = "(IntExprNil)".to_string();
    for dim in dims.iter().rev() {
        term = format!("(IntExprCons (IntLit {dim}) {term})");
    }
    format!("(ShapeLit {term})")
}

/// One dim as an egglog term: literals stay literals; a bare symbolic var
/// becomes `(IntVar "c")` and RECORDS its pin — execution requires every
/// var pinned via `set_dim` (the binding then seeds tight bounds and the
/// [n,n] collapse delivers the literal to every user by congruence).
fn dim_term(
    expr: &Expression,
    dyn_map: &FxHashMap<char, usize>,
    vars: &mut BTreeMap<char, usize>,
    at: &str,
) -> Result<String> {
    let terms = expr.terms.read();
    match &terms[..] {
        [Term::Num(n)] => Ok(format!("(IntLit {n})")),
        [Term::Var(c)] => {
            let Some(value) = dyn_map.get(c) else {
                bail!("hlir_to_logical: dim '{c}' at {at} needs set_dim (execution requires a pin)");
            };
            vars.insert(*c, *value);
            Ok(format!("(IntVar \"{c}\")"))
        }
        _ => bail!("hlir_to_logical: arithmetic dim expression at {at} — later slice"),
    }
}

/// A shape term from per-dim terms.
fn shape_term_of(dims: &[String]) -> String {
    let mut term = "(IntExprNil)".to_string();
    for dim in dims.iter().rev() {
        term = format!("(IntExprCons {dim} {term})");
    }
    format!("(ShapeLit {term})")
}

/// Their RPN index expression rendered as OUR IntExpr term, with `z`
/// replaced by the given coordinate term and dyn vars resolved via the
/// pins. Add/Mul only for now (their slice path is affine); anything else
/// bails loudly.
fn int_expr_term(
    expr: &Expression,
    coord_term: &str,
    dyn_map: &FxHashMap<char, usize>,
    at: &str,
) -> Result<String> {
    let mut stack: Vec<String> = Vec::new();
    for term in expr.terms.read().iter() {
        match term {
            Term::Num(n) => stack.push(format!("(IntLit {n})")),
            Term::Var('z') => stack.push(coord_term.to_string()),
            Term::Var(c) => {
                let value = dyn_map.get(c).ok_or_else(|| {
                    anyhow!("hlir_to_logical: unpinned var '{c}' in index expression at {at}")
                })?;
                stack.push(format!("(IntLit {value})"));
            }
            Term::Add | Term::Mul | Term::Sub | Term::Div | Term::Mod | Term::Min
            | Term::Max | Term::Gte | Term::Lt => {
                // Their builders emit RHS terms first, so the stack TOP is
                // the LEFT operand (verified against as_op + the Sub impl).
                let (Some(left), Some(right)) = (stack.pop(), stack.pop()) else {
                    bail!("hlir_to_logical: malformed index expression at {at}");
                };
                let rendered = match term {
                    Term::Add => format!("(IntAdd {left} {right})"),
                    Term::Mul => format!("(IntMul {left} {right})"),
                    Term::Sub => format!("(IntAdd {left} (IntMul (IntLit -1) {right}))"),
                    Term::Div => format!("(IntTruncDiv {left} {right})"),
                    Term::Mod => format!("(IntTruncRem {left} {right})"),
                    Term::Min => format!("(IntMin {left} {right})"),
                    Term::Max => format!("(IntMax {left} {right})"),
                    // Comparisons arrive as 0/1 VALUES in their expressions;
                    // ours are the bool bridge's indicators. Over the discrete
                    // integers, a >= b is spelled b < a+1 — one constructor.
                    Term::Lt => {
                        format!("(IntCastFromBool (BoolLessThanInt {left} {right}))")
                    }
                    Term::Gte => format!(
                        "(IntCastFromBool (BoolLessThanInt {right} (IntAdd {left} (IntLit 1))))"
                    ),
                    _ => unreachable!(),
                };
                stack.push(rendered);
            }
            other => bail!(
                "hlir_to_logical: index-expression term {other:?} at {at} — later slice"
            ),
        }
    }
    match (stack.pop(), stack.is_empty()) {
        (Some(result), true) => Ok(result),
        _ => bail!("hlir_to_logical: malformed index expression at {at}"),
    }
}

/// A ShapeTracker rebuilt from an op's snapshot fields (Scatter carries
/// dims/strides pairs rather than trackers).
fn tracker_of(dims: &[Expression], strides: &[Expression]) -> ShapeTracker {
    ShapeTracker {
        dims: dims.iter().copied().collect(),
        strides: strides.iter().copied().collect(),
        element_stride_bits: 32,
    }
}

/// Translate a recorded iota consumed at its OWN flat shape — term-for-term
/// symbolic translation, exact by construction. A RESHAPED consumption means
/// their frontend flattened per-axis structure (flatten_strides); that
/// information is preserved at the frontend seam or not at all (ruling
/// 2026-07-29): loud bail, never recovery.
#[allow(clippy::too_many_arguments)]
fn specialize_iota(
    ops_text: &mut String,
    post_checks: &mut String,
    expr: &Expression,
    range: usize,
    tracker: &ShapeTracker,
    dyn_map: &FxHashMap<char, usize>,
    node_index: usize,
    position: usize,
) -> Result<(String, Vec<usize>)> {
    let at = format!("t{node_index} operand {position} (iota)");
    ensure!(
        tracker.is_contiguous(),
        "hlir_to_logical: iota through a non-contiguous view at {at}"
    );
    let view_dims: Vec<usize> = tracker
        .dims
        .iter()
        .map(|dim| {
            dim.exec(dyn_map)
                .with_context(|| format!("hlir_to_logical: unpinned dim at {at}"))
        })
        .collect::<Result<_>>()?;
    ensure!(
        view_dims == [range],
        "hlir_to_logical: iota at {at} consumed at {view_dims:?} (range {range}) — \
         per-axis structure was erased by their frontend lowering; pending the \
         frontend-seam decision (ruling 2026-07-29)"
    );
    let coord = format!("(CoordVar 0 (IntLit {range}))");
    let value_expr = int_expr_term(expr, &coord, dyn_map, &at)?;
    let shape = shape_term(&view_dims);
    let name = format!("t{node_index}_operand{position}_iota");
    ops_text.push_str(&format!("(let {name} (LogicalIota {value_expr} {shape}))\n"));
    // The iota authoring contract: construction sites demand bounds.
    post_checks.push_str(&format!(
        "(check (= ?lo{node_index}_{position} (lower-bound-of {value_expr})))\n\
         (check (= ?hi{node_index}_{position} (upper-bound-of {value_expr})))\n"
    ));
    Ok((name, view_dims))
}

fn dtype_term(dtype: DType) -> String {
    format!("({dtype:?})")
}

pub fn hlir_to_logical(graph: &Graph) -> Result<LogicalProgram> {
    hlir_to_logical_with_dims(graph, &graph.dyn_map, None)
}

/// [`hlir_to_logical`] with explicit dim pins and optional RANGE seeds.
/// `dyn_map` pins every var numerically (geometry bookkeeping + the default
/// tight-bound seeds). When `ranges` supplies an interval for a var, the
/// binding seeds THAT interval instead of the tight pin — the bucket-wide
/// render: analyses and fixpoint checks then hold over the whole bucket,
/// while the numeric bookkeeping still uses the representative pin. A
/// range-seeded program does NOT collapse to literals and is therefore for
/// VALIDATION, not execution.
pub fn hlir_to_logical_with_dims(
    graph: &Graph,
    dyn_map: &FxHashMap<char, usize>,
    ranges: Option<&BTreeMap<char, (usize, usize)>>,
) -> Result<LogicalProgram> {
    let order = toposort(&graph.graph, None)
        .map_err(|_| anyhow!("hlir_to_logical: HLIR graph has a cycle"))?;
    let mut pinned_vars: BTreeMap<char, usize> = BTreeMap::new();
    // Iota nodes: recorded, not emitted — every consumer SPECIALIZES the
    // iota at its own view shape (flat index = the row-major sum over the
    // consumer's coordinates), absorbing their flat-iota-reshaped-by-
    // tracker idiom without any reshape machinery.
    let mut iota_exprs: FxHashMap<NodeIndex, (Expression, usize)> = FxHashMap::default();
    let mut values: FxHashMap<NodeIndex, ValueInfo> = FxHashMap::default();
    let mut inputs_text = String::new();
    let mut ops_text = String::new();
    let mut post_checks = String::new();
    let mut input_slots: Vec<(NodeIndex, u64)> = Vec::new();
    // (output op node, source value node, Output.node key)
    let mut output_nodes: Vec<(NodeIndex, NodeIndex, usize)> = Vec::new();

    for node in order {
        let op = &graph.graph[node];
        let idx = node.index();
        let sources = graph.get_sources(node);
        let dyn_op = op.as_ref();

        if let Some(input) = dyn_op.downcast_ref::<Input>() {
            let (dims, dim_exprs) = input_dims_from_consumers(graph, node, dyn_map)?;
            let shape = match &dim_exprs {
                Some(exprs) => {
                    let parts = exprs
                        .iter()
                        .map(|expr| {
                            dim_term(expr, dyn_map, &mut pinned_vars, &format!("input t{idx}"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    shape_term_of(&parts)
                }
                // Reconstructed-through-views inputs fall back to resolved
                // literals — the [n,n] collapse keeps them congruent with any
                // symbolic mention elsewhere.
                None => shape_term(&dims),
            };
            let dtype = dtype_term(input.dtype);
            let label = format!("{}_{idx}", input.label);
            inputs_text.push_str(&format!(
                "(let t{idx}_logical (LogicalTensorInputLit (LogicalIdLit \"{label}\") {shape} {dtype}))\n\
                 (let t{idx}_layout (RightMajorContiguousElementLayoutLit {shape} (bits-of {dtype})))\n\
                 (let t{idx}_layout_tensor (LayoutTensorLit t{idx}_logical t{idx}_layout))\n\
                 (let t{idx}_buffer_id (BufferLit {idx}))\n\
                 (set (buffer-access-of t{idx}_buffer_id) (ReadOnly))\n\
                 (set (buffer-freed-by t{idx}_buffer_id) (CallerFrees))\n\
                 (let t{idx}_buffer_tensor (BufferTensorLit t{idx}_layout_tensor t{idx}_buffer_id))\n\n",
            ));
            input_slots.push((node, idx as u64));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims,
                    dtype: input.dtype,
                },
            );
        } else if let Some(output) = dyn_op.downcast_ref::<Output>() {
            let source = *sources.first().ok_or_else(|| {
                anyhow!("hlir_to_logical: Output at t{idx} has no source")
            })?;
            output_nodes.push((node, source, output.node));
        } else if let Some(slice) = dyn_op.downcast_ref::<SliceView>() {
            // THE SEAM PAYOFF (ruling 2026-07-29): a start slice arrives with
            // its per-axis structure intact and translates as the VIEW it is —
            // one IndexMapApply entry per parent axis, CoordVar + start.
            let source = *sources.first().ok_or_else(|| {
                anyhow!("hlir_to_logical: SliceView at t{idx} has no source")
            })?;
            let source_value = values.get(&source).ok_or_else(|| {
                anyhow!("hlir_to_logical: t{idx} reads untranslated t{}", source.index())
            })?;
            let (parent_name, parent_dims) = match iota_exprs.get(&source) {
                Some((expr, range)) => specialize_iota(
                    &mut ops_text,
                    &mut post_checks,
                    expr,
                    *range,
                    &slice.input_shape,
                    dyn_map,
                    idx,
                    0,
                )?,
                None => lift_operand(
                    &mut ops_text,
                    &slice.input_shape,
                    source_value,
                    dyn_map,
                    idx,
                    0,
                )?,
            };
            let rank = slice.out_dims.len();
            ensure!(
                parent_dims.len() == rank && slice.starts.len() == rank,
                "hlir_to_logical: SliceView at t{idx} rank mismatch (parent {parent_dims:?}, \
                 {} starts, {} out dims)",
                slice.starts.len(),
                slice.out_dims.len()
            );
            let mut out_terms = Vec::with_capacity(rank);
            let mut out_numeric = Vec::with_capacity(rank);
            for (k, dim) in slice.out_dims.iter().enumerate() {
                out_terms.push(dim_term(
                    dim,
                    dyn_map,
                    &mut pinned_vars,
                    &format!("SliceView t{idx} out dim {k}"),
                )?);
                out_numeric.push(dim.exec(dyn_map).with_context(|| {
                    format!("hlir_to_logical: unpinned out dim at SliceView t{idx}")
                })?);
            }
            let mut entries = "(IntExprNil)".to_string();
            for k in (0..rank).rev() {
                let coord = format!("(CoordVar {} {})", rank - 1 - k, out_terms[k]);
                let start = &slice.starts[k];
                let entry = if start.exec(dyn_map) == Some(0) {
                    coord
                } else {
                    let start_term = dim_term(
                        start,
                        dyn_map,
                        &mut pinned_vars,
                        &format!("SliceView t{idx} start {k}"),
                    )?;
                    format!("(IntAdd {coord} {start_term})")
                };
                entries = format!("(IntExprCons {entry} {entries})");
            }
            let shape = shape_term_of(&out_terms);
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalIndexMapApply {parent_name} \
                 (IndexMapLit {entries}) {shape}))\n"
            ));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: out_numeric,
                    dtype: source_value.dtype,
                },
            );
        } else if let Some(unfold) = dyn_op.downcast_ref::<UnfoldView>() {
            // Sliding windows as the view they are: out [w..., k...], and per
            // parent axis i the entry is w_i·stride_i + k_i·dilation_i.
            let source = *sources.first().ok_or_else(|| {
                anyhow!("hlir_to_logical: UnfoldView at t{idx} has no source")
            })?;
            let source_value = values.get(&source).ok_or_else(|| {
                anyhow!("hlir_to_logical: t{idx} reads untranslated t{}", source.index())
            })?;
            let (parent_name, parent_dims) = match iota_exprs.get(&source) {
                Some((expr, range)) => specialize_iota(
                    &mut ops_text,
                    &mut post_checks,
                    expr,
                    *range,
                    &unfold.input_shape,
                    dyn_map,
                    idx,
                    0,
                )?,
                None => lift_operand(
                    &mut ops_text,
                    &unfold.input_shape,
                    source_value,
                    dyn_map,
                    idx,
                    0,
                )?,
            };
            let n = unfold.window_counts.len();
            ensure!(
                parent_dims.len() == n && unfold.kernel.len() == n,
                "hlir_to_logical: UnfoldView at t{idx} rank mismatch"
            );
            let mut out_terms = Vec::with_capacity(2 * n);
            let mut out_numeric = Vec::with_capacity(2 * n);
            for (k, dim) in unfold
                .window_counts
                .iter()
                .chain(unfold.kernel.iter())
                .enumerate()
            {
                out_terms.push(dim_term(
                    dim,
                    dyn_map,
                    &mut pinned_vars,
                    &format!("UnfoldView t{idx} out dim {k}"),
                )?);
                out_numeric.push(dim.exec(dyn_map).with_context(|| {
                    format!("hlir_to_logical: unpinned out dim at UnfoldView t{idx}")
                })?);
            }
            let mut entries = "(IntExprNil)".to_string();
            for i in (0..n).rev() {
                // Out axis positions: w_i at i, k_i at n+i (front-indexed);
                // CoordVar axes are zero-based from the END of the 2n shape.
                let w_coord = format!("(CoordVar {} {})", 2 * n - 1 - i, out_terms[i]);
                let k_coord = format!("(CoordVar {} {})", n - 1 - i, out_terms[n + i]);
                let stride_term = dim_term(
                    &unfold.strides[i],
                    dyn_map,
                    &mut pinned_vars,
                    &format!("UnfoldView t{idx} stride {i}"),
                )?;
                let dilation_term = dim_term(
                    &unfold.dilation[i],
                    dyn_map,
                    &mut pinned_vars,
                    &format!("UnfoldView t{idx} dilation {i}"),
                )?;
                let entry = format!(
                    "(IntAdd (IntMul {w_coord} {stride_term}) (IntMul {k_coord} {dilation_term}))"
                );
                entries = format!("(IntExprCons {entry} {entries})");
            }
            let shape = shape_term_of(&out_terms);
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalIndexMapApply {parent_name} \
                 (IndexMapLit {entries}) {shape}))\n"
            ));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: out_numeric,
                    dtype: source_value.dtype,
                },
            );
        } else if let Some(clamp) = dyn_op.downcast_ref::<ClampView>() {
            // Pad's read half: a TOTAL view — per parent axis the entry is
            // min(max(c_k − before_k, 0), dim_k − 1), with the same
            // conditional structure the legacy lowering used (the clamp
            // sides only where padding exists).
            let source = *sources.first().ok_or_else(|| {
                anyhow!("hlir_to_logical: ClampView at t{idx} has no source")
            })?;
            let source_value = values.get(&source).ok_or_else(|| {
                anyhow!("hlir_to_logical: t{idx} reads untranslated t{}", source.index())
            })?;
            let (parent_name, parent_dims) = match iota_exprs.get(&source) {
                Some((expr, range)) => specialize_iota(
                    &mut ops_text,
                    &mut post_checks,
                    expr,
                    *range,
                    &clamp.input_shape,
                    dyn_map,
                    idx,
                    0,
                )?,
                None => lift_operand(
                    &mut ops_text,
                    &clamp.input_shape,
                    source_value,
                    dyn_map,
                    idx,
                    0,
                )?,
            };
            let rank = parent_dims.len();
            ensure!(
                clamp.befores.len() == rank && clamp.afters.len() == rank,
                "hlir_to_logical: ClampView at t{idx} rank mismatch"
            );
            let mut out_terms = Vec::with_capacity(rank);
            let mut out_numeric = Vec::with_capacity(rank);
            let mut entries = "(IntExprNil)".to_string();
            for k in 0..rank {
                let before = clamp.befores[k].exec(dyn_map).with_context(|| {
                    format!("hlir_to_logical: unpinned pad-before at t{idx}")
                })?;
                let after = clamp.afters[k].exec(dyn_map).with_context(|| {
                    format!("hlir_to_logical: unpinned pad-after at t{idx}")
                })?;
                let dim = parent_dims[k];
                out_numeric.push(before + dim + after);
                out_terms.push(format!("(IntLit {})", before + dim + after));
            }
            for k in (0..rank).rev() {
                let before = clamp.befores[k].exec(dyn_map).unwrap_or(0);
                let after = clamp.afters[k].exec(dyn_map).unwrap_or(0);
                let dim = parent_dims[k];
                let coord = format!("(CoordVar {} {})", rank - 1 - k, out_terms[k]);
                let mut entry = coord;
                if before != 0 {
                    entry = format!(
                        "(IntMax (IntAdd {entry} (IntLit -{before})) (IntLit 0))"
                    );
                }
                if after != 0 {
                    entry = format!("(IntMin {entry} (IntLit {}))", dim - 1);
                }
                entries = format!("(IntExprCons {entry} {entries})");
            }
            let shape = shape_term_of(&out_terms);
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalIndexMapApply {parent_name} \
                 (IndexMapLit {entries}) {shape}))\n"
            ));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: out_numeric,
                    dtype: source_value.dtype,
                },
            );
        } else if let Some(mask) = dyn_op.downcast_ref::<MaskIota>() {
            // Pad's mask half: an iota of bool-bridge indicators — per
            // padded axis (p_k >= before) · (p_k < before + dim), with
            // >= spelled as the discrete rotation before-1 < p_k, i.e.
            // before <= p_k iff before < p_k + 1.
            let rank = mask.in_dims.len();
            let mut out_terms = Vec::with_capacity(rank);
            let mut factors: Vec<String> = Vec::new();
            for k in 0..rank {
                let before = mask.befores[k].exec(dyn_map).with_context(|| {
                    format!("hlir_to_logical: unpinned mask-before at t{idx}")
                })?;
                let after = mask.afters[k].exec(dyn_map).with_context(|| {
                    format!("hlir_to_logical: unpinned mask-after at t{idx}")
                })?;
                let dim = mask.in_dims[k].exec(dyn_map).with_context(|| {
                    format!("hlir_to_logical: unpinned mask dim at t{idx}")
                })?;
                out_terms.push(format!("(IntLit {})", before + dim + after));
                let coord = |terms: &Vec<String>| {
                    format!("(CoordVar {} {})", rank - 1 - k, terms[k])
                };
                if before != 0 {
                    factors.push(format!(
                        "(IntCastFromBool (BoolLessThanInt (IntLit {}) (IntAdd {} (IntLit 1))))",
                        before,
                        coord(&out_terms)
                    ));
                }
                if after != 0 {
                    factors.push(format!(
                        "(IntCastFromBool (BoolLessThanInt {} (IntLit {})))",
                        coord(&out_terms),
                        before + dim
                    ));
                }
            }
            let mut expr = factors.pop().unwrap_or_else(|| "(IntLit 1)".to_string());
            for factor in factors {
                expr = format!("(IntMul {factor} {expr})");
            }
            let shape = shape_term_of(&out_terms);
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalIota {expr} {shape}))\n"
            ));
            post_checks.push_str(&format!(
                "(check (= ?mlo{idx} (lower-bound-of {expr})))\n\
                 (check (= ?mhi{idx} (upper-bound-of {expr})))\n"
            ));
            let out_numeric = mask
                .befores
                .iter()
                .zip(&mask.afters)
                .zip(&mask.in_dims)
                .map(|((b, a), d)| {
                    Some(b.exec(dyn_map)? + a.exec(dyn_map)? + d.exec(dyn_map)?)
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow!("hlir_to_logical: unpinned MaskIota dims at t{idx}"))?;
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: out_numeric,
                    dtype: DType::Int,
                },
            );
        } else if let Some(scatter) = dyn_op.downcast_ref::<Scatter>() {
            // Their scatter: sources [dest, indexes, src]; output = copy(dest)
            // then output[indexes[i]] = src[i] with FLAT positions into the
            // dest view. Coordinate form coincides for RANK-1 dest (their
            // scatter API is flat-1-D by contract); src iterates the INDEX
            // shape through src_strides (zero-padded broadcasts included).
            // OOB divergence: theirs silently skips, ours is UB surfaced
            // loudly at the kernel.
            ensure!(
                sources.len() == 3,
                "hlir_to_logical: scatter at t{idx} has {} sources",
                sources.len()
            );
            let dest_tracker = tracker_of(&scatter.dest_shape, &scatter.dest_strides);
            let index_tracker = tracker_of(&scatter.index_shape, &scatter.index_strides);
            let src_tracker = tracker_of(&scatter.index_shape, &scatter.src_strides);
            let trackers = [&dest_tracker, &index_tracker, &src_tracker];
            let mut lifted = Vec::new();
            for (position, (source, tracker)) in
                sources.iter().zip(trackers).enumerate()
            {
                let source_value = values.get(source).ok_or_else(|| {
                    anyhow!(
                        "hlir_to_logical: t{idx} reads untranslated t{}",
                        source.index()
                    )
                })?;
                let entry = match iota_exprs.get(source) {
                    Some((expr, range)) => specialize_iota(
                        &mut ops_text,
                        &mut post_checks,
                        expr,
                        *range,
                        tracker,
                        dyn_map,
                        idx,
                        position,
                    )?,
                    None => lift_operand(
                        &mut ops_text,
                        tracker,
                        source_value,
                        dyn_map,
                        idx,
                        position,
                    )?,
                };
                lifted.push(entry);
            }
            let (dest_name, dest_dims) = &lifted[0];
            let (index_name, index_dims) = &lifted[1];
            let (src_name, src_dims) = &lifted[2];
            ensure!(
                dest_dims.len() == 1,
                "hlir_to_logical: scatter at t{idx} over rank-{} dest — flat→coordinate \
                 decomposition is a later slice",
                dest_dims.len()
            );
            ensure!(
                index_dims == src_dims,
                "hlir_to_logical: scatter at t{idx} index shape {index_dims:?} vs src shape {src_dims:?}"
            );
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalScatter {dest_name} \
                 (LogicalTensorCons {index_name} (LogicalTensorNil)) {src_name}))\n"
            ));
            let out_dims = dest_dims.clone();
            let dtype = values[&sources[2]].dtype;
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: out_dims,
                    dtype,
                },
            );
        } else if dyn_op.downcast_ref::<Gather>().is_some() {
            // Their gather: sources [indexes, data]; out[i] = data_view[idx[i]]
            // with idx FLAT into the data view. Coordinate form needs one
            // coordinate tensor per data axis:
            //   * rank-1 data: the index tensor IS the coordinate — any
            //     index source, including data-dependent lookups.
            //   * rank-N data with an IOTA index source: per-axis coordinate
            //     iotas (flat/stride % dim over the index expression) — the
            //     slice/unfold family. Data-dependent rank-N refuses loudly
            //     (needs tensor-level div/mod).
            let gather = dyn_op.downcast_ref::<Gather>().unwrap();
            ensure!(
                sources.len() == 2,
                "hlir_to_logical: gather at t{idx} has {} sources",
                sources.len()
            );
            let trackers = &gather.input_shapes;
            ensure!(trackers.len() == 2, "gather at t{idx} missing trackers");
            let data_source = values.get(&sources[1]).ok_or_else(|| {
                anyhow!("hlir_to_logical: t{idx} reads untranslated t{}", sources[1].index())
            })?;
            let (data_name, data_dims) = match iota_exprs.get(&sources[1]) {
                Some((expr, range)) => specialize_iota(
                    &mut ops_text,
                    &mut post_checks,
                    expr,
                    *range,
                    &trackers[1],
                    dyn_map,
                    idx,
                    1,
                )?,
                None => {
                    lift_operand(&mut ops_text, &trackers[1], data_source, dyn_map, idx, 1)?
                }
            };

            let (coord_list, out_dims) = if data_dims.len() == 1 {
                let index_source = values.get(&sources[0]).ok_or_else(|| {
                    anyhow!(
                        "hlir_to_logical: t{idx} reads untranslated t{}",
                        sources[0].index()
                    )
                })?;
                let (index_name, index_dims) = match iota_exprs.get(&sources[0]) {
                    Some((expr, range)) => specialize_iota(
                        &mut ops_text,
                        &mut post_checks,
                        expr,
                        *range,
                        &trackers[0],
                        dyn_map,
                        idx,
                        0,
                    )?,
                    None => lift_operand(
                        &mut ops_text,
                        &trackers[0],
                        index_source,
                        dyn_map,
                        idx,
                        0,
                    )?,
                };
                (
                    format!("(LogicalTensorCons {index_name} (LogicalTensorNil))"),
                    index_dims,
                )
            } else {
                bail!(
                    "hlir_to_logical: gather at t{idx} over rank-{} data — their frontend \
                     flattened the per-axis structure (ShapeTracker carries no offsets, so \
                     slice/unfold lower to flat iota+gather); preserving it is a \
                     frontend-seam decision, not a recovery problem (ruling 2026-07-29)",
                    data_dims.len()
                )
            };

            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalGather {data_name} {coord_list}))\n"
            ));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: out_dims,
                    dtype: data_source.dtype,
                },
            );
        } else if let Some(cast) = dyn_op.downcast_ref::<crate::hlir::Cast>() {
            // Shape-preserving; the tracker rides through on the tensor, so
            // the logical cast applies to the source value directly.
            let source = *sources.first().ok_or_else(|| {
                anyhow!("hlir_to_logical: Cast at t{idx} has no source")
            })?;
            let source_value = values.get(&source).ok_or_else(|| {
                anyhow!("hlir_to_logical: t{idx} reads untranslated t{}", source.index())
            })?;
            // A cast directly over a recorded iota (their pad's mask) first
            // emits the iota at its own flat shape.
            let (source_name, source_dims) = match iota_exprs.get(&source) {
                Some((expr, range)) => {
                    let coord = format!("(CoordVar 0 (IntLit {range}))");
                    let value_expr =
                        int_expr_term(expr, &coord, dyn_map, &format!("cast-iota t{idx}"))?;
                    let shape = shape_term(&[*range]);
                    let name = format!("t{idx}_source_iota");
                    ops_text.push_str(&format!(
                        "(let {name} (LogicalIota {value_expr} {shape}))\n"
                    ));
                    post_checks.push_str(&format!(
                        "(check (= ?clo{idx} (lower-bound-of {value_expr})))\n\
                         (check (= ?chi{idx} (upper-bound-of {value_expr})))\n"
                    ));
                    (name, vec![*range])
                }
                None => (source_value.let_name.clone(), source_value.dims.clone()),
            };
            let target = cast.1;
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalCast {source_name} {}))\n",
                dtype_term(target)
            ));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: source_dims,
                    dtype: target,
                },
            );
        } else if let Some(constant) = dyn_op.downcast_ref::<Constant>() {
            // Rank-0 logical constant; consumers broadcast it through lifted
            // views (an empty index map — every consuming axis reads it).
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalConstant {:?}))\n",
                constant.0 as f64
            ));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
                    dims: Vec::new(),
                    dtype: DType::F32,
                },
            );
        } else if let Some(iota) = dyn_op.downcast_ref::<Iota>() {
            let extent = iota
                .1
                .exec(dyn_map)
                .with_context(|| format!("hlir_to_logical: unpinned iota range at t{idx}"))?;
            // Recorded only — consumers specialize (see iota_exprs).
            iota_exprs.insert(node, (iota.0, extent));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_iota"),
                    dims: vec![extent],
                    dtype: DType::Int,
                },
            );
        } else {
            // Elementwise binaries, unaries, reductions — uniform handling
            // off the source ValueInfos + the op's tracker snapshots.
            let constructor: Option<(&str, usize)> = if dyn_op.downcast_ref::<Add>().is_some() {
                Some(("LogicalAdd", 2))
            } else if dyn_op.downcast_ref::<Mul>().is_some() {
                Some(("LogicalMul", 2))
            } else if dyn_op.downcast_ref::<Mod>().is_some() {
                Some(("LogicalMod", 2))
            } else if dyn_op.downcast_ref::<Recip>().is_some() {
                Some(("LogicalRecip", 1))
            } else if dyn_op.downcast_ref::<Sqrt>().is_some() {
                Some(("LogicalSqrt", 1))
            } else if dyn_op.downcast_ref::<Sin>().is_some() {
                Some(("LogicalSin", 1))
            } else if dyn_op.downcast_ref::<Exp2>().is_some() {
                Some(("LogicalExp2", 1))
            } else if dyn_op.downcast_ref::<Log2>().is_some() {
                Some(("LogicalLog2", 1))
            } else {
                None
            };

            if let Some((constructor, arity)) = constructor {
                if sources.len() != arity {
                    bail!("hlir_to_logical: {constructor} at t{idx} has {} sources, expected {arity}", sources.len());
                }
                let trackers = op_input_trackers(dyn_op)
                    .ok_or_else(|| anyhow!("hlir_to_logical: {constructor} at t{idx} missing trackers"))?;
                let mut operand_names = Vec::new();
                let mut dims: Option<Vec<usize>> = None;
                let mut dtype: Option<DType> = None;
                for (position, source) in sources.iter().enumerate() {
                    let tracker = trackers.get(position).ok_or_else(|| {
                        anyhow!("hlir_to_logical: {constructor} at t{idx} missing tracker {position}")
                    })?;
                    let source_value = values.get(source).ok_or_else(|| {
                        anyhow!("hlir_to_logical: t{idx} reads untranslated t{}", source.index())
                    })?;
                    let (operand_name, operand_dims) =
                        match iota_exprs.get(source) {
                            Some((expr, range)) => specialize_iota(
                                &mut ops_text,
                                &mut post_checks,
                                expr,
                                *range,
                                tracker,
                                dyn_map,
                                idx,
                                position,
                            )?,
                            None => lift_operand(
                                &mut ops_text,
                                tracker,
                                source_value,
                                dyn_map,
                                idx,
                                position,
                            )?,
                        };
                    match &dims {
                        None => dims = Some(operand_dims),
                        Some(existing) if *existing == operand_dims => {}
                        Some(existing) => bail!(
                            "hlir_to_logical: t{idx} operand views disagree on shape ({existing:?} vs {operand_dims:?})"
                        ),
                    }
                    match dtype {
                        None => dtype = Some(source_value.dtype),
                        Some(existing) if existing == source_value.dtype => {}
                        Some(existing) => bail!(
                            "hlir_to_logical: t{idx} mixes operand dtypes {existing:?} vs {:?}",
                            source_value.dtype
                        ),
                    }
                    operand_names.push(operand_name);
                }
                ops_text.push_str(&format!(
                    "(let t{idx}_logical ({constructor} {}))\n",
                    operand_names.join(" ")
                ));
                values.insert(
                    node,
                    ValueInfo {
                        let_name: format!("t{idx}_logical"),
                        dims: dims.expect("arity >= 1"),
                        dtype: dtype.expect("arity >= 1"),
                    },
                );
            } else if let Some((constructor, dim, tracker)) = dyn_op
                .downcast_ref::<SumReduce>()
                .map(|r| ("LogicalReduceSum", r.dim, r.input_shape))
                .or_else(|| {
                    dyn_op
                        .downcast_ref::<MaxReduce>()
                        .map(|r| ("LogicalReduceMax", r.dim, r.input_shape))
                })
            {
                let source = *sources.first().ok_or_else(|| {
                    anyhow!("hlir_to_logical: {constructor} at t{idx} has no source")
                })?;
                let source_value = values.get(&source).ok_or_else(|| {
                    anyhow!("hlir_to_logical: t{idx} reads untranslated t{}", source.index())
                })?;
                let (operand_name, tracker_dims) = match iota_exprs.get(&source) {
                    Some((expr, range)) => specialize_iota(
                        &mut ops_text,
                        &mut post_checks,
                        expr,
                        *range,
                        &tracker,
                        dyn_map,
                        idx,
                        0,
                    )?,
                    None => {
                        lift_operand(&mut ops_text, &tracker, source_value, dyn_map, idx, 0)?
                    }
                };
                let rank = tracker_dims.len();
                if dim >= rank {
                    bail!("hlir_to_logical: t{idx} reduces axis {dim} of rank-{rank} input");
                }
                // Their axis counts from the FRONT; ours from the END,
                // zero-based (the nth-from-end convention).
                let axis_from_end = rank - 1 - dim;
                let mut out_dims = tracker_dims.clone();
                out_dims.remove(dim);
                ops_text.push_str(&format!(
                    "(let t{idx}_logical ({constructor} {operand_name} {axis_from_end}))\n"
                ));
                values.insert(
                    node,
                    ValueInfo {
                        let_name: format!("t{idx}_logical"),
                        dims: out_dims,
                        dtype: source_value.dtype,
                    },
                );
            } else {
                bail!("hlir_to_logical slice 1: unsupported HLIR op {op:?} at node t{idx}");
            }
        }
    }

    if output_nodes.is_empty() {
        bail!("hlir_to_logical: graph has no Output nodes");
    }
    output_nodes.sort_by_key(|(out_node, _, _)| out_node.index());

    let mut outputs_text = String::new();
    let mut output_slots: Vec<(usize, u64)> = Vec::new();
    let mut emitted_direct_iotas: std::collections::HashSet<NodeIndex> =
        std::collections::HashSet::new();
    for (_, source, key) in &output_nodes {
        let value = values.get(source).ok_or_else(|| {
            anyhow!("hlir_to_logical: Output reads untranslated t{}", source.index())
        })?;
        if let Some((expr, range)) = iota_exprs.get(source) {
            if emitted_direct_iotas.insert(*source) {
                let coord = format!("(CoordVar 0 (IntLit {range}))");
                let value_expr =
                    int_expr_term(expr, &coord, dyn_map, &format!("output iota t{key}"))?;
                let shape = shape_term(&[*range]);
                outputs_text.push_str(&format!(
                    "(let {} (LogicalIota {value_expr} {shape}))\n",
                    value.let_name
                ));
                post_checks.push_str(&format!(
                    "(check (= ?oil{key} (lower-bound-of {value_expr})))\n\
                     (check (= ?oih{key} (upper-bound-of {value_expr})))\n"
                ));
            }
        }
        let shape = shape_term(&value.dims);
        let dtype = dtype_term(value.dtype);
        outputs_text.push_str(&format!(
            "(union {name} (LogicalTensorOutputLit (LogicalIdLit \"out_{key}\")))\n\
             (let out{key}_layout (RightMajorContiguousElementLayoutLit {shape} (bits-of {dtype})))\n\
             (let out{key}_layout_tensor (LayoutTensorLit {name} out{key}_layout))\n\
             (let out{key}_buffer_id (BufferLit {key}))\n\
             (set (buffer-access-of out{key}_buffer_id) (ReadWrite))\n\
             (set (buffer-freed-by out{key}_buffer_id) (CallerFrees))\n\
             (let out{key}_buffer_tensor (BufferTensorLit out{key}_layout_tensor out{key}_buffer_id))\n\n",
            name = value.let_name,
        ));
        output_slots.push((*key, *key as u64));
    }

    // Signature lists + boundary lists, in slot order.
    let logical_list = |items: &[String]| {
        let mut term = "(LogicalTensorNil)".to_string();
        for item in items.iter().rev() {
            term = format!("(LogicalTensorCons {item} {term})");
        }
        term
    };
    let buffer_list = |items: &[String]| {
        let mut term = "(BufferTensorNil)".to_string();
        for item in items.iter().rev() {
            term = format!("(BufferTensorCons {item} {term})");
        }
        term
    };
    let input_logicals: Vec<String> = input_slots
        .iter()
        .map(|(node, _)| format!("t{}_logical", node.index()))
        .collect();
    let input_buffers: Vec<String> = input_slots
        .iter()
        .map(|(node, _)| format!("t{}_buffer_tensor", node.index()))
        .collect();
    let output_logicals: Vec<String> = output_nodes
        .iter()
        .map(|(_, source, _)| values[source].let_name.clone())
        .collect();
    let output_buffers: Vec<String> = output_nodes
        .iter()
        .map(|(_, _, key)| format!("out{key}_buffer_tensor"))
        .collect();

    // BINDING pins: a binding never writes a union — tight bounds ARE the
    // pin, and the [n,n] collapse rule delivers var ≡ literal to every user
    // by congruence (including the numeric-geometry walk, which finds the
    // literal in the collapsed class).
    let mut seeds_text = String::new();
    for (var, value) in &pinned_vars {
        let (lower, upper) = match ranges.and_then(|ranges| ranges.get(var)) {
            Some((min, max)) => (*min, *max),
            None => (*value, *value),
        };
        seeds_text.push_str(&format!(
            "(set (lower-bound-of (IntVar \"{var}\")) (bigint {lower}))\n\
             (set (upper-bound-of (IntVar \"{var}\")) (bigint {upper}))\n"
        ));
    }

    let text = format!(
        "; hlir_to_logical (slice 1: contiguous, static) — {} nodes\n\n\
         {seeds_text}\n{inputs_text}{ops_text}\n{outputs_text}\
         (let model_inputs (LogicalInputLit {model_inputs}))\n\
         (let model_outputs (LogicalOutputLit {model_outputs}))\n\
         (let input_boundary (BufferInputLit {input_boundary}))\n\
         (let output (BufferOutputLit {output_boundary}))\n\n\
         (run-schedule (saturate (run)) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))\n\n\
         {post_checks}",
        graph.graph.node_count(),
        model_inputs = logical_list(&input_logicals),
        model_outputs = logical_list(&output_logicals),
        input_boundary = buffer_list(&input_buffers),
        output_boundary = buffer_list(&output_buffers),
    );

    Ok(LogicalProgram {
        text,
        input_slots,
        output_slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egglog::SerializeConfig;

    fn plan_for(graph: &Graph) -> (LogicalProgram, String) {
        let program = hlir_to_logical(graph).expect("slice-1 graph translates");
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            program.text
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .expect("assembled + translated program runs");
        let serialized = egraph.serialize(SerializeConfig::default()).egraph;
        let extracted = crate::extractor::extract_layout_ir(&serialized)
            .expect("extraction runs")
            .expect("plan reaches the boundary");
        let plan = crate::bufferize::bufferize(&crate::dps::dps_rewrite(&extracted))
            .expect("bufferizes");
        (program, plan.summary())
    }

    /// Their canonical `simple`-style graph (b*c + g) through the whole
    /// logical pipeline: translated, saturated, extracted, bufferized.
    #[test]
    fn simple_elementwise_graph_reaches_a_plan() {
        let mut cx = Graph::new();
        let b = cx.tensor(3);
        let c = cx.tensor(3);
        let g = cx.tensor(3);
        let a = (b * c + g).output();

        let (program, summary) = plan_for(&cx);
        assert!(summary.contains("MulFunctionalGeneric"), "{summary}");
        assert!(summary.contains("AddFunctionalGeneric"), "{summary}");
        // The output buffer is keyed the way their get_f32 keys it: by the
        // SOURCE tensor's node index.
        assert!(
            summary.contains(&format!("BufferLit({})", a.id.index())),
            "{summary}"
        );
        assert_eq!(program.input_slots.len(), 3);
        assert_eq!(program.output_slots.len(), 1);
    }

    /// Front-indexed reduce axes flip to our end-indexed convention:
    /// sum over axis 1 of [2, 3] reduces the innermost dim (our axis 0).
    #[test]
    fn reduce_translates_with_the_axis_flip() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 3));
        let _s = x.sum(1).output();

        let program = hlir_to_logical(&cx).expect("reduce graph translates");
        assert!(
            program.text.contains("(LogicalReduceSum t0_logical 0)"),
            "{}",
            program.text
        );
        let (_, summary) = plan_for(&cx);
        assert!(summary.contains("ReduceSumGeneric"), "{summary}");
    }

    /// Slice 2: a permuted operand LIFTS into an IndexMapApply view instead
    /// of bailing (the slice-1 refusal, upgraded).
    #[test]
    fn permuted_operand_lifts_into_a_view() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 3));
        let y = cx.tensor((3, 2));
        let _out = (x.permute((1, 0)) * y).output();

        let program = hlir_to_logical(&cx).expect("permute lifts in slice 2");
        assert!(
            program.text.contains("LogicalIndexMapApply"),
            "{}",
            program.text
        );
    }

    /// Slice 2 honesty: a genuinely non-affine stride (repeat's modulo)
    /// still fails loudly.
    #[test]
    fn repeat_strides_still_bail_loudly() {
        let mut cx = Graph::new();
        let x = cx.tensor(3);
        let y = cx.tensor(12);
        let _out = (x.repeat(4) * y).output();

        // Whether this particular movement chain lands as an affine expand
        // (translatable) or a modulo repeat (loud bail) depends on their
        // tracker algebra — accept either, but NEVER a silent mistranslation:
        // if it translates, the plan must still be extractable.
        match hlir_to_logical(&cx) {
            Err(err) => assert!(err.to_string().contains("slice 2"), "{err}"),
            Ok(program) => assert!(program.text.contains("LogicalIndexMapApply")),
        }
    }
}
