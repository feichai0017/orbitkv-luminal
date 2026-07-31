//! M3 Step 0: the logical-model recorder.
//!
//! GraphTensor methods emit their LOGICAL ops here as the graph is built,
//! beside their HLIR emission — the two worlds coexist until the interim
//! translator retires family by family (each retirement gated by the
//! certification test asserting recorder ≡ translator at the e-class
//! level). The deep point (ruling 2026-07-30): movement methods emit
//! `IndexMapApply` views DIRECTLY from their own parameters, at the source
//! of truth — replacing tracker-lift reconstruction entirely.
//!
//! Model/binding split (M3 Step 1): the recorder emits MODEL text only —
//! input declarations, ops, output naming, signature lists. Boundary
//! vocabulary (layouts, buffers, access, freed-by, Bool8 casts) is the
//! runtime binding generator's business (`reference_binding`), never the
//! model's.
//!
//! Coverage is honest: any construct the recorder does not understand
//! POISONS it with a reason — the first reason wins, the native path
//! refuses loudly at load, and their pipeline is untouched. Every operand
//! resolution cross-checks the tensor's tracker dims against the recorded
//! dims, so a direct tracker mutation the recorder never saw (the ways a
//! frontend method can bypass the movement API) poisons instead of
//! silently mistranslating.

use crate::dtype::DType;
use crate::shape::{Expression, Term};
use rustc_hash::FxHashMap;

/// Handle to a tracker-level view value. Lives on `GraphTensor` (`Copy`);
/// `None` means "my logical value is the recorder's value for my node id".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewId(pub u32);

/// One index-map entry, in-memory. Movement composition happens on this
/// tree — substituting our OWN just-emitted terms, never reconstructing
/// from strides.
#[derive(Debug, Clone, PartialEq)]
pub enum MapEntry {
    /// The consuming view's coordinate, zero-based FROM THE END (the
    /// de Bruijn house convention), with its extent.
    Coord { from_end: usize, extent: Expression },
    /// A dim-expression literal (a number or a symbolic dim var).
    Lit(Expression),
    Add(Box<MapEntry>, Box<MapEntry>),
    Mul(Box<MapEntry>, Expression),
    Div(Box<MapEntry>, Expression),
    Rem(Box<MapEntry>, Expression),
    Min(Box<MapEntry>, Box<MapEntry>),
    Max(Box<MapEntry>, Box<MapEntry>),
}

impl MapEntry {
    /// Substitute every Coord leaf (a previous-out coordinate) with its
    /// replacement in the new out space — the movement-composition step.
    fn substitute(&self, replacement: &[MapEntry], prev_rank: usize) -> MapEntry {
        match self {
            MapEntry::Coord { from_end, .. } => {
                replacement[prev_rank - 1 - from_end].clone()
            }
            MapEntry::Lit(value) => MapEntry::Lit(value.clone()),
            MapEntry::Add(a, b) => MapEntry::Add(
                Box::new(a.substitute(replacement, prev_rank)),
                Box::new(b.substitute(replacement, prev_rank)),
            ),
            MapEntry::Mul(a, e) => {
                MapEntry::Mul(Box::new(a.substitute(replacement, prev_rank)), e.clone())
            }
            MapEntry::Div(a, e) => {
                MapEntry::Div(Box::new(a.substitute(replacement, prev_rank)), e.clone())
            }
            MapEntry::Rem(a, e) => {
                MapEntry::Rem(Box::new(a.substitute(replacement, prev_rank)), e.clone())
            }
            MapEntry::Min(a, b) => MapEntry::Min(
                Box::new(a.substitute(replacement, prev_rank)),
                Box::new(b.substitute(replacement, prev_rank)),
            ),
            MapEntry::Max(a, b) => MapEntry::Max(
                Box::new(a.substitute(replacement, prev_rank)),
                Box::new(b.substitute(replacement, prev_rank)),
            ),
        }
    }
}

/// A tracker-level view: `base_node` seen through `entries` (listed from
/// the PARENT's outermost axis inward, the house map convention), with the
/// view's own output dims.
#[derive(Debug, Clone)]
pub struct ViewValue {
    pub base_node: usize,
    pub entries: Vec<MapEntry>,
    pub dims: Vec<Expression>,
}

/// A movement transform, as the frontend method states it — its own
/// parameters, not a tracker diff.
#[derive(Debug, Clone)]
pub enum Movement {
    /// out dim i = in dim axes[i] (front-based, their convention).
    Permute(Vec<usize>),
    /// New broadcast dim inserted at front position `axis`.
    ExpandDim { axis: usize, size: Expression },
    /// Size-1 front dim at `axis` removed (the squeeze).
    RemoveDim { axis: usize },
    /// dims[axis] = old/inner (outer), inner inserted after (their
    /// split_dims): parent coord = outer·inner_size + inner.
    SplitDims { axis: usize, inner: Expression },
    /// axis2 moved adjacent then merged into axis1 (their merge_dims):
    /// axis1 reads merged/inner, axis2 reads merged%inner.
    MergeDims { axis1: usize, axis2: usize },
    /// Per-axis tile (their repeat): dim → dim·r, coord reads % old dim.
    Repeat(Vec<Expression>),
    /// Zero-start slice: same coords, smaller extents (in-bounds shrink).
    Shrink { new_dims: Vec<Expression> },
}

#[derive(Debug, Clone)]
struct NodeValue {
    name: String,
    dims: Vec<Expression>,
    #[allow(dead_code)]
    dtype: DType,
}

#[derive(Debug, Default)]
pub struct LogicalRecorder {
    /// HLIR node index -> the emitted logical value for that op's result.
    node_values: FxHashMap<usize, NodeValue>,
    views: Vec<ViewValue>,
    /// Realized view lets, memoized by rendered (base, entries, dims) key.
    view_lets: FxHashMap<String, String>,
    ops_text: String,
    post_checks: String,
    /// (source node, output key) pairs in .output() order.
    outputs: Vec<(usize, usize)>,
    /// Input node ids in creation order (their lets are in ops_text).
    inputs: Vec<usize>,
    view_counter: u32,
    poisoned: Option<String>,
}

impl LogicalRecorder {
    /// First poison reason wins; everything after is a no-op.
    pub fn poison(&mut self, reason: impl Into<String>) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason.into());
        }
    }

    pub fn poisoned(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    fn dim_term(expr: &Expression) -> Result<String, String> {
        let terms = expr.terms.read();
        match &terms[..] {
            [Term::Num(n)] => Ok(format!("(IntLit {n})")),
            // Symbolic dims stay IntVar unconditionally — pins are
            // BINDING-side bounds seeds, never model content.
            [Term::Var(c)] => Ok(format!("(IntVar \"{c}\")")),
            other => Err(format!("arithmetic dim expression {other:?}")),
        }
    }

    fn shape_term(dims: &[Expression]) -> Result<String, String> {
        let mut term = "(IntExprNil)".to_string();
        for dim in dims.iter().rev() {
            term = format!("(IntExprCons {} {term})", Self::dim_term(dim)?);
        }
        Ok(format!("(ShapeLit {term})"))
    }

    fn dtype_term(dtype: DType) -> String {
        format!("({dtype:?})")
    }

    fn entry_term(entry: &MapEntry) -> Result<String, String> {
        Ok(match entry {
            MapEntry::Coord { from_end, extent } => {
                format!("(CoordVar {from_end} {})", Self::dim_term(extent)?)
            }
            MapEntry::Lit(value) => Self::dim_term(value)?,
            MapEntry::Add(a, b) => format!(
                "(IntAdd {} {})",
                Self::entry_term(a)?,
                Self::entry_term(b)?
            ),
            MapEntry::Mul(a, e) => {
                format!("(IntMul {} {})", Self::entry_term(a)?, Self::dim_term(e)?)
            }
            MapEntry::Div(a, e) => format!(
                "(IntTruncDiv {} {})",
                Self::entry_term(a)?,
                Self::dim_term(e)?
            ),
            MapEntry::Rem(a, e) => format!(
                "(IntTruncRem {} {})",
                Self::entry_term(a)?,
                Self::dim_term(e)?
            ),
            MapEntry::Min(a, b) => format!(
                "(IntMin {} {})",
                Self::entry_term(a)?,
                Self::entry_term(b)?
            ),
            MapEntry::Max(a, b) => format!(
                "(IntMax {} {})",
                Self::entry_term(a)?,
                Self::entry_term(b)?
            ),
        })
    }

    /// Record an input declaration. Name matches the interim translator's
    /// convention exactly ("{label}_{idx}") so certification programs
    /// unify the input classes by constructor identity.
    pub fn input(&mut self, node: usize, label: &str, dims: &[Expression], dtype: DType) {
        let shape = match Self::shape_term(dims) {
            Ok(shape) => shape,
            Err(reason) => return self.poison(format!("input t{node}: {reason}")),
        };
        let name = format!("rec_t{node}");
        self.ops_text.push_str(&format!(
            "(let {name} (LogicalTensorInputLit (LogicalIdLit \"{label}_{node}\") {shape} {}))\n",
            Self::dtype_term(dtype),
        ));
        self.node_values.insert(
            node,
            NodeValue {
                name,
                dims: dims.to_vec(),
                dtype,
            },
        );
        self.inputs.push(node);
    }

    /// Resolve an operand tensor to its logical let name, realizing its
    /// view if it carries one. Cross-checks tracker dims against recorded
    /// dims — the tripwire for tracker mutations the recorder never saw.
    fn resolve(
        &mut self,
        node: usize,
        view: Option<ViewId>,
        tracker_dims: &[Expression],
    ) -> Result<String, String> {
        let (name, dims) = match view {
            None => {
                let value = self
                    .node_values
                    .get(&node)
                    .ok_or_else(|| format!("t{node} has no recorded logical value"))?;
                (value.name.clone(), value.dims.clone())
            }
            Some(ViewId(v)) => {
                let view = self.views[v as usize].clone();
                (self.realize_view(&view)?, view.dims)
            }
        };
        if dims.len() != tracker_dims.len()
            || dims
                .iter()
                .zip(tracker_dims)
                .any(|(a, b)| a.to_usize() != b.to_usize() || a.to_usize().is_none() && a != b)
        {
            return Err(format!(
                "t{node}: tracker dims {tracker_dims:?} diverged from recorded dims {dims:?} \
                 (a tracker was mutated outside the movement API)"
            ));
        }
        Ok(name)
    }

    fn realize_view(&mut self, view: &ViewValue) -> Result<String, String> {
        let base = self
            .node_values
            .get(&view.base_node)
            .ok_or_else(|| format!("view base t{} has no recorded value", view.base_node))?
            .name
            .clone();
        let mut entries_term = "(IntExprNil)".to_string();
        for entry in view.entries.iter().rev() {
            let entry_term = Self::entry_term(entry)?;
            entries_term = format!("(IntExprCons {entry_term} {entries_term})");
        }
        let shape = Self::shape_term(&view.dims)?;
        let key = format!("{base}|{entries_term}|{shape}");
        if let Some(existing) = self.view_lets.get(&key) {
            return Ok(existing.clone());
        }
        let name = format!("rec_view{}", self.view_counter);
        self.view_counter += 1;
        self.ops_text.push_str(&format!(
            "(let {name} (LogicalIndexMapApply {base} (IndexMapLit {entries_term}) {shape}))\n"
        ));
        self.view_lets.insert(key, name.clone());
        Ok(name)
    }

    /// Identity entries for an unviewed tensor of these dims: parent axis
    /// p (outermost-first) reads the like-positioned output coordinate.
    fn identity_entries(dims: &[Expression]) -> Vec<MapEntry> {
        let rank = dims.len();
        (0..rank)
            .map(|p| MapEntry::Coord {
                from_end: rank - 1 - p,
                extent: dims[p].clone(),
            })
            .collect()
    }

    /// Apply a movement to a tensor's current view (or to its identity),
    /// producing the composed view. Composition substitutes the movement's
    /// coordinate replacement into our own entry terms — syntax we just
    /// emitted, never reconstruction.
    pub fn apply_movement(
        &mut self,
        node: usize,
        current: Option<ViewId>,
        current_dims: &[Expression],
        movement: Movement,
    ) -> Option<ViewId> {
        if self.poisoned.is_some() {
            return None;
        }
        let (base_node, entries, prev_dims) = match current {
            Some(ViewId(v)) => {
                let view = &self.views[v as usize];
                (view.base_node, view.entries.clone(), view.dims.clone())
            }
            None => {
                let Some(value) = self.node_values.get(&node) else {
                    self.poison(format!("movement on t{node} with no recorded value"));
                    return None;
                };
                (node, Self::identity_entries(&value.dims), value.dims.clone())
            }
        };
        if prev_dims.len() != current_dims.len() {
            self.poison(format!(
                "movement on t{node}: tracker rank {} diverged from recorded rank {}",
                current_dims.len(),
                prev_dims.len()
            ));
            return None;
        }
        let prev_rank = prev_dims.len();

        // For each previous-out FRONT position p: its replacement in the
        // new out space (Coord from-end in the new space, or a literal).
        let (replacement, new_dims): (Vec<MapEntry>, Vec<Expression>) = match movement {
            Movement::Permute(axes) => {
                if axes.len() != prev_rank {
                    self.poison(format!("permute arity {} vs rank {prev_rank}", axes.len()));
                    return None;
                }
                let mut replacement = vec![MapEntry::Lit(0.into()); prev_rank];
                for (q, &p) in axes.iter().enumerate() {
                    replacement[p] = MapEntry::Coord {
                        from_end: prev_rank - 1 - q,
                        extent: prev_dims[p].clone(),
                    };
                }
                let new_dims = axes.iter().map(|&p| prev_dims[p].clone()).collect();
                (replacement, new_dims)
            }
            Movement::ExpandDim { axis, size } => {
                if axis > prev_rank {
                    self.poison(format!("expand_dim axis {axis} vs rank {prev_rank}"));
                    return None;
                }
                let new_rank = prev_rank + 1;
                let replacement = (0..prev_rank)
                    .map(|p| {
                        let q = if p < axis { p } else { p + 1 };
                        MapEntry::Coord {
                            from_end: new_rank - 1 - q,
                            extent: prev_dims[p].clone(),
                        }
                    })
                    .collect();
                let mut new_dims = prev_dims.clone();
                new_dims.insert(axis, size);
                (replacement, new_dims)
            }
            Movement::RemoveDim { axis } => {
                if axis >= prev_rank || prev_dims[axis].to_usize() != Some(1) {
                    self.poison(format!(
                        "remove_dim axis {axis} of dims {prev_dims:?} (must be a size-1 axis)"
                    ));
                    return None;
                }
                let new_rank = prev_rank - 1;
                let replacement = (0..prev_rank)
                    .map(|p| {
                        if p == axis {
                            MapEntry::Lit(0.into())
                        } else {
                            let q = if p < axis { p } else { p - 1 };
                            MapEntry::Coord {
                                from_end: new_rank - 1 - q,
                                extent: prev_dims[p].clone(),
                            }
                        }
                    })
                    .collect();
                let mut new_dims = prev_dims.clone();
                new_dims.remove(axis);
                (replacement, new_dims)
            }
            Movement::SplitDims { axis, inner } => {
                if axis >= prev_rank {
                    self.poison(format!("split_dims axis {axis} vs rank {prev_rank}"));
                    return None;
                }
                let outer = (prev_dims[axis] / inner.clone()).simplify();
                let new_rank = prev_rank + 1;
                let replacement = (0..prev_rank)
                    .map(|p| {
                        if p == axis {
                            MapEntry::Add(
                                Box::new(MapEntry::Mul(
                                    Box::new(MapEntry::Coord {
                                        from_end: new_rank - 1 - axis,
                                        extent: outer.clone(),
                                    }),
                                    inner.clone(),
                                )),
                                Box::new(MapEntry::Coord {
                                    from_end: new_rank - 1 - (axis + 1),
                                    extent: inner.clone(),
                                }),
                            )
                        } else {
                            let q = if p < axis { p } else { p + 1 };
                            MapEntry::Coord {
                                from_end: new_rank - 1 - q,
                                extent: prev_dims[p].clone(),
                            }
                        }
                    })
                    .collect();
                let mut new_dims = prev_dims.clone();
                new_dims[axis] = outer;
                new_dims.insert(axis + 1, inner);
                (replacement, new_dims)
            }
            Movement::MergeDims { axis1, axis2 } => {
                if axis1 >= axis2 || axis2 >= prev_rank {
                    self.poison(format!("merge_dims ({axis1},{axis2}) vs rank {prev_rank}"));
                    return None;
                }
                let inner = prev_dims[axis2].clone();
                let merged = (prev_dims[axis1] * prev_dims[axis2]).simplify();
                let new_rank = prev_rank - 1;
                let merged_coord = MapEntry::Coord {
                    from_end: new_rank - 1 - axis1,
                    extent: merged.clone(),
                };
                let replacement = (0..prev_rank)
                    .map(|p| {
                        if p == axis1 {
                            MapEntry::Div(Box::new(merged_coord.clone()), inner.clone())
                        } else if p == axis2 {
                            MapEntry::Rem(Box::new(merged_coord.clone()), inner.clone())
                        } else {
                            let q = if p < axis2 { p } else { p - 1 };
                            MapEntry::Coord {
                                from_end: new_rank - 1 - q,
                                extent: prev_dims[p].clone(),
                            }
                        }
                    })
                    .collect();
                let mut new_dims = prev_dims.clone();
                new_dims[axis1] = merged;
                new_dims.remove(axis2);
                (replacement, new_dims)
            }
            Movement::Repeat(repeats) => {
                if repeats.len() != prev_rank {
                    self.poison(format!("repeat arity {} vs rank {prev_rank}", repeats.len()));
                    return None;
                }
                let replacement = (0..prev_rank)
                    .map(|p| {
                        if repeats[p].to_usize() == Some(1) {
                            MapEntry::Coord {
                                from_end: prev_rank - 1 - p,
                                extent: prev_dims[p].clone(),
                            }
                        } else {
                            let tiled = (prev_dims[p] * repeats[p]).simplify();
                            MapEntry::Rem(
                                Box::new(MapEntry::Coord {
                                    from_end: prev_rank - 1 - p,
                                    extent: tiled,
                                }),
                                prev_dims[p].clone(),
                            )
                        }
                    })
                    .collect();
                let new_dims = prev_dims
                    .iter()
                    .zip(&repeats)
                    .map(|(d, r)| (*d * *r).simplify())
                    .collect();
                (replacement, new_dims)
            }
            Movement::Shrink { new_dims } => {
                if new_dims.len() != prev_rank {
                    self.poison(format!("shrink arity {} vs rank {prev_rank}", new_dims.len()));
                    return None;
                }
                let replacement = (0..prev_rank)
                    .map(|p| MapEntry::Coord {
                        from_end: prev_rank - 1 - p,
                        extent: new_dims[p].clone(),
                    })
                    .collect();
                (replacement, new_dims)
            }
        };

        // Substitute: every Coord leaf in the base entries reads a
        // previous-out coordinate, replaced by the movement's replacement.
        let composed = entries
            .iter()
            .map(|entry| entry.substitute(&replacement, prev_rank))
            .collect();
        let id = ViewId(self.views.len() as u32);
        self.views.push(ViewValue {
            base_node,
            entries: composed,
            dims: new_dims,
        });
        Some(id)
    }

    /// Record an op with the given constructor and operand tensors.
    /// `extra` is appended after the operands (e.g. a reduce axis or a
    /// dtype term). Returns nothing — failures poison.
    #[allow(clippy::too_many_arguments)]
    pub fn op(
        &mut self,
        node: usize,
        constructor: &str,
        operands: &[(usize, Option<ViewId>, Vec<Expression>)],
        extra: &str,
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) {
        if self.poisoned.is_some() {
            return;
        }
        let mut names = Vec::with_capacity(operands.len());
        for (op_node, view, tracker_dims) in operands {
            match self.resolve(*op_node, *view, tracker_dims) {
                Ok(name) => names.push(name),
                Err(reason) => return self.poison(format!("{constructor} at t{node}: {reason}")),
            }
        }
        let name = format!("rec_t{node}");
        let extra = if extra.is_empty() {
            String::new()
        } else {
            format!(" {extra}")
        };
        self.ops_text.push_str(&format!(
            "(let {name} ({constructor} {}{extra}))\n",
            names.join(" ")
        ));
        self.node_values.insert(
            node,
            NodeValue {
                name,
                dims: out_dims,
                dtype: out_dtype,
            },
        );
    }

    /// Record a seam-node view op: the node's logical value is an
    /// IndexMapApply of its operand through entries built DIRECTLY from
    /// the seam node's own parameters (SliceView starts, UnfoldView
    /// window arithmetic, ClampView clamps) — the source of truth.
    pub fn view_op(
        &mut self,
        node: usize,
        operand: (usize, Option<ViewId>, Vec<Expression>),
        entries: &[MapEntry],
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) {
        if self.poisoned.is_some() {
            return;
        }
        let (op_node, view, tracker_dims) = operand;
        let operand_name = match self.resolve(op_node, view, &tracker_dims) {
            Ok(name) => name,
            Err(reason) => return self.poison(format!("view op at t{node}: {reason}")),
        };
        let mut entries_term = "(IntExprNil)".to_string();
        for entry in entries.iter().rev() {
            match Self::entry_term(entry) {
                Ok(term) => entries_term = format!("(IntExprCons {term} {entries_term})"),
                Err(reason) => return self.poison(format!("view op at t{node}: {reason}")),
            }
        }
        let shape = match Self::shape_term(&out_dims) {
            Ok(shape) => shape,
            Err(reason) => return self.poison(format!("view op at t{node}: {reason}")),
        };
        let name = format!("rec_t{node}");
        self.ops_text.push_str(&format!(
            "(let {name} (LogicalIndexMapApply {operand_name} (IndexMapLit {entries_term}) {shape}))\n"
        ));
        self.node_values.insert(
            node,
            NodeValue {
                name,
                dims: out_dims,
                dtype: out_dtype,
            },
        );
    }

    /// Record a source op (no operands) with pre-rendered argument text.
    pub fn source_op(
        &mut self,
        node: usize,
        constructor: &str,
        args: &str,
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) {
        if self.poisoned.is_some() {
            return;
        }
        let name = format!("rec_t{node}");
        self.ops_text
            .push_str(&format!("(let {name} ({constructor} {args}))\n"));
        self.node_values.insert(
            node,
            NodeValue {
                name,
                dims: out_dims,
                dtype: out_dtype,
            },
        );
    }

    /// Record a pad-mask indicator iota (the MaskIota seam node): per
    /// padded axis, (before <= p) · (p < before + dim), each indicator a
    /// bool-bridge cast, >= spelled as the discrete rotation
    /// before < p + 1 — the translator's exact forms, factor order and
    /// fold associativity included (certification depends on it).
    pub fn record_mask_iota(
        &mut self,
        node: usize,
        befores: &[Expression],
        afters: &[Expression],
        in_dims: &[Expression],
    ) {
        if self.poisoned.is_some() {
            return;
        }
        let rank = in_dims.len();
        let mut out_dims = Vec::with_capacity(rank);
        let mut out_terms = Vec::with_capacity(rank);
        for k in 0..rank {
            let out_dim = (befores[k] + in_dims[k] + afters[k]).simplify();
            match Self::dim_term(&out_dim) {
                Ok(term) => out_terms.push(term),
                Err(reason) => return self.poison(format!("mask iota at t{node}: {reason}")),
            }
            out_dims.push(out_dim);
        }
        let mut factors: Vec<String> = Vec::new();
        for k in 0..rank {
            let coord = format!("(CoordVar {} {})", rank - 1 - k, out_terms[k]);
            let before = befores[k];
            let after = afters[k];
            let (Ok(before_term), Ok(bound_term)) = (
                Self::dim_term(&before),
                Self::dim_term(&(before + in_dims[k]).simplify()),
            ) else {
                return self.poison(format!("mask iota at t{node}: symbolic pad bound"));
            };
            if before != Expression::from(0) {
                factors.push(format!(
                    "(IntCastFromBool (BoolLessThanInt {before_term} (IntAdd {coord} (IntLit 1))))"
                ));
            }
            if after != Expression::from(0) {
                factors.push(format!(
                    "(IntCastFromBool (BoolLessThanInt {coord} {bound_term}))"
                ));
            }
        }
        let mut expr = factors.pop().unwrap_or_else(|| "(IntLit 1)".to_string());
        for factor in factors {
            expr = format!("(IntMul {factor} {expr})");
        }
        let shape = match Self::shape_term(&out_dims) {
            Ok(shape) => shape,
            Err(reason) => return self.poison(format!("mask iota at t{node}: {reason}")),
        };
        self.source_op(node, "LogicalIota", &format!("{expr} {shape}"), out_dims, DType::Int);
    }

    /// Append post-schedule authoring checks (iota bounds pairs).
    pub fn post_check(&mut self, text: &str) {
        self.post_checks.push_str(text);
    }

    /// Record an output designation for a source node (Phase A: the source
    /// must be an unviewed recorded value; view outputs poison until the
    /// native materialization story is certified).
    pub fn output(
        &mut self,
        source_node: usize,
        view: Option<ViewId>,
        tracker_dims: &[Expression],
        key: usize,
    ) {
        if self.poisoned.is_some() {
            return;
        }
        if view.is_some() {
            return self.poison(format!(
                "output of a viewed tensor t{source_node} (native view-output pending)"
            ));
        }
        let Some(value) = self.node_values.get(&source_node) else {
            return self.poison(format!("output of unrecorded t{source_node}"));
        };
        if value.dims.len() != tracker_dims.len()
            || value
                .dims
                .iter()
                .zip(tracker_dims)
                .any(|(a, b)| a.to_usize() != b.to_usize() || a.to_usize().is_none() && a != b)
        {
            return self.poison(format!(
                "output t{source_node}: tracker dims {tracker_dims:?} diverged from recorded                  dims {:?} (a tracker was mutated outside the movement API)",
                value.dims
            ));
        }
        self.outputs.push((source_node, key));
    }

    /// MODEL text for certification: input declarations + ops only — no
    /// output unions (a union would FORCE the equality certification must
    /// CHECK), no signature lists, no boundaries.
    pub fn certification_text(&self) -> Result<&str, String> {
        match &self.poisoned {
            Some(reason) => Err(format!("recorder poisoned: {reason}")),
            None => Ok(&self.ops_text),
        }
    }

    /// The recorded logical name for a node (certification checks).
    pub fn value_name(&self, node: usize) -> Option<&str> {
        self.node_values.get(&node).map(|value| value.name.as_str())
    }

    /// Full MODEL text (M3 Step 1): declarations + ops + output naming +
    /// signature lists + post-schedule authoring checks. Boundary text is
    /// the binding generator's business.
    pub fn model_text(&self) -> Result<String, String> {
        if let Some(reason) = &self.poisoned {
            return Err(format!("recorder poisoned: {reason}"));
        }
        let mut text = self.ops_text.clone();
        // Models have no signature (ruling 2026-07-30): naming is the
        // whole story — bindings attach buffers by name.
        for (source, key) in &self.outputs {
            let name = &self.node_values[source].name;
            text.push_str(&format!(
                "(union {name} (LogicalTensorNamed (LogicalIdLit \"out_{key}\")))\n"
            ));
        }
        Ok(text)
    }

    /// The post-schedule authoring checks (appended after the schedule).
    pub fn post_checks(&self) -> &str {
        &self.post_checks
    }

    /// (source node, key) pairs, for the binding generator.
    pub fn output_slots(&self) -> &[(usize, usize)] {
        &self.outputs
    }

    /// Input node ids, for the binding generator.
    pub fn input_nodes(&self) -> &[usize] {
        &self.inputs
    }

    /// M3 Step 1 assembly: the natively-recorded MODEL plus the reference
    /// BINDING (defaults matching the interim translator's boundary text),
    /// as a runnable [`crate::hlir_to_logical::LogicalProgram`]. The
    /// binding vocabulary is the reference runtime's — a different runtime
    /// binds differently against the same model.
    pub fn native_program(&self) -> Result<crate::hlir_to_logical::LogicalProgram, String> {
        let (pre, input_slots, output_slots, post_checks) = self.native_parts()?;
        Ok(crate::hlir_to_logical::LogicalProgram {
            text: format!("{pre}{}{post_checks}", crate::reference_binding::SCHEDULE),
            input_slots,
            output_slots,
        })
    }

    /// The native assembly SPLIT at the schedule, so a runtime can inject
    /// binding seeds (dynamic-dim bounds) before saturation: returns
    /// (pre-schedule text, input slots, output slots, post-schedule checks).
    #[allow(clippy::type_complexity)]
    pub fn native_parts(
        &self,
    ) -> Result<
        (
            String,
            Vec<(petgraph::graph::NodeIndex, u64)>,
            Vec<(usize, u64)>,
            String,
        ),
        String,
    > {
        let mut text = self.model_text()?;
        let mut input_slots = Vec::new();
        let mut input_buffer_tensors = Vec::new();
        for &node in &self.inputs {
            let value = &self.node_values[&node];
            let shape = Self::shape_term(&value.dims)?;
            let stem = format!("nat{node}");
            text.push_str(&crate::reference_binding::input_binding(
                &stem,
                node,
                &value.name,
                &shape,
                &crate::reference_binding::width_term(value.dtype),
            ));
            input_buffer_tensors.push(format!("{stem}_buffer_tensor"));
            input_slots.push((petgraph::graph::NodeIndex::new(node), node as u64));
        }
        let mut output_slots = Vec::new();
        let mut output_buffer_tensors = Vec::new();
        for &(source, key) in &self.outputs {
            let value = &self.node_values[&source];
            let shape = Self::shape_term(&value.dims)?;
            let stem = format!("natout{key}");
            text.push_str(&crate::reference_binding::output_binding(
                &stem,
                key,
                &value.name,
                &shape,
                value.dtype,
            ));
            output_buffer_tensors.push(format!("{stem}_buffer_tensor"));
            output_slots.push((key, key as u64));
        }
        text.push_str(&crate::reference_binding::boundary_lists(
            &input_buffer_tensors,
            &output_buffer_tensors,
            "nat_input_boundary",
            "nat_output_boundary",
        ));
        Ok((text, input_slots, output_slots, self.post_checks.clone()))
    }
}
