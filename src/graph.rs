//! The graph a model is authored into: dynamic-dim assumptions plus the
//! LOGICAL structure the recorder captures (M3 Step 4b: their HLIR
//! petgraph, e-graph search spaces, and compile ladder are DELETED — the
//! recorder is the only path to the e-graph, and runtimes own
//! load/bind/with_ops/search).

use petgraph::stable_graph::NodeIndex;
use rustc_hash::FxHashMap;

use crate::dtype::DType;
use crate::frontend::GraphTensor;
use crate::shape::ToShape;

/// A bucket for a dynamic dimension, defining a range of valid values.
/// For an exact value, use `min == max` (zero-length range).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimBucket {
    pub min: usize,
    pub max: usize,
    representative_override: Option<usize>,
}

impl DimBucket {
    /// Create a new bucket covering `[min, max]` inclusive.
    /// For an exact value, pass `min == max`.
    pub fn new(min: usize, max: usize) -> Self {
        assert!(min <= max, "DimBucket min ({min}) must be <= max ({max})");
        DimBucket {
            min,
            max,
            representative_override: None,
        }
    }

    /// Override the representative value used during search profiling.
    /// Must be within `[min, max]`.
    pub fn representative(mut self, val: usize) -> Self {
        assert!(
            val >= self.min && val <= self.max,
            "Representative {val} must be in [{}, {}]",
            self.min,
            self.max
        );
        self.representative_override = Some(val);
        self
    }

    /// The representative value used during search profiling.
    /// Defaults to midpoint `(min + max) / 2`.
    pub fn representative_value(&self) -> usize {
        self.representative_override
            .unwrap_or((self.min + self.max) / 2)
    }

    /// Check if `val` falls within this bucket's range.
    pub fn contains(&self, val: usize) -> bool {
        val >= self.min && val <= self.max
    }
}

#[derive(Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// The logical-model recorder — GraphTensor methods emit their
    /// logical ops here; it IS the graph (absorbed into this struct at
    /// M3 Step 4e).
    pub logical: crate::graph::LogicalGraph,
    /// The tensor-id mint. Ids are plain sequence numbers (the NodeIndex
    /// type is kept as the id vocabulary; the petgraph it once indexed is
    /// gone) — they key recorder rows, binding slots, and set_data.
    next_id: u32,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Mint a fresh tensor id.
    pub(crate) fn mint_id(&mut self) -> NodeIndex {
        let id = NodeIndex::new(self.next_id as usize);
        self.next_id += 1;
        id
    }

    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor_dtyped("", shape, DType::default())
    }

    /// Create a new tensor with shape S and this dtype. Dtype is DECLARED
    /// at creation (purity ruling 2026-07-30: as_dtype is gone — a
    /// different dtype downstream is a logical cast, never a mutation of
    /// the declaration).
    pub fn tensor_dtyped(&mut self, shape: impl ToShape, dtype: DType) -> GraphTensor {
        self.named_tensor_dtyped("", shape, dtype)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: impl ToString, shape: impl ToShape) -> GraphTensor {
        self.named_tensor_dtyped(name, shape, DType::default())
    }

    /// Named + dtyped input declaration — the one true constructor.
    pub fn named_tensor_dtyped(
        &mut self,
        name: impl ToString,
        shape: impl ToShape,
        dtype: DType,
    ) -> GraphTensor {
        let name = name.to_string();
        let id = self.mint_id();
        let tensor = GraphTensor {
            id,
            graph_ref: self,
            dims: shape.to_shape().into_iter().collect(),
            dtype,
            logical_value: None,
        };
        let logical = self
            .logical
            .input(id.index(), &name, &tensor.dims(), tensor.dtype);
        tensor.with_logical(logical)
    }
}

// ---------------------------------------------------------------------
// THE LOGICAL GRAPH (M3 Step 4e: absorbed into this module — the
// recorder IS the graph; LogicalGraph keeps its name, Austin's ruling
// 2026-08-01). Formerly src/logical_graph.rs:
// ---------------------------------------------------------------------
// The LOGICAL GRAPH — the model the frontend actually builds (renamed
// from logical_recorder, ruling 2026-07-31: this IS the durable thing;
// absorbed into this module at Step 4e; the HLIR StableGraph it once
// stood beside is deleted).
//
// GraphTensor methods emit their LOGICAL ops here as the graph is built,
// beside their HLIR emission — the two worlds coexist until the interim
// translator retires family by family (each retirement gated by the
// certification test asserting recorder ≡ translator at the e-class
// level). The deep point (ruling 2026-07-30): movement methods emit
// `IndexMapApply` views DIRECTLY from their own parameters, at the source
// of truth — replacing tracker-lift reconstruction entirely.
//
// Model/binding split (M3 Step 1): the recorder emits MODEL text only —
// input declarations, ops, output naming, signature lists. Boundary
// vocabulary (layouts, buffers, access, freed-by, Bool8 casts) is the
// runtime binding generator's business (`reference_binding`), never the
// model's.
//
// Coverage is honest: any construct the recorder does not understand
// POISONS it with a reason — the first reason wins, the native path
// refuses loudly at load, and their pipeline is untouched. Every operand
// resolution cross-checks the tensor's tracker dims against the recorded
// dims, so a direct tracker mutation the recorder never saw (the ways a
// frontend method can bypass the movement API) poisons instead of
// silently mistranslating.


use crate::shape::{Expression, Term};
use anyhow::{bail, Result as AnyResult};

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

/// A logical VALUE id — the SSA identity every tensor handle carries
/// (M3 Step 4a: LogicalGraph mints its own ids; HLIR node indices remain
/// only as transitional binding-slot keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueId(pub u32);

/// An operand as a record call sees it: the handle's value plus its
/// tracker dims (the divergence tripwire input).
pub type Operand = (Option<ValueId>, Vec<Expression>);

/// How a value renders: SSA rows are vocabulary-thin (constructor string
/// + operand ids + aux text), but two constructors wrap some operands in
/// a LogicalTensorList.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderForm {
    /// ({constructor} {operands...} {aux})
    Plain,
    /// (LogicalGather data (Cons c1 (Cons c2 ...)))
    GatherList,
    /// (LogicalScatter init (Cons ...) src) — src is the LAST operand.
    ScatterList,
}

/// One SSA value: an op applied to operand values. Views are ORDINARY
/// values (constructor = LogicalIndexMapApply) that additionally keep
/// their map entries STRUCTURED so movement composition works on trees,
/// never on text.
#[derive(Debug, Clone)]
struct Value {
    constructor: String,
    operands: Vec<ValueId>,
    /// Rendered trailing arguments (axis numbers, dtype terms, iota
    /// expr+shape, map+shape for views...).
    aux: String,
    form: RenderForm,
    /// Structured map entries — present exactly on movement-composed
    /// views, consumed by apply_movement composition.
    entries: Option<Vec<MapEntry>>,
    dims: Vec<Expression>,
    #[allow(dead_code)]
    dtype: DType,
    /// For inputs: the transitional binding-slot key (HLIR node index —
    /// their set_data keying; dies with the HLIR pipeline).
    input_slot: Option<usize>,
    /// The input declaration label ("{label}_{slot}").
    input_label: Option<String>,
}

#[derive(Debug, Default)]
pub struct LogicalGraph {
    values: Vec<Value>,
    /// (value, output key) pairs in .output() order.
    outputs: Vec<(ValueId, usize)>,
    post_checks: String,
    poisoned: Option<String>,
}

impl LogicalGraph {
    /// First poison reason wins; everything after is a no-op.
    pub fn poison(&mut self, reason: impl Into<String>) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason.into());
        }
    }

    pub fn poisoned(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    /// Read-only rows for visualization, one per value in ValueId order:
    /// (constructor, operand ids, dims, dtype, input label).
    #[allow(clippy::type_complexity)]
    pub(crate) fn viz_rows(
        &self,
    ) -> impl Iterator<Item = (&str, &[ValueId], &[Expression], DType, Option<&str>)> {
        self.values.iter().map(|value| {
            (
                value.constructor.as_str(),
                value.operands.as_slice(),
                value.dims.as_slice(),
                value.dtype,
                value.input_label.as_deref(),
            )
        })
    }

    /// The recorded output designations, in .output() order.
    pub(crate) fn viz_outputs(&self) -> &[(ValueId, usize)] {
        &self.outputs
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

    /// `owner_shape` is the consuming view's OUT shape term — the box the
    /// map's coordinates are formals of (scoped CoordVar: ownership rides
    /// in the term; the extent field is gone, extents are the owner's own
    /// dims).
    fn entry_term(entry: &MapEntry, owner_shape: &str) -> Result<String, String> {
        Ok(match entry {
            MapEntry::Coord { from_end, extent: _ } => {
                format!("(CoordVar {owner_shape} {from_end})")
            }
            MapEntry::Lit(value) => Self::dim_term(value)?,
            MapEntry::Add(a, b) => format!(
                "(IntAdd {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::entry_term(b, owner_shape)?
            ),
            MapEntry::Mul(a, e) => {
                format!("(IntMul {} {})", Self::entry_term(a, owner_shape)?, Self::dim_term(e)?)
            }
            MapEntry::Div(a, e) => format!(
                "(IntTruncDiv {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::dim_term(e)?
            ),
            MapEntry::Rem(a, e) => format!(
                "(IntTruncRem {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::dim_term(e)?
            ),
            MapEntry::Min(a, b) => format!(
                "(IntMin {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::entry_term(b, owner_shape)?
            ),
            MapEntry::Max(a, b) => format!(
                "(IntMax {} {})",
                Self::entry_term(a, owner_shape)?,
                Self::entry_term(b, owner_shape)?
            ),
        })
    }

    fn push(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(value);
        id
    }

    /// Resolve an operand: it must carry a value, and its tracker dims
    /// must agree with the recorded dims — the tripwire for tracker
    /// mutations the graph never saw.
    fn resolve(&mut self, operand: &Operand, at: &str) -> Result<ValueId, String> {
        let (value, tracker_dims) = operand;
        let Some(id) = value else {
            return Err(format!("{at}: operand has no recorded logical value"));
        };
        let dims = &self.values[id.0 as usize].dims;
        if dims.len() != tracker_dims.len()
            || dims
                .iter()
                .zip(tracker_dims)
                .any(|(a, b)| a.to_usize() != b.to_usize() || a.to_usize().is_none() && a != b)
        {
            return Err(format!(
                "{at}: tracker dims {tracker_dims:?} diverged from recorded dims {dims:?} \
                 (a tracker was mutated outside the movement API)"
            ));
        }
        Ok(*id)
    }

    /// Record an input declaration. Label keys keep the transitional
    /// HLIR-node-index convention (their set_data keying).
    pub fn input(
        &mut self,
        slot: usize,
        label: &str,
        dims: &[Expression],
        dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let shape = match Self::shape_term(dims) {
            Ok(shape) => shape,
            Err(reason) => {
                self.poison(format!("input t{slot}: {reason}"));
                return None;
            }
        };
        let full_label = format!("{label}_{slot}");
        // Boolean inputs cross the boundary as Bool8 (the Bool8 ruling:
        // models compute in 1-bit Bool; bindings state the byte
        // representation). Outputs get their boundary cast from the
        // binding generator; INPUTS get it here — the wire tensor is
        // declared Bool8 and a LogicalCast hands the model its Bool
        // value, so the boundary buffer's dtype-of row (Bool8) agrees
        // with its 8-bit layout (typed-buffers landing B, 2026-08-11).
        let wire_dtype_term = match dtype {
            DType::Bool => "(Bool8)".to_string(),
            other => Self::dtype_term(other),
        };
        let aux = format!("(LogicalIdLit \"{full_label}\") {shape} {wire_dtype_term}");
        let lit = self.push(Value {
            constructor: "LogicalTensorInputLit".to_string(),
            operands: Vec::new(),
            aux,
            form: RenderForm::Plain,
            entries: None,
            dims: dims.to_vec(),
            dtype,
            input_slot: Some(slot),
            input_label: Some(full_label),
        });
        if dtype == DType::Bool {
            return self.op(
                slot,
                "LogicalCast",
                &[(Some(lit), dims.to_vec())],
                "(Bool)",
                dims.to_vec(),
                DType::Bool,
            );
        }
        Some(lit)
    }

    /// Record an op over operand values.
    pub fn op(
        &mut self,
        at: usize,
        constructor: &str,
        operands: &[Operand],
        extra: &str,
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let mut ids = Vec::with_capacity(operands.len());
        for operand in operands {
            match self.resolve(operand, &format!("{constructor} at t{at}")) {
                Ok(id) => ids.push(id),
                Err(reason) => {
                    self.poison(reason);
                    return None;
                }
            }
        }
        Some(self.push(Value {
            constructor: constructor.to_string(),
            operands: ids,
            aux: extra.to_string(),
            form: RenderForm::Plain,
            entries: None,
            dims: out_dims,
            dtype: out_dtype,
            input_slot: None,
            input_label: None,
        }))
    }

    /// Record a seam-node view: an IndexMapApply of the operand through
    /// entries built from the seam's own parameters.
    pub fn view_op(
        &mut self,
        at: usize,
        operand: &Operand,
        entries: &[MapEntry],
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let base = match self.resolve(operand, &format!("view op at t{at}")) {
            Ok(id) => id,
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        };
        self.push_view(at, base, entries.to_vec(), out_dims, out_dtype)
    }

    fn push_view(
        &mut self,
        at: usize,
        base: ValueId,
        entries: Vec<MapEntry>,
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        let shape = match Self::shape_term(&out_dims) {
            Ok(shape) => shape,
            Err(reason) => {
                self.poison(format!("view at t{at}: {reason}"));
                return None;
            }
        };
        // The map's DOMAIN TAG (ruling 2026-08-11): an IndexMapLit
        // carries the source shape it substitutes into — the parent's
        // own dims, written here at the single mint site so the
        // apply/map coherence tripwire can never fire on recorder output.
        let source_dims = self.values[base.0 as usize].dims.clone();
        let source_shape = match Self::shape_term(&source_dims) {
            Ok(term) => term,
            Err(reason) => {
                self.poison(format!("view at t{at}: {reason}"));
                return None;
            }
        };
        let mut entries_term = "(IntExprNil)".to_string();
        for entry in entries.iter().rev() {
            match Self::entry_term(entry, &shape) {
                Ok(term) => entries_term = format!("(IntExprCons {term} {entries_term})"),
                Err(reason) => {
                    self.poison(format!("view at t{at}: {reason}"));
                    return None;
                }
            }
        }
        Some(self.push(Value {
            constructor: "LogicalIndexMapApply".to_string(),
            operands: vec![base],
            aux: format!("(IndexMapLit {entries_term} {source_shape}) {shape}"),
            form: RenderForm::Plain,
            entries: Some(entries),
            dims: out_dims,
            dtype: out_dtype,
            input_slot: None,
            input_label: None,
        }))
    }

    /// Record a source op (no operands) with pre-rendered argument text.
    pub fn source_op(
        &mut self,
        _at: usize,
        constructor: &str,
        args: &str,
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        Some(self.push(Value {
            constructor: constructor.to_string(),
            operands: Vec::new(),
            aux: args.to_string(),
            form: RenderForm::Plain,
            entries: None,
            dims: out_dims,
            dtype: out_dtype,
            input_slot: None,
            input_label: None,
        }))
    }

    /// Record a pad-mask indicator iota (see the pad seam): per padded
    /// axis, (before <= p) · (p < before + dim) as bool-bridge casts.
    /// Record a LogicalIota: the value expression is authored over the
    /// FLAT index (the frontend's `'z'`) and rewritten here into a true
    /// COORDINATE FUNCTION over the declared shape —
    /// `z := Σ CoordVar(shape, axis) · row-major-stride` — so the
    /// recorded model is per-coordinate at ANY rank (Design A, ruling
    /// 2026-08-06: the rank-1-plus-recorded-splits detour and its silent
    /// symbolic-total collapse are gone; symbolic dims render as IntVar
    /// through `shape_term`). Extent-1 axes contribute no summand (their
    /// coordinate is identically 0); a fully degenerate shape records
    /// `(IntLit 0)`. The authoring-contract bounds check pair rides
    /// every iota.
    /// Record a LogicalIota from a COORDINATE-FUNCTION expression (P1
    /// ruling 2026-08-07): the value expression is authored over
    /// `Term::Coord(k)` atoms — one per output axis, minted by
    /// `Graph::iota`'s closure — and lowers per-axis: coords become
    /// `(CoordVar shape axis)`, named symbols become `(IntVar "c")`.
    /// There is no flat-'z' form anymore ('z' in an iota expression is an
    /// ordinary named symbol; flat-index authoring is a rank-1 iota plus
    /// recorded reshapes). Extent-1 axes substitute `(IntLit 0)` (their
    /// coordinate is identically zero); a `Coord(k)` with `k >= rank` is
    /// a leaked atom and poisons loudly. The authoring-contract bounds
    /// pair rides every iota.
    pub fn record_iota(
        &mut self,
        at: usize,
        expr: &Expression,
        dims: &[Expression],
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let shape = match Self::shape_term(dims) {
            Ok(term) => term,
            Err(reason) => {
                self.poison(format!("iota at t{at}: {reason}"));
                return None;
            }
        };
        let rank = dims.len();
        let coord_terms: Vec<String> = (0..rank)
            .map(|k| {
                if dims[k] == Expression::from(1) {
                    "(IntLit 0)".to_string()
                } else {
                    format!("(CoordVar {shape} {})", rank - 1 - k)
                }
            })
            .collect();
        let value_expr = match int_expr_term(
            expr,
            &coord_terms,
            &format!("recorder iota t{at}"),
        ) {
            Ok(text) => text,
            Err(err) => {
                self.poison(format!("iota at t{at}: {err}"));
                return None;
            }
        };
        let logical = self.source_op(
            at,
            "LogicalIota",
            &format!("{value_expr} {shape}"),
            dims.to_vec(),
            DType::Int,
        );
        self.post_check(&format!(
            "(check (= ?reclo{at} (lower-bound-of {value_expr})))\n\
             (check (= ?rechi{at} (upper-bound-of {value_expr})))\n"
        ));
        logical
    }

    pub fn record_mask_iota(
        &mut self,
        at: usize,
        befores: &[Expression],
        afters: &[Expression],
        in_dims: &[Expression],
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let rank = in_dims.len();
        let mut out_dims = Vec::with_capacity(rank);
        let mut out_terms = Vec::with_capacity(rank);
        for k in 0..rank {
            let out_dim = (befores[k] + in_dims[k] + afters[k]).simplify();
            match Self::dim_term(&out_dim) {
                Ok(term) => out_terms.push(term),
                Err(reason) => {
                    self.poison(format!("mask iota at t{at}: {reason}"));
                    return None;
                }
            }
            out_dims.push(out_dim);
        }
        let out_shape_term = match Self::shape_term(&out_dims) {
            Ok(term) => term,
            Err(reason) => {
                self.poison(format!("mask iota at t{at}: {reason}"));
                return None;
            }
        };
        let mut factors: Vec<String> = Vec::new();
        for k in 0..rank {
            let coord = format!("(CoordVar {out_shape_term} {})", rank - 1 - k);
            let before = befores[k];
            let after = afters[k];
            let (Ok(before_term), Ok(bound_term)) = (
                Self::dim_term(&before),
                Self::dim_term(&(before + in_dims[k]).simplify()),
            ) else {
                self.poison(format!("mask iota at t{at}: symbolic pad bound"));
                return None;
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
            Err(reason) => {
                self.poison(format!("mask iota at t{at}: {reason}"));
                return None;
            }
        };
        let logical =
            self.source_op(at, "LogicalIota", &format!("{expr} {shape}"), out_dims, DType::Int);
        // The authoring-contract bounds pair — uniform with record_iota
        // (Design A fold-in, 2026-08-06): every recorded iota's value
        // expression must have derivable bounds, or the fixpoint refuses.
        self.post_check(&format!(
            "(check (= ?reclo{at} (lower-bound-of {expr})))\n\
             (check (= ?rechi{at} (upper-bound-of {expr})))\n"
        ));
        logical
    }

    /// Record a coordinate-form gather.
    pub fn record_gather(
        &mut self,
        at: usize,
        data: &Operand,
        coords: &[Operand],
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let mut ids = Vec::with_capacity(coords.len() + 1);
        match self.resolve(data, &format!("gather at t{at}")) {
            Ok(id) => ids.push(id),
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        }
        for coord in coords {
            match self.resolve(coord, &format!("gather at t{at}")) {
                Ok(id) => ids.push(id),
                Err(reason) => {
                    self.poison(reason);
                    return None;
                }
            }
        }
        Some(self.push(Value {
            constructor: "LogicalGather".to_string(),
            operands: ids,
            aux: String::new(),
            form: RenderForm::GatherList,
            entries: None,
            dims: out_dims,
            dtype: out_dtype,
            input_slot: None,
            input_label: None,
        }))
    }

    /// Record a coordinate-form scatter (operands: init, coords..., src).
    pub fn record_scatter(
        &mut self,
        at: usize,
        init: &Operand,
        coords: &[Operand],
        src: &Operand,
        out_dims: Vec<Expression>,
        out_dtype: DType,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let mut ids = Vec::with_capacity(coords.len() + 2);
        match self.resolve(init, &format!("scatter at t{at}")) {
            Ok(id) => ids.push(id),
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        }
        for coord in coords {
            match self.resolve(coord, &format!("scatter at t{at}")) {
                Ok(id) => ids.push(id),
                Err(reason) => {
                    self.poison(reason);
                    return None;
                }
            }
        }
        match self.resolve(src, &format!("scatter at t{at}")) {
            Ok(id) => ids.push(id),
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        }
        Some(self.push(Value {
            constructor: "LogicalScatter".to_string(),
            operands: ids,
            aux: String::new(),
            form: RenderForm::ScatterList,
            entries: None,
            dims: out_dims,
            dtype: out_dtype,
            input_slot: None,
            input_label: None,
        }))
    }

    /// Identity entries: parent axis p reads the like-positioned coord.
    fn identity_entries(dims: &[Expression]) -> Vec<MapEntry> {
        let rank = dims.len();
        (0..rank)
            .map(|p| MapEntry::Coord {
                from_end: rank - 1 - p,
                extent: dims[p].clone(),
            })
            .collect()
    }

    /// Apply a movement: composes onto an existing view value (a new
    /// value over the SAME base — the intermediate view goes dead and is
    /// elided at render) or wraps identity entries over a plain value.
    pub fn apply_movement(
        &mut self,
        at: usize,
        current: &Operand,
        movement: Movement,
    ) -> Option<ValueId> {
        if self.poisoned.is_some() {
            return None;
        }
        let current_id = match self.resolve(current, &format!("movement at t{at}")) {
            Ok(id) => id,
            Err(reason) => {
                self.poison(reason);
                return None;
            }
        };
        let value = &self.values[current_id.0 as usize];
        let (base, entries, prev_dims) = match &value.entries {
            Some(entries) => (
                value.operands[0],
                entries.clone(),
                value.dims.clone(),
            ),
            None => (
                current_id,
                Self::identity_entries(&value.dims),
                value.dims.clone(),
            ),
        };
        let out_dtype = value.dtype;
        let prev_rank = prev_dims.len();

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
                let outer = (prev_dims[axis] / inner).simplify();
                let new_rank = prev_rank + 1;
                let replacement = (0..prev_rank)
                    .map(|p| {
                        if p == axis {
                            MapEntry::Add(
                                Box::new(MapEntry::Mul(
                                    Box::new(MapEntry::Coord {
                                        from_end: new_rank - 1 - axis,
                                        extent: outer,
                                    }),
                                    inner,
                                )),
                                Box::new(MapEntry::Coord {
                                    from_end: new_rank - 1 - (axis + 1),
                                    extent: inner,
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
                    extent: merged,
                };
                let replacement = (0..prev_rank)
                    .map(|p| {
                        if p == axis1 {
                            MapEntry::Div(Box::new(merged_coord.clone()), inner)
                        } else if p == axis2 {
                            MapEntry::Rem(Box::new(merged_coord.clone()), inner)
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

        let composed: Vec<MapEntry> = entries
            .iter()
            .map(|entry| entry.substitute(&replacement, prev_rank))
            .collect();
        self.push_view(at, base, composed, new_dims, out_dtype)
    }

    /// Append post-schedule authoring checks (iota bounds pairs).
    pub fn post_check(&mut self, text: &str) {
        self.post_checks.push_str(text);
    }

    /// Record an output designation. Phase A: views stay refused until
    /// the native view-output story is proven.
    pub fn output(&mut self, at: usize, operand: &Operand, key: usize) {
        if self.poisoned.is_some() {
            return;
        }
        let id = match self.resolve(operand, &format!("output of t{at}")) {
            Ok(id) => id,
            Err(reason) => return self.poison(reason),
        };
        // Outputs of view VALUES are fine — the binding puts a contiguous
        // boundary on the value and search prices the materialization.
        // (The genuinely divergent case — their pipeline's non-contiguous
        // materialize path — already poisons via its gather1d.)
        self.outputs.push((id, key));
    }

    /// The live set: every value transitively reachable from the outputs,
    /// plus every input declaration (bindings enumerate all inputs).
    pub(crate) fn live_set(&self) -> Vec<bool> {
        let mut live = vec![false; self.values.len()];
        let mut stack: Vec<ValueId> = self.outputs.iter().map(|(id, _)| *id).collect();
        for (index, value) in self.values.iter().enumerate() {
            if value.input_slot.is_some() {
                stack.push(ValueId(index as u32));
            }
        }
        while let Some(id) = stack.pop() {
            let index = id.0 as usize;
            if live[index] {
                continue;
            }
            live[index] = true;
            stack.extend(self.values[index].operands.iter().copied());
        }
        live
    }

    fn render_value(&self, id: ValueId) -> String {
        let value = &self.values[id.0 as usize];
        let name = |id: &ValueId| format!("v{}", id.0);
        match value.form {
            RenderForm::Plain => {
                let mut parts: Vec<String> = value.operands.iter().map(name).collect();
                if !value.aux.is_empty() {
                    parts.push(value.aux.clone());
                }
                format!("(let v{} ({} {}))\n", id.0, value.constructor, parts.join(" "))
            }
            RenderForm::GatherList => {
                let data = name(&value.operands[0]);
                let mut list = "(LogicalTensorNil)".to_string();
                for coord in value.operands[1..].iter().rev() {
                    list = format!("(LogicalTensorCons {} {list})", name(coord));
                }
                format!("(let v{} ({} {data} {list}))\n", id.0, value.constructor)
            }
            RenderForm::ScatterList => {
                let init = name(&value.operands[0]);
                let src = name(value.operands.last().unwrap());
                let mut list = "(LogicalTensorNil)".to_string();
                for coord in value.operands[1..value.operands.len() - 1].iter().rev() {
                    list = format!("(LogicalTensorCons {} {list})", name(coord));
                }
                format!(
                    "(let v{} ({} {init} {list} {src}))\n",
                    id.0, value.constructor
                )
            }
        }
    }

    /// The rendered MODEL: live values in SSA order (creation order is
    /// topological — operands precede consumers), dead values elided,
    /// plus the output NAME annotations.
    pub fn model_text(&self) -> Result<String, String> {
        if let Some(reason) = &self.poisoned {
            return Err(format!("logical graph poisoned: {reason}"));
        }
        let live = self.live_set();
        let mut text = String::new();
        for index in 0..self.values.len() {
            if live[index] {
                text.push_str(&self.render_value(ValueId(index as u32)));
            }
        }
        for (id, key) in &self.outputs {
            text.push_str(&format!(
                "(union v{} (LogicalTensorNamed (LogicalIdLit \"out_{key}\")))\n",
                id.0
            ));
        }
        Ok(text)
    }

    /// The post-schedule authoring checks.
    pub fn post_checks(&self) -> &str {
        &self.post_checks
    }

    /// The native assembly SPLIT at the schedule (binding seeds inject
    /// before saturation): (pre-schedule text, input slots, output slots,
    /// post-schedule checks).
    #[allow(clippy::type_complexity)]
    pub fn native_parts(
        &self,
    ) -> Result<
        (
            String,
            Vec<InputSlot>,
            Vec<OutputSlot>,
            String,
        ),
        String,
    > {
        let mut text = self.model_text()?;
        let mut input_slots = Vec::new();
        let mut input_buffer_tensors = Vec::new();
        let mut next_buffer: i64 = 0;
        for (index, value) in self.values.iter().enumerate() {
            let Some(slot) = value.input_slot else { continue };
            let shape = Self::shape_term(&value.dims)?;
            let stem = format!("nat{slot}");
            let buffer = next_buffer;
            next_buffer += 1;
            text.push_str(&crate::reference_binding::input_binding(
                &stem,
                buffer as usize,
                &format!("v{index}"),
                &shape,
                &crate::reference_binding::width_term(value.dtype),
            ));
            input_buffer_tensors.push(format!("{stem}_buffer_tensor"));
            input_slots.push(InputSlot {
                tensor: petgraph::graph::NodeIndex::new(slot),
                buffer,
                size: slot as u64,
                value_name: format!("v{index}"),
            });
        }
        let mut output_slots = Vec::new();
        let mut output_buffer_tensors = Vec::new();
        for (id, key) in &self.outputs {
            let value = &self.values[id.0 as usize];
            let shape = Self::shape_term(&value.dims)?;
            let stem = format!("natout{key}");
            let buffer = next_buffer;
            next_buffer += 1;
            text.push_str(&crate::reference_binding::output_binding(
                &stem,
                buffer as usize,
                &format!("v{}", id.0),
                &shape,
                value.dtype,
            ));
            output_buffer_tensors.push(format!("{stem}_buffer_tensor"));
            output_slots.push(OutputSlot {
                tensor: petgraph::graph::NodeIndex::new(*key),
                buffer,
                size: *key as u64,
            });
        }
        text.push_str(&crate::reference_binding::boundary_lists(
            &input_buffer_tensors,
            &output_buffer_tensors,
            "nat_input_boundary",
            "nat_output_boundary",
        ));
        Ok((text, input_slots, output_slots, self.post_checks.clone()))
    }

    /// The assembled native program (model + reference-binding defaults).
    pub fn native_program(&self) -> Result<LogicalProgram, String> {
        let (pre, input_slots, output_slots, post_checks) = self.native_parts()?;
        Ok(LogicalProgram {
            text: format!("{pre}{}{post_checks}", crate::reference_binding::SCHEDULE),
            input_slots,
            output_slots,
        })
    }
}

// ─── Survivors of the interim translator (M3 Topic D) ───

/// One bound input: the graph tensor it carries, the buffer the runtime
/// allocated for it, and its declared size. Buffer ids are an internal,
/// sequential, binding-time allocation — inputs first, outputs after —
/// never derived from graph node indices (the retired HLIR keyspace).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSlot {
    pub tensor: petgraph::graph::NodeIndex,
    pub buffer: i64,
    pub size: u64,
    /// The input's SSA value name in the model text (`v{index}`) — the
    /// handle binding-time seeds (value ranges) attach to.
    pub value_name: String,
}

/// One bound output. See [`InputSlot`] for the allocation discipline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSlot {
    pub tensor: petgraph::graph::NodeIndex,
    pub buffer: i64,
    pub size: u64,
}

/// The assembled program plus the I/O binding tables the runtime needs
/// (moved here when the interim translator was deleted, M3 Topic D).
/// Buffer ids are runtime-internal sequential allocations; the runtime's
/// role-split maps translate tensor identities to buffers.
#[derive(Debug, Clone)]
pub struct LogicalProgram {
    /// Model + binding + schedule + authoring-contract checks. Run as
    /// `format!("{}\n\n{}", egglog_snippet::assembled_program(), text)`.
    pub text: String,
    /// Bound inputs in signature order.
    pub input_slots: Vec<InputSlot>,
    /// Bound outputs in output-slot order.
    pub output_slots: Vec<OutputSlot>,
}

/// Their RPN index expression rendered as OUR IntExpr term, with `z`
/// replaced by the given coordinate term and dyn vars resolved via the
/// pins. Add/Mul only for now (their slice path is affine); anything else
/// bails loudly.
pub(crate) fn int_expr_term(
    expr: &Expression,
    coord_terms: &[String],
    at: &str,
) -> AnyResult<String> {
    let mut stack: Vec<String> = Vec::new();
    for term in expr.terms.read().iter() {
        match term {
            Term::Num(n) => stack.push(format!("(IntLit {n})")),
            // Symbolic vars stay IntVar in the model — pins are
            // BINDING-side bounds seeds, never model content (same rule
            // as dim_term; the R3 fix, 2026-08-06). No character is
            // special: 'z' is an ordinary named symbol (P1, 2026-08-07).
            Term::Var(c) => stack.push(format!("(IntVar \"{c}\")")),
            // Coordinate atoms substitute their axis's CoordVar term; an
            // out-of-range axis is a coord Expression that leaked out of
            // its own iota — refuse loudly.
            Term::Coord(k) => match coord_terms.get(*k as usize) {
                Some(term) => stack.push(term.clone()),
                None => bail!(
                    "coordinate atom c{k} at {at}: out of range for rank {} — a \
                     coordinate Expression escaped its iota's value function",
                    coord_terms.len()
                ),
            },
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
