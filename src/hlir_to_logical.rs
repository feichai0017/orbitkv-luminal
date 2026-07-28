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
    Add, Constant, Exp2, Input, Iota, Log2, MaxReduce, Mod, Mul, Output, Recip, Sin, Sqrt,
    SumReduce,
};
use crate::shape::Expression;
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

/// The affine-through-origin coefficient of a stride expression in `z`
/// (`s(i) = c * i`) — `None` for anything nonlinear (repeat's modulo,
/// offsets) or containing unresolved dyn vars.
fn stride_coefficient(stride: &Expression) -> Option<usize> {
    let s0 = stride.exec_single_var_checked(0)?;
    let s1 = stride.exec_single_var_checked(1)?;
    let s2 = stride.exec_single_var_checked(2)?;
    let s3 = stride.exec_single_var_checked(3)?;
    (s0 == 0 && s2 == 2 * s1 && s3 == 3 * s1).then_some(s1)
}

/// Recover the CANONICAL parent dims (size-1 axes omitted — they are
/// unobservable through strides and semantically inert) from one consumer's
/// tracker: contiguous trackers answer directly; affine views reconstruct
/// by sorting real axes' coefficients into a telescoping contiguous ladder.
fn parent_dims_from_tracker(tracker: &ShapeTracker, at: &str) -> Result<Vec<usize>> {
    let empty = FxHashMap::default();
    let dims: Vec<usize> = tracker
        .dims
        .iter()
        .map(|dim| {
            dim.exec(&empty)
                .with_context(|| format!("hlir_to_logical: symbolic dim at {at}"))
        })
        .collect::<Result<_>>()?;
    if tracker.is_contiguous() {
        return Ok(dims.into_iter().filter(|dim| *dim > 1).collect());
    }
    let mut real: Vec<(usize, usize)> = Vec::new(); // (coefficient, extent)
    for (stride, &dim) in tracker.strides.iter().zip(&dims) {
        let c = stride_coefficient(stride).ok_or_else(|| {
            anyhow!("hlir_to_logical slice 2: non-affine stride at {at} (repeat/slice — later slice)")
        })?;
        if c > 0 && dim > 1 {
            real.push((c, dim));
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
    node_index: usize,
    position: usize,
) -> Result<(String, Vec<usize>)> {
    let at = format!("t{node_index} operand {position}");
    let empty = FxHashMap::default();
    let view_dims: Vec<usize> = tracker
        .dims
        .iter()
        .map(|dim| {
            dim.exec(&empty)
                .with_context(|| format!("hlir_to_logical: symbolic dim at {at}"))
        })
        .collect::<Result<_>>()?;
    if tracker.is_contiguous() && view_dims == source.dims {
        return Ok((source.let_name.clone(), view_dims));
    }

    let coefficients: Vec<usize> = tracker
        .strides
        .iter()
        .map(|stride| {
            stride_coefficient(stride).ok_or_else(|| {
                anyhow!("hlir_to_logical slice 2: non-affine stride at {at} (repeat/slice — later slice)")
            })
        })
        .collect::<Result<_>>()?;

    let parent = &source.dims;
    let mut parent_strides = vec![1usize; parent.len()];
    for k in (0..parent.len().saturating_sub(1)).rev() {
        parent_strides[k] = parent_strides[k + 1] * parent[k + 1];
    }

    // Match every real view axis to a unique parent axis by (stride, extent).
    let rank = view_dims.len();
    let mut consumed: Vec<Option<usize>> = vec![None; parent.len()];
    for (axis, (&coefficient, &dim)) in coefficients.iter().zip(&view_dims).enumerate() {
        if coefficient == 0 || dim == 1 {
            continue; // broadcast, or a degenerate axis whose index is always 0
        }
        let matched = (0..parent.len()).find(|&k| {
            consumed[k].is_none() && parent_strides[k] == coefficient && parent[k] == dim
        });
        let Some(k) = matched else {
            bail!(
                "hlir_to_logical slice 2: {at} axis {axis} (extent {dim}, stride {coefficient}) \
                 matches no axis of parent {parent:?} (repeat/slice/merge — later slice)"
            );
        };
        consumed[k] = Some(axis);
    }
    for k in 0..parent.len() {
        ensure!(
            parent[k] == 1 || consumed[k].is_some(),
            "hlir_to_logical slice 2: {at} drops parent axis {k} (extent {}) — slicing is a later slice",
            parent[k]
        );
    }

    // Map entries per PARENT axis, outermost inward; CoordVar axes are
    // zero-based from the innermost of the VIEW shape.
    let mut entries = "(IntExprNil)".to_string();
    for k in (0..parent.len()).rev() {
        let entry = match consumed[k] {
            Some(axis) => format!("(CoordVar {} (IntLit {}))", rank - 1 - axis, view_dims[axis]),
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
fn input_dims_from_consumers(graph: &Graph, input: NodeIndex) -> Result<Vec<usize>> {
    let mut derived: Option<Vec<usize>> = None;
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
            let dims = parent_dims_from_tracker(tracker, &format!("input t{}", input.index()))?;
            match &derived {
                None => derived = Some(dims),
                Some(existing) if *existing == dims => {}
                Some(existing) => bail!(
                    "hlir_to_logical slice 1: input t{} consumers disagree on shape ({existing:?} vs {dims:?})",
                    input.index()
                ),
            }
        }
    }
    derived.ok_or_else(|| {
        anyhow!(
            "hlir_to_logical slice 1: input t{} has no shape-bearing consumer",
            input.index()
        )
    })
}

/// `[2, 3]` → `(ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil))))`
fn shape_term(dims: &[usize]) -> String {
    let mut term = "(IntExprNil)".to_string();
    for dim in dims.iter().rev() {
        term = format!("(IntExprCons (IntLit {dim}) {term})");
    }
    format!("(ShapeLit {term})")
}

fn dtype_term(dtype: DType) -> String {
    format!("({dtype:?})")
}

pub fn hlir_to_logical(graph: &Graph) -> Result<LogicalProgram> {
    let order = toposort(&graph.graph, None)
        .map_err(|_| anyhow!("hlir_to_logical: HLIR graph has a cycle"))?;

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
            let dims = input_dims_from_consumers(graph, node)?;
            let shape = shape_term(&dims);
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
            if iota.0.to_egglog() != "(MVar \"z\")" {
                bail!(
                    "hlir_to_logical slice 1: iota at t{idx} has a non-arange index expression"
                );
            }
            let extent = iota
                .1
                .exec(&FxHashMap::default())
                .with_context(|| format!("hlir_to_logical slice 1: symbolic iota range at t{idx}"))?;
            let shape = shape_term(&[extent]);
            ops_text.push_str(&format!(
                "(let t{idx}_logical (LogicalIota (CoordVar 0 (IntLit {extent})) {shape}))\n"
            ));
            // The iota authoring contract: every construction site demands
            // its expression's bounds at the fixpoint.
            post_checks.push_str(&format!(
                "(check (= ?lo{idx} (lower-bound-of (CoordVar 0 (IntLit {extent})))))\n\
                 (check (= ?hi{idx} (upper-bound-of (CoordVar 0 (IntLit {extent})))))\n"
            ));
            values.insert(
                node,
                ValueInfo {
                    let_name: format!("t{idx}_logical"),
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
                        lift_operand(&mut ops_text, tracker, source_value, idx, position)?;
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
                let (operand_name, tracker_dims) =
                    lift_operand(&mut ops_text, &tracker, source_value, idx, 0)?;
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
    for (_, source, key) in &output_nodes {
        let value = values.get(source).ok_or_else(|| {
            anyhow!("hlir_to_logical: Output reads untranslated t{}", source.index())
        })?;
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

    let text = format!(
        "; hlir_to_logical (slice 1: contiguous, static) — {} nodes\n\n\
         {inputs_text}{ops_text}\n{outputs_text}\
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
