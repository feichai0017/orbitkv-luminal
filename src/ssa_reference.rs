//! `SsaReferenceRuntime`: the CPU reference executor for logical-SSA
//! bufferized plans (ruling 2026-07-28: lives in luminal; once the path is
//! COMPLETE it replaces `ReferenceRuntime` — not before).
//!
//! Executes a [`BufferIrGraph`] directly: every buffer is a `Vec<f32>` sized
//! from the plan's annotated numeric geometry (dims × 32-bit elements —
//! slice 1 is f32-only and bails loudly otherwise); compute nodes dispatch
//! through each op's own `reference_execute` kernel (registry-style, no
//! match statements — an op without a kernel refuses by name). Caller data
//! binds by the numeric `BufferLit` id, the same key `hlir_to_logical`
//! derives from HLIR node indices, so differential tests against
//! `ReferenceRuntime` bind identically on both sides.

use anyhow::{Context, Result, anyhow, ensure};
use petgraph::algo::toposort;
use rustc_hash::FxHashMap;

use crate::buffer_tensor_ir::ReferenceKernelCtx;
use crate::bufferize::{BufferId, BufferIrGraph, BufferNode};

/// The reference backend's implementation inventory: every registered op
/// EXCEPT the zero-copy view (ruling 2026-07-28: the reference runtime
/// implements views by literally materializing into contiguous layout —
/// the kernels are layout-blind, so every operand buffer must be its own
/// contiguous home).
pub fn reference_allow_list() -> Vec<&'static str> {
    crate::layout_ir::ops::built_in_matchers()
        .iter()
        .map(|matcher| matcher.egglog_constructor())
        .filter(|constructor| *constructor != "LayoutTensorOpIndexMapApplyViewGeneric")
        .collect()
}

#[derive(Default)]
pub struct SsaReferenceRuntime {
    plan: Option<BufferIrGraph>,
    /// Caller-staged data by numeric `BufferLit` id, consumed at `execute`.
    staged: FxHashMap<i64, Vec<f32>>,
    /// Post-execute storage, kept for `get_f32`.
    storage: FxHashMap<BufferId, Vec<f32>>,
    /// `BufferLit` id → plan buffer, built at `load_plan`.
    lit_index: FxHashMap<i64, BufferId>,
}

impl SsaReferenceRuntime {
    pub fn load_plan(&mut self, plan: BufferIrGraph) {
        self.lit_index = plan
            .buffers
            .values()
            .filter_map(|buffer| buffer.lit.map(|lit| (lit, buffer.id.clone())))
            .collect();
        self.plan = Some(plan);
        self.storage.clear();
    }

    /// Stage caller data for the boundary buffer with this `BufferLit` id.
    pub fn set_data(&mut self, id: impl Into<i64>, data: Vec<f32>) {
        self.staged.insert(id.into(), data);
    }

    fn numel(dims: &[i64]) -> usize {
        dims.iter().product::<i64>().max(0) as usize
    }

    pub fn execute(&mut self) -> Result<()> {
        let plan = self.plan.as_ref().ok_or_else(|| anyhow!("no plan loaded"))?;

        // Materialize every buffer: staged caller data where provided
        // (length-checked against the annotated geometry), zeros otherwise.
        let mut storage: FxHashMap<BufferId, Vec<f32>> = FxHashMap::default();
        for (id, buffer) in &plan.buffers {
            let dims = buffer.dims.as_ref().ok_or_else(|| {
                anyhow!("buffer {} has no numeric geometry (symbolic dims are not executable yet)", buffer.label)
            })?;
            let bits = buffer.element_bits.ok_or_else(|| {
                anyhow!("buffer {} has no element bit width", buffer.label)
            })?;
            ensure!(bits == 32, "slice 1 executes f32 only; buffer {} is {bits}-bit", buffer.label);
            let numel = Self::numel(dims);
            let data = match buffer.lit.and_then(|lit| self.staged.get(&lit)) {
                Some(staged) => {
                    ensure!(
                        staged.len() == numel,
                        "staged data for {} has {} elements, buffer holds {numel}",
                        buffer.label,
                        staged.len()
                    );
                    staged.clone()
                }
                None => vec![0.0; numel],
            };
            storage.insert(id.clone(), data);
        }

        // Inputs that the plan reads MUST have been staged — zeros would be
        // silently wrong numbers, and silence is the one forbidden failure.
        for node in plan.dag.node_weights() {
            if let BufferNode::BufferInput { slots } = node {
                for slot in slots {
                    let buffer = plan
                        .buffers
                        .get(&slot.buffer)
                        .ok_or_else(|| anyhow!("input slot references unknown buffer"))?;
                    let lit = buffer.lit.ok_or_else(|| {
                        anyhow!("input buffer {} has no BufferLit id to bind by", buffer.label)
                    })?;
                    ensure!(
                        self.staged.contains_key(&lit),
                        "input buffer {} (BufferLit {lit}) was never set_data",
                        buffer.label
                    );
                }
            }
        }

        // Execute in dependency order (anti-edges are real edges, so WAR
        // ordering rides the same toposort).
        let order = toposort(&plan.dag, None)
            .map_err(|_| anyhow!("bufferized plan has a cycle"))?;
        for index in order {
            match &plan.dag[index] {
                BufferNode::BufferInput { .. } | BufferNode::BufferOutput { .. } => {}
                BufferNode::BufferCopy { src, dst } => {
                    let data = storage
                        .get(src)
                        .ok_or_else(|| anyhow!("copy reads unknown buffer"))?
                        .clone();
                    let dest = storage
                        .get_mut(dst)
                        .ok_or_else(|| anyhow!("copy writes unknown buffer"))?;
                    ensure!(data.len() == dest.len(), "copy length mismatch");
                    *dest = data;
                }
                BufferNode::Compute { op, reads, writes, .. } => {
                    let mut operands = Vec::with_capacity(reads.len());
                    let mut operand_dims = Vec::with_capacity(reads.len());
                    for id in reads {
                        operands.push(
                            storage
                                .get(id)
                                .ok_or_else(|| anyhow!("{} reads unknown buffer", op.label()))?
                                .clone(),
                        );
                        let dims = plan.buffers[id]
                            .dims
                            .as_ref()
                            .ok_or_else(|| anyhow!("{} operand lacks geometry", op.label()))?;
                        operand_dims.push(dims.iter().map(|d| *d as usize).collect());
                    }
                    let mut dests = Vec::with_capacity(writes.len());
                    for id in writes {
                        let len = storage
                            .get(id)
                            .ok_or_else(|| anyhow!("{} writes unknown buffer", op.label()))?
                            .len();
                        dests.push(vec![0.0; len]);
                    }
                    let mut ctx = ReferenceKernelCtx { operands, operand_dims, dests };
                    op.reference_execute(&mut ctx)
                        .with_context(|| format!("executing {}", op.label()))?;
                    for (id, data) in writes.iter().zip(ctx.dests) {
                        *storage.get_mut(id).expect("write buffer exists") = data;
                    }
                }
            }
        }

        self.storage = storage;
        Ok(())
    }

    /// The contents of the boundary buffer with this `BufferLit` id.
    pub fn get_f32(&self, id: impl Into<i64>) -> Result<&Vec<f32>> {
        let id = id.into();
        let buffer = self
            .lit_index
            .get(&id)
            .ok_or_else(|| anyhow!("no boundary buffer with BufferLit {id}"))?;
        self.storage
            .get(buffer)
            .ok_or_else(|| anyhow!("buffer for BufferLit {id} has no contents (execute first)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CompileOptions, Graph};
    use crate::hlir::ReferenceRuntime;
    use crate::op::Runtime;
    use crate::hlir_to_logical::hlir_to_logical;
    use egglog::SerializeConfig;

    /// Their graph → our whole pipeline → an executed plan.
    fn run_ssa(cx: &Graph, inputs: &[(petgraph::graph::NodeIndex, Vec<f32>)]) -> SsaReferenceRuntime {
        let program = hlir_to_logical(cx).expect("translates");
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            program.text
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph.parse_and_run_program(None, &text).expect("program runs");
        let serialized = egraph.serialize(SerializeConfig::default()).egraph;
        let allow = reference_allow_list();
        let extracted = crate::extractor::extract_layout_ir_with_ops(&serialized, Some(&allow))
            .expect("extracts")
            .expect("plan");
        let plan = crate::bufferize::bufferize(&crate::dps::dps_rewrite(&extracted))
            .expect("bufferizes");
        let mut rt = SsaReferenceRuntime::default();
        rt.load_plan(plan);
        for (node, data) in inputs {
            rt.set_data(node.index() as i64, data.clone());
        }
        rt.execute().expect("executes");
        rt
    }

    fn assert_close(ours: &[f32], theirs: &[f32]) {
        assert_eq!(ours.len(), theirs.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(theirs).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "element {index}: ours {a} vs theirs {b}"
            );
        }
    }

    /// THE DIFFERENTIAL: their `simple`-test graph (a = b*c + g and
    /// d = sin(b*c / e)) through BOTH pipelines — their egglog search +
    /// ReferenceRuntime vs our translation + saturation + extraction +
    /// bufferization + SsaReferenceRuntime — must agree numerically.
    #[test]
    fn differential_simple_elementwise_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let b = cx.tensor(3);
            let c = cx.tensor(3);
            let g = cx.tensor(3);
            let e = cx.tensor(3);
            let a = (b * c + g).output();
            let d = (b * c / e).sin().output();
            (cx, b, c, g, e, a, d)
        };
        let b_data = vec![1.0, 2.0, 3.0];
        let c_data = vec![4.0, 5.0, 6.0];
        let g_data = vec![0.5, -1.5, 2.5];
        let e_data = vec![2.0, 4.0, 8.0];

        // Theirs: first viable graph, no real search.
        let (mut cx, b, c, g, e, a, d) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(b.id, b_data.clone());
        theirs.set_data(c.id, c_data.clone());
        theirs.set_data(g.id, g_data.clone());
        theirs.set_data(e.id, e_data.clone());
        theirs.execute(&cx.dyn_map);
        let their_a = theirs.get_f32(a.id).clone();
        let their_d = theirs.get_f32(d.id).clone();

        // Ours, same graph shape, same data.
        let (cx2, b2, c2, g2, e2, a2, d2) = build();
        let ours = run_ssa(
            &cx2,
            &[
                (b2.id, b_data),
                (c2.id, c_data),
                (g2.id, g_data),
                (e2.id, e_data),
            ],
        );
        assert_close(ours.get_f32(a2.id.index() as i64).unwrap(), &their_a);
        assert_close(ours.get_f32(d2.id.index() as i64).unwrap(), &their_d);
    }

    /// Slice-2 differential: a permuted operand (transpose view) through
    /// both pipelines.
    #[test]
    fn differential_permuted_mul_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let y = cx.tensor((3, 2));
            let out = (x.permute((1, 0)) * y).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let (mut cx, x, y, out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.set_data(y.id, y_data.clone());
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(out.id).clone();

        let (cx2, x2, y2, out2) = build();
        let ours = run_ssa(&cx2, &[(x2.id, x_data), (y2.id, y_data)]);
        assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
    }

    /// Slice-2 differential: subtraction routes through their Neg (a
    /// broadcast constant) — rank-0 LogicalConstant + lifted broadcast view.
    #[test]
    fn differential_subtraction_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(4);
            let y = cx.tensor(4);
            let out = (x - y).output();
            (cx, x, y, out)
        };
        let x_data = vec![10.0, 20.0, 30.0, 40.0];
        let y_data = vec![1.0, 2.0, 3.0, 4.0];

        let (mut cx, x, y, out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.set_data(y.id, y_data.clone());
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(out.id).clone();

        let (cx2, x2, y2, out2) = build();
        let ours = run_ssa(&cx2, &[(x2.id, x_data), (y2.id, y_data)]);
        assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
    }

    /// THE MATMUL DIFFERENTIAL: their fully-decomposed frontend matmul
    /// (movement views + Mul + SumReduce) through our whole pipeline —
    /// slice-2 lifting translating their expand/permute stride patterns.
    #[test]
    fn differential_matmul_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let a = cx.tensor((2, 3));
            let b = cx.tensor((3, 4));
            let c = a.matmul(b).output();
            (cx, a, b, c)
        };
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();

        let (mut cx, a, b, c) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(a.id, a_data.clone());
        theirs.set_data(b.id, b_data.clone());
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(c.id).clone();

        let (cx2, a2, b2, c2) = build();
        let ours = run_ssa(&cx2, &[(a2.id, a_data), (b2.id, b_data)]);
        assert_close(ours.get_f32(c2.id.index() as i64).unwrap(), &expected);
    }

    /// Reduction differential: sum over the front axis of a [2, 3] tensor,
    /// crossing the axis-convention flip and the reduce kernel.
    #[test]
    fn differential_sum_reduce_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let s = x.sum(0).output();
            (cx, x, s)
        };
        let x_data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];

        let (mut cx, x, s) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(s.id).clone();

        let (cx2, x2, s2) = build();
        let ours = run_ssa(&cx2, &[(x2.id, x_data)]);
        assert_close(ours.get_f32(s2.id.index() as i64).unwrap(), &expected);
    }
}
