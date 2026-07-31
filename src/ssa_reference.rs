//! `SsaReferenceRuntime`: the CPU reference executor for logical-SSA
//! bufferized plans (ruling 2026-07-28: lives in luminal; once the path is
//! COMPLETE it replaces `ReferenceRuntime` — not before).
//!
//! Executes a [`BufferIrGraph`] directly: every buffer is a [`TypedBuffer`] sized
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

use crate::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};
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

/// M3 Step 2: what `load` captured from a natively-recorded Graph — the
/// pre-schedule program text (model + reference-binding defaults), the
/// I/O slots, the post-schedule authoring checks, plus whatever the
/// binding calls accumulate before `search` assembles and saturates.
struct NativeSpec {
    pre_schedule: String,
    input_slots: Vec<(petgraph::graph::NodeIndex, u64)>,
    output_slots: Vec<(usize, u64)>,
    post_checks: String,
    binding_seeds: String,
    ops: Option<Vec<&'static str>>,
}

#[derive(Default)]
pub struct SsaReferenceRuntime {
    plan: Option<BufferIrGraph>,
    /// Caller-staged data by numeric `BufferLit` id, consumed at `execute`.
    staged: FxHashMap<i64, Vec<f32>>,
    /// Post-execute storage, kept for `get_f32` / `get_bool`.
    storage: FxHashMap<BufferId, TypedBuffer>,
    /// `BufferLit` id → plan buffer, built at `load_plan`.
    lit_index: FxHashMap<i64, BufferId>,
    /// M3 Step 2 native-ladder state (`load` → bind → `with_ops` → `search`).
    native: Option<NativeSpec>,
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

    /// M3 Step 2, the native entry ladder: LOAD a natively-recorded graph
    /// (the model + reference-binding defaults; loud if the recorder is
    /// poisoned) — then bind, choose allowable ops, and `search`.
    pub fn load(graph: &crate::graph::Graph) -> Result<Self> {
        let (pre_schedule, input_slots, output_slots, post_checks) = graph
            .logical
            .native_parts()
            .map_err(|reason| anyhow!("native load refused: {reason}"))?;
        let mut runtime = Self::default();
        runtime.native = Some(NativeSpec {
            pre_schedule,
            input_slots,
            output_slots,
            post_checks,
            binding_seeds: String::new(),
            ops: None,
        });
        Ok(runtime)
    }

    /// BINDING: seed a dynamic dim's range (bounds-on-vars — never a pin).
    pub fn bind_dyn_range(&mut self, var: char, lower: u64, upper: u64) -> Result<()> {
        let spec = self.native.as_mut().ok_or_else(|| anyhow!("bind before load"))?;
        spec.binding_seeds.push_str(&format!(
            "(set (lower-bound-of (IntVar \"{var}\")) (bigint {lower}))\n\
             (set (upper-bound-of (IntVar \"{var}\")) (bigint {upper}))\n"
        ));
        Ok(())
    }

    /// The ALLOWABLE-OPS inventory for this runtime (per-runtime API,
    /// deliberately unstandardized — ruling 2026-07-30).
    pub fn with_ops(&mut self, ops: Vec<&'static str>) -> Result<()> {
        let spec = self.native.as_mut().ok_or_else(|| anyhow!("with_ops before load"))?;
        spec.ops = Some(ops);
        Ok(())
    }

    /// SEARCH: one saturation to fixpoint discovers the implementations;
    /// selection then prices every candidate by EXECUTING its bufferized
    /// plan on this runtime with the given data; the winner loads.
    pub fn search(
        &mut self,
        input_data: &FxHashMap<i64, Vec<f32>>,
        options: &crate::implementation_search::ImplementationSearchOptions,
    ) -> Result<crate::implementation_search::SearchOutcome> {
        let spec = self.native.take().ok_or_else(|| anyhow!("search before load"))?;
        let text = format!(
            "{}{}{}{}",
            spec.pre_schedule,
            spec.binding_seeds,
            crate::reference_binding::SCHEDULE,
            spec.post_checks
        );
        let program = crate::logical_graph::LogicalProgram {
            text,
            input_slots: spec.input_slots,
            output_slots: spec.output_slots,
        };
        let full = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            program.text
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &full)
            .map_err(|err| anyhow!("native saturation failed: {err}"))?;
        let serialized = egraph
            .serialize(egglog::SerializeConfig::default())
            .egraph;
        let outcome = crate::implementation_search::search_implementations_with_ops(
            &serialized,
            &program,
            input_data,
            options,
            spec.ops,
        )?;
        self.load_plan(outcome.best_plan.clone());
        Ok(outcome)
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
        // The element bit width picks the TYPED representation: 32 = f32;
        // 1 (the logical Bool width) and 8 (the reference binding's
        // byte-backed Bool layout) are both stored byte-backed. Anything
        // else refuses loudly.
        let mut storage: FxHashMap<BufferId, TypedBuffer> = FxHashMap::default();
        for (id, buffer) in &plan.buffers {
            let dims = buffer.dims.as_ref().ok_or_else(|| {
                anyhow!("buffer {} has no numeric geometry (symbolic dims are not executable yet)", buffer.label)
            })?;
            let bits = buffer.element_bits.ok_or_else(|| {
                anyhow!("buffer {} has no element bit width", buffer.label)
            })?;
            let numel = Self::numel(dims);
            let staged = buffer.lit.and_then(|lit| self.staged.get(&lit));
            let data = match bits {
                32 => match staged {
                    Some(staged) => {
                        ensure!(
                            staged.len() == numel,
                            "staged data for {} has {} elements, buffer holds {numel}",
                            buffer.label,
                            staged.len()
                        );
                        TypedBuffer::F32(staged.clone())
                    }
                    None => TypedBuffer::F32(vec![0.0; numel]),
                },
                // 1 = the logical Bool width, 8 = Bool8 (the boundary
                // code type, ruling 2026-07-30); both live as Bool8 codes
                // in reference storage.
                1 | 8 => {
                    ensure!(
                        staged.is_none(),
                        "buffer {} is boolean; set_data stages f32 only (a \
                         Bool8 staging surface does not exist yet)",
                        buffer.label
                    );
                    TypedBuffer::Bool8(vec![0u8; numel])
                }
                other => anyhow::bail!(
                    "buffer {} has unsupported element width {other} bits (f32 and bool only)",
                    buffer.label
                ),
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
                    ensure!(
                        data.type_name() == dest.type_name(),
                        "copy between {} and {} buffers",
                        data.type_name(),
                        dest.type_name()
                    );
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
                        let existing = storage
                            .get(id)
                            .ok_or_else(|| anyhow!("{} writes unknown buffer", op.label()))?;
                        dests.push(match existing {
                            TypedBuffer::F32(values) => TypedBuffer::F32(vec![0.0; values.len()]),
                            TypedBuffer::Bool8(bits) => TypedBuffer::Bool8(vec![0u8; bits.len()]),
                        });
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

    /// The f32 contents of the boundary buffer with this `BufferLit` id.
    /// Loud on a boolean buffer — use [`Self::get_bool`] for those.
    pub fn get_f32(&self, id: impl Into<i64>) -> Result<&Vec<f32>> {
        self.get_typed(id.into())?.as_f32()
    }

    /// The Bool8 codes of the boundary buffer with this `BufferLit` id
    /// (each element exactly 0 or 1 — the two legal codes).
    pub fn get_bool8(&self, id: impl Into<i64>) -> Result<&Vec<u8>> {
        self.get_typed(id.into())?.as_bool8()
    }

    fn get_typed(&self, id: i64) -> Result<&TypedBuffer> {
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
        use egglog::SerializeConfig;

    /// Their graph → our whole pipeline → an executed plan.
    /// M3 Topic C: the differentials' "ours" side runs the NATIVE path —
    /// recorder model + reference binding, with the graph's dyn pins
    /// injected as tight bounds seeds (the binding interface; the [n,n]
    /// collapse delivers literals by congruence).
    fn run_ssa(cx: &Graph, inputs: &[(petgraph::graph::NodeIndex, Vec<f32>)]) -> SsaReferenceRuntime {
        let (pre, input_slots, output_slots, post) =
            cx.logical.native_parts().expect("recorder clean for a covered graph");
        let mut vars: Vec<_> = cx.dyn_map.iter().collect();
        vars.sort();
        let mut seeds = String::new();
        for (var, value) in vars {
            seeds.push_str(&format!(
                "(set (lower-bound-of (IntVar \"{var}\")) (bigint {value}))\n\
                 (set (upper-bound-of (IntVar \"{var}\")) (bigint {value}))\n"
            ));
        }
        let program = crate::logical_graph::LogicalProgram {
            text: format!(
                "{pre}{seeds}{}{post}",
                crate::reference_binding::SCHEDULE
            ),
            input_slots,
            output_slots,
        };
        run_ssa_program(program, inputs)
    }



    /// The pipeline from an assembled LogicalProgram — shared by the
    /// translator path (run_ssa) and the native recorder path (M3).
    fn run_ssa_program(
        program: crate::logical_graph::LogicalProgram,
        inputs: &[(petgraph::graph::NodeIndex, Vec<f32>)],
    ) -> SsaReferenceRuntime {
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

    /// DYNAMIC DIMS over the bounds interface: the model declares
    /// `(IntVar "a")`, the binding seeds tight bounds from set_dim, the
    /// [n,n] collapse delivers the literal to the geometry walk — and the
    /// SAME symbolic graph shape re-renders per pin (the per-bucket model).
    #[test]
    fn differential_dynamic_dim_against_reference_runtime() {
        for pin in [3usize, 5usize] {
            let build = |dim: usize| {
                let mut cx = Graph::new();
                cx.set_dim('a', dim);
                let x = cx.tensor(('a', 2));
                let y = cx.tensor(('a', 2));
                let out = (x * y).output();
                (cx, x, y, out)
            };
            let data_x: Vec<f32> = (0..pin * 2).map(|v| v as f32 + 1.0).collect();
            let data_y: Vec<f32> = (0..pin * 2).map(|v| (v as f32) * 0.5 - 1.0).collect();

            let (mut cx, x, y, out) = build(pin);
            cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
            let mut theirs = cx.search(
                ReferenceRuntime::default(),
                CompileOptions::default().search_graph_limit(1),
            );
            theirs.set_data(x.id, data_x.clone());
            theirs.set_data(y.id, data_y.clone());
            theirs.execute(&cx.dyn_map);
            let expected = theirs.get_f32(out.id).clone();

            let (cx2, x2, y2, out2) = build(pin);
            let program = cx2.logical.native_program().expect("native program");
            assert!(
                program.text.contains("(IntVar \"a\")"),
                "the model must stay symbolic:\n{}",
                program.text
            );
            // The pin arrives as BINDING seeds, not model content: run_ssa
            // injects (bigint {pin}) bounds from the graph's dyn_map — the
            // execution below at both pins is the proof.
            let ours = run_ssa(&cx2, &[(x2.id, data_x), (y2.id, data_y)]);
            assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
        }
    }

    /// SLICE differential: their nonzero-start slice lowers to
    /// iota(z + start) + flat gather — the general-iota expression walker,
    /// the coordinate-form gather bridge (rank-1 data), and both kernels,
    /// against their runtime.
    #[test]
    fn differential_slice_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(8);
            let out = (x.slice(2..6) + x.slice(1..5)).output();
            (cx, x, out)
        };
        let x_data: Vec<f32> = (0..8).map(|v| (v * v) as f32).collect();

        let (mut cx, x, out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(out.id).clone();

        let (cx2, x2, out2) = build();
        let program = cx2.logical.native_program().expect("native program");
        assert!(
            program.text.contains("LogicalIndexMapApply"),
            "the slice must arrive as a view:\n{}",
            program.text
        );
        let ours = run_ssa(&cx2, &[(x2.id, x_data)]);
        assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
    }

    /// THE SEAM PAYOFF: a 2-D nonzero-start slice arrives structure-intact
    /// (SliceView) and translates as the view it is — while THEIR side of
    /// this same test runs the SliceView's legacy iota+gather lowering, so
    /// this differential proves BOTH halves of the seam at once.
    #[test]
    fn differential_two_dim_slice_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((4, 5));
            let out = x.slice((1..3, 2..5)).output();
            (cx, x, out)
        };
        let x_data: Vec<f32> = (0..20).map(|v| v as f32 * 1.5).collect();

        let (mut cx, x, out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(out.id).clone();

        let (cx2, x2, out2) = build();
        let program = cx2.logical.native_program().expect("native program");
        assert!(
            program.text.contains("LogicalIndexMapApply"),
            "the slice must arrive as a view:\n{}",
            program.text
        );
        let ours = run_ssa(&cx2, &[(x2.id, x_data)]);
        assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
    }

    /// UNFOLD differential through the seam: sliding windows (with a
    /// dilated variant) arrive structure-intact as UnfoldView and translate
    /// as two-coordinate affine view entries; their side runs the legacy
    /// flat iota+gather lowering.
    #[test]
    fn differential_unfold_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(8);
            let plain = x.unfold(3, 2, 1).output(); // windows at 0,2,4
            let y = cx.tensor(10);
            let dilated = y.unfold(3, 2, 2).output(); // effective window 5
            (cx, x, y, plain, dilated)
        };
        let x_data: Vec<f32> = (0..8).map(|v| (v * v) as f32).collect();
        let y_data: Vec<f32> = (0..10).map(|v| v as f32 * 3.0 - 5.0).collect();

        let (mut cx, x, y, plain, dilated) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.set_data(y.id, y_data.clone());
        theirs.execute(&cx.dyn_map);
        let expected_plain = theirs.get_f32(plain.id).clone();
        let expected_dilated = theirs.get_f32(dilated.id).clone();

        let (cx2, x2, y2, plain2, dilated2) = build();
        let program = cx2.logical.native_program().expect("native program");
        assert!(
            program.text.contains("LogicalIndexMapApply"),
            "unfold must arrive as a view:\n{}",
            program.text
        );
        let ours = run_ssa(&cx2, &[(x2.id, x_data), (y2.id, y_data)]);
        assert_close(ours.get_f32(plain2.id.index() as i64).unwrap(), &expected_plain);
        assert_close(
            ours.get_f32(dilated2.id.index() as i64).unwrap(),
            &expected_dilated,
        );
    }

    /// PAD differential, 1-D, THROUGH THE BOOL BRIDGE with zero frontend
    /// changes: their clamp iota (Min/Max terms) + mask iota (Gte/Lt as
    /// indicator values) + cast + blend translate directly — comparisons
    /// become IntCastFromBool(BoolLessThanInt ...) indicators, decided
    /// masks collapse via bounds, undecided ones evaluate in the kernels.
    /// Zero fill and nonzero fill both compared against their runtime.
    #[test]
    fn differential_pad_against_reference_runtime() {
        for fill in [0.0f32, 2.5f32] {
            let build = |fill: f32| {
                let mut cx = Graph::new();
                let x = cx.tensor(4);
                let out = x.pad((1, 2), fill).output();
                (cx, x, out)
            };
            let x_data = vec![10.0, 20.0, 30.0, 40.0];

            let (mut cx, x, out) = build(fill);
            cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
            let mut theirs = cx.search(
                ReferenceRuntime::default(),
                CompileOptions::default().search_graph_limit(1),
            );
            theirs.set_data(x.id, x_data.clone());
            theirs.execute(&cx.dyn_map);
            let expected = theirs.get_f32(out.id).clone();

            let (cx2, x2, out2) = build(fill);
            let program = cx2.logical.native_program().expect("native program");
            assert!(
                program.text.contains("IntCastFromBool"),
                "the mask must ride the bool bridge:\n{}",
                program.text
            );
            let ours = run_ssa(&cx2, &[(x2.id, x_data)]);
            assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
        }
    }

    /// RANK-2 PAD differential through the seam nodes: asymmetric padding
    /// on both axes (one axis before-only, one both sides), both fills —
    /// the case the flat lowering made untranslatable. Their side runs the
    /// legacy lowerings out of the seam nodes' to_egglog.
    #[test]
    fn differential_rank2_pad_against_reference_runtime() {
        for fill in [0.0f32, -1.5f32] {
            let build = |fill: f32| {
                let mut cx = Graph::new();
                let x = cx.tensor((3, 4));
                let out = x.pad(((1, 0), (2, 1)), fill).output();
                (cx, x, out)
            };
            let x_data: Vec<f32> = (0..12).map(|v| v as f32 + 1.0).collect();

            let (mut cx, x, out) = build(fill);
            cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
            let mut theirs = cx.search(
                ReferenceRuntime::default(),
                CompileOptions::default().search_graph_limit(1),
            );
            theirs.set_data(x.id, x_data.clone());
            theirs.execute(&cx.dyn_map);
            let expected = theirs.get_f32(out.id).clone();

            let (cx2, x2, out2) = build(fill);
            let program = cx2.logical.native_program().expect("native program");
            assert!(
                program.text.contains("IntMax") && program.text.contains("IntCastFromBool"),
                "clamp view + indicator mask expected:\n{}",
                program.text
            );
            let ours = run_ssa(&cx2, &[(x2.id, x_data)]);
            assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
        }
    }


    /// COORDINATE-FORM GATHER differential (ruling 2026-07-31): the
    /// primary gather — one Int coordinate tensor per data axis — records
    /// LogicalGather directly (rank-N native, no flatten trick); their
    /// runtime executes the transitional flat-index HLIR lowering as the
    /// oracle.
    #[test]
    fn differential_native_coordinate_gather() {
        let build = || {
            let mut cx = Graph::new();
            let data = cx.tensor((3, 4));
            let row = cx.tensor_dtyped((2, 3), crate::dtype::DType::Int);
            let col = cx.tensor_dtyped((2, 3), crate::dtype::DType::Int);
            let out = data.gather(&[row, col]).output();
            (cx, data, row, col, out)
        };
        let data_vals: Vec<f32> = (0..12).map(|v| v as f32 * 1.5 + 1.0).collect();
        let row_ints = vec![0i32, 2, 1, 2, 0, 1];
        let col_ints = vec![3i32, 0, 2, 3, 1, 0];
        let row_vals: Vec<f32> = row_ints.iter().map(|v| *v as f32).collect();
        let col_vals: Vec<f32> = col_ints.iter().map(|v| *v as f32).collect();

        let (mut cx, data, row, col, out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(data.id, data_vals.clone());
        theirs.set_data(row.id, row_ints);
        theirs.set_data(col.id, col_ints);
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(out.id).clone();

        let (cx2, data2, row2, col2, out2) = build();
        let program = cx2.logical.native_program().expect("native program");
        assert!(
            program.text.contains("(LogicalGather rec_t"),
            "coordinate-form gather expected in the model:\n{}",
            program.text
        );
        let ours = run_ssa_program(
            program,
            &[
                (data2.id, data_vals),
                (row2.id, row_vals),
                (col2.id, col_vals),
            ],
        );
        assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
    }

    /// COORDINATE-FORM SCATTER differential: dest updated at (row, col)
    /// coordinate positions with src — value semantics, against their
    /// flat-Scatter lowering.
    #[test]
    fn differential_native_coordinate_scatter() {
        let build = || {
            let mut cx = Graph::new();
            let dest = cx.tensor((3, 4));
            let row = cx.tensor_dtyped(4, crate::dtype::DType::Int);
            let col = cx.tensor_dtyped(4, crate::dtype::DType::Int);
            let src = cx.tensor(4);
            let out = dest.scatter(&[row, col], src).output();
            (cx, dest, row, col, src, out)
        };
        let dest_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let row_ints = vec![0i32, 1, 2, 1];
        let col_ints = vec![1i32, 3, 0, 0];
        let row_vals: Vec<f32> = row_ints.iter().map(|v| *v as f32).collect();
        let col_vals: Vec<f32> = col_ints.iter().map(|v| *v as f32).collect();
        let src_vals = vec![100.0, 200.0, 300.0, 400.0];

        let (mut cx, dest, row, col, src, out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(dest.id, dest_vals.clone());
        theirs.set_data(row.id, row_ints);
        theirs.set_data(col.id, col_ints);
        theirs.set_data(src.id, src_vals.clone());
        theirs.execute(&cx.dyn_map);
        let expected = theirs.get_f32(out.id).clone();

        let (cx2, dest2, row2, col2, src2, out2) = build();
        let program = cx2.logical.native_program().expect("native program");
        assert!(
            program.text.contains("(LogicalScatter rec_t"),
            "coordinate-form scatter expected in the model:\n{}",
            program.text
        );
        let ours = run_ssa_program(
            program,
            &[
                (dest2.id, dest_vals),
                (row2.id, row_vals),
                (col2.id, col_vals),
                (src2.id, src_vals),
            ],
        );
        assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
    }

    /// Uncovered constructs POISON the recorder with an attributable
    /// reason — the native path refuses loudly, never mistranslates.
    #[test]
    fn recorder_poisons_on_uncovered_family() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 3));
        let y = cx.tensor_dtyped(6, crate::dtype::DType::Int);
        let _out = x.flatten().gather1d(y).output();
        let reason = cx.logical.poisoned().expect("gather poisons the recorder");
        assert!(reason.contains("gather"), "attributable reason: {reason}");
        assert!(cx.logical.native_program().is_err());
    }

    /// M3 STEP 1: THE FIRST NATIVE DIFFERENTIAL — the recorder's model +
    /// the reference binding generator, with NO translator anywhere,
    /// against their full search + runtime.
    #[test]
    fn differential_native_recorder_simple_elementwise() {
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
        let expected_a = theirs.get_f32(a.id).clone();
        let expected_d = theirs.get_f32(d.id).clone();

        let (cx2, b2, c2, g2, e2, a2, d2) = build();
        let program = cx2.logical.native_program().expect("native program");
        let ours = run_ssa_program(
            program,
            &[
                (b2.id, b_data),
                (c2.id, c_data),
                (g2.id, g_data),
                (e2.id, e_data),
            ],
        );
        assert_close(ours.get_f32(a2.id.index() as i64).unwrap(), &expected_a);
        assert_close(ours.get_f32(d2.id.index() as i64).unwrap(), &expected_d);
    }

    /// TYPED-BUFFERS differential: lt produces a genuinely BOOLEAN
    /// intermediate (byte-backed u8 in reference storage; the logical dtype
    /// stays 1-bit), cast bridges it back to f32 as exact 0/1 indicators,
    /// and blend arithmetic runs downstream — element-for-element against
    /// their full search + ReferenceRuntime.
    #[test]
    fn differential_less_than_cast_against_reference_runtime() {
        use crate::dtype::DType;
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let y = cx.tensor((2, 3));
            let out = (x.lt(y).cast(DType::F32) * 3.0 + 1.0).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 5.0, 2.0, 8.0, -1.0, 0.0];
        let y_data = vec![2.0, 4.0, 2.0, 9.0, -2.0, 0.5];

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
        let program = cx2.logical.native_program().expect("native program");
        assert!(
            program.text.contains("LogicalLessThan") && program.text.contains("LogicalCast"),
            "comparison + cast expected in the model:\n{}",
            program.text
        );
        let ours = run_ssa(&cx2, &[(x2.id, x_data), (y2.id, y_data)]);
        assert_close(ours.get_f32(out2.id.index() as i64).unwrap(), &expected);
    }

    /// BOOL8 BOUNDARY differential (ruling 2026-07-30): a bare lt output
    /// crosses the boundary as Bool8 — the translator inserts the
    /// LogicalCast to Bool8, the boundary layout speaks (bits-of (Bool8)),
    /// and get_bool8 yields exactly the two legal codes — against their
    /// runtime's native Vec<bool> for the same graph.
    #[test]
    fn differential_bool8_boundary_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3));
            let y = cx.tensor((2, 3));
            let out = x.lt(y).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 5.0, 2.0, 8.0, -1.0, 0.0];
        let y_data = vec![2.0, 4.0, 2.0, 9.0, -2.0, 0.5];

        let (mut cx, x, y, out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.set_data(y.id, y_data.clone());
        theirs.execute(&cx.dyn_map);
        // Their get_f32 panics on Bool outputs; read the typed buffer the
        // way get_f32 locates it and take the native bool vector.
        let expected: Vec<bool> = {
            let output_id = theirs
                .graph
                .node_indices()
                .find(|n| {
                    if let Some(crate::hlir::Output { node, .. }) =
                        (**theirs.graph[*n]).as_any().downcast_ref::<crate::hlir::Output>()
                    {
                        *node == out.id.index()
                    } else {
                        false
                    }
                })
                .expect("their output node");
            theirs.buffers.get(&output_id).expect("their bool buffer").to_bool_vec()
        };

        let (cx2, x2, y2, out2) = build();
        let program = cx2.logical.native_program().expect("native program");
        assert!(
            program.text.contains("(LogicalCast rec_t") && program.text.contains("(Bool8)"),
            "boundary Bool8 cast expected in the binding:\n{}",
            program.text
        );
        assert!(
            program.text.contains("(bits-of (Bool8))"),
            "Bool8 boundary layout width expected:\n{}",
            program.text
        );
        let ours = run_ssa(&cx2, &[(x2.id, x_data), (y2.id, y_data)]);
        let codes = ours.get_bool8(out2.id.index() as i64).expect("bool8 boundary");
        assert_eq!(codes.len(), expected.len());
        for (index, (code, truth)) in codes.iter().zip(&expected).enumerate() {
            assert!(*code <= 1, "ill-formed Bool8 code {code} at {index}");
            assert_eq!(
                *code == 1,
                *truth,
                "element {index}: our code {code} vs their {truth}"
            );
        }
    }

    /// RESHAPE differentials: split (mixed-radix group entries), merge
    /// (div/rem digit entries), and flatten (a multi-axis merge run) — all
    /// read structurally off the tracker strides, no seam nodes needed.
    #[test]
    fn differential_reshapes_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let a = cx.tensor(12);
            let b = cx.tensor((3, 4));
            let split_out = (a.split_dims(0, 4) * b).output(); // [12] -> [3,4]
            let c = cx.tensor((3, 4));
            let d = cx.tensor(12);
            let merge_out = (c.merge_dims(0, 1) * d).output(); // [3,4] -> [12]
            let e = cx.tensor((2, 3, 2));
            let f = cx.tensor(12);
            let flatten_out = (e.flatten() * f).output(); // [2,3,2] -> [12]
            (cx, a, b, c, d, e, f, split_out, merge_out, flatten_out)
        };
        let v12a: Vec<f32> = (0..12).map(|v| v as f32 + 1.0).collect();
        let v12b: Vec<f32> = (0..12).map(|v| v as f32 * 0.5 - 2.0).collect();
        let v12c: Vec<f32> = (0..12).map(|v| (v * v) as f32).collect();
        let v12d: Vec<f32> = (0..12).map(|v| v as f32 - 6.0).collect();
        let v12e: Vec<f32> = (0..12).map(|v| v as f32 * 1.5).collect();
        let v12f: Vec<f32> = (0..12).map(|v| 12.0 - v as f32).collect();

        let (mut cx, a, b, c, d, e, f, split_out, merge_out, flatten_out) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(a.id, v12a.clone());
        theirs.set_data(b.id, v12b.clone());
        theirs.set_data(c.id, v12c.clone());
        theirs.set_data(d.id, v12d.clone());
        theirs.set_data(e.id, v12e.clone());
        theirs.set_data(f.id, v12f.clone());
        theirs.execute(&cx.dyn_map);
        let expected_split = theirs.get_f32(split_out.id).clone();
        let expected_merge = theirs.get_f32(merge_out.id).clone();
        let expected_flatten = theirs.get_f32(flatten_out.id).clone();

        let (cx2, a2, b2, c2, d2, e2, f2, split2, merge2, flatten2) = build();
        let ours = run_ssa(
            &cx2,
            &[
                (a2.id, v12a),
                (b2.id, v12b),
                (c2.id, v12c),
                (d2.id, v12d),
                (e2.id, v12e),
                (f2.id, v12f),
            ],
        );
        assert_close(ours.get_f32(split2.id.index() as i64).unwrap(), &expected_split);
        assert_close(ours.get_f32(merge2.id.index() as i64).unwrap(), &expected_merge);
        assert_close(
            ours.get_f32(flatten2.id.index() as i64).unwrap(),
            &expected_flatten,
        );
    }

    /// REPEAT differential: tiling strides (z % d) lift into IntTruncRem
    /// map entries.
    #[test]
    fn differential_repeat_against_reference_runtime() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(3);
            let y = cx.tensor(12);
            let out = (x.repeat(4) * y).output();
            (cx, x, y, out)
        };
        let x_data = vec![1.0, 2.0, 3.0];
        let y_data: Vec<f32> = (0..12).map(|v| v as f32 + 0.5).collect();

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
