//! M4 Phase 3 pin, RESTATED at Phase 5: CUDA codegen strings are
//! IDENTICAL whether `CodegenCtx` geometry comes from the shared buffer
//! table (the pre-Phase-3 device path) or from the plan node's own
//! `SlotDescriptor`s (the Phase-3 device path) — FOR NODES WITHOUT
//! COMPOSED ACCESS. The Phase-3 wording ("no views are electable on
//! real backends") was the zero-behavior premise; Phase 5 flips it
//! deliberately: the view op is now claimed through the plan-transparent
//! class, so view-consumer nodes carry composed access their descriptors
//! know and the buffer table never can. On those nodes the descriptor
//! route MUST diverge (it emits the Phase-4 strided read-through); the
//! divergence itself is pinned below, and the strided string gates +
//! `view_admission` own the read-through's content.
//!
//! Set `CODEGEN_DUMP_DIR` to also write every generated source to disk
//! (used to diff before/after captures across the Phase-3 landing).

use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode};
use luminal::dtype::PlanDtype;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::{kernels, CudaRuntime};
use std::collections::HashMap;

/// The PRE-Phase-3 construction, replicated verbatim: per-node geometry
/// looked up in the shared buffer table by BufferId.
fn sources_via_buffer_table(plan: &BufferIrGraph) -> Vec<(String, String)> {
    let geometry: HashMap<BufferId, (Vec<usize>, PlanDtype)> = plan
        .buffers
        .iter()
        .map(|(id, buffer)| {
            let dims: Vec<usize> = buffer
                .dims
                .as_ref()
                .expect("plan buffer has numeric geometry")
                .iter()
                .map(|&d| usize::try_from(d).unwrap_or(0))
                .collect();
            (id.clone(), (dims, buffer.dtype.expect("plan buffer has dtype")))
        })
        .collect();
    let mut out = Vec::new();
    for node in plan.dag.node_weights() {
        let BufferNode::Compute { op, reads, writes, .. } = node else { continue };
        let label = op.label().to_string();
        if label == "BufferAlloc" || label == "BufferFree" {
            continue;
        }
        let kernel = kernels::codegen_for(op.as_ref())
            .unwrap_or_else(|| panic!("elected op {label} has no codegen row"));
        let ctx = kernels::CodegenCtx {
            operand_dims: reads.iter().map(|id| geometry[id].0.clone()).collect(),
            operand_dtypes: reads.iter().map(|id| geometry[id].1).collect(),
            dest_dims: writes.iter().map(|id| geometry[id].0.clone()).collect(),
            dest_dtypes: writes.iter().map(|id| geometry[id].1).collect(),
            composed_access: reads.iter().map(|_| None).collect(),
        };
        for (i, launch) in (kernel.codegen)(op.as_ref(), &ctx)
            .unwrap_or_else(|e| panic!("codegen for {label}: {e}"))
            .into_iter()
            .enumerate()
        {
            out.push((format!("{label}#{i}"), launch.source));
        }
    }
    out
}

fn searched_plan(
    build: impl FnOnce(
        &mut luminal::graph::Graph,
    ) -> FxHashMap<luminal::prelude::petgraph::graph::NodeIndex, luminal::buffer_tensor_ir::TypedBuffer>,
) -> BufferIrGraph {
    let mut cx = luminal::graph::Graph::new();
    let data = build(&mut cx);
    let mut rt = CudaRuntime::load(&cx).expect("load");
    let outcome = rt
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("search under the CUDA allow list");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    rt.plan().expect("plan loaded").clone()
}

fn representative_plans() -> Vec<(&'static str, BufferIrGraph)> {
    vec![
        ("elementwise", searched_plan(|cx| {
            let a = cx.tensor((2usize, 3usize));
            let b = cx.tensor((2usize, 3usize));
            let _ = ((a + b) * a).output();
            [
                (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
                (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
            ]
            .into_iter()
            .collect()
        })),
        ("matmul", searched_plan(|cx| {
            let x = cx.tensor((4usize, 8usize));
            let w = cx.tensor((8usize, 3usize));
            let _ = x.matmul(w).output();
            [
                (x.id, vec![0.5f32; 32].into()),
                (w.id, vec![0.25f32; 24].into()),
            ]
            .into_iter()
            .collect()
        })),
        ("mul_sum", searched_plan(|cx| {
            let a = cx.tensor((3usize, 4usize));
            let b = cx.tensor((3usize, 4usize));
            let _ = (a * b).sum(1).output();
            [
                (a.id, vec![1.0f32; 12].into()),
                (b.id, vec![2.0f32; 12].into()),
            ]
            .into_iter()
            .collect()
        })),
    ]
}

/// The Phase-3 device path: geometry from the node's own descriptors.
/// The third tuple slot records whether the node read through a fold
/// (any operand carrying composed access) — the Phase-5 restatement
/// keys on it.
fn sources_via_descriptors(plan: &BufferIrGraph) -> Vec<(String, String, bool)> {
    let mut out = Vec::new();
    for node in plan.dag.node_weights() {
        let BufferNode::Compute { op, reads, writes, operand_info, result_info, .. } = node
        else {
            continue;
        };
        let label = op.label().to_string();
        if label == "BufferAlloc" || label == "BufferFree" {
            continue;
        }
        assert_eq!(operand_info.len(), reads.len(), "{label}: operand descriptors parallel reads");
        assert_eq!(result_info.len(), writes.len(), "{label}: result descriptors parallel writes");
        // Phase 5: composed access is now LEGAL here — the view op is
        // electable, folded views hand their consumers the access. The
        // old zero-behavior assert (composed_access always None) died
        // with the premise; the caller now pins where divergence from
        // the buffer-table route is required vs forbidden.
        let folded = operand_info.iter().any(|slot| slot.composed_access.is_some());
        let kernel = kernels::codegen_for(op.as_ref())
            .unwrap_or_else(|| panic!("elected op {label} has no codegen row"));
        let ctx = kernels::CodegenCtx::from_descriptors(&label, operand_info, result_info)
            .unwrap_or_else(|e| panic!("descriptor ctx for {label}: {e}"));
        for (i, launch) in (kernel.codegen)(op.as_ref(), &ctx)
            .unwrap_or_else(|e| panic!("codegen for {label}: {e}"))
            .into_iter()
            .enumerate()
        {
            out.push((format!("{label}#{i}"), launch.source, folded));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// M4 Phase 4: strided READS through synthetic ComposedAccess descriptors —
// string-level gates, host-side (no device). Each test builds a CodegenCtx
// through `from_descriptors` (the only codegen path) and asserts the
// generated source contains the exact index expressions and per-axis
// bounds traps; the flat `a[i]` fast path must stay byte-identical.
// ---------------------------------------------------------------------------

mod strided {
    use luminal::buffer_tensor_ir::BufferTensorIrOp;
    use luminal::bufferize::{AccessHop, BufferId, ComposedAccess, SlotDescriptor};
    use luminal::dtype::PlanDtype;
    use luminal::index_expr::IotaExpr;
    use luminal_cuda_lite::{kernels, ops};

    fn slot(dims: Vec<i64>, access: Option<ComposedAccess>) -> SlotDescriptor {
        SlotDescriptor {
            value: luminal::prelude::egraph_serialize::ClassId::from("val$synthetic"),
            buffer: BufferId::Allocated(0),
            dims: Some(dims),
            element_bits: Some(32),
            dtype: Some(PlanDtype::F32),
            composed_access: access,
        }
    }

    fn one_hop(entries: Option<Vec<IotaExpr>>, parent_dims: Vec<i64>) -> ComposedAccess {
        ComposedAccess { hops: vec![AccessHop { entries, parent_dims: Some(parent_dims) }] }
    }

    /// Generate the single kernel source for `op` with the given
    /// descriptors, through the table row (the real dispatch path).
    fn generate(
        op: &dyn BufferTensorIrOp,
        operand_info: &[SlotDescriptor],
        result_info: &[SlotDescriptor],
    ) -> String {
        let ctx = kernels::CodegenCtx::from_descriptors(op.label(), operand_info, result_info)
            .expect("descriptor ctx builds");
        let row = kernels::codegen_for(op).expect("codegen row");
        let launches = (row.codegen)(op, &ctx).expect("codegen succeeds");
        assert_eq!(launches.len(), 1, "single-launch op");
        launches.into_iter().next().unwrap().source
    }

    fn assert_contains(source: &str, needles: &[&str]) {
        for needle in needles {
            assert!(
                source.contains(needle),
                "generated source missing `{needle}`:\n{source}"
            );
        }
    }

    /// Transpose: out [3,2] reading parent [2,3] at (c1, c0), through
    /// the Copy row's unary template.
    #[test]
    fn transpose_read_indexes_the_parent_at_swapped_coords() {
        let access = one_hop(
            Some(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)]), // parent axis 0 ← c1, axis 1 ← c0
            vec![2, 3],
        );
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot(vec![3, 2], Some(access)), slot(vec![3, 2], None)],
            &[slot(vec![3, 2], None)],
        );
        assert_contains(
            &source,
            &[
                // out-coordinate prelude over [3,2]
                "long long c1 = (long long)(rem % 2ULL); rem /= 2ULL;",
                "long long c0 = (long long)(rem % 3ULL); rem /= 3ULL;",
                // hop 0 entries + per-axis bounds traps
                "long long a_h0_0 = c1;",
                "if (a_h0_0 < 0 || a_h0_0 >= 2LL) __trap();",
                "long long a_h0_1 = c0;",
                "if (a_h0_1 < 0 || a_h0_1 >= 3LL) __trap();",
                // flat index over the parent's row-major strides [3,1]
                "long long a_idx = a_h0_0 * 3LL + a_h0_1 * 1LL;",
                "out[i] = a[a_idx];",
            ],
        );
        assert!(!source.contains("a[i]"), "flat read must be rewritten:\n{source}");
    }

    /// Zero-base slice with pitch > cols: out [4,5] over parent [4,8]
    /// (identity coords, larger row pitch), on one operand of a binary
    /// add — the other operand stays flat `b[i]`.
    #[test]
    fn pitched_slice_read_on_one_binary_operand_keeps_the_other_flat() {
        let access = one_hop(
            Some(vec![IotaExpr::Coord(1), IotaExpr::Coord(0)]), // parent axis 0 ← c0, axis 1 ← c1
            vec![4, 8],
        );
        let op = ops::add::AddFunctionalDps;
        let source = generate(
            &op,
            &[
                slot(vec![4, 5], Some(access)),
                slot(vec![4, 5], None),
                slot(vec![4, 5], None),
            ],
            &[slot(vec![4, 5], None)],
        );
        assert_contains(
            &source,
            &[
                "long long a_h0_0 = c0;",
                "if (a_h0_0 < 0 || a_h0_0 >= 4LL) __trap();",
                "long long a_h0_1 = c1;",
                "if (a_h0_1 < 0 || a_h0_1 >= 8LL) __trap();",
                // pitch 8, not the value's 5
                "long long a_idx = a_h0_0 * 8LL + a_h0_1 * 1LL;",
                "out[i] = a[a_idx] + b[i];",
            ],
        );
    }

    /// Broadcast-shaped map: out [2,3] reading parent [1,3] with a Lit 0
    /// entry on the broadcast axis.
    #[test]
    fn broadcast_read_pins_the_broadcast_axis_to_zero() {
        let access = one_hop(
            Some(vec![IotaExpr::Lit(0), IotaExpr::Coord(0)]), // parent axis 0 ← 0, axis 1 ← c1
            vec![1, 3],
        );
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot(vec![2, 3], Some(access)), slot(vec![2, 3], None)],
            &[slot(vec![2, 3], None)],
        );
        assert_contains(
            &source,
            &[
                "long long a_h0_0 = 0LL;",
                "if (a_h0_0 < 0 || a_h0_0 >= 1LL) __trap();",
                "long long a_h0_1 = c1;",
                "if (a_h0_1 < 0 || a_h0_1 >= 3LL) __trap();",
                "long long a_idx = a_h0_0 * 3LL + a_h0_1 * 1LL;",
                "out[i] = a[a_idx];",
            ],
        );
    }

    /// Two-hop chain, composed at codegen time: hop 0 transposes out
    /// [3,2] into [2,3] coordinates; hop 1 maps those into parent [4,3]
    /// with a +1 row offset. Hop 1's entries must be evaluated at hop
    /// 0's OUTPUTS (`a_h0_*`), never at the out coords.
    #[test]
    fn two_hop_chain_feeds_hop0_outputs_into_hop1() {
        let access = ComposedAccess {
            hops: vec![
                AccessHop {
                    entries: Some(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)]),
                    parent_dims: Some(vec![2, 3]),
                },
                AccessHop {
                    // parent axis 0 ← hop0 coord c0 (= a_h0_0) + 1; axis 1 ← hop0 c1 (= a_h0_1)
                    entries: Some(vec![
                        IotaExpr::Add(Box::new(IotaExpr::Coord(1)), Box::new(IotaExpr::Lit(1))),
                        IotaExpr::Coord(0),
                    ]),
                    parent_dims: Some(vec![4, 3]),
                },
            ],
        };
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot(vec![3, 2], Some(access)), slot(vec![3, 2], None)],
            &[slot(vec![3, 2], None)],
        );
        assert_contains(
            &source,
            &[
                "long long a_h0_0 = c1;",
                "long long a_h0_1 = c0;",
                "long long a_h1_0 = (a_h0_0 + 1LL);",
                "if (a_h1_0 < 0 || a_h1_0 >= 4LL) __trap();",
                "long long a_h1_1 = a_h0_1;",
                "if (a_h1_1 < 0 || a_h1_1 >= 3LL) __trap();",
                // the LAST hop's parent is the residence: strides of [4,3]
                "long long a_idx = a_h1_0 * 3LL + a_h1_1 * 1LL;",
                "out[i] = a[a_idx];",
            ],
        );
    }

    /// Reduce over a transposed input: ReduceSum(axis_from_end=0) on a
    /// [2,3] value reading parent [3,2] — the reduced coordinate is the
    /// loop variable, and the read goes through the chain.
    #[test]
    fn reduce_reads_through_the_composed_chain() {
        let access = one_hop(
            Some(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)]), // parent axis 0 ← c1, axis 1 ← c0
            vec![3, 2],
        );
        let op = ops::reduce_sum::ReduceSumDps { axis: 0 };
        let source = generate(
            &op,
            &[slot(vec![2, 3], Some(access)), slot(vec![2], None)],
            &[slot(vec![2], None)],
        );
        assert_contains(
            &source,
            &[
                // c0 (outside the reduced axis) rebuilt before the loop
                "long long c0 = (long long)(rem % 2ULL); rem /= 2ULL;",
                // the reduced coordinate is the loop variable
                "long long c1 = (long long)r;",
                "long long a_h0_0 = c1;",
                "if (a_h0_0 < 0 || a_h0_0 >= 3LL) __trap();",
                "long long a_h0_1 = c0;",
                "if (a_h0_1 < 0 || a_h0_1 >= 2LL) __trap();",
                "long long a_idx = a_h0_0 * 2LL + a_h0_1 * 1LL;",
                "float v = a[a_idx];",
                "acc = acc + v;",
            ],
        );
    }

    /// Cast keeps its conversion around the strided read.
    #[test]
    fn cast_wraps_the_strided_read() {
        let access = one_hop(Some(vec![IotaExpr::Coord(0)]), vec![4]);
        let op = ops::cast::CastDps;
        let mut operand = slot(vec![4], Some(access));
        operand.dtype = Some(PlanDtype::F32);
        let mut dest = slot(vec![4], None);
        dest.dtype = Some(PlanDtype::Int);
        let source =
            generate(&op, &[operand, dest.clone()], &[dest]);
        assert_contains(&source, &["out[i] = (int)a[a_idx];"]);
    }

    /// `entries: None` on ANY hop is a loud codegen bail, never identity.
    #[test]
    fn unparsed_entries_on_any_hop_refuse_loudly() {
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        // Hop 0 unparsed.
        let access = one_hop(None, vec![2, 3]);
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Copy",
            &[slot(vec![3, 2], Some(access)), slot(vec![3, 2], None)],
            &[slot(vec![3, 2], None)],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("unparsed hop must refuse");
        assert!(err.to_string().contains("beyond the parsed expression subset"), "got: {err}");
        // Hop 1 unparsed behind a good hop 0.
        let access = ComposedAccess {
            hops: vec![
                AccessHop {
                    entries: Some(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)]),
                    parent_dims: Some(vec![2, 3]),
                },
                AccessHop { entries: None, parent_dims: Some(vec![4, 3]) },
            ],
        };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Copy",
            &[slot(vec![3, 2], Some(access)), slot(vec![3, 2], None)],
            &[slot(vec![3, 2], None)],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("unparsed hop 1 must refuse");
        assert!(err.to_string().contains("hop 1"), "got: {err}");
    }

    /// A composed access on a RESULT descriptor refuses at the single
    /// codegen entry point (strided writes are CL-4b).
    #[test]
    fn result_composed_access_refuses_at_from_descriptors() {
        let access = one_hop(Some(vec![IotaExpr::Coord(0)]), vec![4]);
        let err = kernels::CodegenCtx::from_descriptors(
            "ProbeOp",
            &[slot(vec![4], None)],
            &[slot(vec![4], Some(access))],
        )
        .expect_err("result access must refuse");
        assert!(err.to_string().contains("strided writes"), "got: {err}");
    }

    /// A composed access on the DPS dest OPERAND slot refuses in the
    /// template (same CL-4b line).
    #[test]
    fn dest_operand_composed_access_refuses_in_the_template() {
        let access = one_hop(Some(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)]), vec![3, 2]);
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Copy",
            &[slot(vec![3, 2], None), slot(vec![3, 2], Some(access))],
            &[slot(vec![3, 2], None)],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("dest operand access must refuse");
        assert!(err.to_string().contains("strided writes"), "got: {err}");
    }

    /// Train-2B moved the expression-carrying kernels' READ sides into
    /// the lowered set (`tests/composed_read_families.rs` owns those
    /// pins); their WRITE sides stay fail-closed — a composed access on
    /// the DPS dest operand slot refuses with the CL-4b line, never a
    /// silently dense write through a view.
    #[test]
    fn expression_kernel_write_sides_stay_fail_closed() {
        let dest_access = || one_hop(Some(vec![IotaExpr::Coord(0)]), vec![4]);
        let mut coord = slot(vec![4], None);
        coord.dtype = Some(PlanDtype::Int);

        // Gather: dest0 at slot rank+1.
        let op = ops::gather::GatherDps { rank: 1 };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Gather",
            &[slot(vec![4], None), coord.clone(), slot(vec![4], Some(dest_access()))],
            &[slot(vec![4], None)],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("gather dest access must fail closed");
        assert!(err.to_string().contains("strided writes"), "got: {err}");

        // Scatter: dest0 at slot rank+2.
        let op = ops::scatter::ScatterFunctionalDps { rank: 1 };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "ScatterFunctional",
            &[
                slot(vec![4], None),
                slot(vec![2], None),
                coord.clone(),
                slot(vec![4], Some(dest_access())),
            ],
            &[slot(vec![4], None)],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("scatter dest access must fail closed");
        assert!(err.to_string().contains("strided writes"), "got: {err}");

        // Materialize: dest0 at slot 1.
        let op = ops::index_map_apply_materialize::IndexMapApplyMaterializeDps {
            entries: Some(vec![IotaExpr::Coord(0)]),
        };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "IndexMapApplyMaterialize",
            &[slot(vec![4], None), slot(vec![4], Some(dest_access()))],
            &[slot(vec![4], None)],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("materialize dest access must fail closed");
        assert!(err.to_string().contains("strided writes"), "got: {err}");

        // Iota (dest-only signature): still the require-flat refusal.
        let op = ops::iota::IotaDps { expr: Some(IotaExpr::Coord(0)) };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Iota",
            &[slot(vec![4], Some(dest_access()))],
            &[slot(vec![4], None)],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("iota must fail closed");
        assert!(err.to_string().contains("does not lower"), "got: {err}");
    }

    /// Descriptor-less operands still emit the flat `a[i]` kernel
    /// byte-identically to the pre-Phase-4 template (hard pin).
    #[test]
    fn flat_path_is_byte_identical() {
        let op = ops::add::AddFunctionalDps;
        let source = generate(
            &op,
            &[slot(vec![2, 3], None), slot(vec![2, 3], None), slot(vec![2, 3], None)],
            &[slot(vec![2, 3], None)],
        );
        assert_eq!(
            source,
            r#"extern "C" __global__ void k(const float* a, const float* b, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}"#
        );
    }
}

/// The None-dims contract, descriptor-side: symbolic geometry refuses
/// loudly (mirror of the executor's buffer-table bail), never a silent
/// zero-extent kernel.
#[test]
fn descriptor_ctx_bails_loudly_on_missing_numerics() {
    use luminal::bufferize::SlotDescriptor;
    let filled = SlotDescriptor {
        value: luminal::prelude::egraph_serialize::ClassId::from("val$x"),
        buffer: luminal::bufferize::BufferId::Allocated(0),
        dims: Some(vec![2, 3]),
        element_bits: Some(32),
        dtype: Some(PlanDtype::F32),
        composed_access: None,
    };
    let symbolic = SlotDescriptor { dims: None, ..filled.clone() };
    let err = kernels::CodegenCtx::from_descriptors("ProbeOp", &[symbolic], &[filled.clone()])
        .expect_err("symbolic operand dims must refuse");
    assert!(err.to_string().contains("ProbeOp operand lacks geometry"), "got: {err}");
    let untyped = SlotDescriptor { dtype: None, ..filled.clone() };
    let err = kernels::CodegenCtx::from_descriptors("ProbeOp", &[filled.clone()], &[untyped])
        .expect_err("missing dest dtype must refuse");
    assert!(err.to_string().contains("ProbeOp dest lacks dtype"), "got: {err}");
    let ok = kernels::CodegenCtx::from_descriptors("ProbeOp", &[filled.clone()], &[filled])
        .expect("filled descriptors build");
    assert_eq!(ok.operand_dims, vec![vec![2, 3]]);
    assert_eq!(ok.composed_access, vec![None]);
}

/// RE-PINNED ONCE at Phase 5 (view electability). Justification per
/// flip:
///  * NON-FOLDED nodes: the Phase-3 equality pin stands unchanged —
///    descriptor-derived codegen is string-identical to the
///    buffer-table replication.
///  * FOLDED nodes (an operand carries composed access): equality is
///    now IMPOSSIBLE BY DESIGN — the buffer table never knew the
///    folded view's map, which is exactly the Phase-3 bug class the
///    descriptors were built to fix. The pin flips to REQUIRED
///    DIVERGENCE: the descriptor route must emit a different (strided
///    read-through) source than the flat replication. The matmul
///    fixture flips from all-equal to folded (its broadcast/permute
///    movement now folds); elementwise and mul_sum stay all-equal.
#[test]
fn codegen_strings_via_descriptors_match_the_buffer_table() {
    let mut folded_seen = 0usize;
    for (name, plan) in representative_plans() {
        let via_table = sources_via_buffer_table(&plan);
        let via_descriptors = sources_via_descriptors(&plan);
        assert!(!via_table.is_empty(), "{name}: no compute kernels generated");
        assert_eq!(
            via_table.len(),
            via_descriptors.len(),
            "{name}: both routes generate the same kernel sequence"
        );
        for ((t_label, t_source), (d_label, d_source, folded)) in
            via_table.iter().zip(&via_descriptors)
        {
            assert_eq!(t_label, d_label, "{name}: kernel order agrees between routes");
            if *folded {
                folded_seen += 1;
                assert_ne!(
                    t_source, d_source,
                    "{name}/{d_label}: a folded operand must change the generated \
                     read (the buffer table cannot know the composed access)"
                );
            } else {
                assert_eq!(
                    t_source, d_source,
                    "{name}/{d_label}: descriptor-derived codegen must be \
                     string-identical to the buffer table on non-folded nodes"
                );
            }
        }
        if let Ok(dir) = std::env::var("CODEGEN_DUMP_DIR") {
            let dir = std::path::Path::new(&dir);
            std::fs::create_dir_all(dir).expect("dump dir");
            for (i, (label, source, _)) in via_descriptors.iter().enumerate() {
                let file = dir.join(format!("{name}_{i:02}_{}.cu", label.replace('#', "_")));
                std::fs::write(file, source).expect("dump write");
            }
        }
        let folded_here = via_descriptors.iter().filter(|(_, _, f)| *f).count();
        println!(
            "[{name}] {} kernels: {} identical via both paths, {} folded (divergence required)",
            via_table.len(),
            via_table.len() - folded_here,
            folded_here
        );
    }
    assert!(
        folded_seen > 0,
        "Phase 5: at least one representative plan must fold a view \
         (the matmul fixture's movement is foldable)"
    );
}
