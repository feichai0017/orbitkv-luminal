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

/// The PRE-Phase-3 construction, restated for the corrected contract:
/// per-node geometry looked up in the shared BUFFER table by BufferId —
/// which now means the buffer's own carried layout (the RESIDENT's
/// layout), never a plan `dims`/`dtype` field. That is precisely why the
/// route still diverges on folded operands: the residence's layout is
/// the parent's, and the operand wanted the view's.
fn sources_via_buffer_table(plan: &BufferIrGraph<luminal_cuda_lite::CudaLayout>) -> Vec<(String, String)> {
    let geometry: HashMap<BufferId, (Vec<usize>, PlanDtype)> = plan
        .buffers
        .iter()
        .map(|(id, buffer)| {
            let dims = buffer
                .layout
                .mirror
                .literal_extents()
                .expect("plan buffer's layout has literal extents");
            (id.clone(), (dims, buffer.layout.dtype.expect("plan buffer's layout is typed")))
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
            // PROTOTYPE (Option B): the buffer table's layout is the
            // WRITER's (resident) layout — for a folded operand that is
            // the parent's dense layout, so this route stays flat and
            // the folded divergence below is required exactly as before.
            operand_layouts: reads
                .iter()
                .map(|id| plan.buffers[id].layout.clone())
                .collect(),
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
) -> BufferIrGraph<luminal_cuda_lite::CudaLayout> {
    let mut cx = luminal::graph::Graph::new();
    let data = build(&mut cx);
    let mut rt = CudaRuntime::load(&cx).expect("load");
    let outcome = rt
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("search under the CUDA allow list");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    rt.plan().expect("plan loaded").clone()
}

fn representative_plans() -> Vec<(&'static str, BufferIrGraph<luminal_cuda_lite::CudaLayout>)> {
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
fn sources_via_descriptors(plan: &BufferIrGraph<luminal_cuda_lite::CudaLayout>) -> Vec<(String, String, bool)> {
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
        // Option B: the divergence discriminator is the slot LAYOUT —
        // an operand whose own elected layout is not the direct read.
        // (A view whose composed layout IS direct would be a flat read
        // on both routes, correctly.)
        let folded = operand_info.iter().any(|slot| {
            let dims = slot
                .layout
                .mirror
                .literal_extents()
                .expect("elected slot layouts are literal in these fixtures");
            !kernels::layout_is_direct(&slot.layout, &dims)
        });
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
    use luminal::bufferize::{BufferId, SlotDescriptor};
    use luminal::dtype::PlanDtype;
    use luminal::index_expr::IotaExpr;
    use luminal_cuda_lite::{kernels, ops};

    use luminal::layouts::{
        BitWidthTerm, ElementOffsetExpressionLayout, IntExprTerm, MirrorLayout,
        RightMajorContiguousElementLayout, ShapeTerm, StridedElementLayout,
    };
    use luminal_cuda_lite::CudaLayout;

    fn lit(v: i64) -> IntExprTerm {
        IntExprTerm::Lit(v)
    }
    fn coord(axis_from_end: i64) -> IntExprTerm {
        IntExprTerm::Coord { axis_from_end }
    }
    fn mul(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Mul(Box::new(a), Box::new(b))
    }
    fn add(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Add(Box::new(a), Box::new(b))
    }
    fn shape(dims: &[i64]) -> ShapeTerm {
        ShapeTerm(dims.iter().map(|&d| lit(d)).collect())
    }
    fn typed(mirror: MirrorLayout) -> CudaLayout {
        CudaLayout { mirror, dtype: Some(PlanDtype::F32) }
    }
    fn rm_layout(dims: &[i64]) -> CudaLayout {
        typed(MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape: shape(dims),
            width: BitWidthTerm(32),
        }))
    }
    fn strided_layout(dims: &[i64], chain: Vec<IntExprTerm>) -> CudaLayout {
        typed(MirrorLayout::Strided(StridedElementLayout {
            shape: shape(dims),
            chain,
            width: BitWidthTerm(32),
        }))
    }
    fn offset_layout(dims: &[i64], offset: IntExprTerm) -> CudaLayout {
        typed(MirrorLayout::ElementOffset(ElementOffsetExpressionLayout {
            offset,
            shape: shape(dims),
            width: BitWidthTerm(32),
        }))
    }

    /// A slot whose layout is the DIRECT read (row-major over its dims).
    /// EVERYTHING the codegen needs — extents, dtype, read path — comes
    /// from that one layout: the descriptor has no dims/dtype/hop fields
    /// left to fill (corrected contract, 2026-08-31).
    fn slot(dims: Vec<i64>) -> SlotDescriptor<CudaLayout> {
        slot_l(rm_layout(&dims))
    }

    /// PROTOTYPE (Option B): a slot carrying its OWN elected layout —
    /// the one vocabulary every family reads through.
    fn slot_l(layout: CudaLayout) -> SlotDescriptor<CudaLayout> {
        SlotDescriptor {
            value: luminal::prelude::egraph_serialize::ClassId::from("val$synthetic"),
            buffer: BufferId::Allocated(0),
            layout,
        }
    }

    /// The same slot with a different dtype fact on its carried layout
    /// (a RUNTIME-side field: dtype rides `CudaLayout`, never the plan).
    fn slot_dt(dims: Vec<i64>, dtype: PlanDtype) -> SlotDescriptor<CudaLayout> {
        let mut s = slot(dims);
        s.layout.dtype = Some(dtype);
        s
    }

    /// A deliberately NON-DIRECT layout for the fail-closed write-side
    /// pins: any strided form over the slot's own dims will do.
    fn nondirect(dims: &[i64]) -> CudaLayout {
        strided_layout(dims, vec![mul(coord(0), lit(2))])
    }

    /// Generate the single kernel source for `op` with the given
    /// descriptors, through the table row (the real dispatch path).
    fn generate(
        op: &dyn BufferTensorIrOp,
        operand_info: &[SlotDescriptor<CudaLayout>],
        result_info: &[SlotDescriptor<CudaLayout>],
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

    /// Transpose, OPTION B: out [3,2] reading its parent's bytes through
    /// the SLOT'S OWN composed strided layout (shape [3,2], chain
    /// from-end [coord0*3, coord1] — element (i,j) at parent flat
    /// j*3+i), through the Copy row's unary template. NO hop chain is
    /// supplied at all — the layout alone drives the read (the
    /// hop-machinery death demonstration).
    #[test]
    fn transpose_read_indexes_the_parent_at_swapped_coords() {
        let layout = strided_layout(&[3, 2], vec![mul(coord(0), lit(3)), coord(1)]);
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot_l(layout), slot(vec![3, 2])],
            &[slot(vec![3, 2])],
        );
        assert_contains(
            &source,
            &[
                // out-coordinate prelude over [3,2]
                "long long c1 = (long long)(rem % 2ULL); rem /= 2ULL;",
                "long long c0 = (long long)(rem % 3ULL); rem /= 3ULL;",
                // the layout's offset expression, lowered directly
                "long long a_idx = (c1 * 3LL) + c0;",
                // ONE final trap against the strided span (1 + (2-1)*3 + (3-1) = 6)
                "if (a_idx < 0 || a_idx >= (((1LL + ((2LL + -1LL) * 3LL)) + (3LL + -1LL)))) __trap();",
                "out[i] = a[a_idx];",
            ],
        );
        assert!(!source.contains("a[i]"), "flat read must be rewritten:\n{source}");
        assert!(
            !source.contains("a_h0_0"),
            "Option B: no hop variables — the layout drives the read:\n{source}"
        );
    }

    /// Zero-base slice with pitch > cols: out [4,5] over parent [4,8]
    /// (identity coords, larger row pitch), on one operand of a binary
    /// add — the other operand stays flat `b[i]`.
    #[test]
    fn pitched_slice_read_on_one_binary_operand_keeps_the_other_flat() {
        // OPTION B: the slice value's composed layout — shape [4,5],
        // chain from-end [coord0 (stride 1), coord1*8 (the parent's
        // pitch)]. The other operand keeps its direct layout and stays
        // flat `b[i]`.
        let layout = strided_layout(&[4, 5], vec![coord(0), mul(coord(1), lit(8))]);
        let op = ops::add::AddFunctionalDps;
        let source = generate(
            &op,
            &[
                slot_l(layout),
                slot(vec![4, 5]),
                slot(vec![4, 5]),
            ],
            &[slot(vec![4, 5])],
        );
        assert_contains(
            &source,
            &[
                // pitch 8, not the value's 5
                "long long a_idx = c1 + (c0 * 8LL);",
                // span trap: 1 + (5-1) + (4-1)*8 = 29
                "if (a_idx < 0 || a_idx >= (((1LL + (5LL + -1LL)) + ((4LL + -1LL) * 8LL)))) __trap();",
                "out[i] = a[a_idx] + b[i];",
            ],
        );
    }

    /// Broadcast-shaped map: out [2,3] reading parent [1,3] with a Lit 0
    /// entry on the broadcast axis.
    #[test]
    fn broadcast_read_pins_the_broadcast_axis_to_zero() {
        // OPTION B: the broadcast value's composed layout — shape [2,3],
        // chain from-end [coord0 (stride 1), 0 (the dead axis residue)].
        let layout = strided_layout(&[2, 3], vec![coord(0), lit(0)]);
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot_l(layout), slot(vec![2, 3])],
            &[slot(vec![2, 3])],
        );
        assert_contains(
            &source,
            &[
                // the dead axis contributes the literal zero residue
                "long long a_idx = c1 + 0LL;",
                // span trap: 1 + (3-1) + 0 = 3 (the PARENT ROW's reach)
                "if (a_idx < 0 || a_idx >= (((1LL + (3LL + -1LL)) + 0LL))) __trap();",
                "out[i] = a[a_idx];",
            ],
        );
    }

    /// Two folds, OPTION B: the e-graph composes — the slot carries ONE
    /// layout whose offset expression is the whole composition (here the
    /// synthetic composition of a transpose then a +1-row offset into a
    /// [4,3] parent: (c1+1)*3 + c0), spelled as an offset-EXPRESSION
    /// form. Pins the offset-form BOUNDS HONESTY COST: an offset
    /// function does not disclose its reach (`SpanExpr` deliberately
    /// unimplemented), so the ONLY trap left is non-negativity — the
    /// per-hop parent-extent traps of the hop machinery are gone, and
    /// nothing bounds the read from above at this layer.
    #[test]
    fn composed_offset_form_reads_one_expression_and_loses_the_reach_trap() {
        let layout = offset_layout(
            &[3, 2],
            add(mul(add(coord(0), lit(1)), lit(3)), coord(1)),
        );
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot_l(layout), slot(vec![3, 2])],
            &[slot(vec![3, 2])],
        );
        assert_contains(
            &source,
            &[
                "long long a_idx = (((c1 + 1LL) * 3LL) + c0);",
                // the only remaining trap: non-negativity
                "if (a_idx < 0) __trap();",
                "out[i] = a[a_idx];",
            ],
        );
        assert!(
            !source.contains("a_idx >="),
            "offset-form layouts disclose no reach — no upper-bound trap exists:\n{source}"
        );
        assert!(!source.contains("a_h0_0"), "no hop variables:\n{source}");
    }

    /// Reduce over a transposed input, OPTION B: ReduceSum(axis_from_end
    /// = 0) on a [2,3] value whose SLOT LAYOUT is the transpose
    /// composition over a [3,2] parent (element (c0,c1) at parent flat
    /// c1*2 + c0). The reduced coordinate is the loop variable, and the
    /// read is the layout's own offset expression — no hop chain, no
    /// per-hop parent-extent traps, one span trap.
    #[test]
    fn reduce_reads_through_the_slot_layout() {
        // shape [2,3]; from-end coord(0) = c1, coord(1) = c0.
        let layout = strided_layout(&[2, 3], vec![mul(coord(0), lit(2)), coord(1)]);
        let op = ops::reduce_sum::ReduceSumDps { axis: 0 };
        let source = generate(&op, &[slot_l(layout), slot(vec![2])], &[slot(vec![2])]);
        assert_contains(
            &source,
            &[
                // c0 (outside the reduced axis) rebuilt before the loop
                "long long c0 = (long long)(rem % 2ULL); rem /= 2ULL;",
                // the reduced coordinate is the loop variable
                "long long c1 = (long long)r;",
                // the layout's expression, lowered directly
                "long long a_idx = (c1 * 2LL) + c0;",
                "float v = a[a_idx];",
                "acc = acc + v;",
            ],
        );
        assert!(!source.contains("a_h0_0"), "no hop variables:\n{source}");
    }

    /// Cast keeps its conversion around the layout-expression read.
    #[test]
    fn cast_wraps_the_strided_read() {
        // A reversed rank-1 read: chain [-coord + 3] spelled as
        // ((coord0 * -1) + 3) — non-direct, so the expression path runs.
        let layout = strided_layout(&[4], vec![add(mul(coord(0), lit(-1)), lit(3))]);
        let op = ops::cast::CastDps;
        // The dtypes ride the slots' CARRIED LAYOUTS (the runtime's own
        // type), not a descriptor field.
        let operand = slot_l(layout);
        let dest = slot_dt(vec![4], PlanDtype::Int);
        let source = generate(&op, &[operand, dest.clone()], &[dest]);
        assert_contains(&source, &["out[i] = (int)a[a_idx];"]);
    }

    /// OPTION B fail-closed analogues of the old `entries: None` hop
    /// refusals: a layout the lowerer cannot spell numerically bails
    /// loudly — never identity, never a guessed extent.
    ///
    /// NOTE THE MOVED SEAM. A SYMBOLIC domain now refuses one step
    /// EARLIER, at `from_descriptors`, because the slot's extents ARE
    /// its layout's domain (there is no dims field to disagree with).
    /// The domain-mismatch refusal survives only where two DIFFERENT
    /// slots' layouts disagree — here an operand whose domain is not the
    /// destination's, which is what the elementwise template reads
    /// against.
    #[test]
    fn unlowerable_layouts_refuse_loudly() {
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        // Symbolic extent in the layout's own domain — refused at ctx build.
        let layout = typed(MirrorLayout::Strided(StridedElementLayout {
            shape: ShapeTerm(vec![IntExprTerm::Var("n".to_string()), lit(2)]),
            chain: vec![coord(0), mul(coord(1), lit(2))],
            width: BitWidthTerm(32),
        }));
        let err = kernels::CodegenCtx::from_descriptors(
            "Copy",
            &[slot_l(layout), slot(vec![3, 2])],
            &[slot(vec![3, 2])],
        )
        .expect_err("symbolic layout extents must refuse");
        assert!(err.to_string().contains("symbolic layout extents"), "got: {err}");
        // An operand layout whose DOMAIN is not the destination's: the
        // template reads at the dest's coordinates, so this is a real
        // incoherence and refuses in the template, never reinterprets.
        let layout = strided_layout(&[2, 3], vec![coord(0), mul(coord(1), lit(3))]);
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Copy",
            &[slot_l(layout), slot(vec![3, 2])],
            &[slot(vec![3, 2])],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("foreign-domain layout must refuse");
        assert!(
            err.to_string().contains("differ from dest extents"),
            "the template names the incoherence: {err}"
        );
    }

    /// A NON-DIRECT LAYOUT on a RESULT descriptor refuses at the single
    /// codegen entry point (strided writes are CL-4b). Under the
    /// corrected contract this is the ONLY spelling of the refusal —
    /// the hop-carrying variant died with the hop machinery, and this is
    /// a CAPABILITY refusal (the backend lowers no strided write), never
    /// a re-check of an e-graph premise.
    #[test]
    fn result_non_direct_layout_refuses_at_from_descriptors() {
        let t_layout = strided_layout(&[4], vec![mul(coord(0), lit(2))]);
        let err = kernels::CodegenCtx::from_descriptors(
            "ProbeOp",
            &[slot(vec![4])],
            &[slot_l(t_layout)],
        )
        .expect_err("result non-direct layout must refuse");
        assert!(err.to_string().contains("non-direct layout"), "got: {err}");
    }

    /// A non-direct LAYOUT on the DPS dest OPERAND slot refuses in the
    /// template (same CL-4b line, Option B spelling).
    #[test]
    fn dest_operand_non_direct_layout_refuses_in_the_template() {
        let t_layout = strided_layout(&[3, 2], vec![mul(coord(0), lit(3)), coord(1)]);
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Copy",
            &[slot(vec![3, 2]), slot_l(t_layout)],
            &[slot(vec![3, 2])],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("dest operand non-direct layout must refuse");
        assert!(err.to_string().contains("strided writes"), "got: {err}");
    }

    /// Train-2B moved the expression-carrying kernels' READ sides into
    /// the lowered set (`tests/composed_read_families.rs` owns those
    /// pins); their WRITE sides stay fail-closed — a composed access on
    /// the DPS dest operand slot refuses with the CL-4b line, never a
    /// silently dense write through a view.
    #[test]
    fn expression_kernel_write_sides_stay_fail_closed() {
        // The write-side refusal now keys on the dest OPERAND slot's
        // own carried layout being non-direct — the one discriminator.
        let dest_nondirect = || slot_l(nondirect(&[4]));
        let coord = slot_dt(vec![4], PlanDtype::Int);

        // Gather: dest0 at slot rank+1.
        let op = ops::gather::GatherDps { rank: 1 };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Gather",
            &[slot(vec![4]), coord.clone(), dest_nondirect()],
            &[slot(vec![4])],
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
                slot(vec![4]),
                slot(vec![2]),
                coord.clone(),
                dest_nondirect(),
            ],
            &[slot(vec![4])],
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
            &[slot(vec![4]), dest_nondirect()],
            &[slot(vec![4])],
        )
        .expect("ctx builds");
        let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
            .expect_err("materialize dest access must fail closed");
        assert!(err.to_string().contains("strided writes"), "got: {err}");

        // Iota (dest-only signature): still the require-flat refusal.
        let op = ops::iota::IotaDps { expr: Some(IotaExpr::Coord(0)) };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "Iota",
            &[dest_nondirect()],
            &[slot(vec![4])],
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
            &[slot(vec![2, 3]), slot(vec![2, 3]), slot(vec![2, 3])],
            &[slot(vec![2, 3])],
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

/// THE LOUD-BAIL CONTRACT, descriptor-side, RESPELLED (corrected
/// contract, 2026-08-31): the descriptor has no `dims`/`dtype` fields
/// to leave empty any more — every numeric a kernel needs comes from
/// the slot's CARRIED LAYOUT. So the refusals move onto the layout: a
/// symbolic domain and a missing dtype fact both bail loudly, never a
/// silent zero-extent kernel and never a guessed representation.
#[test]
fn descriptor_ctx_bails_loudly_on_unusable_layouts() {
    use luminal::bufferize::SlotDescriptor;
    use luminal::layouts::{
        BitWidthTerm, IntExprTerm, MirrorLayout, RightMajorContiguousElementLayout, ShapeTerm,
    };
    use luminal_cuda_lite::CudaLayout;
    let rm = |shape: ShapeTerm| {
        MirrorLayout::RightMajor(RightMajorContiguousElementLayout { shape, width: BitWidthTerm(32) })
    };
    let lit_shape = ShapeTerm(vec![IntExprTerm::Lit(2), IntExprTerm::Lit(3)]);
    let filled = SlotDescriptor {
        value: luminal::prelude::egraph_serialize::ClassId::from("val$x"),
        buffer: luminal::bufferize::BufferId::Allocated(0),
        layout: CudaLayout { mirror: rm(lit_shape.clone()), dtype: Some(PlanDtype::F32) },
    };
    let symbolic = SlotDescriptor {
        layout: CudaLayout {
            mirror: rm(ShapeTerm(vec![IntExprTerm::Var("n".to_string()), IntExprTerm::Lit(3)])),
            dtype: Some(PlanDtype::F32),
        },
        ..filled.clone()
    };
    let err = kernels::CodegenCtx::from_descriptors("ProbeOp", &[symbolic], &[filled.clone()])
        .expect_err("symbolic layout extents must refuse");
    assert!(err.to_string().contains("symbolic layout extents"), "got: {err}");
    let untyped = SlotDescriptor {
        layout: CudaLayout { mirror: rm(lit_shape), dtype: None },
        ..filled.clone()
    };
    let err = kernels::CodegenCtx::from_descriptors("ProbeOp", &[filled.clone()], &[untyped])
        .expect_err("a missing dtype fact must refuse");
    assert!(err.to_string().contains("carries no dtype fact"), "got: {err}");
    let ok = kernels::CodegenCtx::from_descriptors("ProbeOp", &[filled.clone()], &[filled])
        .expect("filled descriptors build");
    assert_eq!(ok.operand_dims, vec![vec![2, 3]]);
    assert!(ok.non_direct_operand(0).is_none(), "the direct form is the flat fast path");
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
