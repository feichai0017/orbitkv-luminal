//! M4 Phase 3 zero-behavior pin: CUDA codegen strings are IDENTICAL
//! whether `CodegenCtx` geometry comes from the shared buffer table
//! (the pre-Phase-3 device path) or from the plan node's own
//! `SlotDescriptor`s (the Phase-3 device path). No views are electable
//! on real backends yet, so descriptors equal buffer-table dims today —
//! this test pins that equality string-for-string on representative
//! searched plans, host-side (no device needed).
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
fn sources_via_descriptors(plan: &BufferIrGraph) -> Vec<(String, String)> {
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
        // Zero-behavior pin's premise, checked explicitly: no view is
        // electable on this backend, so no slot carries composed access.
        for slot in operand_info {
            assert!(
                slot.composed_access.is_none(),
                "{label}: unexpected composed access on a real-backend plan"
            );
        }
        let kernel = kernels::codegen_for(op.as_ref())
            .unwrap_or_else(|| panic!("elected op {label} has no codegen row"));
        let ctx = kernels::CodegenCtx::from_descriptors(&label, operand_info, result_info)
            .unwrap_or_else(|e| panic!("descriptor ctx for {label}: {e}"));
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

#[test]
fn codegen_strings_via_descriptors_match_the_buffer_table() {
    for (name, plan) in representative_plans() {
        let via_table = sources_via_buffer_table(&plan);
        let via_descriptors = sources_via_descriptors(&plan);
        assert!(!via_table.is_empty(), "{name}: no compute kernels generated");
        assert_eq!(
            via_table, via_descriptors,
            "{name}: descriptor-derived codegen must be string-identical to the buffer table"
        );
        if let Ok(dir) = std::env::var("CODEGEN_DUMP_DIR") {
            let dir = std::path::Path::new(&dir);
            std::fs::create_dir_all(dir).expect("dump dir");
            for (i, (label, source)) in via_descriptors.iter().enumerate() {
                let file = dir.join(format!("{name}_{i:02}_{}.cu", label.replace('#', "_")));
                std::fs::write(file, source).expect("dump write");
            }
        }
        println!("[{name}] {} kernels, string-identical via both paths", via_table.len());
    }
}
