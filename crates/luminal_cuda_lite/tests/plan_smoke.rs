//! CL-1 plan-layer smoke: everything up to the device boundary runs on
//! any host — load, bind, search under the CUDA allow list, plan
//! inspection, codegen for every elected compute node — and `execute`
//! without the `device` feature refuses loudly.

use luminal::bufferize::BufferNode;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::{kernels, CudaRuntime};

/// The claim-set pin, RESTATED for M4 Phase 5. The old pin ("every CUDA
/// claim is in the reference inventory") assumed one claim class; the
/// allow list now derives TWO, and only one of them can be pinned
/// against the reference:
///
///  * KERNEL-BEARING claims (a codegen row exists) must stay a subset
///    of the reference inventory — every op the device executes has a
///    host reference to differential against.
///  * PLAN-TRANSPARENT claims (declared effects prove the planner folds
///    them; see `luminal_cuda_lite::plan_transparent`) are asserted
///    separately and are deliberately NOT in the reference runtime's
///    inventory: per ruling aff22598 the reference runtime is
///    permanently materialize-only, so its allow list excludes the view
///    op by design while CUDA-lite claims it. A subset pin over the
///    whole claim set would therefore fail on exactly the op Phase 5
///    exists to admit.
#[test]
fn kernel_bearing_claims_subset_reference_and_transparent_class_asserted() {
    let reference = luminal_reference::reference_allow_list();
    let cuda = CudaRuntime::allow_list();
    assert!(!cuda.is_empty(), "CUDA claims nothing");

    // Partition the claim set by the SAME derivations the allow list
    // used: a codegen row for the constructor's label = kernel-bearing;
    // prototype effects = plan-transparent. No name lists.
    let kernel_labels: Vec<&'static str> =
        kernels::cuda_kernels().iter().map(|k| k.label).collect();
    let registry = luminal_cuda_lite::ops::cuda_registry();
    let mut transparent_claims = 0usize;
    for op in &cuda {
        let stripped = op.trim_start_matches("LayoutTensorOp");
        let kernel_bearing = kernel_labels
            .iter()
            .any(|l| stripped == *l || stripped.trim_end_matches("Generic") == *l);
        if kernel_bearing {
            assert!(
                reference.contains(op),
                "kernel-bearing claim {op} not in the reference inventory"
            );
        } else {
            // Claimable without a kernel ONLY through the transparent
            // class: re-derive it from the registered prototype.
            let entry = registry
                .iter()
                .find(|e| e.matcher.egglog_constructor() == *op)
                .unwrap_or_else(|| panic!("claim {op} has no registered matcher"));
            assert!(
                luminal_cuda_lite::plan_transparent(entry.prototype.as_ref()),
                "{op} has neither a codegen row nor plan-transparent effects"
            );
            transparent_claims += 1;
        }
    }
    assert!(
        transparent_claims > 0,
        "Phase 5: the transparent class must be non-empty (the view op is electable)"
    );
    // And the transparent class is exactly what the reference refuses:
    // none of its members are in the materialize-only reference inventory.
    for entry in &registry {
        if luminal_cuda_lite::plan_transparent(entry.prototype.as_ref()) {
            assert!(
                !reference.contains(&entry.matcher.egglog_constructor()),
                "{} is plan-transparent yet the reference (materialize-only, \
                 ruling aff22598) claims it — the pin's premise changed",
                entry.matcher.egglog_constructor()
            );
        }
    }
    // CL-1b: the expression-carrying ops are claimed now.
    for present in ["Iota", "Gather", "ScatterFunctional", "IndexMapApplyMaterialize"] {
        assert!(
            cuda.iter().any(|op| op.contains(present)),
            "{present} missing from the CL-1b claim set"
        );
    }
}

#[test]
fn search_produces_a_codegen_complete_plan() {
    let mut cx = luminal::graph::Graph::new();
    let a = cx.tensor((2usize, 3usize));
    let b = cx.tensor((2usize, 3usize));
    let _out = ((a + b) * a).output();

    let mut rt = CudaRuntime::load(&cx).expect("load");
    let data: FxHashMap<_, _> = [
        (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
        (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
    ]
    .into_iter()
    .collect();
    let outcome = rt
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("search under the CUDA allow list");
    assert!(outcome.plans_profiled > 0, "no plans profiled");

    // Every elected compute node must have a codegen row — the allow
    // list promised only what the table generates.
    let plan = rt.plan().expect("plan loaded");
    let mut computes = 0usize;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute { op, .. } = node {
            computes += 1;
            let label = op.label();
            if label == "BufferAlloc" || label == "BufferFree" {
                continue;
            }
            assert!(
                kernels::codegen_for(op.as_ref()).is_some(),
                "elected op {label} has no codegen row"
            );
        }
    }
    assert!(computes > 0, "plan has no compute nodes");

    rt.set_data(a.id, vec![1.0f32, 2., 3., 4., 5., 6.]);
    rt.set_data(b.id, vec![10.0f32, 20., 30., 40., 50., 60.]);
    #[cfg(not(feature = "device"))]
    {
        // Without the device feature, execute refuses loudly.
        let err = rt.execute().expect_err("execute must refuse without a device");
        assert!(
            err.to_string().contains("device"),
            "refusal must name the missing feature: {err}"
        );
    }
    #[cfg(feature = "device")]
    {
        // With a device: NVRTC-compile, launch on the GPU, and match
        // the hand-computed numerics: (a+b)*a.
        rt.execute().expect("device execute");
        let got = rt.get_f32(_out.id).expect("output payload");
        assert_eq!(got, &vec![11.0f32, 44., 99., 176., 275., 396.]);
    }
}

#[test]
fn codegen_emits_wellformed_sources() {
    // String-level check on a representative binary kernel: generate
    // Add over (2,3) f32 and eyeball the load-bearing pieces.
    use luminal::dtype::PlanDtype;
    use luminal::layouts::{
        BitWidthTerm, IntExprTerm, MirrorLayout, RightMajorContiguousElementLayout, ShapeTerm,
    };
    fn rm_layout(dims: &[i64]) -> luminal_cuda_lite::layouts::CudaLayout {
        luminal_cuda_lite::layouts::CudaLayout {
            mirror: MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
                shape: ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d)).collect()),
                width: BitWidthTerm(32),
            }),
            dtype: Some(PlanDtype::F32),
        }
    }
    let ctx = kernels::CodegenCtx {
        operand_dims: vec![vec![2, 3], vec![2, 3], vec![2, 3]],
        operand_dtypes: vec![PlanDtype::F32, PlanDtype::F32, PlanDtype::F32],
        dest_dims: vec![vec![2, 3]],
        dest_dtypes: vec![PlanDtype::F32],
        // The slot layouts ARE the read paths (the hop chain is retired):
        // all three are dense row-major, so every read simplifies to the
        // identity and the body collapses to the pre-Option-B text.
        operand_layouts: vec![rm_layout(&[2, 3]), rm_layout(&[2, 3]), rm_layout(&[2, 3])],
    };
    let add = luminal_cuda_lite::ops::add::AddFunctionalDps;
    let kernel = kernels::codegen_for(&add).expect("add has a row");
    let launches = (kernel.codegen)(&add, &ctx).expect("codegen");
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].n, 6);
    assert!(launches[0].source.contains("__global__ void k("));
    assert!(launches[0].source.contains("a[i] + b[i]"));
}
