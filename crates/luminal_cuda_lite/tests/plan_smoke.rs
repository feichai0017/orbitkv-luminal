//! CL-1 plan-layer smoke: everything up to the device boundary runs on
//! any host — load, bind, search under the CUDA allow list, plan
//! inspection, codegen for every elected compute node — and `execute`
//! without the `device` feature refuses loudly.

use luminal::bufferize::BufferNode;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::{kernels, CudaRuntime};

/// CUDA-lite's inventory, on its own terms.
///
/// This deliberately does NOT compare against the reference runtime. These
/// are different runtimes with different jobs — the reference is
/// canonical-layout-only and out-of-place; CUDA-lite admits views and
/// in-place ties at CL-4 — and their op sets are expected to diverge, in
/// both directions, as a matter of course. A cross-runtime subset assertion
/// would just be a tripwire that fires on ordinary progress.
#[test]
fn allow_list_claims_the_expression_carrying_ops() {
    let cuda = CudaRuntime::allow_list();
    assert!(!cuda.is_empty(), "CUDA claims nothing");
    // CL-1b: the expression-carrying ops are claimed now.
    for present in [
        "Iota",
        "Gather",
        "ScatterFunctional",
        "IndexMapApplyMaterialize",
    ] {
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
        let err = rt
            .execute()
            .expect_err("execute must refuse without a device");
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
    let ctx = kernels::CodegenCtx {
        operand_dims: vec![vec![2, 3], vec![2, 3], vec![2, 3]],
        operand_dtypes: vec![PlanDtype::F32, PlanDtype::F32, PlanDtype::F32],
        dest_dims: vec![vec![2, 3]],
        dest_dtypes: vec![PlanDtype::F32],
    };
    let add = luminal_cuda_lite::ops::add::AddFunctionalDps;
    let kernel = kernels::codegen_for(&add).expect("add has a row");
    let launches = (kernel.codegen)(&add, &ctx).expect("codegen");
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].n, 6);
    assert!(launches[0].source.contains("__global__ void k("));
    assert!(launches[0].source.contains("a[i] + b[i]"));
}
