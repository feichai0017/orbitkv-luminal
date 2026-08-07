use luminal::{
    egglog_utils::run_egglog,
    op::{IntoEgglogOp, Runtime},
};

use crate::runtime::CudaRuntime;

fn qwen_node_count(program: &str, root: &str) -> usize {
    let mut ops = <CudaRuntime as Runtime>::Ops::into_vec();
    ops.extend(<luminal::hlir::HLIROps as IntoEgglogOp>::into_vec());
    let egraph = run_egglog(program, root, &ops, true).expect("egglog saturation failed");
    egraph
        .enodes
        .values()
        .filter(|(label, _)| label == "Qwen3Moe")
        .count()
}

fn qwen_competing_kernel_sum_count(program: &str, root: &str) -> usize {
    let mut ops = <CudaRuntime as Runtime>::Ops::into_vec();
    ops.extend(<luminal::hlir::HLIROps as IntoEgglogOp>::into_vec());
    let egraph = run_egglog(program, root, &ops, true).expect("egglog saturation failed");

    let qwen_kind_classes = egraph
        .eclasses
        .iter()
        .filter_map(|(class, (_, nodes))| {
            nodes
                .iter()
                .any(|node| egraph.enodes[node].0 == "Qwen3Moe")
                .then_some(class)
        })
        .collect::<std::collections::HashSet<_>>();

    egraph
        .eclasses
        .values()
        .filter(|(_, ir_nodes)| {
            let contains_qwen = ir_nodes.iter().any(|node| {
                let (label, children) = &egraph.enodes[node];
                label == "Op"
                    && children
                        .first()
                        .is_some_and(|kind| qwen_kind_classes.contains(kind))
            });
            let contains_kernel_sum = ir_nodes.iter().any(|node| {
                let (label, children) = &egraph.enodes[node];
                label == "Op"
                    && children.first().is_some_and(|kind| {
                        egraph.eclasses[kind]
                            .1
                            .iter()
                            .any(|kind_node| egraph.enodes[kind_node].0 == "KernelSum")
                    })
            });
            contains_qwen && contains_kernel_sum
        })
        .count()
}

fn rewritten_dtype(program: &str, fused_output: &str) -> String {
    // A top-level `(let probe (dtype output))` performs the table lookup while
    // parsing, before the rewrite schedule has populated `dtype(output)`. Use
    // a final-phase rule instead: it observes the table after all dtype_prop
    // saturation has finished, then unions a typed probe into the rooted
    // sentinel e-class. Bool is only the always-present sentinel; the floating
    // child records the final observation.
    let dtype_root = "qwen3_moe_test_dtype_root";
    let program = format!(
        r#"{program}
(datatype Qwen3MoeTestDType (Qwen3MoeTestDTypeValue DType))
(let {dtype_root} (Qwen3MoeTestDTypeValue (Bool)))
(rule
    ((= ?observed_dtype (dtype {fused_output})))
    ((union {dtype_root} (Qwen3MoeTestDTypeValue ?observed_dtype)))
    :name "observe Qwen3Moe rewritten dtype"
    :ruleset base_cleanup)
"#
    );
    let mut ops = <CudaRuntime as Runtime>::Ops::into_vec();
    ops.extend(<luminal::hlir::HLIROps as IntoEgglogOp>::into_vec());
    let egraph = run_egglog(&program, dtype_root, &ops, true).expect("egglog saturation failed");
    let root = egraph.roots.first().expect("missing dtype root");
    let (_, nodes) = egraph.eclasses.get(root).expect("missing dtype e-class");
    nodes
        .iter()
        .filter_map(|node| {
            let (label, children) = &egraph.enodes[node];
            (label == "Qwen3MoeTestDTypeValue").then(|| &children[0])
        })
        .flat_map(|dtype_class| &egraph.eclasses[dtype_class].1)
        .map(|dtype_node| egraph.enodes[dtype_node].0.as_str())
        .find(|label| matches!(*label, "F16" | "Bf16" | "F32"))
        .expect("dtype root did not contain a floating dtype")
        .to_string()
}

/// The fixture is the real symbolic-token Hugging Face PT2 topology for one
/// Qwen/Qwen3-30B-A3B `Qwen3MoeSparseMoeBlock`, not a handwritten Rust graph.
/// This only saturates the e-graph, so it requires neither CUDA nor the .so.
#[test]
fn qwen3_moe_rule_fires_on_huggingface_pt2_graph() {
    let program = include_str!("fixtures/qwen3_moe_torch_compile_bf16_dynamic.egg");
    let root = include_str!("fixtures/qwen3_moe_torch_compile_bf16_dynamic.root").trim();
    assert_eq!(
        qwen_node_count(program, root),
        1,
        "expected exactly one fused Qwen3Moe alternative for one HF sparse-MoE block"
    );
}

#[test]
fn qwen3_moe_commits_lowered_expert_reduction() {
    for (program, root) in [
        (
            include_str!("fixtures/qwen3_moe_torch_compile_bf16_dynamic.egg"),
            include_str!("fixtures/qwen3_moe_torch_compile_bf16_dynamic.root").trim(),
        ),
        (
            include_str!("fixtures/qwen3_moe_torch_compile_f16_dynamic.egg"),
            include_str!("fixtures/qwen3_moe_torch_compile_f16_dynamic.root").trim(),
        ),
    ] {
        assert_eq!(
            qwen_competing_kernel_sum_count(program, root),
            0,
            "Qwen3Moe output retained a selectable lowered KernelSum fallback"
        );
    }
}

/// Full-model compilation can elide the isolated block's trailing zero-add
/// reshape and feed the flat expert reduction directly to the residual path.
/// The fusion boundary must therefore be the semantic top-k reduction, not a
/// view representation introduced only when the block is exported alone.
#[test]
fn qwen3_moe_rule_does_not_require_standalone_output_reshape() {
    let fixture = include_str!("fixtures/qwen3_moe_torch_compile_bf16_dynamic.egg");
    let mut program = fixture
        .lines()
        .take_while(|line| !line.starts_with("(let t322 "))
        .collect::<Vec<_>>()
        .join("\n");
    program.push_str("\n(let t322 (Output t321 321 false))\n");

    assert_eq!(
        qwen_node_count(&program, "t322"),
        1,
        "expected fusion at the MoE reduction without the standalone reshape"
    );
}

#[test]
fn qwen3_moe_rule_accepts_fp16_model_dtype() {
    let program = include_str!("fixtures/qwen3_moe_torch_compile_f16_dynamic.egg");
    let root = include_str!("fixtures/qwen3_moe_torch_compile_f16_dynamic.root").trim();
    assert_eq!(qwen_node_count(program, root), 1);
}

/// The original semantic reduction accumulates in F32, while the fused C ABI
/// writes model-width storage. Exercise the complete Egglog schedule and
/// verify that the dtype fixed point retains the HostOp's ABI dtype after the
/// union, rather than merely checking that a one-shot `set` exists in source.
#[test]
fn qwen3_moe_dtype_propagation_preserves_fused_storage_width() {
    assert_eq!(
        rewritten_dtype(
            include_str!("fixtures/qwen3_moe_torch_compile_bf16_dynamic.egg"),
            "t323",
        ),
        "Bf16"
    );
    assert_eq!(
        rewritten_dtype(
            include_str!("fixtures/qwen3_moe_torch_compile_f16_dynamic.egg"),
            "t325",
        ),
        "F16"
    );
}

/// Egglog's `dtype(IR)` table is separate from the dtype field stored inside
/// Qwen3Moe. The matched Hugging Face reduction can accumulate in F32, but the
/// fused C ABI writes model-width F16/BF16. Stamp the unioned e-class after the
/// union so downstream kernels allocate and access the same storage width.
#[test]
fn qwen3_moe_rewrite_stamps_the_fused_abi_dtype_after_union() {
    let rewrite = include_str!("../host/qwen3_moe/qwen3_moe_rewrite.egg");
    assert_eq!(
        rewrite.matches("(set (dtype ?qwen3_moe) (Bf16))").count(),
        2,
        "dynamic and token-1 BF16 rules must stamp BF16 storage"
    );
    assert_eq!(
        rewrite.matches("(set (dtype ?qwen3_moe) (F16))").count(),
        2,
        "dynamic and token-1 FP16 rules must stamp FP16 storage"
    );
    for action in rewrite.split("(let ?qwen3_moe").skip(1) {
        let union = action.find("(union ").expect("missing Qwen3Moe union");
        let delete = action
            .find("(delete (dtype ?qwen3_moe))")
            .expect("missing stale Qwen3Moe dtype deletion");
        let dtype = action
            .find("(set (dtype ?qwen3_moe)")
            .expect("missing Qwen3Moe dtype stamp");
        assert!(
            union < delete && delete < dtype,
            "stale dtype must be deleted after union and before the ABI stamp"
        );
    }
}

#[test]
fn qwen3_moe_rule_fires_on_bf16_token1_decode_graph() {
    let program = include_str!("fixtures/qwen3_moe_torch_compile_bf16_token1.egg");
    let root = include_str!("fixtures/qwen3_moe_torch_compile_bf16_token1.root").trim();
    assert_eq!(qwen_node_count(program, root), 1);
}

#[test]
fn qwen3_moe_rule_fires_on_fp16_token1_decode_graph() {
    let program = include_str!("fixtures/qwen3_moe_torch_compile_f16_token1.egg");
    let root = include_str!("fixtures/qwen3_moe_torch_compile_f16_token1.root").trim();
    assert_eq!(qwen_node_count(program, root), 1);
}

/// Keep the ABI/rewrite boundary explicit: the fused node consumes router
/// logits, while the router projection remains available for cuBLASLt.
#[test]
fn qwen3_moe_rewrite_keeps_router_projection_outside() {
    let rewrite = include_str!("../host/qwen3_moe/qwen3_moe_rewrite.egg");
    let action = rewrite
        .split_once("(let ?qwen3_moe")
        .map(|(_, action)| action)
        .expect("missing Qwen3Moe action");

    assert!(action.contains("?router_logits"));
    assert!(action.contains("?gate_up_proj"));
    assert!(action.contains("?down_proj"));
    assert_eq!(rewrite.matches("(let ?qwen3_moe").count(), 4);
    assert_eq!(rewrite.matches("(subsume (Op (Sum").count(), 4);
    assert!(!action.contains("gate.weight"));
    assert!(
        !rewrite
            .lines()
            .any(|line| line.starts_with("        (= ?t4 "))
    );
    assert!(
        !rewrite
            .lines()
            .any(|line| line.starts_with("        (= ?t5 "))
    );
}
