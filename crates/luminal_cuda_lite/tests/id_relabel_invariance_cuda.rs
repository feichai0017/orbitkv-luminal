//! ID-RELABEL INVARIANCE — the MEASUREMENT, cuda-lite twin (the cuBLASLt
//! MARKER vocabulary and a real `LayoutRenderer`).
//!
//! Same ruling, same harness and same digests as
//! `tests/test_runtime/tests/id_relabel_invariance.rs`; what changes is
//! the vocabulary under test. Here the fixtures are RECORDED MINI
//! PROGRAMS (`mini_conv`, `mini_llama3`, `mini_qwen3` — graphs, shapes
//! and seeds copied from the election rows), the matcher set is
//! `cuda_matchers()` / `cuda_matchers_with_cublaslt()` under the matching
//! allow list, and layouts are rendered by `CudaLayoutRenderer` — which
//! reaches `luminal::layouts`, whose reader sorts the `NodeId`s of a
//! class to choose which same-constructor spelling it reads. That site is
//! one of the reasons this twin exists.
//!
//! It PRINTS, it never asserts: the known id-order dependence sites are
//! exactly what is being surveyed.
//!
//! EXPLICIT RUN ONLY. Every row here is `#[ignore]`d: nine searches per
//! fixture at the harness budget over the mini graphs costs tens of
//! minutes (the cuBLASLt rows are the expensive ones), and the file
//! asserts NOTHING — leaving it in the default suite would buy CI hours
//! of silence. Same reasoning that keeps the render oracle's `oracle_`
//! rows out of `cargo test -p luminal_cuda_lite -- --skip oracle_`.
//!
//! Run it:
//! `cargo test -p luminal_cuda_lite --test id_relabel_invariance_cuda -- \
//!   --ignored --nocapture --test-threads=1`
//! (CPU only — no `device` feature, `StaticProfiler`, seed 0, the harness
//! budget).

use std::collections::HashMap;
use std::fmt::Write as _;

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::dtype::PlanDtype;
use luminal::graph::{Graph, LogicalProgram};
use luminal::implementation_search::{search_implementations_with_runtime, StaticProfiler};
use luminal::layout_ir::OpMatcher;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal::test_support::digest::{extracted_digest, mask_ids, plan_digest};
use luminal::test_support::relabel::relabel_egraph;
use luminal_cuda_lite::layouts::{CudaLayout, CudaLayoutRenderer};
use luminal_cuda_lite::{CudaBindings, CudaRuntime};

/// How many relabelings each fixture is measured against.
const K: u64 = 8;

fn matchers(cublaslt: bool) -> Vec<Box<dyn OpMatcher>> {
    if cublaslt {
        luminal_cuda_lite::ops::cuda_matchers_with_cublaslt()
    } else {
        luminal_cuda_lite::ops::cuda_matchers()
    }
}

fn allow_list(cublaslt: bool) -> Vec<&'static str> {
    if cublaslt {
        CudaRuntime::allow_list_with_cublaslt()
    } else {
        CudaRuntime::allow_list()
    }
}

/// `CudaRuntime::load` → `bind_dyn_range` (every dyn var pinned, sorted,
/// as the election rows do) → saturate → serialize. The e-graph is built
/// ONCE per fixture; the relabelings are drawn from it, so nothing below
/// depends on egglog re-running.
fn saturate(cx: &Graph, cublaslt: bool) -> EGraph {
    let (pre_schedule, input_slots, output_slots, post_checks, _labeled) = cx
        .logical
        .bound_parts(&CudaBindings)
        .unwrap_or_else(|e| panic!("cuda load: {e}"));
    let mut vars: Vec<_> = cx.dyn_map.iter().collect();
    vars.sort();
    let mut binding_seeds = String::new();
    for (var, value) in vars {
        let name: luminal::shape::Symbol = *var;
        let _ = write!(
            binding_seeds,
            "(set (lower-bound-of (IntVar \"{name}\")) (bigint {value}))\n\
             (set (upper-bound-of (IntVar \"{name}\")) (bigint {value}))\n"
        );
    }
    let program = LogicalProgram {
        text: format!(
            "{pre_schedule}{binding_seeds}{}{post_checks}",
            CudaBindings::SCHEDULE
        ),
        input_slots,
        output_slots,
    };
    let full = format!(
        "{}\n\n{}",
        luminal::egglog_snippet::assembled_program_for(&matchers(cublaslt)),
        program.text
    );
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(None, &full)
        .unwrap_or_else(|err| panic!("cuda-lite saturation failed: {err}"));
    egraph
        .serialize(luminal::prelude::egglog::SerializeConfig::default())
        .egraph
}

fn first_diff(base: &str, other: &str) -> Option<String> {
    for (index, (left, right)) in base.lines().zip(other.lines()).enumerate() {
        if left != right {
            return Some(format!("line {index}: base={left:?} relabeled={right:?}"));
        }
    }
    let (base_lines, other_lines) = (base.lines().count(), other.lines().count());
    if base_lines != other_lines {
        return Some(format!(
            "line count differs: base={base_lines} relabeled={other_lines}"
        ));
    }
    None
}

/// The deterministic path: min-cost extraction under the allow list, the
/// DPS rewrite, the REAL rendered-layout table, and bufferization.
fn deterministic(egraph: &EGraph, cublaslt: bool) -> (String, String, String) {
    let allow = allow_list(cublaslt);
    let extracted = luminal::extractor::extract_layout_ir_with_ops_and_matchers(
        egraph,
        Some(allow.as_slice()),
        matchers(cublaslt),
    );
    let graph = match extracted {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            return (
                "<no graph>".into(),
                "<no graph>".into(),
                "<no graph>".into(),
            )
        }
        Err(err) => {
            let text = format!("<extract error: {err:#}>");
            return (text.clone(), text.clone(), text);
        }
    };
    let dps = luminal::dps::dps_rewrite(&graph);
    let mut cache: HashMap<(ClassId, Option<PlanDtype>), CudaLayout> = HashMap::new();
    let built =
        luminal::extractor::rendered_layout_table(egraph, &dps, &CudaLayoutRenderer, &mut cache)
            .and_then(|table| luminal::bufferize::bufferize(&dps, &table));
    match built {
        Ok(plan) => (
            extracted_digest(&graph),
            plan_digest(&plan),
            mask_ids(&plan.summary()),
        ),
        Err(err) => {
            let text = format!("<plan build error: {err:#}>");
            (extracted_digest(&graph), text.clone(), text)
        }
    }
}

/// The seeded search path, at the harness budget on seed 0.
fn searched(
    egraph: &EGraph,
    cx: &Graph,
    pairs: &[(NodeIndex, TypedBuffer)],
    cublaslt: bool,
) -> (String, String) {
    let (pre_schedule, input_slots, output_slots, _post, _labeled) = cx
        .logical
        .bound_parts(&CudaBindings)
        .unwrap_or_else(|e| panic!("cuda load: {e}"));
    let program = LogicalProgram {
        text: pre_schedule,
        input_slots,
        output_slots,
    };
    let data: FxHashMap<NodeIndex, TypedBuffer> = pairs.iter().cloned().collect();
    let outcome = search_implementations_with_runtime(
        egraph,
        &program,
        &data,
        &luminal::test_support::harness_search_options(),
        Some(allow_list(cublaslt)),
        matchers(cublaslt),
        &CudaLayoutRenderer,
        &mut StaticProfiler,
    );
    match outcome {
        Ok(outcome) => (
            format!(
                "plans_profiled={} refusals=[{}]\n{}",
                outcome.plans_profiled,
                outcome.refusal_breakdown.summary(),
                plan_digest(&outcome.best_plan)
            ),
            mask_ids(&outcome.best_plan.summary()),
        ),
        Err(err) => {
            let text = format!("<search refused: {err:#}>");
            (text.clone(), text)
        }
    }
}

fn survey(label: &str, cx: &Graph, pairs: &[(NodeIndex, TypedBuffer)], cublaslt: bool) {
    let egraph = saturate(cx, cublaslt);
    let (base_extract, base_plan, base_summary) = deterministic(&egraph, cublaslt);
    let (base_search, base_search_summary) = searched(&egraph, cx, pairs, cublaslt);

    let mut diffs: Vec<Option<(u64, String)>> = vec![None; 5];
    for seed in 1..=K {
        let relabeled = relabel_egraph(&egraph, seed);
        let (extract, plan, summary) = deterministic(&relabeled, cublaslt);
        let (search, search_summary) = searched(&relabeled, cx, pairs, cublaslt);
        let observed = [
            (&base_extract, extract),
            (&base_plan, plan),
            (&base_search, search),
            (&base_summary, summary),
            (&base_search_summary, search_summary),
        ];
        for (slot, (base, other)) in observed.into_iter().enumerate() {
            if diffs[slot].is_none() {
                diffs[slot] = first_diff(base, &other).map(|detail| (seed, detail));
            }
        }
    }

    let names = [
        "deterministic",
        "plan",
        "search",
        "summary_order",
        "search_summary_order",
    ];
    let verdict = |slot: usize| if diffs[slot].is_some() { "no" } else { "yes" };
    println!(
        "RELABEL {label} K={K} deterministic={} plan={} search={} summary_order={} \
         search_summary_order={}",
        verdict(0),
        verdict(1),
        verdict(2),
        verdict(3),
        verdict(4)
    );
    for (slot, name) in names.iter().enumerate() {
        if let Some((seed, detail)) = &diffs[slot] {
            println!("RELABEL {label}   {name} FIRST DIFF seed={seed} {detail}");
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture graphs — copied from the election rows (same shapes, same
// seeds, same input values) so this twin measures the graphs the render
// oracle and the cuBLASLt board already talk about.
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random values — the seeding discipline of
/// `examples/support/mod.rs` (same `(n, seed)`, same values).
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

struct Fixture {
    cx: Graph,
    pairs: Vec<(NodeIndex, TypedBuffer)>,
}

fn mini_conv() -> Fixture {
    use mini_conv::MiniConvNet;
    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let x = cx.tensor((1, 1, 5, 5));
    let _out = model.forward(x).output();
    let pairs = vec![
        (x.id, weights(25, 1).into()),
        (model.conv1.weight.id, weights(18, 2).into()),
        (model.conv2.weight.id, weights(54, 3).into()),
        (model.head.weight.id, weights(6, 4).into()),
    ];
    Fixture { cx, pairs }
}

fn mini_llama3() -> Fixture {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use mini_llama3::MiniLlama3;

    const VOCAB: usize = 5;
    const D: usize = 8;
    let mut cx = Graph::new();
    let model = MiniLlama3::new(VOCAB, D, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor_dtyped(1, DType::Int);
    let k_cache = cx.tensor((4, 4));
    let v_cache = cx.tensor((4, 4));
    let gather_idx = cx.tensor_dtyped(2, DType::Int);
    let scatter_idx = cx.tensor_dtyped(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _caches_out) =
        model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits.output();

    let block = &model.blocks[0];
    let pairs = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 1).into()),
        (block.wq.weight.id, weights(D * D, 2).into()),
        (block.wk.weight.id, weights(D * 4, 3).into()),
        (block.wv.weight.id, weights(D * 4, 4).into()),
        (block.wo.weight.id, weights(D * D, 5).into()),
        (block.gate.weight.id, weights(D * 12, 6).into()),
        (block.up.weight.id, weights(D * 12, 7).into()),
        (block.down.weight.id, weights(12 * D, 8).into()),
        (k_cache.id, weights(16, 9).into()),
        (v_cache.id, weights(16, 10).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    Fixture { cx, pairs }
}

fn mini_qwen3() -> Fixture {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use mini_qwen3::MiniQwen3;

    const VOCAB: usize = 5;
    const D: usize = 8;
    const HD: usize = 2;
    let mut cx = Graph::new();
    let model = MiniQwen3::new(VOCAB, D, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor_dtyped(1, DType::Int);
    let k_cache = cx.tensor((4, 4));
    let v_cache = cx.tensor((4, 4));
    let gather_idx = cx.tensor_dtyped(2, DType::Int);
    let scatter_idx = cx.tensor_dtyped(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _caches_out) =
        model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits.output();

    let block = &model.blocks[0];
    let (q_norm, k_norm) = block.qk_norm.expect("qwen3 block carries QK-norm");
    let pairs = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 1).into()),
        (block.wq.weight.id, weights(D * D, 2).into()),
        (block.wk.weight.id, weights(D * 4, 3).into()),
        (block.wv.weight.id, weights(D * 4, 4).into()),
        (block.wo.weight.id, weights(D * D, 5).into()),
        (block.gate.weight.id, weights(D * 12, 6).into()),
        (block.up.weight.id, weights(D * 12, 7).into()),
        (block.down.weight.id, weights(12 * D, 8).into()),
        (q_norm.id, weights(HD, 11).into()),
        (k_norm.id, weights(HD, 12).into()),
        (k_cache.id, weights(16, 9).into()),
        (v_cache.id, weights(16, 10).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    Fixture { cx, pairs }
}

// ---------------------------------------------------------------------------
// The measurement — numbered so `--test-threads=1` reads top to bottom,
// and `#[ignore]`d because it is a survey, not a test (see the module doc).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "measurement, not a test: tens of minutes and no assertion; run with --ignored"]
fn relabel_01_conv() {
    let f = mini_conv();
    survey("conv", &f.cx, &f.pairs, false);
}

#[test]
#[ignore = "measurement, not a test: tens of minutes and no assertion; run with --ignored"]
fn relabel_02_llama3() {
    let f = mini_llama3();
    survey("llama3", &f.cx, &f.pairs, false);
}

#[test]
#[ignore = "measurement, not a test: tens of minutes and no assertion; run with --ignored"]
fn relabel_03_qwen3() {
    let f = mini_qwen3();
    survey("qwen3", &f.cx, &f.pairs, false);
}

#[test]
#[ignore = "measurement, not a test: tens of minutes and no assertion; run with --ignored"]
fn relabel_04_conv_cublaslt() {
    let f = mini_conv();
    survey("conv_cublaslt", &f.cx, &f.pairs, true);
}

#[test]
#[ignore = "measurement, not a test: tens of minutes and no assertion; run with --ignored"]
fn relabel_05_llama3_cublaslt() {
    let f = mini_llama3();
    survey("llama3_cublaslt", &f.cx, &f.pairs, true);
}

#[test]
#[ignore = "measurement, not a test: tens of minutes and no assertion; run with --ignored"]
fn relabel_06_qwen3_cublaslt() {
    let f = mini_qwen3();
    survey("qwen3_cublaslt", &f.cx, &f.pairs, true);
}
