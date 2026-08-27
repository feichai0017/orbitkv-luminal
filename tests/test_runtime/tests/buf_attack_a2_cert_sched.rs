//! A2 ATTACK PROBES (certificates + scheduling lens, P1 review 2026-08-26).
//! Observational, against the CURRENT planner (no P1 code exists):
//!   1. single transpose view to bound output — the alloc+copy+free baseline
//!      P1 claims to eliminate;
//!   2. transpose-roundtrip (two stacked views composing to identity) — does
//!      the e-graph/extraction hand the planner ONE composed view, TWO
//!      stacked views, or zero (welded back to the parent class)? This
//!      decides whether P1's per-hop extent oracle ever sees multi-hop
//!      chains in practice;
//!   3. fan-out: the matmul value reaches a bound output through a view AND
//!      feeds a mid-graph elementwise consumer — pins the copy/anti-edge
//!      shape whose donated counterpart the admission argument must order;
//!   4. the same matmul value bound to TWO output slots (one direct, one
//!      through a view) — pins today's seed application (direct slot seeds,
//!      view slot delivery-copies) and the seen_poisons dedup P1 rides.
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;

const GENOME: &[&str] = &[
    "LayoutTensorOpCublasLt",
    "LayoutTensorOpIndexMapApplyViewGeneric",
];

fn report(name: &str, text: &str) {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, GENOME);
    let dps = luminal::dps::dps_rewrite(&graph);
    let mut views = 0usize;
    for node in dps.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            let ins: Vec<String> = op
                .inputs
                .iter()
                .map(|i| format!("{}={}", i.port, i.value))
                .collect();
            let outs: Vec<String> = op
                .outputs
                .iter()
                .map(|o| format!("{}:dims={:?}", o.eclass, o.dims))
                .collect();
            if op.op.label().contains("IndexMapApplyView") {
                views += 1;
            }
            println!("[{name}] DPS {}: ins={ins:?} outs={outs:?}", op.op.label());
        }
    }
    let plan = luminal::bufferize::bufferize(&dps).expect("bufferize");
    let summary = plan.summary();
    println!("[{name}] plan:\n{summary}");
    // Count plan NODES inside the ops section only (the r10 probe's string
    // count double-counted anti-edge lines like "BufferCopy -> BufferFree").
    let mut copies = 0usize;
    let mut allocs = 0usize;
    let mut frees = 0usize;
    let mut in_ops = false;
    for line in summary.lines() {
        if line.starts_with("ops (") {
            in_ops = true;
            continue;
        }
        if in_ops && !line.starts_with(' ') || line.starts_with("anti (") {
            in_ops = false;
        }
        if !in_ops {
            continue;
        }
        let t = line.trim_start();
        if t.starts_with("BufferCopy") {
            copies += 1;
        } else if t.starts_with("BufferAlloc") {
            allocs += 1;
        } else if t.starts_with("BufferFree") {
            frees += 1;
        }
    }
    println!(
        "[{name}] view-nodes(dps)={views} plan: allocs={allocs} copies={copies} frees={frees}"
    );
}

/// Baseline P1 target: matmul -> transpose view -> bound output.
#[test]
fn a2_single_view_to_bound() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize));
        let w = cx.tensor((8usize, 3usize));
        let _ = x.matmul(w).transpose(0, 1).output();
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.t", &text);
}

/// Two stacked transposes composing to identity: does the planner ever SEE
/// a multi-hop view chain, or does composition normalize it away?
#[test]
fn a2_double_view_roundtrip() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize));
        let w = cx.tensor((8usize, 3usize));
        let _ = x.matmul(w).transpose(0, 1).transpose(0, 1).output();
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.t.t", &text);
}

/// Fan-out: the matmul value goes to a bound output through a view AND to a
/// mid-graph elementwise consumer whose result is a second output.
#[test]
fn a2_view_fanout() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize));
        let w = cx.tensor((8usize, 3usize));
        let c = cx.tensor((4usize, 3usize));
        let y = x.matmul(w);
        let _ = y.transpose(0, 1).output();
        let _ = (y * c).output();
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.fanout", &text);
}

/// Same matmul value bound to two slots: direct + through a transpose view.
/// FINDING (2026-08-26): this configuration is unreachable through the
/// native recorder — both outputs mint `natout3_layout` with conflicting
/// shapes ((3,4) vs (4,3)) and egglog rejects the shadowing before the
/// planner runs. The advocate's case (c) is therefore cold via this
/// frontend spelling; pinned as a should_panic.
#[test]
#[should_panic(expected = "Shadowing")]
fn a2_two_slots_same_value() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize));
        let w = cx.tensor((8usize, 3usize));
        let y = x.matmul(w);
        let _ = y.transpose(0, 1).output();
        let _ = y.output();
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.2slots", &text);
}
