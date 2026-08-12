//! The recording proof: the full ~2,200-node yolo11n graph records
//! CLEANLY on the native recorder (no poison, program assembles) with
//! the per-scale DFL respelling in place. Saturation/search of a graph
//! this size is deliberately NOT exercised here — it is the documented
//! heavy path (run `cargo run -p yolo_v11` attended; the parked-era
//! README reports >10 min / 30+ GB for the fixpoint on this model).

use luminal::prelude::*;
use yolo_v11::model::{IMG_SIZE, YoloV11};

#[test]
fn full_graph_records_cleanly() {
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let yolo = YoloV11::init(&mut cx);
    let _logits = yolo.forward(img).output();
    let program = cx.logical.native_program().expect("recorder clean — no poison");
    assert!(
        program.text.contains("(LogicalIndexMapApply"),
        "conv unfolds record as index-map views"
    );
    // The respelling holds: per-scale anchors exist as separate inputs.
    let labels: Vec<String> = cx.logical.input_labels().into_iter().map(|(l, _)| l).collect();
    for i in 0..3 {
        assert!(labels.contains(&format!("yolo.anchors.{i}")), "per-scale anchors {i}");
    }
}

/// The saturation probe (run explicitly, WATCHDOGGED — the graph is
/// ~2,200 nodes and the parked-era README reports >10 min / 30+ GB
/// for its fixpoint): random weights, minimal search budget.
#[test]
#[ignore = "heavy — 3GB-watchdog kill after ~4min on 2026-08-12 (parked-era: >10min/30GB); run by name under an RSS watchdog"]
fn saturation_probe() {
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::ssa_reference::SsaReferenceRuntime;
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let yolo = YoloV11::init(&mut cx);
    let _logits = yolo.forward(img).output();
    let _ = (img, yolo);
    let mut pairs: Vec<(petgraph::graph::NodeIndex, luminal::buffer_tensor_ir::TypedBuffer)> =
        Vec::new();
    for (seed, (label, id)) in cx.logical.input_labels().into_iter().enumerate() {
        let _ = label;
        // Shapes come from the graph itself; fill deterministic junk.
        let n: usize = cx
            .logical
            .input_static_len(id)
            .expect("static input extent");
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
            .collect();
        pairs.push((id, data.into()));
    }
    let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
    let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
    rt.search(
        &data,
        &ImplementationSearchOptions {
            generations: 1,
            generation_size: 1,
            mutations: 0,
            trials: 1,
            seed: 0,
        },
    )
    .expect("search finds a plan");
}
