//! The recording proof: the full ~2,200-node yolo11n graph records
//! CLEANLY on the native recorder (no poison, program assembles) with
//! the per-scale DFL respelling in place. Saturation/search of a graph
//! this size is deliberately NOT exercised here — it is the documented
//! heavy path (run `cargo run -p yolo_v11` attended; the parked-era
//! README reports >10 min / 30+ GB for the fixpoint on this model).

use luminal::prelude::*;
use yolo_v11::model::{IMG_SIZE, YoloV11};

#[test]
#[cfg_attr(not(feature = "zoo-proofs"), ignore = "zoo fidelity proof: a full search + decode loop (llama3 measured at 185s). The zoo is not part of the default test path — run explicitly, e.g. `cargo test -p llama3 -- --ignored`.")]
fn full_graph_records_cleanly() {
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let yolo = YoloV11::init(&mut cx);
    let _logits = yolo.forward(img).output();
    let program = cx.logical.bound_program(&luminal_reference::ReferenceBindings).expect("recorder clean — no poison");
    assert!(
        program.text.contains("(LogicalIndexMapApply"),
        "conv unfolds record as index-map views"
    );
    // The respelling holds: per-scale anchors exist as separate inputs.
    let labels: Vec<String> = cx.logical.input_specs().into_iter().map(|s| s.label).collect();
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
    use luminal_reference::ReferenceRuntime;
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let yolo = YoloV11::init(&mut cx);
    let _logits = yolo.forward(img).output();
    let _ = (img, yolo);
    let mut pairs: Vec<(petgraph::graph::NodeIndex, luminal::buffer_tensor_ir::TypedBuffer)> =
        Vec::new();
    for (seed, spec) in cx.logical.input_specs().into_iter().enumerate() {
        // Geometry comes from the graph itself; fill deterministic junk.
        assert_eq!(
            spec.dtype,
            luminal::prelude::DType::F32,
            "junk-fill is F32-only; '{}' declares {:?}",
            spec.label,
            spec.dtype
        );
        let n: usize = spec
            .dims
            .iter()
            .map(|d| d.to_usize().expect("static input extent"))
            .product();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
            .collect();
        pairs.push((spec.id, data.into()));
    }
    let mut rt = ReferenceRuntime::load(&cx).expect("native load");
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
