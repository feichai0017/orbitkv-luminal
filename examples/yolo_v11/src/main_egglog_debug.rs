//! Diagnostic binary: builds a small slice of the YOLO graph (configurable via
//! `YOLO_DEBUG_LAYERS`), runs the luminal egglog pipeline manually, and prints
//! the per-rule match counts and timings. Useful for figuring out which rules
//! cause the e-graph to balloon on conv-heavy networks.
//!
//! Run with:
//!   YOLO_DEBUG_LAYERS=2 cargo run -p yolo_v11 --release --bin yolo_v11_egglog_debug
//!   YOLO_DEBUG_LAYERS=3 cargo run -p yolo_v11 --release --bin yolo_v11_egglog_debug
//!   YOLO_DEBUG_LAYERS=4 cargo run -p yolo_v11 --release --bin yolo_v11_egglog_debug

use std::time::Instant;
use std::sync::Arc;

use luminal::egglog_utils::*;
use luminal::op::EgglogOp;
use luminal::op::IntoEgglogOp;
use luminal::prelude::*;
use luminal_cuda_lite::runtime::CudaRuntime;

#[path = "model.rs"]
mod model;
use model::*;

fn main() {
    let n_layers: usize = std::env::var("YOLO_DEBUG_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    println!("Building YOLO subset with {n_layers} backbone layers");

    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));

    // Build only first n backbone layers from the YoloV11 init.
    let conv0 = Conv::new("model.0.conv", 3, C0, 3, 2, 1, &mut cx);
    let mut x = conv0.forward(img);
    if n_layers >= 2 {
        let conv1 = Conv::new("model.1.conv", C0, C1, 3, 2, 1, &mut cx);
        x = conv1.forward(x);
    }
    if n_layers >= 3 {
        let c3k2_2 = C3k2::new("model.2", C1, C2, 1, false, 0.25, true, &mut cx);
        x = c3k2_2.forward(x);
    }
    if n_layers >= 4 {
        let conv3 = Conv::new("model.3.conv", C2, C2, 3, 2, 1, &mut cx);
        x = conv3.forward(x);
    }
    if n_layers >= 5 {
        let c3k2_4 = C3k2::new("model.4", C2, C3, 1, false, 0.25, true, &mut cx);
        x = c3k2_4.forward(x);
    }
    if n_layers >= 6 {
        let conv5 = Conv::new("model.5.conv", C3, C3, 3, 2, 1, &mut cx);
        x = conv5.forward(x);
    }
    if n_layers >= 7 {
        let c3k2_6 = C3k2::new("model.6", C3, C3, 1, true, 0.5, true, &mut cx);
        x = c3k2_6.forward(x);
    }
    if n_layers >= 8 {
        let conv7 = Conv::new("model.7.conv", C3, C4, 3, 2, 1, &mut cx);
        x = conv7.forward(x);
    }
    if n_layers >= 9 {
        let c3k2_8 = C3k2::new("model.8", C4, C4, 1, true, 0.5, true, &mut cx);
        x = c3k2_8.forward(x);
    }
    if n_layers >= 10 {
        let sppf_9 = Sppf::new("model.9", C4, C4, 5, &mut cx);
        x = sppf_9.forward(x);
    }
    if n_layers >= 11 {
        let c2psa_10 = C2psa::new("model.10", C4, C4, 1, 0.5, &mut cx);
        x = c2psa_10.forward(x);
    }
    let _ = x.output();

    // Skip the auto-loop-rolling prepass (it's private to luminal::graph),
    // it doesn't usually find anything useful for YOLO anyway since the layers
    // aren't repeats.
    let cx = cx;

    // Replicate build_search_space's egglog construction so we can call
    // run_egglog_with_report and see the per-rule statistics.
    let mut ops: Vec<Arc<Box<dyn EgglogOp>>> = <(luminal_cuda_lite::kernel::Ops, luminal_cuda_lite::host::Ops) as IntoEgglogOp>::into_vec();
    ops.extend(<luminal::hlir::HLIROps as IntoEgglogOp>::into_vec());
    println!("Op set size: {} ops", ops.len());

    let (program, root) = hlir_to_egglog(&cx);
    println!("Egglog program length: {} bytes", program.len());

    let cleanup_mode = std::env::var("YOLO_DEBUG_CLEANUP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    println!("Running egglog with cleanup={cleanup_mode}...");
    let t0 = Instant::now();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_egglog_with_report(&program, &root, &ops, cleanup_mode)
    }));
    let elapsed = t0.elapsed();
    println!("  total elapsed: {:?}", elapsed);

    match res {
        Ok(Ok((egraph, report))) => {
            println!(
                "  egraph: {} enodes / {} eclasses / {} roots",
                egraph.enodes.len(),
                egraph.eclasses.len(),
                egraph.roots.len()
            );
            println!("\n=== EARLY stage took {:?} ===", report.early.total_time);
            print_top_rules(&report.early, 30);
            println!("\n=== FULL stage took {:?} ===", report.full.total_time);
            print_top_rules(&report.full, 30);

            // Report eclass-label statistics: which op kinds have how many
            // alternatives, and which are HLIR-only (would-be cascade victims).
            print_eclass_stats(&egraph);
        }
        Ok(Err(e)) => {
            println!("egglog ERROR: {e:?}");
        }
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&'static str>() {
                s.to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            println!("egglog PANIC: {msg}");
        }
    }
}

fn print_eclass_stats(egraph: &SerializedEGraph) {
    use std::collections::BTreeMap;
    // Count enodes per eclass, group eclasses by their "label set"
    let mut hlir_only = BTreeMap::<String, usize>::new();
    let mut no_alt = Vec::<(String, usize)>::new();
    for (cid, (_class_type, enodes_in_class)) in &egraph.eclasses {
        let labels: Vec<&str> = enodes_in_class
            .iter()
            .filter_map(|n| egraph.enodes.get(n).map(|(l, _)| l.as_str()))
            .collect();
        if labels.is_empty() {
            continue;
        }
        // Identify HLIR-only eclasses (no Kernel*/cuBLAS alternative)
        let has_kernel_alt = labels.iter().any(|l| l.contains("Kernel") || l.contains("cublas") || l.contains("Tile"));
        let has_hlir = labels.iter().any(|l| !l.contains("Kernel") && !l.contains("cublas") && !l.contains("Tile") && !l.contains("[..."));
        if has_hlir && !has_kernel_alt {
            // Only HLIR ops, no kernel alternative — would cascade if cleaned up
            for l in &labels {
                if !l.starts_with("Op") && !l.contains("[..]") {
                    *hlir_only.entry(l.to_string()).or_default() += 1;
                }
            }
            no_alt.push((cid.to_string(), labels.len()));
        }
    }
    println!("\n=== HLIR-only eclasses (no kernel alternative) ===");
    for (label, count) in hlir_only.iter() {
        println!("  {label}: {count}");
    }
    println!("Total HLIR-only eclasses: {}", no_alt.len());
}


fn print_top_rules(report: &EgglogStageReport, n: usize) {
    let mut entries: Vec<(String, usize, std::time::Duration)> = report
        .num_matches_per_rule
        .iter()
        .map(|(k, &v)| {
            let t = report
                .search_and_apply_time_per_rule
                .get(k)
                .copied()
                .unwrap_or_default();
            (k.clone(), v, t)
        })
        .collect();
    // Sort by time (descending) so we surface the heaviest rules first.
    entries.sort_by(|a, b| b.2.cmp(&a.2));
    println!("Top rules by time (matches, total time):");
    for (name, matches, time) in entries.iter().take(n) {
        if name.contains('(') {
            continue;
        }
        println!("  {:>14} matches  {:>10?}  {}", matches, time, name);
    }
    let total_matches: usize = entries.iter().map(|e| e.1).sum();
    let total_time: std::time::Duration = entries.iter().map(|e| e.2).sum();
    println!(
        "Sum: {} matches across {} rules, {:?} total rule time",
        total_matches,
        entries.len(),
        total_time
    );
}

// Make CudaRuntime referenced so the Runtime::Ops are linked in.
#[allow(dead_code)]
fn _force_link() -> CudaRuntime {
    CudaRuntime::initialize(luminal_cuda_lite::cudarc::driver::CudaContext::new(0).unwrap().default_stream())
}
