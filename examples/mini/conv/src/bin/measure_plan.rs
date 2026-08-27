//! MiniConvNet (yolo-family) demo on the reference runtime.
//! Run: cargo run --release -p mini_conv

use luminal::prelude::*;
use mini_conv::MiniConvNet;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6).collect()
}

fn main() {
    let __t_rec = std::time::Instant::now();
    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let x = cx.tensor((1, 1, 5, 5));
    let out = model.forward(x).output();
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (x.id, weights(25, 1).into()),
        (model.conv1.weight.id, weights(18, 2).into()),
        (model.conv2.weight.id, weights(54, 3).into()),
        (model.head.weight.id, weights(6, 4).into()),
    ];
    measure_plan(&cx, &pairs, __t_rec);
}

#[allow(dead_code)]
fn measure_plan(
    cx: &Graph,
    pairs: &[(petgraph::graph::NodeIndex, TypedBuffer)],
    t0: std::time::Instant,
) {
    let rec_us = t0.elapsed().as_micros();
    let model = match cx.logical.model_text() {
        Ok(m) => m,
        Err(e) => {
            println!("RECORD-POISONED: {e}");
            return;
        }
    };
    let rows = model.lines().filter(|l| !l.trim().is_empty()).count();
    let applies = model.matches("(LogicalIndexMapApply").count();
    let mut depth = std::collections::HashMap::new();
    let mut max_chain = 0usize;
    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for line in model.lines() {
        if let Some(rest) = line.strip_prefix("(let v") {
            let id: usize = rest.split_whitespace().next().unwrap().parse().unwrap();
            let d = if rest.contains("(LogicalIndexMapApply v") {
                let op: usize = rest
                    .split("(LogicalIndexMapApply v")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                depth.get(&op).copied().unwrap_or(0) + 1
            } else {
                0
            };
            if d > 0 {
                *hist.entry(d).or_insert(0) += 1;
            }
            max_chain = max_chain.max(d);
            depth.insert(id, d);
        }
    }
    println!("MODEL rows={rows} applies={applies} max_apply_chain={max_chain} record_us={rec_us}");
    println!("CHAIN_DEPTH_HIST {hist:?}");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rt = luminal_reference::ReferenceRuntime::load(cx).expect("native load");
        let mut vars: Vec<_> = cx.dyn_map.iter().collect();
        vars.sort();
        for (var, value) in vars {
            rt.bind_dyn_range(*var, *value as u64, *value as u64).expect("dyn pin binds");
        }
        let data = pairs.iter().cloned().collect();
        let t = std::time::Instant::now();
        let outcome = rt
            .search(&data, &luminal::test_support::harness_search_options())
            .expect("search finds a plan");
        let search_ms = t.elapsed().as_millis();
        println!("SEARCH wall_ms={search_ms} [{}]", outcome.timings.summary());
        println!("REFUSALS {}", outcome.refusal_breakdown.summary());
        let summary = outcome.best_plan.summary();
        let mut label_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut in_ops = false;
        let mut total = 0usize;
        for line in summary.lines() {
            if line.starts_with("ops (") {
                in_ops = true;
                continue;
            }
            if line.starts_with("anti (") {
                in_ops = false;
            }
            if in_ops {
                if let Some(rest) = line.strip_prefix("  ") {
                    let label = rest.split(':').next().unwrap_or("").to_string();
                    *label_counts.entry(label).or_insert(0) += 1;
                    total += 1;
                }
            }
        }
        let buffers_line = summary
            .lines()
            .find(|l| l.starts_with("buffers ("))
            .unwrap_or("buffers (?)")
            .to_string();
        let allocs = summary
            .lines()
            .filter(|l| l.trim_start().starts_with("alloc#"))
            .count();
        println!("PLAN total_ops={total} {buffers_line} allocated_buffers={allocs}");
        for (label, n) in &label_counts {
            println!("PLANOP {label} {n}");
        }
        for (id, v) in pairs {
            rt.set_data(*id, v.clone());
        }
        let t = std::time::Instant::now();
        rt.execute().expect("winner executes");
        println!("EXECUTE wall_ms={}", t.elapsed().as_millis());
    }));
    if let Err(e) = result {
        let msg = e
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic>");
        println!("LADDER PANICKED: {msg}");
    }
}

