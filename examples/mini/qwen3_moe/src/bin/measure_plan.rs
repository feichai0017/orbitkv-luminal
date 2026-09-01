//! MiniQwen3Moe (qwen3_moe family) demo on the reference runtime.
//! Run: cargo run --release -p mini_qwen3_moe

use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal_nn::FeedForward;
use mini_qwen3_moe::MiniQwen3Moe;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

fn main() {
    let __t_rec = std::time::Instant::now();
    const VOCAB: usize = 5;
    const D: usize = 4;
    let mut cx = Graph::new();
    let model = MiniQwen3Moe::new(VOCAB, D, 2, 1, 2, 1, &mut cx);
    let ids = cx.tensor_dtyped(1, DType::Int);
    let k_cache = cx.tensor((4, D));
    let v_cache = cx.tensor((4, D));
    let gather_idx = cx.tensor_dtyped(2, DType::Int);
    let scatter_idx = cx.tensor_dtyped(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _) = model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let logits = logits.output();
    let block = &model.blocks[0];
    let FeedForward::Moe(moe) = &block.ff else {
        unreachable!()
    };
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![2i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 1).into()),
        (block.wq.weight.id, weights(D * D, 2).into()),
        (block.wk.weight.id, weights(D * D, 3).into()),
        (block.wv.weight.id, weights(D * D, 4).into()),
        (block.wo.weight.id, weights(D * D, 5).into()),
        (moe.router.id, weights(D * 2, 6).into()),
        (moe.expert_weights.id, weights(2 * D * D, 7).into()),
        (k_cache.id, weights(4 * D, 8).into()),
        (v_cache.id, weights(4 * D, 9).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
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
            rt.bind_dyn_range(*var, *value as u64, *value as u64)
                .expect("dyn pin binds");
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
