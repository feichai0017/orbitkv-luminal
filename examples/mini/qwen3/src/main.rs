//! MiniQwen3 demo on the REFERENCE RUNTIME: the llama-3 skeleton plus
//! Qwen3's per-head QK RMSNorm. One decode step through the native
//! ladder. Run: cargo run --release -p mini_qwen3

use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal::reference::ReferenceRuntime;
use mini_qwen3::MiniQwen3;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6).collect()
}

fn main() {
    const VOCAB: usize = 5;
    const D: usize = 8;
    const HD: usize = 2; // head_dim = D / n_heads
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
    let logits = logits.output();

    let block = &model.blocks[0];
    let (q_norm, k_norm) = block.qk_norm.expect("qwen3 block carries QK-norm");
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
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
    let data = pairs.iter().cloned().collect();
    let mut rt = ReferenceRuntime::load(&cx).expect("native load");
    let outcome = rt
        .search(&data, &ImplementationSearchOptions::default())
        .expect("search finds a plan");
    println!("search: {}", outcome.timings.summary());
    println!("refusals: {}", outcome.refusal_breakdown.summary());
    for (id, values) in &pairs {
        rt.set_data(*id, values.clone());
    }
    rt.execute().expect("winner executes");
    println!("logits: {:?}", rt.get_f32(logits.id).unwrap());
}
