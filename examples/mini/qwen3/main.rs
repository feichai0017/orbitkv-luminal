//! MiniQwen3 demo on the REFERENCE RUNTIME: the llama-3 skeleton plus
//! Qwen3's per-head QK RMSNorm. One decode step through the native
//! ladder. Run: cargo run --release --example mini_qwen3

use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::*;
use luminal::shape::Expression;
use luminal::ssa_reference::SsaReferenceRuntime;
use luminal_nn::MiniQwen3;

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
        model.forward(ids, &caches, gather_idx, scatter_idx, Expression::from(1usize));
    let logits = logits.output();

    let block = &model.blocks[0];
    let (q_norm, k_norm) = block.qk_norm.expect("qwen3 block carries QK-norm");
    let pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
        (ids.id, vec![3.0]),
        (model.embed.weight.id, weights(VOCAB * D, 1)),
        (block.wq.weight.id, weights(D * D, 2)),
        (block.wk.weight.id, weights(D * 4, 3)),
        (block.wv.weight.id, weights(D * 4, 4)),
        (block.wo.weight.id, weights(D * D, 5)),
        (block.gate.weight.id, weights(D * 12, 6)),
        (block.up.weight.id, weights(D * 12, 7)),
        (block.down.weight.id, weights(12 * D, 8)),
        (q_norm.id, weights(HD, 11)),
        (k_norm.id, weights(HD, 12)),
        (k_cache.id, weights(16, 9)),
        (v_cache.id, weights(16, 10)),
        (gather_idx.id, vec![0.0, 1.0]),
        (scatter_idx.id, vec![1.0]),
    ];
    let data = pairs.iter().cloned().collect();
    let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
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
