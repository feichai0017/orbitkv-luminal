//! MiniGemma3 demo on the REFERENCE RUNTIME — FULL gemma anatomy:
//! sandwich norms, decoupled head_dim, QK-norm, scale folded into Q,
//! in-graph dual-theta split-half RoPE, sliding-window local layer +
//! global layer, GeGLU, √d embedding scaling with unscaled tied head.
//! One decode step through the native ladder.
//! Run: cargo run --release -p mini_gemma3

use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal::ssa_reference::SsaReferenceRuntime;
use luminal_nn::{rope_pairing_matrix, rope_tables_split_half};
use mini_gemma3::MiniGemma3;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6).collect()
}

fn main() {
    const VOCAB: usize = 5;
    const D: usize = 6;
    const FF: usize = 8;
    const NH: usize = 2;
    const NKV: usize = 1;
    const HD: usize = 4; // q_dim = 8 ≠ d = 6 — decoupled head_dim
    const Q_DIM: usize = NH * HD;
    const KV_DIM: usize = NKV * HD;
    const SLOTS: usize = 4;
    const LAYERS: usize = 2;

    let mut cx = Graph::new();
    let ids = cx.tensor_dtyped(1, DType::Int);
    let caches: Vec<_> = (0..LAYERS)
        .map(|_| (cx.tensor((SLOTS, KV_DIM)), cx.tensor((SLOTS, KV_DIM))))
        .collect();
    let gather_idx = cx.tensor_dtyped(2, DType::Int);
    let scatter_idx = cx.tensor_dtyped(1, DType::Int);
    let rope_inputs: Vec<_> = (0..LAYERS)
        .map(|_| (cx.tensor((1, HD)), cx.tensor((1, HD))))
        .collect();
    let rope_rot = cx.tensor((HD, HD));
    let model = MiniGemma3::new(VOCAB, D, FF, NH, NKV, HD, LAYERS, 1, 2, &mut cx);
    let (logits, _caches_out) = model.forward(
        ids,
        &caches,
        gather_idx,
        scatter_idx,
        IntExpr::from(1usize),
        &rope_inputs,
        rope_rot,
    );
    let logits = logits.output();

    let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 199).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
        (rope_rot.id, rope_pairing_matrix(HD, false).into()),
        (model.final_norm.weight.expect("weighted").id, weights(D, 660).into()),
    ];
    for (layer, block) in model.blocks.iter().enumerate() {
        let (cos_table, sin_table) =
            rope_tables_split_half(&[1.0], HD, block.rope_theta, block.pos_scale);
        pairs.push((rope_inputs[layer].0.id, cos_table.into()));
        pairs.push((rope_inputs[layer].1.id, sin_table.into()));
    }
    for (layer, block) in model.blocks.iter().enumerate() {
        let seed = |slot: usize| 600 + layer * 20 + slot;
        pairs.push((block.wq.weight.id, weights(D * Q_DIM, seed(0)).into()));
        pairs.push((block.wk.weight.id, weights(D * KV_DIM, seed(1)).into()));
        pairs.push((block.wv.weight.id, weights(D * KV_DIM, seed(2)).into()));
        pairs.push((block.wo.weight.id, weights(Q_DIM * D, seed(3)).into()));
        pairs.push((block.gate.weight.id, weights(D * FF, seed(4)).into()));
        pairs.push((block.up.weight.id, weights(D * FF, seed(5)).into()));
        pairs.push((block.down.weight.id, weights(FF * D, seed(6)).into()));
        pairs.push((block.input_norm.weight.expect("weighted").id, weights(D, seed(7)).into()));
        pairs.push((block.post_attn_norm.weight.expect("weighted").id, weights(D, seed(8)).into()));
        pairs.push((block.pre_ff_norm.weight.expect("weighted").id, weights(D, seed(9)).into()));
        pairs.push((block.post_ff_norm.weight.expect("weighted").id, weights(D, seed(10)).into()));
        pairs.push((block.q_norm.id, weights(HD, seed(11)).into()));
        pairs.push((block.k_norm.id, weights(HD, seed(12)).into()));
        pairs.push((caches[layer].0.id, weights(SLOTS * KV_DIM, 300 + layer).into()));
        pairs.push((caches[layer].1.id, weights(SLOTS * KV_DIM, 320 + layer).into()));
    }
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
