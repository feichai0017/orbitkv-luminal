//! MiniQwen3 (llama-3 skeleton + per-head QK RMSNorm, one decode step)
//! on CUDA-lite: reference run vs device run on identical seeded
//! inputs, compared through the disclosed layout. Canonical dims from
//! `examples/mini/qwen3/src/bin/measure_plan.rs`.
//!
//! Run: cargo run -p luminal_cuda_lite --example qwen3 --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("qwen3");
}

#[cfg(feature = "device")]
fn main() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use mini_qwen3::MiniQwen3;
    use support::weights;

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
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
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

    if let Err(e) =
        support::device::run_differential("qwen3", &cx, &pairs, &[("logits", logits.id)])
    {
        eprintln!("qwen3: FAIL: {e:#}");
        std::process::exit(1);
    }
}
