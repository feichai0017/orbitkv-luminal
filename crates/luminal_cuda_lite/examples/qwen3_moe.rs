//! MiniQwen3Moe (MoE decoder, one decode step) on CUDA-lite: reference
//! run vs device run on identical seeded inputs, compared through the
//! disclosed layout. Canonical dims from
//! `examples/mini/qwen3_moe/src/bin/measure_plan.rs`.
//!
//! Run: cargo run -p luminal_cuda_lite --example qwen3_moe --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("qwen3_moe");
}

#[cfg(feature = "device")]
fn main() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use luminal_nn::FeedForward;
    use mini_qwen3_moe::MiniQwen3Moe;
    use support::weights;

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
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
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

    if let Err(e) =
        support::device::run_differential("qwen3_moe", &cx, &pairs, &[("logits", logits.id)])
    {
        eprintln!("qwen3_moe: FAIL: {e:#}");
        std::process::exit(1);
    }
}
