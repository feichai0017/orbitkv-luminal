//! Logical, batched Llama 3 decode graph using a page-table cache layout.
//!
//! The page-table allocator and runtime execution loop belong in runtime
//! crates. This crate specifies only the fixed-width logical graph.

pub use llama3::{Llama3, Llama3Dims};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{GraphTensor, Ns};
use luminal_nn::KvCachePool;

/// Fixed batch width of the step-invariant tick graph.
pub const ROWS: usize = 2;

pub struct BatchStep {
    pub cx: Graph,
    pub model: Llama3,
    pub tokens: GraphTensor,
    pub rope_cos: GraphTensor,
    pub rope_sin: GraphTensor,
    pub rope_rot: GraphTensor,
    pub gather_idx: GraphTensor,
    pub scatter_idx: GraphTensor,
    pub mask: GraphTensor,
    pub pool: KvCachePool,
    pub logits: GraphTensor,
    pub cache_outs: Vec<(GraphTensor, GraphTensor)>,
}

impl BatchStep {
    pub fn build(dims: &Llama3Dims, slots: usize) -> Self {
        let mut cx = Graph::new();
        let model = Llama3::init(&mut cx, dims);
        let tokens = cx.tensor_dtyped(ROWS, DType::Int);
        let rope_cos = cx.tensor((ROWS, dims.head_dim));
        let rope_sin = cx.tensor((ROWS, dims.head_dim));
        let rope_rot = cx.tensor((dims.head_dim, dims.head_dim));
        let gather_idx = cx.tensor_dtyped(slots, DType::Int);
        let scatter_idx = cx.tensor_dtyped(ROWS, DType::Int);
        let mask = cx.tensor((ROWS, slots));
        let pool = KvCachePool::new(
            &mut cx,
            dims.layers,
            slots,
            dims.kv_dim(),
            &Ns::root().child("cache"),
        );

        let mut x = model.embed.forward(tokens);
        let mut cache_outs = Vec::with_capacity(model.blocks.len());
        for (layer, block) in model.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward_rope_masked(
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
                gather_idx,
                scatter_idx,
                mask,
                rope_cos,
                rope_sin,
                rope_rot,
            );
            x = next;
            cache_outs.push((k_cache.output(), v_cache.output()));
        }
        let logits = model.lm_head.forward(model.final_norm.forward(x)).output();

        Self {
            cx,
            model,
            tokens,
            rope_cos,
            rope_sin,
            rope_rot,
            gather_idx,
            scatter_idx,
            mask,
            pool,
            logits,
            cache_outs,
        }
    }
}
