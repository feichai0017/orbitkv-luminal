//! Qwen/Qwen3-30B-A3B as pure logical ops — 100% of THIS checkpoint
//! (ruling 2026-08-12): 48 layers, hidden 2048, GQA 32/4 at head_dim
//! 128, per-head QK-norm (learned, eps 1e-6) before rope, rope theta
//! 1e6 unscaled, UNTIED lm_head, and the MoE FFN on every layer — 128
//! experts, top-8, NO shared expert, router in F32 with the Qwen3
//! scoring order (softmax over all experts FIRST, then top-k, then
//! renormalize — norm_topk_prob) via [`luminal_nn::MoETopK`].

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{GraphTensor, Ns};
use luminal_nn::{Embedding, KvCachePool, LayerNorm, Linear, MoETopK};

#[derive(Clone)]
pub struct Qwen3MoeDims {
    pub vocab: usize,
    pub hidden: usize,
    pub moe_intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub experts: usize,
    pub top_k: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl Qwen3MoeDims {
    pub fn qwen3_30b_a3b() -> Self {
        Self {
            vocab: 151_936,
            hidden: 2048,
            moe_intermediate: 768,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 4,
            layers: 48,
            experts: 128,
            top_k: 8,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab: 23,
            hidden: 8,
            moe_intermediate: 6,
            head_dim: 4,
            n_heads: 2,
            n_kv_heads: 1,
            layers: 2,
            experts: 4,
            top_k: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

pub struct Qwen3MoeBlock {
    /// Per-expert (gate, up, down) handles — the HF checkpoint
    /// anatomy; the MoE stacks them IN-GRAPH.
    pub expert_parts: Vec<(GraphTensor, GraphTensor, GraphTensor)>,
    pub attn_norm: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: GraphTensor,
    pub k_norm: GraphTensor,
    pub ffn_norm: LayerNorm,
    pub moe: MoETopK,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl Qwen3MoeBlock {
    fn new(l: usize, d: &Qwen3MoeDims, cx: &mut Graph) -> Self {
        let ns = Ns::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        let experts = mlp.child("experts");
        let expert_parts: Vec<(GraphTensor, GraphTensor, GraphTensor)> = (0..d.experts)
            .map(|e| {
                let expert = experts.index(e);
                (
                    cx.named_tensor(
                        expert.child("gate_proj").leaf("weight"),
                        (d.moe_intermediate, d.hidden),
                    ),
                    cx.named_tensor(
                        expert.child("up_proj").leaf("weight"),
                        (d.moe_intermediate, d.hidden),
                    ),
                    cx.named_tensor(
                        expert.child("down_proj").leaf("weight"),
                        (d.hidden, d.moe_intermediate),
                    ),
                )
            })
            .collect();
        Self {
            attn_norm: LayerNorm::new(
                d.hidden,
                true,
                false,
                false,
                d.rms_eps,
                &ns.child("input_layernorm"),
                cx,
            ),
            wq: Linear::new_permuted(d.hidden, d.q_dim(), false, &attn.child("q_proj"), cx),
            wk: Linear::new_permuted(d.hidden, d.kv_dim(), false, &attn.child("k_proj"), cx),
            wv: Linear::new_permuted(d.hidden, d.kv_dim(), false, &attn.child("v_proj"), cx),
            wo: Linear::new_permuted(d.q_dim(), d.hidden, false, &attn.child("o_proj"), cx),
            q_norm: cx.named_tensor(attn.child("q_norm").leaf("weight"), d.head_dim),
            k_norm: cx.named_tensor(attn.child("k_norm").leaf("weight"), d.head_dim),
            ffn_norm: LayerNorm::new(
                d.hidden,
                true,
                false,
                false,
                d.rms_eps,
                &ns.child("post_attention_layernorm"),
                cx,
            ),
            moe: MoETopK::from_per_expert(
                Linear::new_permuted(d.hidden, d.experts, false, &mlp.child("gate"), cx),
                &expert_parts,
                d.top_k,
            ),
            expert_parts,
            n_heads: d.n_heads,
            n_kv_heads: d.n_kv_heads,
            head_dim: d.head_dim,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.attn_norm.forward(x);
        let q = luminal_nn::rotary_apply(
            luminal_nn::rms_norm_heads(self.wq.forward(normed), self.head_dim, self.q_norm, 1e-6),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = luminal_nn::rotary_apply(
            luminal_nn::rms_norm_heads(self.wk.forward(normed), self.head_dim, self.k_norm, 1e-6),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let (attn, k_cache, v_cache) = luminal_nn::paged_attention_positional(
            q,
            k,
            self.wv.forward(normed),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            q_pos,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            None,
            1.0 / (self.head_dim as f32).sqrt(),
        );
        let x = x + self.wo.forward(attn);
        let ff = self.moe.forward(self.ffn_norm.forward(x));
        (x + ff, k_cache, v_cache)
    }
}

pub struct Qwen3Moe {
    pub dims: Qwen3MoeDims,
    pub embed: Embedding,
    pub blocks: Vec<Qwen3MoeBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: Linear,
}

impl Qwen3Moe {
    pub fn init(cx: &mut Graph, dims: &Qwen3MoeDims) -> Self {
        let blocks = (0..dims.layers)
            .map(|l| Qwen3MoeBlock::new(l, dims, cx))
            .collect();
        Self {
            dims: dims.clone(),
            embed: Embedding::new(
                dims.vocab,
                dims.hidden,
                &Ns::root().child("model").child("embed_tokens"),
                cx,
            ),
            blocks,
            final_norm: LayerNorm::new(
                dims.hidden,
                true,
                false,
                false,
                dims.rms_eps,
                &Ns::root().child("model").child("norm"),
                cx,
            ),
            lm_head: Linear::new_permuted(
                dims.hidden,
                dims.vocab,
                false,
                &Ns::root().child("lm_head"),
                cx,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
        pool: &KvCachePool,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(tokens.dtype, DType::Int);
        let mut x = self.embed.forward(tokens);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward(
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
                gather_idx,
                scatter_idx,
                q_pos,
                rope_cos,
                rope_sin,
                rope_rot,
            );
            x = next;
            caches_out.push((k_cache, v_cache));
        }
        let logits = self.lm_head.forward(self.final_norm.forward(x));
        (logits, caches_out)
    }
}
