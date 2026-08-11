//! Qwen3 as PURE LOGICAL OPS (conversion directive 2026-08-10; this
//! crate is the zoo's exemplar): the model is authored entirely through
//! luminal_nn constructs on the native recorder — no residency markers,
//! no HLIR, no backend types, no in-graph dtype juggling. Weight-ness is
//! not authored (a weight is an ordinary named input; storage residency
//! is runtime-binding business), and the reference runtime computes in
//! f32, so the checkpoint's bf16 is a HOST staging concern (see
//! `weights.rs`). Numeric-precision policy (bf16 matmuls, f32 norms)
//! returns with the backend re-seat as binding/runtime configuration.
//!
//! RoPE is the concat-free pairing-matrix spelling ([`luminal_nn::rotary_apply`])
//! with host-precomputed tables — the rejoin-divergence workaround — and
//! the KV cache is the scatter/gather paged form, so no concat-of-slices
//! road ever forms in the recorded graph.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{Embedding, GatedFfn, LayerNorm, Linear, LlamaBlock};

/// Architecture hyperparameters. `qwen3_4b` is the real model;
/// `tiny` keeps the identical anatomy at smoke-test scale.
#[derive(Clone)]
pub struct QwenDims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl QwenDims {
    /// Qwen/Qwen3-4B. Note head_dim (128) is DECOUPLED from hidden:
    /// q_dim = 32·128 = 4096 ≠ hidden = 2560.
    pub fn qwen3_4b() -> Self {
        Self {
            vocab: 151_936,
            hidden: 2560,
            intermediate: 9728,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 8,
            layers: 36,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
        }
    }

    /// Same anatomy (GQA, decoupled head_dim, QK-norm, SwiGLU, tied
    /// head) at scalar-test scale.
    pub fn tiny() -> Self {
        Self {
            vocab: 31,
            hidden: 16,
            intermediate: 24,
            head_dim: 8,
            n_heads: 2,
            n_kv_heads: 1,
            layers: 2,
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

/// The model: an embedding (tied to the lm head), a stack of
/// rope-threaded [`LlamaBlock`]s with QK-norm, and the final RMS norm.
/// Every parameter is a named input tensor; [`Qwen::weight_bindings`]
/// is the HF-checkpoint-key → handle map the loader walks.
pub struct Qwen {
    pub dims: QwenDims,
    pub embed: Embedding,
    pub blocks: Vec<LlamaBlock>,
    pub final_norm: LayerNorm,
}

impl Qwen {
    pub fn init(cx: &mut Graph, dims: &QwenDims) -> Self {
        let blocks = (0..dims.layers).map(|l| Self::block(l, dims, cx)).collect();
        Self {
            dims: dims.clone(),
            // HF stores embed_tokens as (vocab, hidden) — the natural
            // Embedding orientation; `reverse` is the tied lm head.
            embed: Embedding::new(dims.vocab, dims.hidden, cx),
            blocks,
            final_norm: LayerNorm::new(
                dims.hidden,
                Some("model.norm.weight"),
                None,
                false,
                dims.rms_eps,
                cx,
            ),
        }
    }

    /// Literal construction: `LlamaBlock::new` couples head_dim to
    /// hidden/n_heads and mints weightless norms; Qwen3 needs the
    /// decoupled head_dim and learned norm weights, and every field is
    /// pub for exactly this. HF linears are (out, in) — `new_permuted`
    /// binds them without host transposition.
    fn block(l: usize, d: &QwenDims, cx: &mut Graph) -> LlamaBlock {
        let name = |suffix: &str| format!("model.layers.{l}.{suffix}");
        LlamaBlock {
            ffn_kind: GatedFfn::SwiGlu,
            attn_norm: LayerNorm::new(
                d.hidden,
                Some(&name("input_layernorm.weight")),
                None,
                false,
                d.rms_eps,
                cx,
            ),
            wq: Linear::new_permuted(d.hidden, d.q_dim(), false, cx),
            wk: Linear::new_permuted(d.hidden, d.kv_dim(), false, cx),
            wv: Linear::new_permuted(d.hidden, d.kv_dim(), false, cx),
            wo: Linear::new_permuted(d.q_dim(), d.hidden, false, cx),
            qk_norm: Some((
                cx.named_tensor(name("self_attn.q_norm.weight"), d.head_dim),
                cx.named_tensor(name("self_attn.k_norm.weight"), d.head_dim),
            )),
            ffn_norm: LayerNorm::new(
                d.hidden,
                Some(&name("post_attention_layernorm.weight")),
                None,
                false,
                d.rms_eps,
                cx,
            ),
            gate: Linear::new_permuted(d.hidden, d.intermediate, false, cx),
            up: Linear::new_permuted(d.hidden, d.intermediate, false, cx),
            down: Linear::new_permuted(d.intermediate, d.hidden, false, cx),
            n_heads: d.n_heads,
            n_kv_heads: d.n_kv_heads,
            head_dim: d.head_dim,
        }
    }

    /// One decode step over the paged cache. `tokens`/`q_pos` are (s,)
    /// Int; rope tables (s, head_dim) and the pairing matrix are
    /// host-built inputs; caches are one (slots, kv_dim) pair per layer.
    /// Returns (logits (s, vocab), per-layer cache outs).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(tokens.dtype, DType::Int);
        assert_eq!(caches.len(), self.blocks.len());
        let mut x = self.embed.forward(tokens);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward_rope(
                x,
                caches[layer].0,
                caches[layer].1,
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
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }

    /// The HF-checkpoint-key → input-handle map. Qwen3-4B ties lm_head
    /// to the embedding, so the combined file has no lm_head.weight and
    /// none is listed.
    pub fn weight_bindings(&self) -> Vec<(String, GraphTensor)> {
        let mut map = vec![("model.embed_tokens.weight".to_string(), self.embed.weight)];
        for (l, block) in self.blocks.iter().enumerate() {
            let name = |suffix: &str| format!("model.layers.{l}.{suffix}");
            let (q_norm, k_norm) = block.qk_norm.expect("qwen blocks carry qk-norm");
            map.push((name("self_attn.q_proj.weight"), block.wq.weight));
            map.push((name("self_attn.k_proj.weight"), block.wk.weight));
            map.push((name("self_attn.v_proj.weight"), block.wv.weight));
            map.push((name("self_attn.o_proj.weight"), block.wo.weight));
            map.push((name("self_attn.q_norm.weight"), q_norm));
            map.push((name("self_attn.k_norm.weight"), k_norm));
            map.push((
                name("input_layernorm.weight"),
                block.attn_norm.weight.expect("learned attn norm"),
            ));
            map.push((
                name("post_attention_layernorm.weight"),
                block.ffn_norm.weight.expect("learned ffn norm"),
            ));
            map.push((name("mlp.gate_proj.weight"), block.gate.weight));
            map.push((name("mlp.up_proj.weight"), block.up.weight));
            map.push((name("mlp.down_proj.weight"), block.down.weight));
        }
        map.push((
            "model.norm.weight".to_string(),
            self.final_norm.weight.expect("learned final norm"),
        ));
        map
    }
}
