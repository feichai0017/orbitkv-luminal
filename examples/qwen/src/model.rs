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
use luminal::prelude::{GraphTensor, Ns};
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
/// Every parameter is a named input tensor whose LABEL is its HF
/// checkpoint key — the loader walks `input_specs()` and matches labels
/// against the checkpoint (label-driven staging, ruling 2026-08-13).
/// Qwen3-4B ties lm_head to the embedding, so no lm_head.weight input
/// exists.
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
        }
    }

    /// Literal construction: `LlamaBlock::new` couples head_dim to
    /// hidden/n_heads and mints weightless norms; Qwen3 needs the
    /// decoupled head_dim and learned norm weights, and every field is
    /// pub for exactly this. HF linears are (out, in) — `new_permuted`
    /// binds them without host transposition.
    fn block(l: usize, d: &QwenDims, cx: &mut Graph) -> LlamaBlock {
        let ns = Ns::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        LlamaBlock {
            ffn_kind: GatedFfn::SwiGlu,
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
            qk_norm: Some((
                cx.named_tensor(attn.child("q_norm").leaf("weight"), d.head_dim),
                cx.named_tensor(attn.child("k_norm").leaf("weight"), d.head_dim),
            )),
            ffn_norm: LayerNorm::new(
                d.hidden,
                true,
                false,
                false,
                d.rms_eps,
                &ns.child("post_attention_layernorm"),
                cx,
            ),
            gate: Linear::new_permuted(d.hidden, d.intermediate, false, &mlp.child("gate_proj"), cx),
            up: Linear::new_permuted(d.hidden, d.intermediate, false, &mlp.child("up_proj"), cx),
            down: Linear::new_permuted(d.intermediate, d.hidden, false, &mlp.child("down_proj"), cx),
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

}
