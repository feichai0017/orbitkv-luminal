//! unsloth/gemma-3-4b-it (text tower) as pure logical ops — 100% of
//! THIS checkpoint's features (ruling 2026-08-12): 34 layers in the
//! 5-local:1-global alternation (globals at l+1 ≡ 0 mod 6), sliding
//! window 1024 on locals (enforced by mask — no eviction), dual rope
//! thetas (10k local / 1M global) with linear position scaling /8 on
//! globals only, per-head QK-norm (learned, eps 1e-6) before rope,
//! attention scale 1/16 FOLDED INTO Q (query_pre_attn_scalar =
//! head_dim), sandwich norms (four per layer, the Gemma (1+w) pattern
//! pre-baked host-side at combine), GeGLU FFN, TIED embeddings with
//! the sqrt(hidden) normalizer IN-GRAPH over the unscaled table (the
//! parked example pre-scaled the table and duplicated an unscaled head
//! copy — same math, this spelling keeps one table and matches the HF
//! config's tie + normalizer directly). No logit softcaps (Gemma 3
//! dropped them for QK-norm).

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{GraphTensor, Ns};
use luminal_nn::{Embedding, GemmaBlock, KvCachePool, LayerNorm, Linear};

#[derive(Clone)]
pub struct Gemma3Dims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub window: usize,
    pub sliding_pattern: usize,
    pub rms_eps: f32,
}

impl Gemma3Dims {
    pub fn gemma3_4b() -> Self {
        Self {
            vocab: 262_208,
            hidden: 2560,
            intermediate: 10_240,
            head_dim: 256,
            n_heads: 8,
            n_kv_heads: 4,
            layers: 34,
            window: 1024,
            sliding_pattern: 6,
            rms_eps: 1e-6,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab: 31,
            hidden: 16,
            intermediate: 24,
            head_dim: 4,
            n_heads: 4,
            n_kv_heads: 2,
            layers: 2, // layer 0 local, layer 1 global (pattern 2)
            window: 3,
            sliding_pattern: 2,
            rms_eps: 1e-6,
        }
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn is_local(&self, layer: usize) -> bool {
        (layer + 1) % self.sliding_pattern != 0
    }
}

pub struct Gemma3 {
    pub dims: Gemma3Dims,
    pub embed: Embedding,
    pub blocks: Vec<GemmaBlock>,
    pub final_norm: LayerNorm,
}

impl Gemma3 {
    pub fn init(cx: &mut Graph, dims: &Gemma3Dims) -> Self {
        let blocks = (0..dims.layers).map(|l| Self::block(l, dims, cx)).collect();
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
            )
            .with_unit_offset(),
        }
    }

    fn block(l: usize, d: &Gemma3Dims, cx: &mut Graph) -> GemmaBlock {
        let local = d.is_local(l);
        let ns = Ns::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        let rms = |ns: &Ns, cx: &mut Graph| {
            LayerNorm::new(d.hidden, true, false, false, d.rms_eps, ns, cx).with_unit_offset()
        };
        GemmaBlock {
            input_norm: rms(&ns.child("input_layernorm"), cx),
            post_attn_norm: rms(&ns.child("post_attention_layernorm"), cx),
            pre_ff_norm: rms(&ns.child("pre_feedforward_layernorm"), cx),
            post_ff_norm: rms(&ns.child("post_feedforward_layernorm"), cx),
            wq: Linear::new_permuted(
                d.hidden,
                d.n_heads * d.head_dim,
                false,
                &attn.child("q_proj"),
                cx,
            ),
            wk: Linear::new_permuted(d.hidden, d.kv_dim(), false, &attn.child("k_proj"), cx),
            wv: Linear::new_permuted(d.hidden, d.kv_dim(), false, &attn.child("v_proj"), cx),
            wo: Linear::new_permuted(
                d.n_heads * d.head_dim,
                d.hidden,
                false,
                &attn.child("o_proj"),
                cx,
            ),
            q_norm: cx.named_tensor(attn.child("q_norm").leaf("weight"), d.head_dim),
            k_norm: cx.named_tensor(attn.child("k_norm").leaf("weight"), d.head_dim),
            gate: Linear::new_permuted(
                d.hidden,
                d.intermediate,
                false,
                &mlp.child("gate_proj"),
                cx,
            ),
            up: Linear::new_permuted(d.hidden, d.intermediate, false, &mlp.child("up_proj"), cx),
            down: Linear::new_permuted(
                d.intermediate,
                d.hidden,
                false,
                &mlp.child("down_proj"),
                cx,
            ),
            local,
            window: d.window,
            rope_theta: if local { 10_000.0 } else { 1_000_000.0 },
            pos_scale: if local { 1.0 } else { 1.0 / 8.0 },
            n_heads: d.n_heads,
            n_kv_heads: d.n_kv_heads,
            head_dim: d.head_dim,
        }
    }

    /// One decode step. Local and global layers consume their own rope
    /// tables (dual theta + position scaling), fed as two (s, head_dim)
    /// pairs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: GraphTensor,
        q_pos: GraphTensor,
        rope_local: (GraphTensor, GraphTensor),
        rope_global: (GraphTensor, GraphTensor),
        rope_rot: GraphTensor,
        pool: &KvCachePool,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(tokens.dtype, DType::Int);
        // The HF gemma normalizer: hidden states scale by sqrt(hidden)
        // right after the (unscaled) embedding lookup.
        let mut x = self.embed.forward(tokens) * (self.dims.hidden as f32).sqrt();
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (cos, sin) = if block.local { rope_local } else { rope_global };
            let (next, k_cache, v_cache) = block.forward_positional(
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
                gather_idx,
                scatter_idx,
                q_pos,
                cos,
                sin,
                rope_rot,
            );
            x = next;
            caches_out.push((k_cache, v_cache));
        }
        // Tied head over the unscaled table.
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}
