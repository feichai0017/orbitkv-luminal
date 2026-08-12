//! Meta-Llama-3-8B-Instruct as PURE LOGICAL OPS — the second zoo
//! conversion (per-HF-model fidelity ruling 2026-08-12: this example
//! is exactly `NousResearch/Meta-Llama-3-8B-Instruct`, every
//! architectural feature of that checkpoint and no other's). Anatomy:
//! 32 pre-RMSNorm layers (LEARNED norm weights, eps 1e-5), GQA 32/8
//! heads at head_dim 128, split-half rope at theta 500 000 with NO
//! scaling (this checkpoint's rope_scaling is null — the 3.1-family
//! frequency ramp belongs to the fp8 example), SwiGLU FFN, UNTIED
//! lm_head, slot-pool KV cache ([`luminal_nn::KvCachePool`] — cache
//! structure is model definition; this model uses the position-slots
//! form). Projections are UNFUSED: the old [v;q;k]/[gate;up] byte
//! fusion was a GEMV batching choice, not architecture, and the
//! unfused spelling keeps original HF tensor names and no slice
//! triple. Rope is the concat-free pairing-matrix spelling.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{Embedding, GatedFfn, KvCachePool, LayerNorm, Linear, LlamaBlock};

#[derive(Clone)]
pub struct Llama3Dims {
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

impl Llama3Dims {
    /// NousResearch/Meta-Llama-3-8B-Instruct.
    pub fn llama3_8b() -> Self {
        Self {
            vocab: 128_256,
            hidden: 4096,
            intermediate: 14_336,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 8,
            layers: 32,
            rope_theta: 500_000.0,
            rms_eps: 1e-5,
        }
    }

    /// Same anatomy (GQA, learned norms, untied head, SwiGLU) at
    /// smoke-test scale.
    pub fn tiny() -> Self {
        Self {
            vocab: 29,
            hidden: 16,
            intermediate: 24,
            head_dim: 4,
            n_heads: 4,
            n_kv_heads: 2,
            layers: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

pub struct Llama3 {
    pub dims: Llama3Dims,
    pub embed: Embedding,
    pub blocks: Vec<LlamaBlock>,
    pub final_norm: LayerNorm,
    /// UNTIED head — a real (vocab, hidden) checkpoint tensor.
    pub lm_head: Linear,
}

impl Llama3 {
    pub fn init(cx: &mut Graph, dims: &Llama3Dims) -> Self {
        let blocks = (0..dims.layers).map(|l| Self::block(l, dims, cx)).collect();
        Self {
            dims: dims.clone(),
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
            lm_head: Linear::new_permuted(dims.hidden, dims.vocab, false, cx),
        }
    }

    /// Literal construction: learned per-layer norm weights at this
    /// checkpoint's eps 1e-5, NO qk-norm (a Qwen3 feature, absent
    /// here), HF (out, in) linears bound untransposed.
    fn block(l: usize, d: &Llama3Dims, cx: &mut Graph) -> LlamaBlock {
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
            qk_norm: None,
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

    /// One decode step over the slot pool; see the qwen exemplar for
    /// the step-invariant contract (everything per-step is DATA).
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
        assert_eq!(pool.layers.len(), self.blocks.len());
        let mut x = self.embed.forward(tokens);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward_rope(
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

    pub fn weight_bindings(&self) -> Vec<(String, GraphTensor)> {
        let mut map = vec![
            ("model.embed_tokens.weight".to_string(), self.embed.weight),
            ("lm_head.weight".to_string(), self.lm_head.weight),
        ];
        for (l, block) in self.blocks.iter().enumerate() {
            let name = |suffix: &str| format!("model.layers.{l}.{suffix}");
            map.push((name("self_attn.q_proj.weight"), block.wq.weight));
            map.push((name("self_attn.k_proj.weight"), block.wk.weight));
            map.push((name("self_attn.v_proj.weight"), block.wv.weight));
            map.push((name("self_attn.o_proj.weight"), block.wo.weight));
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
