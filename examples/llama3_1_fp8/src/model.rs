//! nvidia/Llama-3.1-8B-Instruct-FP8 as pure logical ops. Fidelity notes
//! (ruling 2026-08-12 — every feature of THIS checkpoint): Llama-3.1
//! anatomy = Llama-3's plus the llama3-type ROPE SCALING RAMP (factor
//! 8.0, low_freq 1.0, high_freq 4.0, original_max 8192 — the parked
//! example omitted this; we don't) and per-tensor E4M3FN quantization
//! of every layer linear with static input scales (modelopt
//! calibration). Quantization is MODEL DEFINITION: the weights stage
//! and store as F8E4M3 buffers and the quantize/dequant chain is
//! spelled in model text ([`luminal_nn::Fp8Linear`]). Embeddings and
//! the (untied) lm_head are bf16 in the checkpoint — numeric tensors,
//! not quantization — staged f32 like every zoo example.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{Embedding, Fp8Linear, KvCachePool, LayerNorm, Linear};

#[derive(Clone)]
pub struct Fp8Dims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_low_freq_factor: f32,
    pub rope_high_freq_factor: f32,
    pub rope_original_max: f32,
    pub rms_eps: f32,
}

impl Fp8Dims {
    /// nvidia/Llama-3.1-8B-Instruct-FP8.
    pub fn llama3_1_8b_fp8() -> Self {
        Self {
            vocab: 128_256,
            hidden: 4096,
            intermediate: 14_336,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 8,
            layers: 32,
            rope_theta: 500_000.0,
            rope_factor: 8.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_original_max: 8192.0,
            rms_eps: 1e-5,
        }
    }

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
            rope_factor: 8.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_original_max: 16.0,
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

/// One fp8 layer: the Llama block anatomy with every projection an
/// [`Fp8Linear`]. Example-level for now — promoted to luminal_nn when a
/// second fp8 model lands.
pub struct Fp8Block {
    pub attn_norm: LayerNorm,
    pub wq: Fp8Linear,
    pub wk: Fp8Linear,
    pub wv: Fp8Linear,
    pub wo: Fp8Linear,
    pub ffn_norm: LayerNorm,
    pub gate: Fp8Linear,
    pub up: Fp8Linear,
    pub down: Fp8Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl Fp8Block {
    fn new(l: usize, d: &Fp8Dims, cx: &mut Graph) -> Self {
        let name = |suffix: &str| format!("model.layers.{l}.{suffix}");
        Self {
            attn_norm: LayerNorm::new(
                d.hidden,
                Some(&name("input_layernorm.weight")),
                None,
                false,
                d.rms_eps,
                cx,
            ),
            wq: Fp8Linear::new(d.hidden, d.q_dim(), cx),
            wk: Fp8Linear::new(d.hidden, d.kv_dim(), cx),
            wv: Fp8Linear::new(d.hidden, d.kv_dim(), cx),
            wo: Fp8Linear::new(d.q_dim(), d.hidden, cx),
            ffn_norm: LayerNorm::new(
                d.hidden,
                Some(&name("post_attention_layernorm.weight")),
                None,
                false,
                d.rms_eps,
                cx,
            ),
            gate: Fp8Linear::new(d.hidden, d.intermediate, cx),
            up: Fp8Linear::new(d.hidden, d.intermediate, cx),
            down: Fp8Linear::new(d.intermediate, d.hidden, cx),
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
            self.wq.forward(normed),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = luminal_nn::rotary_apply(
            self.wk.forward(normed),
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
        let ff_in = self.ffn_norm.forward(x);
        let ff = self
            .down
            .forward(self.gate.forward(ff_in).silu() * self.up.forward(ff_in));
        (x + ff, k_cache, v_cache)
    }
}

pub struct Llama31Fp8 {
    pub dims: Fp8Dims,
    pub embed: Embedding,
    pub blocks: Vec<Fp8Block>,
    pub final_norm: LayerNorm,
    pub lm_head: Linear,
}

impl Llama31Fp8 {
    pub fn init(cx: &mut Graph, dims: &Fp8Dims) -> Self {
        let blocks = (0..dims.layers)
            .map(|l| Fp8Block::new(l, dims, cx))
            .collect();
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

    /// (key, handle, kind) — fp8 weights, f32 scales, and numeric
    /// (bf16-in-file) tensors need different staging conversions.
    pub fn weight_bindings(&self) -> Vec<(String, GraphTensor, WeightKind)> {
        use WeightKind::*;
        let mut map = vec![
            ("model.embed_tokens.weight".to_string(), self.embed.weight, Numeric),
            ("lm_head.weight".to_string(), self.lm_head.weight, Numeric),
            (
                "model.norm.weight".to_string(),
                self.final_norm.weight.expect("learned final norm"),
                Numeric,
            ),
        ];
        for (l, block) in self.blocks.iter().enumerate() {
            let name = |suffix: &str| format!("model.layers.{l}.{suffix}");
            for (proj, fp8) in [
                ("self_attn.q_proj", &block.wq),
                ("self_attn.k_proj", &block.wk),
                ("self_attn.v_proj", &block.wv),
                ("self_attn.o_proj", &block.wo),
                ("mlp.gate_proj", &block.gate),
                ("mlp.up_proj", &block.up),
                ("mlp.down_proj", &block.down),
            ] {
                map.push((name(&format!("{proj}.weight")), fp8.weight, Fp8));
                map.push((name(&format!("{proj}.input_scale")), fp8.input_scale, Numeric));
                map.push((name(&format!("{proj}.weight_scale")), fp8.weight_scale, Numeric));
            }
            map.push((
                name("input_layernorm.weight"),
                block.attn_norm.weight.expect("learned attn norm"),
                Numeric,
            ));
            map.push((
                name("post_attention_layernorm.weight"),
                block.ffn_norm.weight.expect("learned ffn norm"),
                Numeric,
            ));
        }
        map
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WeightKind {
    /// f32/bf16/f16 in the checkpoint → staged f32.
    Numeric,
    /// E4M3FN bytes → staged as an F8E4M3 buffer, codes untouched.
    Fp8,
}
