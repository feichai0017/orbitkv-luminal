//! MINI MODELS (rulings 2026-08-07/10): one small, runnable
//! representative per example-model FAMILY, named for the model it
//! represents, defined here in core — the runtimes' example directories
//! only RUN them. Coverage: llama/paged_llama (MiniLlama3), qwen
//! (MiniQwen3 — adds per-head QK RMSNorm), gemma (MiniGemma3 — GeGLU;
//! known fidelity gaps doc-noted on the struct), qwen3_moe
//! (MiniQwen3Moe), gemma4_moe (MiniGemma4Moe — adds final logit
//! soft-capping), whisper (MiniWhisper — encoder self-attention +
//! cross-attention), yolo (MiniConvNet), flux2 (MiniDit — adaLN
//! modulation, double/single-stream joint attention, interleaved-pair
//! multi-axis RoPE).

use crate::{DecoderBlock, Embedding, FeedForward, GatedFfn, Linear, LlamaBlock, MoE, ConvND};
use luminal::prelude::*;
use luminal::shape::Expression;

/// Plain multi-head attention, no mask, no cache: q (sq, d) attends over
/// k/v (sk, d) — the encoder/cross-attention primitive.
pub fn attention(
    q: GraphTensor,
    k: GraphTensor,
    v: GraphTensor,
    n_heads: usize,
    head_dim: usize,
) -> GraphTensor {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sq = q.dims()[0];
    let sk = k.dims()[0];
    let q = q.split_dims(1, head_dim).permute((1, 0, 2)); // (nh, sq, hd)
    let k = k.split_dims(1, head_dim).permute((1, 2, 0)); // (nh, hd, sk)
    let v = v.split_dims(1, head_dim).permute((1, 0, 2)); // (nh, sk, hd)
    let scores = q.matmul(k) * scale; // (nh, sq, sk)
    let weights = scores.softmax(2);
    let out = weights.matmul(v); // (nh, sq, hd)
    let _ = (sq, sk);
    out.permute((1, 0, 2)).merge_dims(1, 2) // (sq, nh·hd)
}

/// Shared GQA-decoder assembly behind the family minis: embed →
/// N × LlamaBlock (paged KV cache) → final RMSNorm → tied logits. Each
/// family keeps its own NAMED front door (ruling 2026-08-10: minis are
/// named for the model they represent, not parameterized as llama) so
/// family-specific constructs accrete in one visible place.
fn gqa_lm_new(
    vocab: usize,
    d: usize,
    ff: usize,
    n_heads: usize,
    n_kv_heads: usize,
    layers: usize,
    ffn: GatedFfn,
    qk_norm: bool,
    cx: &mut Graph,
) -> (Embedding, Vec<LlamaBlock>, crate::LayerNorm) {
    let blocks = (0..layers)
        .map(|_| {
            let block = LlamaBlock::new_with_ffn(d, ff, n_heads, n_kv_heads, ffn, cx);
            if qk_norm { block.with_qk_norm(cx) } else { block }
        })
        .collect();
    (
        Embedding::new(vocab, d, cx),
        blocks,
        crate::LayerNorm::new(d, None, None, false, 1e-5, cx),
    )
}

fn gqa_lm_forward(
    embed: &Embedding,
    blocks: &[LlamaBlock],
    final_norm: &crate::LayerNorm,
    ids: GraphTensor,
    caches: &[(GraphTensor, GraphTensor)],
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: Expression,
) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
    let mut x = embed.forward(ids);
    let mut caches_out = Vec::with_capacity(blocks.len());
    for (layer, block) in blocks.iter().enumerate() {
        let (next, kc, vc) = block.forward(
            x,
            caches[layer].0,
            caches[layer].1,
            gather_idx,
            scatter_idx,
            prev_seq,
        );
        x = next;
        caches_out.push((kc, vc));
    }
    let logits = embed.reverse(final_norm.forward(x));
    (logits, caches_out)
}

macro_rules! gqa_lm_forward_impl {
    () => {
        /// ids (s,) Int + one (k, v) cache pair per layer → (logits, caches').
        pub fn forward(
            &self,
            ids: GraphTensor,
            caches: &[(GraphTensor, GraphTensor)],
            gather_idx: GraphTensor,
            scatter_idx: GraphTensor,
            prev_seq: Expression,
        ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
            gqa_lm_forward(
                &self.embed,
                &self.blocks,
                &self.final_norm,
                ids,
                caches,
                gather_idx,
                scatter_idx,
                prev_seq,
            )
        }
    };
}

/// MiniLlama3 — the llama/paged_llama family: RMS pre-norms, GQA over a
/// paged KV cache, SwiGLU. (RoPE deferred by ruling, as everywhere.)
pub struct MiniLlama3 {
    pub embed: Embedding,
    pub blocks: Vec<LlamaBlock>,
    pub final_norm: crate::LayerNorm,
}

impl MiniLlama3 {
    pub fn new(
        vocab: usize,
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        layers: usize,
        cx: &mut Graph,
    ) -> Self {
        let (embed, blocks, final_norm) =
            gqa_lm_new(vocab, d, ff, n_heads, n_kv_heads, layers, GatedFfn::SwiGlu, false, cx);
        Self { embed, blocks, final_norm }
    }
    gqa_lm_forward_impl!();
}

/// MiniQwen3 — the qwen family: the llama-3 skeleton plus Qwen3's
/// per-head QK RMSNorm on q/k (the construct the qwen example adds).
pub struct MiniQwen3 {
    pub embed: Embedding,
    pub blocks: Vec<LlamaBlock>,
    pub final_norm: crate::LayerNorm,
}

impl MiniQwen3 {
    pub fn new(
        vocab: usize,
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        layers: usize,
        cx: &mut Graph,
    ) -> Self {
        let (embed, blocks, final_norm) =
            gqa_lm_new(vocab, d, ff, n_heads, n_kv_heads, layers, GatedFfn::SwiGlu, true, cx);
        Self { embed, blocks, final_norm }
    }
    gqa_lm_forward_impl!();
}

/// MiniGemma3 — the gemma family at FULL ANATOMY (ruling 2026-08-10:
/// minis exercise every architectural construct, shrinking only
/// shapes). Beyond the skeleton: √d EMBEDDING SCALING in-graph with the
/// tied lm_head left UNSCALED; alternating LOCAL (sliding-window,
/// θ=10k) and GLOBAL (full-context, θ=1M, pos·⅛ scaling) layers; and
/// per layer the whole [`GemmaBlock`] construct set — sandwich norms,
/// decoupled head_dim, QK-norm, scale-folded-into-Q, in-graph
/// split-half RoPE, GeGLU. `pattern` = every pattern-th layer is
/// global (gemma uses 6; the ratio is a shape, the alternation the
/// construct).
pub struct MiniGemma3 {
    pub embed: Embedding,
    pub blocks: Vec<crate::GemmaBlock>,
    pub final_norm: crate::LayerNorm,
    d: usize,
}

impl MiniGemma3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        layers: usize,
        window: usize,
        pattern: usize,
        cx: &mut Graph,
    ) -> Self {
        Self {
            embed: Embedding::new(vocab, d, cx),
            blocks: (0..layers)
                .map(|layer| {
                    let local = (layer + 1) % pattern != 0;
                    crate::GemmaBlock::new(d, ff, n_heads, n_kv_heads, head_dim, local, window, cx)
                })
                .collect(),
            final_norm: crate::LayerNorm::new(d, Some("NormWeight"), None, false, 1e-6, cx),
            d,
        }
    }

    /// ids (s,) Int, per-layer caches, per-layer rope tables (cos, sin)
    /// — host-built from each block's role theta/pos_scale — plus the
    /// shared split-half pairing matrix → (logits, caches'). Embeddings
    /// scale by √d in-graph; the tied logits head reads the UNSCALED
    /// table (gemma's convention).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: Expression,
        rope: &[(GraphTensor, GraphTensor)],
        rope_rot: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = self.embed.forward(ids) * (self.d as f32).sqrt();
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, kc, vc) = block.forward(
                x,
                caches[layer].0,
                caches[layer].1,
                gather_idx,
                scatter_idx,
                prev_seq,
                rope[layer].0,
                rope[layer].1,
                rope_rot,
            );
            x = next;
            caches_out.push((kc, vc));
        }
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}

/// Shared MoE-decoder assembly (the [`MoE`] layer itself lives in
/// luminal_nn — the minis only assemble it): embed → N × DecoderBlock
/// with the MoE feed-forward → final LayerNorm → tied logits.
fn moe_lm_new(
    vocab: usize,
    d: usize,
    experts: usize,
    top_k: usize,
    n_heads: usize,
    layers: usize,
    cx: &mut Graph,
) -> (Embedding, Vec<DecoderBlock>, crate::LayerNorm) {
    (
        Embedding::new(vocab, d, cx),
        (0..layers)
            .map(|_| DecoderBlock {
                wq: Linear::new(d, d, false, cx),
                wk: Linear::new(d, d, false, cx),
                wv: Linear::new(d, d, false, cx),
                wo: Linear::new(d, d, false, cx),
                ff: FeedForward::Moe(MoE {
                    expert_weights: cx.named_tensor("Experts", (experts, d, d)),
                    router: cx.named_tensor("Router", (d, experts)),
                    k: top_k,
                }),
                n_heads,
                n_kv_heads: n_heads,
                head_dim: d / n_heads,
            })
            .collect(),
        crate::LayerNorm::new(d, None, None, true, 1e-5, cx),
    )
}

fn moe_lm_forward(
    embed: &Embedding,
    blocks: &[DecoderBlock],
    final_norm: &crate::LayerNorm,
    ids: GraphTensor,
    caches: &[(GraphTensor, GraphTensor)],
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: Expression,
) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
    let mut x = embed.forward(ids);
    let mut caches_out = Vec::with_capacity(blocks.len());
    for (layer, block) in blocks.iter().enumerate() {
        let (next, kc, vc) = block.forward(
            x,
            caches[layer].0,
            caches[layer].1,
            gather_idx,
            scatter_idx,
            prev_seq,
        );
        x = next;
        caches_out.push((kc, vc));
    }
    let logits = embed.reverse(final_norm.forward(x));
    (logits, caches_out)
}

/// MiniQwen3Moe — the qwen3_moe family: MoE decoder blocks over the
/// paged cache. HONEST GAP: the example also carries QK-norm and
/// top-k=8 routing with renormalized weights (mini routes k=1); those
/// ride the pending fidelity ruling.
pub struct MiniQwen3Moe {
    pub embed: Embedding,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: crate::LayerNorm,
}

impl MiniQwen3Moe {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        d: usize,
        experts: usize,
        top_k: usize,
        n_heads: usize,
        layers: usize,
        cx: &mut Graph,
    ) -> Self {
        let (embed, blocks, final_norm) =
            moe_lm_new(vocab, d, experts, top_k, n_heads, layers, cx);
        Self { embed, blocks, final_norm }
    }

    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: Expression,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        moe_lm_forward(
            &self.embed,
            &self.blocks,
            &self.final_norm,
            ids,
            caches,
            gather_idx,
            scatter_idx,
            prev_seq,
        )
    }
}

/// MiniGemma4Moe — the gemma4_moe family: the MoE decoder plus the
/// construct only this family has, FINAL LOGIT SOFT-CAPPING:
/// `tanh(logits / cap) · cap`. HONEST GAP: sandwich norms and QK-norm
/// from the example ride the pending fidelity ruling.
pub struct MiniGemma4Moe {
    pub embed: Embedding,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: crate::LayerNorm,
    pub logit_softcap: f32,
}

impl MiniGemma4Moe {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        d: usize,
        experts: usize,
        top_k: usize,
        n_heads: usize,
        layers: usize,
        cx: &mut Graph,
    ) -> Self {
        let (embed, blocks, final_norm) =
            moe_lm_new(vocab, d, experts, top_k, n_heads, layers, cx);
        Self { embed, blocks, final_norm, logit_softcap: 30.0 }
    }

    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: Expression,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let (logits, caches_out) = moe_lm_forward(
            &self.embed,
            &self.blocks,
            &self.final_norm,
            ids,
            caches,
            gather_idx,
            scatter_idx,
            prev_seq,
        );
        let capped = (logits * (1.0 / self.logit_softcap)).tanh() * self.logit_softcap;
        (capped, caches_out)
    }
}

/// The whisper-family mini: one encoder block (bidirectional
/// self-attention, GELU FFN) and one decoder cross-attention block that
/// attends over the encoder output — the construct nothing else covers.
pub struct MiniWhisper {
    pub enc_norm: crate::LayerNorm,
    pub enc_wq: Linear,
    pub enc_wk: Linear,
    pub enc_wv: Linear,
    pub enc_wo: Linear,
    pub enc_up: Linear,
    pub enc_down: Linear,
    pub dec_norm: crate::LayerNorm,
    pub dec_wq: Linear,
    pub dec_wk: Linear,
    pub dec_wv: Linear,
    pub dec_wo: Linear,
    pub dec_up: Linear,
    pub dec_down: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl MiniWhisper {
    pub fn new(d: usize, ff: usize, n_heads: usize, cx: &mut Graph) -> Self {
        let linear = |a, b, cx: &mut Graph| Linear::new(a, b, false, cx);
        Self {
            enc_norm: crate::LayerNorm::new(d, None, None, true, 1e-5, cx),
            enc_wq: linear(d, d, cx),
            enc_wk: linear(d, d, cx),
            enc_wv: linear(d, d, cx),
            enc_wo: linear(d, d, cx),
            enc_up: linear(d, ff, cx),
            enc_down: linear(ff, d, cx),
            dec_norm: crate::LayerNorm::new(d, None, None, true, 1e-5, cx),
            dec_wq: linear(d, d, cx),
            dec_wk: linear(d, d, cx),
            dec_wv: linear(d, d, cx),
            dec_wo: linear(d, d, cx),
            dec_up: linear(d, ff, cx),
            dec_down: linear(ff, d, cx),
            n_heads,
            head_dim: d / n_heads,
        }
    }

    /// audio (s_enc, d) + token activations (s_dec, d) → (s_dec, d).
    pub fn forward(&self, audio: GraphTensor, tokens: GraphTensor) -> GraphTensor {
        // Encoder block: pre-LN self-attention + GELU FFN, residuals.
        let normed = self.enc_norm.forward(audio);
        let self_attn = attention(
            self.enc_wq.forward(normed),
            self.enc_wk.forward(normed),
            self.enc_wv.forward(normed),
            self.n_heads,
            self.head_dim,
        );
        let enc = audio + self.enc_wo.forward(self_attn);
        let enc = enc
            + self
                .enc_down
                .forward(self.enc_up.forward(enc).gelu_fast_tanh_approximation());

        // Decoder cross-attention block: queries from tokens, keys and
        // values from the ENCODER OUTPUT.
        let normed = self.dec_norm.forward(tokens);
        let cross = attention(
            self.dec_wq.forward(normed),
            self.dec_wk.forward(enc),
            self.dec_wv.forward(enc),
            self.n_heads,
            self.head_dim,
        );
        let x = tokens + self.dec_wo.forward(cross);
        x + self
            .dec_down
            .forward(self.dec_up.forward(x).gelu_fast_tanh_approximation())
    }
}

/// The yolo-family mini: two valid-padding conv layers with relu, then a
/// linear classification head over the flattened features.
pub struct MiniConvNet {
    pub conv1: ConvND,
    pub conv2: ConvND,
    pub head: Linear,
    classes: usize,
}

impl MiniConvNet {
    /// Input (ch_in, h, w) with h = w = 5 and 3×3 valid convs: 5→3→1.
    pub fn new(ch_in: usize, c1: usize, c2: usize, classes: usize, cx: &mut Graph) -> Self {
        Self {
            conv1: ConvND::new(ch_in, c1, [3, 3], [1, 1], [1, 1], [0, 0], false, cx),
            conv2: ConvND::new(c1, c2, [3, 3], [1, 1], [1, 1], [0, 0], false, cx),
            head: Linear::new(c2, classes, false, cx),
            classes,
        }
    }

    /// x (1, ch_in, 5, 5) → logits (classes,).
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = self.conv1.forward(x).relu(); // (1, c1, 3, 3)
        let x = self.conv2.forward(x).relu(); // (1, c2, 1, 1)
        let flat = x.flatten(); // (c2,)
        let logits = self.head.forward(flat.expand_lhs(1)); // (1, classes)
        let _ = self.classes;
        logits.squeeze(0)
    }
}

/// MiniDit — the flux2 family mini. Family-unique constructs carried
/// faithfully: sinusoidal t/guidance conditioning summed through SiLU
/// MLPs; SHARED adaLN modulation tables cut into (shift, scale, gate)
/// triples; gated residuals `x += gate ⊙ sublayer`; no-affine
/// LayerNorms; one DOUBLE-stream block (separate img/txt weights, one
/// joint attention over [txt ‖ img]); one SINGLE-stream block (fused
/// qkv+mlp in-projection, fused out-projection over [attn ‖ mlp]);
/// per-head QK RMSNorm; interleaved-pair multi-axis RoPE from host
/// tables; non-causal maskless SDPA; AdaLayerNormContinuous head with
/// the REVERSED (scale, shift) order. Patchify/VAE/scheduler are
/// host-side in the flux2 example and stay outside the family constructs.
pub struct MiniDit {
    pub x_embed: Linear,
    pub ctx_embed: Linear,
    pub t_mlp1: Linear,
    pub t_mlp2: Linear,
    pub g_mlp1: Linear,
    pub g_mlp2: Linear,
    pub mod_img: Linear,
    pub mod_txt: Linear,
    pub mod_single: Linear,
    pub norm_out: Linear,
    pub proj_out: Linear,
    pub img_q: Linear,
    pub img_k: Linear,
    pub img_v: Linear,
    pub img_out: Linear,
    pub txt_q: Linear,
    pub txt_k: Linear,
    pub txt_v: Linear,
    pub txt_out: Linear,
    pub img_qnorm: GraphTensor,
    pub img_knorm: GraphTensor,
    pub txt_qnorm: GraphTensor,
    pub txt_knorm: GraphTensor,
    pub ff_in: Linear,
    pub ff_out: Linear,
    pub ctx_ff_in: Linear,
    pub ctx_ff_out: Linear,
    pub single_proj: Linear,
    /// The single-stream out-projection, SPLIT into its attn-rows and
    /// mlp-rows halves: out = attn·W_attn + mlp·W_mlp — algebraically
    /// identical to flux2's fused `to_out @ [attn ‖ mlp]`, spelled
    /// without the concat (rejoin-divergence workaround; the fused
    /// spelling returns with the divergence ruling).
    pub single_out_attn: Linear,
    pub single_out_mlp: Linear,
    pub single_qnorm: GraphTensor,
    pub single_knorm: GraphTensor,
    ln: crate::LayerNorm, // no-affine LayerNorm, shared (stateless)
    d: usize,
    n_heads: usize,
    head_dim: usize,
    mlp: usize,
    t_half: usize,
    s_txt: usize,
}

impl MiniDit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        txt_dim: usize,
        d: usize,
        n_heads: usize,
        mlp: usize,
        t_half: usize,
        s_txt: usize,
        cx: &mut Graph,
    ) -> Self {
        let head_dim = d / n_heads;
        Self {
            x_embed: Linear::new(in_channels, d, false, cx),
            ctx_embed: Linear::new(txt_dim, d, false, cx),
            t_mlp1: Linear::new(2 * t_half, d, false, cx),
            t_mlp2: Linear::new(d, d, false, cx),
            g_mlp1: Linear::new(2 * t_half, d, false, cx),
            g_mlp2: Linear::new(d, d, false, cx),
            mod_img: Linear::new(d, 6 * d, false, cx),
            mod_txt: Linear::new(d, 6 * d, false, cx),
            mod_single: Linear::new(d, 3 * d, false, cx),
            norm_out: Linear::new(d, 2 * d, false, cx),
            proj_out: Linear::new(d, in_channels, false, cx),
            img_q: Linear::new(d, d, false, cx),
            img_k: Linear::new(d, d, false, cx),
            img_v: Linear::new(d, d, false, cx),
            img_out: Linear::new(d, d, false, cx),
            txt_q: Linear::new(d, d, false, cx),
            txt_k: Linear::new(d, d, false, cx),
            txt_v: Linear::new(d, d, false, cx),
            txt_out: Linear::new(d, d, false, cx),
            img_qnorm: cx.named_tensor("ImgQNorm", head_dim),
            img_knorm: cx.named_tensor("ImgKNorm", head_dim),
            txt_qnorm: cx.named_tensor("TxtQNorm", head_dim),
            txt_knorm: cx.named_tensor("TxtKNorm", head_dim),
            ff_in: Linear::new(d, 2 * mlp, false, cx),
            ff_out: Linear::new(mlp, d, false, cx),
            ctx_ff_in: Linear::new(d, 2 * mlp, false, cx),
            ctx_ff_out: Linear::new(mlp, d, false, cx),
            single_proj: Linear::new(d, 3 * d + 2 * mlp, false, cx),
            single_out_attn: Linear::new(d, d, false, cx),
            single_out_mlp: Linear::new(mlp, d, false, cx),
            single_qnorm: cx.named_tensor("SglQNorm", head_dim),
            single_knorm: cx.named_tensor("SglKNorm", head_dim),
            ln: crate::LayerNorm::new(d, None, None, true, 1e-6, cx),
            d,
            n_heads,
            head_dim,
            mlp,
            t_half,
            s_txt,
        }
    }

    /// Sinusoidal embedding of a (1,) scalar: [cos(1000x·fᵢ) ‖ sin(1000x·fᵢ)],
    /// fᵢ = 10000^(-i/half) — flip_sin_to_cos ordering, as flux2.
    fn sinusoid(&self, x: GraphTensor) -> GraphTensor {
        let mut cos_parts = Vec::with_capacity(self.t_half);
        let mut sin_parts = Vec::with_capacity(self.t_half);
        for i in 0..self.t_half {
            let freq = (-(i as f32) * (10000f32).ln() / self.t_half as f32).exp();
            let arg = x * (1000.0 * freq);
            cos_parts.push(arg.cos());
            sin_parts.push(arg.sin());
        }
        let mut parts = cos_parts;
        parts.extend(sin_parts);
        let mut cat = parts[0];
        for part in &parts[1..] {
            cat = cat.concat_along(*part, 0);
        }
        cat.unsqueeze(0) // (1, 2·half)
    }

    /// latent (s_img, in_ch), text (s_txt, txt_dim), t (1,), guidance (1,),
    /// rope tables (s_txt+s_img, head_dim), the interleaved pairing
    /// matrix (head_dim, head_dim), and a zeros base (s_txt+s_img, d)
    /// for the scatter-assembled joint sequence → velocity (s_img, in_ch).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: GraphTensor,
        text: GraphTensor,
        t: GraphTensor,
        guidance: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
        joint_base: GraphTensor,
    ) -> GraphTensor {
        let (d, mlp, s_txt) = (self.d, self.mlp, self.s_txt);
        let temb = self.t_mlp2.forward(self.t_mlp1.forward(self.sinusoid(t)).silu())
            + self.g_mlp2.forward(self.g_mlp1.forward(self.sinusoid(guidance)).silu()); // (1, d)
        let cond = temb.silu();
        let m_img = self.mod_img.forward(cond); // (1, 6d): 2 × (shift, scale, gate)
        let m_txt = self.mod_txt.forward(cond);
        let m_single = self.mod_single.forward(cond); // (1, 3d)
        let triple = |m: GraphTensor, set: usize| {
            let base = set * 3 * d;
            (
                m.slice_along(base..base + d, 1),          // shift
                m.slice_along(base + d..base + 2 * d, 1),  // scale
                m.slice_along(base + 2 * d..base + 3 * d, 1), // gate
            )
        };
        let ada = |x: GraphTensor, scale: GraphTensor, shift: GraphTensor| {
            let dims = x.dims();
            x * (scale + 1.0).expand(dims.clone()) + shift.expand(dims)
        };
        let gate = |x: GraphTensor, g: GraphTensor| {
            let dims = x.dims();
            x * g.expand(dims)
        };
        let heads = |x: GraphTensor| x.split_dims(1, self.head_dim).permute((1, 0, 2)); // (H,S,hd)
        let unheads = |x: GraphTensor| x.permute((1, 0, 2)).merge_dims(1, 2); // (S,d)
        let head_rms = |x: GraphTensor, weight: GraphTensor| {
            let dims = x.dims();
            let inv = ((x * x).mean(2) + 1e-6).sqrt().reciprocal(); // (H,S)
            x * inv.unsqueeze(2).expand(dims.clone())
                * weight.unsqueeze(0).unsqueeze(0).expand(dims)
        };
        let rope = |x: GraphTensor| {
            // Interleaved-pair rotation via the pairing matrix — the
            // concat-free spelling (rejoin-divergence workaround):
            // rope(x) = x ⊙ cos + (x @ R) ⊙ sin on (H, S, hd).
            let dims = x.dims();
            let rotated = x.matmul(rope_rot);
            x * rope_cos.unsqueeze(0).expand(dims.clone())
                + rotated * rope_sin.unsqueeze(0).expand(dims)
        };
        let sdpa = |q: GraphTensor, k: GraphTensor, v: GraphTensor| {
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let scores = q.matmul(k.permute((0, 2, 1))) * scale; // (H,S,S)
            scores.softmax(2).matmul(v) // (H,S,hd)
        };
        let swiglu = |u: GraphTensor| {
            u.slice_along(0..mlp, 1).silu() * u.slice_along(mlp..2 * mlp, 1)
        };

        // ---- double-stream block (txt first in every concat/split) ----
        let (shift0, scale0, gate0) = triple(m_img, 0);
        let (shift1, scale1, gate1) = triple(m_img, 1);
        let (c_shift0, c_scale0, c_gate0) = triple(m_txt, 0);
        let (c_shift1, c_scale1, c_gate1) = triple(m_txt, 1);
        let mut img = self.x_embed.forward(latent); // (s_img, d)
        let mut txt = self.ctx_embed.forward(text); // (s_txt, d)
        let img_n = ada(self.ln.forward(img), scale0, shift0);
        let txt_n = ada(self.ln.forward(txt), c_scale0, c_shift0);
        let q_img = head_rms(heads(self.img_q.forward(img_n)), self.img_qnorm);
        let k_img = head_rms(heads(self.img_k.forward(img_n)), self.img_knorm);
        let q_txt = head_rms(heads(self.txt_q.forward(txt_n)), self.txt_qnorm);
        let k_txt = head_rms(heads(self.txt_k.forward(txt_n)), self.txt_knorm);
        // V's sequence concat happens FLAT, before the head split — the
        // head reshape commutes with row concat, and pads over matmul
        // outputs (compute) never form the pure-view stack the rejoin
        // divergence needs. q/k concat after head_rms (a compute) for
        // the same reason.
        let v_all = heads(
            self.txt_v
                .forward(txt_n)
                .concat_along(self.img_v.forward(img_n), 0),
        );
        let attn = unheads(sdpa(
            rope(q_txt.concat_along(q_img, 1)),
            rope(k_txt.concat_along(k_img, 1)),
            v_all,
        )); // (s, d)
        let attn_txt = attn.slice_along(0..s_txt, 0);
        let attn_img = attn.slice_along(s_txt.., 0);
        img = img + gate(self.img_out.forward(attn_img), gate0);
        txt = txt + gate(self.txt_out.forward(attn_txt), c_gate0);
        let ff = swiglu(self.ff_in.forward(ada(self.ln.forward(img), scale1, shift1)));
        img = img + gate(self.ff_out.forward(ff), gate1);
        let c_ff = swiglu(self.ctx_ff_in.forward(ada(self.ln.forward(txt), c_scale1, c_shift1)));
        txt = txt + gate(self.ctx_ff_out.forward(c_ff), c_gate1);

        // ---- single-stream block over [txt ‖ img] ----
        // The joint sequence assembles by SCATTER writes into a zero
        // base (the paged-attention family's own row-assembly spelling)
        // instead of concat's pad+add: the head SLICES this tensor, and
        // a slice distributing down to a pad's clamp view re-creates
        // the rejoin-divergence stack (measured: stage-8 probe). Scatter
        // is a compute write — the slice stops there.
        let graph = latent.graph();
        let txt_positions = graph.arange(s_txt);
        let img_positions = graph.iota(latent.dims()[0], move |c| c[0] + s_txt);
        let mut hidden = crate::scatter_rows(
            img,
            img_positions,
            crate::scatter_rows(txt, txt_positions, joint_base, d),
            d,
        ); // (s, d)
        let (s_shift, s_scale, s_gate) = triple(m_single, 0);
        let normed = ada(self.ln.forward(hidden), s_scale, s_shift);
        let proj = self.single_proj.forward(normed); // (s, 3d + 2·mlp)
        let q = head_rms(heads(proj.slice_along(0..d, 1)), self.single_qnorm);
        let k = head_rms(heads(proj.slice_along(d..2 * d, 1)), self.single_knorm);
        let v = heads(proj.slice_along(2 * d..3 * d, 1));
        let attn = unheads(sdpa(rope(q), rope(k), v)); // (s, d)
        let mlp_out = swiglu(proj.slice_along(3 * d..3 * d + 2 * mlp, 1)); // (s, mlp)
        // Fused out-projection over [attn ‖ mlp], spelled as the
        // row-split sum (see the single_out_* field note).
        hidden = hidden
            + gate(
                self.single_out_attn.forward(attn) + self.single_out_mlp.forward(mlp_out),
                s_gate,
            );

        // ---- AdaLayerNormContinuous head: (scale, shift) — REVERSED ----
        let img_final = hidden.slice_along(s_txt.., 0); // (s_img, d)
        let head = self.norm_out.forward(cond); // (1, 2d)
        let scale = head.slice_along(0..d, 1);
        let shift = head.slice_along(d..2 * d, 1);
        self.proj_out
            .forward(ada(self.ln.forward(img_final), scale, shift))
    }
}

/// Host-side interleaved-pair RoPE tables for the mini DiT grid
/// (mirrors flux2's host-precomputed tables): rows are the s_txt text
/// tokens with ids (0,0,0,ℓ), then the h·w image tokens with ids
/// (0,hi,wi,0) row-major. Four axes × 2 dims each (half = 1 per axis, so
/// the frequency is θ⁰ = 1 and θ drops out); every cos/sin value is
/// written twice — the repeat_interleave that matches adjacent-pair
/// rotation. Tables are (s_txt + h·w, 8); head_dim must be 8.
pub fn mini_dit_rope_tables(s_txt: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut ids: Vec<[f32; 4]> = (0..s_txt).map(|l| [0.0, 0.0, 0.0, l as f32]).collect();
    for hi in 0..h {
        for wi in 0..w {
            ids.push([0.0, hi as f32, wi as f32, 0.0]);
        }
    }
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for id in ids {
        for axis in 0..4 {
            for _ in 0..2 {
                cos.push(id[axis].cos());
                sin.push(id[axis].sin());
            }
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::{
        attention, mini_dit_rope_tables, MiniConvNet, MiniDit, MiniGemma3, MiniGemma4Moe,
        MiniLlama3, MiniQwen3, MiniQwen3Moe, MiniWhisper,
    };
    use crate::test_refs::*;
    use luminal::prelude::*;
    use luminal::shape::Expression;

    fn ref_gelu_tanh(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|v| {
                let scaled = 1.5957691216 * v * (1.0 + 0.044715 * v * v);
                v / (1.0 + (-scaled).exp())
            })
            .collect()
    }

    /// Plain multi-head attention reference: q rows attend over all k/v
    /// rows, per head.
    fn ref_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sq: usize,
        sk: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let d = n_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0f32; sq * d];
        for row in 0..sq {
            for h in 0..n_heads {
                let q_h = &q[row * d + h * head_dim..][..head_dim];
                let scores: Vec<f32> = (0..sk)
                    .map(|col| {
                        let k_h = &k[col * d + h * head_dim..][..head_dim];
                        q_h.iter().zip(k_h).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let denom: f32 = exps.iter().sum();
                for (col, e) in exps.iter().enumerate() {
                    let v_h = &v[col * d + h * head_dim..][..head_dim];
                    for (dim, value) in v_h.iter().enumerate() {
                        out[row * d + h * head_dim + dim] += e / denom * value;
                    }
                }
            }
        }
        out
    }

    /// Per-head RMS norm over a flat (groups·head_dim) row with a learned
    /// (head_dim,) weight — the QK-norm reference (eps 1e-6, matching
    /// `rms_norm_heads`).
    fn ref_rms_head_norm(x: &[f32], head_dim: usize, w: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for head in x.chunks(head_dim) {
            let ms: f32 = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            out.extend(head.iter().zip(w).map(|(v, weight)| v * inv * weight));
        }
        out
    }

    /// One llama-block reference step (rms → GQA attention → residual →
    /// rms → gated FFN → residual), gate activation supplied; optional
    /// Qwen3-style QK-norm weight seeds.
    #[allow(clippy::too_many_arguments)]
    fn ref_llama_block(
        x: &[f32],
        seeds: (usize, usize, usize, usize, usize, usize, usize),
        qk_seeds: Option<(usize, usize)>,
        d: usize,
        ff: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        gather: &[usize],
        scatter_slot: usize,
        gate_act: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<f32> {
        let (wq_s, wk_s, wv_s, wo_s, gate_s, up_s, down_s) = seeds;
        let normed = ref_rms_norm(x, 1e-5);
        let mut q = ref_matmul(&normed, &weights(d * d, wq_s), d, d);
        let mut k_new = ref_matmul(&normed, &weights(d * kv_dim, wk_s), d, kv_dim);
        if let Some((q_seed, k_seed)) = qk_seeds {
            q = ref_rms_head_norm(&q, head_dim, &weights(head_dim, q_seed));
            k_new = ref_rms_head_norm(&k_new, head_dim, &weights(head_dim, k_seed));
        }
        let v_new = ref_matmul(&normed, &weights(d * kv_dim, wv_s), d, kv_dim);
        let attn = ref_paged_step_gqa(
            &q,
            &k_new,
            &v_new,
            k_cache,
            v_cache,
            gather,
            scatter_slot,
            n_heads,
            n_kv,
            head_dim,
            scatter_slot,
            None,
            1.0 / (head_dim as f32).sqrt(),
        );
        let attn_proj = ref_matmul(&attn, &weights(d * d, wo_s), d, d);
        let x1: Vec<f32> = x.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();
        let ff_in = ref_rms_norm(&x1, 1e-5);
        let gate = gate_act(&ref_matmul(&ff_in, &weights(d * ff, gate_s), d, ff));
        let up = ref_matmul(&ff_in, &weights(d * ff, up_s), d, ff);
        let hidden: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| a * b).collect();
        let ffo = ref_matmul(&hidden, &weights(ff * d, down_s), ff, d);
        x1.iter().zip(&ffo).map(|(a, b)| a + b).collect()
    }

    fn seeds_for(layer: usize) -> (usize, usize, usize, usize, usize, usize, usize) {
        let b = 200 + layer * 10;
        (b, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6)
    }

    /// Family harness: one NAMED GQA-decoder mini (ruling 2026-08-10) —
    /// TWO blocks, one decode step, default search budget. Depth was
    /// pinned at one block while random genomes could choice-cycle;
    /// two-phase sampling (2026-08-07) made copy welds unconstructible.
    fn mini_gqa_family(family: &str, gate_act: &dyn Fn(&[f32]) -> Vec<f32>) {
        const VOCAB: usize = 5;
        const D: usize = 8;
        const FF: usize = 12;
        const NH: usize = 4;
        const NKV: usize = 2;
        const HD: usize = 2;
        const KV_DIM: usize = NKV * HD;
        const SLOTS: usize = 4;
        const CTX: usize = 2;
        const LAYERS: usize = 2;
        let token = 3usize;

        let mut cx = Graph::new();
        let ids = cx.tensor_dtyped(1, DType::Int);
        let caches: Vec<_> = (0..LAYERS)
            .map(|_| (cx.tensor((SLOTS, KV_DIM)), cx.tensor((SLOTS, KV_DIM))))
            .collect();
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let step = Expression::from(1usize);
        let (logits, caches_out, embed, blocks) = match family {
            "llama3" => {
                let model = MiniLlama3::new(VOCAB, D, FF, NH, NKV, LAYERS, &mut cx);
                let (logits, caches_out) =
                    model.forward(ids, &caches, gather_idx, scatter_idx, step);
                (logits, caches_out, model.embed, model.blocks)
            }
            "qwen3" => {
                let model = MiniQwen3::new(VOCAB, D, FF, NH, NKV, LAYERS, &mut cx);
                let (logits, caches_out) =
                    model.forward(ids, &caches, gather_idx, scatter_idx, step);
                (logits, caches_out, model.embed, model.blocks)
            }
            other => panic!("unknown mini family {other}"),
        };
        let logits = logits.output();
        let caches_out: Vec<_> = caches_out
            .into_iter()
            .map(|(k, v)| (k.output(), v.output()))
            .collect();

        let embed_w = weights(VOCAB * D, 199);
        let mut pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (ids.id, vec![token as f32]),
            (embed.weight.id, embed_w.clone()),
            (gather_idx.id, vec![0.0, 1.0]),
            (scatter_idx.id, vec![1.0]),
        ];
        let qk_seeds_for = |layer: usize| (260 + layer * 2, 261 + layer * 2);
        let mut ref_caches: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for (layer, block) in blocks.iter().enumerate() {
            let (wq_s, wk_s, wv_s, wo_s, gate_s, up_s, down_s) = seeds_for(layer);
            pairs.push((block.wq.weight.id, weights(D * D, wq_s)));
            pairs.push((block.wk.weight.id, weights(D * KV_DIM, wk_s)));
            pairs.push((block.wv.weight.id, weights(D * KV_DIM, wv_s)));
            pairs.push((block.wo.weight.id, weights(D * D, wo_s)));
            pairs.push((block.gate.weight.id, weights(D * FF, gate_s)));
            pairs.push((block.up.weight.id, weights(D * FF, up_s)));
            pairs.push((block.down.weight.id, weights(FF * D, down_s)));
            if let Some((q_norm, k_norm)) = block.qk_norm {
                let (q_seed, k_seed) = qk_seeds_for(layer);
                pairs.push((q_norm.id, weights(HD, q_seed)));
                pairs.push((k_norm.id, weights(HD, k_seed)));
            }
            let kc = weights(SLOTS * KV_DIM, 300 + layer);
            let vc = weights(SLOTS * KV_DIM, 320 + layer);
            pairs.push((caches[layer].0.id, kc.clone()));
            pairs.push((caches[layer].1.id, vc.clone()));
            ref_caches.push((kc, vc));
        }

        // Scalar reference.
        let mut x: Vec<f32> = embed_w[token * D..(token + 1) * D].to_vec();
        for layer in 0..LAYERS {
            let (kc, vc) = &mut ref_caches[layer];
            let qk_seeds = blocks[layer].qk_norm.map(|_| qk_seeds_for(layer));
            x = ref_llama_block(
                &x,
                seeds_for(layer),
                qk_seeds,
                D,
                FF,
                KV_DIM,
                NH,
                NKV,
                HD,
                kc,
                vc,
                &[0, 1],
                1,
                gate_act,
            );
        }
        let x = ref_rms_norm(&x, 1e-5);
        let ref_logits: Vec<f32> = (0..VOCAB)
            .map(|v| (0..D).map(|i| x[i] * embed_w[v * D + i]).sum())
            .collect();

        let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
        let mut rt = luminal::ssa_reference::SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(
            &data,
            &luminal::implementation_search::ImplementationSearchOptions::default(),
        )
        .expect("search finds a plan");
        for (id, values) in &pairs {
            rt.set_data(*id, values.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
        for layer in 0..LAYERS {
            assert_close(rt.get_f32(caches_out[layer].0.id).unwrap(), &ref_caches[layer].0);
            assert_close(rt.get_f32(caches_out[layer].1.id).unwrap(), &ref_caches[layer].1);
        }
    }

    #[test]
    fn mini_llama3_matches_scalar_reference() {
        mini_gqa_family("llama3", &|x| {
            x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
        });
    }

    #[test]
    fn mini_qwen3_matches_scalar_reference() {
        mini_gqa_family("qwen3", &|x| {
            x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
        });
    }

    /// Table-and-matrix rope reference — mirror of `rotary_apply`:
    /// out = x ⊙ cos + (x @ R) ⊙ sin, per head, single row.
    fn ref_rotary_apply(
        x: &[f32],
        head_dim: usize,
        cos: &[f32],
        sin: &[f32],
        rot: &[f32],
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for head in x.chunks(head_dim) {
            for i in 0..head_dim {
                let rotated: f32 = (0..head_dim)
                    .map(|source| head[source] * rot[source * head_dim + i])
                    .sum();
                out.push(head[i] * cos[i] + rotated * sin[i]);
            }
        }
        out
    }

    /// MiniGemma3 at FULL gemma anatomy vs a complete scalar reference:
    /// two layers (layer 0 LOCAL: window mask + θ=10k; layer 1 GLOBAL:
    /// θ=1M + pos·⅛ scaling), sandwich norms, decoupled head_dim
    /// (n_heads·head_dim = 8 ≠ d = 6), QK-norm, scale folded into Q,
    /// in-graph rope, GeGLU, √d embedding scaling with unscaled tied
    /// head. WINDOW = 1 so the local mask provably bites (gathered
    /// position 0 masked at q_pos = 1).
    #[test]
    fn mini_gemma3_matches_scalar_reference() {
        const VOCAB: usize = 5;
        const D: usize = 6;
        const FF: usize = 8;
        const NH: usize = 2;
        const NKV: usize = 1;
        const HD: usize = 4; // q_dim = 8 ≠ d = 6 — decoupled
        const Q_DIM: usize = NH * HD;
        const KV_DIM: usize = NKV * HD;
        const SLOTS: usize = 4;
        const CTX: usize = 2;
        const LAYERS: usize = 2;
        const WINDOW: usize = 1;
        const PATTERN: usize = 2; // layer 0 local, layer 1 global
        let token = 3usize;
        let q_pos = 1usize;

        let mut cx = Graph::new();
        let ids = cx.tensor_dtyped(1, DType::Int);
        let caches: Vec<_> = (0..LAYERS)
            .map(|_| (cx.tensor((SLOTS, KV_DIM)), cx.tensor((SLOTS, KV_DIM))))
            .collect();
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let rope_inputs: Vec<_> = (0..LAYERS)
            .map(|_| (cx.tensor((1, HD)), cx.tensor((1, HD))))
            .collect();
        let rope_rot = cx.tensor((HD, HD));
        let model = MiniGemma3::new(VOCAB, D, FF, NH, NKV, HD, LAYERS, WINDOW, PATTERN, &mut cx);
        let (logits, caches_out) = model.forward(
            ids,
            &caches,
            gather_idx,
            scatter_idx,
            Expression::from(q_pos),
            &rope_inputs,
            rope_rot,
        );
        let logits = logits.output();
        let caches_out: Vec<_> = caches_out
            .into_iter()
            .map(|(k, v)| (k.output(), v.output()))
            .collect();

        let seeds = |layer: usize, slot: usize| 600 + layer * 20 + slot;
        let embed_w = weights(VOCAB * D, 199);
        let rot_matrix = crate::rope_pairing_matrix(HD, false);
        // Per-layer tables from each block's role parameters.
        let role_tables: Vec<(Vec<f32>, Vec<f32>)> = model
            .blocks
            .iter()
            .map(|block| {
                crate::rope_tables_split_half(
                    &[q_pos as f32],
                    HD,
                    block.rope_theta,
                    block.pos_scale,
                )
            })
            .collect();
        let mut pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (ids.id, vec![token as f32]),
            (model.embed.weight.id, embed_w.clone()),
            (gather_idx.id, vec![0.0, 1.0]),
            (scatter_idx.id, vec![1.0]),
            (rope_rot.id, rot_matrix.clone()),
            (model.final_norm.weight.expect("weighted").id, weights(D, 660)),
        ];
        for (layer, (cos_table, sin_table)) in role_tables.iter().enumerate() {
            pairs.push((rope_inputs[layer].0.id, cos_table.clone()));
            pairs.push((rope_inputs[layer].1.id, sin_table.clone()));
        }
        let mut ref_caches: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for (layer, block) in model.blocks.iter().enumerate() {
            pairs.push((block.wq.weight.id, weights(D * Q_DIM, seeds(layer, 0))));
            pairs.push((block.wk.weight.id, weights(D * KV_DIM, seeds(layer, 1))));
            pairs.push((block.wv.weight.id, weights(D * KV_DIM, seeds(layer, 2))));
            pairs.push((block.wo.weight.id, weights(Q_DIM * D, seeds(layer, 3))));
            pairs.push((block.gate.weight.id, weights(D * FF, seeds(layer, 4))));
            pairs.push((block.up.weight.id, weights(D * FF, seeds(layer, 5))));
            pairs.push((block.down.weight.id, weights(FF * D, seeds(layer, 6))));
            pairs.push((block.input_norm.weight.expect("weighted").id, weights(D, seeds(layer, 7))));
            pairs.push((block.post_attn_norm.weight.expect("weighted").id, weights(D, seeds(layer, 8))));
            pairs.push((block.pre_ff_norm.weight.expect("weighted").id, weights(D, seeds(layer, 9))));
            pairs.push((block.post_ff_norm.weight.expect("weighted").id, weights(D, seeds(layer, 10))));
            pairs.push((block.q_norm.id, weights(HD, seeds(layer, 11))));
            pairs.push((block.k_norm.id, weights(HD, seeds(layer, 12))));
            let kc = weights(SLOTS * KV_DIM, 300 + layer);
            let vc = weights(SLOTS * KV_DIM, 320 + layer);
            pairs.push((caches[layer].0.id, kc.clone()));
            pairs.push((caches[layer].1.id, vc.clone()));
            ref_caches.push((kc, vc));
        }

        // ---- scalar reference ----
        let wrms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            ref_rms_norm(x, 1e-6).iter().zip(w).map(|(v, w)| v * w).collect()
        };
        let mul = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(x, y)| x * y).collect()
        };
        let add = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(x, y)| x + y).collect()
        };
        let mut x: Vec<f32> = embed_w[token * D..(token + 1) * D]
            .iter()
            .map(|v| v * (D as f32).sqrt())
            .collect();
        for layer in 0..LAYERS {
            let local = (layer + 1) % PATTERN != 0;
            let (cos_table, sin_table) = &role_tables[layer];
            let scale = 1.0 / (HD as f32).sqrt();
            let (kc, vc) = &mut ref_caches[layer];
            let h = wrms(&x, &weights(D, seeds(layer, 7)));
            let q = ref_matmul(&h, &weights(D * Q_DIM, seeds(layer, 0)), D, Q_DIM);
            let q = ref_rms_head_norm(&q, HD, &weights(HD, seeds(layer, 11)));
            let q: Vec<f32> = q.iter().map(|v| v * scale).collect(); // folded into Q
            let q = ref_rotary_apply(&q, HD, cos_table, sin_table, &rot_matrix);
            let k = ref_matmul(&h, &weights(D * KV_DIM, seeds(layer, 1)), D, KV_DIM);
            let k = ref_rms_head_norm(&k, HD, &weights(HD, seeds(layer, 12)));
            let k = ref_rotary_apply(&k, HD, cos_table, sin_table, &rot_matrix);
            let v = ref_matmul(&h, &weights(D * KV_DIM, seeds(layer, 2)), D, KV_DIM);
            let attn = ref_paged_step_gqa(
                &q,
                &k,
                &v,
                kc,
                vc,
                &[0, 1],
                1,
                NH,
                NKV,
                HD,
                q_pos,
                local.then_some(WINDOW),
                1.0, // scale already folded into q
            );
            let attn_out = ref_matmul(&attn, &weights(Q_DIM * D, seeds(layer, 3)), Q_DIM, D);
            x = add(&x, &wrms(&attn_out, &weights(D, seeds(layer, 8))));
            let ff_in = wrms(&x, &weights(D, seeds(layer, 9)));
            let gate = ref_gelu_tanh(&ref_matmul(&ff_in, &weights(D * FF, seeds(layer, 4)), D, FF));
            let up = ref_matmul(&ff_in, &weights(D * FF, seeds(layer, 5)), D, FF);
            let ff = ref_matmul(&mul(&gate, &up), &weights(FF * D, seeds(layer, 6)), FF, D);
            x = add(&x, &wrms(&ff, &weights(D, seeds(layer, 10))));
        }
        let x = wrms(&x, &weights(D, 660));
        let ref_logits: Vec<f32> = (0..VOCAB)
            .map(|v| (0..D).map(|i| x[i] * embed_w[v * D + i]).sum())
            .collect();

        let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
        let mut rt = luminal::ssa_reference::SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(
            &data,
            &luminal::implementation_search::ImplementationSearchOptions::default(),
        )
        .expect("search finds a plan");
        for (id, values) in &pairs {
            rt.set_data(*id, values.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
        for layer in 0..LAYERS {
            assert_close(rt.get_f32(caches_out[layer].0.id).unwrap(), &ref_caches[layer].0);
            assert_close(rt.get_f32(caches_out[layer].1.id).unwrap(), &ref_caches[layer].1);
        }
    }

    /// MoE-family harness: 1 block + embed + tied logits, one decode
    /// step; `softcap` = the gemma4_moe final-logit soft-capping.
    fn mini_moe_family(softcap: Option<f32>) {
        const VOCAB: usize = 5;
        const D: usize = 4;
        const E: usize = 2;
        const NH: usize = 2;
        const SLOTS: usize = 4;
        const CTX: usize = 2;
        let token = 2usize;

        let mut cx = Graph::new();
        let ids = cx.tensor_dtyped(1, DType::Int);
        let k_cache = cx.tensor((SLOTS, D));
        let v_cache = cx.tensor((SLOTS, D));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let caches = vec![(k_cache, v_cache)];
        let step = Expression::from(1usize);
        let (logits, caches_out, embed, blocks) = match softcap {
            None => {
                let model = MiniQwen3Moe::new(VOCAB, D, E, 1, NH, 1, &mut cx);
                let (logits, caches_out) =
                    model.forward(ids, &caches, gather_idx, scatter_idx, step);
                (logits, caches_out, model.embed, model.blocks)
            }
            Some(cap) => {
                let model = MiniGemma4Moe::new(VOCAB, D, E, 1, NH, 1, &mut cx);
                assert_eq!(model.logit_softcap, cap, "family cap is fixed");
                let (logits, caches_out) =
                    model.forward(ids, &caches, gather_idx, scatter_idx, step);
                (logits, caches_out, model.embed, model.blocks)
            }
        };
        let logits = logits.output();
        let (kc_out, vc_out) = (caches_out[0].0.output(), caches_out[0].1.output());

        let embed_w = weights(VOCAB * D, 400);
        let block = &blocks[0];
        let crate::FeedForward::Moe(moe) = &block.ff else { unreachable!() };
        let pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (ids.id, vec![token as f32]),
            (embed.weight.id, embed_w.clone()),
            (block.wq.weight.id, weights(D * D, 401)),
            (block.wk.weight.id, weights(D * D, 402)),
            (block.wv.weight.id, weights(D * D, 403)),
            (block.wo.weight.id, weights(D * D, 404)),
            (moe.router.id, weights(D * E, 405)),
            (moe.expert_weights.id, weights(E * D * D, 406)),
            (k_cache.id, weights(SLOTS * D, 407)),
            (v_cache.id, weights(SLOTS * D, 408)),
            (gather_idx.id, vec![0.0, 1.0]),
            (scatter_idx.id, vec![1.0]),
        ];

        // Scalar reference: embed row → block (attn + MoE ffn) → LN →
        // tied logits.
        let x: Vec<f32> = embed_w[token * D..(token + 1) * D].to_vec();
        let mut kc = weights(SLOTS * D, 407);
        let mut vc = weights(SLOTS * D, 408);
        let router = weights(D * E, 405);
        let experts = weights(E * D * D, 406);
        let ff = move |x: &[f32]| ref_moe_k1(x, &router, &experts, D, E);
        let x2 = ref_block_step(
            &x,
            &weights(D * D, 401),
            &weights(D * D, 402),
            &weights(D * D, 403),
            &weights(D * D, 404),
            &ff,
            &mut kc,
            &mut vc,
            &[0, 1],
            1,
            NH,
            D / NH,
            D,
        );
        let x2 = ref_layer_norm(&x2, 1e-5);
        let mut ref_logits: Vec<f32> = (0..VOCAB)
            .map(|v| (0..D).map(|i| x2[i] * embed_w[v * D + i]).sum())
            .collect();
        if let Some(cap) = softcap {
            ref_logits = ref_logits.iter().map(|v| (v / cap).tanh() * cap).collect();
        }

        // embed + block + tied logits is deep enough that the 8-genome
        // harness budget usually cycles out — default budget.
        let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
        let mut rt = luminal::ssa_reference::SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(
            &data,
            &luminal::implementation_search::ImplementationSearchOptions::default(),
        )
        .expect("search finds a plan");
        for (id, values) in &pairs {
            rt.set_data(*id, values.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
        assert_close(rt.get_f32(kc_out.id).unwrap(), &kc);
        assert_close(rt.get_f32(vc_out.id).unwrap(), &vc);
    }

    #[test]
    fn mini_qwen3_moe_matches_scalar_reference() {
        mini_moe_family(None);
    }

    #[test]
    fn mini_gemma4_moe_matches_scalar_reference() {
        mini_moe_family(Some(30.0));
    }

    /// MiniWhisper: encoder self-attention + decoder CROSS-attention —
    /// the construct nothing else exercises.
    #[test]
    fn mini_whisper_matches_scalar_reference() {
        const D: usize = 4;
        const FF: usize = 6;
        const NH: usize = 2;
        const HD: usize = D / NH;
        const S_ENC: usize = 2;

        let mut cx = Graph::new();
        let model = MiniWhisper::new(D, FF, NH, &mut cx);
        let audio = cx.tensor((S_ENC, D));
        let tokens = cx.tensor((1, D));
        let out = model.forward(audio, tokens).output();

        let audio_vals = weights(S_ENC * D, 500);
        let token_vals = weights(D, 501);
        let pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (audio.id, audio_vals.clone()),
            (tokens.id, token_vals.clone()),
            (model.enc_wq.weight.id, weights(D * D, 502)),
            (model.enc_wk.weight.id, weights(D * D, 503)),
            (model.enc_wv.weight.id, weights(D * D, 504)),
            (model.enc_wo.weight.id, weights(D * D, 505)),
            (model.enc_up.weight.id, weights(D * FF, 506)),
            (model.enc_down.weight.id, weights(FF * D, 507)),
            (model.dec_wq.weight.id, weights(D * D, 508)),
            (model.dec_wk.weight.id, weights(D * D, 509)),
            (model.dec_wv.weight.id, weights(D * D, 510)),
            (model.dec_wo.weight.id, weights(D * D, 511)),
            (model.dec_up.weight.id, weights(D * FF, 512)),
            (model.dec_down.weight.id, weights(FF * D, 513)),
        ];

        // Scalar reference.
        let rows_matmul = |x: &[f32], w: &[f32], rows: usize, a: usize, b: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(rows * b);
            for r in 0..rows {
                out.extend(ref_matmul(&x[r * a..(r + 1) * a], w, a, b));
            }
            out
        };
        let ln_rows = |x: &[f32], rows: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(x.len());
            for r in 0..rows {
                out.extend(ref_layer_norm(&x[r * D..(r + 1) * D], 1e-5));
            }
            out
        };
        // Encoder.
        let normed = ln_rows(&audio_vals, S_ENC);
        let q = rows_matmul(&normed, &weights(D * D, 502), S_ENC, D, D);
        let k = rows_matmul(&normed, &weights(D * D, 503), S_ENC, D, D);
        let v = rows_matmul(&normed, &weights(D * D, 504), S_ENC, D, D);
        let sa = ref_attention(&q, &k, &v, S_ENC, S_ENC, NH, HD);
        let sa_proj = rows_matmul(&sa, &weights(D * D, 505), S_ENC, D, D);
        let enc1: Vec<f32> = audio_vals.iter().zip(&sa_proj).map(|(a, b)| a + b).collect();
        let hidden = ref_gelu_tanh(&rows_matmul(&enc1, &weights(D * FF, 506), S_ENC, D, FF));
        let ffo = rows_matmul(&hidden, &weights(FF * D, 507), S_ENC, FF, D);
        let enc: Vec<f32> = enc1.iter().zip(&ffo).map(|(a, b)| a + b).collect();
        // Decoder cross-attention.
        let normed = ref_layer_norm(&token_vals, 1e-5);
        let q = ref_matmul(&normed, &weights(D * D, 508), D, D);
        let k = rows_matmul(&enc, &weights(D * D, 509), S_ENC, D, D);
        let v = rows_matmul(&enc, &weights(D * D, 510), S_ENC, D, D);
        let cross = ref_attention(&q, &k, &v, 1, S_ENC, NH, HD);
        let cross_proj = ref_matmul(&cross, &weights(D * D, 511), D, D);
        let x1: Vec<f32> = token_vals.iter().zip(&cross_proj).map(|(a, b)| a + b).collect();
        let hidden = ref_gelu_tanh(&ref_matmul(&x1, &weights(D * FF, 512), D, FF));
        let ffo = ref_matmul(&hidden, &weights(FF * D, 513), FF, D);
        let expected: Vec<f32> = x1.iter().zip(&ffo).map(|(a, b)| a + b).collect();

        let rt = luminal::test_support::run_ssa(&cx, &pairs);
        assert_close(rt.get_f32(out.id).expect("out"), &expected);
        let _ = attention;
    }

    /// MiniConvNet: 3×3 valid convs 5→3→1 + relu + linear head.
    #[test]
    fn mini_convnet_matches_scalar_reference() {
        const C1: usize = 2;
        const C2: usize = 3;
        const CLASSES: usize = 2;

        let mut cx = Graph::new();
        let model = MiniConvNet::new(1, C1, C2, CLASSES, &mut cx);
        let x = cx.tensor((1, 1, 5, 5));
        let out = model.forward(x).output();

        let x_vals = weights(25, 600);
        let w1 = weights(C1 * 9, 601);
        let w2 = weights(C2 * C1 * 9, 602);
        let wh = weights(C2 * CLASSES, 603);
        let pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (x.id, x_vals.clone()),
            (model.conv1.weight.id, w1.clone()),
            (model.conv2.weight.id, w2.clone()),
            (model.head.weight.id, wh.clone()),
        ];

        // Scalar reference: valid 3×3 convs; ConvND weight layout is
        // (ch_out, ch_in·kh·kw) with kernel-major within a channel.
        let conv = |input: &[f32], w: &[f32], ch_in: usize, ch_out: usize, h: usize| -> Vec<f32> {
            let oh = h - 2;
            let mut out = vec![0f32; ch_out * oh * oh];
            for co in 0..ch_out {
                for oy in 0..oh {
                    for ox in 0..oh {
                        let mut acc = 0f32;
                        for ci in 0..ch_in {
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    acc += input[ci * h * h + (oy + ky) * h + (ox + kx)]
                                        * w[co * ch_in * 9 + ci * 9 + ky * 3 + kx];
                                }
                            }
                        }
                        out[co * oh * oh + oy * oh + ox] = acc.max(0.0);
                    }
                }
            }
            out
        };
        let f1 = conv(&x_vals, &w1, 1, C1, 5); // (C1, 3, 3), relu applied
        let f2 = conv(&f1, &w2, C1, C2, 3); // (C2, 1, 1), relu applied
        let expected = ref_matmul(&f2, &wh, C2, CLASSES);

        let rt = luminal::test_support::run_ssa(&cx, &pairs);
        assert_close(rt.get_f32(out.id).expect("logits"), &expected);
    }

    /// MiniDit vs a full scalar reference: 1 double + 1 single block,
    /// d=16, 2 heads (head_dim 8 = the 4-axis rope table width), 4 image
    /// tokens (2×2 grid) + 2 text tokens, adaLN conditioning from
    /// t/guidance scalars. The two ordering conventions the recon flagged
    /// as silent-mismatch bait — (shift, scale, gate) in block triples vs
    /// (scale, shift) at norm_out, and txt-before-img in every
    /// concat/split — are exercised by construction.
    #[test]
    #[ignore = "BLOCKED on the rejoin-divergence ruling: three concat/view spellings \
                fixed (matmul rope, flat V concat, split out-projection, scatter-\
                assembled joint sequence) and the graph still finds a slice-through-\
                elementwise-distribution road into a view stack (stage-8 probe). The \
                adaLN broadcast-modulation architecture generates these roads \
                structurally; unblock = stratified composition or structural map \
                entries. Probes: probe_dit_stages / probe_dit_round_driver."]
    fn mini_dit_matches_scalar_reference() {
        const IN_CH: usize = 4;
        const TXT_DIM: usize = 6;
        const D: usize = 16;
        const NH: usize = 2;
        const HD: usize = 8;
        const MLP: usize = 6;
        const T_HALF: usize = 2;
        const T_CH: usize = 2 * T_HALF;
        const S_TXT: usize = 2;
        const GRID: usize = 2;
        const S_IMG: usize = GRID * GRID;
        const S: usize = S_TXT + S_IMG;

        let mut cx = Graph::new();
        let model = MiniDit::new(IN_CH, TXT_DIM, D, NH, MLP, T_HALF, S_TXT, &mut cx);
        let latent = cx.tensor((S_IMG, IN_CH));
        let text = cx.tensor((S_TXT, TXT_DIM));
        let t = cx.tensor(1);
        let guidance = cx.tensor(1);
        let rope_cos = cx.tensor((S, HD));
        let rope_sin = cx.tensor((S, HD));
        let rope_rot = cx.tensor((HD, HD));
        let joint_base = cx.tensor((S, D));
        let velocity = model
            .forward(latent, text, t, guidance, rope_cos, rope_sin, rope_rot, joint_base)
            .output();

        let (cos_table, sin_table) = mini_dit_rope_tables(S_TXT, GRID, GRID);
        let rot_matrix = crate::rope_pairing_matrix(HD, true);
        let latent_vals = weights(S_IMG * IN_CH, 540);
        let text_vals = weights(S_TXT * TXT_DIM, 541);
        let (t_val, g_val) = (0.35f32, 0.8f32);
        let pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (latent.id, latent_vals.clone()),
            (text.id, text_vals.clone()),
            (t.id, vec![t_val]),
            (guidance.id, vec![g_val]),
            (rope_cos.id, cos_table.clone()),
            (rope_sin.id, sin_table.clone()),
            (rope_rot.id, rot_matrix.clone()),
            (joint_base.id, vec![0.0; S * D]),
            (model.x_embed.weight.id, weights(IN_CH * D, 500)),
            (model.ctx_embed.weight.id, weights(TXT_DIM * D, 501)),
            (model.t_mlp1.weight.id, weights(T_CH * D, 502)),
            (model.t_mlp2.weight.id, weights(D * D, 503)),
            (model.g_mlp1.weight.id, weights(T_CH * D, 504)),
            (model.g_mlp2.weight.id, weights(D * D, 505)),
            (model.mod_img.weight.id, weights(D * 6 * D, 506)),
            (model.mod_txt.weight.id, weights(D * 6 * D, 507)),
            (model.mod_single.weight.id, weights(D * 3 * D, 508)),
            (model.norm_out.weight.id, weights(D * 2 * D, 509)),
            (model.proj_out.weight.id, weights(D * IN_CH, 510)),
            (model.img_q.weight.id, weights(D * D, 511)),
            (model.img_k.weight.id, weights(D * D, 512)),
            (model.img_v.weight.id, weights(D * D, 513)),
            (model.img_out.weight.id, weights(D * D, 514)),
            (model.txt_q.weight.id, weights(D * D, 515)),
            (model.txt_k.weight.id, weights(D * D, 516)),
            (model.txt_v.weight.id, weights(D * D, 517)),
            (model.txt_out.weight.id, weights(D * D, 518)),
            (model.img_qnorm.id, weights(HD, 519)),
            (model.img_knorm.id, weights(HD, 520)),
            (model.txt_qnorm.id, weights(HD, 521)),
            (model.txt_knorm.id, weights(HD, 522)),
            (model.ff_in.weight.id, weights(D * 2 * MLP, 523)),
            (model.ff_out.weight.id, weights(MLP * D, 524)),
            (model.ctx_ff_in.weight.id, weights(D * 2 * MLP, 525)),
            (model.ctx_ff_out.weight.id, weights(MLP * D, 526)),
            (model.single_proj.weight.id, weights(D * (3 * D + 2 * MLP), 527)),
            (model.single_out_attn.weight.id, weights(D * D, 531)),
            (model.single_out_mlp.weight.id, weights(MLP * D, 532)),
            (model.single_qnorm.id, weights(HD, 529)),
            (model.single_knorm.id, weights(HD, 530)),
        ];

        // ---- scalar reference ----
        // Row-wise helpers (test_refs' are single-row).
        let matmul_rows = |x: &[f32], w: &[f32], rows: usize, in_w: usize, out_w: usize| {
            let mut out = Vec::with_capacity(rows * out_w);
            for row in 0..rows {
                out.extend(ref_matmul(&x[row * in_w..(row + 1) * in_w], w, in_w, out_w));
            }
            out
        };
        let ln_rows = |x: &[f32], rows: usize| {
            let width = x.len() / rows;
            let mut out = Vec::with_capacity(x.len());
            for row in 0..rows {
                out.extend(ref_layer_norm(&x[row * width..(row + 1) * width], 1e-6));
            }
            out
        };
        // adaLN: x·(1+scale)+shift, modulation rows broadcast over rows.
        let ada_rows = |x: &[f32], scale: &[f32], shift: &[f32], rows: usize| {
            let width = x.len() / rows;
            let mut out = Vec::with_capacity(x.len());
            for row in 0..rows {
                for col in 0..width {
                    out.push(x[row * width + col] * (1.0 + scale[col]) + shift[col]);
                }
            }
            out
        };
        let gate_rows = |x: &[f32], g: &[f32], rows: usize| {
            let width = x.len() / rows;
            (0..rows)
                .flat_map(|row| (0..width).map(move |col| x[row * width + col] * g[col]))
                .collect::<Vec<f32>>()
        };
        // Per-head QK-norm over (rows, D) with heads side by side.
        let head_norm_rows = |x: &[f32], w: &[f32], rows: usize| {
            let mut out = Vec::with_capacity(x.len());
            for row in 0..rows {
                out.extend(ref_rms_head_norm(&x[row * D..(row + 1) * D], HD, w));
            }
            out
        };
        // Interleaved-pair rope over (rows, D): per head, per pair:
        // x'[2m] = x[2m]·cos − x[2m+1]·sin; x'[2m+1] = x[2m+1]·cos + x[2m]·sin.
        let rope_rows = |x: &[f32], rows: usize| {
            let mut out = x.to_vec();
            for row in 0..rows {
                for head in 0..NH {
                    for pair in 0..HD / 2 {
                        let base = row * D + head * HD + 2 * pair;
                        let (c0, s0) = (cos_table[row * HD + 2 * pair], sin_table[row * HD + 2 * pair]);
                        let (c1, s1) = (
                            cos_table[row * HD + 2 * pair + 1],
                            sin_table[row * HD + 2 * pair + 1],
                        );
                        let (even, odd) = (x[base], x[base + 1]);
                        out[base] = even * c0 - odd * s0;
                        out[base + 1] = odd * c1 + even * s1;
                    }
                }
            }
            out
        };
        let swiglu_rows = |u: &[f32], rows: usize| {
            let mut out = Vec::with_capacity(rows * MLP);
            for row in 0..rows {
                let row = &u[row * 2 * MLP..(row + 1) * 2 * MLP];
                out.extend(
                    ref_silu(&row[..MLP])
                        .iter()
                        .zip(&row[MLP..])
                        .map(|(a, b)| a * b),
                );
            }
            out
        };
        let add = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(x, y)| x + y).collect()
        };

        // Conditioning.
        let sinusoid = |x: f32| -> Vec<f32> {
            let args: Vec<f32> = (0..T_HALF)
                .map(|i| 1000.0 * x * (-(i as f32) * (10000f32).ln() / T_HALF as f32).exp())
                .collect();
            args.iter()
                .map(|a| a.cos())
                .chain(args.iter().map(|a| a.sin()))
                .collect()
        };
        let temb = add(
            &ref_matmul(
                &ref_silu(&ref_matmul(&sinusoid(t_val), &weights(T_CH * D, 502), T_CH, D)),
                &weights(D * D, 503),
                D,
                D,
            ),
            &ref_matmul(
                &ref_silu(&ref_matmul(&sinusoid(g_val), &weights(T_CH * D, 504), T_CH, D)),
                &weights(D * D, 505),
                D,
                D,
            ),
        );
        let cond = ref_silu(&temb);
        let m_img = ref_matmul(&cond, &weights(D * 6 * D, 506), D, 6 * D);
        let m_txt = ref_matmul(&cond, &weights(D * 6 * D, 507), D, 6 * D);
        let m_single = ref_matmul(&cond, &weights(D * 3 * D, 508), D, 3 * D);
        let triple = |m: &[f32], set: usize| {
            let base = set * 3 * D;
            (
                m[base..base + D].to_vec(),           // shift
                m[base + D..base + 2 * D].to_vec(),   // scale
                m[base + 2 * D..base + 3 * D].to_vec(), // gate
            )
        };

        // Double-stream block.
        let (shift0, scale0, gate0) = triple(&m_img, 0);
        let (shift1, scale1, gate1) = triple(&m_img, 1);
        let (c_shift0, c_scale0, c_gate0) = triple(&m_txt, 0);
        let (c_shift1, c_scale1, c_gate1) = triple(&m_txt, 1);
        let mut img = matmul_rows(&latent_vals, &weights(IN_CH * D, 500), S_IMG, IN_CH, D);
        let mut txt = matmul_rows(&text_vals, &weights(TXT_DIM * D, 501), S_TXT, TXT_DIM, D);
        let img_n = ada_rows(&ln_rows(&img, S_IMG), &scale0, &shift0, S_IMG);
        let txt_n = ada_rows(&ln_rows(&txt, S_TXT), &c_scale0, &c_shift0, S_TXT);
        let q_img = head_norm_rows(
            &matmul_rows(&img_n, &weights(D * D, 511), S_IMG, D, D),
            &weights(HD, 519),
            S_IMG,
        );
        let k_img = head_norm_rows(
            &matmul_rows(&img_n, &weights(D * D, 512), S_IMG, D, D),
            &weights(HD, 520),
            S_IMG,
        );
        let v_img = matmul_rows(&img_n, &weights(D * D, 513), S_IMG, D, D);
        let q_txt = head_norm_rows(
            &matmul_rows(&txt_n, &weights(D * D, 515), S_TXT, D, D),
            &weights(HD, 521),
            S_TXT,
        );
        let k_txt = head_norm_rows(
            &matmul_rows(&txt_n, &weights(D * D, 516), S_TXT, D, D),
            &weights(HD, 522),
            S_TXT,
        );
        let v_txt = matmul_rows(&txt_n, &weights(D * D, 517), S_TXT, D, D);
        // txt first, then rope, then joint non-causal attention.
        let concat_rows = |a: &[f32], b: &[f32]| {
            let mut joined = a.to_vec();
            joined.extend_from_slice(b);
            joined
        };
        let q = rope_rows(&concat_rows(&q_txt, &q_img), S);
        let k = rope_rows(&concat_rows(&k_txt, &k_img), S);
        let v = concat_rows(&v_txt, &v_img);
        let attn = ref_attention(&q, &k, &v, S, S, NH, HD);
        let attn_txt = &attn[..S_TXT * D];
        let attn_img = &attn[S_TXT * D..];
        img = add(
            &img,
            &gate_rows(&matmul_rows(attn_img, &weights(D * D, 514), S_IMG, D, D), &gate0, S_IMG),
        );
        txt = add(
            &txt,
            &gate_rows(&matmul_rows(attn_txt, &weights(D * D, 518), S_TXT, D, D), &c_gate0, S_TXT),
        );
        let ff = swiglu_rows(
            &matmul_rows(
                &ada_rows(&ln_rows(&img, S_IMG), &scale1, &shift1, S_IMG),
                &weights(D * 2 * MLP, 523),
                S_IMG,
                D,
                2 * MLP,
            ),
            S_IMG,
        );
        img = add(
            &img,
            &gate_rows(&matmul_rows(&ff, &weights(MLP * D, 524), S_IMG, MLP, D), &gate1, S_IMG),
        );
        let c_ff = swiglu_rows(
            &matmul_rows(
                &ada_rows(&ln_rows(&txt, S_TXT), &c_scale1, &c_shift1, S_TXT),
                &weights(D * 2 * MLP, 525),
                S_TXT,
                D,
                2 * MLP,
            ),
            S_TXT,
        );
        txt = add(
            &txt,
            &gate_rows(&matmul_rows(&c_ff, &weights(MLP * D, 526), S_TXT, MLP, D), &c_gate1, S_TXT),
        );

        // Single-stream block over [txt ‖ img].
        let mut hidden = concat_rows(&txt, &img);
        let (s_shift, s_scale, s_gate) = triple(&m_single, 0);
        let normed = ada_rows(&ln_rows(&hidden, S), &s_scale, &s_shift, S);
        let proj = matmul_rows(&normed, &weights(D * (3 * D + 2 * MLP), 527), S, D, 3 * D + 2 * MLP);
        let width = 3 * D + 2 * MLP;
        let slice_cols = |x: &[f32], from: usize, to: usize| {
            let mut out = Vec::with_capacity(S * (to - from));
            for row in 0..S {
                out.extend_from_slice(&x[row * width + from..row * width + to]);
            }
            out
        };
        let q = rope_rows(
            &head_norm_rows(&slice_cols(&proj, 0, D), &weights(HD, 529), S),
            S,
        );
        let k = rope_rows(
            &head_norm_rows(&slice_cols(&proj, D, 2 * D), &weights(HD, 530), S),
            S,
        );
        let v = slice_cols(&proj, 2 * D, 3 * D);
        let attn = ref_attention(&q, &k, &v, S, S, NH, HD);
        let mlp_out = swiglu_rows(&slice_cols(&proj, 3 * D, 3 * D + 2 * MLP), S);
        // Row-split fused out-projection (mirrors single_out_attn/_mlp).
        let out_sum = add(
            &matmul_rows(&attn, &weights(D * D, 531), S, D, D),
            &matmul_rows(&mlp_out, &weights(MLP * D, 532), S, MLP, D),
        );
        hidden = add(&hidden, &gate_rows(&out_sum, &s_gate, S));

        // AdaLayerNormContinuous head — (scale, shift), REVERSED order.
        let img_final = &hidden[S_TXT * D..];
        let head = ref_matmul(&cond, &weights(D * 2 * D, 509), D, 2 * D);
        let (scale, shift) = (&head[..D], &head[D..]);
        let expected = matmul_rows(
            &ada_rows(&ln_rows(img_final, S_IMG), scale, shift, S_IMG),
            &weights(D * IN_CH, 510),
            S_IMG,
            D,
            IN_CH,
        );

        let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
        let mut rt = luminal::ssa_reference::SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(
            &data,
            &luminal::implementation_search::ImplementationSearchOptions::default(),
        )
        .expect("search finds a plan");
        for (id, values) in &pairs {
            rt.set_data(*id, values.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(velocity.id).expect("velocity"), &expected);
    }

    /// MICRO-PROBES splitting the rope bomb (rope-alone RSS-killed at
    /// ~4GB in 5s): the two sick graphs are the ONLY graphs using sin
    /// or concat_along, so each stage here exercises one atom — sin,
    /// sin-of-scaled, cos (= sin(π/2 − x)), pad, concat (pad+add), and
    /// the trig+concat angle table. Run:
    /// cargo test --release -p luminal_nn probe_trig_concat -- --ignored --nocapture
    #[test]
    #[ignore = "diagnosis probe — run explicitly by name (release, bounded)"]
    fn probe_trig_concat() {
        let budget = luminal::implementation_search::ImplementationSearchOptions {
            generations: 1,
            generation_size: 1,
            mutations: 1,
            trials: 1,
            seed: 0,
        };
        let run = |label: &str, cx: &Graph, pairs: &[(petgraph::graph::NodeIndex, Vec<f32>)]| {
            let start = std::time::Instant::now();
            let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
            let mut rt =
                luminal::ssa_reference::SsaReferenceRuntime::load(cx).expect("native load");
            match rt.search(&data, &budget) {
                Ok(outcome) => eprintln!(
                    "[micro-probe] {label}: wall {:.1}s | {}",
                    start.elapsed().as_secs_f64(),
                    outcome.timings.summary()
                ),
                Err(err) => eprintln!(
                    "[micro-probe] {label}: wall {:.1}s | search refused: {err:#}",
                    start.elapsed().as_secs_f64()
                ),
            }
        };

        // 1: sin alone
        {
            let mut cx = Graph::new();
            let x = cx.tensor(1);
            let _ = x.sin().output();
            run("sin-alone", &cx, &[(x.id, vec![0.5])]);
        }
        // 2: sin of scaled input
        {
            let mut cx = Graph::new();
            let x = cx.tensor(1);
            let _ = (x * 0.37).sin().output();
            run("sin-scaled", &cx, &[(x.id, vec![0.5])]);
        }
        // 3: cos = sin(π/2 − x)
        {
            let mut cx = Graph::new();
            let x = cx.tensor(1);
            let _ = x.cos().output();
            run("cos", &cx, &[(x.id, vec![0.5])]);
        }
        // 4: pad alone (no trig)
        {
            let mut cx = Graph::new();
            let x = cx.tensor((1, 1));
            let _ = x.pad_along(0, 1, 1, 0.).output();
            run("pad-alone", &cx, &[(x.id, vec![0.5])]);
        }
        // 5: concat of two (1,1) tensors (pad + add, no trig)
        {
            let mut cx = Graph::new();
            let a = cx.tensor((1, 1));
            let b = cx.tensor((1, 1));
            let _ = a.concat_along(b, 1).output();
            run("concat-alone", &cx, &[(a.id, vec![0.5]), (b.id, vec![0.7])]);
        }
        // 6: the angle table — cos/sin of scaled positions, concatenated
        {
            let mut cx = Graph::new();
            let pos = cx.tensor(1);
            let c0 = pos.cos().unsqueeze(1);
            let c1 = (pos * 0.01).cos().unsqueeze(1);
            let _ = c0.concat_along(c1, 1).output();
            run("angle-table", &cx, &[(pos.id, vec![1.0])]);
        }
        // 7: rank-3 movement round-trip (split → merge, no slices)
        {
            let mut cx = Graph::new();
            let x = cx.tensor((1, 8));
            let _ = x.split_dims(1, 4).merge_dims(1, 2).output();
            run("split-merge", &cx, &[(x.id, weights(8, 11))]);
        }
        // 8: THE REJOIN — slice halves of a rank-3, concat them back
        // (the composition-rows divergence pattern), no trig involved
        {
            let mut cx = Graph::new();
            let x = cx.tensor((1, 8));
            let heads = x.split_dims(1, 4); // (1, 2, 4)
            let x1 = heads.slice_along(0..2, 2);
            let x2 = heads.slice_along(2..4, 2);
            let _ = x2.concat_along(x1, 2).merge_dims(1, 2).output();
            run("rejoin", &cx, &[(x.id, weights(8, 12))]);
        }
        // 9: broadcast-multiply — angle row expanded onto the sliced half
        {
            let mut cx = Graph::new();
            let x = cx.tensor((1, 8));
            let table = cx.tensor((1, 2));
            let heads = x.split_dims(1, 4); // (1, 2, 4)
            let x1 = heads.slice_along(0..2, 2); // (1, 2, 2)
            let bcast = table.unsqueeze(1).expand(x1.dims());
            let _ = (x1 * bcast).merge_dims(1, 2).output();
            run("broadcast-mul", &cx, &[(x.id, weights(8, 13)), (table.id, weights(2, 14))]);
        }
    }

    /// CONSTRUCT-ISOLATION PROBES for the gemma3 memory explosion
    /// (RSS-KILL 5.8GB in 12s, isolated sweep 2026-08-10): each
    /// sub-graph exercises exactly ONE of the constructs the full-
    /// anatomy gemma added — (a) in-graph split-half rope, (b) sliding-
    /// window paged attention, (c) weighted sandwich norms. 1-genome
    /// budget; run in a capped process — the stage whose print never
    /// appears is the bomb. Run:
    /// cargo test --release -p luminal_nn probe_gemma_constructs -- --ignored --nocapture
    #[test]
    #[ignore = "diagnosis probe — run explicitly by name (release, bounded)"]
    fn probe_gemma_constructs() {
        let budget = luminal::implementation_search::ImplementationSearchOptions {
            generations: 1,
            generation_size: 1,
            mutations: 1,
            trials: 1,
            seed: 0,
        };
        let run = |label: &str, cx: &Graph, pairs: &[(petgraph::graph::NodeIndex, Vec<f32>)]| {
            let start = std::time::Instant::now();
            let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
            let mut rt =
                luminal::ssa_reference::SsaReferenceRuntime::load(cx).expect("native load");
            match rt.search(&data, &budget) {
                Ok(outcome) => eprintln!(
                    "[gemma-probe] {label}: wall {:.1}s | {}",
                    start.elapsed().as_secs_f64(),
                    outcome.timings.summary()
                ),
                Err(err) => eprintln!(
                    "[gemma-probe] {label}: wall {:.1}s | search refused: {err:#}",
                    start.elapsed().as_secs_f64()
                ),
            }
        };

        // (a) rope alone — now the TABLE-AND-MATRIX spelling (the
        // rejoin-divergence workaround). The original slice/neg/concat
        // spelling detonated here (~4GB in 5s); this stage is the
        // positive control that the workaround saturates cleanly.
        {
            let mut cx = Graph::new();
            let x = cx.tensor((1, 8));
            let cos = cx.tensor((1, 4));
            let sin = cx.tensor((1, 4));
            let rot = cx.tensor((4, 4));
            let out = crate::rotary_apply(x, 4, cos, sin, rot).output();
            let _ = out;
            let (cos_table, sin_table) = crate::rope_tables_split_half(&[1.0], 4, 10_000.0, 1.0);
            let pairs = vec![
                (x.id, weights(8, 1)),
                (cos.id, cos_table),
                (sin.id, sin_table),
                (rot.id, crate::rope_pairing_matrix(4, false)),
            ];
            run("rope-alone", &cx, &pairs);
        }

        // (b) windowed paged attention alone (window = 1, tiny dims).
        {
            let mut cx = Graph::new();
            let q = cx.tensor((1, 4));
            let k_new = cx.tensor((1, 4));
            let v_new = cx.tensor((1, 4));
            let k_cache = cx.tensor((4, 4));
            let v_cache = cx.tensor((4, 4));
            let gather_idx = cx.tensor_dtyped(2, DType::Int);
            let scatter_idx = cx.tensor_dtyped(1, DType::Int);
            let (attn, kc, vc) = crate::paged_attention_windowed(
                q,
                k_new,
                v_new,
                k_cache,
                v_cache,
                gather_idx,
                scatter_idx,
                Expression::from(1usize),
                1,
                1,
                4,
                Some(1),
                0.5,
            );
            let _ = (attn.output(), kc.output(), vc.output());
            let pairs = vec![
                (q.id, weights(4, 2)),
                (k_new.id, weights(4, 3)),
                (v_new.id, weights(4, 4)),
                (k_cache.id, weights(16, 5)),
                (v_cache.id, weights(16, 6)),
                (gather_idx.id, vec![0.0, 1.0]),
                (scatter_idx.id, vec![1.0]),
            ];
            run("window-alone", &cx, &pairs);
        }

        // (c) weighted sandwich norms alone: x + post(w·(x·W)) shape.
        {
            let mut cx = Graph::new();
            let x = cx.tensor((1, 6));
            let w = cx.tensor((6, 6));
            let pre = crate::LayerNorm::new(6, Some("Pre"), None, false, 1e-6, &mut cx);
            let post = crate::LayerNorm::new(6, Some("Post"), None, false, 1e-6, &mut cx);
            let out = (x + post.forward(pre.forward(x).matmul(w))).output();
            let _ = out;
            let pairs = vec![
                (x.id, weights(6, 7)),
                (w.id, weights(36, 8)),
                (pre.weight.expect("w").id, weights(6, 9)),
                (post.weight.expect("w").id, weights(6, 10)),
            ];
            run("sandwich-alone", &cx, &pairs);
        }
    }

    /// ROUND DRIVER on the FULL MiniDit graph (stage 7 detonates at one
    /// genome while stage 6 saturates in 10s): runs the main ruleset one
    /// round at a time with table-size deltas and top-firing rules, so
    /// the igniting family names itself. Bounded: 200 rounds, bail on a
    /// >200k-tuple round. Run:
    /// cargo test --release -p luminal_nn probe_dit_round_driver -- --ignored --nocapture
    #[test]
    #[ignore = "diagnosis probe — run explicitly by name (release, bounded)"]
    fn probe_dit_round_driver() {
        const IN_CH: usize = 4;
        const TXT_DIM: usize = 6;
        const D: usize = 16;
        const NH: usize = 2;
        const HD: usize = 8;
        const MLP: usize = 6;
        const T_HALF: usize = 2;
        const S_TXT: usize = 2;
        const GRID: usize = 2;
        const S: usize = S_TXT + GRID * GRID;

        let mut cx = Graph::new();
        let model = MiniDit::new(IN_CH, TXT_DIM, D, NH, MLP, T_HALF, S_TXT, &mut cx);
        let latent = cx.tensor((GRID * GRID, IN_CH));
        let text = cx.tensor((S_TXT, TXT_DIM));
        let t = cx.tensor(1);
        let guidance = cx.tensor(1);
        let rope_cos = cx.tensor((S, HD));
        let rope_sin = cx.tensor((S, HD));
        let rope_rot = cx.tensor((HD, HD));
        let joint_base = cx.tensor((S, D));
        let _velocity = model
            .forward(latent, text, t, guidance, rope_cos, rope_sin, rope_rot, joint_base)
            .output();
        let (pre, _inputs, _outputs, _post) =
            cx.logical.native_parts().expect("recorder clean");
        let full = format!("{}\n\n{pre}", luminal::egglog_snippet::assembled_program());
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph.parse_and_run_program(None, &full).expect("body loads");
        let sizes = |egraph: &mut egglog::EGraph| -> rustc_hash::FxHashMap<String, isize> {
            let out = egraph
                .parse_and_run_program(None, "(print-size)")
                .expect("sizes");
            let mut map = rustc_hash::FxHashMap::default();
            for chunk in &out {
                for line in chunk.to_string().lines() {
                    if let Some((name, count)) = line.rsplit_once(": ") {
                        if let Ok(count) = count.trim().parse::<isize>() {
                            map.insert(name.trim().to_string(), count);
                        }
                    }
                }
            }
            map
        };
        let mut previous = sizes(&mut egraph);
        for round in 1..=200 {
            let start = std::time::Instant::now();
            let round_out = egraph.parse_and_run_program(None, "(run 1)").expect("round");
            let current = sizes(&mut egraph);
            let total: isize = current.values().sum();
            let mut deltas: Vec<(String, isize)> = current
                .iter()
                .map(|(name, &count)| {
                    (name.clone(), count - previous.get(name).copied().unwrap_or(0))
                })
                .filter(|(_, delta)| *delta != 0)
                .collect();
            deltas.sort_by_key(|(_, delta)| -*delta);
            let grew: isize = deltas.iter().map(|(_, delta)| *delta).sum();
            let top: Vec<String> = deltas
                .iter()
                .take(6)
                .map(|(name, delta)| format!("{name} {delta:+}"))
                .collect();
            if grew > 400 || round % 10 == 0 || grew == 0 {
                eprintln!(
                    "[dit-rounds] round {round}: total {total} ({grew:+}) in {:.2}s | {}",
                    start.elapsed().as_secs_f64(),
                    top.join(", ")
                );
            }
            if grew > 2000 {
                for chunk in &round_out {
                    let egglog::CommandOutput::RunSchedule(report) = chunk else {
                        continue;
                    };
                    let mut rules: Vec<(String, usize)> = report
                        .num_matches_per_rule
                        .iter()
                        .map(|(name, &matches)| (name.to_string(), matches))
                        .collect();
                    rules.sort_by_key(|(_, matches)| std::cmp::Reverse(*matches));
                    for (name, matches) in rules.iter().take(4) {
                        let flat: String =
                            name.split_whitespace().collect::<Vec<_>>().join(" ");
                        eprintln!(
                            "[dit-rounds]   x{matches} {}",
                            flat.chars().take(110).collect::<String>()
                        );
                    }
                }
            }
            if grew > 200_000 {
                eprintln!("[dit-rounds] BAIL: runaway round");
                break;
            }
            if grew == 0 {
                eprintln!("[dit-rounds] SATURATED at round {round}");
                break;
            }
            previous = current;
        }
    }

    /// STAGE-BISECT PROBE for the MiniDit saturation blowup (2026-08-10:
    /// the batch run burned 10+ minutes inside egglog free-join at
    /// bounded ~2GB RSS — a tuple/time explosion, not memory). Builds
    /// the DiT graph construct by construct with a 1-genome budget and
    /// prints saturation wall per stage: the first stage that never
    /// prints (or jumps seconds → minutes) names the construct. Run:
    /// cargo test --release -p luminal_nn probe_dit_stages -- --ignored --nocapture
    #[test]
    #[ignore = "diagnosis probe — run explicitly by name (release, bounded)"]
    fn probe_dit_stages() {
        const IN_CH: usize = 4;
        const TXT_DIM: usize = 6;
        const D: usize = 16;
        const NH: usize = 2;
        const HD: usize = 8;
        const MLP: usize = 6;
        const T_HALF: usize = 2;
        const S_TXT: usize = 2;
        const GRID: usize = 2;
        const S_IMG: usize = GRID * GRID;
        const S: usize = S_TXT + S_IMG;
        let budget = luminal::implementation_search::ImplementationSearchOptions {
            generations: 1,
            generation_size: 1,
            mutations: 1,
            trials: 1,
            seed: 0,
        };

        for stage in [1usize, 2, 3, 4, 5, 6, 8, 9, 10, 7] {
            let start = std::time::Instant::now();
            let mut cx = Graph::new();
            let model = MiniDit::new(IN_CH, TXT_DIM, D, NH, MLP, T_HALF, S_TXT, &mut cx);
            let latent = cx.tensor((S_IMG, IN_CH));
            let text = cx.tensor((S_TXT, TXT_DIM));
            let t = cx.tensor(1);
            let guidance = cx.tensor(1);
            let rope_cos = cx.tensor((S, HD));
            let rope_sin = cx.tensor((S, HD));
            let rope_rot = cx.tensor((HD, HD));
            let joint_base = cx.tensor((S, D));
            let ln = crate::LayerNorm::new(D, None, None, true, 1e-6, &mut cx);

            // ---- partial forward, mirroring MiniDit::forward ----
            let temb = model
                .t_mlp2
                .forward(model.t_mlp1.forward(model.sinusoid(t)).silu())
                + model
                    .g_mlp2
                    .forward(model.g_mlp1.forward(model.sinusoid(guidance)).silu());
            let cond = temb.silu();
            let m_img = model.mod_img.forward(cond);
            let m_txt = model.mod_txt.forward(cond);
            let m_single = model.mod_single.forward(cond);
            let triple = |m: GraphTensor, set: usize| {
                let base = set * 3 * D;
                (
                    m.slice_along(base..base + D, 1),
                    m.slice_along(base + D..base + 2 * D, 1),
                    m.slice_along(base + 2 * D..base + 3 * D, 1),
                )
            };
            let ada = |x: GraphTensor, scale: GraphTensor, shift: GraphTensor| {
                let dims = x.dims();
                x * (scale + 1.0).expand(dims.clone()) + shift.expand(dims)
            };
            let gate = |x: GraphTensor, g: GraphTensor| {
                let dims = x.dims();
                x * g.expand(dims)
            };
            let heads = |x: GraphTensor| x.split_dims(1, HD).permute((1, 0, 2));
            let unheads = |x: GraphTensor| x.permute((1, 0, 2)).merge_dims(1, 2);
            let head_rms = |x: GraphTensor, weight: GraphTensor| {
                let dims = x.dims();
                let inv = ((x * x).mean(2) + 1e-6).sqrt().reciprocal();
                x * inv.unsqueeze(2).expand(dims.clone())
                    * weight.unsqueeze(0).unsqueeze(0).expand(dims)
            };
            let rope = |x: GraphTensor| {
                // matmul-form rotation — mirrors MiniDit::forward
                let dims = x.dims();
                let rotated = x.matmul(rope_rot);
                x * rope_cos.unsqueeze(0).expand(dims.clone())
                    + rotated * rope_sin.unsqueeze(0).expand(dims)
            };
            let sdpa = |q: GraphTensor, k: GraphTensor, v: GraphTensor| {
                let scale = 1.0 / (HD as f32).sqrt();
                (q.matmul(k.permute((0, 2, 1))) * scale).softmax(2).matmul(v)
            };
            let swiglu =
                |u: GraphTensor| u.slice_along(0..MLP, 1).silu() * u.slice_along(MLP..2 * MLP, 1);

            let out: GraphTensor = if stage == 7 {
                // The REAL full forward (head included) — 1-genome
                // budget: discriminates graph divergence from
                // search-loop memory at the default 64-genome budget.
                model.forward(latent, text, t, guidance, rope_cos, rope_sin, rope_rot, joint_base)
            } else if stage == 1 {
                m_single
            } else {
                let (shift0, scale0, gate0) = triple(m_img, 0);
                let (shift1, scale1, gate1) = triple(m_img, 1);
                let (c_shift0, c_scale0, c_gate0) = triple(m_txt, 0);
                let (c_shift1, c_scale1, c_gate1) = triple(m_txt, 1);
                let img = model.x_embed.forward(latent);
                let txt = model.ctx_embed.forward(text);
                let img_n = ada(ln.forward(img), scale0, shift0);
                let txt_n = ada(ln.forward(txt), c_scale0, c_shift0);
                if stage == 2 {
                    img_n + txt_n.mean(0).expand_lhs((S_IMG,))
                } else {
                    let q_img = head_rms(heads(model.img_q.forward(img_n)), model.img_qnorm);
                    let k_img = head_rms(heads(model.img_k.forward(img_n)), model.img_knorm);
                    let q_txt = head_rms(heads(model.txt_q.forward(txt_n)), model.txt_qnorm);
                    let k_txt = head_rms(heads(model.txt_k.forward(txt_n)), model.txt_knorm);
                    let v_all = heads(
                        model
                            .txt_v
                            .forward(txt_n)
                            .concat_along(model.img_v.forward(img_n), 0),
                    );
                    let q_all = q_txt.concat_along(q_img, 1);
                    if stage == 3 {
                        unheads(q_all)
                    } else if stage == 4 {
                        unheads(rope(q_all))
                    } else {
                        let attn = unheads(sdpa(
                            rope(q_all),
                            rope(k_txt.concat_along(k_img, 1)),
                            v_all,
                        ));
                        let attn_txt = attn.slice_along(0..S_TXT, 0);
                        let attn_img = attn.slice_along(S_TXT.., 0);
                        let img = img + gate(model.img_out.forward(attn_img), gate0);
                        let txt = txt + gate(model.txt_out.forward(attn_txt), c_gate0);
                        let ff = swiglu(model.ff_in.forward(ada(ln.forward(img), scale1, shift1)));
                        let img = img + gate(model.ff_out.forward(ff), gate1);
                        let c_ff = swiglu(
                            model
                                .ctx_ff_in
                                .forward(ada(ln.forward(txt), c_scale1, c_shift1)),
                        );
                        let txt = txt + gate(model.ctx_ff_out.forward(c_ff), c_gate1);
                        if stage == 5 {
                            img + txt.mean(0).expand_lhs((S_IMG,))
                        } else {
                            let graph = latent.graph();
                            let txt_positions = graph.arange(S_TXT);
                            let img_positions =
                                graph.iota(S_IMG, move |c| c[0] + S_TXT);
                            let hidden = crate::scatter_rows(
                                img,
                                img_positions,
                                crate::scatter_rows(txt, txt_positions, joint_base, D),
                                D,
                            );
                            let (s_shift, s_scale, s_gate) = triple(m_single, 0);
                            let normed = ada(ln.forward(hidden), s_scale, s_shift);
                            let proj = model.single_proj.forward(normed);
                            let q = head_rms(heads(proj.slice_along(0..D, 1)), model.single_qnorm);
                            let k =
                                head_rms(heads(proj.slice_along(D..2 * D, 1)), model.single_knorm);
                            let v = heads(proj.slice_along(2 * D..3 * D, 1));
                            let attn = unheads(sdpa(rope(q), rope(k), v));
                            let mlp_out = swiglu(proj.slice_along(3 * D..3 * D + 2 * MLP, 1));
                            let hidden_out = hidden
                                + gate(
                                    model.single_out_attn.forward(attn)
                                        + model.single_out_mlp.forward(mlp_out),
                                    s_gate,
                                );
                            // Stage 6→7 delta sub-bisect: 8 = +slice,
                            // 9 = +no-affine LN, 10 = +adaLN head
                            // modulation. Stage 7 (the real forward)
                            // adds proj_out on top of stage 10's shape.
                            if stage == 6 {
                                hidden_out
                            } else {
                                let img_final = hidden_out.slice_along(S_TXT.., 0);
                                if stage == 8 {
                                    img_final
                                } else if stage == 9 {
                                    ln.forward(img_final)
                                } else {
                                    let head = model.norm_out.forward(cond);
                                    let scale_head = head.slice_along(0..D, 1);
                                    let shift_head = head.slice_along(D..2 * D, 1);
                                    ada(ln.forward(img_final), scale_head, shift_head)
                                }
                            }
                        }
                    }
                }
            };
            let out = out.output();
            let _ = out;

            let pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
                (latent.id, weights(S_IMG * IN_CH, 540)),
                (text.id, weights(S_TXT * TXT_DIM, 541)),
                (t.id, vec![0.35]),
                (guidance.id, vec![0.8]),
                (rope_cos.id, mini_dit_rope_tables(S_TXT, GRID, GRID).0),
                (rope_sin.id, mini_dit_rope_tables(S_TXT, GRID, GRID).1),
                (rope_rot.id, crate::rope_pairing_matrix(HD, true)),
                (joint_base.id, vec![0.0; S * D]),
                (model.x_embed.weight.id, weights(IN_CH * D, 500)),
                (model.ctx_embed.weight.id, weights(TXT_DIM * D, 501)),
                (model.t_mlp1.weight.id, weights(2 * T_HALF * D, 502)),
                (model.t_mlp2.weight.id, weights(D * D, 503)),
                (model.g_mlp1.weight.id, weights(2 * T_HALF * D, 504)),
                (model.g_mlp2.weight.id, weights(D * D, 505)),
                (model.mod_img.weight.id, weights(D * 6 * D, 506)),
                (model.mod_txt.weight.id, weights(D * 6 * D, 507)),
                (model.mod_single.weight.id, weights(D * 3 * D, 508)),
                (model.norm_out.weight.id, weights(D * 2 * D, 509)),
                (model.proj_out.weight.id, weights(D * IN_CH, 510)),
                (model.img_q.weight.id, weights(D * D, 511)),
                (model.img_k.weight.id, weights(D * D, 512)),
                (model.img_v.weight.id, weights(D * D, 513)),
                (model.img_out.weight.id, weights(D * D, 514)),
                (model.txt_q.weight.id, weights(D * D, 515)),
                (model.txt_k.weight.id, weights(D * D, 516)),
                (model.txt_v.weight.id, weights(D * D, 517)),
                (model.txt_out.weight.id, weights(D * D, 518)),
                (model.img_qnorm.id, weights(HD, 519)),
                (model.img_knorm.id, weights(HD, 520)),
                (model.txt_qnorm.id, weights(HD, 521)),
                (model.txt_knorm.id, weights(HD, 522)),
                (model.ff_in.weight.id, weights(D * 2 * MLP, 523)),
                (model.ff_out.weight.id, weights(MLP * D, 524)),
                (model.ctx_ff_in.weight.id, weights(D * 2 * MLP, 525)),
                (model.ctx_ff_out.weight.id, weights(MLP * D, 526)),
                (model.single_proj.weight.id, weights(D * (3 * D + 2 * MLP), 527)),
                (model.single_out_attn.weight.id, weights(D * D, 531)),
                (model.single_out_mlp.weight.id, weights(MLP * D, 532)),
                (model.single_qnorm.id, weights(HD, 529)),
                (model.single_knorm.id, weights(HD, 530)),
            ];
            let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
            let mut rt =
                luminal::ssa_reference::SsaReferenceRuntime::load(&cx).expect("native load");
            match rt.search(&data, &budget) {
                Ok(outcome) => eprintln!(
                    "[dit-probe] stage {stage}: wall {:.1}s | {}",
                    start.elapsed().as_secs_f64(),
                    outcome.timings.summary()
                ),
                Err(err) => eprintln!(
                    "[dit-probe] stage {stage}: wall {:.1}s | search refused: {err:#}",
                    start.elapsed().as_secs_f64()
                ),
            }
        }
    }
}
