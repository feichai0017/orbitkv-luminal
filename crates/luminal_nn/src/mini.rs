//! MINI MODELS (ruling 2026-08-07): one small, runnable representative
//! per example-model FAMILY, defined here in core — the runtimes' example
//! directories only RUN them. Coverage: llama/qwen/paged (MiniLlama),
//! gemma (MiniLlama with GeGLU), the MoE decoders (MiniMoe), whisper
//! (MiniWhisper — encoder self-attention + cross-attention, the one
//! construct no other family exercises), yolo (MiniConvNet). flux2's
//! diffusion transformer is NOT yet covered (adaLN-modulated blocks —
//! deferred, noted honestly).

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

/// MiniLlama: embed → N × LlamaBlock (paged KV cache) → final RMSNorm →
/// tied logits. `GatedFfn::GeGlu` makes it the gemma-family mini.
pub struct MiniLlama {
    pub embed: Embedding,
    pub blocks: Vec<LlamaBlock>,
    pub final_norm: crate::LayerNorm,
}

impl MiniLlama {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        layers: usize,
        ffn: GatedFfn,
        cx: &mut Graph,
    ) -> Self {
        let ffn_of = |_: usize| match ffn {
            GatedFfn::SwiGlu => GatedFfn::SwiGlu,
            GatedFfn::GeGlu => GatedFfn::GeGlu,
        };
        Self {
            embed: Embedding::new(vocab, d, cx),
            blocks: (0..layers)
                .map(|layer| LlamaBlock::new_with_ffn(d, ff, n_heads, n_kv_heads, ffn_of(layer), cx))
                .collect(),
            final_norm: crate::LayerNorm::new(d, None, None, false, 1e-5, cx),
        }
    }

    /// ids (s,) Int + one (k, v) cache pair per layer → (logits, caches').
    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: Expression,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = self.embed.forward(ids);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
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
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}

/// MiniMoe: embed → N × DecoderBlock with the MoE feed-forward → final
/// LayerNorm → tied logits (the qwen3_moe/gemma4_moe family).
pub struct MiniMoe {
    pub embed: Embedding,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: crate::LayerNorm,
}

impl MiniMoe {
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
        Self {
            embed: Embedding::new(vocab, d, cx),
            blocks: (0..layers)
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
            final_norm: crate::LayerNorm::new(d, None, None, true, 1e-5, cx),
        }
    }

    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: Expression,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = self.embed.forward(ids);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
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
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
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

#[cfg(test)]
mod tests {
    use super::{attention, MiniConvNet, MiniLlama, MiniMoe, MiniWhisper};
    use crate::test_refs::*;
    use crate::GatedFfn;
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

    /// One llama-block reference step (rms → GQA attention → residual →
    /// rms → gated FFN → residual), gate activation supplied.
    #[allow(clippy::too_many_arguments)]
    fn ref_llama_block(
        x: &[f32],
        seeds: (usize, usize, usize, usize, usize, usize, usize),
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
        let q = ref_matmul(&normed, &weights(d * d, wq_s), d, d);
        let k_new = ref_matmul(&normed, &weights(d * kv_dim, wk_s), d, kv_dim);
        let v_new = ref_matmul(&normed, &weights(d * kv_dim, wv_s), d, kv_dim);
        let attn = ref_paged_step_gqa(
            &q, &k_new, &v_new, k_cache, v_cache, gather, scatter_slot, n_heads, n_kv, head_dim,
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

    /// Family test: MiniLlama (or gemma via GeGlu) — ONE block, one
    /// decode step, default search budget. Depth stays at one block until
    /// the genome-sampling ruling lands: at 2+ blocks (block + embed +
    /// tied logits) even 64 random genomes can all choice-cycle — the
    /// measured cliff (see rung 6).
    fn mini_llama_family(ffn: GatedFfn, gate_act: &dyn Fn(&[f32]) -> Vec<f32>) {
        const VOCAB: usize = 5;
        const D: usize = 8;
        const FF: usize = 12;
        const NH: usize = 4;
        const NKV: usize = 2;
        const HD: usize = 2;
        const KV_DIM: usize = NKV * HD;
        const SLOTS: usize = 4;
        const CTX: usize = 2;
        const LAYERS: usize = 1;
        let token = 3usize;

        let mut cx = Graph::new();
        let model = MiniLlama::new(VOCAB, D, FF, NH, NKV, LAYERS, ffn, &mut cx);
        let ids = cx.tensor_dtyped(1, DType::Int);
        let caches: Vec<_> = (0..LAYERS)
            .map(|_| (cx.tensor((SLOTS, KV_DIM)), cx.tensor((SLOTS, KV_DIM))))
            .collect();
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let (logits, caches_out) =
            model.forward(ids, &caches, gather_idx, scatter_idx, Expression::from(1usize));
        let logits = logits.output();
        let caches_out: Vec<_> = caches_out
            .into_iter()
            .map(|(k, v)| (k.output(), v.output()))
            .collect();

        let embed_w = weights(VOCAB * D, 199);
        let mut pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (ids.id, vec![token as f32]),
            (model.embed.weight.id, embed_w.clone()),
            (gather_idx.id, vec![0.0, 1.0]),
            (scatter_idx.id, vec![1.0]),
        ];
        let mut ref_caches: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for (layer, block) in model.blocks.iter().enumerate() {
            let (wq_s, wk_s, wv_s, wo_s, gate_s, up_s, down_s) = seeds_for(layer);
            pairs.push((block.wq.weight.id, weights(D * D, wq_s)));
            pairs.push((block.wk.weight.id, weights(D * KV_DIM, wk_s)));
            pairs.push((block.wv.weight.id, weights(D * KV_DIM, wv_s)));
            pairs.push((block.wo.weight.id, weights(D * D, wo_s)));
            pairs.push((block.gate.weight.id, weights(D * FF, gate_s)));
            pairs.push((block.up.weight.id, weights(D * FF, up_s)));
            pairs.push((block.down.weight.id, weights(FF * D, down_s)));
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
            x = ref_llama_block(
                &x,
                seeds_for(layer),
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
    fn mini_llama_matches_scalar_reference() {
        mini_llama_family(GatedFfn::SwiGlu, &|x| {
            x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
        });
    }

    #[test]
    fn mini_gemma_matches_scalar_reference() {
        mini_llama_family(GatedFfn::GeGlu, &ref_gelu_tanh);
    }

    /// MiniMoe: 1 block + embed + tied logits, one decode step (run_ssa
    /// harness budget — single layer).
    #[test]
    fn mini_moe_matches_scalar_reference() {
        const VOCAB: usize = 5;
        const D: usize = 4;
        const E: usize = 2;
        const NH: usize = 2;
        const SLOTS: usize = 4;
        const CTX: usize = 2;
        let token = 2usize;

        let mut cx = Graph::new();
        let model = MiniMoe::new(VOCAB, D, E, 1, NH, 1, &mut cx);
        let ids = cx.tensor_dtyped(1, DType::Int);
        let k_cache = cx.tensor((SLOTS, D));
        let v_cache = cx.tensor((SLOTS, D));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let caches = vec![(k_cache, v_cache)];
        let (logits, caches_out) =
            model.forward(ids, &caches, gather_idx, scatter_idx, Expression::from(1usize));
        let logits = logits.output();
        let (kc_out, vc_out) = (caches_out[0].0.output(), caches_out[0].1.output());

        let embed_w = weights(VOCAB * D, 400);
        let block = &model.blocks[0];
        let crate::FeedForward::Moe(moe) = &block.ff else { unreachable!() };
        let pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (ids.id, vec![token as f32]),
            (model.embed.weight.id, embed_w.clone()),
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
        let ref_logits: Vec<f32> = (0..VOCAB)
            .map(|v| (0..D).map(|i| x2[i] * embed_w[v * D + i]).sum())
            .collect();

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
}
