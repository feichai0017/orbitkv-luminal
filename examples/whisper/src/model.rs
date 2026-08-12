//! openai/whisper-tiny.en as pure logical ops — 100% of THIS
//! checkpoint (ruling 2026-08-12): conv frontend (k3s1 + k3s2, exact
//! erf GELU), a stored sinusoidal position table on the encoder and a
//! LEARNED position table on the decoder, torch-style LayerNorm
//! (mean + std, learned weight AND bias, eps 1e-5), the HF whisper
//! projection-bias asymmetry (q/v/out biased, k unbiased), plain MHA
//! 6×64 with 1/8 on the scores, causal cached decoder self-attention
//! over the slot pool, UNCACHED cross-attention whose K/V recompute
//! from the encoder output every step (encoder + decoder are ONE
//! graph — faithful to the parked example), exact-erf GELU MLPs, and
//! the TIED output head. No rope, no RMSNorm, no GQA.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal::shape::Expression;
use luminal_nn::{Embedding, KvCachePool, LayerNorm, Linear};

#[derive(Clone)]
pub struct WhisperDims {
    pub n_mels: usize,
    pub audio_ctx: usize,
    pub state: usize,
    pub heads: usize,
    pub audio_layers: usize,
    pub text_layers: usize,
    pub text_ctx: usize,
    pub vocab: usize,
    pub ff: usize,
    pub eps: f32,
}

impl WhisperDims {
    pub fn whisper_tiny_en() -> Self {
        Self {
            n_mels: 80,
            audio_ctx: 1500,
            state: 384,
            heads: 6,
            audio_layers: 4,
            text_layers: 4,
            text_ctx: 448,
            vocab: 51_864,
            ff: 1536,
            eps: 1e-5,
        }
    }

    pub fn tiny() -> Self {
        Self {
            n_mels: 4,
            audio_ctx: 6,
            state: 8,
            heads: 2,
            audio_layers: 2,
            text_layers: 2,
            text_ctx: 8,
            vocab: 19,
            ff: 12,
            eps: 1e-5,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.state / self.heads
    }

    /// Mel frames before the stride-2 conv halves them.
    pub fn mel_frames(&self) -> usize {
        self.audio_ctx * 2
    }
}

fn conv1d_bias(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> GraphTensor {
    let padded = x.pad(
        vec![
            (Expression::from(0), Expression::from(0)),
            (Expression::from(padding), Expression::from(padding)),
        ],
        0.0,
    );
    let unfolded = padded
        .unfold([1usize, kernel], [1usize, stride], [1usize, 1usize])
        .squeeze(2);
    let flat = unfolded.permute((1, 0, 2)).merge_dims(1, 2);
    let out = flat.matmul(weight.permute((1, 0)));
    let n_windows = out.dims()[0];
    (out + bias.expand_dim(0, n_windows)).transpose(0, 1)
}

/// The HF whisper bias asymmetry: q/v/out biased, k unbiased.
pub struct WhisperAttn {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    heads: usize,
    head_dim: usize,
}

impl WhisperAttn {
    fn new(prefix: &str, d: &WhisperDims, cx: &mut Graph) -> Self {
        // Names ride the Linear's own tensors; binding uses weight_bindings.
        let _ = prefix;
        Self {
            q: Linear::new_permuted(d.state, d.state, true, cx),
            k: Linear::new_permuted(d.state, d.state, false, cx),
            v: Linear::new_permuted(d.state, d.state, true, cx),
            out: Linear::new_permuted(d.state, d.state, true, cx),
            heads: d.heads,
            head_dim: d.head_dim(),
        }
    }

    /// Bidirectional maskless attention (encoder self / cross): q from
    /// `x`, k/v from `kv_source`, scale 1/sqrt(head_dim) on scores.
    fn bidirectional(&self, x: GraphTensor, kv_source: GraphTensor) -> GraphTensor {
        let attn = luminal_nn::attention(
            self.q.forward(x),
            self.k.forward(kv_source),
            self.v.forward(kv_source),
            self.heads,
            self.head_dim,
        );
        self.out.forward(attn)
    }

    /// Causal cached decoder self-attention over the slot pool.
    #[allow(clippy::too_many_arguments)]
    fn causal_cached(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        q_pos: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let (attn, k_cache, v_cache) = luminal_nn::paged_attention_positional(
            self.q.forward(x),
            self.k.forward(x),
            self.v.forward(x),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            q_pos,
            self.heads,
            self.heads, // MHA: no KV grouping
            self.head_dim,
            None,
            1.0 / (self.head_dim as f32).sqrt(),
        );
        (self.out.forward(attn), k_cache, v_cache)
    }
}

fn layer_norm(d: &WhisperDims, cx: &mut Graph) -> LayerNorm {
    // torch LayerNorm: mean-centered, learned weight AND bias.
    LayerNorm::new(d.state, Some("w"), Some("b"), true, d.eps, cx)
}

pub struct EncoderLayer {
    pub attn_norm: LayerNorm,
    pub attn: WhisperAttn,
    pub ff_norm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct DecoderLayer {
    pub self_norm: LayerNorm,
    pub self_attn: WhisperAttn,
    pub cross_norm: LayerNorm,
    pub cross_attn: WhisperAttn,
    pub ff_norm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct Whisper {
    pub dims: WhisperDims,
    pub conv1_w: GraphTensor,
    pub conv1_b: GraphTensor,
    pub conv2_w: GraphTensor,
    pub conv2_b: GraphTensor,
    pub enc_pos: GraphTensor,
    pub enc_layers: Vec<EncoderLayer>,
    pub enc_final_norm: LayerNorm,
    pub embed: Embedding,
    pub dec_pos: GraphTensor,
    pub dec_layers: Vec<DecoderLayer>,
    pub dec_final_norm: LayerNorm,
}

impl Whisper {
    pub fn init(cx: &mut Graph, d: &WhisperDims) -> Self {
        let enc_layers = (0..d.audio_layers)
            .map(|_| EncoderLayer {
                attn_norm: layer_norm(d, cx),
                attn: WhisperAttn::new("enc", d, cx),
                ff_norm: layer_norm(d, cx),
                fc1: Linear::new_permuted(d.state, d.ff, true, cx),
                fc2: Linear::new_permuted(d.ff, d.state, true, cx),
            })
            .collect();
        let dec_layers = (0..d.text_layers)
            .map(|_| DecoderLayer {
                self_norm: layer_norm(d, cx),
                self_attn: WhisperAttn::new("dec", d, cx),
                cross_norm: layer_norm(d, cx),
                cross_attn: WhisperAttn::new("cross", d, cx),
                ff_norm: layer_norm(d, cx),
                fc1: Linear::new_permuted(d.state, d.ff, true, cx),
                fc2: Linear::new_permuted(d.ff, d.state, true, cx),
            })
            .collect();
        Self {
            dims: d.clone(),
            conv1_w: cx.named_tensor("conv1.weight", (d.state, d.n_mels * 3)),
            conv1_b: cx.named_tensor("conv1.bias", d.state),
            conv2_w: cx.named_tensor("conv2.weight", (d.state, d.state * 3)),
            conv2_b: cx.named_tensor("conv2.bias", d.state),
            enc_pos: cx.named_tensor("enc.pos", (d.audio_ctx, d.state)),
            enc_layers,
            enc_final_norm: layer_norm(d, cx),
            embed: Embedding::new(d.vocab, d.state, cx),
            dec_pos: cx.named_tensor("dec.pos", (d.text_ctx, d.state)),
            dec_layers,
            dec_final_norm: layer_norm(d, cx),
        }
    }

    /// mel (n_mels, 2·audio_ctx) → encoder output (audio_ctx, state).
    pub fn encode(&self, mel: GraphTensor) -> GraphTensor {
        let x = conv1d_bias(mel, self.conv1_w, self.conv1_b, 3, 1, 1).gelu();
        let x = conv1d_bias(x, self.conv2_w, self.conv2_b, 3, 2, 1).gelu();
        let mut x = x.transpose(0, 1) + self.enc_pos;
        for layer in &self.enc_layers {
            x = x + layer.attn.bidirectional(layer.attn_norm.forward(x), layer.attn_norm.forward(x));
            let ff_in = layer.ff_norm.forward(x);
            x = x + layer.fc2.forward(layer.fc1.forward(ff_in).gelu());
        }
        self.enc_final_norm.forward(x)
    }

    /// One decode step: token (1,) Int at position q_pos, over the
    /// per-layer self-attention slot pools; cross-attention K/V
    /// recompute from `xa` (the encoder output) in the same graph.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step(
        &self,
        token: GraphTensor,
        q_pos: GraphTensor,
        xa: GraphTensor,
        pool: &KvCachePool,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(token.dtype, DType::Int);
        let pos_row = luminal_nn::gather_rows(self.dec_pos, q_pos, self.dims.state);
        let mut x = self.embed.forward(token) + pos_row;
        let mut caches_out = Vec::with_capacity(self.dec_layers.len());
        for (layer_index, layer) in self.dec_layers.iter().enumerate() {
            let (attn, k_cache, v_cache) = layer.self_attn.causal_cached(
                layer.self_norm.forward(x),
                pool.layers[layer_index].0,
                pool.layers[layer_index].1,
                gather_idx,
                scatter_idx,
                q_pos,
            );
            x = x + attn;
            caches_out.push((k_cache, v_cache));
            x = x + layer
                .cross_attn
                .bidirectional(layer.cross_norm.forward(x), xa);
            let ff_in = layer.ff_norm.forward(x);
            x = x + layer.fc2.forward(layer.fc1.forward(ff_in).gelu());
        }
        let x = self.dec_final_norm.forward(x);
        (self.embed.reverse(x), caches_out)
    }

    pub fn weight_bindings(&self) -> Vec<(String, GraphTensor)> {
        let mut map = vec![
            ("model.encoder.conv1.weight".to_string(), self.conv1_w),
            ("model.encoder.conv1.bias".to_string(), self.conv1_b),
            ("model.encoder.conv2.weight".to_string(), self.conv2_w),
            ("model.encoder.conv2.bias".to_string(), self.conv2_b),
            ("model.encoder.embed_positions.weight".to_string(), self.enc_pos),
            ("model.decoder.embed_tokens.weight".to_string(), self.embed.weight),
            ("model.decoder.embed_positions.weight".to_string(), self.dec_pos),
        ];
        let mut attn = |prefix: String, a: &WhisperAttn, map: &mut Vec<(String, GraphTensor)>| {
            map.push((format!("{prefix}.q_proj.weight"), a.q.weight));
            map.push((format!("{prefix}.q_proj.bias"), a.q.bias.expect("q bias")));
            map.push((format!("{prefix}.k_proj.weight"), a.k.weight));
            map.push((format!("{prefix}.v_proj.weight"), a.v.weight));
            map.push((format!("{prefix}.v_proj.bias"), a.v.bias.expect("v bias")));
            map.push((format!("{prefix}.out_proj.weight"), a.out.weight));
            map.push((format!("{prefix}.out_proj.bias"), a.out.bias.expect("out bias")));
        };
        let mut norm = |prefix: String, n: &LayerNorm, map: &mut Vec<(String, GraphTensor)>| {
            map.push((format!("{prefix}.weight"), n.weight.expect("ln weight")));
            map.push((format!("{prefix}.bias"), n.bias.expect("ln bias")));
        };
        for (l, layer) in self.enc_layers.iter().enumerate() {
            let p = format!("model.encoder.layers.{l}");
            norm(format!("{p}.self_attn_layer_norm"), &layer.attn_norm, &mut map);
            attn(format!("{p}.self_attn"), &layer.attn, &mut map);
            norm(format!("{p}.final_layer_norm"), &layer.ff_norm, &mut map);
            map.push((format!("{p}.fc1.weight"), layer.fc1.weight));
            map.push((format!("{p}.fc1.bias"), layer.fc1.bias.expect("fc1 bias")));
            map.push((format!("{p}.fc2.weight"), layer.fc2.weight));
            map.push((format!("{p}.fc2.bias"), layer.fc2.bias.expect("fc2 bias")));
        }
        norm("model.encoder.layer_norm".to_string(), &self.enc_final_norm, &mut map);
        for (l, layer) in self.dec_layers.iter().enumerate() {
            let p = format!("model.decoder.layers.{l}");
            norm(format!("{p}.self_attn_layer_norm"), &layer.self_norm, &mut map);
            attn(format!("{p}.self_attn"), &layer.self_attn, &mut map);
            norm(format!("{p}.encoder_attn_layer_norm"), &layer.cross_norm, &mut map);
            attn(format!("{p}.encoder_attn"), &layer.cross_attn, &mut map);
            norm(format!("{p}.final_layer_norm"), &layer.ff_norm, &mut map);
            map.push((format!("{p}.fc1.weight"), layer.fc1.weight));
            map.push((format!("{p}.fc1.bias"), layer.fc1.bias.expect("fc1 bias")));
            map.push((format!("{p}.fc2.weight"), layer.fc2.weight));
            map.push((format!("{p}.fc2.bias"), layer.fc2.bias.expect("fc2 bias")));
        }
        norm("model.decoder.layer_norm".to_string(), &self.dec_final_norm, &mut map);
        map
    }
}
