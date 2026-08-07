//! FULL-MODEL tests, smallest first (Austin's directive 2026-08-06):
//! whole models composed from the nn modules, run end-to-end through the
//! native ladder (load → search → execute) against scalar references
//! computed in plain loops. Rungs: (1) MLP; (2) a decoder block with a
//! KV cache — Embedding → paged attention → FFN → tied logits; (3) the
//! same block with the MoE FFN; (4) a two-layer decoder with LayerNorm
//! driven for TWO decode steps, the caches flowing runtime-out →
//! runtime-in between steps.

use crate::{Embedding, Linear, MoE, paged_attention};
use luminal::prelude::*;
use luminal::shape::Expression;

/// The smallest true model: Linear → relu → Linear → relu → Linear.
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Mlp {
    /// `dims` = [in, hidden.., out]; a relu follows every layer but the
    /// last.
    pub fn new(dims: &[usize], cx: &mut Graph) -> Self {
        assert!(dims.len() >= 2, "an MLP needs at least in and out dims");
        let layers = dims
            .windows(2)
            .map(|pair| Linear::new(pair[0], pair[1], true, cx))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, mut x: GraphTensor) -> GraphTensor {
        let last = self.layers.len() - 1;
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if index != last {
                x = x.relu();
            }
        }
        x
    }
}

/// The FFN flavor a decoder block carries.
pub enum FeedForward {
    /// up → relu → down.
    Dense { up: Linear, down: Linear },
    /// Mixture of experts (top-k routing).
    Moe(MoE),
}

impl FeedForward {
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        match self {
            FeedForward::Dense { up, down } => down.forward(up.forward(x).relu()),
            FeedForward::Moe(moe) => moe.forward(x),
        }
    }
}

/// MODEL 2/3: one decoder block over a paged KV cache. Token ids embed,
/// attention reads/writes the cache, the FFN applies, and logits come
/// from the tied embedding (`Embedding::reverse`). Residuals around both
/// sublayers; optional pre-norms arrive with [`TinyDecoder`].
pub struct DecoderBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ff: FeedForward,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl DecoderBlock {
    /// x (s, d_model) + cache slots → (x', k_cache', v_cache').
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: Expression,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let (attn, k_cache, v_cache) = paged_attention(
            self.wq.forward(x),
            self.wk.forward(x),
            self.wv.forward(x),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            prev_seq,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );
        let x = x + self.wo.forward(attn);
        let x = x + self.ff.forward(x);
        (x, k_cache, v_cache)
    }
}

/// MODEL 4: a two-layer decoder — per-layer pre-LayerNorm, per-layer KV
/// caches, tied logits through a final norm. One forward = one decode
/// step; multi-step decode re-runs the (shape-specialized) program with
/// the cache OUTPUTS flowing back in as the next step's cache INPUTS.
pub struct TinyDecoder {
    pub embed: Embedding,
    pub norms: Vec<crate::LayerNorm>,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: crate::LayerNorm,
}

impl TinyDecoder {
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
            x = self.norms[layer].forward(x);
            let (next, k_cache, v_cache) = block.forward(
                x,
                caches[layer].0,
                caches[layer].1,
                gather_idx,
                scatter_idx,
                prev_seq,
            );
            x = next;
            caches_out.push((k_cache, v_cache));
        }
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}

#[cfg(test)]
mod tests {
    use super::{DecoderBlock, FeedForward, Mlp, TinyDecoder};
    use crate::{Embedding, Linear, MoE};
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::shape::Expression;
    use luminal::ssa_reference::SsaReferenceRuntime;
    use rustc_hash::FxHashMap;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    /// Deterministic pseudo-random weights (no RNG dependency; values in
    /// roughly [-0.6, 0.6] so activations stay in a well-conditioned
    /// range).
    fn weights(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
            .collect()
    }

    /// MODEL 1: a full 4→8→6→3 MLP, batch 2, through the search ladder —
    /// every layer's weights and biases bound as named tensors, the whole
    /// forward against a scalar reference.
    #[test]
    fn mlp_forward_matches_scalar_reference() {
        const DIMS: [usize; 4] = [4, 8, 6, 3];
        const BATCH: usize = 2;

        let mut cx = Graph::new();
        let model = Mlp::new(&DIMS, &mut cx);
        let x = cx.tensor((BATCH, DIMS[0]));
        let out = model.forward(x).output();

        let x_data = weights(BATCH * DIMS[0], 7);
        let mut layer_data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for (index, pair) in DIMS.windows(2).enumerate() {
            layer_data.push((
                weights(pair[0] * pair[1], index),
                weights(pair[1], index + 50),
            ));
        }

        // Scalar reference.
        let mut activation = x_data.clone();
        let mut width = DIMS[0];
        for (index, pair) in DIMS.windows(2).enumerate() {
            let (w, b) = &layer_data[index];
            let (in_w, out_w) = (pair[0], pair[1]);
            let mut next = vec![0f32; BATCH * out_w];
            for r in 0..BATCH {
                for c in 0..out_w {
                    let mut acc = b[c];
                    for k in 0..in_w {
                        acc += activation[r * width + k] * w[k * out_w + c];
                    }
                    next[r * out_w + c] =
                        if index != DIMS.len() - 2 { acc.max(0.0) } else { acc };
                }
            }
            activation = next;
            width = out_w;
        }

        let mut data = FxHashMap::default();
        data.insert(x.id, x_data.clone());
        for (layer, (w, b)) in model.layers.iter().zip(&layer_data) {
            data.insert(layer.weight.id, w.clone());
            data.insert(layer.bias.unwrap().id, b.clone());
        }
        let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
        let outcome = rt
            .search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        eprintln!(
            "[mlp] search attribution: {} (plans profiled {}, cache hits {})",
            outcome.timings.summary(),
            outcome.plans_profiled,
            outcome.fingerprint_hits
        );
        rt.set_data(x.id, x_data);
        for (layer, (w, b)) in model.layers.iter().zip(&layer_data) {
            rt.set_data(layer.weight.id, w.clone());
            rt.set_data(layer.bias.unwrap().id, b.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &activation);
    }




    // ── shared scalar-reference pieces (single query row, s = 1) ──

    /// y = x · W, x (1, in), W (in, out).
    fn ref_matmul(x: &[f32], w: &[f32], in_w: usize, out_w: usize) -> Vec<f32> {
        (0..out_w)
            .map(|c| (0..in_w).map(|k| x[k] * w[k * out_w + c]).sum())
            .collect()
    }

    /// One decode step of paged attention for a single new token at
    /// position `prev_seq`, all context causally visible.
    #[allow(clippy::too_many_arguments)]
    fn ref_paged_step(
        q: &[f32],
        k_new: &[f32],
        v_new: &[f32],
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        gather: &[usize],
        scatter_slot: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let kv_dim = n_heads * head_dim; // kv_groups = 1 in every model test
        k_cache[scatter_slot * kv_dim..(scatter_slot + 1) * kv_dim].copy_from_slice(k_new);
        v_cache[scatter_slot * kv_dim..(scatter_slot + 1) * kv_dim].copy_from_slice(v_new);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0f32; kv_dim];
        for h in 0..n_heads {
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let scores: Vec<f32> = gather
                .iter()
                .map(|slot| {
                    let k_row = &k_cache[slot * kv_dim + h * head_dim..][..head_dim];
                    q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for (j, slot) in gather.iter().enumerate() {
                let v_row = &v_cache[slot * kv_dim + h * head_dim..][..head_dim];
                for (d, v) in v_row.iter().enumerate() {
                    out[h * head_dim + d] += exps[j] / denom * v;
                }
            }
        }
        out
    }

    /// LayerNorm as the nn module computes it: mean-subtract, then
    /// x / sqrt(mean(x²) + eps), no affine.
    fn ref_layer_norm(x: &[f32], epsilon: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let centered: Vec<f32> = x.iter().map(|v| v - mean).collect();
        let ms: f32 = centered.iter().map(|v| v * v).sum::<f32>() / n;
        let inv = 1.0 / (ms + epsilon).sqrt();
        centered.iter().map(|v| v * inv).collect()
    }

    /// MoE forward for one row, k = 1: softmax routing, best expert's
    /// matmul scaled by its routing weight.
    fn ref_moe_k1(x: &[f32], router: &[f32], experts: &[f32], d: usize, e_count: usize) -> Vec<f32> {
        let logits: Vec<f32> = (0..e_count)
            .map(|e| (0..d).map(|i| x[i] * router[i * e_count + e]).sum())
            .collect();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let denom: f32 = exps.iter().sum();
        let best = (0..e_count)
            .max_by(|a, b| logits[*a].partial_cmp(&logits[*b]).unwrap())
            .unwrap();
        let weight = exps[best] / denom;
        let w = &experts[best * d * d..(best + 1) * d * d];
        ref_matmul(x, w, d, d).iter().map(|v| v * weight).collect()
    }

    /// The whole decoder-block reference: x' = x + Wo·attn; x'' = x' + ff(x').
    #[allow(clippy::too_many_arguments)]
    fn ref_block_step(
        x: &[f32],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        ff: &dyn Fn(&[f32]) -> Vec<f32>,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        gather: &[usize],
        scatter_slot: usize,
        n_heads: usize,
        head_dim: usize,
        d: usize,
    ) -> Vec<f32> {
        let q = ref_matmul(x, wq, d, d);
        let k = ref_matmul(x, wk, d, d);
        let v = ref_matmul(x, wv, d, d);
        let attn = ref_paged_step(
            &q, &k, &v, k_cache, v_cache, gather, scatter_slot, n_heads, head_dim,
        );
        let attn_proj = ref_matmul(&attn, wo, d, d);
        let x1: Vec<f32> = x.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();
        let ff_out = ff(&x1);
        x1.iter().zip(&ff_out).map(|(a, b)| a + b).collect()
    }

    struct BlockFixture {
        cx: Graph,
        block: DecoderBlock,
        embed: Embedding,
        ids: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    }

    const D: usize = 4; // d_model = n_heads · head_dim
    const N_HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const VOCAB: usize = 5;
    const SLOTS: usize = 4;
    const CTX: usize = 2;
    const PREV_SEQ: usize = 1;
    const FF_HIDDEN: usize = 6;
    const EXPERTS: usize = 2;

    fn block_fixture(ff: fn(&mut Graph) -> FeedForward) -> (BlockFixture, GraphTensor, GraphTensor, GraphTensor) {
        let mut cx = Graph::new();
        let embed = Embedding::new(VOCAB, D, &mut cx);
        let block = DecoderBlock {
            wq: Linear::new(D, D, false, &mut cx),
            wk: Linear::new(D, D, false, &mut cx),
            wv: Linear::new(D, D, false, &mut cx),
            wo: Linear::new(D, D, false, &mut cx),
            ff: ff(&mut cx),
            n_heads: N_HEADS,
            n_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
        };
        let ids = cx.tensor_dtyped(1, DType::Int);
        let k_cache = cx.tensor((SLOTS, D));
        let v_cache = cx.tensor((SLOTS, D));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let x = embed.forward(ids);
        let (x, kc, vc) = block.forward(
            x,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            Expression::from(PREV_SEQ),
        );
        let logits = embed.reverse(x).output();
        let kc = kc.output();
        let vc = vc.output();
        (
            BlockFixture { cx, block, embed, ids, k_cache, v_cache, gather_idx, scatter_idx },
            logits,
            kc,
            vc,
        )
    }

    /// Everything the block binds, with deterministic weights; returns
    /// (tensor-keyed data map, scalar-side copies).
    #[allow(clippy::type_complexity)]
    fn block_data(fx: &BlockFixture) -> (FxHashMap<petgraph::graph::NodeIndex, Vec<f32>>, Vec<(petgraph::graph::NodeIndex, Vec<f32>)>) {
        let token = 3usize;
        let mut pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
            (fx.ids.id, vec![token as f32]),
            (fx.embed.weight.id, weights(VOCAB * D, 1)),
            (fx.block.wq.weight.id, weights(D * D, 2)),
            (fx.block.wk.weight.id, weights(D * D, 3)),
            (fx.block.wv.weight.id, weights(D * D, 4)),
            (fx.block.wo.weight.id, weights(D * D, 5)),
            (fx.k_cache.id, weights(SLOTS * D, 6)),
            (fx.v_cache.id, weights(SLOTS * D, 8)),
            (fx.gather_idx.id, vec![0.0, 1.0]),
            (fx.scatter_idx.id, vec![1.0]),
        ];
        match &fx.block.ff {
            FeedForward::Dense { up, down } => {
                pairs.push((up.weight.id, weights(D * FF_HIDDEN, 9)));
                pairs.push((down.weight.id, weights(FF_HIDDEN * D, 10)));
            }
            FeedForward::Moe(moe) => {
                pairs.push((moe.router.id, weights(D * EXPERTS, 9)));
                pairs.push((moe.expert_weights.id, weights(EXPERTS * D * D, 10)));
            }
        }
        (pairs.iter().cloned().collect(), pairs)
    }

    /// Scalar reference for the whole block graph: embed → block → tied
    /// logits, plus the updated caches.
    fn block_reference(fx: &BlockFixture) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let embed_w = weights(VOCAB * D, 1);
        let token = 3usize;
        let x: Vec<f32> = embed_w[token * D..(token + 1) * D].to_vec();
        let mut k_cache = weights(SLOTS * D, 6);
        let mut v_cache = weights(SLOTS * D, 8);
        let (wq, wk, wv, wo) =
            (weights(D * D, 2), weights(D * D, 3), weights(D * D, 4), weights(D * D, 5));
        let ff: Box<dyn Fn(&[f32]) -> Vec<f32>> = match &fx.block.ff {
            FeedForward::Dense { .. } => {
                let (up, down) = (weights(D * FF_HIDDEN, 9), weights(FF_HIDDEN * D, 10));
                Box::new(move |x: &[f32]| {
                    let hidden: Vec<f32> = ref_matmul(x, &up, D, FF_HIDDEN)
                        .iter()
                        .map(|v| v.max(0.0))
                        .collect();
                    ref_matmul(&hidden, &down, FF_HIDDEN, D)
                })
            }
            FeedForward::Moe(_) => {
                let (router, experts) = (weights(D * EXPERTS, 9), weights(EXPERTS * D * D, 10));
                Box::new(move |x: &[f32]| ref_moe_k1(x, &router, &experts, D, EXPERTS))
            }
        };
        let x2 = ref_block_step(
            &x, &wq, &wk, &wv, &wo, &*ff, &mut k_cache, &mut v_cache, &[0, 1], 1, N_HEADS,
            HEAD_DIM, D,
        );
        // Tied logits: x2 · Eᵀ.
        let logits: Vec<f32> = (0..VOCAB)
            .map(|v| (0..D).map(|i| x2[i] * embed_w[v * D + i]).sum())
            .collect();
        (logits, k_cache, v_cache)
    }

    /// MODEL 2: the full decoder block (dense FFN) through the DEFAULT
    /// search ladder, with the stage attribution printed.
    #[test]
    fn decoder_block_matches_scalar_reference() {
        let (fx, logits, kc, vc) = block_fixture(|cx| FeedForward::Dense {
            up: Linear::new(D, FF_HIDDEN, false, cx),
            down: Linear::new(FF_HIDDEN, D, false, cx),
        });
        let (data, pairs) = block_data(&fx);
        let (ref_logits, ref_kc, ref_vc) = block_reference(&fx);

        let mut rt = SsaReferenceRuntime::load(&fx.cx).expect("native load");
        let outcome = rt
            .search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        eprintln!(
            "[decoder-block] search attribution: {} (plans profiled {}, cache hits {})",
            outcome.timings.summary(),
            outcome.plans_profiled,
            outcome.fingerprint_hits
        );
        for (id, values) in pairs {
            rt.set_data(id, values);
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
        assert_close(rt.get_f32(kc.id).expect("k cache"), &ref_kc);
        assert_close(rt.get_f32(vc.id).expect("v cache"), &ref_vc);
    }

    /// MODEL 3: the same block with the MoE FFN (harness search budget —
    /// the flagship default-budget run is model 2).
    #[test]
    fn moe_decoder_block_matches_scalar_reference() {
        let (fx, logits, kc, vc) = block_fixture(|cx| {
            FeedForward::Moe(MoE {
                expert_weights: cx.named_tensor("Experts", (EXPERTS, D, D)),
                router: cx.named_tensor("Router", (D, EXPERTS)),
                k: 1,
            })
        });
        let (_, pairs) = block_data(&fx);
        let (ref_logits, ref_kc, ref_vc) = block_reference(&fx);

        let rt = luminal::test_support::run_ssa(&fx.cx, &pairs);
        assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
        assert_close(rt.get_f32(kc.id).expect("k cache"), &ref_kc);
        assert_close(rt.get_f32(vc.id).expect("v cache"), &ref_vc);
    }

    /// MODEL 4: a TWO-LAYER decoder with pre-LayerNorms, decoded for TWO
    /// steps — the step-1 cache OUTPUTS feed the step-2 cache INPUTS
    /// through the binding surface (each step is its own shape-specialized
    /// program; harness search budget per step).
    #[test]
    fn tiny_decoder_two_steps_match_scalar_reference() {
        const LAYERS: usize = 2;
        const EPS: f32 = 1e-5;
        let tokens = [3usize, 1];

        // Scalar caches persist across BOTH steps, per layer.
        let mut ref_k: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];
        let mut ref_v: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];
        let mut runtime_k: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];
        let mut runtime_v: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];

        for (step, token) in tokens.iter().enumerate() {
            let prev_seq = step; // tokens already in the cache
            let ctx = step + 1; // slots visible this step
            let mut cx = Graph::new();
            let embed = Embedding::new(VOCAB, D, &mut cx);
            let mut norms = Vec::new();
            let mut blocks = Vec::new();
            for _ in 0..LAYERS {
                norms.push(crate::LayerNorm::new(D, None, None, true, EPS, &mut cx));
                blocks.push(DecoderBlock {
                    wq: Linear::new(D, D, false, &mut cx),
                    wk: Linear::new(D, D, false, &mut cx),
                    wv: Linear::new(D, D, false, &mut cx),
                    wo: Linear::new(D, D, false, &mut cx),
                    ff: FeedForward::Dense {
                        up: Linear::new(D, FF_HIDDEN, false, &mut cx),
                        down: Linear::new(FF_HIDDEN, D, false, &mut cx),
                    },
                    n_heads: N_HEADS,
                    n_kv_heads: N_HEADS,
                    head_dim: HEAD_DIM,
                });
            }
            let model = TinyDecoder {
                embed,
                norms,
                blocks,
                final_norm: crate::LayerNorm::new(D, None, None, true, EPS, &mut cx),
            };
            let ids = cx.tensor_dtyped(1, DType::Int);
            let cache_inputs: Vec<(GraphTensor, GraphTensor)> = (0..LAYERS)
                .map(|_| (cx.tensor((SLOTS, D)), cx.tensor((SLOTS, D))))
                .collect();
            let gather_idx = cx.tensor_dtyped(ctx, DType::Int);
            let scatter_idx = cx.tensor_dtyped(1, DType::Int);
            let (logits, caches_out) = model.forward(
                ids,
                &cache_inputs,
                gather_idx,
                scatter_idx,
                Expression::from(prev_seq),
            );
            let logits = logits.output();
            let caches_out: Vec<_> = caches_out
                .into_iter()
                .map(|(k, v)| (k.output(), v.output()))
                .collect();

            // Per-layer weights, deterministic and step-independent.
            let layer_weights = |layer: usize| {
                let base = 20 + layer * 10;
                (
                    weights(D * D, base),
                    weights(D * D, base + 1),
                    weights(D * D, base + 2),
                    weights(D * D, base + 3),
                    weights(D * FF_HIDDEN, base + 4),
                    weights(FF_HIDDEN * D, base + 5),
                )
            };
            let embed_w = weights(VOCAB * D, 1);
            let mut pairs: Vec<(petgraph::graph::NodeIndex, Vec<f32>)> = vec![
                (ids.id, vec![*token as f32]),
                (model.embed.weight.id, embed_w.clone()),
                (gather_idx.id, (0..ctx).map(|s| s as f32).collect()),
                (scatter_idx.id, vec![step as f32]),
            ];
            for layer in 0..LAYERS {
                let (wq, wk, wv, wo, up, down) = layer_weights(layer);
                let block = &model.blocks[layer];
                pairs.push((block.wq.weight.id, wq));
                pairs.push((block.wk.weight.id, wk));
                pairs.push((block.wv.weight.id, wv));
                pairs.push((block.wo.weight.id, wo));
                let FeedForward::Dense { up: up_l, down: down_l } = &block.ff else {
                    unreachable!()
                };
                pairs.push((up_l.weight.id, up));
                pairs.push((down_l.weight.id, down));
                pairs.push((cache_inputs[layer].0.id, runtime_k[layer].clone()));
                pairs.push((cache_inputs[layer].1.id, runtime_v[layer].clone()));
            }

            // Scalar reference for this step.
            let mut x: Vec<f32> = embed_w[token * D..(token + 1) * D].to_vec();
            let gather: Vec<usize> = (0..ctx).collect();
            for layer in 0..LAYERS {
                let (wq, wk, wv, wo, up, down) = layer_weights(layer);
                x = ref_layer_norm(&x, EPS);
                let ff = move |x: &[f32]| {
                    let hidden: Vec<f32> = ref_matmul(x, &up, D, FF_HIDDEN)
                        .iter()
                        .map(|v| v.max(0.0))
                        .collect();
                    ref_matmul(&hidden, &down, FF_HIDDEN, D)
                };
                x = ref_block_step(
                    &x,
                    &wq,
                    &wk,
                    &wv,
                    &wo,
                    &ff,
                    &mut ref_k[layer],
                    &mut ref_v[layer],
                    &gather,
                    step,
                    N_HEADS,
                    HEAD_DIM,
                    D,
                );
            }
            let x = ref_layer_norm(&x, EPS);
            let ref_logits: Vec<f32> = (0..VOCAB)
                .map(|v| (0..D).map(|i| x[i] * embed_w[v * D + i]).sum())
                .collect();

            // Two layers double the genome decision points: the harness
            // budget's 8 genomes all hit choice-cycle discards, so this
            // model runs the DEFAULT budget (64 genomes).
            let data: FxHashMap<_, _> = pairs.iter().cloned().collect();
            let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
            rt.search(&data, &ImplementationSearchOptions::default())
                .expect("search finds a plan");
            for (id, values) in &pairs {
                rt.set_data(*id, values.clone());
            }
            rt.execute().expect("winner executes");
            assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
            for layer in 0..LAYERS {
                let k_out = rt.get_f32(caches_out[layer].0.id).expect("k cache").clone();
                let v_out = rt.get_f32(caches_out[layer].1.id).expect("v cache").clone();
                assert_close(&k_out, &ref_k[layer]);
                assert_close(&v_out, &ref_v[layer]);
                runtime_k[layer] = k_out; // runtime-out → next step's runtime-in
                runtime_v[layer] = v_out;
            }
        }
    }
}
