//! MiniGemma3 — the gemma family's mini model (rulings 2026-08-07/10,
//! relocation 2026-08-13): the model DEFINITION lives here in the
//! example crate; luminal_nn keeps only the building blocks
//! ([`GemmaBlock`], [`Embedding`], [`LayerNorm`], rope helpers).

use luminal::prelude::*;
use luminal::shape::Expression;
use luminal_nn::{Embedding, GemmaBlock, LayerNorm};

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
    pub blocks: Vec<GemmaBlock>,
    pub final_norm: LayerNorm,
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
            embed: Embedding::new(vocab, d, &Ns::root().child("model").child("embed_tokens"), cx),
            blocks: (0..layers)
                .map(|layer| {
                    let local = (layer + 1) % pattern != 0;
                    let layer_ns = Ns::root().child("model").child("layers").index(layer);
                    GemmaBlock::new(
                        d, ff, n_heads, n_kv_heads, head_dim, local, window, &layer_ns, cx,
                    )
                })
                .collect(),
            final_norm: LayerNorm::new(
                d, true, false, false, 1e-6, &Ns::root().child("model").child("norm"), cx,
            )
            .with_unit_offset(),
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
