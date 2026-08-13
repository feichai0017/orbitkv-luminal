//! MiniLlama3 — one small, runnable representative of the
//! llama/paged_llama example-model family (mini relocation ruling
//! 2026-08-13: model definitions live in their example crates;
//! luminal_nn keeps only the building blocks).

use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal_nn::{Embedding, GatedFfn, LayerNorm, LlamaBlock};

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
) -> (Embedding, Vec<LlamaBlock>, LayerNorm) {
    let model = Ns::root().child("model");
    let blocks = (0..layers)
        .map(|l| {
            let layer_ns = model.child("layers").index(l);
            let block = LlamaBlock::new_with_ffn(d, ff, n_heads, n_kv_heads, ffn, &layer_ns, cx);
            if qk_norm { block.with_qk_norm(&layer_ns, cx) } else { block }
        })
        .collect();
    (
        Embedding::new(vocab, d, &model.child("embed_tokens"), cx),
        blocks,
        LayerNorm::new(d, false, false, false, 1e-5, &model.child("norm"), cx),
    )
}

fn gqa_lm_forward(
    embed: &Embedding,
    blocks: &[LlamaBlock],
    final_norm: &LayerNorm,
    ids: GraphTensor,
    caches: &[(GraphTensor, GraphTensor)],
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: IntExpr,
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

/// MiniLlama3 — the llama/paged_llama family: RMS pre-norms, GQA over a
/// paged KV cache, SwiGLU. (RoPE deferred by ruling, as everywhere.)
pub struct MiniLlama3 {
    pub embed: Embedding,
    pub blocks: Vec<LlamaBlock>,
    pub final_norm: LayerNorm,
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

    /// ids (s,) Int + one (k, v) cache pair per layer → (logits, caches').
    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
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
}
