//! MiniLlama3 fidelity: the mini vs a complete scalar reference
//! (trusted-validator doctrine; moved from luminal_nn mini.rs tests,
//! 2026-08-13).

use luminal::prelude::*;
use luminal::shape::IntExpr;
use scalar_refs::*;

use mini_llama3::MiniLlama3;

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
    let step = IntExpr::from(1usize);
    let (logits, caches_out, embed, blocks) = match family {
        "llama3" => {
            let model = MiniLlama3::new(VOCAB, D, FF, NH, NKV, LAYERS, &mut cx);
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
    let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![token as i32].into()),
        (embed.weight.id, embed_w.clone().into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    let qk_seeds_for = |layer: usize| (260 + layer * 2, 261 + layer * 2);
    let mut ref_caches: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for (layer, block) in blocks.iter().enumerate() {
        let (wq_s, wk_s, wv_s, wo_s, gate_s, up_s, down_s) = seeds_for(layer);
        pairs.push((block.wq.weight.id, weights(D * D, wq_s).into()));
        pairs.push((block.wk.weight.id, weights(D * KV_DIM, wk_s).into()));
        pairs.push((block.wv.weight.id, weights(D * KV_DIM, wv_s).into()));
        pairs.push((block.wo.weight.id, weights(D * D, wo_s).into()));
        pairs.push((block.gate.weight.id, weights(D * FF, gate_s).into()));
        pairs.push((block.up.weight.id, weights(D * FF, up_s).into()));
        pairs.push((block.down.weight.id, weights(FF * D, down_s).into()));
        if let Some((q_norm, k_norm)) = block.qk_norm {
            let (q_seed, k_seed) = qk_seeds_for(layer);
            pairs.push((q_norm.id, weights(HD, q_seed).into()));
            pairs.push((k_norm.id, weights(HD, k_seed).into()));
        }
        let kc = weights(SLOTS * KV_DIM, 300 + layer);
        let vc = weights(SLOTS * KV_DIM, 320 + layer);
        pairs.push((caches[layer].0.id, kc.clone().into()));
        pairs.push((caches[layer].1.id, vc.clone().into()));
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
    let mut rt = luminal_reference::ReferenceRuntime::load(&cx).expect("native load");
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
