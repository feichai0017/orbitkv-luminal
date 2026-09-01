//! MiniGemma3 fidelity vs a complete scalar reference, plus the
//! construct-isolation probes (moved from luminal_nn's mini.rs, 2026-08-13).

use luminal::prelude::*;
use luminal::shape::IntExpr;
use mini_gemma3::MiniGemma3;
use scalar_refs::*;

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
        IntExpr::from(q_pos),
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
    let rot_matrix = luminal_nn::rope_pairing_matrix(HD, false);
    // Per-layer tables from each block's role parameters.
    let role_tables: Vec<(Vec<f32>, Vec<f32>)> = model
        .blocks
        .iter()
        .map(|block| {
            luminal_nn::rope_tables_split_half(
                &[q_pos as f32],
                HD,
                block.rope_theta,
                block.pos_scale,
            )
        })
        .collect();
    let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![token as i32].into()),
        (model.embed.weight.id, embed_w.clone().into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
        (rope_rot.id, rot_matrix.clone().into()),
        (
            model.final_norm.weight.expect("weighted").id,
            weights(D, 660).into(),
        ),
    ];
    for (layer, (cos_table, sin_table)) in role_tables.iter().enumerate() {
        pairs.push((rope_inputs[layer].0.id, cos_table.clone().into()));
        pairs.push((rope_inputs[layer].1.id, sin_table.clone().into()));
    }
    let mut ref_caches: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for (layer, block) in model.blocks.iter().enumerate() {
        pairs.push((
            block.wq.weight.id,
            weights(D * Q_DIM, seeds(layer, 0)).into(),
        ));
        pairs.push((
            block.wk.weight.id,
            weights(D * KV_DIM, seeds(layer, 1)).into(),
        ));
        pairs.push((
            block.wv.weight.id,
            weights(D * KV_DIM, seeds(layer, 2)).into(),
        ));
        pairs.push((
            block.wo.weight.id,
            weights(Q_DIM * D, seeds(layer, 3)).into(),
        ));
        pairs.push((
            block.gate.weight.id,
            weights(D * FF, seeds(layer, 4)).into(),
        ));
        pairs.push((block.up.weight.id, weights(D * FF, seeds(layer, 5)).into()));
        pairs.push((
            block.down.weight.id,
            weights(FF * D, seeds(layer, 6)).into(),
        ));
        pairs.push((
            block.input_norm.weight.expect("weighted").id,
            weights(D, seeds(layer, 7)).into(),
        ));
        pairs.push((
            block.post_attn_norm.weight.expect("weighted").id,
            weights(D, seeds(layer, 8)).into(),
        ));
        pairs.push((
            block.pre_ff_norm.weight.expect("weighted").id,
            weights(D, seeds(layer, 9)).into(),
        ));
        pairs.push((
            block.post_ff_norm.weight.expect("weighted").id,
            weights(D, seeds(layer, 10)).into(),
        ));
        pairs.push((block.q_norm.id, weights(HD, seeds(layer, 11)).into()));
        pairs.push((block.k_norm.id, weights(HD, seeds(layer, 12)).into()));
        let kc = weights(SLOTS * KV_DIM, 300 + layer);
        let vc = weights(SLOTS * KV_DIM, 320 + layer);
        pairs.push((caches[layer].0.id, kc.clone().into()));
        pairs.push((caches[layer].1.id, vc.clone().into()));
        ref_caches.push((kc, vc));
    }

    // ---- scalar reference ----
    let wrms = |x: &[f32], w: &[f32]| -> Vec<f32> {
        // Gemma (1+w): the reference mirrors the in-graph unit offset.
        ref_rms_norm(x, 1e-6)
            .iter()
            .zip(w)
            .map(|(v, w)| v * (1.0 + w))
            .collect()
    };
    let mul = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x * y).collect() };
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
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
        let qw1: Vec<f32> = weights(HD, seeds(layer, 11))
            .iter()
            .map(|w| 1.0 + w)
            .collect();
        let q = ref_rms_head_norm(&q, HD, &qw1);
        let q: Vec<f32> = q.iter().map(|v| v * scale).collect(); // folded into Q
        let q = ref_rotary_apply(&q, HD, cos_table, sin_table, &rot_matrix);
        let k = ref_matmul(&h, &weights(D * KV_DIM, seeds(layer, 1)), D, KV_DIM);
        let kw1: Vec<f32> = weights(HD, seeds(layer, 12))
            .iter()
            .map(|w| 1.0 + w)
            .collect();
        let k = ref_rms_head_norm(&k, HD, &kw1);
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
        let gate = ref_gelu_tanh(&ref_matmul(
            &ff_in,
            &weights(D * FF, seeds(layer, 4)),
            D,
            FF,
        ));
        let up = ref_matmul(&ff_in, &weights(D * FF, seeds(layer, 5)), D, FF);
        let ff = ref_matmul(&mul(&gate, &up), &weights(FF * D, seeds(layer, 6)), FF, D);
        x = add(&x, &wrms(&ff, &weights(D, seeds(layer, 10))));
    }
    let x = wrms(&x, &weights(D, 660));
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
        assert_close(
            rt.get_f32(caches_out[layer].0.id).unwrap(),
            &ref_caches[layer].0,
        );
        assert_close(
            rt.get_f32(caches_out[layer].1.id).unwrap(),
            &ref_caches[layer].1,
        );
    }
}

/// CONSTRUCT-ISOLATION PROBES for the gemma3 memory explosion
/// (RSS-KILL 5.8GB in 12s, isolated sweep 2026-08-10): each
/// sub-graph exercises exactly ONE of the constructs the full-
/// anatomy gemma added — (a) in-graph split-half rope, (b) sliding-
/// window paged attention, (c) weighted sandwich norms. 1-genome
/// budget; run in a capped process — the stage whose print never
/// appears is the bomb. Run:
/// cargo test --release -p mini_gemma3 probe_gemma_constructs -- --ignored --nocapture
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
    let run = |label: &str, cx: &Graph, pairs: &[(petgraph::graph::NodeIndex, TypedBuffer)]| {
        let start = std::time::Instant::now();
        let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
        let mut rt = luminal_reference::ReferenceRuntime::load(cx).expect("native load");
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
        let out = luminal_nn::rotary_apply(x, 4, cos, sin, rot).output();
        let _ = out;
        let (cos_table, sin_table) = luminal_nn::rope_tables_split_half(&[1.0], 4, 10_000.0, 1.0);
        let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
            (x.id, weights(8, 1).into()),
            (cos.id, cos_table.into()),
            (sin.id, sin_table.into()),
            (rot.id, luminal_nn::rope_pairing_matrix(4, false).into()),
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
        let (attn, kc, vc) = luminal_nn::paged_attention_windowed(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            IntExpr::from(1usize),
            1,
            1,
            4,
            Some(1),
            0.5,
        );
        let _ = (attn.output(), kc.output(), vc.output());
        let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
            (q.id, weights(4, 2).into()),
            (k_new.id, weights(4, 3).into()),
            (v_new.id, weights(4, 4).into()),
            (k_cache.id, weights(16, 5).into()),
            (v_cache.id, weights(16, 6).into()),
            (gather_idx.id, vec![0i32, 1].into()),
            (scatter_idx.id, vec![1i32].into()),
        ];
        run("window-alone", &cx, &pairs);
    }

    // (c) weighted sandwich norms alone: x + post(w·(x·W)) shape.
    {
        let mut cx = Graph::new();
        let x = cx.tensor((1, 6));
        let w = cx.tensor((6, 6));
        let pre = luminal_nn::LayerNorm::new(
            6,
            true,
            false,
            false,
            1e-6,
            &Ns::root().child("pre"),
            &mut cx,
        );
        let post = luminal_nn::LayerNorm::new(
            6,
            true,
            false,
            false,
            1e-6,
            &Ns::root().child("post"),
            &mut cx,
        );
        let out = (x + post.forward(pre.forward(x).matmul(w))).output();
        let _ = out;
        let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
            (x.id, weights(6, 7).into()),
            (w.id, weights(36, 8).into()),
            (pre.weight.expect("w").id, weights(6, 9).into()),
            (post.weight.expect("w").id, weights(6, 10).into()),
        ];
        run("sandwich-alone", &cx, &pairs);
    }
}
