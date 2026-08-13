//! Scalar-reference fidelity test for MiniWhisper (moved from
//! luminal_nn's mini.rs tests, relocation ruling 2026-08-13).

use luminal::prelude::*;
use scalar_refs::*;
use mini_whisper::MiniWhisper;

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
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (audio.id, audio_vals.clone().into()),
        (tokens.id, token_vals.clone().into()),
        (model.enc_wq.weight.id, weights(D * D, 502).into()),
        (model.enc_wk.weight.id, weights(D * D, 503).into()),
        (model.enc_wv.weight.id, weights(D * D, 504).into()),
        (model.enc_wo.weight.id, weights(D * D, 505).into()),
        (model.enc_up.weight.id, weights(D * FF, 506).into()),
        (model.enc_down.weight.id, weights(FF * D, 507).into()),
        (model.dec_wq.weight.id, weights(D * D, 508).into()),
        (model.dec_wk.weight.id, weights(D * D, 509).into()),
        (model.dec_wv.weight.id, weights(D * D, 510).into()),
        (model.dec_wo.weight.id, weights(D * D, 511).into()),
        (model.dec_up.weight.id, weights(D * FF, 512).into()),
        (model.dec_down.weight.id, weights(FF * D, 513).into()),
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
}
