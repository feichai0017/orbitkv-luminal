//! MiniWhisper (encoder + cross-attention decoder) demo on the reference
//! runtime. Run: cargo run --release --example mini_whisper

use luminal::prelude::*;
use luminal_nn::MiniWhisper;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6).collect()
}

fn main() {
    const D: usize = 4;
    const FF: usize = 6;
    let mut cx = Graph::new();
    let model = MiniWhisper::new(D, FF, 2, &mut cx);
    let audio = cx.tensor((2, D));
    let tokens = cx.tensor((1, D));
    let out = model.forward(audio, tokens).output();
    let pairs = vec![
        (audio.id, weights(2 * D, 1)),
        (tokens.id, weights(D, 2)),
        (model.enc_wq.weight.id, weights(D * D, 3)),
        (model.enc_wk.weight.id, weights(D * D, 4)),
        (model.enc_wv.weight.id, weights(D * D, 5)),
        (model.enc_wo.weight.id, weights(D * D, 6)),
        (model.enc_up.weight.id, weights(D * FF, 7)),
        (model.enc_down.weight.id, weights(FF * D, 8)),
        (model.dec_wq.weight.id, weights(D * D, 9)),
        (model.dec_wk.weight.id, weights(D * D, 10)),
        (model.dec_wv.weight.id, weights(D * D, 11)),
        (model.dec_wo.weight.id, weights(D * D, 12)),
        (model.dec_up.weight.id, weights(D * FF, 13)),
        (model.dec_down.weight.id, weights(FF * D, 14)),
    ];
    let rt = luminal::test_support::run_ssa(&cx, &pairs);
    println!("decoder out: {:?}", rt.get_f32(out.id).unwrap());
}
