//! MiniWhisper (encoder + cross-attention decoder) demo on the reference
//! runtime. Run: cargo run --release -p mini_whisper

use luminal::prelude::*;
use mini_whisper::MiniWhisper;

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
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (audio.id, weights(2 * D, 1).into()),
        (tokens.id, weights(D, 2).into()),
        (model.enc_wq.weight.id, weights(D * D, 3).into()),
        (model.enc_wk.weight.id, weights(D * D, 4).into()),
        (model.enc_wv.weight.id, weights(D * D, 5).into()),
        (model.enc_wo.weight.id, weights(D * D, 6).into()),
        (model.enc_up.weight.id, weights(D * FF, 7).into()),
        (model.enc_down.weight.id, weights(FF * D, 8).into()),
        (model.dec_wq.weight.id, weights(D * D, 9).into()),
        (model.dec_wk.weight.id, weights(D * D, 10).into()),
        (model.dec_wv.weight.id, weights(D * D, 11).into()),
        (model.dec_wo.weight.id, weights(D * D, 12).into()),
        (model.dec_up.weight.id, weights(D * FF, 13).into()),
        (model.dec_down.weight.id, weights(FF * D, 14).into()),
    ];
    let rt = luminal::test_support::run_ssa(&cx, &pairs);
    println!("decoder out: {:?}", rt.get_f32(out.id).unwrap());
}
