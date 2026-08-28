//! MiniWhisper (encoder + cross-attention decoder) on CUDA-lite:
//! reference run vs device run on identical seeded inputs, compared
//! through the disclosed layout. Canonical dims from
//! `examples/mini/whisper/src/bin/measure_plan.rs`.
//!
//! Run: cargo run -p luminal_cuda_lite --example whisper --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("whisper");
}

#[cfg(feature = "device")]
fn main() {
    use luminal::prelude::*;
    use mini_whisper::MiniWhisper;
    use support::weights;

    const D: usize = 4;
    const FF: usize = 6;
    let mut cx = Graph::new();
    let model = MiniWhisper::new(D, FF, 2, &mut cx);
    let audio = cx.tensor((2, D));
    let tokens = cx.tensor((1, D));
    let out = model.forward(audio, tokens).output();
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
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

    if let Err(e) =
        support::device::run_differential("whisper", &cx, &pairs, &[("out", out.id)])
    {
        eprintln!("whisper: FAIL: {e:#}");
        std::process::exit(1);
    }
}
