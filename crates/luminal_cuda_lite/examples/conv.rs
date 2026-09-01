//! MiniConvNet (yolo-family) on CUDA-lite: reference run vs device run
//! on identical seeded inputs, compared through the disclosed layout.
//! Canonical dims from `examples/mini/conv/src/bin/measure_plan.rs`.
//!
//! Run: cargo run -p luminal_cuda_lite --example conv --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("conv");
}

#[cfg(feature = "device")]
fn main() {
    use luminal::prelude::*;
    use mini_conv::MiniConvNet;
    use support::weights;

    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let x = cx.tensor((1, 1, 5, 5));
    let out = model.forward(x).output();
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (x.id, weights(25, 1).into()),
        (model.conv1.weight.id, weights(18, 2).into()),
        (model.conv2.weight.id, weights(54, 3).into()),
        (model.head.weight.id, weights(6, 4).into()),
    ];

    if let Err(e) =
        support::device::run_differential("conv", &cx, &pairs, &[("logits", out.id)])
    {
        eprintln!("conv: FAIL: {e:#}");
        std::process::exit(1);
    }
}
