//! MiniConvNet (yolo-family) demo on the reference runtime.
//! Run: cargo run --release -p mini_conv

use luminal::prelude::*;
use mini_conv::MiniConvNet;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

fn main() {
    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let x = cx.tensor((1, 1, 5, 5));
    let out = model.forward(x).output();
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (x.id, weights(25, 1).into()),
        (model.conv1.weight.id, weights(18, 2).into()),
        (model.conv2.weight.id, weights(54, 3).into()),
        (model.head.weight.id, weights(6, 4).into()),
    ];
    let rt = luminal_reference::harness::run_reference(&cx, &pairs);
    println!("logits: {:?}", rt.get_f32(out.id).unwrap());
}
