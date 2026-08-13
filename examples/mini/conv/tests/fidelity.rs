//! MiniConvNet fidelity test against the scalar reference (moved from
//! luminal_nn::mini's test module, 2026-08-13 mini relocation).

use luminal::prelude::*;
use luminal_nn::test_refs::*;
use mini_conv::*;

/// MiniConvNet: 3×3 valid convs 5→3→1 + relu + linear head.
#[test]
fn mini_convnet_matches_scalar_reference() {
    const C1: usize = 2;
    const C2: usize = 3;
    const CLASSES: usize = 2;

    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, C1, C2, CLASSES, &mut cx);
    let x = cx.tensor((1, 1, 5, 5));
    let out = model.forward(x).output();

    let x_vals = weights(25, 600);
    let w1 = weights(C1 * 9, 601);
    let w2 = weights(C2 * C1 * 9, 602);
    let wh = weights(C2 * CLASSES, 603);
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (x.id, x_vals.clone().into()),
        (model.conv1.weight.id, w1.clone().into()),
        (model.conv2.weight.id, w2.clone().into()),
        (model.head.weight.id, wh.clone().into()),
    ];

    // Scalar reference: valid 3×3 convs; ConvND weight layout is
    // (ch_out, ch_in·kh·kw) with kernel-major within a channel.
    let conv = |input: &[f32], w: &[f32], ch_in: usize, ch_out: usize, h: usize| -> Vec<f32> {
        let oh = h - 2;
        let mut out = vec![0f32; ch_out * oh * oh];
        for co in 0..ch_out {
            for oy in 0..oh {
                for ox in 0..oh {
                    let mut acc = 0f32;
                    for ci in 0..ch_in {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                acc += input[ci * h * h + (oy + ky) * h + (ox + kx)]
                                    * w[co * ch_in * 9 + ci * 9 + ky * 3 + kx];
                            }
                        }
                    }
                    out[co * oh * oh + oy * oh + ox] = acc.max(0.0);
                }
            }
        }
        out
    };
    let f1 = conv(&x_vals, &w1, 1, C1, 5); // (C1, 3, 3), relu applied
    let f2 = conv(&f1, &w2, C1, C2, 3); // (C2, 1, 1), relu applied
    let expected = ref_matmul(&f2, &wh, C2, CLASSES);

    let rt = luminal::test_support::run_ssa(&cx, &pairs);
    assert_close(rt.get_f32(out.id).expect("logits"), &expected);
}
