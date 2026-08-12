use luminal::prelude::*;

/// A simple linear layer
pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
    permute: bool,
}

impl Linear {
    pub fn new(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (inp, out)),
            bias: if bias {
                Some(cx.named_tensor("Bias", out))
            } else {
                None
            },
            permute: false,
        }
    }

    pub fn new_permuted(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (out, inp)),
            bias: if bias {
                Some(cx.named_tensor("Bias", out))
            } else {
                None
            },
            permute: true,
        }
    }
}

impl Linear {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let output = input.matmul(if self.permute {
            self.weight.permute((1, 0))
        } else {
            self.weight
        });
        if let Some(bias) = self.bias {
            output + bias.expand_lhs(&output.dims()[..output.dims().len() - 1])
        } else {
            output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::ssa_reference::SsaReferenceRuntime;
    use rustc_hash::FxHashMap;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    /// The M3 ladder end-to-end (load → search → execute → read): the
    /// first nn-module test on the native path. Hand-computed reference.
    #[test]
    fn linear_forward_matches_hand_reference() {
        let mut cx = Graph::new();
        let model = Linear::new(3, 4, false, &mut cx);
        let x = cx.tensor((2, 3));
        let out = model.forward(x).output();

        let x_data = vec![1., 2., 3., 4., 5., 6.];
        let w_data: Vec<f32> = (1..=12).map(|v| v as f32 * 0.1).collect();
        let mut expected = vec![0f32; 8];
        for r in 0..2 {
            for c in 0..4 {
                expected[r * 4 + c] =
                    (0..3).map(|k| x_data[r * 3 + k] * w_data[k * 4 + c]).sum();
            }
        }

        let mut data = FxHashMap::default();
        data.insert(x.id, x_data.clone().into());
        data.insert(model.weight.id, w_data.clone().into());
        let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(x.id, x_data);
        rt.set_data(model.weight.id, w_data);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Bias broadcasts over the batch dimension.
    #[test]
    fn linear_bias_broadcasts_over_the_batch() {
        let mut cx = Graph::new();
        let model = Linear::new(2, 3, true, &mut cx);
        let x = cx.tensor((2, 2));
        let out = model.forward(x).output();

        let x_data = vec![1., 2., 3., 4.];
        let w_data = vec![1., 0., 2., 0., 1., 3.];
        let b_data = vec![0.5, -1.0, 0.25];
        let mut expected = vec![0f32; 6];
        for r in 0..2 {
            for c in 0..3 {
                expected[r * 3 + c] = (0..2)
                    .map(|k| x_data[r * 2 + k] * w_data[k * 3 + c])
                    .sum::<f32>()
                    + b_data[c];
            }
        }

        let mut data = FxHashMap::default();
        data.insert(x.id, x_data.clone().into());
        data.insert(model.weight.id, w_data.clone().into());
        data.insert(model.bias.unwrap().id, b_data.clone().into());
        let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(x.id, x_data);
        rt.set_data(model.weight.id, w_data);
        rt.set_data(model.bias.unwrap().id, b_data);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }
}
