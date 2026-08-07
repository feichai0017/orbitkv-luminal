//! FULL-MODEL tests, smallest first (Austin's directive 2026-08-06):
//! whole models composed from the nn modules, run end-to-end through the
//! native ladder (load → search → execute) against scalar references
//! computed in plain loops. Model 1 is a two-hidden-layer MLP; the next
//! rungs are a single decoder block with a KV cache, then a tiny
//! multi-layer decoder.

use crate::Linear;
use luminal::prelude::*;

/// The smallest true model: Linear → relu → Linear → relu → Linear.
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Mlp {
    /// `dims` = [in, hidden.., out]; a relu follows every layer but the
    /// last.
    pub fn new(dims: &[usize], cx: &mut Graph) -> Self {
        assert!(dims.len() >= 2, "an MLP needs at least in and out dims");
        let layers = dims
            .windows(2)
            .map(|pair| Linear::new(pair[0], pair[1], true, cx))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, mut x: GraphTensor) -> GraphTensor {
        let last = self.layers.len() - 1;
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if index != last {
                x = x.relu();
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::Mlp;
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

    /// Deterministic pseudo-random weights (no RNG dependency; values in
    /// roughly [-0.6, 0.6] so activations stay in a well-conditioned
    /// range).
    fn weights(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
            .collect()
    }

    /// MODEL 1: a full 4→8→6→3 MLP, batch 2, through the search ladder —
    /// every layer's weights and biases bound as named tensors, the whole
    /// forward against a scalar reference.
    #[test]
    fn mlp_forward_matches_scalar_reference() {
        const DIMS: [usize; 4] = [4, 8, 6, 3];
        const BATCH: usize = 2;

        let mut cx = Graph::new();
        let model = Mlp::new(&DIMS, &mut cx);
        let x = cx.tensor((BATCH, DIMS[0]));
        let out = model.forward(x).output();

        let x_data = weights(BATCH * DIMS[0], 7);
        let mut layer_data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for (index, pair) in DIMS.windows(2).enumerate() {
            layer_data.push((
                weights(pair[0] * pair[1], index),
                weights(pair[1], index + 50),
            ));
        }

        // Scalar reference.
        let mut activation = x_data.clone();
        let mut width = DIMS[0];
        for (index, pair) in DIMS.windows(2).enumerate() {
            let (w, b) = &layer_data[index];
            let (in_w, out_w) = (pair[0], pair[1]);
            let mut next = vec![0f32; BATCH * out_w];
            for r in 0..BATCH {
                for c in 0..out_w {
                    let mut acc = b[c];
                    for k in 0..in_w {
                        acc += activation[r * width + k] * w[k * out_w + c];
                    }
                    next[r * out_w + c] =
                        if index != DIMS.len() - 2 { acc.max(0.0) } else { acc };
                }
            }
            activation = next;
            width = out_w;
        }

        let mut data = FxHashMap::default();
        data.insert(x.id, x_data.clone());
        for (layer, (w, b)) in model.layers.iter().zip(&layer_data) {
            data.insert(layer.weight.id, w.clone());
            data.insert(layer.bias.unwrap().id, b.clone());
        }
        let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(x.id, x_data);
        for (layer, (w, b)) in model.layers.iter().zip(&layer_data) {
            rt.set_data(layer.weight.id, w.clone());
            rt.set_data(layer.bias.unwrap().id, b.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &activation);
    }
}
