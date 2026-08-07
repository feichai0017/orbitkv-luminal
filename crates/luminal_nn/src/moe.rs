use luminal::prelude::*;

/// A layer of E experts and a router
pub struct MoE {
    pub expert_weights: GraphTensor, // [E, in, out]
    pub router: GraphTensor,         // [in, E]
    pub k: usize,
}

impl MoE {
    pub fn forward(&self, activations: GraphTensor) -> GraphTensor {
        let n = activations.dims().len();
        let e_dim = *self.router.dims().last().unwrap();
        let (_, in_size, out_size) = self.expert_weights.dims3();
        let io = in_size * out_size;
        let k_expr = Expression::from(self.k);

        // 1. Routing probabilities: [batch.., E]
        let routing_weights = activations.matmul(self.router).softmax(n - 1);

        // 2. Top-k expert indices: [batch.., k] (Int)
        let top_k_indices = routing_weights.topk_indexes(self.k, n - 1);

        // 3. Gather top-k routing values: [batch.., k]
        //    flat_idx = batch_row * E + expert_idx
        //    iota(z / k * E) gives batch_row * E at each position in [batch.., k]
        let row_offsets = activations
            .graph()
            .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
        let routing_flat_idx =
            (row_offsets.cast(DType::F32) + top_k_indices.cast(DType::F32)).cast(DType::Int);
        let top_k_values = routing_weights.gather1d(routing_flat_idx); // [batch.., k]

        // 4. Gather expert weight matrices: [batch.., k, in, out]
        //    flat_idx[.., ki, i, o] = expert_idx[.., ki] * in*out + i * out + o
        let base = (top_k_indices * io).cast(DType::F32); // [batch.., k]
        let within = activations
            .graph()
            .iota(Expression::from('z'), (in_size, out_size))
            .cast(DType::F32); // [in, out] values 0..in*out-1

        // Expand base to [batch.., k, in, out]
        let n_base = base.dims().len();
        let exp_base = base
            .expand_dim(n_base, in_size)
            .expand_dim(n_base + 1, out_size);

        // Expand within to [batch.., k, in, out]
        let mut exp_within = within;
        for (i, dim) in base.dims().iter().enumerate() {
            exp_within = exp_within.expand_dim(i, *dim);
        }

        let expert_flat_idx = (exp_base + exp_within).cast(DType::Int);
        let gathered = self.expert_weights.gather1d(expert_flat_idx); // [batch.., k, in, out]

        // 5. Batched matmul: [batch.., k, 1, in] @ [batch.., k, in, out] → [batch.., k, out]
        let expanded_act = activations
            .expand_dim(n - 1, self.k) // [batch.., k, in]
            .unsqueeze(n); // [batch.., k, 1, in]
        let expert_out = expanded_act.matmul(gathered).squeeze(n); // [batch.., k, out]

        // 6. Weighted sum over experts: [batch.., k, out] * [batch.., k, 1] → sum(k) → [batch.., out]
        let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len()); // [batch.., k, 1]
        let weights_exp = weights_exp.expand(expert_out.dims());
        (expert_out * weights_exp).sum(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::MoE;
    use luminal::prelude::*;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    /// Full MoE forward (router softmax → topk → routing-value and
    /// expert-weight gathers → batched matmul → weighted sum) against a
    /// scalar reference, k=1 over 2 experts. Runs the plain extraction
    /// path — the chain rides stable_argsort's rank scatter, both flat
    /// gather sugars, and Int↔F32 index arithmetic.
    #[test]
    fn moe_forward_matches_scalar_reference() {
        const E: usize = 2;
        const IN: usize = 2;
        const OUT: usize = 2;
        const BATCH: usize = 3;

        let mut cx = Graph::new();
        let model = MoE {
            expert_weights: cx.named_tensor("Experts", (E, IN, OUT)),
            router: cx.named_tensor("Router", (IN, E)),
            k: 1,
        };
        let x = cx.tensor((BATCH, IN));
        let out = model.forward(x).output();

        let x_vals = vec![1.0f32, 0.5, -1.0, 2.0, 0.25, -0.75];
        // Router picks expert 0 for positive-x0-heavy rows, expert 1 otherwise
        // (logits differ per row; no ties).
        let router_vals = vec![2.0f32, -1.0, -0.5, 1.5];
        let expert_vals: Vec<f32> = (0..E * IN * OUT).map(|v| v as f32 * 0.3 - 1.0).collect();

        // Scalar reference.
        let mut expected = vec![0.0f32; BATCH * OUT];
        for b in 0..BATCH {
            let xr = &x_vals[b * IN..(b + 1) * IN];
            let mut logits = [0.0f32; E];
            for (e, logit) in logits.iter_mut().enumerate() {
                *logit = (0..IN).map(|i| xr[i] * router_vals[i * E + e]).sum();
            }
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let best = (0..E).max_by(|a, b| logits[*a].partial_cmp(&logits[*b]).unwrap()).unwrap();
            let weight = exps[best] / denom;
            let w = &expert_vals[best * IN * OUT..(best + 1) * IN * OUT];
            for o in 0..OUT {
                expected[b * OUT + o] =
                    (0..IN).map(|i| xr[i] * w[i * OUT + o]).sum::<f32>() * weight;
            }
        }

        let rt = luminal::test_support::run_ssa(
            &cx,
            &[
                (x.id, x_vals),
                (model.router.id, router_vals),
                (model.expert_weights.id, expert_vals),
            ],
        );
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }
}
