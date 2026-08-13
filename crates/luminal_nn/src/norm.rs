use luminal::prelude::*;

/// A simple layer norm with an optional weight and bias
#[derive(Default)]
pub struct LayerNorm {
    pub weight: Option<GraphTensor>,
    pub bias: Option<GraphTensor>,
    mean_norm: bool,
    epsilon: f32,
    /// Gemma's (1+w) parameterization: the checkpoint weight enters
    /// AS-IS and the graph applies `1 + w` (the parameterization is
    /// model definition — ruling 2026-08-12; the hf.rs pre-bake died).
    unit_offset: bool,
}

impl LayerNorm {
    pub fn new(
        dim: usize,
        weight: Option<&str>,
        bias: Option<&str>,
        mean_norm: bool,
        epsilon: f32,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: weight.map(|w| cx.named_tensor(w, dim)),
            bias: bias.map(|b| cx.named_tensor(b, dim)),
            mean_norm,
            epsilon,
            unit_offset: false,
        }
    }

    /// The Gemma (1+w) form: multiply by `1 + weight` in-graph.
    pub fn with_unit_offset(mut self) -> Self {
        self.unit_offset = true;
        self
    }
}

impl LayerNorm {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        if self.mean_norm {
            input = input.mean_norm(input.rank() - 1);
        }
        input = input.std_norm(input.rank() - 1, self.epsilon);
        if let Some(w) = self.weight {
            let w = if self.unit_offset { w + 1.0 } else { w };
            input *= w.expand_lhs(&input.dims()[..input.dims().len() - 1]);
        }
        if let Some(b) = self.bias {
            input += b.expand_lhs(&input.dims()[..input.dims().len() - 1]);
        }
        input
    }
}
