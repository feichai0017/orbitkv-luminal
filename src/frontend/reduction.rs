use crate::prelude::*;

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum(self, axes: impl ToAxes) -> GraphTensor {
        self.reduce("LogicalReduceSum", axes)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max(self, axes: impl ToAxes) -> GraphTensor {
        self.reduce("LogicalReduceMax", axes)
    }

    /// One recorded reduce per axis; the operand for the FIRST reduce is
    /// self's value, later reduces consume the previous reduce's result.
    fn reduce(self, constructor: &str, axes: impl ToAxes) -> GraphTensor {
        let (mut dims, mut id) = (self.dims(), self.id);
        let mut axes = axes.to_axes();
        let mut operand_value = self.logical_value;
        for dim in 0..axes.len() {
            let operand_dims = dims.clone();
            id = self.graph().mint_id();
            if constructor == "LogicalReduceMax" {
                // The empty max has no value (extent-0 ruling
                // 2026-08-13): the reduced axis contracts to >= 1 —
                // static extents discharge trivially; symbolic ones
                // refuse unless the binding's range excludes 0.
                let extent = operand_dims[axes[dim]];
                let at = id.index();
                self.graph()
                    .logical
                    .require_extent_at_least(at, &extent, 1, "reduce_max axis");
            }
            let rank = operand_dims.len();
            let axis_from_end = rank - 1 - axes[dim];
            let mut out_dims = operand_dims.clone();
            out_dims.remove(axes[dim]);
            operand_value = self.graph().logical.op(
                id.index(),
                constructor,
                &[(operand_value, operand_dims)],
                &axis_from_end.to_string(),
                out_dims.clone(),
                self.dtype,
            );
            dims = out_dims;
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, dims, self.graph_ref, self.dtype).with_logical(operand_value)
    }

    /// Reduce a dimension of the tensor by taking the minimum of all elements along that axis.
    pub fn min(self, axes: impl ToAxes) -> GraphTensor {
        -(-self).max(axes)
    }

    /// Reduce a dimension of the tensor by taking the mean of all elements along that axis.
    pub fn mean(self, axes: impl ToAxes) -> GraphTensor {
        let reduced_elements = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<IntExpr>();
        self.sum(axes) / reduced_elements
    }

    /// Reduce a dimension of the tensor by multiplying all elements along that axis.
    pub fn prod(self, axes: impl ToAxes) -> GraphTensor {
        self.log().sum(axes).exp()
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::unary::tests::test_unary;
    use candle_core::{Device, Tensor};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_sum(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
            test_unary((rows, cols), |a| a.sum(1), |a| a.sum(1).unwrap());
            test_unary(
                (rows, cols, depth),
                |a| a.sum((0, 2)),
                |a| a.sum((0, 2)).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_max(rows in 1usize..8, cols in 1usize..8) {
            test_unary((rows, cols), |a| a.max(1), |a| a.max(1).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_min(rows in 1usize..8, cols in 1usize..8) {
            test_unary((rows, cols), |a| a.min(1), |a| a.min(1).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mean(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
            test_unary((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap());
            let denom = (rows * depth) as f32;
            test_unary(
                (rows, cols, depth),
                |a| a.mean((0, 2)),
                |a| {
                    let denom = Tensor::from_vec(vec![denom; cols], cols, a.device()).unwrap();
                    (a.sum(2).unwrap().sum(0).unwrap() / denom).unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_prod(rows in 1usize..8, cols in 1usize..8) {
            test_unary(
                (rows, cols),
                |a| a.prod(1),
                |a| {
                    let v = a.to_vec2::<f32>().unwrap();
                    let out: Vec<f32> = v.iter().map(|row| row.iter().product()).collect();
                    Tensor::from_vec(out, v.len(), &Device::Cpu).unwrap()
                },
            );
        }
    }
}
