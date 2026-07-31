use itertools::Itertools;

use crate::{
    hlir::{Gather, Scatter},
    prelude::*,
};

impl GraphTensor {
    /// Swap dimensions of the tensor
    pub fn permute(mut self, axes: impl ToAxes) -> GraphTensor {
        let axes = axes.to_axes();
        let current_dims = self.legacy_tracker.dims.to_vec();
        self.logical_view = self.graph().logical.apply_movement(
            self.id.index(),
            self.logical_view,
            &current_dims,
            crate::logical_recorder::Movement::Permute(axes.clone()),
        );
        self.legacy_tracker.permute(axes);
        self
    }

    /// Swap 2 dimensions. This is a view-only operation and does not materialize a new tensor
    pub fn transpose(self, dim0: usize, dim1: usize) -> GraphTensor {
        let num_dims = self.legacy_tracker.len();
        assert!(
            dim0 < num_dims && dim1 < num_dims,
            "transpose dimensions ({dim0}, {dim1}) out of bounds for tensor with {num_dims} dimensions"
        );
        let mut perm_axes: Vec<usize> = (0..num_dims).collect();
        perm_axes.swap(dim0, dim1);
        self.permute(perm_axes)
    }

    /// Transpose a 2D tensor
    pub fn t(self) -> GraphTensor {
        assert_eq!(self.legacy_tracker.len(), 2, ".t() supports only 2D tensors");
        self.transpose(0, 1)
    }

    /// Broadcast tensor along a new dimension
    pub fn expand_dim(mut self, axis: usize, size: impl Into<Expression>) -> GraphTensor {
        let size = size.into();
        let current_dims = self.legacy_tracker.dims.to_vec();
        self.logical_view = self.graph().logical.apply_movement(
            self.id.index(),
            self.logical_view,
            &current_dims,
            crate::logical_recorder::Movement::ExpandDim {
                axis,
                size: size.clone(),
            },
        );
        self.legacy_tracker.expand_dim(axis, size);
        self
    }

    /// Broadcast tensor along new dimensions on the right-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    pub fn expand_rhs(mut self, shape: impl ToShape) -> GraphTensor {
        let orig_dims = self.legacy_tracker.len();
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            self = self.expand_dim(orig_dims + i, s);
        }
        self
    }

    /// Tile a tensor along its existing dimensions without materializing a new buffer.
    pub fn repeat(mut self, repeats: impl ToShape) -> GraphTensor {
        let repeats = repeats.to_shape();
        let current_dims = self.legacy_tracker.dims.to_vec();
        self.logical_view = self.graph().logical.apply_movement(
            self.id.index(),
            self.logical_view,
            &current_dims,
            crate::logical_recorder::Movement::Repeat(repeats.clone()),
        );
        self.legacy_tracker.repeat(repeats);
        self
    }

    /// Broadcast tensor along new dimensions on the left-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    /// PyTorch-style expand: every size-1 axis whose target size differs
    /// broadcasts to it (a squeeze + broadcast expand per axis — pure,
    /// recorder-covered). Non-1 axes must already match.
    pub fn expand(mut self, new_shape: impl ToShape) -> GraphTensor {
        let target = new_shape.to_shape();
        assert_eq!(target.len(), self.rank(), "expand rank mismatch");
        for (axis, target_dim) in target.into_iter().enumerate() {
            let current = self.dims()[axis];
            if current == target_dim {
                continue;
            }
            assert_eq!(
                current,
                Expression::from(1),
                "expand: axis {axis} is {current}, only size-1 axes broadcast"
            );
            self = self.squeeze(axis).expand_dim(axis, target_dim);
        }
        self
    }

    pub fn expand_lhs(mut self, shape: impl ToShape) -> GraphTensor {
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            self = self.expand_dim(i, s);
        }
        self
    }

    pub fn expand_to_shape_on_axes(
        mut self,
        shape: impl ToShape,
        axes: impl ToAxes,
    ) -> GraphTensor {
        self.graph()
            .logical
            .poison(format!("expand_to_shape_on_axes at t{} (recorder Phase B)", self.id.index()));
        let shape = shape.to_shape();
        let axes = axes.to_axes();
        assert_eq!(shape.len(), self.legacy_tracker.len() + axes.len());
        for axis in axes.into_iter().sorted() {
            self = self.expand_dim(axis, shape[axis]);
        }
        self
    }

    /// Merge two dimensions together
    pub fn merge_dims(mut self, axis1: usize, axis2: usize) -> GraphTensor {
        let current_dims = self.legacy_tracker.dims.to_vec();
        self.logical_view = self.graph().logical.apply_movement(
            self.id.index(),
            self.logical_view,
            &current_dims,
            crate::logical_recorder::Movement::MergeDims { axis1, axis2 },
        );
        self.legacy_tracker.merge_dims(axis1, axis2);
        self
    }

    /// Flatten all dimensions into a single 1D tensor.
    pub fn flatten(mut self) -> GraphTensor {
        while self.legacy_tracker.len() > 1 {
            self = self.merge_dims(0, 1);
        }
        self
    }

    //// Split a dim into 2 dims, new dim is placed directly after original dim
    pub fn split_dims(mut self, axis: usize, new_dim_size: impl Into<Expression>) -> GraphTensor {
        let new_dim_size = new_dim_size.into();
        let current_dims = self.legacy_tracker.dims.to_vec();
        self.logical_view = self.graph().logical.apply_movement(
            self.id.index(),
            self.logical_view,
            &current_dims,
            crate::logical_recorder::Movement::SplitDims {
                axis,
                inner: new_dim_size,
            },
        );
        self.legacy_tracker.split_dims(axis, new_dim_size);
        self
    }

    /// add a new dimension of size 1 at the specified place
    pub fn unsqueeze(self, dim: usize) -> GraphTensor {
        assert!(self.legacy_tracker.len() < 10, "Shape is maxed out at 10 dimensions");
        self.expand_dim(dim, 1)
    }

    /// remove a dimension of size 1
    pub fn squeeze(mut self, axis: usize) -> GraphTensor {
        assert_eq!(
            self.dims()[axis],
            Expression::from(1),
            "Only dimensions of size 1 can be squeezed!"
        );
        let current_dims = self.legacy_tracker.dims.to_vec();
        self.logical_view = self.graph().logical.apply_movement(
            self.id.index(),
            self.logical_view,
            &current_dims,
            crate::logical_recorder::Movement::RemoveDim { axis },
        );
        self.legacy_tracker.remove_dim(axis);
        self
    }

    /// Gather elements along an axis using per-element indices (ONNX GatherElements semantics).
    ///
    /// `output[i0,..,ik] = self[i0,..,i_{axis-1}, indices[i0,..,ik], i_{axis+1},..,ik]`
    ///
    /// indices must have the same rank as self and the same shape as the output.
    pub fn gather_elements(self, indexes: GraphTensor, axis: usize) -> GraphTensor {
        self.graph()
            .logical
            .poison(format!("gather_elements at t{} (recorder Phase B)", self.id.index()));
        let dims = self.dims();
        let rank = dims.len();
        let out_shape: Vec<usize> = indexes
            .dims()
            .iter()
            .map(|d| {
                d.to_usize()
                    .expect("gather_elements: index dim must be concrete")
            })
            .collect();

        // Row-major strides: stride[i] = prod(dims[i+1..])
        let strides: Vec<usize> = (0..rank)
            .map(|i| {
                dims[i + 1..]
                    .iter()
                    .map(|d| d.to_usize().unwrap())
                    .product()
            })
            .collect();

        // Normalize negative indices for axis dim
        let axis_dim = dims[axis].to_usize().unwrap();
        let idx_f32 = indexes.cast(DType::F32);
        let zero = idx_f32
            .graph()
            .constant_float(0.0)
            .expand_rhs(idx_f32.legacy_tracker);
        let adj = idx_f32
            .graph()
            .constant_float(axis_dim as f32)
            .expand_rhs(idx_f32.legacy_tracker);
        let is_neg = idx_f32.lt(zero).cast(DType::F32);
        let idx_normalized = (idx_f32 + (is_neg * adj)).cast(DType::Int);

        // Non-axis flat index via iota + flatten_strides
        let axis_exprs: Vec<Expression> = (0..rank)
            .map(|d| {
                if d == axis {
                    Expression::from(0)
                } else {
                    Expression::from('z') * strides[d]
                }
            })
            .collect();
        let out_shape_expr: Vec<Expression> =
            out_shape.iter().map(|&s| Expression::from(s)).collect();
        let non_axis_flat = self
            .graph()
            .iota(flatten_strides(&out_shape_expr, &axis_exprs), out_shape);

        // Axis contribution from the runtime index values
        let stride_tensor = self
            .graph()
            .constant(strides[axis])
            .expand_rhs(idx_normalized.legacy_tracker);
        let flat_idx = non_axis_flat + idx_normalized * stride_tensor;

        self.gather(flat_idx)
    }

    /// Scatter updates into a copy of self at positions specified by per-element indices along an axis.
    ///
    /// ONNX ScatterElements semantics:
    /// `output[i0,..,i_{a-1}, indices[i0,..,ik], i_{a+1},..,ik] = updates[i0,..,ik]`
    ///
    /// indices and updates must have the same shape.
    /// Overlapping writes: last write wins.
    pub fn scatter_elements(
        self,
        indices: GraphTensor,
        updates: GraphTensor,
        axis: usize,
    ) -> GraphTensor {
        self.graph()
            .logical
            .poison(format!("scatter_elements at t{} (recorder Phase B)", self.id.index()));
        let data_dims = self.dims();
        let rank = data_dims.len();
        let idx_shape: Vec<usize> = indices
            .dims()
            .iter()
            .map(|d| {
                d.to_usize()
                    .expect("scatter_elements: index dim must be concrete")
            })
            .collect();

        // Row-major strides for data
        let strides: Vec<usize> = (0..rank)
            .map(|i| {
                data_dims[i + 1..]
                    .iter()
                    .map(|d| d.to_usize().unwrap())
                    .product()
            })
            .collect();

        // Normalize negative indices for axis dim
        let axis_dim = data_dims[axis].to_usize().unwrap();
        let idx_f32 = indices.cast(DType::F32);
        let zero = idx_f32
            .graph()
            .constant_float(0.0)
            .expand_rhs(idx_f32.legacy_tracker);
        let adj = idx_f32
            .graph()
            .constant_float(axis_dim as f32)
            .expand_rhs(idx_f32.legacy_tracker);
        let is_neg = idx_f32.lt(zero).cast(DType::F32);
        let idx_normalized = (idx_f32 + (is_neg * adj)).cast(DType::Int);

        // Non-axis flat index via iota + flatten_strides
        let axis_exprs: Vec<Expression> = (0..rank)
            .map(|d| {
                if d == axis {
                    Expression::from(0)
                } else {
                    Expression::from('z') * strides[d]
                }
            })
            .collect();
        let idx_shape_expr: Vec<Expression> =
            idx_shape.iter().map(|&s| Expression::from(s)).collect();
        let non_axis_flat = self.graph().iota(
            flatten_strides(&idx_shape_expr, &axis_exprs),
            idx_shape.clone(),
        );

        // Axis contribution from the runtime index values
        let stride_tensor = self
            .graph()
            .constant(strides[axis])
            .expand_rhs(idx_normalized.legacy_tracker);
        let flat_dest = non_axis_flat + idx_normalized * stride_tensor;

        // Flatten to 1D using materialize + reshape
        let flat_dest_1d = flat_dest.flatten();
        let flat_updates = updates.flatten();
        let flat_data = self.flatten();

        // Use HLIR Scatter: dest[indexes[i]] = src[i]
        let output_flat = flat_updates.scatter(flat_dest_1d, flat_data);

        // Reshape back to data_shape
        let data_shape_usize: Vec<usize> =
            data_dims.iter().map(|d| d.to_usize().unwrap()).collect();
        let mut result = output_flat;
        result.legacy_tracker = ShapeTracker::new(data_shape_usize);

        result
    }

    /// Scatter updates into a copy of self using multi-dimensional index vectors (ONNX ScatterND semantics).
    ///
    /// `indices` has shape [S0, ..., Sq-2, K] where K <= rank(data).
    /// `updates` has shape [S0, ..., Sq-2, D_K, ..., D_{r-1}].
    /// For each batch element (s0, ..., sq-2):
    ///   multi_idx = indices[s0, ..., sq-2, :]
    ///   output[multi_idx[0], ..., multi_idx[K-1], :, ..] = updates[s0, ..., sq-2, :, ..]
    pub fn scatter_nd(self, indices: GraphTensor, updates: GraphTensor) -> GraphTensor {
        self.graph()
            .logical
            .poison(format!("scatter_nd at t{} (recorder Phase B)", self.id.index()));
        let indices = indices.cast(DType::Int);
        let data_dims = self.dims();
        let data_rank = data_dims.len();
        let idx_dims = indices.dims();
        let idx_rank = idx_dims.len();

        let data_shape: Vec<usize> = data_dims
            .iter()
            .map(|d| d.to_usize().expect("scatter_nd: data dim must be concrete"))
            .collect();
        let idx_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| {
                d.to_usize()
                    .expect("scatter_nd: indices dim must be concrete")
            })
            .collect();

        let k = idx_shape[idx_rank - 1]; // last dim of indices = number of index dimensions
        assert!(k <= data_rank, "scatter_nd: K must be <= data rank");

        // Batch shape = indices shape without last dim: [S0, ..., Sq-2]
        let batch_shape: Vec<usize> = idx_shape[..idx_rank - 1].to_vec();
        let batch_numel: usize = batch_shape.iter().product::<usize>().max(1);

        // Trailing shape = data_shape[K..]
        let trailing_shape: Vec<usize> = data_shape[k..].to_vec();
        let trailing_numel: usize = trailing_shape.iter().product::<usize>().max(1);

        // Row-major strides for data
        let data_strides: Vec<usize> = (0..data_rank)
            .map(|i| data_shape[i + 1..].iter().product::<usize>().max(1))
            .collect();

        // Flatten batch dims of indices to [batch_numel, K] using materialize + reshape
        let mut indices_flat = indices;
        if idx_rank > 2 {
            indices_flat.legacy_tracker = ShapeTracker::new(vec![batch_numel, k]);
        }
        // indices_flat: [batch_numel, K] or [K] if idx_rank == 1

        // For each k_dim, extract the slice and multiply by stride
        let mut flat_base: Option<GraphTensor> = None;
        for (k_dim, &stride) in data_strides.iter().enumerate().take(k) {
            let idx_k = indices_flat.slice_along(k_dim..k_dim + 1, indices_flat.dims().len() - 1);
            let idx_k = idx_k.squeeze(idx_k.dims().len() - 1);

            let stride_tensor = self.graph().constant(stride).expand_rhs(idx_k.legacy_tracker);
            let contribution = idx_k * stride_tensor;

            flat_base = Some(match flat_base {
                Some(fb) => fb + contribution,
                None => contribution,
            });
        }
        let flat_base = flat_base.unwrap();

        let mut full_flat_dest = if trailing_shape.is_empty() || trailing_numel == 1 {
            flat_base
        } else {
            // Expand flat_base to [batch_numel, trailing_numel]
            let mut base_expanded = flat_base.expand_dim(1, trailing_numel);

            let trailing_rank = trailing_shape.len();
            for (ti, d) in (k..data_rank).enumerate() {
                let ar = self.graph().arange(data_shape[d]);
                let mut ar_shaped = ar;
                for _ in ti + 1..trailing_rank {
                    let n = ar_shaped.dims().len();
                    ar_shaped = ar_shaped.expand_dim(n, 1);
                }
                for _ in 0..ti {
                    ar_shaped = ar_shaped.expand_dim(0, 1);
                }
                ar_shaped.legacy_tracker.expand(trailing_shape.clone());
                // Flatten trailing dims using materialize + reshape
                let mut ar_flat = ar_shaped;
                ar_flat.legacy_tracker = ShapeTracker::new(vec![trailing_numel]);
                // Expand to [batch_numel, trailing_numel]
                ar_flat = ar_flat.expand_dim(0, batch_numel);

                let stride_tensor = self
                    .graph()
                    .constant(data_strides[d])
                    .expand_rhs(ar_flat.legacy_tracker);
                base_expanded += ar_flat * stride_tensor;
            }
            base_expanded
        };

        full_flat_dest = full_flat_dest.flatten();

        // Flatten data out
        let flat_updates = updates.flatten();
        let flat_data = self.flatten();

        // Use HLIR Scatter: dest[indexes[i]] = src[i]
        let output_flat = flat_updates.scatter(full_flat_dest, flat_data);

        // Reshape back to data_shape
        let mut result = output_flat;
        result.legacy_tracker = ShapeTracker::new(data_shape);

        result
    }

    pub fn gather(self, indexes: GraphTensor) -> GraphTensor {
        self.graph()
            .logical
            .poison(format!("gather at t{} (recorder Phase B)", self.id.index()));
        assert_eq!(
            indexes.dtype,
            DType::Int,
            "Gather indexes must have an integer dtype!"
        );
        let id = self.graph().add_op(
            Gather {
                input_shapes: vec![indexes.legacy_tracker, self.legacy_tracker],
                ..Default::default()
            },
            &[indexes.id, self.id],
        );
        GraphTensor::from_id(id, indexes.legacy_tracker.contiguous(), self.graph_ref, self.dtype)
    }

    /// Scatter self (src) into dest at flat 1D positions given by indexes.
    /// output = copy(dest); output[indexes[i]] = src[i]
    pub fn scatter(self, indexes: GraphTensor, dest: GraphTensor) -> GraphTensor {
        self.graph()
            .logical
            .poison(format!("scatter at t{} (recorder Phase B)", self.id.index()));
        assert_eq!(
            indexes.dtype,
            DType::Int,
            "Scatter indexes must have an integer dtype!"
        );
        // Pad src_strides with leading zero-strides when src has lower rank
        // than indexes. A zero stride reads the same src element at every
        // index position — matches PyTorch's broadcast semantics for
        // `x[idx] = scalar`. Without this, KernelScatter::compile calls
        // flatten_strides(index_shape, src_strides) with mismatched lengths
        // and panics with `assertion `left == right` failed, left: 1 right: 0`.
        let mut src_strides = self.legacy_tracker.strides.to_vec();
        let target_rank = indexes.legacy_tracker.dims.len();
        while src_strides.len() < target_rank {
            src_strides.insert(0, Expression::from(0));
        }
        let id = self.graph().add_op(
            Scatter {
                dest_shape: dest.legacy_tracker.dims.to_vec(),
                dest_strides: dest.legacy_tracker.strides.to_vec(),
                index_shape: indexes.legacy_tracker.dims.to_vec(),
                index_strides: indexes.legacy_tracker.strides.to_vec(),
                src_strides,
            },
            &[dest.id, indexes.id, self.id],
        );
        GraphTensor::from_id(id, dest.legacy_tracker.contiguous(), self.graph_ref, self.dtype)
    }

    /// Extracts sliding local windows from an input tensor.
    pub fn unfold(
        self,
        kernel: impl ToShape,
        strides: impl ToShape,
        dilation: impl ToShape,
    ) -> GraphTensor {

        let (kernel, strides, dilation) =
            (kernel.to_shape(), strides.to_shape(), dilation.to_shape());

        assert_eq!(
            self.legacy_tracker.len(),
            kernel.len(),
            "Kernel must be same number of dimensions as tensor!"
        );
        assert_eq!(
            self.legacy_tracker.len(),
            strides.len(),
            "Strides must be same number of dimensions as tensor!"
        );
        assert_eq!(
            self.legacy_tracker.len(),
            dilation.len(),
            "Dilation must be same number of dimensions as tensor!"
        );

        // Compute input strides (row-major contiguous)
        let dims = self.dims();
        let n = dims.len();
        let mut in_strides = vec![Expression::from(1); n];
        let mut acc = Expression::from(1);
        for (dim, in_stride) in dims.iter().zip(&mut in_strides).rev() {
            *in_stride = acc;
            acc *= dim;
        }

        // Per-dim window counts
        let mut win = Vec::with_capacity(n);
        for (((dim, k), s), d) in dims.iter().zip(&kernel).zip(&strides).zip(&dilation) {
            let effective_window = *d * (*k - 1) + 1;
            win.push((*dim - effective_window).floor_div(s) + 1);
        }

        // [win..., kernel...]
        let mut final_shape: Vec<Expression> = win.into_iter().map(|e| e.simplify()).collect();
        final_shape.extend(kernel.iter().copied());

        // Structure-preserving seam (see SliceView): the per-axis window
        // structure stays first-class; UnfoldView::to_egglog reproduces the
        // legacy flat iota+gather lowering for the existing pipeline.
        let window_counts = final_shape[..n].to_vec();
        let id = self.graph().add_op(
            crate::hlir::UnfoldView {
                kernel: kernel.to_vec(),
                strides: strides.to_vec(),
                dilation: dilation.to_vec(),
                window_counts: window_counts.clone(),
                input_shape: self.legacy_tracker,
            },
            &[self.id],
        );
        // Out shape [win..., k...] (rank 2n): parent axis p reads
        // win_p·stride_p + k_p·dilation_p — window arithmetic straight
        // from the unfold's own parameters.
        let out_rank = 2 * n;
        let entries: Vec<crate::logical_recorder::MapEntry> = (0..n)
            .map(|p| {
                crate::logical_recorder::MapEntry::Add(
                    Box::new(crate::logical_recorder::MapEntry::Mul(
                        Box::new(crate::logical_recorder::MapEntry::Coord {
                            from_end: out_rank - 1 - p,
                            extent: window_counts[p],
                        }),
                        strides[p].into(),
                    )),
                    Box::new(crate::logical_recorder::MapEntry::Mul(
                        Box::new(crate::logical_recorder::MapEntry::Coord {
                            from_end: out_rank - 1 - (n + p),
                            extent: kernel[p],
                        }),
                        dilation[p].into(),
                    )),
                )
            })
            .collect();
        let operand = (self.id.index(), self.logical_view, self.dims());
        self.graph().logical.view_op(
            id.index(),
            operand,
            &entries,
            final_shape.clone(),
            self.dtype,
        );
        GraphTensor::from_id(
            id,
            ShapeTracker::new(final_shape),
            self.graph_ref,
            self.dtype,
        )
    }

    /// Take a slice of a tensor along multiple dimensions.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10));
    /// let b = a.slice((2..4, 1..)); // 2x9 tensor
    /// assert_eq!(b.dims(), vec![Expression::from(2), Expression::from(9)]);
    /// ```
    pub fn slice(mut self, slice: impl ToSlice) -> GraphTensor {
        let mut ranges = slice.to_range_vec();
        ranges.extend(
            self.dims()
                .iter()
                .skip(ranges.len())
                .map(|d| (0.into(), *d)),
        ); // Make sure we have a range per dim
        if ranges.iter().any(|(st, _)| *st != 0) {
            // Start slices are VIEWS. The structure-preserving SliceView node
            // keeps the per-axis starts first-class (the logical-SSA seam);
            // its to_egglog lowers to the same iota+gather the frontend used
            // to build eagerly here, so the existing pipeline is unchanged.
            let mut new_dims = vec![];
            let mut starts = vec![];
            for (dim, (start, end)) in self.dims().into_iter().zip(ranges) {
                starts.push(start);
                new_dims.push(dim.min(end) - start);
            }
            let id = self.graph().add_op(
                crate::hlir::SliceView {
                    starts: starts.clone(),
                    out_dims: new_dims.clone(),
                    input_shape: self.legacy_tracker,
                },
                &[self.id],
            );
            // The seam node's own parameters ARE the view: parent axis p
            // reads out coordinate p plus its start.
            let rank = new_dims.len();
            let entries: Vec<crate::logical_recorder::MapEntry> = (0..rank)
                .map(|p| {
                    let coord = crate::logical_recorder::MapEntry::Coord {
                        from_end: rank - 1 - p,
                        extent: new_dims[p],
                    };
                    if starts[p] == Expression::from(0) {
                        coord
                    } else {
                        crate::logical_recorder::MapEntry::Add(
                            Box::new(coord),
                            Box::new(crate::logical_recorder::MapEntry::Lit(starts[p])),
                        )
                    }
                })
                .collect();
            let operand = (self.id.index(), self.logical_view, self.dims());
            self.graph().logical.view_op(
                id.index(),
                operand,
                &entries,
                new_dims.clone(),
                self.dtype,
            );
            GraphTensor::from_id(id, ShapeTracker::new(new_dims), self.graph_ref, self.dtype)
        } else {
            // No start slices so no iota needed, just reduce the shape down
            let mut new_dims = self.legacy_tracker.dims.to_vec();
            for (sh, (_, end)) in new_dims.iter_mut().zip(&ranges) {
                *sh = sh.min(*end);
            }
            let current_dims = self.legacy_tracker.dims.to_vec();
            self.logical_view = self.graph().logical.apply_movement(
                self.id.index(),
                self.logical_view,
                &current_dims,
                crate::logical_recorder::Movement::Shrink {
                    new_dims: new_dims.clone(),
                },
            );
            for (sh, new_dim) in self.legacy_tracker.dims.iter_mut().zip(new_dims) {
                *sh = new_dim;
            }
            self
        }
    }

    /// Take a slice of a tensor along a dimension.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10));
    /// let b = a.slice_along(4.., 1); // 5x6 tensor
    /// assert_eq!(b.dims(), vec![Expression::from(5), Expression::from(6)]);
    /// ```
    pub fn slice_along(self, slice: impl SliceRange, axis: usize) -> GraphTensor {
        let mut s = vec![(Expression::from(0), Expression::from(i64::MAX)); axis + 1];
        s[axis] = slice.bounds();
        self.slice(s)
    }

    // /// Cut out 'size' elements every 'spacing' elements on a dimension. 'size' must be smaller than the dimension
    // pub fn excise(mut self, spacing: usize, size: usize) -> GraphTensor {
    //     let n_dims = self.legacy_tracker.len();
    //     // Pad out to a multiple of spacing + size
    //     let total_size = (self.legacy_tracker.dims[n_dims - 1] + ((spacing + size) - 1))
    //         / (spacing + size)
    //         * (spacing + size);
    //     let padding = total_size - self.legacy_tracker.dims[self.legacy_tracker.indexes[n_dims - 1]];
    //     self.legacy_tracker.padding[self.legacy_tracker.indexes[n_dims - 1]].1 = padding;

    //     self = self.contiguous();
    //     // Expand a new dimension to do the slicing on
    //     let n_rows = total_size / (spacing + size);
    //     self.legacy_tracker.expand_dim(n_dims, spacing + size);
    //     // self = self.contiguous();
    //     self.legacy_tracker.dims[self.legacy_tracker.indexes[n_dims - 1]] = n_rows;
    //     self.legacy_tracker.fake[self.legacy_tracker.indexes[n_dims]] = false;

    //     // Slice
    //     self.legacy_tracker.mask[self.legacy_tracker.indexes[n_dims]].1 = spacing.into();

    //     self = self.contiguous();

    //     self.legacy_tracker.remove_dim(n_dims);
    //     self
    // }

    /// Pad out dimensions of a tensor with an element
    pub fn pad(self, padding: impl ToPad, elem: f32) -> GraphTensor {
        self.graph()
            .logical
            .poison(format!("pad at t{} (recorder Phase B)", self.id.index()));
        let mut padding = padding.to_pad_vec();
        padding.extend(vec![(0.into(), 0.into()); self.legacy_tracker.len() - padding.len()]); // Make sure we have a padding per dim
        if padding.iter().all(|(s, e)| *s == 0 && *e == 0) {
            return self;
        }
        // Structure-preserving seam (see SliceView): the read half is a
        // total clamped VIEW, the mask half a per-axis-structured indicator
        // iota; both nodes lower to the legacy flat forms for the existing
        // pipeline via their to_egglog.
        let dims = self.dims();
        let befores: Vec<Expression> = padding.iter().map(|(s, _)| *s).collect();
        let afters: Vec<Expression> = padding.iter().map(|(_, e)| *e).collect();
        let out_dims: Vec<Expression> = dims
            .iter()
            .zip(&padding)
            .map(|(d, (s, e))| *d + *s + *e)
            .collect();

        let clamped_id = self.graph().add_op(
            crate::hlir::ClampView {
                befores: befores.clone(),
                afters: afters.clone(),
                input_shape: self.legacy_tracker,
            },
            &[self.id],
        );
        let clamped = GraphTensor::from_id(
            clamped_id,
            ShapeTracker::new(out_dims.clone()),
            self.graph_ref,
            self.dtype,
        );

        let mask_id = self.graph().add_op(
            crate::hlir::MaskIota {
                befores,
                afters,
                in_dims: dims.to_vec(),
            },
            &[],
        );
        let mask = GraphTensor::from_id(
            mask_id,
            ShapeTracker::new(out_dims),
            self.graph_ref,
            DType::Int,
        )
        .cast(self.dtype);

        let masked = clamped * mask;
        if elem == 0.0 {
            masked
        } else {
            masked + ((1.0 - mask) * elem)
        }
    }

    /// Pad along an existing dimension
    pub fn pad_along(
        self,
        left: impl Into<Expression>,
        right: impl Into<Expression>,
        axis: usize,
        elem: f32,
    ) -> GraphTensor {
        let mut p = vec![(Expression::from(0), Expression::from(0)); axis + 1];
        p[axis] = (left.into(), right.into());
        self.pad(p, elem)
    }

    /// Concat along an existing dimension
    pub fn concat_along(self, rhs: GraphTensor, axis: usize) -> GraphTensor {
        // Pad and add
        self.pad_along(0, rhs.dims()[axis], axis, 0.)
            + rhs.pad_along(self.dims()[axis], 0, axis, 0.)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        frontend::{binary::tests::test_binary, unary::tests::test_unary},
        prelude::*,
        tests::assert_exact,
    };
    use candle_core::{IndexOp, Tensor};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_pad_1d(len in 1usize..64, left in 0usize..6, right in 0usize..6) {
            test_unary(
                len,
                |a| a.pad((left, right), 0.),
                |a| a.pad_with_zeros(0, left, right).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_pad_2d(rows in 1usize..32, cols in 1usize..32, top in 0usize..6, bottom in 0usize..6, left in 0usize..6, right in 0usize..6) {
            test_unary(
                (rows, cols),
                |a| a.pad(((top, bottom), (left, right)), 0.),
                |a| {
                    a.pad_with_zeros(0, top, bottom)
                        .unwrap()
                        .pad_with_zeros(1, left, right)
                        .unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_slice_pad(
            rows in 3usize..32,
            cols in 3usize..32,
            start_row in 0usize..32,
            end_row in 1usize..32,
            start_col in 0usize..32,
            end_col in 1usize..32,
            pad_top in 0usize..6,
            pad_bottom in 0usize..6,
            pad_left in 0usize..6,
            pad_right in 0usize..6,
        ) {
            prop_assume!(start_row < end_row && end_row <= rows);
            prop_assume!(start_col < end_col && end_col <= cols);
            test_unary(
                (rows, cols),
                |a| a.slice((start_row..end_row, start_col..end_col)).pad(((pad_top, pad_bottom), (pad_left, pad_right)), 0.),
                |a| {
                    a.i((start_row..end_row, start_col..end_col))
                        .unwrap()
                        .pad_with_zeros(0, pad_top, pad_bottom)
                        .unwrap()
                        .pad_with_zeros(1, pad_left, pad_right)
                        .unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_transpose(rows in 1usize..32, cols in 1usize..32) {
            test_unary(
                (rows, cols),
                |a| a.transpose(0, 1) * 1.0,
                |a| a.transpose(0, 1).unwrap(),
            );
        }
    }

    #[test]
    fn test_unfold() {
        // Need all this code because candle doesnt do unfold
        #[allow(clippy::too_many_arguments)]
        pub fn unfold_nd_f32(
            x: &[f32],
            shape: &[usize],
            strides: &[usize],
            kernel: &[usize],
            step: &[usize],
            dilation: &[usize],
            pad_before: &[usize],
            pad_after: &[usize],
        ) -> Vec<f32> {
            let n = shape.len();
            assert!(n > 0);
            assert_eq!(strides.len(), n);
            assert_eq!(kernel.len(), n);
            assert_eq!(step.len(), n);
            assert_eq!(dilation.len(), n);
            assert_eq!(pad_before.len(), n);
            assert_eq!(pad_after.len(), n);

            for d in 0..n {
                assert!(kernel[d] > 0);
                assert!(step[d] > 0);
                assert!(dilation[d] > 0);
                assert!(shape[d] > 0);
            }

            // Effective kernel size per dim: (K-1)*d + 1
            let eff_kernel: Vec<usize> =
                (0..n).map(|d| (kernel[d] - 1) * dilation[d] + 1).collect();

            // Output spatial shape (number of windows) per dim
            let mut out_shape = vec![0usize; n];
            for d in 0..n {
                let padded = shape[d] + pad_before[d] + pad_after[d];
                if padded < eff_kernel[d] {
                    return Vec::new();
                }
                out_shape[d] = (padded - eff_kernel[d]) / step[d] + 1;
            }

            let windows = prod(&out_shape);
            let window_elems = prod(kernel);
            let mut out = vec![0.0f32; windows * window_elems];

            // Precompute helpers
            let k_mul = row_major_multipliers(kernel);

            // Current output window position (row-major)
            let mut out_pos = vec![0usize; n];

            for w in 0..windows {
                if w > 0 {
                    incr_row_major(&mut out_pos, &out_shape);
                }

                // Window start in padded coordinates
                let start_padded: Vec<usize> = (0..n).map(|d| out_pos[d] * step[d]).collect();

                let base_out = w * window_elems;

                // Iterate kernel elements (flattened)
                for ke in 0..window_elems {
                    let k_idx = unravel_row_major(ke, kernel, &k_mul);

                    let mut flat: isize = 0;
                    let mut in_bounds = true;

                    for d in 0..n {
                        let p = start_padded[d] + k_idx[d] * dilation[d];
                        let logical = p as isize - pad_before[d] as isize;

                        if logical < 0 || logical >= shape[d] as isize {
                            in_bounds = false;
                            break;
                        }
                        flat += logical * strides[d] as isize;
                    }

                    let out_idx = base_out + ke;
                    out[out_idx] = if in_bounds { x[flat as usize] } else { 0.0 };
                }
            }

            out
        }

        // -------- helpers --------

        fn prod(xs: &[usize]) -> usize {
            xs.iter().copied().product()
        }

        fn row_major_multipliers(shape: &[usize]) -> Vec<usize> {
            let n = shape.len();
            let mut mul = vec![1usize; n];
            let mut acc = 1usize;
            for d in (0..n).rev() {
                mul[d] = acc;
                acc *= shape[d];
            }
            mul
        }

        fn unravel_row_major(mut idx: usize, shape: &[usize], mul: &[usize]) -> Vec<usize> {
            let n = shape.len();
            let mut coords = vec![0usize; n];
            for d in 0..n {
                coords[d] = idx / mul[d];
                idx %= mul[d];
            }
            coords
        }

        fn incr_row_major(pos: &mut [usize], shape: &[usize]) {
            for d in (0..pos.len()).rev() {
                pos[d] += 1;
                if pos[d] < shape[d] {
                    return;
                }
                pos[d] = 0;
            }
        }

        test_unary(
            5,
            |a| a.unfold(3, 1, 1),
            |a| {
                Tensor::new(
                    unfold_nd_f32(
                        &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        a.dims(),
                        a.stride(),
                        &[3],
                        &[1],
                        &[1],
                        &[0],
                        &[0],
                    ),
                    a.device(),
                )
                .unwrap()
            },
        );
        test_unary(
            (8, 10),
            |a| a.pad(((0, 2), (4, 4)), 0.).unfold((2, 3), (1, 2), (2, 1)),
            |a| {
                Tensor::new(
                    unfold_nd_f32(
                        &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        a.dims(),
                        a.stride(),
                        &[2, 3],
                        &[1, 2],
                        &[2, 1],
                        &[0, 4],
                        &[2, 3],
                    ),
                    a.device(),
                )
                .unwrap()
            },
        );
    }

    #[test]
    fn test_unfold_floor_div_shape_for_odd_window_numerator() {
        let mut cx = Graph::new();
        let inp = cx.tensor((80, 3000));
        let out = inp.pad(((0, 0), (1, 1)), 0.).unfold((1, 3), (1, 2), (1, 1));
        assert_eq!(out.dims(), &[80, 1500, 1, 3]);
    }

    #[test]
    fn test_unsqueeze() {
        let mut cx = Graph::new();
        let inp = cx.tensor((2, 2, 3));
        let out1 = inp.unsqueeze(1);
        let out2 = inp.unsqueeze(3);
        assert_eq!(out1.dims(), &[2, 1, 2, 3]);
        assert_eq!(out2.dims(), &[2, 2, 3, 1]);
        test_unary(
            (1, 3),
            |a| a.squeeze(0).expand_dim(0, 2) * 1.,
            |a| a.broadcast_as((2, 3)).unwrap(),
        );
        test_unary((2, 1, 3), |a| a.squeeze(1), |a| a.reshape((2, 3)).unwrap());
    }

    #[test]
    fn test_concat() {
        test_binary(
            17,
            32,
            |a, b| a.concat_along(b, 0),
            |a, b| Tensor::cat(&[a, b], 0).unwrap(),
        );
        test_binary(
            (10, 4),
            (10, 6),
            |a, b| a.concat_along(b, 1),
            |a, b| Tensor::cat(&[a, b], 1).unwrap(),
        );
        test_binary(
            (4, 10),
            (6, 10),
            |a, b| a.concat_along(b, 0),
            |a, b| Tensor::cat(&[a, b], 0).unwrap(),
        );
        test_unary(
            (4, 10),
            |a| a.concat_along(a, 0),
            |a| Tensor::cat(&[a.clone(), a], 0).unwrap(),
        );
    }

    #[test]
    fn test_gather_and_scatter_inverse() {
        let mut cx = Graph::new();
        let data = cx.tensor((2, 3));
        let indexes = cx.tensor_dtyped(4, DType::Int);
        let gathered = data.gather(indexes).output();
        // Inverse permutation via scatter: scatter arange at perm positions into zeros
        let perm = cx.tensor_dtyped(6, DType::Int);
        let values = cx.arange(6);
        let zeros = cx.iota(Expression::from(0usize), 6);
        let inv = values.scatter(perm, zeros).cast(DType::F32).output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        rt.set_data(data.id, vec![0., 1., 2., 3., 4., 5.]);
        rt.set_data(indexes.id, vec![5, 0, 3, 2]);
        rt.set_data(perm.id, vec![3, 2, 4, 1, 5, 0]);
        rt.execute(&cx.dyn_map);
        assert_eq!(*rt.get_f32(gathered.id), vec![5., 0., 3., 2.]);
        assert_eq!(*rt.get_f32(inv.id), vec![5., 3., 1., 0., 2., 4.]);
    }

    #[test]
    fn test_scatter_basic() {
        let mut cx = Graph::new();
        let src = cx.tensor(3);
        let indexes = cx.tensor_dtyped(3, DType::Int);
        let dest = cx.tensor(5);
        let result = src.scatter(indexes, dest).output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        rt.set_data(src.id, vec![10., 20., 30.]);
        rt.set_data(indexes.id, vec![1, 3, 4]);
        rt.set_data(dest.id, vec![0., 0., 0., 0., 0.]);
        rt.execute(&cx.dyn_map);
        assert_eq!(*rt.get_f32(result.id), vec![0., 10., 0., 20., 30.]);
    }

    #[test]
    fn test_scatter_into_nonzero_dest() {
        let mut cx = Graph::new();
        let src = cx.tensor(1);
        let indexes = cx.tensor_dtyped(1, DType::Int);
        let dest = cx.tensor(5);
        let result = src.scatter(indexes, dest).output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        rt.set_data(src.id, vec![99.]);
        rt.set_data(indexes.id, vec![2]);
        rt.set_data(dest.id, vec![1., 2., 3., 4., 5.]);
        rt.execute(&cx.dyn_map);
        assert_eq!(*rt.get_f32(result.id), vec![1., 2., 99., 4., 5.]);
    }

    #[test]
    fn test_scatter_all_positions() {
        let mut cx = Graph::new();
        let src = cx.tensor(4);
        let indexes = cx.tensor_dtyped(4, DType::Int);
        let dest = cx.tensor(4);
        let result = src.scatter(indexes, dest).output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        rt.set_data(src.id, vec![40., 30., 20., 10.]);
        rt.set_data(indexes.id, vec![3, 2, 1, 0]);
        rt.set_data(dest.id, vec![1., 2., 3., 4.]);
        rt.execute(&cx.dyn_map);
        assert_eq!(*rt.get_f32(result.id), vec![10., 20., 30., 40.]);
    }

    #[test]
    fn test_repeat_is_view_only() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3));
        let repeated = a.repeat((2, 2));

        assert_eq!(repeated.id, a.id);
        assert_eq!(
            repeated.dims(),
            vec![Expression::from(4usize), Expression::from(6usize)]
        );
    }

    #[test]
    fn test_repeat_runtime_values() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3));
        let repeated = (a.repeat((2, 2)) * 1.0).output();

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        rt.set_data(a.id, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        rt.execute(&cx.dyn_map);

        assert_exact(
            rt.get_f32(repeated.id),
            &[
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 4.0, 5.0, 6.0, //
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
            ],
        );
    }

    //     // #[test]
    //     // fn test_cumsum() {
    //     //     let mut cx = Graph::new();
    //     //     let a = cx.constant(1.).expand_dim(0, 3);
    //     //     let b = a.cumsum_last_dim().retrieve();
    //     //     let c = a
    //     //         .expand_dim(1, 3)
    //     //         .permute((1, 0))
    //     //         .cumsum_last_dim()
    //     //         .permute((1, 0))
    //     //         .retrieve();
    //     //     cx.execute();

    //     //     assert_exact(&b.data(), &[1., 2., 3.]);
    //     //     assert_exact(&c.data(), &[1., 1., 1., 2., 2., 2., 3., 3., 3.]);
    //     // }

    //     // #[test]
    //     // fn test_pool_1d() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor(5).set([1., 2., 3., 4., 5.]);
    //     //     let inp2 = cx
    //     //         .tensor((2, 5))
    //     //         .set([[15., 14., 13., 12., 11.], [1., 2., 3., 4., 5.]]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();
    //     //     // Stride 2
    //     //     let out2 = inp1.pool_last_dim(3, 2, 1).retrieve();
    //     //     // Stride 3
    //     //     let out3 = inp1.pool_last_dim(3, 3, 1).retrieve();
    //     //     // Dilation 2
    //     //     let out4 = inp1.pool_last_dim(3, 1, 2).retrieve();
    //     //     // Dilation 2 Padding 1
    //     //     let out5 = inp1.pad(((1, 1),)).pool_last_dim(3, 1, 2).retrieve();
    //     //     // Stride 1 Batch 2
    //     //     let out6 = inp2.pool_last_dim(3, 1, 1).retrieve();
    //     //     // Stride 3
    //     //     let out7 = inp2.pool_last_dim(3, 3, 1).retrieve();
    //     //     // Dilation 2
    //     //     let out8 = inp2.pool_last_dim(3, 1, 2).retrieve();
    //     //     // Dilation 2 Padding 1
    //     //     let out9 = inp2.pad(((0, 0), (1, 1))).pool_last_dim(3, 1, 2).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(&out1.data(), &[1., 2., 3., 2., 3., 4., 3., 4., 5.]);
    //     //     assert_exact(&out2.data(), &[1., 2., 3., 3., 4., 5.]);
    //     //     assert_exact(&out3.data(), &[1., 2., 3.]);
    //     //     assert_exact(&out4.data(), &[1., 3., 5.]);
    //     //     assert_exact(&out5.data(), &[0., 2., 4., 1., 3., 5., 2., 4., 0.]);
    //     //     assert_exact(
    //     //         &out6.data(),
    //     //         &[
    //     //             15., 14., 13., 14., 13., 12., 13., 12., 11., 1., 2., 3., 2., 3., 4., 3., 4., 5.,
    //     //         ],
    //     //     );
    //     //     assert_exact(&out7.data(), &[15., 14., 13., 1., 2., 3.]);
    //     //     assert_exact(&out8.data(), &[15., 13., 11., 1., 3., 5.]);
    //     //     assert_exact(
    //     //         &out9.data(),
    //     //         &[
    //     //             0., 14., 12., 15., 13., 11., 14., 12., 0., 0., 2., 4., 1., 3., 5., 2., 4., 0.,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_1d_dims() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor((4, 4)).set(vec![
    //     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    //     //     ]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(
    //     //         &out1.data(),
    //     //         &[
    //     //             1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13.,
    //     //             14., 15., 14., 15., 16.,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_2d() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor((4, 4)).set(vec![
    //     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    //     //     ]);
    //     //     // 3x3 kernel
    //     //     let out1 = inp1
    //     //         // Pool first dim first by moving it to end
    //     //         .permute((1, 0))
    //     //         .pool_last_dim(3, 1, 1)
    //     //         // Now move other dim to end
    //     //         .permute((1, 2, 0))
    //     //         .pool_last_dim(3, 1, 1)
    //     //         // Now swap middle two dims
    //     //         .permute((0, 2, 1, 3))
    //     //         // Now merge both pooled dimensions
    //     //         .reshape((4, 3, 3))
    //     //         .retrieve();

    //     //     cx.execute();

    //     //     assert_exact(
    //     //         &out1.data(),
    //     //         &[
    //     //             1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00,
    //     //             7.00, 8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00,
    //     //             14.00, 15.00, 6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_1d_dilation() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor(5).set(vec![1., 2., 3., 4., 5.]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(2, 1, 2).retrieve();
    //     //     // Stride 2
    //     //     let out2 = inp1.pool_last_dim(2, 2, 2).retrieve();
    //     //     // Stride 3
    //     //     let out3 = inp1.pool_last_dim(2, 3, 2).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
    //     //     assert_exact(&out2.data(), &[1., 3., 3., 5.]);
    //     //     assert_exact(&out3.data(), &[1., 3.]);
    //     // }

    //     // #[test]
    //     // fn test_rotate_half() {
    //     //     let mut cx = Graph::new();
    //     //     let a = cx.tensor((3, 2));
    //     //     a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
    //     //     let x1 = a.slice((.., ..1)).contiguous();
    //     //     let x2 = a.slice((.., 1..)).contiguous();
    //     //     let c = (-x2).concat_along(x1, 1);
    //     //     c.retrieve();
    //     //     cx.execute();

    //     //     let d_dev = Cpu::default();
    //     //     let d_a = d_dev.tensor_from_vec(
    //     //         vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
    //     //         (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
    //     //     );
    //     //     let d_x1 = d_a.clone().slice((.., ..1));
    //     //     let d_x2 = d_a.slice((.., 1..));
    //     //     let d_c = (-d_x2, d_x1)
    //     //         .concat_along(dfdx::shapes::Axis::<1>)
    //     //         .realize::<Rank2<3, 2>>();

    //     //     assert_close(&c.data(), &d_c.as_vec());
    //     // }
}
