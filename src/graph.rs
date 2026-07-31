//! The graph a model is authored into: dynamic-dim assumptions plus the
//! LOGICAL structure the recorder captures (M3 Step 4b: their HLIR
//! petgraph, e-graph search spaces, and compile ladder are DELETED — the
//! recorder is the only path to the e-graph, and runtimes own
//! load/bind/with_ops/search).

use petgraph::stable_graph::NodeIndex;
use rustc_hash::FxHashMap;

use crate::dtype::DType;
use crate::frontend::GraphTensor;
use crate::shape::ToShape;

/// A bucket for a dynamic dimension, defining a range of valid values.
/// For an exact value, use `min == max` (zero-length range).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimBucket {
    pub min: usize,
    pub max: usize,
    representative_override: Option<usize>,
}

impl DimBucket {
    /// Create a new bucket covering `[min, max]` inclusive.
    /// For an exact value, pass `min == max`.
    pub fn new(min: usize, max: usize) -> Self {
        assert!(min <= max, "DimBucket min ({min}) must be <= max ({max})");
        DimBucket {
            min,
            max,
            representative_override: None,
        }
    }

    /// Override the representative value used during search profiling.
    /// Must be within `[min, max]`.
    pub fn representative(mut self, val: usize) -> Self {
        assert!(
            val >= self.min && val <= self.max,
            "Representative {val} must be in [{}, {}]",
            self.min,
            self.max
        );
        self.representative_override = Some(val);
        self
    }

    /// The representative value used during search profiling.
    /// Defaults to midpoint `(min + max) / 2`.
    pub fn representative_value(&self) -> usize {
        self.representative_override
            .unwrap_or((self.min + self.max) / 2)
    }

    /// Check if `val` falls within this bucket's range.
    pub fn contains(&self, val: usize) -> bool {
        val >= self.min && val <= self.max
    }
}

#[derive(Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// The logical-model recorder — GraphTensor methods emit their
    /// logical ops here; it IS the graph (absorbed into this struct at
    /// M3 Step 4e).
    pub logical: crate::logical_graph::LogicalGraph,
    /// The tensor-id mint. Ids are plain sequence numbers (the NodeIndex
    /// type is kept as the id vocabulary; the petgraph it once indexed is
    /// gone) — they key recorder rows, binding slots, and set_data.
    next_id: u32,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Mint a fresh tensor id.
    pub(crate) fn mint_id(&mut self) -> NodeIndex {
        let id = NodeIndex::new(self.next_id as usize);
        self.next_id += 1;
        id
    }

    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor_dtyped("", shape, DType::default())
    }

    /// Create a new tensor with shape S and this dtype. Dtype is DECLARED
    /// at creation (purity ruling 2026-07-30: as_dtype is gone — a
    /// different dtype downstream is a logical cast, never a mutation of
    /// the declaration).
    pub fn tensor_dtyped(&mut self, shape: impl ToShape, dtype: DType) -> GraphTensor {
        self.named_tensor_dtyped("", shape, dtype)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: impl ToString, shape: impl ToShape) -> GraphTensor {
        self.named_tensor_dtyped(name, shape, DType::default())
    }

    /// Named + dtyped input declaration — the one true constructor.
    pub fn named_tensor_dtyped(
        &mut self,
        name: impl ToString,
        shape: impl ToShape,
        dtype: DType,
    ) -> GraphTensor {
        let name = name.to_string();
        let id = self.mint_id();
        let tensor = GraphTensor {
            id,
            graph_ref: self,
            dims: shape.to_shape().into_iter().collect(),
            dtype,
            logical_value: None,
        };
        let logical = self
            .logical
            .input(id.index(), &name, &tensor.dims(), tensor.dtype);
        tensor.with_logical(logical)
    }
}
