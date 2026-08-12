//! Meta-Llama-3-8B-Instruct over the PAGE-TABLE cache driver — the
//! multi-sequence form: one slot pool serves several sequences, the
//! host [`luminal_nn::PageTable`] maps logical positions to physical
//! slots, and each BATCHED tick's visibility (causality +
//! cross-sequence isolation + unwritten-slot exclusion) arrives as one
//! (rows, slots) additive mask ([`luminal_nn::paged_attention_masked`]).
//! The model definition is examples/llama3's — one model, two cache
//! drivers, per the cache-builder ruling. The graph is step-invariant
//! at a fixed batch width (ROWS = 2): one search, one execute per
//! tick, each row carrying any sequence's next token.

pub use llama3::model::{Llama3, Llama3Dims};
pub use llama3::{hf, weights};

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex, TypedBuffer};
use luminal::ssa_reference::SsaReferenceRuntime;
use luminal_nn::{CacheState, KvCachePool, PageTable};
use std::error::Error;

/// Fixed batch width of the step-invariant tick graph.
pub const ROWS: usize = 2;

pub struct BatchStep {
    pub cx: Graph,
    pub model: Llama3,
    pub tokens: GraphTensor,
    pub rope_cos: GraphTensor,
    pub rope_sin: GraphTensor,
    pub rope_rot: GraphTensor,
    pub gather_idx: GraphTensor,
    pub scatter_idx: GraphTensor,
    pub mask: GraphTensor,
    pub pool: KvCachePool,
    pub logits: GraphTensor,
    pub cache_outs: Vec<(GraphTensor, GraphTensor)>,
}

impl BatchStep {
    pub fn build(dims: &Llama3Dims, slots: usize) -> Self {
        let mut cx = Graph::new();
        let model = Llama3::init(&mut cx, dims);
        let tokens = cx.tensor_dtyped(ROWS, DType::Int);
        let rope_cos = cx.tensor((ROWS, dims.head_dim));
        let rope_sin = cx.tensor((ROWS, dims.head_dim));
        let rope_rot = cx.tensor((dims.head_dim, dims.head_dim));
        let gather_idx = cx.tensor_dtyped(slots, DType::Int);
        let scatter_idx = cx.tensor_dtyped(ROWS, DType::Int);
        let mask = cx.tensor((ROWS, slots));
        let pool = KvCachePool::new(&mut cx, dims.layers, slots, dims.kv_dim());

        let mut x = model.embed.forward(tokens);
        let mut cache_outs = Vec::with_capacity(model.blocks.len());
        for (layer, block) in model.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward_rope_masked(
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
                gather_idx,
                scatter_idx,
                mask,
                rope_cos,
                rope_sin,
                rope_rot,
            );
            x = next;
            cache_outs.push((k_cache, v_cache));
        }
        let logits = model
            .lm_head
            .forward(model.final_norm.forward(x))
            .output();
        let cache_outs: Vec<_> = cache_outs
            .into_iter()
            .map(|(k, v)| (k.output(), v.output()))
            .collect();
        Self {
            cx,
            model,
            tokens,
            rope_cos,
            rope_sin,
            rope_rot,
            gather_idx,
            scatter_idx,
            mask,
            pool,
            logits,
            cache_outs,
        }
    }
}

/// The tick driver: PageTable slot allocation + the searched runtime.
pub struct Ticker {
    step: BatchStep,
    rt: SsaReferenceRuntime,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    pub table: PageTable,
    pub state: CacheState,
    head_dim: usize,
}

impl Ticker {
    pub fn start(
        step: BatchStep,
        weight_pairs: &[(NodeIndex, TypedBuffer)],
        options: &ImplementationSearchOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let dims = step.model.dims.clone();
        let slots = step.pool.slots;
        let positions: Vec<f32> = (0..slots).map(|p| p as f32).collect();
        let (cos_table, sin_table) =
            luminal_nn::rope_tables_split_half(&positions, dims.head_dim, dims.rope_theta, 1.0);
        let rot = luminal_nn::rope_pairing_matrix(dims.head_dim, false);
        let state = step.pool.zero_state();

        let mut search_data: FxHashMap<NodeIndex, TypedBuffer> =
            weight_pairs.iter().cloned().collect();
        search_data.insert(step.tokens.id, vec![0i32; ROWS].into());
        search_data.insert(
            step.rope_cos.id,
            cos_table[..ROWS * dims.head_dim].to_vec().into(),
        );
        search_data.insert(
            step.rope_sin.id,
            sin_table[..ROWS * dims.head_dim].to_vec().into(),
        );
        search_data.insert(step.rope_rot.id, rot.clone().into());
        search_data.insert(
            step.gather_idx.id,
            (0..slots as i32).collect::<Vec<i32>>().into(),
        );
        search_data.insert(step.scatter_idx.id, vec![0i32, 1].into());
        search_data.insert(step.mask.id, vec![-1e9f32; ROWS * slots].into());
        for (layer, (k, v)) in step.pool.layers.iter().enumerate() {
            search_data.insert(k.id, state.k[layer].clone().into());
            search_data.insert(v.id, state.v[layer].clone().into());
        }

        let mut rt = SsaReferenceRuntime::load(&step.cx)?;
        let outcome = rt.search(&search_data, options)?;
        println!(
            "search: {} plans profiled, best {:.1} ms",
            outcome.plans_profiled,
            outcome.best_nanos as f64 / 1e6,
        );

        for (id, data) in weight_pairs {
            rt.set_data(*id, data.clone());
        }
        rt.set_data(step.rope_rot.id, rot);
        rt.set_data(
            step.gather_idx.id,
            (0..slots as i32).collect::<Vec<i32>>(),
        );

        Ok(Self {
            step,
            rt,
            cos_table,
            sin_table,
            table: PageTable::new(slots),
            state,
            head_dim: dims.head_dim,
        })
    }

    /// One batched tick: exactly [`ROWS`] (sequence, token) rows — every
    /// row advances its sequence by one token. Returns the (ROWS, vocab)
    /// logits.
    pub fn tick(&mut self, rows: &[(usize, u32)]) -> Result<Vec<f32>, Box<dyn Error>> {
        assert_eq!(rows.len(), ROWS, "the tick graph has a fixed batch width");
        // Positions BEFORE allocation drive this tick's rope rows.
        let positions: Vec<usize> = rows
            .iter()
            .map(|(sequence, _)| self.table.sequences[*sequence].len())
            .collect();
        let requests: Vec<(usize, usize)> = rows.iter().map(|(s, _)| (*s, 1)).collect();
        let (scatter, mask, total_s) = self.table.tick_dense(&requests)?;
        assert_eq!(total_s, ROWS);

        let tokens: Vec<i32> = rows.iter().map(|(_, t)| *t as i32).collect();
        let mut cos = Vec::with_capacity(ROWS * self.head_dim);
        let mut sin = Vec::with_capacity(ROWS * self.head_dim);
        for pos in &positions {
            cos.extend_from_slice(&self.cos_table[pos * self.head_dim..(pos + 1) * self.head_dim]);
            sin.extend_from_slice(&self.sin_table[pos * self.head_dim..(pos + 1) * self.head_dim]);
        }

        self.rt.set_data(self.step.tokens.id, tokens);
        self.rt.set_data(self.step.rope_cos.id, cos);
        self.rt.set_data(self.step.rope_sin.id, sin);
        self.rt.set_data(self.step.scatter_idx.id, scatter);
        self.rt.set_data(self.step.mask.id, mask);
        self.state.stage(&mut self.rt, &self.step.pool);
        self.rt.execute()?;
        self.state.absorb(&self.rt, &self.step.cache_outs)?;
        Ok(self.rt.get_f32(self.step.logits.id)?.clone())
    }
}
