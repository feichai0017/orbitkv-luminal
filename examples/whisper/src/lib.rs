//! openai/whisper-tiny.en on the native ladder. Encoder and decoder
//! live in ONE step-invariant graph (faithful to the parked example):
//! the mel spectrogram is an input, the encoder recomputes every
//! execute, the decoder advances one token per step over the slot-pool
//! self-attention caches (positions as DATA — the legacy slice-bound
//! hack is dead), and cross-attention K/V recompute from the encoder
//! output. One search, one execute per token.

pub mod audio;
pub mod hf;
pub mod model;
pub mod weights;

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex, Ns, TypedBuffer};
use luminal::ssa_reference::SsaReferenceRuntime;
use luminal_nn::{CacheState, KvCachePool, PositionSlots};
use model::{Whisper, WhisperDims};
use std::error::Error;

pub const TOKEN_SOT: u32 = 50_257;
pub const TOKEN_NO_TIMESTAMPS: u32 = 50_362;
pub const TOKEN_EOT: u32 = 50_256;

pub struct TranscribeStep {
    pub cx: Graph,
    pub model: Whisper,
    pub mel: GraphTensor,
    pub token: GraphTensor,
    pub q_pos: GraphTensor,
    pub gather_idx: GraphTensor,
    pub scatter_idx: GraphTensor,
    pub pool: KvCachePool,
    pub logits: GraphTensor,
    pub cache_outs: Vec<(GraphTensor, GraphTensor)>,
}

impl TranscribeStep {
    pub fn build(dims: &WhisperDims) -> Self {
        let mut cx = Graph::new();
        let model = Whisper::init(&mut cx, dims);
        let mel = cx.tensor((dims.n_mels, dims.mel_frames()));
        let token = cx.tensor_dtyped(1, DType::Int);
        let q_pos = cx.tensor_dtyped(1, DType::Int);
        let gather_idx = cx.tensor_dtyped(dims.text_ctx, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let pool = KvCachePool::new(
            &mut cx,
            dims.text_layers,
            dims.text_ctx,
            dims.state,
            &Ns::root().child("cache"),
        );
        let xa = model.encode(mel);
        let (logits, cache_outs) =
            model.decode_step(token, q_pos, xa, &pool, gather_idx, scatter_idx);
        let logits = logits.output();
        let cache_outs: Vec<_> = cache_outs
            .into_iter()
            .map(|(k, v)| (k.output(), v.output()))
            .collect();
        Self {
            cx,
            model,
            mel,
            token,
            q_pos,
            gather_idx,
            scatter_idx,
            pool,
            logits,
            cache_outs,
        }
    }
}

pub struct Transcriber {
    step: TranscribeStep,
    rt: SsaReferenceRuntime,
    pub state: CacheState,
    pub slots: PositionSlots,
}

impl Transcriber {
    pub fn start(
        step: TranscribeStep,
        mel: &[f32],
        weight_pairs: &[(NodeIndex, TypedBuffer)],
        options: &ImplementationSearchOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let dims = step.model.dims.clone();
        let slots = PositionSlots::new(dims.text_ctx);
        let state = step.pool.zero_state();

        let mut search_data: FxHashMap<NodeIndex, TypedBuffer> =
            weight_pairs.iter().cloned().collect();
        search_data.insert(step.mel.id, mel.to_vec().into());
        search_data.insert(step.token.id, vec![0i32].into());
        search_data.insert(step.q_pos.id, vec![0i32].into());
        search_data.insert(step.gather_idx.id, slots.full_gather());
        search_data.insert(step.scatter_idx.id, vec![0i32].into());
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
        rt.set_data(step.mel.id, mel.to_vec());
        rt.set_data(step.gather_idx.id, slots.full_gather());

        Ok(Self {
            step,
            rt,
            state,
            slots,
        })
    }

    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, Box<dyn Error>> {
        let (scatter, q_pos) = self.slots.step()?;
        self.rt.set_data(self.step.token.id, vec![token as i32]);
        self.rt.set_data(self.step.q_pos.id, q_pos);
        self.rt.set_data(self.step.scatter_idx.id, scatter);
        self.state.stage(&mut self.rt, &self.step.pool);
        self.rt.execute()?;
        self.state.absorb(&self.rt, &self.step.cache_outs)?;
        Ok(self.rt.get_f32(self.step.logits.id)?.clone())
    }
}

/// Whisper's greedy pick with token suppression: special ids
/// (>= SOT) are suppressed; EOT is allowed EXCEPT on the very first
/// generated token (avoids empty transcripts).
pub fn greedy_pick(logits: &[f32], first_step: bool, suppress_from: u32) -> u32 {
    let mut best = (0u32, f32::NEG_INFINITY);
    for (index, &value) in logits.iter().enumerate() {
        let id = index as u32;
        if id >= suppress_from && id != TOKEN_EOT {
            continue;
        }
        if id == TOKEN_EOT && first_step {
            continue;
        }
        if value > best.1 {
            best = (id, value);
        }
    }
    best.0
}
