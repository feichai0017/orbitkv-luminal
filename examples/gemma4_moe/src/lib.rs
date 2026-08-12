//! google/gemma-4-26B-A4B on the native ladder — the heterogeneous
//! zoo example: per-role head dims and KV widths (the heterogeneous
//! pool), per-role rope of DIFFERENT table widths (sliding 256 full /
//! full 512 partial-0.25), parallel dense+MoE, learned residual
//! scalars, logit softcap. Position-slots driver; step-invariant
//! decode.

pub mod hf;
pub mod model;
pub mod weights;

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex, TypedBuffer};
use luminal::ssa_reference::SsaReferenceRuntime;
use luminal_nn::{CacheState, KvCachePool, PositionSlots};
use model::{Gemma4Dims, Gemma4Moe};
use std::error::Error;
use std::io::Write as _;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const EOS_TOKEN: u32 = 106; // <end_of_turn>
const STOP_TOKEN: u32 = 1; // <eos>

pub struct Gemma4RunConfig {
    pub repo_id: String,
    pub prompt: String,
    /// Truncated instantiation (f32 reference cannot hold 32 layers).
    pub layers: usize,
    pub max_seq: usize,
    pub gen_tokens: usize,
    pub random_weights: bool,
    pub repetition_penalty: f32,
    pub search: ImplementationSearchOptions,
}

impl Default for Gemma4RunConfig {
    fn default() -> Self {
        Self {
            repo_id: "google/gemma-4-26B-A4B".to_string(),
            prompt: "Explain what a neural network is in a paragraph.".to_string(),
            layers: 1,
            max_seq: 64,
            gen_tokens: 8,
            random_weights: false,
            repetition_penalty: 1.05,
            search: ImplementationSearchOptions {
                generations: 2,
                generation_size: 4,
                mutations: 2,
                trials: 1,
                seed: 0,
            },
        }
    }
}

pub struct DecodeStep {
    pub cx: Graph,
    pub model: Gemma4Moe,
    pub token: GraphTensor,
    pub q_pos: GraphTensor,
    pub rope_sliding: (GraphTensor, GraphTensor, GraphTensor),
    pub rope_full: (GraphTensor, GraphTensor, GraphTensor),
    pub gather_idx: GraphTensor,
    pub scatter_idx: GraphTensor,
    pub pool: KvCachePool,
    pub logits: GraphTensor,
    pub cache_outs: Vec<(GraphTensor, GraphTensor)>,
}

impl DecodeStep {
    pub fn build(dims: &Gemma4Dims, max_seq: usize) -> Self {
        let mut cx = Graph::new();
        let model = Gemma4Moe::init(&mut cx, dims);
        let token = cx.tensor_dtyped(1, DType::Int);
        let q_pos = cx.tensor_dtyped(1, DType::Int);
        let rope_sliding = (
            cx.tensor((1, dims.sliding_head_dim)),
            cx.tensor((1, dims.sliding_head_dim)),
            cx.tensor((dims.sliding_head_dim, dims.sliding_head_dim)),
        );
        let rope_full = (
            cx.tensor((1, dims.full_head_dim)),
            cx.tensor((1, dims.full_head_dim)),
            cx.tensor((dims.full_head_dim, dims.full_head_dim)),
        );
        let gather_idx = cx.tensor_dtyped(max_seq, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let pool = KvCachePool::new_heterogeneous(&mut cx, max_seq, &dims.kv_dims());
        let (logits, cache_outs) = model.forward(
            token,
            q_pos,
            rope_sliding,
            rope_full,
            &pool,
            gather_idx,
            scatter_idx,
        );
        let logits = logits.output();
        let cache_outs: Vec<_> = cache_outs
            .into_iter()
            .map(|(k, v)| (k.output(), v.output()))
            .collect();
        Self {
            cx,
            model,
            token,
            q_pos,
            rope_sliding,
            rope_full,
            gather_idx,
            scatter_idx,
            pool,
            logits,
            cache_outs,
        }
    }
}

pub struct Decoder {
    step: DecodeStep,
    rt: SsaReferenceRuntime,
    sliding_tables: (Vec<f32>, Vec<f32>),
    full_tables: (Vec<f32>, Vec<f32>),
    pub state: CacheState,
    pub slots: PositionSlots,
    sliding_head_dim: usize,
    full_head_dim: usize,
}

impl Decoder {
    pub fn start(
        step: DecodeStep,
        weight_pairs: &[(NodeIndex, TypedBuffer)],
        options: &ImplementationSearchOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let dims = step.model.dims.clone();
        let max_seq = step.pool.slots;
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        // Per-ROLE rope: sliding = head_dim 256 theta 10k full rotary;
        // full = head_dim 512 theta 1M PARTIAL 0.25 (zero-angle lanes
        // pass through the pairing form).
        let sliding_tables = luminal_nn::rope_tables_split_half(
            &positions,
            dims.sliding_head_dim,
            10_000.0,
            1.0,
        );
        let full_tables = luminal_nn::rope_tables_partial(
            &positions,
            dims.full_head_dim,
            1_000_000.0,
            dims.full_partial_rotary,
        );
        let rot_sliding = luminal_nn::rope_pairing_matrix(dims.sliding_head_dim, false);
        let rot_full = luminal_nn::rope_pairing_matrix(dims.full_head_dim, false);
        let slots = PositionSlots::new(max_seq);
        let state = step.pool.zero_state();

        let mut search_data: FxHashMap<NodeIndex, TypedBuffer> =
            weight_pairs.iter().cloned().collect();
        search_data.insert(step.token.id, vec![0i32].into());
        search_data.insert(step.q_pos.id, vec![0i32].into());
        search_data.insert(
            step.rope_sliding.0.id,
            sliding_tables.0[..dims.sliding_head_dim].to_vec().into(),
        );
        search_data.insert(
            step.rope_sliding.1.id,
            sliding_tables.1[..dims.sliding_head_dim].to_vec().into(),
        );
        search_data.insert(step.rope_sliding.2.id, rot_sliding.clone().into());
        search_data.insert(
            step.rope_full.0.id,
            full_tables.0[..dims.full_head_dim].to_vec().into(),
        );
        search_data.insert(
            step.rope_full.1.id,
            full_tables.1[..dims.full_head_dim].to_vec().into(),
        );
        search_data.insert(step.rope_full.2.id, rot_full.clone().into());
        search_data.insert(step.gather_idx.id, slots.full_gather());
        search_data.insert(step.scatter_idx.id, vec![0i32].into());
        for (layer, (k, v)) in step.pool.layers.iter().enumerate() {
            search_data.insert(k.id, state.k[layer].clone().into());
            search_data.insert(v.id, state.v[layer].clone().into());
        }

        let mut rt = SsaReferenceRuntime::load(&step.cx)?;
        let outcome = rt.search(&search_data, options)?;
        println!(
            "search: {} plans profiled, best {:.1} ms\n{}",
            outcome.plans_profiled,
            outcome.best_nanos as f64 / 1e6,
            outcome.timings.summary()
        );

        for (id, data) in weight_pairs {
            rt.set_data(*id, data.clone());
        }
        rt.set_data(step.rope_sliding.2.id, rot_sliding);
        rt.set_data(step.rope_full.2.id, rot_full);
        rt.set_data(step.gather_idx.id, slots.full_gather());

        Ok(Self {
            step,
            rt,
            sliding_tables,
            full_tables,
            state,
            slots,
            sliding_head_dim: dims.sliding_head_dim,
            full_head_dim: dims.full_head_dim,
        })
    }

    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, Box<dyn Error>> {
        let (scatter, q_pos) = self.slots.step()?;
        let pos = self.slots.pos - 1;
        let s_row = pos * self.sliding_head_dim..(pos + 1) * self.sliding_head_dim;
        let f_row = pos * self.full_head_dim..(pos + 1) * self.full_head_dim;
        self.rt.set_data(self.step.token.id, vec![token as i32]);
        self.rt.set_data(self.step.q_pos.id, q_pos);
        self.rt.set_data(
            self.step.rope_sliding.0.id,
            self.sliding_tables.0[s_row.clone()].to_vec(),
        );
        self.rt.set_data(
            self.step.rope_sliding.1.id,
            self.sliding_tables.1[s_row].to_vec(),
        );
        self.rt.set_data(
            self.step.rope_full.0.id,
            self.full_tables.0[f_row.clone()].to_vec(),
        );
        self.rt.set_data(
            self.step.rope_full.1.id,
            self.full_tables.1[f_row].to_vec(),
        );
        self.rt.set_data(self.step.scatter_idx.id, scatter);
        self.state.stage(&mut self.rt, &self.step.pool);
        self.rt.execute()?;
        self.state.absorb(&self.rt, &self.step.cache_outs)?;
        Ok(self.rt.get_f32(self.step.logits.id)?.clone())
    }
}

fn gemma_chat_prompt(user_prompt: &str) -> String {
    format!("<bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n")
}

fn argmax_with_penalty(logits: &[f32], seen: &FxHashMap<u32, ()>, penalty: f32) -> u32 {
    let mut best = (0u32, f32::NEG_INFINITY);
    for (index, &raw) in logits.iter().enumerate() {
        let mut value = raw;
        if seen.contains_key(&(index as u32)) {
            if value > 0.0 {
                value /= penalty;
            } else {
                value *= penalty;
            }
        }
        if value > best.1 {
            best = (index as u32, value);
        }
    }
    best.0
}

pub fn run_gemma4_moe(config: Gemma4RunConfig) -> Result<(), Box<dyn Error>> {
    let mut dims = Gemma4Dims::gemma4_26b_a4b();
    assert!(config.layers >= 1 && config.layers <= dims.layers);
    if config.layers < dims.layers {
        println!(
            "NOTE: instantiating {} of {} layers — real weights, real pipeline, truncated \
             depth; the text below is a pipeline demonstration, not the model's real output.",
            config.layers, dims.layers
        );
    }
    dims.layers = config.layers;

    let (tokenizer, model_dir) = if config.random_weights {
        (None, None)
    } else {
        let model_dir = hf::prepare_hf_model(&config.repo_id)?;
        println!("Using model directory: {}", model_dir.display());
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|err| err as Box<dyn Error>)?;
        (Some(tokenizer), Some(model_dir))
    };

    let prompt_tokens: Vec<u32> = match &tokenizer {
        Some(tokenizer) => tokenizer
            .encode(gemma_chat_prompt(&config.prompt).as_str(), false)
            .map_err(|err| err as Box<dyn Error>)?
            .get_ids()
            .to_vec(),
        None => vec![1, 2, 3],
    };
    if prompt_tokens.len() + config.gen_tokens > config.max_seq {
        return Err(format!(
            "prompt ({}) + gen_tokens ({}) exceeds max_seq ({})",
            prompt_tokens.len(),
            config.gen_tokens,
            config.max_seq
        )
        .into());
    }

    println!("Recording the decode-step graph ({} layers)...", dims.layers);
    let step = DecodeStep::build(&dims, config.max_seq);
    let pairs = match &model_dir {
        Some(dir) => {
            println!("Loading weights...");
            weights::load_safetensors_weights(&step.model, dir)?
        }
        None => weights::random_weights(&step.model),
    };
    println!("Searching (one step-invariant graph, profiled with real data)...");
    let mut decoder = Decoder::start(step, &pairs, &config.search)?;
    drop(pairs);

    println!(
        "Prompt: {} tokens, generating up to {}",
        prompt_tokens.len(),
        config.gen_tokens
    );
    let prefill_start = Instant::now();
    let mut logits = Vec::new();
    for &token in &prompt_tokens {
        logits = decoder.step(token)?;
    }
    let prefill = prefill_start.elapsed();

    let mut seen: FxHashMap<u32, ()> = FxHashMap::default();
    let mut step_times: Vec<Duration> = Vec::new();
    let mut generated = 0usize;
    while generated < config.gen_tokens {
        let next = argmax_with_penalty(&logits, &seen, config.repetition_penalty);
        seen.insert(next, ());
        generated += 1;
        if next == EOS_TOKEN || next == STOP_TOKEN {
            break;
        }
        if let Some(tokenizer) = &tokenizer {
            let piece = tokenizer
                .decode(&[next], true)
                .map_err(|err| err as Box<dyn Error>)?;
            print!("{piece}");
            std::io::stdout().flush()?;
        } else {
            print!("[{next}] ");
            std::io::stdout().flush()?;
        }
        if generated == config.gen_tokens {
            break;
        }
        let start = Instant::now();
        logits = decoder.step(next)?;
        step_times.push(start.elapsed());
    }
    println!();
    println!(
        "  prefill: {:.2} s for {} tokens",
        prefill.as_secs_f64(),
        prompt_tokens.len()
    );
    if !step_times.is_empty() {
        let per_token =
            step_times.iter().sum::<Duration>().as_secs_f64() / step_times.len() as f64;
        println!("  decode: {per_token:.2} s/token");
    }
    Ok(())
}
