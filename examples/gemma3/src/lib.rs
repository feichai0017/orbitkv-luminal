//! unsloth/gemma-3-4b-it on the native ladder — step-invariant decode
//! with DUAL rope tables (local 10k / global 1M with /8 scaling) fed
//! per layer role, the sliding-window mask on locals, and the
//! position-slots cache driver.

pub mod hf;
pub mod model;
pub mod weights;

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex, Ns, TypedBuffer};
use luminal::ssa_reference::SsaReferenceRuntime;
use luminal_nn::{CacheState, KvCachePool, PositionSlots};
use model::{Gemma3, Gemma3Dims};
use std::error::Error;
use std::io::Write as _;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const EOS_TOKEN: u32 = 106; // <end_of_turn>
const STOP_TOKEN: u32 = 1; // <eos>

pub struct Gemma3RunConfig {
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

impl Default for Gemma3RunConfig {
    fn default() -> Self {
        Self {
            repo_id: "unsloth/gemma-3-4b-it".to_string(),
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
    pub model: Gemma3,
    pub token: GraphTensor,
    pub q_pos: GraphTensor,
    pub rope_local: (GraphTensor, GraphTensor),
    pub rope_global: (GraphTensor, GraphTensor),
    pub rope_rot: GraphTensor,
    pub gather_idx: GraphTensor,
    pub scatter_idx: GraphTensor,
    pub pool: KvCachePool,
    pub logits: GraphTensor,
    pub cache_outs: Vec<(GraphTensor, GraphTensor)>,
}

impl DecodeStep {
    pub fn build(dims: &Gemma3Dims, max_seq: usize) -> Self {
        let mut cx = Graph::new();
        let model = Gemma3::init(&mut cx, dims);
        let token = cx.tensor_dtyped(1, DType::Int);
        let q_pos = cx.tensor_dtyped(1, DType::Int);
        let rope_local = (
            cx.tensor((1, dims.head_dim)),
            cx.tensor((1, dims.head_dim)),
        );
        let rope_global = (
            cx.tensor((1, dims.head_dim)),
            cx.tensor((1, dims.head_dim)),
        );
        let rope_rot = cx.tensor((dims.head_dim, dims.head_dim));
        let gather_idx = cx.tensor_dtyped(max_seq, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let pool = KvCachePool::new(
            &mut cx,
            dims.layers,
            max_seq,
            dims.kv_dim(),
            &Ns::root().child("cache"),
        );
        let (logits, cache_outs) = model.forward(
            token,
            q_pos,
            rope_local,
            rope_global,
            rope_rot,
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
            rope_local,
            rope_global,
            rope_rot,
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
    local_tables: (Vec<f32>, Vec<f32>),
    global_tables: (Vec<f32>, Vec<f32>),
    pub state: CacheState,
    pub slots: PositionSlots,
    head_dim: usize,
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
        // Dual-theta tables: locals at 10k unscaled, globals at 1M with
        // the /8 linear position scaling.
        let local_tables =
            luminal_nn::rope_tables_split_half(&positions, dims.head_dim, 10_000.0, 1.0);
        let global_tables =
            luminal_nn::rope_tables_split_half(&positions, dims.head_dim, 1_000_000.0, 1.0 / 8.0);
        let rot = luminal_nn::rope_pairing_matrix(dims.head_dim, false);
        let slots = PositionSlots::new(max_seq);
        let state = step.pool.zero_state();

        let mut search_data: FxHashMap<NodeIndex, TypedBuffer> =
            weight_pairs.iter().cloned().collect();
        search_data.insert(step.token.id, vec![0i32].into());
        search_data.insert(step.q_pos.id, vec![0i32].into());
        search_data.insert(
            step.rope_local.0.id,
            local_tables.0[..dims.head_dim].to_vec().into(),
        );
        search_data.insert(
            step.rope_local.1.id,
            local_tables.1[..dims.head_dim].to_vec().into(),
        );
        search_data.insert(
            step.rope_global.0.id,
            global_tables.0[..dims.head_dim].to_vec().into(),
        );
        search_data.insert(
            step.rope_global.1.id,
            global_tables.1[..dims.head_dim].to_vec().into(),
        );
        search_data.insert(step.rope_rot.id, rot.clone().into());
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
        rt.set_data(step.rope_rot.id, rot);
        rt.set_data(step.gather_idx.id, slots.full_gather());

        Ok(Self {
            step,
            rt,
            local_tables,
            global_tables,
            state,
            slots,
            head_dim: dims.head_dim,
        })
    }

    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, Box<dyn Error>> {
        let (scatter, q_pos) = self.slots.step()?;
        let pos = self.slots.pos - 1;
        let row = pos * self.head_dim..(pos + 1) * self.head_dim;
        self.rt.set_data(self.step.token.id, vec![token as i32]);
        self.rt.set_data(self.step.q_pos.id, q_pos);
        self.rt.set_data(
            self.step.rope_local.0.id,
            self.local_tables.0[row.clone()].to_vec(),
        );
        self.rt.set_data(
            self.step.rope_local.1.id,
            self.local_tables.1[row.clone()].to_vec(),
        );
        self.rt.set_data(
            self.step.rope_global.0.id,
            self.global_tables.0[row.clone()].to_vec(),
        );
        self.rt.set_data(
            self.step.rope_global.1.id,
            self.global_tables.1[row].to_vec(),
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

pub fn run_gemma3(config: Gemma3RunConfig) -> Result<(), Box<dyn Error>> {
    let mut dims = Gemma3Dims::gemma3_4b();
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
