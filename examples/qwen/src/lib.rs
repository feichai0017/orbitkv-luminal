//! Qwen3 on the native ladder — the pure-logical conversion exemplar.
//!
//! The decode graph is STEP-INVARIANT: s = 1, a fixed (max_seq, kv_dim)
//! paged cache per layer, and every per-step quantity (token id, query
//! position, rope table row, scatter slot, cache state) arrives as
//! DATA. That is what the positional-mask attention buys: one
//! `search`, then one `execute` per token — prefill is the same graph
//! fed the prompt token by token. Cache state flows runtime-out →
//! runtime-in between steps through `get_f32`/`set_data` (the caches
//! are functional in/out pairs, not in-place buffers).

pub mod hf;
pub mod model;
pub mod weights;

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex, TypedBuffer};
use luminal::ssa_reference::SsaReferenceRuntime;
use model::{Qwen, QwenDims};
use std::error::Error;
use std::io::Write as _;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const EOS_TOKEN: u32 = 151645; // <|im_end|>
const STOP_TOKEN: u32 = 151643; // <|endoftext|>

pub struct QwenRunConfig {
    pub repo_id: String,
    pub prompt: String,
    /// Layers actually instantiated. The reference runtime stages f32
    /// on the host, so full depth (36 layers ≈ 16 GB of parameters
    /// alone) does not fit a small machine — the default truncates and
    /// says so loudly. Full-depth fidelity is backend re-seat business.
    pub layers: usize,
    pub max_seq: usize,
    pub gen_tokens: usize,
    /// Skip the HF download and run the real anatomy on deterministic
    /// pseudo-random parameters.
    pub random_weights: bool,
    pub repetition_penalty: f32,
    pub search: ImplementationSearchOptions,
}

impl Default for QwenRunConfig {
    fn default() -> Self {
        Self {
            repo_id: "Qwen/Qwen3-4B".to_string(),
            prompt: "Explain what a neural network is in a paragraph.".to_string(),
            layers: 1,
            max_seq: 64,
            gen_tokens: 8,
            random_weights: false,
            repetition_penalty: 1.05,
            // Small on purpose: every candidate plan is profiled by
            // EXECUTING it, and a real-hyperparameter step is seconds
            // of interpreter time.
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

/// The recorded step graph and its input/output handles.
pub struct DecodeStep {
    pub cx: Graph,
    pub model: Qwen,
    pub token: GraphTensor,
    pub q_pos: GraphTensor,
    pub rope_cos: GraphTensor,
    pub rope_sin: GraphTensor,
    pub rope_rot: GraphTensor,
    pub gather_idx: GraphTensor,
    pub scatter_idx: GraphTensor,
    pub caches: Vec<(GraphTensor, GraphTensor)>,
    pub logits: GraphTensor,
    pub cache_outs: Vec<(GraphTensor, GraphTensor)>,
    pub max_seq: usize,
}

impl DecodeStep {
    pub fn build(dims: &QwenDims, max_seq: usize) -> Self {
        let mut cx = Graph::new();
        let model = Qwen::init(&mut cx, dims);
        let token = cx.tensor_dtyped(1, DType::Int);
        let q_pos = cx.tensor_dtyped(1, DType::Int);
        let rope_cos = cx.tensor((1, dims.head_dim));
        let rope_sin = cx.tensor((1, dims.head_dim));
        let rope_rot = cx.tensor((dims.head_dim, dims.head_dim));
        let gather_idx = cx.tensor_dtyped(max_seq, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let caches: Vec<_> = (0..dims.layers)
            .map(|_| {
                (
                    cx.tensor((max_seq, dims.kv_dim())),
                    cx.tensor((max_seq, dims.kv_dim())),
                )
            })
            .collect();
        let (logits, cache_outs) = model.forward(
            token,
            q_pos,
            rope_cos,
            rope_sin,
            rope_rot,
            &caches,
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
            rope_cos,
            rope_sin,
            rope_rot,
            gather_idx,
            scatter_idx,
            caches,
            logits,
            cache_outs,
            max_seq,
        }
    }
}

/// The step driver: owns the searched runtime, the host rope tables and
/// the per-layer cache state; `step(token)` returns the logits row.
pub struct Decoder {
    step: DecodeStep,
    rt: SsaReferenceRuntime,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    pub k_state: Vec<Vec<f32>>,
    pub v_state: Vec<Vec<f32>>,
    head_dim: usize,
    pub pos: usize,
}

impl Decoder {
    /// Load, search once (profiling with the supplied weights and the
    /// position-0 step inputs), and stage the step-invariant constants.
    pub fn start(
        step: DecodeStep,
        weight_pairs: &[(NodeIndex, TypedBuffer)],
        options: &ImplementationSearchOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let dims = step.model.dims.clone();
        let max_seq = step.max_seq;
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let (cos_table, sin_table) =
            luminal_nn::rope_tables_split_half(&positions, dims.head_dim, dims.rope_theta, 1.0);
        let rot = luminal_nn::rope_pairing_matrix(dims.head_dim, false);
        let gather: Vec<i32> = (0..max_seq as i32).collect();
        let zero_cache = vec![0f32; max_seq * dims.kv_dim()];

        let mut search_data: FxHashMap<NodeIndex, TypedBuffer> =
            weight_pairs.iter().cloned().collect();
        search_data.insert(step.token.id, vec![0i32].into());
        search_data.insert(step.q_pos.id, vec![0i32].into());
        search_data.insert(step.rope_cos.id, cos_table[..dims.head_dim].to_vec().into());
        search_data.insert(step.rope_sin.id, sin_table[..dims.head_dim].to_vec().into());
        search_data.insert(step.rope_rot.id, rot.clone().into());
        search_data.insert(step.gather_idx.id, gather.clone().into());
        search_data.insert(step.scatter_idx.id, vec![0i32].into());
        for (k, v) in &step.caches {
            search_data.insert(k.id, zero_cache.clone().into());
            search_data.insert(v.id, zero_cache.clone().into());
        }

        let mut rt = SsaReferenceRuntime::load(&step.cx)?;
        let outcome = rt.search(&search_data, options)?;
        println!(
            "search: {} plans profiled, best {:.1} ms\n{}",
            outcome.plans_profiled,
            outcome.best_nanos as f64 / 1e6,
            outcome.timings.summary()
        );

        // Step-invariant inputs stage once; per-step data overwrites the rest.
        for (id, data) in weight_pairs {
            rt.set_data(*id, data.clone());
        }
        rt.set_data(step.rope_rot.id, rot);
        rt.set_data(step.gather_idx.id, gather);

        let layers = dims.layers;
        Ok(Self {
            step,
            rt,
            cos_table,
            sin_table,
            k_state: vec![zero_cache.clone(); layers],
            v_state: vec![zero_cache; layers],
            head_dim: dims.head_dim,
            pos: 0,
        })
    }

    /// Feed one token at the current position; returns the logits row.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, Box<dyn Error>> {
        if self.pos >= self.step.max_seq {
            return Err(format!("sequence exceeded max_seq {}", self.step.max_seq).into());
        }
        let row = self.pos * self.head_dim..(self.pos + 1) * self.head_dim;
        self.rt.set_data(self.step.token.id, vec![token as i32]);
        self.rt.set_data(self.step.q_pos.id, vec![self.pos as i32]);
        self.rt
            .set_data(self.step.rope_cos.id, self.cos_table[row.clone()].to_vec());
        self.rt
            .set_data(self.step.rope_sin.id, self.sin_table[row].to_vec());
        self.rt
            .set_data(self.step.scatter_idx.id, vec![self.pos as i32]);
        for (layer, (k, v)) in self.step.caches.iter().enumerate() {
            self.rt.set_data(k.id, self.k_state[layer].clone());
            self.rt.set_data(v.id, self.v_state[layer].clone());
        }
        self.rt.execute()?;
        for (layer, (k, v)) in self.step.cache_outs.iter().enumerate() {
            self.k_state[layer] = self.rt.get_f32(k.id)?.clone();
            self.v_state[layer] = self.rt.get_f32(v.id)?.clone();
        }
        self.pos += 1;
        Ok(self.rt.get_f32(self.step.logits.id)?.clone())
    }
}

fn qwen3_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
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

pub fn run_qwen(config: QwenRunConfig) -> Result<(), Box<dyn Error>> {
    let mut dims = QwenDims::qwen3_4b();
    assert!(config.layers >= 1 && config.layers <= dims.layers);
    if config.layers < dims.layers {
        println!(
            "NOTE: instantiating {} of {} layers — real weights, real pipeline, truncated \
             depth (the f32 reference runtime cannot hold the full stack); the text below \
             is a pipeline demonstration, not the model's real output.",
            config.layers, dims.layers
        );
    }
    dims.layers = config.layers;

    let (tokenizer, weight_source): (Option<Tokenizer>, _) = if config.random_weights {
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
            .encode(qwen3_chat_prompt(&config.prompt).as_str(), false)
            .map_err(|err| err as Box<dyn Error>)?
            .get_ids()
            .to_vec(),
        // Random-weight mode has no vocabulary; use fixed token ids.
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
    let pairs = match &weight_source {
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
