mod model;

use luminal::prelude::*;
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use model::*;
use std::{io::Write, path::PathBuf, time::Duration};
use tokenizers::Tokenizer;
use tracing::{span, Level};

const REPO_ID: &str = "NousResearch/Meta-Llama-3-8B-Instruct";

/// Get the model directory, respecting HF_HUB_CACHE or HF_HOME environment variables.
/// Falls back to "setup/" for backward compatibility.
fn get_model_dir() -> PathBuf {
    // Check HF_HUB_CACHE first, then derive from HF_HOME, then use default
    let cache_dir = std::env::var("HF_HUB_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HF_HOME")
                .ok()
                .map(|h| PathBuf::from(h).join("hub"))
        })
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
                .unwrap_or_else(|_| PathBuf::from(".cache/huggingface/hub"))
        });

    // HF cache structure: models--<org>--<repo>/snapshots/<revision>/
    let repo_dir = cache_dir.join(format!("models--{}", REPO_ID.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");

    // Find the snapshot directory (use the first/only one, or latest modified)
    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        if let Some(snapshot) = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        {
            let path = snapshot.path();
            // Verify required files exist
            if path.join("tokenizer.json").exists() && path.join("model_combined.safetensors").exists() {
                return path;
            }
        }
    }

    // No valid model directory found
    eprintln!("Error: Model files not found!");
    eprintln!("Please run setup.py first to download the model:");
    eprintln!("  cd examples/llama/setup && uv run setup.py");
    eprintln!();
    eprintln!("You can set HF_HUB_CACHE to control where files are stored:");
    eprintln!("  export HF_HUB_CACHE=/path/to/cache");
    std::process::exit(1);
}

// This example compiles and runs Llama 3 8B on CUDA.

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 5;
    let search_graphs = 5; // the number of graphs we want to search during compilation
    let prompt = "Hello, how are you";

    // Set up tracing to perfetto
    let trace_session = luminal_tracing::subscriber()
        // .perfetto("trace.pftrace")
        .env_filter(format!(
            "{}=trace,luminal=debug,luminal_cuda=debug",
            env!("CARGO_PKG_NAME")
        ))
        .init();

    // Set up cuda context and stream
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Get model directory (respects HF_HUB_CACHE env var)
    let model_dir = get_model_dir();
    println!("Using model directory: {}", model_dir.display());

    // Tokenize prompt
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut sentence = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();

    // Allocate kv cache
    let mut kv_cache = KVCache::new(&stream, max_seq_len);

    // Create compute graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let model = model::Llama::init(&mut cx);
    let logits = model.forward(input, token_ids, &kv_cache).output();

    // Build search space
    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    // Load model weights from safetensors file
    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    let weights_path = model_dir.join("model_combined.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    // Run search process
    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 0);
    runtime.set_data(input, vec![1]);
    runtime.set_data(token_ids, vec![0]);
    runtime = cx.search(runtime, search_graphs);
    kv_cache.reset();

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    // Decode loop
    let mut prev_seq = 0;
    let mut fwd_durations = vec![];
    for i in 0..gen_tokens {
        let start = std::time::Instant::now();
        let span = if i == 0 {
            span!(Level::INFO, "prefill")
        } else {
            span!(Level::INFO, "decode")
        };
        let _entered = span.enter();

        // Set runtime dimensions
        let seq_len = sentence.len();
        cx.set_dim('s', seq_len);
        cx.set_dim('p', prev_seq);

        // Set input data
        runtime.set_data(
            input,
            sentence.iter().map(|i| *i as i32).collect::<Vec<_>>(),
        );
        runtime.set_data(
            token_ids,
            (prev_seq as i32..(seq_len + prev_seq) as i32).collect::<Vec<_>>(),
        );

        // Execute forward pass
        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);

        // Sample next token
        let sample_span = span!(Level::INFO, "sample");
        let _sample_entered = sample_span.enter();
        sentence = vec![*sample(&logits_data, VOCAB_SIZE).last().unwrap()];
        prev_seq += seq_len;
        print!("{}", tokenizer.decode(&sentence, true).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();

    // Report benchmarks
    println!("  TTFT: {:.2} ms", fwd_durations[0].as_secs_f64() * 1e3);
    println!(
        "  TPOT: {:.2} ms",
        (fwd_durations.iter().sum::<Duration>() / fwd_durations.len() as u32).as_secs_f64() * 1e3
    );
    runtime.print_execution_stats();
    // Dump cuda trace to timeline
    trace_session.stop();
    if let Some(path) = trace_session.perfetto_path {
        runtime.record_cuda_perfetto_trace(path);
    }
}

#[tracing::instrument(skip_all)]
fn sample(logits: &[f32], vocab_size: usize) -> Vec<u32> {
    logits
        .chunks_exact(vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0 as u32
        })
        .collect()
}
