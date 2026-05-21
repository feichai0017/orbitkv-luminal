use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use luminal::{graph::BuildSearchSpaceOptions, prelude::*};
use luminal_cuda_lite::runtime::CudaRuntime;

#[path = "/workspaces/luminal/examples/gemma4_moe/src/model.rs"]
mod gemma4_moe_model;
#[path = "/workspaces/luminal/examples/gemma/src/model.rs"]
mod gemma_model;
#[path = "/workspaces/luminal/examples/llama/src/model.rs"]
mod llama_model;
#[path = "/workspaces/luminal/examples/paged_llama/src/model.rs"]
mod paged_llama_model;
#[path = "/workspaces/luminal/examples/qwen3_moe/src/model.rs"]
mod qwen3_moe_model;
#[path = "/workspaces/luminal/examples/qwen/src/model.rs"]
mod qwen_model;
#[path = "/workspaces/luminal/examples/whisper/src/model.rs"]
mod whisper_model;

struct Case {
    name: &'static str,
    build: fn() -> Graph,
    options: BuildSearchSpaceOptions,
}

fn main() -> anyhow::Result<()> {
    let out_dir = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/workspaces/luminal/scratch/egglog_repros/programs"));
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;

    let cases = [
        Case {
            name: "llama",
            build: build_llama,
            options: BuildSearchSpaceOptions::new().max_memory_mib(500),
        },
        Case {
            name: "paged_llama",
            build: build_paged_llama,
            options: BuildSearchSpaceOptions::default(),
        },
        Case {
            name: "qwen",
            build: build_qwen,
            options: BuildSearchSpaceOptions::default(),
        },
        Case {
            name: "qwen3_moe",
            build: build_qwen3_moe,
            options: BuildSearchSpaceOptions::default(),
        },
        Case {
            name: "gemma",
            build: build_gemma,
            options: BuildSearchSpaceOptions::default(),
        },
        Case {
            name: "gemma4_moe",
            build: build_gemma4_moe,
            options: BuildSearchSpaceOptions::default(),
        },
        Case {
            name: "whisper",
            build: build_whisper,
            options: BuildSearchSpaceOptions::default(),
        },
    ];

    for case in cases {
        export_case(&out_dir, &case)?;
    }

    Ok(())
}

fn export_case(out_dir: &Path, case: &Case) -> anyhow::Result<()> {
    let start = Instant::now();
    let mut graph = (case.build)();
    let program = graph.export_egglog_search_program_with_options::<CudaRuntime>(case.options);
    let path = out_dir.join(format!("{}.egg", case.name));
    fs::write(&path, program.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!(
        "{:<12} {:>10} bytes {:>8} lines in {:.3}s -> {}",
        case.name,
        program.len(),
        program.lines().count(),
        start.elapsed().as_secs_f64(),
        path.display(),
    );
    Ok(())
}

fn output_caches(cache_outputs: &[(GraphTensor, GraphTensor)]) {
    for (k_out, v_out) in cache_outputs {
        k_out.output();
        v_out.output();
    }
}

fn build_llama() -> Graph {
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = llama_model::KVCache::new(&mut cx, 4096);
    let (logits, cache_outputs) =
        llama_model::Llama::init(&mut cx).forward(input, pos_ids, &kv_cache);
    logits.output();
    output_caches(&cache_outputs);
    cx.set_dim('s', 1);
    cx.set_dim('p', 1);
    cx
}

fn build_paged_llama() -> Graph {
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let q_pos = cx.named_tensor("q_pos", 's').as_dtype(DType::Int);
    let scatter_idx = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let attn_mask = cx.named_tensor("attn_mask", ('s', 'c'));
    let kv_cache = paged_llama_model::PagedKVCache::new(&mut cx, 8192);
    let (logits, cache_outputs) = paged_llama_model::Llama::init(&mut cx).forward(
        input,
        q_pos,
        scatter_idx,
        gather_idx,
        attn_mask,
        &kv_cache,
    );
    logits.output();
    output_caches(&cache_outputs);
    cx
}

fn build_qwen() -> Graph {
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = qwen_model::KVCache::new(&mut cx, 4096);
    let (logits, cache_outputs) =
        qwen_model::Qwen::init(&mut cx).forward(input, pos_ids, &kv_cache);
    logits.output();
    output_caches(&cache_outputs);
    cx
}

fn build_qwen3_moe() -> Graph {
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = qwen3_moe_model::KVCache::new(&mut cx, 4096);
    let (logits, cache_outputs) =
        qwen3_moe_model::Qwen3MoE::init(&mut cx).forward(input, pos_ids, &kv_cache);
    logits.output();
    output_caches(&cache_outputs);
    cx
}

fn build_gemma() -> Graph {
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = gemma_model::KVCache::new(&mut cx, 4096);
    let (logits, cache_outputs) =
        gemma_model::Gemma::init(&mut cx).forward(input, pos_ids, &kv_cache);
    logits.output();
    output_caches(&cache_outputs);
    cx
}

fn build_gemma4_moe() -> Graph {
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = gemma4_moe_model::KVCache::new(&mut cx, 4096);
    let (logits, cache_outputs) =
        gemma4_moe_model::Gemma4MoE::init(&mut cx).forward(input, pos_ids, &kv_cache);
    logits.output();
    output_caches(&cache_outputs);
    cx
}

fn build_whisper() -> Graph {
    let mut cx = Graph::default();
    let mel = cx
        .named_tensor(
            "mel",
            (whisper_model::N_MELS, whisper_model::N_AUDIO_CTX * 2),
        )
        .persist();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = whisper_model::KVCache::new(&mut cx, whisper_model::N_TEXT_CTX);
    let whisper = whisper_model::Whisper::init(&mut cx);
    let xa = whisper.encoder.forward(mel);
    let (logits, cache_outputs) = whisper.decoder.forward(input, pos_ids, xa, &kv_cache);
    logits.output();
    output_caches(&cache_outputs);
    cx
}
