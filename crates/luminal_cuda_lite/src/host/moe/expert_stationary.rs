//! Launchers for expert_stationary.cu: the MoE block as four stream-ordered
//! launches (index build, phase 1, phase 2, reduce) with each expert's
//! weights read once per row-tile instead of once per (token, expert) pair.
//!
//! Only worth it when tokens-per-expert exceeds ~1. Measured against the
//! shipped GEMV at gpt-oss-120b shapes (E=128, hidden=inter=2880, top_k=4,
//! full MoE block including the index build and the reduce):
//!
//! | seq |    1 |    8 |   32 |   64 |  128 |  256 |  512 |
//! |-----|------|------|------|------|------|------|------|
//! |  x  | 0.51 | 0.74 | 0.99 | 1.42 | 1.92 | 2.30 | 2.52 |
//!
//! so `fused.rs` keeps the GEMV below [`MIN_PAIRS`] and decode never crosses
//! it. See `decode.rs` for the path this replaces at prefill widths.

use std::sync::{Arc, OnceLock};

use crate::{
    compile_module_image_for_current_device,
    cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
};

const SOURCE: &str = include_str!("expert_stationary.cu");
const BLOCK_THREADS: u32 = 256;
const WARPS: usize = 8;
/// Rows per warp and tokens per tile; must match the instantiated kernels.
const ES_R: usize = 8;

/// Pair count at or above which the expert-stationary path is used. The
/// crossover measures at ~128 pairs (seq 32); 256 keeps a margin so that a
/// skewed router — which lowers the average tokens-per-expert — cannot land
/// us on the wrong side of it.
pub const MIN_PAIRS: usize = 256;

/// Dispatch predicate. `LUMINAL_MOE_ES=0` forces the GEMV everywhere (kill
/// switch); `LUMINAL_MOE_ES_MIN_PAIRS` moves the crossover.
pub fn use_expert_stationary(num_pairs: usize) -> bool {
    use std::sync::OnceLock;
    static MIN: OnceLock<Option<usize>> = OnceLock::new();
    let min = *MIN.get_or_init(|| {
        if std::env::var("LUMINAL_MOE_ES").is_ok_and(|v| v == "0") {
            return None;
        }
        Some(
            std::env::var("LUMINAL_MOE_ES_MIN_PAIRS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(MIN_PAIRS),
        )
    });
    min.is_some_and(|m| num_pairs >= m)
}

struct EsKernels {
    _module: Arc<CudaModule>,
    index: CudaFunction,
    phase1: CudaFunction,
    phase2: CudaFunction,
    reduce: CudaFunction,
}

static KERNEL: OnceLock<EsKernels> = OnceLock::new();

fn kernel(stream: &Arc<CudaStream>) -> &'static EsKernels {
    KERNEL.get_or_init(|| {
        let image = compile_module_image_for_current_device(stream.context(), SOURCE)
            .expect("moe expert-stationary kernel should compile");
        let module = stream
            .context()
            .load_module(image)
            .expect("moe expert-stationary module should load");
        let f = |name: &str| {
            module
                .load_function(name)
                .unwrap_or_else(|e| panic!("{name} should exist: {e}"))
        };
        EsKernels {
            index: f("moe_build_expert_index"),
            phase1: f("moe_phase1_es_m4_r8"),
            phase2: f("moe_phase2_es_m4_r8"),
            reduce: f("moe_phase2_es_reduce"),
            _module: module,
        }
    })
}

/// Force the NVRTC compile (idempotent); called at extract time so the cost
/// lands outside timed profiling trials.
pub fn warm(stream: &Arc<CudaStream>) {
    let _ = kernel(stream);
}

/// Scratch this path needs beyond the GEMV's `hidden` buffer, in bytes:
/// `expert_off[E+1]`, `sorted_pairs[pairs]`, `partial[pairs, hidden]`.
pub fn scratch_bytes(num_experts: usize, num_pairs: usize, hidden_dim: usize) -> usize {
    (num_experts + 1) * 4 + num_pairs * 4 + num_pairs * hidden_dim * 4
}

/// The MoE block for `seq` tokens, expert-stationary. `expert_off_ptr`,
/// `sorted_pairs_ptr` and `partial_ptr` are the three scratch regions sized
/// by [`scratch_bytes`]; the rest match `decode::fused_moe_decode`.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_expert_stationary(
    stream: &Arc<CudaStream>,
    x_ptr: u64,
    gu_q_ptr: u64,
    gu_scale_ptr: u64,
    gu_bias_ptr: u64,
    dn_q_ptr: u64,
    dn_scale_ptr: u64,
    dn_bias_ptr: u64,
    topk_ids_ptr: u64,
    topk_w_ptr: u64,
    hidden_scratch_ptr: u64,
    expert_off_ptr: u64,
    sorted_pairs_ptr: u64,
    partial_ptr: u64,
    out_ptr: u64,
    hidden_dim: usize,
    inter: usize,
    top_k: usize,
    seq: usize,
    num_experts: usize,
    idx_row_stride: usize,
    alpha: f32,
    limit: f32,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        hidden_dim.is_multiple_of(32) && inter.is_multiple_of(32),
        "expert-stationary MoE requires 32-aligned dims: hidden={hidden_dim}, inter={inter}"
    );
    // Phase 1 pairs gate/up rows within a warp, so a warp's row block must
    // start even and hold whole pairs.
    anyhow::ensure!(
        ES_R.is_multiple_of(2),
        "ES_R must be even (gate/up interleaving)"
    );
    anyhow::ensure!(idx_row_stride >= top_k, "idx_row_stride must be >= top_k");
    // The index build is a single block; its shared counters are per-expert.
    anyhow::ensure!(
        num_experts <= 1024,
        "expert-stationary index build supports <= 1024 experts, got {num_experts}"
    );
    if seq == 0 || top_k == 0 {
        return Ok(());
    }

    let k = kernel(stream);
    let (h, i, tk, s, ne, stride) = (
        hidden_dim as i32,
        inter as i32,
        top_k as i32,
        seq as i32,
        num_experts as i32,
        idx_row_stride as i32,
    );
    let prof = crate::hostop_profile::enabled();

    // ── index: counting sort of the (token, k) pairs by expert ──
    let t0 = std::time::Instant::now();
    unsafe {
        stream
            .launch_builder(&k.index)
            .arg(&topk_ids_ptr)
            .arg(&expert_off_ptr)
            .arg(&sorted_pairs_ptr)
            .arg(&s)
            .arg(&tk)
            .arg(&stride)
            .arg(&ne)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: (2 * num_experts * 4) as u32,
            })?;
    }
    if prof {
        stream.synchronize()?;
        crate::hostop_profile::record("  MoE idx: expert counting sort", t0.elapsed());
    }

    // grid.x tiles the output rows, grid.y is the expert.
    let rows_per_block = (WARPS * ES_R) as u32;
    let tiles = |rows: usize| LaunchConfig {
        grid_dim: (
            (rows as u32).div_ceil(rows_per_block).max(1),
            num_experts as u32,
            1,
        ),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    let t1 = std::time::Instant::now();
    unsafe {
        stream
            .launch_builder(&k.phase1)
            .arg(&x_ptr)
            .arg(&gu_q_ptr)
            .arg(&gu_scale_ptr)
            .arg(&gu_bias_ptr)
            .arg(&expert_off_ptr)
            .arg(&sorted_pairs_ptr)
            .arg(&hidden_scratch_ptr)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .arg(&alpha)
            .arg(&limit)
            .launch(tiles(2 * inter))?;
    }
    if prof {
        stream.synchronize()?;
        crate::hostop_profile::record("  MoE p1: gate_up ES + swiglu", t1.elapsed());
    }

    let t2 = std::time::Instant::now();
    unsafe {
        stream
            .launch_builder(&k.phase2)
            .arg(&hidden_scratch_ptr)
            .arg(&dn_q_ptr)
            .arg(&dn_scale_ptr)
            .arg(&dn_bias_ptr)
            .arg(&expert_off_ptr)
            .arg(&sorted_pairs_ptr)
            .arg(&topk_w_ptr)
            .arg(&partial_ptr)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .launch(tiles(hidden_dim))?;
    }
    if prof {
        stream.synchronize()?;
        crate::hostop_profile::record("  MoE p2: down ES", t2.elapsed());
    }

    // ── reduce: sum each token's top_k partial rows, in pair order ──
    let t3 = std::time::Instant::now();
    let elems = seq * hidden_dim;
    unsafe {
        stream
            .launch_builder(&k.reduce)
            .arg(&partial_ptr)
            .arg(&out_ptr)
            .arg(&h)
            .arg(&tk)
            .arg(&s)
            .launch(LaunchConfig {
                grid_dim: (elems.div_ceil(256).min(4096).max(1) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    if prof {
        stream.synchronize()?;
        crate::hostop_profile::record("  MoE reduce: topk mix", t3.elapsed());
    }
    Ok(())
}
