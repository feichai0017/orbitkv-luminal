//! `SinkAttention` — host op wrapping the FA3/Hopper AttentionSink paged
//! batch-prefill kernel.
//!
//! Runtime inputs (7): `q` (s, nq*hd) bf16, `k_pool`/`v_pool`
//! (num_slots, nkv*hd) bf16 (post-scatter pool states), `kv_indices` (c,)
//! Int (compact page table, page_size 1), `qo_indptr`/`kv_indptr` (r,) Int
//! on DEVICE (read back per execute), `sinks` (nq,) F32.
//!
//! Output: (nq, s, hd) F32 — the layout+dtype of the reference chain's
//! attention output point, produced by a fused transpose+upcast kernel.
//! Decode is the same kernel at qo_len=1 (no SM90 decode-with-sink kernel).
//!
//! The rewrite rule (sink_attention.egg) matches the paged gpt-oss sink
//! attention chain and unions this op in; which host-mask Input feeds the
//! chain ("mask_sliding" vs "mask_full") selects window_left.

use std::sync::Arc;

use luminal::{
    egglog_utils::api::{Rule, SortDef, sort},
    egglog_utils::base::{EXPRESSION, F64, OP_KIND},
    egglog_utils::{SerializedEGraph, extract_expr},
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::cudarc::driver::{CudaStream, DevicePtr, result};

use super::super::{DeviceBuffer, HostOp};
use super::jit;
use super::{INT_WORKSPACE_SIZE, bytes_to_i32_vec, flashinfer_workspaces, page_locked_workspace};

/// Grow-only device scratch (q transpose in, kernel output out), reused
/// across calls instead of a per-call alloc + trailing sync. Stream-ordered
/// reuse on the same stream is safe (each execute's indptr-readback sync
/// drains prior consumers); on a stream change (tests) the old buffers are
/// leaked rather than dropped, since their context may be gone. Bounded by
/// the largest tick, so leaking on replace/grow costs nothing real.
static SCRATCH: std::sync::Mutex<Option<(usize, crate::cudarc::driver::CudaSlice<u8>)>> =
    std::sync::Mutex::new(None);

/// Sink value making the kernel's sink-augmented softmax degenerate to plain
/// softmax. NOT -inf: a fully-masked row would give `-inf - -inf = NaN`.
/// -1e30 underflows `exp2` to exactly 0 while staying finite.
const NEUTRAL_SINK: f32 = -1.0e30;
/// Indptr readback + FA3 plan, memoized for the span of ONE graph execution.
///
/// Every layer's attention op reads the same two index buffers and builds the
/// same plan from them: the indptrs are graph Inputs written once per step, and
/// a model's layers share `num_qo_heads`/`num_kv_heads`/`head_dim`. Measured on
/// gpt-oss-120b (36 layers, 16-way decode) the readback alone was 51.7 ms of a
/// 71.8 ms tick — 72% — because each read ends in `stream.synchronize()`. The
/// graph therefore drained the pipeline 72 times per tick and never overlapped
/// host-side planning with GPU work. Prefill was worse: 638 ms of 700.
///
/// The key is deliberately conservative: same execution, same buffers, same
/// lengths, same head geometry; anything else re-reads and re-plans. The
/// GENERATION component is what makes staleness impossible — the indptr
/// CONTENTS change every tick while the buffer pointers do not, so a
/// pointer-only key would happily serve the previous tick's plan.
type PlanKey = (u64, u64, u64, usize, usize, usize, usize, usize);

#[derive(Clone)]
struct CachedPlan {
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    plan_info: [i64; 16],
    plan_info_len: i32,
}

static PLAN_CACHE: std::sync::Mutex<Option<(PlanKey, CachedPlan)>> = std::sync::Mutex::new(None);

/// Bumped once per `execute`. The plan cache keys on it so a plan cannot
/// survive into a later execution, where the indptr buffers may well have been
/// rewritten at the same device addresses -- the pointers alone cannot tell
/// those apart.
static EXEC_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn bump_exec_generation() {
    EXEC_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn exec_generation() -> u64 {
    EXEC_GENERATION.load(std::sync::atomic::Ordering::Relaxed)
}

/// Kill switch: `LUMINAL_FA3_PLAN_CACHE=0` restores the read-and-plan-per-layer
/// behaviour. Kept because reusing one plan across layers is the kind of change
/// that would fail silently — wrong attention still produces fluent text — so
/// being able to A/B correctness inside one process, without a rebuild, is
/// worth a branch that is predicted-taken forever.
fn plan_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LUMINAL_FA3_PLAN_CACHE").map_or(true, |v| v != "0"))
}

/// `LUMINAL_FA3_PLAN_VERIFY=1`: on every cache HIT, redo the readback and the
/// plan and assert the result is byte-identical to what was cached.
///
/// This is how the cache is shown to be a semantic no-op. Comparing two SERVER
/// PROCESSES cannot show it: the schedule search is stochastic, so two starts
/// compile different (equally valid) schedules whose float association differs,
/// and a one-token divergence tells you nothing about the cache. Verifying
/// inside one process holds the schedule fixed and isolates the change.
fn plan_verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LUMINAL_FA3_PLAN_VERIFY").is_ok_and(|v| v != "0"))
}

static NEUTRAL_SINKS: std::sync::Mutex<Option<(usize, crate::cudarc::driver::CudaSlice<u8>)>> =
    std::sync::Mutex::new(None);

#[derive(Debug)]
pub struct SinkAttention {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// The 's' (batch tokens) dimension expression.
    pub batch_dim: Expression,
    /// Softmax scale; 0.0 = default `1/sqrt(head_dim)`.
    pub sm_scale: f64,
    /// FlashInfer window_left convention (visible previous positions);
    /// -1 = full attention. Selects the swa .so variant at compile time.
    pub window_left: i64,
    /// Whether the matched chain had per-head sink logits. `false` is plain
    /// softmax (llama3): no sink atoms, so no 7th input to wire. The kernel has
    /// no sink-free mode and rejects null, but its math degenerates exactly —
    /// `m_new = (log_sink > m) ? log_sink : m`,
    /// `d_new = exp2(log_sink - m_new) + d*scale` — so `execute` feeds a
    /// neutral buffer instead of branching the kernel.
    pub has_sinks: bool,
}

impl Default for SinkAttention {
    fn default() -> Self {
        Self {
            num_qo_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            batch_dim: Expression::default(),
            sm_scale: 0.0,
            window_left: -1,
            has_sinks: true,
        }
    }
}

impl EgglogOp for SinkAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "SinkAttention",
            &[
                ("num_qo_heads", EXPRESSION),
                ("num_kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("batch_dim", EXPRESSION),
                ("sm_scale", F64),
                ("window_left", F64),
                ("has_sinks", F64),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        // q, k_pool, v_pool, kv_indices, qo_indptr, kv_indptr[, sinks]
        if self.has_sinks { 7 } else { 6 }
    }

    fn rewrites(&self) -> Vec<Rule> {
        // The FA3 kernels are Hopper-only (sm_90a WGMMA/TMA): emit no rules
        // on other architectures so the search never selects the op there.
        if crate::device_compute_major() != 9 {
            return vec![];
        }
        vec![
            Rule::raw(include_str!("sink_attention.egg")),
            // Sink-free variant (llama3 and other plain-softmax GQA models),
            // kept in its own file — see its header.
            Rule::raw(include_str!("paged_attention.egg")),
        ]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let num_qo_heads = extract_expr(egraph, kind_children[0], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let num_kv_heads = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let head_dim = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let batch_dim = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let sm_scale: f64 = egraph.enodes[kind_children[4]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let window_left = egraph.enodes[kind_children[5]]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            .round() as i64;
        let has_sinks = egraph.enodes[kind_children[6]]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            != 0.0;

        let extracted = Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            batch_dim,
            sm_scale,
            window_left,
            has_sinks,
        };

        // JIT at extract time so the ~45s nvcc cost never lands inside a
        // GA profiling trial (same rationale as FlashInferAttention).
        let _ = jit::ensure_compiled_fa3(head_dim, window_left >= 0);

        // The rule passes the FLAT gather index (proof anchor); recover the
        // compact per-token page table the kernel consumes.
        let flat_idx_node = input_enodes[3];
        let gather_idx = super::find_indptrs::try_find_compact_gather_idx(egraph, flat_idx_node)
            .expect("SinkAttention matched a gather without recoverable compact gather_idx");
        let mut final_inputs = vec![
            input_enodes[0], // q (bf16)
            input_enodes[1], // k_pool
            input_enodes[2], // v_pool
            gather_idx,      // compact kv_indices
            input_enodes[4], // qo_indptr
            input_enodes[5], // kv_indptr
        ];
        if has_sinks {
            final_inputs.push(input_enodes[6]); // sinks (f32)
        }

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, final_inputs)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for SinkAttention {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        _dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let want = if self.has_sinks { 7 } else { 6 };
        anyhow::ensure!(
            inputs.len() == want,
            "SinkAttention expects {want} inputs (has_sinks={}), got {}",
            self.has_sinks,
            inputs.len()
        );
        let buf = |n: NodeIndex, what: &str| -> anyhow::Result<DeviceBuffer> {
            buffers
                .get(&n)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("SinkAttention: missing buffer for {what}"))
        };
        let q = buf(inputs[0], "q")?;
        let k_pool = buf(inputs[1], "k_pool")?;
        let v_pool = buf(inputs[2], "v_pool")?;
        let kv_indices = buf(inputs[3], "kv_indices")?;
        let qo_indptr_buf = buf(inputs[4], "qo_indptr")?;
        let kv_indptr_buf = buf(inputs[5], "kv_indptr")?;
        let mut _neutral_hold = None;
        let sinks_ptr: u64 = if self.has_sinks {
            buf(inputs[6], "sinks")?.ptr()
        } else {
            let bytes = self.num_qo_heads * std::mem::size_of::<f32>();
            let mut guard = NEUTRAL_SINKS.lock().unwrap();
            if guard.as_ref().is_none_or(|(cap, _)| *cap < bytes) {
                let host = vec![NEUTRAL_SINK; self.num_qo_heads];
                let mut dev = stream.alloc_zeros::<u8>(bytes)?;
                stream.memcpy_htod(bytemuck::cast_slice(&host), &mut dev)?;
                *guard = Some((bytes, dev));
            }
            let ptr = {
                let (_, dev) = guard.as_ref().unwrap();
                let (p, _sync) = dev.device_ptr(stream);
                p
            };
            _neutral_hold = Some(guard);
            ptr
        };
        let out = buf(self_node, "output")?;
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        let lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
        let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = flashinfer_workspaces(stream);

        // Read the indptrs back to the host: the FA3 plan is host-side and
        // runs once per execute. The first read's synchronize also drains
        // the previous execute's async traffic out of the shared pinned plan
        // buffer and the scratch pools before this call rewrites them.
        let read_device_i32s = |b: DeviceBuffer| -> anyhow::Result<Vec<i32>> {
            let mut host_bytes = vec![0u8; b.len()];
            unsafe {
                result::memcpy_dtoh_async(&mut host_bytes, b.ptr(), stream.cu_stream())?;
            }
            stream.synchronize()?;
            Ok(bytes_to_i32_vec(host_bytes))
        };

        let plan_key: PlanKey = (
            exec_generation(),
            qo_indptr_buf.ptr(),
            kv_indptr_buf.ptr(),
            qo_indptr_buf.len(),
            kv_indptr_buf.len(),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
        );
        let cached = if plan_cache_enabled() {
            let guard = PLAN_CACHE.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_ref() {
                Some((k, c)) if *k == plan_key => Some(c.clone()),
                _ => None,
            }
        } else {
            None
        };

        let CachedPlan {
            qo_indptr,
            kv_indptr,
            mut plan_info,
            plan_info_len,
        } = match cached {
            // Same execution, same buffers, same geometry: this is the plan the
            // previous layer already built. Reusing it skips the planning AND
            // the two pipeline drains the readback needs to get its inputs.
            Some(c) => {
                if plan_verify_enabled() {
                    let qo = read_device_i32s(qo_indptr_buf)?;
                    let kv = read_device_i32s(kv_indptr_buf)?;
                    assert_eq!(qo, c.qo_indptr, "FA3 plan cache: qo_indptr diverged");
                    assert_eq!(kv, c.kv_indptr, "FA3 plan cache: kv_indptr diverged");
                    let mut klen: Vec<i32> = kv.windows(2).map(|w| w[1] - w[0]).collect();
                    let (mut qo_m, mut kv_m) = (qo.clone(), kv.clone());
                    let mut pinfo = [0i64; 16];
                    let mut plen: i32 = 0;
                    let pl = page_locked_workspace();
                    let ret = unsafe {
                        (lib.prefill_plan)(
                            float_ws_ptr as *mut std::ffi::c_void,
                            super::FLOAT_WORKSPACE_SIZE,
                            int_ws_ptr as *mut std::ffi::c_void,
                            pl.0 as *mut std::ffi::c_void,
                            INT_WORKSPACE_SIZE,
                            qo_m.as_mut_ptr(),
                            kv_m.as_mut_ptr(),
                            klen.as_mut_ptr(),
                            *qo.last().unwrap() as i32,
                            (qo.len() - 1) as i32,
                            self.num_qo_heads as i32,
                            self.num_kv_heads as i32,
                            1,
                            cu_stream,
                            pinfo.as_mut_ptr(),
                            &mut plen,
                        )
                    };
                    assert_eq!(ret, 0, "FA3 plan cache: verification plan failed");
                    assert_eq!(plen, c.plan_info_len, "FA3 plan cache: plan_info_len diverged");
                    assert_eq!(pinfo, c.plan_info, "FA3 plan cache: plan_info diverged");
                }
                c
            }
            None => {
                let mut qo_indptr = read_device_i32s(qo_indptr_buf)?;
                let mut kv_indptr = read_device_i32s(kv_indptr_buf)?;
                anyhow::ensure!(
                    qo_indptr.len() == kv_indptr.len() && qo_indptr.len() >= 2,
                    "SinkAttention: malformed indptrs (qo len {}, kv len {})",
                    qo_indptr.len(),
                    kv_indptr.len()
                );
                // page_size = 1: per-sequence kv length in tokens == pages.
                let mut kv_len_arr: Vec<i32> =
                    kv_indptr.windows(2).map(|w| w[1] - w[0]).collect();
                let batch_size = qo_indptr.len() - 1;
                let nnz_qo = *qo_indptr.last().unwrap() as usize;
                let page_locked = page_locked_workspace();
                let mut plan_info = [0i64; 16];
                let mut plan_info_len: i32 = 0;
                let plan_ret = unsafe {
                        (lib.prefill_plan)(
                            float_ws_ptr as *mut std::ffi::c_void,
                            super::FLOAT_WORKSPACE_SIZE,
                            int_ws_ptr as *mut std::ffi::c_void,
                            page_locked.0 as *mut std::ffi::c_void,
                            INT_WORKSPACE_SIZE,
                            qo_indptr.as_mut_ptr(),
                            kv_indptr.as_mut_ptr(),
                            kv_len_arr.as_mut_ptr(),
                            nnz_qo as i32,
                            batch_size as i32,
                            self.num_qo_heads as i32,
                            self.num_kv_heads as i32,
                            /*page_size=*/ 1,
                            cu_stream,
                            plan_info.as_mut_ptr(),
                        &mut plan_info_len,
                    )
                };
                anyhow::ensure!(plan_ret == 0, "SinkAttention: fa3 plan failed ({plan_ret})");
                let fresh = CachedPlan {
                    qo_indptr,
                    kv_indptr,
                    plan_info,
                    plan_info_len,
                };
                *PLAN_CACHE.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some((plan_key, fresh.clone()));
                fresh
            }
        };

        let nnz_qo = *qo_indptr.last().unwrap() as usize;
        let total_pages = *kv_indptr.last().unwrap() as usize;
        anyhow::ensure!(
            kv_indices.len() >= total_pages * std::mem::size_of::<i32>(),
            "SinkAttention: kv_indices buffer smaller than kv_indptr total"
        );

        // Kernel-native (s, heads, dim) bf16 scratch (front half: transposed
        // q in, back half: kernel out), from the grow-only pool. Reuse is
        // stream-ordered; the refresh-path sync above covers growth.
        let temp_bytes = (nnz_qo * self.num_qo_heads * self.head_dim * 2).max(1);
        let mut scratch_guard = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let stream_key = stream.cu_stream() as usize;
        let needs_new = !matches!(&*scratch_guard,
            Some((key, buf)) if *key == stream_key && buf.len() >= 2 * temp_bytes);
        if needs_new {
            stream.synchronize()?; // in-flight users of the old scratch
            let buf = unsafe { stream.alloc::<u8>((2 * temp_bytes).next_power_of_two())? };
            if let Some((_, old)) = scratch_guard.take() {
                std::mem::forget(old); // context may be gone; never free
            }
            *scratch_guard = Some((stream_key, buf));
        }
        let base_ptr = scratch_guard.as_ref().unwrap().1.device_ptr(stream).0;
        let (q_temp_ptr, temp_ptr) = (base_ptr, base_ptr + temp_bytes as u64);

        // The graph's q is (heads, s, dim) — the same heads-major layout
        // world as the output point — but the kernel reads token-major
        // (s, heads, dim) q. The layouts are byte-identical at s == 1
        // (decode), which is how this survived every single-token path;
        // prefill (s > 1) needs the transpose.
        let qtr_ret = unsafe {
            (lib.transpose_q_bf16)(
                q.ptr() as *const std::ffi::c_void,
                q_temp_ptr as *mut std::ffi::c_void,
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.head_dim as i32,
                cu_stream,
            )
        };
        anyhow::ensure!(
            qtr_ret == 0,
            "SinkAttention: q transpose failed ({qtr_ret})"
        );

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        let run_ret = unsafe {
            (lib.prefill_run)(
                int_ws_ptr as *mut std::ffi::c_void,
                plan_info.as_mut_ptr(),
                plan_info_len,
                q_temp_ptr as *mut std::ffi::c_void,
                k_pool.ptr() as *mut std::ffi::c_void,
                v_pool.ptr() as *mut std::ffi::c_void,
                kv_indices.ptr() as *mut i32,
                sinks_ptr as *mut f32,
                temp_ptr as *mut std::ffi::c_void,
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                /*page_size=*/ 1,
                sm_scale,
                self.window_left as i32,
                cu_stream,
            )
        };
        anyhow::ensure!(run_ret == 0, "SinkAttention: fa3 run failed ({run_ret})");

        let tr_ret = unsafe {
            (lib.transpose_output_f32)(
                temp_ptr as *const std::ffi::c_void,
                out.ptr() as *mut std::ffi::c_void,
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.head_dim as i32,
                cu_stream,
            )
        };
        anyhow::ensure!(tr_ret == 0, "SinkAttention: output transpose failed");

        // No trailing sync: scratch is pooled (never freed), so the enqueued
        // kernels own it under stream ordering; the next tick's plan-refresh
        // path syncs before touching shared host buffers or growing scratch.
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.batch_dim * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4 // F32 output
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("SinkAttention")
    }
}
