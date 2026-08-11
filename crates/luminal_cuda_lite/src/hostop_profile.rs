//! Per-host-op timing, enabled by `LUMINAL_HOSTOP_PROFILE=1`.
//!
//! `EXEC_PROFILE`'s `host_op_call_ms` is one number for every host op in the
//! graph. For a 36-layer MoE model that is 72 calls of two very different
//! kinds, and the aggregate cannot say which kind owns the time — nor whether
//! the time is kernel work, host-side planning, or a `stream.synchronize()`
//! draining the pipeline mid-graph.
//!
//! Ops register named sub-phases with [`phase`], so a single run answers both
//! "which op" and "which part of it".
//!
//! Off by default; one relaxed atomic load per call when off.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: std::sync::Once = std::sync::Once::new();

/// name -> (calls, total). BTreeMap so the report order is stable between runs.
static TOTALS: Mutex<BTreeMap<&'static str, (u64, Duration)>> = Mutex::new(BTreeMap::new());

/// `LUMINAL_HOSTOP_PROFILE=sync`: synchronize after EVERY exec op and charge
/// the elapsed time to it.
///
/// Necessary because the default accounting measures CPU enqueue time, and
/// once the FA3 plan cache removed the mid-graph drains, essentially all GPU
/// work lands in one sync at the end of `execute` — a single number that says
/// nothing about which op produced it. Serializing distorts the total (it
/// forbids overlap) but attributes correctly, which is what a "where does the
/// time go" question needs. Cost is one sync per op, ~181 per tick here,
/// negligible against a 640 ms prefill tick.
pub fn sync_mode() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LUMINAL_HOSTOP_PROFILE").is_ok_and(|v| v == "sync")
    })
}

/// The slowest individual ops seen this execution: (duration, name, index).
/// A per-name total hides the case where one of 109 identically-named CUDA
/// graph launches owns the whole tick.
static SLOWEST: Mutex<Vec<(Duration, &'static str, usize)>> = Mutex::new(Vec::new());

pub fn record_indexed(name: &'static str, idx: usize, d: Duration) {
    if !enabled() {
        return;
    }
    record(name, d);
    let mut g = SLOWEST.lock().unwrap_or_else(|e| e.into_inner());
    g.push((d, name, idx));
    if g.len() > 512 {
        g.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        g.truncate(16);
    }
}

pub fn enabled() -> bool {
    INIT.call_once(|| {
        ENABLED.store(
            std::env::var_os("LUMINAL_HOSTOP_PROFILE").is_some_and(|v| v != "0"),
            Relaxed,
        );
    });
    ENABLED.load(Relaxed)
}

pub fn record(name: &'static str, d: Duration) {
    if !enabled() {
        return;
    }
    let mut g = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let e = g.entry(name).or_insert((0, Duration::ZERO));
    e.0 += 1;
    e.1 += d;
}

/// Time a sub-phase inside a host op. `name` must be a literal so the report
/// key is stable.
pub fn phase<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    record(name, t.elapsed());
    out
}

/// Print and clear. Called once per execute by the runtime, so each line is
/// "per graph execution" — for a decode tick that is per token-batch.
pub fn report_and_reset() {
    if !enabled() {
        return;
    }
    let mut g = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_empty() {
        return;
    }
    let total: Duration = g
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(_, (_, d))| *d)
        .sum();
    eprintln!("HOSTOP_PROFILE total_ms={:.3}", total.as_secs_f64() * 1e3);
    for (name, (calls, d)) in g.iter() {
        eprintln!(
            "  {:<34} calls={:<4} total_ms={:>8.3} per_call_us={:>8.1}",
            name,
            calls,
            d.as_secs_f64() * 1e3,
            d.as_secs_f64() * 1e6 / *calls as f64,
        );
    }
    g.clear();
    drop(g);

    let mut sl = SLOWEST.lock().unwrap_or_else(|e| e.into_inner());
    if !sl.is_empty() {
        sl.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        eprintln!("  --- slowest individual ops ---");
        for (d, name, idx) in sl.iter().take(10) {
            eprintln!("  #{idx:<5} {name:<28} {:>9.3} ms", d.as_secs_f64() * 1e3);
        }
        sl.clear();
    }
}

/// Monotonic counter bumped once per `execute()`.
///
/// Host ops use it to cache work that is invariant *within* one graph
/// execution but must not leak across executions — the FA3 indptr readback and
/// plan being the motivating case: every layer's attention op reads the same
/// two index buffers and builds the same plan, but their contents change from
/// one tick to the next.
pub static GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn bump_generation() -> u64 {
    GENERATION.fetch_add(1, Relaxed) + 1
}

pub fn generation() -> u64 {
    GENERATION.load(Relaxed)
}
