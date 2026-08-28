//! Shared support for the per-model example applications (Austin's
//! spec: "in the CL runtime crate, in the examples folder, there should
//! be a little application that runs each model").
//!
//! Cargo idiom for shared example code: `examples/support/mod.rs` is
//! not itself an example — auto-discovery only picks up `examples/*.rs`
//! and `examples/*/main.rs` — and each example pulls it in with
//! `mod support;`.
//!
//! The differential discipline (house doctrine): identical seeded
//! synthetic inputs on BOTH runtimes — reference host run first for the
//! expected outputs, then CUDA-lite record → search → execute → fetch
//! through the disclosed layout — and a loud bail on any divergence.
#![allow(dead_code)] // each example compiles this module independently and uses a subset

/// Deterministic pseudo-random values — the seeding discipline copied
/// VERBATIM from the mini measure harnesses
/// (`examples/mini/*/src/bin/measure_plan.rs`): same `(n, seed)` gives
/// the same values on both runtimes, so the differential is exact.
pub fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6).collect()
}

/// The stub path for builds WITHOUT the `device` feature: the CUDA-lite
/// crate can load/search/inspect plans anywhere, but `execute` refuses
/// without a CUDA device, so the differential cannot run here.
pub fn require_device(example: &str) {
    println!(
        "{example}: SKIP — this example requires the `device` feature (and a CUDA device).\n\
         Re-run with: cargo run -p luminal_cuda_lite --example {example} --features device"
    );
}

#[cfg(feature = "device")]
pub mod device {
    use anyhow::{anyhow, bail, Context, Result};
    use luminal::buffer_tensor_ir::TypedBuffer;
    use luminal::bufferize::{walk_layout_index, BufferNode};
    use luminal::graph::Graph;
    use luminal::prelude::{FxHashMap, NodeIndex};
    use luminal_cuda_lite::CudaRuntime;
    use luminal_reference::ReferenceRuntime;

    /// Reference host run: the same `load → bind dyn pins → search →
    /// set_data → execute` ladder as the mini measure harnesses
    /// (`examples/mini/*/src/bin/measure_plan.rs`) and
    /// `luminal_reference::harness::run_reference`, on the shared
    /// harness budget.
    fn run_reference(
        cx: &Graph,
        pairs: &[(NodeIndex, TypedBuffer)],
    ) -> Result<ReferenceRuntime> {
        let mut rt = ReferenceRuntime::load(cx).context("reference load")?;
        let mut vars: Vec<_> = cx.dyn_map.iter().collect();
        vars.sort();
        for (var, value) in vars {
            rt.bind_dyn_range(*var, *value as u64, *value as u64)
                .context("reference dyn pin")?;
        }
        let data: FxHashMap<NodeIndex, TypedBuffer> = pairs.iter().cloned().collect();
        rt.search(&data, &luminal::test_support::harness_search_options())
            .context("reference search")?;
        for (id, v) in pairs {
            rt.set_data(*id, v.clone());
        }
        rt.execute().context("reference execute")?;
        Ok(rt)
    }

    /// Read a device output DENSELY through its disclosed layout — the
    /// escape-and-disclose readback copied from
    /// `crates/luminal_cuda_lite/tests/device_fidelity.rs::walked_dense`:
    /// a view-elected output returns its BACKING buffer's bytes (possibly
    /// parent-sized) plus the elected layout, so the honest comparison
    /// walks each element `[i0, i1, ...]` through the hop chain —
    /// `luminal::bufferize::walk_layout_index` is the trusted reader.
    /// A dense election walks the identity, so this is the universal
    /// readback.
    fn walked_dense(rt: &CudaRuntime, out: NodeIndex) -> Result<Vec<f32>> {
        let (data, binding) = rt.fetch(out).context("escape-and-disclose fetch")?;
        let bytes = match data {
            TypedBuffer::F32(values) => values,
            other => bail!("output is {}, not f32", other.type_name()),
        };
        let dims = binding
            .dims
            .clone()
            .ok_or_else(|| anyhow!("symbolic output dims — numeric readback refuses"))?;
        let base_dims = rt
            .plan()
            .ok_or_else(|| anyhow!("plan not loaded"))?
            .buffers
            .get(&binding.buffer)
            .and_then(|record| record.dims.clone())
            .ok_or_else(|| anyhow!("backing buffer has no numeric geometry"))?;
        let numel: usize = dims.iter().map(|&d| d as usize).product();
        let rank = dims.len();
        let mut dense = Vec::with_capacity(numel);
        let mut coords = vec![0usize; rank];
        for _ in 0..numel {
            let flat = walk_layout_index(
                binding.composed_access.as_ref(),
                &dims,
                &base_dims,
                &coords,
            )
            .context("the walker reads the disclosed layout")?;
            dense.push(bytes[flat]);
            for axis in (0..rank).rev() {
                coords[axis] += 1;
                if coords[axis] < dims[axis] as usize {
                    break;
                }
                coords[axis] = 0;
            }
        }
        Ok(dense)
    }

    /// Elementwise comparison at the device_fidelity epsilon
    /// (`tests/device_fidelity.rs::assert_close`):
    /// `tol = 1e-5.max(|reference| * 1e-5)` — relative 1e-5 with an
    /// absolute 1e-5 floor. Loud on the first divergent element. The
    /// bail predicate is the NEGATED must-hold condition `!(diff <= tol)`
    /// — assert_close's `assert!(diff <= tol)` verbatim — so a NaN
    /// device element (diff = NaN, for which `diff > tol` is FALSE)
    /// bails loudly instead of sailing through; every accepted element
    /// then satisfies `diff <= tol`, so the max_abs fold never sees NaN.
    // Clippy's suggested `diff > tol` is EXACTLY the NaN-silent bug this
    // predicate fixes — incomparability (NaN) must bail, not pass.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn compare(want: &[f32], got: &[f32], what: &str) -> Result<f32> {
        if want.len() != got.len() {
            bail!(
                "{what}: length mismatch — reference {} vs device {}",
                want.len(),
                got.len()
            );
        }
        let mut max_abs = 0f32;
        for (i, (w, g)) in want.iter().zip(got).enumerate() {
            let tol = 1e-5f32.max(w.abs() * 1e-5);
            let diff = (w - g).abs();
            if !(diff <= tol) {
                bail!(
                    "{what}: element {i} diverges — reference {w} vs device {g} \
                     (|delta| {diff:.3e} !<= tol {tol:.3e})"
                );
            }
            max_abs = max_abs.max(diff);
        }
        Ok(max_abs)
    }

    /// Plan statistics: kernel launches (Compute nodes), whole-buffer
    /// copies (BufferCopy nodes), distinct buffers, and output slots
    /// split into direct vs escaped (view-elected: `composed_access`
    /// disclosed, backing buffer escapes to the caller).
    struct PlanStats {
        kernels: usize,
        copies: usize,
        buffers: usize,
        outputs: usize,
        escaped: usize,
    }

    fn plan_stats(rt: &CudaRuntime) -> Result<PlanStats> {
        let plan = rt.plan().ok_or_else(|| anyhow!("plan not loaded"))?;
        let mut stats = PlanStats {
            kernels: 0,
            copies: 0,
            buffers: plan.buffers.len(),
            outputs: 0,
            escaped: 0,
        };
        for idx in plan.dag.node_indices() {
            match &plan.dag[idx] {
                BufferNode::BufferInput { .. } => {}
                BufferNode::Compute { .. } => stats.kernels += 1,
                BufferNode::BufferCopy { .. } => stats.copies += 1,
                BufferNode::BufferOutput { slots } => {
                    for slot in slots {
                        stats.outputs += 1;
                        if slot.composed_access.is_some() {
                            stats.escaped += 1;
                        }
                    }
                }
            }
        }
        Ok(stats)
    }

    /// The whole differential, shared by every runnable example:
    ///
    /// 1. reference host run (expected outputs),
    /// 2. CUDA-lite `load → bind dyn pins → search` on the SAME harness
    ///    budget (`luminal::test_support::harness_search_options` — the
    ///    budget device_fidelity and the mini measure harnesses use),
    /// 3. plan stats + refusal counters (all zero expected — the ladder
    ///    acceptance from `tests/ladder_refusals.rs`; nonzero FAILS),
    /// 4. device execute, fetch through the disclosed layout, compare
    ///    at the device_fidelity epsilon.
    pub fn run_differential(
        name: &str,
        cx: &Graph,
        pairs: &[(NodeIndex, TypedBuffer)],
        outputs: &[(&str, NodeIndex)],
    ) -> Result<()> {
        // 1. Reference run first — the expected outputs.
        let t = std::time::Instant::now();
        let reference = run_reference(cx, pairs).context("reference half")?;
        let mut expected = Vec::new();
        for (label, id) in outputs {
            expected.push(reference.get_f32(*id).with_context(|| format!("reference {label}"))?.clone());
        }
        println!("{name}: reference OK ({} ms)", t.elapsed().as_millis());

        // 2. CUDA-lite: record → search (harness budget) → plan.
        let mut rt = CudaRuntime::load(cx).context("cuda load")?;
        let mut vars: Vec<_> = cx.dyn_map.iter().collect();
        vars.sort();
        for (var, value) in vars {
            rt.bind_dyn_range(*var, *value as u64, *value as u64)
                .context("cuda dyn pin")?;
        }
        let data: FxHashMap<NodeIndex, TypedBuffer> = pairs.iter().cloned().collect();
        let t = std::time::Instant::now();
        let outcome = rt
            .search(&data, &luminal::test_support::harness_search_options())
            .context("cuda search")?;
        let search_ms = t.elapsed().as_millis();
        println!(
            "{name}: search {search_ms} ms | plans profiled {} | [{}]",
            outcome.plans_profiled,
            outcome.timings.summary()
        );

        // 3. Refusal counters — all zero expected (ladder acceptance).
        let b = &outcome.refusal_breakdown;
        println!("{name}: refusals {}", b.summary());
        if b.extract_refusals != 0 || b.plan_build_refusals != 0 || b.execute_refusals != 0 {
            bail!(
                "nonzero search refusals — the ladder expects zero with views admitted: {}",
                b.summary()
            );
        }
        let stats = plan_stats(&rt)?;
        println!(
            "{name}: plan kernels={} copies={} buffers={} outputs={} escaped={}",
            stats.kernels, stats.copies, stats.buffers, stats.outputs, stats.escaped
        );

        // 4. Execute on device; fetch through the disclosed layout.
        for (id, v) in pairs {
            rt.set_data(*id, v.clone());
        }
        let t = std::time::Instant::now();
        rt.execute().context("device execute")?;
        let execute_ms = t.elapsed().as_millis();
        println!("{name}: execute {execute_ms} ms");

        for ((label, id), want) in outputs.iter().zip(&expected) {
            let got = walked_dense(&rt, *id).with_context(|| format!("device {label}"))?;
            let max_abs = compare(want, &got, label)?;
            println!(
                "{name}: {label} matches reference ({} elements, max |delta| {max_abs:.3e})",
                want.len()
            );
        }
        println!("{name}: PASS (search {search_ms} ms, execute {execute_ms} ms)");
        Ok(())
    }
}
