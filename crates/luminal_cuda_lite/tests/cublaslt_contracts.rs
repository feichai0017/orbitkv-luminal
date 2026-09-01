//! Train 3, Items 4+5 — the DEVICE-GATED cuBLASLt contract tests:
//! compiled under `--features device` on any host (the compile check is
//! this train's gate) and EXECUTED by the orchestrator's A100 pass.
//!
//! Coverage:
//!  * the four contract forms on real device buffers (direct
//!    `device_call::dispatch` over hand-built `LtCall`s, ROW
//!    convention): DEFAULT-epilogue forms execute green; BIAS-epilogue
//!    forms pin the loud refusal (the measured library restriction —
//!    no BIAS/RELU_BIAS on a ROW-order D);
//!  * the TF32 strictness detector assertion (contract 5);
//!  * a deliberate ld-bounds violation refused loudly BEFORE dispatch
//!    (contract 4 — including the rows==1 case the library's own check
//!    is vacuous on);
//!  * NUMERICS POLICY (Item 4): the marker-elected plan compares
//!    against the decomposed route TOLERANCE-based only — see
//!    `assert_close` for the reduction-order contract.
#![cfg(feature = "device")]

use cudarc::driver::{CudaContext, CudaSlice};
use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::BufferNode;
use luminal::prelude::{FxHashMap, NodeIndex};

/// The universal escape-and-disclose readback (the device_fidelity
/// pattern): fetch the backing bytes + binding, walk each output
/// element through the disclosed layout. Dense elections walk the
/// identity, view elections the composed chain.
fn walked_dense(rt: &CudaRuntime, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt.fetch(out).expect("escape-and-disclose fetch");
    let bytes = match data {
        TypedBuffer::F32(values) => values,
        other => panic!("output is {}, not f32", other.type_name()),
    };
    // The value's shape and read path both come from the RETURNED
    // LAYOUT; there is no `dims` field and no hop chain any more.
    luminal_cuda_lite::layouts::dense_f32(bytes, &binding.layout)
        .expect("the returned layout reads dense over its backing buffer")
}
use luminal_cuda_lite::ops::cublaslt::device_call;
use luminal_cuda_lite::ops::cublaslt::exec::{CSource, LtCall, LtDesc};
use luminal_cuda_lite::ops::cublaslt::CublasLtForm;
use luminal_cuda_lite::CudaRuntime;
use std::sync::Arc;

/// REDUCTION-ORDER CONTRACT (Item 4, documented at the comparison
/// site): bit-exactness between a vendor GEMM and the decomposed
/// mul+reduce route is IMPOSSIBLE IN PRINCIPLE — cublasLtMatmul's
/// reduction order is algorithm-dependent and unspecified (split-k,
/// tiling, FMA contraction), while the decomposed route reduces in
/// linear axis order. Marker-elected results are therefore compared
/// TOLERANCE-based at the device_fidelity epsilon
/// (`tol = 1e-5.max(|reference| * 1e-5)`), and NOTHING in this tree
/// may claim or test bit-equality against the decomposed route.
/// NaN is incomparable and bails (the negated-predicate idiom).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn assert_close(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let tol = 1e-5f32.max(w.abs() * 1e-5);
        let diff = (w - g).abs();
        assert!(
            diff <= tol,
            "{what}: element {i} diverges — expected {w}, got {g} (|delta| {diff:.3e} > tol {tol:.3e})"
        );
    }
}

/// Deterministic values (the shared example seeding discipline).
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// Host reference for one call: ROW-order walk (the bridge's ROW
/// convention — every descriptor is CUBLASLT_ORDER_ROW, ld = row
/// pitch) of D = act(op(A)op(B) + beta*C + bias), alpha = 1 (the
/// fixed literal).
fn host_reference(
    call: &LtCall,
    a: &[f32],
    b: &[f32],
    c: Option<&[f32]>,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let (m, n, k) = (call.m as usize, call.n as usize, call.k as usize);
    let mut d = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                // ROW storage with the descriptor's ld; op applied.
                let a_v = if call.trans_a {
                    a[kk * call.a.ld as usize + row] // A' is k x m
                } else {
                    a[row * call.a.ld as usize + kk] // A' is m x k
                };
                let b_v = if call.trans_b {
                    b[col * call.b.ld as usize + kk] // B' is n x k
                } else {
                    b[kk * call.b.ld as usize + col] // B' is k x n
                };
                acc += (a_v as f64) * (b_v as f64);
            }
            if let Some(c) = c {
                acc += c[row * call.c.ld as usize + col] as f64;
            }
            if let Some(bias) = bias {
                acc += bias[row] as f64;
            }
            let mut v = acc as f32;
            if call.relu {
                v = v.max(0.0);
            }
            d[row * call.d.ld as usize + col] = v;
        }
    }
    d
}

fn to_device(stream: &Arc<cudarc::driver::CudaStream>, host: &[f32]) -> CudaSlice<u8> {
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut slice = stream.alloc_zeros::<u8>(bytes.len().max(1)).expect("alloc");
    stream.memcpy_htod(&bytes, &mut slice).expect("H2D");
    slice
}

fn from_device(stream: &Arc<cudarc::driver::CudaStream>, slice: &CudaSlice<u8>) -> Vec<f32> {
    let mut host = vec![0u8; slice.len()];
    stream.memcpy_dtoh(slice, &mut host).expect("D2H");
    host.chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// Build the canonical contiguous call for one form (m=3, n=4, k=5) —
/// the bridge's ROW convention: dense row-major operands, ld = the
/// row pitch (= cols).
fn call_for(form: CublasLtForm) -> LtCall {
    let (m, n, k) = (3i64, 4i64, 5i64);
    LtCall {
        form,
        m,
        n,
        k,
        trans_a: false,
        trans_b: false,
        a: LtDesc::row(m, k, k),
        b: LtDesc::row(k, n, n),
        c: LtDesc::row(m, n, n),
        d: LtDesc::row(m, n, n),
        c_source: if form.has_c() {
            CSource::Operand(2)
        } else {
            CSource::AliasD
        },
        beta_is_one: form.has_c(),
        relu: false,
        bias_operand: form.has_bias().then(|| if form.has_c() { 3 } else { 2 }),
    }
}

/// Contract 5: the TF32 strictness detector runs (once) at handle
/// creation and must be green on the A100 (strict FP32, no TF32
/// fallback in effect).
#[test]
fn tf32_strictness_detector_is_green() {
    device_call::assert_compute_strictness().expect(
        "strict CUBLAS_COMPUTE_32F must be in effect (TF32 is graph-modeled, never a flag)",
    );
}

/// The four contract forms, each under the bridge's ROW convention:
/// the DEFAULT-epilogue forms (Base, Accumulate) execute green on real
/// buffers, compared tolerance-based against the host walk (see
/// `assert_close`'s reduction-order contract); the BIAS-epilogue forms
/// (Bias, AccumulateBias) are refused LOUDLY before dispatch — the
/// MEASURED A100 finding (2026-08-28 probe): the library returns
/// CUBLAS_STATUS_NOT_SUPPORTED for BIAS/RELU_BIAS whenever D is
/// CUBLASLT_ORDER_ROW (any A/B order), and the API's per-D-row bias
/// cannot express the marker's sibling-frame bias through a COL
/// re-description of the row-major destination. No bytes may move on a
/// refused form.
#[test]
fn all_four_contract_forms_execute_green() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    for form in CublasLtForm::ALL {
        let call = call_for(form);
        let (m, n, k) = (call.m as usize, call.n as usize, call.k as usize);
        let a = weights(m * k, 1);
        let b = weights(k * n, 2);
        let c = form.has_c().then(|| weights(m * n, 3));
        let bias = form.has_bias().then(|| weights(m, 4));

        let dev_a = to_device(&stream, &a);
        let dev_b = to_device(&stream, &b);
        let mut operands: Vec<&CudaSlice<u8>> = vec![&dev_a, &dev_b];
        let dev_c = c.as_ref().map(|c| to_device(&stream, c));
        let dev_bias = bias.as_ref().map(|v| to_device(&stream, v));
        if let Some(dc) = dev_c.as_ref() {
            operands.push(dc);
        }
        if let Some(db) = dev_bias.as_ref() {
            operands.push(db);
        }
        let mut dest = stream.alloc_zeros::<u8>(m * n * 4).expect("dest alloc");

        if form.has_bias() {
            let err = device_call::dispatch(&call, &operands, &mut dest, &stream)
                .expect_err("bias-epilogue forms must be refused under the ROW convention");
            let msg = format!("{err:#}");
            assert!(msg.contains("refused BEFORE dispatch"), "{form:?}: {msg}");
            assert!(
                msg.contains("ROW-order D"),
                "{form:?} refusal must name the finding: {msg}"
            );
            stream.synchronize().expect("sync");
            assert!(
                from_device(&stream, &dest).iter().all(|&v| v == 0.0),
                "{form:?}: no bytes may move on a refused dispatch"
            );
            continue;
        }

        device_call::dispatch(&call, &operands, &mut dest, &stream)
            .unwrap_or_else(|e| panic!("{form:?} dispatch: {e:#}"));
        stream.synchronize().expect("sync");

        let got = from_device(&stream, &dest);
        let want = host_reference(&call, &a, &b, c.as_deref(), bias.as_deref());
        assert_close(&want, &got, &format!("{form:?}"));
    }
}

/// Contract 4: a deliberate ld-bounds violation is refused loudly
/// BEFORE dispatch — including the rows==1 shape whose ld the library
/// itself would happily accept (its check is vacuous there). The
/// destination stays all-zero: no bytes moved.
#[test]
fn ld_bounds_violation_refuses_before_dispatch() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    // rows==1 (the shape whose ld the library never dereferences in
    // ROW order): D is 1x8 — needs 8 elements; give it 4.
    let call = LtCall {
        form: CublasLtForm::Base,
        m: 1,
        n: 8,
        k: 2,
        trans_a: false,
        trans_b: false,
        a: LtDesc::row(1, 2, 2),
        b: LtDesc::row(2, 8, 8),
        c: LtDesc::row(1, 8, 8),
        d: LtDesc::row(1, 8, 8),
        c_source: CSource::AliasD,
        beta_is_one: false,
        relu: false,
        bias_operand: None,
    };
    let dev_a = to_device(&stream, &weights(2, 1));
    let dev_b = to_device(&stream, &weights(16, 2));
    let mut dest = stream.alloc_zeros::<u8>(4 * 4).expect("short dest"); // 4 f32s, needs 8
    let err = device_call::dispatch(&call, &[&dev_a, &dev_b], &mut dest, &stream)
        .expect_err("the short D buffer must be refused BEFORE dispatch");
    let msg = format!("{err:#}");
    assert!(msg.contains("refused BEFORE dispatch"), "{msg}");
    stream.synchronize().expect("sync");
    assert!(
        from_device(&stream, &dest).iter().all(|&v| v == 0.0),
        "no bytes may move on a refused dispatch"
    );
}

/// Item 4 END TO END: the marker-elected plan (searched with the
/// marker vocabulary, executed through the host-call arm) against the
/// decomposed route (default vocabulary, NVRTC kernels), tolerance-based
/// per the reduction-order contract in `assert_close`.
#[test]
fn marker_elected_plan_matches_decomposed_route_tolerance_based() {
    let build = || {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor((4usize, 8usize));
        let b = cx.tensor((8usize, 3usize));
        let out = a.matmul(b).output();
        (cx, a, b, out)
    };
    let data_for = |a: NodeIndex, b: NodeIndex| -> FxHashMap<NodeIndex, TypedBuffer> {
        [
            (a, TypedBuffer::from(weights(32, 1))),
            (b, TypedBuffer::from(weights(24, 2))),
        ]
        .into_iter()
        .collect()
    };
    // The seeded budget the CPU election pin measured green (see
    // tests/cublaslt_election.rs).
    let options = luminal::implementation_search::ImplementationSearchOptions {
        generations: 12,
        generation_size: 16,
        mutations: 4,
        trials: 1,
        seed: 0,
    };

    // Marker-elected route.
    let (cx, a, b, out) = build();
    let mut fused = CudaRuntime::load_with_cublaslt(&cx).expect("load fused");
    let data = data_for(a.id, b.id);
    fused.search(&data, &options).expect("fused search");
    let elected =
        fused.plan().expect("plan").dag.node_weights().any(
            |n| matches!(n, BufferNode::Compute { op, .. } if op.label().starts_with("CublasLt")),
        );
    assert!(
        elected,
        "the fused route must actually elect the marker for this comparison"
    );
    fused.set_data(a.id, weights(32, 1));
    fused.set_data(b.id, weights(24, 2));
    fused.execute().expect("fused execute");
    // The marker-elected output is the sandwich's sibling VIEW — it
    // escapes with a composed layout, so the honest readback walks the
    // disclosed layout (get_f32 refuses non-row-major backings by design).
    let got = walked_dense(&fused, out.id);

    // Decomposed route (default vocabulary — no marker in the assembly).
    let (cx, a, b, out) = build();
    let mut plain = CudaRuntime::load(&cx).expect("load plain");
    let data = data_for(a.id, b.id);
    plain
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("plain search");
    plain.set_data(a.id, weights(32, 1));
    plain.set_data(b.id, weights(24, 2));
    plain.execute().expect("plain execute");
    let want = walked_dense(&plain, out.id);

    // TOLERANCE-BASED, never bit-equality (reduction-order contract).
    assert_close(&want, &got, "marker vs decomposed 4x8x3");
}
