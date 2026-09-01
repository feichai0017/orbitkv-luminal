//! Per-plan proofs for the exemplar's wiring — no HF download, seconds
//! to run. The anatomy itself (QK-norm, rope threading, positional
//! mask, paged cache) is validated against scalar references in
//! luminal_nn; these tests prove the CRATE's loop: build → search once
//! → execute per token, cache state flowing runtime-out → runtime-in.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::TypedBuffer;
use qwen::model::QwenDims;
use qwen::{DecodeStep, Decoder, weights};

fn smoke_search() -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 2,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
    }
}

/// Three decode steps at tiny dims: logits stay finite, the cache write
/// frontier advances one row per step (rows beyond the frontier stay
/// zero), and a second identical decoder reproduces the logits exactly.
#[test]
fn tiny_decode_loop_is_deterministic_and_advances_the_cache() {
    let dims = QwenDims::tiny();
    let max_seq = 4usize;
    let kv_dim = dims.kv_dim();

    let run = || {
        let step = DecodeStep::build(&dims, max_seq);
        let pairs = weights::random_weights(&step.cx);
        let mut decoder = Decoder::start(step, &pairs, &smoke_search()).expect("search");
        let mut rows = Vec::new();
        for token in [1u32, 2, 0] {
            rows.push(decoder.step(token).expect("step"));
        }
        (rows, decoder)
    };

    let (rows, decoder) = run();
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), dims.vocab, "logits row {index}");
        assert!(
            row.iter().all(|v| v.is_finite()),
            "non-finite logits at step {index}"
        );
    }
    // After 3 steps the frontier sits at row 3: rows 0..2 written, row 3 zero.
    for (layer, k_state) in decoder.k_state.iter().enumerate() {
        for row in 0..max_seq {
            let slice = &k_state[row * kv_dim..(row + 1) * kv_dim];
            let written = slice.iter().any(|v| *v != 0.0);
            if row < 3 {
                assert!(written, "layer {layer} cache row {row} never written");
            } else {
                assert!(!written, "layer {layer} cache row {row} written early");
            }
        }
    }

    let (rows_again, _) = run();
    assert_eq!(rows, rows_again, "decode loop is not deterministic");
}

/// Vocab-scale gather pin: coordinate-form `gather_rows` across what
/// used to be the 2^24 flat-exactness boundary — table (300_000, 64)
/// with row-id payloads; any index corruption under any searched plan
/// reads a neighbouring row and mismatches loudly. (The smuggling this
/// originally guarded against is gone — typed buffers — but the pin
/// stays: it exercises the primary gather spelling at scale.)
#[test]
fn embedding_scale_row_gather_stays_exact() {
    const ROWS: usize = 300_000;
    const D: usize = 64;
    let picks: Vec<usize> = vec![0, 1, 262_143, 262_144, 262_145, 299_998, 299_999];

    let mut cx = Graph::new();
    let table = cx.tensor((ROWS, D));
    let idx = cx.tensor_dtyped(picks.len(), DType::Int);
    let rows = luminal_nn::gather_rows(table, idx, D).output();

    // data[r, c] = r — every element of a fetched row must equal the
    // requested row id exactly (row ids < 2^24 are f32-exact).
    let mut data = vec![0f32; ROWS * D];
    for (r, chunk) in data.chunks_mut(D).enumerate() {
        chunk.fill(r as f32);
    }
    let idx_vals: Vec<i32> = picks.iter().map(|r| *r as i32).collect();

    let rt = luminal_reference::harness::run_reference(
        &cx,
        &[(table.id, data.into()), (idx.id, idx_vals.into())],
    );
    let out = rt.get_f32(rows.id).expect("gathered rows");
    assert_eq!(out.len(), picks.len() * D);
    for (which, row) in picks.iter().enumerate() {
        for c in 0..D {
            assert_eq!(
                out[which * D + c],
                *row as f32,
                "row {row} column {c} read a neighbouring row"
            );
        }
    }
}

/// THE RESURRECTED PROBE (typed-buffers acceptance, 2026-08-11): this
/// exact graph — a MATERIALIZED flat index idx·2560 + col at Qwen3-4B
/// vocab magnitude (3.9e8) with the base subtracted — is the probe
/// that exposed the Int-in-f32 smuggling: under f32 storage it failed
/// nondeterministically (plan-dependent rounding to multiples of 32).
/// Under typed buffers the arithmetic runs in native i32 (checked,
/// non-wrapping) and the residuals are exact under EVERY plan. The
/// probe that found the bug is the proof of the fix.
#[test]
fn vocab_scale_flat_index_arithmetic_stays_exact() {
    const HIDDEN: usize = 2560;
    const LAST_ROW: usize = 151_935; // Qwen3-4B vocab − 1
    let base = (LAST_ROW * HIDDEN) as i64; // 388,953,600 — fits i32

    let mut cx = Graph::new();
    let idx = cx.tensor_dtyped(1, DType::Int);
    let flat = (idx * HIDDEN).expand_dim(1, HIDDEN) + cx.arange(HIDDEN as i32).expand_dim(0, 1);
    let base_t = cx.constant(base).expand_dim(0, 1).expand_dim(1, HIDDEN);
    let residual = (flat - base_t).output();

    // Landing D: the flat products are provable because the caller
    // DECLARES the index range — the proof chain runs idx ∈ [0, vocab)
    // through the interval rules to "every partial sum fits i32."
    let mut rt = luminal_reference::ReferenceRuntime::load(&cx).expect("native load");
    rt.bind_value_range(idx.id, 0, LAST_ROW as i64)
        .expect("range binds");
    let mut data = luminal::prelude::FxHashMap::default();
    data.insert(idx.id, TypedBuffer::from(vec![LAST_ROW as i32]));
    rt.search(
        &data,
        &luminal::implementation_search::ImplementationSearchOptions {
            generations: 2,
            generation_size: 4,
            mutations: 2,
            trials: 1,
            seed: 0,
        },
    )
    .expect("proven flat-index arithmetic implements");
    rt.set_data(idx.id, vec![LAST_ROW as i32]);
    rt.execute().expect("executes");
    let out = rt.get_i32(residual.id).expect("residual");
    assert_eq!(out.len(), HIDDEN);
    for (column, value) in out.iter().enumerate() {
        assert_eq!(
            *value, column as i32,
            "flat index arithmetic lost exactness at column {column}"
        );
    }
}
