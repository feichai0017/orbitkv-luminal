//! Per-plan proofs for the exemplar's wiring — no HF download, seconds
//! to run. The anatomy itself (QK-norm, rope threading, positional
//! mask, paged cache) is validated against scalar references in
//! luminal_nn; these tests prove the CRATE's loop: build → search once
//! → execute per token, cache state flowing runtime-out → runtime-in.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
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
        let pairs = weights::random_weights(&step.model);
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

/// Vocab-scale gather exactness: the reference runtime stores Int
/// buffer VALUES in f32 (exact only below 2^24), so a MATERIALIZED
/// flat index (row·D + col) at embedding scale rounds — and whether it
/// materializes is a plan decision, which made the flat-sugar
/// `gather_rows` fail nondeterministically at Qwen3-4B magnitudes
/// (found by this crate's original probe, 2026-08-11). `gather_rows`
/// is now COORDINATE-form: per-axis coordinates stay below their axis
/// extents and never overflow. This test drives it across the 2^24
/// flat boundary — table (300_000, 64), flat range up to 1.92e7 — with
/// row-id payloads: any index rounding under any searched plan reads a
/// neighbouring row and mismatches loudly.
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
    let idx_vals: Vec<f32> = picks.iter().map(|r| *r as f32).collect();

    let rt = luminal::test_support::run_ssa(
        &cx,
        &[(table.id, data), (idx.id, idx_vals)],
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
