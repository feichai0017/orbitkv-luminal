//! Offline per-plan proofs for the qwen3_moe crate loop (no HF
//! download) — the tiny dims keep 4 experts top-2 routed per step:
//! the anatomy is validated against scalar references in luminal_nn;
//! these prove the crate's decode loop over the position-slots driver.

use qwen3_moe::model::Qwen3MoeDims;
use qwen3_moe::{DecodeStep, Decoder, weights};
use luminal::implementation_search::ImplementationSearchOptions;

fn smoke_search() -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 2,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
    }
}

/// Three decode steps at tiny dims: finite logits, the cache write
/// frontier advances via the PositionSlots driver, bitwise-identical
/// reruns.
#[test]
fn tiny_decode_loop_is_deterministic_and_advances_the_cache() {
    let dims = Qwen3MoeDims::tiny();
    let max_seq = 4usize;
    let kv_dim = dims.kv_dim();

    let run = || {
        let step = DecodeStep::build(&dims, max_seq);
        let pairs = weights::random_weights(&step);
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
    for (layer, k_state) in decoder.state.k.iter().enumerate() {
        for slot in 0..max_seq {
            let written = k_state[slot * kv_dim..(slot + 1) * kv_dim]
                .iter()
                .any(|v| *v != 0.0);
            assert_eq!(
                written,
                slot < 3,
                "layer {layer} slot {slot} write-frontier mismatch"
            );
        }
    }

    let (rows_again, _) = run();
    assert_eq!(rows, rows_again, "decode loop is not deterministic");
}
