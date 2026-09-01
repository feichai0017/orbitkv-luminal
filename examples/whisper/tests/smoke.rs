//! Offline proofs for the whisper crate loop: encoder + decoder in one
//! graph at tiny dims — forced-token steps then greedy — with the
//! decoder cache frontier advancing and bitwise-deterministic reruns.

use luminal::implementation_search::ImplementationSearchOptions;
use whisper::model::WhisperDims;
use whisper::{TranscribeStep, Transcriber, weights};

fn smoke_search() -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 2,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
    }
}

#[test]
#[cfg_attr(
    not(feature = "zoo-proofs"),
    ignore = "zoo fidelity proof: a full search + transcribe loop. ALSO CURRENTLY FAILING (pre-existing, not caused by the ignore): `attempt to add with overflow` in the u64 heuristic_total sum (implementation_search.rs) under dynamic-bounds pricing — suspected i64::MAX slice-end literals reaching midpoint estimation. Bisected to f2464436 and earlier. Run with `cargo test -p whisper -- --ignored`."
)]
fn tiny_transcribe_loop_is_deterministic_and_advances_the_cache() {
    let dims = WhisperDims::tiny();
    let mel: Vec<f32> = (0..dims.n_mels * dims.mel_frames())
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();

    let run = || {
        let step = TranscribeStep::build(&dims);
        let pairs = weights::random_weights(&step.cx);
        let mut transcriber =
            Transcriber::start(step, &mel, &pairs, &smoke_search()).expect("search");
        let mut rows = Vec::new();
        for token in [1u32, 2, 0] {
            rows.push(transcriber.step(token).expect("step"));
        }
        (rows, transcriber)
    };

    let (rows, transcriber) = run();
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), dims.vocab, "logits row {index}");
        assert!(row.iter().all(|v| v.is_finite()), "step {index} non-finite");
    }
    for (layer, k_state) in transcriber.state.k.iter().enumerate() {
        for slot in 0..dims.text_ctx {
            let written = k_state[slot * dims.state..(slot + 1) * dims.state]
                .iter()
                .any(|v| *v != 0.0);
            assert_eq!(written, slot < 3, "layer {layer} slot {slot} frontier");
        }
    }
    let (rows_again, _) = run();
    assert_eq!(rows, rows_again, "transcribe loop is not deterministic");
}
