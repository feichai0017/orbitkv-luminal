//! The cross-driver equivalence proof: sequence A decoded ALONE through
//! llama3's position-slots driver must produce the SAME logits as A
//! decoded in a batched page-table tick interleaved with an unrelated
//! sequence B — the isolation mask makes B invisible to A even though
//! they share one pool and one execute. Same tiny dims, same
//! deterministic weights, exact equality demanded.

use llama3::model::Llama3Dims;
use llama3::weights;
use luminal::implementation_search::ImplementationSearchOptions;
use paged_llama3::{BatchStep, ROWS, Ticker};

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
fn batched_sequences_isolate_exactly() {
    let dims = Llama3Dims::tiny();
    let slots = 8usize;

    // Reference: A alone through the position-slots driver.
    let solo = {
        let step = llama3::DecodeStep::build(&dims, slots);
        let pairs = weights::random_weights(&step.cx);
        let mut decoder = llama3::Decoder::start(step, &pairs, &smoke_search()).expect("search");
        let mut rows = Vec::new();
        for token in [1u32, 2, 3] {
            rows.push(decoder.step(token).expect("step"));
        }
        rows
    };

    // Batched: A interleaved with B in shared-pool page-table ticks.
    // Weight data is deterministic by construction, so both graphs
    // carry identical parameters.
    let step = BatchStep::build(&dims, slots);
    let pairs = weights::random_weights(&step.cx);
    let mut ticker = Ticker::start(step, &pairs, &smoke_search()).expect("search");
    let a = ticker.table.new_sequence();
    let b = ticker.table.new_sequence();
    let a_tokens = [1u32, 2, 3];
    let b_tokens = [4u32, 1, 4];
    let mut a_logits = Vec::new();
    for step_index in 0..3 {
        let logits = ticker
            .tick(&[(a, a_tokens[step_index]), (b, b_tokens[step_index])])
            .expect("tick");
        assert_eq!(logits.len(), ROWS * dims.vocab);
        a_logits.push(logits[..dims.vocab].to_vec());
    }

    for (step_index, (solo_row, batched_row)) in solo.iter().zip(&a_logits).enumerate() {
        for (column, (x, y)) in solo_row.iter().zip(batched_row).enumerate() {
            assert!(
                (x - y).abs() <= 1e-5 * x.abs().max(1.0),
                "step {step_index} column {column}: solo {x} vs batched {y} — \
                 sequence B leaked into A's attention"
            );
        }
    }
}
