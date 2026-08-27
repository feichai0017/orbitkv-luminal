//! TestRuntime — the tests-side runtime vocabulary (rehoming ruling
//! 2026-08-13).
//!
//! The reference runtime implements only PLAIN spellings of the logical
//! ops; the fused ops (AddMulFused, MatMulFused) were deleted from it at
//! commit aff22598 and their test estate died with them. The estate that
//! needs fused/view variants lives HERE instead: extraction and egglog
//! assembly are runtime-injectable
//! (`luminal::extractor::extract_layout_ir_with_matchers`,
//! `luminal::egglog_snippet::assembled_program_for`), so this crate is a
//! MATCHER LIST plus fixture runners — no runtime machinery duplicated.

pub mod add_mul_fused;

pub use add_mul_fused::{AddMulFused, AddMulFusedDps, AddMulFusedMatcher};

// ---------------------------------------------------------------------------
// the included `src/extractor.rs` text resolves `crate::dtype`,
// `crate::layout_ir`, `crate::logical_op`, and `crate::reference` against
// these, so its types ARE luminal's types. Not part of this crate's
// vocabulary — reach for `luminal::…` directly in tests.
// ---------------------------------------------------------------------------
pub mod dtype {
    pub use luminal::dtype::*;
}
pub mod layout_ir {
    pub use luminal::layout_ir::*;
}
pub mod logical_op {
    pub use luminal::logical_op::*;
}
pub mod reference {
    pub mod ops {
        pub use luminal_reference::ops::*;
    }
}

use luminal::layout_ir::ExtractedGraph;
use luminal::layout_ir::OpMatcher;

/// THE TestRuntime vocabulary: the reference registry, plus the op
/// variants the reference runtime deliberately does not implement — the
/// view op and the fused add+mul pair. Tests in this crate extract and
/// assemble against exactly this list.
pub fn matchers() -> Vec<Box<dyn OpMatcher>> {
    let mut matchers = luminal_reference::ops::built_in_matchers();
    matchers.push(Box::new(luminal_reference::ops::IndexMapApplyViewMatcher));
    matchers.push(Box::new(AddMulFusedMatcher));
    matchers
}

/// The assembled egglog program for this runtime's vocabulary, plus the
/// fixture script, run to saturation and serialized — the raw material
/// for every extraction below. Panics on any failure: these are fixtures.
pub fn serialize_fixture(script_text: &str) -> luminal::prelude::egraph_serialize::EGraph {
    use egglog::SerializeConfig;

    let preamble = luminal::egglog_snippet::assembled_program_for(&matchers());
    let program = format!("{preamble}\n\n{script_text}");
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(None, &program)
        .unwrap_or_else(|err| panic!("egglog failed on fixture: {err}"));
    egraph.serialize(SerializeConfig::default()).egraph
}

/// Deterministic (min-cost) extraction of a fixture script on this
/// runtime's vocabulary.
pub fn extract_fixture(script_text: &str) -> ExtractedGraph {
    let serialized = serialize_fixture(script_text);
    luminal::extractor::extract_layout_ir_with_matchers(&serialized, matchers())
        .expect("extraction succeeds")
        .expect("fixture produced no extracted graph")
}

/// Build a TOTAL genome over a fixture's produced classes: each class takes
/// the first preference (an implementation constructor name) it can satisfy,
/// falling back to its first candidate — the producer index is
/// deterministically sorted, so the same preferences always build the same
/// genome. (Adapted from `luminal_reference::harness::genome_preferring`, which
/// is public but hard-wired to the built-in producer index; this one runs
/// over THIS runtime's matcher set via the genome seam.)
pub fn genome_preferring(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
    preferences: &[&str],
) -> luminal::extractor::Genome {
    let index = luminal::extractor::producer_index_with_matchers(egraph, matchers());
    let mut genome = luminal::extractor::Genome::default();
    for (class, candidates) in index {
        let pick = preferences
            .iter()
            .find_map(|preferred| {
                candidates
                    .iter()
                    .find(|(name, _)| name.as_str() == *preferred)
            })
            .or_else(|| candidates.first())
            .expect("produced classes have candidates");
        genome.choices.insert(class, pick.1.clone());
    }
    genome
}

/// Genome-driven fixture extraction (the selection adapter's walk) plus the
/// plan fingerprint the search dedups on.
pub fn extract_fixture_with_genome(
    script_text: &str,
    preferences: &[&str],
) -> (ExtractedGraph, u64) {
    let serialized = serialize_fixture(script_text);
    let genome = genome_preferring(&serialized, preferences);
    let graph =
        luminal::extractor::extract_layout_ir_with_genome_and_matchers(&serialized, &genome, matchers())
            .expect("genome extraction runs")
            .expect("genome extraction reaches the boundary");
    let fingerprint = luminal::extractor::plan_fingerprint(&graph);
    (graph, fingerprint)
}
