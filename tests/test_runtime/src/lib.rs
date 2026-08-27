//! TestRuntime — the tests-side runtime vocabulary (rehoming ruling
//! 2026-08-13).
//!
//! THE SPLIT this crate exists to hold: the reference runtime is
//! functional, out-of-place and view-free. Its kernel table carries only
//! `*FunctionalDps` types and `reference_allow_list()` is *derived* from
//! that table, so a mutating or view-shaped op registered there is
//! extractable but can never be selected — dead weight paid for on every
//! saturation. The op shapes that exercise `Bufferizable` / `ToDps` /
//! the bufferizer's aliasing machinery live HERE instead, and this crate
//! owns them outright: instance, DPS form, matcher and `.egg` rewrites,
//! one folder per op under [`ops`].
//!
//! It depends on NO other runtime crate — not even the reference one.
//! The 22 plain functional ops are forked here too, kernels omitted.
//! These are two different runtimes with two different jobs, and their
//! vocabularies are EXPECTED to diverge: this op list will be whittled
//! down to the shapes the bufferizer contracts actually need, while the
//! reference registry moves toward canonical-layout-only. Nothing is
//! kept in sync on purpose.
//!
//! No runtime machinery is duplicated: extraction and egglog assembly
//! are runtime-injectable
//! (`luminal::extractor::extract_layout_ir_with_matchers`,
//! `luminal::egglog_snippet::assembled_program_for`), so this crate is a
//! MATCHER LIST, its own op folders, and fixture runners. It is
//! plan-level only — no kernels, no executor: everything it asserts is a
//! property of an `ExtractedGraph` or a `BufferIrGraph`.

pub mod ops;

pub use ops::{
    AddMulFused, AddMulFusedDps, AddMulFusedMatcher, IndexMapApplyView, IndexMapApplyViewMatcher,
};

use std::path::PathBuf;

use luminal::layout_ir::ExtractedGraph;
use luminal::layout_ir::OpMatcher;

/// THE TestRuntime vocabulary, every entry owned by this crate: 22
/// forked functional ops, the metadata view op, the fused add+mul pair,
/// and the 12 mutating forms. Tests here extract and assemble against
/// exactly this list.
pub fn matchers() -> Vec<Box<dyn OpMatcher>> {
    let mut matchers = ops::functional::functional_matchers();
    matchers.push(Box::new(IndexMapApplyViewMatcher));
    matchers.push(Box::new(AddMulFusedMatcher));
    matchers.extend(ops::mutating::mutating_matchers());
    matchers
}

/// A fixture owned by THIS crate, under `fixtures/`.
///
/// Every fixture this runtime uses lives here, including forked copies of
/// scripts the reference corpus also carries. The fork is deliberate: the
/// reference copy cannot name a mutating or view constructor (an
/// undeclared constructor is an egglog PARSE failure), and the two
/// vocabularies are expected to diverge. Nothing here reaches into the
/// core script tree.
pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

/// [`extract_fixture`] on one of this crate's own fixtures, by file name.
pub fn extract_fixture_by_name(name: &str) -> ExtractedGraph {
    let source = std::fs::read_to_string(fixture_path(name))
        .unwrap_or_else(|_| panic!("fixture script {name} readable"));
    extract_fixture(&source)
}

/// [`extract_fixture_by_name`] restricted to an allow-list of
/// `LayoutTensorOp` constructor names — forces extraction through
/// specific implementations so a test can pin one spelling end to end.
/// Runs over THIS runtime's matcher set.
pub fn extract_fixture_with_ops(name: &str, allowed: &[&str]) -> ExtractedGraph {
    try_extract_fixture_with_ops(name, allowed)
        .expect("extraction succeeds")
        .unwrap_or_else(|| panic!("fixture {name} produced no extracted graph"))
}

/// The fallible form of [`extract_fixture_with_ops`] — used to assert
/// that an unsatisfiable filter refuses, rather than silently widening.
pub fn try_extract_fixture_with_ops(
    name: &str,
    allowed: &[&str],
) -> anyhow::Result<Option<ExtractedGraph>> {
    let source = std::fs::read_to_string(fixture_path(name))
        .unwrap_or_else(|_| panic!("fixture script {name} readable"));
    try_extract_text_with_ops(&source, allowed)
}

fn try_extract_text_with_ops(
    script_text: &str,
    allowed: &[&str],
) -> anyhow::Result<Option<ExtractedGraph>> {
    let serialized = serialize_fixture(script_text);
    luminal::extractor::extract_layout_ir_with_ops_and_matchers(
        &serialized,
        Some(allowed),
        matchers(),
    )
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
/// genome. Runs over THIS runtime's matcher set via the genome seam.
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
    let graph = luminal::extractor::extract_layout_ir_with_genome_and_matchers(
        &serialized,
        &genome,
        matchers(),
    )
    .expect("genome extraction runs")
    .expect("genome extraction reaches the boundary");
    let fingerprint = luminal::extractor::plan_fingerprint(&graph);
    (graph, fingerprint)
}
