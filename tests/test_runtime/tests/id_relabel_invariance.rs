//! ID-RELABEL INVARIANCE — the MEASUREMENT (TestRuntime vocabulary).
//!
//! Ruling (Austin, 2026-09-02): *"Assume that all node ids and eclass ids
//! are random every time. If you want to find an eclass you need to find
//! it. Use that in your design."*
//!
//! This file does not enforce that ruling; it MEASURES how far the tree
//! currently is from it. For every fixture it runs the pipeline on the
//! serialized e-graph and on K random relabelings of the same e-graph
//! (`luminal::test_support::relabel`), then compares ID-FREE digests
//! (`luminal::test_support::digest`). Everything it learns is PRINTED,
//! one `RELABEL` line per fixture; nothing is asserted, because the
//! known id-order dependence sites (the extractor's `producer_index_from`
//! sort, `sampling_space`'s `BTreeMap<ClassId,_>` walks, `render_layout`'s
//! `ids.sort()`, `roots.sort_by_key(ToString::to_string)`,
//! `BufferIrGraph::sort_key`, `stable_key`) are exactly what is being
//! surveyed — an assertion here would just be a red suite with no reading
//! in it.
//!
//! Run it: `cargo test -p test_runtime --test id_relabel_invariance -- --nocapture`.
//!
//! Three questions per fixture:
//!
//!  * `deterministic` — does `extract_layout_ir_with_matchers` (the
//!    min-cost extractor, no genome) plus its mock bufferization produce
//!    the same plan under relabeling?
//!  * `search` — does `search_implementations_with_runtime` at
//!    `harness_search_options()` (seed 0) elect the same plan, profile the
//!    same number of candidates, and refuse for the same reasons?
//!  * `summary_order` — does `BufferIrGraph::summary()` (the text the
//!    golden files pin) keep its LINE ORDER once ids are masked? Its
//!    buffer rows sort on `format!("0:{eclass}")`, so this is the direct
//!    probe of that one site.

use std::collections::BTreeSet;
use std::path::PathBuf;

use luminal::bufferize::BufferIrGraph;
use luminal::graph::LogicalProgram;
use luminal::implementation_search::{search_implementations_with_runtime, StaticProfiler};
use luminal::layout_ir::{LayoutRenderer, LayoutTensorInfo};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::FxHashMap;
use luminal::test_support::digest::{extracted_digest, mask_ids, plan_digest};
use luminal::test_support::relabel::relabel_egraph;
use luminal::test_support::MockLayout;

/// How many relabelings each fixture is measured against.
const K: u64 = 8;

/// The mock renderer: every value's layout renders to its own layout
/// class, exactly as `luminal::test_support::mock_layout_table` builds
/// the table for the deterministic path — so the two paths differ in
/// SEARCH, never in rendering.
struct MockRenderer;

impl LayoutRenderer<MockLayout> for MockRenderer {
    fn render_layout(
        &self,
        _egraph: &EGraph,
        value: &LayoutTensorInfo,
    ) -> anyhow::Result<MockLayout> {
        Ok(MockLayout(value.layout.eclass.clone()))
    }
}

/// The first line at which two renderings part company — the "first
/// differing element" the survey reports for every `no`.
fn first_diff(base: &str, other: &str) -> Option<String> {
    for (index, (left, right)) in base.lines().zip(other.lines()).enumerate() {
        if left != right {
            return Some(format!("line {index}: base={left:?} relabeled={right:?}"));
        }
    }
    let (base_lines, other_lines) = (base.lines().count(), other.lines().count());
    if base_lines != other_lines {
        return Some(format!(
            "line count differs: base={base_lines} relabeled={other_lines}"
        ));
    }
    None
}

/// The deterministic path: min-cost extraction, then the DPS rewrite and
/// mock bufferization the search path also runs.
fn deterministic(egraph: &EGraph) -> (String, String, String) {
    let extracted =
        luminal::extractor::extract_layout_ir_with_matchers(egraph, test_runtime::matchers());
    let graph = match extracted {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            return (
                "<no graph>".into(),
                "<no graph>".into(),
                "<no graph>".into(),
            )
        }
        Err(err) => {
            let text = format!("<extract error: {err:#}>");
            return (text.clone(), text.clone(), text);
        }
    };
    let dps = luminal::dps::dps_rewrite(&graph);
    match luminal::test_support::bufferize_mock(&dps) {
        Ok(plan) => (
            extracted_digest(&graph),
            plan_digest(&plan),
            mask_ids(&plan.summary()),
        ),
        Err(err) => {
            let text = format!("<bufferize error: {err:#}>");
            (extracted_digest(&graph), text.clone(), text)
        }
    }
}

/// The seeded search path, at the harness budget on seed 0.
fn searched(egraph: &EGraph) -> (String, String) {
    let program = LogicalProgram {
        text: String::new(),
        input_slots: Vec::new(),
        output_slots: Vec::new(),
    };
    let data: FxHashMap<luminal::prelude::NodeIndex, luminal::prelude::TypedBuffer> =
        FxHashMap::default();
    let outcome = search_implementations_with_runtime(
        egraph,
        &program,
        &data,
        &luminal::test_support::harness_search_options(),
        None,
        test_runtime::matchers(),
        &MockRenderer,
        &mut StaticProfiler,
    );
    match outcome {
        Ok(outcome) => {
            let accounting = format!(
                "plans_profiled={} refusals=[{}]",
                outcome.plans_profiled,
                outcome.refusal_breakdown.summary()
            );
            let plan: &BufferIrGraph<MockLayout> = &outcome.best_plan;
            (
                format!("{accounting}\n{}", plan_digest(plan)),
                mask_ids(&plan.summary()),
            )
        }
        Err(err) => {
            let text = format!("<search refused: {err:#}>");
            (text.clone(), text)
        }
    }
}

/// The `.egg` fixtures this runtime owns, in file-name order.
fn fixtures() -> Vec<String> {
    let dir: PathBuf = test_runtime::fixture_path("any.egg")
        .parent()
        .expect("fixtures directory")
        .to_path_buf();
    let names: BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {dir:?}: {e}"))
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".egg"))
        .collect();
    names.into_iter().collect()
}

/// One fixture's whole survey. Returns the `RELABEL` lines it printed so
/// the caller can echo a roll-up at the end.
fn survey(label: &str, egraph: &EGraph) -> Vec<String> {
    let (base_extract, base_plan, base_summary) = deterministic(egraph);
    let (base_search, base_search_summary) = searched(egraph);

    let mut extract_diff: Option<(u64, String)> = None;
    let mut plan_diff: Option<(u64, String)> = None;
    let mut summary_diff: Option<(u64, String)> = None;
    let mut search_diff: Option<(u64, String)> = None;
    let mut search_summary_diff: Option<(u64, String)> = None;

    for seed in 1..=K {
        let relabeled = relabel_egraph(egraph, seed);
        let (extract, plan, summary) = deterministic(&relabeled);
        let (search, search_summary) = searched(&relabeled);
        if extract_diff.is_none() {
            extract_diff = first_diff(&base_extract, &extract).map(|d| (seed, d));
        }
        if plan_diff.is_none() {
            plan_diff = first_diff(&base_plan, &plan).map(|d| (seed, d));
        }
        if summary_diff.is_none() {
            summary_diff = first_diff(&base_summary, &summary).map(|d| (seed, d));
        }
        if search_diff.is_none() {
            search_diff = first_diff(&base_search, &search).map(|d| (seed, d));
        }
        if search_summary_diff.is_none() {
            search_summary_diff =
                first_diff(&base_search_summary, &search_summary).map(|d| (seed, d));
        }
    }

    let verdict = |diff: &Option<(u64, String)>| if diff.is_some() { "no" } else { "yes" };
    let mut lines = vec![format!(
        "RELABEL {label} K={K} deterministic={} plan={} search={} summary_order={} search_summary_order={}",
        verdict(&extract_diff),
        verdict(&plan_diff),
        verdict(&search_diff),
        verdict(&summary_diff),
        verdict(&search_summary_diff),
    )];
    for (what, diff) in [
        ("deterministic", &extract_diff),
        ("plan", &plan_diff),
        ("search", &search_diff),
        ("summary_order", &summary_diff),
        ("search_summary_order", &search_summary_diff),
    ] {
        if let Some((seed, detail)) = diff {
            lines.push(format!(
                "RELABEL {label}   {what} FIRST DIFF seed={seed} {detail}"
            ));
        }
    }
    for line in &lines {
        println!("{line}");
    }
    lines
}

#[test]
fn id_relabel_invariance_survey() {
    let mut roll_up = Vec::new();
    for name in fixtures() {
        let source = std::fs::read_to_string(test_runtime::fixture_path(&name))
            .unwrap_or_else(|e| panic!("fixture {name}: {e}"));
        let egraph = test_runtime::serialize_fixture(&source);
        roll_up.extend(survey(name.trim_end_matches(".egg"), &egraph));
    }
    println!("\n--- RELABEL roll-up ---");
    for line in roll_up {
        println!("{line}");
    }
}
