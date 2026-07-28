//! The logical-SSA selection search: a mutation-only hill climb over
//! per-value producer genomes, profiled on the real `SsaReferenceRuntime` —
//! luminal's search shape (no cost models, profile the real thing, keep the
//! best, mutate) over our genome representation.
//!
//! Genomes that fail to extract (cycles, contract violations) are discarded
//! and replaced with fresh random rolls — the repair strategy. Many genomes
//! build the same plan (dead rows are unread), so every built plan is
//! fingerprinted and duplicates reuse the cached measurement instead of
//! burning profile time (the plan-hash dedup ruling, 2026-07-27).

use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashMap;

use crate::bufferize::BufferIrGraph;
use crate::extractor::{self, Genome};
use crate::hlir_to_logical::LogicalProgram;
use crate::ssa_reference::SsaReferenceRuntime;

#[derive(Debug, Clone)]
pub struct LogicalSearchOptions {
    pub generations: usize,
    pub generation_size: usize,
    /// Point mutations per offspring. Mutations hit ANY producer class —
    /// dead rows included, deliberately: a dead-row mutation is free now and
    /// pre-stages the choice a later route flip lands on.
    pub mutations: usize,
    pub trials: usize,
    pub seed: u64,
}

impl Default for LogicalSearchOptions {
    fn default() -> Self {
        Self {
            generations: 8,
            generation_size: 8,
            mutations: 2,
            trials: 3,
            seed: 0,
        }
    }
}

#[derive(Debug)]
pub struct SearchOutcome {
    pub best_plan: BufferIrGraph,
    pub best_genome: Genome,
    pub best_nanos: u128,
    /// Plans actually profiled (distinct fingerprints).
    pub plans_profiled: usize,
    /// Candidates answered from the fingerprint cache without re-profiling.
    pub fingerprint_hits: usize,
}

/// Search the saturated e-graph for the fastest executable plan, profiling
/// with the given caller data. Deterministic for a fixed seed.
pub fn search_logical(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<i64, Vec<f32>>,
    options: &LogicalSearchOptions,
) -> Result<SearchOutcome> {
    let index = extractor::producer_index(egraph);
    ensure!(!index.is_empty(), "no producer classes to search over");
    let classes: Vec<_> = index.keys().cloned().collect();
    let mut rng = StdRng::seed_from_u64(options.seed);

    let mut random_genome = |rng: &mut StdRng| {
        let mut genome = Genome::default();
        for (class, candidates) in &index {
            let pick = &candidates[rng.random_range(0..candidates.len())];
            genome.choices.insert(class.clone(), pick.1.clone());
        }
        genome
    };
    let mutate = |parent: &Genome, rng: &mut StdRng, count: usize| {
        let mut child = parent.clone();
        for _ in 0..count {
            let class = &classes[rng.random_range(0..classes.len())];
            let candidates = &index[class];
            let pick = &candidates[rng.random_range(0..candidates.len())];
            child.choices.insert(class.clone(), pick.1.clone());
        }
        child
    };

    // fingerprint → measured nanos (the dedup cache).
    let mut cache: FxHashMap<u64, u128> = FxHashMap::default();
    let mut plans_profiled = 0usize;
    let mut fingerprint_hits = 0usize;
    let mut best: Option<(u128, Genome, BufferIrGraph)> = None;

    let mut profile_plan = |plan: &BufferIrGraph, trials: usize| -> Result<u128> {
        let mut runtime = SsaReferenceRuntime::default();
        runtime.load_plan(plan.clone());
        for (id, data) in input_data {
            runtime.set_data(*id, data.clone());
        }
        runtime.execute()?; // warmup + validity
        let mut best_nanos = u128::MAX;
        for _ in 0..trials.max(1) {
            let start = Instant::now();
            runtime.execute()?;
            best_nanos = best_nanos.min(start.elapsed().as_nanos());
        }
        Ok(best_nanos)
    };

    for generation in 0..options.generations {
        let mut candidates: Vec<Genome> = Vec::with_capacity(options.generation_size);
        match &best {
            None => {
                for _ in 0..options.generation_size {
                    candidates.push(random_genome(&mut rng));
                }
            }
            Some((_, parent, _)) => {
                let parent = parent.clone();
                for _ in 0..options.generation_size {
                    candidates.push(mutate(&parent, &mut rng, options.mutations));
                }
            }
        }

        for genome in candidates {
            // Extraction failure = invalid genome (cycle, contract breach):
            // discard; the next generation's fresh mutations are the repair.
            let Ok(Some(graph)) = extractor::extract_layout_ir_with_genome(egraph, &genome)
            else {
                continue;
            };
            let fingerprint = extractor::plan_fingerprint(&graph);
            let nanos = match cache.get(&fingerprint) {
                Some(nanos) => {
                    fingerprint_hits += 1;
                    *nanos
                }
                None => {
                    let Ok(plan) =
                        crate::bufferize::bufferize(&crate::dps::dps_rewrite(&graph))
                    else {
                        continue;
                    };
                    let Ok(nanos) = profile_plan(&plan, options.trials) else {
                        continue;
                    };
                    cache.insert(fingerprint, nanos);
                    plans_profiled += 1;
                    if best.as_ref().is_none_or(|(best_nanos, _, _)| nanos < *best_nanos) {
                        best = Some((nanos, genome.clone(), plan));
                    }
                    continue;
                }
            };
            if best.as_ref().is_none_or(|(best_nanos, _, _)| nanos < *best_nanos) {
                let Ok(plan) = crate::bufferize::bufferize(&crate::dps::dps_rewrite(&graph))
                else {
                    continue;
                };
                best = Some((nanos, genome.clone(), plan));
            }
        }

        if best.is_none() && generation + 1 == options.generations {
            break;
        }
    }

    let (best_nanos, best_genome, best_plan) =
        best.ok_or_else(|| anyhow!("no candidate genome produced an executable plan"))?;
    let _ = program; // binding tables travel with the caller; kept for future bucket plumbing
    Ok(SearchOutcome {
        best_plan,
        best_genome,
        best_nanos,
        plans_profiled,
        fingerprint_hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CompileOptions, Graph};
    use crate::hlir::ReferenceRuntime;
    use crate::hlir_to_logical::hlir_to_logical;
    use crate::op::Runtime;
    use egglog::SerializeConfig;

    /// A REAL selection space (x+y and x*y from shared inputs offers the
    /// fused kernel vs the pair, plus commuted and mutating variants): the
    /// search must return a numerically correct plan, and the fingerprint
    /// cache must absorb duplicate plans.
    #[test]
    fn search_returns_a_correct_plan_and_dedups_duplicate_plans() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(4);
            let y = cx.tensor(4);
            let a = (x + y).output();
            let m = (x * y).output();
            (cx, x, y, a, m)
        };
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let y_data = vec![10.0, 20.0, 30.0, 40.0];

        // Their numbers.
        let (mut cx, x, y, a, m) = build();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut theirs = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        theirs.set_data(x.id, x_data.clone());
        theirs.set_data(y.id, y_data.clone());
        theirs.execute(&cx.dyn_map);
        let their_a = theirs.get_f32(a.id).clone();
        let their_m = theirs.get_f32(m.id).clone();

        // Our search.
        let (cx2, x2, y2, a2, m2) = build();
        let program = hlir_to_logical(&cx2).expect("translates");
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            program.text
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph.parse_and_run_program(None, &text).expect("program runs");
        let serialized = egraph.serialize(SerializeConfig::default()).egraph;

        let mut inputs = FxHashMap::default();
        inputs.insert(x2.id.index() as i64, x_data.clone());
        inputs.insert(y2.id.index() as i64, y_data.clone());
        let outcome = search_logical(
            &serialized,
            &program,
            &inputs,
            &LogicalSearchOptions::default(),
        )
        .expect("search finds an executable plan");

        assert!(
            outcome.fingerprint_hits > 0,
            "small space, many genomes: the plan cache must fire \
             (profiled {}, hits {})",
            outcome.plans_profiled,
            outcome.fingerprint_hits
        );

        let mut runtime = SsaReferenceRuntime::default();
        runtime.load_plan(outcome.best_plan.clone());
        runtime.set_data(x2.id.index() as i64, x_data);
        runtime.set_data(y2.id.index() as i64, y_data);
        runtime.execute().expect("best plan executes");
        let ours_a = runtime.get_f32(a2.id.index() as i64).unwrap();
        let ours_m = runtime.get_f32(m2.id.index() as i64).unwrap();
        for (ours, theirs) in [(ours_a, &their_a), (ours_m, &their_m)] {
            assert_eq!(ours.len(), theirs.len());
            for (index, (lhs, rhs)) in ours.iter().zip(theirs).enumerate() {
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * rhs.abs().max(1.0),
                    "element {index}: ours {lhs} vs theirs {rhs}"
                );
            }
        }
    }
}
