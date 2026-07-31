//! IMPLEMENTATION search (renamed from logical_search, ruling 2026-07-30):
//! there is NO search at the logical level — saturation discovers
//! implementations; this module only SELECTS among them, pricing every
//! candidate by executing its bufferized plan on the runtime.
//!
//! A mutation-only hill climb over
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
use std::collections::BTreeMap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashMap;

use crate::bufferize::BufferIrGraph;
use crate::extractor::{self, Genome};
use crate::graph::LogicalProgram;
use crate::ssa_reference::SsaReferenceRuntime;

#[derive(Debug, Clone)]
pub struct ImplementationSearchOptions {
    pub generations: usize,
    pub generation_size: usize,
    /// Point mutations per offspring. Mutations hit ANY producer class —
    /// dead rows included, deliberately: a dead-row mutation is free now and
    /// pre-stages the choice a later route flip lands on.
    pub mutations: usize,
    pub trials: usize,
    pub seed: u64,
}

impl Default for ImplementationSearchOptions {
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
pub fn search_implementations(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<i64, Vec<f32>>,
    options: &ImplementationSearchOptions,
) -> Result<SearchOutcome> {
    search_implementations_with_ops(egraph, program, input_data, options, None)
}

/// As [`search_implementations`], with the runtime's ALLOWABLE-OPS
/// inventory made explicit (M3 Step 2: per-runtime, unstandardized).
pub fn search_implementations_with_ops(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<i64, Vec<f32>>,
    options: &ImplementationSearchOptions,
    allow_override: Option<Vec<&'static str>>,
) -> Result<SearchOutcome> {
    let allow = allow_override.unwrap_or_else(crate::ssa_reference::reference_allow_list);
    let index = extractor::producer_index_with_ops(egraph, Some(&allow));
    ensure!(!index.is_empty(), "no producer classes to search over");
    let classes: Vec<_> = index.keys().cloned().collect();
    let mut rng = StdRng::seed_from_u64(options.seed);

    let random_genome = |rng: &mut StdRng| {
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

    let profile_plan = |plan: &BufferIrGraph, trials: usize| -> Result<u128> {
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
            let Ok(Some(graph)) =
                extractor::extract_layout_ir_with_genome_and_ops(egraph, &genome, Some(&allow))
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


/// One bucket combination's finished search: the dim ranges it covers, the
/// representative pins it was searched at, and the winning plan.
#[derive(Debug)]
pub struct BucketPlan {
    pub ranges: BTreeMap<char, (usize, usize)>,
    pub representative: FxHashMap<char, usize>,
    pub program: LogicalProgram,
    pub outcome: SearchOutcome,
}

/// Range-seeded bucketed search, mirroring their per-bucket model: one
/// Cartesian combination of `DimBucket`s per search, each combination run
/// TWICE — a bucket-wide RANGE-seeded render whose fixpoint checks prove
/// the model sound over the whole interval (validation only; ranges do not
/// collapse), then a representative-pinned render that is searched and
/// profiled. `select_bucket` picks the covering plan at runtime.
///
/// Slice note (documented divergence from their symbolic LLIR): each
/// winning plan is STATIC at its representative; executing at another pin
/// re-renders — genome transfer across renders is future work.
pub fn bucketed_search_implementations(
    graph: &crate::graph::Graph,
    dim_buckets: &BTreeMap<char, Vec<crate::graph::DimBucket>>,
    input_data: impl Fn(&FxHashMap<char, usize>) -> FxHashMap<i64, Vec<f32>>,
    options: &ImplementationSearchOptions,
) -> Result<Vec<BucketPlan>> {
    ensure!(!dim_buckets.is_empty(), "no dim buckets supplied");

    // M3 Topic C: buckets assemble NATIVELY — one recorder model, per-
    // bucket binding seeds (ranges for the bucket-wide validation render,
    // tight [n,n] pins for the representative render). The model text
    // never changes across buckets; only the binding does.
    let (pre, input_slots, output_slots, post) = graph
        .logical
        .native_parts()
        .map_err(|reason| anyhow!("native load refused: {reason}"))?;
    let seeds_text = |seeds: &BTreeMap<char, (u64, u64)>| {
        let mut text = String::new();
        for (var, (lower, upper)) in seeds {
            text.push_str(&format!(
                "(set (lower-bound-of (IntVar \"{var}\")) (bigint {lower}))\n\
                 (set (upper-bound-of (IntVar \"{var}\")) (bigint {upper}))\n"
            ));
        }
        text
    };
    let assemble = |seeds: &BTreeMap<char, (u64, u64)>| crate::graph::LogicalProgram {
        text: format!(
            "{pre}{}{}{post}",
            seeds_text(seeds),
            crate::reference_binding::SCHEDULE
        ),
        input_slots: input_slots.clone(),
        output_slots: output_slots.clone(),
    };

    // Cartesian combinations, dims in sorted order (their bucket_combinations).
    let dims: Vec<&char> = dim_buckets.keys().collect();
    let mut combos: Vec<Vec<usize>> = vec![Vec::new()];
    for dim in &dims {
        let count = dim_buckets[*dim].len();
        combos = combos
            .into_iter()
            .flat_map(|combo| {
                (0..count).map(move |index| {
                    let mut next = combo.clone();
                    next.push(index);
                    next
                })
            })
            .collect();
    }

    let mut plans = Vec::new();
    for combo in combos {
        let mut ranges = BTreeMap::new();
        let mut representative = graph.dyn_map.clone();
        for (dim, bucket_index) in dims.iter().zip(&combo) {
            let bucket = &dim_buckets[*dim][*bucket_index];
            ranges.insert(**dim, (bucket.min, bucket.max));
            representative.insert(**dim, bucket.representative_value());
        }

        // Bucket-wide soundness: the range-seeded render must run its whole
        // fixpoint (authoring-contract checks included) over the interval.
        let mut validation_seeds: BTreeMap<char, (u64, u64)> = BTreeMap::new();
        for (dim, value) in &representative {
            validation_seeds.insert(*dim, (*value as u64, *value as u64));
        }
        for (dim, (min, max)) in &ranges {
            validation_seeds.insert(*dim, (*min as u64, *max as u64));
        }
        let validation = assemble(&validation_seeds);
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            validation.text
        );
        crate::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} fails bucket-wide validation: {err}"))?;

        // Representative render: pinned via tight bounds, searched, profiled.
        let mut pin_seeds: BTreeMap<char, (u64, u64)> = BTreeMap::new();
        for (dim, value) in &representative {
            pin_seeds.insert(*dim, (*value as u64, *value as u64));
        }
        let program = assemble(&pin_seeds);
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            program.text
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} representative render fails: {err}"))?;
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;
        let data = input_data(&representative);
        let outcome = search_implementations(&serialized, &program, &data, options)?;
        plans.push(BucketPlan { ranges, representative, program, outcome });
    }
    Ok(plans)
}

/// The covering bucket plan for a concrete dim assignment, if any.
pub fn select_bucket<'a>(
    plans: &'a [BucketPlan],
    dims: &FxHashMap<char, usize>,
) -> Option<&'a BucketPlan> {
    plans.iter().find(|plan| {
        plan.ranges.iter().all(|(dim, (min, max))| {
            dims.get(dim).is_some_and(|value| value >= min && value <= max)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
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

        // GOLDEN (pinned: x + y and x * y on the fixed data).
        let their_a = vec![11.0, 22.0, 33.0, 44.0];
        let their_m = vec![10.0, 40.0, 90.0, 160.0];

        // Our search.
        let (cx2, x2, y2, a2, m2) = build();
        let program = cx2.logical.native_program().expect("native program");
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
        let outcome = search_implementations(
            &serialized,
            &program,
            &inputs,
            &ImplementationSearchOptions::default(),
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

    /// BUCKETED: two buckets over 'a', each validated bucket-wide
    /// (range seeds) and searched at its representative; selection covers
    /// runtime dims; each bucket's plan agrees with ReferenceRuntime at its
    /// representative.
    #[test]
    fn bucketed_search_validates_searches_and_selects() {
        use crate::graph::DimBucket;

        let build = |dim: usize| {
            let mut cx = Graph::new();
            cx.set_dim('a', dim);
            let x = cx.tensor(('a', 2));
            let y = cx.tensor(('a', 2));
            let out = (x * y).output();
            (cx, x, y, out)
        };

        let buckets: BTreeMap<char, Vec<DimBucket>> = [(
            'a',
            vec![DimBucket::new(2, 4), DimBucket::new(5, 9)],
        )]
        .into();

        // The graph used for SEARCH: built at any pin (per-bucket renders
        // re-pin), sharing the same HLIR shape.
        let (cx, x, y, _out) = build(3);
        let data_for = |rep: &FxHashMap<char, usize>| {
            let n = rep[&'a'] * 2;
            let mut data = FxHashMap::default();
            data.insert(x.id.index() as i64, (0..n).map(|v| v as f32 + 1.0).collect());
            data.insert(y.id.index() as i64, (0..n).map(|v| v as f32 * 0.5).collect());
            data
        };
        let plans = bucketed_search_implementations(
            &cx,
            &buckets,
            data_for,
            &ImplementationSearchOptions::default(),
        )
        .expect("bucketed search completes");
        assert_eq!(plans.len(), 2, "one plan per bucket");

        // Selection covers each bucket; out-of-range dims select nothing.
        let mut dims = FxHashMap::default();
        dims.insert('a', 3usize);
        assert!(select_bucket(&plans, &dims).unwrap().ranges[&'a'] == (2, 4));
        dims.insert('a', 7usize);
        assert!(select_bucket(&plans, &dims).unwrap().ranges[&'a'] == (5, 9));
        dims.insert('a', 20usize);
        assert!(select_bucket(&plans, &dims).is_none());

        // Numeric agreement at each bucket's representative.
        for plan in &plans {
            let rep = plan.representative[&'a'];
            // GOLDEN (computed: out = x * y with x[i] = i+1, y[i] = i*0.5
            // — the data_for closure's values at this representative).
            let n = rep * 2;
            let expected: Vec<f32> = (0..n).map(|v| (v as f32 + 1.0) * (v as f32 * 0.5)).collect();
            let data = data_for(&plan.representative);

            let mut runtime = SsaReferenceRuntime::default();
            runtime.load_plan(plan.outcome.best_plan.clone());
            for (id, values) in &data {
                runtime.set_data(*id, values.clone());
            }
            runtime.execute().expect("bucket plan executes at representative");
            let ours = runtime
                .get_f32(plan.program.output_slots[0].1 as i64)
                .unwrap();
            assert_eq!(ours.len(), expected.len());
            for (index, (lhs, rhs)) in ours.iter().zip(&expected).enumerate() {
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * rhs.abs().max(1.0),
                    "bucket {:?} element {index}: ours {lhs} vs theirs {rhs}",
                    plan.ranges
                );
            }
        }
    }

}
