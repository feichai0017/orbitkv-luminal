//! M4 PHASE 5 ACCEPTANCE (CPU side): the view op is ELECTABLE on
//! CUDA-lite — real searched plans fold movement to producer redirects
//! and consumers read through the composed access.
//!
//! Mirrors `plan_smoke` on view-heavy fixtures, through the REAL
//! CudaRuntime ladder (load → search under the CUDA allow list → plan
//! inspection). Everything here is device-free: electing and folding a
//! view is planner work; only the read-through happens on the device
//! (the device differentials pin that half).
//!
//! Assertion discipline:
//!  * movement that folds is PINNED as zero materialize nodes — the op
//!    whose whole purpose is materializing an index map
//!    (`IndexMapApplyMaterialize`, label = IR identity) must not appear;
//!  * NO unfolded-view compute nodes — re-checked here by the same
//!    effect-predicate shape the plan validator uses (the validator in
//!    `luminal::bufferize` stays the fence; this keeps the acceptance
//!    test honest if the fence ever moves);
//!  * buffer/copy counts are PINNED per fixture (regression tripwires
//!    for the folded shape);
//!  * consumers' operand descriptors must CARRY the composed access
//!    (Phase-3 machinery), checked by EVALUATING the hop chain against
//!    the hand-computed map — hop count is the e-graph's business.

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::{BufferIrGraph, BufferNode, ComposedAccess};
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;

/// Search budget for the view fixtures: profiling is static (bytes
/// moved), so generations are cheap — enough sampling that the
/// all-views plan is reliably in the profiled set, seeded for
/// deterministic pins.
fn view_search_options() -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 4,
        generation_size: 8,
        mutations: 4,
        trials: 1,
        seed: 0,
    }
}

/// Load → search on the CUDA runtime; return the best plan.
fn plan_for(cx: &Graph, inputs: &[(NodeIndex, TypedBuffer)]) -> BufferIrGraph {
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    let data: FxHashMap<NodeIndex, TypedBuffer> = inputs.iter().cloned().collect();
    let outcome = rt.search(&data, &view_search_options()).expect("cuda search");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    rt.plan().expect("plan loaded").clone()
}

/// The plan-shape audit shared by every fixture. Returns
/// (compute_count, copy_count, buffer_count, composed slots).
fn audit(plan: &BufferIrGraph) -> (usize, usize, usize, Vec<(String, usize, ComposedAccess)>) {
    let mut computes = 0usize;
    let mut copies = 0usize;
    let mut composed = Vec::new();
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::BufferCopy { .. } => copies += 1,
            BufferNode::Compute { op, reads, writes, ties, operand_info, result_info } => {
                let label = op.label();
                if label == "BufferAlloc" || label == "BufferFree" {
                    continue;
                }
                computes += 1;

                // ZERO materialize nodes for foldable movement: every
                // fixture's movement is within the parsed expression
                // subset, so the materializing spelling must lose to
                // the fold. (Label = IR identity, house policy.)
                assert_ne!(
                    label, "IndexMapApplyMaterialize",
                    "foldable movement was materialized:\n{}",
                    plan.summary()
                );

                // NO unfolded-view compute nodes — the same
                // effect-predicate shape `validate_plan` fences on.
                let derives = |result: usize| ties.iter().any(|(_, r)| *r == result);
                let view_shaped = !reads.is_empty()
                    && !writes.is_empty()
                    && (0..reads.len()).all(|o| !op.operand_reads_memory(o))
                    && (0..writes.len())
                        .all(|r| !op.result_writes_memory(r) && derives(r));
                assert!(!view_shaped, "unfolded view ({label}) reached the plan");

                // Every kernel-bearing elected op has a codegen row.
                assert!(
                    luminal_cuda_lite::kernels::codegen_for(op.as_ref()).is_some(),
                    "elected op {label} has no codegen row"
                );

                for (slot, info) in operand_info.iter().enumerate() {
                    if let Some(access) = &info.composed_access {
                        composed.push((label.to_string(), slot, access.clone()));
                    }
                }
                for info in result_info {
                    assert!(
                        info.composed_access.is_none(),
                        "{label}: a compute RESULT is produced by the node, never through a fold"
                    );
                }
            }
            _ => {}
        }
    }
    (computes, copies, plan.buffers.len(), composed)
}

/// Evaluate a composed-access chain at one out-coordinate (the Phase-3
/// probe walk): hops[0] is the outermost fold, each hop's outputs feed
/// the next; returns final parent coordinates. Loud on unparsed hops.
fn chain_eval(access: &ComposedAccess, out_coord: &[usize]) -> Vec<i64> {
    let mut coords: Vec<usize> = out_coord.to_vec();
    for (k, hop) in access.hops.iter().enumerate() {
        let entries = hop
            .entries
            .as_ref()
            .unwrap_or_else(|| panic!("hop {k} beyond the parsed subset (fail-closed)"));
        let next: Vec<i64> = entries.iter().map(|e| e.eval(&coords)).collect();
        assert!(next.iter().all(|&v| v >= 0), "negative parent index at hop {k}");
        coords = next.iter().map(|&v| v as usize).collect();
    }
    coords.iter().map(|&v| v as i64).collect()
}

/// TRANSPOSE CONSUMER: x(2,3) permuted then multiplied. The searched
/// plan must fold the permute and hand the mul a swap map.
#[test]
fn transpose_consumer_folds_and_carries_the_swap_map() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize));
    let c = cx.tensor((3usize, 2usize));
    let _out = (x.permute((1, 0)) * c).output();

    let plan = plan_for(
        &cx,
        &[
            (x.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
            (c.id, vec![1.0f32; 6].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    // One real kernel (the mul), no copies, three buffers (x, c, out).
    assert_eq!(computes, 1, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 3, "plan shape drifted:\n{}", plan.summary());
    assert!(!composed.is_empty(), "no operand carries composed access:\n{}", plan.summary());
    let (label, slot, access) = &composed[0];
    assert_eq!(label, "MulFunctionalGeneric");
    let last = access.hops.last().unwrap();
    assert_eq!(last.parent_dims.as_deref(), Some(&[2i64, 3][..]));
    for i in 0..3usize {
        for j in 0..2usize {
            assert_eq!(
                chain_eval(access, &[i, j]),
                vec![j as i64, i as i64],
                "transpose: mul operand {slot} out ({i},{j}) must address parent ({j},{i})"
            );
        }
    }
}

/// SLICE CONSUMER: rows 1..3 of a (4,6), multiplied. Fold + offset map.
#[test]
fn slice_consumer_folds_and_carries_the_offset_map() {
    let mut cx = Graph::new();
    let x = cx.tensor((4usize, 6usize));
    let c = cx.tensor((2usize, 6usize));
    let _out = (x.slice((1..3, ..)) * c).output();

    let plan = plan_for(
        &cx,
        &[
            (x.id, (0..24).map(|v| v as f32).collect::<Vec<f32>>().into()),
            (c.id, vec![1.0f32; 12].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    assert_eq!(computes, 1, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 3, "plan shape drifted:\n{}", plan.summary());
    assert!(!composed.is_empty(), "no operand carries composed access:\n{}", plan.summary());
    let (_, _, access) = &composed[0];
    let last = access.hops.last().unwrap();
    assert_eq!(last.parent_dims.as_deref(), Some(&[4i64, 6][..]));
    for i in 0..2usize {
        for j in 0..6usize {
            assert_eq!(
                chain_eval(access, &[i, j]),
                vec![i as i64 + 1, j as i64],
                "slice: out ({i},{j}) must address parent ({},{j})",
                i + 1
            );
        }
    }
}

/// BROADCAST CONSUMER: a (3,) row broadcast over (2,3), multiplied.
/// Views read through non-injective maps legally (stride-0 axis).
#[test]
fn broadcast_consumer_folds_and_carries_the_stride0_map() {
    let mut cx = Graph::new();
    let x = cx.tensor(3usize);
    let c = cx.tensor((2usize, 3usize));
    let _out = (x.expand_dim(0, 2) * c).output();

    let plan = plan_for(
        &cx,
        &[
            (x.id, vec![1.0f32, 2., 3.].into()),
            (c.id, vec![1.0f32; 6].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    assert_eq!(computes, 1, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 3, "plan shape drifted:\n{}", plan.summary());
    assert!(!composed.is_empty(), "no operand carries composed access:\n{}", plan.summary());
    let (_, _, access) = &composed[0];
    let last = access.hops.last().unwrap();
    assert_eq!(last.parent_dims.as_deref(), Some(&[3i64][..]));
    for i in 0..2usize {
        for j in 0..3usize {
            assert_eq!(
                chain_eval(access, &[i, j]),
                vec![j as i64],
                "broadcast: out ({i},{j}) must address parent ({j}) for every i"
            );
        }
    }
}

/// CHAINED-MATMUL-SHAPED: (a·b)·c through the decomposed frontend
/// spelling (expand/permute movement + mul + sum at both stages). All
/// movement is foldable, so the plan is exactly the four kernels —
/// two muls, two reduces — with zero materializes and zero copies.
#[test]
fn chained_matmul_folds_all_movement() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize));
    let b = cx.tensor((3usize, 4usize));
    let c = cx.tensor((4usize, 2usize));
    let _out = a.matmul(b).matmul(c).output();

    let plan = plan_for(
        &cx,
        &[
            (a.id, vec![1.0f32; 6].into()),
            (b.id, vec![1.0f32; 12].into()),
            (c.id, vec![1.0f32; 8].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    // 2 broadcast-muls + 2 reduces; inputs a,b,c + the four kernel
    // results (out is the last reduce's destination) = 7 buffers.
    assert_eq!(computes, 4, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 7, "plan shape drifted:\n{}", plan.summary());
    // Both muls read at least one operand through a composed access
    // (the expand_dim broadcasts and the rhs permute+expand).
    let mul_slots = composed
        .iter()
        .filter(|(label, _, _)| label == "MulFunctionalGeneric")
        .count();
    assert!(
        mul_slots >= 2,
        "expected both broadcast-muls to read through folds, got {mul_slots}:\n{}",
        plan.summary()
    );
}
