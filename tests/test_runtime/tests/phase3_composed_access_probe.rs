//! M4 PHASE 3 PLAN PIN: a folded view no longer discards its index map —
//! the consumer's operand SlotDescriptor carries the composed access
//! (entries parsed from the e-graph, enode-anchored, + the parent dims).
//! The test runtime admits views (harness matchers), so these plans are
//! CPU-exercisable; real backends still exclude views from their allow
//! lists, which is exactly the Phase-3 zero-behavior pin.
//!
//! Assertion discipline: hop COUNT is the e-graph's business (a chain may
//! reach the planner as stacked hops or as one welded apply — both are
//! legal spellings of the same access), so the chain is checked by
//! EVALLING it end-to-end against the hand-computed composite map, plus
//! the last hop's parent_dims against the true parent extents.

use luminal::bufferize::{BufferNode, ComposedAccess};
use luminal::graph::Graph;

/// Evaluate a composed-access chain at one out-coordinate: hops[0] is the
/// outermost fold (entries over the slot's own coordinates), each hop's
/// outputs are the next hop's coordinates; returns the final parent
/// coordinates. Panics (test-loud) on an unparsed hop.
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

/// Bufferize a frontend program with views preferred, and return every
/// (op label, operand slot index, access) that carries composed access.
fn composed_slots(text: &str, prefer: &[&str]) -> (luminal::bufferize::BufferIrGraph, Vec<(String, usize, ComposedAccess)>) {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, prefer);
    let dps = luminal::dps::dps_rewrite(&graph);
    let plan = luminal::bufferize::bufferize(&dps).expect("bufferize");
    let mut found = Vec::new();
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute { op, operand_info, result_info, .. } = node {
            for (slot, info) in operand_info.iter().enumerate() {
                if let Some(access) = &info.composed_access {
                    found.push((op.label().to_string(), slot, access.clone()));
                }
            }
            for info in result_info {
                assert!(
                    info.composed_access.is_none(),
                    "{}: a compute RESULT is produced by the node, never through a fold",
                    op.label()
                );
            }
        }
    }
    (plan, found)
}

/// TRANSPOSE VIEW: `x.permute((1,0))` feeding a mul. The folded view's
/// map must land on the mul's operand descriptor: out (i,j) -> parent
/// (j,i), parent dims (2,3).
#[test]
fn transpose_view_consumer_carries_the_swap_map() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((2usize, 3usize));
        let c = cx.tensor((3usize, 2usize));
        let _ = (x.permute((1, 0)) * c).output();
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    };
    let (plan, found) = composed_slots(&text, &["LayoutTensorOpIndexMapApplyViewGeneric"]);
    assert!(
        !found.is_empty(),
        "no operand carries composed access — was the view elected?\n{}",
        plan.summary()
    );
    let (label, slot, access) = &found[0];
    println!("composed access on {label} operand {slot}: {} hop(s)", access.hops.len());
    let last = access.hops.last().unwrap();
    assert_eq!(
        last.parent_dims.as_deref(),
        Some(&[2i64, 3][..]),
        "the innermost hop indexes into the true parent extents"
    );
    for i in 0..3usize {
        for j in 0..2usize {
            assert_eq!(
                chain_eval(access, &[i, j]),
                vec![j as i64, i as i64],
                "transpose: out ({i},{j}) must address parent ({j},{i})"
            );
        }
    }
}

/// 2-HOP CHAIN: slice rows 1..3 of a (4,6), then transpose — composite
/// out (a,b) -> parent (b+1, a). The chain may reach the planner stacked
/// (2 hops, un-normalized) or welded by the e-graph into one apply; the
/// EVALUATED composite and the innermost parent dims are the invariant.
#[test]
fn sliced_transpose_chain_composes_to_the_offset_swap() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 6usize));
        let c = cx.tensor((6usize, 2usize));
        let _ = (x.slice((1..3, ..)).permute((1, 0)) * c).output();
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    };
    let (plan, found) = composed_slots(&text, &["LayoutTensorOpIndexMapApplyViewGeneric"]);
    assert!(
        !found.is_empty(),
        "no operand carries composed access — was the view chain elected?\n{}",
        plan.summary()
    );
    let (label, slot, access) = &found[0];
    println!(
        "composed access on {label} operand {slot}: {} hop(s) (stacked or welded — both legal)",
        access.hops.len()
    );
    let last = access.hops.last().unwrap();
    assert_eq!(
        last.parent_dims.as_deref(),
        Some(&[4i64, 6][..]),
        "the innermost hop indexes into x's true extents"
    );
    for a in 0..6usize {
        for b in 0..2usize {
            assert_eq!(
                chain_eval(access, &[a, b]),
                vec![b as i64 + 1, a as i64],
                "slice+transpose: out ({a},{b}) must address parent ({},{a})",
                b + 1
            );
        }
    }
}

/// THE r10 CHAINED-MATMUL FIXTURE (CublasLt + views): the interior view
/// feeding the second call must carry parsed composed access on that
/// call's operand descriptor — the un-normalized planner-side record of
/// what the fold discarded before Phase 3.
#[test]
fn r10_chained_matmuls_carry_composed_access() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize));
        let w1 = cx.tensor((8usize, 3usize));
        let w2 = cx.tensor((3usize, 5usize));
        let _ = x.matmul(w1).matmul(w2).output();
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    };
    let (plan, found) = composed_slots(
        &text,
        &["LayoutTensorOpCublasLt", "LayoutTensorOpIndexMapApplyViewGeneric"],
    );
    println!(
        "r10 chain: {} composed slot(s): {:?}",
        found.len(),
        found.iter().map(|(l, s, a)| (l.clone(), *s, a.hops.len())).collect::<Vec<_>>()
    );
    assert!(
        !found.is_empty(),
        "the interior view folded but no descriptor records it\n{}",
        plan.summary()
    );
    for (label, slot, access) in &found {
        for (k, hop) in access.hops.iter().enumerate() {
            assert!(
                hop.entries.is_some(),
                "{label} operand {slot} hop {k}: entries beyond the parsed subset — \
                 the r10 view maps are plain coordinate maps and must parse"
            );
            assert!(
                hop.parent_dims.is_some(),
                "{label} operand {slot} hop {k}: parent dims must ride extraction"
            );
        }
    }
}
