//! THE VIEW OP's estate, rehomed from core's `src/test_support.rs`.
//!
//! `IndexMapApplyView` is the only op declaring BOTH `operand_reads_memory
//! == false` and `result_writes_memory == false` — a metadata op that
//! moves no bytes. That declaration shape is what drives the bufferizer's
//! view fold, its non-writing-tie ordering, and the `validate_plan` arm
//! that refuses an unfolded view. The reference runtime has no kernel for
//! it and never registered it, so its witnesses belong here, on the
//! vocabulary that does.
//!
//! Adapted from the pre-move bodies: paths retargeted to the DEP-WORLD
//! (`luminal::`, the same build this crate links) and the fixture runners
//! pointed at this crate's own `fixtures/` — including its forked copy of
//! `basic_program.egg`. Nothing here reaches into the core script tree.

use luminal::bufferize;
use luminal::layout_ir::Access;
use luminal::test_support::{MockOp, TestGraph};

use test_runtime::IndexMapApplyView;

/// THE REAL VIEW OP, plan level (Step 3): `IndexMapApplyView` feeding a
/// compute op contributes ZERO plan nodes — the result binds its parent's
/// buffer and the consumer's kernel reads that storage directly. Same
/// shape as core's `MockView` pin (which stays beside the engine, in
/// `src/bufferize.rs`'s suite), but proving the SHIPPED op's declared
/// contract — un-read operand, un-written result, Must(0→0) — drives the
/// fold. Mock and real op are deliberately redundant: one pins the
/// engine, the other pins that a matcher-authored op reaches it.
#[test]
fn real_view_op_feeds_compute_with_zero_plan_nodes() {
    use luminal::bufferize::BufferNode;

    let mut g = TestGraph::new();
    let x = g.input("x", "B", Access::ReadWrite, "rm");
    let v = g.op(Box::new(IndexMapApplyView), &[&x], &[("v", "row0")])[0].clone();
    let r = g.op(
        Box::new(MockOp {
            reads: vec![true],
            ..Default::default()
        }),
        &[&v],
        &[("r", "rm")],
    )[0]
    .clone();
    g.output(&r, "D");
    let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

    assert_eq!(
        plan.value_buffer[&v],
        plan.value_buffer[&x],
        "the view derives its parent's buffer:\n{}",
        plan.summary()
    );
    for idx in plan.dag.node_indices() {
        if let BufferNode::Compute { op, reads, .. } = &plan.dag[idx] {
            assert_ne!(
                op.label(),
                "IndexMapApplyViewGeneric",
                "views are folded:\n{}",
                plan.summary()
            );
            if op.label() == "MockOp" {
                assert_eq!(
                    reads[0], plan.value_buffer[&x],
                    "the consumer reads the parent buffer directly"
                );
            }
        }
    }
}

/// THE REAL VIEW OP bound to an output slot on a DIFFERENT buffer than
/// its parent's: the Must tie binds the view into the parent's storage,
/// and the boundary promise is honored by exactly one BufferCopy into the
/// slot's buffer — the accepted price of returning a view (span-aware
/// seeding through views is the recorded future refinement). The view
/// itself still contributes no compute node.
#[test]
fn real_view_op_to_output_slot_pays_a_boundary_copy() {
    use luminal::bufferize::BufferNode;

    let mut g = TestGraph::new();
    let x = g.input("x", "B", Access::ReadWrite, "rm");
    let v = g.op(Box::new(IndexMapApplyView), &[&x], &[("v", "row0")])[0].clone();
    g.output(&v, "D");
    let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

    assert_eq!(
        plan.value_buffer[&v],
        plan.value_buffer[&x],
        "the view value lives in its parent's buffer:\n{}",
        plan.summary()
    );
    let copies: Vec<_> = plan
        .dag
        .node_indices()
        .filter_map(|idx| match &plan.dag[idx] {
            BufferNode::BufferCopy { src, dst } => Some((src.clone(), dst.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        copies.len(),
        1,
        "exactly one boundary copy honors the slot:\n{}",
        plan.summary()
    );
    assert_eq!(
        copies[0].0, plan.value_buffer[&x],
        "copied from the parent's buffer"
    );
    assert!(
        plan.dag
            .node_indices()
            .all(|idx| !matches!(&plan.dag[idx], BufferNode::Compute { .. })),
        "no kernel runs — a view plus a transport:\n{}",
        plan.summary()
    );
}

/// STAGE 7 / STEP 4, the view-feeds-compute boundary fixture end to end:
/// the transpose view of the READ-ONLY input extracts as the zero-cost
/// view op and folds to zero plan nodes, so the whole program is ONE
/// kernel — Sqrt reading the caller's input buffer directly (specialized
/// against the composed layout) and writing its seeded output buffer.
/// Zero allocations, zero copies.
#[test]
fn view_feeds_compute_fixture_runs_one_kernel_on_the_input_buffer() {
    use luminal::bufferize::{BufferId, BufferNode};

    let graph = test_runtime::extract_fixture_by_name("boundary_view_feeds_compute.egg");
    let plan = bufferize::bufferize(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

    assert!(
        plan.buffers
            .keys()
            .all(|id| matches!(id, BufferId::Boundary(_))),
        "zero allocations:\n{}",
        plan.summary()
    );
    let launch: Vec<BufferId> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::BufferInput { slots } => Some(slots.iter().map(|slot| slot.buffer.clone())),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(launch.len(), 1, "one caller input:\n{}", plan.summary());

    let mut computes = 0;
    for idx in plan.dag.node_indices() {
        match &plan.dag[idx] {
            BufferNode::BufferCopy { .. } => panic!("zero copies:\n{}", plan.summary()),
            BufferNode::Compute {
                op, reads, writes, ..
            } => {
                computes += 1;
                assert_eq!(op.label(), "SqrtFunctionalGeneric", "{}", plan.summary());
                assert_eq!(
                    reads[0], launch[0],
                    "the kernel reads the input buffer directly through the folded view"
                );
                assert_ne!(
                    writes[0],
                    launch[0],
                    "the read-only input is never written:\n{}",
                    plan.summary()
                );
            }
            _ => {}
        }
    }
    assert_eq!(
        computes,
        1,
        "the view contributed zero plan nodes:\n{}",
        plan.summary()
    );
}

/// STAGE 7 / STEP 4, the write-into-viewed-buffer boundary fixture: a
/// writer demanding the viewed buffer in place is REJECTED while a
/// view-reader is live (Exp is dataflow-independent of Sqrt, and the
/// analyzer is region-blind besides), so Sqrt degrades to a fresh
/// allocation and ONE boundary copy honors y@input-buffer — ordered
/// after Exp's read by the WAR anti edge. Exp still reads the viewed
/// buffer directly through the folded view.
#[test]
fn write_into_viewed_buffer_fixture_degrades_to_copy() {
    use luminal::bufferize::{BufferId, BufferNode};

    let graph = test_runtime::extract_fixture_by_name("boundary_write_into_viewed_buffer.egg");
    let plan = bufferize::bufferize(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

    let launch: Vec<BufferId> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::BufferInput { slots } => Some(slots.iter().map(|slot| slot.buffer.clone())),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(launch.len(), 1, "one caller input:\n{}", plan.summary());
    let viewed = launch[0].clone();

    let mut copies = Vec::new();
    let mut sqrt_writes = None;
    let mut exp_reads = None;
    for idx in plan.dag.node_indices() {
        match &plan.dag[idx] {
            BufferNode::BufferCopy { src, dst } => copies.push((src.clone(), dst.clone())),
            BufferNode::Compute {
                op, reads, writes, ..
            } => match op.label() {
                "SqrtFunctionalGeneric" => sqrt_writes = Some(writes[0].clone()),
                "ExpFunctionalGeneric" => exp_reads = Some(reads[0].clone()),
                "BufferAlloc" | "BufferFree" => {}
                other => panic!("unexpected kernel {other}:\n{}", plan.summary()),
            },
            _ => {}
        }
    }
    assert_eq!(
        exp_reads.expect("exp kernel present"),
        viewed,
        "exp reads the viewed buffer directly through the folded view"
    );
    let sqrt_writes = sqrt_writes.expect("sqrt kernel present");
    assert!(
        matches!(sqrt_writes, BufferId::Allocated(_)),
        "the in-place wish was rejected:\n{}",
        plan.summary()
    );
    assert_eq!(copies.len(), 1, "one boundary copy:\n{}", plan.summary());
    assert_eq!(
        copies[0].0, sqrt_writes,
        "copied from the rejected writer's allocation"
    );
    assert_eq!(
        copies[0].1, viewed,
        "into the demanded output slot's buffer"
    );
}

/// EXTRACTION PREFERS THE VIEW: where an IndexMapApply's consumer accepts
/// the COMPOSED layout, the free view op wins over the materializing
/// gather (declared-effect cost: 0 vs 2). In basic_program both apply
/// sites now extract as views — the transpose-onto-z site keeps a
/// layout-conversion CopyGeneric AFTER its view (z's output slot demands
/// a non-composed contiguous layout), and no Materialize survives
/// anywhere.
#[test]
fn extraction_prefers_the_view_op_where_the_layout_is_composed() {
    use luminal::layout_ir::ExtractedNode;

    let graph = test_runtime::extract_fixture_by_name("basic_program.egg");
    let mut views = 0;
    let mut materializes = 0;
    let mut conversion_copies = 0;
    for node in graph.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            match op.op.label() {
                "IndexMapApplyViewGeneric" => views += 1,
                "IndexMapApplyMaterialize" => materializes += 1,
                "CopyGeneric" => conversion_copies += 1,
                _ => {}
            }
        }
    }
    assert_eq!(views, 2, "both apply sites extract as views");
    assert_eq!(materializes, 0, "no materializing gather survives");
    assert_eq!(
        conversion_copies, 1,
        "the non-composed output slot re-lays-out through one copy kernel"
    );
}
