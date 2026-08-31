//! ATTACKER C — P1+P2 COMPOSITION PROBES (Austin 2026-08-26 review).
//!
//! P1 (donation through zero-movement views) cannot be run — it is not
//! implemented. These probes therefore pin, against the CURRENT planner, the
//! exact composed-world properties both briefs rely on, by constructing the
//! post-P1 plan shapes with the machinery that exists today (dest-tie
//! seeding, which is P1's mechanism minus the view hop):
//!
//!  * diamond: a value seeded into a BOUND buffer that is simultaneously the
//!    alias parent of a mid-graph view with its own consumer — is the bound
//!    buffer protected from later writers, and does the plan certify?
//!  * same-e-class input/output cohabitation under a seed (P1 brief's unrun
//!    case (a)): rejection without a May permit, admission with one.
//!  * the composed "mm-chain under P1" plan shape: interior folded view
//!    between two seeded/allocated writers, terminal op writing the bound
//!    buffer directly — full certificate run.
//!  * DETERMINISM (M4 Phase 1, approved 2026-08-26): buffer dims are now a
//!    WRITER-IDENTITY join — computed only from the residents that supply
//!    the buffer's bytes (staged inputs, memory-writing compute results,
//!    copy destinations), iterated in deterministic plan-node order. The
//!    old first-wins join over std-HashMap iteration of `value_buffer`
//!    (measured nondeterministic by this file's original probe) is gone;
//!    the flipped probe below pins the new law, and two new gates pin the
//!    conflicting-writer loud bail and the per-node descriptor schema.
use luminal::bufferize::{BufferId, BufferNode};
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{
    Access, BufferInfo, ExtractedDag, ExtractedEdge, ExtractedGraph, ExtractedNode, FreedBy,
    InputNode, LayoutInfo, LayoutTensorInfo, LogicalInfo, OpInput, OpNode, OutputNode, OutputSlot,
    Provenance,
};
use luminal::prelude::petgraph;
use luminal::test_support::{EmptyOp, MockOp, MockView, TestGraph};
use petgraph::algo::has_path_connecting;
use petgraph::graph::NodeIndex;

fn computes<'a>(
    plan: &'a luminal::bufferize::BufferIrGraph<luminal::test_support::MockLayout>,
    label: &str,
) -> Vec<(NodeIndex, Vec<BufferId>, Vec<BufferId>)> {
    plan.dag
        .node_indices()
        .filter_map(|i| match &plan.dag[i] {
            BufferNode::Compute { op, reads, writes, .. } if op.label() == label => {
                Some((i, reads.clone(), writes.clone()))
            }
            _ => None,
        })
        .collect()
}

fn copies(plan: &luminal::bufferize::BufferIrGraph<luminal::test_support::MockLayout>) -> Vec<(NodeIndex, BufferId, BufferId)> {
    plan.dag
        .node_indices()
        .filter_map(|i| match &plan.dag[i] {
            BufferNode::BufferCopy { src, dst, .. } => Some((i, src.clone(), dst.clone())),
            _ => None,
        })
        .collect()
}

/// ATTACK 1+2 (the diamond, dest-tie analogue of the P1+P2 world): value r is
/// SEEDED into bound buffer D (the exact post-P1 residence), a view of r
/// aliases D into the middle of the graph, a consumer reads through it, and an
/// in-place accumulator then attacks the shared bound buffer. Required
/// outcomes: (i) the bound buffer becomes the view's alias parent (value_buffer
/// maps the view's value to the Boundary id); (ii) the accumulator's write is
/// VETOED (r's END_OF_PROGRAM slot read never happens-before anything,
/// bufferize.rs:837-842) and repaired out-of-place; (iii) D keeps exactly one
/// writer; (iv) the whole plan still certifies.
#[test]
fn diamond_bound_buffer_is_alias_parent_and_survives_writer_attack() {
    let mut g = TestGraph::new();
    let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")]).remove(0);
    // chain root writer: writes r into e's storage (dest tie), seeded to D
    let r = g
        .op(
            Box::new(MockOp { reads: vec![false], in_place_operand: Some(0), not_conflicting: false }),
            &[&e],
            &[("r", "rm")],
        )
        .remove(0);
    // the mid-graph view of the seeded value: aliases the BOUND buffer
    let v = g.op(Box::new(MockView), &[&r], &[("v", "view")]).remove(0);
    // consumer through the view
    let c = g
        .op(
            Box::new(MockOp { reads: vec![true], in_place_operand: None, not_conflicting: false }),
            &[&v],
            &[("c", "rm")],
        )
        .remove(0);
    // the attacker: an accumulator demanding to overwrite the shared storage
    let a = g
        .op(
            Box::new(MockOp { reads: vec![true], in_place_operand: Some(0), not_conflicting: false }),
            &[&v],
            &[("a", "rm")],
        )
        .remove(0);
    g.output(&r, "D"); // seed proposal: e -> D
    g.output(&c, "E");
    g.output(&a, "F");
    let plan = luminal::test_support::bufferize_mock(&g.build()).expect("the diamond must still certify");
    println!("{}", plan.summary());

    // (i) seed applied; the view's value resides in the BOUND buffer.
    let d = plan.value_buffer[&r].clone();
    assert!(matches!(d, BufferId::Boundary(_)), "r seeded into D:\n{}", plan.summary());
    assert_eq!(plan.value_buffer[&v], d, "the bound buffer is the view's alias parent");
    assert_eq!(plan.value_buffer[&e], d, "chain root poison rides D");

    // (ii)+(iii) the accumulator was rejected: it writes fresh storage, and D
    // has exactly one writing compute (the seeded chain-root writer).
    let mocks = computes(&plan, "MockOp");
    let writers_of_d: Vec<_> = mocks.iter().filter(|(_, _, w)| w.contains(&d)).collect();
    assert_eq!(writers_of_d.len(), 1, "exactly one writer of the bound buffer");
    assert!(
        writers_of_d[0].1.is_empty() || writers_of_d[0].1[0] != d || writers_of_d[0].2[0] == d,
        "the sole writer is the chain root"
    );
    // RULING 2026-08-27 (repair destinations are fresh single-writer
    // buffers): the accumulator's operand is a FOLDED view, so its repair
    // copy lands the parent bytes in a FRESHLY minted buffer the
    // accumulator READS (re-rooted fold) while WRITING its own result
    // buffer — the two are distinct Allocated buffers now, where the
    // pre-ruling plan copied into the result buffer itself.
    let acc = mocks
        .iter()
        .find(|(_, reads, writes)| {
            !reads.is_empty()
                && reads[0] != d
                && reads[0] != writes[0]
                && matches!(reads[0], BufferId::Allocated(_))
                && matches!(writes[0], BufferId::Allocated(_))
        })
        .expect("the repaired accumulator reads the fresh repair buffer, writes its own");
    // its bytes were carried out of D by a repair copy into the buffer it READS
    assert!(
        copies(&plan).iter().any(|(_, src, dst)| *src == d && *dst == acc.1[0]),
        "repair copy D -> the fresh single-writer repair buffer:\n{}",
        plan.summary()
    );
    // no BufferCopy writes D either (no delivery needed: src == dest elided)
    assert!(
        copies(&plan).iter().all(|(_, _, dst)| *dst != d),
        "nothing else writes the bound buffer:\n{}",
        plan.summary()
    );

    // (iv) the consumer reads the bound buffer, data-ordered after the writer.
    let consumer = mocks
        .iter()
        .find(|(_, reads, writes)| !reads.is_empty() && reads[0] == d && writes[0] != d)
        .expect("consumer reads D through the folded view");
    let writer_idx = writers_of_d[0].0;
    assert!(
        has_path_connecting(&plan.dag, writer_idx, consumer.0, None),
        "consumer ordered after the bound buffer's writer"
    );
}

/// ATTACK 2 hard case (P1 brief's unrun (a), same-e-class): the bound output
/// buffer IS an input buffer. A seed into it makes the chain-root op write the
/// buffer its other operand reads — same-op read, excused ONLY by a trusted
/// May permit (bufferize.rs:1040-1055). Without the permit the seed must
/// degrade to exactly the copy plan; with it, zero-copy in-place.
#[test]
fn seed_into_cohabited_input_buffer_needs_may_permit() {
    // (1) no permit: rejected, degrade to copy
    let mut g = TestGraph::new();
    let x = g.input("x", "D", Access::ReadWrite, "rm");
    let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")]).remove(0);
    let r = g
        .op(
            Box::new(MockOp { reads: vec![true, false], in_place_operand: Some(1), not_conflicting: false }),
            &[&x, &e],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "D");
    let plan = luminal::test_support::bufferize_mock(&g.build()).expect("must certify (degraded)");
    println!("no-permit:\n{}", plan.summary());
    assert!(
        matches!(plan.value_buffer[&r], BufferId::Allocated(_)),
        "seed rejected: r relocates to fresh storage:\n{}",
        plan.summary()
    );
    let cps = copies(&plan);
    assert_eq!(cps.len(), 1, "one boundary delivery copy:\n{}", plan.summary());
    assert_eq!(cps[0].2, plan.value_buffer[&x], "the delivery overwrites D");

    // (2) with the trusted permit: admitted, zero copies, computes in caller storage
    let mut g = TestGraph::new();
    let x = g.input("x", "D", Access::ReadWrite, "rm");
    let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")]).remove(0);
    let r = g
        .op(
            Box::new(MockOp { reads: vec![true, false], in_place_operand: Some(1), not_conflicting: true }),
            &[&x, &e],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "D");
    let plan = luminal::test_support::bufferize_mock(&g.build()).expect("must certify (admitted)");
    println!("with-permit:\n{}", plan.summary());
    assert!(
        matches!(plan.value_buffer[&r], BufferId::Boundary(_)),
        "seed admitted under the May permit:\n{}",
        plan.summary()
    );
    assert_eq!(copies(&plan).len(), 0, "zero copies:\n{}", plan.summary());
}

/// ATTACK 3 (chained writers with an interior folded view — the composed
/// "mm.mm under P1" plan shape, built with today's machinery): writer m1 into
/// alloc storage, folded view of m1's result, terminal writer m2 reading
/// through the view and SEEDED to write the bound buffer directly. The full
/// pipeline (analysis, anti edges, optimize, certificate, lowering tripwires)
/// must accept: one alloc, one free, zero copies, free after m2's read.
#[test]
fn chain_with_interior_view_and_terminal_bound_write_certifies() {
    let mut g = TestGraph::new();
    let x = g.input("x", "X", Access::ReadOnly, "rm");
    let e1 = g.op(Box::new(EmptyOp), &[], &[("e1", "rm")]).remove(0);
    let m1 = g
        .op(
            Box::new(MockOp { reads: vec![true, false], in_place_operand: Some(1), not_conflicting: false }),
            &[&x, &e1],
            &[("m1", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&m1], &[("v", "view")]).remove(0);
    let e2 = g.op(Box::new(EmptyOp), &[], &[("e2", "rm")]).remove(0);
    let m2 = g
        .op(
            Box::new(MockOp { reads: vec![true, false], in_place_operand: Some(1), not_conflicting: false }),
            &[&v, &e2],
            &[("m2", "rm")],
        )
        .remove(0);
    g.output(&m2, "D");
    let plan = luminal::test_support::bufferize_mock(&g.build()).expect("composed chain must certify");
    println!("{}", plan.summary());

    assert_eq!(copies(&plan).len(), 0, "zero copies:\n{}", plan.summary());
    let m1_buf = plan.value_buffer[&m1].clone();
    assert!(matches!(m1_buf, BufferId::Allocated(_)));
    assert_eq!(plan.value_buffer[&v], m1_buf, "view folded onto m1's storage");
    assert!(
        matches!(plan.value_buffer[&m2], BufferId::Boundary(_)),
        "terminal writer computes directly into the bound buffer:\n{}",
        plan.summary()
    );
    // m2 reads m1's alloc THROUGH the view and writes the boundary: the exact
    // producer->consumer-across-buffers shape the composed P1+P2 world makes.
    let mocks = computes(&plan, "MockOp");
    let m2_node = mocks
        .iter()
        .find(|(_, _, w)| matches!(w[0], BufferId::Boundary(_)))
        .expect("terminal writer");
    assert_eq!(m2_node.1[0], m1_buf, "m2 reads m1's buffer via the folded view");
    // one alloc, one free, and the free is ordered after m2's read.
    let allocs = computes(&plan, "BufferAlloc");
    let frees = computes(&plan, "BufferFree");
    assert_eq!(allocs.len(), 1);
    assert_eq!(frees.len(), 1);
    assert_eq!(frees[0].1[0], m1_buf);
    assert!(
        has_path_connecting(&plan.dag, m2_node.0, frees[0].0, None),
        "m1's buffer freed only after the through-view reader"
    );
}

// ---------------------------------------------------------------------------
// ATTACK 4 (determinism): hand-built graphs with REAL dims, so the geometry
// annotation actually runs its dims join — since M4 Phase 1 the
// WRITER-IDENTITY join (writers vote in plan-node order; view readers never
// vote; conflicting writers bail loudly).
// ---------------------------------------------------------------------------

struct DimsGraph {
    dag: ExtractedDag,
    producers: std::collections::HashMap<luminal::prelude::egraph_serialize::ClassId, NodeIndex>,
    slots: Vec<OutputSlot>,
    next: u32,
}

impl DimsGraph {
    fn new() -> Self {
        DimsGraph {
            dag: ExtractedDag::new(),
            producers: Default::default(),
            slots: Vec::new(),
            next: 0,
        }
    }
    fn fresh(&mut self) -> u32 {
        self.next += 1;
        self.next
    }
    fn value(&self, name: &str, layout: &str, dims: &[i64]) -> LayoutTensorInfo {
        LayoutTensorInfo {
            eclass: luminal::prelude::egraph_serialize::ClassId::from(format!("val${name}")),
            label: name.to_string(),
            tooltip: String::new(),
            shape: None,
            dtype: None,
            dtype_enum: Some(PlanDtype::F32),
            dims: Some(dims.to_vec()),
            element_bits: Some(32),
            logical: LogicalInfo {
                eclass: luminal::prelude::egraph_serialize::ClassId::from(format!("logical${name}")),
                label: name.to_string(),
                tooltip: String::new(),
                op: None,
                children: Vec::new(),
            },
            layout: LayoutInfo {
                eclass: luminal::prelude::egraph_serialize::ClassId::from(format!("layout${layout}")),
                label: layout.to_string(),
                tooltip: String::new(),
            },
        }
    }
    fn buffer(&mut self, name: &str) -> BufferInfo {
        let n = self.fresh();
        BufferInfo {
            lit: None,
            tensor_eclass: luminal::prelude::egraph_serialize::ClassId::from(format!("buftensor${n}")),
            tensor_label: name.to_string(),
            tensor_tooltip: String::new(),
            id_eclass: luminal::prelude::egraph_serialize::ClassId::from(format!("buf${name}")),
            id_label: name.to_string(),
            id_tooltip: String::new(),
            access: Some(Access::ReadWrite),
            freed_by: Some(FreedBy::Caller),
        }
    }
    fn input(&mut self, name: &str, buffer: &str, dims: &[i64]) -> LayoutTensorInfo {
        let value = self.value(name, "rm", dims);
        let buffer = self.buffer(buffer);
        let node = self
            .dag
            .add_node(ExtractedNode::BufferInput(InputNode { value: value.clone(), buffer }));
        self.producers.insert(value.eclass.clone(), node);
        value
    }
    fn op(
        &mut self,
        iface: Box<dyn luminal::layout_ir::LayoutIrOp>,
        inputs: &[&LayoutTensorInfo],
        name: &str,
        layout: &str,
        dims: &[i64],
    ) -> LayoutTensorInfo {
        let n = self.fresh();
        let out = self.value(name, layout, dims);
        let op_inputs: Vec<OpInput> = inputs
            .iter()
            .enumerate()
            .map(|(i, value)| OpInput { port: format!("in{i}"), value: value.eclass.clone() })
            .collect();
        let node = self.dag.add_node(ExtractedNode::LayoutOp(OpNode {
            op: iface,
            provenance: Provenance::Synthesized { id: n },
            inputs: op_inputs,
            outputs: vec![out.clone()],
            tooltip: String::new(),
            heuristic_cost: 1,
        }));
        for (i, value) in inputs.iter().enumerate() {
            let producer = *self.producers.get(&value.eclass).expect("operand producer");
            self.dag.add_edge(
                producer,
                node,
                ExtractedEdge { value: value.eclass.clone(), port: format!("in{i}") },
            );
        }
        self.producers.insert(out.eclass.clone(), node);
        out
    }
    fn output(&mut self, value: &luminal::prelude::egraph_serialize::ClassId, buffer: &str) {
        let index = self.slots.len();
        let buffer = self.buffer(buffer);
        self.slots.push(OutputSlot { index, value: value.clone(), buffer });
    }
    fn build(mut self) -> ExtractedGraph {
        let slots = std::mem::take(&mut self.slots);
        let node = self.dag.add_node(ExtractedNode::BufferOutput(OutputNode {
            eclass: luminal::prelude::egraph_serialize::ClassId::from("output$0"),
            label: "output".to_string(),
            tooltip: String::new(),
            slots: slots.clone(),
        }));
        for slot in &slots {
            let producer = *self.producers.get(&slot.value).expect("output producer");
            self.dag.add_edge(
                producer,
                node,
                ExtractedEdge { value: slot.value.clone(), port: format!("out {}", slot.index) },
            );
        }
        if let ExtractedNode::BufferOutput(output) = &mut self.dag[node] {
            output.slots = slots;
        }
        ExtractedGraph { dag: self.dag, outputs: vec![node] }
    }
}

/// FLIPPED PIN (was: the first-wins nondeterminism measurement). A view
/// whose numel DIFFERS from its parent's (a broadcast-flavored view),
/// admitted onto the parent's CALLER-PINNED input buffer — the exact
/// resident mix P2-on-real-backends creates. Under the WRITER-IDENTITY
/// dims join (M4 Phase 1, approved 2026-08-26) the caller buffer's dims
/// come ONLY from the resident that supplies its bytes — the staged input
/// x at (2,3). The view reads THROUGH the buffer at (2,2,3) but produces
/// no plan node and can never vote. We bufferize the SAME graph 60 times
/// in one process: the old probe measured first-wins flicker between
/// (2,3) and (2,2,3); the law now demands exactly {(2,3)}, every run.
#[test]
fn caller_buffer_dims_writer_identity_join_is_deterministic() {
    let mut outcomes: std::collections::BTreeSet<Vec<i64>> = Default::default();
    for _ in 0..60 {
        let mut g = DimsGraph::new();
        let x = g.input("x", "X", &[2, 3]);
        let v = g.op(Box::new(MockView), &[&x], "v", "view", &[2, 2, 3]);
        let c = g.op(
            Box::new(MockOp { reads: vec![true], in_place_operand: None, not_conflicting: false }),
            &[&v],
            "c",
            "rm",
            &[2, 2, 3],
        );
        g.output(&c.eclass, "E");
        let plan = luminal::test_support::bufferize_mock(&g.build()).expect("bufferizes");
        // sanity: the view really is resident in x's caller buffer
        let xbuf = plan.value_buffer[&x.eclass].clone();
        assert!(matches!(xbuf, BufferId::Boundary(_)));
        assert_eq!(plan.value_buffer[&v.eclass], xbuf, "view admitted onto the pinned buffer");
        let dims = plan.buffers[&xbuf].dims.clone().expect("annotated");
        outcomes.insert(dims);
    }
    println!("distinct dims outcomes for the caller buffer over 60 runs: {outcomes:?}");
    // THE PIN: writer identity, not resident hash-order. The staged input
    // is the buffer's only writer, so its geometry — and only its geometry —
    // names the storage, in every run.
    assert_eq!(
        outcomes,
        std::collections::BTreeSet::from([vec![2, 3]]),
        "the caller buffer's dims must be the WRITER's geometry, every run"
    );
}

/// GATE (conflicting writers): two writers of one buffer that disagree on
/// geometry are a planner contradiction and must BAIL LOUDLY, both writers
/// named — never resolved by iteration order. Recipe: an in-place
/// accumulator admitted onto the caller's ReadWrite input buffer (the
/// attack-3c admission from the P2 probes) whose result claims DIFFERENT
/// dims than the staged input — the buffer then has two disagreeing
/// writers: the input stage at (2,3) and the compute result at (3,4).
#[test]
fn conflicting_writers_bail_loudly_with_both_named() {
    let mut g = DimsGraph::new();
    let x = g.input("x", "B", &[2, 3]);
    let r = g.op(
        Box::new(MockOp { reads: vec![true], in_place_operand: Some(0), not_conflicting: false }),
        &[&x],
        "r",
        "rm",
        &[3, 4],
    );
    g.output(&r.eclass, "E");
    let err = luminal::test_support::bufferize_mock(&g.build()).expect_err(
        "two writers with disagreeing geometry on one buffer must be rejected",
    );
    let text = format!("{err:#}");
    println!("rejected: {text}");
    assert!(
        text.contains("buffer geometry contradiction"),
        "the writer-join door must name itself: {text}"
    );
    assert!(
        text.contains("[2, 3]") && text.contains("[3, 4]"),
        "both writers' dims must be named: {text}"
    );
}

/// GATE (per-node descriptor schema, approved 2026-08-26b): every lowered
/// Compute node carries per-slot operand/result descriptors — value +
/// buffer identity from the BufferTensor slots, dims/dtype from the
/// extraction — parallel to `reads`/`writes`. Since M4 Phase 3 the
/// composed-access slot is FILLED at view-fold time: the view-reading
/// operand carries the fold's hop (fail-closed `entries: None` here —
/// `MockView` exposes no numeric map), result slots stay direct.
/// BufferCopy nodes carry the copied value. In this all-literal fixture,
/// every MockOp slot must be FILLED.
#[test]
fn every_compute_node_carries_filled_slot_descriptors() {
    let mut g = DimsGraph::new();
    let x = g.input("x", "X", &[2, 3]);
    let v = g.op(Box::new(MockView), &[&x], "v", "view", &[2, 2, 3]);
    let c = g.op(
        Box::new(MockOp { reads: vec![true], in_place_operand: None, not_conflicting: false }),
        &[&v],
        "c",
        "rm",
        &[2, 2, 3],
    );
    g.output(&c.eclass, "E");
    let plan = luminal::test_support::bufferize_mock(&g.build()).expect("bufferizes");

    let mut computes = 0usize;
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::Compute { op, reads, writes, operand_info, result_info, .. } => {
                computes += 1;
                assert_eq!(operand_info.len(), reads.len(), "{}: operand_info parallels reads", op.label());
                assert_eq!(result_info.len(), writes.len(), "{}: result_info parallels writes", op.label());
                for (slot, id) in operand_info.iter().zip(reads) {
                    assert_eq!(&slot.buffer, id, "{}: descriptor buffer = reads entry", op.label());
                }
                for (slot, id) in result_info.iter().zip(writes) {
                    assert_eq!(&slot.buffer, id, "{}: descriptor buffer = writes entry", op.label());
                }
                if op.label() == "MockOp" {
                    for slot in operand_info.iter().chain(result_info) {
                        assert!(slot.dims.is_some(), "MockOp slot dims filled (value {})", slot.value);
                        assert!(slot.dtype.is_some(), "MockOp slot dtype filled (value {})", slot.value);
                        assert!(slot.element_bits.is_some(), "MockOp slot bits filled (value {})", slot.value);
                    }
                    // Phase 3: the operand READS THROUGH the folded view, so
                    // its descriptor records the fold — one hop, fail-closed
                    // entries (MockView has no numeric map), parent dims =
                    // x's literal extents. Results are produced here: direct.
                    let access = operand_info[0]
                        .composed_access
                        .as_ref()
                        .expect("the view-reading operand carries the fold's composed access");
                    assert_eq!(access.hops.len(), 1, "one folded view, one hop");
                    assert_eq!(access.hops[0].entries, None, "MockView is mapless: fail-closed");
                    assert_eq!(access.hops[0].parent_dims, Some(vec![2, 3]), "parent extents ride the hop");
                    for slot in result_info {
                        assert!(slot.composed_access.is_none(), "results are never folds");
                    }
                }
            }
            BufferNode::BufferCopy { value, .. } => {
                assert_eq!(value, &c.eclass, "the delivery copy names the value it transports");
            }
            _ => {}
        }
    }
    assert!(computes >= 1, "fixture must lower at least one compute:\n{}", plan.summary());
    // The consumer reads the view VALUE (v) out of the parent's buffer —
    // the descriptor records the through-view identity the composed-access
    // slot will describe in Phase 3.
    let consumer = plan
        .dag
        .node_weights()
        .find_map(|n| match n {
            BufferNode::Compute { op, operand_info, .. } if op.label() == "MockOp" => {
                Some(operand_info[0].clone())
            }
            _ => None,
        })
        .expect("the consumer compute");
    assert_eq!(consumer.value, v.eclass, "operand descriptor names the VIEW value");
    assert_eq!(
        consumer.buffer,
        plan.value_buffer[&x.eclass],
        "…resident in the parent's caller buffer"
    );
    assert_eq!(consumer.dims, Some(vec![2, 2, 3]), "…at the view's own geometry");
}

/// COMPOSED DOOR (attack 1+4, P1xP2): an output slot bound to a VIEW OF A
/// POISON. The poison ledger is producer-keyed (bufferize.rs:1150-1159), so
/// the slot-binds-poison rejection (bufferize.rs:1236-1244) is laundered by
/// one view. TODAY the seed walk stops at the view (bufferize.rs:658-660), the
/// poison lands in System storage, and the undefined-delivery program either
/// bails or panics in optimize (the freed-never-written buffer). UNDER P1 the
/// walk would cross the view to the poison root with ZERO writing hops, the
/// poison would fold on the BOUNDARY buffer (buffer_tensor_ir.rs:1219-1221),
/// no free exists to panic on — and the caller receives undefined bytes
/// SILENTLY. This probe pins today's (loud-ish) behavior as the baseline P1
/// must not degrade.
#[test]
fn output_slot_bound_to_view_of_poison_current_behavior() {
    let graph = {
        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")]).remove(0);
        let v = g.op(Box::new(MockView), &[&e], &[("v", "view")]).remove(0);
        g.output(&v, "D");
        g.build()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| luminal::test_support::bufferize_mock(&graph)));
    match result {
        Ok(Ok(plan)) => panic!(
            "SILENT: undefined bytes delivered to a bound output today:\n{}",
            plan.summary()
        ),
        Ok(Err(e)) => println!("REJECTED loudly today: {e:#}"),
        Err(_) => println!(
            "PANIC today (freed-never-written System buffer, \
             buffer_tensor_ir.rs:1353) — loud, but a panic where the \
             discipline demands a Result; under P1 even this disappears"
        ),
    }
}
