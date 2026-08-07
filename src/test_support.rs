//! Test-only support: hand-author `ExtractedGraph`s at the level under test
//! (the analogue of writing a `.mlir` file by hand for MLIR's analysis tests),
//! plus a fixture path that runs a real egglog script through the extractor.
//!
//! Assignment rule: any case expressible with default-interface ops should be an
//! egg script tested via [`extract_fixture`]; the [`TestGraph`] builder is only
//! for cases *defined by* a non-default `Bufferizable` interface (declared
//! must-share ties, may-share permits, accumulators), which by design have no
//! egglog surface.

use std::collections::HashMap;
use std::fs;

use egraph_serialize::ClassId;
use petgraph::graph::NodeIndex;

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::extractor;
use crate::layout_ir::{
    AliasInfo, BufferInfo, Bufferizable, ExtractedDag, ExtractedEdge,
    Access, ExtractedGraph, ExtractedNode, FreedBy, InputNode, LayoutInfo, LayoutIrOp,
    LayoutTensorInfo,
    LogicalInfo, OpInput, OpNode, OutputNode, OutputSlot, Sharing,
};

// =============================================================================
// Mock ops (interface-defined behaviors the real op set cannot express)
// =============================================================================

/// A configurable op for exercising the analyzer and planner.
///
/// * `reads[i]` — does operand `i` read its buffer?
/// * `in_place_operand` — which operand (if any) declares result 0 as an
///   aliasing value (its in-place candidate; a dest tie, since MockOp writes
///   its result).
#[derive(Debug, Clone, Default)]
pub struct MockOp {
    pub reads: Vec<bool>,
    pub in_place_operand: Option<usize>,
    /// Grants the may-share permit for EVERY operand against the tied result
    /// (the unconditional, trusted permission).
    pub not_conflicting: bool,
}

impl MockOp {
    /// A write-only destination on operand 0 aliasing result 0 (the shape of
    /// every DPS dest operand).
    pub fn write_only_dest() -> Self {
        MockOp {
            reads: vec![false],
            in_place_operand: Some(0),
            ..Default::default()
        }
    }
}

impl OpSlotNames for MockOp {}

impl BufferTensorIrOp for MockOp {
    fn label(&self) -> &str {
        "MockOp"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        self.reads.get(operand).copied().unwrap_or(false)
    }
}

impl Bufferizable for MockOp {
    fn alias_info(&self) -> Vec<AliasInfo> {
        let mut info = Vec::new();
        if let Some(operand) = self.in_place_operand {
            info.push(AliasInfo { operand, result: 0, sharing: Sharing::Must });
            if self.not_conflicting {
                for read in 0..self.reads.len() {
                    info.push(AliasInfo { operand: read, result: 0, sharing: Sharing::May });
                }
            }
        }
        info
    }
}

impl crate::layout_ir::ToDps for MockOp {
    fn to_dps(&self) -> Option<Box<dyn crate::layout_ir::LayoutIrOp>> {
        None // mocks declare their interface directly; no DPS rewrite
    }
}

impl LayoutIrOp for MockOp {}

/// A pure view op (à la `tensor.extract_slice`): its single result ALIASES
/// operand 0's storage under the result's own layout — a derived buffer,
/// never interchangeable with the parent. What makes it a VIEW is its
/// declared memory effects, not a tie kind: it writes nothing, so its tie is
/// not a dest tie (seeding never crosses it), and an ADMITTED view folds to
/// nothing in the plan. A REJECTED view repairs like every other tie — the
/// result gets fresh storage initialized by copying the parent's bytes (a
/// view over a copy of the buffer, layout unchanged).
#[derive(Debug, Clone)]
pub struct MockView;

impl OpSlotNames for MockView {}

impl BufferTensorIrOp for MockView {
    fn label(&self) -> &str {
        "MockView"
    }

    fn operand_reads_memory(&self, _operand: usize) -> bool {
        false // metadata op: no bytes observed
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // metadata op: no bytes produced
    }
}

impl Bufferizable for MockView {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo { operand: 0, result: 0, sharing: Sharing::Must }]
    }
}

impl crate::layout_ir::ToDps for MockView {
    fn to_dps(&self) -> Option<Box<dyn crate::layout_ir::LayoutIrOp>> {
        None // nothing is written: there is no destination to pass
    }
}

impl LayoutIrOp for MockView {}

/// An alloc-like op (à la `tensor.empty`): its single result is undefined
/// storage, so reading it is never a conflict.
#[derive(Debug, Clone)]
pub struct EmptyOp;

impl OpSlotNames for EmptyOp {}

impl BufferTensorIrOp for EmptyOp {
    fn label(&self) -> &str {
        "Empty"
    }

    fn result_is_undefined(&self, _result: usize) -> bool {
        true
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false
    }
}

impl Bufferizable for EmptyOp {}

impl crate::layout_ir::ToDps for EmptyOp {
    fn to_dps(&self) -> Option<Box<dyn crate::layout_ir::LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for EmptyOp {}

// =============================================================================
// TestGraph: hand-authored ExtractedGraphs
// =============================================================================

/// Builds a well-formed [`ExtractedGraph`] node by node. Buffer and layout
/// identities are controlled by name so tests can express cohabitation (two
/// values sharing one buffer) and layout (in)equality precisely.
pub struct TestGraph {
    dag: ExtractedDag,
    /// The node producing each value; every operand must have one (asserted).
    producers: HashMap<ClassId, NodeIndex>,
    slots: Vec<OutputSlot>,
    next: u32,
}

impl TestGraph {
    pub fn new() -> Self {
        TestGraph {
            dag: ExtractedDag::new(),
            producers: HashMap::new(),
            slots: Vec::new(),
            next: 0,
        }
    }

    fn fresh(&mut self) -> u32 {
        self.next += 1;
        self.next
    }

    fn value_info(&self, name: &str, layout: &str) -> LayoutTensorInfo {
        LayoutTensorInfo {
            eclass: ClassId::from(format!("val${name}")),
            label: name.to_string(),
            tooltip: String::new(),
            shape: None,
            dtype: None,
            dims: None,
            element_bits: None,
            logical: LogicalInfo {
                eclass: ClassId::from(format!("logical${name}")),
                label: name.to_string(),
                tooltip: String::new(),
                op: None,
                children: Vec::new(),
            },
            layout: LayoutInfo {
                eclass: ClassId::from(format!("layout${layout}")),
                label: layout.to_string(),
                tooltip: String::new(),
            },
        }
    }

    /// Buffer identities are keyed by `buffer` name: two bindings naming the
    /// same buffer share one `id_eclass` (cohabitation), distinct names get
    /// distinct identities.
    fn buffer_info(
        &mut self,
        buffer: &str,
        access: Option<Access>,
        freed_by: Option<FreedBy>,
    ) -> BufferInfo {
        let n = self.fresh();
        BufferInfo {
            lit: None,
            tensor_eclass: ClassId::from(format!("buftensor${n}")),
            tensor_label: buffer.to_string(),
            tensor_tooltip: String::new(),
            id_eclass: ClassId::from(format!("buf${buffer}")),
            id_label: buffer.to_string(),
            id_tooltip: String::new(),
            access,
            freed_by,
        }
    }

    /// A program input: value `name` (with layout `layout`) living in `buffer`.
    pub fn input(&mut self, name: &str, buffer: &str, access: Access, layout: &str) -> ClassId {
        self.input_binding(name, buffer, Some(access), Some(FreedBy::Caller), layout)
    }

    /// The fully-explicit binding builder: `None` models a program that OMITS
    /// a boundary declaration (for input-validation tests — well-formed
    /// programs always declare).
    pub fn input_binding(
        &mut self,
        name: &str,
        buffer: &str,
        access: Option<Access>,
        freed_by: Option<FreedBy>,
        layout: &str,
    ) -> ClassId {
        let value = self.value_info(name, layout);
        let eclass = value.eclass.clone();
        let buffer = self.buffer_info(buffer, access, freed_by);
        let node = self
            .dag
            .add_node(ExtractedNode::BufferInput(InputNode { value, buffer }));
        self.producers.insert(eclass.clone(), node);
        eclass
    }

    /// An op with the given interface. One output value per `(name, layout)`
    /// pair. Adds real dataflow edges from each operand's producer.
    pub fn op(
        &mut self,
        iface: Box<dyn LayoutIrOp>,
        inputs: &[&ClassId],
        outputs: &[(&str, &str)],
    ) -> Vec<ClassId> {
        let n = self.fresh();
        let output_infos: Vec<LayoutTensorInfo> = outputs
            .iter()
            .map(|(name, layout)| self.value_info(name, layout))
            .collect();
        let result_classes: Vec<ClassId> =
            output_infos.iter().map(|info| info.eclass.clone()).collect();
        let op_inputs: Vec<OpInput> = inputs
            .iter()
            .enumerate()
            .map(|(index, value)| OpInput {
                port: format!("in{index}"),
                value: (*value).clone(),
            })
            .collect();
        let node = self.dag.add_node(ExtractedNode::LayoutOp(OpNode {
            op: iface,
            provenance: crate::layout_ir::Provenance::Synthesized { id: n },
            inputs: op_inputs,
            outputs: output_infos,
            tooltip: String::new(),
            cost: 1,
            copies: 0,
        }));
        for (index, value) in inputs.iter().enumerate() {
            let producer = *self
                .producers
                .get(*value)
                .unwrap_or_else(|| panic!("operand {value} has no producer"));
            self.dag.add_edge(
                producer,
                node,
                ExtractedEdge {
                    value: (*value).clone(),
                    port: format!("in{index}"),
                },
            );
        }
        for eclass in &result_classes {
            self.producers.insert(eclass.clone(), node);
        }
        result_classes
    }

    /// Pin `value` into `buffer` as the next output slot.
    pub fn output(&mut self, value: &ClassId, buffer: &str) {
        let index = self.slots.len();
        let buffer = self.buffer_info(buffer, Some(Access::ReadWrite), Some(FreedBy::Caller));
        self.slots.push(OutputSlot {
            index,
            value: value.clone(),
            buffer,
        });
    }

    /// Finalize: emit the `BufferOutput` node (with edges from every slot
    /// value's producer, so topological order is correct) and return the graph.
    pub fn build(mut self) -> ExtractedGraph {
        let slots = std::mem::take(&mut self.slots);
        let node = self.dag.add_node(ExtractedNode::BufferOutput(OutputNode {
            eclass: ClassId::from("output$0"),
            label: "output".to_string(),
            tooltip: String::new(),
            slots: slots.clone(),
        }));
        for slot in &slots {
            let producer = *self
                .producers
                .get(&slot.value)
                .unwrap_or_else(|| panic!("output value {} has no producer", slot.value));
            self.dag.add_edge(
                producer,
                node,
                ExtractedEdge {
                    value: slot.value.clone(),
                    port: format!("out {}", slot.index),
                },
            );
        }
        // Re-insert slots into the node weight (cloned above for edge wiring).
        if let ExtractedNode::BufferOutput(output) = &mut self.dag[node] {
            output.slots = slots;
        }
        ExtractedGraph {
            dag: self.dag,
            outputs: vec![node],
        }
    }
}

// =============================================================================
// Fixture path: real egglog scripts through the real extractor
// =============================================================================

/// Run `test_scripts/<script>` through egglog (with the full preamble) and
/// hand back the serialized e-graph — the selection tooling's raw material.
pub fn serialize_fixture(script: &str) -> egraph_serialize::EGraph {
    use egglog::SerializeConfig;

    let preamble = crate::egglog_snippet::assembled_program();
    let source = fs::read_to_string(format!("src/egglog/checkpoint_5/test_scripts/{script}"))
        .unwrap_or_else(|_| panic!("fixture script {script} readable"));
    let program = format!("{preamble}\n\n{source}");

    let mut egraph = crate::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(Some(script.to_string()), &program)
        .unwrap_or_else(|err| panic!("egglog failed on fixture {script}: {err}"));
    egraph.serialize(SerializeConfig::default()).egraph
}

/// Build a TOTAL genome over a fixture's produced classes: each class takes
/// the first preference (an implementation constructor name) it can satisfy,
/// falling back to its first candidate — the producer index is
/// deterministically sorted, so the same preferences always build the same
/// genome.
pub fn genome_preferring(
    egraph: &egraph_serialize::EGraph,
    preferences: &[&str],
) -> extractor::Genome {
    let index = extractor::producer_index(egraph);
    let mut genome = extractor::Genome::default();
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
    script: &str,
    preferences: &[&str],
) -> (ExtractedGraph, u64) {
    let egraph = serialize_fixture(script);
    let genome = genome_preferring(&egraph, preferences);
    let graph = extractor::extract_layout_ir_with_genome(&egraph, &genome)
        .expect("genome extraction runs")
        .expect("genome extraction reaches the boundary");
    let fingerprint = extractor::plan_fingerprint(&graph);
    (graph, fingerprint)
}

/// Run `test_scripts/<script>` through egglog (with the full preamble) and the
/// real extractor, returning the extracted graph. Panics on any failure — these
/// are test fixtures.
pub fn extract_fixture(script: &str) -> ExtractedGraph {
    use egglog::SerializeConfig;

    let preamble = crate::egglog_snippet::assembled_program();
    let source = fs::read_to_string(format!("src/egglog/checkpoint_5/test_scripts/{script}"))
        .unwrap_or_else(|_| panic!("fixture script {script} readable"));
    let program = format!("{preamble}\n\n{source}");

    let mut egraph = crate::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(Some(script.to_string()), &program)
        .unwrap_or_else(|err| panic!("egglog failed on fixture {script}: {err}"));
    let serialized = egraph.serialize(SerializeConfig::default()).egraph;
    extractor::extract_layout_ir(&serialized)
        .expect("extraction succeeds")
        .unwrap_or_else(|| panic!("fixture {script} produced no extracted graph"))
}

/// [`extract_fixture`] restricted to an allow-list of LayoutTensorOp
/// constructor names — forces extraction through specific implementations so
/// a test can exercise one op end to end. Panics if the program is not
/// implementable within the list; use [`try_extract_fixture_with_ops`] to
/// assert that failure itself.
pub fn extract_fixture_with_ops(script: &str, allowed: &[&str]) -> ExtractedGraph {
    try_extract_fixture_with_ops(script, allowed)
        .expect("extraction succeeds")
        .unwrap_or_else(|| panic!("fixture {script} produced no extracted graph"))
}

/// The fallible form of [`extract_fixture_with_ops`].
pub fn try_extract_fixture_with_ops(
    script: &str,
    allowed: &[&str],
) -> anyhow::Result<Option<ExtractedGraph>> {
    use egglog::SerializeConfig;

    let preamble = crate::egglog_snippet::assembled_program();
    let source = fs::read_to_string(format!("src/egglog/checkpoint_5/test_scripts/{script}"))
        .unwrap_or_else(|_| panic!("fixture script {script} readable"));
    let program = format!("{preamble}\n\n{source}");

    let mut egraph = crate::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(Some(script.to_string()), &program)
        .unwrap_or_else(|err| panic!("egglog failed on fixture {script}: {err}"));
    let serialized = egraph.serialize(SerializeConfig::default()).egraph;
    extractor::extract_layout_ir_with_ops(&serialized, Some(allowed))
}

#[cfg(test)]
mod harness_tests {
    use super::*;
    use crate::bufferize;

    /// The builder produces a graph the real pipeline accepts end to end.
    #[test]
    fn builder_graph_bufferizes() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let results = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        );
        g.output(&results[0], "B");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");
        // Out-of-place: y in a fresh alloc, then a copy into pinned B.
        let summary = plan.summary();
        assert!(summary.contains("MockOp"), "{summary}");
        assert!(summary.contains("BufferCopy"), "{summary}");
    }

    /// The fixture path runs a real egg script through egglog + the extractor.
    #[test]
    fn fixture_extracts_and_bufferizes() {
        let graph = extract_fixture("boundary_pass_through.egg");
        let plan = bufferize::bufferize(&graph).expect("bufferizes");
        let summary = plan.summary();
        assert!(summary.contains("ops (0):"), "{summary}");
    }

    /// The canonical WAR hazard, through the REAL pipeline: y = Sqrt(x) lands in
    /// x's buffer via a boundary copy while Exp(x) still reads x. The plan must
    /// carry an Anti edge ordering Exp's read before the copy's write.
    #[test]
    fn war_hazard_gets_anti_edge() {
        use crate::bufferize::{BufferNode, EdgeKind};
        use petgraph::visit::EdgeRef;

        let graph = extract_fixture("boundary_war_hazard.egg");
        let plan = bufferize::bufferize(&graph).expect("bufferizes");

        let anti: Vec<_> = plan
            .dag
            .edge_references()
            .filter(|e| e.weight().kind == EdgeKind::Anti)
            .collect();
        // 1 WAR (Exp's read before the copy overwriting x's buffer) + 2
        // lifetime (each fresh buffer's src-reading copy before its free).
        assert_eq!(anti.len(), 3, "expected 1 WAR + 2 lifetime antis:\n{}", plan.summary());
        let war: Vec<_> = anti
            .iter()
            .filter(|e| {
                !matches!(&plan.dag[e.target()], BufferNode::Compute { writes, .. }
                    if writes.is_empty())
            })
            .collect();
        assert_eq!(war.len(), 1, "expected exactly one WAR anti:\n{}", plan.summary());
        let edge = war[0];
        // Source must be the Exp compute node; target the copy into x's buffer.
        match &plan.dag[edge.source()] {
            BufferNode::Compute { op, .. } => assert_eq!(op.label(), "ExpFunctionalGeneric"),
            other => panic!("anti edge source should be ExpFunctionalGeneric, got {other:?}"),
        }
        assert!(matches!(&plan.dag[edge.target()], BufferNode::BufferCopy { .. }));
    }

    /// Copy-vs-copy WAR: out0 passes x onward into C (copy B->C reads B) while
    /// out1 materializes f(x) into B (copy alloc->B writes B). The two copies
    /// are unordered by dataflow; the plan must order read-of-B first.
    #[test]
    fn copy_reading_buffer_ordered_before_copy_writing_it() {
        use crate::bufferize::{BufferNode, EdgeKind};
        use petgraph::visit::EdgeRef;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&x, "C"); // copy B -> C (reads B)
        g.output(&y, "B"); // copy alloc -> B (writes B)
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        let anti: Vec<_> = plan
            .dag
            .edge_references()
            .filter(|e| e.weight().kind == EdgeKind::Anti)
            .collect();
        // 1 WAR (the B-reading copy before the B-writing copy) + 1 lifetime
        // (the B-writing copy's src-read before y's buffer is freed).
        assert_eq!(anti.len(), 2, "expected 1 WAR + 1 lifetime anti:\n{}", plan.summary());
        let war: Vec<_> = anti
            .iter()
            .filter(|e| {
                !matches!(&plan.dag[e.target()], BufferNode::Compute { writes, .. }
                    if writes.is_empty())
            })
            .collect();
        assert_eq!(war.len(), 1, "expected exactly one WAR anti:\n{}", plan.summary());
        let edge = war[0];
        // Direction: the copy READING B (the B->C pass-onward) must run before
        // the copy WRITING B (the alloc->B overwrite).
        match &plan.dag[edge.source()] {
            BufferNode::BufferCopy { src, .. } => assert_eq!(src, &plan.value_buffer[&x]),
            other => panic!("anti source should be the B-reading copy, got {other:?}"),
        }
        match &plan.dag[edge.target()] {
            BufferNode::BufferCopy { dst, .. } => assert_eq!(dst, &plan.value_buffer[&x]),
            other => panic!("anti target should be the B-writing copy, got {other:?}"),
        }
    }

    /// A buffer both passed through to one slot and copy-overwritten for another
    /// is unsatisfiable (one buffer, two final values). Input-program validation
    /// rejects the program up front — the demanded value y is not among B's
    /// inputs, so it cannot cohabit with the promised pass-through.
    #[test]
    fn passthrough_plus_copy_into_same_buffer_is_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&x, "B"); // pass-through: B's final contents promised to be x
        g.output(&y, "B"); // copy alloc -> B: overwrites them
        let err = bufferize::bufferize(&g.build()).unwrap_err();
        assert!(err.to_string().contains("unsupported program"), "{err}");
        assert!(err.to_string().contains("distinct final values"), "{err}");
    }

    /// OUT-OF-PLACE RESOLUTION, accumulator case: op2 accumulates into x
    /// (reads its own destination) but a sibling reader of x forces rejection.
    /// The plan must copy x's contents into the fresh result buffer BEFORE op2,
    /// retarget op2's operand to it, and never write x's pinned buffer.
    #[test]
    fn rejected_accumulator_copies_contents_and_retargets() {
        use crate::bufferize::{BufferId, BufferNode};
        use petgraph::visit::EdgeRef;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        // sibling reader, unordered with the accumulator -> forces rejection
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true], // accumulator: reads its own destination
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&x],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&y, "C");
        g.output(&r, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        // Find the accumulator's compute node.
        let (acc_idx, acc_reads, acc_writes) = plan
            .dag
            .node_indices()
            .find_map(|idx| match &plan.dag[idx] {
                BufferNode::Compute { reads, writes, .. }
                    if !reads.is_empty() && plan.value_buffer[&r] == writes[0] =>
                {
                    Some((idx, reads.clone(), writes.clone()))
                }
                _ => None,
            })
            .expect("accumulator node");
        // Retargeted: reads its own fresh result buffer, not x's pinned B.
        assert_eq!(acc_reads[0], acc_writes[0], "{}", plan.summary());
        assert!(matches!(acc_writes[0], BufferId::Allocated(_)));
        // PINNING (critique test i): must NOT share the pinned input buffer.
        assert_ne!(acc_writes[0], plan.value_buffer[&x]);
        // A copy feeds the accumulator: src = x's buffer, dst = the fresh alloc,
        // dataflow-ordered before the op.
        let copy_edge = plan
            .dag
            .edges_directed(acc_idx, petgraph::Direction::Incoming)
            .find(|e| matches!(&plan.dag[e.source()], BufferNode::BufferCopy { .. }))
            .expect("copy feeds the accumulator");
        match &plan.dag[copy_edge.source()] {
            BufferNode::BufferCopy { src, dst } => {
                assert_eq!(src, &plan.value_buffer[&x]);
                assert_eq!(dst, &acc_writes[0]);
            }
            _ => unreachable!(),
        }
    }

    /// OUT-OF-PLACE RESOLUTION, write-only case: a rejected write-only
    /// destination retargets with NO contents copy (poison contents are
    /// irrelevant), and must not share the operand's storage.
    #[test]
    fn rejected_write_only_dest_retargets_without_copy() {
        use crate::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        // Interior operand: v is another op's result with a sibling reader.
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("v", "rm")],
        )[0]
        .clone();
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&v],
            &[("y", "rm")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![false], // write-only destination
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&y, "C");
        g.output(&r, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        let (reads, writes) = plan
            .dag
            .node_indices()
            .find_map(|idx| match &plan.dag[idx] {
                BufferNode::Compute { reads, writes, .. }
                    if !reads.is_empty() && plan.value_buffer[&r] == writes[0] =>
                {
                    Some((reads.clone(), writes.clone()))
                }
                _ => None,
            })
            .expect("dest-taking node");
        assert_eq!(reads[0], writes[0], "{}", plan.summary());
        // PINNING (critique test ii): must NOT share the interior operand's alloc.
        assert_ne!(writes[0], plan.value_buffer[&v]);
        assert!(matches!(writes[0], BufferId::Allocated(_)));
        // No copy targets the retargeted buffer (write-only: nothing to preserve).
        let copies_into_target = plan
            .dag
            .node_indices()
            .filter(|&idx| {
                matches!(&plan.dag[idx], BufferNode::BufferCopy { dst, .. } if dst == &writes[0])
            })
            .count();
        assert_eq!(copies_into_target, 0, "{}", plan.summary());
    }

    /// CHAINED in-place destinations, now with MULTI-HOP destination seeding:
    /// OpA writes into poison e producing r1; OpB uses r1 as its write-only
    /// destination producing r2, bound to output D. Bottom-up analysis decides
    /// OpB first (union r1~r2), then OpA (union e~r1); the seed walks D's slot
    /// back through BOTH hops to the chain root e, and since both hops are
    /// admitted in place the whole chain binds to D itself — zero allocations,
    /// zero copies, every op computing directly into the output storage.
    #[test]
    fn chained_destinations_collapse_onto_the_output_buffer() {
        use crate::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let r1 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&e],
            &[("r1", "rm")],
        )[0]
        .clone();
        let r2 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&r1],
            &[("r2", "rm")],
        )[0]
        .clone();
        g.output(&r2, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert_eq!(plan.value_buffer[&e], plan.value_buffer[&r1]);
        assert_eq!(plan.value_buffer[&r1], plan.value_buffer[&r2]);
        assert!(
            matches!(plan.value_buffer[&r2], BufferId::Boundary(_)),
            "the chain seeds into the output buffer itself:\n{}",
            plan.summary()
        );
        assert!(
            plan.value_buffer
                .values()
                .all(|b| !matches!(b, BufferId::Allocated(_))),
            "zero allocations:\n{}",
            plan.summary()
        );
        let copies = plan
            .dag
            .node_indices()
            .filter(|&idx| matches!(&plan.dag[idx], BufferNode::BufferCopy { .. }))
            .count();
        assert_eq!(copies, 0, "zero copies:\n{}", plan.summary());
    }

    /// THE MUTATING TIER, end to end: extraction restricted to
    /// SqrtMutatingGeneric proves the one-buffer kernel through analysis and
    /// planning — the tie is admitted (the op's own read is the same-use
    /// exemption), the result rides x's caller buffer, and the boundary
    /// passes through. Zero copies, zero allocations, no permits involved.
    #[test]
    fn mutating_sqrt_bufferizes_zero_copy_in_place() {
        use crate::bufferize::{BufferId, BufferNode};

        let graph = extract_fixture_with_ops(
            "boundary_in_place_mutation.egg",
            &["LayoutTensorOpSqrtMutatingGeneric"],
        );
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        assert!(
            plan.buffers.keys().all(|id| matches!(id, BufferId::Boundary(_))),
            "zero allocations:\n{}",
            plan.summary()
        );
        let mut computes = 0;
        for idx in plan.dag.node_indices() {
            match &plan.dag[idx] {
                BufferNode::BufferCopy { .. } => panic!("zero copies:\n{}", plan.summary()),
                BufferNode::Compute { op, reads, writes, .. } => {
                    computes += 1;
                    assert_eq!(op.label(), "SqrtMutatingGeneric");
                    assert_eq!(reads.len(), 1, "one buffer in:\n{}", plan.summary());
                    assert_eq!(reads[0], writes[0], "mutated in place:\n{}", plan.summary());
                    assert!(matches!(&writes[0], BufferId::Boundary(_)));
                }
                _ => {}
            }
        }
        assert_eq!(computes, 1);
    }

    /// THE MAY-SHARE PERMIT, end to end: z = x + x back into x's buffer,
    /// extraction restricted to the input-alias-safe mutating add. The rhs
    /// read aliases the mutated storage; the permit (whose all-layouts-equal
    /// precondition the egglog match discharged) excuses it, and the whole
    /// accumulation is zero-copy in the caller's buffer.
    #[test]
    fn alias_safe_add_accumulates_x_plus_x_in_place() {
        use crate::bufferize::{BufferId, BufferNode};

        let graph = extract_fixture_with_ops(
            "boundary_alias_safe_add.egg",
            &["LayoutTensorOpAddMutatingInputAliasSafeGeneric"],
        );
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        assert!(
            plan.buffers.keys().all(|id| matches!(id, BufferId::Boundary(_))),
            "zero allocations:\n{}",
            plan.summary()
        );
        let mut computes = 0;
        for idx in plan.dag.node_indices() {
            match &plan.dag[idx] {
                BufferNode::BufferCopy { .. } => panic!("zero copies:\n{}", plan.summary()),
                BufferNode::Compute { op, reads, writes, .. } => {
                    computes += 1;
                    assert_eq!(op.label(), "AddMutatingInputAliasSafeGeneric");
                    assert_eq!(reads.len(), 2, "{}", plan.summary());
                    assert_eq!(reads[0], writes[0], "lhs mutated in place");
                    assert_eq!(reads[1], writes[0], "rhs reads the SAME storage (permitted)");
                }
                _ => {}
            }
        }
        assert_eq!(computes, 1);
    }

    /// THE CONTRAST: the same x + x program through the PLAIN mutating add,
    /// which declares no permit — absence is restrict semantics. The rhs
    /// aliasing the mutated storage rejects the tie; the generic repair
    /// relocates (copy-in to a fresh buffer, mutate there) and the boundary
    /// copies back. Sound, two copies, one allocation — the price of the
    /// missing permit.
    #[test]
    fn plain_mutating_add_on_x_plus_x_degrades_to_copies() {
        use crate::bufferize::{BufferId, BufferNode};

        let graph = extract_fixture_with_ops(
            "boundary_alias_safe_add.egg",
            &["LayoutTensorOpAddMutatingGeneric"],
        );
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        assert_eq!(allocs, 1, "one relocation:\n{}", plan.summary());
        let copies = plan
            .dag
            .node_indices()
            .filter(|&idx| matches!(&plan.dag[idx], BufferNode::BufferCopy { .. }))
            .count();
        assert_eq!(copies, 2, "copy-in + boundary copy:\n{}", plan.summary());
    }

    /// The allow-list is a hard filter: a program not implementable within
    /// it fails extraction loudly, never silently substitutes.
    #[test]
    fn op_filter_excluding_every_implementation_fails_loudly() {
        let err = try_extract_fixture_with_ops(
            "boundary_in_place_mutation.egg",
            &["LayoutTensorOpExpMutatingGeneric"], // no Exp in this program
        )
        .expect_err("no implementation for sqrt is allowed");
        assert!(err.to_string().contains("failed to extract"), "{err}");
    }

    /// Boundary in-place mutation through the FUNCTIONAL op, after the
    /// engine stopped checking layouts: the same-op read of x against the
    /// seeded destination has no unconditional permit, so the seed is
    /// rejected and the plan degrades soundly — fresh allocation, boundary
    /// copy back into the caller's buffer. No Anti edge is needed: the copy
    /// is dataflow-ordered after the buffer's only reader (its source IS the
    /// sqrt's output). (The zero-copy lowering for this program is the
    /// MutatingGeneric op, which extraction does not yet prefer — the
    /// recorded extraction-preference decision.)
    #[test]
    fn boundary_mutation_via_functional_degrades_to_copy() {
        use crate::bufferize::{BufferId, BufferNode, EdgeKind};
        use petgraph::visit::EdgeRef;

        let graph = extract_fixture("boundary_in_place_mutation.egg");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        assert_eq!(allocs, 1, "one relocated destination:\n{}", plan.summary());
        let copies = plan
            .dag
            .node_indices()
            .filter(|&idx| matches!(&plan.dag[idx], BufferNode::BufferCopy { .. }))
            .count();
        assert_eq!(copies, 1, "one boundary copy:\n{}", plan.summary());
        // No WAR anti is needed (the copy is dataflow-ordered after the
        // read); the one anti is lifetime — the boundary copy's src-read
        // before the fresh buffer's free.
        let anti: Vec<_> = plan
            .dag
            .edge_references()
            .filter(|edge| edge.weight().kind == EdgeKind::Anti)
            .collect();
        assert_eq!(anti.len(), 1, "one lifetime anti only:\n{}", plan.summary());
        assert!(
            matches!(&plan.dag[anti[0].target()], BufferNode::Compute { writes, .. }
                if writes.is_empty()),
            "the sole anti targets the free:\n{}",
            plan.summary()
        );
    }

    /// One value bound to TWO output buffers: the chain-root poison can only
    /// live in one of them. The seed goes to the lowest slot (D); the other
    /// slot (E) is served by a copy out of D.
    #[test]
    fn seed_ties_break_to_lowest_slot() {
        use crate::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&e],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        g.output(&r, "E");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert!(
            matches!(plan.value_buffer[&r], BufferId::Boundary(_)),
            "r seeds into an output buffer:\n{}",
            plan.summary()
        );
        let copies: Vec<(&BufferId, &BufferId)> = plan
            .dag
            .node_indices()
            .filter_map(|idx| match &plan.dag[idx] {
                BufferNode::BufferCopy { src, dst } => Some((src, dst)),
                _ => None,
            })
            .collect();
        assert_eq!(copies.len(), 1, "one copy serves the losing slot:\n{}", plan.summary());
        assert_eq!(
            copies[0].0,
            &plan.value_buffer[&r],
            "the copy reads the seeded buffer:\n{}",
            plan.summary()
        );
        assert_ne!(copies[0].0, copies[0].1);
    }

    /// A seed whose write would clobber a value an UNORDERED op still reads is
    /// rejected, and the plan degrades to exactly the unseeded shape: fresh
    /// allocation, boundary copy, and the WAR machinery ordering the copy
    /// after the reader.
    #[test]
    fn rejected_seed_degrades_to_copy_with_war_edge() {
        use crate::bufferize::{BufferId, BufferNode, EdgeKind};
        use petgraph::visit::EdgeRef;

        let mut g = TestGraph::new();
        let x = g.input("x", "D", Access::ReadWrite, "rm");
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&e],
            &[("r", "rm")],
        )[0]
        .clone();
        // An unordered reader of x keeps x's storage (buffer D) live.
        let s = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("s", "rm")],
        )[0]
        .clone();
        g.output(&r, "D"); // seed proposal: e ↦ D — must be rejected
        g.output(&s, "E");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert!(
            matches!(plan.value_buffer[&r], BufferId::Allocated(_)),
            "the rejected seed falls back to a fresh allocation:\n{}",
            plan.summary()
        );
        let copies_into_d = plan
            .dag
            .node_indices()
            .filter(|&idx| {
                matches!(&plan.dag[idx], BufferNode::BufferCopy { dst, .. }
                    if dst == &plan.value_buffer[&x])
            })
            .count();
        assert_eq!(copies_into_d, 1, "boundary copy restored:\n{}", plan.summary());
        // 1 WAR (the reader before the boundary copy) + 2 lifetime (each
        // fresh buffer's src-reading copy before its free).
        let anti: Vec<_> = plan
            .dag
            .edge_references()
            .filter(|edge| edge.weight().kind == EdgeKind::Anti)
            .collect();
        assert_eq!(anti.len(), 3, "1 WAR + 2 lifetime antis:\n{}", plan.summary());
        let war = anti
            .iter()
            .filter(|e| {
                !matches!(&plan.dag[e.target()], BufferNode::Compute { writes, .. }
                    if writes.is_empty())
            })
            .count();
        assert_eq!(war, 1, "reader ordered before the copy:\n{}", plan.summary());
    }

    /// RANK 6 + the CRITICAL WAR fix regression: two inputs swapping buffers
    /// (out0 = b into P, out1 = a into Q, with a@P and b@Q) produce two copies
    /// that each read the other's destination. Both anti edges must be added
    /// (judged against the FROZEN dataflow ordering — the executed-miscompile
    /// fix), making the plan a cycle: a loud "unschedulable" error, never a
    /// silent wrong-order schedule.
    #[test]
    fn buffer_swap_fails_loudly_not_silently() {
        let mut g = TestGraph::new();
        let a = g.input("a", "P", Access::ReadWrite, "rm");
        let b = g.input("b", "Q", Access::ReadWrite, "rm");
        g.output(&b, "P"); // copy Q -> P (reads Q, writes P)
        g.output(&a, "Q"); // copy P -> Q (reads P, writes Q)
        let err = bufferize::bufferize(&g.build()).unwrap_err();
        assert!(err.to_string().contains("unschedulable"), "{err}");
    }

    /// MAJOR review finding, now caught at the door: two DIFFERENT computed
    /// values demanded in the SAME output buffer is a one-buffer-two-values
    /// contradiction. Input-program validation rejects the program outright
    /// (it is not a planning failure — no plan can satisfy it at buffer
    /// granularity), so the WAW contradiction can no longer reach planning
    /// through materialized outputs. The final-bindings check remains as the
    /// trusted kernel's backstop for the pass-through sliver.
    #[test]
    fn two_materialized_values_into_one_buffer_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y1 = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y1", "rm")],
        )[0]
        .clone();
        let y2 = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y2", "rm")],
        )[0]
        .clone();
        g.output(&y1, "D");
        g.output(&y2, "D"); // same buffer, two distinct computed values
        let err = bufferize::bufferize(&g.build()).unwrap_err();
        assert!(err.to_string().contains("unsupported program"), "{err}");
        assert!(err.to_string().contains("distinct final values"), "{err}");
    }

    /// PARKED PROGRAM CLASS (Decision 1b, 2026-07-08): mutating one boundary
    /// view of a buffer while another view of the same buffer must survive to
    /// the output demands two distinct final values in one buffer — sound only
    /// under REGION reasoning (the views are disjoint rows), which the planner
    /// deliberately does not do. Input-program validation rejects it up front;
    /// the fixture stays as the acceptance test for future region support.
    #[test]
    fn aliased_views_in_place_fixture_rejected_as_unsupported() {
        let graph = extract_fixture("boundary_aliased_views_in_place.egg");
        let err = bufferize::bufferize(&graph).unwrap_err();
        assert!(err.to_string().contains("unsupported program"), "{err}");
        assert!(err.to_string().contains("pass-throughs"), "{err}");
    }

    /// RANK 8: materializing into a Read-granted destination is a hard error
    /// (write obligation without write rights)...
    #[test]
    fn copy_into_read_granted_buffer_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadOnly, "rm");
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&y, "B"); // materialize a computed value into the Read buffer
        let err = bufferize::bufferize(&g.build()).unwrap_err();
        assert!(err.to_string().contains("read-only buffer"), "{err}");
    }

    /// Declaration satisfiability (the last inheritor of the final-bindings
    /// check, now at input validation): two pass-through demands on one
    /// buffer under the SAME layout but with DIFFERENT values are
    /// contradictory — the buffer cannot end holding both.
    #[test]
    fn same_layout_pass_through_contradiction_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.input("y", "B", Access::ReadWrite, "rm");
        g.output(&x, "B");
        g.output(&y, "B");
        let err = bufferize::bufferize(&g.build()).unwrap_err();
        assert!(err.to_string().contains("same layout"), "{err}");
        assert!(err.to_string().contains("invalid input program"), "{err}");
    }

    /// ...while passing the Read buffer's own value through is legal.
    #[test]
    fn passthrough_of_read_granted_buffer_is_legal() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadOnly, "rm");
        g.output(&x, "B");
        let plan = bufferize::bufferize(&g.build()).expect("pass-through is legal");
        assert!(plan.summary().contains("ops (0):"), "{}", plan.summary());
    }

    /// Numeric geometry rides extraction: literal dims and bit widths are
    /// walked off the e-graph terms onto every value info — the surface the
    /// SsaReferenceRuntime sizes its buffers from.
    #[test]
    fn extraction_carries_numeric_dims_and_bits() {
        use crate::layout_ir::ExtractedNode;

        let graph = extract_fixture("matmul_fused_example.egg");
        let matmul_out = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label() == "MatMulFusedGeneric" => {
                    Some(op.outputs[0].clone())
                }
                _ => None,
            })
            .expect("fused matmul extracted");
        assert_eq!(matmul_out.dims.as_deref(), Some(&[2, 4][..]));
        assert_eq!(matmul_out.element_bits, Some(32));
    }

    /// FIXED ISSUE PROOF (root-caused and fixed 2026-07-29, Austin-approved):
    /// this sound div/mod iota gather program once fired the axis-support
    /// tripwire. Root cause: the subst CoordVar arm instantiated coordinates
    /// with map entries OUTSIDE the coordinate's own range — an out-of-box
    /// instantiation of box-true equations (e.g. a view-space quotient
    /// box-true equal to 1), whose per-representative image unions then
    /// welded false equalities into literal classes. The in-range
    /// instantiation guard on that arm (the well-definedness condition of
    /// subst-of; see the preamble subst contract) removes the corruption;
    /// this test — the original reproducer, formerly the expect-panic pin —
    /// now asserts CLEAN saturation and stands as the fix's proof.
    #[test]
    fn div_mod_iota_gather_with_layout_saturates_cleanly() {
        let body = r#"
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let flat (IntAdd (IntMul (CoordVar out_shape 1) (IntLit 3)) (CoordVar out_shape 0)))
(let value
  (IntAdd
    (IntMul (IntAdd (IntTruncRem (IntTruncDiv flat (IntLit 3)) (IntLit 2)) (IntLit 1)) (IntLit 5))
    (IntAdd (IntTruncRem flat (IntLit 3)) (IntLit 2))))
(let row_coord (IntTruncDiv value (IntLit 5)))
(let col_coord (IntTruncRem value (IntLit 5)))
(let row_iota (LogicalIota row_coord out_shape))
(let col_iota (LogicalIota col_coord out_shape))
(let data_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 5) (IntExprNil)))))
(let data_logical (LogicalTensorInputLit (LogicalIdLit "data") data_shape (F32)))
(let gathered
  (LogicalGather data_logical
    (LogicalTensorCons row_iota (LogicalTensorCons col_iota (LogicalTensorNil)))))
(let data_layout (RightMajorContiguousElementLayoutLit data_shape (bits-of (F32))))
(let data_layout_tensor (LayoutTensorLit data_logical data_layout))
(run-schedule (saturate (run)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!("{}\n\n{}", crate::egglog_snippet::assembled_program(), body);
        crate::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &full)
            .expect("sound div/mod gather program saturates cleanly under the subst range guard");
    }

    /// The bool bridge (Austin's design 2026-07-29): decided comparisons
    /// collapse to their literals, undecided indicators stay bits, and the
    /// pad pattern's clamp/mask bounds derive — the fixture's checks and
    /// fail-pins are the proof.
    #[test]
    fn bool_bridge_fixture_holds() {
        let _ = serialize_fixture("bool_bridge_example.egg");
    }

    /// GENOME WALK, consistent-fused: both boundary classes choose the SAME
    /// fused enode; instance dedup (keyed by concrete enode) yields ONE
    /// kernel claiming both output slots into the caller's buffers.
    #[test]
    fn genome_fused_choice_dedups_one_instance_claiming_both_slots() {
        use crate::bufferize::BufferId;
        use crate::layout_ir::ExtractedNode;

        let (graph, _) = extract_fixture_with_genome(
            "add_mul_fused.egg",
            &["LayoutTensorOpAddMulFusedGeneric"],
        );
        let computes = graph
            .dag
            .node_weights()
            .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
            .count();
        assert_eq!(computes, 1, "instance dedup across both claimed slots");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");
        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        assert_eq!(allocs, 0, "both slots land in caller buffers:\n{}", plan.summary());
    }

    /// GENOME WALK, the MIXED plan (the paper-walk's scenario 3, now
    /// executable): the add value chooses the fused kernel, the mul value
    /// chooses the standalone Mul. The fused instance's unclaimed mul slot
    /// writes a fresh WASTE destination — allocated, written, freed unread —
    /// instead of double-writing the caller's mul buffer (which the
    /// two-writers tripwire would reject). Wasteful, correct, priced by
    /// profiling: the single-mutation bridge between the pair and fused
    /// plans.
    #[test]
    fn genome_mixed_choice_mints_a_waste_destination() {
        use crate::bufferize::BufferId;
        use crate::layout_ir::ExtractedNode;

        let (graph, _) = extract_fixture_with_genome(
            "add_mul_fused.egg",
            &[
                "LayoutTensorOpMulFunctionalGeneric",
                "LayoutTensorOpAddMulFusedGeneric",
            ],
        );
        let computes = graph
            .dag
            .node_weights()
            .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
            .count();
        assert_eq!(computes, 2, "fused kernel + standalone mul");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");
        let summary = plan.summary();
        assert!(summary.contains("AddMulFusedGeneric"), "{summary}");
        assert!(summary.contains("MulFunctionalGeneric"), "{summary}");
        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        assert_eq!(allocs, 1, "the unclaimed fused slot writes scratch:\n{summary}");
    }

    /// Many genomes, one plan: the fingerprint identifies the PLAN, so the
    /// search can skip re-profiling genomes that build what it already
    /// timed (plan-hash dedup ruling, 2026-07-27).
    #[test]
    fn genome_plan_fingerprints_identify_plans() {
        let (_, fused_a) = extract_fixture_with_genome(
            "add_mul_fused.egg",
            &["LayoutTensorOpAddMulFusedGeneric"],
        );
        let (_, fused_b) = extract_fixture_with_genome(
            "add_mul_fused.egg",
            &["LayoutTensorOpAddMulFusedGeneric"],
        );
        let (_, mixed) = extract_fixture_with_genome(
            "add_mul_fused.egg",
            &[
                "LayoutTensorOpMulFunctionalGeneric",
                "LayoutTensorOpAddMulFusedGeneric",
            ],
        );
        assert_eq!(fused_a, fused_b, "same genome, same plan, same fingerprint");
        assert_ne!(fused_a, mixed, "different plans, different fingerprints");
    }

    /// Dead rows under a REAL genome: with the fused matmul chosen for the
    /// output, the genome's rows for the product and the broadcast views
    /// (fallback-filled) are never read — one kernel, no intermediate.
    #[test]
    fn genome_matmul_fused_choice_swallows_the_intermediate() {
        use crate::bufferize::BufferId;
        use crate::layout_ir::ExtractedNode;

        let (graph, _) = extract_fixture_with_genome(
            "matmul_fused_example.egg",
            &["LayoutTensorOpMatMulFusedGeneric"],
        );
        let computes = graph
            .dag
            .node_weights()
            .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
            .count();
        assert_eq!(computes, 1, "dead rows: views and product never demanded");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");
        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        assert_eq!(allocs, 0, "no intermediate buffer:\n{}", plan.summary());
    }

    /// CHAIN FUSION, fused route: selecting MatMulFusedGeneric demands
    /// neither the broadcast views nor the [M, N, K] product — their genome
    /// rows are dead, and NO intermediate buffer exists. One kernel, three
    /// caller buffers, zero allocations, zero copies.
    #[test]
    fn matmul_fused_route_swallows_the_intermediate() {
        use crate::bufferize::{BufferId, BufferNode};
        use crate::layout_ir::ExtractedNode;

        let graph = extract_fixture_with_ops(
            "matmul_fused_example.egg",
            &["LayoutTensorOpMatMulFusedGeneric"],
        );
        let computes = graph
            .dag
            .node_weights()
            .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
            .count();
        assert_eq!(computes, 1, "one fused kernel, nothing else");

        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");
        let summary = plan.summary();
        assert!(summary.contains("MatMulFusedGeneric"), "{summary}");
        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        assert_eq!(allocs, 0, "the product is never materialized:\n{summary}");
        let copies = plan
            .dag
            .node_indices()
            .filter(|&idx| matches!(&plan.dag[idx], BufferNode::BufferCopy { .. }))
            .count();
        assert_eq!(copies, 0, "zero-copy:\n{summary}");
    }

    /// CHAIN FUSION, unfused route: the SAME e-graph, restricted to the
    /// decomposed implementations, materializes the [2, 4, 3] product into a
    /// fresh allocation that the reduce consumes — the buffer the fused route
    /// proves unnecessary.
    #[test]
    fn matmul_unfused_route_materializes_the_intermediate() {
        use crate::bufferize::BufferId;

        let graph = extract_fixture_with_ops(
            "matmul_fused_example.egg",
            &[
                "LayoutTensorOpIndexMapApplyViewGeneric",
                "LayoutTensorOpMulFunctionalGeneric",
                "LayoutTensorOpReduceSumGeneric",
            ],
        );
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");
        let summary = plan.summary();
        assert!(summary.contains("MulFunctionalGeneric"), "{summary}");
        assert!(summary.contains("ReduceSumGeneric"), "{summary}");
        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        assert_eq!(allocs, 1, "exactly the product buffer:\n{summary}");
    }

    /// Cost sanity: with every implementation available, the deterministic
    /// extractor prefers the fused kernel (3 slots) over the decomposed
    /// route (product write + reduce read make it strictly costlier).
    #[test]
    fn matmul_default_extraction_prefers_the_fused_kernel() {
        let graph = extract_fixture("matmul_fused_example.egg");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");
        let summary = plan.summary();
        assert!(summary.contains("MatMulFusedGeneric"), "{summary}");
        assert!(
            !summary.contains("ReduceSumGeneric"),
            "the decomposed route must lose on cost:\n{summary}"
        );
    }

    /// The pinned-plan fixture list, shared by the pin test and the
    /// regenerator below.
    const GOLDEN_SCRIPTS: &[&str] = &[
        "add_mul_fused",
        "basic_program",
        "matmul_fused_example",
        "boundary_aliased_views",
        "boundary_donated_input",
        "boundary_gather",
        "boundary_in_place_mutation",
        "boundary_iota",
        "boundary_pass_through",
        "boundary_scalar",
        "boundary_scatter",
        "boundary_view_feeds_compute",
        "boundary_war_hazard",
        "boundary_write_into_viewed_buffer",
        "transformer",
    ];

    /// Regenerate output/<stem>.bufferized.txt — run explicitly by name:
    /// `cargo test regenerate_golden_plans -- --ignored` (no env vars, by
    /// ruling 2026-08-06; invocation-by-name IS the programmatic opt-in).
    /// Regenerated diffs are REVIEWED, never rubber-stamped — a golden
    /// change must trace to an intended ruling (e.g. 2026-08-06: the
    /// transformer golden had pinned a float-reassociated residual sum the
    /// dtype-gate removed).
    #[test]
    #[ignore = "golden regenerator — run explicitly by name"]
    fn regenerate_golden_plans() {
        for stem in GOLDEN_SCRIPTS {
            let graph = extract_fixture(&format!("{stem}.egg"));
            let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect(stem);
            fs::write(format!("output/{stem}.bufferized.txt"), plan.summary())
                .expect("golden writes");
        }
    }

    /// RANK 3: the goldens are ENFORCED pins, not documentation. Every boundary
    /// script's plan summary must match the checked-in output/<stem>.bufferized.txt
    /// byte for byte (and, via the `anti (N):` section, pin WAR edges).
    #[test]
    fn golden_plans_are_pinned() {
        for stem in GOLDEN_SCRIPTS {
            let golden_path = format!("output/{stem}.bufferized.txt");
            let graph = extract_fixture(&format!("{stem}.egg"));
            let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect(stem);
            let golden = fs::read_to_string(&golden_path).unwrap_or_else(|_| {
                panic!("{golden_path} missing — run regenerate_golden_plans by name")
            });
            assert_eq!(plan.summary(), golden, "plan for {stem} diverged from golden");
        }
    }

    /// The DPS rewrite: every capable op gains one poison destination per
    /// result (trailing operands), produced by synthesized Poison nodes whose
    /// values carry the tied result's LAYOUT (the equivalence gate keys on it).
    #[test]
    fn dps_rewrite_appends_tied_poison_destinations() {
        use crate::layout_ir::ExtractedNode;
        let graph = extract_fixture("boundary_in_place_mutation.egg");
        let rewritten = crate::dps::dps_rewrite(&graph);

        // The DPS form keeps the base op's label (label policy: IR names are
        // never edited), so DPS-ness is witnessed semantically: among
        // extractable ops only DPS forms answer to_dps() = None.
        let (op, result_layout) = rewritten
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op)
                    if op.op.label() == "SqrtFunctionalGeneric" && op.op.to_dps().is_none() =>
                {
                    Some((op.clone(), op.outputs[0].layout.eclass.clone()))
                }
                _ => None,
            })
            .expect("Sqrt was rewritten to its DPS form");
        assert_eq!(op.inputs.len(), 2, "input + one destination");
        assert!(op.inputs[1].port.starts_with("dest"));
        // The poison producer exists and its value carries the result's layout.
        let poison = rewritten
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(p) if p.op.label() == "Poison" => Some(p.clone()),
                _ => None,
            })
            .expect("Poison producer synthesized");
        assert_eq!(poison.outputs[0].eclass, op.inputs[1].value);
        assert_eq!(poison.outputs[0].layout.eclass, result_layout);
    }

    /// The zero-input source path (R7): iota extracts with an EMPTY operand
    /// list, and after the DPS rewrite its appended destination is the op's
    /// ONLY operand — the degenerate case of the trailing-destination
    /// convention, proven end to end on the boundary_iota fixture.
    #[test]
    fn iota_extracts_as_zero_input_source() {
        use crate::layout_ir::ExtractedNode;
        let graph = extract_fixture("boundary_iota.egg");
        let extracted = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label() == "IotaGeneric" => Some(op.clone()),
                _ => None,
            })
            .expect("iota op extracted");
        assert!(extracted.inputs.is_empty(), "a source op has no operands");
        assert_eq!(extracted.outputs.len(), 1);

        let rewritten = crate::dps::dps_rewrite(&graph);
        let dps = rewritten
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op)
                    if op.op.label() == "IotaGeneric" && op.op.to_dps().is_none() =>
                {
                    Some(op.clone())
                }
                _ => None,
            })
            .expect("iota rewritten to its DPS form");
        assert_eq!(dps.inputs.len(), 1, "the destination is the only operand");
        assert!(dps.inputs[0].port.starts_with("dest"));
    }

    /// The in-place scatter path (R8 dual — the user's in-place ruling):
    /// forced to the MUTATING implementation, the KV-cache update writes
    /// straight into the caller's cache buffer — no result allocation, no
    /// BufferCopy, and the operand slots read init/src/coord0/coord1.
    /// (Full extraction currently prefers the functional form on a cost
    /// tie and REPAIRS the in-place demand with a copy — the golden pins
    /// that honest outcome; extraction preference is a deferred lever.)
    #[test]
    fn scatter_mutating_updates_the_cache_in_place() {
        use crate::layout_ir::ExtractedNode;
        let graph = extract_fixture_with_ops(
            "boundary_scatter.egg",
            &["LayoutTensorOpScatterMutatingGeneric", "LayoutTensorOpIotaGeneric"],
        );
        let scatter = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label() == "ScatterMutatingGeneric" => {
                    Some(op.clone())
                }
                _ => None,
            })
            .expect("mutating scatter extracted");
        assert_eq!(scatter.inputs.len(), 4, "init + src + one coord per init axis");
        assert_eq!(scatter.inputs[0].port, "init");
        assert_eq!(scatter.inputs[1].port, "src");
        assert_eq!(scatter.inputs[2].port, "coord0");
        assert_eq!(scatter.inputs[3].port, "coord1");

        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph))
            .expect("in-place scatter bufferizes");
        let summary = plan.summary();
        assert!(summary.contains("ScatterMutatingGeneric"), "{summary}");
        assert!(
            !summary.contains("BufferCopy"),
            "in-place must be zero-copy: {summary}"
        );
    }

    /// A scatter whose coordinate shapes genuinely disagree with src's
    /// shape survives the gated main stratum untouched (fail-open) and
    /// dies inside the terminal stratum's coordinate-shape-lock closure —
    /// egglog's own merge machinery raises the error.
    #[test]
    fn scatter_with_disagreeing_src_and_coordinate_shapes_dies_in_the_terminal_stratum() {
        let preamble = crate::egglog_snippet::assembled_program();
        let script = r#"
(let cache_shape (ShapeLit (IntExprCons (IntLit 6) (IntExprCons (IntLit 4) (IntExprNil)))))
(let cache (LogicalTensorInputLit (LogicalIdLit "cache") cache_shape (F32)))
(let row_shape (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil)))))
(let wide_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let src (LogicalTensorInputLit (LogicalIdLit "src") wide_shape (F32)))
(let position (LogicalTensorInputLit (LogicalIdLit "position") row_shape (Int)))
(let column (LogicalTensorInputLit (LogicalIdLit "column") row_shape (Int)))
(let bad_scatter
  (LogicalScatter cache
    (LogicalTensorCons position (LogicalTensorCons column (LogicalTensorNil)))
    src))
(run-schedule (saturate (run)) (saturate (run fixpoint-invariants)))
"#;
        let program = format!("{preamble}

{script}");
        let mut egraph = crate::egglog_snippet::new_egraph();
        let err = egraph
            .parse_and_run_program(None, &program)
            .expect_err("the shape lock must collide in the terminal stratum");
        assert!(
            err.to_string().contains("shape-of"),
            "the error should come from the shape-of merge: {err}"
        );
    }

    /// The variable-arity path (R8): gather's rank is DATA read out of the
    /// e-graph by its matcher — the first data-carrying instance. The
    /// extracted op has 1 + rank operands in declared order, and after the
    /// DPS rewrite the destination is the trailing operand.
    #[test]
    fn gather_extracts_with_rank_counted_operands() {
        use crate::layout_ir::ExtractedNode;
        let graph = extract_fixture("boundary_gather.egg");
        let extracted = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label() == "GatherGeneric" => Some(op.clone()),
                _ => None,
            })
            .expect("gather op extracted");
        assert_eq!(extracted.inputs.len(), 3, "data + one coord per data axis");
        assert_eq!(extracted.inputs[0].port, "data");
        assert_eq!(extracted.inputs[1].port, "coord0");
        assert_eq!(extracted.inputs[2].port, "coord1");

        let rewritten = crate::dps::dps_rewrite(&graph);
        let dps = rewritten
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op)
                    if op.op.label() == "GatherGeneric" && op.op.to_dps().is_none() =>
                {
                    Some(op.clone())
                }
                _ => None,
            })
            .expect("gather rewritten to its DPS form");
        assert_eq!(dps.inputs.len(), 4, "the destination trails the coords");
        assert_eq!(dps.inputs[3].port, "dest0");
    }

    /// The terminal-stratum checker (user ruling 2026-07-23: every
    /// invariant lives in egglog): a gather whose coordinate shapes are
    /// GENUINELY unequal survives the gated main stratum untouched
    /// (fail-open, no race), then dies inside (run fixpoint-invariants) —
    /// the unguarded shape closure writes both shapes into one :no-merge
    /// slot and egglog's own merge machinery raises the error. Inline
    /// program — the pipeline sweep never runs intentionally ill-formed
    /// scripts.
    #[test]
    fn gather_with_disagreeing_coordinate_shapes_dies_in_the_terminal_stratum() {
        let preamble = crate::egglog_snippet::assembled_program();
        let script = r#"
(let three_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let four_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let ids_three (LogicalTensorInputLit (LogicalIdLit "ids_three") three_shape (Int)))
(let ids_four (LogicalTensorInputLit (LogicalIdLit "ids_four") four_shape (Int)))
(let ids_three_layout (RightMajorContiguousElementLayoutLit three_shape (bits-of (Int))))
(let ids_four_layout (RightMajorContiguousElementLayoutLit four_shape (bits-of (Int))))
(let ids_three_lt (LayoutTensorLit ids_three ids_three_layout))
(let ids_four_lt (LayoutTensorLit ids_four ids_four_layout))
(let data_shape (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 7) (IntExprNil)))))
(let data (LogicalTensorInputLit (LogicalIdLit "data") data_shape (F32)))
(let data_layout (RightMajorContiguousElementLayoutLit data_shape (bits-of (F32))))
(let data_lt (LayoutTensorLit data data_layout))
(let bad_gather
  (LogicalGather data
    (LogicalTensorCons ids_three
      (LogicalTensorCons ids_four (LogicalTensorNil)))))
(let out_layout (RightMajorContiguousElementLayoutLit three_shape (bits-of (F32))))
(let out_lt (LayoutTensorLit bad_gather out_layout))
(let out_buffer (BufferLit 500))
(set (buffer-access-of out_buffer) (ReadWrite))
(set (buffer-freed-by out_buffer) (CallerFrees))
(let output
  (BufferOutputLit (BufferTensorCons (BufferTensorLit out_lt out_buffer) (BufferTensorNil))))
(run-schedule (saturate (run)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let program = format!("{preamble}\n\n{script}");
        let mut egraph = crate::egglog_snippet::new_egraph();
        let err = egraph
            .parse_and_run_program(None, &program)
            .expect_err("the shape closure must collide in the terminal stratum");
        assert!(
            err.to_string().contains("shape-of"),
            "the error should come from the shape-of merge: {err}"
        );
    }

    /// The missing-bounds half of the iota gate (user ruling 2026-07-23:
    /// no seeding — the lattice's top stays ABSENCE): the CONSTRUCTION
    /// SITE emits the bounds demand as post-schedule checks, and a check
    /// whose lookup finds nothing fails the program loudly at the
    /// fixpoint. An iota over an expression nothing bounds — an unseeded
    /// IntVar — dies there.
    #[test]
    fn iota_with_no_derivable_bounds_dies_in_the_terminal_stratum() {
        let preamble = crate::egglog_snippet::assembled_program();
        let script = r#"
(let mystery_var (IntVar "mystery_var"))
(let unsafe_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let unbounded_iota (LogicalIota mystery_var unsafe_shape))
(run-schedule (saturate (run)) (saturate (run fixpoint-invariants)))
(check (= ?demanded_lower (lower-bound-of mystery_var)))
(check (= ?demanded_upper (upper-bound-of mystery_var)))
"#;
        let program = format!("{preamble}\n\n{script}");
        let mut egraph = crate::egglog_snippet::new_egraph();
        let err = egraph
            .parse_and_run_program(None, &program)
            .expect_err("the constructor-site demand must fail on the absent bound");
        assert!(
            err.to_string().contains("lower-bound-of"),
            "the error should cite the absent bounds demand: {err}"
        );
    }

    /// Idempotency: DPS forms answer to_dps() = None, so a second rewrite is a
    /// no-op (same node and edge counts).
    #[test]
    fn dps_rewrite_is_idempotent() {
        let graph = extract_fixture("basic_program.egg");
        let once = crate::dps::dps_rewrite(&graph);
        let twice = crate::dps::dps_rewrite(&once);
        assert_eq!(once.dag.node_count(), twice.dag.node_count());
        assert_eq!(once.dag.edge_count(), twice.dag.edge_count());
    }

    /// A view is free: the consumer reads the PARENT's buffer directly, and
    /// the view op leaves no node in the plan (folded like a poison producer).
    #[test]
    fn view_reads_parent_buffer_with_zero_plan_nodes() {
        use crate::bufferize::BufferNode;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        let r = g.op(
            Box::new(MockOp { reads: vec![true], ..Default::default() }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert_eq!(
            plan.value_buffer[&v], plan.value_buffer[&x],
            "the view derives its parent's buffer:\n{}",
            plan.summary()
        );
        for idx in plan.dag.node_indices() {
            if let BufferNode::Compute { op, reads, .. } = &plan.dag[idx] {
                assert_ne!(op.label(), "MockView", "views are folded:\n{}", plan.summary());
                // Storage nodes (alloc/free) read no operands; the invariant
                // under test is about the view's CONSUMER.
                if op.label() == "MockOp" {
                    assert_eq!(
                        reads[0], plan.value_buffer[&x],
                        "the consumer reads the parent buffer directly"
                    );
                }
            }
        }
    }

    /// A view of READ-ONLY storage is legal: it writes nothing, so the
    /// writability veto does not apply (previously any in-place candidate on a
    /// Read-granted buffer was refused — correct only for writers).
    #[test]
    fn view_of_read_only_input_is_admitted() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadOnly, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        let r = g.op(
            Box::new(MockOp { reads: vec![true], ..Default::default() }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let plan = bufferize::bufferize(&g.build()).expect("a view writes nothing — Read is fine");
        assert_eq!(plan.value_buffer[&v], plan.value_buffer[&x]);
    }

    /// REGION-BLIND conservatism, pinned (user decision): a writer into a
    /// viewed buffer conflicts with a live, unordered read THROUGH the view —
    /// even though the view's layout differs (the regions might be disjoint;
    /// without the interval oracle we must assume overlap). The optional
    /// writer yields out-of-place; the view, decided in phase 1, stands.
    #[test]
    fn writer_into_viewed_buffer_yields_out_of_place() {
        use crate::bufferize::BufferId;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        let s = g.op(
            Box::new(MockOp { reads: vec![true], ..Default::default() }),
            &[&v],
            &[("s", "rm")],
        )[0]
        .clone();
        let r_w = g.op(Box::new(MockOp::write_only_dest()), &[&x], &[("rW", "rm")])[0].clone();
        g.output(&r_w, "D");
        g.output(&s, "E");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert!(
            matches!(plan.value_buffer[&r_w], BufferId::Allocated(_)),
            "the writer yields to a fresh allocation:\n{}",
            plan.summary()
        );
        assert_eq!(plan.value_buffer[&v], plan.value_buffer[&x]);
    }

    /// Chained views resolve link by link to the root storage.
    #[test]
    fn chained_views_bind_through_to_the_root_buffer() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v1 = g.op(Box::new(MockView), &[&x], &[("v1", "row0")])[0].clone();
        let v2 = g.op(Box::new(MockView), &[&v1], &[("v2", "cell0")])[0].clone();
        let r = g.op(
            Box::new(MockOp { reads: vec![true], ..Default::default() }),
            &[&v2],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");
        assert_eq!(plan.value_buffer[&v2], plan.value_buffer[&x]);
        assert_eq!(plan.value_buffer[&v1], plan.value_buffer[&x]);
    }

    /// A view bound straight to an output slot on its parent's own buffer is a
    /// pure pass-through: zero copies, zero allocations, and the final binding
    /// carries the VIEW's layout (region-aware boundary bookkeeping).
    #[test]
    fn view_passthrough_to_output_slot() {
        use crate::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        g.output(&v, "B");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert!(plan.buffers.keys().all(|id| matches!(id, BufferId::Boundary(_))));
        assert!(
            plan.dag
                .node_indices()
                .all(|idx| !matches!(&plan.dag[idx], BufferNode::BufferCopy { .. })),
            "pass-through needs no copy:\n{}",
            plan.summary()
        );
    }

    /// THE REAL VIEW OP, plan level (Step 3): `IndexMapApplyView` feeding a
    /// compute op contributes ZERO plan nodes — the result binds its parent's
    /// buffer and the consumer's kernel reads that storage directly. Same
    /// shape as the MockView pin above, but proving the shipped op's declared
    /// contract (un-read operand, un-written result, Must(0→0)) drives the
    /// fold.
    #[test]
    fn real_view_op_feeds_compute_with_zero_plan_nodes() {
        use crate::bufferize::BufferNode;
        use crate::ssa_reference::ops::IndexMapApplyView;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(IndexMapApplyView), &[&x], &[("v", "row0")])[0].clone();
        let r = g.op(
            Box::new(MockOp { reads: vec![true], ..Default::default() }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert_eq!(
            plan.value_buffer[&v], plan.value_buffer[&x],
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
        use crate::bufferize::BufferNode;
        use crate::ssa_reference::ops::IndexMapApplyView;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(IndexMapApplyView), &[&x], &[("v", "row0")])[0].clone();
        g.output(&v, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        assert_eq!(
            plan.value_buffer[&v], plan.value_buffer[&x],
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
        assert_eq!(copies[0].0, plan.value_buffer[&x], "copied from the parent's buffer");
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
        use crate::bufferize::{BufferId, BufferNode};

        let graph = extract_fixture("boundary_view_feeds_compute.egg");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        assert!(
            plan.buffers.keys().all(|id| matches!(id, BufferId::Boundary(_))),
            "zero allocations:\n{}",
            plan.summary()
        );
        let launch: Vec<BufferId> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferInput { slots } => {
                    Some(slots.iter().map(|slot| slot.buffer.clone()))
                }
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(launch.len(), 1, "one caller input:\n{}", plan.summary());

        let mut computes = 0;
        for idx in plan.dag.node_indices() {
            match &plan.dag[idx] {
                BufferNode::BufferCopy { .. } => panic!("zero copies:\n{}", plan.summary()),
                BufferNode::Compute { op, reads, writes, .. } => {
                    computes += 1;
                    assert_eq!(op.label(), "SqrtFunctionalGeneric", "{}", plan.summary());
                    assert_eq!(
                        reads[0], launch[0],
                        "the kernel reads the input buffer directly through the folded view"
                    );
                    assert_ne!(
                        writes[0], launch[0],
                        "the read-only input is never written:\n{}",
                        plan.summary()
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            computes, 1,
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
        use crate::bufferize::{BufferId, BufferNode};

        let graph = extract_fixture("boundary_write_into_viewed_buffer.egg");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        let launch: Vec<BufferId> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferInput { slots } => {
                    Some(slots.iter().map(|slot| slot.buffer.clone()))
                }
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
                BufferNode::Compute { op, reads, writes, .. } => match op.label() {
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
        assert_eq!(copies[0].0, sqrt_writes, "copied from the rejected writer's allocation");
        assert_eq!(copies[0].1, viewed, "into the demanded output slot's buffer");
    }

    /// ALLOC/FREE PHASE 3, the donated-input fixture end to end: a READ-ONLY
    /// ProgramFrees input (read-then-destroy — Access and FreedBy orthogonal)
    /// gets exactly one BufferFree consuming the donated buffer, and the
    /// certificate's lifetime arms hold: the free is ordered after the
    /// kernel's read (the anti edge), no alloc exists for caller storage,
    /// and the donated buffer backs no output slot.
    #[test]
    fn donated_input_fixture_frees_the_donated_buffer() {
        use crate::bufferize::{BufferId, BufferNode};

        let graph = extract_fixture("boundary_donated_input.egg");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        assert!(
            plan.buffers.keys().all(|id| matches!(id, BufferId::Boundary(_))),
            "no allocations:\n{}",
            plan.summary()
        );
        let launch: Vec<BufferId> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferInput { slots } => {
                    Some(slots.iter().map(|slot| slot.buffer.clone()))
                }
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(launch.len(), 1, "one caller input:\n{}", plan.summary());

        let mut frees = 0;
        for idx in plan.dag.node_indices() {
            match &plan.dag[idx] {
                BufferNode::BufferCopy { .. } => panic!("no copies:\n{}", plan.summary()),
                BufferNode::Compute { op, reads, .. } => {
                    if op.label() == "BufferFree" {
                        frees += 1;
                        assert_eq!(
                            reads[0], launch[0],
                            "the free consumes the donated input buffer:\n{}",
                            plan.summary()
                        );
                    }
                }
                _ => {}
            }
        }
        assert_eq!(frees, 1, "donated storage is freed exactly once:\n{}", plan.summary());
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
        use crate::layout_ir::ExtractedNode;

        let graph = extract_fixture("basic_program.egg");
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

    /// MUST-ALLOCATE OUTPUTS — a pinned gap, not a feature. basic_program's
    /// output buffers (7, 8, 10) appear only in BufferOutputLit, never in BufferInputLit;
    /// under existence-is-BufferInputLit-membership that storage does NOT exist at
    /// launch, so the plan itself is responsible for allocating and returning
    /// it. Today the planner cannot express that obligation: every boundary
    /// buffer is interned caller-owned, silently coercing "allocate and
    /// return" into "the caller passes it in pre-allocated". This test pins
    /// the coercion; when the Alloc-node / plan-manifest stage lands, these
    /// buffers must become system-owned allocations that escape, and this
    /// test must be flipped to assert exactly that.
    #[test]
    fn must_alloc_outputs_are_coerced_to_caller_provided_storage() {
        use std::collections::HashSet;

        use crate::bufferize::{BufferId, BufferNode, Owner};

        let graph = extract_fixture("basic_program.egg");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");

        // Storage that exists at launch = the buffers backing BufferInput
        // nodes (the extractor admits those from BufferInputLit membership).
        let launch_existing: HashSet<BufferId> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferInput { slots } => Some(slots.iter()),
                _ => None,
            })
            .flatten()
            .map(|slot| slot.buffer.clone())
            .collect();

        // Every output destination that is not launch-existing storage is a
        // must-allocate output.
        let must_alloc: Vec<BufferId> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferOutput { slots } => Some(slots.iter()),
                _ => None,
            })
            .flatten()
            .filter(|slot| !launch_existing.contains(&slot.buffer))
            .map(|slot| slot.buffer.clone())
            .collect();
        assert_eq!(
            must_alloc.len(),
            3,
            "basic_program has three must-allocate outputs (z, w, left_view):\n{}",
            plan.summary()
        );

        for buffer in &must_alloc {
            let info = &plan.buffers[buffer];
            // THE COERCION: must-allocate storage is still pinned
            // caller-owned — the plan demands a handle the caller was never
            // supposed to provide, instead of allocating and returning it.
            assert!(
                matches!(buffer, BufferId::Boundary(_)),
                "must-allocate output still interned as a pinned boundary buffer"
            );
            assert!(
                matches!(info.owner, Owner::Caller),
                "must-allocate output still recorded caller-owned: {}",
                info.label
            );
        }
    }

    /// End to end through the pipeline: every DPS destination is admitted in
    /// place into its poison's fresh storage — reads[dest] == writes[tied] for
    /// every compute node, and Poison producers are folded (no compute node).
    #[test]
    fn dps_destinations_admitted_and_poisons_folded() {
        use crate::bufferize::BufferNode;
        let graph = extract_fixture("basic_program.egg");
        let plan = bufferize::bufferize(&crate::dps::dps_rewrite(&graph)).expect("bufferizes");
        let mut computes = 0;
        for idx in plan.dag.node_indices() {
            if let BufferNode::Compute { op, reads, writes, .. } = &plan.dag[idx] {
                assert_ne!(op.label(), "Poison", "poison producers must be folded");
                if reads.is_empty() || writes.is_empty() {
                    continue; // storage nodes (alloc/free): no DPS shape
                }
                computes += 1;
                // Trailing destination operands read the buffer they write.
                let dests = writes.len();
                let data = reads.len() - dests;
                for (j, write) in writes.iter().enumerate() {
                    assert_eq!(&reads[data + j], write, "{}", plan.summary());
                }
            }
        }
        assert!(computes > 0);
    }

    /// COMMITTED-WRITER INTERFERENCE (review-confirmed miscompile, fixed):
    /// two ops each claim the SAME poison value as their write-only
    /// destination; r2 is dead, r1 is the output. Whichever candidate the
    /// bottom-up order decides first is admitted blind (the other's read set
    /// is not yet aliased); the reads-only scan then let the second one
    /// through too, and the seed bound BOTH unordered writers into the
    /// caller's output buffer — a scheduler picking the dead writer last
    /// left r2's bytes where the boundary promised r1. The committed-writer
    /// check (try_in_place step 3, MLIR's write-side interference) must
    /// reject the second candidate: the output value ends up the buffer's
    /// only writer, whichever decision order the toposort produces.
    #[test]
    fn unordered_second_writer_of_shared_destination_is_rejected() {
        use crate::bufferize::BufferNode;

        let mut g = TestGraph::new();
        let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")])[0].clone();
        let r2 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&p],
            &[("r2", "rm")],
        )[0]
        .clone();
        let r1 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&p],
            &[("r1", "rm")],
        )[0]
        .clone();
        g.output(&r1, "D");
        let plan = bufferize::bufferize(&g.build()).expect("bufferizes");

        // The two writers must not share storage: exactly one of the two
        // in-place candidates survives, so r1's buffer has ONE compute writer.
        assert_ne!(
            plan.value_buffer[&r1], plan.value_buffer[&r2],
            "unordered writers must not share storage:\n{}",
            plan.summary()
        );
        let writers_of_r1_buffer = plan
            .dag
            .node_indices()
            .filter(|&idx| {
                // A BufferAlloc lists the buffer but installs no binding —
                // it is not a writer in the sense under test.
                matches!(&plan.dag[idx], BufferNode::Compute { reads, writes, .. }
                    if !reads.is_empty() && writes.contains(&plan.value_buffer[&r1]))
            })
            .count();
        assert_eq!(
            writers_of_r1_buffer, 1,
            "the output value's buffer has exactly one writer:\n{}",
            plan.summary()
        );
    }

    /// Multi-destination DPS (review finding: previously unexercised): an
    /// AddMulFused node run through dps_rewrite gains TWO destinations, each
    /// tied to its own result — the pairs must land in DISTINCT allocations,
    /// with reads[dest_j] == writes[j] for both.
    #[test]
    fn multi_destination_pairs_get_distinct_allocations() {
        use crate::bufferize::BufferNode;
        use crate::ssa_reference::ops::AddMulFused;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.input("y", "C", Access::ReadWrite, "rm");
        let results = g.op(Box::new(AddMulFused), &[&x, &y], &[("sum", "rm"), ("prod", "rm")]);
        let (sum, prod) = (results[0].clone(), results[1].clone());
        g.output(&sum, "D");
        g.output(&prod, "E");

        let rewritten = crate::dps::dps_rewrite(&g.build());
        let plan = bufferize::bufferize(&rewritten).expect("bufferizes");

        // Each (poison, result) pair on its own allocation — never shared.
        assert_ne!(
            plan.value_buffer[&sum], plan.value_buffer[&prod],
            "{}",
            plan.summary()
        );
        // The fused compute reads both destinations it writes, positionally.
        let (reads, writes) = plan
            .dag
            .node_indices()
            .find_map(|idx| match &plan.dag[idx] {
                BufferNode::Compute { op, reads, writes, .. } if op.label() == "AddMulFusedGeneric" => {
                    Some((reads.clone(), writes.clone()))
                }
                _ => None,
            })
            .expect("fused DPS node present");
        assert_eq!(reads.len(), 4, "2 data + 2 destinations");
        assert_eq!(&reads[2], &writes[0]);
        assert_eq!(&reads[3], &writes[1]);
    }

    /// Slot tables render each declared Must tie as one full-width row
    /// (`in ↔ out`) whose single port serves both compass sides — the tied
    /// result's edge leaves the SAME row its destination entered. Pinned on
    /// the multi-tie op because no fixture extracts one: both pairs must
    /// span, and both results must dock at their tie rows' east sides.
    #[test]
    fn slot_tables_render_ties_as_spanning_rows() {
        use crate::ssa_reference::ops::AddMulFused;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.input("y", "C", Access::ReadWrite, "rm");
        let results = g.op(Box::new(AddMulFused), &[&x, &y], &[("sum", "rm"), ("prod", "rm")]);
        g.output(&results[0], "D");
        g.output(&results[1], "E");

        let rewritten = crate::dps::dps_rewrite(&g.build());
        let dot = rewritten.to_dot();
        for span in ["dest0 ↔ out0</TD>", "dest1 ↔ out1</TD>"] {
            assert!(dot.contains(span), "missing tie row {span:?} in:\n{dot}");
        }
        for dock in [":p_dest0:e ->", ":p_dest1:e ->"] {
            assert!(dot.contains(dock), "tied result not docked at {dock:?} in:\n{dot}");
        }
    }
}




































#[cfg(test)]
mod intcoordvar_probe {
    //! Probes for the (IntCoordVar LogicalTensor i64) constructibility
    //! analysis. Preamble is never touched on disk; programs assemble in
    //! memory. Run: cargo test intcoordvar_probe -- --nocapture

    /// Step-by-step runner: prints each step's Ok/Err so every claim is
    /// separately observable.
    fn run_steps(egraph: &mut egglog::EGraph, steps: &[(&str, &str)]) {
        for (label, prog) in steps {
            match egraph.parse_and_run_program(None, prog) {
                Ok(lines) => {
                    println!("X {label}: Ok");
                    for l in lines {
                        println!("X {label} out: {l}");
                    }
                }
                Err(e) => println!("X {label}: Err: {e}"),
            }
        }
    }

    /// Two-sort mutually recursive mini-datatype (a stand-in for
    /// LogicalTensor <-> IndexMap <-> IntCoordVar). Emulates the
    /// self-referential coordinate tag via the name+union route and
    /// observes: cyclic-class closure under congruence, extraction, and
    /// the nonce cost (structurally identical views never merge).
    #[test]
    fn x_probe_cyclic_selftag() {
        let mut egraph = crate::egglog_snippet::new_egraph();
        run_steps(
            &mut egraph,
            &[
                (
                    "setup",
                    r#"
(datatype*
  (PT (PTInput String)
      (PTName String)
      (PTView PT PMap))
  (PMap (PMapLit PCoord))
  (PCoord (PCoordVar PT i64)))
(let data (PTInput "data"))
(let nm (PTName "v1"))
(let v (PTView data (PMapLit (PCoordVar nm 0))))
(union nm v)
"#,
                ),
                (
                    "cyclic-check",
                    "(check (= v (PTView data (PMapLit (PCoordVar v 0)))))",
                ),
                ("extract-v", "(extract v)"),
                ("extract-v-variants", "(extract v 4)"),
                (
                    "nonce-no-merge",
                    r#"
(let n1 (PTName "nonce1"))
(let n2 (PTName "nonce2"))
(let w1 (PTView data (PMapLit (PCoordVar n1 0))))
(let w2 (PTView data (PMapLit (PCoordVar n2 0))))
(union n1 w1)
(union n2 w2)
(fail (check (= w1 w2)))
"#,
                ),
                (
                    "hashcons-baseline",
                    r#"
(let u1 (PTView data (PMapLit (PCoordVar data 0))))
(let u2 (PTView data (PMapLit (PCoordVar data 0))))
(check (= u1 u2))
"#,
                ),
            ],
        );
    }

    /// Today's behavior on the user's own discriminating example: a
    /// 3-vector with slices v[0..2] and v[1..3]. Both slices' coordinate
    /// is the identical term (CoordVar out_shape 0); the VIEW terms stay
    /// distinct; an identically-written view hash-conses (free CSE).
    #[test]
    fn x_probe_slice_views_today() {
        let mut full = crate::egglog_snippet::assembled_program().to_string();
        full.push_str(
            r#"
(let vec_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprNil))))
(let v_in (LogicalTensorInputLit (LogicalIdLit "v") vec_shape (F32)))
(let coord (CoordVar out_shape 0))
(let slice_a (LogicalIndexMapApply v_in
  (IndexMapLit (IntExprCons coord (IntExprNil))) out_shape))
(let slice_b (LogicalIndexMapApply v_in
  (IndexMapLit (IntExprCons (IntAdd coord (IntLit 1)) (IntExprNil))) out_shape))
(let slice_a2 (LogicalIndexMapApply v_in
  (IndexMapLit (IntExprCons (CoordVar out_shape 0) (IntExprNil))) out_shape))
(let v_layout (RightMajorContiguousElementLayoutLit vec_shape (bits-of (F32))))
(let v_lt (LayoutTensorLit v_in v_layout))
(run-schedule (saturate (run)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#,
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        match egraph.parse_and_run_program(None, &full) {
            Ok(_) => println!("X slice-fixture: Ok"),
            Err(e) => {
                println!("X slice-fixture: Err: {e}");
                return;
            }
        }
        run_steps(
            &mut egraph,
            &[
                ("cse-check", "(check (= slice_a slice_a2))"),
                ("distinct-views", "(fail (check (= slice_a slice_b)))"),
                (
                    "shared-coord-term",
                    "(check (= coord (CoordVar out_shape 0)))",
                ),
                (
                    "corruption-pairs",
                    r#"
(fail (check (= (IntLit 0) (IntLit 1))))
(fail (check (= (IntLit 0) (IntLit 2))))
(fail (check (= (IntLit 0) (IntLit 3))))
(fail (check (= (IntLit 0) (IntLit 4))))
(fail (check (= (IntLit 0) (IntLit 5))))
(fail (check (= (IntLit 1) (IntLit 2))))
(fail (check (= (IntLit 1) (IntLit 3))))
(fail (check (= (IntLit 1) (IntLit 4))))
(fail (check (= (IntLit 1) (IntLit 5))))
(fail (check (= (IntLit 2) (IntLit 3))))
(fail (check (= (IntLit 2) (IntLit 4))))
(fail (check (= (IntLit 2) (IntLit 5))))
(fail (check (= (IntLit 3) (IntLit 4))))
(fail (check (= (IntLit 3) (IntLit 5))))
(fail (check (= (IntLit 4) (IntLit 5))))
"#,
                ),
            ],
        );
    }
}

/// The test harness's search budget — the SAME genetic algorithm as the
/// module-level ladder tests, sized for a suite of hundreds of graphs
/// (ruling 2026-08-06: everything in the main tree runs the genetic
/// implementation search — there is no plain-walk bypass). Deterministic
/// (fixed seed); 2 generations × 4 genomes exercises random genomes plus
/// the mutation step without profiling 64 candidates per differential.
pub fn harness_search_options() -> crate::implementation_search::ImplementationSearchOptions {
    crate::implementation_search::ImplementationSearchOptions {
        generations: 2,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
    }
}

/// M3 Step 4b: the NATIVE test harness — recorder model + reference
/// binding + dyn pins as tight [n,n] bounds seeds, saturated, then the
/// GENETIC IMPLEMENTATION SEARCH picks the winning plan (executing every
/// candidate with the given data), which executes and stays loaded for
/// output reads. The frontend candle differentials and the ssa_reference
/// differentials both run through here — the same load → bind → search →
/// execute ladder as the nn module tests, on the harness budget above.
pub fn run_ssa(cx: &crate::graph::Graph, inputs: &[(petgraph::graph::NodeIndex, Vec<f32>)]) -> crate::ssa_reference::SsaReferenceRuntime {
    let mut rt = crate::ssa_reference::SsaReferenceRuntime::load(cx)
        .expect("recorder clean for a covered graph");
    let mut vars: Vec<_> = cx.dyn_map.iter().collect();
    vars.sort();
    for (var, value) in vars {
        rt.bind_dyn_range(*var, *value as u64, *value as u64)
            .expect("dyn pin binds");
    }
    let data: rustc_hash::FxHashMap<_, _> = inputs.iter().cloned().collect();
    rt.search(&data, &harness_search_options())
        .expect("search finds a plan");
    for (node, values) in inputs {
        rt.set_data(*node, values.clone());
    }
    rt.execute().expect("winner executes");
    rt
}


#[cfg(test)]
mod stage4b_probes {

    /// (History: this module held the Step-4b degenerate-broadcast
    /// unsoundness pin — the stride recovery walks welding the zero
    /// class. Fixed by the testimony landing; the repro is now the LIVE
    /// regression `degenerate_broadcast_runs_clean` below.)
    ///
    /// PINNED KNOWN GAP (Step 4b): a PURE-IDENTITY graph — an input
    /// directly output(), no ops — panics the axis-extent-lock tripwire
    /// natively (input and output bindings share one BufferLit and one
    /// logical value). The model zoo's identity-add idiom (Topic E, yolo)
    /// exists for exactly this; the real fix is binding-level (4d/M4
    /// aliasing/donation design).
    #[test]
    fn pinned_pure_identity_output() {
        let mut cx = crate::graph::Graph::new();
        let a = cx.tensor(2);
        let b = a.output();
        let rt = crate::test_support::run_ssa(&cx, &[(a.id, vec![1.0, 2.0])]);
        let got = rt.get_f32(b.id).unwrap();
        assert_eq!(got, &vec![1.0, 2.0]);
    }

    /// Austin's stride-destructuring contract (chain world, 2026-08-04):
    /// live axes yield their stride class, stride-1 axes yield Unit, a
    /// zero slot on a live axis is the DETERMINED broadcast Some(Zero),
    /// and a provably extent-1 slot is None — the free parameter each
    /// consumer resolves for itself.
    #[test]
    fn chain_strides_destructure_contract() {
        use crate::extractor::{chain_strides, ChainStride};
        let body = r#"
(let psh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let p (RightMajorContiguousElementLayoutLit psh (bits-of (F32))))
(let plog (LogicalTensorInputLit (LogicalIdLit "p") psh (F32)))
(let plt (LayoutTensorLit plog p))
(let osh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil))))))
(let v (LogicalIndexMapApply plog (IndexMapLit (IntExprCons (CoordVar osh 2) (IntExprCons (CoordVar osh 0) (IntExprNil)))) osh))
(let dsh (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 2) (IntExprNil)))))
(let d (RightMajorContiguousElementLayoutLit dsh (bits-of (F32))))
(run-schedule (saturate (run)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!("{}\n\n{body}", crate::egglog_snippet::assembled_program());
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph.parse_and_run_program(None, &full).expect("program runs");
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;

        let by_let = |name: &str| {
            serialized
                .class_data
                .iter()
                .find(|(_, data)| data.extra.get("let").map(String::as_str) == Some(name))
                .map(|(class, _)| class.clone())
                .unwrap_or_else(|| panic!("let {name} not found"))
        };

        // Parent (2,3) right-major: strides [3, 1].
        let p = chain_strides(&serialized, &by_let("p")).expect("parent destructures");
        assert_eq!(p.len(), 2);
        assert!(matches!(p[0], Some(ChainStride::Expr(_))), "{p:?}");
        assert_eq!(p[1], Some(ChainStride::Unit), "{p:?}");

        // Degenerate (1,2): the extent-1 slot is the FREE parameter.
        let d = chain_strides(&serialized, &by_let("d")).expect("degenerate destructures");
        assert_eq!(d[0], None, "extent-1 slot must be the consumer's choice: {d:?}");
        assert_eq!(d[1], Some(ChainStride::Unit), "{d:?}");

        // Broadcast view (2,5,3): [3, DETERMINED 0, 1].
        let v_logical = by_let("v");
        let view_layout = serialized
            .nodes
            .iter()
            .find_map(|(_, node)| {
                if node.op != "LayoutTensorLit" {
                    return None;
                }
                let logical = node.children.first()?;
                if serialized.nid_to_cid(logical) != &v_logical {
                    return None;
                }
                // The materializing copy pairs the SAME logical with an
                // RM target layout — skip it; the view's own layout is
                // the composed (non-contiguous) one.
                let layout = serialized.nid_to_cid(node.children.get(1)?).clone();
                let is_rm = serialized.nodes.iter().any(|(nid2, n2)| {
                    n2.op == "RightMajorContiguousElementLayoutLit"
                        && serialized.nid_to_cid(nid2) == &layout
                });
                if is_rm { None } else { Some(layout) }
            })
            .expect("view LayoutTensor exists");
        let v = chain_strides(&serialized, &view_layout).expect("view destructures");
        assert!(matches!(v[0], Some(ChainStride::Expr(_))), "{v:?}");
        assert_eq!(v[1], Some(ChainStride::Zero), "broadcast axis is determined: {v:?}");
        assert_eq!(v[2], Some(ChainStride::Unit), "{v:?}");
    }

    /// SATURATION PROFILER (Austin commissioned 2026-08-05): run each
    /// specimen's FULL schedule once and read egglog's own RunReport —
    /// per-rule search+apply times and per-ruleset totals. Suspect under
    /// test: the a9fb4b55 commit (map-range lock + diagonal arms)
    /// roughly doubled movement-shard wall time.
    /// Run: RRX=1 cargo test rrx_profile -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic — run explicitly by name"]
    fn rrx_profile() {
        let specimens: Vec<(&str, String)> = vec![
            ("slice_pad(27x10)", {
                let mut cx = crate::graph::Graph::new();
                let a = cx.tensor((27usize, 10usize));
                let _out = a
                    .slice((2..6, 7..10))
                    .pad(((1usize, 2usize), (1usize, 0usize)), 0.)
                    .output();
                let (pre, _is, _os, post) = cx.logical.native_parts().expect("recorder clean");
                format!("{pre}{}{post}", crate::reference_binding::SCHEDULE)
            }),
            ("batch_matmul(2,3,4,5)", {
                let mut cx = crate::graph::Graph::new();
                let a = cx.tensor((2usize, 3usize, 4usize));
                let b = cx.tensor((4usize, 5usize));
                let _out = a.matmul(b).output();
                let (pre, _is, _os, post) = cx.logical.native_parts().expect("recorder clean");
                format!("{pre}{}{post}", crate::reference_binding::SCHEDULE)
            }),
        ];
        // The fixed floor: parse + declare the assembled preamble alone.
        {
            let preamble = crate::egglog_snippet::assembled_program();
            let start = std::time::Instant::now();
            let mut egraph = crate::egglog_snippet::new_egraph();
            egraph.parse_and_run_program(None, &preamble).expect("preamble loads");
            eprintln!(
                "[prof] ===== preamble only (parse+declare, no body/schedule): {:.2}s, {} lines =====",
                start.elapsed().as_secs_f64(),
                preamble.lines().count()
            );
        }
        for (name, body) in specimens {
            let full = format!("{}\n\n{body}", crate::egglog_snippet::assembled_program());
            let mut egraph = crate::egglog_snippet::new_egraph();
            let start = std::time::Instant::now();
            let outputs = egraph.parse_and_run_program(None, &full).expect("program runs");
            let wall = start.elapsed().as_secs_f64();
            eprintln!("\n[prof] ===== {name}: total wall {wall:.2}s =====");
            for chunk in &outputs {
                let egglog::CommandOutput::RunSchedule(report) = chunk else {
                    continue;
                };
                let mut rulesets: Vec<(String, f64)> = report
                    .search_and_apply_time_per_ruleset
                    .iter()
                    .map(|(name, time)| (name.to_string(), time.as_secs_f64()))
                    .collect();
                rulesets.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (ruleset, secs) in rulesets.iter().take(6) {
                    let label = if ruleset.is_empty() { "(default)" } else { ruleset };
                    let rebuild = report
                        .rebuild_time_per_ruleset
                        .iter()
                        .find(|(name, _)| name.as_ref() == label || (label == "(default)" && name.is_empty()))
                        .map(|(_, time)| time.as_secs_f64())
                        .unwrap_or(0.0);
                    let merge = report
                        .merge_time_per_ruleset
                        .iter()
                        .find(|(name, _)| name.as_ref() == label || (label == "(default)" && name.is_empty()))
                        .map(|(_, time)| time.as_secs_f64())
                        .unwrap_or(0.0);
                    eprintln!(
                        "[prof] ruleset {label:<28} search+apply {secs:>7.3}s  rebuild {rebuild:>7.3}s  merge {merge:>7.3}s"
                    );
                }
                let mut rules: Vec<(String, f64, usize)> = report
                    .search_and_apply_time_per_rule
                    .iter()
                    .map(|(name, time)| {
                        let matches = report
                            .num_matches_per_rule
                            .get(name)
                            .copied()
                            .unwrap_or(0);
                        (name.to_string(), time.as_secs_f64(), matches)
                    })
                    .collect();
                rules.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (rule, secs, matches) in rules.iter().take(15) {
                    let flat: String = rule.split_whitespace().collect::<Vec<_>>().join(" ");
                    let head: String = flat.chars().take(110).collect();
                    eprintln!("[prof] {secs:>7.3}s x{matches:<6} {head}");
                }
            }
        }
    }

    /// G7 MAP-ENTRY RANGE LOCK (Austin ruled 2026-08-05): the
    /// adversary's degenerate bypass — a map reading an extent-1 parent
    /// axis at a 5-extent coordinate — must now DIE LOUDLY in the
    /// invariants stratum instead of composing silently.
    #[test]
    fn map_range_lock_fires_on_degenerate_bypass() {
        let body = r#"
(let psh (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil)))))
(let plog (LogicalTensorInputLit (LogicalIdLit "p") psh (F32)))
(let p (RightMajorContiguousElementLayoutLit psh (bits-of (F32))))
(let plt (LayoutTensorLit plog p))
(let osh (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 4) (IntExprNil)))))
(let v (LogicalIndexMapApply plog (IndexMapLit (IntExprCons (CoordVar osh 1) (IntExprCons (CoordVar osh 0) (IntExprNil)))) osh))
(run-schedule (saturate (run)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!("{}\n\n{body}", crate::egglog_snippet::assembled_program());
        let err = crate::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &full)
            .expect_err("out-of-range map over a degenerate axis must panic");
        assert!(
            err.to_string().contains("range lock"),
            "wrong failure: {err}"
        );
    }

    /// LIVE REGRESSION (was the pinned Step-4b unsoundness): the
    /// rank-0 -> [1] broadcast that used to weld the zero class (the
    /// recovery walks turned "presentations [1] and [0] are both valid"
    /// into IntExpr equalities 0 ≡ 1 ≡ 32). With recovery deleted and
    /// strided-presentation testimony in its place, it runs end-to-end
    /// with every tripwire live.
    #[test]
    fn degenerate_broadcast_runs_clean() {
        let mut cx = crate::graph::Graph::new();
        let a = cx.tensor(1);
        let b = (a * 2.0).output();
        let rt = crate::test_support::run_ssa(&cx, &[(a.id, vec![0.5])]);
        let got = rt.get_f32(b.id).unwrap();
        assert!((got[0] - 1.0).abs() < 1e-6, "{got:?}");
    }
}

/// Deep-dive landing 1/6 regression: the subst-of in-range guard.
/// LANDED form = smallest-possible-box interval arm + structural
/// same-extent arm; LEGACY form (reconstructed by reverse surgery) =
/// entry within the coordinate's CURRENT interval. Five standalone
/// egglog scenarios, exact verdicts asserted for both forms.
///
/// On the two-phase scenarios: pins happen ONCE, pre-run (Austin's
/// ruling — buckets are separate egglog programs; there is no
/// operational mid-run pin). The late `(set (upper-bound-of ...))`
/// models a DERIVED tightening arriving late in saturation, plus
/// seed-order sensitivity generally: the legacy guard's verdict depends
/// on WHEN a monotone fact lands; the landed guard's does not. The core
/// defect is fixpoint-level regardless: legacy admits dyn-extent
/// entries that are out-of-box under admissible valuations.
#[cfg(test)]
mod subst_guard_study {
    const LANDED_GUARD: &str = "    (= ?entry_lower (lower-bound-of ?entry))\n    (= ?entry_upper (upper-bound-of ?entry))\n    (>= ?entry_lower (bigint 0))\n    (= ?extent_lower (lower-bound-of ?extent))\n    (<= ?entry_upper (- ?extent_lower (bigint 1)))";

    const LEGACY_GUARD: &str = "    (= ?entry_lower (lower-bound-of ?entry))\n    (= ?entry_upper (upper-bound-of ?entry))\n    (= ?coord_lower (lower-bound-of ?expr))\n    (= ?coord_upper (upper-bound-of ?expr))\n    (>= ?entry_lower ?coord_lower)\n    (<= ?entry_upper ?coord_upper)";

    /// The structural arm's rule text as landed (comment elided; the
    /// unique premise pair suffices as the removal anchor).
    const STRUCTURAL_ARM_ANCHOR: &str = "(rule\n  (\n    (subst-demand ?expr ?map ?source_shape)\n    (= ?expr (CoordVar ?source_shape ?axis))\n    (= ?map (IndexMapLit ?entries))\n    (= ?entry (expr-list-nth-from-end ?entries ?axis))\n    (= ?source_shape (ShapeLit ?source_dims))\n    (= ?extent (expr-list-nth-from-end ?source_dims ?axis))\n    (= ?entry (CoordVar ?entry_shape ?entry_axis))\n    (= ?entry_shape (ShapeLit ?entry_dims))\n    (= ?entry_extent (expr-list-nth-from-end ?entry_dims ?entry_axis))\n    (= ?entry_extent ?extent)\n  )\n  ((set (subst-of ?expr ?map ?source_shape) ?entry))\n)";

    fn variant(text: &str, name: &str) -> String {
        match name {
            "landed" => {
                assert!(text.contains(LANDED_GUARD), "landed guard text drifted");
                assert!(
                    text.contains(STRUCTURAL_ARM_ANCHOR),
                    "structural arm text drifted"
                );
                text.to_string()
            }
            "legacy" => {
                assert!(text.contains(LANDED_GUARD), "landed guard text drifted");
                let t = text.replacen(LANDED_GUARD, LEGACY_GUARD, 1);
                assert!(t.contains(STRUCTURAL_ARM_ANCHOR), "structural arm text drifted");
                t.replacen(STRUCTURAL_ARM_ANCHOR, "; [study: structural arm removed]", 1)
            }
            other => panic!("unknown variant {other}"),
        }
    }

    fn scenarios() -> Vec<(&'static str, String)> {
        let sg1_common = "\
(let sgn (IntVar \"sgn\"))\n\
(set (lower-bound-of sgn) (bigint 1))\n\
(set (upper-bound-of sgn) (bigint 8))\n\
(let sg_cout_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))\n\
(let sg_cout (CoordVar sg_cout_shape 0))\n\
(let sg_entry (IntAdd sg_cout (IntLit 5)))\n\
(let sg_src (ShapeLit (IntExprCons sgn (IntExprNil))))\n\
(let sg_map (IndexMapLit (IntExprCons sg_entry (IntExprNil))))\n\
(let sg_coord (CoordVar sg_src 0))\n\
(subst-demand sg_coord sg_map sg_src)\n\
(run-schedule (saturate (run)))\n";
        let sg4_common = "\
(let s4n (IntVar \"s4n\"))\n\
(set (lower-bound-of s4n) (bigint 1))\n\
(set (upper-bound-of s4n) (bigint 8))\n\
(let s4_entry (IntLit 5))\n\
(let s4_src (ShapeLit (IntExprCons s4n (IntExprNil))))\n\
(let s4_map (IndexMapLit (IntExprCons s4_entry (IntExprNil))))\n\
(let s4_coord (CoordVar s4_src 0))\n\
(subst-demand s4_coord s4_map s4_src)\n\
(run-schedule (saturate (run)))\n";
        vec![
            ("sg1_admits", format!("{sg1_common}(check (= (subst-of sg_coord sg_map sg_src) sg_entry))\n")),
            ("sg1_tighten", format!("{sg1_common}(set (upper-bound-of sgn) (bigint 1))\n(run-schedule (saturate (run)))\n")),
            ("sg2_static", "\
(let s2_cout_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))\n\
(let s2_cout (CoordVar s2_cout_shape 0))\n\
(let s2_entry (IntAdd s2_cout (IntLit 1)))\n\
(let s2_src (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))\n\
(let s2_map (IndexMapLit (IntExprCons s2_entry (IntExprNil))))\n\
(let s2_coord (CoordVar s2_src 0))\n\
(subst-demand s2_coord s2_map s2_src)\n\
(run-schedule (saturate (run)))\n\
(check (= (subst-of s2_coord s2_map s2_src) s2_entry))\n".to_string()),
            ("sg3_identity", "\
(let s3n (IntVar \"s3n\"))\n\
(set (lower-bound-of s3n) (bigint 1))\n\
(set (upper-bound-of s3n) (bigint 8))\n\
(let s3_entry_shape (ShapeLit (IntExprCons s3n (IntExprCons s3n (IntExprNil)))))\n\
(let s3_entry (CoordVar s3_entry_shape 1))\n\
(let s3_src (ShapeLit (IntExprCons s3n (IntExprNil))))\n\
(let s3_map (IndexMapLit (IntExprCons s3_entry (IntExprNil))))\n\
(let s3_coord (CoordVar s3_src 0))\n\
(subst-demand s3_coord s3_map s3_src)\n\
(run-schedule (saturate (run)))\n\
(check (= (subst-of s3_coord s3_map s3_src) s3_entry))\n".to_string()),
            ("sg4_admits", format!("{s4}(check (= (subst-of s4_coord s4_map s4_src) s4_entry))\n", s4 = sg4_common)),
            ("sg4_tighten", format!("{s4}(set (upper-bound-of s4n) (bigint 1))\n(run-schedule (saturate (run)))\n", s4 = sg4_common)),
        ]
    }

    fn run_verdict(text: &str) -> &'static str {
        let mut egraph = crate::egglog_snippet::new_egraph();
        match egraph.parse_and_run_program(None, text) {
            Ok(_) => "ok",
            Err(err) => {
                let e = err.to_string();
                if e.contains("crossed IntExpr bounds") {
                    "panic-crossed-bounds"
                } else if e.contains("distinct integer literals") {
                    "panic-distinct-literals"
                } else if e.contains("Check failed") || e.contains("check failed") {
                    "refused"
                } else {
                    eprintln!("STUDY-ERR detail: {}", &e[..e.len().min(400)]);
                    "other-error"
                }
            }
        }
    }

    #[test]
    fn subst_guard_landed_vs_legacy() {
        // (variant, scenario) -> expected verdict. "refused" = the image
        // correctly did not fire (fail-closed); the legacy admissions and
        // detonations are the documented unsoundness.
        let expected = [
            ("landed", "sg1_admits", "refused"),
            ("landed", "sg1_tighten", "ok"),
            ("landed", "sg2_static", "ok"),
            ("landed", "sg3_identity", "ok"),
            ("landed", "sg4_admits", "refused"),
            ("landed", "sg4_tighten", "ok"),
            ("legacy", "sg1_admits", "ok"),
            ("legacy", "sg1_tighten", "panic-crossed-bounds"),
            ("legacy", "sg2_static", "ok"),
            ("legacy", "sg3_identity", "ok"),
            ("legacy", "sg4_admits", "ok"),
            ("legacy", "sg4_tighten", "panic-crossed-bounds"),
        ];
        let base = crate::egglog_snippet::assembled_program();
        for var_name in ["landed", "legacy"] {
            let varied = variant(&base, var_name);
            for (scen_name, tail) in scenarios() {
                let verdict = run_verdict(&format!("{varied}\n{tail}"));
                let want = expected
                    .iter()
                    .find(|(v, s, _)| *v == var_name && *s == scen_name)
                    .map(|(_, _, w)| *w)
                    .unwrap();
                eprintln!("STUDY {var_name:>7} | {scen_name:<12} | {verdict}");
                assert_eq!(
                    verdict, want,
                    "guard regression: {var_name}/{scen_name} expected {want}, got {verdict}"
                );
            }
        }
    }
}

