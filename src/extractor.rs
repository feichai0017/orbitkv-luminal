use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use egraph_serialize::{ClassId, EGraph, Node, NodeId};
use petgraph::graph::{DiGraph, NodeIndex};

use crate::layout_ir::{
    Access, BufferInfo, ExtractedDag, ExtractedEdge, ExtractedGraph, ExtractedNode,
    ExtractionSite, FreedBy, InputNode, LayoutInfo, LayoutIrOp, LayoutTensorInfo, LogicalInfo,
    OpInput, OpMatcher, OpNode, OutputNode, OutputSlot, ops::built_in_matchers,
};
use crate::logical_op::{LogicalRender, logical_op_for};

#[derive(Debug)]
struct Extractor<'a> {
    egraph: &'a EGraph,
    /// The op registry: egglog constructor name → registered matcher, built
    /// from [`built_in_matchers`] (optionally filtered by the allow-list —
    /// the test/debug lever that forces extraction through specific ops;
    /// structural plumbing — inputs, outputs, buffer lists — is never
    /// filtered). This registry is the ONLY dispatch: an enode whose label
    /// has no entry here simply offers no implementation candidate.
    matchers: HashMap<&'static str, Box<dyn OpMatcher>>,
    class_nodes: HashMap<ClassId, Vec<NodeId>>,
    render_class_nodes: HashMap<ClassId, Vec<NodeId>>,
    op_specs: HashMap<ClassId, Vec<OpSpec>>,
    producer_index: HashMap<ClassId, Vec<ProducerRef>>,
    input_terminals: HashMap<ClassId, InputInfo>,
    /// The search genome, when this walk is genome-driven (see [`Genome`]).
    /// `None` = the deterministic fixture extractor (min-cost tooling).
    genome: Option<&'a Genome>,
    memo: HashMap<ClassId, Option<Plan>>,
}

#[derive(Debug, Clone)]
struct InputInfo {
    buffer_tensor_class: ClassId,
    buffer_tensor_enode: NodeId,
    buffer_id_class: ClassId,
    logical_name: String,
}

#[derive(Debug, Clone)]
struct Plan {
    cost: u32,
    copies: u32,
    source_eclass: Option<ClassId>,
    source_enode: Option<NodeId>,
    selected_output_index: Option<usize>,
    input_list: Vec<ClassId>,
    output_list: Vec<ClassId>,
    kind: PlanKind,
    children: Vec<PlanChild>,
    metadata: Vec<PlanMeta>,
}

#[derive(Debug, Clone)]
struct PlanChild {
    port: String,
    class: ClassId,
}

#[derive(Debug, Clone)]
struct PlanMeta {
    name: &'static str,
    class: ClassId,
}

/// Extractor-internal ONLY. `PlanKind` is the selection/cost IR at *e-graph*
/// granularity — it includes plumbing (buffer-list cons/nil, boundary literals)
/// that has no place in the clean dataflow output. It is deliberately private and
/// must never leak out of this module: the public artifact is [`ExtractedGraph`]
/// (whose nodes are [`ExtractedNode`]), which every `Plan` is lowered into by
/// `build_extracted_graph`. If you find yourself wanting to expose a `PlanKind`
/// (or `Plan`) across the module boundary, lower it to an `ExtractedNode` instead.
#[derive(Debug, Clone)]
enum PlanKind {
    Input(InputInfo),
    BufferOutputLit,
    BufferTensorCons,
    BufferTensorNil,
    BufferTensorLit {
        buffer_id_class: ClassId,
        logical_name: String,
    },
    LayoutIr(Box<dyn LayoutIrOp>),
}

#[derive(Debug, Clone)]
struct OpSpec {
    inputs: Vec<ClassId>,
    outputs: Vec<ClassId>,
}

#[derive(Debug, Clone)]
struct ProducerRef {
    op_class: ClassId,
    spec_index: usize,
    output_index: usize,
}

pub fn extract_layout_ir(egraph: &EGraph) -> Result<Option<ExtractedGraph>> {
    extract_layout_ir_with_ops(egraph, None)
}

/// [`extract_layout_ir`] restricted to an allow-list of LayoutTensorOp
/// constructor names — the test/debug lever for exercising a specific
/// implementation. `None` allows every op; a program not implementable
/// within the list fails extraction loudly.
///
/// Both this and [`extract_layout_ir`] are the DETERMINISTIC FIXTURE
/// extractor (min-cost, tie-broken) — tooling for fixtures and goldens,
/// not the selection mechanism. The search path is
/// [`extract_layout_ir_with_genome`].
pub fn extract_layout_ir_with_ops(
    egraph: &EGraph,
    allowed_ops: Option<&[&str]>,
) -> Result<Option<ExtractedGraph>> {
    let allowed = allowed_ops.map(|ops| ops.iter().map(|op| op.to_string()).collect());
    let mut extractor = Extractor::new(egraph, allowed, None);
    extractor.extract()
}

/// One genome choice: the concrete implementation enode that produces the
/// keyed LayoutTensor class, and which of its output slots carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerChoice {
    pub enode: NodeId,
    pub output_index: usize,
}

/// The search genome: a per-LayoutTensor-class producer selection. The
/// genome is the ONLY authority under [`extract_layout_ir_with_genome`] —
/// it replaces both the cost choice and first-emission slot claiming. The
/// contract is TOTALITY over produced classes: a demanded class that has
/// producers but no entry fails extraction loudly (no silent substitution).
/// Entries for classes the walk never demands are dead rows — legal and
/// free (the reachability-kill semantics).
#[derive(Debug, Clone, Default)]
pub struct Genome {
    pub choices: HashMap<ClassId, ProducerChoice>,
}

/// Genome-driven extraction — the selection adapter's walk. Starts from the
/// binding outputs and instantiates exactly the genome's chosen producer
/// per demanded class; multi-output instances dedup by enode; output slots
/// the genome does NOT assign to their instance write anonymous waste
/// destinations (fresh synthetic values, allocated and freed unread —
/// waste-allowed, priced by profiling).
#[allow(dead_code)] // selection-adapter API: test harness here; lib export in the luminal graft
pub fn extract_layout_ir_with_genome(
    egraph: &EGraph,
    genome: &Genome,
) -> Result<Option<ExtractedGraph>> {
    extract_layout_ir_with_genome_and_ops(egraph, genome, None)
}

/// Genome-driven extraction under a backend implementation allow-list (the
/// genome and the inventory compose: choices must come from allowed ops).
#[allow(dead_code)] // selection-adapter API: test harness here; lib export in the luminal graft
pub fn extract_layout_ir_with_genome_and_ops(
    egraph: &EGraph,
    genome: &Genome,
    allowed_ops: Option<&[&str]>,
) -> Result<Option<ExtractedGraph>> {
    let allowed = allowed_ops.map(|ops| ops.iter().map(|op| op.to_string()).collect());
    let mut extractor = Extractor::new(egraph, allowed, Some(genome));
    extractor.extract()
}

/// Every LayoutTensor class's candidate producers, as
/// `(implementation constructor name, choice)` pairs sorted for
/// determinism — the raw material genome construction and mutation draw
/// from. Classes with no producers (boundary inputs) are absent.
#[allow(dead_code)] // selection-adapter API: test harness here; lib export in the luminal graft
pub fn producer_index(
    egraph: &EGraph,
) -> std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>> {
    producer_index_with_ops(egraph, None)
}

/// [`producer_index`] restricted to a backend's implementation allow-list —
/// the genome space only offers what the executing backend implements.
#[allow(dead_code)] // selection-adapter API: test harness here; lib export in the luminal graft
pub fn producer_index_with_ops(
    egraph: &EGraph,
    allowed_ops: Option<&[&str]>,
) -> std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>> {
    let allowed = allowed_ops.map(|ops| ops.iter().map(|op| op.to_string()).collect());
    let extractor = Extractor::new(egraph, allowed, None);
    let mut index = std::collections::BTreeMap::new();
    for (class, producers) in &extractor.producer_index {
        let mut entries: Vec<(String, ProducerChoice)> = Vec::new();
        for producer in producers {
            let Some(node_ids) = extractor.class_nodes.get(&producer.op_class) else {
                continue;
            };
            for node_id in node_ids {
                let Some(node) = extractor.egraph.nodes.get(node_id) else {
                    continue;
                };
                if node.subsumed || !extractor.matchers.contains_key(node.op.as_str()) {
                    continue;
                }
                entries.push((
                    node.op.clone(),
                    ProducerChoice {
                        enode: node_id.clone(),
                        output_index: producer.output_index,
                    },
                ));
            }
        }
        if !entries.is_empty() {
            entries.sort_by_key(|(name, choice)| {
                (name.clone(), choice.enode.to_string(), choice.output_index)
            });
            index.insert(class.clone(), entries);
        }
    }
    index
}

/// A stable fingerprint of a plan's SHAPE: the chosen instances (enode +
/// claimed slots) and the dataflow between them. Many genomes map to one
/// plan (dead rows are unread), so the search hashes the built plan and
/// skips re-profiling duplicates (ruling 2026-07-27).
#[allow(dead_code)] // selection-adapter API: test harness here; lib export in the luminal graft
pub fn plan_fingerprint(graph: &ExtractedGraph) -> u64 {
    use petgraph::visit::EdgeRef;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for index in graph.dag.node_indices() {
        match &graph.dag[index] {
            ExtractedNode::LayoutOp(op) => {
                "op".hash(&mut hasher);
                op.op.label().hash(&mut hasher);
                if let crate::layout_ir::Provenance::Extracted {
                    source_enode,
                    selected_output_index,
                    ..
                } = &op.provenance
                {
                    source_enode.to_string().hash(&mut hasher);
                    selected_output_index.hash(&mut hasher);
                }
                for output in &op.outputs {
                    output.eclass.to_string().hash(&mut hasher);
                }
            }
            ExtractedNode::BufferInput(input) => {
                "in".hash(&mut hasher);
                input.value.eclass.to_string().hash(&mut hasher);
            }
            ExtractedNode::BufferOutput(output) => {
                "out".hash(&mut hasher);
                for slot in &output.slots {
                    slot.index.hash(&mut hasher);
                    slot.value.to_string().hash(&mut hasher);
                }
            }
        }
    }
    for edge in graph.dag.edge_references() {
        edge.source().index().hash(&mut hasher);
        edge.target().index().hash(&mut hasher);
        edge.weight().port.hash(&mut hasher);
        edge.weight().value.to_string().hash(&mut hasher);
    }
    hasher.finish()
}

impl<'a> Extractor<'a> {
    fn new(
        egraph: &'a EGraph,
        allowed_ops: Option<HashSet<String>>,
        genome: Option<&'a Genome>,
    ) -> Self {
        let matchers = built_in_matchers()
            .into_iter()
            .filter(|matcher| {
                allowed_ops
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(matcher.egglog_constructor()))
            })
            .map(|matcher| (matcher.egglog_constructor(), matcher))
            .collect();
        let class_nodes = class_nodes(egraph);
        let render_class_nodes = render_class_nodes(egraph);
        let (op_specs, producer_index) = collect_op_specs(egraph, &render_class_nodes);
        let output_buffer_classes = collect_output_buffer_classes(egraph, &class_nodes);
        let input_buffer_classes = collect_input_buffer_classes(egraph, &class_nodes);
        let input_terminals = collect_input_terminals(
            egraph,
            &render_class_nodes,
            &output_buffer_classes,
            &input_buffer_classes,
        );

        Self {
            egraph,
            matchers,
            class_nodes,
            render_class_nodes,
            op_specs,
            producer_index,
            input_terminals,
            genome,
            memo: HashMap::new(),
        }
    }

    fn extract(&mut self) -> Result<Option<ExtractedGraph>> {
        let roots = output_root_classes(self.egraph);
        if roots.is_empty() {
            return Ok(None);
        }

                for root in &roots {
            if self.best_plan(root, &mut HashSet::new()).is_none() {
                bail!("failed to extract LayoutIR graph from BufferOutputLit eclass {root}");
            }
        }

        Ok(Some(self.build_extracted_graph(&roots)?))
    }

    fn best_plan(&mut self, class: &ClassId, visiting: &mut HashSet<ClassId>) -> Option<Plan> {
        if let Some(plan) = self.memo.get(class) {
            return plan.clone();
        }
        if !visiting.insert(class.clone()) {
            return None;
        }

        let mut best = self.input_terminals.get(class).map(|input| Plan {
            cost: 0,
            copies: 0,
            source_eclass: None,
            source_enode: None,
            selected_output_index: None,
            input_list: Vec::new(),
            output_list: Vec::new(),
            kind: PlanKind::Input(input.clone()),
            children: Vec::new(),
            metadata: Vec::new(),
        });

        let mut candidates = Vec::new();
        let node_ids = self.class_nodes.get(class).cloned().unwrap_or_default();
        for node_id in node_ids {
            let Some(node) = self.egraph.nodes.get(&node_id) else {
                continue;
            };
            if let Some(candidate) = self.candidate_for_node(&node_id, node) {
                candidates.push(candidate);
            }
        }
        candidates.extend(self.producer_candidates_for_output(class));

        // GENOME AUTHORITY (the selection adapter): when a genome drives the
        // walk, a class with producers is produced by EXACTLY its chosen
        // enode/slot — never by cost, never by first-emission claiming. A
        // produced class missing from the genome violates the total-genome
        // contract: candidates empty out and extraction fails loudly at the
        // root (fail-open, no silent substitution).
        if let Some(genome) = self.genome {
            if self.producer_index.contains_key(class) {
                match genome.choices.get(class) {
                    Some(choice) => candidates.retain(|candidate| {
                        candidate.source_enode.as_ref() == Some(&choice.enode)
                            && candidate.selected_output_index == Some(choice.output_index)
                    }),
                    None => candidates.clear(),
                }
            }
        }

        for candidate in candidates {
            let mut cost = candidate.base_cost();
            let mut copies = candidate.copy_count();
            let mut child_plans = Vec::with_capacity(candidate.children.len());

            for child in &candidate.children {
                let Some(child_plan) = self.best_plan(&child.class, visiting) else {
                    child_plans.clear();
                    break;
                };
                cost += child_plan.cost;
                copies += child_plan.copies;
                child_plans.push(child.clone());
            }

            if child_plans.len() != candidate.children.len() {
                continue;
            }

            let plan = Plan {
                cost,
                copies,
                source_eclass: candidate
                    .source_eclass
                    .clone()
                    .or_else(|| Some(class.clone())),
                source_enode: candidate.source_enode,
                selected_output_index: candidate.selected_output_index,
                input_list: candidate.input_list,
                output_list: candidate.output_list,
                kind: candidate.kind,
                children: child_plans,
                metadata: candidate.metadata,
            };

            if self.is_better(&plan, best.as_ref()) {
                best = Some(plan);
            }
        }

        visiting.remove(class);
        self.memo.insert(class.clone(), best.clone());
        best
    }

    fn candidate_for_node(&self, node_id: &NodeId, node: &Node) -> Option<Candidate> {
        if node.subsumed || node.op == "[...]" {
            return None;
        }

        let op = node.op.as_str();
        match op {
            "BufferOutputLit" => Some(Candidate::structural(
                node_id,
                PlanKind::BufferOutputLit,
                self.children(node, &[("outputs", 0)])?,
            )),
            "BufferTensorCons" => Some(Candidate::structural(
                node_id,
                PlanKind::BufferTensorCons,
                self.children(node, &[("head", 0), ("tail", 1)])?,
            )),
            "BufferTensorNil" => Some(Candidate::structural(
                node_id,
                PlanKind::BufferTensorNil,
                vec![],
            )),
            "BufferTensorLit" => {
                let layout_tensor_class = self.child_class(node, 0)?;
                let buffer_id_class = self.child_class(node, 1)?;
                Some(Candidate::structural(
                    node_id,
                    PlanKind::BufferTensorLit {
                        buffer_id_class: buffer_id_class.clone(),
                        logical_name: self
                            .logical_name_from_layout_tensor(&layout_tensor_class)
                            .unwrap_or_else(|| layout_tensor_class.to_string()),
                    },
                    vec![PlanChild {
                        port: "tensor".to_string(),
                        class: layout_tensor_class,
                    }],
                ))
            }
            _ => None,
        }
    }

    fn producer_candidates_for_output(&self, output_class: &ClassId) -> Vec<Candidate> {
        let Some(producers) = self.producer_index.get(output_class) else {
            return Vec::new();
        };

        let mut candidates = Vec::new();
        for producer in producers {
            let Some(spec) = self
                .op_specs
                .get(&producer.op_class)
                .and_then(|specs| specs.get(producer.spec_index))
            else {
                continue;
            };
            let Some(node_ids) = self.class_nodes.get(&producer.op_class) else {
                continue;
            };

            for node_id in node_ids {
                let Some(node) = self.egraph.nodes.get(node_id) else {
                    continue;
                };
                if let Some(candidate) = self.candidate_for_layout_op(producer, spec, node_id, node)
                {
                    candidates.push(candidate);
                }
            }
        }

        candidates
    }

    fn candidate_for_layout_op(
        &self,
        producer: &ProducerRef,
        spec: &OpSpec,
        node_id: &NodeId,
        node: &Node,
    ) -> Option<Candidate> {
        if node.subsumed || node.op == "[...]" {
            return None;
        }

        // The registry IS the dispatch: an enode whose constructor has no
        // registered matcher (unknown, or excluded by the allow-list — which
        // filters IMPLEMENTATIONS only) offers no candidate here. A MATCHED
        // enode, by contrast, is extractable by construction (the match rules
        // discharged applicability in egglog — see the OpMatcher validity
        // contract), so a metadata slot that fails to resolve is schema
        // drift between the matcher's slot spec and the preamble's
        // constructor arity: a bug, and it panics rather than silently
        // shrinking the candidate space.
        let matcher = self.matchers.get(node.op.as_str())?;
        let metadata = self.metadata(node, matcher.metadata_slots()).unwrap_or_else(|| {
            panic!(
                "schema drift: {} enode {node_id} does not satisfy its matcher's metadata slots {:?}",
                node.op,
                matcher.metadata_slots(),
            )
        });
        let op: Box<dyn LayoutIrOp> = matcher.extract(&ExtractionSite {
            egraph: self.egraph,
            node_id,
            node,
        });

        let children = self.op_children(&spec.inputs, op.as_ref());
        Some(Candidate::layout_ir(
            producer.op_class.clone(),
            node_id,
            producer.output_index,
            spec.inputs.clone(),
            spec.outputs.clone(),
            op,
            children,
            metadata,
        ))
    }

    fn op_children(&self, inputs: &[ClassId], op: &dyn LayoutIrOp) -> Vec<PlanChild> {
        inputs
            .iter()
            .enumerate()
            .map(|(index, class)| PlanChild {
                port: op.operand_name(index),
                class: class.clone(),
            })
            .collect()
    }

    fn children(&self, node: &Node, ports: &[(&'static str, usize)]) -> Option<Vec<PlanChild>> {
        ports
            .iter()
            .map(|(port, index)| {
                Some(PlanChild {
                    port: (*port).to_string(),
                    class: self.child_class(node, *index)?,
                })
            })
            .collect()
    }

    fn metadata(&self, node: &Node, ports: &[(&'static str, usize)]) -> Option<Vec<PlanMeta>> {
        ports
            .iter()
            .map(|(name, index)| {
                Some(PlanMeta {
                    name,
                    class: self.child_class(node, *index)?,
                })
            })
            .collect()
    }

    fn child_class(&self, node: &Node, index: usize) -> Option<ClassId> {
        let child_id = node.children.get(index)?;
        self.egraph
            .nodes
            .get(child_id)
            .map(|child| child.eclass.clone())
    }

    fn render_buffer_id(&self, class: &ClassId) -> String {
        self.render_class_prefer(class, 3, Some("BufferLit"))
    }

    fn layout_tensor_details(&self, class: &ClassId) -> Vec<(String, String)> {
        self.renderer().layout_tensor_details(class)
    }

    fn layout_tensor_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        self.renderer().layout_tensor_parts(class)
    }

    fn buffer_tensor_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        self.renderer().buffer_tensor_parts(class)
    }

    fn class_let_name(&self, class: &ClassId) -> Option<String> {
        self.renderer().class_let_name(class)
    }

    fn class_type(&self, class: &ClassId) -> Option<String> {
        self.renderer().class_type(class)
    }

    fn layout_tensor_label(&self, class: &ClassId) -> String {
        self.renderer().layout_tensor_label(class)
    }

    fn logical_label(&self, class: &ClassId) -> String {
        self.renderer().logical_label(class)
    }

    fn logical_details(&self, class: &ClassId) -> Vec<(String, String)> {
        self.renderer().logical_details(class)
    }

    fn logical_children(&self, class: &ClassId) -> Vec<(&'static str, ClassId)> {
        self.renderer().logical_children(class)
    }

    fn layout_label(&self, class: &ClassId) -> String {
        self.renderer().layout_label(class)
    }

    fn canonical_layout(&self, class: &ClassId) -> String {
        self.renderer().canonical_layout(class)
    }

    fn layout_details(&self, class: &ClassId) -> Vec<(String, String)> {
        self.renderer().layout_details(class)
    }

    fn readable_shape(&self, class: &ClassId) -> Option<String> {
        self.renderer().readable_shape(class)
    }

    fn readable_index_map(&self, class: &ClassId) -> Option<String> {
        self.renderer().readable_index_map(class)
    }

    fn render_class_prefer(
        &self,
        class: &ClassId,
        depth: usize,
        preferred_op: Option<&str>,
    ) -> String {
        self.renderer()
            .render_class_prefer(class, depth, preferred_op)
    }

    fn render_layout_tensor_list(&self, classes: &[ClassId]) -> String {
        let items = classes
            .iter()
            .enumerate()
            .map(|(index, class)| {
                format!("{index}:{}", self.renderer().layout_tensor_summary(class))
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{items}]")
    }

    /// Plan preference: (cost, copies, label) as before, then a CONTENT-based
    /// stable key. The e-graph unions commutative variants (`IntAdd(x,y)` =
    /// `IntAdd(y,x)`) into one class; without a content key, which variant wins
    /// depends on hash-iteration order and flips run to run. Rendering the
    /// source e-node resolves children to let-names/literals, which are stable
    /// across runs — making the (user-blessed) arbitrary tie-break deterministic.
    fn is_better(&self, plan: &Plan, best: Option<&Plan>) -> bool {
        let Some(best) = best else {
            return true;
        };
        (plan.cost, plan.copies, plan_label(plan), self.stable_key(plan))
            < (best.cost, best.copies, plan_label(best), self.stable_key(best))
    }

    fn stable_key(&self, plan: &Plan) -> String {
        plan.source_enode
            .as_ref()
            .map(|enode| self.renderer().render_node(enode, 3))
            .unwrap_or_default()
    }

    fn renderer(&self) -> ClassRenderer<'_> {
        ClassRenderer {
            egraph: self.egraph,
            class_nodes: &self.render_class_nodes,
        }
    }

    fn logical_name_from_layout_tensor(&self, class: &ClassId) -> Option<String> {
        self.renderer().logical_name_from_layout_tensor(class)
    }
}

#[derive(Debug)]
struct Candidate {
    source_eclass: Option<ClassId>,
    source_enode: Option<NodeId>,
    selected_output_index: Option<usize>,
    input_list: Vec<ClassId>,
    output_list: Vec<ClassId>,
    kind: PlanKind,
    children: Vec<PlanChild>,
    metadata: Vec<PlanMeta>,
}

impl Candidate {
    fn structural(source_enode: &NodeId, kind: PlanKind, children: Vec<PlanChild>) -> Self {
        Self {
            source_eclass: None,
            source_enode: Some(source_enode.clone()),
            selected_output_index: None,
            input_list: Vec::new(),
            output_list: Vec::new(),
            kind,
            children,
            metadata: Vec::new(),
        }
    }

    fn layout_ir(
        source_eclass: ClassId,
        source_enode: &NodeId,
        selected_output_index: usize,
        input_list: Vec<ClassId>,
        output_list: Vec<ClassId>,
        op: Box<dyn LayoutIrOp>,
        children: Vec<PlanChild>,
        metadata: Vec<PlanMeta>,
    ) -> Self {
        Self {
            source_eclass: Some(source_eclass),
            source_enode: Some(source_enode.clone()),
            selected_output_index: Some(selected_output_index),
            input_list,
            output_list,
            kind: PlanKind::LayoutIr(op),
            children,
            metadata,
        }
    }

    fn base_cost(&self) -> u32 {
        match &self.kind {
            PlanKind::Input(_) => 0,
            PlanKind::BufferOutputLit
            | PlanKind::BufferTensorCons
            | PlanKind::BufferTensorNil
            | PlanKind::BufferTensorLit { .. } => 0,
            // No per-op special-casing. Cost is a proxy for the data the op moves:
            // one unit per tensor it READS and per result it WRITES, taken from
            // the op's own declared memory effects — so an unnecessary copy is
            // strictly more expensive than not copying, and a metadata view
            // (reads nothing, writes nothing) is honestly free. Every current
            // compute op reads all operands and writes all results, so for them
            // this equals the old slot count.
            //
            // TODO(cost): make this the sum of tensor *sizes* (product of dims),
            // with every symbolic/dynamic dimension assumed to be 100. That is an
            // extraction-only sampling assumption — a stand-in for "typical input
            // sizes" — so copies are penalized by bytes moved, not just op count.
            PlanKind::LayoutIr(op) => {
                let reads = (0..self.input_list.len())
                    .filter(|&operand| op.operand_reads_memory(operand))
                    .count();
                let writes = (0..self.output_list.len())
                    .filter(|&result| op.result_writes_memory(result))
                    .count();
                (reads + writes) as u32
            }
        }
    }

    fn copy_count(&self) -> u32 {
        // Copies are no longer special-cased in extraction; they are just ops that
        // cost what they move (see `base_cost`). Kept returning 0 so the secondary
        // tie-break degenerates to the arbitrary label order.
        0
    }
}

struct ClassRenderer<'a> {
    egraph: &'a EGraph,
    class_nodes: &'a HashMap<ClassId, Vec<NodeId>>,
}

/// The renderer's implementation of the [`LogicalRender`] callbacks: the
/// bridge each [`crate::logical_op::LogicalOp`] formats itself through.
/// Carries the recursion guard (`visiting`) so `child_expr` cycles fall back
/// to labels exactly as direct recursion did.
struct LogicalRenderCtx<'r, 'a, 'v> {
    renderer: &'r ClassRenderer<'a>,
    visiting: &'v mut HashSet<ClassId>,
}

impl LogicalRender for LogicalRenderCtx<'_, '_, '_> {
    fn child_expr(&mut self, node: &Node, index: usize) -> String {
        child_class(self.renderer.egraph, node, index)
            .map(|class| self.renderer.readable_logical_expr(&class, self.visiting))
            .unwrap_or_else(|| "?".to_string())
    }

    fn child_short(
        &mut self,
        node: &Node,
        index: usize,
        depth: usize,
        prefer: Option<&str>,
    ) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .map(|class| self.renderer.render_class_prefer(&class, depth, prefer))
    }

    fn child_shape(&mut self, node: &Node, index: usize) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .and_then(|class| self.renderer.readable_shape(&class))
    }

    fn child_index_map(&mut self, node: &Node, index: usize) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .and_then(|class| self.renderer.readable_index_map(&class))
    }

    fn child_int_expr(&mut self, node: &Node, index: usize) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .map(|class| self.renderer.readable_expr(&class, &mut HashSet::new()))
    }
}

impl<'a> ClassRenderer<'a> {
    fn class_let_name(&self, class: &ClassId) -> Option<String> {
        self.egraph
            .class_data
            .get(class)
            .and_then(|data| data.extra.get("let"))
            .cloned()
    }

    fn class_type(&self, class: &ClassId) -> Option<String> {
        self.egraph
            .class_data
            .get(class)
            .and_then(|data| data.typ.clone())
    }

    fn render_class_prefer(
        &self,
        class: &ClassId,
        depth: usize,
        preferred_op: Option<&str>,
    ) -> String {
        if depth == 0 {
            return class.to_string();
        }

        let Some(node_ids) = self.class_nodes.get(class) else {
            return class.to_string();
        };
        let Some(node_id) = choose_render_node(self.egraph, node_ids, preferred_op) else {
            return class.to_string();
        };

        self.render_node(node_id, depth)
    }

    fn render_class_with_op(&self, class: &ClassId, depth: usize, op: &str) -> Option<String> {
        let node_id = self.node_with_op(class, op)?;
        Some(self.render_node(node_id, depth))
    }

    fn node_with_op(&self, class: &ClassId, op: &str) -> Option<&NodeId> {
        self.class_nodes.get(class)?.iter().find(|node_id| {
            self.egraph
                .nodes
                .get(*node_id)
                .is_some_and(|node| node.op == op)
        })
    }

    fn render_node(&self, node_id: &NodeId, depth: usize) -> String {
        if depth == 0 {
            return self
                .egraph
                .nodes
                .get(node_id)
                .map(|node| node.eclass.to_string())
                .unwrap_or_else(|| node_id.to_string());
        }

        let Some(node) = self.egraph.nodes.get(node_id) else {
            return node_id.to_string();
        };
        if node.children.is_empty() {
            return node.op.clone();
        }

        let args = node
            .children
            .iter()
            .filter_map(|child_id| self.egraph.nodes.get(child_id))
            .map(|child| self.render_class_prefer(&child.eclass, depth - 1, None))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}({args})", node.op)
    }

    fn display_name(&self, class: &ClassId, fallback: impl FnOnce() -> String) -> String {
        self.class_let_name(class).unwrap_or_else(fallback)
    }

    fn logical_name_from_layout_tensor(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LayoutTensorLit" {
                continue;
            }
            let logical_class = child_class(self.egraph, node, 0)?;
            if let Some(name) = self.logical_name_from_logical(&logical_class) {
                return Some(name);
            }
        }
        None
    }

    fn logical_name_from_logical(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LogicalTensorInputLit" && node.op != "LogicalTensorNamed" {
                continue;
            }
            let id_class = child_class(self.egraph, node, 0)?;
            return Some(self.render_class_prefer(&id_class, 2, Some("LogicalIdLit")));
        }
        None
    }

    fn layout_tensor_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LayoutTensorLit" {
                continue;
            }
            return Some((
                child_class(self.egraph, node, 0)?,
                child_class(self.egraph, node, 1)?,
            ));
        }
        None
    }

    fn buffer_tensor_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "BufferTensorLit" {
                continue;
            }
            return Some((
                child_class(self.egraph, node, 0)?,
                child_class(self.egraph, node, 1)?,
            ));
        }
        None
    }

    fn layout_tensor_label(&self, class: &ClassId) -> String {
        self.display_name(class, || {
            self.layout_tensor_parts(class)
                .map(|(logical, _)| self.logical_label(&logical))
                .or_else(|| self.logical_name_from_layout_tensor(class))
                .unwrap_or_else(|| class.to_string())
        })
    }

    fn layout_tensor_summary(&self, class: &ClassId) -> String {
        let label = self.layout_tensor_label(class);
        let Some((logical, layout)) = self.layout_tensor_parts(class) else {
            return label;
        };
        format!(
            "{label}(logical={}, layout={})",
            self.logical_label(&logical),
            self.canonical_layout(&layout)
        )
    }

    fn logical_label(&self, class: &ClassId) -> String {
        self.display_name(class, || {
            let Some(node_id) = self.choose_logical_node(class) else {
                return class.to_string();
            };
            let Some(node) = self.egraph.nodes.get(node_id) else {
                return class.to_string();
            };
            match logical_op_for(node.op.as_str()) {
                Some(op) => op.display_label(
                    node,
                    &mut LogicalRenderCtx { renderer: self, visiting: &mut HashSet::new() },
                ),
                None => self.render_node(node_id, 8),
            }
        })
    }

    fn logical_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let mut details = self.class_details(class);
        details.push((
            "expr".to_string(),
            self.readable_logical_expr(class, &mut HashSet::new()),
        ));
        if let Some(shape) = self.logical_shape(class) {
            details.push(("shape".to_string(), shape));
        }
        if let Some(dtype) = self.logical_dtype(class) {
            details.push(("dtype".to_string(), dtype));
        }
        details
    }

    fn logical_children(&self, class: &ClassId) -> Vec<(&'static str, ClassId)> {
        let Some(node_id) = self.choose_logical_node(class) else {
            return Vec::new();
        };
        let Some(node) = self.egraph.nodes.get(node_id) else {
            return Vec::new();
        };

        let ports: &[(&str, usize)] = logical_op_for(node.op.as_str())
            .map(|op| op.child_ports())
            .unwrap_or(&[]);

        ports
            .iter()
            .filter_map(|(port, index)| {
                let child = child_class(self.egraph, node, *index)?;
                if child == *class {
                    None
                } else {
                    Some((*port, child))
                }
            })
            .collect()
    }

    fn logical_op_name(&self, class: &ClassId) -> Option<String> {
        let node_id = self.choose_logical_node(class)?;
        let node = self.egraph.nodes.get(node_id)?;
        Some(node.op.clone())
    }

    fn choose_logical_node(&self, class: &ClassId) -> Option<&NodeId> {
        let node_ids = self.class_nodes.get(class)?;
        for op in crate::logical_op::built_in_logical_ops() {
            if let Some(node_id) = node_ids.iter().find(|node_id| {
                self.egraph
                    .nodes
                    .get(*node_id)
                    .is_some_and(|node| node.op == op.egglog_constructor())
            }) {
                return Some(node_id);
            }
        }
        choose_render_node(self.egraph, node_ids, None)
    }

    fn readable_logical_expr(&self, class: &ClassId, visiting: &mut HashSet<ClassId>) -> String {
        if !visiting.insert(class.clone()) {
            return self.logical_label(class);
        }

        let rendered = self
            .choose_logical_node(class)
            .and_then(|node_id| {
                let node = self.egraph.nodes.get(node_id)?;
                Some(match logical_op_for(node.op.as_str()) {
                    Some(op) => {
                        op.readable_expr(node, &mut LogicalRenderCtx { renderer: self, visiting })
                    }
                    None => self.render_node(node_id, 16),
                })
            })
            .unwrap_or_else(|| class.to_string());

        visiting.remove(class);
        rendered
    }

    fn layout_label(&self, class: &ClassId) -> String {
        let canonical = self.canonical_layout_short_label(class);
        match self.class_let_name(class) {
            Some(name) if name != canonical => format!("{name}\n{canonical}"),
            Some(name) => name,
            None => canonical,
        }
    }

    fn canonical_layout_short_label(&self, class: &ClassId) -> String {
        if let Some(summary) = self.contiguous_layout_summary(class) {
            summary
        } else if let Some(summary) = self.left_major_layout_summary(class) {
            summary
        } else if let Some(summary) = self.strided_layout_summary(class) {
            summary
        } else if self
            .node_with_op(class, "ElementOffsetExpressionLayoutLit")
            .is_some()
        {
            "ElementOffset".to_string()
        } else if self
            .node_with_op(class, "BitOffsetExpressionLayoutLit")
            .is_some()
        {
            "BitOffset".to_string()
        } else {
            self.render_class_prefer(class, 6, None)
        }
    }

    fn canonical_layout(&self, class: &ClassId) -> String {
        if let Some(summary) = self.contiguous_layout_inline(class) {
            return summary;
        }
        if let Some(summary) = self.left_major_layout_inline(class) {
            return summary;
        }
        if let Some(summary) = self.strided_layout_inline(class) {
            return summary;
        }
        for op in LAYOUT_RENDER_OPS {
            if let Some(rendered) = self.render_class_with_op(class, 16, op) {
                return rendered;
            }
        }
        self.render_class_prefer(class, 16, None)
    }

    fn layout_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let mut details = self.class_details(class);
        details.push(("canonical".to_string(), self.canonical_layout(class)));
        if let Some((shape, bits)) = self.contiguous_layout_shape_bits(class) {
            details.push(("shape".to_string(), shape));
            details.push(("bits".to_string(), bits));
        } else if let Some((shape, bits)) = self.left_major_layout_shape_bits(class) {
            details.push(("shape".to_string(), shape));
            details.push(("bits".to_string(), bits));
        } else if let Some((shape, _strides, bits)) =
            self.strided_layout_shape_strides_bits(class)
        {
            details.push(("shape".to_string(), shape));
            details.push(("bits".to_string(), bits));
        }
        if let Some(contiguous) =
            self.render_class_with_op(class, 16, "RightMajorContiguousElementLayoutLit")
        {
            details.push(("right_major_contiguous".to_string(), contiguous));
        }
        if let Some(left_major) =
            self.render_class_with_op(class, 16, "LeftMajorContiguousElementLayoutLit")
        {
            details.push(("left_major_contiguous".to_string(), left_major));
        }
        if let Some(strided) = self.strided_layout_inline(class) {
            details.push(("strided".to_string(), strided));
        }
        if let Some(element_offset) =
            self.render_class_with_op(class, 16, "ElementOffsetExpressionLayoutLit")
        {
            details.push(("element_offset".to_string(), element_offset));
        }
        details.push((
            "bit_offset".to_string(),
            self.render_class_with_op(class, 32, "BitOffsetExpressionLayoutLit")
                .unwrap_or_else(|| "<none>".to_string()),
        ));
        details
    }

    fn contiguous_layout_summary(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.contiguous_layout_shape_bits(class)?;
        Some(format!("RightMajorContiguous\n{shape}\n{bits}b"))
    }

    fn contiguous_layout_inline(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.contiguous_layout_shape_bits(class)?;
        Some(format!(
            "RightMajorContiguous(shape={shape}, bits={bits})"
        ))
    }

    fn contiguous_layout_shape_bits(&self, class: &ClassId) -> Option<(String, String)> {
        let node_id = self.node_with_op(class, "RightMajorContiguousElementLayoutLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let shape_class = child_class(self.egraph, node, 0)?;
        let bits_class = child_class(self.egraph, node, 1)?;
        Some((
            self.readable_shape(&shape_class)
                .unwrap_or_else(|| self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))),
            self.readable_bit_width(&bits_class),
        ))
    }

    fn left_major_layout_summary(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.left_major_layout_shape_bits(class)?;
        Some(format!("LeftMajorContiguous\n{shape}\n{bits}b"))
    }

    fn left_major_layout_inline(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.left_major_layout_shape_bits(class)?;
        Some(format!("LeftMajorContiguous(shape={shape}, bits={bits})"))
    }

    fn left_major_layout_shape_bits(&self, class: &ClassId) -> Option<(String, String)> {
        let node_id = self.node_with_op(class, "LeftMajorContiguousElementLayoutLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let shape_class = child_class(self.egraph, node, 0)?;
        let bits_class = child_class(self.egraph, node, 1)?;
        Some((
            self.readable_shape(&shape_class)
                .unwrap_or_else(|| self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))),
            self.readable_bit_width(&bits_class),
        ))
    }

    fn strided_layout_summary(&self, class: &ClassId) -> Option<String> {
        let (shape, strides, bits) = self.strided_layout_shape_strides_bits(class)?;
        Some(format!("Strided\n{shape}\n{strides}\n{bits}b"))
    }

    fn strided_layout_inline(&self, class: &ClassId) -> Option<String> {
        let (shape, strides, bits) = self.strided_layout_shape_strides_bits(class)?;
        Some(format!(
            "Strided(shape={shape}, strides={strides}, bits={bits})"
        ))
    }

    fn strided_layout_shape_strides_bits(
        &self,
        class: &ClassId,
    ) -> Option<(String, String, String)> {
        let node_id = self.node_with_op(class, "StridedElementLayoutLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let shape_class = child_class(self.egraph, node, 0)?;
        let strides_class = child_class(self.egraph, node, 1)?;
        let bits_class = child_class(self.egraph, node, 2)?;
        Some((
            self.readable_shape(&shape_class)
                .unwrap_or_else(|| self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))),
            self.readable_expr_list_display(&strides_class)
                .unwrap_or_else(|| self.render_class_prefer(&strides_class, 16, Some("IntExprCons"))),
            self.readable_bit_width(&bits_class),
        ))
    }

    fn readable_shape(&self, class: &ClassId) -> Option<String> {
        let node_id = self.node_with_op(class, "ShapeLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let dims_class = child_class(self.egraph, node, 0)?;
        self.readable_expr_list_display(&dims_class)
    }

    // Widths are wrapped terms (BitWidthLit i64); labels want the bare
    // number, so unwrap one level before rendering.
    fn readable_bit_width(&self, class: &ClassId) -> String {
        self.node_with_op(class, "BitWidthLit")
            .and_then(|node_id| {
                let node = self.egraph.nodes.get(node_id)?;
                let value_class = child_class(self.egraph, node, 0)?;
                Some(self.render_class_prefer(&value_class, 2, None))
            })
            .unwrap_or_else(|| self.render_class_prefer(class, 4, None))
    }

    fn readable_index_map(&self, class: &ClassId) -> Option<String> {
        let node_id = self.node_with_op(class, "IndexMapLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let exprs_class = child_class(self.egraph, node, 0)?;
        self.readable_expr_list_display(&exprs_class)
    }

    fn readable_expr_list_display(&self, class: &ClassId) -> Option<String> {
        let exprs = self.readable_expr_list(class, &mut HashSet::new())?;
        Some(format!("[{}]", exprs.join(", ")))
    }

    fn readable_expr_list(
        &self,
        class: &ClassId,
        visiting: &mut HashSet<ClassId>,
    ) -> Option<Vec<String>> {
        if !visiting.insert(class.clone()) {
            return None;
        }

        let result = if self.node_with_op(class, "IntExprNil").is_some() {
            Some(Vec::new())
        } else {
            let cons_id = self.node_with_op(class, "IntExprCons")?;
            let cons = self.egraph.nodes.get(cons_id)?;
            let head_class = child_class(self.egraph, cons, 0)?;
            let tail_class = child_class(self.egraph, cons, 1)?;
            let mut dims = vec![self.readable_expr(&head_class, &mut HashSet::new())];
            dims.extend(self.readable_expr_list(&tail_class, visiting)?);
            Some(dims)
        };

        visiting.remove(class);
        result
    }

    fn readable_expr(&self, class: &ClassId, visiting: &mut HashSet<ClassId>) -> String {
        if !visiting.insert(class.clone()) {
            return class.to_string();
        }

        let rendered = self
            .node_with_op(class, "IntLit")
            .and_then(|node_id| {
                let node = self.egraph.nodes.get(node_id)?;
                let value_class = child_class(self.egraph, node, 0)?;
                Some(self.render_class_prefer(&value_class, 2, None))
            })
            .or_else(|| {
                // A dynamic-dimension symbol renders as its bare name. After
                // IntLit: a pinned IntVar's class holds both, and the concrete
                // value reads better.
                self.node_with_op(class, "IntVar").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let name_class = child_class(self.egraph, node, 0)?;
                    Some(
                        self.render_class_prefer(&name_class, 2, None)
                            .trim_matches('"')
                            .to_string(),
                    )
                })
            })
            .or_else(|| {
                // Compact per user: v<axis> — the extent rides in tooltips,
                // not labels (a coordinate variable's identity is (axis,
                // extent), but graphs balloon if every leaf spells both).
                self.node_with_op(class, "CoordVar").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let axis_class = child_class(self.egraph, node, 0)?;
                    Some(format!(
                        "v{}",
                        self.render_class_prefer(&axis_class, 2, None)
                    ))
                })
            })
            .or_else(|| {
                self.node_with_op(class, "IntAdd").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let lhs = child_class(self.egraph, node, 0)?;
                    let rhs = child_class(self.egraph, node, 1)?;
                    Some(format!(
                        "({} + {})",
                        self.readable_expr(&lhs, visiting),
                        self.readable_expr(&rhs, visiting)
                    ))
                })
            })
            .or_else(|| {
                self.node_with_op(class, "IntMul").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let lhs = child_class(self.egraph, node, 0)?;
                    let rhs = child_class(self.egraph, node, 1)?;
                    Some(format!(
                        "({} * {})",
                        self.readable_expr(&lhs, visiting),
                        self.readable_expr(&rhs, visiting)
                    ))
                })
            })
            .or_else(|| {
                // The division family and lattice pair render function-style:
                // the rounding mode / lattice direction is the constructor's
                // identity, so it must stay visible.
                ["IntTruncDiv", "IntTruncRem", "IntCeilDiv", "IntMin", "IntMax"]
                    .iter()
                    .zip(["tdiv", "trem", "ceildiv", "min", "max"])
                    .find_map(|(op, name)| {
                        let node_id = self.node_with_op(class, op)?;
                        let node = self.egraph.nodes.get(node_id)?;
                        let dividend = child_class(self.egraph, node, 0)?;
                        let divisor = child_class(self.egraph, node, 1)?;
                        Some(format!(
                            "{name}({}, {})",
                            self.readable_expr(&dividend, visiting),
                            self.readable_expr(&divisor, visiting)
                        ))
                    })
            })
            .unwrap_or_else(|| self.render_class_prefer(class, 8, None));

        visiting.remove(class);
        rendered
    }

    fn class_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let mut details = Vec::new();
        if let Some(typ) = self.class_type(class) {
            details.push(("type".to_string(), typ));
        }
        if let Some(name) = self.class_let_name(class) {
            details.push(("let".to_string(), name));
        }
        details
    }

    fn layout_tensor_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let Some(node_ids) = self.class_nodes.get(class) else {
            return Vec::new();
        };

        for node_id in node_ids {
            let Some(node) = self.egraph.nodes.get(node_id) else {
                continue;
            };
            if node.op != "LayoutTensorLit" {
                continue;
            }

            let Some(logical_class) = child_class(self.egraph, node, 0) else {
                continue;
            };
            let Some(layout_class) = child_class(self.egraph, node, 1) else {
                continue;
            };

            let mut details = self.class_details(class);
            details.push(("logical".to_string(), self.logical_label(&logical_class)));
            details.push(("logical_eclass".to_string(), logical_class.to_string()));
            if let Some(shape) = self.logical_shape(&logical_class) {
                details.push(("shape".to_string(), shape));
            }
            if let Some(dtype) = self.logical_dtype(&logical_class) {
                details.push(("dtype".to_string(), dtype));
            }
            if !details.iter().any(|(key, _)| key == "shape") {
                if let Some(shape) = self.layout_shape(&layout_class) {
                    details.push(("shape".to_string(), shape));
                }
            }
            if !details.iter().any(|(key, _)| key == "dtype") {
                if let Some(dtype) = self.layout_dtype(&layout_class) {
                    details.push(("dtype".to_string(), dtype));
                }
            }
            details.push(("layout".to_string(), self.canonical_layout(&layout_class)));
            details.push(("layout_eclass".to_string(), layout_class.to_string()));
            return details;
        }

        Vec::new()
    }

    fn logical_shape(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LogicalTensorInputLit" {
                continue;
            }
            let shape_class = child_class(self.egraph, node, 1)?;
            return Some(
                self.readable_shape(&shape_class).unwrap_or_else(|| {
                    self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))
                }),
            );
        }
        None
    }

    fn logical_dtype(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LogicalTensorInputLit" {
                continue;
            }
            let dtype_class = child_class(self.egraph, node, 2)?;
            return Some(self.render_class_prefer(&dtype_class, 4, None));
        }
        None
    }

    // ---- numeric geometry (the executor/translator surface): literal shapes
    // walked straight off the e-graph terms, never parsed from rendered
    // strings. `None` = symbolic or absent — consumers bail loudly.

    /// A primitive i64 class: any member node whose op parses as an integer.
    fn numeric_i64(&self, class: &ClassId) -> Option<i64> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if let Ok(value) = node.op.parse::<i64>() {
                return Some(value);
            }
        }
        None
    }

    /// An IntExpr class holding a literal: the IntLit member's value.
    fn numeric_int_expr(&self, class: &ClassId) -> Option<i64> {
        let node_id = self.node_with_op(class, "IntLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let value_class = child_class(self.egraph, node, 0)?;
        self.numeric_i64(&value_class)
    }

    /// A fully-literal IntExprList, walked cons-by-cons BY E-CLASS (a class's
    /// serialized representative may be a function node — the R8 lesson).
    fn numeric_expr_list(&self, class: &ClassId) -> Option<Vec<i64>> {
        let mut dims = Vec::new();
        let mut current = class.clone();
        loop {
            if let Some(node_id) = self.node_with_op(&current, "IntExprCons").cloned() {
                let node = self.egraph.nodes.get(&node_id)?;
                dims.push(self.numeric_int_expr(&child_class(self.egraph, node, 0)?)?);
                current = child_class(self.egraph, node, 1)?;
            } else if self.node_with_op(&current, "IntExprNil").is_some() {
                return Some(dims);
            } else {
                return None;
            }
        }
    }

    /// A Shape class with fully-literal dims.
    fn numeric_shape(&self, class: &ClassId) -> Option<Vec<i64>> {
        let node_id = self.node_with_op(class, "ShapeLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        self.numeric_expr_list(&child_class(self.egraph, node, 0)?)
    }

    /// The numeric `BufferLit` value of a BufferId class, when literal.
    fn numeric_buffer_lit(&self, class: &ClassId) -> Option<i64> {
        let node_id = self.node_with_op(class, "BufferLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        self.numeric_i64(&child_class(self.egraph, node, 0)?)
    }

    /// The layout class's extents, numerically (mirrors [`Self::layout_shape`]).
    fn numeric_layout_dims(&self, class: &ClassId) -> Option<Vec<i64>> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            let shape_child = match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit"
                | "LeftMajorContiguousElementLayoutLit"
                | "StridedElementLayoutLit" => 0,
                "ElementOffsetExpressionLayoutLit" | "BitOffsetExpressionLayoutLit" => 1,
                _ => continue,
            };
            let shape_class = child_class(self.egraph, node, shape_child)?;
            return self.numeric_shape(&shape_class);
        }
        None
    }

    /// The layout class's element bit width, numerically (mirrors
    /// [`Self::layout_dtype`]'s constructor positions).
    fn numeric_layout_bits(&self, class: &ClassId) -> Option<i64> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            let bits_child = match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit" | "LeftMajorContiguousElementLayoutLit" => 1,
                "StridedElementLayoutLit" | "ElementOffsetExpressionLayoutLit"
                | "BitOffsetExpressionLayoutLit" => 2,
                _ => continue,
            };
            let bits_class = child_class(self.egraph, node, bits_child)?;
            let lit = self.node_with_op(&bits_class, "BitWidthLit")?;
            let lit_node = self.egraph.nodes.get(lit)?;
            return self.numeric_i64(&child_class(self.egraph, lit_node, 0)?);
        }
        None
    }

    fn layout_shape(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit"
                | "LeftMajorContiguousElementLayoutLit"
                | "StridedElementLayoutLit" => {
                    let shape_class = child_class(self.egraph, node, 0)?;
                    return Some(self.readable_shape(&shape_class).unwrap_or_else(|| {
                        self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))
                    }));
                }
                _ => {}
            }
        }
        None
    }

    fn layout_dtype(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit" | "LeftMajorContiguousElementLayoutLit" => {
                    let bits_class = child_class(self.egraph, node, 1)?;
                    return Some(self.readable_bit_width(&bits_class));
                }
                "StridedElementLayoutLit" => {
                    let bits_class = child_class(self.egraph, node, 2)?;
                    return Some(self.readable_bit_width(&bits_class));
                }
                _ => {}
            }
        }
        None
    }
}


impl<'a> Extractor<'a> {
    fn plan(&self, class: &ClassId) -> Result<&Plan> {
        self.memo
            .get(class)
            .and_then(Option::as_ref)
            .with_context(|| format!("no extracted plan for eclass {class}"))
    }

    fn build_extracted_graph(&self, roots: &[ClassId]) -> Result<ExtractedGraph> {
        let mut builder = IrBuilder {
            extractor: self,
            dag: DiGraph::new(),
            value_producer: HashMap::new(),
            op_nodes: HashMap::new(),
        };
        let mut outputs = Vec::with_capacity(roots.len());
        for root in roots {
            outputs.push(builder.emit_output(root)?);
        }
        Ok(ExtractedGraph {
            dag: builder.dag,
            outputs,
        })
    }

    // ---- structured info builders for the Layout IR DAG ----

    fn layout_tensor_info(&self, class: &ClassId) -> LayoutTensorInfo {
        let label = self.layout_tensor_label(class);
        let details = self.layout_tensor_details(class);
        let shape = find_detail(&details, "shape");
        let dtype = find_detail(&details, "dtype");
        let mut lines = self.source_lines(Some(class), None);
        push_details(&mut lines, &details);
        let tooltip = join_tooltip(lines);

        let (logical, layout) = match self.layout_tensor_parts(class) {
            Some((logical_class, layout_class)) => (
                self.logical_info(&logical_class, &mut HashSet::new()),
                self.layout_info(&layout_class),
            ),
            None => (
                LogicalInfo {
                    eclass: class.clone(),
                    label: class.to_string(),
                    tooltip: String::new(),
                    op: None,
                    children: Vec::new(),
                },
                LayoutInfo {
                    eclass: class.clone(),
                    label: class.to_string(),
                    tooltip: String::new(),
                },
            ),
        };

        let (dims, element_bits) = match self.layout_tensor_parts(class) {
            Some((_, layout_class)) => {
                let renderer = self.renderer();
                (
                    renderer.numeric_layout_dims(&layout_class),
                    renderer.numeric_layout_bits(&layout_class),
                )
            }
            None => (None, None),
        };

        LayoutTensorInfo {
            eclass: class.clone(),
            label,
            tooltip,
            shape,
            dtype,
            dims,
            element_bits,
            logical,
            layout,
        }
    }

    fn logical_info(&self, class: &ClassId, visiting: &mut HashSet<ClassId>) -> LogicalInfo {
        let label = self.logical_label(class);
        let tooltip = self.logical_tooltip(class);
        let children = if visiting.insert(class.clone()) {
            let children = self
                .logical_children(class)
                .into_iter()
                .map(|(port, child)| (port.to_string(), self.logical_info(&child, visiting)))
                .collect();
            visiting.remove(class);
            children
        } else {
            Vec::new()
        };
        let op = if children.is_empty() {
            None
        } else {
            self.logical_op_name(class)
        };
        LogicalInfo {
            eclass: class.clone(),
            label,
            tooltip,
            op,
            children,
        }
    }

    fn logical_op_name(&self, class: &ClassId) -> Option<String> {
        self.renderer().logical_op_name(class)
    }

    fn layout_info(&self, class: &ClassId) -> LayoutInfo {
        LayoutInfo {
            eclass: class.clone(),
            label: self.layout_label(class),
            tooltip: self.layout_tooltip(class),
        }
    }

    fn buffer_info(
        &self,
        buffer_tensor_class: &ClassId,
        buffer_tensor_enode: Option<&NodeId>,
        tensor_class: &ClassId,
        buffer_id_class: &ClassId,
    ) -> BufferInfo {
        let tensor_label = self
            .class_let_name(buffer_tensor_class)
            .unwrap_or_else(|| buffer_tensor_class.to_string());
        let tensor_tooltip = self.buffer_tensor_tooltip(
            buffer_tensor_class,
            buffer_tensor_enode,
            tensor_class,
            buffer_id_class,
        );
        let rendered = self.render_buffer_id(buffer_id_class);
        let id_label = match self.class_let_name(buffer_id_class) {
            Some(name) if name != rendered => format!("{name}\n{rendered}"),
            Some(name) => name,
            None => rendered,
        };
        let id_tooltip = self.buffer_id_tooltip(buffer_id_class);
        let lit = self.renderer().numeric_buffer_lit(buffer_id_class);
        BufferInfo {
            tensor_eclass: buffer_tensor_class.clone(),
            tensor_label,
            tensor_tooltip,
            id_eclass: buffer_id_class.clone(),
            id_label,
            id_tooltip,
            access: self.buffer_access(buffer_id_class),
            freed_by: self.buffer_freed_by(buffer_id_class),
            lit,
        }
    }

    /// Look up the buffer's contents permission via the `buffer-access-of`
    /// function: its entries serialize as `buffer-access-of` nodes (child 0 =
    /// the BufferId) living in the e-class of their Access value. `None` =
    /// the program declared nothing, which input-program validation rejects
    /// for every buffer — declarations are always explicit.
    fn buffer_access(&self, buffer_id_class: &ClassId) -> Option<Access> {
        for (node_id, node) in &self.egraph.nodes {
            if node.subsumed || node.op != "buffer-access-of" {
                continue;
            }
            let Some(arg_class) = child_class(self.egraph, node, 0) else {
                continue;
            };
            if &arg_class != buffer_id_class {
                continue;
            }
            let access_class = self.egraph.nid_to_cid(node_id);
            if self.renderer().node_with_op(access_class, "ReadOnly").is_some() {
                return Some(Access::ReadOnly);
            }
            return Some(Access::ReadWrite);
        }
        None
    }

    /// Look up storage deallocation responsibility via `buffer-freed-by`.
    /// `None` = undeclared, which input-program validation rejects for every
    /// buffer — there is deliberately no default.
    fn buffer_freed_by(&self, buffer_id_class: &ClassId) -> Option<FreedBy> {
        for (node_id, node) in &self.egraph.nodes {
            if node.subsumed || node.op != "buffer-freed-by" {
                continue;
            }
            let Some(arg_class) = child_class(self.egraph, node, 0) else {
                continue;
            };
            if &arg_class != buffer_id_class {
                continue;
            }
            let freed_class = self.egraph.nid_to_cid(node_id);
            if self.renderer().node_with_op(freed_class, "ProgramFrees").is_some() {
                return Some(FreedBy::Program);
            }
            return Some(FreedBy::Caller);
        }
        None
    }

    // ---- tooltip builders (ported from the former GraphBuilder) ----

    fn source_lines(&self, eclass: Option<&ClassId>, enode: Option<&NodeId>) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(eclass) = eclass {
            push_detail(&mut lines, "eclass", eclass);
            if let Some(typ) = self.class_type(eclass) {
                push_detail(&mut lines, "type", typ);
            }
            if let Some(name) = self.class_let_name(eclass) {
                push_detail(&mut lines, "let", name);
            }
        }
        if let Some(enode) = enode {
            push_detail(&mut lines, "enode", enode);
        }
        lines
    }

    fn logical_tooltip(&self, class: &ClassId) -> String {
        let mut lines = self.source_lines(Some(class), None);
        push_details(&mut lines, &self.logical_details(class));
        join_tooltip(lines)
    }

    fn layout_tooltip(&self, class: &ClassId) -> String {
        let mut lines = self.source_lines(Some(class), None);
        push_details(&mut lines, &self.layout_details(class));
        join_tooltip(lines)
    }

    fn buffer_tensor_tooltip(
        &self,
        class: &ClassId,
        source_enode: Option<&NodeId>,
        tensor: &ClassId,
        buffer_id: &ClassId,
    ) -> String {
        let mut lines = self.source_lines(Some(class), source_enode);
        push_detail(&mut lines, "tensor_eclass", tensor);
        push_detail(&mut lines, "buffer_id_eclass", buffer_id);
        if let Some((literal_tensor, literal_buffer_id)) = self.buffer_tensor_parts(class) {
            push_detail(&mut lines, "literal_tensor_eclass", literal_tensor);
            push_detail(&mut lines, "literal_buffer_id_eclass", literal_buffer_id);
        }
        push_detail(&mut lines, "buffer_id", self.render_buffer_id(buffer_id));
        push_details(&mut lines, &self.layout_tensor_details(tensor));
        join_tooltip(lines)
    }

    fn buffer_id_tooltip(&self, class: &ClassId) -> String {
        let mut lines = self.source_lines(Some(class), None);
        push_detail(&mut lines, "value", self.render_buffer_id(class));
        join_tooltip(lines)
    }

    fn output_tooltip(&self, class: &ClassId, source_enode: Option<&NodeId>) -> String {
        join_tooltip(self.source_lines(Some(class), source_enode))
    }

    fn op_tooltip(&self, class: &ClassId, plan: &Plan) -> String {
        let mut lines = Vec::new();
        push_detail(&mut lines, "selected_output_eclass", class);
        if let Some(op_eclass) = &plan.source_eclass {
            push_detail(&mut lines, "op_eclass", op_eclass);
        }
        if let Some(enode) = &plan.source_enode {
            push_detail(&mut lines, "concrete_enode", enode);
        }
        if let Some(index) = plan.selected_output_index {
            push_detail(&mut lines, "selected_output_index", index);
        }
        push_detail(&mut lines, "cost", plan.cost);
        push_detail(&mut lines, "copies", plan.copies);
        push_details(&mut lines, &self.layout_tensor_details(class));
        if !plan.input_list.is_empty() {
            push_detail(
                &mut lines,
                "input_layout_tensors",
                self.render_layout_tensor_list(&plan.input_list),
            );
        }
        if !plan.output_list.is_empty() {
            push_detail(
                &mut lines,
                "output_layout_tensors",
                self.render_layout_tensor_list(&plan.output_list),
            );
        }
        for meta in &plan.metadata {
            let value = if is_layout_metadata(meta.name) {
                self.canonical_layout(&meta.class)
            } else if meta.name == "shape" {
                self.readable_shape(&meta.class).unwrap_or_else(|| {
                    self.render_class_prefer(
                        &meta.class,
                        metadata_render_depth(meta.name),
                        metadata_preferred_op(meta.name),
                    )
                })
            } else if meta.name == "index_map" {
                self.readable_index_map(&meta.class).unwrap_or_else(|| {
                    self.render_class_prefer(
                        &meta.class,
                        metadata_render_depth(meta.name),
                        metadata_preferred_op(meta.name),
                    )
                })
            } else {
                self.render_class_prefer(
                    &meta.class,
                    metadata_render_depth(meta.name),
                    metadata_preferred_op(meta.name),
                )
            };
            push_detail(&mut lines, meta.name, value);
        }
        join_tooltip(lines)
    }
}

/// Walks the memoized plans into the [`ExtractedGraph`] DAG. Nodes are ops plus
/// input/output boundaries; edges carry the LayoutTensor value flowing between
/// a producer and a consumer.
struct IrBuilder<'e, 'a> {
    extractor: &'e Extractor<'a>,
    dag: ExtractedDag,
    /// Value e-class -> the node that produces it (an op or an input boundary).
    value_producer: HashMap<ClassId, NodeIndex>,
    /// Op identity -> its node, so multi-output ops are emitted exactly once.
    op_nodes: HashMap<(ClassId, NodeId), NodeIndex>,
}

impl<'e, 'a> IrBuilder<'e, 'a> {
    /// Whether an instantiated op's output slot BELONGS to its output class.
    /// Without a genome every slot is claimed (the deterministic extractor's
    /// first-emission behavior). Under a genome, a slot is claimed only if
    /// the genome maps that class to exactly this enode and slot — the
    /// genome, not emission order, decides ownership.
    fn slot_claimed(&self, output: &ClassId, enode: &NodeId, slot: usize) -> bool {
        match self.extractor.genome {
            None => true,
            Some(genome) => genome
                .choices
                .get(output)
                .is_some_and(|choice| &choice.enode == enode && choice.output_index == slot),
        }
    }

    fn ensure_value(&mut self, class: &ClassId) -> Result<NodeIndex> {
        if let Some(index) = self.value_producer.get(class) {
            return Ok(*index);
        }
        let plan = self.extractor.plan(class)?.clone();
        match &plan.kind {
            PlanKind::Input(info) => {
                let value = self.extractor.layout_tensor_info(class);
                let buffer = self.extractor.buffer_info(
                    &info.buffer_tensor_class,
                    Some(&info.buffer_tensor_enode),
                    class,
                    &info.buffer_id_class,
                );
                let index = self
                    .dag
                    .add_node(ExtractedNode::BufferInput(InputNode { value, buffer }));
                self.value_producer.insert(class.clone(), index);
                Ok(index)
            }
            PlanKind::LayoutIr(op) => {
                let op_eclass = plan
                    .source_eclass
                    .clone()
                    .with_context(|| format!("op plan for {class} missing source eclass"))?;
                let source_enode = plan
                    .source_enode
                    .clone()
                    .with_context(|| format!("op plan for {class} missing source enode"))?;
                let key = (op_eclass.clone(), source_enode.clone());
                if let Some(index) = self.op_nodes.get(&key) {
                    self.value_producer.insert(class.clone(), *index);
                    return Ok(*index);
                }

                let outputs = plan
                    .output_list
                    .iter()
                    .enumerate()
                    .map(|(slot, output)| {
                        let mut info = self.extractor.layout_tensor_info(output);
                        if !self.slot_claimed(output, &source_enode, slot) {
                            // WASTE DESTINATION (genome walks only): this
                            // instance computes the slot, but the genome
                            // assigned the class to a different producer. A
                            // fresh synthetic value identity (the poison-id
                            // idiom) makes bufferize allocate scratch instead
                            // of double-writing the class's real home.
                            info.eclass =
                                ClassId::from(format!("genome$waste${source_enode}${slot}"));
                            info.label = format!("{} (unclaimed)", info.label);
                        }
                        info
                    })
                    .collect::<Vec<_>>();
                let inputs = plan
                    .children
                    .iter()
                    .map(|child| OpInput {
                        port: child.port.clone(),
                        value: child.class.clone(),
                    })
                    .collect::<Vec<_>>();
                let tooltip = self.extractor.op_tooltip(class, &plan);
                let node = OpNode {
                    op: op.clone(),
                    provenance: crate::layout_ir::Provenance::Extracted {
                        op_eclass,
                        source_enode: source_enode.clone(),
                        selected_output_index: plan.selected_output_index.unwrap_or(0),
                    },
                    inputs,
                    outputs,
                    tooltip,
                    cost: plan.cost,
                    copies: plan.copies,
                };
                let index = self.dag.add_node(ExtractedNode::LayoutOp(node));
                self.op_nodes.insert(key, index);
                for (slot, output) in plan.output_list.iter().enumerate() {
                    if self.slot_claimed(output, &source_enode, slot) {
                        self.value_producer.insert(output.clone(), index);
                    }
                }
                self.value_producer.insert(class.clone(), index);

                for child in &plan.children {
                    let producer = self.ensure_value(&child.class)?;
                    self.dag.add_edge(
                        producer,
                        index,
                        ExtractedEdge {
                            value: child.class.clone(),
                            port: child.port.clone(),
                        },
                    );
                }
                Ok(index)
            }
            other => bail!("expected value-producing plan at {class}, found {other:?}"),
        }
    }

    fn emit_output(&mut self, class: &ClassId) -> Result<NodeIndex> {
        let plan = self.extractor.plan(class)?.clone();
        match &plan.kind {
            PlanKind::BufferOutputLit => {
                let outputs_list = only_child(&plan, "outputs")?;
                let mut slots = Vec::new();
                self.collect_output_buffers(&outputs_list, 0, &mut slots)?;

                let label = self
                    .extractor
                    .class_let_name(class)
                    .unwrap_or_else(|| class.to_string());
                let tooltip = self
                    .extractor
                    .output_tooltip(class, plan.source_enode.as_ref());
                let output_slots = slots
                    .iter()
                    .map(|(index, value, _, buffer)| OutputSlot {
                        index: *index,
                        value: value.clone(),
                        buffer: buffer.clone(),
                    })
                    .collect();
                let output_index = self.dag.add_node(ExtractedNode::BufferOutput(OutputNode {
                    eclass: class.clone(),
                    label,
                    tooltip,
                    slots: output_slots,
                }));
                for (index, value, producer, _) in &slots {
                    self.dag.add_edge(
                        *producer,
                        output_index,
                        ExtractedEdge {
                            value: value.clone(),
                            port: format!("out {index}"),
                        },
                    );
                }
                Ok(output_index)
            }
            other => bail!("expected BufferOutputLit at extracted root {class}, found {other:?}"),
        }
    }

    fn collect_output_buffers(
        &mut self,
        list_class: &ClassId,
        index: usize,
        slots: &mut Vec<(usize, ClassId, NodeIndex, BufferInfo)>,
    ) -> Result<usize> {
        let plan = self.extractor.plan(list_class)?.clone();
        match &plan.kind {
            PlanKind::BufferTensorCons => {
                let head = only_child(&plan, "head")?;
                let tail = only_child(&plan, "tail")?;
                self.emit_output_buffer(&head, index, slots)?;
                self.collect_output_buffers(&tail, index + 1, slots)
            }
            PlanKind::BufferTensorNil => Ok(index),
            other => bail!("expected BufferTensorList at {list_class}, found {other:?}"),
        }
    }

    fn emit_output_buffer(
        &mut self,
        class: &ClassId,
        index: usize,
        slots: &mut Vec<(usize, ClassId, NodeIndex, BufferInfo)>,
    ) -> Result<()> {
        let plan = self.extractor.plan(class)?.clone();
        match &plan.kind {
            PlanKind::BufferTensorLit {
                buffer_id_class, ..
            } => {
                let tensor = only_child(&plan, "tensor")?;
                let producer = self.ensure_value(&tensor)?;
                let buffer = self.extractor.buffer_info(
                    class,
                    plan.source_enode.as_ref(),
                    &tensor,
                    buffer_id_class,
                );
                slots.push((index, tensor, producer, buffer));
                Ok(())
            }
            other => bail!("expected BufferTensorLit at output {class}, found {other:?}"),
        }
    }
}

fn find_detail(details: &[(String, String)], key: &str) -> Option<String> {
    details
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
}

fn only_child(plan: &Plan, port: &str) -> Result<ClassId> {
    plan.children
        .iter()
        .find(|child| child.port == port)
        .map(|child| child.class.clone())
        .with_context(|| format!("missing {port} child for {:?}", plan.kind))
}

fn push_detail(lines: &mut Vec<String>, key: &str, value: impl ToString) {
    let value = tooltip_value(value);
    if value.is_empty() {
        return;
    }
    let line = format!("{key}={value}");
    if !lines.contains(&line) {
        lines.push(line);
    }
}

fn push_details(lines: &mut Vec<String>, details: &[(String, String)]) {
    for (key, value) in details {
        push_detail(lines, key, value);
    }
}

fn join_tooltip(lines: Vec<String>) -> String {
    lines.join("\n")
}

fn tooltip_value(value: impl ToString) -> String {
    const MAX_FIELD_CHARS: usize = 2_000;

    let mut value = value
        .to_string()
        .replace('"', "'")
        .replace(['\n', '\r', '\t'], " ");
    if value.chars().count() <= MAX_FIELD_CHARS {
        return value;
    }

    value = value.chars().take(MAX_FIELD_CHARS).collect();
    value.push_str("...");
    value
}

fn metadata_preferred_op(name: &str) -> Option<&'static str> {
    match name {
        "layout" | "out_layout" | "add_out_layout" | "mul_out_layout" => {
            Some("RightMajorContiguousElementLayoutLit")
        }
        "shape" => Some("ShapeLit"),
        "index_map" => Some("IndexMapLit"),
        "buffer_id" => Some("BufferLit"),
        _ => None,
    }
}

fn metadata_render_depth(name: &str) -> usize {
    match name {
        "index_map" => 32,
        "layout" | "out_layout" | "add_out_layout" | "mul_out_layout" | "shape" => 16,
        "axis" | "buffer_id" => 4,
        _ => 12,
    }
}

fn is_layout_metadata(name: &str) -> bool {
    matches!(
        name,
        "layout" | "out_layout" | "add_out_layout" | "mul_out_layout"
    )
}

fn plan_label(plan: &Plan) -> String {
    match &plan.kind {
        PlanKind::Input(input) => format!("Input:{}", input.logical_name),
        PlanKind::BufferOutputLit => "BufferOutputLit".to_string(),
        PlanKind::BufferTensorCons => "BufferTensorCons".to_string(),
        PlanKind::BufferTensorNil => "BufferTensorNil".to_string(),
        PlanKind::BufferTensorLit { logical_name, .. } => format!("BufferTensorLit:{logical_name}"),
        PlanKind::LayoutIr(op) => op.label().to_string(),
    }
}

fn class_nodes(egraph: &EGraph) -> HashMap<ClassId, Vec<NodeId>> {
    let mut classes: HashMap<ClassId, Vec<NodeId>> = HashMap::new();
    for (node_id, node) in &egraph.nodes {
        if node.subsumed || node.op == "[...]" {
            continue;
        }
        classes
            .entry(node.eclass.clone())
            .or_default()
            .push(node_id.clone());
    }
    classes
}

fn render_class_nodes(egraph: &EGraph) -> HashMap<ClassId, Vec<NodeId>> {
    let mut classes: HashMap<ClassId, Vec<NodeId>> = HashMap::new();
    for (node_id, node) in &egraph.nodes {
        if node.op == "[...]" {
            continue;
        }
        classes
            .entry(node.eclass.clone())
            .or_default()
            .push(node_id.clone());
    }
    classes
}

fn collect_op_specs(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
) -> (
    HashMap<ClassId, Vec<OpSpec>>,
    HashMap<ClassId, Vec<ProducerRef>>,
) {
    let mut op_specs: HashMap<ClassId, Vec<OpSpec>> = HashMap::new();
    let mut producer_index: HashMap<ClassId, Vec<ProducerRef>> = HashMap::new();

    for (op_class, node_ids) in class_nodes {
        for node_id in node_ids {
            let Some(node) = egraph.nodes.get(node_id) else {
                continue;
            };
            if node.op != "LayoutTensorOpLit" {
                continue;
            }

            let Some(input_list_class) = child_class(egraph, node, 0) else {
                continue;
            };
            let Some(output_list_class) = child_class(egraph, node, 1) else {
                continue;
            };
            let Some(inputs) = layout_tensor_list_items(
                egraph,
                class_nodes,
                &input_list_class,
                &mut HashSet::new(),
            ) else {
                continue;
            };
            let Some(outputs) = layout_tensor_list_items(
                egraph,
                class_nodes,
                &output_list_class,
                &mut HashSet::new(),
            ) else {
                continue;
            };

            let specs = op_specs.entry(op_class.clone()).or_default();
            if specs
                .iter()
                .any(|spec| spec.inputs == inputs && spec.outputs == outputs)
            {
                continue;
            }

            let spec_index = specs.len();
            specs.push(OpSpec {
                inputs,
                outputs: outputs.clone(),
            });

            for (output_index, output_class) in outputs.into_iter().enumerate() {
                producer_index
                    .entry(output_class)
                    .or_default()
                    .push(ProducerRef {
                        op_class: op_class.clone(),
                        spec_index,
                        output_index,
                    });
            }
        }
    }

    (op_specs, producer_index)
}

fn layout_tensor_list_items(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
    list_class: &ClassId,
    visiting: &mut HashSet<ClassId>,
) -> Option<Vec<ClassId>> {
    if !visiting.insert(list_class.clone()) {
        return None;
    }

    let node_ids = class_nodes.get(list_class)?;
    for node_id in node_ids {
        let node = egraph.nodes.get(node_id)?;
        match node.op.as_str() {
            "LayoutTensorNil" => {
                visiting.remove(list_class);
                return Some(Vec::new());
            }
            "LayoutTensorCons" => {
                let head = child_class(egraph, node, 0)?;
                let tail = child_class(egraph, node, 1)?;
                let mut items = layout_tensor_list_items(egraph, class_nodes, &tail, visiting)?;
                items.insert(0, head);
                visiting.remove(list_class);
                return Some(items);
            }
            _ => {}
        }
    }

    visiting.remove(list_class);
    None
}

fn output_root_classes(egraph: &EGraph) -> Vec<ClassId> {
    let mut roots = egraph
        .nodes
        .values()
        .filter(|node| !node.subsumed && node.op == "BufferOutputLit")
        .map(|node| node.eclass.clone())
        .collect::<Vec<_>>();
    roots.sort_by_key(ToString::to_string);
    roots.dedup();
    roots
}

fn collect_output_buffer_classes(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
) -> HashSet<ClassId> {
    let mut output_buffers = HashSet::new();
    let mut visited_lists = HashSet::new();

    for node in egraph
        .nodes
        .values()
        .filter(|node| !node.subsumed && node.op == "BufferOutputLit")
    {
        let Some(list_class) = child_class(egraph, node, 0) else {
            continue;
        };
        collect_buffer_list(
            egraph,
            class_nodes,
            &list_class,
            &mut visited_lists,
            &mut output_buffers,
        );
    }

    output_buffers
}

fn collect_input_buffer_classes(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
) -> HashSet<ClassId> {
    let mut input_buffers = HashSet::new();
    let mut visited_lists = HashSet::new();

    for node in egraph
        .nodes
        .values()
        .filter(|node| !node.subsumed && node.op == "BufferInputLit")
    {
        let Some(list_class) = child_class(egraph, node, 0) else {
            continue;
        };
        collect_buffer_list(
            egraph,
            class_nodes,
            &list_class,
            &mut visited_lists,
            &mut input_buffers,
        );
    }

    input_buffers
}

fn collect_buffer_list(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
    list_class: &ClassId,
    visited_lists: &mut HashSet<ClassId>,
    buffers: &mut HashSet<ClassId>,
) {
    if !visited_lists.insert(list_class.clone()) {
        return;
    }

    let Some(node_ids) = class_nodes.get(list_class) else {
        return;
    };

    for node_id in node_ids {
        let Some(node) = egraph.nodes.get(node_id) else {
            continue;
        };
        if node.op != "BufferTensorCons" {
            continue;
        }
        if let Some(buffer_class) = child_class(egraph, node, 0) {
            buffers.insert(buffer_class);
        }
        if let Some(tail_class) = child_class(egraph, node, 1) {
            collect_buffer_list(egraph, class_nodes, &tail_class, visited_lists, buffers);
        }
    }
}

fn collect_input_terminals(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
    output_buffer_classes: &HashSet<ClassId>,
    input_buffer_classes: &HashSet<ClassId>,
) -> HashMap<ClassId, InputInfo> {
    let mut terminals = HashMap::new();
    let renderer = ClassRenderer {
        egraph,
        class_nodes,
    };
    let has_explicit_inputs = !input_buffer_classes.is_empty();

    for (node_id, node) in egraph
        .nodes
        .iter()
        .filter(|(_, node)| !node.subsumed && node.op == "BufferTensorLit")
    {
        if has_explicit_inputs {
            if !input_buffer_classes.contains(&node.eclass) {
                continue;
            }
        } else if output_buffer_classes.contains(&node.eclass) {
            continue;
        }
        let Some(layout_tensor_class) = child_class(egraph, node, 0) else {
            continue;
        };
        let Some(buffer_id_class) = child_class(egraph, node, 1) else {
            continue;
        };
        terminals
            .entry(layout_tensor_class.clone())
            .or_insert_with(|| InputInfo {
                buffer_tensor_class: node.eclass.clone(),
                buffer_tensor_enode: node_id.clone(),
                buffer_id_class: buffer_id_class.clone(),
                logical_name: renderer
                    .logical_name_from_layout_tensor(&layout_tensor_class)
                    .unwrap_or_else(|| layout_tensor_class.to_string()),
            });
    }

    terminals
}

fn choose_render_node<'a>(
    egraph: &'a EGraph,
    node_ids: &'a [NodeId],
    preferred_op: Option<&str>,
) -> Option<&'a NodeId> {
    if let Some(preferred_op) = preferred_op {
        if let Some(node_id) = node_ids.iter().find(|node_id| {
            egraph
                .nodes
                .get(*node_id)
                .is_some_and(|node| node.op == preferred_op)
        }) {
            return Some(node_id);
        }
    }

    if let Some(node_id) = node_ids.iter().find(|node_id| {
        egraph
            .nodes
            .get(*node_id)
            .is_some_and(|node| node.children.is_empty() && is_simple_literal(&node.op))
    }) {
        return Some(node_id);
    }

    for render_op in RENDER_PREFERRED_OPS {
        if let Some(node_id) = node_ids.iter().find(|node_id| {
            egraph
                .nodes
                .get(*node_id)
                .is_some_and(|node| node.op == *render_op)
        }) {
            return Some(node_id);
        }
    }

    node_ids.iter().min_by_key(|node_id| {
        egraph
            .nodes
            .get(*node_id)
            .map(|node| node.op.as_str())
            .unwrap_or_default()
    })
}

const RENDER_PREFERRED_OPS: &[&str] = &[
    "BufferLit",
    "LogicalIdLit",
    "LayoutTensorLit",
    "LogicalTensorInputLit",
    "LogicalTensorNamed",
    "RightMajorContiguousElementLayoutLit",
    "LeftMajorContiguousElementLayoutLit",
    "StridedElementLayoutLit",
    "IndexMapLit",
    "ShapeLit",
    "IntExprCons",
    "IntExprNil",
    "IntLit",
    "IntVar",
    "CoordVar",
    "F32",
    "F64",
    "Int",
    "Bool",
];

const LAYOUT_RENDER_OPS: &[&str] = &[
    "RightMajorContiguousElementLayoutLit",
    "LeftMajorContiguousElementLayoutLit",
    "StridedElementLayoutLit",
    "ElementOffsetExpressionLayoutLit",
    "BitOffsetExpressionLayoutLit",
];

fn is_simple_literal(op: &str) -> bool {
    op.parse::<i64>().is_ok() || op.starts_with('"')
}

fn child_class(egraph: &EGraph, node: &Node, index: usize) -> Option<ClassId> {
    let child_id = node.children.get(index)?;
    egraph.nodes.get(child_id).map(|child| child.eclass.clone())
}
