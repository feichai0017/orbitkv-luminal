//! ID-FREE DIGESTS — how two plans (or two extracted graphs) are compared
//! when the identities that spell them are assumed random.
//!
//! Ruling (Austin, 2026-09-02): serialized `ClassId`/`NodeId` strings are
//! opaque keys within ONE serialized e-graph. A comparison across two
//! runs therefore may not mention them — not directly, not through a
//! hash, and not through a sort order. [`crate::extractor::plan_fingerprint`]
//! hashes id STRINGS, so it is an equality test inside one process and
//! useless here; these digests exist because that is the only thing it
//! can be.
//!
//! THE NAMING SCHEME (the whole trick): nothing is named by an id.
//!
//!  * OPS are named by their POSITION in a canonical topological order.
//!    The order is Kahn's algorithm over the plan's own dependence edges
//!    (data AND anti); the ready set is broken first by the node's own
//!    id-free SEED (op label, arity, tie table, the `BufferLit`s it
//!    touches) and then by a STRUCTURAL COLOUR — Weisfeiler-Leman
//!    refinement over `(port, colour)` multisets of predecessors and
//!    successors, run to a stable partition. Neither half mentions an id.
//!    Colour refinement is a known INCOMPLETE canonicalization, so
//!    genuinely symmetric nodes can still tie; the digest COUNTS those
//!    ties in its header rather than pretending they do not exist.
//!  * BUFFERS are named `pin{lit}` when they are boundary storage
//!    carrying a `BufferLit` (the CALLER's own numbering — a program
//!    literal no relabeling touches), `in{i}` for a boundary buffer an
//!    input slot pins without one, otherwise `b{n}` in FIRST-USE order
//!    along the canonical op order. Naming a pinned buffer by its literal
//!    keeps an id-dependent INPUT SLOT ORDER visible in exactly one place
//!    (the `inputs` section) instead of renaming the whole plan.
//!  * VALUES are named after the slot that produces them — `v@{buffer}`
//!    for an input slot's value, `v.op{k}.r{j}` for a result — and
//!    `v.x{n}` in first-encounter order for the rest (an operand reading
//!    a folded view has no producing slot).
//!  * OUTPUT SLOTS are named by their index, which the program declares.
//!
//! Text that a runtime may have spelled from an id (a label falling back
//! to `class.to_string()`) goes through [`mask_ids`] first, so a label is
//! evidence about STRUCTURE and never about numbering.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use petgraph::Direction;
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, NodeIndexable};

use crate::bufferize::{BufferId, BufferIrGraph, BufferNode, EdgeKind, Owner, PlanLayout};
use crate::layout_ir::{
    Access, ExtractedGraph, ExtractedNode, FreedBy, LayoutTensorInfo, LogicalInfo,
};
use crate::prelude::egraph_serialize::ClassId;

/// Replace every `-<digits>` id tail with `-N`, so text that fell back to
/// spelling an e-class survives as structure without carrying a number.
/// (`BufferLit(10)` and other program literals are untouched — they are
/// the caller's own numbering, stable across relabelings.)
pub fn mask_ids(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0usize;
    while index < bytes.len() {
        let ch = bytes[index];
        if ch == b'-' {
            let mut end = index + 1;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > index + 1 {
                out.push_str("-N");
                index = end;
                continue;
            }
        }
        out.push(ch as char);
        index += 1;
    }
    out
}

/// A cheap, order-sensitive string hash — used ONLY to keep structural
/// signatures short. It never sees an id.
fn digest64(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The canonical order machinery, shared by both digests: a graph whose
/// nodes carry a `seed` (their own id-free description) and whose edges
/// carry a port label.
struct Canonicalizer {
    /// `signature[node]` — the refined structural colour.
    signature: Vec<u64>,
    /// The canonical topological order over the nodes that were offered.
    order: Vec<NodeIndex>,
    /// How many times the ready set could not be broken by
    /// `(signature, seed)` — an honest ambiguity count, not a silent pick.
    ambiguous_ties: usize,
}

/// Refine every node to a STRUCTURAL SIGNATURE, then put `members` in a
/// canonical topological order.
///
/// The signature is Weisfeiler-Leman colour refinement over the whole
/// graph: round 0 is the node's own id-free description (`seed`), and
/// each round rewrites a node's colour from its own colour plus the
/// multiset of `(port, colour)` of its predecessors AND successors.
/// Refinement stops when the partition stops splitting. It is a
/// well-known INCOMPLETE canonicalization — genuinely symmetric nodes
/// keep one colour — which is exactly why the caller is handed
/// `ambiguous_ties` instead of a promise.
///
/// * `seed(node)` — the node's own id-free description.
/// * `parents(node)` / `children(node)` — `(port, neighbour)` pairs.
/// * `order_parents(node)` — every predecessor the topological order must
///   respect (dataflow AND ordering edges).
fn canonicalize(
    members: &[NodeIndex],
    all_nodes: &[NodeIndex],
    node_count: usize,
    seed: &dyn Fn(NodeIndex) -> String,
    parents: &dyn Fn(NodeIndex) -> Vec<(String, NodeIndex)>,
    children: &dyn Fn(NodeIndex) -> Vec<(String, NodeIndex)>,
    order_parents: &dyn Fn(NodeIndex) -> Vec<NodeIndex>,
) -> Canonicalizer {
    // Every input the refinement reads is computed ONCE: the seeds and the
    // adjacency lists are pure functions of the graph, and recomputing them
    // inside a sort comparator would make this quadratic.
    let mut seed_text: Vec<String> = vec![String::new(); node_count];
    let mut parent_list: Vec<Vec<(String, NodeIndex)>> = vec![Vec::new(); node_count];
    let mut child_list: Vec<Vec<(String, NodeIndex)>> = vec![Vec::new(); node_count];
    for &node in all_nodes {
        seed_text[node.index()] = seed(node);
        parent_list[node.index()] = parents(node);
        child_list[node.index()] = children(node);
    }

    // --- colour refinement -------------------------------------------
    let mut signature = vec![0u64; node_count];
    for &node in all_nodes {
        signature[node.index()] = digest64(&seed_text[node.index()]);
    }
    let distinct = |signature: &[u64]| -> usize {
        all_nodes
            .iter()
            .map(|node| signature[node.index()])
            .collect::<BTreeSet<u64>>()
            .len()
    };
    let mut colours = distinct(&signature);
    for _ in 0..16 {
        let mut next = signature.clone();
        for &node in all_nodes {
            let render = |pairs: &[(String, NodeIndex)]| -> String {
                let mut parts: Vec<String> = pairs
                    .iter()
                    .map(|(port, other)| format!("{port}={:016x}", signature[other.index()]))
                    .collect();
                parts.sort();
                parts.join(",")
            };
            next[node.index()] = digest64(&format!(
                "{:016x}|<{}|>{}",
                signature[node.index()],
                render(&parent_list[node.index()]),
                render(&child_list[node.index()])
            ));
        }
        signature = next;
        let refined = distinct(&signature);
        if refined == colours {
            break; // the partition is stable
        }
        colours = refined;
    }

    // --- Kahn over the ordering edges, ready set broken structurally ---
    let member_set: HashSet<NodeIndex> = members.iter().copied().collect();
    let mut pending: HashMap<NodeIndex, usize> = HashMap::new();
    let mut successors: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for &node in members {
        let blockers: Vec<NodeIndex> = order_parents(node)
            .into_iter()
            .filter(|parent| member_set.contains(parent))
            .collect();
        pending.insert(node, blockers.len());
        for parent in blockers {
            successors.entry(parent).or_default().push(node);
        }
    }
    // The SEED leads and the refined colour only breaks its ties: the seed
    // is the human-meaningful half (op label, arity, boundary literals),
    // so the canonical order groups by op instead of by an opaque hash,
    // and a difference surfaces on the line that caused it.
    let key = |node: NodeIndex| (seed_text[node.index()].clone(), signature[node.index()]);
    let mut ready: Vec<NodeIndex> = members
        .iter()
        .copied()
        .filter(|node| pending[node] == 0)
        .collect();
    let mut order = Vec::with_capacity(members.len());
    let mut ambiguous_ties = 0usize;
    while !ready.is_empty() {
        ready.sort_by_key(|&node| key(node));
        if ready.len() > 1 && key(ready[0]) == key(ready[1]) {
            ambiguous_ties += 1;
        }
        let node = ready.remove(0);
        order.push(node);
        for child in successors.get(&node).cloned().unwrap_or_default() {
            let slot = pending.get_mut(&child).expect("member");
            *slot -= 1;
            if *slot == 0 {
                ready.push(child);
            }
        }
    }
    // A cycle would leave members unplaced; the plan DAG has none, but
    // never silently drop nodes from a digest.
    if order.len() < members.len() {
        let mut leftover: Vec<NodeIndex> = members
            .iter()
            .copied()
            .filter(|node| !order.contains(node))
            .collect();
        leftover.sort_by_key(|&node| key(node));
        ambiguous_ties += leftover.len();
        order.extend(leftover);
    }

    Canonicalizer {
        signature,
        order,
        ambiguous_ties,
    }
}

/// A first-come name allocator (buffers, values) — the "first use in the
/// canonical order" rule, made explicit.
struct Namer<K: std::hash::Hash + Eq + Clone> {
    names: HashMap<K, String>,
    prefix: &'static str,
    next: usize,
}

impl<K: std::hash::Hash + Eq + Clone> Namer<K> {
    fn new(prefix: &'static str) -> Self {
        Self {
            names: HashMap::new(),
            prefix,
            next: 0,
        }
    }

    fn pin(&mut self, key: &K, name: String) {
        self.names.entry(key.clone()).or_insert(name);
    }

    fn name(&mut self, key: &K) -> String {
        if let Some(name) = self.names.get(key) {
            return name.clone();
        }
        let name = format!("{}{}", self.prefix, self.next);
        self.next += 1;
        self.names.insert(key.clone(), name.clone());
        name
    }

    fn known(&self, key: &K) -> Option<&String> {
        self.names.get(key)
    }
}

// ===========================================================================
// The plan digest
// ===========================================================================

fn owner_text(owner: Owner) -> &'static str {
    match owner {
        Owner::Caller => "caller",
        Owner::System => "system",
    }
}

fn access_text(access: Access) -> &'static str {
    match access {
        Access::ReadOnly => "ro",
        Access::ReadWrite => "rw",
    }
}

fn freed_text(freed: FreedBy) -> &'static str {
    match freed {
        FreedBy::Caller => "caller",
        FreedBy::Program => "program",
    }
}

/// An ID-FREE rendering of a bufferized plan.
///
/// Two plans that differ only by an e-graph relabeling digest
/// identically; two plans that elect different ops, wire them
/// differently, or share buffers differently do not. Carried layouts
/// (`L`) are NOT read — core never interprets them and neither does this.
pub fn plan_digest<L: PlanLayout>(plan: &BufferIrGraph<L>) -> String {
    let dag = &plan.dag;
    let node_count = dag.node_bound();

    // The executable members: compute ops and bufferizer copies.
    let members: Vec<NodeIndex> = dag
        .node_indices()
        .filter(|&index| {
            matches!(
                dag[index],
                BufferNode::Compute { .. } | BufferNode::BufferCopy { .. }
            )
        })
        .collect();

    // A boundary buffer's `BufferLit` is the caller's own numbering, so it
    // anchors an op to the program's boundary without naming an identity.
    let boundary_lits = |ids: &[BufferId]| -> Vec<i64> {
        let mut lits: Vec<i64> = ids
            .iter()
            .filter_map(|id| plan.buffers.get(id).and_then(|buffer| buffer.lit))
            .collect();
        lits.sort_unstable();
        lits
    };
    let seed = |index: NodeIndex| -> String {
        match &dag[index] {
            BufferNode::Compute {
                op,
                reads,
                writes,
                ties,
                ..
            } => format!(
                "compute:{}:r{}:w{}:t{ties:?}:lits{:?}/{:?}",
                op.label(),
                reads.len(),
                writes.len(),
                boundary_lits(reads),
                boundary_lits(writes),
            ),
            BufferNode::BufferCopy { src, dst } => format!(
                "copy:lits{:?}/{:?}",
                boundary_lits(std::slice::from_ref(src)),
                boundary_lits(std::slice::from_ref(dst)),
            ),
            BufferNode::BufferInput { slots } => format!(
                "input:{}:lits{:?}",
                slots.len(),
                boundary_lits(
                    &slots
                        .iter()
                        .map(|binding| binding.buffer.clone())
                        .collect::<Vec<_>>()
                ),
            ),
            BufferNode::BufferOutput { slots } => format!(
                "output:{}:lits{:?}",
                slots.len(),
                boundary_lits(
                    &slots
                        .iter()
                        .map(|slot| slot.buffer.clone())
                        .collect::<Vec<_>>()
                ),
            ),
        }
    };
    // Refinement walks DATA edges both ways; an anti edge is an ordering
    // obligation, not a dataflow fact, so it constrains the ORDER only.
    let data_parents = |index: NodeIndex| -> Vec<(String, NodeIndex)> {
        dag.edges_directed(index, Direction::Incoming)
            .filter(|edge| edge.weight().kind == EdgeKind::Data)
            .map(|edge| (edge.weight().port.clone(), edge.source()))
            .collect()
    };
    let data_children = |index: NodeIndex| -> Vec<(String, NodeIndex)> {
        dag.edges_directed(index, Direction::Outgoing)
            .filter(|edge| edge.weight().kind == EdgeKind::Data)
            .map(|edge| (edge.weight().port.clone(), edge.target()))
            .collect()
    };
    let order_parents = |index: NodeIndex| -> Vec<NodeIndex> {
        dag.edges_directed(index, Direction::Incoming)
            .map(|edge| edge.source())
            .collect()
    };
    let all_nodes: Vec<NodeIndex> = dag.node_indices().collect();
    let canonical = canonicalize(
        &members,
        &all_nodes,
        node_count,
        &seed,
        &data_parents,
        &data_children,
        &order_parents,
    );
    let op_position: HashMap<NodeIndex, usize> = canonical
        .order
        .iter()
        .enumerate()
        .map(|(slot, &index)| (index, slot))
        .collect();

    // --- names -------------------------------------------------------
    let mut buffers: Namer<BufferId> = Namer::new("b");
    let mut values: Namer<ClassId> = Namer::new("v.x");
    // A boundary buffer's `BufferLit` is the caller's OWN numbering — a
    // program literal, untouched by any relabeling — so it is the one
    // name the plan genuinely knows for a pinned buffer. Using it keeps
    // an id-order-dependent INPUT SLOT ORDER visible in exactly one place
    // (the `inputs` section) instead of renaming the whole plan.
    for (id, buffer) in &plan.buffers {
        if let (BufferId::Boundary(_), Some(lit)) = (id, buffer.lit) {
            buffers.pin(id, format!("pin{lit}"));
        }
    }
    let mut input_slots: Vec<(usize, ClassId, BufferId)> = Vec::new();
    for index in dag.node_indices() {
        if let BufferNode::BufferInput { slots } = &dag[index] {
            for (slot, binding) in slots.iter().enumerate() {
                buffers.pin(&binding.buffer, format!("in{slot}"));
                let name = buffers.name(&binding.buffer);
                values.pin(&binding.value, format!("v@{name}"));
                input_slots.push((slot, binding.value.clone(), binding.buffer.clone()));
            }
        }
    }
    for (slot, &index) in canonical.order.iter().enumerate() {
        if let BufferNode::Compute { result_info, .. } = &dag[index] {
            for (result, info) in result_info.iter().enumerate() {
                values.pin(&info.value, format!("v.op{slot}.r{result}"));
            }
        }
    }
    // First-use naming, walked in canonical order.
    for &index in &canonical.order {
        match &dag[index] {
            BufferNode::Compute {
                reads,
                writes,
                operand_info,
                result_info,
                ..
            } => {
                for info in operand_info {
                    let _ = values.name(&info.value);
                }
                for buffer in reads.iter().chain(writes) {
                    let _ = buffers.name(buffer);
                }
                for info in result_info {
                    let _ = values.name(&info.value);
                }
            }
            BufferNode::BufferCopy { src, dst } => {
                let _ = buffers.name(src);
                let _ = buffers.name(dst);
            }
            _ => {}
        }
    }
    let mut output_slots: Vec<(usize, ClassId, BufferId)> = Vec::new();
    for index in dag.node_indices() {
        if let BufferNode::BufferOutput { slots } = &dag[index] {
            let mut slots: Vec<_> = slots.iter().collect();
            slots.sort_by_key(|slot| slot.index);
            for slot in slots {
                let _ = buffers.name(&slot.buffer);
                let _ = values.name(&slot.value);
                output_slots.push((slot.index, slot.value.clone(), slot.buffer.clone()));
            }
        }
    }
    // Buffers no op and no boundary ever touched: named last, in an
    // id-free deterministic order (their own declared properties, then
    // the planner's own allocation counter, which is not an e-graph id).
    let mut orphans: Vec<&BufferId> = plan
        .buffers
        .keys()
        .filter(|id| buffers.known(id).is_none())
        .collect();
    orphans.sort_by_key(|id| {
        let buffer = &plan.buffers[*id];
        (
            mask_ids(&buffer.label),
            buffer.lit,
            owner_text(buffer.owner),
            access_text(buffer.access),
            match id {
                BufferId::Allocated(minted) => (1u8, *minted),
                BufferId::Boundary(_) => (0u8, 0),
            },
        )
    });
    for id in orphans {
        let _ = buffers.name(id);
    }

    // CLOSING SWEEPS. Every value still unnamed is named after the buffer
    // it lives in, walked in BUFFER-NAME order. This matters: the two
    // remaining sources of values are `Buffer::backs` (a `HashMap`, so no
    // order at all) and `value_buffer` (a `BTreeMap` keyed on `ClassId` —
    // which is an ID SORT, exactly what this module may not read). Values
    // that cohabit one buffer share the name, which is the honest answer:
    // the plan is what assigns them, and nothing id-free tells them apart.
    let mut by_name: Vec<(String, BufferId)> = plan
        .buffers
        .keys()
        .map(|id| (buffers.name(id), id.clone()))
        .collect();
    by_name.sort_by(|left, right| left.0.cmp(&right.0));
    for (name, id) in &by_name {
        if let Some(buffer) = plan.buffers.get(id) {
            values.pin(&buffer.backs, format!("v@{name}"));
        }
    }
    let mut residents: Vec<(String, ClassId)> = plan
        .value_buffer
        .iter()
        .map(|(value, buffer)| {
            (
                buffers.known(buffer).cloned().unwrap_or_default(),
                value.clone(),
            )
        })
        .collect();
    residents.sort_by(|left, right| left.0.cmp(&right.0));
    for (name, value) in &residents {
        values.pin(value, format!("v@{name}"));
    }

    // --- render ------------------------------------------------------
    let mut out = String::new();
    let _ = writeln!(out, "plan-digest v1");
    let _ = writeln!(
        out,
        "counts: ops={} buffers={} inputs={} outputs={} values={} ambiguous_ties={}",
        canonical.order.len(),
        plan.buffers.len(),
        input_slots.len(),
        output_slots.len(),
        plan.value_buffer.len(),
        canonical.ambiguous_ties
    );

    input_slots.sort_by_key(|(slot, _, _)| *slot);
    let _ = writeln!(out, "inputs ({}):", input_slots.len());
    for (slot, value, buffer) in &input_slots {
        let _ = writeln!(
            out,
            "  in{slot} value={} buffer={}",
            values.name(value),
            buffers.name(buffer)
        );
    }

    let mut buffer_rows: Vec<(String, String)> = Vec::new();
    for (id, buffer) in &plan.buffers {
        let name = buffers.name(id);
        let backs = values.name(&buffer.backs);
        let kind = match id {
            BufferId::Boundary(_) => "boundary",
            BufferId::Allocated(_) => "allocated",
        };
        buffer_rows.push((
            name.clone(),
            format!(
                "  {name} {kind} owner={} access={} freed_by={} lit={:?} backs={backs} label={:?}",
                owner_text(buffer.owner),
                access_text(buffer.access),
                freed_text(buffer.freed_by),
                buffer.lit,
                mask_ids(&buffer.label),
            ),
        ));
    }
    buffer_rows.sort();
    let _ = writeln!(out, "buffers ({}):", buffer_rows.len());
    for (_, row) in buffer_rows {
        let _ = writeln!(out, "{row}");
    }

    let _ = writeln!(out, "ops ({}):", canonical.order.len());
    for (slot, &index) in canonical.order.iter().enumerate() {
        match &dag[index] {
            BufferNode::Compute {
                op,
                reads,
                writes,
                ties,
                operand_info,
                result_info,
            } => {
                let operands: Vec<String> = reads
                    .iter()
                    .enumerate()
                    .map(|(operand, buffer)| {
                        let value = operand_info
                            .get(operand)
                            .map(|info| values.name(&info.value))
                            .unwrap_or_else(|| "?".to_string());
                        format!("{}:{value}", buffers.name(buffer))
                    })
                    .collect();
                let results: Vec<String> = writes
                    .iter()
                    .enumerate()
                    .map(|(result, buffer)| {
                        let value = result_info
                            .get(result)
                            .map(|info| values.name(&info.value))
                            .unwrap_or_else(|| "?".to_string());
                        format!("{}:{value}", buffers.name(buffer))
                    })
                    .collect();
                let _ = writeln!(
                    out,
                    "  op{slot} {} ties={ties:?} reads=[{}] writes=[{}]",
                    op.label(),
                    operands.join(", "),
                    results.join(", ")
                );
            }
            BufferNode::BufferCopy { src, dst } => {
                let _ = writeln!(
                    out,
                    "  op{slot} BufferCopy src={} dst={}",
                    buffers.name(src),
                    buffers.name(dst)
                );
            }
            _ => unreachable!("only compute and copy nodes are ordered"),
        }
    }

    let mut anti: Vec<String> = Vec::new();
    for edge in dag.edge_references() {
        if edge.weight().kind != EdgeKind::Anti {
            continue;
        }
        let name = |index: NodeIndex| match op_position.get(&index) {
            Some(slot) => format!("op{slot}"),
            None => match &dag[index] {
                BufferNode::BufferInput { .. } => "input".to_string(),
                BufferNode::BufferOutput { .. } => "output".to_string(),
                _ => "?".to_string(),
            },
        };
        anti.push(format!(
            "  {} -> {} buffer={}",
            name(edge.source()),
            name(edge.target()),
            buffers.name(&edge.weight().buffer)
        ));
    }
    anti.sort();
    let _ = writeln!(out, "anti ({}):", anti.len());
    for line in anti {
        let _ = writeln!(out, "{line}");
    }

    output_slots.sort_by_key(|(slot, _, _)| *slot);
    let _ = writeln!(out, "outputs ({}):", output_slots.len());
    for (slot, value, buffer) in &output_slots {
        let _ = writeln!(
            out,
            "  out{slot} value={} buffer={}",
            values.name(value),
            buffers.name(buffer)
        );
    }

    // The value→buffer assignment, spelled through the same names: this
    // is where buffer SHARING (in-place chains, cohabitation) shows up.
    let mut assignment: Vec<String> = plan
        .value_buffer
        .iter()
        .map(|(value, buffer)| format!("  {} -> {}", values.name(value), buffers.name(buffer)))
        .collect();
    assignment.sort();
    let _ = writeln!(out, "assignment ({}):", assignment.len());
    for line in assignment {
        let _ = writeln!(out, "{line}");
    }

    out
}

// ===========================================================================
// The extracted-graph digest
// ===========================================================================

/// The logical spine of a value, to `depth` levels — constructor names
/// only (eager fields; the lazy label/tooltip renderers are never forced).
fn logical_spine(info: &LogicalInfo, depth: usize) -> String {
    let head = info.op.clone().unwrap_or_else(|| "leaf".to_string());
    if depth == 0 || info.children.is_empty() {
        return head;
    }
    let children: Vec<String> = info
        .children
        .iter()
        .map(|(port, child)| format!("{port}={}", logical_spine(child, depth - 1)))
        .collect();
    format!("{head}({})", children.join(","))
}

/// The id-free facts of a LayoutTensor: the numeric/plan facts that a
/// relabeling must not touch, plus masked display text.
fn tensor_facts(info: &LayoutTensorInfo) -> String {
    format!(
        "label={:?} shape={:?} dtype={:?} dtype_enum={:?} dims={:?} bits={:?} logical={} \
         layout_class_sort={:?}",
        mask_ids(&info.label),
        info.shape.as_deref().map(mask_ids),
        info.dtype.as_deref().map(mask_ids),
        info.dtype_enum,
        info.dims,
        info.element_bits,
        logical_spine(&info.logical, 3),
        info.layout
            .eclass
            .as_ref()
            .rsplit_once('-')
            .map(|(sort, _)| sort.to_string()),
    )
}

/// An ID-FREE rendering of a deterministic [`ExtractedGraph`].
///
/// Node identity is position in a canonical topological order; values are
/// named after the producing slot; boundary rows keep the program's own
/// numbering (slot index, `BufferLit`) and their declared access /
/// deallocation contracts.
pub fn extracted_digest(graph: &ExtractedGraph) -> String {
    let dag = &graph.dag;
    let node_count = dag.node_bound();

    let members: Vec<NodeIndex> = dag
        .node_indices()
        .filter(|&index| !matches!(dag[index], ExtractedNode::BufferOutput(_)))
        .collect();

    let seed = |index: NodeIndex| -> String {
        match &dag[index] {
            ExtractedNode::BufferInput(input) => format!(
                "input:lit={:?}:access={:?}:freed={:?}:{}",
                input.buffer.lit,
                input.buffer.access,
                input.buffer.freed_by,
                tensor_facts(&input.value)
            ),
            ExtractedNode::LayoutOp(op) => format!(
                "op:{}:in{}:out{}:cost{}",
                op.op.label(),
                op.inputs.len(),
                op.outputs.len(),
                op.heuristic_cost
            ),
            ExtractedNode::BufferOutput(output) => format!("output:{}", output.slots.len()),
        }
    };
    let parents = |index: NodeIndex| -> Vec<(String, NodeIndex)> {
        dag.edges_directed(index, Direction::Incoming)
            .map(|edge| (edge.weight().port.clone(), edge.source()))
            .collect()
    };
    let children = |index: NodeIndex| -> Vec<(String, NodeIndex)> {
        dag.edges_directed(index, Direction::Outgoing)
            .map(|edge| (edge.weight().port.clone(), edge.target()))
            .collect()
    };
    let order_parents = |index: NodeIndex| -> Vec<NodeIndex> {
        parents(index).into_iter().map(|(_, p)| p).collect()
    };
    let all_nodes: Vec<NodeIndex> = dag.node_indices().collect();
    let canonical = canonicalize(
        &members,
        &all_nodes,
        node_count,
        &seed,
        &parents,
        &children,
        &order_parents,
    );

    // Names: `i{k}` for boundary inputs, `n{k}` for ops, both by canonical
    // position; values named after the slot that produces them.
    let mut node_name: HashMap<NodeIndex, String> = HashMap::new();
    let mut values: Namer<ClassId> = Namer::new("v.x");
    let mut inputs = 0usize;
    let mut ops = 0usize;
    for &index in &canonical.order {
        match &dag[index] {
            ExtractedNode::BufferInput(input) => {
                node_name.insert(index, format!("i{inputs}"));
                values.pin(&input.value.eclass, format!("v.i{inputs}"));
                inputs += 1;
            }
            ExtractedNode::LayoutOp(op) => {
                node_name.insert(index, format!("n{ops}"));
                for (slot, output) in op.outputs.iter().enumerate() {
                    values.pin(&output.eclass, format!("v.n{ops}.r{slot}"));
                }
                ops += 1;
            }
            ExtractedNode::BufferOutput(_) => {}
        }
    }
    // Operand values, named in CANONICAL order: everything below reads
    // names rather than minting them, so no rendering loop's own
    // iteration order (petgraph's edge indices, a node-index scan) can
    // decide a name.
    for &index in &canonical.order {
        if let ExtractedNode::LayoutOp(op) = &dag[index] {
            for input in &op.inputs {
                let _ = values.name(&input.value);
            }
        }
    }

    let mut out = String::new();
    let _ = writeln!(out, "extracted-digest v1");
    let _ = writeln!(
        out,
        "counts: nodes={} edges={} inputs={inputs} ops={ops} outputs={} ambiguous_ties={}",
        dag.node_count(),
        dag.edge_count(),
        graph.outputs.len(),
        canonical.ambiguous_ties
    );

    for &index in &canonical.order {
        let name = &node_name[&index];
        match &dag[index] {
            ExtractedNode::BufferInput(input) => {
                let _ = writeln!(
                    out,
                    "{name} INPUT lit={:?} access={:?} freed_by={:?} \
                     tensor_label={:?} id_label={:?} value={} {}",
                    input.buffer.lit,
                    input.buffer.access,
                    input.buffer.freed_by,
                    mask_ids(&input.buffer.tensor_label),
                    mask_ids(&input.buffer.id_label),
                    values.name(&input.value.eclass),
                    tensor_facts(&input.value),
                );
            }
            ExtractedNode::LayoutOp(op) => {
                let provenance = match &op.provenance {
                    crate::layout_ir::Provenance::Extracted {
                        selected_output_index,
                        ..
                    } => format!("extracted:selected_output={selected_output_index}"),
                    crate::layout_ir::Provenance::Synthesized { .. } => "synthesized".to_string(),
                };
                let _ = writeln!(
                    out,
                    "{name} OP {} cost={} provenance={provenance}",
                    op.op.label(),
                    op.heuristic_cost
                );
                for (slot, input) in op.inputs.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "  operand[{slot}] port={:?} value={}",
                        input.port,
                        values.name(&input.value)
                    );
                }
                for (slot, output) in op.outputs.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "  result[{slot}] value={} {}",
                        values.name(&output.eclass),
                        tensor_facts(output)
                    );
                }
            }
            ExtractedNode::BufferOutput(_) => {}
        }
    }

    // Every dataflow edge, spelled through canonical names.
    let mut edges: Vec<String> = Vec::new();
    for edge in dag.edge_references() {
        let source = node_name
            .get(&edge.source())
            .cloned()
            .unwrap_or_else(|| "OUT".to_string());
        let target = node_name
            .get(&edge.target())
            .cloned()
            .unwrap_or_else(|| "OUT".to_string());
        edges.push(format!(
            "  {source} -> {target} port={:?} value={}",
            edge.weight().port,
            values.name(&edge.weight().value)
        ));
    }
    edges.sort();
    let _ = writeln!(out, "edges ({}):", edges.len());
    for line in edges {
        let _ = writeln!(out, "{line}");
    }

    // Output boundaries: the program's own slot numbering.
    let mut slots: Vec<(usize, String)> = Vec::new();
    let mut boundary_labels: BTreeSet<String> = BTreeSet::new();
    for index in dag.node_indices() {
        if let ExtractedNode::BufferOutput(output) = &dag[index] {
            boundary_labels.insert(mask_ids(&output.label));
            for slot in &output.slots {
                slots.push((
                    slot.index,
                    format!(
                        "  out{} value={} lit={:?} access={:?} freed_by={:?} id_label={:?}",
                        slot.index,
                        values.name(&slot.value),
                        slot.buffer.lit,
                        slot.buffer.access,
                        slot.buffer.freed_by,
                        mask_ids(&slot.buffer.id_label),
                    ),
                ));
            }
        }
    }
    slots.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let _ = writeln!(out, "outputs ({}):", slots.len());
    for (_, line) in slots {
        let _ = writeln!(out, "{line}");
    }
    let _ = writeln!(
        out,
        "output_boundary_labels: {:?}",
        boundary_labels.into_iter().collect::<Vec<_>>()
    );

    // A signature census: how many distinct structural signatures the
    // graph carries (a blunt check that the signature actually separates).
    let census: BTreeMap<u64, usize> =
        canonical
            .order
            .iter()
            .fold(BTreeMap::new(), |mut census, &index| {
                *census
                    .entry(canonical.signature[index.index()])
                    .or_insert(0) += 1;
                census
            });
    let _ = writeln!(
        out,
        "distinct_signatures: {} over {} nodes",
        census.len(),
        canonical.order.len()
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockOp, TestGraph, bufferize_mock};

    fn chain() -> ExtractedGraph {
        let mut graph = TestGraph::new();
        let x = graph.input("x", "bx", Access::ReadOnly, "rm");
        let y = graph.input("y", "by", Access::ReadOnly, "rm");
        let add = MockOp {
            reads: vec![true, true],
            ..Default::default()
        };
        let sum = graph.op(Box::new(add.clone()), &[&x, &y], &[("sum", "rm")]);
        let out = graph.op(Box::new(add), &[&sum[0], &y], &[("out", "rm")]);
        graph.output(&out[0], "bout");
        graph.build()
    }

    #[test]
    fn the_digests_are_stable_and_mention_no_identity() {
        let graph = chain();
        let plan = bufferize_mock(&graph).expect("the chain bufferizes");
        let extracted = extracted_digest(&graph);
        let planned = plan_digest(&plan);
        assert_eq!(extracted, extracted_digest(&chain()));
        assert_eq!(
            planned,
            plan_digest(&bufferize_mock(&chain()).expect("bufferizes"))
        );
        // TestGraph spells its identities `val$x`, `buf$bx`, ... — the
        // digests must not carry any of them.
        for spelling in ["val$", "buf$", "layout$", "buftensor$", "logical$"] {
            assert!(
                !extracted.contains(spelling),
                "extracted digest leaked an identity ({spelling}):\n{extracted}"
            );
            assert!(
                !planned.contains(spelling),
                "plan digest leaked an identity ({spelling}):\n{planned}"
            );
        }
    }

    /// Two plans over structurally DIFFERENT graphs must digest apart —
    /// the digest is not allowed to be blind.
    #[test]
    fn a_different_graph_digests_differently() {
        let mut graph = TestGraph::new();
        let x = graph.input("x", "bx", Access::ReadOnly, "rm");
        let y = graph.input("y", "by", Access::ReadOnly, "rm");
        let add = MockOp {
            reads: vec![true, true],
            ..Default::default()
        };
        // One op instead of two: a different election.
        let sum = graph.op(Box::new(add), &[&x, &y], &[("sum", "rm")]);
        graph.output(&sum[0], "bout");
        let shorter = graph.build();
        assert_ne!(extracted_digest(&shorter), extracted_digest(&chain()));
        assert_ne!(
            plan_digest(&bufferize_mock(&shorter).expect("bufferizes")),
            plan_digest(&bufferize_mock(&chain()).expect("bufferizes"))
        );
    }

    #[test]
    fn mask_ids_hides_numbering_but_keeps_structure() {
        assert_eq!(mask_ids("Layout-2216"), "Layout-N");
        assert_eq!(
            mask_ids("function-55-LayoutTensorOpAdd"),
            "function-N-LayoutTensorOpAdd"
        );
        assert_eq!(mask_ids("BufferLit(10)"), "BufferLit(10)");
        assert_eq!(mask_ids("a-b"), "a-b");
    }
}
