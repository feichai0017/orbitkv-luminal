//! ID-FREE E-CLASS LOCATION — describe the class you want, then query the
//! e-nodes inside it.
//!
//! RULING (Austin, 2026-09-02): *"You need to use the egraph via its APIs.
//! You need to search for the eclass you're looking for by describing what
//! it should be and once you find it, you can query the enodes inside it."*
//!
//! THE CONTRACT this module exists to enforce:
//!
//!  * A serialized [`ClassId`] (`"Layout-2216"`) is a CREATION-ORDER
//!    COUNTER. It is mostly stable for one binary, guaranteed by nothing,
//!    and renumbers on any edit to the estate. Nothing a test asserts may
//!    depend on it, and nothing this module returns to a caller may spell
//!    one — not in a signature, not in an ordering, not in a tie-break.
//!    Ids cross this API only as OPAQUE KEYS inside one process: you get a
//!    [`ClassId`]/[`NodeId`] back from a search and hand it straight to
//!    another call.
//!  * A test must not pin a COST/LABEL TIE by accident either. Where two
//!    e-nodes are the same constructor at the same cost, "the extractor
//!    picks this one" is an id-order fact, not a semantic one. The
//!    election helpers below (`elect_each`, `elect_by_signature`) exist so
//!    a test can force EACH candidate in turn and assert what that
//!    candidate's OWN terms say — the per-e-node pattern.
//!
//! WHAT REPLACES THE ID. Every observable string this module produces is a
//! SIGNATURE: the e-node's constructor plus its children rendered
//! recursively, with leaves spelled by recorder let-name, input name, or
//! literal value. See [`Locator::signature`].
//!
//! This generalizes the ad-hoc walkers that grew across the marker board
//! (`d_layout_class` / `class_ops` / `child_class` / `node_in_class` in the
//! cuBLASLt bias-premise test, `buffer_list_classes` in the cuBLASLt
//! election core, `output_stem`'s `natout\d+` text scrape in the
//! null-tensor probe, `genome_electing` / `elect_all` /
//! `per_enode_election_sweep` on the round-3 attack board) into one place.
//!
//! ```ignore
//! use luminal::test_support::locate::Locator;
//!
//! let loc = Locator::new(&egraph);
//! // Describe the class, don't name it.
//! let site = loc.find_one_class(|c| c.has_op("CublasLtLogicalMatmulSite"));
//! // Then query the e-nodes inside it.
//! for cand in loc.candidates(&index, &out_class) {
//!     let genome = loc.elect_by_signature(&index, &base, &out_class, &cand.signature);
//!     // ... assert what THIS candidate's own descriptors say.
//! }
//! ```

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use egraph_serialize::{ClassId, EGraph, Node, NodeId};

use crate::extractor::{Genome, ProducerChoice};

/// The genome sampling index shape produced by
/// [`crate::extractor::producer_index_with_matchers`]: class → its
/// candidate producers as `(implementation constructor name, choice)`.
pub type ProducerIndex = BTreeMap<ClassId, Vec<(String, ProducerChoice)>>;

/// DEPTH BOUND for [`Locator::signature`] (see the type docs for why a
/// bound is required at all). Four class levels below the e-node is enough
/// to separate the spellings the marker board distinguishes — a cuBLASLt
/// op reaches its operand descriptors' operation constructors and layout
/// tensors, and its site's logical operands, within it.
pub const DEFAULT_SIGNATURE_DEPTH: usize = 4;

/// How many alternative spellings of one class a signature prints before
/// it elides. Alternatives are SORTED before truncation, so the elision is
/// content-determined, never node-order determined.
const MAX_ALTERNATIVES: usize = 4;

/// One producer candidate of a class, described rather than identified.
///
/// `signature` is the id-free canonical rendering of `enode`
/// ([`Locator::signature`]); `enode` is an opaque handle, valid only
/// within the process that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The implementation constructor name (`LayoutTensorOpCublasLtBias`, …).
    pub constructor: String,
    /// Which of the producing instance's output slots carries the class.
    pub output_index: usize,
    /// The id-free canonical rendering of `enode`.
    pub signature: String,
    /// Opaque handle to the producing e-node. NEVER assert on its text.
    pub enode: NodeId,
}

impl Candidate {
    /// One display line — constructor, output slot, FULL signature. Used
    /// by the panics below so a failure shows the whole field rather than
    /// an id.
    pub fn describe(&self) -> String {
        format!(
            "{} [out {}] {}",
            self.constructor, self.output_index, self.signature
        )
    }

    /// [`Candidate::describe`] with the signature elided in the middle —
    /// for the progress lines a sweep prints per candidate, where a
    /// kilobyte of term is noise. Never use it to identify a candidate:
    /// elision can make two distinct candidates read alike.
    pub fn describe_short(&self) -> String {
        const HEAD: usize = 200;
        const TAIL: usize = 60;
        let sig: Vec<char> = self.signature.chars().collect();
        let sig = if sig.len() <= HEAD + TAIL + 3 {
            self.signature.clone()
        } else {
            format!(
                "{}...{}",
                sig[..HEAD].iter().collect::<String>(),
                sig[sig.len() - TAIL..].iter().collect::<String>()
            )
        };
        format!("{} [out {}] {}", self.constructor, self.output_index, sig)
    }
}

/// A DESCRIPTION of one e-class: what constructors it holds, what the
/// recorder called it, and the e-nodes inside it. Handed to the predicate
/// of [`Locator::find_class`].
pub struct ClassView<'l, 'e> {
    loc: &'l Locator<'e>,
    class: ClassId,
}

impl<'l, 'e> ClassView<'l, 'e> {
    /// The class handle — opaque; pass it back to the locator, never
    /// assert on its text.
    pub fn class(&self) -> &ClassId {
        &self.class
    }

    /// Every constructor name in the class (subsumed spellings included,
    /// exactly like the marker's own value readers).
    pub fn ops(&self) -> BTreeSet<String> {
        self.loc
            .class_nodes(&self.class)
            .iter()
            .map(|n| n.op.clone())
            .collect()
    }

    /// Does the class hold a node with this constructor?
    pub fn has_op(&self, name: &str) -> bool {
        self.loc
            .class_nodes(&self.class)
            .iter()
            .any(|n| n.op == name)
    }

    /// The recorder / fixture `let` name bound to this class, if any —
    /// `class_data[class].extra["let"]`.
    pub fn let_name(&self) -> Option<String> {
        self.loc.let_name(&self.class)
    }

    /// The node with this constructor. When the class holds several
    /// spellings of one constructor, the one with the SMALLEST signature
    /// wins — a content-determined pick, never node order. Use
    /// [`ClassView::nodes_with_op`] when the multiplicity is the point.
    pub fn node(&self, op: &str) -> Option<&'e Node> {
        self.nodes_with_op(op).into_iter().next()
    }

    /// Every node with this constructor, ordered by signature.
    pub fn nodes_with_op(&self, op: &str) -> Vec<&'e Node> {
        let mut nodes: Vec<&'e Node> = self
            .loc
            .class_nodes(&self.class)
            .iter()
            .copied()
            .filter(|n| n.op == op)
            .collect();
        nodes.sort_by_cached_key(|n| self.loc.signature_of(n));
        nodes
    }

    /// Every node in the class, ordered by `(constructor, signature)`.
    pub fn nodes(&self) -> Vec<&'e Node> {
        self.loc.nodes_in(&self.class)
    }

    /// The class's own rendering — its let name when it has one, else the
    /// sorted set of spellings inside it. Id-free; safe to print in a
    /// failure message.
    pub fn signature(&self) -> String {
        self.loc.class_signature(&self.class)
    }
}

/// The fluent, id-free view over a serialized e-graph.
///
/// Cheap to construct; holds a private render memo, so keep one per
/// e-graph rather than rebuilding it per lookup.
pub struct Locator<'e> {
    egraph: &'e EGraph,
    depth: usize,
    memo: RefCell<HashMap<(ClassId, usize), Rc<str>>>,
}

impl<'e> Locator<'e> {
    /// A locator at [`DEFAULT_SIGNATURE_DEPTH`].
    pub fn new(egraph: &'e EGraph) -> Self {
        Self {
            egraph,
            depth: DEFAULT_SIGNATURE_DEPTH,
            memo: RefCell::new(HashMap::new()),
        }
    }

    /// Same locator at a different signature depth. Deeper separates more
    /// spellings and costs more; shallower is faster and coarser.
    pub fn with_depth(self, depth: usize) -> Self {
        Self {
            egraph: self.egraph,
            depth,
            memo: RefCell::new(HashMap::new()),
        }
    }

    /// The e-graph this locator reads.
    pub fn egraph(&self) -> &'e EGraph {
        self.egraph
    }

    // ------------------------------------------------------------------
    // (a) ROOTS
    // ------------------------------------------------------------------

    /// The boundary OUTPUT buffer-tensor classes IN OUTPUT-SLOT ORDER —
    /// the `BufferOutputLit` root's `BufferTensorCons` spine, walked head
    /// by head. Slot order is the program's, not the e-graph's.
    ///
    /// Panics if the e-graph holds two structurally different output
    /// lists: an ambiguous boundary is a fixture bug, and silently picking
    /// one is exactly the id-order dependence this module removes.
    pub fn outputs(&self) -> Vec<ClassId> {
        self.buffer_spine("BufferOutputLit")
    }

    /// The boundary INPUT buffer-tensor classes in slot order (the
    /// `BufferInputLit` spine). Empty when the program declares no
    /// explicit input list.
    pub fn inputs(&self) -> Vec<ClassId> {
        self.buffer_spine("BufferInputLit")
    }

    /// The class bound to this `let` name in the program text, if any.
    /// Let names come from the SOURCE, so they are stable across runs and
    /// across estate edits in a way class ids are not.
    pub fn let_class(&self, name: &str) -> Option<ClassId> {
        self.egraph.class_data.iter().find_map(|(class, data)| {
            (data.extra.get("let").map(String::as_str) == Some(name)).then(|| class.clone())
        })
    }

    /// The LOGICAL TENSOR class of the boundary input called `name` — the
    /// `LogicalTensorInputLit` / `LogicalTensorNamed` whose `LogicalIdLit`
    /// carries that string. Matches with or without the egglog string
    /// quotes, so both `"x"` and `x` find it.
    pub fn input_class(&self, name: &str) -> Option<ClassId> {
        self.egraph
            .nodes
            .values()
            .filter(|n| n.op == "LogicalTensorInputLit" || n.op == "LogicalTensorNamed")
            .find(|n| self.logical_name(&n.eclass).as_deref() == Some(name))
            .map(|n| n.eclass.clone())
    }

    /// The input NAME carried by a logical-tensor class, unquoted.
    pub fn logical_name(&self, logical: &ClassId) -> Option<String> {
        for node in self.class_nodes(logical) {
            if node.op != "LogicalTensorInputLit" && node.op != "LogicalTensorNamed" {
                continue;
            }
            let id_class = self.try_child(node, 0)?;
            for id_node in self.class_nodes(&id_class) {
                if id_node.op != "LogicalIdLit" {
                    continue;
                }
                let payload = self.try_child(id_node, 0)?;
                if let Some(text) = self.class_nodes(&payload).first().map(|n| n.op.as_str()) {
                    return Some(text.trim_matches('"').to_string());
                }
            }
        }
        None
    }

    /// The recorder OUTPUT STEMS in slot order, READ OFF THE E-GRAPH.
    ///
    /// The recorder names an output's boundary lets `{stem}_layout_tensor`
    /// / `{stem}_buffer_id` / `{stem}_buffer_tensor` (see
    /// `Graph::bound_parts`, where the stem is `natout{K}` and `K` is the
    /// output node's index). K shifts whenever recorder numbering does, so
    /// tests must never hardcode it — nor scrape it out of the program
    /// text, which is what this replaces.
    ///
    /// Panics when a slot's stem cannot be derived: a boundary whose lets
    /// are unnamed has no stem to report, and returning a placeholder
    /// would hide the fixture bug.
    pub fn output_stems(&self) -> Vec<String> {
        self.outputs()
            .iter()
            .map(|class| {
                self.boundary_stem(class).unwrap_or_else(|| {
                    panic!(
                        "output slot {} has no derivable stem: no `{{stem}}_buffer_tensor` / \
                         `_layout_tensor` / `_buffer_id` let name reaches it",
                        self.class_signature(class)
                    )
                })
            })
            .collect()
    }

    fn boundary_stem(&self, buffer_tensor: &ClassId) -> Option<String> {
        let strip = |name: String, suffix: &str| name.strip_suffix(suffix).map(str::to_string);
        if let Some(stem) = self
            .let_name(buffer_tensor)
            .and_then(|n| strip(n, "_buffer_tensor"))
        {
            return Some(stem);
        }
        let lit = self.view(buffer_tensor).node("BufferTensorLit")?;
        if let Some(stem) = self
            .try_child(lit, 0)
            .and_then(|c| self.let_name(&c))
            .and_then(|n| strip(n, "_layout_tensor"))
        {
            return Some(stem);
        }
        self.try_child(lit, 1)
            .and_then(|c| self.let_name(&c))
            .and_then(|n| strip(n, "_buffer_id"))
    }

    // ------------------------------------------------------------------
    // (b) DESCRIPTION-BASED SEARCH
    // ------------------------------------------------------------------

    /// A description of one class.
    pub fn view<'l>(&'l self, class: &ClassId) -> ClassView<'l, 'e> {
        ClassView {
            loc: self,
            class: class.clone(),
        }
    }

    /// EVERY class matching the description, ordered by class signature.
    ///
    /// Classes whose signatures TIE are genuinely indistinguishable by
    /// description: their relative order is unspecified and asserting on
    /// it would be exactly the id-order tie-break this module removes.
    /// Narrow the predicate — or use [`Locator::find_one_class`] — rather
    /// than indexing into the result.
    pub fn find_class(&self, predicate: impl Fn(&ClassView<'_, 'e>) -> bool) -> Vec<ClassId> {
        let mut hits: Vec<ClassId> = self
            .egraph
            .classes()
            .keys()
            .filter(|class| predicate(&self.view(class)))
            .cloned()
            .collect();
        hits.sort_by_cached_key(|class| self.class_digest(class));
        hits
    }

    /// The UNIQUE class matching the description. Panics listing every
    /// match's signature when the description is not sharp enough (or
    /// matches nothing) — the intended failure mode: sharpen the
    /// description instead of indexing.
    pub fn find_one_class(&self, predicate: impl Fn(&ClassView<'_, 'e>) -> bool) -> ClassId {
        let hits = self.find_class(predicate);
        match hits.len() {
            1 => hits.into_iter().next().expect("checked length"),
            _ => panic!(
                "find_one_class: description matched {} classes, expected exactly 1\n{}",
                hits.len(),
                hits.iter()
                    .map(|c| format!("  - {}", self.class_digest(c)))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        }
    }

    /// PATH STEP: the class of `node`'s child `index`. Panics naming the
    /// constructor and the arity when the slot is absent — a path step off
    /// the end of a term is a fixture bug, not a `None`.
    pub fn child(&self, node: &Node, index: usize) -> ClassId {
        self.try_child(node, index).unwrap_or_else(|| {
            panic!(
                "{} has no child {index} (arity {})",
                node.op,
                node.children.len()
            )
        })
    }

    /// [`Locator::child`], fallible.
    pub fn try_child(&self, node: &Node, index: usize) -> Option<ClassId> {
        let id = node.children.get(index)?;
        Some(self.egraph.nodes.get(id)?.eclass.clone())
    }

    /// Every e-node in the class, ordered by `(constructor, signature)` —
    /// a content order, so `.first()` on the result is reproducible
    /// without depending on node ids.
    pub fn nodes_in(&self, class: &ClassId) -> Vec<&'e Node> {
        let mut nodes = self.class_nodes(class);
        nodes.sort_by_cached_key(|n| (n.op.clone(), self.signature_of(n)));
        nodes
    }

    /// The node with this constructor in this class — see
    /// [`ClassView::node`] for the multiplicity rule.
    pub fn node_in(&self, class: &ClassId, op: &str) -> Option<&'e Node> {
        self.view(class).node(op)
    }

    /// The `IntLit` value a class carries, if it carries one — the leaf
    /// read every geometry walk ends at. Generalizes the `parse_dim`
    /// copies on the marker boards; `None` means the extent is symbolic
    /// (or absent), never that it is zero.
    pub fn int_literal(&self, class: &ClassId) -> Option<i64> {
        for node in self.class_nodes(class) {
            if node.op != "IntLit" {
                continue;
            }
            let payload = self.try_child(node, 0)?;
            if let Some(value) = self
                .class_nodes(&payload)
                .iter()
                .find_map(|n| n.op.parse::<i64>().ok())
            {
                return Some(value);
            }
        }
        None
    }

    /// The recorder / fixture `let` name of a class, if any.
    pub fn let_name(&self, class: &ClassId) -> Option<String> {
        self.egraph
            .class_data
            .get(class)
            .and_then(|data| data.extra.get("let"))
            .cloned()
    }

    // ------------------------------------------------------------------
    // (c) SIGNATURES
    // ------------------------------------------------------------------

    /// THE ID-FREE CANONICAL RENDERING of an e-node:
    /// `Constructor(child, child, …)`, children resolved CLASS by class
    /// and rendered recursively.
    ///
    /// LEAVES are spelled by content: a class with a `let` name renders as
    /// that name (`nat0_layout_tensor`), a nullary constructor as itself
    /// (`CublasLtOperationT`), a literal payload as its text (`3`, `"x"`).
    /// No class id and no node id is ever emitted — a class that renders
    /// nothing else prints as its sorted CONSTRUCTOR SET (`{IntAdd|IntMul}`)
    /// or as `{}` when it is empty.
    ///
    /// TIES IN CHILD RENDERING are broken by STRUCTURAL CONTENT: a class
    /// with several spellings renders every one of them, sorts the
    /// strings, dedups, and joins with `|` — so which spelling the
    /// e-graph happens to store first never shows. Subsumed spellings are
    /// skipped (they are retired terms) unless the whole class is
    /// subsumed, in which case they are all that is left to describe it.
    ///
    /// DEPTH BOUND: [`DEFAULT_SIGNATURE_DEPTH`] class levels, one consumed
    /// per class hop. At the bound a class falls back to its constructor
    /// set, and an e-node to `Constructor/arity`.
    ///
    /// CYCLE HANDLING: the depth bound IS the cycle handling. A saturated
    /// e-graph is cyclic in general (the layout copy/view re-description
    /// 2-cycles are the standing example), so a cycle simply renders as
    /// its own unrolling until the bound cuts it. No path state is kept,
    /// which is what makes the per-`(class, depth)` render memo sound.
    ///
    /// Signatures are stable across runs and across e-class renumbering,
    /// but NOT across estate edits that change the terms themselves —
    /// which is the point: a changed term should change the description.
    pub fn signature(&self, enode: &NodeId) -> String {
        match self.egraph.nodes.get(enode) {
            Some(node) => self.render_node(node, self.depth),
            None => "{}".to_string(),
        }
    }

    /// [`Locator::signature`] at an explicit depth.
    pub fn signature_with_depth(&self, enode: &NodeId, depth: usize) -> String {
        match self.egraph.nodes.get(enode) {
            Some(node) => self.render_node(node, depth),
            None => "{}".to_string(),
        }
    }

    /// The id-free rendering of a CLASS: its let name when it has one,
    /// else the joined renderings of the spellings inside it.
    pub fn class_signature(&self, class: &ClassId) -> String {
        self.render_class(class, self.depth)
    }

    /// The CHEAP id-free description of a class: its let name when it has
    /// one, else its sorted constructor set (`{IntAdd|IntMul}`) — the
    /// depth-0 rendering. Used for ordering and failure messages, where a
    /// full [`Locator::class_signature`] can run to kilobytes.
    pub fn class_digest(&self, class: &ClassId) -> String {
        self.render_class(class, 0)
    }

    /// [`Locator::signature`] for a node reached by walking (the nodes
    /// [`Locator::nodes_in`] and [`ClassView::node`] hand back), which
    /// carry no id of their own.
    pub fn signature_of(&self, node: &Node) -> String {
        self.render_node(node, self.depth)
    }

    fn render_node(&self, node: &Node, depth: usize) -> String {
        if node.children.is_empty() {
            return node.op.clone();
        }
        if depth == 0 {
            return format!("{}/{}", node.op, node.children.len());
        }
        let args: Vec<String> = node
            .children
            .iter()
            .map(|id| match self.egraph.nodes.get(id) {
                Some(child) => self.render_class(&child.eclass, depth - 1),
                None => "{}".to_string(),
            })
            .collect();
        format!("{}({})", node.op, args.join(","))
    }

    fn render_class(&self, class: &ClassId, depth: usize) -> String {
        if let Some(name) = self.let_name(class) {
            return name;
        }
        let key = (class.clone(), depth);
        // Two statements on purpose: the borrow must die here, because
        // `render_node` below recurses straight back into this function.
        let hit = self.memo.borrow().get(&key).cloned();
        if let Some(hit) = hit {
            return hit.to_string();
        }
        let all = self.class_nodes(class);
        let live: Vec<&Node> = all.iter().copied().filter(|n| !n.subsumed).collect();
        let nodes = if live.is_empty() { all } else { live };
        // CONSTRUCTORS describe a class; egglog FUNCTION ROWS
        // (`shape-of`, `layout-of`, `input-layout-tensor-list-of`, ...)
        // merely point at it from a derived table, and rendering them
        // buries the description under the whole analysis. Keep them
        // only when nothing else is left to say.
        let constructors: Vec<&Node> = nodes
            .iter()
            .copied()
            .filter(|n| !is_function_row(&n.op))
            .collect();
        let nodes = if constructors.is_empty() {
            nodes
        } else {
            constructors
        };

        let rendered = if nodes.is_empty() {
            "{}".to_string()
        } else if depth == 0 {
            let ops: BTreeSet<&str> = nodes.iter().map(|n| n.op.as_str()).collect();
            format!("{{{}}}", ops.into_iter().collect::<Vec<_>>().join("|"))
        } else {
            let mut alternatives: Vec<String> = nodes
                .iter()
                .map(|node| self.render_node(node, depth))
                .collect();
            alternatives.sort();
            alternatives.dedup();
            if alternatives.len() == 1 {
                alternatives.remove(0)
            } else {
                let elided = alternatives.len() > MAX_ALTERNATIVES;
                alternatives.truncate(MAX_ALTERNATIVES);
                format!(
                    "{{{}{}}}",
                    alternatives.join("|"),
                    if elided { "|.." } else { "" }
                )
            }
        };
        self.memo
            .borrow_mut()
            .insert(key, Rc::from(rendered.as_str()));
        rendered
    }

    /// The class's nodes, in the e-graph's own order. PRIVATE on purpose:
    /// that order is node-id order, and everything public sorts it by
    /// content before handing it out.
    fn class_nodes(&self, class: &ClassId) -> Vec<&'e Node> {
        match self.egraph.classes().get(class) {
            Some(c) => c
                .nodes
                .iter()
                .filter_map(|id| self.egraph.nodes.get(id))
                .collect(),
            None => Vec::new(),
        }
    }

    fn buffer_spine(&self, root_op: &str) -> Vec<ClassId> {
        let mut spines: Vec<Vec<ClassId>> = Vec::new();
        for node in self.egraph.nodes.values().filter(|n| n.op == root_op) {
            let Some(list) = self.try_child(node, 0) else {
                continue;
            };
            let spine = self.walk_cons(&list, "BufferTensorCons", "BufferTensorNil");
            if !spines.contains(&spine) {
                spines.push(spine);
            }
        }
        match spines.len() {
            0 => Vec::new(),
            1 => spines.remove(0),
            _ => panic!(
                "{root_op} names {} structurally different boundary lists; the boundary must be \
                 unambiguous (picking one would be an e-graph-order tie-break)",
                spines.len()
            ),
        }
    }

    /// Walk a cons spine head by head, in list order.
    pub fn walk_cons(&self, list: &ClassId, cons_op: &str, nil_op: &str) -> Vec<ClassId> {
        let mut out = Vec::new();
        let mut current = Some(list.clone());
        let mut guard = 0usize;
        while let Some(class) = current {
            guard += 1;
            assert!(guard <= 4096, "{cons_op} spine did not terminate");
            let view = self.view(&class);
            if view.has_op(nil_op) {
                break;
            }
            let cells: BTreeSet<(ClassId, ClassId)> = view
                .nodes_with_op(cons_op)
                .into_iter()
                .filter_map(|cons| Some((self.try_child(cons, 0)?, self.try_child(cons, 1)?)))
                .collect();
            let mut cells = cells.into_iter();
            let Some((head, tail)) = cells.next() else {
                break;
            };
            assert!(
                cells.next().is_none(),
                "{cons_op} cell holds several structurally different (head, tail) pairs; the list \
                 is ambiguous and picking one would be an e-graph-order tie-break"
            );
            out.push(head);
            current = Some(tail);
        }
        out
    }

    // ------------------------------------------------------------------
    // (c) CANDIDATES
    // ------------------------------------------------------------------

    /// The producer index over an explicit runtime matcher set — the
    /// raw material [`Locator::candidates`] and the election helpers read.
    /// Build it once and pass it around; it is not cheap.
    pub fn producer_index(
        &self,
        matchers: Vec<Box<dyn crate::layout_ir::OpMatcher>>,
    ) -> ProducerIndex {
        crate::extractor::producer_index_with_matchers(self.egraph, matchers)
    }

    /// This class's producer candidates, DESCRIBED — ordered by
    /// `(constructor, signature, output_index)`.
    ///
    /// The producer index's own order is `(name, enode.to_string(),
    /// output_index)` — an id-order dependence. This re-sorts on the
    /// signature so the returned order is content-determined; two entries
    /// that still tie are the same constructor over the same rendered
    /// term, and the election helpers refuse to pick between them.
    pub fn candidates(&self, index: &ProducerIndex, class: &ClassId) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = index
            .get(class)
            .into_iter()
            .flatten()
            .map(|(constructor, choice)| Candidate {
                constructor: constructor.clone(),
                output_index: choice.output_index,
                signature: self.signature(&choice.enode),
                enode: choice.enode.clone(),
            })
            .collect();
        out.sort_by(|a, b| {
            (&a.constructor, &a.signature, a.output_index).cmp(&(
                &b.constructor,
                &b.signature,
                b.output_index,
            ))
        });
        out
    }

    /// The class's single candidate with this constructor.
    ///
    /// Panics printing EVERY candidate of the class when two or more share
    /// the constructor — the failure a cost/label tie-break used to hide:
    /// the same constructor at the same cost with the same label, chosen
    /// by id order. Take the printed signatures and assert per candidate
    /// ([`Locator::elect_each`]) instead.
    pub fn assert_unique_candidate(
        &self,
        index: &ProducerIndex,
        class: &ClassId,
        constructor: &str,
    ) -> Candidate {
        let all = self.candidates(index, class);
        let matching: Vec<&Candidate> = all
            .iter()
            .filter(|c| c.constructor == constructor)
            .collect();
        if matching.len() == 1 {
            return matching[0].clone();
        }
        panic!(
            "assert_unique_candidate: {} candidate(s) named {constructor} in the class, expected \
             exactly 1 — an id-order tie-break would decide this one.\ncandidates of {}:\n{}",
            matching.len(),
            self.class_signature(class),
            all.iter()
                .map(|c| format!("  - {}", c.describe()))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    // ------------------------------------------------------------------
    // (d) ELECTION
    // ------------------------------------------------------------------

    /// FORCE one e-node: `base` with that e-node's choice written into
    /// EVERY class where it is a candidate. `None` when it is a candidate
    /// for no demanded class at all (the per-e-node sweeps report and skip
    /// those).
    ///
    /// `base` supplies the rest of the genome — viability-aware election
    /// steers every OTHER class, including the escalated copy/materialize
    /// routes the forced e-node's operands may need. Core owns no
    /// preference order, so the base is the caller's (runtimes pass their
    /// own `genome_with_ordering` result).
    pub fn force_enode(
        &self,
        index: &ProducerIndex,
        base: &Genome,
        enode: &NodeId,
    ) -> Option<Genome> {
        let mut genome = base.clone();
        let mut forced = 0usize;
        for (class, candidates) in index {
            if let Some((_, choice)) = candidates.iter().find(|(_, c)| &c.enode == enode) {
                genome.choices.insert(class.clone(), choice.clone());
                forced += 1;
            }
        }
        (forced > 0).then_some(genome)
    }

    /// ELECT THE CANDIDATE THIS SIGNATURE DESCRIBES. Panics printing every
    /// candidate of the class when the signature matches none or several.
    pub fn elect_by_signature(
        &self,
        index: &ProducerIndex,
        base: &Genome,
        class: &ClassId,
        signature: &str,
    ) -> Genome {
        let all = self.candidates(index, class);
        let matching: Vec<&Candidate> = all.iter().filter(|c| c.signature == signature).collect();
        let candidate = match matching.len() {
            1 => matching[0],
            n => panic!(
                "elect_by_signature: {n} candidate(s) match\n  {signature}\ncandidates of {}:\n{}",
                self.class_signature(class),
                all.iter()
                    .map(|c| format!("  - {}", c.describe()))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        };
        self.force_enode(index, base, &candidate.enode)
            .expect("a candidate of this class is a candidate somewhere")
    }

    /// ONE GENOME PER CANDIDATE of the class, in [`Locator::candidates`]
    /// order — the per-e-node pattern's driver. Assert on what EACH
    /// elected candidate's own terms say; never on which one an untouched
    /// extraction happens to pick.
    pub fn elect_each(
        &self,
        index: &ProducerIndex,
        base: &Genome,
        class: &ClassId,
    ) -> Vec<(Candidate, Genome)> {
        self.candidates(index, class)
            .into_iter()
            .filter_map(|candidate| {
                let genome = self.force_enode(index, base, &candidate.enode)?;
                Some((candidate, genome))
            })
            .collect()
    }
}

/// Is this constructor name an egglog FUNCTION (a derived table row)
/// rather than a datatype CONSTRUCTOR? The estate spells functions in
/// kebab-case (`shape-of`, `int-subst-of`, `expr-list-length-of`) and
/// constructors in CamelCase, so the leading character plus a hyphen
/// decides it. Negative literals (`-1`) start with the hyphen itself and
/// are never mistaken for functions.
fn is_function_row(op: &str) -> bool {
    op.contains('-') && op.starts_with(|c: char| c.is_ascii_lowercase())
}
