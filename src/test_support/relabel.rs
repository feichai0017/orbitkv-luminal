//! RANDOM-RELABEL INVARIANCE — the harness for Austin's ruling
//! (2026-09-02): *"Assume that all node ids and eclass ids are random
//! every time. If you want to find an eclass you need to find it. Use
//! that in your design."*
//!
//! Serialized identities (`ClassId` = `"Layout-2216"`, `NodeId` =
//! `"function-55-LayoutTensorOpAdd"`) are OPAQUE KEYS inside one
//! serialized e-graph and nothing more. No decision, ordering,
//! tie-break, or test may read their string values or their sort order.
//!
//! This module manufactures the counterexample: a consistent random
//! BIJECTION over every `ClassId` and every `NodeId` of a serialized
//! e-graph, applied everywhere an id appears, with the `IndexMap`
//! insertion orders shuffled as well — because a genuinely fresh egglog
//! run would have produced a different order too, and iteration order is
//! part of what an id-order dependence reads.
//!
//! WHAT IS PRESERVED (deliberately, so id-*shape* readers keep working
//! and only id-*value* readers can notice):
//!
//!  * a `ClassId`'s SORT-NAME prefix — the serializer writes
//!    `"{sort}-{value.rep()}"` and recovers the pair by splitting on the
//!    LAST `-` (`vendor/egglog-checkout/src/serialize.rs:249`), so
//!    `"Layout-2216"` becomes `"Layout-77"`, never `"IntExpr-77"`;
//!  * a `NodeId`'s FUNCTION-NAME suffix — the serializer writes
//!    `"function-{offset}-{name}"` and splits the tag off the FRONT
//!    (`serialize.rs:268`), so `"function-55-LayoutTensorOpAdd"` becomes
//!    `"function-3-LayoutTensorOpAdd"`;
//!  * `primitive-…` / `dummy-…` node ids, which EMBED a class id, are
//!    relabeled through the very same class bijection, so a primitive
//!    node keeps naming its own class;
//!  * `split-…` node ids are relabeled recursively.
//!
//! WHAT CHANGES: every numeric part, and the iteration order of `nodes`,
//! `class_data` and `root_eclasses`.
//!
//! The permutation is drawn from the seed over the numbers the e-graph
//! ALREADY uses (a permutation of the existing pool, never fresh
//! numbers), which makes the map bijective by construction and keeps id
//! string lengths in the same distribution.

use std::collections::{BTreeSet, HashMap};

use crate::prelude::egraph_serialize::{ClassId, EGraph, Node, NodeId};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// A consistent random bijection over one serialized e-graph's
/// identities, plus the insertion-order permutations that go with it.
///
/// Built from a graph, then applied to it (or, via [`Relabeling::inverse`],
/// to the relabeled result to recover the original exactly).
#[derive(Debug, Clone)]
pub struct Relabeling {
    classes: HashMap<ClassId, ClassId>,
    nodes: HashMap<NodeId, NodeId>,
    /// Output position `p` takes the source entry at index `node_order[p]`.
    node_order: Vec<usize>,
    class_data_order: Vec<usize>,
    root_order: Vec<usize>,
}

/// `Relabeling::new(egraph, seed).apply(egraph)` — the one-call form.
pub fn relabel_egraph(egraph: &EGraph, seed: u64) -> EGraph {
    Relabeling::new(egraph, seed).apply(egraph)
}

/// A `ClassId` splits into (sort-name prefix, numeric rep) on the LAST
/// `-`; sort names may themselves contain `-`, which is exactly why the
/// serializer splits from the right.
fn split_class(id: &str) -> Option<(&str, u64)> {
    let (prefix, digits) = id.rsplit_once('-')?;
    Some((prefix, digits.parse::<u64>().ok()?))
}

/// The four serialized `NodeId` shapes (`serialize.rs::to_node_id`).
enum NodeShape<'a> {
    /// `function-<offset>-<name>`; `name` may contain `-`.
    Function { offset: u64, name: &'a str },
    /// `primitive-<ClassId>` / `dummy-<ClassId>`.
    Wrapped { tag: &'a str, class: &'a str },
    /// `split-<NodeId>`.
    Split { inner: &'a str },
    /// Anything else — left alone rather than mangled.
    Opaque,
}

fn split_node(id: &str) -> NodeShape<'_> {
    if let Some(rest) = id.strip_prefix("function-") {
        if let Some((offset, name)) = rest.split_once('-') {
            if let Ok(offset) = offset.parse::<u64>() {
                return NodeShape::Function { offset, name };
            }
        }
        return NodeShape::Opaque;
    }
    if let Some(rest) = id.strip_prefix("split-") {
        return NodeShape::Split { inner: rest };
    }
    for tag in ["primitive", "dummy"] {
        if let Some(rest) = id.strip_prefix(tag).and_then(|r| r.strip_prefix('-')) {
            return NodeShape::Wrapped { tag, class: rest };
        }
    }
    NodeShape::Opaque
}

/// A permutation of a number pool, drawn from `rng`: the i-th smallest
/// member maps to the i-th member of a shuffle of the same pool. Keeps
/// the map bijective and the number range unchanged.
fn permute_pool(pool: &BTreeSet<u64>, rng: &mut StdRng) -> HashMap<u64, u64> {
    let sorted: Vec<u64> = pool.iter().copied().collect();
    let mut shuffled = sorted.clone();
    shuffled.shuffle(rng);
    sorted.into_iter().zip(shuffled).collect()
}

fn shuffled_order(len: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    order.shuffle(rng);
    order
}

fn invert_order(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; order.len()];
    for (position, &source) in order.iter().enumerate() {
        inverse[source] = position;
    }
    inverse
}

impl Relabeling {
    /// Draw the bijection for `egraph` from `seed`.
    pub fn new(egraph: &EGraph, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // --- every class id the graph mentions, including the ones
        // embedded inside primitive/dummy node ids ---------------------
        let mut class_universe: BTreeSet<ClassId> = BTreeSet::new();
        for (id, node) in &egraph.nodes {
            class_universe.insert(node.eclass.clone());
            let mut spelling: &str = id.as_ref();
            loop {
                match split_node(spelling) {
                    NodeShape::Split { inner } => spelling = inner,
                    NodeShape::Wrapped { class, .. } => {
                        class_universe.insert(ClassId::from(class));
                        break;
                    }
                    _ => break,
                }
            }
        }
        class_universe.extend(egraph.class_data.keys().cloned());
        class_universe.extend(egraph.root_eclasses.iter().cloned());

        let reps: BTreeSet<u64> = class_universe
            .iter()
            .filter_map(|id| split_class(id.as_ref()).map(|(_, rep)| rep))
            .collect();
        let rep_map = permute_pool(&reps, &mut rng);
        let relabel_class = |id: &ClassId| -> ClassId {
            match split_class(id.as_ref()) {
                Some((prefix, rep)) => match rep_map.get(&rep) {
                    Some(fresh) => ClassId::from(format!("{prefix}-{fresh}")),
                    None => id.clone(),
                },
                None => id.clone(),
            }
        };
        let classes: HashMap<ClassId, ClassId> = class_universe
            .iter()
            .map(|id| (id.clone(), relabel_class(id)))
            .collect();

        // --- the function-row pool -----------------------------------
        let offsets: BTreeSet<u64> = egraph
            .nodes
            .keys()
            .filter_map(|id| match split_node(id.as_ref()) {
                NodeShape::Function { offset, .. } => Some(offset),
                _ => None,
            })
            .collect();
        let offset_map = permute_pool(&offsets, &mut rng);

        // A node id is relabeled STRUCTURALLY: the row number through the
        // offset permutation, an embedded class through the class
        // bijection, a split wrapper recursively.
        fn relabel_node_spelling(
            spelling: &str,
            offset_map: &HashMap<u64, u64>,
            relabel_class: &dyn Fn(&ClassId) -> ClassId,
        ) -> String {
            match split_node(spelling) {
                NodeShape::Function { offset, name } => {
                    let fresh = offset_map.get(&offset).copied().unwrap_or(offset);
                    format!("function-{fresh}-{name}")
                }
                NodeShape::Wrapped { tag, class } => {
                    format!("{tag}-{}", relabel_class(&ClassId::from(class)))
                }
                NodeShape::Split { inner } => format!(
                    "split-{}",
                    relabel_node_spelling(inner, offset_map, relabel_class)
                ),
                NodeShape::Opaque => spelling.to_string(),
            }
        }
        let nodes: HashMap<NodeId, NodeId> = egraph
            .nodes
            .keys()
            .map(|id| {
                (
                    id.clone(),
                    NodeId::from(relabel_node_spelling(
                        id.as_ref(),
                        &offset_map,
                        &relabel_class,
                    )),
                )
            })
            .collect();

        // --- and the insertion orders --------------------------------
        let node_order = shuffled_order(egraph.nodes.len(), &mut rng);
        let class_data_order = shuffled_order(egraph.class_data.len(), &mut rng);
        let root_order = shuffled_order(egraph.root_eclasses.len(), &mut rng);

        Self {
            classes,
            nodes,
            node_order,
            class_data_order,
            root_order,
        }
    }

    /// The class bijection (identity on ids this relabeling never saw).
    pub fn class(&self, id: &ClassId) -> ClassId {
        self.classes.get(id).cloned().unwrap_or_else(|| id.clone())
    }

    /// The node bijection (identity on ids this relabeling never saw).
    pub fn node(&self, id: &NodeId) -> NodeId {
        self.nodes.get(id).cloned().unwrap_or_else(|| id.clone())
    }

    /// The relabeling that undoes this one — inverted maps AND inverted
    /// insertion orders, so `r.inverse().apply(&r.apply(g)) == g`
    /// including iteration order.
    pub fn inverse(&self) -> Self {
        Self {
            classes: self
                .classes
                .iter()
                .map(|(from, to)| (to.clone(), from.clone()))
                .collect(),
            nodes: self
                .nodes
                .iter()
                .map(|(from, to)| (to.clone(), from.clone()))
                .collect(),
            node_order: invert_order(&self.node_order),
            class_data_order: invert_order(&self.class_data_order),
            root_order: invert_order(&self.root_order),
        }
    }

    /// Rewrite `egraph` under this bijection. Every id is mapped in every
    /// position it occurs — `nodes` keys, `Node::eclass`, `Node::children`,
    /// `class_data` keys, `root_eclasses` — and the three iteration orders
    /// are permuted. `op`, arity, `cost` and `subsumed` are carried
    /// verbatim.
    ///
    /// Panics if `egraph` is not the shape this relabeling was drawn for
    /// (its `add_node` would otherwise report a duplicate, which is the
    /// same bug found one step later).
    pub fn apply(&self, egraph: &EGraph) -> EGraph {
        assert_eq!(
            egraph.nodes.len(),
            self.node_order.len(),
            "relabeling drawn for a different e-graph (node count)"
        );
        assert_eq!(
            egraph.class_data.len(),
            self.class_data_order.len(),
            "relabeling drawn for a different e-graph (class_data count)"
        );
        assert_eq!(
            egraph.root_eclasses.len(),
            self.root_order.len(),
            "relabeling drawn for a different e-graph (root count)"
        );

        let mut out = EGraph::default();
        let source_nodes: Vec<(&NodeId, &Node)> = egraph.nodes.iter().collect();
        for &position in &self.node_order {
            let (id, node) = source_nodes[position];
            out.add_node(
                self.node(id),
                Node {
                    op: node.op.clone(),
                    children: node.children.iter().map(|c| self.node(c)).collect(),
                    eclass: self.class(&node.eclass),
                    cost: node.cost,
                    subsumed: node.subsumed,
                },
            );
        }
        let source_class_data: Vec<_> = egraph.class_data.iter().collect();
        for &position in &self.class_data_order {
            let (id, data) = source_class_data[position];
            out.class_data.insert(self.class(id), data.clone());
        }
        out.root_eclasses = self
            .root_order
            .iter()
            .map(|&position| self.class(&egraph.root_eclasses[position]))
            .collect();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The class partition of a serialized graph, spelled WITHOUT ids:
    /// for every class, the multiset of its nodes' `(op, arity)`. Two
    /// graphs agreeing here have the same partition up to relabeling.
    fn partition_shape(egraph: &EGraph) -> BTreeMap<String, Vec<(String, usize)>> {
        let mut shape: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
        for node in egraph.nodes.values() {
            shape
                .entry(node.eclass.to_string())
                .or_default()
                .push((node.op.clone(), node.children.len()));
        }
        for members in shape.values_mut() {
            members.sort();
        }
        shape
    }

    /// A real serialized e-graph, assembled from CORE alone (no runtime
    /// matcher set): shapes, layouts, a logical input and a view apply —
    /// enough sorts, enough classes, and real `function-<row>-<Name>` /
    /// `primitive-<sort>-<rep>` node ids.
    fn fixture() -> EGraph {
        let body = r#"
(let psh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let p (RightMajorContiguousElementLayoutLit psh (bits-of (F32))))
(let plog (LogicalTensorInputLit (LogicalIdLit "p") psh (F32)))
(let plt (LayoutTensorLit plog p))
(let osh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil))))))
(let v (LogicalIndexMapApply plog (IndexMapLit (IntExprCons (CoordVar osh 2) (IntExprCons (CoordVar osh 0) (IntExprNil))) psh) osh))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!(
            "{}\n\n{body}",
            crate::egglog_snippet::assembled_program_for(&[])
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &full)
            .expect("the core fixture program runs");
        egraph.serialize(egglog::SerializeConfig::default()).egraph
    }

    #[test]
    fn relabel_is_a_bijection_the_inverse_undoes_exactly() {
        let original = fixture();
        for seed in 0..4u64 {
            let relabeling = Relabeling::new(&original, seed);
            let relabeled = relabeling.apply(&original);
            let back = relabeling.inverse().apply(&relabeled);
            assert_eq!(
                back, original,
                "seed {seed}: relabel-then-inverse must reproduce the e-graph exactly"
            );
            // IndexMap equality is order-insensitive, so pin the orders too.
            assert!(
                back.nodes.keys().eq(original.nodes.keys()),
                "seed {seed}: the inverse must restore node insertion order"
            );
            assert!(
                back.class_data.keys().eq(original.class_data.keys()),
                "seed {seed}: the inverse must restore class_data insertion order"
            );
            assert_eq!(back.root_eclasses, original.root_eclasses);
        }
    }

    #[test]
    fn relabel_preserves_op_arity_subsumption_and_the_class_partition() {
        let original = fixture();
        let relabeling = Relabeling::new(&original, 7);
        let relabeled = relabeling.apply(&original);

        assert_eq!(relabeled.nodes.len(), original.nodes.len());
        assert_eq!(relabeled.class_data.len(), original.class_data.len());
        assert_eq!(relabeled.root_eclasses.len(), original.root_eclasses.len());

        for (id, node) in &original.nodes {
            let fresh = &relabeled.nodes[&relabeling.node(id)];
            assert_eq!(fresh.op, node.op, "op must survive relabeling");
            assert_eq!(
                fresh.children.len(),
                node.children.len(),
                "children arity must survive relabeling"
            );
            assert_eq!(
                fresh.subsumed, node.subsumed,
                "the subsumed flag must survive relabeling"
            );
            assert_eq!(fresh.cost, node.cost, "cost must survive relabeling");
            assert_eq!(
                fresh.eclass,
                relabeling.class(&node.eclass),
                "a node must land in its class's image"
            );
            for (before, after) in node.children.iter().zip(&fresh.children) {
                assert_eq!(*after, relabeling.node(before), "children map consistently");
            }
        }

        // Nodes that shared a class still share one, and no two classes merged.
        let before: Vec<_> = partition_shape(&original).into_values().collect();
        let after: Vec<_> = partition_shape(&relabeled).into_values().collect();
        let mut before_sorted = before;
        let mut after_sorted = after;
        before_sorted.sort();
        after_sorted.sort();
        assert_eq!(
            before_sorted, after_sorted,
            "the class partition must be preserved"
        );
        for (id, node) in &original.nodes {
            for (other_id, other) in &original.nodes {
                assert_eq!(
                    node.eclass == other.eclass,
                    relabeled.nodes[&relabeling.node(id)].eclass
                        == relabeled.nodes[&relabeling.node(other_id)].eclass,
                    "class co-membership must be preserved exactly"
                );
            }
        }
    }

    #[test]
    fn relabel_actually_moves_the_ids_and_the_iteration_order() {
        let original = fixture();
        let relabeled = relabel_egraph(&original, 1);
        assert!(
            original
                .nodes
                .keys()
                .zip(relabeled.nodes.keys())
                .any(|(a, b)| a != b),
            "a relabeling that changes nothing would prove nothing"
        );
        // Sort-name prefixes survive: a Layout class stays a Layout class.
        let sorts = |graph: &EGraph| -> BTreeSet<String> {
            graph
                .nodes
                .values()
                .filter_map(|node| {
                    split_class(node.eclass.as_ref()).map(|(prefix, _)| prefix.to_string())
                })
                .collect()
        };
        assert_eq!(
            sorts(&original),
            sorts(&relabeled),
            "sort-name prefixes must survive relabeling"
        );
    }
}
