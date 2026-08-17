//! Recover the compact row-index input from a lowered `gather_rows` flat index.

use luminal::egglog_utils::{ClassId, NodeId, SerializedEGraph};

/// Find the compact row-index tensor used to build a `gather_rows` flat index.
///
/// `gather_rows(data, indices, d)` lowers the index tensor as:
/// `indices * d + arange(d)`. FlashInfer wants `indices`, not the expanded
/// `(c, d)` flat index.
/// Peel dtype casts: a translated graph indexes with i64 where the engine
/// holds i32, so Casts appear at arbitrary levels of the index arithmetic.
fn peel_casts<'a>(egraph: &'a SerializedEGraph, node: &'a NodeId) -> &'a NodeId {
    let mut cur = node;
    for _ in 0..8 {
        // Elementwise fusion wraps index arithmetic in FusionStart/End regions
        // before the op decodes its inputs; step through them like casts.
        let wrapper = resolve_op_with_kind(egraph, cur, "Cast")
            .or_else(|| resolve_op_with_kind(egraph, cur, "FusionEnd"))
            .or_else(|| resolve_op_with_kind(egraph, cur, "FusionStart"));
        let Some(cast) = wrapper else {
            return cur;
        };
        let (_, children) = &egraph.enodes[cast];
        // Ops differ in whether a dtype child sits between the kind and the
        // input list; find the child that actually IS the IList.
        let ilist = children.iter().skip(1).find(|c| {
            egraph.eclasses.get(*c).is_some_and(|(_, members)| {
                members
                    .iter()
                    .any(|m| matches!(egraph.enodes[m].0.as_str(), "ICons" | "INil"))
            })
        });
        let Some(ilist) = ilist else { return cur };
        let inner = walk_ilist_simple(egraph, ilist);
        let Some(first) = inner.first() else { return cur };
        cur = first;
    }
    cur
}

pub fn try_find_compact_gather_idx<'a>(
    egraph: &'a SerializedEGraph,
    flat_idx_node: &'a NodeId,
) -> Option<&'a NodeId> {
    let flat = peel_casts(egraph, flat_idx_node);
    let add_inputs = logical_binary_inputs(egraph, flat, "Add")?;
    for input in add_inputs {
        let input = peel_casts(egraph, input);
        let Some(mul_inputs) = logical_binary_inputs(egraph, input, "Mul") else {
            continue;
        };
        for mul_input in mul_inputs {
            let mul_input = peel_casts(egraph, mul_input);
            if let Some(input_node) = resolve_input_node(egraph, mul_input) {
                return Some(input_node);
            }
        }
    }
    None
}

fn walk_ilist_simple<'a>(
    egraph: &'a SerializedEGraph,
    ilist_eclass: &'a ClassId,
) -> Vec<&'a NodeId> {
    let mut inputs = Vec::new();
    let mut current = resolve_first_node(egraph, ilist_eclass);

    loop {
        let (label, children) = &egraph.enodes[current];
        if label == "INil" {
            break;
        }
        if label != "ICons" {
            break;
        }
        let ir_node = resolve_first_ir_node(egraph, &children[0]);
        inputs.push(ir_node);
        current = resolve_first_node(egraph, &children[1]);
    }

    inputs
}

fn resolve_first_node<'a>(egraph: &'a SerializedEGraph, eclass: &ClassId) -> &'a NodeId {
    &egraph.eclasses[eclass].1[0]
}

fn resolve_first_ir_node<'a>(egraph: &'a SerializedEGraph, eclass: &ClassId) -> &'a NodeId {
    let nodes = &egraph.eclasses[eclass].1;
    for node in nodes {
        let label = &egraph.enodes[node].0;
        if label == "Op" || label == "Input" {
            return node;
        }
    }
    &nodes[0]
}

fn resolve_input_node<'a>(egraph: &'a SerializedEGraph, node: &'a NodeId) -> Option<&'a NodeId> {
    let class = egraph.node_to_class.get(node)?;
    egraph.eclasses[class]
        .1
        .iter()
        .find(|candidate| egraph.enodes[*candidate].0 == "Input")
}

fn resolve_op_with_kind<'a>(
    egraph: &'a SerializedEGraph,
    node: &'a NodeId,
    kind_substr: &str,
) -> Option<&'a NodeId> {
    let class = egraph.node_to_class.get(node)?;
    for candidate in &egraph.eclasses[class].1 {
        let (label, children) = &egraph.enodes[candidate];
        if label != "Op" || children.is_empty() {
            continue;
        }
        let kind = resolve_first_node(egraph, &children[0]);
        if egraph.enodes[kind].0.contains(kind_substr) {
            return Some(candidate);
        }
    }
    None
}

fn logical_binary_inputs<'a>(
    egraph: &'a SerializedEGraph,
    node: &'a NodeId,
    op_name: &str,
) -> Option<Vec<&'a NodeId>> {
    if let Some(op_node) = resolve_op_with_kind(egraph, node, op_name) {
        let (_, children) = &egraph.enodes[op_node];
        return Some(walk_ilist_simple(egraph, &children[1]));
    }

    let (label, children) = &egraph.enodes[node];
    if label != "Op" || children.len() < 2 {
        return None;
    }
    let kind = resolve_first_node(egraph, &children[0]);
    if egraph.enodes[kind].0.contains("CudaBinaryElementwise") {
        let opcode_class = egraph.enodes[kind].1.first()?;
        let opcode_node = resolve_first_node(egraph, opcode_class);
        if egraph.enodes[opcode_node].0.trim_matches('"') != op_name {
            return None;
        }
        return Some(
            walk_ilist_simple(egraph, &children[1])
                .into_iter()
                .map(|input| unwrap_fusion_start(egraph, input))
                .collect(),
        );
    }
    if !egraph.enodes[kind].0.contains("FusionEnd") {
        return None;
    }
    let fe_inputs = walk_ilist_simple(egraph, &children[1]);
    let elem = *fe_inputs.first()?;
    let (elem_label, elem_children) = &egraph.enodes[elem];
    if elem_label != "Op" || elem_children.len() < 2 {
        return None;
    }
    let elem_kind = resolve_first_node(egraph, &elem_children[0]);
    if !egraph.enodes[elem_kind].0.contains("CudaBinaryElementwise") {
        return None;
    }
    let opcode_class = egraph.enodes[elem_kind].1.first()?;
    let opcode_node = resolve_first_node(egraph, opcode_class);
    if egraph.enodes[opcode_node].0.trim_matches('"') != op_name {
        return None;
    }
    Some(
        walk_ilist_simple(egraph, &elem_children[1])
            .into_iter()
            .map(|input| unwrap_fusion_start(egraph, input))
            .collect(),
    )
}

fn unwrap_fusion_start<'a>(egraph: &'a SerializedEGraph, node: &'a NodeId) -> &'a NodeId {
    let (label, children) = &egraph.enodes[node];
    if label != "Op" || children.len() < 2 {
        return node;
    }
    let kind = resolve_first_node(egraph, &children[0]);
    if !egraph.enodes[kind].0.contains("FusionStart") {
        return node;
    }
    walk_ilist_simple(egraph, &children[1])
        .first()
        .copied()
        .unwrap_or(node)
}
