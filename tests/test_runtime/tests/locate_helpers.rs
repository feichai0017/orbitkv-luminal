//! UNIT TESTS for the ID-FREE e-class locator (`luminal::test_support::locate`).
//!
//! Every assertion here is about the CONTRACT the module exists to keep:
//! roots are reached by walking the boundary term, classes are reached by
//! DESCRIPTION, e-nodes are named by an id-free SIGNATURE, and forcing a
//! signature into the genome elects exactly that e-node. Nothing below
//! spells a serialized `ClassId`, and nothing below hardcodes a recorder
//! counter (`natout{K}`, `nat{K}`) — the two things that make a marker-board
//! test rot on an unrelated estate edit.

use std::collections::BTreeSet;

use luminal::graph::Graph;
use luminal::layout_ir::{ExtractedNode, Provenance};
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use luminal::test_support::locate::Locator;

/// The election preferences the marker boards run with.
const PIN: &[&str] = &[
    "LayoutTensorOpCublasLtAccumulateBias",
    "LayoutTensorOpCublasLtBias",
    "LayoutTensorOpCublasLtAccumulate",
    "LayoutTensorOpCublasLt",
    "LayoutTensorOpIndexMapApplyViewGeneric",
];

/// The recorder's own m=1 matmul, bound and saturated — a program whose
/// boundary lets are `nat{K}` / `natout{K}` shaped.
fn matmul_egraph() -> EGraph {
    let mut cx = Graph::new();
    let x = cx.tensor((1usize, 4usize));
    let w = cx.tensor((4usize, 3usize));
    let _out = x.matmul(w).output();
    let text = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;
    test_runtime::serialize_fixture(&text)
}

/// `x[4,8] @ w[8,3] + b[3]`, spelled as `luminal_nn::Linear` with bias
/// spells it — the graph whose cuBLASLt bias form the LeftMajor premise
/// is stated over.
fn linear_with_bias_egraph() -> EGraph {
    let mut cx = Graph::new();
    let x = cx.tensor((4usize, 8usize));
    let w = cx.tensor((8usize, 3usize));
    let b = cx.tensor(3usize);
    let product = x.matmul(w);
    let dims = product.dims();
    let biased = product + b.expand_lhs(&dims[..dims.len() - 1]);
    let _out = biased.output();
    let text = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;
    test_runtime::serialize_fixture(&text)
}

/// The layout-tensor class behind a boundary buffer-tensor slot.
fn layout_tensor_of(loc: &Locator<'_>, buffer_tensor: &ClassId) -> ClassId {
    loc.child(
        loc.node_in(buffer_tensor, "BufferTensorLit")
            .expect("a boundary slot is a BufferTensorLit"),
        0,
    )
}

// ===========================================================================
// (a) ROOTS
// ===========================================================================

/// The boundary is REACHED, not named: the output spine walk finds one
/// slot, the input spine finds two, and each slot is a real buffer tensor.
#[test]
fn roots_are_found_by_walking_the_boundary_term() {
    let egraph = matmul_egraph();
    let loc = Locator::new(&egraph);

    let outputs = loc.outputs();
    let inputs = loc.inputs();
    println!(
        "roots: {} output slot(s), {} input slot(s)",
        outputs.len(),
        inputs.len()
    );
    assert_eq!(outputs.len(), 1, "one bound output slot");
    assert_eq!(inputs.len(), 2, "x and w cross the boundary");
    for slot in outputs.iter().chain(inputs.iter()) {
        assert!(
            loc.view(slot).has_op("BufferTensorLit"),
            "every boundary slot is a BufferTensorLit: {}",
            loc.class_digest(slot)
        );
    }
    // Slots are DISTINCT classes — a spine walk that fell over would
    // repeat one.
    let distinct: BTreeSet<&ClassId> = inputs.iter().collect();
    assert_eq!(distinct.len(), inputs.len(), "input slots are distinct");
}

/// The recorder's output STEM is read off the e-graph, never scraped out
/// of the program text and never hardcoded. The proof it is the real stem:
/// the boundary lets it names resolve back through `let_class`.
#[test]
fn output_stems_come_from_the_egraph_and_round_trip() {
    let egraph = matmul_egraph();
    let loc = Locator::new(&egraph);

    let stems = loc.output_stems();
    println!("stems: {stems:?}");
    assert_eq!(stems.len(), loc.outputs().len(), "one stem per slot");
    for (stem, slot) in stems.iter().zip(loc.outputs()) {
        assert!(
            stem.starts_with("natout"),
            "the recorder's output stems are natout-shaped, got {stem}"
        );
        // The COUNTER in `natout{K}` is deliberately not asserted: it is
        // the output node's index and shifts whenever recorder numbering
        // does. What must hold is that the stem names the real lets.
        let layout_tensor = loc
            .let_class(&format!("{stem}_layout_tensor"))
            .unwrap_or_else(|| panic!("{stem}_layout_tensor is bound"));
        assert_eq!(
            layout_tensor,
            layout_tensor_of(&loc, &slot),
            "the stem's layout-tensor let is the slot's own layout tensor"
        );
        assert!(
            loc.let_class(&format!("{stem}_buffer_id")).is_some(),
            "{stem}_buffer_id is bound"
        );
    }
}

/// `let_class` finds a class by the name the SOURCE gave it, and
/// `input_class` finds a boundary input by the name its `LogicalIdLit`
/// carries.
#[test]
fn let_name_and_input_name_lookup() {
    let egraph = matmul_egraph();
    let loc = Locator::new(&egraph);

    // Round-trip: every boundary slot's let name resolves back to it.
    for slot in loc.inputs() {
        let name = loc
            .let_name(&slot)
            .expect("the recorder names every boundary buffer tensor");
        assert_eq!(loc.let_class(&name), Some(slot.clone()), "let round-trip");
    }
    assert_eq!(loc.let_class("no_such_let_name_anywhere"), None);

    // Input names come off the logical term, unquoted.
    let names: Vec<String> = loc
        .inputs()
        .iter()
        .map(|slot| {
            let layout_tensor = layout_tensor_of(&loc, slot);
            let logical = loc.child(
                loc.node_in(&layout_tensor, "LayoutTensorLit")
                    .expect("a buffer tensor names a layout tensor"),
                0,
            );
            loc.logical_name(&logical).expect("a named boundary input")
        })
        .collect();
    println!("input names: {names:?}");
    for name in &names {
        assert_eq!(
            loc.input_class(name).map(|c| loc.logical_name(&c)),
            Some(Some(name.clone())),
            "input_class round-trips the name {name}"
        );
    }
    assert_eq!(loc.input_class("not_an_input"), None);
}

// ===========================================================================
// (b) DESCRIPTION-BASED SEARCH
// ===========================================================================

/// THE BIAS-PREMISE WALK WITH NO IDS. The ad-hoc version
/// (`crates/luminal_cuda_lite/tests/cublaslt_bias_premise.rs`) is
/// `d_layout_class`: root `LayoutTensorOpCublasLt*` e-node → child 3
/// `CublasLtOutputDDescriptor` → child 1 `LayoutTensorLit` → child 1 = the
/// D LAYOUT class. Here the root is reached by DESCRIPTION and every hop
/// is a path step, and the premise itself — that the bias form's D holds
/// the left-major spelling — is checked on the class the walk lands on.
#[test]
fn find_class_reaches_the_bias_d_layout_class() {
    let egraph = linear_with_bias_egraph();
    let loc = Locator::new(&egraph);

    let bias_classes = loc.find_class(|class| class.has_op("LayoutTensorOpCublasLtBias"));
    println!("classes holding a bias form: {}", bias_classes.len());
    assert!(
        !bias_classes.is_empty(),
        "the per-feature bias must mint LayoutTensorOpCublasLtBias"
    );

    let mut checked = 0usize;
    for class in &bias_classes {
        for op in loc.view(class).nodes_with_op("LayoutTensorOpCublasLtBias") {
            // slot 3 is the D descriptor on every cuBLASLt form
            let d_descriptor = loc.child(op, 3);
            let descriptor = loc
                .node_in(&d_descriptor, "CublasLtOutputDDescriptor")
                .expect("slot 3 holds the D descriptor");
            let layout_tensor = loc.child(descriptor, 1);
            let lit = loc
                .node_in(&layout_tensor, "LayoutTensorLit")
                .expect("the D descriptor names a layout tensor");
            let d_layout = loc.child(lit, 1);
            println!("  D layout class: {}", loc.class_digest(&d_layout));
            assert!(
                loc.view(&d_layout)
                    .has_op("LeftMajorContiguousElementLayoutLit"),
                "THE LEFT-MAJOR D PREMISE: a minted bias form's D layout class must hold the \
                 left-major spelling, got {}",
                loc.class_digest(&d_layout)
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "at least one bias form was walked");
}

/// `find_one_class` is the sharp form: it panics rather than pick between
/// two matches, because picking would be the id-order tie-break this
/// module exists to remove.
#[test]
#[should_panic(expected = "description matched")]
fn find_one_class_refuses_an_unsharp_description() {
    let egraph = matmul_egraph();
    let loc = Locator::new(&egraph);
    // Every layout-tensor class matches — deliberately unsharp.
    loc.find_one_class(|class| class.has_op("LayoutTensorLit"));
}

// ===========================================================================
// (c) SIGNATURES
// ===========================================================================

/// THE SERIALIZED CLASS-ID SHAPE, `[A-Za-z_][A-Za-z0-9_]*-[0-9]+`, scanned
/// by hand rather than by `regex` so this runtime crate takes on no new
/// dependency for one test. Returns the offending substring.
///
/// It matches `Layout-2216` and `i64-4` (what egglog's serializer mints)
/// while leaving negative literals (`-1`, no identifier before the hyphen)
/// and the estate's kebab-case function names (`shape-of`,
/// `expr-list-nth-from-end`, no digits after the hyphen) alone.
fn class_id_shaped(text: &str) -> Option<String> {
    let bytes: Vec<char> = text.chars().collect();
    for (i, c) in bytes.iter().enumerate() {
        if *c != '-' {
            continue;
        }
        // digits after the hyphen
        let mut end = i + 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == i + 1 {
            continue;
        }
        // an identifier before it
        let mut start = i;
        while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == '_') {
            start -= 1;
        }
        if start < i && (bytes[start].is_ascii_alphabetic() || bytes[start] == '_') {
            return Some(bytes[start..end].iter().collect());
        }
    }
    None
}

/// NO SIGNATURE MAY SPELL A CLASS ID. Serialized ids look like
/// `Layout-2216` / `i64-4` — an identifier, a hyphen, digits. The scan
/// below runs over every candidate signature in the whole producer index
/// of two different programs.
#[test]
fn signatures_never_spell_a_class_id() {
    // The shape really is what serialized ids look like.
    let egraph = matmul_egraph();
    let sample = egraph
        .nodes
        .values()
        .next()
        .expect("a saturated program has nodes")
        .eclass
        .to_string();
    assert!(
        class_id_shaped(&sample).is_some(),
        "the scan must actually match a serialized class id, got {sample}"
    );

    for (label, egraph) in [
        ("matmul", matmul_egraph()),
        ("linear+bias", linear_with_bias_egraph()),
    ] {
        let loc = Locator::new(&egraph);
        let index = loc.producer_index(test_runtime::matchers());
        let mut checked = 0usize;
        for class in index.keys() {
            for candidate in loc.candidates(&index, class) {
                if let Some(id) = class_id_shaped(&candidate.signature) {
                    panic!(
                        "{label}: signature of {} spells the class id `{id}`:\n{}",
                        candidate.constructor, candidate.signature
                    );
                }
                checked += 1;
            }
            if let Some(id) = class_id_shaped(&loc.class_digest(class)) {
                panic!(
                    "{label}: class digest spells the class id `{id}`: {}",
                    loc.class_digest(class)
                );
            }
        }
        println!("{label}: {checked} candidate signature(s) are id-free");
        assert!(checked > 0, "{label}: the index is not empty");
    }
}

/// A signature separates the spellings that differ and identifies the ones
/// that do not: the m=1 boundary class carries four cuBLASLt readings, and
/// the four signatures are four distinct strings.
#[test]
fn signatures_separate_the_readings_of_one_class() {
    let egraph = matmul_egraph();
    let loc = Locator::new(&egraph);
    let index = loc.producer_index(test_runtime::matchers());
    let out = layout_tensor_of(&loc, &loc.outputs()[0]);

    let cublaslt: Vec<_> = loc
        .candidates(&index, &out)
        .into_iter()
        .filter(|c| c.constructor.starts_with("LayoutTensorOpCublasLt"))
        .collect();
    println!("{} cuBLASLt candidate(s):", cublaslt.len());
    for candidate in &cublaslt {
        println!("  {}", candidate.describe_short());
    }
    assert!(cublaslt.len() >= 2, "the degenerate frame reads many ways");
    let distinct: BTreeSet<&str> = cublaslt.iter().map(|c| c.signature.as_str()).collect();
    assert_eq!(
        distinct.len(),
        cublaslt.len(),
        "same-constructor candidates must be told apart by their signatures"
    );

    // ...and the constructor alone cannot: this is exactly the tie a
    // cost/label assertion used to decide by id order.
    let same_constructor = loc.candidates(&index, &out);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // the panic below is the assertion
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loc.assert_unique_candidate(&index, &out, "LayoutTensorOpCublasLt")
    }));
    std::panic::set_hook(previous);
    assert!(
        err.is_err(),
        "assert_unique_candidate must refuse a constructor with {} candidates",
        same_constructor.len()
    );
}

// ===========================================================================
// (d) ELECTION
// ===========================================================================

/// FORCING A SIGNATURE ELECTS THAT E-NODE: the extracted plan's op carries
/// the very source e-node whose signature was asked for.
#[test]
fn elect_by_signature_elects_that_signature() {
    let egraph = matmul_egraph();
    let loc = Locator::new(&egraph);
    let index = loc.producer_index(test_runtime::matchers());
    let base = test_runtime::genome_preferring(&egraph, PIN);
    let out = layout_tensor_of(&loc, &loc.outputs()[0]);

    let candidates: Vec<_> = loc
        .candidates(&index, &out)
        .into_iter()
        .filter(|c| c.constructor.starts_with("LayoutTensorOpCublasLt"))
        .collect();
    assert!(candidates.len() >= 2, "several readings to choose between");

    for candidate in &candidates {
        let genome = loc.elect_by_signature(&index, &base, &out, &candidate.signature);
        let graph = luminal::extractor::extract_layout_ir_with_genome_and_matchers(
            &egraph,
            &genome,
            test_runtime::matchers(),
        )
        .expect("forced extraction runs")
        .expect("forced extraction reaches the boundary");

        let elected: Vec<String> = graph
            .dag
            .node_weights()
            .filter_map(|node| match node {
                ExtractedNode::LayoutOp(op) => match &op.provenance {
                    Provenance::Extracted { source_enode, .. } => Some(loc.signature(source_enode)),
                    Provenance::Synthesized { .. } => None,
                },
                _ => None,
            })
            .collect();
        assert!(
            elected.contains(&candidate.signature),
            "the plan must carry the e-node whose signature was elected\nwanted:\n{}\ngot:\n{}",
            candidate.signature,
            elected.join("\n")
        );
    }
    println!(
        "{} forced election(s) each elected their own e-node",
        candidates.len()
    );
}

/// `elect_each` is the per-e-node driver: one genome per candidate, each
/// electing its own e-node, and never the same one twice.
#[test]
fn elect_each_yields_one_genome_per_candidate() {
    let egraph = matmul_egraph();
    let loc = Locator::new(&egraph);
    let index = loc.producer_index(test_runtime::matchers());
    let base = test_runtime::genome_preferring(&egraph, PIN);
    let out = layout_tensor_of(&loc, &loc.outputs()[0]);

    let elections = loc.elect_each(&index, &base, &out);
    println!("elect_each produced {} election(s)", elections.len());
    assert_eq!(
        elections.len(),
        loc.candidates(&index, &out).len(),
        "every candidate of a produced class is electable somewhere"
    );
    for (candidate, genome) in &elections {
        let choice = genome
            .choices
            .get(&out)
            .expect("the forced class carries a row");
        assert_eq!(
            choice.enode, candidate.enode,
            "the genome's row for the class is the candidate's own e-node"
        );
    }
}
