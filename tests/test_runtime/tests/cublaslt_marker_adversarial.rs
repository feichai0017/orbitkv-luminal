//! ADVERSARIAL fixtures for the MARKER design.
//!
//! Fixture 4: the round-1 mirror FATAL — sliced sources (storage k=7,
//! product k=3). The marker's logical-shape unification must REFUSE the
//! site; the decomposed route must survive.
//!
//! Fixture 7: degenerate extents (m=1, the extent-1 pointing weld, and
//! the all-ones corner) — must not panic; readings stay bounded and
//! every elected spec is sound.

use std::collections::BTreeSet;

use luminal::layout_ir::ExtractedNode;
use luminal::prelude::egraph_serialize::{ClassId, Node};
use luminal::test_support::locate::Locator;
use test_runtime::cublaslt_marker::{CublasLt, LtMatmulSpec};

const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

const PIN: &[&str] = &[
    "LayoutTensorOpCublasLtAccumulateBias",
    "LayoutTensorOpCublasLtBias",
    "LayoutTensorOpCublasLtAccumulate",
    "LayoutTensorOpCublasLt",
    // ROUND 10 (view admission in ELECTION): the sibling site's result is
    // routed to the recorder's boundary value by a transpose VIEW; prefer
    // the view op over materialize/copy wherever both produce a class.
    "LayoutTensorOpIndexMapApplyViewGeneric",
];

/// Parameterized handwritten 2D matmul skeleton (ported from the round-1
/// adversarial estate). Geometry args are raw IntExpr text so storage
/// extents can deliberately mismatch the product coordinates.
fn matmul_2d(
    m: &str,
    n: &str,
    k_prod: &str,
    a_rows: &str,
    a_cols: &str,
    b_rows: &str,
    b_cols: &str,
) -> String {
    format!(
        r#"(let a_shape (ShapeLit (IntExprCons {a_rows} (IntExprCons {a_cols} (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons {b_rows} (IntExprCons {b_cols} (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprCons {k_prod} (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output
  (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
"#
    )
}

fn cublaslt_ops(graph: &luminal::layout_ir::ExtractedGraph) -> Vec<CublasLt> {
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => {
                (*op.op).as_any().downcast_ref::<CublasLt>().cloned()
            }
            _ => None,
        })
        .collect()
}

// ===========================================================================
// Fixture 4 — sliced-source counterexample. product (m n k) = (2, 4, 3),
// A stored [2,7] (col slice), B stored [7,4] (row slice). Round-1 mirror
// silently minted k=7. The marker must refuse: storage k=7 does not unify
// with product k=3.
// ===========================================================================
#[test]
fn fixture4_sliced_source_refused() {
    let fx = format!(
        "{}{SCHEDULE}\n",
        matmul_2d(
            "(IntLit 2)",
            "(IntLit 4)",
            "(IntLit 3)",
            "(IntLit 2)",
            "(IntLit 7)",
            "(IntLit 7)",
            "(IntLit 4)",
        )
    );
    let serialized = test_runtime::serialize_fixture(&fx);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    let ops = serialized
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .count();
    println!("ADVERSARIAL sliced-source: {sites} sites, {ops} cublaslt op enodes");
    assert_eq!(sites, 0, "logical-shape unification refuses the marker");
    assert_eq!(ops, 0, "no site, no op");

    // The decomposed route survives: extraction still reaches the boundary
    // even with the genome asking for cublaslt everywhere.
    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let ops = cublaslt_ops(&graph);
    assert!(ops.is_empty(), "no cublaslt candidate exists to elect");
    let computes = graph
        .dag
        .node_weights()
        .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
        .count();
    println!("  decomposed plan has {computes} compute nodes");
    assert!(computes > 0, "decomposed route reaches the boundary");
}

// ===========================================================================
// Fixture 7 — degenerate extents must not panic; readings stay bounded.
// ===========================================================================

// ---------------------------------------------------------------------------
// The m=1 sweep's per-e-node oracle: what does THIS cuBLASLt e-node's own
// descriptor term say the call frame is? Reads the e-graph only, through
// `luminal::test_support::locate` — no class id is spelled anywhere.
// ---------------------------------------------------------------------------

/// The reading one cuBLASLt op e-node commits to.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Reading {
    trans_a: bool,
    trans_b: bool,
    lda: i64,
    ldb: i64,
    ldd: i64,
    /// Descriptor A's / B's layout tensor — opaque in-process handles used
    /// to find this e-node's kernel in the elected plan, never asserted on.
    a_lt: ClassId,
    b_lt: ClassId,
}

/// The storage extents of a logical tensor, off its `shape-of` row.
fn storage_dims(loc: &Locator<'_>, logical: &ClassId) -> Vec<i64> {
    let shape = loc
        .find_one_class(|c| {
            c.nodes_with_op("shape-of")
                .iter()
                .any(|n| loc.try_child(n, 0).as_ref() == Some(logical))
        })
        .clone();
    let lit = loc
        .node_in(&shape, "ShapeLit")
        .expect("a shape class holds a ShapeLit");
    loc.walk_cons(&loc.child(lit, 0), "IntExprCons", "IntExprNil")
        .iter()
        .map(|dim| {
            loc.int_literal(dim)
                .expect("this fixture's extents are all literal")
        })
        .collect()
}

/// The operand descriptor at `slot` of a cuBLASLt op e-node, and whether
/// its operation constructor is the TRANSPOSED one.
fn descriptor<'e>(
    loc: &Locator<'e>,
    op: &Node,
    slot: usize,
    constructor: &str,
) -> (&'e Node, bool) {
    let class = loc.child(op, slot);
    let desc = loc
        .node_in(&class, constructor)
        .unwrap_or_else(|| panic!("slot {slot} holds a {constructor}"));
    let operation = loc.child(desc, 2);
    let view = loc.view(&operation);
    assert!(
        view.has_op("CublasLtOperationN") || view.has_op("CublasLtOperationT"),
        "a descriptor's operation slot holds one of the two operation constructors, got {}",
        view.signature()
    );
    (desc, view.has_op("CublasLtOperationT"))
}

/// THIS e-node's own reading, derived from its own site triple and its own
/// descriptors: the operation constructors fix trans_a/trans_b, and the
/// COL-view row counts fix the lds.
///
/// The general oracle (`per_enode_election_sweep`'s `reading_ld` on the
/// round-3 board) additionally lets a PADDED layout override the row count
/// with its own pitch. That override cannot fire here, and the assertion
/// below is the reason: every operand of this fixture is described by a
/// CONTIGUOUS element layout, whose pitch is by construction one of its own
/// extents. If an estate change ever makes an operand strided, this fails
/// on the precondition rather than on a confusing ld mismatch.
fn read_enode(loc: &Locator<'_>, op: &Node) -> Reading {
    let site_class = loc.child(op, 0);
    let site = loc
        .node_in(&site_class, "CublasLtLogicalMatmulSite")
        .expect("slot 0 holds the site");
    let a_storage = storage_dims(loc, &loc.child(site, 0));
    let d_storage = storage_dims(loc, &loc.child(site, 2));
    assert_eq!(a_storage.len(), 2, "2-D fixture");
    assert_eq!(d_storage.len(), 2, "2-D fixture");
    // Unswapped frame (round 10): the call's m/n are the SITE's own out
    // extents, and k is the a-storage extent that is not m.
    let (lm, ln) = (d_storage[0], d_storage[1]);
    let lk = if a_storage[0] == lm {
        a_storage[1]
    } else {
        a_storage[0]
    };

    let (a_desc, trans_a) = descriptor(loc, op, 1, "CublasLtOperandADescriptor");
    let (b_desc, trans_b) = descriptor(loc, op, 2, "CublasLtOperandBDescriptor");
    let a_lt = loc.child(a_desc, 1);
    let b_lt = loc.child(b_desc, 1);
    for (role, lt) in [("A", &a_lt), ("B", &b_lt)] {
        let layout = loc.child(
            loc.node_in(lt, "LayoutTensorLit")
                .expect("a descriptor names a layout tensor"),
            1,
        );
        let view = loc.view(&layout);
        assert!(
            view.has_op("RightMajorContiguousElementLayoutLit")
                || view.has_op("LeftMajorContiguousElementLayoutLit"),
            "operand {role} is not described by a contiguous element layout, so the padded-pitch \
             override of the general ld oracle may apply and this sweep's rows rule is no longer \
             sufficient: {}",
            view.signature()
        );
    }

    Reading {
        trans_a,
        trans_b,
        lda: if trans_a { lk } else { lm },
        ldb: if trans_b { ln } else { lk },
        ldd: lm,
        a_lt,
        b_lt,
    }
}

/// The cuBLASLt COL-order leading-dimension clamps on literals: lda >=
/// rows(A), ldb >= rows(B), ldd >= m, ldc == ldd. Every elected frame must
/// pass them, whichever one the sweep forced.
fn assert_call_sound(tag: &str, spec: &LtMatmulSpec) {
    let (m, n, k) = spec.mnk_lits();
    let rows_a = if spec.trans_a { k } else { m };
    let rows_b = if spec.trans_b { n } else { k };
    for (name, ld, rows) in [
        ("lda", spec.lda.literal(), rows_a),
        ("ldb", spec.ldb.literal(), rows_b),
        ("ldd", spec.ldd.literal(), m),
    ] {
        let ld = ld.unwrap_or_else(|| panic!("{tag}: {name} is symbolic on a literal fixture"));
        assert!(
            ld >= rows,
            "{tag}: cuBLASLt COL clamp violated — {name}={ld} < rows={rows}"
        );
    }
    assert_eq!(
        spec.ldc.literal(),
        spec.ldd.literal(),
        "{tag}: C rides the D layout, so ldc must equal ldd"
    );
}

/// m=1 (live-recorder geometry x[1,4] @ w[4,3]).
///
/// ORIGINAL INTENT (round 1, re-pinned round 10). The degenerate matmul
/// must not panic: saturation stays bounded (the original site plus the
/// transpose-sandwich sibling), and the elected call frame is sound. Round
/// 10 additionally pinned the SPEC FIELDS to one frame — `!trans_a`,
/// `trans_b`, `lda=1, ldb=3, ldd=1` — on the reasoning that "at m=1 the
/// original site presents a legal unswapped call directly ... and WINS THE
/// ELECTION over the sibling".
///
/// WHY THOSE FIELD PINS WERE RETIRED (ruling 2026-09-02, Austin: tests must
/// never pin a cost/label TIE by accident). The boundary output class
/// carries FOUR cuBLASLt readings of the same bytes — {A op N, A op T} x
/// {B op N, B op T} — same constructor, same cost, same label. Which one an
/// untouched election returns is decided by e-node ORDER inside the
/// producer index (`(name, enode.to_string(), output_index)`), which is
/// creation-order and guaranteed by nothing. All four are sound spellings
/// of the same GEMV, so "the original site wins" was never a compiler
/// promise to assert.
///
/// WHAT IS ASSERTED NOW — the per-e-node pattern, as
/// `per_enode_election_sweep` does it. The site is located BY DESCRIPTION
/// (the site whose two logical operands are the boundary inputs), its
/// cuBLASLt candidates are listed BY SIGNATURE, and each candidate in turn
/// is FORCED into the genome; the parsed spec must then be THAT
/// candidate's own descriptors' reading — its own operation constructors,
/// its own COL row counts — and must pass the COL clamps. The round-10
/// numbers survive as the reading of the (A op N, B op T) candidate, now
/// checked alongside the other three rather than in place of them.
#[test]
fn fixture7_m1_degenerate_no_panic() {
    use luminal::graph::Graph;
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((1usize, 4usize));
        let w = cx.tensor((4usize, 3usize));
        let _out = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    // serialize_fixture panics if saturation panics.
    let serialized = test_runtime::serialize_fixture(&text);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    println!(
        "fixture7 m=1: {} nodes, {sites} site(s)",
        serialized.nodes.len()
    );
    // ROUND-10 RE-PIN (was 1): original + transpose-sandwich sibling.
    assert_eq!(
        sites, 2,
        "the degenerate matmul carries original + sibling sites"
    );

    let loc = Locator::new(&serialized);

    // DESCRIBE THE SITE, don't name it: the ORIGINAL site is the one whose
    // two logical operands are the boundary inputs themselves (the
    // sibling's operands are index-map applies of them).
    let boundary_logicals: BTreeSet<ClassId> = loc
        .inputs()
        .iter()
        .map(|buffer_tensor| {
            let lit = loc
                .node_in(buffer_tensor, "BufferTensorLit")
                .expect("a boundary slot is a BufferTensorLit");
            let layout_tensor = loc.child(lit, 0);
            let lt = loc
                .node_in(&layout_tensor, "LayoutTensorLit")
                .expect("a buffer tensor names a layout tensor");
            loc.child(lt, 0)
        })
        .collect();
    assert_eq!(boundary_logicals.len(), 2, "x and w cross the boundary");
    let original_site = loc.find_one_class(|class| {
        class.node("CublasLtLogicalMatmulSite").is_some_and(|site| {
            boundary_logicals.contains(&loc.child(site, 0))
                && boundary_logicals.contains(&loc.child(site, 1))
        })
    });

    // The class the election actually decides: the boundary output's own
    // layout tensor.
    let out_buffer_tensor = loc
        .outputs()
        .first()
        .cloned()
        .expect("one bound output slot");
    let out_layout_tensor = loc.child(
        loc.node_in(&out_buffer_tensor, "BufferTensorLit")
            .expect("the output slot is a BufferTensorLit"),
        0,
    );
    println!(
        "  output slot {:?} -> layout tensor {}",
        loc.output_stems(),
        loc.class_digest(&out_layout_tensor)
    );

    let index = loc.producer_index(test_runtime::matchers());
    let base = test_runtime::genome_preferring(&serialized, PIN);
    let candidates: Vec<_> = loc
        .candidates(&index, &out_layout_tensor)
        .into_iter()
        .filter(|candidate| {
            candidate.constructor.starts_with("LayoutTensorOpCublasLt")
                && loc.child(&serialized.nodes[&candidate.enode], 0) == original_site
        })
        .collect();
    println!(
        "  {} cuBLASLt candidate(s) on the boundary class",
        candidates.len()
    );
    assert!(
        candidates.len() >= 2,
        "the degenerate frame is read several ways; a single candidate would mean the \
         multiplicity this sweep exists for has gone"
    );

    let mut swept = 0usize;
    for candidate in &candidates {
        let want = read_enode(&loc, &serialized.nodes[&candidate.enode]);
        let genome =
            loc.elect_by_signature(&index, &base, &out_layout_tensor, &candidate.signature);
        let graph = luminal::extractor::extract_layout_ir_with_genome_and_matchers(
            &serialized,
            &genome,
            test_runtime::matchers(),
        )
        .expect("forced extraction runs")
        .expect("forced extraction reaches the boundary");
        let ops = cublaslt_ops(&graph);
        assert_eq!(ops.len(), 1, "one kernel per forced election");
        let spec = ops[0].spec.as_ref().expect("spec parses");
        println!(
            "  elected {} => m={} n={} k={} trans_a={} trans_b={} lda={} ldb={} ldd={}",
            candidate.describe_short(),
            spec.m,
            spec.n,
            spec.k,
            spec.trans_a,
            spec.trans_b,
            spec.lda,
            spec.ldb,
            spec.ldd
        );

        // The frame's SHAPE is the same GEMV whichever reading is elected.
        assert_eq!(spec.mnk_lits(), (1, 3, 4));
        // ...and every field is THIS e-node's own reading, never a sibling's.
        assert_eq!(
            (
                spec.desc_a_layout_tensor.clone(),
                spec.desc_b_layout_tensor.clone()
            ),
            (want.a_lt.clone(), want.b_lt.clone()),
            "the elected kernel must be the FORCED e-node's, not a sibling reading"
        );
        assert_eq!(
            spec.trans_a, want.trans_a,
            "trans_a is THIS e-node's reading"
        );
        assert_eq!(
            spec.trans_b, want.trans_b,
            "trans_b is THIS e-node's reading"
        );
        assert_eq!(
            spec.lda.literal(),
            Some(want.lda),
            "lda is THIS e-node's reading"
        );
        assert_eq!(
            spec.ldb.literal(),
            Some(want.ldb),
            "ldb is THIS e-node's reading"
        );
        assert_eq!(
            spec.ldd.literal(),
            Some(want.ldd),
            "ldd is THIS e-node's reading"
        );
        assert_call_sound("fixture7 m=1", spec);
        swept += 1;
    }
    assert_eq!(
        swept,
        candidates.len(),
        "every cuBLASLt candidate of the boundary class was forced and checked"
    );

    // The round-10 numbers, no longer as "the winner" but as one member of
    // the swept set: the direct call A = x[1,4] COL 1x4 ld 1 op N,
    // B = w[4,3] COL 3x4 ld 3 op T.
    let direct = candidates
        .iter()
        .map(|candidate| read_enode(&loc, &serialized.nodes[&candidate.enode]))
        .find(|reading| !reading.trans_a && reading.trans_b)
        .expect("the direct (op N, op T) frame is still one of the readings");
    assert_eq!((direct.lda, direct.ldb, direct.ldd), (1, 3, 1));
}

/// The extent-1 pointing-weld geometry: m=1, k=1, n=3 (A [1,1], B [1,3]).
/// Round 1 aborted here; round 2 forced a canonical single reading to
/// protect its :no-merge functions. Descriptor TERMS make the structural
/// same-extent weld legal multiplicity: both readings of the same bytes
/// mint, the election picks either, and the parser's cross-checks hold.
#[test]
fn fixture7_extent1_weld_single_canonical_reading() {
    let fx = format!(
        "{}{SCHEDULE}\n",
        matmul_2d(
            "(IntLit 1)",
            "(IntLit 3)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 3)",
        )
    );
    let serialized = test_runtime::serialize_fixture(&fx);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    let a_readings = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtOperandADescriptor")
        .count();
    println!("fixture7 weld: {sites} site(s), {a_readings} A reading(s)");
    // ROUND-10 RE-PIN (was 1): original + transpose-sandwich sibling(s).
    // The weld can congruence several sibling spellings into the same
    // term; what matters is boundedness and that every elected spec is
    // sound (checked below).
    assert!(
        (2..=4).contains(&sites),
        "bounded sites despite the coordinate weld: {sites}"
    );
    assert!(a_readings >= 1, "the welded bytes are read at least once");

    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let ops = cublaslt_ops(&graph);
    assert_eq!(ops.len(), 1, "election picks one reading");
    let spec = ops[0]
        .spec
        .as_ref()
        .expect("spec parses and cross-validates");
    println!(
        "  m={} n={} k={} trans_a={} trans_b={}",
        spec.m, spec.n, spec.k, spec.trans_a, spec.trans_b
    );
    // ROUND-10 RE-PIN (was (3,1,1)): same direct-call election as the m=1
    // fixture above — at m=1 the original site is COL-presenting and wins.
    assert_eq!(spec.mnk_lits(), (1, 3, 1));
}

/// All-ones corner m=n=k=1: LogicalMul commutativity can present the role
/// swap and coordinate welds make every map pattern congruent — observe
/// what the marker does. Duplicate SITES are acceptable (each is a sound
/// 1x1 call); a panic or a wrong descriptor is not.
#[test]
fn fixture7_all_ones_corner_observed() {
    let fx = format!(
        "{}{SCHEDULE}\n",
        matmul_2d(
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
        )
    );
    let serialized = test_runtime::serialize_fixture(&fx);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    println!("fixture7 all-ones: {sites} site(s)");
    assert!(sites >= 1, "the 1x1x1 matmul still marks");

    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let ops = cublaslt_ops(&graph);
    assert_eq!(ops.len(), 1, "one op elected at the boundary");
    let spec = ops[0].spec.as_ref().expect("spec parses");
    assert_eq!(spec.mnk_lits(), (1, 1, 1));
    assert_eq!(spec.lda, 1);
    assert_eq!(spec.ldb, 1);
    assert_eq!(spec.ldd, 1);
}
