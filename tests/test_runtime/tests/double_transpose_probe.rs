//! DOUBLE-TRANSPOSE COLLAPSE PROBE (round-10 termination prerequisite,
//! Austin 2026-08-25: "we'll have to make sure that this does cons /
//! terminate. I am relying on that").
//!
//! The transpose-rewrite design terminates iff apply(apply(x, T), T)
//! rejoins x's class, so the rewrite's re-fire on its own output
//! hash-conses shut. Static finding: every LogicalIndexMapApply premise in
//! the preamble matches a SINGLE apply and concludes layout facts only —
//! no rule unions a nested apply back onto its grandparent. This probe
//! pins the consequence empirically, and doubles as the RED baseline the
//! collapse rule must flip.
const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

#[test]
fn double_transpose_does_not_rejoin_without_a_rule() {
    let fx = format!(
        r#"(let x_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let t_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 2) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") x_shape (F32)))
(let x_lt (LayoutTensorLit x (RightMajorContiguousElementLayoutLit x_shape (bits-of (F32)))))
(let t1 (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar t_shape 0)
    (IntExprCons (CoordVar t_shape 1) (IntExprNil))) x_shape)
  t_shape))
(let t2 (LogicalIndexMapApply t1
  (IndexMapLit (IntExprCons (CoordVar x_shape 0)
    (IntExprCons (CoordVar x_shape 1) (IntExprNil))) t_shape)
  x_shape))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    // Anchor the three logical classes structurally: x is the input; t1 is
    // the apply whose child is x's class; t2 is the apply whose child is
    // t1's class.
    let x_class = s
        .nodes
        .values()
        .find(|n| n.op == "LogicalTensorInputLit")
        .expect("input exists")
        .eclass
        .clone();
    let applies: Vec<_> = s
        .nodes
        .values()
        .filter(|n| n.op == "LogicalIndexMapApply")
        .collect();
    assert!(
        applies.len() >= 2,
        "both applies must survive to the egraph"
    );
    let t1_class = applies
        .iter()
        .find(|n| {
            n.children
                .first()
                .and_then(|c| s.nodes.get(c))
                .map(|c| c.eclass == x_class)
                .unwrap_or(false)
        })
        .expect("t1 anchored at x")
        .eclass
        .clone();
    let t2 = applies
        .iter()
        .find(|n| {
            n.children
                .first()
                .and_then(|c| s.nodes.get(c))
                .map(|c| c.eclass == t1_class)
                .unwrap_or(false)
        })
        .expect("t2 anchored at t1");
    let rejoined = t2.eclass == x_class;
    println!(
        "double-transpose rejoin: t2 class {:?} vs x class {:?} -> rejoined = {rejoined}",
        t2.eclass, x_class
    );
    // TODAY'S PINNED FACT: no rejoin. The round-10 collapse rule must flip
    // this assertion (and when it does, flip it deliberately — that flip IS
    // the termination anchor landing).
    assert!(
        !rejoined,
        "a logical double-transpose collapse now exists; round 10's tower is anchored — \
         update this probe to assert rejoin and cite the rule"
    );
}
