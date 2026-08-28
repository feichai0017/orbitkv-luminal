//! Train 3, Item 5 — the CPU-runnable contract tests: the executor-owned
//! validation and call planning for the cuBLASLt host call, no device
//! required. The device-gated halves live in `tests/cublaslt_contracts.rs`.

use luminal::prelude::egraph_serialize::ClassId;
use luminal_cuda_lite::ops::cublaslt::exec::{
    plan_call, plan_call_from_spec, validate_ld_bounds, CSource, LtDesc,
};
use luminal_cuda_lite::ops::cublaslt::{
    CuDim, CuEpilogue, CublasLt, CublasLtForm, LtMatmulSpec,
};

fn cid(s: &str) -> ClassId {
    ClassId::from(s)
}

/// A hand-built canonical spec: m x n = k-contracted call, contiguous
/// COL readings (ld = rows — the SPEC side keeps the frozen estate's
/// COL disclosure; the bridge re-expresses them as ROW descriptors,
/// see `exec.rs`'s ROW CONVENTION), no decoration.
fn base_spec(m: i64, n: i64, k: i64) -> LtMatmulSpec {
    LtMatmulSpec {
        form: CublasLtForm::Base,
        m: CuDim::Literal(m),
        n: CuDim::Literal(n),
        k: CuDim::Literal(k),
        trans_a: false,
        trans_b: false,
        lda: CuDim::Literal(m),
        ldb: CuDim::Literal(k),
        ldc: CuDim::Literal(m),
        ldd: CuDim::Literal(m),
        order_col: true,
        has_c: false,
        has_bias: false,
        epilogue: CuEpilogue::Default,
        logical_a: cid("a"),
        logical_b: cid("b"),
        logical_out: cid("out"),
        logical_site_out: cid("site_out"),
        desc_a_layout_tensor: cid("a_lt"),
        desc_b_layout_tensor: cid("b_lt"),
        c_tensor: None,
        bias_tensor: None,
        desc_a_buffer: None,
        desc_b_buffer: None,
        d_buffer: None,
    }
}

// ---------------------------------------------------------------------------
// Contract 4: the ld bounds validator — OUR check, because the library's
// own ld check is self-consistency only and VACUOUS at rows == 1.
// ---------------------------------------------------------------------------

#[test]
fn ld_bounds_accepts_contiguous_row_layouts() {
    // 4x3 ROW-contiguous: ld = 3 (row pitch), needs 3*3+3 = 12 elements.
    validate_ld_bounds("A", &LtDesc { rows: 4, cols: 3, ld: 3 }, 12).expect("exact fit");
    // Padded: ld = 6 over the same view needs 6*3+3 = 21.
    validate_ld_bounds("A", &LtDesc { rows: 4, cols: 3, ld: 6 }, 21).expect("padded fit");
    validate_ld_bounds("A", &LtDesc { rows: 4, cols: 3, ld: 6 }, 64).expect("slack");
}

#[test]
fn ld_bounds_rejects_the_rows_one_vacuous_case() {
    // THE load-bearing case (verified hardware finding): at rows == 1
    // the LIBRARY accepts any ld — its check is vacuous (in ROW order
    // a single row never dereferences ld) — so a too-small buffer
    // would be read out of bounds without a word. OUR check must
    // reject: 1x8 needs cols = 8 elements regardless of ld; the
    // buffer holds 4.
    let err = validate_ld_bounds("A", &LtDesc { rows: 1, cols: 8, ld: 1 }, 4)
        .expect_err("rows==1 with a short buffer must be refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("refused BEFORE dispatch"), "{msg}");
    assert!(msg.contains("vacuous"), "the refusal must name the vacuous library check: {msg}");
    // And the same descriptor over an adequate buffer passes.
    validate_ld_bounds("A", &LtDesc { rows: 1, cols: 8, ld: 1 }, 8).expect("adequate");
}

#[test]
fn ld_bounds_rejects_short_buffers_and_degenerate_geometry() {
    // ld too large for the buffer.
    validate_ld_bounds("B", &LtDesc { rows: 4, cols: 3, ld: 8 }, 12)
        .expect_err("ld 8 over 12 elements (needs 8*3+3 = 27)");
    // Zero/negative ld and empty geometry are refused outright.
    validate_ld_bounds("B", &LtDesc { rows: 4, cols: 3, ld: 0 }, 64).expect_err("ld 0");
    validate_ld_bounds("B", &LtDesc { rows: 0, cols: 3, ld: 1 }, 64).expect_err("rows 0");
    validate_ld_bounds("B", &LtDesc { rows: 4, cols: 0, ld: 4 }, 64).expect_err("cols 0");
}

#[test]
fn ld_bounds_gate_runs_for_every_descriptor_at_plan_validation() {
    let call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    // Operand element counts in Lit order [a, b]; dest = D.
    // (ROW bridge: A = 5x4 ld 4 -> 20, B = 3x5 ld 5 -> 15, D = 4x3
    // ld 3 -> 12 — numerically the same fits as the old COL pins.)
    call.validate_against(&[20, 15], 12).expect("all descriptors fit");
    call.validate_against(&[19, 15], 12).expect_err("A one element short");
    call.validate_against(&[20, 14], 12).expect_err("B one element short");
    call.validate_against(&[20, 15], 11).expect_err("D one element short");
    call.validate_against(&[20], 12).expect_err("operand count != Lit arity");
}

// ---------------------------------------------------------------------------
// Contract 3: descriptor construction — the C descriptor ALWAYS exists;
// the no-C forms alias D (a valid Cdesc mirroring D — Cdesc=NULL is the
// segfault the hardware campaign found).
// ---------------------------------------------------------------------------

#[test]
fn no_c_forms_always_carry_a_valid_cdesc_aliasing_d() {
    let call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    assert_eq!(call.c_source, CSource::AliasD);
    assert_eq!(call.c, call.d, "the aliased Cdesc mirrors the D descriptor exactly");
    assert!(!call.beta_is_one, "beta = 0.0f on the no-C forms: C is never read");
    // D is the EXECUTOR's dense row-major dest: ROW m x n, ld = n —
    // NEVER the spec's ldd (which describes the claimed e-graph layout
    // over the recorder's buffer; consuming it was the orientation bug).
    assert_eq!(call.d, LtDesc { rows: 4, cols: 3, ld: 3 });
}

#[test]
fn c_fold_forms_read_c_from_operand_two_with_structural_beta_one() {
    let mut spec = base_spec(4, 3, 5);
    spec.form = CublasLtForm::Accumulate;
    spec.has_c = true;
    spec.c_tensor = Some(cid("c_lt"));
    let call = plan_call_from_spec(&spec).expect("plan");
    assert_eq!(call.c_source, CSource::Operand(2), "contract order [a, b, c]");
    assert!(call.beta_is_one, "beta = 1.0f is STRUCTURAL on the C-fold forms");
    assert_eq!(call.c, call.d, "C rides the D layout by rule guard");
}

#[test]
fn bias_epilogue_forms_refuse_at_plan_time() {
    // THE MEASURED A100 FINDING (2026-08-28 probe): the library
    // refuses BIAS/RELU_BIAS whenever D is CUBLASLT_ORDER_ROW, and
    // the marker's sibling-frame per-D-row bias cannot be expressed
    // through a COL re-description of the executor's row-major dest —
    // the bridge refuses the bias forms LOUDLY before any descriptor
    // is built.
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        let mut spec = base_spec(4, 3, 5);
        spec.form = form;
        spec.has_c = form.has_c();
        spec.has_bias = true;
        if form.has_c() {
            spec.c_tensor = Some(cid("c_lt"));
        }
        spec.bias_tensor = Some(cid("bias_lt"));
        spec.epilogue = CuEpilogue::Bias;
        let err = plan_call_from_spec(&spec).expect_err("bias form must refuse at plan time");
        let msg = format!("{err:#}");
        assert!(msg.contains("NOT dispatchable under"), "{msg}");
        assert!(msg.contains("ROW"), "the refusal must name the convention: {msg}");
    }
}

#[test]
fn row_bridge_flips_the_spec_col_readings() {
    // THE ROW RE-EXPRESSION: a spec COL `r x c / ld` reading of the
    // operand bytes is the ROW `c x r / ld` reading of the transposed
    // matrix, so the bridge swaps dims and FLIPS the transpose op; the
    // spec's ld carries over verbatim.
    //
    // Spec N/N (COL A' = m x k = 4x5 ld 4; COL B' = k x n = 5x3 ld 5)
    // => ROW A' = 5x4 ld 4 at T, ROW B' = 3x5 ld 5 at T.
    let call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    assert!(call.trans_a && call.trans_b, "N/N spec => T/T ROW call");
    assert_eq!(call.a, LtDesc { rows: 5, cols: 4, ld: 4 });
    assert_eq!(call.b, LtDesc { rows: 3, cols: 5, ld: 5 });

    // Spec trans_a: COL A' stored [k, m] = 5x4 ld 5 presented as
    // op(A') = m x k => ROW A' = 4x5 ld 5 at N.
    let mut spec = base_spec(4, 3, 5);
    spec.trans_a = true;
    spec.lda = CuDim::Literal(5); // contiguous COL: ld = rows' = k
    let call = plan_call_from_spec(&spec).expect("plan");
    assert_eq!(call.a, LtDesc { rows: 4, cols: 5, ld: 5 });
    assert_eq!(call.b, LtDesc { rows: 3, cols: 5, ld: 5 });
    assert!(!call.trans_a && call.trans_b);
}

// ---------------------------------------------------------------------------
// Contract 2 (scalar scope): there is NO runtime scalar channel. The
// call carries no alpha at all and beta only as the structural
// `beta_is_one` bit — a compile-time property of `LtCall`'s type (no
// f32 field exists to smuggle a runtime scalar through); these tests
// pin the structural derivation per form.
// ---------------------------------------------------------------------------

#[test]
fn beta_is_structural_per_form_and_nothing_else() {
    // The bias forms refuse at plan time (see
    // `bias_epilogue_forms_refuse_at_plan_time`); the structural-beta
    // pin runs over the dispatchable forms.
    for form in [CublasLtForm::Base, CublasLtForm::Accumulate] {
        let mut spec = base_spec(4, 3, 5);
        spec.form = form;
        spec.has_c = form.has_c();
        if form.has_c() {
            spec.c_tensor = Some(cid("c_lt"));
        }
        let call = plan_call_from_spec(&spec).expect("plan");
        assert_eq!(
            call.beta_is_one,
            form.has_c(),
            "beta is a function of the FORM alone (the C-fold decorator), never data"
        );
    }
}

// ---------------------------------------------------------------------------
// Loud bails: symbolic geometry and missing specs refuse before any
// descriptor is built.
// ---------------------------------------------------------------------------

#[test]
fn symbolic_geometry_is_a_loud_pre_dispatch_refusal() {
    let mut spec = base_spec(4, 3, 5);
    spec.k = CuDim::Symbolic(cid("k_class"));
    let err = plan_call_from_spec(&spec).expect_err("symbolic k");
    assert!(format!("{err:#}").contains("SYMBOLIC"), "{err:#}");
}

#[test]
fn an_elected_op_without_a_parsed_spec_refuses() {
    let op = CublasLt { form: CublasLtForm::Base, spec: None };
    let err = plan_call(&op).expect_err("no spec");
    assert!(format!("{err:#}").contains("no parsed LtMatmulSpec"), "{err:#}");
}
