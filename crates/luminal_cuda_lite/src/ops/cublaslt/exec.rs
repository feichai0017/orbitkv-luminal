//! The cuBLASLt HOST-CALL contract layer — CPU-side, device-free.
//!
//! Train 3 (registry + host-call dispatch): the four marker contracts
//! execute as ONE host library call (`cublasLtMatmul`), not an NVRTC
//! kernel. This module turns an elected [`CublasLt`] op (its parsed
//! [`LtMatmulSpec`]) into a fully-resolved [`LtCall`] — plain numerics
//! and finite classifications only — and carries the executor-side
//! validation the library itself does NOT provide. Everything here is
//! unit-testable without a device; the cudarc dispatch lives in
//! `device_call` (feature-gated) and consumes an `LtCall` verbatim.
//!
//! THE A100 EXECUTOR CONTRACTS (verified findings; load-bearing):
//! 1. F32-only end to end (inputs, outputs, CUBLAS_COMPUTE_32F).
//! 2. POINTER_MODE_HOST with LITERAL scalars: alpha = 1.0f always
//!    (the marker has no alpha channel); beta is STRUCTURAL — 1.0f on
//!    the C-fold (Accumulate) forms, 0.0f otherwise. There is no
//!    runtime scalar channel: [`LtCall`] has NO alpha/beta float
//!    fields, only [`LtCall::beta_is_one`], and the literals live in
//!    the dispatch site as compile-time constants.
//! 3. C = D aliasing with a VALID Cdesc on the no-C forms
//!    (Cdesc = NULL segfaults): [`LtCall::c`] is NON-OPTIONAL — every
//!    call carries a C descriptor; [`CSource::AliasD`] says "pass the
//!    D pointer" and beta = 0.0f guarantees C is never read.
//! 4. ld semantics: the library's own ld check is self-consistency
//!    only and VACUOUS at rows == 1 (in ROW order a single row never
//!    dereferences ld) — [`validate_ld_bounds`] is the REAL bounds
//!    validation (`ld*(rows-1) + cols <= element count` in ROW order),
//!    asserted loudly at dispatch for A, B, C, and D. Emitted lds are
//!    clamped to `>= 1` at plan time.
//! 5. TF32 is graph-modeled, never a flag: the dispatch sets
//!    CUBLAS_COMPUTE_32F explicitly and `device_call` carries a
//!    startup detector assertion at handle creation.
//!
//! THE ROW CONVENTION (Train-3 orientation fix; measured on the A100
//! with the 4x8x3 dump — see `tests/cublaslt_contracts.rs`):
//! every emitted layout descriptor declares CUBLASLT_ORDER_ROW. This
//! DECLARES REALITY, in two halves:
//!
//!  * A and B: the marker spec's readings are COL views over the
//!    operand buffers' bytes (frozen estate convention R9/R10). A COL
//!    `r x c / ld` view of a byte range IS the ROW `c x r / ld` view
//!    of the transposed matrix — same bytes, same pitch — so the
//!    bridge re-expresses each operand reading as ROW by swapping the
//!    descriptor dims and FLIPPING the transpose op. The spec's lds
//!    carry over verbatim (a COL view's ld and the underlying
//!    row-major storage's row pitch are the same number, padded
//!    layouts included).
//!  * D (and C, which rides D's layout by rule guard): the EXECUTOR'S
//!    destination convention is authoritative — the CL executor is
//!    out-of-place and materializes every result value DENSE
//!    ROW-MAJOR in the value's own dims (the disclosure downstream
//!    walks exactly that). The spec's ldd/ldc describe the CLAIMED
//!    e-graph layout over the RECORDER's out buffer — a buffer the
//!    executor never writes — so the bridge derives D from the call
//!    frame alone: ROW `m x n` with ld = n (the dense row pitch).
//!    Writing the spec's COL D descriptor into the fresh dest was the
//!    orientation bug: bytes landed COL-major (element (r,c) at
//!    c*m + r) while the disclosure reads row-major (r*n + c) —
//!    element 0 agreed, element 1 did not.
//!
//! The CM-swap alternative (compute D^T with swapped roles under COL
//! defaults) is REJECTED: cuBLASLt's bias epilogue adds bias[i] to
//! row i of the API's D, and the marker's bias contract is per-row of
//! the call's D (length m) — a role swap would silently turn it into
//! a per-column bias. ROW order keeps bias semantics intact.

use anyhow::{bail, Result};

use super::{CuDim, CuEpilogue, CublasLt, CublasLtForm, LtMatmulSpec};

/// One descriptor's ROW-order geometry: `rows x cols` with leading
/// dimension `ld` = the ROW pitch (elements between consecutive rows),
/// all resolved literals (elements, not bytes). Every descriptor the
/// bridge emits is declared CUBLASLT_ORDER_ROW at dispatch — see the
/// module doc's ROW CONVENTION.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtDesc {
    pub rows: i64,
    pub cols: i64,
    pub ld: i64,
}

/// Where the C pointer comes from. The C DESCRIPTOR always exists
/// (contract 3); only the pointer source varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CSource {
    /// No-C forms: pass the D pointer as C (beta = 0.0f, C never read).
    AliasD,
    /// C-fold forms: the Lit operand at this index (contract order
    /// `[a, b, c, bias?]` puts c at 2); beta = 1.0f.
    Operand(usize),
}

/// The fully-resolved host call: every number the dispatch needs,
/// nothing the dispatch may reinterpret. NO scalar fields beyond the
/// structural `beta_is_one` — see module doc contract 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LtCall {
    pub form: CublasLtForm,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub a: LtDesc,
    pub b: LtDesc,
    /// ALWAYS present (contract 3): on the no-C forms this mirrors `d`.
    pub c: LtDesc,
    pub d: LtDesc,
    pub c_source: CSource,
    /// beta is STRUCTURAL: `true` exactly on the C-fold forms.
    pub beta_is_one: bool,
    pub relu: bool,
    /// Lit operand index of the bias vector, on the bias forms.
    pub bias_operand: Option<usize>,
}

fn literal(dim: &CuDim, what: &str) -> Result<i64> {
    match dim.literal() {
        Some(v) => Ok(v),
        None => bail!(
            "cuBLASLt dispatch: {what} is SYMBOLIC — binding symbolic geometry \
             from the dyn map at dispatch is not wired in this landing (loud \
             bail, never a guess)"
        ),
    }
}

/// The REAL ld bounds validation (contract 4), in ROW order. The
/// library's own check is self-consistency only — `ld >= cols` when
/// there is more than one row — and VACUOUS at rows == 1 (a single row
/// never dereferences ld), so a too-small buffer would be read/written
/// out of bounds without a word. This check is the one that counts:
/// `ld*(rows-1) + cols <= elems`, plus positivity.
pub fn validate_ld_bounds(who: &str, desc: &LtDesc, elems: usize) -> Result<()> {
    if desc.rows < 1 || desc.cols < 1 {
        bail!(
            "cuBLASLt {who}: empty descriptor geometry {}x{} — refused before dispatch",
            desc.rows,
            desc.cols
        );
    }
    if desc.ld < 1 {
        bail!("cuBLASLt {who}: ld {} < 1 — refused before dispatch", desc.ld);
    }
    let needed = desc
        .ld
        .checked_mul(desc.rows - 1)
        .and_then(|v| v.checked_add(desc.cols))
        .ok_or_else(|| {
            anyhow::anyhow!("cuBLASLt {who}: ld*(rows-1)+cols overflows i64 — refused")
        })?;
    if needed as i128 > elems as i128 {
        bail!(
            "cuBLASLt {who}: descriptor {}x{} ld {} needs {} elements but the \
             buffer holds {} — out-of-bounds access refused BEFORE dispatch \
             (the library's own ld check is vacuous at rows==1)",
            desc.rows,
            desc.cols,
            desc.ld,
            needed,
            elems
        );
    }
    Ok(())
}

/// Resolve an elected [`CublasLt`] op into the host call. Loud on a
/// missing spec (an op elected without its parsed marker spec is
/// malformed) and on symbolic geometry.
pub fn plan_call(op: &CublasLt) -> Result<LtCall> {
    let Some(spec) = op.spec.as_ref() else {
        bail!(
            "cuBLASLt dispatch: elected {} carries no parsed LtMatmulSpec — \
             the marker's extract() did not resolve this site",
            op.form.constructor_name()
        );
    };
    plan_call_from_spec(spec)
}

/// [`plan_call`] over the spec alone (test seam).
pub fn plan_call_from_spec(spec: &LtMatmulSpec) -> Result<LtCall> {
    // BIAS-EPILOGUE REFUSAL (measured on the A100, 2026-08-28 probe —
    // see the ROW CONVENTION module doc): cublasLtMatmulAlgoGetHeuristic
    // returns CUBLAS_STATUS_NOT_SUPPORTED for CUBLASLT_EPILOGUE_BIAS /
    // RELU_BIAS whenever D is CUBLASLT_ORDER_ROW (any A/B order;
    // DEFAULT and RELU are supported under every order combination).
    // The bias vector's length is pinned to D's ROW count by the API,
    // and the marker's bias contract is per-row of the SIBLING call's
    // D (length m = the recorder's feature dim) — re-expressing D as
    // COL over the executor's row-major dest transposes the frame and
    // would need a per-COLUMN bias, which the API does not have. Under
    // the frozen estate + the executor's dense row-major destination
    // there is NO correct dispatch for the bias forms: refuse loudly,
    // never land wrong bytes.
    if spec.form.has_bias() {
        bail!(
            "cuBLASLt {}: the bias-epilogue contracts are NOT dispatchable under \
             the ROW convention — the A100 library refuses BIAS/RELU_BIAS with a \
             ROW-order D (measured CUBLAS_STATUS_NOT_SUPPORTED), and the API's \
             per-D-row bias cannot express the marker's per-row-of-the-sibling-D \
             vector through a COL re-description of the row-major destination; \
             refusing before any descriptor is built",
            spec.form.constructor_name()
        );
    }
    let m = literal(&spec.m, "m")?;
    let n = literal(&spec.n, "n")?;
    let k = literal(&spec.k, "k")?;

    // THE ROW CONVENTION (module doc): the spec's readings are COL
    // views (frozen estate convention); a COL `r x c / ld` view of the
    // operand bytes IS the ROW `c x r / ld` view of the transposed
    // matrix, so the bridge flips each operand's transpose op and
    // swaps the descriptor dims. The op algebra is then the same shape
    // as before, with the FLIPPED trans:
    //   A: op(A') is m x k  =>  A' is (m,k) at N, (k,m) at T
    //   B: op(B') is k x n  =>  B' is (k,n) at N, (n,k) at T
    //   D:                      m x n (no transD)
    let trans_a = !spec.trans_a;
    let trans_b = !spec.trans_b;
    let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
    let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };
    let (d_rows, d_cols) = (m, n);

    // Operand lds carry over VERBATIM from the spec (a COL view's ld
    // and the row-major storage's row pitch are the same number,
    // padded layouts included), clamped to >= 1 (contract 4's
    // emission rule).
    let ld = |dim: &CuDim, what: &str| -> Result<i64> { Ok(literal(dim, what)?.max(1)) };
    let a = LtDesc { rows: a_rows, cols: a_cols, ld: ld(&spec.lda, "lda")? };
    let b = LtDesc { rows: b_rows, cols: b_cols, ld: ld(&spec.ldb, "ldb")? };
    // D is the EXECUTOR's destination, not the spec's claim: the CL
    // executor materializes every result DENSE ROW-MAJOR in the
    // value's dims (the out-of-place convention the disclosure walks),
    // so ld = n — the dense row pitch. The spec's ldd describes the
    // claimed e-graph layout over the RECORDER's out buffer, which the
    // executor never writes; consuming it here was the orientation
    // bug.
    let d = LtDesc { rows: d_rows, cols: d_cols, ld: n.max(1) };
    // C rides the D layout by rule guard (the marker cross-checks the
    // layout classes), so Cdesc == Ddesc geometry on EVERY form — the
    // valid-Cdesc contract for the no-C forms comes for free. On the
    // C-fold forms the executor owes a C operand buffer holding the
    // call-frame C dense row-major (the dispatch site enforces it).
    let c = d;

    let c_source = if spec.form.has_c() {
        // Contract order [a, b, c, bias?]: c is Lit operand 2.
        CSource::Operand(2)
    } else {
        CSource::AliasD
    };
    let bias_operand = spec.form.has_bias().then(|| match spec.form {
        CublasLtForm::Bias => 2,
        CublasLtForm::AccumulateBias => 3,
        _ => unreachable!("has_bias() is true only on the bias forms"),
    });
    let relu = matches!(spec.epilogue, CuEpilogue::Relu | CuEpilogue::ReluBias);

    Ok(LtCall {
        form: spec.form,
        m,
        n,
        k,
        trans_a,
        trans_b,
        a,
        b,
        c,
        d,
        c_source,
        beta_is_one: spec.form.has_c(),
        relu,
        bias_operand,
    })
}

impl LtCall {
    /// Validate every descriptor against its backing buffer's element
    /// count — the pre-dispatch gate (contract 4). `elems` in Lit
    /// operand order `[a, b, c?, bias?]`, then the destination.
    pub fn validate_against(&self, operand_elems: &[usize], dest_elems: usize) -> Result<()> {
        if operand_elems.len() != self.form.lit_arity() {
            bail!(
                "cuBLASLt {}: {} operand buffers for Lit arity {}",
                self.form.constructor_name(),
                operand_elems.len(),
                self.form.lit_arity()
            );
        }
        validate_ld_bounds("A", &self.a, operand_elems[0])?;
        validate_ld_bounds("B", &self.b, operand_elems[1])?;
        validate_ld_bounds("D", &self.d, dest_elems)?;
        match self.c_source {
            CSource::AliasD => validate_ld_bounds("C(=D)", &self.c, dest_elems)?,
            CSource::Operand(i) => {
                let Some(&elems) = operand_elems.get(i) else {
                    bail!("cuBLASLt: C operand index {i} out of operand range");
                };
                validate_ld_bounds("C", &self.c, elems)?;
            }
        }
        if let Some(i) = self.bias_operand {
            let Some(&elems) = operand_elems.get(i) else {
                bail!("cuBLASLt: bias operand index {i} out of operand range");
            };
            // The bias vector is length m (one entry per D row —
            // independent of storage order; this is why the ROW
            // convention was chosen over the CM-swap trick).
            if (elems as i128) < self.m as i128 {
                bail!(
                    "cuBLASLt bias: buffer holds {elems} elements, epilogue reads m = {} \
                     — refused before dispatch",
                    self.m
                );
            }
        }
        Ok(())
    }
}
