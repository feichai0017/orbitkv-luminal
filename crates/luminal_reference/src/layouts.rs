//! The reference runtime's LAYOUT vocabulary and decoder (resident-
//! geometry cleanup, ruling 2026-08-31; Option B rework). Core's
//! bufferizer is generic over an opaque layout type it only clones and
//! transports; THIS runtime instantiates it with a TYPED layout — the
//! shared mirror vocabulary (`luminal::layouts`, a core convenience
//! module the bufferizer itself never calls) plus the value's `dtype-of`
//! fact. The dtype is the RUNTIME's own extraction-side knowledge folded
//! into the runtime's own type at decode time — it is not plan
//! vocabulary (the plan carries `RefLayout` opaquely), and it is what
//! makes Option-B plans self-contained for `load_plan` callers: staging,
//! allocation, and readback all read the carried layout instead of a
//! table an external caller never had.

use anyhow::Result;
use egraph_serialize::EGraph;
use luminal::layout_ir::LayoutTensorInfo;

/// The reference runtime's opaque plan-layout type: the decoded mirror
/// layout plus the value's dtype fact. `dtype: None` is representable
/// (a value without a `dtype-of` row) and bails loudly at USE — staging,
/// allocation typing, readback — never silently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefLayout {
    pub mirror: luminal::layouts::MirrorLayout,
    pub dtype: Option<luminal::dtype::PlanDtype>,
}

/// The reference runtime's bufferized plan.
pub type ReferencePlan = luminal::bufferize::BufferIrGraph<RefLayout>;

/// The extraction-side layout decoder: minimal-faithful — the five
/// constructor spellings decode into the mirror structs, preferring the
/// most-structured spelling present (a decoding preference only; all
/// spellings of a class denote one function), with the value's dtype
/// fact carried alongside. No normalization, no analysis; failure is
/// loud and refuses the plan. Pure in `(layout class, dtype fact)` —
/// the decoder cache contract.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceLayoutDecoder;

impl luminal::layout_ir::LayoutDecoder<RefLayout> for ReferenceLayoutDecoder {
    fn decode(&self, egraph: &EGraph, value: &LayoutTensorInfo) -> Result<RefLayout> {
        Ok(RefLayout {
            mirror: luminal::layouts::decode_layout_for(
                egraph,
                &value.layout.eclass,
                "reference layout decoder",
            )?,
            dtype: value.dtype_enum,
        })
    }
}
