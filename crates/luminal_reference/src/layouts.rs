//! The reference runtime's LAYOUT vocabulary and renderer (resident-
//! geometry cleanup, ruling 2026-08-31). Core's bufferizer is generic
//! over an opaque layout type it only clones and transports; THIS
//! runtime instantiates it with the shared mirror vocabulary
//! (`luminal::layouts` — the five egglog constructors, field-for-field,
//! a core convenience module the bufferizer itself never calls) and
//! registers its renderer beside its op matchers.

use anyhow::Result;
use egraph_serialize::{ClassId, EGraph};

/// The reference runtime's opaque plan-layout type.
pub type RefLayout = luminal::layouts::MirrorLayout;

/// The reference runtime's bufferized plan.
pub type ReferencePlan = luminal::bufferize::BufferIrGraph<RefLayout>;

/// The extraction-side layout renderer: minimal-faithful — the five
/// constructor spellings render into the mirror structs, preferring the
/// most-structured spelling present (a rendering preference only; all
/// spellings of a class denote one function). No normalization, no
/// analysis; failure is loud and refuses the plan.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceLayoutRenderer;

impl luminal::layout_ir::LayoutRenderer<RefLayout> for ReferenceLayoutRenderer {
    fn render_layout(&self, egraph: &EGraph, class: &ClassId) -> Result<RefLayout> {
        luminal::layouts::render_layout_for(egraph, class, "reference layout renderer")
    }
}
