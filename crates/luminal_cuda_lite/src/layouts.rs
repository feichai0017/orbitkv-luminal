//! The CUDA-lite runtime's LAYOUT vocabulary and renderer (resident-
//! geometry cleanup, ruling 2026-08-31). Core's bufferizer is generic
//! over an opaque layout type; this backend instantiates it with the
//! shared mirror vocabulary (`luminal::layouts`, a core convenience
//! module the bufferizer itself never calls) — its own choice, not a
//! core contract: a backend may bring layouts core has never heard of.

use anyhow::Result;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};

/// The CUDA-lite runtime's opaque plan-layout type.
pub type CudaLayout = luminal::layouts::MirrorLayout;

/// The CUDA-lite bufferized plan.
pub type CudaPlan = luminal::bufferize::BufferIrGraph<CudaLayout>;

/// The extraction-side layout renderer: minimal-faithful — the five
/// constructor spellings render into the mirror structs, preferring the
/// most-structured spelling present (a rendering preference only). No
/// normalization, no analysis; failure is loud and refuses the plan.
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaLayoutRenderer;

impl luminal::layout_ir::LayoutRenderer<CudaLayout> for CudaLayoutRenderer {
    fn render_layout(&self, egraph: &EGraph, class: &ClassId) -> Result<CudaLayout> {
        luminal::layouts::render_layout_for(egraph, class, "cuda-lite layout renderer")
    }
}
