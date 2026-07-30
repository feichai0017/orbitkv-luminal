//! The dtype-changing materializer: `out[i] = convert(in[i])` to the target
//! dtype carried in the egglog term. Never a view — the element bit width
//! changes underfoot, so the result is always fresh storage in the
//! canonical layout minted for the OUTPUT's dtype. Functional only.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, ToDps};

/// `CastGeneric(input) -> out`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cast;

impl OpSlotNames for Cast {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Cast {
    fn label(&self) -> &str {
        "CastGeneric"
    }
}

impl Bufferizable for Cast {}

impl ToDps for Cast {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(CastDps))
    }
}

impl LayoutIrOp for Cast {}

/// Destination-passing form of [`Cast`]:
///
/// ```text
/// CastGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastDps;

impl OpSlotNames for CastDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CastDps {
    fn reference_execute(
        &self,
        ctx: &mut crate::buffer_tensor_ir::ReferenceKernelCtx,
    ) -> anyhow::Result<()> {
        // The conversion is driven by the BUFFER types the plan annotated —
        // the op needs no dtype field of its own. Covered pairs only;
        // anything else refuses loudly (never a silent reinterpretation).
        use crate::buffer_tensor_ir::TypedBuffer;
        match (&ctx.operands[0], &mut ctx.dests[0]) {
            // Same-type: value-preserving copy (their Int-iota → F32 path
            // stores integer VALUES in f32, so this stays exact).
            (TypedBuffer::F32(input), TypedBuffer::F32(dest)) => {
                anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
                dest.copy_from_slice(input);
            }
            (TypedBuffer::Bool8(input), TypedBuffer::Bool8(dest)) => {
                anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
                dest.copy_from_slice(input);
            }
            // The indicator bridge: bool -> float is exactly 0.0 / 1.0.
            (TypedBuffer::Bool8(input), TypedBuffer::F32(dest)) => {
                anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
                for (out, code) in dest.iter_mut().zip(input) {
                    // The Bool8 invariant, enforced at the read: only the
                    // two legal codes exist; anything else is ill-formed
                    // data, not a truthy byte.
                    anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                    *out = f32::from(*code);
                }
            }
            (TypedBuffer::F32(_), TypedBuffer::Bool8(_)) => {
                anyhow::bail!(
                    "cast f32 -> Bool8 is not a reinterpretation: the != 0 \
                     reading is a PROJECTION and must appear as an explicit \
                     comparison in the model (LessThan), never as a cast"
                );
            }
        }
        Ok(())
    }

    fn label(&self) -> &str {
        "CastGeneric" // DPS forms keep the IR name
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for CastDps {
    fn alias_info(&self) -> Vec<crate::layout_ir::AliasInfo> {
        vec![crate::layout_ir::AliasInfo {
            operand: 1,
            result: 0,
            sharing: crate::layout_ir::Sharing::Must,
        }]
    }
}

impl ToDps for CastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for CastDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpCastGeneric` enodes and produces [`Cast`]
/// instances. Metadata children: `dtype` at child 1 (the target), and
/// `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct CastMatcher;

impl OpMatcher for CastMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCastGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("cast/match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("cast/match_functional.egg"),
            },
        ]
    }


    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("dtype", 1), ("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Cast)
    }
}
