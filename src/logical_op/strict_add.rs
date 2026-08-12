use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The dynamic-checked Int add (Rust strict_* naming, ruling 2026-08-11): overflow is a loud kernel panic. NO ring rewrites — Strict ops are opaque, the escape hatch where bounds proofs cannot reach; plain LogicalAdd is the proof-gated ring citizen.
#[derive(Debug, Clone, Copy)]
pub struct LogicalStrictAdd;

impl LogicalOp for LogicalStrictAdd {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalStrictAdd"
    }

    fn display_name(&self) -> &'static str {
        "strict add"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("lhs", 0), ("rhs", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalStrictAdd({}, {})",
            ctx.child_expr(node, 0),
            ctx.child_expr(node, 1)
        )
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("strict_add/constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("strict_add/dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("strict_add/value_bounds.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("strict_add/shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("strict_add/forward_layout.egg"),
            },
        ]
    }
}
