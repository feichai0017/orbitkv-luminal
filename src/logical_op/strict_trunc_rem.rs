use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The dynamic-checked truncated remainder; see LogicalStrictTruncDiv.
#[derive(Debug, Clone, Copy)]
pub struct LogicalStrictTruncRem;

impl LogicalOp for LogicalStrictTruncRem {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalStrictTruncRem"
    }

    fn display_name(&self) -> &'static str {
        "strict trunc rem"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("numerator", 0), ("denominator", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalStrictTruncRem({}, {})",
            ctx.child_expr(node, 0),
            ctx.child_expr(node, 1)
        )
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("strict_trunc_rem/constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("strict_trunc_rem/dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("strict_trunc_rem/value_bounds.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("strict_trunc_rem/shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("strict_trunc_rem/forward_layout.egg"),
            },
        ]
    }
}
