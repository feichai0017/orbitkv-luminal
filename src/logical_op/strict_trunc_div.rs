use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The dynamic-checked truncated division: zero divisor or the MIN/-1 corner is a loud kernel panic; see LogicalStrictAdd.
#[derive(Debug, Clone, Copy)]
pub struct LogicalStrictTruncDiv;

impl LogicalOp for LogicalStrictTruncDiv {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalStrictTruncDiv"
    }

    fn display_name(&self) -> &'static str {
        "strict trunc div"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("numerator", 0), ("denominator", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalStrictTruncDiv({}, {})",
            ctx.child_expr(node, 0),
            ctx.child_expr(node, 1)
        )
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("strict_trunc_div/constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("strict_trunc_div/dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("strict_trunc_div/value_bounds.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("strict_trunc_div/shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("strict_trunc_div/forward_layout.egg"),
            },
        ]
    }
}
