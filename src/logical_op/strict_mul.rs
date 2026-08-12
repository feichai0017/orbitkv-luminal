use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The dynamic-checked Int mul; see LogicalStrictAdd.
#[derive(Debug, Clone, Copy)]
pub struct LogicalStrictMul;

impl LogicalOp for LogicalStrictMul {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalStrictMul"
    }

    fn display_name(&self) -> &'static str {
        "strict mul"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("lhs", 0), ("rhs", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalStrictMul({}, {})",
            ctx.child_expr(node, 0),
            ctx.child_expr(node, 1)
        )
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("strict_mul/constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("strict_mul/dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("strict_mul/value_bounds.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("strict_mul/shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("strict_mul/forward_layout.egg"),
            },
        ]
    }
}
