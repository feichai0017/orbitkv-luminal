use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Coordinate-form scatter: `out[coords(c)] = src[c]`, elsewhere init.
/// The coordinate tensors ride behind the list child, so only `init` and
/// `src` are direct ports.
#[derive(Debug, Clone, Copy)]
pub struct LogicalScatter;

impl LogicalOp for LogicalScatter {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalScatter"
    }

    fn display_name(&self) -> &'static str {
        "scatter"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("init", 0), ("src", 2)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let init = ctx.child_expr(node, 0);
        let coords = ctx
            .child_short(node, 1, 6, None)
            .unwrap_or_else(|| "?".to_string());
        let src = ctx.child_expr(node, 2);
        format!("LogicalScatter(init={init}, coords={coords}, src={src})")
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("scatter/constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("scatter/dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("scatter/shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("scatter/forward_layout.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("scatter/forward_layout_2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Seed,
                text: include_str!("scatter/seed.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Seed,
                text: include_str!("scatter/seed_2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("scatter/fixpoint.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("scatter/fixpoint_2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("scatter/fixpoint_3.egg"),
            },
        ]
    }
}
