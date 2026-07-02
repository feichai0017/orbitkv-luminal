//! PT2 graph nodes -> Luminal Graph translation.
//!
//! Walks the parsed PT2 graph and constructs an equivalent Luminal computation graph.

mod attention;
mod binary;
mod conv;
mod dispatch;
mod movement;
mod movement_dynamic;
mod reduction;
mod tensor;
mod unary;

use std::collections::HashMap;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

use anyhow::{Context, Result};
use luminal::graph::Graph;
use luminal::prelude::*;

use crate::pt2_expr::parse_sympy_expr_with_ranges;
use crate::pt2_parser::{InputKind, ParsedPT2, SymDimMap};
use crate::pt2_schema::*;
use crate::pt2_util;

/// Result of translating a PT2 graph to a Luminal graph.
pub struct TranslatedGraph {
    /// The luminal computation graph.
    pub graph: Graph,
    /// Node IDs for user inputs (in order).
    pub user_input_ids: Vec<(String, NodeIndex)>,
    /// Node IDs for outputs (in order).
    pub output_ids: Vec<(String, NodeIndex)>,
    /// Symbolic dimension mapping.
    pub sym_map: SymDimMap,
}

/// Main translation entry point.
pub fn translate(parsed: &ParsedPT2) -> Result<TranslatedGraph> {
    let mut translator = Translator::new(parsed)?;
    translator.translate_graph()?;
    Ok(translator.finish())
}

pub(crate) struct Translator<'a> {
    pub(crate) parsed: &'a ParsedPT2,
    pub(crate) graph: Graph,
    /// Maps tensor name -> GraphTensor
    pub(crate) tensors: HashMap<String, GraphTensor>,
    pub(crate) sym_map: SymDimMap,
    pub(crate) user_input_ids: Vec<(String, NodeIndex)>,
    pub(crate) output_ids: Vec<(String, NodeIndex)>,
    /// Extra tensor metadata from inlined subgraphs.
    pub(crate) extra_tensor_values: HashMap<String, TensorMeta>,
    /// index_put results whose destination is a graph input (e.g. HF
    /// StaticCache K/V buffers): result node id -> the destination input
    /// tensor. Declared graph outputs matching these are emitted against the
    /// input instead — the runtime owns that buffer as persistent state and
    /// the fused kernels update it in place, so the output is a state read.
    /// This also keeps loop-rolled graphs free of output edges into layer
    /// bodies (see luminal_trainium docs/luminal_loop_rolling_issue.md).
    pub(crate) input_backed_write_backs: HashMap<NodeIndex, GraphTensor>,
}

impl<'a> Translator<'a> {
    fn new(parsed: &'a ParsedPT2) -> Result<Self> {
        let sym_map = parsed.build_sym_dim_map();
        Ok(Self {
            parsed,
            graph: Graph::new(),
            tensors: HashMap::new(),
            sym_map,
            user_input_ids: Vec::new(),
            output_ids: Vec::new(),
            extra_tensor_values: HashMap::new(),
            input_backed_write_backs: HashMap::new(),
        })
    }

    fn translate_graph(&mut self) -> Result<()> {
        self.create_inputs()?;

        let nodes = &self.parsed.program.graph_module.graph.nodes;
        for (i, node) in nodes.iter().enumerate() {
            self.translate_node(node)
                .with_context(|| format!("Failed to translate node {i}: {}", node.target))?;
        }

        let output_names = self.parsed.output_names();
        for name in &output_names {
            let tensor = self.get_tensor(name)?;
            // In-place write-backs into graph inputs (StaticCache updates)
            // alias the runtime-owned buffer: emit the output against the
            // input node itself so readback serves the post-run state bytes.
            if let Some(dest) = self.input_backed_write_backs.get(&tensor.id) {
                dest.output();
                self.output_ids.push((name.clone(), dest.id));
                continue;
            }
            let tensor = if tensor.dtype == DType::Bool {
                tensor.cast(DType::Int).cast(DType::Bool)
            } else if tensor.dtype == DType::Int {
                tensor
            } else {
                tensor + 0.0
            };
            tensor.output();
            self.output_ids.push((name.clone(), tensor.id));
        }
        self.debug_dump_translated_tensors()?;
        self.debug_mark_translated_outputs()?;
        self.debug_write_node_manifest()?;

        Ok(())
    }

    fn create_inputs(&mut self) -> Result<()> {
        let inputs = self.parsed.classify_inputs();
        for input in &inputs {
            match input {
                InputKind::Parameter {
                    graph_name,
                    original_name,
                } => {
                    let meta = self
                        .parsed
                        .tensor_meta(graph_name)
                        .with_context(|| format!("Missing tensor meta for param {graph_name}"))?;
                    let shape = self.tensor_meta_to_shape(meta)?;
                    let dtype = pt2_util::torch_dtype_int_to_luminal(meta.dtype);
                    let tensor = self
                        .graph
                        .named_tensor(original_name, shape)
                        .as_dtype(dtype);
                    tensor.persist();
                    self.tensors.insert(graph_name.clone(), tensor);
                }
                InputKind::Buffer {
                    graph_name,
                    original_name,
                } => {
                    let meta = self
                        .parsed
                        .tensor_meta(graph_name)
                        .with_context(|| format!("Missing tensor meta for buffer {graph_name}"))?;
                    let shape = self.tensor_meta_to_shape(meta)?;
                    let dtype = pt2_util::torch_dtype_int_to_luminal(meta.dtype);
                    let tensor = self
                        .graph
                        .named_tensor(original_name, shape)
                        .as_dtype(dtype);
                    tensor.persist();
                    self.tensors.insert(graph_name.clone(), tensor);
                }
                InputKind::UserInput { graph_name } => {
                    let meta = self
                        .parsed
                        .tensor_meta(graph_name)
                        .with_context(|| format!("Missing tensor meta for input {graph_name}"))?;
                    let shape = self.tensor_meta_to_shape(meta)?;
                    let dtype = pt2_util::torch_dtype_int_to_luminal(meta.dtype);
                    let tensor = self.graph.named_tensor(graph_name, shape).as_dtype(dtype);
                    self.user_input_ids.push((graph_name.clone(), tensor.id));
                    self.tensors.insert(graph_name.clone(), tensor);
                }
            }
        }
        Ok(())
    }

    fn finish(self) -> TranslatedGraph {
        TranslatedGraph {
            graph: self.graph,
            user_input_ids: self.user_input_ids,
            output_ids: self.output_ids,
            sym_map: self.sym_map,
        }
    }

    // --- Helper methods ---

    pub(crate) fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.extra_tensor_values
            .get(name)
            .or_else(|| self.parsed.tensor_meta(name))
    }

    pub(crate) fn get_tensor(&self, name: &str) -> Result<GraphTensor> {
        self.tensors
            .get(name)
            .copied()
            .with_context(|| format!("Unknown tensor: {name}"))
    }

    fn debug_dump_translated_tensors(&self) -> Result<()> {
        let Some(path) = env::var_os("LUMINAL_PYTHON_DEBUG_TENSOR_NAMES_FILE") else {
            return Ok(());
        };
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open {:?}", path))?;
        writeln!(file, "translated_tensors={}", names.len())?;
        for name in names {
            if let Some(tensor) = self.tensors.get(name) {
                let has_meta = self.parsed.tensor_meta(name).is_some();
                writeln!(
                    file,
                    "{name}\tdtype={:?}\tshape={:?}\tpt2_meta={has_meta}",
                    tensor.dtype, tensor.shape.dims
                )?;
            }
        }
        Ok(())
    }

    fn debug_mark_translated_outputs(&mut self) -> Result<()> {
        let Ok(raw_names) = env::var("LUMINAL_PYTHON_DEBUG_OUTPUT_TENSORS") else {
            return Ok(());
        };
        let names: Vec<String> = raw_names
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        for name in names {
            if self.output_ids.iter().any(|(existing, _)| existing == &name) {
                continue;
            }
            let tensor = self.get_tensor(&name)?;
            if self.parsed.tensor_meta(&name).is_none() {
                anyhow::bail!(
                    "debug output tensor {name:?} has no PT2 tensor metadata; choose a PT2 tensor name from LUMINAL_PYTHON_DEBUG_TENSOR_NAMES_FILE"
                );
            }
            let tensor = tensor.output();
            self.output_ids.push((name, tensor.id));
        }
        Ok(())
    }

    /// LUMINAL_PYTHON_DEBUG_NODE_MANIFEST=<path> writes one line per
    /// translated FX node (name, target, input tensor names, dtype, shape) so
    /// debug taps can be located by graph structure without re-exporting.
    fn debug_write_node_manifest(&self) -> Result<()> {
        let Some(path) = env::var_os("LUMINAL_PYTHON_DEBUG_NODE_MANIFEST") else {
            return Ok(());
        };
        use std::fmt::Write as _;
        let mut out = String::new();
        for node in &self.parsed.program.graph_module.graph.nodes {
            let input_names: Vec<&str> = node
                .inputs
                .iter()
                .filter_map(|input| match &input.arg {
                    Argument::Tensor(t) => Some(t.as_tensor.name.as_str()),
                    _ => None,
                })
                .collect();
            for output in &node.outputs {
                let Some(name) = output.as_tensor.as_ref().map(|t| t.name.as_str()) else {
                    continue;
                };
                let (dtype, shape) = match self.tensors.get(name) {
                    Some(tensor) => (
                        format!("{:?}", tensor.dtype),
                        format!("{:?}", tensor.shape.dims),
                    ),
                    None => ("?".to_string(), "?".to_string()),
                };
                writeln!(
                    out,
                    "{name}\t{target}\t[{inputs}]\t{dtype}\t{shape}",
                    target = node.target,
                    inputs = input_names.join(",")
                )
                .expect("write to string");
            }
        }
        std::fs::write(&path, out)
            .with_context(|| format!("write node manifest to {path:?}"))?;
        Ok(())
    }

    pub(crate) fn get_input_tensor(&self, node: &Node, idx: usize) -> Result<GraphTensor> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        let name = arg.as_tensor_name().with_context(|| {
            format!("Input {idx} of {} is not a tensor: {:?}", node.target, arg)
        })?;
        self.get_tensor(name)
    }

    pub(crate) fn get_int_arg(&self, node: &Node, idx: usize) -> Result<i64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        if let Some(v) = arg.as_int() {
            return Ok(v);
        }
        // Fall through to symbolic-aware resolution. Op-arg slots like `dim`
        // and `axis` are always concrete in practice, but with dynamic shapes
        // PT2 occasionally hands us a SymInt that is fully bound at export
        // time (e.g. an `unsqueeze` whose dim was derived from `len(shape)`);
        // accept those when they reduce to a concrete int rather than failing
        // with the misleading "not an int" diagnostic.
        if let Some(expr) = self.resolve_arg_as_expression(arg)
            && let Some(v) = expr.to_usize()
        {
            return Ok(v as i64);
        }
        anyhow::bail!("Input {idx} of {} is not an int: {:?}", node.target, arg)
    }

    pub(crate) fn get_float_arg(&self, node: &Node, idx: usize) -> Result<f64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        if let Some(f) = arg.as_float() {
            return Ok(f);
        }
        if let Some(i) = arg.as_int() {
            return Ok(i as f64);
        }
        anyhow::bail!("Input {idx} of {} is not a float: {:?}", node.target, arg)
    }

    pub(crate) fn get_ints_arg(&self, node: &Node, idx: usize) -> Result<Vec<i64>> {
        use crate::pt2_schema::SymIntEntry;
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        // Symbolic int lists: tolerate them as long as every entry is a
        // bound concrete value. Prevents false "not an int list" failures on
        // graphs where torch.export emits sym_ints for what is dimensionally
        // a static parameter (kernel sizes, etc. with dynamic batch).
        if let Some(entries) = arg.as_sym_ints() {
            let mut out = Vec::with_capacity(entries.len());
            for entry in entries {
                let v = match entry {
                    SymIntEntry::Int(i) => Some(i.as_int),
                    SymIntEntry::Name(s) => self
                        .resolve_sym_int(&s.as_name)
                        .and_then(|e| e.to_usize().map(|u| u as i64)),
                };
                match v {
                    Some(n) => out.push(n),
                    None => {
                        anyhow::bail!(
                            "Input {idx} of {} contains an unresolved sym_int entry",
                            node.target
                        )
                    }
                }
            }
            return Ok(out);
        }
        arg.as_ints()
            .map(|v| v.to_vec())
            .with_context(|| format!("Input {idx} of {} is not int list: {:?}", node.target, arg))
    }

    pub(crate) fn get_expr_arg(&self, node: &Node, idx: usize) -> Result<Expression> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        self.resolve_arg_as_expression(arg).with_context(|| {
            format!(
                "Input {idx} of {} cannot be resolved to Expression: {:?}",
                node.target, arg
            )
        })
    }

    pub(crate) fn get_exprs_arg(&self, node: &Node, idx: usize) -> Result<Vec<Expression>> {
        use crate::pt2_schema::SymIntEntry;
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        if let Some(ints) = arg.as_ints() {
            return Ok(ints.iter().map(|&v| Expression::from(v)).collect());
        }
        if let Some(entries) = arg.as_sym_ints() {
            return entries
                .iter()
                .map(|entry| match entry {
                    SymIntEntry::Int(i) => Ok(Expression::from(i.as_int)),
                    SymIntEntry::Name(s) => self
                        .resolve_sym_int(&s.as_name)
                        .with_context(|| format!("Cannot resolve sym_int: {}", s.as_name)),
                })
                .collect();
        }
        anyhow::bail!(
            "Input {idx} of {} is not int list or sym_int list: {:?}",
            node.target,
            arg
        )
    }

    pub(crate) fn get_bool_arg(&self, node: &Node, idx: usize) -> Result<bool> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        arg.as_bool()
            .with_context(|| format!("Input {idx} of {} is not a bool: {:?}", node.target, arg))
    }

    pub(crate) fn tensor_meta_to_shape(&self, meta: &TensorMeta) -> Result<Vec<Expression>> {
        meta.sizes
            .iter()
            .map(|s| self.dim_size_to_expr(s))
            .collect()
    }

    pub(crate) fn dim_size_to_expr(&self, dim: &DimSize) -> Result<Expression> {
        match dim {
            DimSize::Int(i) => Ok(Expression::from(i.as_int)),
            DimSize::Expr(e) => self.resolve_expr_value(&e.as_expr).with_context(|| {
                format!(
                    "Cannot resolve symbolic dimension expression: {}",
                    e.as_expr.expr_str
                )
            }),
        }
    }

    pub(crate) fn resolve_sym_int(&self, name: &str) -> Option<Expression> {
        let sym_int_values = &self.parsed.program.graph_module.graph.sym_int_values;
        if let Some(val) = sym_int_values.get(name) {
            if let Some(expr_str) = val
                .get("as_expr")
                .and_then(|e| e.get("expr_str"))
                .and_then(|s| s.as_str())
                && let Some(expr) = self.resolve_expr_str(expr_str)
            {
                return Some(expr);
            }
            if let Some(hint) = val
                .get("as_expr")
                .and_then(|e| e.get("hint"))
                .and_then(|h| h.get("as_int"))
                .and_then(|v| v.as_i64())
            {
                return Some(Expression::from(hint));
            }
        }
        None
    }

    pub(crate) fn resolve_arg_as_expression(&self, arg: &Argument) -> Option<Expression> {
        if let Some(v) = arg.as_int() {
            return Some(Expression::from(v));
        }
        if let Some(name) = arg.as_sym_int_name() {
            return self.resolve_sym_int(name);
        }
        if let Argument::Expr(e) = arg {
            return self.resolve_expr_value(&e.as_expr);
        }
        None
    }

    pub(crate) fn resolve_expr_str(&self, expr_str: &str) -> Option<Expression> {
        parse_sympy_expr_with_ranges(expr_str, &self.sym_map.sym_to_char, &self.sym_map.ranges)
            .or_else(|| {
                crate::pt2_parser::extract_symbol_name_pub(expr_str)
                    .and_then(|sym| self.sym_map.sym_to_char.get(&sym).copied())
                    .map(Expression::from)
            })
    }

    pub(crate) fn resolve_expr_value(&self, expr: &ExprValue) -> Option<Expression> {
        self.resolve_expr_str(&expr.expr_str).or_else(|| {
            expr.hint
                .as_ref()
                .and_then(|h| h.as_int())
                .map(Expression::from)
        })
    }
}
