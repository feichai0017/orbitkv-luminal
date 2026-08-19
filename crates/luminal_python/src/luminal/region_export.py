"""Strict ``torch.export`` normalization for compiler-owned FX regions."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import torch
from torch import fx

from .pt2 import (
    _build_dynamic_shapes_from_gm,
    _decomp_table,
    _drop_dead_data_dependent_ops,
    _drop_input_guards,
    _export_kwargs,
    _strip_symint_placeholders,
)


def export_region(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    dynamic_range: tuple[int, int] | None = None,
) -> torch.export.ExportedProgram:
    """Normalize an FX region without storage access or static fallback.

    Dynamo may expose a symbolic tensor dimension as a separate ``SymInt``
    argument.  Replace that argument with a tensor-size operation before
    re-exporting so the exported program has a tensor-only runtime signature.
    The caller's graph is never modified.
    """

    graph = copy.deepcopy(graph).eval()
    _strip_data_attr(graph)
    inputs, _, strip_ok = _strip_symint_placeholders(graph, list(example_inputs))
    if not strip_ok:
        raise RuntimeError(
            "cannot export region: a SymInt input could not be derived from "
            "a tensor dimension"
        )

    dynamic_shapes = _build_dynamic_shapes_from_gm(graph, dynamic_range)
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in inspect.signature(graph.forward).parameters.values()
    )
    if dynamic_shapes is not None and not has_varargs:
        dynamic_shapes = dynamic_shapes["args"]
    try:
        exported = torch.export.export(
            graph,
            tuple(inputs),
            dynamic_shapes=dynamic_shapes,
            **_export_kwargs(),
        )
        _drop_input_guards(exported)
        _drop_dead_data_dependent_ops(exported.graph_module)
        exported = exported.run_decompositions(_decomp_table())
        _set_range_constraints(exported, dynamic_range)
        return exported
    except Exception as error:
        raise RuntimeError(
            "torch.export failed for compiler-owned FX region: "
            f"{type(error).__name__}: {error}"
        ) from error


def _strip_data_attr(graph: fx.GraphModule) -> None:
    """Treat ``Tensor.data`` as identity inside an inference-only region.

    Dynamo represents ``tensor.data`` with the private ``_get_data_attr`` op.
    If that op reaches ``torch.export``, export may lift the aliased tensor as a
    constant instead of preserving the original region input. Luminal regions
    run without autograd, so the detached alias has the same observable value
    and storage semantics as its source tensor.
    """

    target = getattr(torch._C._autograd, "_get_data_attr", None)
    if target is None:
        return

    changed = False
    for node in list(graph.graph.nodes):
        if node.op != "call_function" or node.target is not target:
            continue
        if len(node.args) != 1 or node.kwargs:
            raise RuntimeError("unexpected _get_data_attr invocation")
        node.replace_all_uses_with(node.args[0])
        graph.graph.erase_node(node)
        changed = True

    if changed:
        graph.graph.lint()
        graph.recompile()


def _set_range_constraints(
    exported: torch.export.ExportedProgram,
    dynamic_range: tuple[int, int] | None,
) -> None:
    from torch.utils._sympy.value_ranges import ValueRanges

    used_symbols = set()
    for node in exported.graph_module.graph.nodes:
        for value in torch.utils._pytree.tree_leaves(node.meta.get("val")):
            values = (
                (*value.shape, *value.stride())
                if isinstance(value, torch.Tensor)
                else (value,)
            )
            for item in values:
                if isinstance(item, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                    used_symbols.update(item.node.expr.free_symbols)
    for symbol in list(exported.range_constraints):
        if symbol not in used_symbols:
            del exported.range_constraints[symbol]
        elif dynamic_range is not None and dynamic_range[0] != dynamic_range[1]:
            bounds = ValueRanges(max(2, dynamic_range[0]), dynamic_range[1])
            exported.range_constraints[symbol] &= bounds
