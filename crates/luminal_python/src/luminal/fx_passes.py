"""Pre-export FX passes that shrink pathological graphs before luminal sees them.

Dynamo fully unrolls Python loops in model code. When a loop is a recognizable
algorithm, the unrolled form is both enormous and unnecessary: Qwen3.5/3.6's
gated delta-net (`torch_chunk_gated_delta_rule`) spends a 63-iteration
sequential loop computing a unit-lower-triangular inverse, which accounts for
~80% of the exported graph (~40k nodes on the real 4B config). These passes
recognize such motifs and emit the closed form instead.

Running before `torch.export` means the giant graph is never built at all —
neither exported, nor translated, nor turned into an e-graph.
"""

from __future__ import annotations

import operator

import torch

__all__ = ["apply_pre_export_passes"]


def _tuple_index(idx, expect_row, i):
    """Match `(..., i, :i)` (row) or `(..., :i, :i)` (block) index tuples."""
    if not (isinstance(idx, tuple) and len(idx) == 3 and idx[0] is Ellipsis):
        return False
    first, second = idx[1], idx[2]
    if not (isinstance(second, slice) and second.start is None and second.stop == i and second.step is None):
        return False
    if expect_row:
        return first == i
    return isinstance(first, slice) and first.start is None and first.stop == i and first.step is None


def _unwrap_clone(node):
    """`x.clone()` is semantically transparent for our purposes."""
    while node is not None and node.op == "call_method" and node.target == "clone":
        node = node.args[0]
    return node


def _is_strictly_lower(node, depth=6):
    """Conservatively prove `node` is strictly lower triangular.

    The closed form below is only valid for a nilpotent (strictly lower
    triangular) operand. We accept the shapes HF's delta-rule produces:
    `masked_fill(triu(..., diagonal=0), 0)` possibly wrapped in
    sign/scale-preserving elementwise ops. Anything unproven is skipped, which
    just leaves the slow unrolled path intact.
    """
    seen = 0
    while node is not None and seen < depth:
        seen += 1
        if node.op == "call_method" and node.target in ("masked_fill", "masked_fill_"):
            mask = node.args[1]
            fill = node.args[2] if len(node.args) > 2 else node.kwargs.get("value")
            if fill != 0:
                return False
            mask = _unwrap_clone(mask)
            if mask is not None and (
                (mask.op == "call_method" and mask.target == "triu")
                or (mask.op == "call_function" and mask.target is torch.triu)
            ):
                diagonal = (
                    mask.args[1]
                    if len(mask.args) > 1
                    else mask.kwargs.get("diagonal", 0)
                )
                return diagonal == 0
            return False
        if node.op == "call_method" and node.target in ("tril",):
            diagonal = node.args[1] if len(node.args) > 1 else node.kwargs.get("diagonal", 0)
            return diagonal <= -1
        # sign/scale-preserving elementwise wrappers keep the zero pattern
        if node.op == "call_function" and node.target in (operator.neg, operator.mul, torch.neg, torch.mul):
            node = _unwrap_clone(node.args[0])
            continue
        if node.op == "call_method" and node.target in ("neg", "mul", "float", "to"):
            node = _unwrap_clone(node.args[0])
            continue
        return False
    return False


def _match_iteration(base, setitem, i):
    """Verify one iteration: `base[..., i, :i] = row + (row.unsqueeze(-1) * blk).sum(-2)`.

    Returns the set of nodes belonging to this iteration, or None.
    """
    if not _tuple_index(setitem.args[1], True, i):
        return None
    add = setitem.args[2]
    if not (add.op == "call_function" and add.target in (operator.add, torch.add)):
        return None
    row_a, summed = add.args
    if not (summed.op == "call_method" and summed.target == "sum" and summed.args[1] == -2):
        return None
    mul = summed.args[0]
    if not (mul.op == "call_function" and mul.target in (operator.mul, torch.mul)):
        return None
    unsq, blk = mul.args
    if not (unsq.op == "call_method" and unsq.target == "unsqueeze" and unsq.args[1] == -1):
        return None
    if _unwrap_clone(unsq.args[0]) is not _unwrap_clone(row_a):
        return None

    row_get = _unwrap_clone(row_a)
    blk_get = _unwrap_clone(blk)
    for node, is_row in ((row_get, True), (blk_get, False)):
        if not (
            node.op == "call_function"
            and node.target is operator.getitem
            and node.args[0] is base
            and _tuple_index(node.args[1], is_row, i)
        ):
            return None
    nodes = {setitem, add, summed, mul, unsq, row_get, blk_get}
    for n in (row_a, blk):  # the clone wrappers, if present
        if n.op == "call_method" and n.target == "clone":
            nodes.add(n)
    return nodes


def _find_forward_substitution(graph):
    """Find maximal `T = A + A@T` forward-substitution loops over a shared base."""
    matches = []
    for base in list(graph.nodes):
        setitems = [
            u
            for u in base.users
            if u.op == "call_function" and u.target is operator.setitem and u.args[0] is base
        ]
        if len(setitems) < 4:
            continue
        val = base.meta.get("example_value", base.meta.get("val"))
        if val is None or val.ndim < 2 or val.shape[-1] != val.shape[-2]:
            continue
        n = int(val.shape[-1])
        by_index = {}
        for s in setitems:
            idx = s.args[1]
            if isinstance(idx, tuple) and len(idx) == 3 and isinstance(idx[1], int):
                by_index[idx[1]] = s
        if sorted(by_index) != list(range(1, n)):
            continue  # must cover every row exactly once
        owned = set()
        ok = True
        for i in range(1, n):
            got = _match_iteration(base, by_index[i], i)
            if got is None:
                ok = False
                break
            owned |= got
        if not ok or not _is_strictly_lower(_unwrap_clone(base)):
            continue
        matches.append((base, owned, n))
    return matches


def _emit_neumann_inverse(graph, base, n):
    """Emit `sum_{k=1}^{n-1} A^k` == post-loop value, via repeated squaring.

    (I - A)^-1 = I + sum_k A^k for nilpotent A, and the loop leaves exactly the
    sum (the `+ eye` happens in model code afterwards). Uses the identity
    (I+M)(I+P) = I + M + P + M@P to avoid materializing an identity matrix:
    ~5 squarings + 5 matmuls for n=64, versus n-1 sequential iterations.
    """
    anchor = base
    created = set()

    def emit(target, args):
        nonlocal anchor
        with graph.inserting_after(anchor):
            node = graph.call_function(target, args)
        anchor = node
        created.add(node)
        return node

    acc, pw, k = base, base, 1
    while k * 2 < n:
        sq = emit(torch.matmul, (pw, pw))
        cross = emit(torch.matmul, (acc, sq))
        s1 = emit(operator.add, (acc, sq))
        acc = emit(operator.add, (s1, cross))
        pw, k = sq, k * 2
    return acc, created


def apply_pre_export_passes(gm: torch.fx.GraphModule) -> int:
    """Rewrite recognized motifs in-place. Returns the number of rewrites."""
    graph = gm.graph
    rewrites = 0
    for base, owned, n in _find_forward_substitution(graph):
        closed, created = _emit_neumann_inverse(graph, base, n)
        for node in list(base.users):
            if node in owned or node in created:
                continue
            node.replace_input_with(base, closed)
        for node in reversed(list(graph.nodes)):
            if node in owned:
                graph.erase_node(node)
        rewrites += 1
    if rewrites:
        graph.lint()
        gm.recompile()
    return rewrites
