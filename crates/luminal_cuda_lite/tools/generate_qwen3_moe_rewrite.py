"""Generate Qwen3-MoE rewrites from real torch.compile PT2 translations.

FP16 and BF16 do not translate to the same Luminal topology, so this tool
accepts one symbolic-token capture for each dtype and emits two exact staged
matches. Both replace the post-router graph with the same Qwen3Moe HostOp.
The router projection itself deliberately remains outside the match.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


STAGE_SIZE = 20
BOUNDARY = {
    0: "?hidden_states",
    2: "?gate_up_proj",
    3: "?down_proj",
    5: "?router_logits",
}


@dataclass(frozen=True)
class Capture:
    name: str
    dtype: str
    dynamic_tokens: bool
    source: str
    nodes: dict[int, str]
    dependencies: dict[int, set[int]]
    first_node: int
    last_node: int
    boundary_first_use: dict[str, int]


def _all_nodes(source: str) -> dict[int, str]:
    nodes = {}
    for line in source.splitlines():
        match = re.fullmatch(r"\(let t(\d+) (.*)\)", line)
        if match is not None:
            nodes[int(match.group(1))] = match.group(2)
    return nodes


def _transform(expression: str, first_node: int, last_node: int) -> str:
    expression = expression.replace('(MVar "a")', "?tokens")
    for index, variable in BOUNDARY.items():
        expression = re.sub(rf"\bt{index}\b", variable, expression)
    expression = re.sub(
        r"\bt(\d+)\b",
        lambda match: f"?t{match.group(1)}"
        if first_node <= int(match.group(1)) <= last_node
        else match.group(0),
        expression,
    )
    return expression


def parse_capture(
    path: Path, expected_dtype: str, *, dynamic_tokens: bool
) -> Capture:
    source = path.read_text(encoding="utf-8")
    all_nodes = _all_nodes(source)
    if not all_nodes:
        raise ValueError(f"{path}: no PT2 nodes found")

    input_dtype = re.fullmatch(
        r'\(Input 0 "[^"]+" \((Bf16|F16)\)\)', all_nodes[0]
    )
    if input_dtype is None or input_dtype.group(1) != expected_dtype:
        raise ValueError(f"{path}: expected Input 0 dtype {expected_dtype}")

    output_index = max(all_nodes)
    output = re.fullmatch(r"\(Output t(\d+) \d+ false\)", all_nodes[output_index])
    if output is None:
        raise ValueError(f"{path}: expected the final node to be the model Output")

    # Fuse through the reduction across the eight selected experts, whose
    # result is the ABI's flat [tokens, hidden] output. Do not include the
    # standalone module's trailing view/reshape representation: torch.compile
    # can eliminate that representation when the block is embedded in a full
    # transformer layer and feed this reduction directly to the residual add.
    # Matching the semantic MoE output keeps the rewrite independent of its
    # surrounding consumer while preserving any downstream layout operation.
    output_producer = int(output.group(1))
    moe_outputs = [
        index
        for index, expression in all_nodes.items()
        if index <= output_producer
        and expression.startswith("(Op (Sum (ECons ")
        and ") (MNum 8) (ECons" in expression
        and "(ECons (MNum 2048) (ENil))" in expression
    ]
    if not moe_outputs:
        raise ValueError(f"{path}: could not find the final top-8 expert reduction")
    last_node = max(moe_outputs)
    first_node = 6

    has_symbolic_tokens = '(MVar "a")' in source
    if has_symbolic_tokens != dynamic_tokens:
        expected = "symbolic" if dynamic_tokens else "static token-1"
        raise ValueError(f"{path}: expected a {expected} capture")
    if not dynamic_tokens and "(MNum 1)" not in all_nodes[5]:
        raise ValueError(f"{path}: static capture does not have one token")

    # Enforce the ABI boundary encoded below: t0 is hidden states, t1 is the
    # router weight, t2/t3 are expert weights, and t4/t5 are the router GEMM.
    if "ICons t0 (ICons t1" not in all_nodes.get(4, ""):
        raise ValueError(f"{path}: t4 is not the expected router multiply")
    if "(ICons t4 (INil))" not in all_nodes.get(5, ""):
        raise ValueError(f"{path}: t5 is not the expected router reduction")

    nodes = {}
    dependencies = {}
    boundary_first_use = {}
    for index in range(first_node, last_node + 1):
        raw = all_nodes.get(index)
        if raw is None:
            raise ValueError(f"{path}: missing post-router node t{index}")
        for boundary_index, variable in BOUNDARY.items():
            if re.search(rf"\bt{boundary_index}\b", raw):
                boundary_first_use.setdefault(variable, index)
        nodes[index] = _transform(raw, first_node, last_node)
        dependencies[index] = {
            int(dependency)
            for dependency in re.findall(r"\bt(\d+)\b", raw)
            if first_node <= int(dependency) <= last_node
        }

    missing = set(BOUNDARY.values()) - set(boundary_first_use)
    if missing:
        raise ValueError(f"{path}: unused ABI boundaries: {sorted(missing)}")

    return Capture(
        name=f"{expected_dtype.lower()}_{'dynamic' if dynamic_tokens else 'token1'}",
        dtype=expected_dtype,
        dynamic_tokens=dynamic_tokens,
        source=source,
        nodes=nodes,
        dependencies=dependencies,
        first_node=first_node,
        last_node=last_node,
        boundary_first_use=boundary_first_use,
    )


def _live_nodes(capture: Capture, cut: int) -> list[int]:
    return [
        node
        for node in range(capture.first_node, cut + 1)
        if any(
            node in capture.dependencies[later]
            for later in range(cut + 1, capture.last_node + 1)
        )
    ]


def _relation_args(capture: Capture, cut: int, live: list[int]) -> list[str]:
    boundaries = [
        variable
        for variable in BOUNDARY.values()
        if capture.boundary_first_use[variable] <= cut
    ]
    tokens = ["?tokens"] if capture.dynamic_tokens else []
    return [*boundaries, *tokens, *[f"?t{node}" for node in live]]


def _generate_capture(capture: Capture) -> tuple[list[str], list[str]]:
    stages = []
    start = capture.first_node
    while start <= capture.last_node:
        end = min(start + STAGE_SIZE - 1, capture.last_node)
        stages.append((start, end))
        start = end + 1

    declarations = []
    for stage_index, (_, end) in enumerate(stages[:-1]):
        sorts = [
            "IR"
            for variable in BOUNDARY.values()
            if capture.boundary_first_use[variable] <= end
        ]
        if capture.dynamic_tokens:
            sorts.append("Expression")
        sorts.extend("IR" for _ in _live_nodes(capture, end))
        declarations.append(
            f"(relation qwen3_moe_{capture.name}_stage_{stage_index} "
            f"({' '.join(sorts)}))"
        )

    rules = [
        f"; {capture.dtype}: PT2 nodes t{capture.first_node}..t{capture.last_node}",
    ]
    previous_live = []
    for stage_index, (start, end) in enumerate(stages):
        facts = []
        if stage_index > 0:
            previous_end = stages[stage_index - 1][1]
            args = " ".join(
                _relation_args(capture, previous_end, previous_live)
            )
            facts.append(
                f"        (qwen3_moe_{capture.name}_stage_{stage_index - 1} {args})"
            )
        facts.extend(
            f"        (= ?t{node} {capture.nodes[node]})"
            for node in range(start, end + 1)
        )

        if stage_index == len(stages) - 1:
            facts.extend(
                f"        (= ({capture.dtype}) (dtype {variable}))"
                for variable in BOUNDARY.values()
            )
            tokens = "?tokens" if capture.dynamic_tokens else "(MNum 1)"
            actions = [
                f"        (let ?qwen3_moe (Op (Qwen3Moe {tokens} ({capture.dtype}))",
                "            (ICons ?hidden_states (ICons ?router_logits (ICons ?gate_up_proj (ICons ?down_proj (INil)))))))",
                f"        (union ?t{capture.last_node} ?qwen3_moe)",
                # Qwen3Moe's dtype field is also its C-ABI storage contract.
                # The matched reference reduction may already have installed
                # F32 in dtype(IR). Subsuming the reduction does not remove
                # that function-table row, and dtype's `:merge new` is not a
                # lattice for conflicting dtype values. Delete the stale row
                # before stamping the unioned e-class with the model-width
                # F16/BF16 storage that the fused C ABI actually writes.
                f"        (delete (dtype ?qwen3_moe))",
                f"        (set (dtype ?qwen3_moe) ({capture.dtype}))",
                f"        (subsume {capture.nodes[capture.last_node]})",
            ]
        else:
            current_live = _live_nodes(capture, end)
            args = " ".join(_relation_args(capture, end, current_live))
            actions = [
                f"        (qwen3_moe_{capture.name}_stage_{stage_index} {args})"
            ]

        rules.extend(
            [
                f"; Stage {stage_index}: PT2 nodes t{start}..t{end}",
                "(rule",
                "    (",
                *facts,
                "    )",
                "    (",
                *actions,
                "    )",
                f'    :name "Qwen3-MoE {capture.name} post-router stage {stage_index}"',
                "    :ruleset glumoe",
                ")",
            ]
        )
        previous_live = _live_nodes(capture, end)

    return rules, declarations


def generate(
    bf16_dynamic_source: Path,
    f16_dynamic_source: Path,
    bf16_token1_source: Path,
    f16_token1_source: Path,
) -> tuple[str, str]:
    captures = [
        parse_capture(bf16_dynamic_source, "Bf16", dynamic_tokens=True),
        parse_capture(f16_dynamic_source, "F16", dynamic_tokens=True),
        parse_capture(bf16_token1_source, "Bf16", dynamic_tokens=False),
        parse_capture(f16_token1_source, "F16", dynamic_tokens=False),
    ]
    rules = [
        "; @generated by tools/generate_qwen3_moe_rewrite.py; do not hand edit.",
        "; Exact torch.compile PT2 matches for the Qwen3-30B-A3B block.",
        "; Router GEMM t4/t5 intentionally remains outside the fused HostOp.",
    ]
    declarations = [
        "; @generated by tools/generate_qwen3_moe_rewrite.py; do not hand edit.",
        "; Relations split exact PT2 matches into bounded joins.",
    ]
    for capture in captures:
        capture_rules, capture_declarations = _generate_capture(capture)
        rules.extend(capture_rules)
        declarations.extend(capture_declarations)
    return "\n".join(rules) + "\n", "\n".join(declarations) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16_dynamic_source", type=Path)
    parser.add_argument("f16_dynamic_source", type=Path)
    parser.add_argument("bf16_token1_source", type=Path)
    parser.add_argument("f16_token1_source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="also preserve the two source captures as rewrite fixtures",
    )
    args = parser.parse_args()

    rewrite, declarations = generate(
        args.bf16_dynamic_source,
        args.f16_dynamic_source,
        args.bf16_token1_source,
        args.f16_token1_source,
    )
    args.destination.write_text(rewrite, encoding="utf-8")
    args.destination.with_name("qwen3_moe_declarations.egg").write_text(
        declarations, encoding="utf-8"
    )
    if args.fixture_dir is not None:
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        for source, name in (
            (
                args.bf16_dynamic_source,
                "qwen3_moe_torch_compile_bf16_dynamic.egg",
            ),
            (
                args.f16_dynamic_source,
                "qwen3_moe_torch_compile_f16_dynamic.egg",
            ),
            (
                args.bf16_token1_source,
                "qwen3_moe_torch_compile_bf16_token1.egg",
            ),
            (
                args.f16_token1_source,
                "qwen3_moe_torch_compile_f16_token1.egg",
            ),
        ):
            (args.fixture_dir / name).write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )
            root = source.with_suffix(".root")
            (args.fixture_dir / name.replace(".egg", ".root")).write_text(
                root.read_text(encoding="utf-8"), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
