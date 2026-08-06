"""Export and inspect a Hugging Face Qwen3-MoE sparse MoE block on CPU.

This script captures only ``Qwen3MoeSparseMoeBlock`` rather than the complete
causal language model. It uses the production Qwen3-30B-A3B configuration and
BF16 random parameters/activations, saves the decomposed PT2 artifact, and asks
Luminal's translation-only APIs for the pre-backend HLIR DOT and egglog
representations. No CUDA backend or GPU is involved.

Run from ``crates/luminal_python``:

    uv run python examples/export_qwen3_moe_block.py --tokens 16

Inspect an existing PT2 artifact without rebuilding its 1.1 GB random weights:

    uv run python examples/export_qwen3_moe_block.py \
        --existing-pt2 ../../target/qwen3_moe/qwen3_moe_block_t16.pt2

Artifacts default to ``<repo>/target/qwen3_moe`` (already ignored by git).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from luminal import translate_pt2_to_dot, translate_pt2_to_egglog
from luminal.pt2 import _decomp_table, _export_kwargs


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "target" / "qwen3_moe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-pt2",
        type=Path,
        help="Skip export and write DOT/egglog sidecars for this PT2 file.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=16,
        help="Concrete token count used to seed export (default: 16).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size (default: 1).",
    )
    parser.add_argument(
        "--dynamic-tokens",
        action="store_true",
        help="Export the sequence dimension as symbolic instead of static.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Hugging Face config to load (default: {MODEL_ID}).",
    )
    return parser.parse_args()


def load_sparse_moe_block(model_id: str) -> tuple[torch.nn.Module, object]:
    """Instantiate the production sparse MoE block with random BF16 weights."""
    from transformers import AutoConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    config = AutoConfig.from_pretrained(model_id)
    # `PreTrainedModel.__init__` normally resolves this private dispatch knob
    # to grouped_mm for supported MoE models. We instantiate the block directly,
    # so reproduce that full-model choice explicitly; leaving it as None selects
    # Qwen's eager, data-dependent Python loop, which cannot be exported.
    config._experts_implementation = "grouped_mm"
    block = Qwen3MoeSparseMoeBlock(config).eval().to(dtype=torch.bfloat16)
    return block, config


def export_block(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    dynamic_tokens: bool,
) -> torch.export.ExportedProgram:
    dynamic_shapes = None
    if dynamic_tokens:
        token_dim = torch.export.Dim("tokens", min=1)
        dynamic_shapes = ({1: token_dim},)

    exported = torch.export.export(
        block,
        (hidden_states,),
        dynamic_shapes=dynamic_shapes,
        **_export_kwargs(),
    )
    return exported.run_decompositions(_decomp_table())


def write_graph_artifacts(pt2_path: Path) -> tuple[Path, Path, Path]:
    """Write human-readable DOT and exact egglog sidecars for a PT2 graph."""
    dot_path = pt2_path.with_suffix(".dot")
    egg_path = pt2_path.with_suffix(".egg")
    root_path = pt2_path.with_suffix(".root")

    dot_path.write_text(translate_pt2_to_dot(str(pt2_path)), encoding="utf-8")
    egg_program, root = translate_pt2_to_egglog(str(pt2_path))
    egg_path.write_text(egg_program, encoding="utf-8")
    root_path.write_text(f"{root}\n", encoding="utf-8")
    return dot_path, egg_path, root_path


def main() -> None:
    args = parse_args()
    if args.existing_pt2 is not None:
        pt2_path = args.existing_pt2.expanduser().resolve()
        if not pt2_path.is_file():
            raise FileNotFoundError(pt2_path)
        dot_path, egg_path, root_path = write_graph_artifacts(pt2_path)
        print(
            "Inspected existing Qwen3-MoE PT2 graph\n"
            f"  PT2:         {pt2_path}\n"
            f"  Luminal DOT: {dot_path}\n"
            f"  Egglog:      {egg_path}\n"
            f"  Root:        {root_path}"
        )
        return

    if args.tokens < 1 or args.batch_size < 1:
        raise ValueError("--tokens and --batch-size must both be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"t{args.tokens}"
    if args.dynamic_tokens:
        suffix += "_dynamic"
    pt2_path = args.output_dir / f"qwen3_moe_block_{suffix}.pt2"

    with torch.no_grad():
        block, config = load_sparse_moe_block(args.model_id)
        hidden_states = torch.randn(
            args.batch_size,
            args.tokens,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cpu",
        )
        exported = export_block(
            block,
            hidden_states,
            dynamic_tokens=args.dynamic_tokens,
        )

    torch.export.save(exported, pt2_path)
    dot_path, egg_path, root_path = write_graph_artifacts(pt2_path)

    print(
        "Exported Qwen3-MoE sparse block\n"
        f"  model:       {args.model_id}\n"
        f"  dtype:       {hidden_states.dtype}\n"
        f"  input:       {tuple(hidden_states.shape)}\n"
        f"  experts:     {config.num_experts}\n"
        f"  top-k:       {config.num_experts_per_tok}\n"
        f"  dynamic T:   {args.dynamic_tokens}\n"
        f"  PT2:         {pt2_path}\n"
        f"  Luminal DOT: {dot_path}\n"
        f"  Egglog:      {egg_path}\n"
        f"  Root:        {root_path}"
    )


if __name__ == "__main__":
    main()
