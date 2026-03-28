"""Helpers for caching Llama 3.1-8B test artifacts under pytest cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig, LlamaForCausalLM

# This code is designed to be deleted.
# We should not need to cache pt2 or onnx files to get reasonable compile performance.

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
INPUT_IDS_LIST = [1, 2, 3, 4]
INPUT_IDS = torch.tensor([INPUT_IDS_LIST], dtype=torch.long)
ARTIFACT_SCHEMA_VERSION = 1
ONNX_OPSET_VERSION = 20
PT2_STRICT = False

_REF_LOGITS_META_KEY = "luminal_python/llama38b_artifacts/ref_logits_v1"
_ONNX_META_KEY = "luminal_python/llama38b_artifacts/onnx_v1"
_PT2_META_KEY = "luminal_python/llama38b_artifacts/pt2_v1"


@dataclass(frozen=True)
class Llama38BArtifactBundle:
    ref_logits_path: Path
    onnx_path: Path | None = None
    pt2_path: Path | None = None
    weights_path: Path | None = None


def ensure_onnx_bundle(cache, cache_dir: Path) -> Llama38BArtifactBundle:
    """Ensure ONNX artifacts and shared reference logits exist in pytest cache."""
    ref_logits_path = cache_dir / "ref_logits.pt"
    onnx_dir = cache_dir / "onnx"
    onnx_path = onnx_dir / "llama38b.onnx"

    ref_metadata = _ref_logits_metadata()
    onnx_metadata = _onnx_metadata()
    needs_ref_logits = cache.get(_REF_LOGITS_META_KEY, None) != ref_metadata or not (
        ref_logits_path.is_file()
    )
    needs_onnx = cache.get(_ONNX_META_KEY, None) != onnx_metadata or not (
        onnx_path.is_file()
    )

    if needs_ref_logits or needs_onnx:
        print(f"Generating cached ONNX artifacts for {MODEL_ID} in {cache_dir}")
        if needs_ref_logits:
            ref_logits_path.unlink(missing_ok=True)
        if needs_onnx:
            shutil.rmtree(onnx_dir, ignore_errors=True)
            onnx_dir.mkdir(parents=True, exist_ok=True)

        model = _load_model()

        if needs_ref_logits:
            ref_logits = _compute_ref_logits(model)
            torch.save(ref_logits, ref_logits_path)
            cache.set(_REF_LOGITS_META_KEY, ref_metadata)

        if needs_onnx:
            torch.onnx.export(
                model,
                (INPUT_IDS,),
                str(onnx_path),
                opset_version=ONNX_OPSET_VERSION,
                input_names=["input_ids"],
                output_names=["logits"],
            )
            cache.set(_ONNX_META_KEY, onnx_metadata)

    return Llama38BArtifactBundle(ref_logits_path=ref_logits_path, onnx_path=onnx_path)


def ensure_pt2_bundle(cache, cache_dir: Path) -> Llama38BArtifactBundle:
    """Ensure PT2 artifacts and shared reference logits exist in pytest cache."""
    ref_logits_path = cache_dir / "ref_logits.pt"
    pt2_dir = cache_dir / "pt2"
    pt2_path = pt2_dir / "llama38b.pt2"
    weights_path = pt2_dir / "llama38b_weights.safetensors"

    ref_metadata = _ref_logits_metadata()
    pt2_metadata = _pt2_metadata()
    needs_ref_logits = cache.get(_REF_LOGITS_META_KEY, None) != ref_metadata or not (
        ref_logits_path.is_file()
    )
    needs_pt2 = cache.get(_PT2_META_KEY, None) != pt2_metadata or not (
        pt2_path.is_file() and weights_path.is_file()
    )

    if needs_ref_logits or needs_pt2:
        print(f"Generating cached PT2 artifacts for {MODEL_ID} in {cache_dir}")
        if needs_ref_logits:
            ref_logits_path.unlink(missing_ok=True)
        if needs_pt2:
            shutil.rmtree(pt2_dir, ignore_errors=True)
            pt2_dir.mkdir(parents=True, exist_ok=True)

        model = _load_model()

        if needs_ref_logits:
            ref_logits = _compute_ref_logits(model)
            torch.save(ref_logits, ref_logits_path)
            cache.set(_REF_LOGITS_META_KEY, ref_metadata)

        if needs_pt2:
            exported_program = torch.export.export(
                model, (INPUT_IDS,), strict=PT2_STRICT
            )
            torch.export.save(exported_program, str(pt2_path))

            state_dict = {
                key: value.float().clone()
                for key, value in exported_program.state_dict.items()
            }
            save_file(state_dict, str(weights_path))
            cache.set(_PT2_META_KEY, pt2_metadata)

    return Llama38BArtifactBundle(
        ref_logits_path=ref_logits_path,
        pt2_path=pt2_path,
        weights_path=weights_path,
    )


def _load_model() -> LlamaForCausalLM:
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.use_cache = False
    config._attn_implementation = "eager"

    return LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float32,
    ).eval()


def _compute_ref_logits(model: LlamaForCausalLM) -> torch.Tensor:
    with torch.no_grad():
        return model(INPUT_IDS).logits.clone()


def _ref_logits_metadata() -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "input_ids": INPUT_IDS_LIST,
        "device": "cpu",
        "torch_dtype": "float32",
        "use_cache": False,
        "attn_implementation": "eager",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def _onnx_metadata() -> dict[str, object]:
    return {
        **_ref_logits_metadata(),
        "artifact_type": "onnx",
        "opset_version": ONNX_OPSET_VERSION,
    }


def _pt2_metadata() -> dict[str, object]:
    return {
        **_ref_logits_metadata(),
        "artifact_type": "pt2",
        "strict": PT2_STRICT,
    }
