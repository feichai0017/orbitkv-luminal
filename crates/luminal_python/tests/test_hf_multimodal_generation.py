"""Hugging Face multimodal image-text-to-text smoke tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from luminal import luminal_backend

MODEL_ID = "google/gemma-3-4b-it"
_CUDA_BACKEND_AVAILABLE = (
    os.getenv("LUMINAL_BACKEND", "native").lower() == "cuda"
    and torch.cuda.is_available()
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _CUDA_BACKEND_AVAILABLE,
        reason="Gemma 3 multimodal tests require the CUDA backend",
    ),
]


@dataclass(frozen=True)
class HFMultimodalCase:
    case_id: str
    messages_builder: Callable[[Path], list[dict]]
    max_new_tokens: int
    expects_pixel_values: bool


@dataclass(frozen=True)
class Gemma3MultimodalBundle:
    model: AutoModelForImageTextToText
    processor: AutoProcessor
    device: torch.device
    dtype: torch.dtype


def _build_text_only_messages(_: Path) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a concise assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "In one short sentence, explain what a compiler does.",
                }
            ],
        },
    ]


def _build_image_to_text_messages(image_path: Path) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        },
    ]


MULTIMODAL_CASES = [
    HFMultimodalCase(
        case_id="chat_text_only",
        messages_builder=_build_text_only_messages,
        max_new_tokens=12,
        expects_pixel_values=False,
    ),
    HFMultimodalCase(
        case_id="image_to_text",
        messages_builder=_build_image_to_text_messages,
        max_new_tokens=16,
        expects_pixel_values=True,
    ),
]


def _model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _set_greedy_generation(model) -> None:
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def _move_to_device(
    encoded: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
            continue

        moved = value.to(device)
        if moved.is_floating_point():
            moved = moved.to(dtype=dtype)
        result[key] = moved
    return result


def _encode_case(
    bundle: Gemma3MultimodalBundle,
    case: HFMultimodalCase,
    image_path: Path,
) -> dict[str, torch.Tensor]:
    encoded = bundle.processor.apply_chat_template(
        case.messages_builder(image_path),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    encoded = _move_to_device(dict(encoded), bundle.device, bundle.dtype)

    if case.expects_pixel_values:
        assert "pixel_values" in encoded
    assert "input_ids" in encoded
    return encoded


def _generate_kwargs(
    bundle: Gemma3MultimodalBundle,
    encoded: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> dict:
    tokenizer = bundle.processor.tokenizer
    return dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def _logits_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 5e-2
    if dtype == torch.float16:
        return 1e-2
    return 1e-3


@pytest.fixture(scope="module")
def gemma3_multimodal_bundle() -> Gemma3MultimodalBundle:
    device = torch.device("cuda")
    dtype = _model_dtype(device)

    config = AutoConfig.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (
        AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            config=config,
            torch_dtype=dtype,
        )
        .eval()
        .to(device)
    )
    _set_greedy_generation(model)
    return Gemma3MultimodalBundle(
        model=model, processor=processor, device=device, dtype=dtype
    )


class TestHFMultimodalGeneration:
    @pytest.mark.parametrize("case", MULTIMODAL_CASES, ids=lambda case: case.case_id)
    def test_generate_matches_eager(
        self,
        case: HFMultimodalCase,
        gemma3_multimodal_bundle: Gemma3MultimodalBundle,
        hf_multimodal_image_path: Path,
    ):
        encoded = _encode_case(gemma3_multimodal_bundle, case, hf_multimodal_image_path)
        kwargs = _generate_kwargs(
            gemma3_multimodal_bundle, encoded, case.max_new_tokens
        )

        with torch.no_grad():
            eager_output = gemma3_multimodal_bundle.model.generate(**kwargs)

        compiled_model = torch.compile(
            gemma3_multimodal_bundle.model, backend=luminal_backend
        )
        with torch.no_grad():
            compiled_output = compiled_model.generate(**kwargs)

        torch.testing.assert_close(compiled_output, eager_output)

    @pytest.mark.parametrize("case", MULTIMODAL_CASES, ids=lambda case: case.case_id)
    def test_forward_logits_match_eager(
        self,
        case: HFMultimodalCase,
        gemma3_multimodal_bundle: Gemma3MultimodalBundle,
        hf_multimodal_image_path: Path,
    ):
        encoded = _encode_case(gemma3_multimodal_bundle, case, hf_multimodal_image_path)

        with torch.no_grad():
            eager_out = gemma3_multimodal_bundle.model(**encoded)

        compiled_model = torch.compile(
            gemma3_multimodal_bundle.model, backend=luminal_backend
        )
        with torch.no_grad():
            compiled_out = compiled_model(**encoded)

        atol = _logits_tolerance(gemma3_multimodal_bundle.dtype)
        torch.testing.assert_close(
            compiled_out.logits,
            eager_out.logits,
            atol=atol,
            rtol=1e-3,
        )
