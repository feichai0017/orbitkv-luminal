from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import test_models as tm

from luminal import luminal_backend


Args = tuple[torch.Tensor, ...]
Kwargs = dict[str, torch.Tensor]
InputFactory = Callable[[torch.device], tuple[Args, Kwargs]]


@dataclass(frozen=True)
class UnaryCase:
    id: str
    model_factory: Callable[[], torch.nn.Module]
    input_factory: InputFactory


UNARY_CASES: list[UnaryCase] = [
    UnaryCase(
        id="sigmoid",
        model_factory=tm.SigmoidTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="sigmoid_in_expression",
        model_factory=tm.SigmoidInExpressionModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),
    UnaryCase(
        id="tanh",
        model_factory=tm.TanhTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="tanh_in_expression",
        model_factory=tm.TanhInExpressionModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),
    UnaryCase(
        id="relu",
        model_factory=tm.ReluTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="relu_all_negative",
        model_factory=tm.ReluAllNegativeModel,
        input_factory=lambda device: ((-torch.rand((5, 5), device=device),), {}),
    ),
    UnaryCase(
        id="relu_in_expression",
        model_factory=tm.ReluInExpressionModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="abs",
        model_factory=tm.AbsTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="abs_all_negative",
        model_factory=tm.AbsAllNegativeModel,
        input_factory=lambda device: ((-torch.rand((5, 5), device=device),), {}),
    ),
    UnaryCase(
        id="abs_in_expression",
        model_factory=tm.AbsInExpressionModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="neg",
        model_factory=tm.NegTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="neg_all_positive",
        model_factory=tm.NegAllPositiveModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),
    UnaryCase(
        id="neg_in_expression",
        model_factory=tm.NegInExpressionModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 2 - 1,), {}),
    ),
    UnaryCase(
        id="clip",
        model_factory=tm.ClipTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 4 - 2,), {}),
    ),
    UnaryCase(
        id="clip_min_only",
        model_factory=tm.ClipMinOnlyTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 4 - 2,), {}),
    ),
    UnaryCase(
        id="clip_max_only",
        model_factory=tm.ClipMaxOnlyTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 4 - 2,), {}),
    ),
]


class TestUnaryOps:
    @pytest.mark.parametrize("case", UNARY_CASES, ids=lambda case: case.id)
    def test_matches_eager(self, case: UnaryCase, device: torch.device) -> None:
        model = case.model_factory().to(device)
        compiled_model = torch.compile(model, backend=luminal_backend)
        args, kwargs = case.input_factory(device)
        torch.testing.assert_close(
            compiled_model(*args, **kwargs),
            model(*args, **kwargs),
            atol=1e-5,
            rtol=1e-5,
        )
