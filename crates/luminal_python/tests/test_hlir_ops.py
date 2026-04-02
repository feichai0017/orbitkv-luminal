from dataclasses import dataclass
from typing import Callable

import torch
from test_models import (
    AddAddTestModel,
    AddConstantTestModel,
    AddTestModel,
    # And model
    AndTestModel,
    CastBoolToFloatModel,
    # Cast models
    CastDoubleToFloatModel,
    CastInComputationGraphModel,
    CastInt32ToFloatModel,
    CastInt64ToFloatModel,
    CastNegativeValuesModel,
    CastScalarValueModel,
    CastWith2DTensorModel,
    CeilInExpressionModel,
    CeilNegativeModel,
    # Ceil models
    CeilTestModel,
    ClipMinOnlyTestModel,
    # Clip models
    ClipTestModel,
    # Concat models
    ConcatAxis0Model,
    ConcatAxis1Model,
    ConcatInExpressionModel,
    ConcatNegativeAxisModel,
    ConcatThreeTensorsModel,
    Constant1DArrayFloatModel,
    Constant2DMatrixFloatModel,
    ConstantBoolConversionModel,
    ConstantFloat64ConversionModel,
    ConstantInt32ConversionModel,
    ConstantInt64ConversionModel,
    ConstantInt64RawDataModel,
    ConstantMultipleInGraphModel,
    ConstantNegativeValuesModel,
    ConstantRawDataFloatModel,
    # Constant models
    ConstantScalarFloatModel,
    ConstantZeroValueModel,
    CosTestModel,
    DivTestModel,
    # Equal models
    EqualBroadcastModel,
    EqualTestModel,
    EqualWithConstantModel,
    # Erf model
    ErfTestModel,
    # Expand model
    ExpandTestModel,
    FloorInExpressionModel,
    FloorNegativeModel,
    # Floor models
    FloorTestModel,
    # Gather models
    Gather1DModel,
    Gather2DAxis0Model,
    Gather2DAxis1Model,
    GatherConstantFoldModel,
    # GatherElements model
    GatherElementsTestModel,
    GatherElementsLargeTestModel,
    GatherEmbeddingModel,
    GatherNegativeIndicesModel,
    # Gemm model
    GemmTestModel,
    # GreaterOrEqual models
    GreaterOrEqualTestModel,
    GreaterOrEqualWithConstantModel,
    # Greater models
    GreaterTestModel,
    GreaterWithConstantModel,
    # IsNaN model
    IsNaNTestModel,
    # LayerNormalization model
    LayerNormTestModel,
    LessBroadcastModel,
    # LessOrEqual models
    LessOrEqualTestModel,
    LessOrEqualWithConstantModel,
    # Less models
    LessTestModel,
    LessWithConstantModel,
    LinearLayerModel,
    # Multi-op chain models
    ManualLayerNormModel,
    # MatMul models
    MatMul2DModel,
    MatMulBatchedModel,
    # Max models
    MaxTestModel,
    MaxWithConstantModel,
    # Min models
    MinTestModel,
    MinWithConstantModel,
    MLPBlockModel,
    ModByConstantModel,
    # Mod models
    ModTestModel,
    MulTestModel,
    # Not model
    NotTestModel,
    # OneHot model
    OneHotTestModel,
    # Or model
    OrTestModel,
    PowByConstantModel,
    # Pow models
    PowTestModel,
    ReduceMax3DAxis1Model,
    ReduceMaxAllAxesModel,
    # ReduceMax models
    ReduceMaxAxis0Model,
    ReduceMaxAxis1Model,
    ReduceMaxInExpressionModel,
    ReduceMaxKeepDimsModel,
    ReduceMaxMultiAxisKeepDimsModel,
    ReduceMaxMultiAxisModel,
    ReduceMaxNegativeAxisModel,
    ReduceMean3DAxis1Model,
    ReduceMeanAllAxesModel,
    # ReduceMean models
    ReduceMeanAxis0Model,
    ReduceMeanAxis1Model,
    ReduceMeanInExpressionModel,
    ReduceMeanKeepDimsModel,
    ReduceMeanMultiAxisKeepDimsModel,
    ReduceMeanMultiAxisModel,
    ReduceMeanNegativeAxisModel,
    ReduceMin3DAxis1Model,
    ReduceMinAllAxesModel,
    # ReduceMin models
    ReduceMinAxis0Model,
    ReduceMinAxis1Model,
    ReduceMinInExpressionModel,
    ReduceMinKeepDimsModel,
    ReduceMinMultiAxisKeepDimsModel,
    ReduceMinMultiAxisModel,
    ReduceMinNegativeAxisModel,
    ReduceSum3DAxis1Model,
    ReduceSumAllAxesModel,
    # ReduceSum models
    ReduceSumAxis0Model,
    ReduceSumAxis1Model,
    ReduceSumInExpressionModel,
    ReduceSumKeepDimsModel,
    ReduceSumMultiAxisKeepDimsModel,
    ReduceSumMultiAxisModel,
    ReduceSumNegativeAxisModel,
    # Activation function models
    ReluTestModel,
    Reshape3Dto2DModel,
    ReshapeAfterOpsModel,
    ReshapeInExpressionModel,
    ReshapeInferFirstDimModel,
    ReshapeInferLastDimModel,
    ReshapeRoundtripModel,
    ReshapeTo3DModel,
    # Reshape models
    ReshapeToFlatModel,
    ReshapeToMatrixModel,
    ScaledDotProductModel,
    ScatterElementsAxis0TestModel,
    # ScatterElements models
    ScatterElementsTestModel,
    # ScatterND model
    ScatterNDTestModel,
    ShapeReshapeBatchFlattenModel,
    ShapeReshapeKeepBatchModel,
    SigmoidTestModel,
    SinTestModel,
    SliceMultiAxisTestModel,
    # Slice models
    SliceTestModel,
    SoftmaxDim0TestModel,
    # Softmax models
    SoftmaxTestModel,
    # Split model
    SplitTestModel,
    SqrtTestModel,
    SqueezeAllDimsModel,
    # Squeeze models
    SqueezeAxisModel,
    SqueezeInExpressionModel,
    SqueezeMultipleAxesModel,
    SqueezeNegativeAxisModel,
    SubTestModel,
    TanhTestModel,
    TopKIndicesTestModel,
    # TopK models
    TopKValuesTestModel,
    Transpose3DTestModel,
    Transpose4DTestModel,
    TransposeInExpressionModel,
    TransposeReverseTestModel,
    TransposeTestModel,
    TrilDiagonalTestModel,
    # Trilu models
    TrilTestModel,
    TriuDiagonalTestModel,
    TriuTestModel,
    # Unsqueeze models
    UnsqueezeAxis0Model,
    UnsqueezeMiddleModel,
    # Where models
    WhereSelfSelectModel,
    WhereTestModel,
    WhereWithConstantModel,
    # Xor model
    XorTestModel,
)

from luminal import luminal_backend


Args = tuple[torch.Tensor, ...]
Kwargs = dict[str, torch.Tensor]
InputFactory = Callable[[torch.device], tuple[Args, Kwargs]]


@dataclass(frozen=True)
class OpCase:
    id: str
    model_factory: Callable[[], torch.nn.Module]
    input_factory: InputFactory
    atol: float | None = None
    rtol: float | None = None


# Arithmetic and unary-style operation cases
ARITHMETIC_UNARY_CASES: list[OpCase] = [
    OpCase(
        id="add",
        model_factory=AddTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="mul",
        model_factory=MulTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="div",
        model_factory=DivTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="add_broadcast",
        model_factory=AddTestModel,
        input_factory=lambda device: ((torch.rand(5, device=device),), {}),
    ),

    OpCase(
        id="add_constant",
        model_factory=AddConstantTestModel,
        input_factory=lambda device: ((torch.rand(5, device=device),), {}),
    ),

    OpCase(
        id="sub",
        model_factory=SubTestModel,
        input_factory=lambda device: ((torch.rand((10, 10), device=device),), {}),
    ),

    OpCase(
        id="sub_broadcast",
        model_factory=SubTestModel,
        input_factory=lambda device: ((torch.rand((10, 10), device=device),), {}),
    ),

    OpCase(
        id="sqrt",
        model_factory=SqrtTestModel,
        input_factory=lambda device: ((torch.rand(100, device=device),), {}),
    ),

    OpCase(
        id="sin",
        model_factory=SinTestModel,
        input_factory=lambda device: ((torch.rand(100, device=device),), {}),
    ),

    OpCase(
        id="cos",
        model_factory=CosTestModel,
        input_factory=lambda device: ((torch.rand(100, device=device),), {}),
    ),

    OpCase(
        id="constant_scalar_float",
        model_factory=ConstantScalarFloatModel,
        input_factory=lambda device: ((torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_1d_array_float",
        model_factory=Constant1DArrayFloatModel,
        input_factory=lambda device: ((torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_2d_matrix_float",
        model_factory=Constant2DMatrixFloatModel,
        input_factory=lambda device: ((torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(device),), {}),
    ),

    OpCase(
        id="constant_raw_data_float",
        model_factory=ConstantRawDataFloatModel,
        input_factory=lambda device: ((torch.tensor([1.0, 2.0, 3.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_int32_conversion",
        model_factory=ConstantInt32ConversionModel,
        input_factory=lambda device: ((torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_int64_conversion",
        model_factory=ConstantInt64ConversionModel,
        input_factory=lambda device: ((torch.tensor([2.0, 3.0, 4.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_float64_conversion",
        model_factory=ConstantFloat64ConversionModel,
        input_factory=lambda device: ((torch.tensor([10.0, 20.0, 30.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_bool_conversion",
        model_factory=ConstantBoolConversionModel,
        input_factory=lambda device: ((torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_int64_raw_data",
        model_factory=ConstantInt64RawDataModel,
        input_factory=lambda device: ((torch.tensor([10.0, 20.0, 30.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_negative_values",
        model_factory=ConstantNegativeValuesModel,
        input_factory=lambda device: ((torch.tensor([100.0, 200.0, 300.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_zero_value",
        model_factory=ConstantZeroValueModel,
        input_factory=lambda device: ((torch.tensor([1.0, 2.0, 3.0, 4.0]).to(device),), {}),
    ),

    OpCase(
        id="constant_multiple_in_graph",
        model_factory=ConstantMultipleInGraphModel,
        input_factory=lambda device: ((torch.tensor([5.0, 6.0, 7.0]).to(device),), {}),
    ),

    OpCase(
        id="cast_double_to_float",
        model_factory=CastDoubleToFloatModel,
        input_factory=lambda device: ((torch.tensor([1.123456789, 2.987654321, 3.555555555, 4.111111111], dtype=torch.float64).to(device),), {}),
    ),

    OpCase(
        id="cast_int32_to_float",
        model_factory=CastInt32ToFloatModel,
        input_factory=lambda device: ((torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).to(device),), {}),
    ),

    OpCase(
        id="cast_int64_to_float",
        model_factory=CastInt64ToFloatModel,
        input_factory=lambda device: ((torch.tensor([100, 200, 300, 400], dtype=torch.int64).to(device),), {}),
    ),

    OpCase(
        id="cast_bool_to_float",
        model_factory=CastBoolToFloatModel,
        input_factory=lambda device: ((torch.tensor([True, False, True, False, True, False], dtype=torch.bool).to(device),), {}),
    ),

    OpCase(
        id="cast_in_computation_graph",
        model_factory=CastInComputationGraphModel,
        input_factory=lambda device: ((torch.tensor([10, 20, 30], dtype=torch.int32).to(device),), {}),
    ),

    OpCase(
        id="cast_with_2d_tensor",
        model_factory=CastWith2DTensorModel,
        input_factory=lambda device: ((torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64).to(device),), {}),
    ),

    OpCase(
        id="cast_negative_values",
        model_factory=CastNegativeValuesModel,
        input_factory=lambda device: ((torch.tensor([-10, -5, 0, 5, 10], dtype=torch.int32).to(device),), {}),
    ),

    OpCase(
        id="cast_scalar_value",
        model_factory=CastScalarValueModel,
        input_factory=lambda device: ((torch.tensor([42.123456], dtype=torch.float64).to(device),), {}),
    ),

    OpCase(
        id="mod",
        model_factory=ModTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) * 10.0,), {}),
    ),

    OpCase(
        id="mod_broadcast",
        model_factory=ModTestModel,
        input_factory=lambda device: ((torch.rand(5, device=device) * 10.0,), {}),
    ),

    OpCase(
        id="mod_by_constant",
        model_factory=ModByConstantModel,
        input_factory=lambda device: ((torch.tensor([7.0, 9.0, 11.0]).to(device),), {}),
    ),

    OpCase(
        id="floor",
        model_factory=FloorTestModel,
        input_factory=lambda device: ((torch.tensor([1.2, 2.7, 3.0, 4.9, 5.5]).to(device),), {}),
    ),

    OpCase(
        id="floor_negative",
        model_factory=FloorNegativeModel,
        input_factory=lambda device: ((torch.tensor([-1.2, -2.7, -0.1, 0.9, -3.5]).to(device),), {}),
    ),

    OpCase(
        id="floor_in_expression",
        model_factory=FloorInExpressionModel,
        input_factory=lambda device: ((torch.tensor([1.5, 2.8, 3.3, 4.1]).to(device),), {}),
    ),

    OpCase(
        id="ceil",
        model_factory=CeilTestModel,
        input_factory=lambda device: ((torch.tensor([1.2, 2.7, 3.0, 4.9, 5.5]).to(device),), {}),
    ),

    OpCase(
        id="ceil_negative",
        model_factory=CeilNegativeModel,
        input_factory=lambda device: ((torch.tensor([-1.2, -2.7, -0.1, 0.9, -3.5]).to(device),), {}),
    ),

    OpCase(
        id="ceil_in_expression",
        model_factory=CeilInExpressionModel,
        input_factory=lambda device: ((torch.tensor([1.5, 2.8, 3.3, 4.1]).to(device),), {}),
    ),

    OpCase(
        id="pow",
        model_factory=PowTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) + 0.1,), {}),
        atol=0.0001,
        rtol=0.0001,
    ),

    OpCase(
        id="pow_broadcast",
        model_factory=PowTestModel,
        input_factory=lambda device: ((torch.rand(5, device=device) + 0.1,), {}),
        atol=0.0001,
        rtol=0.0001,
    ),

    OpCase(
        id="pow_by_constant",
        model_factory=PowByConstantModel,
        input_factory=lambda device: ((torch.tensor([2.0, 3.0, 4.0]).to(device),), {}),
        atol=0.0001,
        rtol=0.0001,
    ),

    OpCase(
        id="relu",
        model_factory=ReluTestModel,
        input_factory=lambda device: ((torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device),), {}),
    ),

    OpCase(
        id="sigmoid",
        model_factory=SigmoidTestModel,
        input_factory=lambda device: ((torch.rand((4, 8), device=device) * 4.0 - 2.0,), {}),
        atol=1e-05,
    ),

    OpCase(
        id="tanh",
        model_factory=TanhTestModel,
        input_factory=lambda device: ((torch.rand((4, 8), device=device) * 4.0 - 2.0,), {}),
        atol=1e-05,
    ),

    OpCase(
        id="clip",
        model_factory=ClipTestModel,
        input_factory=lambda device: ((torch.rand((4, 5), device=device) * 2.0 - 1.0,), {}),
    ),

    OpCase(
        id="clip_min_only",
        model_factory=ClipMinOnlyTestModel,
        input_factory=lambda device: ((torch.rand((4, 5), device=device) * 2.0 - 1.0,), {}),
    ),

    OpCase(
        id="erf",
        model_factory=ErfTestModel,
        input_factory=lambda device: ((torch.linspace(-2.0, 2.0, 16, device=device),), {}),
        atol=0.0001,
    ),
]
# Tensor movement and shape transformation cases
MOVEMENT_CASES: list[OpCase] = [
    OpCase(
        id="transpose_2d",
        model_factory=TransposeTestModel,
        input_factory=lambda device: ((torch.rand((5, 10), device=device),), {}),
    ),

    OpCase(
        id="transpose_3d",
        model_factory=Transpose3DTestModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
    ),

    OpCase(
        id="transpose_4d",
        model_factory=Transpose4DTestModel,
        input_factory=lambda device: ((torch.rand((1, 3, 224, 224), device=device),), {}),
    ),

    OpCase(
        id="transpose_reverse",
        model_factory=TransposeReverseTestModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4, 5), device=device),), {}),
    ),

    OpCase(
        id="transpose_in_expression",
        model_factory=TransposeInExpressionModel,
        input_factory=lambda device: ((torch.rand((5, 10), device=device),), {}),
    ),

    OpCase(
        id="transpose_square_matrix",
        model_factory=TransposeTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="reshape_2d_to_1d",
        model_factory=ReshapeToFlatModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
    ),

    OpCase(
        id="reshape_1d_to_2d",
        model_factory=ReshapeToMatrixModel,
        input_factory=lambda device: ((torch.rand(12, device=device),), {}),
    ),

    OpCase(
        id="reshape_1d_to_3d",
        model_factory=ReshapeTo3DModel,
        input_factory=lambda device: ((torch.rand(24, device=device),), {}),
    ),

    OpCase(
        id="reshape_infer_last_dim",
        model_factory=ReshapeInferLastDimModel,
        input_factory=lambda device: ((torch.rand(12, device=device),), {}),
    ),

    OpCase(
        id="reshape_infer_first_dim",
        model_factory=ReshapeInferFirstDimModel,
        input_factory=lambda device: ((torch.rand(12, device=device),), {}),
    ),

    OpCase(
        id="reshape_3d_to_2d",
        model_factory=Reshape3Dto2DModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
    ),

    OpCase(
        id="reshape_in_expression",
        model_factory=ReshapeInExpressionModel,
        input_factory=lambda device: ((torch.rand(12, device=device),), {}),
    ),

    OpCase(
        id="reshape_roundtrip",
        model_factory=ReshapeRoundtripModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
    ),

    OpCase(
        id="reshape_after_ops",
        model_factory=ReshapeAfterOpsModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
    ),

    OpCase(
        id="shape_reshape_batch_flatten",
        model_factory=ShapeReshapeBatchFlattenModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
    ),

    OpCase(
        id="shape_reshape_view_batch",
        model_factory=ShapeReshapeKeepBatchModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
    ),

    OpCase(
        id="squeeze_axis",
        model_factory=SqueezeAxisModel,
        input_factory=lambda device: ((torch.rand(1, 3, 4, device=device),), {}),
    ),

    OpCase(
        id="squeeze_all_dims",
        model_factory=SqueezeAllDimsModel,
        input_factory=lambda device: ((torch.rand(1, 3, 1, 4, device=device),), {}),
    ),

    OpCase(
        id="squeeze_multiple_axes",
        model_factory=SqueezeMultipleAxesModel,
        input_factory=lambda device: ((torch.rand(1, 3, 1, 4, device=device),), {}),
    ),

    OpCase(
        id="squeeze_negative_axis",
        model_factory=SqueezeNegativeAxisModel,
        input_factory=lambda device: ((torch.rand(3, 4, 1, device=device),), {}),
    ),

    OpCase(
        id="squeeze_in_expression",
        model_factory=SqueezeInExpressionModel,
        input_factory=lambda device: ((torch.rand(1, 5, device=device),), {}),
    ),

    OpCase(
        id="concat_axis0",
        model_factory=ConcatAxis0Model,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
    ),

    OpCase(
        id="concat_axis1",
        model_factory=ConcatAxis1Model,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
    ),

    OpCase(
        id="concat_three_tensors",
        model_factory=ConcatThreeTensorsModel,
        input_factory=lambda device: ((torch.rand((4, 4), device=device),), {}),
    ),

    OpCase(
        id="concat_negative_axis",
        model_factory=ConcatNegativeAxisModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
    ),

    OpCase(
        id="concat_in_expression",
        model_factory=ConcatInExpressionModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
    ),

    OpCase(
        id="unsqueeze",
        model_factory=UnsqueezeAxis0Model,
        input_factory=lambda device: ((torch.rand(6, device=device),), {}),
    ),

    OpCase(
        id="unsqueeze_middle",
        model_factory=UnsqueezeMiddleModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
    ),

    OpCase(
        id="expand",
        model_factory=ExpandTestModel,
        input_factory=lambda device: ((torch.rand((1, 4), device=device),), {}),
    ),

    OpCase(
        id="slice_1d",
        model_factory=SliceTestModel,
        input_factory=lambda device: ((torch.rand(5, device=device),), {}),
    ),

    OpCase(
        id="slice_2d",
        model_factory=SliceMultiAxisTestModel,
        input_factory=lambda device: ((torch.rand(4, 4, device=device),), {}),
    ),

    OpCase(
        id="split",
        model_factory=SplitTestModel,
        input_factory=lambda device: ((torch.rand(3, 4, device=device),), {}),
    ),
]
# Comparison, logical, and selection cases
COMPARISON_CASES: list[OpCase] = [
    OpCase(
        id="less",
        model_factory=LessTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="less_broadcast",
        model_factory=LessBroadcastModel,
        input_factory=lambda device: ((torch.rand(5, device=device),), {}),
    ),

    OpCase(
        id="less_with_constant",
        model_factory=LessWithConstantModel,
        input_factory=lambda device: ((torch.tensor([0.1, 0.5, 0.9]).to(device),), {}),
    ),

    OpCase(
        id="equal",
        model_factory=EqualTestModel,
        input_factory=lambda device: ((torch.randint(0, 3, (5, 5)).float().to(device),), {}),
    ),

    OpCase(
        id="equal_broadcast",
        model_factory=EqualBroadcastModel,
        input_factory=lambda device: ((torch.randint(0, 3, (5,)).float().to(device),), {}),
    ),

    OpCase(
        id="equal_with_constant",
        model_factory=EqualWithConstantModel,
        input_factory=lambda device: ((torch.tensor([1.0, 0.5, 3.0]).to(device),), {}),
    ),

    OpCase(
        id="where",
        model_factory=WhereTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) - 0.5,), {}),
    ),

    OpCase(
        id="where_self_select",
        model_factory=WhereSelfSelectModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device) - 0.5,), {}),
    ),

    OpCase(
        id="where_with_constant",
        model_factory=WhereWithConstantModel,
        input_factory=lambda device: ((torch.tensor([-0.5, 0.5, -1.0]).to(device),), {}),
    ),

    OpCase(
        id="max",
        model_factory=MaxTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="max_with_constant",
        model_factory=MaxWithConstantModel,
        input_factory=lambda device: ((torch.rand(5, device=device),), {}),
    ),

    OpCase(
        id="min",
        model_factory=MinTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="min_with_constant",
        model_factory=MinWithConstantModel,
        input_factory=lambda device: ((torch.rand(5, device=device),), {}),
    ),

    OpCase(
        id="less_or_equal",
        model_factory=LessOrEqualTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="less_or_equal_with_constant",
        model_factory=LessOrEqualWithConstantModel,
        input_factory=lambda device: ((torch.rand(3, device=device),), {}),
    ),

    OpCase(
        id="greater_or_equal",
        model_factory=GreaterOrEqualTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="greater_or_equal_with_constant",
        model_factory=GreaterOrEqualWithConstantModel,
        input_factory=lambda device: ((torch.rand(3, device=device),), {}),
    ),

    OpCase(
        id="not",
        model_factory=NotTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="and",
        model_factory=AndTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="or",
        model_factory=OrTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="xor",
        model_factory=XorTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="greater",
        model_factory=GreaterTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="greater_with_constant",
        model_factory=GreaterWithConstantModel,
        input_factory=lambda device: ((torch.rand(8, device=device),), {}),
    ),

    OpCase(
        id="isnan",
        model_factory=IsNaNTestModel,
        input_factory=lambda device: ((torch.rand((3, 3), device=device),), {}),
    ),
]
# Reduction-style cases
REDUCTION_CASES: list[OpCase] = [
    OpCase(
        id="reduce_sum_axis0",
        model_factory=ReduceSumAxis0Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_axis1",
        model_factory=ReduceSumAxis1Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_keepdims",
        model_factory=ReduceSumKeepDimsModel,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_all_axes",
        model_factory=ReduceSumAllAxesModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_3d_axis1",
        model_factory=ReduceSum3DAxis1Model,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_multi_axis",
        model_factory=ReduceSumMultiAxisModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_multi_axis_keepdims",
        model_factory=ReduceSumMultiAxisKeepDimsModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_negative_axis",
        model_factory=ReduceSumNegativeAxisModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_sum_in_expression",
        model_factory=ReduceSumInExpressionModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_axis0",
        model_factory=ReduceMaxAxis0Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_axis1",
        model_factory=ReduceMaxAxis1Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_keepdims",
        model_factory=ReduceMaxKeepDimsModel,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_all_axes",
        model_factory=ReduceMaxAllAxesModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_3d_axis1",
        model_factory=ReduceMax3DAxis1Model,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_multi_axis",
        model_factory=ReduceMaxMultiAxisModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_multi_axis_keepdims",
        model_factory=ReduceMaxMultiAxisKeepDimsModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_negative_axis",
        model_factory=ReduceMaxNegativeAxisModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_max_in_expression",
        model_factory=ReduceMaxInExpressionModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_axis0",
        model_factory=ReduceMinAxis0Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_axis1",
        model_factory=ReduceMinAxis1Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_keepdims",
        model_factory=ReduceMinKeepDimsModel,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_all_axes",
        model_factory=ReduceMinAllAxesModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_3d_axis1",
        model_factory=ReduceMin3DAxis1Model,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_multi_axis",
        model_factory=ReduceMinMultiAxisModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_multi_axis_keepdims",
        model_factory=ReduceMinMultiAxisKeepDimsModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_negative_axis",
        model_factory=ReduceMinNegativeAxisModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_min_in_expression",
        model_factory=ReduceMinInExpressionModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_axis0",
        model_factory=ReduceMeanAxis0Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_axis1",
        model_factory=ReduceMeanAxis1Model,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_keepdims",
        model_factory=ReduceMeanKeepDimsModel,
        input_factory=lambda device: ((torch.rand((4, 5), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_all_axes",
        model_factory=ReduceMeanAllAxesModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_3d_axis1",
        model_factory=ReduceMean3DAxis1Model,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_multi_axis",
        model_factory=ReduceMeanMultiAxisModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_multi_axis_keepdims",
        model_factory=ReduceMeanMultiAxisKeepDimsModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_negative_axis",
        model_factory=ReduceMeanNegativeAxisModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="reduce_mean_in_expression",
        model_factory=ReduceMeanInExpressionModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="softmax",
        model_factory=SoftmaxTestModel,
        input_factory=lambda device: ((torch.rand((4, 8), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="softmax_dim0",
        model_factory=SoftmaxDim0TestModel,
        input_factory=lambda device: ((torch.rand((4, 8), device=device),), {}),
        atol=1e-05,
    ),
]
# Indexing, gather/scatter, and masking cases
INDEXING_CASES: list[OpCase] = [
    OpCase(
        id="gather_1d",
        model_factory=Gather1DModel,
        input_factory=lambda device: ((torch.rand(6, device=device),), {}),
    ),

    OpCase(
        id="gather_embedding",
        model_factory=GatherEmbeddingModel,
        input_factory=lambda device: ((torch.tensor([0, 2, 5, 1]).to(device),), {}),
    ),

    OpCase(
        id="gather_2d_axis0",
        model_factory=Gather2DAxis0Model,
        input_factory=lambda device: ((torch.tensor([2, 0, 4, 1]).to(device),), {}),
    ),

    OpCase(
        id="gather_2d_axis1",
        model_factory=Gather2DAxis1Model,
        input_factory=lambda device: ((torch.rand(4, 5, device=device),), {}),
    ),

    OpCase(
        id="gather_negative_indices",
        model_factory=GatherNegativeIndicesModel,
        input_factory=lambda device: ((torch.tensor([-1, -2, 0, 2]).to(device),), {}),
    ),

    OpCase(
        id="gather_constant_fold",
        model_factory=GatherConstantFoldModel,
        input_factory=lambda device: ((torch.tensor([1.0, 2.0, 3.0]).to(device),), {}),
    ),

    OpCase(
        id="tril",
        model_factory=TrilTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="triu",
        model_factory=TriuTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="tril_diagonal",
        model_factory=TrilDiagonalTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="triu_diagonal",
        model_factory=TriuDiagonalTestModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="gather_elements",
        model_factory=GatherElementsTestModel,
        input_factory=lambda device: ((torch.rand((2, 3), device=device),), {}),
    ),

    OpCase(
        id="gather_elements_large",
        model_factory=GatherElementsLargeTestModel,
        input_factory=lambda device: ((torch.rand((4, 8), device=device),), {}),
    ),

    OpCase(
        id="topk_values",
        model_factory=TopKValuesTestModel,
        input_factory=lambda device: ((torch.rand(4, 8, device=device),), {}),
    ),

    OpCase(
        id="topk_indices",
        model_factory=TopKIndicesTestModel,
        input_factory=lambda device: ((torch.rand(4, 8, device=device),), {}),
    ),

    OpCase(
        id="onehot",
        model_factory=OneHotTestModel,
        input_factory=lambda device: ((torch.tensor([0, 2, 4, 1, 3], device=device),), {}),
    ),

    OpCase(
        id="scatter_elements",
        model_factory=ScatterElementsTestModel,
        input_factory=lambda device: ((torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device),), {}),
    ),

    OpCase(
        id="scatter_elements_axis0",
        model_factory=ScatterElementsAxis0TestModel,
        input_factory=lambda device: ((torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device),), {}),
    ),

    OpCase(
        id="scatter_nd",
        model_factory=ScatterNDTestModel,
        input_factory=lambda device: ((torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=device),), {}),
    ),
]
# Multi-op and NN composite graph cases
COMPOSITE_CASES: list[OpCase] = [
    OpCase(
        id="linear_layer",
        model_factory=LinearLayerModel,
        input_factory=lambda device: ((torch.rand((5, 5), device=device),), {}),
    ),

    OpCase(
        id="matmul_2d",
        model_factory=MatMul2DModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="matmul_batched",
        model_factory=MatMulBatchedModel,
        input_factory=lambda device: ((torch.rand((2, 3, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="layer_norm",
        model_factory=ManualLayerNormModel,
        input_factory=lambda device: ((torch.rand((2, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="scaled_dot_product_attention",
        model_factory=ScaledDotProductModel,
        input_factory=lambda device: ((torch.rand((4, 8), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="mlp_block",
        model_factory=MLPBlockModel,
        input_factory=lambda device: ((torch.rand((2, 8), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="layernorm",
        model_factory=LayerNormTestModel,
        input_factory=lambda device: ((torch.rand((2, 4), device=device),), {}),
        atol=1e-05,
    ),

    OpCase(
        id="gemm",
        model_factory=GemmTestModel,
        input_factory=lambda device: ((torch.rand((3, 4), device=device),), {}),
        atol=1e-05,
    ),
]
ALL_CASES: list[OpCase] = [
    *ARITHMETIC_UNARY_CASES,
    *MOVEMENT_CASES,
    *COMPARISON_CASES,
    *REDUCTION_CASES,
    *INDEXING_CASES,
    *COMPOSITE_CASES,
]


class TestHLIROps:
    @pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.id)
    def test_matches_eager(self, case: OpCase, device: torch.device) -> None:
        model = case.model_factory().to(device)
        compiled_model = torch.compile(model, backend=luminal_backend)
        args, kwargs = case.input_factory(device)
        eager_output = model(*args, **kwargs)
        compiled_output = compiled_model(*args, **kwargs)
        # Mirror torch.allclose defaults for omitted tolerances.
        atol = 1e-8 if case.atol is None else case.atol
        rtol = 1e-5 if case.rtol is None else case.rtol
        torch.testing.assert_close(compiled_output, eager_output, atol=atol, rtol=rtol)

    def test_add_add_dynamic_shapes(self, device: torch.device) -> None:
        model = AddAddTestModel().to(device)
        compiled_model = torch.compile(model, backend=luminal_backend)
        x = torch.rand((5, 5), device=device)
        torch.testing.assert_close(compiled_model(x), model(x))
        other_x = torch.rand((5,), device=device)
        torch.testing.assert_close(compiled_model(other_x), model(other_x))
