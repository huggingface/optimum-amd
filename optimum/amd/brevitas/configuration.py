# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BrevitasQuantizationConfig:
    """
    QuantizationConfig is the configuration class handling all the ONNX Runtime quantization parameters.

    Args:
        replace_mha_with_quantizable (`bool`, defaults to `False`):
            TODO
        weights_bitwidth (`int`, defaults to `8`):
            Bitwidth of the weights quantization. For example, with `weights_bitwidth=8`, each weight value is quantized on 8 bits.
        activations_bitwidth (`int`, defaults to `8`):
            Bitwidth of the activations quantization.
        weights_param_method (`str`, defaults to `stats`):
            TODO
        weights_symmetric (`bool`, defaults to `True`):
            Whether to use symmetric quantization on the weights.
        scale_precision (`str`, defaults to `"float_scale"`):
            Precise the constraints on the scale. Can either be `"float_scale"` (arbitrary scales), or `"power_of_two_scale"` (scales constrainted to be a power of 2).
        weights_quant_granularity (`str`, defaults to `"per_tensor"`):
            The granularity of the quantization of the weights. This parameter can either be:
            - `"per_tensor"`: A single scale (and possibly zero-point) is used for one weight matrix.
            - `"per_channel"`: Each column (outer dimension) of the weight matrix has its own scale (and possibly zero-point).
            - `"per_group"`: Each column of the weight matrix may have several scales, grouped by `weight_group_size`.
        weights_group_size (`Optional[int]`, defaults to `None`):
            Group size to use for the weights in case `weights_quant_granularity="per_group"`. Defaults to `128` in this case, to `None` otherwise.
        quantize_zero_point (`bool`, defaults to `True`):
            TODO
        activations_param_method (`List[str]`):
            TODO
        is_static (`bool`, defaults to `False`):
            Whether to apply static quantization or dynamic quantization.
        activations_symmetric (`bool`, defaults to `False`):
            Whether to use symmetric quantization on the activations.
        activations_quant_granularity (`str`, defaults to `"per_tensor"`):
            The granularity of the quantization of the activations. This parameter can either be `"per_tensor"`, `"per_row"` or `"per_group"`. In case static quantization is used (`is_static=True`), only `"per_tensor"` may be used.
        activations_group_size (`int`, defaults to `None`):
            Group size to use for the activations in case `activations_quant_granularity="per_group"`. Defaults to `64` in this case, to `None` otherwise.
        activations_equalization (`Optional[str]`, defaults to `"cross_layer"`):
            Whether to apply apply activation equalization (SmoothQuant). Possible options are:
            - `None`: No activation equalization.
            - `"cross_layer"`: TODO
            - `"layerwise"`: TODO
        apply_weight_equalization (`bool`, defaults to `False`):
            TODO
        apply_gptq (`bool`, defaults to `False`):
            Whether to apply GPTQ algorithm for quantizing the weights.
        gptq_act_oder (`Optional[bool]`, defaults to `None`):
            Whether to use activations reordering (act-order, also known as desc-act) when `apply_gptq=True`. If `apply_gptq=True`, defaults to `False`.
    """

    replace_mha_with_quantizable: bool = False
    weights_bitwidth: int = 8
    activations_bitwidth: int = 8
    weights_param_method: Literal["stats", "mse"] = "stats"
    weights_symmetric: bool = True
    scale_precision: Literal["float_scale", "power_of_two_scale"] = "float_scale"
    weights_quant_granularity: Literal["per_tensor", "per_channel", "per_group"] = "per_tensor"
    weights_group_size: Optional[int] = None
    quantize_zero_point: bool = True
    activations_param_method: Literal["stats", "mse"] = "stats"
    is_static: bool = False
    activations_symmetric: bool = False
    activations_quant_granularity: Literal["per_tensor", "per_row", "per_group"] = "per_tensor"
    activations_group_size: Optional[int] = None
    activations_equalization: Literal[None, "layerwise", "cross_layer"] = "cross_layer"
    apply_weight_equalization: bool = False
    apply_gptq: bool = False
    gptq_act_oder: Optional[bool] = None

    def __post_init__(self):
        if self.activations_quant_granularity == "per_group" and self.activations_group_size is None:
            self.activations_group_size = 64

        if self.weights_quant_granularity == "per_group" and self.weights_group_size is None:
            self.weights_group_size = 128

        if self.apply_gptq and self.gptq_act_oder is None:
            self.gptq_act_oder = False

        if self.is_static and self.activations_quant_granularity != "per_tensor":
            raise ValueError(
                f'Static quantization with activations_quant_granularity="{self.activations_quant_granularity}" is not supported. The quantization granularity must be activations_quant_granularity="per_tensor" when using static quantization.'
            )

        if self.activations_quant_granularity == "per_row" and not self.replace_mha_with_quantizable:
            raise ValueError("Per-row activations quantization requires setting replace_mha_with_quantizable to True.")

    def requires_fx_graph(self):
        return self.activations_equalization == "cross_layer" or self.apply_weight_equalization
