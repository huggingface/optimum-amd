# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class BrevitasQuantizationConfig:
    """
    QuantizationConfig is the configuration class handling all the Brevitas quantization parameters.

    Args:
        weights_bitwidth (`int`, defaults to `8`):
            Bitwidth of the weights quantization. For example, with `weights_bitwidth=8`, each weight value is quantized on 8 bits.
        activations_bitwidth (`Optional[int]`, defaults to `8`):
            Bitwidth of the activations quantization.
        weights_only (`bool`, defaults to `False`):
            If set to `True`, only weights are to be quantized, otherwise activations are quantized as well.
        weights_calibration_method (`str`, defaults to `stats`):
            Strategy to use to estimate the quantization parameters (scale, zero-point) for the weights. Two strategies are available:
            - `"stats"`: Use min-max to estimate the range to quantize on.
            - `"mse"`: Use mean-square error between the unquantized weights and quantized weights to estimate the range to quantize on.
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
            When set to True, the unquantized value 0.0 is exactly representable as a quantized value: the zero point. When set to False, a quantization range [a, b] is exactly reprensentable (no rounding on a and b), but the unquantized value zero is not exactly representable.
        activations_calibration_method (`List[str]`):
            Strategy to use to estimate the quantization parameters (scale, zero-point) for the activations. Two strategies are available:
            - `"stats"`: Use min-max to estimate the range to quantize on.
            - `"mse"`: Use mean-square error between the unquantized activations and quantized activations to estimate the range to quantize on.
        is_static (`bool`, defaults to `False`):
            Whether to apply static quantization or dynamic quantization.
        activations_symmetric (`bool`, defaults to `False`):
            Whether to use symmetric quantization on the activations.
        activations_quant_granularity (`str`, defaults to `"per_tensor"`):
            The granularity of the quantization of the activations. This parameter can either be `"per_tensor"`, `"per_row"` or `"per_group"`. In case static quantization is used (`is_static=True`), only `"per_tensor"` may be used.
        activations_group_size (`int`, defaults to `None`):
            Group size to use for the activations in case `activations_quant_granularity="per_group"`. Defaults to `64` in this case, to `None` otherwise.
        activations_equalization (`Optional[str]`, defaults to `"cross_layer"`):
            Whether to apply activation equalization (SmoothQuant). Possible options are:
            - `None`: No activation equalization.
            - `"layerwise"`: Apply SmoothQuant as described in https://arxiv.org/abs/2211.10438. The activation rescaling will be added as multiplication node, that is not fused within a preceding layer.
            - `"cross_layer"`: Apply SmoothQuant, and fuse the activation rescaling within a preceding layer when possible (example: nn.LayerNorm followed by nn.Linear). This is achieved through a graph capture of the model using [torch.fx](https://pytorch.org/docs/stable/fx.html#module-torch.fx).
        apply_weight_equalization (`bool`, defaults to `False`):
            Applies weight equalization across layers, following https://arxiv.org/abs/1906.04721. This parameter is useful for models whose activation function is linear or piecewise-linear (like ReLU, used in OPT model), and allows to reduce the quantization error of the weights by balancing scales across layers.
        apply_bias_correction (`bool`, defaults to `False`):
            Applies bias correction to compensate for changes in activation bias caused by quantization.
        apply_gptq (`bool`, defaults to `False`):
            Whether to apply GPTQ algorithm for quantizing the weights.
        gptq_act_order (`Optional[bool]`, defaults to `None`):
            Whether to use activations reordering (act-order, also known as desc-act) when `apply_gptq=True`. If `apply_gptq=True`, defaults to `False`.
    """

    weights_bitwidth: int = 8
    activations_bitwidth: Optional[int] = 8
    weights_only: bool = False
    weights_calibration_method: Literal["stats", "mse"] = "stats"
    weights_symmetric: bool = True
    scale_precision: Literal["float_scale", "power_of_two_scale"] = "float_scale"
    weights_quant_granularity: Literal["per_tensor", "per_channel", "per_group"] = "per_channel"
    weights_group_size: Optional[int] = None
    quantize_zero_point: bool = True
    activations_calibration_method: Optional[Literal["stats", "mse"]] = "mse"
    is_static: bool = False
    activations_symmetric: Optional[bool] = False
    activations_quant_granularity: Optional[Literal["per_tensor", "per_row", "per_group"]] = "per_tensor"
    activations_group_size: Optional[int] = None
    activations_equalization: Optional[Literal[None, "layerwise", "cross_layer"]] = None
    apply_weight_equalization: bool = False
    apply_bias_correction: bool = False
    apply_gptq: bool = False
    gptq_act_order: Optional[bool] = None
    device: str = "auto"
    gpu_device_map: Optional[Dict[int, float]] = None
    cpu_device_map: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.activations_quant_granularity == "per_group" and self.activations_group_size is None:
            self.activations_group_size = 64

        if self.weights_quant_granularity == "per_group" and self.weights_group_size is None:
            self.weights_group_size = 128

        if self.apply_gptq and self.gptq_act_order is None:
            self.gptq_act_order = False

        if self.is_static and self.activations_quant_granularity != "per_tensor":
            raise ValueError(
                f'Static quantization with activations_quant_granularity="{self.activations_quant_granularity}" is not supported. The quantization granularity must be activations_quant_granularity="per_tensor" when using static quantization.'
            )

        if self.weights_quant_granularity == "per_group" and self.weights_calibration_method == "mse":
            raise ValueError(
                'The quantization configuration `weights_quant_granularity="per_group"` is not supported along `weights_calibration_method="mse"`. Per group MSE weight quantization is not supported.'
            )

        if self.scale_precision == "power_of_two_scale" and (
            not self.weights_symmetric or not self.activations_symmetric
        ):
            raise ValueError(
                'The quantization configuration `scale_precision="power_of_two_scale"` is not supported along `weights_symmetric=True` or along `activations_symmetric=True`. Asymmetric quantization with power-of-two scale is not supported.'
            )

        if self.scale_precision == "power_of_two_scale" and self.weights_quant_granularity == "per_group":
            raise ValueError(
                'The quantization configuration `scale_precision="power_of_two_scale"` is not supported along `weights_quant_granularity="per_group"`. Per group quantization with power-of-two scale factors is not supported.'
            )

        if not self.is_static and self.activations_quant_granularity == "per_group" and not self.activations_symmetric:
            raise ValueError(
                'The quantization configuration `activations_quant_granularity="per_group"` is not supported along `activations_symmetric=False`. Asymmetric dynamic per group quantization is not supported.'
            )

        if self.scale_precision == "power_of_two_scale" and not self.is_static:
            raise ValueError(
                'The quantization configuration `scale_precision="power_of_two_scale"` is not supported along `is_static=False`. Dynamic activation quantization with power-of-two scale factor is not supported.'
            )

        if self.weights_only:
            self.activations_bitwidth = None
            self.activations_symmetric = None
            self.activations_equalization = None
            self.activations_group_size = None
            self.activations_calibration_method = None

    def requires_fx_graph(self):
        return self.activations_equalization == "cross_layer" or self.apply_weight_equalization


class AutoQuantizationConfig:
    @staticmethod
    def ipu_transformers_config(
        activations_equalization: Optional[Literal[None, "layerwise", "cross_layer"]] = None,
        apply_weight_equalization: bool = False,
        apply_bias_correction: bool = False,
        apply_gptq: bool = False,
        gptq_act_order: bool = False,
        gpu_device_map: Optional[Dict[int, float]] = None,
        cpu_device_map: Optional[Dict[int, float]] = None,
    ):
        return BrevitasQuantizationConfig(
            weights_quant_granularity="per_tensor",
            activations_equalization=activations_equalization,
            apply_weight_equalization=apply_weight_equalization,
            apply_bias_correction=apply_bias_correction,
            apply_gptq=apply_gptq,
            gptq_act_order=gptq_act_order,
            gpu_device_map=gpu_device_map,
            cpu_device_map=cpu_device_map,
        )
