# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""Configuration classes for quantization with RyzenAI."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import vai_q_onnx
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType

from optimum.configuration_utils import BaseConfig


@dataclass
class QuantizationConfig:
    """
    QuantizationConfig is the configuration class handling all the RyzenAI quantization parameters.

    Args:
        is_static (`bool`):
            Whether to apply static quantization or dynamic quantization.
        format (`QuantFormat`):
            Targeted RyzenAI quantization representation format.
            For the Operator Oriented (QOperator) format, all the quantized operators have their own ONNX definitions.
            For the Tensor Oriented (QDQ) format, the model is quantized by inserting QuantizeLinear / DeQuantizeLinear
            operators.
        calibration_method (`CalibrationMethod`):
            The method chosen to calculate the activations quantization parameters using the calibration dataset.
        activations_dtype (`QuantType`, defaults to `QuantType.QUInt8`):
            The quantization data types to use for the activations.
        activations_symmetric (`bool`, defaults to `False`):
            Whether to apply symmetric quantization on the activations.
        weights_dtype (`QuantType`, defaults to `QuantType.QInt8`):
            The quantization data types to use for the weights.
        weights_symmetric (`bool`, defaults to `True`):
            Whether to apply symmetric quantization on the weights.
        enable_dpu (`bool`, defaults to `True`):
            Determines whether to generate a quantized model that is suitable for the DPU. If set to True, the quantization
            process will create a model that is optimized for DPU computations.

    """

    format: QuantFormat = QuantFormat.QDQ
    calibration_method: CalibrationMethod = vai_q_onnx.PowerOfTwoMethod.MinMSE
    activations_dtype: QuantType = QuantType.QUInt8
    activations_symmetric: bool = True
    weights_dtype: QuantType = QuantType.QInt8
    weights_symmetric: bool = True
    enable_dpu: bool = True

    @staticmethod
    def quantization_type_str(activations_dtype: QuantType, weights_dtype: QuantType) -> str:
        return (
            f"{'s8' if activations_dtype == QuantType.QInt8 else 'u8'}"
            f"/"
            f"{'s8' if weights_dtype == QuantType.QInt8 else 'u8'}"
        )

    @property
    def use_symmetric_calibration(self) -> bool:
        return self.activations_symmetric and self.weights_symmetric

    def __str__(self):
        return (
            f"{self.format} ("
            f"schema: {QuantizationConfig.quantization_type_str(self.activations_dtype, self.weights_dtype)}, "
            f"enable_dpu: {self.enable_dpu})"
        )


class RyzenAIConfig(BaseConfig):
    """
    RyzenAIConfig is the configuration class handling all the VitisAI parameters related to the ONNX IR model export,
     and quantization parameters.

    Attributes:
        opset (`Optional[int]`, defaults to `None`):
            ONNX opset version to export the model with.
        quantization (`Optional[QuantizationConfig]`, defaults to `None`):
            Specify a configuration to quantize ONNX model
    """

    CONFIG_NAME = "ryzenai_config.json"
    FULL_CONFIGURATION_FILE = "ryzenai_config.json"

    def __init__(
        self,
        opset: Optional[int] = None,
        quantization: Optional[QuantizationConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.opset = opset
        self.quantization = self.dataclass_to_dict(quantization)
        self.optimum_version = kwargs.pop("optimum_version", None)

    @staticmethod
    def dataclass_to_dict(config) -> dict:
        new_config = {}
        if config is None:
            return new_config
        if isinstance(config, dict):
            return config
        for k, v in asdict(config).items():
            if isinstance(v, Enum):
                v = v.name
            elif isinstance(v, list):
                v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
            new_config[k] = v
        return new_config
