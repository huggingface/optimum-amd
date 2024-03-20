# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""Configuration classes for quantization with RyzenAI."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal, Optional

import vai_q_onnx

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

    format: Literal["qdq", "qop", "vitisqdq"] = "qdq"
    calibration_method: Literal["nonoverflow", "mse", "minmax", "entropy", "percentile"] = "mse"
    activations_dtype: Literal["uint8", "int8"] = "uint8"
    activations_symmetric: bool = True
    weights_dtype: Literal["uint8", "int8"] = "int8"
    weights_symmetric: bool = True
    enable_dpu: bool = True

    def __post_init__(self):
        self.format = self._map_format(self.format)
        self.calibration_method = self._map_calibration_method(self.calibration_method)
        self.activations_dtype, self.weights_dtype = self._map_dtypes(self.activations_dtype, self.weights_dtype)

    @staticmethod
    def _map_format(format_str):
        mapping = {
            "qdq": vai_q_onnx.QuantFormat.QDQ,
            "qop": vai_q_onnx.QuantFormat.QOperator,
            "vitisqdq": vai_q_onnx.VitisQuantFormat.QDQ,
        }
        return QuantizationConfig._map_value(mapping, format_str, "format")

    @staticmethod
    def _map_calibration_method(method_str):
        mapping = {
            "nonoverflow": vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            "mse": vai_q_onnx.PowerOfTwoMethod.MinMSE,
            "minmax": vai_q_onnx.CalibrationMethod.MinMax,
            "entropy": vai_q_onnx.CalibrationMethod.Entropy,
            "percentile": vai_q_onnx.CalibrationMethod.Percentile,
        }
        return QuantizationConfig._map_value(mapping, method_str, "calibration method")

    @staticmethod
    def _map_dtypes(activations_dtype_str, weights_dtype_str):
        mapping = {
            "uint8": vai_q_onnx.QuantType.QUInt8,
            "int8": vai_q_onnx.QuantType.QInt8,
            "uint16": vai_q_onnx.VitisQuantType.QUInt16,
            "int16": vai_q_onnx.VitisQuantType.QInt16,
            "uint32": vai_q_onnx.VitisQuantType.QUInt32,
            "int32": vai_q_onnx.VitisQuantType.QInt32,
            "float16": vai_q_onnx.VitisQuantType.QFloat16,
            "bfloat16": vai_q_onnx.VitisQuantType.QBFloat16,
        }
        activations_dtype = QuantizationConfig._map_value(mapping, activations_dtype_str, "activations dtype")
        weights_dtype = QuantizationConfig._map_value(mapping, weights_dtype_str, "weights dtype")
        return activations_dtype, weights_dtype

    @staticmethod
    def _map_value(mapping, value, name):
        try:
            return mapping[value]
        except KeyError:
            valid_values = ", ".join(f'"{v}"' for v in mapping.keys())
            raise ValueError(f'{name} only supports the following values: {valid_values}. Received "{value}".')

    @staticmethod
    def quantization_type_str(activations_dtype, weights_dtype) -> str:
        str_mapping = {
            vai_q_onnx.QuantType.QUInt8: "u8",
            vai_q_onnx.QuantType.QInt8: "s8",
            vai_q_onnx.VitisQuantType.QUInt16: "u16",
            vai_q_onnx.VitisQuantType.QInt16: "s16",
            vai_q_onnx.VitisQuantType.QUInt32: "u32",
            vai_q_onnx.VitisQuantType.QInt32: "s32",
            vai_q_onnx.VitisQuantType.QFloat16: "f16",
            vai_q_onnx.VitisQuantType.QBFloat16: "bf16",
        }
        activations_str = str_mapping.get(activations_dtype)
        weights_str = str_mapping.get(weights_dtype)
        if activations_str is None or weights_str is None:
            raise ValueError("Unsupported quantization type")
        return f"{activations_str}/{weights_str}"

    @property
    def use_symmetric_calibration(self) -> bool:
        return self.activations_symmetric and self.weights_symmetric

    def __str__(self):
        return (
            f"{self.format} ("
            f"schema: {QuantizationConfig.quantization_type_str(self.activations_dtype, self.weights_dtype)}, "
            f"enable_dpu: {self.enable_dpu})"
        )


class AutoQuantizationConfig:
    @staticmethod
    def ipu_cnn_config():
        return QuantizationConfig(
            format="qdq",
            calibration_method="mse",
            activations_dtype="int8",
            activations_symmetric=True,
            weights_dtype="int8",
            weights_symmetric=True,
            enable_dpu=True,
        )

    @staticmethod
    def cpu_cnn_config(
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        enable_dpu: bool = False,
    ):
        return QuantizationConfig(
            format="qdq",
            calibration_method="minmax",
            activations_dtype="uint8",
            activations_symmetric=use_symmetric_activations,
            weights_dtype="int8",
            weights_symmetric=use_symmetric_weights,
            enable_dpu=enable_dpu,
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
