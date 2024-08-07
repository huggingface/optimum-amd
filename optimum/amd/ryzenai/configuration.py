# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
"""Configuration classes for quantization with RyzenAI."""

from dataclasses import asdict
from enum import Enum
from typing import Optional

from onnxruntime.quantization import CalibrationMethod, QuantType
from quark.onnx.calibrate import PowerOfTwoMethod
from quark.onnx.quantization.config.config import QuantizationConfig

from optimum.configuration_utils import BaseConfig


class AutoQuantizationConfig:
    @staticmethod
    def npu_cnn_config():
        return QuantizationConfig(
            calibrate_method=PowerOfTwoMethod.MinMSE,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            enable_npu_cnn=True,
            extra_options={"ActivationSymmetric": True},
        )

    @staticmethod
    def npu_transformer_config():
        return QuantizationConfig(
            calibrate_method=CalibrationMethod.MinMax,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            enable_npu_transformer=True,
        )

    @staticmethod
    def cpu_cnn_config(
        include_cle: bool = True,
        include_fast_ft: bool = True,
        extra_options: dict = None,
    ):
        return QuantizationConfig(
            calibrate_method=CalibrationMethod.Percentile,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            include_cle=include_cle,
            include_fast_ft=include_fast_ft,
            extra_options=extra_options,
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
