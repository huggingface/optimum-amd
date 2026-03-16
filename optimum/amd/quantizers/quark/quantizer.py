from enum import Enum
from typing import (
    Iterable,
    Union,
)

import torch

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config
from transformers.quantizers import HfQuantizer, HfQuantizerPlugin
from transformers.utils.quantization_config import QuantizationConfigMixin

class QuantizationMethod(str, Enum):
    QUARK = "quark"

class QuarkPlugin(HfQuantizerPlugin):
    @staticmethod
    def get_quantizer():
        return QuarkQuantizer
    
    @staticmethod
    def get_config():
        return QuarkConfig

class QuarkConfig(QuantizationConfigMixin):
    def __init__(
        self,
        qconfig: Union[Config],
        dataset: Iterable = None,
    ):
        self._qconfig = qconfig
        self._dataset = dataset

    @property
    def quant_method(self):
        return QuantizationMethod.QUARK


class QuarkQuantizer(HfQuantizer):
    requires_calibration = False
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.pre_quantized is True:
            raise ValueError("Pre-quantized models are not supported by QuarkQuantizer")

        self._quantizer = ModelQuantizer(quantization_config._qconfig)

    def _process_model_before_weight_loading(self, model, **kwargs):
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        if not self._quantizer.is_all_dynamic and self.quantization_config.dataset is None:
            raise ValueError("A calibration dataset is required for the quantization method.")

        with torch.inference_mode():
            qmodel = self._quantizer.quantize_model(model, self._quantization_config.dataset)

        return qmodel

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return False
