# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import unittest

import torch
from parameterized import parameterized
from testing_utils import SUPPORTED_MODELS_TINY, get_quantized_model

from brevitas.nn.quant_linear import QuantLinear
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector, DynamicActQuantProxyFromInjector


def _get_all_model_ids(model_type: str):
    if isinstance(SUPPORTED_MODELS_TINY[model_type], str):
        return [SUPPORTED_MODELS_TINY[model_type]]
    else:
        return list(SUPPORTED_MODELS_TINY[model_type].keys())


class TestQuantization(unittest.TestCase):
    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_dynamic_quantization(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            model = get_quantized_model(
                model_id,
                is_static=False,
                apply_gptq=False,
                apply_weight_equalization=False,
                activations_equalization=None,
            )

            found_quant_linear = False
            for _, submodule in model.named_modules():
                if isinstance(submodule, QuantLinear):
                    self.assertTrue(isinstance(submodule.input_quant, DynamicActQuantProxyFromInjector))
                    found_quant_linear = True
                    break
            self.assertTrue(found_quant_linear)

    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_static_quantization(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            model = get_quantized_model(
                model_id,
                is_static=True,
                apply_gptq=False,
                apply_weight_equalization=False,
                activations_equalization=None,
            )

            found_quant_linear = False
            for _, submodule in model.named_modules():
                if isinstance(submodule, QuantLinear):
                    self.assertFalse(isinstance(submodule.input_quant, DynamicActQuantProxyFromInjector))
                    self.assertTrue(isinstance(submodule.input_quant, ActQuantProxyFromInjector))
                    found_quant_linear = True
                    break
            self.assertTrue(found_quant_linear)

    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_fx_static_quantization(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            model = get_quantized_model(
                model_id,
                is_static=True,
                apply_gptq=False,
                apply_weight_equalization=False,
                activations_equalization="cross_layer",
            )

            self.assertTrue(isinstance(model, torch.fx.GraphModule))

    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_gptq(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            model = get_quantized_model(
                model_id,
                is_static=True,
                apply_gptq=True,
                apply_weight_equalization=False,
                activations_equalization=None,
            )

            found_quant_linear = False
            for _, submodule in model.named_modules():
                if isinstance(submodule, QuantLinear):
                    self.assertFalse(isinstance(submodule.input_quant, DynamicActQuantProxyFromInjector))
                    self.assertTrue(isinstance(submodule.input_quant, ActQuantProxyFromInjector))
                    found_quant_linear = True
                    break
            self.assertTrue(found_quant_linear)

    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_weight_equalization(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            model = get_quantized_model(
                model_id,
                is_static=True,
                apply_gptq=False,
                apply_weight_equalization=True,
                activations_equalization=None,
            )

            self.assertTrue(isinstance(model, torch.fx.GraphModule))

    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_weights_only_quantization(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            _ = get_quantized_model(
                model_id,
                weights_only=True,
            )
