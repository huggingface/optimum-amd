# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import unittest

import torch
from brevitas.nn.quant_linear import QuantLinear
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector, DynamicActQuantProxyFromInjector
from parameterized import parameterized

from optimum.amd import BrevitasQuantizationConfig, BrevitasQuantizer
from optimum.amd.brevitas.data_utils import get_dataset_for_model
from transformers import AutoTokenizer


SUPPORTED_MODELS = {"llama": "fxmarty/tiny-llama-fast-tokenizer", "opt": "hf-internal-testing/tiny-random-OPTModel"}


class TestQuantization(unittest.TestCase):
    @parameterized.expand(SUPPORTED_MODELS.keys())
    def test_dynamic_quantization(self, model_type: str):
        model_id = SUPPORTED_MODELS[model_type]
        qconfig = BrevitasQuantizationConfig(
            is_static=False,
            apply_gptq=False,
            apply_weight_equalization=False,
            activations_equalization=None,
            replace_mha_with_quantizable=False,
        )
        quantizer = BrevitasQuantizer.from_pretrained(model_id)

        model = quantizer.quantize(qconfig, calibration_dataset=None)

        found_quant_linear = False
        for _, submodule in model.named_modules():
            if isinstance(submodule, QuantLinear):
                self.assertTrue(isinstance(submodule.input_quant, DynamicActQuantProxyFromInjector))
                found_quant_linear = True
                break
        self.assertTrue(found_quant_linear)

    @parameterized.expand(SUPPORTED_MODELS.keys())
    def test_static_quantization(self, model_type: str):
        model_id = SUPPORTED_MODELS[model_type]
        qconfig = BrevitasQuantizationConfig(
            is_static=True,
            apply_gptq=False,
            apply_weight_equalization=False,
            activations_equalization=None,
            replace_mha_with_quantizable=False,
        )
        quantizer = BrevitasQuantizer.from_pretrained(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the data for calibration and evaluation.
        calibration_dataset = get_dataset_for_model(
            model_id,
            qconfig=qconfig,
            dataset_name="wikitext2",
            tokenizer=tokenizer,
            nsamples=128,
            seqlen=64,
            split="train",
        )

        model = quantizer.quantize(qconfig, calibration_dataset=calibration_dataset)

        found_quant_linear = False
        for _, submodule in model.named_modules():
            if isinstance(submodule, QuantLinear):
                self.assertFalse(isinstance(submodule.input_quant, DynamicActQuantProxyFromInjector))
                self.assertTrue(isinstance(submodule.input_quant, ActQuantProxyFromInjector))
                found_quant_linear = True
                break
        self.assertTrue(found_quant_linear)

    @parameterized.expand(SUPPORTED_MODELS.keys())
    def test_fx_static_quantization(self, model_type: str):
        model_id = SUPPORTED_MODELS[model_type]
        qconfig = BrevitasQuantizationConfig(
            is_static=True,
            apply_gptq=False,
            apply_weight_equalization=False,
            activations_equalization="cross_layer",
            replace_mha_with_quantizable=False,
        )
        quantizer = BrevitasQuantizer.from_pretrained(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the data for calibration and evaluation.
        calibration_dataset = get_dataset_for_model(
            model_id,
            qconfig=qconfig,
            dataset_name="wikitext2",
            tokenizer=tokenizer,
            nsamples=128,
            seqlen=64,
            split="train",
        )

        model = quantizer.quantize(qconfig, calibration_dataset=calibration_dataset)

        self.assertTrue(isinstance(model, torch.fx.GraphModule))

        # TODO: add asserts

    @parameterized.expand(SUPPORTED_MODELS.keys())
    def test_gptq(self, model_type: str):
        model_id = SUPPORTED_MODELS[model_type]
        qconfig = BrevitasQuantizationConfig(
            is_static=True,
            apply_gptq=True,
            apply_weight_equalization=False,
            activations_equalization=None,
            replace_mha_with_quantizable=False,
        )
        quantizer = BrevitasQuantizer.from_pretrained(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the data for calibration and evaluation.
        calibration_dataset = get_dataset_for_model(
            model_id,
            qconfig=qconfig,
            dataset_name="wikitext2",
            tokenizer=tokenizer,
            nsamples=128,
            seqlen=64,
            split="train",
        )

        model = quantizer.quantize(qconfig, calibration_dataset=calibration_dataset)

        found_quant_linear = False
        for _, submodule in model.named_modules():
            if isinstance(submodule, QuantLinear):
                self.assertFalse(isinstance(submodule.input_quant, DynamicActQuantProxyFromInjector))
                self.assertTrue(isinstance(submodule.input_quant, ActQuantProxyFromInjector))
                found_quant_linear = True
                break
        self.assertTrue(found_quant_linear)

    @parameterized.expand(SUPPORTED_MODELS.keys())
    def test_weight_equalization(self, model_type: str):
        model_id = SUPPORTED_MODELS[model_type]
        qconfig = BrevitasQuantizationConfig(
            is_static=True,
            apply_gptq=False,
            apply_weight_equalization=True,
            activations_equalization=None,
            replace_mha_with_quantizable=False,
        )
        quantizer = BrevitasQuantizer.from_pretrained(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the data for calibration and evaluation.
        calibration_dataset = get_dataset_for_model(
            model_id,
            qconfig=qconfig,
            dataset_name="wikitext2",
            tokenizer=tokenizer,
            nsamples=128,
            seqlen=64,
            split="train",
        )

        model = quantizer.quantize(qconfig, calibration_dataset=calibration_dataset)

        self.assertTrue(isinstance(model, torch.fx.GraphModule))
