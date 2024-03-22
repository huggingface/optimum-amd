# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import gc
import os
import tempfile
import unittest

import huggingface_hub
import numpy as np
import onnx
import onnxruntime
import pytest
import requests
import torch
from parameterized import parameterized
from PIL import Image
from testing_models import (
    PYTORCH_MODELS,
    RYZEN_PREQUANTIZED_MODEL_CUSTOM_TASKS,
    RYZEN_PREQUANTIZED_MODEL_IMAGE_CLASSIFICATION,
    RYZEN_PREQUANTIZED_MODEL_IMAGE_SEGMENTATION,
    RYZEN_PREQUANTIZED_MODEL_IMAGE_TO_IMAGE,
    RYZEN_PREQUANTIZED_MODEL_OBJECT_DETECTION,
)
from testing_utils import (
    DEFAULT_CACHE_DIR,
    DEFAULT_VAIP_CONFIG,
    DEFAULT_VAIP_CONFIG_TRANSFORMERS,
    RyzenAITestCaseMixin,
    get_models_to_test,
)

from optimum.amd import BrevitasQuantizationConfig, BrevitasQuantizer
from optimum.amd.brevitas.data_utils import get_dataset_for_model
from optimum.amd.brevitas.export import onnx_export_from_quantized_model
from optimum.amd.ryzenai import (
    RyzenAIModel,
    RyzenAIModelForCausalLM,
    RyzenAIModelForCustomTasks,
    RyzenAIModelForImageClassification,
    RyzenAIModelForImageToImage,
    RyzenAIModelForObjectDetection,
    RyzenAIModelForSemanticSegmentation,
    pipeline,
)
from optimum.utils import (
    DummyInputGenerator,
    logging,
)
from transformers import AutoTokenizer
from transformers.testing_utils import slow


logger = logging.get_logger()


def load_model_and_input(model_id, repo_type="model"):
    all_files = huggingface_hub.list_repo_files(model_id, repo_type=repo_type)
    file_name = [name for name in all_files if name.endswith(".onnx")][0]

    onnx_model_path = huggingface_hub.hf_hub_download(model_id, file_name)
    model = onnx.load(onnx_model_path)

    input_name = model.graph.input[0].name
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in input_shape]
    input_shape[0] = 1

    ort_input = DummyInputGenerator.random_float_tensor(input_shape, framework="np")

    return file_name, ort_input, input_name


class RyzenAIModelIntegrationTest(unittest.TestCase, RyzenAITestCaseMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_MODEL_ID = "amd/resnet50"

    def test_load_model_from_hub(self):
        os.environ["XLNX_ENABLE_CACHE"] = "0"
        os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

        model = RyzenAIModel.from_pretrained(self.TEST_MODEL_ID, vaip_config=DEFAULT_VAIP_CONFIG)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertListEqual(model.providers, ["VitisAIExecutionProvider", "CPUExecutionProvider"])

    def test_load_model_with_invalid_config_path(self):
        with self.assertRaises(ValueError):
            RyzenAIModel.from_pretrained(self.TEST_MODEL_ID, vaip_config=".\\invalid_path\\vaip_config.json")

    def test_load_model_no_config_path(self):
        with self.assertRaises(ValueError):
            RyzenAIModel.from_pretrained(self.TEST_MODEL_ID)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["XLNX_ENABLE_CACHE"] = "0"
            os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

            model = RyzenAIModel.from_pretrained(self.TEST_MODEL_ID, vaip_config=DEFAULT_VAIP_CONFIG)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue("ResNet_int.onnx" in folder_contents)


class RyzenAIModelForImageClassificationIntegrationTest(unittest.TestCase, RyzenAITestCaseMixin):
    @parameterized.expand(RYZEN_PREQUANTIZED_MODEL_IMAGE_CLASSIFICATION)
    @pytest.mark.prequantized_model_test
    def test_model(self, model_id):
        cache_dir = DEFAULT_CACHE_DIR
        cache_key = model_id.replace("/", "_").lower()

        file_name, ort_input, input_name = load_model_and_input(model_id)

        vaip_config = DEFAULT_VAIP_CONFIG
        outputs_ipu, outputs_cpu = self.prepare_outputs(
            model_id, RyzenAIModelForImageClassification, ort_input, vaip_config, cache_dir, cache_key, file_name
        )

        self.assertIn("logits", outputs_ipu)
        self.assertIn("logits", outputs_cpu)

        self.assertTrue(np.allclose(outputs_ipu.logits, outputs_cpu.logits, atol=1e-4))

        current_ops = self.get_ops(cache_dir, cache_key)
        baseline_ops = self.get_baseline_ops(cache_key)

        self.assertEqual(baseline_ops["all"], current_ops["all"], f"Total operators do not match! {current_ops}")
        self.assertEqual(baseline_ops["dpu"], current_ops["dpu"], f"DPU operators do not match! {current_ops}")

        gc.collect()

    @parameterized.expand(
        ["mohitsha/timm-resnet18-onnx-quantized-ryzen", "mohitsha/transformers-resnet18-onnx-quantized-ryzen"]
    )
    @slow
    def test_pipeline(self, model_id):
        os.environ["XLNX_ENABLE_CACHE"] = "0"
        os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

        pipe = pipeline("image-classification", model=model_id, vaip_config=DEFAULT_VAIP_CONFIG)

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        outputs = pipe(image)[0]

        self.assertGreaterEqual(outputs["score"], 0.0)


class RyzenAIModelForObjectDetectionIntegrationTest(unittest.TestCase, RyzenAITestCaseMixin):
    PIPELINE_SUPPORTED_MODEL_ARCH = [
        "yolov3",
        "yolov5",
        "yolov8",
        "yolox",
    ]

    @parameterized.expand(list(RYZEN_PREQUANTIZED_MODEL_OBJECT_DETECTION.values()))
    @pytest.mark.prequantized_model_test
    def test_model(self, model_id):
        cache_dir = DEFAULT_CACHE_DIR
        cache_key = model_id.replace("/", "_").lower()

        file_name, ort_input, input_name = load_model_and_input(model_id)

        vaip_config = DEFAULT_VAIP_CONFIG
        outputs_ipu, outputs_cpu = self.prepare_outputs(
            model_id, RyzenAIModelForObjectDetection, ort_input, vaip_config, cache_dir, cache_key, file_name
        )

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        current_ops = self.get_ops(cache_dir, cache_key)
        baseline_ops = self.get_baseline_ops(cache_key)

        self.assertEqual(baseline_ops["all"], current_ops["all"], f"Total operators do not match! {current_ops}")
        self.assertEqual(baseline_ops["dpu"], current_ops["dpu"], f"DPU operators do not match! {current_ops}")

        gc.collect()

    @parameterized.expand(PIPELINE_SUPPORTED_MODEL_ARCH)
    @slow
    def test_pipeline(self, model_arch):
        os.environ["XLNX_ENABLE_CACHE"] = "0"
        os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

        model_id = RYZEN_PREQUANTIZED_MODEL_OBJECT_DETECTION[model_arch]
        pipe = pipeline("object-detection", model=model_id, vaip_config=DEFAULT_VAIP_CONFIG, model_type=model_arch)

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        outputs = pipe(image)

        self.assertTrue(len(outputs) > 0)

        for pred in outputs:
            self.assertIn("box", pred)
            self.assertIn("label", pred)
            self.assertIn("score", pred)
            self.assertGreaterEqual(pred["score"], 0.0)

    @slow
    def test_pipeline_model_is_none(self):
        os.environ["XLNX_ENABLE_CACHE"] = "0"
        os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

        pipe = pipeline("object-detection", vaip_config=DEFAULT_VAIP_CONFIG)

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        outputs = pipe(image)

        self.assertTrue(len(outputs) > 0)

        for pred in outputs:
            self.assertIn("box", pred)
            self.assertIn("label", pred)
            self.assertIn("score", pred)
            self.assertGreaterEqual(pred["score"], 0.0)


class RyzenAIModelForSemanticSegmentationIntegrationTest(unittest.TestCase, RyzenAITestCaseMixin):
    @parameterized.expand(RYZEN_PREQUANTIZED_MODEL_IMAGE_SEGMENTATION)
    @pytest.mark.prequantized_model_test
    def test_model(self, model_id):
        cache_dir = DEFAULT_CACHE_DIR
        cache_key = model_id.replace("/", "_").lower()

        file_name, ort_input, input_name = load_model_and_input(model_id)

        vaip_config = DEFAULT_VAIP_CONFIG
        outputs_ipu, outputs_cpu = self.prepare_outputs(
            model_id, RyzenAIModelForSemanticSegmentation, ort_input, vaip_config, cache_dir, cache_key, file_name
        )

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        current_ops = self.get_ops(cache_dir, cache_key)
        baseline_ops = self.get_baseline_ops(cache_key)

        self.assertEqual(baseline_ops["all"], current_ops["all"], f"Total operators do not match! {current_ops}")
        self.assertEqual(baseline_ops["dpu"], current_ops["dpu"], f"DPU operators do not match! {current_ops}")

        gc.collect()


class RyzenAIModelForImageToImageIntegrationTest(unittest.TestCase, RyzenAITestCaseMixin):
    @parameterized.expand(RYZEN_PREQUANTIZED_MODEL_IMAGE_TO_IMAGE)
    @pytest.mark.prequantized_model_test
    def test_model(self, model_id):
        cache_dir = DEFAULT_CACHE_DIR
        cache_key = model_id.replace("/", "_").lower()

        file_name, ort_input, input_name = load_model_and_input(model_id)

        vaip_config = DEFAULT_VAIP_CONFIG
        outputs_ipu, outputs_cpu = self.prepare_outputs(
            model_id, RyzenAIModelForImageToImage, ort_input, vaip_config, cache_dir, cache_key, file_name
        )

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        current_ops = self.get_ops(cache_dir, cache_key)
        baseline_ops = self.get_baseline_ops(cache_key)

        self.assertEqual(baseline_ops["all"], current_ops["all"], f"Total operators do not match! {current_ops}")
        self.assertEqual(baseline_ops["dpu"], current_ops["dpu"], f"DPU operators do not match! {current_ops}")

        gc.collect()


class RyzenAIModelForCustomTasksIntegrationTest(unittest.TestCase, RyzenAITestCaseMixin):
    @parameterized.expand(RYZEN_PREQUANTIZED_MODEL_CUSTOM_TASKS)
    @pytest.mark.prequantized_model_test
    def test_model(self, model_id):
        cache_dir = DEFAULT_CACHE_DIR
        cache_key = model_id.replace("/", "_").lower()

        file_name, ort_input, input_name = load_model_and_input(model_id)
        ort_input = {input_name: ort_input}

        vaip_config = DEFAULT_VAIP_CONFIG
        outputs_ipu, outputs_cpu = self.prepare_outputs(
            model_id, RyzenAIModelForCustomTasks, ort_input, vaip_config, cache_dir, cache_key, file_name
        )

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        current_ops = self.get_ops(cache_dir, cache_key)
        baseline_ops = self.get_baseline_ops(cache_key)

        self.assertEqual(baseline_ops["all"], current_ops["all"], f"Total operators do not match! {current_ops}")
        self.assertEqual(baseline_ops["dpu"], current_ops["dpu"], f"DPU operators do not match! {current_ops}")

        gc.collect()


class RyzenAIModelForCausalLMIntegrationTest(unittest.TestCase, RyzenAITestCaseMixin):
    SUPPORTED_ARCHITECTURES = {
        "opt",
        "llama",
        "mistral",
    }

    @parameterized.expand(
        get_models_to_test(
            PYTORCH_MODELS,
            library_name="transformers",
            supported_archs=SUPPORTED_ARCHITECTURES,
            tasks="text-generation-with-past",
        )
    )
    @pytest.mark.brevitas_quantized_decoder_llms_test
    def test_model(self, test_name: str, model_type: str, model_id: str, task: str):
        dataset_name = "wikitext2"
        num_calib_samples = 10

        quantization_dir = tempfile.TemporaryDirectory()

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        quantization_config = BrevitasQuantizationConfig(
            apply_gptq=False,
            apply_weight_equalization=False,
            activations_equalization="layerwise",
            is_static=False,
            weights_symmetric=True,
            activations_symmetric=False,
        )

        # Load the data for calibration and evaluation.
        calibration_dataset = get_dataset_for_model(
            model_id,
            qconfig=quantization_config,
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            nsamples=num_calib_samples,
            seqlen=64,
            split="train",
            device=None,
            fuse_sequences=False,
        )

        # quantize model
        quantizer = BrevitasQuantizer.from_pretrained(model_id, device_map="cpu")

        quantized_model = quantizer.quantize(quantization_config, calibration_dataset)

        # export model
        onnx_export_from_quantized_model(quantized_model, quantization_dir.name)

        # inference
        cache_dir = DEFAULT_CACHE_DIR
        cache_key = model_id.replace("/", "_").lower()
        vaip_config = DEFAULT_VAIP_CONFIG_TRANSFORMERS

        prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = tokenizer(prompt, return_tensors="np")
        ort_inputs = {key: np.array(inputs[key], dtype=np.int64) for key in inputs.keys()}

        if model_type in {"llama", "mistral"}:
            attention_mask = torch.tensor(inputs["attention_mask"])
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            ort_inputs["position_ids"] = position_ids.numpy()

        outputs_ipu, outputs_cpu = self.prepare_outputs(
            quantization_dir.name, RyzenAIModelForCausalLM, ort_inputs, vaip_config, cache_dir, cache_key
        )

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        current_ops = self.get_ops(cache_dir, cache_key)
        baseline_ops = self.get_baseline_ops(cache_key)

        self.assertEqual(baseline_ops["all"], current_ops["all"], f"Total operators do not match! {current_ops}")
        self.assertEqual(
            baseline_ops["matmulinteger"], current_ops["matmulinteger"], f"MATMULINTEGERs do not match! {current_ops}"
        )

        quantization_dir.cleanup()
