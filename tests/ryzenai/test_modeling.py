# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import gc
import os
import unittest

import huggingface_hub
import numpy as np
import onnx
from parameterized import parameterized
from testing_utils import (
    RYZEN_PRETRAINED_MODEL_CUSTOM_TASKS,
    RYZEN_PRETRAINED_MODEL_IMAGE_CLASSIFICATION,
    RYZEN_PRETRAINED_MODEL_IMAGE_SEGMENTATION,
    RYZEN_PRETRAINED_MODEL_IMAGE_TO_IMAGE,
    RYZEN_PRETRAINED_MODEL_OBJECT_DETECTION,
)

from optimum.amd.ryzenai import (
    RyzenAIModelForCustomTasks,
    RyzenAIModelForImageClassification,
    RyzenAIModelForImageSegmentation,
    RyzenAIModelForImageToImage,
)
from optimum.utils import (
    DummyInputGenerator,
    logging,
)
from transformers import set_seed


logger = logging.get_logger()


SEED = 42


class RyzenAIModelForImageClassificationIntegrationTest(unittest.TestCase):
    @parameterized.expand(RYZEN_PRETRAINED_MODEL_IMAGE_CLASSIFICATION)
    def test_model(self, model_id):
        set_seed(SEED)

        all_files = huggingface_hub.list_repo_files(model_id, repo_type="model")
        file_name = [name for name in all_files if name.endswith(".onnx")][0]

        onnx_model_path = huggingface_hub.hf_hub_download(model_id, file_name)
        model = onnx.load(onnx_model_path)

        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        input_shape = [dim.dim_value for dim in input_shape]
        input_shape[0] = 1

        ort_input = DummyInputGenerator.random_float_tensor(input_shape, framework="np")

        def run(model_id, file_name, ort_input, use_cpu_runner, compile_reserve_const_data):
            os.environ["XLNX_ENABLE_CACHE"] = "0"
            os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

            os.environ["USE_CPU_RUNNER"] = "1" if use_cpu_runner else "0"
            os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1" if compile_reserve_const_data else "0"

            model = RyzenAIModelForImageClassification.from_pretrained(
                model_id, file_name=file_name, vaip_config=".\\vaip_config.json"
            )

            outputs = model(ort_input)

            return outputs

        output_ipu = run(model_id, file_name, ort_input, use_cpu_runner=0, compile_reserve_const_data=0)
        output_cpu = run(model_id, file_name, ort_input, use_cpu_runner=1, compile_reserve_const_data=1)

        self.assertIn("logits", output_ipu)
        self.assertIn("logits", output_cpu)

        self.assertTrue(np.allclose(output_ipu.logits, output_cpu.logits, atol=1e-4))

        gc.collect()


class RyzenAIModelForObjectDetectionIntegrationTest(unittest.TestCase):
    @parameterized.expand(RYZEN_PRETRAINED_MODEL_OBJECT_DETECTION)
    def test_model(self, model_id):
        set_seed(SEED)

        all_files = huggingface_hub.list_repo_files(model_id, repo_type="model")
        file_name = [name for name in all_files if name.endswith(".onnx")][0]

        onnx_model_path = huggingface_hub.hf_hub_download(model_id, file_name)
        model = onnx.load(onnx_model_path)

        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        input_shape = [dim.dim_value for dim in input_shape]
        input_shape[0] = 1

        ort_input = DummyInputGenerator.random_float_tensor(input_shape, framework="np")

        def run(model_id, file_name, ort_input, use_cpu_runner, compile_reserve_const_data):
            os.environ["XLNX_ENABLE_CACHE"] = "0"
            os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

            os.environ["USE_CPU_RUNNER"] = "1" if use_cpu_runner else "0"
            os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1" if compile_reserve_const_data else "0"

            model = RyzenAIModelForImageSegmentation.from_pretrained(
                model_id, file_name=file_name, vaip_config=".\\vaip_config.json"
            )

            outputs = model(ort_input)

            return outputs

        outputs_ipu = run(model_id, file_name, ort_input, use_cpu_runner=0, compile_reserve_const_data=0)
        outputs_cpu = run(model_id, file_name, ort_input, use_cpu_runner=1, compile_reserve_const_data=1)

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        gc.collect()


class RyzenAIModelForImageSegmentationIntegrationTest(unittest.TestCase):
    @parameterized.expand(RYZEN_PRETRAINED_MODEL_IMAGE_SEGMENTATION)
    def test_model(self, model_id):
        set_seed(SEED)

        all_files = huggingface_hub.list_repo_files(model_id, repo_type="model")
        file_name = [name for name in all_files if name.endswith(".onnx")][0]

        onnx_model_path = huggingface_hub.hf_hub_download(model_id, file_name)
        model = onnx.load(onnx_model_path)

        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        input_shape = [dim.dim_value for dim in input_shape]
        input_shape[0] = 1

        ort_input = DummyInputGenerator.random_float_tensor(input_shape, framework="np")

        def run(model_id, file_name, ort_input, use_cpu_runner, compile_reserve_const_data):
            os.environ["XLNX_ENABLE_CACHE"] = "0"
            os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

            os.environ["USE_CPU_RUNNER"] = "1" if use_cpu_runner else "0"
            os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1" if compile_reserve_const_data else "0"

            model = RyzenAIModelForImageSegmentation.from_pretrained(
                model_id, file_name=file_name, vaip_config=".\\vaip_config.json"
            )

            outputs = model(ort_input)

            return outputs

        outputs_ipu = run(model_id, file_name, ort_input, use_cpu_runner=0, compile_reserve_const_data=0)
        outputs_cpu = run(model_id, file_name, ort_input, use_cpu_runner=1, compile_reserve_const_data=1)

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        gc.collect()


class RyzenAIModelForImageToImageIntegrationTest(unittest.TestCase):
    @parameterized.expand(RYZEN_PRETRAINED_MODEL_IMAGE_TO_IMAGE)
    def test_model(self, model_id):
        set_seed(SEED)

        all_files = huggingface_hub.list_repo_files(model_id, repo_type="model")
        file_name = [name for name in all_files if name.endswith(".onnx")][0]

        onnx_model_path = huggingface_hub.hf_hub_download(model_id, file_name)
        model = onnx.load(onnx_model_path)

        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        input_shape = [dim.dim_value for dim in input_shape]
        input_shape[0] = 1

        ort_input = DummyInputGenerator.random_float_tensor(input_shape, framework="np")

        def run(model_id, file_name, ort_input, use_cpu_runner, compile_reserve_const_data):
            os.environ["XLNX_ENABLE_CACHE"] = "0"
            os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

            os.environ["USE_CPU_RUNNER"] = "1" if use_cpu_runner else "0"
            os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1" if compile_reserve_const_data else "0"

            model = RyzenAIModelForImageToImage.from_pretrained(
                model_id, file_name=file_name, vaip_config=".\\vaip_config.json"
            )

            outputs = model(ort_input)

            return outputs

        outputs_ipu = run(model_id, file_name, ort_input, use_cpu_runner=0, compile_reserve_const_data=0)
        outputs_cpu = run(model_id, file_name, ort_input, use_cpu_runner=1, compile_reserve_const_data=1)

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        gc.collect()


class RyzenAIModelForCustomTasksIntegrationTest(unittest.TestCase):
    @parameterized.expand(RYZEN_PRETRAINED_MODEL_CUSTOM_TASKS)
    def test_model_vision(self, model_id):
        set_seed(SEED)

        all_files = huggingface_hub.list_repo_files(model_id, repo_type="model")
        file_name = [name for name in all_files if name.endswith(".onnx")][0]

        onnx_model_path = huggingface_hub.hf_hub_download(model_id, file_name)
        model = onnx.load(onnx_model_path)

        input_name = model.graph.input[0].name
        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        input_shape = [dim.dim_value for dim in input_shape]
        input_shape[0] = 1

        ort_input = DummyInputGenerator.random_float_tensor(input_shape, framework="np")
        ort_input = {input_name: ort_input}

        def run(model_id, file_name, ort_input, use_cpu_runner, compile_reserve_const_data):
            os.environ["XLNX_ENABLE_CACHE"] = "0"
            os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

            os.environ["USE_CPU_RUNNER"] = "1" if use_cpu_runner else "0"
            os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1" if compile_reserve_const_data else "0"

            model = RyzenAIModelForCustomTasks.from_pretrained(
                model_id, file_name=file_name, vaip_config=".\\vaip_config.json"
            )

            outputs = model(**ort_input)

            return outputs

        outputs_ipu = run(model_id, file_name, ort_input, use_cpu_runner=0, compile_reserve_const_data=0)
        outputs_cpu = run(model_id, file_name, ort_input, use_cpu_runner=1, compile_reserve_const_data=1)

        for output_ipu, output_cpu in zip(outputs_ipu.values(), outputs_cpu.values()):
            self.assertTrue(np.allclose(output_ipu, output_cpu, atol=1e-4))

        gc.collect()
