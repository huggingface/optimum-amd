# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import unittest
from functools import partial
from typing import Dict

import pytest
import timm
import torch
from datasets import load_dataset
from parameterized import parameterized
from testing_utils import PYTORCH_TIMM_MODEL, PYTORCH_TIMM_MODEL_SUBSET

from optimum.amd.ryzenai import (
    AutoQuantizationConfig,
    RyzenAIModelForImageClassification,
    RyzenAIOnnxQuantizer,
)
from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager
from transformers import PretrainedConfig
from transformers.testing_utils import slow


def _get_models_to_test(export_models_dict: Dict, library_name: str = "timm"):
    models_to_test = []
    for model_type, model_names_tasks in export_models_dict.items():
        task_config_mapping = TasksManager.get_supported_tasks_for_model_type(
            model_type, "onnx", library_name=library_name
        )

        if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
            tasks = list(task_config_mapping.keys())
            model_tasks = {model_names_tasks: tasks}
        else:
            model_tasks = model_names_tasks  # possibly, test different tasks on different models

        for model_name, tasks in model_tasks.items():
            for task in tasks:
                models_to_test.append(
                    (
                        f"{model_type}_{task}_{model_name}",
                        model_type,
                        model_name,
                        task,
                    )
                )

        return sorted(models_to_test)


class TestTimmQuantization(unittest.TestCase):
    def _quantize(
        self,
        model_name: str,
    ):
        dataset_name = "imagenet-1k"
        num_calib_samples = 10

        export_dir = tempfile.TemporaryDirectory()
        quantization_dir = tempfile.TemporaryDirectory()

        # export
        main_export(
            model_name_or_path=model_name,
            output=export_dir.name,
            task="image-classification",
            opset=13,
            batch_size=1,
            no_dynamic_axes=True,
        )
        config = PretrainedConfig.from_pretrained(export_dir.name)

        # preprocess config
        data_config = timm.data.resolve_data_config(pretrained_cfg=config.to_dict())
        transforms = timm.data.create_transform(**data_config, is_training=False)

        def preprocess_fn(ex, transforms):
            image = ex["image"]
            if image.mode == "L":
                # Three channels.
                print("WARNING: converting greyscale to RGB")
                image = image.convert("RGB")
            pixel_values = transforms(image)

            return {"pixel_values": pixel_values}

        # quantize model
        quantizer = RyzenAIOnnxQuantizer.from_pretrained(export_dir.name)
        quantization_config = AutoQuantizationConfig.ipu_cnn_config()

        train_calibration_dataset = quantizer.get_calibration_dataset(
            "imagenet-1k",
            preprocess_function=partial(preprocess_fn, transforms=transforms),
            num_samples=num_calib_samples,
            dataset_split="train",
            preprocess_batch=False,
            streaming=True,
        )

        quantizer.quantize(
            quantization_config=quantization_config, dataset=train_calibration_dataset, save_dir=quantization_dir.name
        )

        def run(use_cpu_runner, compile_reserve_const_data):
            os.environ["XLNX_ENABLE_CACHE"] = "0"
            os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"

            os.environ["USE_CPU_RUNNER"] = "1" if use_cpu_runner else "0"
            os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1" if compile_reserve_const_data else "0"

            ryzen_model = RyzenAIModelForImageClassification.from_pretrained(
                quantization_dir.name,
                vaip_config=".\\tests\\ryzenai\\vaip_config.json",
            )

            evaluation_set = load_dataset(dataset_name, split="validation", streaming=True, trust_remote_code=True)
            data = next(iter(evaluation_set))

            pixel_values = preprocess_fn(data, transforms)["pixel_values"]
            outputs = ryzen_model(pixel_values.unsqueeze(0))

            return outputs

        output_ipu = run(use_cpu_runner=0, compile_reserve_const_data=0)
        output_cpu = run(use_cpu_runner=1, compile_reserve_const_data=1)

        self.assertTrue(torch.allclose(output_ipu.logits, output_cpu.logits, atol=1e-4))

        export_dir.cleanup()
        quantization_dir.cleanup()

    @parameterized.expand(_get_models_to_test(PYTORCH_TIMM_MODEL_SUBSET, library_name="timm"))
    def test_timm_quantization_subset(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
    ):
        self._quantize(model_name=model_name)

    @parameterized.expand(_get_models_to_test(PYTORCH_TIMM_MODEL, library_name="timm"))
    @pytest.mark.quant_test
    @slow
    def test_timm_quantization(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
    ):
        self._quantize(model_name=model_name)
