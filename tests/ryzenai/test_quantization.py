# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from typing import Dict

import torch
from datasets import Dataset
from parameterized import parameterized
from testing_utils import PYTORCH_TIMM_MODEL

from optimum.amd.ryzenai import QuantizationConfig, RyzenAIOnnxQuantizer
from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager
from transformers import PretrainedConfig


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
    @parameterized.expand(_get_models_to_test(PYTORCH_TIMM_MODEL))
    def test_quantization_integration(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
    ):
        export_dir = tempfile.TemporaryDirectory()
        quantization_dir = tempfile.TemporaryDirectory()

        main_export(
            model_name_or_path=model_name,
            output=export_dir.name,
            task=task,
        )

        quantizer = RyzenAIOnnxQuantizer.from_pretrained(export_dir.name)

        # NOTE: Vitis AI Quantizer 3.5 is broken with enable_dpu=True.
        quantization_config = QuantizationConfig(enable_dpu=False)

        cfg = PretrainedConfig.from_pretrained(export_dir.name)
        if (
            hasattr(cfg, "pretrained_cfg")
            and "input_size" in cfg.pretrained_cfg
            and len(cfg.pretrained_cfg["input_size"]) == 3
        ):
            cfg_input_size = cfg.pretrained_cfg["input_size"]
            shape = (cfg_input_size[0], cfg_input_size[1], cfg_input_size[2])
        else:
            shape = (3, 224, 224)

        my_dict = {"pixel_values": [torch.rand(shape) for i in range(10)]}
        dataset = Dataset.from_dict(my_dict)
        dataset = dataset.with_format("torch")

        quantizer.quantize(quantization_config=quantization_config, dataset=dataset, save_dir=quantization_dir.name)

        export_dir.cleanup()
        quantization_dir.cleanup()
