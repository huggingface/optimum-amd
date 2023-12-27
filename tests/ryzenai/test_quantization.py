# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path
from typing import Dict

import evaluate
import timm
import torch
from datasets import Dataset, load_dataset
from parameterized import parameterized
from testing_utils import PYTORCH_TIMM_MODEL
from tqdm import tqdm

from optimum.amd.ryzenai import QuantizationConfig, RyzenAIModelForImageClassification, RyzenAIOnnxQuantizer
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

        quantization_config = QuantizationConfig(enable_dpu=True)

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

    def test_quantization_quality(self):
        dataset_name = "imagenet-1k"
        model_name = "timm/res2next50.in1k"

        timm_model = timm.create_model("res2next50.in1k", pretrained=True)
        timm_model = timm_model.eval()

        export_dir = tempfile.TemporaryDirectory()
        quantization_dir = tempfile.TemporaryDirectory()

        main_export(model_name_or_path=model_name, output=export_dir.name, task="image-classification", opset=13)

        static_onnx_path = RyzenAIModelForImageClassification.reshape(
            Path(export_dir.name) / "model.onnx",
            input_shape_dict={"pixel_values": [1, 3, 224, 224]},
            output_shape_dict={"logits": [1, 1000]},
        )

        quantizer = RyzenAIOnnxQuantizer.from_pretrained(export_dir.name, file_name=static_onnx_path.name)

        quantization_config = QuantizationConfig()

        calibration_set = load_dataset(dataset_name, split="train", streaming=True)
        calibration_data = {"pixel_values": []}

        data_config = timm.data.resolve_model_data_config(timm_model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        iterable_calibration_set = iter(calibration_set)
        for i in range(100):
            pil_image = next(iterable_calibration_set)["image"]
            if pil_image.mode == "L":
                # Three channels.
                print("WARNING: converting greyscale to RGB")
                pil_image = pil_image.convert("RGB")
            pixel_values = transforms(pil_image)
            calibration_data["pixel_values"].append(pixel_values)

        calibration_dataset = Dataset.from_dict(calibration_data)
        calibration_dataset = calibration_dataset.with_format("torch")

        quantizer.quantize(
            quantization_config=quantization_config, dataset=calibration_dataset, save_dir=quantization_dir.name
        )

        evaluation_set = load_dataset(dataset_name, split="validation", streaming=True)

        # TODO: Evaluate on VitisAIExecutionProvider.
        ryzen_model = RyzenAIModelForImageClassification.from_pretrained(
            quantization_dir.name,
            vaip_config="C:\\Users\\Mohit\\Work\\optimum-amd-hf\\vaip_config.json",
            provider="VitisAIExecutionProvider",
        )

        iterable_evaluation_set = iter(evaluation_set)

        timm_evals = []
        quantized_evals = []
        reference_labels = []
        for i in tqdm(range(300), desc="Inference..."):
            data = next(iterable_evaluation_set)

            reference_labels.append(data["label"])

            pil_image = data["image"]

            if pil_image.mode == "L":
                # Three channels.
                print("WARNING: converting greyscale to RGB")
                pil_image = pil_image.convert("RGB")
            pixel_values = transforms(pil_image).unsqueeze(0)

            res = ryzen_model(pixel_values)
            predicted_id = torch.argmax(res.logits, dim=-1).item()
            quantized_evals.append(predicted_id)

            res = timm_model(pixel_values)
            predicted_id = torch.argmax(res, dim=-1).item()
            timm_evals.append(predicted_id)

        accuracy = evaluate.load("accuracy")

        quantized_accuracy = accuracy.compute(references=reference_labels, predictions=quantized_evals)["accuracy"]
        timm_accuracy = accuracy.compute(references=reference_labels, predictions=timm_evals)["accuracy"]

        print("quantized_accuracy", quantized_accuracy)
        print("timm_accuracy", timm_accuracy)

        # TODO: Should get to 1% accuracy drop.
        self.assertTrue((timm_accuracy - quantized_accuracy) / timm_accuracy < 0.05)

        export_dir.cleanup()
        quantization_dir.cleanup()
