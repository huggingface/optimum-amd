# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import torch
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from onnx import numpy_helper
from parameterized import parameterized
from testing_utils import SUPPORTED_MODELS_TINY, VALIDATE_EXPORT_ON_SHAPES, get_quantized_model

from optimum.exporters import TasksManager
from optimum.exporters.onnx import (
    export_models,
    validate_models_outputs,
)
from optimum.exporters.onnx.utils import _get_submodels_and_onnx_configs
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.utils.testing_utils import grid_parameters
from transformers.modeling_utils import get_parameter_dtype


def _get_models_to_test(export_models_dict: Dict, library_name: str = "transformers"):
    models_to_test = []
    for model_type, model_names_tasks in export_models_dict.items():
        model_type = model_type.replace("_", "-")
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
                onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                    model_type=model_type,
                    exporter="onnx",
                    task=task,
                    model_name=model_name,
                    library_name=library_name,
                )

                models_to_test.append(
                    (f"{model_type}_{task}_DQ", model_type, model_name, task, onnx_config_constructor, False)
                )
                models_to_test.append(
                    (f"{model_type}_{task}_QDQ", model_type, model_name, task, onnx_config_constructor, True)
                )
    return sorted(models_to_test)


def export_and_validate(
    model: torch.nn.Module,
    task: str,
    export_output_dir: str,
    onnx_config_class_constructor,
    shapes_to_validate: Dict,
    qdq_weights: bool,
):
    export_manager = StdQCDQONNXManager
    if qdq_weights:
        export_manager.change_weight_export(export_weight_q_node=True)

    with torch.no_grad(), brevitas_proxy_export_mode(model, export_manager=export_manager):
        library_name = TasksManager._infer_library_from_model(model)
        framework = "pt"
        dtype = get_parameter_dtype(model) if framework == "pt" else model.dtype

        if "bfloat16" in str(dtype):
            float_dtype = "bf16"
        elif "float16" in str(dtype):
            float_dtype = "fp16"
        else:
            float_dtype = "fp32"

        output = Path(export_output_dir)
        if not output.exists():
            output.mkdir(parents=True)

        onnx_config, models_and_onnx_configs = _get_submodels_and_onnx_configs(
            model=model,
            task=task,
            monolith=False,
            custom_onnx_configs={},
            custom_architecture=False,
            float_dtype=float_dtype,
            _variant="default",
            library_name=library_name,
        )

        model.config.save_pretrained(output)

        onnx_files_subpaths = [key + ".onnx" for key in models_and_onnx_configs.keys()]

        input_shapes = {}
        for input_name in DEFAULT_DUMMY_SHAPES.keys():
            input_shapes[input_name] = DEFAULT_DUMMY_SHAPES[input_name]

        _, onnx_outputs = export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output_dir=output,
            output_names=onnx_files_subpaths,
            input_shapes=input_shapes,
            device="cpu",
            dtype=float_dtype,
            no_dynamic_axes=False,
            do_constant_folding=False,
        )

    onnx_config = onnx_config_class_constructor(model.config)

    input_shapes_iterator = grid_parameters(shapes_to_validate, yield_dict=True, add_test_name=False)

    for input_shapes in input_shapes_iterator:
        validate_models_outputs(
            models_and_onnx_configs=models_and_onnx_configs,
            onnx_named_outputs=onnx_outputs,
            atol=onnx_config.ATOL_FOR_VALIDATION,
            output_dir=output,
            input_shapes=input_shapes,
            use_subprocess=False,
        )


class TestOnnxExport(unittest.TestCase):
    @parameterized.expand(_get_models_to_test(SUPPORTED_MODELS_TINY))
    def test_dynamic_quantization(
        self, test_name, model_type, model_name, task, onnx_config_class_constructor, qdq_weights: bool
    ):
        model = get_quantized_model(
            model_name,
            is_static=False,
            apply_gptq=False,
            apply_weight_equalization=False,
            activations_equalization=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Check that PyTorch and ORT outputs match on various shapes.
            export_and_validate(
                model=model,
                task=task,
                export_output_dir=tmpdir,
                onnx_config_class_constructor=onnx_config_class_constructor,
                shapes_to_validate=VALIDATE_EXPORT_ON_SHAPES,
                qdq_weights=qdq_weights,
            )

            onnx_model = onnx.load(os.path.join(tmpdir, "model.onnx"))

            if qdq_weights:
                for node in onnx_model.graph.node:
                    if node.op_type == "Constant":
                        for attrib in node.attribute:
                            new_array = numpy_helper.to_array(attrib.t)
                            if len(new_array.shape) >= 2 and new_array.dtype in [np.uint8, np.int8]:
                                break
                else:
                    self.assertTrue(False, "Did not found an int8/uint8 serialized weight")
            else:
                for node in onnx_model.graph.node:
                    if node.op_type == "Constant":
                        for attrib in node.attribute:
                            new_array = numpy_helper.to_array(attrib.t)
                            if len(new_array.shape) >= 2 and new_array.dtype in [np.uint8, np.int8]:
                                self.assertTrue(False, "Found uint8/int8 serialized weights while we should not")
