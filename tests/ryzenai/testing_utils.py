# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import json
import os
from typing import Dict

from optimum.exporters import TasksManager
from transformers import set_seed


SEED = 42

BASELINE_OPERATORS_JSON = os.path.normpath("./tests/ryzenai/operators_baseline.json")  # For RyzenSDK 1.1

DEFAULT_CACHE_DIR = "ryzen_cache"


def get_models_to_test(
    export_models_dict: Dict, library_name: str = "timm", supported_archs: list = None, tasks: list = None
):
    models_to_test = []
    for model_type, model_names_tasks in export_models_dict.items():
        if supported_archs is not None and model_type not in supported_archs:
            continue

        if tasks is not None and not isinstance(tasks, list):
            tasks = [tasks]

        if tasks is None:
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(
                model_type, "onnx", library_name=library_name
            )

            if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
                tasks = list(task_config_mapping.keys())
                model_tasks = {model_names_tasks: tasks}
            else:
                model_tasks = model_names_tasks  # possibly, test different tasks on different models
        else:
            model_tasks = {model_names_tasks: tasks}

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


def parse_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        result = {"all": 0, "dpu": 0, "cpu": 0, "matmulinteger": 0}
        for entry in data["deviceStat"]:
            result[entry["name"].lower()] = entry["nodeNum"]
        return result


class RyzenAITestCaseMixin:
    def run_model(
        self,
        model_class,
        model_id,
        ort_input,
        use_cpu_runner,
        compile_reserve_const_data,
        cache_dir=None,
        cache_key=None,
        file_name=None,
    ):
        os.environ["XLNX_ENABLE_CACHE"] = "0"
        os.environ["XLNX_USE_SHARED_CONTEXT"] = "1"
        os.environ["USE_CPU_RUNNER"] = "1" if use_cpu_runner else "0"
        os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1" if compile_reserve_const_data else "0"

        provider_options = {}
        if cache_dir:
            provider_options["cacheDir"] = cache_dir
        if cache_key:
            provider_options["cacheKey"] = cache_key

        model_instance = model_class.from_pretrained(model_id, file_name=file_name, provider_options=provider_options)

        if isinstance(ort_input, dict):
            outputs = model_instance(**ort_input)
        else:
            outputs = model_instance(ort_input)

        return outputs

    def prepare_outputs(self, model_id, model_class, ort_input, cache_dir=None, cache_key=None, file_name=None):
        set_seed(SEED)
        output_ipu = self.run_model(
            model_class,
            model_id,
            ort_input,
            use_cpu_runner=0,
            compile_reserve_const_data=0,
            cache_dir=cache_dir,
            cache_key=cache_key,
            file_name=file_name,
        )

        output_cpu = self.run_model(
            model_class,
            model_id,
            ort_input,
            use_cpu_runner=1,
            compile_reserve_const_data=1,
            file_name=file_name,
        )

        return output_ipu, output_cpu

    def get_ops(self, cache_dir, cache_key):
        result = parse_json(os.path.join(cache_dir, cache_key, "vitisai_ep_report.json"))
        return result

    def get_baseline_ops(self, key):
        with open(BASELINE_OPERATORS_JSON, "r") as json_file:
            data = json.load(json_file)
            return data[key]
