import unittest

import torch
import torch_zendnn_plugin
from parameterized import parameterized

from optimum.exporters.tasks import TasksManager
from optimum.utils import logging
from transformers import AutoConfig


logger = logging.get_logger()


SEED = 42
SUPPORTED_MODELS_TINY = {
    "bert": {"hf-internal-testing/tiny-random-bert": ["feature-extraction", "text-classification"]},
}


def _get_all_model_ids(model_type: str):
    if isinstance(SUPPORTED_MODELS_TINY[model_type], str):
        return [SUPPORTED_MODELS_TINY[model_type]]
    else:
        return list(SUPPORTED_MODELS_TINY[model_type].keys())


def load_model(model_name: str, task: str):
    model = TasksManager.get_model_from_task(
        task=task, model_name_or_path=model_name, framework="pt", library_name="transformers"
    )

    return model


def get_dummy_inputs(model_type: str, model_name: str, task: str):
    config = AutoConfig.from_pretrained(model_name)
    onnx_config = TasksManager.get_exporter_config_constructor(
        exporter="onnx", task=task, model_type=model_type, model_name=model_name
    )(config)

    dummy_inputs = onnx_config.generate_dummy_inputs()

    return dummy_inputs


class TestZenDNN(unittest.TestCase):
    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_torch_compile(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            for task in SUPPORTED_MODELS_TINY[model_type][model_id]:
                model = load_model(model_id, task)
                torch.compile(model, backend="zentorch")

    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_logits(self, model_type: str):
        for model_id in _get_all_model_ids(model_type):
            for task in SUPPORTED_MODELS_TINY[model_type][model_id]:
                dummy_inputs = get_dummy_inputs(model_type, model_id, task)
                model = load_model(model_id, task)
                model.eval()

                torch._dynamo.reset()
                inductor_model = torch.compile(model, backend="inductor")
                with torch.inference_mode():
                    inductor_out = inductor_model(**dummy_inputs)

                torch._dynamo.reset()
                zentorch_model = torch.compile(model, backend="zentorch")
                with torch.inference_mode():
                    zentorch_out = zentorch_model(**dummy_inputs)

                for key in inductor_out:
                    self.assertTrue(torch.allclose(inductor_out[key], zentorch_out[key], rtol=1e-3))
