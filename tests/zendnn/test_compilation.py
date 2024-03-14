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
    "t5": {"hf-internal-testing/tiny-random-t5": ["text2text-generation"]},
    "bart": {"hf-internal-testing/tiny-random-bart": ["text2text-generation"]},
    "bert": {"hf-internal-testing/tiny-random-bert": ["fill-mask"]},
    "roberta": {"hf-internal-testing/tiny-random-roberta": ["fill-mask"]},
    "distilbert": {"hf-internal-testing/tiny-random-distilbert": ["fill-mask"]},
    "gpt2": {"hf-internal-testing/tiny-random-GPT2LMHeadModel": ["text-generation"]},
    "gptj": {"hf-internal-testing/tiny-random-GPTJForCausalLM": ["text-generation"]},
    "gpt-neo": {"hf-internal-testing/tiny-random-GPTNeoForCausalLM": ["text-generation"]},
    "gpt-neox": {"hf-internal-testing/tiny-random-GPTNeoXForCausalLM": ["text-generation"]},
    "opt": {"hf-internal-testing/tiny-random-OPTForCausalLM": ["text-generation"]},
    "mpt": {"hf-internal-testing/tiny-random-MptForCausalLM": ["text-generation"]},
    "llama": {"hf-internal-testing/tiny-random-LlamaForCausalLM": ["text-generation"]},
    "phi": {"hf-internal-testing/tiny-random-PhiForCausalLM": ["text-generation"]},
    "mistral": {"hf-internal-testing/tiny-random-MistralForCausalLM": ["text-generation"]},
    "bloom": {"hf-internal-testing/tiny-random-BloomForCausalLM": ["text-generation"]},
    "falcon": {"hf-internal-testing/tiny-random-FalconForCausalLM": ["text-generation"]},
    "gpt-bigcode": {"hf-internal-testing/tiny-random-GPTBigCodeForCausalLM": ["text-generation"]},
    # "Yi": {"hf-internal-testing/tiny-random-Yi": ["text-generation"]}, # just llama
    # "vicuna": {"hf-internal-testing/tiny-random-vicuna": ["text-generation"]}, #just llama
    # "zephyr": {"hf-internal-testing/tiny-random-zephyr": ["text-generation"]}, # just mistral
    # "SantaCoder": {"hf-internal-testing/tiny-random-SantaCoder": ["text-generation"]}, # just gpt2
    # "distilgpt2": {"hf-internal-testing/tiny-random-GPT2LMHeadModel": ["text-generation"]}, # just gpt2
    # "xlnet": {"hf-internal-testing/tiny-random-xlnet": ["fill-mask"]}, # missing onnx config
    # "gemma": {"fxmarty/tiny-random-GemmaForCausalLM": ["text-generation"]}, # missing onnx config
    # "blip": {"hf-internal-testing/tiny-random-BlipModel": ["image-to-text"]}, # missing onnx config
    # "blip2": {"hf-internal-testing/tiny-random-Blip2Model": ["image-to-text"]}, # missing onnx config
    # "starcoder2": {"hf-internal-testing/tiny-random-Starcoder2ForCausalLM": ["text-generation"]}, # next transformers release
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
        exporter="onnx", task=task, model_type=model_type, model_name=model_name, library_name="transformers"
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
    def test_model_logits(self, model_type: str):
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

                torch.testing.assert_close(inductor_out.logits, zentorch_out.logits, rtol=1e-3, atol=1e-5)
