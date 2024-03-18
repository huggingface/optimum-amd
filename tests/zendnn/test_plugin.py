import unittest

import torch
import torch_zendnn_plugin  # noqa: F401
from parameterized import parameterized

from optimum.exporters.tasks import TasksManager
from optimum.utils import logging


logger = logging.get_logger()


SEED = 42
SUPPORTED_MODELS_TINY = {
    # text encoder
    "bert": {"hf-internal-testing/tiny-random-bert": ["text-classification"]},
    "xlnet": {"hf-internal-testing/tiny-random-xlnet": ["text-classification"]},
    "roberta": {"hf-internal-testing/tiny-random-roberta": ["text-classification"]},
    "distilbert": {"hf-internal-testing/tiny-random-distilbert": ["text-classification"]},
    # image encoder
    "vit": {"hf-internal-testing/tiny-random-ViTForImageClassification": ["image-classification"]},
}
SUPPORTED_MODELS_TINY_TEXT_GENERATION = {
    # text decoder
    "gemma": {"fxmarty/tiny-random-GemmaForCausalLM": ["text-generation"]},
    "phi": {"hf-internal-testing/tiny-random-PhiForCausalLM": ["text-generation"]},
    "opt": {"hf-internal-testing/tiny-random-OPTForCausalLM": ["text-generation"]},
    "mpt": {"hf-internal-testing/tiny-random-MptForCausalLM": ["text-generation"]},
    "gpt2": {"hf-internal-testing/tiny-random-GPT2LMHeadModel": ["text-generation"]},
    "gptj": {"hf-internal-testing/tiny-random-GPTJForCausalLM": ["text-generation"]},
    "bloom": {"hf-internal-testing/tiny-random-BloomForCausalLM": ["text-generation"]},
    "llama": {"hf-internal-testing/tiny-random-LlamaForCausalLM": ["text-generation"]},
    "falcon": {"hf-internal-testing/tiny-random-FalconForCausalLM": ["text-generation"]},
    "mistral": {"hf-internal-testing/tiny-random-MistralForCausalLM": ["text-generation"]},
    "gpt-neo": {"hf-internal-testing/tiny-random-GPTNeoForCausalLM": ["text-generation"]},
    "gpt-neox": {"hf-internal-testing/tiny-random-GPTNeoXForCausalLM": ["text-generation"]},
    "gpt-bigcode": {"hf-internal-testing/tiny-random-GPTBigCodeForCausalLM": ["text-generation"]},
    # "yi": {"hf-internal-testing/tiny-random-Yi": ["text-generation"]},  # just llama
    # "vicuna": {"hf-internal-testing/tiny-random-vicuna": ["text-generation"]},  # just llama
    # "zephyr": {"hf-internal-testing/tiny-random-zephyr": ["text-generation"]},  # just mistral
    # "santacoder": {"hf-internal-testing/tiny-random-SantaCoder": ["text-generation"]},  # just gpt2
    # "distilgpt2": {"hf-internal-testing/tiny-random-GPT2LMHeadModel": ["text-generation"]},  # just gpt2
    # "starcoder2": {"hf-internal-testing/tiny-random-Starcoder2ForCausalLM": ["text-generation"]}, # next transformers release
    # text encoder-decoder
    "t5": {"hf-internal-testing/tiny-random-t5": ["text2text-generation"]},
    "bart": {"hf-internal-testing/tiny-random-bart": ["text2text-generation"]},
    # # automatic speech recognition
    "whisper": {"hf-internal-testing/tiny-random-WhisperForConditionalGeneration": ["automatic-speech-recognition"]},
}

SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION = {
    # # stable diffusion
    "stable-diffusion": {"hf-internal-testing/tiny-stable-diffusion-torch": ["stable-diffusion"]},
}


def load_model_or_pipe(model_name: str, task: str):
    model_or_pipe = TasksManager.get_model_from_task(task=task, model_name_or_path=model_name, framework="pt")
    return model_or_pipe


def get_dummy_inputs(task: str):
    if task in ["fill-mask", "text-generation", "text-classification"]:
        dummy_inputs = {
            "input_ids": torch.randint(low=0, high=2, size=(2, 10), dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

    elif task in ["image-classification"]:
        dummy_inputs = {
            "pixel_values": torch.rand(size=(2, 3, 30, 30), dtype=torch.float),
        }

    elif task in ["text2text-generation"]:
        dummy_inputs = {
            "input_ids": torch.randint(low=0, high=2, size=(2, 10), dtype=torch.long),
            "decoder_input_ids": torch.randint(low=0, high=2, size=(2, 10), dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

    elif task in ["automatic-speech-recognition"]:
        dummy_inputs = {
            "input_values": torch.rand(size=(2, 10), dtype=torch.float),
        }

    elif task in ["image-to-text"]:
        dummy_inputs = {
            "pixel_values": torch.rand(size=(2, 3, 14, 14), dtype=torch.float),
            "input_ids": torch.randint(low=0, high=2, size=(2, 10), dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

    elif task in ["stable-diffusion"]:
        dummy_inputs = {
            "prompt": ["This is test prompt 1", "This is test prompt 2"],
        }

    else:
        raise ValueError(f"Task {task} not supported")

    return dummy_inputs


def torch_compile_and_forward(model, inputs, backend):
    model = torch.compile(model, backend=backend)
    out_logits = model(**inputs).logits

    return out_logits


def torch_compile_and_generate(model, inputs, backend):
    model = torch.compile(model, backend=backend)
    out_ids = model.generate(**inputs, min_new_tokens=10, max_new_tokens=10, pad_token_id=0)

    return out_ids


def torch_compile_and_diffuse(pipe, inputs, backend):
    pipe.unet = torch.compile(pipe.unet, backend=backend)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend=backend)
    out_images = pipe(**inputs, num_inference_steps=2, output_type="pt").images

    return out_images


class TestZenDNNPlugin(unittest.TestCase):
    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_text_classification_model(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                torch.manual_seed(SEED)
                inputs = get_dummy_inputs(task)
                model = load_model_or_pipe(model_id, task)

                inductor_logits = torch_compile_and_forward(model, inputs, backend="inductor")
                zentorch_logits = torch_compile_and_forward(model, inputs, backend="zentorch")
                torch.testing.assert_close(inductor_logits, zentorch_logits, rtol=1e-3, atol=1e-5)

                logger.info(f"Model {model_id} for task {task} passed the test.")

    @parameterized.expand(SUPPORTED_MODELS_TINY_TEXT_GENERATION.keys())
    def test_text_generation_model(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY_TEXT_GENERATION[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                torch.manual_seed(SEED)
                inputs = get_dummy_inputs(task)
                model = load_model_or_pipe(model_id, task)

                inductor_logits = torch_compile_and_forward(model, inputs, backend="inductor")
                zentorch_logits = torch_compile_and_forward(model, inputs, backend="zentorch")
                torch.testing.assert_close(inductor_logits, zentorch_logits, rtol=1e-3, atol=1e-5)

                inductor_ids = torch_compile_and_generate(model, inputs, backend="inductor")
                zentorch_ids = torch_compile_and_generate(model, inputs, backend="zentorch")
                torch.testing.assert_close(inductor_ids, zentorch_ids, rtol=1e-3, atol=1e-5)

                logger.info(f"Model {model_id} for task {task} passed the test.")

    @parameterized.expand(SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION.keys())
    def test_image_diffusion_pipe(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                torch.manual_seed(SEED)
                inputs = get_dummy_inputs(task)
                pipe = load_model_or_pipe(model_id, task)

                inductor_images = torch_compile_and_diffuse(pipe, inputs, backend="inductor")
                zentorch_images = torch_compile_and_diffuse(pipe, inputs, backend="zentorch")
                torch.testing.assert_close(inductor_images, zentorch_images, rtol=1e-3, atol=1e-5)

                logger.info(f"Model {model_id} for task {task} passed the test.")
