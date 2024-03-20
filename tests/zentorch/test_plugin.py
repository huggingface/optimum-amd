import os
import unittest

import requests
import torch
import zentorch  # noqa: F401
from diffusers import DiffusionPipeline
from parameterized import parameterized
from PIL import Image

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    AutoTokenizer,
)
from transformers.pipelines.audio_utils import ffmpeg_read


EAGER_DEBUG = os.environ.get("EAGER_DEBUG", "0") == "1"  # set to 1 to run the model in eager mode

SEED = 42
BATCH_SIZE = 2
TEXT_GENERATION_KWARGS = {
    "min_new_tokens": 10,
    "max_new_tokens": 10,
    # use output_logits in transformers next release
    # https://github.com/huggingface/transformers/issues/14498#issuecomment-1953014058
    "output_scores": True,
    "return_dict_in_generate": True,
}
IMAGE_DIFFUSION_KWARGS = {"num_inference_steps": 2, "output_type": "pt"}

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
    # "yi": {"hf-internal-testing/tiny-random-yi": ["text-generation"]},  # just llama
    # "vicuna": {"hf-internal-testing/tiny-random-vicuna": ["text-generation"]},  # just llama
    # "zephyr": {"hf-internal-testing/tiny-random-zephyr": ["text-generation"]},  # just mistral
    # "santacoder": {"hf-internal-testing/tiny-random-santacoder": ["text-generation"]},  # just gpt2
    # "distilgpt2": {"hf-internal-testing/tiny-random-distilgpt2": ["text-generation"]},  # just gpt2
    # "starcoder2": {"hf-internal-testing/tiny-random-starcoder2": ["text-generation"]}, # next transformers release
    # text encoder-decoder
    "t5": {"hf-internal-testing/tiny-random-t5": ["text2text-generation"]},
    "bart": {"hf-internal-testing/tiny-random-bart": ["text2text-generation"]},
    # automatic speech recognition
    "whisper": {"openai/whisper-tiny": ["automatic-speech-recognition"]},
    # image to text
    "blip": {"hf-internal-testing/tiny-random-BlipForConditionalGeneration": ["image-to-text"]},
    "blip2": {"hf-internal-testing/tiny-random-Blip2ForConditionalGeneration": ["image-to-text"]},
}

SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION = {
    # stable diffusion
    "stable-diffusion": {"hf-internal-testing/tiny-stable-diffusion-torch": ["stable-diffusion"]},
    "stable-diffusion-xl": {"hf-internal-testing/tiny-stable-diffusion-xl-pipe": ["stable-diffusion-xl"]},
}


def load_model_or_pipe(name_or_path: str, task: str):
    torch.manual_seed(SEED)

    if task == "text-classification":
        return AutoModelForSequenceClassification.from_pretrained(name_or_path)
    elif task == "image-classification":
        return AutoModelForImageClassification.from_pretrained(name_or_path)
    elif task == "text-generation":
        return AutoModelForCausalLM.from_pretrained(name_or_path)
    elif task == "text2text-generation":
        return AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
    elif task == "automatic-speech-recognition":
        return AutoModelForSpeechSeq2Seq.from_pretrained(name_or_path)
    elif task == "image-to-text":
        return AutoModelForVision2Seq.from_pretrained(name_or_path)
    elif task in ["stable-diffusion", "stable-diffusion-xl"]:
        return DiffusionPipeline.from_pretrained(name_or_path)
    else:
        raise ValueError(f"Task {task} not supported")


def get_dummy_inputs(model_type: str, model_id: str, task: str):
    if task in ["text-classification", "text-generation", "text2text-generation"]:
        processor = AutoTokenizer.from_pretrained(model_id)
        text = ["This is a test sentence"] * BATCH_SIZE
        dummy_inputs = processor(text=text, return_tensors="pt")

    elif task in ["image-to-text", "image-classification"]:
        processor = AutoImageProcessor.from_pretrained(model_id)
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        images = [Image.open(requests.get(image_url, stream=True).raw)] * BATCH_SIZE
        dummy_inputs = processor(images=images, return_tensors="pt")

    elif task in ["audio-classification", "automatic-speech-recognition"]:
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
        audios = [ffmpeg_read(requests.get(audio_url).content, processor.sampling_rate)] * BATCH_SIZE
        dummy_inputs = processor(raw_speech=audios, return_tensors="pt")

    elif task in ["stable-diffusion", "stable-diffusion-xl"]:
        dummy_inputs = {"prompt": ["This is test prompt"] * BATCH_SIZE}

    else:
        raise ValueError(f"Task {task} not supported")

    return dummy_inputs


def load_and_compile_model(model_id, task, backend):
    model = load_model_or_pipe(model_id, task)

    torch._dynamo.reset()
    model = torch.compile(model, backend=backend)

    return model


def load_and_compile_pipeline(model_id, task, backend):
    pipe = load_model_or_pipe(model_id, task)

    torch._dynamo.reset()
    pipe.unet = torch.compile(pipe.unet, backend=backend)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend=backend)

    return pipe


class TestZenTorchPlugin(unittest.TestCase):
    @parameterized.expand(SUPPORTED_MODELS_TINY.keys())
    def test_text_classification_model(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                inputs = get_dummy_inputs(model_type, model_id, task)

                if EAGER_DEBUG:
                    model = load_model_or_pipe(model_id, task)
                    _ = model(**inputs).logits
                    continue

                inductor_model = load_and_compile_model(model_id, task, backend="inductor")
                inductor_logits = inductor_model(**inputs).logits

                zentorch_model = load_and_compile_model(model_id, task, backend="zentorch")
                zentorch_logits = zentorch_model(**inputs).logits

                torch.testing.assert_close(inductor_logits, zentorch_logits, rtol=1e-3, atol=1e-5)

    @parameterized.expand(SUPPORTED_MODELS_TINY_TEXT_GENERATION.keys())
    def test_text_generation_model(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY_TEXT_GENERATION[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                inputs = get_dummy_inputs(model_type, model_id, task)

                if EAGER_DEBUG:
                    model = load_model_or_pipe(model_id, task)
                    _ = model.generate(**inputs, **TEXT_GENERATION_KWARGS)
                    continue

                inductor_model = load_and_compile_model(model_id, task, backend="inductor")
                # use logits in transformers next release
                # https://github.com/huggingface/transformers/issues/14498#issuecomment-1953014058
                inductor_scores = inductor_model.generate(**inputs, **TEXT_GENERATION_KWARGS).scores
                inductor_scores = torch.concat(inductor_scores)

                zentorch_model = load_and_compile_model(model_id, task, backend="zentorch")
                # use logits in transformers next release
                # https://github.com/huggingface/transformers/issues/14498#issuecomment-1953014058
                zentorch_scores = zentorch_model.generate(**inputs, **TEXT_GENERATION_KWARGS).scores
                zentorch_scores = torch.concat(zentorch_scores)

                torch.testing.assert_close(inductor_scores, zentorch_scores, rtol=1e-3, atol=1e-5)

    @parameterized.expand(SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION.keys())
    def test_image_diffusion_pipe(self, model_type: str):
        model_id_and_tasks = SUPPORTED_MODELS_TINY_IMAGE_DIFFUSION[model_type]

        for model_id, tasks in model_id_and_tasks.items():
            for task in tasks:
                inputs = get_dummy_inputs(model_type, model_id, task)

                if EAGER_DEBUG:
                    pipe = load_model_or_pipe(model_id, task)
                    pipe(**inputs, **IMAGE_DIFFUSION_KWARGS).images
                    continue

                # we get a new instance of the pipe for every backend to avoid any side effects
                inductor_pipe = load_and_compile_pipeline(model_id, task, backend="inductor")
                inductor_images = inductor_pipe(**inputs, **IMAGE_DIFFUSION_KWARGS).images

                zentorch_pipe = load_and_compile_pipeline(model_id, task, backend="zentorch")
                zentorch_images = zentorch_pipe(**inputs, **IMAGE_DIFFUSION_KWARGS).images

                torch.testing.assert_close(inductor_images, zentorch_images, rtol=1e-3, atol=1e-5)
